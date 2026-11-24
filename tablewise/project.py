"""A Tablewise project: one dataset, its saved decisions, and the five stages. Every decision is a JSON file in the
workspace folder, so you can close the app and carry on later (and see exactly what was approved)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import discover as d1
from . import ontology as d2
from .ai import AI
from .ask import Asker
from .engine import Engine, metric_problems, propose_metrics
from .monitor import METHODS, Monitor, monitor_problems, propose_monitors
from .views import build_views


@dataclass
class Settings:
    max_rows_per_table: int = 200_000       # rows read from each table to study it
    max_key_values: int = 2_000_000          # values read from a key column: links are checked against ALL of them
    max_bytes_per_query: int = 1_000_000_000  # BigQuery only: one query may read at most this much
    model: str = "claude-sonnet-5-5"
    price_in: float = 2.00                   # USD per million tokens, only used to count spend
    price_out: float = 10.00
    ai_budget_usd: float = 0.50              # per session; the AI stops after this
    lookback: int = 12                       # past periods that define "normal" for alerts
    stale_data_alerts: bool = False          # alert when the newest complete period has no rows at all
    email_alerts: bool = True                # needs GMAIL_ADDRESS and GMAIL_APP_PASSWORD environment variables


def _split_key(text):
    return [k.strip() for k in str(text or "").replace(",", "+").split("+") if k.strip()]


class Project:
    def __init__(self, backend, workspace, settings=None, ai=None, api_key=None, log=print):
        self.backend, self.dir = backend, Path(workspace)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.s = settings or Settings()
        self.log = log
        self.ai = ai or AI(self.dir / "ai_cache.json", self.s.model, self.s.price_in, self.s.price_out,
                           self.s.ai_budget_usd, api_key=api_key)
        self.tables = self.schemas = self.profile = self.pk_candidates = None
        self.links = pd.DataFrame()
        self.pks, self.approved_links = {}, []
        self.specs = self.info = None
        self.ontology, self.engine, self.view_checks = None, None, None
        self.candidate_metrics, self.metrics = [], {}
        self.candidate_monitors, self.monitors = [], []
        self._load_saved()

    # ---------- files ----------
    def _file(self, name):
        return self.dir / name

    def _read(self, name, default):
        try:
            return json.loads(self._file(name).read_text())
        except Exception:
            return default

    def _write(self, name, data):
        self._file(name).write_text(json.dumps(data, indent=2, default=str))

    def _load_saved(self):
        dec = self._read("decisions.json", {})
        self.pks = dec.get("primary_keys", {})
        self.approved_links = dec.get("links", [])
        onto = self._read("ontology.json", None)
        if onto and onto.get("view_columns"):
            self.ontology = onto
            try:
                self.engine = Engine(self.backend, onto, self.s.max_bytes_per_query)
            except Exception as e:
                self.log(f"Saved ontology couldn't be loaded ({e}); build the views again.")
                self.engine = None
        if self.engine:
            self.metrics = {m["name"]: m for m in self._read("metrics.json", [])}
            self.monitors = self._read("monitors.json", [])

    @property
    def stage(self):
        """How far this project has got: 0 nothing, 1 keys and links saved, 2 views built, 3 metrics, 4 checks."""
        return (4 if self.monitors else 3 if self.metrics else 2 if self.engine else 1 if self.pks else 0)

    # =================================== Stage 1 ===================================
    def analyse(self):
        """Read every table, study every column, suggest keys and links, and let the AI check their meaning."""
        b, s = self.backend, self.s
        self.tables, self.schemas, notes = {}, {}, []
        for t in b.list_tables():
            df, schema, total = b.read_table(t, s.max_rows_per_table)
            self.tables[t], self.schemas[t] = df, schema
            notes.append({"table": t, "rows": total, "read": len(df), "columns": len(df.columns)})
        self.profile = d1.profile_columns(self.tables, self.schemas)
        self.pk_candidates = d1.find_primary_keys(self.tables, self.profile)

        def read_key(t, k):
            return b.read_column(t, k, s.max_key_values) if self.tables_total(notes, t) > len(self.tables[t]) \
                else self.tables[t][k]
        links = d1.find_links(self.tables, self.profile, self.pk_candidates, read_key, chosen_keys=self.pks or None)
        self.links = d1.ai_check_links(links, self.tables, self.ai, log=self.log)
        return pd.DataFrame(notes)

    @staticmethod
    def tables_total(notes, t):
        return next((n["rows"] or 0 for n in notes if n["table"] == t), 0)

    def key_rows(self):
        """One row per table: the key in use (your saved choice, else the best suggestion) and the options."""
        rows = []
        for t in self.tables:
            cands = self.pk_candidates[self.pk_candidates.table == t]
            top = cands.iloc[0]
            if t in self.pks:
                key = self.pks[t]
            else:
                key = top.key if top.key and top.status in ("strong", "needs review") else []
            mine = cands[cands.key.map(lambda k: list(k) == list(key))]
            r = mine.iloc[0] if len(mine) else None
            rows.append({"table": t, "primary key": " + ".join(key),
                         "status": r.status if r is not None else ("your choice" if key else "none"),
                         "evidence": r.evidence if r is not None else "",
                         "other options": " | ".join(" + ".join(c.key) + f" ({c.status})" for c in cands.itertuples()
                                                     if c.key and list(c.key) != list(key))})
        return pd.DataFrame(rows)

    def link_rows(self):
        """Every candidate link, strongest first. Approved = your saved choice, else 'strong'."""
        cols = ["approve", "from", "to", "compare", "exact %", "format %", "zeros %", "status", "cardinality",
                "per parent", "notes"]
        if self.links is None or self.links.empty:
            return pd.DataFrame(columns=cols)
        saved = {(l["from"], l["to"]): l for l in self._read("decisions.json", {}).get("links", [])}
        rejected = {(l["from"], l["to"]) for l in self._read("decisions.json", {}).get("rejected_links", [])}
        rows = []
        for _, r in self.links.iterrows():
            key = (r["from"], r["to"])
            prev = saved.get(key)
            approve = True if prev else False if key in rejected else r.status == "strong"
            notes = list(r["flags"])
            if r.get("ai_verdict"):
                notes.append(f"AI: {r['ai_verdict']} – {r.get('ai_reason', '')}")
            rows.append({"approve": approve, "from": r["from"], "to": r["to"],
                         "compare": prev["match"] if prev else ("exact" if r.status == "strong" else r.best_mode),
                         "exact %": r.exact_pct, "format %": r.format_pct, "zeros %": r.zeros_pct, "status": r.status,
                         "cardinality": r.cardinality, "per parent": r.per_parent, "notes": "; ".join(notes)})
        return pd.DataFrame(rows, columns=cols)

    def save_decisions(self, keys, link_rows):
        """keys: {table: 'col' or 'col_a + col_b' or ''}; link_rows: rows like link_rows() returns.
        Returns (messages, links_need_rerun)."""
        msgs, pks = [], {}
        for t, text in keys.items():
            cols = _split_key(text)
            bad = [c for c in cols if c not in self.tables[t].columns]
            if bad:
                raise ValueError(f"{t} has no column {bad}")
            pks[t] = cols
        by_key = {(r["from"], r["to"]): r for _, r in self.links.iterrows()} if len(self.links) else {}
        approved, rejected = [], []
        for _, row in pd.DataFrame(link_rows).iterrows():
            r = by_key.get((row["from"], row["to"]))
            if r is None:
                continue
            mode = str(row["compare"]).strip().lower()
            if mode not in d1.MATCH_MODES:
                raise ValueError(f"{row['from']}: 'compare' must be exact, format or zeros (not '{row['compare']}')")
            rec = {"from": r["from"], "to": r["to"], "from_table": r.from_table, "from_column": r.from_column,
                   "to_table": r.to_table, "to_column": r.to_column, "match": mode,
                   "match_pct": float(r[f"{mode}_pct"]), "cardinality": r.cardinality,
                   "link_name": r.get("link_name", ""), "approved": bool(row["approve"])}
            (approved if rec["approved"] else rejected).append(rec)
        for l in approved:
            if pks.get(l["to_table"]) != [l["to_column"]]:
                msgs.append(f"check: {l['from']} -> {l['to']} points to {l['to_column']}, which is not the key you chose for {l['to_table']}")
        used = d1.keys_used_for_links(self.pk_candidates, self.pks or None)
        changed = sorted(t for t, k in pks.items() if len(k) == 1 and used.get(t) != k[0])
        self.pks, self.approved_links = pks, approved
        self._write("decisions.json", {"primary_keys": pks, "links": approved, "rejected_links": rejected})
        msgs.insert(0, f"Saved {sum(bool(v) for v in pks.values())} primary keys and {len(approved)} relationships.")
        if changed:
            msgs.append(f"You changed the key of {changed}: links to the new key are searched again now.")
        return msgs, bool(changed)

    def map(self):
        return (d1.map_text(self.tables, self.pks, self.approved_links),
                d1.map_svg(self.tables, self.pks, self.approved_links))

    # =================================== Stage 2 ===================================
    def understand(self):
        if not self.pks and not self.approved_links:
            raise ValueError("Save your keys and relationships first.")
        self.specs, self.info = d2.understand_tables(self.tables, self.schemas, self.profile, self.pks,
                                                     self.approved_links, self.ai, log=self.log)

    def ontology_rows(self):
        """Four tables to review, pre-filled with your saved choices: objects, columns, merges, negatives."""
        prev = self._read("ontology.json", {}).get("tables", {})
        objs, cols, merges, negs = [], [], [], []
        for t, specs in self.specs.items():
            pt, info = prev.get(t, {}), self.info[t]
            objs.append({"table": t, "one row is a": pt.get("entity", info["entity"]),
                         "description": pt.get("description", info["description"]),
                         "keep one row per key": pt.get("dedupe", info["duplicate_keys"] > 0),
                         "repeated keys": info["duplicate_keys"]})
            for c, sp in specs.items():
                pc = pt.get("columns", {}).get(c, {})
                cols.append({"table": t, "column": c, "type": pc.get("type", sp["type"]),
                             "personal": pc.get("personal", sp["personal"]), "meaning": pc.get("meaning", sp["meaning"]),
                             "notes": d2.notes_for(sp)})
                prev_map = pc.get("mapping")
                for old, new in sp["mapping"].items():
                    merges.append({"apply": (old in prev_map) if prev_map is not None else True, "table": t, "column": c,
                                   "value": old, "becomes": ((prev_map or {}).get(old, new) if prev_map else new) or "",
                                   "why": sp["mapping_reasons"].get(old, "")})
                if sp["negatives"]:
                    ng = sp["negatives"]
                    negs.append({"table": t, "column": c, "negatives": ng["count"],
                                 "examples": ", ".join(f"{k} ({v}x)" for k, v in ng["examples"].items()),
                                 "rule": pc.get("negative_rule", sp["negative_rule"]), "why": ng.get("reason", "")})
        return (pd.DataFrame(objs), pd.DataFrame(cols),
                pd.DataFrame(merges, columns=["apply", "table", "column", "value", "becomes", "why"]),
                pd.DataFrame(negs, columns=["table", "column", "negatives", "examples", "rule", "why"]))

    def save_ontology(self, objs, cols, merges, negs):
        choices = {"tables": {}, "columns": {}, "mapping": {}, "negative_rule": {}}
        for _, r in pd.DataFrame(objs).iterrows():
            choices["tables"][r["table"]] = {"entity": str(r["one row is a"]), "description": str(r["description"]),
                                             "dedupe": bool(r["keep one row per key"])}
        for _, r in pd.DataFrame(cols).iterrows():
            typ, pers = str(r["type"]).strip().lower(), str(r["personal"]).strip().lower()
            if typ not in d2.TYPES:
                raise ValueError(f"{r['table']}.{r['column']}: type must be one of {d2.TYPES}")
            if pers not in d2.PERSONAL_CHOICES:
                raise ValueError(f"{r['table']}.{r['column']}: personal must be one of {d2.PERSONAL_CHOICES}")
            choices["columns"][(r["table"], r["column"])] = {"type": typ, "personal": pers, "meaning": str(r["meaning"] or "")}
        for (t, c), sp in ((k, self.specs[k[0]][k[1]]) for k in choices["columns"]):
            choices["mapping"][(t, c)] = {}
        for _, r in pd.DataFrame(merges).iterrows():
            if bool(r["apply"]):
                choices["mapping"].setdefault((r["table"], r["column"]), {})[r["value"]] = (str(r["becomes"]).strip() or None)
        for _, r in pd.DataFrame(negs).iterrows():
            rule = str(r["rule"]).strip().lower()
            if rule not in d2.NEGATIVE_RULES:
                raise ValueError(f"{r['table']}.{r['column']}: rule must be one of {d2.NEGATIVE_RULES}")
            choices["negative_rule"][(r["table"], r["column"])] = rule
        prev = self._read("ontology.json", {})
        self.ontology = d2.build_ontology(self.specs, self.info, self.pks, self.approved_links, self.tables, choices,
                                          salt=prev.get("salt"), source=self.backend.label)
        self._write("ontology.json", self.ontology)
        return self.ontology

    def build_views(self):
        if not self.ontology:
            raise ValueError("Save the ontology first.")
        self.view_checks = build_views(self.backend, self.ontology, log=self.log)
        self._write("ontology.json", self.ontology)
        self.engine = Engine(self.backend, self.ontology, self.s.max_bytes_per_query)
        if self.engine.unsafe_links:
            self.log(f"Not used for joins in metrics, because the target isn't a unique key: {self.engine.unsafe_links}")
        return self.view_checks

    # =================================== Stage 3 ===================================
    def propose_metrics(self, manual=()):
        self._need_engine()
        saved = self._read("metrics.json", [])
        self.candidate_metrics = propose_metrics(self.engine, self.ai, saved=saved, manual=manual, log=self.log)
        return self.metric_rows()

    def metric_rows(self):
        saved = {m["name"] for m in self._read("metrics.json", [])}
        rows = []
        for m in self.candidate_metrics:
            ok = not m["problems"]
            rows.append({"approve": ok and ((m["name"] in saved) if saved else not m.get("auto")),
                         "name": m["name"], "label": m["label"], "table": m["table"],
                         "date column": m.get("time_column") or "", "unit": m.get("unit", ""),
                         "now": m.get("preview"), "sql": m["sql"], "definition": m.get("definition", ""),
                         "problems": "; ".join(m["problems"])})
        return pd.DataFrame(rows, columns=["approve", "name", "label", "table", "date column", "unit", "now", "sql",
                                           "definition", "problems"])

    def save_metrics(self, rows):
        """Ticked rows become approved metrics. A row you edited (or added) is tested again first; one that fails
        its checks is not approved."""
        cands = {m["name"]: m for m in self.candidate_metrics}
        approved, refused = {}, []
        for _, r in pd.DataFrame(rows).iterrows():
            if not str(r.get("name") or "").strip():
                continue
            name = str(r["name"]).strip()
            base = cands.get(name, {"higher_is": "neutral"})
            m = {k: v for k, v in base.items() if k not in ("problems", "auto", "saved", "preview")}
            m.update(name=name, label=str(r["label"] or name), table=str(r["table"]), sql=str(r["sql"]),
                     time_column=str(r["date column"] or "") or None, unit=str(r["unit"] or ""),
                     definition=str(r["definition"] or ""))
            probs = metric_problems(self.engine, m)
            if name in cands:
                cands[name].update({**m, "problems": probs})
            else:
                self.candidate_metrics.append({**m, "problems": probs})
            if not bool(r["approve"]):
                continue
            if probs:
                refused.append(f"{name}: {'; '.join(probs)}")
                continue
            approved[name] = m
        self.metrics = approved
        self._write("metrics.json", list(approved.values()))
        return [f"Saved {len(approved)} metrics."] + [f"not approved, {p}" for p in refused]

    # =================================== Stage 4 ===================================
    def asker(self):
        self._need_engine()
        if not self.metrics:
            raise ValueError("Approve at least one metric first.")
        return Asker(self.engine, self.ai, self.metrics, self._file("questions_log.csv"))

    def ask(self, question):
        return self.asker().answer(question)

    # =================================== Stage 5 ===================================
    def propose_monitors(self):
        self._need_engine()
        saved = self._read("monitors.json", [])
        self.candidate_monitors = propose_monitors(self.engine, self.ai, self.metrics, saved=saved, log=self.log)
        return self.monitor_rows()

    def monitor_rows(self):
        saved = {m["name"] for m in self._read("monitors.json", [])}
        rows = []
        for mo in self.candidate_monitors:
            ok = not mo["problems"]
            rows.append({"approve": ok and (bool(mo.get("saved")) if saved else True), "name": mo["name"],
                         "metric": mo.get("metric"), "split by": mo.get("group_by") or "", "every": mo.get("grain"),
                         "method": mo.get("method"), "threshold": mo.get("threshold"), "min rows": mo.get("min_rows", 10),
                         "why": mo.get("why", ""), "problems": "; ".join(mo["problems"])})
        return pd.DataFrame(rows, columns=["approve", "name", "metric", "split by", "every", "method", "threshold",
                                           "min rows", "why", "problems"])

    def save_monitors(self, rows):
        cands = {m["name"]: m for m in self.candidate_monitors}
        out, refused = [], []
        for _, r in pd.DataFrame(rows).iterrows():
            if not bool(r["approve"]):
                continue
            name = str(r["name"]).strip()
            try:
                thr = float(r["threshold"])
            except (TypeError, ValueError):
                thr = None
            mo = {**{k: v for k, v in cands.get(name, {}).items() if k not in ("problems", "saved")},
                  "name": name, "metric": str(r["metric"]), "group_by": str(r["split by"] or "") or None,
                  "grain": str(r["every"]).strip().lower(), "method": str(r["method"]).strip().lower(),
                  "threshold": thr, "min_rows": int(float(r["min rows"] or 0)), "why": str(r.get("why") or "")}
            probs = monitor_problems(self.engine, self.metrics, mo)
            if probs:
                refused.append(f"{name}: {'; '.join(probs)}")
                continue
            out.append(mo)
        self.monitors = out
        self._write("monitors.json", out)
        return [f"Saved {len(out)} checks."] + [f"not saved, {p}" for p in refused]

    def monitor(self):
        self._need_engine()
        return Monitor(self.engine, self.ai, self.metrics, self.s.lookback, self.s.stale_data_alerts,
                       dataset_label=self.backend.label)

    def run_checks(self, email=None):
        return self.monitor().run(self.monitors, email=self.s.email_alerts if email is None else email)

    def tasks(self):
        a = self.monitor().load_actions()
        if a.empty:
            return a
        order = a["status"].map({"open": 0, "assigned": 1, "closed": 2}).fillna(3)
        return a.assign(_o=order).sort_values(["_o", "created_at"], ascending=[True, False]).drop(columns="_o")

    def update_task(self, action_id, status, owner=""):
        return self.monitor().update_action(action_id, status, owner)

    def _need_engine(self):
        if not self.engine:
            raise ValueError("Build the clean views first (step 3).")


METHOD_NAMES = list(METHODS)
