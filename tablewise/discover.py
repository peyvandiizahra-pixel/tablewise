"""Stage 1: what identifies each row (primary keys) and which columns point to other tables (relationships).
Only general signals are used, nothing about any particular dataset."""
from __future__ import annotations

import json
import re
from itertools import combinations

import pandas as pd

ID_WORDS = {"id", "key", "code", "ref", "reference", "number", "num", "no", "uuid", "guid", "identifier"}


def name_tokens(name):
    """'appt_id' -> ['appt', 'id'];  'orderItemId' -> ['order', 'item', 'id']"""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def singular(word):
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "ches", "shes", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def stem_tokens(table):
    """'order_items' -> ['order', 'item'];  'triage_requests' -> ['triage', 'request']"""
    toks = name_tokens(table)
    return toks[:-1] + [singular(toks[-1])] if toks else toks


def shape(value):
    """The FORM of a value without the value: 'CL001' -> 'AA999', '627-684-9103' -> '999-999-9999'."""
    s = re.sub(r"\d", "9", re.sub(r"[A-Za-z]", "A", str(value)))
    return re.sub(r"A{6,}|9{13,}", lambda m: m.group(0)[0] + "+", s)[:24]


# ---- Three ways to compare values. Nothing is changed in the data: these are only used to MEASURE. ----
def exact_text(s):
    """The value exactly as stored (a whole-number decimal like 2071.0 is written 2071)."""
    s = s.dropna()
    if pd.api.types.is_float_dtype(s) and len(s) and (s == s.round()).all():
        s = s.astype("int64")
    return s.astype(str)


def format_text(s):
    """Ignoring upper/lower case, spaces and punctuation: ' CL-001' and 'cl001' look the same."""
    return exact_text(s).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)


def zeros_text(s):
    """...and also ignoring zeros at the start of the number part: 'CL001' and 'CL1' look the same."""
    return format_text(s).str.replace(r"^([a-z]*)0+(?=\d)", r"\1", regex=True)


MATCH_MODES = {"exact": exact_text, "format": format_text, "zeros": zeros_text}
MODE_LABEL = {"exact": "exact", "format": "ignoring case/spaces/punctuation", "zeros": "also ignoring leading zeros"}


def column_kind(s, bq_type):
    """integer, decimal, date, boolean or text. Uses the stored type, or the values if everything is text."""
    if bq_type in ("DATE", "DATETIME", "TIMESTAMP", "TIME"):
        return "date"
    if bq_type in ("BOOL", "BOOLEAN"):
        return "boolean"
    v = s.dropna()
    if not len(v):
        return "empty"
    if bq_type in ("INT64", "INTEGER"):
        return "integer"
    if bq_type in ("FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"):
        v = pd.to_numeric(v, errors="coerce").dropna()
        return "integer" if len(v) and (v == v.round()).all() else "decimal"
    v = v.astype(str).str.strip().sample(min(len(v), 5000), random_state=0)
    if v.str.lower().isin(["true", "false", "yes", "no", "y", "n"]).mean() >= 0.95:
        return "boolean"
    if v.str.fullmatch(r"-?(0|[1-9]\d{0,17})").mean() >= 0.95:     # digits, no leading zero
        return "integer"
    if v.str.fullmatch(r"-?\d+\.\d+").mean() >= 0.95:
        return "decimal"
    if pd.to_datetime(v, errors="coerce", format="mixed", utc=True).notna().mean() >= 0.9:
        return "date"
    return "text"


def profile_columns(tables, schemas):
    rows = []
    for t, df in tables.items():
        n = len(df)
        for c in df.columns:
            s = df[c]
            kind = column_kind(s, schemas[t].get(c, "STRING"))
            ex = exact_text(s)
            filled = len(ex)
            shapes = ex.sample(min(filled, 2000), random_state=0).map(shape) if filled else pd.Series(dtype=str)
            top_shape = shapes.value_counts() if filled else pd.Series(dtype=int)
            nums = pd.to_numeric(ex, errors="coerce") if kind == "integer" else None
            d = ex.nunique()
            sequence = bool(kind == "integer" and filled and nums.min() >= 0 and nums.min() <= 10
                            and (nums.max() - nums.min() + 1) <= 1.5 * d)
            rows.append({
                "table": t, "column": c, "bq_type": schemas[t].get(c, "STRING"), "kind": kind,
                "rows": n, "filled_pct": round(100 * filled / n, 1) if n else 0.0,
                "distinct": d, "distinct_format": format_text(s).nunique(), "distinct_zeros": zeros_text(s).nunique(),
                "unique_pct": round(100 * d / filled, 2) if filled else 0.0,
                "avg_len": round(ex.str.len().mean(), 1) if filled else 0.0,
                "shape": top_shape.index[0] if len(top_shape) else "",
                "shape_pct": round(100 * top_shape.iloc[0] / top_shape.sum()) if len(top_shape) else 0,
                "counter": sequence,          # looks like 1, 2, 3 ... N
                "id_name": bool(set(name_tokens(c)) & ID_WORDS),
            })
    return pd.DataFrame(rows)


# ===================================== primary keys =====================================
def names_other_table(column, table, all_tables):
    """'registered_practice' in patients names the table 'practices': it probably POINTS to a key elsewhere."""
    col = name_tokens(column)
    for other in all_tables:
        stem = stem_tokens(other)
        if other != table and stem and any(col[i:i + len(stem)] == stem for i in range(len(col))):
            return other
    return None


def find_primary_keys(tables, prof):
    out = []
    for t, df in tables.items():
        p = prof[prof.table == t].set_index("column")
        stem = stem_tokens(t)
        full_copies = int(df.duplicated().sum())
        found = []
        for c, r in p.iterrows():
            if r.kind in ("decimal", "boolean", "empty") or r.avg_len > 40 or r.filled_pct < 90 or r.unique_pct < 90:
                continue
            score, why = 0, []
            u, f = r.unique_pct, r.filled_pct
            score += 40 if u == 100 else 30 if u >= 99 else 20 if u >= 95 else 10
            score += 20 if f == 100 else 15 if f >= 99 else 10 if f >= 95 else 0
            why += [f"{u:g}% unique", f"{f:g}% filled"]
            if r.id_name:
                score += 15
                why.append("name sounds like an ID")
            if any(tok[:3] == s[:3] for tok in name_tokens(c) if len(tok) >= 3 and tok not in ID_WORDS for s in stem):
                score += 10
                why.append(f"name sounds like '{t}'")
            if r.shape_pct >= 90:
                score += 10
                why.append(f"{r.shape_pct}% share one form '{r['shape']}'")
            pointer = names_other_table(c, t, tables)
            if pointer:
                score -= 20
                why.append(f"named after '{pointer}', so probably points there")
            if r.kind == "date":
                score -= 30
                why.append("a date/time: unique by accident")
            if c == df.columns[0]:
                score += 5
                why.append("first column")
            hidden = r.distinct - r.distinct_format
            if hidden > 0:
                why.append(f"{hidden} values differ only in case/spaces/punctuation")
            if u < 100 and full_copies:
                why.append(f"the table has {full_copies} rows that are complete copies of another row")
            status = "strong" if u == 100 and f == 100 and score >= 70 else "needs review" if score >= 50 else "weak"
            found.append({"table": t, "key": [c], "score": score, "status": status, "unique_pct": u,
                          "filled_pct": f, "evidence": " · ".join(why)})

        # No single column is clean enough: try pairs of columns (e.g. customer + date)
        if not any(x["status"] == "strong" for x in found):
            usable = [c for c, r in p.iterrows()
                      if r.kind not in ("decimal", "empty") and r.avg_len <= 40 and r.filled_pct >= 95 and r.distinct >= 2]
            usable = sorted(usable, key=lambda c: -p.loc[c, "distinct"])[:20]
            for a, b in combinations(usable, 2):
                pair = df[[a, b]].dropna()
                u = 100 * len(pair.drop_duplicates()) / max(len(pair), 1)
                if u < 99 or max(p.loc[a, "unique_pct"], p.loc[b, "unique_pct"]) >= 99.9:
                    continue
                f = 100 * len(pair) / max(len(df), 1)
                hint = sum(bool(p.loc[x, "id_name"] or names_other_table(x, t, tables) or p.loc[x, "kind"] == "date")
                           for x in (a, b))
                score = (40 if u == 100 else 30) + (20 if f == 100 else 10) + 10 * hint
                found.append({"table": t, "key": [a, b], "score": score,
                              "status": "strong" if u == 100 and f == 100 and hint == 2 else "needs review",
                              "unique_pct": round(u, 2), "filled_pct": round(f, 1),
                              "evidence": f"together {u:.4g}% unique · {f:.4g}% filled · {hint} of 2 look like IDs/dates/links"})
        found.sort(key=lambda x: -x["score"])
        out += found[:4] or [{"table": t, "key": [], "score": 0, "status": "none found", "unique_pct": None,
                              "filled_pct": None, "evidence": "no column or pair of columns identifies the rows"}]
    return pd.DataFrame(out)


def keys_used_for_links(pk_cands, chosen_keys=None):
    """The key of each table that links are searched against: your saved choice, else its best suggestion."""
    used = {}
    for t, g in pk_cands.groupby("table", sort=False):
        top = g.iloc[0]
        if len(top.key) == 1 and top.status in ("strong", "needs review"):
            used[t] = top.key[0]
    for t, key in (chosen_keys or {}).items():
        used.pop(t, None)
        if len(key) == 1:
            used[t] = key[0]
    return used


# ===================================== relationships =====================================
def name_evidence(column, parent_table, parent_key):
    """2.5 = named after that table, and the name ENDS with it  ('user_id' -> users, 'clinician' -> clinicians)
       2   = the table's name appears in it                       ('patient_nhs' -> patients)
       1   = a partial hint                                       ('triage_ref' -> triage_requests)
       0   = no hint from the names"""
    col, stem = name_tokens(column), stem_tokens(parent_table)
    key = name_tokens(parent_key)

    def contains(seq):
        return bool(seq) and any(col[i:i + len(seq)] == seq for i in range(len(col)))

    def ends_with(seq):
        core = col[:-1] if col and col[-1] in ID_WORDS else col
        return bool(seq) and core[-len(seq):] == seq

    if contains(stem):
        return 2.5 if ends_with(stem) else 2
    if key and not set(key) <= ID_WORDS and contains(key):
        return 2.5
    for a in col:
        for b in stem:
            if a not in ID_WORDS and len(a) >= 3 and len(b) >= 3 and (a == b or b.startswith(a) or a.startswith(b)):
                return 1
    return 0


def find_links(tables, prof, pk_cands, read_key, chosen_keys=None, min_match=0.5):
    """For every column, check if its values are found among another table's key values. Measures three
    ways of comparing (exact / ignoring format / also ignoring leading zeros) and changes nothing."""
    prof_i = prof.set_index(["table", "column"])
    own_key = {t: k for t, k in keys_used_for_links(pk_cands, chosen_keys).items() if t in tables}
    parents = []
    for t, k in own_key.items():
        values = read_key(t, k)
        # an Index is built once and then looked up very fast, even with millions of key values
        parents.append({"table": t, "key": k, "info": prof_i.loc[(t, k)],
                        "keys": {m: pd.Index(pd.unique(f(values))) for m, f in MATCH_MODES.items()},
                        "distinct": values.nunique()})
    found = []
    for t, df in tables.items():
        for c in df.columns:
            ci = prof_i.loc[(t, c)]
            if ci.kind not in ("integer", "text") or ci.avg_len > 40 or ci.distinct < 2:
                continue
            counts = {m: f(df[c]).value_counts() for m, f in MATCH_MODES.items()}   # each value once, with its count
            total = counts["exact"].sum()
            if not total:
                continue
            for par in parents:
                if par["table"] == t:
                    continue                                   # links inside one table: not covered yet
                found_in = {m: par["keys"][m].get_indexer(counts[m].index) >= 0 for m in MATCH_MODES}
                pct = {m: counts[m][found_in[m]].sum() / total for m in MATCH_MODES}
                best = max(pct, key=lambda m: (pct[m], m == "exact"))
                if pct[best] < min_match:
                    continue
                name = name_evidence(c, par["table"], par["key"])
                counter = bool(par["info"].counter and ci.kind == "integer")
                small = par["distinct"] < 20
                if own_key.get(t) == c and counter and name < 2:
                    continue                                   # two tables numbered 1, 2, 3...: not a link
                hit = counts[best][found_in[best]]
                child_unique = bool((counts["exact"] == 1).all())
                flags = []
                if counter:
                    flags.append("the key is just counting numbers 1, 2, 3...")
                if small:
                    flags.append(f"the key has only {par['distinct']} values")
                if best != "exact":
                    flags.append(f"matches only {MODE_LABEL[best]}")
                found.append({
                    "from": f"{t}.{c}", "to": f"{par['table']}.{par['key']}",
                    "from_table": t, "from_column": c, "to_table": par["table"], "to_column": par["key"],
                    "exact_pct": round(100 * pct["exact"], 1), "format_pct": round(100 * pct["format"], 1),
                    "zeros_pct": round(100 * pct["zeros"], 1), "best_mode": best,
                    "values_found_pct": round(100 * found_in[best].mean(), 1),
                    "cardinality": "one-to-one" if child_unique else "many-to-one",
                    "per_parent": round(hit.sum() / max(len(hit), 1), 1),
                    "parents_unused_pct": round(100 * (1 - len(hit) / max(len(par["keys"][best]), 1)), 1),
                    "name": name, "counter_or_small": counter or small, "flags": flags,
                    "child_shape": ci["shape"], "parent_shape": par["info"]["shape"],
                    "child_kind": ci.kind, "filled_pct": ci.filled_pct})
    links = pd.DataFrame(found)
    if links.empty:
        return links

    def decide(group):
        group = group.sort_values(["name", "exact_pct", "zeros_pct"], ascending=False).copy()
        status = pd.Series("needs review", index=group.index)
        eligible = []
        for i, r in group.iterrows():
            if r["counter_or_small"] and r["name"] < 2:       # common values + no clear name = coincidence
                status[i] = "unlikely"
                r["flags"].append("needs a clear name match because the values are so common")
            else:
                eligible.append(i)
        if eligible:
            best_name = group.loc[eligible, "name"].max()
            best = [i for i in eligible if group.at[i, "name"] == best_name]
            for i in eligible:
                if i not in best:
                    status[i] = "unlikely"
                    group.at[i, "flags"].append(f"'{group.at[best[0], 'to_table']}' fits the column name better")
            fits = [i for i in best if max(group.at[i, "exact_pct"], group.at[i, "zeros_pct"]) >= 90]
            for i in best:
                r = group.loc[i]
                if len(fits) > 1 and i in fits:
                    others = [group.at[j, "to_table"] for j in fits if j != i]
                    r["flags"].append(f"ambiguous: also fits {others}")
                elif r.exact_pct >= 95 and r["name"] >= 1:
                    status[i] = "strong"
        group["status"] = status
        return group

    links = pd.concat([decide(g) for _, g in links.groupby("from", sort=False)])
    links["score"] = (0.6 * links[["exact_pct", "format_pct", "zeros_pct"]].max(axis=1)
                      + 10 * links["name"] + 0.1 * links["values_found_pct"]
                      - 30 * (links["status"] == "unlikely")).round()
    order = {"strong": 0, "needs review": 1, "unlikely": 2}
    return links.sort_values(by=["status", "score"], key=lambda s: s.map(order) if s.name == "status" else -s) \
                .reset_index(drop=True)


def bridges(links):
    """A table pointing to two other tables connects them many-to-many (e.g. orders <-> products via order_items)."""
    out = []
    for t, g in pd.DataFrame(links).groupby("from_table"):
        parents = sorted(set(g["to_table"]))
        for a, b in combinations(parents, 2):
            out.append(f"{a} <-> {b}: many-to-many, through {t}")
    return out


# ===================================== the AI checks the meaning =====================================
AI_CHECK_SYSTEM = (
    "You review relationships that a program proposed between database tables, using value matching. "
    "Matching values can be a coincidence (for example two tables both numbered 1, 2, 3), so judge whether the "
    "column really refers to the other table. You only see table names, column names, types and value forms "
    "(A = letter, 9 = digit), never the values. Answer only with JSON.")


def ai_check_links(links, tables, ai, max_items=80, log=print):
    """The AI never approves anything. It can only move a link to "needs review" or out of it."""
    if links.empty:
        return links
    links = links.copy()
    todo = links.sort_values("score", ascending=False).head(max_items)
    schema = {t: list(df.columns) for t, df in tables.items()}
    items = [{"id": int(i), "column": r["from"], "column_type": r["child_kind"], "column_form": r["child_shape"],
              "points_to": r["to"], "key_form": r["parent_shape"],
              "values_found": f"{max(r.exact_pct, r.format_pct, r.zeros_pct):g}% ({MODE_LABEL[r.best_mode]})",
              "cardinality": r.cardinality} for i, r in todo.iterrows()]
    prompt = (f"Tables and their columns:\n{json.dumps(schema)}\n\nProposed relationships:\n{json.dumps(items)}\n\n"
              "For each one return {\"id\": ..., \"verdict\": \"yes\" | \"no\" | \"unsure\", "
              "\"reason\": \"one short sentence\", \"link_name\": \"a short verb phrase, e.g. 'is booked with'\"}. "
              "Return a JSON list only.")
    try:
        answers = {a["id"]: a for a in ai.ask_json(AI_CHECK_SYSTEM, prompt, expect=list) if isinstance(a, dict) and "id" in a}
    except Exception as e:
        log(f"AI check skipped: {e}")
        return links.assign(ai_verdict="", ai_reason="", link_name="")
    links["ai_verdict"] = [answers.get(i, {}).get("verdict", "") for i in links.index]
    links["ai_reason"] = [answers.get(i, {}).get("reason", "") for i in links.index]
    links["link_name"] = [answers.get(i, {}).get("link_name", "") for i in links.index]
    for i, r in links.iterrows():
        if r.status == "strong" and r.ai_verdict == "no":
            links.at[i, "status"] = "needs review"
            r["flags"].append("the AI doubts it")
        elif r.status == "needs review" and r.ai_verdict == "no" and r.counter_or_small:
            links.at[i, "status"] = "unlikely"
            r["flags"].append("the AI says it's a coincidence")
    # A column with NO good link at all, where the names don't match but the AI recognises the meaning
    # (e.g. 'client' -> customers): its best AI-approved candidate goes back to "needs review"
    for _, g in links.groupby("from"):
        if (g.status != "unlikely").any():
            continue
        yes = g[g.ai_verdict == "yes"].sort_values("score", ascending=False)
        if len(yes):
            i = yes.index[0]
            links.at[i, "status"] = "needs review"
            links.at[i, "flags"].append("the AI thinks it's real although the names don't match")
    return links


# ===================================== the map =====================================
def map_text(tables, pks, links):
    lines = []
    for t in tables:
        key = " + ".join(pks.get(t, [])) or "no key"
        lines.append(f"{t}  (key: {key}, {len(tables[t]):,} rows)")
        for l in links:
            if l["from_table"] == t:
                how = "" if l["match"] == "exact" else f", compared {MODE_LABEL[l['match']]}"
                name = f" '{l['link_name']}'" if l.get("link_name") else ""
                lines.append(f"   {l['from_column']} -> {l['to']}{name}  ({l['cardinality']}, {l['match_pct']:g}% match{how})")
    extra = bridges(links) if links else []
    if extra:
        lines.append("\nConnected through a third table:\n   " + "\n   ".join(extra))
    return "\n".join(lines)


def map_svg(tables, pks, links):
    """A drawing of the approved model, if Graphviz is installed; else None."""
    try:
        import graphviz
        g = graphviz.Digraph(graph_attr={"rankdir": "LR", "bgcolor": "transparent"},
                             node_attr={"shape": "box", "style": "rounded", "fontname": "Arial", "fontsize": "11"},
                             edge_attr={"fontname": "Arial", "fontsize": "9", "color": "#52514e"})
        for t in tables:
            g.node(t, f"{t}\nkey: {' + '.join(pks.get(t, [])) or '-'}")
        for l in links:
            label = l["from_column"] + (" (N:1)" if l["cardinality"] == "many-to-one" else " (1:1)")
            g.edge(l["from_table"], l["to_table"], label=label, style="solid" if l["match"] == "exact" else "dashed")
        return g.pipe(format="svg").decode()
    except Exception:
        return None
