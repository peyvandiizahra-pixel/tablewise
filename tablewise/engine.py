"""The query engine: SQL is built only from approved metrics, approved links and listed values. Values travel
as query parameters, and when a table can be reached in two ways, both are offered by name and never mixed."""
from __future__ import annotations

import re

import pandas as pd

from .views import LinkGroups, column_sql, lit

GRAINS = {"day": "{d}", "week": "DATE_TRUNC({d}, WEEK(MONDAY))", "month": "DATE_TRUNC({d}, MONTH)"}


class Engine:
    def __init__(self, backend, ontology, max_bytes=1_000_000_000):
        if not ontology.get("view_columns"):
            raise ValueError("Build the clean views first.")
        self.backend, self.ontology, self.max_bytes = backend, ontology, max_bytes
        self.groups = LinkGroups(ontology)
        self.meta = self._view_meta()
        self.unsafe_links = [f"{l['from']} -> {l['to']}" for l in ontology["links"] if not self.safe_join(l)]

    # ---------- what each view contains ----------
    def _view_meta(self):
        meta = {}
        for t, tb in self.ontology["tables"].items():
            cols = {}
            for c, sp in tb["columns"].items():
                out = column_sql(self.ontology, self.groups, t, c, sp)
                if not out:
                    continue
                name, linked = out[0], (t, c) in self.groups
                typ = "number" if name.endswith("_age_years") else "identifier" if linked else sp["type"]
                hashed = sp["personal"] in ("hash", "drop") or (linked and self.groups.hashed[self.groups.find((t, c))])
                values = sorted({sp["mapping"].get(v, v) for v in sp.get("values", [])} - {None}) if typ == "category" else []
                cols[name] = {"type": typ, "meaning": sp["meaning"], "hashed": hashed, "values": values,
                              "is_datetime": typ == "date" and (sp["has_time"] or sp["bq_type"] in ("DATETIME", "TIMESTAMP"))}
            meta[t] = cols
        return meta

    def safe_join(self, l):
        """A link is followed only if it points to its table's key and that key is unique in the view.
        Otherwise one row could match several rows there and be counted more than once."""
        return (self.ontology["tables"][l["to_table"]]["key"] == [l["to_column"]]
                and self.ontology.get("unique_keys", {}).get(l["to_table"], True))

    def join_paths(self, base, max_hops=3):
        """Every way to reach a parent table from `base` by following approved links (child -> parent)."""
        out, frontier = [[]], [[]]
        for _ in range(max_hops):
            nxt = []
            for path in frontier:
                here = path[-1]["to_table"] if path else base
                seen = {base} | {l["to_table"] for l in path}
                nxt += [path + [l] for l in self.ontology["links"]
                        if l["from_table"] == here and l["to_table"] not in seen and self.safe_join(l)]
            out += nxt
            frontier = nxt
        return out

    def dimensions_for(self, base):
        """Columns a metric of table `base` can be grouped or filtered by: its own and its parents' categories.
        When a parent can be reached in two ways, the name says which way: 'branches.region via account > branch'."""
        paths = self.join_paths(base)
        ends = [p[-1]["to_table"] if p else base for p in paths]
        dims = {}
        for path, t in zip(paths, ends):
            via = f" via {' > '.join(l['from_column'] for l in path)}" if ends.count(t) > 1 else ""
            for c, m in self.meta[t].items():
                if m["hashed"] or (t == base and c in self.ontology["tables"][t]["key"]):
                    continue
                if m["type"] in ("category", "boolean") or (m["type"] == "identifier" and t != base) or \
                        (m["type"] == "text" and t != base and c not in self.ontology["tables"][t]["key"]):
                    dims[f"{t}.{c}{via}"] = {"table": t, "column": c, "type": m["type"], "values": m["values"], "path": path}
        return dims

    def date_expr(self, alias, t, c):
        return f"DATE({alias}.`{c}`)" if self.meta[t][c]["is_datetime"] else f"{alias}.`{c}`"

    # ---------- a plan -> SQL ----------
    def compile_metric(self, plan, metrics):
        """A plan {metric, group_by, time_grain, start, end, filters} -> (SQL, parameters)."""
        m = metrics[plan["metric"]]
        base, dims = m["table"], self.dimensions_for(m["table"])
        aliases, joins, params, where = {(): "t"}, [], [], []
        model_ref = self.backend.model_ref

        def alias(path):
            """One join per link on the way; two different ways to the same table get two different joins."""
            for k in range(1, len(path) + 1):
                key = tuple(f"{l['from']}>{l['to']}" for l in path[:k])
                if key not in aliases:
                    a = aliases[key] = f"j{len(aliases)}"
                    l = path[k - 1]
                    joins.append(f"LEFT JOIN {model_ref(l['to_table'])} AS {a} "
                                 f"ON {aliases[key[:-1]]}.`{l['from_column']}` = {a}.`{l['to_column']}`")
            return aliases[tuple(f"{l['from']}>{l['to']}" for l in path)]

        def param(value, typ):
            params.append((f"p{len(params)}", typ, value))
            return f"@p{len(params) - 1}"

        select, group = [], []
        grain = plan.get("time_grain") or "none"
        if grain != "none":
            if not m.get("time_column"):
                raise ValueError(f"'{m['label']}' has no date, so it can't be shown over time.")
            select.append(GRAINS[grain].format(d=self.date_expr("t", base, m["time_column"])) + " AS `period`")
            group.append("`period`")
        for d in plan.get("group_by") or []:
            if d not in dims:
                raise ValueError(f"'{d}' can't group '{m['label']}'. Options: {sorted(dims)}")
            col = dims[d]["column"]
            name = col if sum(dims.get(x, {}).get("column") == col for x in plan["group_by"]) == 1 else re.sub(r"\W+", "_", d)
            select.append(f"{alias(dims[d]['path'])}.`{col}` AS `{name}`")
            group.append(f"`{name}`")
        if (plan.get("start") or plan.get("end")) and not m.get("time_column"):
            raise ValueError(f"'{m['label']}' has no date, so it can't be limited to a period. "
                             "Use a metric that has a date, or SQL.")
        if plan.get("start") or plan.get("end"):
            d = self.date_expr("t", base, m["time_column"])
            if plan.get("start"):
                where.append(f"{d} >= {param(plan['start'], 'DATE')}")
            if plan.get("end"):
                where.append(f"{d} <= {param(plan['end'], 'DATE')}")
        for f in plan.get("filters") or []:
            d = dims.get(f.get("column"))
            if not d:
                raise ValueError(f"Can't filter on '{f.get('column')}'. Options: {sorted(dims)}")
            vals = f.get("values") if isinstance(f.get("values"), list) else [f.get("values", f.get("value"))]
            if d["values"]:
                bad = [v for v in vals if v not in d["values"]]
                if bad:
                    raise ValueError(f"{f['column']} has no value {bad}. It has {d['values']}")
            typ = "BOOL" if d["type"] == "boolean" else "STRING"
            if typ == "STRING":
                vals = [str(v) for v in vals]
            neg = "NOT " if f.get("op") == "not_in" else ""
            where.append(f"{alias(d['path'])}.`{d['column']}` {neg}IN UNNEST({param(vals, 'ARRAY<' + typ + '>')})")
        sql = ("SELECT\n  " + ",\n  ".join(select + [f"{m['sql']} AS `value`", "COUNT(*) AS `rows`"])
               + f"\nFROM {model_ref(base)} AS t" + "".join("\n" + j for j in joins)
               + ("\nWHERE " + "\n  AND ".join(where) if where else "")
               + ("\nGROUP BY " + ", ".join(group) if group else "")
               + ("\nORDER BY `period`" if grain != "none" else "\nORDER BY `value` DESC" if group else "")
               + "\nLIMIT 1001")
        return sql, params

    def run_sql(self, sql, params=(), max_rows=None):
        return self.backend.query(sql, params, max_rows=max_rows)

    @staticmethod
    def inline_params(sql, params):
        """The SQL with its parameters written in, for showing it and for saving it elsewhere."""
        for name, typ, value in sorted(params, key=lambda p: -len(p[0])):
            if typ.startswith("ARRAY<"):
                txt = "UNNEST([" + ", ".join(lit(v) if isinstance(v, str) else str(v).upper() for v in value) + "])"
                sql = sql.replace(f"UNNEST(@{name})", txt)
            else:
                sql = sql.replace(f"@{name}", f"DATE {lit(value)}" if typ == "DATE" else lit(value))
        return sql

    def catalog_text(self, metrics=None):
        """What the AI is told about the data: views, columns, values, links, date ranges, approved metrics."""
        lines = []
        for t, cols in self.meta.items():
            tb = self.ontology["tables"][t]
            lines.append(f"\nTABLE {t} -- {tb['entity']}: {tb['description']}")
            for c, m in cols.items():
                bits = [m["type"]]
                if m["hashed"]:
                    bits.append("a code that stands for a person: only for joining or counting distinct")
                if m["values"]:
                    bits.append("values: " + ", ".join(repr(v) for v in m["values"]))
                if f"{t}.{c}" in self.ontology.get("date_ranges", {}):
                    bits.append("from {} to {}".format(*self.ontology["date_ranges"][f"{t}.{c}"]))
                if m["meaning"]:
                    bits.append(m["meaning"])
                lines.append(f"  {c}: " + "; ".join(bits))
        lines.append("\nLINKS (child -> parent):")
        lines += [f"  {l['from']} -> {l['to']}" + (f" ({l['link_name']})" if l.get("link_name") else "")
                  for l in self.ontology["links"]]
        if metrics:
            lines.append("\nAPPROVED METRICS:")
            for m in metrics.values():
                lines.append(f"  {m['name']}: {m['label']} [{m['unit']}] table {m['table']}, "
                             f"date {m.get('time_column') or 'none'} -- {m['definition']}")
                lines.append(f"     group_by/filter options: {list(self.dimensions_for(m['table']))}")
        return "\n".join(lines)


# =====================================================================================================
# Metrics: the AI proposes, the code tests each one on the data, you approve
# =====================================================================================================
METRIC_SYSTEM = ("You design business metrics on clean database views. Each metric is ONE aggregate SQL expression "
                 "(BigQuery) over ONE table, using the alias t. Answer only with JSON.")


def metric_problems(eng, m):
    """Why a metric can't be trusted, or [] if it passes: its columns exist, its text values exist, it runs,
    and the result makes sense. Sets m['preview'] to its value on the whole data."""
    meta = eng.meta
    t = m.get("table")
    if t not in meta:
        return [f"no table '{t}'"]
    probs = []
    if re.search(r"\b(SELECT|FROM|JOIN|INSERT|UPDATE|DELETE|DROP)\b|;", m.get("sql", ""), re.I):
        probs.append("the SQL must be one expression, not a query")
    for c in re.findall(r"\bt\.`?(\w+)`?", m.get("sql", "")):
        if c not in meta[t]:
            probs.append(f"{t} has no column '{c}'")
        elif meta[t][c]["hashed"] and not re.search(rf"COUNT\(\s*DISTINCT\s+t\.`?{c}`?\s*\)", m["sql"], re.I):
            probs.append(f"'{c}' is a personal code: it can only be counted, not shown")
    for c, v in re.findall(r"\bt\.`?(\w+)`?\s*(?:=|!=|<>)\s*'([^']*)'", m.get("sql", "")):
        if c in meta[t] and meta[t][c]["values"] and v not in meta[t][c]["values"]:
            probs.append(f"{c} never equals '{v}' (it has {meta[t][c]['values']})")
    tc = m.get("time_column")
    if tc and (tc not in meta[t] or meta[t][tc]["type"] != "date"):
        probs.append(f"'{tc}' is not a date column of {t}")
    if probs:
        return probs
    try:
        value = eng.run_sql(f"SELECT {m['sql']} AS value FROM {eng.backend.model_ref(t)} AS t").iloc[0, 0]
    except Exception as e:
        return ["the query failed: " + str(getattr(e, "message", e)).splitlines()[0][:200]]
    m["preview"] = None if pd.isna(value) else float(value)
    if m["preview"] is None:
        probs.append("gives no value on your data")
    elif m.get("unit") == "%" and not 0 <= m["preview"] <= 100:
        probs.append(f"a percentage of {m['preview']:.1f} is impossible")
    return probs


def propose_metrics(eng, ai, saved=(), manual=(), log=print):
    """Your saved metrics come back exactly as you approved them (your own `manual` ones win over them); the AI's
    ideas are only added under names you haven't used. Plain row counts are always offered too."""
    prompt = (eng.catalog_text() + "\n\nPropose 6 to 10 useful metrics a manager would track. Rules: one table per "
              "metric; 'sql' is an aggregate expression using alias t (COUNT, COUNTIF, SUM, AVG, COUNT(DISTINCT), "
              "SAFE_DIVIDE); compare a category only to its listed values, spelled exactly; for rates use "
              "100 * SAFE_DIVIDE(...) and unit '%'; 'time_column' is a date column of that table, or null. "
              "Return a JSON list of {\"name\": \"snake_case\", \"label\": \"...\", \"table\": \"...\", \"sql\": \"...\", "
              "\"time_column\": ..., \"unit\": \"...\", \"higher_is\": \"better|worse|neutral\", "
              "\"definition\": \"one plain sentence\"}.")
    try:
        found = ai.ask_json(METRIC_SYSTEM, prompt, expect=list)
    except Exception as e:
        log(f"The AI could not propose metrics: {e}")
        found = []
    ai_found = bool(found)
    found += [{"name": f"{t}_count", "auto": ai_found,
               "label": f"Number of {eng.ontology['tables'][t]['entity'].lower()} records", "table": t, "sql": "COUNT(*)",
               "time_column": next((c for c, m in eng.meta[t].items() if m["type"] == "date"), None),
               "unit": "rows", "higher_is": "neutral", "definition": f"Count of rows in {t}."} for t in eng.meta]
    out, seen = [], set()
    for m in list(manual) + [{**s, "saved": True} for s in saved] + [x for x in found if isinstance(x, dict)]:
        if m.get("name") in seen or not m.get("sql"):
            continue
        m = {"time_column": None, "unit": "", "higher_is": "neutral", "definition": "", "label": m.get("name"), **m}
        seen.add(m["name"])
        m["problems"] = metric_problems(eng, m)
        out.append(m)
    return out
