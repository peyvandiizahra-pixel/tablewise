"""Stage 2: one clean view per table, built from the reviewed ontology. Raw tables are never changed.
The SQL is written in BigQuery's dialect; the local backend translates it for DuckDB."""
from __future__ import annotations

import pandas as pd

from .ontology import BOOL_FALSE, BOOL_TRUE, MISSING_WORDS

MODES = ["exact", "format", "zeros"]


def lit(v):
    """A safe SQL string literal."""
    return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"


def text_sql(c):
    """Trimmed text; words like 'n/a' or 'unknown' become NULL."""
    base = f"TRIM(CAST(`{c}` AS STRING))"
    return f"(CASE WHEN LOWER({base}) IN ({', '.join(lit(x) for x in MISSING_WORDS)}) THEN NULL ELSE {base} END)"


def normalise_sql(s, mode):
    if mode == "exact":
        return s
    f = f"REGEXP_REPLACE(LOWER({s}), r'[^a-z0-9]', '')"
    return f if mode == "format" else f"REGEXP_REPLACE({f}, r'^([a-z]*)0+([0-9])', r'\\1\\2')"


def id_text_sql(c, bq_type):
    """An ID as text, written the way Stage 1 compared it: a whole-number decimal like 2071.0 becomes '2071'."""
    if bq_type in ("FLOAT", "FLOAT64"):
        return (f"(CASE WHEN `{c}` = TRUNC(`{c}`) THEN CAST(SAFE_CAST(`{c}` AS INT64) AS STRING) "
                f"ELSE CAST(`{c}` AS STRING) END)")
    return text_sql(c)


def iso_sql(s):
    """'2024-01-15T10:30:00.123Z' or '2024-01-15 10:30+01:00' -> DATETIME: the time as written, the zone dropped."""
    bare = f"REGEXP_REPLACE(REPLACE({s}, ' ', 'T'), r'(Z|[+-][0-9]{{2}}:?[0-9]{{2}})$', '')"
    return ", ".join([f"COALESCE(SAFE.PARSE_DATETIME('%Y-%m-%dT%H:%M:%E*S', {bare})",
                      f"SAFE.PARSE_DATETIME('%Y-%m-%dT%H:%M:%S', {bare})",
                      f"SAFE.PARSE_DATETIME('%Y-%m-%dT%H:%M', {bare}))"])


def as_date_sql(expr, sp):
    native = sp["bq_type"] in ("DATE", "DATETIME", "TIMESTAMP")
    return expr if (native and sp["bq_type"] == "DATE") or (not native and not sp["has_time"]) else f"DATE({expr})"


class LinkGroups:
    """Columns joined to each other must be written the same way in every view: one group per set of linked
    columns, compared in the strictest way any of its links needs, and coded if any member is personal."""

    def __init__(self, ontology):
        self.parent = {}
        links, tables = ontology["links"], ontology["tables"]
        for l in links:
            self.parent[self.find((l["from_table"], l["from_column"]))] = self.find((l["to_table"], l["to_column"]))
        groups = {}
        for l in links:
            for node in ((l["from_table"], l["from_column"]), (l["to_table"], l["to_column"])):
                groups.setdefault(self.find(node), set()).add(node)
        self.mode, self.hashed, self.string = {}, {}, {}
        for root, members in groups.items():
            modes = [l["match"] for l in links if self.find((l["from_table"], l["from_column"])) == root]
            self.mode[root] = max(modes, key=MODES.index)
            self.hashed[root] = any(tables[t]["columns"][c]["personal"] in ("hash", "drop") for t, c in members)
            types = {tables[t]["columns"][c]["bq_type"] for t, c in members}
            self.string[root] = self.mode[root] != "exact" or self.hashed[root] or len(types) > 1 or "STRING" in types

    def find(self, x):
        while self.parent.setdefault(x, x) != x:
            x = self.parent[x]
        return x

    def __contains__(self, node):
        return node in self.parent


def column_sql(ontology, groups, t, c, sp):
    """(output name, SQL expression) for one column, or None if it's removed from the view."""
    node = (t, c)
    in_group = node in groups
    if sp["personal"] == "drop" and not in_group:
        return None
    if sp["type"] == "identifier" or in_group:
        if in_group and not groups.string[groups.find(node)]:
            return c, f"`{c}`"                                       # e.g. integer ids matched exactly
        mode = groups.mode[groups.find(node)] if in_group else "exact"
        expr = normalise_sql(id_text_sql(c, sp["bq_type"]), mode)
        if sp["personal"] == "hash" or (in_group and groups.hashed[groups.find(node)]):
            expr = f"SUBSTR(TO_HEX(SHA256(CONCAT({lit(ontology['salt'])}, {expr}))), 1, 16)"
        return c, expr
    native = sp["bq_type"] != "STRING"
    if sp["type"] == "number":
        if native:
            expr = f"CAST(`{c}` AS FLOAT64)"
        else:
            clean = f"REGEXP_REPLACE({text_sql(c)}, r'[£$€,\\s]', '')"
            expr = f"SAFE_CAST(REGEXP_EXTRACT({clean}, r'^(-?[0-9]+(?:\\.[0-9]+)?)') AS FLOAT64)"
            factors = sp.get("unit_factors") or {}
            if factors:      # several units in one column: each value in the main unit ('2 hrs' -> 120 mins)
                unit = f"LOWER(REGEXP_EXTRACT({clean}, r'([A-Za-z%]+)$'))"
                whens = " ".join(f"WHEN {lit(u)} THEN {float(f)!r}" for u, f in factors.items())
                expr = f"({expr} * CASE COALESCE({unit}, '') WHEN '' THEN 1 {whens} END)"   # another unit: empty
        rule = sp.get("negative_rule", "keep")
        if rule == "empty":
            expr = f"IF({expr} < 0, NULL, {expr})"
        elif rule == "abs":
            expr = f"ABS({expr})"
        return c, expr
    if sp["type"] == "date":
        if native:
            expr = f"`{c}`"
        else:
            parts = []
            for f in sp["date_formats"]:
                if f == "ISO":
                    parts.append(iso_sql(text_sql(c)))
                elif "%H" in f:
                    parts.append(f"SAFE.PARSE_DATETIME({lit(f)}, {text_sql(c)})")
                elif sp["has_time"]:
                    parts.append(f"DATETIME(SAFE.PARSE_DATE({lit(f)}, {text_sql(c)}))")
                else:
                    parts.append(f"SAFE.PARSE_DATE({lit(f)}, {text_sql(c)})")
            expr = f"COALESCE({', '.join(parts)})" if len(parts) > 1 else (parts[0] if parts else "CAST(NULL AS DATE)")
        if sp["personal"] == "age":
            return f"{c}_age_years", f"CAST(FLOOR(DATE_DIFF(CURRENT_DATE(), {as_date_sql(expr, sp)}, DAY) / 365.25) AS INT64)"
        if sp["personal"] == "drop":
            return None
        return c, expr
    if sp["personal"] == "drop":
        return None
    if sp["type"] == "boolean":
        if native:
            return c, f"`{c}`"
        low = f"LOWER({text_sql(c)})"
        return c, (f"(CASE WHEN {low} IN ({', '.join(map(lit, BOOL_TRUE))}) THEN TRUE "
                   f"WHEN {low} IN ({', '.join(map(lit, BOOL_FALSE))}) THEN FALSE END)")
    if sp["type"] == "category" and sp["mapping"]:
        whens = " ".join(f"WHEN {lit(old)} THEN {lit(new) if new is not None else 'NULL'}"
                         for old, new in sp["mapping"].items())
        return c, f"(CASE {text_sql(c)} {whens} ELSE {text_sql(c)} END)"
    return c, text_sql(c) if not native else f"`{c}`"


def view_sql(backend, ontology, groups, t):
    tb = ontology["tables"][t]
    exprs = {}
    for c, sp in tb["columns"].items():
        col = column_sql(ontology, groups, t, c, sp)
        if col:
            exprs[col[0]] = col[1]
    select = ",\n  ".join(f"{e} AS `{n}`" for n, e in exprs.items())
    sql = f"SELECT\n  {select}\nFROM {backend.raw_ref(t)} AS _raw\nWHERE TRUE"     # BigQuery wants a WHERE next to QUALIFY
    keys = ", ".join(exprs[k] for k in tb["key"] if k in exprs)
    if tb["dedupe"] and keys:
        filled = " + ".join(f"IF(`{c}` IS NULL, 0, 1)" for c in tb["columns"])
        # the most complete row wins; a tie is broken by the row's content, so it's the same row every time
        sql += (f"\nQUALIFY ROW_NUMBER() OVER (PARTITION BY {keys} "
                f"ORDER BY {filled} DESC, FARM_FINGERPRINT(TO_JSON_STRING(_raw))) = 1")
    return exprs, sql


def build_views(backend, ontology, log=print):
    """Creates the views, then checks them. Adds date ranges, unique keys and view columns to the ontology.
    Returns the checks table."""
    groups = LinkGroups(ontology)
    backend.prepare_model("Clean views built from the reviewed ontology. Raw tables are never changed.")
    view_columns = {}
    for t, tb in ontology["tables"].items():
        exprs, sql = view_sql(backend, ontology, groups, t)
        backend.create_view(t, sql, f"{tb['entity']}: {tb['description']}")
        view_columns[t] = exprs
        log(f"View ready: {t} ({len(exprs)} columns)")

    def one_row(sql):
        return backend.query(sql).iloc[0]

    checks, date_ranges, unique_keys = [], {}, {}
    for t, tb in ontology["tables"].items():
        cols = view_columns[t]
        # values that were filled in the raw data but couldn't be read as a number / date / yes-no
        parsed = [c for c, sp in tb["columns"].items()
                  if sp["type"] in ("number", "date", "boolean") and sp["bq_type"] == "STRING" and c in cols]
        if parsed:
            # read with "keep negatives", so negatives you chose to empty aren't counted as unreadable
            sel = ", ".join(f"COUNTIF({text_sql(c)} IS NOT NULL AND "
                            f"{column_sql(ontology, groups, t, c, {**tb['columns'][c], 'negative_rule': 'keep'})[1]} IS NULL) AS `{c}`"
                            for c in parsed)
            r = one_row(f"SELECT {sel} FROM {backend.raw_ref(t)}")
            for c in parsed:
                checks.append({"table": t, "check": f"{c} readable", "ok": int(r[c]) == 0,
                               "detail": f"{int(r[c])} filled values could not be read as {tb['columns'][c]['type']}"})
        key = [k for k in tb["key"] if k in cols]
        if key:
            kexpr = f"`{key[0]}`" if len(key) == 1 else "CONCAT(" + ", '|', ".join(f"CAST(`{k}` AS STRING)" for k in key) + ")"
            r = one_row(f"SELECT COUNT(*) AS n, COUNT(DISTINCT {kexpr}) AS d FROM {backend.model_ref(t)}")
            unique_keys[t] = int(r.n) == int(r.d)
            checks.append({"table": t, "check": f"key {'+'.join(key)} unique", "ok": int(r.n) == int(r.d),
                           "detail": f"{int(r.n)} rows, {int(r.d)} distinct keys"})
        dates = [c for c, sp in tb["columns"].items() if sp["type"] == "date" and c in cols]
        if dates:
            r = one_row("SELECT " + ", ".join(f"MIN({as_date_sql(f'`{d}`', tb['columns'][d])}) AS `min_{d}`, "
                                             f"MAX({as_date_sql(f'`{d}`', tb['columns'][d])}) AS `max_{d}`" for d in dates)
                        + f" FROM {backend.model_ref(t)}")
            for d in dates:
                date_ranges[f"{t}.{d}"] = (str(r[f"min_{d}"])[:10], str(r[f"max_{d}"])[:10])
    for l in ontology["links"]:
        r = one_row(f"SELECT COUNTIF(c.`{l['from_column']}` IS NOT NULL) AS filled, COUNTIF(p.k IS NOT NULL) AS matched "
                    f"FROM {backend.model_ref(l['from_table'])} c LEFT JOIN (SELECT DISTINCT `{l['to_column']}` AS k "
                    f"FROM {backend.model_ref(l['to_table'])}) p ON c.`{l['from_column']}` = p.k")
        pct = 100 * int(r.matched) / max(int(r.filled), 1)
        checks.append({"table": l["from_table"], "check": f"{l['from']} -> {l['to']} joins", "ok": pct >= 95,
                       "detail": f"{pct:.1f}% of filled values find their {l['to_table']} row"})
    ontology["date_ranges"] = date_ranges
    ontology["unique_keys"] = unique_keys
    ontology["view_columns"] = {t: list(c) for t, c in view_columns.items()}
    out = pd.DataFrame(checks, columns=["table", "check", "ok", "detail"])
    out.insert(0, "result", out.pop("ok").map({True: "PASS", False: "CHECK"}))
    return out
