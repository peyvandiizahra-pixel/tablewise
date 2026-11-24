"""Where the data lives. Tablewise writes its SQL once, in BigQuery's dialect; each backend runs it.

- DuckDBBackend: a folder of CSV / Parquet / Excel files, analysed on your own computer. No cloud account needed.
- BigQueryBackend: a BigQuery dataset (needs `pip install tablewise[bigquery]` and Google credentials).
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


class NotFoundError(Exception):
    """A table that doesn't exist (yet)."""


@dataclass
class DryRun:
    columns: list = field(default_factory=list)
    bytes: int = 0


class Backend:
    """What the rest of Tablewise needs from a data store."""
    label = "data"

    def list_tables(self) -> list[str]: ...
    def read_table(self, table, max_rows) -> tuple[pd.DataFrame, dict, int]: ...
    def read_column(self, table, column, max_values) -> pd.Series: ...
    def raw_ref(self, table) -> str: ...
    def model_ref(self, table) -> str: ...
    def prepare_model(self, description): ...
    def create_view(self, table, sql, description): ...
    def query(self, sql, params=(), max_rows=None) -> pd.DataFrame: ...
    def dry_run(self, sql, params=()) -> DryRun: ...
    def load_actions(self, columns) -> pd.DataFrame: ...
    def save_actions(self, df): ...

    def shown_sql(self, sql):
        """The SQL as this backend runs it (for showing it to the user)."""
        return sql

    @property
    def actions_location(self):
        return "the tasks table"


# =====================================================================================================
# BigQuery SQL -> DuckDB SQL
# =====================================================================================================
FUNCS = ["SAFE.PARSE_DATETIME", "SAFE.PARSE_DATE", "PARSE_DATETIME", "PARSE_DATE", "SAFE_CAST", "REGEXP_EXTRACT",
         "REGEXP_REPLACE", "REGEXP_CONTAINS", "COUNTIF", "SAFE_DIVIDE", "DATE_TRUNC", "DATETIME_TRUNC",
         "TIMESTAMP_TRUNC", "DATE_DIFF", "DATETIME_DIFF", "DATE_ADD", "DATE_SUB", "FORMAT_DATE", "FORMAT_DATETIME",
         "TO_HEX", "DATETIME", "DATE", "FARM_FINGERPRINT", "TO_JSON_STRING", "TRUNC", "LOGICAL_OR", "LOGICAL_AND"]
_UNITS = {"DAY": "day", "WEEK": "week", "MONTH": "month", "QUARTER": "quarter", "YEAR": "year", "HOUR": "hour",
          "MINUTE": "minute", "SECOND": "second"}


def _strings(sql):
    """BigQuery literals -> DuckDB literals (\\' -> '', r'..' -> '..')."""
    out, i = [], 0
    while i < len(sql):
        ch = sql[i]
        raw = ch in "rR" and i + 1 < len(sql) and sql[i + 1] == "'" and (i == 0 or not (sql[i - 1].isalnum() or sql[i - 1] == "_"))
        if ch == "'" or raw:
            j = i + (2 if raw else 1)
            buf = []
            while j < len(sql):
                c = sql[j]
                if c == "\\" and not raw and j + 1 < len(sql):
                    buf.append(sql[j + 1])
                    j += 2
                    continue
                if c == "'":
                    break
                buf.append(c)
                j += 1
            out.append("'" + "".join(buf).replace("'", "''") + "'")
            i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _args(s, start):
    """Split the arguments of a call whose '(' is at s[start]. Returns (args, index of the closing ')')."""
    depth, cur, args, i, q = 0, [], [], start, False
    while i < len(s):
        c = s[i]
        if c == "'":
            q = not q
        if not q:
            if c == "(":
                depth += 1
                if depth == 1:
                    i += 1
                    continue
            elif c == ")":
                depth -= 1
                if depth == 0:
                    args.append("".join(cur).strip())
                    return args, i
            elif c == "," and depth == 1:
                args.append("".join(cur).strip())
                cur = []
                i += 1
                continue
        cur.append(c)
        i += 1
    raise ValueError("unbalanced brackets in SQL")


def _unit(u):
    u = re.sub(r"\(.*\)", "", u).strip().upper()          # WEEK(MONDAY) -> WEEK
    return _UNITS.get(u, u.lower())


def _rewrite(name, a):
    if name in ("SAFE.PARSE_DATE", "PARSE_DATE"):
        return f"CAST(try_strptime({a[1]}, {a[0]}) AS DATE)"
    if name in ("SAFE.PARSE_DATETIME", "PARSE_DATETIME"):
        return f"try_strptime({a[1]}, {a[0]})"
    if name == "SAFE_CAST":
        return f"TRY_CAST({a[0]})"
    if name == "REGEXP_EXTRACT":
        return f"regexp_extract({a[0]}, {a[1]}, 1)" if "(" in a[1] else f"regexp_extract({a[0]}, {a[1]})"
    if name == "REGEXP_REPLACE":
        return f"regexp_replace({a[0]}, {a[1]}, {a[2]}, 'g')"
    if name == "REGEXP_CONTAINS":
        return f"regexp_matches({a[0]}, {a[1]})"
    if name == "COUNTIF":
        return f"count_if({a[0]})"
    if name == "SAFE_DIVIDE":
        return f"(CAST({a[0]} AS DOUBLE) / NULLIF({a[1]}, 0))"
    if name in ("DATE_TRUNC", "DATETIME_TRUNC", "TIMESTAMP_TRUNC"):
        unit = _unit(a[1])
        cast = " AS DATE" if name == "DATE_TRUNC" else " AS TIMESTAMP"
        return f"CAST(date_trunc('{unit}', {a[0]}){cast})"
    if name in ("DATE_DIFF", "DATETIME_DIFF"):
        return f"date_diff('{_unit(a[2])}', {a[1]}, {a[0]})"
    if name in ("DATE_ADD", "DATE_SUB"):
        return f"CAST(({a[0]} {'+' if name == 'DATE_ADD' else '-'} {a[1]}) AS DATE)"
    if name in ("FORMAT_DATE", "FORMAT_DATETIME"):
        return f"strftime({a[1]}, {a[0]})"
    if name == "TO_HEX":
        inner = re.match(r"SHA256\((.*)\)$", a[0], re.S)
        return f"sha256({inner.group(1)})" if inner else f"to_hex({a[0]})"
    if name == "DATETIME":
        return f"CAST({a[0]} AS TIMESTAMP)"
    if name == "DATE":
        return f"CAST({a[0]} AS DATE)"
    if name == "FARM_FINGERPRINT":
        return f"hash({a[0]})"
    if name == "TO_JSON_STRING":
        return f"CAST(to_json({a[0]}) AS VARCHAR)"
    if name == "TRUNC":
        return f"trunc({a[0]})"
    if name == "LOGICAL_OR":
        return f"bool_or({a[0]})"
    if name == "LOGICAL_AND":
        return f"bool_and({a[0]})"
    raise KeyError(name)


def _sql_value(v):
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


def translate(sql, params=()):
    """BigQuery Standard SQL (as Tablewise writes it, and the common functions an AI uses) -> DuckDB SQL.
    params: (name, type, value) tuples, written into the SQL as safely quoted literals."""
    for name, typ, value in sorted(params, key=lambda p: -len(p[0])):
        if typ.startswith("ARRAY<"):
            vals = ", ".join(_sql_value(v) for v in value) or "NULL"
            sql = re.sub(rf"UNNEST\(\s*@{name}\b\s*\)", f"({vals})", sql)
            sql = re.sub(rf"IN\s*\(\(({re.escape(vals)})\)\)", rf"IN (\1)", sql)
        elif typ == "DATE":
            sql = re.sub(rf"@{name}\b", f"DATE {_sql_value(str(value))}", sql)
        else:
            sql = re.sub(rf"@{name}\b", _sql_value(value), sql)
    sql = re.sub(r"\bIN\s+UNNEST\s*\(\s*\[(.*?)\]\s*\)", r"IN (\1)", sql, flags=re.S)
    sql = re.sub(r"OPTIONS\(description=(?:'(?:[^'\\]|\\.)*')\)", "", sql)
    sql = _strings(sql)
    sql = re.sub(r"`([\w-]+)\.([\w-]+)\.([\w-]+)`", r'"\2"."\3"', sql)
    sql = re.sub(r"`([\w-]+)\.([\w-]+)`", r'"\1"."\2"', sql)
    sql = re.sub(r"`([^`]+)`", r'"\1"', sql)
    sql = re.sub(r"\bCURRENT_DATE\(\)", "current_date", sql, flags=re.I)
    sql = re.sub(r"\bCURRENT_DATETIME\(\)|\bCURRENT_TIMESTAMP\(\)", "current_timestamp", sql, flags=re.I)
    names = "|".join(re.escape(f) for f in sorted(FUNCS, key=len, reverse=True))
    # BigQuery function names in any case -> UPPER once; the rewrites below write lower-case DuckDB names, so they
    # are never rewritten again
    sql = re.sub(rf"(?<![\w.])({names})(\s*\()", lambda m: m.group(1).upper() + m.group(2), sql, flags=re.I)
    pat = re.compile(rf"(?<![\w.])({names})\s*\(")
    while True:
        ms = list(pat.finditer(sql))
        if not ms:
            break
        m = ms[-1]
        args, end = _args(sql, m.end() - 1)
        sql = sql[:m.start()] + _rewrite(m.group(1), args) + sql[end + 1:]
    sql = sql.replace("%E*S", "%S")          # BigQuery: seconds with any fraction; DuckDB has no %E*S
    for bq, duck in (("FLOAT64", "DOUBLE"), ("INT64", "BIGINT"), ("STRING", "VARCHAR"), ("BOOL", "BOOLEAN"),
                     ("NUMERIC", "DECIMAL(38, 9)"), ("BIGNUMERIC", "DECIMAL(38, 9)")):
        sql = re.sub(rf"\bAS\s+{bq}\b", f"AS {duck}", sql, flags=re.I)
    return sql


# =====================================================================================================
# Local files with DuckDB
# =====================================================================================================
_DUCK_TO_BQ = {"VARCHAR": "STRING", "BIGINT": "INT64", "INTEGER": "INT64", "SMALLINT": "INT64", "TINYINT": "INT64",
               "HUGEINT": "INT64", "UBIGINT": "INT64", "UINTEGER": "INT64", "DOUBLE": "FLOAT64", "FLOAT": "FLOAT64",
               "REAL": "FLOAT64", "BOOLEAN": "BOOL", "DATE": "DATE", "TIMESTAMP": "DATETIME",
               "TIMESTAMP WITH TIME ZONE": "TIMESTAMP"}
DATA_FILES = (".csv", ".tsv", ".txt", ".parquet", ".xlsx", ".xls", ".json", ".jsonl")


def table_name(path):
    """'Customer List (2025).csv' -> 'customer_list_2025'"""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", Path(path).stem).strip("_").lower()
    return name if name and not name[0].isdigit() else f"t_{name}"


class DuckDBBackend(Backend):
    """Every data file in a folder becomes a raw table. Text files are read as text, exactly as written, so
    nothing is cleaned silently: that's Tablewise's job, and you approve it."""
    label = "local files"

    def __init__(self, data_dir, db_file):
        import duckdb
        self.data_dir, self.db_file = Path(data_dir), Path(db_file)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"No folder {self.data_dir}")
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.db_file))
        for schema in ("raw", "model", "app"):
            self.con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        self.files = {}
        for f in sorted(self.data_dir.iterdir()):
            if f.suffix.lower() in DATA_FILES and not f.name.startswith((".", "~")):
                self.files[table_name(f)] = f
        if not self.files:
            raise FileNotFoundError(f"No data files ({', '.join(DATA_FILES)}) in {self.data_dir}")
        for t, f in self.files.items():
            self._load(t, f)

    def _load(self, t, f):
        target = f'"raw"."{t}"'
        path = str(f).replace("'", "''")
        ext = f.suffix.lower()
        if ext in (".csv", ".tsv", ".txt"):
            self.con.execute(f"CREATE OR REPLACE TABLE {target} AS SELECT * FROM read_csv('{path}', all_varchar=true, "
                             f"header=true, null_padding=true, ignore_errors=false)")
        elif ext == ".parquet":
            self.con.execute(f"CREATE OR REPLACE TABLE {target} AS SELECT * FROM read_parquet('{path}')")
        elif ext in (".json", ".jsonl"):
            self.con.execute(f"CREATE OR REPLACE TABLE {target} AS SELECT * FROM read_json_auto('{path}')")
        else:
            df = pd.read_excel(f, dtype=str)
            self.con.register("_xl", df)
            self.con.execute(f"CREATE OR REPLACE TABLE {target} AS SELECT * FROM _xl")
            self.con.unregister("_xl")

    def list_tables(self):
        return list(self.files)

    def _schema(self, schema, t):
        rows = self.con.execute("SELECT column_name, data_type FROM information_schema.columns "
                                "WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position", [schema, t]).fetchall()
        return {c: _DUCK_TO_BQ.get(str(d).upper().split("(")[0], "STRING") for c, d in rows}

    def read_table(self, table, max_rows):
        schema = self._schema("raw", table)
        total = self.con.execute(f'SELECT COUNT(*) FROM "raw"."{table}"').fetchall()[0][0]
        df = self.con.execute(f'SELECT * FROM "raw"."{table}" LIMIT {int(max_rows)}').df()
        df = df.astype(object).where(df.notna(), None)
        return df, schema, total

    def read_column(self, table, column, max_values):
        return self.con.execute(f'SELECT "{column}" FROM "raw"."{table}" LIMIT {int(max_values)}').df()[column]

    def raw_ref(self, t):
        return f"`local.raw.{t}`"

    def model_ref(self, t):
        return f"`local.model.{t}`"

    def model_name(self, ref):
        m = re.fullmatch(r"`?local\.model\.([\w-]+)`?", ref)
        return m.group(1) if m else None

    def prepare_model(self, description):
        self.con.execute('CREATE SCHEMA IF NOT EXISTS "model"')

    def create_view(self, table, sql, description):
        self.con.execute(f'CREATE OR REPLACE VIEW "model"."{table}" AS\n{translate(sql)}')
        try:
            self.con.execute(f'COMMENT ON VIEW "model"."{table}" IS ' + _sql_value(description[:1000]))
        except Exception:
            pass

    def query(self, sql, params=(), max_rows=None):
        df = self.con.execute(translate(sql, params)).df()
        return df.head(max_rows) if max_rows else df

    def dry_run(self, sql, params=()):
        res = self.con.execute(f"SELECT * FROM ({translate(sql, params)}) AS _q LIMIT 0")
        cols = [d[0] for d in res.description]
        res.fetchall()                      # finish the result, or DuckDB keeps the connection busy
        return DryRun(columns=cols, bytes=0)

    def shown_sql(self, sql):
        return translate(sql)

    @property
    def actions_location(self):
        return f"the table app.actions in {self.db_file.name}"

    def load_actions(self, columns):
        exists = self.con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'app' "
                                  "AND table_name = 'actions'").fetchall()[0][0]
        if not exists:
            return pd.DataFrame(columns=columns)
        return self.con.execute('SELECT * FROM "app"."actions"').df()[columns]

    def save_actions(self, df):
        self.con.register("_actions", df)
        self.con.execute('CREATE OR REPLACE TABLE "app"."actions" AS SELECT * FROM _actions')
        self.con.unregister("_actions")

    def close(self):
        self.con.close()


# =====================================================================================================
# BigQuery
# =====================================================================================================
SKIP_TYPES = {"RECORD", "STRUCT", "GEOGRAPHY", "JSON", "BYTES", "RANGE"}


class BigQueryBackend(Backend):
    """A BigQuery dataset. Reading the tables uses BigQuery's row-reading API (no query, so it's free); the clean
    views go to a second dataset, <dataset>_model. Credentials: `gcloud auth application-default login`."""
    label = "BigQuery"

    def __init__(self, project, dataset, model_suffix="_model", max_bytes=1_000_000_000):
        try:
            from google.cloud import bigquery
            from google.api_core.exceptions import NotFound
        except ImportError as e:
            raise ImportError("BigQuery support needs: pip install 'tablewise[bigquery]'") from e
        self.bq, self.NotFound = bigquery, NotFound
        self.project, self.dataset = project, dataset
        self.model_dataset = dataset + model_suffix
        self.max_bytes = max_bytes
        self.client = bigquery.Client(project=project)
        self.location = self.client.get_dataset(f"{project}.{dataset}").location
        self.notes = []

    def list_tables(self):
        out = []
        for item in self.client.list_tables(f"{self.project}.{self.dataset}"):
            if item.table_type == "TABLE":
                out.append(item.table_id)
            else:
                self.notes.append(f"{item.table_id}: skipped, it's a {item.table_type.lower()}")
        return out

    def read_table(self, table, max_rows):
        t = self.client.get_table(f"{self.project}.{self.dataset}.{table}")
        fields = [f for f in t.schema if f.field_type not in SKIP_TYPES and f.mode != "REPEATED"]
        df = self.client.list_rows(t, selected_fields=fields, max_results=max_rows).to_dataframe()
        return df, {f.name: f.field_type for f in fields}, t.num_rows

    def read_column(self, table, column, max_values):
        t = self.client.get_table(f"{self.project}.{self.dataset}.{table}")
        field = [f for f in t.schema if f.name == column]
        return self.client.list_rows(t, selected_fields=field, max_results=max_values).to_dataframe()[column]

    def raw_ref(self, t):
        return f"`{self.project}.{self.dataset}.{t}`"

    def model_ref(self, t):
        return f"`{self.project}.{self.model_dataset}.{t}`"

    def model_name(self, ref):
        m = re.fullmatch(rf"`?{re.escape(self.project)}\.{re.escape(self.model_dataset)}\.([\w-]+)`?", ref)
        return m.group(1) if m else None

    def prepare_model(self, description):
        ds = self.bq.Dataset(f"{self.project}.{self.model_dataset}")
        ds.location, ds.description = self.location, description
        self.client.create_dataset(ds, exists_ok=True)

    def create_view(self, table, sql, description):
        lit = "'" + description[:1000].replace("\\", "\\\\").replace("'", "\\'") + "'"
        self.client.query(f"CREATE OR REPLACE VIEW {self.model_ref(table)}\nOPTIONS(description={lit})\nAS\n{sql}").result()

    def _params(self, params):
        out = []
        for name, typ, value in params:
            if typ.startswith("ARRAY<"):
                out.append(self.bq.ArrayQueryParameter(name, typ[6:-1], value))
            else:
                out.append(self.bq.ScalarQueryParameter(name, typ, dt.date.fromisoformat(value) if typ == "DATE" else value))
        return out

    def query(self, sql, params=(), max_rows=None):
        cfg = self.bq.QueryJobConfig(query_parameters=self._params(params), maximum_bytes_billed=self.max_bytes)
        return self.client.query(sql, job_config=cfg).result(max_results=max_rows).to_dataframe()

    def dry_run(self, sql, params=()):
        cfg = self.bq.QueryJobConfig(query_parameters=self._params(params), dry_run=True, use_query_cache=False)
        job = self.client.query(sql, job_config=cfg)
        return DryRun(columns=[f.name for f in (job.schema or [])], bytes=job.total_bytes_processed or 0)

    @property
    def actions_location(self):
        return f"{self.project}.{self.model_dataset}.actions"

    def load_actions(self, columns):
        try:
            t = self.client.get_table(self.actions_location)
        except self.NotFound:          # only "no table yet": any other error stops here instead of wiping the tasks
            return pd.DataFrame(columns=columns)
        return self.client.list_rows(t).to_dataframe()[columns]

    def save_actions(self, df):
        cfg = self.bq.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.client.load_table_from_dataframe(df, self.actions_location, job_config=cfg).result()

