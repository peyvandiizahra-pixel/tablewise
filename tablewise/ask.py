"""Stage 4: a question becomes a plan from an approved metric (verified), checked SQL (unverified), or one
question back. Then a chart is picked from the shape of the result."""
from __future__ import annotations

import datetime as dt
import decimal
import json
import numbers
import re
import time
from pathlib import Path

import pandas as pd

PLANNER_SYSTEM = ("You turn a question about data into a query plan. Prefer an APPROVED METRIC: its numbers are "
                  "trusted. Write SQL only when no metric can answer. If the question can be read in ways that change "
                  "the answer, ask ONE short question instead of guessing. Answer only with JSON.")
FORBIDDEN = r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|EXPORT|CALL|DECLARE|EXECUTE|BEGIN|LOAD|ATTACH|COPY|INSTALL|PRAGMA)\b"


def strip_strings(sql):
    return re.sub(r"'(?:[^'\\]|\\.)*'", "''", re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S))


class Asker:
    def __init__(self, engine, ai, metrics, log_file=None):
        self.eng, self.ai, self.metrics = engine, ai, metrics
        self.log_file = Path(log_file) if log_file else None

    def planner_prompt(self, question):
        return (self.eng.catalog_text(self.metrics) + f"\n\nToday is {dt.date.today()}.\n\nQuestion: {question}\n\n"
                "Reply with ONE of these JSON objects:\n"
                "1. {\"kind\": \"metric\", \"metric\": \"<name>\", \"group_by\": [\"table.column\", ...at most 2], "
                "\"time_grain\": \"none|day|week|month\", \"start\": \"YYYY-MM-DD\" or null, \"end\": \"YYYY-MM-DD\" or null, "
                "\"filters\": [{\"column\": \"table.column\", \"op\": \"in|not_in\", \"values\": [...]}], "
                "\"explanation\": \"one plain sentence\"}\n"
                "   group_by and filters must come from that metric's options; filter values from the listed values.\n"
                "2. {\"kind\": \"sql\", \"sql\": \"one BigQuery Standard SQL SELECT; refer to tables by their plain names, "
                "e.g. FROM invoices AS i; never output a person's code\", \"explanation\": \"one plain sentence\"}\n"
                "3. {\"kind\": \"clarify\", \"question\": \"one short question\"}\n"
                "Relative dates (last month, this year) are relative to today; if that period is outside the date "
                "ranges listed, ask instead.")

    # ---------- checks for free SQL ----------
    def to_model_tables(self, sql):
        """Plain table names -> full names of the clean views. Anything else after FROM/JOIN is reported."""
        unknown, meta, b = [], self.eng.meta, self.eng.backend
        ctes = {n.lower() for n in re.findall(r"(?:\bWITH|,)\s*(\w+)\s+AS\s*\(", sql, re.I)}

        def swap(m):
            name = m.group(2).strip("`")
            short = name.split(".")[-1]
            if short in meta and (name == short or b.model_name(f"`{name}`") == short):
                return f"{m.group(1)} {b.model_ref(short)}"
            if name.lower() not in ctes and name.upper() != "UNNEST":
                unknown.append(name)
            return m.group(0)

        hidden = {}

        def hide(m):                       # EXTRACT(YEAR FROM x): that FROM is not a table
            hidden[f"__H{len(hidden)}__"] = m.group(0)
            return f"__H{len(hidden) - 1}__"
        out = re.sub(r"\b(EXTRACT|TRIM|SUBSTRING)\s*\([^()]*(?:\([^()]*\)[^()]*)*\)", hide, sql, flags=re.I)
        out = re.sub(r"\b(FROM|JOIN)\s+(`[^`]+`|[\w.\-]+)", swap, out, flags=re.I)
        for k, v in hidden.items():
            out = out.replace(k, v)
        return out, unknown

    def check_free_sql(self, sql):
        checks, meta = [], self.eng.meta
        add = lambda name, ok, detail: checks.append({"check": name, "passed": bool(ok), "detail": detail})
        code = strip_strings(sql).strip().rstrip(";")
        bad = sorted({b.upper() for b in re.findall(FORBIDDEN, code, re.I)})
        is_select = re.match(r"\s*(SELECT|WITH)\b", code, re.I) is not None
        add("Read-only, one query", is_select and not bad and ";" not in code, f"not allowed: {bad}" if bad else "ok")
        full, unknown = self.to_model_tables(sql)
        add("Only the clean views", not unknown, f"unknown tables {unknown}" if unknown else "ok")
        probs = []
        for col, val in re.findall(r"\b(\w+)`?\s*(?:=|!=|<>)\s*'([^']*)'", sql):
            known = [m["values"] for cols in meta.values() for c, m in cols.items() if c == col and m["values"]]
            if known and not any(val in k for k in known):
                probs.append(f"{col} never equals '{val}' (it has {known[0]})")
        add("Text values exist in the data", not probs, "; ".join(probs) or "ok")
        if all(c["passed"] for c in checks):
            try:
                dry = self.eng.backend.dry_run(full)
                hashed = {c.lower() for cols in meta.values() for c, m in cols.items() if m["hashed"]}
                leak = [c for c in dry.columns if c.lower() in hashed]
                add("The database accepts it (dry run)", True, f"would read {dry.bytes / 1e6:,.1f} MB" if dry.bytes else "ok")
                add("Within the size limit", dry.bytes <= self.eng.max_bytes, f"limit {self.eng.max_bytes / 1e9:g} GB")
                add("Shows no personal codes", not leak, f"outputs {leak}" if leak else "ok")
            except Exception as e:
                add("The database accepts it (dry run)", False, str(getattr(e, "message", e)).splitlines()[0][:300])
        return full, checks

    # ---------- the whole answer ----------
    def answer(self, question):
        start, msgs, checks = time.time(), self.planner_prompt(question), []
        res = {"question": question, "status": "failed", "kind": "", "verified": False, "title": question,
               "explanation": "", "sql": "", "params": [], "df": None, "checks": checks, "clarify": ""}
        if not self.ai.available and not self.ai.cache:
            checks.append({"check": "AI", "passed": False, "detail": "Asking questions needs the AI: add your Claude API key."})
            res["seconds"] = 0.0
            return res
        metrics = self.metrics
        for attempt in range(2):
            err, plan = None, {}
            try:
                plan = self.ai.ask_json(PLANNER_SYSTEM, msgs, expect=dict)
            except Exception as e:
                err = f"No usable reply: {e}"
            kind = plan.get("kind")
            res.update(kind=kind or "", explanation=plan.get("explanation", ""))
            try:
                if err and "AI is off" in err:
                    raise ValueError(err)
                if kind == "clarify":
                    res.update(status="clarify", clarify=plan.get("question", "Could you say a bit more?"))
                    break
                if kind == "metric":
                    if plan.get("metric") not in metrics:
                        raise ValueError(f"Unknown metric '{plan.get('metric')}'. Use one of {list(metrics)}")
                    sql, params = self.eng.compile_metric(plan, metrics)
                    m = metrics[plan["metric"]]
                    checks.append({"check": "Built from approved metric", "passed": True,
                                   "detail": f"{m['label']}: {m['definition']}"})
                    df = self.eng.run_sql(sql, params)
                    res.update(status="ok", verified=True, sql=sql, params=params, df=df, plan=plan,
                               title=m["label"] + (" by " + ", ".join(
                                   re.sub(r"^\w+\.(\w+)(?: via (.*))?$",
                                          lambda x: x.group(1) + (f" (via {x.group(2)})" if x.group(2) else ""), g)
                                   for g in plan["group_by"]) if plan.get("group_by") else "")
                               + (f" per {plan['time_grain']}" if plan.get("time_grain") not in (None, "none") else ""))
                    break
                if kind == "sql":
                    full, new_checks = self.check_free_sql(plan.get("sql", ""))
                    if not all(c["passed"] for c in new_checks):
                        raise ValueError("; ".join(f"{c['check']}: {c['detail']}" for c in new_checks if not c["passed"]))
                    checks.extend(new_checks)
                    res.update(status="ok", verified=False, sql=full, df=self.eng.run_sql(full, max_rows=1001))
                    break
                raise ValueError(err or "kind must be metric, sql or clarify")
            except Exception as e:
                err = str(getattr(e, "message", e)).splitlines()[0][:400]
            self.ai.forget(PLANNER_SYSTEM, msgs)       # this plan didn't work: don't answer the same question with it again
            checks.append({"check": f"Attempt {attempt + 1}", "passed": False, "detail": err})
            if "AI is off" in err or "budget" in err:
                break
            msgs = self.planner_prompt(question) + f"\n\nYour previous reply {json.dumps(plan)[:1500]} failed: {err}\nFix it."
        if res["status"] == "ok":
            n = len(res["df"])
            checks.append({"check": "Query runs", "passed": True,
                           "detail": f"{n} rows" if n <= 1000 else "more than 1000 rows: showing the first 1000"})
            res["df"] = res["df"].head(1000)
        res["seconds"] = round(time.time() - start, 1)
        if self.log_file:
            pd.DataFrame([{"time": dt.datetime.now().isoformat(timespec="seconds"), "question": question,
                           "status": res["status"], "kind": res["kind"], "verified": res["verified"],
                           "seconds": res["seconds"], "sql": self.eng.inline_params(res["sql"], res["params"])}]).to_csv(
                self.log_file, mode="a", header=not self.log_file.exists(), index=False)
        return res


# ---------- charts: picked from the shape of the result ----------
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def tidy(df):
    df = df.copy()
    for c in df.columns:
        first = df[c].dropna().iloc[0] if df[c].notna().any() else None
        if str(df[c].dtype).startswith("db") or isinstance(first, (dt.date, dt.datetime)):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        elif isinstance(first, decimal.Decimal):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fmt(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "-"
    if isinstance(v, numbers.Number) and not isinstance(v, bool):
        v = float(v)
        return f"{v:,.0f}" if v.is_integer() else f"{v:,.1f}"
    return str(v)


def make_chart(df):
    """(summary text, plotly figure or None): one row -> a headline; 'period' -> a line; a group -> ranked bars."""
    import plotly.graph_objects as go
    if df is None or df.empty:
        return "**No rows.** The query ran but nothing matched.", None
    df = tidy(df).drop(columns=["rows"], errors="ignore")
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    time_c = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cat = [c for c in df.columns if c not in num and c not in time_c]
    if len(df) == 1:
        parts = [fmt(df.iloc[0][c]) for c in cat + time_c] + [f"{c.replace('_', ' ')}: **{fmt(df.iloc[0][c])}**" for c in num]
        return "### " + " · ".join(parts), None
    if not num:
        return f"**{len(df)} rows**", None
    y = "value" if "value" in num else num[-1]
    fig, note = go.Figure(), ""
    if time_c:
        x = time_c[0]
        if cat:
            label = df[cat].astype(object).where(df[cat].notna(), "(empty)").astype(str).agg(" · ".join, axis=1)
            groups = list(df.groupby(label, sort=True))
        else:
            groups = [(None, df)]
        if len(groups) > len(SERIES):
            return f"**{len(df)} rows** · too many groups for one chart: see the table", None
        for i, (name, g) in enumerate(groups):
            g = g.sort_values(x)
            fig.add_scatter(x=g[x], y=g[y], mode="lines+markers", name=str(name) if name is not None else y,
                            line=dict(color=SERIES[i], width=2), marker=dict(size=8),
                            hovertemplate=f"{name or ''}<br>%{{x|%d %b %Y}}<br>%{{y:,.4~g}}<extra></extra>")
        legend, xt = len(groups) > 1, x
    elif not cat:
        if len(num) < 2:
            return f"**{len(df)} rows**", None
        x = next(c for c in num if c != y)
        d = df[[x, y]].sort_values(x)
        note = f" · the chart shows the first 60 of {len(d)}" if len(d) > 60 else ""
        d = d.head(60)
        fig.add_bar(x=d[x].map(lambda v: f"{v:g}" if isinstance(v, float) else str(v)), y=d[y],
                    marker=dict(color=SERIES[0]), hovertemplate=f"{x} %{{x}}<br>%{{y:,.4~g}}<extra></extra>")
        fig.update_layout(bargap=0.35)
        fig.update_xaxes(type="category")
        legend, xt = False, x
    else:
        if len(cat) > 1:
            return f"**{len(df)} rows** · two groupings: see the table", None
        x = cat[0]
        d = df[[x, y]].copy()
        d[x] = d[x].where(d[x].notna(), "(empty)").astype(str)
        note = f" · the chart shows the top 25 of {len(d)}" if len(d) > 25 else ""
        d = d.sort_values(y, ascending=False).head(25).iloc[::-1]
        fig.add_bar(x=d[y], y=d[x], orientation="h", marker=dict(color=SERIES[0]),
                    hovertemplate=f"%{{y}}<br>%{{x:,.4~g}}<extra></extra>")
        try:
            fig.update_traces(marker_cornerradius=4)
        except Exception:
            pass
        fig.update_layout(bargap=0.35)
        fig.update_yaxes(type="category")
        legend, xt = False, ""
    fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=420, showlegend=legend,
                      font=dict(family="Inter, Arial, sans-serif", size=13, color=INK_2), margin=dict(l=10, r=20, t=10, b=10),
                      hoverlabel=dict(bgcolor="white", font_color=INK), legend=dict(orientation="h", y=1.1, x=0))
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zeroline=False, title=xt)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    return f"**{len(df)} rows**{note}", fig
