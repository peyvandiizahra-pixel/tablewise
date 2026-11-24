"""The Tablewise app: five steps, each ending in your decision, then questions and alerts.
Start it with `tablewise app` (see the README)."""
from __future__ import annotations

import html
import traceback

import gradio as gr
import pandas as pd

from .ask import make_chart, tidy

CSS = """
.gradio-container {max-width: 1280px !important}
#tw-head h1 {margin-bottom: 0}
.tw-note {color: #52514e; font-size: 0.92em}
"""
INTRO = """# Tablewise
**Point it at messy tables. It finds the keys and links, cleans the data into views, and turns questions into
checked SQL, charts and alerts. Nothing is used until you approve it.**"""


def _b(v):
    """A tick from an edited table: True/False, 'true'/'false', 1/0."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "y", "✓")
    return bool(v) and not pd.isna(v)


def _err(e):
    return f"**Couldn't do that:** {html.escape(str(e))}"


def create_app(project):
    p = project
    log_lines = []
    p.log = lambda *a: log_lines.append(" ".join(str(x) for x in a))

    def take_log():
        out = "\n".join(f"- {html.escape(l)}" for l in log_lines)
        log_lines.clear()
        return out

    def status():
        stage = ["nothing approved yet", "keys and links approved", "clean views built", "metrics approved",
                 "checks approved"][p.stage]
        return f"<span class='tw-note'>Data: <b>{html.escape(p.backend.label)}</b> · {stage} · {html.escape(p.ai.status())}</span>"

    # ------------------------------------------------------------------ handlers
    def use_key(key):
        p.ai.set_key(key)
        return ("Key in use for this session. It is kept in memory only and never saved." if key.strip()
                else "AI switched off."), "", status()

    def analyse():
        try:
            notes = p.analyse()
            msg = (f"Read {len(notes)} tables. Found {len(p.links[p.links.status != 'unlikely']) if len(p.links) else 0} "
                   f"likely relationships and rejected {int((p.links.status == 'unlikely').sum()) if len(p.links) else 0} "
                   "coincidences. Review them in step 1.")
            extra = take_log()
            return (msg + ("\n\n" + extra if extra else ""), notes, p.key_rows(), p.link_rows(), status())
        except Exception as e:
            traceback.print_exc()
            return _err(e), None, None, None, status()

    def save_keys(keys_df, links_df):
        try:
            keys = {r["table"]: r["primary key"] for _, r in keys_df.iterrows()}
            links_df = links_df.copy()
            links_df["approve"] = links_df["approve"].map(_b)
            msgs, rerun = p.save_decisions(keys, links_df)
            if rerun:
                p.analyse()
                msgs.append("Links were searched again against your keys: check the list, then save again.")
            text, svg = p.map()
            return ("\n".join(f"- {m}" for m in msgs), p.key_rows(), p.link_rows(),
                    svg or "", f"```\n{text}\n```", status())
        except Exception as e:
            return _err(e), gr.skip(), gr.skip(), gr.skip(), gr.skip(), status()

    def understand():
        try:
            if p.tables is None:
                p.analyse()
            p.understand()
            o, c, m, n = p.ontology_rows()
            extra = take_log()
            return (("Done. Check each table below, then press **Save and build the clean views**."
                     + ("\n\n" + extra if extra else "")), o, c, m, n)
        except Exception as e:
            return _err(e), None, None, None, None

    def save_onto(o, c, m, n):
        try:
            m = m.copy()
            if len(m):
                m["apply"] = m["apply"].map(_b)
            o = o.copy()
            o["keep one row per key"] = o["keep one row per key"].map(_b)
            p.save_ontology(o, c, m, n)
            checks = p.build_views()
            bad = int((checks.result != "PASS").sum())
            msg = (f"Clean views built. {len(checks) - bad} of {len(checks)} checks passed."
                   + (f" **{bad} need a look** (see below)." if bad else " Next: step 3."))
            extra = take_log()
            return msg + ("\n\n" + extra if extra else ""), checks, status()
        except Exception as e:
            traceback.print_exc()
            return _err(e), None, status()

    def suggest_metrics():
        try:
            rows = p.propose_metrics()
            extra = take_log()
            return ("A metric that fails a check can't be approved; edit its SQL and save to test it again. "
                    "Add your own in an empty row." + ("\n\n" + extra if extra else "")), rows
        except Exception as e:
            return _err(e), None

    def save_metrics(rows):
        try:
            rows = rows.copy()
            rows["approve"] = rows["approve"].map(_b)
            msgs = p.save_metrics(rows)
            return "\n".join(f"- {m}" for m in msgs), p.metric_rows(), status(), gr.update(choices=examples())
        except Exception as e:
            return _err(e), gr.skip(), status(), gr.skip()

    def examples():
        return [f"Show the {m['label'].lower()}" + (" by month" if m.get("time_column") else "")
                for m in list(p.metrics.values())[:5]]

    def ask(question):
        question = (question or "").strip()
        hidden = (gr.update(value=None, visible=False), gr.update(value=None, visible=False))
        if not question:
            return ("Type a question first.",) + hidden + ("", "", None, status())
        try:
            res = p.ask(question)
        except Exception as e:
            return (_err(e),) + hidden + ("", "", None, status())
        checks = pd.DataFrame(res["checks"] or [], columns=["check", "passed", "detail"])
        checks["passed"] = checks["passed"].map({True: "PASS", False: "FAIL"})
        badge = ("**Verified:** built from an approved metric." if res["verified"]
                 else "**Unverified:** new SQL that passed the safety checks. Check it before you rely on it.")
        how = f"{badge}\n\n**What it does:** {res['explanation'] or '-'}\n\nTook {res['seconds']} seconds."
        sql = p.backend.shown_sql(p.engine.inline_params(res["sql"], res["params"])) if res["sql"] else ""
        if res["status"] == "clarify":
            return ((f"**I need one detail first:** {res['clarify']}\n\nAdd it to your question and ask again.",)
                    + hidden + (how, "", checks, status()))
        if res["status"] != "ok":
            return (("**I couldn't build a query that passes every check, so there's no answer.** Open "
                     "*How I got this answer* to see why, or ask in another way.",) + hidden + (how, sql, checks, status()))
        summary, fig = make_chart(res["df"])
        table = tidy(res["df"])
        table[table.select_dtypes("float").columns] = table.select_dtypes("float").round(2)
        for c in table.select_dtypes("datetime").columns:          # whole days: show 2026-06-22, not 00:00:00
            if (table[c].dropna().dt.normalize() == table[c].dropna()).all():
                table[c] = table[c].dt.strftime("%Y-%m-%d")
        text = f"#### {res['title']}\n{summary}\n\n{res['explanation']}\n\n{badge}"
        return (text, gr.update(value=fig, visible=fig is not None), gr.update(value=table, visible=True),
                how, sql, checks, status())

    def suggest_checks():
        try:
            rows = p.propose_monitors()
            extra = take_log()
            return ("Methods: unusual (threshold = standard deviations), change (percent), above / below (a limit)."
                    + ("\n\n" + extra if extra else "")), rows
        except Exception as e:
            return _err(e), None

    def save_checks(rows):
        try:
            rows = rows.copy()
            rows["approve"] = rows["approve"].map(_b)
            return "\n".join(f"- {m}" for m in p.save_monitors(rows)), status()
        except Exception as e:
            return _err(e), status()

    def tasks_view():
        try:
            t = p.tasks()
        except Exception:
            t = pd.DataFrame()
        ids = list(t["action_id"]) if len(t) else []
        shown = t[["action_id", "status", "owner", "metric", "segment", "period", "value", "expected", "message"]] if len(t) else t
        return shown, gr.update(choices=ids, value=ids[0] if ids else None)

    def run_checks():
        try:
            r = p.run_checks()
            failed = f" {len(r['failed'])} failed: " + "; ".join(r["failed"]) if r["failed"] else ""
            msg = f"Ran {r['checked']} checks: {r['fired']} fired, {len(r['new'])} new tasks, {r['email']}.{failed}"
        except Exception as e:
            msg = _err(e)
        return (msg,) + tasks_view() + (status(),)

    def update_task(action_id, st, owner):
        if not action_id:
            return ("Choose a task first.",) + tasks_view()
        try:
            msg = p.update_task(action_id, st, (owner or "").strip())
        except Exception as e:
            msg = _err(e)
        return (msg,) + tasks_view()

    # ------------------------------------------------------------------ layout
    with gr.Blocks(title="Tablewise") as app:
        with gr.Row(elem_id="tw-head"):
            gr.Markdown(INTRO)
        head = gr.HTML(status())

        with gr.Tab("Start"):
            gr.Markdown("**Your Claude API key** (optional, but the AI steps need it). It stays in this app's memory "
                        "only: it is never written to a file, and it's gone when you close the app.")
            with gr.Row():
                key_box = gr.Textbox(label="Claude API key", type="password", placeholder="sk-ant-...", scale=4)
                key_btn = gr.Button("Use this key", scale=1)
            key_msg = gr.Markdown()
            gr.Markdown(f"**Your data:** {html.escape(p.backend.label)}. Reading it is free and changes nothing.")
            go_btn = gr.Button("Analyse my data", variant="primary")
            go_msg = gr.Markdown()
            notes_df = gr.Dataframe(label="Tables read", interactive=False, visible=True)

        with gr.Tab("1 · Keys and links"):
            gr.Markdown("**Primary keys:** type the column that identifies each row (two columns: `a + b`; none: leave "
                        "empty). **Relationships:** tick to approve; *compare* says how values are matched: `exact`, "
                        "`format` (ignoring case, spaces and punctuation) or `zeros` (also leading zeros). Nothing is "
                        "cleaned silently: you see all three percentages.")
            keys_df = gr.Dataframe(label="Primary keys", interactive=True, wrap=True, static_columns=[0, 2, 3, 4])
            links_df = gr.Dataframe(label="Relationships (strongest first; 'unlikely' ones were rejected as coincidences)",
                                    interactive=True, wrap=True, static_columns=[1, 2, 4, 5, 6, 7, 8, 9, 10],
                                    datatype=["bool", "str", "str", "str", "number", "number", "number", "str", "str",
                                              "number", "str"],
                                    column_widths=["72px", "210px", "220px", "84px", "78px", "84px", "78px", "110px",
                                                   "112px", "80px", "420px"])
            save_keys_btn = gr.Button("Save keys and links", variant="primary")
            keys_msg = gr.Markdown()
            map_svg = gr.HTML()
            map_txt = gr.Markdown()

        with gr.Tab("2 · Meaning and clean views"):
            gr.Markdown("The code reads every column (types, date formats, units, personal data) and the AI adds "
                        "meanings and merges of values that mean the same thing. **personal:** `keep`, `drop` (removed "
                        "from the views), `hash` (replaced by a code that still joins) or `age` (birth dates).")
            und_btn = gr.Button("Understand the columns", variant="primary")
            und_msg = gr.Markdown()
            objs_df = gr.Dataframe(label="Tables", interactive=True, wrap=True, static_columns=[0, 4],
                                   datatype=["str", "str", "str", "bool", "number"])
            cols_df = gr.Dataframe(label="Columns (edit type, personal or meaning)", interactive=True, wrap=True,
                                   static_columns=[0, 1, 5], max_height=420)
            merges_df = gr.Dataframe(label="Values to merge (untick to refuse; edit 'becomes')", interactive=True,
                                     wrap=True, static_columns=[1, 2, 3, 5],
                                     datatype=["bool", "str", "str", "str", "str", "str"])
            negs_df = gr.Dataframe(label="Negative numbers (rule: keep, empty or abs)", interactive=True, wrap=True,
                                   static_columns=[0, 1, 2, 3, 5])
            build_btn = gr.Button("Save and build the clean views", variant="primary")
            build_msg = gr.Markdown()
            checks_df = gr.Dataframe(label="Checks on the clean views", interactive=False, wrap=True)

        with gr.Tab("3 · Metrics"):
            gr.Markdown("A metric is one trusted number, defined once (e.g. *overdue invoice rate*). Each one is tested on "
                        "your data before it can be approved.")
            sug_btn = gr.Button("Suggest metrics", variant="primary")
            sug_msg = gr.Markdown()
            metrics_df = gr.Dataframe(interactive=True, wrap=True, static_columns=[6, 9],
                                      datatype=["bool", "str", "str", "str", "str", "str", "number", "str", "str", "str"])
            save_m_btn = gr.Button("Save metrics", variant="primary")
            save_m_msg = gr.Markdown()

        with gr.Tab("4 · Ask"):
            with gr.Row():
                q = gr.Textbox(label="Your question", scale=5, placeholder="e.g. Card spending by customer segment")
                ask_btn = gr.Button("Ask", variant="primary", scale=1)
            ex = gr.Dropdown(label="Examples", choices=examples(), interactive=True)
            out_text = gr.Markdown()
            out_plot = gr.Plot(show_label=False, visible=False)
            out_table = gr.Dataframe(label="Result", interactive=False, wrap=True, visible=False)
            with gr.Accordion("How I got this answer", open=False):
                out_how = gr.Markdown()
                out_sql = gr.Code(label="SQL", language="sql", interactive=False)
                out_checks = gr.Dataframe(label="Checks", interactive=False, wrap=True)

        with gr.Tab("5 · Alerts"):
            gr.Markdown("Each check re-runs approved SQL (no AI cost) and compares the **last complete period** with the "
                        "ones before it. A new alert becomes a task, and an email if you set it up.")
            sug_c_btn = gr.Button("Suggest checks")
            sug_c_msg = gr.Markdown()
            mon_df = gr.Dataframe(interactive=True, wrap=True, static_columns=[9],
                                  column_widths=["72px", "220px", "150px", "300px", "80px", "100px", "90px", "80px", "300px", "200px"],
                                  datatype=["bool", "str", "str", "str", "str", "str", "number", "number", "str", "str"])
            save_c_btn = gr.Button("Save checks")
            save_c_msg = gr.Markdown()
            run_btn = gr.Button("Run checks now", variant="primary")
            run_msg = gr.Markdown()
            tasks_df = gr.Dataframe(label="Tasks (open first)", interactive=False, wrap=True,
                                    column_widths=["120px", "90px", "90px", "170px", "220px", "170px", "80px", "180px", "420px"])
            with gr.Row():
                task_id = gr.Dropdown(label="Task", choices=[])
                task_status = gr.Dropdown(["open", "assigned", "closed"], value="assigned", label="Status")
                task_owner = gr.Textbox(label="Owner (optional)")
                task_btn = gr.Button("Update task")
            task_msg = gr.Markdown()

        # ------------------------------------------------------------------ wiring
        key_btn.click(use_key, key_box, [key_msg, key_box, head])
        go_btn.click(analyse, None, [go_msg, notes_df, keys_df, links_df, head])
        save_keys_btn.click(save_keys, [keys_df, links_df], [keys_msg, keys_df, links_df, map_svg, map_txt, head])
        und_btn.click(understand, None, [und_msg, objs_df, cols_df, merges_df, negs_df])
        build_btn.click(save_onto, [objs_df, cols_df, merges_df, negs_df], [build_msg, checks_df, head])
        sug_btn.click(suggest_metrics, None, [sug_msg, metrics_df])
        save_m_btn.click(save_metrics, metrics_df, [save_m_msg, metrics_df, head, ex])
        ex.change(lambda v: v or gr.skip(), ex, q)
        ask_outs = [out_text, out_plot, out_table, out_how, out_sql, out_checks, head]
        ask_btn.click(ask, q, ask_outs)
        q.submit(ask, q, ask_outs)
        sug_c_btn.click(suggest_checks, None, [sug_c_msg, mon_df])
        save_c_btn.click(save_checks, mon_df, [save_c_msg, head])
        run_btn.click(run_checks, None, [run_msg, tasks_df, task_id, head])
        task_btn.click(update_task, [task_id, task_status, task_owner], [task_msg, tasks_df, task_id])
        app.load(tasks_view, None, [tasks_df, task_id])
    return app


def launch(project, share=False, port=None, open_browser=True):
    create_app(project).launch(share=share, server_port=port, inbrowser=open_browser, css=CSS,
                               theme=gr.themes.Soft(primary_hue="teal"))
