"""End to end on the finance demo data: all five stages, with Claude replaced by a script (no key, no cost).
Run:  pytest -q"""
import json
import re
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from tablewise.backends import DuckDBBackend, translate          # noqa: E402
from tablewise.project import Project, Settings                  # noqa: E402
import make_finance_data                                          # noqa: E402

FAKE_KEY = "fake-key-for-tests-NEVER-WRITTEN-123"

# ======================= a scripted Claude =======================
SYN = {"CARD": "Card payment", "card payment": "Card payment", "DD": "Direct Debit", "current account": "Current",
       "savings": "Savings", "sme": "SME", "retail": "Retail", "active": "Active", "PAID": "Paid", "paid": "Paid",
       "overdue": "Overdue", "card": "Card", "bank transfer": "Bank transfer", "groceries": "Groceries",
       "bills": "Bills", "north ": "North", "LONDON": "London"}
METRICS = [
    {"name": "transaction_count", "label": "Transactions", "table": "transactions", "sql": "COUNT(*)",
     "time_column": "posted_at", "unit": "transactions", "higher_is": "neutral", "definition": "Number of transactions."},
    {"name": "card_spend", "label": "Card spending", "table": "transactions",
     "sql": "-SUM(IF(t.txn_type = 'Card payment', t.amount, 0))", "time_column": "posted_at", "unit": "GBP",
     "higher_is": "neutral", "definition": "Money spent by card."},
    {"name": "overdue_rate", "label": "Overdue invoice rate", "table": "invoices", "unit": "%", "higher_is": "worse",
     "sql": "100 * SAFE_DIVIDE(COUNTIF(t.status = 'Overdue'), COUNT(*))", "time_column": "issued_on",
     "definition": "Share of invoices that are overdue."},
    {"name": "avg_invoice", "label": "Average invoice", "table": "invoices", "sql": "AVG(t.amount_due)",
     "time_column": "issued_on", "unit": "GBP", "higher_is": "neutral", "definition": "Mean invoice amount."},
    {"name": "bad_value", "label": "Broken on purpose", "table": "invoices", "sql": "COUNTIF(t.status = 'overdue')",
     "time_column": "issued_on", "unit": "invoices", "definition": "uses a spelling that was merged away"},
    {"name": "bad_column", "label": "Broken on purpose", "table": "customers", "sql": "COUNT(t.surname)",
     "time_column": None, "unit": "rows", "definition": "a column that doesn't exist"},
]
BRANCH = "branches.branch_name via account > branch_id"
MONITORS = [
    {"name": "Card payment burst at a branch", "metric": "transaction_count", "group_by": BRANCH, "grain": "week",
     "method": "unusual", "threshold": 3, "min_rows": 5, "why": "Could be fraud or a system fault."},
    {"name": "Overdue invoices rising", "metric": "overdue_rate", "group_by": None, "grain": "week",
     "method": "above", "threshold": 25, "min_rows": 20, "why": "Cash flow risk."},
    {"name": "broken", "metric": "nope", "grain": "week", "method": "unusual", "threshold": 3},
]


def plan_for(prompt):
    q = re.search(r"Question: (.*)\n", prompt).group(1)
    retry = "failed:" in prompt
    if q.startswith("Transactions per week by branch"):
        return {"kind": "metric", "metric": "transaction_count", "group_by": [BRANCH if retry else "branches.branch_name"],
                "time_grain": "week", "start": "2026-04-01", "end": None, "filters": [], "explanation": "Weekly count per branch."}
    if q.startswith("Overdue rate by month"):
        return {"kind": "metric", "metric": "overdue_rate", "group_by": [], "time_grain": "month", "start": None,
                "end": None, "filters": [], "explanation": "Monthly overdue share."}
    if q.startswith("Card spending by segment"):
        return {"kind": "metric", "metric": "card_spend", "group_by": ["customers.segment"], "time_grain": "none",
                "start": None, "end": None, "filters": [], "explanation": "Card spending per customer segment."}
    if q.startswith("Average payment by method"):
        return {"kind": "sql", "sql": "SELECT p.method, ROUND(AVG(p.amount), 2) AS avg_payment FROM payments AS p "
                                      "GROUP BY p.method ORDER BY avg_payment DESC", "explanation": "Average payment per method."}
    if q.startswith("Show customer emails"):
        return {"kind": "sql", "sql": "SELECT email FROM customers LIMIT 5", "explanation": "emails"}
    if q.startswith("Delete"):
        return {"kind": "sql", "sql": "DELETE FROM payments", "explanation": "no"}
    if q.startswith("Who is our best customer"):
        return {"kind": "clarify", "question": "Best by money spent, by number of accounts, or by paying on time?"}
    raise KeyError(q)


def answer(system, prompt):
    if "Proposed relationships" in prompt:
        items = json.loads(prompt.split("Proposed relationships:\n", 1)[1].split("\n\nFor each", 1)[0])
        out = []
        for it in items:
            col = it["column"].split(".")[1]
            toks = [x for x in re.split(r"[^a-z]+", col.lower()) if x and x not in ("id", "ref", "code", "no")]
            parent = it["points_to"].split(".")[0].lower()
            ok = any(x[:4] == parent[:4] or x in parent for x in toks if len(x) >= 3)
            out.append({"id": it["id"], "verdict": "yes" if ok else "no", "reason": "names", "link_name": "belongs to"})
        return out
    if system.startswith("You document a database table"):
        t = re.search(r"Table '(\w+)'", prompt).group(1)
        cols = json.loads(prompt.split("Columns:\n", 1)[1].split("\n\nReturn", 1)[0])
        out = {}
        for c in cols:
            info = {"meaning": f"the {c['column'].replace('_', ' ')}", "personal": False}
            if c.get("values"):
                info["merge"] = {v: SYN[v] for v in c["values"] if v in SYN}
                info["missing"] = [v for v in c["values"] if v.lower() == "n/a"]
            if c.get("negative_values"):
                info["negative_rule"], info["negative_reason"] = "keep", "money out of the account is negative"
            out[c["column"]] = info
        return {"entity": t[:-1].title(), "description": f"One {t[:-1]}.", "columns": out}
    if system.startswith("You design business metrics"):
        return METRICS
    if system.startswith("You turn a question"):
        return plan_for(prompt)
    if system.startswith("You choose what to monitor"):
        return MONITORS
    if system.startswith("You explain data alerts"):
        return {a["action_id"]: f"{a['metric']} went {a['direction']} in {a['segment']}." for a in json.loads(prompt.split("\n\nReturn")[0])}
    raise KeyError(system[:60])


class FakeMessages:
    calls = 0

    def create(self, model, max_tokens, system, messages, betas=None, fallbacks=None):
        assert betas == ["server-side-fallback-2026-07-01"] and fallbacks == "default"
        FakeMessages.calls += 1
        text = "```json\n" + json.dumps(answer(system, messages[0]["content"])) + "\n```"
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                                     usage=types.SimpleNamespace(input_tokens=3000, output_tokens=500))


def scripted(project):
    project.ai.set_key(FAKE_KEY)                       # exercise the real key handling...
    project.ai.client = types.SimpleNamespace(beta=types.SimpleNamespace(messages=FakeMessages()))  # ...then script it
    return project


# ======================= the test =======================
@pytest.fixture(scope="module")
def project(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tw")
    make_finance_data.make(tmp / "data", seed=7)
    backend = DuckDBBackend(tmp / "data", tmp / "ws" / "local.duckdb")
    return scripted(Project(backend, tmp / "ws", Settings(), log=lambda *a: None))


def test_all_five_stages(project):
    p = project
    # ---- Stage 1: keys and links ----
    p.analyse()
    keys = p.key_rows().set_index("table")["primary key"].to_dict()
    assert keys == {"accounts": "account_number", "branches": "branch_id", "customers": "customer_id",
                    "invoices": "invoice_no", "payments": "payment_id", "transactions": "txn_id"}
    links = p.link_rows()
    real = {("accounts.branch_id", "zeros" if False else "exact"), ("customers.home_branch", "exact"),
            ("invoices.customer_ref", "exact"), ("accounts.customer", "zeros"), ("payments.invoice", "format"),
            ("transactions.account", "format")}
    found = links[links.status != "unlikely"]
    assert {(r["from"], r["compare"]) for _, r in found.iterrows()} == real
    assert set(links[links.status == "unlikely"]["from"]) >= {"payments.installment_no"}, "coincidence rejected"
    links.loc[links.status != "unlikely", "approve"] = True
    msgs, rerun = p.save_decisions(keys, links)
    assert not rerun and "6 relationships" in msgs[0]

    # ---- Stage 2: ontology and clean views ----
    p.understand()
    objs, cols, merges, negs = p.ontology_rows()
    c = cols.set_index(["table", "column"])
    assert c.loc[("customers", "full_name"), "personal"] == "drop"
    assert c.loc[("customers", "date_of_birth"), "personal"] == "age"
    assert c.loc[("transactions", "posted_at"), "type"] == "date"
    assert "converted to 'gbp' from p" in c.loc[("transactions", "amount"), "notes"]
    assert negs.set_index(["table", "column"]).loc[("transactions", "amount"), "rule"] == "keep"
    p.save_ontology(objs, cols, merges, negs)
    checks = p.build_views()
    assert (checks.result == "PASS").all(), checks[checks.result != "PASS"]
    con = p.backend.con
    tx = con.execute('SELECT * FROM "model"."transactions"').df()
    assert set(tx.txn_type) == {"Card payment", "Direct Debit", "Transfer", "Faster Payment", "Salary"}
    raw = con.execute('SELECT txn_id, amount FROM "raw"."transactions"').df().set_index("txn_id")["amount"]
    clean = tx.set_index("txn_id")["amount"]
    pence = raw[raw.str.endswith("p")].index[:20]
    assert all(abs(clean[i] - int(raw[i][:-1]) / 100) < 1e-6 for i in pence), "pence -> pounds"
    cu = con.execute('SELECT * FROM "model"."customers"').df()
    assert len(cu) == 800 and cu.customer_id.is_unique, "the repeated customer is kept once"
    assert not {"full_name", "email", "phone", "postcode", "date_of_birth"} & set(cu.columns)
    assert "date_of_birth_age_years" in cu.columns

    # ---- Stage 3: metrics ----
    rows = p.propose_metrics()
    probs = rows.set_index("name")["problems"]
    assert probs["bad_value"] and probs["bad_column"] and not probs["overdue_rate"]
    rows["approve"] = rows["problems"] == ""
    p.save_metrics(rows)
    assert {"transaction_count", "card_spend", "overdue_rate", "avg_invoice"} <= set(p.metrics)
    assert "bad_value" not in p.metrics

    # ---- Stage 4: questions ----
    r = p.ask("Transactions per week by branch")
    assert r["status"] == "ok" and r["verified"], r["checks"]
    assert any(c["check"] == "Attempt 1" for c in r["checks"]), "an ambiguous branch is fixed on the 2nd try"
    assert "via account > branch_id" in r["title"]
    r = p.ask("Overdue rate by month")
    assert r["status"] == "ok" and len(r["df"]) == 7 and r["df"]["value"].iloc[-1] > r["df"]["value"].iloc[0]
    r = p.ask("Card spending by segment")
    assert r["status"] == "ok" and set(r["df"]["segment"]) == {"Retail", "SME", "Small Business", "Premier"}
    r = p.ask("Average payment by method")
    assert r["status"] == "ok" and not r["verified"]
    assert p.ask("Show customer emails")["status"] == "failed", "personal columns are not in the views"
    assert p.ask("Delete all payments")["status"] == "failed", "read-only"
    assert p.ask("Who is our best customer?")["status"] == "clarify"

    # ---- Stage 5: alerts ----
    mrows = p.propose_monitors()
    assert mrows.set_index("name").loc["broken", "problems"]
    p.save_monitors(mrows)
    res = p.run_checks(email=False)
    burst = [a for a in res["new"] if a["check_name"] == "Card payment burst at a branch"]
    assert [a["segment"] for a in burst] == ["branch_name = Birmingham New St"], burst
    assert any(a["check_name"] == "Overdue invoices rising" for a in res["new"])
    assert not p.run_checks(email=False)["new"], "the same alert is not raised twice"
    assert "assigned" in p.update_task(burst[0]["action_id"], "assigned", "Zahra")
    assert p.tasks().set_index("action_id").loc[burst[0]["action_id"], "owner"] == "Zahra"


def test_the_key_is_never_written(project):
    for f in project.dir.rglob("*"):
        if f.is_file():
            assert FAKE_KEY.encode() not in f.read_bytes(), f"the API key was written to {f}"


def test_saved_decisions_come_back(project):
    again = Project(project.backend, project.dir, log=lambda *a: None, api_key="")
    assert again.stage == 4 and set(again.metrics) == set(project.metrics)
    assert again.ask("Overdue rate by month")["status"] == "ok", "answered from the cache, with no key"


def test_translate():
    assert translate("SELECT COUNTIF(x = 'a') FROM `local.model.t`") == 'SELECT count_if(x = \'a\') FROM "model"."t"'
    sql = translate("SELECT DATE_TRUNC(d, WEEK(MONDAY)), SAFE_DIVIDE(a, b) WHERE c IN UNNEST(@p0)",
                    [("p0", "ARRAY<STRING>", ["it's", "b"])])
    assert "date_trunc('week', d)" in sql and "IN ('it''s', 'b')" in sql and "NULLIF(b, 0)" in sql
    assert translate("SELECT date_trunc(d, month)") == "SELECT CAST(date_trunc('month', d) AS DATE)"
