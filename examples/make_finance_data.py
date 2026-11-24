"""Make the demo dataset: six messy tables of a small, made-up bank. Everything is fake.

    python examples/make_finance_data.py            # writes examples/finance/*.csv

The mess is the kind real exports have: four date formats in one column, amounts written '£1,250.00',
'12.30 GBP' or '1230p', IDs that lost their leading zeros or changed case, the same label spelled five ways,
a repeated customer, and placeholders like 'n/a'. Two problems are hidden for the alerts to find:
  1. a burst of card payments at one branch in the last complete week (2026-06-22 .. 2026-06-28)
  2. the share of overdue invoices climbs over the last four weeks
"""
import argparse
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

START, END = date(2025, 12, 1), date(2026, 6, 28)          # a Monday .. a Sunday: 30 complete weeks
FIRST = ["Amira", "Ben", "Chloe", "Daniel", "Ella", "Farhan", "Grace", "Hassan", "Isla", "Jack", "Leila", "Mohammed",
         "Nora", "Oliver", "Priya", "Reza", "Sara", "Tom", "Yusuf", "Zahra"]
LAST = ["Ahmadi", "Brown", "Clarke", "Davies", "Evans", "Green", "Hughes", "Khan", "Lewis", "Patel", "Roberts",
        "Smith", "Taylor", "Walker", "Wilson", "Wright"]


def day(r):
    return START + timedelta(days=r.randrange((END - START).days + 1))


def fmt_date(d, r):
    return r.choice([d.strftime("%Y-%m-%d"), d.strftime("%d/%m/%Y"), d.strftime("%d %b %Y"), d.strftime("%Y%m%d")
                     if r.random() < .05 else d.strftime("%d/%m/%Y")])


def fmt_time(t, r):
    return r.choice([t.strftime("%Y-%m-%dT%H:%M:%SZ"), t.strftime("%d/%m/%Y %H:%M"), t.strftime("%Y-%m-%d %H:%M")])


def money(x, r, allow_units=True):
    """-12.3 -> '-12.30'; 1250 -> '£1,250.00' or '1250.00 GBP' or '125000p'."""
    if x < 0:
        return f"{x:.2f}"
    k = r.random()
    if allow_units and k < .12:
        return f"{x:.2f} GBP"
    if allow_units and k < .18:
        return f"{round(x * 100)}p"
    return f"£{x:,.2f}" if k < .6 else f"{x:.2f}"


def make(out, seed=7):
    r = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)

    branches = pd.DataFrame({
        "branch_id": [1, 2, 3, 4, 5],
        "branch_name": ["Camden", "Leeds Central", "Birmingham New St", "Manchester", "Croydon"],
        "region": ["London", "North", "Midlands", "north ", "LONDON"],
        "opened_on": ["2004-05-17", "12/09/2011", "03 Mar 2015", "2019-01-07", "28/02/2021"]})

    n_cust = 800
    cust_ids = [f"CUS{i:05d}" for i in range(1, n_cust + 1)]
    customers = pd.DataFrame({
        "customer_id": cust_ids,
        "full_name": [f"{r.choice(FIRST)} {r.choice(LAST)}" for _ in cust_ids],
        "email": [f"user{i}@example.com" if r.random() > .03 else "n/a" for i in range(n_cust)],
        "phone": [f"07{r.randint(100, 999)} {r.randint(100000, 999999)}" for _ in cust_ids],
        "date_of_birth": [fmt_date(date(1950, 1, 1) + timedelta(days=r.randrange(20000)), r) for _ in cust_ids],
        "postcode": [r.choice(["NW1 7AB", "LS1 4DY", "B2 4QA", "M1 1AE", "CR0 2AB"]) for _ in cust_ids],
        "segment": [r.choice(["Retail", "Retail", "retail", "SME", "sme", "Small Business", "Premier"]) for _ in cust_ids],
        "home_branch": [r.randint(1, 5) for _ in cust_ids],
        "joined_on": [fmt_date(day(r) - timedelta(days=r.randrange(2000)), r) for _ in cust_ids]})
    customers = pd.concat([customers, customers.iloc[[41]]], ignore_index=True)      # one customer exported twice

    n_acc = 1200
    acc_ids = [f"AC{100000 + i}" for i in range(1, n_acc + 1)]
    acc_cust = [r.randrange(n_cust) for _ in acc_ids]
    acc_branch = [int(customers.home_branch[c]) if r.random() < .8 else r.randint(1, 5) for c in acc_cust]
    accounts = pd.DataFrame({
        "account_number": acc_ids,
        # a third of the IDs lost their leading zeros on the way: CUS00042 -> CUS42
        "customer": [cust_ids[c] if r.random() < .67 else f"CUS{c + 1}" for c in acc_cust],
        "branch_id": acc_branch,
        "account_type": [r.choice(["Current", "Current", "current account", "Savings", "savings", "ISA", "Business"])
                         for _ in acc_ids],
        "opened": [fmt_date(day(r) - timedelta(days=r.randrange(3000)), r) for _ in acc_ids],
        "overdraft_limit": [r.choice(["£500.00", "500", "£1,000.00", "0", "-", "250"]) for _ in acc_ids],
        "status": [r.choice(["Active"] * 6 + ["active", "Closed", "Frozen", "N/A"]) for _ in acc_ids]})

    txns = []
    kinds = [("Card payment", -1, 5, 120), ("CARD", -1, 5, 120), ("card payment", -1, 5, 120), ("Direct Debit", -1, 20, 400),
             ("DD", -1, 20, 400), ("Transfer", -1, 10, 900), ("Faster Payment", 1, 10, 900), ("Salary", 1, 1200, 4200)]
    cats = {"Card payment": ["Groceries", "groceries", "Travel", "Eating out", "Shopping", "n/a"],
            "Direct Debit": ["Bills", "bills", "Utilities"], "Transfer": ["Transfer"], "Faster Payment": ["Transfer"],
            "Salary": ["Income"]}

    def txn(acc, when, kind=None, amount=None):
        name, sign, lo, hi = kind or r.choice(kinds)
        base = {"card": "Card payment", "dd": "Direct Debit"}.get(name.lower().split()[0], name)
        amt = amount if amount is not None else sign * round(r.uniform(lo, hi), 2)
        txns.append({"account": acc if r.random() < .9 else acc.lower(), "posted_at": fmt_time(when, r),
                     "amount": money(amt, r), "txn_type": name,
                     "category": r.choice(cats.get(base, ["Other"]))})

    for _ in range(20000):
        t = datetime.combine(day(r), datetime.min.time()) + timedelta(minutes=r.randrange(7 * 60, 23 * 60))
        txn(r.choice(acc_ids), t)
    # hidden problem 1: a burst of card payments on branch 3's accounts in the last complete week
    branch3 = [a for a, b in zip(acc_ids, acc_branch) if b == 3]
    for _ in range(420):
        t = datetime(2026, 6, 22) + timedelta(days=r.randrange(7), minutes=r.randrange(60, 1380))
        txn(r.choice(branch3), t, kind=("Card payment", -1, 150, 900))
    r.shuffle(txns)
    transactions = pd.DataFrame([{"txn_id": i, **x} for i, x in enumerate(txns, 1)])

    invoices, payments = [], []
    for i in range(1, 3001):
        issued = day(r)
        weeks_to_end = (END - issued).days // 7
        overdue_p = .08 if weeks_to_end >= 4 else [.38, .3, .22, .16][weeks_to_end]   # hidden problem 2
        due = issued + timedelta(days=30)
        amt = round(r.uniform(80, 4000), 2)
        status = "Overdue" if r.random() < overdue_p else r.choice(["Paid", "Paid", "paid", "PAID", "Open", "Cancelled"])
        if status == "Overdue":
            status = r.choice(["Overdue", "Overdue", "overdue"])
        inv = f"INV-26-{i:05d}"
        parts = r.choice([1, 1, 1, 2, 3])
        paid_days = sorted(issued + timedelta(days=r.randrange(5, 45)) for _ in range(parts))
        if status.lower() == "paid" and paid_days[-1] > END:
            status = "Open"                                  # not paid yet when the data was exported
        invoices.append({"invoice_no": inv, "customer_ref": r.choice(cust_ids), "issued_on": fmt_date(issued, r),
                         "due_on": fmt_date(due, r), "amount_due": money(amt, r, allow_units=False), "status": status})
        if status.lower() == "paid":
            for k, paid in enumerate(paid_days, 1):
                ref = r.choice([inv, inv, inv, inv.lower(), inv.replace("-", "")])
                payments.append({"invoice": ref, "paid_on": fmt_date(paid, r),
                                 "amount": money(round(amt / parts, 2), r, allow_units=False),
                                 "method": r.choice(["Card", "card", "BACS", "Bank transfer", "bank transfer", "Cheque"]),
                                 "installment_no": k})
    invoices = pd.DataFrame(invoices)
    payments = pd.DataFrame([{"payment_id": f"PAY{i:06d}", **p} for i, p in enumerate(payments, 1)])

    for name, df in {"branches": branches, "customers": customers, "accounts": accounts, "transactions": transactions,
                     "invoices": invoices, "payments": payments}.items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name:13s} {len(df):6,} rows")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(Path(__file__).parent / "finance"))
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    make(Path(a.out), a.seed)
