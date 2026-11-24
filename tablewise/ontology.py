"""Stage 2: understand every column (type, date formats, units, personal data, meaning), then turn your reviewed
choices into the ontology that the clean views are built from."""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime as _dt

import pandas as pd

from .discover import exact_text, name_tokens, stem_tokens

MISSING_WORDS = ["", "null", "none", "n/a", "na", "nan", "-", "--", "unknown", "?", "not known", "not recorded"]
BOOL_TRUE, BOOL_FALSE = ["true", "yes", "y", "1"], ["false", "no", "n", "0"]
NUMBER_RE = r"^(?:[£$€]|(?i:GBP|USD|EUR))?\s*-?(0|[1-9][\d,]*)(\.\d+)?\s*[A-Za-z%]{0,8}$"   # also 'GBP 12.00'   # '£1,200.50', '15 mins', '-10' (not '0738...')
EMAIL_RE = r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}"
PHONE_RE = r"\+?[\d\s().-]{10,24}"
NAME_VALUE_RE = r"((Dr|Mr|Mrs|Ms|Miss|Prof)\.? )?[A-Z][a-z'-]+( [A-Z][a-z'-]+){1,3}"   # 'Dr Sarah Jones', 'Tom Hall'
# Column names that identify a person, matched on whole name parts: 'customerEmail' and 'first_name' count,
# 'product_name' and 'branch_name' don't.
PERSONAL_WORDS = {"forename", "surname", "firstname", "lastname", "fullname", "email", "phone", "mobile", "telephone",
                  "tel", "address", "street", "postcode", "zip", "zipcode", "dob", "birthday", "birthdate", "ssn",
                  "passport", "nino", "iban"}
PERSONAL_PHRASES = [["first", "name"], ["last", "name"], ["full", "name"], ["given", "name"], ["e", "mail"],
                    ["post", "code"], ["zip", "code"], ["birth", "date"], ["date", "of", "birth"], ["birth", "year"],
                    ["year", "of", "birth"], ["national", "id"], ["national", "insurance"], ["social", "security"],
                    ["nhs", "number"], ["nhs", "no"], ["nhs", "num"], ["nhs", "id"], ["sort", "code"],
                    ["card", "number"], ["card", "no"], ["tax", "id"]]
# 'name' only identifies a person after one of these words ('customer_name'), or alone in a table of people
PERSON_WORDS = {"patient", "person", "people", "user", "customer", "client", "member", "staff", "employee", "clinician",
                "doctor", "gp", "nurse", "carer", "practitioner", "contact", "kin", "guardian", "parent", "mother",
                "father", "owner", "buyer", "seller", "sender", "recipient", "referrer", "author", "caller", "payee",
                "payer", "holder", "account_holder", "applicant", "borrower", "guarantor", "beneficiary"}
DATE_FORMATS = ["%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d",
                "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y", "%d/%m/%y", "%m/%d/%y",
                "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%b %d, %Y", "%Y%m%d"]
# ISO timestamps with a zone or fractions of a second ('2024-01-15T10:30:00.123Z'): read in SQL as the time written,
# without the zone. In date_formats it is called "ISO".
ISO_RE = r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?"
# Units that convert to one another: each unit's size in the group's first unit. Currencies only convert within one
# currency (pounds and pence), never between currencies.
UNIT_GROUPS = [
    {"min": 1, "mins": 1, "minute": 1, "minutes": 1, "s": 1 / 60, "sec": 1 / 60, "secs": 1 / 60, "second": 1 / 60,
     "seconds": 1 / 60, "h": 60, "hr": 60, "hrs": 60, "hour": 60, "hours": 60, "d": 1440, "day": 1440, "days": 1440,
     "wk": 10080, "wks": 10080, "week": 10080, "weeks": 10080},
    {"month": 1, "months": 1, "mo": 1, "mos": 1, "year": 12, "years": 12, "yr": 12, "yrs": 12, "y": 12,
     "week": 7 / 30.4375, "weeks": 7 / 30.4375, "wk": 7 / 30.4375, "day": 1 / 30.4375, "days": 1 / 30.4375},
    {"kg": 1, "kgs": 1, "g": .001, "gram": .001, "grams": .001, "mg": 1e-6, "mcg": 1e-9, "ug": 1e-9,
     "lb": .45359237, "lbs": .45359237, "st": 6.35029318, "stone": 6.35029318},
    {"cm": 1, "mm": .1, "m": 100, "km": 100000, "in": 2.54, "inch": 2.54, "inches": 2.54, "ft": 30.48},
    {"ml": 1, "l": 1000, "cl": 10, "dl": 100},
    {"gbp": 1, "p": .01, "pence": .01},
    {"usd": 1, "cents": .01},
    {"eur": 1},
]
ID_VALUE_RE = r"[A-Za-z]{1,6}[-_/]?[A-Za-z]{0,4}[-_/]?\d{2,}"
DAY_MONTH_PAIRS = [("%d/%m/%Y", "%m/%d/%Y"), ("%d-%m-%Y", "%m-%d-%Y"), ("%d/%m/%y", "%m/%d/%y"),
                   ("%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M")]
DOSE = r"(\d+(?:\.\d+)?)\s*(?:mg|mcg|µg|ug|g|ml|l|%|iu|units?)(?![A-Za-z])"
TYPES = ["identifier", "number", "date", "boolean", "category", "text"]
PERSONAL_CHOICES = ["keep", "drop", "hash", "age"]
NEGATIVE_RULES = ["keep", "empty", "abs"]


def personal_by_name(column, table):
    """Does the column NAME say it identifies a person?"""
    toks = name_tokens(column)
    if set(toks) & PERSONAL_WORDS or any(toks[i:i + len(p)] == p for p in PERSONAL_PHRASES for i in range(len(toks))):
        return True
    if "name" in toks:
        i = toks.index("name")
        if i and toks[i - 1] in PERSON_WORDS:            # customer_name, holder_name
            return True
        stem = stem_tokens(table)
        if toks == ["name"] and stem and stem[-1] in PERSON_WORDS:     # customers.name (but not branches.name)
            return True
    return False


def looks_like_phone(x):
    """'07700 900123', '+44 7700 900123', '627-684-9103': 10 to 15 digits, and a leading + or 0 or separators.
    Not a plain number (an ID, a timestamp), not a zero-padded ID like 0001234567, not a date like 2024-01-15."""
    x = str(x).strip()
    if not re.fullmatch(PHONE_RE, x) or re.fullmatch(r"-?\d+\.\d+", x):
        return False
    if not 10 <= len(re.sub(r"\D", "", x)) <= 15:
        return False
    if x[0] == "+" or re.search(r"[\s().-]", x):
        return True
    return x[0] == "0" and x[1] != "0"


def is_nhs_number(x):
    """A valid NHS number: 10 digits whose last digit is the modulus-11 check digit."""
    d = re.sub(r"[\s-]", "", str(x))
    if not re.fullmatch(r"\d{10}", d):
        return False
    check = 11 - sum(int(d[i]) * (10 - i) for i in range(9)) % 11
    return check != 10 and check % 11 == int(d[9])


def personal_reason(t, c, s, v, bq_type, typ):
    """Why a column looks personal, or '' if it doesn't: from its name, the form of its values, and free text."""
    if typ == "boolean":
        return ""
    if personal_by_name(c, t):
        return "its name says it identifies a person"
    if typ == "date" or not len(v):
        return ""
    vals = v.astype(str) if bq_type == "STRING" else exact_text(s)
    vals = vals.sample(min(len(vals), 500), random_state=0)
    if vals.str.fullmatch(EMAIL_RE).mean() > .5:
        return "the values look like email addresses"
    if vals.map(is_nhs_number).mean() > .9:
        return "the values are valid NHS numbers"
    if typ != "number" and vals.map(looks_like_phone).mean() > .5:
        return "the values look like phone numbers"
    toks = name_tokens(c)
    if typ in ("text", "category") and toks and stem_tokens(c)[-1] in PERSON_WORDS \
            and vals.str.fullmatch(NAME_VALUE_RE).mean() > .5:
        return "the values look like people's names"
    if typ == "text" and bq_type == "STRING" and (vals.str.split().str.len() >= 4).mean() >= .5:
        return "free text: it may contain names or ID numbers"
    return ""


def clean_strings(s):
    """Values as text, trimmed, with words like 'n/a' or 'unknown' treated as empty."""
    v = s.dropna().astype(str).str.strip()
    return v[~v.str.lower().isin(MISSING_WORDS)]


def compact_dates(v):
    """'20240115': eight digits that read as a date (yyyymmdd), so not a number."""
    s = v.sample(min(len(v), 500), random_state=0)
    eight = s.str.fullmatch(r"(19|20)\d{6}")
    return eight.mean() >= .9 and pd.to_datetime(s[eight], format="%Y%m%d", errors="coerce").notna().mean() >= .9


def unit_factors(units):
    """How to bring every unit in a number column to its most common unit: {} for one unit or none,
    {unit: multiplier} when they convert ('hrs': 60 in a 'mins' column), None when they can't be converted."""
    found = units.value_counts()
    if len(found) <= 1:
        return {}
    main = found.index[0]
    for group in UNIT_GROUPS:
        if all(u in group for u in found.index):
            return {u: group[u] / group[main] for u in found.index}
    return None


def conversion_note(sp):
    f = sp.get("unit_factors") or {}
    others = [u for u in f if u != sp["unit"]]
    return (f"converted to '{sp['unit']}' from " + ", ".join(f"{u} (x{f[u]:g})" for u in others)) if others else ""


def date_formats(v):
    """Which date formats appear in a column, and in which order to try them. Python and BigQuery use the
    same format codes, so the formats found here are used as they are in the SQL views."""
    v = v.sample(min(len(v), 2000), random_state=0)
    parses = {}
    for f in DATE_FORMATS + ["ISO"]:
        ok = set()
        for x in v.unique():
            if f == "ISO":
                if re.fullmatch(ISO_RE, x):
                    ok.add(x)
                continue
            try:
                _dt.strptime(x, f)
                ok.add(x)
            except ValueError:
                pass
        if ok:
            parses[f] = ok
    if not parses:
        return [], 0.0, None
    order, notes = {}, []
    for day_first, month_first in DAY_MONTH_PAIRS:
        a, b = parses.get(day_first, set()), parses.get(month_first, set())
        if a or b:
            only_a, only_b = len(a - b), len(b - a)       # e.g. 27/02 can only be day-first
            order[(day_first, month_first)] = "day" if only_a >= only_b else "month"
            if a & b and not (only_a and only_b) and only_a == only_b:
                notes.append(f"'{day_first[:5]}' dates never show a day above 12: day/month order is a guess")
    counts = v.value_counts()
    chosen = sorted(parses, key=lambda f: -counts[counts.index.isin(parses[f])].sum())
    for (d, m), first in order.items():                  # the winning order is tried first
        if d in chosen and m in chosen:
            win, lose = (d, m) if first == "day" else (m, d)
            chosen.remove(lose)
            chosen.insert(chosen.index(win) + 1, lose)
    covered = set().union(*parses.values())
    share = counts[counts.index.isin(covered)].sum() / counts.sum()
    return chosen, share, "; ".join(notes) or None


def understand_column(t, c, s, bq_type, prof_row, role):
    """Proposes how to read one column: a dict the review shows and the views use."""
    v = clean_strings(s) if bq_type == "STRING" else s.dropna()
    spec = {"bq_type": bq_type, "role": role, "type": "text", "personal": "keep", "personal_why": "", "meaning": "",
            "date_formats": [], "date_order_note": None, "has_time": False, "unit": "", "unit_factors": {},
            "unit_problem": "", "mapping": {}, "mapping_reasons": {}, "negatives": None, "negative_rule": "keep",
            "values": [], "form": prof_row["shape"], "filled_pct": prof_row["filled_pct"]}
    if role in ("key", "link"):
        spec["type"] = "identifier"
    elif bq_type in ("DATE", "DATETIME", "TIMESTAMP"):
        spec.update(type="date", has_time=bq_type != "DATE")
    elif bq_type in ("BOOL", "BOOLEAN"):
        spec["type"] = "boolean"
    elif bq_type in ("INT64", "INTEGER", "FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"):
        spec["type"] = "number"
    elif len(v):
        low = v.str.lower()
        numeric = v.str.fullmatch(NUMBER_RE).mean() >= .9 and not compact_dates(v)
        units = v.str.extract(r"([A-Za-z%]+)\s*$")[0].dropna().str.lower() if numeric else None
        factors = unit_factors(units) if numeric else None
        if low.isin(BOOL_TRUE + BOOL_FALSE).mean() >= .95 and low.nunique() <= 6:
            spec["type"] = "boolean"
        elif numeric and factors is not None:
            spec.update(type="number", unit=units.value_counts().index[0] if len(units) else "", unit_factors=factors)
        else:
            if numeric:      # numbers in units that don't convert ('15m', '2h'): one number per value would be wrong
                spec["unit_problem"] = ("numbers in units that can't be converted to one another ("
                                        + ", ".join(units.value_counts().index[:6]) + "): kept as text")
            formats, share, note = date_formats(v)
            if share >= .8:
                spec.update(type="date", date_formats=formats, date_order_note=note,
                            has_time=any("%H" in f or f == "ISO" for f in formats))
            elif prof_row["distinct_format"] <= 30:
                spec["type"] = "category"
    if spec["type"] in ("text", "category") and role == "column" and bq_type == "STRING" and len(v) \
            and prof_row["distinct_format"] > 30 and v.str.fullmatch(ID_VALUE_RE).mean() >= .9:
        spec["type"] = "identifier"           # 'SKU-0012', 'ORD000123': codes, even when nothing links to them
    if spec["type"] == "number":
        nums = pd.to_numeric(v, errors="coerce") if bq_type != "STRING" else pd.to_numeric(
            v.str.replace(r"(?i)^(GBP|USD|EUR)", "", regex=True).str.replace(r"[£$€,\s]", "", regex=True).str.extract(r"^(-?\d+(?:\.\d+)?)")[0], errors="coerce")
        neg = v[nums < 0]
        if len(neg):
            spec["negatives"] = {"count": int(len(neg)), "examples": {str(k): int(n) for k, n in neg.value_counts().head(5).items()}}
    spec["personal_why"] = personal_reason(t, c, s, v, bq_type, spec["type"])
    if spec["personal_why"]:
        spec["personal"] = "hash" if role in ("key", "link") else "age" if spec["type"] == "date" else "drop"
    if spec["type"] == "category" and spec["personal"] == "keep":
        spec["values"] = [str(x) for x in v.value_counts().head(40).index]
    return spec


def changes_dose(old, new):
    a, b = sorted(re.findall(DOSE, str(old), re.I)), sorted(re.findall(DOSE, str(new), re.I))
    return bool(a or b) and a != b


ONTO_SYSTEM = ("You document a database table so that analysts and an AI can query it safely. You see column names, "
               "types and value forms (A = letter, 9 = digit). Only for category columns do you see the real values. "
               "Answer only with JSON.")


def ai_describe_table(t, specs, pks, links, ai, log=print):
    cols = []
    for c, sp in specs.items():
        item = {"column": c, "type": sp["type"], "form": sp["form"], "filled_pct": sp["filled_pct"]}
        if sp["values"]:
            item["values"] = sp["values"]
        if sp["negatives"]:
            item["negative_values"] = sp["negatives"]["count"]
        cols.append(item)
    lk = [f"{l['from']} -> {l['to']}" for l in links if t in (l["from_table"], l["to_table"])]
    prompt = (f"Table '{t}', primary key {pks.get(t)}. Links: {lk}\nColumns:\n{json.dumps(cols)}\n\n"
              "Return {\"entity\": \"singular name of what one row is, e.g. Invoice\", "
              "\"description\": \"one sentence\", \"columns\": {\"<column>\": {"
              "\"meaning\": \"short plain-English meaning\", "
              "\"personal\": true if it identifies or describes a person (name, contact, birth date, a national, "
              "health or bank ID number), else false, "
              "\"merge\": {\"raw value\": \"standard value\"} ONLY for columns with values: map values that mean "
              "EXACTLY the same thing to one clear standard value; never merge different meanings; keep doses and "
              "amounts ('500mg', '£5') exactly; leave out anything you are unsure of, "
              "\"missing\": [values that mean unknown or not recorded], "
              "\"negative_rule\": only for columns with negative_values: \"keep\" if negatives make sense, "
              "\"empty\" if they are errors, \"abs\" if the sign was flipped by mistake, "
              "\"negative_reason\": \"one sentence\"}}}")
    try:
        return ai.ask_json(ONTO_SYSTEM, prompt, expect=dict)
    except Exception as e:
        log(f"{t}: AI description skipped ({e})")
        return {}


def final_mapping(values_counts, merges, missing):
    """Every observed spelling -> its final value: AI merges first, then case/spacing variants of the same
    word become the most common spelling. None = empty."""
    out, reasons, mapped = {}, {}, {}
    for v in values_counts.index:
        if v in missing:
            mapped[v] = None
            reasons[v] = "the AI says it means 'unknown'"
        elif v in merges and merges[v] != v and not changes_dose(v, merges[v]):
            mapped[v] = merges[v]
            reasons[v] = "AI: same meaning"
        else:
            mapped[v] = v
    totals = {}
    for v, n in values_counts.items():
        if mapped[v] is not None:
            totals[mapped[v]] = totals.get(mapped[v], 0) + n
    spelling = {}
    for target, n in sorted(totals.items(), key=lambda x: -x[1]):
        spelling.setdefault(re.sub(r"\s+", " ", target.strip().lower()), target)
    for v in values_counts.index:
        final = mapped[v] if mapped[v] is None else spelling[re.sub(r"\s+", " ", mapped[v].strip().lower())]
        if final != v:
            out[v] = final
            reasons.setdefault(v, "same word, different case or spacing")
    return out, {k: reasons[k] for k in out}


def understand_tables(tables, schemas, profile, pks, links, ai, log=print):
    """Stage 2 analysis for every table: (column specs, table info)."""
    link_cols = {(l["from_table"], l["from_column"]) for l in links}
    specs_all, info_all = {}, {}
    for t, df in tables.items():
        p = profile[profile.table == t].set_index("column")
        specs = {}
        for c in df.columns:
            role = "key" if c in pks.get(t, []) else "link" if (t, c) in link_cols else "column"
            specs[c] = understand_column(t, c, df[c], schemas[t].get(c, "STRING"), p.loc[c], role)
        ans = ai_describe_table(t, specs, pks, links, ai, log)
        for c, sp in specs.items():
            a = (ans.get("columns") or {}).get(c, {})
            sp["meaning"] = a.get("meaning", "")
            if a.get("personal") and sp["personal"] == "keep":
                sp["personal"] = "hash" if sp["role"] in ("key", "link") else "age" if sp["type"] == "date" else "drop"
                sp["personal_why"] = "the AI says it identifies or describes a person"
            if sp["negatives"] and a.get("negative_rule") in NEGATIVE_RULES:
                sp["negative_rule"] = a["negative_rule"]
                sp["negatives"]["reason"] = a.get("negative_reason", "")
            if sp["type"] == "category" and sp["personal"] == "keep":
                counts = clean_strings(df[c]).value_counts()
                merges = {k: v for k, v in (a.get("merge") or {}).items() if isinstance(v, str) and k in counts.index}
                missing = {m for m in (a.get("missing") or []) if m in counts.index}
                sp["mapping"], sp["mapping_reasons"] = final_mapping(counts, merges, missing)
        info_all[t] = {"entity": ans.get("entity", t), "description": ans.get("description", ""),
                       "duplicate_keys": int(df[pks[t]].duplicated().sum()) if pks.get(t) else 0}
        specs_all[t] = specs
    return specs_all, info_all


def notes_for(sp):
    return "; ".join(filter(None, [
        f"form {sp['form']}", f"{sp['filled_pct']:g}% filled", sp["role"] if sp["role"] != "column" else "",
        f"personal: {sp['personal_why']}" if sp.get("personal_why") else "",
        f"formats {', '.join(sp['date_formats'])}" if sp["date_formats"] else "",
        sp["date_order_note"] or "", f"unit '{sp['unit']}'" if sp["unit"] else "", conversion_note(sp),
        sp.get("unit_problem") or ""]))


def build_ontology(specs, info, pks, links, tables, choices, salt=None, source=""):
    """Your reviewed choices -> the ontology the views are built from.
    choices = {"tables": {t: {"entity", "description", "dedupe"}},
               "columns": {(t, c): {"type", "personal", "meaning"}},
               "mapping": {(t, c): {old: new or None, ...}}  (only the merges you kept),
               "negative_rule": {(t, c): rule}}"""
    salt = salt or hashlib.sha256(f"{source}{time.time()}".encode()).hexdigest()[:16]
    out = {}
    for t, cols in specs.items():
        tc = choices["tables"].get(t, {})
        columns = {}
        for c, sp in cols.items():
            ch = choices["columns"].get((t, c), {})
            typ = ch.get("type", sp["type"])
            fmts, has_time = sp["date_formats"], sp["has_time"]
            if typ == "date" and sp["bq_type"] == "STRING" and not fmts:
                # you changed it to a date: find its formats now, or the view would get an empty column
                fmts = date_formats(clean_strings(tables[t][c]))[0]
                has_time = any("%H" in f or f == "ISO" for f in fmts)
            personal = ch.get("personal", sp["personal"])
            if personal == "age" and typ != "date":
                personal = "drop"
            columns[c] = {"bq_type": sp["bq_type"], "role": sp["role"], "unit": sp["unit"], "form": sp["form"],
                          "date_formats": fmts, "has_time": has_time, "unit_factors": sp.get("unit_factors") or {},
                          "type": typ, "personal": personal, "meaning": ch.get("meaning", sp["meaning"]),
                          "mapping": choices["mapping"].get((t, c), sp["mapping"]),
                          "negative_rule": choices["negative_rule"].get((t, c), sp["negative_rule"]),
                          "values": sp["values"]}
        out[t] = {"entity": tc.get("entity", info[t]["entity"]), "description": tc.get("description", info[t]["description"]),
                  "key": pks.get(t, []), "dedupe": bool(tc.get("dedupe", info[t]["duplicate_keys"] > 0) and info[t]["duplicate_keys"]),
                  "columns": columns}
    return {"source": source, "salt": salt, "saved": time.strftime("%Y-%m-%d %H:%M"), "tables": out, "links": links}
