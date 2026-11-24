"""Stage 5: the AI proposes checks on approved metrics; each run re-uses the approved SQL (no AI cost) and judges
the last complete period against the periods before it. New alerts become tasks, and optionally an email."""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import os
import re
import smtplib
import ssl
from email.message import EmailMessage

import pandas as pd

from .ask import fmt, tidy

MONITOR_SYSTEM = "You choose what to monitor in an organisation's data, to catch problems early. Answer only with JSON."
METHODS = {"unusual": "far outside its normal range", "change": "a big change since the period before",
           "above": "above a limit", "below": "below a limit"}
ACTION_COLS = ["action_id", "created_at", "check_name", "metric", "segment", "period", "value", "expected",
               "direction", "message", "status", "owner", "updated_at"]


def monitor_problems(eng, metrics, mo):
    m = metrics.get(mo.get("metric"))
    if not m:
        return [f"unknown metric '{mo.get('metric')}'"]
    probs = []
    if not m.get("time_column"):
        probs.append("this metric has no date, so it can't be watched over time")
    if mo.get("group_by") and mo["group_by"] not in eng.dimensions_for(m["table"]):
        probs.append(f"can't split it by '{mo['group_by']}'")
    if mo.get("grain") not in ("day", "week", "month"):
        probs.append("grain must be day, week or month")
    if mo.get("method") not in METHODS:
        probs.append(f"method must be one of {list(METHODS)}")
    if not isinstance(mo.get("threshold"), (int, float)):
        probs.append("threshold must be a number")
    return probs


def propose_monitors(eng, ai, metrics, saved=(), log=print):
    """Your saved checks come back exactly as you approved them; the AI's ideas are only added when they watch
    something new, under a name you haven't used."""
    prompt = (eng.catalog_text(metrics) + "\n\nSuggest 3 to 8 checks that would catch real problems early. Return a JSON "
              "list of {\"name\": \"short title\", \"metric\": \"<approved metric name>\", \"group_by\": \"table.column\" "
              "from that metric's options, or null, \"grain\": \"day|week|month\", "
              "\"method\": \"unusual|change|above|below\", \"threshold\": number (unusual: how many standard "
              "deviations from normal, e.g. 3; change: percent, e.g. 30; above/below: a limit in the metric's unit), "
              "\"min_rows\": the fewest records a period needs before it is judged, \"why\": \"one sentence\"}.")
    try:
        found = ai.ask_json(MONITOR_SYSTEM, prompt, expect=list)
    except Exception as e:
        log(f"The AI could not suggest checks: {e}")
        found = []
    found = [{**s, "saved": True} for s in saved] + [mo for mo in found if isinstance(mo, dict)]
    if not any(not monitor_problems(eng, metrics, {"group_by": None, "grain": "week", **mo}) for mo in found):
        # nothing usable: one simple check per dated metric
        found += [{"name": f"{m['label']}: unusual week", "metric": m["name"], "group_by": None, "grain": "week",
                   "method": "unusual", "threshold": 3, "min_rows": 10, "why": "A simple default check."}
                  for m in metrics.values() if m.get("time_column")]
    out, seen = [], set()
    for mo in found:
        if not isinstance(mo, dict):
            continue
        mo = {"group_by": None, "grain": "week", "min_rows": 10, "why": "", **mo}
        key = (mo.get("metric"), mo.get("group_by"), mo.get("method"), mo.get("grain"))
        name = mo.get("name") or str(key)
        if key in seen or name in seen:        # saved checks come first, so they win over a similar AI idea
            continue
        seen.update([key, name])
        mo["name"] = name
        mo["problems"] = monitor_problems(eng, metrics, mo)
        out.append(mo)
    return out


def period_start(d, grain):
    return d if grain == "day" else d - dt.timedelta(days=d.weekday()) if grain == "week" else d.replace(day=1)


def next_period(p, grain):
    return p + dt.timedelta(days=1 if grain == "day" else 7) if grain != "month" else (p + dt.timedelta(days=32)).replace(day=1)


def last_complete(end, grain):
    """The newest period that has fully ended by the day `end`."""
    current = period_start(end, grain)
    return current if next_period(current, grain) - dt.timedelta(days=1) <= end else period_start(current - dt.timedelta(days=1), grain)


def empty_means_zero(m):
    """True for a metric that is a plain COUNT(...), COUNTIF(...) or SUM(...): a period with no rows really is 0, and
    a drop to 0 must be seen. A rate or an average has no value at all in an empty period."""
    if "empty_means_zero" in m:
        return bool(m["empty_means_zero"])
    s = m["sql"].strip()
    head = re.match(r"(COUNT|COUNTIF|SUM)\s*\(", s, re.I)
    if not head:
        return False
    depth = 0
    for i in range(head.end() - 1, len(s)):
        depth += {"(": 1, ")": -1}.get(s[i], 0)
        if depth == 0:
            return i == len(s) - 1
    return False


class Monitor:
    def __init__(self, engine, ai, metrics, lookback=12, stale_data_alerts=False, dataset_label="your data"):
        self.eng, self.ai, self.metrics = engine, ai, metrics
        self.lookback, self.stale, self.label = lookback, stale_data_alerts, dataset_label

    def latest_date(self, m):
        """The newest date in the metric's table, ignoring dates in the future (bookings ahead, typos like 2204)."""
        d_ = self.eng.date_expr("t", m["table"], m["time_column"])
        d = self.eng.run_sql(f"SELECT MAX(IF({d_} <= CURRENT_DATE(), {d_}, NULL)) AS d "
                             f"FROM {self.eng.backend.model_ref(m['table'])} AS t").iloc[0, 0]
        return pd.Timestamp(d).date() if d is not None and not pd.isna(d) else None

    def evaluate(self, mo, as_of=None):
        """Runs one approved check. The last COMPLETE period is compared with the periods before it."""
        m, grain, thr = self.metrics[mo["metric"]], mo["grain"], float(mo["threshold"])
        today = dt.date.today()
        end = as_of or self.latest_date(m)
        if end is None:
            return []
        last = last_complete(min(end, today), grain)
        alerts = []
        newest = last_complete(today, grain)
        if self.stale and as_of is None and last < newest:
            alerts.append({"action_id": hashlib.sha1(f"{mo['name']}|no data|{newest}".encode()).hexdigest()[:12],
                           "check_name": mo["name"], "metric": m["label"], "segment": "all",
                           "period": f"{grain} of {newest}", "value": 0.0,
                           "expected": f"no {m['table']} rows since {end}", "direction": "down (no data)"})
        periods = [last]
        for _ in range(self.lookback):
            periods.insert(0, period_start(periods[0] - dt.timedelta(days=1), grain))
        plan = {"metric": m["name"], "group_by": [mo["group_by"]] if mo.get("group_by") else [], "time_grain": grain,
                "start": periods[0].isoformat(), "end": (next_period(last, grain) - dt.timedelta(days=1)).isoformat()}
        sql, params = self.eng.compile_metric(plan, self.metrics)
        df = tidy(self.eng.run_sql(sql, params))
        if df.empty:
            return alerts
        df["period"] = pd.to_datetime(df["period"]).dt.date
        seg_col = next((c for c in df.columns if c not in ("period", "value", "rows")), None)
        zero_if_empty = empty_means_zero(m)
        for seg, g in (df.groupby(seg_col, dropna=False) if seg_col else [(None, df)]):
            # A period with no rows is missing from the result: 0 for a count or a sum, empty for a rate or an
            # average. Only from the group's first period in the window, so a new branch isn't read as a drop.
            g = g.set_index("period")[["value", "rows"]]
            g = g.reindex([p for p in periods if p >= g.index.min()])
            g["rows"] = g["rows"].fillna(0)
            if zero_if_empty:
                g["value"] = g["value"].fillna(0)
            if last not in g.index or pd.isna(g.at[last, "value"]):
                continue
            # Too few records make a rate or an average jumpy. A count is judged however small.
            if not zero_if_empty and g.at[last, "rows"] < int(mo.get("min_rows") or 0):
                continue
            value = float(g.at[last, "value"])
            hist = g.loc[g.index < last, "value"].astype(float).dropna().tail(self.lookback)
            fire, expected, direction = False, "", "up"
            if mo["method"] == "unusual" and len(hist) >= 4:
                mean, sd = hist.mean(), hist.std(ddof=1) or max(abs(hist.mean()) * 0.01, 1e-9)
                low, high = mean - thr * sd, mean + thr * sd
                fire, expected, direction = not low <= value <= high, f"normal {low:,.1f} to {high:,.1f}", "up" if value > mean else "down"
            elif mo["method"] == "change" and len(hist):
                prev = hist.iloc[-1]
                pct = (value - prev) / abs(prev) * 100 if prev else float("inf")
                fire, expected, direction = abs(pct) >= thr, f"{prev:,.1f} the period before ({pct:+.0f}%)", "up" if value > prev else "down"
            elif mo["method"] == "above":
                fire, expected = value > thr, f"limit {thr:g}"
            elif mo["method"] == "below":
                fire, expected, direction = value < thr, f"limit {thr:g}", "down"
            if fire:
                segment = "all" if seg_col is None else f"{seg_col} = {seg}"
                alerts.append({"action_id": hashlib.sha1(f"{mo['name']}|{segment}|{last}".encode()).hexdigest()[:12],
                               "check_name": mo["name"], "metric": m["label"], "segment": segment,
                               "period": f"{grain} of {last}", "value": round(value, 3), "expected": expected,
                               "direction": direction + (" (worse)" if (m.get("higher_is") == "worse") == (direction == "up")
                                                         and m.get("higher_is") in ("worse", "better") else "")})
        return alerts

    def explain(self, alerts):
        """One short AI sentence per alert (cached). Falls back to a plain sentence."""
        base = {a["action_id"]: f"{a['metric']} ({a['segment']}) was {fmt(a['value'])} in the {a['period']}: {a['expected']}."
                for a in alerts}
        if not alerts:
            return base
        try:
            ans = self.ai.ask_json("You explain data alerts to a busy manager in one plain sentence each. Answer only with JSON.",
                                   json.dumps([{k: a[k] for k in ("action_id", "metric", "segment", "period", "value",
                                                                  "expected", "direction")} for a in alerts])
                                   + "\n\nReturn {\"<action_id>\": \"one sentence: what happened and what to check first\"}.",
                                   expect=dict)
            return {k: ans.get(k) or v for k, v in base.items()}
        except Exception:
            return base

    def load_actions(self):
        return self.eng.backend.load_actions(ACTION_COLS)

    def save_actions(self, df):
        df = df[ACTION_COLS].copy()
        for c in ACTION_COLS:
            df[c] = df[c].astype(float) if c == "value" else df[c].astype(str)
        self.eng.backend.save_actions(df)

    def send_email(self, new):
        """Gmail, if GMAIL_ADDRESS and GMAIL_APP_PASSWORD are set as environment variables (never in a file)."""
        addr, pw = os.environ.get("GMAIL_ADDRESS"), os.environ.get("GMAIL_APP_PASSWORD")
        if not (addr and pw):
            return "no email (set GMAIL_ADDRESS and GMAIL_APP_PASSWORD to get one)"
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = f"{len(new)} new alert(s) in {self.label}", addr, addr
        msg.set_content("\n".join(f"- {a['message']}  [{a['action_id']}]" for a in new))
        rows = "".join(f"<tr><td>{html.escape(a['metric'])}</td><td>{html.escape(a['segment'])}</td>"
                       f"<td>{html.escape(a['period'])}</td><td><b>{fmt(a['value'])}</b></td><td>{html.escape(a['expected'])}</td>"
                       f"<td>{html.escape(a['message'])}</td></tr>" for a in new)
        msg.add_alternative(f"<p>{len(new)} new alert(s) in <b>{html.escape(self.label)}</b>.</p>"
                            "<table border='1' cellpadding='6' style='border-collapse:collapse'>"
                            "<tr><th>Metric</th><th>Where</th><th>When</th><th>Value</th><th>Expected</th><th>Note</th></tr>"
                            f"{rows}</table>", subtype="html")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context(), timeout=30) as s:
            s.login(addr, pw)
            s.send_message(msg)
        return f"emailed to {addr}"

    def run(self, monitors, email=True):
        alerts, failed = [], []
        for mo in monitors:
            try:
                alerts += self.evaluate(mo)
            except Exception as e:        # one broken check doesn't stop the others
                failed.append(f"{mo.get('name')}: {str(getattr(e, 'message', e)).splitlines()[0][:200]}")
        actions = self.load_actions()
        new = [a for a in alerts if a["action_id"] not in set(actions["action_id"])]
        notes = self.explain(new)
        now = dt.datetime.now().isoformat(timespec="seconds")
        for a in new:
            a.update(message=notes[a["action_id"]], status="open", owner="", created_at=now, updated_at=now)
        if new:
            actions = pd.concat([actions, pd.DataFrame(new)[ACTION_COLS]], ignore_index=True) if len(actions) \
                else pd.DataFrame(new)[ACTION_COLS]
            self.save_actions(actions)
        try:
            sent = self.send_email(new) if (email and new) else "no email"
        except Exception as e:            # the tasks are saved already: say why the email didn't go
            sent = f"the email FAILED ({type(e).__name__}: {str(e)[:120]}), but the new tasks are saved"
        return {"checked": len(monitors), "fired": len(alerts), "new": new, "email": sent, "actions": actions,
                "failed": failed}

    def update_action(self, action_id, status, owner=""):
        actions = self.load_actions()
        hit = actions["action_id"] == action_id
        if not hit.any():
            return f"No task {action_id}"
        actions.loc[hit, "status"] = status
        if owner:
            actions.loc[hit, "owner"] = owner
        actions.loc[hit, "updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        self.save_actions(actions)
        return f"Task {action_id}: {status}" + (f", owner {owner}" if owner else "")
