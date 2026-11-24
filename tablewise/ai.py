"""Every AI call goes through here: Claude, a spending limit, and a cache so the same question is never paid twice.

The API key is only ever held in memory: it comes from the ANTHROPIC_API_KEY environment variable or is typed into
the app, and it is never written to a file, the cache or a log.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path


class AIUnavailable(RuntimeError):
    """No key, no credit, or the budget is spent."""


class AI:
    def __init__(self, cache_file, model="claude-sonnet-5-5", price_in=2.00, price_out=10.00, budget_usd=0.50,
                 api_key=None):
        self.cache_file = Path(cache_file)
        self.model, self.price_in, self.price_out, self.budget = model, price_in, price_out, budget_usd
        self.spent, self.calls, self.cache_hits, self.problem = 0.0, 0, 0, ""
        self.client = None
        try:
            self.cache = json.loads(self.cache_file.read_text())
        except Exception:
            self.cache = {}
        self.set_key(api_key or os.environ.get("ANTHROPIC_API_KEY"))

    # ---- the key ----
    def set_key(self, api_key):
        """Use this key from now on (memory only). An empty key switches the AI off."""
        api_key = (api_key or "").strip()
        if not api_key:
            self.client, self.problem = None, "no Claude API key"
            return
        import anthropic
        self.client, self.problem = anthropic.Anthropic(api_key=api_key, max_retries=2), ""

    @property
    def available(self):
        return self.client is not None and not self.problem

    def status(self):
        state = f"Claude ({self.model})" if self.available else f"off: {self.problem}"
        return (f"AI: {state} · spent this session ${self.spent:.3f} of ${self.budget:.2f} · "
                f"{self.calls} calls, {self.cache_hits} answered from the cache")

    # ---- the cache ----
    def _key(self, system, prompt):
        return hashlib.sha256(json.dumps([self.model, system, prompt]).encode()).hexdigest()

    def _save(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_file.with_name(self.cache_file.name + ".tmp")     # write, then swap: never half a file
        tmp.write_text(json.dumps(self.cache))
        tmp.replace(self.cache_file)

    def forget(self, system, prompt):
        """Remove one answer from the cache, e.g. a query plan that turned out not to work."""
        if self.cache.pop(self._key(system, prompt), None) is not None:
            self._save()

    # ---- asking ----
    def ask(self, system, prompt, max_tokens=16000, use_cache=True, check=None):
        """The same question again is answered from the cache, for free. `check(text)` raises if an answer can't be
        used; only answers that pass it are cached, so a cut-off or broken answer is asked again next time."""
        key = self._key(system, prompt)
        cached = self.cache.get(key) if use_cache else None
        if cached:
            try:
                if check:
                    check(cached["text"])
                self.cache_hits += 1
                return cached["text"]
            except Exception:
                pass
        if not self.available:
            raise AIUnavailable(f"The AI is off ({self.problem}). Add your Claude API key to use it.")
        if self.spent >= self.budget:
            raise AIUnavailable(f"This session reached its AI budget of ${self.budget:.2f}. Raise it in the settings.")
        import anthropic
        text = None
        for attempt in range(2):
            try:
                # fallbacks="default": if a safety filter declines the request (rare), the API answers it with a
                # fallback model in the same call instead of returning nothing
                resp = self.client.beta.messages.create(
                    model=self.model, max_tokens=max_tokens, system=system,
                    messages=[{"role": "user", "content": prompt}],
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default")
                self.spent += (resp.usage.input_tokens * self.price_in + resp.usage.output_tokens * self.price_out) / 1e6
                self.calls += 1
                if resp.stop_reason == "refusal":
                    why = getattr(getattr(resp, "stop_details", None), "category", None)
                    raise ValueError("Claude declined this request" + (f" ({why})" if why else "") + ". Try rewording it.")
                if resp.stop_reason == "max_tokens":
                    raise ValueError(f"The AI's answer was cut off at {max_tokens} tokens, so it can't be used.")
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                break
            except anthropic.APIStatusError as e:
                msg = str(e).lower()
                if e.status_code in (401, 402, 403) or (e.status_code in (400, 429) and any(
                        w in msg for w in ("credit", "billing", "spend limit", "spend cap"))):
                    self.problem = f"Claude said: {str(e)[:160]}"
                    raise AIUnavailable(self.problem) from e
                if attempt == 1:
                    raise
                time.sleep(5)                      # overloaded or a server error: wait, then try once more
            except anthropic.APIConnectionError:
                if attempt == 1:
                    raise
                time.sleep(5)
        if check:
            check(text)                            # raises: a bad answer is NOT cached
        self.cache[key] = {"text": text, "model": self.model, "time": time.strftime("%Y-%m-%d %H:%M")}
        self._save()
        return text

    def ask_json(self, system, prompt, expect=None, **kw):
        """ask(), then read the JSON out of the answer. An answer that isn't valid JSON of the expected shape raises
        and is never cached."""
        return parse_json(self.ask(system, prompt, check=lambda text: parse_json(text, expect), **kw), expect)


def parse_json(text, expect=None):
    """The JSON in an AI answer (with or without ```json fences). expect=list or dict also checks its shape;
    a list wrapped in an object ({"metrics": [...]}) is accepted where a list is expected."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    body = m.group(1) if m else text
    start = min([i for i in (body.find("{"), body.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("The AI didn't return JSON:\n" + text[:300])
    end = max(body.rfind("}"), body.rfind("]"))
    data = json.loads(body[start:end + 1])
    if expect is list and isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        data = lists[0] if len(lists) == 1 else data
    if expect is not None and not isinstance(data, expect):
        raise ValueError(f"The AI returned a {type(data).__name__} where a {expect.__name__} was expected:\n{text[:300]}")
    return data
