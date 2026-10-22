"""Typed loopback-only systemone client (mocked in worker tests)."""
from __future__ import annotations

import json
import urllib.request

from .contracts import local_url


class SystemOneClient:
    def __init__(self, base_url: str, timeout_s: int = 120):
        self.base_url = local_url(base_url)
        self.timeout_s = timeout_s

    def ask(self, state, questions, trunk_id: str, cache_bypass: bool = False,
            id_slot=None) -> dict:
        if isinstance(questions, list):
            payload_q = {}
            for q in questions:
                item = {k: v for k, v in q.items() if k in ("type", "instructions", "criteria") and v is not None}
                if q.get("type") == "choice" and "criteria" not in item:
                    options = q.get("options")
                    item["criteria"] = options if isinstance(options, dict) else {str(o): str(o) for o in options or []}
                payload_q[q["question_id"]] = item
        else:
            payload_q = questions
        payload = {"state": state, "questions": payload_q, "trunk_id": trunk_id,
                   "cache_bypass": bool(cache_bypass)}
        if id_slot is not None:
            payload["id_slot"] = id_slot
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.base_url + "/v1/systemone", data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.load(resp)

    def tokenize(self, text: str) -> list:
        body = json.dumps({"content": text}).encode("utf-8")
        req = urllib.request.Request(self.base_url + "/tokenize", data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.load(resp)
        tokens = data.get("tokens", data.get("ids"))
        if not isinstance(tokens, list) or any(not isinstance(t, int) or isinstance(t, bool) for t in tokens):
            raise ValueError("tokenizer response lacks integer token IDs")
        return tokens
