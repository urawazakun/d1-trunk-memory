"""Receipted matched-geometry text trunk acceptance; no process control."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import time

from .client import SystemOneClient
from .chunking import split_text
from .config import Config
from .contracts import atomic_json, digest
from .runner import can_start_request


def compare_answers(reference, candidate):
    if not isinstance(reference.get("answers"), dict) or not reference["answers"]:
        raise ValueError("missing reference answers")
    if reference.get("model") != candidate.get("model") or reference["answers"] != candidate.get("answers"):
        raise ValueError("matched trunk answer mismatch (no tolerance)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8101")
    args = parser.parse_args()
    receipt = replace(Config(), inference_enabled=True, receipt_path=args.receipt, base_url=args.base).receipt()
    if not receipt:
        raise SystemExit("accepted operator receipt required")
    model = SystemOneClient(args.base)
    state = split_text(("The exchange resumed Bitcoin (BTC) deposits after maintenance. "
                        "The weather agency also measured regional rainfall.\n") * 80, model.tokenize)[0][0]
    trunk_id = "intel-accept:v1:" + digest([state, receipt["model_digest"], receipt["template_digest"]])
    seed_q = [{"question_id": "seed", "type": "noul", "instructions": "Does this text specifically report on Bitcoin (BTC)?"}]
    new_q = [{"question_id": "later", "type": "noul", "instructions": "Does this text report a resumption of deposit services?"}]
    calls = []
    args.out.mkdir(parents=True, exist_ok=True)
    def ask(label, text, qs, key, bypass=False):
        if not can_start_request(datetime.now(timezone.utc)):
            raise RuntimeError("operator quiet/drain window; stopped before next call")
        started = time.monotonic()
        response = model.ask(text, qs, key, cache_bypass=bypass, id_slot=0)
        calls.append({"case": label, "response": response, "wall_ms": (time.monotonic() - started) * 1000})
        atomic_json(args.out / "raw.json", calls)
        return response
    try:
        for i in range(2):
            ask(f"warmup-{i}", state, new_q, trunk_id, True)
        ask("seed", state, seed_q, trunk_id)
        reference = ask("matched-bypass-new-question", state, new_q, trunk_id, True)
        warm = ask("warm-new-question", state, new_q, trunk_id)
        compare_answers(reference, warm)
        ask("disturb-B", "A separate weather observation: rainfall has stopped.", seed_q, trunk_id + "B")
        revisited = ask("revisit-A", state, new_q, trunk_id)
        compare_answers(reference, revisited)
        summary = {"pass_exact": True, "receipt": receipt, "actual_state_tokens": len(model.tokenize(state)),
                   "note": "Every answer field compared exactly within process/slot; cache residency/hit must be correlated with operator server logs. Eviction/restart acceptance belongs to ops, no shared server restart performed."}
    except Exception as error:
        summary = {"pass_exact": False, "error": f"{type(error).__name__}: {error}", "receipt": receipt}
        atomic_json(args.out / "summary.json", summary)
        raise
    atomic_json(args.out / "summary.json", summary)


if __name__ == "__main__":
    main()
