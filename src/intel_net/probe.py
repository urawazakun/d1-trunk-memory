"""Mechanical specificity fixtures and receipted native-noul shadow probe.

Fixtures test explicit named-asset disclosures versus unrelated facts, not
market forecast quality. No model-generated labels are used.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .contracts import atomic_json, digest, utc_iso
from .runner import can_start_request


def fixtures(bank, names=None):
    """Build mechanical fixtures for every ``*.specific`` bank question.

    Positives are explicit named-asset operational disclosures (contain the
    display name and TICKER); negatives are unrelated factual fixtures with
    the name/ticker absent. ``names`` maps subject -> display name.
    """
    names = names or {}
    items = []
    negatives = ["The city weather office recorded 18 millimeters of rainfall today.",
                 "The railway operator announced a timetable change for the east line.",
                 "The hospital opened a new outpatient clinic on Monday.",
                 "The public library extended its opening hours for winter.",
                 "A coastal research agency measured seawater temperature yesterday."]
    for q in bank["questions"]:
        if not q["question_id"].endswith(".specific"):
            continue
        subject = q["subject"]
        name = names.get(subject, subject.upper())
        positives = [f"Today's issuer notice: transfers of {name} ({subject.upper()}) will be paused for a network upgrade tomorrow.",
                     f"The exchange announced today that deposits for {name} ({subject.upper()}) have resumed after maintenance."]
        for i, text in enumerate(positives):
            items.append({"id": f"{subject}-pos-{i}", "question": q, "state": text, "truth": True,
                          "truth_rule": "explicit named-asset operational disclosure", "keyword_free": False})
        for i, text in enumerate(negatives):
            # Mechanical negative: no subject token/display name in a
            # deliberately unrelated nonfinancial factual fixture.
            items.append({"id": f"{subject}-neg-{i}", "question": q, "state": text, "truth": False,
                          "truth_rule": "unrelated fixture with target name/ticker absent", "keyword_free": True})
    return items


def evaluate(rows, threshold=.7):
    positive = negative = tp = fp = keyword_fp = invalid = 0
    for row in rows:
        p = row.get("p_yes")
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(p) or not 0 <= p <= 1:
            invalid += 1
            continue
        prediction = p >= threshold
        if row["truth"]:
            positive += 1
            tp += prediction
        else:
            negative += 1
            fp += prediction
            keyword_fp += prediction and row.get("keyword_free", False)
    recall = tp / positive if positive else None
    fpr = fp / negative if negative else None
    passed = invalid == 0 and positive >= 50 and negative >= 200 and recall >= .8 and fpr <= .05 and keyword_fp == 0
    return {"policy_version": "mechanical-specificity-v1-proposal", "threshold": threshold,
            "positives": positive, "negatives": negative, "invalid": invalid, "true_positive": tp,
            "false_positive": fp, "keyword_free_false_positive": keyword_fp, "recall": recall,
            "false_positive_rate": fpr, "pass": passed,
            "proposed_gate": {"min_positive": 50, "min_negative": 200, "min_recall": .8,
                              "max_false_positive_rate": .05, "max_keyword_free_false_positive": 0},
            "limitation": "procedural named-disclosure specificity only; no evidence of semantic or forecast accuracy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--bank", type=Path, default=Path("examples/questions.yaml"))
    parser.add_argument("--assets", type=Path, default=Path("examples/assets.json"))
    parser.add_argument("--base", default="http://127.0.0.1:8101")
    parser.add_argument("--fixtures-only", action="store_true")
    args = parser.parse_args()
    from .questions import _load_assets, load_bank
    tickers, names = _load_assets(args.assets)
    dataset = fixtures(load_bank(args.bank, assets_path=args.assets), names=names)
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "fixtures.json", dataset)
    if args.fixtures_only:
        return
    config = replace(Config(), base_url=args.base, receipt_path=args.receipt or Config().receipt_path, inference_enabled=True)
    receipt = config.receipt()
    if not receipt:
        raise SystemExit("accepted cutover receipt required; only fixtures were written")
    from .client import SystemOneClient
    model = SystemOneClient(config.base_url)
    rows = []
    for item in dataset:
        if not can_start_request(datetime.now(timezone.utc), quiet_minutes=config.quiet_minutes):
            atomic_json(args.out / "partial.json", rows)
            raise SystemExit("operator quiet window: stopped; partial results retained")
        question = item["question"]
        trunk = "intel-probe:v1:" + digest({"state": item["state"], "model": receipt["model_digest"],
                                            "template": receipt["template_digest"]})
        response = model.ask(item["state"], [question], trunk)
        answer = response.get("answers", {}).get(question["question_id"], {})
        rows.append({**item, "p_yes": answer.get("noul"), "response": response, "available_at": utc_iso()})
        atomic_json(args.out / "raw.json", rows)
    atomic_json(args.out / "summary.json", evaluate(rows))


if __name__ == "__main__":
    main()
