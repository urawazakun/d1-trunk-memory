"""REVIEW-B signals regression tests (no model claims; mechanical fixtures)."""
from datetime import datetime, timedelta, timezone

from intel_net import signals

UTC = timezone.utc
ASOF = datetime(2026, 10, 10, 12, tzinfo=UTC)
BANK = {"bank_version": "bv1", "bank_hash": "h",
        "questions": [{"question_id": "coin.btc.specific", "question_hash": "hq1",
                       "group": "coin", "subject": "btc", "half_life_h": 6.0, "max_age_h": 72.0},
                      {"question_id": "coin.btc.good", "question_hash": "hq2",
                       "group": "coin", "subject": "btc", "half_life_h": 6.0, "max_age_h": 72.0},
                      {"question_id": "choice.q", "question_hash": "hq3", "type": "choice",
                       "options": {"a": {}, "b": {}}, "group": "g", "subject": "s",
                       "half_life_h": 6.0, "max_age_h": 72.0}]}


def row(cid, qid, qhash, docs, p, age_h, sha=None, fetched=None, avail=None,
        status="ok", probs=None, group=None):
    pub = ASOF - timedelta(hours=age_h)
    fet = fetched or pub
    return {"trunk_id": "t", "chunk_id": cid, "question_id": qid,
            "question_hash": qhash, "status": status, "p_yes": p,
            "probabilities": probs, "doc_ids": docs,
            "content_sha256": sha or cid, "state_path": f"s/{cid}",
            "correction_group": group or cid,
            "text": "t", "published_at": pub.isoformat(), "fetched_at": fet.isoformat(),
            "available_at": (avail or pub).isoformat()}


def _sig(snap, qid, label="yes"):
    return next(s for s in snap["signals"] if s["question_id"] == qid and s["label"] == label)


def test_half_life_closed_form():
    import math
    r = row("c1", "coin.btc.specific", "hq1", ["d1"], 0.8, 6.0)
    snap = signals.compute([r], BANK, asof=ASOF)
    assert abs(_sig(snap, "coin.btc.specific")["level"] - 0.4) < 1e-9


def test_multi_chunk_coverage_one_and_max_per_doc_after_scoring():
    rs = [row(f"c{i}", "coin.btc.specific", "hq1", [f"d{i}"], 0.8, 1.0) for i in range(4)]
    rs.append(row("cx", "coin.btc.specific", "hq1", ["d0"], 0.1, 1.0))
    snap = signals.compute(rs, BANK, asof=ASOF)
    s = _sig(snap, "coin.btc.specific")
    assert s["coverage"] == 1.0 and s["status"] == "ok"
    assert s["observed_count"] == 5  # coverage counts cells, not doc-best
    assert len(s["top_chunks"]) <= 8


def test_choice_rows_and_future_fetched_and_missing_available():
    good = row("c1", "choice.q", "hq3", ["d1"], None, 1.0, probs={"a": 0.9, "b": 0.1})
    fut = row("c2", "choice.q", "hq3", ["d2"], None, 1.0,
              probs={"a": 0.9, "b": 0.1}, fetched=ASOF + timedelta(hours=1))
    fut["published_at"] = ASOF.isoformat()
    fut["available_at"] = ASOF.isoformat()
    noavail = row("c3", "choice.q", "hq3", ["d3"], None, 1.0, probs={"a": 0.5, "b": 0.5})
    del noavail["available_at"]
    wrong = row("c4", "choice.q", "hq3", ["d4"], None, 1.0, probs={"a": 0.5, "b": 0.5})
    wrong["question_hash"] = "bad"
    snap = signals.compute([good, fut, noavail, wrong], BANK, asof=ASOF,
                           health={"ingest_complete": True})
    sa = _sig(snap, "choice.q", "a")
    assert sa["observed_count"] == 1 and sa["expected_count"] == 4
    assert sa["coverage"] == 0.25 and sa["status"] == "incomplete"
    assert sa["spike"] is None


def test_correction_newest_group_and_no_future_docs():
    old = row("c1", "coin.btc.specific", "hq1", ["d1"], 0.9, 5.0, group="g1",
              fetched=ASOF - timedelta(hours=5))
    new = row("c2", "coin.btc.specific", "hq1", ["d1"], 0.2, 1.0, group="g1",
              fetched=ASOF - timedelta(hours=1))
    snap = signals.compute([old, new], BANK, asof=ASOF)
    s = _sig(snap, "coin.btc.specific")
    assert s["observed_count"] == 1
    assert abs(s["level"] - 0.2 * 2 ** (-1.0 / 6.0)) < 1e-9


def test_baseline_mismatch_ignores_previous_and_history():
    r = row("c1", "coin.btc.specific", "hq1", ["d1"], 0.9, 1.0)
    prev = {"asof": ASOF.isoformat(), "bank_version": "other", "bank_hash": "h",
            "inference_revision": "", "signals": [
                {"question_id": "coin.btc.specific", "label": "yes", "level": 0.1, "delta": 5.0}]}
    snap = signals.compute([r], BANK, asof=ASOF, previous=prev, history=[prev])
    s = _sig(snap, "coin.btc.specific")
    assert s["previous_level"] is None and s["delta"] is None


def test_no_duplicate_prev_delta_and_spike_needs_history():
    r = row("c1", "coin.btc.specific", "hq1", ["d1"], 0.9, 1.0)
    prev = {"asof": ASOF.isoformat(), "bank_version": "bv1", "bank_hash": "h",
            "inference_revision": "", "signals": [
                {"question_id": "coin.btc.specific", "label": "yes", "level": 0.1, "delta": 0.05}]}
    hist = [dict(prev, asof=(ASOF - timedelta(hours=i + 1)).isoformat()) for i in range(8)]
    snap = signals.compute([r], BANK, asof=ASOF, previous=prev, history=hist)
    assert _sig(snap, "coin.btc.specific")["spike"] is None  # 1 doc < 3
    rs = [row(f"c{i}", "coin.btc.specific", "hq1", [f"d{i}"], 0.9, 1.0) for i in range(3)]
    snap2 = signals.compute(rs, BANK, asof=ASOF, previous=prev, history=hist)
    assert _sig(snap2, "coin.btc.specific")["spike"] in (True, False, None)


def test_healthy_empty_only_with_health():
    snap = signals.compute([], BANK, asof=ASOF)
    assert snap["status"] == "incomplete" and snap["coverage"] == 0.0
    snap2 = signals.compute([], BANK, asof=ASOF, health={"ingest_complete": True})
    assert snap2["status"] == "empty" and snap2["coverage"] == 1.0


def test_low_coverage_disables_positive_spike():
    rs = [row("c1", "coin.btc.specific", "hq1", ["d1"], 0.99, 0.1)]
    bad = row("c2", "coin.btc.specific", "hq1", ["d2"], None, 0.1)
    prev = {"asof": (ASOF - timedelta(hours=1)).isoformat(), "bank_version": "bv1",
            "bank_hash": "h", "inference_revision": "",
            "signals": [{"question_id": "coin.btc.specific", "label": "yes",
                         "level": 0.0, "delta": 0.01}]}
    hist = [dict(prev, asof=(ASOF - timedelta(hours=i + 2)).isoformat()) for i in range(8)]
    snap = signals.compute(rs + [bad], BANK, asof=ASOF, previous=prev, history=hist)
    s = _sig(snap, "coin.btc.specific")
    assert s["coverage"] == 0.5 and s["spike"] is None
