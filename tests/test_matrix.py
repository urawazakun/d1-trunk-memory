import os
import uuid
from pathlib import Path as _P
import pytest as _pt

@_pt.fixture
def wtmp():
    d = _P(".tmp-t") / ("t-" + uuid.uuid4().hex[:12])
    d.mkdir(parents=True, exist_ok=True)
    yield d
import json

from intel_net import ingest, matrix, registry


def _mk(wtmp, docs):
    v = wtmp / "vault"
    (v / "d").mkdir(parents=True)
    for i, d in enumerate(docs):
        (v / "d" / f"{i}.json").write_text(json.dumps(d), encoding="utf-8")
    con = registry.connect(wtmp / "t.db")
    bank = {"bank_version": "v1", "bank_hash": "h",
            "questions": [
                {"question_id": "q1", "question_hash": "h1", "group": "crypto",
                 "subject": "btc", "type": "noul", "instructions": "up?", "options": None,
                 "half_life_h": 6, "max_age_h": 72, "active": True},
                {"question_id": "q2", "question_hash": "h2", "group": "crypto",
                 "subject": "btc", "type": "choice", "instructions": "pick",
                 "options": {"A": "yes", "B": "no"},
                 "half_life_h": 6, "max_age_h": 72, "active": True}]}
    registry.register_questions(con, bank)
    ingest.ingest(con, v, wtmp / "chunks", asof="2026-10-10T11:00:00+00:00",
                  bank=bank, model_digest="m", template_digest="t")
    return con, bank


def _doc(url="http://x/a", fetched="2026-10-10T10:00:00+00:00", text="BTC up"):
    return {"id": "a", "source": "s", "fetched_at": fetched,
            "published_at": "2026-10-10T09:00:00+00:00", "url": url,
            "title": "T", "summary": text, "raw": {"text": text}}


def test_partial_and_invalid_retry(wtmp):
    con, bank = _mk(wtmp, [_doc()])
    tasks = matrix.pending(con, bank, asof="2026-10-10T12:00:00+00:00")
    assert len(tasks) == 1 and len(tasks[0]["questions"]) == 2
    task = tasks[0]
    s = matrix.record(con, task, {"answers": {"q1": {"type": "noul", "noul": 0.8},
                                              "q2": {"type": "choice", "probabilities": {"ZZZ": 1.0}}}},
                      available_at="2026-10-10T12:00:00+00:00", wall_ms=5)
    assert (s["ok"], s["invalid"], s["missing"]) == (1, 1, 0)
    tasks2 = matrix.pending(con, bank, asof="2026-10-10T12:00:00+00:00")
    assert len(tasks2) == 1 and [q["question_id"] for q in tasks2[0]["questions"]] == ["q2"]
    # nonfinite never becomes zero
    matrix.record(con, tasks2[0], {"answers": {"q2": {"type": "choice", "probabilities": {"A": float("inf"), "B": 0.0}}}},
                  available_at="2026-10-10T12:01:00+00:00")
    rows = matrix.feature_rows(con, bank, asof="2026-10-10T12:02:00+00:00")
    q2 = [r for r in rows if r["question_id"] == "q2"][0]
    assert q2["status"] == "invalid" and q2["p_yes"] is None
    con.close()


def test_correction_point_in_time(wtmp):
    old = _doc(fetched="2026-10-10T08:00:00+00:00", text="BTC up big")
    new = _doc(fetched="2026-10-10T10:00:00+00:00", text="correction: BTC flat")
    con, bank = _mk(wtmp, [old, new])
    early = matrix.feature_rows(con, bank, asof="2026-10-10T09:00:00+00:00")
    late = matrix.feature_rows(con, bank, asof="2026-10-10T11:00:00+00:00")
    assert early[0]["content_sha256"] != late[0]["content_sha256"]  # no later leakage into earlier snapshot
    con.close()


def test_dedup_across_sources(wtmp):
    con, bank = _mk(wtmp, [_doc(url="http://x/a"), _doc(url="http://y/b")])
    rows = matrix.feature_rows(con, bank, asof="2026-10-10T11:00:00+00:00")
    assert {r["content_sha256"] for r in rows} and len({r["chunk_id"] for r in rows}) == 1
    con.close()


def test_idempotent_record_and_trunk_id_shape(wtmp):
    con, bank = _mk(wtmp, [_doc()])
    task = matrix.pending(con, bank, asof="2026-10-10T12:00:00+00:00")[0]
    assert task["trunk_id"].startswith("intel:v1:") and len(task["trunk_id"].encode()) < 128
    resp = {"answers": {"q1": {"type": "noul", "noul": 0.2},
                        "q2": {"type": "choice", "probabilities": {"A": 0.6, "B": 0.4}}}}
    matrix.record(con, task, resp, available_at="2026-10-10T12:00:00+00:00")
    matrix.record(con, task, resp, available_at="2026-10-10T12:00:00+00:00")
    assert matrix.pending(con, bank, asof="2026-10-10T12:00:00+00:00") == []
    con.close()
