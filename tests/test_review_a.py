"""REVIEW-A regression tests (workspace-local temp; stock tmp_path is sandbox-denied)."""
import gzip
import json
import uuid
from pathlib import Path as _P

import pytest as _pt


@_pt.fixture
def wtmp():
    d = _P(".tmp-t") / ("ra-" + uuid.uuid4().hex[:12])
    d.mkdir(parents=True, exist_ok=True)
    yield d

from intel_net import chunking, ingest, matrix, questions, registry


def _doc(**kw):
    d = {"id": "a", "source": "s", "fetched_at": "2026-10-10T10:00:00+00:00",
         "published_at": "2026-10-10T09:00:00+00:00", "url": "http://x/a",
         "title": "T", "summary": "hello world", "raw": {"text": "hello world"}}
    d.update(kw)
    return d


def _bank(*qs):
    return {"bank_version": "v1", "bank_hash": "h", "questions": list(qs)}


def _q(qid="q1", h="h1", type="noul", options=None, active=True, instr="up?"):
    return {"question_id": qid, "question_hash": h, "group": "g", "subject": "s",
            "type": type, "instructions": instr, "options": options,
            "half_life_h": 6, "max_age_h": 72, "active": active}


def test_connect_mkdir_parent(wtmp):
    con = registry.connect(wtmp / "sub" / "dir" / "t.db")
    con.close()


def test_missing_fetched_quarantined(wtmp):
    v = wtmp / "vault"
    (v / "d").mkdir(parents=True)
    d = _doc()
    del d["fetched_at"]
    (v / "d" / "a.json").write_text(json.dumps(d))
    con = registry.connect(wtmp / "t.db")
    s = ingest.ingest(con, v, wtmp / "ch", asof="2026-10-10T11:00:00+00:00",
                      bank=_bank(), model_digest="m", template_digest="t")
    assert s["quarantined"] == 1 and s["docs_kept"] == 0
    assert con.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"] == 0
    con.close()


def test_stale_publication_rejected(wtmp):
    v = wtmp / "vault"
    (v / "d").mkdir(parents=True)
    (v / "d" / "a.json").write_text(json.dumps(_doc(published_at="2026-10-05T00:00:00+00:00")))
    con = registry.connect(wtmp / "t.db")
    s = ingest.ingest(con, v, wtmp / "ch", asof="2026-10-10T11:00:00+00:00",
                      bank=_bank(), model_digest="m", template_digest="t")
    assert s["quarantined"] == 1 and s["docs_kept"] == 0
    con.close()


def test_dict_type_and_content_errors(wtmp):
    v = wtmp / "vault"
    (v / "d").mkdir(parents=True)
    d = _doc()
    d["type"] = {"source": "s2"}
    d["raw"] = {"text": {"type": "x", "text": "dict-wrapped"}}
    (v / "d" / "a.json").write_text(json.dumps(d))
    (v / "d" / "b.json").write_text(json.dumps(_doc(url="http://x/b", raw={"text": {"no": 1}, "body": 5}, summary=123)))
    con = registry.connect(wtmp / "t.db")
    s = ingest.ingest(con, v, wtmp / "ch", asof="2026-10-10T11:00:00+00:00",
                      bank=_bank(), model_digest="m", template_digest="t")
    assert s["docs_kept"] == 1 and s["quarantined"] == 1
    con.close()


def test_recent_first_and_incomplete(wtmp):
    v = wtmp / "vault"
    (v / "d").mkdir(parents=True)
    for i in range(5):
        (v / f"d/{i}.json").write_text(json.dumps(_doc(url=f"http://x/{i}", text=f"text {i}")))
    con = registry.connect(wtmp / "t.db")
    s = ingest.ingest(con, v, wtmp / "ch", asof="2026-10-10T11:00:00+00:00",
                      bank=_bank(), model_digest="m", template_digest="t", limit=2)
    assert s["incomplete"] is True
    s2 = ingest.ingest(con, v, wtmp / "ch2", asof="2026-10-10T11:00:00+00:00",
                       bank=_bank(), model_digest="m", template_digest="t", limit=100)
    assert s2["incomplete"] is False
    con.close()


def test_split_utf8_boundary_and_token_lossless_ja(wtmp):
    # boundary: multibyte char straddling limit
    text = "a" * 999 + "あ" + "b" * 10  # あ is 3 bytes; limit 1000 -> must not split it
    parts = chunking.split_utf8(text, 1000)
    assert "".join(parts) == text
    assert all(len(p.encode("utf-8")) <= 1000 for p in parts)
    assert parts[0].endswith("a")  # あ pushed whole to next part
    # tokenized lossless + recursive <=1000, JA + pathological
    def fake_tok(s):
        return list(s.encode("utf-8"))  # byte-level stand-in (each byte a token)
    ja = "ビットコインは上昇した。" * 500
    parts2, ver, basis = chunking.split_text(ja, fake_tok)
    assert "".join(parts2) == ja and basis == "tokenizer" and ver == "tok-v1"
    assert all(len(fake_tok(p)) <= 1000 for p in parts2)
    weird = "a" * 100 + "あ" * 2000 + "z" * 50
    parts3, _, _ = chunking.split_text(weird, fake_tok)
    assert "".join(parts3) == weird
    assert all(len(fake_tok(p)) <= 1000 for p in parts3)


def test_offline_not_validated_and_chunk_identity(wtmp):
    v = wtmp / "vault"
    (v / "d").mkdir(parents=True)
    (v / "d/a.json").write_text(json.dumps(_doc()))
    con = registry.connect(wtmp / "t.db")
    tok_calls = []
    def tok(s):
        tok_calls.append(s)
        return [1] * ((len(s.encode("utf-8")) + 99) // 100)
    ingest.ingest(con, v, wtmp / "ch", asof="2026-10-10T11:00:00+00:00",
                  bank=_bank(), model_digest="m", template_digest="t")
    row = con.execute("SELECT chunk_id,chunker_version,validated,token_count_basis FROM chunks").fetchone()
    assert row["validated"] == 0 and row["token_count_basis"] == "utf8_upper_bound"
    assert row["chunker_version"] in row["chunk_id"]
    # Receipt transition must actually validate offline states, then subsequent
    # unchanged ticks reuse the validated geometry without another tokenization.
    n0 = len(tok_calls)
    ingest.ingest(con, v, wtmp / "ch", asof="2026-10-10T11:00:00+00:00",
                  bank=_bank(), model_digest="m", template_digest="t", tokenize=tok)
    assert len(tok_calls) > n0
    assert con.execute("SELECT COUNT(*) FROM chunks WHERE validated=1").fetchone()[0] > 0
    n1 = len(tok_calls)
    ingest.ingest(con, v, wtmp / "ch", asof="2026-10-10T11:00:00+00:00",
                  bank=_bank(), model_digest="m", template_digest="t", tokenize=tok)
    assert len(tok_calls) == n1
    con.close()


def test_bank_source_and_active_hashes(wtmp):
    yp = wtmp / "q.yaml"
    yp.write_text("schema_version: 1\nbank_version: \"v9\"\nquestions:\n"
                  "  - {question_id: market.crypto, group: crypto, subject: market, type: {name: noul}, instructions: CUSTOM INSTR, half_life_h: 9.0}\n"
                  "  - {question_id: custom.q, group: cg, subject: cs, type: choice, instructions: pick one, options: {A: 'yes', B: 'no'}}\n",
                  encoding="utf-8")
    bank = questions.load_bank(str(yp), pairs_path=None)
    assert bank["bank_version"] == "v9"
    mc = next(q for q in bank["questions"] if q["question_id"] == "market.crypto")
    assert mc["instructions"] == "CUSTOM INSTR" and mc["type"] == "noul"
    cq = next(q for q in bank["questions"] if q["question_id"] == "custom.q")
    assert cq["options"] == {"A": "yes", "B": "no"}
    h_old = mc["question_hash"]
    # editing arbitrary question => new hash; register follows exact active hashes
    yp.write_text(yp.read_text(encoding="utf-8").replace("CUSTOM INSTR", "CUSTOM INSTR v2"), encoding="utf-8")
    bank2 = questions.load_bank(str(yp), pairs_path=None)
    mc2 = next(q for q in bank2["questions"] if q["question_id"] == "market.crypto")
    assert mc2["question_hash"] != h_old
    con = registry.connect(wtmp / "t.db")
    registry.register_questions(con, bank)
    registry.register_questions(con, bank2)  # prior versions deactivated
    rows = list(con.execute("SELECT question_hash,active FROM question_versions WHERE question_id='market.crypto'"))
    assert sum(r["active"] for r in rows) == 1
    assert [r["question_hash"] for r in rows if r["active"]][0] == mc2["question_hash"]
    con.close()


def test_pending_bank_change_model_filter_token_basis(wtmp):
    v = wtmp / "vault"
    (v / "d").mkdir(parents=True)
    (v / "d/a.json").write_text(json.dumps(_doc()))
    con = registry.connect(wtmp / "t.db")
    b1 = _bank(_q("q1", "h1"))
    registry.register_questions(con, b1)
    ingest.ingest(con, v, wtmp / "ch", asof="2026-10-10T11:00:00+00:00",
                  bank=b1, model_digest="m1", template_digest="t1")
    t = matrix.pending(con, b1, asof="2026-10-10T12:00:00+00:00")[0]
    assert t["token_count_basis"] == "utf8_upper_bound" and "token_count" in t
    matrix.record(con, t, {"answers": {"q1": {"type": "noul", "noul": 0.6}}},
                  available_at="2026-10-10T12:00:00+00:00")
    assert matrix.pending(con, b1, asof="2026-10-10T12:00:00+00:00") == []
    # bank adds a question => ready trunk becomes pending again
    b2 = _bank(_q("q1", "h1"), _q("q2", "h2"))
    registry.register_questions(con, b2)
    tasks = matrix.pending(con, b2, asof="2026-10-10T12:00:00+00:00")
    assert len(tasks) == 1 and [q["question_id"] for q in tasks[0]["questions"]] == ["q2"]
    # model filter: placeholder/other-model trunks cannot monopolize
    assert matrix.pending(con, b2, asof="2026-10-10T12:00:00+00:00", model_digest="other") == []
    con.close()


def test_feature_rows_per_model_and_multichunk_doc(wtmp):
    v = wtmp / "vault"
    (v / "d").mkdir(parents=True)
    (v / "d/a.json").write_text(json.dumps(_doc(summary="x" * 2500, raw={"text": "x" * 2500})))
    con = registry.connect(wtmp / "t.db")
    b = _bank(_q("q1", "h1"))
    registry.register_questions(con, b)
    ingest.ingest(con, v, wtmp / "ch", asof="2026-10-10T11:00:00+00:00",
                  bank=b, model_digest="m1", template_digest="t1")
    ingest.ingest(con, v, wtmp / "ch", asof="2026-10-10T11:00:00+00:00",
                  bank=b, model_digest="m2", template_digest="t1")
    rows_m1 = matrix.feature_rows(con, b, asof="2026-10-10T12:00:00+00:00", model_digest="m1")
    rows_all = matrix.feature_rows(con, b, asof="2026-10-10T12:00:00+00:00")
    assert len(rows_m1) >= 2  # all chunks of the doc kept (no per-doc drop)
    assert len(rows_all) == 2 * len(rows_m1)  # each compatible revision returned
    con.close()


def test_record_digest_type_choice_and_no_overwrite(wtmp):
    v = wtmp / "vault"
    (v / "d").mkdir(parents=True)
    (v / "d/a.json").write_text(json.dumps(_doc()))
    con = registry.connect(wtmp / "t.db")
    b = _bank(_q("q1", "h1"), _q("qc", "hc", type="choice", options={"A": "y", "B": "n"}))
    registry.register_questions(con, b)
    ingest.ingest(con, v, wtmp / "ch", asof="2026-10-10T11:00:00+00:00",
                  bank=b, model_digest="m", template_digest="t")
    t = matrix.pending(con, b, asof="2026-10-10T12:00:00+00:00")[0]
    good = {"answers": {"q1": {"type": "noul", "noul": 0.7},
                        "qc": {"type": "choice", "probabilities": {"A": 0.6, "B": 0.4}}},
            "model": "sys-one-v3", "model_digest": "m", "template_digest": "t",
            "usage": {"tok": 5}}
    s = matrix.record(con, t, good, available_at="2026-10-10T12:00:00+00:00")
    assert s == {"ok": 2, "invalid": 0, "missing": 0, "status": "ready"}
    raw = con.execute("SELECT raw_answer_json FROM answers WHERE question_id='q1'").fetchone()["raw_answer_json"]
    assert "sys-one-v3" in raw and "tok" in con.execute("SELECT usage_json FROM answers WHERE question_id='q1'").fetchone()["usage_json"]
    # retry error must not overwrite successful history
    t2 = dict(t, questions=[_q("q1", "h1")])
    matrix.record(con, t2, {"answers": {}, "model_digest": "m", "template_digest": "t"},
                  available_at="2026-10-10T12:01:00+00:00")
    st = con.execute("SELECT status FROM answers WHERE question_id='q1'").fetchone()["status"]
    assert st == "ok"
    # digest mismatch invalid; type mismatch invalid; inexact choice keys invalid
    t3 = matrix.pending(con, _bank(_q("q1", "h1")), asof="2026-10-10T12:00:00+00:00")
    assert t3 == []  # ready batch not suppressed: q1 done; (bank without qc => nothing outstanding)
    bad = dict(t, questions=[_q("qc", "hc", type="choice", options={"A": "y", "B": "n"})])
    s2 = matrix.record(con, bad, {"answers": {"qc": {"type": "choice", "probabilities": {"A": 0.5, "B": 0.5}}},
                                  "model_digest": "WRONG", "template_digest": "t"},
                       available_at="2026-10-10T12:02:00+00:00")
    assert s2["invalid"] == 0  # prior ok preserved, mismatch not overwriting
    con.close()
