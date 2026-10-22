import os
import uuid
from pathlib import Path as _P
import pytest as _pt

@_pt.fixture
def wtmp():
    d = _P(".tmp-t") / ("t-" + uuid.uuid4().hex[:12])
    d.mkdir(parents=True, exist_ok=True)
    yield d
import gzip
import json
from datetime import datetime, timezone

from intel_net import ingest, registry


def _vault(wtmp):
    v = wtmp / "vault"
    (v / "2026" / "10" / "10" / "src").mkdir(parents=True)
    return v


def _doc(doc_id="a1", fetched="2026-10-10T10:00:00+00:00", published="2026-10-10T09:00:00+00:00",
         text="BTC rises", url="http://x/a", title="T"):
    return {"id": doc_id, "source": "src", "fetched_at": fetched, "published_at": published,
            "url": url, "title": title, "summary": text, "raw": {"text": text}}


def test_reingest_idempotent_and_gzip(wtmp):
    v = _vault(wtmp)
    (v / "2026" / "10" / "10" / "src" / "a.json").write_text(json.dumps(_doc()), encoding="utf-8")
    with gzip.open(v / "2026" / "10" / "10" / "src" / "b.json.gz", "wt", encoding="utf-8") as f:
        json.dump(_doc("b1", text="ETH falls", url="http://x/b"), f)
    con = registry.connect(wtmp / "t.db")
    bank = {"bank_version": "v", "questions": []}
    s1 = ingest.ingest(con, v, wtmp / "chunks", asof="2026-10-10T11:00:00+00:00",
                       bank=bank, model_digest="m", template_digest="t")
    s2 = ingest.ingest(con, v, wtmp / "chunks", asof="2026-10-10T11:00:00+00:00",
                       bank=bank, model_digest="m", template_digest="t")
    assert s1["docs_kept"] == 2 and s2["trunks"] == 0
    assert con.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"] == 2
    con.close()


def test_quarantine_future_and_malformed(wtmp):
    v = _vault(wtmp)
    (v / "2026" / "10" / "10" / "src" / "f.json").write_text(
        json.dumps(_doc(published="2026-10-11T00:00:00+00:00")), encoding="utf-8")
    (v / "2026" / "10" / "10" / "src" / "bad.json").write_text("{nope", encoding="utf-8")
    con = registry.connect(wtmp / "t.db")
    s = ingest.ingest(con, v, wtmp / "chunks", asof="2026-10-10T11:00:00+00:00",
                      bank={"bank_version": "v", "questions": []},
                      model_digest="m", template_digest="t")
    assert s["quarantined"] == 2 and s["docs_kept"] == 0
    con.close()


def test_same_short_id_different_sources(wtmp):
    v = _vault(wtmp)
    (v / "2026" / "10" / "10" / "src" / "a.json").write_text(json.dumps(_doc("same")), encoding="utf-8")
    d2 = _doc("same", text="other text here", url="http://y")
    d2["source"] = "other"
    (v / "2026" / "10" / "10" / "src" / "b.json").write_text(json.dumps(d2), encoding="utf-8")
    con = registry.connect(wtmp / "t.db")
    ingest.ingest(con, v, wtmp / "chunks", asof="2026-10-10T11:00:00+00:00",
                  bank={"bank_version": "v", "questions": []}, model_digest="m", template_digest="t")
    assert con.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"] == 2
    con.close()


def test_long_ja_chunk_bounds(wtmp):
    from intel_net import chunking
    text = "ビットコインは上昇した。" * 500
    parts, ver, basis = chunking.split_text(text)
    assert len(parts) > 1 and basis == "utf8_upper_bound"
    assert all(len(p.encode("utf-8")) <= 1000 for p in parts)


def test_missing_pubtime_uses_fetched_flag(wtmp):
    v = _vault(wtmp)
    d = _doc(published=None)
    d.pop("published_at")
    (v / "2026" / "10" / "10" / "src" / "a.json").write_text(json.dumps(d), encoding="utf-8")
    con = registry.connect(wtmp / "t.db")
    ingest.ingest(con, v, wtmp / "chunks", asof="2026-10-10T11:00:00+00:00",
                  bank={"bank_version": "v", "questions": []}, model_digest="m", template_digest="t")
    row = con.execute("SELECT publication_basis,published_at,fetched_at FROM documents").fetchone()
    assert row["publication_basis"] == "fetched_fallback" and row["published_at"] == row["fetched_at"]
    con.close()
