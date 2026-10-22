import json
import uuid
from pathlib import Path as _P

import pytest as _pt


@_pt.fixture
def wtmp():
    d = _P(".tmp-t") / ("t-" + uuid.uuid4().hex[:12])
    d.mkdir(parents=True, exist_ok=True)
    yield d
from intel_net import registry


def test_schema_and_crash_resume(wtmp):
    db = wtmp / "t.db"
    con = registry.connect(db)
    con.execute("INSERT INTO documents VALUES('k','p','s','id','h','u','2026-10-10T09:00:00+00:00','2026-10-10T10:00:00+00:00','published','g',NULL)")
    con.commit()
    con.close()
    con = registry.connect(db)  # resume after crash: schema + row survive
    assert con.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"] == 1
    assert con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    con.close()


def test_register_questions_idempotent(wtmp):
    from intel_net import questions
    assets = wtmp / "assets.json"
    assets.write_text(json.dumps({"assets": [{"ticker": "btc", "name": "Bitcoin"}]}), encoding="utf-8")
    con = registry.connect(wtmp / "t.db")
    bank = questions.load_bank("examples/questions.yaml", assets_path=str(assets))
    registry.register_questions(con, bank)
    n1 = con.execute("SELECT COUNT(*) c FROM question_versions").fetchone()["c"]
    registry.register_questions(con, bank)
    n2 = con.execute("SELECT COUNT(*) c FROM question_versions").fetchone()["c"]
    assert n1 == n2 and n1 >= 12
    ids = {r["question_id"] for r in con.execute("SELECT question_id FROM question_versions")}
    assert "asset.btc.specific" in ids and "market.crypto" in ids
    con.close()
