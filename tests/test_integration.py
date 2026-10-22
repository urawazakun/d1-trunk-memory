from dataclasses import replace
import json

from intel_net.config import Config
from intel_net.contracts import atomic_json, parse_time, utc_iso
from intel_net.runner import tick


def setup_config(tmp_path):
    primary = tmp_path / "ssd"
    vault = tmp_path / "vault"
    path = vault / "2026" / "10" / "10" / "agency" / "item.json"
    atomic_json(path, {"id": "doc", "source": "agency", "title": "Bitcoin (BTC) upgrade",
                       "summary": "Deposits resumed today.", "fetched_at": "2026-10-10T08:00:00+00:00",
                       "published_at": "2026-10-10T08:00:00+00:00", "url": "https://agency.example/notice"})
    bank = tmp_path / "questions.yaml"
    bank.write_text("schema_version: 1\nbank_version: integration-v1\nquestions:\n"
                    "  - question_id: market.crypto\n    type: noul\n    group: crypto\n"
                    "    subject: market\n    instructions: Does this report concern cryptocurrencies?\n", encoding="utf-8")
    atomic_json(tmp_path / "assets.json", {"assets": [{"ticker": "btc", "name": "Bitcoin"}]})
    return replace(Config(), db_path=primary / "intel-net.db", snapshot_dir=primary / "signals",
                   vault_dir=vault, chunk_dir=tmp_path / "hdd" / "chunks", mirror_dir=tmp_path / "hdd" / "mirror",
                   question_bank=bank, assets_path=tmp_path / "assets.json",
                   receipt_path=tmp_path / "receipt.json", quiet_minutes=12)


def test_offline_to_receipted_model_preserves_pending_and_answers(monkeypatch, tmp_path):
    from intel_net import client, matrix, questions, registry
    cfg = setup_config(tmp_path)
    now = parse_time("2026-10-10T09:30:00+00:00")
    called = []
    class FakeClient:
        def __init__(self, *args, **kw): pass
        def tokenize(self, text): return list(range(len(text)))
        def ask(self, state, bank_questions, trunk_id, **kwargs):
            called.append(trunk_id)
            return {"model": "d1-model-name", "answers": {q["question_id"]: {"type": "noul", "noul": .9}
                                                           for q in bank_questions}}
    monkeypatch.setattr(client, "SystemOneClient", FakeClient)
    first = tick(cfg, now_fn=lambda: now)
    assert not called and first["inference"] == "awaiting-cutover-receipt"
    atomic_json(cfg.receipt_path, {"status": "accepted", "base_url": cfg.base_url, "decision_prefix_cache": True,
                                  "model_digest": "model-sha", "template_digest": "template-sha",
                                  "binary_digest": "binary-sha", "server_generation": "boot", "per_slot_context": 4096})
    second = tick(replace(cfg, inference_enabled=True), now_fn=lambda: now)
    assert called and second["answered_batches"] >= 1
    from intel_net.consumer import read_snapshot
    # computed_at is wall-clock: read with a wide age bound so this frozen-asof
    # test stays valid after its fixture date.
    assert read_snapshot(cfg.snapshot_dir, asof=utc_iso(), max_age_h=24 * 365)["coverage"] == 1
    con = registry.connect(cfg.db_path)
    bank = questions.load_bank(cfg.question_bank, assets_path=cfg.assets_path)
    rows = matrix.feature_rows(con, bank, asof=now, model_digest="model-sha", template_digest="template-sha")
    assert rows and all(row["p_yes"] == .9 for row in rows)
    assert all(row["model_digest"] == "model-sha" for row in rows)
    assert not matrix.pending(con, bank, asof=now, model_digest="model-sha", template_digest="template-sha")
    con.close()


def test_priority_tick_does_not_call_tokenizer_or_inference(monkeypatch, tmp_path):
    from intel_net import client
    cfg = setup_config(tmp_path)
    atomic_json(cfg.receipt_path, {"status": "accepted", "base_url": cfg.base_url, "decision_prefix_cache": True,
                                  "model_digest": "m", "template_digest": "t", "binary_digest": "b",
                                  "server_generation": "boot", "per_slot_context": 4096})
    class NoCalls:
        def __init__(self, *args, **kwargs): pass
        def tokenize(self, *args): raise AssertionError("tokenizer called during priority window")
        def ask(self, *args): raise AssertionError("inference called during priority window")
    monkeypatch.setattr(client, "SystemOneClient", NoCalls)
    result = tick(replace(cfg, inference_enabled=True), now_fn=lambda: parse_time("2026-10-10T09:01:00+00:00"))
    assert result["priority_paused"] and result["answered_batches"] == 0


def test_live_ingest_budget_advances_past_unchanged_documents(tmp_path):
    from intel_net import ingest, registry
    cfg = setup_config(tmp_path)
    for i in range(4):
        atomic_json(cfg.vault_dir / "2026/10/10/agency" / f"more-{i}.json",
                    {"id": f"d{i}", "source": "agency", "title": "item", "summary": f"notice {i}",
                     "fetched_at": "2026-10-10T08:00:00+00:00", "published_at": "2026-10-10T08:00:00+00:00"})
    con = registry.connect(cfg.db_path)
    summaries = []
    for _ in range(3):
        summaries.append(ingest.ingest(con, cfg.vault_dir, cfg.chunk_dir, asof="2026-10-10T09:30:00+00:00",
            bank={"questions": []}, model_digest="m", template_digest="t", limit=2))
    assert [s["incomplete"] for s in summaries] == [True, True, False]
    assert con.execute("SELECT COUNT(*) FROM ingest_state").fetchone()[0] == 5
    con.close()
