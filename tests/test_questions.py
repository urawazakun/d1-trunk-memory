import json

from intel_net import questions


def _assets(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_bank_seeds_groups_and_generates_assets(tmp_path):
    assets = _assets(tmp_path / "assets.json", {"assets": [
        {"ticker": "btc", "name": "Bitcoin"},
        {"ticker": "eth", "name": "Ethereum"},
        {"ticker": "unknowncoin"},  # no display name: never guessed
    ]})
    bank = questions.load_bank("examples/questions.yaml", assets_path=assets)
    ids = {q["question_id"] for q in bank["questions"]}
    assert {"market.crypto", "market.rates", "market.fx", "market.equities",
            "world.conflict", "freshness.new"} <= ids
    assert "asset.btc.specific" in ids and "asset.btc.good" in ids
    assert "asset.eth.specific" in ids
    assert "asset.unknowncoin.specific" not in ids  # unknown names never guessed
    assert all(q["question_hash"] for q in bank["questions"]) and bank["bank_hash"]


def test_legacy_pairs_shape(tmp_path):
    assets = _assets(tmp_path / "pairs.json", {"pairs": ["btc_jpy"]})
    bank = questions.load_bank("examples/questions.yaml", assets_path=assets)
    ids = {q["question_id"] for q in bank["questions"]}
    assert "asset.btc.specific" in ids


def test_revision_on_rewording():
    h1 = questions._qid_hash("noul", "is it up?", None)
    h2 = questions._qid_hash("noul", "is it UP?", None)
    assert h1 != h2


def test_no_network_lookup():
    bank = questions.load_bank("examples/questions.yaml", assets_path=None)
    assert not any(q["question_id"].startswith("asset.") for q in bank["questions"])
