from intel_net.probe import evaluate, fixtures
import pytest
from intel_net.acceptance import compare_answers


def test_proposal_requires_sample_counts_and_no_keyword_negative_admission():
    rows = [{"truth": True, "p_yes": .9, "keyword_free": False} for _ in range(50)]
    rows += [{"truth": False, "p_yes": .2, "keyword_free": True} for _ in range(200)]
    assert evaluate(rows)["pass"]
    rows[-1]["p_yes"] = .7
    result = evaluate(rows)
    assert result["false_positive_rate"] == .005 and not result["pass"]
    rows[-1]["p_yes"] = float("nan")
    assert evaluate(rows)["invalid"] == 1 and not evaluate(rows)["pass"]


def test_fixtures_use_explicit_rules_no_model_labels():
    bank = {"questions": [{"question_id": "asset.btc.specific", "subject": "btc"}]}
    data = fixtures(bank, names={"btc": "Bitcoin"})
    assert len(data) == 7
    assert sum(row["truth"] for row in data) == 2
    assert all(row.get("truth_rule") for row in data)
    assert all("BTC" in row["state"] for row in data if row["truth"])
    assert all("Bitcoin" in row["state"] for row in data if row["truth"])
    assert all("BTC" not in row["state"] for row in data if not row["truth"])


def test_exact_acceptance_rejects_tiny_changes_and_missing_answers():
    response = {"model": "d1", "answers": {"q": {"type": "noul", "noul": .9}}}
    compare_answers(response, response)
    with pytest.raises(ValueError):
        compare_answers(response, {"model": "d1", "answers": {"q": {"type": "noul", "noul": .900000001}}})
    with pytest.raises(ValueError):
        compare_answers({"answers": {}}, {"answers": {}})
