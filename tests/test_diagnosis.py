"""Hypotheses follow measured fit, with no unsupported feature/structure claim."""

import pytest

from jinyiquant.diagnosis import learning_findings


def audit(tiny=True, ratios=(90., 70.), reproduced=True):
    return {"passed": True, "findings": {"small_sample_learning_passed": tiny,
        "same_seed_reproduction_passed": reproduced},
        "runs": [{"arm": "correct", "train_mse_to_constant_ratio": v} for v in ratios]}


def test_underfit_recommends_one_optimization_factor_without_claiming_cause():
    result = learning_findings(audit())
    assert result["training_fit"]["all_worse_than_constant"] is True
    assert result["rules"][0]["status"] == "未观察到"
    assert result["next_step"]["factor"] == "model.learning_rate"
    assert result["result"] == "证据不足" and result["model_promoted"] is False
    assert result["rules"][-1]["status"] == "证据不足"


def test_tiny_failure_is_distinguished_from_full_budget_underfit():
    result = learning_findings(audit(tiny=False))
    assert result["rules"][0]["status"] == "已观察到"
    assert "小样本" in result["next_step"]["reason"]


@pytest.mark.parametrize("values", [(), (0.4, 0.6), (None,), (float("nan"),)])
def test_absent_or_better_training_fit_does_not_invent_structure_change(values):
    result = learning_findings(audit(ratios=values))
    assert result["next_step"]["factor"] is None
    assert all(rule["status"] == "证据不足" for rule in result["rules"][1:])


def test_failed_reproduction_prevents_confident_training_diagnosis():
    result = learning_findings(audit(reproduced=False))
    assert result["training_fit"]["all_worse_than_constant"] is None
    assert result["next_step"]["factor"] is None
