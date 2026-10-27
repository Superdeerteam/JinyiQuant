"""Deterministic-object comparison: paired daily evidence without fabricated seeds."""

import numpy as np
import pandas as pd
import pytest

from jinyiquant.comparison import (
    compare_evaluation_objects,
    validate_object_policy,
)
from jinyiquant.evaluation import EVALUATION_VERSION

CHANGE = {"factor": "model.epochs", "baseline": 1, "candidate": 2, "hypothesis": "unit test"}
BASIS = {"calibration_report": "reports/unit-test.json", "calibration_sha256": "0" * 64,
         "practical_improvement": 0.001, "noise_multiplier": 2.0}


def policy(**overrides):
    value = {
        "primary_metric": "rank_ic_mean", "direction": "maximize", "min_improvement": 0.001,
        "max_metric_regressions": {"turnover_mean": 0.05},
        "bootstrap_samples": 200, "block_days": 5, "confidence": 0.9, "seed": 7,
        "threshold_rationale": "unit test threshold", "change": CHANGE, "threshold_basis": BASIS,
    }
    value.update(overrides)
    return value


def evaluation_object(object_id, daily_rank_ic, *, turnover=0.5, version=EVALUATION_VERSION,
                      economic_status="computed"):
    values = np.asarray(daily_rank_ic, dtype=float)
    dates = pd.bdate_range("2023-11-01", periods=len(values))
    return {
        "object_id": object_id,
        "metrics": {"evaluation_version": version, "rank_ic_mean": float(values.mean()),
                    "turnover_mean": turnover,
                    "label_overlap": {"overlapping_pair_count": 0},
                    "economic_metrics": {"status": economic_status}},
        "daily": pd.DataFrame({"date": dates, "rank_ic": values}),
    }


def test_seed_grid_keys_are_rejected_instead_of_ignored():
    with pytest.raises(ValueError, match="policy keys"):
        validate_object_policy({**policy(), "min_seeds": 2})


def test_identical_objects_cannot_clear_a_practical_threshold():
    baseline = evaluation_object("baseline", np.full(20, 0.05))
    candidate = evaluation_object("candidate", np.full(20, 0.05))
    comparison, decision = compare_evaluation_objects(baseline, candidate, policy())
    assert comparison["object_mode"] == "deterministic_aggregate"
    assert "no independent seed replicas" in comparison["seed_semantics"]
    assert comparison["improvement"] == pytest.approx(0.0)
    assert decision["status"] == "reject"
    assert "below preregistered practical threshold" in " ".join(decision["failed_gates"])


def test_consistent_improvement_is_accepted_with_a_strict_interval():
    baseline = evaluation_object("baseline", np.full(20, 0.05))
    candidate = evaluation_object("candidate", np.full(20, 0.07))
    comparison, decision = compare_evaluation_objects(baseline, candidate, policy())
    assert comparison["improvement"] == pytest.approx(0.02)
    assert comparison["confidence_interval"]["low"] == pytest.approx(0.02)
    assert comparison["confidence_interval"]["high"] == pytest.approx(0.02)
    assert decision["status"] == "accept"
    assert decision["automatic_promotion"] is False


def test_noisy_difference_is_inconclusive_not_accept():
    pattern = np.array([0.02, -0.015] * 10)
    baseline = evaluation_object("baseline", np.full(20, 0.05))
    candidate = evaluation_object("candidate", np.full(20, 0.05) + pattern)
    comparison, decision = compare_evaluation_objects(baseline, candidate, policy())
    assert comparison["improvement"] == pytest.approx(pattern.mean())
    assert comparison["confidence_interval"]["low"] < 0
    assert decision["status"] == "inconclusive"
    assert "confidence interval does not establish" in " ".join(decision["missing_evidence"])


def test_mixed_or_foreign_evaluation_versions_raise():
    baseline = evaluation_object("baseline", np.full(20, 0.05))
    with pytest.raises(ValueError, match="mixes evaluation versions"):
        compare_evaluation_objects(baseline, evaluation_object("candidate", np.full(20, 0.06), version="eval-v1"),
                                   policy())
    with pytest.raises(ValueError, match="must declare evaluation_version"):
        foreign = evaluation_object("candidate", np.full(20, 0.06))
        foreign["metrics"].pop("evaluation_version")
        compare_evaluation_objects(baseline, foreign, policy())
    with pytest.raises(ValueError, match="must be evaluated by"):
        stale = evaluation_object("candidate", np.full(20, 0.06), version="eval-v1")
        old = evaluation_object("baseline", np.full(20, 0.05), version="eval-v1")
        compare_evaluation_objects(old, stale, policy())


def test_undefined_economic_status_blocks_the_comparison():
    baseline = evaluation_object("baseline", np.full(20, 0.05))
    candidate = evaluation_object("candidate", np.full(20, 0.07),
                                  economic_status="undefined_single_side_rate_not_configured")
    comparison, decision = compare_evaluation_objects(baseline, candidate, policy())
    assert decision["status"] == "inconclusive"
    assert "undefined_single_side_rate_not_configured" in " ".join(decision["missing_evidence"])
    assert comparison["objects"]["candidate"]["gate"]["economic_usable"] is False


def test_recorded_metric_must_match_the_daily_artifact():
    baseline = evaluation_object("baseline", np.full(20, 0.05))
    candidate = evaluation_object("candidate", np.full(20, 0.07))
    candidate["metrics"]["rank_ic_mean"] = 0.5
    with pytest.raises(ValueError, match="does not match daily artifact"):
        compare_evaluation_objects(baseline, candidate, policy())


def test_days_outside_declared_periods_are_not_silently_dropped():
    baseline = evaluation_object("baseline", np.full(20, 0.05))
    candidate = evaluation_object("candidate", np.full(20, 0.07))
    dates = baseline["daily"]["date"]
    comparison, decision = compare_evaluation_objects(
        baseline, candidate, policy(), periods=[(dates.iloc[0], dates.iloc[9])])
    assert decision["status"] == "inconclusive"
    assert "fall outside the declared periods" in " ".join(decision["missing_evidence"])


def test_identity_control_can_never_promote():
    baseline = evaluation_object("baseline", np.full(20, 0.05))
    candidate = evaluation_object("candidate", np.full(20, 0.07))
    identity = {"factor": "control.identity", "baseline": None, "candidate": None,
                "hypothesis": "identity"}
    comparison, decision = compare_evaluation_objects(baseline, candidate,
                                                      policy(change=identity))
    assert decision["status"] == "reject"
    assert "identity control" in " ".join(decision["failed_gates"])
