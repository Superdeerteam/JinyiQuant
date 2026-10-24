"""Behaviour checks for the capacity-study gate decision."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def decision_module():
    spec = importlib.util.spec_from_file_location(
        "decide_capacity_20261024", ROOT / "scripts/decide_capacity_20261024.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prereg_stub():
    return {
        "study_id": "test-study",
        "reference": {"version": "v2_60d_csz", "steps": 9000, "epochs": 0.6007,
                      "weights_sha256": "0" * 64, "hidden_dim": 40},
        "reused_arm": {"hidden_dim": 40},
        "pre_registered_gates": {
            "gates": [
                {"id": "rank_ic_mean_improvement", "metric": "rank_ic", "statistic": "pooled mean",
                 "op": "gte", "value": 0.01},
                {"id": "long_short_mean_degradation", "metric": "long_short_return", "statistic": "pooled mean",
                 "op": "gte", "value": -0.0001},
                {"id": "turnover_degradation", "metric": "turnover", "statistic": "pooled mean",
                 "op": "lte", "value": 0.03},
                {"id": "worst_day_long_short", "metric": "long_short_return",
                 "statistic": "minimum daily long_short_return", "op": "gte", "value": -0.00013896761},
            ],
            "bootstrap_gate": {"applies_to": ["rank_ic", "long_short_return", "turnover"]},
        },
    }


def comparison_stub(arm, rank_ic, long_short, turnover, worst=None, cis=None):
    cis = cis or {"rank_ic": (0.02, 0.03), "long_short_return": (0.0001, 0.0002), "turnover": (-0.05, -0.01)}
    rows = []
    for metric, delta in (("rank_ic", rank_ic), ("long_short_return", long_short), ("turnover", turnover)):
        low, high = cis[metric]
        rows.append({"pair": f"{arm}-minus-v2_60d_csz", "metric": metric, "days": 65,
                     "mean_delta": delta, "block5_ci_low": low, "block5_ci_high": high})
    worst = worst if worst is not None else -0.00005
    return {"paired_deltas": {"pooled": rows},
            "pooled_metrics": {arm: {"min_daily_long_short_return": worst,
                                     "worst_day": "2024-03-07", "worst_day_window": "mar2024"}}}


def arm(arm_name="w80_9000", group="capacity", hidden_dim=80, max_steps=9000, share=0.6007):
    return {"arm": arm_name, "group": group, "hidden_dim": hidden_dim, "max_steps": max_steps,
            "expected_epoch_share": share, "expected_samples_seen": 288000}


def test_compare_uses_the_frozen_operators_only():
    module = decision_module()
    assert module.compare(0.01, "gte", 0.01) is True
    assert module.compare(0.0099, "gte", 0.01) is False
    assert module.compare(0.03, "lte", 0.03) is True
    assert module.compare(-0.0001, "gte", -0.0001) is True
    with pytest.raises(ValueError, match="unsupported gate operator"):
        module.compare(0.0, "gt", 0.0)


def test_bootstrap_gate_needs_the_favourable_side_of_zero():
    module = decision_module()
    passing = comparison_stub("w80_9000", 0.02, 0.0002, -0.01)
    results = module.bootstrap_gates(passing, "w80_9000", "v2_60d_csz",
                                     {"applies_to": ["rank_ic", "long_short_return", "turnover"]})
    assert [row["passed"] for row in results] == [True, True, True]
    straddling = comparison_stub("w80_9000", 0.02, 0.0002, -0.01,
                                 cis={"rank_ic": (-0.01, 0.03), "long_short_return": (-0.0001, 0.0002),
                                      "turnover": (-0.05, 0.01)})
    results = module.bootstrap_gates(straddling, "w80_9000", "v2_60d_csz",
                                     {"applies_to": ["rank_ic", "long_short_return", "turnover"]})
    assert [row["passed"] for row in results] == [False, False, False]
    assert all(row["excludes_zero"] is False for row in results)


def test_arm_decision_names_failed_gates_and_registers_the_compared_factor():
    module = decision_module()
    prereg = prereg_stub()
    reference = {**prereg["reference"], "hidden_dim": prereg["reused_arm"]["hidden_dim"]}

    passed = module.arm_decision(comparison_stub("w80_9000", 0.02, 0.0002, -0.01),
                                 prereg, arm(), reference)
    assert passed["passed"] is True and passed["failed_gates"] == []
    assert passed["changed_vs_v2"] == ["capacity"]

    failed = module.arm_decision(comparison_stub("w80_9000", -0.002, -0.000005, 0.045, worst=-0.000225,
                                                 cis={"rank_ic": (-0.017, 0.004),
                                                      "long_short_return": (-0.000124, 0.000076),
                                                      "turnover": (0.022, 0.072)}),
                                 prereg, arm(), reference)
    assert failed["passed"] is False
    assert failed["failed_gates"] == ["rank_ic_mean_improvement", "turnover_degradation", "worst_day_long_short",
                                      "bootstrap_rank_ic", "bootstrap_long_short_return", "bootstrap_turnover"]
    assert failed["exit_rule_applies"] is True

    volume = module.arm_decision(comparison_stub("w40_full2epoch", 0.02, 0.0002, -0.01), prereg,
                                 arm("w40_full2epoch", "training_volume", 40, 29966, 2.0), reference)
    assert volume["changed_vs_v2"] == ["training_epochs"]


def test_no_passing_version_hands_the_slot_back_and_skips_step3():
    module = decision_module()
    failed = {"arm": "w80_9000", "passed": False, "pooled_rank_ic_delta": -0.002,
              "pooled_long_short_delta": -0.000005, "hidden_dim": 80,
              "numeric_gates": [{"id": "turnover_degradation", "observed": 0.045}], "group": "capacity"}
    winner = module.select_winner([failed])
    assert winner["strongest_passing_version"] is None
    assert winner["closest_non_passing"] == "w80_9000"
    step3 = module.step3_decision([failed], winner)
    assert step3["triggered"] is False and step3["winner_if_triggered"] is None

    passing = {"arm": "w80_9000", "passed": True, "pooled_rank_ic_delta": 0.02,
               "pooled_long_short_delta": 0.0002, "hidden_dim": 80,
               "numeric_gates": [{"id": "turnover_degradation", "observed": -0.01}], "group": "capacity"}
    winner = module.select_winner([failed, passing])
    assert winner["strongest_passing_version"] == "w80_9000"
    step3 = module.step3_decision([failed, passing], winner)
    assert step3["triggered"] is True and step3["winner_if_triggered"] == "w80_9000"
