"""Behavioral checks for fixed-compute coverage decisions and the frozen 40-day budget."""

import numpy as np
import pandas as pd
import pytest

from jinyiquant.io import file_sha256, write_json
from modified_tlob.coverage import (
    baseline_seed_noise,
    coverage_decision,
    declared_exposure,
    minimum_mean_gain,
)
from modified_tlob.ensemble import SEEDS, paired_block_intervals
from scripts.run_training_coverage_arms import (
    arm_declaration,
    frozen_limits,
    reuse_measurements,
)

NOISE = {"rank_ic_mean": 0.0116, "long_short_return_mean": 0.000104,
         "turnover_mean": 0.0371, "min_daily_long_short_return": 0.000139}


def test_exposure_shows_equal_compute_with_fewer_pool_epochs():
    exposure = declared_exposure({10: 79920, 40: 319632}, steps=1500, batch_size=32)
    assert exposure["sample_exposures"] == 48000
    assert exposure["arms"]["10"]["epochs"] == pytest.approx(48000 / 79920)
    assert exposure["arms"]["40"]["epochs"] == pytest.approx(48000 / 319632)
    # The fixed budget is only 15% of the 40-day pool, so this is not a test of more history.
    assert exposure["arms"]["40"]["epochs"] < 0.16 < exposure["arms"]["10"]["epochs"]
    assert "not the value of more history" in exposure["interpretation"]
    with pytest.raises(ValueError, match="exactly the 10-day and 40-day arms"):
        declared_exposure({10: 10}, steps=1, batch_size=1)
    with pytest.raises(ValueError, match="positive integers"):
        declared_exposure({10: 10, 40: 40}, steps=0, batch_size=1)


def test_seed_noise_uses_windows_as_strata_and_never_pools_days():
    metrics = {period: {"rank_ic_mean": {str(seed): value for seed, value in
                                        zip(SEEDS, values)}}
               for period, values in {"research_a": [0.01, 0.02, 0.03], "research_b": [0.02, 0.04, 0.06]}.items()}
    result = baseline_seed_noise(metrics, metrics=("rank_ic_mean",))
    # Sample variances 1e-4 and 4e-4 average to 2.5e-4; sqrt(2.5e-4) is the declared noise unit.
    assert result["noise"]["rank_ic_mean"] == pytest.approx(np.sqrt(2.5e-4))
    assert result["per_window"]["research_b"]["rank_ic_mean"]["seed_std"] == pytest.approx(np.sqrt(4e-4))
    assert result["per_window"]["research_a"]["rank_ic_mean"]["spread"] == pytest.approx(0.02)
    with pytest.raises(ValueError, match="at least two baseline windows"):
        baseline_seed_noise({"research_a": metrics["research_a"]}, metrics=("rank_ic_mean",))
    broken = {period: {"rank_ic_mean": dict(values["rank_ic_mean"])} for period, values in metrics.items()}
    broken["research_a"]["rank_ic_mean"].pop("47")
    with pytest.raises(ValueError, match="exactly seeds"):
        baseline_seed_noise(broken, metrics=("rank_ic_mean",))
    flat = {period: {"rank_ic_mean": {str(seed): 0.01 for seed in SEEDS}} for period in metrics}
    with pytest.raises(ValueError, match="must be positive"):
        baseline_seed_noise(flat, metrics=("rank_ic_mean",))


def test_materiality_floor_is_the_larger_of_practical_and_measured_noise():
    noise = {"noise": {"rank_ic_mean": 0.0116}}
    assert minimum_mean_gain(noise, practical_improvement=0.002, noise_multiplier=1.0)[
        "minimum_mean_gain"] == pytest.approx(0.0116)
    assert minimum_mean_gain(noise, practical_improvement=0.02, noise_multiplier=1.0)[
        "minimum_mean_gain"] == pytest.approx(0.02)
    with pytest.raises(ValueError, match="positive finite"):
        minimum_mean_gain(noise, practical_improvement=0.0, noise_multiplier=1.0)
    with pytest.raises(ValueError, match="primary metric"):
        minimum_mean_gain({"noise": {}}, practical_improvement=0.002, noise_multiplier=1.0)


def paired_frame(rank_ic=0.02, long_short=0.00005, turnover=-0.01):
    days = np.arange(10)
    return pd.DataFrame({"period": ["research_a"] * 10 + ["research_b"] * 10,
                         "date": list(pd.date_range("2023-11-01", periods=10)) +
                                 list(pd.date_range("2024-03-01", periods=10)),
                         "rank_ic": rank_ic, "long_short_return": long_short, "turnover": turnover,
                         "_day": np.concatenate([days, days])})


def decision_for(paired, *, summaries=None, cost=True, minimum=0.0116):
    intervals = [paired_block_intervals(paired.drop(columns="_day"), block_days=block, samples=200)
                 for block in (5, 2)]
    rows = summaries or [{"period": period, "worst_day_delta": 0.0}
                         for period in ("research_a", "research_b")]
    exposure = declared_exposure({10: 79920, 40: 319632}, steps=1500, batch_size=32)
    return coverage_decision(paired.drop(columns="_day"), rows, intervals,
                             {"minimum_mean_gain": minimum, "noise": NOISE}, cost_measured=cost,
                             exposure=exposure)


def test_decision_accepts_only_with_gain_interval_and_cost():
    result = decision_for(paired_frame())
    assert result["result"] == "accept"
    assert result["action"] == "retain_40_day_candidate_for_next_stage"
    assert result["checks"]["minimum_mean_gain"] and result["checks"]["both_windows_improve"]
    assert all(row["positive_rank_ic_lower"] for row in result["interval_checks"])
    assert result["exposure"]["sample_exposures"] == 48000
    assert result["model_promoted"] is False and result["final_views"] == 0


@pytest.mark.parametrize("failure", ["gain", "window", "gross", "turnover", "worst", "cost"])
def test_point_gates_reject_without_search(failure):
    paired, summaries, cost = paired_frame(), None, True
    if failure == "gain":
        paired["rank_ic"] = 0.005
    elif failure == "window":
        paired.loc[paired.period == "research_a", "rank_ic"] = -0.001
    elif failure == "gross":
        paired["long_short_return"] = -0.00001
    elif failure == "turnover":
        paired["turnover"] = 0.004
    elif failure == "worst":
        summaries = [{"period": "research_a", "worst_day_delta": -0.0002},
                     {"period": "research_b", "worst_day_delta": 0.0}]
    else:
        cost = False
    result = decision_for(paired, summaries=summaries, cost=cost)
    assert result["result"] == ("inconclusive" if failure == "cost" else "reject")
    assert result["action"] == "retain_10_day_reference"
    assert not result["model_promoted"]
    # A rejected gain claim must never be reported as evidence against longer history.
    assert "not the value of more history" in result["interpretation"]


def test_positive_mean_with_uncertain_blocks_is_inconclusive():
    paired = paired_frame()
    paired["rank_ic"] = np.tile([-0.02] * 5 + [0.06] * 5, 2)
    assert decision_for(paired)["result"] == "inconclusive"
    with pytest.raises(ValueError, match="complete"):
        decision_for(paired.iloc[:19])


def test_frozen_limits_round_measured_peaks_and_require_both_windows():
    measurements = [{"window": "research_a", "arm": "40", "process_peak_rss_mb": 5000.0,
                     "preprocessing_seconds": 400.0, "seconds_per_step": 0.08},
                    {"window": "research_b", "arm": "40", "process_peak_rss_mb": 5200.0,
                     "preprocessing_seconds": 420.0, "seconds_per_step": 0.09},
                    {"window": "research_a", "arm": "10", "process_peak_rss_mb": 2800.0,
                     "preprocessing_seconds": 100.0, "seconds_per_step": 0.09}]
    limits = frozen_limits(measurements, training_steps=1500)
    assert limits["max_rss_mb"] == 6656.0  # round_up_256(1.25 * 5200)
    assert limits["wall_seconds"] == 1200.0  # round_up_300(1.25 * (420 + 1500 * 0.09) + 300)
    assert limits["basis"]["measured_peak_rss_mb"] == 5200.
    assert limits["basis"]["declared_samples_seen"] == 48000
    measured = limits["basis"]["measured_preprocessing_seconds"]
    assert measured == 420. and limits["basis"]["windows_measured"] == ["research_a", "research_b"]
    with pytest.raises(ValueError, match="Both 40-day windows"):
        frozen_limits(measurements[:1], training_steps=1500)


def test_reused_measurements_must_match_the_frozen_training_sets(tmp_path):
    dates = pd.to_datetime(["2023-10-31 09:30"])
    selection = []
    for window in ("research_a", "research_b"):
        for count in (10, 40):
            artifact = tmp_path / f"{window}_{count}.parquet"
            pd.DataFrame({"date": dates, "instrument": ["A"], "target": [0.01],
                          "label_end": dates + pd.Timedelta(minutes=30)}).to_parquet(artifact, index=False)
            selection.append({"window": window, "training_days": count, "artifact": str(artifact),
                              "sha256": file_sha256(artifact)})
    frozen = {"training_selection": selection}
    measurements = []
    for item in selection:
        declaration = arm_declaration(frozen, item["window"], str(item["training_days"]))
        identity = {key: declaration[key] for key in ("samples", "keys_sha256", "labels_sha256")}
        measurements.append({"window": item["window"], "arm": str(item["training_days"]),
                             "identity": identity, "frozen_identity": identity})
    write_json(tmp_path / "measurements.json", {"measurements": measurements})
    reused = reuse_measurements(tmp_path, frozen)
    assert len(reused["measurements"]) == 4 and len(reused["measurements_sha256"]) == 64
    tampered = [dict(item) for item in measurements]
    tampered[0]["identity"] = {**tampered[0]["identity"], "keys_sha256": "0" * 64}
    write_json(tmp_path / "measurements.json", {"measurements": tampered})
    with pytest.raises(ValueError, match="different training set"):
        reuse_measurements(tmp_path, frozen)
    write_json(tmp_path / "measurements.json", {"measurements": measurements[:3]})
    with pytest.raises(ValueError, match="both windows and both arms"):
        reuse_measurements(tmp_path, frozen)


def test_arm_declaration_binds_the_frozen_artifact(tmp_path):
    dates = pd.to_datetime(["2023-10-31 09:30", "2023-10-31 10:00"])
    artifact = tmp_path / "research_a_10_training_targets.parquet"
    pd.DataFrame({"date": dates, "instrument": ["A", "B"], "target": [0.01, 0.02],
                  "label_end": dates + pd.Timedelta(minutes=30)}).to_parquet(artifact, index=False)
    frozen = {"training_selection": [{"window": "research_a", "training_days": 10,
                                      "artifact": str(artifact), "sha256": file_sha256(artifact)}]}
    declaration = arm_declaration(frozen, "research_a", "10")
    assert declaration["samples"] == 2
    assert len(declaration["keys_sha256"]) == 64 and len(declaration["labels_sha256"]) == 64
    pd.DataFrame({"date": dates, "instrument": ["A", "C"], "target": [0.01, 0.02],
                  "label_end": dates + pd.Timedelta(minutes=30)}).to_parquet(artifact, index=False)
    with pytest.raises(ValueError, match="artifact changed"):
        arm_declaration(frozen, "research_a", "10")
    write_json(tmp_path / "empty.json", {})
    with pytest.raises(ValueError, match="no research_a 40-day"):
        arm_declaration(frozen, "research_a", "40")
