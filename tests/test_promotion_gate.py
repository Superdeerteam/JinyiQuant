"""Pre-registered promotion gate: ordering, basis re-derivation, decision rule.

The fixtures build a coherent pre-registration by hand - measured basis artifact,
pre-registration JSON, registration record, periods, baseline and candidate
object archives - so each test can break exactly one link and check that the gate
refuses instead of deciding.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jinyiquant.evaluation import EVALUATION_VERSION, aggregate_metric
from scripts.promotion_gate import main

REGISTERED_AT = "2026-10-27T00:00:00+00:00"
BASIS_CREATED_AT = "2026-10-26T00:00:00+00:00"
BASELINE_STARTED_AT = "2026-10-01T00:00:00+00:00"
CANDIDATE_STARTED_AT = "2026-10-27T01:00:00+00:00"
PERIODS = [["2023-11-01", "2023-11-30"], ["2024-01-02", "2024-01-31"]]
DATES = [day.strftime("%Y-%m-%d") for day in
         list(pd.bdate_range("2023-11-01", "2023-11-30")) + list(pd.bdate_range("2024-01-02", "2024-01-31"))]
BASELINE_LEVEL = 0.07
PLACEBO_SCALE = 0.003
RELATIVE_FLOOR = 0.02
PRIMARY_GRID = 1e-4
SECONDARY_SCALES = {"ic_mean": 0.0022, "ic_ir": 0.15, "long_short_sharpe": 1.2,
                    "long_short_return_mean": 3e-5, "min_daily_long_short_return": 4e-5,
                    "turnover_mean": 0.012}
NET5_SCALE = 2e-4
ORIGINAL_APPLIES_TO = "training run and submission inference package, host RSS across all subprocesses"
NEW_APPLIES_TO = ("platform inference/submission path only (DAI data read, model load, full-interval "
                  "inference and score write-out inside the submission notebook); training memory is "
                  "reported only and never gated")


def _round_up(grid: float, value: float) -> float:
    return float(np.ceil(value / grid - 1e-12) * grid)


def _expected_thresholds() -> dict:
    raw = max(PLACEBO_SCALE, RELATIVE_FLOOR * BASELINE_LEVEL)
    return {
        "primary_metric": "rank_ic_mean",
        "relative_floor": RELATIVE_FLOOR,
        "min_improvement": _round_up(PRIMARY_GRID, raw),
        "secondary_limits": {metric: _round_up(1e-6, value) for metric, value in SECONDARY_SCALES.items()},
        "net5_limit": _round_up(1e-7, NET5_SCALE),
    }


def _daily(rank_ic: np.ndarray, dates: list[str]) -> pd.DataFrame:
    rank_ic = np.asarray(rank_ic, dtype=float)
    wave = np.cos(np.arange(len(rank_ic)) * 0.7)
    spread = rank_ic * 0.004 + 1e-5 * wave
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "rank_ic": rank_ic,
        "ic": rank_ic * 1.01 + 1e-4 * wave,
        "long_short_return": spread,
        "turnover": 1.1 + 0.01 * wave,
        "net_return": spread - 5e-4,
    })


def _metrics(daily: pd.DataFrame) -> dict:
    return {
        "evaluation_version": EVALUATION_VERSION,
        "rank_ic_mean": aggregate_metric(daily["rank_ic"].to_numpy(float), "mean", 252.0),
        "rank_ic_ir": aggregate_metric(daily["rank_ic"].to_numpy(float), "ir", 252.0),
        "ic_mean": aggregate_metric(daily["ic"].to_numpy(float), "mean", 252.0),
        "ic_ir": aggregate_metric(daily["ic"].to_numpy(float), "ir", 252.0),
        "long_short_sharpe": aggregate_metric(daily["long_short_return"].to_numpy(float), "sharpe", 252.0),
        "long_short_return_mean": aggregate_metric(daily["long_short_return"].to_numpy(float), "mean", 252.0),
        "min_daily_long_short_return": aggregate_metric(daily["long_short_return"].to_numpy(float), "min", 252.0),
        "turnover_mean": aggregate_metric(daily["turnover"].to_numpy(float), "mean", 252.0),
        "label_overlap": {"overlapping_pair_count": 0,
                          "overlapping_labels_treated_as_independent_trades": False},
        "economic_metrics": {"status": "computed",
                             "overlapping_labels_treated_as_independent_trades": False,
                             "returns_compounded": False,
                             "daily_returns_summed_across_days": False},
    }


def _write_json(path: Path, document: dict) -> str:
    payload = (json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_object(directory: Path, object_id: str, daily: pd.DataFrame,
                 started_at: str = CANDIDATE_STARTED_AT) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    daily.to_csv(directory / "daily_0bp.csv", index=False)
    daily.to_csv(directory / "daily_5bp.csv", index=False)
    _write_json(directory / "metrics.json", {
        "schema_version": 1, "object_id": object_id, "evaluation_version": EVALUATION_VERSION,
        "metrics_0bp": _metrics(daily), "metrics_5bp": _metrics(daily)})
    _write_json(directory / "manifest.json", {"schema_version": 1, "started_at": started_at})
    _write_json(directory / "change.json", {"factor": "model.epochs", "baseline": 1, "candidate": 2,
                                            "hypothesis": "unit test candidate"})
    return directory


def write_basis(directory: Path, *, min_improvement: float | None = None) -> tuple[Path, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    derived = _expected_thresholds()
    if min_improvement is not None:
        derived["min_improvement"] = min_improvement
    measurements = {
        "estimator": {"primary_grid": PRIMARY_GRID, "relative_floor": RELATIVE_FLOOR,
                      "secondary_grids": {metric: 1e-6 for metric in SECONDARY_SCALES},
                      "net5_grid": 1e-7},
        "levels": {"nine_blend": {"rank_ic_mean": BASELINE_LEVEL}},
        "placebo_scale": {"rank_ic_mean": PLACEBO_SCALE, **SECONDARY_SCALES,
                          "net_return_mean_5bp": NET5_SCALE},
        "derived_thresholds": derived | {"baseline_primary_level": BASELINE_LEVEL},
        "days": len(DATES),
        "periods": ["nov2023", "jan2024"],
    }
    path = directory / "volatility.json"
    _write_json(path, {"schema_version": 1, "status": "completed", "created_at": BASIS_CREATED_AT,
                       "evaluation_version": EVALUATION_VERSION, "baseline_object": "nine_blend",
                       "measurements": measurements})
    return path, derived


def write_prereg(root: Path, basis: Path, derived: dict, baseline: Path, *,
                 registered_at: str = REGISTERED_AT, min_improvement: float | None = None,
                 periods: list[list[str]] | None = None) -> tuple[Path, Path]:
    thresholds = _expected_thresholds()
    policy = {
        "primary_metric": "rank_ic_mean",
        "direction": "maximize",
        "min_improvement": thresholds["min_improvement"] if min_improvement is None else min_improvement,
        "max_metric_regressions": dict(thresholds["secondary_limits"]),
        "bootstrap_samples": 400,
        "block_days": 2,
        "confidence": 0.9,
        "seed": 11,
        "threshold_rationale": "measured placebo scale and the relative floor",
        "change": {"factor": "model.epochs", "baseline": 1, "candidate": 2, "hypothesis": "unit test"},
        "threshold_basis": {"calibration_report": str(basis), "calibration_sha256": _sha(basis),
                            "practical_improvement": 0.001, "noise_multiplier": 1.0},
    }
    prereg = {
        "schema_version": 1,
        "registration_id": "b4-gate-unit-test",
        "registered_at": registered_at,
        "threshold_basis": policy["threshold_basis"],
        "object_policy_template": policy,
        "derived_thresholds": {"primary_metric": "rank_ic_mean",
                               "min_improvement": policy["min_improvement"],
                               "secondary_limits": derived["secondary_limits"],
                               "net5_limit": derived["net5_limit"]},
        "net5_limit": derived["net5_limit"],
        "baseline_object": {"path": str(baseline)},
        "periods": {"declared": periods or PERIODS},
        "resource_budget": {"platform_ram_limit_mib": 24576, "headroom_fraction": 0.15,
                            "promotion_peak_rss_mib": 20889,
                            "applies_to": ORIGINAL_APPLIES_TO},
        "evidence_rules": {"date_evidence_unit": "consecutive trading-day blocks inside a period",
                           "seeds_are_not_date_evidence": True,
                           "stocks_within_a_day_are_not_resampled": True,
                           "minimum_periods": 2, "minimum_days": 40,
                           "required_positive_fraction": 0.75},
        "supersedes": {"retired_thresholds": [{"name": "min_rank_ic_gain", "value": 0.0046}],
                       "usable_for_promotion": False},
        "economic_risk_boundary": {
            "stress_single_side_rate_bp": 5.0, "stress_scenario_locked": True,
            "facts": {"turnover_local": 1.10195, "turnover_required_for_zero_net_at_5bp": 0.52912},
            "collapse_rule": "a turnover gain that collapses the spread is not economical at 5bp",
            "must_not": ["relax the 5bp stress scenario"],
        },
        "metric_caliber_layers": {
            "official_turnover": "cross-submission rank percentile; not computable locally",
            "local_proxy": {"definition": "U / 2", "value": 1.10195, "theoretical_max": 2.0,
                            "use": "monotone directional evidence only"},
            "cost_scenario_independence": "turnover is rate independent",
        },
        "pre_registration_completeness": {
            "improvement_amount": {"metric": "rank_ic_mean", "value": 0.0034},
            "secondary_degradation_lines": {"metrics": ["ic_mean", "turnover_mean"]},
            "resource_budget": {"platform_ram_limit_mib": 24576, "promotion_peak_rss_mib": 20889},
        },
    }
    prereg_path = root / "prereg.json"
    _write_json(prereg_path, prereg)
    registration_path = root / "registration.json"
    _write_json(registration_path, {"schema_version": 1, "registration_id": "b4-gate-unit-test",
                                    "prereg_sha256": _sha(prereg_path),
                                    "registered_at": registered_at})
    return prereg_path, registration_path


def write_scope_patch(root: Path, prereg_path: Path, registration_path: Path, *,
                      created_at: str = "2026-10-27T00:30:00+00:00",
                      limit_mib: int = 20889, min_improvement: float | None = None,
                      resource_gate: dict | None = None,
                      new_applies_to: str = NEW_APPLIES_TO) -> dict:
    """Append a scope patch to the registration record; it may never move a number."""
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    assert not registration.get("scope_patches"), "registration already carries a scope patch"
    original_sha = _sha(registration_path)
    policy = prereg["object_policy_template"]
    document = {
        "schema_version": 1, "patch_id": "unit-test-scope-patch", "kind": "scope_patch",
        "created_at": created_at,
        "patched_registration": {
            "registration_id": registration["registration_id"],
            "registered_at": registration["registered_at"],
            "prereg_path": str(prereg_path), "prereg_sha256": _sha(prereg_path),
            "registration_path": str(registration_path),
            "registration_sha256": original_sha},
        "supersedes": {"resource_budget.applies_to": prereg["resource_budget"]["applies_to"]},
        "sets": {"resource_budget.applies_to": new_applies_to,
                 "resource_budget.training_memory": "report_only",
                 "resource_gate": resource_gate or {"training": "report_only",
                                                    "inference_submission": "gated"}},
        "scope": {"gated": {"artifact_kind": "inference_submission", "limit_mib": limit_mib},
                  "report_only": {"artifact_kind": "training", "path": "training run"}},
        "unchanged": {
            "platform_ram_limit_mib": prereg["resource_budget"]["platform_ram_limit_mib"],
            "headroom_fraction": prereg["resource_budget"]["headroom_fraction"],
            "promotion_peak_rss_mib": limit_mib,
            "primary_metric": policy["primary_metric"],
            "min_improvement": policy["min_improvement"] if min_improvement is None else min_improvement,
            "max_metric_regressions": policy["max_metric_regressions"],
            "net5_limit": prereg["net5_limit"],
            "bootstrap": {key: policy[key] for key in ("bootstrap_samples", "block_days",
                                                       "confidence", "seed")},
            "threshold_basis_sha256": policy["threshold_basis"]["calibration_sha256"]},
        "direction": "narrowing a mis-specified applicability, not relaxing a quality threshold",
        "reason": "unit test scope patch",
    }
    amended = dict(registration)
    amended["scope_patches"] = [document]
    amended["amendment_history"] = [{"amended_at": created_at, "change": "unit test scope patch",
                                     "previous_registration_sha256": original_sha,
                                     "previous_registration": registration}]
    _write_json(registration_path, amended)
    assert root  # keep the helper signature explicit about the workspace root
    return {"path": registration_path, "original_sha256": original_sha}


def build_case(tmp_path: Path, *, candidate_shift: float = 0.006, noise: float = 0.0,
               periods: list[list[str]] | None = None,
               candidate_started_at: str = CANDIDATE_STARTED_AT,
               resources: float | None = 18000.0, isolate_primary: bool = False,
               artifact_kind: str = "inference_submission",
               scope_patch: bool = False) -> dict:
    dates = DATES if periods is None else [
        day.strftime("%Y-%m-%d") for period in periods
        for day in pd.bdate_range(period[0], period[1])]
    baseline_daily = _daily(np.full(len(dates), BASELINE_LEVEL), dates)
    shift = np.full(len(dates), candidate_shift)
    if noise:
        noise_values = np.random.default_rng(7).normal(0.0, noise, size=len(dates))
        shift = shift + (noise_values - noise_values.mean())
    candidate_daily = _daily(np.full(len(dates), BASELINE_LEVEL) + shift, dates)
    if isolate_primary:
        for column in ("ic", "long_short_return", "turnover", "net_return"):
            candidate_daily[column] = baseline_daily[column].to_numpy()
    baseline_dir = write_object(tmp_path / "baseline", "nine_blend", baseline_daily, BASELINE_STARTED_AT)
    candidate_dir = write_object(tmp_path / "candidate", "candidate_x", candidate_daily,
                                 candidate_started_at)
    basis, derived = write_basis(tmp_path / "basis")
    prereg_path, registration_path = write_prereg(tmp_path, basis, derived, baseline_dir,
                                                  periods=periods)
    resources_path = None
    if resources is not None:
        resources_path = tmp_path / "resources.json"
        _write_json(resources_path, {"peak_rss_mib": resources, "wall_seconds": 120.0,
                                     "artifact_kind": artifact_kind})
    if scope_patch:
        write_scope_patch(tmp_path, prereg_path, registration_path)
    return {"prereg": prereg_path, "registration": registration_path, "candidate": candidate_dir,
            "baseline": baseline_dir, "resources": resources_path}


def run_gate(case: dict, output: Path) -> int:
    argv = ["--candidate", str(case["candidate"]), "--baseline", str(case["baseline"]),
            "--prereg", str(case["prereg"]), "--registration", str(case["registration"]),
            "--output", str(output)]
    if case["resources"] is not None:
        argv += ["--resources", str(case["resources"])]
    return main(argv)


def decision(output: Path) -> dict:
    return json.loads((output / "decision.json").read_text(encoding="utf-8"))


def test_measured_improvement_is_accepted(tmp_path):
    case = build_case(tmp_path, candidate_shift=0.006)
    code = run_gate(case, tmp_path / "out-accept")
    result = decision(tmp_path / "out-accept")
    assert code == 0
    assert result["status"] == "accept"
    assert result["improvement"] == pytest.approx(0.006, abs=1e-9)
    assert result["min_improvement"] == pytest.approx(_expected_thresholds()["min_improvement"])
    assert result["automatic_promotion"] is False
    gate = json.loads((tmp_path / "out-accept" / "gate.json").read_text(encoding="utf-8"))
    assert gate["ordering"]["prereg_before_candidate"] is True
    assert gate["resources"]["passed"] is True
    assert gate["comparison"]["bootstrap"]["stock_resampling"] is False
    assert "no independent seed replicas" in gate["comparison"]["seed_semantics"]


def test_improvement_below_the_measured_threshold_is_rejected(tmp_path):
    case = build_case(tmp_path, candidate_shift=0.001)
    code = run_gate(case, tmp_path / "out-reject")
    result = decision(tmp_path / "out-reject")
    assert code == 3
    assert result["status"] == "reject"
    assert any("below the measured threshold" in item for item in result["failed_gates"])


def test_noisy_difference_is_inconclusive(tmp_path):
    case = build_case(tmp_path, candidate_shift=0.006, noise=0.12, isolate_primary=True)
    code = run_gate(case, tmp_path / "out-noise")
    result = decision(tmp_path / "out-noise")
    assert code == 4
    assert result["status"] == "inconclusive"
    assert any("does not exclude zero" in item for item in result["missing_evidence"])


def test_gain_concentrated_in_one_period_is_rejected(tmp_path):
    periods = [["2023-11-01", "2023-11-15"], ["2023-11-16", "2023-11-30"],
               ["2024-01-02", "2024-01-31"]]
    case = build_case(tmp_path, candidate_shift=-0.0005, periods=periods)
    daily = pd.read_csv(case["candidate"] / "daily_0bp.csv", parse_dates=["date"])
    first = (daily["date"] < pd.Timestamp("2023-11-16")).to_numpy()
    rank_ic = daily["rank_ic"].to_numpy(dtype=float)
    rank_ic[first] += 0.06
    boosted = _daily(rank_ic, [day.strftime("%Y-%m-%d") for day in daily["date"]])
    case["candidate"] = write_object(tmp_path / "candidate_two", "candidate_x", boosted)
    code = run_gate(case, tmp_path / "out-repeat")
    result = decision(tmp_path / "out-repeat")
    assert code == 3
    assert result["status"] == "reject"
    assert any("repeats in" in item for item in result["failed_gates"])


def test_resource_over_budget_is_rejected(tmp_path):
    case = build_case(tmp_path, candidate_shift=0.006, resources=26000.0)
    code = run_gate(case, tmp_path / "out-memory")
    result = decision(tmp_path / "out-memory")
    assert code == 3
    assert any("peak host RSS" in item for item in result["failed_gates"])


def test_absent_resource_evidence_is_inconclusive(tmp_path):
    case = build_case(tmp_path, candidate_shift=0.006, resources=None)
    code = run_gate(case, tmp_path / "out-noresource")
    result = decision(tmp_path / "out-noresource")
    assert code == 4
    assert any("resource evidence absent" in item for item in result["missing_evidence"])


def test_candidate_started_before_registration_is_refused(tmp_path):
    case = build_case(tmp_path, candidate_shift=0.006,
                      candidate_started_at="2026-10-26T12:00:00+00:00")
    code = run_gate(case, tmp_path / "out-order")
    result = decision(tmp_path / "out-order")
    assert code == 2
    assert result["status"] == "invalid"
    assert "before the pre-registration" in result["refused_reason"]


def test_preregistration_changed_after_registration_is_refused(tmp_path):
    case = build_case(tmp_path, candidate_shift=0.006)
    document = json.loads(case["prereg"].read_text(encoding="utf-8"))
    document["object_policy_template"]["min_improvement"] = 0.0
    _write_json(case["prereg"], document)
    code = run_gate(case, tmp_path / "out-tamper")
    result = decision(tmp_path / "out-tamper")
    assert code == 2
    assert "changed after registration" in result["refused_reason"]


def test_threshold_that_disagrees_with_the_measured_basis_is_refused(tmp_path):
    case = build_case(tmp_path, candidate_shift=0.006)
    basis, derived = write_basis(tmp_path / "basis2")
    prereg_path, registration_path = write_prereg(tmp_path, basis, derived, case["baseline"],
                                                  min_improvement=0.0046)
    case["prereg"], case["registration"] = prereg_path, registration_path
    code = run_gate(case, tmp_path / "out-basis")
    result = decision(tmp_path / "out-basis")
    assert code == 2
    assert "re-derived" in result["refused_reason"]


def test_seed_slots_are_refused_by_the_object_policy(tmp_path):
    case = build_case(tmp_path, candidate_shift=0.006)
    document = json.loads(case["prereg"].read_text(encoding="utf-8"))
    document["object_policy_template"]["min_seeds"] = 3
    _write_json(case["prereg"], document)
    _write_json(case["registration"], {"schema_version": 1, "prereg_sha256": _sha(case["prereg"]),
                                       "registered_at": REGISTERED_AT})
    code = run_gate(case, tmp_path / "out-seeds")
    result = decision(tmp_path / "out-seeds")
    assert code == 2
    assert "policy" in result["refused_reason"]


def _write_run_dir(directory: Path, scores: pd.DataFrame, targets: pd.DataFrame,
                   started_at: str = CANDIDATE_STARTED_AT) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(directory / "predictions.parquet", index=False)
    targets.to_parquet(directory / "targets.parquet", index=False)
    _write_json(directory / "metrics.json", {"schema_version": 1, "evaluation_version": EVALUATION_VERSION})
    _write_json(directory / "manifest.json", {"schema_version": 1, "started_at": started_at})
    _write_json(directory / "change.json", {"factor": "model.epochs", "baseline": 1, "candidate": 2,
                                            "hypothesis": "unit test candidate"})
    return directory


def _panel(dates: list[str], scale: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]
    hours = [int(value[:2]) for value in times]
    minutes = [int(value[3:]) for value in times]
    instruments = [f"{index:06d}.SZ" for index in range(1, 9)]
    rows = []
    for day_index, day in enumerate(dates):
        for time_index, (hour, minute) in enumerate(zip(hours, minutes)):
            stamp = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)
            for instrument_index, instrument in enumerate(instruments):
                target = scale * (np.sin(0.7 * instrument_index + 1.3 * day_index) + 0.3 * time_index)
                rows.append({"date": stamp, "instrument": instrument, "target": target})
    targets = pd.DataFrame(rows)
    return targets, targets[["date", "instrument"]].assign(score=-targets["target"])


def test_run_directory_with_predictions_is_reevaluated_at_both_cost_scenarios(tmp_path):
    case = build_case(tmp_path, candidate_shift=0.006)
    targets, negated = _panel(DATES, 1e-3)
    baseline_run = _write_run_dir(tmp_path / "run_baseline", negated, targets, BASELINE_STARTED_AT)
    candidate_run = _write_run_dir(tmp_path / "run_candidate",
                                   negated.assign(score=-negated["score"]), targets)
    case["baseline"], case["candidate"] = baseline_run, candidate_run
    code = run_gate(case, tmp_path / "out-rundir")
    result = decision(tmp_path / "out-rundir")
    gate = json.loads((tmp_path / "out-rundir" / "gate.json").read_text(encoding="utf-8"))
    assert code == 0
    assert result["status"] == "accept"
    assert gate["objects"]["candidate"]["kind"] == "run_dir_reevaluated_0bp_5bp"
    assert result["improvement"] > 0.5


def test_registered_pre_registration_matches_the_measured_basis():
    from scripts.promotion_gate import DEFAULT_PREREG, DEFAULT_REGISTRATION, verify_basis, verify_registration

    registration = verify_registration(DEFAULT_PREREG, DEFAULT_REGISTRATION)
    prereg = registration["prereg"]
    basis = verify_basis(prereg, registration["registered_at"])
    assert basis["checks"]["min_improvement_recomputed"] == pytest.approx(
        prereg["object_policy_template"]["min_improvement"])
    assert basis["checks"]["min_improvement_recomputed"] == pytest.approx(0.0034)
    assert prereg["supersedes"]["usable_for_promotion"] is False
    retired = {item["value"] for item in prereg["supersedes"]["retired_thresholds"]}
    assert 0.0046 in retired
    assert prereg["object_policy_template"]["min_improvement"] != pytest.approx(0.0046)
    assert basis["report"]["measurements"]["days"] >= prereg["evidence_rules"]["minimum_days"]
    assert basis["report"]["measurements"]["placebo_scale"]["rank_ic_mean"] <= \
        prereg["object_policy_template"]["min_improvement"]
    boundary = prereg["economic_risk_boundary"]
    assert boundary["stress_single_side_rate_bp"] == 5.0
    assert boundary["stress_scenario_locked"] is True
    assert boundary["facts"]["turnover_required_for_zero_net_at_5bp"] == pytest.approx(0.52912)
    assert boundary["facts"]["gross_required_multiple_at_5bp"] == pytest.approx(2.0828)
    assert any("relax the 5bp" in item for item in boundary["must_not"])
    layers = prereg["metric_caliber_layers"]
    assert layers["local_proxy"]["value"] == pytest.approx(1.10195)
    assert "rank percentile" in layers["official_turnover"]
    assert "rate-independent" in layers["cost_scenario_independence"]
    completeness = prereg["pre_registration_completeness"]
    assert completeness["improvement_amount"]["metric"] == "rank_ic_mean"
    assert "turnover_mean" in completeness["secondary_degradation_lines"]["metrics"]
    assert completeness["resource_budget"]["promotion_peak_rss_mib"] == 20889


def test_pre_registration_without_the_economic_boundary_block_is_refused(tmp_path):
    case = build_case(tmp_path, candidate_shift=0.006)
    document = json.loads(case["prereg"].read_text(encoding="utf-8"))
    document.pop("economic_risk_boundary")
    _write_json(case["prereg"], document)
    _write_json(case["registration"], {"schema_version": 1, "prereg_sha256": _sha(case["prereg"]),
                                       "registered_at": REGISTERED_AT})
    code = run_gate(case, tmp_path / "out-blocks")
    result = decision(tmp_path / "out-blocks")
    assert code == 2
    assert "economic_risk_boundary" in result["refused_reason"]


def test_training_artifact_over_the_inference_line_is_not_rejected(tmp_path):
    case = build_case(tmp_path, candidate_shift=0.006, resources=22000.0,
                      artifact_kind="training", scope_patch=True)
    code = run_gate(case, tmp_path / "out-training")
    result = decision(tmp_path / "out-training")
    gate = json.loads((tmp_path / "out-training" / "gate.json").read_text(encoding="utf-8"))
    assert code == 0
    assert result["status"] == "accept"
    assert result["failed_gates"] == []
    assert gate["resources"]["gated"] is False
    assert gate["resources"]["resource_gate_source"] == "scope_patch"
    assert gate["resources"]["peak_rss_mib"] == pytest.approx(22000.0)
    assert gate["resources"]["training_memory"] == "report_only"
    assert result["resource_scope"] == {
        "applies_to": NEW_APPLIES_TO, "artifact_kind": "training", "gated": False,
        "training_memory": "report_only", "scope_patch": gate["scope_patch"]["path"]}


def test_inference_artifact_over_the_line_is_rejected(tmp_path):
    case = build_case(tmp_path, candidate_shift=0.006, resources=22000.0,
                      artifact_kind="inference_submission", scope_patch=True)
    code = run_gate(case, tmp_path / "out-inference")
    result = decision(tmp_path / "out-inference")
    assert code == 3
    assert result["status"] == "reject"
    assert any("inference/submission line 20889" in item for item in result["failed_gates"])
    # A legal inference peak still passes the same case.
    case["resources"] = tmp_path / "resources-legal.json"
    _write_json(case["resources"], {"peak_rss_mib": 20000.0, "artifact_kind": "inference_submission"})
    code = run_gate(case, tmp_path / "out-inference-legal")
    assert code == 0
    assert decision(tmp_path / "out-inference-legal")["status"] == "accept"


def test_scope_patch_narrows_the_resource_gate_without_becoming_a_threshold_registration(tmp_path):
    # The patch is written after the candidate started and it still does not move the
    # ordering anchor: that stays the threshold registration timestamp.
    case = build_case(tmp_path / "ordered", candidate_shift=0.006, resources=22000.0,
                      artifact_kind="training")
    patch_info = write_scope_patch(tmp_path / "ordered", case["prereg"], case["registration"],
                                   created_at="2026-10-27T02:00:00+00:00")
    code = run_gate(case, tmp_path / "out-ordered")
    gate = json.loads((tmp_path / "out-ordered" / "gate.json").read_text(encoding="utf-8"))
    assert code == 0
    assert pd.Timestamp(gate["ordering"]["preregistered_at"]) == pd.Timestamp(REGISTERED_AT)
    assert gate["ordering"]["scope_patch_used_for_ordering"] is False
    assert gate["scope_patch"]["takes_part_in_ordering"] is False
    assert gate["scope_patch"]["created_at"] == "2026-10-27T02:00:00+00:00"
    # The amendment keeps the pre-amendment registration bytes reproducible by hash.
    assert gate["scope_patch"]["pre_amendment_registration_sha256"] == patch_info["original_sha256"]
    assert gate["scope_patch"]["record"] == "registration.scope_patches[0]"

    # Without the patch the registered scope still gates every artifact kind, so the
    # same training peak is rejected: the narrowing really is carried by the patch.
    unpatched = build_case(tmp_path / "unpatched", candidate_shift=0.006, resources=22000.0,
                           artifact_kind="training")
    code = run_gate(unpatched, tmp_path / "out-unpatched")
    gate = json.loads((tmp_path / "out-unpatched" / "gate.json").read_text(encoding="utf-8"))
    assert code == 3
    assert gate["resources"]["resource_gate_source"] == "registered"
    assert gate["resources"]["gated"] is True

    # A patch may not move a threshold, and it may not widen the gate back to training.
    moved = build_case(tmp_path / "moved", candidate_shift=0.006, resources=18000.0)
    write_scope_patch(tmp_path / "moved", moved["prereg"], moved["registration"],
                      min_improvement=0.002)
    code = run_gate(moved, tmp_path / "out-moved")
    assert code == 2
    assert "disagrees with the registered" in decision(tmp_path / "out-moved")["refused_reason"]

    widened = build_case(tmp_path / "widened", candidate_shift=0.006, resources=18000.0)
    write_scope_patch(tmp_path / "widened", widened["prereg"], widened["registration"],
                      resource_gate={"training": "gated", "inference_submission": "gated"})
    code = run_gate(widened, tmp_path / "out-widened")
    assert code == 2
    assert "may only narrow" in decision(tmp_path / "out-widened")["refused_reason"]
