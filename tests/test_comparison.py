import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jinyiquant.comparison import _block_indices, compare_runs, validate_policy
from jinyiquant.evaluation import EVALUATION_VERSION, METRIC_SPEC, aggregate_metric, evaluate
from jinyiquant.io import digest, file_sha256

CONTRACT_METRIC_BLOCKS = {
    "evaluation_version": EVALUATION_VERSION,
    "label_overlap": {"overlapping_pair_count": 0, "independent_label_periods": True,
                      "overlapping_labels_treated_as_independent_trades": False},
    "economic_metrics": {
        "status": "computed", "single_side_rate": 0.0,
        "gross_return_mean": 0.003, "cost_mean": 0.0, "net_return_mean": 0.003,
        "turnover_mean": 0.3, "total_abs_weight_change_mean": 0.6,
        "breakeven_single_side_rate": None,
        "overlapping_labels_treated_as_independent_trades": False,
        "returns_compounded": False, "daily_returns_summed_across_days": False,
    },
}


@pytest.fixture
def policy(tmp_path):
    calibration_root = tmp_path / "calibration"
    calibration_root.mkdir()
    calibration_runs = [run_fixture(calibration_root, "baseline", seed, month, {})
                        for month in (1, 2) for seed in (3, 8)]
    for path in calibration_runs:
        edit(path, "manifest", lambda m: m.update(started_at="2025-03-01T07:00:00Z",
                                                 ended_at="2025-03-01T08:00:00Z"))
    summaries = {}
    for metric, (column, method, _) in METRIC_SPEC.items():
        values = [aggregate_metric(pd.read_csv(path / "daily.csv")[column].to_numpy(), method)
                  for path in calibration_runs if column in pd.read_csv(path / "daily.csv")]
        if values:
            summaries[metric] = {"n": len(values), "mean": float(np.mean(values)),
                                 "std": float(np.std(values, ddof=1)), "min": min(values), "max": max(values)}
    report = {"status": "completed", "created_at": "2025-03-01T08:30:00Z",
              "runs": [str(p) for p in calibration_runs], "metrics": summaries}
    save(calibration_root, "calibration", report)
    return {
        "primary_metric": "rank_ic_mean", "direction": "maximize", "min_improvement": 0.015,
        "max_metric_regressions": {"long_short_return_mean": 0.001, "turnover_mean": 0.05},
        "max_wall_ratio": 1.5, "max_memory_ratio": 1.2, "min_seeds": 2, "min_periods": 2,
        "bootstrap_samples": 100, "block_days": 2, "confidence": 0.95, "seed": 8,
        "threshold_rationale": "Fixture: measured baseline variation below 0.01, practical gain 0.015.",
        "change": {"factor": "model.alpha", "baseline": 1, "candidate": 2, "hypothesis": "Higher alpha improves fit"},
        "threshold_basis": {"calibration_report": str(calibration_root / "calibration.json"),
                            "calibration_sha256": digest(report), "practical_improvement": 0.0001,
                            "noise_multiplier": 1.0},
    }


def save(path, name, value):
    (path / f"{name}.json").write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def archive(path):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"] = [{"path": item.name, "bytes": item.stat().st_size, "sha256": file_sha256(item)}
                             for item in sorted(path.iterdir()) if item.name != "manifest.json"]
    save(path, "manifest", manifest)


def run_fixture(root, arm, seed, month, policy, *, improvement=0.04):
    path = root / f"{arm}-{seed}-{month}"
    path.mkdir()
    period = {"start": f"2025-{month:02}-01T00:00:00", "end": f"2025-{month + 1:02}-01T00:00:00"}
    dates = pd.bdate_range(period["start"], periods=12)
    rank_ic = np.array([0.10, 0.12, 0.10, 0.08, 0.11, 0.13, 0.09, 0.11, 0.07, 0.10, 0.09, 0.12])
    if arm == "candidate":
        rank_ic = rank_ic + improvement
    daily = pd.DataFrame({"date": dates, "rank_ic": rank_ic, "ic": rank_ic,
                          "long_short_return": np.repeat(0.003, len(dates)),
                          "turnover": np.repeat(0.3, len(dates)),
                          "sample_count": 100, "cross_section_count": 1,
                          "eligible_rank_ic_count": 1, "valid_rank_ic_count": 1,
                          "eligible_ic_count": 1, "valid_ic_count": 1})
    daily.to_csv(path / "daily.csv", index=False)
    pd.DataFrame({"date": dates, "instrument": "A", "score": rank_ic}).to_parquet(path / "predictions.parquet")
    (path / "source.zip").write_bytes(b"fixture source snapshot")
    config = {"seed": seed, "evaluation_split": "validation", "model": {"alpha": 1 if arm == "baseline" else 2},
              "data": {"version": "demo-v1", "lookback": 30, "bars_path": "input.parquet"},
              "splits": {"train": {"start": "2024-01-01", "end": "2025-01-01"}, "validation": period},
              "evaluation": {"periods_per_year": 252}, "budget": {"wall_seconds": 60, "max_rss_mb": 1024},
              "acceptance": {"repro_atol": 1e-10, "repro_rtol": 1e-8}}
    manifest = {"run_id": path.name, "status": "completed", "started_at": "2025-03-01T10:00:00Z",
                "ended_at": "2025-03-01T10:10:00Z",
                "comparison_policy_registered_at": "2025-03-01T09:00:00Z",
                "comparison_policy_sha256": digest(policy), "resources": {"wall_seconds": 10, "peak_rss_mb": 200},
                "reproducibility": {"passed": True, "reference_run_id": path.name + "-original", "atol": 1e-10, "rtol": 1e-8},
                "config_sha256": digest(config),
                "code": {"content_sha256": digest({"fixture.py": "example"}), "files": {"fixture.py": "example"},
                         "snapshot_sha256": file_sha256(path / "source.zip")}}
    (path / "config.sha256").write_text(digest(config), encoding="utf-8")
    section_hash = digest([date.isoformat() for date in dates])
    coverage = {"eligible_section_count": len(dates), "valid_section_count": len(dates),
                "eligible_sections_sha256": section_hash, "valid_sections_sha256": section_hash}
    metrics = {"rank_ic_mean": float(rank_ic.mean()), "ic_mean": float(rank_ic.mean()),
               "rank_ic_ir": aggregate_metric(rank_ic, "ir"),
               "long_short_return_mean": 0.003, "turnover_mean": 0.3,
               "cross_section_count": len(dates),
               "sample_keys_sha256": "a" * 64, "targets_sha256": "b" * 64,
               "correlation_coverage": {"schema_version": 1, "rank_ic": coverage, "ic": coverage},
               **CONTRACT_METRIC_BLOCKS}
    save(path, "manifest", manifest)
    save(path, "config", config)
    save(path, "metrics", metrics)
    save(path, "data", {"files": {"bars": {"sha256": "ab" * 32}}, "version": "v1"})
    save(path, "checks", {"passed": True, "checks": [{"name": "behavior", "passed": True}]})
    archive(path)
    reference = path.with_name(path.name + "-original")
    shutil.copytree(path, reference)
    manifest["run_id"] = reference.name
    manifest.pop("reproducibility")
    save(reference, "manifest", manifest)
    archive(reference)
    return path


def suite(tmp_path, policy, **kwargs):
    baseline, candidate = [], []
    for month in (1, 2):
        for seed in (3, 8):
            baseline.append(run_fixture(tmp_path, "baseline", seed, month, policy))
            candidate.append(run_fixture(tmp_path, "candidate", seed, month, policy, **kwargs))
    return baseline, candidate


def edit(path, filename, update):
    value = json.loads((path / f"{filename}.json").read_text(encoding="utf-8"))
    update(value)
    save(path, filename, value)


def edit_verified_metrics(path, update):
    """Keep a rerun and its reference consistent when probing semantic evidence."""
    for run in (path, path.with_name(path.name + "-original")):
        edit(run, "metrics", update)
        archive(run)


def evaluated_suite(tmp_path, policy, *, baseline_scores, candidate_scores,
                    section_targets=((0.01, 0.02, 0.03, 0.04), (0.01, 0.02, 0.03, 0.04))):
    """Archive actual evaluation of two four-stock timestamps per trading day."""
    baseline, candidate = suite(tmp_path, policy)
    for arm, scores in ((baseline, baseline_scores), (candidate, candidate_scores)):
        for path in arm:
            dates = pd.to_datetime(pd.read_csv(path / "daily.csv")["date"])
            rows = [
                {"date": day + pd.Timedelta(hours=10, minutes=30 * section),
                 "instrument": instrument, "score": scores[section][stock],
                 "target": section_targets[section][stock]}
                for day in dates for section in range(2)
                for stock, instrument in enumerate(("A", "B", "C", "D"))
            ]
            frame = pd.DataFrame(rows)
            predictions = frame[["date", "instrument", "score"]]
            metrics, daily = evaluate(predictions, frame[["date", "instrument", "target"]],
                                      {"quantile": 0.25, "periods_per_year": 252,
                                       "min_cross_section": 3, "single_side_rate": 0.0})
            for run in (path, path.with_name(path.name + "-original")):
                save(run, "metrics", metrics)
                daily.to_csv(run / "daily.csv", index=False)
                predictions.to_parquet(run / "predictions.parquet", index=False)
                archive(run)
    return baseline, candidate


def test_verified_multiseed_multiperiod_gain_is_accepted_deterministically(tmp_path, policy):
    baseline, candidate = suite(tmp_path, policy)
    comparison, decision = compare_runs(baseline, candidate, policy)
    assert decision["status"] == "accept"
    assert decision["automatic_promotion"] is False
    assert comparison["improvement"] == pytest.approx(0.04)
    assert comparison["confidence_interval"]["low"] == pytest.approx(0.04)
    assert comparison == compare_runs(baseline, candidate, policy)[0]
    json.dumps([comparison, decision], allow_nan=False)


def test_single_win_never_accepted(tmp_path, policy):
    baseline = run_fixture(tmp_path, "baseline", 3, 1, policy)
    candidate = run_fixture(tmp_path, "candidate", 3, 1, policy)
    _, decision = compare_runs([baseline], [candidate], policy)
    assert decision["status"] == "inconclusive"
    assert any("seeds" in issue for issue in decision["missing_evidence"])
    assert any("periods" in issue for issue in decision["missing_evidence"])


def test_constant_candidate_sections_cannot_create_an_accepted_gain(tmp_path, policy):
    baseline, candidate = evaluated_suite(
        tmp_path, policy, baseline_scores=((1, 2, 3, 4), (4, 3, 2, 1)),
        candidate_scores=((1, 2, 3, 4), (7, 7, 7, 7)),
    )
    comparison, decision = compare_runs(baseline, candidate, policy)
    # The old finite-only averaging produced a gain of 1.0 on every day.
    assert comparison["improvement"] == pytest.approx(1)
    assert decision["status"] == "reject"
    assert any("incomplete rank_ic correlation coverage" in gate for gate in decision["failed_gates"])
    assert not decision["missing_evidence"]


def test_incomplete_baseline_cannot_support_candidate_acceptance(tmp_path, policy):
    baseline, candidate = evaluated_suite(
        tmp_path, policy, baseline_scores=((1, 3, 2, 4), (7, 7, 7, 7)),
        candidate_scores=((1, 2, 3, 4), (1, 2, 3, 4)),
    )
    _, decision = compare_runs(baseline, candidate, policy)
    assert decision["status"] == "inconclusive"
    assert not decision["failed_gates"]
    assert any("incomplete rank_ic correlation coverage" in issue for issue in decision["missing_evidence"])


def test_constant_target_sections_do_not_disqualify_an_otherwise_valid_gain(tmp_path, policy):
    baseline, candidate = evaluated_suite(
        tmp_path, policy, baseline_scores=((1, 3, 2, 4), (4, 3, 2, 1)),
        candidate_scores=((1, 2, 3, 4), (7, 7, 7, 7)),
        section_targets=((0.01, 0.02, 0.03, 0.04), (0.02, 0.02, 0.02, 0.02)),
    )
    comparison, decision = compare_runs(baseline, candidate, policy)
    assert decision["status"] == "accept"
    assert comparison["improvement"] == pytest.approx(0.2)


@pytest.mark.parametrize("evidence", ["metrics", "daily"])
def test_legacy_archives_without_correlation_evidence_are_inconclusive(tmp_path, policy, evidence):
    baseline, candidate = suite(tmp_path, policy)
    if evidence == "metrics":
        edit_verified_metrics(candidate[0], lambda value: value.pop("correlation_coverage"))
    else:
        for run in (candidate[0], candidate[0].with_name(candidate[0].name + "-original")):
            daily = pd.read_csv(run / "daily.csv").drop(columns="eligible_rank_ic_count")
            daily.to_csv(run / "daily.csv", index=False)
            archive(run)
    _, decision = compare_runs(baseline, candidate, policy)
    assert decision["status"] == "inconclusive"
    assert not decision["failed_gates"]
    assert any("correlation coverage evidence" in issue for issue in decision["missing_evidence"])


@pytest.mark.parametrize("evidence", ["null", "list", "unsupported_schema", "missing_eligible_count"])
def test_incomplete_coverage_metadata_is_inconclusive_without_population_errors(tmp_path, policy, evidence):
    baseline, candidate = suite(tmp_path, policy)
    def remove_evidence(value):
        if evidence in {"null", "list"}:
            value["correlation_coverage"] = None if evidence == "null" else []
        elif evidence == "unsupported_schema":
            value["correlation_coverage"]["schema_version"] = 99
        else:
            value["correlation_coverage"]["rank_ic"].pop("eligible_section_count")
    edit_verified_metrics(candidate[0], remove_evidence)
    _, decision = compare_runs(baseline, candidate, policy)
    assert decision["status"] == "inconclusive"
    assert not decision["failed_gates"]
    assert any("correlation coverage evidence" in issue for issue in decision["missing_evidence"])


@pytest.mark.parametrize("primary,secondary,column", [
    ("rank_ic_ir", "long_short_return_mean", "rank_ic"),
    ("ic_mean", "rank_ic_mean", "rank_ic"),
    ("rank_ic_mean", "ic_mean", "ic"),
])
def test_equal_counts_with_different_timestamp_identities_fail_policy_gates(
    tmp_path, policy, primary, secondary, column,
):
    policy = {**policy, "primary_metric": primary,
              "max_metric_regressions": {secondary: 0.1, "turnover_mean": 0.05}}
    baseline, candidate = suite(tmp_path, policy)
    edit_verified_metrics(candidate[0], lambda value: value["correlation_coverage"][column].update(
        valid_sections_sha256=digest(["different prediction timestamp"]),
    ))
    _, decision = compare_runs(baseline, candidate, policy)
    assert decision["status"] == "reject"
    assert any(f"incomplete {column} correlation coverage" in gate for gate in decision["failed_gates"])
    assert not decision["missing_evidence"]


def test_correlation_coverage_is_only_required_for_policy_correlations(tmp_path, policy):
    policy = {**policy, "primary_metric": "long_short_return_mean", "min_improvement": 0.001,
              "max_metric_regressions": {"turnover_mean": 0.05}}
    baseline, candidate = evaluated_suite(
        tmp_path, policy, baseline_scores=((1, 2, 3, 4), (4, 3, 2, 1)),
        candidate_scores=((1, 2, 3, 4), (7, 7, 7, 7)),
    )
    for path in baseline + candidate:
        edit_verified_metrics(path, lambda value: value.pop("correlation_coverage"))
    comparison, decision = compare_runs(baseline, candidate, policy)
    assert comparison["improvement"] == pytest.approx(0.015)
    assert decision["status"] == "accept"


@pytest.mark.parametrize("mismatch", ["metadata_total", "daily_total", "population_hash", "population_count"])
def test_inconsistent_or_unfair_correlation_populations_cannot_be_compared(tmp_path, policy, mismatch):
    baseline, candidate = suite(tmp_path, policy)
    path = candidate[0]
    if mismatch == "metadata_total":
        edit_verified_metrics(path, lambda value: value["correlation_coverage"]["rank_ic"].update(
            eligible_section_count=13,
        ))
    elif mismatch == "population_hash":
        edit_verified_metrics(path, lambda value: value["correlation_coverage"]["rank_ic"].update(
            eligible_sections_sha256="c" * 64, valid_sections_sha256="c" * 64,
        ))
    else:
        for run in (path, path.with_name(path.name + "-original")):
            daily = pd.read_csv(run / "daily.csv")
            daily.loc[0, "eligible_rank_ic_count"] = 0
            daily.loc[0, "valid_rank_ic_count"] = 0
            if mismatch == "population_count":
                daily.loc[0, "rank_ic"] = np.nan
                section_hash = digest([pd.Timestamp(date).isoformat() for date in daily["date"].iloc[1:]])
                edit(run, "metrics", lambda value: value["correlation_coverage"]["rank_ic"].update(
                    eligible_section_count=11, valid_section_count=11,
                    eligible_sections_sha256=section_hash, valid_sections_sha256=section_hash,
                ))
                edit(run, "metrics", lambda value: value.update(
                    rank_ic_mean=aggregate_metric(daily["rank_ic"].to_numpy(), "mean"),
                    rank_ic_ir=aggregate_metric(daily["rank_ic"].to_numpy(), "ir"),
                ))
            daily.to_csv(run / "daily.csv", index=False)
            archive(run)
    with pytest.raises(ValueError, match="correlation coverage|target-eligible"):
        compare_runs(baseline, candidate, policy)


def test_finite_daily_correlation_cannot_claim_zero_valid_sections(tmp_path, policy):
    baseline, candidate = suite(tmp_path, policy)
    path = candidate[0]
    for run in (path, path.with_name(path.name + "-original")):
        daily = pd.read_csv(run / "daily.csv")
        daily.loc[0, "valid_rank_ic_count"] = 0
        section_hash = digest([pd.Timestamp(date).isoformat() for date in daily["date"].iloc[1:]])
        edit(run, "metrics", lambda value: value["correlation_coverage"]["rank_ic"].update(
            valid_section_count=11, valid_sections_sha256=section_hash,
        ))
        daily.to_csv(run / "daily.csv", index=False)
        archive(run)
    with pytest.raises(ValueError, match="daily values contradict valid correlation counts"):
        compare_runs(baseline, candidate, policy)


@pytest.mark.parametrize("field", ["budget", "evaluation", "splits", "data"])
def test_unfair_pairs_fail_closed(tmp_path, policy, field):
    baseline, candidate = suite(tmp_path, policy)
    edit(candidate[0], "config", lambda value: value[field].update(unfair=True))
    with pytest.raises(ValueError, match="unfair"):
        compare_runs(baseline, candidate, policy)


def test_final_data_cannot_select_model(tmp_path, policy):
    baseline, candidate = suite(tmp_path, policy)
    edit(candidate[0], "config", lambda value: value.update(evaluation_split="final"))
    with pytest.raises(ValueError, match="final-check"):
        compare_runs(baseline, candidate, policy)


@pytest.mark.parametrize("issue", ["no_rerun", "tolerance", "late_policy", "policy_changed"])
def test_missing_predeclared_evidence_is_inconclusive(tmp_path, policy, issue):
    baseline, candidate = suite(tmp_path, policy)
    def mutate(value):
        if issue == "no_rerun":
            value.pop("reproducibility")
        elif issue == "tolerance":
            value["reproducibility"].update(atol=1)
        elif issue == "late_policy":
            value.update(comparison_policy_registered_at="2025-03-01T11:00:00Z")
        else:
            value.update(comparison_policy_sha256="changed")
    edit(candidate[0], "manifest", mutate)
    _, decision = compare_runs(baseline, candidate, policy)
    assert decision["status"] == "inconclusive"
    assert decision["missing_evidence"]


@pytest.mark.parametrize("issue", ["cost", "secondary", "checks"])
def test_primary_gain_cannot_override_other_failures(tmp_path, policy, issue):
    baseline, candidate = suite(tmp_path, policy)
    if issue == "cost":
        edit(candidate[0], "manifest", lambda value: value["resources"].update(wall_seconds=100))
    elif issue == "secondary":
        daily = pd.read_csv(candidate[0] / "daily.csv")
        daily["turnover"] = 0.9
        daily.to_csv(candidate[0] / "daily.csv", index=False)
        edit(candidate[0], "metrics", lambda value: value.update(turnover_mean=0.9))
        archive(candidate[0])
    else:
        edit(candidate[0], "checks", lambda value: value.update(passed=False))
        archive(candidate[0])
    _, decision = compare_runs(baseline, candidate, policy)
    assert decision["status"] == "reject"
    assert decision["failed_gates"]


def test_daily_metric_tampering_is_detected(tmp_path, policy):
    baseline, candidate = suite(tmp_path, policy)
    edit(candidate[0], "metrics", lambda value: value.update(rank_ic_mean=0.99))
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        compare_runs(baseline, candidate, policy)


def test_modifying_daily_and_metrics_together_does_not_evade_inventory(tmp_path, policy):
    baseline, candidate = suite(tmp_path, policy)
    daily = pd.read_csv(candidate[0] / "daily.csv")
    daily["rank_ic"] += 0.5
    daily.to_csv(candidate[0] / "daily.csv", index=False)
    edit(candidate[0], "metrics", lambda value: value.update(rank_ic_mean=float(daily["rank_ic"].mean())))
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        compare_runs(baseline, candidate, policy)


def test_missing_inventory_is_inconclusive(tmp_path, policy):
    baseline, candidate = suite(tmp_path, policy)
    edit(candidate[0], "manifest", lambda value: value.pop("artifacts"))
    _, decision = compare_runs(baseline, candidate, policy)
    assert decision["status"] == "inconclusive"


def test_claimed_rerun_with_absent_actual_reference_is_inconclusive(tmp_path, policy):
    baseline, candidate = suite(tmp_path, policy)
    edit(candidate[0], "manifest", lambda value: value["reproducibility"].update(reference_run_id="missing"))
    _, decision = compare_runs(baseline, candidate, policy)
    assert decision["status"] == "inconclusive"
    assert any("reference run archive absent" in reason for reason in decision["missing_evidence"])


def test_actual_rerun_prediction_difference_overrides_passed_boolean(tmp_path, policy):
    baseline, candidate = suite(tmp_path, policy)
    predictions = pd.read_parquet(candidate[0] / "predictions.parquet")
    predictions["score"] += 0.1
    predictions.to_parquet(candidate[0] / "predictions.parquet")
    archive(candidate[0])
    _, decision = compare_runs(baseline, candidate, policy)
    assert decision["status"] == "reject"
    assert any("actual rerun predictions" in reason for reason in decision["failed_gates"])


def test_changed_software_cannot_be_compared(tmp_path, policy):
    baseline, candidate = suite(tmp_path, policy)
    def change_source(value):
        value["code"]["files"] = {"fixture.py": "different"}
        value["code"]["content_sha256"] = digest(value["code"]["files"])
    edit(candidate[0], "manifest", change_source)
    edit(candidate[0].with_name(candidate[0].name + "-original"), "manifest", change_source)
    with pytest.raises(ValueError, match="software implementations"):
        compare_runs(baseline, candidate, policy)


def test_duplicate_run_ids_across_arms_are_rejected(tmp_path, policy):
    baseline, candidate = suite(tmp_path, policy)
    edit(candidate[0], "manifest", lambda value: value.update(run_id=baseline[0].name))
    with pytest.raises(ValueError, match="run IDs"):
        compare_runs(baseline, candidate, policy)


def test_bootstrap_keeps_contiguous_blocks_and_does_not_wrap():
    indices = _block_indices(12, 3, np.random.default_rng(9)).reshape(-1, 3)
    assert np.all(np.diff(indices, axis=1) == 1)
    assert indices.min() >= 0
    assert indices.max() < 12


def test_thresholds_cannot_be_silently_invented(policy):
    invalid = copy.deepcopy(policy)
    invalid.pop("min_improvement")
    with pytest.raises(ValueError, match="policy keys"):
        validate_policy(invalid)
    invalid = copy.deepcopy(policy)
    invalid["min_seeds"] = 1
    with pytest.raises(ValueError, match="min_seeds"):
        validate_policy(invalid)


def change_recipe(path, update, *, metadata=None):
    """Update both archived reruns, preserving the artifact/checksum contract."""
    for run in (path, path.with_name(path.name + "-original")):
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        update(config)
        save(run, "config", config)
        (run / "config.sha256").write_text(digest(config), encoding="utf-8")
        edit(run, "manifest", lambda m: m.update(config_sha256=digest(config)))
        if metadata:
            edit(run, "data", metadata)
        archive(run)


@pytest.mark.parametrize("factor,value", [("lookback", 60), ("feature_columns", ["close", "volume"])])
def test_declared_data_factor_keeps_sample_and_source_checks(tmp_path, policy, factor, value):
    baseline_value = 30 if factor == "lookback" else ["close"]
    policy["change"] = {"factor": f"data.{factor}", "baseline": baseline_value,
                        "candidate": value, "hypothesis": "Test one data factor"}
    baseline, candidate = suite(tmp_path, policy)
    for path in baseline:
        change_recipe(path, lambda c: c["data"].update({factor: baseline_value}))
    for path in candidate:
        change_recipe(path, lambda c: (c["model"].update(alpha=1), c["data"].update({factor: value})))
    # Feature metadata is allowed only when it agrees with the declared config.
    if factor == "feature_columns":
        for arm, fields in ((baseline, baseline_value), (candidate, value)):
            for path in arm:
                change_recipe(path, lambda c: None, metadata=lambda d: d.update(feature_columns=fields))
        calibration = json.loads(Path(policy["threshold_basis"]["calibration_report"]).read_text())
        for path in calibration["runs"]:
            change_recipe(Path(path), lambda c: c["data"].update(feature_columns=baseline_value),
                          metadata=lambda d: d.update(feature_columns=baseline_value))
    comparison, decision = compare_runs(baseline, candidate, policy)
    assert decision["result"] == "保留", decision
    assert comparison["change"]["factor"] == f"data.{factor}"
    edit_verified_metrics(candidate[0], lambda m: m.update(sample_keys_sha256="different"))
    with pytest.raises(ValueError, match="evaluated sample_keys"):
        compare_runs(baseline, candidate, policy)


def test_declared_history_change_does_not_allow_second_factor(tmp_path, policy):
    policy["change"].update(factor="data.lookback", baseline=30, candidate=60)
    baseline, candidate = suite(tmp_path, policy)
    for path in candidate:
        change_recipe(path, lambda c: c["data"].update(lookback=60))
    with pytest.raises(ValueError, match="undeclared"):
        compare_runs(baseline, candidate, policy)


@pytest.mark.parametrize("issue", ["below_practical", "below_noise", "forged_summary", "late_calibration",
                                  "changed_source", "different_calibration_data"])
def test_unverified_threshold_cannot_retain_candidate(tmp_path, policy, issue):
    calibration_path = Path(policy["threshold_basis"]["calibration_report"])
    report = json.loads(calibration_path.read_text())
    if issue == "below_practical":
        policy["threshold_basis"]["practical_improvement"] = policy["min_improvement"] * 2
    elif issue == "below_noise":
        path = Path(report["runs"][0])
        daily = pd.read_csv(path / "daily.csv")
        daily["rank_ic"] += .2
        daily.to_csv(path / "daily.csv", index=False)
        edit(path, "metrics", lambda m: m.update(rank_ic_mean=float(daily.rank_ic.mean())))
        archive(path)
        values = [json.loads((Path(p) / "metrics.json").read_text())["rank_ic_mean"] for p in report["runs"]]
        report["metrics"]["rank_ic_mean"] = {"n": 4, "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)), "min": min(values), "max": max(values)}
    elif issue == "forged_summary":
        report["metrics"]["rank_ic_mean"]["std"] = 0.9
    elif issue == "late_calibration":
        report["created_at"] = "2025-03-01T09:30:00Z"
    elif issue == "changed_source":
        edit(Path(report["runs"][0]), "metrics", lambda m: m.update(rank_ic_mean=0.9))
    else:
        path = Path(report["runs"][0])
        edit(path, "data", lambda d: d.update(version="different-snapshot"))
        archive(path)
    if issue in {"forged_summary", "late_calibration", "below_noise"}:
        save(calibration_path.parent, "calibration", report)
        policy["threshold_basis"]["calibration_sha256"] = digest(report)
    baseline, candidate = suite(tmp_path, policy)
    _, decision = compare_runs(baseline, candidate, policy)
    assert decision["result"] == "证据不足"
    assert any("calibration" in reason for reason in decision["missing_evidence"])


def test_complete_zero_gain_reverts_but_missing_replication_is_inconclusive(tmp_path, policy):
    baseline, candidate = suite(tmp_path, policy, improvement=0)
    _, decision = compare_runs(baseline, candidate, policy)
    assert decision["result"] == "回退"
    assert decision["action"] == "retain_baseline"
    edit(candidate[0], "manifest", lambda m: m.pop("reproducibility"))
    _, decision = compare_runs(baseline, candidate, policy)
    assert decision["result"] == "证据不足"


def test_matching_pairs_still_require_one_evaluation_population_across_seeds(tmp_path, policy):
    baseline, candidate = suite(tmp_path, policy)
    for path in (baseline[0], candidate[0]):
        edit_verified_metrics(path, lambda m: m.update(sample_keys_sha256="different-seed-population"))
    with pytest.raises(ValueError, match="across seeds"):
        compare_runs(baseline, candidate, policy)

@pytest.mark.parametrize("case", ["both_missing", "both_old", "mixed"])
def test_comparison_rejects_runs_without_the_current_evaluation_version(tmp_path, policy, case):
    """Every compared run must be eval-v2; removing the check makes these tests fail."""
    baseline, candidate = suite(tmp_path, policy)
    def drop(value):
        value.pop("evaluation_version", None)
    if case == "both_missing":
        for path in (baseline[0], candidate[0]):
            edit_verified_metrics(path, drop)
    elif case == "both_old":
        for path in (baseline[0], candidate[0]):
            edit_verified_metrics(path, lambda value: value.update(evaluation_version="eval-v1"))
    else:
        for path in (baseline[0], candidate[0]):
            edit_verified_metrics(path, drop)
        edit_verified_metrics(candidate[0], lambda value: value.update(evaluation_version=EVALUATION_VERSION))
    with pytest.raises(ValueError, match="evaluation_version"):
        compare_runs(baseline, candidate, policy)


@pytest.mark.parametrize("mutation,expected_status,reason_fragment", [
    ("overlap", "overlapping_labels", "overlapping label periods"),
    ("overlap_flag", "overlapping_labels", "overlap block treats"),
    ("economic_undefined", "undefined_single_side_rate_not_configured", "undefined/descriptive by contract"),
    ("economic_descriptive", "descriptive_returns_undefined_overlapping_labels",
     "undefined/descriptive by contract"),
    ("economic_missing", "economic_block_absent", "economic_metrics block absent"),
    ("label_overlap_missing", "label_overlap_evidence_absent", "label_overlap evidence absent"),
])
def test_gate_blocks_comparison_acceptance(tmp_path, policy, mutation, expected_status, reason_fragment):
    """Comparison-level gate: removing ``if not gate['passed']`` makes this test accept."""
    baseline, candidate = suite(tmp_path, policy)
    comparison, decision = compare_runs(baseline, candidate, policy)
    assert decision["status"] == "accept"
    assert comparison["evidence_gates"]["evaluation"][candidate[0].name]["passed"] is True

    def mutate(value):
        if mutation == "overlap":
            value["label_overlap"] = {"overlapping_pair_count": 1,
                                      "independent_label_periods": False,
                                      "overlapping_labels_treated_as_independent_trades": False}
        elif mutation == "overlap_flag":
            value["label_overlap"]["overlapping_labels_treated_as_independent_trades"] = True
        elif mutation == "economic_undefined":
            value["economic_metrics"]["status"] = "undefined_single_side_rate_not_configured"
        elif mutation == "economic_descriptive":
            value["economic_metrics"]["status"] = "descriptive_returns_undefined_overlapping_labels"
        elif mutation == "economic_missing":
            value.pop("economic_metrics")
        else:
            value.pop("label_overlap")

    edit_verified_metrics(candidate[0], mutate)
    comparison, decision = compare_runs(baseline, candidate, policy)
    assert decision["status"] == "inconclusive"
    assert any(reason_fragment in issue for issue in decision["missing_evidence"])
    gate = comparison["evidence_gates"]["evaluation"][candidate[0].name]
    assert gate["passed"] is False
    assert gate["competition_status"] == expected_status or gate["economic_status"] == expected_status
    assert any(reason_fragment in reason for reason in gate["reasons"])
