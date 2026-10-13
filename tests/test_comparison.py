import copy
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from jinyiquant.comparison import _block_indices, compare_runs, validate_policy
from jinyiquant.io import digest, file_sha256


@pytest.fixture
def policy():
    return {
        "primary_metric": "rank_ic_mean", "direction": "maximize", "min_improvement": 0.015,
        "max_metric_regressions": {"long_short_return_mean": 0.001, "turnover_mean": 0.05},
        "max_wall_ratio": 1.5, "max_memory_ratio": 1.2, "min_seeds": 2, "min_periods": 2,
        "bootstrap_samples": 100, "block_days": 2, "confidence": 0.95, "seed": 8,
        "threshold_rationale": "Fixture: measured baseline variation below 0.01, practical gain 0.015.",
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
                          "sample_count": 100, "cross_section_count": 1})
    daily.to_csv(path / "daily.csv", index=False)
    pd.DataFrame({"date": dates, "instrument": "A", "score": rank_ic}).to_parquet(path / "predictions.parquet")
    (path / "source.zip").write_bytes(b"fixture source snapshot")
    config = {"seed": seed, "evaluation_split": "validation", "model": {"alpha": 1 if arm == "baseline" else 2},
              "data": {"version": "demo-v1", "lookback": 30, "bars_path": "input.parquet"},
              "splits": {"train": {"start": "2024-01-01", "end": "2025-01-01"}, "validation": period},
              "evaluation": {"periods_per_year": 252}, "budget": {"wall_seconds": 60, "max_rss_mb": 1024},
              "acceptance": {"repro_atol": 1e-10, "repro_rtol": 1e-8}}
    manifest = {"run_id": path.name, "status": "completed", "started_at": "2025-03-01T10:00:00Z",
                "comparison_policy_registered_at": "2025-03-01T09:00:00Z",
                "comparison_policy_sha256": digest(policy), "resources": {"wall_seconds": 10, "peak_rss_mb": 200},
                "reproducibility": {"passed": True, "reference_run_id": path.name + "-original", "atol": 1e-10, "rtol": 1e-8},
                "config_sha256": digest(config),
                "code": {"content_sha256": digest({"fixture.py": "example"}), "files": {"fixture.py": "example"},
                         "snapshot_sha256": file_sha256(path / "source.zip")}}
    (path / "config.sha256").write_text(digest(config), encoding="utf-8")
    metrics = {"rank_ic_mean": float(rank_ic.mean()), "long_short_return_mean": 0.003, "turnover_mean": 0.3,
               "sample_keys_sha256": "a" * 64, "targets_sha256": "b" * 64}
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
