"""Behavior checks for the eval-v2 re-evaluation run (t7 artifact invariants)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/reeval-a-20261026"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def latest_run() -> Path:
    runs = sorted(path for path in RUNS.glob("*") if (path / "manifest.json").is_file())
    assert runs, "re-evaluation run archive is missing"
    return runs[-1]


@pytest.mark.skipif(not RUNS.is_dir(), reason="re-evaluation run archive is not present")
def test_manifest_records_contract_and_re_evaluated_provenance():
    run = latest_run()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["evaluation_version"] == "eval-v2"
    assert manifest["run_id"] == run.name
    assert manifest["plan_sha256"] == sha256_file(run / "plan.json")
    assert manifest["config_sha256"] == digest(config)
    assert manifest["evaluation_source"]["evaluation_py_sha256"] == \
        sha256_file(ROOT / "src/jinyiquant/evaluation.py")
    # The archive records the comparison.py it actually ran with. Archives are never
    # rewritten, so a later hardening change leaves the recorded hash frozen here.
    comparison_sha = manifest["evaluation_source"]["comparison_py_sha256"]
    assert isinstance(comparison_sha, str) and len(comparison_sha) == 64
    assert all(character in "0123456789abcdef" for character in comparison_sha)
    reference = manifest["re_evaluated_from"]["reference_nine_blend"]["source_run_ids"]
    assert len(reference) == 9
    for candidate in ("candidate_tlob_3seed", "candidate_mlp_3seed", "candidate_patch_3seed"):
        assert len(manifest["re_evaluated_from"][candidate]["source_run_ids"]) == 3
    assert plan["no_training"] is True
    assert [window["name"] for window in plan["windows"]] == \
        ["nov2023", "jan2024", "mar2024", "apr2024", "jun2024"]
    assert plan["sealed_interval"] == ["2024-10-08", "2025-01-01"]
    assert plan["bootstrap"]["block_days"] == 5 and plan["bootstrap"]["draws"] == 4000
    assert [obj["id"] for obj in plan["objects"]] == [
        "reference_nine_blend", "candidate_tlob_3seed", "candidate_mlp_3seed",
        "candidate_patch_3seed", "baseline_constant_zero", "baseline_label_mean",
        "baseline_ridge_default", "control_shuffled_reference", "control_sign_flipped_reference"]
    assert len(manifest["source_archives"]["replays"]) == 5
    for replay in manifest["source_archives"]["replays"]:
        path = ROOT / replay["run_dir"] / "manifest.json"
        assert sha256_file(path) == replay["manifest_sha256"]


@pytest.mark.skipif(not RUNS.is_dir(), reason="re-evaluation run archive is not present")
def test_every_object_has_eval_v2_metrics_and_cost_scenarios():
    run = latest_run()
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    for item in plan["objects"]:
        metrics = json.loads((run / "objects" / item["id"] / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["evaluation_version"] == "eval-v2"
        assert metrics["metrics_0bp"]["evaluation_version"] == "eval-v2"
        assert metrics["metrics_5bp"]["evaluation_version"] == "eval-v2"
        scenarios = metrics["cost_scenarios"]["scenarios"]
        assert scenarios["0bp"]["single_side_rate"] == 0.0
        assert scenarios["5bp"]["single_side_rate"] == 0.0005
        assert metrics["cost_scenarios"]["turnover_conversion_factor"] == 0.5
        assert scenarios["5bp"]["economic_metrics"]["evaluation_version"] == "eval-v2"
    reference = json.loads((run / "objects/reference_nine_blend/metrics.json").read_text(encoding="utf-8"))
    assert reference["metrics_5bp"]["economic_metrics"]["status"] == "computed"
    assert reference["metrics_5bp"]["economic_metrics"]["breakeven_single_side_rate"] > 0
    for baseline in ("baseline_constant_zero", "baseline_label_mean"):
        metrics = json.loads((run / "objects" / baseline / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["gate"]["5bp"]["passed"] is False
        assert metrics["gate"]["5bp"]["reasons"]


@pytest.mark.skipif(not RUNS.is_dir(), reason="re-evaluation run archive is not present")
def test_paired_daily_table_stays_inside_months_and_outside_the_sealed_interval():
    run = latest_run()
    paired = pd.read_csv(run / "comparison" / "paired_daily.csv")
    assert set(paired["object_id"]) == {
        "candidate_tlob_3seed", "candidate_mlp_3seed", "candidate_patch_3seed",
        "baseline_constant_zero", "baseline_label_mean", "baseline_ridge_default",
        "control_shuffled_reference", "control_sign_flipped_reference"}
    assert len(paired) == 8 * 104
    assert sorted(paired["month"].unique()) == ["2023-11", "2024-01", "2024-03", "2024-04", "2024-06"]
    dates = pd.to_datetime(paired["date"])
    assert dates.min() >= pd.Timestamp("2023-11-01")
    assert dates.max() < pd.Timestamp("2024-07-01")
    assert ((dates >= pd.Timestamp("2024-10-08")) & (dates < pd.Timestamp("2025-01-01"))).sum() == 0
    intervals = json.loads((run / "comparison" / "intervals.json").read_text(encoding="utf-8"))
    reference = intervals["intervals"]["candidate_tlob_3seed"]["rank_ic"]
    assert reference["block_days"] == 5 and reference["draws"] == 4000
    assert reference["defined_days"] == 104
    assert sorted(reference["blocks_per_month"]) == ["2023-11", "2024-01", "2024-03", "2024-04", "2024-06"]


@pytest.mark.skipif(not RUNS.is_dir(), reason="re-evaluation run archive is not present")
def test_degradation_controls_behave_as_declared():
    run = latest_run()
    controls = json.loads((run / "comparison" / "controls.json").read_text(encoding="utf-8"))
    shuffled = controls["degradation_controls"]["control_shuffled_reference"]
    assert shuffled["passed"] is True
    assert abs(shuffled["observed_rank_ic_mean"]) < 0.01
    flipped = controls["degradation_controls"]["control_sign_flipped_reference"]
    assert flipped["passed"] is True
    assert flipped["observed_rank_ic_mean"] == pytest.approx(
        -flipped["reference_rank_ic_mean"], abs=1e-12)
    assert flipped["observed_spread_mean"] == pytest.approx(
        -flipped["reference_spread_mean"], abs=1e-12)
    registration = json.loads((run / "research_area_registration.json").read_text(encoding="utf-8"))
    assert registration["sealed_intersection"] == []
    assert registration["day_count"] == 104
    assert registration["evaluated_from"] == "2023-11-01"
    assert registration["evaluated_to"] == "2024-06-28"
    assert all(window["frozen_replay_coverage"] for window in registration["windows"])
    checks = json.loads((run / "checks.json").read_text(encoding="utf-8"))
    assert checks["passed"] is True
    assert all(item["passed"] for item in checks["checks"])
