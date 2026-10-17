"""Duration decisions use final, fair pairs and cannot average away a failed window or seed."""

import copy

import pandas as pd
import pytest
from test_training_round import archive, reseal

from jinyiquant.io import file_sha256, read_json, write_json
from scripts.analyze_training_duration import (
    PRIMARY,
    RISKS,
    analyze,
    collect_runs,
    duration_gate,
    reference_drift,
    select_budget,
)


def registration():
    return {"schema_version": 1, "experiment_id": "duration-test", "created_at_utc": "2026-01-01T00:00:00Z",
            "experiment": {"periods": ["first", "second"], "screen_seed": 11,
                           "replication_seeds": [29, 47], "budgets": [1500, 5000, 10000]},
            "calibration": {"pooled_seed_std": {m: .01 for m in (*PRIMARY, *RISKS)}},
            "limitations": ["Toy data, no final access"], "references": []}


def passing_rows(seeds=(11,)):
    return [{"period": period, "seed": seed,
             "delta": {m: .02 if m in PRIMARY else 0.0 for m in (*PRIMARY, *RISKS)}}
            for period in ("first", "second") for seed in seeds]


def intervals():
    return [{"period": "all", "metric": m, "interval": [.001, .03]} for m in PRIMARY]


@pytest.mark.parametrize("metric", [*PRIMARY, *RISKS])
def test_any_window_failure_blocks_screen_despite_large_other_window_gain(metric):
    rows = passing_rows()
    rows[0]["delta"][metric] = .010001 if metric == "turnover" else -.010001
    rows[1]["delta"][metric] = -100 if metric == "turnover" else 100
    gate = duration_gate(rows, intervals(), registration())
    assert gate["complete_grid"] and not gate["passed"]
    assert any(not row["passed"] and row["period"] == "first" and row["metric"] == metric
               for row in gate["rows"])


def test_replication_needs_complete_seed_grid_and_same_metric_positive_interval():
    reg, rows = registration(), passing_rows((11, 29, 47))
    assert duration_gate(rows, intervals(), reg, replication=True)["passed"]
    assert not duration_gate(rows[:-1], intervals(), reg, replication=True)["passed"]
    rows[-1]["delta"]["rank_ic"] = -.000001
    assert not duration_gate(rows, intervals(), reg, replication=True)["passed"]
    rows = passing_rows((11, 29, 47))
    for row in rows:
        row["delta"]["long_short_return"] = .001
    ci = intervals()
    ci[0]["interval"] = [-.01, .03]
    assert not duration_gate(rows, ci, reg, replication=True)["passed"]
    assert duration_gate([r for r in rows if r["seed"] == 11], ci, reg)["passed"]


def test_boundary_risk_limits_and_practical_threshold_are_inclusive():
    rows = passing_rows()
    for row in rows:
        row["delta"] = {m: .01 if m in (*PRIMARY, "turnover") else -.01 for m in (*PRIMARY, *RISKS)}
    assert duration_gate(rows, intervals(), registration())["passed"]
    rows[0]["delta"]["opening_rank_ic"] = None
    assert not duration_gate(rows, intervals(), registration())["passed"]


def test_selection_prefers_shorter_sub_noise_gain_and_excludes_risk_failure():
    pooled = registration()["calibration"]["pooled_seed_std"]
    gates = {"5000": {"passed": True, "mean_delta": {"rank_ic": .02, "long_short_return": .02}},
             "10000": {"passed": True, "mean_delta": {"rank_ic": .025, "long_short_return": .025}}}
    assert select_budget(gates, pooled) == 5000
    gates["10000"]["mean_delta"]["rank_ic"] = .04
    assert select_budget(gates, pooled) == 10000
    gates["10000"]["passed"] = False
    assert select_budget(gates, pooled) == 5000
    gates["5000"]["passed"] = False
    assert select_budget(gates, pooled) is None
    gates["5000"]["passed"] = gates["10000"]["passed"] = True
    gates["10000"]["mean_delta"] = {"rank_ic": .02, "long_short_return": 100}
    assert select_budget(gates, pooled) == 5000


def fixture_stage(tmp_path):
    reg, attempts = registration(), []
    stage = tmp_path / "stage"
    stage.mkdir()
    for period, start in (("first", "2023-07-18"), ("second", "2023-07-25")):
        for steps in reg["experiment"]["budgets"]:
            path = archive(tmp_path / f"{period}_{steps}", epochs=12, start=start, times=("09:30", "14:30"))
            config = read_json(path / "config.json")
            config["model"]["max_steps"] = steps
            config["acceptance"] = {"repro_atol": 1e-6, "repro_rtol": 1e-5}
            write_json(path / "config.json", config)
            curve = [{"epoch": .1, "steps": 500, "validation_rank_ic_mean": 100, "diagnostic_extra": "kept"},
                     {"epoch": 1, "steps": steps, "train_mse": .01, "validation_mse": .01,
                      "validation_rank_ic_mean": 1, "validation_long_short_return_mean": .01}]
            write_json(path / "learning_curve.json", curve)
            manifest = read_json(path / "manifest.json")
            manifest["started_at"] = "2026-01-02T00:00:00Z"
            write_json(path / "manifest.json", manifest)
            reseal(path)
            attempts.append({"name": path.name, "period": period, "seed": 11, "steps": steps,
                             "run": str(path), "exit_code": 0})
            if steps == 1500:
                reg["references"].append({"period": period, "seed": 11, "run": str(path),
                                          "sha256": {"manifest.json": file_sha256(path / "manifest.json")}})
    write_json(stage / "attempts.json", attempts)
    write_json(stage / "plan.json", {"runs": attempts})
    write_json(stage / "manifest.json", {"status": "completed"})
    reg_path = tmp_path / "preregistration.json"
    write_json(reg_path, reg)
    return reg_path, stage


def test_end_to_end_uses_existing_strict_comparison_preserves_inputs_and_never_picks_curve_peak(tmp_path):
    reg_path, stage = fixture_stage(tmp_path)
    inputs = [p for p in tmp_path.rglob("*") if p.is_file()]
    before = {str(p): file_sha256(p) for p in inputs}
    destination = tmp_path / "analysis"
    result = analyze(reg_path, [stage], destination)
    assert result["selected_screen_budget"] is None and result["retained_budget"] == 1500
    assert result["historical_reference_drift"]["passed"]
    assert result["final_views"] == 0 and not result["model_promoted"]
    assert before == {str(p): file_sha256(p) for p in inputs}
    comparison = read_json(destination / "comparison_1500_5000/comparison.json")
    assert comparison["plan"]["change"]["factor"] == "model.max_steps"
    assert comparison["checks"]["paired_keys_labels_data_and_single_factor"]
    assert len(result["descriptive_longer_budget_comparison"]["pairs"]) == 2
    curve = read_json(destination / "learning_curves.json")[0]["curve"]
    assert curve[0]["diagnostic_extra"] == "kept" and curve[0]["validation_rank_ic_mean"] == 100
    assert (destination / "training_duration.png").stat().st_size > 1000
    brief = (destination / "summary.txt").read_text(encoding="utf-8")
    assert brief.startswith("固定配方训练步数实验")
    assert "初筛两窗口等权均值（seed 11" in brief and "1500 | 1.000 | 1.000000" in brief
    assert "first | 11 | 5000" in brief and "second | 11 | 10000" in brief
    assert "成功运行已记录总耗时 0.20 分钟" in brief and "不等同纯训练速度" in brief
    assert "平均改善条件未通过：RankIC" in brief and "门槛 >= 0.010000" in brief
    assert "复验：未触发，不追加种子训练" in brief
    assert result["retained_weight_source"]["run"] == str(tmp_path / "second_1500")
    assert str(tmp_path / "second_1500") in brief
    assert len(pd.read_csv(destination / "terminal_metrics.csv")) == 6
    with pytest.raises(FileExistsError, match="overwrite"):
        analyze(reg_path, [stage], destination)


@pytest.mark.parametrize("fault", ["stage_failed", "attempt_failed", "duplicate", "short_run", "late_rules"])
def test_incomplete_ambiguous_or_posthoc_runs_are_rejected(tmp_path, fault):
    reg_path, stage = fixture_stage(tmp_path)
    reg, attempts = read_json(reg_path), read_json(stage / "attempts.json")
    if fault == "stage_failed":
        write_json(stage / "manifest.json", {"status": "failed"})
    elif fault == "attempt_failed":
        attempts[0]["exit_code"] = 1
    elif fault == "duplicate":
        attempts[-1] = {**attempts[0], "name": attempts[-1]["name"]}
    elif fault == "short_run":
        write_json(tmp_path / "first_1500/learning_curve.json", [{"epoch": 1, "steps": 1499}])
    else:
        reg["created_at_utc"] = "2026-02-01T00:00:00Z"
    write_json(stage / "attempts.json", attempts)
    with pytest.raises(ValueError):
        collect_runs(reg, [stage])


def test_old_source_reference_is_only_a_drift_check_and_reports_changed_keys(tmp_path):
    reg_path, stage = fixture_stage(tmp_path)
    reg = read_json(reg_path)
    records, _ = collect_runs(reg, [stage])
    changed = tmp_path / "changed_reference"
    changed.mkdir()
    source = tmp_path / "first_1500"
    for filename in ("config.json", "predictions.parquet"):
        (changed / filename).write_bytes((source / filename).read_bytes())
    predictions = pd.read_parquet(changed / "predictions.parquet")
    predictions.loc[0, "instrument"] = "DIFFERENT"
    predictions.to_parquet(changed / "predictions.parquet", index=False)
    write_json(changed / "manifest.json", {"artifacts": []})
    reseal(changed)
    reg = copy.deepcopy(reg)
    reg["references"][0].update(run=str(changed), sha256={"manifest.json": file_sha256(changed / "manifest.json")})
    result = reference_drift(records, reg)
    assert not result["passed"] and not result["rows"][0]["keys_equal"]
    assert all(not row["strict_comparison_arm"] for row in result["rows"])


def fine_stage(tmp_path):
    """A grid whose minimum (500) is below the explicit 1500-step baseline anchor."""
    reg, attempts = registration(), []
    reg["experiment"]["budgets"] = [500, 750, 1000, 1500]
    reg["experiment"]["baseline_budget"] = 1500
    stage = tmp_path / "stage"
    stage.mkdir()
    for period, start in (("first", "2023-07-18"), ("second", "2023-07-25")):
        for steps in reg["experiment"]["budgets"]:
            path = archive(tmp_path / f"{period}_{steps}", epochs=12, start=start, times=("09:30", "14:30"))
            config = read_json(path / "config.json")
            config["model"]["max_steps"] = steps
            config["acceptance"] = {"repro_atol": 1e-6, "repro_rtol": 1e-5}
            write_json(path / "config.json", config)
            curve = [{"epoch": .1, "steps": 250, "validation_rank_ic_mean": 100, "diagnostic_extra": "kept"},
                     {"epoch": 1, "steps": steps, "train_mse": .01, "validation_mse": .01,
                      "validation_rank_ic_mean": 1, "validation_long_short_return_mean": .01}]
            write_json(path / "learning_curve.json", curve)
            manifest = read_json(path / "manifest.json")
            manifest["started_at"] = "2026-01-02T00:00:00Z"
            write_json(path / "manifest.json", manifest)
            reseal(path)
            attempts.append({"name": path.name, "period": period, "seed": 11, "steps": steps,
                             "run": str(path), "exit_code": 0})
            if steps == 1500:
                reg["references"].append({"period": period, "seed": 11, "run": str(path),
                                          "sha256": {"manifest.json": file_sha256(path / "manifest.json")}})
    write_json(stage / "attempts.json", attempts)
    write_json(stage / "plan.json", {"runs": attempts})
    write_json(stage / "manifest.json", {"status": "completed"})
    reg_path = tmp_path / "preregistration.json"
    write_json(reg_path, reg)
    return reg_path, stage


def test_explicit_baseline_below_grid_minimum_stays_the_comparison_anchor(tmp_path):
    reg_path, stage = fine_stage(tmp_path)
    result = analyze(reg_path, [stage], tmp_path / "analysis")
    assert result["baseline_budget"] == 1500 and result["retained_budget"] == 1500
    assert result["historical_reference_drift"]["passed"]
    for candidate in (500, 750, 1000):
        comparison = read_json(tmp_path / f"analysis/comparison_1500_{candidate}/comparison.json")
        assert comparison["plan"]["change"]["baseline"] == 1500
        assert comparison["plan"]["change"]["candidate"] == candidate
    assert not (tmp_path / "analysis/comparison_500_1500").exists()
    assert result["retained_weight_source"]["run"] == str(tmp_path / "second_1500")
    assert result["retained_weight_source"]["source"] == "historical_1500_reference"
    brief = (tmp_path / "analysis/summary.txt").read_text(encoding="utf-8")
    assert "旧 1500 步参考预测一致性：通过" in brief
    assert "500 对 1500 步初筛" in brief and "1000 对 1500 步初筛" in brief


def test_baseline_budget_must_be_one_of_the_registered_budgets(tmp_path):
    reg_path, stage = fine_stage(tmp_path)
    reg = read_json(reg_path)
    reg["experiment"]["baseline_budget"] = 999
    write_json(reg_path, reg)
    with pytest.raises(ValueError, match="baseline_budget"):
        analyze(reg_path, [stage], tmp_path / "analysis")


def recovery_stages(tmp_path, *, status="failed"):
    reg_path, stopped = fixture_stage(tmp_path)
    ordered = sorted(read_json(stopped / "attempts.json"), key=lambda row: (row["steps"], row["period"]))
    failure = {**ordered[4], "run": str(tmp_path / "failed_worker"), "exit_code": 1073807364}
    failed_path = tmp_path / "failed_worker"
    failed_path.mkdir()
    (failed_path / "config.json").write_bytes((tmp_path / "first_10000/config.json").read_bytes())
    write_json(failed_path / "manifest.json", {"status": status, "exit_code": 1073807364,
               "error": {"reason": "worker terminated; exact cause unknown"},
               "resources": {"wall_seconds": 60, "peak_rss_mb": 70}})
    write_json(failed_path / "learning_curve.json", [{"steps": 3000, "epoch": 1.2}])
    write_json(stopped / "plan.json", {"runs": ordered})
    write_json(stopped / "attempts.json", ordered[:4] + [failure])
    write_json(stopped / "manifest.json", {"status": status, "error": {"reason": "worker exit"}})
    recovered = tmp_path / "recovered"
    recovered.mkdir()
    write_json(recovered / "manifest.json", {"status": "completed"})
    write_json(recovered / "attempts.json", ordered[4:])
    write_json(recovered / "plan.json", {"runs": ordered[4:],
               "recovery": {"original_stage": str(stopped), "reason": "restart with unchanged seeds"}})
    return reg_path, stopped, recovered


def test_explicit_recovery_keeps_failed_and_unexecuted_audit_and_uses_six_complete_cells(tmp_path):
    reg_path, stopped, recovered = recovery_stages(tmp_path)
    with pytest.raises(ValueError, match="not completed"):
        collect_runs(read_json(reg_path), [stopped, recovered])
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    before = {str(p): file_sha256(p) for p in files}
    result = analyze(reg_path, [stopped, recovered], tmp_path / "analysis", recover_failed_stages=True)
    audit = result["recovery_audit"]
    assert audit["enabled"] and audit["excluded_failed_attempts"] == 1
    assert audit["unexecuted_original_cells"] == 1 and audit["complete_unique_runs"] == 6
    old, new = result["stages"]
    assert old["recovered_successful_prefix"] and not new["recovered_successful_prefix"]
    failed = old["attempts"][-1]
    assert failed["exit_code"] == 1073807364 and not failed["included_in_comparison"]
    assert failed["evidence_sha256"]["manifest.json"] == file_sha256(tmp_path / "failed_worker/manifest.json")
    assert old["unexecuted"][0]["name"] == "second_10000"
    assert new["declared_recovery"]["original_stage"] == str(stopped)
    assert result["retained_budget"] == 1500 and not result["model_promoted"]
    brief = (tmp_path / "analysis/summary.txt").read_text(encoding="utf-8")
    assert "失败尝试 1 次；原计划未执行单元 1 个" in brief
    assert "失败尝试已记录耗时 1.00 分钟；合计 1.20 分钟" in brief
    assert "exit=1073807364；耗时 1.00 分钟；峰值 RSS 70.00 MiB" in brief
    assert before == {str(p): file_sha256(p) for p in files}


@pytest.mark.parametrize("fault", ["running", "gap", "out_of_order", "no_failed_exit", "duplicate_success",
                                  "incomplete_recovery"])
def test_recovery_does_not_hide_live_missing_reordered_or_duplicate_runs(tmp_path, fault):
    reg_path, stopped, recovered = recovery_stages(tmp_path, status="interrupted")
    attempts = read_json(stopped / "attempts.json")
    if fault == "running":
        write_json(stopped / "manifest.json", {"status": "running"})
    elif fault == "gap":
        attempts.pop(1)
    elif fault == "out_of_order":
        attempts[0], attempts[1] = attempts[1], attempts[0]
    elif fault == "no_failed_exit":
        attempts[-1]["exit_code"] = 0
    else:
        later = read_json(recovered / "attempts.json")
        later = later + [attempts[0]] if fault == "duplicate_success" else later[:1]
        write_json(recovered / "attempts.json", later)
        write_json(recovered / "plan.json", {"runs": later})
    write_json(stopped / "attempts.json", attempts)
    with pytest.raises(ValueError):
        collect_runs(read_json(reg_path), [stopped, recovered], recover_failed_stages=True)
