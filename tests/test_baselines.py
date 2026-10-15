"""Forward-date baseline behavior, exact sample parity, and undefined constant scores."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jinyiquant.baselines import BaselinePlan, load_plan, paired_estimates, run_suite, variant
from jinyiquant.config import RunConfig, load_config
from jinyiquant.evaluation import evaluate
from jinyiquant.io import read_json, write_json
from jinyiquant.runner import run
from scripts.tlob_fixture import create_fixture

ROOT = Path(__file__).resolve().parents[1]


def plan_raw(base="config.json"):
    return {"schema_version": 1, "experiment_id": "baseline-test", "base_config": base, "pilot_run": "pilot",
        "seeds": [11, 29, 47], "periods": [
            {"name": "first", "train": {"start": "2020-01-02", "end": "2020-01-03"},
             "validation": {"start": "2020-01-03", "end": "2020-01-07"}},
            {"name": "second", "train": {"start": "2020-01-06", "end": "2020-01-07"},
             "validation": {"start": "2020-01-07", "end": "2020-01-10"}}],
        "total_wall_seconds": 300, "per_run_wall_seconds": 60, "max_rss_mb": 2048,
        "max_total_rss_mb": 4096, "max_train_samples": 100,
        "block_days": 2, "bootstrap_samples": 100, "confidence": .95, "bootstrap_seed": 3,
        "reproduce_period": "first", "rationale": "test behavior only", "speed_estimate": {"test": True}}


@pytest.mark.parametrize("case", ["two_seeds", "duplicate_seed", "overlap", "train_after_validation", "final"])
def test_invalid_grid_fails_before_reading_data(tmp_path, case):
    base_path = create_fixture(tmp_path / "fixture")
    base = read_json(base_path)
    base["splits"]["final"] = {"start": "2020-01-10", "end": "2020-01-13"}
    write_json(base_path, base)
    raw = plan_raw(str(base_path))
    if case == "two_seeds":
        raw["seeds"] = [11, 29]
    elif case == "duplicate_seed":
        raw["seeds"] = [11, 29, 29]
    elif case == "overlap":
        raw["periods"][1]["validation"]["start"] = "2020-01-06"
    elif case == "train_after_validation":
        raw["periods"][0]["train"]["end"] = "2020-01-04"
    else:
        raw["periods"][1]["validation"]["end"] = "2020-01-11"
    write_json(tmp_path / "plan.json", raw)
    with pytest.raises(ValueError):
        load_plan(tmp_path / "plan.json")


def test_constant_fractional_mean_has_no_correlations_or_ranked_portfolios():
    keys = pd.DataFrame({"date": pd.to_datetime(["2020-01-02 10:00"] * 1000),
                         "instrument": [str(i) for i in range(1000)]})
    metrics, daily, sections = evaluate(keys.assign(score=.00001234567890123),
        keys.assign(target=np.arange(1000) / 1000), {"quantile": .2, "periods_per_year": 252, "min_cross_section": 3},
        return_sections=True, undefined_constant_portfolios=True)
    assert metrics["ic_mean"] is None and metrics["rank_ic_mean"] is None
    assert metrics["long_short_return_mean"] is None and metrics["turnover_mean"] is None
    assert sections.ranking_defined.eq(False).all() and daily.mse.gt(0).all()
    json.dumps(metrics, allow_nan=False)


def test_paired_blocks_do_not_treat_seed_copies_as_independent_dates():
    plan = BaselinePlan.model_validate(plan_raw())
    rows = [{"period": p.name, "seed": s, "model": m, "date": str(d), "mse": 2 if m == "modified_tlob" else 1,
             "ic": .2 if m == "modified_tlob" else np.nan if m == "mean" else .1,
             "rank_ic": .2 if m == "modified_tlob" else np.nan if m == "mean" else .1,
             "long_short_return": .002 if m == "modified_tlob" else np.nan if m == "mean" else .001,
             "turnover": 1 if m != "mean" else np.nan}
            for p in plan.periods for s in plan.seeds for m in ("mean", "ridge", "modified_tlob") for d in range(4)]
    result = paired_estimates(pd.DataFrame(rows), plan)
    delta = next(x for x in result["comparisons"] if x["period"] == "first" and x["reference"] == "ridge" and x["metric"] == "rank_ic")
    assert delta["days"] == 4 and delta["mean"] == pytest.approx(.1)
    np.testing.assert_allclose(delta["interval"], [.1, .1])
    assert next(x for x in result["comparisons"] if x["reference"] == "mean" and x["metric"] == "rank_ic")["status"] == "undefined"
    with pytest.raises(ValueError, match="Incomplete"):
        paired_estimates(pd.DataFrame(rows).iloc[1:], plan)


def test_variants_freeze_samples_evaluation_and_budget(tmp_path):
    config = load_config(create_fixture(tmp_path / "fixture"))
    raw = config.snapshot()
    raw["splits"]["final"] = {"start": "2020-01-10", "end": "2020-01-13"}
    config = RunConfig.model_validate(raw)
    plan = BaselinePlan.model_validate(plan_raw())
    models = [variant(config, plan, plan.periods[0], 11, name, tmp_path) for name in ("mean", "ridge", "modified_tlob")]
    for field in ("data", "splits", "evaluation", "budget", "acceptance"):
        assert all(getattr(m, field) == getattr(models[0], field) for m in models)


def test_real_pipeline_writes_exact_curve_and_timestamp_errors(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    cfg = load_config(create_fixture(tmp_path / "fixture"))
    raw = cfg.snapshot()
    raw["logging"]["learning_curves"] = True
    raw["evaluation"]["constant_portfolios"] = "undefined"
    path, code = run(RunConfig.model_validate(raw))
    assert code == 0, (path / "worker.log").read_text()
    curve = read_json(path / "learning_curve.json")
    assert [row["epoch"] for row in curve] == [0, 1, 2]
    prediction = pd.read_parquet(path / "scored_predictions.parquet").merge(
        pd.read_parquet(path / "targets.parquet"), on=["date", "instrument"], validate="one_to_one")
    assert curve[-1]["validation_mse"] == pytest.approx(np.square(prediction.score - prediction.target).mean())
    sections = pd.read_csv(path / "by_timestamp.csv")
    assert len(sections) == 2 and sections.mse.gt(0).all()
    assert (path / "learning_curve.png").stat().st_size > 1000


def test_complete_suite_archives_fair_grid_reproduction_and_coverage(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    base_path = create_fixture(tmp_path / "fixture")
    raw = read_json(base_path)
    raw["splits"]["final"] = {"start": "2020-01-10", "end": "2020-01-13"}
    write_json(base_path, raw)
    base = load_config(base_path)
    pilot, code = run(base)
    assert code == 0
    plan = plan_raw(str(base_path))
    plan.update(pilot_run=str(pilot), parallel_runs=2)
    write_json(tmp_path / "plan.json", plan)
    destination, code = run_suite(tmp_path / "plan.json", tmp_path / "suite")
    assert code == 0, (destination / "worker.log").read_text()
    assert read_json(destination / "checks.json")["passed"]
    assert read_json(destination / "reproduction.json")["passed"]
    result = read_json(destination / "metrics.json")
    assert result["grid_runs"] == 18 and result["final_views"] == 0 and not result["model_promoted"]
    assert len(pd.read_csv(destination / "by_timestamp.csv")) == 45
    coverage = pd.read_csv(destination / "coverage_by_timestamp.csv")
    assert set(coverage.split) == {"train", "validation"}
    assert coverage.missing_answer_fraction.eq(0).all()
    assert (destination / "baseline_overview.png").stat().st_size > 1000
