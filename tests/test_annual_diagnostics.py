from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jinyiquant.evaluation import evaluate
from jinyiquant.io import read_json
from scripts.analyze_annual_baseline import (
    coverage_table,
    diagnose_policy,
    distribution_summary,
    expected_timestamps,
    write_reports,
)

TIMES = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]
EVALUATION = {"quantile": .25, "periods_per_year": 252, "min_cross_section": 3}


@pytest.fixture
def annual_case():
    plan = {"evaluation_dates": ["2022-01-28", "2022-02-07"], "sample_times": TIMES,
            "trading_calendar": ["2021-12-31", "2022-01-27", "2022-01-28", "2022-02-07"]}
    grid = expected_timestamps(plan)
    universe = pd.DataFrame([(timestamp, stock) for timestamp in grid for stock in "ABCD"],
                            columns=["date", "instrument"])
    predictions = universe.assign(score=np.tile([1., 2., 3., 4.], len(grid)))
    predictions.loc[predictions.date.ge("2022-02-01"), "score"] *= -1
    targets = universe.assign(target=np.tile([.01, -.02, .03, .04], len(grid)))
    january = {"fit_month": "202201", "sha256": "a" * 64, "last_training_date": "2021-12-31",
               "train_end_exclusive": "2022-01-01"}
    february = {"fit_month": "202202", "sha256": "b" * 64, "last_training_date": "2022-01-28",
                "train_end_exclusive": "2022-02-01"}
    sources = {"202201": {"fixed": january, "monthly": january},
               "202202": {"fixed": january, "monthly": february}}
    cfg = SimpleNamespace(acceptance=SimpleNamespace(min_coverage=.99), evaluation=EVALUATION)
    return predictions, targets, universe, plan, sources, cfg


def run_case(case, policy="fixed"):
    predictions, targets, universe, plan, sources, cfg = case
    return diagnose_policy(predictions, targets, universe, plan, sources, policy, cfg)


@pytest.mark.parametrize("lost", ["predictions", "targets", "universe"])
def test_independent_calendar_detects_missing_entire_day(annual_case, lost):
    pred, target, universe, plan, _, cfg = annual_case
    frames = {"predictions": pred, "targets": target, "universe": universe}
    frames[lost] = frames[lost][frames[lost].date.lt("2022-02-01")]
    with pytest.raises(ValueError, match="incomplete independent date/time grid"):
        coverage_table(frames["universe"], frames["predictions"], frames["targets"],
                       expected_timestamps(plan), cfg.acceptance.min_coverage)


def test_day_chunk_carry_preserves_turnover_and_signal_changes_across_month(annual_case):
    pred, target, _, _, _, _ = annual_case
    actual = run_case(annual_case)
    _, _, reference = evaluate(pred, target, EVALUATION, return_sections=True,
                               undefined_constant_portfolios=True)
    for metric in ["turnover", "score_change_mae", "rank_persistence", "rank_ic", "long_short_return"]:
        np.testing.assert_allclose(actual[metric], reference[metric], equal_nan=True)
    assert pd.isna(actual.iloc[0].turnover)
    assert actual.loc[actual.timestamp.eq("2022-02-07 09:30"), "turnover"].item() == 2
    assert actual.loc[actual.timestamp.eq("2022-02-07 09:30"), "signal_common_count"].item() == 4


def test_weight_age_uses_independent_calendar_and_training_origin(annual_case):
    fixed, monthly = run_case(annual_case), run_case(annual_case, "monthly")
    assert fixed[fixed.month.eq("2022-02")].weight_age_trading_days.eq(3).all()
    assert monthly[monthly.month.eq("2022-02")].weight_age_trading_days.eq(1).all()
    assert monthly[monthly.month.eq("2022-02")].weight_age_calendar_days.eq(10).all()
    assert monthly.weight_changed.sum() == 1
    assert not fixed.weight_changed.any()
    annual_case[4]["202202"]["monthly"]["last_training_date"] = "2022-02-07"
    with pytest.raises(ValueError, match="before their training boundary"):
        run_case(annual_case, "monthly")


def test_constant_signal_remains_undefined_and_does_not_create_turnover(annual_case):
    pred, _, _, _, _, _ = annual_case
    pred.loc[pred.date.eq("2022-01-28 14:30"), "score"] = 5
    actual = run_case(annual_case)
    constant = actual[actual.timestamp.eq("2022-01-28 14:30")].iloc[0]
    assert np.isnan(constant.rank_ic) and np.isnan(constant.long_short_return)
    assert constant.undefined_rank_ic and not constant.negative_rank_ic
    assert not constant.ranking_defined
    assert pd.isna(actual[actual.timestamp.eq("2022-02-07 09:30")].iloc[0].turnover)
    assert actual[actual.timestamp.eq("2022-02-07 09:30")].iloc[0].score_change_mae > 0


def test_missing_target_keys_are_counted_without_hiding_missing_predictions(annual_case):
    pred, target, universe, plan, _, _ = annual_case
    sparse_target = target.drop(index=0)
    coverage = coverage_table(universe, pred, sparse_target, expected_timestamps(plan), .75)
    assert coverage.iloc[0].target_coverage == .75
    assert coverage.iloc[0].missing_target_count == 1
    with pytest.raises(ValueError, match="Predictions missing target keys"):
        coverage_table(universe, pred.drop(index=0), target, expected_timestamps(plan), .75)


def test_reports_retain_each_metric_each_timestamp_without_annual_aggregates(annual_case, tmp_path):
    fixed, monthly = run_case(annual_case), run_case(annual_case, "monthly")
    detail = pd.concat([fixed, monthly], ignore_index=True)
    write_reports(detail, tmp_path)
    exported = pd.read_csv(tmp_path / "by_timestamp.csv")
    assert len(exported) == 2 * 2 * 8
    assert not exported.duplicated(["policy", "timestamp"]).any()
    assert exported.groupby(["policy", "date"]).size().eq(8).all()
    for file in tmp_path.glob("*distributions.csv"):
        summary = pd.read_csv(file)
        assert "month" in summary and summary.month.notna().all()
        assert "mean" not in summary.columns
    assert not list(tmp_path.glob("*metrics.json"))
    assert (tmp_path / "2022-01.html").exists() and (tmp_path / "2022-02.html").exists()
    paired = pd.read_csv(tmp_path / "paired_by_timestamp.csv")
    assert len(paired) == 16 and paired.rank_ic_delta.eq(0).all()
    assert paired.turnover_delta.isna().sum() == 1
    findings = read_json(tmp_path / "monthly_failure_locations.json")
    february = [r for r in findings if r["month"] == "2022-02" and r["metric"] == "rank_ic"]
    assert len(february) == 2 and all(r["negative_count"] == 1 for r in february)
    with pytest.raises(ValueError, match="must retain month"):
        distribution_summary(detail, ["policy"], ["rank_ic"])


def test_reserved_year_cannot_enter_analysis_calendar():
    with pytest.raises(ValueError, match="2022 research year"):
        expected_timestamps({"evaluation_dates": ["2024-07-01"], "sample_times": TIMES})
