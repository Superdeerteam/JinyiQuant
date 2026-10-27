"""Calibration checks for the single versioned evaluation contract (eval-v2).

Expected values are derived by hand or by an independent helper, not read back
from the implementation under test.
"""

import json

import numpy as np
import pandas as pd
import pytest

import jinyiquant.evaluation as evaluation
from jinyiquant.evaluation import (
    CONTRACT,
    EVALUATION_VERSION,
    METRIC_SPEC,
    TURNOVER_CONVERSION_FACTOR,
    aggregate_metric,
    align_predictions,
    drift_weights,
    evaluate,
    label_end,
    neutral_weights,
    rebalance_economics,
    total_abs_weight_change,
)
from jinyiquant.io import digest

RATE = 0.001
CONFIG = {"quantile": 0.2, "periods_per_year": 252, "min_cross_section": 3, "single_side_rate": RATE}
NO_RATE = {"quantile": 0.2, "periods_per_year": 252, "min_cross_section": 3}


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def assert_close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    assert abs(float(actual) - float(expected)) <= tolerance


def timed_case(pairs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build prediction/target frames from (timestamp, scores, targets) tuples."""
    rows = []
    for timestamp, scores, targets in pairs:
        assert len(scores) == len(targets)
        for index, (score, target) in enumerate(zip(scores, targets)):
            rows.append({"date": timestamp, "instrument": f"S{index:02d}",
                         "score": float(score), "target": float(target)})
    frame = pd.DataFrame(rows)
    return frame[["date", "instrument", "score"]], frame[["date", "instrument", "target"]]


def hand_case() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three stocks, one long and one short per side, two adjacent labels."""
    return timed_case([
        ("2025-01-02 10:00:00", [3.0, 2.0, 1.0], [0.02, 0.01, -0.03]),
        ("2025-01-02 10:30:00", [3.0, 2.0, 1.0], [0.02, 0.01, -0.03]),
    ])


def nested_keys(value) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(nested_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.extend(nested_keys(item))
    return keys


def test_contract_declares_units_legs_cost_and_annualization():
    metrics = evaluate(*hand_case(), CONFIG)[0]
    contract = metrics["contract"]
    assert contract["evaluation_version"] == EVALUATION_VERSION
    assert contract["label"]["horizon_minutes"] == 30
    assert "30-minute VWAP" in contract["label"]["return_unit"]
    assert "13:30" in contract["label"]["clock"] and "overnight" in contract["label"]["clock"]
    assert contract["portfolio"]["leg_capital"] == 1.0
    assert contract["portfolio"]["gross_exposure"] == 2.0
    assert contract["portfolio"]["net_exposure"] == 0.0
    assert "max(1, floor(n * quantile))" in contract["portfolio"]["tail_size"]
    assert contract["cost"]["cost_model"] == "cost = single_side_rate * U"
    assert "turnover = U / 2" in contract["cost"]["turnover"]
    assert "conversion_factor = 0.5" in contract["cost"]["turnover"]
    assert "not annualized" in contract["aggregation"]["ic_ir"]
    assert "sqrt(periods_per_year)" in contract["aggregation"]["sharpe"]
    assert metrics["aggregation"]["periods_per_year"] == 252
    assert metrics["aggregation"]["sharpe_annualization_factor"] == pytest.approx(np.sqrt(252))
    portfolio = metrics["portfolio"]
    assert portfolio["leg_capital"] == 1.0 and portfolio["gross_exposure"] == 2.0
    assert portfolio["net_exposure"] == 0.0
    assert portfolio["return_on_gross_notional_definition"] == "spread / 2.0"


def test_every_metric_product_carries_evaluation_version():
    metrics, daily, sections = evaluate(*hand_case(), CONFIG, return_sections=True)
    assert metrics["evaluation_version"] == EVALUATION_VERSION == "eval-v2"
    assert metrics["contract"] == CONTRACT
    assert metrics["competition_metrics"]["evaluation_version"] == EVALUATION_VERSION
    assert metrics["economic_metrics"]["evaluation_version"] == EVALUATION_VERSION
    assert daily["evaluation_version"].eq(EVALUATION_VERSION).all()
    assert sections["evaluation_version"].eq(EVALUATION_VERSION).all()
    assert digest(metrics["contract"]) == digest(CONTRACT)


def test_no_relative_rank_or_weighted_total_is_produced():
    metrics = evaluate(*hand_case(), CONFIG)[0]
    forbidden = ("percentile", "composite", "rank_pct", "综合")
    keys = nested_keys(metrics)
    assert not any(token in key.lower() for key in keys for token in forbidden)
    functions = [name for name in dir(evaluation) if not name.startswith("_")]
    assert not any(token in name.lower() for name in functions for token in forbidden)
    assert not any(key in metrics for key in ("total_score", "weighted_score", "platform_score"))
    json.dumps(metrics, allow_nan=False)


def test_repeated_calls_produce_byte_identical_metric_json():
    targets = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
    pairs = []
    for day in (2, 3, 6):
        for clock in ("10:00:00", "10:30:00", "11:00:00"):
            scores = list(range(10)) if day != 6 else list(range(9, -1, -1))
            pairs.append((f"2025-01-{day:02d} {clock}", scores, targets))
    predictions, labels = timed_case(pairs)
    first_metrics, first_daily = evaluate(predictions, labels, CONFIG)
    second_metrics, second_daily = evaluate(predictions, labels, CONFIG)
    assert canonical(first_metrics) == canonical(second_metrics)
    assert digest(first_metrics) == digest(second_metrics)
    pd.testing.assert_frame_equal(first_daily, second_daily)
    reordered_metrics, reordered_daily = evaluate(
        predictions.iloc[::-1].reset_index(drop=True),
        labels.sample(frac=1, random_state=5).reset_index(drop=True),
        CONFIG,
    )
    assert canonical(reordered_metrics) == canonical(first_metrics)
    pd.testing.assert_frame_equal(reordered_daily, first_daily)


def test_hand_computed_rebalance_cost_matches_to_1e_12():
    target = pd.Series({"A": 1.0, "B": 0.0, "C": -1.0})
    pretrade = pd.Series({"A": 0.8, "B": 0.1, "C": -0.9})
    result = rebalance_economics(target, pretrade, gross_return=0.004, single_side_rate=RATE)
    assert_close(result["total_abs_weight_change"], 0.4)
    assert_close(result["cost"], RATE * 0.4)
    assert_close(result["net_return"], 0.004 - RATE * 0.4)
    assert_close(result["turnover"], 0.2)
    assert result["turnover_conversion_factor"] == TURNOVER_CONVERSION_FACTOR == 0.5
    assert result["evaluation_version"] == EVALUATION_VERSION
    # Independent recomputation of the same U.
    assert_close(total_abs_weight_change(target, pretrade), 0.4)


def test_hand_computed_evaluate_economics_matches_to_1e_12():
    metrics, daily, sections = evaluate(*hand_case(), CONFIG, return_sections=True)
    expected_u = 0.03 / 1.05 + 0.08 / 1.05
    assert pd.isna(sections["total_abs_weight_change"].iloc[0])
    assert_close(sections["total_abs_weight_change"].iloc[1], expected_u)
    assert_close(sections["turnover"].iloc[1], expected_u / 2)
    assert_close(sections["cost"].iloc[1], RATE * expected_u)
    assert_close(sections["net_return"].iloc[1], 0.05 - RATE * expected_u)
    drifted = drift_weights(pd.Series({"A": 1.0, "B": 0.0, "C": -1.0}),
                            pd.Series({"A": 0.02, "B": 0.01, "C": -0.03}))
    assert_close(drifted["A"], 1.02 / 1.05)
    assert_close(drifted["B"], 0.0)
    assert_close(drifted["C"], -0.97 / 1.05)
    economics = metrics["economic_metrics"]
    assert economics["status"] == "computed"
    assert economics["defined_rebalance_count"] == 1
    assert economics["undefined_rebalance_count"] == 1
    assert_close(economics["gross_return_mean"], 0.05)
    assert_close(economics["total_abs_weight_change_mean"], expected_u)
    assert_close(economics["turnover_mean"], expected_u / 2)
    assert_close(economics["cost_mean"], RATE * expected_u)
    assert_close(economics["net_return_mean"], 0.05 - RATE * expected_u)
    assert_close(economics["breakeven_single_side_rate"], 0.05 / expected_u)
    assert economics["turnover_conversion_factor"] == 0.5
    assert_close(metrics["turnover_mean"], expected_u / 2)
    assert daily["long_short_return"].iloc[0] == pytest.approx(0.05)


def test_missing_cost_rate_is_undefined_and_counted_not_zero():
    metrics = evaluate(*hand_case(), NO_RATE)[0]
    economics = metrics["economic_metrics"]
    assert economics["single_side_rate"] is None
    assert economics["status"] == "undefined_single_side_rate_not_configured"
    assert economics["cost_mean"] is None and economics["net_return_mean"] is None
    assert economics["total_abs_weight_change_mean"] is not None
    assert economics["breakeven_single_side_rate"] is not None
    assert metrics["undefined_cost_count"] == metrics["cross_section_count"]
    assert json.loads(canonical(metrics))["economic_metrics"]["cost_mean"] is None


@pytest.mark.parametrize(("size", "expected_tail"), [(3, 1), (5, 1), (10, 2), (20, 4)])
def test_tail_size_is_floor_n_quantile_with_minimum_one(size, expected_tail):
    scores = pd.Series(np.arange(size, dtype=float), index=[f"S{i}" for i in range(size)])
    long, short, net, tail_size = neutral_weights(scores, 0.2)
    assert tail_size == expected_tail
    assert_close(long.sum(), 1.0)
    assert_close(short.sum(), 1.0)
    assert_close(long.sum() + short.sum(), 2.0)
    assert_close(net.sum(), 0.0)


def test_boundary_ties_share_the_boundary_weight():
    scores = pd.Series([1.0, 1.0, 4.0, 4.0], index=list("ABCD"))
    long, short, _, tail_size = neutral_weights(scores, 0.25)
    assert tail_size == 1
    assert_close(long["C"], 0.5) and assert_close(long["D"], 0.5)
    assert_close(short["A"], 0.5) and assert_close(short["B"], 0.5)


@pytest.mark.parametrize("constant", [0.0, 7.0])
def test_constant_and_zero_scores_are_undefined_and_not_zero_filled(constant):
    pairs = []
    for clock in ("10:00:00", "10:30:00"):
        pairs.append((f"2025-01-02 {clock}", [constant] * 5, [0.01, -0.02, 0.03, -0.04, 0.05]))
    predictions, labels = timed_case(pairs)
    metrics, daily, sections = evaluate(predictions, labels, CONFIG, return_sections=True)
    assert metrics["ic_mean"] is None and metrics["rank_ic_mean"] is None
    assert metrics["undefined_ic_count"] == 2
    assert daily["ic"].isna().all() and daily["rank_ic"].isna().all()
    assert daily["long_short_return"].eq(0).all()
    assert sections["net_weight_gross"].eq(0).all()
    assert json.loads(canonical(metrics))["ic_mean"] is None
    assert "not stock-selection ability" in metrics["contract"]["portfolio"]["constant_scores"]
    undefined_metrics, undefined_daily = evaluate(
        predictions, labels, CONFIG, undefined_constant_portfolios=True)
    assert undefined_metrics["long_short_return_mean"] is None
    assert undefined_daily["long_short_return"].isna().all()


def test_single_instrument_constant_column_still_requires_a_real_ranking():
    pairs = []
    for day in (2, 3):
        pairs.append((f"2025-01-{day:02d} 10:00:00", [5.0, 1.0, 2.0, 4.0, 3.0],
                      [0.01, 0.02, 0.03, 0.04, 0.05]))
    predictions, labels = timed_case(pairs)
    predictions.loc[predictions.instrument.eq("S02"), "score"] = 9.0
    metrics, _, sections = evaluate(predictions, labels, CONFIG, return_sections=True)
    assert sections["ranking_defined"].all()
    assert metrics["rank_ic_mean"] is not None and metrics["undefined_ic_count"] == 0
    with pytest.raises(ValueError, match="cross section"):
        evaluate(predictions[predictions.instrument.eq("S00")],
                 labels[labels.instrument.eq("S00")], CONFIG)


def test_shuffled_scores_have_no_rank_information_and_no_promotion():
    rng = np.random.default_rng(19)
    instruments = 30
    target_values = rng.normal(0.0, 0.02, instruments)
    rows = []
    for day in range(1, 41):
        timestamp = f"2025-{1 + (day - 1) // 28:02d}-{1 + (day - 1) % 28:02d} 10:00:00"
        shuffled = rng.permutation(target_values)
        for index in range(instruments):
            rows.append({"date": timestamp, "instrument": f"S{index:02d}",
                         "score": shuffled[index], "target": target_values[index]})
    frame = pd.DataFrame(rows)
    predictions = frame[["date", "instrument", "score"]]
    labels = frame[["date", "instrument", "target"]]
    metrics = evaluate(predictions, labels, NO_RATE)[0]
    assert abs(metrics["rank_ic_mean"]) < 0.1
    assert abs(metrics["ic_mean"]) < 0.1
    ordered = evaluate(frame[["date", "instrument"]].assign(score=frame["target"]), labels, NO_RATE)[0]
    assert ordered["rank_ic_mean"] > 0.99
    keys = nested_keys(metrics)
    assert not any("promot" in key.lower() or "decision" in key.lower() for key in keys)


def test_reversing_scores_flips_ic_and_long_short_signs():
    targets = [0.01, 0.02, 0.03, 0.04, 0.05]
    pairs = [(f"2025-01-{day:02d} {clock}", list(range(5)), targets)
             for day in (2, 3, 6) for clock in ("10:00:00", "10:30:00")]
    predictions, labels = timed_case(pairs)
    forward = evaluate(predictions, labels, CONFIG)[0]
    reverse = evaluate(predictions.assign(score=-predictions["score"]), labels, CONFIG)[0]
    assert_close(reverse["rank_ic_mean"], -forward["rank_ic_mean"])
    assert_close(reverse["ic_mean"], -forward["ic_mean"])
    assert_close(reverse["long_short_return_mean"], -forward["long_short_return_mean"])
    assert forward["long_short_return_mean"] > 0 and reverse["long_short_return_mean"] < 0
    assert reverse["turnover_mean"] is not None and reverse["turnover_mean"] > 0


def test_ic_ir_and_sharpe_follow_declared_annualization():
    targets = [0.01, 0.02, 0.03, 0.04]
    pairs = [(f"2025-01-{day:02d} 10:00:00", scores, targets)
             for day, scores in ((2, [1, 2, 3, 4]), (3, [1, 2, 3, 4]), (6, [4, 3, 2, 1]))]
    predictions, labels = timed_case(pairs)
    metrics, daily = evaluate(predictions, labels, CONFIG)
    assert daily["rank_ic"].tolist() == pytest.approx([1.0, 1.0, -1.0])
    assert_close(metrics["ic_ir"], daily["ic"].mean() / daily["ic"].std(ddof=1))
    assert_close(metrics["rank_ic_ir"], daily["rank_ic"].mean() / daily["rank_ic"].std(ddof=1))
    expected_sharpe = (daily["long_short_return"].mean() / daily["long_short_return"].std(ddof=1)
                       * np.sqrt(252))
    assert_close(metrics["long_short_sharpe"], expected_sharpe)
    assert_close(aggregate_metric(np.array([0.1, 0.2, 0.3]), "ir"), 2.0)
    assert_close(aggregate_metric(np.array([0.1, 0.2, 0.3]), "sharpe"), 2.0 * np.sqrt(252))


def test_overlapping_labels_cannot_be_independent_trades():
    targets = [0.01, 0.02, 0.03, 0.04, 0.05]
    predictions, labels = timed_case([
        ("2025-01-02 10:00:00", [0, 1, 2, 3, 4], targets),
        ("2025-01-02 10:15:00", [4, 3, 2, 1, 0], list(reversed(targets))),
        ("2025-01-02 10:50:00", [0, 1, 2, 3, 4], targets),
    ])
    metrics, daily, sections = evaluate(predictions, labels, CONFIG, return_sections=True)
    assert sections["label_overlap"].tolist() == [False, True, False]
    assert pd.isna(sections["total_abs_weight_change"].iloc[0])
    assert pd.isna(sections["total_abs_weight_change"].iloc[1])
    assert pd.notna(sections["total_abs_weight_change"].iloc[2])
    assert metrics["overlapping_label_section_count"] == 1
    assert metrics["label_overlap"]["independent_label_periods"] is False
    economics = metrics["economic_metrics"]
    assert economics["status"] == "descriptive_returns_undefined_overlapping_labels"
    assert economics["overlapping_labels_treated_as_independent_trades"] is False
    assert economics["gross_return_mean"] is None
    assert economics["net_return_mean"] is None
    assert economics["breakeven_single_side_rate"] is None
    # Trade-side quantities stay defined; only the return pairing is invalid.
    assert economics["defined_rebalance_count"] == 1
    assert economics["cost_mean"] is not None and economics["turnover_mean"] is not None
    assert metrics["undefined_turnover_count"] == 2
    assert_close(daily["long_short_return"].iloc[0], 0.04)
    assert "cumulative_return" not in metrics


def test_label_clock_matches_official_sampling_schedule():
    assert label_end("2025-01-02 09:30") == pd.Timestamp("2025-01-02 10:00")
    assert label_end("2025-01-02 11:30") == pd.Timestamp("2025-01-02 13:30")
    assert label_end("2025-01-02 14:30") == pd.Timestamp("2025-01-02 15:00")
    grid = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]
    for previous, following in zip(grid, grid[1:]):
        assert label_end(f"2025-01-02 {previous}") == pd.Timestamp(f"2025-01-02 {following}")
    with pytest.raises(ValueError):
        label_end("2025-01-02 10:00", 0)


def test_helper_level_recomputation_matches_evaluate():
    targets = [0.01, 0.02, 0.03, 0.04, 0.05]
    pairs = [(f"2025-01-{day:02d} {clock}", [1, 2, 3, 4, 5], targets)
             for day in (2, 3) for clock in ("10:00:00", "10:30:00")]
    predictions, labels = timed_case(pairs)
    metrics, daily, sections = evaluate(predictions, labels, CONFIG, return_sections=True)
    aligned = align_predictions(predictions, labels)
    previous_weights = previous_returns = previous_timestamp = None
    for index, (timestamp, section) in enumerate(aligned.groupby("date", sort=True)):
        instruments = section["instrument"].to_numpy()
        scores = pd.Series(section["score"].to_numpy(dtype=float), index=instruments)
        returns = pd.Series(section["target"].to_numpy(dtype=float), index=instruments)
        long, short, net, tail_size = neutral_weights(scores, 0.2)
        row = sections.iloc[index]
        assert row["tail_size"] == tail_size
        assert row["long_position_count"] == int((long > 0).sum())
        assert_close(row["long_return"], float(np.dot(long.to_numpy(), returns.to_numpy())))
        assert_close(row["short_return"], float(np.dot(short.to_numpy(), returns.to_numpy())))
        if previous_weights is not None:
            if timestamp == label_end(previous_timestamp):
                pretrade = drift_weights(previous_weights, previous_returns)
            else:
                pretrade = previous_weights
            expected_u = total_abs_weight_change(net, pretrade)
            assert_close(row["total_abs_weight_change"], expected_u)
            assert_close(row["cost"], RATE * expected_u)
            assert_close(row["net_return"], row["long_short_return"] - RATE * expected_u)
        previous_weights, previous_returns, previous_timestamp = net, returns, timestamp
    for name, (column, method, _) in METRIC_SPEC.items():
        expected = aggregate_metric(daily[column].to_numpy(), method, 252)
        if expected is None:
            assert metrics[name] is None
        else:
            assert_close(metrics[name], expected)

def test_full_replacement_calibration_uses_the_union_of_both_books():
    """U=2 for building a two-leg book from cash; union mutation must be caught."""
    target = pd.Series({"A": 1.0, "B": -1.0})
    built = rebalance_economics(target, pd.Series(dtype=float), gross_return=0.0,
                                single_side_rate=RATE)
    assert_close(built["total_abs_weight_change"], 2.0)
    assert_close(built["cost"], 2 * RATE)
    assert_close(built["turnover"], 1.0)
    assert built["turnover_conversion_factor"] == 0.5
    replaced = rebalance_economics(target, pd.Series({"C": 1.0, "D": -1.0}),
                                   gross_return=0.0, single_side_rate=RATE)
    assert_close(replaced["total_abs_weight_change"], 4.0)
    assert_close(replaced["cost"], 4 * RATE)
    assert_close(replaced["turnover"], 2.0)
    # This is the mutation the calibration must catch: truncating the union to
    # target.index silently drops the pre-trade-only stocks and halves U.
    truncated = total_abs_weight_change(
        target, pd.Series({"C": 1.0, "D": -1.0}).reindex(target.index, fill_value=0.0))
    assert_close(truncated, 2.0)
    assert truncated != replaced["total_abs_weight_change"]
    residual = total_abs_weight_change(target, pd.Series({"A": 1.0, "B": -1.0, "Z": 0.5}))
    assert_close(residual, 0.5)
    assert_close(total_abs_weight_change(target, pd.Series({"A": 1.0, "B": -1.0, "Z": 0.5})
                                         .reindex(target.index, fill_value=0.0)), 0.0)
