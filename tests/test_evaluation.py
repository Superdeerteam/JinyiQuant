import json

import numpy as np
import pandas as pd
import pytest

from jinyiquant.evaluation import EVALUATION_VERSION, align_predictions, evaluate
from jinyiquant.io import digest

CONFIG = {"quantile": 0.25, "periods_per_year": 252, "min_cross_section": 3}


def frames(scores=(1, 2, 3, 4), targets=(0.01, 0.02, 0.03, 0.04), days=2):
    keys = pd.DataFrame([
        {"date": f"2025-01-{day + 2:02} 10:00:00", "instrument": instrument}
        for day in range(days) for instrument in ("A", "B", "C", "D")
    ])
    return keys.assign(score=np.tile(scores, days).astype(float)), keys.assign(target=np.tile(targets, days).astype(float))


def test_alignment_uses_keys_not_input_positions():
    predictions, targets = frames()
    metrics, daily = evaluate(predictions.iloc[::-1], targets.sample(frac=1, random_state=3), CONFIG)
    assert metrics["rank_ic_mean"] == pytest.approx(1)
    assert metrics["ic_mean"] == pytest.approx(1)
    assert metrics["long_short_return_mean"] == pytest.approx(0.03)
    assert metrics["turnover_mean"] == 0
    assert metrics["long_short_sharpe"] is None
    assert metrics["local_approximation"] is True
    assert daily["sample_count"].tolist() == [4, 4]
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "extra", "nan", "infinity", "label_in_prediction"])
def test_bad_predictions_fail_without_silent_dropping(problem):
    predictions, targets = frames()
    if problem == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[:1]])
    elif problem == "missing":
        predictions = predictions.iloc[:-1]
    elif problem == "extra":
        predictions.loc[0, "instrument"] = "X"
    elif problem == "nan":
        predictions.loc[0, "score"] = np.nan
    elif problem == "infinity":
        predictions.loc[0, "score"] = np.inf
    else:
        predictions["target"] = 42
    with pytest.raises(ValueError):
        align_predictions(predictions, targets)


def test_missing_targets_are_not_imputed():
    predictions, targets = frames()
    targets.loc[0, "target"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        evaluate(predictions, targets, CONFIG)


def test_constant_score_is_undefined_ic_and_neutral_portfolio():
    predictions, targets = frames(scores=(7, 7, 7, 7))
    metrics, daily = evaluate(predictions, targets, CONFIG)
    assert metrics["rank_ic_mean"] is None
    assert metrics["ic_mean"] is None
    assert metrics["undefined_rank_ic_count"] == 2
    assert daily["rank_ic"].isna().all()
    assert daily["long_short_return"].eq(0).all()
    assert metrics["rank_ic_ir"] is None
    assert daily["eligible_rank_ic_count"].tolist() == [1, 1]
    assert daily["valid_rank_ic_count"].tolist() == [0, 0]
    for column in ("rank_ic", "ic"):
        coverage = metrics["correlation_coverage"][column]
        assert coverage["eligible_section_count"] == 2
        assert coverage["valid_section_count"] == 0
        assert coverage["valid_sections_sha256"] == digest([])
    json.dumps(metrics, allow_nan=False)


def test_partial_intraday_coverage_retains_timestamp_identity_and_daily_counts():
    predictions, targets = frames(days=1)
    constant_predictions = predictions.assign(date="2025-01-02 10:30:00", score=7)
    later_targets = targets.assign(date="2025-01-02 10:30:00")
    predictions = pd.concat([constant_predictions, predictions]).sample(frac=1, random_state=4)
    targets = pd.concat([targets, later_targets]).sample(frac=1, random_state=5)
    metrics, daily = evaluate(predictions, targets, CONFIG)
    assert daily["rank_ic"].tolist() == pytest.approx([1])
    assert daily["cross_section_count"].tolist() == [2]
    for column in ("rank_ic", "ic"):
        assert daily[f"eligible_{column}_count"].tolist() == [2]
        assert daily[f"valid_{column}_count"].tolist() == [1]
        assert metrics["correlation_coverage"][column] == {
            "eligible_section_count": 2, "valid_section_count": 1,
            "eligible_sections_sha256": digest(["2025-01-02T10:00:00", "2025-01-02T10:30:00"]),
            "valid_sections_sha256": digest(["2025-01-02T10:00:00"]),
        }
    ordered_metrics, _ = evaluate(predictions.iloc[::-1], targets.iloc[::-1], CONFIG)
    assert metrics["correlation_coverage"] == ordered_metrics["correlation_coverage"]


def test_constant_targets_are_excluded_from_eligible_correlation_populations():
    predictions, targets = frames(days=1)
    predictions = pd.concat([predictions, predictions.assign(date="2025-01-02 10:30:00", score=7)])
    targets = pd.concat([targets, targets.assign(date="2025-01-02 10:30:00", target=0.02)])
    metrics, daily = evaluate(predictions, targets, CONFIG)
    assert metrics["cross_section_count"] == 2
    assert metrics["rank_ic_mean"] == pytest.approx(1)
    assert metrics["undefined_rank_ic_count"] == 1
    for column in ("rank_ic", "ic"):
        assert daily[f"eligible_{column}_count"].tolist() == [1]
        assert daily[f"valid_{column}_count"].tolist() == [1]
        assert metrics["correlation_coverage"][column] == {
            "eligible_section_count": 1, "valid_section_count": 1,
            "eligible_sections_sha256": digest(["2025-01-02T10:00:00"]),
            "valid_sections_sha256": digest(["2025-01-02T10:00:00"]),
        }


def test_rank_and_linear_correlation_eligibility_use_their_own_target_representation():
    predictions, targets = frames(targets=(0, 1e-300, 2e-300, 3e-300), days=1)
    metrics, daily = evaluate(predictions, targets, CONFIG)
    assert metrics["rank_ic_mean"] == pytest.approx(1)
    assert metrics["ic_mean"] is None
    assert daily["eligible_rank_ic_count"].tolist() == [1]
    assert daily["eligible_ic_count"].tolist() == [0]
    assert metrics["correlation_coverage"]["ic"] == {
        "eligible_section_count": 0, "valid_section_count": 0,
        "eligible_sections_sha256": digest([]), "valid_sections_sha256": digest([]),
    }


def test_boundary_ties_share_weight_independent_of_instrument_order():
    predictions, targets = frames(scores=(1, 1, 4, 4), targets=(0, 0.02, 0.04, 0.08))
    metrics, _ = evaluate(predictions, targets, CONFIG)
    assert metrics["long_return_mean"] == pytest.approx(0.06)
    assert metrics["short_return_mean"] == pytest.approx(0.01)
    # Average-rank Pearson, rather than a formula that silently assumes no ties.
    assert metrics["rank_ic_mean"] == pytest.approx(0.8944271909999159)


def test_daily_statistics_weight_timestamps_equally_and_measure_turnover():
    p, t = frames(days=1)
    second_p = p.assign(date="2025-01-02 10:30:00", score=[4, 3, 2, 1])
    second_t = t.assign(date="2025-01-02 10:30:00")
    metrics, daily = evaluate(pd.concat([p, second_p]), pd.concat([t, second_t]), CONFIG)
    assert daily["rank_ic"].iloc[0] == pytest.approx(0)
    assert daily["long_short_return"].iloc[0] == pytest.approx(0)
    # U is drift-aware: adjacent labels drift the pre-trade weights by the previous label.
    expected_u = 2.04 / 1.03 + 2.07 / 1.03
    assert daily["total_abs_weight_change"].iloc[0] == pytest.approx(expected_u)
    assert daily["turnover"].iloc[0] == pytest.approx(expected_u / 2)
    assert metrics["turnover_mean"] == pytest.approx(expected_u / 2)
    assert metrics["day_count"] == 1
    assert metrics["cross_section_count"] == 2
    assert metrics["undefined_turnover_count"] == 1
    assert metrics["rebalance_count"] == 1
    assert metrics["evaluation_version"] == EVALUATION_VERSION
    assert daily["evaluation_version"].eq(EVALUATION_VERSION).all()
    json.dumps(metrics, allow_nan=False)


def test_first_turnover_and_one_day_ratios_remain_undefined():
    p, t = frames(days=1)
    metrics, daily = evaluate(p, t, CONFIG)
    assert metrics["turnover_mean"] is None
    assert metrics["rank_ic_ir"] is None
    assert metrics["long_short_sharpe"] is None
    assert daily["turnover"].isna().all()


def test_too_small_section_is_an_explicit_failure():
    p, t = frames(days=1)
    with pytest.raises(ValueError, match="cross section"):
        evaluate(p.iloc[:2], t.iloc[:2], CONFIG)
