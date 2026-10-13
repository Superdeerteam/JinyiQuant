import json

import numpy as np
import pandas as pd
import pytest

from jinyiquant.evaluation import align_predictions, evaluate

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
    json.dumps(metrics, allow_nan=False)


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
    assert daily["turnover"].iloc[0] == pytest.approx(2)
    assert metrics["day_count"] == 1
    assert metrics["cross_section_count"] == 2


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
