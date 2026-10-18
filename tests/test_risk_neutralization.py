import numpy as np
import pandas as pd
import pytest

from jinyiquant.evaluation import evaluate
from modified_tlob.risk import (
    INDUSTRY_DUMMIES,
    REGRESSORS,
    STYLE_FACTORS,
    portfolio_net_weights,
    previous_exposure_date,
    residualize,
    section_panel,
    trailing_median,
    winsorized_zscore,
)

KEYS = ["date", "instrument"]


def snapshot(seed=0):
    rng = np.random.default_rng(seed)
    instruments = [f"{index:06d}.SZ" for index in range(6)]
    frame = pd.DataFrame({"instrument": instruments})
    for factor in STYLE_FACTORS:
        frame[factor] = rng.normal(size=len(frame))
    for industry in INDUSTRY_DUMMIES:
        frame[industry] = 0
    frame["ret"] = rng.normal(scale=0.01, size=len(frame))
    frame["weights"] = rng.random(len(frame))
    return frame, instruments


def test_previous_exposure_date_is_strictly_before_the_signal_date():
    dates = np.array(["2023-10-30", "2023-10-31", "2023-11-01"], dtype="datetime64[ns]")
    assert previous_exposure_date(dates, "2023-11-01 09:30:00") == pd.Timestamp("2023-10-31")
    assert previous_exposure_date(dates, "2023-11-02 14:30:00") == pd.Timestamp("2023-11-01")
    with pytest.raises(ValueError, match="strictly before"):
        previous_exposure_date(dates, "2023-10-30 10:00:00")
    same_day = np.array(["2023-11-01"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="strictly before"):
        previous_exposure_date(same_day, "2023-11-01 10:00:00")


def test_section_panel_drops_uncovered_names_and_never_reads_ret_or_weights():
    frame, instruments = snapshot()
    scores = list(np.linspace(-1, 1, len(instruments))) + [0.5]
    panel = section_panel(scores, instruments + ["999999.SZ"], frame)
    assert len(panel) == len(instruments)
    assert "ret" not in panel and "weights" not in panel
    assert set(panel.columns) == {"instrument", "score", *REGRESSORS}

    residual = residualize(winsorized_zscore(panel["score"])[0], panel[list(REGRESSORS)])
    changed = frame.copy()
    changed["ret"] = np.linspace(-9, 9, len(changed))
    changed["weights"] = 0.0
    panel_changed = section_panel(scores, instruments + ["999999.SZ"], changed)
    residual_changed = residualize(winsorized_zscore(panel_changed["score"])[0],
                                   panel_changed[list(REGRESSORS)])
    assert np.array_equal(residual[0], residual_changed[0])


def test_residual_is_orthogonal_to_the_regressors_and_linear():
    rng = np.random.default_rng(7)
    count = 240
    design = np.column_stack([
        rng.normal(size=(count, len(STYLE_FACTORS))),
        np.eye(len(INDUSTRY_DUMMIES))[rng.integers(0, len(INDUSTRY_DUMMIES), size=count)],
    ])
    standardized = rng.normal(size=count)
    residual, explained = residualize(standardized, design)
    assert np.allclose(design.T @ residual, 0, atol=1e-10)
    assert 0 <= explained <= 1
    scaled, scaled_explained = residualize(3.0 * standardized + 7.0, design)
    assert np.allclose(scaled, 3.0 * residual, atol=1e-10)
    assert scaled_explained == pytest.approx(explained)


def test_winsorized_zscore_clips_outliers_and_handles_a_constant_section():
    values = np.array([0.001, 0.002, -0.001, 0.0, 5.0])
    standardized, clipped = winsorized_zscore(values)
    assert clipped == 1
    assert standardized.mean() == pytest.approx(0, abs=1e-12)
    assert standardized.std(ddof=0) == pytest.approx(1)
    assert standardized.max() < (5.0 - np.median(values)) / 1.4826 / np.median(
        np.abs(values - np.median(values)))

    constant, clipped = winsorized_zscore(np.full(8, 0.25))
    assert clipped == 0
    assert np.array_equal(constant, np.zeros(8))


def test_portfolio_net_weights_reproduce_the_local_long_short_return():
    rng = np.random.default_rng(11)
    count, per_section = 120, 2
    keys = pd.DataFrame([{"date": f"2025-01-{day + 2:02} 10:00:00", "instrument": f"{index:04d}.SZ"}
                         for day in range(per_section) for index in range(count)])
    scores = rng.normal(size=len(keys))
    scores[::7] = scores[0]  # boundary ties must share weight identically
    targets = rng.normal(scale=0.01, size=len(keys)) + 4 * scores
    predictions = keys.assign(score=scores)
    answer = keys.assign(target=targets)
    _, _, sections = evaluate(predictions, answer, {"quantile": 0.2, "periods_per_year": 252,
                                                    "min_cross_section": 3}, return_sections=True)
    for row, (_, group) in zip(sections.itertuples(), predictions.groupby("date", sort=True)):
        weights = portfolio_net_weights(group["score"], quantile=0.2)
        expected = float(np.dot(weights, answer.loc[group.index, "target"]))
        assert row.long_short_return == pytest.approx(expected, abs=1e-12)


def test_trailing_median_uses_only_the_dates_strictly_before_the_reference():
    dates = np.array(pd.bdate_range("2024-01-01", periods=25), dtype="datetime64[ns]")
    series = pd.Series(np.arange(25, dtype=float), index=pd.DatetimeIndex(dates))
    reference = pd.Timestamp(dates[24])
    baseline = trailing_median(series, dates, before=reference, window=20)
    assert baseline == pytest.approx(np.median(np.arange(4, 24)))

    changed = series.copy()
    changed.iloc[24] = 1e6       # the reference day itself is never used
    changed.iloc[3] = -1e6       # outside the twenty-day window
    assert trailing_median(changed, dates, before=reference, window=20) == pytest.approx(baseline)
    changed.iloc[4:24] += 100.0  # every included day shifts the baseline
    assert trailing_median(changed, dates, before=reference, window=20) == pytest.approx(baseline + 100)
