"""Behavior checks for the platform-style five-metric report helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.report_platform_metrics import COMPONENTS, composite, percentile, stress_buckets


def test_percentile_direction_and_ties():
    values = np.array([1.0, 2.0, 3.0])
    assert np.allclose(percentile(values, higher=True), [0.0, 0.5, 1.0])
    assert np.allclose(percentile(values, higher=False), [1.0, 0.5, 0.0])
    assert np.allclose(percentile(np.array([2.0, 2.0, 2.0]), higher=True), [0.5, 0.5, 0.5])


def test_composite_is_the_mean_percentile_and_gap_decomposes():
    frame = pd.DataFrame({
        "ic": [0.05, 0.06, 0.07, 0.08],
        "ic_ir": [1.0, 2.0, 3.0, 4.0],
        "sharpe": [10.0, 20.0, 30.0, 40.0],
        "stress": [0.01, 0.02, 0.03, 0.04],
        "turnover": [1.4, 1.3, 1.2, 1.1],
    })
    scored = composite(frame.copy())
    assert np.allclose(scored["composite"], scored[[f"p_{name}" for name in COMPONENTS]].mean(axis=1))
    assert np.allclose(scored["gap_to_target"], 0.85 - scored["composite"])
    contributions = scored[[f"gap_contrib_{name}" for name in COMPONENTS]].sum(axis=1)
    assert np.allclose(contributions, scored["gap_to_target"])
    assert scored.loc[3, "p_turnover"] == 1.0
    assert scored.loc[0, "p_turnover"] == 0.0


def test_stress_buckets_cover_every_scored_day():
    daily = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        "rank_ic": [0.01, 0.02, 0.03, 0.04],
    })
    conditions = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        "vol_ratio": [0.5, 1.0, 1.5, 0.9],
        "amount_ratio": [0.8, 1.2, 0.7, 1.1],
    })
    buckets = stress_buckets(daily, conditions)
    assert buckets["high_vol"]["days"] + buckets["low_vol"]["days"] == len(daily)
    assert buckets["high_amount"]["days"] + buckets["low_amount"]["days"] == len(daily)
    assert buckets["high_vol"]["days"] == 2
    assert buckets["high_amount"]["days"] == 2
    with pytest.raises(ValueError):
        stress_buckets(daily, conditions.iloc[:-1])
