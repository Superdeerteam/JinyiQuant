"""Coverage must not hide short, sparse, filled or stale source windows."""

import numpy as np
import pandas as pd

from jinyiquant.data import build_batch, history_diagnostics


def test_history_reports_raw_observations_separately_from_padding_and_fill():
    keys = pd.DataFrame({"date": pd.to_datetime(["2023-08-02 10:00"] * 3),
                         "instrument": ["A", "B", "C"]})
    bars = pd.DataFrame({"date": pd.to_datetime(["2023-08-01 14:59", "2023-08-01 15:00",
                                                "2023-08-02 09:59", "2023-08-02 10:00"]),
                         "instrument": ["A", "A", "A", "B"],
                         "close": [1., np.nan, np.nan, 2.], "volume": [2., 3., np.nan, 3.]})
    config = {"feature_columns": ["close", "volume"], "lookback": 4, "min_history": 1,
              "missing_policy": "ffill", "max_ffill_steps": 2}
    batch = build_batch(bars, keys, config)
    quality = history_diagnostics(batch).set_index("instrument")
    assert batch.coverage["coverage"] == 2 / 3
    assert quality.loc["A", "history_row_count"] == 3
    assert quality.loc["A", "effective_history_length"] == 2
    assert quality.loc["A", "complete_history_length"] == 1
    assert quality.loc["A", "missing_fraction"] == 5 / 8
    assert quality.loc["A", "source_missing_fraction"] == 0.5
    assert quality.loc["A", "latest_bar_age_minutes"] == 1
    assert quality.loc["A", "latest_effective_age_minutes"] == 1140
    assert quality.loc["A", "history_span_minutes"] == 1140
    assert quality.loc["B", "effective_history_length"] == 1
    assert quality.loc["B", "missing_fraction"] == .75
    assert quality.loc["B", "latest_effective_age_minutes"] == 0
    assert quality.loc["C", "effective_history_length"] == 0
    assert quality.loc["C", "missing_fraction"] == 1
    assert pd.isna(quality.loc["C", "latest_effective_age_minutes"])
    assert pd.isna(quality.loc["C", "source_missing_fraction"])
    assert not quality.full_observed_window.any()
    assert batch.coverage["history_quality"]["full_observed_window_samples"] == 0
    # Filling can create finite values without increasing the effective source length.
    assert np.isfinite(batch.x).all()
    assert batch.x[0, -1, 0] == 1


def test_complete_window_and_missing_window_have_distinct_quality():
    keys = pd.DataFrame({"date": pd.to_datetime(["2023-08-02 10:00"]), "instrument": ["A"]})
    bars = pd.DataFrame({"date": pd.date_range("2023-08-02 09:57", periods=4, freq="min"),
                         "instrument": "A", "close": 1.})
    config = {"feature_columns": ["close"], "lookback": 4}
    full = history_diagnostics(build_batch(bars, keys, config)).iloc[0]
    missing = history_diagnostics(build_batch(bars.assign(close=np.nan), keys, config)).iloc[0]
    assert full.full_observed_window and full.missing_fraction == 0
    assert full.latest_effective_age_minutes == 0 and full.effective_history_length == 4
    assert missing.history_row_count == 4 and missing.effective_history_length == 0
    assert missing.source_missing_fraction == 1 and pd.isna(missing.latest_effective_age_minutes)
