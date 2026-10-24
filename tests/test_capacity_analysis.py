"""Behavior checks for the capacity-study comparison builder."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def analysis_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_capacity_20261024", ROOT / "scripts/analyze_capacity_20261024.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def daily(dates, **columns):
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "rank_ic": 0.0, "ic": 0.0,
                          "long_short_return": 0.0, "turnover": 1.0, "sample_count": 10})
    for name, values in columns.items():
        frame[name] = values
    return frame


def test_pooled_daily_concatenates_windows_chronologically_and_rejects_bad_days():
    module = analysis_module()
    frames = {("w2", "v"): daily(["2024-01-02", "2024-01-03"]),
              ("w1", "v"): daily(["2023-11-01", "2023-11-02"])}
    pooled = module.pooled_daily(frames, ["w1", "w2"], "v")
    assert pooled.window.tolist() == ["w1", "w1", "w2", "w2"]
    assert pooled.date.is_monotonic_increasing
    with pytest.raises(ValueError, match="not chronological"):
        module.pooled_daily(frames, ["w2", "w1"], "v")
    frames[("w2", "v")] = daily(["2023-11-02"])
    with pytest.raises(ValueError, match="duplicate days"):
        module.pooled_daily(frames, ["w1", "w2"], "v")


def test_paired_delta_is_candidate_minus_reference_and_requires_shared_days():
    module = analysis_module()
    dates = ["2023-11-01", "2023-11-02"]
    frames = {("w1", "candidate"): daily(dates, rank_ic=[0.3, 0.4]),
              ("w1", "reference"): daily(dates, rank_ic=[0.1, 0.5])}
    delta = module.paired_delta(frames, ["w1"], "candidate", "reference", "rank_ic")
    assert delta.tolist() == pytest.approx([0.2, -0.1])
    frames[("w1", "reference")] = daily(dates[:1], rank_ic=[0.1])
    with pytest.raises(ValueError, match="do not share prediction days"):
        module.paired_delta(frames, ["w1"], "candidate", "reference", "rank_ic")


def test_pooled_metrics_weights_days_equally_and_reports_the_worst_day():
    module = analysis_module()
    pooled = pd.DataFrame({
        "window": ["w1", "w1", "w2"],
        "date": pd.to_datetime(["2023-11-01", "2023-11-02", "2024-01-02"]),
        "rank_ic": [0.1, 0.2, 0.6], "ic": [0.0, 0.0, 0.0],
        "long_short_return": [0.001, -0.002, 0.003],
        "turnover": [1.0, 1.1, 1.2], "sample_count": [10, 20, 30]})
    result = module.pooled_metrics(pooled, 252)
    assert result["days"] == 3
    assert result["rank_ic_mean"] == pytest.approx(0.3)
    assert result["long_short_return_mean"] == pytest.approx((0.001 - 0.002 + 0.003) / 3)
    assert result["worst_day"] == "2023-11-02"
    assert result["worst_day_window"] == "w1"
    assert result["min_daily_long_short_return"] == pytest.approx(-0.002)
    assert result["samples_per_window"] == {"w1": 30, "w2": 30}


def test_finite_keeps_the_artifact_json_safe():
    module = analysis_module()
    assert module.finite(float("nan")) is None
    assert module.finite(np.inf) is None
    assert module.finite(np.float64(1.5)) == 1.5
    assert module.finite(np.int64(3)) == 3
    assert module.finite(True) is True
