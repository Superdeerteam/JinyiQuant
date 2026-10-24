"""Behavior checks for the frozen-version forward replay helpers."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def replay_module():
    spec = importlib.util.spec_from_file_location(
        "replay_model_versions", ROOT / "scripts/replay_model_versions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_window_keys_keep_only_approved_clocks_inside_the_half_open_window():
    replay = replay_module()
    times = ["09:30", "10:00", "14:30"]
    universe = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01 09:25", "2024-01-01 09:30", "2024-01-01 10:00",
                                "2024-01-03 14:30", "2024-01-04 09:30"]),
        "instrument": ["A", "A", "A", "A", "A"]})
    bundle = SimpleNamespace(universe=universe)
    config = {"sample_times": times, "label_horizon_minutes": 30}
    keys = replay.window_keys(bundle, config, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04"))
    assert keys.date.tolist() == [pd.Timestamp("2024-01-01 09:30"), pd.Timestamp("2024-01-01 10:00"),
                                  pd.Timestamp("2024-01-03 14:30")]
    assert keys.date.dt.strftime("%H:%M").isin(times).all()


def test_daily_counts_separate_missing_answers_from_history_fallback():
    replay = replay_module()
    keys = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02 09:30", "2024-01-02 09:30", "2024-01-02 10:00",
                                "2024-01-02 10:00", "2024-01-03 09:30"]),
        "instrument": ["A", "B", "A", "B", "A"]})
    targets = keys.iloc[[0, 1, 2, 4]].reset_index(drop=True)
    fallback = keys.iloc[[0]].assign(no_history=True)
    counts = replay.daily_counts(keys, targets, fallback).set_index("date")
    assert counts.loc["2024-01-02", "expected_sample_count"] == 4
    assert counts.loc["2024-01-02", "scored_sample_count"] == 3
    assert counts.loc["2024-01-02", "missing_answer_count"] == 1
    assert counts.loc["2024-01-02", "no_history_count"] == 1
    assert counts.loc["2024-01-03", "missing_answer_count"] == 0
    assert counts.loc["2024-01-03", "no_history_count"] == 0


def test_pair_list_uses_plan_pairs_and_rejects_unknown_versions():
    replay = replay_module()
    default_names = {"v0_july_submitted", "v0_local_oct", "v1_oct_csz", "v2_60d_csz"}
    plan = {"versions": [{"name": name} for name in default_names]}
    assert replay.pair_list(plan) == list(replay.DEFAULT_PAIRS)
    custom = {"versions": [{"name": "a"}, {"name": "b"}], "pairs": [["a", "b"]]}
    assert replay.pair_list(custom) == [("a", "b")]
    with pytest.raises(ValueError, match="Invalid comparison pair"):
        replay.pair_list({"versions": [{"name": "a"}], "pairs": [["a", "missing"]]})


def test_block_delta_interval_covers_the_mean_and_is_degenerate_for_constant_days():
    replay = replay_module()
    rising = pd.DataFrame({"delta": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = replay._block_delta(rising, "delta", 2, 2000, 7)
    assert result["days"] == 5
    assert result["mean"] == pytest.approx(3.0)
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]
    constant = pd.DataFrame({"delta": [0.25] * 10})
    result = replay._block_delta(constant, "delta", 2, 2000, 7)
    assert result["ci_low"] == pytest.approx(0.25)
    assert result["ci_high"] == pytest.approx(0.25)
