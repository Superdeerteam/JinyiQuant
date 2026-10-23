"""Opening-history recovery must be causal, bounded and neutral for usable inputs."""

import json
import re
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from jinyiquant.config import ModelConfig
from jinyiquant.data import build_batch, preprocess_bars, validate_keys
from jinyiquant.io import file_sha256, write_json
from modified_tlob import submission
from modified_tlob.adapter import TLOBAdapter
from modified_tlob.schema import lob_columns


def history_case():
    data = {"feature_columns": list(lob_columns(3)), "lookback": 8, "min_history": 1,
            "missing_policy": "ffill", "max_ffill_steps": 2, "missing_sentinel": -1,
            "price_scale": 100, "history_days": 10, "sample_times": ["09:30", "10:00"],
            "label_horizon_minutes": 30}
    rows = []
    for instrument, dates in (
        ("RECOVER", pd.date_range("2024-02-08 14:23", periods=8, freq="min")),
        ("HEALTHY", pd.date_range("2024-02-19 09:23", periods=8, freq="min")),
        ("RECOVER", pd.to_datetime(["2024-02-19 09:59"])),
    ):
        for index, date in enumerate(dates):
            rows.append({"date": date, "instrument": instrument,
                         **{name: 1000 + index * 3 + j for j, name in enumerate(data["feature_columns"])}})
    bars = pd.DataFrame(rows)
    keys = validate_keys(pd.DataFrame([
        {"date": pd.Timestamp(f"2024-02-19 {clock}"), "instrument": name}
        for clock in data["sample_times"] for name in ("HEALTHY", "RECOVER", "NEVER")]))
    return bars, keys, data


class Query:
    def __init__(self, bars, pool):
        self.bars, self.pool, self.calls = bars, pool, []

    def __call__(self, sql, *, filters):
        assert not any(value in sql.lower() for value in ("target", "vwap", "label"))
        table = re.search(r"FROM (\w+)", sql).group(1)
        assert table in {"bars", "pool"}
        first, last = map(pd.Timestamp, filters["date"])
        self.calls.append({"sql": sql, "filters": filters})
        columns = sql.split(" FROM ")[0].removeprefix("SELECT ").split(", ")
        frame = self.pool if table == "pool" else self.bars
        frame = frame.loc[frame.date.between(first, last), columns].copy()
        if "QUALIFY" in sql:
            assert "WHERE date >= TIMESTAMP" in sql and "AND date < TIMESTAMP" in sql
            assert sql.index("WHERE") < sql.index("QUALIFY")
            assert len(filters["instrument"]) <= 50
            frame = frame.loc[frame.date.lt(last) & frame.instrument.isin(filters["instrument"])]
            limit = int(re.search(r"<= (\d+)$", sql).group(1))
            frame = frame.sort_values("date").groupby("instrument", sort=False).tail(limit)
        return SimpleNamespace(df=lambda: frame.sample(frac=1, random_state=17).reset_index(drop=True))


def recovery(query, data, warm, *, deadline=None):
    return submission.HistoryRecovery(query, "bars", data, warm,
        {"max_backfill_days": 366, "instrument_batch_size": 50},
        time.monotonic() + 60 if deadline is None else deadline,
        SimpleNamespace(event=lambda *args, **kwargs: None))


def test_recovery_crosses_holiday_caches_quotes_and_preserves_usable_windows():
    bars, keys, data = history_case()
    first = keys.date.min()
    warm = first - pd.Timedelta(days=data["history_days"])
    prepared = preprocess_bars(bars, data)
    query = Query(bars, keys)
    handler = recovery(query, data, warm)
    for date, group in keys.groupby("date", sort=True):
        base = submission.section_history(prepared.loc[prepared.date.ge(warm)], group, data["lookback"])
        before = build_batch(base, group, data)
        after = handler.recover(base, before)
        usable = before.effective_rows.sum(axis=1) >= data["min_history"]
        np.testing.assert_array_equal(after.x[usable], before.x[usable])
        np.testing.assert_array_equal(after.observed[usable], before.observed[usable])
        np.testing.assert_array_equal(after.history_dates[usable], before.history_dates[usable])
        if date == first:
            expected = build_batch(prepared, group, data)
            np.testing.assert_array_equal(after.x, expected.x)
            np.testing.assert_array_equal(after.history_dates, expected.history_dates)
            assert after.coverage["usable_samples"] == 2
            # Repeating the same opening consumes cached rows, not another query.
            again = handler.recover(base, before)
            np.testing.assert_array_equal(again.x, after.x)
    assert len(query.calls) == 1
    assert set(query.calls[0]["filters"]["instrument"]) == {"RECOVER", "NEVER"}
    assert pd.Timestamp(query.calls[0]["filters"]["date"][1]) == warm
    assert handler.cache["RECOVER"].date.lt(warm).all()
    # Prices were converted once: cached 1000-cent ask becomes 10 yuan.
    assert handler.cache["RECOVER"].iloc[0].ask_price1 == 10


def test_all_missing_full_window_does_not_pull_older_rows_outside_the_lookback():
    bars, keys, data = history_case()
    group = keys.loc[keys.date.eq(keys.date.min()) & keys.instrument.eq("HEALTHY")]
    prepared = preprocess_bars(bars.loc[bars.instrument.eq("HEALTHY")], data)
    prepared[data["feature_columns"]] = np.nan
    before = build_batch(prepared, group, data)
    query = Query(bars, keys)
    after = recovery(query, data, pd.Timestamp("2024-02-09 09:30")).recover(prepared, before)
    assert after is before and not query.calls


def test_backfill_query_failure_and_deadline_are_not_hidden_as_mean_fallback():
    bars, keys, data = history_case()
    group = keys.loc[keys.date.eq(keys.date.min())]
    empty = preprocess_bars(bars.iloc[:0], data)
    batch = build_batch(empty, group, data)
    def unavailable(*args, **kwargs):
        raise RuntimeError("provider unavailable")
    handler = recovery(unavailable, data, pd.Timestamp("2024-02-09 09:30"))
    with pytest.raises(RuntimeError, match="provider unavailable"):
        handler.recover(empty, batch)
    handler = recovery(unavailable, data, pd.Timestamp("2024-02-09 09:30"), deadline=0)
    with pytest.raises(TimeoutError, match="history recovery"):
        handler.recover(empty, batch)


def make_package(tmp_path, data, monkeypatch, *, strict=False):
    config = ModelConfig(kind="modified_tlob", hidden_dim=8, num_layers=2, batch_size=3,
                         device="cpu", cpu_threads=1, target_scaling="train_std")
    model = TLOBAdapter(config, data, seed=11)
    model.trained, model.intercept, model.target_scale = True, .00123, .003
    model.export_text(tmp_path / "modified_tlob_weights.json")
    contract = {"weights_sha256": file_sha256(tmp_path / "modified_tlob_weights.json"), "device": "cpu",
                "platform_price_scale": data["price_scale"], "min_coverage": .99, "wall_seconds": 60}
    if not strict:
        contract["history_policy"] = {"mode": "recover_then_mean", "max_backfill_days": 366,
                                      "instrument_batch_size": 50}
    write_json(tmp_path / "submission_config.json", contract)
    monkeypatch.setattr(submission, "capture_source", lambda _: {"scope": "test"})
    monkeypatch.setattr(submission, "environment", lambda: {"determinism": {}, "dependencies": {}})
    monkeypatch.setattr(TLOBAdapter, "fit", lambda *args, **kwargs: pytest.fail("Inference trained a model"))
    return model, contract


def test_platform_opening_recovers_then_uses_saved_mean_without_dropping_keys(tmp_path, monkeypatch):
    bars, keys, data = history_case()
    model, contract = make_package(tmp_path, data, monkeypatch)
    prepared = preprocess_bars(bars, data)
    start, end = keys.date.min(), keys.date.max()
    warm = start - pd.Timedelta(days=data["history_days"])
    before = build_batch(prepared.loc[prepared.date.ge(warm)], keys, data)
    old_scores = model.predict(before)
    result = submission.predict_range({"bar1m": "bars", "instruments": "pool"}, start, end,
                                      package_root=tmp_path, query=Query(bars, keys))
    assert result[["date", "instrument"]].equals(keys)
    usable = before.effective_rows.sum(axis=1) >= data["min_history"]
    np.testing.assert_array_equal(result.factor.to_numpy()[usable], old_scores[usable])
    assert result.loc[result.instrument.eq("NEVER"), "factor"].eq(model.intercept).all()
    opening = keys.loc[keys.date.eq(start)]
    expected = model.predict(build_batch(prepared, opening, data))
    np.testing.assert_array_equal(result.loc[result.date.eq(start), "factor"].to_numpy(), expected)
    assert file_sha256(tmp_path / "modified_tlob_weights.json") == contract["weights_sha256"]
    manifest = json.loads(next((tmp_path / "inference_runs").glob("*/manifest.json")).read_text())
    assert manifest["status"] == "completed"
    assert manifest["history_handling"]["recovered_samples"] == 1
    assert manifest["history_handling"]["fallback_samples"] == 2
    assert manifest["history_handling"]["data_quality"] == "fallback_used"


@pytest.mark.parametrize("later_available", [False, True])
def test_initial_empty_opening_is_neutral_but_entirely_empty_source_fails(tmp_path, monkeypatch, later_available):
    bars, keys, data = history_case()
    model, _ = make_package(tmp_path, data, monkeypatch)
    bars = bars.loc[bars.date.gt(keys.date.min())] if later_available else bars.iloc[:0]
    def call():
        return submission.predict_range({"bar1m": "bars", "instruments": "pool"},
            keys.date.min(), keys.date.max(), package_root=tmp_path, query=Query(bars, keys))
    if later_available:
        result = call()
        assert result.loc[result.date.eq(keys.date.min()), "factor"].eq(model.intercept).all()
    else:
        with pytest.raises(ValueError, match="No usable historical quotes in the entire prediction range"):
            call()
        manifest = json.loads(next((tmp_path / "inference_runs").glob("*/manifest.json")).read_text())
        assert manifest["status"] == "failed" and manifest["history_handling"]["usable_samples"] == 0


def test_legacy_strict_coverage_contract_remains_strict(tmp_path, monkeypatch):
    bars, keys, data = history_case()
    make_package(tmp_path, data, monkeypatch, strict=True)
    with pytest.raises(ValueError, match=r"Insufficient history coverage.*usable=1/3.*required=0.99"):
        submission.predict_range({"bar1m": "bars", "instruments": "pool"}, keys.date.min(), keys.date.max(),
                                 package_root=tmp_path, query=Query(bars, keys))
