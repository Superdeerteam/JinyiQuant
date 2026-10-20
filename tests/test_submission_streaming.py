"""Daily platform reads must preserve the original fixed-weight inference contract."""

import hashlib
import json
import re
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


def _case(policy="ffill"):
    features = list(lob_columns(3))
    data = {
        "feature_columns": features, "lookback": 8, "min_history": 2,
        "missing_policy": policy, "max_ffill_steps": 2, "missing_sentinel": -1,
        "price_scale": 100, "history_days": 2, "sample_times": ["09:30", "10:00"],
        "label_horizon_minutes": 30,
    }
    rows = []

    def add(date, instrument, value=0, missing=False):
        values = {name: (1000 + value + i if "price" in name else 100 + value + i)
                  for i, name in enumerate(features)}
        if missing:
            values = dict.fromkeys(features, np.nan)
        rows.append({"date": pd.Timestamp(date), "instrument": instrument, **values})

    for day_index, day in enumerate(pd.date_range("2022-01-29", "2022-02-04")):
        for minute in range(23, 31):
            add(day + pd.Timedelta(hours=9, minutes=minute), "ACTIVE", day_index * 20 + minute)
        add(day + pd.Timedelta(hours=10), "ACTIVE", day_index * 20 + 40)
        add(day + pd.Timedelta(hours=9, minutes=30), "ALL_MISSING", missing=True)
    for minute in range(31, 39):
        add(f"2022-01-29 09:{minute}", "SUSPENDED", minute)
        add(f"2022-01-29 09:{minute}", "NEW_TO_POOL", minute + 50)
    # A suspended instrument retains January history even after history_days have
    # elapsed. An all-missing new row still occupies its position in the window.
    add("2022-02-04 09:30", "SUSPENDED", missing=True)
    add("2022-01-30 09:30", "NEW_TO_POOL", missing=True)
    add("2022-02-01 00:00", "ACTIVE", 60)
    add("2022-02-04 00:00", "ACTIVE", 70)
    add("2022-01-29 09:29:59", "OUTSIDE_WARM_START", 99)
    add("2022-02-04 10:01", "ACTIVE", 999)
    bars = pd.DataFrame(rows)
    # Both sentinel conversion and bounded filling are part of the equivalence,
    # including a missing entry whose preceding row is in the previous day.
    bars.loc[(bars.instrument == "ACTIVE") & bars.date.eq("2022-02-01 00:00"),
             features[0]] = -1
    bars.loc[(bars.instrument == "ACTIVE") & bars.date.eq("2022-02-01 09:30"),
             features[3]] = np.inf
    instruments = ["ACTIVE", "SUSPENDED", "ALL_MISSING", "NO_HISTORY"]
    keys = []
    for day in pd.date_range("2022-01-31", "2022-02-04"):
        for sample in data["sample_times"]:
            for instrument in instruments + (["NEW_TO_POOL"] if day.day >= 3 and day.month == 2 else []):
                keys.append({"date": pd.Timestamp(f"{day.date()} {sample}"), "instrument": instrument})
    return bars, validate_keys(pd.DataFrame(keys)), data


class _Query:
    """Inclusive, unsorted provider responses with extra boundary rows."""

    def __init__(self, bars, pool):
        self.frames = {"bars": bars, "pool": pool}
        self.calls = []

    def __call__(self, sql, *, filters):
        match = re.fullmatch(r"SELECT (.+) FROM (\w+)", sql)
        assert match is not None
        columns, table = match[1].split(", "), match[2]
        assert table in self.frames, "Inference must never request labels"
        assert len(columns) == len(set(columns))
        assert not {"target", "vwap_return", "label_end"}.intersection(columns)
        start, end = map(pd.Timestamp, filters["date"])
        self.calls.append({"table": table, "columns": columns, "start": start, "end": end})
        frame = self.frames[table]
        # Deliberately over-return adjacent days, as well as the inclusive end
        # boundary. The caller owns the exact half-open intermediate partition.
        frame = frame.loc[frame.date.between(start - pd.Timedelta(days=1),
                                            end + pd.Timedelta(days=1)), columns]
        return SimpleNamespace(df=lambda: frame.sample(frac=1, random_state=17).reset_index(drop=True))


def _assert_batch_equal(actual, expected):
    pd.testing.assert_frame_equal(actual.keys, expected.keys)
    np.testing.assert_array_equal(actual.x, expected.x)
    np.testing.assert_array_equal(actual.observed, expected.observed)
    # assert_array_equal treats corresponding NaT positions as equal; ordinary
    # elementwise equality would incorrectly reject identical absent histories.
    np.testing.assert_array_equal(actual.history_dates, expected.history_dates)
    assert actual.history_feature_count == expected.history_feature_count
    assert actual.coverage == expected.coverage


@pytest.mark.parametrize("policy", ["zero", "ffill"])
@pytest.mark.parametrize("last", ["2022-02-04 10:00", "2022-02-04 00:00"])
def test_daily_windows_match_full_history_with_suspensions_missing_rows_and_boundaries(policy, last):
    bars, keys, data = _case(policy)
    data["sample_times"].append("00:00")
    keys = validate_keys(pd.concat([keys, pd.DataFrame({
        "date": pd.to_datetime(["2022-02-01 00:00", "2022-02-04 00:00"]),
        "instrument": ["ACTIVE", "ACTIVE"],
    })], ignore_index=True))
    warm_start, end = pd.Timestamp("2022-01-29 09:30"), pd.Timestamp(last)
    whole = preprocess_bars(bars.loc[bars.date.between(warm_start, end)], data)
    query = _Query(bars, keys)
    seen_sections, seen_rows = [], 0
    for window, metadata in submission.iter_bar_days(query, "bars", warm_start, end, data):
        assert {"start", "end", "rows", "retained_rows", "preprocessed_bars_sha256"} <= metadata.keys()
        first, stop = pd.Timestamp(metadata["start"]), pd.Timestamp(metadata["end"])
        # A midnight end belongs to the final zero-length day, not the
        # preceding day whose upper boundary is that same midnight.
        inclusive = stop == end and stop < first.normalize() + pd.Timedelta(days=1)
        mask = (whole.date >= first) & (whole.date <= stop if inclusive else whole.date < stop)
        current = whole.loc[mask].reset_index(drop=True)
        history = whole.loc[whole.date < first].groupby("instrument", sort=False).tail(data["lookback"])
        expected_window = validate_keys(pd.concat([history, current], ignore_index=True))
        pd.testing.assert_frame_equal(validate_keys(window), expected_window)
        assert metadata["rows"] == len(current)
        assert metadata["retained_rows"] == len(
            expected_window.groupby("instrument", sort=False).tail(data["lookback"]))
        assert metadata["preprocessed_bars_sha256"] == hashlib.sha256(
            pd.util.hash_pandas_object(current, index=False).to_numpy().tobytes()).hexdigest()
        seen_rows += metadata["rows"]
        eligible = keys.loc[(keys.date >= first) & (keys.date <= stop if inclusive else keys.date < stop)
                            & (keys.date <= end)]
        for date, group in eligible.groupby("date", sort=True):
            selected = window.loc[window.instrument.isin(group.instrument) & (window.date <= date)]
            selected = selected.groupby("instrument", sort=False).tail(data["lookback"])
            actual = build_batch(selected, group, data)
            expected = build_batch(whole, group, data)
            _assert_batch_equal(actual, expected)
            seen_sections.append(date)
    assert seen_rows == len(whole)
    assert seen_sections == sorted(keys.loc[keys.date <= end, "date"].unique())
    assert query.calls[0]["start"] == warm_start
    assert query.calls[-1]["end"] == end
    for call in query.calls:
        assert pd.Timedelta(0) <= call["end"] - call["start"] <= pd.Timedelta(days=1)
        assert call["columns"] == ["date", "instrument", *data["feature_columns"]]


def test_daily_reader_rejects_duplicate_source_keys():
    bars, keys, data = _case()
    duplicate = bars.loc[(bars.instrument == "ACTIVE") & bars.date.eq("2022-02-01 09:30")]
    query = _Query(pd.concat([bars, duplicate], ignore_index=True), keys)
    with pytest.raises(ValueError, match="duplicate"):
        list(submission.iter_bar_days(query, "bars", pd.Timestamp("2022-01-29 09:30"),
                                      pd.Timestamp("2022-02-04 10:00"), data))


def test_platform_streaming_predictions_equal_legacy_fixed_tlob_and_keep_audit(tmp_path, monkeypatch):
    bars, keys, data = _case()
    start, end = keys.date.min(), keys.date.max()
    warm_start = start - pd.Timedelta(days=data["history_days"])
    whole = preprocess_bars(bars.loc[bars.date.between(warm_start, end)], data)
    config = ModelConfig(kind="modified_tlob", hidden_dim=8, num_layers=2, num_heads=1,
                         device="cpu", cpu_threads=1, batch_size=3, target_scaling="train_std")
    model = TLOBAdapter(config, data, seed=29)
    # A fixed synthetic checkpoint exercises the real network and JSON loader
    # without training or reading any competition weights or labels.
    model.trained, model.intercept, model.target_scale = True, 0.00123, 0.003
    weight_path = tmp_path / "modified_tlob_weights.json"
    model.export_text(weight_path)
    write_json(tmp_path / "submission_config.json", {
        "weights_sha256": file_sha256(weight_path), "device": "cpu",
        "platform_price_scale": 100, "min_coverage": 0, "wall_seconds": 60,
    })
    expected = pd.concat([
        batch.keys.assign(factor=model.predict(batch))
        for _, group in keys.groupby("date", sort=True)
        for batch in [build_batch(whole, group, data)]
    ], ignore_index=True)
    calls = []

    def check_bounded_batch(window, group, config):
        date = group.date.iloc[0]
        assert group.date.nunique() == 1
        assert window.date.le(date).all()
        assert set(window.instrument) <= set(group.instrument)
        assert window.groupby("instrument").size().le(data["lookback"]).all()
        actual = build_batch(window, group, config)
        _assert_batch_equal(actual, build_batch(whole, group, data))
        calls.append(date)
        return actual

    monkeypatch.setattr(submission, "build_batch", check_bounded_batch)
    monkeypatch.setattr(submission, "capture_source", lambda _: {"kind": "synthetic_test"})
    monkeypatch.setattr(submission, "environment", lambda: {"determinism": {}, "dependencies": {}})
    monkeypatch.setattr(TLOBAdapter, "fit", lambda *_args, **_kwargs: pytest.fail("Inference trained a model"))
    query = _Query(bars, keys)
    actual = submission.predict_range({"bar1m": "bars", "instruments": "pool"}, start, end,
                                      package_root=tmp_path, query=query)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert calls == sorted(keys.date.unique())
    assert actual.loc[actual.instrument.isin(["NO_HISTORY", "ALL_MISSING"]), "factor"].eq(model.intercept).all()
    assert actual.instrument.eq("NEW_TO_POOL").any()
    assert file_sha256(weight_path) == json.loads((tmp_path / "submission_config.json").read_text())["weights_sha256"]
    assert all(call["end"] - call["start"] <= pd.Timedelta(days=1)
               for call in query.calls if call["table"] == "bars")
    archive = next((tmp_path / "inference_runs").glob("*/manifest.json")).parent
    assert json.loads((archive / "manifest.json").read_text())["status"] == "completed"
    events = [json.loads(line) for line in (archive / "events.jsonl").read_text().splitlines()]
    required = {"schema_version", "timestamp_utc", "level", "stage", "event_type",
                "run_id", "experiment_id", "context"}
    assert events and all(required <= event.keys() for event in events)
    assert events[-1]["context"]["status"] == "completed"
    assert len([e for e in events if e["stage"] == "preprocessing" and e["event_type"] == "completed"]) == len(calls)
    record = json.loads((archive / "data.json").read_text())
    assert record["prediction_keys"] == len(keys)
    assert record["label_input"] is False
