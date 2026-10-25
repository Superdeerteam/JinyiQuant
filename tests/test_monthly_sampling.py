from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from jinyiquant.config import DataConfig, MonthlySampling, RunConfig
from jinyiquant.data import (
    build_batch,
    complete_trading_calendar,
    load_data,
    monthly_sampled_days,
    previous_trading_days,
)

DAYS = ["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06", "2022-01-07", "2022-01-10",
        "2022-01-11", "2022-01-12", "2022-01-13", "2022-01-14",
        "2022-02-07", "2022-02-08", "2022-02-09", "2022-02-10", "2022-02-11", "2022-02-14",
        "2022-03-01"]
TIMES = ["09:30", "10:00", "10:30"]


def config(**kwargs):
    defaults = dict(feature_columns=["close", "volume"], lookback=3, min_history=1,
                    missing_policy="ffill", max_ffill_steps=1, sample_times=["10:00", "10:30"],
                    price_scale=1.0, missing_sentinel=None, source="test", version="v1", history_days=1,
                    bars_path=None, labels_path=None, instruments_path=None, instrument_map_path=None,
                    allowed_start="2020-01-01", allowed_end="2025-01-01")
    return SimpleNamespace(**(defaults | kwargs))


def sampling(days_per_month=2):
    return SimpleNamespace(start="2022-01-01", end="2022-03-01", days_per_month=days_per_month)


def frames():
    stamps = [pd.Timestamp(f"{day} {clock}") for day in DAYS for clock in TIMES]
    bars = pd.DataFrame({"date": stamps, "instrument": "A",
                         "close": np.arange(len(stamps), dtype=float), "volume": 1.0})
    label_stamps = [pd.Timestamp(f"{day} {clock}") for day in DAYS for clock in TIMES[1:]]
    labels = pd.DataFrame({"date": label_stamps, "instrument": "A",
                           "target": np.arange(len(label_stamps), dtype=float)})
    universe = labels[["date", "instrument"]].copy()
    return bars, labels, universe


def write_inputs(root, bars, labels, universe):
    paths = {"bars_path": root / "bars", "labels_path": root / "labels",
             "instruments_path": root / "universe.csv"}
    paths["bars_path"].mkdir()
    paths["labels_path"].mkdir()
    for month, frame in bars.groupby(bars.date.dt.strftime("%Y%m")):
        frame.to_csv(paths["bars_path"] / f"{month}.0.csv", index=False)
    for month, frame in labels.groupby(labels.date.dt.strftime("%Y-%m")):
        frame.to_csv(paths["labels_path"] / f"labels_{month}.csv", index=False)
    universe.to_csv(paths["instruments_path"], index=False)
    return paths


def test_monthly_sampled_days_are_centered_stratified_and_deterministic():
    calendar = pd.DatetimeIndex(pd.to_datetime(DAYS))
    days, months = monthly_sampled_days(calendar, "2022-01-01", "2022-03-01", 2)
    assert [str(day.date()) for day in days] == ["2022-01-05", "2022-01-12", "2022-02-08", "2022-02-11"]
    assert [month["month"] for month in months] == ["2022-01", "2022-02"]
    assert months[0]["complete_trading_days"] == 10 and months[0]["indices"] == [2, 7]
    assert months[1]["complete_trading_days"] == 6 and months[1]["indices"] == [1, 4]
    again, _ = monthly_sampled_days(calendar, "2022-01-01", "2022-03-01", 2)
    assert days.equals(again)


def test_monthly_sampling_fails_closed_on_short_months_and_misaligned_ranges():
    calendar = pd.DatetimeIndex(pd.to_datetime(DAYS))
    with pytest.raises(ValueError, match="2022-02 has 6 complete trading days"):
        monthly_sampled_days(calendar, "2022-01-01", "2022-03-01", 7)
    with pytest.raises(ValueError, match="month-aligned"):
        monthly_sampled_days(calendar, "2022-01-04", "2022-03-01", 2)
    with pytest.raises(ValueError, match="month-aligned"):
        monthly_sampled_days(calendar, "2022-01-01", "2022-03-15", 2)


def test_complete_calendar_requires_every_sampling_time():
    bars, labels, _ = frames()
    universe = pd.concat([labels[["date", "instrument"]],
                          pd.DataFrame({"date": [pd.Timestamp("2022-01-17 10:00")], "instrument": "A"})])
    calendar = complete_trading_calendar(universe, ["10:00", "10:30"])
    assert pd.Timestamp("2022-01-17") not in calendar
    assert len(calendar) == len(DAYS)


def test_previous_trading_days_use_the_independent_calendar():
    calendar = pd.DatetimeIndex(pd.to_datetime(DAYS))
    days = pd.DatetimeIndex(pd.to_datetime(["2022-01-05", "2022-02-08"]))
    previous = previous_trading_days(calendar, days)
    assert [str(day.date()) for day in previous] == ["2022-01-04", "2022-02-07"]
    with pytest.raises(ValueError, match="preceding"):
        previous_trading_days(calendar, pd.DatetimeIndex(pd.to_datetime([DAYS[0]])))


def test_sampled_read_keeps_selected_days_context_and_evaluation_rows(tmp_path):
    bars, labels, universe = frames()
    paths = write_inputs(tmp_path, bars, labels, universe)
    cfg = config(**paths, monthly_sampling=sampling(), cache_dir=tmp_path / "cache")
    bundle = load_data(cfg, "2022-01-01", "2022-03-15")

    sampled = ["2022-01-05", "2022-01-12", "2022-02-08", "2022-02-11"]
    context = ["2022-01-04", "2022-01-11", "2022-02-07", "2022-02-10"]
    bar_days = sorted(bundle.bars.date.dt.strftime("%Y-%m-%d").unique().tolist())
    assert bar_days == sorted(sampled + context + ["2022-03-01"])
    assert sorted(bundle.labels.date.dt.strftime("%Y-%m-%d").unique().tolist()) == sorted(sampled + ["2022-03-01"])
    assert sorted(bundle.instruments.date.dt.strftime("%Y-%m-%d").unique().tolist()) == sorted(sampled + ["2022-03-01"])

    audit = bundle.metadata["monthly_sampling"]
    assert audit["days"] == 4 and audit["days_per_month"] == 2
    assert audit["selected_days"] == ["2022-01-05", "2022-01-12", "2022-02-08", "2022-02-11"]
    assert audit["context_days"] == ["2022-01-04", "2022-01-11", "2022-02-07", "2022-02-10"]
    assert [month["month"] for month in audit["months"]] == ["2022-01", "2022-02"]
    assert audit["history_end"].startswith("2022-02-28")
    assert bundle.metadata["bars_start"] == "2022-01-01 00:00:00"

    # A 128-slot window must reach the retained previous trading day.
    keys = bundle.labels.loc[bundle.labels.date.eq(pd.Timestamp("2022-01-05 10:00")), ["date", "instrument"]]
    batch = build_batch(bundle.bars, keys, cfg)
    assert batch.coverage["history_min"] == 3
    assert batch.observed.all()

    # Monthly projections must match the same rows read as one contiguous range.
    full = load_data(config(**paths), "2022-01-01", "2022-03-15")
    expected = full.bars.loc[full.bars.date.dt.strftime("%Y-%m-%d").isin(bar_days)].reset_index(drop=True)
    pd.testing.assert_frame_equal(bundle.bars, expected)


def test_sampled_projection_cache_is_monthly_verified_and_selection_keyed(tmp_path):
    bars, labels, universe = frames()
    paths = write_inputs(tmp_path, bars, labels, universe)
    cfg = config(**paths, monthly_sampling=sampling(), cache_dir=tmp_path / "cache")
    cold = load_data(cfg, "2022-01-01", "2022-03-15")
    assert [event["event"] for event in cold.cache_events].count("cache_written") >= 5
    warm = load_data(cfg, "2022-01-01", "2022-03-15")
    assert {event["event"] for event in warm.cache_events} == {"cache_hit"}
    pd.testing.assert_frame_equal(cold.bars, warm.bars)
    assert cold.metadata == warm.metadata

    # A different day selection must not reuse the cached rows.
    other = load_data(config(**paths, monthly_sampling=sampling(days_per_month=1),
                             cache_dir=tmp_path / "cache"), "2022-01-01", "2022-03-15")
    assert [event["event"] for event in other.cache_events].count("cache_miss") >= 1
    assert other.metadata["monthly_sampling"]["days"] == 2


def test_sampled_read_requires_the_sampling_start(tmp_path):
    bars, labels, universe = frames()
    cfg = config(**write_inputs(tmp_path, bars, labels, universe), monthly_sampling=sampling())
    with pytest.raises(ValueError, match="sampling range"):
        load_data(cfg, "2022-02-01", "2022-03-15")


def test_sampled_read_requires_an_independent_calendar(tmp_path):
    bars, labels, universe = frames()
    paths = write_inputs(tmp_path, bars, labels, universe)
    cfg = config(bars_path=paths["bars_path"], labels_path=paths["labels_path"],
                 monthly_sampling=sampling())
    with pytest.raises(ValueError, match="instruments calendar"):
        load_data(cfg, "2022-01-01", "2022-03-15")


def test_config_requires_month_alignment_and_exact_training_coverage():
    with pytest.raises(ValidationError, match="month-aligned"):
        MonthlySampling(start="2022-01-04", end="2022-03-01")
    with pytest.raises(ValidationError, match="month-aligned"):
        MonthlySampling(start="2022-01-01", end="2022-03-05")
    with pytest.raises(ValidationError, match="start < end"):
        MonthlySampling(start="2022-03-01", end="2022-03-01")

    raw = {"experiment_id": "sampling-test", "hypothesis": "test", "seed": 11,
           "data": {"bars_path": "bars", "labels_path": "labels", "instruments_path": "universe",
                    "source": "test", "version": "v1", "feature_columns": ["close", "volume"],
                    "lookback": 3, "sample_times": ["10:00", "10:30"],
                    "monthly_sampling": {"start": "2022-02-01", "end": "2022-03-01"}},
           "splits": {"train": {"start": "2022-01-01", "end": "2022-03-01"},
                      "validation": {"start": "2022-03-01", "end": "2022-03-10"},
                      "final": {"start": "2022-03-10", "end": "2022-04-01"}}}
    with pytest.raises(ValidationError, match="exactly cover the training split"):
        RunConfig.model_validate(raw)
    raw["data"]["monthly_sampling"] = {"start": "2022-01-01", "end": "2022-03-01"}
    assert RunConfig.model_validate(raw).data.monthly_sampling.days_per_month == 6


def test_data_config_keeps_sampling_inside_the_allowed_interval():
    with pytest.raises(ValidationError, match="allowed development interval"):
        DataConfig(bars_path="bars", labels_path="labels", source="test", version="v1",
                   allowed_start="2022-02-01", allowed_end="2025-01-01",
                   monthly_sampling={"start": "2022-01-01", "end": "2022-03-01"})
