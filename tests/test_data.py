from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from jinyiquant.config import DataConfig
from jinyiquant.data import build_batch, load_data, select_targets


def config(**kwargs):
    defaults = dict(feature_columns=["close", "volume"], lookback=3, min_history=2,
                    missing_policy="ffill", max_ffill_steps=1, sample_times=["10:00", "10:30"],
                    price_scale=1.0, missing_sentinel=None, source="test", version="v1", history_days=1,
                    bars_path=None, labels_path=None, instruments_path=None, instrument_map_path=None)
    return SimpleNamespace(**(defaults | kwargs))


def bars():
    return pd.DataFrame({"date": pd.to_datetime(["2020-01-02 09:58", "2020-01-02 09:59",
                                               "2020-01-02 10:00", "2020-01-02 10:01"]),
                         "instrument": ["A"] * 4, "close": [10.0, np.nan, np.nan, 1000.0],
                         "volume": [1.0, 2.0, 3.0, 4.0]})


def test_causal_window_fill_mask_and_future_invariance():
    source = bars()
    keys = pd.DataFrame({"date": [pd.Timestamp("2020-01-02 10:00")], "instrument": ["A"]})
    batch = build_batch(source, keys, config())
    assert batch.x.shape == (1, 3, 2)
    np.testing.assert_array_equal(batch.x[0, :, 0], [10.0, 10.0, 0.0])
    np.testing.assert_array_equal(batch.observed[0, :, 0], [True, False, False])
    assert batch.history_dates[0, -1] == np.datetime64("2020-01-02T10:00")
    source.loc[3, ["close", "volume"]] = -123456
    mutated = build_batch(source, keys, config())
    np.testing.assert_array_equal(mutated.x, batch.x)
    np.testing.assert_array_equal(mutated.observed, batch.observed)


def test_longer_lookback_keeps_the_same_trailing_window():
    dates = pd.date_range("2020-01-02 09:55", periods=6, freq="min")
    source = pd.DataFrame({"date": dates, "instrument": ["A"] * 6,
                           "close": np.arange(6.0) + 10.0, "volume": np.arange(6.0) + 1.0})
    keys = pd.DataFrame({"date": [pd.Timestamp("2020-01-02 10:00")], "instrument": ["A"]})
    long = build_batch(source, keys, config(lookback=5, missing_policy="zero"))
    short = build_batch(source, keys, config(lookback=3))
    np.testing.assert_array_equal(long.x[:, -3:], short.x)
    np.testing.assert_array_equal(long.observed[:, -3:], short.observed)
    np.testing.assert_array_equal(long.history_dates[:, -3:], short.history_dates)


def test_missing_stock_retains_key_padding_and_reports_coverage():
    keys = pd.DataFrame({"date": pd.to_datetime(["2020-01-02 10:00"] * 2), "instrument": ["B", "A"]})
    batch = build_batch(bars(), keys, config())
    assert batch.keys["instrument"].tolist() == ["A", "B"]
    assert batch.coverage["coverage"] == 0.5
    assert batch.coverage["min_section_coverage"] == 0.5
    assert batch.coverage["insufficient_history"] == 1
    assert not batch.observed[1].any()
    assert (batch.x[1] == 0).all()
    assert np.isnat(batch.history_dates[1]).all()


def test_duplicate_bar_and_invalid_keys_fail_instead_of_deduplicating():
    keys = bars().loc[[2], ["date", "instrument"]]
    with pytest.raises(ValueError, match="duplicate"):
        build_batch(pd.concat([bars(), bars().iloc[[0]]]), keys, config())
    with pytest.raises(ValueError, match="sampling time"):
        build_batch(bars(), bars().loc[[0], ["date", "instrument"]], config())
    with pytest.raises(ValueError, match="Future label"):
        build_batch(bars(), keys, config(feature_columns=["vwap_return"]))


def test_target_purging_missing_labels_and_inferred_lunch_clock():
    source = pd.DataFrame({"date": pd.to_datetime(["2020-01-02 10:00", "2020-01-02 10:30",
                                                 "2020-01-02 11:30", "2020-01-02 14:30"]),
                           "instrument": ["A"] * 4, "target": [0.1, np.nan, 0.2, 0.3]})
    selected = select_targets(source, {"start": "2020-01-02", "end": "2020-01-02 15:00"})
    assert selected["target"].tolist() == [0.1, 0.2]
    assert selected.iloc[1]["label_end"] == pd.Timestamp("2020-01-02 13:30")
    assert selected.attrs["selection"]["boundary_purged"] == 1
    assert selected.attrs["selection"]["missing_labels"] == 1
    assert "assumed" in selected.attrs["selection"]["label_clock"]


def test_configured_sample_subset_selects_only_requested_times_and_preserves_boundary_counts():
    source = pd.DataFrame({"date": pd.to_datetime(["2020-01-01 10:00", "2020-01-02 10:00",
                                                 "2020-01-02 10:30", "2020-01-02 14:30"]),
                           "instrument": ["A"] * 4, "target": [0.0, 0.1, 0.2, 0.3]})
    split = {"start": "2020-01-02", "end": "2020-01-02 15:00"}
    selected = select_targets(source, split, sample_times=["10:00", "14:30"])
    assert selected["date"].tolist() == [pd.Timestamp("2020-01-02 10:00")]
    assert selected["target"].tolist() == [0.1]
    assert selected.attrs["selection"]["requested"] == 3
    assert selected.attrs["selection"]["selected"] == 1
    assert selected.attrs["selection"]["outside_sample_times"] == 1
    assert selected.attrs["selection"]["boundary_purged"] == 1
    batch = build_batch(bars(), selected[["date", "instrument"]], config(sample_times=["10:00", "14:30"]))
    assert len(batch.keys) == 1
    legacy = select_targets(source, split)
    assert legacy["target"].tolist() == [0.1, 0.2]
    assert legacy.attrs["selection"]["outside_sample_times"] == 0


@pytest.mark.parametrize("time", ["11:01", "11:15", "11:29"])
def test_config_rejects_sampling_times_unsupported_by_inferred_label_clock(time):
    with pytest.raises(ValidationError, match="supported trading-clock time"):
        DataConfig(bars_path="unused", labels_path="unused", source="test", version="v1", sample_times=[time])


@pytest.mark.parametrize("time,end", [("11:00", "11:30"), ("11:30", "13:30"), ("14:30", "15:00")])
def test_configured_boundary_times_match_the_inferred_label_clock(time, end):
    cfg = DataConfig(bars_path="unused", labels_path="unused", source="test", version="v1", sample_times=[time])
    source = pd.DataFrame({"date": [pd.Timestamp("2020-01-02 " + time)], "instrument": ["A"], "target": [0.1]})
    selected = select_targets(source, {"start": "2020-01-02", "end": "2020-01-03"}, sample_times=cfg.sample_times)
    assert selected["label_end"].tolist() == [pd.Timestamp("2020-01-02 " + end)]
    assert selected.attrs["selection"]["outside_sample_times"] == 0


def test_explicit_label_end_and_invalid_horizon():
    source = pd.DataFrame({"date": [pd.Timestamp("2020-01-02 11:30")], "instrument": ["A"],
                           "target": [0.2], "label_end": [pd.Timestamp("2020-01-02 13:30")]})
    assert len(select_targets(source, {"start": "2020-01-02", "end": "2020-01-02 13:31"})) == 1
    assert len(select_targets(source, {"start": "2020-01-02", "end": "2020-01-02 13:30"})) == 0
    source["label_end"] = source["date"]
    with pytest.raises(ValueError, match="later"):
        select_targets(source, {"start": "2020-01-02", "end": "2020-01-03"})
    source["label_end"] = source["date"] + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="shorter"):
        select_targets(source, {"start": "2020-01-02", "end": "2020-01-03"})


def test_all_missing_history_is_not_counted_as_usable():
    source = bars()
    source[["close", "volume"]] = np.nan
    batch = build_batch(source, source.loc[[2], ["date", "instrument"]], config())
    assert batch.coverage["usable_samples"] == 0
    assert batch.coverage["coverage"] == 0
    assert not batch.observed.any()
    assert (batch.x == 0).all()


def test_allowed_envelope_rejects_future_request_and_clamps_warmup(tmp_path):
    path = tmp_path / "bars.parquet"
    source = bars()
    source.loc[0, "date"] = pd.Timestamp("2019-12-31 10:00")
    source.to_parquet(path)
    cfg = config(bars_path=path, allowed_start="2020-01-01", allowed_end="2025-01-01", history_days=10)
    bundle = load_data(cfg, start="2020-01-02", end="2020-01-03")
    assert bundle.bars.date.min() >= pd.Timestamp("2020-01-01")
    assert bundle.metadata["bars_start"] == "2020-01-01 00:00:00"
    with pytest.raises(ValueError, match="ends after"):
        load_data(cfg, start="2020-01-02", end="2026-01-01")


def test_projected_parquet_loading_unit_normalization_and_fingerprints(tmp_path):
    source = bars()
    source["close"] = [1000, -1, 1200, 9000]
    source["large_unused_column"] = "unrequested"
    bars_path = tmp_path / "bars.parquet"
    labels_path = tmp_path / "labels.parquet"
    source.to_parquet(bars_path)
    pd.DataFrame({"date": pd.to_datetime(["2020-01-02 10:00", "2020-01-03 10:00"]),
                  "instrument": ["A", "A"], "vwap_return": [0.1, 0.2]}).to_parquet(labels_path)
    bundle = load_data(config(bars_path=bars_path, labels_path=labels_path, price_scale=100,
                              missing_sentinel=-1), "2020-01-02", "2020-01-03")
    assert list(bundle.bars) == ["date", "instrument", "close", "volume"]
    assert bundle.bars["close"].iloc[0] == 10.0
    assert np.isnan(bundle.bars["close"].iloc[1])
    assert bundle.bars["volume"].iloc[0] == 1.0
    assert bundle.labels["target"].tolist() == [0.1]
    assert len(bundle.metadata["files"]["bars"][0]["sha256"]) == 64
    assert bundle.metadata["files"]["bars"][0]["bytes"] == bars_path.stat().st_size


def test_instrument_id_requires_explicit_map_and_never_guesses(tmp_path):
    path = tmp_path / "bars.feather"
    mapping = tmp_path / "map.csv"
    source = bars().drop(columns="instrument")
    source["instrument_id"] = 7
    source.to_feather(path)
    with pytest.raises(ValueError, match="explicit instrument_map_path"):
        load_data(config(bars_path=path))
    pd.DataFrame({"instrument_id": [7], "instrument": ["600001.SH"]}).to_csv(mapping, index=False)
    bundle = load_data(config(bars_path=path, instrument_map_path=mapping))
    assert bundle.bars["instrument"].unique().tolist() == ["600001.SH"]
    assert bundle.labels.empty
    pd.DataFrame({"instrument_id": [7, 7], "instrument": ["A", "B"]}).to_csv(mapping, index=False)
    with pytest.raises(ValueError, match="one-to-one"):
        load_data(config(bars_path=path, instrument_map_path=mapping))


def alignment_config(tmp_path, labels, universe, **kwargs):
    bars_path, labels_path, universe_path = (tmp_path / name for name in ["bars.csv", "labels.csv", "universe.csv"])
    bars().to_csv(bars_path, index=False)
    labels.to_csv(labels_path, index=False)
    universe.to_csv(universe_path, index=False)
    return config(bars_path=bars_path, labels_path=labels_path, instruments_path=universe_path, **kwargs)


def test_extra_answers_are_excluded_reported_and_valid_answers_unchanged(tmp_path):
    labels = pd.DataFrame({"date": ["2020-01-02 10:00"] * 3,
                           "instrument": ["A", "B", "OUTSIDE"], "target": [0.2, -0.4, 99.0]})
    universe = labels.loc[:1, ["date", "instrument"]]
    cfg = alignment_config(tmp_path, labels, universe, cache_dir=tmp_path / "cache")
    bundle = load_data(cfg)
    assert bundle.labels["instrument"].tolist() == ["A", "B"]
    assert bundle.labels["target"].tolist() == [0.2, -0.4]
    assert bundle.instruments["instrument"].tolist() == ["A", "B"]
    alignment = {"policy": "exclude", "labels_input_rows": 3, "outside_universe_label_rows": 1,
                 "outside_universe_label_instruments": 1, "labels_retained_rows": 2}
    assert bundle.metadata["alignment"] == alignment
    assert bundle.metadata["label_rows"] == 2
    assert bundle.data_events == [{"event": "outside_universe_labels_excluded", **alignment}]
    # Cache reads must preserve the same audit counts and retained label values.
    warm = load_data(cfg)
    pd.testing.assert_frame_equal(bundle.labels, warm.labels)
    assert warm.data_events == bundle.data_events
    assert warm.metadata == bundle.metadata
    assert all(event["event"] == "cache_hit" for event in warm.cache_events)
    # An unavailable pool security is still predicted and lowers coverage.
    batch = build_batch(bundle.bars, bundle.instruments, cfg)
    assert batch.keys["instrument"].tolist() == ["A", "B"]
    assert batch.coverage["coverage"] == 0.5
    assert not batch.observed[1].any()


def test_labels_outside_point_in_time_universe_rejected_in_error_mode(tmp_path):
    labels = pd.DataFrame({"date": ["2020-01-02 10:00"], "instrument": ["A"], "target": [1]})
    universe = pd.DataFrame({"date": ["2020-01-02 10:00"], "instrument": ["B"]})
    cfg = alignment_config(tmp_path, labels, universe, outside_universe_labels="error")
    with pytest.raises(ValueError, match="outside the point-in-time"):
        load_data(cfg)


def test_label_membership_uses_timestamp_not_union_of_universe(tmp_path):
    labels = pd.DataFrame({"date": ["2020-01-02 10:00", "2020-01-02 10:30", "2020-01-02 10:30"],
                           "instrument": ["A", "A", "B"], "target": [0.1, 0.2, 0.3]})
    universe = labels.loc[[0, 2], ["date", "instrument"]]
    bundle = load_data(alignment_config(tmp_path, labels, universe))
    assert bundle.labels["target"].tolist() == [0.1, 0.3]
    assert bundle.metadata["alignment"]["outside_universe_label_rows"] == 1
    assert bundle.metadata["alignment"]["outside_universe_label_instruments"] == 1


def test_all_outside_labels_return_empty_with_auditable_counts(tmp_path):
    labels = pd.DataFrame({"date": ["2020-01-02 10:00", "2020-01-02 10:30"],
                           "instrument": ["A", "A"], "target": [1, 2]})
    universe = labels[["date", "instrument"]].assign(instrument="B")
    bundle = load_data(alignment_config(tmp_path, labels, universe))
    assert bundle.labels.empty
    assert bundle.instruments["instrument"].tolist() == ["B", "B"]
    assert bundle.metadata["alignment"] == {
        "policy": "exclude", "labels_input_rows": 2, "outside_universe_label_rows": 2,
        "outside_universe_label_instruments": 1, "labels_retained_rows": 0,
    }
    selected = select_targets(bundle.labels, {"start": "2020-01-02", "end": "2020-01-03"})
    assert selected.empty


def test_exclusion_policy_does_not_hide_duplicate_label_keys(tmp_path):
    labels = pd.DataFrame({"date": ["2020-01-02 10:00"] * 2,
                           "instrument": ["OUTSIDE"] * 2, "target": [1, 2]})
    universe = pd.DataFrame({"date": ["2020-01-02 10:00"], "instrument": ["A"]})
    with pytest.raises(ValueError, match="duplicate"):
        load_data(alignment_config(tmp_path, labels, universe))


def test_tar_archive_is_not_silently_scanned(tmp_path):
    archive = tmp_path / "bars.tar.gz"
    archive.write_bytes(b"unused")
    with pytest.raises(ValueError, match="selected monthly feather"):
        load_data(config(bars_path=archive))


def test_verified_cache_cold_warm_outputs_identical_and_parse_is_skipped(tmp_path, monkeypatch):
    import jinyiquant.data as module

    path = tmp_path / "bars.parquet"
    bars().to_parquet(path)
    cfg = config(bars_path=path, cache_dir=tmp_path / "cache")
    cold = load_data(cfg, "2020-01-02", "2020-01-03")
    assert [event["event"] for event in cold.cache_events] == ["cache_miss", "cache_written"]

    def fail_source_parse(*args, **kwargs):
        raise AssertionError("A valid warm cache must skip source parsing")

    monkeypatch.setattr(module, "_read_file", fail_source_parse)
    warm = load_data(cfg, "2020-01-02", "2020-01-03")
    pd.testing.assert_frame_equal(cold.bars, warm.bars)
    pd.testing.assert_frame_equal(cold.labels, warm.labels)
    assert cold.metadata == warm.metadata
    assert [event["event"] for event in warm.cache_events] == ["cache_hit"]


def test_changed_source_content_invalidates_cache(tmp_path):
    path = tmp_path / "bars.parquet"
    source = bars()
    source.to_parquet(path)
    cfg = config(bars_path=path, cache_dir=tmp_path / "cache")
    before = load_data(cfg)
    source.loc[0, "close"] = 999.0
    source.to_parquet(path)
    after = load_data(cfg)
    assert after.bars.loc[0, "close"] == 999.0
    assert before.metadata["data_sha256"] != after.metadata["data_sha256"]
    assert after.cache_events[0]["event"] == "cache_miss"
    assert before.cache_events[0]["cache_key"] != after.cache_events[0]["cache_key"]


@pytest.mark.parametrize("corrupt", ["parquet", "manifest"])
def test_corrupt_cache_is_detected_rebuilt_and_reported(tmp_path, corrupt):
    path = tmp_path / "bars.parquet"
    bars().to_parquet(path)
    cfg = config(bars_path=path, cache_dir=tmp_path / "cache")
    cold = load_data(cfg)
    cache_path = next((tmp_path / "cache" / "bars").glob("*.parquet" if corrupt == "parquet" else "*.json"))
    cache_path.write_bytes(b"corrupted contents")
    repaired = load_data(cfg)
    pd.testing.assert_frame_equal(cold.bars, repaired.bars)
    assert cold.metadata == repaired.metadata
    assert repaired.cache_events[0]["event"] == "cache_invalid"
    assert repaired.cache_events[0]["reason"]
    assert repaired.cache_events[0]["recovery"] == "rebuild_from_source"
    assert repaired.cache_events[-1]["event"] == "cache_written"
    assert load_data(cfg).cache_events[0]["event"] == "cache_hit"


def test_cache_keeps_labels_and_features_in_separate_projections(tmp_path):
    source = bars()
    source["target"] = [0.1, 0.2, 0.3, 0.4]
    path = tmp_path / "combined.parquet"
    source.to_parquet(path)
    cfg = config(bars_path=path, labels_path=path, cache_dir=tmp_path / "cache")
    bundle = load_data(cfg)
    assert "target" not in bundle.bars
    cached_bars = pd.read_parquet(next((tmp_path / "cache" / "bars").glob("*.parquet")))
    cached_labels = pd.read_parquet(next((tmp_path / "cache" / "labels").glob("*.parquet")))
    assert "target" not in cached_bars
    assert "close" not in cached_labels
    assert "target" in cached_labels


def test_cache_projection_key_separates_date_ranges(tmp_path):
    path = tmp_path / "bars.parquet"
    bars().to_parquet(path)
    cfg = config(bars_path=path, cache_dir=tmp_path / "cache", history_days=0)
    early = load_data(cfg, "2020-01-02 09:58", "2020-01-02 10:00")
    later = load_data(cfg, "2020-01-02 10:00", "2020-01-02 10:02")
    assert len(early.bars) == len(later.bars) == 2
    assert early.bars.date.max() < later.bars.date.min()
    assert early.cache_events[0]["cache_key"] != later.cache_events[0]["cache_key"]
