"""Bounded chunked bar loading must reproduce the whole-frame bar matrix exactly.

The 927-day window cannot be read as one pandas frame (132 days: 5 GB resident, 19.3 GB
transient), so bars arrive in bounded chunks and are written straight into the canonical
float32 arrays. These tests pin the three things that make that substitution safe: the arrays
are bit-identical to ``bar_arrays`` over the whole window, chunk boundaries never reorder or
drop an instrument's rows, and an absent month or empty block is a no-op rather than a
misalignment.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jinyiquant.data import (
    BarArraysBuilder,
    bar_arrays,
    bar_arrays_streamed,
    build_batch,
    load_data,
)
from jinyiquant.pool import StreamingPool

FEATURES = ["open", "volume", "bid_price1", "ask_price1"]


def config(**kwargs):
    defaults = dict(feature_columns=FEATURES, lookback=3, min_history=1, missing_policy="ffill",
                    max_ffill_steps=5, sample_times=["10:00", "10:30"], price_scale=100,
                    missing_sentinel=-1, source="test", version="v1", history_days=1,
                    allowed_start="2020-01-01", allowed_end="2024-10-08", bars_path=None,
                    labels_path=None, instruments_path=None, instrument_map_path=None)
    return SimpleNamespace(**(defaults | kwargs))


def source_frame(days=2, instruments=("A", "B", "C"), per_day=8, seed=5):
    """Raw source rows: integer units, sentinel gaps and a shuffled row order."""
    rng = np.random.default_rng(seed)
    stamps = pd.concat([pd.Series(pd.date_range(f"2020-01-0{2 + day} 09:31", periods=per_day,
                                               freq="1min")) for day in range(days)], ignore_index=True)
    frames = []
    for number, name in enumerate(instruments):
        values = rng.integers(1000, 2000, size=(len(stamps), len(FEATURES))).astype(np.int64)
        values[rng.random(values.shape) < 0.05] = -1          # configured missing sentinel
        frames.append(pd.DataFrame({"date": stamps, "instrument": name,
                                    **dict(zip(FEATURES, values.T))}))
    frame = pd.concat(frames, ignore_index=True)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def assemble(frame, chunk_rows, *, shuffle=False):
    """Feed the frame as chronological per-day chunks, optionally shuffled in order."""
    ordered = frame.sort_values(["date", "instrument"], kind="stable").reset_index(drop=True)
    blocks, current = [], None
    for row in ordered.itertuples(index=False):
        if current is None or len(current) >= chunk_rows:
            current = []
            blocks.append(current)
        current.append(row)
    chunks = [pd.DataFrame(block, columns=ordered.columns) for block in blocks]
    builder = BarArraysBuilder(FEATURES, chunk_rows=max(chunk_rows, max(len(c) for c in chunks)))
    order = list(range(len(chunks)))
    if shuffle:
        order = list(np.random.default_rng(3).permutation(len(chunks)))
    for index in order:
        builder.observe(chunks[index])
    for index in order:
        builder.append(chunks[index])
    return builder.result()


def assert_identical(one, other):
    assert one.values.tobytes() == other.values.tobytes()
    assert one.dates.tobytes() == other.dates.tobytes()
    np.testing.assert_array_equal(one.offsets, other.offsets)
    assert list(one.names) == list(other.names)


@pytest.mark.parametrize("chunk_rows", [1, 3, 7, 24, 10_000])
def test_chunked_assembly_is_bitwise_identical_to_the_whole_frame(chunk_rows):
    frame = source_frame()
    expected = bar_arrays(frame, FEATURES)
    assert_identical(assemble(frame, chunk_rows), expected)


def test_chunk_order_does_not_change_the_matrix():
    """Source writers emit record batches in arbitrary order; only the spans may be trusted."""
    frame = source_frame(days=3, per_day=6)
    expected = bar_arrays(frame, FEATURES)
    assert_identical(assemble(frame, 4, shuffle=True), expected)


def test_chunk_boundary_keeps_instrument_and_time_alignment():
    """A block that splits an instrument mid-stream still leaves one contiguous clock."""
    frame = source_frame(days=2, per_day=8)
    expected = bar_arrays(frame, FEATURES)
    for chunk_rows in (5, 9, 13):
        got = assemble(frame, chunk_rows)
        assert_identical(got, expected)
        for code, name in enumerate(got.names):
            block = slice(int(got.offsets[code]), int(got.offsets[code + 1]))
            dates = got.dates[block]
            assert (np.diff(dates.astype("int64")) > 0).all()
            assert len(dates) == (expected.offsets[code + 1] - expected.offsets[code])


def test_empty_block_is_a_no_op_and_duplicate_keys_are_rejected():
    frame = source_frame(instruments=("A",), per_day=4)
    builder = BarArraysBuilder(FEATURES, chunk_rows=8)
    builder.observe(frame.sort_values("date").reset_index(drop=True))
    builder.observe(frame.iloc[:0])
    ordered = frame.sort_values("date").reset_index(drop=True)
    builder.append(ordered)
    builder.append(ordered.iloc[:0])
    assert_identical(builder.result(), bar_arrays(frame, FEATURES))
    duplicate = pd.concat([ordered, ordered.iloc[:1]], ignore_index=True)
    clash = BarArraysBuilder(FEATURES)
    with pytest.raises(ValueError, match="duplicate"):
        clash.observe(duplicate)


def test_interleaved_date_spans_are_rejected_instead_of_reordered():
    """Two blocks covering the same instrument must not be silently reordered."""
    frame = source_frame(instruments=("A",), per_day=6).sort_values("date").reset_index(drop=True)
    builder = BarArraysBuilder(FEATURES, chunk_rows=16)
    builder.observe(frame.iloc[:4])
    builder.observe(frame.iloc[2:8])
    with pytest.raises(ValueError, match="overlap in time"):
        builder.layout()


def test_unfilled_matrix_is_refused():
    frame = source_frame(instruments=("A",), per_day=4).sort_values("date").reset_index(drop=True)
    builder = BarArraysBuilder(FEATURES)
    builder.observe(frame)
    with pytest.raises(ValueError, match="never filled"):
        builder.result()


def write_monthly_source(tmp_path, frame, mapping=True):
    """One feather per month, with shuffled record batches like the real archive."""
    bars_dir = tmp_path / "bars"
    bars_dir.mkdir()
    frame = frame.assign(instrument_id=pd.factorize(frame["instrument"])[0] + 1)
    for month, group in frame.groupby(frame["date"].dt.strftime("%Y-%m"), sort=True):
        group = group.drop(columns="instrument")
        shuffled = pd.concat([group.iloc[index::3] for index in range(3)], ignore_index=True)
        shuffled.to_feather(bars_dir / f"{month}.0.feather")
    map_path = None
    if mapping:
        map_path = tmp_path / "map.csv"
        frame[["instrument", "instrument_id"]].drop_duplicates().to_csv(map_path, index=False)
    return bars_dir, map_path


@pytest.mark.parametrize("chunk_rows", [3, 17, 100_000])
def test_streamed_files_match_the_frame_reader(tmp_path, chunk_rows):
    frame = source_frame(days=3, per_day=5, instruments=("A", "B", "C", "D"))
    bars_dir, map_path = write_monthly_source(tmp_path, frame)
    cfg = config(bars_path=bars_dir, instrument_map_path=map_path)
    streamed, records, audit = bar_arrays_streamed(bars_dir, cfg, start="2020-01-02", end="2020-01-06",
                                                  mapping=pd.read_csv(map_path), chunk_rows=chunk_rows)
    whole = load_data(cfg, start="2020-01-02", end="2020-01-06")
    assert_identical(streamed, bar_arrays(whole.bars, FEATURES))
    assert audit["passes"] == 2 and audit["chunks"] >= 1
    assert audit["rows"] == streamed.rows == whole.metadata["bar_rows"]
    assert [record["sha256"] for record in records] == [record["sha256"]
                                                        for record in whole.metadata["files"]["bars"]]


def test_streamed_load_returns_arrays_and_the_same_pool(tmp_path):
    frame = source_frame(days=3, per_day=5)
    bars_dir, map_path = write_monthly_source(tmp_path, frame)
    stamps = sorted(frame["date"].unique())[3:6]
    keys = pd.DataFrame({"date": stamps, "instrument": "A"})
    cfg = config(bars_path=bars_dir, instrument_map_path=map_path,
                 sample_times=sorted({stamp.strftime("%H:%M") for stamp in stamps}))
    streamed = load_data(cfg, start="2020-01-02", end="2020-01-06", stream_bars=True)
    assert streamed.bars is None and streamed.arrays is not None
    assert streamed.metadata["bars_streamed"]["mode"] == "bounded_chunks"
    assert streamed.metadata["bar_rows"] == streamed.arrays.rows
    assert any(event["event"] == "cache_bypassed" for event in streamed.cache_events)
    materialised = load_data(cfg, start="2020-01-02", end="2020-01-06")
    assert_identical(streamed.arrays, bar_arrays(materialised.bars, FEATURES))
    pool = StreamingPool.from_arrays(streamed.arrays, keys, cfg)
    reference = build_batch(materialised.bars, keys, cfg)
    batch = pool.batch(np.arange(len(keys)))
    assert batch.x.tobytes() == reference.x.tobytes()
    assert np.array_equal(batch.observed, reference.observed)
    assert batch.history_dates.tobytes() == reference.history_dates.tobytes()
    assert pool.coverage == reference.coverage


def test_missing_month_and_empty_block(tmp_path):
    """A month with no file contributes nothing; a block with no rows still assembles."""
    january = source_frame(days=1, per_day=4)
    march = source_frame(days=1, per_day=4, seed=11).assign(
        date=lambda d: d["date"] + pd.DateOffset(months=2))
    frame = pd.concat([january, march], ignore_index=True)
    bars_dir, map_path = write_monthly_source(tmp_path, frame)
    cfg = config(bars_path=bars_dir, instrument_map_path=map_path)
    mapping = pd.read_csv(map_path)
    streamed, _, audit = bar_arrays_streamed(bars_dir, cfg, start="2020-01-01", end="2020-04-01",
                                            mapping=mapping, chunk_rows=5)
    whole = bar_arrays(load_data(cfg, start="2020-01-01", end="2020-04-01").bars, FEATURES)
    assert_identical(streamed, whole)
    assert audit["source_files"] == 2            # January and March; February does not exist
    assert streamed.rows == len(frame)
    empty, _, empty_audit = bar_arrays_streamed(bars_dir, cfg, start="2020-01-20", end="2020-01-21",
                                               mapping=mapping, chunk_rows=5)
    assert empty.rows == 0 and empty.values.shape == (0, len(FEATURES))
    np.testing.assert_array_equal(empty.offsets, np.zeros(1, np.int64))
    assert empty_audit["rows"] == 0 and empty_audit["chunks"] == 0
    assert_identical(empty, bar_arrays(load_data(cfg, start="2020-01-20", end="2020-01-21").bars, FEATURES))


def test_streamed_reader_refuses_archives_and_keeps_the_sealed_guard(tmp_path):
    path = tmp_path / "bars.rar"
    path.write_bytes(b"not read")
    with pytest.raises(ValueError, match="extract the archive"):
        bar_arrays_streamed(path, config(), start="2020-01-02", end="2020-01-03")
    bars_dir, _ = write_monthly_source(tmp_path, source_frame(days=1))
    with pytest.raises(ValueError, match="ends after"):
        load_data(config(bars_path=bars_dir, allowed_end="2020-06-01"), start="2020-01-02",
                  end="2024-12-01", stream_bars=True, chunk_rows=4)
