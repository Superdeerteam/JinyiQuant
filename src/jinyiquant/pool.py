"""Streaming training pool: sample keys plus raw bars, windows built per batch.

The materialised pool keeps one ``(samples, lookback, features)`` float32 window tensor per
process (24.9 GB for the 132-day window, 173 GB projected for 920 days, which does not fit
the 172 GB host). This module keeps only what a window is made of - the sample keys and one
canonical float32 bar matrix - and assembles the windows of the keys a sampler asks for:

    pool = StreamingPool.build(bars, keys, config)     # keys + bars, no window tensor
    batch = pool.batch(indices)                        # Batch for those keys only

Nothing about the training recipe changes: the keys keep the order ``select_targets`` gave
them, the sampler still owns the draw, and every window comes from
:func:`jinyiquant.data.window_arrays`, the same builder the materialised pool calls. A
streamed batch is therefore bitwise identical to the same rows of the materialised pool,
which ``scripts/verify_streaming_pool.py`` and ``tests/test_streaming_pool.py`` assert.

The default training path stays materialised: :func:`training_pool` only returns a
:class:`StreamingPool` when ``streaming=True`` is passed explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    FORBIDDEN_FEATURES,
    KEYS,
    BarArrays,
    Batch,
    approved_sample_minutes,
    bar_arrays,
    build_batch,
    config_option,
    history_diagnostics,
    history_quality,
    instrument_codes,
    lob_trade_columns,
    validate_keys,
    validate_sample_times,
    window_arrays,
    window_coverage,
    window_geometry,
    window_parameters,
)

__all__ = ["StreamingPool", "training_pool"]


def _bar_counts(arrays: BarArrays, codes: np.ndarray, begin: np.ndarray, stop: np.ndarray,
                history_feature_count: int, *, instruments_per_chunk: int = 256,
                block_rows: int = 1_048_576):
    """Per-key observed and effective row counts, computed from bounded prefix sums.

    ``observed`` is the number of finite feature values in the window (what ``observed.mean()``
    measures on a materialised pool) and ``effective`` counts rows with any finite primary
    feature (what ``Batch.effective_rows.sum(axis=1)`` measures). Both are integers, so the
    derived coverage is bit-identical to the materialised path. Prefix sums are built per
    instrument chunk over bounded bar blocks, so the transients never exceed a slice of the
    bar matrix - the whole point of not materialising windows.
    """
    observed = np.zeros(len(codes), dtype=np.int64)
    effective = np.zeros(len(codes), dtype=np.int64)
    if not arrays.rows or not (codes >= 0).any():
        return observed, effective
    present = np.flatnonzero(codes >= 0)
    order = np.argsort(codes[present], kind="stable")
    rows, ordered = present[order], codes[present][order]
    bounds = np.searchsorted(ordered, np.arange(len(arrays.names) + 1))
    for first in range(0, len(arrays.names), instruments_per_chunk):
        last = min(first + instruments_per_chunk, len(arrays.names))
        selected = rows[bounds[first]:bounds[last]]
        low, high = int(arrays.offsets[first]), int(arrays.offsets[last])
        if not len(selected) or low == high:
            continue
        size = high - low
        prefix_finite = np.zeros(size + 1, dtype=np.int64)
        prefix_effective = np.zeros(size + 1, dtype=np.int64)
        for offset in range(0, size, block_rows):
            limit = min(offset + block_rows, size)
            finite = np.isfinite(arrays.values[low + offset:low + limit])
            prefix_finite[offset + 1:limit + 1] = np.cumsum(finite.sum(axis=1, dtype=np.int64)) + prefix_finite[offset]
            prefix_effective[offset + 1:limit + 1] = (
                np.cumsum(finite[:, :history_feature_count].any(axis=1).astype(np.int64))
                + prefix_effective[offset])
        local = arrays.offsets[codes[selected]] - low
        observed[selected] = prefix_finite[local + stop[selected]] - prefix_finite[local + begin[selected]]
        effective[selected] = prefix_effective[local + stop[selected]] - prefix_effective[local + begin[selected]]
    return observed, effective


@dataclass(frozen=False, eq=False)
class StreamingPool:
    """Training pool that stores the sample index and raw bars, not the window tensor.

    ``keys`` is the same sorted key table :func:`build_batch` would return, so target
    alignment, sample masks and coverage keep their meaning. ``coverage`` is the pool-level
    coverage of the materialised path (a streamed batch carries it unchanged, because a batch
    is a draw from this pool, not a pool of its own).
    """

    keys: pd.DataFrame
    arrays: BarArrays
    codes: np.ndarray
    begin: np.ndarray
    stop: np.ndarray
    settings: dict[str, Any]
    history_feature_count: int
    coverage: dict[str, Any]
    observed_count: np.ndarray = field(repr=False)
    effective_count: np.ndarray = field(repr=False)
    sample_times: Any = None
    features: Any = None
    approved_minutes: np.ndarray | None = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def build(cls, bars: pd.DataFrame, keys: pd.DataFrame, config: Any, *,
              history_quality_summary: bool = True) -> "StreamingPool":
        """Index ``keys`` against the bars frame exactly like :func:`build_batch` would.

        ``history_quality_summary`` reproduces the materialised coverage summary with one
        chunked pass over the keys and no retained windows; it is the only build step whose
        cost grows with the key count rather than the bar count, so it can be switched off
        when a caller only needs the index.
        """
        features = list(config_option(config, "feature_columns"))
        if FORBIDDEN_FEATURES.intersection(features):
            raise ValueError("Future label columns cannot be input features")
        bars = validate_keys(bars, "bars")
        if any(column not in bars for column in features):
            raise ValueError("Missing configured feature columns")
        return cls.from_arrays(bar_arrays(bars, features), keys, config,
                               history_quality_summary=history_quality_summary)

    @classmethod
    def from_arrays(cls, arrays: BarArrays, keys: pd.DataFrame, config: Any, *,
                    history_quality_summary: bool = True) -> "StreamingPool":
        """Index ``keys`` against a canonical bar matrix that was built elsewhere.

        ``load_data(..., stream_bars=True)`` produces that matrix in bounded chunks, so a
        window that never fits as a pandas frame can still be trained on; the matrix is the
        same one :meth:`build` derives from a frame, so the pool is identical either way.
        """
        features = list(config_option(config, "feature_columns"))
        history_feature_count = 12 if tuple(features) == lob_trade_columns() else len(features)
        if FORBIDDEN_FEATURES.intersection(features):
            raise ValueError("Future label columns cannot be input features")
        if arrays.width != len(features):
            raise ValueError(f"Bar matrix width {arrays.width} does not match "
                             f"{len(features)} configured features")
        keys = validate_keys(keys[KEYS], "sample keys")
        times = config_option(config, "sample_times", [])
        key_dates = validate_sample_times(keys, times)
        settings = window_parameters(config)
        codes = instrument_codes(arrays, keys["instrument"])
        begin, stop = window_geometry(arrays, codes, key_dates, settings["lookback"])
        observed_count, effective_count = _bar_counts(arrays, codes, begin, stop, history_feature_count)
        enough = effective_count >= settings["minimum"]
        coverage = window_coverage(keys, enough, observed_count=int(observed_count.sum()),
                                   observed_size=int(len(keys) * settings["lookback"] * len(features)),
                                   history_count=stop - begin)
        pool = cls(keys=keys, arrays=arrays, codes=codes, begin=begin, stop=stop, settings=settings,
                   history_feature_count=history_feature_count, coverage=coverage,
                   observed_count=observed_count, effective_count=effective_count, sample_times=times,
                   features=features, approved_minutes=approved_sample_minutes(times))
        if history_quality_summary:
            pool.coverage["history_quality"] = history_quality(
                pool.history_diagnostics(), features, history_feature_count)
        return pool

    # ------------------------------------------------------------------ pool surface
    def __len__(self) -> int:
        return len(self.keys)

    @property
    def effective_counts(self) -> np.ndarray:
        """Rows with any finite primary feature, i.e. ``Batch.effective_rows.sum(axis=1)``."""
        return self.effective_count

    @property
    def fit_mask(self) -> np.ndarray:
        """The mask ``runner`` derives from ``batch.effective_rows.sum(axis=1) >= min_history``."""
        return self.effective_count >= int(self.settings["minimum"])

    def nbytes(self) -> dict[str, int]:
        """Pool footprint by component; the window tensor is absent by construction.

        ``keys`` counts the timestamp column plus one pointer per instrument string (the
        strings themselves are shared), which is what a key costs beyond its geometry.
        """
        index = {"keys": int(self.keys["date"].to_numpy().nbytes + len(self.keys) * 8),
                 "instrument_codes": int(self.codes.nbytes), "window_geometry": int(self.begin.nbytes + self.stop.nbytes),
                 "row_counts": int(self.observed_count.nbytes + self.effective_count.nbytes)}
        return {**self.arrays.nbytes(), **index}

    # ------------------------------------------------------------------ batching
    def _indexes(self, indices) -> np.ndarray:
        positions = np.asarray(indices)
        if positions.ndim != 1 or positions.dtype.kind not in "iu":
            raise ValueError("Streaming batch indices must be a 1-D integer array")
        if len(positions) and (positions.min() < 0 or positions.max() >= len(self.keys)):
            raise ValueError("Streaming batch index outside the pool")
        return positions.astype(np.int64, copy=False)

    def batch(self, indices) -> Batch:
        """Build one batch's windows; the caller's sampler decides which keys and their order."""
        positions = self._indexes(indices)
        keys = self.keys.iloc[positions].reset_index(drop=True)
        # A streamed batch repeats the key-time contract on every draw, so no batch can carry
        # an unapproved timestamp even when the pool was built from approved keys.
        validate_sample_times(keys, self.sample_times, self.approved_minutes)
        lookback = int(self.settings["lookback"])
        x = np.zeros((len(positions), lookback, self.arrays.width), dtype=np.float32)
        observed = np.zeros(x.shape, dtype=bool)
        history_dates = np.full((len(positions), lookback), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        window_arrays(self.arrays, self.codes[positions], self.begin[positions], self.stop[positions],
                      x, observed, history_dates, history_feature_count=self.history_feature_count,
                      **self.settings)
        return Batch(keys, x, observed, history_dates, self.coverage, self.history_feature_count)

    def history_diagnostics(self, *, chunk_rows: int = 8192) -> pd.DataFrame:
        """Per-key history quality, streamed in key chunks through the shared diagnostic."""
        if chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")
        frames = [history_diagnostics(self.batch(np.arange(start, min(start + chunk_rows, len(self.keys)))))
                  for start in range(0, len(self.keys), chunk_rows)]
        if not frames:
            return history_diagnostics(self.batch(np.zeros(0, dtype=np.int64)))
        return pd.concat(frames, ignore_index=True)


def training_pool(bars: pd.DataFrame, keys: pd.DataFrame, config: Any, *, streaming: bool = False,
                  arrays: BarArrays | None = None):
    """The default materialised :class:`Batch`, or a :class:`StreamingPool` when asked for.

    The switch is off by default and lives here - not in the run configuration - so no
    existing pipeline can silently change which pool it trains on. ``arrays`` supplies the
    canonical matrix a bounded bar read already produced; it implies the streaming pool and
    skips the frame entirely.
    """
    if not streaming:
        if arrays is not None:
            raise ValueError("A prebuilt BarArrays only feeds the streaming pool")
        return build_batch(bars, keys, config)
    if arrays is not None:
        return StreamingPool.from_arrays(arrays, keys, config)
    return StreamingPool.build(bars, keys, config)
