"""Versioned local data readers and label-window selection.

All input timestamps are exchange-local, timezone-naive bar *end* timestamps.
Instrument IDs are never inferred from security codes: an explicit map is required.
This module owns the shared data contract (input schema, label clock, split boundaries)
and the chronological guards that use it; ``modified_tlob.schema`` re-exports the schema
part. Reading, window construction and label selection live here; the training entry
stays in ``modified_tlob/train.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

# --------------------------------------------------------------------------- data contract
# One definition site for the input schema and the chronological boundaries. The model
# layer (``modified_tlob.schema``) re-exports this contract instead of owning it, so the
# data layer never imports the model layer.

CONTRACT_VERSION = 2

KEYS = ["date", "instrument"]
FORBIDDEN_FEATURES = frozenset({"target", "vwap_return", "label_end", "end_price", "vwap"})

TRADE_COLUMNS = ("volume", "amount", "deal_number")
LABEL_HORIZON_MINUTES = 30

# Final-check interval [SEALED_START, SEALED_END): the only sanctioned way to open it is a
# recorded one-time final_access claim, taken right before evaluation. Phase A never uses it.
SEALED_START = pd.Timestamp("2024-10-08")
SEALED_END = pd.Timestamp("2025-01-01")
MAX_REPORTED_BOUNDARY_SAMPLES = 5

# Rows per bounded bar block for the streamed reader: the transient float64 chunk plus its
# float32 copy stay around 40 MB, independent of how many days the window spans.
DEFAULT_BAR_CHUNK_ROWS = 262_144


def config_option(config, name: str, default=None):
    """Read one option from a mapping or an attribute object (resolved configs)."""
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def lob_columns(levels: int = 3) -> tuple[str, ...]:
    """Explicit input order matching upstream ask-price/size, bid-price/size per level."""
    if levels not in {3, 5}:
        raise ValueError("Only three or five LOB levels are configured")
    return tuple(f"{side}_{kind}{level}" for level in range(1, levels + 1)
                 for side, kind in (("ask", "price"), ("ask", "volume"), ("bid", "price"), ("bid", "volume")))


def lob_trade_columns() -> tuple[str, ...]:
    return lob_columns(3) + TRADE_COLUMNS


def validate_input_schema(fields, book_representation="raw", trade_features="zero") -> int:
    fields = tuple(fields)
    if fields not in (lob_columns(3), lob_columns(5), lob_trade_columns()):
        raise ValueError("Unexpected LOB field order")
    if book_representation not in ("raw", "relative_mid_bps") or trade_features not in ("zero", "log1p"):
        raise ValueError("Unsupported LOB input transformation")
    if trade_features != "zero" and fields != lob_trade_columns():
        raise ValueError("Trade features require three LOB levels followed by volume, amount, deal_number")
    return 12 if fields == lob_trade_columns() else len(fields)


def timestamps(values, name: str) -> pd.Series:
    """Exchange-local timezone-naive bar end timestamps with explicit failure messages."""
    try:
        result = pd.to_datetime(values, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name}: invalid timestamp") from exc
    if result.isna().any():
        raise ValueError(f"{name}: missing timestamp")
    if isinstance(result.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{name}: use timezone-naive Asia/Shanghai timestamps")
    return result.astype("datetime64[ns]")


def validate_keys(frame: pd.DataFrame, name: str = "data") -> pd.DataFrame:
    """Reject ambiguous joins instead of silently dropping duplicate source rows."""
    if any(key not in frame for key in KEYS):
        raise ValueError(f"{name}: required columns are {KEYS}")
    result = frame.copy()
    result["date"] = timestamps(result["date"], f"{name}.date")
    if result.empty:
        return result.reset_index(drop=True)
    instruments = result["instrument"]
    if instruments.isna().any() or not instruments.map(lambda x: isinstance(x, str)).all():
        raise ValueError(f"{name}: instrument must contain nonempty strings")
    if instruments.str.strip().eq("").any() or instruments.ne(instruments.str.strip()).any():
        raise ValueError(f"{name}: empty or whitespace-padded instrument")
    if result.duplicated(KEYS).any():
        raise ValueError(f"{name}: duplicate (date, instrument) keys")
    return result.sort_values(KEYS, kind="stable").reset_index(drop=True)


def _minutes(dates: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(dates)
    return stamps.dt.hour * 60 + stamps.dt.minute


def required_label_end(dates: pd.Series, horizon_minutes: int = LABEL_HORIZON_MINUTES) -> pd.Series:
    """Forward end of a label horizon; the 11:30 sample resumes after the lunch break."""
    stamps = pd.to_datetime(dates)
    required = stamps + pd.Timedelta(minutes=horizon_minutes)
    lunch = _minutes(stamps).eq(690) & (horizon_minutes == LABEL_HORIZON_MINUTES)
    required.loc[lunch] = stamps.loc[lunch] + pd.Timedelta(hours=2)
    return required


def known_session_horizon(dates: pd.Series, horizon_minutes: int = LABEL_HORIZON_MINUTES) -> pd.Series:
    """Whether the horizon stays inside one continuous session; overnight windows are unknown."""
    minute = _minutes(dates)
    end_minute = minute + horizon_minutes
    lunch = minute.eq(690) & (horizon_minutes == LABEL_HORIZON_MINUTES)
    return ((minute >= 570) & (end_minute <= 690)) | ((minute >= 780) & (end_minute <= 900)) | lunch


def boundary_samples(frame: pd.DataFrame, boundary) -> pd.DataFrame:
    """Samples whose forward label window reaches the exclusive boundary."""
    return frame.loc[frame["label_end"] >= pd.Timestamp(boundary)]


def assert_label_windows_before(frame: pd.DataFrame, boundary, *, name: str = "samples") -> None:
    """Fail with the offending samples when a label window reaches the exclusive boundary."""
    boundary = pd.Timestamp(boundary)
    crossing = boundary_samples(frame, boundary)
    if len(crossing):
        columns = [column for column in ("date", "instrument", "label_end") if column in crossing]
        detail = crossing.head(MAX_REPORTED_BOUNDARY_SAMPLES)[columns].to_dict("records")
        raise ValueError(f"{name}: {len(crossing)} label windows reach the exclusive boundary {boundary}; "
                         f"boundary samples: {detail}")


@dataclass(frozen=True)
class FinalAccessClaim:
    """Proof that one sealed final-check view was recorded in the audit index.

    ``tracking.claim_final_access`` is the only sanctioned minting path: it inserts the
    audit row and enforces ``final_access.max_views``. The type exists so the read path
    cannot be opened by flipping a boolean; building one by hand is an explicit,
    greppable act that leaves no audit row behind.
    """
    scope: str
    run_id: str
    granted_at: str
    window: tuple[str, str]


def validate_sealed_claim(config: Any, claim: Any) -> FinalAccessClaim:
    """Reject a bare flag or a claim minted for a different final window."""
    if not isinstance(claim, FinalAccessClaim):
        raise TypeError("sealed_access must be a FinalAccessClaim recorded by tracking.claim_final_access; "
                        "a bare flag cannot open the sealed interval")
    expected = _sealed_claim_window(config)
    if expected is not None and tuple(claim.window) != expected:
        raise ValueError(f"sealed claim window {tuple(claim.window)} does not match the configured "
                         f"final split {expected}")
    return claim


def _sealed_claim_window(config: Any) -> tuple[str, str] | None:
    """The configured final-split window, or None when the config declares none."""
    splits = config_option(config, "splits")
    final = config_option(splits, "final") if splits is not None else None
    start = config_option(final, "start") if final is not None else None
    end = config_option(final, "end") if final is not None else None
    if start is None or end is None:
        return None
    return (pd.Timestamp(start).isoformat(), pd.Timestamp(end).isoformat())


def assert_development_read(start, end, *, sealed_access: FinalAccessClaim | None = None) -> None:
    """Refuse any read whose window could open the sealed final-check interval.

    ``start``/``end`` are the effective file-read window; ``None`` means unbounded. The
    sealed interval opens only for a ``FinalAccessClaim`` minted by
    ``tracking.claim_final_access``, which records the one-time view in the audit index.
    A bare boolean cannot be accepted: it would leave no audit record behind.
    """
    if sealed_access is not None:
        if not isinstance(sealed_access, FinalAccessClaim):
            raise TypeError("sealed_access must be a FinalAccessClaim recorded by "
                            "tracking.claim_final_access; a bare flag cannot open the sealed interval")
        return
    lower = None if start is None else pd.Timestamp(start)
    upper = None if end is None else pd.Timestamp(end)
    if (lower is None or lower < SEALED_END) and (upper is None or upper > SEALED_START):
        raise ValueError(f"Sealed final interval [{SEALED_START}, {SEALED_END}) cannot be read: requested window "
                         f"[{start}, {end}) overlaps it; development reads end before {SEALED_START}. "
                         "Opening it requires the recorded one-time final_access claim (see runner.run), "
                         "which phase A does not use")


READER_FORMAT_VERSION = 1

__all__ = ["Batch", "DataBundle", "DaySelection", "build_batch", "complete_trading_calendar",
           "history_diagnostics", "load_data", "monthly_sampled_days", "preprocess_bars",
           "previous_trading_days", "select_targets", "validate_keys"]


@dataclass
class DataBundle:
    """One reading of the data contract.

    ``bars`` is the window frame of the default path. With ``stream_bars=True`` the same
    window is instead delivered as ``arrays`` (the canonical float32 matrix) and ``bars`` is
    ``None``; a caller that still wants a frame fails loudly on it rather than seeing an
    empty one.
    """

    bars: pd.DataFrame | None
    labels: pd.DataFrame
    metadata: dict[str, Any]
    instruments: pd.DataFrame | None = None
    cache_events: list[dict[str, Any]] = field(default_factory=list)
    data_events: list[dict[str, Any]] = field(default_factory=list)
    arrays: "BarArrays | None" = None

    @property
    def universe(self) -> pd.DataFrame | None:
        return self.instruments


@dataclass
class Batch:
    keys: pd.DataFrame
    x: np.ndarray
    observed: np.ndarray
    history_dates: np.ndarray
    coverage: dict[str, Any]
    history_feature_count: int | None = None

    @property
    def effective_rows(self):
        """Optional auxiliary inputs cannot create usable primary-input history."""
        return self.observed[..., :self.history_feature_count].any(axis=2)


def _get(config: Any, name: str, default: Any = None) -> Any:
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _time_filter(frame: pd.DataFrame, start: Any, end: Any) -> pd.DataFrame:
    if "date" not in frame:
        return frame
    frame = frame.copy()
    frame["date"] = timestamps(frame["date"], "date")
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= frame["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= frame["date"] < pd.Timestamp(end)
    return frame.loc[mask].reset_index(drop=True)


def _arrow_filter(schema: pa.Schema, start: Any, end: Any) -> ds.Expression | None:
    expression = None
    if "date" in schema.names:
        for value, operation in [(start, "start"), (end, "end")]:
            if value is not None:
                scalar = pa.scalar(pd.Timestamp(value), type=schema.field("date").type)
                clause = ds.field("date") >= scalar if operation == "start" else ds.field("date") < scalar
                expression = clause if expression is None else expression & clause
    return expression


def _columns(available: list[str], requested: list[str]) -> list[str]:
    return list(dict.fromkeys(column for column in requested if column in available))


def _file_blocks(path: Path, requested: list[str], start: Any, end: Any,
                 selection: DaySelection | None, chunk_rows: int):
    """One source file's rows in ascending date order, at most ``chunk_rows`` rows per block.

    A source file is the coarsest block: the writers of these archives emit a month's record
    batches in arbitrary order (each batch sorted, the batches not), so the file is read once
    and its rows ordered by date before slicing. Blocks are then chronological, which is what
    lets pass 2 place a chunk without ever holding the window.
    """
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    frame = _selected(_read_file(path, requested, start, end), selection)
    if frame.empty:
        return
    stamps = frame["date"].to_numpy(dtype="datetime64[ns]")
    order = np.argsort(stamps, kind="stable")
    for begin in range(0, len(order), chunk_rows):
        yield frame.iloc[order[begin:begin + chunk_rows]]


def _read_file(path: Path, requested: list[str], start: Any, end: Any) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        dataset = ds.dataset(path, format="parquet")
        return dataset.to_table(
            columns=_columns(dataset.schema.names, requested),
            filter=_arrow_filter(dataset.schema, start, end),
        ).to_pandas()
    if suffix in {".feather", ".arrow"}:
        parts = []
        with pa.memory_map(str(path), "r") as source:
            reader = ipc.open_file(source)
            columns = _columns(reader.schema.names, requested)
            # Arrow IPC lacks parquet's predicate pushdown. Bound temporary memory
            # to one record batch and retain only the requested date range.
            for index in range(reader.num_record_batches):
                table = pa.Table.from_batches([reader.get_batch(index)]).select(columns)
                if "date" in columns:
                    mask = None
                    for value, operation in [(start, pc.greater_equal), (end, pc.less)]:
                        if value is not None:
                            scalar = pa.scalar(pd.Timestamp(value), type=table["date"].type)
                            clause = operation(table["date"], scalar)
                            mask = clause if mask is None else pc.and_(mask, clause)
                    if mask is not None:
                        table = table.filter(mask)
                if table.num_rows:
                    parts.append(table.to_pandas())
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    if suffix == ".csv":
        parts = []
        for chunk in pd.read_csv(path, usecols=lambda c: c in requested, chunksize=100_000,
                                 dtype={"instrument": "string"}):
            chunk = _time_filter(chunk, start, end)
            if len(chunk):
                parts.append(chunk)
        return pd.concat(parts, ignore_index=True) if parts else pd.read_csv(path, nrows=0,
                                                                           usecols=lambda c: c in requested)
    raise ValueError(f"Unsupported data file {path}; use parquet, feather/arrow, csv, or monthly label RAR")


def _month_overlaps(name: str, start: Any, end: Any) -> bool:
    match = re.search(r"(?<!\d)(20\d{2})[-_]?([01]\d)(?!\d)", Path(name).name)
    if not match:
        return True
    month = pd.Timestamp(f"{match[1]}-{match[2]}-01")
    next_month = month + pd.offsets.MonthBegin(1)
    return (end is None or month < pd.Timestamp(end)) and (start is None or next_month > pd.Timestamp(start))


def _read_rar(path: Path, requested: list[str], start: Any, end: Any) -> tuple[pd.DataFrame, list[str]]:
    executable = shutil.which("tar")
    if executable is None:
        raise ValueError("RAR input requires libarchive tar; alternatively extract the required monthly parquet files")
    listing = subprocess.run([executable, "-tf", str(path)], capture_output=True, check=True,
                             text=True, encoding="utf-8")
    members = [name for name in listing.stdout.splitlines()
               if name.endswith(".parquet") and _month_overlaps(name, start, end)]
    if not members:
        raise ValueError(f"No monthly parquet members overlap requested dates: {path}")
    frames = []
    for member in members:
        payload = subprocess.run([executable, "-xOf", str(path), member], capture_output=True, check=True).stdout
        source = pq.ParquetFile(pa.BufferReader(payload))
        frame = source.read(columns=_columns(source.schema_arrow.names, requested)).to_pandas()
        frames.append(_time_filter(frame, start, end))
    return pd.concat(frames, ignore_index=True), members


def _digest(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Data changed while fingerprinting: {path}")
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": after.st_size}


def _cached_read(path: Path, requested: list[str], start: Any, end: Any, source_record: dict,
                 source_state: tuple[int, int], cache_dir: Path | None,
                 events: list[dict], selection: DaySelection | None = None) -> tuple[pd.DataFrame, dict]:
    """Use a verified projection cache; never include runtime cache state in provenance."""
    identity = {"reader_version": READER_FORMAT_VERSION, "source_sha256": source_record["sha256"],
                "format": path.suffix.lower(), "columns": list(dict.fromkeys(requested)),
                "start": None if start is None else str(pd.Timestamp(start)),
                "end": None if end is None else str(pd.Timestamp(end)),
                "pandas_version": pd.__version__, "pyarrow_version": pa.__version__}
    if selection is not None:
        # Keep unsampled cache identities unchanged; a day selection is an extra dimension.
        identity["selection"] = selection.identity()
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_file = None if cache_dir is None else cache_dir / f"{cache_key}.parquet"
    cache_manifest = None if cache_dir is None else cache_dir / f"{cache_key}.json"
    context = {"cache_key": cache_key, "source": str(path.resolve()),
               "role": None if cache_dir is None else cache_dir.name}
    if cache_manifest is not None and (cache_manifest.exists() or cache_file.exists()):
        try:
            document = json.loads(cache_manifest.read_text(encoding="utf-8"))
            if document["identity"] != identity:
                raise ValueError("Cache identity does not match the requested source projection")
            actual = _digest(cache_file)
            if actual["sha256"] != document["parquet_sha256"] or actual["bytes"] != document["parquet_bytes"]:
                raise ValueError("Cache parquet checksum mismatch")
            frame = pd.read_parquet(cache_file)
            if len(frame) != document["rows"] or list(frame.columns) != document["columns"]:
                raise ValueError("Cache shape differs from its manifest")
            if not set(frame.columns).issubset(requested):
                raise ValueError("Cache contains unrequested columns")
            if not isinstance(document["source_extra"], dict):
                raise ValueError("Malformed cache source metadata")
            events.append({"event": "cache_hit", **context})
            return frame, document["source_extra"]
        except Exception as exc:
            # A cache is disposable. Retain the reason and rebuild from the
            # independently hashed original source; source failures propagate.
            events.append({"event": "cache_invalid", **context,
                           "reason": f"{type(exc).__name__}: {exc}", "recovery": "rebuild_from_source"})
    elif cache_dir is not None:
        events.append({"event": "cache_miss", **context})
    if path.suffix.lower() == ".rar":
        frame, members = _read_rar(path, requested, start, end)
        extra = {"members": members}
    else:
        frame, extra = _read_file(path, requested, start, end), {}
    if selection is not None:
        frame = frame.loc[selection.keep(frame["date"])].reset_index(drop=True)
    if cache_dir is not None:
        source_after = path.stat()
        if source_state != (source_after.st_size, source_after.st_mtime_ns):
            raise ValueError(f"Data changed during read; refusing to populate cache: {path}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Unique temporary names and atomic replaces prevent partially written
        # entries being read. A concurrent two-file publication mismatch simply
        # fails checksum validation and rebuilds; it cannot return stale data.
        token = uuid.uuid4().hex
        temporary = cache_dir / f".{cache_key}.{token}.parquet.partial"
        temporary_manifest = cache_dir / f".{cache_key}.{token}.json.partial"
        frame.to_parquet(temporary, index=False)
        content_record = _digest(temporary)
        document = {"identity": identity, "parquet_sha256": content_record["sha256"],
                    "parquet_bytes": content_record["bytes"], "rows": len(frame),
                    "columns": list(frame.columns), "source_extra": extra}
        temporary_manifest.write_text(json.dumps(document, sort_keys=True, allow_nan=False), encoding="utf-8")
        temporary.replace(cache_file)
        temporary_manifest.replace(cache_manifest)
        events.append({"event": "cache_written", **context})
    return frame, extra


def _source_files(path: Path, start: Any, end: Any, cache_dir: Path | None) -> list[Path]:
    """Source files of one role that overlap ``[start, end)``, in a stable order."""
    return sorted(file for file in path.rglob("*") if file.is_file()
                  and file.suffix.lower() in {".parquet", ".feather", ".arrow", ".csv"}
                  and (cache_dir is None or not file.resolve().is_relative_to(cache_dir.parent.resolve()))
                  and _month_overlaps(file.name, start, end)) if path.is_dir() else [path]


def _read(path_value: Any, requested: list[str], start: Any, end: Any,
          cache_dir: Path | None = None, cache_events: list[dict] | None = None,
          selection: DaySelection | None = None, *, sealed_access: FinalAccessClaim | None = None
          ) -> tuple[pd.DataFrame, list[dict]]:
    if start is not None or end is not None:
        # One choke point for every windowed market-data read (including verified caches).
        assert_development_read(start, end, sealed_access=sealed_access)
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Data source does not exist: {path}")
    if str(path).lower().endswith((".tar.gz", ".tgz", ".tar")):
        raise ValueError("Extract selected monthly feather files from the bar archive first; "
                         "provide instrument_map_path for instrument_id. Full archive scans are not automatic.")
    files = _source_files(path, start, end, cache_dir)
    if not files:
        raise ValueError(f"No data files overlap requested dates: {path}")
    frames, records = [], []
    cache_events = [] if cache_events is None else cache_events
    for file in files:
        before = file.stat()
        record = _digest(file)
        frame, extra = _cached_read(file, requested, start, end, record,
                                   (before.st_size, before.st_mtime_ns), cache_dir, cache_events, selection)
        record.update(extra)
        after = file.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Data changed during read: {file}")
        frames.append(frame)
        records.append(record)
    return pd.concat(frames, ignore_index=True), records


def _map_instruments(frame: pd.DataFrame, mapping: pd.DataFrame | None, name: str) -> pd.DataFrame:
    if "instrument" in frame:
        if mapping is not None and "instrument_id" in frame:
            matched = frame.merge(mapping.rename(columns={"instrument": "_mapped_instrument"}),
                                  on="instrument_id", how="left", validate="many_to_one")
            if matched["_mapped_instrument"].isna().any() or not matched["instrument"].eq(matched["_mapped_instrument"]).all():
                raise ValueError(f"{name}: instrument disagrees with explicit instrument_id map")
        return frame
    if "instrument_id" not in frame:
        raise ValueError(f"{name}: missing instrument or instrument_id")
    if mapping is None:
        raise ValueError(f"{name}: instrument_id requires an explicit instrument_map_path; never infer IDs")
    mapped = frame.merge(mapping, on="instrument_id", how="left", validate="many_to_one", sort=False)
    if mapped["instrument"].isna().any():
        raise ValueError(f"{name}: unmapped instrument_id")
    return mapped


@dataclass(frozen=True)
class DaySelection:
    """Keep only the listed calendar days inside [start, end); other dates pass through."""

    start: pd.Timestamp
    end: pd.Timestamp
    days: pd.DatetimeIndex

    def keep(self, dates: pd.Series) -> pd.Series:
        inside = (dates >= self.start) & (dates < self.end)
        return ~inside | dates.dt.normalize().isin(self.days)

    def identity(self) -> dict[str, Any]:
        stamps = "\n".join(day.strftime("%Y-%m-%d") for day in sorted(self.days))
        return {"start": str(self.start), "end": str(self.end), "days": len(self.days),
                "days_sha256": hashlib.sha256(stamps.encode()).hexdigest()}


def complete_trading_calendar(universe: pd.DataFrame, sample_times: Sequence[str]) -> pd.DatetimeIndex:
    """Days whose independent universe grid contains every configured sampling time."""
    grid = list(dict.fromkeys(sample_times))
    stamps = timestamps(universe["date"], "universe.date")
    kept = stamps.loc[stamps.dt.strftime("%H:%M").isin(grid)]
    counts = kept.groupby(kept.dt.normalize()).nunique()
    return pd.DatetimeIndex(counts.index[counts == len(grid)].sort_values())


def monthly_sampled_days(calendar: Any, start: Any, end: Any,
                         days_per_month: int) -> tuple[pd.DatetimeIndex, list[dict]]:
    """Centered stratified choice of complete trading days in every month of [start, end)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.day != 1 or end.day != 1 or start >= end:
        raise ValueError("Monthly sampling requires month-aligned start < end")
    if days_per_month < 1:
        raise ValueError("days_per_month must be positive")
    dates = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(pd.to_datetime(calendar)).normalize())))
    selected, months = [], []
    month = start
    while month < end:
        following = month + pd.offsets.MonthBegin(1)
        available = dates[(dates >= month) & (dates < following)]
        if len(available) < days_per_month:
            raise ValueError(f"{month:%Y-%m} has {len(available)} complete trading days; "
                             f"{days_per_month} per month cannot be met")
        indices = [(2 * i + 1) * len(available) // (2 * days_per_month) for i in range(days_per_month)]
        chosen = available[indices]
        selected.extend(chosen)
        months.append({"month": f"{month:%Y-%m}", "complete_trading_days": len(available),
                       "indices": indices, "selected": [str(day.date()) for day in chosen]})
        month = following
    return pd.DatetimeIndex(selected), months


def previous_trading_days(calendar: Any, days: Any) -> pd.DatetimeIndex:
    """One independently registered trading day before each selected day, for bar history."""
    dates = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(pd.to_datetime(calendar)).normalize())))
    positions = dates.get_indexer(pd.DatetimeIndex(pd.to_datetime(days)).normalize())
    if (positions <= 0).any():
        raise ValueError("Every sampled day requires a preceding calendar trading day")
    return pd.DatetimeIndex(sorted(set(dates[positions - 1])))


def _sampling_fields(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    get = value.get if isinstance(value, dict) else lambda name: getattr(value, name)
    fields = {"start": pd.Timestamp(get("start")), "end": pd.Timestamp(get("end")),
              "days_per_month": int(get("days_per_month"))}
    if fields["start"].tzinfo or fields["end"].tzinfo:
        raise ValueError("Monthly sampling requires timezone-naive timestamps")
    if fields["days_per_month"] < 1:
        raise ValueError("days_per_month must be positive")
    return fields


def load_data(config: Any, start: Any = None, end: Any = None, *,
              sealed_access: FinalAccessClaim | None = None, stream_bars: bool = False,
              chunk_rows: int = DEFAULT_BAR_CHUNK_ROWS) -> DataBundle:
    """Read only selected dates/columns; bars include configured historical warm-up.

    ``stream_bars`` reads the same bar window in bounded chunks straight into the canonical
    float32 arrays (:func:`bar_arrays_streamed`) instead of one pandas frame, and returns them
    as ``bundle.arrays`` with ``bundle.bars`` left ``None``. Projection, mapping, labels,
    universe alignment and provenance records are unchanged, so both readings are
    interchangeable for callers that only window the bars.

    Label values are retained separately. No label-derived columns enter bars or
    build_batch; inference can use the bars and universe without the label table.

    With ``monthly_sampling`` the training interval keeps only evenly spaced
    complete trading days (and one previous trading day of bar history, the
    original 128-bar recipe's warm-up), one monthly source file at a time, cached
    per projection.

    The sealed final-check interval cannot be read: the effective window
    ``[warm_start, end)`` must stay before ``SEALED_START`` unless the caller
    has already recorded the one-time final access (never in phase A).
    """
    features = list(config_option(config, "feature_columns"))
    if FORBIDDEN_FEATURES.intersection(features):
        raise ValueError("Future label columns cannot be input features")
    allowed_start, allowed_end = config_option(config, "allowed_start"), config_option(config, "allowed_end")
    start = start if start is not None else allowed_start
    end = end if end is not None else allowed_end
    if start is not None and allowed_start is not None and pd.Timestamp(start) < pd.Timestamp(allowed_start):
        raise ValueError("Requested data starts before the allowed development interval")
    if end is not None and allowed_end is not None and pd.Timestamp(end) > pd.Timestamp(allowed_end):
        raise ValueError("Requested data ends after the allowed development interval")
    sampling = _sampling_fields(config_option(config, "monthly_sampling"))
    if sampling is not None and (start is None or pd.Timestamp(start) != sampling["start"]):
        raise ValueError("Monthly sampling reads must start exactly at the sampling range")
    history_days = config_option(config, "history_days", 10)
    warm_start = None if start is None else (
        sampling["start"] if sampling is not None
        else pd.Timestamp(start) - pd.Timedelta(days=history_days))
    if warm_start is not None and allowed_start is not None:
        warm_start = max(warm_start, pd.Timestamp(allowed_start))
    if end is None and config_option(config, "labels_path"):
        raise ValueError(f"Development reads with labels require an explicit end before the sealed interval "
                         f"[{SEALED_START}, {SEALED_END})")
    if sealed_access is not None:
        sealed_access = validate_sealed_claim(config, sealed_access)
    if start is not None or end is not None:
        assert_development_read(warm_start if warm_start is not None else start, end, sealed_access=sealed_access)
    records = {}
    cache_events = []
    cache_root = config_option(config, "cache_dir")

    def read_role(role, path, columns, first, last, selection=None):
        directory = None if cache_root is None else Path(cache_root) / role
        if directory is not None and Path(path).is_dir() and Path(path).resolve().is_relative_to(Path(cache_root).resolve()):
            raise ValueError("cache_dir must not be the same directory as, or an ancestor of, an input directory")
        return _read(path, columns, first, last, directory, cache_events, selection,
                     sealed_access=sealed_access)

    mapping = None
    map_path = config_option(config, "instrument_map_path")
    if map_path:
        mapping, records["instrument_map"] = read_role("instrument_map", map_path,
                                                      ["instrument_id", "instrument"], None, None)
        if set(mapping.columns) != {"instrument_id", "instrument"} or mapping.isna().any().any():
            raise ValueError("Instrument map must contain non-null instrument_id and instrument")
        mapping = mapping.drop_duplicates().reset_index(drop=True)
        if mapping["instrument_id"].duplicated().any() or mapping["instrument"].duplicated().any():
            raise ValueError("Instrument map must be one-to-one")
    instruments, rows_selection, bars_selection, sampling_audit = None, None, None, None
    if sampling is not None:
        if not config_option(config, "instruments_path"):
            raise ValueError("Monthly sampling requires the independent instruments calendar")
        raw_universe, records["instruments"] = read_role("instruments", config_option(config, "instruments_path"),
                                                         KEYS, start, end)
        universe = validate_keys(raw_universe, "instruments")
        calendar = complete_trading_calendar(universe, config_option(config, "sample_times", []))
        days, months = monthly_sampled_days(calendar, sampling["start"], sampling["end"],
                                            sampling["days_per_month"])
        context = previous_trading_days(calendar, days)
        history_end = max(sampling["start"], sampling["end"] - pd.Timedelta(days=history_days))
        rows_selection = DaySelection(sampling["start"], sampling["end"], days)
        bars_selection = DaySelection(sampling["start"], history_end, days.union(context))
        instruments = universe.loc[rows_selection.keep(universe["date"])].reset_index(drop=True)
        if instruments.empty:
            raise ValueError("Monthly sampling selected no universe rows")
        sampling_audit = {
            "start": str(sampling["start"]), "end": str(sampling["end"]),
            "days_per_month": sampling["days_per_month"], "days": len(days),
            "selected_days": [str(day.date()) for day in days],
            "context_days": [str(day.date()) for day in context],
            "history_end": str(history_end), "months": months,
            "calendar_source": "independent instruments table",
            "rule": "centered stratified index floor((2i+1)n/(2k)) inside each calendar month",
            "reader": "one monthly source projection per cache entry"}
    arrays, bars_streamed = None, None
    if stream_bars:
        # Same projection, filters and preprocessing as the frame path below, assembled in
        # bounded chunks so the window never exists as one frame.
        arrays, records["bars"], bars_streamed = bar_arrays_streamed(
            config_option(config, "bars_path"), config, start=warm_start, end=end,
            selection=bars_selection, mapping=mapping, chunk_rows=chunk_rows,
            sealed_access=sealed_access)
        cache_events.append({"event": "cache_bypassed", "role": "bars",
                             "reason": "bounded_chunked_bars"})
        bars = None
    else:
        bars, records["bars"] = read_role("bars", config_option(config, "bars_path"),
                                          ["date", "instrument", "instrument_id", *features], warm_start, end,
                                          bars_selection)
        bars = preprocess_bars(_map_instruments(bars, mapping, "bars"), config)
    label_path = config_option(config, "labels_path")
    if label_path:
        labels, records["labels"] = read_role("labels", label_path,
                                              [*KEYS, "instrument_id", "target", "vwap_return", "label_end"],
                                              start, end, rows_selection)
        labels = validate_keys(_map_instruments(labels, mapping, "labels"), "labels")
        if "target" in labels and "vwap_return" in labels:
            if not np.allclose(labels["target"], labels["vwap_return"], equal_nan=True):
                raise ValueError("Conflicting target and vwap_return columns")
            labels = labels.drop(columns="vwap_return")
        elif "vwap_return" in labels:
            labels = labels.rename(columns={"vwap_return": "target"})
        if "target" not in labels:
            raise ValueError("labels: missing target or vwap_return")
        labels["target"] = pd.to_numeric(labels["target"], errors="raise")
        if "label_end" in labels:
            labels["label_end"] = timestamps(labels["label_end"], "labels.label_end")
    else:
        labels = pd.DataFrame(columns=[*KEYS, "target"])
    outside_policy = config_option(config, "outside_universe_labels", "exclude")
    if outside_policy not in {"exclude", "error"}:
        raise ValueError("outside_universe_labels must be exclude or error")
    alignment = {"policy": outside_policy, "labels_input_rows": len(labels),
                 "outside_universe_label_rows": 0, "outside_universe_label_instruments": 0,
                 "labels_retained_rows": len(labels)}
    data_events = []
    if config_option(config, "instruments_path") and instruments is None:
        instruments, records["instruments"] = read_role("instruments", config_option(config, "instruments_path"),
                                                        KEYS, start, end)
        instruments = validate_keys(instruments, "instruments")
    if instruments is not None and len(labels):
        # Membership is exact at the prediction timestamp, not the union
        # of securities seen at any time. Keep the universe unchanged.
        invalid = labels.merge(instruments.assign(_member=True), on=KEYS, how="left",
                               validate="one_to_one", sort=False)["_member"].isna()
        if invalid.any():
            if outside_policy == "error":
                raise ValueError("Label keys include securities outside the point-in-time universe")
            alignment["outside_universe_label_rows"] = int(invalid.sum())
            alignment["outside_universe_label_instruments"] = int(labels.loc[invalid, "instrument"].nunique())
            labels = labels.loc[~invalid].reset_index(drop=True)
            alignment["labels_retained_rows"] = len(labels)
            data_events.append({"event": "outside_universe_labels_excluded", **alignment})
    payload = json.dumps({"files": records, "monthly_sampling": sampling_audit},
                         sort_keys=True, separators=(",", ":")).encode()
    metadata = {"schema_version": 1, "data_contract_version": CONTRACT_VERSION,
                "source": config_option(config, "source"), "version": config_option(config, "version"),
                "data_sha256": hashlib.sha256(payload).hexdigest(), "files": records,
                "requested_start": None if start is None else str(pd.Timestamp(start)),
                "requested_end": None if end is None else str(pd.Timestamp(end)),
                "bars_start": None if warm_start is None else str(warm_start),
                "bar_rows": int(arrays.rows) if arrays is not None else len(bars),
                "bars_streamed": bars_streamed, "label_rows": len(labels), "alignment": alignment,
                "missing_targets": int((~np.isfinite(labels["target"].to_numpy(dtype=float))).sum()),
                "feature_columns": features, "price_scale": config_option(config, "price_scale", 1.0),
                "label_definition": ("provided target; end_price/future_vwap - 1 is inferred from the "
                                     "official vwap_return field description; original labels are used unchanged"),
                "timezone": "Asia/Shanghai (naive)", "history_days": history_days,
                "monthly_sampling": sampling_audit}
    return DataBundle(bars=bars, labels=labels, metadata=metadata, instruments=instruments,
                      cache_events=cache_events, data_events=data_events, arrays=arrays)


def preprocess_bars(bars: pd.DataFrame, config: Any) -> pd.DataFrame:
    """Shared source-unit conversion and missing-value policy for local/platform inputs."""
    features = list(config_option(config, "feature_columns"))
    bars = validate_keys(bars, "bars")
    for column in features:
        if column not in bars:
            raise ValueError(f"bars: missing feature {column}")
        bars[column] = pd.to_numeric(bars[column], errors="raise").astype(np.float64)
        sentinel = config_option(config, "missing_sentinel")
        if sentinel is not None:
            bars[column] = bars[column].mask(bars[column].eq(sentinel))
        bars[column] = bars[column].replace([np.inf, -np.inf], np.nan)
        if tuple(features) == lob_trade_columns() and column in features[12:]:
            bars[column] = bars[column].mask(bars[column] < 0)
        if column in {"open", "high", "low", "close", "pre_close", "amount"} or re.match(r"^(ask|bid)_price\d+$", column):
            bars[column] /= config_option(config, "price_scale", 1.0)
    bars = bars[[*KEYS, *features]]
    return bars


def select_targets(labels: pd.DataFrame, split: Any, horizon_minutes: int = 30,
                   *, sample_times: Sequence[str] | None = None, boundary: Any = None) -> pd.DataFrame:
    """Purge labels whose future observation window leaves the selected split.

    Explicit label_end is preferred. In its absence, 11:30 ends at 13:30 for
    30-minute labels, following the provided eight-section sampling schedule.
    This inferred trading clock is recorded; overnight windows are excluded.
    Optional sampling times select a subset without changing existing rejection
    counts, which describe all labels within the requested split.

    ``boundary`` is the exclusive start of the next interval (validation.start for a
    training selection). Every selected sample must have ``label_end < boundary``; a split
    that reaches into the next interval fails with its boundary samples instead of
    silently training on a window that overlaps it.
    """
    frame = validate_keys(labels, "labels")
    if "target" not in frame:
        if "vwap_return" not in frame:
            raise ValueError("labels: missing target")
        frame = frame.rename(columns={"vwap_return": "target"})
    start, end = pd.Timestamp(config_option(split, "start")), pd.Timestamp(config_option(split, "end"))
    boundary = end if boundary is None else pd.Timestamp(boundary)
    in_split = (frame["date"] >= start) & (frame["date"] < end)
    frame = frame.loc[in_split].copy()
    frame["target"] = pd.to_numeric(frame["target"], errors="raise")
    finite = np.isfinite(frame["target"].to_numpy(dtype=float))
    required_end = required_label_end(frame["date"], horizon_minutes)
    if "label_end" in frame:
        frame["label_end"] = timestamps(frame["label_end"], "label_end")
        if (frame["label_end"] <= frame["date"]).any():
            raise ValueError("label_end must be later than prediction date")
        if (frame["label_end"] < required_end).any():
            raise ValueError("label_end is shorter than the required prediction horizon")
        known_horizon = frame["label_end"].dt.normalize().eq(frame["date"].dt.normalize())
    else:
        frame["label_end"] = required_end
        known_horizon = known_session_horizon(frame["date"], horizon_minutes)
    if boundary < end:
        # The split itself reaches into the next interval: refuse it, naming the samples.
        assert_label_windows_before(frame, boundary, name="training split")
        raise ValueError(f"Training split [{start}, {end}) reaches the exclusive boundary {boundary}")
    approved_time = (pd.Series(True, index=frame.index) if sample_times is None
                     else frame["date"].dt.strftime("%H:%M").isin(sample_times))
    valid = finite & known_horizon & (frame["label_end"] < end) & approved_time
    selected = frame.loc[valid].reset_index(drop=True)
    assert_label_windows_before(selected, boundary, name="selected training samples")
    selected.attrs["selection"] = {"requested": len(frame), "selected": len(selected),
                                   "missing_labels": int((~finite).sum()),
                                   "unknown_session_horizon": int((~known_horizon).sum()),
                                   "boundary_purged": int((frame["label_end"] >= end).sum()),
                                   "outside_sample_times": int((~approved_time).sum()),
                                   "label_clock": ("explicit_label_end" if "label_end" in labels
                                                   else "assumed_trading_clock_1130_to_1330")}
    return selected


@dataclass
class BarArrays:
    """Canonical bar matrix and per-instrument geometry shared by both pool paths.

    One float32 copy holds every configured feature, with every non-finite source value set
    to NaN (the missing marker the windower expects). Instruments are name-sorted and the
    rows of one instrument are time-sorted, so a sample's trailing window is the contiguous
    slice ``[offsets[code] + begin, offsets[code] + stop)``. The materialised pool and the
    streaming pool therefore gather the same bytes from these arrays through
    :func:`window_arrays`, which is the only place a window is assembled.
    """

    values: np.ndarray
    dates: np.ndarray
    offsets: np.ndarray
    names: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.dates.shape[0])

    @property
    def width(self) -> int:
        return int(self.values.shape[1])

    def nbytes(self) -> dict[str, int]:
        """In-memory footprint of the canonical arrays, which bounds a streaming pool."""
        return {"bars": int(self.values.nbytes), "bar_dates": int(self.dates.nbytes),
                "instrument_offsets": int(self.offsets.nbytes)}


def _feature_matrix(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    """The one float64 -> float32 feature conversion: whole window and one chunk alike."""
    return frame[list(features)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32, copy=True)


def bar_arrays(bars: pd.DataFrame, features: Sequence[str]) -> BarArrays:
    """Sort once and keep one float32 bar copy; every non-finite value becomes NaN."""
    frame = bars.sort_values(["instrument", "date"], kind="stable")
    codes, names = pd.factorize(frame["instrument"])
    values = _feature_matrix(frame, features)
    values[~np.isfinite(values)] = np.nan
    counts = np.bincount(codes, minlength=len(names)).astype(np.int64)
    return BarArrays(values=values, dates=frame["date"].to_numpy(dtype="datetime64[ns]"),
                     offsets=np.concatenate((np.zeros(1, np.int64), np.cumsum(counts))),
                     names=np.asarray(names, dtype=object))


# --------------------------------------------------------------------------- bounded bar loading
# The 927-day window no longer fits because of the canonical bar matrix, not the window
# tensor. Reading the whole window into one pandas frame costs several float64 copies of it
# (132 days: 5 GB resident, 19.3 GB transient), which extrapolates past the host at 927 days.
# The two-pass reader below builds the same float32 arrays with peak memory
# ``target arrays + one source chunk``:
#
#   pass 1  key columns only      -> per-instrument row counts (the layout)
#   pass 2  one bounded chunk     -> values written straight into the preallocated matrix
#
# ``tests/test_bar_streaming.py`` recomputes over arbitrary chunk boundaries and
# ``scripts/verify_bar_streaming.py`` hashes both paths on a real window.

class BarArraysBuilder:
    """Two-pass assembly of the canonical instrument-major bar matrix from keyed chunks.

    ``observe`` takes pass 1 (the key columns of every chunk) and ``append`` takes pass 2 (the
    preprocessed values of the *same* chunks in the same order). The layout is exactly
    ``bars.sort_values(["instrument", "date"], kind="stable")``: instrument is the outer order
    and each instrument's rows are in time order, so the result is bit-identical to
    :func:`bar_arrays` over the concatenated window while every temporary is bounded by one
    chunk. Source chunk order is irrelevant - pass 1 records each chunk's date span per
    instrument and pass 2 places chunks by span, rejecting spans that interleave or repeat a
    key. Source files written by parallel writers (row batches in arbitrary order) therefore
    assemble correctly, and an inconsistent source fails loudly instead of reordering rows.
    """

    def __init__(self, features: Sequence[str], *, chunk_rows: int = DEFAULT_BAR_CHUNK_ROWS):
        if chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")
        self.features = list(features)
        self.chunk_rows = int(chunk_rows)
        self.chunks = 0
        self.observed_rows = 0
        self.peak_chunk_rows = 0
        self.values: np.ndarray | None = None
        self.dates: np.ndarray | None = None
        self.offsets: np.ndarray | None = None
        self.names: np.ndarray | None = None
        self._seen: set[str] = set()
        self._spans: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._code_of: dict[str, int] | None = None
        self._pending: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._appended: list[int] = []
        self._written = None

    # ------------------------------------------------------------------ pass 1
    def observe(self, chunk: pd.DataFrame) -> None:
        """Record one chunk's per-instrument date span and row count (the layout input)."""
        self.chunks += 1
        if chunk.empty:
            empty = np.zeros(0, np.int64)
            self._spans.append((np.empty(0, dtype=object), empty,
                                np.zeros(0, "datetime64[ns]"), np.zeros(0, "datetime64[ns]")))
            return
        keys = validate_keys(chunk[KEYS], "bars chunk")
        self.observed_rows += len(keys)
        self.peak_chunk_rows = max(self.peak_chunk_rows, len(keys))
        names = keys["instrument"].to_numpy(dtype=object)
        stamps = keys["date"].to_numpy(dtype="datetime64[ns]")
        unique, codes = np.unique(names, return_inverse=True)
        counts = np.bincount(codes, minlength=len(unique)).astype(np.int64)
        bounds = np.concatenate(([0], np.cumsum(counts)))
        grouped = stamps[np.argsort(codes, kind="stable")]
        self._spans.append((unique, counts,
                            np.minimum.reduceat(grouped, bounds[:-1]),
                            np.maximum.reduceat(grouped, bounds[:-1])))
        self._seen.update(unique.tolist())

    # ------------------------------------------------------------------ layout
    def layout(self) -> None:
        """Freeze instrument order and offsets and allocate the target arrays once.

        Each ``(chunk, instrument)`` pair becomes a run with a date span; ordering the runs of
        one instrument by span start reproduces the time order of the whole window without
        ever holding it.
        """
        if self._code_of is not None:
            return
        names = pd.Index(sorted(self._seen)) if self._seen else pd.Index([], dtype=object)
        self._code_of = {name: code for code, name in enumerate(names)}
        if not names.size:
            self.values = np.empty((0, len(self.features)), dtype=np.float32)
            self.dates = np.empty(0, dtype="datetime64[ns]")
            self.offsets = np.zeros(1, np.int64)
            self.names = np.asarray([], dtype=object)
            self._written = np.zeros(0, np.int64)
            return
        codes, counts, mins, maxs, sizes = [], [], [], [], []
        for unique, count, low, high in self._spans:
            codes.append(np.fromiter((self._code_of[name] for name in unique), np.int64, len(unique)))
            counts.append(count)
            mins.append(low.astype(np.int64))
            maxs.append(high.astype(np.int64))
            sizes.append(len(unique))
        codes, counts = np.concatenate(codes), np.concatenate(counts)
        mins, maxs = np.concatenate(mins), np.concatenate(maxs)
        totals = np.bincount(codes, weights=counts, minlength=len(names)).astype(np.int64)
        offsets = np.concatenate((np.zeros(1, np.int64), np.cumsum(totals))).astype(np.int64)
        order = np.lexsort((mins, codes))
        ordered_codes, ordered_counts = codes[order], counts[order]
        ordered_min, ordered_max = mins[order], maxs[order]
        clash = np.zeros(len(order), dtype=bool)
        clash[1:] = (ordered_codes[1:] == ordered_codes[:-1]) & (ordered_min[1:] <= ordered_max[:-1])
        if clash.any():
            name = names[ordered_codes[np.flatnonzero(clash)[0]]]
            raise ValueError(f"bars chunks overlap in time for {name}: a date range is split "
                             "across two chunks, so their order is ambiguous")
        written_before = np.concatenate((np.zeros(1, np.int64), np.cumsum(ordered_counts)))[:-1]
        group = np.concatenate(([0], np.flatnonzero(ordered_codes[1:] != ordered_codes[:-1]) + 1))
        group_sizes = np.diff(np.concatenate((group, [len(order)])))
        starts_ordered = offsets[ordered_codes] + written_before - np.repeat(written_before[group], group_sizes)
        starts = np.empty_like(starts_ordered)
        starts[order] = starts_ordered
        bounds = np.concatenate(([0], np.cumsum(sizes)))
        self._pending = [(codes[bounds[i]:bounds[i + 1]], counts[bounds[i]:bounds[i + 1]],
                          starts[bounds[i]:bounds[i + 1]]) for i in range(self.chunks)]
        self.values = np.empty((int(offsets[-1]), len(self.features)), dtype=np.float32)
        self.dates = np.empty(int(offsets[-1]), dtype="datetime64[ns]")
        self.offsets = offsets
        self.names = np.asarray(list(names), dtype=object)
        self._written = np.zeros(len(names), dtype=np.int64)

    # ------------------------------------------------------------------ pass 2
    def append(self, chunk: pd.DataFrame) -> None:
        """Write one chunk's preprocessed values into the matrix at its laid-out positions."""
        self.layout()
        index = len(self._appended)
        if index >= self.chunks:
            raise ValueError("bar chunks were appended more often than observed")
        codes, expected, starts = self._pending[index]
        frame = chunk.reset_index(drop=True)
        if len(frame) > self.chunk_rows:
            raise ValueError(f"bar chunk of {len(frame)} rows exceeds chunk_rows={self.chunk_rows}")
        if not len(frame):
            if len(expected):
                raise ValueError("bar source changed between the layout and value passes")
            self._appended.append(index)
            return
        chunk_codes = np.fromiter((self._code_of[name] for name in frame["instrument"].to_numpy(dtype=object)),
                                  dtype=np.int64, count=len(frame))
        counts = np.bincount(chunk_codes, minlength=len(self.names))
        if not np.array_equal(counts[codes], expected):
            raise ValueError("bar source changed between the layout and value passes")
        stamps = frame["date"].to_numpy(dtype="datetime64[ns]")
        # Instrument-major inside the chunk, then its own time order: the global stable sort is
        # the concatenation of these blocks because the layout proved the spans are disjoint.
        order = np.lexsort((stamps, chunk_codes))
        ordered_codes = chunk_codes[order]
        start_of = np.zeros(len(self.names), dtype=np.int64)
        start_of[codes] = starts
        group_start = np.searchsorted(ordered_codes, ordered_codes, side="left")
        destination = start_of[ordered_codes] + (np.arange(len(order)) - group_start)
        values = _feature_matrix(frame.iloc[order], self.features)
        values[~np.isfinite(values)] = np.nan
        self.values[destination] = values
        self.dates[destination] = stamps[order]
        self._written += np.bincount(ordered_codes, minlength=len(self.names))
        self._appended.append(index)

    # ------------------------------------------------------------------ result
    def result(self) -> BarArrays:
        """The canonical arrays; refuses a half-filled matrix instead of returning one."""
        self.layout()
        if self._appended != list(range(self.chunks)):
            raise ValueError(f"{len(self._appended)} of {self.chunks} bar chunks were never filled")
        if not np.array_equal(self._written, np.diff(self.offsets)):
            raise ValueError("bar matrix is not fully written; refusing a partial result")
        return BarArrays(values=self.values, dates=self.dates, offsets=self.offsets, names=self.names)


def _selected(frame: pd.DataFrame, selection: DaySelection | None) -> pd.DataFrame:
    """Apply a monthly day selection exactly as the cached projection does."""
    return frame if selection is None else frame.loc[selection.keep(frame["date"])].reset_index(drop=True)


def bar_arrays_streamed(path_value: Any, config: Any, *, start: Any = None, end: Any = None,
                        selection: DaySelection | None = None, mapping: pd.DataFrame | None = None,
                        chunk_rows: int = DEFAULT_BAR_CHUNK_ROWS,
                        sealed_access: FinalAccessClaim | None = None
                        ) -> tuple[BarArrays, list[dict], dict]:
    """Build the canonical ``BarArrays`` straight from source files, one bounded chunk at a time.

    Same projection, filters, instrument mapping and preprocessing as
    ``load_data`` + :func:`bar_arrays`; the only difference is that no frame ever holds the
    whole window, so peak memory is the target arrays plus one chunk. Verified caches are
    bypassed (they key on a whole-window projection), so the source is read twice: pass 1
    reads the key columns, pass 2 the configured features. Returns the arrays, the same
    per-file provenance records ``_read`` produces, and an audit of the chunking.
    """
    if start is not None or end is not None:
        assert_development_read(start, end, sealed_access=sealed_access)
    if not path_value:
        raise ValueError("Bounded bar streaming needs data.bars_path")
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Data source does not exist: {path}")
    if str(path).lower().endswith((".tar.gz", ".tgz", ".tar", ".rar")):
        raise ValueError("Bounded bar streaming reads extracted monthly feather/parquet files; "
                         "extract the archive first or load the window as a frame")
    features = list(config_option(config, "feature_columns"))
    if FORBIDDEN_FEATURES.intersection(features):
        raise ValueError("Future label columns cannot be input features")
    files = _source_files(path, start, end, None)
    if not files:
        raise ValueError(f"No data files overlap requested dates: {path}")
    key_columns = list(dict.fromkeys([*KEYS, "instrument_id"]))
    value_columns = list(dict.fromkeys([*key_columns, *features]))
    builder = BarArraysBuilder(features, chunk_rows=chunk_rows)
    states, records = {}, []
    for file in files:
        before = file.stat()
        states[file] = (before.st_size, before.st_mtime_ns)
        records.append(_digest(file))
        for chunk in _file_blocks(file, key_columns, start, end, selection, chunk_rows):
            builder.observe(_map_instruments(chunk, mapping, "bars"))
    builder.layout()
    for file in files:
        for chunk in _file_blocks(file, value_columns, start, end, selection, chunk_rows):
            builder.append(preprocess_bars(_map_instruments(chunk, mapping, "bars"), config))
        after = file.stat()
        if (after.st_size, after.st_mtime_ns) != states[file]:
            raise ValueError(f"Data changed during read: {file}")
    arrays = builder.result()
    audit = {"mode": "bounded_chunks", "chunk_rows": builder.chunk_rows, "chunks": builder.chunks,
             "max_chunk_rows": int(builder.peak_chunk_rows), "source_files": len(files),
             "passes": 2, "rows": arrays.rows, "instruments": len(arrays.names),
             "cache": "bypassed (whole-window projection keys)",
             "layout": "instrument-major, date ascending (same as bars.sort_values(['instrument','date']))"}
    return arrays, records, audit


def instrument_codes(arrays: BarArrays, instruments: pd.Series) -> np.ndarray:
    """Bar-group code per key; -1 keeps a key whose instrument has no bars addressable."""
    if not arrays.rows:
        return np.full(len(instruments), -1, dtype=np.int64)
    return pd.Index(arrays.names).get_indexer(pd.Index(instruments)).astype(np.int64)


def window_parameters(config: Any) -> dict:
    """Window geometry and missing-value policy, validated once for both pool paths."""
    lookback, minimum = int(config_option(config, "lookback")), int(config_option(config, "min_history", 1))
    if not 1 <= minimum <= lookback:
        raise ValueError("min_history must be within [1, lookback]")
    policy, limit = config_option(config, "missing_policy", "zero"), int(config_option(config, "max_ffill_steps", 0))
    if policy not in {"zero", "ffill"} or limit < 0:
        raise ValueError("Invalid missing-value policy")
    return {"lookback": lookback, "minimum": minimum, "policy": policy, "limit": limit}


MINUTE_NS = 60_000_000_000


def approved_sample_minutes(times) -> np.ndarray | None:
    """Minute-of-day values whose ``strftime("%H:%M")`` spelling is approved by the config."""
    if not times:
        return None
    spelled = {str(value) for value in times}
    return np.array([minute for minute in range(1440)
                     if f"{minute // 60:02d}:{minute % 60:02d}" in spelled], dtype=np.int32)


def validate_sample_times(keys: pd.DataFrame, times, approved: np.ndarray | None = None) -> np.ndarray:
    """Approved whole-minute keys only; a streamed batch repeats this on every draw.

    The clock test is the numeric form of ``date.dt.strftime("%H:%M").isin(times)`` - identical
    for every timestamp, because that is exactly how the string is spelled - so a per-batch
    draw can re-run it without pandas' formatting cost.
    """
    dates = keys["date"].to_numpy(dtype="datetime64[ns]")
    minutes = dates.astype(np.int64) // MINUTE_NS
    if approved is None:
        approved = approved_sample_minutes(times)
    if approved is not None and not np.isin(minutes % 1440, approved).all():
        raise ValueError("Sample keys contain an unapproved sampling time")
    if np.any(dates.astype(np.int64) % MINUTE_NS):
        raise ValueError("Sample timestamps must be precise whole minutes")
    return dates


def window_geometry(arrays: BarArrays, codes: np.ndarray, key_dates: np.ndarray,
                    lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-key ``(begin, stop)``: the trailing ``lookback`` bars ending at the key timestamp."""
    stop = np.zeros(len(codes), dtype=np.int64)
    present = codes >= 0
    if arrays.rows and present.any():
        positions = np.flatnonzero(present)
        order = np.argsort(codes[positions], kind="stable")
        positions, ordered = positions[order], codes[positions][order]
        bounds = np.searchsorted(ordered, np.arange(len(arrays.names) + 1))
        for code in np.unique(ordered):
            rows = positions[bounds[code]:bounds[code + 1]]
            local = arrays.dates[arrays.offsets[code]:arrays.offsets[code + 1]]
            stop[rows] = np.searchsorted(local, key_dates[rows], side="right")
    return np.maximum(stop - lookback, 0), stop


def forward_fill(values: np.ndarray, limit: int) -> np.ndarray:
    """Fill NaN runs of at most ``limit`` steps along time, matching ``DataFrame.ffill``.

    The gap is measured from the previous finite value in the same (row, feature) series, so
    the result is identical to the per-window pandas call this replaces.
    """
    finite = np.isfinite(values)
    if finite.all():                     # nothing to fill: no temporaries at all
        return values
    index = np.arange(values.shape[1], dtype=np.int32)
    last = np.maximum.accumulate(np.where(finite, index[None, :, None], np.int32(-1)), axis=1)
    gap = index[None, :, None] - last
    fill = (~finite) & (last >= 0) & (gap <= limit)
    if fill.any():
        rows, columns, features = np.nonzero(fill)
        values[rows, columns, features] = values[rows, last[rows, columns, features], features]
    return values


def relative_price_log_volume(window: np.ndarray, features: Sequence[str]) -> None:
    """Apply ``price_rel_vol_log`` in place to one block of gathered windows.

    Prices become ``(p - ref) / ref`` with ``ref`` the mid price ``(ask_price1 + bid_price1)/2``
    at the window's own last step, so the transform is causal, per (instrument, window) and
    independent of the instrument's absolute price level. A missing or non-positive mid zeroes
    that window's price channels instead of dividing by it. Volumes become ``log1p(v)`` for
    ``v >= 0``. Applied after forward fill and before the final NaN->0 fill on the one window
    path shared by the materialised, streaming and inference builders.
    """
    names = list(features)
    if len(names) != window.shape[2]:
        raise ValueError("feature names do not match the window width")
    try:
        ask, bid = names.index("ask_price1"), names.index("bid_price1")
    except ValueError:
        raise ValueError("price_rel_vol_log requires ask_price1 and bid_price1") from None
    mid = 0.5 * (window[:, -1, ask] + window[:, -1, bid])
    valid = np.isfinite(mid) & (mid > 0)
    reference = np.where(valid, mid, 1.0)[:, None, None]
    prices = [index for index, name in enumerate(names) if "price" in name]
    if prices:
        values = (window[:, :, prices] - reference) / reference
        values[~valid] = 0.0
        window[:, :, prices] = values
    volumes = [index for index, name in enumerate(names) if "volume" in name]
    if volumes:
        window[:, :, volumes] = np.log1p(np.maximum(window[:, :, volumes], 0.0))


def window_arrays(arrays: BarArrays, codes: np.ndarray, begin: np.ndarray, stop: np.ndarray,
                  x: np.ndarray, observed: np.ndarray, history_dates: np.ndarray, *,
                  lookback: int, minimum: int, policy: str, limit: int,
                  history_feature_count: int, features: Sequence[str] = (),
                  feature_transform: str = "none", chunk: int | None = None):
    """Assemble every key's trailing window into the preallocated outputs.

    The single window builder: the materialised pool fills one pool-sized array set and the
    streaming pool fills one batch's worth, so both paths produce the same values by
    construction. Windows are right-aligned (short history pads the front with NaN, which
    ``observed`` reports as unobserved and ``x`` reports as zero), ``enough`` counts rows with
    any finite primary feature, and ``observed`` is measured before any filling. ``features``
    and ``feature_transform`` apply the configured input transform after forward fill and before
    the final zero fill, so training, validation and prediction windows see the same values.
    Returns ``(enough, history_count)``.
    """
    if feature_transform not in ("none", "price_rel_vol_log"):
        raise ValueError(f"Unsupported feature_transform: {feature_transform!r}")
    count = len(codes)
    if not arrays.rows:
        return np.zeros(count, dtype=bool), np.zeros(count, dtype=np.int64)
    # Bounded temporaries: the gather alone is (chunk, lookback, features) float32, and a
    # forward fill adds a few integer intermediates of the same shape.
    chunk = chunk or (2048 if policy == "ffill" and limit else 8192)
    columns = np.arange(lookback)
    enough = np.zeros(count, dtype=bool)
    for start_row in range(0, count, chunk):
        rows = slice(start_row, min(start_row + chunk, count))
        relative = columns[None, :] - (lookback - (stop[rows] - begin[rows]))[:, None]
        valid = relative >= 0
        base = np.where(codes[rows] >= 0, arrays.offsets[np.maximum(codes[rows], 0)], 0)
        absolute = np.clip(base[:, None] + begin[rows][:, None] + np.where(valid, relative, 0),
                           0, max(arrays.rows - 1, 0))
        window = arrays.values[absolute]
        window[~valid] = np.nan
        observed[rows] = np.isfinite(window)
        dates = arrays.dates[absolute]
        dates[~valid] = np.datetime64("NaT", "ns")
        history_dates[rows] = dates
        enough[rows] = (np.isfinite(window[:, :, :history_feature_count]).any(axis=2).sum(axis=1) >= minimum)
        if policy == "ffill" and limit:
            # Interval trades are flows; filling them repeats a past interval's activity.
            window[:, :, :history_feature_count] = forward_fill(window[:, :, :history_feature_count], limit)
        if feature_transform != "none":
            relative_price_log_volume(window, features)
        np.nan_to_num(window, copy=False, nan=0.0)
        x[rows] = window
    return enough, stop - begin


def window_coverage(keys: pd.DataFrame, enough: np.ndarray, *, observed_count: int,
                    observed_size: int, history_count: np.ndarray) -> dict:
    """Coverage from counts, so a streamed pool can report the identical numbers."""
    coverage_by_time = keys.assign(usable=enough).groupby("date", observed=True)["usable"].agg(["size", "sum"])
    sections = [{"date": str(date), "expected": int(row["size"]), "usable": int(row["sum"]),
                 "coverage": float(row["sum"] / row["size"])} for date, row in coverage_by_time.iterrows()]
    return {"expected_samples": len(keys), "samples": len(keys), "usable_samples": int(enough.sum()),
            "insufficient_history": int((~enough).sum()),
            "coverage": float(enough.mean()) if len(keys) else 0.0,
            "min_section_coverage": min((row["coverage"] for row in sections), default=0.0),
            "observed_fraction": (observed_count / observed_size) if observed_size else 0.0,
            "history_min": int(history_count.min()) if len(keys) else 0,
            "history_max": int(history_count.max()) if len(keys) else 0,
            "by_section": sections}


def build_batch(bars: pd.DataFrame, keys: pd.DataFrame, config: Any) -> Batch:
    """Build fixed-size trailing windows without accepting or accessing labels."""
    features = list(config_option(config, "feature_columns"))
    history_feature_count = 12 if tuple(features) == lob_trade_columns() else len(features)
    if FORBIDDEN_FEATURES.intersection(features):
        raise ValueError("Future label columns cannot be input features")
    bars = validate_keys(bars, "bars")
    keys = validate_keys(keys[KEYS], "sample keys")
    if any(column not in bars for column in features):
        raise ValueError("Missing configured feature columns")
    key_dates = validate_sample_times(keys, config_option(config, "sample_times", []))
    settings = window_parameters(config)
    feature_transform = config_option(config, "feature_transform", "none")
    arrays = bar_arrays(bars, features)
    codes = instrument_codes(arrays, keys["instrument"])
    begin, stop = window_geometry(arrays, codes, key_dates, settings["lookback"])
    x = np.zeros((len(keys), settings["lookback"], len(features)), dtype=np.float32)
    observed = np.zeros_like(x, dtype=bool)
    history_dates = np.full((len(keys), settings["lookback"]), np.datetime64("NaT", "ns"),
                            dtype="datetime64[ns]")
    enough, history_count = window_arrays(arrays, codes, begin, stop, x, observed, history_dates,
                                          history_feature_count=history_feature_count,
                                          features=tuple(features), feature_transform=feature_transform,
                                          **settings)
    # Insufficient windows remain in the batch to preserve universe alignment.
    # The runner rejects their coverage when acceptance.min_coverage is not met.
    coverage = window_coverage(keys, enough, observed_count=int(observed.sum()),
                               observed_size=int(observed.size), history_count=history_count)
    batch = Batch(keys, x, observed, history_dates, coverage, history_feature_count)
    coverage["history_quality"] = history_quality(history_diagnostics(batch), features, history_feature_count)
    return batch


def history_quality(quality: pd.DataFrame, features: Sequence[str], history_feature_count: int) -> dict:
    """Summary of one key table's history quality, shared by both pool paths."""
    return {
        "definition": "Observed means finite before filling; effective rows use primary history features; complete rows have all",
        "history_features": list(features)[:history_feature_count],
        "age_unit": "elapsed calendar minutes (includes lunch, overnight and holidays); null means no history",
        "full_observed_window_samples": int(quality.full_observed_window.sum()),
        "statistics": {column: {"min": _finite_stat(quality[column], "min"),
                                "median": _finite_stat(quality[column], "median"),
                                "max": _finite_stat(quality[column], "max")}
                       for column in ("effective_history_length", "complete_history_length", "missing_fraction",
                                      "latest_effective_age_minutes", "history_span_minutes")}}


def _finite_stat(values, operation):
    values = values.dropna()
    if values.empty:
        return None
    value = getattr(values, operation)()
    return float(value) if pd.notna(value) else None


def history_diagnostics(batch: Batch) -> pd.DataFrame:
    """Per-key source quality, measured before ffill/zero padding; never inferred from coverage."""
    dates, observed = batch.history_dates, batch.observed
    present = ~np.isnat(dates)
    effective = batch.effective_rows
    complete = observed.all(axis=2)
    count = present.sum(axis=1)
    cutoff = batch.keys.date.to_numpy(dtype="datetime64[ns]")
    # Integer extrema avoid NaT propagation in max/min, including all-missing windows.
    integer = dates.astype(np.int64)
    last_effective = np.where(effective, integer, np.iinfo(np.int64).min).max(axis=1).astype("datetime64[ns]")
    first = np.where(present, integer, np.iinfo(np.int64).max).min(axis=1).astype("datetime64[ns]")
    last = dates[:, -1]
    span = (last - first) / np.timedelta64(1, "m")
    span[count == 0] = np.nan
    feature_count = observed.shape[2]
    source_slots = count * feature_count
    return batch.keys.assign(
        history_row_count=count, effective_history_length=effective.sum(axis=1),
        complete_history_length=complete.sum(axis=1),
        missing_fraction=1 - observed.mean(axis=(1, 2)),
        source_missing_fraction=1 - np.divide(observed.sum(axis=(1, 2)), source_slots,
                                              out=np.full(len(count), np.nan), where=source_slots > 0),
        latest_bar_age_minutes=(cutoff - last) / np.timedelta64(1, "m"),
        latest_effective_age_minutes=(cutoff - last_effective) / np.timedelta64(1, "m"),
        history_span_minutes=span, full_observed_window=complete.all(axis=1))
