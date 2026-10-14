"""Versioned local data readers and causal, label-free history windows.

All input timestamps are exchange-local, timezone-naive bar *end* timestamps.
Instrument IDs are never inferred from security codes: an explicit map is required.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import uuid
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

KEYS = ["date", "instrument"]
FORBIDDEN_FEATURES = {"target", "vwap_return", "label_end", "end_price", "vwap"}
READER_FORMAT_VERSION = 1


@dataclass
class DataBundle:
    bars: pd.DataFrame
    labels: pd.DataFrame
    metadata: dict[str, Any]
    instruments: pd.DataFrame | None = None
    cache_events: list[dict[str, Any]] = field(default_factory=list)
    data_events: list[dict[str, Any]] = field(default_factory=list)

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


def _get(config: Any, name: str, default: Any = None) -> Any:
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _dates(values: pd.Series, name: str) -> pd.Series:
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
    result["date"] = _dates(result["date"], f"{name}.date")
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


def _time_filter(frame: pd.DataFrame, start: Any, end: Any) -> pd.DataFrame:
    if "date" not in frame:
        return frame
    frame = frame.copy()
    frame["date"] = _dates(frame["date"], "date")
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
                 events: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Use a verified projection cache; never include runtime cache state in provenance."""
    identity = {"reader_version": READER_FORMAT_VERSION, "source_sha256": source_record["sha256"],
                "format": path.suffix.lower(), "columns": list(dict.fromkeys(requested)),
                "start": None if start is None else str(pd.Timestamp(start)),
                "end": None if end is None else str(pd.Timestamp(end)),
                "pandas_version": pd.__version__, "pyarrow_version": pa.__version__}
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


def _read(path_value: Any, requested: list[str], start: Any, end: Any,
          cache_dir: Path | None = None, cache_events: list[dict] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Data source does not exist: {path}")
    if str(path).lower().endswith((".tar.gz", ".tgz", ".tar")):
        raise ValueError("Extract selected monthly feather files from the bar archive first; "
                         "provide instrument_map_path for instrument_id. Full archive scans are not automatic.")
    files = sorted(file for file in path.rglob("*") if file.is_file()
                   and file.suffix.lower() in {".parquet", ".feather", ".arrow", ".csv"}
                   and (cache_dir is None or not file.resolve().is_relative_to(cache_dir.parent.resolve()))
                   and _month_overlaps(file.name, start, end)) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No data files overlap requested dates: {path}")
    frames, records = [], []
    cache_events = [] if cache_events is None else cache_events
    for file in files:
        before = file.stat()
        record = _digest(file)
        frame, extra = _cached_read(file, requested, start, end, record,
                                   (before.st_size, before.st_mtime_ns), cache_dir, cache_events)
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


def load_data(config: Any, start: Any = None, end: Any = None) -> DataBundle:
    """Read only selected dates/columns; bars include configured historical warm-up.

    Label values are retained separately. No label-derived columns enter bars or
    build_batch; inference can use the bars and universe without the label table.
    """
    features = list(_get(config, "feature_columns"))
    if FORBIDDEN_FEATURES.intersection(features):
        raise ValueError("Future label columns cannot be input features")
    allowed_start, allowed_end = _get(config, "allowed_start"), _get(config, "allowed_end")
    start = start if start is not None else allowed_start
    end = end if end is not None else allowed_end
    if start is not None and allowed_start is not None and pd.Timestamp(start) < pd.Timestamp(allowed_start):
        raise ValueError("Requested data starts before the allowed development interval")
    if end is not None and allowed_end is not None and pd.Timestamp(end) > pd.Timestamp(allowed_end):
        raise ValueError("Requested data ends after the allowed development interval")
    warm_start = None if start is None else pd.Timestamp(start) - pd.Timedelta(days=_get(config, "history_days", 10))
    if warm_start is not None and allowed_start is not None:
        warm_start = max(warm_start, pd.Timestamp(allowed_start))
    records = {}
    cache_events = []
    cache_root = _get(config, "cache_dir")

    def read_role(role, path, columns, first, last):
        directory = None if cache_root is None else Path(cache_root) / role
        if directory is not None and Path(path).is_dir() and Path(path).resolve().is_relative_to(Path(cache_root).resolve()):
            raise ValueError("cache_dir must not be the same directory as, or an ancestor of, an input directory")
        return _read(path, columns, first, last, directory, cache_events)

    mapping = None
    map_path = _get(config, "instrument_map_path")
    if map_path:
        mapping, records["instrument_map"] = read_role("instrument_map", map_path,
                                                      ["instrument_id", "instrument"], None, None)
        if set(mapping.columns) != {"instrument_id", "instrument"} or mapping.isna().any().any():
            raise ValueError("Instrument map must contain non-null instrument_id and instrument")
        mapping = mapping.drop_duplicates().reset_index(drop=True)
        if mapping["instrument_id"].duplicated().any() or mapping["instrument"].duplicated().any():
            raise ValueError("Instrument map must be one-to-one")
    bars, records["bars"] = read_role("bars", _get(config, "bars_path"),
                                      ["date", "instrument", "instrument_id", *features], warm_start, end)
    bars = validate_keys(_map_instruments(bars, mapping, "bars"), "bars")
    bars = preprocess_bars(bars, config)
    label_path = _get(config, "labels_path")
    if label_path:
        labels, records["labels"] = read_role("labels", label_path,
                                              [*KEYS, "instrument_id", "target", "vwap_return", "label_end"],
                                              start, end)
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
            labels["label_end"] = _dates(labels["label_end"], "labels.label_end")
    else:
        labels = pd.DataFrame(columns=[*KEYS, "target"])
    instruments = None
    outside_policy = _get(config, "outside_universe_labels", "exclude")
    if outside_policy not in {"exclude", "error"}:
        raise ValueError("outside_universe_labels must be exclude or error")
    alignment = {"policy": outside_policy, "labels_input_rows": len(labels),
                 "outside_universe_label_rows": 0, "outside_universe_label_instruments": 0,
                 "labels_retained_rows": len(labels)}
    data_events = []
    if _get(config, "instruments_path"):
        instruments, records["instruments"] = read_role("instruments", _get(config, "instruments_path"),
                                                        KEYS, start, end)
        instruments = validate_keys(instruments, "instruments")
        if len(labels):
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
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    metadata = {"schema_version": 1, "source": _get(config, "source"), "version": _get(config, "version"),
                "data_sha256": hashlib.sha256(payload).hexdigest(), "files": records,
                "requested_start": None if start is None else str(pd.Timestamp(start)),
                "requested_end": None if end is None else str(pd.Timestamp(end)),
                "bars_start": None if warm_start is None else str(warm_start),
                "bar_rows": len(bars), "label_rows": len(labels), "alignment": alignment,
                "missing_targets": int((~np.isfinite(labels["target"].to_numpy(dtype=float))).sum()),
                "feature_columns": features, "price_scale": _get(config, "price_scale", 1.0),
                "label_definition": ("provided target; end_price/future_vwap - 1 is inferred from the "
                                     "official vwap_return field description; original labels are used unchanged"),
                "timezone": "Asia/Shanghai (naive)", "history_days": _get(config, "history_days", 10)}
    return DataBundle(bars=bars, labels=labels, metadata=metadata, instruments=instruments,
                      cache_events=cache_events, data_events=data_events)


def preprocess_bars(bars: pd.DataFrame, config: Any) -> pd.DataFrame:
    """Shared source-unit conversion and missing-value policy for local/platform inputs."""
    features = list(_get(config, "feature_columns"))
    bars = validate_keys(bars, "bars")
    for column in features:
        if column not in bars:
            raise ValueError(f"bars: missing feature {column}")
        bars[column] = pd.to_numeric(bars[column], errors="raise").astype(np.float64)
        sentinel = _get(config, "missing_sentinel")
        if sentinel is not None:
            bars[column] = bars[column].mask(bars[column].eq(sentinel))
        bars[column] = bars[column].replace([np.inf, -np.inf], np.nan)
        if column in {"open", "high", "low", "close", "pre_close", "amount"} or re.match(r"^(ask|bid)_price\d+$", column):
            bars[column] /= _get(config, "price_scale", 1.0)
    bars = bars[[*KEYS, *features]]
    return bars


def select_targets(labels: pd.DataFrame, split: Any, horizon_minutes: int = 30) -> pd.DataFrame:
    """Purge labels whose future observation window leaves the selected split.

    Explicit label_end is preferred. In its absence, 11:30 ends at 13:30 for
    30-minute labels, following the provided eight-section sampling schedule.
    This inferred trading clock is recorded; overnight windows are excluded.
    """
    frame = validate_keys(labels, "labels")
    if "target" not in frame:
        if "vwap_return" not in frame:
            raise ValueError("labels: missing target")
        frame = frame.rename(columns={"vwap_return": "target"})
    start, end = pd.Timestamp(_get(split, "start")), pd.Timestamp(_get(split, "end"))
    in_split = (frame["date"] >= start) & (frame["date"] < end)
    frame = frame.loc[in_split].copy()
    frame["target"] = pd.to_numeric(frame["target"], errors="raise")
    finite = np.isfinite(frame["target"].to_numpy(dtype=float))
    required_end = frame["date"] + pd.Timedelta(minutes=horizon_minutes)
    minute = frame["date"].dt.hour * 60 + frame["date"].dt.minute
    lunch = minute.eq(690) & (horizon_minutes == 30)
    required_end.loc[lunch] = frame.loc[lunch, "date"] + pd.Timedelta(hours=2)
    if "label_end" in frame:
        frame["label_end"] = _dates(frame["label_end"], "label_end")
        if (frame["label_end"] <= frame["date"]).any():
            raise ValueError("label_end must be later than prediction date")
        if (frame["label_end"] < required_end).any():
            raise ValueError("label_end is shorter than the required prediction horizon")
        known_horizon = frame["label_end"].dt.normalize().eq(frame["date"].dt.normalize())
    else:
        frame["label_end"] = required_end
        end_minute = minute + horizon_minutes
        known_horizon = (((minute >= 570) & (end_minute <= 690))
                         | ((minute >= 780) & (end_minute <= 900)) | lunch)
    valid = finite & known_horizon & (frame["label_end"] < end)
    selected = frame.loc[valid].reset_index(drop=True)
    selected.attrs["selection"] = {"requested": len(frame), "selected": len(selected),
                                   "missing_labels": int((~finite).sum()),
                                   "unknown_session_horizon": int((~known_horizon).sum()),
                                   "boundary_purged": int((frame["label_end"] >= end).sum()),
                                   "label_clock": ("explicit_label_end" if "label_end" in labels
                                                   else "assumed_trading_clock_1130_to_1330")}
    return selected


def build_batch(bars: pd.DataFrame, keys: pd.DataFrame, config: Any) -> Batch:
    """Build fixed-size trailing windows without accepting or accessing labels."""
    features = list(_get(config, "feature_columns"))
    if FORBIDDEN_FEATURES.intersection(features):
        raise ValueError("Future label columns cannot be input features")
    bars = validate_keys(bars, "bars")
    keys = validate_keys(keys[KEYS], "sample keys")
    if any(column not in bars for column in features):
        raise ValueError("Missing configured feature columns")
    times = _get(config, "sample_times", [])
    if times and not keys["date"].dt.strftime("%H:%M").isin(times).all():
        raise ValueError("Sample keys contain an unapproved sampling time")
    if not keys["date"].eq(keys["date"].dt.floor("min")).all():
        raise ValueError("Sample timestamps must be precise whole minutes")
    lookback, minimum = int(_get(config, "lookback")), int(_get(config, "min_history", 1))
    if not 1 <= minimum <= lookback:
        raise ValueError("min_history must be within [1, lookback]")
    policy, limit = _get(config, "missing_policy", "zero"), int(_get(config, "max_ffill_steps", 0))
    if policy not in {"zero", "ffill"} or limit < 0:
        raise ValueError("Invalid missing-value policy")
    x = np.zeros((len(keys), lookback, len(features)), dtype=np.float32)
    observed = np.zeros_like(x, dtype=bool)
    history_dates = np.full((len(keys), lookback), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    groups = {}
    for instrument, frame in bars.groupby("instrument", sort=False, observed=True):
        frame = frame.sort_values("date")
        values = frame[features].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        values[~np.isfinite(values)] = np.nan
        groups[instrument] = (frame["date"].to_numpy(dtype="datetime64[ns]"), values)
    enough = np.zeros(len(keys), dtype=bool)
    history_count = np.zeros(len(keys), dtype=np.int64)
    for index, (date, instrument) in enumerate(keys.itertuples(index=False, name=None)):
        if instrument not in groups:
            continue
        dates, values = groups[instrument]
        stop = int(np.searchsorted(dates, date.to_datetime64(), side="right"))
        begin = max(0, stop - lookback)
        length = stop - begin
        if not length:
            continue
        history_count[index] = length
        window = values[begin:stop].copy()
        observed[index, -length:] = np.isfinite(window)
        enough[index] = int(np.isfinite(window).any(axis=1).sum()) >= minimum
        if policy == "ffill" and limit:
            window = pd.DataFrame(window).ffill(limit=limit).to_numpy(dtype=np.float32)
        x[index, -length:] = np.nan_to_num(window, nan=0.0)
        history_dates[index, -length:] = dates[begin:stop]
    # Insufficient windows remain in the batch to preserve universe alignment.
    # The runner rejects their coverage when acceptance.min_coverage is not met.
    coverage_by_time = keys.assign(usable=enough).groupby("date", observed=True)["usable"].agg(["size", "sum"])
    sections = [{"date": str(date), "expected": int(row["size"]), "usable": int(row["sum"]),
                 "coverage": float(row["sum"] / row["size"])} for date, row in coverage_by_time.iterrows()]
    coverage = {"expected_samples": len(keys), "samples": len(keys), "usable_samples": int(enough.sum()),
                "insufficient_history": int((~enough).sum()),
                "coverage": float(enough.mean()) if len(keys) else 0.0,
                "min_section_coverage": min((row["coverage"] for row in sections), default=0.0),
                "observed_fraction": float(observed.mean()) if observed.size else 0.0,
                "history_min": int(history_count.min()) if len(keys) else 0,
                "history_max": int(history_count.max()) if len(keys) else 0,
                "by_section": sections}
    return Batch(keys, x, observed, history_dates, coverage)
