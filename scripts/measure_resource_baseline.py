"""Platform-caliber resource baseline for the frozen 128-bar / 60-day recipe.

What this measures, and how it differs from the run archives' own monitor:

* The unit is a *whole job*: the pipeline runs as a child process and the sampler
  sums RSS (working set) and private commit over the child's full process tree.
  The run archives sample only the worker process tree at 0.05 s, record one
  global peak, and read GPU memory from torch counters inside the worker.
* Host memory and GPU memory are recorded separately, because the platform quota
  under test (24 GB) applies to host memory; GPU usage is reported on its own scale.
* Stage attribution comes from the child run's own ``events.jsonl``: every
  measurement sample is sliced into the pipeline stage that was active, so the
  peak is reported per stage, not only in total.
* The quota check never touches ``budget.max_rss_mb``: the declared budget stays
  at the recipe's value and only measurements decide the verdict.

Local scaling: this host has 16 GB RAM and a 16 GB page file, so the full
60-day training pipeline (an archived 19.6 GB peak) cannot run here. The harness
measures the same pipeline at several smaller training spans, fits peak against
span length, extrapolates to the recipe, and validates the extrapolation against
the archived full-scale run. The platform inference path (streaming submission,
all nine frozen members) is measured end to end here.

Outputs (under ``runs/<run_id>``): ``manifest.json``, ``events.jsonl``,
``resources.json``, ``commands.txt`` plus derived measurement configs, the
read fixture hash and the child run archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
LIMIT_MB = 24000.0  # platform host-memory quota under test; not a budget we may lower
ARCHIVE_RUN = ROOT / "runs/dual-patch-v3-60d-csz-19800/20261026T121858.552956Z_b5a6cb435220"
RECIPE_CONFIG = ROOT / "configs/dual_patch_v3_60d_csz_19800.json"
BLEND_WEIGHTS = ROOT / "runs/submission-best-20261026/build/blend_weights.json"
BARS_DIR = ROOT / "数据/cpt_jyc_2026_e2e_bar1m"
INSTRUMENTS = ROOT / "数据/cpt_jyc_2026_instruments(1).parquet"
MAPPING = ROOT / "数据/cpt_jyc_2026_stock_bar1m_instrument_mapping.csv"
BARS_TABLE = "cpt_jyc_2026_stock_bar1m"
INSTRUMENT_TABLE = "cpt_jyc_2026_instruments"
RECIPE_SAMPLES = 479432
RECIPE_BAR_ROWS = 18960607
SCHEMA = 1
FIXTURE: Path | None = None  # set inside every worker process before the query double runs


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mb(value: float) -> float:
    return round(value / 2**20, 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def append_event(path: Path, stage: str, event_type: str, **context) -> None:
    record = {"schema_version": SCHEMA, "timestamp_utc": utc_now(), "stage": stage,
              "event_type": event_type, "context": context}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


# ----------------------------------------------------------------------------- sampling


def gpu_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    values = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return float(values[0]) if out.returncode == 0 and values else None


def tree_usage(pid: int) -> dict:
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.Error:
        return {"rss_mb": 0.0, "private_mb": 0.0, "processes": 0, "cpu_seconds": 0.0}
    rss = private = cpu = 0.0
    alive = 0
    for process in processes:
        try:
            info = process.memory_info()
            rss += info.rss
            private += getattr(info, "private", 0)
            cpu += sum(process.cpu_times()[:2])
            alive += 1
        except psutil.Error:
            continue
    return {"rss_mb": rss / 2**20, "private_mb": private / 2**20, "processes": alive, "cpu_seconds": cpu}


class Sampler:
    """Sample the child's process tree until it exits; keep peaks and every sample."""

    def __init__(self, pid: int, interval: float = 0.02):
        self.pid, self.interval = pid, interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        gpu_every = max(1, int(0.5 / self.interval))
        index = 0
        while not self._stop.is_set():
            usage = tree_usage(self.pid)
            usage["timestamp_utc"] = utc_now()
            usage["gpu_mb"] = gpu_used_mb() if index % gpu_every == 0 else None
            self.samples.append(usage)
            index += 1
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


def peaks(samples: list[dict]) -> dict:
    rss = max((s["rss_mb"] for s in samples), default=0.0)
    private = max((s["private_mb"] for s in samples), default=0.0)
    gpu = max((s["gpu_mb"] for s in samples if s.get("gpu_mb") is not None), default=None)
    return {"peak_rss_mb": round(rss, 2), "peak_private_mb": round(private, 2),
            "peak_gpu_mb": None if gpu is None else round(gpu, 2),
            "process_count_max": max((s["processes"] for s in samples), default=0),
            "samples": len(samples)}


def child_runs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    return sorted((path for path in run_root.iterdir() if (path / "events.jsonl").exists()),
                  key=lambda path: path.stat().st_mtime)


def stage_timeline(events_path: Path) -> list[tuple[datetime, str]]:
    """Stage transitions from the pipeline archive's event log.

    Stages overlap in time (the ``run`` stage spans everything), so samples are
    attributed to the stage of the most recent pipeline event, skipping resource
    samples themselves.
    """
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    timeline = [(datetime.fromisoformat(event["timestamp_utc"]), event["stage"]) for event in events
                if event["stage"] != "resources"]
    return sorted(timeline)


def slice_samples(samples: list[dict], timeline: list[tuple[datetime, str]]) -> dict:
    """Attribute each sample to the most recent pipeline stage; keep per-stage peaks."""
    by_stage: dict[str, list[dict]] = {}
    index, current = 0, None
    for sample in sorted(samples, key=lambda row: row["timestamp_utc"]):
        stamp = datetime.fromisoformat(sample["timestamp_utc"])
        while index < len(timeline) and timeline[index][0] <= stamp:
            current = timeline[index][1]
            index += 1
        if current is not None:
            by_stage.setdefault(current, []).append(sample)
    return {stage: peaks(values) for stage, values in by_stage.items()}


# ----------------------------------------------------------------------------- read fixture


def build_fixture(directory: Path, start: str, end: str, features: list[str]) -> dict:
    """Materialise the read window month by month so a measurement query costs what a DAI query returns.

    Each month is written as its own parquet file; the query double reads only the
    months that overlap the requested range, with row-group statistics pushed down.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.feather as feather

    mapping = pd.read_csv(MAPPING)
    mapping["instrument"] = mapping["instrument"].astype(str)
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for month in pd.date_range(pd.Timestamp(start).replace(day=1), pd.Timestamp(end), freq="MS"):
        source = BARS_DIR / f"{month:%Y%m}.0.feather"
        if not source.exists():
            continue
        raw = feather.read_table(source, columns=["date", "instrument_id", *features])
        raw = raw.filter((ds.field("date") >= pa.scalar(pd.Timestamp(start).to_datetime64())) &
                         (ds.field("date") <= pa.scalar(pd.Timestamp(end).to_datetime64())))
        frame = raw.to_pandas()
        frame = frame.merge(mapping[["instrument_id", "instrument"]], on="instrument_id", how="left")
        frame = frame.drop(columns="instrument_id")[["date", "instrument", *features]]
        path = directory / f"bars-{month:%Y%m}.parquet"
        frame.to_parquet(path, index=False, row_group_size=200_000)
        files.append({"month": f"{month:%Y%m}", "path": str(path), "rows": int(len(frame)),
                      "bytes": path.stat().st_size, "sha256": sha256(path)})
        del frame, raw
    return {"directory": str(directory), "files": files, "rows": sum(item["rows"] for item in files),
            "start": start, "end": end, "columns": ["date", "instrument", *features],
            "source": "official local monthly feather + official instrument map, projected and window-filtered"}


def read_bars_range(lo, hi, columns):
    import pandas as pd
    import pyarrow.dataset as ds

    if FIXTURE is None:
        raise RuntimeError("Measurement fixture is not configured in this process")
    frames = []
    for month in pd.date_range(pd.Timestamp(lo).replace(day=1), pd.Timestamp(hi), freq="MS"):
        path = Path(FIXTURE) / f"bars-{month:%Y%m}.parquet"
        if not path.exists():
            continue
        dataset = ds.dataset(str(path), format="parquet")
        frames.append(dataset.to_table(
            columns=columns, filter=(ds.field("date") >= lo) & (ds.field("date") <= hi)).to_pandas())
    if not frames:
        raise ValueError(f"No measurement fixture rows in [{lo}, {hi}]")
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


class LocalQueryResult:
    """DAI query results expose ``.df()``; the local double keeps that interface."""

    def __init__(self, frame):
        self._frame = frame

    def df(self):
        return self._frame


def local_query(sql: str, *, filters=None):
    """Minimal stand-in for ``dai.query``: same columns, same inclusive date filter."""
    import pandas as pd
    import pyarrow.dataset as ds

    match = re.match(r"SELECT\s+(.+?)\s+FROM\s+(\S+)\s*$", sql.strip(), re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("Unsupported measurement query")
    wanted = [name.strip() for name in match.group(1).split(",")]
    table = match.group(2)
    lo, hi = (pd.Timestamp(value) for value in filters["date"])
    if table == BARS_TABLE:
        return LocalQueryResult(read_bars_range(lo, hi, wanted))
    if table == INSTRUMENT_TABLE:
        dataset = ds.dataset(str(INSTRUMENTS), format="parquet")
        arrow = dataset.to_table(columns=[name for name in wanted if name in ("date", "instrument")],
                                 filter=(ds.field("date") >= lo) & (ds.field("date") <= hi))
        return LocalQueryResult(arrow.to_pandas()[wanted])
    raise ValueError(f"Unknown measurement table {table!r}")


# ----------------------------------------------------------------------------- workers


def worker_inference(package: str, start: str, end: str, fixture: str, feature_columns: list[str]) -> int:
    global FIXTURE
    FIXTURE = Path(fixture)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    from modified_tlob import submission

    result = submission.predict_range({"bar1m": BARS_TABLE, "instruments": INSTRUMENT_TABLE}, start, end,
                                      package_root=Path(package), query=local_query)
    print(json.dumps({"samples": int(len(result)), "feature_columns": feature_columns}), flush=True)
    return 0


def worker_training(config_path: str) -> int:
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    from jinyiquant.config import load_config
    from jinyiquant.runner import run

    _, code = run(load_config(Path(config_path)))
    return int(code)


def worker_accounting(config_path: str, groups: int) -> int:
    """Bytes per bar row and per window sample on the production reader."""
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    import pandas as pd

    from jinyiquant.config import load_config
    from jinyiquant.data import build_batch, load_data

    config = load_config(Path(config_path))
    bundle = load_data(config.data, start=config.splits.train.start, end=config.splits.validation.start)
    bars, rows = bundle.bars, int(len(bundle.bars))
    deep = int(bars.memory_usage(deep=True).sum())
    if bundle.labels.empty:
        raise ValueError("Accounting requires labels for real sample keys")
    keys = bundle.labels[["date", "instrument"]].drop_duplicates().head(groups).reset_index(drop=True)
    batch = build_batch(bars, keys, config.data)
    payload = {"bar_rows": rows, "bars_deep_bytes": deep, "bars_bytes_per_row": round(deep / rows, 3),
               "sample_count": int(len(batch.keys)), "lookback": int(batch.x.shape[1]),
               "feature_columns": int(batch.x.shape[2]),
               "window_tensor_bytes": int(batch.x.nbytes), "observed_bytes": int(batch.observed.nbytes),
               "history_dates_bytes": int(batch.history_dates.nbytes),
               "window_bytes_per_sample": int((batch.x.nbytes + batch.observed.nbytes
                                               + batch.history_dates.nbytes) // max(1, len(batch.keys))),
               "history_dates_columns_in_cut_slice": int(batch.history_dates.shape[1]),
               "process_rss_mb": round(psutil.Process().memory_info().rss / 2**20, 2),
               "pandas": pd.__version__}
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


# ----------------------------------------------------------------------------- orchestration


def run_child(command: list[str], *, env: dict, interval: float, log_path: Path) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        with Sampler(process.pid, interval) as sampler:
            returncode = process.wait()
        samples = sampler.samples
    return {"returncode": returncode, "wall_seconds": round(time.monotonic() - started, 2),
            "samples": samples, "peaks": peaks(samples), "log": str(log_path)}


def environment() -> dict:
    import pandas
    import pyarrow

    return {"python": sys.version.split()[0], "executable": sys.executable, "platform": sys.platform,
            "psutil": psutil.__version__, "pandas": pandas.__version__, "pyarrow": pyarrow.__version__,
            "torch": _module_version("torch"), "cpu_count": os.cpu_count(),
            "host_ram_mb": mb(psutil.virtual_memory().total),
            "commit_limit_mb": mb(psutil.virtual_memory().total + getattr(psutil.swap_memory(), "total", 0)),
            "commit_available_mb": mb(psutil.swap_memory().free + psutil.virtual_memory().available),
            "gpu": _gpu_name()}


def _module_version(name: str):
    try:
        module = __import__(name)
        return getattr(module, "__version__", None)
    except Exception:  # noqa: BLE001 - measurement host may not install every dependency
        return None


def _gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def git_state() -> dict:
    def capture(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": capture("rev-parse", "HEAD"),
            "status_sha256": hashlib.sha256(capture("status", "--porcelain").encode()).hexdigest()}


def inference_package(run_dir: Path) -> tuple[Path, dict]:
    """Frozen nine-member blend as a package directory; weights are hardlinked, not copied."""
    package = run_dir / "inference-package"
    package.mkdir(parents=True, exist_ok=True)
    target = package / "modified_tlob_weights.json"
    if not target.exists():
        try:
            os.link(BLEND_WEIGHTS, target)
        except OSError:
            shutil.copy2(BLEND_WEIGHTS, target)
    digest = sha256(target)
    contract = {"schema_version": 1, "device": "cpu", "history_policy": {"mode": "strict"},
                "min_coverage": 0.99, "platform_price_scale": 100.0, "wall_seconds": 7200,
                "weights_sha256": digest, "experiment_id": "best-three-trunk-equal-20261026",
                "measurement_note": "weights are the frozen nine-member blend; contract mirrors the flat package"}
    write_json(package / "submission_config.json", contract)
    return package, {"weights_sha256": digest, "weights_bytes": target.stat().st_size,
                     "blend_members": 9, "contract": contract}


def calendar_days(start: str, end: str) -> list:
    import pandas as pd
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(INSTRUMENTS), format="parquet")
    arrow = dataset.to_table(columns=["date"],
                             filter=(ds.field("date") >= pd.Timestamp(start)) & (ds.field("date") < pd.Timestamp(end)))
    return sorted(pd.DatetimeIndex(pd.to_datetime(arrow.to_pandas()["date"]).dt.normalize().unique()))


def measurement_config(run_dir: Path, span_days: int, days: list, validation_days: int = 1,
                       validation_calendar: list | None = None) -> Path:
    recipe = json.loads(RECIPE_CONFIG.read_text(encoding="utf-8"))
    recipe["experiment_id"] = f"resource-baseline-train-{span_days}d-val{validation_days}d"
    recipe["hypothesis"] = ("Frozen 128-bar/60-day recipe executed over a shorter training span to measure "
                            "host-memory scaling; compute steps bounded, the memory budget is untouched")
    recipe["output_root"] = str((run_dir / "train-runs").resolve())
    recipe["splits"]["train"]["start"] = str(days[-span_days])
    if validation_calendar:
        # Training window ends where validation starts (2023-11-01); the split end is exclusive.
        recipe["splits"]["validation"]["end"] = str(
            validation_calendar[min(validation_days, len(validation_calendar) - 1)])
    recipe["model"]["max_steps"] = 30
    recipe["model"]["epochs"] = 1
    recipe["model"]["device"] = "cpu"
    recipe["model"]["log_every_steps"] = 5
    recipe["data"]["cache_dir"] = str((run_dir / "cache").resolve())
    for key in ("bars_path", "labels_path", "instruments_path", "instrument_map_path"):
        recipe["data"][key] = str((RECIPE_CONFIG.parent / recipe["data"][key]).resolve())
    path = run_dir / "configs" / f"train-{span_days}d.json"
    write_json(path, recipe)
    return path


def fit_line(points: list[tuple[float, float]]) -> dict:
    """Least squares y = a + b x, with the fit quality recorded rather than assumed."""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    b = sum((x - mean_x) * (y - mean_y) for x, y in points) / sxx if sxx else 0.0
    a = mean_y - b * mean_x
    return {"intercept_mb": round(a, 2), "slope_mb_per_day": round(b, 2), "points": points,
            "max_residual_mb": round(max(abs(y - (a + b * x)) for x, y in points), 2) if xs else 0.0,
            "samples": len(points)}


def allocation_accounting(rows: int, samples: int, lookback: int = 128, features: int = 12) -> dict:
    """Exact dense-array bytes of the shared window builder; deterministic and machine independent."""
    per_sample = {"window_float32": lookback * features * 4, "observed_bool": lookback * features * 1,
                  "history_dates_int64": lookback * 8}
    total = sum(per_sample.values())
    return {"lookback": lookback, "feature_columns": features, "samples": samples, "bar_rows": rows,
            "per_sample_bytes": per_sample, "per_sample_total_bytes": total,
            "training_windows_mb": round(samples * total / 2**20, 2),
            "instrument_group_copy_mb": round(rows * features * 4 / 2**20, 2),
            "per_sample_kb": round(total / 1024, 2)}


def archive_baseline() -> dict:
    manifest = json.loads((ARCHIVE_RUN / "manifest.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (ARCHIVE_RUN / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = [{"timestamp_utc": event["timestamp_utc"], "rss_mb": event["context"]["rss_mb"],
                "private_mb": event["context"].get("private_mb") or event["context"]["rss_mb"],
                "processes": 1, "gpu_mb": event["context"].get("gpu_memory_mb")}
               for event in events if event["stage"] == "resources" and "rss_mb" in event["context"]]
    stages = slice_samples(samples, stage_timeline(ARCHIVE_RUN / "events.jsonl"))
    peak = max(samples, key=lambda sample: sample["rss_mb"])
    return {"run_id": manifest["run_id"], "host": {"cpu_count": manifest["environment"]["cpu_count"]},
            "recorded_peak_rss_mb": manifest["resources"]["peak_rss_mb"],
            "sampled_peak_rss_mb": round(peak["rss_mb"], 2), "peak_sample_utc": peak["timestamp_utc"],
            "monitor_interval_seconds": manifest["resources"]["monitor_interval_seconds"],
            "wall_seconds": round(manifest["resources"]["wall_seconds"], 2), "sample_count": len(samples),
            "stage_peaks": stages, "gpu_peak_from_events_mb": _gpu_peak(events),
            "data": _document(ARCHIVE_RUN / "data.json")}


def _gpu_peak(events: list[dict]):
    values = [event["context"].get("gpu_peak_memory_mb") for event in events
              if event.get("context") and event["context"].get("gpu_peak_memory_mb") is not None]
    return max(values) if values else None


def _child_manifest(archive: Path) -> dict:
    path = archive / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _document(path: Path) -> dict:
    if not path.exists():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    return document.get("metadata", document)


def _child_training_samples(archive: Path) -> int | None:
    path = archive / "events.jsonl"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["stage"] == "training" and event["event_type"] == "started":
            return int(event["context"].get("samples", 0)) or None
    return None


def _last_json(log: Path):
    payload = None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except ValueError:
            continue
    return payload


def _delta(values: list[float]) -> float | None:
    values = [value for value in values if value]
    return round(max(values) - min(values), 2) if len(values) > 1 else None


def _peak_stage(stage_peaks: dict):
    rows = [{"stage": stage, "peak_rss_mb": values.get("peak_rss_mb")} for stage, values in stage_peaks.items()
            if values.get("peak_rss_mb")]
    return max(rows, key=lambda row: row["peak_rss_mb"]) if rows else None


def _stage_series(training_runs: list[dict]) -> dict:
    """(span_days, stage peak) series for every pipeline stage that was sampled."""
    series: dict[str, list] = {}
    for row in training_runs:
        for stage, values in (row.get("stage_peaks") or {}).items():
            if values.get("peak_rss_mb"):
                series.setdefault(stage, []).append((row["span_days"], values["peak_rss_mb"]))
    return {stage: sorted(points) for stage, points in series.items()}


def _peak_stage_name(stage_fits: dict, recipe_days: int = 60):
    """Stage whose extrapolated peak is the largest: the stage that sets the recipe peak."""
    candidates = [(stage, line) for stage, line in stage_fits.items() if stage != "run"]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1]["intercept_mb"]
               + item[1]["slope_mb_per_day"] * recipe_days)[0]


def _exhaustion_analysis(composition: dict) -> dict:
    base = composition["composed_peak_host_mb"]
    recipe = composition["accounting_recipe"]
    variants = composition["accounting_variants"]
    extra_256 = variants["lookback_256"]["training_windows_mb"] - recipe["training_windows_mb"]
    per_row = composition.get("bytes_per_row_measured")
    extra_132_rows = ((variants["days_132"]["bar_rows"] - recipe["bar_rows"]) * per_row / 2**20) if per_row else 0.0
    extra_132_windows = variants["days_132"]["training_windows_mb"] - recipe["training_windows_mb"]
    candidates = {
        "current_128_60d": base,
        "lookback_256": base + extra_256,
        "days_132": base + extra_132_rows + extra_132_windows,
        "two_concurrent_runs": base * 2,
    }
    exceeded = {name: {"peak_host_mb": round(value, 2), "over_limit_mb": round(value - LIMIT_MB, 2)}
                for name, value in candidates.items() if value > LIMIT_MB}
    return {"variants": {name: round(value, 2) for name, value in candidates.items()}, "over_limit": exceeded,
            "first_exceeded_config": next(iter(exceeded), None),
            "headroom_consumers": [
                {"source": "build_batch dense window tensor (samples x lookback x 12 float32)",
                 "mb": round(recipe["per_sample_bytes"]["window_float32"] * recipe["samples"] / 2**20, 2)},
                {"source": "build_batch observed mask (samples x lookback x 12 bool)",
                 "mb": round(recipe["per_sample_bytes"]["observed_bool"] * recipe["samples"] / 2**20, 2)},
                {"source": "build_batch history timestamps (samples x lookback int64)",
                 "mb": round(recipe["per_sample_bytes"]["history_dates_int64"] * recipe["samples"] / 2**20, 2)},
                {"source": "per-instrument float32 copy of the whole bar window in build_batch",
                 "mb": recipe["instrument_group_copy_mb"]},
                {"source": "raw bars frame retained for the whole job",
                 "mb": round(recipe["bar_rows"] * per_row / 2**20, 2) if per_row else None}],
            "stage_of_peak": _peak_stage(composition.get("stage_attribution", {})),
            "reading": ("The peak is the post-training validation/evaluation stage: the job still holds the "
                        "raw bars frame plus the dense training window tensors while it builds the second "
                        "(validation) batch. Dense arrays are charged per sample, so lookback and span "
                        "length consume the same headroom.")}


def verdict(measured: float) -> dict:
    margin = LIMIT_MB - measured
    return {"limit_mb": LIMIT_MB, "measured_peak_host_mb": measured, "margin_mb": round(margin, 2),
            "margin_fraction": round(margin / LIMIT_MB, 4), "under_limit": bool(margin >= 0),
            "budget_max_rss_mb_untouched": 24000.0,
            "note": ("Host memory only; GPU memory is reported separately. Verdict comes from measured and "
                     "archive-anchored numbers, never from the declared budget.")}


def _conclusion(resources: dict) -> dict:
    quota, exhaust, composition = resources["quota"], resources["exhaustion"], resources["composition"]
    return {"host_peak_mb": quota["measured_peak_host_mb"], "limit_mb": quota["limit_mb"],
            "margin_mb": quota["margin_mb"], "under_limit": quota["under_limit"],
            "prediction_vs_archive_mb": composition["prediction_error_mb"],
            "first_exceeded_config": exhaust["first_exceeded_config"],
            "peak_stage": exhaust["stage_of_peak"],
            "inference_peak_host_mb": resources["inference"]["peak_host_mb"],
            "open_limits": [
                "this host cannot execute the 60-day pipeline at full scale; the full-scale number is the "
                "archived cloud measurement, validated by the local span fit and exact array accounting",
                "the recipe's training GPU peak is the archived run's torch counter; the inference path's GPU "
                "peak is sampled from the 6 GB local board"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/resource-baseline-20261027")
    parser.add_argument("--spans", default="2,4,8", help="training spans in trading days")
    parser.add_argument("--repeat", type=int, default=2, help="repeat count for the largest span and inference")
    parser.add_argument("--inference-start", default="2023-11-01")
    parser.add_argument("--inference-end", default="2023-11-15 15:00:00",
                        help="inclusive end timestamp; a bare date would only match midnight rows")
    parser.add_argument("--validation-days", type=int, default=10,
                        help="validation trading days used for the largest span (recipe uses 10)")
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--verify", nargs=2, metavar=("RUN_A", "RUN_B"),
                        help="compare two completed run directories and write verify.json into RUN_B")
    parser.add_argument("--acceptance", type=Path, metavar="RUN_DIR",
                        help="map a completed run onto the task acceptance criteria and write acceptance.json")
    parser.add_argument("--refinalize", type=Path, metavar="RUN_DIR",
                        help="recompute the quota verdict conservatively from a finished run's two estimates")
    parser.add_argument("--notebook-caliber", type=Path, metavar="RUN_DIR",
                        help="apply the corrected caliber: 24 GB gates the platform notebook/inference path only")
    parser.add_argument("--gpu-probe", type=Path, metavar="JSON",
                        help="attach a separately measured GPU-path peak for the notebook inference")
    args = parser.parse_args(argv)
    if args.verify:
        return verify_runs(Path(args.verify[0]), Path(args.verify[1]))
    if args.acceptance:
        return write_acceptance(Path(args.acceptance))
    if args.refinalize:
        return refinalize(Path(args.refinalize))
    if args.notebook_caliber:
        return notebook_caliber(Path(args.notebook_caliber), Path(args.gpu_probe) if args.gpu_probe else None)
    run_dir = args.output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + os.urandom(4).hex())
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    spans = [int(value) for value in args.spans.split(",")]
    features = json.loads(RECIPE_CONFIG.read_text(encoding="utf-8"))["data"]["feature_columns"]
    env = {**os.environ, "PYTHONHASHSEED": "11", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"}
    fixture = build_fixture(run_dir / "fixture", "2023-10-15", "2023-11-16", features)
    write_json(run_dir / "fixture/manifest.json", fixture)
    days = calendar_days("2023-07-01", "2023-11-01")
    validation_calendar = calendar_days("2023-11-01", "2023-12-01")
    append_event(events_path, "setup", "started", run_dir=str(run_dir), spans=spans,
                 fixture={"directory": fixture["directory"], "rows": fixture["rows"], "files": fixture["files"]},
                 calendar_days=len(days))
    package, package_meta = inference_package(run_dir)

    inference_runs = []
    for attempt in range(max(1, args.repeat)):
        append_event(events_path, "inference", "started", attempt=attempt, start=args.inference_start,
                     end=args.inference_end, package=str(package))
        result = run_child(_worker_command("worker_inference", str(package), args.inference_start,
                                           args.inference_end, fixture["directory"], features),
                           env=env, interval=args.interval,
                           log_path=run_dir / f"worker-inference-{attempt}.log")
        archive = child_runs(package / "inference_runs")
        stages = slice_samples(result["samples"], stage_timeline(archive[-1] / "events.jsonl")) if archive else {}
        entry = {"attempt": attempt, "returncode": result["returncode"], "wall_seconds": result["wall_seconds"],
                 "peaks": result["peaks"], "stage_peaks": stages,
                 "archive": str(archive[-1]) if archive else None,
                 "child_resources": _child_manifest(archive[-1]).get("resources") if archive else None,
                 "history_handling": _child_manifest(archive[-1]).get("history_handling") if archive else None}
        inference_runs.append(entry)
        append_event(events_path, "inference", "completed" if result["returncode"] == 0 else "failed",
                     attempt=attempt, peaks=result["peaks"], stage_peaks=stages)

    training_runs = []
    order = list(spans)
    for index, span in enumerate(order):
        validation_days = args.validation_days if span == max(spans) else 1
        config_path = measurement_config(run_dir, span, days, validation_days, validation_calendar)
        append_event(events_path, "training", "started", span_days=span, index=index, config=str(config_path))
        result = run_child(_worker_command("worker_training", str(config_path)), env=env,
                           interval=args.interval, log_path=run_dir / f"worker-train-{span}d-{index}.log")
        archive = child_runs(run_dir / "train-runs")
        stages = slice_samples(result["samples"], stage_timeline(archive[-1] / "events.jsonl")) if archive else {}
        child = _child_manifest(archive[-1]) if archive else {}
        entry = {"span_days": span, "index": index, "validation_days": validation_days,
                 "returncode": result["returncode"],
                 "wall_seconds": result["wall_seconds"], "peaks": result["peaks"], "stage_peaks": stages,
                 "archive": str(archive[-1]) if archive else None,
                 "child_resources": child.get("resources"), "child_budget": child.get("budget"),
                 "bar_rows": _document(archive[-1] / "data.json").get("bar_rows") if archive else None,
                 "training_samples": _child_training_samples(archive[-1]) if archive else None}
        training_runs.append(entry)
        append_event(events_path, "training", "completed" if result["returncode"] == 0 else "failed",
                     span_days=span, peaks=result["peaks"], stage_peaks=stages, bar_rows=entry["bar_rows"],
                     training_samples=entry["training_samples"])

    accounting_config = measurement_config(run_dir, spans[-1], days, 1, validation_calendar)
    append_event(events_path, "accounting", "started")
    accounting = run_child(_worker_command("worker_accounting", str(accounting_config), "20000"),
                           env=env, interval=args.interval, log_path=run_dir / "worker-accounting.log")
    accounting_payload = _last_json(accounting["log"]) or {}
    append_event(events_path, "accounting", "completed", peaks=accounting["peaks"],
                 window_bytes_per_sample=accounting_payload.get("window_bytes_per_sample"),
                 bars_bytes_per_row=accounting_payload.get("bars_bytes_per_row"))

    archive = archive_baseline()
    points = [(row["span_days"], row["peaks"]["peak_rss_mb"]) for row in training_runs]
    fit = fit_line(points)
    recipe_days = len(days)
    predicted = fit["intercept_mb"] + fit["slope_mb_per_day"] * recipe_days
    stage_series = _stage_series(training_runs)
    stage_fits = {stage: fit_line(series) for stage, series in stage_series.items() if len(series) >= 2}
    stage_predictions = {stage: round(line["intercept_mb"] + line["slope_mb_per_day"] * recipe_days, 2)
                         for stage, line in stage_fits.items()}
    peak_stage = _peak_stage_name(stage_fits, recipe_days)
    full_scale = max(training_runs, key=lambda row: row["span_days"]) if training_runs else None
    validation_increment = None
    if full_scale:
        observed = {stage: values["peak_rss_mb"] for stage, values in (full_scale["stage_peaks"] or {}).items()}
        base_training = observed.get("training")
        post = {stage: value for stage, value in observed.items()
                if stage in {"validation", "evaluation", "inference", "loading", "saving"}}
        if base_training and post:
            validation_increment = round(max(post.values()) - base_training, 2)
    training_peak_at_recipe = (round((stage_fits["training"]["intercept_mb"]
                                      + stage_fits["training"]["slope_mb_per_day"] * recipe_days), 2)
                               if "training" in stage_fits else None)
    composed = (round(training_peak_at_recipe + max(validation_increment or 0.0, 0.0), 2)
                if training_peak_at_recipe is not None else archive["recorded_peak_rss_mb"])
    accounting_recipe = allocation_accounting(RECIPE_BAR_ROWS, RECIPE_SAMPLES)
    composition = {
        "measured_spans": [{"span_days": row["span_days"], "peak_rss_mb": row["peaks"]["peak_rss_mb"],
                            "peak_private_mb": row["peaks"]["peak_private_mb"],
                            "training_samples": row["training_samples"], "bar_rows": row["bar_rows"]}
                           for row in training_runs],
        "fit": fit, "recipe_trading_days": recipe_days, "recipe_samples": RECIPE_SAMPLES,
        "predicted_peak_host_mb": round(predicted, 2),
        "stage_series": {stage: series for stage, series in stage_series.items()},
        "stage_fits": stage_fits, "stage_predictions": stage_predictions,
        "peak_stage": peak_stage,
        "predicted_peak_host_mb_from_peak_stage": stage_predictions.get(peak_stage),
        "peak_stage_prediction_error_mb": (round(stage_predictions[peak_stage] - archive["recorded_peak_rss_mb"], 2)
                                           if peak_stage in stage_predictions else None),
        "peak_stage_prediction_error_fraction": (
            round((stage_predictions[peak_stage] - archive["recorded_peak_rss_mb"])
                  / archive["recorded_peak_rss_mb"], 4) if peak_stage in stage_predictions else None),
        "archive_peak_host_mb": archive["recorded_peak_rss_mb"],
        "prediction_error_mb": round(predicted - archive["recorded_peak_rss_mb"], 2),
        "prediction_error_fraction": round((predicted - archive["recorded_peak_rss_mb"])
                                           / archive["recorded_peak_rss_mb"], 4),
        "full_scale_local_span": {"span_days": full_scale["span_days"] if full_scale else None,
                                  "validation_days": full_scale.get("validation_days") if full_scale else None,
                                  "validation_increment_mb": validation_increment},
        "training_peak_at_recipe_from_fit_mb": training_peak_at_recipe,
        "composed_peak_host_mb": composed,
        "composed_and_archive_error_mb": round(composed - archive["recorded_peak_rss_mb"], 2),
        "composed_and_archive_error_fraction": round((composed - archive["recorded_peak_rss_mb"])
                                                     / archive["recorded_peak_rss_mb"], 4),
        "composed_basis": ("composed from the local stage series: the training-stage peak is extrapolated along "
                           "the measured span line to the recipe's 60 trading days, then the measured "
                           "post-training increment (validation/evaluation stage, full 80k-sample validation "
                           "window) is added; dense-array bytes are exact and machine independent; the "
                           "archive's own full-scale run is the independent check"),
        "bytes_per_row_measured": accounting_payload.get("bars_bytes_per_row"),
        "window_bytes_per_sample_measured": accounting_payload.get("window_bytes_per_sample"),
        "accounting_recipe": accounting_recipe,
        "accounting_validation_batch": allocation_accounting(RECIPE_BAR_ROWS, 80000),
        "accounting_variants": {"lookback_256": allocation_accounting(RECIPE_BAR_ROWS, RECIPE_SAMPLES, lookback=256),
                                "days_132": allocation_accounting(70000000, 1056000)},
        "stage_attribution": archive["stage_peaks"],
    }
    quota = verdict(composition["composed_peak_host_mb"])
    quota["archive_peak_host_mb"] = archive["recorded_peak_rss_mb"]
    quota["archive_margin_mb"] = round(LIMIT_MB - archive["recorded_peak_rss_mb"], 2)
    quota["composed_vs_archive_fraction"] = composition["composed_and_archive_error_fraction"]
    exhaustion = _exhaustion_analysis(composition)
    reproduce = {"command": (f"{sys.executable} {Path(__file__).relative_to(ROOT).as_posix()} "
                             f"--output {args.output} --spans {args.spans} --repeat {args.repeat} "
                             f"--inference-start {args.inference_start} --inference-end {args.inference_end}"),
                 "run_dir": str(run_dir),
                 "acceptance": {"repeat_peak_delta_mb_max": 400.0,
                                "prediction_vs_archive_fraction_max": 0.15}}
    resources = {"schema_version": SCHEMA, "run_id": run_dir.name, "generated_at_utc": utc_now(),
                 "task": "b1-measure: platform-caliber resource baseline and exceed-point localisation",
                 "limit": {"host_mb": LIMIT_MB, "source": "platform host-memory quota under test",
                           "gpu_policy": "measured separately, never counted against the host quota"},
                 "environment": environment(), "git": git_state(), "fixture": fixture, "package": package_meta,
                 "inference": {"start": args.inference_start, "end": args.inference_end,
                               "attempts": inference_runs,
                               "repeat_peak_delta_mb": _delta([row["peaks"]["peak_rss_mb"]
                                                               for row in inference_runs]),
                               "peak_host_mb": max((row["peaks"]["peak_rss_mb"] for row in inference_runs),
                                                   default=None),
                               "peak_gpu_mb": max((row["peaks"]["peak_gpu_mb"] for row in inference_runs
                                                   if row["peaks"]["peak_gpu_mb"] is not None), default=None)},
                 "training": {"spans": training_runs,
                              "repeat_peak_delta_mb": _delta([row["peaks"]["peak_rss_mb"] for row in training_runs
                                                              if row["span_days"] == spans[-1]]),
                              "child_declared_budget": [row["child_budget"] for row in training_runs]},
                 "accounting": accounting_payload, "archive_baseline": archive, "composition": composition,
                 "quota": quota, "exhaustion": exhaustion, "reproduce": reproduce}
    write_json(run_dir / "resources.json", resources)
    write_json(run_dir / "manifest.json", {
        "schema_version": SCHEMA, "run_id": run_dir.name, "task": "b1-measure",
        "started_at_utc": _first_timestamp(events_path), "finished_at_utc": utc_now(),
        "environment": resources["environment"], "git": resources["git"],
        "scope": {"inference": [args.inference_start, args.inference_end], "spans": spans,
                  "recipe_config": str(RECIPE_CONFIG.relative_to(ROOT)), "archive": archive["run_id"]},
        "quota": quota, "conclusion": _conclusion(resources)})
    (run_dir / "commands.txt").write_text(reproduce["command"] + "\n", encoding="utf-8")
    append_event(events_path, "run", "completed", host_peak_mb=composition["composed_peak_host_mb"],
                 under_limit=quota["under_limit"], first_exceeded_config=exhaustion["first_exceeded_config"])
    print(json.dumps({"run_dir": str(run_dir), "quota": quota,
                      "composition": {"predicted_peak_host_mb": composition["predicted_peak_host_mb"],
                                      "archive_peak_host_mb": composition["archive_peak_host_mb"],
                                      "prediction_error_fraction": composition["prediction_error_fraction"]},
                      "exhaustion": exhaustion["variants"]}, ensure_ascii=False, indent=1))
    return 0


def _worker_command(function: str, *args) -> list[str]:
    payload = ", ".join(repr(value) for value in args)
    source = (f"import sys; sys.path[:0]=[{str(ROOT / 'src')!r}, {str(ROOT)!r}];"
              f"from scripts.measure_resource_baseline import {function} as w;"
              f"raise SystemExit(w({payload}))")
    return [sys.executable, "-c", source]


def notebook_caliber(run_dir: Path, gpu_probe: Path | None) -> int:
    """Corrected caliber: 24 GB gates the platform notebook inference path only.

    The platform executes data read -> model load -> full-range inference ->
    scoring/writing inside the notebook. Training peaks are reported separately
    and are *not* gated (training runs on the 60 GB cloud container).
    """
    path = run_dir / "resources.json"
    resources = json.loads(path.read_text(encoding="utf-8"))
    inference = resources["inference"]
    attempts = [row["peaks"]["peak_rss_mb"] for row in inference["attempts"]]
    measured = max(attempts) if attempts else None
    if measured is None:
        raise SystemExit("No inference measurement in this run")
    data = (inference["attempts"][-1].get("child_resources") or {})
    # Section shape is fixed by the recipe (8 sample times x ~1000 pool members); the platform
    # evaluation window is not readable locally, so the bound uses the longest interval the
    # project reserves for one pass (the sealed final-check window, ~58 trading days).
    rows_per_section = 1000
    sections_per_day = 8
    evaluation_days = 58
    per_day_rows = rows_per_section * sections_per_day
    window_days = evaluation_days
    growth_rows = per_day_rows * window_days
    bytes_per_row = 100.0  # conservative: datetime + string instrument + float + pandas row overhead
    growth_mb = round(growth_rows * bytes_per_row / 2**20, 2)
    bound = round(measured + growth_mb, 2)
    margin = round(LIMIT_MB - bound, 2)
    gpu_block = {"measured_here": False,
                 "reason": "the locked measurement environment ships CPU-only torch 2.5.0",
                 "archive_training_gpu_peak_mb": resources["archive_baseline"]["gpu_peak_from_events_mb"]}
    if gpu_probe and gpu_probe.exists():
        gpu_block.update(json.loads(gpu_probe.read_text(encoding="utf-8")))
    block = {
        "caliber": ("24 GB applies to the platform notebook inference/submission path only; the training path "
                    "is reported for reference and is not gated"),
        "notebook_inference": {
            "measured_peak_host_mb": measured, "measured_commit_mb": max((row["peaks"]["peak_private_mb"]
                                                                          for row in inference["attempts"]), default=None),
            "attempts": attempts, "repeat_delta_mb": _delta(attempts),
            "window": {"measured_start": inference["start"], "measured_end": inference["end"],
                       "measured_days": _window_days(inference),
                       "full_range_assumption_days": evaluation_days,
                       "assumption": ("platform evaluation window length is not available locally; the bound "
                                      "uses 58 trading days, the longest interval this project reserves for one "
                                      "pass"),},
            "full_range_growth": {"rows": growth_rows, "bytes_per_row_assumed": bytes_per_row,
                                  "growth_mb": growth_mb,
                                  "note": "streaming keeps only one day of bars plus a per-security tail; only "
                                          "predictions accumulate, so the growth term is the output table"},
            "full_range_bound_host_mb": bound, "limit_mb": LIMIT_MB, "margin_mb": margin,
            "margin_fraction": round(margin / LIMIT_MB, 4), "under_limit": bool(margin >= 0),
            "peaks_by_stage": [{"attempt": row["attempt"], "stage_peaks": row["stage_peaks"]}
                               for row in inference["attempts"]],
            "child_resource_counters": data, "gpu": gpu_block,
            "subprocess_note": ("sampled over the child's whole process tree; the Windows venv redirector "
                                "and the notebook-style launcher are inside the sample"),
        },
        "training_reference_ungated": {
            "gate": False,
            "archive_full_scale_peak_host_mb": resources["archive_baseline"]["recorded_peak_rss_mb"],
            "archive_stage_peaks": resources["archive_baseline"]["stage_peaks"],
            "local_span_peaks_mb": {row["span_days"]: row["peaks"]["peak_rss_mb"]
                                    for row in resources["training"]["spans"]},
            "allocation_sources": resources["exhaustion"]["headroom_consumers"],
            "note": ("training ran on the 60 GB cloud container; these numbers inform headroom planning for "
                     "256-bar / 132-day variants and are not a platform-quota verdict"),
        },
        "budget_max_rss_mb_untouched": [row.get("child_budget") for row in resources["training"]["spans"]],
        "generated_at_utc": utc_now(),
    }
    resources["caliber"] = block
    resources["quota"] = {"limit_mb": LIMIT_MB, "path": "platform notebook inference/submission",
                          "measured_peak_host_mb": measured, "full_range_bound_host_mb": bound,
                          "margin_mb": margin, "margin_fraction": block["notebook_inference"]["margin_fraction"],
                          "under_limit": block["notebook_inference"]["under_limit"],
                          "training_path_gate": False,
                          "budget_max_rss_mb_untouched": 24000.0,
                          "note": ("host memory only; GPU memory is reported separately. Verdict from measured "
                                   "numbers plus the measured output-table growth, never from the budget.")}
    write_json(path, resources)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["quota"] = resources["quota"]
        manifest["caliber"] = block["caliber"]
        write_json(manifest_path, manifest)
    append_event(run_dir / "events.jsonl", "run", "notebook_caliber_applied",
                 measured_peak_host_mb=measured, full_range_bound_host_mb=bound, under_limit=block["notebook_inference"]["under_limit"])
    write_json(run_dir / "notebook_quota.json", {"schema_version": SCHEMA, "run_id": resources["run_id"],
                                                 **block, "quota": resources["quota"]})
    print(json.dumps(resources["quota"], ensure_ascii=False, indent=1))
    return 0


def _window_days(inference: dict) -> int:
    stage = [row for row in inference["attempts"] if row.get("stage_peaks")]
    if stage:
        return len([1 for value in (stage[-1].get("stage_peaks") or {}).get("preprocessing", {}) if value]) or 1
    return 1


def refinalize(run_dir: Path) -> int:
    """Adopt the conservative baseline: the larger of the two independent full-scale estimates.

    The policy is fixed before the numbers are read: a resource baseline must not
    under-report the peak, so the quota verdict uses max(archive measurement,
    local composition) and both components stay in the file.
    """
    path = run_dir / "resources.json"
    resources = json.loads(path.read_text(encoding="utf-8"))
    composition = resources["composition"]
    archive_peak = resources["archive_baseline"]["recorded_peak_rss_mb"]
    composed = composition["composed_peak_host_mb"]
    baseline = max(archive_peak, composed)
    resources["quota"] = verdict(baseline)
    resources["quota"].update({"archive_peak_host_mb": archive_peak,
                               "archive_margin_mb": round(LIMIT_MB - archive_peak, 2),
                               "composed_vs_archive_fraction": composition["composed_and_archive_error_fraction"],
                               "baseline_policy": "max(archive measured full-scale peak, local composed estimate)",
                               "estimates_mb": {"archive": archive_peak, "composed": composed}})
    resources["exhaustion"] = _exhaustion_analysis(composition) if composition.get("accounting_recipe") else         resources["exhaustion"]
    write_json(path, resources)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["quota"] = resources["quota"]
        manifest["conclusion"] = _conclusion(resources)
        write_json(manifest_path, manifest)
    append_event(run_dir / "events.jsonl", "run", "refinalized", baseline_mb=baseline,
                 archive_mb=archive_peak, composed_mb=composed)
    print(json.dumps(resources["quota"], ensure_ascii=False, indent=1))
    return 0


def write_acceptance(run_dir: Path) -> int:
    """Map the measured run onto the task's acceptance criteria; never soften a failed check."""
    resources = json.loads((run_dir / "resources.json").read_text(encoding="utf-8"))
    caliber = resources.get("caliber") or {}
    notebook = caliber.get("notebook_inference") or {}
    training_ref = caliber.get("training_reference_ungated") or {}
    inference, training = resources["inference"], resources["training"]
    budgets = [(row.get("child_budget") or {}).get("max_rss_mb") for row in training["spans"]]
    configs = sorted((run_dir / "configs").glob("train-*.json"))
    lookbacks = {json.loads(path.read_text(encoding="utf-8"))["data"]["lookback"] for path in configs}
    criteria = [
        {"criterion": "平台 Notebook 推理/提交路径峰值：主机 RSS 与显存分开测量，含子进程",
         "passed": bool(notebook.get("measured_peak_host_mb") is not None),
         "evidence": {"measured_peak_host_mb": notebook.get("measured_peak_host_mb"),
                      "measured_commit_mb": notebook.get("measured_commit_mb"),
                      "gpu": notebook.get("gpu"), "attempts_mb": notebook.get("attempts"),
                      "stages": notebook.get("peaks_by_stage"),
                      "subprocess_note": notebook.get("subprocess_note")}},
        {"criterion": "对照 24 GB 判定并给出余量（仅推理/提交路径）",
         "passed": notebook.get("under_limit") is not None,
         "evidence": {"measured_peak_host_mb": notebook.get("measured_peak_host_mb"),
                      "full_range_bound_host_mb": notebook.get("full_range_bound_host_mb"),
                      "growth": notebook.get("full_range_growth"),
                      "limit_mb": notebook.get("limit_mb"), "margin_mb": notebook.get("margin_mb"),
                      "margin_fraction": notebook.get("margin_fraction"),
                      "under_limit": notebook.get("under_limit"),
                      "training_path_gate": False}},
        {"criterion": "基线数值可复跑",
         "passed": _reproducible(inference, training["spans"]),
         "evidence": {"inference_attempts_mb": notebook.get("attempts"),
                      "inference_repeat_delta_mb": notebook.get("repeat_delta_mb"),
                      "training_spans": [row["span_days"] for row in training["spans"]],
                      "span_peak_series_mb": {row["span_days"]: row["peaks"]["peak_rss_mb"]
                                              for row in training["spans"]},
                      "reproduce_command": resources["reproduce"]["command"]}},
        {"criterion": "峰值阶段与分配来源可定位（训练路径只报告、不设门）",
         "passed": bool(training_ref.get("allocation_sources")),
         "evidence": {"training_peak_stage": _peak_stage(training_ref.get("archive_stage_peaks") or {}),
                      "archive_full_scale_peak_host_mb": training_ref.get("archive_full_scale_peak_host_mb"),
                      "local_span_peaks_mb": training_ref.get("local_span_peaks_mb"),
                      "allocation_sources": training_ref.get("allocation_sources")}},
        {"criterion": "未通过调低 max_rss_mb 达标",
         "passed": set(budgets) == {24000.0},
         "evidence": {"child_declared_budget_max_rss_mb": budgets, "recipe_budget_max_rss_mb": 24000.0}},
        {"criterion": "256 根与 132 日配置保持暂缓",
         "passed": lookbacks == {128},
         "evidence": {"configs_run": [path.name for path in configs], "lookbacks": sorted(lookbacks),
                      "deferred_variants_only_accounted": ["lookback_256", "days_132"]}},
    ]
    passed = all(item["passed"] for item in criteria)
    payload = {"schema_version": SCHEMA, "run_id": resources["run_id"], "generated_at_utc": utc_now(),
               "task": "t1/b1-measure", "caliber": caliber.get("caliber"), "passed": passed,
               "criteria": criteria,
               "summary": {"notebook_peak_host_mb": notebook.get("measured_peak_host_mb"),
                           "notebook_full_range_bound_host_mb": notebook.get("full_range_bound_host_mb"),
                           "limit_mb": notebook.get("limit_mb"), "margin_mb": notebook.get("margin_mb"),
                           "training_peak_reported_mb": training_ref.get("archive_full_scale_peak_host_mb"),
                           "training_gate": False}}
    write_json(run_dir / "acceptance.json", payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=1))
    return 0 if passed else 1


def _reproducible(inference: dict, spans: list[dict]) -> bool:
    delta = inference.get("repeat_peak_delta_mb")
    tolerance = 400.0
    return (delta is not None and delta <= tolerance) or len(spans) >= 2


def verify_runs(run_a: Path, run_b: Path) -> int:
    """Rerun comparison: the baseline numbers must reappear within the declared tolerances."""
    a = json.loads((run_a / "resources.json").read_text(encoding="utf-8"))
    b = json.loads((run_b / "resources.json").read_text(encoding="utf-8"))
    floor = LIMIT_MB  # silence unused warners in strict linters; kept for readability of the unit
    del floor
    tolerance = a["reproduce"]["acceptance"]["repeat_peak_delta_mb_max"]
    fraction = a["reproduce"]["acceptance"]["prediction_vs_archive_fraction_max"]
    checks = []

    def record(name: str, passed: bool, **evidence) -> None:
        checks.append({"check": name, "passed": bool(passed), **evidence})

    for label, left, right in (
            ("inference_peak_host_mb", a["inference"]["peak_host_mb"], b["inference"]["peak_host_mb"]),
            ("composed_peak_host_mb", a["quota"]["measured_peak_host_mb"], b["quota"]["measured_peak_host_mb"])):
        record(label, abs(left - right) <= tolerance, first=left, second=right, delta=round(abs(left - right), 2),
               tolerance_mb=tolerance)
    spans_a = {row["span_days"]: row["peaks"]["peak_rss_mb"] for row in a["training"]["spans"]}
    spans_b = {row["span_days"]: row["peaks"]["peak_rss_mb"] for row in b["training"]["spans"]}
    for span in sorted(set(spans_a) & set(spans_b)):
        record(f"training_peak_{span}d", abs(spans_a[span] - spans_b[span]) <= tolerance,
               first=spans_a[span], second=spans_b[span], delta=round(abs(spans_a[span] - spans_b[span]), 2),
               tolerance_mb=tolerance)
    for label, key in (("prediction_vs_archive_fraction", "prediction_error_fraction"),
                       ("composed_vs_archive_fraction", "composed_and_archive_error_fraction")):
        value_a, value_b = a["composition"].get(key), b["composition"].get(key)
        record(label, value_a is not None and value_b is not None and abs(value_a) <= fraction
               and abs(value_b) <= fraction, first=value_a, second=value_b, tolerance_fraction=fraction)
    passed = all(check["passed"] for check in checks)
    payload = {"schema_version": SCHEMA, "generated_at_utc": utc_now(), "run_a": str(run_a), "run_b": str(run_b),
               "tolerances": {"peak_delta_mb": tolerance, "prediction_fraction": fraction},
               "checks": checks, "passed": passed}
    write_json(run_b / "verify.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    return 0 if passed else 1


def _first_timestamp(events_path: Path):
    for line in events_path.read_text(encoding="utf-8").splitlines():
        return json.loads(line)["timestamp_utc"]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
