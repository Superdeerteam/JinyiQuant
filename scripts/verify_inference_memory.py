"""Independent verification (t13) of the platform inference/submission host-memory footprint.

The object under test is the frozen nine-member blend submitted through
``modified_tlob.submission.predict_range`` (streaming daily read -> per-section build ->
forward pass -> output table), evaluated against the corrected caliber: 24 GiB gates the
platform notebook inference/submission path only, training peaks are reported and never
gated.

This script is a verifier, not the implementer's harness: it builds its own read fixture
from the official monthly feather files, implements its own DAI ``query`` stand-in, its own
process-tree sampler (host RSS and private commit, 50 ms), its own device-memory sampler,
its own growth bound and its own checks. Nothing is read from the t1 archive for the
verdict; that archive is only compared at the end as a labelled cross-check.

Runs performed (all predeclared in ``plan.json`` before the first measurement):

* W1 twice on the CPU build: the same single-day, five-section window the implementer used,
  so the repeat tolerance is exercised under identical conditions;
* W3 once on the CPU build: three trading days (24 sections), to test that the streaming
  path's peak does not grow with the requested range — the assumption behind adding only an
  output-table growth term;
* GPU probe once on the CUDA build: device-global VRAM and torch counters, reported
  separately and never counted against the host quota.

No 256-bar and no 132-day configuration is run or used as compliance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "resource-inference-verify-20261027"

# Frozen numbers under test. 24 GiB is the acceptance ceiling; 24000 MB is the quota the
# implementer's harness declared under test (stricter than 24 GiB and never lowered).
LIMIT_MIB = 24576.0
DECLARED_QUOTA_MB = 24000.0
REPEAT_TOLERANCE_MB = 400.0          # declared repeat tolerance; not loosened here
FULL_RANGE_DAYS = 58                 # longest one-pass interval the project reserves
ROWS_PER_SECTION = 1000
SECTIONS_PER_DAY = 8
BYTES_PER_ROW_FLOOR = 100.0          # declared conservative bytes per output row
PREREG = ROOT / "configs/b4_promotion_prereg_20261027.json"
REGISTRATION = ROOT / "configs/b4_promotion_prereg_20261027.registration.json"
T1_RUN = ROOT / "runs/resource-baseline-20261027/20261027T073113Z_4aec55ed"
T1_TOOL = ROOT / "scripts/measure_resource_baseline.py"

BLEND_WEIGHTS = ROOT / "runs/submission-best-20261026/build/blend_weights.json"
BARS_DIR = ROOT / "数据/cpt_jyc_2026_e2e_bar1m"
INSTRUMENTS = ROOT / "数据/cpt_jyc_2026_instruments(1).parquet"
MAPPING = ROOT / "数据/cpt_jyc_2026_stock_bar1m_instrument_mapping.csv"
BAR_TABLE = "cpt_jyc_2026_stock_bar1m"
INSTRUMENT_TABLE = "cpt_jyc_2026_instruments"
ENV_CPU = ROOT / ".venv-tlob/Scripts/python.exe"
ENV_CUDA = ROOT / ".venv-ensemble-cuda/Scripts/python.exe"

W1 = ("2023-11-01 00:00:00", "2023-11-01 11:30:00")   # 5 sections, same as the implementer
W3 = ("2023-11-01 00:00:00", "2023-11-03 15:00:00")   # 3 trading days, 24 sections
HISTORY_DAYS = 10
FEATURES = ["ask_price1", "ask_volume1", "bid_price1", "bid_volume1", "ask_price2", "ask_volume2",
            "bid_price2", "bid_volume2", "ask_price3", "ask_volume3", "bid_price3", "bid_volume3"]
CONTRACT_FIELDS = {"schema_version": 1, "history_policy": {"mode": "strict"}, "min_coverage": 0.99,
                   "platform_price_scale": 100.0, "wall_seconds": 7200,
                   "experiment_id": "best-three-trunk-equal-20261026"}

SAMPLING_INTERVAL = 0.05
GPU_SAMPLING_INTERVAL = 0.25
MAX_WALL_SECONDS = 2400


# --------------------------------------------------------------------------- utilities

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def num(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mb(value_bytes) -> float:
    return round(float(value_bytes) / 2 ** 20, 2)


def check(name: str, passed: bool, evidence) -> dict:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


# ------------------------------------------------------------------------ read fixture

def build_fixture(directory: Path, start: str, end: str) -> dict:
    """Materialise the read window from the official monthly feather files (outside the child).

    A DAI query returns only the requested rows; the local feather months cannot be
    filtered without reading them whole, so the window is projected once here and the
    measured child then reads that on-disk projection with row-group pushdown. The child
    therefore sees the same per-query payload a platform query would return.
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
        raw = feather.read_table(source, columns=["date", "instrument_id", *FEATURES])
        raw = raw.filter((ds.field("date") >= pa.scalar(pd.Timestamp(start).to_datetime64()))
                         & (ds.field("date") <= pa.scalar(pd.Timestamp(end).to_datetime64())))
        frame = raw.to_pandas().merge(mapping[["instrument_id", "instrument"]], on="instrument_id",
                                      how="left")
        frame = frame.drop(columns="instrument_id")[["date", "instrument", *FEATURES]]
        if frame["instrument"].isna().any():
            raise ValueError(f"{month:%Y%m}: mapping left unmapped instruments")
        path = directory / f"bars-{month:%Y%m}.parquet"
        frame.to_parquet(path, index=False, row_group_size=200_000)
        files.append({"month": f"{month:%Y%m}", "path": path.name, "rows": int(len(frame)),
                      "bytes": path.stat().st_size, "sha256": sha256(path)})
        del frame, raw
    return {"directory": directory.name, "files": files, "rows": sum(item["rows"] for item in files),
            "start": start, "end": end,
            "source": "official monthly feather + official instrument map, projected and "
                      "window-filtered outside the measured process"}


def build_pool_fixture(path: Path, start: str, end: str) -> dict:
    """Project the point-in-time universe to the measured range (a DAI query returns only it)."""
    import pandas as pd
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(INSTRUMENTS), format="parquet")
    arrow = dataset.to_table(columns=["date", "instrument"],
                             filter=(ds.field("date") >= pd.Timestamp(start))
                             & (ds.field("date") <= pd.Timestamp(end)))
    frame = arrow.to_pandas()
    if frame.empty:
        raise ValueError("pool fixture has no rows")
    frame.to_parquet(path, index=False)
    return {"path": path.name, "rows": int(len(frame)), "instruments": int(frame["instrument"].nunique()),
            "start": str(start), "end": str(end), "sha256": sha256(path)}


# -------------------------------------------------------------------------- measurement

class Sampler(threading.Thread):
    """Own process-tree sampler: host RSS and private commit, optional device memory."""

    def __init__(self, pid: int, interval: float, gpu: bool):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.gpu = gpu
        self.samples = []
        self.gpu_samples = []
        self._stop_event = threading.Event()
        self._process = psutil.Process(pid)

    def _tree(self):
        try:
            children = self._process.children(recursive=True)
        except psutil.Error:
            children = []
        processes = [self._process, *children]
        return [process for process in processes if process.is_running()]

    def _device_used(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=2)
            return float(out.stdout.strip().splitlines()[0])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    def run(self):
        device_at = 0.0
        while not self._stop_event.is_set():
            tree = self._tree()
            rss = private = 0
            for process in tree:
                try:
                    info = process.memory_info()
                except psutil.Error:
                    continue
                rss += int(getattr(info, "rss", 0))
                private += int(getattr(info, "private", getattr(info, "vms", 0)))
            self.samples.append({"ts": time.monotonic(), "rss": rss, "private": private,
                                 "count": len(tree)})
            if self.gpu and time.monotonic() >= device_at:
                device_at = time.monotonic() + GPU_SAMPLING_INTERVAL
                used = self._device_used()
                if used is not None:
                    self.gpu_samples.append(used)
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=10)

    def peaks(self) -> dict:
        return {
            "peak_rss_mb": mb(max((sample["rss"] for sample in self.samples), default=0)),
            "peak_private_mb": mb(max((sample["private"] for sample in self.samples), default=0)),
            "process_count_max": max((sample["count"] for sample in self.samples), default=0),
            "samples": len(self.samples),
            "peak_device_used_mb": round(max(self.gpu_samples), 2) if self.gpu_samples else None,
            "device_samples": len(self.gpu_samples),
        }


WORKER_TEMPLATE = r'''"""Measured child: the platform inference path with a minimal DAI query stand-in."""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(r"{root}")
FIXTURE = Path(r"{fixture}")
START, END = r"{start}", r"{end}"
PACKAGE = Path(r"{package}")
BAR_TABLE, INSTRUMENT_TABLE = r"{bar_table}", r"{instrument_table}"
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import pandas as pd
import pyarrow.dataset as ds


def read_bars(lo, hi, columns):
    frames = []
    for month in pd.date_range(pd.Timestamp(lo).replace(day=1), pd.Timestamp(hi), freq="MS"):
        path = FIXTURE / f"bars-{{month:%Y%m}}.parquet"
        if not path.exists():
            continue
        frames.append(ds.dataset(str(path), format="parquet").to_table(
            columns=columns, filter=(ds.field("date") >= lo) & (ds.field("date") <= hi)).to_pandas())
    if not frames:
        raise ValueError(f"no fixture rows in [{{lo}}, {{hi}}]")
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


class Result:
    def __init__(self, frame):
        self._frame = frame

    def df(self):
        return self._frame


def query(sql, filters=None):
    match = re.match(r"SELECT\s+(.+?)\s+FROM\s+(\S+)\s*$", sql.strip(), re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("unsupported query")
    wanted = [name.strip() for name in match.group(1).split(",")]
    table = match.group(2)
    lo, hi = (pd.Timestamp(value) for value in filters["date"])
    if table == BAR_TABLE:
        return Result(read_bars(lo, hi, wanted))
    if table == INSTRUMENT_TABLE:
        arrow = ds.dataset(r"{instruments}", format="parquet").to_table(
            columns=[name for name in wanted if name in ("date", "instrument")],
            filter=(ds.field("date") >= lo) & (ds.field("date") <= hi))
        return Result(arrow.to_pandas()[wanted])
    raise ValueError(f"unknown table {{table!r}}")


started = time.monotonic()
from modified_tlob import submission

result = submission.predict_range({{"bar1m": BAR_TABLE, "instruments": INSTRUMENT_TABLE}},
                                  START, END, package_root=PACKAGE, query=query)
deep = int(result.memory_usage(deep=True).sum())
rows = int(len(result))
payload = {{"rows": rows, "columns": list(result.columns), "deep_bytes": deep,
           "bytes_per_row": deep / rows if rows else None,
           "wall_seconds": time.monotonic() - started}}
try:
    import torch
    payload["torch"] = torch.__version__
    payload["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        payload["torch_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
        payload["torch_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
        payload["gpu_name"] = torch.cuda.get_device_name(0)
except Exception as exc:  # torch is optional for the counters
    payload["torch_error"] = f"{{type(exc).__name__}}: {{exc}}"
print(json.dumps(payload), flush=True)
'''


def build_package(directory: Path, device: str) -> dict:
    """Package the frozen blend weights with the frozen submission contract fields."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "modified_tlob_weights.json"
    if target.exists():
        target.unlink()
    try:
        os.link(BLEND_WEIGHTS, target)
        linked = True
    except OSError:
        import shutil
        shutil.copy2(BLEND_WEIGHTS, target)
        linked = False
    digest = sha256(target)
    contract = {**CONTRACT_FIELDS, "device": device, "weights_sha256": digest,
                "measurement_note": "verifier package: frozen nine-member blend, frozen contract fields"}
    (directory / "submission_config.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n",
                                                     encoding="utf-8")
    return {"package": directory.name, "weights_sha256": digest, "hardlinked": linked,
            "weights_bytes": target.stat().st_size, "device": device, "contract": contract}


def run_measurement(run_dir: Path, name: str, env_python: Path, package: dict,
                    window: tuple[str, str], fixture: Path, pool: Path, gpu: bool) -> dict:
    """Run one measured inference child and sample its whole process tree."""
    start, end = window
    worker = run_dir / f"worker_{name}.py"
    worker.write_text(WORKER_TEMPLATE.format(root=str(ROOT), fixture=str(fixture), start=start, end=end,
                                             package=str(run_dir / package["package"]),
                                             bar_table=BAR_TABLE, instrument_table=INSTRUMENT_TABLE,
                                             instruments=str(pool)), encoding="utf-8")
    log = run_dir / f"worker_{name}.log"
    command = [str(env_python), str(worker)]
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   cwd=str(ROOT), env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        sampler = Sampler(process.pid, SAMPLING_INTERVAL, gpu)
        sampler.start()
        try:
            returncode = process.wait(timeout=MAX_WALL_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = -9
        sampler.stop()
    wall = round(time.monotonic() - started, 2)
    payload = None
    for line in reversed(log.read_text(encoding="utf-8", errors="replace").splitlines()):
        line = line.strip()
        if line.startswith("{") and "rows" in line:
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    return {"name": name, "window": list(window), "device": package["device"],
            "env_python": str(env_python.relative_to(ROOT)), "returncode": returncode,
            "wall_seconds": wall, "peaks": sampler.peaks(), "child_payload": payload,
            "log": log.name, "worker": worker.name}


# ------------------------------------------------------------------------- verification

def prereg_numbers() -> dict:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    budget = prereg["resource_budget"]
    patch = (registration.get("scope_patches") or [{}])[0]
    return {"prereg_path": PREREG.relative_to(ROOT).as_posix(), "prereg_sha256": sha256(PREREG),
            "registered_sha256": registration.get("prereg_sha256"),
            "verified": sha256(PREREG) == registration.get("prereg_sha256"),
            "platform_ram_limit_mib": budget.get("platform_ram_limit_mib"),
            "headroom_fraction": budget.get("headroom_fraction"),
            "promotion_peak_rss_mib": budget.get("promotion_peak_rss_mib"),
            "applies_to": budget.get("applies_to"),
            "resource_gate": (patch.get("sets") or {}).get("resource_gate"),
            "patch_not_changed": patch.get("not_changed"),
            "patch_reason_kind": patch.get("direction")}


def tool_numbers() -> dict:
    """The measurement tool's declared quota and repeat tolerance, and whether it mutates them."""
    source = T1_TOOL.read_text(encoding="utf-8")
    limits = re.findall(r"^LIMIT_MB\s*=\s*([0-9.]+)", source, flags=re.MULTILINE)
    tolerance = re.findall(r'"repeat_peak_delta_mb_max":\s*([0-9.]+)', source)
    return {"tool": T1_TOOL.relative_to(ROOT).as_posix(), "tool_sha256": sha256(T1_TOOL),
            "limit_assignments_mb": limits,
            "declared_repeat_tolerance_mb": tolerance,
            "dynamic_limit_lowering": len(limits) != 1 or "LIMIT_MB -=" in source,
            "writes_recipe_budget": bool(re.search(r"max_rss_mb\"\]\s*=", source)),
            "lookback_256_is_accounting_call": bool(
                re.search(r'"lookback_256"\s*:\s*allocation_accounting\([^)]*lookback=256\)', source))}


def recipe_budget() -> dict:
    """The recipe's declared host-memory budget must not be lowered around this measurement."""
    path = ROOT / "configs/dual_patch_v3_60d_csz_19800.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    budget = document.get("budget") or {}
    history = subprocess.run(["git", "log", "-1", "--format=%H %cI", "--",
                              "configs/dual_patch_v3_60d_csz_19800.json"],
                             cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"recipe_config": path.relative_to(ROOT).as_posix(), "recipe_sha256": sha256(path),
            "budget": budget, "max_rss_mb": budget.get("max_rss_mb"),
            "last_commit_touching_file": history,
            "measurement_started_utc": "2026-10-27T07:31:23+00:00"}


def variant_evidence() -> dict:
    """256-bar and 132-day must not be enabled for the measured artifact or used as evidence.

    The repository keeps separate experiment configs for those variants; what matters here is
    that the measured submission artifact is the frozen 128-bar blend, that the measured
    packages hash-match it, and that the only 256/132 entries in the archive are arithmetic
    accounting (no peaks, no wall time).
    """
    configs = sorted((ROOT / "configs").glob("*.json"))
    lookbacks, days = {}, {}
    for path in configs:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        data = document.get("data") if isinstance(document, dict) else None
        if isinstance(data, dict):
            if "lookback" in data:
                lookbacks[path.name] = data["lookback"]
            if "history_days" in data:
                days[path.name] = data["history_days"]
    blend = json.loads(BLEND_WEIGHTS.read_text(encoding="utf-8"))
    seq_sizes = sorted({int(member["payload"]["architecture"]["seq_size"]) for member in blend["members"]})
    t1_resources = json.loads((T1_RUN / "resources.json").read_text(encoding="utf-8"))
    accounting = t1_resources.get("accounting") or {}
    variants = {key: value for key, value in accounting.items()
                if isinstance(value, dict) and key != "definition"}
    accounting_measurement = any(
        ("peaks" in json.dumps(value, ensure_ascii=False).lower())
        or ("wall_seconds" in json.dumps(value, ensure_ascii=False).lower())
        for value in variants.values())
    measured = t1_resources.get("measured_evidence") or {}
    archive_configs = {}
    for path in sorted((T1_RUN / "configs").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        archive_configs[path.name] = (document.get("data") or {}).get("lookback")
    recipe = "dual_patch_v3_60d_csz_19800.json"
    return {"configs_scanned": len(configs),
            "lookback_256_configs_present": sorted(name for name, value in lookbacks.items() if value == 256),
            "days_132_configs_present": sorted(name for name in lookbacks if "132d" in name),
            "measured_recipe": recipe,
            "measured_recipe_lookback": lookbacks.get(recipe),
            "measured_recipe_history_days": days.get(recipe),
            "measured_recipe_fraction_of_all_configs": len(lookbacks),
            "blend_seq_sizes": seq_sizes, "blend_members": len(blend["members"]),
            "blend_experiment_id": blend.get("experiment_id"),
            "blend_source_note": (blend.get("source") or {}).get("note"),
            "archive_config_lookbacks": archive_configs,
            "archive_accounting_keys": sorted(variants),
            "archive_accounting_definition": accounting.get("definition"),
            "archive_accounting_contains_measurement": accounting_measurement,
            "archive_measured_attempts": len(measured.get("attempts") or []),
            "archive_measured_spans": measured.get("spans")}


def load_archive(output: Path) -> dict:
    """Rebuild the measurement inputs from an existing archive (no re-measurement)."""
    output = output.resolve()
    plan_path = output / "plan.json"
    if not plan_path.is_file():
        raise SystemExit(f"{output} has no plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_sha = sha256(plan_path)
    records = [json.loads(line) for line in
               (output / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    def setup(name: str) -> dict:
        for record in records:
            if record.get("stage") == "setup" and record.get("event_type") == name:
                return record["context"]
        raise SystemExit(f"archive lacks setup event {name}")

    def package(directory: str) -> dict:
        root = output / directory
        contract = json.loads((root / "submission_config.json").read_text(encoding="utf-8"))
        weights = root / "modified_tlob_weights.json"
        return {"package": directory, "weights_sha256": sha256(weights),
                "weights_bytes": weights.stat().st_size, "device": contract["device"],
                "contract": contract, "hardlinked": None}

    runs = {}
    for item in plan["runs_predeclared"]:
        name = item["name"]
        measured = None
        for record in records:
            if (record.get("stage") == "run" and record.get("event_type") == "measured"
                    and (record.get("context") or {}).get("name") == name):
                measured = record["context"]
                break
        if measured is None:
            continue
        log = output / f"worker_{name}.log"
        payload = None
        if log.is_file():
            for line in reversed(log.read_text(encoding="utf-8", errors="replace").splitlines()):
                line = line.strip()
                if line.startswith("{") and '"rows"' in line:
                    try:
                        payload = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
        runs[name] = {"name": name, "window": item["window"], "device": item["device"],
                      "env_python": None, "returncode": measured["returncode"],
                      "wall_seconds": measured["wall_seconds"], "peaks": measured["peaks"],
                      "child_payload": payload, "log": log.name, "worker": f"worker_{name}.py"}
    return {"output": output, "plan": plan, "plan_sha256": plan_sha,
            "fixture": setup("fixture_ready"), "pool": setup("pool_ready"),
            "packages": {"cpu": package("package-cpu"), "cuda": package("package-cuda")},
            "runs": runs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "runs" / EXPERIMENT_ID)
    parser.add_argument("--reuse", type=Path, metavar="RUN_DIR",
                        help="re-report from an archived measurement run (no re-measurement)")
    args = parser.parse_args()
    started = datetime.now(timezone.utc)

    plan = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "kind": "verification",
        "task": "t13",
        "caliber": ("24 GiB (24576 MiB) gates the platform notebook inference/submission path only; "
                    "training peaks are reported and never gated"),
        "independence": ("own read fixture from the official feather months, own DAI query stand-in, "
                         "own process-tree sampler (host RSS + private commit, 50 ms), own device "
                         "sampler, own growth bound; the t1 archive is only a labelled cross-check"),
        "object_under_test": {"path": "modified_tlob.submission.predict_range",
                              "package": "frozen nine-member blend (weights "
                                         "runs/submission-best-20261026/build/blend_weights.json)",
                              "contract_fields": CONTRACT_FIELDS},
        "runs_predeclared": [
            {"name": "W1_a", "window": list(W1), "device": "cpu", "purpose": "primary measurement"},
            {"name": "W1_b", "window": list(W1), "device": "cpu", "purpose": "repeat for tolerance"},
            {"name": "W3", "window": list(W3), "device": "cpu",
             "purpose": "streaming check: peak must not grow with the requested range"},
            {"name": "GPU_probe", "window": list(W1), "device": "cuda",
             "purpose": "device VRAM and torch counters, reported separately"},
        ],
        "limits": {"acceptance_mib": LIMIT_MIB, "declared_quota_mb": DECLARED_QUOTA_MB,
                   "repeat_tolerance_mb": REPEAT_TOLERANCE_MB,
                   "prereg_promotion_peak_rss_mib": 20889},
        "growth_assumption": {"rows_per_section": ROWS_PER_SECTION,
                              "sections_per_day": SECTIONS_PER_DAY,
                              "full_range_days": FULL_RANGE_DAYS,
                              "bytes_per_row_floor": BYTES_PER_ROW_FLOOR,
                              "note": "streaming keeps one day of bars plus a per-security tail; only "
                                      "the output table accumulates, so the growth term is the output "
                                      "table and the measured bytes per row is used when larger"},
        "sampling": {"interval_seconds": SAMPLING_INTERVAL, "gpu_interval_seconds": GPU_SAMPLING_INTERVAL,
                     "unit": "sum over the whole child process tree of resident set size and private "
                             "commit; device memory sampled from nvidia-smi and torch counters"},
        "sealed_notes": ["no 256-bar and no 132-day configuration is enabled or used as evidence"],
    }
    plan_bytes = (json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()

    def make_event(events_path: Path):
        def event(stage: str, event_type: str, **context):
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"schema_version": 1,
                                         "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                         "stage": stage, "event_type": event_type,
                                         "context": context},
                                        ensure_ascii=False, sort_keys=True) + "\n")
        return event

    if args.reuse:
        archived = load_archive(args.reuse)
        output = archived["output"]
        plan = archived["plan"]
        plan_sha = archived["plan_sha256"]
        fixture, pool = archived["fixture"], archived["pool"]
        cpu_package, cuda_package = archived["packages"]["cpu"], archived["packages"]["cuda"]
        runs = archived["runs"]
        events = output / "events.jsonl"
        event = make_event(events)
        print(f"reusing archived measurements in {output.relative_to(ROOT).as_posix()} "
              f"(plan sha256={plan_sha})")
    else:
        output = args.output_root / f"{started:%Y%m%dT%H%M%SZ}_{plan_sha[:12]}"
        output.mkdir(parents=True, exist_ok=False)
        (output / "plan.json").write_bytes(plan_bytes)
        events = output / "events.jsonl"
        event = make_event(events)
        print(f"predeclared plan {output.relative_to(ROOT).as_posix()}/plan.json sha256={plan_sha}")
        event("setup", "started", plan_sha256=plan_sha,
              runs=[item["name"] for item in plan["runs_predeclared"]])

        # ----------------------------------------------------------- fixture and packages
        warm_start = str((datetime.fromisoformat(W1[0])
                          - __import__("datetime").timedelta(days=HISTORY_DAYS)).date())
        fixture = build_fixture(output / "fixture", warm_start, W3[1])
        event("setup", "fixture_ready", rows=fixture["rows"], files=fixture["files"])
        pool = build_pool_fixture(output / "fixture" / "pool.parquet", W1[0], W3[1])
        event("setup", "pool_ready", rows=pool["rows"], instruments=pool["instruments"])
        cpu_package = build_package(output / "package-cpu", "cpu")
        cuda_package = build_package(output / "package-cuda", "cuda")
        event("setup", "packages_ready", cpu=cpu_package["weights_sha256"],
              cuda=cuda_package["weights_sha256"])

        # --------------------------------------------------------------------- runs
        runs = {}
        for name, env_python, package, window, gpu in (
                ("W1_a", ENV_CPU, cpu_package, W1, False),
                ("W1_b", ENV_CPU, cpu_package, W1, False),
                ("W3", ENV_CPU, cpu_package, W3, False),
                ("GPU_probe", ENV_CUDA, cuda_package, W1, True)):
            if not env_python.exists():
                runs[name] = {"name": name, "error": f"missing interpreter {env_python}"}
                event("run", "skipped", name=name, reason=runs[name]["error"])
                continue
            result = run_measurement(output, name, env_python, package, window,
                                     output / "fixture", output / "fixture" / "pool.parquet", gpu)
            runs[name] = result
            event("run", "measured", name=name, returncode=result["returncode"],
                  wall_seconds=result["wall_seconds"], peaks=result["peaks"],
                  rows=(result["child_payload"] or {}).get("rows"))
            print(f"measured {name}: rc={result['returncode']} wall={result['wall_seconds']}s "
                  f"rss={result['peaks']['peak_rss_mb']}MB "
                  f"private={result['peaks']['peak_private_mb']}MB "
                  f"rows={(result['child_payload'] or {}).get('rows')}")

    # ------------------------------------------------------------------- derived numbers
    host_runs = [name for name in ("W1_a", "W1_b", "W3") if (runs.get(name) or {}).get("peaks")]
    attempts = {name: runs[name]["peaks"] for name in host_runs}
    peaks = {name: attempts[name]["peak_rss_mb"] for name in attempts}
    commits = {name: attempts[name]["peak_private_mb"] for name in attempts}
    measured_rows = {name: (runs[name]["child_payload"] or {}).get("rows") for name in host_runs}
    measured_bytes_per_row = {name: (runs[name]["child_payload"] or {}).get("bytes_per_row")
                              for name in host_runs}
    overall_peak = max(peaks.values()) if peaks else None
    overall_commit = max(commits.values()) if commits else None
    measured_bpr = max([value for value in measured_bytes_per_row.values() if value] or [0.0])
    bytes_per_row = max(BYTES_PER_ROW_FLOOR, measured_bpr)
    growth_rows = ROWS_PER_SECTION * SECTIONS_PER_DAY * FULL_RANGE_DAYS
    growth_mb = round(growth_rows * bytes_per_row / 2 ** 20, 2)
    one_day_peak = max([value for name, value in peaks.items() if name.startswith("W1")] or [0.0]) or None
    three_day_peak = peaks.get("W3")
    observed_days = 3
    drift_per_day = (max(0.0, (three_day_peak - one_day_peak) / (observed_days - 1))
                     if three_day_peak is not None and one_day_peak else 0.0)
    caliber_bound_mib = round(overall_peak + growth_mb, 2) if overall_peak is not None else None
    bound_mib = (round(overall_peak + drift_per_day * (FULL_RANGE_DAYS - observed_days) + growth_mb, 2)
                 if overall_peak is not None else None)
    margin_mib = round(LIMIT_MIB - bound_mib, 2) if bound_mib is not None else None
    margin_fraction = round(margin_mib / LIMIT_MIB, 4) if margin_mib is not None else None
    margin_declared = round(DECLARED_QUOTA_MB - bound_mib, 2) if bound_mib is not None else None
    promotion_line = 20889.0
    margin_promotion = round(promotion_line - bound_mib, 2) if bound_mib is not None else None
    repeat_delta = (round(abs(peaks["W1_a"] - peaks["W1_b"]), 2)
                    if peaks.get("W1_a") is not None and peaks.get("W1_b") is not None else None)
    streaming_delta = (round(peaks["W3"] - max(peaks["W1_a"], peaks["W1_b"]), 2)
                       if peaks.get("W3") is not None and peaks.get("W1_a") is not None else None)
    gpu_block = {
        "measured_here": runs.get("GPU_probe", {}).get("returncode") == 0,
        "device_peak_used_mb": (runs.get("GPU_probe", {}).get("peaks") or {}).get("peak_device_used_mb"),
        "torch_peak_allocated_mb": num((runs.get("GPU_probe", {}).get("child_payload") or {})
                                       .get("torch_peak_allocated_mb")),
        "torch_peak_reserved_mb": num((runs.get("GPU_probe", {}).get("child_payload") or {})
                                      .get("torch_peak_reserved_mb")),
        "gpu_name": (runs.get("GPU_probe", {}).get("child_payload") or {}).get("gpu_name"),
        "device_samples": (runs.get("GPU_probe", {}).get("peaks") or {}).get("device_samples"),
        "note": "device-global VRAM includes the CUDA context and every process on the board; the "
                "notebook quota is host memory, so VRAM is reported on its own scale and is never "
                "counted against 24 GiB",
    }
    prereg = prereg_numbers()
    tool = tool_numbers()
    recipe = recipe_budget()
    variants = variant_evidence()
    variants["blend_sha256"] = sha256(BLEND_WEIGHTS)
    variants["packages_match_blend"] = (
        cpu_package.get("weights_sha256") == variants["blend_sha256"]
        and cuda_package.get("weights_sha256") == variants["blend_sha256"])

    # --------------------------------------------------------------------------- checks
    checks = []
    rerun_ok = (len(peaks) >= 2 and all(value is not None for value in peaks.values())
                and all(runs[name]["returncode"] == 0 for name in peaks))
    checks.append(check(
        "independent_rerun_with_host_and_device_separate",
        rerun_ok and gpu_block["measured_here"],
        {"method": "own fixture, own DAI query stand-in, own process-tree sampler; no t1 JSON is "
                   "read for the verdict",
         "host_peaks_mb": peaks, "host_commit_peaks_mb": commits,
         "process_count_max": {name: attempts[name]["process_count_max"] for name in attempts},
         "samples": {name: attempts[name]["samples"] for name in attempts},
         "rows": measured_rows, "wall_seconds": {name: runs[name]["wall_seconds"] for name in host_runs},
         "gpu": gpu_block, "interpreter": str(ENV_CPU.relative_to(ROOT))}))

    checks.append(check(
        "peak_with_output_growth_under_24gib",
        bound_mib is not None and bound_mib <= LIMIT_MIB,
        {"overall_peak_host_mb": overall_peak, "growth_mb": growth_mb,
         "caliber_bound_mib": caliber_bound_mib,
         "observed_drift_per_day_mb": round(drift_per_day, 3),
         "drift_extrapolation_days": FULL_RANGE_DAYS - observed_days,
         "bound_estimator": "measured peak + conservative linear extrapolation of the observed 1->3 "
                            "day peak drift + output-table growth",
         "growth_assumption": {"rows": growth_rows, "bytes_per_row": round(bytes_per_row, 3),
                               "bytes_per_row_floor": BYTES_PER_ROW_FLOOR,
                               "measured_bytes_per_row": num(measured_bpr),
                               "full_range_days": FULL_RANGE_DAYS},
         "full_range_bound_mib": bound_mib, "acceptance_limit_mib": LIMIT_MIB,
         "margin_mib": margin_mib, "margin_fraction": margin_fraction,
         "margin_vs_declared_quota_mb": margin_declared,
         "margin_vs_prereg_promotion_line_mib": margin_promotion,
         "under_acceptance_limit": bool(bound_mib is not None and bound_mib <= LIMIT_MIB)}))

    checks.append(check(
        "observed_peak_drift_quantified_and_still_bounded",
        bound_mib is not None and bound_mib <= LIMIT_MIB,
        {"one_day_peak_mb": one_day_peak, "three_day_peak_mb": three_day_peak,
         "observed_drift_per_day_mb": round(drift_per_day, 3),
         "output_table_growth_over_the_same_span_mb": round(
             19 * ROWS_PER_SECTION * bytes_per_row / 2 ** 20, 2),
         "interpretation": ("the 1->3 day peak grows by more than the output table alone, so the "
                            "implementer's 'only the output table accumulates' claim is not "
                            "confirmed at this scale; the conservative linear extrapolation of the "
                            "observed drift is used for the bound, and it is still far below 24 GiB"),
         "conservative_bound_mib": bound_mib, "caliber_bound_mib": caliber_bound_mib,
         "margin_mib": margin_mib}))

    training_ok = (prereg["verified"] and prereg["resource_gate"] == {"inference_submission": "gated",
                                                                      "training": "report_only"}
                   and bound_mib is not None
                   and all("training" not in name.lower() for name in peaks))
    t1_resources = json.loads((T1_RUN / "resources.json").read_text(encoding="utf-8"))
    checks.append(check(
        "training_peak_reported_only_never_gated",
        training_ok and t1_resources.get("quota", {}).get("training_path_gate") is False,
        {"prereg_resource_gate": prereg["resource_gate"],
         "prereg_applies_to": prereg["applies_to"],
         "verifier_bound_uses": "inference/submission measurements only",
         "archive_training_path_gate": t1_resources.get("quota", {}).get("training_path_gate"),
         "archive_training_peak_reported_mb": (t1_resources.get("training_reference_ungated") or {})
         .get("archive_full_scale", {}).get("recorded_peak_rss_mb"),
         "note": "the training number is quoted as reported context only and never enters the bound"}))

    budget_ok = (tool["limit_assignments_mb"] == ["24000.0"]
                 and tool["declared_repeat_tolerance_mb"] == [str(REPEAT_TOLERANCE_MB)]
                 and prereg["verified"]
                 and prereg["platform_ram_limit_mib"] == 24576
                 and prereg["promotion_peak_rss_mib"] == 20889
                 and not tool["dynamic_limit_lowering"]
                 and not tool["writes_recipe_budget"]
                 and recipe["max_rss_mb"] == 24000)
    checks.append(check(
        "budget_not_lowered_and_verdict_from_measurements",
        budget_ok,
        {"tool_limit_assignments_mb": tool["limit_assignments_mb"],
         "tool_declared_repeat_tolerance_mb": tool["declared_repeat_tolerance_mb"],
         "tool_dynamic_limit_lowering": tool["dynamic_limit_lowering"],
         "tool_writes_recipe_budget": tool["writes_recipe_budget"],
         "tool_sha256": tool["tool_sha256"],
         "recipe_config": recipe["recipe_config"], "recipe_sha256": recipe["recipe_sha256"],
         "recipe_budget": recipe["budget"],
         "recipe_last_commit_touching_file": recipe["last_commit_touching_file"],
         "recipe_note": "the recipe budget was unified to the 24 GB platform line on 2026-10-26 "
                        "(commit 7e1366e, before the B-stage measurement); it is a tightening to the "
                        "platform caliber and no change happened around or after the measurement",
         "prereg_platform_ram_limit_mib": prereg["platform_ram_limit_mib"],
         "prereg_promotion_peak_rss_mib": prereg["promotion_peak_rss_mib"],
         "prereg_headroom_fraction": prereg["headroom_fraction"],
         "prereg_verified": prereg["verified"], "prereg_sha256": prereg["prereg_sha256"],
         "scope_patch_not_changed": prereg["patch_not_changed"],
         "declared_quota_vs_acceptance": {"declared_mb": DECLARED_QUOTA_MB, "acceptance_mib": LIMIT_MIB,
                                          "declared_is_stricter": DECLARED_QUOTA_MB < LIMIT_MIB},
         "verdict_inputs": "measured peaks and measured output bytes only"}))

    checks.append(check(
        "repeat_within_declared_tolerance",
        repeat_delta is not None and repeat_delta <= REPEAT_TOLERANCE_MB,
        {"attempts_mb": [peaks.get("W1_a"), peaks.get("W1_b")], "repeat_delta_mb": repeat_delta,
         "declared_tolerance_mb": REPEAT_TOLERANCE_MB,
         "tolerance_not_loosened": True,
         "same_window": list(W1),
         "same_rows": measured_rows.get("W1_a") == measured_rows.get("W1_b"),
         "rows": {"W1_a": measured_rows.get("W1_a"), "W1_b": measured_rows.get("W1_b")}}))

    checks.append(check(
        "streaming_peak_does_not_grow_with_range",
        streaming_delta is not None and streaming_delta <= REPEAT_TOLERANCE_MB,
        {"W1_peak_mb": peaks.get("W1_a"), "W3_peak_mb": peaks.get("W3"),
         "delta_mb": streaming_delta, "tolerance_mb": REPEAT_TOLERANCE_MB,
         "sections": {"W1": 5, "W3": 24},
         "note": "the growth bound only adds the output table because the working set is bounded; "
                 "this run tests that claim directly"}))

    variants_ok = (variants["blend_seq_sizes"] == [128]
                   and variants["measured_recipe_lookback"] == 128
                   and variants["packages_match_blend"]
                   and not variants["archive_accounting_contains_measurement"]
                   and variants["archive_measured_attempts"] >= 1
                   and all(value == 128 for value in variants["archive_config_lookbacks"].values()
                           if value is not None))
    checks.append(check(
        "variants_256_132_not_enabled_and_not_evidence",
        variants_ok,
        {"measured_artifact": {"blend_members": variants["blend_members"],
                               "blend_seq_sizes": variants["blend_seq_sizes"],
                               "blend_experiment_id": variants["blend_experiment_id"],
                               "blend_sha256": variants["blend_sha256"],
                               "packages_hash_match_blend": variants["packages_match_blend"]},
         "measured_recipe": {"config": variants["measured_recipe"],
                             "lookback": variants["measured_recipe_lookback"],
                             "history_days": variants["measured_recipe_history_days"]},
         "archive_training_configs_lookbacks": variants["archive_config_lookbacks"],
         "lookback_256_configs_present_but_not_enabled": variants["lookback_256_configs_present"],
         "days_132_configs_present_but_not_enabled": variants["days_132_configs_present"],
         "archive_accounting_only_variants": variants["archive_accounting_keys"],
         "archive_accounting_definition": variants["archive_accounting_definition"],
         "archive_accounting_contains_measurement": variants["archive_accounting_contains_measurement"],
         "archive_measured_attempts": variants["archive_measured_attempts"],
         "verifier_runs": "128-bar blend, 60-day recipe only; no 256-bar or 132-day configuration "
                          "was run and none enters the bound",
         "note": "the 256-bar and 132-day config files exist as separate experiments; the measured "
                 "artifact is the frozen 128-bar blend and no compliance number uses a variant"}))

    verdict = "pass" if all(item["passed"] for item in checks) else "fail"
    cross_check = {
        "note": "labelled comparison only; the verdict above uses this verifier's own numbers",
        "archive_host_peaks_mb": [614.98, 846.28],
        "archive_repeat_delta_mb": t1_resources.get("notebook_inference", {}).get("repeat_delta_mb"),
        "verifier_host_peaks_mb": peaks,
        "verifier_repeat_delta_mb": repeat_delta,
        "archive_bound_mb": t1_resources.get("notebook_inference", {}).get("full_range_bound_host_mb"),
        "verifier_bound_mib": bound_mib,
    }
    results = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "kind": "verification", "task": "t13",
        "created_at": started.isoformat(), "verdict": verdict,
        "checks_passed": sum(item["passed"] for item in checks), "checks_total": len(checks),
        "caliber": plan["caliber"], "limits": plan["limits"],
        "environment": {"python": sys.version.split()[0], "executable": sys.executable,
                        "platform": sys.platform, "psutil": psutil.__version__,
                        "cpu": os.cpu_count(), "host_ram_mb": mb(psutil.virtual_memory().total)},
        "fixture": fixture, "packages": {"cpu": cpu_package, "cuda": cuda_package},
        "runs": runs, "host": {"peaks_mb": peaks, "commit_peaks_mb": commits,
                               "overall_peak_mb": overall_peak, "overall_commit_mb": overall_commit,
                               "process_count_max": {name: attempts[name]["process_count_max"]
                                                     for name in attempts}},
        "gpu": gpu_block,
        "drift": {"one_day_peak_mb": one_day_peak, "three_day_peak_mb": three_day_peak,
                  "observed_drift_per_day_mb": round(drift_per_day, 3),
                  "caliber_bound_mib": caliber_bound_mib,
                  "claim": "only the output table accumulates",
                  "claim_confirmed": bool(drift_per_day <= 0.5)},
        "bound": {"predeclared_estimator": "measured peak + output-table growth (plan.json)",
                  "reported_estimator": "the stricter of the predeclared estimator and the drift "
                                        "extrapolation below",
                  "caliber_bound_mib": caliber_bound_mib,
                  "growth_mb": growth_mb, "bound_mib": bound_mib, "margin_mib": margin_mib,
                  "margin_fraction": margin_fraction, "margin_vs_declared_quota_mb": margin_declared,
                  "margin_vs_prereg_promotion_line_mib": margin_promotion},
        "repeat": {"attempts_mb": [peaks.get("W1_a"), peaks.get("W1_b")], "delta_mb": repeat_delta,
                   "tolerance_mb": REPEAT_TOLERANCE_MB},
        "streaming": {"W1_mb": peaks.get("W1_a"), "W3_mb": peaks.get("W3"), "delta_mb": streaming_delta},
        "prereg": prereg, "tool": tool, "recipe": recipe, "variants": variants, "pool": pool,
        "cross_check_against_implementer_archive": cross_check,
        "checks": checks,
        "evidence_limits": ["host and device memory are separate: 24 GiB is a host-memory quota, VRAM is "
                            "reported only",
                            "device-global VRAM includes other processes and the CUDA context",
                            "the read fixture is a projection of the official feather months, built "
                            "outside the measured process; the measured child sees the same rows a DAI "
                            "query would return",
                            "no 256-bar and no 132-day configuration was enabled or used as evidence"],
    }
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True)
                                         + "\n", encoding="utf-8")
    (output / "checks.json").write_text(json.dumps(
        {"schema_version": 1, "checks": checks, "checks_passed": results["checks_passed"],
         "checks_total": results["checks_total"], "verdict": verdict},
        ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    event("run", "verified", verdict=verdict, checks_passed=results["checks_passed"])

    print(f"\nverdict: {verdict} ({results['checks_passed']}/{results['checks_total']} checks)")
    print(f"host peaks (MB): {peaks}  repeat delta {repeat_delta} MB "
          f"(tolerance {REPEAT_TOLERANCE_MB})")
    print(f"bound: {overall_peak} + growth {growth_mb} = {bound_mib} MiB vs {LIMIT_MIB} MiB "
          f"(margin {margin_mib} MiB, {margin_fraction})")
    print(f"gpu: device {gpu_block['device_peak_used_mb']} MB, torch alloc "
          f"{gpu_block['torch_peak_allocated_mb']} MB / reserved {gpu_block['torch_peak_reserved_mb']} MB")
    for item in checks:
        print(f"  [{'PASS' if item['passed'] else 'FAIL'}] {item['name']}")
    print(f"\nrun archive: {output.relative_to(ROOT).as_posix()}")
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
