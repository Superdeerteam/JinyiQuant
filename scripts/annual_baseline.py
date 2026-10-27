"""Causal 2022 fixed-weight/monthly-refit diagnosis with a frozen execution plan."""

import argparse
import copy
import gc
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from jinyiquant.checks import verify_artifacts
from jinyiquant.config import Period, RunConfig
from jinyiquant.data import build_batch
from jinyiquant.evaluation import EVALUATION_VERSION
from jinyiquant.io import digest, file_sha256, read_json, utc_now, write_json
from jinyiquant.runner import _kill_tree, inference_keys
from jinyiquant.tracking import Recorder, artifacts, capture_source, environment, resource_usage
from modified_tlob.schema import lob_columns
from scripts.run_seed_ensemble import fit_worker, guarded_data, process_tree_usage, record_data

ROOT = Path(__file__).resolve().parents[1]
EXP = "annual-baseline-2022-20261017"
TIMES = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]
MONTHS = [f"2022{month:02}" for month in range(1, 13)]
RESERVED = ["2023-08-03/2023-08-05", "2024-07", "2024-Q4"]


def _days(values):
    if not values or any(not isinstance(value, str) for value in values):
        raise ValueError("Trading dates must be ISO date strings")
    dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if (dates.tz is not None or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
            or dates.strftime("%Y-%m-%d").tolist() != list(values)):
        raise ValueError("Trading dates must be unique, sorted exchange-local dates")
    return dates


def timestamps(days, times=TIMES):
    return [f"{day}T{clock}:00" for day in days for clock in times]


def validate_window(window, calendar):
    """Require the last ten independently registered trading days before each month."""
    dates, train, evaluation = _days(calendar), _days(window["train_dates"]), _days(window["evaluation_dates"])
    start, end = pd.Timestamp(window["evaluation_start"]), pd.Timestamp(window["evaluation_end_exclusive"])
    if (window["name"] not in MONTHS or start != pd.Timestamp(window["name"] + "01")
            or end != start + pd.offsets.MonthBegin(1)
            or pd.Timestamp(window["train_end_exclusive"]) != start
            or pd.Timestamp(window["train_start"]) != train[0]
            or len(train) != 10 or train[0] < pd.Timestamp("2021-12-01")
            or train.tolist() != dates[dates < start][-10:].tolist()
            or len(evaluation) < 10
            or evaluation.tolist() != dates[(dates >= start) & (dates < end)].tolist()
            or window["last_training_date"] != window["train_dates"][-1]
            or window["expected_timestamps"] != timestamps(window["evaluation_dates"])):
        raise ValueError("Noncausal window, wrong last-ten-day rule, incomplete month or reserved period")


def validate_recipe(raw):
    cfg = RunConfig.model_validate(raw)
    required = {"kind": "modified_tlob", "max_steps": 1500, "learning_rate": 1e-4,
                "batch_size": 32, "epochs": 12, "book_representation": "raw", "target_scaling": "train_std",
                "time_loss_weighting": "none", "trade_features": "zero", "hidden_dim": 40,
                "num_layers": 4, "num_heads": 1, "is_sin_emb": True}
    if (cfg.seed != 11 or cfg.evaluation_split != "validation" or cfg.data.lookback != 128
            or tuple(cfg.data.feature_columns) != lob_columns(3)
            or list(cfg.data.sample_times) != TIMES or list(cfg.data.training_sample_times or ()) != TIMES
            or any(getattr(cfg.model, key) != value for key, value in required.items())
            or cfg.logging.learning_curves):
        raise ValueError("Annual diagnosis must preserve the frozen seed11 current recipe")
    return cfg


def _input_paths(data_root):
    def existing(*names):
        candidates = [data_root / name for name in names if (data_root / name).exists()]
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one data source from: {names}")
        return candidates[0]

    paths = {"bars_path": existing("bars", "cpt_jyc_2026_e2e_bar1m"),
             "labels_path": existing("labels", "cpt_jyc_2026_vwap"),
             "instruments_path": existing("cpt_jyc_2026_instruments.parquet", "cpt_jyc_2026_instruments(1).parquet"),
             "instrument_map_path": data_root / "cpt_jyc_2026_stock_bar1m_instrument_mapping.csv"}
    if any(not path.exists() for path in paths.values()):
        raise FileNotFoundError("data-root requires unpacked standard bars, labels, universe and mapping files")
    sources = [paths["instruments_path"], paths["instrument_map_path"]]
    for role in ("bars_path", "labels_path"):
        months = set()
        for path in sorted(paths[role].rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".parquet", ".feather", ".arrow", ".csv"}:
                continue
            match = re.search(r"(?<!\d)(20\d{2})[-_]?([01]\d)(?!\d)", path.name)
            if not match:
                raise ValueError("Monthly input filenames must identify their year and month")
            month = f"{match[1]}{match[2]}"
            if "202112" <= month <= "202212":
                months.add(month)
                sources.append(path)
        if months != {"202112", *MONTHS}:
            raise ValueError(f"Incomplete 2021-12/2022 monthly input files: {role}")
    return paths, sources


def prepare(output, data_root, *, cache_dir=None, chunk_days=5, max_rss_mb=6144,
            job_wall_seconds=3600, total_wall_seconds=28800):
    """Freeze metadata/calendar before decoding any training or evaluation answers."""
    output, data_root = Path(output).resolve(), Path(data_root).resolve()
    if output.exists():
        raise FileExistsError("Use a new annual output directory; old plans are never overwritten")
    if min(chunk_days, max_rss_mb, job_wall_seconds, total_wall_seconds) <= 0:
        raise ValueError("Execution budgets must be positive")
    paths, sources = _input_paths(data_root)
    base = read_json(ROOT / "configs/fixed_seed_ensemble_20261017_plan.json")["configs"]["research_a"]
    validate_recipe(base)
    raw_dates = pd.to_datetime(pd.read_parquet(paths["instruments_path"], columns=["date"]).date)
    if raw_dates.isna().any() or isinstance(raw_dates.dtype, pd.DatetimeTZDtype):
        raise ValueError("Universe calendar must use exchange-local dates")
    dates = pd.DatetimeIndex(sorted(raw_dates.dt.normalize().unique()))
    dates = dates[(dates >= "2021-12-01") & (dates < "2023-01-01")]
    calendar = dates.strftime("%Y-%m-%d").tolist()
    output.mkdir(parents=True, exist_ok=False)
    cache_dir = Path(cache_dir).resolve() if cache_dir else output / "cache"
    windows, configs = [], {}
    for month in MONTHS:
        start = pd.Timestamp(month + "01")
        end = start + pd.offsets.MonthBegin(1)
        training = dates[dates < start][-10:].strftime("%Y-%m-%d").tolist()
        evaluation = dates[(dates >= start) & (dates < end)].strftime("%Y-%m-%d").tolist()
        if len(training) != 10:
            raise ValueError("Incomplete pre-year training calendar")
        window = {"name": month, "train_start": training[0], "train_end_exclusive": str(start.date()),
                  "train_dates": training, "last_training_date": training[-1],
                  "evaluation_start": str(start.date()), "evaluation_end_exclusive": str(end.date()),
                  "evaluation_dates": evaluation, "expected_timestamps": timestamps(evaluation)}
        validate_window(window, calendar)
        raw = copy.deepcopy(base)
        raw.update(experiment_id=EXP, seed=11, output_root=str(output),
                   hypothesis="Diagnose fixed and monthly-refit current recipe across every 2022 day and time")
        raw["data"].update({key: str(value) for key, value in paths.items()}, cache_dir=str(cache_dir))
        raw["splits"]["train"] = {"start": window["train_start"], "end": window["train_end_exclusive"]}
        raw["splits"]["validation"] = {"start": window["evaluation_start"], "end": window["evaluation_end_exclusive"]}
        raw["budget"].update(max_rss_mb=max_rss_mb, wall_seconds=job_wall_seconds)
        configs[month] = validate_recipe(raw).snapshot()
        windows.append(window)
    registry = output / "roles.json"
    write_json(registry, {"schema_version": 1, "evaluation_version": EVALUATION_VERSION,
                         "evaluation_year": 2022, "reserved": RESERVED,
                         "windows": windows, "allowed_label_roles": ["research_train", "research_evaluation"],
                         "rule": "Only 2021-12 training and 2022 train/evaluation; no backward weight evaluation"})
    checksums = {str(path): file_sha256(path) for path in sources}
    stats = {str(path): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in sources}
    evaluation_dates = [day for day in calendar if day.startswith("2022-")]
    plan = {"schema_version": 2, "experiment_id": EXP, "created_at_utc": utc_now(),
            "windows": windows, "configs": configs, "config_sha256": {k: digest(v) for k, v in configs.items()},
            "registry_path": str(registry), "registry_sha256": file_sha256(registry),
            "data_root": str(data_root), "source_inputs_sha256": checksums, "source_input_stat": stats,
            "trading_calendar": calendar, "evaluation_dates": evaluation_dates, "sample_times": TIMES,
            "expected_grid": timestamps(evaluation_dates), "calendar_source": "independent instruments table",
            "recipe": "seed11/raw12/all8/ten trading days/1500 steps/train_std/no smoothing",
            "budget": {"fits_max": 12, "chunk_days": chunk_days, "max_rss_mb": max_rss_mb,
                       "job_wall_seconds": job_wall_seconds, "total_wall_seconds": total_wall_seconds},
            "code": capture_source(output), "reserved": RESERVED,
            "rationale": "Complete 2022 research calendar; existing 2023/2024 weights and held-out periods untouched",
            "sources": ["https://qlib.readthedocs.io/en/stable/component/report.html"]}
    write_json(output / "plan.json", plan)
    write_json(output / "plan_identity.json", {"sha256": file_sha256(output / "plan.json")})
    return plan


def load_plan(root, *, verify_inputs=True):
    root = Path(root).resolve()
    if file_sha256(root / "plan.json") != read_json(root / "plan_identity.json")["sha256"]:
        raise ValueError("Frozen annual plan changed")
    plan = read_json(root / "plan.json")
    if file_sha256(plan["registry_path"]) != plan["registry_sha256"]:
        raise ValueError("Frozen annual role registration changed")
    if (plan["sample_times"] != TIMES or [w["name"] for w in plan["windows"]] != MONTHS
            or plan["evaluation_dates"] != [d for d in plan["trading_calendar"] if d.startswith("2022-")]
            or plan["expected_grid"] != timestamps(plan["evaluation_dates"])):
        raise ValueError("Frozen annual calendar/grid is incomplete")
    for window in plan["windows"]:
        validate_window(window, plan["trading_calendar"])
        raw = plan["configs"][window["name"]]
        cfg = validate_recipe(raw)
        if (digest(raw) != plan["config_sha256"][window["name"]]
                or cfg.splits.train.start != pd.Timestamp(window["train_start"])
                or cfg.splits.train.end != pd.Timestamp(window["train_end_exclusive"])
                or cfg.splits.validation.start != pd.Timestamp(window["evaluation_start"])
                or cfg.splits.validation.end != pd.Timestamp(window["evaluation_end_exclusive"])):
            raise ValueError("Configuration differs from its frozen window")
    for relative, checksum in plan["code"]["files"].items():
        if file_sha256(ROOT / relative) != checksum:
            raise ValueError(f"Source changed after plan preparation: {relative}")
    for filename, checksum in plan["source_inputs_sha256"].items():
        path = Path(filename)
        actual = {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        if (actual != plan["source_input_stat"][filename]
                or (verify_inputs and file_sha256(path) != checksum)):
            raise ValueError(f"Frozen dataset changed: {filename}")
    return plan


def validate_data_snapshot(plan, metadata):
    for records in metadata["files"].values():
        for source in records:
            if plan["source_inputs_sha256"].get(source["path"]) != source["sha256"]:
                raise ValueError("Loaded input differs from the frozen dataset")


def verify_completed(root, month, kind, plan=None):
    root = Path(root).resolve()
    plan = plan or load_plan(root, verify_inputs=False)
    out = root / kind / month
    manifest = read_json(out / "manifest.json")
    if (manifest["status"] != "completed" or manifest["exit_code"] != 0
            or manifest["plan_sha256"] != file_sha256(root / "plan.json")
            or manifest["config_sha256"] != plan["config_sha256"][month]
            or manifest["run_id"] != f"{kind}-{month}-{digest(str(root))[:12]}"
            or read_json(out / "config.json") != plan["configs"][month]
            or not verify_artifacts(out)["passed"]):
        raise ValueError(f"Completed job identity or artifacts differ: {out}")
    expected = ({"model.pt", "modified_tlob_weights.json", "training.json", "training_identity.json", "checks.json"}
                if kind == "fit" else {"fixed.parquet", "monthly.parquet", "weights.json", "history_coverage.parquet"})
    if not expected.issubset({item["path"] for item in manifest["artifacts"]}):
        raise ValueError("Completed job is missing required artifacts")
    if kind == "fit":
        window = next(w for w in plan["windows"] if w["name"] == month)
        if (read_json(out / "training_identity.json")["dates"] != window["train_dates"]
                or read_json(out / "training.json")["steps"] != 1500
                or read_json(out / "checks.json")["passed"] is not True):
            raise ValueError("Training identity/steps/checks differ from the frozen plan")
    elif read_json(out / "weights.json") != weight_sources(root, month, plan):
        raise ValueError("Inference weight provenance differs from completed training")
    return manifest


def weight_sources(root, month, plan):
    result = {}
    evaluation_start = pd.Timestamp(next(w for w in plan["windows"] if w["name"] == month)["evaluation_start"])
    for arm, fitted in (("fixed", "202201"), ("monthly", month)):
        window = next(w for w in plan["windows"] if w["name"] == fitted)
        verify_completed(root, fitted, "fit", plan)
        if pd.Timestamp(window["train_end_exclusive"]) > evaluation_start:
            raise ValueError("A weight cannot predict before its training cutoff")
        path = Path(root) / "fit" / fitted / "model.pt"
        result[arm] = {"path": str(path), "sha256": file_sha256(path), "fit_month": fitted,
                       "train_dates": window["train_dates"], "last_training_date": window["last_training_date"],
                       "train_end_exclusive": window["train_end_exclusive"],
                       "config_sha256": plan["config_sha256"][fitted],
                       "fit_manifest_sha256": file_sha256(path.parent / "manifest.json")}
    return result


def _infer(plan, window, config, root, out, rec):
    from modified_tlob.adapter import TLOBAdapter

    sources = weight_sources(root, window["name"], plan)
    models = {path: TLOBAdapter.load(Path(path), device=config.model.device, recorder=rec)
              for path in {source["path"] for source in sources.values()}}
    frames, coverage = {arm: [] for arm in sources}, []
    count = plan["budget"]["chunk_days"]
    for offset in range(0, len(window["evaluation_dates"]), count):
        days = window["evaluation_dates"][offset:offset + count]
        first, last = pd.Timestamp(days[0]), pd.Timestamp(days[-1]) + pd.Timedelta(days=1)
        bundle = guarded_data(plan, config, window, first, last, labels=False)
        validate_data_snapshot(plan, bundle.metadata)
        record_data(rec, out / f"data_{offset}.json", bundle)
        keys = inference_keys(bundle, Period(start=first, end=last), config.data)
        if sorted(keys.date.dt.strftime("%Y-%m-%dT%H:%M:%S").unique()) != timestamps(days):
            raise ValueError("Inference universe differs from the independent date/eight-time grid")
        rec.event("preprocessing", "started", dates=days, samples=len(keys), labels_loaded=False)
        batch = build_batch(bundle.bars, keys, config.data)
        cutoff = batch.keys.date.to_numpy(dtype="datetime64[ns]")[:, None]
        if not (((batch.history_dates <= cutoff) | np.isnat(batch.history_dates)).all()
                and batch.coverage["min_section_coverage"] >= config.acceptance.min_coverage):
            raise ValueError("Causality/history coverage failed")
        detail = batch.keys.assign(observed=batch.effective_rows.any(axis=1),
            short=batch.effective_rows.sum(axis=1) < config.data.min_history).groupby("date").agg(
                universe_count=("instrument", "size"), observed_count=("observed", "sum"),
                short_history_count=("short", "sum")).reset_index()
        detail["history_coverage"] = detail.observed_count / detail.universe_count
        detail["timestamp"] = detail.pop("date").dt.strftime("%Y-%m-%dT%H:%M:%S")
        coverage.append(detail)
        rec.event("preprocessing", "completed", dates=days, resources=resource_usage())
        scores = {}
        for arm, source in sources.items():
            path = source["path"]
            rec.event("inference", "started", arm=arm, dates=days, labels_loaded=False,
                      shared_prediction=path in scores)
            if path not in scores:
                scores[path] = models[path].predict(batch)
            if len(scores[path]) != len(batch.keys) or not np.isfinite(scores[path]).all():
                raise ValueError("Incomplete or nonfinite predictions")
            frames[arm].append(batch.keys.assign(score=scores[path]))
            rec.event("inference", "completed", arm=arm, resources=resource_usage())
        del batch, bundle, scores
        gc.collect()
    for arm in sources:
        pd.concat(frames[arm], ignore_index=True).to_parquet(out / f"{arm}.parquet", index=False)
    pd.concat(coverage, ignore_index=True).to_parquet(out / "history_coverage.parquet", index=False)
    if sources != weight_sources(root, window["name"], plan):
        raise ValueError("Weights changed during inference")
    write_json(out / "weights.json", sources)


def worker(root, month, kind):
    plan = load_plan(root, verify_inputs=False)
    window = next(w for w in plan["windows"] if w["name"] == month)
    config = validate_recipe(plan["configs"][month])
    out = root / kind / month
    rec = Recorder(out, EXP)
    rec.run_id = read_json(out / "manifest.json")["run_id"]
    try:
        if kind == "fit":
            fit_worker(plan, window, config, out, rec)
            validate_data_snapshot(plan, read_json(out / "data.json"))
        elif kind == "infer":
            _infer(plan, window, config, root, out, rec)
        else:
            raise ValueError("Unknown annual worker kind")
        rec.event("run", "worker_completed", kind=kind)
    except BaseException as exc:
        rec.failure("worker", exc)
        raise


def _end_job(out, rec, status, started, peak, error=None):
    resources = {"wall_seconds": time.monotonic() - started, "peak_rss_mb": peak}
    rec.event("run", "ended", status=status, resources=resources)
    manifest = read_json(out / "manifest.json")
    manifest.update(status=status, exit_code=0 if status == "completed" else 1,
                    ended_at=utc_now(), resources=resources, artifacts=artifacts(out))
    if error:
        manifest["error"] = {"reason": str(error), "recovery": "preserve failed job; no automatic retraining"}
    write_json(out / "manifest.json", manifest)


def launch(root, plan, month, kind, remaining):
    out = root / kind / month
    out.mkdir(parents=True, exist_ok=False)
    rec = Recorder(out, EXP)
    rec.run_id = f"{kind}-{month}-{digest(str(root))[:12]}"
    config = plan["configs"][month]
    write_json(out / "config.json", config)
    runtime = environment()
    runtime["determinism"].update(device=config["model"]["device"], deterministic_algorithms=True,
        note="Requested CUDA; actual device/preflight and determinism recorded by worker events")
    write_json(out / "manifest.json", {"schema_version": 1, "run_id": rec.run_id, "experiment_id": EXP,
        "status": "running", "kind": kind, "month": month, "started_at": utc_now(),
        "plan_sha256": file_sha256(root / "plan.json"), "config_sha256": plan["config_sha256"][month],
        "code": plan["code"], "environment": runtime})
    rec.event("run", "started", kind=kind, month=month)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT), "PYTHONHASHSEED": "11",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    started, peak, last_sample = time.monotonic(), 0., 0.
    process = None
    try:
        with (out / "worker.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "scripts.annual_baseline", "worker", str(root),
                "--month", month, "--kind", kind], stdout=log, stderr=log, cwd=ROOT, env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                try:
                    usage = process_tree_usage(process.pid)
                    peak = max(peak, usage["rss_mb"])
                    if elapsed - last_sample >= 5:
                        rec.event("resources", "sample", **usage, elapsed_seconds=elapsed)
                        last_sample = elapsed
                except psutil.NoSuchProcess:
                    pass
                if elapsed > min(remaining, plan["budget"]["job_wall_seconds"]) or peak > plan["budget"]["max_rss_mb"]:
                    raise TimeoutError("Annual job wall time/RSS budget exceeded")
                time.sleep(.2)
        if process.returncode:
            raise RuntimeError(f"Annual worker failed ({process.returncode}); see {out / 'worker.log'}")
        _end_job(out, rec, "completed", started, peak)
        verify_completed(root, month, kind, plan)
    except BaseException as exc:
        if process is not None and process.poll() is None:
            _kill_tree(process)
        rec.failure("supervision", exc)
        _end_job(out, rec, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", started, peak, exc)
        raise
    print(f"{kind} {month} completed", flush=True)


def analyze(root):
    from scripts.analyze_annual_baseline import analyze as analyze_outputs

    return analyze_outputs(root)


def run(root, data_root=None, **options):
    root = Path(root).resolve()
    if not (root / "plan.json").exists():
        if data_root is None:
            raise ValueError("A new run requires --data-root pointing to unpacked standard data")
        prepare(root, data_root, **options)
        plan = load_plan(root, verify_inputs=False)
    else:
        plan = load_plan(root)
    previous = read_json(root / "manifest.json") if (root / "manifest.json").exists() else {}
    if previous.get("status") == "completed":
        for month in MONTHS:
            for kind in ("fit", "infer"):
                verify_completed(root, month, kind, plan)
        if not verify_artifacts(Path(previous["analysis_path"]))["passed"]:
            raise ValueError("Completed annual analysis artifacts changed")
        print("Completed annual run verified; no new fits or evaluation", flush=True)
        return
    spent = previous.get("resources", {}).get("wall_seconds", 0.)
    started = time.monotonic()
    rec = Recorder(root, EXP)
    manifest = {"schema_version": 1, "evaluation_version": EVALUATION_VERSION,
                "run_id": rec.run_id, "experiment_id": EXP,
                "status": "running", "started_at": previous.get("started_at", utc_now()),
                "plan_sha256": file_sha256(root / "plan.json"), "code": plan["code"],
                "environment": environment(), "reserved_period_views": 0,
                "resources": {"wall_seconds": spent}, "resumed": bool(previous)}
    write_json(root / "manifest.json", manifest)
    try:
        rec.event("run", "resumed" if previous else "started", training_fits_max=12, prior_wall_seconds=spent)
        for month in MONTHS:
            for kind in ("fit", "infer"):
                out = root / kind / month
                if out.exists():
                    verify_completed(root, month, kind, plan)
                    continue
                remaining = plan["budget"]["total_wall_seconds"] - spent - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Annual total wall budget exhausted")
                launch(root, plan, month, kind, remaining)
                manifest["resources"] = {"wall_seconds": spent + time.monotonic() - started}
                write_json(root / "manifest.json", manifest)
        analysis_path = analyze(root)
        if spent + time.monotonic() - started > plan["budget"]["total_wall_seconds"]:
            raise TimeoutError("Annual total wall budget exhausted during analysis")
        manifest.update(status="completed", ended_at=utc_now(), exit_code=0, analysis_path=str(analysis_path))
        rec.event("run", "completed", training_fits=12, reserved_period_views=0)
    except BaseException as exc:
        rec.failure("experiment", exc)
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        ended_at=utc_now(), exit_code=1, error={"reason": str(exc)})
        raise
    finally:
        manifest["resources"] = {"wall_seconds": spent + time.monotonic() - started}
        write_json(root / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "worker", "analyze"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--month", choices=MONTHS)
    parser.add_argument("--kind", choices=["fit", "infer"])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--chunk-days", type=int, default=5)
    parser.add_argument("--max-rss-mb", type=float, default=6144)
    parser.add_argument("--job-wall-seconds", type=float, default=3600)
    parser.add_argument("--total-wall-seconds", type=float, default=28800)
    args = parser.parse_args()
    options = {key: getattr(args, key) for key in
               ("cache_dir", "chunk_days", "max_rss_mb", "job_wall_seconds", "total_wall_seconds")}
    if args.action == "worker":
        if not args.month or not args.kind:
            parser.error("worker requires --month and --kind")
        worker(args.root.resolve(), args.month, args.kind)
    elif args.action == "analyze":
        analyze(args.root.resolve())
    elif args.action == "prepare":
        if args.data_root is None:
            parser.error("prepare requires --data-root")
        prepare(args.root, args.data_root, **options)
    else:
        run(args.root, args.data_root, **options)


if __name__ == "__main__":
    main()
