"""Measure and freeze the 40-day budget, then fit and compare the fixed 10/40-day arms.

Only one factor changes between arms: how many training dates the same 1500-step,
batch-32, learning-rate 1e-4, raw12, eight-timepoint recipe sees. The six 10-day
baseline fits are reused only after their training keys, labels and recipe are proved
identical to the frozen 10-day arm; no baseline is refitted for convenience.
"""

import argparse
import copy
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from jinyiquant.checks import batch_checks, check, prediction_checks
from jinyiquant.config import EvaluationConfig, Period, RunConfig
from jinyiquant.data import build_batch, history_diagnostics, select_targets
from jinyiquant.evaluation import evaluate
from jinyiquant.io import digest, file_sha256, read_json, utc_now, write_json
from jinyiquant.runner import _answer_coverage_check, _kill_tree, inference_keys, save_checks
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir, resource_usage
from modified_tlob.coverage import (
    DELTA_METRICS,
    SEEDS,
    baseline_seed_noise,
    coverage_decision,
    declared_exposure,
    minimum_mean_gain,
)
from modified_tlob.ensemble import paired_block_intervals
from scripts.run_seed_ensemble import (
    claim_training,
    complete_correlations,
    evaluation_chunks,
    frame_digest,
    guarded_data,
    process_tree_usage,
    record_data,
    small_batch,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "training-coverage-arms-20261018"
ARMS = ("10", "40")
KEYS = ["date", "instrument"]
BASELINE_RUN = ROOT / "runs/fixed-seed-ensemble-20261017/20261017T064944.073834Z_193b03ead7ff"
SCORING_RUN = ROOT / "runs/fixed-seed-ensemble-20261017/analysis-recovery/20261017T091219.300500Z_7c40b5b5d26d"
PILOT_STEPS = 100
EXPOSURE_STEPS, EXPOSURE_BATCH = 1500, 32
WINDOWS = ("research_a", "research_b")
METRIC_KEYS = ("rank_ic_mean", "long_short_return_mean", "min_daily_long_short_return")


class BudgetExceeded(RuntimeError):
    pass


def job_recorder(output):
    rec = Recorder(output, EXPERIMENT)
    rec.run_id = output.name + "_" + digest(str(output.resolve()))[:12]
    return rec


def round_up(value, unit):
    return float(int(np.ceil(float(value) / unit)) * unit)


def process_peak_mib():
    info = psutil.Process().memory_info()
    return float(getattr(info, "peak_wset", info.rss)) / 2**20


def evaluation_config(config):
    """Evaluation settings come from the frozen arm; never re-specified here."""
    return EvaluationConfig.model_validate(config["evaluation"])


def supervise(command, output, *, cap_rss_mb, cap_seconds, rec, seed=SEEDS[0], interval=5.0):
    """Independent watchdog over the real worker process tree, including its children."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT), "PYTHONHASHSEED": str(seed),
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    started, peak, last = time.monotonic(), 0.0, 0.0
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            while process.poll() is None:
                try:
                    usage = process_tree_usage(process.pid)
                    peak = max(peak, usage["rss_mb"])
                    elapsed = time.monotonic() - started
                    if elapsed - last >= interval:
                        rec.event("resources", "sample", **usage, elapsed_seconds=elapsed,
                                  caps={"wall_seconds": cap_seconds, "rss_mb": cap_rss_mb})
                        last = elapsed
                    if elapsed > cap_seconds or peak > cap_rss_mb:
                        _kill_tree(process)
                        rec.event("budget", "exceeded", level="ERROR", elapsed_seconds=elapsed, peak_rss_mb=peak,
                                  caps={"wall_seconds": cap_seconds, "rss_mb": cap_rss_mb})
                        raise BudgetExceeded(f"Declared wall/RSS cap exceeded: {output.name}")
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.2)
        finally:
            if process.poll() is None:
                _kill_tree(process)
    return {"returncode": process.returncode, "wall_seconds": time.monotonic() - started, "peak_rss_mb": peak}


# ---------------------------------------------------------------- resource measurement


def arm_declaration(frozen, window_name, arm):
    """Frozen training target artifact for one window and arm, with content fingerprints."""
    entry = next((e for e in frozen["training_selection"]
                  if e["window"] == window_name and str(e["training_days"]) == arm), None)
    if entry is None:
        raise ValueError(f"Frozen plan has no {window_name} {arm}-day training selection")
    if file_sha256(entry["artifact"]) != entry["sha256"]:
        raise ValueError("Frozen training target artifact changed")
    frame = pd.read_parquet(entry["artifact"])
    return {"artifact": entry["artifact"], "artifact_sha256": entry["sha256"], "samples": len(frame),
            "keys_sha256": frame_digest(frame[KEYS]),
            "labels_sha256": frame_digest(frame[[*KEYS, "target"]])}


def window_scope(frozen, window_name, arm):
    """Registered train/evaluation ranges; the widened arm start stays inside the window."""
    window = next(w for w in frozen["windows"] if w["name"] == window_name)
    dates = window["arms"][arm]["train_dates"]
    return {"train_start": dates[0], "train_end_exclusive": window["arms"][arm]["train_end_exclusive"],
            "evaluation_start": window["evaluation_start"],
            "evaluation_end_exclusive": window["evaluation_end_exclusive"],
            "evaluation_dates": window["evaluation_dates"]}


def measurement(frozen_path, window_name, arm, output, rec):
    """Resource pilot on the real preprocessing path plus a short optimizer probe."""
    import torch

    from modified_tlob.adapter import TLOBAdapter

    frozen = read_json(frozen_path)
    window = next(w for w in frozen["windows"] if w["name"] == window_name)
    config = RunConfig.model_validate({**window["arms"][arm]["config"], "seed": SEEDS[0]})
    declaration = arm_declaration(frozen, window_name, arm)
    plan = {"registry_path": frozen["registry_path"], "registry_sha256": frozen["registry_sha256"]}
    marks, started = [], time.monotonic()

    def mark(stage):
        marks.append({"stage": stage, "elapsed_seconds": time.monotonic() - started,
                      "rss_mb": resource_usage()["rss_mb"], "peak_rss_mb": process_peak_mib()})

    mark("start")
    scope = window_scope(frozen, window_name, arm)
    bundle = guarded_data(plan, config, scope, config.splits.train.start, config.splits.train.end, labels=True)
    mark("bars_and_labels_loaded")
    times = config.data.training_sample_times or config.data.sample_times
    targets = select_targets(bundle.labels, config.splits.train, config.data.label_horizon_minutes,
                             sample_times=times)
    dates = sorted(targets.date.dt.strftime("%Y-%m-%d").unique())
    identity = {"samples": len(targets), "keys_sha256": frame_digest(targets[KEYS]),
                "labels_sha256": frame_digest(targets[[*KEYS, "target"]])}
    if dates != window["arms"][arm]["train_dates"]:
        raise ValueError("Measurement data path does not reproduce the frozen training dates")
    if identity != {key: declaration[key] for key in identity}:
        raise ValueError("Measurement data path does not reproduce the frozen training set")
    mark("targets_selected")
    preprocessing_started = time.monotonic()
    batch = build_batch(bundle.bars, targets[KEYS], config.data)
    preprocessing_seconds = time.monotonic() - preprocessing_started
    mark("batch_built")
    # Mirror the fit worker: the raw history is released before optimization, and the
    # per-key diagnostics artifacts are written while the batch is resident.
    targets = batch.keys.merge(targets, on=KEYS, validate="one_to_one")
    history_diagnostics(batch).to_parquet(output / "train_history.parquet", index=False)
    targets.to_parquet(output / "train_targets.parquet", index=False)
    write_json(output / "coverage.json", batch.coverage)
    mark("diagnostics_written")
    del bundle
    gc.collect()
    mark("bars_released")
    mask = batch.effective_rows.sum(axis=1) >= config.data.min_history
    adapter = TLOBAdapter(config.model.model_copy(update={"max_steps": PILOT_STEPS,
                                                         "log_every_steps": PILOT_STEPS}),
                          config.data, seed=SEEDS[0], recorder=rec)
    training_started = time.monotonic()
    progress = adapter.fit(batch, targets.target.to_numpy(), sample_mask=mask)
    training_seconds = time.monotonic() - training_started
    if progress["steps"] != PILOT_STEPS:
        raise ValueError("Resource pilot did not run the declared probe steps")
    result = {"window": window_name, "arm": arm, "samples": len(targets), "training_days": len(dates),
              "stages": marks, "preprocessing_seconds": preprocessing_seconds, "pilot_steps": PILOT_STEPS,
              "pilot_seconds": training_seconds, "seconds_per_step": training_seconds / PILOT_STEPS,
              "pilot_samples_seen": int(progress["samples_seen"]), "target_scale": adapter.target_scale,
              "identity": identity, "frozen_identity": {key: declaration[key] for key in identity},
              "gpu_peak_memory_mb": (torch.cuda.max_memory_allocated() / 2**20
                                     if torch.cuda.is_available() else None),
              "torch": str(torch.__version__), "device": str(adapter.device),
              "process_peak_rss_mb": process_peak_mib()}
    mark("training_probe_completed")
    del batch, adapter, progress
    gc.collect()
    write_json(output / "measurement.json", result)
    rec.event("measurement", "completed", window=window_name, arm=arm, samples=len(targets),
              training_days=len(dates), preprocessing_seconds=preprocessing_seconds,
              pilot_steps=PILOT_STEPS, pilot_seconds=training_seconds,
              peak_rss_mb=result["process_peak_rss_mb"], gpu_peak_memory_mb=result["gpu_peak_memory_mb"])


def frozen_limits(measurements, *, training_steps, factor=1.25):
    """Executable caps from measured peaks; not a forecast about what a run will need."""
    broad = [m for m in measurements if m["arm"] == "40"]
    if len(broad) < 2:
        raise ValueError("Both 40-day windows must be measured before the budget is frozen")
    peak = max(m["process_peak_rss_mb"] for m in broad)
    preprocessing = max(m["preprocessing_seconds"] for m in broad)
    per_step = max(m["seconds_per_step"] for m in broad)
    steps_seconds = per_step * training_steps
    total_memory = psutil.virtual_memory().total / 2**20
    return {"max_rss_mb": round_up(peak * factor, 256),
            "wall_seconds": round_up(factor * (preprocessing + steps_seconds) + 300, 300),
            "basis": {"measured_peak_rss_mb": peak, "measured_preprocessing_seconds": preprocessing,
                      "measured_seconds_per_step": per_step, "declared_training_steps": training_steps,
                      "declared_samples_seen": training_steps * EXPOSURE_BATCH,
                      "rss_formula": f"round_up_256({factor} x measured peak host RSS)",
                      "wall_formula": (f"round_up_300({factor} x (measured preprocessing + {training_steps} x "
                                       "measured seconds/step) + 300s save/check slack)"),
                      "windows_measured": sorted(m["window"] for m in broad)},
            "host": {"total_memory_mb": total_memory, "measured_peak_below_system_memory": peak < total_memory}}


# ---------------------------------------------------------------- one evaluator for both arms


def score_predictions(predictions, targets, config):
    """Score on the frozen answer-key intersection; identical code path for both arms."""
    scored = predictions.merge(targets[KEYS], on=KEYS, validate="one_to_one")
    return evaluate(scored, targets, config, return_sections=True, undefined_constant_portfolios=True)


def baseline_window_metrics(frozen):
    """Re-score the six archived 10-day baseline fits against the frozen evaluation targets."""
    evaluation = evaluation_config(frozen["windows"][0]["arms"]["10"]["config"])
    views, costs, window_metrics = {}, {}, {}
    for name in WINDOWS:
        targets_path = SCORING_RUN / f"{name}_targets.parquet"
        targets = pd.read_parquet(targets_path)
        views[name] = {"targets": str(targets_path), "targets_sha256": file_sha256(targets_path), "seeds": {}}
        window_metrics[name] = {}
        for seed in SEEDS:
            prediction_path = BASELINE_RUN / f"inference/{name}/seed{seed}_predictions.parquet"
            predictions = pd.read_parquet(prediction_path)
            metrics, daily, _ = score_predictions(predictions, targets, evaluation)
            archived = SCORING_RUN / f"{name}_{'baseline' if seed == SEEDS[0] else f'seed{seed}'}_daily.csv"
            saved = pd.read_csv(archived)
            if (len(saved) != len(daily) or not np.array_equal(saved.date.astype(str), daily.date.astype(str))
                    or any(not np.allclose(saved[column], daily[column], atol=1e-12, rtol=1e-9)
                           for column in DELTA_METRICS)):
                raise ValueError("Archived baseline daily metrics do not reproduce from its own predictions")
            views[name]["seeds"][str(seed)] = {
                "prediction_path": str(prediction_path), "prediction_sha256": file_sha256(prediction_path),
                "daily_archive": str(archived), "metrics": metrics,
                "metrics_reproduced_from_predictions": True}
            for metric in (*METRIC_KEYS, "turnover_mean"):
                window_metrics[name].setdefault(metric, {})[str(seed)] = metrics[metric]
        cost_path = BASELINE_RUN / f"inference/{name}/inference_cost.json"
        costs[name] = read_json(cost_path) if cost_path.is_file() else None
    return views, costs, window_metrics


# ---------------------------------------------------------------- budget freeze


def build_frozen(frozen, prior_sha, measurements, limits, views, costs, noise, gain, coverage):
    """Same windows, dates and recipe; only the measured budget and the predeclared gates change."""
    plan = copy.deepcopy(frozen)
    by_key = {(m["window"], m["arm"]): m for m in measurements}
    for window in plan["windows"]:
        if window["name"] not in WINDOWS:
            continue
        counts = {}
        for arm in ARMS:
            measured = by_key[(window["name"], arm)]
            counts[int(arm)] = measured["samples"]
            if arm == "40":
                # Only the newly measured arm receives the measured caps; the 10-day arm declaration
                # is left exactly as the six reused baseline fits declared it.
                window["arms"][arm]["config"]["budget"] = {
                    "max_train_samples": int(measured["samples"]),
                    "max_rss_mb": limits["max_rss_mb"], "wall_seconds": limits["wall_seconds"]}
            window["arms"][arm]["frozen_identity"] = dict(measured["frozen_identity"])
        window["exposure"] = declared_exposure(counts, steps=EXPOSURE_STEPS, batch_size=EXPOSURE_BATCH)
        window["baseline_reuse"] = {str(seed): views[window["name"]]["seeds"][str(seed)] for seed in SEEDS}
        window["scoring_targets"] = {"path": views[window["name"]]["targets"],
                                     "sha256": views[window["name"]]["targets_sha256"]}
        window["baseline_inference_cost"] = costs.get(window["name"])
    plan.update(
        supersedes={"prior_sha256": prior_sha,
                    "reason": "Adds the measured executable budget (peak host memory, wall time, exact pool-sized "
                              "sample caps) and the baseline-only decision gates required before any 40-day fit."},
        execution_status="frozen_not_trained_budget_frozen",
        measured_budget={"limits": limits, "measurements": measurements},
        calibration={**noise, "minimum_mean_gain": gain,
                     "basis": "Six archived 10-day baseline fits on research_a/research_b, re-scored from their "
                              "frozen predictions and evaluation targets; frozen before any 40-day fit.",
                     "baseline_inference_cost": costs},
        policy={"primary_metric": "paired daily all-eight-time RankIC, seed-mean per day before pairing",
                 "direction": "maximize",
                 "minimum_mean_gain": gain["minimum_mean_gain"],
                 "minimum_mean_gain_basis": gain,
                 "noise": noise["noise"],
                 "guardrails": {"both_windows_improve": True, "gross_mean_nondecreasing": True,
                                "turnover_mean_nonincreasing": True,
                                "worst_day_delta_at_least": -noise["noise"]["min_daily_long_short_return"],
                                "gross_interval_lower_at_least": -noise["noise"]["long_short_return_mean"],
                                "turnover_interval_upper_at_most": noise["noise"]["turnover_mean"]},
                 "statistics": {"bootstrap_samples": 20000, "bootstrap_seed": 20261017, "confidence": 0.95,
                                "block_days": [5, 2], "paired": True, "window_stratified": True,
                                "sensitivity_rule": "Report both block lengths; conflicting conclusions are "
                                                    "inconclusive, and neither is chosen after seeing results."},
                 "exposure_policy": "Steps, batch size, learning rate and recipe stay fixed for this round; a larger "
                                    "pool gets no extra steps because that would confound coverage with compute.",
                 "interpretation_rule": "A non-accept result bounds what equal-compute date expansion achieves; it "
                                        "does not establish that more history is uninformative."},
        acceptance={**frozen.get("acceptance", {}),
                    "thresholds_status": "calibrated from six archived 10-day baseline fits before any 40-day fit",
                    "minimum_mean_gain": gain["minimum_mean_gain"],
                    "guardrail_tolerances": dict(noise["noise"])},
        diagnostic_budget={"new_training": 6, "fits_planned": 6, "inference_windows": 2,
                           "seed_predictions": 2 * len(SEEDS), "baseline_fits_retrained": 0,
                           "resource_pilot_steps": PILOT_STEPS, "resource_pilots": 4})
    return plan


def frozen_budget(prior_path, output_path, output_root, *, practical_improvement, noise_multiplier,
                  measurements_from=None):
    """Run the four resource pilots, calibrate the gates on the baselines and freeze both."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Frozen plans are immutable; select a new output name")
    prior_path = Path(prior_path).resolve()
    frozen = read_json(prior_path)
    if [w["name"] for w in frozen["windows"] if w["role"] == "research_used"] != list(WINDOWS):
        raise ValueError("The two research origins are required for this round")
    run = new_run_dir(Path(output_root))
    rec = Recorder(run, EXPERIMENT)
    started = time.monotonic()
    write_json(run / "manifest.json", {"schema_version": 1, "run_id": run.name, "experiment_id": EXPERIMENT,
        "started_at": utc_now(), "status": "running", "new_training": 0, "final_views": 0,
        "code": capture_source(run), "environment": environment(), "budget_only": True})
    measurements, checks = [], []
    try:
        rec.event("run", "started", new_training=0, resource_pilot_steps=PILOT_STEPS)
        reused = None
        if measurements_from is not None:
            reused = reuse_measurements(Path(measurements_from), frozen)
            measurements = reused["measurements"]
            rec.event("measurement", "reused", source=reused["source"],
                      measurements_sha256=reused["measurements_sha256"], pilots=len(measurements))
        else:
            for name in WINDOWS:
                for arm in ARMS:
                    path = run / f"measure_{name}_{arm}"
                    rec.event("measurement", "started", window=name, arm=arm)
                    launch(prior_path, path, "measure", name, arm=arm, frozen=str(prior_path),
                           cap={"wall_seconds": 2400, "rss_mb": 12288})
                    measurements.append(read_json(path / "measurement.json"))
        limits = frozen_limits(measurements, training_steps=EXPOSURE_STEPS)
        views, costs, window_metrics = baseline_window_metrics(frozen)
        noise = baseline_seed_noise(window_metrics)
        gain = minimum_mean_gain(noise, practical_improvement=practical_improvement,
                                 noise_multiplier=noise_multiplier)
        old_cap = frozen["windows"][0]["arms"]["10"]["config"]["budget"]["max_train_samples"]
        pooled = [m["samples"] for m in measurements if m["arm"] == "40"]
        checks.append(check("forty_day_pool_exceeds_old_sample_cap", min(pooled) > old_cap,
                            old_cap=old_cap, pooled_40=pooled))
        checks.append(check("measured_peak_within_system_memory",
                            limits["host"]["measured_peak_below_system_memory"], **limits["host"],
                            frozen_rss_cap_mb=limits["max_rss_mb"]))
        checks.append(check("baseline_metrics_reproduce_from_frozen_predictions",
                            all(seed["metrics_reproduced_from_predictions"]
                                for view in views.values() for seed in view["seeds"].values())))
        plan = build_frozen(frozen, file_sha256(prior_path), measurements, limits, views, costs, noise, gain,
                            {"practical_improvement": practical_improvement,
                             "noise_multiplier": noise_multiplier})
        if reused is not None:
            plan["measured_budget"]["reused_measurements"] = {key: value for key, value in reused.items()
                                                              if key != "measurements"}
        research = [i for i, window in enumerate(frozen["windows"]) if window["name"] in WINDOWS]
        checks.append(check("ten_day_arm_declaration_untouched",
                            all(plan["windows"][i]["arms"]["10"]["config"]
                                == frozen["windows"][i]["arms"]["10"]["config"] for i in research)))
        checks.append(check("recipe_and_dates_unchanged",
                            all(plan["windows"][i]["arms"][arm]["train_dates"]
                                == frozen["windows"][i]["arms"][arm]["train_dates"]
                                and plan["windows"][i]["arms"][arm]["config"] | {"budget": None}
                                == frozen["windows"][i]["arms"][arm]["config"] | {"budget": None}
                                for i in research for arm in ARMS)))
        if not save_checks(run, checks):
            raise ValueError("Budget freeze checks failed")
        write_json(output_path, plan)
        write_json(run / "frozen.json", plan)
        write_json(run / "measurements.json", {"measurements": measurements, "limits": limits,
                                               "calibration": noise, "minimum_mean_gain": gain,
                                               "baseline_seed_metrics": window_metrics,
                                               "reused_measurements": None if reused is None else
                                               {key: value for key, value in reused.items()
                                                if key != "measurements"}})
        rec.event("freeze", "completed", frozen=str(output_path),
                  frozen_sha256=file_sha256(output_path), limits=limits,
                  minimum_mean_gain=gain["minimum_mean_gain"])
        finish(run, "completed", 0, resources={"wall_seconds": time.monotonic() - started})
    except BaseException as exc:
        rec.failure("budget_freeze", exc)
        finish(run, "failed", 1, error={"reason": str(exc)})
        raise
    return run


def reuse_measurements(source, frozen):
    """Reuse recorded pilots only when they were taken on the same frozen training set."""
    source = Path(source)
    document = read_json(source / "measurements.json")
    measurements = document["measurements"]
    if sorted((m["window"], m["arm"]) for m in measurements) != sorted((name, arm) for name in WINDOWS
                                                                     for arm in ARMS):
        raise ValueError("Reusable measurements must cover both windows and both arms once")
    for item in measurements:
        declaration = arm_declaration(frozen, item["window"], item["arm"])
        if (item["frozen_identity"] != {key: declaration[key]
                                        for key in ("samples", "keys_sha256", "labels_sha256")}
                or item["identity"] != item["frozen_identity"]):
            raise ValueError("Reused measurement was taken against a different training set")
    return {"source": str(source.resolve()),
            "measurements_sha256": file_sha256(source / "measurements.json"), "measurements": measurements}


# ---------------------------------------------------------------- execution plan


def frozen_targets(name):
    path = SCORING_RUN / f"{name}_targets.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen evaluation targets are missing: {path}")
    return {"path": str(path), "sha256": file_sha256(path)}


TRAINING_IDENTITY_FILES = ("src/jinyiquant/data.py", "src/jinyiquant/config.py", "src/jinyiquant/checks.py",
                           "src/jinyiquant/tracking.py", "src/jinyiquant/io.py", "modified_tlob/adapter.py",
                           "modified_tlob/model.py", "modified_tlob/schema.py", "modified_tlob/device.py")


def source_identity(archives):
    """Prove reused baseline weights came from the same data/model/training code as this tree."""
    import hashlib
    import zipfile

    result = {}
    for archive_path in sorted({str(path) for path in archives}):
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            rows = {}
            for name in TRAINING_IDENTITY_FILES:
                current = ROOT / name
                if name not in names or not current.is_file():
                    raise ValueError(f"Training source snapshot cannot be compared: {name}")
                archived = hashlib.sha256(archive.read(name)).hexdigest()
                rows[name] = {"archived_sha256": archived, "current_sha256": file_sha256(current),
                              "identical": archived == file_sha256(current)}
        if not all(row["identical"] for row in rows.values()):
            raise ValueError(f"Training code differs from the archived baseline snapshot: {archive_path}")
        result[archive_path] = {"archive_sha256": file_sha256(Path(archive_path)), "files": rows}
    if len({tuple((name, row["archived_sha256"]) for name, row in value["files"].items())
            for value in result.values()}) != 1:
        raise ValueError("The six baseline snapshots do not share one training source")
    return result


def prepare(frozen_path, output_path, output_root):
    """Freeze the execution plan: reuse proved baselines, declare the six candidate fits."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Frozen plans are immutable; select a new output name")
    frozen_path = Path(frozen_path).resolve()
    frozen = read_json(frozen_path)
    run = new_run_dir(Path(output_root))
    rec = Recorder(run, EXPERIMENT)
    write_json(run / "manifest.json", {"schema_version": 1, "run_id": run.name, "experiment_id": EXPERIMENT,
        "started_at": utc_now(), "status": "running", "new_training": 0, "final_views": 0,
        "code": capture_source(run), "environment": environment()})
    try:
        rec.event("run", "started", new_training=0, mode="prepare")
        if not str(frozen["acceptance"]["thresholds_status"]).startswith("calibrated"):
            raise ValueError("Decision thresholds must be calibrated before execution")
        if file_sha256(frozen["registry_path"]) != frozen["registry_sha256"]:
            raise ValueError("Shared data-role registration changed")
        windows, inventory = [], {}
        for name in WINDOWS:
            window = next(w for w in frozen["windows"] if w["name"] == name)
            arms = {}
            for arm in ARMS:
                declaration = arm_declaration(frozen, name, arm)
                inventory[declaration["artifact"]] = declaration["artifact_sha256"]
                arms[arm] = {"config": copy.deepcopy(window["arms"][arm]["config"]),
                             "train_dates": window["arms"][arm]["train_dates"],
                             "train_end_exclusive": window["arms"][arm]["train_end_exclusive"],
                             "samples": declaration["samples"],
                             "frozen_identity": {key: declaration[key]
                                                 for key in ("samples", "keys_sha256", "labels_sha256")}}
            baseline = {}
            for seed in SEEDS:
                fitting = BASELINE_RUN / f"fits/{name}_s{seed}"
                identity = read_json(fitting / "training_identity.json")
                archived_config = read_json(fitting / "config.json")
                differences = {key for key in set(archived_config) | set(arms["10"]["config"])
                               if archived_config.get(key) != arms["10"]["config"].get(key)}
                if differences - {"experiment_id", "hypothesis", "seed"}:
                    raise ValueError(f"{name} seed {seed} baseline recipe differs from the frozen 10-day arm")
                for key in ("samples", "keys_sha256", "labels_sha256"):
                    if identity[key] != arms["10"]["frozen_identity"][key]:
                        raise ValueError(f"{name} seed {seed} baseline was fitted on different training data")
                weights = fitting / "model.pt"
                predictions = BASELINE_RUN / f"inference/{name}/seed{seed}_predictions.parquet"
                inventory[str(weights)] = file_sha256(weights)
                inventory[str(predictions)] = file_sha256(predictions)
                baseline[str(seed)] = {"fit_dir": str(fitting), "weights": str(weights),
                                       "weights_sha256": inventory[str(weights)],
                                       "predictions": str(predictions),
                                       "predictions_sha256": inventory[str(predictions)],
                                       "training_identity": identity,
                                       "config_differences": sorted(differences)}
            targets = frozen_targets(name)
            inventory[targets["path"]] = targets["sha256"]
            scope = window_scope(frozen, name, "40")
            windows.append({"name": name, "train_start": scope["train_start"],
                            "train_end_exclusive": scope["train_end_exclusive"],
                            "evaluation_start": scope["evaluation_start"],
                            "evaluation_end_exclusive": scope["evaluation_end_exclusive"],
                            "evaluation_dates": scope["evaluation_dates"], "arms": arms,
                            "exposure": declared_exposure({10: arms["10"]["samples"], 40: arms["40"]["samples"]},
                                                          steps=EXPOSURE_STEPS, batch_size=EXPOSURE_BATCH),
                            "baseline": baseline, "targets": targets,
                            "fit_once_before_evaluation": True})
        limits = frozen["measured_budget"]["limits"]
        identity_rows = source_identity([BASELINE_RUN / f"fits/{name}_s{seed}/source.zip"
                                         for name in WINDOWS for seed in SEEDS])
        plan = {"schema_version": 1, "experiment_id": EXPERIMENT, "created_at_utc": utc_now(),
                "frozen_path": str(frozen_path), "frozen_sha256": file_sha256(frozen_path),
                "hypothesis": frozen["hypothesis"], "only_factor": "training dates 10 -> 40",
                "recipe_fixed": {"raw12": True, "sample_times": 8, "lookback": 128, "loss": "MSE",
                                 "target_scaling": "train_std", "learning_rate": 1e-4, "max_steps": 1500,
                                 "batch_size": 32, "smoothing": "none"},
                "training_source_identity": identity_rows,
                "seeds": list(SEEDS), "deployment_seed": 11, "windows": windows,
                "policy": frozen["policy"], "calibration": frozen["calibration"],
                "statistics": frozen["policy"]["statistics"],
                "budget": {"local_peak_rss_cap_mib": limits["max_rss_mb"],
                           "per_fit_wall_cap_seconds": limits["wall_seconds"], "fit_attempts_max": 6,
                           "fit_stage_wall_cap_seconds": 6 * limits["wall_seconds"] + 2 * 1800,
                           "inference_chunk_days": 5, "inference_windows": len(WINDOWS)},
                "registry_path": frozen["registry_path"], "registry_sha256": frozen["registry_sha256"],
                "inputs_sha256": inventory, "final_views": 0,
                "access_policy": "research_a/research_b are already-used research data; confirmation, locked and "
                                 "legacy final ranges stay closed this round.",
                "notes": "Training attempts are capped at six; the six 10-day baseline fits are reused, not refitted."}
        write_json(run / "plan.json", plan)
        write_json(output_path, plan)
        rec.event("prepare", "completed", plan=str(output_path), plan_sha256=file_sha256(output_path),
                  fits_planned=6, baseline_fits_reused=6)
        finish(run, "completed", 0, resources={"wall_seconds": 0.0})
    except BaseException as exc:
        rec.failure("prepare", exc)
        finish(run, "failed", 1, error={"reason": str(exc)})
        raise
    return run


# ---------------------------------------------------------------- workers


def fit_worker(plan, window, arm, config, output, rec):
    from modified_tlob.adapter import TLOBAdapter

    declaration = window["arms"][arm]
    rec.event("data", "loading", role="research_train", arm=arm)
    bundle = guarded_data(plan, config, window, config.splits.train.start, config.splits.train.end, labels=True)
    record_data(rec, output / "data.json", bundle)
    times = config.data.training_sample_times or config.data.sample_times
    targets = select_targets(bundle.labels, config.splits.train, config.data.label_horizon_minutes,
                             sample_times=times)
    dates = sorted(targets.date.dt.strftime("%Y-%m-%d").unique())
    identity = {"samples": len(targets), "keys_sha256": frame_digest(targets[KEYS]),
                "labels_sha256": frame_digest(targets[[*KEYS, "target"]])}
    if dates != declaration["train_dates"]:
        raise ValueError("Training dates differ from the frozen declaration")
    if identity != declaration["frozen_identity"]:
        raise ValueError("Training keys or labels differ from the frozen declaration")
    if len(targets) > config.budget.max_train_samples:
        raise ValueError("Training sample budget exceeded")
    rec.event("preprocessing", "started", samples=len(targets), arm=arm)
    preprocessing_started = time.monotonic()
    batch = build_batch(bundle.bars, targets[KEYS], config.data)
    preprocessing_seconds = time.monotonic() - preprocessing_started
    targets = batch.keys.merge(targets, on=KEYS, validate="one_to_one")
    mask = batch.effective_rows.sum(axis=1) >= config.data.min_history
    checks = batch_checks(batch, targets, config.splits.train, config.acceptance.min_coverage)
    checks.append(_answer_coverage_check(
        inference_keys(bundle, config.splits.train, config.data, sample_times=times), targets,
        config.acceptance.min_coverage, "train_answer_coverage", "instruments"))
    checks.append(check("declared_training_days_and_all_eight_times",
                        len(dates) == len(declaration["train_dates"])
                        and targets.groupby(targets.date.dt.normalize()).date.nunique().eq(len(times)).all(),
                        training_days=len(dates), sample_times=len(times)))
    if not save_checks(output, checks):
        raise ValueError("Training data checks failed")
    history_diagnostics(batch).to_parquet(output / "train_history.parquet", index=False)
    targets.to_parquet(output / "train_targets.parquet", index=False)
    write_json(output / "selection.json", targets.attrs.get("selection", {}))
    write_json(output / "coverage.json", batch.coverage)
    write_json(output / "training_identity.json", {**identity, "dates": dates, "arm": arm})
    rec.event("data", "validated", samples=len(targets), dates=dates, checks_passed=True)
    rec.event("preprocessing", "completed", elapsed_seconds=preprocessing_seconds, resources=resource_usage())
    del bundle
    gc.collect()
    model = TLOBAdapter(config.model, config.data, seed=config.seed, recorder=rec)
    rec.event("training", "started", seed=config.seed, arm=arm, evaluation_labels_loaded=False)
    progress = model.fit(batch, targets.target.to_numpy(), sample_mask=mask)
    if config.model.max_steps is None or progress["steps"] != config.model.max_steps:
        raise ValueError("Did not complete exactly the frozen optimizer steps")
    rec.event("training", "completed", **progress)
    write_json(output / "training.json", {**progress, "target_scale": model.target_scale, "arm": arm,
        "training_days": len(dates), "prediction_units": "original_label_return",
        "evaluation_labels_loaded": False,
        "training_std": float(targets.target.to_numpy(dtype=np.float32)[mask].astype(np.float64).std())})
    rec.event("saving", "started")
    model.save(output / "model.pt")
    model.export_text(output / "modified_tlob_weights.json")
    rec.event("saving", "completed", sha256=file_sha256(output / "model.pt"))
    probe = small_batch(batch)
    expected = model.predict(probe)
    for filename in ("model.pt", "modified_tlob_weights.json"):
        restored = TLOBAdapter.load(output / filename, device=config.model.device)
        checks.append(check("save_load_" + filename,
                            np.allclose(expected, restored.predict(probe), atol=config.acceptance.repro_atol,
                                        rtol=config.acceptance.repro_rtol)))
        del restored
    checks.append(check("training_only_target_scale", np.isclose(model.target_scale,
        targets.target.to_numpy(dtype=np.float32)[mask].astype(np.float64).std(), atol=1e-14, rtol=1e-10)))
    if not save_checks(output, checks):
        raise ValueError("Model save/load or scale check failed")
    del probe, batch, model, targets
    gc.collect()


def infer_worker(plan, window, arm, config, paths, output, rec):
    import torch

    from modified_tlob.adapter import TLOBAdapter

    def sync():
        if config.model.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    seeds = tuple(sorted(paths))
    checks, frames, costs = [], {seed: [] for seed in seeds}, []
    rec.event("loading", "started", arm=arm, paths={str(s): str(p) for s, p in paths.items()})
    models = {seed: TLOBAdapter.load(path, device=config.model.device, recorder=rec) for seed, path in paths.items()}
    weight_hashes = {str(seed): file_sha256(path) for seed, path in paths.items()}
    rec.event("loading", "completed", weight_hashes=weight_hashes)
    for index, (dates, first, last) in enumerate(evaluation_chunks(window)):
        started = time.monotonic()
        bundle = guarded_data(plan, config, window, first, last, labels=False)
        record_data(rec, output / f"data_{index}.json", bundle)
        keys = inference_keys(bundle, Period(start=first, end=last), config.data)
        rec.event("preprocessing", "started", chunk=index, samples=len(keys), labels_loaded=False)
        batch = build_batch(bundle.bars, keys, config.data)
        preprocessing_seconds = time.monotonic() - started
        cutoffs = batch.keys.date.to_numpy(dtype="datetime64[ns]")[:, None]
        checks += [check(f"causal_history_{index}", ((batch.history_dates <= cutoffs)
                      | np.isnat(batch.history_dates)).all()),
                   check(f"history_coverage_{index}", batch.coverage["min_section_coverage"]
                         >= config.acceptance.min_coverage)]
        rec.event("data", "validated", chunk=index, labels_loaded=False, samples=len(keys))
        rec.event("preprocessing", "completed", chunk=index, elapsed_seconds=preprocessing_seconds)
        probe = small_batch(batch)
        for seed in seeds:
            models[seed].predict(probe)
        rec.event("inference", "started", chunk=index, samples=len(keys), label_input=False)
        order = list(seeds) if index % 2 == 0 else list(seeds)[::-1]
        scores, timings = {}, {}
        for seed in order:
            sync()
            started_seed = time.monotonic()
            scores[seed] = batch.keys.assign(score=models[seed].predict(batch))
            sync()
            timings[str(seed)] = time.monotonic() - started_seed
        for seed in seeds:
            frames[seed].append(scores[seed])
        checks.append(check(f"identical_prediction_keys_{index}",
                            all(scores[seed][KEYS].equals(scores[seeds[0]][KEYS]) for seed in seeds)))
        costs.append({"chunk": index, "dates": dates, "samples": len(keys), "timing_order": order,
                      "shared_preprocessing_seconds": preprocessing_seconds, "per_seed_seconds": timings,
                      "same_keys_sha256": frame_digest(batch.keys), "resources": resource_usage()})
        if index == 0:
            stocks = set(probe.keys.instrument)
            checks.extend(prediction_checks(models[seeds[0]], probe,
                                            bundle.bars[bundle.bars.instrument.isin(stocks)], config.data,
                                            config.acceptance.repro_atol, config.acceptance.repro_rtol))
        rec.event("inference", "completed", **costs[-1])
        history_diagnostics(batch).to_parquet(output / f"evaluation_history_{index}.parquet", index=False)
        del batch, probe, bundle, scores
        gc.collect()
    for seed in seeds:
        pd.concat(frames[seed], ignore_index=True).to_parquet(output / f"seed{seed}_predictions.parquet", index=False)
    checks.append(check("weights_frozen_across_all_chunks",
                        weight_hashes == {str(seed): file_sha256(path) for seed, path in paths.items()}))
    write_json(output / "inference_cost.json", {
        "arm": arm, "chunks": costs,
        "total_seconds": sum(sum(c["per_seed_seconds"].values()) + c["shared_preprocessing_seconds"]
                             for c in costs),
        "shared_preprocessing_seconds": sum(c["shared_preprocessing_seconds"] for c in costs),
        "measurement": "Synchronized CUDA; same full input keys; warmup excluded; shared preprocessing "
                       "reported separately", "platform_verified": False})
    if not save_checks(output, checks):
        raise ValueError("Inference behavior checks failed")


# ---------------------------------------------------------------- supervision


def launch(plan_path, output, kind, window_name, *, arm, seed=None, paths=None, remaining=None, cap=None,
           frozen=None):
    """Run one worker under an independent watchdog; never silently extend a cap."""
    output.mkdir(parents=True, exist_ok=False)
    plan = read_json(plan_path)
    window = next(w for w in plan["windows"] if w["name"] == window_name)
    config = copy.deepcopy(window["arms"][arm]["config"])
    if seed is not None:
        config["seed"] = seed
    write_json(output / "config.json", config)
    write_json(output / "job.json", {"kind": kind, "window": window_name, "arm": arm, "seed": seed,
                                     "paths": paths or {}, "frozen": frozen or ""})
    rec = job_recorder(output)
    started = time.monotonic()
    write_json(output / "manifest.json", {"schema_version": 1, "run_id": rec.run_id, "experiment_id": EXPERIMENT,
        "started_at": utc_now(), "status": "running", "kind": kind, "window": window_name, "arm": arm,
        "seed": seed, "plan_sha256": file_sha256(plan_path), "code": capture_source(output),
        "environment": environment(), "new_training": int(kind == "fit")})
    rec.event("run", "started", kind=kind, window=window_name, arm=arm, seed=seed)
    caps = cap or {"wall_seconds": min(plan["budget"]["per_fit_wall_cap_seconds"], remaining or 3600),
                   "rss_mb": plan["budget"]["local_peak_rss_cap_mib"]}
    command = [sys.executable, "-m", "scripts.run_training_coverage_arms", "worker",
               str(Path(plan_path).resolve()), "--output", str(output)]
    try:
        result = supervise(command, output, cap_rss_mb=caps["rss_mb"], cap_seconds=caps["wall_seconds"],
                           rec=rec, seed=seed or SEEDS[0])
    except BudgetExceeded as exc:
        finish(output, "budget_exceeded", 124,
               resources={"wall_seconds": time.monotonic() - started},
               error={"reason": str(exc), "recovery": "stop; no silent budget extension or extra fits"})
        raise
    finish(output, "completed" if result["returncode"] == 0 else "failed", result["returncode"],
           resources={"wall_seconds": result["wall_seconds"], "peak_rss_mb": result["peak_rss_mb"]})
    if result["returncode"]:
        raise RuntimeError(f"Worker failed; see {output / 'worker.log'}")
    print(f"{kind} {window_name} arm={arm} seed={seed}: {result['wall_seconds']:.1f}s "
          f"peak={result['peak_rss_mb']:.0f}MiB", flush=True)
    return result


def execute(plan_path, root):
    plan = read_json(plan_path)
    started = time.monotonic()
    rec = Recorder(root, EXPERIMENT)
    write_json(root / "manifest.json", {"schema_version": 1, "run_id": root.name, "experiment_id": EXPERIMENT,
        "started_at": utc_now(), "status": "running", "plan_sha256": file_sha256(plan_path),
        "code": capture_source(root), "environment": environment(), "final_views": 0})
    write_json(root / "preregistration.json", plan)
    attempts = []
    try:
        rec.event("run", "started", fits_planned=6, seeds=list(SEEDS))
        if file_sha256(plan["registry_path"]) != plan["registry_sha256"]:
            raise ValueError("Shared data-role registration changed")
        for path, checksum in plan["inputs_sha256"].items():
            if file_sha256(path) != checksum:
                raise ValueError(f"Frozen input changed: {path}")
        inference_paths = {}
        for window in plan["windows"]:
            name, identities = window["name"], []
            for seed in SEEDS:
                if len(attempts) >= plan["budget"]["fit_attempts_max"]:
                    raise RuntimeError("Training attempt cap exhausted")
                path = root / "fits" / f"{name}_s{seed}"
                claim = ROOT / "runs" / EXPERIMENT / "training_claims" / f"{name}_s{seed}.json"
                claim_training(claim, path, name, seed, plan_path)
                attempts.append({"window": name, "arm": "40", "seed": seed, "path": str(path),
                                 "status": "started"})
                write_json(root / "attempts.json", attempts)
                remaining = plan["budget"]["fit_stage_wall_cap_seconds"] - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Fit stage budget exhausted")
                launch(plan_path, path, "fit", name, arm="40", seed=seed, remaining=remaining)
                attempts[-1]["status"] = "completed"
                write_json(root / "attempts.json", attempts)
                identities.append(read_json(path / "training_identity.json"))
            if any({key: identity[key] for key in ("samples", "keys_sha256", "labels_sha256")}
                   != window["arms"]["40"]["frozen_identity"] for identity in identities):
                raise ValueError("Candidate training keys/labels differ across seeds or from the declaration")
            paths = {str(seed): str(root / "fits" / f"{name}_s{seed}" / "model.pt") for seed in SEEDS}
            write_json(root / f"{name}_weights.json",
                       {seed: {"path": value, "sha256": file_sha256(value)} for seed, value in paths.items()})
            inference_path = root / "inference" / name
            launch(plan_path, inference_path, "infer", name, arm="40", paths=paths)
            inference_paths[name] = inference_path
        analyze(plan, root, inference_paths, rec)
        if time.monotonic() - started > plan["budget"]["fit_stage_wall_cap_seconds"]:
            raise TimeoutError("Fit stage budget exhausted during scoring")
        rec.event("run", "completed", training_attempts=len(attempts), final_views=0)
        finish(root, "completed", 0, resources={"wall_seconds": time.monotonic() - started})
    except BaseException as exc:
        rec.failure("experiment", exc)
        write_json(root / "decision.json", {"result": "inconclusive", "action": "retain_10_day_reference",
            "reason": str(exc), "training_attempts": len(attempts), "final_views": 0, "model_promoted": False})
        finish(root, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", 1,
               resources={"wall_seconds": time.monotonic() - started}, error={"reason": str(exc)})
        raise


# ---------------------------------------------------------------- analysis


def analyze(plan, root, inference_paths, rec):
    """Score both arms with one code path, then pair the seed-mean daily deltas."""
    checks, summaries, paired_rows, per_seed, costs = [], [], [], {}, {}
    for window in plan["windows"]:
        name = window["name"]
        if file_sha256(window["targets"]["path"]) != window["targets"]["sha256"]:
            raise ValueError("Frozen evaluation targets changed")
        targets = pd.read_parquet(window["targets"]["path"])
        config = evaluation_config(window["arms"]["40"]["config"])
        seed_daily, seed_metrics, population = {}, {}, {}
        for arm in ARMS:
            for seed in SEEDS:
                path = (Path(inference_paths[name]) / f"seed{seed}_predictions.parquet" if arm == "40"
                        else Path(window["baseline"][str(seed)]["predictions"]))
                predictions = pd.read_parquet(path)
                metrics, daily, sections = score_predictions(predictions, targets, config)
                label = f"{name}_{arm}_s{seed}"
                checks.append(_answer_coverage_check(predictions[KEYS], targets, 0.99,
                                                     label + "_answer_coverage", "instruments"))
                checks.append(check(label + "_complete_correlation_coverage",
                                    complete_correlations(metrics, 80)))
                checks.append(check(label + "_exact_ten_day_eight_time_grid",
                                    len(sections) == 80 and sections.groupby("date").size().eq(8).all()
                                    and len(daily) == len(window["evaluation_dates"])))
                seed_daily[(arm, seed)] = daily.set_index("date")[list(DELTA_METRICS)]
                seed_metrics.setdefault(str(seed), {})[arm] = {key: metrics[key] for key in
                                                               (*METRIC_KEYS, "turnover_mean")}
                population.setdefault(arm, set()).add(metrics["sample_keys_sha256"] + metrics["targets_sha256"])
        if any(len(value) != 1 for value in population.values()) or population["10"] != population["40"]:
            raise ValueError("Arms were scored on different keys or answers")
        checks.append(check(f"{name}_identical_scoring_population", population["10"] == population["40"]))
        if any(not seed_daily[(arm, seed)].index.equals(seed_daily[(arm, SEEDS[0])].index)
               for arm in ARMS for seed in SEEDS):
            raise ValueError("Seed daily grids differ inside one arm")
        arm_daily = {arm: pd.concat([seed_daily[(arm, seed)] for seed in SEEDS]).groupby(level=0).mean()
                     for arm in ARMS}
        if not arm_daily["10"].index.equals(arm_daily["40"].index):
            raise ValueError("Arms were scored on different evaluation days")
        delta = arm_daily["40"] - arm_daily["10"]
        frame = delta.reset_index()
        frame["period"] = name
        paired_rows.append(frame)
        for seed in SEEDS:
            per_seed.setdefault(name, {}).setdefault(str(seed), {})["window_metrics"] = seed_metrics[str(seed)]
            per_seed[name][str(seed)]["mean_delta"] = {
                column: float((seed_daily[("40", seed)][column] - seed_daily[("10", seed)][column]).mean())
                for column in DELTA_METRICS}
        summaries.append({"period": name,
                          "arm_window_means": {arm: {column: float(arm_daily[arm][column].mean())
                                                     for column in DELTA_METRICS} for arm in ARMS},
                          "worst_day_delta": float(arm_daily["40"].long_short_return.min()
                                                   - arm_daily["10"].long_short_return.min()),
                          "min_day_delta": float(delta.long_short_return.min()),
                          "days": len(delta), "exposure": window["exposure"], "per_seed": per_seed[name]})
        costs[name] = read_json(Path(inference_paths[name]) / "inference_cost.json")
    paired = pd.concat(paired_rows, ignore_index=True)
    paired.to_csv(root / "paired_daily.csv", index=False)
    statistics = plan["statistics"]
    intervals = [paired_block_intervals(paired, block_days=block, samples=statistics["bootstrap_samples"],
                                        seed=statistics["bootstrap_seed"])
                 for block in statistics["block_days"]]
    write_json(root / "paired_intervals.json", intervals)
    decision = coverage_decision(paired, summaries, intervals, plan["policy"], cost_measured=bool(costs),
                                 exposure={window["name"]: window["exposure"] for window in plan["windows"]})
    write_json(root / "metrics.json", {"windows": summaries, "inference_cost": costs,
                                       "baseline_inference_cost": {name: read_json(BASELINE_RUN / f"inference/{name}/inference_cost.json")
                                                                   if (BASELINE_RUN / f"inference/{name}/inference_cost.json").is_file() else None
                                                                   for name in WINDOWS},
                                       "exposure": {window["name"]: window["exposure"] for window in plan["windows"]},
                                       "calibration": plan["calibration"]["noise"],
                                       "minimum_mean_gain": plan["policy"]["minimum_mean_gain"]})
    write_json(root / "decision.json", decision)
    if not save_checks(root, checks):
        raise ValueError("Scoring coverage checks failed")
    rec.event("evaluation", "decision", **decision)
    print(decision["result"], decision["action"], decision["mean_rank_ic_delta"], flush=True)


def rescore(plan_path, execution, destination):
    """Recompute the analysis from frozen weights and predictions; no training, no raw data."""
    execution = Path(execution).resolve()
    plan = read_json(plan_path)
    if read_json(execution / "manifest.json")["status"] != "completed":
        raise ValueError("Rescoring requires a completed execution archive")
    output = new_run_dir(Path(destination).resolve())
    rec = Recorder(output, EXPERIMENT)
    started = time.monotonic()
    write_json(output / "manifest.json", {"schema_version": 1, "run_id": output.name, "experiment_id": EXPERIMENT,
        "started_at": utc_now(), "status": "running", "new_training": 0, "new_inference": 0,
        "execution_source": str(execution), "final_views": 0, "code": capture_source(output),
        "environment": environment()})
    try:
        rec.event("run", "started", mode="rescore", new_training=0)
        write_json(output / "preregistration.json", plan)
        paths = {window["name"]: execution / "inference" / window["name"] for window in plan["windows"]}
        for name, path in paths.items():
            if read_json(path / "manifest.json")["status"] != "completed":
                raise ValueError(f"Completed candidate inference required: {name}")
        analyze(plan, output, paths, rec)
        finish(output, "completed", 0, resources={"wall_seconds": time.monotonic() - started})
    except BaseException as exc:
        rec.failure("rescore", exc)
        finish(output, "failed", 1, error={"reason": str(exc)})
        raise
    return output


# ---------------------------------------------------------------- research record


def conclusion_text(decision, exposure, windows):
    """State what this round measured, without claiming anything about more history."""
    def span(arm):
        values = sorted({round(window["exposure"]["arms"][arm]["epochs"], 4) for window in windows})
        return f"{values[0]:.2f}" if values[-1] - values[0] < 5e-3 else f"{values[0]:.2f}-{values[-1]:.2f}"

    per_window = ", ".join(f"{key}: {value:+.6f}" for key, value in decision["period_mean_deltas"].items())
    failed = [name for name, passed in decision["checks"].items() if not passed]
    head = (f"At the frozen exposure ({exposure['sample_exposures']} sample exposures = 1500 steps x 32 batch, "
            f"i.e. {span('10')} epochs of the 10-day pool but only {span('40')} epochs of the 40-day pool), the paired "
            f"daily all-eight-time RankIC difference (40-day minus 10-day, seed-mean) was "
            f"{decision['mean_rank_ic_delta']:+.6f} against a predeclared floor of "
            f"{decision['minimum_mean_gain']:.6f}; per window {per_window}.")
    if decision["result"] == "accept":
        return (head + " The candidate clears the predeclared gain, interval, replication, guardrail and cost gates, "
                "so 40 training dates are retained for the next, separately registered stage.")
    if decision["result"] == "inconclusive":
        return (head + " The interval or cost evidence is insufficient, so neither arm is promoted and the "
                "10-day reference is retained. This is not evidence that more history is uninformative; it is a "
                "limit of this equal-compute round with two research windows and three seeds.")
    return (head + f" The candidate fails the predeclared gates {', '.join(failed) or 'unspecified'}, so the "
            "10-day reference is retained. A pooled gain driven by one window while the other degrades is a "
            "replication failure, not an improvement, and it is not evidence that more history is uninformative: the "
            "40-day pool is only partially traversed at this fixed step count, and no steps or seeds were added "
            "mid-round.")


def record(plan_path, execution, output_path):
    """Write the structured research record from a completed archive; never hand-edited."""
    plan = read_json(plan_path)
    execution = Path(execution).resolve()
    manifest = read_json(execution / "manifest.json")
    if manifest["status"] != "completed":
        raise ValueError("A research record requires a completed execution archive")
    decision = read_json(execution / "decision.json")
    metrics = read_json(execution / "metrics.json")
    intervals = read_json(execution / "paired_intervals.json")
    attempts = read_json(execution / "attempts.json")
    frozen = read_json(plan["frozen_path"])
    fits = [{"window": attempt["window"], "arm": attempt["arm"], "seed": attempt["seed"],
             "run": attempt["path"],
             "resources": read_json(Path(attempt["path"]) / "manifest.json").get("resources"),
             "weights_sha256": file_sha256(Path(attempt["path"]) / "model.pt")} for attempt in attempts]
    exposures = {window["name"]: window["exposure"] for window in plan["windows"]}
    document = {
        "schema_version": 1, "experiment_id": EXPERIMENT, "recorded_at_utc": utc_now(),
        "hypothesis": plan["hypothesis"], "only_factor": plan["only_factor"],
        "recipe_fixed": plan["recipe_fixed"], "seeds": plan["seeds"], "deployment_seed": plan["deployment_seed"],
        "budget_freeze": {"frozen_path": plan["frozen_path"], "frozen_sha256": plan["frozen_sha256"],
                          "limits": frozen["measured_budget"]["limits"],
                          "measurements": frozen["measured_budget"]["measurements"],
                          "calibration": frozen["calibration"],
                          "minimum_mean_gain": frozen["policy"]["minimum_mean_gain"]},
        "baseline_reuse": {window["name"]: {seed: {key: row[key] for key in
                                                   ("weights_sha256", "predictions_sha256", "config_differences")}
                                            for seed, row in window["baseline"].items()}
                           for window in plan["windows"]},
        "execution": {"run_dir": str(execution),
                      "manifest_sha256": file_sha256(execution / "manifest.json"),
                      "plan_path": str(Path(plan_path).resolve()), "plan_sha256": file_sha256(plan_path),
                      "new_training_fits": len(fits), "reused_baseline_fits": 6, "fits": fits,
                      "inference_cost": metrics["inference_cost"],
                      "baseline_inference_cost": metrics["baseline_inference_cost"]},
        "exposure": exposures,
        "results": {"windows": metrics["windows"], "paired_intervals": intervals, "decision": decision},
        "conclusion": conclusion_text(decision, exposures[WINDOWS[0]], plan["windows"]),
        "retained_version": "10-day seed11 research reference (unchanged)",
        "candidate_status": "not promoted" if decision["result"] != "accept" else "promoted to next stage",
        "model_promoted": False, "final_views": 0, "platform_verified": False,
        "limitations": [
            "Two research windows and three seeds are selection data, not independent confirmation.",
            "Local gross metrics exclude fees, slippage and overlapping-horizon effects.",
            "The 40-day pool is only partially traversed at the frozen 1500 steps, so this compares arms at equal "
            "compute rather than measuring the value of more history.",
            "Confirmation, locked and legacy final ranges were not opened this round; the platform time and memory "
            "limits remain unverified.",
        ],
        "next_steps": [
            "If coverage matters, register a separate equal-epoch round with explicit step scaling rather than "
            "changing steps inside this one.",
            "Only after a passing research decision, register the confirmation opening with its own frozen gates.",
        ],
    }
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Research records are append-only; select a new output name")
    write_json(output_path, document)
    return output_path


# ---------------------------------------------------------------- entry point


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("budget", "prepare", "run", "worker", "rescore", "record"))
    parser.add_argument("path", type=Path, help="frozen config, execution plan, or execution archive")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--practical-improvement", type=float, default=0.002)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--measurements-from", type=Path)
    parser.add_argument("--execution", type=Path)
    args = parser.parse_args()
    root = args.output_root or ROOT / "runs" / EXPERIMENT
    if args.command == "budget":
        run = frozen_budget(
            (args.prior or ROOT / "configs/training_coverage_20261017_v3_frozen.json"),
            args.output or ROOT / "configs/training_coverage_20261017_v4_frozen.json", root,
            practical_improvement=args.practical_improvement, noise_multiplier=args.noise_multiplier,
            measurements_from=args.measurements_from)
        print(str(run), flush=True)
    elif args.command == "prepare":
        run = prepare(args.path, args.output or ROOT / "configs/training_coverage_20261018_v4_plan.json", root)
        print(str(run), flush=True)
    elif args.command == "run":
        run = new_run_dir(Path(root).resolve())
        print(str(run), flush=True)
        execute(args.path.resolve(), run)
    elif args.command == "rescore":
        print(str(rescore(args.path.resolve(), args.execution or args.path,
                          args.output or Path(root) / "rescore")), flush=True)
    elif args.command == "record":
        if args.execution is None or args.output is None:
            parser.error("record requires --execution <completed run> and --output <record.json>")
        print(str(record(args.path.resolve(), args.execution, args.output)), flush=True)
    else:
        output = args.output.resolve()
        rec = job_recorder(output)
        try:
            plan, job = read_json(args.path), read_json(output / "job.json")
            if file_sha256(args.path) != read_json(output / "manifest.json")["plan_sha256"]:
                raise ValueError("Frozen plan changed")
            config = RunConfig.model_validate(read_json(output / "config.json"))
            window = next(w for w in plan["windows"] if w["name"] == job["window"])
            if job["kind"] == "fit":
                fit_worker(plan, window, job["arm"], config, output, rec)
            elif job["kind"] == "infer":
                infer_worker(plan, window, job["arm"], config,
                             {int(seed): Path(path) for seed, path in job["paths"].items()}, output, rec)
            else:
                measurement(job["frozen"], job["window"], job["arm"], output, rec)
            rec.event("run", "completed", kind=job["kind"])
        except BaseException as exc:
            rec.failure("worker", exc)
            raise


if __name__ == "__main__":
    main()
