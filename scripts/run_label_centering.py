"""Fit the fixed 10-day/1500-step recipe on timestamp-centered official labels.

The only training treatment is the label: ``y* = official return - mean official
return over every training stock at the same full timestamp``. The mean is computed
once per timestamp over the complete selected training cross-section, never from a
random mini-batch. ``target_std`` remains the raw-label standard deviation, so the
target scale is unchanged. Inference emits the relative score directly, without
labels and without adding back any future cross-section mean. Evaluation keeps the
official raw labels; raw-label MSE and the centered training loss are reported
separately.

The comparison is the archived raw-label 10-day/1500-step three-seed arm. The
original calibrated gain floor and worst-day guardrail are reused unchanged. Seed 11
is shown only as progress; all six fits always run. This "demean only" choice has a
public cross-sectional precedent (Qlib processors) but is a minimal assumption here,
not a verified recipe.
"""

import argparse
import copy
import difflib
import hashlib
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.checks import check
from jinyiquant.config import EvaluationConfig, RunConfig
from jinyiquant.evaluation import evaluate
from jinyiquant.io import digest, file_sha256, read_json, utc_now, write_json
from jinyiquant.runner import _answer_coverage_check, save_checks
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir
from modified_tlob.coverage import DELTA_METRICS, arm_exposure, coverage_decision
from modified_tlob.ensemble import SEEDS
from scripts.run_coverage_step_scaling import analysis
from scripts.run_seed_ensemble import claim_training, complete_correlations
from scripts.run_training_coverage_arms import (
    TRAINING_IDENTITY_FILES,
    BudgetExceeded,
    arm_declaration,
    fit_worker,
    infer_worker,
    score_predictions,
    supervise,
    window_scope,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "label-centering-20261018"
KEYS = ["date", "instrument"]
WINDOWS = ("research_a", "research_b")
REFERENCE, CANDIDATE = "ref10", "cand_centered"
STEPS = 1500
BATCH = 32
FULL_SEEDS = tuple(SEEDS)
PROGRESS_SEEDS = (11,)
FROZEN = ROOT / "configs/training_coverage_20261017_v4_frozen.json"
REFERENCE_RUN = ROOT / "runs/fixed-seed-ensemble-20261017/20261017T064944.073834Z_193b03ead7ff"
SCORING_RUN = ROOT / "runs/fixed-seed-ensemble-20261017/analysis-recovery/20261017T091219.300500Z_7c40b5b5d26d"
PLAN = ROOT / "configs/label_centering_20261018_plan.json"
RUNS = ROOT / "runs" / EXPERIMENT
TREATMENT_FILES = ("modified_tlob/adapter.py", "src/jinyiquant/config.py")
QLIB_PRECEDENT = ("https://github.com/microsoft/qlib/blob/main/qlib/data/dataset/processor.py",
                  "Qlib applies cross-sectional processors per datetime; this round borrows only the "
                  "per-timestamp mean subtraction and does not claim an equivalent validated label recipe.")


def job_recorder(output):
    rec = Recorder(output, EXPERIMENT)
    rec.run_id = output.name + "_" + digest(str(output.resolve()))[:12]
    return rec


def fit_dir(root, name, seed):
    return Path(root) / "fits" / f"{name}_s{seed}"


def evaluation_config(window):
    return EvaluationConfig.model_validate(window["arms"][REFERENCE]["config"]["evaluation"])


# ------------------------------------------------------------------ label statistics


def label_centering_stats(targets):
    """Describe the full-timestamp centering without storing the mean vector."""
    dates = pd.to_datetime(targets["date"])
    raw = targets["target"].to_numpy(dtype=np.float32).astype(np.float64)
    frame = pd.DataFrame({"timestamp": dates.to_numpy(), "raw": raw})
    grouped = frame.groupby("timestamp", sort=False)["raw"]
    means = grouped.transform("mean").to_numpy()
    centered = raw - means
    timestamp_means = grouped.mean()
    counts = grouped.size()
    values = timestamp_means.to_numpy(dtype=np.float64)
    return {
        "formula": "official return - mean official return over every training stock at the same full timestamp",
        "label_source": "official training labels only",
        "scope": "every selected training row of each full timestamp; computed once before mini-batches "
                 "and before the min-history mask",
        "target_std_basis": "raw official training label standard deviation (not the centered standard deviation)",
        "timestamps": int(len(timestamp_means)), "rows": int(len(raw)),
        "min_rows_per_timestamp": int(counts.min()),
        "raw_mean": float(raw.mean()), "raw_std": float(raw.std()),
        "centered_mean": float(centered.mean()), "centered_std": float(centered.std()),
        "mean_abs_timestamp_mean": float(np.abs(values).mean()),
        "rms_timestamp_mean": float(np.sqrt(np.mean(values ** 2))),
        "max_abs_timestamp_mean": float(np.abs(values).max()),
        "max_abs_centered_timestamp_mean": float(
            pd.DataFrame({"timestamp": frame["timestamp"], "centered": centered})
            .groupby("timestamp", sort=False)["centered"].mean().abs().max()),
        "timestamp_means_sha256": digest([
            {"timestamp": str(timestamp), "mean": float(value)}
            for timestamp, value in timestamp_means.items()]),
    }


def assert_only_centering_changes(reference, candidate, label):
    """Candidate recipe differs from the raw-label 10-day arm in label centering only."""
    differing = {key for key in set(reference) | set(candidate) if reference.get(key) != candidate.get(key)}
    allowed = {"experiment_id", "hypothesis", "output_root", "budget", "model"}
    if differing - allowed:
        raise ValueError(f"{label}: unexpected recipe differences {sorted(differing - allowed)}")
    model_keys = sorted(key for key in set(reference["model"]) | set(candidate["model"])
                        if reference["model"].get(key) != candidate["model"].get(key))
    if model_keys != ["target_centering"]:
        raise ValueError(f"{label}: only model.target_centering may change, saw {model_keys}")
    if (reference["model"].get("target_centering", "none"),
            candidate["model"].get("target_centering")) != ("none", "timestamp_mean_train"):
        raise ValueError(f"{label}: centering change is not none -> timestamp_mean_train")
    if candidate["model"]["target_scaling"] != "train_std" or candidate["model"]["max_steps"] != STEPS:
        raise ValueError(f"{label}: raw-label train_std scaling and {STEPS} steps must stay fixed")
    if set(reference["budget"]) != set(candidate["budget"]):
        raise ValueError(f"{label}: budget schema must not change")


# ------------------------------------------------------------------ reusable reference


def reference_arm(frozen, name):
    """Prove the archived raw-label arm is the frozen 10-day/1500 recipe before reuse."""
    declaration = arm_declaration(frozen, name, "10")
    window = next(w for w in frozen["windows"] if w["name"] == name)
    entry = {"days": 10, "steps": STEPS, "samples": declaration["samples"],
             "frozen_identity": {key: declaration[key] for key in ("samples", "keys_sha256", "labels_sha256")},
             "artifact": declaration["artifact"], "artifact_sha256": declaration["artifact_sha256"],
             "config": copy.deepcopy(window["arms"]["10"]["config"]),
             "train_dates": list(window["arms"]["10"]["train_dates"]),
             "train_end_exclusive": window["arms"]["10"]["train_end_exclusive"], "per_seed": {}}
    for seed in FULL_SEEDS:
        fitting = REFERENCE_RUN / f"fits/{name}_s{seed}"
        identity = read_json(fitting / "training_identity.json")
        expected = {**entry["frozen_identity"], "dates": entry["train_dates"]}
        if {key: identity.get(key) for key in expected} != expected:
            raise ValueError(f"{name} seed {seed} reference was fitted on different training data")
        archived = read_json(fitting / "config.json")
        differing = {key for key in set(archived) | set(entry["config"])
                     if archived.get(key) != entry["config"].get(key)}
        if differing - {"experiment_id", "hypothesis", "seed"}:
            raise ValueError(f"{name} seed {seed} reference recipe differs from the frozen arm: "
                             f"{sorted(differing)}")
        training = read_json(fitting / "training.json")
        if training["steps"] != STEPS:
            raise ValueError(f"{name} seed {seed} reference did not run {STEPS} steps")
        if not np.isclose(training["target_scale"], training["training_std"], atol=1e-14, rtol=1e-10):
            raise ValueError(f"{name} seed {seed} reference did not scale by its raw label std")
        predictions = REFERENCE_RUN / f"inference/{name}/seed{seed}_predictions.parquet"
        if not predictions.is_file():
            raise FileNotFoundError(predictions)
        entry["per_seed"][str(seed)] = {
            "fit_dir": str(fitting), "weights": str(fitting / "model.pt"),
            "weights_sha256": file_sha256(fitting / "model.pt"),
            "predictions": str(predictions), "predictions_sha256": file_sha256(predictions),
            "training": {key: training[key] for key in ("loss", "target_scale", "training_std", "steps",
                                                        "samples", "samples_seen")}}
    return entry


def source_identity(archives, delta_path):
    """Check the reused fits share the training code; diff the two declared treatment files."""
    result, non_treatment = {}, {}
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
                              "identical": archived == file_sha256(current),
                              "treatment_file": name in TREATMENT_FILES}
                if name not in TREATMENT_FILES and archived != rows[name]["current_sha256"]:
                    raise ValueError(f"Training code differs from the archived snapshot: {name} in {archive_path}")
        signatures = {tuple((name, row["archived_sha256"]) for name, row in rows.items()
                            if name not in TREATMENT_FILES)}
        if len(signatures) != 1:
            raise ValueError("The reused snapshots do not share one non-treatment training source")
        non_treatment.update({name: row for name, row in rows.items() if name not in TREATMENT_FILES})
        result[archive_path] = {"archive_sha256": file_sha256(Path(archive_path)), "files": rows}
    with zipfile.ZipFile(archives[0]) as archive:
        old_adapter = archive.read(TREATMENT_FILES[0]).decode("utf-8")
        old_config = archive.read(TREATMENT_FILES[1]).decode("utf-8")
    current = {name: (ROOT / name).read_text(encoding="utf-8") for name in TREATMENT_FILES}
    diff = list(difflib.unified_diff(old_adapter.splitlines(), current[TREATMENT_FILES[0]].splitlines(),
                                     fromfile="archived/" + TREATMENT_FILES[0],
                                     tofile="current/" + TREATMENT_FILES[0], lineterm=""))
    diff += list(difflib.unified_diff(old_config.splitlines(), current[TREATMENT_FILES[1]].splitlines(),
                                      fromfile="archived/" + TREATMENT_FILES[1],
                                      tofile="current/" + TREATMENT_FILES[1], lineterm=""))
    if "target_centering" not in current[TREATMENT_FILES[0]] or "target_centering" in old_adapter:
        raise ValueError("The treatment diff does not contain the declared timestamp centering")
    if "target_centering" not in current[TREATMENT_FILES[1]] or "target_centering" in old_config:
        raise ValueError("The treatment diff does not contain the declared configuration field")
    if "timestamp_mean_train" not in current[TREATMENT_FILES[0]]:
        raise ValueError("The treatment diff is missing the timestamp_mean_train contract")
    write_json(delta_path, {
        "schema_version": 1, "treatment_files": list(TREATMENT_FILES),
        "archived_adapter_sha256": hashlib.sha256(old_adapter.encode()).hexdigest(),
        "current_adapter_sha256": file_sha256(ROOT / TREATMENT_FILES[0]),
        "archived_config_sha256": hashlib.sha256(old_config.encode()).hexdigest(),
        "current_config_sha256": file_sha256(ROOT / TREATMENT_FILES[1]),
        "diff_sha256": hashlib.sha256("\n".join(diff).encode()).hexdigest(), "diff": diff,
        "note": "The archived reference predates the inert date-decay loss-weighting code; this candidate "
                "sets time_loss_weighting=none, so only the new target-centering branch is active."})
    return result, non_treatment


# ------------------------------------------------------------------ plan freeze


def prepare(output_path=PLAN, *, frozen_path=FROZEN, output_root=RUNS):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Frozen plans are immutable; select a new output name")
    frozen_path = Path(frozen_path).resolve()
    frozen = read_json(frozen_path)
    if file_sha256(frozen["registry_path"]) != frozen["registry_sha256"]:
        raise ValueError("Shared data-role registration changed")
    if read_json(REFERENCE_RUN / "manifest.json")["status"] == "running":
        raise ValueError("The reference run is still running")
    run = new_run_dir(Path(output_root))
    rec = Recorder(run, EXPERIMENT)
    write_json(run / "manifest.json", {"schema_version": 1, "run_id": run.name, "experiment_id": EXPERIMENT,
        "started_at": utc_now(), "status": "running", "new_training": 0, "final_views": 0, "mode": "prepare",
        "code": capture_source(run), "environment": environment()})
    try:
        rec.event("run", "started", mode="prepare", steps=STEPS, seeds=list(FULL_SEEDS))
        limits = frozen["measured_budget"]["limits"]
        policy = copy.deepcopy(frozen["policy"])
        if (limits["max_rss_mb"], limits["wall_seconds"]) != (9472.0, 900.0):
            raise ValueError("The reused measured budget is not the frozen one")
        windows, inventory, centering = [], {}, {}
        for name in WINDOWS:
            scope = window_scope(frozen, name, "10")
            reference = reference_arm(frozen, name)
            targets = SCORING_RUN / f"{name}_targets.parquet"
            if not targets.is_file():
                raise FileNotFoundError(targets)
            inventory[str(targets)] = file_sha256(targets)
            for row in reference["per_seed"].values():
                inventory[row["predictions"]] = row["predictions_sha256"]
                inventory[row["weights"]] = row["weights_sha256"]
            artifact = Path(reference["artifact"])
            if file_sha256(artifact) != reference["artifact_sha256"]:
                raise ValueError("Frozen training target artifact changed")
            artifact_frame = pd.read_parquet(artifact)
            stats = label_centering_stats(artifact_frame)
            if stats["timestamps"] != len(reference["train_dates"]) * 8:
                raise ValueError(f"{name}: centering does not cover every training timestamp")
            if stats["max_abs_centered_timestamp_mean"] > 1e-12:
                raise ValueError(f"{name}: planned centered labels are not timestamp-zero-mean")
            centering[name] = stats
            candidate = {"days": 10, "steps": STEPS, "samples": reference["samples"],
                         "frozen_identity": dict(reference["frozen_identity"]),
                         "artifact": reference["artifact"], "artifact_sha256": reference["artifact_sha256"],
                         "train_dates": list(reference["train_dates"]),
                         "train_end_exclusive": reference["train_end_exclusive"],
                         "config": copy.deepcopy(reference["config"])}
            candidate["config"]["experiment_id"] = EXPERIMENT
            candidate["config"]["hypothesis"] = (
                "Centering training labels within each full timestamp, while keeping the raw-label target "
                "standard deviation, reduces unlearnable market-level noise and improves out-of-sample "
                "ranking enough to clear the retained raw-label 10-day reference under the unchanged gates.")
            candidate["config"]["output_root"] = str(RUNS / "fits")
            candidate["config"]["model"]["target_centering"] = "timestamp_mean_train"
            candidate["config"]["budget"] = {"max_train_samples": reference["samples"],
                                             "max_rss_mb": limits["max_rss_mb"],
                                             "wall_seconds": limits["wall_seconds"]}
            assert_only_centering_changes(reference["config"], candidate["config"], name)
            RunConfig.model_validate(candidate["config"])
            inventory[reference["artifact"]] = reference["artifact_sha256"]
            windows.append({
                "name": name, "train_start": scope["train_start"],
                "train_end_exclusive": scope["train_end_exclusive"],
                "evaluation_start": scope["evaluation_start"],
                "evaluation_end_exclusive": scope["evaluation_end_exclusive"],
                "evaluation_dates": scope["evaluation_dates"],
                "arms": {REFERENCE: reference, CANDIDATE: candidate},
                "exposure": {"batch_size": BATCH, "steps": STEPS,
                             "arms": {arm: arm_exposure(spec["samples"], steps=STEPS, batch_size=BATCH)
                                      for arm, spec in {REFERENCE: reference, CANDIDATE: candidate}.items()},
                             "interpretation": "Both arms use 1500 x 32 = 48,000 sample exposures on the same "
                                               "10 training dates and seeds; only the training-label definition "
                                               "differs."},
                "targets": {"path": str(targets), "sha256": inventory[str(targets)]},
                "fit_once_before_evaluation": True})
        identity_rows, non_treatment = source_identity(
            [REFERENCE_RUN / f"fits/{name}_s{seed}/source.zip" for name in WINDOWS for seed in FULL_SEEDS],
            run / "source_delta.json")
        inventory[str(run / "source_delta.json")] = file_sha256(run / "source_delta.json")
        plan = {
            "schema_version": 1, "experiment_id": EXPERIMENT, "created_at_utc": utc_now(),
            "hypothesis": "The raw-label 10-day/1500 reference spends capacity fitting a market-wide level that its "
                          "cross-sectional LOB inputs cannot predict. Subtracting the full-timestamp training-pool "
                          "mean removes that unlearnable component while keeping the raw-label target_std, so the "
                          "MSE should focus on relative returns and improve out-of-sample RankIC under the "
                          "unchanged gates.",
            "only_factor": "training label raw official return -> official return minus the mean official return "
                           "over every training stock at the same full timestamp; target_std stays the raw-label "
                           "std; network, inputs, 1500 steps, batch order, MSE, seeds and dates fixed",
            "source_frozen": {"path": str(frozen_path), "sha256": file_sha256(frozen_path)},
            "reference_source": {
                "run": str(REFERENCE_RUN),
                "manifest_status": read_json(REFERENCE_RUN / "manifest.json")["status"],
                "note": "The six raw-label fits completed; that run failed only after training while writing its "
                        "analysis. Its own prediction files and the completed analysis-recovery archive are reused.",
                "scoring_run": str(SCORING_RUN)},
            "recipe_fixed": {"raw12": True, "sample_times": 8, "lookback": 128, "loss": "MSE",
                             "target_scaling": "train_std", "target_std_fit": "raw official selected training "
                             "labels, full 10-day pool",
                             "learning_rate": 1e-4, "batch_size": BATCH, "steps": STEPS,
                             "sampler": "torch.randperm(seed) unchanged", "smoothing": "none",
                             "time_loss_weighting": "none"},
            "label_centering": {
                "formula": "y* = official return - group mean by full timestamp over the selected training pool",
                "cross_section": "one complete timestamp of the frozen 10-day training pool, all stocks and all "
                                 "eight sample times; never a random mini-batch",
                "timestamps_per_window": {name: centering[name]["timestamps"] for name in WINDOWS},
                "per_window": centering,
                "inference": "direct relative score from the fitted network times the raw-label target_std; no "
                             "label read and no future timestamp mean added back",
                "evaluation": "official raw labels for every reported ranking, long-short, turnover and raw MSE; "
                              "a centered-label diagnostic is reported separately and never selects the model",
                "public_precedent": {"url": QLIB_PRECEDENT[0], "use": QLIB_PRECEDENT[1]},
                "assumption_status": "minimal unverified assumption for this project; the public precedent does "
                                     "not establish that demean-only labels improve this model"},
            "windows": windows, "training_source_identity": identity_rows,
            "non_treatment_source": non_treatment,
            "source_delta": {"path": str(run / "source_delta.json"),
                             "sha256": inventory[str(run / "source_delta.json")],
                             "note": "Only adapter.py and config.py differ from the archived reference snapshot; "
                                     "every other training file is byte-identical."},
            "policy": policy, "calibration": frozen["calibration"], "statistics": policy["statistics"],
            "progress": {"seeds": list(PROGRESS_SEEDS),
                         "rule": "Seed 11 is printed and archived only as progress; all six fits always run and "
                                 "only the complete three-seed comparison can accept or reject an improvement.",
                         "mandatory_full_replay": True},
            "steps": {"reference": STEPS, "candidate": STEPS},
            "seeds": {"progress": list(PROGRESS_SEEDS), "full": list(FULL_SEEDS)},
            "deployment_seed": 11,
            "budget": {"local_peak_rss_cap_mib": limits["max_rss_mb"],
                       "per_fit_wall_cap_seconds": limits["wall_seconds"], "fit_attempts_max": 6,
                       "fit_stage_wall_cap_seconds": 6 * limits["wall_seconds"] + 2 * 1800,
                       "inference_chunk_days": 5, "inference_windows": len(WINDOWS)},
            "measured_budget": {"limits": limits, "source": str(frozen_path)},
            "registry_path": frozen["registry_path"], "registry_sha256": frozen["registry_sha256"],
            "inputs_sha256": inventory, "final_views": 0,
            "access_policy": "research_a/research_b are already-used research data; confirmation, locked and "
                             "legacy final ranges stay closed this round.",
            "notes": "Six mandatory centered-label fits. The six raw-label reference fits are reused, never "
                     "refitted. No other label transform, seed, step count or window is searched this round.",
            "retained_version": "10-day seed11 research reference (unchanged)",
        }
        checks = [
            check("reference_reuses_three_seeds_per_window",
                  all(len(window["arms"][REFERENCE]["per_seed"]) == len(FULL_SEEDS) for window in windows)),
            check("candidate_differs_only_in_label_centering",
                  all(assert_only_centering_changes(
                      window["arms"][REFERENCE]["config"], window["arms"][CANDIDATE]["config"],
                      window["name"]) is None for window in windows)),
            check("planned_labels_cover_every_timestamp_and_are_zero_mean",
                  all(row["timestamps"] == 80 and row["max_abs_centered_timestamp_mean"] < 1e-12
                      for row in centering.values())),
            check("unchanged_original_gain_floor_and_worst_day_tolerance",
                  np.isclose(policy["minimum_mean_gain"], 0.01164665094206816, atol=1e-15, rtol=0)
                  and np.isclose(policy["noise"]["min_daily_long_short_return"],
                                 0.00013896761043335348, atol=1e-18, rtol=0),
                  minimum_mean_gain=policy["minimum_mean_gain"],
                  worst_day_tolerance=policy["noise"]["min_daily_long_short_return"]),
        ]
        if not save_checks(run, checks):
            raise ValueError("Plan freeze checks failed")
        write_json(output_path, plan)
        write_json(run / "plan.json", plan)
        rec.event("prepare", "completed", plan=str(output_path), plan_sha256=file_sha256(output_path),
                  mandatory_fits=6, progress_seeds=list(PROGRESS_SEEDS))
        finish(run, "completed", 0, resources={"wall_seconds": 0.0})
    except BaseException as exc:
        rec.failure("prepare", exc)
        finish(run, "failed", 1, error={"reason": str(exc)})
        raise
    return run


# ------------------------------------------------------------------ worker supervision


def launch(plan_path, output, kind, window_name, *, seed=None, paths=None, cap=None):
    output.mkdir(parents=True, exist_ok=False)
    plan = read_json(plan_path)
    window = next(w for w in plan["windows"] if w["name"] == window_name)
    config = copy.deepcopy(window["arms"][CANDIDATE]["config"])
    if seed is not None:
        config["seed"] = seed
    write_json(output / "config.json", config)
    write_json(output / "job.json", {"kind": kind, "window": window_name, "arm": CANDIDATE, "seed": seed,
                                     "paths": paths or {}})
    rec = job_recorder(output)
    started = time.monotonic()
    write_json(output / "manifest.json", {"schema_version": 1, "run_id": rec.run_id, "experiment_id": EXPERIMENT,
        "started_at": utc_now(), "status": "running", "kind": kind, "window": window_name, "arm": CANDIDATE,
        "seed": seed, "plan_sha256": file_sha256(plan_path), "code": capture_source(output),
        "environment": environment(), "new_training": int(kind == "fit")})
    rec.event("run", "started", kind=kind, window=window_name, arm=CANDIDATE, seed=seed)
    caps = cap or {"wall_seconds": plan["budget"]["per_fit_wall_cap_seconds"],
                   "rss_mb": plan["budget"]["local_peak_rss_cap_mib"]}
    command = [sys.executable, "-m", "scripts.run_label_centering", "worker",
               str(Path(plan_path).resolve()), "--output", str(output)]
    try:
        result = supervise(command, output, cap_rss_mb=caps["rss_mb"], cap_seconds=caps["wall_seconds"],
                           rec=rec, seed=seed or FULL_SEEDS[0])
    except BudgetExceeded as exc:
        finish(output, "budget_exceeded", 124, resources={"wall_seconds": time.monotonic() - started},
               error={"reason": str(exc), "recovery": "stop; no silent budget extension or extra fits"})
        raise
    finish(output, "completed" if result["returncode"] == 0 else "failed", result["returncode"],
           resources={"wall_seconds": result["wall_seconds"], "peak_rss_mb": result["peak_rss_mb"]})
    if result["returncode"]:
        raise RuntimeError(f"Worker failed; see {output / 'worker.log'}")
    print(f"{kind} {window_name} seed={seed}: {result['wall_seconds']:.1f}s "
          f"peak={result['peak_rss_mb']:.0f}MiB", flush=True)
    return result


def worker(plan_path, output):
    rec = job_recorder(output)
    try:
        plan, job = read_json(plan_path), read_json(output / "job.json")
        if file_sha256(plan_path) != read_json(output / "manifest.json")["plan_sha256"]:
            raise ValueError("Frozen plan changed")
        config = RunConfig.model_validate(read_json(output / "config.json"))
        window = next(w for w in plan["windows"] if w["name"] == job["window"])
        if job["kind"] == "fit":
            fit_worker(plan, window, job["arm"], config, output, rec)
        else:
            infer_worker(plan, window, job["arm"], config,
                         {int(seed): Path(path) for seed, path in job["paths"].items()}, output, rec)
        rec.event("run", "completed", kind=job["kind"])
    except BaseException as exc:
        rec.failure("worker", exc)
        raise


# ------------------------------------------------------------------ execution


def prior_claims(stem):
    rows = []
    for claim in sorted((RUNS / "training_claims").glob(stem + "*.json")):
        record = read_json(claim)
        archive = Path(record["run"])
        rows.append({"claim": str(claim), "archive": str(archive),
                     "claimed_at_utc": record.get("claimed_at_utc"),
                     "complete": (archive / "model.pt").is_file() and (archive / "training.json").is_file()})
    return rows


def claim_fit(plan_path, name, seed, path):
    stem = f"{name}_s{seed}"
    prior = prior_claims(stem)
    if any(row["complete"] for row in prior):
        raise ValueError("Training claim already spent; no additional fits allowed")
    attempt = len(prior) + 1
    claim = RUNS / "training_claims" / (f"{stem}.json" if attempt == 1 else f"{stem}_attempt{attempt}.json")
    claim_training(claim, path, name, seed, plan_path)
    return attempt, prior


def fit_candidate(plan_path, root, plan, window, seed, rec, attempts, started):
    name = window["name"]
    if len(attempts) >= plan["budget"]["fit_attempts_max"]:
        raise RuntimeError("Training attempt cap exhausted")
    path = fit_dir(root, name, seed)
    attempt, prior = claim_fit(plan_path, name, seed, path)
    if prior:
        write_json(root / "recoveries.json", {
            "interrupted_archives": prior,
            "reason": "an earlier attempt at this seed was interrupted before it saved a model, so it left no "
                      "result to select on"})
        rec.event("training", "retry", window=name, seed=seed, attempt=attempt,
                  interrupted_archives=[row["archive"] for row in prior])
    attempts.append({"window": name, "arm": CANDIDATE, "seed": seed, "steps": STEPS,
                     "attempt": attempt, "path": str(path), "status": "started",
                     "prior_interrupted_archives": [row["archive"] for row in prior]})
    write_json(root / "attempts.json", attempts)
    remaining = plan["budget"]["fit_stage_wall_cap_seconds"] - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("Fit stage budget exhausted")
    launch(plan_path, path, "fit", name, seed=seed,
           cap={"wall_seconds": min(plan["budget"]["per_fit_wall_cap_seconds"], remaining),
                "rss_mb": plan["budget"]["local_peak_rss_cap_mib"]})
    training = read_json(path / "training.json")
    stats = training.get("target_centering_stats")
    expected = plan["label_centering"]["per_window"][name]
    if training["steps"] != STEPS or training["samples"] != window["arms"][CANDIDATE]["samples"]:
        raise ValueError("Centered fit did not complete the declared steps and samples")
    if training.get("target_centering") != "timestamp_mean_train" or not isinstance(stats, dict):
        raise ValueError("Centered fit did not record its timestamp centering")
    if stats["timestamps"] != expected["timestamps"] or stats["mask_excluded_rows"] != 0:
        raise ValueError("Centered fit did not use the complete training cross-sections")
    if not np.isclose(training["target_scale"], expected["raw_std"], atol=1e-14, rtol=1e-10):
        raise ValueError("Centered fit target_std is not the raw-label standard deviation")
    if not np.isclose(training["centered_target_std"], expected["centered_std"], atol=1e-14, rtol=1e-10):
        raise ValueError("Centered fit target statistics differ from the frozen plan")
    if not training["centered_target_std"] < training["raw_target_std"]:
        raise ValueError("Centering did not reduce target variance")
    attempts[-1]["status"] = "completed"
    attempts[-1]["training"] = {key: training[key] for key in
                                ("loss", "target_scale", "raw_target_std", "centered_target_std", "steps",
                                 "samples", "samples_seen", "loss_units", "prediction_units")}
    attempts[-1]["centering"] = {key: stats[key] for key in
                                 ("timestamps", "rows", "min_rows_per_timestamp", "mask_excluded_rows",
                                  "raw_std", "centered_std", "max_abs_centered_timestamp_mean")}
    write_json(root / "attempts.json", attempts)


def fit_checks(plan, fit_root):
    rows = []
    for window in plan["windows"]:
        name = window["name"]
        expected = plan["label_centering"]["per_window"][name]
        for seed in FULL_SEEDS:
            fitting = fit_dir(fit_root, name, seed)
            training = read_json(fitting / "training.json")
            stats = training["target_centering_stats"]
            identity = read_json(fitting / "training_identity.json")
            frozen = {**window["arms"][CANDIDATE]["frozen_identity"], "dates": window["arms"][CANDIDATE]["train_dates"]}
            if {key: identity.get(key) for key in frozen} != frozen:
                raise ValueError(f"{name} seed {seed}: centered fit used different training keys or labels")
            rows.append({"window": name, "seed": seed, "steps": training["steps"],
                         "samples": training["samples"], "target_scale": training["target_scale"],
                         "raw_target_std": training["raw_target_std"],
                         "centered_target_std": training["centered_target_std"],
                         "prediction_units": training["prediction_units"],
                         "timestamps": stats["timestamps"], "rows": stats["rows"],
                         "mask_excluded_rows": stats["mask_excluded_rows"],
                         "max_abs_centered_timestamp_mean": stats["max_abs_centered_timestamp_mean"],
                         "target_scale_matches_plan_raw_std": bool(np.isclose(
                             training["target_scale"], expected["raw_std"], atol=1e-14, rtol=1e-10))})
    return [check("candidate_runs_all_six_declared_fits", len(rows) == 6, rows=rows),
            check("centered_fits_complete_1500_steps", all(row["steps"] == STEPS for row in rows), rows=rows),
            check("centered_fits_cover_every_full_timestamp",
                  all(row["timestamps"] == 80 and row["rows"] == row["samples"] and row["mask_excluded_rows"] == 0
                      for row in rows), rows=rows),
            check("centered_labels_are_zero_mean_within_timestamp",
                  all(row["max_abs_centered_timestamp_mean"] < 1e-12 for row in rows), rows=rows),
            check("target_std_is_raw_label_std_not_centered",
                  all(row["target_scale_matches_plan_raw_std"] and row["centered_target_std"] < row["raw_target_std"]
                      for row in rows), rows=rows),
            check("candidate_predictions_are_relative_scores",
                  all(row["prediction_units"] == "relative_return_score" for row in rows), rows=rows)]


def centered_error(predictions, targets, evaluation):
    """Diagnostic MSE against per-timestamp-centered official labels; never a gate."""
    scored = predictions.merge(targets[KEYS], on=KEYS, validate="one_to_one")
    aligned = scored.merge(targets, on=KEYS, validate="one_to_one")
    # Center in float64: float32 subtraction at the label scale can collapse
    # near-ties and change ranks, which this diagnostic must not do.
    aligned["target"] = aligned["target"].astype(np.float64)
    aligned["target"] = aligned["target"] - aligned.groupby("date")["target"].transform("mean")
    return evaluate(scored, aligned, evaluation, return_sections=True, undefined_constant_portfolios=True)


def error_report(plan, fit_root, paths):
    report = {}
    for window in plan["windows"]:
        name = window["name"]
        targets = pd.read_parquet(window["targets"]["path"])
        evaluation = evaluation_config(window)
        rows = {}
        for arm, by_seed in paths[name].items():
            rows[arm] = {}
            for seed, path in sorted(by_seed.items()):
                predictions = pd.read_parquet(path)
                centered, daily, _ = centered_error(predictions, targets, evaluation)
                official = evaluate(predictions.merge(targets[KEYS], on=KEYS, validate="one_to_one"), targets,
                                    evaluation, return_sections=True, undefined_constant_portfolios=True)
                official_metrics, official_daily, _ = official
                for column in ("rank_ic", "long_short_return", "turnover"):
                    if not np.allclose(official_daily[column], daily[column], atol=1e-10, rtol=1e-9,
                                       equal_nan=True):
                        raise ValueError(f"{name} {arm} seed {seed}: centered labels changed {column}")
                training = (read_json(fit_dir(fit_root, name, seed) / "training.json")
                            if arm == CANDIDATE else window["arms"][REFERENCE]["per_seed"][str(seed)]["training"])
                rows[arm][str(seed)] = {
                    "official_mse_raw_label": official_metrics["mse"],
                    "diagnostic_mse_centered_label": centered["mse"],
                    "official_mae_raw_label": official_metrics["mae"],
                    "diagnostic_mae_centered_label": centered["mae"],
                    "official_ic_raw_label": official_metrics["ic_mean"],
                    "diagnostic_ic_centered_label": centered["ic_mean"],
                    "training_loss_optimized_units": training.get("loss"),
                    "training_loss_units": (training.get("loss_units") or
                                            ("raw_label_mse" if arm == REFERENCE else None)),
                    "training_centered_loss": (training.get("loss") if arm == CANDIDATE else None),
                    "training_raw_loss": (None if arm == CANDIDATE else training.get("loss")),
                    "target_scale_raw_label_std": training.get("target_scale"),
                    "centered_target_std": training.get("centered_target_std")}
        report[name] = rows
    return report


def score_all(plan, paths, checks):
    """Score every arm and seed on the official labels through one code path, keeping full metrics."""
    scored = {}
    for window in plan["windows"]:
        name = window["name"]
        if file_sha256(window["targets"]["path"]) != window["targets"]["sha256"]:
            raise ValueError("Frozen evaluation targets changed")
        targets = pd.read_parquet(window["targets"]["path"])
        evaluation = evaluation_config(window)
        daily, metrics, population = {}, {}, {}
        for arm, by_seed in paths[name].items():
            for seed, path in sorted(by_seed.items()):
                predictions = pd.read_parquet(path)
                values, frame, sections = score_predictions(predictions, targets, evaluation)
                label = f"{name}_{arm}_s{seed}"
                checks.append(_answer_coverage_check(predictions[KEYS], targets, 0.99,
                                                     label + "_answer_coverage", "instruments"))
                checks.append(check(label + "_complete_correlation_coverage",
                                    complete_correlations(values, 80)))
                checks.append(check(label + "_exact_ten_day_eight_time_grid",
                                    len(sections) == 80 and sections.groupby("date").size().eq(8).all()
                                    and len(frame) == len(window["evaluation_dates"])))
                daily[(arm, seed)] = frame.set_index("date")[list(DELTA_METRICS)]
                metrics[(arm, seed)] = values
                population.setdefault(arm, set()).add(values["sample_keys_sha256"] + values["targets_sha256"])
        if any(len(value) != 1 for value in population.values()):
            raise ValueError(f"{name}: seeds were scored on different keys")
        if len({next(iter(value)) for value in population.values()}) != 1:
            raise ValueError(f"{name}: arms were scored on different keys or answers")
        scored[name] = {"daily": daily, "metrics": metrics, "targets": targets}
    return scored


def run_progress(plan, root, candidate_paths, rec):
    """Seed-11 progress only: archived and printed, never used to stop or decide."""
    checks = []
    paths = {window["name"]: {
        REFERENCE: {11: window["arms"][REFERENCE]["per_seed"]["11"]["predictions"]},
        CANDIDATE: candidate_paths[window["name"]][CANDIDATE]} for window in plan["windows"]}
    scored = score_all(plan, paths, checks)
    paired, intervals, summaries = analysis(plan, scored, CANDIDATE, REFERENCE, PROGRESS_SEEDS)
    paired.to_csv(root / "progress_seed11_paired_daily.csv", index=False)
    progress = {
        "schema_version": 1, "descriptive_only": True, "seeds": list(PROGRESS_SEEDS),
        "note": "Progress display only. The round always completes seeds 29 and 47; this comparison cannot "
                "accept, reject or stop the experiment.",
        "mean_rank_ic_delta": float(paired.rank_ic.mean()),
        "period_mean_deltas": {str(key): float(value) for key, value in
                               paired.groupby("period").rank_ic.mean().items()},
        "worst_day_delta": {row["period"]: row["worst_day_delta"] for row in summaries},
        "gross_mean_delta": float(paired.long_short_return.mean()),
        "turnover_mean_delta": float(paired.turnover.mean()),
        "intervals": intervals, "per_window": summaries,
        "checks": {row["name"]: row["passed"] for row in checks}}
    write_json(root / "progress_seed11.json", progress)
    if not save_checks(root, checks):
        raise ValueError("Seed-11 progress scoring checks failed")
    rec.event("evaluation", "progress_seed11", descriptive_only=True,
              mean_rank_ic_delta=progress["mean_rank_ic_delta"],
              period_mean_deltas=progress["period_mean_deltas"],
              worst_day_delta=progress["worst_day_delta"])
    print("progress seed11 (not a decision)", progress["mean_rank_ic_delta"],
          progress["period_mean_deltas"], flush=True)
    return progress


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
        rec.event("run", "started", fits_planned=6, progress_seeds=list(PROGRESS_SEEDS),
                  full_seeds=list(FULL_SEEDS))
        if file_sha256(plan["registry_path"]) != plan["registry_sha256"]:
            raise ValueError("Shared data-role registration changed")
        for path, checksum in plan["inputs_sha256"].items():
            if file_sha256(path) != checksum:
                raise ValueError(f"Frozen input changed: {path}")
        candidate_paths = {}
        for window in plan["windows"]:
            fit_candidate(plan_path, root, plan, window, 11, rec, attempts, started)
        for window in plan["windows"]:
            name = window["name"]
            inference = root / "inference" / name
            launch(plan_path, inference, "infer", name,
                   paths={str(11): str(fit_dir(root, name, 11) / "model.pt")})
            candidate_paths[name] = {CANDIDATE: {11: str(inference / "seed11_predictions.parquet")}}
        progress = run_progress(plan, root, candidate_paths, rec)
        for window in plan["windows"]:
            for seed in (29, 47):
                fit_candidate(plan_path, root, plan, window, seed, rec, attempts, started)
        for window in plan["windows"]:
            name = window["name"]
            inference = root / "inference_full" / name
            launch(plan_path, inference, "infer", name,
                   paths={str(seed): str(fit_dir(root, name, seed) / "model.pt") for seed in FULL_SEEDS})
            candidate_paths[name][CANDIDATE] = {seed: str(inference / f"seed{seed}_predictions.parquet")
                                                for seed in FULL_SEEDS}
        run_full(plan, root, candidate_paths, progress, rec, attempts)
        rec.event("run", "completed", training_attempts=len(attempts), final_views=0)
        finish(root, "completed", 0, resources={"wall_seconds": time.monotonic() - started})
    except BaseException as exc:
        rec.failure("experiment", exc)
        write_json(root / "decision.json", {"result": "inconclusive", "action": "retain_10_day_reference",
            "reason": str(exc), "training_attempts": len(attempts), "final_views": 0, "model_promoted": False})
        finish(root, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", 1,
               resources={"wall_seconds": time.monotonic() - started}, error={"reason": str(exc)})
        raise


def run_full(plan, root, candidate_paths, progress, rec, attempts, *, fit_root=None, inference_root=None):
    fit_root, inference_root = Path(fit_root or root), Path(inference_root or root)
    checks = fit_checks(plan, fit_root)
    paths = {}
    for window in plan["windows"]:
        name = window["name"]
        paths[name] = {
            REFERENCE: {int(seed): row["predictions"] for seed, row in
                        window["arms"][REFERENCE]["per_seed"].items()},
            CANDIDATE: candidate_paths[name][CANDIDATE]}
    scored = score_all(plan, paths, checks)
    errors = error_report(plan, fit_root, paths)
    costs = {window["name"]: read_json(inference_root / "inference_full" / window["name"] / "inference_cost.json")
             for window in plan["windows"]}
    for window in plan["windows"]:
        name = window["name"]
        earlier = pd.read_parquet(inference_root / "inference" / name / "seed11_predictions.parquet")
        later = pd.read_parquet(paths[name][CANDIDATE][11])
        checks.append(check(f"{name}_separate_inference_processes_agree",
                            len(earlier) == len(later)
                            and np.allclose(earlier.score, later.score, atol=1e-6, rtol=1e-5)))
    paired, intervals, summaries = analysis(plan, scored, CANDIDATE, REFERENCE, FULL_SEEDS)
    paired.to_csv(root / "paired_daily.csv", index=False)
    write_json(root / "paired_intervals.json", intervals)
    exposure = {window["name"]: window["exposure"] for window in plan["windows"]}
    decision = coverage_decision(
        paired, summaries, intervals, plan["policy"], cost_measured=bool(costs), exposure=exposure,
        only_factor=plan["only_factor"],
        interpretation="A non-accept result rejects the claim that full-timestamp label demeaning improves the "
                       "retained raw-label 10-day/1500 reference under the unchanged gates. It does not show that "
                       "every cross-sectional label transform or target scale is uninformative.",
        limitations="Two research windows and three seeds are selection data; local gross metrics exclude fees "
                    "and slippage; the platform time and memory limits remain unverified.")
    decision["action"] = ("retain_centered_label_candidate_for_confirmation"
                          if decision["result"] == "accept" else "retain_10_day_reference")
    decision["stage"] = "full"
    decision["progress_seed11"] = {"descriptive_only": True,
                                   "mean_rank_ic_delta": progress["mean_rank_ic_delta"],
                                   "period_mean_deltas": progress["period_mean_deltas"],
                                   "worst_day_delta": progress["worst_day_delta"]}
    write_json(root / "decision.json", decision)
    write_json(root / "metrics.json", {
        "label_centering": plan["label_centering"],
        "windows": {row["period"]: {"reference_comparison": row, "error_report": errors[row["period"]]}
                    for row in summaries},
        "progress_seed11": decision["progress_seed11"],
        "inference_cost": costs, "exposure": exposure,
        "calibration": plan["calibration"]["noise"], "minimum_mean_gain": plan["policy"]["minimum_mean_gain"],
        "costs": {"per_fit_wall_cap_seconds": plan["budget"]["per_fit_wall_cap_seconds"],
                  "local_peak_rss_cap_mib": plan["budget"]["local_peak_rss_cap_mib"],
                  "fits": attempts}})
    if not save_checks(root, checks):
        raise ValueError("Full scoring checks failed")
    rec.event("evaluation", "decision", result=decision["result"], action=decision["action"],
              mean_rank_ic_delta=decision["mean_rank_ic_delta"],
              period_mean_deltas=decision["period_mean_deltas"], checks=decision["checks"])
    print("decision", decision["result"], decision["action"], decision["mean_rank_ic_delta"],
          decision["period_mean_deltas"], flush=True)


# ------------------------------------------------------------------ recovery


def recover(plan_path, source, output_root=RUNS):
    """Re-score a completed six-fit archive whose diagnostic centering failed; never retrain."""
    plan = read_json(plan_path)
    source = Path(source).resolve()
    manifest = read_json(source / "manifest.json")
    if manifest.get("plan_sha256") != file_sha256(plan_path):
        raise ValueError("Recovery source was produced by a different frozen plan")
    if manifest.get("status") not in ("failed", "completed"):
        raise ValueError("Recovery source is not a finished archive")
    attempts = read_json(source / "attempts.json")
    if len([row for row in attempts if row["status"] == "completed"]) != 6:
        raise ValueError("Recovery requires all six completed fits")
    for window in plan["windows"]:
        name = window["name"]
        for seed in FULL_SEEDS:
            fitting = fit_dir(source, name, seed)
            if not all((fitting / filename).is_file()
                       for filename in ("model.pt", "training.json", "training_identity.json")):
                raise FileNotFoundError(f"Recovery source is missing fit artifacts: {fitting}")
            if not (source / "inference_full" / name / f"seed{seed}_predictions.parquet").is_file():
                raise FileNotFoundError(f"Recovery source is missing predictions for {name} seed {seed}")
        if not (source / "inference" / name / "seed11_predictions.parquet").is_file():
            raise FileNotFoundError(f"Recovery source is missing progress predictions for {name}")
    run = new_run_dir(Path(output_root))
    rec = Recorder(run, EXPERIMENT)
    write_json(run / "manifest.json", {"schema_version": 1, "run_id": run.name, "experiment_id": EXPERIMENT,
        "started_at": utc_now(), "status": "running", "mode": "recover-analysis", "new_training": 0,
        "source_run": str(source), "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "plan_sha256": file_sha256(plan_path), "code": capture_source(run), "environment": environment(),
        "final_views": 0})
    write_json(run / "preregistration.json", plan)
    write_json(run / "attempts.json", attempts)
    try:
        rec.event("run", "started", mode="recover-analysis", source_run=str(source), new_training=0)
        write_json(run / "recovery.json", {
            "source_run": str(source), "source_manifest_sha256": file_sha256(source / "manifest.json"),
            "reason": "The six fits and inferences completed, but the original archive failed while writing the "
                      "diagnostic centered-label report: float32 subtraction collapsed two near-tied labels and "
                      "changed one day's rank IC. The diagnostic now centers in float64. No model is retrained.",
            "reused_attempts": attempts, "new_training": 0})
        candidate_paths = {window["name"]: {CANDIDATE: {
            seed: str(source / "inference_full" / window["name"] / f"seed{seed}_predictions.parquet")
            for seed in FULL_SEEDS}} for window in plan["windows"]}
        progress_paths = {window["name"]: {CANDIDATE: {
            11: str(source / "inference" / window["name"] / "seed11_predictions.parquet")}}
            for window in plan["windows"]}
        progress = run_progress(plan, run, progress_paths, rec)
        run_full(plan, run, candidate_paths, progress, rec, attempts,
                 fit_root=source, inference_root=source)
        rec.event("run", "completed", mode="recover-analysis", source_run=str(source), new_training=0)
        finish(run, "completed", 0, resources={"wall_seconds": 0.0})
    except BaseException as exc:
        rec.failure("recovery", exc)
        write_json(run / "decision.json", {"result": "inconclusive", "action": "retain_10_day_reference",
            "reason": str(exc), "training_attempts": 0, "final_views": 0, "model_promoted": False})
        finish(run, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", 1,
               resources={"wall_seconds": 0.0}, error={"reason": str(exc)})
        raise
    return run


# ------------------------------------------------------------------ research record


def recorded_fits(root):
    rows = []
    if not (Path(root) / "attempts.json").is_file():
        return rows
    for item in read_json(Path(root) / "attempts.json"):
        if item["status"] != "completed":
            continue
        rows.append({"window": item["window"], "seed": item["seed"], "steps": item["steps"],
                     "run": item["path"], "attempt": item.get("attempt"),
                     "resources": read_json(Path(item["path"]) / "manifest.json").get("resources"),
                     "weights_sha256": file_sha256(Path(item["path"]) / "model.pt"),
                     "training": item.get("training"), "centering": item.get("centering")})
    return rows


def conclusion_text(decision, metrics, progress):
    failed = [key for key, passed in decision["checks"].items() if not passed]
    per_window = ", ".join(f"{key}: {value:+.6f}" for key, value in decision["period_mean_deltas"].items())
    mse = []
    for name, rows in metrics["windows"].items():
        arms = rows["error_report"]
        seed = sorted(arms[CANDIDATE])[0]
        mse.append(f"{name}: raw-label MSE {arms[CANDIDATE][seed]['official_mse_raw_label']:.6e}, "
                   f"centered-label MSE {arms[CANDIDATE][seed]['diagnostic_mse_centered_label']:.6e}, "
                   f"training centered loss {arms[CANDIDATE][seed]['training_centered_loss']:.6e} "
                   f"(seed {seed})")
    head = (f"With only the training label changed (raw official return -> official return minus the full-timestamp "
            f"training-pool mean, raw-label target_std kept), the seed-mean paired daily all-eight-time RankIC "
            f"difference against the retained raw-label 10-day/1500 reference was "
            f"{decision['mean_rank_ic_delta']:+.6f} against the unchanged floor of "
            f"{decision['minimum_mean_gain']:.6f}; per window {per_window}. Seed 11 progress was "
            f"{progress['mean_rank_ic_delta']:+.6f} and is not part of the decision.")
    if decision["result"] == "accept":
        tail = (" The candidate clears the unchanged gain, replication, guardrail, interval and cost gates, so the "
                "centered-label 10-day recipe may be registered for a separately frozen confirmation stage. It is "
                "not yet a platform-verified submission.")
    else:
        tail = (f" The candidate fails the predeclared gates {', '.join(failed) or 'unspecified'}, so the raw-label "
                "10-day reference is retained. The three seeds are complete; this is not evidence that every "
                "cross-sectional label transform is uninformative.")
    return head + " Error reporting, kept separate from ranking: " + "; ".join(mse) + "." + tail


def record(plan_path, execution, output_path):
    plan = read_json(plan_path)
    execution = Path(execution).resolve()
    if read_json(execution / "manifest.json")["status"] != "completed":
        raise ValueError("A research record requires a completed execution archive")
    decision = read_json(execution / "decision.json")
    metrics = read_json(execution / "metrics.json")
    progress = read_json(execution / "progress_seed11.json")
    fits = recorded_fits(execution)
    document = {
        "schema_version": 1, "experiment_id": EXPERIMENT, "recorded_at_utc": utc_now(),
        "hypothesis": plan["hypothesis"], "only_factor": plan["only_factor"],
        "recipe_fixed": plan["recipe_fixed"], "label_centering": plan["label_centering"],
        "steps": plan["steps"], "seeds": plan["seeds"], "deployment_seed": plan["deployment_seed"],
        "plan": {"path": str(Path(plan_path).resolve()), "sha256": file_sha256(plan_path)},
        "reference_source": plan["reference_source"],
        "gates": {"policy": plan["policy"], "minimum_mean_gain": plan["policy"]["minimum_mean_gain"],
                  "worst_day_tolerance": plan["policy"]["noise"]["min_daily_long_short_return"],
                  "thresholds_status": "copied unchanged from the frozen 20261017 v4 calibration"},
        "reused_arms": {window["name"]: {seed: {key: row[key] for key in
                                                    ("weights_sha256", "predictions_sha256", "training")}
                                             for seed, row in window["arms"][REFERENCE]["per_seed"].items()}
                        for window in plan["windows"]},
        "execution": {"run_dir": str(execution), "manifest_sha256": file_sha256(execution / "manifest.json"),
                      "new_training_fits": len(fits), "fits": fits,
                      "progress_seed11": progress,
                      "retries": (read_json(execution / "recoveries.json")
                                  if (execution / "recoveries.json").is_file() else None),
                      "recovery": (read_json(execution / "recovery.json")
                                   if (execution / "recovery.json").is_file() else None)},
        "exposure": {window["name"]: window["exposure"] for window in plan["windows"]},
        "results": {"decision": decision, "windows": metrics["windows"],
                    "inference_cost": metrics["inference_cost"],
                    "intervals": read_json(execution / "paired_intervals.json")},
        "conclusion": conclusion_text(decision, metrics, progress),
        "retained_version": ("raw-label 10-day seed11 research reference (unchanged)"
                             if decision["result"] != "accept"
                             else "centered-label 10-day candidate (research stage only)"),
        "candidate_status": ("not promoted" if decision["result"] != "accept"
                             else "promoted to a separately frozen confirmation stage"),
        "model_promoted": False, "final_views": 0, "platform_verified": False,
        "limitations": [
            "Two research windows and three seeds are selection data, not independent confirmation.",
            "Demeaning only is a minimal assumption with a public cross-sectional-processing precedent; this "
            "round does not establish that it is the best label transform or that it works elsewhere.",
            "The target_std remains the raw-label standard deviation by design, so the centered and raw arms "
            "share one scale rather than an optimally rescaled centered target.",
            "Raw-label MSE and centered-label MSE measure different targets; neither replaces the official "
            "competition score.",
            "Confirmation, locked and legacy final ranges were not opened; the platform time and memory limits "
            "remain unverified.",
        ],
        "next_steps": [
            "If rejected, test a different scaling basis or label transform only after a new pre-registration; "
            "do not retune this round after seeing its result.",
            "If accepted, run a separately frozen confirmation on untouched dates before any platform claim.",
        ],
    }
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Research records are append-only; select a new output name")
    write_json(output_path, document)
    return output_path


# ------------------------------------------------------------------ entry point


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "worker", "run", "recover", "record"))
    parser.add_argument("path", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--execution", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(str(prepare(args.output or PLAN, frozen_path=args.frozen or FROZEN,
                          output_root=args.output_root or RUNS)), flush=True)
    elif args.command == "worker":
        worker(args.path.resolve(), args.output.resolve())
    elif args.command == "run":
        root = new_run_dir(Path(args.output_root or RUNS).resolve())
        print(str(root), flush=True)
        execute(args.path.resolve(), root)
    elif args.command == "recover":
        if args.execution is None:
            parser.error("recover requires --execution <six-fit archive>")
        print(str(recover(args.path.resolve(), args.execution, args.output_root or RUNS)), flush=True)
    else:
        if args.execution is None or args.output is None:
            parser.error("record requires --execution <completed run> and --output <record.json>")
        print(str(record(args.path.resolve(), args.execution, args.output)), flush=True)


if __name__ == "__main__":
    main()
