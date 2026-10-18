"""Test fixed ten-trading-day recency loss weights inside the 40-day pool.

The equal-weight 40-day/6000-step arm is already archived and failed only the frozen
worst-day guardrail. This round changes one factor: every training sample's MSE term is
multiplied by ``2 ** (-age / 10)``, where age 0 is the newest training date, and the
weights are normalized to mean one over the selected training pool. The sampler, batch
order, steps, seeds, network, preprocessing and full-40-day ``target_std`` are unchanged.
The equal-weight arm is the direct mechanism control; promotion still compares against
the retained 10-day/1500-step reference with the unchanged gates and guardrails.

All three seeds are mandatory. The seed-11 screen is recorded but never used to stop
early or to claim an improvement.
"""

import argparse
import copy
import difflib
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.checks import check
from jinyiquant.config import RunConfig
from jinyiquant.io import digest, file_sha256, read_json, utc_now, write_json
from jinyiquant.runner import save_checks
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir
from modified_tlob.coverage import DELTA_METRICS, arm_exposure, coverage_decision
from modified_tlob.ensemble import SEEDS, paired_block_intervals
from scripts.run_coverage_step_scaling import analysis, score_windows, screen_decision
from scripts.run_seed_ensemble import claim_training
from scripts.run_training_coverage_arms import (
    TRAINING_IDENTITY_FILES,
    BudgetExceeded,
    arm_declaration,
    fit_worker,
    infer_worker,
    supervise,
    window_scope,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "coverage-date-weighting-20261018"
KEYS = ["date", "instrument"]
WINDOWS = ("research_a", "research_b")
REFERENCE, CONTROL, CANDIDATE = "ref10", "ctrl40", "cand40"
STEPS = {REFERENCE: 1500, CONTROL: 6000, CANDIDATE: 6000}
BATCH = 32
SCREEN_SEEDS, FULL_SEEDS = (11,), tuple(SEEDS)
FROZEN = ROOT / "configs/training_coverage_20261017_v4_frozen.json"
PRIOR_PLAN = ROOT / "configs/coverage_step_scaling_20261018_plan.json"
REFERENCE_RUN = ROOT / "runs/fixed-seed-ensemble-20261017/20261017T064944.073834Z_193b03ead7ff"
SCORING_RUN = ROOT / "runs/fixed-seed-ensemble-20261017/analysis-recovery/20261017T091219.300500Z_7c40b5b5d26d"
CONTROL_SCREEN_RUN = ROOT / "runs/coverage-step-scaling-20261018/20261018T042517.974654Z_b779238cb04a"
CONTROL_FULL_RUN = ROOT / "runs/coverage-step-scaling-20261018/20261018T050004.599534Z_1529c1a26f85"
PLAN = ROOT / "configs/coverage_date_weighting_20261018_plan.json"
RUNS = ROOT / "runs" / EXPERIMENT


def job_recorder(output):
    rec = Recorder(output, EXPERIMENT)
    rec.run_id = output.name + "_" + digest(str(output.resolve()))[:12]
    return rec


def fit_dir(root, name, seed):
    return Path(root) / "fits" / f"{name}_st{STEPS[CANDIDATE]}_s{seed}"


# ------------------------------------------------------------------ plan freeze


def assert_only_weighting_changes(control, candidate, label):
    """The candidate differs from the equal-weight 6000-step arm in loss weighting only."""
    differing = {key for key in set(control) | set(candidate) if control.get(key) != candidate.get(key)}
    if differing - {"experiment_id", "hypothesis", "model"}:
        raise ValueError(f"{label}: unexpected recipe differences {sorted(differing)}")
    model_keys = sorted(key for key in set(control["model"]) | set(candidate["model"])
                        if control["model"].get(key) != candidate["model"].get(key))
    if model_keys != ["time_loss_weighting"]:
        raise ValueError(f"{label}: only time_loss_weighting may change, saw {model_keys}")
    if (control["model"]["time_loss_weighting"], candidate["model"]["time_loss_weighting"]) != (
            "none", "date_decay_10d"):
        raise ValueError(f"{label}: weighting change is not none -> date_decay_10d")
    if (control["model"]["max_steps"] != STEPS[CANDIDATE]
            or candidate["model"]["max_steps"] != STEPS[CANDIDATE]):
        raise ValueError(f"{label}: both arms must run {STEPS[CANDIDATE]} steps")


TREATMENT_FILES = ("modified_tlob/adapter.py", "src/jinyiquant/config.py")


def weighting_source_identity(archives, delta_path):
    """Prove reused fits share the training pipeline; the two treatment files are diffed.

    The archived controls predate the date-weighting code, so their adapter/config hashes
    must differ from this tree. Every other training file must be byte-identical, and the
    exact treatment diff is saved for review instead of being silently tolerated.
    """
    result = {}
    non_treatment = {}
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
    with zipfile.ZipFile(CONTROL_FULL_RUN / "fits/research_a_st6000_s29/source.zip") as archive:
        old_adapter = archive.read("modified_tlob/adapter.py").decode("utf-8")
        old_config = archive.read("src/jinyiquant/config.py").decode("utf-8")
    current = {name: (ROOT / name).read_text(encoding="utf-8") for name in TREATMENT_FILES}
    diff = list(difflib.unified_diff(old_adapter.splitlines(), current["modified_tlob/adapter.py"].splitlines(),
                                     fromfile="archived/modified_tlob/adapter.py",
                                     tofile="current/modified_tlob/adapter.py", lineterm=""))
    diff += list(difflib.unified_diff(old_config.splitlines(), current["src/jinyiquant/config.py"].splitlines(),
                                      fromfile="archived/src/jinyiquant/config.py",
                                      tofile="current/src/jinyiquant/config.py", lineterm=""))
    if "date_decay_10d" not in current["modified_tlob/adapter.py"] or "date_decay_10d" in old_adapter:
        raise ValueError("The treatment diff does not contain the declared date weighting")
    if "date_decay_10d" not in current["src/jinyiquant/config.py"] or "date_decay_10d" in old_config:
        raise ValueError("The treatment diff does not contain the declared configuration literal")
    write_json(delta_path, {"schema_version": 1, "treatment_files": list(TREATMENT_FILES),
                            "archived_adapter_sha256": hashlib.sha256(old_adapter.encode()).hexdigest(),
                            "current_adapter_sha256": file_sha256(ROOT / TREATMENT_FILES[0]),
                            "archived_config_sha256": hashlib.sha256(old_config.encode()).hexdigest(),
                            "current_config_sha256": file_sha256(ROOT / TREATMENT_FILES[1]),
                            "diff_sha256": hashlib.sha256("\n".join(diff).encode()).hexdigest(),
                            "diff": diff})
    return result, non_treatment


def reusable_arm(frozen, name, days, fit_for_seed, predictions_for_seed, *, base_config=None):
    """One archived arm: frozen declaration, config and per-seed artifacts."""
    days = int(days)
    arm = REFERENCE if days == 10 else CONTROL
    declaration = arm_declaration(frozen, name, str(days))
    window = next(w for w in frozen["windows"] if w["name"] == name)
    entry = {"days": days, "steps": STEPS[arm], "samples": declaration["samples"],
             "frozen_identity": {key: declaration[key] for key in ("samples", "keys_sha256", "labels_sha256")},
             "config": copy.deepcopy(base_config or window["arms"][str(days)]["config"]),
             "train_dates": list(window["arms"][str(days)]["train_dates"]),
             "train_end_exclusive": window["arms"][str(days)]["train_end_exclusive"], "per_seed": {}}
    for seed in SEEDS:
        fitting = Path(fit_for_seed(seed))
        identity = read_json(fitting / "training_identity.json")
        expected = {**entry["frozen_identity"], "dates": entry["train_dates"]}
        if {key: identity.get(key) for key in expected} != expected:
            raise ValueError(f"{name} seed {seed} {days}-day arm was fitted on different training data")
        predictions = Path(predictions_for_seed(seed))
        if not predictions.is_file():
            raise FileNotFoundError(predictions)
        archived = read_json(fitting / "config.json")
        differing = {key for key in set(archived) | set(entry["config"])
                     if archived.get(key) != entry["config"].get(key)}
        if differing - {"experiment_id", "hypothesis", "seed"}:
            raise ValueError(f"{name} seed {seed} {days}-day recipe differs from the frozen arm: "
                             f"{sorted(differing)}")
        entry["per_seed"][str(seed)] = {
            "fit_dir": str(fitting), "weights": str(fitting / "model.pt"),
            "weights_sha256": file_sha256(fitting / "model.pt"),
            "predictions": str(predictions), "predictions_sha256": file_sha256(predictions)}
    return entry


def weighting_stats(fit_root):
    """The fitted date weights recorded by the adapter, for archive checks and the record."""
    for line in (Path(fit_root) / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        context = event.get("context", {})
        if event.get("stage") == "training" and event.get("event_type") == "time_loss_weighting_fitted":
            return context
    raise ValueError(f"No loss-weight event in {fit_root}")


def prepare(output_path, *, frozen_path=FROZEN, output_root=RUNS):
    """Freeze the candidate recipe, the reused controls and every gate before training."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Frozen plans are immutable; select a new output name")
    frozen_path = Path(frozen_path).resolve()
    frozen = read_json(frozen_path)
    if file_sha256(frozen["registry_path"]) != frozen["registry_sha256"]:
        raise ValueError("Shared data-role registration changed")
    prior_plan = read_json(PRIOR_PLAN)
    if read_json(CONTROL_FULL_RUN / "manifest.json")["status"] != "completed":
        raise ValueError("The equal-weight 40-day/6000-step control must be a completed archive")
    prior_decision = read_json(CONTROL_FULL_RUN / "decision.json")
    if prior_decision["result"] == "accept":
        raise ValueError("This round only runs when the equal-weight 40-day arm failed its gates")
    run = new_run_dir(Path(output_root))
    rec = Recorder(run, EXPERIMENT)
    write_json(run / "manifest.json", {"schema_version": 1, "run_id": run.name, "experiment_id": EXPERIMENT,
        "started_at": utc_now(), "status": "running", "new_training": 0, "final_views": 0, "mode": "prepare",
        "code": capture_source(run), "environment": environment()})
    try:
        rec.event("run", "started", mode="prepare", candidate_steps=STEPS[CANDIDATE],
                  half_life_trading_days=10)
        limits = prior_plan["measured_budget"]["limits"]
        if (limits["max_rss_mb"], limits["wall_seconds"]) != (9472.0, 1500.0):
            raise ValueError("The reused 6000-step budget is not the frozen one")
        windows, inventory, weight_summary = [], {}, {}
        for name in WINDOWS:
            scope = window_scope(frozen, name, "40")
            reference = reusable_arm(
                frozen, name, 10,
                lambda seed: REFERENCE_RUN / f"fits/{name}_s{seed}",
                lambda seed: REFERENCE_RUN / f"inference/{name}/seed{seed}_predictions.parquet")
            prior_window = next(w for w in prior_plan["windows"] if w["name"] == name)
            control = reusable_arm(
                frozen, name, 40,
                lambda seed: (CONTROL_SCREEN_RUN if seed == SCREEN_SEEDS[0] else CONTROL_FULL_RUN)
                / f"fits/{name}_st{STEPS[CONTROL]}_s{seed}",
                lambda seed: CONTROL_FULL_RUN / f"inference_full/{name}/seed{seed}_predictions.parquet",
                base_config=prior_window["arms"][CANDIDATE]["config"])
            candidate = {"days": 40, "steps": STEPS[CANDIDATE], "samples": control["samples"],
                         "frozen_identity": dict(control["frozen_identity"]),
                         "train_dates": list(control["train_dates"]),
                         "train_end_exclusive": control["train_end_exclusive"],
                         "config": copy.deepcopy(control["config"])}
            candidate["config"]["experiment_id"] = EXPERIMENT
            candidate["config"]["hypothesis"] = (
                "Fixed ten-trading-day recency weights 2**(-age/10) on the 40-day pool improve the "
                "out-of-sample rank IC enough to clear the retained 10-day reference without degrading "
                "worst-day long-short, turnover or gross return.")
            candidate["config"]["model"]["time_loss_weighting"] = "date_decay_10d"
            assert_only_weighting_changes(control["config"], candidate["config"], name)
            targets = SCORING_RUN / f"{name}_targets.parquet"
            if not targets.is_file():
                raise FileNotFoundError(targets)
            inventory[str(targets)] = file_sha256(targets)
            for arm in (reference, control):
                for row in arm["per_seed"].values():
                    inventory[row["predictions"]] = row["predictions_sha256"]
                    inventory[row["weights"]] = row["weights_sha256"]
            weight_summary[name] = {
                "train_dates": len(candidate["train_dates"]),
                "newest_date": candidate["train_dates"][-1],
                "oldest_date": candidate["train_dates"][0],
                "formula": "2 ** (-age_trading_days / 10), age 0 = newest training date",
                "normalization": "sample mean one over the selected 40-day training pool",
                "recent_10_day_weight_share_expected": 0.533,
                "equal_weight_recent_10_day_share": 0.25}
            windows.append({
                "name": name, "train_start": scope["train_start"],
                "train_end_exclusive": scope["train_end_exclusive"],
                "evaluation_start": scope["evaluation_start"],
                "evaluation_end_exclusive": scope["evaluation_end_exclusive"],
                "evaluation_dates": scope["evaluation_dates"],
                "arms": {REFERENCE: reference, CONTROL: control, CANDIDATE: candidate},
                "exposure": {
                    "batch_size": BATCH,
                    "arms": {arm: arm_exposure(spec["samples"], steps=spec["steps"], batch_size=BATCH)
                             for arm, spec in {REFERENCE: reference, CONTROL: control,
                                               CANDIDATE: candidate}.items()},
                    "interpretation": (
                        "The candidate and the equal-weight control both run 6000 x 32 = 192,000 sample "
                        "exposures (0.60 epochs of the 40-day pool). Only the per-sample loss weights differ. "
                        "The retained 10-day reference runs 1500 x 32 = 48,000 exposures.")},
                "targets": {"path": str(targets), "sha256": inventory[str(targets)]},
                "fit_once_before_evaluation": True})
        identity_rows, non_treatment = weighting_source_identity(
            [REFERENCE_RUN / f"fits/{name}_s{seed}/source.zip" for name in WINDOWS for seed in SEEDS]
            + [CONTROL_SCREEN_RUN / f"fits/{name}_st6000_s11/source.zip" for name in WINDOWS]
            + [CONTROL_FULL_RUN / f"fits/{name}_st6000_s{seed}/source.zip"
               for name in WINDOWS for seed in (29, 47)],
            run / "source_delta.json")
        inventory[str(run / "source_delta.json")] = file_sha256(run / "source_delta.json")
        policy = copy.deepcopy(frozen["policy"])
        plan = {
            "schema_version": 1, "experiment_id": EXPERIMENT, "created_at_utc": utc_now(),
            "hypothesis": "The 40-day pool failed its worst-day guardrail because old dates diluted the "
                          "recent regime. Fixed exponential recency weights with a ten-trading-day half-life "
                          "should keep the 40-day sample but restore the recent-date emphasis, clearing the "
                          "unchanged promotion gates against the 10-day/1500 reference.",
            "only_factor": "per-sample MSE weight none -> 2**(-age/10), age 0 = newest training date, "
                           "normalized to mean one on the full selected 40-day pool; sampler, batch order, "
                           "steps, seeds, target_std, network and preprocessing fixed",
            "source_frozen": {"path": str(frozen_path), "sha256": file_sha256(frozen_path)},
            "prior_round": {"equal_weight_execution": str(CONTROL_FULL_RUN),
                            "decision_sha256": file_sha256(CONTROL_FULL_RUN / "decision.json"),
                            "result": prior_decision["result"], "action": prior_decision["action"],
                            "mean_rank_ic_delta": prior_decision["mean_rank_ic_delta"],
                            "period_mean_deltas": prior_decision["period_mean_deltas"],
                            "failed_checks": [key for key, passed in prior_decision["checks"].items()
                                              if not passed]},
            "recipe_fixed": {"raw12": True, "sample_times": 8, "lookback": 128, "loss": "MSE",
                             "target_scaling": "train_std", "target_std_fit": "all 40 selected training rows",
                             "learning_rate": 1e-4, "batch_size": BATCH, "smoothing": "none",
                             "sampler": "torch.randperm(seed) unchanged", "steps": STEPS[CANDIDATE],
                             "date_weighting": "2**(-age/10), mean one, only loss weights"},
            "steps": {"reference": STEPS[REFERENCE], "control": STEPS[CONTROL], "candidate": STEPS[CANDIDATE]},
            "seeds": {"screen": list(SCREEN_SEEDS), "full": list(FULL_SEEDS)}, "deployment_seed": 11,
            "windows": windows, "training_source_identity": identity_rows,
            "non_treatment_source": non_treatment,
            "source_delta": {"path": str(run / "source_delta.json"),
                             "sha256": inventory[str(run / "source_delta.json")],
                             "note": "Only adapter.py and config.py differ from the equal-weight control "
                                     "snapshot; every other training file is byte-identical."},
            "weighting": weight_summary,
            "policy": policy, "calibration": frozen["calibration"], "statistics": policy["statistics"],
            "screen": {"seeds": list(SCREEN_SEEDS),
                       "gates": {"rank_ic_delta_at_least": 0.0,
                                 "worst_day_delta_at_least": -policy["noise"]["min_daily_long_short_return"]},
                       "rule": "Seed 11 is screened and reported first, but the round always completes seeds "
                               "29 and 47: the registered decision is the three-seed comparison, and seed 11 "
                               "alone can neither accept nor reject an improvement.",
                       "mandatory_full_replay": True},
            "budget": {"local_peak_rss_cap_mib": limits["max_rss_mb"],
                       "per_fit_wall_cap_seconds": limits["wall_seconds"], "fit_attempts_max": 8,
                       "fit_stage_wall_cap_seconds": 8 * limits["wall_seconds"] + 2 * 1800,
                       "inference_chunk_days": 5, "inference_windows": len(WINDOWS)},
            "measured_budget": {"limits": limits, "pilots": prior_plan["measured_budget"]["pilots"],
                                "pilot_source": prior_plan["measured_budget"]["pilot_source"],
                                "pilot_sha256": prior_plan["measured_budget"]["pilot_sha256"]},
            "registry_path": frozen["registry_path"], "registry_sha256": frozen["registry_sha256"],
            "inputs_sha256": inventory, "final_views": 0,
            "access_policy": "research_a/research_b are already-used research data; confirmation, locked and "
                             "legacy final ranges stay closed this round.",
            "notes": "Six mandatory 6000-step fits (seed 11 first for the recorded screen, then 29/47). The "
                     "twelve reused fits are never refitted. The equal-weight control isolates the weighting "
                     "effect; promotion is still decided against the 10-day reference.",
        }
        checks = [
            check("reused_arms_have_three_seeds_each",
                  all(len(window["arms"][arm]["per_seed"]) == len(SEEDS)
                      for window in windows for arm in (REFERENCE, CONTROL))),
            check("candidate_differs_only_in_weighting",
                  all(assert_only_weighting_changes(
                      window["arms"][CONTROL]["config"], window["arms"][CANDIDATE]["config"],
                      window["name"]) is None for window in windows),
                  control_steps=STEPS[CONTROL], candidate_steps=STEPS[CANDIDATE]),
            check("prior_round_failed_its_gates", prior_decision["result"] != "accept",
                  prior_result=prior_decision["result"]),
            check("recent_ten_day_expected_share", all(
                row["recent_10_day_weight_share_expected"] == 0.533 for row in weight_summary.values())),
        ]
        if not save_checks(run, checks):
            raise ValueError("Plan freeze checks failed")
        write_json(output_path, plan)
        write_json(run / "plan.json", plan)
        rec.event("prepare", "completed", plan=str(output_path), plan_sha256=file_sha256(output_path),
                  mandatory_fits=2 * len(FULL_SEEDS), per_fit_wall_cap_seconds=limits["wall_seconds"])
        finish(run, "completed", 0, resources={"wall_seconds": 0.0})
    except BaseException as exc:
        rec.failure("prepare", exc)
        finish(run, "failed", 1, error={"reason": str(exc)})
        raise
    return run


# ------------------------------------------------------------------ worker supervision


def launch(plan_path, output, kind, window_name, *, seed=None, paths=None, cap=None):
    """Run one worker under an independent watchdog; a cap is never silently extended."""
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
    command = [sys.executable, "-m", "scripts.run_coverage_date_weighting", "worker",
               str(Path(plan_path).resolve()), "--output", str(output)]
    try:
        result = supervise(command, output, cap_rss_mb=caps["rss_mb"], cap_seconds=caps["wall_seconds"],
                           rec=rec, seed=seed or SEEDS[0])
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
    stem = f"{name}_st{STEPS[CANDIDATE]}_s{seed}"
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
            "reason": "an earlier attempt at this arm and seed was interrupted before it saved a model, so it "
                      "left no result to select on"})
        rec.event("training", "retry", window=name, seed=seed, attempt=attempt,
                  interrupted_archives=[row["archive"] for row in prior])
    attempts.append({"window": name, "arm": CANDIDATE, "seed": seed, "steps": STEPS[CANDIDATE],
                     "attempt": attempt, "path": str(path), "status": "started",
                     "prior_interrupted_archives": [row["archive"] for row in prior]})
    write_json(root / "attempts.json", attempts)
    remaining = plan["budget"]["fit_stage_wall_cap_seconds"] - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("Fit stage budget exhausted")
    launch(plan_path, path, "fit", name, seed=seed,
           cap={"wall_seconds": min(plan["budget"]["per_fit_wall_cap_seconds"], remaining),
                "rss_mb": plan["budget"]["local_peak_rss_cap_mib"]})
    stats = weighting_stats(path)
    if stats.get("method") != "date_decay_10d" or stats["days"] != 40:
        raise ValueError("The fit did not record the declared 40-day date weights")
    if not np.isclose(stats["mean_weight"], 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("Date weights are not normalized to mean one")
    attempts[-1]["status"] = "completed"
    attempts[-1]["weight_stats"] = {key: stats[key] for key in
                                    ("days", "oldest_age_trading_days", "mean_weight", "recent_weight_share")}
    write_json(root / "attempts.json", attempts)


def frozen_arm_paths(window, arm):
    return {int(seed): row["predictions"] for seed, row in window["arms"][arm]["per_seed"].items()}


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
        rec.event("run", "started", candidate_steps=STEPS[CANDIDATE], screen_seeds=list(SCREEN_SEEDS),
                  full_seeds=list(FULL_SEEDS))
        if file_sha256(plan["registry_path"]) != plan["registry_sha256"]:
            raise ValueError("Shared data-role registration changed")
        for path, checksum in plan["inputs_sha256"].items():
            if file_sha256(path) != checksum:
                raise ValueError(f"Frozen input changed: {path}")
        candidate_paths = {}
        for window in plan["windows"]:
            name = window["name"]
            fit_candidate(plan_path, root, plan, window, 11, rec, attempts, started)
            inference = root / "inference" / name
            launch(plan_path, inference, "infer", name,
                   paths={str(11): str(fit_dir(root, name, 11) / "model.pt")})
            candidate_paths[name] = {CANDIDATE: {11: str(inference / "seed11_predictions.parquet")}}
        screen = run_screen(plan, root, candidate_paths, rec)
        write_json(root / "screen.json", screen)
        rec.event("evaluation", "screen", result=screen["decision"]["result"],
                  mean_rank_ic_delta=screen["decision"]["mean_rank_ic_delta"],
                  period_mean_deltas=screen["decision"]["period_mean_deltas"],
                  mandatory_full_replay=True)
        for window in plan["windows"]:
            name = window["name"]
            for seed in (29, 47):
                fit_candidate(plan_path, root, plan, window, seed, rec, attempts, started)
        for window in plan["windows"]:
            name = window["name"]
            inference = root / "inference_full" / name
            launch(plan_path, inference, "infer", name,
                   paths={str(seed): str(fit_dir(root, name, seed) / "model.pt") for seed in FULL_SEEDS})
            candidate_paths[name][CANDIDATE] = {seed: str(inference / f"seed{seed}_predictions.parquet")
                                                for seed in FULL_SEEDS}
        run_full(plan, root, candidate_paths, screen, rec, attempts)
        rec.event("run", "completed", training_attempts=len(attempts), final_views=0)
        finish(root, "completed", 0, resources={"wall_seconds": time.monotonic() - started})
    except BaseException as exc:
        rec.failure("experiment", exc)
        write_json(root / "decision.json", {"result": "inconclusive", "action": "retain_10_day_reference",
            "reason": str(exc), "training_attempts": len(attempts), "final_views": 0, "model_promoted": False})
        finish(root, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", 1,
               resources={"wall_seconds": time.monotonic() - started}, error={"reason": str(exc)})
        raise


def run_screen(plan, root, candidate_paths, rec):
    checks = []
    paths = {window["name"]: {REFERENCE: {11: window["arms"][REFERENCE]["per_seed"]["11"]["predictions"]},
                              CANDIDATE: candidate_paths[window["name"]][CANDIDATE]}
             for window in plan["windows"]}
    scored = score_windows(plan, paths, checks)
    paired, intervals, summaries = analysis(plan, scored, CANDIDATE, REFERENCE, SCREEN_SEEDS)
    paired.to_csv(root / "screen_paired_daily.csv", index=False)
    decision = screen_decision(paired, summaries, intervals, plan)
    decision["mandatory_full_replay"] = True
    decision["next"] = "continue to seeds 29/47 regardless of this screen; only the three-seed decision counts"
    if not save_checks(root, checks):
        raise ValueError("Screen scoring checks failed")
    print("screen", decision["result"], decision["mean_rank_ic_delta"],
          decision["period_mean_deltas"], flush=True)
    return {"decision": decision, "per_window": summaries, "paired_intervals": intervals,
            "steps": STEPS, "exposure": {window["name"]: window["exposure"] for window in plan["windows"]}}


def pair_up(scored_window, name, high, low, seeds):
    """Paired daily deltas (high minus low) for one window; seeds are averaged first."""
    means = {arm: pd.concat([scored_window["daily"][(arm, seed)] for seed in seeds]).groupby(level=0).mean()
             for arm in (high, low)}
    if not means[high].index.equals(means[low].index):
        raise ValueError("Arms were scored on different evaluation days")
    delta = (means[high] - means[low]).reset_index()
    delta["period"] = name
    summary = {"period": name, "days": len(delta),
               "arm_window_means": {arm: {column: float(means[arm][column].mean()) for column in DELTA_METRICS}
                                    for arm in (high, low)},
               "worst_day_delta": float(means[high].long_short_return.min()
                                        - means[low].long_short_return.min()),
               "min_day_delta": float(delta.long_short_return.min()),
               "per_seed_mean_delta": {
                   str(seed): {column: float(scored_window["daily"][(high, seed)][column].mean()
                                             - scored_window["daily"][(low, seed)][column].mean())
                               for column in DELTA_METRICS} for seed in seeds}}
    return delta, summary


def pair_analysis(plan, scored, high, low, seeds):
    frames, summaries = [], []
    for window in plan["windows"]:
        frame, summary = pair_up(scored[window["name"]], window["name"], high, low, seeds)
        frames.append(frame)
        summaries.append(summary)
    paired = pd.concat(frames, ignore_index=True)
    statistics = plan["statistics"]
    intervals = [paired_block_intervals(paired, block_days=block, samples=statistics["bootstrap_samples"],
                                        seed=statistics["bootstrap_seed"])
                 for block in statistics["block_days"]]
    return paired, intervals, summaries


def date_weight_checks(fit_root, plan):
    """The candidate must be the declared 40-day weighted refit with unchanged target scaling."""
    rows = []
    for window in plan["windows"]:
        for seed in FULL_SEEDS:
            fitting = fit_dir(fit_root, window["name"], seed)
            stats = weighting_stats(fitting)
            training = read_json(fitting / "training.json")
            rows.append({
                "window": window["name"], "seed": seed, "steps": training["steps"],
                "samples": training["samples"], "days": stats["days"],
                "mean_weight": stats["mean_weight"], "recent_weight_share": stats["recent_weight_share"],
                "target_scale": training["target_scale"], "training_std": training["training_std"],
                "target_scale_matches_full_pool": bool(np.isclose(
                    training["target_scale"], training["training_std"], atol=1e-14, rtol=1e-10))})
    return [check("candidate_runs_all_six_declared_fits", len(rows) == 6, rows=rows),
            check("candidate_uses_40_day_mean_one_weights",
                  all(row["days"] == 40 and np.isclose(row["mean_weight"], 1.0, atol=1e-6) for row in rows),
                  rows=rows),
            check("candidate_recent_ten_day_share_matches_formula",
                  all(abs(row["recent_weight_share"] - 0.533) < 1e-3 for row in rows), rows=rows),
            check("candidate_target_std_is_full_40_day_fit",
                  all(row["target_scale_matches_full_pool"] for row in rows), rows=rows)]


def run_full(plan, root, candidate_paths, screen, rec, attempts, *, fit_root=None, inference_root=None):
    fit_root = Path(fit_root or root)
    inference_root = Path(inference_root or root)
    checks = date_weight_checks(fit_root, plan)
    paths = {}
    for window in plan["windows"]:
        name = window["name"]
        paths[name] = {REFERENCE: frozen_arm_paths(window, REFERENCE),
                       CONTROL: frozen_arm_paths(window, CONTROL),
                       CANDIDATE: candidate_paths[name][CANDIDATE]}
    scored = score_windows(plan, paths, checks)
    costs = {}
    for window in plan["windows"]:
        name = window["name"]
        made_earlier = pd.read_parquet(inference_root / "inference" / name / "seed11_predictions.parquet")
        made_now = pd.read_parquet(paths[name][CANDIDATE][11])
        checks.append(check(f"{name}_separate_inference_processes_agree",
                            len(made_earlier) == len(made_now)
                            and np.allclose(made_earlier.score, made_now.score, atol=1e-6, rtol=1e-5)))
        costs[name] = read_json(inference_root / "inference_full" / name / "inference_cost.json")
    paired, intervals, summaries = pair_analysis(plan, scored, CANDIDATE, REFERENCE, FULL_SEEDS)
    paired.to_csv(root / "paired_daily.csv", index=False)
    write_json(root / "paired_intervals.json", intervals)
    exposure = {window["name"]: window["exposure"] for window in plan["windows"]}
    decision = coverage_decision(
        paired, summaries, intervals, plan["policy"], cost_measured=bool(costs), exposure=exposure,
        only_factor=plan["only_factor"],
        interpretation="A non-accept result rejects the claim that fixed ten-day recency weighting makes the "
                       "40-day pool beat the retained 10-day reference under the unchanged gates. It does not "
                       "show that recency weighting is useless at every half-life or step count.",
        limitations="Two research windows and three seeds are selection data; local gross metrics exclude fees "
                    "and slippage; the platform time and memory limits remain unverified.")
    decision["action"] = ("retain_date_weighted_40_day_candidate_for_next_stage"
                          if decision["result"] == "accept" else "retain_10_day_reference")
    decision["stage"] = "full"
    decision["screen"] = screen["decision"]
    decision["weighting"] = {"formula": "2**(-age/10)", "half_life_trading_days": 10,
                             "normalization": "mean one over selected 40-day pool",
                             "recent_10_day_weight_share": 0.533}
    weight_paired, weight_intervals, weight_summaries = pair_analysis(
        plan, scored, CANDIDATE, CONTROL, FULL_SEEDS)
    weight_paired.to_csv(root / "weight_effect_daily.csv", index=False)
    weight_effect = {
        "schema_version": 1, "comparison": "cand40 minus ctrl40",
        "scope": "descriptive_mechanism_evidence_no_gates",
        "mean_rank_ic_delta": float(weight_paired.rank_ic.mean()),
        "period_mean_deltas": {str(key): float(value) for key, value in
                               weight_paired.groupby("period").rank_ic.mean().items()},
        "gross_mean_delta": float(weight_paired.long_short_return.mean()),
        "turnover_mean_delta": float(weight_paired.turnover.mean()),
        "per_window": weight_summaries, "intervals": weight_intervals,
        "note": "The control is the archived equal-weight 40-day/6000-step arm. This contrast isolates the "
                "loss weighting at identical pool, steps, seeds and batch order; it does not decide promotion."}
    write_json(root / "weight_effect.json", weight_effect)
    write_json(root / "decision.json", decision)
    write_json(root / "metrics.json", {
        "windows": {row["period"]: {"reference_comparison": row} for row in summaries},
        "weight_effect": weight_effect, "inference_cost": costs, "exposure": exposure,
        "calibration": plan["calibration"]["noise"], "minimum_mean_gain": plan["policy"]["minimum_mean_gain"],
        "costs": {"per_fit_wall_cap_seconds": plan["budget"]["per_fit_wall_cap_seconds"],
                  "local_peak_rss_cap_mib": plan["budget"]["local_peak_rss_cap_mib"],
                  "fits": attempts}})
    if not save_checks(root, checks):
        raise ValueError("Full scoring checks failed")
    rec.event("evaluation", "decision", result=decision["result"], action=decision["action"],
              mean_rank_ic_delta=decision["mean_rank_ic_delta"],
              period_mean_deltas=decision["period_mean_deltas"], checks=decision["checks"])
    rec.event("evaluation", "weight_effect", mean_rank_ic_delta=weight_effect["mean_rank_ic_delta"],
              period_mean_deltas=weight_effect["period_mean_deltas"])
    print("decision", decision["result"], decision["action"], decision["mean_rank_ic_delta"],
          decision["period_mean_deltas"], flush=True)
    print("weight_effect", weight_effect["mean_rank_ic_delta"],
          weight_effect["period_mean_deltas"], flush=True)


# ------------------------------------------------------------------ recovery


def recover(plan_path, source, output_root=RUNS):
    """Re-score a completed six-fit archive whose final check write failed; never retrain."""
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
        for seed in FULL_SEEDS:
            fitting = fit_dir(source, window["name"], seed)
            if not all((fitting / name).is_file() for name in ("model.pt", "training.json", "training_identity.json")):
                raise FileNotFoundError(f"Recovery source is missing fit artifacts: {fitting}")
            prediction = source / "inference_full" / window["name"] / f"seed{seed}_predictions.parquet"
            if not prediction.is_file():
                raise FileNotFoundError(prediction)
    run = new_run_dir(Path(output_root))
    rec = Recorder(run, EXPERIMENT)
    write_json(run / "manifest.json", {"schema_version": 1, "run_id": run.name, "experiment_id": EXPERIMENT,
        "started_at": utc_now(), "status": "running", "mode": "recover-analysis", "new_training": 0,
        "source_run": str(source), "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "plan_sha256": file_sha256(plan_path), "code": capture_source(run), "environment": environment(),
        "final_views": 0})
    write_json(run / "preregistration.json", plan)
    try:
        rec.event("run", "started", mode="recover-analysis", source_run=str(source), new_training=0)
        write_json(run / "recovery.json", {
            "source_run": str(source), "source_manifest_sha256": file_sha256(source / "manifest.json"),
            "reason": "the original archive completed all six fits and inferences but failed while serializing a "
                      "NumPy boolean in checks.json",
            "reused_attempts": attempts, "new_training": 0})
        screen = read_json(source / "screen.json")
        write_json(run / "screen.json", screen)
        candidate_paths = {window["name"]: {CANDIDATE: {
            seed: str(source / "inference_full" / window["name"] / f"seed{seed}_predictions.parquet")
            for seed in FULL_SEEDS}} for window in plan["windows"]}
        run_full(plan, run, candidate_paths, screen, rec, attempts, fit_root=source, inference_root=source)
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
    root = Path(root)
    attempts = read_json(root / "attempts.json") if (root / "attempts.json").is_file() else (
        read_json(root / "recovery.json").get("reused_attempts", []) if (root / "recovery.json").is_file() else [])
    fits = []
    for item in attempts:
        if item["status"] != "completed":
            continue
        fits.append({"window": item["window"], "seed": item["seed"], "steps": item["steps"],
                     "run": item["path"], "attempt": item.get("attempt"),
                     "resources": read_json(Path(item["path"]) / "manifest.json").get("resources"),
                     "weights_sha256": file_sha256(Path(item["path"]) / "model.pt"),
                     "weight_stats": item.get("weight_stats")})
    return fits


def conclusion_text(decision, weight):
    failed = [key for key, passed in decision["checks"].items() if not passed]
    per_window = ", ".join(f"{key}: {value:+.6f}" for key, value in decision["period_mean_deltas"].items())
    head = (f"With only the loss weights changed (none -> 2**(-age/10), mean one), the seed-mean paired daily "
            f"all-eight-time RankIC difference against the retained 10-day/1500 reference was "
            f"{decision['mean_rank_ic_delta']:+.6f} against the unchanged floor of "
            f"{decision['minimum_mean_gain']:.6f}; per window {per_window}. The direct contrast against the "
            f"equal-weight 40-day/6000 arm was {weight['mean_rank_ic_delta']:+.6f} "
            f"({', '.join(f'{k}: {v:+.6f}' for k, v in weight['period_mean_deltas'].items())}).")
    if decision["result"] == "accept":
        return (head + " The candidate clears the unchanged gain, replication, guardrail, interval and cost "
                       "gates, so the date-weighted 40-day plan may be registered for a separately frozen "
                       "confirmation stage. It is not yet a platform-verified submission.")
    return (head + f" The candidate fails the predeclared gates {', '.join(failed) or 'unspecified'}, so the "
                   "10-day reference is retained. The three seed results are complete; this is not evidence "
                   "that every recency half-life or step count is uninformative.")


def record(plan_path, execution, output_path):
    plan = read_json(plan_path)
    execution = Path(execution).resolve()
    if read_json(execution / "manifest.json")["status"] != "completed":
        raise ValueError("A research record requires a completed execution archive")
    decision = read_json(execution / "decision.json")
    screen = read_json(execution / "screen.json")
    weight = read_json(execution / "weight_effect.json")
    fits = recorded_fits(execution)
    document = {
        "schema_version": 1, "experiment_id": EXPERIMENT, "recorded_at_utc": utc_now(),
        "hypothesis": plan["hypothesis"], "only_factor": plan["only_factor"],
        "recipe_fixed": plan["recipe_fixed"], "weighting": plan["weighting"],
        "steps": plan["steps"], "seeds": plan["seeds"], "deployment_seed": plan["deployment_seed"],
        "plan": {"path": str(Path(plan_path).resolve()), "sha256": file_sha256(plan_path)},
        "prior_round": plan["prior_round"],
        "budget_freeze": {"limits": plan["measured_budget"]["limits"],
                          "pilots": plan["measured_budget"]["pilots"],
                          "pilot_source": plan["measured_budget"]["pilot_source"]},
        "gates": {"screen": plan["screen"]["gates"], "decision": plan["policy"],
                  "minimum_mean_gain": plan["policy"]["minimum_mean_gain"]},
        "reused_arms": {window["name"]: {arm: {seed: {key: row[key] for key in
                                                      ("weights_sha256", "predictions_sha256")}
                                               for seed, row in window["arms"][arm]["per_seed"].items()}
                                         for arm in (REFERENCE, CONTROL)}
                        for window in plan["windows"]},
        "execution": {"run_dir": str(execution), "manifest_sha256": file_sha256(execution / "manifest.json"),
                      "new_training_fits": len(fits), "fits": fits,
                      "screen": screen["decision"], "screen_per_window": screen.get("per_window"),
                      "recovery": (read_json(execution / "recovery.json")
                                   if (execution / "recovery.json").is_file() else None)},
        "exposure": {window["name"]: window["exposure"] for window in plan["windows"]},
        "results": {"decision": decision, "weight_effect": weight,
                    "intervals": read_json(execution / "paired_intervals.json"),
                    "per_window": read_json(execution / "metrics.json").get("windows")},
        "conclusion": conclusion_text(decision, weight),
        "retained_version": "10-day seed11 research reference (unchanged)" if decision["result"] != "accept"
                            else "date-weighted 40-day candidate (research stage only)",
        "candidate_status": "not promoted" if decision["result"] != "accept"
                            else "promoted to a separately frozen confirmation stage",
        "model_promoted": False, "final_views": 0, "platform_verified": False,
        "limitations": [
            "Two research windows and three seeds are selection data, not independent confirmation.",
            "The ten-day half-life is this project's predeclared choice; no other half-life was searched.",
            "The equal-weight control isolates the weighting effect but was trained in the prior round; its "
            "source snapshot differs only in the loss-weighting code path.",
            "6000 steps is 0.60 epochs of the 40-day pool; heavier schedules are outside this round.",
            "Local gross metrics exclude fees and slippage; platform time and memory limits remain unverified.",
        ],
        "next_steps": [
            "If accepted, run a separately frozen confirmation on untouched dates before any platform claim.",
            "If rejected, report the weighting effect against the equal-weight control before proposing another "
            "half-life; this round deliberately tested only 10 days.",
        ],
    }
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Research records are append-only; select a new output name")
    write_json(output_path, document)
    return output_path


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
