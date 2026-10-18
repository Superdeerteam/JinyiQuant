"""Raise only the 40-day arm's optimizer steps from 1500 to 6000, then screen it.

The archived coverage round held steps fixed, so the 40-day pool was traversed for about
0.15 epochs; its pooled gain came entirely from research_a while research_b degraded and
the worst-day guardrail failed. This round changes one number: the 40-day arm's
``max_steps`` 1500 -> 6000 (192,000 sample exposures = 0.60 pool epochs, the share the
10-day arm already received). The retained 10-day/1500 arm stays the reference and the
archived 40-day/1500 arm is the step-count control, so the step effect is measured on the
same pool, seeds and batch order. Weights are refitted from the same seed: no 1500-step
optimizer state was saved, so resuming that trajectory would be a different treatment.

Every gate is frozen in the plan before the first 6000-step fit exists.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.checks import check
from jinyiquant.config import EvaluationConfig, RunConfig
from jinyiquant.io import digest, file_sha256, read_json, utc_now, write_json
from jinyiquant.runner import _answer_coverage_check, save_checks
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir
from modified_tlob.coverage import DELTA_METRICS, arm_exposure, coverage_decision
from modified_tlob.ensemble import SEEDS, paired_block_intervals
from scripts.run_seed_ensemble import claim_training, complete_correlations
from scripts.run_training_coverage_arms import (
    BudgetExceeded,
    arm_declaration,
    fit_worker,
    frozen_limits,
    infer_worker,
    reuse_measurements,
    score_predictions,
    source_identity,
    supervise,
    window_scope,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "coverage-step-scaling-20261018"
KEYS = ["date", "instrument"]
WINDOWS = ("research_a", "research_b")
REFERENCE, CONTROL, CANDIDATE = "ref10", "ctrl40", "cand40"
STEPS = {REFERENCE: 1500, CONTROL: 1500, CANDIDATE: 6000}
BATCH = 32
SCREEN_SEEDS, FULL_SEEDS = (11,), tuple(SEEDS)
METRIC_KEYS = ("rank_ic_mean", "long_short_return_mean", "min_daily_long_short_return")
FROZEN = ROOT / "configs/training_coverage_20261017_v4_frozen.json"
REFERENCE_RUN = ROOT / "runs/fixed-seed-ensemble-20261017/20261017T064944.073834Z_193b03ead7ff"
SCORING_RUN = ROOT / "runs/fixed-seed-ensemble-20261017/analysis-recovery/20261017T091219.300500Z_7c40b5b5d26d"
PRIOR_EXECUTION = ROOT / "runs/training-coverage-arms-20261018/20261018T025756.181886Z_3b035283cf60"
PRIOR_MEASUREMENTS = ROOT / "runs/training-coverage-arms-20261018/20261018T024634.978262Z_ea2f53860760"
PLAN = ROOT / "configs/coverage_step_scaling_20261018_plan.json"
RUNS = ROOT / "runs" / EXPERIMENT


def job_recorder(output):
    rec = Recorder(output, EXPERIMENT)
    rec.run_id = output.name + "_" + digest(str(output.resolve()))[:12]
    return rec


def fit_dir(root, name, seed):
    return Path(root) / "fits" / f"{name}_st{STEPS[CANDIDATE]}_s{seed}"


# ------------------------------------------------------------------ plan freeze


def assert_steps_are_the_only_recipe_change(archived, candidate, label):
    """The candidate must differ from the 1500-step control in steps and execution caps only."""
    differing = {key for key in set(archived) | set(candidate) if archived.get(key) != candidate.get(key)}
    if differing - {"budget", "model", "experiment_id", "hypothesis"}:
        raise ValueError(f"{label}: unexpected recipe differences {sorted(differing)}")
    model_keys = sorted(key for key in set(archived["model"]) | set(candidate["model"])
                        if archived["model"].get(key) != candidate["model"].get(key))
    if model_keys != ["max_steps"]:
        raise ValueError(f"{label}: only model.max_steps may change, saw {model_keys}")
    if (archived["model"]["max_steps"], candidate["model"]["max_steps"]) != (STEPS[CONTROL], STEPS[CANDIDATE]):
        raise ValueError(f"{label}: step change is not {STEPS[CONTROL]} -> {STEPS[CANDIDATE]}")
    differing_budget = set(archived["budget"]) ^ set(candidate["budget"])
    if differing_budget - {"max_rss_mb", "wall_seconds", "max_train_samples"}:
        raise ValueError(f"{label}: unexpected budget fields {sorted(differing_budget)}")


def step_loss(events_path, step):
    """Observed training loss at one optimizer step, for trajectory equivalence."""
    for line in Path(events_path).read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if (event.get("stage") == "training" and event.get("event_type") == "progress"
                and event.get("context", {}).get("step") == step):
            return float(event["context"]["loss"])
    raise ValueError(f"No training progress event at step {step} in {events_path}")


def reusable_arm(frozen, name, days, fit_root):
    """One archived arm: its declared training set, its config and its per-seed artifacts."""
    days = int(days)
    arm = REFERENCE if days == 10 else CONTROL
    declaration = arm_declaration(frozen, name, str(days))
    window = next(w for w in frozen["windows"] if w["name"] == name)
    entry = {"days": days, "steps": STEPS[arm], "samples": declaration["samples"],
             "frozen_identity": {key: declaration[key] for key in ("samples", "keys_sha256", "labels_sha256")},
             "config": copy.deepcopy(window["arms"][str(days)]["config"]),
             "train_dates": list(window["arms"][str(days)]["train_dates"]),
             "train_end_exclusive": window["arms"][str(days)]["train_end_exclusive"], "per_seed": {}}
    for seed in SEEDS:
        fitting = Path(fit_root) / f"{name}_s{seed}"
        identity = read_json(fitting / "training_identity.json")
        expected = {**entry["frozen_identity"], "dates": entry["train_dates"]}
        if {key: identity.get(key) for key in expected} != expected:
            raise ValueError(f"{name} seed {seed} {days}-day arm was fitted on different training data")
        predictions = Path(fit_root).parent / "inference" / name / f"seed{seed}_predictions.parquet"
        if not predictions.is_file():
            raise FileNotFoundError(predictions)
        entry["per_seed"][str(seed)] = {"fit_dir": str(fitting), "weights": str(fitting / "model.pt"),
                                        "weights_sha256": file_sha256(fitting / "model.pt"),
                                        "predictions": str(predictions),
                                        "predictions_sha256": file_sha256(predictions)}
        archived = read_json(fitting / "config.json")
        differing = {key for key in set(archived) | set(entry["config"])
                     if archived.get(key) != entry["config"].get(key)}
        if differing - {"experiment_id", "hypothesis", "seed"}:
            raise ValueError(f"{name} seed {seed} {days}-day recipe differs from the frozen arm: "
                             f"{sorted(differing)}")
    return entry


def prepare(output_path, *, frozen_path=FROZEN, prior_execution=PRIOR_EXECUTION,
            measurements_from=PRIOR_MEASUREMENTS, output_root=RUNS):
    """Freeze the candidate recipe, the reused controls and every gate before training."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Frozen plans are immutable; select a new output name")
    frozen_path = Path(frozen_path).resolve()
    frozen = read_json(frozen_path)
    if file_sha256(frozen["registry_path"]) != frozen["registry_sha256"]:
        raise ValueError("Shared data-role registration changed")
    prior_execution = Path(prior_execution).resolve()
    if read_json(prior_execution / "manifest.json")["status"] != "completed":
        raise ValueError("The archived 40-day/1500 arm must be a completed archive")
    prior_decision = read_json(prior_execution / "decision.json")
    if prior_decision["result"] == "accept":
        raise ValueError("This round only runs when the archived 40-day arm failed its gates")
    run = new_run_dir(Path(output_root))
    rec = Recorder(run, EXPERIMENT)
    write_json(run / "manifest.json", {"schema_version": 1, "run_id": run.name, "experiment_id": EXPERIMENT,
        "started_at": utc_now(), "status": "running", "new_training": 0, "final_views": 0, "mode": "prepare",
        "code": capture_source(run), "environment": environment()})
    try:
        rec.event("run", "started", mode="prepare", candidate_steps=STEPS[CANDIDATE])
        reused = reuse_measurements(Path(measurements_from), frozen)
        limits = frozen_limits(reused["measurements"], training_steps=STEPS[CANDIDATE])
        rec.event("budget", "frozen", max_rss_mb=limits["max_rss_mb"], wall_seconds=limits["wall_seconds"],
                  basis=limits["basis"])
        windows, inventory = [], {}
        for name in WINDOWS:
            scope = window_scope(frozen, name, "40")
            reference = reusable_arm(frozen, name, 10, REFERENCE_RUN / "fits")
            control = reusable_arm(frozen, name, 40, prior_execution / "fits")
            candidate = {"days": 40, "steps": STEPS[CANDIDATE], "samples": control["samples"],
                         "frozen_identity": dict(control["frozen_identity"]),
                         "train_dates": list(control["train_dates"]),
                         "train_end_exclusive": control["train_end_exclusive"],
                         "config": copy.deepcopy(control["config"])}
            candidate["config"]["model"]["max_steps"] = STEPS[CANDIDATE]
            candidate["config"]["budget"] = {"max_train_samples": control["samples"],
                                             "max_rss_mb": limits["max_rss_mb"],
                                             "wall_seconds": limits["wall_seconds"]}
            assert_steps_are_the_only_recipe_change(control["config"], candidate["config"], name)
            targets = SCORING_RUN / f"{name}_targets.parquet"
            if not targets.is_file():
                raise FileNotFoundError(targets)
            inventory[str(targets)] = file_sha256(targets)
            arms = {REFERENCE: reference, CONTROL: control, CANDIDATE: candidate}
            for arm in (REFERENCE, CONTROL):
                for row in arms[arm]["per_seed"].values():
                    inventory[row["predictions"]] = row["predictions_sha256"]
                    inventory[row["weights"]] = row["weights_sha256"]
            windows.append({
                "name": name, "train_start": scope["train_start"],
                "train_end_exclusive": scope["train_end_exclusive"],
                "evaluation_start": scope["evaluation_start"],
                "evaluation_end_exclusive": scope["evaluation_end_exclusive"],
                "evaluation_dates": scope["evaluation_dates"], "arms": arms,
                "exposure": {
                    "batch_size": BATCH,
                    "arms": {arm: arm_exposure(spec["samples"], steps=spec["steps"], batch_size=BATCH)
                             for arm, spec in arms.items()},
                    "interpretation": (
                        "The candidate gets four times the optimizer compute of both controls: 6000 x 32 = "
                        "192,000 sample exposures, i.e. 0.60 epochs of the 40-day pool, the share the "
                        "10-day/1500 arm already received. A gain therefore measures the value of extra "
                        "optimization on the same 40-day pool, not the value of more history; separating "
                        "those needs an equal-epoch 10-day/6000-step arm.")},
                "trajectory": {seed: {"step": STEPS[CONTROL],
                                      "observed_loss": step_loss(Path(row["fit_dir"]) / "events.jsonl",
                                                                 STEPS[CONTROL])}
                               for seed, row in control["per_seed"].items()},
                "targets": {"path": str(targets), "sha256": inventory[str(targets)]},
                "fit_once_before_evaluation": True})
        identity_rows = source_identity(
            [REFERENCE_RUN / f"fits/{name}_s{seed}/source.zip" for name in WINDOWS for seed in SEEDS]
            + [prior_execution / f"fits/{name}_s{seed}/source.zip" for name in WINDOWS for seed in SEEDS])
        inventory[str(prior_execution / "decision.json")] = file_sha256(prior_execution / "decision.json")
        policy = copy.deepcopy(frozen["policy"])
        plan = {
            "schema_version": 1, "experiment_id": EXPERIMENT, "created_at_utc": utc_now(),
            "hypothesis": "The 40-day arm failed at 1500 steps because that is only 0.15 epochs of its pool, "
                          "not because the extra dates hurt: at 6000 steps (0.60 epochs, the share the 10-day "
                          "arm already received) it should stop degrading research_b and clear the frozen "
                          "worst-day guardrail.",
            "only_factor": "optimizer steps 1500 -> 6000 for the 40-day arm at fixed pool, network, learning "
                           "rate, batch size, preprocessing and seeds",
            "source_frozen": {"path": str(frozen_path), "sha256": file_sha256(frozen_path)},
            "prior_round": {"execution": str(prior_execution),
                            "decision_sha256": inventory[str(prior_execution / "decision.json")],
                            "result": prior_decision["result"], "action": prior_decision["action"],
                            "mean_rank_ic_delta": prior_decision["mean_rank_ic_delta"],
                            "period_mean_deltas": prior_decision["period_mean_deltas"],
                            "failed_checks": [key for key, passed in prior_decision["checks"].items() if not passed]},
            "recipe_fixed": {"raw12": True, "sample_times": 8, "lookback": 128, "loss": "MSE",
                             "target_scaling": "train_std", "learning_rate": 1e-4, "batch_size": BATCH,
                             "smoothing": "none", "time_loss_weighting": "none"},
            "steps": {"reference": STEPS[REFERENCE], "control": STEPS[CONTROL], "candidate": STEPS[CANDIDATE]},
            "seeds": {"screen": list(SCREEN_SEEDS), "full": list(FULL_SEEDS)}, "deployment_seed": 11,
            "windows": windows, "training_source_identity": identity_rows,
            "policy": policy, "calibration": frozen["calibration"], "statistics": policy["statistics"],
            "screen": {"seeds": list(SCREEN_SEEDS),
                       "gates": {"rank_ic_delta_at_least": 0.0,
                                 "worst_day_delta_at_least": -policy["noise"]["min_daily_long_short_return"]},
                       "rule": "Screen at seed 11 before spending four more fits: the candidate must not "
                               "degrade the retained 10-day reference in either research window and must clear "
                               "the frozen worst-day guardrail in both. A failed screen ends this round; no "
                               "other step count is searched afterwards."},
            "budget": {"local_peak_rss_cap_mib": limits["max_rss_mb"],
                       "per_fit_wall_cap_seconds": limits["wall_seconds"], "fit_attempts_max": 6,
                       "fit_stage_wall_cap_seconds": 6 * limits["wall_seconds"] + 2 * 1800,
                       "inference_chunk_days": 5, "inference_windows": len(WINDOWS)},
            "measured_budget": {"limits": limits, "pilots": reused["measurements"],
                                "pilot_source": reused["source"],
                                "pilot_sha256": reused["measurements_sha256"]},
            "registry_path": frozen["registry_path"], "registry_sha256": frozen["registry_sha256"],
            "inputs_sha256": inventory, "final_views": 0,
            "access_policy": "research_a/research_b are already-used research data; confirmation, locked and "
                             "legacy final ranges stay closed this round.",
            "notes": "Two fits (seed 11) are spent before the screen; four more only if it passes. The twelve "
                     "reused fits are never refitted.",
        }
        checks = [
            check("reused_arms_have_three_seeds_each",
                  all(len(window["arms"][arm]["per_seed"]) == len(SEEDS)
                      for window in windows for arm in (REFERENCE, CONTROL))),
            check("candidate_steps_are_the_only_recipe_change",
                  all(assert_steps_are_the_only_recipe_change(
                      window["arms"][CONTROL]["config"], window["arms"][CANDIDATE]["config"],
                      window["name"]) is None for window in windows),
                  control_steps=STEPS[CONTROL], candidate_steps=STEPS[CANDIDATE]),
            check("prior_round_failed_its_gates", prior_decision["result"] != "accept",
                  prior_result=prior_decision["result"]),
        ]
        if not save_checks(run, checks):
            raise ValueError("Plan freeze checks failed")
        write_json(output_path, plan)
        write_json(run / "plan.json", plan)
        rec.event("prepare", "completed", plan=str(output_path), plan_sha256=file_sha256(output_path),
                  screen_fits=2 * len(SCREEN_SEEDS), conditional_fits=2 * (len(FULL_SEEDS) - 1),
                  per_fit_wall_cap_seconds=limits["wall_seconds"])
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
    command = [sys.executable, "-m", "scripts.run_coverage_step_scaling", "worker",
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


# ------------------------------------------------------------------ scoring and pairing


def evaluation_config(window):
    """Evaluation settings come from the frozen arm; never re-specified here."""
    return EvaluationConfig.model_validate(window["arms"][CONTROL]["config"]["evaluation"])


def score_windows(plan, paths, checks):
    """Score every arm and seed on the frozen answers through one code path.

    ``paths`` maps window name -> arm -> seed -> prediction file.
    """
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
                metrics[(arm, seed)] = {key: values[key] for key in (*METRIC_KEYS, "turnover_mean")}
                population.setdefault(arm, set()).add(values["sample_keys_sha256"] + values["targets_sha256"])
        if any(len(value) != 1 for value in population.values()):
            raise ValueError(f"{name}: seeds were scored on different keys")
        if len({next(iter(value)) for value in population.values()}) != 1:
            raise ValueError(f"{name}: arms were scored on different keys or answers")
        scored[name] = {"daily": daily, "metrics": metrics, "targets": targets}
    return scored


def arm_means(daily, arm, seeds):
    """Seed-mean per evaluation day: seeds are replicates, days are the paired unit."""
    frames = [daily[(arm, seed)] for seed in seeds]
    if any(not frame.index.equals(frames[0].index) for frame in frames):
        raise ValueError("Seed daily grids differ inside one arm")
    return pd.concat(frames).groupby(level=0).mean()


def pair_up(scored_window, name, high, low, seeds):
    """Paired daily deltas (high minus low) for one window; seeds are averaged first."""
    means = {arm: arm_means(scored_window["daily"], arm, seeds) for arm in (high, low)}
    if not means[high].index.equals(means[low].index):
        raise ValueError("Arms were scored on different evaluation days")
    delta = means[high] - means[low]
    frame = delta.reset_index()
    frame["period"] = name
    summary = {"period": name, "days": len(delta),
               "arm_window_means": {arm: {column: float(means[arm][column].mean()) for column in DELTA_METRICS}
                                    for arm in (high, low)},
               "worst_day_delta": float(means[high].long_short_return.min()
                                        - means[low].long_short_return.min()),
               "min_day_delta": float(delta.long_short_return.min()),
               "per_seed_mean_delta": {
                   str(seed): {column: float(scored_window["daily"][(high, seed)][column].mean()
                                             - scored_window["daily"][(low, seed)][column].mean())
                               for column in DELTA_METRICS} for seed in seeds},
               "per_seed_metrics": {str(seed): {arm: scored_window["metrics"][(arm, seed)]
                                                for arm in (high, low)} for seed in seeds}}
    return frame, summary


def analysis(plan, scored, high, low, seeds):
    """Paired deltas for one comparison: per-window summaries, pooled block intervals."""
    frames, summaries = [], []
    for window in plan["windows"]:
        name = window["name"]
        frame, summary = pair_up(scored[name], name, high, low, seeds)
        frames.append(frame)
        summaries.append(summary)
    paired = pd.concat(frames, ignore_index=True)
    statistics = plan["statistics"]
    intervals = [paired_block_intervals(paired, block_days=block, samples=statistics["bootstrap_samples"],
                                        seed=statistics["bootstrap_seed"])
                 for block in statistics["block_days"]]
    return paired, intervals, summaries


def screen_decision(paired, summaries, intervals, plan):
    """Frozen seed-11 screen: no degradation in either window plus the worst-day guardrail."""
    gates = plan["screen"]["gates"]
    means = paired.groupby("period").rank_ic.mean()
    worst = {row["period"]: row["worst_day_delta"] for row in summaries}
    checks = {}
    for name in WINDOWS:
        checks[f"{name}_rank_ic_delta_at_least"] = bool(float(means[name]) >= gates["rank_ic_delta_at_least"])
        checks[f"{name}_worst_day_delta_at_least"] = bool(worst[name] >= gates["worst_day_delta_at_least"])
    passed = all(checks.values())
    return {"schema_version": 1, "result": "pass" if passed else "reject", "seeds": list(SCREEN_SEEDS),
            "comparison": f"{CANDIDATE} minus {REFERENCE}", "checks": checks, "gates": gates,
            "mean_rank_ic_delta": float(paired.rank_ic.mean()),
            "period_mean_deltas": {str(key): float(value) for key, value in means.items()},
            "per_window": {row["period"]: {"worst_day_delta": row["worst_day_delta"],
                                           "arm_window_means": row["arm_window_means"],
                                           "per_seed_mean_delta": row["per_seed_mean_delta"]}
                           for row in summaries},
            "intervals": intervals,
            "descriptive_only": "Single-seed intervals understate uncertainty; the registered decision "
                                "requires the three-seed replay.",
            "rule": plan["screen"]["rule"],
            "next": "continue to seeds 29/47 and the registered three-seed decision" if passed
                    else "end this round and retain the 10-day reference; no other step count is searched"}


def reference_decision(plan, paired, summaries, intervals, exposure, cost_measured):
    """The unchanged archived gates, asked again of the 6000-step candidate."""
    return coverage_decision(
        paired, summaries, intervals, plan["policy"], cost_measured=cost_measured, exposure=exposure,
        only_factor=plan["only_factor"],
        interpretation="A non-accept result rejects the claim that four times the optimizer compute repairs the "
                       "40-day arm at this exposure: 6000 steps is still only 0.60 epochs of the 40-day pool, and "
                       "step counts are frozen for this round.",
        limitations="Two research windows and three seeds are selection data; local gross metrics exclude fees "
                    "and slippage; the platform time and memory limits remain unverified.")


def step_effect(paired, summaries, intervals):
    """Descriptive step contrast on the same pool: it explains, it does not decide."""
    return {"schema_version": 1, "comparison": f"{CANDIDATE} minus {CONTROL}",
            "scope": "descriptive_mechanism_evidence_no_gates",
            "mean_rank_ic_delta": float(paired.rank_ic.mean()),
            "period_mean_deltas": {str(key): float(value) for key, value in
                                   paired.groupby("period").rank_ic.mean().items()},
            "gross_mean_delta": float(paired.long_short_return.mean()),
            "turnover_mean_delta": float(paired.turnover.mean()),
            "per_window": {row["period"]: {"worst_day_delta": row["worst_day_delta"],
                                           "per_seed_mean_delta": row["per_seed_mean_delta"]}
                           for row in summaries},
            "intervals": intervals,
            "note": "Steps are the only treatment, so a positive contrast shows the extra optimization helped on "
                    "the same 40-day pool, seeds and batch order. It cannot show that longer history beats recent "
                    "data at equal compute."}


def trajectory_report(plan, roots, atol=1e-6, rtol=1e-5):
    """Did the 6000-step refit reproduce the archived 1500-step trajectory up to its endpoint?"""
    roots = [Path(root) for root in roots]
    rows = []
    for window in plan["windows"]:
        for seed, expected in window["trajectory"].items():
            events = next((fit_dir(root, window["name"], seed) / "events.jsonl" for root in roots
                           if (fit_dir(root, window["name"], seed) / "events.jsonl").is_file()), None)
            if events is None:
                continue
            actual = step_loss(events, STEPS[CONTROL])
            rows.append({"window": window["name"], "seed": int(seed), "step": STEPS[CONTROL],
                         "control_loss": expected["observed_loss"], "candidate_loss": actual,
                         "matches": bool(abs(actual - expected["observed_loss"])
                                         <= atol + rtol * abs(expected["observed_loss"]))})
    checks = [check("candidate_reproduces_the_control_trajectory_to_step_1500",
                    all(row["matches"] for row in rows) if rows else True, rows=rows)]
    return rows, checks


# ------------------------------------------------------------------ execution


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
            for seed in SCREEN_SEEDS:
                fit_candidate(plan_path, root, plan, window, seed, rec, attempts, started)
            inference = root / "inference" / name
            launch(plan_path, inference, "infer", name,
                   paths={str(seed): str(fit_dir(root, name, seed) / "model.pt") for seed in SCREEN_SEEDS})
            candidate_paths[name] = {CANDIDATE: {seed: str(inference / f"seed{seed}_predictions.parquet")
                                                 for seed in SCREEN_SEEDS}}
        screen = run_screen(plan, root, candidate_paths, rec)
        write_json(root / "screen.json", screen)
        if screen["decision"]["result"] != "pass":
            write_json(root / "decision.json", {
                "schema_version": 1, "stage": "screen", "result": "reject",
                "action": "retain_10_day_reference", "reason": "screen_failed",
                "failed_checks": [key for key, passed in screen["decision"]["checks"].items() if not passed],
                "mean_rank_ic_delta": screen["decision"]["mean_rank_ic_delta"],
                "period_mean_deltas": screen["decision"]["period_mean_deltas"], "screen": screen,
                "training_attempts": len(attempts), "final_views": 0, "model_promoted": False,
                "note": "The round ends here by preregistration: no other step count is searched."})
            rec.event("run", "completed", phase="screen", result="reject", training_attempts=len(attempts))
            finish(root, "completed", 0, resources={"wall_seconds": time.monotonic() - started})
            print("screen reject", screen["decision"]["mean_rank_ic_delta"],
                  screen["decision"]["period_mean_deltas"], flush=True)
            return
        for window in plan["windows"]:
            name = window["name"]
            for seed in [seed for seed in FULL_SEEDS if seed not in SCREEN_SEEDS]:
                fit_candidate(plan_path, root, plan, window, seed, rec, attempts, started)
            inference = root / "inference_full" / name
            launch(plan_path, inference, "infer", name,
                   paths={str(seed): str(fit_dir(root, name, seed) / "model.pt") for seed in FULL_SEEDS})
            candidate_paths[name][CANDIDATE] = {seed: str(inference / f"seed{seed}_predictions.parquet")
                                                for seed in FULL_SEEDS}
        run_full(plan, root, candidate_paths, screen, rec, {name: {seed: str(root / "inference" / name
                                                                      / f"seed{seed}_predictions.parquet")
                                                                   for seed in SCREEN_SEEDS}
                                                             for name in candidate_paths})
        rec.event("run", "completed", phase="full", training_attempts=len(attempts), final_views=0)
        finish(root, "completed", 0, resources={"wall_seconds": time.monotonic() - started})
    except BaseException as exc:
        rec.failure("experiment", exc)
        write_json(root / "decision.json", {"result": "inconclusive", "action": "retain_10_day_reference",
            "reason": str(exc), "training_attempts": len(attempts), "final_views": 0, "model_promoted": False})
        finish(root, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", 1,
               resources={"wall_seconds": time.monotonic() - started}, error={"reason": str(exc)})
        raise


def prior_claims(stem):
    """Claimed fits for one arm; a claim whose archive holds no model produced no result."""
    rows = []
    for claim in sorted((RUNS / "training_claims").glob(stem + "*.json")):
        record = read_json(claim)
        archive = Path(record["run"])
        rows.append({"claim": str(claim), "archive": str(archive),
                     "claimed_at_utc": record.get("claimed_at_utc"),
                     "complete": (archive / "model.pt").is_file() and (archive / "training.json").is_file()})
    return rows


def claim_fit(plan_path, name, seed, path):
    """Claim one fit; retry an interrupted attempt but never refit one that produced a model."""
    stem = f"{name}_st{STEPS[CANDIDATE]}_s{seed}"
    prior = prior_claims(stem)
    if any(row["complete"] for row in prior):
        raise ValueError("Training claim already spent; no additional fits allowed")
    attempt = len(prior) + 1
    claim = RUNS / "training_claims" / (f"{stem}.json" if attempt == 1 else f"{stem}_attempt{attempt}.json")
    claim_training(claim, path, name, seed, plan_path)
    return attempt, prior


def fit_candidate(plan_path, root, plan, window, seed, rec, attempts, started):
    """One refitted 6000-step arm; a retry is only possible while no attempt produced a model."""
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
    attempts[-1]["status"] = "completed"
    write_json(root / "attempts.json", attempts)


def frozen_arm_paths(plan, window):
    """Reused arms of one window: the retained 10-day reference and the 40-day/1500 control."""
    return {arm: {int(seed): row["predictions"] for seed, row in window["arms"][arm]["per_seed"].items()}
            for arm in (REFERENCE, CONTROL)}


def recorded_fits(root):
    """Every completed fit behind this archive, including one reused from a screen archive."""
    fits = []
    if (Path(root) / "screen_source.json").is_file():
        for row in read_json(Path(root) / "screen_source.json")["fits"].values():
            fits.append({"window": row["window"], "seed": row["seed"], "steps": row["steps"], "run": row["run"],
                         "resources": row.get("resources"), "weights_sha256": row["weights_sha256"],
                         "reused_from_screen_archive": True})
    if (Path(root) / "attempts.json").is_file():
        for item in read_json(Path(root) / "attempts.json"):
            if item["status"] != "completed":
                continue
            fits.append({"window": item["window"], "seed": item["seed"], "steps": item["steps"],
                         "run": item["path"], "attempt": item.get("attempt"),
                         "resources": read_json(Path(item["path"]) / "manifest.json").get("resources"),
                         "weights_sha256": file_sha256(Path(item["path"]) / "model.pt")})
    return fits


def screen_source(plan, plan_path, source_run):
    """Prove a passed screen came from this frozen plan and still owns its seed-11 models."""
    source_run = Path(source_run).resolve()
    if read_json(source_run / "manifest.json").get("plan_sha256") != file_sha256(plan_path):
        raise ValueError("Screen archive was produced by a different frozen plan")
    screen = read_json(source_run / "screen.json")
    if screen["decision"]["result"] != "pass":
        raise ValueError("The full stage requires a passed screen")
    fits, predictions = {}, {}
    for window in plan["windows"]:
        name = window["name"]
        fitting = fit_dir(source_run, name, SCREEN_SEEDS[0])
        training = read_json(fitting / "training.json")
        expected = {**window["arms"][CANDIDATE]["frozen_identity"],
                    "dates": window["arms"][CANDIDATE]["train_dates"]}
        identity = read_json(fitting / "training_identity.json")
        if {key: identity.get(key) for key in expected} != expected:
            raise ValueError(f"{name}: the screen fit used a different training set")
        if training["steps"] != STEPS[CANDIDATE]:
            raise ValueError(f"{name}: the screen fit did not run the declared steps")
        weights = fitting / "model.pt"
        prediction = source_run / "inference" / name / f"seed{SCREEN_SEEDS[0]}_predictions.parquet"
        if not (weights.is_file() and prediction.is_file()):
            raise FileNotFoundError(f"Screen artifacts are incomplete for {name}")
        fits[name] = {"window": name, "seed": SCREEN_SEEDS[0], "steps": STEPS[CANDIDATE], "run": str(fitting),
                      "resources": read_json(fitting / "manifest.json").get("resources"),
                      "weights": str(weights), "weights_sha256": file_sha256(weights),
                      "predictions": str(prediction), "predictions_sha256": file_sha256(prediction)}
        predictions[name] = {SCREEN_SEEDS[0]: str(prediction)}
    return {"run": str(source_run), "manifest_sha256": file_sha256(source_run / "manifest.json"),
            "screen_path": str(source_run / "screen.json"),
            "screen_sha256": file_sha256(source_run / "screen.json"),
            "decision": screen["decision"], "fits": fits, "predictions": predictions}


def execute_full(plan_path, root, source_run):
    """Continue a passed screen in a new archive: keep its seed-11 fits, add seeds 29 and 47."""
    plan = read_json(plan_path)
    started = time.monotonic()
    rec = Recorder(root, EXPERIMENT)
    source = screen_source(plan, plan_path, source_run)
    write_json(root / "manifest.json", {"schema_version": 1, "run_id": root.name, "experiment_id": EXPERIMENT,
        "started_at": utc_now(), "status": "running", "stage": "full", "plan_sha256": file_sha256(plan_path),
        "screen_source": source["run"], "screen_source_manifest_sha256": source["manifest_sha256"],
        "code": capture_source(root), "environment": environment(), "final_views": 0})
    write_json(root / "preregistration.json", plan)
    attempts = []
    try:
        rec.event("run", "started", phase="full", screen_source=source["run"],
                  screen_sha256=source["screen_sha256"])
        if file_sha256(plan["registry_path"]) != plan["registry_sha256"]:
            raise ValueError("Shared data-role registration changed")
        for path, checksum in plan["inputs_sha256"].items():
            if file_sha256(path) != checksum:
                raise ValueError(f"Frozen input changed: {path}")
        write_json(root / "screen_source.json", source)
        write_json(root / "screen.json", {"decision": source["decision"], "source": source["run"],
                                          "screen_sha256": source["screen_sha256"]})
        candidate_paths = {}
        for window in plan["windows"]:
            name = window["name"]
            for seed in [seed for seed in FULL_SEEDS if seed not in SCREEN_SEEDS]:
                fit_candidate(plan_path, root, plan, window, seed, rec, attempts, started)
            inference = root / "inference_full" / name
            weights = {str(seed): str(fit_dir(root, name, seed) / "model.pt")
                       for seed in FULL_SEEDS if seed not in SCREEN_SEEDS}
            weights[str(SCREEN_SEEDS[0])] = source["fits"][name]["weights"]
            launch(plan_path, inference, "infer", name, paths=weights)
            candidate_paths[name] = {CANDIDATE: {seed: str(inference / f"seed{seed}_predictions.parquet")
                                                 for seed in FULL_SEEDS}}
        run_full(plan, root, candidate_paths, {"decision": source["decision"]}, rec, source["predictions"],
                 fit_roots=[root, source["run"]])
        rec.event("run", "completed", phase="full", training_attempts=len(attempts), final_views=0)
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
    paths = {window["name"]: {REFERENCE: {SCREEN_SEEDS[0]:
                                          window["arms"][REFERENCE]["per_seed"][str(SCREEN_SEEDS[0])]["predictions"]},
                              CANDIDATE: candidate_paths[window["name"]][CANDIDATE]}
             for window in plan["windows"]}
    scored = score_windows(plan, paths, checks)
    paired, intervals, summaries = analysis(plan, scored, CANDIDATE, REFERENCE, SCREEN_SEEDS)
    paired.to_csv(root / "screen_paired_daily.csv", index=False)
    decision = screen_decision(paired, summaries, intervals, plan)
    rows, trajectory_checks = trajectory_report(plan, root)
    write_json(root / "trajectory.json", {"rows": rows, "atol": 1e-6, "rtol": 1e-5})
    checks.extend(trajectory_checks)
    if not save_checks(root, checks):
        raise ValueError("Screen scoring checks failed")
    rec.event("evaluation", "screen", result=decision["result"],
              mean_rank_ic_delta=decision["mean_rank_ic_delta"],
              period_mean_deltas=decision["period_mean_deltas"], checks=decision["checks"],
              per_window=decision["per_window"])
    print("screen", decision["result"], decision["mean_rank_ic_delta"],
          decision["period_mean_deltas"], flush=True)
    return {"decision": decision, "per_window": summaries, "paired_intervals": intervals,
            "steps": STEPS, "exposure": {window["name"]: window["exposure"] for window in plan["windows"]}}


def run_full(plan, root, candidate_paths, screen, rec, screen_predictions, fit_roots=None):
    checks = []
    rows, trajectory_checks = trajectory_report(plan, fit_roots or [root])
    write_json(root / "trajectory.json", {"rows": rows, "atol": 1e-6, "rtol": 1e-5})
    checks.extend(trajectory_checks)
    paths = {}
    for window in plan["windows"]:
        name = window["name"]
        paths[name] = {**frozen_arm_paths(plan, window), CANDIDATE: candidate_paths[name][CANDIDATE]}
    scored = score_windows(plan, paths, checks)
    costs = {}
    for window in plan["windows"]:
        name = window["name"]
        for seed, earlier in screen_predictions[name].items():
            if seed not in paths[name][CANDIDATE]:
                continue
            made_earlier = pd.read_parquet(earlier)
            made_now = pd.read_parquet(paths[name][CANDIDATE][seed])
            checks.append(check(f"{name}_s{seed}_separate_inference_processes_agree",
                                len(made_earlier) == len(made_now)
                                and np.allclose(made_earlier.score, made_now.score, atol=1e-6, rtol=1e-5)))
        costs[name] = read_json(root / "inference_full" / name / "inference_cost.json")
    paired, intervals, summaries = analysis(plan, scored, CANDIDATE, REFERENCE, FULL_SEEDS)
    paired.to_csv(root / "paired_daily.csv", index=False)
    write_json(root / "paired_intervals.json", intervals)
    steps_paired, steps_intervals, steps_summaries = analysis(plan, scored, CANDIDATE, CONTROL, FULL_SEEDS)
    steps_paired.to_csv(root / "step_effect_daily.csv", index=False)
    step = step_effect(steps_paired, steps_summaries, steps_intervals)
    exposure = {window["name"]: window["exposure"] for window in plan["windows"]}
    decision = reference_decision(plan, paired, summaries, intervals, exposure, cost_measured=bool(costs))
    decision["stage"] = "full"
    decision["screen"] = screen["decision"]
    decision["step_effect"] = {key: step[key] for key in
                               ("comparison", "mean_rank_ic_delta", "period_mean_deltas")}
    write_json(root / "decision.json", decision)
    write_json(root / "step_effect.json", step)
    write_json(root / "metrics.json", {
        "windows": {row["period"]: {"reference_comparison": row} for row in summaries},
        "step_effect": step, "inference_cost": costs, "exposure": exposure,
        "calibration": plan["calibration"]["noise"], "minimum_mean_gain": plan["policy"]["minimum_mean_gain"],
        "costs": {"per_fit_wall_cap_seconds": plan["budget"]["per_fit_wall_cap_seconds"],
                  "local_peak_rss_cap_mib": plan["budget"]["local_peak_rss_cap_mib"],
                  "fits": recorded_fits(root)}})
    if not save_checks(root, checks):
        raise ValueError("Full scoring checks failed")
    rec.event("evaluation", "decision", result=decision["result"], action=decision["action"],
              mean_rank_ic_delta=decision["mean_rank_ic_delta"],
              period_mean_deltas=decision["period_mean_deltas"], checks=decision["checks"])
    rec.event("evaluation", "step_effect", mean_rank_ic_delta=step["mean_rank_ic_delta"],
              period_mean_deltas=step["period_mean_deltas"])
    print("decision", decision["result"], decision["action"], decision["mean_rank_ic_delta"],
          decision["period_mean_deltas"], flush=True)
    print("step_effect", step["mean_rank_ic_delta"], step["period_mean_deltas"], flush=True)


# ------------------------------------------------------------------ research record


def conclusion_text(decision, step, plan):
    failed = [key for key, passed in decision["checks"].items() if not passed]
    per_window = ", ".join(f"{key}: {value:+.6f}" for key, value in decision["period_mean_deltas"].items())
    head = (f"With only the 40-day arm's steps changed ({STEPS[CONTROL]} -> {STEPS[CANDIDATE]}, 192,000 sample "
            f"exposures = 0.60 epochs of the 40-day pool), the seed-mean paired daily all-eight-time RankIC "
            f"difference against the retained 10-day/1500 reference was {decision['mean_rank_ic_delta']:+.6f} "
            f"against the unchanged predeclared floor of {decision['minimum_mean_gain']:.6f}; per window "
            f"{per_window}. The descriptive step contrast against the archived 40-day/1500 arm was "
            f"{step['mean_rank_ic_delta']:+.6f} ({', '.join(f'{k}: {v:+.6f}' for k, v in step['period_mean_deltas'].items())}).")
    if decision["result"] == "accept":
        return (head + " The candidate clears the gain, interval, replication, guardrail and cost gates, so the "
                       "40-day plan is repaired at this exposure and may be registered for a separately frozen "
                       "confirmation stage. This does not show that longer history beats recent data at equal "
                       "compute; that needs a 10-day/6000-step arm.")
    return (head + f" The candidate fails the predeclared gates {', '.join(failed) or 'unspecified'}, so the "
                   "10-day reference is retained. Per preregistration the round ends here: no other step count "
                   "was searched, and this is not evidence that more history is uninformative.")


def record(plan_path, execution, output_path):
    """Write the structured research record from a completed archive; never hand-edited."""
    plan = read_json(plan_path)
    execution = Path(execution).resolve()
    if read_json(execution / "manifest.json")["status"] != "completed":
        raise ValueError("A research record requires a completed execution archive")
    decision = read_json(execution / "decision.json")
    screen = read_json(execution / "screen.json")
    trajectory = read_json(execution / "trajectory.json")
    step = read_json(execution / "step_effect.json") if (execution / "step_effect.json").is_file() else None
    fits = recorded_fits(execution)
    interrupted = [row for stem in {f"{item['window']}_st{STEPS[CANDIDATE]}_s{item['seed']}" for item in fits}
                   for row in prior_claims(stem) if not row["complete"]]
    if step is not None:
        conclusion = conclusion_text(decision, step, plan)
    else:
        conclusion = (f"The seed-11 screen rejected the 6000-step candidate "
                      f"({screen['decision']['mean_rank_ic_delta']:+.6f} pooled RankIC, per window "
                      f"{screen['decision']['period_mean_deltas']}, failed "
                      f"{[k for k, v in screen['decision']['checks'].items() if not v]}), so the four remaining "
                      "fits were not spent and this round ends with the 10-day reference retained.")
    document = {
        "schema_version": 1, "experiment_id": EXPERIMENT, "recorded_at_utc": utc_now(),
        "hypothesis": plan["hypothesis"], "only_factor": plan["only_factor"],
        "recipe_fixed": plan["recipe_fixed"], "steps": plan["steps"], "seeds": plan["seeds"],
        "deployment_seed": plan["deployment_seed"],
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
                      "trajectory_equivalence_to_step_1500": trajectory, "screen": screen["decision"],
                      "screen_per_window": screen.get("per_window"),
                      "interrupted_attempts": interrupted},
        "exposure": {window["name"]: window["exposure"] for window in plan["windows"]},
        "results": {"decision": decision, "step_effect": step,
                    "intervals": read_json(execution / "paired_intervals.json")},
        "conclusion": conclusion,
        "retained_version": "10-day seed11 research reference (unchanged)",
        "candidate_status": "not promoted" if decision["result"] != "accept" else "promoted to a confirmation stage",
        "model_promoted": False, "final_views": 0, "platform_verified": False,
        "limitations": [
            "Two research windows and three seeds are selection data, not independent confirmation.",
            "Steps were scaled for the 40-day arm only, so the candidate has four times the optimizer compute "
            "of the retained reference; a 10-day/6000-step arm is required before claiming that longer history "
            "beats recent data.",
            "6000 steps is still only 0.60 epochs of the 40-day pool; heavier schedules are outside this round.",
            "Local gross metrics exclude fees and slippage; platform time and memory limits remain unverified.",
        ],
        "next_steps": [
            "If the candidate passed, register a 10-day/6000-step arm so coverage and compute are separated at "
            "equal epochs.",
            "If research_b still degraded, the next registered treatment is recent-date loss weighting inside "
            "the 40-day pool at a predeclared half-life, not another step count.",
        ],
    }
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Research records are append-only; select a new output name")
    write_json(output_path, document)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "worker", "run", "continue", "record"))
    parser.add_argument("path", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--prior-execution", type=Path)
    parser.add_argument("--measurements-from", type=Path)
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--screen", type=Path, help="passed screen archive to continue")
    args = parser.parse_args()
    if args.command == "prepare":
        print(str(prepare(args.output or PLAN, frozen_path=args.frozen or FROZEN,
                          prior_execution=args.prior_execution or PRIOR_EXECUTION,
                          measurements_from=args.measurements_from or PRIOR_MEASUREMENTS,
                          output_root=args.output_root or RUNS)), flush=True)
    elif args.command == "worker":
        worker(args.path.resolve(), args.output.resolve())
    elif args.command == "run":
        root = new_run_dir(Path(args.output_root or RUNS).resolve())
        print(str(root), flush=True)
        execute(args.path.resolve(), root)
    elif args.command == "continue":
        if args.screen is None:
            parser.error("continue requires --screen <passed screen archive>")
        root = new_run_dir(Path(args.output_root or RUNS).resolve())
        print(str(root), flush=True)
        execute_full(args.path.resolve(), root, args.screen)
    else:
        if args.execution is None or args.output is None:
            parser.error("record requires --execution <completed run> and --output <record.json>")
        print(str(record(args.path.resolve(), args.execution, args.output)), flush=True)


if __name__ == "__main__":
    main()
