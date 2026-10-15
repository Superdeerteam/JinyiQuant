"""Baseline calibration and preregistered, bounded seed/period comparison suites."""

from pathlib import Path
from time import monotonic

import numpy as np

from .comparison import compare_runs, validate_policy
from .config import Period, RunConfig, load_config
from .evaluation import METRIC_SPEC
from .io import digest, read_json, utc_now, write_json
from .protocol import calibration_evidence, fair_config_pair
from .runner import run
from .tracking import Recorder, capture_source, environment, finish, new_run_dir


def _start_archive(archive, experiment_id, **context):
    recorder = Recorder(archive, experiment_id)
    write_json(archive / "manifest.json", {"schema_version": 1, "run_id": archive.name,
        "experiment_id": experiment_id, "started_at": utc_now(), "status": "running",
        "code": capture_source(archive), "environment": environment()})
    recorder.event("run", "started", **context)
    return recorder


def _variant(config, seed, period):
    raw = config.snapshot()
    raw["seed"] = seed
    raw["splits"]["validation"] = Period.model_validate(period).model_dump(mode="json")
    if raw["evaluation_split"] != "validation":
        raise ValueError("Calibration and comparison must not access the final split")
    return RunConfig.model_validate(raw)


def _grid(seeds, periods):
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Seeds must be nonempty unique nonnegative integers")
    parsed = sorted([Period.model_validate(p) for p in periods], key=lambda p: p.start)
    if not parsed or any(a.end > b.start for a, b in zip(parsed, parsed[1:])):
        raise ValueError("Validation periods must be nonempty and nonoverlapping")
    return [(seed, period.model_dump(mode="json")) for period in parsed for seed in seeds]


def calibrate(config, seeds, periods):
    combinations = _grid(seeds, periods)
    configs = [_variant(config, s, p) for s, p in combinations]
    archive = new_run_dir(config.output_root / "calibrations")
    started = monotonic()
    recorder = _start_archive(archive, config.experiment_id, kind="baseline_calibration")
    write_json(archive / "plan.json", {"registered_at": utc_now(), "config": config.snapshot(),
                                     "seeds": seeds, "periods": periods})
    paths, code = [], 0
    try:
        for cfg in configs:
            path, status = run(cfg)
            recorder.event("calibration", "attempt_completed", child_run=str(path), exit_code=status)
            paths.append(path)
            if status:
                code = status
                break
    except BaseException as exc:
        recorder.failure("calibration", exc)
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        write_json(archive / "calibration.json", {"schema_version": 1, "status": "failed",
            "created_at": utc_now(), "runs": [str(p) for p in paths], "reason": str(exc)})
        finish(archive, "interrupted" if code == 130 else "failed", code, error=str(exc),
               resources={"wall_seconds": monotonic() - started})
        return archive, code
    values = {metric: [] for metric in METRIC_SPEC}
    for path in paths:
        if read_json(path / "manifest.json")["status"] == "completed":
            metrics = read_json(path / "metrics.json")
            for metric in values:
                if metrics.get(metric) is not None:
                    values[metric].append(metrics[metric])
    summary = {metric: {"n": len(v), "mean": float(np.mean(v)) if v else None,
                       "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                       "min": min(v) if v else None, "max": max(v) if v else None}
               for metric, v in values.items()}
    report = {"schema_version": 1, "status": "completed" if code == 0 else "failed", "created_at": utc_now(),
              "runs": [str(p) for p in paths], "seeds": seeds, "periods": periods, "metrics": summary,
              "scope": "Baseline variation across declared seeds/time periods, not candidate evidence",
              "next_step": "Declare practical threshold and regression/cost limits before candidate runs"}
    write_json(archive / "calibration.json", report)
    recorder.event("evaluation", "completed", status=report["status"], metrics=summary)
    finish(archive, report["status"], code, resources={"wall_seconds": monotonic() - started})
    return archive, code


def save_comparison(baselines, candidates, policy, destination):
    destination = new_run_dir(destination)
    started = monotonic()
    recorder = _start_archive(destination, "paired-comparison", kind="comparison")
    try:
        comparison, decision = compare_runs(baselines, candidates, policy)
        write_json(destination / "comparison.json", comparison)
        write_json(destination / "decision.json", decision)
        (destination / "report.txt").write_text(
            f"Decision: {decision['status']}\nAction: {decision['action']}\n"
            f"Failed gates: {decision['failed_gates']}\nMissing evidence: {decision['missing_evidence']}\n"
            "No automatic model replacement; original baseline and candidate archives are retained.\n", encoding="utf-8")
        recorder.event("evaluation", "completed", result=decision["result"])
        finish(destination, "completed", 0, resources={"wall_seconds": monotonic() - started})
        return destination, 0
    except Exception as exc:
        recorder.failure("comparison", exc)
        write_json(destination / "comparison.json", {"schema_version": 1, "status": "invalid",
                                                     "reason": str(exc)})
        write_json(destination / "decision.json", {"schema_version": 1, "status": "reject",
                    "result": "回退",
                    "accepted": False, "action": "retain_baseline", "reason": str(exc), "automatic_promotion": False})
        finish(destination, "failed", 2, error=str(exc), resources={"wall_seconds": monotonic() - started})
        return destination, 2


def suite(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    allowed = {"baseline_config", "candidate_config", "seeds", "periods", "policy", "calibration_report"}
    if set(plan) != allowed:
        raise ValueError(f"Suite plan keys must be exactly {sorted(allowed)}")
    baseline = load_config(plan_path.parent / plan["baseline_config"])
    candidate = load_config(plan_path.parent / plan["candidate_config"])
    policy_raw = validate_policy(plan["policy"])
    policy_raw["threshold_basis"]["calibration_report"] = str(
        (plan_path.parent / policy_raw["threshold_basis"]["calibration_report"]).resolve())
    policy = validate_policy(policy_raw)
    calibration_path = (plan_path.parent / plan["calibration_report"]).resolve()
    if calibration_path != Path(policy["threshold_basis"]["calibration_report"]):
        raise ValueError("Suite calibration must match threshold_basis.calibration_report")
    calibration = read_json(calibration_path)
    if calibration.get("status") != "completed":
        raise ValueError("A completed baseline calibration is required before registering comparison thresholds")
    combinations = _grid(plan["seeds"], plan["periods"])
    if len(plan["seeds"]) < policy["min_seeds"] or len(plan["periods"]) < policy["min_periods"]:
        raise ValueError("Suite grid cannot satisfy the declared seed/period minimum")
    variants = [(_variant(baseline, s, p), _variant(candidate, s, p)) for s, p in combinations]
    for b, c in variants:
        fair_config_pair(b.snapshot(), c.snapshot(), policy["change"])
    registered_at = utc_now()
    calibrated = calibration_evidence(policy, baseline.snapshot(), registered_at=registered_at)
    archive = new_run_dir(baseline.output_root / "suites")
    started = monotonic()
    recorder = _start_archive(archive, baseline.experiment_id, kind="single_factor_suite")
    registration = {"comparison_policy_sha256": digest(policy), "comparison_policy_registered_at": registered_at}
    write_json(archive / "plan.json", {**plan, "policy": policy, **registration,
                                     "baseline_snapshot": baseline.snapshot(),
                                     "candidate_snapshot": candidate.snapshot(), "calibration": calibrated,
                                     "calibration_sha256": digest(calibration),
                                     "max_runs": len(variants) * 4,
                                     "max_worker_wall_seconds": len(variants) * 4 * baseline.budget.wall_seconds})
    baselines, candidates, attempts = [], [], []
    recorder.event("registration", "completed", **registration, change=policy["change"],
                   threshold_basis=policy["threshold_basis"], calibration=calibrated)
    try:
        for b, c in variants:
            for config, arm in ((b, baselines), (c, candidates)):
                original, code = run(config, policy_registration=registration)
                attempts.append(str(original))
                write_json(archive / "attempts.json", attempts)
                recorder.event("experiment", "attempt_completed", child_run=str(original), exit_code=code)
                if code:
                    raise RuntimeError(f"Suite run failed: {original.name} (exit {code})")
                repeated, code = run(config, reference=original, policy_registration=registration)
                attempts.append(str(repeated))
                write_json(archive / "attempts.json", attempts)
                recorder.event("reproducibility", "attempt_completed", child_run=str(repeated), exit_code=code)
                if code:
                    raise RuntimeError(f"Suite rerun failed: {repeated.name} (exit {code})")
                arm.append(repeated)
        comparison, decision = compare_runs(baselines, candidates, policy)
        comparison["all_attempts"] = attempts
        write_json(archive / "comparison.json", comparison)
        write_json(archive / "decision.json", decision)
        (archive / "report.txt").write_text(
            f"Decision: {decision['status']}\nAll {len(attempts)} attempts retained.\n"
            "Metrics are local approximations; comparison does not replace an accepted model.\n", encoding="utf-8")
        recorder.event("evaluation", "completed", result=decision["result"], attempts=len(attempts))
        finish(archive, "completed", 0, resources={"wall_seconds": monotonic() - started})
        return archive, 0
    except BaseException as exc:
        recorder.failure("experiment", exc)
        write_json(archive / "comparison.json", {"schema_version": 1, "status": "failed", "attempts": attempts,
                                                 "reason": str(exc), "registration": registration})
        interrupted = isinstance(exc, KeyboardInterrupt)
        write_json(archive / "decision.json", {"schema_version": 1,
                    "status": "inconclusive" if interrupted else "reject", "accepted": False,
                    "result": "证据不足" if interrupted else "回退",
                    "action": "retain_baseline", "automatic_promotion": False, "reason": str(exc)})
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        finish(archive, "interrupted" if code == 130 else "failed", code, error=str(exc),
               resources={"wall_seconds": monotonic() - started})
        return archive, code
