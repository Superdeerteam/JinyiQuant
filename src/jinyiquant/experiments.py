"""Baseline calibration and preregistered, bounded seed/period comparison suites."""

from pathlib import Path

import numpy as np

from .checks import verify_artifacts
from .comparison import compare_runs, validate_policy
from .config import Period, RunConfig, load_config
from .evaluation import METRIC_SPEC
from .io import digest, read_json, utc_now, write_json
from .runner import run
from .tracking import new_run_dir


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
    write_json(archive / "plan.json", {"registered_at": utc_now(), "config": config.snapshot(),
                                     "seeds": seeds, "periods": periods})
    paths, code = [], 0
    for cfg in configs:
        path, status = run(cfg)
        paths.append(path)
        if status:
            code = status
            break
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
    return archive, code


def save_comparison(baselines, candidates, policy, destination):
    destination = new_run_dir(destination)
    try:
        comparison, decision = compare_runs(baselines, candidates, policy)
        write_json(destination / "comparison.json", comparison)
        write_json(destination / "decision.json", decision)
        (destination / "report.txt").write_text(
            f"Decision: {decision['status']}\nAction: {decision['action']}\n"
            f"Failed gates: {decision['failed_gates']}\nMissing evidence: {decision['missing_evidence']}\n"
            "No automatic model replacement; original baseline and candidate archives are retained.\n", encoding="utf-8")
        return destination, 0
    except Exception as exc:
        write_json(destination / "comparison.json", {"schema_version": 1, "status": "invalid",
                                                     "reason": str(exc)})
        write_json(destination / "decision.json", {"schema_version": 1, "status": "reject",
                    "accepted": False, "action": "retain_baseline", "reason": str(exc), "automatic_promotion": False})
        return destination, 2


def suite(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    allowed = {"baseline_config", "candidate_config", "seeds", "periods", "policy", "calibration_report"}
    if set(plan) != allowed:
        raise ValueError(f"Suite plan keys must be exactly {sorted(allowed)}")
    baseline = load_config(plan_path.parent / plan["baseline_config"])
    candidate = load_config(plan_path.parent / plan["candidate_config"])
    policy = validate_policy(plan["policy"])
    calibration_path = plan_path.parent / plan["calibration_report"]
    calibration = read_json(calibration_path)
    if calibration.get("status") != "completed":
        raise ValueError("A completed baseline calibration is required before registering comparison thresholds")
    combinations = _grid(plan["seeds"], plan["periods"])
    if len(plan["seeds"]) < policy["min_seeds"] or len(plan["periods"]) < policy["min_periods"]:
        raise ValueError("Suite grid cannot satisfy the declared seed/period minimum")
    # Verify the calibration came from this baseline/data recipe; preserve every source run.
    calibration_runs = [Path(p) for p in calibration["runs"]]
    if not calibration_runs:
        raise ValueError("Calibration has no runs")
    for path in calibration_runs:
        if not verify_artifacts(path)["passed"]:
            raise ValueError("Calibration artifact checksum mismatch")
        old = read_json(path / "config.json")
        for field in ("model", "data", "budget", "evaluation", "acceptance"):
            if old[field] != baseline.snapshot()[field]:
                raise ValueError(f"Calibration baseline differs in {field}")
        if old["splits"]["train"] != baseline.snapshot()["splits"]["train"]:
            raise ValueError("Calibration training period differs")
        if read_json(path / "manifest.json")["status"] != "completed":
            raise ValueError("Calibration contains an unsuccessful run")
    actual_configs = [read_json(p / "config.json") for p in calibration_runs]
    actual_seeds = {c["seed"] for c in actual_configs}
    actual_periods = {digest(c["splits"]["validation"]) for c in actual_configs}
    actual_pairs = {(c["seed"], digest(c["splits"]["validation"])) for c in actual_configs}
    if len(actual_pairs) != len(actual_configs) or len(actual_pairs) != len(actual_seeds) * len(actual_periods):
        raise ValueError("Calibration archives must form a complete unique seed/period grid")
    if len(actual_seeds) < policy["min_seeds"] or len(actual_periods) < policy["min_periods"]:
        raise ValueError("Baseline variation needs the declared seed and period counts")
    variants = [(_variant(baseline, s, p), _variant(candidate, s, p)) for s, p in combinations]
    for b, c in variants:
        for field in ("data", "splits", "evaluation", "budget", "acceptance"):
            if b.snapshot()[field] != c.snapshot()[field]:
                raise ValueError(f"Unfair planned comparison: {field} differs")
    archive = new_run_dir(baseline.output_root / "suites")
    registration = {"comparison_policy_sha256": digest(policy), "comparison_policy_registered_at": utc_now()}
    write_json(archive / "plan.json", {**plan, **registration, "calibration_sha256": digest(calibration),
                                     "max_runs": len(variants) * 4,
                                     "max_worker_wall_seconds": len(variants) * 4 * baseline.budget.wall_seconds})
    baselines, candidates, attempts = [], [], []
    try:
        for b, c in variants:
            for config, arm in ((b, baselines), (c, candidates)):
                original, code = run(config, policy_registration=registration)
                attempts.append(str(original))
                write_json(archive / "attempts.json", attempts)
                if code:
                    raise RuntimeError(f"Suite run failed: {original.name} (exit {code})")
                repeated, code = run(config, reference=original, policy_registration=registration)
                attempts.append(str(repeated))
                write_json(archive / "attempts.json", attempts)
                if code:
                    raise RuntimeError(f"Suite rerun failed: {repeated.name} (exit {code})")
                arm.append(repeated)
        comparison, decision = compare_runs(baselines, candidates, policy)
        comparison["calibration"] = {"path": str(calibration_path), "sha256": digest(calibration)}
        comparison["all_attempts"] = attempts
        write_json(archive / "comparison.json", comparison)
        write_json(archive / "decision.json", decision)
        (archive / "report.txt").write_text(
            f"Decision: {decision['status']}\nAll {len(attempts)} attempts retained.\n"
            "Metrics are local approximations; comparison does not replace an accepted model.\n", encoding="utf-8")
        return archive, 0
    except BaseException as exc:
        write_json(archive / "comparison.json", {"schema_version": 1, "status": "failed", "attempts": attempts,
                                                 "reason": str(exc), "registration": registration})
        write_json(archive / "decision.json", {"schema_version": 1, "status": "reject", "accepted": False,
                    "action": "retain_baseline", "automatic_promotion": False, "reason": str(exc)})
        return archive, 130 if isinstance(exc, KeyboardInterrupt) else 1
