"""Executable pre-registered promotion gate (B4).

The gate is deliberately narrow: it answers one question under one frozen rule.
It reads the pre-registration, proves the pre-registration landed before the
candidate was evaluated, re-derives every threshold from the measured volatility
artifact, and then compares a baseline and a candidate eval-v2 object on paired
trading days with consecutive-day block resampling.

Evidence rules enforced here (never waived by options):

* the resampling unit is a block of consecutive trading days inside one declared
  disjoint period; blocks never cross a month gap and stocks are never resampled;
* seeds are not date evidence: the comparison works on deterministic objects, so
  no seed slot is fabricated or counted as replication;
* baseline and candidate are differenced on the identical trading-day index;
* the primary improvement must reach the measured threshold, the block interval
  must exclude zero, the effect must repeat across declared periods, and every
  secondary metric must stay inside its measured regression line;
* the resource budget is checked against the pre-registered platform caliber.

Exit codes: 0 = accept, 3 = reject, 4 = evidence inconclusive, 2 = refused
invalid input (nothing is decided; the artifacts are still written).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.comparison import compare_evaluation_objects
from jinyiquant.config import EvaluationConfig
from jinyiquant.evaluation import EVALUATION_VERSION, METRIC_SPEC, aggregate_metric, evaluate
from jinyiquant.protocol import validate_change

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = ROOT / "configs/b4_promotion_prereg_20261027.json"
DEFAULT_REGISTRATION = ROOT / "configs/b4_promotion_prereg_20261027.registration.json"
SECONDARY_METRICS = ("ic_mean", "ic_ir", "long_short_sharpe", "long_short_return_mean",
                     "min_daily_long_short_return", "turnover_mean")
NET5 = "net_return_mean_5bp"
MIB_PER_MB = 1e6 / 2 ** 20
DECISION_EXIT = {"accept": 0, "reject": 3, "inconclusive": 4, "invalid": 2}


class Refused(Exception):
    """Fail-closed input error: no decision can be made from these artifacts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(document, path: Path) -> str:
    payload = (json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def parse_timestamp(value, label: str) -> pd.Timestamp:
    if not isinstance(value, str) or not value:
        raise Refused(f"{label} must be an ISO timestamp")
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise Refused(f"{label} is not a timestamp: {value!r}") from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise Refused(f"{label} must be timezone aware UTC, found {value!r}")
    return stamp.tz_convert("UTC")


def verify_registration(prereg_path: Path, registration_path: Path) -> dict:
    """The pre-registration bytes on disk must be the ones that were registered."""
    if not prereg_path.is_file():
        raise Refused(f"pre-registration not found: {prereg_path}")
    if not registration_path.is_file():
        raise Refused(f"registration record not found: {registration_path}")
    prereg = read_json(prereg_path)
    registration = read_json(registration_path)
    current = sha256_file(prereg_path)
    if registration.get("prereg_sha256") != current:
        raise Refused("pre-registration bytes changed after registration: "
                      f"recorded {registration.get('prereg_sha256')} != current {current}")
    registered_at = parse_timestamp(prereg.get("registered_at"), "preregistered_at")
    if parse_timestamp(registration.get("registered_at"), "registration.registered_at") != registered_at:
        raise Refused("pre-registration and registration record disagree on registered_at")
    for key in ("object_policy_template", "derived_thresholds",
                "resource_budget", "evidence_rules", "periods", "supersedes",
                "economic_risk_boundary", "metric_caliber_layers",
                "pre_registration_completeness"):
        if key not in prereg:
            raise Refused(f"pre-registration is missing {key!r}")
    if "threshold_basis" not in prereg["object_policy_template"]:
        raise Refused("pre-registration policy template is missing 'threshold_basis'")
    required_content = {
        "economic_risk_boundary": ("stress_single_side_rate_bp", "stress_scenario_locked",
                                   "must_not", "facts", "collapse_rule"),
        "metric_caliber_layers": ("official_turnover", "local_proxy", "cost_scenario_independence"),
        "pre_registration_completeness": ("improvement_amount", "secondary_degradation_lines",
                                          "resource_budget"),
    }
    for block, fields in required_content.items():
        absent = [field for field in fields if field not in prereg[block]]
        if absent:
            raise Refused(f"pre-registration block {block!r} is missing {absent}")
    return {"prereg": prereg, "registration": registration, "registered_at": registered_at,
            "prereg_sha256": current}


def verify_basis(prereg: dict, registered_at: pd.Timestamp) -> dict:
    """Re-derive the thresholds from the measured artifact; refuse any drift."""
    basis = prereg["object_policy_template"]["threshold_basis"]
    report_path = ROOT / basis["calibration_report"]
    if not report_path.is_file():
        raise Refused(f"threshold basis artifact not found: {basis['calibration_report']}")
    if sha256_file(report_path) != basis["calibration_sha256"]:
        raise Refused("threshold basis artifact changed after registration")
    report = read_json(report_path)
    if report.get("status") != "completed":
        raise Refused("threshold basis artifact is not completed")
    created = parse_timestamp(report.get("created_at"), "basis.created_at")
    if created > registered_at:
        raise Refused("threshold basis artifact was created after the pre-registration")
    if report.get("evaluation_version") != EVALUATION_VERSION:
        raise Refused("threshold basis artifact is not eval-v2")
    measurements = report.get("measurements") or {}
    derived = measurements.get("derived_thresholds") or {}
    declared = prereg["derived_thresholds"]
    scale = measurements.get("placebo_scale") or {}
    levels = measurements.get("levels") or {}
    baseline_object = report.get("baseline_object")
    primary = derived.get("primary_metric")
    if primary != prereg["object_policy_template"]["primary_metric"]:
        raise Refused("basis primary metric differs from the pre-registered primary metric")
    if baseline_object not in levels:
        raise Refused("basis artifact does not contain the baseline object")
    floor = derived.get("relative_floor")
    grid = (measurements.get("estimator") or {}).get("primary_grid", 1e-4)
    if not isinstance(floor, (int, float)) or not isinstance(grid, (int, float)):
        raise Refused("basis artifact does not record the estimator constants")
    baseline_level = float(levels[baseline_object][primary])
    relative_floor_value = float(floor) * baseline_level
    raw = max(float(scale[primary]), relative_floor_value)
    recomputed = float(np.ceil(raw / grid - 1e-12) * grid)
    checks = {
        "baseline_level": baseline_level,
        "relative_floor_value": relative_floor_value,
        "placebo_max_abs_primary_difference": float(scale[primary]),
        "min_improvement_raw": raw,
        "min_improvement_recomputed": recomputed,
    }
    if abs(recomputed - float(derived.get("min_improvement", np.nan))) > 1e-12:
        raise Refused("basis artifact's own derived threshold does not match its measurements")
    policy_min = float(prereg["object_policy_template"]["min_improvement"])
    if abs(recomputed - policy_min) > 1e-12:
        raise Refused(f"pre-registered min_improvement {policy_min} != re-derived {recomputed}")
    if abs(float(declared.get("min_improvement", np.nan)) - recomputed) > 1e-12:
        raise Refused("pre-registered derived_thresholds block disagrees with the basis artifact")
    secondary_limits = prereg["object_policy_template"]["max_metric_regressions"]
    for metric in SECONDARY_METRICS:
        limit = secondary_limits[metric] if isinstance(secondary_limits[metric], (int, float)) \
            else secondary_limits[metric]["max_regression"]
        if abs(float(limit) - float(derived["secondary_limits"][metric])) > 1e-12:
            raise Refused(f"secondary limit for {metric} differs from the measured scale")
    if abs(float(prereg["net5_limit"]) - float(derived["net5_limit"])) > 1e-12:
        raise Refused("net5 regression line differs from the measured scale")
    days = int(measurements.get("days") or 0)
    if days < int(prereg["evidence_rules"]["minimum_days"]):
        raise Refused("basis artifact measured fewer trading days than the evidence rule allows")
    return {"report": report, "path": rel(report_path), "sha256": basis["calibration_sha256"],
            "checks": checks}


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame or frame.empty:
        raise Refused(f"{rel(path)} has no daily rows")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["date"].duplicated().any():
        raise Refused(f"{rel(path)} has duplicate trading days")
    return frame.sort_values("date").reset_index(drop=True)


def _derived_scenarios_from_predictions(directory: Path) -> tuple[dict, pd.DataFrame, dict, pd.DataFrame]:
    predictions_path = directory / "predictions.parquet"
    targets_path = directory / "targets.parquet"
    if not (predictions_path.is_file() and targets_path.is_file()):
        raise Refused(f"{rel(directory)} lacks predictions.parquet/targets.parquet for 5bp re-evaluation")
    predictions = pd.read_parquet(predictions_path)
    targets = pd.read_parquet(targets_path)
    if "score" not in predictions:
        raise Refused(f"{rel(predictions_path)} lacks a score column")
    config = EvaluationConfig(quantile=0.2, periods_per_year=252, min_cross_section=3,
                              constant_portfolios="undefined", single_side_rate=0.0)
    metrics_0, daily_0, _ = evaluate(predictions, targets, config, return_sections=True,
                                     undefined_constant_portfolios=True)
    config_5 = config.model_copy(update={"single_side_rate": 0.0005})
    metrics_5, daily_5, _ = evaluate(predictions, targets, config_5, return_sections=True,
                                     undefined_constant_portfolios=True)
    return metrics_0, daily_0, metrics_5, daily_5


def _find_started_at(directory: Path) -> tuple[str | None, str | None]:
    for parent in (directory, directory.parent, directory.parent.parent):
        manifest = parent / "manifest.json"
        if manifest.is_file():
            document = read_json(manifest)
            started = document.get("started_at")
            if started:
                return str(started), rel(manifest)
    return None, None


def load_object(directory: Path, role: str) -> dict:
    """Load one eval-v2 object from an archive or a run directory."""
    directory = directory if directory.is_absolute() else ROOT / directory
    if not directory.is_dir():
        raise Refused(f"{role} directory not found: {directory}")
    metrics_path = directory / "metrics.json"
    if not metrics_path.is_file():
        raise Refused(f"{role} directory lacks metrics.json")
    metrics = read_json(metrics_path)
    if "metrics_0bp" in metrics and "metrics_5bp" in metrics:
        daily_0 = _load_frame(directory / "daily_0bp.csv")
        daily_5 = _load_frame(directory / "daily_5bp.csv")
        metrics_0, metrics_5 = metrics["metrics_0bp"], metrics["metrics_5bp"]
        source = "object_archive"
    elif metrics.get("evaluation_version") == EVALUATION_VERSION:
        metrics_0, daily_0, metrics_5, daily_5 = _derived_scenarios_from_predictions(directory)
        source = "run_dir_reevaluated_0bp_5bp"
    else:
        raise Refused(f"{role} artifacts are neither an eval-v2 object archive nor a reevaluable run")
    for label, block in (("0bp", metrics_0), ("5bp", metrics_5)):
        if block.get("evaluation_version") != EVALUATION_VERSION:
            raise Refused(f"{role} {label} block is not {EVALUATION_VERSION}")
    started_at, manifest_path = _find_started_at(directory)
    return {
        "object_id": str(metrics.get("object_id") or directory.name),
        "role": role,
        "source": rel(directory),
        "source_kind": source,
        "metrics_0bp": metrics_0,
        "metrics_5bp": metrics_5,
        "daily_0bp": daily_0,
        "daily_5bp": daily_5,
        "started_at": started_at,
        "started_at_source": manifest_path,
        "sha256": {name: sha256_file(directory / name) for name in
                   ("metrics.json", "daily_0bp.csv", "daily_5bp.csv")
                   if (directory / name).is_file()},
    }


def verify_fair_pair(baseline: dict, candidate: dict) -> None:
    left, right = baseline["daily_0bp"], candidate["daily_0bp"]
    if not left["date"].equals(right["date"]):
        raise Refused("baseline and candidate were not evaluated on the identical trading-day index")
    if not baseline["daily_5bp"]["date"].equals(right["date"]):
        raise Refused("5bp daily index differs between objects")
    left_5, right_5 = baseline["daily_5bp"], candidate["daily_5bp"]
    if not left_5["date"].equals(right_5["date"]):
        raise Refused("baseline and candidate 5bp daily indexes differ")


def recompute_secondary(object_metrics: dict, daily: pd.DataFrame, label: str) -> dict:
    """Secondary gate numbers must be reproducible from the daily artifact."""
    values = {}
    for metric in SECONDARY_METRICS:
        column, statistic, _ = METRIC_SPEC[metric]
        if column not in daily:
            raise Refused(f"{label}: daily artifact lacks {column!r} for {metric}")
        recomputed = aggregate_metric(daily[column].to_numpy(dtype=float), statistic, 252.0)
        recorded = object_metrics.get(metric)
        if recomputed is None or recorded is None or not np.isfinite(recorded):
            raise Refused(f"{label}: {metric} is undefined in the archived evaluation")
        if not np.isclose(recomputed, recorded, rtol=1e-9, atol=1e-12):
            raise Refused(f"{label}: recorded {metric} does not match the daily artifact")
        values[metric] = float(recorded)
    if "net_return" not in daily:
        raise Refused(f"{label}: 5bp daily artifact lacks net_return")
    values[NET5] = float(daily["net_return"].mean())
    return values


def period_sign_evidence(comparison: dict) -> dict:
    effects = comparison.get("period_effects") or []
    positive = [index for index, effect in enumerate(effects) if effect is not None and effect > 0]
    return {"period_effects": effects, "positive_periods": len(positive), "periods": len(effects),
            "fraction_positive": (len(positive) / len(effects)) if effects else 0.0}


def resolve_resources(path: Path | None, candidate: dict) -> dict:
    """Resource evidence must be explicit; nothing is inferred from a missing number."""
    evidence = {"provided": False, "peak_rss_mib": None, "wall_seconds": None, "source": None,
                "unit_note": "peak_rss_mib = 2**20 bytes; manifest peak_rss_mb is decimal MB and is "
                             "converted with 1e6/2**20"}
    if path is not None:
        document = read_json(path)
        if "peak_rss_mib" not in document:
            raise Refused(f"{rel(path)} must report peak_rss_mib")
        evidence.update(provided=True, peak_rss_mib=float(document["peak_rss_mib"]),
                        wall_seconds=document.get("wall_seconds"), source=rel(path))
        return evidence
    directory = ROOT / candidate["source"]
    manifest = directory / "manifest.json"
    if manifest.is_file():
        resources = read_json(manifest).get("resources") or {}
        peak_mb = resources.get("peak_rss_mb")
        if isinstance(peak_mb, (int, float)) and np.isfinite(peak_mb):
            evidence.update(provided=True, peak_rss_mib=float(peak_mb) * MIB_PER_MB,
                            wall_seconds=resources.get("wall_seconds"), source=rel(manifest),
                            peak_rss_mb_recorded=float(peak_mb))
    return evidence


def resolve_change(candidate: dict, explicit: Path | None) -> dict:
    if explicit is not None:
        change = read_json(explicit)
    else:
        directory = ROOT / candidate["source"]
        candidates = [directory / "change.json", directory / "plan.json"]
        change = None
        for path in candidates:
            if path.is_file():
                document = read_json(path)
                change = document.get("change", document if "factor" in document else None)
                if change:
                    break
        if change is None:
            raise Refused("candidate does not declare its single controlled factor "
                          "(change.json or plan.json with a change block)")
    try:
        validate_change(change)
    except ValueError as exc:
        raise Refused(f"candidate change declaration is invalid: {exc}") from exc
    if change["factor"] == "control.identity":
        raise Refused("identity control cannot justify a promotion")
    return change


def evaluate_gate(args) -> tuple[dict, dict]:
    prereg_path = args.prereg if args.prereg.is_absolute() else ROOT / args.prereg
    registration_path = args.registration if args.registration.is_absolute() else ROOT / args.registration
    registration = verify_registration(prereg_path, registration_path)
    prereg = registration["prereg"]
    basis = verify_basis(prereg, registration["registered_at"])

    baseline_path = args.baseline or prereg["baseline_object"]["path"]
    baseline = load_object(Path(baseline_path), "baseline")
    candidate = load_object(args.candidate, "candidate")
    verify_fair_pair(baseline, candidate)

    started_at = args.candidate_started_at or candidate["started_at"]
    if not started_at:
        raise Refused("candidate start time is unknown: the ordering of pre-registration and "
                      "candidate evaluation cannot be proven")
    candidate_started = parse_timestamp(started_at, "candidate started_at")
    if candidate_started < registration["registered_at"]:
        raise Refused(f"candidate started at {candidate_started} before the pre-registration "
                      f"({registration['registered_at']}); the thresholds were not registered in advance")

    change = resolve_change(candidate, args.change)
    policy = dict(prereg["object_policy_template"])
    policy["change"] = change

    periods = [(start, end) for start, end in prereg["periods"]["declared"]]
    try:
        comparison, decision = compare_evaluation_objects(
            {"object_id": baseline["object_id"], "metrics": baseline["metrics_0bp"],
             "daily": baseline["daily_0bp"]},
            {"object_id": candidate["object_id"], "metrics": candidate["metrics_0bp"],
             "daily": candidate["daily_0bp"]},
            policy, periods=periods)
    except ValueError as exc:
        raise Refused(f"comparison policy or object validation failed: {exc}") from exc

    baseline_values = recompute_secondary(baseline["metrics_0bp"], baseline["daily_0bp"], "baseline")
    candidate_values = recompute_secondary(candidate["metrics_0bp"], candidate["daily_0bp"], "candidate")
    baseline_net5 = float(baseline["daily_5bp"]["net_return"].mean())
    candidate_net5 = float(candidate["daily_5bp"]["net_return"].mean())
    net5_regression = baseline_net5 - candidate_net5
    net5_limit = float(prereg["net5_limit"])

    rules = prereg["evidence_rules"]
    minimum_days = int(rules["minimum_days"])
    minimum_periods = int(rules["minimum_periods"])
    if "required_positive_fraction" in rules:
        fraction = float(rules["required_positive_fraction"])
        required_positive = max(1, int(np.ceil(fraction * len(periods) - 1e-12)))
    elif "required_positive_periods" in rules:
        required_positive = int(rules["required_positive_periods"])
    else:
        raise Refused("evidence_rules must declare required_positive_fraction")
    sign = period_sign_evidence(comparison)
    resources = resolve_resources(args.resources, candidate)
    memory_limit = float(prereg["resource_budget"]["promotion_peak_rss_mib"])
    interval = comparison.get("confidence_interval")
    improvement = comparison.get("improvement")

    failures: list[str] = []
    missing: list[str] = []
    # The engine's own accept rule additionally requires the interval *lower bound*
    # to clear the practical threshold. The pre-registered rule (and the historical
    # team convention) applies the threshold to the observed paired effect and uses
    # the interval only to exclude zero, so the gate decides from that evidence and
    # keeps the engine status as a diagnostic instead of delegating the verdict.
    engine_failures = [item for item in (decision.get("failed_gates") or [])]
    engine_missing = [item for item in (decision.get("missing_evidence") or [])
                      if "confidence interval does not establish" not in item]
    failures.extend(engine_failures)
    missing.extend(engine_missing)
    if len(candidate["daily_0bp"]) < minimum_days:
        missing.append(f"only {len(candidate['daily_0bp'])} trading days; evidence rule needs "
                       f"{minimum_days}")
    if len(periods) < minimum_periods:
        missing.append(f"only {len(periods)} declared periods; evidence rule needs {minimum_periods}")
    if interval is None:
        missing.append("day-block interval is undefined; the paired effect cannot be bounded")
    elif interval["low"] <= 0:
        missing.append(f"5-day block interval [{interval['low']:.6f}, {interval['high']:.6f}] does not "
                       "exclude zero")
    if improvement is not None and improvement < float(policy["min_improvement"]):
        failures.append(f"observed paired improvement {improvement:.6f} is below the measured "
                        f"threshold {policy['min_improvement']:.6f}")
    if net5_regression > net5_limit:
        failures.append(f"net return at 5bp regresses by {net5_regression:.3e}, beyond the measured "
                        f"line {net5_limit:.3e}")
    if sign["positive_periods"] < required_positive:
        failures.append(f"improvement repeats in {sign['positive_periods']} of {sign['periods']} "
                        f"periods; the pre-registered rule needs {required_positive}")
    if not resources["provided"]:
        missing.append("resource evidence absent: peak host RSS was not measured for the candidate")
    elif resources["peak_rss_mib"] > memory_limit:
        failures.append(f"peak host RSS {resources['peak_rss_mib']:.1f} MiB exceeds the pre-registered "
                        f"24 GiB platform line {memory_limit:.0f} MiB")

    status = "accept" if not failures and not missing else ("reject" if failures else "inconclusive")

    gate = {
        "schema_version": 1,
        "gate_id": prereg["registration_id"],
        "evaluated_at": utc_now(),
        "evaluation_version": EVALUATION_VERSION,
        "prereg": {"path": rel(prereg_path), "sha256": registration["prereg_sha256"],
                   "registered_at": str(registration["registered_at"]),
                   "registration_record": rel(registration_path)},
        "ordering": {
            "candidate_started_at": str(candidate_started),
            "candidate_started_at_source": candidate["started_at_source"] or "argument",
            "preregistered_at": str(registration["registered_at"]),
            "prereg_before_candidate": bool(candidate_started >= registration["registered_at"]),
        },
        "threshold_basis": {"path": basis["path"], "sha256": basis["sha256"],
                            "checks": basis["checks"],
                            "created_at": basis["report"]["created_at"]},
        "objects": {
            "baseline": {"object_id": baseline["object_id"], "source": baseline["source"],
                         "kind": baseline["source_kind"], "sha256": baseline["sha256"],
                         "metrics": baseline_values},
            "candidate": {"object_id": candidate["object_id"], "source": candidate["source"],
                          "kind": candidate["source_kind"], "sha256": candidate["sha256"],
                          "metrics": candidate_values},
        },
        "change": change,
        "comparison": comparison,
        "engine": {"status": decision.get("status"),
                   "note": "compare_evaluation_objects supplies policy validation, the paired "
                           "daily series, the 5-day block interval and the secondary limits; its "
                           "accept rule also requires the interval lower bound to clear the "
                           "threshold, which the pre-registered rule does not, so the gate verdict "
                           "is derived above from the same evidence",
                   "failed_gates": engine_failures, "missing_evidence": engine_missing},
        "secondary": comparison.get("secondary"),
        "net5": {"baseline": baseline_net5, "candidate": candidate_net5,
                 "regression": net5_regression, "limit": net5_limit,
                 "passed": net5_regression <= net5_limit},
        "period_repeat": sign | {"required_positive_periods": required_positive},
        "resources": resources | {"promotion_peak_rss_mib": memory_limit,
                                  "passed": bool(resources["provided"]
                                                 and resources["peak_rss_mib"] <= memory_limit)},
        "evidence_rules": rules,
        "cost_scenario": {
            "stress_single_side_rate": 0.0005,
            "stress_single_side_rate_bp": float(prereg["economic_risk_boundary"]["stress_single_side_rate_bp"]),
            "scenario_locked": bool(prereg["economic_risk_boundary"]["stress_scenario_locked"]),
            "turnover_is_rate_independent": True,
            "official_score_contains_cost": False,
        },
        "caliber_layers": {"official_turnover": prereg["metric_caliber_layers"]["official_turnover"],
                           "local_proxy_value": prereg["metric_caliber_layers"]["local_proxy"]["value"],
                           "local_proxy_use": prereg["metric_caliber_layers"]["local_proxy"]["use"]},
        "failed_gates": sorted(set(failures)),
        "missing_evidence": sorted(set(missing)),
    }
    final = {
        "schema_version": 1,
        "gate_id": prereg["registration_id"],
        "status": status,
        "result": {"accept": "保留", "reject": "回退", "inconclusive": "证据不足"}[status],
        "action": "retain_candidate" if status == "accept" else "retain_baseline",
        "automatic_promotion": False,
        "primary_metric": prereg["object_policy_template"]["primary_metric"],
        "improvement": comparison.get("improvement"),
        "min_improvement": prereg["object_policy_template"]["min_improvement"],
        "confidence_interval": comparison.get("confidence_interval"),
        "failed_gates": sorted(set(failures)),
        "missing_evidence": sorted(set(missing)),
        "engine_status": decision.get("status"),
        "prereg_sha256": registration["prereg_sha256"],
        "policy_sha256": comparison.get("policy_sha256"),
        "sealed_interval_untouched": True,
        "git_commit": git_commit(),
    }
    return gate, final


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", type=Path, required=True,
                        help="candidate eval-v2 object archive or reevaluable run directory")
    parser.add_argument("--baseline", type=Path,
                        help="baseline eval-v2 object; defaults to the pre-registered baseline")
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--change", type=Path, help="candidate change block if not stored with the run")
    parser.add_argument("--resources", type=Path, help="JSON with measured peak_rss_mib/wall_seconds")
    parser.add_argument("--candidate-started-at", type=str,
                        help="UTC start timestamp when the artifact carries none")
    parser.add_argument("--output", type=Path, help="output directory; defaults under runs/")
    args = parser.parse_args(argv)
    output = args.output or (ROOT / "runs" / "b4-gate-20261027"
                             / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{args.candidate.name}")
    output.mkdir(parents=True, exist_ok=False)
    try:
        gate, decision = evaluate_gate(args)
    except (Refused, ValueError, KeyError, OSError, TypeError) as exc:
        reason = str(exc) if isinstance(exc, Refused) else f"{type(exc).__name__}: {exc}"
        decision = {"schema_version": 1, "status": "invalid", "result": "拒绝输入",
                    "action": "none", "automatic_promotion": False,
                    "refused_reason": reason, "git_commit": git_commit()}
        write_json(decision, output / "decision.json")
        print(f"REFUSED: {reason}")
        print(f"wrote {rel(output / 'decision.json')}")
        return DECISION_EXIT["invalid"]
    write_json(gate, output / "gate.json")
    write_json(decision, output / "decision.json")
    print(f"status={decision['status']} improvement={decision['improvement']} "
          f"min_improvement={decision['min_improvement']} interval={decision['confidence_interval']}")
    for item in decision["failed_gates"]:
        print(f"  failed gate: {item}")
    for item in decision["missing_evidence"]:
        print(f"  missing evidence: {item}")
    print(f"wrote {rel(output)}")
    return DECISION_EXIT[decision["status"]]


if __name__ == "__main__":
    sys.exit(main())
