"""Fair paired comparisons with synchronized moving-day-block uncertainty.

Comparison is a decision report, not a model deployment/promotion operation.
Policy must be registered before every compared run. Secondary regression limits
are absolute metric units; a scalar uses METRIC_SPEC's documented direction, or
an object specifies {max_regression, direction}. Independent periods must not
overlap. Seeds share each bootstrap day draw; stocks are never resampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import METRIC_SPEC, aggregate_metric
from .io import digest, file_sha256, read_json
from .protocol import (
    calibration_evidence,
    fair_config_pair,
    fair_data_pair,
    recipe,
    validate_change,
    validate_threshold_basis,
    without_locations,
)

_CORRELATION_COVERAGE_FIELDS = {
    "eligible_section_count", "valid_section_count", "eligible_sections_sha256", "valid_sections_sha256",
}


def validate_policy(policy: dict) -> dict:
    required = {
        "primary_metric", "direction", "min_improvement", "max_metric_regressions",
        "max_wall_ratio", "max_memory_ratio", "min_seeds", "min_periods",
        "bootstrap_samples", "block_days", "confidence", "seed", "threshold_rationale",
        "change", "threshold_basis",
    }
    if set(policy) != required:
        raise ValueError(f"comparison policy keys must be exactly {sorted(required)}")
    validate_change(policy["change"])
    validate_threshold_basis(policy)
    if policy["primary_metric"] not in METRIC_SPEC:
        raise ValueError("primary_metric must be a supported daily-derived metric")
    if policy["direction"] not in {"maximize", "minimize"}:
        raise ValueError("direction must be maximize or minimize")
    for key in ("min_improvement", "max_wall_ratio", "max_memory_ratio", "confidence"):
        value = policy[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{key} must be finite numeric")
    if policy["min_improvement"] < 0 or min(policy["max_wall_ratio"], policy["max_memory_ratio"]) <= 0:
        raise ValueError("improvement must be nonnegative and cost limits positive")
    if not 0 < policy["confidence"] < 1:
        raise ValueError("confidence must lie in (0, 1)")
    for key, minimum in (("min_seeds", 2), ("min_periods", 2), ("bootstrap_samples", 100), ("block_days", 1), ("seed", 0)):
        if type(policy[key]) is not int or policy[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if not isinstance(policy["threshold_rationale"], str) or not policy["threshold_rationale"].strip():
        raise ValueError("threshold_rationale must describe baseline variation and practical relevance")
    limits = policy["max_metric_regressions"]
    if not isinstance(limits, dict) or not limits:
        raise ValueError("max_metric_regressions must declare at least one secondary metric")
    for metric, specification in limits.items():
        if metric not in METRIC_SPEC:
            raise ValueError(f"unsupported secondary metric: {metric}")
        _secondary_limit(metric, specification)
    return dict(policy)


def _secondary_limit(metric: str, specification: Any) -> tuple[float, str]:
    if isinstance(specification, dict):
        if set(specification) != {"max_regression", "direction"}:
            raise ValueError("secondary metric specification needs max_regression and direction")
        limit, direction = specification["max_regression"], specification["direction"]
    else:
        limit, direction = specification, METRIC_SPEC[metric][2]
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not np.isfinite(limit) or limit < 0:
        raise ValueError("secondary regression limits must be nonnegative finite numbers")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("secondary direction must be maximize or minimize")
    return float(limit), direction


@dataclass
class _Run:
    path: Path
    manifest: dict
    config: dict
    data: dict
    metrics: dict
    checks: dict
    daily: pd.DataFrame

    @property
    def period(self) -> tuple[str, str]:
        split = self.config["splits"]["validation"]
        return str(split["start"]), str(split["end"])

    @property
    def pair_key(self) -> tuple[int, str, str]:
        return int(self.config["seed"]), *self.period


def _read_run(path: str | Path) -> _Run:
    path = Path(path)
    documents = [read_json(path / f"{name}.json") for name in ("manifest", "config", "data", "metrics", "checks")]
    daily = pd.read_csv(path / "daily.csv")
    if "date" not in daily or daily.empty:
        raise ValueError(f"{path}: daily results require nonempty dates")
    daily["date"] = pd.to_datetime(daily["date"], errors="raise")
    if daily["date"].isna().any() or daily["date"].duplicated().any():
        raise ValueError(f"{path}: duplicate/missing daily dates")
    if not (daily["date"] == daily["date"].dt.normalize()).all():
        raise ValueError(f"{path}: daily dates must be normalized")
    daily = daily.sort_values("date").set_index("date")
    run = _Run(path, *documents, daily)
    if run.config.get("evaluation_split") != "validation":
        raise ValueError("final-check results cannot be used for comparison or model selection")
    start, end = map(pd.Timestamp, run.period)
    if ((daily.index < start.normalize()) | (daily.index >= end)).any():
        raise ValueError(f"{path}: daily results outside validation period")
    return run


def _fair_pair(baseline: _Run, candidate: _Run, change: dict) -> None:
    fair_config_pair(baseline.config, candidate.config, change)
    fair_data_pair(baseline.data, candidate.data, change)
    if not baseline.daily.index.equals(candidate.daily.index):
        raise ValueError("unfair comparison: daily dates do not align exactly")
    for key in ("sample_count", "cross_section_count"):
        if key in baseline.daily or key in candidate.daily:
            if key not in baseline.daily or key not in candidate.daily or not baseline.daily[key].equals(candidate.daily[key]):
                raise ValueError(f"unfair comparison: differing daily {key}")
    for key in ("sample_keys_sha256", "targets_sha256"):
        if baseline.metrics.get(key) != candidate.metrics.get(key):
            raise ValueError(f"unfair comparison: differing evaluated {key}")


def _run_evidence(run: _Run, policy_hash: str) -> tuple[list[str], list[str]]:
    missing, failures = [], []
    run_id = str(run.manifest.get("run_id", run.path.name))
    missing.extend(_artifact_evidence(run))
    if run.manifest.get("status") not in {"completed", "succeeded"}:
        failures.append(f"{run_id}: run did not complete")
    if run.checks.get("passed") is False:
        failures.append(f"{run_id}: acceptance checks failed")
    elif run.checks.get("passed") is not True:
        missing.append(f"{run_id}: acceptance checks evidence absent")
    checks = run.checks.get("checks", [])
    if not checks or any(item.get("passed") is None for item in checks):
        missing.append(f"{run_id}: individual acceptance checks evidence absent")
    if any(item.get("passed") is False for item in checks):
        failures.append(f"{run_id}: individual acceptance checks failed")
    registered = run.manifest.get("comparison_policy_registered_at")
    started = run.manifest.get("started_at")
    valid_time = False
    try:
        registered_time, started_time = pd.Timestamp(registered), pd.Timestamp(started)
        valid_time = (registered_time.tzinfo is not None and started_time.tzinfo is not None
                      and registered_time <= started_time)
    except (ValueError, TypeError):
        pass
    if run.manifest.get("comparison_policy_sha256") != policy_hash or not valid_time:
        missing.append(f"{run_id}: policy not registered before this run")
    repro = run.manifest.get("reproducibility", {})
    acceptance = run.config.get("acceptance", {})
    if not repro or repro.get("passed") is not True:
        missing.append(f"{run_id}: passing rerun evidence absent")
    elif (not repro.get("reference_run_id") or repro["reference_run_id"] == run_id
          or repro.get("atol") != acceptance.get("repro_atol")
          or repro.get("rtol") != acceptance.get("repro_rtol")):
        missing.append(f"{run_id}: rerun reference or preregistered tolerances missing/mismatched")
    else:
        reference_missing, reference_failures = _reference_evidence(run)
        missing.extend(reference_missing)
        failures.extend(reference_failures)
    return missing, failures


def _artifact_evidence(run: _Run) -> list[str]:
    missing = []
    inventory = run.manifest.get("artifacts")
    if not isinstance(inventory, list) or not inventory:
        return [f"{run.path.name}: artifact checksum inventory absent"]
    required = {"config.json", "config.sha256", "data.json", "metrics.json", "daily.csv",
                "checks.json", "predictions.parquet", "source.zip"}
    listed = set()
    for item in inventory:
        relative = item.get("path", "")
        path = (run.path / relative).resolve()
        if not relative or not path.is_relative_to(run.path.resolve()) or relative in listed:
            raise ValueError(f"{run.path}: invalid/duplicate artifact path")
        listed.add(relative)
        if (not path.is_file() or item.get("sha256") != file_sha256(path)
                or item.get("bytes") != path.stat().st_size):
            raise ValueError(f"{run.path}: artifact checksum/size mismatch: {relative}")
    if required - listed:
        missing.append(f"{run.path.name}: required artifact inventory incomplete: {sorted(required - listed)}")
    config_hash = digest(run.config)
    checksum_path = run.path / "config.sha256"
    if run.manifest.get("config_sha256") is None or not checksum_path.is_file():
        missing.append(f"{run.path.name}: configuration checksum evidence absent")
    elif run.manifest["config_sha256"] != config_hash or checksum_path.read_text(encoding="utf-8").strip() != config_hash:
        raise ValueError(f"{run.path}: configuration checksum mismatch")
    code = run.manifest.get("code", {})
    if not code.get("content_sha256") or not code.get("files"):
        missing.append(f"{run.path.name}: source content evidence absent")
    elif digest(code["files"]) != code["content_sha256"]:
        raise ValueError(f"{run.path}: source content checksum mismatch")
    if (code.get("snapshot_sha256") and (run.path / "source.zip").is_file()
            and file_sha256(run.path / "source.zip") != code["snapshot_sha256"]):
        raise ValueError(f"{run.path}: source snapshot checksum mismatch")
    for key in ("sample_keys_sha256", "targets_sha256"):
        if not run.metrics.get(key):
            missing.append(f"{run.path.name}: evaluated sample identity absent: {key}")
    return missing


def _numeric_equal(left: Any, right: Any, atol: float, rtol: float) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_numeric_equal(left[key], right[key], atol, rtol) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_numeric_equal(a, b, atol, rtol) for a, b in zip(left, right))
    if _finite(left) and _finite(right):
        return bool(np.isclose(left, right, atol=atol, rtol=rtol))
    return left == right


def _reference_evidence(run: _Run) -> tuple[list[str], list[str]]:
    evidence = run.manifest["reproducibility"]
    reference_id = evidence["reference_run_id"]
    reference_path = (run.path.parent / reference_id).resolve()
    if reference_path.parent != run.path.parent.resolve() or reference_path == run.path.resolve():
        raise ValueError("rerun reference must be a distinct sibling run directory")
    if not reference_path.is_dir():
        return [f"{run.path.name}: actual reference run archive absent"], []
    try:
        reference = _read_run(reference_path)
    except FileNotFoundError:
        return [f"{run.path.name}: reference archive incomplete"], []
    missing = _artifact_evidence(reference)
    failures = []
    if reference.manifest.get("run_id") != reference_id:
        raise ValueError("reference manifest run_id differs from its directory")
    if reference.manifest.get("status") != "completed" or reference.checks.get("passed") is not True:
        failures.append(f"{run.path.name}: reference run did not complete with passing checks")
    if reference.config != run.config or reference.data != run.data:
        raise ValueError("rerun reference differs in configuration or data")
    if reference.manifest.get("code", {}).get("content_sha256") != run.manifest.get("code", {}).get("content_sha256"):
        raise ValueError("rerun reference uses a different software implementation")
    a, r = evidence["atol"], evidence["rtol"]
    if not _numeric_equal(reference.metrics, run.metrics, a, r):
        failures.append(f"{run.path.name}: actual rerun metrics exceed preregistered tolerance")
    if not (run.path / "predictions.parquet").is_file() or not (reference.path / "predictions.parquet").is_file():
        missing.append(f"{run.path.name}: actual rerun prediction evidence absent")
    else:
        columns = ["date", "instrument"]
        frames = [pd.read_parquet(path / "predictions.parquet").sort_values(columns).reset_index(drop=True)
                  for path in (reference.path, run.path)]
        for frame in frames:
            if (set(frame.columns) != {*columns, "score"} or frame.empty
                    or frame.duplicated(columns).any() or frame.isna().any().any()
                    or not np.isfinite(frame["score"]).all()):
                raise ValueError("invalid archived rerun predictions")
        equal = frames[0][columns].equals(frames[1][columns])
        if not equal or not np.allclose(frames[0]["score"], frames[1]["score"], atol=a, rtol=r):
            failures.append(f"{run.path.name}: actual rerun predictions exceed preregistered tolerance")
    return missing, failures


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and bool(np.isfinite(value))


def _correlation_evidence(run: _Run, columns: set[str], *, candidate: bool) -> tuple[list[str], list[str]]:
    """Require complete target-defined populations for every policy correlation."""
    missing, failures = [], []
    coverage = run.metrics.get("correlation_coverage")
    if (not isinstance(coverage, dict) or type(coverage.get("schema_version")) is not int
            or coverage["schema_version"] != 1):
        return [f"{run.path.name}: correlation coverage evidence absent/unsupported"], []
    for column in sorted(columns):
        entry = coverage.get(column)
        if not isinstance(entry, dict) or not _CORRELATION_COVERAGE_FIELDS.issubset(entry):
            missing.append(f"{run.path.name}: {column} correlation coverage evidence incomplete")
            continue
        eligible, valid = entry["eligible_section_count"], entry["valid_section_count"]
        hashes = [entry[name] for name in ("eligible_sections_sha256", "valid_sections_sha256")]
        if (type(eligible) is not int or type(valid) is not int or not 0 <= valid <= eligible
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in "0123456789abcdef" for char in value) for value in hashes)):
            raise ValueError(f"{run.path}: malformed {column} correlation coverage")
        count_columns = [f"eligible_{column}_count", f"valid_{column}_count", "cross_section_count"]
        if not set(count_columns).issubset(run.daily):
            missing.append(f"{run.path.name}: {column} daily correlation coverage evidence absent")
            continue
        counts = run.daily[count_columns].to_numpy(dtype=float)
        if (not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any()
                or (counts[:, 1] > counts[:, 0]).any() or (counts[:, 0] > counts[:, 2]).any()
                or counts[:, 0].sum() != eligible or counts[:, 1].sum() != valid):
            raise ValueError(f"{run.path}: {column} correlation coverage differs from daily counts")
        if column in run.daily and not np.array_equal(np.isfinite(run.daily[column]), counts[:, 1] > 0):
            raise ValueError(f"{run.path}: {column} daily values contradict valid correlation counts")
        if eligible == 0:
            missing.append(f"{run.path.name}: no target-eligible {column} correlation sections")
        elif valid != eligible or hashes[0] != hashes[1]:
            reason = f"{run.path.name}: incomplete {column} correlation coverage ({valid}/{eligible} eligible sections)"
            (failures if candidate else missing).append(reason)
    return missing, failures


def _fair_correlation_population(baseline: _Run, candidate: _Run, columns: set[str]) -> None:
    coverages = [run.metrics.get("correlation_coverage") for run in (baseline, candidate)]
    if not all(isinstance(coverage, dict) and type(coverage.get("schema_version")) is int
               and coverage["schema_version"] == 1 for coverage in coverages):
        return  # Missing/unsupported evidence is handled by the decision gate.
    for column in columns:
        entries = [coverage.get(column) for coverage in coverages]
        if all(isinstance(entry, dict) and _CORRELATION_COVERAGE_FIELDS.issubset(entry) for entry in entries):
            identities = [(entry.get("eligible_section_count"), entry["eligible_sections_sha256"]) for entry in entries]
            if identities[0] != identities[1]:
                raise ValueError(f"unfair comparison: differing target-eligible {column} correlation sections")
            count = f"eligible_{column}_count"
            if (count in baseline.daily and count in candidate.daily
                    and not np.array_equal(baseline.daily[count], candidate.daily[count])):
                raise ValueError(f"unfair comparison: differing daily target-eligible {column} correlation counts")


def _metric(run: _Run, metric: str, indices: np.ndarray | None = None) -> float | None:
    column, method, _ = METRIC_SPEC[metric]
    if column not in run.daily:
        return None
    values = run.daily[column].to_numpy(dtype=float)
    if indices is not None:
        values = values[indices]
    return aggregate_metric(values, method, run.config["evaluation"]["periods_per_year"])


def _block_indices(length: int, block_days: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, length - block_days + 1, size=int(np.ceil(length / block_days)))
    return (starts[:, None] + np.arange(block_days)).ravel()[:length]


def compare_runs(baseline_dirs: list[str | Path], candidate_dirs: list[str | Path], policy: dict) -> tuple[dict, dict]:
    """Compute comparison and decision; input artifacts are never modified.

    Malformed or unfair input raises ValueError. Valid but incomplete evidence
    returns inconclusive. A failed safety/cost/regression gate returns reject.
    """
    policy = validate_policy(policy)
    if not baseline_dirs or not candidate_dirs:
        raise ValueError("baseline and candidate runs must both be nonempty")
    baselines = [_read_run(path) for path in baseline_dirs]
    candidates = [_read_run(path) for path in candidate_dirs]
    identifiers = [run.manifest.get("run_id") for run in baselines + candidates]
    if None in identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("all baseline/candidate run IDs must be present and unique")
    left, right = {run.pair_key: run for run in baselines}, {run.pair_key: run for run in candidates}
    if len(left) != len(baselines) or len(right) != len(candidates):
        raise ValueError("duplicate seed/period pairs: provide one verified rerun per arm and pair")
    if left.keys() != right.keys():
        raise ValueError("baseline and candidate seed/validation-period pairs must match exactly")
    pairs = [(left[key], right[key]) for key in sorted(left)]
    for baseline, candidate in pairs:
        _fair_pair(baseline, candidate, policy["change"])
    periods = sorted({run.period for run in baselines}, key=lambda period: pd.Timestamp(period[0]))
    for prior, following in zip(periods, periods[1:]):
        if pd.Timestamp(prior[1]) > pd.Timestamp(following[0]):
            raise ValueError("validation periods must be nonoverlapping; overlapping windows are not independent periods")
    for arm in (baselines, candidates):
        if len({digest(recipe(run.config, across_grid=True)) for run in arm}) != 1:
            raise ValueError("an arm must use one fixed configuration across seeds and periods")
    missing: list[str] = []
    failures: list[str] = []
    policy_hash = digest(policy)
    calibration = None
    try:
        registration_times = [pd.Timestamp(run.manifest.get("comparison_policy_registered_at"))
                              for run in baselines + candidates]
        if all(not pd.isna(t) and t.tzinfo is not None for t in registration_times):
            calibration = calibration_evidence(policy, baselines[0].config, registered_at=min(registration_times),
                baseline_data={digest(run.config["splits"]["validation"]): run.data for run in baselines})
        else:
            missing.append("calibration registration timestamps absent")
    except (OSError, KeyError, ValueError, TypeError) as exc:
        missing.append(f"verified baseline calibration unavailable: {exc}")
    for run in baselines + candidates:
        run_missing, run_failures = _run_evidence(run, policy_hash)
        missing.extend(run_missing)
        failures.extend(run_failures)
    correlation_columns = {METRIC_SPEC[metric][0] for metric in {policy["primary_metric"], *policy["max_metric_regressions"]}
                           if METRIC_SPEC[metric][0] in {"rank_ic", "ic"}}
    if correlation_columns:
        for arm, is_candidate in ((baselines, False), (candidates, True)):
            for run in arm:
                run_missing, run_failures = _correlation_evidence(run, correlation_columns, candidate=is_candidate)
                missing.extend(run_missing)
                failures.extend(run_failures)
        for baseline, candidate in pairs:
            _fair_correlation_population(baseline, candidate, correlation_columns)
    sources = {run.manifest.get("code", {}).get("content_sha256") for run in baselines + candidates}
    if len(sources - {None}) > 1:
        raise ValueError("unfair comparison: differing software implementations; rerun both arms with one source version")
    seeds = sorted({key[0] for key in left})
    if len(seeds) < policy["min_seeds"]:
        missing.append(f"need {policy['min_seeds']} seeds; observed {len(seeds)}")
    if len(periods) < policy["min_periods"]:
        missing.append(f"need {policy['min_periods']} nonoverlapping periods; observed {len(periods)}")
    if len(pairs) != len(seeds) * len(periods):
        missing.append("seed-by-period grid incomplete")
    primary = policy["primary_metric"]
    sign = 1 if policy["direction"] == "maximize" else -1
    pair_results = []
    for baseline, candidate in pairs:
        summary = {"seed": baseline.pair_key[0], "period": list(baseline.period),
                   "baseline_run_id": baseline.manifest.get("run_id"),
                   "candidate_run_id": candidate.manifest.get("run_id"),
                   "baseline_path": str(baseline.path), "candidate_path": str(candidate.path),
                   "baseline": None, "candidate": None, "improvement": None, "secondary": {}, "cost": {}}
        for metric in {primary, *policy["max_metric_regressions"]}:
            values = [_metric(run, metric) for run in (baseline, candidate)]
            for run, value in zip((baseline, candidate), values):
                recorded = run.metrics.get(metric)
                if value is None or not _finite(recorded):
                    missing.append(f"{run.path.name}: undefined/missing metric {metric}")
                elif not np.isclose(value, recorded, rtol=1e-9, atol=1e-12):
                    raise ValueError(f"{run.path}: recorded {metric} does not match daily artifact")
            if None in values:
                continue
            bvalue, cvalue = values
            if metric == primary:
                summary.update(baseline=bvalue, candidate=cvalue, improvement=sign * (cvalue - bvalue))
            if metric in policy["max_metric_regressions"]:
                limit, direction = _secondary_limit(metric, policy["max_metric_regressions"][metric])
                regression = (bvalue - cvalue) * (1 if direction == "maximize" else -1)
                summary["secondary"][metric] = {"regression": regression, "limit": limit, "passed": regression <= limit}
                if regression > limit:
                    failures.append(f"seed {baseline.pair_key[0]} period {baseline.period}: {metric} regression exceeds limit")
        for resource, limit_key in (("wall_seconds", "max_wall_ratio"), ("peak_rss_mb", "max_memory_ratio")):
            bvalue = baseline.manifest.get("resources", {}).get(resource)
            cvalue = candidate.manifest.get("resources", {}).get(resource)
            if not _finite(bvalue) or not _finite(cvalue) or bvalue <= 0 or cvalue <= 0:
                missing.append(f"seed {baseline.pair_key[0]}: valid resource evidence absent for {resource}")
                continue
            ratio = cvalue / bvalue
            summary["cost"][resource] = {"ratio": ratio, "limit": policy[limit_key], "passed": ratio <= policy[limit_key]}
            if ratio > policy[limit_key]:
                failures.append(f"seed {baseline.pair_key[0]} period {baseline.period}: {resource} ratio exceeds limit")
        pair_results.append(summary)
    primary_column = METRIC_SPEC[primary][0]
    bootstrap_ready = all(item["improvement"] is not None for item in pair_results)
    for period in periods:
        period_runs = [run for run in baselines + candidates if run.period == period]
        if any(not period_runs[0].daily.index.equals(run.daily.index) for run in period_runs[1:]):
            raise ValueError("all seeds in one period must have identical daily dates")
        for arm in (baselines, candidates):
            identities = {digest(without_locations(run.data)) for run in arm if run.period == period}
            if len(identities) != 1:
                raise ValueError("unfair comparison: data snapshot changed across seeds in one period")
        for field in ("sample_keys_sha256", "targets_sha256"):
            if len({run.metrics.get(field) for run in period_runs}) != 1:
                raise ValueError(f"unfair comparison: evaluated {field} changed across seeds")
        if len(period_runs[0].daily) < 2 * policy["block_days"]:
            missing.append(f"period {period}: need at least two complete temporal blocks")
            bootstrap_ready = False
        for run in period_runs:
            if primary_column not in run.daily or not np.isfinite(run.daily[primary_column]).all():
                missing.append(f"{run.path.name}: primary daily metric has undefined/nonfinite days")
                bootstrap_ready = False
    interval = None
    observed = [item["improvement"] for item in pair_results if item["improvement"] is not None]
    effect = float(np.mean(observed)) if len(observed) == len(pairs) else None
    if bootstrap_ready:
        rng = np.random.default_rng(policy["seed"])
        draws = []
        for _ in range(policy["bootstrap_samples"]):
            indices = {period: _block_indices(len(next(run for run in baselines if run.period == period).daily),
                                             policy["block_days"], rng) for period in periods}
            effects = []
            for baseline, candidate in pairs:
                sampled = indices[baseline.period]
                bvalue, cvalue = _metric(baseline, primary, sampled), _metric(candidate, primary, sampled)
                if bvalue is None or cvalue is None:
                    break
                effects.append(sign * (cvalue - bvalue))
            if len(effects) == len(pairs):
                draws.append(float(np.mean(effects)))
        if len(draws) != policy["bootstrap_samples"]:
            missing.append("bootstrap contains undefined statistics; interval cannot justify acceptance")
        else:
            alpha = (1 - policy["confidence"]) / 2
            low, high = np.quantile(draws, [alpha, 1 - alpha])
            interval = {"low": float(low), "high": float(high), "confidence": policy["confidence"]}
    stability = all(item["improvement"] is not None and item["improvement"] >= 0 for item in pair_results)
    if not missing and observed and not stability and len(observed) == len(pairs):
        failures.append("primary improvement is negative in at least one seed/period pair")
    if not missing and effect is not None and effect < policy["min_improvement"]:
        failures.append("observed primary improvement below preregistered practical threshold")
    if not missing and policy["change"]["factor"] == "control.identity":
        failures.append("identity control cannot justify candidate promotion")
    status = "reject" if failures else "inconclusive"
    if not failures and not missing and interval is not None and interval["low"] > 0 and interval["low"] >= policy["min_improvement"]:
        status = "accept"
    elif not failures and not missing:
        missing.append("confidence interval does not establish the preregistered improvement")
    baseline_values = [item["baseline"] for item in pair_results if item["baseline"] is not None]
    comparison = {
        "schema_version": 1, "metric_scope": "local_approximation", "policy": policy,
        "calibration": calibration, "change": policy["change"],
        "policy_sha256": policy_hash, "primary_metric": primary, "direction": policy["direction"],
        "pairs": pair_results, "seed_count": len(seeds), "period_count": len(periods),
        "baseline_variation": {"minimum": min(baseline_values) if baseline_values else None,
                               "maximum": max(baseline_values) if baseline_values else None,
                               "std": float(np.std(baseline_values, ddof=1)) if len(baseline_values) > 1 else None,
                               "by_seed": {str(seed): [item["baseline"] for item in pair_results if item["seed"] == seed]
                                           for seed in seeds},
                               "by_period": [{"period": list(period), "values": [item["baseline"] for item in pair_results
                                              if item["period"] == list(period)]} for period in periods]},
        "improvement": effect, "confidence_interval": interval,
        "bootstrap": {"method": "moving consecutive trading-day blocks within disjoint periods",
                      "seed_draws": "synchronized across all seeds; never stock-level iid",
                      "weighting": "equal run-pair metric differences", "samples": policy["bootstrap_samples"],
                      "block_days": policy["block_days"], "seed": policy["seed"]},
        "limitations": ["local approximate metrics; no official combined score",
                        "interval conditions on evaluated seeds and periods; repeated candidate selection needs fresh validation"],
    }
    decision = {"schema_version": 1, "status": status, "accepted": status == "accept",
                "result": {"accept": "保留", "reject": "回退", "inconclusive": "证据不足"}[status],
                "action": "retain_candidate" if status == "accept" else "retain_baseline",
                "failed_gates": sorted(set(failures)), "missing_evidence": sorted(set(missing)),
                "primary_stable_across_pairs": stability, "policy_sha256": policy_hash,
                "automatic_promotion": False}
    return comparison, decision
