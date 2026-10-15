"""Preregister one controlled factor and bind thresholds to measured calibration."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from .checks import verify_artifacts
from .config import ModelConfig
from .evaluation import METRIC_SPEC, aggregate_metric
from .io import digest, read_json

DATA_FACTORS = {"lookback", "history_days", "min_history", "feature_columns", "missing_policy",
                "max_ffill_steps", "price_scale", "missing_sentinel"}
_LOCATIONS = {"path", "bars_path", "labels_path", "instruments_path", "instrument_map_path",
              "cache_dir", "cache_path"}


def without_locations(value):
    if isinstance(value, dict):
        return {k: without_locations(v) for k, v in value.items() if k not in _LOCATIONS}
    if isinstance(value, list):
        return [without_locations(v) for v in value]
    return value


def validate_change(change):
    if not isinstance(change, dict) or set(change) != {"factor", "baseline", "candidate", "hypothesis"}:
        raise ValueError("change needs exactly factor, baseline, candidate and hypothesis")
    factors = {f"data.{key}" for key in DATA_FACTORS} | {f"model.{key}" for key in ModelConfig.model_fields}
    if change["factor"] not in factors | {"control.identity"}:
        raise ValueError("change.factor must name one supported model/data field or control.identity")
    if not isinstance(change["hypothesis"], str) or not change["hypothesis"].strip():
        raise ValueError("change.hypothesis must be nonempty")
    if change["factor"] == "control.identity":
        if change["baseline"] is not None or change["candidate"] is not None:
            raise ValueError("identity control requires null factor values")
    elif change["baseline"] == change["candidate"]:
        raise ValueError("the declared factor must actually change")


def recipe(config, *, across_grid=False):
    value = deepcopy(config)
    for key in ("hypothesis", "experiment_id", "output_root"):
        value.pop(key, None)
    if across_grid:
        value.pop("seed", None)
        value["splits"].pop("validation", None)
    return without_locations(value)


def fair_config_pair(baseline, candidate, change):
    """No wildcard exemptions: remove exactly the registered field, then compare."""
    validate_change(change)
    left, right = recipe(baseline), recipe(candidate)
    if change["factor"] != "control.identity":
        section, field = change["factor"].split(".")
        for value, arm in ((left, "baseline"), (right, "candidate")):
            if field not in value.get(section, {}) or value[section][field] != change[arm]:
                raise ValueError(f"unfair comparison: {arm} differs from preregistered {change['factor']}")
            del value[section][field]
    if left != right:
        raise ValueError("unfair comparison: undeclared configuration difference outside the single factor")


def fair_data_pair(baseline, candidate, change):
    if not baseline or not candidate:
        raise ValueError("unfair comparison: missing data identity")
    left, right = without_locations(baseline), without_locations(candidate)
    factor = change["factor"]
    # Only metadata mechanically derived from the named factor can differ.
    derived = {"data.feature_columns": {"feature_columns"}, "data.price_scale": {"price_scale"},
               "data.history_days": {"history_days", "bars_start", "bar_rows"}}.get(factor, set())
    for field in derived:
        if field in {"feature_columns", "price_scale", "history_days"}:
            for value, arm in ((left, "baseline"), (right, "candidate")):
                if value.get(field) != change[arm]:
                    raise ValueError(f"unfair comparison: metadata contradicts registered {field}")
        left.pop(field, None)
        right.pop(field, None)
    if left != right:
        raise ValueError("unfair comparison: differing data identity outside the single factor")


def validate_threshold_basis(policy):
    basis = policy["threshold_basis"]
    required = {"calibration_report", "calibration_sha256", "practical_improvement", "noise_multiplier"}
    if not isinstance(basis, dict) or set(basis) != required:
        raise ValueError(f"threshold_basis needs exactly {sorted(required)}")
    if not isinstance(basis["calibration_report"], str) or not basis["calibration_report"]:
        raise ValueError("threshold_basis requires a calibration report path")
    checksum = basis["calibration_sha256"]
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("threshold_basis requires a calibration SHA256")
    for field in ("practical_improvement", "noise_multiplier"):
        value = basis[field]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"threshold_basis.{field} must be positive finite numeric")


def calibration_evidence(policy, baseline_config, *, registered_at=None, baseline_data=None):
    """Recompute the reported variation from intact baseline archives, before use."""
    basis = policy["threshold_basis"]
    path = Path(basis["calibration_report"]).resolve()
    report = read_json(path)
    if digest(report) != basis["calibration_sha256"] or report.get("status") != "completed":
        raise ValueError("Calibration report changed or did not complete")
    completed = pd.Timestamp(report.get("created_at"))
    if pd.isna(completed) or completed.tzinfo is None:
        raise ValueError("Calibration requires a UTC completion timestamp")
    if registered_at is not None and completed > pd.Timestamp(registered_at):
        raise ValueError("Calibration must complete before policy registration")
    configs, values, periods = [], [], {}
    metric = policy["primary_metric"]
    column, statistic, _ = METRIC_SPEC[metric]
    for location in report.get("runs", []):
        run = Path(location)
        if not run.is_absolute():
            run = path.parent / run
        if not verify_artifacts(run)["passed"]:
            raise ValueError("Calibration artifact checksum mismatch")
        manifest, config, checks, metrics = [read_json(run / f"{name}.json")
                                            for name in ("manifest", "config", "checks", "metrics")]
        listed = {item["path"] for item in manifest.get("artifacts", [])}
        if {"config.json", "data.json", "metrics.json", "daily.csv", "checks.json"} - listed:
            raise ValueError("Calibration artifact inventory is incomplete")
        if (manifest.get("status") != "completed" or checks.get("passed") is not True
                or config.get("evaluation_split") != "validation"):
            raise ValueError("Calibration requires completed, passing validation runs")
        if recipe(config, across_grid=True) != recipe(baseline_config, across_grid=True):
            raise ValueError("Calibration baseline recipe differs")
        period = config["splits"]["validation"]
        key = digest(period)
        if baseline_data is not None:
            calibrated_data = read_json(run / "data.json")
            observed = baseline_data.get(key)
            if observed is None:
                # New periods may read different rows but must share the frozen
                # source files. A new snapshot requires a fresh calibration.
                source_sets = [without_locations(item.get("files")) for item in baseline_data.values()]
                if not calibrated_data.get("files") or without_locations(calibrated_data["files"]) not in source_sets:
                    raise ValueError("Calibration source snapshot differs from evaluated baseline")
            elif without_locations(calibrated_data) != without_locations(observed):
                raise ValueError("Calibration data identity differs from evaluated baseline")
        daily = pd.read_csv(run / "daily.csv")
        value = aggregate_metric(daily[column].to_numpy(dtype=float), statistic,
                                 periods_per_year=config["evaluation"]["periods_per_year"])
        if value is None or not np.isfinite(value) or not np.isfinite(daily[column]).all():
            raise ValueError("Calibration primary metric is undefined")
        if metrics.get(metric) is None or not np.isclose(value, metrics[metric], atol=1e-12, rtol=1e-9):
            raise ValueError("Calibration metrics contradict daily results")
        periods[key] = period
        configs.append((config["seed"], key))
        values.append(float(value))
        ended = pd.Timestamp(manifest.get("ended_at"))
        if pd.isna(ended) or ended.tzinfo is None or ended > completed:
            raise ValueError("Calibration report predates a run completion")
    seeds = {seed for seed, _ in configs}
    if (len(seeds) < policy["min_seeds"] or len(periods) < policy["min_periods"]
            or len(set(configs)) != len(configs) or len(configs) != len(seeds) * len(periods)):
        raise ValueError("Calibration needs a complete unique seed/period grid")
    ordered = sorted(periods.values(), key=lambda p: pd.Timestamp(p["start"]))
    if any(pd.Timestamp(a["end"]) > pd.Timestamp(b["start"]) for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Calibration periods must not overlap")
    std = float(np.std(values, ddof=1))
    summary = report.get("metrics", {}).get(metric, {})
    expected = {"n": len(values), "mean": float(np.mean(values)), "std": std,
                "min": min(values), "max": max(values)}
    if any(summary.get(key) is None or not np.isclose(summary[key], value, atol=1e-12, rtol=1e-9)
           for key, value in expected.items()):
        raise ValueError("Calibration summary differs from archived measurements")
    required = max(basis["practical_improvement"], std * basis["noise_multiplier"])
    if policy["min_improvement"] + 1e-12 < required:
        raise ValueError("min_improvement is below measured baseline noise or practical relevance")
    return {"path": str(path), "sha256": digest(report), "measured_std": std,
            "required_improvement": required, "seeds": len(seeds), "periods": len(periods)}
