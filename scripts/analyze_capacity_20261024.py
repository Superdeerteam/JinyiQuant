"""Assemble the capacity-20261024 comparison artifact from the frozen replay outputs.

The replay (``scripts/replay_model_versions.py``) scores every version on identical
keys, dates and labels inside one run, and its ``report`` step writes per-window daily
and paired tables. This script turns those artifacts into one machine-readable JSON:
the three-window metric table, per-window and pooled paired daily deltas with the
pre-registered 5-day block bootstrap, each version's worst day, the training-curve
contrast (``learning_curve.json`` checkpoints plus the 100-step ``events.jsonl`` loss
rows) and the recorded exposure, wall time and replay cost.

No metric is edited or re-derived by hand. Every number is either read back from a run
artifact or aggregated from daily rows with the frozen estimator imported from the
replay module, and the per-window deltas are cross-checked against
``report/paired_summary.csv`` before the JSON is written.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jinyiquant.io import file_sha256, read_json, utc_now, write_json  # noqa: E402

PAIR_METRICS = ("rank_ic", "long_short_return", "turnover")
SUMMARY_METRICS = ("rank_ic_mean", "rank_ic_ir", "ic_mean", "long_short_return_mean",
                   "long_short_sharpe", "turnover_mean", "min_daily_long_short_return", "extreme_return")
CURVE_FIELDS = ("steps", "epoch", "elapsed_seconds", "train_mse", "train_mse_over_constant",
                "validation_mse", "validation_mse_over_constant", "validation_rank_ic_mean",
                "validation_ic_mean", "validation_long_short_return_mean", "validation_turnover_mean",
                "validation_min_daily_long_short_return", "validation_scored_samples")
LOSS_GRID = (100, 1000, 3000, 6000, 9000)


def replay_module():
    spec = importlib.util.spec_from_file_location(
        "replay_model_versions", ROOT / "scripts/replay_model_versions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value):
    """JSON-safe scalar: non-finite floats become None instead of NaN."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def load_dailies(run_dir: Path, plan: dict) -> dict:
    frames = {}
    for window in plan["windows"]:
        for version in plan["versions"]:
            path = run_dir / "windows" / window["name"] / version["name"] / "daily.csv"
            frame = pd.read_csv(path)
            frame["date"] = pd.to_datetime(frame["date"])
            frames[(window["name"], version["name"])] = frame
    return frames


def window_table(run_dir: Path, plan: dict, frames: dict) -> list[dict]:
    rows = []
    for window in plan["windows"]:
        name = window["name"]
        for version in plan["versions"]:
            out = run_dir / "windows" / name / version["name"]
            metrics = read_json(out / "metrics.json")
            coverage = read_json(run_dir / "windows" / name / "coverage.json")
            ic = frames[(name, version["name"])]["ic"].dropna()
            ic_ir = (float(ic.mean() / ic.std(ddof=1))
                     if len(ic) > 1 and ic.std(ddof=1) > 0 else None)
            rows.append({
                "window": name, "version": version["name"], "hidden_dim": version.get("hidden_dim"),
                "steps": version.get("steps"), **{metric: finite(metrics[metric]) for metric in SUMMARY_METRICS},
                "ic_ir_daily": finite(ic_ir),
                "day_count": metrics["day_count"], "cross_section_count": metrics["cross_section_count"],
                "sample_count": metrics["sample_count"],
                "prediction_keys": coverage["keys"], "scored_samples": coverage["targets"],
                "no_history_samples": coverage["no_history_keys"],
                "no_history_ratio": finite(coverage["no_history_ratio"]),
                "sample_keys_sha256": metrics["sample_keys_sha256"],
                "targets_sha256": metrics["targets_sha256"],
            })
    return rows


def pooled_daily(frames: dict, windows: list[str], version: str) -> pd.DataFrame:
    """Concatenate a version's daily rows across windows in chronological window order."""
    parts = []
    for name in windows:
        daily = frames[(name, version)]
        part = daily[["date", "rank_ic", "ic", "long_short_return", "turnover", "sample_count"]].copy()
        part.insert(0, "window", name)
        parts.append(part)
    pooled = pd.concat(parts, ignore_index=True)
    if pooled.date.duplicated().any():
        raise ValueError(f"duplicate days while pooling {version}")
    if not pooled.date.is_monotonic_increasing:
        raise ValueError(f"pooled days are not chronological for {version}")
    return pooled.reset_index(drop=True)


def pooled_metrics(pooled: pd.DataFrame, periods_per_year: float) -> dict:
    rank_ic = pooled["rank_ic"].dropna()
    long_short = pooled["long_short_return"].dropna()
    turnover = pooled["turnover"].dropna()

    def sharpe(series: pd.Series):
        if len(series) < 2 or series.std(ddof=1) == 0:
            return None
        return float(series.mean() / series.std(ddof=1) * np.sqrt(periods_per_year))

    rank_ic_ir = (float(rank_ic.mean() / rank_ic.std(ddof=1))
                  if len(rank_ic) > 1 and rank_ic.std(ddof=1) > 0 else None)
    worst = pooled.loc[pooled["long_short_return"].idxmin()]
    return {
        "days": int(len(pooled)),
        "rank_ic_mean": finite(rank_ic.mean()),
        "rank_ic_ir": finite(rank_ic_ir),
        "long_short_return_mean": finite(long_short.mean()),
        "long_short_sharpe": finite(sharpe(long_short)),
        "turnover_mean": finite(turnover.mean()),
        "min_daily_long_short_return": finite(worst["long_short_return"]),
        "worst_day": str(worst["date"].date()), "worst_day_window": worst["window"],
        "samples_per_window": {name: int(pooled.loc[pooled.window == name, "sample_count"].sum())
                               for name in pooled.window.unique()},
    }


def paired_delta(frames: dict, windows: list[str], left: str, right: str, metric: str) -> pd.Series:
    parts = []
    for name in windows:
        left_frame = frames[(name, left)].set_index("date")[metric]
        right_frame = frames[(name, right)].set_index("date")[metric]
        if not left_frame.index.equals(right_frame.index):
            raise ValueError(f"{name}: {left} and {right} do not share prediction days")
        parts.append(left_frame - right_frame)
    return pd.concat(parts).sort_index().rename("delta")


def interval(delta: pd.Series, block: int, draws: int, seed: int) -> tuple[dict, pd.Series]:
    replay = replay_module()
    values = delta.dropna()
    stats = replay._block_delta(pd.DataFrame({"delta": values.to_numpy()}), "delta", block, draws, seed)
    return stats, values


def paired_row(frames: dict, windows: list[str], pair: list[str], metric: str,
               block: int, draws: int, seed: int) -> dict:
    stats, values = interval(paired_delta(frames, windows, pair[0], pair[1], metric), block, draws, seed)
    row = {"pair": f"{pair[0]}-minus-{pair[1]}", "metric": metric,
           "days": stats["days"], "mean_delta": finite(stats["mean"]),
           "block5_ci_low": finite(stats["ci_low"]), "block5_ci_high": finite(stats["ci_high"]),
           "positive_days": int((values > 0).sum()), "negative_days": int((values < 0).sum()),
           "zero_days": int((values == 0).sum())}
    if len(values):
        row.update({"min_delta": finite(values.min()), "min_delta_date": str(values.idxmin().date()),
                    "max_delta": finite(values.max()), "max_delta_date": str(values.idxmax().date())})
    return row


def report_rows(report_dir: Path) -> dict:
    path = report_dir / "paired_summary.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return {(row["window"], row["pair"], row["metric"]): row for _, row in frame.iterrows()}


def verify_report(paired_window: list[dict], report: dict) -> dict:
    if not report:
        raise ValueError("report/paired_summary.csv is missing; run the replay 'report' step first")
    deltas, intervals = [], []
    for row in paired_window:
        key = (row["window"], row["pair"], row["metric"])
        if key not in report:
            raise ValueError(f"report/paired_summary.csv has no row for {key}")
        reference = report[key]
        deltas.append(abs(row["mean_delta"] - float(reference["mean_delta"])))
        intervals.append(abs(row["block5_ci_low"] - float(reference["block5_ci_low"])))
        intervals.append(abs(row["block5_ci_high"] - float(reference["block5_ci_high"])))
    worst = max(deltas + intervals)
    return {"rows_compared": len(paired_window), "max_abs_mean_delta_diff": worst,
            "max_abs_ci_diff": max(intervals), "match": worst <= 1e-12}


def read_events(mirror: Path) -> list[dict]:
    path = mirror / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def training_curve(mirror: Path) -> dict:
    """100-step loss rows, the run-closing summary row, epoch boundaries and curve checkpoints."""
    events = read_events(mirror)
    progress, epoch_events, final_event = [], [], None
    for event in events:
        stage, kind = event.get("stage"), event.get("event_type")
        context = event.get("context", {})
        if stage == "training" and kind == "progress":
            if "step" in context:
                progress.append({"step": int(context["step"]), "loss": float(context["loss"]),
                                 "optimization_loss": finite(context.get("optimization_loss")),
                                 "epoch": context.get("epoch"), "samples_seen": context.get("samples_seen"),
                                 "elapsed_seconds": finite(context.get("elapsed_seconds"))})
            else:
                final_event = {name: finite(context.get(name)) for name in
                               ("steps", "samples_seen", "epochs", "epochs_completed", "loss",
                                "optimization_loss", "elapsed_seconds", "stop_reason",
                                "throughput_samples_per_second")}
        elif kind == "epoch_completed":
            epoch_events.append({"step": context.get("step"), "samples_seen": context.get("samples_seen"),
                                 "epoch": context.get("epoch")})
    progress.sort(key=lambda row: row["step"])
    checkpoints = []
    curve_path = mirror / "learning_curve.json"
    if curve_path.exists():
        checkpoints = [{field: finite(entry[field]) for field in CURVE_FIELDS if field in entry}
                       for entry in read_json(curve_path)]
    config_path = mirror / "config.json"
    logging = read_json(config_path).get("logging", {}) if config_path.exists() else {}
    losses = {row["step"]: row["loss"] for row in progress}
    grid = {str(step): finite(losses[step]) for step in LOSS_GRID if step in losses}
    grid.update({str(row["step"]): finite(row["loss"]) for row in progress
                 if row["step"] > 9000 and row["step"] % 3000 == 0})
    return {
        "learning_curves_enabled": bool(logging.get("learning_curves")),
        "learning_curve_every_steps": logging.get("learning_curve_every_steps"),
        "progress_rows": len(progress),
        "progress": {"step": [row["step"] for row in progress],
                     "loss": [row["loss"] for row in progress],
                     "epoch": [row["epoch"] for row in progress],
                     "samples_seen": [row["samples_seen"] for row in progress],
                     "elapsed_seconds": [row["elapsed_seconds"] for row in progress]},
        "loss_at_steps": grid,
        "final_training_event": final_event,
        "last_progress": progress[-1] if progress else None,
        "epoch_events": epoch_events,
        "checkpoints": checkpoints,
    }


def exposure(version: dict, summary: dict | None, record: dict | None, curve: dict,
             reference_version: str) -> dict:
    name = version["name"]
    entry = {"hidden_dim": version.get("hidden_dim"), "declared_steps": version.get("steps"),
             "declared_epochs": version.get("epochs"),
             "events_last_step": curve["last_progress"]["step"] if curve["last_progress"] else None,
             "events_last_samples_seen": curve["last_progress"]["samples_seen"]
             if curve["last_progress"] else None,
             "events_epoch_boundaries": curve["epoch_events"]}
    final = curve["final_training_event"]
    if final:
        entry.update({"events_final_steps": final.get("steps"),
                      "events_final_samples_seen": final.get("samples_seen"),
                      "events_final_epochs": final.get("epochs"),
                      "events_final_stop_reason": final.get("stop_reason")})
    arm = next((row for row in summary["arms"] if row["arm"] == name), None) if summary else None
    if arm:
        entry.update({"run_id": arm["run_id"], "steps": arm["steps"], "samples_seen": arm["samples_seen"],
                      "epochs_achieved": finite(arm["epochs_achieved"]),
                      "wall_seconds": finite(arm["wall_seconds"]),
                      "training_seconds": finite(arm.get("training_seconds")),
                      "prep_seconds": finite(arm.get("prep_seconds")),
                      "stop_reason": arm.get("stop_reason"),
                      "peak_rss_mb": finite(arm.get("peak_rss_mb")),
                      "budget_wall_seconds": arm.get("budget_wall_seconds"),
                      "budget_margin": finite(arm.get("budget_margin")),
                      "curve_checkpoint_count": arm["curve_checkpoints"]["count"],
                      "source": "runs/model-versions/capacity-20261024/summary.json"})
    reference = None
    if record and name == reference_version:
        reference = next((row for row in record["arms"] if row["arm"] == record.get("reference_arm", "60d")),
                         None)
    if reference:
        entry.update({"run_id": reference["run_id"], "steps": reference["steps"],
                      "samples_seen": reference["samples_seen_exposures"],
                      "epochs_achieved": finite(reference["epochs"]),
                      "wall_seconds": finite(reference["wall_seconds"]),
                      "training_seconds": finite(reference.get("training_elapsed_seconds")),
                      "source": "runs/model-versions/v2-cloud-20261023/record.json arms[arm=60d]"})
    return entry


def end_of_run_validation(version: dict, summary: dict | None, record: dict | None,
                          reference_version: str) -> dict | None:
    """Within-training validation numbers for the reused reference run (no curve checkpoints)."""
    name = version["name"]
    if summary and name != reference_version:
        arm = next((row for row in summary["arms"] if row["arm"] == name), None)
        if arm and arm.get("validation_metrics"):
            return {"steps": arm.get("steps"), "source": "runs/model-versions/capacity-20261024/summary.json",
                    "metrics": {key: finite(value) for key, value in arm["validation_metrics"].items()}}
    if record and name == reference_version:
        arm = next((row for row in record["arms"] if row["arm"] == record.get("reference_arm", "60d")), None)
        if arm and arm.get("metrics"):
            return {"steps": arm.get("steps"), "source": "runs/model-versions/v2-cloud-20261023/record.json",
                    "metrics": {key: finite(value) for key, value in arm["metrics"].items()}}
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/capacity_20261024/replay.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "modified_tlob/capacity_20261024.json")
    parser.add_argument("--trainer-summary", type=Path,
                        default=ROOT / "runs/model-versions/capacity-20261024/summary.json")
    parser.add_argument("--reference-record", type=Path,
                        default=ROOT / "runs/model-versions/v2-cloud-20261023/record.json")
    parser.add_argument("--reference-version", default="v2_60d_csz",
                        help="plan version backed by --reference-record; other versions read the trainer summary")
    parser.add_argument("--continuation-version", default="w40_full2epoch",
                        help="version that continues the reference run past step 9000 (two-epoch arm)")
    parser.add_argument("--preregistration", type=Path,
                        default=ROOT / "configs/capacity_20261024/experiments.json")
    args = parser.parse_args(argv)

    replay = replay_module()
    plan, _ = replay.load_plan(args.plan)
    run_dir = args.run_dir.resolve()
    windows = [window["name"] for window in plan["windows"]]
    versions = [version["name"] for version in plan["versions"]]
    block = int(plan.get("block_days", 5))
    draws = int(plan.get("bootstrap_draws", 20000))
    seed = int(plan["seed"])
    manifest = read_json(run_dir / "manifest.json")
    frames = load_dailies(run_dir, plan)

    window_rows = window_table(run_dir, plan, frames)
    digests = {}
    for row in window_rows:
        entry = digests.setdefault(row["window"], {"sample_keys_sha256": set(), "targets_sha256": set()})
        entry["sample_keys_sha256"].add(row["sample_keys_sha256"])
        entry["targets_sha256"].add(row["targets_sha256"])

    pooled = {version: pooled_daily(frames, windows, version) for version in versions}
    pooled_metrics_by_version = {
        version: pooled_metrics(pooled[version], plan["evaluation"]["periods_per_year"])
        for version in versions}
    paired_window, paired_pooled = [], []
    for pair in plan["pairs"]:
        for metric in PAIR_METRICS:
            for name in windows:
                row = paired_row(frames, [name], pair, metric, block, draws, seed)
                row["window"] = name
                paired_window.append(row)
            paired_pooled.append(paired_row(frames, windows, pair, metric, block, draws, seed))

    report_check = verify_report(paired_window, report_rows(run_dir / "report"))
    prereg = read_json(args.preregistration)
    prereg_pairs = [list(pair) for pair in prereg["evaluation"]["pairs"]]
    prereg_windows = [{"name": window["name"], "start": window["start"], "end": window["end"]}
                      for window in prereg["evaluation"]["windows"]]
    plan_windows = [{"name": window["name"], "start": window["start"], "end": window["end"]}
                    for window in plan["windows"]]
    summary = read_json(args.trainer_summary) if args.trainer_summary.exists() else None
    record = read_json(args.reference_record) if args.reference_record.exists() else None

    curves, exposures, cadence = {}, {}, []
    expected_curves = prereg["runtime_budget"]["curve_evaluations"]
    for version in plan["versions"]:
        mirror = Path(version["weights"]).resolve().parent
        curve = training_curve(mirror)
        curves[version["name"]] = curve
        exposures[version["name"]] = exposure(version, summary, record, curve, args.reference_version)
        if version["name"] in expected_curves:
            cadence.append({"arm": version["name"], "observed_checkpoints": len(curve["checkpoints"]),
                            "preregistered_curve_evaluations": expected_curves[version["name"]],
                            "match": len(curve["checkpoints"]) == expected_curves[version["name"]],
                            "learning_curve_every_steps": curve["learning_curve_every_steps"],
                            "checkpoint_steps": [row["steps"] for row in curve["checkpoints"]]})

    end_validation = {version["name"]: end_of_run_validation(version, summary, record,
                                                            args.reference_version)
                      for version in plan["versions"]}
    reference_name, continuation_name = args.reference_version, args.continuation_version
    continuity = None
    if continuation_name in versions and reference_name in versions:
        reference_rows = dict(zip(curves[reference_name]["progress"]["step"],
                                  curves[reference_name]["progress"]["loss"]))
        continuation_rows = dict(zip(curves[continuation_name]["progress"]["step"],
                                     curves[continuation_name]["progress"]["loss"]))
        shared = sorted(step for step in reference_rows if step in continuation_rows and step <= 9000)
        differences = [abs(reference_rows[step] - continuation_rows[step]) for step in shared]
        continuity = {"reference": reference_name, "continuation": continuation_name,
                      "compared_steps": len(shared),
                      "max_abs_loss_difference": max(differences) if differences else None,
                      "note": "the two-epoch arm's first 9000 updates reproduce the reused reference run's "
                              "100-step loss rows; its extra epochs are a continuation of the same trajectory, "
                              "not a second independent sample"}
    matched_steps = [str(step) for step in LOSS_GRID]
    curve_contrasts = {
        "matched_step_loss": {version: {step: curves[version]["loss_at_steps"].get(step)
                                        for step in matched_steps} for version in versions},
        "checkpoint_validation_rank_ic": {
            version: {str(row["steps"]): row.get("validation_rank_ic_mean")
                      for row in curves[version]["checkpoints"]} for version in versions},
        "notes": [
            "training volume: v2_60d_csz and w40_full2epoch share the 100-step loss rows up to step 9000, "
            "so the matched-step losses isolate the two-epoch continuation that only w40_full2epoch has",
            "capacity: v2_60d_csz, w80_9000 and w160_9000 share the same 9000-step schedule, so their "
            "matched-step losses and step-9000 validation metrics compare widths at equal update budget",
            "v2_60d_csz trained with logging.learning_curves=false, so it has no learning_curve.json "
            "checkpoints; its curve is the 100-step events.jsonl loss rows and its single end-of-run validation",
        ],
    }
    exposure_checks = []
    for version in plan["versions"]:
        entry = exposures[version["name"]]
        declared, events_steps, recorded = entry.get("declared_steps"), entry.get("events_final_steps"),             entry.get("steps")
        values = [value for value in (declared, events_steps, recorded) if value is not None]
        exposure_checks.append({"version": version["name"], "declared_steps": declared,
                                "events_final_steps": events_steps, "recorded_steps": recorded,
                                "steps_consistent": len(set(values)) == 1 and bool(values),
                                "sources_compared": len(values),
                                "samples_seen": entry.get("samples_seen"),
                                "epochs_achieved": entry.get("epochs_achieved")})
    coverage = {}
    for window in plan["windows"]:
        entries = read_json(run_dir / "windows" / window["name"] / "coverage.json")
        coverage[window["name"]] = {"keys": entries["keys"], "targets": entries["targets"],
                                    "no_history_keys": entries["no_history_keys"],
                                    "no_history_ratio": finite(entries["no_history_ratio"]),
                                    "prediction_seconds": sum(chunk["seconds"] for chunk in entries["chunks"])}
    replay_cost = {"run_id": manifest["run_id"], "experiment_id": manifest["experiment_id"],
                   "device": manifest["device"], "wall_seconds": finite(manifest["resources"]["wall_seconds"]),
                   "code_commit": manifest["source"]["commit"], "windows": coverage,
                   "versions": len(versions),
                   "scored_rows": sum(entry["targets"] for entry in coverage.values()) * len(versions)}

    payload = {
        "schema_version": 1,
        "kind": "capacity_study_development_replay_comparison",
        "study_id": "capacity-20261024",
        "generated_utc": utc_now(),
        "metric_scope": "local approximate jinyiquant.evaluation (gross forward label returns, no fees, "
                        "slippage or execution); not the official platform score",
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan.resolve()), "seed": seed,
                 "experiment_id": plan["experiment_id"], "windows": windows,
                 "pairs": [list(pair) for pair in plan["pairs"]],
                 "bootstrap": {"block_days": block, "draws": draws, "quantiles": [0.025, 0.975],
                               "method": "moving consecutive-day block bootstrap of the paired daily delta "
                                         "series, estimator scripts/replay_model_versions.py::_block_delta"},
                 "pooled_series_rule": prereg["evaluation"]["primary_series"],
                 "pooled_metrics_rule": "equal weight per paired trading day across the three windows; not "
                                        "sample-weighted"},
        "replay": replay_cost,
        "window_metrics": window_rows,
        "pooled_metrics": pooled_metrics_by_version,
        "paired_deltas": {"per_window": paired_window, "pooled": paired_pooled},
        "curves": {"per_version": curves, "contrasts": curve_contrasts,
                   "end_of_run_validation": end_validation,
                   "validation_split_note": "the learning-curve checkpoints and the end-of-run validation "
                                            "numbers score the V2 validation split 2023-11-01..2023-11-14 "
                                            "(10 days, 80 sections, 79,920 samples); they are within-training "
                                            "diagnostics, not the three development windows"},
        "exposure": exposures,
        "verification": {
            "weights_sha256": [{"version": name, "expected_sha256": entry.get("expected_sha256"),
                                "observed_sha256": manifest["weights"][name]["sha256"],
                                "match": (entry["expected_sha256"] == manifest["weights"][name]["sha256"]
                                          if entry.get("expected_sha256") else None)}
                               for name, entry in zip(versions, plan["versions"])],
            "identical_evaluation_scope_per_window": [
                {"window": name, "distinct_sample_keys_sha256": len(value["sample_keys_sha256"]),
                 "distinct_targets_sha256": len(value["targets_sha256"]),
                 "all_versions_identical": len(value["sample_keys_sha256"]) == 1
                 and len(value["targets_sha256"]) == 1}
                for name, value in digests.items()],
            "report_agreement": report_check,
            "preregistration": {"source": str(args.preregistration),
                                "pairs_match": [list(pair) for pair in plan["pairs"]] == prereg_pairs,
                                "windows_match": plan_windows == prereg_windows},
            "curve_cadence": cadence,
            "training_continuity": continuity,
            "exposure_readback": exposure_checks,
        },
        "commands": {
            "replay": ["python scripts/replay_model_versions.py run configs/capacity_20261024/replay.json",
                       "python scripts/replay_model_versions.py report "
                       "configs/capacity_20261024/replay.json "
                       f"{run_dir}"],
            "analysis": [f"python scripts/analyze_capacity_20261024.py --run-dir {run_dir} "
                         "--out modified_tlob/capacity_20261024.json"],
            "interpreter_note": "the replay loads torch checkpoints, so use an interpreter with torch "
                                "(this host: .venv-ensemble-cuda, torch 2.11.0+cu128); the analysis itself "
                                "only needs numpy/pandas",
        },
    }
    write_json(args.out, payload)
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
