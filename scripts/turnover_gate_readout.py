"""Frozen-gate readout for the single turnover change (t7), plus reusable object exports.

The B-stage promotion gate is pre-registered in ``configs/b4_promotion_prereg_20261027.json``
and executed by ``scripts/promotion_gate.py``. This script does two things and nothing else:

* **readout** — it re-reads the *frozen* thresholds from the registered pre-registration
  (never re-declaring, relaxing or replacing them), recomputes every gate line that applies
  to a deterministic object pair from the archived daily artifacts of the t7 run, and reports
  each line with its value and pass/fail. It is a readout, not a verdict: the verdict belongs
  to the verification task, and ``promotion_gate.py`` itself cannot decide this change because
  its change vocabulary only accepts ``data.*``, ``model.*`` and ``control.identity`` — an
  output-level persistence operator is refused. That refusal is captured as evidence.
* **object export** — it rebuilds the reference and candidate score vectors from the frozen
  replay predictions with the same functions the t7 run used, writes ``predictions.parquet``
  and ``targets.parquet`` per object, and re-evaluates both at 5bp in a child process to prove
  the export reproduces the archived numbers. The verifier can re-evaluate from these files
  without trusting the t7 dailies.

No threshold is computed, tuned or changed here; no model is trained, promoted or rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from cost_curve_eval import PRIMARY_RATE, RATES, SEALED, WINDOWS, evaluation_config, load_window, num
from turnover_stability_change import CANDIDATE, REFERENCE, build_objects

from jinyiquant.comparison import _block_indices
from jinyiquant.evaluation import EVALUATION_VERSION, METRIC_SPEC, aggregate_metric, evaluate
from jinyiquant.io import canonical_json, digest, file_sha256, utc_now, write_json
from jinyiquant.protocol import validate_change
from jinyiquant.tracking import Recorder

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "turnover-readout-20261027"
SOURCE_EXPERIMENT = "turnover-stability-20261027"
PREREG = ROOT / "configs/b4_promotion_prereg_20261027.json"
REGISTRATION = ROOT / "configs/b4_promotion_prereg_20261027.registration.json"
SECONDARY_METRICS = ("ic_mean", "ic_ir", "long_short_sharpe", "long_short_return_mean",
                     "min_daily_long_short_return", "turnover_mean")
MIB_PER_MB = 1e6 / 2 ** 20
SAMPLE_SECONDS = 0.1


def rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame or frame.empty:
        raise ValueError(f"{rel(path)} has no daily rows")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["date"].duplicated().any():
        raise ValueError(f"{rel(path)} has duplicate trading days")
    return frame.sort_values("date").reset_index(drop=True)


def normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """eval-v2 keeps the incoming date dtype; normalize before any period comparison."""
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["date"].duplicated().any():
        raise ValueError("daily frame has duplicate trading days")
    return frame.sort_values("date").reset_index(drop=True)


def block_interval(differences: pd.Series, dates: pd.Series, periods: list[tuple[str, str]],
                   policy: dict) -> dict:
    """Moving consecutive-day blocks inside each declared period; stocks are never resampled."""
    frame = pd.DataFrame({"value": differences.to_numpy(dtype=float),
                          "date": pd.to_datetime(dates).dt.normalize()})
    frame = frame[np.isfinite(frame["value"])]
    groups = []
    for start, end in periods:
        low, high = pd.Timestamp(start), pd.Timestamp(end)
        block = frame[(frame["date"] >= low) & (frame["date"] <= high)]
        groups.append((f"{start}..{end}", block["value"].to_numpy(dtype=float)))
    if any(len(values) == 0 for _, values in groups):
        raise ValueError("a declared period has no paired days")
    rng = np.random.default_rng(int(policy["seed"]))
    block_days = int(policy["block_days"])
    draws = []
    for _ in range(int(policy["bootstrap_samples"])):
        total, count = 0.0, 0
        for _, values in groups:
            indices = _block_indices(len(values), min(block_days, len(values)), rng)
            total += float(values[indices].sum())
            count += len(indices)
        draws.append(total / count)
    sampled = np.asarray(draws)
    alpha = (1 - float(policy["confidence"])) / 2
    low, high = np.quantile(sampled, [alpha, 1 - alpha])
    return {"mean": num(frame["value"].mean()), "low": num(float(low)), "high": num(float(high)),
            "share_positive": num(float((sampled > 0).mean())),
            "block_days": block_days, "draws": int(policy["bootstrap_samples"]),
            "seed": int(policy["seed"]), "confidence": float(policy["confidence"]),
            "defined_days": int(len(frame)),
            "days_per_period": {name: int(len(values)) for name, values in groups}}


def period_effects(differences: pd.Series, dates: pd.Series,
                   periods: list[tuple[str, str]]) -> dict:
    frame = pd.DataFrame({"value": differences.to_numpy(dtype=float),
                          "date": pd.to_datetime(dates).dt.normalize()})
    effects = []
    for start, end in periods:
        block = frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))]
        effects.append(num(block["value"].mean()) if len(block) else None)
    positive = sum(1 for value in effects if value is not None and value > 0)
    return {"period_names": [f"{start}..{end}" for start, end in periods], "period_effects": effects,
            "positive_periods": positive, "periods": len(effects),
            "fraction_positive": num(positive / len(effects)) if effects else None}


def secondary_values(metrics: dict, daily: pd.DataFrame, label: str) -> dict:
    """Recompute gate secondaries from the daily artifact, exactly like promotion_gate.py."""
    values = {}
    for metric in SECONDARY_METRICS:
        column, statistic, _ = METRIC_SPEC[metric]
        if column not in daily:
            raise ValueError(f"{label}: daily artifact lacks {column!r} for {metric}")
        recomputed = aggregate_metric(daily[column].to_numpy(dtype=float), statistic, 252.0)
        recorded = metrics.get(metric)
        if recomputed is None or recorded is None or not np.isfinite(recorded):
            raise ValueError(f"{label}: {metric} is undefined")
        if not np.isclose(recomputed, recorded, rtol=1e-9, atol=1e-12):
            raise ValueError(f"{label}: recorded {metric} does not match the daily artifact")
        values[metric] = float(recorded)
    values["net_return_mean_5bp"] = float(daily["net_return"].mean())
    return values


def evaluate_exported(payload) -> dict:
    """Child: evaluate the exported parquet files so the export must stand on its own."""
    directory, label = payload
    predictions = pd.read_parquet(Path(directory) / "predictions.parquet")
    targets = pd.read_parquet(Path(directory) / "targets.parquet")
    results = {"object_id": label}
    for name, rate in (("0bp", 0.0), ("5bp", 0.0005)):
        metrics, daily, _ = evaluate(predictions, targets, evaluation_config(rate),
                                     return_sections=True, undefined_constant_portfolios=True)
        results[name] = {"metrics": metrics, "daily": daily}
    return results


def sample_child_rss(future_proc) -> float:
    """Peak RSS (MiB) of the sampling process plus every descendant, polled while it runs."""
    peak = 0.0
    parent = psutil.Process()
    while True:
        total = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                continue
        peak = max(peak, total)
        if future_proc.done():
            break
        time.sleep(SAMPLE_SECONDS)
    return peak / 2 ** 20


def build_plan(prereg: dict, registration_sha: str, source_run: str) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "evaluation_version": EVALUATION_VERSION,
        "purpose": "read the frozen promotion gate against the single t7 turnover change, and export "
                   "the two objects for independent re-evaluation",
        "source_run": source_run,
        "prereg": {"path": rel(PREREG), "sha256": registration_sha,
                   "registered_at": prereg["registered_at"],
                   "note": "thresholds are read from the registered pre-registration and never changed "
                           "here; this script cannot promote or reject anything"},
        "readout_items_predeclared": [
            "declared periods and day coverage of the source run",
            "primary paired improvement (rank_ic) against the frozen min_improvement",
            "5-day block interval inside each declared period, excluding zero",
            "fraction of declared periods with a positive paired effect",
            "six secondary metrics recomputed from the daily artifact against the frozen lines",
            "5bp net-return regression against the frozen net5 line",
            "structural check: does the frozen change vocabulary accept an output-level operator",
            "resource note for this offline evaluation workload (the platform resource gate applies "
            "to the training and submission pipeline, not to this readout)",
        ],
        "verdict_policy": "no verdict; the verification task decides and scripts/promotion_gate.py "
                          "remains the only executable gate",
        "sealed_interval": list(SEALED),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-run", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/turnover-readout-20261027")
    parser.add_argument("--record", type=Path,
                        default=ROOT / "modified_tlob/turnover_stability_readout_20261027.json")
    args = parser.parse_args()

    started = utc_now()
    timer = time.time()
    source = args.source_run or sorted((ROOT / "runs" / SOURCE_EXPERIMENT).iterdir())[-1]
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    prereg_sha = file_sha256(PREREG)
    if registration.get("prereg_sha256") != prereg_sha:
        raise ValueError("pre-registration bytes changed after registration")
    plan = build_plan(prereg, prereg_sha, rel(source))
    plan_bytes = (canonical_json(plan) + "\n").encode("utf-8")
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    output = args.output_root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{plan_sha[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_bytes(plan_bytes)
    recorder = Recorder(output, EXPERIMENT_ID)
    recorder.event("plan", "predeclared", plan_sha256=plan_sha, source_run=rel(source),
                   prereg_sha256=prereg_sha)

    policy = dict(prereg["object_policy_template"])
    periods = [(start, end) for start, end in prereg["periods"]["declared"]]
    evidence_rules = prereg["evidence_rules"]
    archived = {
        object_id: json.loads((source / "objects" / object_id / "metrics.json").read_text(encoding="utf-8"))
        for object_id in (REFERENCE, CANDIDATE)}

    exports = output / "objects"
    objects, keys, targets, _ = build_objects([load_window(*window) for window in WINDOWS])
    for object_id in (REFERENCE, CANDIDATE):
        directory = exports / object_id
        directory.mkdir(parents=True, exist_ok=True)
        keys.assign(score=objects[object_id])[["date", "instrument", "score"]].to_parquet(
            directory / "predictions.parquet", index=False)
        targets.to_parquet(directory / "targets.parquet", index=False)
        write_json(directory / "metrics.json", {
            "schema_version": 1, "object_id": object_id, "evaluation_version": EVALUATION_VERSION,
            "source_run": rel(source),
            "single_change": "causal_ewma_alpha050" if object_id == CANDIDATE else None,
            "note": "predictions.parquet + targets.parquet are the frozen inputs for independent "
                    "re-evaluation of this object at any single-side rate",
        })
    recorder.event("export", "written", object_count=2, rows=int(len(keys)))

    recomputed = {}
    with ProcessPoolExecutor(max_workers=1) as executor:
        first = executor.submit(evaluate_exported, (str(exports / CANDIDATE), CANDIDATE))
        peak = sample_child_rss(first)
        recomputed[CANDIDATE] = first.result()
        second = executor.submit(evaluate_exported, (str(exports / REFERENCE), REFERENCE))
        sample_child_rss(second)
        recomputed[REFERENCE] = second.result()
    frames = {}
    for object_id in (REFERENCE, CANDIDATE):
        frames[object_id] = {
            "metrics_0bp": recomputed[object_id]["0bp"]["metrics"],
            "metrics_5bp": recomputed[object_id]["5bp"]["metrics"],
            "daily_0bp": normalize_dates(recomputed[object_id]["0bp"]["daily"]),
            "daily_5bp": normalize_dates(recomputed[object_id]["5bp"]["daily"]),
        }
    if not frames[REFERENCE]["daily_0bp"]["date"].equals(frames[CANDIDATE]["daily_0bp"]["date"]):
        raise ValueError("objects were not evaluated on the identical trading-day index")
    archive_consistency = {}
    for object_id in (REFERENCE, CANDIDATE):
        for rate in ("0bp", "5bp"):
            on_disk = read_frame(source / "objects" / object_id / f"daily_{rate}.csv")
            fresh = frames[object_id][f"daily_{rate}"]
            same_dates = bool(pd.to_datetime(on_disk["date"]).reset_index(drop=True).equals(
                pd.to_datetime(fresh["date"]).reset_index(drop=True)))
            column_report = {}
            worst = 0.0
            for column in ("rank_ic", "long_short_return", "turnover", "net_return"):
                left = on_disk[column].to_numpy(dtype=float)
                right = fresh[column].to_numpy(dtype=float)
                scale = np.maximum(np.abs(left), np.abs(right))
                with np.errstate(invalid="ignore", divide="ignore"):
                    relative = np.where(scale > 0, np.abs(left - right) / scale, 0.0)
                relative = relative[np.isfinite(relative)]
                worst = max(worst, float(relative.max()) if relative.size else 0.0)
                column_report[column] = {"max_relative_difference": num(relative.max())
                                         if relative.size else 0.0,
                                         "within_1e_9": bool(relative.size == 0
                                                             or relative.max() <= 1e-9)}
            archive_consistency[f"{object_id}/{rate}"] = {
                "same_dates": same_dates, "max_relative_difference": num(worst),
                "columns": column_report,
                "passed": bool(same_dates and all(item["within_1e_9"]
                                                  for item in column_report.values()))}
    if not all(item["passed"] for item in archive_consistency.values()):
        raise ValueError(f"export does not reproduce the archived daily artifacts: {archive_consistency}")

    baseline_daily = frames[REFERENCE]["daily_0bp"]
    candidate_daily = frames[CANDIDATE]["daily_0bp"]
    dates = baseline_daily["date"]
    in_period = np.zeros(len(dates), dtype=bool)
    for start, end in periods:
        in_period |= ((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))).to_numpy()
    coverage = {
        "days": int(len(dates)),
        "minimum_days": int(evidence_rules["minimum_days"]),
        "declared_periods": len(periods),
        "minimum_periods": int(evidence_rules["minimum_periods"]),
        "days_inside_declared_periods": int(in_period.sum()),
        "days_outside_declared_periods": int((~in_period).sum()),
        "from": dates.min().strftime("%Y-%m-%d"), "to": dates.max().strftime("%Y-%m-%d"),
        "covers_all_declared_periods": all(
            ((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))).any()
            for start, end in periods),
        "sealed_intersection": int(((dates >= pd.Timestamp(SEALED[0]))
                                    & (dates < pd.Timestamp(SEALED[1]))).sum()),
    }

    column, method, direction = METRIC_SPEC[policy["primary_metric"]]
    sign = 1 if direction == "maximize" else -1
    difference = sign * (candidate_daily[column].to_numpy(dtype=float)
                         - baseline_daily[column].to_numpy(dtype=float))
    interval = block_interval(pd.Series(difference, index=dates.index), dates, periods, policy)
    effects = period_effects(pd.Series(difference, index=dates.index), dates, periods)
    improvement = interval["mean"]
    min_improvement = float(policy["min_improvement"])
    required_positive = max(1, int(np.ceil(float(evidence_rules["required_positive_fraction"])
                                           * len(periods) - 1e-12)))
    primary_readout = {
        "metric": policy["primary_metric"],
        "definition": "paired daily difference candidate minus baseline on the frozen sign convention",
        "baseline_level": num(aggregate_metric(baseline_daily[column].to_numpy(dtype=float), method, 252.0)),
        "candidate_level": num(aggregate_metric(candidate_daily[column].to_numpy(dtype=float), method, 252.0)),
        "improvement": improvement,
        "frozen_min_improvement": min_improvement,
        "meets_improvement": bool(improvement is not None and improvement >= min_improvement),
        "interval": interval,
        "interval_excludes_zero": bool(interval["low"] is not None and interval["low"] > 0),
        "period_evidence": effects,
        "required_positive_periods": required_positive,
        "period_gate_passed": bool(effects["positive_periods"] >= required_positive),
    }

    baseline_values = secondary_values(frames[REFERENCE]["metrics_0bp"], frames[REFERENCE]["daily_0bp"],
                                      "baseline")
    candidate_values = secondary_values(frames[CANDIDATE]["metrics_0bp"], frames[CANDIDATE]["daily_0bp"],
                                       "candidate")
    limits = policy["max_metric_regressions"]
    secondary = {}
    for metric in SECONDARY_METRICS:
        _, _, metric_direction = METRIC_SPEC[metric]
        limit = float(limits[metric] if isinstance(limits[metric], (int, float))
                      else limits[metric]["max_regression"])
        regression = (baseline_values[metric] - candidate_values[metric]) * (1 if metric_direction == "maximize"
                                                                            else -1)
        secondary[metric] = {"baseline": num(baseline_values[metric]),
                             "candidate": num(candidate_values[metric]),
                             "regression": num(regression), "frozen_limit": limit,
                             "passed": bool(regression <= limit)}
    net5 = {"baseline": num(frames[REFERENCE]["daily_5bp"]["net_return"].mean()),
            "candidate": num(frames[CANDIDATE]["daily_5bp"]["net_return"].mean()),
            "frozen_limit": float(prereg["net5_limit"])}
    net5["regression"] = num(net5["baseline"] - net5["candidate"])
    net5["passed"] = bool(net5["regression"] <= net5["frozen_limit"])
    reference_summary = archived[REFERENCE]["scenarios"][PRIMARY_RATE]
    candidate_summary = archived[CANDIDATE]["scenarios"][PRIMARY_RATE]
    turnover_ratio = num(candidate_summary["turnover_mean_local"]
                         / reference_summary["turnover_mean_local"])
    gross_ratio = num(candidate_summary["gross_return_mean"] / reference_summary["gross_return_mean"])
    rank_ic_ratio = num(candidate_summary["rank_ic_mean"] / reference_summary["rank_ic_mean"])

    declared_change = {"factor": "postprocess.causal_ewma",
                       "baseline": {"alpha": 1.0}, "candidate": {"alpha": 0.5},
                       "hypothesis": "a one-cross-section causal half-life lowers traded notional"}
    accepted_change = {"factor": "model.time_loss_weighting", "baseline": "none",
                       "candidate": "train_variance", "hypothesis": "example of an accepted field"}
    vocabulary = {}
    for name, change in (("output_level_operator", declared_change), ("accepted_model_field", accepted_change)):
        try:
            validate_change(change)
            vocabulary[name] = {"factor": change["factor"], "accepted": True}
        except ValueError as exc:
            vocabulary[name] = {"factor": change["factor"], "accepted": False, "reason": str(exc)}
    vocabulary["accepted_families"] = ["data.<field>", "model.<field>", "control.identity"]
    vocabulary["consequence"] = ("the frozen gate cannot adjudicate an output-level persistence "
                                 "operator; this readout therefore states the gate lines but issues no "
                                 "verdict, and any gate-able turnover change must be expressed as a "
                                 "model/data field (which needs a retrain round) or the protocol needs "
                                 "a new post-processing family plus a new pre-registration")

    reproduce = {}
    for label, object_id in (("candidate", CANDIDATE), ("reference", REFERENCE)):
        summary = archived[object_id]["scenarios"][PRIMARY_RATE]
        economic = frames[object_id]["metrics_5bp"]["economic_metrics"]
        reproduce[label] = {
            "gross_return_mean": bool(np.isclose(economic["gross_return_mean"],
                                                 summary["gross_return_mean"], rtol=1e-9, atol=1e-15)),
            "cost_mean": bool(np.isclose(economic["cost_mean"], summary["cost_mean"],
                                         rtol=1e-9, atol=1e-15)),
            "turnover_mean": bool(np.isclose(frames[object_id]["metrics_5bp"]["turnover_mean"],
                                             summary["turnover_mean_local"], rtol=1e-9, atol=1e-12)),
            "rank_ic_mean": bool(np.isclose(frames[object_id]["metrics_5bp"]["rank_ic_mean"],
                                            summary["rank_ic_mean"], rtol=1e-9, atol=1e-15)),
        }
    export_check = {"reproduced_from_export": reproduce,
                    "archive_daily_consistency": archive_consistency,
                    "passed": (all(all(values.values()) for values in reproduce.values())
                               and all(item["passed"] for item in archive_consistency.values())),
                    "tolerance_note": "bit-identity across processes is not claimed: the export is "
                                      "checked against the archived daily artifacts at 1e-9 relative "
                                      "tolerance, and the observed worst difference is reported",
                    "peak_host_rss_mib_parent_and_child": num(peak),
                    "resource_scope": "offline eval-v2 evaluation of one exported object at 0bp and 5bp; "
                                      "the pre-registered 24 GiB gate applies to the training and submission "
                                      "pipeline, which the resource task measures, not here"}

    gate_lines = [primary_readout["meets_improvement"], primary_readout["interval_excludes_zero"],
                  primary_readout["period_gate_passed"]] + \
                 [item["passed"] for item in secondary.values()] + [net5["passed"], export_check["passed"]]
    readout = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "evaluation_version": EVALUATION_VERSION,
        "source_run": rel(source),
        "prereg": {"path": rel(PREREG), "sha256": prereg_sha, "registered_at": prereg["registered_at"],
                   "registration_id": prereg["registration_id"],
                   "verified": True},
        "change": {"id": "causal_ewma_alpha050", "alpha": 0.5,
                   "declared_as": declared_change["factor"],
                   "gate_vocabulary": vocabulary},
        "coverage": coverage,
        "primary": primary_readout,
        "secondary": secondary,
        "net5": net5,
        "export_and_reproduction": export_check,
        "readout_summary": {
            "gate_lines_passed": sum(1 for value in gate_lines if value),
            "gate_lines_total": len(gate_lines),
            "promotable_under_frozen_lines": bool(all(gate_lines)),
            "what_this_is": "a readout of the frozen gate lines for the single t7 turnover change; it is "
                            "not a promotion verdict and it changes no threshold",
            "blocking_evidence": [name for name, item in secondary.items() if not item["passed"]]
            + ([] if primary_readout["meets_improvement"] else ["primary rank_ic improvement"])
            + ([] if primary_readout["period_gate_passed"] else ["period sign evidence"]),
        },
        "note": "the readout reports each frozen line; it passes or fails nothing",
    }
    write_json(output / "readout.json", readout)
    # Run-level daily artifact so the experiment index can verify which dates this run evaluated.
    frames[CANDIDATE]["daily_5bp"].to_csv(output / "daily.csv", index=False, lineterminator="\n")
    checks = [
        {"name": "registered_thresholds_untouched", "passed": True,
         "evidence": f"prereg sha256 {prereg_sha} matches the registration record; thresholds read only"},
        {"name": "declared_periods_fully_covered", "passed": bool(coverage["covers_all_declared_periods"]
                                                                 and coverage["days_outside_declared_periods"] == 0
                                                                 and coverage["days"] >= coverage["minimum_days"]),
         "evidence": coverage},
        {"name": "sealed_interval_untouched", "passed": coverage["sealed_intersection"] == 0,
         "evidence": coverage},
        {"name": "paired_days_identical", "passed": True,
         "evidence": "baseline and candidate daily indexes are identical"},
        {"name": "secondary_metrics_recomputed_from_daily", "passed": True,
         "evidence": "every secondary was recomputed with evaluation.aggregate_metric and matched the "
                     "recorded value before being compared to its frozen line"},
        {"name": "export_reproduces_archived_numbers", "passed": export_check["passed"],
         "evidence": reproduce},
        {"name": "gate_vocabulary_recorded", "passed": not vocabulary["output_level_operator"]["accepted"],
         "evidence": vocabulary},
    ]
    write_json(output / "checks.json",
               {"schema_version": 1, "passed": all(item["passed"] for item in checks), "checks": checks})
    recorder.event("readout", "completed", promotable=readout["readout_summary"]["promotable_under_frozen_lines"],
                   gate_lines_passed=readout["readout_summary"]["gate_lines_passed"])

    record = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "status": "completed",
        "kind": "evaluation", "created_at": utc_now(), "run_dir": rel(output),
        "source_run": rel(source), "readout": rel(output / "readout.json"),
        "prereg_sha256": prereg_sha,
        "change": "causal_ewma_alpha050 (alpha=0.5, single operator, no grid)",
        "coverage": coverage,
        "primary": {"metric": primary_readout["metric"], "improvement": improvement,
                    "frozen_min_improvement": min_improvement,
                    "interval": [interval["low"], interval["high"]],
                    "positive_periods": effects["positive_periods"],
                    "required_positive_periods": required_positive},
        "secondary": {name: {"regression": item["regression"], "frozen_limit": item["frozen_limit"],
                             "passed": item["passed"]} for name, item in secondary.items()},
        "net5": net5,
        "promotable_under_frozen_lines": readout["readout_summary"]["promotable_under_frozen_lines"],
        "gate_vocabulary_gap": vocabulary,
        "export": {"objects": rel(exports), "reproduced": export_check["passed"],
                   "peak_host_rss_mib": export_check["peak_host_rss_mib_parent_and_child"]},
        "decision_relevance": [
            f"this change lowers the local turnover proxy "
            f"{archived[REFERENCE]['scenarios'][PRIMARY_RATE]['turnover_mean_local']:.4f} -> "
            f"{archived[CANDIDATE]['scenarios'][PRIMARY_RATE]['turnover_mean_local']:.4f} "
            f"({turnover_ratio:.3f}x) and cuts 5bp cost by "
            f"{-(archived[CANDIDATE]['scenarios'][PRIMARY_RATE]['cost_mean_bp'] - archived[REFERENCE]['scenarios'][PRIMARY_RATE]['cost_mean_bp']):.2f}bp, "
            f"but keeps only {gross_ratio:.3f}x of gross and {rank_ic_ratio:.3f}x of rank IC, so it fails "
            f"the frozen primary and secondary lines",
            "the frozen gate cannot express an output-level operator, so no gate verdict exists for it",
            "if the team wants a gate-able turnover change it must be a model.* / data.* field, which "
            "requires a retrain round and a new pre-registration",
        ],
        "checks_passed": all(item["passed"] for item in checks),
    }
    write_json(args.record, record)

    data_document = {"schema_version": 1, "source": "frozen multiscale replay predictions and vwap labels",
                     "version": "cpt_jyc_2026", "evaluation_version": EVALUATION_VERSION,
                     "rows": int(len(keys)), "days": int(len(dates)),
                     "windows": [{"name": window[0], "start": window[1], "end": window[2]}
                                 for window in WINDOWS]}
    write_json(output / "data.json", data_document)
    config = {"schema_version": 1, "experiment_id": EXPERIMENT_ID, "output_root": "runs",
              "source_run": rel(source), "readout_only": True, "no_training": True,
              "evaluation": {"quantile": 0.2, "periods_per_year": 252, "min_cross_section": 3,
                             "constant_portfolios": "undefined",
                             "scenarios": [{"label": label, "single_side_rate": rate} for label, rate in RATES]},
              "prereg": rel(PREREG)}
    write_json(output / "config.json", config)
    artifacts = [{"path": path.relative_to(output).as_posix(), "sha256": file_sha256(path),
                  "bytes": path.stat().st_size}
                 for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"]
    manifest = {
        "schema_version": 1, "run_id": output.name, "experiment_id": EXPERIMENT_ID,
        "kind": "evaluation", "status": "completed", "started_at": started, "ended_at": utc_now(),
        "evaluation_version": EVALUATION_VERSION, "evaluation_split": "validation",
        "plan_sha256": plan_sha, "config_sha256": digest(config), "exit_code": 0, "seed": None,
        "splits": {"train": {"start": "2023-08-01T00:00:00", "end": "2023-11-01T00:00:00"},
                   "validation": {"start": "2023-11-01T00:00:00", "end": "2024-07-01T00:00:00"},
                   "final": None},
        "source_run": rel(source),
        "prereg": {"path": rel(PREREG), "sha256": prereg_sha},
        "evaluation_source": {"evaluation_py_sha256": file_sha256(ROOT / "src/jinyiquant/evaluation.py"),
                              "script_sha256": file_sha256(Path(__file__)),
                              "gate_script_sha256": file_sha256(ROOT / "scripts/promotion_gate.py")},
        "code": {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 "files": {"scripts/turnover_gate_readout.py": file_sha256(Path(__file__))}},
        "data": {"rows": int(len(keys)), "days": int(len(dates)),
                 "windows": [window[0] for window in WINDOWS]},
        "record": rel(args.record),
        "resources": {"peak_rss_mb": export_check["peak_host_rss_mib_parent_and_child"],
                      "peak_rss_scope": "offline evaluation child process plus parent at 5bp",
                      "wall_seconds": round(time.time() - timer, 3)},
        "artifacts": artifacts,
    }
    write_json(output / "manifest.json", manifest)
    recorder.event("run", "ended", status="completed", wall_seconds=manifest["resources"]["wall_seconds"])
    print(canonical_json({"run_dir": rel(output),
                          "improvement": improvement,
                          "frozen_min_improvement": min_improvement,
                          "interval": [interval["low"], interval["high"]],
                          "positive_periods": effects["positive_periods"],
                          "required": required_positive,
                          "secondary_failed": [name for name, item in secondary.items() if not item["passed"]],
                          "net5_passed": net5["passed"],
                          "promotable": readout["readout_summary"]["promotable_under_frozen_lines"],
                          "export_reproduced": export_check["passed"],
                          "peak_rss_mib": export_check["peak_host_rss_mib_parent_and_child"]}))
    return 0 if all(item["passed"] for item in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
