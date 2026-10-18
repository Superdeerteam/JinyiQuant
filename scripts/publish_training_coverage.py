"""Audit and publish monthly-only coverage research evidence, without model execution."""

import argparse
from pathlib import Path

import pandas as pd

from jinyiquant.checks import verify_artifacts
from jinyiquant.io import file_sha256, read_json, utc_now, write_json
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir
from scripts.diagnose_training_coverage import INDEX, exported_detail, quarter_jobs

ROOT = Path(__file__).resolve().parents[1]


def records(frame):
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def publish(plan_path, history_run, inference_run, output, provenance=None):
    """No averages across months; paired differences always retain a timestamp/month."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new research record; never overwrite historical evidence")
    plan = read_json(plan_path)
    runs = [Path(plan["run_dir"]), Path(history_run), Path(inference_run)]
    for run in runs:
        if read_json(run / "manifest.json")["status"] != "completed" or not verify_artifacts(run)["passed"]:
            raise ValueError(f"Incomplete or modified evidence archive: {run}")
    checks = [read_json(run / "checks.json") for run in runs]
    if any(not c["passed"] or c["new_training"] != 0 or c["reserved_label_views"] != 0 for c in checks):
        raise ValueError("Research scope or artifact checks failed")
    if checks[2]["additional_inference"] > 6 or checks[2]["reused_combinations"] != 6:
        raise ValueError("Supplemental inference budget violated")
    index = read_json(INDEX)
    old = exported_detail()
    quarterly = pd.read_csv(Path(inference_run) / "by_timestamp.csv", parse_dates=["timestamp"])
    history = pd.read_csv(Path(history_run) / "by_timestamp.csv", parse_dates=["timestamp"])
    expected = pd.DatetimeIndex(old.loc[old.policy.eq("fixed"), "timestamp"]).sort_values()
    for frame in [quarterly, history]:
        actual = pd.DatetimeIndex(frame.timestamp).sort_values()
        if not actual.equals(expected):
            raise ValueError("Incomplete or duplicate monthly/eight-time evidence")
    baseline = old.loc[old.policy.eq("fixed")]
    counts = quarterly[["timestamp", "sample_count"]].merge(
        baseline[["timestamp", "sample_count", "universe_count"]], on="timestamp",
        suffixes=("_quarterly", "_original"), validate="one_to_one")
    if not counts.sample_count_quarterly.eq(counts.sample_count_original).all():
        raise ValueError("Quarterly and original scored populations differ")
    quality_counts = history[["timestamp", "universe_count"]].merge(
        counts[["timestamp", "universe_count"]], on="timestamp", suffixes=("_history", "_original"))
    if not quality_counts.universe_count_history.eq(quality_counts.universe_count_original).all():
        raise ValueError("Source quality and scoring universe denominators differ")
    coverage = ["universe_count", "prediction_count", "target_count", "target_coverage", "prediction_coverage"]
    quarterly = quarterly.merge(baseline[["timestamp", *coverage]], on="timestamp", validate="one_to_one")
    calendar_path = Path(plan["calendar_source"])
    if file_sha256(calendar_path) != plan["calendar_sha256"]:
        raise ValueError("Official universe calendar changed")
    days = pd.read_parquet(calendar_path, columns=["date"]).date.dt.normalize().unique()
    calendar = pd.DatetimeIndex(days).sort_values()
    origin_sources = {job["month"]: job for job in quarter_jobs(index)}
    last = pd.DatetimeIndex([origin_sources[t.strftime("%Y%m")]["weight"]["last_training_date"]
                            for t in quarterly.timestamp])
    current = pd.DatetimeIndex(quarterly.timestamp.dt.normalize())
    quarterly["weight_age_calendar_days"] = (current - last).days
    quarterly["weight_age_trading_days"] = calendar.searchsorted(current, side="right") - calendar.searchsorted(last, side="right")
    quarterly["last_training_date"] = last.strftime("%Y-%m-%d")
    comparisons = []
    for policy in ["fixed", "monthly"]:
        baseline = old.loc[old.policy.eq(policy)]
        columns = ["timestamp", "rank_ic", "ic", "long_short_return", "turnover"]
        paired = quarterly[columns + ["month", "time"]].merge(baseline[columns], on="timestamp",
                    validate="one_to_one", suffixes=("_quarterly", "_reference"))
        for metric in columns[1:]:
            paired[metric + "_delta"] = paired[metric + "_quarterly"] - paired[metric + "_reference"]
        paired["reference"] = policy
        comparisons.append(paired)
    paired = pd.concat(comparisons, ignore_index=True)
    output.mkdir(parents=True, exist_ok=False)
    files, monthly_rows = [], []
    for month, group in quarterly.groupby("month", sort=True):
        quality = history.loc[history.month.eq(month)]
        row = {"month": month, "origin": str(group.weight_fit_month.iloc[0]), "sections": len(group)}
        for metric in ["rank_ic", "ic", "long_short_return", "turnover"]:
            median = group[metric].median()
            row[metric + "_median"] = float(median) if pd.notna(median) else None
            row[metric + "_negative_sections"] = int(group[metric].lt(0).sum())
            row[metric + "_undefined_sections"] = int(group[metric].isna().sum())
        monthly_rows.append(row)
        doc = {"schema_version": 1, "month": month, "quarterly": records(group.assign(timestamp=group.timestamp.astype(str))),
               "paired_references": records(paired.loc[paired.month.eq(month)].assign(
                   timestamp=lambda x: x.timestamp.astype(str))),
               "history": records(quality.assign(timestamp=quality.timestamp.astype(str))),
               "monthly_summary": row,
               "month_time": records(group.groupby(["month", "time"], as_index=False)[
                   ["rank_ic", "long_short_return", "turnover"]].median())}
        target = output / f"{month}.json"
        write_json(target, doc)
        files.append({"path": target.name, "sha256": file_sha256(target)})
    summary = {"schema_version": 1, "created_at_utc": utc_now(), "status": "completed_with_label_clock_limitation",
               "publication_provenance": provenance,
               "hypothesis": plan["hypothesis"], "plan": str(Path(plan_path).resolve()),
               "plan_sha256": file_sha256(plan_path), "run_archives": [str(r.resolve()) for r in runs],
               "archive_manifests_sha256": {str(r): file_sha256(r / "manifest.json") for r in runs},
               "cost_seconds": {read_json(r / "manifest.json").get("kind", "freeze"):
                                (pd.Timestamp(read_json(r / "manifest.json")["ended_at"]) -
                                 pd.Timestamp(read_json(r / "manifest.json")["started_at"])).total_seconds()
                                for r in runs},
               "monthly_results": monthly_rows, "files": files,
               "checks": {"passed": True, "new_training": 0, "additional_weight_month_inferences": 6,
                          "reused_weight_month_combinations": 6, "months": 12, "sample_times": 8,
                          "timestamp_rows": len(quarterly), "reserved_label_views": 0,
                          "annual_aggregates_exported": False,
                          "actual_endpoint_verification_complete": checks[0]["actual_endpoint_verification_complete"]},
               "limitations": ["Official download lacks label_end. Midnight training boundaries are proven using the official no-overnight same-day-close upper bound; exact intraday/lunch endpoints remain unverified.",
                               "Local gross label returns omit fees, execution and official BARRA processing.",
                               "One historical year/seed; origin, weight age and market conditions are confounded. No causal age effect or 40-day improvement is established.",
                               "History ages refer to source timestamps, not proof that every quoted field economically refreshed.",
                               "Monthly history summaries describe timestamp-level stock distributions; do not interpret medians of medians as pooled-stock quantiles."],
               "decision": "Retain current 10-day seed11 reference. No training-window promotion; next stage requires baseline-only threshold and resource calibration.",
               "model_promoted": False, "literature": plan["literature"]}
    write_json(output / "index.json", summary)
    return output / "index.json"


def archive_publication(args):
    run = new_run_dir(ROOT / "runs/training-coverage-publication")
    rec = Recorder(run, "training-coverage-20261017")
    code = capture_source(run)
    write_json(run / "manifest.json", {"status": "running", "run_id": run.name,
               "experiment_id": rec.experiment_id, "kind": "monthly_evidence_export",
               "started_at": utc_now(), "code": code, "environment": environment()})
    rec.event("startup", "started", new_training=0, additional_inference=0)
    try:
        result = publish(args.plan, args.history_run, args.inference_run, args.output,
                         {"run_dir": str(run.resolve()), "code": code})
        write_json(run / "checks.json", read_json(result)["checks"])
        write_json(run / "metrics.json", {"output": str(result.resolve()), "sha256": file_sha256(result)})
        rec.event("export", "completed", output=str(result.resolve()))
        finish(run, "completed", 0)
        return result
    except BaseException as exc:
        rec.failure("export", exc)
        finish(run, "failed", 1, error={"reason": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/training_coverage_20261017_v3_frozen.json")
    parser.add_argument("--history-run", type=Path, required=True)
    parser.add_argument("--inference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "modified_tlob/training_coverage_20261017")
    args = parser.parse_args()
    print(archive_publication(args))
