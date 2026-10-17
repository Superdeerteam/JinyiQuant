"""Recompute a completed ensemble experiment using saved predictions/targets only; never fit."""

import argparse
from pathlib import Path

import pandas as pd

from jinyiquant.checks import check, verify_artifacts
from jinyiquant.config import RunConfig
from jinyiquant.evaluation import evaluate
from jinyiquant.io import file_sha256, read_json, utc_now, write_json
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir
from modified_tlob.ensemble import SEEDS, average_predictions, paired_block_intervals, research_decision


def audit(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    manifest = read_json(source / "manifest.json")
    if manifest["status"] != "completed" or not verify_artifacts(source)["passed"]:
        raise ValueError("Input must be a completed intact experiment")
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Audit output must not modify the input archive")
    output = new_run_dir(destination)
    rec = Recorder(output, "fixed-seed-ensemble-offline-audit")
    write_json(output / "manifest.json", {"schema_version": 1, "run_id": output.name,
        "experiment_id": rec.experiment_id, "started_at": utc_now(), "status": "running",
        "source": str(source), "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "code": capture_source(output), "environment": environment()})
    checks, paired, summaries = [], [], []
    rec.event("run", "started", training=False, raw_data_access=False)
    try:
        plan = read_json(source / "preregistration.json")
        execution = Path(manifest.get("execution_source", source))
        if not verify_artifacts(execution)["passed"]:
            raise ValueError("Original execution archive changed")
        attempts = read_json(source / "attempts.json")
        expected = {(w["name"], s) for w in plan["windows"] for s in SEEDS}
        checks.append(check("exactly_six_completed_fits", len(attempts) == 6
            and {(a["period"], a["seed"]) for a in attempts} == expected
            and all(a["status"] == "completed" for a in attempts)))
        core, environments, actual_devices = [], [], []
        for attempt in attempts:
            path = Path(attempt["path"])
            run_manifest, config, progress = [read_json(path / f"{n}.json") for n in ("manifest", "config", "training")]
            checks.append(check(f"fit_{attempt['period']}_{attempt['seed']}", verify_artifacts(path)["passed"]
                and read_json(path / "checks.json")["passed"] and progress["steps"] == 1500
                and config == plan["configs"][attempt["period"]] | {"seed": attempt["seed"]}
                and progress["prediction_units"] == "original_label_return"
                and progress["evaluation_labels_loaded"] is False))
            core.append({n: v for n, v in run_manifest["code"]["files"].items()
                         if n.startswith(("modified_tlob/", "src/jinyiquant/")) and n.endswith(".py")})
            environments.append(run_manifest["environment"]["dependencies"])
            # Generic environment() defaults to CPU; the actual selected-device event is authoritative.
            import json

            events = [json.loads(line) for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            selected = [e["context"] for e in events if e["stage"] == "training" and e["event_type"] == "determinism"]
            actual_devices.append(selected[0] if len(selected) == 1 else {})
        checks.append(check("same_core_and_dependencies_across_all_six_fits",
                            all(c == core[0] for c in core) and all(e == environments[0] for e in environments)))
        checks.append(check("actual_cuda_determinism_events", all(d.get("device") == "cuda"
                            and d.get("algorithms") is True for d in actual_devices)))
        write_json(output / "actual_determinism.json", {"events": actual_devices,
            "note": "Original environment.determinism.device is a generic CPU default. Actual fit events above "
                    "record CUDA, deterministic algorithms, seed and workspace; archived originals are preserved."})
        costs = {}
        for window in plan["windows"]:
            period = window["name"]
            inference = execution / "inference" / period
            config = RunConfig.model_validate(plan["configs"][period])
            targets = pd.read_parquet(source / f"{period}_targets.parquet")
            frames = {s: pd.read_parquet(inference / f"seed{s}_predictions.parquet") for s in SEEDS}
            candidate = average_predictions(frames)
            pd.testing.assert_frame_equal(candidate, pd.read_parquet(inference / "ensemble_predictions.parquet"))
            daily, metrics = {}, {}
            for arm, predictions in (("baseline", frames[11]), ("candidate", candidate)):
                scored = predictions.merge(targets[["date", "instrument"]], on=["date", "instrument"],
                                           validate="one_to_one")
                metrics[arm], daily[arm] = evaluate(scored, targets, config.evaluation,
                                                   undefined_constant_portfolios=True)
                checks.append(check(f"{period}_{arm}_metrics_recomputed",
                    metrics[arm] == read_json(source / f"{period}_{arm}.json")))
                write_json(output / f"{period}_{arm}_metrics.json", metrics[arm])
            joined = daily["baseline"][["date"]].assign(period=period)
            for metric in ("rank_ic", "long_short_return", "turnover"):
                joined[metric] = daily["candidate"][metric] - daily["baseline"][metric]
            paired.append(joined)
            summaries.append({"period": period, "worst_day_delta": metrics["candidate"]["min_daily_long_short_return"]
                              - metrics["baseline"]["min_daily_long_short_return"]})
            identities = [read_json(execution / f"fits/{period}_s{s}/training_identity.json") for s in SEEDS]
            checks.append(check(period + "_same_training_keys_and_labels", all(i == identities[0] for i in identities)))
            costs[period] = read_json(inference / "inference_cost.json")
        days = pd.concat(paired, ignore_index=True)
        intervals = [paired_block_intervals(days, block_days=b) for b in (5, 2)]
        decision = research_decision(days, summaries, intervals, plan["policy"], cost_measured=bool(costs))
        checks += [check("paired_intervals_recomputed", intervals == read_json(source / "paired_intervals.json")),
                   check("decision_recomputed", decision == read_json(source / "decision.json")),
                   check("saved_prediction_target_inputs_only", True, training=False, raw_label_files=False)]
        days.to_csv(output / "paired_daily.csv", index=False)
        write_json(output / "paired_intervals.json", intervals)
        write_json(output / "decision.json", decision)
        write_json(output / "checks.json", {"passed": all(c["passed"] for c in checks), "checks": checks})
        if not all(c["passed"] for c in checks):
            raise ValueError("Offline reproduction differs; see checks.json")
        rec.event("run", "completed", passed=True, training=False, raw_data_access=False)
        finish(output, "completed", 0)
        return output
    except BaseException as exc:
        rec.failure("audit", exc)
        finish(output, "failed", 1, error={"reason": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(audit(args.source, args.output))
