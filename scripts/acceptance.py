"""Run the complete engineering acceptance from a clean generated fixture.

Outputs remain under the explicitly supplied directory. This does not test alpha.
"""

import argparse
from pathlib import Path

from jinyiquant.checks import verify_artifacts
from jinyiquant.config import RunConfig, load_config
from jinyiquant.demo import create_demo
from jinyiquant.experiments import calibrate, suite
from jinyiquant.io import digest, read_json, write_json
from jinyiquant.runner import failed_config, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = create_demo(args.output)
    config = load_config(config_path)
    results = []
    original, code = run(config)
    results.append({"name": "complete_pipeline", "passed": code == 0, "path": str(original)})
    repeated, code = run(config, reference=original)
    results.append({"name": "reproducibility", "passed": code == 0, "path": str(repeated)})
    results.append({"name": "artifact_integrity", "passed": verify_artifacts(original)["passed"]})
    for field, value in (("wall_seconds", 0.01), ("max_rss_mb", 1)):
        raw = config.snapshot()
        raw["budget"][field] = value
        failed, code = run(RunConfig.model_validate(raw))
        manifest = read_json(failed / "manifest.json")
        results.append({"name": field + "_budget", "passed": code == 124 and manifest["status"] == "budget_exceeded",
                        "path": str(failed)})
    raw = config.snapshot()
    raw["data"]["bars_path"] = str(config_path.parent / "missing.parquet")
    failed, code = run(RunConfig.model_validate(raw))
    results.append({"name": "missing_data_failure", "passed": code != 0 and (failed / "failure.json").exists(),
                    "path": str(failed)})
    bad_path = config_path.parent / "invalid.json"
    write_json(bad_path, {"experiment_id": "bad", "api_key": "fake-private-value"})
    try:
        load_config(bad_path)
    except Exception as exc:
        failed, code = failed_config(bad_path, exc)
        events = (failed / "events.jsonl").read_text(encoding="utf-8")
        results.append({"name": "invalid_config_redaction", "passed": code == 2 and "fake-private-value" not in events,
                        "path": str(failed)})
    periods = [{"start": "2020-01-20", "end": "2020-01-27"}, {"start": "2020-01-27", "end": "2020-02-03"}]
    calibration, code = calibrate(config, [11, 29], periods)
    results.append({"name": "baseline_variation", "passed": code == 0, "path": str(calibration)})
    # Identity candidate: zero effect is known a priori; passing this MUST NOT produce acceptance.
    calibrated = read_json(calibration / "calibration.json")
    threshold = max(1e-12, calibrated["metrics"]["rank_ic_mean"]["std"])
    policy = {"primary_metric": "rank_ic_mean", "direction": "maximize", "min_improvement": threshold,
              "max_metric_regressions": {"long_short_return_mean": 0.0, "turnover_mean": 0.0},
              "max_wall_ratio": 10.0, "max_memory_ratio": 2.0, "min_seeds": 2, "min_periods": 2,
              "bootstrap_samples": 200, "block_days": 2, "confidence": 0.95, "seed": 123,
              "threshold_rationale": "Engineering identity control: noise measured across baseline seeds/periods; "
              "tiny practical floor tests rejection of zero effect. Cost limits are fixture limits.",
              "change": {"factor": "control.identity", "baseline": None, "candidate": None,
                         "hypothesis": "Identical arms cannot justify improvement"},
              "threshold_basis": {"calibration_report": str(calibration / "calibration.json"),
                                  "calibration_sha256": digest(calibrated),
                                  "practical_improvement": 1e-12, "noise_multiplier": 1.0}}
    plan = {"baseline_config": str(config_path), "candidate_config": str(config_path), "seeds": [11, 29],
            "periods": periods, "policy": policy, "calibration_report": str(calibration / "calibration.json")}
    plan_path = config_path.parent / "suite.json"
    write_json(plan_path, plan)
    comparison, code = suite(plan_path)
    decision = read_json(comparison / "decision.json")
    results.append({"name": "identity_candidate_not_promoted", "passed": code == 0
                    and decision["result"] == "回退" and not decision["accepted"], "path": str(comparison)})
    # Exercise a real, permitted data-recipe change without weakening sample fairness.
    candidate = config.snapshot()
    candidate["data"]["lookback"] += 1
    candidate_path = config_path.parent / "candidate-history.json"
    write_json(candidate_path, candidate)
    history_policy = {**policy, "change": {"factor": "data.lookback", "baseline": config.data.lookback,
        "candidate": candidate["data"]["lookback"], "hypothesis": "Engineering check of controlled history change"}}
    history_plan = {**plan, "candidate_config": str(candidate_path), "policy": history_policy}
    history_plan_path = config_path.parent / "suite-history.json"
    write_json(history_plan_path, history_plan)
    history, code = suite(history_plan_path)
    compared = read_json(history / "comparison.json")
    results.append({"name": "single_history_factor_with_fixed_evaluation", "passed": code == 0
                    and compared.get("change", {}).get("factor") == "data.lookback"
                    and len(compared.get("pairs", [])) == 4, "path": str(history)})
    report = {"schema_version": 1, "scope": "synthetic engineering evidence only", "checks": results,
              "passed": all(r["passed"] for r in results)}
    write_json(config_path.parent / "acceptance.json", report)
    print(config_path.parent / "acceptance.json", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
