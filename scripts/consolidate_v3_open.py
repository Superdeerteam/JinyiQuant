"""Consolidate the closed V3-open round into a versioned research record.

The seed-11 opening-residual candidate was screened against the gate frozen in
``configs/v3_open_20261024_gate.json`` and failed the primary improvement family, so seed 29 was
never trained, nothing was packaged and nothing was submitted; V2-CSZ stays the submission model.
This script only archives already-produced artefacts into ``modified_tlob/v3_open_20261024.json``:
it trains nothing, scores nothing and edits no threshold.  It refuses to write when the frozen gate
hash, the screen verdict or the invariance review disagree with the recorded evidence.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jinyiquant.io import file_sha256, read_json, utc_now, write_json  # noqa: E402

EXPERIMENT = "v3-open-20261024"
GATE_PATH = Path("configs/v3_open_20261024_gate.json")
MODEL_CONFIG = Path("configs/model_v3_open_0930_residual.json")
PREFLIGHT = Path("runs/v3-open-20261024/preflight.json")
BASELINE_METRICS = Path("runs/v3-open-20261024/baseline_metrics.json")
RECOMPUTE = Path("runs/v3-open-20261024/baseline_recompute_check.json")
REVIEW = Path("runs/v3-open-20261024/invariance_review.json")
SCREEN = Path("runs/v3-open-20261024/seed11_gate.json")
EVAL = Path("runs/v3-open-20261024/seed11/open_residual_eval.json")
STYLE = Path("runs/v3-open-20261024/seed11/delta_style_diagnostic.json")
TRAINING = Path("runs/v3-open-20261024/seed11/training_summary.json")
DEFAULT_OUTPUT = Path("modified_tlob/v3_open_20261024.json")
V2_PACKAGE = Path("D:/进益资本杯/dist/v2-submission-20261023.zip")
V2_PLATFORM_SCORE = 0.23340
MONTHS = ("nov2023", "jan2024", "mar2024")
VIEWS = ("processed", "raw_full", "raw_common")


def resolve(path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def artifact(relative_path) -> dict:
    path = resolve(relative_path)
    return {"path": Path(relative_path).as_posix(), "bytes": path.stat().st_size,
            "sha256": file_sha256(path)}


def directory_artifact(relative_path) -> dict:
    path = resolve(relative_path)
    files = [item for item in path.rglob("*") if item.is_file()]
    return {"path": Path(relative_path).as_posix(), "files": len(files),
            "bytes": sum(item.stat().st_size for item in files)}


def mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def check_map(screen: dict) -> dict[str, bool]:
    return {item["id"]: bool(item["passed"]) for item in screen["checks"]}


def monthly_block(screen: dict, evaluation: dict, gate: dict) -> dict:
    primary = screen["measurements"]["primary_processed_0930_rank_ic"]
    floors = gate["thresholds"]["monthly_non_degradation"]["floors"]
    candidate = evaluation["nine_thirty"]["processed"]["candidate"]["monthly"]
    block = {}
    for month in MONTHS:
        delta = primary["month_deltas"][month]
        block[month] = {
            "days": candidate[month]["days"],
            "baseline_rank_ic_mean": candidate[month]["rank_ic_mean"] - delta,
            "candidate_rank_ic_mean": candidate[month]["rank_ic_mean"],
            "rank_ic_delta": delta,
            "floor": floors[month],
            "meets_floor": delta >= floors[month],
            "candidate_long_short_mean": candidate[month]["long_short_return_mean"],
        }
    return block


def guardrail_block(screen: dict, gate: dict, checks: dict[str, bool]) -> dict:
    measured = screen["measurements"]["full_day_processed_guardrails"]
    thresholds = gate["thresholds"]["degradation_guardrails"]
    reference = gate["baseline"]["headline"]
    rows = {
        "turnover_full_day_processed": {
            "baseline": thresholds["turnover"]["baseline"],
            "candidate": measured["turnover_full_day_processed"],
            "threshold": thresholds["turnover"]["threshold"], "rule": thresholds["turnover"]["rule"],
        },
        "rank_ic_ir_full_day_processed": {
            "baseline": thresholds["rank_ic_ir"]["baseline"],
            "candidate": measured["rank_ic_ir_full_day_processed"],
            "threshold": thresholds["rank_ic_ir"]["threshold"], "rule": thresholds["rank_ic_ir"]["rule"],
        },
        "long_short_mean_full_day_processed": {
            "baseline": thresholds["long_short_mean"]["baseline"],
            "candidate": measured["long_short_mean_full_day_processed"],
            "threshold": thresholds["long_short_mean"]["threshold"],
            "rule": thresholds["long_short_mean"]["rule"],
        },
        "long_short_sharpe_full_day_processed": {
            "baseline": thresholds["long_short_sharpe"]["baseline"],
            "candidate": measured["long_short_sharpe_full_day_processed"],
            "threshold": thresholds["long_short_sharpe"]["threshold"],
            "rule": thresholds["long_short_sharpe"]["rule"],
        },
        "worst_day_full_day_processed": {
            "baseline": thresholds["worst_day"]["baseline"],
            "candidate": measured["worst_day_full_day_processed"],
            "candidate_date": measured["worst_day_date"],
            "threshold": thresholds["worst_day"]["threshold"], "rule": thresholds["worst_day"]["rule"],
        },
        "rank_ic_1000_raw_full": {
            "baseline": reference["rank_ic_1000_raw_full_window_mean"],
            "candidate_delta": screen["measurements"]["ten_oclock_raw_full"]["pooled_delta_mean"],
            "threshold": thresholds["rank_ic_1000_raw_full"]["threshold"],
            "rule": thresholds["rank_ic_1000_raw_full"]["rule"],
        },
    }
    ids = {"turnover_full_day_processed": "guardrail_turnover",
           "rank_ic_ir_full_day_processed": "guardrail_rank_ic_ir",
           "long_short_mean_full_day_processed": "guardrail_long_short_mean",
           "long_short_sharpe_full_day_processed": "guardrail_long_short_sharpe",
           "worst_day_full_day_processed": "guardrail_worst_day",
           "rank_ic_1000_raw_full": "guardrail_rank_ic_1000_raw_full"}
    for name, row in rows.items():
        row["passed"] = checks[ids[name]]
    return rows


def stress_block(screen: dict, checks: dict[str, bool]) -> dict:
    buckets = screen["measurements"]["stress_processed"]
    block = {}
    for name, bucket in buckets.items():
        block[name] = {**bucket,
                       "rank_ic_passed": checks[f"stress_{name}_rank_ic"],
                       "long_short_passed": checks[f"stress_{name}_long_short"]}
    return block


def per_clock_block(evaluation: dict) -> dict:
    block = {}
    for view in VIEWS:
        candidate_rows = {row["time"]: row for row in evaluation["per_clock"][view]}
        baseline_rows = {row["time"]: row for row in evaluation["baseline_per_clock"][view]}
        rows = []
        for time in candidate_rows:
            cand, base = candidate_rows[time], baseline_rows[time]
            rows.append({
                "time": time, "days": cand["days"],
                "baseline_rank_ic_mean": base["rank_ic_mean"],
                "candidate_rank_ic_mean": cand["rank_ic_mean"],
                "rank_ic_delta": cand["rank_ic_mean"] - base["rank_ic_mean"],
                "baseline_ic_mean": base["ic_mean"], "candidate_ic_mean": cand["ic_mean"],
                "baseline_long_short_mean": base["long_short_return_mean"],
                "candidate_long_short_mean": cand["long_short_return_mean"],
                "baseline_turnover_mean": base["turnover_mean"],
                "candidate_turnover_mean": cand["turnover_mean"],
            })
        block[view] = rows
    return block


def build_record() -> dict:
    gate = read_json(resolve(GATE_PATH))
    preflight = read_json(resolve(PREFLIGHT))
    screen = read_json(resolve(SCREEN))
    evaluation = read_json(resolve(EVAL))
    style = read_json(resolve(STYLE))
    training = read_json(resolve(TRAINING))
    review = read_json(resolve(REVIEW))
    recompute = read_json(resolve(RECOMPUTE))
    model_config = read_json(resolve(MODEL_CONFIG))

    gate_sha = file_sha256(resolve(GATE_PATH))
    if screen["gate"]["sha256"] != gate_sha or not screen["gate"]["thresholds_verified_unchanged"]:
        raise SystemExit("frozen gate changed after the seed-11 screen; refusing to write")
    if screen["verdict"] != "fail":
        raise SystemExit(f"expected the recorded seed-11 screen to fail, got {screen['verdict']!r}")
    if evaluation["inputs"]["gate_sha256"] != gate_sha:
        raise SystemExit("evaluation was produced against a different gate")
    if review["verdict"] != "pass" or review["summary"]["checks_failed"]:
        raise SystemExit("invariance review is not a clean pass")

    checks = check_map(screen)
    measured = screen["measurements"]
    candidate = screen["candidate"]
    head = candidate["open_residual"]
    package = {"path": str(V2_PACKAGE), "exists": V2_PACKAGE.is_file()}
    if V2_PACKAGE.is_file():
        package.update({"bytes": V2_PACKAGE.stat().st_size, "sha256": file_sha256(V2_PACKAGE),
                        "mtime_utc": mtime_utc(V2_PACKAGE)})

    return {
        "schema_version": 1,
        "revision": 1,
        "experiment_id": EXPERIMENT,
        "created_at_utc": utc_now(),
        "task": "T10 finalise: archive the failed V3-open seed-11 screen and commit the branch",
        "hypothesis": {
            "statement": model_config["hypothesis"],
            "expected_improvement": (
                "pooled processed 09:30 RankIC delta versus V2-CSZ >= one baseline block-5 standard "
                "error (0.004156918), with the raw view not degrading and no full-day guardrail breach"
            ),
            "acceptance": "pass every check of the pre-registered gate on seed 11 before training seed 29",
        },
        "branch": {
            "name": preflight["branch"]["name"],
            "base_commit": preflight["base_commit"]["sha"],
            "base_commit_short": preflight["base_commit"]["sha_short"],
            "worktree": preflight["worktree"]["path"],
            "main_merged": preflight["branch"]["main_merged"],
            "commit_message": "V3-open 09:30 开盘残差头：seed11 未过风格中性准入，保留 V2-CSZ 基线",
        },
        "scope": gate["scope"],
        "preregistration": {
            "gate_id": gate["gate_id"],
            "path": artifact(GATE_PATH),
            "created_at_utc": gate["created_at_utc"],
            "frozen_before_head_training": gate["frozen_before_head_training"],
            "freeze_evidence": gate["freeze_evidence"],
            "decision_rules": gate["decision_rules"],
            "thresholds": gate["thresholds"],
            "evaluation_protocol": gate["evaluation_protocol"],
        },
        "baseline": {
            "version": gate["baseline"]["version"],
            "seed": gate["baseline"]["seed"],
            "weights_sha256": gate["baseline"]["weights_sha256"],
            "base_state_sha256": screen["candidate"]["baseline"]["base_state_sha256"],
            "replay_run_id": gate["baseline"]["replay_run_id"],
            "replay_run_dir": gate["baseline"]["replay_run_dir"],
            "headline": gate["baseline"]["headline"],
            "noise_units": gate["baseline"]["noise_units"],
            "recompute_check": {
                "all_passed": recompute["all_passed"],
                "gate_self_test_passed": recompute["gate_self_test"]["all_passed"],
                "rank_ic_0930_abs_diff": recompute["headline"]["rank_ic_0930_abs_diff"],
                "rank_ic_1000_abs_diff": recompute["headline"]["rank_ic_1000_abs_diff"],
                "per_window_max_metric_abs_diff": {
                    month: recompute["per_window"][month]["max_metric_abs_diff"] for month in MONTHS
                },
            },
            "submission_score": {
                "platform_total_score": V2_PLATFORM_SCORE,
                "provenance": (
                    "user-reported platform total score for the V2 submission recorded in the team plan "
                    "(T10); it is not reproducible locally and no V3-open package was ever submitted"
                ),
            },
        },
        "training": {
            "run_id": training["archive"]["run_id"],
            "archive": directory_artifact(
                Path("runs/v3-open-20261024/open-residual") / training["archive"]["run_id"]
            ),
            "status": training["archive"]["status"],
            "started_at_utc": training["archive"]["started_at"],
            "ended_at_utc": training["archive"]["ended_at"],
            "invocation": training["invocation"],
            "budget": {
                "declared_config_max_steps": model_config["model"]["max_steps"],
                "head_steps": head["steps"],
                "learning_rate": head["learning_rate"],
                "batch_size": head["batch_size"],
                "optimizer": head["optimizer"],
                "optimizer_scope": head["optimizer_scope"],
                "training_samples": head["samples"],
                "train_range": [model_config["splits"]["train"]["start"],
                                model_config["splits"]["train"]["end"]],
                "sample_time": head["sample_time"],
                "target_centering": head["target_centering"],
                "target_space": head["target_space"],
            },
            "actual": {
                "steps": head["steps"],
                "samples": head["samples"],
                "trunk_parameters": head["trunk_parameters"],
                "head_parameters": head["head_parameters"],
                "initial_mse": head["initial_mse"],
                "final_mse": head["final_mse"],
                "mse_reduction": head["mse_reduction"],
                "wall_seconds": training["resources"]["wall_seconds"],
                "cpu_seconds": training["resources"]["cpu_seconds"],
                "device": training["resources"]["device"],
                "gpu_peak_memory_mb": training["resources"]["gpu_peak_memory_mb"],
                "rss_mb": training["resources"]["rss_mb"],
            },
            "in_sample_prediction_units": training["in_sample"],
            "weights_sha256": training["model"]["weights_sha256"],
            "checkpoint_sha256": training["model"]["checkpoint_sha256"],
            "checkpoint_schema_version": training["model"]["checkpoint_schema_version"],
            "base_weights_file_sha256": training["model"]["base_weights_sha256"],
            "base_state_sha256": training["model"]["base_state_sha256"],
            "checks": training["checks"],
            "data_identity": training["data_identity"],
            "capacity_note": training["capacity_note"],
        },
        "implementation": {
            "files": review["reviewed_files_sha256"],
            "checkpoint": {
                "schema_version": training["model"]["checkpoint_schema_version"],
                "head_state_sha256": review["head"]["trained_head_state_sha256"],
                "zero_init_correction_max_abs": review["head"]["zero_init_correction_max_abs"],
                "legacy_schemas_still_load": True,
            },
            "invariance_review": {
                "verdict": review["verdict"],
                "reviewed_base_commit": review["reviewed_base_commit"],
                "checks_passed": review["summary"]["checks_passed"],
                "checks_total": review["summary"]["checks_total"],
                "max_abs_diffs": review["max_abs_diffs"],
                "pytest": review["summary"]["pytest"],
                "observations": review["observations"],
            },
            "decisions": [
                "trunk freeze is enforced by the optimiser scope plus inference-mode trunk outputs; "
                "requires_grad_(False) on the trunk switches oneDNN to packed kernels and perturbs "
                "non-opening outputs by ~1e-7 relative",
                "only history-valid 09:30 keys with an exchange time of 09:30 receive the correction; "
                "every other key is bit-identical to V2-CSZ and missing-history keys fall back to the "
                "frozen intercept",
            ],
        },
        "results": {
            "windows": evaluation["windows"],
            "primary_processed_0930_rank_ic": measured["primary_processed_0930_rank_ic"],
            "raw_full_0930_rank_ic": measured["raw_full_0930_rank_ic"],
            "monthly": monthly_block(screen, evaluation, gate),
            "full_day_processed_guardrails": guardrail_block(screen, gate, checks),
            "stress_processed": stress_block(screen, checks),
            "per_clock": per_clock_block(evaluation),
            "nine_thirty_candidate": {
                key: value for key, value in evaluation["nine_thirty"]["processed"]["candidate"].items()
                if key != "monthly"
            },
            "nine_thirty_paired_delta": evaluation["nine_thirty"]["processed"]["paired_delta"],
            "full_day_paired_delta": evaluation["full_day"]["processed"]["paired_delta"],
            "prediction_invariance": {
                month: candidate["predictions"][month] for month in MONTHS
            },
            "daily_diffs": evaluation["daily_diffs"],
            "daily_full_day": evaluation["daily_full_day"],
            "reproduction": evaluation["reproduction"],
            "evaluation_checks": evaluation["checks"],
            "evaluation_resources": evaluation["resources"],
        },
        "gate_evidence": {
            "stage": screen["stage"],
            "created_at_utc": screen["created_at_utc"],
            "gate_sha256": gate_sha,
            "checks_total": len(screen["checks"]),
            "checks_passed": sum(1 for value in checks.values() if value),
            "checks_failed": sum(1 for value in checks.values() if not value),
            "checks": screen["checks"],
            "verification": screen["verification"],
            "protocol": screen["protocol"],
        },
        "style_diagnostic": {
            "source": artifact(STYLE),
            "purpose": style["purpose"],
            "gate_input": False,
            "timestamps": style["timestamps"],
            "style_industry_r2": style["style_industry_r2"],
            "delta_std_mean": style["delta_std_mean"],
            "corr_vs_baseline_score": style["corr_vs_baseline_score"],
            "style_correlations_mean": style["style_correlations_mean"],
        },
        "verdict": {
            "verdict": screen["verdict"],
            "stage": screen["stage"],
            "decision": screen["decision"],
            "failures": screen["failures"],
            "explanation": (
                "the raw 09:30 RankIC gain (+0.017226) does not survive the pre-registered style/industry "
                "residualized view (+0.000440, 10.6% of the noise floor), the block-5/block-2 intervals "
                "straddle zero and only 1/3 months improve, so the candidate is rejected"
            ),
            "kept_model": {
                "model": "V2-CSZ (v2_60d_csz, seed 11)",
                "platform_total_score": V2_PLATFORM_SCORE,
                "reason": "the frozen gate keeps the baseline unless every promotion check passes",
            },
            "seed29": {"trained": False, "artifacts_present": False},
            "packaged": False,
            "submitted": False,
            "why_not_submitted": [
                "the frozen gate rejected the candidate at the seed-11 screen, so seed 29, the package "
                "and the platform submission were never authorised",
                "V2-CSZ keeps the submission slot and its V2 package was not modified",
            ],
            "package_untouched": package,
        },
        "limitations": [
            *screen["limitations"],
            "the style diagnostic explains the delta but is not a gate input; it is archived as context",
            "the checkpoint payload carries seed 11 and schema_version 4 but no experiment/version label, "
            "so a packaging provenance step would have to supply the v3-open-20261024 tag",
            "no packaging or platform validation ran for V3-open because the gate stopped the round",
        ],
        "not_committed": {
            "data": "D:/进益资本杯/数据 and all parquet/feather inputs stay outside git (.gitignore)",
            "weights": "open-residual checkpoint, modified_tlob_weights.json and predictions stay in runs/",
            "runs": "runs/v3-open-20261024 (archives, JSON evidence, parquet) is ignored by git",
            "dist": "no V3-open package exists; the V2 package in D:/进益资本杯/dist is untouched",
        },
        "reproduction": {
            "artifacts": {
                "gate": artifact(GATE_PATH),
                "model_config": artifact(MODEL_CONFIG),
                "baseline_metrics": artifact(BASELINE_METRICS),
                "baseline_recompute_check": artifact(RECOMPUTE),
                "invariance_review": artifact(REVIEW),
                "seed11_gate": artifact(SCREEN),
                "seed11_eval": artifact(EVAL),
                "seed11_style_diagnostic": artifact(STYLE),
                "seed11_training_summary": artifact(TRAINING),
            },
            "commands": [
                "PYTHONPATH=src D:/进益资本杯/.venv-ensemble-cuda/Scripts/python.exe "
                "scripts/freeze_v3_open_gate.py",
                "PYTHONPATH=src D:/进益资本杯/.venv-ensemble-cuda/Scripts/python.exe "
                "scripts/train_open_residual.py configs/model_v3_open_0930_residual.json "
                "--base-weights D:/进益资本杯/runs/model-versions/v2-cloud-20261023/60d/"
                "modified_tlob_weights.json --expected-base-sha256 "
                "3e44451670320c282f7421b77d773bc2af8190cbdf02a62c2a1f7e6b9284c0aa --steps 1500 --device cuda",
                "PYTHONPATH=src D:/进益资本杯/.venv-ensemble-cuda/Scripts/python.exe "
                "scripts/predict_open_residual.py --checkpoint "
                "runs/v3-open-20261024/open-residual/20261025T052524.593107Z_5c4a968e2d0e/model.pt "
                "--output runs/v3-open-20261024/seed11",
                "PYTHONPATH=src D:/进益资本杯/.venv-ensemble-cuda/Scripts/python.exe "
                "scripts/evaluate_open_residual.py --seed-dir runs/v3-open-20261024/seed11",
                "PYTHONPATH=src D:/进益资本杯/.venv-ensemble-cuda/Scripts/python.exe "
                "scripts/consolidate_v3_open.py",
            ],
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    record = build_record()
    output = resolve(args.output)
    if output.exists():
        existing = read_json(output)
        if existing.get("experiment_id") != record["experiment_id"]:
            raise SystemExit(f"{output} belongs to another experiment; refusing to overwrite")
        if existing.get("verdict", {}).get("verdict") != record["verdict"]["verdict"]:
            raise SystemExit("refusing to rewrite a previously recorded verdict")
    write_json(output, record)
    checks = record["gate_evidence"]
    print(f"wrote {output.relative_to(ROOT).as_posix()} bytes={output.stat().st_size}")
    print(f"verdict={record['verdict']['verdict']} checks={checks['checks_passed']}/"
          f"{checks['checks_total']} failed={checks['checks_failed']} "
          f"delta={record['results']['primary_processed_0930_rank_ic']['pooled_delta_mean']:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
