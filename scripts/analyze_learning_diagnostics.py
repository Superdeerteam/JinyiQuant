"""Independently audit completed diagnostic predictions and their target scale."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.checks import check, verify_artifacts
from jinyiquant.io import read_json, utc_now, write_json
from jinyiquant.tracking import Recorder, capture_source, environment, finish


def independent_daily(predictions, targets):
    keys = ["date", "instrument"]
    if predictions.duplicated(keys).any() or targets.duplicated(keys).any():
        raise ValueError("Independent audit requires unique prediction and answer keys")
    joined = predictions.merge(targets[keys + ["target"]], on=keys, validate="one_to_one")
    if len(joined) != len(targets) or not np.isfinite(joined[["score", "target"]]).all().all():
        raise ValueError("Independent audit requires complete finite answer alignment")
    rows = []
    for timestamp, section in joined.groupby("date"):
        ranks = section[["score", "target"]].rank(method="average").to_numpy()
        values = section[["score", "target"]].to_numpy()
        rows.append({"date": pd.Timestamp(timestamp).normalize(),
                     "rank_ic": float(np.corrcoef(ranks.T)[0, 1]) if (np.ptp(ranks, axis=0) > 0).all() else np.nan,
                     "ic": float(np.corrcoef(values.T)[0, 1]) if (np.ptp(values, axis=0) > 0).all() else np.nan})
    result = pd.DataFrame(rows).groupby("date", as_index=False)[["rank_ic", "ic"]].mean()
    result["date"] = result.date.dt.strftime("%Y-%m-%d")
    return result


def analyze(reference, destination):
    reference, destination = Path(reference).resolve(), Path(destination).resolve()
    manifest = read_json(reference / "manifest.json")
    if manifest["status"] != "completed":
        raise ValueError("Analyze only completed diagnostics; preserve partial attempts separately")
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    experiment_id = manifest["experiment_id"] + "-audit"
    recorder = Recorder(destination, experiment_id)
    write_json(destination / "manifest.json", {"schema_version": 1, "run_id": destination.name,
        "experiment_id": experiment_id, "status": "running", "started_at": utc_now(),
        "reference": str(reference), "code": capture_source(destination), "environment": environment()})
    recorder.event("run", "started", reference=str(reference), raw_data_reopened=False)
    try:
        checks = [check("reference_artifacts", verify_artifacts(reference)["passed"])]
        rows, predictions = [], {}
        for attempt in read_json(reference / "attempts.json"):
            path = Path(attempt["run"])
            keys = (attempt["period"], attempt["seed"])
            arm = attempt["arm"]
            permutation = pd.read_parquet(reference / f"{keys[0]}_{keys[1]}_permutation.parquet")
            variance = float(np.var(permutation.target.to_numpy(dtype=float)))
            if variance <= 0:
                raise ValueError("Training answers are constant; scale ratio undefined")
            predicted = pd.read_parquet(path / "predictions.parquet")
            targets = pd.read_parquet(reference / (keys[0] + "_data") / "targets.parquet")
            independent = independent_daily(predicted, targets)
            stored = pd.read_csv(path / "daily.csv")
            checked = independent.merge(stored, on="date", suffixes=("_independent", "_stored"),
                                         validate="one_to_one", how="outer")
            matched = len(checked) == len(independent) == len(stored)
            for metric in ("rank_ic", "ic"):
                matched = matched and np.allclose(checked[metric + "_independent"], checked[metric + "_stored"],
                                                  atol=1e-12, rtol=1e-10, equal_nan=True)
            checks.append(check("independent_correlations", matched, **attempt))
            metrics = read_json(path / "metrics.json")
            initial = pd.read_parquet(path / "initial_predictions.parquet")
            initial_daily = independent_daily(initial, targets)
            rows.append({**attempt, "training_target_variance": variance,
                "train_mse": metrics["train_mse"], "train_mse_to_constant_ratio": metrics["train_mse"] / variance,
                "prediction_std": float(predicted.score.std(ddof=0)),
                "evaluation_target_std": float(targets.target.std(ddof=0)),
                "initial_rank_ic": float(initial_daily.rank_ic.mean()),
                "rank_ic": float(independent.rank_ic.mean()), "ic": float(independent.ic.mean())})
            predictions.setdefault(keys, {})[arm] = predicted
        pairs = []
        for (period, seed), arms in predictions.items():
            paired = arms["correct"].merge(arms["shuffled"], on=["date", "instrument"],
                                          suffixes=("_correct", "_shuffled"), validate="one_to_one", how="outer")
            finite = np.isfinite(paired[["score_correct", "score_shuffled"]]).all().all()
            checks.append(check("identical_pair_keys", finite, period=period, seed=seed))
            pairs.append({"period": period, "seed": seed, "prediction_correlation": float(np.corrcoef(
                paired[["score_correct", "score_shuffled"]].to_numpy().T)[0, 1])})
        passed = all(item["passed"] for item in checks)
        diagnostic = read_json(reference / "metrics.json")
        underfit = all(row["train_mse_to_constant_ratio"] > 1 for row in rows)
        result = {"schema_version": 1, "passed": passed, "runs": rows, "pairs": pairs,
            "findings": {"small_sample_learning_passed": diagnostic["small_sample"]["learning_passed"],
                "same_seed_reproduction_passed": diagnostic["small_sample"]["reproduction"]["passed"],
                "negative_control_status": diagnostic["negative_control"]["status"],
                "all_control_runs_worse_than_fitted_constant_on_train": underfit,
                "training_fit_conclusion": "underfit_at_frozen_budget" if underfit else "see_individual_runs",
                "no_quality_improvement_claim": True, "model_promoted": False},
            "interpretation": {"scale_ratio": "Training MSE divided by best fitted constant MSE; >1 is worse",
                "independent_metrics": "NumPy correlation of original values / average ranks, then equal timestamp/day means",
                "limits": "Short, fixed two-epoch control runs; high prediction correlation does not identify a leakage source",
                "decision": "Diagnostic evidence only; no promotion or additional tuning"}}
        write_json(destination / "metrics.json", result)
        write_json(destination / "checks.json", {"passed": passed, "checks": checks})
        recorder.event("evaluation", "completed", passed=passed)
        finish(destination, "completed" if passed else "failed", 0 if passed else 1,
               resources={"wall_seconds": time.monotonic() - started})
        return result
    except BaseException as exc:
        recorder.failure("audit", exc)
        finish(destination, "failed", 1, error=str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.reference, args.output)
    print(json.dumps({"passed": result["passed"], "archive": str(args.output.resolve())}))
    raise SystemExit(0 if result["passed"] else 1)
