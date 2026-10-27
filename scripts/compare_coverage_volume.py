"""Read-only paired comparison for the monthly-coverage vs update-volume study.

Plan schema: {schema_version, experiment_id, hypothesis, bootstrap_samples,
bootstrap_seed, block_days, arms: {label: run_dir}, pairs: [{label, baseline,
candidate, question}], seed_groups: {label: {seed: arm_label}}}.

Each arm directory must hold config.json, predictions.parquet and
targets.parquet.  Daily metrics are recomputed from those artifacts with the
same evaluator the training run uses, so archived and fresh runs stay
comparable; nothing is retrained and no artifact is modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.comparison import _block_indices, archived_cost_scenarios
from jinyiquant.config import load_config
from jinyiquant.evaluation import EVALUATION_VERSION, evaluate

SUMMARY = ("rank_ic_mean", "long_short_return_mean", "turnover_mean",
           "min_daily_long_short_return", "rank_ic_ir")
DAILY = ("rank_ic", "long_short_return", "turnover")


def _load_arm(directory: Path) -> dict:
    config = load_config(directory / "config.json")
    predictions = pd.read_parquet(directory / "predictions.parquet")
    targets = pd.read_parquet(directory / "targets.parquet")
    scored = predictions.merge(targets[["date", "instrument"]], on=["date", "instrument"],
                               how="inner", validate="one_to_one")
    metrics, daily, _ = evaluate(scored, targets, config.evaluation, return_sections=True,
                                 undefined_constant_portfolios=True)
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    return {"directory": str(directory), "keys": scored[["date", "instrument"]],
            "metrics": metrics, "daily": daily.set_index("date")[list(DAILY)],
            "seed": config.seed}


def _interval(days: np.ndarray, block_days: int, samples: int, seed: int) -> list[float] | None:
    if len(days) < 2:
        return None
    rng = np.random.default_rng(seed)
    block = min(block_days, len(days))
    draws = np.array([days[_block_indices(len(days), block, rng)].mean() for _ in range(samples)])
    return np.quantile(draws, [0.025, 0.975]).tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    base = args.plan.parent
    arms = {label: _load_arm((base / path).resolve()) for label, path in plan["arms"].items()}
    reference = next(iter(arms.values()))["keys"]
    for label, arm in arms.items():
        joined = reference.merge(arm["keys"], on=["date", "instrument"], how="outer", indicator=True)
        if (joined._merge != "both").any() or len(joined) != len(reference):
            raise ValueError(f"Arm {label} is not scored on the reference key set")

    result = {"schema_version": 1, "experiment_id": plan["experiment_id"],
              "evaluation_version": EVALUATION_VERSION,
              "hypothesis": plan["hypothesis"], "arms": {}, "pairs": [], "seed_groups": {}}
    for label, arm in arms.items():
        result["arms"][label] = {"directory": arm["directory"], "seed": arm["seed"],
                                 "evaluation_version": arm["metrics"]["evaluation_version"],
                                 "economic_metrics": arm["metrics"]["economic_metrics"],
                                 "metrics": {key: float(arm["metrics"][key]) for key in SUMMARY}}
    first_arm = arms[next(iter(arms))]["directory"]
    result["evaluation_contract"] = archived_cost_scenarios(first_arm)

    for pair in plan["pairs"]:
        left, right = arms[pair["baseline"]], arms[pair["candidate"]]
        joined = left["daily"].join(right["daily"], lsuffix="_baseline", rsuffix="_candidate", how="inner")
        entry = {"label": pair["label"], "question": pair["question"],
                 "baseline": pair["baseline"], "candidate": pair["candidate"],
                 "days": int(len(joined)), "differences": "candidate minus baseline", "metrics": {}}
        for index, metric in enumerate(DAILY):
            delta = (joined[f"{metric}_candidate"] - joined[f"{metric}_baseline"]).to_numpy(float)
            if not np.isfinite(delta).all():
                entry["metrics"][metric] = {"status": "undefined", "mean": None, "interval": None}
                continue
            entry["metrics"][metric] = {
                "status": "estimated" if len(delta) >= 2 else "insufficient_days",
                "mean": float(delta.mean()),
                "interval": _interval(delta, int(plan["block_days"]),
                                      int(plan["bootstrap_samples"]),
                                      int(plan["bootstrap_seed"]) + index),
                "positive_days": int((delta > 0).sum()),
                "negative_days": int((delta < 0).sum()),
            }
        result["pairs"].append(entry)

    for label, group in plan.get("seed_groups", {}).items():
        rows = []
        for seed_text, arm_label in group.items():
            metrics = arms[arm_label]["metrics"]
            rows.append({"seed": int(seed_text),
                         **{key: float(metrics[key]) for key in SUMMARY}})
        frame = pd.DataFrame(rows).sort_values("seed")
        result["seed_groups"][label] = {
            "seeds": [int(value) for value in frame.seed],
            "mean": {key: float(frame[key].mean()) for key in SUMMARY},
            "std": {key: float(frame[key].std(ddof=1)) for key in SUMMARY},
            "range": {key: [float(frame[key].min()), float(frame[key].max())] for key in SUMMARY},
        }

    output = args.plan.with_name(args.plan.stem.replace("_plan", "") + ".json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
