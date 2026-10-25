"""Pick the training-schedule arm from a per-seed run matrix under pre-registered rules.

Plan schema: {schema_version, experiment_id, hypothesis, baseline: {label, runs},
candidates: [{label, runs}], bootstrap_samples, bootstrap_seed, block_days,
guardrail: {turnover_max, min_rank_ic_gain}} where every `runs` maps a seed
string to a run directory holding config.json, predictions.parquet and
targets.parquet.

Per candidate: metrics per seed, mean/std across seeds, same-seed paired daily
differences against the baseline, a 5-day moving-block bootstrap interval on the
seed-averaged daily difference, and explicit pass/fail against the pre-registered
guardrail. Read-only: no training, no artifact mutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.comparison import _block_indices
from jinyiquant.config import load_config
from jinyiquant.evaluation import evaluate

DAILY = ("rank_ic", "long_short_return", "turnover")
SUMMARY = ("rank_ic_mean", "long_short_return_mean", "turnover_mean", "min_daily_long_short_return")


def load_arm(directory: Path) -> dict:
    config = load_config(directory / "config.json")
    predictions = pd.read_parquet(directory / "predictions.parquet")
    targets = pd.read_parquet(directory / "targets.parquet")
    scored = predictions.merge(targets[["date", "instrument"]], on=["date", "instrument"],
                               how="inner", validate="one_to_one")
    metrics, daily, _ = evaluate(scored, targets, config.evaluation, return_sections=True,
                                 undefined_constant_portfolios=True)
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    return {"metrics": {key: float(metrics[key]) for key in SUMMARY},
            "daily": daily.set_index("date")[list(DAILY)]}


def run_group(base: Path, entry: dict) -> dict:
    return {"label": entry["label"],
            "seeds": {str(seed): load_arm((base / path).resolve()) for seed, path in entry["runs"].items()}}


def mean_interval(delta: pd.Series, block: int, draws: int, seed: int) -> list[float] | None:
    values = delta.to_numpy(dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    draws_arr = np.array([values[_block_indices(len(values), min(block, len(values)), rng)].mean()
                          for _ in range(draws)])
    return np.quantile(draws_arr, [0.025, 0.975]).tolist()


def summarise(group: dict) -> dict:
    seeds = sorted(group["seeds"], key=int)
    frame = pd.DataFrame([{"seed": int(seed), **group["seeds"][seed]["metrics"]} for seed in seeds])
    return {"label": group["label"], "seeds": [int(seed) for seed in seeds],
            "per_seed": frame.to_dict(orient="records"),
            "mean": {key: float(frame[key].mean()) for key in SUMMARY},
            "std": {key: float(frame[key].std(ddof=1)) if len(frame) > 1 else None for key in SUMMARY}}


def compare(candidate: dict, baseline: dict, plan: dict) -> dict:
    shared = sorted(set(candidate["seeds"]) & set(baseline["seeds"]), key=int)
    if not shared:
        return {"status": "no_shared_seeds"}
    deltas = {}
    for metric in DAILY:
        per_seed = []
        for seed in shared:
            left = candidate["seeds"][seed]["daily"][metric]
            right = baseline["seeds"][seed]["daily"][metric]
            joined = pd.concat([left.rename("candidate"), right.rename("baseline")], axis=1).dropna()
            per_seed.append(joined["candidate"] - joined["baseline"])
        pooled = pd.concat(per_seed, axis=1).mean(axis=1)
        interval = mean_interval(pooled, int(plan["block_days"]), int(plan["bootstrap_samples"]),
                                 int(plan["bootstrap_seed"]) + DAILY.index(metric))
        deltas[metric] = {"mean": float(pooled.mean()), "interval": interval,
                          "seeds": [int(seed) for seed in shared],
                          "per_seed_mean": {str(seed): float(series.mean()) for seed, series in zip(shared, per_seed)}}
    return deltas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    base = args.plan.resolve().parent
    baseline = run_group(base, plan["baseline"])
    base_summary = summarise(baseline)
    guardrail = plan["guardrail"]
    results = {"schema_version": 1, "experiment_id": plan["experiment_id"], "hypothesis": plan["hypothesis"],
               "baseline": base_summary, "candidates": [],
               "guardrail": guardrail, "decision_rule": plan.get("decision_rule", {})}
    for entry in plan["candidates"]:
        group = run_group(base, entry)
        summary = summarise(group)
        diffs = compare(group, baseline, plan)
        turnover_ok = None
        if summary["mean"]["turnover_mean"] is not None and guardrail.get("turnover_max"):
            turnover_ok = summary["mean"]["turnover_mean"] <= float(guardrail["turnover_max"])
        rank_ic = diffs.get("rank_ic")
        gain_ok = None
        ci_ok = None
        if rank_ic:
            gain_ok = rank_ic["mean"] >= float(guardrail["min_rank_ic_gain"])
            ci_ok = rank_ic["interval"] is not None and rank_ic["interval"][0] > 0
        results["candidates"].append({"label": summary["label"], "seeds": summary["seeds"],
                                      "mean": summary["mean"], "std": summary["std"],
                                      "per_seed": summary["per_seed"], "paired_vs_baseline": diffs,
                                      "guardrail": {"turnover_max": guardrail.get("turnover_max"),
                                                    "turnover_ok": turnover_ok,
                                                    "min_rank_ic_gain": guardrail.get("min_rank_ic_gain"),
                                                    "rank_ic_gain_ok": gain_ok,
                                                    "rank_ic_ci_above_zero": ci_ok,
                                                    "pass": bool(turnover_ok) and bool(gain_ok) and bool(ci_ok)}})
    ranked = sorted(results["candidates"], key=lambda item: item["mean"]["rank_ic_mean"], reverse=True)
    results["ranking_by_mean_rank_ic"] = [item["label"] for item in ranked]
    results["passing"] = [item["label"] for item in results["candidates"] if item["guardrail"]["pass"]]
    output = args.plan.with_name(args.plan.stem.replace("_plan", "") + "_result.json")
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ranking": results["ranking_by_mean_rank_ic"], "passing": results["passing"],
                      "table": [{k: item[k] for k in ("label", "seeds", "mean", "std")} for item in ranked]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
