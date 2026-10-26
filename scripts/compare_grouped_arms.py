"""Score frozen training arms on the shared validation window and compare them daily.

An arm is either a single run or an equal-weight seed ensemble (scores averaged per
date/instrument before evaluation, matching the V3 decision's ensemble convention).
Daily paired differences use the same 5-day moving-block bootstrap as the selection
sweeps, so a block interval is not a day-level independence claim.
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

SUMMARY = ("rank_ic_mean", "turnover_mean", "long_short_return_mean", "min_daily_long_short_return",
           "long_short_sharpe", "rank_ic_ir")
DAILY = ("rank_ic", "long_short_return", "turnover")


def scores(path: Path) -> pd.Series:
    frame = pd.read_parquet(path / "predictions.parquet")
    return frame.set_index(["date", "instrument"])["score"]


def load_arm(paths: list[Path]) -> tuple[dict, pd.DataFrame]:
    joined = pd.concat([scores(path) for path in paths], axis=1).mean(axis=1)
    config = load_config(paths[0] / "config.json")
    targets = pd.read_parquet(paths[0] / "targets.parquet")
    scored = joined.rename("score").reset_index().merge(
        targets[["date", "instrument"]], on=["date", "instrument"], how="inner", validate="one_to_one")
    metrics, daily, _ = evaluate(scored, targets, config.evaluation, return_sections=True,
                                 undefined_constant_portfolios=True)
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    spread = {"score_std": float(joined.std()), "score_abs_mean": float(joined.abs().mean())}
    return {key: float(metrics[key]) for key in SUMMARY} | spread, daily.set_index("date")


def mean_interval(values: pd.Series, block: int, draws: int, seed: int) -> list[float]:
    array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([array[_block_indices(len(array), min(block, len(array)), rng)].mean()
                     for _ in range(draws)])
    return np.quantile(boot, [0.025, 0.975]).round(8).tolist()


def compare(left: dict, right: dict, plan: dict) -> dict:
    result = {}
    for position, metric in enumerate(DAILY):
        a, b = left["daily"][metric], right["daily"][metric]
        joint = pd.concat([a.rename("left"), b.rename("right")], axis=1).dropna()
        delta = joint["left"] - joint["right"]
        result[metric] = {"mean": round(float(delta.mean()), 8),
                          "interval": mean_interval(delta, int(plan["block_days"]),
                                                    int(plan["bootstrap_samples"]),
                                                    int(plan["bootstrap_seed"]) + position)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    root = args.plan.resolve().parent.parent
    arms = {}
    for name, entry in plan["arms"].items():
        paths = [root / path for path in entry["runs"]]
        for path in paths:
            if not (path / "metrics.json").exists():
                raise SystemExit(f"{name}: missing run {path}")
        metrics, daily = load_arm(paths)
        arms[name] = {"role": entry["role"], "seeds": [path.parent.name for path in paths],
                      "metrics": metrics, "daily": daily}
        print(name, {key: round(metrics[key], 6) for key in ("rank_ic_mean", "turnover_mean",
                                                            "long_short_return_mean", "score_std")})
    pairs = []
    guardrail = plan["guardrail"]
    for left_name, right_name in plan["pairs"]:
        left, right = arms[left_name], arms[right_name]
        deltas = compare(left, right, plan)
        pass_gain = deltas["rank_ic"]["mean"] >= float(guardrail["min_rank_ic_gain"])
        pass_ci = deltas["rank_ic"]["interval"][0] > 0
        pass_turnover = left["metrics"]["turnover_mean"] <= float(guardrail["turnover_max"])
        pairs.append({"left": left_name, "right": right_name, "deltas": deltas,
                      "guardrail": {"min_rank_ic_gain": guardrail["min_rank_ic_gain"],
                                    "turnover_max": guardrail["turnover_max"],
                                    "rank_ic_gain_ok": pass_gain, "rank_ic_ci_above_zero": pass_ci,
                                    "turnover_ok": pass_turnover,
                                    "pass": pass_gain and pass_ci and pass_turnover}})
        print("pair", left_name, "-", right_name, deltas["rank_ic"])
    document = {"schema_version": 1, "experiment_id": plan["experiment_id"], "hypothesis": plan["hypothesis"],
                "design": {"block_days": plan["block_days"], "bootstrap_samples": plan["bootstrap_samples"],
                           "bootstrap_seed": plan["bootstrap_seed"], "guardrail": guardrail},
                "arms": {name: {key: value for key, value in arm.items() if key != "daily"}
                         for name, arm in arms.items()},
                "pairs": pairs, "decision": plan["decision"]}
    output = args.output or args.plan.with_name(args.plan.stem.replace("_plan", "") + "_record.json")
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
