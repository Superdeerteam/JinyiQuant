"""Score frozen training arms on the shared validation window and compare them daily.

LEGACY CALIBER (stage A): this script's pre-registered guardrail used to decide
promotion. That promotion model is retired. The script now only reports
descriptive paired differences: every guardrail row carries
``usable_for_promotion=false`` and the written record sets
``promotion_allowed=false``. ``promotion_decision`` raises
``PromotionDisabledError`` for any caller that tries to turn a comparison into an
accept/reject decision; no legacy conclusion may authorise promotion.

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

from jinyiquant.comparison import _block_indices, declared_cost_scenarios
from jinyiquant.config import load_config
from jinyiquant.evaluation import EVALUATION_VERSION, evaluate

SUMMARY = ("rank_ic_mean", "turnover_mean", "long_short_return_mean", "min_daily_long_short_return",
           "long_short_sharpe", "rank_ic_ir")
DAILY = ("rank_ic", "long_short_return", "turnover")
LEGACY_CALIBER = {
    "caliber_id": "legacy-pre-eval-v2",
    "status": "disabled",
    "promotion_allowed": False,
    "legacy_mark": "旧口径结论——不得用于晋升",
}


class PromotionDisabledError(RuntimeError):
    """Raised when a caller tries to promote/eliminate from the legacy guardrail."""


def legacy_guardrail(left: dict, deltas: dict, guardrail: dict) -> dict:
    """Compute the retired guardrail fields, explicitly marked as not-for-promotion."""
    pass_gain = deltas["rank_ic"]["mean"] >= float(guardrail["min_rank_ic_gain"])
    pass_ci = deltas["rank_ic"]["interval"][0] > 0
    pass_turnover = left["metrics"]["turnover_mean"] <= float(guardrail["turnover_max"])
    return {
        "min_rank_ic_gain": guardrail["min_rank_ic_gain"],
        "turnover_max": guardrail["turnover_max"],
        "rank_ic_gain_ok": pass_gain,
        "rank_ic_ci_above_zero": pass_ci,
        "turnover_ok": pass_turnover,
        "pass": pass_gain and pass_ci and pass_turnover,
        "legacy": True,
        "usable_for_promotion": False,
        "usable_for_elimination": False,
        "legacy_status": LEGACY_CALIBER["legacy_mark"],
    }


def promotion_decision(*_args, **_kwargs):
    """The retired accept/reject path: refuse to produce a promotion decision."""
    raise PromotionDisabledError(
        "grouped-arm guardrail promotion is retired (旧口径结论——不得用于晋升); "
        "use eval-v2 once the contract lands")


def scores(path: Path) -> pd.Series:
    frame = pd.read_parquet(path / "predictions.parquet")
    return frame.set_index(["date", "instrument"])["score"]


def load_arm(paths: list[Path]):
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
    values = {key: float(metrics[key]) for key in SUMMARY} | spread
    values.update(evaluation_version=metrics["evaluation_version"],
                  economic_metrics=metrics["economic_metrics"])
    answer = targets.merge(scored[["date", "instrument"]], on=["date", "instrument"],
                           how="inner", validate="one_to_one")
    return values, daily.set_index("date"), scored, answer, config.evaluation


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
        metrics, daily, scored, answer, evaluation = load_arm(paths)
        arms[name] = {"role": entry["role"], "seeds": [path.parent.name for path in paths],
                      "metrics": metrics, "daily": daily, "_scored": scored,
                      "_targets": answer, "_evaluation": evaluation}
        print(name, {key: round(metrics[key], 6) for key in ("rank_ic_mean", "turnover_mean",
                                                            "long_short_return_mean", "score_std")})
    pairs = []
    guardrail = plan["guardrail"]
    for left_name, right_name in plan["pairs"]:
        left, right = arms[left_name], arms[right_name]
        deltas = compare(left, right, plan)
        pairs.append({"left": left_name, "right": right_name, "deltas": deltas,
                      "guardrail": legacy_guardrail(left, deltas, guardrail)})
        print("pair", left_name, "-", right_name, deltas["rank_ic"])
    first_left = arms[plan["pairs"][0][0]]
    evaluation_contract = declared_cost_scenarios(
        first_left.pop("_scored"), first_left.pop("_targets"), first_left.pop("_evaluation"),
        undefined_constant_portfolios=True)
    document = {"schema_version": 1, "experiment_id": plan["experiment_id"],
                "evaluation_version": EVALUATION_VERSION,
                "evaluation_contract": evaluation_contract,
                "hypothesis": plan["hypothesis"],
                "design": {"block_days": plan["block_days"], "bootstrap_samples": plan["bootstrap_samples"],
                           "bootstrap_seed": plan["bootstrap_seed"], "guardrail": guardrail},
                "legacy_caliber": LEGACY_CALIBER, "promotion_allowed": False,
                "arms": {name: {key: value for key, value in arm.items()
                                if key != "daily" and not key.startswith("_")}
                         for name, arm in arms.items()},
                "pairs": pairs,
                "decision": {"text": plan["decision"], "status": "disabled_legacy_caliber",
                             "usable_for_promotion": False,
                             "legacy_status": LEGACY_CALIBER["legacy_mark"]}}
    output = args.output or args.plan.with_name(args.plan.stem.replace("_plan", "") + "_record.json")
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
