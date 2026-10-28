"""Paired out-of-sample comparison of the R14 arm against its budget-matched control.

Both models are scored on the recipe's evaluation split (``validation`` — 2023-11-01..2023-11-15
in the 441-day recipe, a window neither model trained on) with the recipe's own evaluation block,
so the difference is a strictly paired, same-split, same-budget reading.

This is *not* the pre-registered eight-window screen from
``configs/screen10_prereg_20261028.json``: the machine that runs the R14 arm holds no 2024 bars
(``202401..202408`` are absent from ``/root/data/cpt_jyc_2026_e2e_bar1m`` on hz-4), so the screen
has to be replayed where the 2024 data and the control replay windows live. The validation window
here overlaps the first half of the screen's ``nov2023`` window, which is stated in the output so
the reading is not mistaken for a screen result.

    python scripts/compare_r14_validation.py --recipe configs/r14-pretrain-441d-csz-seed11.json \
        --run /mnt/work/runs/r14-pretrain-441d-csz-seed11/<stamp> --point r14-eq132 \
        --control-run /mnt/work/runs/scaling-441d-seed11/<stamp> --control-point scaling-1.32 \
        --output .tmp_r14/validation_comparison.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from compare_blend_replay import block_interval  # noqa: E402
from jinyiquant.config import load_config  # noqa: E402
from jinyiquant.data import load_data, select_targets  # noqa: E402
from jinyiquant.evaluation import evaluate  # noqa: E402
from jinyiquant.io import utc_now, write_json  # noqa: E402

SCHEMA = "r14-validation-comparison/1"


def split_record(config) -> dict:
    split = getattr(config.splits, config.evaluation_split)
    return {"name": config.evaluation_split, "start": str(split.start), "end": str(split.end)}


def validation_targets(config) -> pd.DataFrame:
    """Rebuild the evaluation split's targets exactly as replay_model_versions does."""
    split = getattr(config.splits, config.evaluation_split)
    bundle = load_data(config.data, start=split.start, end=split.end)
    targets = select_targets(bundle.labels, {"start": split.start, "end": split.end},
                             config.data.label_horizon_minutes,
                             sample_times=config.data.sample_times)
    if targets.empty:
        raise ValueError("The evaluation split has no eligible targets")
    return targets


def read_predictions(run_dir: str, point: str) -> pd.DataFrame:
    path = Path(run_dir) / "budget_points" / f"{point}_predictions.parquet"
    frame = pd.read_parquet(path)
    for column in ("date", "instrument", "score"):
        if column not in frame:
            raise ValueError(f"{path} is missing the {column!r} column")
    return frame


def scored_frame(predictions: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Inner-join to the scored keys; both arms must score the identical key set."""
    return predictions.merge(targets[["date", "instrument"]], on=["date", "instrument"],
                             how="inner", validate="one_to_one")


def evaluate_arm(predictions: pd.DataFrame, targets: pd.DataFrame, evaluation: dict) -> dict:
    scored = scored_frame(predictions, targets)
    if scored.empty:
        raise ValueError("No scored predictions after exact key alignment")
    metrics, daily, _sections = evaluate(
        scored, targets, evaluation, return_sections=True,
        undefined_constant_portfolios=evaluation.get("constant_portfolios") == "undefined")
    return {"metrics": metrics, "daily": daily, "scored_samples": int(len(scored))}


def paired(candidate: dict, control: dict, *, block: int, draws: int, seed: int) -> dict:
    """Per-day paired differences between two evaluation frames on the same dates."""
    left = candidate["daily"][["date", "rank_ic", "turnover"]].rename(
        columns={"rank_ic": "candidate_rank_ic", "turnover": "candidate_turnover"})
    right = control["daily"][["date", "rank_ic", "turnover"]].rename(
        columns={"rank_ic": "control_rank_ic", "turnover": "control_turnover"})
    joined = left.merge(right, on="date", how="inner", validate="one_to_one")
    rank = joined.dropna(subset=["candidate_rank_ic", "control_rank_ic"])
    turnover = joined.dropna(subset=["candidate_turnover", "control_turnover"])
    delta = rank.candidate_rank_ic.to_numpy() - rank.control_rank_ic.to_numpy()
    turnover_delta = turnover.candidate_turnover.to_numpy() - turnover.control_turnover.to_numpy()
    return {
        "days": int(len(joined)),
        "rank_ic": {
            "candidate_mean": float(rank.candidate_rank_ic.mean()),
            "control_mean": float(rank.control_rank_ic.mean()),
            "delta": float(delta.mean()) if len(delta) else None,
            "positive_days": int((delta > 0).sum()),
            "interval": block_interval(delta, block=block, draws=draws, seed=seed),
        },
        "turnover": {
            "candidate_mean": float(turnover.candidate_turnover.mean()),
            "control_mean": float(turnover.control_turnover.mean()),
            "delta": float(turnover_delta.mean()) if len(turnover_delta) else None,
            "interval": block_interval(turnover_delta, block=block, draws=draws, seed=seed),
        },
    }


def assert_same_keys(candidate: pd.DataFrame, control: pd.DataFrame) -> None:
    left = set(map(tuple, candidate[["date", "instrument"]].to_numpy()))
    right = set(map(tuple, control[["date", "instrument"]].to_numpy()))
    if left != right:
        raise ValueError(f"The two arms scored different key sets: {len(left - right)} only in the "
                         f"candidate, {len(right - left)} only in the control")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--run", required=True, help="candidate run directory")
    parser.add_argument("--point", required=True, help="candidate budget point name")
    parser.add_argument("--control-run", required=True)
    parser.add_argument("--control-point", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--block", type=int, default=5)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261028)
    options = parser.parse_args(argv)

    config = load_config(options.recipe)
    targets = validation_targets(config)
    evaluation = (config.evaluation.model_dump() if hasattr(config.evaluation, "model_dump")
                  else dict(config.evaluation))
    candidate_predictions = read_predictions(options.run, options.point)
    control_predictions = read_predictions(options.control_run, options.control_point)
    assert_same_keys(candidate_predictions, control_predictions)
    candidate = evaluate_arm(candidate_predictions, targets, evaluation)
    control = evaluate_arm(control_predictions, targets, evaluation)
    record = {
        "schema_version": SCHEMA,
        "written_at": utc_now(),
        "recipe": str(options.recipe),
        "split": split_record(config),
        "candidate": {"run": str(options.run), "point": options.point,
                      "scored_samples": candidate["scored_samples"],
                      "metrics": candidate["metrics"]},
        "control": {"run": str(options.control_run), "point": options.control_point,
                    "scored_samples": control["scored_samples"],
                    "metrics": control["metrics"]},
        "paired": paired(candidate, control, block=options.block, draws=options.draws,
                         seed=options.seed),
        "note": ("validation split only: paired same-split, same-budget evidence. This is not the "
                 "pre-registered eight-window screen and it overlaps the screen's nov2023 window, "
                 "so it must not be read as a screen result."),
    }
    write_json(Path(options.output), record)
    print(json.dumps(record, sort_keys=True, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
