"""Compare the two smoothed submission candidates on the same 104 delivered trading days.

Reads only archived eval-v2 replay predictions -- no training, no new inference, no sealed
data. Caliber is the project one: eval-v2 (quantile 0.2, min_cross_section 3, undefined
constant portfolios), the five delivery windows, daily rank_ic pooled across the 104 trading
days, and a 5-day moving-block bootstrap on the paired daily difference.

The 441d@2.64 candidate is a reproduction check: its unsmoothed blend must land on the
published 0.074242 and its smoothed version on the published +0.0082356 improvement, or the
caliber here is wrong and nothing else in the output may be used.

Usage
    /root/py312 scripts/rung_candidate_compare.py --out /root/runs/rung-candidate-compare-20261028/compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from jinyiquant.config import EvaluationConfig  # noqa: E402
from jinyiquant.evaluation import evaluate  # noqa: E402
from verify_turnover_change import causal_ewma  # noqa: E402
from compare_blend_replay import block_interval  # noqa: E402

WINDOWS = ["nov2023", "jan2024", "mar2024", "apr2024", "jun2024"]
PAIRS = [
    ("441d-2.64", "927d-1.32"),
    ("441d-2.64", "441-6seed"),
    ("441d-2.64", "rung-6"),
    ("441-6seed", "rung-6"),
]
ALPHA = 0.9111677303420447
BASELINE_RANK_IC = 0.070687271
ARCHIVED_UNSMOOTHED = ("/root/runs/archived-441/daily_0bp_blend.csv")
ARCHIVED_SMOOTHED = ("/root/runs/archived-441/daily_0bp_ewma_match.csv")
INCUMBENT_TURNOVER = 1.1019545306708005

CANDIDATES = {
    "441d-2.64": {
        "members": [
            ("/root/runs/441-replay-subset/dose-replay-20261027/batch-441d-seed11/"
             "20261027T205912.221663Z_177502e7e8b9", "s11-2.64"),
            ("/root/runs/441-replay-subset/dose-replay-20261027/batch-441d-seed29/"
             "20261027T230138.712292Z_3317a578512c", "s29-2.64"),
            ("/root/runs/441-replay-subset/dose-replay-20261027/batch-441d-seed47/"
             "20261028T004911.418444Z_e7eed5e69132", "s47-2.64"),
        ],
    },
    "441d-2.64-lotB": {
        "members": [
            ("/root/runs/441-lotb-replay-20261028/20261028T080248.891975Z_44bca9f467b1", "s53-264"),
            ("/root/runs/441-lotb-replay-20261028/20261028T080248.891975Z_44bca9f467b1", "s59-264"),
            ("/root/runs/441-lotb-replay-20261028/20261028T080248.891975Z_44bca9f467b1", "s61-264"),
        ],
    },
    "441-6seed": {
        "members": [
            ("/root/runs/441-replay-subset/dose-replay-20261027/batch-441d-seed11/"
             "20261027T205912.221663Z_177502e7e8b9", "s11-2.64"),
            ("/root/runs/441-replay-subset/dose-replay-20261027/batch-441d-seed29/"
             "20261027T230138.712292Z_3317a578512c", "s29-2.64"),
            ("/root/runs/441-replay-subset/dose-replay-20261027/batch-441d-seed47/"
             "20261028T004911.418444Z_e7eed5e69132", "s47-2.64"),
            ("/root/runs/441-lotb-replay-20261028/20261028T080248.891975Z_44bca9f467b1", "s53-264"),
            ("/root/runs/441-lotb-replay-20261028/20261028T080248.891975Z_44bca9f467b1", "s59-264"),
            ("/root/runs/441-lotb-replay-20261028/20261028T080248.891975Z_44bca9f467b1", "s61-264"),
        ],
    },
    "rung-6": {
        "members": [
            ("/root/runs/441-replay-subset/dose-replay-20261027/batch-441d-seed11/"
             "20261027T205912.221663Z_177502e7e8b9", "s11-2.64"),
            ("/root/runs/441-replay-subset/dose-replay-20261027/batch-441d-seed29/"
             "20261027T230138.712292Z_3317a578512c", "s29-2.64"),
            ("/root/runs/441-replay-subset/dose-replay-20261027/batch-441d-seed47/"
             "20261028T004911.418444Z_e7eed5e69132", "s47-2.64"),
            ("/root/runs/927d-replay-20261028-batchA/"
             "20261028T073610.848937Z_f89d015d6ae2", "s11-132"),
            ("/root/runs/927d-replay-20261028-batchA/"
             "20261028T073610.848937Z_f89d015d6ae2", "s29-132"),
            ("/root/runs/927d-replay-20261028-batchA/"
             "20261028T073610.848937Z_f89d015d6ae2", "s47-132"),
        ],
    },
    "927d-1.32": {
        "members": [
            ("/root/runs/927d-replay-20261028-batchA/"
             "20261028T073610.848937Z_f89d015d6ae2", "s11-132"),
            ("/root/runs/927d-replay-20261028-batchA/"
             "20261028T073610.848937Z_f89d015d6ae2", "s29-132"),
            ("/root/runs/927d-replay-20261028-batchA/"
             "20261028T073610.848937Z_f89d015d6ae2", "s47-132"),
        ],
    },
}


def evaluation_config(rate: float | None) -> EvaluationConfig:
    return EvaluationConfig(quantile=0.2, periods_per_year=252, min_cross_section=3,
                            constant_portfolios="undefined", single_side_rate=rate)


def evaluate_scores(keys: pd.DataFrame, targets: pd.DataFrame, scores: np.ndarray,
                    rate: float | None) -> tuple[dict, pd.DataFrame]:
    predictions = keys.assign(score=scores)[["date", "instrument", "score"]]
    metrics, daily, _ = evaluate(predictions, targets, evaluation_config(rate),
                                 return_sections=True, undefined_constant_portfolios=True)
    return metrics, daily


def load_window(members: list[tuple[str, str]], window: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    first = Path(members[0][0]) / "windows" / window
    targets = pd.read_parquet(first / "targets.parquet")
    targets = targets.sort_values(["date", "instrument"]).reset_index(drop=True)
    keys = targets[["date", "instrument"]]
    if keys.duplicated().any():
        raise SystemExit(f"{window}: duplicate target keys")
    scores = []
    for base, version in members:
        predictions = pd.read_parquet(Path(base) / "windows" / window / version / "predictions.parquet")
        merged = predictions.merge(keys, on=["date", "instrument"], how="inner",
                                   validate="one_to_one")
        merged = merged.sort_values(["date", "instrument"]).reset_index(drop=True)
        if len(merged) != len(keys) or not merged[["date", "instrument"]].equals(keys):
            raise SystemExit(f"{window}/{version}: prediction coverage or key order mismatch")
        scores.append(merged["score"].to_numpy(dtype=float))
    blend = np.mean(np.stack(scores), axis=0)
    return keys, targets[["date", "instrument", "target"]], blend


def run_candidate(name: str, spec: dict) -> dict:
    frames = {"unsmoothed": [], "smoothed": [], "smoothed_5bp": []}
    per_window = {}
    for window in WINDOWS:
        keys, targets, blend = load_window(spec["members"], window)
        _, daily_raw = evaluate_scores(keys, targets, blend, 0.0)
        smoothed = causal_ewma(keys, blend, ALPHA)
        _, daily_sm = evaluate_scores(keys, targets, smoothed, 0.0)
        _, daily_sm5 = evaluate_scores(keys, targets, smoothed, 0.0005)
        for label, frame in (("unsmoothed", daily_raw), ("smoothed", daily_sm),
                             ("smoothed_5bp", daily_sm5)):
            frame = frame.copy()
            frame["window"] = window
            frames[label].append(frame)
        per_window[window] = {
            "days": int(len(daily_sm)),
            "unsmoothed_rank_ic": float(daily_raw["rank_ic"].mean()),
            "smoothed_rank_ic": float(daily_sm["rank_ic"].mean()),
            "smoothed_turnover": float(daily_sm["turnover"].mean()),
        }
        print(f"  {name} {window}: raw {per_window[window]['unsmoothed_rank_ic']:.5f} -> "
              f"smoothed {per_window[window]['smoothed_rank_ic']:.5f}", flush=True)
    pooled = {label: pd.concat(chunks, ignore_index=True) for label, chunks in frames.items()}
    smoothed = pooled["smoothed"]
    turnover = float(smoothed["turnover"].mean())
    rank_ic = float(smoothed["rank_ic"].mean())
    return {
        "candidate": name,
        "alpha": ALPHA,
        "days": int(len(smoothed)),
        "rank_ic_unsmoothed": float(pooled["unsmoothed"]["rank_ic"].mean()),
        "rank_ic_smoothed": rank_ic,
        "rank_ic_smoothed_5bp": float(pooled["smoothed_5bp"]["rank_ic"].mean()),
        "turnover_smoothed": turnover,
        "turnover_degradation": turnover - INCUMBENT_TURNOVER,
        "improvement_vs_baseline": rank_ic - BASELINE_RANK_IC,
        "long_short_mean": float(smoothed["long_short_return"].mean()),
        "per_window": per_window,
        "daily": smoothed[["date", "window", "rank_ic", "turnover", "long_short_return"]],
        "daily_smoothed": smoothed[["date", "rank_ic"]],
        "daily_unsmoothed": pooled["unsmoothed"][["date", "rank_ic"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    results = {}
    for name, spec in CANDIDATES.items():
        print(f"[{name}]", flush=True)
        results[name] = run_candidate(name, spec)

    a, b = "441d-2.64", "927d-1.32"
    pairs_out = []
    for left_name, right_name in PAIRS:
        if left_name not in results or right_name not in results:
            continue
        left = results[left_name]["daily"].rename(columns={"rank_ic": "rank_ic_a"})
        right = results[right_name]["daily"].rename(columns={"rank_ic": "rank_ic_b"})
        paired = left[["date", "window", "rank_ic_a"]].merge(
            right[["date", "rank_ic_b"]], on="date", how="inner", validate="one_to_one")
        paired["delta"] = paired["rank_ic_b"] - paired["rank_ic_a"]
        delta = paired["delta"].to_numpy(dtype=float)
        interval = block_interval(delta, block=5, draws=20000)
        pairs_out.append({
            "left": left_name, "right": right_name, "days": int(len(delta)),
            "mean_delta": float(delta.mean()),
            "ci_low": interval["ci_low"], "ci_high": interval["ci_high"],
            "positive_days": int((delta > 0).sum()),
            "delta_by_window": {str(window): float(group["delta"].mean())
                                for window, group in paired.groupby("window", sort=False)},
        })

    archived_unsmoothed = Path(ARCHIVED_UNSMOOTHED)
    archived_smoothed = Path(ARCHIVED_SMOOTHED)
    reproduction = {}
    for label, path in (("unsmoothed", archived_unsmoothed), ("smoothed", archived_smoothed)):
        if not path.is_file():
            reproduction[label] = {"archived": str(path), "present": False}
            continue
        archived = pd.read_csv(path, usecols=["date", "rank_ic"])
        fresh = results[a]["daily_%s" % label][["date", "rank_ic"]]
        merged = archived.merge(fresh, on="date", suffixes=("_archived", "_fresh"),
                                validate="one_to_one")
        gap = (merged["rank_ic_fresh"] - merged["rank_ic_archived"]).abs()
        reproduction[label] = {
            "archived": str(path), "present": True, "days": int(len(merged)),
            "archived_mean": float(merged["rank_ic_archived"].mean()),
            "fresh_mean": float(merged["rank_ic_fresh"].mean()),
            "max_abs_difference": float(gap.max()),
            "ok": bool(len(merged) == 104 and gap.max() < 1e-9),
        }

    summary = {
        "alpha": ALPHA,
        "baseline_rank_ic": BASELINE_RANK_IC,
        "incumbent_turnover": INCUMBENT_TURNOVER,
        "candidates": {name: {k: v for k, v in result.items()
                              if k not in ("daily", "daily_smoothed", "daily_unsmoothed")}
                       for name, result in results.items()},
        "pairs": pairs_out,
        "reproduction_check": reproduction,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "candidates"},
                     ensure_ascii=False, indent=1))
    for name in CANDIDATES:
        row = summary["candidates"][name]
        print(f"{name}: rank_ic {row['rank_ic_smoothed']:.6f} turn {row['turnover_smoothed']:.4f} "
              f"improvement {row['improvement_vs_baseline']:+.6f} ls {row['long_short_mean']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
