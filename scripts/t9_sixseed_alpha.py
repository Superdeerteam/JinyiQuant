"""t9: re-match the causal-EWMA alpha for the six-seed 441d@2.64 blend.

Reads the archived per-window predictions of the six 441-day @2.64 checkpoints
(lotA seeds 11/29/47 and lotB seeds 53/59/61), scans alpha so that the smoothed
turnover lands inside the incumbent's +-1% band, and reports the paired daily rank_ic
difference against the registered three-seed candidate at its published alpha.

Nothing is retrained, no frozen object is overwritten, and the sealed interval is never
touched: only the five development windows (nov2023/jan2024/mar2024/apr2024/jun2024).

Usage (on the replay container, from the repository root):

    source /root/run_env.sh
    export MBB_SCRIPTS_DIR=/root/runs/t7-caliber/scripts   # corrected block interval
    python scripts/t9_sixseed_alpha.py --out /root/runs/t9-sixseed-20261028/alpha_scan.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/root/repo_featnorm")
SCRIPTS_DIR = Path(os.environ.get("MBB_SCRIPTS_DIR") or (REPO / "scripts"))
for _p in reversed((str(SCRIPTS_DIR), str(REPO / "scripts"), str(REPO))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# compare_blend_replay must be imported before rung_candidate_compare: the latter prepends
# the repository's own scripts directory to sys.path, which would shadow the corrected copy.
import compare_blend_replay as cbr  # noqa: E402
import rung_candidate_compare as rc  # noqa: E402
from verify_turnover_change import causal_ewma  # noqa: E402

if not hasattr(cbr, "archived_single_block_interval"):
    raise SystemExit(f"stale compare_blend_replay loaded from {cbr.__file__} "
                     "(set MBB_SCRIPTS_DIR to the corrected copy)")

WINDOWS = ["nov2023", "jan2024", "mar2024", "apr2024", "jun2024"]
INCUMBENT_ALPHA = 0.9111677303420447
EXPECTED_INCUMBENT = {"rank_ic": 0.07884498746583304, "turnover": 1.099551972883452}
TURNOVER_CENTER = 1.1019545306708005  # reported incumbent turnover, the matching target
BAND_FRACTION = 0.01
GRID = [round(0.85 + 0.005 * i, 5) for i in range(31)]  # 0.850 .. 1.000 step 0.005
REFINE_STEP = 0.001
BLOCK_DAYS = 5
DRAWS = 20000
BOOT_SEED = 20261028

LOT_A = [
    ("s11", "/root/runs/441-replay-subset/dose-replay-20261027/batch-441d-seed11/"
            "20261027T205912.221663Z_177502e7e8b9", "s11-2.64"),
    ("s29", "/root/runs/441-replay-subset/dose-replay-20261027/batch-441d-seed29/"
            "20261027T230138.712292Z_3317a578512c", "s29-2.64"),
    ("s47", "/root/runs/441-replay-subset/dose-replay-20261027/batch-441d-seed47/"
            "20261028T004911.418444Z_e7eed5e69132", "s47-2.64"),
]
LOT_B = [
    ("s53", "/root/runs/441-lotb-replay-20261028/20261028T080248.891975Z_44bca9f467b1", "s53-264"),
    ("s59", "/root/runs/441-lotb-replay-20261028/20261028T080248.891975Z_44bca9f467b1", "s59-264"),
    ("s61", "/root/runs/441-lotb-replay-20261028/20261028T080248.891975Z_44bca9f467b1", "s61-264"),
]
SIX = [(name, base, version) for name, base, version in LOT_A + LOT_B]
THREE = list(LOT_A)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_pooled(members):
    """Pool the five development windows into one date-sorted (keys, targets, blend)."""
    keys_parts, target_parts, score_parts, per_window = [], [], [], {}
    for window in WINDOWS:
        keys, targets, blend = rc.load_window([(base, version) for _, base, version in members],
                                              window)
        keys_parts.append(keys)
        target_parts.append(targets)
        score_parts.append(blend)
        per_window[window] = int(pd.to_datetime(keys["date"]).dt.normalize().nunique())
    keys = pd.concat(keys_parts, ignore_index=True)
    targets = pd.concat(target_parts, ignore_index=True)
    scores = np.concatenate(score_parts)
    order = np.argsort(keys["date"].to_numpy(), kind="stable")
    keys = keys.iloc[order].reset_index(drop=True)
    targets = targets.iloc[order].reset_index(drop=True)
    return keys, targets, scores[order], per_window


def smoothed_daily(keys, targets, scores, alpha):
    smoothed = causal_ewma(keys, scores, alpha)
    metrics, daily = rc.evaluate_scores(keys, targets, smoothed, 0.0)
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.sort_values("date").reset_index(drop=True)
    return metrics, daily, smoothed


def scan(keys, targets, scores, alphas):
    low = TURNOVER_CENTER * (1 - BAND_FRACTION)
    high = TURNOVER_CENTER * (1 + BAND_FRACTION)
    points = []
    for alpha in alphas:
        sm = causal_ewma(keys, scores, alpha)
        _, daily = rc.evaluate_scores(keys, targets, sm, 0.0)
        turnover = float(daily["turnover"].mean())
        points.append({"alpha": float(alpha), "turnover_mean": turnover,
                       "in_band": bool(low <= turnover <= high)})
        print(f"  alpha={alpha:.5f} turnover={turnover:.6f} in_band={points[-1]['in_band']}",
              flush=True)
    return points


def interval(values, *, archived: bool = False):
    fn = cbr.archived_single_block_interval if archived else cbr.block_interval
    return fn(np.asarray(values, dtype=float), block=BLOCK_DAYS, draws=DRAWS, seed=BOOT_SEED)


def paired_block(series_six: pd.Series, series_three: pd.Series, windows: dict):
    index = series_six.index.intersection(series_three.index)
    delta = (series_six.loc[index] - series_three.loc[index]).sort_index()
    values = delta.to_numpy(dtype=float)
    new = interval(values)
    old = interval(values, archived=True)
    per_window = {}
    for window, days in windows.items():
        chunk = delta.loc[(delta.index >= WINDOW_BOUNDS[window][0])
                          & (delta.index < WINDOW_BOUNDS[window][1])]
        per_window[window] = {"days": int(len(chunk)),
                              "mean": float(chunk.mean()) if len(chunk) else None,
                              "positive": bool(len(chunk) and chunk.mean() > 0)}
    return {"candidate": "six_seed", "control": "three_seed",
            "days": int(len(values)), "mean": float(values.mean()),
            "ci_low": new["ci_low"], "ci_high": new["ci_high"],
            "excludes_zero": bool(new["ci_low"] > 0 or new["ci_high"] < 0),
            "archived_ci_low": old["ci_low"], "archived_ci_high": old["ci_high"],
            "archived_excludes_zero": bool(old["ci_low"] > 0 or old["ci_high"] < 0),
            "block_days": BLOCK_DAYS, "draws": DRAWS, "seed": BOOT_SEED,
            "positive_windows": sum(1 for row in per_window.values() if row["positive"]),
            "windows_total": len(per_window), "per_window": per_window}


WINDOW_BOUNDS = {
    "nov2023": ("2023-11-01", "2023-12-01"),
    "jan2024": ("2024-01-01", "2024-02-01"),
    "mar2024": ("2024-03-01", "2024-04-01"),
    "apr2024": ("2024-04-01", "2024-05-01"),
    "jun2024": ("2024-06-01", "2024-07-01"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--refine", type=float, default=REFINE_STEP)
    args = parser.parse_args()

    print("[load] six-seed pooled windows", flush=True)
    keys, targets, blend_six, window_days = load_pooled(SIX)
    print(f"[load] rows={len(keys)} days={pd.to_datetime(keys['date']).dt.normalize().nunique()}",
          flush=True)
    print("[load] three-seed pooled windows", flush=True)
    keys3, targets3, blend_three, _ = load_pooled(THREE)
    if not keys.equals(keys3):
        raise SystemExit("six-seed and three-seed key sets differ; pairing impossible")
    if not targets.equals(targets3):
        raise SystemExit("six-seed and three-seed targets differ")

    band = (TURNOVER_CENTER * (1 - BAND_FRACTION), TURNOVER_CENTER * (1 + BAND_FRACTION))
    print(f"[scan] target turnover {TURNOVER_CENTER:.10f} band {band}", flush=True)
    grid_points = scan(keys, targets, blend_six, GRID)

    in_band = [p for p in grid_points if p["in_band"]]
    if not in_band:
        raise SystemExit("no alpha on the 0.005 grid lands in the turnover band")
    best = min(in_band, key=lambda p: abs(p["turnover_mean"] - TURNOVER_CENTER))
    lo = max(0.0, best["alpha"] - 0.01)
    hi = min(1.0, best["alpha"] + 0.01)
    refine_grid = [round(lo + args.refine * i, 6)
                   for i in range(int(round((hi - lo) / args.refine)) + 1)]
    print(f"[refine] {refine_grid[0]} .. {refine_grid[-1]} step {args.refine}", flush=True)
    refine_points = scan(keys, targets, blend_six, refine_grid)
    in_band_fine = [p for p in refine_points if p["in_band"]]
    pool = in_band_fine or refine_points
    chosen = min(pool, key=lambda p: abs(p["turnover_mean"] - TURNOVER_CENTER))
    alpha_star = float(chosen["alpha"])

    print(f"[alpha*] {alpha_star} turnover {chosen['turnover_mean']:.6f}", flush=True)
    metrics_six, daily_six, smoothed_six = smoothed_daily(keys, targets, blend_six, alpha_star)
    metrics_three, daily_three, _ = smoothed_daily(keys, targets, blend_three, INCUMBENT_ALPHA)

    incumbent_gap = {
        "rank_ic": float(metrics_three["rank_ic_mean"]) - EXPECTED_INCUMBENT["rank_ic"],
        "turnover": float(metrics_three["turnover_mean"]) - EXPECTED_INCUMBENT["turnover"],
    }
    if abs(incumbent_gap["rank_ic"]) > 5e-6 or abs(incumbent_gap["turnover"]) > 5e-6:
        raise SystemExit(f"incumbent reproduction drifted: {incumbent_gap}")

    six_series = daily_six.set_index("date")["rank_ic"]
    three_series = daily_three.set_index("date")["rank_ic"]
    paired_star = paired_block(six_series, three_series, window_days)

    # Same-alpha contrast: six versus three at the incumbent alpha (t6 published +0.000945).
    metrics_six_inc, daily_six_inc, _ = smoothed_daily(keys, targets, blend_six, INCUMBENT_ALPHA)
    paired_inc = paired_block(daily_six_inc.set_index("date")["rank_ic"], three_series,
                              window_days)

    def arm_summary(metrics, daily):
        worst = daily.loc[daily["rank_ic"].idxmin()]
        return {"rank_ic_mean": float(metrics["rank_ic_mean"]),
                "turnover_mean": float(metrics["turnover_mean"]),
                "long_short_return_mean": float(metrics["long_short_return_mean"]),
                "ic_ir": float(metrics.get("ic_ir", float("nan"))),
                "days": int(len(daily)),
                "worst_day": {"date": str(pd.Timestamp(worst["date"]).date()),
                              "rank_ic": float(worst["rank_ic"])},
                "daily_rank_ic_p05": float(np.quantile(daily["rank_ic"], 0.05)),
                "all_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                for k, v in metrics.items()}}

    payload = {
        "schema_version": 1,
        "experiment_id": "t9-sixseed-alpha-20261028",
        "generated_at_utc": utc_now(),
        "inputs": {
            "members": [{"seed": name, "run_dir": base, "version": version}
                        for name, base, version in SIX],
            "windows": WINDOWS,
            "window_days": window_days,
            "rows": int(len(keys)),
            "days": int(pd.to_datetime(keys["date"]).dt.normalize().nunique()),
            "blend": "equal weight 1/6 on the six 441d@2.64 member scores",
        },
        "caliber": {
            "evaluation_version": "eval-v2",
            "smoothing": {"operator": "causal_ewma", "state_reset": "trading_day"},
            "bootstrap": {"corrected": "scripts/compare_blend_replay.block_interval",
                          "archived": "scripts/compare_blend_replay."
                                      "archived_single_block_interval",
                          "block_days": BLOCK_DAYS, "draws": DRAWS, "seed": BOOT_SEED},
            "incumbent_alpha": INCUMBENT_ALPHA,
        },
        "turnover_band": {"center": TURNOVER_CENTER, "fraction": BAND_FRACTION,
                          "low": band[0], "high": band[1]},
        "alpha_scan": {"grid": GRID, "grid_points": grid_points,
                       "refine_grid": refine_grid, "refine_points": refine_points},
        "selected_alpha": {"alpha": alpha_star,
                           "turnover_mean": float(chosen["turnover_mean"]),
                           "distance_to_center_relative":
                               abs(chosen["turnover_mean"] - TURNOVER_CENTER) / TURNOVER_CENTER,
                           "in_band": True,
                           "rule": "grid 0.85..1.00 step 0.005, then 0.001 refine around the "
                                   "best in-band grid point, then closest to the band center; "
                                   "only turnover is read during the scan"},
        "at_alpha_star": {"six_seed": arm_summary(metrics_six, daily_six)},
        "incumbent_three_seed": {"alpha": INCUMBENT_ALPHA,
                                  "arm": arm_summary(metrics_three, daily_three),
                                  "reproduction_gap": incumbent_gap},
        "paired_six_minus_three": {"at_alpha_star": paired_star,
                                    "at_incumbent_alpha": paired_inc},
        "checks": {
            "incumbent_reproduced": {"expected_rank_ic": EXPECTED_INCUMBENT["rank_ic"],
                                      "expected_turnover": EXPECTED_INCUMBENT["turnover"],
                                      "gap": incumbent_gap, "tolerance": 5e-6},
            "t6_same_alpha_delta": {"published_mean": 0.000945396641743281,
                                     "published_ci": [0.000284, 0.001586],
                                     "recomputed_mean": paired_inc["mean"],
                                     "recomputed_ci": [paired_inc["ci_low"],
                                                       paired_inc["ci_high"]]},
        },
        "notes": [
            "only the five development windows are read; the sealed interval is untouched",
            "the incumbent is reproduced from its archived member predictions at the published "
            "alpha and must match the registered object before any comparison is read",
            "the corrected whole-series block interval is the gate; the retired single-block "
            "interval is printed side by side for continuity only",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    args.out.write_bytes(raw)
    print(json.dumps({"out": str(args.out), "sha256": sha256_bytes(raw), "bytes": len(raw),
                      "alpha_star": alpha_star,
                      "turnover_at_alpha_star": float(chosen["turnover_mean"]),
                      "rank_ic_six": float(metrics_six["rank_ic_mean"]),
                      "rank_ic_three": float(metrics_three["rank_ic_mean"]),
                      "paired_mean": paired_star["mean"],
                      "paired_ci": [paired_star["ci_low"], paired_star["ci_high"]],
                      "paired_archived_ci": [paired_star["archived_ci_low"],
                                             paired_star["archived_ci_high"]],
                      "positive_windows": paired_star["positive_windows"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
