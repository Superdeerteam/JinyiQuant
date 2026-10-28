"""Ad-hoc review check: does the R6/R8 tree's 14:30 edge overlap the R12 tilt's 14:30 edge?

Read-only over existing artifacts (8 screen windows only). For each window:
  * restrict to the 14:30 section (slot 6),
  * z-score per date: trunk (r441 3-seed blend), tree (R6 LightGBM), r12 tilt composite,
  * evaluate 14:30-section rank_ic for base / +tree / +tilt / +both at fixed weight 0.2,
  * Spearman(tree_z, tilt_z) on the 14:30 rows.
Writes runs/r6r8-20261029/review/r12_1430_overlap.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import r6r8_screen as R6  # noqa: E402

R12 = ROOT / "runs/r12-slot1430-tilt-20261028/20261028T1430Z_r12tilt"
W = R6.REPLAY / "windows"
OUT = ROOT / "runs/r6r8-20261029/review"
SLOT_MIN = 14 * 60 + 30
KEYS = ["date", "instrument"]
W_W = 0.2


def _z(frame: pd.DataFrame, col: str) -> np.ndarray:
    g = frame.groupby("date")[col]
    return ((frame[col] - g.transform("mean")) / g.transform("std")).to_numpy(dtype=float)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    out = {"tilt_slot": "14:30", "w": W_W, "windows": {}, "pooled": {}}
    variants = ("base", "tree", "tilt", "both", "both_half")
    daily = {v: [] for v in variants}
    corr_rows = []
    for window in R6.WINDOWS:
        keys = pd.read_parquet(W / window / "keys.parquet")
        targets = pd.read_parquet(W / window / "targets.parquet")
        keys = keys.merge(targets[KEYS], on=KEYS, how="inner")
        keys = keys[keys["date"].dt.hour * 60 + keys["date"].dt.minute == SLOT_MIN]
        keys = keys.sort_values(KEYS).reset_index(drop=True)
        tg = keys.merge(targets, on=KEYS, how="inner")
        # trunk raw blend
        acc = np.zeros(len(keys))
        for v in R6.BLENDS["r441"]:
            p = pd.read_parquet(W / window / v / "predictions.parquet")
            acc += keys.merge(p, on=KEYS, how="left")["score"].to_numpy()
        tree = pd.read_parquet(R6.RUN / "screen" / "predictions" / f"{window}.parquet")
        tilt = pd.read_parquet(R12 / "windows" / window / "r12_tilt" / "predictions.parquet")
        frame = keys.assign(trunk=acc / 3.0)
        frame = frame.merge(tree, on=KEYS, how="left")
        frame = frame.merge(tilt[KEYS + ["score"]].rename(columns={"score": "tilt"}),
                            on=KEYS, how="left")
        frame["trunk_z"] = _z(frame, "trunk")
        frame["tree_z"] = _z(frame, "score")
        frame["tilt_z"] = _z(frame, "tilt")
        corr = _spearman(frame["tree_z"], frame["tilt_z"])
        scores = {
            "base": frame["trunk_z"],
            "tree": (1 - W_W) * frame["trunk_z"] + W_W * frame["tree_z"],
            "tilt": (1 - W_W) * frame["trunk_z"] + W_W * frame["tilt_z"],
            "both": (1 - 2 * W_W) * frame["trunk_z"] + W_W * frame["tree_z"] + W_W * frame["tilt_z"],
            "both_half": (1 - W_W) * frame["trunk_z"] + 0.5 * W_W * frame["tree_z"] + 0.5 * W_W * frame["tilt_z"],
        }
        stats = {}
        for name in variants:
            d = R6._daily(keys, tg, scores[name].to_numpy(), name)
            daily[name].append(d.assign(window=window))
            stats[name] = float(d["rank_ic"].mean())
        out["windows"][window] = {
            "rank_ic": stats,
            "tree_gain": stats["tree"] - stats["base"],
            "tilt_gain": stats["tilt"] - stats["base"],
            "both_gain": stats["both"] - stats["base"],
            "overlap_loss": (stats["tree"] - stats["base"]) + (stats["tilt"] - stats["base"])
                            - (stats["both"] - stats["base"]),
            "spearman_tree_tilt": corr,
        }
        corr_rows.append(corr)
        print(window, {k: round(v, 5) for k, v in stats.items()}, "corr %.3f" % corr, flush=True)
    for name in variants:
        d = pd.concat(daily[name], ignore_index=True)
        out["pooled"][name] = R6._pooled(d, R6.WINDOWS)
    gains = {w: out["windows"][w] for w in out["windows"]}
    out["pooled"]["tree_gain"] = out["pooled"]["tree"]["rank_ic"] - out["pooled"]["base"]["rank_ic"]
    out["pooled"]["tilt_gain"] = out["pooled"]["tilt"]["rank_ic"] - out["pooled"]["base"]["rank_ic"]
    out["pooled"]["both_gain"] = out["pooled"]["both"]["rank_ic"] - out["pooled"]["base"]["rank_ic"]
    out["spearman_mean"] = float(np.mean(corr_rows))
    out["windows_tree_gain_positive"] = sum(g["tree_gain"] > 0 for g in gains.values())
    out["windows_tilt_gain_positive"] = sum(g["tilt_gain"] > 0 for g in gains.values())
    out["verdict"] = ("overlap" if out["spearman_mean"] > 0.3 or
                      out["pooled"]["both_gain"] < 0.6 * (out["pooled"]["tree_gain"] + out["pooled"]["tilt_gain"])
                      else "complementary")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r12_1430_overlap.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                      for k, v in out["pooled"].items() if not isinstance(v, dict)}, ensure_ascii=False))
    print("spearman_mean %.3f verdict %s" % (out["spearman_mean"], out["verdict"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
