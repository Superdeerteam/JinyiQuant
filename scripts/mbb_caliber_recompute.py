#!/usr/bin/env python3
"""Recompute historical paired comparisons under a corrected 5-day moving-block bootstrap.

The house bootstrap (``scripts/compare_blend_replay.py:104`` ``block_interval``) draws ONE
block of five consecutive daily values per replicate and averages that block.  Its interval is
therefore the sampling interval of a *single block mean*, not of the pooled daily mean: it is
wider by roughly ``sqrt(n / 5)``.  Every "not significant" verdict recorded today was produced
with that caliber.

This script keeps that caliber as ``old`` and adds a ``new`` caliber that resamples the whole
series:

    n_blocks = ceil(n / block)
    replicate: draw ``n_blocks`` random start indices, take the block starting at each one
               (wrapping modulo n), concatenate them into a series of length
               ``n_blocks * block`` (approximately n), average it.

20000 draws, seed 20261028.  Point estimates are not touched: they stay the archived
daily-series means, and the script aborts if a recomputed point estimate does not reproduce
its archived value exactly.  No registered threshold and no historical verdict file is
modified; the output is a side-by-side recomputation only.

Usage (on hz-t3, after ``source /root/run_env.sh``):

    python /root/runs/mbb-caliber-20261028/recompute.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root")
REPO = ROOT / "repo_featnorm"
RUNS = ROOT / "runs"
BLOCK = 5
DRAWS = 20000
BOOT_SEED = 20261028

FEATNORM_RUN = RUNS / "featnorm-replay-20261028" / "20261028T071015.646405Z_8dd7831f6bdb"
EXT_A = RUNS / "927d-replay-20261028-batchA" / "20261028T073610.848937Z_f89d015d6ae2"
EXT_B = RUNS / "927d-replay-20261028-batchB" / "20261028T082014.218918Z_881f3c9bc12f"

HERE = Path(__file__).resolve().parent
ARCHIVED = HERE / "archived"
OUT = HERE / "recompute.json"

sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import compare_blend_replay as cbr  # noqa: E402
import featnorm_decision as fd  # noqa: E402
import rung_candidate_compare as rc  # noqa: E402

def _block_interval_new(values) -> dict:
    """Whole-series 5-day moving-block bootstrap (the corrected caliber)."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = vals.size
    if n < 2:
        return {"days": int(n), "n_blocks": None, "resampled_length": None,
                "mean": None, "ci_low": None, "ci_high": None}
    n_blocks = math.ceil(n / BLOCK)
    rng = np.random.default_rng(BOOT_SEED)
    starts = rng.integers(0, n, size=(DRAWS, n_blocks))
    index = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % n
    means = vals[index.reshape(DRAWS, -1)].mean(axis=1)
    return {
        "days": int(n),
        "n_blocks": int(n_blocks),
        "resampled_length": int(n_blocks * BLOCK),
        "mean": float(vals.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "iid_standard_error": float(vals.std(ddof=1) / math.sqrt(n)),
    }


def _calibers(values) -> tuple[dict, dict]:
    vals = np.asarray(values, dtype=float)
    old = cbr.block_interval(vals, block=BLOCK, draws=DRAWS, seed=BOOT_SEED)
    old = {k: old[k] for k in ("days", "mean", "ci_low", "ci_high")}
    return old, new_block_interval(vals)


def _excludes_zero(interval: dict) -> bool | None:
    low, high = interval.get("ci_low"), interval.get("ci_high")
    if low is None or high is None:
        return None
    return bool(low > 0.0 or high < 0.0)


def _row(row_id: str, label: str, series, archived: dict, *, extra: dict | None = None) -> dict:
    vals = np.asarray(pd.Series(series).to_numpy(dtype=float), dtype=float)
    vals = vals[np.isfinite(vals)]
    old, new = _calibers(vals)
    point = float(vals.mean())
    archived_point = archived.get("point")
    gap = None if archived_point is None else abs(point - archived_point)
    old_ex = _excludes_zero(old)
    new_ex = _excludes_zero(new)
    half_old = None if old["ci_low"] is None else (old["ci_high"] - old["ci_low"]) / 2.0
    half_new = None if new["ci_low"] is None else (new["ci_high"] - new["ci_low"]) / 2.0
    row = {
        "id": row_id,
        "label": label,
        "n": int(vals.size),
        "point": point,
        "archived_point": archived_point,
        "point_gap": gap,
        "point_matches_archived_exactly": bool(gap is not None and point == archived_point),
        "old_caliber": {
            "ci": [old["ci_low"], old["ci_high"]],
            "half_width": half_old,
            "excludes_zero": old_ex,
            "archived_ci": archived.get("old_ci"),
            "matches_archived_ci": (
                None if archived.get("old_ci") is None else
                bool(old["ci_low"] == archived["old_ci"][0] and old["ci_high"] == archived["old_ci"][1])
            ),
        },
        "new_caliber": {
            "ci": [new["ci_low"], new["ci_high"]],
            "half_width": half_new,
            "excludes_zero": new_ex,
            "n_blocks": new["n_blocks"],
            "resampled_length": new["resampled_length"],
            "iid_standard_error": new["iid_standard_error"],
        },
        "verdict_flips": bool(old_ex is not None and new_ex is not None and old_ex != new_ex),
    }
    if half_old and half_new:
        row["half_width_ratio_old_over_new"] = half_old / half_new
    if extra:
        row.update(extra)
    return row


def _rung_results() -> dict:
    wanted = ["441d-2.64", "441-6seed", "rung-6", "927d-1.32"]
    out = {}
    for name in wanted:
        print(f"[arm] {name}", flush=True)
        out[name] = rc.run_candidate(name, rc.CANDIDATES[name])
    return out


def _rung_pair(results: dict, left: str, right: str) -> pd.DataFrame:
    left_frame = results[left]["daily"].rename(columns={"rank_ic": "rank_ic_a"})
    right_frame = results[right]["daily"].rename(columns={"rank_ic": "rank_ic_b"})
    paired = left_frame[["date", "window", "rank_ic_a"]].merge(
        right_frame[["date", "rank_ic_b"]], on="date", how="inner", validate="one_to_one")
    paired["delta"] = paired["rank_ic_b"] - paired["rank_ic_a"]
    return paired


def _featnorm_series(point: str) -> tuple[pd.Series, dict[str, pd.Series]]:
    per_window = {window: fd.seed_delta(FEATNORM_RUN, window, point, "rank_ic") for window in fd.WINDOWS}
    return pd.concat(list(per_window.values())).sort_index(), per_window


def _extension_series() -> tuple[pd.Series, dict[str, pd.Series]]:
    per_window = {
        window: fd.seed_delta(
            EXT_A, window, "264", "rank_ic",
            candidate_template="s{seed}-264", control_template="s{seed}-132",
            seeds=fd.SEEDS, extra_dirs=(EXT_B,))
        for window in fd.WINDOWS
    }
    return pd.concat(list(per_window.values())).sort_index(), per_window


def _window_check(per_window: dict[str, pd.Series], archived_windows: dict) -> dict:
    check = {}
    for window, series in per_window.items():
        recomputed = float(series.mean())
        archived = float(archived_windows[window]["mean"])
        check[str(window)] = {"recomputed": recomputed, "archived": archived,
                              "gap": abs(recomputed - archived)}
    return check


def _archived_json(name: str) -> dict:
    return json.loads((ARCHIVED / name).read_text(encoding="utf-8"))


def main() -> int:
    compare4 = _archived_json("compare4.json")
    featnorm = _archived_json("featnorm-decision.json")
    extension = _archived_json("927d-extension-decision.json")
    sealed = _archived_json("sealed_result.json")

    archived_pairs = {(p["left"], p["right"]): p for p in compare4["pairs"]}
    rows: list[dict] = []

    results = _rung_results()
    for pair in (("441d-2.64", "927d-1.32"), ("441d-2.64", "441-6seed"),
                 ("441d-2.64", "rung-6"), ("441-6seed", "rung-6")):
        archived = archived_pairs[pair]
        paired = _rung_pair(results, *pair)
        windows_check = {
            str(window): {
                "recomputed": float(group["delta"].mean()),
                "archived": float(archived["delta_by_window"][str(window)]),
                "gap": abs(float(group["delta"].mean()) - float(archived["delta_by_window"][str(window)])),
            }
            for window, group in paired.groupby("window", sort=False)
        }
        rows.append(_row(
            f"{pair[1]} - {pair[0]}",
            f"{pair[1]} minus {pair[0]} (5 windows, 104 days, ensemble daily RankIC)",
            paired["delta"], {"point": archived["mean_delta"], "old_ci": [archived["ci_low"], archived["ci_high"]]},
            extra={
                "kind": "rung-candidate ensemble pair",
                "positive_days": int((paired["delta"] > 0).sum()),
                "archived_positive_days": archived["positive_days"],
                "windows": windows_check,
                "max_window_gap": max(w["gap"] for w in windows_check.values()),
            },
        ))

    for point, label in (("132", "data-1.32"), ("264", "data-2.64")):
        archived_point = featnorm["points"][label]["pooled"]
        series, per_window = _featnorm_series(point)
        windows_check = _window_check(per_window, featnorm["points"][label]["windows"])
        rows.append(_row(
            f"feature-norm {label} candidate - control",
            f"price-rel-vol-log head vs control, seeds 11/29/47 ({label}, eval 5 windows/104 days)",
            series,
            {"point": archived_point["mean"], "old_ci": [archived_point["ci_low"], archived_point["ci_high"]]},
            extra={
                "kind": "feature-transform decision (daily.csv, per-seed delta then seed mean)",
                "windows": windows_check,
                "max_window_gap": max(w["gap"] for w in windows_check.values()),
            },
        ))

    ext_pooled = extension["points"]["data-1.32"]["pooled"]
    ext_series, ext_per_window = _extension_series()
    ext_check = _window_check(ext_per_window, extension["points"]["data-1.32"]["windows"])
    rows.append(_row(
        "927d 2.64 - 1.32",
        "927-day rung, scale 2.64 minus scale 1.32, same-seed paired (eval 5 windows/104 days)",
        ext_series,
        {"point": ext_pooled["mean"], "old_ci": [ext_pooled["ci_low"], ext_pooled["ci_high"]]},
        extra={
            "kind": "data-scaling extension decision (per-seed delta then seed mean)",
            "windows": ext_check,
            "max_window_gap": max(w["gap"] for w in ext_check.values()),
            "archived_positive_windows": extension["positive_windows"],
        },
    ))

    sealed_daily = pd.DataFrame(sealed["paired_daily"])
    sealed_daily["date"] = pd.to_datetime(sealed_daily["date"])
    sealed_daily = sealed_daily.sort_values("date")
    sealed_archived = sealed["paired_candidate_minus_reference"]
    rows.append(_row(
        "sealed candidate - reference nine",
        "sealed 2024Q4 window (2024-10-08..2025-01-01, 61 days), archived daily series only",
        sealed_daily["delta"],
        {"point": sealed_archived["mean_delta"],
         "old_ci": [sealed_archived["ci_low"], sealed_archived["ci_high"]]},
        extra={
            "kind": "sealed-window side-by-side (no sealed data re-read)",
            "positive_days": int((sealed_daily["delta"] > 0).sum()),
            "archived_positive_days": sealed_archived["positive_days"],
        },
    ))

    flips = [r["id"] for r in rows if r["verdict_flips"]]
    unchanged = [r["id"] for r in rows if not r["verdict_flips"]]
    mismatched = [r["id"] for r in rows if not r["point_matches_archived_exactly"]]

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "caliber": {
            "block_days": BLOCK,
            "draws": DRAWS,
            "bootstrap_seed": BOOT_SEED,
            "old": "scripts/compare_blend_replay.py:104 block_interval — one 5-day block mean per replicate",
            "new": "n_blocks = ceil(n/5) blocks with random starts and wrap-around, concatenated to "
                   "length n_blocks*5, then averaged",
            "point_estimate": "archived daily-series mean, recomputed and compared bit for bit",
        },
        "archived_sources": {
            "rung_pairs": str(ARCHIVED / "compare4.json"),
            "feature_transform": str(ARCHIVED / "featnorm-decision.json"),
            "data_scaling_extension": str(ARCHIVED / "927d-extension-decision.json"),
            "sealed": str(ARCHIVED / "sealed_result.json"),
        },
        "rows": rows,
        "summary": {
            "rows": len(rows),
            "point_estimate_mismatches": mismatched,
            "verdict_flips": flips,
            "verdict_unchanged": unchanged,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    _print_table(rows)
    if mismatched:
        print(f"\nPOINT ESTIMATE MISMATCH: {mismatched} — do not use these rows", flush=True)
        return 2
    return 0


def _fmt(value, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}"


def _print_table(rows: list[dict]) -> None:
    header = (f"{'comparison':44s} {'n':>4s} {'point':>10s} {'old CI':>22s} {'new CI':>22s} "
              f"{'nblk':>4s} {'hw x':>5s} {'old!=0':>6s} {'new!=0':>6s} {'flip':>5s}")
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        old = row["old_caliber"]["ci"]
        new = row["new_caliber"]["ci"]
        old_text = f"[{_fmt(old[0])}, {_fmt(old[1])}]"
        new_text = f"[{_fmt(new[0])}, {_fmt(new[1])}]"
        print(f"{row['id'][:44]:44s} {row['n']:4d} {_fmt(row['point']):>10s} {old_text:>22s} "
              f"{new_text:>22s} {row['new_caliber']['n_blocks']:4d} "
              f"{row.get('half_width_ratio_old_over_new', float('nan')):5.2f} "
              f"{str(row['old_caliber']['excludes_zero']):>6s} "
              f"{str(row['new_caliber']['excludes_zero']):>6s} "
              f"{str(row['verdict_flips']):>5s}")
    print()
    for row in rows:
        print(f"{row['id']}: archived_point_gap={row['point_gap']:.3e} "
              f"old_ci_matches_archived={row['old_caliber']['matches_archived_ci']}"
              + (f" max_window_gap={row['max_window_gap']:.3e}" if "max_window_gap" in row else ""))


if __name__ == "__main__":
    sys.exit(main())
