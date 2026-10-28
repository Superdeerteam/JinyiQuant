"""Paired candidate-vs-control evaluation for the market-context A/B/C replay.

Reads archived eval-v2 replay predictions only: no training, no new inference, no sealed
data. Caliber is the project one -- eval-v2 (quantile 0.2, min_cross_section 3, undefined
constant portfolios, single_side_rate 0.0 so turnover stays the eval-v2 ``U / 2``), the
project causal EWMA (alpha=0.9111677303420447) applied to each arm's cross-sectionally
z-scored scores, daily rank_ic pooled across trading days, and paired candidate-minus-control
daily differences with the 5-day moving-block bootstrap of
``scripts.compare_blend_replay.block_interval`` (ceil(n / 5) random-start wrap-around blocks,
20000 draws).  The retired single-block caliber is printed side by side and recorded as
``archived_ci_*`` for continuity only; every decision uses the corrected interval.

Pairing is per (seed, trading day): each arm is a replay version with a seed, so the
difference is only taken within one seed and one trading day, then pooled across seeds and
windows. Arms are also compared through their equal-weight blend of seeds, which is what the
published 441d@2.64 reference caliber is.

Entry points
    * ``--run-dir``: a replay directory produced by
      ``scripts/replay_model_versions.py run configs/mktctx_replay_20261028_plan.json``.
    * ``--sources``: an explicit ``{"arms": {arm: {seed: {"run_dir", "version"}}}}`` map so
      archived runs (the three 441d@2.64 seed runs of the A arm) can be re-checked.
    * ``--self-test``: exercises the same code path on synthetic data with a known paired
      difference, an identical-arms control, and AR(1) paired differences whose block
      interval must widen relative to the iid interval and match the block-inflation theory.

Usage
    python scripts/mktctx_compare.py --plan configs/mktctx_replay_20261028_plan.json \\
        --run-dir runs/mktctx-20261028/<run-id> --out runs/mktctx-20261028/comparison.json
    python scripts/mktctx_compare.py --plan configs/mktctx_replay_20261028_plan.json \\
        --sources runs/mktctx-20261028/reference_a5_sources.json --check-arm A \\
        --check-windows nov2023,jan2024,mar2024,apr2024,jun2024 \\
        --expect-rank-ic 0.078845 --expect-turnover 1.0996 --tolerance 0.0005 \\
        --out runs/mktctx-20261028/reference_a5.json
    python scripts/mktctx_compare.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO / "scripts"):
    sys.path.insert(0, str(_path))

from jinyiquant.config import EvaluationConfig  # noqa: E402
from jinyiquant.evaluation import EVALUATION_VERSION, evaluate  # noqa: E402
from compare_blend_replay import (  # noqa: E402
    archived_single_block_interval, block_interval)
from verify_turnover_change import causal_ewma  # noqa: E402

ALPHA = 0.9111677303420447
BLOCK_DAYS = 5
DEFAULT_DRAWS = 20000
TURNOVER_CONVERSION = 0.5
KEYS = ["date", "instrument"]
PAIR_ORDER = (("C", "B"), ("C", "A"), ("B", "A"))
BLEND = "blend"


# --------------------------------------------------------------------------------------- #
# evaluation caliber
# --------------------------------------------------------------------------------------- #
def evaluation_config(rate: float | None = 0.0) -> EvaluationConfig:
    return EvaluationConfig(quantile=0.2, periods_per_year=252, min_cross_section=3,
                            constant_portfolios="undefined", single_side_rate=rate)


def daily_scores(keys: pd.DataFrame, targets: pd.DataFrame, scores,
                 *, smooth: bool = True) -> pd.DataFrame:
    """One eval-v2 daily row per trading day for a score vector aligned to ``keys``."""
    values = np.asarray(scores, dtype=float)
    if len(values) != len(keys):
        raise SystemExit(f"score length {len(values)} != key length {len(keys)}")
    if smooth:
        values = causal_ewma(keys, values, ALPHA)
    predictions = keys[KEYS].assign(score=values)
    scored = predictions.merge(targets[KEYS], on=KEYS, how="inner", validate="one_to_one")
    if scored.empty:
        raise SystemExit("no scored rows after aligning predictions to targets")
    _, daily, _ = evaluate(scored, targets, evaluation_config(0.0), return_sections=True,
                           undefined_constant_portfolios=True)
    return daily[["date", "rank_ic", "turnover", "long_short_return"]].sort_values(
        "date").reset_index(drop=True)


def load_window(run_dir: str | Path, window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(run_dir) / "windows" / window
    if not base.is_dir():
        raise SystemExit(f"missing window directory: {base}")
    targets = pd.read_parquet(base / "targets.parquet").sort_values(KEYS).reset_index(drop=True)
    keys_path = base / "keys.parquet"
    if keys_path.is_file():
        keys = pd.read_parquet(keys_path)
    else:
        keys = targets[KEYS]
    keys = keys.sort_values(KEYS).reset_index(drop=True)[KEYS]
    if keys.duplicated().any():
        raise SystemExit(f"{base}: duplicate window keys")
    return keys, targets[KEYS + ["target"]]


def load_scores(run_dir: str | Path, window: str, version: str,
                keys: pd.DataFrame) -> np.ndarray:
    path = Path(run_dir) / "windows" / window / version / "predictions.parquet"
    if not path.is_file():
        raise SystemExit(f"missing predictions: {path}")
    predictions = pd.read_parquet(path)[KEYS + ["score"]]
    merged = predictions.merge(keys, on=KEYS, how="inner",
                               validate="one_to_one").sort_values(KEYS).reset_index(drop=True)
    if len(merged) != len(keys) or not merged[KEYS].equals(keys):
        raise SystemExit(f"{path}: prediction coverage or key order mismatch")
    return merged["score"].to_numpy(dtype=float)


# --------------------------------------------------------------------------------------- #
# interval + pairing
# --------------------------------------------------------------------------------------- #
def _number(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def interval_summary(values, *, block: int, draws: int, seed: int) -> dict:
    """5-day moving-block bootstrap through ``compare_blend_replay.block_interval``.

    The corrected interval leads; the retired single-block interval is kept as ``archived_ci_*``
    so a reader can see why historical readings were conservative.
    """
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    interval = block_interval(data, block=block, draws=draws, seed=seed)
    retired = archived_single_block_interval(data, block=block, draws=draws, seed=seed)
    return {"days": int(interval["days"]) if interval["days"] is not None else 0,
            "mean": _number(interval["mean"]),
            "ci_low": _number(interval["ci_low"]),
            "ci_high": _number(interval["ci_high"]),
            "archived_ci_low": _number(retired["ci_low"]),
            "archived_ci_high": _number(retired["ci_high"])}


def _excludes_zero(row: dict, *, prefix: str = "") -> bool | None:
    low, high = row.get(f"{prefix}ci_low"), row.get(f"{prefix}ci_high")
    if low is None or high is None:
        return None
    return bool(low > 0.0 or high < 0.0)


def _seed_key(seed: str):
    seed = str(seed)
    return (0, int(seed)) if seed.isdigit() else (1, seed)


def _daily_column(frame: pd.DataFrame) -> pd.Series:
    return frame.set_index("date")["rank_ic"]


def pair_stats(series: dict, windows: list[str], candidate: str, control: str,
               *, block: int, draws: int, seed: int) -> dict:
    """Per-window and pooled paired daily rank_ic differences, plus blend-level check."""
    per_window: dict[str, dict] = {}
    pooled: list[np.ndarray] = []
    per_seed_pooled: dict[str, list[np.ndarray]] = {}
    blend_per_window: dict[str, dict] = {}
    blend_pooled: list[np.ndarray] = []
    for window in windows:
        cand = series.get(window, {}).get(candidate)
        ctrl = series.get(window, {}).get(control)
        if not cand or not ctrl:
            continue
        common = [s for s in cand if s != BLEND and s in ctrl]
        deltas: list[np.ndarray] = []
        for seed_name in common:
            left = _daily_column(cand[seed_name])
            right = _daily_column(ctrl[seed_name])
            index = left.index.intersection(right.index)
            if len(index) == 0:
                continue
            delta = (left.loc[index] - right.loc[index]).to_numpy(dtype=float)
            deltas.append(delta)
            per_seed_pooled.setdefault(seed_name, []).append(delta)
        window_delta = (np.concatenate(deltas) if deltas else np.array([], dtype=float))
        row = {"seeds": common, **interval_summary(window_delta, block=block, draws=draws,
                                                   seed=seed)}
        row["positive"] = bool(row["mean"] is not None and row["mean"] > 0)
        row["excludes_zero"] = _excludes_zero(row)
        row["archived_excludes_zero"] = _excludes_zero(row, prefix="archived_")
        per_window[window] = row
        if len(window_delta):
            pooled.append(window_delta)
        if BLEND in cand and BLEND in ctrl:
            left = _daily_column(cand[BLEND])
            right = _daily_column(ctrl[BLEND])
            index = left.index.intersection(right.index)
            blend_delta = (left.loc[index] - right.loc[index]).to_numpy(dtype=float)
            if len(blend_delta):
                blend_pooled.append(blend_delta)
            blend_per_window[window] = interval_summary(blend_delta, block=block, draws=draws,
                                                       seed=seed)

    pooled_values = np.concatenate(pooled) if pooled else np.array([], dtype=float)
    summary = interval_summary(pooled_values, block=block, draws=draws, seed=seed)
    summary["excludes_zero"] = _excludes_zero(summary)
    summary["archived_excludes_zero"] = _excludes_zero(summary, prefix="archived_")
    summary["windows"] = len(per_window)
    summary["positive_windows"] = sum(1 for row in per_window.values() if row["positive"])
    summary["positive_window_names"] = [w for w, row in per_window.items() if row["positive"]]
    per_seed = {name: interval_summary(np.concatenate(chunks), block=block, draws=draws,
                                       seed=seed)
                for name, chunks in per_seed_pooled.items()}
    seed_point_mean = (float(np.mean([row["mean"] for row in per_seed.values()]))
                       if per_seed else None)
    blend_summary = interval_summary(np.concatenate(blend_pooled), block=block, draws=draws,
                                     seed=seed) if blend_pooled else None
    if blend_summary is not None:
        blend_summary["windows"] = len(blend_per_window)
        blend_summary["positive_windows"] = sum(
            1 for row in blend_per_window.values()
            if row["mean"] is not None and row["mean"] > 0)
    return {"candidate": candidate, "control": control,
            "per_window": per_window, "pooled": summary,
            "per_seed": per_seed, "seed_point_mean": seed_point_mean,
            "blend": {"per_window": blend_per_window, "pooled": blend_summary}}


def arm_levels(series: dict, windows: list[str], arms) -> dict:
    levels = {}
    for arm in arms:
        frames = [series[window][arm][BLEND].assign(window=window)
                  for window in windows if arm in series.get(window, {})]
        if not frames:
            continue
        pooled = pd.concat(frames, ignore_index=True)
        levels[arm] = {"days": int(len(pooled)),
                       "rank_ic_mean": float(pooled["rank_ic"].mean()),
                       "turnover_mean": float(pooled["turnover"].mean()),
                       "long_short_mean": float(pooled["long_short_return"].mean())}
    return levels


# --------------------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------------------- #
def arm_map_from_plan(plan: dict, run_dir: str | Path) -> dict:
    arms: dict[str, dict[str, dict]] = {}
    for version in plan.get("versions", []):
        arm, seed = version.get("arm"), version.get("seed")
        if arm is None or seed is None:
            continue
        arms.setdefault(str(arm), {})[str(seed)] = {"run_dir": str(run_dir),
                                                    "version": version["name"]}
    return arms


def load_sources(path: str | Path) -> tuple[dict, dict]:
    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    arms = payload.get("arms", payload)
    resolved: dict[str, dict[str, dict]] = {}
    for arm, seeds in arms.items():
        resolved[str(arm)] = {}
        for seed, spec in seeds.items():
            run_dir = Path(spec["run_dir"])
            if not run_dir.is_absolute():
                run_dir = (source_path.parent / run_dir).resolve()
            resolved[str(arm)][str(seed)] = {"run_dir": str(run_dir),
                                             "version": spec["version"]}
    return resolved, payload


def resolve_pair_arms(plan: dict | None, arms: dict) -> list[tuple[str, str]]:
    if plan and plan.get("pairs"):
        version_to_arm = {version["name"]: str(version["arm"]) for version in
                          plan.get("versions", []) if version.get("arm") and version.get("name")}
        pairs = []
        for left, right in plan["pairs"]:
            left_arm = version_to_arm.get(left, left)
            right_arm = version_to_arm.get(right, right)
            if left_arm in arms and right_arm in arms:
                pairs.append((left_arm, right_arm))
        if pairs:
            return pairs
    available = {arm: i for i, arm in enumerate(arms)}
    return [(cand, ctrl) for cand, ctrl in PAIR_ORDER
            if cand in available and ctrl in available]


# --------------------------------------------------------------------------------------- #
# main comparison
# --------------------------------------------------------------------------------------- #
def evaluate_arms(arms: dict, windows: list[str], *, smooth: bool = True) -> dict:
    series: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
    for window in windows:
        by_dir: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for seeds in arms.values():
            for spec in seeds.values():
                run_dir = str(spec["run_dir"])
                if run_dir not in by_dir:
                    by_dir[run_dir] = load_window(run_dir, window)
        reference_dir = next(iter(by_dir))
        reference_keys, reference_targets = by_dir[reference_dir]
        for run_dir, (keys, targets) in by_dir.items():
            if len(keys) != len(reference_keys) or not keys[KEYS].equals(reference_keys[KEYS]):
                raise SystemExit(f"{window}: key frames differ between {run_dir} and "
                                 f"{reference_dir}")
            if not targets.equals(reference_targets):
                raise SystemExit(f"{window}: target frames differ between {run_dir} and "
                                 f"{reference_dir}")
        keys, targets = reference_keys, reference_targets
        series[window] = {}
        for arm, seeds in arms.items():
            arm_series: dict[str, pd.DataFrame] = {}
            stack = []
            for seed_name in sorted(seeds, key=_seed_key):
                spec = seeds[seed_name]
                score = load_scores(spec["run_dir"], window, spec["version"], keys)
                stack.append(score)
                arm_series[seed_name] = daily_scores(keys, targets, score, smooth=smooth)
            blend = np.mean(np.stack(stack), axis=0)
            arm_series[BLEND] = daily_scores(keys, targets, blend, smooth=smooth)
            series[window][arm] = arm_series
        print(f"[{window}] keys={len(keys)} targets={len(targets)} "
              f"arms={sorted(series[window])}", flush=True)
    return series


def reference_check(series: dict, windows: list[str], arm: str, *,
                    expect_rank_ic: float, expect_turnover: float,
                    tolerance: float) -> dict:
    frames = [series[window][arm][BLEND] for window in windows
              if arm in series.get(window, {})]
    if not frames:
        return {"ok": False, "error": f"arm {arm} not evaluated on {windows}"}
    pooled = pd.concat(frames, ignore_index=True)
    rank_ic = float(pooled["rank_ic"].mean())
    turnover = float(pooled["turnover"].mean())
    diff_ic = abs(rank_ic - expect_rank_ic)
    diff_to = abs(turnover - expect_turnover)
    return {"arm": arm, "windows": windows, "days": int(len(pooled)),
            "rank_ic": rank_ic, "turnover": turnover,
            "expected": {"rank_ic": expect_rank_ic, "turnover": expect_turnover},
            "abs_diff": {"rank_ic": diff_ic, "turnover": diff_to},
            "tolerance": tolerance,
            "ok": bool(diff_ic <= tolerance and diff_to <= tolerance)}


def _interval_cells(row: dict) -> str:
    if row["mean"] is None:
        return f"{'-':>11}{'-':>11}{'-':>11}"
    return (f"{row['mean']:>+11.5f}{row['ci_low']:>+12.5f}{row['ci_high']:>+12.5f}")


def print_pair_table(name: str, result: dict) -> None:
    print(f"\n{name}  (candidate minus control, daily rank_ic, smoothed)")
    print(f"{'window':<12}{'days':>6}{'mean':>11}{'new_ci_low':>12}{'new_ci_high':>12}"
          f"{'old_ci_low':>12}{'old_ci_high':>12}  positive  new!=0  old!=0")
    for window, row in result["per_window"].items():
        if row["mean"] is None:
            print(f"{window:<12}{row['days']:>6}{'-':>11}")
            continue
        print(f"{window:<12}{row['days']:>6}{_interval_cells(row)}"
              f"{row['archived_ci_low']:>+12.5f}{row['archived_ci_high']:>+12.5f}"
              f"  {'yes' if row['positive'] else 'no ':>8}"
              f"  {str(row['excludes_zero']):>6}  {str(row['archived_excludes_zero']):>6}")
    pooled = result["pooled"]
    if pooled["mean"] is not None:
        print(f"{'pooled':<12}{pooled['days']:>6}{_interval_cells(pooled)}"
              f"{pooled['archived_ci_low']:>+12.5f}{pooled['archived_ci_high']:>+12.5f}"
              f"  {pooled['positive_windows']}/{pooled['windows']:<7}"
              f"  {str(pooled['excludes_zero']):>6}  {str(pooled['archived_excludes_zero']):>6}")


# --------------------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------------------- #
def _synthetic_case(*, seed: int, windows: tuple[str, ...], days: int, instruments: int,
                    times: int, seeds: tuple[int, ...], signal: float, noise: float,
                    noise_extra: float) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    series: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
    frame_store: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for window in windows:
        dates = pd.bdate_range("2023-01-02", periods=days)
        clock = pd.to_timedelta(np.arange(times) * 30, unit="m") + pd.Timedelta(
            hours=9, minutes=30)
        timestamps = pd.DatetimeIndex([day + offset for day in dates for offset in clock])
        keys = pd.MultiIndex.from_product(
            [timestamps, [f"S{i:03d}" for i in range(instruments)]],
            names=KEYS).to_frame(index=False).sort_values(KEYS).reset_index(drop=True)
        # decimal period returns (1% scale) so the eval-v2 notional base stays valid
        target = 0.01 * rng.standard_normal(len(keys))
        keys = keys.assign(target=target)
        grouped = keys.groupby("date")["target"]
        z_target = ((target - grouped.transform("mean"))
                    / grouped.transform(lambda values: values.std(ddof=0))).to_numpy()
        targets = keys[KEYS + ["target"]]
        series[window] = {"B": {}, "C": {}}
        base_scores: list[np.ndarray] = []
        candidate_scores: list[np.ndarray] = []
        for seed_number in seeds:
            base = 0.35 * z_target + noise * rng.standard_normal(len(keys))
            candidate = (base + signal * z_target
                         + noise_extra * rng.standard_normal(len(keys)))
            base_scores.append(base)
            candidate_scores.append(candidate)
            series[window]["B"][str(seed_number)] = daily_scores(keys, targets, base)
            series[window]["C"][str(seed_number)] = daily_scores(keys, targets, candidate)
        series[window]["B"][BLEND] = daily_scores(
            keys, targets, np.mean(np.stack(base_scores), axis=0))
        series[window]["C"][BLEND] = daily_scores(
            keys, targets, np.mean(np.stack(candidate_scores), axis=0))
        frame_store[window] = (keys[KEYS], targets)
    return series, frame_store


def _ar1(phi: float, n: int, rng, *, mean: float = 0.004, sigma: float = 0.01) -> np.ndarray:
    innovations = rng.standard_normal(n) * sigma
    values = np.empty(n)
    values[0] = mean + innovations[0] / np.sqrt(max(1.0 - phi * phi, 1e-6))
    for index in range(1, n):
        values[index] = mean + phi * (values[index - 1] - mean) + innovations[index]
    return values


def run_self_test(out_path: str | Path | None = None, *, draws: int = 2000,
                  draws_fast: int = 500, reps: int = 300) -> dict:
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: dict) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {json.dumps(detail)}", flush=True)

    # 1. known positive paired difference, two windows, three seeds.
    series, _ = _synthetic_case(seed=11, windows=("w1", "w2"), days=24, instruments=40,
                                times=3, seeds=(11, 29, 47), signal=0.12, noise=0.9,
                                noise_extra=0.05)
    result = pair_stats(series, ["w1", "w2"], "C", "B", block=BLOCK_DAYS, draws=draws,
                        seed=20261028)
    pooled = result["pooled"]
    ok = (pooled["mean"] is not None and pooled["mean"] > 0
          and pooled["ci_low"] is not None and pooled["ci_low"] > 0
          and pooled["positive_windows"] == 2 and len(result["per_seed"]) == 3)
    check("known_positive_delta", ok, {"pooled": pooled,
                                       "per_window_mean": {w: row["mean"] for w, row in
                                                           result["per_window"].items()}})
    if ok:
        deltas = []
        for window in ("w1", "w2"):
            for seed_name in ("11", "29", "47"):
                left = series[window]["C"][seed_name].set_index("date")["rank_ic"]
                right = series[window]["B"][seed_name].set_index("date")["rank_ic"]
                index = left.index.intersection(right.index)
                deltas.append((left.loc[index] - right.loc[index]).to_numpy(dtype=float))
        delta = np.concatenate(deltas)
        se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
        half = 0.5 * (pooled["ci_high"] - pooled["ci_low"])
        ratio = half / (1.96 * se) if se > 0 else None
        check("known_delta_ci_matches_iid_width",
              ratio is not None and 0.7 <= ratio <= 1.6,
              {"mean": pooled["mean"], "ci": [pooled["ci_low"], pooled["ci_high"]],
               "iid_standard_error": se, "half_width_over_iid_half": ratio,
               "note": "the corrected whole-series block interval must sit close to the iid "
                       "interval of the pooled daily mean for a serially uncorrelated delta"})
        archived_half = 0.5 * (pooled["archived_ci_high"] - pooled["archived_ci_low"])
        archived_ratio = archived_half / (1.96 * se) if se > 0 else None
        sqrt_blocks = float(np.sqrt(len(delta) / BLOCK_DAYS))
        check("retired_caliber_is_sqrt_blocks_too_wide",
              archived_ratio is not None
              and abs(archived_ratio / sqrt_blocks - 1.0) <= 0.35,
              {"archived_half_width_over_iid_half": archived_ratio,
               "sqrt_days_over_block": sqrt_blocks,
               "note": "the pre-2026-10-28 single-block interval is sqrt(n / block) too wide; "
                       "it stays visible side by side and never feeds a verdict"})

    # 2. identical arms must pair to exactly zero (no smoothing leakage, no key drift).
    null_series, _ = _synthetic_case(seed=29, windows=("w1", "w2"), days=24,
                                     instruments=40, times=3, seeds=(11, 29, 47),
                                     signal=0.0, noise=0.9, noise_extra=0.0)
    for window, arms in null_series.items():
        arms["C"] = dict(arms["B"])
    null = pair_stats(null_series, ["w1", "w2"], "C", "B", block=BLOCK_DAYS, draws=draws,
                      seed=20261028)
    null_pooled = null["pooled"]
    exact = (null_pooled["mean"] == 0.0 and null_pooled["ci_low"] == 0.0
             and null_pooled["ci_high"] == 0.0)
    check("identical_arms_exact_zero", exact, {"pooled": null_pooled})

    # 3. row order must not matter: the loaders sort, the EWMA reindexes to the key frame.
    keys, targets = _synthetic_case(seed=37, windows=("w1",), days=8, instruments=25,
                                    times=3, seeds=(11,), signal=0.1, noise=0.9,
                                    noise_extra=0.1)[1]["w1"]
    rng = np.random.default_rng(5)
    scores = rng.standard_normal(len(keys))
    baseline = daily_scores(keys, targets, scores)
    order = rng.permutation(len(keys))
    permuted = daily_scores(keys.iloc[order].reset_index(drop=True),
                            targets.iloc[order].reset_index(drop=True),
                            scores[order])
    same = bool(np.allclose(baseline["rank_ic"], permuted["rank_ic"], atol=1e-12)
                and np.allclose(baseline["turnover"], permuted["turnover"], atol=1e-12))
    check("row_order_invariance", same, {"max_abs_diff": float(
        np.max(np.abs(baseline["rank_ic"].to_numpy() - permuted["rank_ic"].to_numpy())))})

    # 4. time correlation: the 5-day block interval must widen with AR(1) persistence and
    #    track the theoretical standard deviation of a 5-day mean (random start + wrap).
    dependence = {}
    for phi in (0.0, 0.6, 0.9):
        rng = np.random.default_rng(101)
        widths, brackets, points = [], 0, []
        for _ in range(reps):
            values = _ar1(phi, 120, rng)
            block = block_interval(values, block=BLOCK_DAYS, draws=draws_fast, seed=7)
            widths.append(block["ci_high"] - block["ci_low"])
            brackets += int(block["ci_low"] <= block["mean"] <= block["ci_high"])
            points.append(block["mean"])
        points = np.asarray(points, dtype=float)
        dependence[str(phi)] = {"mean_width": float(np.mean(widths)),
                                "mean_point": float(points.mean()),
                                "bracket_rate": brackets / reps}
    base_width = dependence["0.0"]["mean_width"]
    for phi in (0.0, 0.6, 0.9):
        row = dependence[str(phi)]
        factor = 1.0 + 2.0 * sum((BLOCK_DAYS - lag) / BLOCK_DAYS * phi ** lag
                                 for lag in range(1, BLOCK_DAYS))
        row["width_ratio"] = row["mean_width"] / base_width if base_width else None
        # _ar1 draws innovation innovations with sd sigma, so the stationary sd is
        # sigma / sqrt(1 - phi^2); the block-mean spread carries both terms.
        row["theory_ratio"] = float(np.sqrt(factor / (1.0 - phi ** 2)))
    check("block_ci_width_tracks_persistence",
          all(abs(dependence[str(phi)]["width_ratio"]
                  / dependence[str(phi)]["theory_ratio"] - 1.0) <= 0.35
              for phi in (0.0, 0.6, 0.9)),
          dependence)
    check("block_ci_stays_centered",
          all(abs(dependence[str(phi)]["mean_point"] - 0.004) <= 0.002
              and dependence[str(phi)]["bracket_rate"] >= 0.95 for phi in (0.0, 0.6, 0.9)),
          {phi: {"mean_point": dependence[phi]["mean_point"],
                 "bracket_rate": dependence[phi]["bracket_rate"]} for phi in dependence})

    payload = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "checks": checks, "ok": all(item["ok"] for item in checks),
               "generator": {"instruments": 40, "times": 3, "signal": 0.12,
                             "note": "synthetic frames, not replayed predictions"}}
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(payload, indent=1, allow_nan=False),
                                  encoding="utf-8")
    print(f"\nself-test {'passed' if payload['ok'] else 'FAILED'}")
    return payload


# --------------------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan")
    parser.add_argument("--run-dir")
    parser.add_argument("--sources")
    parser.add_argument("--arms", help="comma separated arm subset, e.g. A or B,C")
    parser.add_argument("--windows", help="comma separated window subset in plan order")
    parser.add_argument("--out")
    parser.add_argument("--no-smooth", action="store_true",
                        help="compare raw scores; default applies the project EWMA")
    parser.add_argument("--block", type=int, default=BLOCK_DAYS)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("--check-arm")
    parser.add_argument("--check-windows")
    parser.add_argument("--expect-rank-ic", type=float)
    parser.add_argument("--expect-turnover", type=float)
    parser.add_argument("--tolerance", type=float, default=0.0005)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        payload = run_self_test(args.out)
        return 0 if payload["ok"] else 1
    if not args.out:
        parser.error("--out is required for a comparison run")

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8")) if args.plan else None
    plan_windows = [window["name"] for window in plan.get("windows", [])] if plan else []
    if args.sources:
        arms, source_payload = load_sources(args.sources)
        windows = plan_windows or [str(name) for name in source_payload.get("windows", [])]
    elif args.run_dir:
        if plan is None:
            parser.error("--run-dir requires --plan")
        arms = arm_map_from_plan(plan, args.run_dir)
        windows = plan_windows
    else:
        parser.error("one of --run-dir, --sources or --self-test is required")
    if args.arms:
        requested_arms = [name.strip() for name in args.arms.split(",") if name.strip()]
        missing = [name for name in requested_arms if name not in arms]
        if missing:
            parser.error(f"unknown arms: {missing}")
        arms = {name: arms[name] for name in requested_arms}
    if args.windows:
        requested = [name.strip() for name in args.windows.split(",") if name.strip()]
        unknown = [name for name in requested if name not in windows]
        if unknown:
            parser.error(f"unknown windows: {unknown}")
        windows = requested
    if not windows:
        parser.error("no windows to evaluate")

    draws = args.draws
    if plan and args.draws == DEFAULT_DRAWS:
        draws = int(plan.get("bootstrap_draws", DEFAULT_DRAWS))
    block = args.block
    if plan and args.block == BLOCK_DAYS:
        block = int(plan.get("comparison", {}).get("block_days", BLOCK_DAYS))
    bootstrap_seed = (args.bootstrap_seed if args.bootstrap_seed is not None
                      else int(plan.get("seed", 20261026)) if plan else 20261026)

    print(f"arms: {json.dumps({arm: {seed: spec['version'] for seed, spec in seeds.items()} for arm, seeds in arms.items()})}")
    print(f"windows: {windows}")
    print(f"block={block} draws={draws} bootstrap_seed={bootstrap_seed} "
          f"smooth={not args.no_smooth}")

    series = evaluate_arms(arms, windows, smooth=not args.no_smooth)
    pairs = resolve_pair_arms(plan, arms)
    pair_results = {}
    for candidate, control in pairs:
        result = pair_stats(series, windows, candidate, control, block=block, draws=draws,
                            seed=bootstrap_seed)
        pair_results[f"{candidate}-{control}"] = result
        print_pair_table(f"{candidate} - {control}", result)

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": (plan.get("experiment_id") if plan else None),
        "caliber": {
            "evaluation_version": EVALUATION_VERSION,
            "quantile": 0.2, "periods_per_year": 252, "min_cross_section": 3,
            "constant_portfolios": "undefined", "single_side_rate": 0.0,
            "turnover_conversion_factor": TURNOVER_CONVERSION,
            "smoothing": {"operator": "causal_ewma", "alpha": ALPHA,
                          "applied": not args.no_smooth},
            "pairing": ("candidate minus control on eval-v2 daily rank_ic, paired by "
                        "(arm, seed, trading day), then pooled across seeds and windows"),
            "bootstrap": {"method": "scripts/compare_blend_replay.block_interval",
                          "block_days": block, "draws": draws,
                          "seed": bootstrap_seed,
                          "resampling": "ceil(n / block) random-start wrap-around blocks over "
                                        "the pooled daily vector ordered window -> seed -> day, "
                                        "concatenated to ceil(n / block) * block days and "
                                        "averaged per replicate (corrected 2026-10-28)",
                          "corrected_at_utc": "2026-10-28",
                          "archived_secondary": {
                              "method": "scripts/compare_blend_replay."
                                        "archived_single_block_interval",
                              "resampling": "one 5-day block mean per replicate",
                              "role": "side-by-side reproduction of the retired interval, never "
                                      "used for a decision"}},
            "decision_metric": "rank_ic paired difference with the corrected interval",
        },
        "windows": windows,
        "sources": ({arm: {seed: spec for seed, spec in seeds.items()}
                     for arm, seeds in arms.items()}),
        "arms": arm_levels(series, windows, list(arms)),
        "pairs": pair_results,
    }
    if args.check_arm:
        if args.expect_rank_ic is None or args.expect_turnover is None:
            parser.error("--check-arm requires --expect-rank-ic and --expect-turnover")
        check_windows = ([name.strip() for name in args.check_windows.split(",")]
                         if args.check_windows else windows)
        unknown = [name for name in check_windows if name not in windows]
        if unknown:
            parser.error(f"unknown windows for --check-arm: {unknown}")
        check = reference_check(series, check_windows, args.check_arm,
                                expect_rank_ic=args.expect_rank_ic,
                                expect_turnover=args.expect_turnover,
                                tolerance=args.tolerance)
        payload["reference_check"] = check
        print(f"\nreference check [{args.check_arm}] on {check_windows}: "
              f"rank_ic={check.get('rank_ic')} turnover={check.get('turnover')} "
              f"ok={check.get('ok')}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, allow_nan=False), encoding="utf-8")
    print(f"\nwrote {out_path}")
    if "reference_check" in payload and not payload["reference_check"]["ok"]:
        print("REFERENCE CHECK FAILED: caliber does not reproduce the published A arm")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
