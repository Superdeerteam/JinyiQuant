"""Ten-month post-processing study over frozen replay scores (zero training).

Stages
1. load the dev10 replay predictions (18 frozen checkpoints, ten development months)
   and build the declared arms: six submission incumbents, four dose rungs and the
   441d 2.64 candidate.
2. apply the three pre-declared transforms (per-timestamp winsorize, volatility
   scaling against the 20-day trailing realised volatility, cross-sectional centred
   average rank) and their declared compositions.
3. evaluate every arm and variant on the eval-v2 contract, gross and under the local
   BARRA-style neutralization approximation, and run the registered paired-day
   month-aware block bootstrap.
4. execute the registered null checks, the adoption gate and the nine-blend
   five-month confirmation (A4), then write results.json and daily.csv.

Read-only over archived predictions; the transforms train nothing and read no
future label (sigma at day d uses days d-1 and earlier only).

    python scripts/dev10_postprocess.py [--run <run_dir>] [--out <dir>] [--skip-nine]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jinyiquant.config import EvaluationConfig  # noqa: E402
from jinyiquant.evaluation import EVALUATION_VERSION, aggregate_metric  # noqa: E402
from scripts.compare_multiscale_replay import (  # noqa: E402
    arm_scores,
    evaluate_scores,
    load_exposure,
    load_window,
    neutralize,
)
from scripts.report_platform_metrics import (  # noqa: E402
    pooled_stress,
    stress_buckets,
    stress_conditions,
)

PLAN = ROOT / "configs/dev10_replay_20261028_plan.json"
RUN_ROOT = ROOT / "runs/dev10-replay-20261028"
OUT_ROOT = ROOT / "runs/dev10-postprocess-20261028"
LABELS = ROOT / "数据/cpt_jyc_2026_vwap"
EXPOSURE = [ROOT / f"数据/cpt_jyc_2026_exposure/cpt_jyc_2026_exposure_{y}_part_0001.parquet" for y in (2023, 2024)]
SCREEN_RUN = ROOT / "runs/multiscale-replay-20261026/20261026T133118.869042Z_bcdc779dc66b"
VERIFY_RUN = ROOT / "runs/multiscale-replay-verify-20261026/20261026T140057.561539Z_1a57d4b38acc"
NINE_MEMBERS = ("tlob_s11", "tlob_s29", "tlob_s47", "mlp_s11", "mlp_s29", "mlp_s47",
                "patch_s11", "patch_s29", "patch_s47")
OLD_FIVE = ("nov2023", "jan2024", "mar2024", "apr2024", "jun2024")

CFG = EvaluationConfig(quantile=0.2, periods_per_year=252, min_cross_section=3,
                       constant_portfolios="undefined", single_side_rate=0.0)
TRANSFORMS = ("raw", "vol", "vol_rank", "vol_winsor", "winsor", "winsor_rank", "rank", "all3")
# Pre-declared preference order for near-ties inside one standard error (prereg).
PRIORITY = ("vol", "vol_winsor", "vol_rank", "all3", "winsor", "winsor_rank", "rank")
BLOCK_DAYS = 5
DRAWS = 4000
SEED = 20261028
CONFIDENCE = 0.95
GATE_RANK_IC = -0.002
GATE_TURNOVER = 0.10
N1_TOL = 1e-9
N2_TOL = 5e-4
VOL_WINDOW = 20
VOL_MIN_HISTORY = 10
VOL_FLOOR_QUANTILE = 0.05
ARM_GROUPS = {
    "incumbent6": ("tlob_", "mlp_"),
    "rung132": ("r132_",),
    "rung240": ("r240_",),
    "rung441": ("r441_",),
    "cand441p264": ("r441p_",),
}
DAILY_COLUMNS = ("date", "ic", "rank_ic", "long_short_return", "turnover", "net_return", "cost")


def sharpe_delta_interval(raw: np.ndarray, cand: np.ndarray, months: np.ndarray, *, block: int,
                          draws: int, seed: int, confidence: float,
                          periods_per_year: int = 252) -> dict:
    """Paired block-bootstrap interval for the realised Sharpe delta (registered statistic)."""
    frame = pd.DataFrame({"raw": np.asarray(raw, dtype=float), "cand": np.asarray(cand, dtype=float),
                          "month": np.asarray(months)})
    frame = frame[np.isfinite(frame["raw"]) & np.isfinite(frame["cand"])]
    if frame.empty:
        return {"mean": None, "se": None, "low": None, "high": None, "days": 0}

    def sharpe(values: np.ndarray) -> float:
        if len(values) < 2:
            return float("nan")
        std = float(np.std(values, ddof=1))
        return float("nan") if std == 0.0 else float(np.mean(values) / std * np.sqrt(periods_per_year))

    def simulate(index: np.ndarray) -> float:
        return sharpe(frame["cand"].to_numpy(dtype=float)[index]) - sharpe(frame["raw"].to_numpy(dtype=float)[index])

    point = sharpe(frame["cand"].to_numpy(dtype=float)) - sharpe(frame["raw"].to_numpy(dtype=float))
    groups = [np.flatnonzero(frame["month"].to_numpy() == key) for key in sorted(frame["month"].unique())]
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=float)
    for draw in range(draws):
        picked = []
        for positions in groups:
            n = len(positions)
            width = min(block, n)
            starts = rng.integers(0, n, size=(n + width - 1) // width)
            local = (starts[:, None] + np.arange(width)[None, :]) % n
            picked.append(positions[np.sort(local.reshape(-1)[:n])])
        sampled[draw] = simulate(np.sort(np.concatenate(picked)))
    sampled = sampled[np.isfinite(sampled)]
    alpha = (1.0 - confidence) / 2.0
    return {"mean": None if not np.isfinite(point) else float(point),
            "se": float(np.std(sampled, ddof=1)) if len(sampled) > 1 else None,
            "low": float(np.quantile(sampled, alpha)) if len(sampled) else None,
            "high": float(np.quantile(sampled, 1 - alpha)) if len(sampled) else None,
            "days": int(len(frame)), "block_days": block, "draws": draws, "seed": seed,
            "months": int(frame["month"].nunique()),
            "sampled_draws": int(len(sampled))}


def month_of(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime("%Y-%m")


def block_bootstrap(diffs: np.ndarray, months: np.ndarray, *, block: int, draws: int,
                    seed: int, confidence: float) -> dict:
    """Moving-block bootstrap keeping every block inside one month (registered estimator)."""
    frame = pd.DataFrame({"diff": np.asarray(diffs, dtype=float), "month": np.asarray(months)})
    frame = frame[np.isfinite(frame["diff"])]
    if frame.empty:
        return {"mean": None, "se": None, "low": None, "high": None, "share_positive": None,
                "days": 0}
    groups = [(key, frame.loc[frame.month == key, "diff"].to_numpy(dtype=float))
              for key in sorted(frame.month.unique())]
    if not groups or any(len(values) < 2 for _, values in groups):
        # Fall back to the pooled moving block when a month has fewer than two days.
        groups = [("pooled", frame["diff"].to_numpy(dtype=float))]
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=float)
    for draw in range(draws):
        total, count = 0.0, 0
        for _, values in groups:
            n = len(values)
            width = min(block, n)
            starts = rng.integers(0, n, size=(n + width - 1) // width)
            indices = (starts[:, None] + np.arange(width)[None, :]) % n
            take = indices.reshape(-1)[:n]
            total += float(values[take].sum())
            count += n
        sampled[draw] = total / count
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(frame["diff"].mean()),
        "se": float(sampled.std(ddof=1)),
        "low": float(np.quantile(sampled, alpha)),
        "high": float(np.quantile(sampled, 1 - alpha)),
        "share_positive": float((sampled > 0).mean()),
        "days": int(len(frame)),
        "block_days": block,
        "draws": draws,
        "seed": seed,
        "months": int(frame["month"].nunique()),
    }


# --------------------------------------------------------------------------- sigma
def sigma_panel(labels: pd.DataFrame) -> pd.DataFrame:
    """Trailing realised volatility from (day, instrument, vwap_return) rows.

    sigma_i(d) uses only trading days strictly before d, so a label of day d can
    never leak into day d's scale.
    """
    frame = labels[np.isfinite(labels.vwap_return)]
    frame = frame.assign(square=frame.vwap_return.to_numpy(dtype=float) ** 2)
    msq = frame.groupby(["day", "instrument"], sort=True).square.mean()
    panel = msq.unstack("instrument")
    sigma = np.sqrt(panel.rolling(VOL_WINDOW, min_periods=VOL_MIN_HISTORY).mean().shift(1))
    floor = sigma.apply(lambda row: row.dropna().quantile(VOL_FLOOR_QUANTILE)
                        if row.notna().any() else np.nan, axis=1)
    sigma = sigma.clip(lower=floor.to_numpy()[:, None], axis=0)
    filled = int(sigma.isna().sum().sum())
    # Declared fallback: cross-sectional date median, then the previous day's own
    # scale for any residual gap (causal); everything else is dropped.
    sigma = sigma.apply(lambda row: row.fillna(row.median()) if row.notna().any() else row, axis=1)
    sigma = sigma.apply(lambda column: column.ffill(), axis=0)  # prior-day sigma only, never future
    sigma = sigma.clip(lower=1e-8)
    still = int(sigma.isna().sum().sum())
    sigma = sigma.dropna(how="all")
    return sigma, filled, still


def load_sigma(months: list[str]) -> pd.DataFrame:
    """Long (day, instrument, sigma) table for the requested label months."""
    frames = []
    for month in months:
        path = LABELS / f"cpt_jyc_2026_vwap_{month}.parquet"
        frame = pd.read_parquet(path, columns=["date", "instrument", "vwap_return"])
        frame["day"] = pd.to_datetime(frame["date"]).dt.normalize()
        frames.append(frame[["day", "instrument", "vwap_return"]])
    sigma, filled, unfilled = sigma_panel(pd.concat(frames, ignore_index=True))
    long = sigma.stack().rename("sigma").reset_index()
    long.columns = ["day", "instrument", "sigma"]
    long["fills"] = filled
    long["unfilled"] = unfilled
    return long


# ---------------------------------------------------------------------- transforms
def winsorize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["date", "instrument", "score"]].copy()
    quantiles = out.groupby("date").score.quantile([0.01, 0.99]).unstack()
    out["score"] = out.score.clip(lower=out.date.map(quantiles[0.01]),
                                  upper=out.date.map(quantiles[0.99]))
    return out


def centered_rank(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["date", "instrument", "score"]].copy()
    grouped = out.groupby("date").score
    ranks = grouped.rank(method="average")
    size = grouped.transform("size").astype(float)
    out["score"] = (ranks - (size + 1.0) / 2.0) / (size / 2.0)
    return out


def volatility_scale(frame: pd.DataFrame, sigma: pd.DataFrame) -> pd.DataFrame:
    out = frame[["date", "instrument", "score"]].copy()
    out["day"] = pd.to_datetime(out.date).dt.normalize()
    merged = out.merge(sigma[["day", "instrument", "sigma"]], on=["day", "instrument"], how="left")
    merged["sigma"] = merged.sigma.fillna(merged.groupby("day").sigma.transform("median"))
    merged["score"] = np.where(merged.sigma.notna(),
                               merged.score.to_numpy(dtype=float) / merged.sigma.to_numpy(dtype=float),
                               merged.score.to_numpy(dtype=float))
    return merged[["date", "instrument", "score"]]


def variants_of(scores: pd.DataFrame, sigma: pd.DataFrame) -> dict[str, pd.DataFrame]:
    winsor = winsorize(scores)
    vol = volatility_scale(scores, sigma)
    vol_winsor = winsorize(vol)
    return {
        "raw": scores[["date", "instrument", "score"]].copy(),
        "winsor": winsor,
        "vol": vol,
        "vol_rank": centered_rank(vol),
        "vol_winsor": vol_winsor,
        "winsor_rank": centered_rank(winsor),
        "rank": centered_rank(scores),
        "all3": centered_rank(vol_winsor),
    }


# ---------------------------------------------------------------------- evaluation
def evaluate_variant(frame: pd.DataFrame, targets: pd.DataFrame, exposure, *, neutralized: bool):
    if neutralized:
        frame = neutralize(frame, exposure[1], exposure[2], quantile=0.2)
        metrics, daily, _ = evaluate_scores(frame, targets, CFG, "processed_score")
    else:
        metrics, daily, _ = evaluate_scores(frame, targets, CFG, "score")
    return metrics, daily[list(DAILY_COLUMNS) if set(DAILY_COLUMNS) <= set(daily) else
                          [c for c in DAILY_COLUMNS if c in daily]].copy()


def summarize(daily: pd.DataFrame, *, neutralized: bool, stress: float | None = None) -> dict:
    return {
        "days": int(len(daily)),
        "ic": aggregate_metric(daily.ic, "mean"),
        "ic_ir": aggregate_metric(daily.ic, "ir"),
        "rank_ic": aggregate_metric(daily.rank_ic, "mean"),
        "rank_ic_ir": aggregate_metric(daily.rank_ic, "ir"),
        "long_short_sharpe": aggregate_metric(daily.long_short_return, "sharpe"),
        "long_short_mean": aggregate_metric(daily.long_short_return, "mean"),
        "turnover": aggregate_metric(daily.turnover, "mean"),
        "worst_day_long_short": aggregate_metric(daily.long_short_return, "min"),
        "lower_tail_1pct_long_short": aggregate_metric(daily.long_short_return, "lower_tail"),
        "net_return_5bp": aggregate_metric(daily.long_short_return - 0.001 * daily.turnover, "mean"),
        "stress_rank_ic": stress,
        "track": "neutralized" if neutralized else "gross",
    }


def member_rank_ic(run: Path, window: str, member: str) -> float | None:
    path = run / "windows" / window / member / "metrics.json"
    if not path.exists():
        return None
    return float(json.loads(path.read_text(encoding="utf-8")).get("rank_ic_mean"))


def latest_run() -> Path:
    if not RUN_ROOT.exists():
        raise SystemExit(f"missing {RUN_ROOT}; run the replay first")
    runs = sorted(p for p in RUN_ROOT.iterdir() if (p / "plan.json").exists() and (p / "windows").exists())
    if not runs:
        raise SystemExit(f"no completed replay run under {RUN_ROOT}")
    return runs[-1]


def window_list(plan: dict) -> list[dict]:
    return plan["windows"]


def build_arms(plan: dict) -> dict[str, list[str]]:
    names = [version["name"] for version in plan["versions"]]
    arms = {}
    for arm, prefixes in ARM_GROUPS.items():
        members = [name for name in names if name.startswith(prefixes)]
        if arm == "incumbent6":
            members = [name for name in members if name.startswith(("tlob_", "mlp_"))]
        if not members:
            raise SystemExit(f"arm {arm}: no plan members")
        arms[arm] = members
    return arms


def run(analysis_id: str, skip_nine: bool) -> dict:
    replay_run = latest_run()
    replay_plan = json.loads((replay_run / "plan.json").read_text(encoding="utf-8"))
    windows = window_list(replay_plan)
    arms = build_arms(replay_plan)
    exposure = load_exposure({"_path": str(PLAN)})
    sigma = load_sigma([month for month in pd.period_range("2023-08", "2024-09", freq="M")
                        .strftime("%Y-%m")])
    print(f"[load] sigma rows={len(sigma)} fills={int(sigma.fills.max())} "
          f"unfilled={int(sigma.unfilled.max())}", flush=True)

    results = {
        "schema_version": 1,
        "analysis_id": analysis_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "evaluation_version": EVALUATION_VERSION,
        "replay_run": str(replay_run.relative_to(ROOT)).replace("\\", "/"),
        "prereg": {"path": "configs/dev10_postprocess_prereg_20261028.json"},
        "windows": {window["name"]: {"start": window["start"], "end": window["end"]} for window in windows},
        "arms": {arm: members for arm, members in arms.items()},
        "transforms": list(TRANSFORMS),
        "sigma": {"window_days": VOL_WINDOW, "min_history": VOL_MIN_HISTORY,
                  "floor_quantile": VOL_FLOOR_QUANTILE,
                  "fill_count": int(sigma.fills.max()), "unfilled_count": int(sigma.unfilled.max())},
        "metrics": {},
        "per_window": {},
        "deltas": {},
        "null_checks": {},
        "gates": {},
        "winner": None,
        "power": {},
        "nine_blend": {},
    }

    daily_rows: list[pd.DataFrame] = []
    raw_by_arm: dict[str, dict[str, pd.DataFrame]] = {}
    stress_by_arm: dict[str, dict[str, dict[str, dict]]] = {}
    for window in windows:
        name = window["name"]
        cond = stress_conditions(pd.Timestamp(window["start"]), pd.Timestamp(window["end"]),
                                 set(pd.read_parquet(replay_run / "windows" / name / "keys.parquet")
                                     .instrument.unique()))
        results["per_window"][name] = {"days": int(len(cond))}
        all_members = [member for members in arms.values() for member in members]
        keys, targets, frames = load_window(replay_run, name, all_members)
        coverage = {"keys": int(len(keys)), "targets": int(len(targets)),
                    "members": {member: int(len(frames[member])) for member in all_members},
                    "ratio": float(len(targets) / len(keys))}
        results["per_window"][name]["coverage"] = coverage
        results["per_window"][name]["stress_days"] = int(cond.vol_ratio.notna().sum())
        for arm, members in arms.items():
            scores = arm_scores(frames, {member: 1.0 / len(members) for member in members})
            variants = variants_of(scores, sigma)
            raw_by_arm.setdefault(arm, {})
            for variant, frame in variants.items():
                for neutral in (False, True):
                    metrics, daily = evaluate_variant(frame, targets, exposure, neutralized=neutral)
                    daily_metrics = summarize(daily, neutralized=neutral)
                    if neutral:
                        buckets = stress_buckets(daily, cond)
                        stress_by_arm.setdefault(arm, {}).setdefault(variant, {})[name] = buckets
                        daily_metrics["stress_rank_ic"] = None
                    else:
                        raw_by_arm[arm][variant] = daily
                        daily_metrics["stress_rank_ic"] = None
                    metrics.update(daily_metrics)
                    results["metrics"].setdefault(arm, {}).setdefault(variant, {})[
                        "neutralized" if neutral else "gross"] = metrics
                    part = daily.copy()
                    part["window"] = name
                    part["arm"] = arm
                    part["variant"] = variant
                    part["track"] = "neutralized" if neutral else "gross"
                    daily_rows.append(part)
            print(f"[eval] {name} {arm:<12} done", flush=True)

    # ------------------------------------------------------------ pooled aggregates
    daily = pd.concat(daily_rows, ignore_index=True)
    daily["month"] = month_of(daily.date)
    pooled = {arm: {} for arm in arms}
    for arm in arms:
        for variant in TRANSFORMS:
            pooled[arm][variant] = {}
            for track in ("gross", "neutralized"):
                part = daily[(daily.arm == arm) & (daily.variant == variant) & (daily.track == track)]
                summary = summarize(part, neutralized=(track == "neutralized"))
                if track == "neutralized":
                    per_window = stress_by_arm.get(arm, {}).get(variant, {})
                    summary["stress_rank_ic"] = (pooled_stress(per_window)["stress"]
                                                 if len(per_window) == len(windows) else None)
                pooled[arm][variant][track] = summary
                if track == "gross":
                    results["metrics"][arm][variant]["gross"] = summary
                else:
                    results["metrics"][arm][variant]["neutralized"] = summary
    # Store per-window aggregate metrics (gross daily rows per window already pooled above).
    for window in windows:
        name = window["name"]
        for arm in arms:
            for variant in TRANSFORMS:
                for track in ("gross", "neutralized"):
                    part = daily[(daily.window == name) & (daily.arm == arm)
                                 & (daily.variant == variant) & (daily.track == track)]
                    results["per_window"][name].setdefault("metrics", {}).setdefault(arm, {}).setdefault(
                        variant, {})[track] = summarize(part, neutralized=(track == "neutralized"))

    # ------------------------------------------------------------ paired deltas + gate
    gate_order = {name: index for index, name in enumerate(PRIORITY)}
    for arm in arms:
        for variant in TRANSFORMS:
            if variant == "raw":
                continue
            for track in ("gross", "neutralized"):
                base = daily[(daily.arm == arm) & (daily.variant == "raw") & (daily.track == track)][
                    ["date", "long_short_return", "rank_ic", "turnover", "cost"]]
                cand = daily[(daily.arm == arm) & (daily.variant == variant) & (daily.track == track)][
                    ["date", "long_short_return", "rank_ic", "turnover", "cost"]]
                merged = base.merge(cand, on="date", suffixes=("_raw", "_cand"), validate="one_to_one")
                delta = {
                    "long_short_return": block_bootstrap(
                        merged.long_short_return_cand.to_numpy() - merged.long_short_return_raw.to_numpy(),
                        month_of(merged.date).to_numpy(), block=BLOCK_DAYS, draws=DRAWS, seed=SEED,
                        confidence=CONFIDENCE),
                    "rank_ic": block_bootstrap(
                        merged.rank_ic_cand.to_numpy() - merged.rank_ic_raw.to_numpy(),
                        month_of(merged.date).to_numpy(), block=BLOCK_DAYS, draws=DRAWS, seed=SEED,
                        confidence=CONFIDENCE),
                    "turnover": block_bootstrap(
                        merged.turnover_cand.to_numpy() - merged.turnover_raw.to_numpy(),
                        month_of(merged.date).to_numpy(), block=BLOCK_DAYS, draws=DRAWS, seed=SEED,
                        confidence=CONFIDENCE),
                }
                delta["long_short_sharpe"] = sharpe_delta_interval(
                    merged.long_short_return_raw.to_numpy(dtype=float),
                    merged.long_short_return_cand.to_numpy(dtype=float),
                    month_of(merged.date).to_numpy(), block=BLOCK_DAYS, draws=DRAWS, seed=SEED,
                    confidence=CONFIDENCE)
                results["deltas"].setdefault(arm, {}).setdefault(variant, {})[track] = delta
                if track == "gross":
                    passed = (delta["long_short_return"]["low"] is not None
                              and delta["long_short_return"]["low"] > 0
                              and delta["rank_ic"]["mean"] is not None
                              and delta["rank_ic"]["mean"] >= GATE_RANK_IC
                              and delta["turnover"]["mean"] is not None
                              and delta["turnover"]["mean"] <= GATE_TURNOVER)
                    results["gates"].setdefault(arm, {})[variant] = {
                        "pass": bool(passed),
                        "rule": {"long_short_return_ci_low_gt_0": delta["long_short_return"]["low"],
                                 "rank_ic_point_ge_minus_0.002": delta["rank_ic"]["mean"],
                                 "turnover_point_le_plus_0.10": delta["turnover"]["mean"]},
                        "primary_track": "gross",
                        "sharpe_delta_interval": delta["long_short_sharpe"],
                        "neutralized_gate": {
                            "long_short_return_ci_low_gt_0": results["deltas"][arm][variant]["neutralized"]["long_short_return"]["low"],
                            "rank_ic_point_ge_minus_0.002": results["deltas"][arm][variant]["neutralized"]["rank_ic"]["mean"],
                            "turnover_point_le_plus_0.10": results["deltas"][arm][variant]["neutralized"]["turnover"]["mean"],
                        },
                    }

    # ------------------------------------------------------------------- null checks
    n1 = {}
    worst = 0.0
    for arm in arms:
        base = raw_by_arm[arm]["raw"]
        cand = raw_by_arm[arm]["rank"]
        merged = base.merge(cand, on="date", suffixes=("_raw", "_rank"), validate="one_to_one")
        diffs = {
            "long_short_return": float((merged.long_short_return_raw - merged.long_short_return_rank).abs().max()),
            "turnover": float((merged.turnover_raw - merged.turnover_rank).abs().max()),
            "rank_ic": float((merged.rank_ic_raw - merged.rank_ic_rank).abs().max()),
        }
        worst = max(worst, max(diffs.values()))
        n1[arm] = {"max_abs_daily_diff": diffs, "pass": max(diffs.values()) <= N1_TOL}
    n1_ok = all(entry["pass"] for entry in n1.values()) and worst <= N1_TOL

    n2 = {}
    n2_worst = 0.0
    for name in OLD_FIVE:
        for member in arms["incumbent6"]:
            current = member_rank_ic(replay_run, name, member)
            archived_run = SCREEN_RUN if name in ("nov2023", "jan2024", "mar2024") else VERIFY_RUN
            archived = member_rank_ic(archived_run, name, member)
            if current is None or archived is None:
                continue
            diff = abs(current - archived)
            n2_worst = max(n2_worst, diff)
            n2[f"{name}/{member}"] = {"new": current, "archived": archived, "abs_diff": diff}
    n2_ok = n2_worst <= N2_TOL

    n3 = {}
    for name, entry in results["per_window"].items():
        coverage = entry["coverage"]
        members = coverage["members"]
        bad = [member for member, count in members.items() if count < coverage["keys"] * 0.99]
        n3[name] = {"keys": coverage["keys"], "targets": coverage["targets"],
                    "ratio": coverage["ratio"], "members_below_0.99": bad,
                    "pass": coverage["ratio"] >= 0.99 and not bad}
    n3_ok = all(entry["pass"] for entry in n3.values())
    results["null_checks"] = {
        "N1_rank_selection_invariance": {"pass": bool(n1_ok), "tolerance": N1_TOL, "arms": n1},
        "N2_five_month_reproduction": {"pass": bool(n2_ok), "tolerance": N2_TOL,
                                       "max_abs_diff": n2_worst, "per_member": n2},
        "N3_coverage": {"pass": bool(n3_ok), "windows": n3},
    }

    # ------------------------------------------------------------------ power + winner
    best = None
    for variant in TRANSFORMS:
        if variant == "raw":
            continue
        delta = results["deltas"]["incumbent6"][variant]["gross"]
        old_part = daily[(daily.arm == "incumbent6") & (daily.variant == variant) & (daily.track == "gross")
                         & (daily.window.isin(OLD_FIVE))][["date", "long_short_return"]]
        old_base = daily[(daily.arm == "incumbent6") & (daily.variant == "raw") & (daily.track == "gross")
                         & (daily.window.isin(OLD_FIVE))][["date", "long_short_return"]]
        merged = old_base.merge(old_part, on="date", suffixes=("_raw", "_cand"))
        old_ci = block_bootstrap(merged.long_short_return_cand.to_numpy() - merged.long_short_return_raw.to_numpy(),
                                 month_of(merged.date).to_numpy(), block=BLOCK_DAYS, draws=DRAWS, seed=SEED,
                                 confidence=CONFIDENCE)
        results["power"][variant] = {
            "days_104": old_ci["days"], "se_104": old_ci["se"], "days_203": delta["long_short_return"]["days"],
            "se_203": delta["long_short_return"]["se"],
            "se_ratio": (None if not delta["long_short_return"]["se"] or not old_ci["se"]
                         else old_ci["se"] / delta["long_short_return"]["se"]),
        }
        gate = results["gates"]["incumbent6"].get(variant, {})
        if not gate.get("pass"):
            continue
        interval = delta["long_short_sharpe"]
        if interval["mean"] is None:
            continue
        score = interval["mean"]
        se = interval["se"] or 0.0
        candidate = {"variant": variant, "long_short_sharpe_delta": score,
                     "long_short_sharpe_se": se,
                     "long_short_sharpe_ci": [interval["low"], interval["high"]],
                     "long_short_return_delta": delta["long_short_return"]["mean"],
                     "rank_ic_delta": delta["rank_ic"]["mean"],
                     "turnover_delta": delta["turnover"]["mean"]}
        if best is None:
            best = candidate
            continue
        tolerance = max(se, best["long_short_sharpe_se"])
        gap = score - best["long_short_sharpe_delta"]
        if gap > tolerance:
            best = candidate
        elif abs(gap) <= tolerance and gate_order.get(variant, 99) < gate_order.get(best["variant"], 99):
            best = candidate
    results["winner"] = best

    # ------------------------------------------------------------ nine-blend A4 check
    if not skip_nine:
        nine_plan = json.loads((SCREEN_RUN / "plan.json").read_text(encoding="utf-8"))
        nine_windows = [window for window in nine_plan["windows"] if window["name"] in OLD_FIVE]
        nine_windows += [window for window in json.loads((VERIFY_RUN / "plan.json").read_text(encoding="utf-8"))["windows"]
                         if window["name"] in OLD_FIVE and window["name"] not in {w["name"] for w in nine_windows}]
        nine_rows: list[pd.DataFrame] = []
        for window in nine_windows:
            name = window["name"]
            source = SCREEN_RUN if name in ("nov2023", "jan2024", "mar2024") else VERIFY_RUN
            keys, targets, frames = load_window(source, name, list(NINE_MEMBERS))
            scores = arm_scores(frames, {member: 1.0 / len(NINE_MEMBERS) for member in NINE_MEMBERS})
            for variant, frame in variants_of(scores, sigma).items():
                for neutral in (False, True):
                    _, daily = evaluate_variant(frame, targets, exposure, neutralized=neutral)
                    part = daily.copy()
                    part["window"] = name
                    part["variant"] = variant
                    part["track"] = "neutralized" if neutral else "gross"
                    nine_rows.append(part)
            print(f"[nine] {name} done", flush=True)
        nine_daily = pd.concat(nine_rows, ignore_index=True)
        nine_daily["month"] = month_of(nine_daily.date)
        nine = {}
        for variant in TRANSFORMS:
            nine[variant] = {}
            for track in ("gross", "neutralized"):
                part = nine_daily[(nine_daily.variant == variant) & (nine_daily.track == track)]
                nine[variant][track] = summarize(part, neutralized=(track == "neutralized"))
            if variant == "raw":
                continue
            delta = {}
            for track in ("gross", "neutralized"):
                base = nine_daily[(nine_daily.variant == "raw") & (nine_daily.track == track)][
                    ["date", "long_short_return", "rank_ic", "turnover"]]
                cand = nine_daily[(nine_daily.variant == variant) & (nine_daily.track == track)][
                    ["date", "long_short_return", "rank_ic", "turnover"]]
                merged = base.merge(cand, on="date", suffixes=("_raw", "_cand"))
                delta[track] = {
                    "long_short_return_mean": float((merged.long_short_return_cand - merged.long_short_return_raw).mean()),
                    "rank_ic_mean": float((merged.rank_ic_cand - merged.rank_ic_raw).mean()),
                    "turnover_mean": float((merged.turnover_cand - merged.turnover_raw).mean()),
                }
            nine[variant]["delta"] = delta
        confirmations = {}
        for variant in TRANSFORMS:
            if variant == "raw":
                continue
            ten = results["deltas"]["incumbent6"][variant]["gross"]
            five = nine[variant]["delta"]["gross"]
            same_sign = (np.sign(ten["long_short_return"]["mean"]) == np.sign(five["long_short_return_mean"]))
            magnitude = (None if not five["long_short_return_mean"]
                         else ten["long_short_return"]["mean"] / five["long_short_return_mean"])
            confirmations[variant] = {"same_sign": bool(same_sign), "magnitude_ratio": magnitude,
                                      "ten_month_delta": ten["long_short_return"]["mean"],
                                      "five_month_delta": five["long_short_return_mean"]}
        results["nine_blend"] = {"windows": [window["name"] for window in nine_windows],
                                 "variants": nine, "confirmation": confirmations}

    # ---------------------------------------------------------------------- persist
    out_dir = OUT_ROOT / analysis_id
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [column for column in ("date", "window", "arm", "variant", "track", *DAILY_COLUMNS[1:])
               if column in daily]
    daily_out = daily[columns].copy()
    daily_out.to_csv(out_dir / "daily.csv", index=False)
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    results["_out_dir"] = str(out_dir.relative_to(ROOT)).replace("\\", "/")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-nine", action="store_true", help="skip the archived nine-blend confirmation")
    args = parser.parse_args()
    global RUN_ROOT
    if args.run is not None:
        RUN_ROOT = args.run.resolve()
    analysis_id = (args.out.name if args.out is not None
                   else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "Z")
    global OUT_ROOT
    if args.out is not None:
        OUT_ROOT = args.out.resolve().parent
    results = run(analysis_id, skip_nine=args.skip_nine)
    checks = results["null_checks"]
    print(f"[null] N1={checks['N1_rank_selection_invariance']['pass']} "
          f"N2={checks['N2_five_month_reproduction']['pass']} "
          f"N3={checks['N3_coverage']['pass']}", flush=True)
    for arm in results["gates"]:
        passed = [variant for variant, gate in results["gates"][arm].items() if gate["pass"]]
        print(f"[gate] {arm:<12} pass={passed}", flush=True)
    print(f"[winner] {results['winner']}", flush=True)
    print(f"[out] {results['_out_dir']}", flush=True)
    if not checks["N1_rank_selection_invariance"]["pass"]:
        print("N1 failed: rank transform changed selection; analysis invalid", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
