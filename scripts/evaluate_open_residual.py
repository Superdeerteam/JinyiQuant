"""Score the V3-open 09:30 residual-head candidate against the frozen V2-CSZ baseline.

Both weight sets are scored on exactly the archived replay keys and the official label
files of the three pre-registered development months (``nov2023``, ``jan2024``,
``mar2024``).  The script

1. re-derives the baseline daily series from the archived V2 predictions and verifies it
   against the frozen ``baseline_metrics.json`` (same code path, 1e-12 tolerance),
2. derives the candidate ``raw_full`` / ``raw_common`` / ``processed`` views with the
   frozen local style/industry residualization,
3. computes per-day 09:30 and full-day series, paired baseline->candidate deltas with the
   gate's block-5 / block-2 moving-block intervals, monthly means, worst days, full-day
   turnover / ICIR / long-short Sharpe and the six volatility/volume stress buckets, and
4. reports every frozen gate threshold next to its candidate value as *reference numbers*.

The formal seed-11 admission verdict is the reviewer's next task; this archive only
publishes the candidate evidence.  Metrics stay the local ``jinyiquant.evaluation``
approximation (gross forward label returns, no fees, no official score).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jinyiquant.io import file_sha256, read_json, utc_now, write_json  # noqa: E402

DATA_ROOT = Path("D:/进益资本杯/数据")
PLAN_PATH = ROOT / "configs" / "label_contrast_20261024_plan.json"
GATE_PATH = ROOT / "configs" / "v3_open_20261024_gate.json"
BASELINE_METRICS = ROOT / "runs" / "v3-open-20261024" / "baseline_metrics.json"
REPLAY_RUN = Path("D:/进益资本杯/runs/label-contrast-eval-20261024/20261024T171905.318633Z_ad9cc8a7400e")
OUT_ROOT = ROOT / "runs" / "v3-open-20261024"
BASELINE = "v2_60d_csz"
WINDOWS = ("nov2023", "jan2024", "mar2024")
VIEWS = ("raw_full", "raw_common", "processed")
CLOCK = "09:30"
SECONDARY_CLOCK = "10:00"
REPRODUCTION_TOLERANCE = 1e-12


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def load_contract() -> dict:
    """Plan contract with the worktree-safe absolute data paths."""
    plan = read_json(PLAN_PATH)
    plan["data"].update({
        "bars_path": str(DATA_ROOT / "cpt_jyc_2026_e2e_bar1m"),
        "labels_path": str(DATA_ROOT / "cpt_jyc_2026_vwap"),
        "instruments_path": str(DATA_ROOT / "cpt_jyc_2026_instruments(1).parquet"),
        "instrument_map_path": str(DATA_ROOT / "cpt_jyc_2026_stock_bar1m_instrument_mapping.csv"),
        "cache_dir": None,
    })
    return plan


def window_views(predictions: pd.DataFrame, targets: pd.DataFrame, evaluation, exposure_by_date: dict,
                 exposure_dates: np.ndarray, lce) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """The frozen three-view scoring recipe (identical to the baseline replay)."""
    metrics_full, daily_full, sections_full = lce.score_full(predictions, targets, evaluation)
    frame, diagnostics = lce.neutralize(predictions, exposure_by_date, exposure_dates,
                                        quantile=evaluation.quantile)
    if len(frame) < 0.99 * len(predictions):
        raise ValueError("more than 1% of keys lack a prior-day exposure")
    scored = {}
    for view, column in (("raw_common", "raw_score"), ("processed", "processed_score")):
        metrics, daily, sections, keys_sha = lce.score_common(frame, column, targets, evaluation)
        scored[view] = {"metrics": metrics, "daily": daily, "sections": sections, "keys_sha": keys_sha}
    if scored["raw_common"]["keys_sha"] != scored["processed"]["keys_sha"]:
        raise ValueError("raw and processed views use different samples")
    scored["raw_full"] = {"metrics": metrics_full, "daily": daily_full, "sections": sections_full,
                          "keys_sha": metrics_full["sample_keys_sha256"]}
    return scored, frame, diagnostics


def concat_windows(frames_by_window: dict, value: str) -> tuple[np.ndarray, list[np.ndarray]]:
    parts = [frames_by_window[name][value].to_numpy(dtype=float) for name in WINDOWS]
    return np.concatenate(parts), parts


def paired_metric(candidate_frames: dict, baseline_frames: dict, metric: str) -> pd.DataFrame:
    """Aligned per-day candidate/baseline values and their delta (missing rows dropped)."""
    frames = []
    for name in WINDOWS:
        left = candidate_frames[name][["date", metric]].rename(columns={metric: "candidate"})
        right = baseline_frames[name][["date", metric]].rename(columns={metric: "baseline"})
        joined = left.merge(right, on="date", how="inner", validate="one_to_one")
        joined["window"] = name
        frames.append(joined)
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[np.isfinite(frame["candidate"]) & np.isfinite(frame["baseline"])]
    frame["delta"] = frame["candidate"] - frame["baseline"]
    return frame.reset_index(drop=True)


def parts_of(frame: pd.DataFrame, column: str) -> list[np.ndarray]:
    return [frame.loc[frame["window"] == name, column].to_numpy(dtype=float) for name in WINDOWS]


def turnover_stats(frame: pd.DataFrame, column: str, fvg) -> dict:
    parts = [values[np.isfinite(values)] for values in parts_of(frame, column)]
    parts = [values for values in parts if len(values)]
    draws = fvg.bootstrap_draws(parts)
    values = np.concatenate(parts)
    return {"mean": float(values.mean()), "daily_std": float(values.std(ddof=1)),
            "block5": fvg.interval(draws.mean(axis=1))}


def delta_stats(frame: pd.DataFrame, fvg) -> dict:
    """Paired daily delta: mean, moving-block intervals, sign counts, worst days."""
    values, parts = concat_windows({name: frame.loc[frame["window"] == name]
                                    for name in WINDOWS}, "delta")
    draws = fvg.bootstrap_draws(parts)
    ordered = np.sort(values)
    return {
        "days": int(len(values)),
        "mean": float(values.mean()),
        "daily_std": float(values.std(ddof=1)),
        "block5": fvg.interval(draws.mean(axis=1)),
        "block2": fvg.interval(fvg.bootstrap_draws(parts, block_days=2).mean(axis=1)),
        "positive_days": int((values > 0).sum()),
        "negative_days": int((values < 0).sum()),
        "p05_daily": float(np.quantile(values, 0.05)),
        "worst_five_day_mean": float(ordered[:5].mean()),
    }


def series_block(frame: pd.DataFrame, column: str, fvg, *, annualize: bool) -> dict:
    values, parts = concat_windows({name: frame.loc[frame["window"] == name] for name in WINDOWS}, column)
    return fvg.series_stats(values, parts, annualize=annualize)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", type=Path,
                        help="Candidate prediction dir (default runs/v3-open-20261024/seed11)")
    parser.add_argument("--label", default="seed11")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    seed_dir = (args.seed_dir or OUT_ROOT / args.label).resolve()
    output = (args.output or seed_dir / "open_residual_eval.json").resolve()
    predictions_dir = seed_dir / "predictions"
    prediction_checks = read_json(seed_dir / "prediction_checks.json")
    if not prediction_checks.get("all_passed"):
        raise ValueError("candidate predictions did not pass the invariance checks")
    fvg = module(ROOT / "scripts" / "freeze_v3_open_gate.py", "freeze_v3_open_gate")
    lce = module(ROOT / "scripts" / "label_contrast_evaluation.py", "label_contrast_evaluation")
    plan = load_contract()
    gate = read_json(GATE_PATH)
    baseline_metrics = read_json(BASELINE_METRICS)
    evaluation = fvg.evaluation_config(plan)
    exposure_dates, exposure_by_date, volatility = fvg.load_exposure()
    volume = fvg.load_volume(["2023-09", "2023-10", "2023-11", "2023-12",
                              "2024-01", "2024-02", "2024-03"])
    started = time.monotonic()

    versions = {"baseline": {}, "candidate": {}}
    window_records, checks = [], []
    for window in plan["windows"]:
        name = window["name"]
        if name not in WINDOWS:
            continue
        targets, counts = lce.window_targets(plan, window)
        archived = pd.read_parquet(REPLAY_RUN / "windows" / name / BASELINE / "predictions.parquet")
        candidate = pd.read_parquet(predictions_dir / name / "predictions.parquet")
        if set(archived.columns) != {"date", "instrument", "score"} or set(candidate.columns) != {
                "date", "instrument", "score"}:
            raise ValueError(f"{name}: predictions are not plain score tables")
        if not archived[["date", "instrument"]].equals(candidate[["date", "instrument"]]):
            raise ValueError(f"{name}: candidate and baseline keys differ")
        for label, predictions in (("baseline", archived), ("candidate", candidate)):
            scored, _, _ = window_views(predictions, targets, evaluation, exposure_by_date,
                                        exposure_dates, lce)
            versions[label][name] = scored
        for view in VIEWS:
            if versions["candidate"][name][view]["metrics"]["sample_keys_sha256"] != versions[
                    "baseline"][name][view]["metrics"]["sample_keys_sha256"]:
                raise ValueError(f"{name}/{view}: candidate and baseline samples differ")
            if versions["candidate"][name][view]["metrics"]["targets_sha256"] != versions[
                    "baseline"][name][view]["metrics"]["targets_sha256"]:
                raise ValueError(f"{name}/{view}: candidate and baseline targets differ")
        window_records.append({
            "name": name, "label": window["label"],
            "start": str(pd.Timestamp(window["start"]))[:10], "end": str(pd.Timestamp(window["end"]))[:10],
            "trading_days": int(candidate["date"].dt.normalize().nunique()),
            "prediction_keys": int(len(candidate)), "target_keys": int(len(targets)),
            "targets": counts,
            "targets_sha256": versions["candidate"][name]["raw_full"]["metrics"]["targets_sha256"],
            "sample_keys_sha256": versions["candidate"][name]["raw_full"]["metrics"]["sample_keys_sha256"],
            "sections": int(len(versions["candidate"][name]["raw_full"]["sections"])),
        })
        print(json.dumps({"window": name, "keys": int(len(candidate)), "targets": int(len(targets))}),
              flush=True)

    # ---------------------------------------------------------------- daily series
    def clock_frames(label: str, view: str, clock: str) -> dict:
        return {name: fvg.daily_series(versions[label][name][view]["sections"], clock)
                for name in WINDOWS}

    pooled = {label: {} for label in versions}
    for label in versions:
        for view in VIEWS:
            frames = []
            for name in WINDOWS:
                daily = versions[label][name][view]["daily"].copy()
                daily["window"] = name
                frames.append(daily)
            pooled[label][view] = pd.concat(frames, ignore_index=True).sort_values(
                ["window", "date"]).reset_index(drop=True)

    # ------------------------------------------------- baseline self-reproduction
    archived_dispersion = baseline_metrics["nine_thirty_baseline"]
    archived_headline = baseline_metrics["headline"]["rank_ic_0930_window_mean"]
    archived_full_day = baseline_metrics["pooled_full_day"]
    archived_stress = baseline_metrics["stress_regimes"]["buckets"]
    reproduction = {}
    for view in VIEWS:
        base_0930 = clock_frames("baseline", view, CLOCK)
        rank_ic = series_block(pd.concat([base_0930[name].assign(window=name) for name in WINDOWS],
                                         ignore_index=True), "rank_ic", fvg, annualize=False)
        turnover = turnover_stats(pd.concat([base_0930[name].assign(window=name) for name in WINDOWS],
                                            ignore_index=True), "turnover", fvg)
        window_mean = float(np.mean([base_0930[name]["rank_ic"].mean() for name in WINDOWS]))
        full = pooled["baseline"][view]
        full_ic = series_block(full, "rank_ic", fvg, annualize=False)
        full_ls = series_block(full, "long_short_return", fvg, annualize=True)
        full_to = turnover_stats(full, "turnover", fvg)
        recorded = archived_dispersion[view]
        recorded_full = archived_full_day[view]
        diffs = {
            "rank_ic_0930_pooled_mean": abs(rank_ic["mean"] - recorded["rank_ic"]["mean"]),
            "rank_ic_0930_block5_se": abs(rank_ic["block5"]["se"] - recorded["rank_ic"]["block5"]["se"]),
            "turnover_0930_mean": abs(turnover["mean"] - recorded["turnover"]["mean"]),
            "full_day_rank_ic_ir": abs(full_ic["rank_ic_ir"] - recorded_full["rank_ic_full_day"]["rank_ic_ir"]),
            "full_day_long_short_sharpe": abs(
                full_ls["long_short_sharpe"] - recorded_full["long_short_full_day"]["long_short_sharpe"]),
            "full_day_long_short_mean": abs(
                full_ls["mean"] - recorded_full["long_short_full_day"]["mean"]),
            "full_day_turnover_mean": abs(full_to["mean"] - recorded_full["turnover_full_day"]["mean"]),
            "full_day_worst_day": abs(
                float(full["long_short_return"].min())
                - recorded_full["worst_day_full_day"]["long_short_return"]),
        }
        if view in archived_headline:
            diffs["rank_ic_0930_window_mean"] = abs(window_mean - archived_headline[view])
        reproduction[view] = {"max_abs_diff": max(diffs.values()), "diffs": diffs,
                              "passed": max(diffs.values()) <= REPRODUCTION_TOLERANCE}
    if not all(item["passed"] for item in reproduction.values()):
        raise ValueError("baseline recomputation does not reproduce baseline_metrics.json")

    # ------------------------------------------------------------ 09:30 candidate
    nine_thirty = {}
    for view in VIEWS:
        cand_frames = clock_frames("candidate", view, CLOCK)
        base_frames = clock_frames("baseline", view, CLOCK)
        paired = {metric: paired_metric(cand_frames, base_frames, metric)
                  for metric in ("rank_ic", "long_short_return", "turnover")}
        cand_concat = pd.concat([cand_frames[name].assign(window=name) for name in WINDOWS],
                                ignore_index=True)
        nine_thirty[view] = {
            "candidate": {
                "rank_ic": series_block(cand_concat, "rank_ic", fvg, annualize=False),
                "long_short_return": series_block(cand_concat, "long_short_return", fvg, annualize=True),
                "turnover": turnover_stats(cand_concat, "turnover", fvg),
                "monthly": {name: {"days": int(len(cand_frames[name])),
                                   "rank_ic_mean": float(cand_frames[name]["rank_ic"].mean()),
                                   "long_short_return_mean": float(
                                       cand_frames[name]["long_short_return"].mean())}
                            for name in WINDOWS},
                "worst_days": [
                    {"window": row.window, "date": str(row.date)[:10],
                     "long_short_return": float(row.long_short_return),
                     "rank_ic": float(row.rank_ic)}
                    for row in cand_concat.sort_values("long_short_return").head(5).itertuples()],
            },
            "baseline": {
                "rank_ic": series_block(pd.concat([base_frames[name].assign(window=name)
                                                   for name in WINDOWS], ignore_index=True),
                                        "rank_ic", fvg, annualize=False),
                "long_short_return": series_block(pd.concat([base_frames[name].assign(window=name)
                                                             for name in WINDOWS], ignore_index=True),
                                                  "long_short_return", fvg, annualize=True),
                "turnover": turnover_stats(pd.concat([base_frames[name].assign(window=name)
                                                      for name in WINDOWS], ignore_index=True),
                                           "turnover", fvg),
            },
            "paired_delta": {
                "rank_ic": delta_stats(paired["rank_ic"], fvg),
                "long_short_return": delta_stats(paired["long_short_return"], fvg),
                "turnover": delta_stats(paired["turnover"], fvg),
                "monthly_rank_ic_mean_delta": {
                    name: float(paired["rank_ic"].loc[paired["rank_ic"]["window"] == name, "delta"].mean())
                    for name in WINDOWS},
                "monthly_long_short_mean_delta": {
                    name: float(paired["long_short_return"].loc[
                        paired["long_short_return"]["window"] == name, "delta"].mean())
                    for name in WINDOWS},
                "monthly_turnover_mean_delta": {
                    name: float(paired["turnover"].loc[paired["turnover"]["window"] == name, "delta"].mean())
                    for name in WINDOWS},
            },
        }

    # -------------------------------------------------------------- full-day views
    full_day = {}
    for view in VIEWS:
        cand = pooled["candidate"][view]
        base = pooled["baseline"][view]
        paired = {metric: paired_metric({name: cand.loc[cand["window"] == name].reset_index(drop=True)
                                        for name in WINDOWS},
                                        {name: base.loc[base["window"] == name].reset_index(drop=True)
                                         for name in WINDOWS}, metric)
                  for metric in ("rank_ic", "long_short_return", "turnover")}
        worst = cand.loc[cand["long_short_return"].idxmin()]
        full_day[view] = {
            "candidate": {
                "rank_ic": series_block(cand, "rank_ic", fvg, annualize=False),
                "long_short_return": series_block(cand, "long_short_return", fvg, annualize=True),
                "turnover": turnover_stats(cand, "turnover", fvg),
                "worst_day": {"window": worst["window"], "date": str(worst["date"])[:10],
                              "long_short_return": float(worst["long_short_return"])},
            },
            "baseline": {
                "rank_ic": series_block(base, "rank_ic", fvg, annualize=False),
                "long_short_return": series_block(base, "long_short_return", fvg, annualize=True),
                "turnover": turnover_stats(base, "turnover", fvg),
                "worst_day": {
                    "window": base.loc[base["long_short_return"].idxmin(), "window"],
                    "date": str(base.loc[base["long_short_return"].idxmin(), "date"])[:10],
                    "long_short_return": float(base["long_short_return"].min())},
            },
            "paired_delta": {metric: delta_stats(paired[metric], fvg)
                             for metric in ("rank_ic", "long_short_return", "turnover")},
        }

    # ------------------------------------------------------------- stress buckets
    all_days = [pd.Timestamp(day) for day in sorted(pd.unique(pd.concat(
        [versions["candidate"][name]["processed"]["sections"]["date"]
         for name in WINDOWS]).dt.normalize()))]
    regime_frame = fvg.regimes_for_days(all_days, exposure_dates, volatility, volume)
    regime_frame["date"] = pd.to_datetime(regime_frame["date"])
    stress = {}
    for view in VIEWS:
        cand_concat = pd.concat([clock_frames("candidate", view, CLOCK)[name].assign(window=name)
                                 for name in WINDOWS], ignore_index=True)
        joined = cand_concat.merge(regime_frame[["date", "vol_high", "volume_high"]], on="date",
                                   how="left", validate="one_to_one")
        if joined[["vol_high", "volume_high"]].isna().any().any():
            raise ValueError("regime states are missing for some candidate days")
        conditions = {
            "high_volatility": joined["vol_high"].to_numpy(),
            "low_volatility": ~joined["vol_high"].to_numpy(),
            "high_volume": joined["volume_high"].to_numpy(),
            "low_volume": ~joined["volume_high"].to_numpy(),
            "high_volatility_high_volume": (joined["vol_high"] & joined["volume_high"]).to_numpy(),
            "high_volatility_low_volume": (joined["vol_high"] & ~joined["volume_high"]).to_numpy(),
        }
        buckets = {}
        for state, mask in conditions.items():
            part = joined.loc[mask].sort_values("date")
            ic_parts = [part.loc[part["window"] == name, "rank_ic"].to_numpy(dtype=float)
                        for name in WINDOWS if (part["window"] == name).any()]
            ls_parts = [part.loc[part["window"] == name, "long_short_return"].to_numpy(dtype=float)
                        for name in WINDOWS if (part["window"] == name).any()]
            buckets[state] = {
                "days": int(len(part)),
                "rank_ic_mean": float(part["rank_ic"].mean()),
                "rank_ic_se": fvg.histogram_se(ic_parts),
                "long_short_return_mean": float(part["long_short_return"].mean()),
                "long_short_se": fvg.histogram_se(ls_parts),
                "months": {name: int((part["window"] == name).sum()) for name in WINDOWS},
                "baseline_rank_ic_mean": float(archived_stress[view][state]["rank_ic_mean"]),
                "baseline_long_short_return_mean": float(
                    archived_stress[view][state]["long_short_return_mean"]),
            }
        stress[view] = buckets

    # ------------------------------------------------------ frozen gate reference
    thresholds = gate["thresholds"]
    primary = thresholds["primary_0930_rank_ic"]
    guardrails = thresholds["degradation_guardrails"]
    stress_cfg = thresholds["stress_scenarios"]["buckets"]
    months_required = next(item["threshold"] for item in primary["checks"]
                           if item["id"] == "months_positive")
    ten_oclock = {}
    for view in VIEWS:
        parts = [clock_frames("candidate", view, SECONDARY_CLOCK)[name]["rank_ic"].to_numpy(dtype=float)
                 for name in WINDOWS]
        base_parts = [clock_frames("baseline", view, SECONDARY_CLOCK)[name]["rank_ic"].to_numpy(dtype=float)
                      for name in WINDOWS]
        paired = paired_metric(clock_frames("candidate", view, SECONDARY_CLOCK),
                               clock_frames("baseline", view, SECONDARY_CLOCK), "rank_ic")
        ten_oclock[view] = {"candidate_window_mean": float(np.mean([part.mean() for part in parts])),
                            "baseline_window_mean": float(np.mean([part.mean() for part in base_parts])),
                            "paired_delta_mean": float(paired["delta"].mean())}
    month_deltas = nine_thirty["processed"]["paired_delta"]["monthly_rank_ic_mean_delta"]
    cand_0930 = nine_thirty["processed"]
    cand_full = full_day["processed"]
    gate_reference = {
        "note": "numeric reference against the frozen gate only; the formal seed-11 admission "
                "verdict belongs to the next review task, not to this archive",
        "primary_0930_rank_ic_processed": {
            "delta_mean": cand_0930["paired_delta"]["rank_ic"]["mean"],
            "minimum_gain": primary["minimum_gain"],
            "mean_delta_at_least_minimum_gain": bool(
                cand_0930["paired_delta"]["rank_ic"]["mean"] > 0
                and cand_0930["paired_delta"]["rank_ic"]["mean"] >= primary["minimum_gain"]),
            "block5_ci_low": cand_0930["paired_delta"]["rank_ic"]["block5"]["ci_low"],
            "block5_lower_positive": bool(cand_0930["paired_delta"]["rank_ic"]["block5"]["ci_low"] > 0),
            "block2_ci_low": cand_0930["paired_delta"]["rank_ic"]["block2"]["ci_low"],
            "block2_lower_positive": bool(cand_0930["paired_delta"]["rank_ic"]["block2"]["ci_low"] > 0),
            "month_deltas": month_deltas,
            "months_positive": int(sum(value > 0 for value in month_deltas.values())),
            "months_required": int(months_required),
        },
        "raw_view_consistency": {
            "view": "raw_full",
            "delta_mean": nine_thirty["raw_full"]["paired_delta"]["rank_ic"]["mean"],
            "threshold": primary["raw_view_consistency"]["threshold"],
            "meets": bool(nine_thirty["raw_full"]["paired_delta"]["rank_ic"]["mean"]
                          >= primary["raw_view_consistency"]["threshold"]),
        },
        "monthly_non_degradation": {
            name: {"delta_mean": month_deltas[name],
                   "floor": thresholds["monthly_non_degradation"]["floors"][name],
                   "meets": bool(month_deltas[name]
                                 >= thresholds["monthly_non_degradation"]["floors"][name])}
            for name in WINDOWS
        },
        "degradation_guardrails": {
            "turnover_full_day_processed": {
                "candidate": cand_full["candidate"]["turnover"]["mean"],
                "threshold": guardrails["turnover"]["threshold"],
                "baseline": guardrails["turnover"]["baseline"],
                "meets": bool(cand_full["candidate"]["turnover"]["mean"]
                              <= guardrails["turnover"]["threshold"])},
            "rank_ic_ir_full_day_processed": {
                "candidate": cand_full["candidate"]["rank_ic"]["rank_ic_ir"],
                "threshold": guardrails["rank_ic_ir"]["threshold"],
                "baseline": guardrails["rank_ic_ir"]["baseline"],
                "meets": bool(cand_full["candidate"]["rank_ic"]["rank_ic_ir"]
                              >= guardrails["rank_ic_ir"]["threshold"])},
            "long_short_mean_full_day_processed": {
                "candidate": cand_full["candidate"]["long_short_return"]["mean"],
                "threshold": guardrails["long_short_mean"]["threshold"],
                "baseline": guardrails["long_short_mean"]["baseline"],
                "meets": bool(cand_full["candidate"]["long_short_return"]["mean"]
                              >= guardrails["long_short_mean"]["threshold"])},
            "long_short_sharpe_full_day_processed": {
                "candidate": cand_full["candidate"]["long_short_return"]["long_short_sharpe"],
                "threshold": guardrails["long_short_sharpe"]["threshold"],
                "baseline": guardrails["long_short_sharpe"]["baseline"],
                "meets": bool(cand_full["candidate"]["long_short_return"]["long_short_sharpe"]
                              >= guardrails["long_short_sharpe"]["threshold"])},
            "worst_day_full_day_processed": {
                "candidate": cand_full["candidate"]["worst_day"]["long_short_return"],
                "candidate_date": cand_full["candidate"]["worst_day"]["date"],
                "threshold": guardrails["worst_day"]["threshold"],
                "baseline": guardrails["worst_day"]["baseline"],
                "meets": bool(cand_full["candidate"]["worst_day"]["long_short_return"]
                              >= guardrails["worst_day"]["threshold"])},
            "rank_ic_1000_raw_full": {
                "candidate_window_mean": ten_oclock["raw_full"]["candidate_window_mean"],
                "baseline_window_mean": ten_oclock["raw_full"]["baseline_window_mean"],
                "paired_delta_mean": ten_oclock["raw_full"]["paired_delta_mean"],
                "threshold": guardrails["rank_ic_1000_raw_full"]["threshold"],
                "meets": bool(ten_oclock["raw_full"]["paired_delta_mean"]
                              >= guardrails["rank_ic_1000_raw_full"]["threshold"])},
        },
        "stress_scenarios_processed": {
            state: {"days": stress["processed"][state]["days"],
                    "rank_ic_mean": stress["processed"][state]["rank_ic_mean"],
                    "rank_ic_floor": stress_cfg[state]["rank_ic_floor"],
                    "rank_ic_meets": bool(stress["processed"][state]["rank_ic_mean"]
                                          >= stress_cfg[state]["rank_ic_floor"]),
                    "long_short_return_mean": stress["processed"][state]["long_short_return_mean"],
                    "long_short_floor": stress_cfg[state]["long_short_floor"],
                    "long_short_meets": bool(stress["processed"][state]["long_short_return_mean"]
                                             >= stress_cfg[state]["long_short_floor"])}
            for state in stress_cfg},
    }

    # --------------------------------------------------------------- daily diff file
    diff_frames = []
    for name in WINDOWS:
        for view in VIEWS:
            for clock in plan["analysis"]["timestamps"]:
                cand = fvg.daily_series(versions["candidate"][name][view]["sections"], clock)
                base = fvg.daily_series(versions["baseline"][name][view]["sections"], clock)
                merged = cand.merge(base, on="date", suffixes=("_candidate", "_baseline"),
                                    validate="one_to_one")
                for metric in ("rank_ic", "long_short_return", "turnover"):
                    merged[f"{metric}_delta"] = merged[f"{metric}_candidate"] - merged[f"{metric}_baseline"]
                merged["window"] = name
                merged["clock"] = clock
                merged["view"] = view
                diff_frames.append(merged)
    daily_diffs = pd.concat(diff_frames, ignore_index=True)
    ordered = ["window", "date", "clock", "view", "rank_ic_candidate", "rank_ic_baseline", "rank_ic_delta",
               "long_short_return_candidate", "long_short_return_baseline", "long_short_return_delta",
               "turnover_candidate", "turnover_baseline", "turnover_delta"]
    daily_diffs = daily_diffs[ordered].sort_values(["window", "view", "clock", "date"]).reset_index(drop=True)
    daily_diffs_path = seed_dir / "daily_diffs.parquet"
    daily_diffs.to_parquet(daily_diffs_path, index=False)

    full_day_rows = []
    for label in ("baseline", "candidate"):
        for view in VIEWS:
            frame = pooled[label][view].copy()
            frame["view"] = view
            frame["label"] = label
            full_day_rows.append(frame[["window", "date", "view", "label", "rank_ic",
                                        "long_short_return", "turnover"]])
    full_day_table = pd.concat(full_day_rows, ignore_index=True)
    full_day_path = seed_dir / "daily_full_day.parquet"
    full_day_table.to_parquet(full_day_path, index=False)

    expected_rows = len(VIEWS) * len(plan["analysis"]["timestamps"]) * sum(
        record["trading_days"] for record in window_records)
    checks = [
        {"id": "prediction_invariance_archive", "passed": bool(prediction_checks["all_passed"])},
        {"id": "baseline_metrics_reproduced", "passed": all(item["passed"] for item in reproduction.values()),
         "max_abs_diff": max(item["max_abs_diff"] for item in reproduction.values())},
        {"id": "candidate_and_baseline_samples_identical", "passed": True,
         "windows": [record["name"] for record in window_records]},
        {"id": "daily_diffs_complete", "passed": bool(len(daily_diffs) == expected_rows),
         "rows": int(len(daily_diffs)), "expected_rows": int(expected_rows)},
    ]
    evidence = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "experiment_id": "v3-open-20261024",
        "stage": f"{args.label}_open_residual_evaluation",
        "label": args.label,
        "candidate": prediction_checks["candidate"],
        "frozen_baseline": {"version": BASELINE, **prediction_checks["frozen_baseline"]},
        "inputs": {
            "predictions_dir": str(predictions_dir),
            "prediction_checks": str(seed_dir / "prediction_checks.json"),
            "prediction_checks_sha256": file_sha256(seed_dir / "prediction_checks.json"),
            "gate": str(GATE_PATH), "gate_sha256": file_sha256(GATE_PATH),
            "baseline_metrics": str(BASELINE_METRICS),
            "baseline_metrics_sha256": file_sha256(BASELINE_METRICS),
            "replay_run": str(REPLAY_RUN),
            "plan": str(PLAN_PATH), "plan_sha256": file_sha256(PLAN_PATH),
        },
        "conventions": {
            "metric_scope": "local_approximation",
            "views": list(VIEWS),
            "daily": "equal mean over the eight sampled timestamps",
            "paired_delta": "candidate minus V2-CSZ baseline on the same date, window and clock",
            "block_bootstrap": {"block_days": [5, 2], "samples": fvg.SAMPLES,
                                "seed": fvg.BOOTSTRAP_SEED, "confidence": fvg.CONFIDENCE,
                                "rule": "contiguous moving blocks inside each month; pooled draws "
                                        "weighted by days"},
            "turnover": "0.5 * sum(abs(net weight_t - net weight_t-1)); the first sampled "
                        "timestamp of each window stays null",
            "not_used": "final/locked check ranges; no threshold was edited",
        },
        "reproduction": reproduction,
        "windows": window_records,
        "per_clock": {view: lce.by_time(pd.concat([versions["candidate"][name][view]["sections"]
                                                   for name in WINDOWS], ignore_index=True))
                      for view in VIEWS},
        "baseline_per_clock": {view: lce.by_time(pd.concat([versions["baseline"][name][view]["sections"]
                                                            for name in WINDOWS], ignore_index=True))
                               for view in VIEWS},
        "nine_thirty": nine_thirty,
        "full_day": full_day,
        "ten_oclock": ten_oclock,
        "stress": stress,
        "gate_reference": gate_reference,
        "daily_diffs": {"path": str(daily_diffs_path), "rows": int(len(daily_diffs)),
                        "sha256": file_sha256(daily_diffs_path)},
        "daily_full_day": {"path": str(full_day_path), "rows": int(len(full_day_table)),
                           "sha256": file_sha256(full_day_path)},
        "checks": checks,
        "all_evidence_checks_passed": all(item["passed"] for item in checks),
        "resources": {"wall_seconds": float(time.monotonic() - started)},
    }
    write_json(output, evidence)
    if not evidence["all_evidence_checks_passed"]:
        failed = [item for item in checks if not item["passed"]]
        raise ValueError(f"evaluation evidence checks failed: {failed}")
    print(json.dumps({"exit_code": 0, "output": str(output),
                      "rank_ic_0930_processed_mean": nine_thirty["processed"]["candidate"]["rank_ic"]["mean"],
                      "rank_ic_0930_processed_delta": nine_thirty["processed"]["paired_delta"]["rank_ic"]["mean"],
                      "full_day_turnover": full_day["processed"]["candidate"]["turnover"]["mean"],
                      "daily_diffs_rows": int(len(daily_diffs))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
