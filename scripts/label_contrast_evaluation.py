"""Compare the V2 RAW / CENTER label arms with the frozen V2-CSZ baseline.

The replay run (``scripts/replay_model_versions.py run``) scores the three frozen
weight sets on identical keys and raw official labels.  This entry point reuses
that run, reproduces the archived raw-score metrics, replays the frozen local
style-residualization convention of ``scripts/diagnose_risk_neutralization.py``
(previous-day exposure snapshot, median +/- 3 * 1.4826 * MAD winsorization,
section z-score, equal-weight OLS on 10 styles + 32 industries) and writes one
structured record.  The decision framework is pre-registered in the plan file
before the metrics of this round are computed.

No model is trained or refitted here; label transforms are never inverted and no
score is re-centred.  All metrics stay the local ``jinyiquant.evaluation``
approximation: gross forward label returns, no fees, no execution, no official
score.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jinyiquant.data import select_targets  # noqa: E402
from jinyiquant.evaluation import METRIC_SPEC, aggregate_metric, evaluate  # noqa: E402
from jinyiquant.io import digest, file_sha256, read_json, utc_now, write_json  # noqa: E402
from modified_tlob.ensemble import paired_block_intervals  # noqa: E402
from modified_tlob.risk import (  # noqa: E402
    INDUSTRY_DUMMIES,
    REGRESSORS,
    STYLE_FACTORS,
    previous_exposure_date,
    residualize,
    section_panel,
    winsorized_zscore,
)

KEYS = ["date", "instrument"]
VARIANTS = ("raw_full", "raw_common", "processed")
DELTA_METRICS = ("rank_ic", "long_short_return", "turnover")
REPORTED = ("rank_ic_mean", "rank_ic_ir", "ic_mean", "long_short_return_mean",
            "long_short_sharpe", "long_return_mean", "short_return_mean",
            "turnover_mean", "extreme_return", "min_daily_long_short_return")
POOLED_COLUMNS = ("rank_ic", "ic", "long_return", "short_return", "long_short_return", "turnover")


def resolve(path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def json_float(value):
    return None if value is None or not np.isfinite(value) else float(value)


def month_overlaps(name: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """Same month-file selection rule as ``jinyiquant.data._month_overlaps``."""
    match = re.search(r"(?<!\d)(20\d{2})[-_]?([01]\d)(?!\d)", Path(name).name)
    if not match:
        return True
    month = pd.Timestamp(f"{match[1]}-{match[2]}-01")
    next_month = month + pd.offsets.MonthBegin(1)
    return month < end and next_month > start


# ------------------------------------------------------------------ plan and inputs


def load_plan(path: Path) -> dict:
    plan = read_json(path)
    required = {"schema_version", "experiment_id", "data", "evaluation", "windows", "versions",
                "decision_framework", "statistics", "analysis"}
    missing = required - set(plan)
    if missing:
        raise ValueError(f"plan is missing {sorted(missing)}")
    if plan["experiment_id"] != "label-contrast-20261024":
        raise ValueError("plan experiment_id does not match this entry point")
    plan["analysis"]["archive_path"] = str((path.parent / plan["analysis"]["archive_path"]).resolve())
    for name in ("bars_path", "labels_path", "instruments_path", "instrument_map_path", "cache_dir"):
        if plan["data"].get(name):
            plan["data"][name] = str((path.parent / plan["data"][name]).resolve())
    for version in plan["versions"]:
        version["weights"] = str((path.parent / version["weights"]).resolve())
    if plan["evaluation"].get("constant_portfolios") != "undefined":
        raise ValueError("label contrast requires undefined constant portfolios")
    if tuple(plan["statistics"]["block_days"]) != (5, 2):
        raise ValueError("plan block lengths differ from the frozen paired convention")
    if plan["analysis"]["timestamps"] != ["09:30", "10:00", "10:30", "11:00", "11:30",
                                          "13:30", "14:00", "14:30"]:
        raise ValueError("plan timestamp grid differs from the frozen eight sections")
    return plan


def window_targets(plan: dict, window: dict) -> tuple[pd.DataFrame, dict]:
    """Reconstruct exactly the replay target set: official labels, universe, purge."""
    start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
    data = plan["data"]
    label_dir = resolve(data["labels_path"])
    files = [file for file in sorted(label_dir.rglob("*"))
             if file.is_file() and file.suffix.lower() == ".parquet"
             and month_overlaps(file.name, start, end)]
    if not files:
        raise ValueError(f"no label files overlap {window['name']}")
    frames = []
    for file in files:
        frame = pd.read_parquet(file, columns=["date", "instrument", "vwap_return"])
        frame["date"] = pd.to_datetime(frame["date"])
        frames.append(frame.loc[(frame["date"] >= start) & (frame["date"] < end)])
    labels = pd.concat(frames, ignore_index=True)
    if labels.duplicated(KEYS).any():
        raise ValueError(f"{window['name']}: duplicate label keys")
    instruments = pd.read_parquet(resolve(data["instruments_path"]), columns=KEYS)
    instruments["date"] = pd.to_datetime(instruments["date"])
    universe = instruments.loc[(instruments["date"] >= start)
                               & (instruments["date"] < end), KEYS].drop_duplicates()
    merged = labels.merge(universe.assign(_member=True), on=KEYS, how="left", validate="one_to_one")
    outside = int(merged["_member"].isna().sum())
    labels = merged.loc[merged["_member"].notna(), ["date", "instrument", "vwap_return"]]
    labels = labels.rename(columns={"vwap_return": "target"}).reset_index(drop=True)
    targets = select_targets(labels, {"start": start, "end": end},
                             int(data["label_horizon_minutes"]),
                             sample_times=list(data["sample_times"]))
    counts = {"label_rows": int(len(labels)), "outside_universe_rows": outside,
              "selected_targets": int(len(targets)),
              "label_files": sorted(file.name for file in files)}
    return targets[KEYS + ["target"]].reset_index(drop=True), counts


def load_exposure(plan: dict) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Read every exposure snapshot needed by the replayed windows, with frozen checks."""
    paths = [resolve("数据/cpt_jyc_2026_exposure/cpt_jyc_2026_exposure_2023_part_0001.parquet"),
             resolve("数据/cpt_jyc_2026_exposure/cpt_jyc_2026_exposure_2024_part_0001.parquet")]
    columns = ["date", "instrument", *STYLE_FACTORS, *INDUSTRY_DUMMIES]
    frame = pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if frame.duplicated(KEYS).any():
        raise ValueError("exposure table contains duplicate date/instrument keys")
    numeric = frame[[*STYLE_FACTORS, *INDUSTRY_DUMMIES]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("exposure table contains undefined regressors")
    if not frame[list(INDUSTRY_DUMMIES)].isin([0, 1]).all().all():
        raise ValueError("industry dummies must be binary")
    if not (frame[list(INDUSTRY_DUMMIES)].sum(axis=1) == 1).all():
        raise ValueError("industry dummies must sum to one per stock")
    dates = np.array(sorted(frame["date"].unique()), dtype="datetime64[ns]")
    by_date = {date: group for date, group in frame.groupby("date", sort=True)}
    return frame, dates, by_date


def load_replay(run_dir: Path, plan: dict) -> tuple[dict, dict]:
    summary = read_json(run_dir / "summary_metrics.json")
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError("replay run is not completed")
    if manifest.get("experiment_id") != plan["experiment_id"]:
        raise ValueError("replay run experiment_id differs from the plan")
    recorded = {(entry["window"], entry["version"]): entry for entry in summary["responses"]}
    return recorded, manifest


# ------------------------------------------------------------------ scoring helpers


def evaluation_config(plan: dict):
    from jinyiquant.config import EvaluationConfig
    block = plan["evaluation"]
    return EvaluationConfig(quantile=float(block["quantile"]),
                            periods_per_year=int(block["periods_per_year"]),
                            min_cross_section=int(block["min_cross_section"]),
                            constant_portfolios="undefined")


def metric_block(metrics: dict, daily: pd.DataFrame) -> dict:
    result = {metric: json_float(metrics[metric]) for metric in REPORTED}
    ir_values = daily["ic"].to_numpy(dtype=float)
    ir_values = ir_values[np.isfinite(ir_values)]
    result["ic_ir"] = (json_float(float(ir_values.mean() / ir_values.std(ddof=1)))
                       if len(ir_values) > 1 and ir_values.std(ddof=1) > 0 else None)
    result["day_count"] = int(metrics["day_count"])
    result["cross_section_count"] = int(metrics["cross_section_count"])
    result["sample_count"] = int(metrics["sample_count"])
    result["undefined_rank_ic_count"] = int(metrics["undefined_rank_ic_count"])
    series = daily.set_index("date")["long_short_return"].dropna()
    result["worst_day"] = None if series.empty else {
        "date": str(series.idxmin())[:10], "long_short_return": float(series.min())}
    return result


def pooled_metrics(daily: pd.DataFrame) -> dict:
    """Aggregate the pooled daily series with the frozen local metric conventions."""
    result = {name: json_float(aggregate_metric(daily[column].to_numpy(), method, 252.0))
              for name, (column, method, _) in METRIC_SPEC.items()}
    ic_values = daily["ic"].to_numpy(dtype=float)
    ic_values = ic_values[np.isfinite(ic_values)]
    result["ic_ir"] = (json_float(float(ic_values.mean() / ic_values.std(ddof=1)))
                       if len(ic_values) > 1 and ic_values.std(ddof=1) > 0 else None)
    result["day_count"] = int(len(daily))
    result["sample_count"] = int(daily["sample_count"].sum())
    series = daily.set_index("date")["long_short_return"].dropna()
    result["worst_day"] = None if series.empty else {
        "date": str(series.idxmin())[:10], "long_short_return": float(series.min())}
    return result


def by_time(sections: pd.DataFrame) -> list[dict]:
    frame = sections.assign(clock=pd.to_datetime(sections["timestamp"]).dt.strftime("%H:%M"))
    rows = []
    for clock, group in frame.groupby("clock", sort=True):
        rows.append({"time": clock, "sections": int(len(group)),
                     "days": int(group["date"].nunique()),
                     "rank_ic_mean": json_float(group["rank_ic"].mean()),
                     "ic_mean": json_float(group["ic"].mean()),
                     "long_short_return_mean": json_float(group["long_short_return"].mean()),
                     "turnover_mean": json_float(group["turnover"].mean())})
    return rows


def score_full(predictions: pd.DataFrame, targets: pd.DataFrame, evaluation) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Same path as ``replay_model_versions.evaluate_version``."""
    scored = predictions.merge(targets[KEYS], on=KEYS, how="inner", validate="one_to_one")
    if scored.empty:
        raise ValueError("no scored predictions after key alignment")
    metrics, daily, sections = evaluate(scored, targets, evaluation, return_sections=True,
                                        undefined_constant_portfolios=True)
    return metrics, daily, sections


def neutralize(predictions: pd.DataFrame, exposure_by_date: dict, exposure_dates: np.ndarray, *,
               quantile: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen style/industry residualization, previous-day snapshot per timestamp."""
    frames, diagnostics = [], []
    for timestamp, section in predictions.groupby("date", sort=True):
        signal_date = pd.Timestamp(timestamp).normalize()
        snapshot_date = previous_exposure_date(exposure_dates, signal_date)
        if not snapshot_date < signal_date:
            raise ValueError(f"{timestamp}: exposure snapshot is not strictly prior")
        panel = section_panel(section["score"], section["instrument"],
                              exposure_by_date[snapshot_date])
        standardized, clipped = winsorized_zscore(panel["score"].to_numpy(dtype=float))
        design = panel[list(REGRESSORS)].to_numpy(dtype=float)
        residual, explained = residualize(standardized, design)
        scored = panel[["instrument"]].copy()
        scored["date"] = timestamp
        scored["raw_score"] = panel["score"].to_numpy(dtype=float)
        scored["processed_score"] = residual
        frames.append(scored)
        diagnostics.append({"date": signal_date, "timestamp": timestamp,
                            "snapshot_date": snapshot_date, "n_fit": int(len(panel)),
                            "clipped": int(clipped), "style_explained": float(explained),
                            "design_sha256": digest({"design": design.round(12).tolist()})})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(diagnostics)


def score_common(frame: pd.DataFrame, column: str, targets: pd.DataFrame, evaluation):
    """Score one variant on prediction ∩ target keys shared by every variant."""
    common = frame[KEYS].merge(targets[KEYS], on=KEYS, validate="one_to_one")
    scored = frame.merge(common, on=KEYS, how="inner", validate="one_to_one")
    answer = targets.merge(common, on=KEYS, validate="one_to_one")
    if len(scored) != len(answer) or len(scored) != len(common):
        raise ValueError("common sample construction changed the key set")
    table = scored[KEYS].assign(score=scored[column].to_numpy(dtype=float))
    metrics, daily, sections = evaluate(table, answer, evaluation, return_sections=True,
                                        undefined_constant_portfolios=True)
    ordered = answer.sort_values(KEYS)
    keys_sha = digest(ordered.assign(date=ordered["date"].astype(str))[KEYS].to_dict("records"))
    return metrics, daily, sections, keys_sha


# ------------------------------------------------------------------ paired statistics


def paired_block(plan: dict, daily_by: dict, left: str, right: str, variant: str) -> dict:
    """Per-day paired deltas; the block intervals pool only the pre-registered dev months."""
    frames = {}
    for label, version in (("left", left), ("right", right)):
        frame = daily_by[(version, variant)].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frames[label] = frame.set_index(["window", "date"])[list(DELTA_METRICS)]
    table = (frames["left"] - frames["right"]).reset_index().rename(columns={"window": "period"})
    if table.duplicated(["period", "date"]).any():
        raise ValueError("paired daily table has duplicate keys")
    if not np.isfinite(table[list(DELTA_METRICS)].to_numpy(dtype=float)).all():
        raise ValueError(f"paired daily table of {left}-{right} {variant} has undefined values")
    months = list(plan["analysis"]["development_months"])
    decision_sample = table.loc[table["period"].isin(months)].reset_index(drop=True)
    statistics = plan["statistics"]
    blocks = {f"block_{days}": paired_block_intervals(
        decision_sample, block_days=int(days), samples=int(statistics["bootstrap_samples"]),
        seed=int(statistics["bootstrap_seed"])) for days in statistics["block_days"]}
    rows = {}
    for name, result in blocks.items():
        rows[name] = {}
        for row in result["rows"]:
            rows[name].setdefault(row["period"], {})[row["metric"]] = {
                "days": int(row["days"]), "mean": json_float(row["mean"]),
                "ci_low": json_float(row["interval"][0]), "ci_high": json_float(row["interval"][1])}
    per_month = table.groupby("period")[list(DELTA_METRICS)].mean()
    per_month = {period: {metric: json_float(row[metric]) for metric in DELTA_METRICS}
                 for period, row in per_month.iterrows()}
    counts = {}
    for metric in DELTA_METRICS:
        values = decision_sample[metric]
        counts[metric] = {
            "positive_days": int((values > 0).sum()), "negative_days": int((values < 0).sum()),
            "zero_days": int((values == 0).sum()),
            "worst_day": str(decision_sample.loc[values.idxmin(), "date"])[:10],
            "worst_delta": json_float(values.min())}
    return {"pair": f"{left}-minus-{right}", "variant": variant, "blocks": rows,
            "per_month_mean_delta": per_month, "day_counts": counts,
            "pooled": {metric: json_float(decision_sample[metric].mean()) for metric in DELTA_METRICS},
            "decision_months": months,
            "daily": [{"window": row.period, "date": str(row.date)[:10],
                       "rank_ic": json_float(row.rank_ic),
                       "long_short_return": json_float(row.long_short_return),
                       "turnover": json_float(row.turnover)}
                      for row in table.itertuples()]}


def decide(plan: dict, pairs: dict, pooled: dict) -> dict:
    """Apply the pre-registered decision framework; never re-select after seeing metrics."""
    framework = plan["decision_framework"]
    baseline = "v2_60d_csz"
    decision = {"baseline": baseline, "framework": framework,
                "candidates": {}, "rank_ic_primary_variant": "raw_full"}
    for candidate in ("v2_raw", "v2_center"):
        key = f"{candidate}-minus-{baseline}"
        primary = pairs[(key, "raw_full")]
        entry = {"pair": key, "variant": "raw_full",
                 "blocks": primary["blocks"],
                 "per_month_mean_delta": primary["per_month_mean_delta"],
                 "pooled_mean_delta": primary["pooled"],
                 "day_counts": primary["day_counts"],
                 "checks": {}, "processed_view": {}}
        block5 = entry["blocks"]["block_5"]["all"]["rank_ic"]
        block2 = entry["blocks"]["block_2"]["all"]["rank_ic"]
        months = [entry["per_month_mean_delta"][name]["rank_ic"]
                  for name in plan["analysis"]["development_months"]]
        improvement = {"mean_positive": bool(block5["mean"] > 0),
                       "block5_lower_positive": bool(block5["ci_low"] > 0),
                       "block2_lower_positive": bool(block2["ci_low"] > 0),
                       "positive_month_count": int(sum(1 for value in months if value > 0)),
                       "two_of_three_months_positive": bool(sum(1 for value in months if value > 0) >= 2)}
        long_short_ci_high = entry["blocks"]["block_5"]["all"]["long_short_return"]["ci_high"]
        long_short_degraded = bool(entry["pooled_mean_delta"]["long_short_return"] < 0
                                   and long_short_ci_high < 0)
        candidate_turnover = pooled[(candidate, "raw_full")]["turnover_mean"]
        baseline_turnover = pooled[(baseline, "raw_full")]["turnover_mean"]
        turnover_increase = (candidate_turnover / baseline_turnover - 1.0
                             if baseline_turnover else None)
        turnover_degraded = bool(turnover_increase is not None and turnover_increase > 0.10)
        candidate_worst = pooled[(candidate, "raw_full")]["worst_day"]
        baseline_worst = pooled[(baseline, "raw_full")]["worst_day"]
        worst_ratio = ((candidate_worst["long_short_return"] - baseline_worst["long_short_return"])
                       / abs(baseline_worst["long_short_return"])
                       if baseline_worst["long_short_return"] else None)
        worst_degraded = bool(worst_ratio is not None and worst_ratio < -0.25)
        entry["checks"] = {
            "rank_ic_improvement": improvement,
            "guardrails": {
                "long_short": {"pooled_mean_delta": entry["pooled_mean_delta"]["long_short_return"],
                               "block5_ci_high": long_short_ci_high,
                               "material_degradation": long_short_degraded},
                "turnover": {"candidate_mean": candidate_turnover,
                             "baseline_mean": baseline_turnover,
                             "relative_increase": json_float(turnover_increase),
                             "material_degradation": turnover_degraded},
                "worst_day": {"candidate": candidate_worst, "baseline": baseline_worst,
                              "relative_delta": json_float(worst_ratio),
                              "material_degradation": worst_degraded}}}
        improved = all(improvement[name] for name in
                       ("mean_positive", "block5_lower_positive", "block2_lower_positive",
                        "two_of_three_months_positive"))
        degraded = long_short_degraded or turnover_degraded or worst_degraded
        if improved and not degraded:
            verdict = "better_than_csz"
        elif improved and degraded:
            verdict = "guardrail_degraded"
        else:
            verdict = "not_better"
        seed29 = bool(verdict == "better_than_csz"
                      or (block5["mean"] > 0 and not degraded and block5["ci_low"] > -0.005))
        entry["verdict"] = verdict
        entry["seed29_followup"] = seed29
        processed = pairs[(key, "processed")]
        entry["processed_view"] = {
            "pooled_mean_rank_ic_delta": processed["pooled"]["rank_ic"],
            "block5_all": processed["blocks"]["block_5"]["all"],
            "block2_all": processed["blocks"]["block_2"]["all"],
            "per_month_mean_delta": {name: processed["per_month_mean_delta"][name]["rank_ic"]
                                     for name in plan["analysis"]["development_months"]}}
        decision["candidates"][candidate] = entry
    candidates = decision["candidates"]
    trigger = [name for name in candidates if candidates[name]["seed29_followup"]]
    decision["seed29_recommendation"] = {
        "run_seed29_for": trigger,
        "reason": ("pre-registered trigger met" if trigger
                   else "no candidate shows a positive RankIC point estimate with intact guardrails"),
        "keep_csz": [name for name in candidates
                     if candidates[name]["verdict"] == "not_better"]}
    return decision


# ------------------------------------------------------------------ archive


def build_archive(plan_path: Path, replay_dir: Path, output: Path) -> dict:
    started = time.monotonic()
    plan = load_plan(plan_path)
    evaluation = evaluation_config(plan)
    recorded, manifest = load_replay(replay_dir, plan)
    exposure, exposure_dates, exposure_by_date = load_exposure(plan)
    del exposure
    versions = {version["name"]: version for version in plan["versions"]}
    baseline = "v2_60d_csz"
    candidates = [version["name"] for version in plan["versions"] if version["role"] == "candidate"]

    windows_out, results, eight_time_points, style_out = [], [], [], {}
    daily_by, coverage_out, reproduction = {}, {}, {}
    label_files = set()
    for window in plan["windows"]:
        name = window["name"]
        targets, counts = window_targets(plan, window)
        label_files.update(counts["label_files"])
        window_dir = replay_dir / "windows" / name
        keys = pd.read_parquet(window_dir / "keys.parquet")
        days = sorted(pd.unique(keys["date"].dt.normalize()))
        windows_out.append({"name": name, "label": window["label"],
                            "start": str(pd.Timestamp(window["start"]))[:10],
                            "end": str(pd.Timestamp(window["end"]))[:10],
                            "trading_days": int(len(days)),
                            "prediction_keys": int(len(keys)),
                            "target_keys": int(len(targets)),
                            "targets": counts,
                            "used_for_decision": name in plan["analysis"]["development_months"]})
        window_reproduction = {}
        window_designs = {}
        for version in plan["versions"]:
            vname = version["name"]
            predictions = pd.read_parquet(window_dir / vname / "predictions.parquet")
            if set(predictions.columns) != {*KEYS, "score"} or predictions.empty:
                raise ValueError(f"{name}/{vname}: not a score table")
            if predictions.duplicated(KEYS).any():
                raise ValueError(f"{name}/{vname}: duplicate prediction keys")
            archived = recorded[(name, vname)]

            metrics_full, daily_full, sections_full = score_full(predictions, targets, evaluation)
            diffs = []
            for metric in REPORTED:
                left, right = metrics_full[metric], archived[metric]
                if (left is None) != (right is None):
                    raise ValueError(f"{name}/{vname}: undefined metric mismatch for {metric}")
                if left is not None:
                    diffs.append(abs(left - right))
            diff = max(diffs) if diffs else 0.0
            if diff > 1e-12:
                raise ValueError(f"{name}/{vname}: raw metrics do not reproduce the replay ({diff})")
            if metrics_full["sample_keys_sha256"] != archived["sample_keys_sha256"]:
                raise ValueError(f"{name}/{vname}: sample keys differ from the replay")
            if metrics_full["targets_sha256"] != archived["targets_sha256"]:
                raise ValueError(f"{name}/{vname}: reconstructed targets differ from the replay")
            if len(targets) != archived["sample_count"]:
                raise ValueError(f"{name}/{vname}: target count differs from the replay")
            window_reproduction[vname] = {"max_metric_abs_diff": float(diff),
                                          "sample_keys_sha256": metrics_full["sample_keys_sha256"],
                                          "targets_sha256": metrics_full["targets_sha256"]}

            frame, diagnostics = neutralize(predictions, exposure_by_date, exposure_dates,
                                            quantile=evaluation.quantile)
            if len(frame) < 0.99 * len(predictions):
                raise ValueError(f"{name}/{vname}: more than 1% of keys lack a prior-day exposure")
            if diagnostics["timestamp"].nunique() != len(days) * 8:
                raise ValueError(f"{name}/{vname}: residualized section grid is not 8 per day")
            for row in diagnostics.itertuples():
                previous = window_designs.setdefault(row.timestamp, row.design_sha256)
                if previous != row.design_sha256:
                    raise ValueError(f"{name}: regression design changes across versions")

            common = frame[KEYS].merge(targets[KEYS], on=KEYS, validate="one_to_one")
            sample_ratio = len(common) / len(targets)
            if sample_ratio < 0.99:
                raise ValueError(f"{name}/{vname}: exposure-common sample below 99% of targets")

            blocks = {}
            for variant, column in (("raw_common", "raw_score"), ("processed", "processed_score")):
                metrics, daily, sections, keys_sha = score_common(frame, column, targets, evaluation)
                blocks[variant] = {"metrics": metrics, "daily": daily, "sections": sections,
                                   "keys_sha": keys_sha}
            if blocks["raw_common"]["keys_sha"] != blocks["processed"]["keys_sha"]:
                raise ValueError(f"{name}/{vname}: raw and processed samples differ")
            blocks["raw_full"] = {"metrics": metrics_full, "daily": daily_full,
                                  "sections": sections_full,
                                  "keys_sha": metrics_full["sample_keys_sha256"]}

            for variant in VARIANTS:
                metrics = blocks[variant]["metrics"]
                daily = blocks[variant]["daily"]
                sections = blocks[variant]["sections"]
                results.append({"window": name, "version": vname, "variant": variant,
                                **metric_block(metrics, daily)})
                eight_time_points.append({"window": name, "version": vname, "variant": variant,
                                          "rows": by_time(sections)})
                daily_by.setdefault((vname, variant), []).append(daily.assign(window=name))
            style_out[f"{name}/{vname}"] = {
                "mean_style_explained": json_float(diagnostics["style_explained"].mean()),
                "max_style_explained": json_float(diagnostics["style_explained"].max()),
                "clipped_fraction": json_float(diagnostics["clipped"].sum()
                                               / diagnostics["n_fit"].sum()),
                "min_panel_size": int(diagnostics["n_fit"].min()),
                "exposure_dropped_fraction": json_float(1 - len(frame) / len(predictions)),
                "exposure_common_ratio_of_targets": json_float(sample_ratio),
                "snapshot_dates": sorted({str(value)[:10] for value in diagnostics["snapshot_date"]}),
                "designs_sha256": digest({str(key): value for key, value in window_designs.items()})}
            coverage_out[f"{name}/{vname}"] = {"predictions": int(len(predictions)),
                                               "exposure_rows": int(len(frame)),
                                               "targets": int(len(targets)),
                                               "common": int(len(common))}
        reproduction[name] = window_reproduction

    daily_by = {key: pd.concat(parts, ignore_index=True) for key, parts in daily_by.items()}
    pooled = {}
    for vname in versions:
        for variant in VARIANTS:
            daily = daily_by[(vname, variant)]
            selection = daily[daily["window"].isin(plan["analysis"]["development_months"])]
            selection = selection.sort_values(["window", "date"]).reset_index(drop=True)
            pooled[(vname, variant)] = pooled_metrics(selection)

    pairs = {}
    for left in candidates:
        for variant in ("raw_full", "processed"):
            result = paired_block(plan, daily_by, left, baseline, variant)
            key = f"{left}-minus-{baseline}"
            pairs[(key, variant)] = result
    # raw_common is sample-identical to processed; record its rank_ic delta too.
    for left in candidates:
        key = f"{left}-minus-{baseline}"
        pairs[(key, "raw_common")] = paired_block(plan, daily_by, left, baseline, "raw_common")

    decision = decide(plan, pairs, pooled)
    pairs_out = [result for result in pairs.values()]

    training = read_json(ROOT / "runs" / "label-contrast-20261024" / "training_evidence.json")
    training_rows = {}
    for arm in ("raw", "center"):
        block = training["arms"][arm]
        training_rows[f"v2_{arm}"] = {
            "arm": arm, "run_id": block["run_id"], "wall_seconds": block["wall_seconds"],
            "loss": block["fit"]["loss"], "loss_units": block["fit"]["loss_units"],
            "target_centering": block["fit"]["target_centering"],
            "target_scale": block["fit"]["target_scale"],
            "raw_target_std": block["fit"]["raw_target_std"],
            "centered_target_std": block["fit"]["centered_target_std"],
            "validation_rank_ic_mean": block["validation_metrics"]["rank_ic_mean"],
            "validation_rank_ic_ir": block["validation_metrics"]["rank_ic_ir"]}
    baseline_block = training["baseline_csz"]
    training_rows[baseline] = {
        "arm": "csz", "run_id": baseline_block["run_id"], "wall_seconds": None,
        "loss": baseline_block["fit"]["loss"], "loss_units": baseline_block["fit"]["loss_units"],
        "target_centering": "timestamp_zscore_train_rms_matched",
        "target_scale": baseline_block["fit"]["target_scale"],
        "raw_target_std": baseline_block["fit"]["raw_target_std"],
        "centered_target_std": baseline_block["fit"]["centered_target_std"],
        "validation_rank_ic_mean": baseline_block["validation_metrics"]["rank_ic_mean"],
        "validation_rank_ic_ir": baseline_block["validation_metrics"]["rank_ic_ir"]}

    weights = {version["name"]: {"path": str(resolve(version["weights"]).relative_to(ROOT)),
                                 "sha256": file_sha256(resolve(version["weights"])),
                                 "expected_sha256": version["expected_sha256"],
                                 "label": version["label"], "role": version["role"]}
               for version in plan["versions"]}
    for name, entry in weights.items():
        if entry["sha256"] != entry["expected_sha256"]:
            raise ValueError(f"{name}: weight checksum changed")

    archive = {
        "schema_version": 1,
        "experiment_id": plan["experiment_id"],
        "created_at_utc": utc_now(),
        "preregistration": {"path": str(plan_path.relative_to(ROOT)),
                            "sha256": file_sha256(plan_path),
                            "decision_framework": plan["decision_framework"],
                            "statistics": plan["statistics"],
                            "analysis": plan["analysis"]},
        "scope": ("local descriptive comparison of three label transforms on identical raw official "
                  "labels; no official score, no promotion"),
        "execution": {"replay_run_id": manifest["run_id"],
                      "replay_run_dir": str(replay_dir.relative_to(ROOT)),
                      "replay_wall_seconds": manifest.get("resources", {}).get("wall_seconds"),
                      "analysis_script": str(Path(__file__).resolve().relative_to(ROOT)),
                      "analysis_script_sha256": file_sha256(Path(__file__).resolve()),
                      "weights": weights,
                      "inputs_sha256": {
                          "labels_files": sorted(label_files),
                          "exposure_2023": file_sha256(resolve(
                              "数据/cpt_jyc_2026_exposure/cpt_jyc_2026_exposure_2023_part_0001.parquet")),
                          "exposure_2024": file_sha256(resolve(
                              "数据/cpt_jyc_2026_exposure/cpt_jyc_2026_exposure_2024_part_0001.parquet")),
                          "instruments": file_sha256(resolve(plan["data"]["instruments_path"]))}},
        "conventions": {
            "evaluation": f"jinyiquant.evaluation local-v1, quantile {plan['evaluation']['quantile']}, "
                          f"min_cross_section {plan['evaluation']['min_cross_section']}, "
                          f"periods_per_year {plan['evaluation']['periods_per_year']}, "
                          "constant portfolios undefined",
            "label_use": plan["analysis"]["label_use"],
            "rank_ic": "Pearson correlation of average ranks within one prediction timestamp, "
                       "equally averaged over that day's timestamps",
            "rank_ic_ir": "daily mean / sample std (ddof=1), not annualized",
            "long_short_return": "equal-weight top/bottom 20% gross forward return spread; "
                                 "no fees, execution, overlap or compounding",
            "turnover": "0.5*sum(abs(net_weight_t-net_weight_previous)) between consecutive "
                        "sampled timestamps including overnight; first section undefined",
            "worst_day": "minimum daily long_short_return over the reported period",
            "style_residualization": plan["analysis"]["style_residualization"],
            "official_score": "not implemented; official platform preprocessing unavailable"},
        "versions": [{"name": version["name"], "label": version["label"], "role": version["role"],
                      "target_centering": training_rows[version["name"]]["target_centering"],
                      "prediction_units_note": (
                          "score = network output * raw-label training std; centered arms keep the "
                          "centered label units and are never re-centred or inverse-transformed")}
                     for version in plan["versions"]],
        "training_losses": {
            "note": training["loss_comparability_note"],
            "per_arm": training_rows,
            "shared_targets_sha256": training["shared_input_identity"]["targets_parquet_sha256"]},
        "windows": windows_out,
        "reproduction": {"note": "recomputed raw-score metrics must match the replay run exactly; "
                                 "targets are reconstructed from the official label files",
                         "per_window": reproduction},
        "results": results,
        "development_months": {"months": plan["analysis"]["development_months"],
                               "pooled_metrics": {f"{vname}/{variant}": pooled[(vname, variant)]
                                                  for vname in versions for variant in VARIANTS}},
        "eight_time_points": eight_time_points,
        "style_residualization": style_out,
        "coverage": coverage_out,
        "paired": pairs_out,
        "decision": decision,
        "limitations": [
            "The official platform preprocessing/evaluation is unavailable; every metric is the local "
            "jinyiquant.evaluation approximation and the style residualization is the frozen 2026-10-18 "
            "local convention, not the official procedure.",
            "Only seed 11 exists for RAW and CENTER; the CSZ baseline is also seed 11. A single-seed "
            "contrast cannot separate the label effect from seed noise, which is why the decision "
            "framework treats a win as a replication trigger.",
            "raw_full metrics cover every prediction/target key; raw_common and processed metrics cover "
            "the previous-day-exposure-common sample, so the residualization contrast is sample-identical.",
            "Missing-history keys keep the model's fitted training-target fallback (a constant per arm), "
            "exactly as the frozen predict contract does; no label is read for them.",
            "Long-short returns are gross forward label returns with overlapping 30-minute horizons; "
            "block bootstrap describes day-level sampling noise only, and same-day stocks are never "
            "treated as independent evidence.",
            "july2023 precedes every version's training window and is reported as context only; the "
            "three development months are the pre-registered decision sample."]}

    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, archive)
    archive["_wall_seconds"] = time.monotonic() - started
    return archive


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("plan", type=Path)
    build.add_argument("replay_dir", type=Path)
    build.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    plan = load_plan(args.plan.resolve())
    output = args.output or Path(plan["analysis"]["archive_path"])
    archive = build_archive(args.plan.resolve(), args.replay_dir.resolve(), output.resolve())
    print(output)
    print(f"wall {archive['_wall_seconds']:.1f}s")
    for name, entry in archive["decision"]["candidates"].items():
        print(name, entry["verdict"], "seed29", entry["seed29_followup"],
              {metric: entry["checks"]["rank_ic_improvement"][metric]
               for metric in ("mean_positive", "block5_lower_positive", "two_of_three_months_positive")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
