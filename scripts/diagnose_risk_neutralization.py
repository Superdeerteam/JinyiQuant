"""Replay three label schemes through one frozen local risk-residual pipeline.

The raw-label reference, the centered-label control and the CSZ candidate are
re-scored on exactly the same prior-day exposure snapshots, the same equal-weight
style + industry regression and the same evaluated sample. Both the raw and the
residualized version of every score are reported, together with long-short net
exposures, daily stress views and block-bootstrap intervals.

Nothing is trained, no scheme is selected and no accept/reject gate is applied.
The official preprocessing module is unavailable, so every convention is a
frozen local approximation recorded in the plan and in the run manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.config import EvaluationConfig
from jinyiquant.evaluation import evaluate
from jinyiquant.io import digest, file_sha256, read_json, utc_now, write_json
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir, resource_usage
from modified_tlob.ensemble import paired_block_intervals
from modified_tlob.risk import (
    INDUSTRY_DUMMIES,
    REGRESSORS,
    STYLE_FACTORS,
    portfolio_net_weights,
    previous_exposure_date,
    residualize,
    section_panel,
    trailing_median,
    winsorized_zscore,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "risk-neutralization-20261018"
KEYS = ["date", "instrument"]
WINDOWS = ("research_a", "research_b")
SCHEMES = ("ref10", "cand_centered", "cand_csz")
SEEDS = (11, 29, 47)
VARIANTS = ("raw", "processed")
TIMESTAMPS = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")
REPORTED = ("rank_ic_mean", "rank_ic_ir", "long_short_return_mean", "long_short_sharpe",
            "long_return_mean", "short_return_mean", "turnover_mean", "extreme_return",
            "min_daily_long_short_return")
DELTA_METRICS = ("rank_ic", "long_short_return", "turnover")
PAIRS = {
    "cand_csz_minus_ref10": ("cand_csz", "ref10"),
    "cand_centered_minus_ref10": ("cand_centered", "ref10"),
    "cand_csz_minus_cand_centered": ("cand_csz", "cand_centered"),
}


def resolve(path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def json_float(value):
    return None if value is None or not np.isfinite(value) else float(value)


# ------------------------------------------------------------------ frozen inputs


def load_plan(path) -> dict:
    plan = read_json(resolve(path))
    required = {"schema_version", "experiment_id", "risk_model", "sample", "schemes", "seeds",
                "inputs", "evaluation", "stress", "statistics", "budget"}
    missing = required - set(plan)
    if missing:
        raise ValueError(f"plan is missing {sorted(missing)}")
    if plan["experiment_id"] != EXPERIMENT:
        raise ValueError("plan experiment_id does not match this entry point")
    model = plan["risk_model"]
    if list(model["style_factors"]) != list(STYLE_FACTORS) or \
            list(model["industry_variables"]) != list(INDUSTRY_DUMMIES):
        raise ValueError("plan risk model differs from modified_tlob.risk")
    if model["regression"]["excluded_columns"] != ["ret", "weights", "float_market_cap",
                                                   "industry_level1_code"]:
        raise ValueError("plan excluded columns differ from the frozen module")
    if tuple(plan["seeds"]) != SEEDS or tuple(plan["evaluation"].get("reported", ())) != REPORTED:
        raise ValueError("plan seeds or reported metrics differ from this entry point")
    return plan


def exposure_frame(plan) -> pd.DataFrame:
    paths = [resolve(path) for path in plan["inputs"]["exposure"]]
    frame = pd.concat([pd.read_parquet(path, columns=["date", "instrument", *STYLE_FACTORS,
                                                     *INDUSTRY_DUMMIES, "ret"])
                       for path in paths], ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if frame.duplicated(["date", "instrument"]).any():
        raise ValueError("exposure table contains duplicate date/instrument keys")
    values = frame[["ret", *STYLE_FACTORS, *INDUSTRY_DUMMIES]]
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("exposure table contains undefined values")
    if not frame[list(INDUSTRY_DUMMIES)].isin([0, 1]).all().all():
        raise ValueError("industry dummies must be binary")
    if not (frame[list(INDUSTRY_DUMMIES)].sum(axis=1) == 1).all():
        raise ValueError("industry dummies must sum to one per stock")
    return frame


def daily_sampled_volume(plan) -> pd.Series:
    frame = pd.concat([pd.read_parquet(resolve(path), columns=["date", "volume"])
                       for path in plan["inputs"]["vwap"]], ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.groupby("date")["volume"].sum().astype(float)


def evaluation_config(plan) -> EvaluationConfig:
    block = plan["evaluation"]
    if block.get("constant_portfolios", "undefined") != "undefined":
        raise ValueError("this diagnostic requires undefined constant portfolios")
    return EvaluationConfig(quantile=float(block["quantile"]),
                            periods_per_year=int(block["periods_per_year"]),
                            min_cross_section=int(block["min_cross_section"]),
                            constant_portfolios="undefined")


def input_inventory(plan) -> dict:
    paths = [resolve(path) for path in plan["inputs"]["exposure"] + plan["inputs"]["vwap"]]
    for window in plan["sample"]["windows"].values():
        paths.append(resolve(window["targets"]["path"]))
    for scheme in SCHEMES:
        for window in WINDOWS:
            for seed in SEEDS:
                paths.append(resolve(plan["schemes"][scheme]["prediction_root"]) / window /
                             f"seed{seed}_predictions.parquet")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(paths)}


# ------------------------------------------------------------------ one window


def window_regimes(window, exposure, exposure_dates, volume, *, window_days=20) -> pd.DataFrame:
    """Prior-day volatility and volume states; only dates strictly before the state are used."""
    volatility = exposure.groupby("date")["ret"].apply(
        lambda values: float(np.median(np.abs(values))))
    rows = []
    for day in sorted(pd.Timestamp(value) for value in window["dates"]):
        previous = previous_exposure_date(exposure_dates, day)
        volatility_state = float(volatility.loc[previous])
        volume_state = float(volume.loc[previous])
        volatility_base = trailing_median(volatility, exposure_dates, before=previous,
                                          window=window_days)
        volume_base = trailing_median(volume, exposure_dates, before=previous, window=window_days)
        if volatility_base <= 0 or volume_base <= 0:
            raise ValueError(f"nonpositive regime baseline before {day}")
        rows.append({"date": day, "snapshot_date": previous,
                     "volatility_state": volatility_state, "volatility_baseline": volatility_base,
                     "volatility_ratio": volatility_state / volatility_base,
                     "volume_state": volume_state, "volume_baseline": volume_base,
                     "volume_ratio": volume_state / volume_base,
                     "vol_high": bool(volatility_state > volatility_base),
                     "volume_high": bool(volume_state > volume_base)})
    return pd.DataFrame(rows)


def neutralize(predictions, exposure_by_date, exposure_dates, *, quantile: float):
    """Residualize every section on its prior-day equal-weight style + industry design."""
    frames, diagnostics, exposure_rows = [], [], []
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

        weights = {"raw": portfolio_net_weights(scored["raw_score"], quantile=quantile),
                   "processed": portfolio_net_weights(scored["processed_score"], quantile=quantile)}
        for factor in REGRESSORS:
            values = panel[factor].to_numpy(dtype=float)
            exposure_rows.append({
                "date": pd.Timestamp(timestamp).normalize(), "timestamp": timestamp, "factor": factor,
                "raw": float(np.dot(weights["raw"], values)),
                "processed": float(np.dot(weights["processed"], values))})
        diagnostics.append({"date": pd.Timestamp(timestamp).normalize(), "timestamp": timestamp,
                            "snapshot_date": snapshot_date, "n_fit": len(panel),
                            "clipped": clipped, "style_explained": explained,
                            "design_sha256": hashlib.sha256(design.tobytes()).hexdigest()})
    return (pd.concat(frames, ignore_index=True), pd.DataFrame(diagnostics),
            pd.DataFrame(exposure_rows))


def score_on_common_sample(frame, targets, evaluation):
    """Score one variant on prediction ∩ target keys; both variants share the frame."""
    common = frame[KEYS].merge(targets[KEYS], on=KEYS, validate="one_to_one")
    scored = frame.merge(common, on=KEYS, validate="one_to_one")
    answer = targets.merge(common, on=KEYS, validate="one_to_one")
    if len(scored) != len(answer) or len(scored) != len(common):
        raise ValueError("common sample construction changed the key set")
    metrics, daily, sections = evaluate(scored, answer, evaluation, return_sections=True,
                                        undefined_constant_portfolios=True)
    ordered = scored.sort_values(KEYS)
    keys = ordered.assign(date=ordered["date"].astype(str))
    return metrics, daily, sections, digest(keys[KEYS].to_dict("records"))


def archived_metrics(plan) -> dict:
    root = resolve(plan["schemes"]["cand_csz"]["prediction_root"]).parent
    return read_json(root / "metrics.json")


def archived_value(archive, window, arm, seed, metric):
    for block in ("reference_comparison", "control_comparison"):
        per_seed = archive["windows"][window][block].get("per_seed_metrics")
        if per_seed and str(seed) in per_seed and arm in per_seed[str(seed)]:
            return per_seed[str(seed)][arm][metric]
    raise KeyError(f"archived {window} {arm} seed {seed} {metric} is unavailable")


def metric_block(metrics) -> dict:
    result = {metric: json_float(metrics[metric]) for metric in REPORTED}
    result["day_count"] = int(metrics["day_count"])
    result["cross_section_count"] = int(metrics["cross_section_count"])
    return result


def analyze_window(name, window, plan, exposure, exposure_by_date, exposure_dates, volume,
                   evaluation, archive, recorder) -> dict:
    targets_path = resolve(window["targets"]["path"])
    if file_sha256(targets_path) != window["targets"]["sha256"]:
        raise ValueError(f"{name}: frozen evaluation targets changed")
    targets = pd.read_parquet(targets_path)
    regimes = window_regimes(window, exposure, exposure_dates, volume,
                             window_days=int(plan["stress"]["window"]))
    days = sorted(pd.Timestamp(value) for value in window["dates"])
    if list(regimes.date) != days:
        raise ValueError(f"{name}: regime grid does not match the frozen evaluation dates")

    section_rows, daily_rows, exposure_frames, metrics_by = [], [], [], {}
    designs, sample_hashes = {}, {}
    for scheme in SCHEMES:
        for seed in SEEDS:
            path = resolve(plan["schemes"][scheme]["prediction_root"]) / name / \
                f"seed{seed}_predictions.parquet"
            predictions = pd.read_parquet(path)
            if set(predictions.columns) != {*KEYS, "score"} or predictions.empty:
                raise ValueError(f"{path} is not a score table")
            frame, diagnostics, section_exposure = neutralize(
                predictions, exposure_by_date, exposure_dates, quantile=evaluation.quantile)
            if len(frame) < 0.99 * len(predictions):
                raise ValueError(f"{path}: more than 1% of keys lack a prior-day exposure")
            if set(diagnostics.date) != set(days):
                raise ValueError(f"{path}: prediction dates differ from the frozen dates")

            diagnostic_by_timestamp = diagnostics.drop(columns="date").set_index("timestamp")
            for timestamp, row in diagnostic_by_timestamp.iterrows():
                previous = designs.setdefault(timestamp, row["design_sha256"])
                if previous != row["design_sha256"]:
                    raise ValueError("the regression design changes across schemes or seeds")
            exposure_frames.append(section_exposure.assign(window=name, scheme=scheme, seed=seed))

            # Conformance: the archived scorer must reproduce on its frozen answer keys.
            unrestricted = predictions.merge(targets[KEYS], on=KEYS, validate="one_to_one")
            reference, _, _ = evaluate(unrestricted, targets, evaluation, return_sections=True,
                                       undefined_constant_portfolios=True)
            for metric in ("rank_ic_mean", "long_short_return_mean", "turnover_mean"):
                recorded = archived_value(archive, name, scheme, seed, metric)
                if not np.isclose(reference[metric], recorded, atol=1e-12, rtol=1e-9):
                    raise ValueError(f"{name} {scheme} seed {seed} {metric} does not reproduce")

            for variant, column in (("raw", "raw_score"), ("processed", "processed_score")):
                scored = frame.rename(columns={column: "score"})[[*KEYS, "score"]]
                metrics, daily, sections, keys_sha = score_on_common_sample(
                    scored, targets, evaluation)
                sample_hashes[(scheme, seed, variant)] = keys_sha
                metrics_by[(scheme, seed, variant)] = metric_block(metrics)

                sections = sections.assign(timestamp=pd.to_datetime(sections["timestamp"]))
                joined = sections.set_index("timestamp").join(diagnostic_by_timestamp, how="left")
                if joined[["style_explained", "clipped", "n_fit"]].isna().any().any():
                    raise ValueError(f"{name} {scheme} seed {seed} {variant}: missing diagnostics")
                section_rows.append(pd.DataFrame({
                    "window": name, "scheme": scheme, "seed": seed, "variant": variant,
                    "date": joined.date, "timestamp": joined.index,
                    "rank_ic": joined.rank_ic.to_numpy(), "long_short_return": joined.long_short_return.to_numpy(),
                    "turnover": joined.turnover.to_numpy(), "sample_count": joined.sample_count.to_numpy(),
                    "style_explained": joined.style_explained.to_numpy(),
                    "clipped": joined.clipped.to_numpy(), "n_fit": joined.n_fit.to_numpy(),
                    "snapshot_date": joined.snapshot_date.to_numpy()}))
                daily_rows.append(daily.assign(window=name, scheme=scheme, seed=seed, variant=variant))
                recorder.event("score", "variant_scored", window=name, scheme=scheme, seed=seed,
                               variant=variant, sections=len(sections),
                               samples=int(metrics["sample_count"]),
                               rank_ic=json_float(metrics["rank_ic_mean"]),
                               turnover=json_float(metrics["turnover_mean"]))

    sections = pd.concat(section_rows, ignore_index=True)
    daily = pd.concat(daily_rows, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    exposures = pd.concat(exposure_frames, ignore_index=True)
    if sections.timestamp.nunique() != len(days) * len(TIMESTAMPS):
        raise ValueError(f"{name}: section grid is not {len(days)} days x 8 timestamps")
    if daily.date.nunique() != len(days):
        raise ValueError(f"{name}: daily grid does not cover the frozen dates")
    if len(set(sample_hashes.values())) != 1:
        raise ValueError(f"{name}: raw and processed variants use different evaluation keys")
    if len(designs) != len(days) * len(TIMESTAMPS):
        raise ValueError(f"{name}: incomplete design grid")

    daily = daily.merge(regimes[["date", "vol_high", "volume_high", "volatility_ratio",
                                 "volume_ratio"]], on="date", how="left", validate="many_to_one")
    if daily[["vol_high", "volume_high"]].isna().any().any():
        raise ValueError(f"{name}: a scored day has no regime state")
    return {"days": days, "regimes": regimes, "sections": sections, "daily": daily,
            "exposures": exposures, "metrics": metrics_by, "designs": designs,
            "sample_keys_sha256": next(iter(sample_hashes.values()))}


# ------------------------------------------------------------------ aggregation


def seed_average(rows) -> dict:
    """Metric mean across the three seeds; undefined if any seed is undefined."""
    result = {}
    for metric in REPORTED:
        values = [row[metric] for row in rows]
        result[metric] = None if any(value is None for value in values) else float(np.mean(values))
    result["day_count"] = int(rows[0]["day_count"])
    result["cross_section_count"] = int(rows[0]["cross_section_count"])
    return result


def mean_daily(daily) -> pd.DataFrame:
    """Seed mean per day; every seed must define the day."""
    grouped = daily.groupby(["window", "scheme", "variant", "date"], sort=True)[list(DELTA_METRICS)]
    if not (grouped.count() == len(SEEDS)).all().all():
        raise ValueError("a seed is undefined on at least one evaluated day")
    return grouped.mean().reset_index()


def paired_table(daily_mean, left, right) -> pd.DataFrame:
    """Per-day difference of two (scheme, variant) cells, renamed to the interval API."""
    frames = {}
    for label, (scheme, variant) in {"left": left, "right": right}.items():
        selection = daily_mean[(daily_mean.scheme == scheme) & (daily_mean.variant == variant)]
        frames[label] = selection.set_index(["window", "date"])[list(DELTA_METRICS)]
    table = (frames["left"] - frames["right"]).reset_index().rename(columns={"window": "period"})
    if table.duplicated(["period", "date"]).any() or not np.isfinite(
            table[list(DELTA_METRICS)].to_numpy()).all():
        raise ValueError("paired delta table is incomplete")
    return table


def intervals(table, plan) -> dict:
    settings = plan["statistics"]
    return {f"block_{block}": paired_block_intervals(
        table, block_days=int(block), samples=int(settings["bootstrap_samples"]),
        seed=int(settings["bootstrap_seed"])) for block in settings["block_days"]}


def metric_delta(left, right) -> dict:
    return {metric: None if left[metric] is None or right[metric] is None
            else float(left[metric] - right[metric]) for metric in REPORTED}


def neutralization_report(window_data) -> dict:
    metrics = window_data["metrics"]
    daily = window_data["daily"]
    sections = window_data["sections"]
    daily_mean = mean_daily(daily)
    report, delta_tables = {}, {}
    for scheme in SCHEMES:
        rows = {variant: [metrics[(scheme, seed, variant)] for seed in SEEDS]
                for variant in VARIANTS}
        per_seed = {str(seed): {variant: metrics[(scheme, seed, variant)] for variant in VARIANTS}
                    for seed in SEEDS}
        seed_mean = {variant: seed_average(rows[variant]) for variant in VARIANTS}
        delta_tables[f"{scheme}_processed_minus_raw"] = paired_table(
            daily_mean, (scheme, "processed"), (scheme, "raw"))
        design = sections[(sections.scheme == scheme) & (sections.variant == "raw")]
        report[scheme] = {
            "per_seed": per_seed, "seed_mean": seed_mean,
            "style_explained": {str(seed): float(design[design.seed == seed].style_explained.mean())
                                for seed in SEEDS},
            "clipped_fraction": {str(seed): float(design[design.seed == seed].clipped.sum()
                                                  / design[design.seed == seed].n_fit.sum())
                                 for seed in SEEDS},
            "seed_mean_delta": metric_delta(seed_mean["processed"], seed_mean["raw"]),
            "per_seed_delta": {str(seed): metric_delta(per_seed[str(seed)]["processed"],
                                                       per_seed[str(seed)]["raw"]) for seed in SEEDS}}
    cross = {}
    for label, (candidate, control) in PAIRS.items():
        entry = {}
        for variant in VARIANTS:
            delta_tables[f"{label}_{variant}"] = paired_table(
                daily_mean, (candidate, variant), (control, variant))
            entry[variant] = {"seed_mean_delta": metric_delta(
                seed_average([metrics[(candidate, seed, variant)] for seed in SEEDS]),
                seed_average([metrics[(control, seed, variant)] for seed in SEEDS]))}
        cross[label] = entry
    return {"schemes": report, "cross_scheme": cross, "delta_tables": delta_tables,
            "daily_mean": daily_mean}


def exposure_report(window_data) -> dict:
    exposures = window_data["exposures"]
    result = {}
    for scheme in SCHEMES:
        per_variant = {}
        for variant in VARIANTS:
            seed_mean = exposures[exposures.scheme == scheme].pivot_table(
                index=["timestamp", "seed"], columns="factor", values=variant).groupby("timestamp").mean()
            styles = {factor: {"mean": float(seed_mean[factor].mean()),
                               "mean_abs": float(seed_mean[factor].abs().mean()),
                               "max_abs": float(seed_mean[factor].abs().max())}
                      for factor in STYLE_FACTORS}
            industry_mean = seed_mean[list(INDUSTRY_DUMMIES)].mean()
            per_variant[variant] = {
                "styles": styles,
                "industry": {"max_abs_mean": float(industry_mean.abs().max()),
                             "mean_abs": float(industry_mean.abs().mean()),
                             "worst": str(industry_mean.abs().idxmax())}}
        result[scheme] = per_variant
    return result


def stress_report(window_data) -> dict:
    daily = window_data["daily"]
    regimes = window_data["regimes"].set_index("date")
    result = {}
    for scheme in SCHEMES:
        entry = {}
        for variant in VARIANTS:
            selection = daily[(daily.scheme == scheme) & (daily.variant == variant)]
            by_day = selection.groupby("date")[list(DELTA_METRICS)].mean().join(regimes)
            buckets = {}
            conditions = {
                "high_volatility": by_day.vol_high, "low_volatility": ~by_day.vol_high,
                "high_volume": by_day.volume_high, "low_volume": ~by_day.volume_high,
                "high_volatility_high_volume": by_day.vol_high & by_day.volume_high,
                "high_volatility_low_volume": by_day.vol_high & ~by_day.volume_high,
                "low_volatility_high_volume": ~by_day.vol_high & by_day.volume_high,
                "low_volatility_low_volume": ~by_day.vol_high & ~by_day.volume_high}
            for state, mask in conditions.items():
                part = by_day[mask]
                buckets[state] = {
                    "days": int(len(part)),
                    "rank_ic": json_float(part.rank_ic.mean() if len(part) else None),
                    "long_short_return": json_float(part.long_short_return.mean() if len(part) else None),
                    "turnover": json_float(part.turnover.mean() if len(part) else None)}
            worst = by_day.sort_values("long_short_return").head(3)
            entry[variant] = {"buckets": buckets,
                              "worst_days": [{"date": str(day)[:10],
                                              "long_short_return": json_float(row.long_short_return),
                                              "rank_ic": json_float(row.rank_ic)}
                                             for day, row in worst.iterrows()]}
        result[scheme] = entry
    per_timestamp = {}
    for scheme in SCHEMES:
        per_timestamp[scheme] = {}
        for variant in VARIANTS:
            selection = window_data["sections"][(window_data["sections"].scheme == scheme)
                                                & (window_data["sections"].variant == variant)]
            by_time = selection.assign(clock=selection.timestamp.dt.strftime("%H:%M")).groupby("clock")[
                ["rank_ic", "long_short_return", "turnover"]].mean()
            per_timestamp[scheme][variant] = {
                clock: {key: json_float(row[key]) for key in ("rank_ic", "long_short_return", "turnover")}
                for clock, row in by_time.iterrows()}
    regimes = window_data["regimes"].copy()
    for column in ("date", "snapshot_date"):
        regimes[column] = regimes[column].astype(str)
    return {"buckets": result, "per_timestamp": per_timestamp,
            "regimes": regimes.to_dict("records")}


def interpret(windows) -> str:
    parts = []
    for name in WINDOWS:
        base = windows[name]["schemes"]["ref10"]["seed_mean"]
        for scheme in ("cand_centered", "cand_csz"):
            raw = windows[name]["schemes"][scheme]["seed_mean"]["raw"]["rank_ic_mean"]
            processed = windows[name]["schemes"][scheme]["seed_mean"]["processed"]["rank_ic_mean"]
            if raw and processed is not None:
                parts.append(f"{name} {scheme} RankIC {raw:.4f} -> {processed:.4f} "
                             f"({100 * processed / raw:.0f}% retained)")
        for variant in VARIANTS:
            value = base[variant]["rank_ic_mean"]
            if value is not None:
                parts.append(f"{name} ref10 {variant} RankIC {value:.4f}")
    return "; ".join(parts)


def daily_pair_extremes(daily) -> dict:
    """Worst paired day per contrast, from the seed-mean daily table."""
    mean = daily.groupby(["scheme", "variant", "date"])[list(DELTA_METRICS)].mean().reset_index()
    pivot = {metric: mean.pivot_table(index="date", columns=["scheme", "variant"], values=metric)
             for metric in DELTA_METRICS}
    result = {}
    for candidate, control in PAIRS.values():
        for variant in VARIANTS:
            delta = pivot["long_short_return"][(candidate, variant)] - \
                pivot["long_short_return"][(control, variant)]
            day = delta.idxmin()
            result[f"{candidate}_minus_{control}|{variant}"] = {
                "worst_day": str(day)[:10],
                "long_short_return_delta": float(delta.loc[day]),
                "rank_ic_delta": float(pivot["rank_ic"][(candidate, variant)].loc[day]
                                       - pivot["rank_ic"][(control, variant)].loc[day]),
                "mean_long_short_return_delta": float(delta.mean())}
    return result


def guardrail_day(daily, day) -> dict:
    """The day that failed the previous round's worst-day guardrail, raw versus processed."""
    mean = daily.groupby(["scheme", "variant", "date"]).long_short_return.mean()
    result = {}
    for scheme in SCHEMES:
        for variant in VARIANTS:
            try:
                result[f"{scheme}|{variant}"] = float(mean.loc[(scheme, variant, pd.Timestamp(day))])
            except KeyError:
                raise ValueError(f"guardrail day {day} is missing for {scheme} {variant}")
    for candidate, control in (("cand_csz", "ref10"), ("cand_centered", "ref10")):
        for variant in VARIANTS:
            result[f"{candidate}_minus_{control}|{variant}"] = \
                result[f"{candidate}|{variant}"] - result[f"{control}|{variant}"]
    return result


def record_run(execution, output) -> int:
    run_dir = resolve(execution)
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("status") != "succeeded":
        raise ValueError(f"{run_dir} is not a completed run")
    metrics = read_json(run_dir / "metrics.json")
    metrics_sha256 = file_sha256(run_dir / "metrics.json")
    reproduced_by = sorted(
        sibling.name for sibling in run_dir.parent.iterdir()
        if sibling.is_dir() and sibling != run_dir and (sibling / "metrics.json").is_file()
        and file_sha256(sibling / "metrics.json") == metrics_sha256)
    windows = {}
    for name in WINDOWS:
        block = metrics["windows"][name]
        daily = pd.read_csv(run_dir / f"daily_{name}.csv", parse_dates=["date"])
        windows[name] = {
            "dates": block["dates"], "sections": block["sections"],
            "sample_keys_sha256": block["sample_keys_sha256"],
            "designs_sha256": block["designs_sha256"],
            "seed_mean": {scheme: {variant: block["schemes"][scheme]["seed_mean"][variant]
                                   for variant in VARIANTS} for scheme in SCHEMES},
            "neutralization_delta": {scheme: {
                "seed_mean": block["schemes"][scheme]["seed_mean_delta"],
                "rank_ic_per_seed": {seed: block["schemes"][scheme]["per_seed_delta"][seed]["rank_ic_mean"]
                                     for seed in map(str, SEEDS)},
                "style_explained": block["schemes"][scheme]["style_explained"],
                "clipped_fraction": block["schemes"][scheme]["clipped_fraction"]}
                for scheme in SCHEMES},
            "cross_scheme": block["cross_scheme"],
            "intervals": {label: {key: [row for row in entry["rows"] if row["period"] == name]
                                  for key, entry in block["intervals"][label].items()}
                          for label in block["intervals"]},
            "exposures": metrics["exposure_summary"][name],
            "stress_buckets": {scheme: {variant: metrics["stress"][name]["buckets"][scheme][variant]["buckets"]
                                        for variant in VARIANTS} for scheme in SCHEMES},
            "worst_days": {scheme: {variant: metrics["stress"][name]["buckets"][scheme][variant]["worst_days"]
                                    for variant in VARIANTS} for scheme in SCHEMES},
            "regimes": metrics["stress"][name]["regimes"],
            "daily_pair_extremes": daily_pair_extremes(daily)}
    record = {
        "schema_version": 1, "experiment_id": EXPERIMENT, "created_at_utc": utc_now(),
        "execution": {"run_id": manifest["run_id"], "run_dir": str(run_dir.relative_to(ROOT)),
                      "plan_sha256": manifest["plan_sha256"], "wall_seconds": manifest.get("wall_seconds"),
                      "inputs": len(manifest["inputs_sha256"]),
                      "metrics_sha256": metrics_sha256,
                      "byte_identical_reproduction": reproduced_by},
        "scope": metrics["scope"], "official_reference": metrics["official_reference"],
        "risk_model": metrics["risk_model"], "seeds": metrics["seeds"], "schemes": metrics["schemes"],
        "windows": windows,
        "guardrail_day_2024-03-07": guardrail_day(
            pd.read_csv(run_dir / "daily_research_b.csv", parse_dates=["date"]), "2024-03-07"),
        "observation": metrics["observation"], "limitations": metrics["limitations"],
        "interpretation_rule": "Descriptive diagnostic. Retaining 85-91% of a raw RankIC after residualization "
                               "means most of the ranking edge is not style exposure; a lower retained share "
                               "would mean part of the edge is. No scheme is promoted, no threshold is applied, "
                               "and research_a/research_b are already-used selection data."}
    write_json(resolve(output), record)
    return 0


def run_analysis(plan_path, output_root) -> int:
    started = time.monotonic()
    run_dir = new_run_dir(Path(output_root))
    recorder = Recorder(run_dir, EXPERIMENT)
    manifest = {"experiment_id": EXPERIMENT, "run_id": run_dir.name, "started_at": utc_now(),
                "plan": str(plan_path), "plan_sha256": file_sha256(plan_path),
                "scope": "diagnostic_only", "status": "running"}
    write_json(run_dir / "manifest.json", manifest)
    try:
        plan = load_plan(plan_path)
        recorder.event("start", "run_started", plan=str(plan_path))

        inventory = input_inventory(plan)
        write_json(run_dir / "preregistration.json", {
            "schema_version": 1, "experiment_id": EXPERIMENT, "plan": plan,
            "plan_sha256": file_sha256(plan_path), "inputs_sha256": inventory,
            "frozen_at_utc": utc_now(),
            "note": "Recorded before any metric of this round was computed. The official "
                    "M.jyc_eval implementation is unavailable; every convention is a local "
                    "approximation."})
        recorder.event("start", "inputs_verified", files=len(inventory))

        exposure = exposure_frame(plan)
        exposure_dates = np.array(sorted(exposure["date"].unique()), dtype="datetime64[ns]")
        exposure_by_date = {date: group for date, group in exposure.groupby("date", sort=True)}
        volume = daily_sampled_volume(plan)
        evaluation = evaluation_config(plan)
        archive = archived_metrics(plan)
        if tuple(plan["sample"]["timestamps"]) != TIMESTAMPS:
            raise ValueError("plan timestamp grid differs from the frozen eight sections")

        windows, exposure_summary, stress = {}, {}, {}
        for name in WINDOWS:
            window = plan["sample"]["windows"][name]
            data = analyze_window(name, window, plan, exposure, exposure_by_date, exposure_dates,
                                  volume, evaluation, archive, recorder)
            recorder.event("analysis", "window_replayed", window=name,
                           sections=len(data["sections"]))
            figures = neutralization_report(data)
            windows[name] = {
                "window": window, "sample_keys_sha256": data["sample_keys_sha256"],
                "dates": [str(day)[:10] for day in data["days"]],
                "sections": len(data["sections"]),
                "designs_sha256": digest({str(key): value for key, value in data["designs"].items()}),
                "schemes": figures["schemes"], "cross_scheme": figures["cross_scheme"],
                "intervals": {key: intervals(table, plan) for key, table in
                              figures["delta_tables"].items()}}
            exposure_summary[name] = exposure_report(data)
            stress[name] = stress_report(data)
            data["sections"].to_parquet(run_dir / f"sections_{name}.parquet", index=False)
            data["daily"].to_csv(run_dir / f"daily_{name}.csv", index=False)
            data["exposures"].to_parquet(run_dir / f"section_exposures_{name}.parquet", index=False)

        wall = time.monotonic() - started
        if wall > plan["budget"]["wall_cap_seconds"]:
            raise RuntimeError(f"wall budget exceeded: {wall:.1f}s")
        if resource_usage()["rss_mb"] > plan["budget"]["peak_rss_cap_mib"]:
            raise RuntimeError("peak RSS exceeded the frozen cap")

        summary = {
            "schema_version": 1, "experiment_id": EXPERIMENT,
            "scope": "local approximation of the official preprocessing; no official score",
            "hypothesis": plan["hypothesis"],
            "official_reference": plan["official_reference"],
            "risk_model": plan["risk_model"],
            "seeds": list(SEEDS), "schemes": {scheme: plan["schemes"][scheme]["label"]
                                             for scheme in SCHEMES},
            "windows": windows,
            "exposure_summary": exposure_summary,
            "stress": stress,
            "observation": interpret(windows),
            "limitations": [
                "The official M.jyc_eval preprocessing is unavailable, so winsorization strength, standardization, exposure timing, regression weights and missing-value handling are frozen local choices, not official parameters.",
                "Exposure snapshots lag the signal by one trading day; a same-day snapshot cannot be shown to be published before the intraday timestamp.",
                "research_a and research_b were already used for scheme selection, so cross-scheme contrasts are descriptive and not independent confirmation.",
                "Long-short returns are gross forward-label returns with no fees, execution, overlap or compounding adjustment.",
                "Two ten-day windows and three seeds bound the statistical evidence; block-bootstrap intervals describe day-level sampling noise only.",
                "No model was trained and no scheme is promoted; the retained reference remains the raw-label 10-day seed-11 run."]}
        write_json(run_dir / "metrics.json", summary)
        recorder.event("save", "metrics_saved", wall_seconds=round(wall, 1))
        capture = capture_source(run_dir)
        manifest |= {"completed_at": utc_now(), "wall_seconds": round(wall, 1),
                     "environment": environment(), "source": capture,
                     "inputs_sha256": inventory}
        write_json(run_dir / "manifest.json", manifest)
        finish(run_dir, "succeeded", 0, resources=resource_usage())
        print(json.dumps({"run_id": run_dir.name, "run_dir": str(run_dir),
                          "observation": summary["observation"]}, ensure_ascii=False))
        return 0
    except BaseException as error:  # noqa: BLE001 - failures must be archived, not masked
        recorder.failure("run", error)
        finish(run_dir, "failed", 1, resources=resource_usage(),
               error={"type": type(error).__name__, "message": str(error)})
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="replay the frozen plan")
    run.add_argument("plan")
    run.add_argument("--output-root", default=str(ROOT / "runs" / EXPERIMENT))
    record = subparsers.add_parser("record", help="write the compact research record")
    record.add_argument("execution")
    record.add_argument("--output", default=str(ROOT / "modified_tlob" /
                                                "risk_neutralization_20261018.json"))
    arguments = parser.parse_args(argv)
    if arguments.command == "record":
        return record_run(arguments.execution, arguments.output)
    return run_analysis(resolve(arguments.plan), arguments.output_root)


if __name__ == "__main__":
    sys.exit(main())
