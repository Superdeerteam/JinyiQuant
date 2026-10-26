"""Compare fixed-ratio TLOB/MLPLOB score blends on three development months.

Reads the six member predictions produced by ``replay_model_versions.py`` and
blends them post hoc (the blend is linear in restored return scores, so this is
identical to scoring a blend checkpoint). Reports per-month metrics, member and
ensemble prediction correlation, volatility/volume stress buckets and
time-of-day behaviour. No training, no weight fitting, no platform access.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jinyiquant.evaluation import evaluate  # noqa: E402
from jinyiquant.io import read_json, utc_now, write_json  # noqa: E402

MEMBERS = ("tlob_s11", "tlob_s29", "tlob_s47", "mlp_s11", "mlp_s29", "mlp_s47")
TL0B = ("tlob_s11", "tlob_s29", "tlob_s47")
MLP = ("mlp_s11", "mlp_s29", "mlp_s47")
ARMS = {
    "tlob_3seed": {name: 1 / 3 for name in TL0B},
    "mlp_3seed": {name: 1 / 3 for name in MLP},
    "blend_75_25": {**{name: 0.75 / 3 for name in TL0B}, **{name: 0.25 / 3 for name in MLP}},
    "blend_50_50": {**{name: 1 / 6 for name in TL0B}, **{name: 1 / 6 for name in MLP}},
    "blend_25_75": {**{name: 0.25 / 3 for name in TL0B}, **{name: 0.75 / 3 for name in MLP}},
}
ARM_ORDER = ("tlob_3seed", "mlp_3seed", "blend_75_25", "blend_50_50", "blend_25_75")
KEYS = ["date", "instrument"]
SAMPLE_TIMES = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")
EVALUATION = {"quantile": 0.2, "periods_per_year": 252, "min_cross_section": 3,
              "constant_portfolios": "undefined"}


def load_window(run_dir: Path, name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base = run_dir / "windows" / name
    keys = pd.read_parquet(base / "keys.parquet")
    targets = pd.read_parquet(base / "targets.parquet")
    frames = {}
    for member in MEMBERS:
        frame = pd.read_parquet(base / member / "predictions.parquet")
        if frame[KEYS].duplicated().any() or not np.isfinite(frame.score).all():
            raise ValueError(f"{name}/{member}: invalid predictions")
        frames[member] = frame.sort_values(KEYS).reset_index(drop=True)
    reference = frames[MEMBERS[0]][KEYS]
    for member, frame in frames.items():
        if not frame[KEYS].equals(reference):
            raise ValueError(f"{name}/{member}: prediction keys differ")
    return keys, targets, frames


def arm_scores(frames: dict, weights: dict) -> pd.DataFrame:
    out = frames[MEMBERS[0]][KEYS].copy()
    score = np.zeros(len(out))
    for member, weight in weights.items():
        score += weight * frames[member].score.to_numpy(dtype=np.float64)
    out["score"] = score
    return out


def monthly_metrics(predictions: pd.DataFrame, targets: pd.DataFrame):
    scored = predictions.merge(targets[KEYS], on=KEYS, how="inner", validate="one_to_one")
    if scored.empty:
        raise ValueError("no scored predictions after exact key alignment")
    return evaluate(scored, targets, EVALUATION, return_sections=True,
                    undefined_constant_portfolios=True)


def section_correlations(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merged = left.merge(right, on=KEYS, suffixes=("_left", "_right"), validate="one_to_one")
    grouped = merged.groupby("date", sort=True)
    out = grouped.apply(
        lambda section: pd.Series({
            "n": len(section),
            "pearson": section.score_left.corr(section.score_right),
            "spearman": section.score_left.rank().corr(section.score_right.rank()),
            "top_overlap": float(
                (lambda left, right: (left & right).sum() / max(1, (left | right).sum()))(
                    section.score_left.rank(pct=True) > 0.8,
                    section.score_right.rank(pct=True) > 0.8)),
        }),
        include_groups=False).reset_index()
    return out


def block_interval(values, *, block: int = 5, draws: int = 20000, seed: int = 20261026) -> dict:
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(values) < 2:
        return {"days": int(len(values)), "mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), size=draws)
    indices = (starts[:, None] + np.arange(block)[None, :]) % len(values)
    means = values[indices].mean(axis=1)
    return {"days": int(len(values)), "mean": float(values.mean()),
            "ci_low": float(np.quantile(means, 0.025)), "ci_high": float(np.quantile(means, 0.975))}


def load_conditions(plan: dict, window: dict, instruments: set) -> pd.DataFrame:
    """Trailing-median volatility and amount regimes from the official vwap table."""
    start = pd.Timestamp(window["start"]) - pd.Timedelta(days=70)
    end = pd.Timestamp(window["end"])
    labels_path = Path(plan["data"]["labels_path"])
    if not labels_path.is_absolute():
        labels_path = (Path(plan["_path"]).parent / labels_path).resolve()
    frames = []
    for month in pd.period_range(start, end, freq="M"):
        path = labels_path / f"{labels_path.name}_{month.strftime('%Y-%m')}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["date", "instrument", "vwap_return", "amount"],
                                filters=[("date", ">=", start), ("date", "<", end)])
        frames.append(frame)
    if not frames:
        raise ValueError(f"{window['name']}: no label months for stress conditions")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame.date.dt.strftime("%H:%M").isin(SAMPLE_TIMES)]
    frame = frame.loc[frame.instrument.isin(instruments)]
    by_time = frame.groupby("date", sort=True).agg(market_return=("vwap_return", "mean"),
                                                   amount=("amount", "sum"))
    by_day = by_time.groupby(by_time.index.normalize()).agg(
        market_vol=("market_return", "std"), amount=("amount", "sum"))
    by_day.index.name = "day"
    by_day["vol_ratio"] = by_day.market_vol / by_day.market_vol.shift(1).rolling(
        20, min_periods=10).median()
    by_day["amount_ratio"] = by_day.amount / by_day.amount.shift(1).rolling(
        20, min_periods=10).median()
    return by_day.reset_index()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--record", type=Path, default=ROOT / "modified_tlob/tlob_mlplob_replay_20261026.json")
    args = parser.parse_args(argv)
    plan = read_json(args.plan)
    plan["_path"] = str(args.plan.resolve())
    run_dir = args.run_dir.resolve()
    record = {
        "schema_version": 1,
        "experiment_id": "tlob-mlplob-replay-20261026",
        "created_at_utc": utc_now(),
        "scope": ("Six frozen V3-recipe checkpoints replayed on 2023-11, 2024-01 and 2024-03; "
                  "TLOB and MLPLOB ensembles blended post hoc at 75:25, 50:50 and 25:75."),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "plan": str(args.plan.resolve().relative_to(ROOT)),
        "arm_definition": {name: {member: round(weight, 6) for member, weight in ARMS[name].items()}
                           for name in ARM_ORDER},
        "windows": {},
        "correlations": {},
        "stress": {},
        "paired": {},
        "evidence_boundary": (
            "Local approximate metric convention: gross forward target returns, no fees, "
            "no execution or market impact. 2023-11, 2024-01 and 2024-03 are used development "
            "windows; 2023-11-01..14 is also the V3 validation month, so none of the three is an "
            "independent confirmation window. Stress buckets use 20-trading-day trailing medians "
            "from the official vwap table; a ~20-day month gives small bucket samples."
        ),
    }
    windows = plan["windows"]
    summary_rows, correlation_rows, member_rows, time_rows = [], [], [], []
    daily_all = pd.DataFrame()
    per_window_daily = {}
    sections_cache = {}
    conditions_cache = {}

    for window in windows:
        name = window["name"]
        keys, targets, frames = load_window(run_dir, name)
        instruments = set(keys.instrument.unique())
        conditions = load_conditions(plan, window, instruments)
        conditions_cache[name] = conditions
        window_metrics = {}
        window_daily = {}
        window_sections = {}
        for arm in ARM_ORDER:
            predictions = arm_scores(frames, ARMS[arm])
            metrics, daily, sections = monthly_metrics(predictions, targets)
            ic = daily.ic
            metrics["ic_ir"] = float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else None
            window_metrics[arm] = {key: metrics[key] for key in (
                "ic_mean", "rank_ic_mean", "rank_ic_ir", "ic_ir", "long_short_return_mean",
                "long_short_sharpe", "turnover_mean", "min_daily_long_short_return", "day_count",
                "cross_section_count", "sample_count")}
            daily = daily.copy()
            daily.insert(0, "arm", arm)
            daily.insert(0, "window", name)
            window_daily[arm] = daily
            window_sections[arm] = sections
            summary_rows.append({"window": name, "arm": arm, **window_metrics[arm]})
        daily_all = pd.concat([daily_all, *window_daily.values()], ignore_index=True)
        per_window_daily[name] = window_daily
        sections_cache[name] = window_sections

        merged = frames[TL0B[0]][KEYS].copy()
        merged["tlob"] = arm_scores(frames, ARMS["tlob_3seed"]).score
        merged["mlp"] = arm_scores(frames, ARMS["mlp_3seed"]).score
        correlation = section_correlations(
            merged[KEYS].assign(score_left=merged.tlob),
            merged[KEYS].assign(score_right=merged.mlp))
        record["correlations"][name] = {
            "ensemble_section_pearson_mean": float(correlation.pearson.mean()),
            "ensemble_section_spearman_mean": float(correlation.spearman.mean()),
            "ensemble_section_spearman_p05": float(correlation.spearman.quantile(0.05)),
            "ensemble_section_spearman_min": float(correlation.spearman.min()),
            "top_quintile_overlap_mean": float(correlation.top_overlap.mean()),
            "sections": int(len(correlation)),
        }
        for section in correlation.itertuples():
            correlation_rows.append({"window": name, "date": section.date, "pearson": section.pearson,
                                     "spearman": section.spearman, "top_overlap": section.top_overlap})
        for left in MEMBERS:
            for right in MEMBERS:
                pair = section_correlations(
                    frames[left].rename(columns={"score": "score_left"}),
                    frames[right].rename(columns={"score": "score_right"}))
                member_rows.append({"window": name, "left": left, "right": right,
                                    "spearman_mean": float(pair.spearman.mean()) if len(pair) else None})
        tlob_daily = window_daily["tlob_3seed"].set_index("date").rank_ic
        mlp_daily = window_daily["mlp_3seed"].set_index("date").rank_ic
        common = tlob_daily.index.intersection(mlp_daily.index)
        record["correlations"][name]["daily_rank_ic_pearson"] = float(
            tlob_daily.loc[common].corr(mlp_daily.loc[common]))
        ls_left = window_daily["tlob_3seed"].set_index("date").long_short_return
        ls_right = window_daily["mlp_3seed"].set_index("date").long_short_return
        record["correlations"][name]["daily_long_short_pearson"] = float(
            ls_left.loc[common].corr(ls_right.loc[common]))

        for arm in ARM_ORDER:
            sections = window_sections[arm]
            clock = pd.to_datetime(sections.timestamp).dt.strftime("%H:%M")
            grouped = sections.assign(clock=clock).groupby("clock", sort=True)
            for value, group in grouped:
                time_rows.append({"window": name, "arm": arm, "time": value,
                                  "rank_ic_mean": float(group.rank_ic.mean()),
                                  "long_short_return_mean": float(group.long_short_return.mean()),
                                  "sections": int(len(group))})

        by_day = pd.DataFrame({"date": daily_all.loc[daily_all.window == name, "date"].unique()})
        by_day["day"] = pd.to_datetime(by_day.date)
        by_day = by_day.merge(conditions, on="day", how="left")
        stress_rows = []
        for arm in ARM_ORDER:
            arm_daily = window_daily[arm].merge(
                by_day[["date", "vol_ratio", "amount_ratio", "market_vol"]], on="date", how="left")
            for regime, mask in (("high_vol", arm_daily.vol_ratio >= 1.0),
                                 ("low_vol", arm_daily.vol_ratio < 1.0),
                                 ("high_amount", arm_daily.amount_ratio >= 1.0),
                                 ("low_amount", arm_daily.amount_ratio < 1.0)):
                part = arm_daily.loc[mask]
                if part.empty:
                    continue
                stress_rows.append({
                    "window": name, "arm": arm, "regime": regime, "days": int(len(part)),
                    "rank_ic_mean": float(part.rank_ic.mean()),
                    "long_short_return_mean": float(part.long_short_return.mean()),
                    "long_short_hit_rate": float((part.long_short_return > 0).mean()),
                    "worst_long_short": float(part.long_short_return.min()),
                    "worst_rank_ic": float(part.rank_ic.min()),
                })
        record["stress"][name] = stress_rows
        record["windows"][name] = {
            "label": window["label"],
            "start": window["start"], "end": window["end"],
            "days": len(window_daily["tlob_3seed"]),
            "arms": window_metrics,
            "tradeable_days_above_trailing_median": {
                "high_vol": int((by_day.vol_ratio >= 1.0).sum()),
                "high_amount": int((by_day.amount_ratio >= 1.0).sum()),
            },
        }

    summary = pd.DataFrame(summary_rows)
    report = run_dir / "report" / "blend_comparison"
    report.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report / "monthly_summary.csv", index=False)
    daily_all.to_csv(report / "daily_all.csv", index=False)
    time_frame = pd.DataFrame(time_rows)
    time_frame.to_csv(report / "time_of_day.csv", index=False)
    member_frame = pd.DataFrame(member_rows)
    member_frame.to_csv(report / "member_correlation.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(report / "ensemble_section_correlation.csv", index=False)
    pd.DataFrame([row for rows in record["stress"].values() for row in rows]).to_csv(
        report / "stress_regimes.csv", index=False)

    paired_rows = []
    for name in [window["name"] for window in windows]:
        daily = per_window_daily[name]
        for left_arm in ("blend_75_25", "blend_50_50", "blend_25_75"):
            left = daily[left_arm].set_index("date")
            for right_arm in ("tlob_3seed", "mlp_3seed"):
                right = daily[right_arm].set_index("date")
                common = left.index.intersection(right.index)
                for metric in ("rank_ic", "long_short_return", "turnover"):
                    delta = (left.loc[common, metric] - right.loc[common, metric]).dropna()
                    interval = block_interval(delta.to_numpy())
                    paired_rows.append({"window": name, "left": left_arm, "right": right_arm,
                                        "metric": metric, "mean_delta": interval["mean"],
                                        "block5_ci_low": interval["ci_low"],
                                        "block5_ci_high": interval["ci_high"],
                                        "positive_days": int((delta > 0).sum()),
                                        "negative_days": int((delta < 0).sum()),
                                        "max_abs_date": str(delta.abs().idxmax()) if len(delta) else None})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(report / "paired_blend.csv", index=False)
    record["paired"] = paired_rows

    # Exact recheck of the prior 2023-11-01..14 validation-window record.
    reference = {
        "tlob_3seed": {"rank_ic_mean": 0.09234883942058289, "long_short_return_mean": 0.0006493823488517845,
                       "turnover_mean": 1.127440775305097},
        "mlp_3seed": {"rank_ic_mean": 0.09293593041939735, "long_short_return_mean": 0.000654415613485363,
                      "turnover_mean": 1.1132089016511126},
        "blend_50_50": {"rank_ic_mean": 0.09382126926525722, "long_short_return_mean": 0.0006636980333684818,
                        "turnover_mean": 1.1015254845656857},
    }
    recheck = {}
    for arm in ("tlob_3seed", "mlp_3seed", "blend_50_50"):
        part = per_window_daily["nov2023"][arm]
        part = part.loc[part.date <= "2023-11-14"]
        recheck[arm] = {
            "days": int(len(part)),
            "rank_ic_mean": float(part.rank_ic.mean()),
            "long_short_return_mean": float(part.long_short_return.mean()),
            "turnover_mean": float(part.turnover.mean()),
            "prior_record": reference[arm],
        }
    record["validation_window_recheck"] = recheck
    # Pooled descriptive read: mean over the three windows (no cross-window interval is claimed).
    pooled = summary.groupby("arm")[["rank_ic_mean", "long_short_return_mean", "turnover_mean"]].mean()
    record["pooled_three_month_means"] = pooled.round(8).to_dict(orient="index")
    record["report_dir"] = str(report.relative_to(ROOT))
    record["headline"] = {
        "cross_trunk_ensemble_spearman": {name: round(value["ensemble_section_spearman_mean"], 4)
                                          for name, value in record["correlations"].items()},
        "daily_rank_ic_pearson": {name: round(value["daily_rank_ic_pearson"], 4)
                                  for name, value in record["correlations"].items()},
        "pooled_rank_ic": {arm: round(float(pooled.loc[arm, "rank_ic_mean"]), 6) for arm in ARM_ORDER},
    }
    write_json(args.record, record)

    order = {arm: index for index, arm in enumerate(ARM_ORDER)}
    table = summary.assign(order=summary.arm.map(order)).sort_values(["window", "order"])
    print(table[["window", "arm", "rank_ic_mean", "long_short_return_mean", "turnover_mean"]]
          .to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\npooled three-month means")
    print(pooled.to_string(float_format=lambda value: f"{value:.5f}"))
    print(f"\nrecord: {args.record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
