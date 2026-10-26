"""Compare multi-scale patch, TLOB, MLPLOB and fixed blends on replay windows.

Reads the per-member predictions produced by ``replay_model_versions.py`` and
evaluates pre-declared equal-weight arms on raw and risk-neutralized scores,
plus volatility/amount stress buckets. No training, no weight fitting.
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

from jinyiquant.config import EvaluationConfig  # noqa: E402
from jinyiquant.evaluation import evaluate  # noqa: E402
from jinyiquant.io import read_json, utc_now, write_json  # noqa: E402
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
SAMPLE_TIMES = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")
METRICS = ("rank_ic_mean", "rank_ic_ir", "ic_mean", "ic_ir", "long_short_return_mean",
           "long_short_sharpe", "turnover_mean", "min_daily_long_short_return", "day_count",
           "sample_count", "score_std")
BLOCK_DAYS = 5
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 20261026


def load_members(plan: dict, run_dir: Path) -> dict:
    groups = {"tlob": [], "mlp": [], "patch": []}
    for version in plan["versions"]:
        name = version["name"]
        for prefix in groups:
            if name.startswith(prefix + "_"):
                groups[prefix].append(name)
    for prefix, names in groups.items():
        if names and len(names) != 3:
            raise ValueError(f"{prefix} requires exactly three seed members, got {names}")
    if not groups["tlob"] or not groups["mlp"]:
        raise ValueError("replay plan requires the TLOB and MLPLOB member groups")
    return groups


def define_arms(groups: dict) -> dict:
    arms = {f"{prefix}_3seed": {name: 1 / 3 for name in names}
            for prefix, names in groups.items() if len(names) == 3}
    if all(len(groups[prefix]) == 3 for prefix in ("tlob", "mlp")):
        arms["blend_tlob_mlp_50_50"] = {**{name: 1 / 6 for name in groups["tlob"]},
                                        **{name: 1 / 6 for name in groups["mlp"]}}
    if all(len(groups[prefix]) == 3 for prefix in ("tlob", "patch")):
        arms["blend_tlob_patch_50_50"] = {**{name: 1 / 6 for name in groups["tlob"]},
                                          **{name: 1 / 6 for name in groups["patch"]}}
    if all(len(groups[prefix]) == 3 for prefix in ("mlp", "patch")):
        arms["blend_mlp_patch_50_50"] = {**{name: 1 / 6 for name in groups["mlp"]},
                                         **{name: 1 / 6 for name in groups["patch"]}}
    if all(len(groups[prefix]) == 3 for prefix in ("tlob", "mlp", "patch")):
        arms["blend_all_equal"] = {name: 1 / 9 for prefix in groups for name in groups[prefix]}
    return arms


def load_window(run_dir: Path, name: str, members: list[str]):
    base = run_dir / "windows" / name
    keys = pd.read_parquet(base / "keys.parquet")
    targets = pd.read_parquet(base / "targets.parquet")
    frames = {}
    for member in members:
        frame = pd.read_parquet(base / member / "predictions.parquet")
        if frame[KEYS].duplicated().any() or not np.isfinite(frame.score).all():
            raise ValueError(f"{name}/{member}: invalid predictions")
        frames[member] = frame.sort_values(KEYS).reset_index(drop=True)
    reference = frames[members[0]][KEYS]
    for member, frame in frames.items():
        if not frame[KEYS].equals(reference):
            raise ValueError(f"{name}/{member}: prediction keys differ")
    return keys, targets, frames


def arm_scores(frames: dict, weights: dict) -> pd.DataFrame:
    out = frames[next(iter(weights))][KEYS].copy()
    score = np.zeros(len(out))
    for member, weight in weights.items():
        score += weight * frames[member].score.to_numpy(dtype=np.float64)
    out["score"] = score
    return out


def load_exposure(plan: dict) -> tuple[pd.DataFrame, dict, np.ndarray]:
    paths = plan.get("inputs", {}).get("exposure")
    if paths is None:
        paths = [str(ROOT / "数据/cpt_jyc_2026_exposure" /
                     f"cpt_jyc_2026_exposure_{year}_part_0001.parquet") for year in (2023, 2024)]
    else:
        base = Path(plan["_path"]).parent
        paths = [str((base / path).resolve()) if not Path(path).is_absolute() else path
                 for path in paths]
    frame = pd.concat([pd.read_parquet(path, columns=["date", "instrument", *STYLE_FACTORS,
                                                     *INDUSTRY_DUMMIES, "ret"])
                       for path in paths], ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if frame.duplicated(["date", "instrument"]).any():
        raise ValueError("exposure table contains duplicate keys")
    values = frame[["ret", *STYLE_FACTORS, *INDUSTRY_DUMMIES]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("exposure table contains undefined values")
    by_date = {date: group for date, group in frame.groupby("date", sort=True)}
    dates = np.sort(frame["date"].unique()).astype("datetime64[ns]")
    return frame, by_date, dates


def neutralize(predictions: pd.DataFrame, exposure_by_date: dict, exposure_dates: np.ndarray,
               *, quantile: float) -> pd.DataFrame:
    frames = []
    for timestamp, section in predictions.groupby("date", sort=True):
        signal_date = pd.Timestamp(timestamp).normalize()
        snapshot_date = previous_exposure_date(exposure_dates, signal_date)
        if not snapshot_date < signal_date:
            raise ValueError(f"{timestamp}: exposure snapshot is not strictly prior")
        panel = section_panel(section["score"], section["instrument"], exposure_by_date[snapshot_date])
        standardized, _ = winsorized_zscore(panel["score"].to_numpy(dtype=float))
        design = panel[list(REGRESSORS)].to_numpy(dtype=float)
        residual, _ = residualize(standardized, design)
        scored = panel[["instrument"]].copy()
        scored["date"] = timestamp
        scored["raw_score"] = panel["score"].to_numpy(dtype=float)
        scored["processed_score"] = residual
        frames.append(scored)
    return pd.concat(frames, ignore_index=True)


def evaluate_scores(scores: pd.DataFrame, targets: pd.DataFrame, evaluation: EvaluationConfig,
                    column: str):
    common = scores[KEYS].merge(targets[KEYS], on=KEYS, how="inner", validate="one_to_one")
    frame = scores.merge(common, on=KEYS, how="inner", validate="one_to_one")
    answer = targets.merge(common, on=KEYS, how="inner", validate="one_to_one")
    scored = frame[KEYS].assign(score=frame[column].to_numpy(dtype=np.float64))
    metrics, daily, sections = evaluate(scored, answer, evaluation, return_sections=True,
                                        undefined_constant_portfolios=True)
    metrics = {key: metrics[key] for key in METRICS if key in metrics}
    metrics["score_std"] = float(scored.score.std())
    ic_std = daily.ic.std(ddof=1)
    metrics["ic_ir"] = float(daily.ic.mean() / ic_std) if ic_std > 0 else None
    return metrics, daily, sections


def block_interval(values, *, block: int, draws: int, seed: int) -> dict:
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(values) < 2:
        return {"days": int(len(values)), "mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), size=draws)
    indices = (starts[:, None] + np.arange(block)[None, :]) % len(values)
    means = values[indices].mean(axis=1)
    return {"days": int(len(values)), "mean": float(values.mean()),
            "ci_low": float(np.quantile(means, 0.025)),
            "ci_high": float(np.quantile(means, 0.975))}


def load_conditions(plan: dict, window: dict, instruments: set) -> pd.DataFrame:
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
        frames.append(pd.read_parquet(path, columns=["date", "instrument", "vwap_return", "amount"],
                                      filters=[("date", ">=", start), ("date", "<", end)]))
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
    by_day.index.name = "date"
    return by_day.reset_index()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plan", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--record", type=Path,
                        default=ROOT / "modified_tlob/multiscale_replay_20261026.json")
    args = parser.parse_args(argv)
    plan = read_json(args.plan)
    plan["_path"] = str(args.plan.resolve())
    run_dir = args.run_dir.resolve()
    groups = load_members(plan, run_dir)
    arms = define_arms(groups)
    evaluation = EvaluationConfig(**plan["evaluation"])
    members = [name for names in groups.values() for name in names]
    _, exposure_by_date, exposure_dates = load_exposure(plan)
    quantile = float(plan["evaluation"]["quantile"])

    record = {"schema_version": 1, "experiment_id": plan["experiment_id"],
              "created_at_utc": utc_now(),
              "scope": ("Frozen V3-recipe TLOB, MLPLOB and dual-scale patch checkpoints replayed on "
                        "2023-11, 2024-01 and 2024-03; pre-declared equal-weight arms scored on raw and "
                        "prior-day-exposure-neutralized scores."),
              "run_dir": str(run_dir.relative_to(ROOT)), "plan": str(args.plan.resolve().relative_to(ROOT)),
              "arm_definition": {arm: {member: round(weight, 6) for member, weight in weights.items()}
                                 for arm, weights in arms.items()},
              "windows": {}, "cross_month": {}, "paired": {}, "stress": {},
              "evidence_boundary": (
                  "Local approximate convention: gross forward target returns, no fees or execution. "
                  "The neutralizer is the frozen local approximation (median+/-3*1.4826*MAD winsorization, "
                  "prior-day equal-weight OLS on 10 style factors and 32 industry dummies); it is not the "
                  "official BARRA implementation. All three months are used development windows, so none "
                  "is independent confirmation. Stress buckets use trailing 20-day medians and have few days.")}
    summary_rows, daily_rows, stress_rows, paired_rows = [], [], [], []
    monthly = {}
    for window in plan["windows"]:
        name = window["name"]
        keys, targets, frames = load_window(run_dir, name, members)
        conditions = load_conditions(plan, window, set(keys.instrument.unique()))
        window_record = {"arms": {}, "neutralization": {}, "stress": {}}
        for arm in arms:
            scores = arm_scores(frames, arms[arm])
            neutral = neutralize(scores, exposure_by_date, exposure_dates, quantile=quantile)
            raw_metrics, raw_daily, _ = evaluate_scores(
                neutral, targets, evaluation, "raw_score")
            processed_metrics, processed_daily, _ = evaluate_scores(
                neutral, targets, evaluation, "processed_score")
            window_record["arms"][arm] = raw_metrics
            window_record["neutralization"][arm] = processed_metrics
            summary_rows.append({"window": name, "arm": arm, "basis": "raw", **raw_metrics})
            summary_rows.append({"window": name, "arm": arm, "basis": "neutralized",
                                 **processed_metrics})
            for basis, daily in (("raw", raw_daily), ("neutralized", processed_daily)):
                frame = daily.copy()
                frame.insert(0, "arm", arm)
                frame.insert(0, "basis", basis)
                frame.insert(0, "window", name)
                daily_rows.append(frame)
            stress = raw_daily.copy()
            stress["date_key"] = pd.to_datetime(stress.date).dt.normalize()
            conditions_frame = conditions.assign(
                date_key=pd.to_datetime(conditions.date).dt.normalize())
            stress = stress.merge(
                conditions_frame[["date_key", "vol_ratio", "amount_ratio", "market_vol"]],
                on="date_key", how="left")
            for regime, mask in (("high_vol", stress.vol_ratio >= 1.0),
                                 ("low_vol", stress.vol_ratio < 1.0),
                                 ("high_amount", stress.amount_ratio >= 1.0),
                                 ("low_amount", stress.amount_ratio < 1.0)):
                part = stress.loc[mask]
                values = {"days": int(len(part)),
                          "rank_ic_mean": float(part.rank_ic.mean()) if len(part) else None,
                          "long_short_return_mean": float(part.long_short_return.mean()) if len(part) else None}
                window_record["stress"].setdefault(regime, {})[arm] = values
                stress_rows.append({"window": name, "arm": arm, "regime": regime, **values})
        monthly[name] = window_record
        record["windows"][name] = window_record
        print(name, "done")
    summary = pd.DataFrame(summary_rows)
    record["cross_month"] = {}
    for arm in arms:
        for basis in ("raw", "neutralized"):
            part = summary[(summary.arm == arm) & (summary.basis == basis)]
            record["cross_month"].setdefault(basis, {})[arm] = {
                "months": int(len(part)),
                "rank_ic_mean": float(part.rank_ic_mean.mean()),
                "rank_ic_min": float(part.rank_ic_mean.min()),
                "rank_ic_positive_months": int((part.rank_ic_mean > 0).sum()),
                "long_short_sharpe_mean": float(part.long_short_sharpe.mean()),
                "turnover_mean": float(part.turnover_mean.mean()),
                "rank_ic_ir_mean": float(part.rank_ic_ir.mean()),
            }
    for left, right in (("blend_tlob_mlp_50_50", "tlob_3seed"),
                        ("patch_3seed", "tlob_3seed"), ("patch_3seed", "mlp_3seed"),
                        ("blend_all_equal", "tlob_3seed"), ("blend_tlob_patch_50_50", "tlob_3seed"),
                        ("blend_mlp_patch_50_50", "mlp_3seed"), ("blend_all_equal", "patch_3seed")):
        if left not in arms or right not in arms:
            continue
        for window in plan["windows"]:
            name = window["name"]
            left_daily = next(row for row in daily_rows
                              if row.window.iloc[0] == name and row.arm.iloc[0] == left
                              and row.basis.iloc[0] == "raw").set_index("date")
            right_daily = next(row for row in daily_rows
                               if row.window.iloc[0] == name and row.arm.iloc[0] == right
                               and row.basis.iloc[0] == "raw").set_index("date")
            common = left_daily.index.intersection(right_daily.index)
            delta = (left_daily.loc[common, "rank_ic"] - right_daily.loc[common, "rank_ic"])
            entry = block_interval(delta, block=BLOCK_DAYS, draws=BOOTSTRAP_DRAWS,
                                   seed=BOOTSTRAP_SEED)
            record["paired"].setdefault(f"{left}_minus_{right}", {})[name] = entry
            paired_rows.append({"pair": f"{left}_minus_{right}", "window": name, **entry})
    record["paired_definition"] = {"block_days": BLOCK_DAYS, "draws": BOOTSTRAP_DRAWS,
                                   "seed": BOOTSTRAP_SEED, "metric": "daily rank_ic"}
    record["stress"] = {row["window"]: {} for row in stress_rows}
    for row in stress_rows:
        record["stress"][row["window"]].setdefault(row["regime"], {})[row["arm"]] = {
            "days": row["days"], "rank_ic_mean": row["rank_ic_mean"],
            "long_short_return_mean": row["long_short_return_mean"]}
    out = args.record.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, record)
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(
        out.with_name(out.stem + "_daily.csv"), index=False)
    pd.DataFrame(stress_rows).to_csv(out.with_name(out.stem + "_stress.csv"), index=False)
    pd.DataFrame(paired_rows).to_csv(out.with_name(out.stem + "_paired.csv"), index=False)
    print("\nCross-month raw RankIC (mean / min / positive months / Sharpe / turnover):")
    for arm in arms:
        row = record["cross_month"]["raw"][arm]
        print(f"{arm:28s} {row['rank_ic_mean']:+.6f} {row['rank_ic_min']:+.6f} "
              f"{row['rank_ic_positive_months']}/{row['months']} "
              f"{row['long_short_sharpe_mean']:.2f} {row['turnover_mean']:.4f}")
    print("\nCross-month neutralized RankIC:")
    for arm in arms:
        row = record["cross_month"]["neutralized"][arm]
        print(f"{arm:28s} {row['rank_ic_mean']:+.6f} {row['rank_ic_min']:+.6f} "
              f"{row['rank_ic_positive_months']}/{row['months']} "
              f"{row['long_short_sharpe_mean']:.2f} {row['turnover_mean']:.4f}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
