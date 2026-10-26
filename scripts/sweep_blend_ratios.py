"""Sweep fixed TLOB/MLPLOB blend ratios on the three replayed months.

The blend is linear in restored return scores, so every ratio can be scored from
the six frozen member predictions without new inference. The sweep reports each
metric curve, a local equal-weight percentile composite and per-month optima.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from compare_blend_replay import ARMS, arm_scores, load_conditions, load_window, monthly_metrics  # noqa: E402

from jinyiquant.io import read_json, utc_now, write_json  # noqa: E402

METRICS = (
    ("rank_ic_mean", True),
    ("ic_mean", True),
    ("ic_ir", True),
    ("long_short_sharpe", True),
    ("turnover_mean", False),
    ("stress_high_vol_rank_ic", True),
)
COMPOSITE = ("ic_mean", "ic_ir", "long_short_sharpe", "stress_high_vol_rank_ic", "turnover_mean")


def grid(step: float = 0.05) -> list[float]:
    count = int(round(1.0 / step))
    return [round(index * step, 4) for index in range(count + 1)]


def month_rows(frames: dict, targets: pd.DataFrame, conditions: pd.DataFrame, tlob, mlp,
               weights: list[float]) -> tuple[list[dict], list[pd.DataFrame]]:
    rows, dailies = [], []
    for weight in weights:
        predictions = frames["tlob_s11"][["date", "instrument"]].assign(
            score=weight * tlob.score.to_numpy() + (1.0 - weight) * mlp.score.to_numpy())
        metrics, daily, _ = monthly_metrics(predictions, targets)
        daily = daily.copy()
        daily.insert(0, "weight_tlob", weight)
        dailies.append(daily)
        ic = daily.ic
        dated = conditions.assign(date=conditions.day.dt.strftime("%Y-%m-%d"))
        merged = daily.merge(dated[["date", "vol_ratio", "amount_ratio"]], on="date", how="left")
        high_vol = merged.loc[merged.vol_ratio >= 1.0, "rank_ic"]
        high_amount = merged.loc[merged.amount_ratio >= 1.0, "rank_ic"]
        rows.append({
            "weight_tlob": weight,
            "rank_ic_mean": metrics["rank_ic_mean"],
            "ic_mean": metrics["ic_mean"],
            "rank_ic_ir": metrics["rank_ic_ir"],
            "ic_ir": float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else np.nan,
            "long_short_sharpe": metrics["long_short_sharpe"],
            "long_short_return_mean": metrics["long_short_return_mean"],
            "min_daily_long_short_return": metrics["min_daily_long_short_return"],
            "turnover_mean": metrics["turnover_mean"],
            "stress_high_vol_rank_ic": float(high_vol.mean()) if len(high_vol) else np.nan,
            "stress_high_amount_rank_ic": float(high_amount.mean()) if len(high_amount) else np.nan,
            "high_vol_days": int(len(high_vol)),
        })
    return rows, dailies


def add_composite(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for metric, higher_better in METRICS:
        if metric not in COMPOSITE:
            continue
        values = frame[metric] if higher_better else -frame[metric]
        frame[f"{metric}_pct"] = values.rank(pct=True)
    frame["composite"] = frame[[f"{metric}_pct" for metric in COMPOSITE]].mean(axis=1)
    return frame


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--record", type=Path, default=ROOT / "modified_tlob/tlob_mlplob_ratio_sweep_20261026.json")
    args = parser.parse_args(argv)
    plan = read_json(args.plan)
    plan["_path"] = str(args.plan.resolve())
    run_dir = args.run_dir.resolve()
    weights = grid(args.step)
    monthly, daily_rows = {}, []
    for window in plan["windows"]:
        name = window["name"]
        keys, targets, frames = load_window(run_dir, name)
        conditions = load_conditions(plan, window, set(keys.instrument.unique()))
        tlob = arm_scores(frames, ARMS["tlob_3seed"])
        mlp = arm_scores(frames, ARMS["mlp_3seed"])
        rows, dailies = month_rows(frames, targets, conditions, tlob, mlp, weights)
        frame = pd.DataFrame(rows)
        frame.insert(0, "window", name)
        frame = add_composite(frame)
        monthly[name] = frame
        for daily in dailies:
            daily.insert(0, "window", name)
            daily_rows.append(daily)
        print(f"[{name}] {len(weights)} ratios evaluated", flush=True)
    stacked = pd.concat(monthly.values(), ignore_index=True)
    pooled = stacked.groupby("weight_tlob")[
        [metric for metric, _ in METRICS] + ["rank_ic_ir", "min_daily_long_short_return"]].mean()
    pooled["composite"] = stacked.groupby("weight_tlob")["composite"].mean()
    best = {
        "composite": float(pooled.composite.idxmax()),
        "rank_ic_mean": float(pooled.rank_ic_mean.idxmax()),
        "ic_mean": float(pooled.ic_mean.idxmax()),
        "long_short_sharpe": float(pooled.long_short_sharpe.idxmax()),
        "turnover_mean": float(pooled.turnover_mean.idxmin()),
        "stress_high_vol_rank_ic": float(pooled.stress_high_vol_rank_ic.idxmax()),
    }
    per_month_best = {
        name: float(frame.loc[frame.composite.idxmax(), "weight_tlob"])
        for name, frame in monthly.items()}
    report = run_dir / "report" / "blend_comparison"
    report.mkdir(parents=True, exist_ok=True)
    stacked.to_csv(report / "ratio_sweep.csv", index=False)
    pooled.to_csv(report / "ratio_sweep_pooled.csv")
    pd.concat(daily_rows, ignore_index=True).to_csv(report / "ratio_sweep_daily.csv", index=False)

    # Composite curve uncertainty: per-window block interval of the composite-vs-endpoint gap
    # is not meaningful for a rank statistic, so report the per-window peak dispersion instead.
    record = {
        "schema_version": 1,
        "experiment_id": "tlob-mlplob-ratio-sweep-20261026",
        "created_at_utc": utc_now(),
        "scope": "Fixed-ratio TLOB/MLPLOB score sweep on the three replayed months; no training or refitting.",
        "grid_step": args.step,
        "grid": weights,
        "metric_direction": {metric: ("higher" if higher else "lower") for metric, higher in METRICS},
        "composite": {
            "members": list(COMPOSITE),
            "normalization": "within-month percentile rank across the sweep grid, equal weight",
            "note": "local proxy for the platform's five-metric percentile score; not a platform score",
        },
        "pooled": pooled.round(8).reset_index().to_dict(orient="records"),
        "best": best,
        "per_month_best_composite": per_month_best,
        "monthly": {name: frame.round(8).to_dict(orient="records") for name, frame in monthly.items()},
        "evidence_boundary": (
            "Three already-used development months; a dense ratio sweep makes the optimum a "
            "descriptive peak, not an independent confirmation. Composite uses a local metric "
            "proxy; platform stress/turnover definitions are unknown."),
    }
    write_json(args.record, record)

    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5), dpi=140)
    for axis, metric, title in (
            (axes[0, 0], "rank_ic_mean", "Pooled RankIC"),
            (axes[0, 1], "long_short_sharpe", "Pooled long-short Sharpe"),
            (axes[1, 0], "turnover_mean", "Pooled turnover (lower better)"),
            (axes[1, 1], "composite", "Composite percentile (local proxy)")):
        values = pooled[metric].to_numpy()
        axis.plot(pooled.index.to_numpy(), values, marker="o", markersize=3)
        axis.set_xlabel("TLOB weight (rest MLPLOB)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        if metric == "composite":
            peak = pooled[metric].idxmax()
            axis.axvline(peak, color="tab:red", linestyle="--", linewidth=1)
            axis.annotate(f"peak {peak:.2f}", (peak, pooled[metric].max()),
                          textcoords="offset points", xytext=(4, 6), color="tab:red")
            for name, frame in monthly.items():
                axis.plot(frame.weight_tlob, frame.composite, linestyle=":", linewidth=1, alpha=0.7,
                          label=name)
            axis.legend(fontsize=7)
    figure.suptitle("TLOB/MLPLOB blend ratio sweep, three development months")
    figure.tight_layout()
    figure.savefig(report / "ratio_sweep.png")
    plt.close(figure)

    print(pooled.round(5).to_string())
    print("best:", best)
    print("per-month composite peak:", per_month_best)
    print(f"record: {args.record}")
    print(f"figure: {report / 'ratio_sweep.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
