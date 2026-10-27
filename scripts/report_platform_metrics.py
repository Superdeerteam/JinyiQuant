"""Report the platform-style five metrics for the current nine-model equal blend.

The platform score is an equal-weight mean of five rank percentiles: IC mean,
IC_IR, long-short Sharpe, stress stability and turnover (lower is better). The
platform field is not observable locally, so this script reports the five raw
metrics under the frozen local approximation and a *local* percentile proxy over
a fixed 16-arm panel (9 frozen checkpoints + 7 pre-declared equal-weight blends).
The proxy is not the official score and must not be quoted as one.

LEGACY CALIBER (stage A): the percentile composite is *descriptive only*. It is
retired as a promotion/elimination input; ``composite`` rejects every use other
than ``"description"`` and ``promotion_input_guard`` raises ``LegacyCaliberError``
for promotion/elimination requests. The written record carries
``promotion_allowed=false`` and the legacy mark. See ``runs/legacy_caliber.json``.

Read-only over archived replay predictions; no training, no weight fitting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jinyiquant.comparison import declared_cost_scenarios  # noqa: E402
from jinyiquant.config import EvaluationConfig  # noqa: E402
from jinyiquant.evaluation import EVALUATION_VERSION, aggregate_metric  # noqa: E402
from scripts.compare_multiscale_replay import (  # noqa: E402
    SAMPLE_TIMES,
    arm_scores,
    define_arms,
    evaluate_scores,
    load_exposure,
    load_window,
    merged_scores,
    neutralize,
)

SCREEN_RUN = ROOT / "runs/multiscale-replay-20261026/20261026T133118.869042Z_bcdc779dc66b"
VERIFY_RUN = ROOT / "runs/multiscale-replay-verify-20261026/20261026T140057.561539Z_1a57d4b38acc"
LABELS = ROOT / "数据/cpt_jyc_2026_vwap"
EXPOSURE = [ROOT / f"数据/cpt_jyc_2026_exposure/cpt_jyc_2026_exposure_{y}_part_0001.parquet"
            for y in (2023, 2024)]
SPEC = ROOT / "runs/submission-best-20261026/build/spec.json"
OUT = ROOT / "modified_tlob/round5_platform_five_20261026.json"
TARGET = 0.85
CFG = EvaluationConfig(quantile=0.2, periods_per_year=252, min_cross_section=3,
                       constant_portfolios="undefined")
BLEND = "blend_all_equal"
COMPONENTS = ("ic", "ic_ir", "sharpe", "stress", "turnover")
HIGHER_IS_BETTER = {"ic": True, "ic_ir": True, "sharpe": True, "stress": True, "turnover": False}
BLOCK_DAYS = 5
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20261026
LEGACY_CALIBER = {
    "caliber_id": "legacy-pre-eval-v2",
    "status": "descriptive_only",
    "promotion_allowed": False,
    "legacy_mark": "旧口径结论——不得用于晋升",
}
DESCRIPTIVE_USES = frozenset({"description", "diagnostic", "report"})


class LegacyCaliberError(RuntimeError):
    """Raised when the retired percentile composite is used for promotion/elimination."""


def promotion_input_guard(use: str) -> dict:
    """Allow descriptive reads of the percentile panel and refuse promotion-shaped ones.

    Any use that is not explicitly descriptive is rejected: the 16-arm percentile
    composite is a legacy caliber conclusion and cannot authorise promotion or
    elimination.
    """
    normalized = use.strip().lower()
    if normalized not in DESCRIPTIVE_USES:
        raise LegacyCaliberError(
            f"{use!r} is not a descriptive use: 16-arm percentile composite is a legacy caliber "
            "conclusion (旧口径结论——不得用于晋升); promotion_allowed=false")
    return {"use": normalized, **LEGACY_CALIBER}


def legacy_record_block() -> dict:
    """Return the flags every written report record must carry."""
    return {**LEGACY_CALIBER, "percentile_panel": "16-arm fixed panel",
            "replacement_contract": "eval-v2 (pending)"}


def run_windows() -> list[tuple[Path, dict]]:
    out = []
    for run in (SCREEN_RUN, VERIFY_RUN):
        plan = json.load(open(run / "plan.json", encoding="utf-8"))
        out.append((run, plan))
    return out


def stress_conditions(start: pd.Timestamp, end: pd.Timestamp, instruments: set[str]) -> pd.DataFrame:
    begin = start - pd.Timedelta(days=70)
    frames = []
    for month in pd.period_range(begin, end, freq="M"):
        path = LABELS / f"{LABELS.name}_{month.strftime('%Y-%m')}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path, columns=["date", "instrument", "vwap_return", "amount"],
                                          filters=[("date", ">=", begin), ("date", "<", end)]))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame.date.dt.strftime("%H:%M").isin(SAMPLE_TIMES)]
    frame = frame.loc[frame.instrument.isin(instruments)]
    by_time = frame.groupby("date", sort=True).agg(market_return=("vwap_return", "mean"),
                                                   amount=("amount", "sum"))
    by_day = by_time.groupby(by_time.index.normalize()).agg(
        market_vol=("market_return", "std"), amount=("amount", "sum"))
    by_day["vol_ratio"] = by_day.market_vol / by_day.market_vol.shift(1).rolling(20, min_periods=10).median()
    by_day["amount_ratio"] = by_day.amount / by_day.amount.shift(1).rolling(20, min_periods=10).median()
    by_day.index.name = "date"
    return by_day.reset_index()


def five(daily: pd.DataFrame) -> dict:
    """Aggregate daily rows with the single contract aggregator (no local formulas)."""
    return {
        "ic": aggregate_metric(daily.ic.to_numpy(dtype=float), "mean"),
        "ic_ir": aggregate_metric(daily.ic.to_numpy(dtype=float), "ir"),
        "sharpe": aggregate_metric(daily.long_short_return.to_numpy(dtype=float), "sharpe"),
        "turnover": aggregate_metric(daily.turnover.to_numpy(dtype=float), "mean"),
        "rank_ic": aggregate_metric(daily.rank_ic.to_numpy(dtype=float), "mean"),
        "days": int(len(daily)),
        "turnover_conversion_factor": 0.5,
        "evaluation_version": EVALUATION_VERSION,
    }


def stress_buckets(daily: pd.DataFrame, cond: pd.DataFrame) -> dict:
    left = daily[["date", "rank_ic"]].copy()
    left["date"] = pd.to_datetime(left.date).dt.normalize()
    merged = left.merge(cond[["date", "vol_ratio", "amount_ratio"]], on="date", how="inner")
    if len(merged) != len(left):
        raise ValueError(f"stress conditions cover {len(merged)} of {len(left)} days")
    masks = {
        "high_vol": merged.vol_ratio >= 1.0,
        "low_vol": merged.vol_ratio < 1.0,
        "high_amount": merged.amount_ratio >= 1.0,
        "low_amount": merged.amount_ratio < 1.0,
    }
    return {name: {"days": int(mask.sum()), "rank_ic_mean": float(merged.loc[mask, "rank_ic"].mean())}
            for name, mask in masks.items()}


def pooled_stress(per_window: dict) -> dict:
    num = {b: 0.0 for b in ("high_vol", "low_vol", "high_amount", "low_amount")}
    den = {b: 0 for b in num}
    for buckets in per_window.values():
        for name, value in buckets.items():
            num[name] += value["rank_ic_mean"] * value["days"]
            den[name] += value["days"]
    pooled = {name: num[name] / den[name] for name in num}
    pooled["stress"] = min(pooled["high_vol"], pooled["high_amount"])
    return pooled


def percentile(values: np.ndarray, higher: bool) -> list[float]:
    out = []
    for value in values:
        worse = np.sum(values < value) if higher else np.sum(values > value)
        ties = int(np.sum(values == value)) - 1
        out.append((worse + 0.5 * ties) / (len(values) - 1))
    return out


def composite(frame: pd.DataFrame, *, use: str = "description") -> pd.DataFrame:
    """Add the five percentiles and the equal-weight composite, for description only.

    ``use`` exists to make the retired promotion path explicit: anything other
    than a descriptive use raises ``LegacyCaliberError``.
    """
    if use.strip().lower() not in DESCRIPTIVE_USES:
        raise LegacyCaliberError(
            f"{use!r}: the 16-arm percentile composite is descriptive only "
            "(旧口径结论——不得用于晋升); promotion input is not allowed")
    for name in COMPONENTS:
        frame[f"p_{name}"] = percentile(frame[name].to_numpy(dtype=float), HIGHER_IS_BETTER[name])
    frame["composite"] = frame[[f"p_{name}" for name in COMPONENTS]].mean(axis=1)
    frame["gap_to_target"] = TARGET - frame["composite"]
    for name in COMPONENTS:
        frame[f"gap_contrib_{name}"] = (TARGET - frame[f"p_{name}"]) / len(COMPONENTS)
    return frame




def main() -> int:
    exposure_frame, exposure_by_date, exposure_dates = load_exposure({
        "_path": str(SPEC), "inputs": {"exposure": [str(p) for p in EXPOSURE]}})
    record = {
        "schema_version": 1,
        "experiment_id": "round5-platform-five-20261026",
        "evaluation_version": EVALUATION_VERSION,
        "target_score": TARGET,
        "metric_scope": "local approximation, not the official platform score",
        "legacy_caliber": legacy_record_block(),
        "promotion_allowed": False,
        "evaluation_contract": {
            "source": "src/jinyiquant/evaluation.py:evaluate",
            "turnover_conversion_factor": 0.5,
            "note": "All competition and economic values below come from the single eval-v2 contract. "
                    "The retired percentile proxy stays in the legacy_caliber block, descriptive only.",
        },
        "definitions": {
            "ic": "mean daily Pearson IC of the winsorized/z-scored/style-residualized score",
            "ic_ir": "daily IC mean / daily IC std (ddof=1); not annualized",
            "sharpe": "daily long-short mean/std * sqrt(252); gross, no fees or impact",
            "stress": "min(pooled high-volatility, pooled high-amount) rank IC, neutralized score",
            "turnover": "eval-v2 turnover = U / 2 with conversion factor 0.5; "
                        "U = sum(|target - pre-trade drifted|); lower is better",
            "percentile": "(worse + 0.5*ties) / (n-1) inside the frozen 16-arm panel",
            "composite": "equal-weight mean of the five component percentiles",
        },
        "neutralization": {
            "exposure": "previous trading day cpt_jyc_2026_exposure snapshot strictly before the signal day",
            "transform": "median +/- 3*1.4826*MAD winsorization, cross-section z-score, equal-weight OLS on 10 style factors and 32 industry dummies",
            "note": "local approximation of the platform BARRA residualization; official details unknown",
        },
        "panel": {
            "definition": "9 frozen checkpoints (TLOB/MLPLOB/dual_patch x seeds 11/29/47) + 7 pre-declared equal-weight combinations",
            "arms": None,
        },
        "windows": {},
        "arms": {},
        "baseline": {},
        "proxy": {},
        "evidence_boundary": "",
    }
    pooled_daily = {"raw": {}, "neutralized": {}}
    pooled_five = {"raw": {}, "neutralized": {}}
    pooled_scored = {"raw": [], "neutralized": []}
    pooled_targets = []
    stress_per_window = {}

    for run, plan in run_windows():
        groups = {"tlob": [], "mlp": [], "patch": []}
        for version in plan["versions"]:
            for prefix in groups:
                if version["name"].startswith(prefix + "_"):
                    groups[prefix].append(version["name"])
        arms = define_arms(groups)
        for members in groups.values():
            for member in members:
                arms[member] = {member: 1.0}
        record["panel"]["arms"] = sorted(arms)
        for window in plan["windows"]:
            name = window["name"]
            keys, targets, frames = load_window(run, name, [m for members in groups.values() for m in members])
            cond = stress_conditions(pd.Timestamp(window["start"]), pd.Timestamp(window["end"]),
                                     set(keys.instrument.unique()))
            record["windows"][name] = {
                "start": window["start"], "end": window["end"],
                "days": int(((cond.date >= pd.Timestamp(window["start"]))
                             & (cond.date < pd.Timestamp(window["end"]))).sum())}
            print(f"[{name}] {len(arms)} arms, {len(cond)} days", flush=True)
            for arm, weights in arms.items():
                scores = arm_scores(frames, weights)
                raw, daily_raw, _ = evaluate_scores(scores, targets, CFG, "score")
                neu_scores = neutralize(scores, exposure_by_date, exposure_dates, quantile=0.2)
                neu, daily_neu, _ = evaluate_scores(neu_scores, targets, CFG, "processed_score")
                if arm == BLEND:
                    raw_scored, answer = merged_scores(scores, targets, "score")
                    neu_scored, _ = merged_scores(neu_scores, targets, "processed_score")
                    pooled_scored["raw"].append(raw_scored)
                    pooled_scored["neutralized"].append(neu_scored)
                    pooled_targets.append(answer)
                buckets = stress_buckets(daily_neu, cond)
                stress_per_window.setdefault(arm, {})[name] = buckets
                record["arms"].setdefault(arm, {})[name] = {"raw": raw, "neutralized": neu,
                                                           "stress": buckets}
                for tag, daily in (("raw", daily_raw), ("neutralized", daily_neu)):
                    part = daily[["date", "ic", "rank_ic", "long_short_return", "turnover"]].copy()
                    part["window"] = name
                    pooled_daily[tag].setdefault(arm, []).append(part)
                print(f"  {arm:<22} neu IC={neu['ic_mean']:.4f} ICIR={neu['ic_ir']:.2f} "
                      f"SR={neu['long_short_sharpe']:5.1f} TO={neu['turnover_mean']:.3f}", flush=True)

    for tag in ("raw", "neutralized"):
        for arm, parts in pooled_daily[tag].items():
            pooled_five[tag][arm] = five(pd.concat(parts, ignore_index=True))

    frame = pd.DataFrame([
        {"arm": arm, **{name: pooled_five["neutralized"][arm][name] for name in COMPONENTS if name != "stress"},
         "stress_level": pooled_stress(stress_per_window[arm])["stress"],
         "rank_ic": pooled_five["neutralized"][arm]["rank_ic"],
         "days": pooled_five["neutralized"][arm]["days"]}
        for arm in pooled_five["neutralized"]
    ]).set_index("arm")
    frame["stress"] = frame["stress_level"]
    frame["stress_ratio"] = frame["stress_level"] / frame["rank_ic"]
    scored = composite(frame.copy())
    combos = [arm for arm in scored.index if arm.endswith("_3seed") or arm.startswith("blend_")]
    local = composite(frame.loc[combos].copy())
    record["baseline"] = {
        "experiment_id": "best-three-trunk-equal-20261026",
        "weights": {name: 1 / 9 for name in sorted(
            m["name"] for m in json.load(open(SPEC, encoding="utf-8"))["members"])},
        "five": {name: float(scored.loc[BLEND, name]) for name in COMPONENTS},
        "stress_buckets": pooled_stress(stress_per_window[BLEND]),
        "stress_ratio": float(scored.loc[BLEND, "stress_ratio"]),
        "rank_ic": float(scored.loc[BLEND, "rank_ic"]),
        "days": int(scored.loc[BLEND, "days"]),
    }
    record["proxy"] = {
        "legacy_caliber": legacy_record_block(),
        "panel_16": {
            "composite": float(scored.loc[BLEND, "composite"]),
            "gap_to_target": float(scored.loc[BLEND, "gap_to_target"]),
            "percentiles": {name: float(scored.loc[BLEND, f"p_{name}"]) for name in COMPONENTS},
            "gap_contributions": {name: float(scored.loc[BLEND, f"gap_contrib_{name}"]) for name in COMPONENTS},
        },
        "panel_7": {
            "composite": float(local.loc[BLEND, "composite"]),
            "gap_to_target": float(local.loc[BLEND, "gap_to_target"]),
            "percentiles": {name: float(local.loc[BLEND, f"p_{name}"]) for name in COMPONENTS},
            "gap_contributions": {name: float(local.loc[BLEND, f"gap_contrib_{name}"]) for name in COMPONENTS},
        },
    }
    if pooled_targets:
        pooled_answer = pd.concat(pooled_targets, ignore_index=True)
        contract = record["evaluation_contract"]
        for basis in ("raw", "neutralized"):
            pooled = pd.concat(pooled_scored[basis], ignore_index=True)
            scenario_block = declared_cost_scenarios(
                pooled, pooled_answer, CFG, undefined_constant_portfolios=True)
            contract[basis] = {
                "evaluation_version": scenario_block["evaluation_version"],
                "source": scenario_block["source"],
                "turnover_conversion_factor": scenario_block["turnover_conversion_factor"],
                "comparison_usable": scenario_block["comparison_usable"],
                "comparison_gate_reasons": scenario_block["comparison_gate_reasons"],
                "cost_definition": scenario_block["definition"],
                "competition_metrics": scenario_block["scenarios"]["0bp"]["competition_metrics"],
                "economic_metrics": {name: block["economic_metrics"]
                                     for name, block in scenario_block["scenarios"].items()},
            }
    record["monthly"] = {
        name: {key: float(value) for key, value in
               {"ic": bucket["neutralized"]["ic_mean"], "ic_ir": bucket["neutralized"]["ic_ir"],
                "sharpe": bucket["neutralized"]["long_short_sharpe"],
                "turnover": bucket["neutralized"]["turnover_mean"],
                "high_vol_rank_ic": bucket["stress"]["high_vol"]["rank_ic_mean"],
                "high_amount_rank_ic": bucket["stress"]["high_amount"]["rank_ic_mean"]}.items()}
        for name, bucket in record["arms"][BLEND].items()
    }
    record["ranking_16"] = scored.reset_index()[
        ["arm", *COMPONENTS, "stress_ratio", "composite", "gap_to_target"]].round(6).to_dict(orient="records")
    record["ranking_7"] = local.reset_index()[
        ["arm", *COMPONENTS, "stress_ratio", "composite", "gap_to_target"]].round(6).to_dict(orient="records")
    record["evidence_boundary"] = (
        "The five values are a local approximation of the platform metrics: neutralization, "
        "stress definition and turnover convention are not the official implementation. The "
        "percentile composite uses only the 16 local arms, not the competition field; its level "
        "is not the platform score and the 0.85 gap is a proxy gap. All five development months "
        "(2023-11, 2024-01, 2024-03, 2024-04, 2024-06) are already-used research data; the "
        "2024-10-08..2025-01-01 final interval stays closed.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(record, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT}")
    print(f"baseline five: {json.dumps(record['baseline']['five'], ensure_ascii=False)}")
    for panel in ("panel_16", "panel_7"):
        p = record["proxy"][panel]
        print(f"{panel}: composite={p['composite']:.4f} gap={p['gap_to_target']:+.4f} "
              f"pct={ {k: round(v, 3) for k, v in p['percentiles'].items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
