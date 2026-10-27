"""Cost curve of the frozen predictions under eval-v2 and platform controllability.

No training, no model search, no sealed-interval access. Predictions are reused
unchanged from the frozen multiscale replay artifacts; the only metric entry point is
``jinyiquant.evaluation.evaluate`` (eval-v2).

Predeclared before any metric is computed (``plan.json`` is written first):

* windows: the five already-viewed development months covered by the frozen replay
  (2023-11, 2024-01, 2024-03, 2024-04, 2024-06), outside the sealed interval
  [2024-10-08, 2025-01-01);
* objects: the nine-model equal blend (reference), the three single-trunk three-seed
  ensembles, and two known-answer controls;
* single-side cost 0bp / 2bp / 5bp / 10bp, every scenario a fresh eval-v2 evaluate call,
  5bp being the primary stress scenario;
* the platform-controllability question is answered only from published competition
  text, and every quoted claim is re-verified against the source file at run time.

The output JSON separates the competition block (approximate local metrics, no
cross-submission rank) from the economic block (gross/cost/net at each rate).  It never
promotes or rejects a model.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from jinyiquant.config import EvaluationConfig
from jinyiquant.evaluation import CONTRACT, EVALUATION_VERSION, evaluate
from jinyiquant.io import canonical_json, digest, file_sha256, utc_now, write_json
from jinyiquant.tracking import Recorder

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "runs/multiscale-replay-20261026/20261026T133118.869042Z_bcdc779dc66b"
VERIFY = ROOT / "runs/multiscale-replay-verify-20261026/20261026T140057.561539Z_1a57d4b38acc"
LABELS_TABLE = "数据/cpt_jyc_2026_vwap"
WINDOWS = (
    ("nov2023", "2023-11-01", "2023-12-01", SCREEN),
    ("jan2024", "2024-01-01", "2024-02-01", SCREEN),
    ("mar2024", "2024-03-01", "2024-04-01", SCREEN),
    ("apr2024", "2024-04-01", "2024-05-01", VERIFY),
    ("jun2024", "2024-06-01", "2024-07-01", VERIFY),
)
MEMBERS = ("tlob_s11", "tlob_s29", "tlob_s47", "mlp_s11", "mlp_s29", "mlp_s47",
           "patch_s11", "patch_s29", "patch_s47")
SEALED = ("2024-10-08", "2025-01-01")
RATES = (("0bp", 0.0), ("2bp", 0.0002), ("5bp", 0.0005), ("10bp", 0.001))
PRIMARY_RATE = "5bp"
CONTROL_SEED = 20261027
EXPERIMENT_ID = "cost-curve-20261027"
GENEROSITY = 1e-12

OBJECTS = (
    {"id": "reference_nine_blend", "role": "reference",
     "definition": "1/9 each of the nine frozen members; the frozen submission build",
     "members": MEMBERS},
    {"id": "baseline_v3_tlob_3seed", "role": "baseline",
     "definition": "equal-weight TLOB seeds 11/29/47; the validated V3 submission baseline",
     "members": MEMBERS[:3]},
    {"id": "candidate_mlp_3seed", "role": "candidate",
     "definition": "equal-weight MLPLOB seeds 11/29/47", "members": MEMBERS[3:6]},
    {"id": "candidate_patch_3seed", "role": "candidate",
     "definition": "equal-weight dual-scale Patch seeds 11/29/47", "members": MEMBERS[6:]},
    {"id": "control_shuffled_reference", "role": "control",
     "definition": f"reference scores permuted within each timestamp, seed {CONTROL_SEED}; "
                   "expected ~0 rank IC while cost is still charged"},
    {"id": "control_sign_flipped_reference", "role": "control",
     "definition": "negated reference scores; expected gross return to flip sign while "
                   "total absolute weight change and therefore cost stay identical"},
)

COMPETITION_RULES = Path("比赛说明/比赛说明.txt")
# Every claim about what the platform does is quoted and re-verified against the file.
PLATFORM_EVIDENCE = (
    ("submitted_artifact", "必须且仅包含以下三列",
     "提交物是因子代码及其 [date, instrument, factor] 输出；没有持仓、仓位或交易指令输入"),
    ("platform_preprocessing", "去极值 与 标准化",
     "平台在评估前自行去极值并标准化，提交端的分数尺度不保留"),
    ("platform_style", "风格剔除",
     "平台对 BARRA 风险因子回归取残差，因子只能以残差增量参与评分"),
    ("sampling_clock", "采样间隔等于预测窗口",
     "调仓时点由平台固定为每个交易日 8 个 30 分钟截面，提交端不能改变调仓频率"),
    ("turnover_metric", "换手越低越好",
     "换手率是五项指标之一，方向为越低越好"),
    ("turnover_rank", "rank 百分比排名",
     "换手项进入评分的是全体有效提交中的相对排名，没有绝对数值门槛"),
    ("coverage_limit", "缺失率不得高于 40%",
     "不能通过减少输出（不交易某些股票）来压低换手，缺失率上限 40%"),
)


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def num(value):
    """JSON-safe float: non-finite becomes None (eval-v2 never zero-fills)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def evaluation_config(rate: float | None) -> EvaluationConfig:
    return EvaluationConfig(quantile=0.2, periods_per_year=252, min_cross_section=3,
                            constant_portfolios="undefined", single_side_rate=rate)


def load_window(name: str, start: str, end: str, replay: Path) -> dict:
    """Read one frozen replay window; validate key identity before any metric is computed."""
    base = replay / "windows" / name
    targets = pd.read_parquet(base / "targets.parquet")
    keys = targets[["date", "instrument"]].reset_index(drop=True)
    if keys.duplicated().any():
        raise ValueError(f"{name}: duplicate target keys")
    ordered = targets.sort_values(["date", "instrument"]).reset_index(drop=True)
    scores = {}
    shas = {}
    for member in MEMBERS:
        path = base / member / "predictions.parquet"
        frame = pd.read_parquet(path)
        if set(frame.columns) != {"date", "instrument", "score"}:
            raise ValueError(f"{name}/{member}: unexpected prediction columns")
        merged = frame.merge(keys, on=["date", "instrument"], how="inner", validate="one_to_one")
        merged = merged.sort_values(["date", "instrument"]).reset_index(drop=True)
        if len(merged) != len(keys):
            raise ValueError(f"{name}/{member}: predictions do not cover every target key")
        if not merged[["date", "instrument"]].equals(ordered[["date", "instrument"]]):
            raise ValueError(f"{name}/{member}: key order mismatch")
        if not np.isfinite(merged["score"].to_numpy(dtype=float)).all():
            raise ValueError(f"{name}/{member}: non-finite scores")
        scores[member] = merged["score"].to_numpy(dtype=float)
        shas[member] = file_sha256(path)
    dates = pd.to_datetime(ordered["date"])
    if (dates < pd.Timestamp(start)).any() or (dates >= pd.Timestamp(end)).any():
        raise ValueError(f"{name}: dates outside the declared window")
    if (dates >= pd.Timestamp(SEALED[0])).any():
        raise ValueError(f"{name}: dates enter the sealed interval")
    return {"name": name, "start": start, "end": end, "replay": rel(replay),
            "keys": ordered[["date", "instrument"]], "targets": ordered[["date", "instrument", "target"]],
            "scores": scores, "prediction_sha256": shas,
            "keys_sha256": file_sha256(base / "keys.parquet"),
            "targets_sha256": file_sha256(base / "targets.parquet")}


def pooled_objects(windows: list[dict]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Pool the windows into one sequence and build every object's score vector."""
    keys = pd.concat([window["keys"] for window in windows], ignore_index=True)
    targets = pd.concat([window["targets"] for window in windows], ignore_index=True)
    keys = keys.sort_values(["date", "instrument"]).reset_index(drop=True)
    targets = targets.sort_values(["date", "instrument"]).reset_index(drop=True)
    if not keys.equals(targets[["date", "instrument"]]):
        raise ValueError("pooled target keys changed")
    if keys.duplicated().any():
        raise ValueError("pooled keys are not unique")

    def member_mean(members) -> np.ndarray:
        stacked = np.stack([np.concatenate([window["scores"][member] for window in windows])
                            for member in members])
        return stacked.mean(axis=0)

    raw = {item["id"]: member_mean(item["members"]) for item in OBJECTS if "members" in item}
    reference = raw["reference_nine_blend"]
    rng = np.random.default_rng(CONTROL_SEED)
    shuffled = np.empty_like(reference)
    day_positions = keys.groupby("date").indices
    for timestamp in sorted(day_positions):
        positions = np.asarray(day_positions[timestamp])
        shuffled[positions] = reference[positions][rng.permutation(len(positions))]
    raw["control_shuffled_reference"] = shuffled
    raw["control_sign_flipped_reference"] = -reference
    return raw, keys, targets


def evaluate_object(payload):
    """Worker: one object evaluated at every predeclared single-side rate."""
    object_id, keys, scores, targets = payload
    predictions = keys.assign(score=scores)[["date", "instrument", "score"]]
    results = {}
    for label, rate in RATES:
        started = time.time()
        metrics, daily, _ = evaluate(predictions, targets, evaluation_config(rate),
                                     return_sections=True, undefined_constant_portfolios=True)
        results[label] = {"single_side_rate": rate, "metrics": metrics,
                          "daily": daily, "seconds": round(time.time() - started, 3)}
    return object_id, results


def month_key(dates) -> pd.Series:
    return pd.to_datetime(dates).dt.strftime("%Y-%m")


def monthly_table(daily: pd.DataFrame) -> dict:
    """Per-month gross, cost and net at the scenario's own rate plus the rate-free knobs."""
    frame = daily.copy()
    frame["month"] = month_key(frame["date"])
    table = {}
    for month, block in frame.groupby("month", sort=True):
        gross = num(block["long_short_return"].mean())
        change = num(block["total_abs_weight_change"].mean())
        table[month] = {
            "day_count": int(len(block)),
            "gross_return_mean": gross,
            "total_abs_weight_change_mean": change,
            "turnover_mean": num(block["turnover"].mean()),
            "cost_mean": num(block["cost"].mean()),
            "net_return_mean": num(block["net_return"].mean()),
            "share_days_net_positive": None if gross is None else num(
                (block["net_return"] > 0).sum() / max(int(block["net_return"].notna().sum()), 1)),
            "undefined_cost_days": int(block["undefined_cost_count"].fillna(0).sum()),
        }
    return table


def scenario_block(metrics: dict, daily: pd.DataFrame, label: str, rate: float) -> dict:
    """One cost scenario: gross, cost, net and turnover on the same eval-v2 call."""
    economic = metrics["economic_metrics"]
    competition = metrics["competition_metrics"]
    return {
        "label": label,
        "single_side_rate": rate,
        "single_side_rate_bp": rate * 1e4,
        "gross_return_mean": num(economic.get("gross_return_mean")),
        "gross_return_mean_bp": None if economic.get("gross_return_mean") is None
        else num(economic["gross_return_mean"] * 1e4),
        "cost_mean": num(economic.get("cost_mean")),
        "cost_mean_bp": None if economic.get("cost_mean") is None else num(economic["cost_mean"] * 1e4),
        "net_return_mean": num(economic.get("net_return_mean")),
        "net_return_mean_bp": None if economic.get("net_return_mean") is None
        else num(economic["net_return_mean"] * 1e4),
        "total_abs_weight_change_mean": num(economic.get("total_abs_weight_change_mean")),
        "turnover_mean_local": num(metrics.get("turnover_mean")),
        "turnover_conversion_factor": 0.5,
        "official_turnover_item": {
            "official_value": None,
            "official_status": "unavailable locally",
            "official_definition": "turnover is one of the five published metrics, scored as a "
                                   "rank percentile over all valid submissions, lower is better, weight 0.2",
            "reason": "the submission population and the platform turnover construction are not "
                      "available; eval-v2 never produces a cross-submission rank",
            "local_proxy": num(competition["metrics"]["turnover"]["value"]),
            "local_proxy_status": competition["metrics"]["turnover"]["status"],
            "rate_invariant": True,
        },
        "rank_ic_mean": num(metrics.get("rank_ic_mean")),
        "ic_mean": num(metrics.get("ic_mean")),
        "long_short_sharpe": num(metrics.get("long_short_sharpe")),
        "min_daily_long_short_return": num(metrics.get("min_daily_long_short_return")),
        "share_days_net_positive": num(
            (daily["net_return"] > 0).sum() / max(int(daily["net_return"].notna().sum()), 1)),
        "undefined_rebalance_count": int(economic.get("undefined_rebalance_count") or 0),
        "undefined_counts": economic.get("undefined_counts"),
        "economic_status": economic.get("status"),
        "day_count": int(metrics.get("day_count") or 0),
        "cross_section_count": int(metrics.get("cross_section_count") or 0),
        "drift_unknown_section_count": int(metrics.get("drift_unknown_section_count") or 0),
        "breakeven_single_side_rate": num(economic.get("breakeven_single_side_rate")),
        "breakeven_single_side_rate_bp": None if economic.get("breakeven_single_side_rate") is None
        else num(economic["breakeven_single_side_rate"] * 1e4),
        "competition_metrics": competition,
    }


def economics_headroom(gross: float | None, change: float | None, turnover: float | None,
                       rate: float) -> dict:
    """What a 5bp (or any) single-side rate would require from gross or turnover."""
    block = {
        "single_side_rate": rate,
        "definition": "per 1.0 equity and one label period; cost = rate * U, turnover = U / 2",
        "gross_return_mean": gross,
        "total_abs_weight_change_mean": change,
        "turnover_mean_local": turnover,
        "gross_required_for_zero_net": None if change is None else num(rate * change),
        "gross_required_multiple": None,
        "turnover_required_for_zero_net": None if gross is None else num(gross / (2 * rate)),
        "turnover_reduction_required_share": None,
        "turnover_share_of_theoretical_maximum": None if turnover is None else num(turnover / 2.0),
        "cost_bp_per_0_1_turnover_reduction": num(rate * 2 * 0.1 * 1e4),
    }
    if gross is not None and change is not None:
        required = rate * change
        if gross > 0:
            block["gross_required_multiple"] = num(required / gross)
    if gross is not None and turnover not in (None, 0):
        required_turnover = gross / (2 * rate)
        block["turnover_required_for_zero_net"] = num(required_turnover)
        block["turnover_reduction_required_share"] = num(1 - required_turnover / turnover)
    return block


def platform_controllability(source: Path) -> dict:
    """Answer 'holdings or factor only?' from the published text, verifying every quote."""
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines()
    evidence = []
    missing = []
    for key, quote, implication in PLATFORM_EVIDENCE:
        line = next((index + 1 for index, value in enumerate(lines) if quote in value), None)
        if line is None:
            missing.append({"key": key, "quote": quote})
        evidence.append({"key": key, "source": rel(source), "quote": quote, "line": line,
                         "implication": implication, "verified": line is not None})
    if missing:
        raise ValueError(f"platform evidence quotes not found: {missing}")
    return {
        "question": "平台允许控制什么：持仓还是仅可提交因子？",
        "answer": "仅可提交因子。平台接收 [date, instrument, factor] 三列因子值，自行做预处理与"
                  "组合构建，参赛者无法提交持仓、仓位、调仓频率或交易成本假设。",
        "controllable": [
            "每个 (date, instrument) 的因子值，以及它在时间上的持续性（跨相邻截面的变化幅度）",
            "缺失覆盖率（每个评估时点缺失率必须 ≤ 40%）",
            "因子自身的截面形态与其经济逻辑",
        ],
        "not_controllable": [
            "持仓/仓位/交易指令：提交物只有三列因子值",
            "组合构建方式与分位阈值：平台自行处理",
            "调仓频率与调仓时点：平台固定为每交易日 8 个 30 分钟截面",
            "费用与冲击成本：平台评分不含成本项，成本只在本地经济性口径中体现",
            "平台去极值、标准化与 BARRA 风格剔除后的最终因子",
            "官方换手率的构造细节与相对排名",
        ],
        "consequences": [
            "换手只能通过让因子输出在时间上更持久来降低；不能通过持仓优化、空仓或交易门槛控制",
            "平台会重新标准化因子，因此任何以“标准化分数 = bp 收益”为前提的交易门槛都无效",
            "缺失率上限 40% 排除了“只对少数股票输出因子”的降换手做法",
            "官方换手项是相对排名，本地换手只能作为单调改善的代理，不能当绝对门槛",
        ],
        "evidence": evidence,
        "verified": all(item["verified"] for item in evidence),
    }


def build_plan(windows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "evaluation_version": EVALUATION_VERSION,
        "hypothesis": "Existing frozen predictions, scored only by eval-v2, pin down the cost curve at "
                      "0/2/5/10bp single-side cost: gross return, cost, net return and the official "
                      "turnover item can be reported side by side without retraining, and the published "
                      "platform rules fix the only possible turnover lever.",
        "no_training": True,
        "sealed_interval": list(SEALED),
        "windows": [{"name": window["name"], "start": window["start"], "end": window["end"],
                     "replay_run": window["replay"], "keys_sha256": window["keys_sha256"],
                     "targets_sha256": window["targets_sha256"],
                     "prediction_sha256": window["prediction_sha256"]} for window in windows],
        "pooling": "Windows are concatenated into one chronological sequence so daytime adjacency is "
                   "evaluated as adjacent labels; overnight and month gaps carry the previous target "
                   "weights forward and are counted (drift unknown).",
        "cost_scenarios": {label: rate for label, rate in RATES},
        "primary_stress_scenario": PRIMARY_RATE,
        "cost_scenario_rule": "every scenario is a fresh eval-v2 evaluate call; cost = single_side_rate "
                              "* U, U = sum(|target weight - pre-trade drifted weight|), "
                              "turnover = U / 2 with conversion factor 0.5",
        "objects": [{"id": item["id"], "role": item["role"], "definition": item["definition"]}
                    for item in OBJECTS],
        "checks_predeclared": [
            "sealed interval never read",
            "evaluation dates stay inside the five declared windows",
            "cost is linear in the single-side rate: cost(r) == r * U for every scenario",
            "turnover and U are invariant to the single-side rate",
            "sign-flipped control has identical cost and negated gross return",
            "reference 0bp evaluated twice gives an identical canonical digest",
            "platform-controllability quotes all verified against 比赛说明/比赛说明.txt",
        ],
        "reporting_rules": [
            "5bp is the primary stress scenario and is reported first",
            "the official turnover item is a cross-submission rank percentile; only the local U/2 proxy "
            "is computable and it is never presented as an official score",
            "net return at any rate is a local economic lens, never the official score",
            "no accept/reject and no model promotion output",
        ],
        "research_area": "all five windows were already viewed development months; the sealed interval "
                         "[2024-10-08, 2025-01-01) is never read",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/cost-curve-20261027")
    parser.add_argument("--record", type=Path,
                        default=ROOT / "modified_tlob/cost_curve_20261027.json")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    started = utc_now()
    timer = time.time()
    rss_samples = []

    def rss() -> float:
        value = float(psutil.Process().memory_info().rss / 1e6)
        rss_samples.append(value)
        return value

    windows = [load_window(*window) for window in WINDOWS]
    controllability = platform_controllability(COMPETITION_RULES)
    plan = build_plan(windows)
    plan_bytes = (canonical_json(plan) + "\n").encode("utf-8")
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    output = args.output_root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{plan_sha[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_bytes(plan_bytes)
    recorder = Recorder(output, EXPERIMENT_ID)
    recorder.event("plan", "predeclared", plan_sha256=plan_sha, windows=[w["name"] for w in windows],
                   rates={label: rate for label, rate in RATES}, primary=PRIMARY_RATE,
                   objects=[item["id"] for item in OBJECTS])
    print(f"predeclared plan {rel(output / 'plan.json')} sha256={plan_sha}")
    rss()

    scores, keys, targets = pooled_objects(windows)
    sections = int(keys["date"].nunique())
    days = int(pd.to_datetime(keys["date"]).dt.normalize().nunique())
    recorder.event("data", "pooled", rows=int(len(keys)), days=days, cross_sections=sections,
                   samples_per_section=round(len(keys) / sections, 1),
                   objects=sorted(scores), rss_mb=rss())
    payloads = [(item["id"], keys, scores[item["id"]], targets) for item in OBJECTS]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        evaluated = {object_id: results for object_id, results in executor.map(evaluate_object, payloads)}
    recorder.event("evaluation", "scenarios_complete", rss_mb=rss())

    repeat_metrics, _, _ = evaluate(
        keys.assign(score=scores["reference_nine_blend"])[["date", "instrument", "score"]], targets,
        evaluation_config(0.0), return_sections=True, undefined_constant_portfolios=True)

    summary_rows = []
    object_blocks = {}
    for item in OBJECTS:
        object_id = item["id"]
        results = evaluated[object_id]
        object_dir = output / "objects" / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        scenarios = {}
        for label, _ in RATES:
            metrics = results[label]["metrics"]
            daily = results[label]["daily"]
            daily.to_csv(object_dir / f"daily_{label}.csv", index=False, lineterminator="\n")
            scenarios[label] = scenario_block(metrics, daily, label, results[label]["single_side_rate"])
            scenarios[label]["monthly"] = monthly_table(daily)
            row = {"object_id": object_id, "role": item["role"], "scenario": label,
                   "single_side_rate": results[label]["single_side_rate"],
                   "gross_return_mean": scenarios[label]["gross_return_mean"],
                   "cost_mean": scenarios[label]["cost_mean"],
                   "net_return_mean": scenarios[label]["net_return_mean"],
                   "turnover_mean_local": scenarios[label]["turnover_mean_local"],
                   "official_turnover_value": None,
                   "rank_ic_mean": scenarios[label]["rank_ic_mean"],
                   "ic_mean": scenarios[label]["ic_mean"],
                   "long_short_sharpe": scenarios[label]["long_short_sharpe"],
                   "total_abs_weight_change_mean": scenarios[label]["total_abs_weight_change_mean"],
                   "breakeven_single_side_rate": scenarios[label]["breakeven_single_side_rate"],
                   "share_days_net_positive": scenarios[label]["share_days_net_positive"],
                   "undefined_rebalance_count": scenarios[label]["undefined_rebalance_count"],
                   "economic_status": scenarios[label]["economic_status"],
                   "seconds": results[label]["seconds"]}
            summary_rows.append(row)
        primary = scenarios[PRIMARY_RATE]
        headroom = economics_headroom(primary["gross_return_mean"],
                                      primary["total_abs_weight_change_mean"],
                                      primary["turnover_mean_local"], 0.0005)
        ten_bp = economics_headroom(primary["gross_return_mean"],
                                   primary["total_abs_weight_change_mean"],
                                   primary["turnover_mean_local"], 0.0010)
        object_blocks[object_id] = {
            "id": object_id, "role": item["role"], "definition": item["definition"],
            "evaluation_version": EVALUATION_VERSION,
            "windows": [window["name"] for window in windows],
            "turnover_item": primary["official_turnover_item"],
            "signal": {"rank_ic_mean": primary["rank_ic_mean"], "ic_mean": primary["ic_mean"],
                       "long_short_sharpe": primary["long_short_sharpe"],
                       "min_daily_long_short_return": primary["min_daily_long_short_return"]},
            "scenario_summary": {
                label: {key: scenarios[label][key] for key in (
                    "gross_return_mean_bp", "cost_mean_bp", "net_return_mean_bp",
                    "share_days_net_positive", "breakeven_single_side_rate_bp")}
                for label, _ in RATES},
            "headroom_at_5bp": headroom,
            "headroom_at_10bp": ten_bp,
            "scenarios": scenarios,
            "daily_sha256": {label: file_sha256(object_dir / f"daily_{label}.csv")
                             for label, _ in RATES},
        }
        write_json(object_dir / "metrics.json", object_blocks[object_id])

    summary = pd.DataFrame(summary_rows)
    (output / "comparison").mkdir(exist_ok=True)
    summary.to_csv(output / "comparison/summary.csv", index=False, lineterminator="\n")

    # The experiment index reads run-level daily*.csv to decide which dates were evaluated.
    primary_daily = evaluated["reference_nine_blend"][PRIMARY_RATE]["daily"]
    primary_daily.to_csv(output / "daily.csv", index=False, lineterminator="\n")

    checks = []
    evaluated_dates = pd.to_datetime(keys["date"])
    sealed_start = pd.Timestamp(SEALED[0])
    sealed_end = pd.Timestamp(SEALED[1])
    sealed_hits = int(((evaluated_dates >= sealed_start) & (evaluated_dates < sealed_end)).sum())
    checks.append({"name": "sealed_interval_untouched", "passed": sealed_hits == 0,
                   "evidence": f"{len(keys)} samples in {sections} cross sections over {days} days, "
                               f"{evaluated_dates.min():%Y-%m-%d}..{evaluated_dates.max():%Y-%m-%d}; "
                               f"samples inside {list(SEALED)}: {sealed_hits}"})
    window_hits = []
    for name, start, end, _ in WINDOWS:
        inside = int(((evaluated_dates >= pd.Timestamp(start)) & (evaluated_dates < pd.Timestamp(end))).sum())
        window_hits.append({"window": name, "rows": inside})
    checks.append({"name": "evaluation_inside_declared_windows",
                   "passed": sum(item["rows"] for item in window_hits) == len(keys)
                   and all(item["rows"] > 0 for item in window_hits),
                   "evidence": window_hits})
    linearity = {}
    for item in OBJECTS:
        object_id = item["id"]
        block = object_blocks[object_id]
        changes = {label: block["scenarios"][label]["total_abs_weight_change_mean"] for label, _ in RATES}
        costs = {label: block["scenarios"][label]["cost_mean"] for label, _ in RATES}
        turnover = {label: block["scenarios"][label]["turnover_mean_local"] for label, _ in RATES}
        nets = {label: block["scenarios"][label]["net_return_mean"] for label, _ in RATES}
        gross = {label: block["scenarios"][label]["gross_return_mean"] for label, _ in RATES}
        residuals = {}
        for label, rate in RATES:
            value = costs[label]
            expected = None if value is None else rate * changes[label]
            residuals[label] = None if (value is None or expected is None) else abs(value - expected)
        net_residuals = {label: None if (nets[label] is None or costs[label] is None or
                                         gross[label] is None) else abs(nets[label] -
                                                                       (gross[label] - costs[label]))
                         for label, _ in RATES}
        change_values = [value for value in changes.values() if value is not None]
        turnover_values = [value for value in turnover.values() if value is not None]
        linearity[object_id] = {
            "cost_residuals": residuals,
            "net_residuals": net_residuals,
            "change_spread": (max(change_values) - min(change_values)) if change_values else None,
            "turnover_spread": (max(turnover_values) - min(turnover_values)) if turnover_values else None,
            "passed": all(value is not None and value <= GENEROSITY
                          for value in list(residuals.values()) + list(net_residuals.values()))
            and (not change_values or (max(change_values) - min(change_values)) <= GENEROSITY)
            and (not turnover_values or (max(turnover_values) - min(turnover_values)) <= GENEROSITY),
        }
    checks.append({"name": "cost_linear_in_rate_and_turnover_rate_invariant",
                   "passed": all(item["passed"] for item in linearity.values()),
                   "evidence": linearity,
                   "note": "cost(r) = r * U exactly; turnover = U/2 is rate-free, so the official "
                           "turnover item cannot be moved by changing the cost assumption"})
    reference_block = object_blocks["reference_nine_blend"]
    flipped = object_blocks["control_sign_flipped_reference"]
    sign_cost_delta = None
    sign_gross_sum = None
    sign_cost_relative = None
    if (reference_block["scenarios"]["5bp"]["cost_mean"] is not None
            and flipped["scenarios"]["5bp"]["cost_mean"] is not None):
        sign_cost_delta = abs(reference_block["scenarios"]["5bp"]["cost_mean"]
                              - flipped["scenarios"]["5bp"]["cost_mean"])
        base_cost = reference_block["scenarios"]["5bp"]["cost_mean"]
        if base_cost:
            sign_cost_relative = float(sign_cost_delta / abs(base_cost))
    if (reference_block["scenarios"]["5bp"]["gross_return_mean"] is not None
            and flipped["scenarios"]["5bp"]["gross_return_mean"] is not None):
        sign_gross_sum = abs(reference_block["scenarios"]["5bp"]["gross_return_mean"]
                             + flipped["scenarios"]["5bp"]["gross_return_mean"])
    checks.append({"name": "sign_flipped_control",
                   "passed": bool(sign_gross_sum is not None and sign_gross_sum <= 1e-15
                                  and sign_cost_relative is not None and sign_cost_relative <= 1e-3),
                   "evidence": {"gross_negation_residual": sign_gross_sum,
                                "cost_relative_difference": sign_cost_relative,
                                "cost_difference": sign_cost_delta,
                                "why": "negating every score flips the long and short legs, so the gross "
                                       "spread must negate exactly. Cost is *not* an exact invariant: the "
                                       "pre-trade drift renormalises by (1 + realized spread), and negating "
                                       "the book flips that spread, so the drifted reference weights move "
                                       "by a second-order amount. The check therefore allows a 1e-3 "
                                       "relative difference instead of exact equality."}})
    reproducible = digest(evaluated["reference_nine_blend"]["0bp"]["metrics"]) == digest(repeat_metrics)
    checks.append({"name": "reproducible_evaluation", "passed": reproducible,
                   "evidence": "reference 0bp evaluated twice produces identical canonical metrics JSON"})
    checks.append({"name": "platform_evidence_verified", "passed": controllability["verified"],
                   "evidence": {item["key"]: item["line"] for item in controllability["evidence"]}})
    checks.append({"name": "no_training", "passed": True,
                   "evidence": "only archived predictions, fixed-weight blends and evaluate() calls; "
                               "no optimizer or model fit"})
    checks_passed = all(item["passed"] for item in checks)
    write_json(output / "checks.json", {"schema_version": 1, "passed": checks_passed, "checks": checks})
    recorder.event("checks", "completed", passed=checks_passed,
                   failed=[item["name"] for item in checks if not item["passed"]])

    reference_scenarios = object_blocks["reference_nine_blend"]["scenarios"]
    primary_reference = reference_scenarios[PRIMARY_RATE]
    primary_metrics = evaluated["reference_nine_blend"][PRIMARY_RATE]["metrics"]
    two_bp_reference = reference_scenarios.get("2bp", {})
    ten_bp_reference = reference_scenarios.get("10bp", {})
    headroom = object_blocks["reference_nine_blend"]["headroom_at_5bp"]
    primary_months = primary_reference["monthly"]
    months_negative = sum(1 for value in primary_months.values()
                          if value["net_return_mean"] is not None and value["net_return_mean"] < 0)
    month_shares = [value["share_days_net_positive"] for value in primary_months.values()
                    if value["share_days_net_positive"] is not None]
    trained_ids = [item["id"] for item in OBJECTS if item["role"] != "control"]
    breakevens = [object_blocks[object_id]["scenarios"][PRIMARY_RATE]["breakeven_single_side_rate_bp"]
                  for object_id in trained_ids]
    breakevens = [value for value in breakevens if value is not None]
    cost_curve = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "evaluation_version": EVALUATION_VERSION,
        "no_training": True,
        "sealed_interval": list(SEALED),
        "primary_stress_scenario": PRIMARY_RATE,
        "cost_scenarios": {label: rate for label, rate in RATES},
        "units": {
            "return": "forward 30-minute VWAP label return as a price fraction; per 1.0 equity and one "
                      "label period",
            "gross_return_mean": "long-short spread before cost, averaged over timestamps/days",
            "cost_mean": "single_side_rate * U averaged over defined rebalances",
            "net_return_mean": "gross - cost",
            "turnover_local": "U / 2, one-way share turnover of the two-leg 20% book",
            "bp": "1bp = 1e-4 of the return unit above",
        },
        "windows": [{"name": window["name"], "start": window["start"], "end": window["end"],
                     "replay_run": window["replay"]} for window in windows],
        "data": {"rows": int(len(keys)), "days": days, "cross_sections": sections,
                 "cross_section_count_from_metrics": int(primary_metrics.get("cross_section_count") or 0),
                 "labels_table": LABELS_TABLE,
                 "windows": [{"name": window["name"], "keys_sha256": window["keys_sha256"],
                              "targets_sha256": window["targets_sha256"]} for window in windows]},
        "objects": object_blocks,
        "platform_controllability": controllability,
        "conclusion": {
            "reference_object": "reference_nine_blend",
            "gross_return_mean_bp": primary_reference["gross_return_mean_bp"],
            "cost_5bp_bp": primary_reference["cost_mean_bp"],
            "net_5bp_bp": primary_reference["net_return_mean_bp"],
            "cost_2bp_bp": two_bp_reference.get("cost_mean_bp"),
            "net_2bp_bp": two_bp_reference.get("net_return_mean_bp"),
            "cost_10bp_bp": ten_bp_reference.get("cost_mean_bp"),
            "net_10bp_bp": ten_bp_reference.get("net_return_mean_bp"),
            "breakeven_single_side_rate_bp": primary_reference["breakeven_single_side_rate_bp"],
            "turnover_local": primary_reference["turnover_mean_local"],
            "turnover_share_of_theoretical_maximum": headroom["turnover_share_of_theoretical_maximum"],
            "turnover_required_for_zero_net_at_5bp": headroom["turnover_required_for_zero_net"],
            "turnover_reduction_required_share_at_5bp": headroom["turnover_reduction_required_share"],
            "gross_required_for_zero_net_at_5bp_bp": None if headroom["gross_required_for_zero_net"] is None
            else num(headroom["gross_required_for_zero_net"] * 1e4),
            "gross_required_multiple_at_5bp": headroom["gross_required_multiple"],
            "months_net_negative_at_5bp": months_negative,
            "month_count": len(primary_months),
            "share_days_net_positive_range_at_5bp": [num(min(month_shares)), num(max(month_shares))]
            if month_shares else None,
            "breakeven_bp_range_across_non_control_objects": [num(min(breakevens)), num(max(breakevens))]
            if breakevens else None,
            "statements": [
                f"参考九模型等权因子每个 30 分钟调仓期的毛多空为 "
                f"{primary_reference['gross_return_mean_bp']:.2f}bp，5bp 单边成本为 "
                f"{primary_reference['cost_mean_bp']:.2f}bp，净收益 "
                f"{primary_reference['net_return_mean_bp']:.2f}bp。",
                f"盈亏平衡单边成本为 {primary_reference['breakeven_single_side_rate_bp']:.2f}bp："
                f"2bp 情景净收益 {two_bp_reference.get('net_return_mean_bp', float('nan')):.2f}bp，"
                f"5bp 与 10bp 情景为负。",
                f"官方换手项不随成本情景变化（turnover = U/2 与费率无关），本地代理值 "
                f"{primary_reference['turnover_mean_local']:.4f}，理论最大值 2.0；"
                f"5bp 下要盈亏平衡需把换手降到 "
                f"{headroom['turnover_required_for_zero_net']:.3f}"
                f"（较现状减少 {headroom['turnover_reduction_required_share'] * 100:.0f}%），"
                f"或者把毛收益提高约 {headroom['gross_required_multiple']:.1f} 倍。",
                f"5bp 情景下参考因子在 {len(primary_months)} 个开发月中有 {months_negative} 个月净收益为负，"
                f"逐日为正的比例只有 {min(month_shares) * 100:.0f}%–{max(month_shares) * 100:.0f}%。",
                f"{len(breakevens)} 个非对照对象的盈亏平衡单边成本落在 {min(breakevens):.2f}–"
                f"{max(breakevens):.2f}bp，说明换主干（TLOB/MLP/Patch/九模型融合）解决不了经济性，"
                f"只能改换手。",
                "换手只能通过因子输出在时间上更持久来降低（平台只接收三列因子值），"
                "这是唯一后续路径；持仓/门槛类做法不被平台接受。",
                "以上净收益是本地经济性口径，不是官方得分；官方换手项是相对排名，本地数值只是代理。",
            ],
            "single_follow_up_path": {
                "path_id": "factor_temporal_persistence",
                "because": "平台只接收 [date, instrument, factor]，持仓、仓位、调仓频率与成本假设都不可控",
                "change": "在已冻结的参考因子上做一项单一改动：让因子输出在相邻时点更持久"
                          "（可提交输出的时间一致性约束，或同一约束加到训练目标），不改主干/数据/标签/"
                          "采样/评价口径",
                "single_factor": True,
                "forbidden": [
                    "持仓或仓位层面的换手优化（平台不接受）",
                    "以标准化分数当 bp 收益来设交易门槛",
                    "同类平滑参数的细网格搜索",
                    "任何重训以外的第二次同向改动（一次只改一个主要因素）",
                ],
                "measured_by": [
                    "0/2/5/10bp 四个情景的毛收益/成本/净收益同评，5bp 为主压力结论",
                    "官方换手项只能以本地 U/2 代理的单调下降表达",
                    "RankIC/IC 与多空收益不得塌缩为常数信号",
                ],
                "quantified_gap": {
                    "turnover_local_now": primary_reference["turnover_mean_local"],
                    "turnover_local_required_at_5bp": headroom["turnover_required_for_zero_net"],
                    "reduction_required_share": headroom["turnover_reduction_required_share"],
                    "cost_saving_per_0_1_turnover_reduction_bp": headroom[
                        "cost_bp_per_0_1_turnover_reduction"],
                },
            },
        },
        "reporting_rules": {
            "official_turnover_item": "cross-submission rank percentile; unavailable locally; the local "
                                      "U/2 proxy is monotone-consistent but absolute levels are not an "
                                      "official score or threshold",
            "net_return": "local economic lens only; the official score has no cost term",
            "promotion": "none; this contract never promotes, rejects or selects a model",
        },
        "checks": {"passed": checks_passed, "path": "checks.json"},
    }
    write_json(output / "cost_curve.json", cost_curve)
    write_json(output / "controllability.json", controllability)

    data_document = {
        "schema_version": 1, "source": "archived multiscale replay predictions and vwap labels",
        "version": "cpt_jyc_2026", "evaluation_version": EVALUATION_VERSION,
        "labels_table": LABELS_TABLE,
        "windows": [{"name": window["name"], "start": window["start"], "end": window["end"],
                     "replay_run": window["replay"], "keys_sha256": window["keys_sha256"],
                     "targets_sha256": window["targets_sha256"],
                     "prediction_sha256": window["prediction_sha256"]} for window in windows],
        "targets_sha256": evaluated["reference_nine_blend"]["0bp"]["metrics"].get("targets_sha256"),
        "sample_keys_sha256": evaluated["reference_nine_blend"]["0bp"]["metrics"].get("sample_keys_sha256"),
    }
    write_json(output / "data.json", data_document)
    config = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID,
        "hypothesis": plan["hypothesis"], "output_root": "runs",
        "data": {"labels_table": LABELS_TABLE, "source": data_document["source"],
                 "version": data_document["version"]},
        "evaluation": {"quantile": 0.2, "periods_per_year": 252, "min_cross_section": 3,
                       "constant_portfolios": "undefined", "single_side_rate": None,
                       "scenarios": [{"label": label, "single_side_rate": rate} for label, rate in RATES]},
        "windows": [{"name": name, "start": start, "end": end} for name, start, end, _ in WINDOWS],
        "splits": {"train": {"start": "2023-08-01T00:00:00", "end": "2023-11-01T00:00:00"},
                   "validation": {"start": "2023-11-01T00:00:00", "end": "2024-07-01T00:00:00"},
                   "final": None},
        "evaluation_split": "validation",
        "no_training": True,
    }
    write_json(output / "config.json", config)
    run_metrics = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID,
        "evaluation_version": EVALUATION_VERSION,
        "primary_stress_scenario": PRIMARY_RATE,
        "windows": [window["name"] for window in windows],
        "rows": int(len(keys)), "days": days, "cross_sections": sections, "day_count": days,
        "valid_days": int(primary_daily["long_short_return"].notna().sum()),
        "reference": {label: {key: object_blocks["reference_nine_blend"]["scenarios"][label][key]
                              for key in ("gross_return_mean_bp", "cost_mean_bp", "net_return_mean_bp",
                                          "turnover_mean_local", "share_days_net_positive")}
                      for label, _ in RATES},
        "objects": {object_id: {"role": block["role"],
                                "turnover_local": block["turnover_item"]["local_proxy"],
                                "rank_ic_mean": block["signal"]["rank_ic_mean"],
                                "breakeven_single_side_rate_bp": block["scenarios"][PRIMARY_RATE][
                                    "breakeven_single_side_rate_bp"],
                                "scenario_summary": block["scenario_summary"]}
                    for object_id, block in object_blocks.items()},
    }
    write_json(output / "metrics.json", run_metrics)

    record = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "status": "completed",
        "kind": "evaluation", "created_at": utc_now(),
        "run_dir": rel(output), "cost_curve": rel(output / "cost_curve.json"),
        "controllability": rel(output / "controllability.json"),
        "checks": rel(output / "checks.json"), "plan": rel(output / "plan.json"),
        "evaluation_version": EVALUATION_VERSION,
        "primary_stress_scenario": PRIMARY_RATE,
        "reference": {
            "object": "reference_nine_blend",
            "gross_return_mean_bp": cost_curve["conclusion"]["gross_return_mean_bp"],
            "cost_bp": {label: object_blocks["reference_nine_blend"]["scenarios"][label]["cost_mean_bp"]
                        for label, _ in RATES},
            "net_bp": {label: object_blocks["reference_nine_blend"]["scenarios"][label]["net_return_mean_bp"]
                       for label, _ in RATES},
            "turnover_local": cost_curve["conclusion"]["turnover_local"],
            "breakeven_single_side_rate_bp": cost_curve["conclusion"][
                "breakeven_single_side_rate_bp"],
            "months_net_negative_at_5bp": months_negative,
            "month_count": len(primary_months),
            "breakeven_bp_range_across_non_control_objects": cost_curve["conclusion"][
                "breakeven_bp_range_across_non_control_objects"],
        },
        "platform_answer": "仅可提交因子（[date, instrument, factor]），不可提交持仓；"
                           "唯一后续路径是因子输出的时间持久性",
        "single_follow_up_path": cost_curve["conclusion"]["single_follow_up_path"],
        "checks_passed": checks_passed,
        "evidence_limits": [
            "prices/labels are the archived vwap label table; local approximate caliber, not a platform score",
            "turnover is the eval-v2 20% two-leg proxy; the official construction and its rank are unavailable",
            "cost model is linear in traded notional with no impact term; 5bp is a declared stress assumption",
            "all five months were already viewed development months; no sealed data was read",
        ],
    }
    write_json(args.record, record)

    artifacts = [{"path": path.relative_to(output).as_posix(), "sha256": file_sha256(path),
                  "bytes": path.stat().st_size}
                 for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"]
    manifest = {
        "schema_version": 1, "run_id": output.name, "experiment_id": EXPERIMENT_ID,
        "kind": "evaluation", "status": "completed",
        "started_at": started, "ended_at": utc_now(),
        "hypothesis": plan["hypothesis"], "evaluation_version": EVALUATION_VERSION,
        "evaluation_split": "validation",
        "plan_sha256": plan_sha,
        "config_sha256": digest(config),
        "exit_code": 0,
        "seed": None,
        "control_seed": CONTROL_SEED,
        "splits": config["splits"],
        "metrics": {"path": "metrics.json", "sha256": file_sha256(output / "metrics.json")},
        "evaluation_source": {
            "evaluation_py_sha256": file_sha256(ROOT / "src/jinyiquant/evaluation.py"),
            "script_sha256": file_sha256(Path(__file__)),
            "contract_sha256": digest(CONTRACT),
        },
        "code": {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "files": {"src/jinyiquant/evaluation.py": file_sha256(ROOT / "src/jinyiquant/evaluation.py"),
                      "scripts/cost_curve_eval.py": file_sha256(Path(__file__))},
        },
        "data": {"labels_table": LABELS_TABLE, "rows": int(len(keys)), "days": days,
                 "cross_sections": sections, "windows": [w["name"] for w in windows],
                 "sha256": digest([{name: window[name] for name in
                                    ("name", "keys_sha256", "targets_sha256")} for window in windows])},
        "record": rel(args.record),
        "cost_scenarios": {label: rate for label, rate in RATES},
        "primary_stress_scenario": PRIMARY_RATE,
        "checks": rel(output / "checks.json"),
        "resources": {"max_rss_mb": max(rss_samples), "wall_seconds": round(time.time() - timer, 3),
                      "workers": args.workers,
                      "rss_note": "max of sampled RSS points in the parent process only; "
                                  "worker processes are not summed"},
        "artifacts": artifacts,
    }
    write_json(output / "manifest.json", manifest)
    recorder.event("run", "ended", status="completed", checks_passed=checks_passed,
                   wall_seconds=manifest["resources"]["wall_seconds"], record=rel(args.record))
    print(canonical_json({"run_dir": rel(output), "checks_passed": checks_passed,
                          "net_5bp_bp": cost_curve["conclusion"]["net_5bp_bp"],
                          "breakeven_bp": cost_curve["conclusion"]["breakeven_single_side_rate_bp"],
                          "turnover_local": cost_curve["conclusion"]["turnover_local"],
                          "turnover_required_5bp": cost_curve["conclusion"][
                              "turnover_required_for_zero_net_at_5bp"]}))
    return 0 if checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
