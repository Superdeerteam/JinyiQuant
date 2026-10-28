#!/usr/bin/env python3
"""融合增量闸门（fusion-increment gate）— R6/R8/R11..R18 的唯一判定入口。

问题只有一个：把某条新路线的预测以单一权重并入已冻结的 441d@1.32 三点种子主干之后，
主干在一批**从未参与该路线设计**的交易日上是否变好。相关性低不算增量，榜单好看也不算。

口径（与 configs/fusion_increment_registration_20261028.json 一一对应，两侧不一致即 abort）
--------------------------------------------------------------------------------------
* 主干 = dev10_replay_20261028_plan 里 r441_s11/r441_s29/r441_s47 的等权 blend。
* 每个臂先把成员原始分数等权平均，再做"按 timestamp 的 cross-sectional z-score + 因果 EWMA"
  （alpha=0.9111677303420447），这正是 scripts/mktctx_compare.py::daily_scores 的 house 口径。
* 融合在平滑后的标准化空间里做凸组合：fused(w) = (1-w)*trunk + w*candidate。
  于是 w=0 逐位复现主干口径、w=1 逐位复现候选口径（端点不变性由 --self-test 守住）。
* w 只在**最早的开发块** 2023-11-01..2023-11-14（nov2023 内那 10 天，早已被 V3 验证花掉，
  不承担任何干净留出角色）上拟合，拟合完冻结；8 个筛查窗与 2 个确认窗全在拟合之后。
* 增量 = 每日 rank_ic(fused) - rank_ic(trunk)，按 (window, trading day) 配对；
  种子在 blend 内部，不作为独立天数，因此配对单位只有交易日。
* 区间 = scripts/compare_blend_replay.py::block_interval，ceil(n/5) 个随机起点 5 日块，
  20000 draws，seed 20261028；旧的单块估计只是并列报出（archived_ci_*），永不门控。

判定（门槛写死在代码里，注册文件必须写同一组数）
----------------------------------------------------------------------
通过 = 165 天池化均值 >= +0.003 且 修正 CI 下界 > 0 且 >= 7/8 筛查窗均值为正
       且 155 天 post-freeze 均值 > 0 且 推理成本可报 且 w > 0。
单模（arm_kind=single_model）只出描述，不出 verdict。

读数纪律
----------------
* screen：8 窗 165 天可以反复看，但每次读都追加到 read_ledger.jsonl（可审计的"看了几次"）。
* confirm：jun2024+sep2024 共 38 天，每个方向只读一次，且必须持队长放行文件，
  且该方向的 screen 读数已经 clears_fusion_gate，且权重必须还是筛查时那一个。
* 十窗/203 天的池化数只允许作为 final_report 出现——一旦给出，确认额度就算花掉。
* 十窗在本日之前已被十几个候选反复使用过：165 天不是新鲜样本，本闸门显式承认选择偏差。

用法
------
    python scripts/fusion_increment_gate.py --mode audit
    python scripts/fusion_increment_gate.py --mode audit --plan runs/r18/plan.json
    python scripts/fusion_increment_gate.py --mode screen --plan runs/r18/plan.json
    python scripts/fusion_increment_gate.py --mode confirm --plan ... --release release.json
    python scripts/fusion_increment_gate.py --mode single --plan ...      # 仅描述
    python scripts/fusion_increment_gate.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _entry in (str(REPO / "src"), str(REPO / "scripts")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from compare_blend_replay import (  # noqa: E402
    archived_single_block_interval,
    block_interval,
)
from jinyiquant.config import EvaluationConfig  # noqa: E402
from jinyiquant.evaluation import EVALUATION_VERSION, evaluate  # noqa: E402
from jinyiquant.io import file_sha256, read_json, write_json  # noqa: E402
from verify_turnover_change import causal_ewma  # noqa: E402

KEYS = ["date", "instrument"]
DEFAULT_REGISTRATION = REPO / "configs/fusion_increment_registration_20261028.json"
DEFAULT_LEDGER = REPO / "runs/fusion-gate/read_ledger.jsonl"
EXECUTABLE = Path(__file__).resolve()

PRIMARY_METRIC = "rank_ic"
SIDE_METRICS = ("long_short_return", "turnover")
# 净收益：eval-v2 的 net_return = long_short_return - single_side_rate * U，U = 2 * turnover。
# 5bp 是 house 的锁定压力场景（b4 注册文件 stress_single_side_rate_bp=5.0，net5_limit=5.62e-05），
# 这里只用同一口径把净收益并列报出，不做任何门控。
NET_METRIC = "net_return_5bp"
COST_SINGLE_SIDE_RATE = 0.0005
REPORT_METRICS = (PRIMARY_METRIC, *SIDE_METRICS, NET_METRIC)
# 纯标注：441 候选晋升时用的 turnover_mean 退化线 0.0318；超过只在 verdict 里打 attention，不改变判决。
TURNOVER_ATTENTION_THRESHOLD = 0.0318
MIN_IMPROVEMENT = 0.003
MIN_POSITIVE_WINDOWS = 7
WINDOWS_TOTAL = 8
SCREEN_DAYS = 165
CONFIRMATION_DAYS = 38
CONFIRMATION_MIN_POSITIVE_DAYS = 21
FIT_DAYS = 10
POST_FREEZE_DAYS = 155
BLOCK_DAYS = 5
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 20261028
WEIGHT_GRID = tuple(round(0.05 * index, 2) for index in range(11))
MAX_FUSION_WEIGHT = 0.5
EWMA_ALPHA = 0.9111677303420447
TIE_TOLERANCE = 1e-9
PLAN_SCHEMA = "fusion-route-plan/1"
FIT_START, FIT_END = "2023-11-01", "2023-11-14"
WINDOW_STARTS = {
    "nov2023": "2023-11-01", "jan2024": "2024-01-01", "feb2024": "2024-02-01",
    "mar2024": "2024-03-01", "apr2024": "2024-04-01", "may2024": "2024-05-01",
    "jun2024": "2024-06-03", "jul2024": "2024-07-01", "aug2024": "2024-08-01",
    "sep2024": "2024-09-02",
}

SCREEN_VERDICTS = (
    "clears_fusion_gate",
    "no_allocation",
    "below_threshold",
    "interval_includes_zero",
    "positive_windows_below_7",
    "post_freeze_mean_not_positive",
    "blocked_missing_cost_report",
    "descriptive_only",
)


# --------------------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------------------- #
def _number(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def day_strings(keys: pd.DataFrame) -> np.ndarray:
    return (pd.to_datetime(keys["date"]).dt.normalize().dt.strftime("%Y-%m-%d")).to_numpy()


def interval_summary(values, *, block: int = BLOCK_DAYS, draws: int = BOOTSTRAP_DRAWS,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    """5 日移动块 bootstrap（复用 compare_blend_replay.block_interval）。

    修正后的区间领先；退役的单块区间只作为 archived_ci_* 并列保留，用于解释历史读数为何偏保守。
    """
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise SystemExit("interval_summary: no finite values")
    interval = block_interval(data, block=block, draws=draws, seed=seed)
    retired = archived_single_block_interval(data, block=block, draws=draws, seed=seed)
    return {
        "days": int(interval["days"] or 0),
        "mean": _number(interval["mean"]),
        "ci_low": _number(interval["ci_low"]),
        "ci_high": _number(interval["ci_high"]),
        "archived_ci_low": _number(retired["ci_low"]),
        "archived_ci_high": _number(retired["ci_high"]),
        "block_days": block,
        "draws": draws,
        "seed": seed,
        "ci_excludes_zero": bool(interval["ci_low"] is not None and interval["ci_low"] > 0.0),
    }


def evaluation_config(rate: float | None = 0.0) -> EvaluationConfig:
    return EvaluationConfig(
        quantile=0.2,
        periods_per_year=252,
        min_cross_section=3,
        constant_portfolios="undefined",
        single_side_rate=rate,
    )


def daily_metrics(keys: pd.DataFrame, targets: pd.DataFrame, scores) -> pd.DataFrame:
    """eval-v2 口径的逐日 rank_ic / turnover / long_short_return / net5（一行一个交易日）。

    net5 用 single_side_rate=5e-4 跑同一个 evaluate，再从 sections 直读 net_return：eval-v2
    里成本只进 cost 与 net_return 两列（evaluation.py:533-534），其余列逐位不变，所以一次
    5e-4 的 evaluate 同时给出 gross 与 5bp 净收益。不能拿 long_short_return 减 5e-4*U 反推：
    日度聚合是"等时间戳均值、逐列跳过 NaN"（evaluation.py:584-585），首日无前序组合、标签
    重叠、常数组合等时点会让 net 与 gross 覆盖的时点集不同，反推会系统性偏。
    """
    values = np.asarray(scores, dtype=float)
    if values.size != len(keys):
        raise SystemExit(f"score length {values.size} != key rows {len(keys)}")
    if not np.isfinite(values).all():
        raise SystemExit("non-finite scores: an arm cannot be scored with gaps")
    predictions = keys[KEYS].assign(score=values)
    _, daily, _ = evaluate(
        predictions,
        targets,
        evaluation_config(COST_SINGLE_SIDE_RATE),
        return_sections=True,
        undefined_constant_portfolios=True,
    )
    missing = [column for column in ("date", "net_return", PRIMARY_METRIC, *SIDE_METRICS)
               if column not in daily.columns]
    if missing:
        raise SystemExit(f"eval-v2 daily frame lacks {missing}")
    frame = daily[["date", *SIDE_METRICS, PRIMARY_METRIC]].copy()
    frame[NET_METRIC] = daily["net_return"].to_numpy(dtype=float)
    if frame["date"].duplicated().any():
        raise SystemExit("eval-v2 returned duplicate trading days")
    return frame.sort_values("date").reset_index(drop=True)


def load_scores(run_dir: str | Path, window: str, version: str, keys: pd.DataFrame) -> np.ndarray:
    path = Path(run_dir) / "windows" / window / version / "predictions.parquet"
    if not path.is_file():
        raise SystemExit(f"missing predictions: {path}")
    predictions = pd.read_parquet(path)[KEYS + ["score"]]
    merged = predictions.merge(keys, on=KEYS, how="inner",
                               validate="one_to_one").sort_values(KEYS).reset_index(drop=True)
    if len(merged) != len(keys) or not merged[KEYS].equals(keys):
        raise SystemExit(f"{path}: prediction coverage or key order mismatch")
    return merged["score"].to_numpy(dtype=float)


def smoothed_arm_score(members: list[dict], run_dir: str | Path, window: str,
                       keys: pd.DataFrame) -> np.ndarray:
    """成员原始分数先等权平均，再做 house 口径的 z-score + 因果 EWMA。"""
    if not members:
        raise SystemExit(f"{window}: arm has no members")
    stack = [load_scores(run_dir, window, member["version"], keys) for member in members]
    blended = np.mean(np.stack(stack, axis=0), axis=0)
    return causal_ewma(keys, blended, EWMA_ALPHA)


def fuse(incumbent: np.ndarray, candidate: np.ndarray, weight: float) -> np.ndarray:
    return (1.0 - weight) * incumbent + weight * candidate


# --------------------------------------------------------------------------------------- #
# registration + plan integrity
# --------------------------------------------------------------------------------------- #
def gate_script_state(registration: dict) -> dict:
    actual = file_sha256(EXECUTABLE)
    pinned = registration.get("gate_script_sha256")
    return {"pinned": pinned, "actual": actual, "matches": pinned == actual}


def load_registration(path: str | Path = DEFAULT_REGISTRATION) -> dict:
    """读注册文件并与代码内写死的门槛逐项对照；任何一侧被改过都直接 abort。"""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"missing registration: {path}")
    registration = read_json(path)
    registration["_path"] = str(path)
    registration["_sha256"] = file_sha256(path)

    roles = registration["window_roles"]
    for label, spec in (("authority", roles["authority"]), ("replay_plan", roles["replay_plan"])):
        pinned_path = Path(REPO / spec["path"])
        if not pinned_path.is_file():
            raise SystemExit(f"missing {label}: {pinned_path}")
        if file_sha256(pinned_path) != spec["sha256"]:
            raise SystemExit(
                f"{pinned_path}: sha256 differs from the pinned {label} ({spec['sha256']}); "
                "the window split or the replay plan was edited")

    estimator = registration["estimator"]
    implementation = Path(REPO / estimator["implementation"].split("::")[0])
    if not implementation.is_file():
        raise SystemExit(f"missing estimator implementation: {implementation}")
    if file_sha256(implementation) != estimator["implementation_sha256"]:
        raise SystemExit(
            f"{implementation}: sha256 differs from the pinned estimator "
            f"({estimator['implementation_sha256']}); block_interval was edited")

    decision = registration["decision"]
    fusion = registration["fusion_increment_definition"]
    checks = [
        ("decision.metric", decision["metric"], PRIMARY_METRIC),
        ("decision.min_improvement", decision["min_improvement"], MIN_IMPROVEMENT),
        ("decision.min_positive_windows", decision["min_positive_windows"], MIN_POSITIVE_WINDOWS),
        ("decision.windows_total", decision["windows_total"], WINDOWS_TOTAL),
        ("decision.ci_must_exclude_zero", decision["ci_must_exclude_zero"], True),
        ("decision.post_freeze_guard.days", decision["post_freeze_guard"]["days"], POST_FREEZE_DAYS),
        ("decision.confirmation.min_positive_days",
         decision["confirmation"]["min_positive_days"], CONFIRMATION_MIN_POSITIVE_DAYS),
        ("decision.confirmation.days", decision["confirmation"]["days"], CONFIRMATION_DAYS),
        ("estimator.block_days", estimator["block_days"], BLOCK_DAYS),
        ("estimator.draws", estimator["draws"], BOOTSTRAP_DRAWS),
        ("estimator.seed", estimator["seed"], BOOTSTRAP_SEED),
        ("fusion.weight_fit.trading_days", fusion["weight_fit"]["trading_days"], FIT_DAYS),
        ("fusion.weight_fit.max_weight", fusion["weight_fit"]["max_weight"], MAX_FUSION_WEIGHT),
        ("fusion.ewma_alpha", fusion["ewma_alpha"], EWMA_ALPHA),
        ("window_roles.screen_days", roles["screen_days"], SCREEN_DAYS),
        ("window_roles.confirmation_days", roles["confirmation_days"], CONFIRMATION_DAYS),
        ("window_roles.post_freeze_days", roles["post_freeze_days"], POST_FREEZE_DAYS),
        ("red_lines.no_threshold_change", registration["red_lines"]["no_threshold_change"], True),
        ("red_lines.no_score_before_registration",
         registration["red_lines"]["no_score_before_registration"], True),
    ]
    for label, registered, expected in checks:
        if registered != expected:
            raise SystemExit(
                f"registration[{label}] = {registered!r} but the gate is pinned to "
                f"{expected!r}; refusing to run with a changed threshold")
    if tuple(float(value) for value in fusion["weight_fit"]["grid"]) != WEIGHT_GRID:
        raise SystemExit(
            f"registration weight grid {fusion['weight_fit']['grid']!r} != {WEIGHT_GRID!r}")
    annotation = (registration.get("annotations") or {}).get("turnover_attention") or {}
    if float(annotation.get("turnover_attention_threshold", -1.0)) != TURNOVER_ATTENTION_THRESHOLD:
        raise SystemExit(
            "registration annotations.turnover_attention threshold differs from the pinned "
            f"{TURNOVER_ATTENTION_THRESHOLD}; the annotation line is registered, not free to edit")
    if annotation.get("does_not_gate") is not True:
        raise SystemExit("turnover_attention must stay a non-gating annotation")
    side_reports = registration.get("mandatory_side_reports") or {}
    if not (side_reports.get(NET_METRIC) or {}).get("required"):
        raise SystemExit(
            f"registration must require the {NET_METRIC} side report (missing net5 means an "
            "incomplete read)")
    if (str(fusion["weight_fit"]["start"])[:10] != FIT_START
            or str(fusion["weight_fit"]["end"])[:10] != FIT_END):
        raise SystemExit("registered weight-fit block differs from 2023-11-01..2023-11-14")
    if len(roles["screen"]) != WINDOWS_TOTAL:
        raise SystemExit(f"registration lists {len(roles['screen'])} screen windows")
    if sorted(roles["confirmation"]) != sorted({"jun2024", "sep2024"}):
        raise SystemExit(f"unexpected confirmation windows: {roles['confirmation']}")
    by_role = {row["name"]: row for row in roles["windows"]}
    if set(by_role) != set(roles["screen"]) | set(roles["confirmation"]):
        raise SystemExit("registered window list does not match the screen/confirmation split")
    screen_days = sum(by_role[name]["trading_days"] for name in roles["screen"])
    confirm_days = sum(by_role[name]["trading_days"] for name in roles["confirmation"])
    if screen_days != SCREEN_DAYS or confirm_days != CONFIRMATION_DAYS:
        raise SystemExit(
            f"registered day counts sum to {screen_days}+{confirm_days}, not "
            f"{SCREEN_DAYS}+{CONFIRMATION_DAYS}")
    return registration


def window_index(registration: dict) -> dict[str, dict]:
    return {row["name"]: row for row in registration["window_roles"]["windows"]}


def load_plan(path: str | Path, registration: dict) -> dict:
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"missing route plan: {path}")
    plan = read_json(path)
    plan["_path"] = str(path)
    plan["_sha256"] = file_sha256(path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise SystemExit(f"{path}: schema_version must be {PLAN_SCHEMA!r}")
    if not plan.get("route"):
        raise SystemExit(f"{path}: route is required")
    if plan.get("arm_kind") not in {"fused", "single_model"}:
        raise SystemExit(f"{path}: arm_kind must be 'fused' or 'single_model'")
    pinned = plan.get("registration") or {}
    if pinned.get("sha256") != registration["_sha256"]:
        raise SystemExit(
            f"{path}: plan pins registration {pinned.get('sha256')!r} but the gate loaded "
            f"{registration['_sha256']!r}; re-author the plan against the registered protocol")
    for arm in ("incumbent", "candidate"):
        block = plan.get(arm) or {}
        if not block.get("run_dir"):
            raise SystemExit(f"{path}: {arm}.run_dir is required")
        members = block.get("members") or []
        if not members or any(not member.get("version") for member in members):
            raise SystemExit(f"{path}: {arm}.members[].version is required")
    if not (plan.get("candidate") or {}).get("label"):
        raise SystemExit(f"{path}: candidate.label is required (one main configuration per route)")
    return plan


# --------------------------------------------------------------------------------------- #
# window data
# --------------------------------------------------------------------------------------- #
def read_window(run_dir: str | Path, window: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(run_dir) / "windows" / window["name"]
    if not base.is_dir():
        raise SystemExit(f"missing window directory: {base}")
    targets = pd.read_parquet(base / "targets.parquet").sort_values(KEYS).reset_index(drop=True)
    keys_path = base / "keys.parquet"
    keys = pd.read_parquet(keys_path) if keys_path.is_file() else targets[KEYS]
    keys = keys.sort_values(KEYS).reset_index(drop=True)[KEYS]
    if keys.duplicated().any():
        raise SystemExit(f"{base}: duplicate window keys")
    if "target" not in targets.columns:
        raise SystemExit(f"{base}: targets.parquet has no 'target' column")
    targets = targets[KEYS + ["target"]]
    if not targets[KEYS].equals(keys):
        raise SystemExit(f"{base}: keys.parquet and targets.parquet disagree on the key set")

    unique = np.unique(day_strings(keys))
    start, end = str(window["start"])[:10], str(window["end"])[:10]
    if unique[0] < start or unique[-1] >= end:
        raise SystemExit(
            f"{base}: trading days {unique[0]}..{unique[-1]} leave the registered window "
            f"[{start}, {end})")
    if len(unique) != int(window["trading_days"]):
        raise SystemExit(
            f"{base}: {len(unique)} trading days but the registration pins "
            f"{window['trading_days']}")
    return keys, targets


class WindowCache:
    """按窗口惰性装载 keys/targets 与两个臂的平滑分数；拟合块与读数共用同一份向量。"""

    def __init__(self, plan: dict, registration: dict):
        self.plan = plan
        self.index = window_index(registration)
        self.incumbent_run = Path(plan["incumbent"]["run_dir"])
        self.candidate_run = Path(plan["candidate"]["run_dir"])
        self._store: dict[str, dict] = {}

    def get(self, window: str) -> dict:
        if window not in self.index:
            raise SystemExit(f"window {window!r} is not registered")
        if window in self._store:
            return self._store[window]
        spec = self.index[window]
        keys, targets = read_window(self.incumbent_run, spec)
        _, candidate_targets = read_window(self.candidate_run, spec)
        if not candidate_targets["target"].equals(targets["target"]):
            raise SystemExit(
                f"{window}: incumbent and candidate windows disagree on targets; "
                "the two arms are not scoring the same labels")
        entry = {
            "window": window,
            "keys": keys,
            "targets": targets,
            "days": day_strings(keys),
            "incumbent": smoothed_arm_score(self.plan["incumbent"]["members"],
                                            self.incumbent_run, window, keys),
            "candidate": smoothed_arm_score(self.plan["candidate"]["members"],
                                            self.candidate_run, window, keys),
        }
        self._store[window] = entry
        return entry


# --------------------------------------------------------------------------------------- #
# weight fit (the only place a parameter is chosen; frozen afterwards)
# --------------------------------------------------------------------------------------- #
def fit_weight(cache: WindowCache, registration: dict) -> dict:
    spec = registration["fusion_increment_definition"]["weight_fit"]
    entry = cache.get(spec["window"])
    days = entry["days"]
    start, end = str(spec["start"])[:10], str(spec["end"])[:10]
    fit_days = np.unique(days[(days >= start) & (days <= end)])
    mask = (days >= start) & (days <= end)
    count = int(len(fit_days))
    if count != int(spec["trading_days"]):
        raise SystemExit(
            f"weight-fit block has {count} trading days but the registration pins "
            f"{spec['trading_days']}")
    keys = entry["keys"][mask].reset_index(drop=True)
    targets = entry["targets"][mask].reset_index(drop=True)
    incumbent, candidate = entry["incumbent"][mask], entry["candidate"][mask]

    profile = []
    for weight in WEIGHT_GRID:
        daily = daily_metrics(keys, targets, fuse(incumbent, candidate, weight))
        series = daily[PRIMARY_METRIC].to_numpy(dtype=float)
        if not np.isfinite(series).any():
            raise SystemExit(f"weight fit: no finite rank_ic at w={weight}")
        profile.append({
            "weight": float(weight),
            "rank_ic_mean": float(np.nanmean(series)),
            "days": int(np.isfinite(series).sum()),
        })
    best = max(row["rank_ic_mean"] for row in profile)
    chosen = min(row["weight"] for row in profile
                 if best - row["rank_ic_mean"] <= TIE_TOLERANCE)
    return {
        "window": spec["window"],
        "start": str(spec["start"])[:10],
        "end": str(spec["end"])[:10],
        "trading_days": count,
        "weight": float(chosen),
        "criterion": spec["criterion"],
        "grid": list(WEIGHT_GRID),
        "tie_break": "smallest weight within 1e-9 of the maximum objective",
        "profile": profile,
        "post_freeze_days": SCREEN_DAYS - count,
    }


# --------------------------------------------------------------------------------------- #
# paired daily read
# --------------------------------------------------------------------------------------- #
def read_windows(cache: WindowCache, windows: list[str], weight: float) -> dict:
    """逐窗算出 trunk/candidate/fused 的日度读数与逐日配对差。"""
    per_window: dict[str, dict] = {}
    pooled: dict[str, list[float]] = {metric: [] for metric in REPORT_METRICS}
    levels = {arm: {metric: [] for metric in REPORT_METRICS}
              for arm in ("trunk", "candidate", "fused")}
    labels: list[str] = []
    dates: list[str] = []
    for window in windows:
        entry = cache.get(window)
        keys, targets = entry["keys"], entry["targets"]
        fused = fuse(entry["incumbent"], entry["candidate"], weight)
        trunk_daily = daily_metrics(keys, targets, entry["incumbent"])
        fused_daily = daily_metrics(keys, targets, fused)
        candidate_daily = daily_metrics(keys, targets, entry["candidate"])
        merged = trunk_daily.merge(fused_daily, on="date", how="inner",
                                   suffixes=("_trunk", "_fused"), validate="one_to_one")
        merged = merged.merge(candidate_daily, on="date", how="inner", validate="one_to_one")
        merged = merged.rename(columns={metric: f"{metric}_candidate"
                                        for metric in REPORT_METRICS})
        expected = len(np.unique(entry["days"]))
        if len(merged) != expected:
            raise SystemExit(f"{window}: {len(merged)} paired daily rows != {expected} days")
        per_window[window] = {
            "trading_days": int(len(merged)),
            "first_day": str(merged["date"].iloc[0]),
            "last_day": str(merged["date"].iloc[-1]),
        }
        for metric in REPORT_METRICS:
            trunk_values = merged[f"{metric}_trunk"].to_numpy(dtype=float)
            candidate_values = merged[f"{metric}_candidate"].to_numpy(dtype=float)
            fused_values = merged[f"{metric}_fused"].to_numpy(dtype=float)
            delta = fused_values - trunk_values
            # 成本类读数在"没有前一组合"的交易日上是未定义（eval-v2 不零填），这些天不能进
            # 区间估计，也不能被当成 0：只在配对差有定义的交易日上做块 bootstrap，并报出计数。
            defined = np.isfinite(delta)
            if not defined.any():
                raise SystemExit(f"{window}: no trading day with a defined {metric} difference")
            defined_delta = delta[defined]
            per_window[window][metric] = {
                "days": int(len(merged)),
                "days_defined": int(defined.sum()),
                "days_undefined": int(len(merged) - int(defined.sum())),
                "trunk_mean": _number(np.nanmean(trunk_values)),
                "candidate_mean": _number(np.nanmean(candidate_values)),
                "fused_mean": _number(np.nanmean(fused_values)),
                "delta_mean": _number(np.nanmean(delta)),
                "interval": interval_summary(defined_delta),
            }
            pooled[metric].extend(float(value) for value in defined_delta)
            levels["trunk"][metric].extend(float(value) for value in trunk_values[defined])
            levels["candidate"][metric].extend(float(value) for value in candidate_values[defined])
            levels["fused"][metric].extend(float(value) for value in fused_values[defined])
        labels.extend([window] * len(merged))
        dates.extend(str(value) for value in merged["date"])
    series = {"days": dates, "windows": labels}
    for metric, values in pooled.items():
        # delta_net_return_5bp 只覆盖配对差有定义的交易日，与 days/windows 的完整列表不对齐。
        series[f"delta_{metric}"] = values
    for arm, block in levels.items():
        for metric, values in block.items():
            series[f"{arm}_{metric}"] = values
    pooled_summary = {}
    for metric, values in pooled.items():
        pooled_summary[metric] = interval_summary(values)
        pooled_summary[metric]["days_defined"] = int(len(values))
        pooled_summary[metric]["days_undefined"] = int(
            sum(per_window[window][metric]["days_undefined"] for window in windows))
    return {
        "weight": float(weight),
        "per_window": per_window,
        "pooled": pooled_summary,
        "series": series,
    }


def positive_windows(read: dict, windows: list[str]) -> dict:
    names = [window for window in windows
             if (read["per_window"][window][PRIMARY_METRIC]["delta_mean"] or 0.0) > 0.0]
    return {"unit": "window", "count": len(names), "of": len(windows), "windows": names}


def positive_days(read: dict) -> dict:
    values = np.asarray(read["series"][f"delta_{PRIMARY_METRIC}"], dtype=float)
    values = values[np.isfinite(values)]
    return {"unit": "trading_day", "count": int((values > 0.0).sum()), "of": int(values.size)}


def post_freeze_guard(read: dict, registration: dict) -> dict:
    spec = registration["fusion_increment_definition"]["weight_fit"]
    start, end = str(spec["start"])[:10], str(spec["end"])[:10]
    windows, days = read["series"]["windows"], read["series"]["days"]
    deltas = read["series"][f"delta_{PRIMARY_METRIC}"]
    keep = [index for index, (window, day) in enumerate(zip(windows, days))
            if not (window == spec["window"] and start <= day <= end)]
    summary = interval_summary([deltas[index] for index in keep])
    summary.update({
        "days_excluded": int(len(days) - len(keep)),
        "excluded_block": f"{spec['window']} {start}..{end}",
        "condition": "pooled paired mean of the days strictly after the weight-fit block > 0",
        "rationale": "只收紧不放松：它只能在 165 天读数已经通过之后再否掉一个方向",
    })
    return summary


# --------------------------------------------------------------------------------------- #
# mandatory side reports
# --------------------------------------------------------------------------------------- #
def cost_report(spec: dict | None, *, arm: str) -> dict:
    """把 inference_cost.json 折算成每交易日秒数；缺文件就是缺读数，不是零成本。"""
    if not spec:
        return {"arm": arm, "available": False, "reason": "plan carries no inference cost spec"}
    paths = list(spec.get("paths") or [])
    if spec.get("path"):
        paths.append(spec["path"])
    if not paths:
        return {"arm": arm, "available": False, "reason": "cost spec lists no files"}
    files = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            return {"arm": arm, "available": False, "reason": f"missing inference cost: {path}"}
        payload = read_json(path)
        chunks = payload.get("chunks") or []
        per_seed = sum(float(sum((chunk.get("per_seed_seconds") or {}).values()))
                       for chunk in chunks)
        shared = payload.get("shared_preprocessing_seconds")
        if shared is None:
            shared = sum(float(chunk.get("shared_preprocessing_seconds") or 0.0)
                         for chunk in chunks)
        total = payload.get("total_seconds")
        if total is None:
            total = per_seed + float(shared or 0.0)
        days = sorted({str(day) for chunk in chunks for day in (chunk.get("dates") or [])})
        peaks = [float(chunk["resources"]["gpu_peak_memory_mb"]) for chunk in chunks
                 if (chunk.get("resources") or {}).get("gpu_peak_memory_mb") is not None]
        files.append({
            "path": str(path),
            "measurement": payload.get("measurement"),
            "platform_verified": bool(payload.get("platform_verified")),
            "total_seconds": _number(total),
            "per_seed_seconds": _number(per_seed),
            "shared_preprocessing_seconds": _number(shared),
            "chunk_days": len(days),
            "gpu_peak_memory_mb": max(peaks) if peaks else None,
        })
    total_seconds = sum(float(row["total_seconds"] or 0.0) for row in files)
    declared = spec.get("days_covered")
    days_covered = int(declared) if declared else max((row["chunk_days"] for row in files),
                                                      default=0)
    if not days_covered:
        return {"arm": arm, "available": False,
                "reason": "inference cost chunks carry no dates and days_covered is unset",
                "files": files}
    return {
        "arm": arm,
        "label": spec.get("label"),
        "available": True,
        "files": files,
        "days_covered": days_covered,
        "total_seconds": _number(total_seconds),
        "seconds_per_trading_day": _number(total_seconds / days_covered),
        "gpu_peak_memory_mb": max((row["gpu_peak_memory_mb"] or 0.0) for row in files) or None,
        "platform_verified": all(row["platform_verified"] for row in files),
        "marginal_note": "主干已经付过钱；融合臂的边际成本就是候选自己的前向 + 一次线性组合",
    }


def turnover_annotation(read: dict) -> dict:
    """纯标注：融合臂相对主干的换手配对差越过晋升线的换手退化量时打旗。

    这条 flag 不进入任何 pass/fail 判断，也不改变 verdict 取值：换手在筛选层只报
    （队长 2026-10-28 决定），真正的换手门在晋升阶段（turnover_mean 退化线 0.0318）；
    这里打旗是为了让晋升阶段一定看到。
    """
    pooled = read["pooled"].get("turnover") or {}
    mean = pooled.get("mean")
    fired = bool(mean is not None and mean > TURNOVER_ATTENTION_THRESHOLD)
    return {
        "turnover_attention": fired,
        "delta_turnover_mean": mean,
        "turnover_attention_threshold": TURNOVER_ATTENTION_THRESHOLD,
        "rule_zh": (f"融合臂相对主干的日度 Δturnover 池化均值 > +{TURNOVER_ATTENTION_THRESHOLD}"
                    "（441 候选晋升时用过的 turnover_mean 退化线）时置 true；只做标注，"
                    "不改变 verdict，换手本身仍然只报不门控"),
        "does_not_gate": True,
    }


def build_side_reports(plan: dict, read: dict) -> dict:
    """多空日均、换手、净收益(5bp)、推理成本四条必须同时出现的旁证。"""
    pooled = read["pooled"]
    per_window = read["per_window"]

    def level(arm: str, metric: str) -> float | None:
        return _number(np.nanmean([row[metric][f"{arm}_mean"] for row in per_window.values()]))

    candidate_cost = cost_report(plan["candidate"].get("cost"), arm="candidate")
    incumbent_cost = cost_report(plan["incumbent"].get("cost"), arm="incumbent")
    net5_available = bool(pooled[NET_METRIC].get("mean") is not None
                          and (pooled[NET_METRIC].get("days_defined") or 0) > 0)
    return {
        "long_short_daily_mean": {
            "caliber": "eval-v2 日度 long_short_return，按窗口等权平均",
            "trunk": level("trunk", "long_short_return"),
            "candidate": level("candidate", "long_short_return"),
            "fused": level("fused", "long_short_return"),
            "paired_delta": pooled["long_short_return"],
            "policy": "描述性：rank_ic 增量为正而多空为负时上报队长，但它不重算闸门结论",
        },
        "turnover": {
            "caliber": "eval-v2 turnover = U/2，日度均值；越低越好",
            "trunk": level("trunk", "turnover"),
            "candidate": level("candidate", "turnover"),
            "fused": level("fused", "turnover"),
            "paired_delta": pooled["turnover"],
            "policy": "只报不门控；通过但换手明显上升的路线由队长在晋升时权衡",
        },
        "net_return_5bp": {
            "caliber": ("直接读 evaluate(single_side_rate=5e-4) sections 的 net_return（日度按等时间戳"
                        "均值、未定义时点跳过不零填）；不能用 long_short_return - 5e-4*U 反推，"
                        "因为 net 与 gross 覆盖的时点集可能不同"),
            "trunk": level("trunk", NET_METRIC),
            "candidate": level("candidate", NET_METRIC),
            "fused": level("fused", NET_METRIC),
            "paired_delta": pooled[NET_METRIC],
            "days_defined": pooled[NET_METRIC].get("days_defined"),
            "days_undefined": pooled[NET_METRIC].get("days_undefined"),
            "policy": ("只报不门控：晋升阶段的 net5_limit = 5.62e-05 才是门；"
                       "但缺 net5 的读数算不完整，同样不允许 clears"),
        },
        "inference_cost": {
            "candidate": candidate_cost,
            "incumbent": incumbent_cost,
            "available": bool(candidate_cost["available"]),
            "fused_marginal_seconds_per_day": candidate_cost.get("seconds_per_trading_day")
            if candidate_cost["available"] else None,
            "policy": "缺成本报告 ⇒ 读数不完整 ⇒ 不允许 clears_fusion_gate",
        },
        "annotations": turnover_annotation(read),
        "complete": bool(candidate_cost["available"] and net5_available),
    }


# --------------------------------------------------------------------------------------- #
# decision
# --------------------------------------------------------------------------------------- #
def decide_screen(*, registration: dict, plan: dict, fit: dict, read: dict, guard: dict,
                  positives: dict, cost: dict, side_complete: bool = True) -> dict:
    decision = registration["decision"]
    pooled = read["pooled"][PRIMARY_METRIC]
    thresholds = {
        "min_improvement": decision["min_improvement"],
        "ci_must_exclude_zero": decision["ci_must_exclude_zero"],
        "min_positive_windows": decision["min_positive_windows"],
        "windows_total": decision["windows_total"],
        "post_freeze_mean_positive": True,
        "cost_report_required": True,
        "weight_must_exceed": 0.0,
        "estimator": "block_interval(block=5, draws=20000, seed=20261028)",
    }
    if plan["arm_kind"] == "single_model":
        return {
            "verdict": "descriptive_only",
            "verdict_zh": "单模只报描述：它没有以注册权重并入主干，不能进入扩种子轮",
            "fail_reasons": [],
            "thresholds": thresholds,
            "annotations": turnover_annotation(read),
            "clears": False,
        }
    failures = []
    mean = pooled["mean"]
    if mean is None or mean < decision["min_improvement"]:
        failures.append("below_threshold")
    if not pooled["ci_low"] or pooled["ci_low"] <= 0.0:
        failures.append("interval_includes_zero")
    if positives["count"] < decision["min_positive_windows"]:
        failures.append("positive_windows_below_7")
    if guard["mean"] is None or guard["mean"] <= 0.0:
        failures.append("post_freeze_mean_not_positive")
    if not cost["available"] or not side_complete:
        # 缺推理成本或缺 5bp 净收益旁证都算读数不完整（注册文件 mandatory_side_reports）。
        failures.append("blocked_missing_cost_report")
    if fit["weight"] <= 0.0:
        return {
            "verdict": "no_allocation",
            "verdict_zh": "拟合块把权重选成 0：这条路线的信息在主干之外没有可分配份额",
            "fail_reasons": sorted(set(failures) | {"no_allocation"}),
            "thresholds": thresholds,
            "annotations": turnover_annotation(read),
            "clears": False,
        }
    verdict = "clears_fusion_gate" if not failures else failures[0]
    return {
        "verdict": verdict,
        "verdict_zh": {
            "clears_fusion_gate": "通过：可以进入下一轮扩种子（多点种子/更多窗），不是提交或晋升",
            "below_threshold": "池化均值低于 +0.003 的注册门槛",
            "interval_includes_zero": "修正 CI 含 0，日配对差不足以排除噪声",
            "positive_windows_below_7": "为正的筛查窗少于 7/8，改善不是普遍方向",
            "post_freeze_mean_not_positive": "冻结后（155 天）均值不为正：增量来自拟合块本身",
            "blocked_missing_cost_report": "推理成本或 5bp 净收益(net5)旁证没报，读数不完整，不允许通过",
        }.get(verdict, verdict),
        "fail_reasons": failures,
        "thresholds": thresholds,
        "annotations": turnover_annotation(read),
        "clears": verdict == "clears_fusion_gate",
    }


def decide_confirmation(*, registration: dict, read: dict, positives: dict) -> dict:
    decision = registration["decision"]["confirmation"]
    pooled = read["pooled"][PRIMARY_METRIC]
    passes = bool(pooled["mean"] is not None and pooled["mean"] > 0.0
                  and positives["count"] >= decision["min_positive_days"])
    return {
        "verdict": "confirmation_passes" if passes else "confirmation_fails",
        "criterion": (f"pooled paired mean > 0 and >= {decision['min_positive_days']}/"
                      f"{decision['days']} positive days"),
        "pooled_mean": pooled["mean"],
        "ci_low": pooled["ci_low"],
        "positive_days": positives["count"],
        "days": pooled["days"],
        "passes": passes,
        "fail_reasons": [] if passes else ["confirmation_criterion_not_met"],
        "annotations": turnover_annotation(read),
        "note_zh": "否决线不是晋升线：38 天无法建立显著性，只用来否掉筛查期侥幸通过的方向",
    }


def headline(result: dict) -> dict:
    """账本里的一行摘要：够复现口径，又不必读整份结果。"""
    pooled = result["read"]["pooled"][PRIMARY_METRIC]
    decision = result["decision"]
    return {
        "weight": result["weight"]["weight"],
        "verdict": decision["verdict"],
        "fail_reasons": decision.get("fail_reasons", []),
        "pooled_mean": pooled["mean"],
        "ci_low": pooled["ci_low"],
        "ci_high": pooled["ci_high"],
        "days": pooled["days"],
        "positive_windows": result["positives"]["count"],
        "post_freeze_mean": result["post_freeze"]["mean"],
        "turnover_delta": result["read"]["pooled"]["turnover"]["mean"],
        "turnover_attention": decision.get("annotations", {}).get("turnover_attention"),
        "net5_delta": result["read"]["pooled"][NET_METRIC]["mean"],
        "long_short_delta": result["read"]["pooled"]["long_short_return"]["mean"],
        "cost_seconds_per_day": result["side_reports"]["inference_cost"]["candidate"].get(
            "seconds_per_trading_day"),
    }


# --------------------------------------------------------------------------------------- #
# read ledger (append-only; screen repeatable, confirmation once per route)
# --------------------------------------------------------------------------------------- #
def ledger_entries(path: str | Path = DEFAULT_LEDGER) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def append_ledger(entry: dict, path: str | Path = DEFAULT_LEDGER) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(entry)
    record.setdefault("read_id", f"{entry['mode']}-{entry['route']}-{uuid.uuid4().hex[:10]}")
    record.setdefault("utc", utc_now())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def confirmation_reads(route: str, path: str | Path = DEFAULT_LEDGER) -> list[dict]:
    return [row for row in ledger_entries(path)
            if row.get("route") == route and row.get("mode") == "confirm"]


# --------------------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------------------- #
def run_audit(registration: dict, plan: dict | None = None,
              ledger: str | Path = DEFAULT_LEDGER) -> dict:
    """不读任何预测：只核对注册、窗口角色、estimator 与（如有）路线计划的一致性。"""
    roles = registration["window_roles"]
    fusion = registration["fusion_increment_definition"]
    decision = registration["decision"]
    script = gate_script_state(registration)
    summary = {
        "mode": "audit",
        "reads_any_prediction": False,
        "registration": {"path": registration["_path"], "sha256": registration["_sha256"]},
        "gate_script": script,
        "estimator": {
            "implementation": registration["estimator"]["implementation"],
            "sha256": file_sha256(Path(REPO / registration["estimator"]["implementation"]
                                       .split("::")[0])),
            "block_days": BLOCK_DAYS,
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
        },
        "window_roles": {
            "authority": roles["authority"],
            "replay_plan": roles["replay_plan"],
            "screen": roles["screen"],
            "confirmation": roles["confirmation"],
            "screen_days": roles["screen_days"],
            "confirmation_days": roles["confirmation_days"],
            "total_days": roles["total_days"],
            "post_freeze_days": roles["post_freeze_days"],
        },
        "weight_fit": {
            "window": fusion["weight_fit"]["window"],
            "start": fusion["weight_fit"]["start"],
            "end": fusion["weight_fit"]["end"],
            "trading_days": fusion["weight_fit"]["trading_days"],
            "grid": fusion["weight_fit"]["grid"],
        },
        "decision": {
            "metric": decision["metric"],
            "min_improvement": decision["min_improvement"],
            "ci_must_exclude_zero": decision["ci_must_exclude_zero"],
            "min_positive_windows": decision["min_positive_windows"],
            "windows_total": decision["windows_total"],
            "post_freeze_guard": decision["post_freeze_guard"],
            "confirmation": decision["confirmation"],
        },
        "mandatory_side_reports": sorted(
            key for key in registration["mandatory_side_reports"] if not key.endswith("_zh")),
        "ledger": {"path": str(Path(ledger)), "reads_total": len(ledger_entries(ledger))},
    }
    if plan is not None:
        route_reads = ledger_entries(ledger)
        summary["plan"] = {
            "path": plan["_path"],
            "sha256": plan["_sha256"],
            "route": plan["route"],
            "arm_kind": plan["arm_kind"],
            "candidate": plan["candidate"]["label"],
            "incumbent_run": plan["incumbent"]["run_dir"],
            "candidate_run": plan["candidate"]["run_dir"],
            "incumbent_versions": [member["version"]
                                   for member in plan["incumbent"]["members"]],
            "candidate_versions": [member["version"]
                                   for member in plan["candidate"]["members"]],
            "candidate_cost_available": cost_report(plan["candidate"].get("cost"),
                                                    arm="candidate")["available"],
            "screening_reads_on_record": len([row for row in route_reads
                                              if row.get("route") == plan["route"]
                                              and row.get("mode") == "screen"]),
            "confirmation_reads_on_record": len(confirmation_reads(plan["route"], ledger)),
        }
    return summary


def run_gate(plan: dict, registration: dict, *, mode: str, ledger: str | Path,
             frozen_weight: float | None = None) -> dict:
    script = gate_script_state(registration)
    if not script["matches"]:
        raise SystemExit(
            "gate script changed since the registration froze it "
            f"(pinned {script['pinned']!r}, actual {script['actual']!r}); restore the pinned "
            "script or re-register — no read may run against an unpinned judge")
    roles = registration["window_roles"]
    windows = list(roles["screen"] if mode in {"screen", "single"} else roles["confirmation"])
    if mode in {"screen", "confirm"} and plan["arm_kind"] != "fused":
        raise SystemExit(
            f"{plan['_path']}: arm_kind=single_model only supports --mode single "
            "(a bare model has no fusion increment to judge)")
    cache = WindowCache(plan, registration)
    if plan["arm_kind"] == "fused":
        fit = fit_weight(cache, registration)
        if frozen_weight is not None and abs(fit["weight"] - float(frozen_weight)) > 1e-12:
            raise SystemExit(
                f"refit detected: screening read froze w={frozen_weight} but the fit now "
                f"returns {fit['weight']}; the weight must not move after a read")
    else:
        fit = {"window": None, "weight": 1.0, "trading_days": 0, "profile": [],
               "criterion": "n/a",
               "note": "single_model: fused(w=1) 就是候选本身，没有拟合"}
    read = read_windows(cache, windows, fit["weight"])
    expected_days = SCREEN_DAYS if mode in {"screen", "single"} else CONFIRMATION_DAYS
    total_days = len(read["series"]["days"])
    if total_days != expected_days:
        raise SystemExit(f"{mode}: {total_days} paired days but the registration pins "
                         f"{expected_days}")
    positives = positive_windows(read, windows)
    days = positive_days(read)
    guard = post_freeze_guard(read, registration)
    side = build_side_reports(plan, read)
    if mode == "confirm":
        decision = decide_confirmation(registration=registration, read=read, positives=days)
    else:
        decision = decide_screen(registration=registration, plan=plan, fit=fit, read=read,
                                 guard=guard, positives=positives,
                                 cost=side["inference_cost"], side_complete=side["complete"])
    result = {
        "mode": mode,
        "evaluation_version": EVALUATION_VERSION,
        "gate": {"path": str(EXECUTABLE), "sha256": script["actual"]},
        "registration": {"path": registration["_path"],
                         "sha256": registration["_sha256"]},
        "plan": {"path": plan["_path"], "sha256": plan["_sha256"], "route": plan["route"],
                 "arm_kind": plan["arm_kind"], "candidate": plan["candidate"]["label"]},
        "windows": windows,
        "paired_days": total_days,
        "weight": fit,
        "read": read,
        "positives": positives,
        "positive_days": days,
        "post_freeze": guard,
        "side_reports": side,
        "decision": decision,
        "confirmation": decision if mode == "confirm" else None,
        "selection_bias": {
            "ten_window_reuse": registration["selection_bias_acknowledgment"][
                "ten_window_reuse_zh"],
            "multiplicity": registration["selection_bias_acknowledgment"]["multiplicity"],
            "reading": "165 天不是新鲜样本；通过只说明'值得扩种子'，不说明有超额",
        },
    }
    recorded = append_ledger({
        "route": plan["route"],
        "mode": mode,
        "plan_path": plan["_path"],
        "plan_sha256": plan["_sha256"],
        "registration_sha256": registration["_sha256"],
        "gate_sha256": script["actual"],
        "windows": windows,
        "headline": headline(result),
    }, ledger)
    result["ledger"] = {"read_id": recorded["read_id"], "utc": recorded["utc"],
                        "route": recorded["route"], "mode": recorded["mode"],
                        "path": str(Path(ledger))}
    return result


def run_confirm(plan: dict, registration: dict, *, ledger, release_path) -> dict:
    route = plan["route"]
    if not release_path:
        raise SystemExit("confirm reads require --release (the captain's one-shot release file)")
    release = read_json(Path(release_path))
    if release.get("route") != route:
        raise SystemExit(f"release file is for route {release.get('route')!r}, not {route!r}")
    if release.get("released_by") != "captain":
        raise SystemExit("release file must be signed released_by='captain'")
    existing = confirmation_reads(route, ledger)
    if existing:
        raise SystemExit(
            f"route {route} already spent its confirmation read: {existing[0]['read_id']} "
            f"({existing[0]['utc']}); the 38 confirmation days are one-shot per direction")
    screening = [row for row in ledger_entries(ledger)
                 if row.get("mode") == "screen" and row.get("route") == route
                 and row.get("read_id") == release.get("screening_read_id")]
    if not screening:
        raise SystemExit(
            f"release references screening read {release.get('screening_read_id')!r} which is "
            "not in the ledger for this route")
    if screening[0]["headline"]["verdict"] != "clears_fusion_gate":
        raise SystemExit(
            f"screening read {screening[0]['read_id']} verdict "
            f"{screening[0]['headline']['verdict']!r} does not unlock confirmation")
    result = run_gate(plan, registration, mode="confirm", ledger=ledger,
                      frozen_weight=screening[0]["headline"]["weight"])
    result["confirmation"]["screening_read_id"] = screening[0]["read_id"]
    result["confirmation"]["release_path"] = str(Path(release_path))
    append_ledger({
        "route": route,
        "mode": "confirm_verdict",
        "plan_path": plan["_path"],
        "registration_sha256": registration["_sha256"],
        "screening_read_id": screening[0]["read_id"],
        "headline": {"passes": result["confirmation"]["passes"],
                     "pooled_mean": result["confirmation"]["pooled_mean"],
                     "positive_days": result["confirmation"]["positive_days"]},
    }, ledger)
    return result


def run_single(plan: dict, registration: dict, *, ledger) -> dict:
    """单模只报描述：不拟合权重、不并入主干、不出闸门结论。"""
    plan = dict(plan)
    plan["arm_kind"] = "single_model"
    result = run_gate(plan, registration, mode="single", ledger=ledger)
    if result["decision"]["verdict"] != "descriptive_only":
        raise SystemExit("single-model mode must never emit a gate verdict")
    return result


# --------------------------------------------------------------------------------------- #
# self-test (synthetic data through the same code path)
# --------------------------------------------------------------------------------------- #
def _synthetic_frames(root: Path, *, days_by_window: dict[str, int], instruments: int,
                      times_per_day: int, rng: np.random.Generator, variant: str) -> None:
    """造一份与回放目录同形的合成数据。

    trunk 与候选共享同一个 latent，但各自带独立噪声实现；融合降噪即产生真实增量。
    variant:
      positive  — 候选在所有交易日都携带 latent（应该通过）
      duplicate — 候选就是 trunk 的副本（应选 w=0）
      noise     — 候选与 latent 无关（不应通过）
      fit_only  — 候选只在拟合块里携带 latent（post-freeze guard 必须否掉）
    """
    for window, count in days_by_window.items():
        days = pd.bdate_range(pd.Timestamp(WINDOW_STARTS[window]), periods=count)
        instruments_index = [f"SYN{index:03d}" for index in range(instruments)]
        rows = []
        for day in days:
            for stamp in range(times_per_day):
                moment = day + pd.Timedelta(hours=9, minutes=30 + 5 * stamp)
                latent = rng.standard_normal(instruments)
                noise_trunk = rng.standard_normal(instruments)
                noise_candidate = rng.standard_normal(instruments)
                target = latent + 0.25 * rng.standard_normal(instruments)
                trunk = 0.35 * latent + noise_trunk
                if variant == "duplicate":
                    candidate = trunk.copy()
                elif variant == "noise":
                    candidate = rng.standard_normal(instruments)
                elif variant == "fit_only":
                    carries = str(day.date()) <= FIT_END
                    candidate = 0.35 * (latent if carries
                                        else rng.standard_normal(instruments)) + noise_candidate
                else:
                    candidate = 0.35 * latent + noise_candidate
                for index, instrument in enumerate(instruments_index):
                    rows.append({"date": moment, "instrument": instrument,
                                 "target": float(target[index]),
                                 "trunk": float(trunk[index]),
                                 "candidate": float(candidate[index])})
        frame = pd.DataFrame(rows)
        base = root / "windows" / window
        base.mkdir(parents=True, exist_ok=True)
        frame[KEYS].to_parquet(base / "keys.parquet", index=False)
        frame[KEYS + ["target"]].to_parquet(base / "targets.parquet", index=False)
        for name, column in (("synth_trunk", "trunk"), ("synth_candidate", "candidate")):
            arm_dir = base / name
            arm_dir.mkdir(parents=True, exist_ok=True)
            frame[KEYS].assign(score=frame[column].to_numpy()).to_parquet(
                arm_dir / "predictions.parquet", index=False)


def synthetic_case(root: str | Path, *, variant: str = "positive", route: str = "R99",
                   registration_sha256: str, instruments: int = 24, times_per_day: int = 2,
                   seed: int = 7) -> dict:
    """构造一条合成路线（plan + run dir + 成本文件），供 --self-test 与测试复用。"""
    root = Path(root)
    rng = np.random.default_rng(seed)
    days_by_window = {"nov2023": 22, "jan2024": 22, "feb2024": 15, "mar2024": 21,
                      "apr2024": 20, "may2024": 20, "jun2024": 19, "jul2024": 23,
                      "aug2024": 22, "sep2024": 19}
    run_dir = root / "runs" / route.lower()
    _synthetic_frames(run_dir, days_by_window=days_by_window, instruments=instruments,
                      times_per_day=times_per_day, rng=rng, variant=variant)
    cost_dir = run_dir / "inference"
    cost_dir.mkdir(parents=True, exist_ok=True)
    for name, per_seed in (("synth_candidate", 12.5), ("synth_trunk", 25.0)):
        write_json(cost_dir / f"{name}_inference_cost.json", {
            "arm": name,
            "chunks": [{"chunk": 0,
                        "dates": [f"2023-11-{day:02d}" for day in (1, 2, 3, 6, 7)],
                        "per_seed_seconds": {"11": per_seed},
                        "resources": {"gpu_peak_memory_mb": 96.0},
                        "samples": 4000,
                        "shared_preprocessing_seconds": 3.0,
                        "timing_order": [11]}],
            "measurement": "synthetic; self-test only",
            "platform_verified": False,
            "shared_preprocessing_seconds": 3.0,
            "total_seconds": per_seed + 3.0,
        })
    plan = {
        "schema_version": PLAN_SCHEMA,
        "experiment_id": f"synthetic-{route.lower()}-{variant}",
        "route": route,
        "arm_kind": "fused",
        "registration": {"path": str(DEFAULT_REGISTRATION), "sha256": registration_sha256},
        "incumbent": {
            "label": "synthetic trunk",
            "run_dir": str(run_dir),
            "members": [{"seed": "11", "version": "synth_trunk"}],
            "cost": {"paths": [str(cost_dir / "synth_trunk_inference_cost.json")],
                     "label": "synthetic trunk"},
        },
        "candidate": {
            "label": f"synthetic candidate ({variant})",
            "run_dir": str(run_dir),
            "members": [{"seed": "11", "version": "synth_candidate"}],
            "cost": {"paths": [str(cost_dir / "synth_candidate_inference_cost.json")],
                     "label": "synthetic candidate"},
        },
    }
    plan_path = root / f"{route.lower()}_{variant}_plan.json"
    write_json(plan_path, plan)
    return {"plan": plan_path, "run_dir": run_dir, "variant": variant, "route": route}


def run_self_test(base: str | Path | None = None) -> dict:
    import tempfile

    checks: list[dict] = []
    with tempfile.TemporaryDirectory(dir=str(base) if base else None) as workdir:
        root = Path(workdir)
        registration = load_registration()
        sha = registration["_sha256"]
        ledger = root / "ledger.jsonl"

        def check(name: str, passed: bool, detail: str = "") -> None:
            if not passed:
                raise SystemExit(f"self-test failed: {name} {detail}")
            checks.append({"check": name, "passed": True, "detail": detail})

        def gate(plan_path: Path, mode: str) -> dict:
            plan = load_plan(plan_path, registration)
            if mode == "single":
                return run_single(plan, registration, ledger=ledger)
            return run_gate(plan, registration, mode=mode, ledger=ledger)

        positive = synthetic_case(root, variant="positive", route="R91",
                                  registration_sha256=sha)
        result = gate(positive["plan"], "screen")
        check("positive-candidate-clears",
              result["decision"]["verdict"] == "clears_fusion_gate",
              f"verdict={result['decision']['verdict']} "
              f"mean={result['read']['pooled'][PRIMARY_METRIC]['mean']:.4f} "
              f"w={result['weight']['weight']}")
        check("weight-is-frozen-in-result", result["weight"]["weight"] in WEIGHT_GRID)
        check("screen-days-are-165", len(result["read"]["series"]["days"]) == SCREEN_DAYS)
        check("ledger-recorded-screen-read",
              [row["mode"] for row in ledger_entries(ledger)] == ["screen"])

        plan_positive = load_plan(positive["plan"], registration)
        entry = WindowCache(plan_positive, registration).get("nov2023")
        fused_scores = fuse(entry["incumbent"], entry["candidate"], result["weight"]["weight"])
        derived = daily_metrics(entry["keys"], entry["targets"], fused_scores)

        # net5 必须直读 evaluate(single_side_rate=5e-4) 的 net_return；逐日恒等式
        # net5 = net0 - cost5 只在 eval-v2 自己的 sections 上成立（同一批"change 有定义"的时点）。
        frame_5 = evaluate(
            entry["keys"][KEYS].assign(score=fused_scores), entry["targets"],
            evaluation_config(COST_SINGLE_SIDE_RATE),
            return_sections=True, undefined_constant_portfolios=True)[1]
        frame_0 = evaluate(
            entry["keys"][KEYS].assign(score=fused_scores), entry["targets"],
            evaluation_config(0.0),
            return_sections=True, undefined_constant_portfolios=True)[1]
        cost_check = derived[["date", NET_METRIC]].merge(
            frame_0[["date", "net_return"]].rename(columns={"net_return": "net0"}),
            on="date", validate="one_to_one").merge(
            frame_5[["date", "cost"]].rename(columns={"cost": "cost5"}),
            on="date", validate="one_to_one")
        identity_gap = float(np.max(np.abs((cost_check["net0"] - cost_check["cost5"])
                                           - cost_check[NET_METRIC])))
        check("net5-is-the-evalv2-net-at-5bp", identity_gap < 1e-12,
              f"max|net0 - cost5 - net5| = {identity_gap:.3e} over {len(cost_check)} days")

        compared = [PRIMARY_METRIC, *SIDE_METRICS]
        left = frame_0[["date", *compared]].sort_values("date").reset_index(drop=True)
        right = frame_5[["date", *compared]].sort_values("date").reset_index(drop=True)
        same_days = bool(left["date"].equals(right["date"]))
        left_values = left[compared].to_numpy(dtype=float)
        right_values = right[compared].to_numpy(dtype=float)
        # 成本参数只该改 cost/net，不该改任何一个 gross 指标，也不该改哪些天有定义。
        same_defined = bool(np.array_equal(np.isnan(left_values), np.isnan(right_values)))
        both = np.isfinite(left_values) & np.isfinite(right_values)
        rate_gap = float(np.max(np.abs(left_values[both] - right_values[both]))) if both.any() else 0.0
        check("cost-rate-leaves-other-metrics-unchanged",
              same_days and same_defined and rate_gap < 1e-12,
              f"max|rate0 - rate5| on rank_ic/turnover/long_short = {rate_gap:.3e}; "
              f"same_defined={same_defined}")

        pooled_5 = result["read"]["pooled"]
        counted = ((pooled_5[NET_METRIC].get("days_defined") or 0)
                   + (pooled_5[NET_METRIC].get("days_undefined") or 0))
        check("net5-undefined-days-are-counted",
              counted == SCREEN_DAYS
              and (pooled_5[NET_METRIC].get("days_defined") or 0) >= 150
              and (pooled_5[PRIMARY_METRIC].get("days_undefined") or 0) == 0,
              f"net5 defined={pooled_5[NET_METRIC].get('days_defined')} "
              f"undefined={pooled_5[NET_METRIC].get('days_undefined')}; "
              f"rank_ic undefined={pooled_5[PRIMARY_METRIC].get('days_undefined')}")

        side = result["side_reports"]
        check("side-reports-carry-net5",
              side["complete"] is True
              and side["net_return_5bp"]["trunk"] is not None
              and side["net_return_5bp"]["fused"] is not None
              and side["net_return_5bp"]["paired_delta"]["days"]
              == side["net_return_5bp"]["days_defined"]
              and side["annotations"]["turnover_attention"] in (True, False),
              f"net5 delta={side['net_return_5bp']['paired_delta']['mean']} "
              f"days={side['net_return_5bp']['paired_delta']['days']}")

        quiet_read = json.loads(json.dumps(result["read"]))
        quiet_read["pooled"]["turnover"]["mean"] = TURNOVER_ATTENTION_THRESHOLD - 0.01
        loud_read = json.loads(json.dumps(result["read"]))
        loud_read["pooled"]["turnover"]["mean"] = TURNOVER_ATTENTION_THRESHOLD + 0.01
        decisions = [
            decide_screen(registration=registration, plan=plan_positive, fit=result["weight"],
                          read=body, guard=result["post_freeze"], positives=result["positives"],
                          cost=side["inference_cost"], side_complete=side["complete"])
            for body in (quiet_read, loud_read)
        ]
        check("turnover-attention-does-not-gate",
              decisions[0]["verdict"] == result["decision"]["verdict"] == decisions[1]["verdict"]
              and (decisions[0]["clears"] == result["decision"]["clears"]
                   == decisions[1]["clears"])
              and decisions[0]["annotations"]["turnover_attention"] is False
              and decisions[1]["annotations"]["turnover_attention"] is True,
              f"verdict {result['decision']['verdict']} unchanged; att quiet->False loud->True")

        direct = block_interval(np.asarray(result["read"]["series"][f"delta_{PRIMARY_METRIC}"],
                                           dtype=float),
                                block=BLOCK_DAYS, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
        pooled = result["read"]["pooled"][PRIMARY_METRIC]
        check("pooled-interval-matches-block_interval",
              abs(direct["mean"] - pooled["mean"]) < 1e-12
              and abs(direct["ci_low"] - pooled["ci_low"]) < 1e-12,
              f"mean={direct['mean']:.6f} ci_low={direct['ci_low']:.6f}")

        duplicate = synthetic_case(root, variant="duplicate", route="R92",
                                   registration_sha256=sha)
        duplicate_result = gate(duplicate["plan"], "screen")
        check("duplicate-candidate-gets-no-allocation",
              duplicate_result["decision"]["verdict"] == "no_allocation"
              and duplicate_result["weight"]["weight"] == 0.0,
              f"w={duplicate_result['weight']['weight']}")

        noise = synthetic_case(root, variant="noise", route="R93", registration_sha256=sha)
        noise_result = gate(noise["plan"], "screen")
        check("noise-candidate-never-clears",
              not noise_result["decision"]["clears"],
              f"w={noise_result['weight']['weight']} "
              f"fail={noise_result['decision']['fail_reasons']}")

        fit_only = synthetic_case(root, variant="fit_only", route="R94", registration_sha256=sha)
        fit_only_result = gate(fit_only["plan"], "screen")
        check("fit-only-gain-is-caught-by-post-freeze-guard",
              fit_only_result["weight"]["weight"] > 0.0
              and fit_only_result["post_freeze"]["mean"] <= 0.0
              and not fit_only_result["decision"]["clears"],
              f"w={fit_only_result['weight']['weight']} "
              f"guard_mean={fit_only_result['post_freeze']['mean']:.4f} "
              f"fail={fit_only_result['decision']['fail_reasons']}")

        single = gate(positive["plan"], "single")
        check("single-model-is-descriptive-only",
              single["decision"]["verdict"] == "descriptive_only")

        entry = WindowCache(load_plan(positive["plan"], registration),
                            registration).get("jan2024")
        trunk = entry["incumbent"]
        check("zero-weight-is-bit-identical-to-trunk",
              np.array_equal(trunk, fuse(trunk, entry["candidate"], 0.0)))
        trunk_daily = daily_metrics(entry["keys"], entry["targets"], trunk)
        zero_daily = daily_metrics(entry["keys"], entry["targets"],
                                  fuse(trunk, entry["candidate"], 0.0))
        check("zero-weight-reproduces-daily-metrics",
              np.allclose(trunk_daily[PRIMARY_METRIC].to_numpy(float),
                          zero_daily[PRIMARY_METRIC].to_numpy(float), atol=1e-12))

        from mktctx_compare import daily_scores as house_daily_scores

        raw = np.mean(np.stack([load_scores(positive["run_dir"], "jan2024", "synth_trunk",
                                            entry["keys"])]), axis=0)
        house = house_daily_scores(entry["keys"], entry["targets"], raw, smooth=True)
        check("arm-matches-house-daily_scores",
              np.allclose(house.sort_values("date")[PRIMARY_METRIC].to_numpy(float),
                          trunk_daily.sort_values("date")[PRIMARY_METRIC].to_numpy(float),
                          atol=1e-12))

        screening_id = [row for row in ledger_entries(ledger)
                        if row["mode"] == "screen" and row["route"] == "R91"][0]["read_id"]
        release = root / "release.json"
        write_json(release, {"route": "R91", "released_by": "captain",
                             "screening_read_id": screening_id})
        plan = load_plan(positive["plan"], registration)
        confirm = run_confirm(plan, registration, ledger=ledger, release_path=release)
        check("confirmation-runs-once-with-release",
              confirm["confirmation"]["passes"] is True,
              f"positive_days={confirm['confirmation']['positive_days']}/38 "
              f"mean={confirm['confirmation']['pooled_mean']:.4f}")
        try:
            run_confirm(plan, registration, ledger=ledger, release_path=release)
        except SystemExit as exc:
            check("second-confirmation-is-refused", "already spent" in str(exc))
        else:
            raise SystemExit("self-test failed: a second confirmation read was allowed")
        try:
            run_confirm(plan, registration, ledger=root / "empty.jsonl", release_path=release)
        except SystemExit as exc:
            check("confirmation-without-screening-read-is-refused",
                  "not in the ledger" in str(exc))
        else:
            raise SystemExit("self-test failed: confirmation ran without a screening read")
        try:
            run_confirm(plan, registration, ledger=ledger, release_path=None)
        except SystemExit as exc:
            check("confirmation-without-release-is-refused", "require --release" in str(exc))
        else:
            raise SystemExit("self-test failed: confirmation ran without a release file")

        tampered = read_json(DEFAULT_REGISTRATION)
        tampered["decision"]["min_improvement"] = 0.001
        tampered_path = root / "tampered.json"
        write_json(tampered_path, tampered)
        try:
            load_registration(tampered_path)
        except SystemExit as exc:
            check("tampered-threshold-aborts", "min_improvement" in str(exc))
        else:
            raise SystemExit("self-test failed: a lowered threshold was accepted")

        tampered = read_json(DEFAULT_REGISTRATION)
        tampered["estimator"]["implementation_sha256"] = "0" * 64
        write_json(tampered_path, tampered)
        try:
            load_registration(tampered_path)
        except SystemExit as exc:
            check("tampered-estimator-aborts", "sha256 differs" in str(exc))
        else:
            raise SystemExit("self-test failed: a swapped estimator was accepted")

        plan_body = read_json(positive["plan"])
        plan_body["registration"]["sha256"] = "f" * 64
        write_json(tampered_path, plan_body)
        try:
            load_plan(tampered_path, registration)
        except SystemExit as exc:
            check("stale-plan-pin-is-refused", "re-author the plan" in str(exc))
        else:
            raise SystemExit("self-test failed: a plan pinned to another registration ran")

        plan_body = read_json(positive["plan"])
        plan_body["registration"]["sha256"] = sha
        plan_body["route"] = "R95"
        write_json(tampered_path, plan_body)
        refit_ledger = root / "refit_ledger.jsonl"
        try:
            run_gate(load_plan(tampered_path, registration), registration, mode="screen",
                     ledger=refit_ledger, frozen_weight=0.0)
        except SystemExit as exc:
            check("refit-after-read-is-refused", "refit detected" in str(exc))
            check("refused-refit-writes-no-ledger", ledger_entries(refit_ledger) == [])
        else:
            raise SystemExit("self-test failed: a refit after the screening read was accepted")

        audit = run_audit(registration, ledger=ledger)
        check("audit-reads-no-prediction", audit["reads_any_prediction"] is False)
        check("audit-lists-eight-screen-windows", len(audit["window_roles"]["screen"]) == 8)
        check("audit-gate-script-matches-registration",
              audit["gate_script"]["matches"] is True)
    return {"self_test": "passed", "checks": checks, "count": len(checks)}


# --------------------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("audit", "screen", "confirm", "single"),
                        default="audit")
    parser.add_argument("--registration", default=str(DEFAULT_REGISTRATION))
    parser.add_argument("--plan")
    parser.add_argument("--release", dest="release_path")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--json-out")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        result = run_self_test()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    registration = load_registration(args.registration)
    if args.mode == "audit":
        plan = load_plan(args.plan, registration) if args.plan else None
        result = run_audit(registration, plan, args.ledger)
    elif args.mode == "screen":
        result = run_gate(load_plan(args.plan, registration), registration, mode="screen",
                          ledger=args.ledger)
    elif args.mode == "confirm":
        result = run_confirm(load_plan(args.plan, registration), registration,
                             ledger=args.ledger, release_path=args.release_path)
    else:
        result = run_single(load_plan(args.plan, registration), registration,
                            ledger=args.ledger)

    if args.json_out:
        write_json(Path(args.json_out), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
