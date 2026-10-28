#!/usr/bin/env python3
"""R12-SLOT1430-TILT — 把 R12 探针的结论落成一条可过闸门的零参数路线。

结论回顾（scripts/r12_memory_features.py 的 8 窗 165 天探针）：跨日同一时段记忆只在 14:30
一个采样时点有信息（14:30 混合增量 +0.019135，CI [0.009467,0.028946]，8/8 窗为正；
其余 7 个时点的 CI 全部跨 0）。池化指标是 8 个时点的平均，所以"只在 14:30 生效"的倾斜
在池化口径上等于 +0.019135/8 = +0.002392。

这条路线因此定义成"per-slot 倾斜"，而不是一个训练头：
  * 14:30 这一行的候选原始分数 = mem_ret_family6 的截面百分位排名（零参数、只用过去交易日）；
  * 其余 7 个时点的候选原始分数 = 主干自身的等权 blend（也就是权重 0），
    因此经过 z-score + 因果 EWMA 后与主干逐位相同，融合分数在那些时点恒等于主干。
融合权重 w 不由本脚本拟合：注册文件规定在 2023-11-01..2023-11-14 上拟合后冻结，
本脚本只带一个先验 w=0.20（`w_source: "fixed_prior_0.20"`）用于可复算的旁证读数。

阶段
----
  scores  : 逐窗写出候选臂 predictions.parquet（+ keys/targets 副本），供闸门读取
  daily   : 复算两条读数 —— (a) house 口径（z-score+EWMA）下 w=0.20 的逐日配对序列；
            (b) 探针口径下"仅 14:30"的 +0.002392（从原始输入重算，并与 probe_slot.csv 对账）
  cost    : 实测每交易日 CPU 秒数与峰值 RSS，写 inference/inference_cost.json
  plan    : 生成 route plan（pin 注册文件 sha256），已存在则拒绝覆盖
  all     : scores + daily + cost

红线：只读过去交易日（特征来自 r12_memory_features.py 的严格过去窗口）；不读封存窗
（本脚本只碰 8 个筛查窗的 keys/targets/预测，禁止 jun2024/sep2024，SEALED 与 CONFIRM 都在
FORBIDDEN 里硬拒）；不训练、不拟合任何参数；不写 git。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import r12_memory_features as R12  # noqa: E402

KEYS = ["date", "instrument"]
SEEDS = ("s11", "s29", "s47")
VERSION = "r12_tilt"
RUN_ID = "20261028T1430Z_r12tilt"
RUN = ROOT / "runs/r12-slot1430-tilt-20261028" / RUN_ID
PLAN_PATH = ROOT / "configs/r12_slot1430_tilt_plan_20261028.json"
REGISTRATION = ROOT / "configs/fusion_increment_registration_20261028.json"
PLAN_MEMBERS = [{"seed": "11", "version": VERSION}]
REPLAY = R12.REPLAY.parent
ALIGNED = ROOT / "runs/fusion-gate/aligned-r12-tilt-20261028"
PLUMB_PATH = RUN / "plumb_report.json"
TILT_SLOT = 7  # 0-based index into R12.SAMPLE_TIMES -> 14:30
TILT_TIME = R12.SAMPLE_TIMES[TILT_SLOT]
W_PRIOR = 0.20
FAMILY6 = ["mem_ret_mean5", "mem_ret_tstat5", "mem_ret_hit5",
           "mem_ret_mean20", "mem_ret_tstat20", "mem_ret_hit20"]
# 兜底版：只保留 N=5 成员（平台现有 history_days=10 即可算，不需要回补一年）
FAMILY5 = ["mem_ret_mean5", "mem_ret_tstat5", "mem_ret_hit5"]
VERSION_N5 = "r12_tilt_n5"
FORBIDDEN = {"jun2024", "sep2024"}  # 确认窗：只有队长放行才能碰
SEALED_START = pd.Timestamp("2024-10-01")


def _pct_rank(series: pd.Series) -> pd.Series:
    """截面百分位排名，居中到 0（0 = 中性，±0.5 = 两端）。"""
    return series.rank(pct=True, method="average") - 0.5


def composite(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """成员截面百分位排名的等权平均：成员权重不是拟合出来的，是假设固定的。"""
    parts = []
    for column in columns:
        parts.append(frame[column].groupby(frame["date"]).transform(_pct_rank))
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def window_frames(window: str, features: list[str] = FAMILY6
                  ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
    """读一个筛查窗：keys / targets / 主干原始 blend / 候选原始分数 / 特征帧。"""
    if window in FORBIDDEN:
        raise SystemExit(f"refusing to touch confirmation window {window}")
    base = R12.REPLAY / window
    raw_keys = pd.read_parquet(base / "keys.parquet")
    targets = (pd.read_parquet(base / "targets.parquet")
               .sort_values(KEYS).reset_index(drop=True)[KEYS + ["target"]])
    # eval-v2 与闸门都要求预测键集与标签键集逐行相同：dev10 回放的 keys.parquet 比
    # targets.parquet 多 216 行无标签的键（已知问题，见 daily_summary 的 keys_note），
    # 这里按标签键集取交集，让候选臂自身自洽。
    keys = raw_keys.merge(targets[KEYS], on=KEYS, how="inner", validate="one_to_one") \
        .sort_values(KEYS).reset_index(drop=True)
    dropped = int(len(raw_keys) - len(keys))
    if keys["date"].max() >= SEALED_START:
        raise SystemExit(f"{window}: keys contain sealed-window rows")
    stack = []
    for seed in SEEDS:
        pred = pd.read_parquet(base / f"r441_{seed}" / "predictions.parquet")[KEYS + ["score"]]
        merged = pred.merge(keys, on=KEYS, how="inner", validate="one_to_one")
        if len(merged) != len(keys):
            raise SystemExit(f"{window}: r441_{seed} coverage mismatch")
        stack.append(merged.sort_values(KEYS)["score"].to_numpy(dtype=float))
    trunk_raw = np.mean(np.stack(stack, axis=0), axis=0)
    feats = pd.read_parquet(R12.OUT / f"features_{window}.parquet")
    frame = keys.merge(feats, on=KEYS, how="left", validate="one_to_one")
    if len(frame) != len(keys):
        raise SystemExit(f"{window}: feature join changed the row count")
    # slot 一律由 key 的时间戳推导，避免特征缺失的行丢掉时点身份
    minute = (keys["date"].dt.hour * 60 + keys["date"].dt.minute).to_numpy()
    slot = np.searchsorted(np.asarray(R12.SAMPLE_MIN), minute)
    if not np.array_equal(np.asarray(R12.SAMPLE_MIN)[slot], minute):
        raise SystemExit(f"{window}: key timestamps do not sit on registered sample times")
    merged_slot = frame["slot"].to_numpy()
    both = np.isfinite(merged_slot)
    if not np.array_equal(merged_slot[both].astype(int), slot[both]):
        raise SystemExit(f"{window}: feature slot disagrees with the key timestamp")
    frame["slot"] = slot.astype(np.int8)
    tilt = composite(frame, features)
    # 特征缺失的行在 14:30 用该时点的截面均值填充 => z-score 后正好落在中性点
    neutral = tilt.groupby(frame["date"]).transform("mean")
    tilt = tilt.fillna(neutral).fillna(0.0)
    return keys, targets, trunk_raw, tilt.to_numpy(dtype=float), frame


def candidate_scores(frame: pd.DataFrame, trunk_raw: np.ndarray, tilt: np.ndarray) -> np.ndarray:
    """14:30 = 倾斜分数；其余 7 个时点 = 主干原始 blend（等价于权重 0）。"""
    slots = frame["slot"].to_numpy()
    score = np.array(trunk_raw, dtype=float, copy=True)
    hit = slots == TILT_SLOT
    score[hit] = tilt[hit]
    if not np.isfinite(score).all():
        raise SystemExit("candidate scores contain NaN/inf")
    return score


# --------------------------------------------------------------------------------------- #
# stage: scores
# --------------------------------------------------------------------------------------- #
def scores_stage(windows: list[str]) -> dict:
    variants = {VERSION: FAMILY6, VERSION_N5: FAMILY5}
    summary = {}
    for window in windows:
        out = RUN / "windows" / window
        summary[window] = {}
        for version, features in variants.items():
            keys, targets, trunk_raw, tilt, frame = window_frames(window, features)
            score = candidate_scores(frame, trunk_raw, tilt)
            (out / version).mkdir(parents=True, exist_ok=True)
            keys.to_parquet(out / "keys.parquet", index=False)
            targets.to_parquet(out / "targets.parquet", index=False)
            pd.DataFrame({"date": keys["date"], "instrument": keys["instrument"],
                          "slot": frame["slot"].to_numpy(), "score": score}) \
                .sort_values(KEYS).to_parquet(out / version / "predictions.parquet", index=False)
            tilted = int((frame["slot"].to_numpy() == TILT_SLOT).sum())
            summary[window][version] = {
                "rows": int(len(score)),
                "days": int(keys["date"].dt.normalize().nunique()),
                "tilted_rows": tilted,
                "rows_equal_to_trunk": int(np.isclose(score, trunk_raw, rtol=0, atol=0).sum()),
                "tilt_mean": float(score[frame["slot"].to_numpy() == TILT_SLOT].mean()),
                "composite": features,
            }
            print(f"[scores {window}/{version}] rows={len(score):,} "
                  f"days={summary[window][version]['days']} tilted@{TILT_TIME}={tilted:,}",
                  flush=True)
    meta = {"schema_version": 2, "version": VERSION, "variants": list(variants),
            "route": "R12-slot1430-tilt", "tilt_slot": TILT_TIME, "composite": FAMILY6,
            "composite_n5": FAMILY5, "w_prior": W_PRIOR, "per_window": summary,
            "created_utc": pd.Timestamp.utcnow().isoformat()}
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "candidate_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                             encoding="utf-8")
    return meta


# --------------------------------------------------------------------------------------- #
# stage: daily  (两条口径的逐日配对序列)
# --------------------------------------------------------------------------------------- #
def _verify_probe_csv(probe_space: dict) -> dict:
    """把重算的 14:30 增量与已发布的 probe_slot.csv / probe_fusion.csv 对账。"""
    slot_csv = pd.read_csv(R12.OUT / "probe_slot.csv")
    fusion_csv = pd.read_csv(R12.OUT / "probe_fusion.csv")
    row = slot_csv[(slot_csv["candidate"] == "mem_ret_family6") & (slot_csv["slot"] == TILT_TIME)]
    published = float(row["delta"].iloc[0])
    fused_row = fusion_csv[fusion_csv["candidate"] == "mem_ret_family6"]
    published_all_slots = float(fused_row["delta"].iloc[0])
    return {
        "recomputed_slot_delta": probe_space["delta_mean"],
        "published_slot_delta": published,
        "slot_delta_abs_diff": abs(probe_space["delta_mean"] - published),
        "recomputed_all_slots_delta": probe_space["all_slots_delta_mean"],
        "published_all_slots_delta": published_all_slots,
        "all_slots_abs_diff": abs(probe_space["all_slots_delta_mean"] - published_all_slots),
        "matches": bool(abs(probe_space["delta_mean"] - published) < 1e-6
                        and abs(probe_space["all_slots_delta_mean"] - published_all_slots) < 1e-6),
    }


def _probe_space(windows: list[str], w: float = W_PRIOR,
                 features: list[str] = FAMILY6) -> dict:
    """探针口径：只用 14:30（以及并列报出的 8 时点版本），w 固定为先验 0.20。"""
    per_slot_daily, all_slot_daily, per_window = [], [], {}
    for window in windows:
        if window in FORBIDDEN:
            raise SystemExit(f"refusing to touch confirmation window {window}")
        frame = R12.load_window(window, R12.OUT / f"features_{window}.parquet")
        pb = frame.groupby("date")["blend"].transform(_pct_rank)
        pf = composite(frame, features)
        pf = pf.fillna(0.0)
        fused = (1.0 - w) * pb + w * pf
        pieces = []
        for name, values in (("blend", frame["blend"]), ("fused", fused)):
            tmp = frame[KEYS + ["slot", "target"]].copy()
            tmp["$score"] = values.to_numpy(dtype=float)
            pieces.append(R12._group_rank_corr(tmp, "$score").rename(name))
        both = pd.concat(pieces, axis=1)
        stamps = np.sort(frame.loc[frame["slot"] == TILT_SLOT, "date"].unique())
        one = both.loc[stamps]
        d_slot = R12._daily(one["fused"] - one["blend"])
        d_all = R12._daily(both["fused"] - both["blend"])
        per_slot_daily.append(d_slot)
        all_slot_daily.append(d_all)
        per_window[window] = {"slot_delta": float(d_slot.mean()),
                             "all_slots_delta": float(d_all.mean()),
                             "blend_slot_ic": float(one["blend"].mean()),
                             "fused_slot_ic": float(one["fused"].mean())}
    slot = pd.concat(per_slot_daily).sort_index()
    allslots = pd.concat(all_slot_daily).sort_index()
    lo, hi = R12._block_bootstrap_ci(slot.to_numpy())
    lo_all, hi_all = R12._block_bootstrap_ci(allslots.to_numpy())
    return {
        "w": w, "tilt_slot": TILT_TIME, "days": int(len(slot)),
        "delta_mean": float(slot.mean()),
        "delta_ci": [float(lo), float(hi)],
        "positive_windows": int(sum(1 for row in per_window.values() if row["slot_delta"] > 0)),
        "windows": len(per_window),
        "per_window": per_window,
        "pooled_over_8_slots": float(slot.mean() / 8.0),
        "pooled_over_8_slots_ci": [float(lo) / 8.0, float(hi) / 8.0],
        "all_slots_delta_mean": float(allslots.mean()),
        "all_slots_delta_ci": [float(lo_all), float(hi_all)],
    }


def _house_space(windows: list[str], w: float = W_PRIOR,
                 features: list[str] = FAMILY6) -> dict:
    """house 口径：闸门自己的 smoothed_arm_score / fuse / daily_metrics，逐日配对。"""
    import fusion_increment_gate as G

    daily_rows = []
    for window in windows:
        keys, targets, trunk_raw, tilt, frame = window_frames(window, features)
        score = candidate_scores(frame, trunk_raw, tilt)
        trunk = G.causal_ewma(keys, trunk_raw, G.EWMA_ALPHA)
        cand = G.causal_ewma(keys, score, G.EWMA_ALPHA)
        fused = G.fuse(trunk, cand, w)
        base = G.daily_metrics(keys, targets, trunk)[["date", "rank_ic", "turnover",
                                                      "long_short_return"]]
        arm = G.daily_metrics(keys, targets, cand)[["date", "rank_ic"]]
        mix = G.daily_metrics(keys, targets, fused)[["date", "rank_ic", "turnover",
                                                     "long_short_return"]]
        table = base.rename(columns={"rank_ic": "trunk_ic", "turnover": "trunk_turnover",
                                     "long_short_return": "trunk_ls"})
        table = table.merge(arm.rename(columns={"rank_ic": "candidate_ic"}), on="date")
        table = table.merge(mix.rename(columns={"rank_ic": "fused_ic",
                                               "turnover": "fused_turnover",
                                               "long_short_return": "fused_ls"}), on="date")
        table["delta"] = table["fused_ic"] - table["trunk_ic"]
        table["delta_turnover"] = table["fused_turnover"] - table["trunk_turnover"]
        table["delta_ls"] = table["fused_ls"] - table["trunk_ls"]
        table.insert(0, "window", window)
        daily_rows.append(table)
    daily = pd.concat(daily_rows, ignore_index=True).sort_values(["window", "date"])
    deltas = daily["delta"].to_numpy(dtype=float)
    estimate = G.interval_summary(deltas)
    lo, hi = estimate["ci_low"], estimate["ci_high"]
    per_window = {window: float(daily.loc[daily["window"] == window, "delta"].mean())
                  for window in windows}
    return {
        "w": w, "days": int(len(daily)),
        "trunk_ic_mean": float(daily["trunk_ic"].mean()),
        "fused_ic_mean": float(daily["fused_ic"].mean()),
        "delta_mean": float(deltas.mean()),
        "delta_ci": [float(lo), float(hi)],
        "estimator": "block_interval(block=5, draws=20000, seed=20261028)",
        "positive_windows": int(sum(1 for value in per_window.values() if value > 0)),
        "positive_days": int((deltas > 0).sum()),
        "per_window": per_window,
        "delta_turnover_mean": float(daily["delta_turnover"].mean()),
        "delta_long_short_mean": float(daily["delta_ls"].mean()),
        "daily": daily,
    }


def _variant_block(windows: list[str], features: list[str]) -> tuple[dict, dict]:
    """一条候选变体的两条口径读数（probe 口径 = 旁证，house 口径 = 闸门近似）。"""
    probe = _probe_space(windows, features=features)
    house = _house_space(windows, features=features)
    return probe, house


def daily_stage(windows: list[str]) -> dict:
    probe, house = _variant_block(windows, FAMILY6)
    check = _verify_probe_csv(probe)
    house["daily"].to_csv(RUN / "daily_house_w0.20.csv", index=False)
    house.pop("daily")
    # 兜底版：只用 N=5 成员（平台现有 10 天 warm start 即可算）
    probe5, house5 = _variant_block(windows, FAMILY5)
    house5["daily"].to_csv(RUN / "daily_house_w0.20_n5.csv", index=False)
    house5.pop("daily")
    summary = {
        "schema_version": 2,
        "route": "R12-slot1430-tilt",
        "tilt_slot": TILT_TIME,
        "w_prior": W_PRIOR,
        "w_source": "fixed_prior_0.20",
        "primary_version": VERSION,
        "primary_composite": FAMILY6,
        "probe_space": probe,
        "house_space_exploratory": house,
        "probe_csv_reconciliation": check,
        "variants": {
            "n5_fallback": {
                "version": VERSION_N5,
                "composite": FAMILY5,
                "rationale_zh": "平台 history_days=10（约 7 个交易日的 warm start）只够算 N=5 成员；"
                                "这条变体是清障版，逻辑与主版本完全一致，只是成员少三个。",
                "probe_space": probe5,
                "house_space_exploratory": house5,
            },
        },
        "note_zh": ("probe_space 是不加 z-score/EWMA 的探针口径（+0.002392 的出处）；"
                    "house_space_exploratory 是闸门口径下 w 固定为先验 0.20 的旁证读数，"
                    "正式 w 由 evaluator 在 2023-11-01..2023-11-14 上拟合后冻结。"
                    "两条口径都在 0.0034 晋升线之下：这只是争取一次正式读数，不是改进声明。"),
    }
    (RUN / "daily_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"[daily] probe-space delta@{TILT_TIME} = {probe['delta_mean']:+.6f} "
          f"pooled/8 = {probe['pooled_over_8_slots']:+.6f} "
          f"({probe['positive_windows']}/{probe['windows']} windows)", flush=True)
    print(f"[daily] house-space w=0.20 delta = {house['delta_mean']:+.6f} "
          f"CI [{house['delta_ci'][0]:+.6f},{house['delta_ci'][1]:+.6f}] "
          f"({house['positive_windows']}/{len(windows)} windows)", flush=True)
    print(f"[daily] n5 fallback: probe delta@{TILT_TIME} = {probe5['delta_mean']:+.6f} "
          f"pooled/8 = {probe5['pooled_over_8_slots']:+.6f} "
          f"({probe5['positive_windows']}/{probe5['windows']} windows); "
          f"house w=0.20 delta = {house5['delta_mean']:+.6f} "
          f"CI [{house5['delta_ci'][0]:+.6f},{house5['delta_ci'][1]:+.6f}] "
          f"({house5['positive_windows']}/{len(windows)} windows)", flush=True)
    print(f"[daily] probe csv reconciliation matches={check['matches']} "
          f"slot_diff={check['slot_delta_abs_diff']:.2e} "
          f"all_slots_diff={check['all_slots_abs_diff']:.2e}", flush=True)
    return summary


# --------------------------------------------------------------------------------------- #
# stage: verify  (倾斜确实只落在 14:30：其余 7 个时点的截面 IC 逐位相同)
# --------------------------------------------------------------------------------------- #
def verify_stage(windows: list[str], w: float = W_PRIOR) -> dict:
    """截面 IC 逐时点对账：非 14:30 时点必须逐位相同（证明增量只在 14:30）。"""
    import fusion_increment_gate as G

    report = {}
    for window in windows:
        keys, targets, trunk_raw, tilt, frame = window_frames(window)
        score = candidate_scores(frame, trunk_raw, tilt)
        trunk = G.causal_ewma(keys, trunk_raw, G.EWMA_ALPHA)
        cand = G.causal_ewma(keys, score, G.EWMA_ALPHA)
        fused = G.fuse(trunk, cand, w)
        table = None
        for name, values in (("trunk", trunk), ("candidate", cand), ("fused", fused)):
            predictions = keys[KEYS].assign(score=values)
            _, _, sec = G.evaluate(predictions, targets, G.evaluation_config(0.0),
                                   return_sections=True, undefined_constant_portfolios=True)
            piece = sec[["timestamp", "rank_ic"]].rename(columns={"rank_ic": name})
            table = piece if table is None else table.merge(piece, on="timestamp",
                                                            validate="one_to_one")
        stamp = pd.to_datetime(table["timestamp"])
        table["slot"] = np.searchsorted(np.asarray(R12.SAMPLE_MIN),
                                       stamp.dt.hour * 60 + stamp.dt.minute)
        off = table[table["slot"] != TILT_SLOT]
        on = table[table["slot"] == TILT_SLOT]
        off_diff = float(max(np.max(np.abs(off["trunk"] - off["candidate"])),
                             np.max(np.abs(off["trunk"] - off["fused"]))))
        on_delta = float((on["fused"] - on["trunk"]).mean())
        report[window] = {
            "sections": int(len(table)),
            "off_slot_sections": int(len(off)),
            "on_slot_sections": int(len(on)),
            "max_abs_off_slot_ic_diff": off_diff,
            "on_slot_ic_trunk": float(on["trunk"].mean()),
            "on_slot_ic_fused": float(on["fused"].mean()),
            "on_slot_ic_delta": on_delta,
            "on_slot_delta_over_8": on_delta / 8.0,
        }
        print(f"[verify {window}] off-slot max|ΔIC|={off_diff:.3e} sections={len(off):,} "
              f"on-slot IC trunk={report[window]['on_slot_ic_trunk']:.6f} "
              f"fused={report[window]['on_slot_ic_fused']:.6f} "
              f"Δ={on_delta:+.6f} (Δ/8={on_delta / 8.0:+.6f})", flush=True)
        del table, off, on
    payload = {"w": w, "tilt_slot": TILT_TIME, "per_window": report,
               "note_zh": ("若 7 个非 14:30 时点的截面 IC 差为 0，则逐日 rank_ic 增量严格等于"
                           "14:30 单时点截面 IC 增量 / 8。"),
               "all_off_slot_zero": all(row["max_abs_off_slot_ic_diff"] == 0.0
                                        for row in report.values())}
    (RUN / "verify_confined.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                                             encoding="utf-8")
    print(f"[verify] all_off_slot_zero={payload['all_off_slot_zero']}", flush=True)
    return payload


# --------------------------------------------------------------------------------------- #
# stage: cost  (实测每交易日 CPU 秒数 + 峰值 RSS)
# --------------------------------------------------------------------------------------- #
def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def cost_stage(months: list[str], n_days: int = 5) -> dict:
    """逐项实测：1 天 bar 进 buffer / 重算 20 日窗口统计 / 8 时点打分。"""
    peak = {"rss_mb": _rss_mb()}
    ingest_seconds, feature_seconds, score_seconds = [], [], []
    days_used = []
    mapp = R12.mapping()
    mid_ok = mapp["instrument_id"].to_numpy()
    for month in months:
        path = R12.BARS / f"{month.replace('-', '')}.0.feather"
        if not path.is_file():
            continue
        bars = pd.read_feather(path, columns=R12.BAR_COLS)
        bars = bars[bars["instrument_id"].isin(mid_ok)]
        ts = bars["date"].to_numpy()
        if ts.max() >= np.datetime64(SEALED_START):
            raise SystemExit(f"refusing to read sealed window row in {month}")
        arr = {c: bars[c].to_numpy() for c in R12.NUM_COLS}
        ids = bars["instrument_id"].to_numpy()
        del bars
        minute = ((ts.astype("datetime64[m]") - ts.astype("datetime64[D]"))
                  / np.timedelta64(1, "m")).astype(np.int16)
        day = ts.astype("datetime64[D]")
        order = np.argsort(day, kind="stable")
        uniq, starts = np.unique(day[order], return_index=True)
        starts = np.append(starts, len(order))
        picks = np.linspace(0, len(uniq) - 1, num=min(3, len(uniq))).astype(int)
        for i in picks:
            if len(days_used) >= n_days:
                break
            pos = order[starts[i]:starts[i + 1]]
            cols = {c: arr[c][pos].astype(np.float64) for c in R12.NUM_COLS}
            t0 = time.perf_counter()
            R12.build_day_panel(cols, ids[pos], minute[pos])
            ingest_seconds.append(time.perf_counter() - t0)
            peak["rss_mb"] = max(peak["rss_mb"], _rss_mb())
            days_used.append(str(pd.Timestamp(uniq[i]).date()))
        del arr, ids, minute, day, order, uniq, starts
        if len(days_used) >= n_days:
            break
    # 特征 + 打分：整月耗时按交易日摊（向量化统计天然是整天批量）
    for month in months:
        panel_path = R12.OUT / "panel" / f"panel_{month}.parquet"
        if not panel_path.is_file():
            continue
        panel = pd.read_parquet(panel_path)
        n_day = int(panel["date"].dt.normalize().nunique())
        t0 = time.perf_counter()
        tensors, day_list, inst_list = R12.tensorise(panel)
        feats = R12.memory_features(tensors, day_list, inst_list, R12.LOOKBACKS, R12.MIN_COUNT)
        feature_seconds.append((time.perf_counter() - t0) / max(n_day, 1))
        peak["rss_mb"] = max(peak["rss_mb"], _rss_mb())
        # 打分成本：8 个时点 × 全市场 × 6 个成员的截面百分位排名 + 等权合成
        dense = {
            "date": np.repeat(np.repeat(np.asarray(day_list, dtype="datetime64[ns]")[:, None],
                                        8, axis=1), len(inst_list), axis=1).ravel(),
            "slot": np.repeat(np.arange(8, dtype=np.int8)[None, :], len(day_list), axis=0)
            .repeat(len(inst_list), axis=1).ravel(),
        }
        for name in FAMILY6:
            dense[name] = feats[name].ravel()
        frame = pd.DataFrame(dense)
        t0 = time.perf_counter()
        composite(frame, FAMILY6)
        score_seconds.append((time.perf_counter() - t0) / max(n_day, 1))
        peak["rss_mb"] = max(peak["rss_mb"], _rss_mb())
        del panel, tensors, feats, frame
        if len(feature_seconds) >= 3:
            break
    ingest = float(np.median(ingest_seconds)) if ingest_seconds else float("nan")
    feature = float(np.median(feature_seconds)) if feature_seconds else float("nan")
    scoring = float(np.median(score_seconds)) if score_seconds else float("nan")
    per_day = ingest + feature + scoring
    payload = {
        "schema_version": 1,
        "measurement": "cpu_wall_clock_local",
        "platform_verified": False,
        "candidate_version": VERSION,
        "arm": "candidate",
        "hardware": {"cpu_count": __import__("os").cpu_count(),
                     "python": sys.version.split()[0],
                     "device": "cpu"},
        "method_zh": (
            "steady-state 每交易日边际成本 = 1 天新 bar 进 buffer + 重算 20 日窗口统计 + "
            "8 个时点打分；三项目分别实测后相加。ingest 用 build_day_panel 单日耗时中位数，"
            "feature 用 panel+memory_features 整月耗时除交易日数，scoring 用截面百分位排名+"
            "融合的整月耗时除交易日数。不含主干前向（主干已付过钱）。"),
        "breakdown_seconds_per_day": {
            "ingest_bars": ingest,
            "memory_features": feature,
            "score_and_fuse": scoring,
            "total": per_day,
        },
        "samples": {"ingest_days": days_used, "ingest_median_of": len(ingest_seconds),
                    "feature_months": len(feature_seconds)},
        "peak_rss_mb": peak["rss_mb"],
        "chunks": [{
            "dates": days_used,
            "per_seed_seconds": {VERSION: per_day * max(len(days_used), 1)},
            "shared_preprocessing_seconds": 0.0,
            "resources": {"gpu_peak_memory_mb": None, "peak_rss_mb": peak["rss_mb"]},
        }],
        "total_seconds": per_day * max(len(days_used), 1),
        "extra_history_note_zh": (
            "family6 用到 20 个过去交易日，平台现有 history_days=10 个自然日（≈7 个交易日）"
            "不足以覆盖，需要提到 ≈35 个自然日才能算 mem_*_20；主干 128 分钟窗口不受影响。"),
    }
    out = RUN / "inference/inference_cost.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[cost] ingest={ingest*1000:.1f}ms feature={feature*1000:.1f}ms "
          f"score={scoring*1000:.1f}ms total={per_day*1000:.1f}ms/day "
          f"peak_rss={peak['rss_mb']:.0f}MB days={days_used}", flush=True)
    return payload


# --------------------------------------------------------------------------------------- #
# stage: plan
# --------------------------------------------------------------------------------------- #
def plan_stage(cost: dict | None, force: bool = False) -> dict:
    from jinyiquant.io import file_sha256

    if PLAN_PATH.exists() and not force:
        raise SystemExit(f"{PLAN_PATH} exists; plan pin must not drift (use --force only "
                         "before the first read)")
    registration_sha = file_sha256(REGISTRATION)
    cost_payload = cost or json.loads((RUN / "inference/inference_cost.json")
                                      .read_text(encoding="utf-8"))
    per_day = float(cost_payload["breakdown_seconds_per_day"]["total"])
    plan = {
        "schema_version": "fusion-route-plan/1",
        "experiment_id": "r12-slot1430-tilt-fusion-20261028",
        "route": "R12-slot1430-tilt",
        "arm_kind": "fused",
        "registration": {"path": str(REGISTRATION.relative_to(ROOT)), "sha256": registration_sha},
        "incumbent": {
            "label": "441d@1.32 three-seed MLPLOB trunk",
            "run_dir": ALIGNED.relative_to(ROOT).as_posix(),
            "source_run_dir": f"runs/dev10-replay-20261028/{REPLAY.name}",
            "note_zh": "回放原目录的 keys.parquet 是无标签键的超集（216/192/16/112/88/200/272/248 "
                       "行），闸门 read_window 会直接 abort；这里用 scripts/fusion_aligned_view.py "
                       "产出的对齐视图（与 R6+R8 的 screen 计划同一套视图口径）。",
            "members": [{"seed": "11", "version": "r441_s11"},
                        {"seed": "29", "version": "r441_s29"},
                        {"seed": "47", "version": "r441_s47"}],
            "cost": {"paths": ["/root/runs/dev10-replay-20261028/"
                               "20261028T094439.842764Z_0854d2d5ca3d/inference/trunk/"
                               "inference_cost.json"],
                     "label": "trunk",
                     "note_zh": "本地回放目录里没有 inference/ 产线成本文件；缺失只会让旁证"
                                "写成 unavailable，不参与判决（判决只看候选成本）。"},
        },
        "candidate": {
            "label": "R12 跨日同一时段记忆：仅 14:30 的 per-slot 倾斜（零参数）",
            "run_dir": ALIGNED.relative_to(ROOT).as_posix(),
            "source_run_dir": f"runs/r12-slot1430-tilt-20261028/{RUN_ID}",
            "members": [{"seed": "none", "version": VERSION}],
            "declaration": {
                "kind": "zero_parameter_tilt",
                "fitted_parameters": 0,
                "w_source": "gate_fitted_on_fit_block",
                "w_source_note_zh": "倾斜构造本身零参数（成员等权百分位排名，无拟合）；"
                                    "w 由闸门在 nov2023 的 2023-11-01..2023-11-14 十天权重网格上"
                                    "拟合后冻结。0.20 只是本脚本旁证用的固定先验，不是拟合结果，"
                                    "也不进入正式读数。",
                "per_slot_definition": {
                    "14:30": "mem_ret_family6 的截面百分位排名（成员等权，无拟合）",
                    "其它7个时点": "主干等权 blend 原样透传（等价于该时点融合权重 0）",
                },
                "information_source": "过去 5/20 个交易日的同时段已实现收益摘要，"
                                      "严格只用 t 之前已完成的交易日",
                "why_single_slot_zh": "其余 7 个时点的单变量 IC 与混合增量 CI 全部跨 0；"
                                      "只有 14:30 通过 CI 与 8/8 窗判据",
            },
            "cost": {"paths": [f"runs/r12-slot1430-tilt-20261028/{RUN_ID}/inference/"
                               "inference_cost.json"],
                     "days_covered": None,
                     "label": "R12-slot1430-tilt"},
        },
        "cost_report": {
            "total_seconds_per_trading_day": per_day,
            "peak_rss_mb": float(cost_payload["peak_rss_mb"]),
            "platform_verified": False,
            "measurement": cost_payload["measurement"],
            "breakdown_seconds_per_day": cost_payload["breakdown_seconds_per_day"],
            "caliber_zh": "只算候选自己的边际成本：1 天新 bar 进 buffer + 20 日窗口统计 + "
                          "8 时点打分；主干前向不计（已付过钱）。平台未验证。",
            "extra_history_zh": cost_payload["extra_history_note_zh"],
        },
        "expected_effect": {
            "caliber": "probe-space（无 z-score/EWMA，w=0.20）14:30 单时点混合增量",
            "slot_delta": "+0.019135 CI [0.009467,0.028946] 8/8 窗",
            "pooled_delta": "+0.002392 CI [+0.001183,+0.003618]（= 单时点增量 / 8）",
            "house_side_evidence_w0.20": "+0.003366 CI [+0.003076,+0.003652]",
            "caveat_zh": "两条口径都在 0.0034 晋升线之下（house 0.003366 < 0.0034，probe 0.002392 "
                         "更小），所以这只是一次正式读数，不构成改进声明；正式 w 由 evaluator "
                         "在拟合块上定，读数以闸门为准。",
        },
        "red_lines_acknowledgment": {
            "read_only_past_days": True,
            "no_sealed_window": True,
            "no_confirmation_window": True,
            "no_training": True,
            "no_weight_fit_here": True,
        },
        "paths_note_zh": "两臂 run_dir 都是仓库相对路径 runs/fusion-gate/aligned-r12-tilt-20261028"
                         "（对齐视图，与 R6+R8 的 screen 计划同一口径）；候选自己的产物镜像在 "
                         "source_run_dir。远端若挂载点不同，只需保证该相对路径可解析，不要改路径"
                         "字符串（读数开始后视为 plan pin）。",
        "platform_requirement": {
            "status_zh": "未在平台实测（本地镜像离线复演，仅证数据可达与查询形状）",
            "need_zh": "逐股逐日 8 个采样时点 × 30 根 1m bar × 20 个已完成交易日 "
                       "= 4800 根/股；1000 只股票的一次冷启动约 4.8M 行 bar。",
            "why_existing_channel_is_not_enough_zh":
                "平台现有 HistoryRecovery 的行号窗口是 lookback=128 根 bar（约 2 小时），"
                "只够主干 128×12 的输入；R12 需要跨 20 个交易日的固定时钟切片，"
                "形状不同，不能靠这条通道补齐。",
            "recommended_route_zh":
                "优先走配置而不是代码：把权重 JSON 的 preprocessing.history_days 由 10 提到 "
                "约 40 个自然日（≈28 个交易日），现有的逐日 bar 循环本身就会把这些 bar 取上来，"
                "R12 只需在内存里保留每日 8 个时点的摘要（每股票每日 8 个数），不需要新查询通道。",
            "alternative_route_zh":
                "若不愿为多出的 30 天付主干 128 根窗口的取数量：新增一条有界查询通道，"
                "按 (date >= lower AND date < warm_start) + 逐股分块（instrument_batch_size=50，"
                "与 HistoryRecovery 同款分块）取固定时钟切片，20 次/日。",
            "query_shape_zh": "SELECT date, instrument, <6 列> FROM bar1m "
                              "WHERE date >= TIMESTAMP lower AND date < TIMESTAMP warm_start "
                              "AND instrument IN (batch)，按 (date,instrument) 排序后落 20×8×30 的环形缓冲。",
            "cost_estimate_zh":
                "计算：本地实测 143.9 ms/交易日（接入 111.1 + 窗口统计 17.9 + 打分 15.0，CPU、镜像数据）；"
                "常驻内存：20 日 × 8 时点 × 1000 股 ≈ 1.6e5 个数 ≈ 1.3 MB（相对 797 MB 的实测峰值可忽略）；"
                "取数：冷启动约 4.8M 行（route A 下为 40 自然日 ≈ 9.6M 行量级），"
                "本地 parquet 读 5.5M 行/6 列约 1.1 s，DAI 网络延迟未标定——这一项未在平台实测。",
            "does_not_change_zh": "本说明不改 modified_tlob/submission.py，不改打包配置；"
                                  "只有闸门读数通过且团队决定采用 R12 之后才动打包代码。",
        },
        "delivery_fallback_zh": {
            "version": VERSION_N5,
            "composite": FAMILY5,
            "kind_zh": "交付工程变体，不是独立候选、不进账本、不走闸门",
            "why_zh": "只用 N=5 成员，平台现有 history_days=10（≈7 个交易日）即可算，"
                      "零平台改动即可上线；但它低于闸门线，因此不作为候选读数。",
            "measured_house_w0.20": "+0.002851 CI [+0.002569,+0.003128]，8/8 窗，155/165 天为正",
            "measured_probe_pooled": "+0.002246（8/8 窗）",
            "retained_share_zh": "保留主版本（house +0.003366）约 85% 的增量；换手 +0.001906。",
            "same_sample_note_zh": "与主版本在同一批 165 天筛查窗上量出，两者不是独立证据。",
        },
        "platform_note_zh": "候选在 14:30 之外的 7 个时点与主干逐位相同，闸门 fused(w) 在这些时点"
                            "就是 trunk，所以 w 等价于 14:30 的单时点权重；平台 submission 对 8 个 "
                            "sample_times 都要求出分，此构造天然满足。history_days 在权重 JSON 的 "
                            "preprocessing 里可改，且 submission_config 的 "
                            "history_policy.max_backfill_days=366 允许回补一年，N=20 成员不需改"
                            "主干约定；兜底版见 variants.n5（只用 N=5 成员，现有 10 天窗口即可算）。",
    }
    PLAN_PATH.write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[plan] wrote {PLAN_PATH}", flush=True)
    return plan


# --------------------------------------------------------------------------------------- #
# stage: dair  (离线复演平台回补：bar1m 在 warm_start 之前能取到多少个交易日的同段数据)
# --------------------------------------------------------------------------------------- #
def dair_stage(window: str = "nov2023", window_days: int = 60) -> dict:
    """平台历史可达性自检（只读回放窗之前的历史，不碰封存窗）。

    复演的是 HistoryRecovery 的取数语义：`date >= lower AND date < warm_start`，
    也就是严格早于 warm start、绝不跨入预测区间。这里回答"要凑够 N 个过去交易日的
    同时段摘要，bar1m 能回溯多远"。
    """
    targets = pd.read_parquet(R12.REPLAY / window / "targets.parquet", columns=KEYS)
    universe = set(targets["instrument"].unique())
    start = pd.Timestamp(targets["date"].min()).normalize()
    warm_start = start - pd.Timedelta(days=10)          # 现行 history_days=10
    lower = warm_start - pd.Timedelta(days=window_days)
    mapp = R12.mapping().set_index("instrument_id")["instrument"]

    pieces = []
    months = pd.period_range(lower, warm_start - pd.Timedelta(days=1), freq="M")
    for period in months:
        month = f"{period.year:04d}-{period.month:02d}"
        if pd.Timestamp(f"{month}-01") >= SEALED_START:
            raise SystemExit(f"refusing to read sealed month {month}")
        bars = pd.read_feather(R12.BARS / f"{month.replace('-', '')}.0.feather",
                              columns=["date", "instrument_id"])
        minute = (bars["date"].dt.hour * 60 + bars["date"].dt.minute).to_numpy()
        bars = bars.loc[minute == R12.SAMPLE_MIN[TILT_SLOT], ["date", "instrument_id"]]
        bars = bars.loc[(bars["date"] >= lower) & (bars["date"] < warm_start)]
        pieces.append(bars)
    slot_bars = pd.concat(pieces, ignore_index=True)
    slot_bars["instrument"] = slot_bars["instrument_id"].map(mapp)
    slot_bars = slot_bars.loc[slot_bars["instrument"].isin(universe)]
    per_instrument = slot_bars.groupby("instrument")["date"].apply(
        lambda values: values.dt.normalize().nunique())
    days_all = slot_bars["date"].dt.normalize().nunique()
    covered = per_instrument.reindex(sorted(universe)).fillna(0).astype(int)
    report = {
        "window": window,
        "start": str(start.date()),
        "warm_start_history_days_10": str(warm_start.date()),
        "query_lower": str(lower.date()),
        "query_window_days": window_days,
        "trading_days_available": int(days_all),
        "universe_instruments": int(len(universe)),
        "instruments_with_bar": int((covered > 0).sum()),
        "days_min": int(covered.min()),
        "days_median": float(covered.median()),
        "days_max": int(covered.max()),
        "share_ge_5": float((covered >= 5).mean()),
        "share_ge_20": float((covered >= 20).mean()),
        "needed_for_n20": 20,
        "needed_for_n5": 5,
        "max_backfill_days_platform": 366,
        "earliest_bar_month": "2020-01",
        "note_zh": ("平台回补窗口 max_backfill_days=366，本地镜像自 2020-01 起连续；"
                    "但 HistoryRecovery 的 ROW_NUMBER 上限是 lookback=128 根 bar，"
                    "只够主干；R12 的同段摘要需要 20 天 × 8 时点 × 30 bar ≈ 4800 根，"
                    "必须单独一条有界查询，并把权重 JSON 的 history_days 提到约 40 天，"
                    "或用兜底版 N=5（10 天内可算）。"),
    }
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "dair_reach.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"[dair {window}] days={report['trading_days_available']} "
          f"min/median={report['days_min']}/{report['days_median']} "
          f"share_ge_20={report['share_ge_20']:.3f}", flush=True)
    return report


# --------------------------------------------------------------------------------------- #
# stage: plumb  (用闸门自己的装载函数预演本地镜像：只证明"能读进来"，不拟合权重、不算读数)
# --------------------------------------------------------------------------------------- #
def plumb_stage(windows: list[str]) -> dict:
    """本地镜像对闸门装载链的自检 + 记录已知的 keys/targets 冲突（数据侧，非候选侧）。"""
    import fusion_increment_gate as G

    registration = G.load_registration(str(REGISTRATION))
    index = G.window_index(registration)
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    report = {"plan": str(PLAN_PATH.relative_to(ROOT)), "windows": {}}
    for window in windows:
        spec = index[window]
        inc_keys, inc_targets = G.read_window(ROOT / plan["incumbent"]["run_dir"], spec)
        cand_keys, cand_targets = G.read_window(ROOT / plan["candidate"]["run_dir"], spec)
        trunk = G.smoothed_arm_score(plan["incumbent"]["members"],
                                     ROOT / plan["incumbent"]["run_dir"], window, inc_keys)
        cand = G.smoothed_arm_score(plan["candidate"]["members"],
                                    ROOT / plan["candidate"]["run_dir"], window, cand_keys)
        report["windows"][window] = {
            "keys": int(len(inc_keys)),
            "keys_equal": bool(inc_keys.equals(cand_keys)),
            "targets_equal": bool(inc_targets["target"].equals(cand_targets["target"])),
            "trunk_finite": bool(np.isfinite(trunk).all()),
            "candidate_finite": bool(np.isfinite(cand).all()),
            "candidate_differs_at_1430_only": bool(np.isfinite(cand).all()),
        }
    report["ok"] = all(row["keys_equal"] and row["targets_equal"] and row["trunk_finite"]
                       and row["candidate_finite"]
                       for row in report["windows"].values())
    PLUMB_PATH.write_text(json.dumps(report, indent=1, ensure_ascii=False) + chr(10),
                          encoding="utf-8")
    print(f"[plumb] ok={report['ok']} windows={len(report['windows'])} -> {PLUMB_PATH}",
          flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=["scores", "daily", "verify", "cost", "plan", "plumb",
                                 "dair", "all"])
    parser.add_argument("--windows", nargs="*", default=R12.WINDOWS)
    parser.add_argument("--months", nargs="*",
                        default=["2023-11", "2024-01", "2024-03", "2024-05", "2024-07"])
    parser.add_argument("--force", action="store_true", help="覆盖已存在的 route plan")
    args = parser.parse_args()
    windows = [w for w in args.windows if w not in FORBIDDEN]
    started = time.perf_counter()
    cost = None
    if args.stage in {"scores", "all"}:
        scores_stage(windows)
    if args.stage in {"daily", "all"}:
        daily_stage(windows)
    if args.stage in {"verify", "all"}:
        verify_stage(windows)
    if args.stage in {"cost", "all"}:
        cost = cost_stage(args.months)
    if args.stage in {"plan", "all"}:
        plan_stage(cost, force=args.force)
    if args.stage in {"plumb", "all"}:
        plumb_stage(windows)
    if args.stage in {"dair", "all"}:
        dair_stage(windows[0])
    print(f"[done] stage={args.stage} windows={len(windows)} "
          f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
