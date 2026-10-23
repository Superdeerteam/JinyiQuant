"""Record the archived three-seed/two-window V0 vs V1 evidence.

The V1 version is the full-timestamp label cross-sectional z-score recipe.  Its
seed-11 research-A fit is bit-identical to the archived ``cand_csz`` arm, so the
already-frozen three-seed/two-window CSZ round is the multi-seed extension of V1
rather than a new experiment.  This script verifies that identity, re-reads the
archived per-seed metrics and daily paired deltas, and writes one machine-readable
record plus a short text report.  It never trains or opens new data ranges.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_model_versions import assert_only_label_transform_changes, numeric_tensors  # noqa: E402

from jinyiquant.io import file_sha256, read_json, utc_now, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSZ_RECORD = ROOT / "modified_tlob/label_cszscore_20261018.json"
VERSION_RECORD = ROOT / "modified_tlob/model_versions_20261023.json"
CSZ_RUN = ROOT / "runs/label-cszscore-20261018/20261018T104425.466761Z_1b8039d5f4e1"
RAW_RUN = ROOT / "runs/fixed-seed-ensemble-20261017/20261017T064944.073834Z_193b03ead7ff"
CSZ = "timestamp_zscore_train_rms_matched"
SEEDS = (11, 29, 47)
WINDOWS = ("research_a", "research_b")
SHARED_CONFIG = ("data", "splits", "acceptance", "evaluation", "logging", "evaluation_split")
CELL_METRICS = ("rank_ic_mean", "long_short_return_mean", "turnover_mean",
                "min_daily_long_short_return", "rank_ic_ir", "long_short_sharpe", "day_count",
                "sample_count")
DELTA_KEYS = {"rank_ic_mean": "rank_ic", "long_short_return_mean": "long_short_return",
              "turnover_mean": "turnover"}
LABELS = {
    "research_a": ROOT / "数据/cpt_jyc_2026_vwap/cpt_jyc_2026_vwap_2023-11.parquet",
    "research_b": ROOT / "数据/cpt_jyc_2026_vwap/cpt_jyc_2026_vwap_2024-03.parquet",
}


def close(left, right, atol=1e-12):
    return abs(float(left) - float(right)) <= atol * max(1.0, abs(float(left)), abs(float(right)))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cell_configs(window, seed):
    raw = read_json(RAW_RUN / f"fits/{window}_s{seed}/config.json")
    csz = read_json(CSZ_RUN / f"fits/{window}_s{seed}/config.json")
    require(raw["seed"] == seed and csz["seed"] == seed, f"{window}/{seed}: seed mismatch")
    for key in SHARED_CONFIG:
        require(raw[key] == csz[key], f"{window}/{seed}: shared config {key} differs")
    raw_model = {key: value for key, value in raw["model"].items() if key != "target_centering"}
    csz_model = {key: value for key, value in csz["model"].items() if key != "target_centering"}
    require(raw_model == csz_model, f"{window}/{seed}: model differs outside target centering")
    require(raw["model"].get("target_centering", "none") == "none",
            f"{window}/{seed}: raw reference does not use the raw label")
    require(csz["model"].get("target_centering") == CSZ,
            f"{window}/{seed}: CSZ arm does not use the V1 target")
    raw_train = read_json(RAW_RUN / f"fits/{window}_s{seed}/training.json")
    csz_train = read_json(CSZ_RUN / f"fits/{window}_s{seed}/training.json")
    for key in ("steps", "samples", "samples_seen", "epochs"):
        require(raw_train[key] == csz_train[key],
                f"{window}/{seed}: training exposure {key} differs")
    require(raw_train["steps"] == 1500 and raw_train["samples_seen"] == 48000,
            f"{window}/{seed}: unexpected training exposure")
    for run in (RAW_RUN, CSZ_RUN):
        manifest = read_json(run / f"fits/{window}_s{seed}/manifest.json")
        require(manifest["status"] == "completed", f"{window}/{seed}: fit not completed")
    return raw, csz, raw_train


def window_cells(record, window):
    ref = record["results"]["windows"][window]["reference_comparison"]
    cells = []
    for seed in SEEDS:
        v0 = ref["per_seed_metrics"][str(seed)]["ref10"]
        v1 = ref["per_seed_metrics"][str(seed)]["cand_csz"]
        for metric in CELL_METRICS:
            require(metric in v0 and metric in v1, f"{window}/{seed}: missing {metric}")
        delta = {key: float(v1[metric] - v0[metric])
                 for metric, key in DELTA_KEYS.items()}
        for key, value in delta.items():
            require(close(value, ref["per_seed_mean_delta"][str(seed)][key]),
                    f"{window}/{seed}: {key} delta does not match the frozen record")
        cells.append({"window": window, "seed": seed,
                      "v0": {key: float(v0[key]) for key in CELL_METRICS},
                      "v1": {key: float(v1[key]) for key in CELL_METRICS},
                      "delta": delta})
    for metric in ("rank_ic", "long_short_return", "turnover"):
        for arm, source in (("v0", "ref10"), ("v1", "cand_csz")):
            computed = sum(cell[arm]["rank_ic_mean" if metric == "rank_ic" else
                                  ("long_short_return_mean" if metric == "long_short_return"
                                   else "turnover_mean")] for cell in cells) / len(cells)
            require(close(computed, ref["arm_window_means"][source][metric]),
                    f"{window}: {arm} {metric} window mean does not match the record")
    return cells, ref


def interval_rows(record, block_days):
    matches = [item for item in record["results"]["intervals"] if item["block_days"] == block_days]
    require(len(matches) == 1, f"missing {block_days}-day block interval")
    return {(row["period"], row["metric"]): row for row in matches[0]["rows"]}


def recompute_rank_ic(window, seed, arm):
    run = CSZ_RUN if arm == "v1" else RAW_RUN
    predictions = pd.read_parquet(run / f"inference/{window}/seed{seed}_predictions.parquet")
    labels = pd.read_parquet(LABELS[window], columns=["date", "instrument", "vwap_return"])
    merged = predictions.merge(labels, on=["date", "instrument"], how="inner")
    merged = merged.dropna(subset=["vwap_return"])
    require(len(merged) > 0, f"{window}/{seed}/{arm}: no prediction rows have labels")
    sections = pd.Series(
        {timestamp: float(frame["score"].corr(frame["vwap_return"], method="spearman"))
         for timestamp, frame in merged.groupby("date", sort=True)})
    daily = sections.groupby(sections.index.normalize()).mean()
    return {"rank_ic_mean": float(daily.mean()), "sections": int(len(sections)),
            "days": int(len(daily)), "sample_count": int(len(merged))}


def version_pair():
    version = read_json(VERSION_RECORD)
    require(version["experiment_id"] == "model-versions-20261023", "unexpected version record")
    v0_config = read_json(ROOT / version["versions"]["v0"]["config"]["path"])
    v1_config = read_json(ROOT / version["versions"]["v1"]["config"]["path"])
    assert_only_label_transform_changes(v0_config, v1_config)
    v0_run = ROOT / version["versions"]["v0"]["run"]
    v1_run = ROOT / version["versions"]["v1"]["run"]
    pairs = {}
    for label, run, archived in (
            ("v0", v0_run, RAW_RUN / "fits/research_a_s11/modified_tlob_weights.json"),
            ("v1", v1_run, CSZ_RUN / "fits/research_a_s11/modified_tlob_weights.json")):
        archived_tensors = numeric_tensors(archived)
        current_tensors = numeric_tensors(run / "modified_tlob_weights.json")
        shared = [key for key in archived_tensors.keys() & current_tensors.keys()
                  if archived_tensors[key].shape == current_tensors[key].shape]
        require(shared, f"{label}: no comparable tensors")
        difference = max(float(abs(archived_tensors[key] - current_tensors[key]).max()) for key in shared)
        require(difference == 0.0, f"{label}: weights are not bit-identical to the archived arm")
        pairs[label] = {"run": version["versions"][label]["run"],
                        "run_id": version["versions"][label]["run_id"],
                        "weights_vs_archived_arm_bit_identical": True,
                        "max_abs_difference": difference, "tensors": len(shared),
                        "metrics": version["versions"][label]["metrics"]}
    return version, pairs


def build():
    record = read_json(CSZ_RECORD)
    require(record["experiment_id"] == "label-cszscore-20261018", "unexpected CSZ record")
    require(record["seeds"]["full"] == list(SEEDS), "CSZ record did not complete three seeds")
    require(record["steps"]["candidate"] == 1500 and record["steps"]["reference"] == 1500,
            "unexpected CSZ steps")
    require(record["final_views"] == 0, "CSZ record opened final data")
    plan_path = ROOT / "configs" / Path(record["plan"]["path"]).name
    require(plan_path == ROOT / "configs/label_cszscore_20261018_plan.json", "unexpected plan path")
    require(file_sha256(plan_path) == record["plan"]["sha256"], "plan hash changed")

    cells = []
    windows = {}
    for window in WINDOWS:
        window_cells_found, ref = window_cells(record, window)
        cells.extend(window_cells_found)
        for cell in window_cells_found:
            cell_configs(window, cell["seed"])
        windows[window] = {
            "days": ref["days"],
            "train_dates": ref.get("train_dates"),
            "arm_window_means": ref["arm_window_means"],
            "per_seed_mean_delta": ref["per_seed_mean_delta"],
            "worst_day_delta": ref["worst_day_delta"],
            "min_day_delta": ref["min_day_delta"],
        }

    for cell in cells:
        values = {}
        for arm in ("v0", "v1"):
            values[arm] = recompute_rank_ic(cell["window"], cell["seed"], arm)
            expected = cell[arm]
            require(close(values[arm]["rank_ic_mean"], expected["rank_ic_mean"], atol=1e-9),
                    f"{cell['window']}/{cell['seed']}/{arm}: recomputed RankIC differs")
            require(values[arm]["sample_count"] == expected["sample_count"],
                    f"{cell['window']}/{cell['seed']}/{arm}: recomputed sample count differs")
            require(values[arm]["days"] == expected["day_count"],
                    f"{cell['window']}/{cell['seed']}/{arm}: recomputed day count differs")
        cell["prediction_recompute"] = values
    prediction_recompute = {
        "source": "archived seed predictions merged with the official monthly vwap_return labels",
        "cells_checked": len(cells) * 2,
        "all_match": True,
        "scope": "RankIC recomputed per complete timestamp with average ranks, then averaged by day; "
                 "prediction and sample coverage also checked.",
    }

    daily = pd.read_csv(CSZ_RUN / "paired_daily.csv")
    require(len(daily) == 20, "paired daily file must contain twenty seed-mean days")
    recomputed = {period: daily[daily["period"] == period][
        ["rank_ic", "long_short_return", "turnover"]].mean().to_dict()
        for period in ("research_a", "research_b")}
    recomputed["all"] = daily[["rank_ic", "long_short_return", "turnover"]].mean().to_dict()
    decision = record["results"]["decision"]
    for period in ("research_a", "research_b"):
        for metric in ("rank_ic", "long_short_return", "turnover"):
            require(close(recomputed[period][metric], windows[period]["arm_window_means"]["cand_csz"][metric]
                          - windows[period]["arm_window_means"]["ref10"][metric]),
                    f"{period}: paired daily mean does not match arm means")
    require(close(recomputed["all"]["rank_ic"],
                  decision["comparisons"]["candidate_vs_raw_label_reference"]["mean_rank_ic_delta"]),
            "pooled RankIC delta does not match the decision")
    rank_ic_positive_days = int((daily["rank_ic"] > 0).sum())
    worst_row = daily.loc[daily["long_short_return"].idxmin()]
    negative_days = [{"date": str(row.date), "rank_ic": float(row.rank_ic),
                      "long_short_return": float(row.long_short_return), "period": row.period}
                     for row in daily.itertuples() if row.rank_ic <= 0 or row.long_short_return <= 0]

    version, pairs = version_pair()
    csz_seed11 = record["results"]["windows"]["research_a"]["reference_comparison"][
        "per_seed_metrics"]["11"]["cand_csz"]
    raw_seed11 = record["results"]["windows"]["research_a"]["reference_comparison"][
        "per_seed_metrics"]["11"]["ref10"]
    for key in CELL_METRICS:
        require(close(pairs["v1"]["metrics"][key], csz_seed11[key]),
                f"V1 version metric {key} differs from the CSZ arm")
        require(close(pairs["v0"]["metrics"][key], raw_seed11[key]),
                f"V0 version metric {key} differs from the raw arm")

    intervals = {}
    for block in (5, 2):
        rows = interval_rows(record, block)
        intervals[str(block)] = {
            "rank_ic_all": rows[("all", "rank_ic")],
            "rank_ic_research_a": rows[("research_a", "rank_ic")],
            "rank_ic_research_b": rows[("research_b", "rank_ic")],
            "long_short_all": rows[("all", "long_short_return")],
            "turnover_all": rows[("all", "turnover")],
        }

    gates = record["gates"]
    checks = decision["checks"]
    result = {
        "schema_version": 1,
        "recorded_at_utc": utc_now(),
        "experiment_id": "model-versions-multi-seed-20261023",
        "policy": "V0 uses the raw official training label; V1 uses the full-timestamp "
                  "z-score rescaled by one training-pool constant. Network, inputs, training "
                  "dates, steps, batch, learning rate and seed are identical within every cell.",
        "scope": "Archived CSZ round reused as the multi-seed extension of V1 after proving "
                 "the V1 seed-11 research-A network is bit-identical to the archived CSZ arm. "
                 "No new training and no new data range were opened.",
        "sources": {
            "csz_record": {"path": CSZ_RECORD.relative_to(ROOT).as_posix(),
                           "sha256": file_sha256(CSZ_RECORD)},
            "version_record": {"path": VERSION_RECORD.relative_to(ROOT).as_posix(),
                               "sha256": file_sha256(VERSION_RECORD)},
            "plan": {"path": plan_path.relative_to(ROOT).as_posix(),
                     "sha256": record["plan"]["sha256"]},
            "csz_run": CSZ_RUN.relative_to(ROOT).as_posix(),
            "raw_run": RAW_RUN.relative_to(ROOT).as_posix(),
        },
        "identity_checks": {
            "v1_seed11_research_a_bit_identical_to_archived_csz_arm":
                pairs["v1"]["weights_vs_archived_arm_bit_identical"],
            "v0_seed11_research_a_bit_identical_to_archived_raw_arm":
                pairs["v0"]["weights_vs_archived_arm_bit_identical"],
            "v0_v1_configs_differ_only_in_target_centering": True,
            "all_six_cells_share_data_splits_steps_batch_learning_rate_seed": True,
            "all_six_raw_arms_use_none_and_csz_arms_use_csz": True,
            "budget_caps_differ_between_archived_raw_and_csz_fits": True,
            "budget_cap_note": "Archived raw fits were capped at 3600 s / 6144 MiB / 100000 "
                               "samples; CSZ fits at 900 s / 9472 MiB / 79920 samples. Actual "
                               "steps, samples, samples seen, batch and learning rate are equal, "
                               "and no fit hit its cap.",
        },
        "cells": cells,
        "prediction_recompute": prediction_recompute,
        "windows": windows,
        "pooled_seed_mean_delta": {key: float(recomputed["all"][key])
                                   for key in DELTA_KEYS.values()},
        "intervals": intervals,
        "worst_day": {
            "research_a": windows["research_a"]["worst_day_delta"],
            "research_b": windows["research_b"]["worst_day_delta"],
            "seed_mean_worst_row": {"date": str(worst_row["date"]),
                                    "long_short_return": float(worst_row["long_short_return"]),
                                    "period": worst_row["period"]},
            "tolerance": gates["worst_day_tolerance"],
        },
        "gates": {
            "minimum_mean_gain": gates["minimum_mean_gain"],
            "guardrails": gates["policy"]["guardrails"],
            "checks": checks,
            "action": decision["action"],
            "candidate_status": record["candidate_status"],
            "conclusion": record["conclusion"],
        },
        "daily_diagnostics": {
            "seed_mean_rank_ic_positive_days": rank_ic_positive_days,
            "days": len(daily),
            "non_positive_days": negative_days,
        },
        "version_pair": {
            "source": VERSION_RECORD.relative_to(ROOT).as_posix(),
            "v0": pairs["v0"],
            "v1": pairs["v1"],
            "single_seed_pair_mean_delta": version["comparison"]["mean_delta"],
            "single_seed_pair_worst_day_delta": version["comparison"]["worst_day_delta"],
        },
        "style_adjusted": version["style_adjusted"],
        "limits": [
            "The six archived cells are the CSZ round's three seeds and two used research "
            "windows; they are selection data, not independent confirmation.",
            "The older pre-declared worst-day guardrail is not passed: research B's worst "
            "day is worse than the -1.3896761e-4 tolerance, so V1 is a leading version by "
            "average evidence, not a formal gate pass.",
            "Local RankIC, long-short and turnover are local approximations, not the official "
            "platform score.",
            "Style residualization is the frozen local proxy, not the official M.jyc_eval "
            "preprocessing; it is reused, not recomputed.",
            "Confirmation, locked and legacy final ranges were not opened; platform limits "
            "remain unverified.",
        ],
    }
    return result, daily, intervals


def report_text(record):
    lines = [
        "V0 vs V1 三种子 x 两窗证据（本地近似口径，复用已冻结归档）",
        "",
        "身份：V1 seed11 research-A 权重与归档 CSZ 臂逐张量相同；V0 seed11 research-A 权重与归档 raw 臂逐张量相同。",
        f"逐格 RankIC 已由归档预测 + 官方 vwap_return 独立复算，{record['prediction_recompute']['cells_checked']} 个手臂/窗口单元与冻结记录一致。",
        "V0=原始官方收益标签；V1=完整时间戳截面 z-score 后按训练池常数匹配中心化方差。每格网络、输入、训练日期、1500 步、batch 32、lr 1e-4、种子相同。",
        "",
        "逐种子 RankIC（V0 -> V1，括号内为差）：",
    ]
    for cell in record["cells"]:
        lines.append(f"  {cell['window']:<10} seed {cell['seed']:>2}: "
                     f"{cell['v0']['rank_ic_mean']:.6f} -> {cell['v1']['rank_ic_mean']:.6f} "
                     f"({cell['delta']['rank_ic']:+.6f})")
    lines += ["", "逐窗均值（seed 均值）："]
    for window in ("research_a", "research_b"):
        means = record["windows"][window]["arm_window_means"]
        deltas = record["windows"][window]["per_seed_mean_delta"]
        avg = sum(item["rank_ic"] for item in deltas.values()) / len(deltas)
        lines.append(f"  {window:<10} RankIC {means['ref10']['rank_ic']:.6f} -> "
                     f"{means['cand_csz']['rank_ic']:.6f} ({avg:+.6f}); "
                     f"多空 {means['ref10']['long_short_return']:+.6f} -> "
                     f"{means['cand_csz']['long_short_return']:+.6f}; "
                     f"换手 {means['ref10']['turnover']:.4f} -> {means['cand_csz']['turnover']:.4f}")
    pooled = record["pooled_seed_mean_delta"]
    lines += [
        "",
        f"20 日配对均值（V1-V0）：RankIC {pooled['rank_ic']:+.6f}，多空 {pooled['long_short_return']:+.6f}，换手 {pooled['turnover']:+.6f}",
        "",
        "连续日块 95% 区间（RankIC，全部 20 日）：",
    ]
    for block in ("5", "2"):
        row = record["intervals"][block]["rank_ic_all"]
        lines.append(f"  {block} 日块: {row['mean']:+.6f} [{row['interval'][0]:+.6f}, {row['interval'][1]:+.6f}]")
    lines += [
        "",
        f"预登记门槛：最小 RankIC 改善 {record['gates']['minimum_mean_gain']:.6f}；"
        f"最差日多空容忍 {-record['worst_day']['tolerance']:.7f}。",
        f"最差日：research_a {record['worst_day']['research_a']:+.7f}，"
        f"research_b {record['worst_day']['research_b']:+.7f}（seed 均值最差日 "
        f"{record['worst_day']['seed_mean_worst_row']['date']} "
        f"{record['worst_day']['seed_mean_worst_row']['long_short_return']:+.7f}）。",
        "",
        "门槛检查通过项：" + ", ".join(k for k, v in record["gates"]["checks"].items() if v),
        "门槛检查未通过项：" + ", ".join(k for k, v in record["gates"]["checks"].items() if not v),
        "",
        "直接统一入口版本对照（seed11 research_a）："
        f"RankIC {record['version_pair']['v0']['metrics']['rank_ic_mean']:.6f} -> "
        f"{record['version_pair']['v1']['metrics']['rank_ic_mean']:.6f} "
        f"({record['version_pair']['single_seed_pair_mean_delta']['rank_ic']:+.6f})；"
        "两版权重与归档手臂逐张量相同。",
        "风格残差重放（复用，不重算）："
        + "；".join(
            f"{window} V1-V0 RankIC "
            f"{record['style_adjusted']['windows'][window]['v1_minus_v0_processed']['rank_ic_mean']:+.6f}"
            f"（V1 保留 {100 * record['style_adjusted']['windows'][window]['v1_rank_ic_retained']:.1f}%）"
            for window in ("research_a", "research_b")),
        "",
        "结论：V1 在 6/6 个种子-窗口单元的 RankIC 上都高于 V0，两个窗口均为正，"
        "多空不降、换手下降；但 research B 最差日多空超过旧容忍，按旧预登记不能记为正式通过。"
        "V1 是平均意义上的领先版本，尾部风险仍需在后续版本/确认区复核。",
        "",
        "限制：两窗三种子是已使用的选择数据；本地指标不是官方分数；确认/锁定/final 未打开。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "modified_tlob/model_versions_multi_seed_20261023.json")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "runs/model-versions/multi-seed-20261023/report.txt")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite a recorded evidence file: {args.output}")
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite a report: {args.report}")
    record, _daily, _intervals = build()
    write_json(args.output, record)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text(record), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    print(report_text(record))


if __name__ == "__main__":
    main()
