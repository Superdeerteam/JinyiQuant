"""Compare preregistered final update budgets; diagnostic checkpoints never select a model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.checks import verify_artifacts
from jinyiquant.io import file_sha256, read_json, utc_now, write_json
from scripts.compare_training_round import compare

PRIMARY = ("rank_ic", "long_short_return")
RISKS = ("worst_day_long_short_return", "opening_rank_ic", "opening_long_short_return", "turnover")
FIELDS = {"rank_ic": "rank_ic_mean", "long_short_return": "long_short_return_mean",
          "turnover": "turnover_mean", "worst_day_long_short_return": "min_daily_long_short_return"}
KEYS = ["date", "instrument"]


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def baseline_budget(experiment):
    """Fine grids may extend below the retained reference; never infer it from the minimum."""
    baseline = experiment.get("baseline_budget", min(experiment["budgets"]))
    if baseline not in experiment["budgets"]:
        raise ValueError("baseline_budget must be one of the preregistered budgets")
    return baseline


def duration_gate(rows, intervals, registration, *, replication=False):
    """Require each paired window/seed to pass; a pooled gain cannot hide a failing cell."""
    experiment = registration["experiment"]
    seeds = [experiment["screen_seed"]]
    if replication:
        seeds += experiment["replication_seeds"]
    expected = {(period, seed) for period in experiment["periods"] for seed in seeds}
    actual = [(row["period"], row["seed"]) for row in rows]
    grid = len(actual) == len(set(actual)) and set(actual) == expected
    checks = []
    for row in rows:
        limits = registration["calibration"]["pooled_seed_std"]
        for metric in (*PRIMARY, *RISKS):
            delta = row["delta"].get(metric)
            lower, upper = (None, limits[metric]) if metric == "turnover" else (
                0 if metric in PRIMARY else -limits[metric], None)
            passed = (_finite(delta) and (lower is None or delta >= lower)
                      and (upper is None or delta <= upper))
            checks.append({"period": row["period"], "seed": row["seed"], "metric": metric,
                           "delta": delta, "lower_bound": lower, "upper_bound": upper, "passed": bool(passed)})
    means, practical = {}, []
    for metric in PRIMARY:
        values = [row["delta"].get(metric) for row in rows]
        mean = float(np.mean(values)) if values and all(_finite(v) for v in values) else None
        means[metric] = mean
        threshold = registration["calibration"]["pooled_seed_std"][metric]
        matching = [r for r in intervals if r["period"] == "all" and r["metric"] == metric]
        ci = matching[0].get("interval") if len(matching) == 1 else None
        positive_ci = (isinstance(ci, list) and len(ci) == 2 and all(_finite(x) for x in ci)
                       and ci[0] > 0)
        above = mean is not None and mean >= threshold
        practical.append({"metric": metric, "mean_delta": mean, "threshold": threshold,
                          "interval": ci, "positive_interval": bool(positive_ci),
                          "passed": bool(above and (positive_ci or not replication))})
    passed = grid and bool(checks) and all(r["passed"] for r in checks) and any(r["passed"] for r in practical)
    return {"passed": bool(passed), "phase": "replication" if replication else "screen",
            "complete_grid": grid, "rows": checks, "practical_improvement": practical,
            "mean_delta": means, "model_promoted": False}


def select_budget(gates, pooled_std):
    """Select only passing budgets; prefer the shorter budget for sub-noise differences."""
    eligible = sorted(int(budget) for budget, gate in gates.items() if gate["passed"])
    if not eligible:
        return None
    winner = eligible[0]
    for budget in eligible[1:]:
        left, right = gates[str(winner)]["mean_delta"], gates[str(budget)]["mean_delta"]
        equivalent = all(abs(right[m] - left[m]) < pooled_std[m] for m in PRIMARY)
        if not equivalent and right["rank_ic"] > left["rank_ic"]:
            winner = budget
    return winner


def collect_runs(registration, stage_paths, *, recover_failed_stages=False):
    """Collect complete cells; explicit recovery preserves stopped-stage attempts in the audit."""
    records, seen_paths, stages = {}, set(), []
    experiment = registration["experiment"]
    for stage in map(lambda p: Path(p).resolve(), stage_paths):
        manifest, plan, attempts = (read_json(stage / name) for name in
                                    ("manifest.json", "plan.json", "attempts.json"))
        status = manifest.get("status")
        recovering = recover_failed_stages and status in {"failed", "interrupted"}
        if status != "completed" and not recovering:
            raise ValueError(f"Stage is not completed: {stage}")
        declared = [r["name"] for r in plan["runs"]]
        actual = [r["name"] for r in attempts]
        if len(actual) != len(set(actual)) or len(declared) != len(set(declared)):
            raise ValueError("Stage plan and attempts must have unique names")
        if recovering:
            if (not attempts or actual != declared[:len(actual)] or len(actual) > len(declared)
                    or any(a.get("exit_code") != 0 for a in attempts[:-1])
                    or type(attempts[-1].get("exit_code")) is not int or attempts[-1]["exit_code"] == 0):
                raise ValueError("Stopped-stage attempts must be an ordered planned prefix ending in one failure")
        elif sorted(actual) != sorted(declared):
            raise ValueError("Stage attempts do not match the complete planned run list")
        declared_rows = {row["name"]: row for row in plan["runs"]}
        audit = {"path": str(stage), "status": status, "recovered_successful_prefix": bool(recovering),
                 "attempts_sha256": file_sha256(stage / "attempts.json"),
                 "manifest_sha256": file_sha256(stage / "manifest.json"),
                 "plan_sha256": file_sha256(stage / "plan.json"), "stage_error": manifest.get("error"),
                 "declared_recovery": plan.get("recovery"),
                 "attempts": [], "unexecuted": [{**row, "status": "not_executed"}
                                                   for row in plan["runs"][len(actual):]] if recovering else []}
        stages.append(audit)
        for attempt in attempts:
            if any(attempt.get(k) != v for k, v in declared_rows[attempt["name"]].items()
                   if k not in {"run", "exit_code"}):
                raise ValueError("Attempt metadata differs from its original planned cell")
            failed = attempt.get("exit_code") != 0
            if failed and not recovering:
                raise ValueError("Failed training attempt cannot be analyzed as completed")
            run = Path(attempt["run"]).resolve()
            config, run_manifest = (read_json(run / name) for name in ("config.json", "manifest.json"))
            period, seed, steps = attempt["period"], attempt["seed"], config["model"]["max_steps"]
            if (period not in experiment["periods"] or seed != config["seed"]
                    or seed not in [experiment["screen_seed"], *experiment["replication_seeds"]]
                    or steps not in experiment["budgets"] or attempt.get("steps", steps) != steps):
                raise ValueError("Attempt is outside the preregistered period/seed/budget grid")
            key = period, seed, steps
            if (not failed and key in records) or run in seen_paths:
                raise ValueError("Duplicate run or period/seed/steps cell")
            seen_paths.add(run)
            if failed:
                if (run_manifest.get("status") not in {"failed", "interrupted"}
                        or run_manifest.get("exit_code") != attempt["exit_code"]):
                    raise ValueError("Failed attempt must retain a matching terminal failure manifest")
                audit["attempts"].append({**attempt, "included_in_comparison": False,
                    "status": run_manifest["status"], "error": run_manifest.get("error"),
                    "resources": run_manifest.get("resources"),
                    "evidence_sha256": {name: file_sha256(run / name) for name in
                        ("manifest.json", "config.json", "checks.json", "events.jsonl", "learning_curve.json")
                        if (run / name).is_file()}})
                continue
            if run_manifest.get("status") != "completed" or config["evaluation_split"] != "validation":
                raise ValueError("Only completed research-validation runs are allowed")
            registered, started = (pd.Timestamp(registration["created_at_utc"]),
                                   pd.Timestamp(run_manifest["started_at"]))
            if (pd.isna(registered) or pd.isna(started) or registered.tzinfo is None
                    or started.tzinfo is None or registered > started):
                raise ValueError("Rules must have been registered before every compared training run")
            curve = read_json(run / "learning_curve.json")
            if not curve or curve[-1]["steps"] != steps:
                raise ValueError("Run did not reach its exact preregistered update budget")
            if any(a["steps"] >= b["steps"] for a, b in zip(curve, curve[1:])):
                raise ValueError("Learning curve must have unique increasing real update counts")
            records[key] = {"path": run, "config": config, "manifest": run_manifest, "curve": curve}
            audit["attempts"].append({**attempt, "included_in_comparison": True, "status": "completed",
                                      "manifest_sha256": file_sha256(run / "manifest.json")})
    required = {(p, experiment["screen_seed"], b) for p in experiment["periods"]
                for b in experiment["budgets"]}
    if not required.issubset(records):
        raise ValueError("Missing preregistered screening cells")
    return records, stages


def paired_rows(comparison, sections):
    """Extract terminal metrics and 09:30 outcomes from the verified full-universe evaluation."""
    summaries = {(r["period"], r["seed"], r["arm"]): r for r in comparison["summary"]}
    sections = sections.copy()
    opening = sections[pd.to_datetime(sections.timestamp).dt.strftime("%H:%M") == "09:30"]
    rows = []
    for pair in comparison["plan"]["pairs"]:
        row = {"period": pair["period"], "seed": pair["seed"]}
        for arm in ("baseline", "candidate"):
            summary = summaries[pair["period"], pair["seed"], arm]
            values = {metric: summary[field] for metric, field in FIELDS.items()}
            local = opening[(opening.period == pair["period"]) & (opening.seed == pair["seed"])
                            & (opening.arm == arm)]
            for metric in PRIMARY:
                numbers = local[metric].to_numpy(dtype=float)
                values["opening_" + metric] = (float(numbers.mean()) if len(numbers)
                                               and np.isfinite(numbers).all() else None)
            row[arm] = values
        row["delta"] = {metric: row["candidate"][metric] - row["baseline"][metric]
                        if all(_finite(row[a][metric]) for a in ("baseline", "candidate")) else None
                        for metric in (*PRIMARY, *RISKS)}
        rows.append(row)
    return rows


def reference_drift(records, registration):
    """Old-source runs are reproduction references, never strict statistical control arms."""
    references = {(r["period"], r["seed"]): r for r in registration["references"]}
    if len(references) != len(registration["references"]):
        raise ValueError("Duplicate historical references")
    for reference in references.values():
        old = Path(reference["run"])
        for filename, checksum in reference["sha256"].items():
            if file_sha256(old / filename) != checksum:
                raise ValueError(f"Historical reference changed: {old / filename}")
        if not verify_artifacts(old)["passed"]:
            raise ValueError(f"Historical reference artifacts changed: {old}")
    rows = []
    baseline = baseline_budget(registration["experiment"])
    for (period, seed, steps), record in sorted(records.items()):
        if steps != baseline:
            continue
        reference = references[period, seed]
        old = Path(reference["run"])
        old_config = read_json(old / "config.json")
        if old_config["splits"] != record["config"]["splits"]:
            raise ValueError("Training/validation windows differ from preregistered references")
        frames = [pd.read_parquet(p / "predictions.parquet").sort_values(KEYS).reset_index(drop=True)
                  for p in (old, record["path"])]
        valid_keys = all(not f.duplicated(KEYS).any() and not f[KEYS].isna().any().any() for f in frames)
        keys_equal = valid_keys and frames[0][KEYS].equals(frames[1][KEYS])
        tolerance = record["config"]["acceptance"]
        atol, rtol = tolerance["repro_atol"], tolerance["repro_rtol"]
        values = [f.score.to_numpy(dtype=float) for f in frames]
        finite = all(np.isfinite(v).all() for v in values)
        close = keys_equal and finite and np.allclose(*values, atol=atol, rtol=rtol)
        rows.append({"period": period, "seed": seed, "reference": str(old), "run": str(record["path"]),
                     "keys_equal": bool(keys_equal), "passed": bool(close), "atol": atol, "rtol": rtol,
                     "max_abs_prediction_difference": float(np.max(np.abs(values[0] - values[1])))
                     if keys_equal and finite else None, "strict_comparison_arm": False})
    return {"passed": bool(rows) and all(r["passed"] for r in rows), "rows": rows}


def _comparison(records, registration, baseline, candidate, destination):
    periods = list(registration["experiment"]["periods"])
    keys = sorted({(p, s) for p, s, b in records if b == baseline}
                  & {(p, s) for p, s, b in records if b == candidate})
    pairs = [{"period": p, "seed": s, "baseline": str(records[p, s, baseline]["path"]),
              "candidate": str(records[p, s, candidate]["path"])} for p, s in keys]
    plan = {"schema_version": 1, "experiment_id": registration["experiment_id"],
            "change": {"factor": "model.max_steps", "baseline": baseline, "candidate": candidate,
                       "hypothesis": "Determine the useful update budget for the fixed all-eight-time recipe."},
            "pairs": pairs, "screen_periods": periods, "bootstrap_samples": 2000, "bootstrap_seed": 20261015}
    plan_path = destination / f"plan_{baseline}_{candidate}.json"
    write_json(plan_path, plan)
    output = destination / f"comparison_{baseline}_{candidate}"
    result = compare(plan_path, output)
    if not result["checks"]["complete_target_eligible_correlation_coverage"]:
        raise ValueError("Incomplete eligible correlation coverage")
    return result, paired_rows(result, pd.read_csv(output / "by_timestamp.csv"))


def plot_curves(records, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    periods = sorted({key[0] for key in records})
    fields = [("train_mse", "Training MSE"), ("validation_mse", "Validation MSE"),
              ("validation_rank_ic_mean", "Validation RankIC"),
              ("validation_long_short_return_mean", "Validation gross long-short (bp)")]
    fig, axes = plt.subplots(len(periods), len(fields), figsize=(17, 4.3 * len(periods)), squeeze=False)
    for row, period in enumerate(periods):
        for col, (field, title) in enumerate(fields):
            ax = axes[row, col]
            for (p, seed, steps), record in sorted(records.items()):
                if p != period:
                    continue
                x = [r["steps"] for r in record["curve"]]
                scale = 10000 if field.endswith("long_short_return_mean") else 1
                y = [r.get(field) * scale if _finite(r.get(field)) else np.nan for r in record["curve"]]
                ax.plot(x, y, marker=".", alpha=.8, label=f"{steps:,} steps / seed {seed}")
            ax.set_title(f"{period.replace('_', ' ')}\n{title}")
            ax.set_xlabel("Optimizer updates (actual)")
            ax.grid(alpha=.2)
            if col == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Fixed recipe training duration | Intermediate checkpoints are diagnostic only", fontsize=14)
    fig.tight_layout(rect=(0, .025, 1, .95))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _summary_lines(decision, terminal, screen_seed):
    def number(value, *, scale=1, digits=6):
        return f"{value * scale:.{digits}f}" if _finite(value) else "未记录/未定义"

    audit = decision["recovery_audit"]
    baseline = decision["baseline_budget"]
    lines = ["固定配方训练步数实验",
             f"已完成 {len(terminal)} 次训练；仅使用研究验证区间，最终区间查看次数为 0。",
             "所有结论仅比较预注册训练终点；中途曲线用于诊断，不据其挑选模型。",
             f"旧 {baseline} 步参考预测一致性：{'通过' if decision['historical_reference_drift']['passed'] else '未通过，阻塞替换结论'}。",
             "", f"初筛两窗口等权均值（seed {screen_seed}；毛收益未扣交易成本）：",
             "预算步数 | 实际轮数均值 | RankIC | 毛多空 bp | 换手 | 两次运行合计分钟 | 峰值 RSS MiB"]
    screen = terminal[terminal.seed == screen_seed]
    for budget, group in screen.groupby("budget", sort=True):
        lines.append(f"{budget} | {group.epochs.mean():.3f} | {group.rank_ic.mean():.6f} | "
                     f"{group.long_short_return.mean() * 10000:.4f} | {group.turnover.mean():.6f} | "
                     f"{group.wall_seconds.sum() / 60:.2f} | {group.peak_rss_mb.max():.2f}")
    lines += ["", "逐窗口/种子终点（包括已执行的复验）：",
              "窗口 | 种子 | 步数 | 实际轮数 | RankIC | 毛多空 bp | 换手 | 分钟 | 峰值 RSS MiB"]
    for row in terminal.sort_values(["period", "seed", "budget"]).to_dict("records"):
        lines.append(f"{row['period']} | {row['seed']} | {row['budget']} | {row['epochs']:.3f} | "
                     f"{row['rank_ic']:.6f} | {row['long_short_return'] * 10000:.4f} | "
                     f"{row['turnover']:.6f} | {row['wall_seconds'] / 60:.2f} | {row['peak_rss_mb']:.2f}")
    failures = [a for stage in decision["stages"] for a in stage["attempts"] if not a["included_in_comparison"]]
    failed_costs = [(a.get("resources") or {}).get("wall_seconds") for a in failures]
    failed_seconds = sum(v for v in failed_costs if _finite(v))
    lines += ["", f"成功运行已记录总耗时 {terminal.wall_seconds.sum() / 60:.2f} 分钟；"
              f"失败尝试已记录耗时 {failed_seconds / 60:.2f} 分钟；"
              f"合计 {(terminal.wall_seconds.sum() + failed_seconds) / 60:.2f} 分钟。",
              "上述耗时是单次运行全过程的墙钟时间，含数据处理、曲线验证、保存及检查，不等同纯训练速度；"
              "未计系统重启停机和分析工具耗时。内存是进程峰值 RSS，不是 GPU 显存。",
              f"失败尝试 {len(failures)} 次；原计划未执行单元 {audit['unexecuted_original_cells']} 个；"
              "失败运行不参与效果比较。"]
    for attempt in failures:
        resources = attempt.get("resources") or {}
        lines.append(f"失败记录：{attempt['period']} / seed {attempt['seed']} / {attempt['steps']} 步预算；"
                     f"exit={attempt['exit_code']}；"
                     f"耗时 {number(resources.get('wall_seconds'), scale=1 / 60, digits=2)} 分钟；"
                     f"峰值 RSS {number(resources.get('peak_rss_mb'), digits=2)} MiB；归档 {attempt['run']}")
    labels = {"rank_ic": "RankIC", "long_short_return": "毛多空",
              "worst_day_long_short_return": "最差日毛多空", "opening_rank_ic": "开盘 RankIC",
              "opening_long_short_return": "开盘毛多空", "turnover": "换手"}
    for phase, gates in (("初筛", decision["screening"]), ("复验", decision["replication"])):
        for budget, gate in gates.items():
            lines += ["", f"{budget} 对 {baseline} 步{phase}：{'通过' if gate['passed'] else '未通过'}；"
                      f"平均 ΔRankIC {number(gate['mean_delta']['rank_ic'])}，"
                      f"平均 Δ毛多空 {number(gate['mean_delta']['long_short_return'], scale=10000)} bp。"]
            for check in gate["rows"]:
                if check["passed"]:
                    continue
                metric = check["metric"]
                scale, unit = (10000, " bp") if "return" in metric else (1, "")
                operator, limit = (">=", check["lower_bound"]) if check["lower_bound"] is not None else (
                    "<=", check["upper_bound"])
                lines.append(f"未通过：{check['period']} / seed {check['seed']} / {labels[metric]}；"
                             f"变化 {number(check['delta'], scale=scale)}{unit}，"
                             f"要求 {operator} {number(limit, scale=scale)}{unit}。")
            for check in gate["practical_improvement"]:
                if check["passed"]:
                    continue
                scale, unit = (10000, " bp") if "return" in check["metric"] else (1, "")
                lines.append(f"平均改善条件未通过：{labels[check['metric']]} "
                             f"{number(check['mean_delta'], scale=scale)}{unit}，"
                             f"门槛 >= {number(check['threshold'], scale=scale)}{unit}"
                             + (f"；同项 95% 日块区间下限须 > 0，实际区间 {check['interval']}（原始单位）"
                                if phase == "复验" else "") + "；至少一项平均改善条件必须通过。")
    recommended = decision["recommend_replication_budget"]
    replication_text = ("已执行，结果见上文" if decision["replication"] else
                        f"建议执行 {recommended} 步候选；分析工具未启动训练" if recommended else
                        "未触发，不追加种子训练")
    lines += ["", f"初筛选择预算：{decision['selected_screen_budget'] or '无候选通过'}；"
              f"本轮保留研究预算：{decision['retained_budget']} 步。",
              f"复验：{replication_text}。",
              "保留权重来源（固定最新窗口和初筛种子，未按最好种子挑选）："
              + decision["retained_weight_source"]["run"],
              "这是本地未扣交易成本的研究结果；旧日期重复研究及少量验证日限制了证据强度，未正式晋升模型。"]
    return lines


def analyze(registration_path, stage_paths, output, *, recover_failed_stages=False):
    registration_path, output = Path(registration_path).resolve(), Path(output).resolve()
    registration = read_json(registration_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {output}")
    if any(not _finite(registration["calibration"]["pooled_seed_std"].get(metric))
           or registration["calibration"]["pooled_seed_std"][metric] <= 0 for metric in (*PRIMARY, *RISKS)):
        raise ValueError("Every noise threshold must be positive and finite")
    records, stages = collect_runs(registration, stage_paths, recover_failed_stages=recover_failed_stages)
    if (any(output.is_relative_to(r["path"]) for r in records.values())
            or any(output.is_relative_to(Path(stage["path"])) for stage in stages)):
        raise ValueError("Analysis must not modify an input archive")
    drift = reference_drift(records, registration)
    output.mkdir(parents=True, exist_ok=False)
    experiment = registration["experiment"]
    baseline = baseline_budget(experiment)
    comparisons, screening, replication = {}, {}, {}
    table = []
    for budget in sorted(set(experiment["budgets"]) - {baseline}):
        comparison, rows = _comparison(records, registration, baseline, budget, output)
        comparisons[str(budget)] = {"path": str(output / f"comparison_{baseline}_{budget}"), "pairs": rows}
        screen = [r for r in rows if r["seed"] == experiment["screen_seed"]]
        screening[str(budget)] = duration_gate(screen, comparison["paired_intervals"], registration)
        if len(rows) > len(screen):
            replication[str(budget)] = duration_gate(rows, comparison["paired_intervals"], registration,
                                                     replication=True)
        for row in rows:
            for arm in ("baseline", "candidate"):
                summary = next(r for r in comparison["summary"] if r["period"] == row["period"]
                               and r["seed"] == row["seed"] and r["arm"] == arm)
                table.append({"budget": baseline if arm == "baseline" else budget, "comparison_budget": budget,
                              "period": row["period"], "seed": row["seed"], "arm": arm, **row[arm],
                              **{k: summary.get(k) for k in ("mse", "ic_mean", "training_samples", "steps",
                                 "epochs", "train_mse", "validation_mse", "validation_mse_over_constant")},
                              **summary["resources"]})
    selected = select_budget(screening, registration["calibration"]["pooled_seed_std"])
    accepted = (selected if selected is not None and drift["passed"]
                and replication.get(str(selected), {}).get("passed") else None)
    descriptive = None
    candidates = sorted(set(experiment["budgets"]) - {baseline})
    if len(candidates) == 2:
        result, rows = _comparison(records, registration, *candidates, output)
        descriptive = {"baseline": candidates[0], "candidate": candidates[1], "pairs": rows,
                       "paired_intervals": result["paired_intervals"], "used_for_selection": False}
    decision = {"schema_version": 1, "created_at": utc_now(), "experiment_id": registration["experiment_id"],
                "registration": str(registration_path), "registration_sha256": file_sha256(registration_path),
                "script_sha256": file_sha256(Path(__file__)), "stages": stages, "comparisons": comparisons,
                "recovery_audit": {"enabled": recover_failed_stages,
                    "stopped_stages": [s["path"] for s in stages if s["recovered_successful_prefix"]],
                    "excluded_failed_attempts": sum(not a["included_in_comparison"]
                                                    for s in stages for a in s["attempts"]),
                    "unexecuted_original_cells": sum(len(s["unexecuted"]) for s in stages),
                    "complete_unique_runs": len(records)},
                "screening": screening, "replication": replication, "historical_reference_drift": drift,
                "baseline_budget": baseline,
                "selected_screen_budget": selected, "accepted_research_budget": accepted,
                "recommend_replication_budget": selected if selected is not None and drift["passed"]
                and str(selected) not in replication else None,
                "retained_budget": accepted or baseline, "descriptive_longer_budget_comparison": descriptive,
                "checkpoint_selection": "Only preregistered terminal budgets; no intermediate curve selection",
                "model_promoted": False, "final_views": 0, "limitations": registration["limitations"]}
    period, seed = experiment["periods"][-1], experiment["screen_seed"]
    retained = (records[period, seed, accepted]["path"] if accepted else Path(next(
        r["run"] for r in registration["references"] if r["period"] == period and r["seed"] == seed)))
    decision["retained_weight_source"] = {
        "run": str(retained), "period": period, "seed": seed, "budget": decision["retained_budget"],
        "source": "replicated_research_candidate" if accepted else f"historical_{baseline}_reference",
        "weights": {name: {"path": str(retained / name), "sha256": file_sha256(retained / name)}
                    for name in ("model.pt", "modified_tlob_weights.json") if (retained / name).is_file()}}
    curve_records = [{"period": p, "seed": s, "budget": b, "run": str(r["path"]), "curve": r["curve"]}
                     for (p, s, b), r in sorted(records.items())]
    write_json(output / "learning_curves.json", curve_records)
    pd.DataFrame([{**{k: v for k, v in r.items() if k != "curve"}, **point}
                  for r in curve_records for point in r["curve"]]).to_csv(output / "learning_curves.csv", index=False)
    terminal = (pd.DataFrame(table).drop(columns=["comparison_budget", "arm"])
                .drop_duplicates(["budget", "period", "seed"]).reset_index(drop=True))
    terminal.to_csv(output / "terminal_metrics.csv", index=False)
    plot_curves(records, output / "training_duration.png")
    lines = _summary_lines(decision, terminal, screen_seed=experiment["screen_seed"])
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision["artifacts"] = {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}
    write_json(output / "decision.json", decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registration", type=Path)
    parser.add_argument("stages", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recover-failed-stages", action="store_true",
                        help="Explicitly merge audited failed/interrupted prefixes with completed recovery stages")
    args = parser.parse_args()
    result = analyze(args.registration, args.stages, args.output,
                     recover_failed_stages=args.recover_failed_stages)
    print(json.dumps({key: result[key] for key in ("selected_screen_budget", "accepted_research_budget",
                                                  "recommend_replication_budget", "retained_budget")}))


if __name__ == "__main__":
    main()
