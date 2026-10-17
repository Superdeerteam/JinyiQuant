"""Date/time diagnostics for frozen annual policies; never export an annual average."""

from __future__ import annotations

import argparse
import gc
import html
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.config import Period, RunConfig
from jinyiquant.data import select_targets
from jinyiquant.evaluation import KEYS, evaluate
from jinyiquant.io import file_sha256, read_json, utc_now, write_json
from jinyiquant.runner import inference_keys
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir, resource_usage
from scripts.run_seed_ensemble import guarded_data, record_data

METRICS = [
    "rank_ic", "ic", "long_return", "short_return", "short_position_return", "long_short_return", "turnover",
    "mse", "mae", "score_change_mae", "rank_persistence", "sample_count",
    "universe_count", "prediction_count", "target_count", "target_coverage", "prediction_coverage",
    "score_std", "target_std", "score_target_std_ratio", "score_p05", "score_p50", "score_p95",
    "target_p05", "target_p50", "target_p95", "weight_age_calendar_days", "weight_age_trading_days",
]
PAIRED_METRICS = [
    "rank_ic", "ic", "long_return", "short_return", "long_short_return", "turnover",
    "mse", "mae", "score_change_mae", "rank_persistence", "score_std", "score_target_std_ratio",
]
CONVENTIONS = {
    "scope": "Local approximate gross forward-label returns; no fees, execution, compounding or official score.",
    "granularity": "Every declared trading date and all eight sample times, separately for each policy.",
    "turnover": "Half L1 change in net long-short weights, consecutive sample times including overnight and month boundaries; first timestamp undefined.",
    "constant_scores": "RankIC, IC and portfolios undefined; missing values are not replaced by zero.",
    "short_return": "Return of the bottom-score basket before negation; short_position_return = -short_return; long_short_return = long_return + short_position_return.",
    "failure_flags": "rank_ic < 0 and long_short_return < 0 are descriptive failure flags, not significance or promotion tests. Undefined values are separate.",
    "summaries": "Monthly and month/time distributions only. No annual or pooled average, IR, Sharpe or aggregate score is exported.",
    "weight_age": "Calendar days and independent-universe trading dates strictly after last training date, through prediction date inclusive.",
    "comparison": "Monthly minus fixed on identical date/time/stock keys; one seed and one research year. Age and calendar conditions are confounded; no causal claim or automatic promotion.",
    "missing_labels": "Target coverage is measured against the independent stock universe at every timestamp; missing targets remain counted, never fabricated. Score distributions and portfolios use the identical available-target stock keys for both policies.",
    "public_method": "https://qlib.readthedocs.io/en/stable/component/report.html",
}


def expected_timestamps(plan):
    """Use the declared independent calendar, never dates inferred from labels."""
    days = plan.get("evaluation_dates") or [d for w in plan["windows"] for d in w["evaluation_dates"]]
    times = plan.get("sample_times") or next(iter(plan["configs"].values()))["data"]["sample_times"]
    if len(set(days)) != len(days) or len(times) != 8 or len(set(times)) != 8:
        raise ValueError("Independent calendar must have unique dates and eight unique times")
    grid = pd.DatetimeIndex([f"{day} {time}" for day in days for time in times]).sort_values()
    if grid.empty or (grid.year != 2022).any():
        raise ValueError("Annual diagnostics may only evaluate the declared 2022 research year")
    if "expected_grid" in plan and not grid.equals(pd.DatetimeIndex(plan["expected_grid"]).sort_values()):
        raise ValueError("Declared expected grid differs from independent calendar")
    return grid


def assert_grid(frame, expected, name, column="date"):
    actual = pd.DatetimeIndex(pd.to_datetime(frame[column]).unique()).sort_values()
    missing, extra = expected.difference(actual), actual.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(f"{name}: incomplete independent date/time grid; missing={missing[:8].tolist()}, extra={extra[:8].tolist()}")


def coverage_table(universe, predictions, targets, expected, minimum):
    """Check timestamps before key joins so entire absent days cannot disappear."""
    frames = {"universe": universe, "prediction": predictions, "target": targets}
    for name, frame in frames.items():
        assert_grid(frame, expected, name)
        if frame.duplicated(KEYS).any():
            raise ValueError(f"Duplicate {name} keys")
    coverage = universe.groupby("date").size().rename("universe_count").to_frame()
    for name, frame in [("prediction", predictions), ("target", targets)]:
        membership = frame[KEYS].merge(universe[KEYS], how="left", on=KEYS, indicator=True,
                                      validate="one_to_one")
        if membership._merge.ne("both").any():
            raise ValueError(f"{name} keys outside independent universe")
        coverage[f"{name}_count"] = frame.groupby("date").size()
        coverage[f"{name}_coverage"] = coverage[f"{name}_count"] / coverage.universe_count
        coverage[f"missing_{name}_count"] = coverage.universe_count - coverage[f"{name}_count"]
        if coverage[f"{name}_coverage"].lt(minimum).any():
            raise ValueError(f"Per-timestamp {name} coverage below {minimum}")
    missing = targets[KEYS].merge(predictions[KEYS], how="left", on=KEYS, indicator=True)
    if missing._merge.ne("both").any():
        raise ValueError("Predictions missing target keys; scores cannot be silently inner-joined away")
    return coverage.reset_index().rename(columns={"date": "timestamp"})


def continuous_sections(predictions, targets, config):
    """Bound memory by day, carrying the prior cross section across every boundary.

    A fresh evaluator sees the previous last section before the current day. Its
    turnover and score-change state is therefore exactly the uninterrupted state;
    the carried section's own metrics are discarded, including its initial NaN.
    """
    merged = predictions.merge(targets[KEYS + ["target"]], on=KEYS, how="inner", validate="one_to_one")
    previous, output = None, []
    for _, day in merged.groupby(merged.date.dt.normalize(), sort=True):
        frame = day if previous is None else pd.concat([previous, day], ignore_index=True)
        _, _, sections = evaluate(frame[KEYS + ["score"]], frame[KEYS + ["target"]], config,
                                  return_sections=True, undefined_constant_portfolios=True)
        output.append(sections if previous is None else sections.iloc[1:])
        previous = day.loc[day.date.eq(day.date.max())]
    return pd.concat(output, ignore_index=True)


def diagnose_policy(predictions, targets, universe, plan, sources, policy, config, history=None):
    expected = expected_timestamps(plan)
    coverage = coverage_table(universe, predictions, targets, expected, config.acceptance.min_coverage)
    sections = continuous_sections(predictions, targets, config.evaluation)
    sections["timestamp"] = pd.to_datetime(sections.timestamp)
    assert_grid(sections, expected, f"{policy} sections", "timestamp")
    if sections.timestamp.duplicated().any():
        raise ValueError("Duplicate evaluated timestamps")
    sections = sections.merge(coverage, on="timestamp", validate="one_to_one")
    sections["short_position_return"] = -sections.short_return
    sections["policy"] = policy
    sections["date"] = sections.timestamp.dt.strftime("%Y-%m-%d")
    sections["month"] = sections.timestamp.dt.strftime("%Y-%m")
    sections["time"] = sections.timestamp.dt.strftime("%H:%M")
    month_ids = sections.timestamp.dt.strftime("%Y%m")
    training = [sources[m][policy] for m in month_ids]
    last = pd.DatetimeIndex([m["last_training_date"] for m in training])
    boundary = pd.DatetimeIndex([m["train_end_exclusive"] for m in training])
    current = pd.DatetimeIndex(sections.timestamp.dt.normalize())
    if ((current <= last) | (current < boundary)).any():
        raise ValueError("Cannot evaluate weights before their training boundary")
    calendar = pd.DatetimeIndex(plan["trading_calendar"]).sort_values().unique()
    if not last.isin(calendar).all() or not current.isin(calendar).all():
        raise ValueError("Weight age dates absent from independent trading calendar")
    sections["weight_sha256"] = [m["sha256"] for m in training]
    sections["weight_fit_month"] = [m["fit_month"] for m in training]
    sections["last_training_date"] = last.strftime("%Y-%m-%d")
    sections["weight_age_calendar_days"] = (current - last).days
    sections["weight_age_trading_days"] = calendar.searchsorted(current, side="right") - calendar.searchsorted(last, side="right")
    sections["weight_changed"] = sections.weight_sha256.ne(sections.weight_sha256.shift())
    sections.loc[0, "weight_changed"] = False
    sections["negative_rank_ic"] = sections.rank_ic.lt(0) & sections.rank_ic.notna()
    sections["negative_long_short"] = sections.long_short_return.lt(0) & sections.long_short_return.notna()
    sections["undefined_rank_ic"] = sections.rank_ic.isna()
    sections["undefined_long_short"] = sections.long_short_return.isna()
    sections["undefined_turnover"] = sections.turnover.isna()
    aligned = predictions.merge(targets[KEYS + ["target"]], on=KEYS, validate="one_to_one")
    distributions = []
    for timestamp, group in aligned.groupby("date", sort=True):
        row = {"timestamp": timestamp}
        for value in ("score", "target"):
            row[f"{value}_std"] = float(group[value].std(ddof=0))
            row.update({f"{value}_p{p:02}": float(group[value].quantile(p / 100)) for p in [5, 50, 95]})
        row["score_target_std_ratio"] = row["score_std"] / row["target_std"] if row["target_std"] > 0 else np.nan
        distributions.append(row)
    sections = sections.merge(pd.DataFrame(distributions), on="timestamp", validate="one_to_one")
    if history is not None:
        history = history.copy()
        history["timestamp"] = pd.to_datetime(history.timestamp)
        assert_grid(history, expected, "inference history coverage", "timestamp")
        overlap = (set(history.columns) & set(sections.columns)) - {"timestamp"}
        for column in overlap:
            if column == "universe_count":
                joined = history.merge(sections[["timestamp", column]], on="timestamp", suffixes=("_history", "_score"))
                if not joined[f"{column}_history"].eq(joined[f"{column}_score"]).all():
                    raise ValueError("Inference and scoring independent universe counts differ")
            else:
                raise ValueError(f"History diagnostics unexpectedly overwrite {column}")
        sections = sections.merge(history.drop(columns=list(overlap)), on="timestamp", validate="one_to_one")
    return sections.sort_values("timestamp", ignore_index=True)


def distribution_summary(detail, keys, columns):
    """Each row must retain a calendar month; no pooled-year aggregate exists."""
    if "month" not in keys:
        raise ValueError("Diagnostic summaries must retain month")
    rows = []
    for group, frame in detail.groupby(keys, sort=True, observed=True):
        group = group if isinstance(group, tuple) else (group,)
        identity = dict(zip(keys, group))
        for column in columns:
            values = pd.to_numeric(frame[column], errors="raise").dropna()
            rows.append(identity | {"metric": column, "sections": len(frame), "valid_count": len(values),
                                   "undefined_count": len(frame) - len(values), "negative_count": int(values.lt(0).sum()),
                                   "zero_count": int(values.eq(0).sum()),
                                   "min": values.min(), "p10": values.quantile(.1), "median": values.median(),
                                   "p90": values.quantile(.9), "max": values.max()})
    return pd.DataFrame(rows)


def paired_differences(detail):
    keys = ["timestamp", "date", "month", "time"]
    fixed = detail[detail.policy.eq("fixed")]
    monthly = detail[detail.policy.eq("monthly")]
    paired = fixed[keys + PAIRED_METRICS].merge(monthly[keys + PAIRED_METRICS], on=keys,
                                             suffixes=("_fixed", "_monthly"), how="outer",
                                             validate="one_to_one", indicator=True)
    if paired._merge.ne("both").any():
        raise ValueError("Policy timestamp grids differ")
    for metric in PAIRED_METRICS:
        paired[f"{metric}_delta"] = paired[f"{metric}_monthly"] - paired[f"{metric}_fixed"]
    return paired.drop(columns="_merge")


def _heatmap(frame, metric):
    pivot = frame.pivot(index="date", columns="time", values=metric)
    finite = np.abs(pivot.to_numpy(dtype=float))
    scale = np.nanquantile(finite, .9) if np.isfinite(finite).any() else 1
    scale = max(float(scale), 1e-12)
    rows = []
    for date, values in pivot.iterrows():
        cells = [f"<th>{html.escape(date)}</th>"]
        for time, value in values.items():
            if pd.isna(value):
                text, color = "undefined", "#ddd"
            else:
                strength = min(abs(float(value)) / scale, 1) * .65
                rgb = "55,100,180" if metric == "turnover" else ("210,55,55" if value < 0 else "35,135,90")
                color = f"rgba({rgb},{strength:.3f})"
                text = f"{value:.6g}"
            cells.append(f'<td style="background:{color}" title="{date} {time}">{text}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><tr><th>Date</th>" + "".join(f"<th>{t}</th>" for t in pivot.columns) + "</tr>" + "".join(rows) + "</table>"


def write_reports(detail, output):
    """All reports derive solely from the exported date/time rows."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "by_timestamp.csv", index=False)
    detail.to_parquet(output / "by_timestamp.parquet", index=False)
    columns = [c for c in METRICS if c in detail]
    columns += [c for c in ("history_coverage", "short_history_count", "observed_count", "observed_fraction") if c in detail]
    monthly = distribution_summary(detail, ["policy", "month"], columns)
    times = distribution_summary(detail, ["policy", "month", "time"], columns)
    monthly.to_csv(output / "monthly_distributions.csv", index=False)
    times.to_csv(output / "monthly_eight_times.csv", index=False)
    aged = detail.assign(age_bucket=pd.cut(detail.weight_age_trading_days,
                        bins=[0, 5, 10, 20, 40, 60, 120, np.inf], right=True,
                        labels=["1-5", "6-10", "11-20", "21-40", "41-60", "61-120", "121+"]))
    distribution_summary(aged, ["policy", "month", "time", "age_bucket"], columns).to_csv(
        output / "monthly_time_weight_age.csv", index=False)
    paired = paired_differences(detail)
    paired.to_csv(output / "paired_by_timestamp.csv", index=False)
    distribution_summary(paired, ["month", "time"], [f"{c}_delta" for c in PAIRED_METRICS]).to_csv(
        output / "paired_month_time_distributions.csv", index=False)
    write_json(output / "conventions.json", CONVENTIONS)
    style = "<style>body{font:14px system-ui;margin:24px;color:#172336}table{border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{border:1px solid #ccd3dc;padding:5px 8px;white-space:nowrap}th{background:#edf1f5}details{margin:20px 0}.scroll{overflow:auto;max-height:70vh}p{max-width:1100px;line-height:1.5}</style>"
    links, findings = [], []
    for month, frame in detail.groupby("month", sort=True):
        body = [f"<h1>{month}：固定权重与每月重训</h1>",
                "<p>本地近似毛收益口径，不含手续费和实际成交。RankIC 或多空收益小于零表示该截面出现负向表现，不代表统计显著。灰色为未定义，不能当作零。换手使用蓝色深浅，不用好坏配色；每张月度热图独立缩放，应比较数值。月及时点分布保留缺失数量，全部指标的逐日逐时值在下方完整表中。</p>"]
        for policy, part in frame.groupby("policy", sort=True):
            body.append(f"<h2>{'固定权重' if policy == 'fixed' else '每月重训'} ({html.escape(policy)})</h2>")
            for metric in ("rank_ic", "long_short_return", "turnover"):
                body.append(f"<h3>{metric}：逐日 × 八时点</h3><div class='scroll'>{_heatmap(part, metric)}</div>")
            compact = times[(times.month == month) & (times.policy == policy) &
                            times.metric.isin(["rank_ic", "long_short_return", "turnover"])].copy()
            body.append("<details><summary>本月各时点分布与缺失数量</summary><div class='scroll'>" +
                        compact.to_html(index=False, na_rep="undefined", float_format=lambda x: f"{x:.7g}") + "</div></details>")
            failures = compact[compact.metric.isin(["rank_ic", "long_short_return"])]
            for metric, rows in failures.groupby("metric"):
                worst = rows.sort_values(["negative_count", "median", "time"], ascending=[False, True, True]).iloc[0]
                findings.append({"month": month, "policy": policy, "metric": metric, "time": worst.time,
                                 "negative_count": int(worst.negative_count), "valid_count": int(worst.valid_count),
                                 "undefined_count": int(worst.undefined_count),
                                 "median": float(worst["median"]) if pd.notna(worst["median"]) else None,
                                 "interpretation": "Most negative sections in this month; descriptive, no significance claim."})
        body.append("<details open><summary>全部指标的逐日逐时完整表</summary><div class='scroll'>" +
                    frame.to_html(index=False, na_rep="undefined", float_format=lambda x: f"{x:.8g}") + "</div></details>")
        delta = paired[paired.month.eq(month)]
        body.append("<details><summary>逐日逐时配对差：每月重训 − 固定权重</summary><div class='scroll'>" +
                    delta.to_html(index=False, na_rep="undefined", float_format=lambda x: f"{x:.8g}") + "</div></details>")
        filename = f"{month}.html"
        (output / filename).write_text("<!doctype html><meta charset='utf-8'>" + style + "".join(body), encoding="utf-8")
        frame.to_csv(output / f"{month}_by_timestamp.csv", index=False)
        links.append(f"<li><a href='{filename}'>{month}：月度详细诊断</a></li>")
    write_json(output / "monthly_failure_locations.json", findings)
    scope = " ".join(html.escape(text) for text in CONVENTIONS.values())
    index = "<!doctype html><meta charset='utf-8'>" + style + "<h1>2022 年基线诊断</h1>"
    index += "<p>请按月查看八个时点的逐日表现。本报告不输出全年或总体平均数、综合评分。权重日龄与市场月份存在混杂，单年份、单种子结果只能用于定位下一步研究问题，不能证明因果或自动晋升模型。</p>"
    index += "<ul>" + "".join(links) + "</ul><p>" + scope + "</p>"
    index += "<p><a href='by_timestamp.csv'>完整逐日逐时明细</a> · <a href='paired_by_timestamp.csv'>两种策略配对差</a> · <a href='monthly_time_weight_age.csv'>按月份、时点和权重日龄分组</a></p>"
    (output / "index.html").write_text(index, encoding="utf-8")


def analyze(root):
    # Deferred import avoids the executor's analyze delegate creating an import cycle.
    from scripts.annual_baseline import load_plan, validate_data_snapshot, verify_completed

    root = Path(root).resolve()
    plan = load_plan(root)
    output = new_run_dir(root / "analysis")
    rec = Recorder(output, plan["experiment_id"])
    write_json(output / "manifest.json", {"status": "running", "run_id": output.name,
               "experiment_id": plan["experiment_id"], "started_at": utc_now(),
               "plan_sha256": file_sha256(root / "plan.json"), "code": capture_source(output),
               "environment": environment(), "kind": "annual_diagnostics"})
    rec.event("evaluation", "started", reserved_period_views=0, annual_aggregates=False)
    try:
        target_frames, universe_frames, history_frames, sources, inputs = [], [], [], {}, {}
        for window in plan["windows"]:
            month = window["name"]
            verify_completed(root, month, "infer", plan)
            source_path = root / "infer" / month / "weights.json"
            sources[month] = read_json(source_path)
            inputs[str(source_path)] = file_sha256(source_path)
            for source in sources[month].values():
                inputs[source["path"]] = source["sha256"]
                inputs[str(Path(source["path"]).parent / "manifest.json")] = source["fit_manifest_sha256"]
            for policy in ("fixed", "monthly"):
                path = root / "infer" / month / f"{policy}.parquet"
                inputs[str(path)] = file_sha256(path)
            cfg = RunConfig.model_validate(plan["configs"][month])
            chunk_days = plan["budget"]["chunk_days"]
            for i in range(0, len(window["evaluation_dates"]), chunk_days):
                days = window["evaluation_dates"][i:i + chunk_days]
                start, end = pd.Timestamp(days[0]), pd.Timestamp(days[-1]) + pd.Timedelta(days=1)
                if start < pd.Timestamp("2022-01-01") or end > pd.Timestamp("2023-01-01"):
                    raise ValueError("Scoring request outside research year")
                bundle = guarded_data(plan, cfg, window, start, end, labels=True)
                validate_data_snapshot(plan, bundle.metadata)
                record_data(rec, output / f"data_{month}_{i}.json", bundle)
                target_frames.append(select_targets(bundle.labels, Period(start=start, end=end),
                                                     horizon_minutes=cfg.data.label_horizon_minutes,
                                                     sample_times=cfg.data.sample_times))
                universe_frames.append(inference_keys(bundle, Period(start=start, end=end), cfg.data))
                del bundle
                gc.collect()
            history_path = root / "infer" / month / "history_coverage.parquet"
            if not history_path.is_file():
                raise FileNotFoundError(f"Per-time inference history coverage missing: {history_path}")
            history_frames.append(pd.read_parquet(history_path))
            inputs[str(history_path)] = file_sha256(history_path)
            rec.event("evaluation", "labels_loaded", month=month, reserved_period_views=0, resources=resource_usage())
        targets, universe = pd.concat(target_frames, ignore_index=True), pd.concat(universe_frames, ignore_index=True)
        del target_frames, universe_frames
        history = pd.concat(history_frames, ignore_index=True)
        targets.to_parquet(output / "targets.parquet", index=False)
        universe.to_parquet(output / "universe_keys.parquet", index=False)
        write_json(output / "input_artifacts.json", inputs)
        write_json(output / "weights.json", sources)
        sections = []
        for policy in ("fixed", "monthly"):
            predictions = pd.concat([pd.read_parquet(root / "infer" / w["name"] / f"{policy}.parquet")
                                     for w in plan["windows"]], ignore_index=True)
            predictions["date"] = pd.to_datetime(predictions.date)
            sections.append(diagnose_policy(predictions, targets, universe, plan, sources, policy, cfg, history))
            rec.event("evaluation", "policy_completed", policy=policy, timestamp_rows=len(sections[-1]),
                      resources=resource_usage())
            del predictions
            gc.collect()
        detail = pd.concat(sections, ignore_index=True)
        write_reports(detail, output)
        if any(file_sha256(path) != sha for path, sha in inputs.items()):
            raise ValueError("Inference artifacts changed during analysis")
        write_json(output / "checks.json", {"passed": True, "year": 2022, "reserved_period_views": 0,
                   "independent_calendar_grid": True, "causal_weight_provenance": True,
                   "cross_month_turnover_continuous": True, "annual_aggregates_exported": False,
                   "timestamp_rows_per_policy": len(expected_timestamps(plan)), "months": 12, "sample_times": 8})
        rec.event("evaluation", "completed", output=str(output), resources=resource_usage())
        finish(output, "completed", 0, resources=resource_usage())
    except BaseException as exc:
        rec.failure("evaluation", exc)
        finish(output, "failed", 1, error={"reason": str(exc)}, resources=resource_usage())
        raise
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    print(analyze(parser.parse_args().root))
