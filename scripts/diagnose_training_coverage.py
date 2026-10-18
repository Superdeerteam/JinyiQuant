"""Quarter-origin inference and source-history diagnostics, with zero training."""

import argparse
import copy
import gc
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.checks import verify_artifacts
from jinyiquant.config import Period, RunConfig
from jinyiquant.data import build_batch, history_diagnostics, load_data, select_targets
from jinyiquant.io import file_sha256, read_json, utc_now, write_json
from jinyiquant.runner import inference_keys
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir, resource_usage
from scripts.analyze_annual_baseline import (
    METRICS,
    assert_grid,
    continuous_sections,
    coverage_table,
    distribution_summary,
)

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "modified_tlob/annual_baseline_2022_20261017/index.json"
EXP = "training-coverage-diagnostic-20261017"
QUALITY = ["history_row_count", "effective_history_length", "complete_history_length", "missing_fraction",
           "source_missing_fraction", "latest_bar_age_minutes", "latest_effective_age_minutes",
           "history_span_minutes"]


def quarter_jobs(index):
    jobs = []
    for number in range(1, 13):
        month = f"2022{number:02}"
        origin = f"2022{((number - 1) // 3) * 3 + 1:02}"
        reuse = "fixed" if origin == "202201" else ("monthly" if month == origin else None)
        jobs.append({"month": month, "origin": origin, "reuse_policy": reuse,
                     "weight": index["weights"][origin]["monthly"]})
    if sum(job["reuse_policy"] is None for job in jobs) != 6:
        raise ValueError("Exactly six supplemental weight/month combinations are allowed")
    return jobs


def exported_detail(index_path=INDEX):
    index = read_json(index_path)
    frames = []
    for entry in index["files"]:
        path = Path(index_path).parent / entry["path"]
        # Git's Windows checkout may change only the terminal LF to CRLF.
        payload = path.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(payload).hexdigest() != entry["sha256"] or len(payload) != entry["bytes"]:
            raise ValueError("Existing monthly export checksum differs")
        doc = read_json(path)
        frames.append(pd.DataFrame(doc["rows"], columns=doc["columns"]))
    detail = pd.concat(frames, ignore_index=True)
    detail["timestamp"] = pd.to_datetime(detail.timestamp)
    return detail


def quality_sections(quality):
    rows = []
    for stamp, group in quality.groupby("date", sort=True):
        row = {"timestamp": stamp, "month": stamp.strftime("%Y-%m"), "time": stamp.strftime("%H:%M"),
               "universe_count": len(group), "full_observed_window_count": int(group.full_observed_window.sum()),
               "full_observed_window_fraction": float(group.full_observed_window.mean()),
               "short_128_count": int(group.effective_history_length.lt(128).sum()),
               "no_effective_history_count": int(group.effective_history_length.eq(0).sum()),
               "latest_effective_not_at_cutoff_count": int(group.latest_effective_age_minutes.gt(0).sum()),
               "latest_effective_age_undefined_count": int(group.latest_effective_age_minutes.isna().sum())}
        for column in QUALITY:
            values = group[column].dropna()
            row.update({f"{column}_{name}": float(value) if pd.notna(value) else None
                        for name, value in [("min", values.min()), ("median", values.median()),
                                            ("p90", values.quantile(.9)), ("max", values.max())]})
        rows.append(row)
    return pd.DataFrame(rows)


def local_config(index, month, data_root, cache_dir=None):
    # The exported index intentionally omits local/remote filesystem locations.
    cfg = read_json(ROOT / "configs/fixed_seed_ensemble_20261017_plan.json")["configs"]["research_a"]
    cfg = copy.deepcopy(cfg)
    cfg["data"].update(index["configs"][month]["data"])
    cfg["data"].update(bars_path=str(data_root / "cpt_jyc_2026_e2e_bar1m"),
                        labels_path=str(data_root / "cpt_jyc_2026_vwap"),
                        instruments_path=str(data_root / "cpt_jyc_2026_instruments(1).parquet"),
                        instrument_map_path=str(data_root / "cpt_jyc_2026_stock_bar1m_instrument_mapping.csv"),
                        cache_dir=None if cache_dir is None else str(cache_dir))
    # Keep CPU preprocessing deterministic; checkpoint execution device is selected separately.
    cfg["model"]["device"] = "cpu"
    cfg["splits"] = index["configs"][month]["splits"]
    return RunConfig.model_validate(cfg)


def history_run(output_root, data_root, chunk_days=5):
    index = read_json(INDEX)
    run = new_run_dir(Path(output_root))
    rec = Recorder(run, EXP)
    write_json(run / "manifest.json", {"status": "running", "run_id": run.name, "experiment_id": EXP,
               "kind": "source_history_quality", "started_at": utc_now(), "code": capture_source(run),
               "environment": environment(), "index_sha256": file_sha256(INDEX)})
    rec.event("startup", "started", new_training=0, additional_inference=0, reserved_label_views=0)
    started, sections, inputs = time.monotonic(), [], {}
    try:
        for window in index["fit_windows"]:
            month, days = window["name"], window["evaluation_dates"]
            cfg = local_config(index, month, Path(data_root))
            for offset in range(0, len(days), chunk_days):
                dates = days[offset:offset + chunk_days]
                first, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)
                raw = cfg.data.model_dump(mode="json")
                raw["labels_path"] = None
                rec.event("data_validation", "started", month=month, dates=dates, labels_loaded=False)
                bundle = load_data(raw, start=first, end=end)
                if not bundle.labels.empty:
                    raise ValueError("History analysis must not load labels")
                for role in bundle.metadata["files"].values():
                    for source in role:
                        path = Path(source["path"])
                        if source["sha256"] != index["data_sha256"].get(path.name):
                            # Universe file had a different filename on Linux, same content.
                            if source["sha256"] not in index["data_sha256"].values():
                                raise ValueError(f"Source differs from annual index: {path.name}")
                        inputs[str(path)] = source["sha256"]
                keys = inference_keys(bundle, Period(start=first, end=end), cfg.data)
                rec.event("preprocessing", "started", month=month, rows=len(keys))
                batch = build_batch(bundle.bars, keys, cfg.data)
                cutoff = batch.keys.date.to_numpy(dtype="datetime64[ns]")[:, None]
                if not ((batch.history_dates <= cutoff) | np.isnat(batch.history_dates)).all():
                    raise ValueError("History contains future bars")
                quality = history_diagnostics(batch)
                detail = quality_sections(quality)
                detail.to_parquet(run / f"history_{month}_{offset}.parquet", index=False)
                sections.append(detail)
                rec.event("preprocessing", "completed", month=month, dates=dates, resources=resource_usage())
                del batch, bundle, quality
                gc.collect()
            rec.event("history_quality", "month_completed", month=month, elapsed_seconds=time.monotonic() - started)
        detail = pd.concat(sections, ignore_index=True)
        expected = pd.DatetimeIndex([f"{d} {t}" for w in index["fit_windows"]
                                    for d in w["evaluation_dates"] for t in index["sample_times"]])
        assert_grid(detail, expected, "source history", "timestamp")
        if detail.timestamp.duplicated().any():
            raise ValueError("Duplicate history timestamp")
        detail.to_csv(run / "by_timestamp.csv", index=False)
        columns = [c for c in detail if c not in {"timestamp", "month", "time"}]
        for name, keys in [("monthly", ["month"]), ("monthly_eight_times", ["month", "time"])]:
            distribution_summary(detail, keys, columns).to_csv(run / f"{name}.csv", index=False)
        write_json(run / "input_artifacts.json", inputs)
        write_json(run / "conventions.json", {
            "age": "Elapsed calendar minutes, including overnight, holidays and lunch; zero only at prediction cutoff. Positive age is descriptive, not automatically an error.",
            "window": "128 observed bar rows can span lunch or previous sessions; missing_fraction includes padding, source_missing_fraction excludes absent rows.",
            "statistics": "Per-timestamp source-stock distributions before ffill; monthly outputs describe distributions of those per-timestamp statistics, never pooled stock quantiles.",
            "reuse": "Reuse existing history_diagnostics/build_batch; annual export lacks per-time complete-window and age detail, so replay source preprocessing without labels or model execution.",
            "annual_average_exported": False})
        checks = {"passed": True, "timestamp_rows": len(detail), "months": 12, "sample_times": 8,
                  "new_training": 0, "additional_inference": 0, "labels_loaded": False,
                  "reserved_label_views": 0, "annual_aggregates_exported": False}
        write_json(run / "checks.json", checks)
        write_json(run / "metrics.json", {"scope": "Monthly/per-time CSV only", "wall_seconds": time.monotonic() - started})
        rec.event("shutdown", "completed", **checks)
        finish(run, "completed", 0, resources=resource_usage())
    except BaseException as exc:
        rec.failure("history_quality", exc)
        finish(run, "failed", 1, error={"reason": str(exc)})
        raise
    return run


def verify_quarter_sources(root, index):
    """Fail before inference if any required existing checkpoint/prediction is missing."""
    jobs = quarter_jobs(index)
    inputs, missing = {}, []
    for job in jobs:
        weight = Path(root) / "fit" / job["origin"] / "model.pt"
        if not weight.is_file():
            missing.append(str(weight))
        elif file_sha256(weight) != job["weight"]["sha256"]:
            raise ValueError(f"Existing checkpoint does not match annual index: {weight}")
        else:
            inputs[str(weight)] = job["weight"]["sha256"]
            manifest = weight.parent / "manifest.json"
            if file_sha256(manifest) != job["weight"]["fit_manifest_sha256"]:
                raise ValueError("Fit manifest differs from the indexed original")
            inputs[str(manifest)] = file_sha256(manifest)
        if job["reuse_policy"]:
            path = Path(root) / "infer" / job["month"] / f"{job['reuse_policy']}.parquet"
            if not path.is_file():
                missing.append(str(path))
            else:
                if not verify_artifacts(path.parent)["passed"]:
                    raise ValueError("Original inference artifact inventory failed")
                source = read_json(path.parent / "weights.json")[job["reuse_policy"]]
                if source["sha256"] != job["weight"]["sha256"]:
                    raise ValueError("Reused prediction checkpoint differs from declared quarter origin")
                inputs[str(path)] = file_sha256(path)
    return jobs, inputs, sorted(set(missing))


def reusable_predictions(path, window, index):
    """Accept an interrupted run's saved month only if it is the declared grid and finite."""
    predictions = pd.read_parquet(path)
    if not {"date", "instrument", "score"}.issubset(predictions.columns):
        raise ValueError(f"Resumed predictions lack required columns: {path}")
    stamp = pd.to_datetime(predictions.date)
    days = pd.DatetimeIndex(sorted(stamp.dt.strftime("%Y-%m-%d").unique()))
    if (not days.equals(pd.DatetimeIndex(window["evaluation_dates"]))
            or sorted(stamp.dt.strftime("%H:%M").unique()) != sorted(index["sample_times"])):
        raise ValueError(f"Resumed predictions do not cover the declared month/eight-time grid: {path}")
    if predictions.duplicated(["date", "instrument"]).any() or not np.isfinite(predictions.score).all():
        raise ValueError(f"Resumed predictions contain duplicates or nonfinite scores: {path}")
    return predictions


def resume_source(resume_from, jobs):
    """Refuse reuse from a run that declared different jobs or a different frozen index."""
    resume_from = Path(resume_from).resolve()
    manifest = read_json(resume_from / "manifest.json")
    if manifest.get("kind") != "quarter_origins" or manifest.get("index_sha256") != file_sha256(INDEX):
        raise ValueError("Resume source is not a quarter-origin run of the same frozen annual index")
    if read_json(resume_from / "jobs.json") != jobs:
        raise ValueError("Resume source declares different weight/month combinations")
    return resume_from


def inference_run(root, output_root, data_root, device="cuda", resume_from=None):
    from modified_tlob.adapter import TLOBAdapter

    index = read_json(INDEX)
    jobs, inputs, missing = verify_quarter_sources(root, index)
    if missing:
        raise FileNotFoundError("Original quarter checkpoint/prediction artifacts missing: " + "; ".join(missing))
    prior = None if resume_from is None else resume_source(resume_from, jobs)
    run = new_run_dir(Path(output_root))
    rec = Recorder(run, EXP)
    write_json(run / "manifest.json", {"status": "running", "run_id": run.name, "experiment_id": EXP,
               "kind": "quarter_origins", "started_at": utc_now(), "code": capture_source(run),
               "environment": environment(), "index_sha256": file_sha256(INDEX)})
    write_json(run / "jobs.json", jobs)
    rec.event("startup", "started", new_training=0, additional_weight_month_inferences_max=6,
              resumed_from=None if prior is None else str(prior))
    sections, previous, model, current_origin, inference_count, resumed = [], None, None, None, 0, []
    existing = exported_detail()
    try:
        for job in jobs:
            month, origin = job["month"], job["origin"]
            window = next(w for w in index["fit_windows"] if w["name"] == month)
            cfg = local_config(index, month, Path(data_root))
            period = cfg.splits.validation
            universe = pd.read_parquet(cfg.data.instruments_path, columns=["date", "instrument"],
                                        filters=[("date", ">=", period.start), ("date", "<", period.end)])
            targets_path = Path(cfg.data.labels_path) / f"cpt_jyc_2026_vwap_{month[:4]}-{month[4:]}.parquet"
            if file_sha256(targets_path) != index["data_sha256"][targets_path.name]:
                raise ValueError("Evaluation labels differ from original annual source")
            targets = select_targets(pd.read_parquet(targets_path, columns=["date", "instrument", "vwap_return"]),
                                     period, sample_times=index["sample_times"])
            targets = targets.merge(universe, on=["date", "instrument"], validate="one_to_one")
            inputs[str(targets_path)] = file_sha256(targets_path)
            if job["reuse_policy"]:
                path = Path(root) / "infer" / month / f"{job['reuse_policy']}.parquet"
                predictions = pd.read_parquet(path)
                rec.event("inference", "reused", month=month, origin=origin, path=str(path))
            else:
                if inference_count + len(resumed) >= 6:
                    raise ValueError("Supplemental inference combination limit exceeded")
                cached = None if prior is None else prior / f"quarterly_{month}.parquet"
                if cached is not None and cached.is_file():
                    predictions = reusable_predictions(cached, window, index)
                    inputs[str(cached)] = file_sha256(cached)
                    resumed.append({"month": month, "origin": origin, "path": str(cached),
                                    "sha256": inputs[str(cached)]})
                    rec.event("inference", "resumed", month=month, origin=origin,
                              sha256=inputs[str(cached)])
                else:
                    inference_count += 1
                    if current_origin != origin:
                        model = TLOBAdapter.load(Path(root) / "fit" / origin / "model.pt", device=device, recorder=rec)
                        current_origin = origin
                    frames = []
                    rec.event("inference", "started", month=month, origin=origin, combination=inference_count)
                    days = window["evaluation_dates"]
                    for offset in range(0, len(days), 5):
                        part = days[offset:offset + 5]
                        start, end = pd.Timestamp(part[0]), pd.Timestamp(part[-1]) + pd.Timedelta(days=1)
                        raw = cfg.data.model_dump(mode="json")
                        raw["labels_path"] = None
                        bundle = load_data(raw, start=start, end=end)
                        if not bundle.labels.empty:
                            raise ValueError("Inference reader loaded labels")
                        for records in bundle.metadata["files"].values():
                            for record in records:
                                if record["sha256"] not in index["data_sha256"].values():
                                    raise ValueError("Inference input changed since annual baseline")
                                inputs[record["path"]] = record["sha256"]
                        keys = inference_keys(bundle, Period(start=start, end=end), cfg.data)
                        batch = build_batch(bundle.bars, keys, cfg.data)
                        cutoff = batch.keys.date.to_numpy(dtype="datetime64[ns]")[:, None]
                        if not ((batch.history_dates <= cutoff) | np.isnat(batch.history_dates)).all():
                            raise ValueError("Inference contains future bars")
                        scores = model.predict(batch)
                        frames.append(keys.assign(score=scores))
                        rec.event("inference", "chunk_completed", month=month, dates=part, resources=resource_usage())
                        del bundle, batch
                        gc.collect()
                    predictions = pd.concat(frames, ignore_index=True)
                    predictions.to_parquet(run / f"quarterly_{month}.parquet", index=False)
            expected = pd.DatetimeIndex([f"{d} {t}" for d in window["evaluation_dates"] for t in index["sample_times"]])
            coverage_table(universe, predictions, targets, expected, cfg.acceptance.min_coverage)
            if not np.isfinite(predictions.score).all():
                raise ValueError("Nonfinite scores")
            # Recompute monthly metrics from saved predictions; carry previous section within quarter.
            prior_target = None if previous is None else previous[1]
            pred = predictions if prior_target is None else pd.concat([previous[0], predictions], ignore_index=True)
            truth = targets if prior_target is None else pd.concat([prior_target, targets], ignore_index=True)
            detail = continuous_sections(pred, truth, cfg.evaluation)
            detail["timestamp"] = pd.to_datetime(detail.timestamp)
            detail = detail.loc[detail.timestamp >= period.start].copy()
            assert_grid(detail, expected, "quarterly scoring", "timestamp")
            detail["short_position_return"] = -detail.short_return
            detail = detail.assign(policy="quarterly", month=detail.timestamp.dt.strftime("%Y-%m"),
                                   time=detail.timestamp.dt.strftime("%H:%M"), weight_fit_month=origin,
                                   weight_sha256=job["weight"]["sha256"])
            baseline = existing.loc[(existing.month == f"{month[:4]}-{month[4:]}") & existing.policy.eq("fixed")]
            if origin == "202201":
                joined = detail.merge(baseline[["timestamp", "rank_ic", "long_short_return"]], on="timestamp",
                                      suffixes=("", "_archived"), validate="one_to_one")
                if not all(np.allclose(joined[c], joined[c + "_archived"], atol=1e-12, rtol=1e-9)
                           for c in ["rank_ic", "long_short_return"]):
                    raise ValueError("Reused January-weight metrics differ from archived export")
            sections.append(detail)
            stamp = predictions.date.max()
            previous = (predictions.loc[predictions.date.eq(stamp)], targets.loc[targets.date.eq(stamp)])
            rec.event("evaluation", "month_completed", month=month, origin=origin)
        detail = pd.concat(sections, ignore_index=True)
        detail.to_csv(run / "by_timestamp.csv", index=False)
        columns = [c for c in METRICS if c in detail]
        distribution_summary(detail, ["policy", "month"], columns).to_csv(run / "monthly.csv", index=False)
        distribution_summary(detail, ["policy", "month", "time"], columns).to_csv(run / "monthly_eight_times.csv", index=False)
        paired = detail.merge(existing.loc[existing.policy.eq("fixed"), ["timestamp", "rank_ic", "long_short_return"]],
                              on="timestamp", suffixes=("_quarterly", "_january"), validate="one_to_one")
        for metric in ["rank_ic", "long_short_return"]:
            paired[metric + "_delta"] = paired[metric + "_quarterly"] - paired[metric + "_january"]
        paired.to_csv(run / "paired_by_timestamp.csv", index=False)
        if any(file_sha256(path) != sha for path, sha in inputs.items()):
            raise ValueError("Source artifact changed during diagnostics")
        write_json(run / "input_artifacts.json", inputs)
        write_json(run / "checks.json", {"passed": True, "new_training": 0, "additional_inference": inference_count,
                   "reused_combinations": 6, "resumed_combinations": len(resumed),
                   "supplemental_combinations_total": inference_count + len(resumed),
                   "resume_source": None if prior is None else str(prior), "resumed": resumed,
                   "reserved_label_views": 0, "annual_aggregates_exported": False})
        write_json(run / "metrics.json", {"scope": "Monthly/per-time CSV only", "causal_weight_age_effect": "not identified"})
        rec.event("shutdown", "completed", additional_inference=inference_count, resumed=len(resumed))
        finish(run, "completed", 0, resources=resource_usage())
    except BaseException as exc:
        rec.failure("quarter_origins", exc)
        finish(run, "failed", 1, error={"reason": str(exc)})
        raise
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["history", "inference", "inventory"])
    parser.add_argument("--annual-root", type=Path, default=ROOT / "runs/annual-baseline-2022-20261017")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/training-coverage-diagnostic")
    parser.add_argument("--data-root", type=Path, default=ROOT / "数据")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume-from", type=Path,
                        help="Reuse verified monthly predictions from an interrupted quarter-origin run")
    args = parser.parse_args()
    if args.mode == "inventory":
        jobs, inputs, missing = verify_quarter_sources(args.annual_root, read_json(INDEX))
        output = new_run_dir(args.output_root)
        write_json(output / "inventory.json", {"jobs": jobs, "inputs": inputs, "missing": missing})
        print(output)
    elif args.mode == "history":
        print(history_run(args.output_root, args.data_root))
    else:
        print(inference_run(args.annual_root, args.output_root, args.data_root, args.device, args.resume_from))
