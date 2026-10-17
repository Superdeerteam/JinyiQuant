"""Prepare, run and audit the fixed forward three-seed experiment (no seed/weight search)."""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from jinyiquant.checks import batch_checks, check, prediction_checks, verify_artifacts
from jinyiquant.config import Period, RunConfig
from jinyiquant.data import build_batch, history_diagnostics, load_data, select_targets
from jinyiquant.evaluation import evaluate
from jinyiquant.io import canonical_json, digest, file_sha256, read_json, utc_now, write_json
from jinyiquant.runner import _answer_coverage_check, _kill_tree, inference_keys, save_checks
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir, resource_usage
from modified_tlob.ensemble import (
    KEYS,
    SEEDS,
    FixedSeedEnsemble,
    average_predictions,
    paired_block_intervals,
    research_decision,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "fixed-seed-ensemble-20261017"


def job_recorder(output):
    rec = Recorder(output, EXPERIMENT)
    rec.run_id = output.name + "_" + digest(str(output.resolve()))[:12]
    return rec


def process_tree_usage(pid):
    parent = psutil.Process(pid)
    processes = [parent, *parent.children(recursive=True)]
    return {"rss_mb": sum(p.memory_info().rss for p in processes) / 2**20,
            "cpu_seconds": sum(sum(p.cpu_times()[:2]) for p in processes), "process_count": len(processes)}


def claim_training(claim, path, period, seed, plan_path):
    record = {"period": period, "seed": seed, "run": str(path), "claimed_at_utc": utc_now(),
              "plan_sha256": file_sha256(plan_path)}
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        with claim.open("x", encoding="utf-8") as handle:
            handle.write(canonical_json(record))
    except FileExistsError:
        old = read_json(claim)
        if (old.get("status") != "pretraining_interrupted" or old.get("training_steps") != 0
                or old.get("plan_sha256") != record["plan_sha256"]):
            raise ValueError("Training claim already spent; no additional fits allowed") from None
        archive = Path(old["run"])
        events = [json.loads(line) for line in (archive / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        if ((archive / "model.pt").exists() or (archive / "training.json").exists()
                or any(e["stage"] == "training" and e["event_type"] in {"started", "progress", "completed"}
                       for e in events)):
            raise ValueError("Cannot recover a claim after training started")
        record["previous_pretraining_attempt"] = old
        write_json(claim, record)


def frame_digest(frame):
    frame = frame.sort_values(KEYS).reset_index(drop=True).copy()
    frame["date"] = frame.date.astype(str)
    return digest(frame.to_dict("records"))


def prepare(protocol_path, plan_path):
    """Freeze the old proposal and its calibration before opening the new research labels."""
    protocol = read_json(protocol_path)
    if plan_path.exists():
        raise FileExistsError("Frozen plan already exists")
    calibration = ROOT / protocol["calibration"]["file"]
    if file_sha256(calibration) != protocol["calibration"]["sha256"]:
        raise ValueError("Historical calibration digest differs")
    old = pd.read_csv(calibration)
    old = old[old.steps == 1500]
    if (len(old) != 6 or old.duplicated(["period", "seed"]).any()
            or set(old.seed) != set(SEEDS) or old.period.nunique() != 2):
        raise ValueError("Calibration needs the original two-window three-seed grid")
    noise = protocol["calibration"]["seed_noise"]
    for metric in ("rank_ic_mean", "long_short_return_mean", "turnover_mean", "min_daily_long_short_return"):
        actual = float(np.sqrt(old.groupby("period")[metric].var(ddof=1).mean()))
        if not np.isclose(actual, noise[metric], atol=1e-14, rtol=1e-10):
            raise ValueError("Historical seed-noise calculation differs")
    windows = [w for w in protocol["candidate_windows"] if w["role"] == "research"]
    if [w["name"] for w in windows] != ["research_a", "research_b"]:
        raise ValueError("The two declared research origins are required")
    overlaps = []
    excluded_synthetic = 0
    for path in (ROOT / "runs").rglob("config.json"):
        raw = read_json(path)
        data = raw.get("data", {}) if isinstance(raw, dict) else {}
        if (data.get("version", "").startswith("demo")
                or data.get("source") in {"synthetic", "generated", "demo"}):
            excluded_synthetic += 1
            continue
        splits = raw.get("splits", {}) if isinstance(raw, dict) else {}
        for role, split in splits.items():
            if role not in {"train", "validation"} or not {"start", "end"}.issubset(split):
                continue
            for w in windows:
                if (pd.Timestamp(split["start"]) < pd.Timestamp(w["evaluation_end_exclusive"])
                        and pd.Timestamp(split["end"]) > pd.Timestamp(w["evaluation_start"])):
                    overlaps.append({"config": str(path), "split": role, "window": w["name"]})
    if overlaps:
        raise ValueError(f"Historical run exposure overlaps proposed windows: {overlaps}")
    base = read_json(ROOT / "configs/training_duration_20261015/july_18_21_s11_steps1500.json")
    configs = {}
    for window in windows:
        raw = read_json(ROOT / "configs/training_duration_20261015/july_18_21_s11_steps1500.json")
        raw.update(experiment_id=EXPERIMENT, hypothesis=protocol["experiment_1"]["hypothesis"],
                   output_root=str(ROOT / "runs" / EXPERIMENT / "fits"))
        raw["data"].update(bars_path=str(ROOT / "数据/cpt_jyc_2026_e2e_bar1m"),
            labels_path=str(ROOT / "数据/cpt_jyc_2026_vwap"),
            instruments_path=str(ROOT / "数据/cpt_jyc_2026_instruments(1).parquet"),
            instrument_map_path=str(ROOT / "数据/cpt_jyc_2026_stock_bar1m_instrument_mapping.csv"),
            cache_dir=str(ROOT / "cache" / EXPERIMENT))
        raw["model"]["log_every_steps"] = 100
        raw["logging"]["learning_curves"] = False
        raw["splits"] = {"train": {"start": window["train_start"], "end": window["train_end_exclusive"]},
                         "validation": {"start": window["evaluation_start"], "end": window["evaluation_end_exclusive"]},
                         "final": {"start": "2024-10-08", "end": "2025-01-01"}}
        configs[window["name"]] = RunConfig.model_validate(raw).snapshot()
    registry_path = ROOT / "runs/data_roles.json"
    registry = {"schema_version": 1, "created_at_utc": utc_now(), "protocol_sha256": file_sha256(protocol_path),
                "windows": protocol["candidate_windows"], "active_experiment": EXPERIMENT,
                "allowed_label_roles": ["research_train", "research_evaluation"],
                "legacy_reserved": {"start": "2023-08-03", "end": "2023-08-04"},
                "enforcement": "This experiment's guarded_data rejects labels outside the registered research ranges; "
                               "not an OS security boundary or proof of absent unrecorded exposure.",
                "historical_run_overlap_count": len(overlaps)}
    registry["synthetic_configs_excluded"] = excluded_synthetic
    if registry_path.exists():
        raise FileExistsError("Existing shared role registry must be reconciled, never overwritten")
    write_json(registry_path, registry)
    inventory = {}
    for config in configs.values():
        first = pd.Timestamp(config["splits"]["train"]["start"]) - pd.Timedelta(days=10)
        last = pd.Timestamp(config["splits"]["validation"]["end"])
        for month in pd.period_range(first, last - pd.Timedelta(days=1), freq="M"):
            for path in (ROOT / f"数据/cpt_jyc_2026_e2e_bar1m/{month.strftime('%Y%m')}.0.feather",
                         ROOT / f"数据/cpt_jyc_2026_vwap/cpt_jyc_2026_vwap_{month}.parquet"):
                inventory[str(path)] = file_sha256(path)
        for name in ("instruments_path", "instrument_map_path"):
            path = Path(config["data"][name])
            inventory[str(path)] = file_sha256(path)
    plan = {"schema_version": 1, "experiment_id": EXPERIMENT, "registered_at_utc": utc_now(),
            "protocol": protocol, "protocol_sha256": file_sha256(protocol_path),
            "registry_path": str(registry_path), "registry_sha256": file_sha256(registry_path),
            "windows": windows, "configs": configs, "seeds": list(SEEDS), "max_training_attempts": 6,
            "policy": {"minimum_mean_gain": protocol["experiment_1"]["minimum_mean_gain"], "noise": noise},
            "statistics": protocol["statistics"], "budget": protocol["execution_budget"],
            "inputs_sha256": inventory, "recipe_reference": base["model"],
            "public_method": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html"}
    write_json(plan_path, plan)
    return plan


def guarded_data(plan, config, window, start, end, *, labels):
    if file_sha256(plan["registry_path"]) != plan["registry_sha256"]:
        raise ValueError("Shared data-role registration changed")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    allowed = [(window["train_start"], window["train_end_exclusive"]),
               (window["evaluation_start"], window["evaluation_end_exclusive"])]
    if not any(pd.Timestamp(lo) <= start < end <= pd.Timestamp(hi) for lo, hi in allowed):
        raise ValueError("Access outside registered research train/evaluation roles")
    data = config.data.model_dump(mode="json")
    if not labels:
        data["labels_path"] = None
    bundle = load_data(data, start=start, end=end)
    if not labels and not bundle.labels.empty:
        raise ValueError("Inference loaded labels")
    return bundle


def record_data(rec, path, bundle):
    write_json(path, bundle.metadata)
    for event in bundle.cache_events:
        rec.event("data_cache", event["event"], **{k: v for k, v in event.items() if k != "event"})
    for event in bundle.data_events:
        rec.event("data", event["event"], level="WARNING", **{k: v for k, v in event.items() if k != "event"})


def small_batch(batch, n=32):
    return replace(batch, keys=batch.keys.iloc[:n].reset_index(drop=True), x=batch.x[:n],
                   observed=batch.observed[:n], history_dates=batch.history_dates[:n])


def fit_worker(plan, window, config, output, rec):
    from modified_tlob.adapter import TLOBAdapter

    rec.event("data", "loading", role="research_train")
    bundle = guarded_data(plan, config, window, config.splits.train.start, config.splits.train.end, labels=True)
    record_data(rec, output / "data.json", bundle)
    targets = select_targets(bundle.labels, config.splits.train, sample_times=config.data.sample_times)
    dates = sorted(targets.date.dt.strftime("%Y-%m-%d").unique())
    if dates != window["train_dates"] or len(targets) > config.budget.max_train_samples:
        raise ValueError("Training dates or sample budget differ from declaration")
    rec.event("preprocessing", "started", samples=len(targets))
    batch = build_batch(bundle.bars, targets[KEYS], config.data)
    targets = batch.keys.merge(targets, on=KEYS, validate="one_to_one")
    mask = batch.effective_rows.sum(axis=1) >= config.data.min_history
    checks = batch_checks(batch, targets, config.splits.train, config.acceptance.min_coverage)
    checks.append(_answer_coverage_check(inference_keys(bundle, config.splits.train, config.data), targets,
                                        config.acceptance.min_coverage, "train_answer_coverage", "instruments"))
    checks.append(check("ten_training_dates_all_eight_times", len(dates) == 10
                        and targets.groupby(targets.date.dt.normalize()).date.nunique().eq(8).all()))
    if not save_checks(output, checks):
        raise ValueError("Training data checks failed")
    history_diagnostics(batch).to_parquet(output / "train_history.parquet", index=False)
    targets.to_parquet(output / "train_targets.parquet", index=False)
    write_json(output / "selection.json", targets.attrs.get("selection", {}))
    write_json(output / "coverage.json", batch.coverage)
    write_json(output / "training_identity.json", {"keys_sha256": frame_digest(targets[KEYS]),
        "labels_sha256": frame_digest(targets[[*KEYS, "target"]]), "dates": dates, "samples": len(targets)})
    rec.event("data", "validated", samples=len(targets), dates=dates, checks_passed=True)
    rec.event("preprocessing", "completed", resources=resource_usage())
    # Release raw history/labels before optimization; no research-evaluation tensors or labels exist here.
    del bundle
    gc.collect()
    model = TLOBAdapter(config.model, config.data, seed=config.seed, recorder=rec)
    rec.event("training", "started", seed=config.seed, evaluation_labels_loaded=False)
    progress = model.fit(batch, targets.target.to_numpy(), sample_mask=mask)
    if progress["steps"] != 1500:
        raise ValueError("Did not complete exactly 1500 optimizer steps")
    rec.event("training", "completed", **progress)
    write_json(output / "training.json", {**progress, "target_scale": model.target_scale,
        "prediction_units": "original_label_return", "evaluation_labels_loaded": False,
        "training_std": float(targets.target.to_numpy(dtype=np.float32)[mask].astype(np.float64).std())})
    rec.event("saving", "started")
    model.save(output / "model.pt")
    model.export_text(output / "modified_tlob_weights.json")
    rec.event("saving", "completed", sha256=file_sha256(output / "model.pt"))
    probe = small_batch(batch)
    expected = model.predict(probe)
    for filename in ("model.pt", "modified_tlob_weights.json"):
        rec.event("loading", "started", filename=filename)
        restored = TLOBAdapter.load(output / filename, device=config.model.device)
        checks.append(check("save_load_" + filename,
            np.allclose(expected, restored.predict(probe), atol=config.acceptance.repro_atol,
                        rtol=config.acceptance.repro_rtol)))
        del restored
        rec.event("loading", "completed", filename=filename)
    checks.append(check("training_only_target_scale", np.isclose(model.target_scale,
        targets.target.to_numpy(dtype=np.float32)[mask].astype(np.float64).std(), atol=1e-14, rtol=1e-10)))
    if not save_checks(output, checks):
        raise ValueError("Model save/load or scale check failed")
    rec.event("validation", "completed", checks_passed=True)
    del probe, batch, model, targets
    gc.collect()


def evaluation_chunks(window, count=5):
    dates = window["evaluation_dates"]
    for offset in range(0, len(dates), count):
        chunk = dates[offset:offset + count]
        yield chunk, pd.Timestamp(chunk[0]), pd.Timestamp(chunk[-1]) + pd.Timedelta(days=1)


def infer_worker(plan, window, config, paths, output, rec, *, pilot=False):
    import torch

    from modified_tlob.adapter import TLOBAdapter

    checks, frames, costs = [], {s: [] for s in SEEDS}, []
    if not pilot:
        rec.event("loading", "started", paths={str(s): str(p) for s, p in paths.items()})
        ensemble = FixedSeedEnsemble.load(paths, recorder=rec)
        weight_hashes = {str(s): file_sha256(p) for s, p in paths.items()}
        rec.event("loading", "completed", weight_hashes=weight_hashes)
    for index, (dates, first, last) in enumerate(evaluation_chunks(window)):
        if pilot:
            dates, last = dates[:1], first + pd.Timedelta(days=1)
        prestarted = time.monotonic()
        bundle = guarded_data(plan, config, window, first, last, labels=False)
        record_data(rec, output / f"data_{index}.json", bundle)
        keys = inference_keys(bundle, Period(start=first, end=last), config.data)
        if pilot:
            keys = keys.iloc[:32]
        rec.event("preprocessing", "started", chunk=index, samples=len(keys), labels_loaded=False)
        batch = build_batch(bundle.bars, keys, config.data)
        preprocessing_seconds = time.monotonic() - prestarted
        cutoffs = batch.keys.date.to_numpy(dtype="datetime64[ns]")[:, None]
        checks += [check(f"causal_history_{index}", ((batch.history_dates <= cutoffs)
                    | np.isnat(batch.history_dates)).all()),
                   check(f"history_coverage_{index}", batch.coverage["min_section_coverage"]
                         >= config.acceptance.min_coverage)]
        rec.event("data", "validated", chunk=index, labels_loaded=False, samples=len(keys))
        rec.event("preprocessing", "completed", chunk=index, elapsed_seconds=preprocessing_seconds)
        if pilot:
            model = TLOBAdapter(config.model, config.data, seed=11, recorder=rec)
            model.network.eval()
            torch.cuda.synchronize()
            start = time.monotonic()
            with torch.inference_mode():
                value = model.network(torch.from_numpy(batch.x).to(model.device))
            torch.cuda.synchronize()
            write_json(output / "pilot.json", {"labels_loaded": False, "training_steps": 0,
                "samples": len(keys), "forward_seconds": time.monotonic() - start,
                "preprocessing_seconds": preprocessing_seconds, "resources": resource_usage(),
                "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
                "finite": bool(torch.isfinite(value).all())})
            checks.append(check("pilot_forward_finite", bool(torch.isfinite(value).all())))
            break
        probe = small_batch(batch)
        # Warm each network, then alternate timing order across chunks to reduce ordering bias.
        for seed in SEEDS:
            ensemble.models[seed].predict(probe)
        scores, timings = {}, {}
        rec.event("inference", "started", chunk=index, samples=len(keys), label_input=False)
        arms = ["baseline", "ensemble"] if index % 2 == 0 else ["ensemble", "baseline"]
        for arm in arms:
            torch.cuda.synchronize()
            started = time.monotonic()
            if arm == "baseline":
                scores[arm] = ensemble.models[11].predict(batch)
            else:
                components = {s: batch.keys.assign(score=ensemble.models[s].predict(batch)) for s in SEEDS}
                scores[arm] = average_predictions(components).score.to_numpy()
                for seed in SEEDS:
                    frames[seed].append(components[seed])
            torch.cuda.synchronize()
            timings[arm] = time.monotonic() - started
        checks.append(check(f"seed11_reference_reused_{index}", np.allclose(scores["baseline"],
            components[11].score, atol=config.acceptance.repro_atol, rtol=config.acceptance.repro_rtol)))
        costs.append({"chunk": index, "dates": dates, "samples": len(keys), "timing_order": arms,
                      "shared_preprocessing_seconds": preprocessing_seconds,
                      "baseline_seconds": timings["baseline"], "ensemble_seconds": timings["ensemble"],
                      "same_keys_sha256": frame_digest(batch.keys), "resources": resource_usage()})
        if index == 0:
            stocks = set(probe.keys.instrument)
            checks.extend(prediction_checks(ensemble, probe, bundle.bars[bundle.bars.instrument.isin(stocks)],
                config.data, config.acceptance.repro_atol, config.acceptance.repro_rtol))
        rec.event("inference", "completed", **costs[-1])
        history_diagnostics(batch).to_parquet(output / f"evaluation_history_{index}.parquet", index=False)
        del batch, probe, bundle, components, scores
        gc.collect()
    if not pilot:
        for seed in SEEDS:
            pd.concat(frames[seed], ignore_index=True).to_parquet(output / f"seed{seed}_predictions.parquet", index=False)
        average_predictions({s: pd.concat(frames[s], ignore_index=True) for s in SEEDS}).to_parquet(
            output / "ensemble_predictions.parquet", index=False)
        checks.append(check("weights_frozen_across_all_chunks", weight_hashes ==
                            {str(s): file_sha256(p) for s, p in paths.items()}))
        write_json(output / "inference_cost.json", {"chunks": costs,
            "baseline_seconds": sum(c["baseline_seconds"] for c in costs),
            "ensemble_seconds": sum(c["ensemble_seconds"] for c in costs),
            "shared_preprocessing_seconds": sum(c["shared_preprocessing_seconds"] for c in costs),
            "measurement": "Synchronized CUDA; same full input keys; warmup excluded; shared preprocessing reported separately",
            "platform_verified": False})
        rec.event("saving", "completed", predictions_frozen=True, prediction_units="original_label_return")
    if not save_checks(output, checks):
        raise ValueError("Inference behavior checks failed")


def complete_correlations(metrics, expected_sections):
    coverage = metrics["correlation_coverage"]
    return all(coverage[name]["eligible_sections_sha256"] == coverage[name]["valid_sections_sha256"]
               and coverage[name]["eligible_section_count"] == coverage[name]["valid_section_count"]
               == expected_sections for name in ("rank_ic", "ic"))


def score_window(plan, window, config, inference, output, rec, *, target_archive=None):
    """Open research answers only after all six weights and predictions are fixed."""
    targets, checks = [], []
    if target_archive is not None:
        targets = pd.read_parquet(target_archive)
        rec.event("loading", "archived_targets_reused", path=str(target_archive), sha256=file_sha256(target_archive))
    else:
        for index, (_, first, last) in enumerate(evaluation_chunks(window)):
            bundle = guarded_data(plan, config, window, first, last, labels=True)
            record_data(rec, output / f"scoring_data_{window['name']}_{index}.json", bundle)
            targets.append(select_targets(bundle.labels, config.splits.validation, sample_times=config.data.sample_times))
            del bundle
            gc.collect()
        targets = pd.concat(targets, ignore_index=True)
    targets.to_parquet(output / f"{window['name']}_targets.parquet", index=False)
    daily, summary = {}, {}
    for arm, filename in (("baseline", "seed11_predictions.parquet"), ("seed29", "seed29_predictions.parquet"),
                          ("seed47", "seed47_predictions.parquet"), ("candidate", "ensemble_predictions.parquet")):
        predictions = pd.read_parquet(inference / filename)
        checks.append(_answer_coverage_check(predictions[KEYS], targets, config.acceptance.min_coverage,
                                            arm + "_answer_coverage", "instruments"))
        scored = predictions.merge(targets[KEYS], on=KEYS, validate="one_to_one")
        metrics, days, sections = evaluate(scored, targets, config.evaluation, return_sections=True,
                                          undefined_constant_portfolios=True)
        checks.append(check(arm + "_complete_correlation_coverage", complete_correlations(metrics, 80)))
        checks.append(check(arm + "_exact_ten_day_eight_time_grid", sorted(days.date.tolist()) == window["evaluation_dates"]
                            and len(sections) == 80 and sections.groupby("date").size().eq(8).all()))
        prefix = output / f"{window['name']}_{arm}"
        write_json(prefix.with_suffix(".json"), metrics)
        days.to_csv(str(prefix) + "_daily.csv", index=False)
        sections.to_csv(str(prefix) + "_by_timestamp.csv", index=False)
        daily[arm], summary[arm] = days, metrics
    if not all(c["passed"] for c in checks):
        write_json(output / f"{window['name']}_checks.json", {"passed": False, "checks": checks})
        raise ValueError("Incomplete scoring population or daily correlation coverage")
    pd.testing.assert_frame_equal(daily["baseline"][["date"]], daily["candidate"][["date"]])
    paired = daily["baseline"][["date"]].copy()
    paired["period"] = window["name"]
    for metric in ("rank_ic", "long_short_return", "turnover"):
        paired[metric] = daily["candidate"][metric] - daily["baseline"][metric]
    row = {"period": window["name"], "baseline": summary["baseline"], "candidate": summary["candidate"],
           "individual_seeds_descriptive_only": {"29": summary["seed29"], "47": summary["seed47"]},
           "worst_day_delta": summary["candidate"]["min_daily_long_short_return"]
               - summary["baseline"]["min_daily_long_short_return"]}
    write_json(output / f"{window['name']}_checks.json", {"passed": True, "checks": checks})
    rec.event("evaluation", "completed", window=window["name"], summary=row)
    return paired, row


def launch(plan_path, output, kind, period, *, seed=11, paths=None, remaining=3600):
    """Independent process watchdog, including preprocessing, training and inference resources."""
    output.mkdir(parents=True, exist_ok=False)
    plan = read_json(plan_path)
    config = plan["configs"][period] | {"seed": seed}
    write_json(output / "config.json", config)
    write_json(output / "job.json", {"kind": kind, "period": period, "paths": paths or {}})
    rec = job_recorder(output)
    started = time.monotonic()
    runtime = environment()
    runtime["determinism"].update(device=config["model"]["device"],
        torch_version=runtime["dependencies"].get("torch"), deterministic_algorithms=True,
        cpu_threads=config["model"]["cpu_threads"],
        note="Explicit device request; successful worker device/determinism events confirm actual execution")
    write_json(output / "manifest.json", {"schema_version": 1, "run_id": rec.run_id,
        "experiment_id": EXPERIMENT, "started_at": utc_now(), "status": "running", "kind": kind,
        "period": period, "seed": seed, "plan_sha256": file_sha256(plan_path),
        "code": capture_source(output), "environment": runtime})
    rec.event("run", "started", kind=kind, seed=seed)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT), "PYTHONHASHSEED": str(seed),
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    cap = min(3600, remaining)
    peak, last_resource = 0., 0.
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "scripts.run_seed_ensemble", "worker", str(plan_path),
                                    "--output", str(output)], stdout=log, stderr=log, env=env,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            while process.poll() is None:
                try:
                    usage = process_tree_usage(process.pid)
                    rss = usage["rss_mb"]
                    peak = max(peak, rss)
                    elapsed = time.monotonic() - started
                    if elapsed - last_resource >= 5:
                        rec.event("resources", "sample", **usage, elapsed_seconds=elapsed)
                        last_resource = elapsed
                    if elapsed > cap or peak > plan["budget"]["local_peak_rss_cap_mib"]:
                        _kill_tree(process)
                        rec.event("budget", "exceeded", level="ERROR", elapsed_seconds=elapsed, peak_rss_mb=peak)
                        finish(output, "budget_exceeded", 124, resources={"wall_seconds": elapsed, "peak_rss_mb": peak},
                               error={"reason": "local wall/RSS cap", "recovery": "stop; do not add fits"})
                        raise RuntimeError(f"Budget exceeded: {output}")
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.2)
        except BaseException as exc:
            if process.poll() is None:
                _kill_tree(process)
            if read_json(output / "manifest.json")["status"] == "running":
                rec.failure("supervision", exc)
                finish(output, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", 130,
                       resources={"wall_seconds": time.monotonic() - started, "peak_rss_mb": peak},
                       error={"reason": str(exc)})
            raise
    resources = {"wall_seconds": time.monotonic() - started, "peak_rss_mb": peak}
    finish(output, "completed" if process.returncode == 0 else "failed", process.returncode, resources=resources)
    if process.returncode:
        raise RuntimeError(f"Worker failed; see {output / 'worker.log'}")
    print(f"{kind} {period} seed={seed}: completed {resources}", flush=True)


def execute(plan_path, root):
    plan = read_json(plan_path)
    started = time.monotonic()
    rec = Recorder(root, EXPERIMENT)
    write_json(root / "manifest.json", {"schema_version": 1, "run_id": root.name, "experiment_id": EXPERIMENT,
        "started_at": utc_now(), "status": "running", "plan_sha256": file_sha256(plan_path),
        "code": capture_source(root), "environment": environment(), "final_views": 0})
    write_json(root / "preregistration.json", plan)
    attempts = []
    try:
        rec.event("run", "started", max_fits=6, weights=[1 / 3] * 3, seeds=list(SEEDS))
        for path, checksum in plan["inputs_sha256"].items():
            if file_sha256(path) != checksum:
                raise ValueError("Source dataset differs from the frozen plan")
        first = plan["windows"][0]["name"]
        launch(plan_path, root / "pilot", "pilot", first)
        inference_paths, archive_paths = {}, {}
        for window in plan["windows"]:
            period, paths, identities = window["name"], {}, []
            for seed in SEEDS:
                if len(attempts) >= plan["max_training_attempts"]:
                    raise RuntimeError("Training cap exhausted")
                path = root / "fits" / f"{period}_s{seed}"
                claim = ROOT / "runs" / EXPERIMENT / "training_claims" / f"{period}_s{seed}.json"
                claim_training(claim, path, period, seed, plan_path)
                attempts.append({"period": period, "seed": seed, "path": str(path), "status": "started"})
                write_json(root / "attempts.json", attempts)
                remaining = plan["budget"]["research_stage_wall_cap_seconds"] - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Research stage budget exhausted")
                launch(plan_path, path, "fit", period, seed=seed, remaining=remaining)
                attempts[-1]["status"] = "completed"
                write_json(root / "attempts.json", attempts)
                paths[str(seed)] = str(path / "model.pt")
                identities.append(read_json(path / "training_identity.json"))
            if any(identity != identities[0] for identity in identities):
                raise ValueError("Training keys/labels differ across seeds")
            archive_paths[period] = paths
        write_json(root / "weights_frozen.json", {period: {s: {"path": p, "sha256": file_sha256(p)}
            for s, p in paths.items()} for period, paths in archive_paths.items()})
        for window in plan["windows"]:
            period = window["name"]
            path = root / "inference" / period
            remaining = plan["budget"]["research_stage_wall_cap_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Research stage budget exhausted")
            launch(plan_path, path, "infer", period, paths=archive_paths[period], remaining=remaining)
            inference_paths[period] = path
        write_json(root / "predictions_frozen.json", {p: {f.name: file_sha256(f) for f in path.glob("*predictions.parquet")}
                                                     for p, path in inference_paths.items()})
        analyze(plan, root, inference_paths, rec)
        if time.monotonic() - started > plan["budget"]["research_stage_wall_cap_seconds"]:
            raise TimeoutError("Research stage budget exhausted during scoring")
        rec.event("run", "completed", training_attempts=len(attempts), final_views=0)
        finish(root, "completed", 0, resources={"wall_seconds": time.monotonic() - started})
    except BaseException as exc:
        rec.failure("experiment", exc)
        write_json(root / "decision.json", {"result": "inconclusive", "action": "retain_seed11_reference",
            "reason": str(exc), "training_attempts": len(attempts), "final_views": 0, "model_promoted": False})
        finish(root, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", 1,
               resources={"wall_seconds": time.monotonic() - started}, error={"reason": str(exc)})
        raise


def analyze(plan, root, inference_paths, rec, *, target_archives=None):
    paired, summaries, costs = [], [], {}
    for window in plan["windows"]:
        config = RunConfig.model_validate(plan["configs"][window["name"]])
        inference = inference_paths[window["name"]]
        if not verify_artifacts(inference)["passed"]:
            raise ValueError("Inference artifacts changed before scoring")
        days, summary = score_window(plan, window, config, inference, root, rec,
                                     target_archive=(target_archives or {}).get(window["name"]))
        paired.append(days)
        summaries.append(summary)
        costs[window["name"]] = read_json(inference / "inference_cost.json")
    paired = pd.concat(paired, ignore_index=True)
    paired.to_csv(root / "paired_daily.csv", index=False)
    intervals = [paired_block_intervals(paired, block_days=b) for b in (5, 2)]
    write_json(root / "paired_intervals.json", intervals)
    write_json(root / "metrics.json", {"windows": summaries, "inference_cost": costs})
    decision = research_decision(paired, summaries, intervals, plan["policy"], cost_measured=bool(costs))
    write_json(root / "decision.json", decision)
    rec.event("evaluation", "decision", **decision)
    print(decision, flush=True)


def recover_analysis(source, destination):
    """Continue scoring from frozen completed predictions, with zero additional fits/inference."""
    source = Path(source).resolve()
    if not verify_artifacts(source)["passed"]:
        raise ValueError("Original execution archive changed")
    original = read_json(source / "manifest.json")
    if original["status"] != "failed":
        raise ValueError("Recovery requires a preserved failed execution")
    plan = read_json(source / "preregistration.json")
    attempts = read_json(source / "attempts.json")
    if len(attempts) != 6 or any(a["status"] != "completed" for a in attempts):
        raise ValueError("Recovery cannot create or repeat training attempts")
    frozen = read_json(source / "predictions_frozen.json")
    inference_paths = {w["name"]: source / "inference" / w["name"] for w in plan["windows"]}
    for period, path in inference_paths.items():
        if read_json(path / "manifest.json")["status"] != "completed" or not verify_artifacts(path)["passed"]:
            raise ValueError("Completed intact inference required")
        if any(file_sha256(path / name) != value for name, value in frozen[period].items()):
            raise ValueError("Frozen predictions changed")
    output = new_run_dir(Path(destination).resolve())
    rec = Recorder(output, EXPERIMENT)
    started = time.monotonic()
    write_json(output / "manifest.json", {"schema_version": 1, "run_id": output.name,
        "experiment_id": EXPERIMENT, "started_at": utc_now(), "status": "running",
        "execution_source": str(source), "execution_manifest_sha256": file_sha256(source / "manifest.json"),
        "code": capture_source(output), "environment": environment(), "new_training_fits": 0,
        "new_inference_runs": 0, "final_views": 0, "recovery_reason": original["error"]})
    write_json(output / "preregistration.json", plan)
    write_json(output / "attempts.json", attempts)
    try:
        rec.event("run", "started", recovery=True, new_training_fits=0, frozen_predictions_reused=True)
        targets = {w["name"]: source / f"{w['name']}_targets.parquet" for w in plan["windows"]
                   if (source / f"{w['name']}_targets.parquet").is_file()}
        analyze(plan, output, inference_paths, rec, target_archives=targets)
        rec.event("run", "completed", new_training_fits=0, final_views=0)
        finish(output, "completed", 0, resources={"wall_seconds": time.monotonic() - started})
        return output
    except BaseException as exc:
        rec.failure("recovery_analysis", exc)
        finish(output, "failed", 1, error={"reason": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "worker", "recover-analysis"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    if args.command == "prepare":
        prepare(args.protocol.resolve(), plan_path)
        print(str(plan_path), flush=True)
    elif args.command == "recover-analysis":
        print(recover_analysis(plan_path, args.output or ROOT / "runs" / EXPERIMENT / "analysis-recovery"))
    elif args.command == "run":
        root = new_run_dir((args.output or ROOT / "runs" / EXPERIMENT).resolve())
        print(str(root), flush=True)
        execute(plan_path, root)
    else:
        output = args.output.resolve()
        rec = job_recorder(output)
        try:
            plan, job = read_json(plan_path), read_json(output / "job.json")
            if file_sha256(plan_path) != read_json(output / "manifest.json")["plan_sha256"]:
                raise ValueError("Frozen plan changed")
            config = RunConfig.model_validate(read_json(output / "config.json"))
            window = next(w for w in plan["windows"] if w["name"] == job["period"])
            if job["kind"] == "fit":
                fit_worker(plan, window, config, output, rec)
            else:
                infer_worker(plan, window, config, {int(s): Path(p) for s, p in job["paths"].items()},
                             output, rec, pilot=job["kind"] == "pilot")
            rec.event("run", "completed", kind=job["kind"])
        except BaseException as exc:
            rec.failure("worker", exc)
            raise


if __name__ == "__main__":
    main()
