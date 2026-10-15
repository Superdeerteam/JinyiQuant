"""Sequential research worker and an independent wall-time / memory watchdog."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from .checks import batch_checks, check, numerical_equal, prediction_checks, verify_artifacts
from .config import RunConfig, load_config
from .data import build_batch, load_data, select_targets
from .evaluation import evaluate
from .io import read_json, redact, utc_now, write_json
from .models import create_model, load_model, model_filename
from .tracking import (
    Recorder,
    capture_source,
    claim_final_access,
    environment,
    finish,
    new_run_dir,
    resource_usage,
)


def save_checks(run_dir, checks):
    value = {"schema_version": 1, "passed": all(c["passed"] for c in checks), "checks": checks}
    write_json(run_dir / "checks.json", value)
    return value["passed"]


def _aligned_targets(batch, targets):
    result = batch.keys.merge(targets, on=["date", "instrument"], how="left", validate="one_to_one")
    if result.target.isna().any():
        raise ValueError("Missing targets after exact key alignment")
    return result


def inference_keys(bundle, split, data_config):
    universe = bundle.universe if bundle.universe is not None else bundle.bars[["date", "instrument"]]
    expected = universe.loc[(universe.date >= split.start) & (universe.date < split.end)]
    expected = expected.loc[expected.date.dt.strftime("%H:%M").isin(data_config.sample_times)]
    # A dummy constant permits reuse of the trading-clock boundary selector; no answers are consulted.
    return select_targets(expected.assign(target=0.0), split, data_config.label_horizon_minutes)[["date", "instrument"]]


def _answer_coverage_check(expected, targets, minimum, name, universe_source):
    observed = targets[["date", "instrument"]].assign(present=1)
    aligned = expected.merge(observed, how="outer", on=["date", "instrument"], indicator=True,
                             validate="one_to_one")
    if (aligned["_merge"] == "right_only").any():
        raise ValueError("Selected targets lie outside the independently generated prediction universe")
    coverage = aligned.groupby("date").present.count() / aligned.groupby("date").size()
    return check(name, len(coverage) > 0 and coverage.min() >= minimum,
                 minimum=float(coverage.min()) if len(coverage) else 0, expected_keys=len(expected),
                 universe_source=universe_source, missing_answers=int(aligned.present.isna().sum()),
                 required=minimum, by_section=[{"date": str(date), "coverage": float(value)}
                                               for date, value in coverage.items()])


def execute_pipeline(config: RunConfig, run_dir: Path):
    rec = Recorder(run_dir, config.experiment_id)
    checks = []
    split = getattr(config.splits, config.evaluation_split)
    np.random.seed(config.seed)
    rec.event("data", "loading", source=config.data.source, version=config.data.version)
    bundle = load_data(config.data, start=config.splits.train.start, end=split.end)
    for event in bundle.cache_events:
        rec.event("data_cache", event["event"], **{k: v for k, v in event.items() if k != "event"})
    for event in bundle.data_events:
        rec.event("data", event["event"], level="WARNING", **{k: v for k, v in event.items() if k != "event"})
    write_json(run_dir / "data.json", bundle.metadata)
    train_targets = select_targets(bundle.labels, config.splits.train, config.data.label_horizon_minutes,
                                   sample_times=config.data.sample_times)
    eval_targets = select_targets(bundle.labels, split, config.data.label_horizon_minutes,
                                  sample_times=config.data.sample_times)
    selection = {"train": train_targets.attrs.get("selection", {}),
                 "evaluation": eval_targets.attrs.get("selection", {})}
    write_json(run_dir / "selection.json", selection)
    if train_targets.empty or eval_targets.empty:
        raise ValueError("Training and evaluation must each contain eligible targets")
    if len(train_targets) > config.budget.max_train_samples:
        raise ValueError("Training sample budget exceeded; choose a smaller declared time range")
    rec.event("data", "validated", training_samples=len(train_targets), evaluation_samples=len(eval_targets))
    rec.event("preprocessing", "started")
    train = build_batch(bundle.bars, train_targets[["date", "instrument"]], config.data)
    validation = build_batch(bundle.bars, inference_keys(bundle, split, config.data), config.data)
    train_targets = _aligned_targets(train, train_targets)
    fit_mask = train.observed.any(axis=2).sum(axis=1) >= config.data.min_history
    no_history = ~validation.observed.any(axis=(1, 2))
    fallback_mask = (validation.observed.any(axis=2).sum(axis=1) < config.data.min_history
                     if config.model.kind == "modified_tlob" else no_history)
    coverage_report = {"train": train.coverage, "evaluation": validation.coverage,
                       "training_samples_used": int(fit_mask.sum()),
                       "training_samples_excluded_insufficient_history": int((~fit_mask).sum()),
                       "no_history_prediction_samples": int(no_history.sum()),
                       "training_mean_fallback_samples": int(fallback_mask.sum()),
                       "no_history_prediction_policy": "training_mean"}
    write_json(run_dir / "coverage.json", coverage_report)
    checks += [dict(c, name="train_" + c["name"]) for c in
               batch_checks(train, train_targets, config.splits.train, config.acceptance.min_coverage)]
    checks += batch_checks(validation, eval_targets, split, config.acceptance.min_coverage)
    universe_source = "instruments" if bundle.universe is not None else "observed_bars"
    checks.append(_answer_coverage_check(inference_keys(bundle, config.splits.train, config.data), train_targets,
                  config.acceptance.min_coverage, "train_expected_universe_coverage", universe_source))
    checks.append(_answer_coverage_check(validation.keys, eval_targets, config.acceptance.min_coverage,
                                        "expected_universe_coverage", universe_source))
    checks.append(check("nonempty_training_history", fit_mask.any(), usable_samples=int(fit_mask.sum())))
    coverage_report["answers"] = {c["name"]: c["details"] for c in checks if "universe_coverage" in c["name"]}
    write_json(run_dir / "coverage.json", coverage_report)
    if not save_checks(run_dir, checks):
        raise ValueError("Data acceptance failed; see checks.json")
    missing_answers = sum(value["missing_answers"] for value in coverage_report["answers"].values())
    if no_history.any() or (~fit_mask).any() or missing_answers:
        rec.event("preprocessing", "partial_coverage_accepted", level="WARNING",
                  no_history_prediction_samples=int(no_history.sum()),
                  training_samples_excluded=int((~fit_mask).sum()),
                  missing_answers=missing_answers,
                  no_history_prediction_policy="training_mean")
    rec.event("preprocessing", "completed", resources=resource_usage())
    model = create_model(config, recorder=rec)
    if config.model.kind == "modified_tlob":
        write_json(run_dir / "device.json", {"selected": str(model.device), "requested": config.model.device})
        manifest = read_json(run_dir / "manifest.json")
        manifest["environment"]["determinism"].update(device=str(model.device),
            torch_version=str(__import__("torch").__version__), deterministic_algorithms=True,
            cpu_threads=config.model.cpu_threads, note="Tolerance-based reproducibility; GPU rounding may vary")
        write_json(run_dir / "manifest.json", manifest)
    rec.event("training", "started", seed=config.seed, samples=int(fit_mask.sum()))
    started = time.monotonic()
    fit_options = {"sample_mask": fit_mask}
    # A sealed final check must never become per-epoch training feedback.
    if config.model.kind == "modified_tlob" and config.evaluation_split == "validation":
        eval_batch = build_batch(bundle.bars, eval_targets[["date", "instrument"]], config.data)
        fit_options["validation"] = (eval_batch, _aligned_targets(eval_batch, eval_targets).target.to_numpy())
    progress = model.fit(train, train_targets.target.to_numpy(), **fit_options)
    elapsed = time.monotonic() - started
    rec.event("training", "progress", **progress, elapsed_seconds=elapsed,
              throughput_samples_per_second=int(fit_mask.sum()) / max(elapsed, 1e-9), resources=resource_usage())
    filename = model_filename(config.model.kind)
    rec.event("saving", "started", path=filename)
    model.save(run_dir / filename)
    if config.model.kind == "modified_tlob":
        model.export_text(run_dir / "modified_tlob_weights.json")
        rec.event("saving", "text_model_saved", path="modified_tlob_weights.json")
    rec.event("saving", "model_saved", path=filename)
    loaded = load_model(run_dir / filename, config, recorder=rec)
    rec.event("loading", "model_loaded", path=filename)
    checks.append(check("save_load_predictions", np.allclose(model.predict(validation), loaded.predict(validation),
                        atol=config.acceptance.repro_atol, rtol=config.acceptance.repro_rtol)))
    normalization = model.validate_training_preprocessing(train, fit_mask)
    checks.append(check("training_only_normalization", **normalization))
    if config.model.kind == "modified_tlob":
        text_model = load_model(run_dir / "modified_tlob_weights.json", config)
        checks.append(check("text_save_load_predictions", np.allclose(model.predict(validation),
                            text_model.predict(validation), atol=config.acceptance.repro_atol,
                            rtol=config.acceptance.repro_rtol)))
    rec.event("inference", "started", label_input=False)
    predictions = validation.keys.copy()
    predictions["score"] = loaded.predict(validation)
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    predictions.rename(columns={"score": "factor"}).to_parquet(run_dir / "submission.parquet", index=False)
    eval_targets.to_parquet(run_dir / "targets.parquet", index=False)
    rec.event("inference", "completed", samples=len(predictions))
    checks += prediction_checks(loaded, validation, bundle.bars, config.data,
                                config.acceptance.repro_atol, config.acceptance.repro_rtol)
    rec.event("validation", "completed", passed=all(c["passed"] for c in checks))
    if not save_checks(run_dir, checks):
        raise ValueError("Behavioral acceptance failed; see checks.json")
    rec.event("evaluation", "started", evaluation_split=config.evaluation_split)
    scored_predictions = predictions.merge(eval_targets[["date", "instrument"]], on=["date", "instrument"],
                                            how="inner", validate="one_to_one")
    scored_predictions.to_parquet(run_dir / "scored_predictions.parquet", index=False)
    metrics, daily = evaluate(scored_predictions, eval_targets, config.evaluation)
    metrics["coverage"] = {"expected_prediction_samples": len(validation.keys), "prediction_samples": len(predictions),
                           "scored_samples": len(scored_predictions),
                           "missing_answer_samples": len(predictions) - len(scored_predictions),
                           "no_history_prediction_samples": int(no_history.sum()),
                           "training_mean_fallback_samples": int(fallback_mask.sum()),
                           "no_history_prediction_policy": "training_mean"}
    expected_daily = validation.keys.assign(day=validation.keys.date.dt.normalize(), no_history=no_history).groupby("day")
    scored_daily = scored_predictions.groupby(scored_predictions.date.dt.normalize()).size()
    dates = pd.to_datetime(daily["date"])
    daily["expected_sample_count"] = dates.map(expected_daily.size()).astype(int)
    daily["prediction_sample_count"] = daily["expected_sample_count"]
    daily["missing_answer_count"] = daily["expected_sample_count"] - dates.map(scored_daily).fillna(0).astype(int)
    daily["no_history_prediction_count"] = dates.map(expected_daily.no_history.sum()).astype(int)
    write_json(run_dir / "metrics.json", metrics)
    daily.to_csv(run_dir / "daily.csv", index=False)
    rec.event("evaluation", "completed", metrics=metrics)
    manifest = read_json(run_dir / "manifest.json")
    manifest["data_version"] = bundle.metadata
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "report.txt").write_text(
        f"Run: {run_dir.name}\nHypothesis: {config.hypothesis}\n"
        f"Evaluation: {config.evaluation_split}; local approximation, not official score.\n"
        f"Checks: {len(checks)} passed.\nEvidence: engineering run only; no model promotion.\n"
        "See metrics.json, daily.csv, coverage.json and immutable source.zip.\n", encoding="utf-8")


def worker(run_dir: Path) -> int:
    config = load_config(run_dir / "config.json")
    rec = Recorder(run_dir, config.experiment_id)
    stop = threading.Event()
    def resources():
        while not stop.wait(config.logging.resource_interval_seconds):
            rec.event("resources", "sample", **resource_usage())
    monitor = threading.Thread(target=resources, daemon=True)
    monitor.start()
    try:
        execute_pipeline(config, run_dir)
        return 0
    except BaseException as exc:
        rec.failure("pipeline", exc)
        checks_path = run_dir / "checks.json"
        checks = read_json(checks_path)["checks"] if checks_path.exists() else []
        checks.append(check("pipeline_completed", False, exception_type=type(exc).__name__, reason=redact(str(exc))))
        save_checks(run_dir, checks)
        write_json(run_dir / "failure.json", {"type": type(exc).__name__, "reason": redact(str(exc))})
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        stop.set()
        monitor.join(timeout=1)


def _kill_tree(process):
    try:
        parent = psutil.Process(process.pid)
        descendants = parent.children(recursive=True)
        for child in reversed(descendants):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
    except psutil.NoSuchProcess:
        pass
    process.wait(timeout=10)


def _reproducibility(run_dir, reference, config):
    prior = read_json(reference / "manifest.json")
    current = read_json(run_dir / "manifest.json")
    if prior["status"] != "completed" or not verify_artifacts(reference)["passed"]:
        raise ValueError("Reference run is not completed or its artifacts were modified")
    if read_json(reference / "config.json") != config.snapshot():
        raise ValueError("Effective configuration changed since reference run")
    if read_json(reference / "checks.json").get("passed") is not True:
        raise ValueError("Reference acceptance checks did not pass")
    if prior["code"]["content_sha256"] != current["code"]["content_sha256"]:
        raise ValueError("Source changed since reference run; use archived source to reproduce")
    if read_json(reference / "data.json") != read_json(run_dir / "data.json"):
        raise ValueError("Data changed since reference run")
    left = pd.read_parquet(reference / "predictions.parquet").sort_values(["date", "instrument"]).reset_index(drop=True)
    right = pd.read_parquet(run_dir / "predictions.parquet").sort_values(["date", "instrument"]).reset_index(drop=True)
    keys_equal = left[["date", "instrument"]].equals(right[["date", "instrument"]])
    a, r = config.acceptance.repro_atol, config.acceptance.repro_rtol
    passed = keys_equal and np.allclose(left.score, right.score, atol=a, rtol=r)
    passed = passed and numerical_equal(read_json(reference / "metrics.json"), read_json(run_dir / "metrics.json"), a, r)
    evidence = {"passed": bool(passed), "reference_run_id": reference.name, "atol": a, "rtol": r,
                "max_abs_prediction_difference": float(np.max(np.abs(left.score - right.score))) if keys_equal else None}
    current["reproducibility"] = evidence
    write_json(run_dir / "manifest.json", current)
    checks = read_json(run_dir / "checks.json")["checks"] + [check("reproducibility", passed, **evidence_without_pass(evidence))]
    save_checks(run_dir, checks)
    return bool(passed)


def evidence_without_pass(value):
    return {k: v for k, v in value.items() if k != "passed"}


def run(config: RunConfig, *, reference: Path | None = None, policy_registration: dict | None = None):
    run_dir = new_run_dir(config.output_root)
    rec = Recorder(run_dir, config.experiment_id)
    started = time.monotonic()
    write_json(run_dir / "config.json", config.snapshot())
    (run_dir / "config.sha256").write_text(config.checksum + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "run_id": run_dir.name, "experiment_id": config.experiment_id,
                "hypothesis": config.hypothesis, "status": "running", "started_at": utc_now(),
                "config_sha256": config.checksum, "seed": config.seed, "budget": config.budget.model_dump(),
                "evaluation_split": config.evaluation_split, **(policy_registration or {})}
    write_json(run_dir / "manifest.json", manifest)
    rec.event("run", "started", config_sha256=config.checksum, budget=config.budget.model_dump())
    process = None
    peak_rss, cpu_seconds = 0.0, 0.0
    status, exit_code, error = "failed", 1, None
    try:
        manifest.update(code=capture_source(run_dir), environment=environment())
        write_json(run_dir / "manifest.json", manifest)
        if config.evaluation_split == "final":
            claim_final_access(config, run_dir.name)
            rec.event("validation", "final_access_claimed", purpose=config.final_access.purpose)
        if reference:
            rec.event("run", "rerun", reference_run_id=reference.name, reason="predeclared reproducibility validation",
                      recovery_state="new_run_no_overwrite")
        env = {**os.environ, "PYTHONHASHSEED": str(config.seed), "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        with (run_dir / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "jinyiquant", "_worker", str(run_dir)],
                                       stdout=log, stderr=log, env=env,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            while process.poll() is None:
                try:
                    p = psutil.Process(process.pid)
                    processes = [p] + p.children(recursive=True)
                    rss = sum(child.memory_info().rss for child in processes) / 2**20
                    peak_rss = max(peak_rss, rss)
                    cpu_seconds = max(cpu_seconds, sum(sum(child.cpu_times()[:2]) for child in processes))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                if time.monotonic() - started > config.budget.wall_seconds or peak_rss > config.budget.max_rss_mb:
                    _kill_tree(process)
                    status, exit_code = "budget_exceeded", 124
                    error = {"reason": "wall-time or memory budget exceeded", "recovery_state": "restart_as_new_run"}
                    rec.event("budget", "exceeded", level="ERROR", **error)
                    break
                time.sleep(0.05)
            else:
                exit_code = process.returncode
                status = "completed" if exit_code == 0 else "interrupted" if exit_code == 130 else "failed"
        if status == "completed" and reference and not _reproducibility(run_dir, reference, config):
            status, exit_code = "failed", 3
            error = {"reason": "reproducibility tolerance exceeded"}
        if status != "completed" and (run_dir / "failure.json").exists():
            error = read_json(run_dir / "failure.json")
    except BaseException as exc:
        if process is not None and process.poll() is None:
            _kill_tree(process)
        rec.failure("run", exc)
        status, exit_code = ("interrupted", 130) if isinstance(exc, KeyboardInterrupt) else ("failed", 1)
        error = {"type": type(exc).__name__, "reason": redact(str(exc))}
    if status != "completed":
        checks = read_json(run_dir / "checks.json")["checks"] if (run_dir / "checks.json").exists() else []
        checks.append(check("run_completed", False, status=status, error=error))
        save_checks(run_dir, checks)
        (run_dir / "report.txt").write_text(f"Run {run_dir.name}: {status}\nReason: {error}\n", encoding="utf-8")
    selected = read_json(run_dir / "device.json")["selected"] if (run_dir / "device.json").exists() else "cpu"
    resources = {"wall_seconds": time.monotonic() - started, "peak_rss_mb": peak_rss,
                 "cpu_seconds": cpu_seconds, "monitor_interval_seconds": .05, "gpu_used": selected == "cuda"}
    finish(run_dir, status, exit_code, resources=resources, error=error)
    return run_dir, exit_code


def failed_config(path: Path, exc: Exception):
    # Even unparseable configurations get an isolated failure archive; never persist unknown fields/secrets.
    run_dir = new_run_dir(path.resolve().parent / "runs")
    write_json(run_dir / "manifest.json", {"schema_version": 1, "run_id": run_dir.name,
               "experiment_id": "invalid-config", "started_at": utc_now(), "config_path": str(path.resolve())})
    rec = Recorder(run_dir, "invalid-config")
    rec.event("run", "started")
    rec.failure("configuration", exc)
    save_checks(run_dir, [check("valid_config", False, exception_type=type(exc).__name__)])
    finish(run_dir, "failed", 2, error={"type": type(exc).__name__, "reason": "Invalid configuration; see events.jsonl"})
    return run_dir, 2
