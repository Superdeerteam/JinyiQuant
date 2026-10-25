"""Real-sample memorization and timestamp-constrained training-label controls."""

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from pydantic import Field, model_validator

from jinyiquant.checks import batch_checks, check, prediction_checks, verify_artifacts
from jinyiquant.comparison import _block_indices
from jinyiquant.config import Period, RunConfig, StrictModel, load_config
from jinyiquant.data import KEYS, Batch, build_batch, history_diagnostics, load_data, select_targets
from jinyiquant.evaluation import evaluate
from jinyiquant.io import digest, read_json, utc_now, write_json
from jinyiquant.runner import (
    _aligned_targets,
    _answer_coverage_check,
    _kill_tree,
    inference_keys,
    save_checks,
)
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir, resource_usage

from .adapter import TLOBAdapter
from .model import parameter_groups


class SmallSample(StrictModel):
    samples: int = Field(ge=2)
    epochs: int = Field(ge=1, le=1000)
    seed: int = Field(ge=0, le=2**32 - 1)
    record_every: int = Field(ge=1)
    max_initial_mse_ratio: float = Field(gt=0, lt=1)
    max_constant_mse_ratio: float = Field(gt=0, lt=1)


class ControlPeriod(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    train: Period
    validation: Period


class NegativeControl(StrictModel):
    seeds: tuple[int, ...]
    sample_times: tuple[str, ...]
    periods: tuple[ControlPeriod, ...]
    bootstrap_samples: int = Field(ge=100)
    block_days: int = Field(ge=1)
    confidence: float = Field(gt=0, lt=1)
    bootstrap_seed: int = Field(ge=0)

    @model_validator(mode="after")
    def independent_repeats(self):
        if len(set(self.seeds)) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Require at least two unique control seeds")
        if any(seed < 0 or seed > 2**32 - 1 for seed in self.seeds):
            raise ValueError("Seeds must be uint32")
        if len(self.periods) < 2 or len({period.name for period in self.periods}) != len(self.periods):
            raise ValueError("Require at least two uniquely named research periods")
        ordered = sorted(self.periods, key=lambda period: period.train.start)
        if any(period.train.end > period.validation.start for period in ordered):
            raise ValueError("Training must end before research evaluation")
        if any(first.validation.end > second.train.start for first, second in zip(ordered, ordered[1:])):
            raise ValueError("Research periods must not overlap")
        return self


class DiagnosticBudget(StrictModel):
    wall_seconds: float = Field(gt=0)
    max_rss_mb: float = Field(gt=0)


class DiagnosticPlan(StrictModel):
    schema_version: int = Field(ge=1, le=1)
    base_config: Path
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    small_sample: SmallSample
    negative_control: NegativeControl
    budget: DiagnosticBudget
    rationale: str = Field(min_length=1)


def load_plan(path):
    path = Path(path).resolve()
    plan = DiagnosticPlan.model_validate(read_json(path))
    config = load_config(path.parent / plan.base_config)
    if not config.model.torch_backend or config.evaluation_split != "validation":
        raise ValueError("Diagnostics require a torch regression model and development validation only")
    controls = []
    for period in plan.negative_control.periods:
        snapshot = config.snapshot()
        snapshot["data"]["sample_times"] = plan.negative_control.sample_times
        snapshot["splits"].update(train=period.train.model_dump(), validation=period.validation.model_dump())
        effective = RunConfig.model_validate(snapshot)
        warm_start = max(effective.data.allowed_start,
                         pd.Timestamp(period.train.start) - pd.Timedelta(days=effective.data.history_days))
        if warm_start < config.splits.final.end and period.validation.end > config.splits.final.start:
            raise ValueError("Research data request would open the frozen final interval")
        controls.append((period.name, effective))
    return plan, config, controls


def permute_training_targets(keys, targets, seed):
    keys = keys.reset_index(drop=True)
    if keys.duplicated(KEYS).any() or keys[KEYS].isna().any().any():
        raise ValueError("Permutation requires unique non-null timestamp/instrument keys")
    targets = np.asarray(targets)
    if targets.shape != (len(keys),) or not np.isfinite(targets).all():
        raise ValueError("Permutation requires aligned finite training targets")
    source = np.arange(len(keys))
    generator = np.random.default_rng(seed)
    sections = []
    for timestamp, indices in keys.groupby("date", sort=True).indices.items():
        source[indices] = generator.permutation(indices)
        sections.append({"date": str(timestamp), "samples": len(indices),
                         "changed_targets": int(np.count_nonzero(targets[indices] != targets[source[indices]])),
                         "fixed_points": int(np.count_nonzero(indices == source[indices]))})
    if not np.array_equal(keys.date.to_numpy(), keys.date.to_numpy()[source]):
        raise AssertionError("Permutation crossed a prediction timestamp")
    mapping = keys.copy()
    mapping["source_instrument"] = keys.instrument.to_numpy()[source]
    mapping["target"] = targets
    mapping["training_target"] = targets[source]
    return targets[source].copy(), mapping, sections


def subset_batch(batch, indices):
    indices = np.asarray(indices)
    return Batch(batch.keys.iloc[indices].reset_index(drop=True), batch.x[indices].copy(),
                 batch.observed[indices].copy(), batch.history_dates[indices].copy(), dict(batch.coverage),
                 batch.history_feature_count)


def _prepare(config, destination, recorder, *, training_only=False):
    destination.mkdir(parents=True, exist_ok=False)
    end = config.splits.train.end if training_only else config.splits.validation.end
    recorder.event("data", "loading", start=str(config.splits.train.start), end=str(end), final_loaded=False)
    bundle = load_data(config.data, start=config.splits.train.start, end=end)
    for event in bundle.cache_events + bundle.data_events:
        recorder.event("data", event["event"], **{key: value for key, value in event.items() if key != "event"})
    write_json(destination / "data.json", bundle.metadata)
    training_times = config.data.training_sample_times or config.data.sample_times
    targets = select_targets(bundle.labels, config.splits.train, config.data.label_horizon_minutes,
                             sample_times=training_times)
    if not len(targets) or len(targets) > config.budget.max_train_samples:
        raise ValueError("Empty or over-budget training selection")
    train = build_batch(bundle.bars, targets[KEYS], config.data)
    targets = _aligned_targets(train, targets)
    train_mask = train.effective_rows.sum(axis=1) >= config.data.min_history
    checks = batch_checks(train, targets, config.splits.train, config.acceptance.min_coverage)
    checks.append(_answer_coverage_check(inference_keys(bundle, config.splits.train, config.data,
                                        sample_times=training_times), targets,
                  config.acceptance.min_coverage, "train_answer_coverage", "instruments"))
    history_diagnostics(train).to_parquet(destination / "train_history.parquet", index=False)
    validation, eval_targets = None, None
    if not training_only:
        eval_targets = select_targets(bundle.labels, config.splits.validation, config.data.label_horizon_minutes,
                                      sample_times=config.data.sample_times)
        validation = build_batch(bundle.bars, inference_keys(bundle, config.splits.validation, config.data), config.data)
        checks += batch_checks(validation, eval_targets, config.splits.validation, config.acceptance.min_coverage)
        checks.append(_answer_coverage_check(validation.keys, eval_targets, config.acceptance.min_coverage,
                                             "evaluation_answer_coverage", "instruments"))
        eval_targets.to_parquet(destination / "targets.parquet", index=False)
        history_diagnostics(validation).to_parquet(destination / "evaluation_history.parquet", index=False)
    checks.append(check("usable_training_samples", train_mask.any()))
    if not save_checks(destination, checks):
        raise ValueError(f"Data checks failed: {destination}")
    recorder.event("data", "validated", train_samples=len(targets), selected=int(train_mask.sum()))
    recorder.event("preprocessing", "completed", resources=resource_usage())
    return bundle, train, targets, train_mask, validation, eval_targets


def _network_scores(model, batch):
    return model._predict(batch)


def _parameter_changes(network, before):
    return {group: float(torch.stack([(parameter.detach().cpu().double() - before[name]).square().sum()
             for name, parameter in network.named_parameters() if name.split(".")[0] == group]).sum().sqrt())
            for group in parameter_groups(network)}


def _mse(predictions, targets):
    return float(np.mean(np.square(np.asarray(predictions, dtype=float) - np.asarray(targets, dtype=float))))


def _fit(config, train, target, mask, validation, eval_targets, destination, source_archive, *, tiny=None):
    run_dir = new_run_dir(destination)
    recorder = Recorder(run_dir, config.experiment_id)
    write_json(run_dir / "config.json", config.snapshot())
    write_json(run_dir / "manifest.json", {"schema_version": 1, "run_id": run_dir.name,
        "experiment_id": config.experiment_id, "status": "running", "started_at": utc_now(),
        "seed": config.seed, "code_archive": str(source_archive), "config_sha256": config.checksum})
    recorder.event("run", "started")
    started = time.monotonic()
    try:
        model = TLOBAdapter(config.model, config.data, seed=config.seed, recorder=recorder)
        before = {name: parameter.detach().cpu().double().clone()
                  for name, parameter in model.network.named_parameters()}
        trace = []
        initial = None

        def observe(adapter, epoch, steps):
            nonlocal initial
            if epoch == 0:
                initial = _network_scores(adapter, validation)
            if tiny is None:
                return
            if epoch not in (0, config.model.epochs) and epoch % tiny.record_every:
                return
            row = {"epoch": epoch, "steps": steps,
                   "mse": _mse(_network_scores(adapter, train), target),
                   "elapsed_seconds": time.monotonic() - started}
            trace.append(row)
            recorder.event("training", "fixed_batch_evaluation", **row)

        recorder.event("training", "started", samples=int(mask.sum()), optimizer="Adam", loss="raw_return_MSE")
        progress = model.fit(train, target, sample_mask=mask, observer=observe)
        if tiny:
            write_json(run_dir / "learning_curve.json", trace)
        recorder.event("training", "completed", **progress)
        changes = _parameter_changes(model.network, before)
        recorder.event("saving", "started")
        model.save(run_dir / "model.pt")
        recorder.event("saving", "completed")
        loaded = TLOBAdapter.load(run_dir / "model.pt", device=str(model.device), recorder=recorder)
        recorder.event("loading", "completed")
        recorder.event("inference", "started", label_input=False)
        scores = model.predict(validation)
        parity = np.allclose(scores, loaded.predict(validation), atol=config.acceptance.repro_atol,
                             rtol=config.acceptance.repro_rtol)
        predictions = validation.keys.assign(score=scores)
        predictions.to_parquet(run_dir / "predictions.parquet", index=False)
        validation.keys.assign(score=initial).to_parquet(run_dir / "initial_predictions.parquet", index=False)
        recorder.event("inference", "completed", samples=len(scores))
        checks = [check("save_load_predictions", parity),
                  check("parameter_groups_updated", all(value > 0 for value in changes.values()), norms=changes),
                  check("finite_predictions", np.isfinite(scores).all())]
        metrics = {"train_mse": _mse(model.predict(train)[mask], np.asarray(target)[mask]),
                   "parameter_delta_norms": changes, "training": progress,
                   "wall_seconds": time.monotonic() - started}
        recorder.event("evaluation", "started")
        if tiny:
            constant_mse = float(np.var(np.asarray(target, dtype=float)))
            initial_mse, final_mse = trace[0]["mse"], trace[-1]["mse"]
            if min(constant_mse, initial_mse) <= 0:
                raise ValueError("Tiny-batch diagnostic needs nonconstant targets and nonzero initial error")
            metrics.update(initial_mse=initial_mse, final_mse=final_mse, constant_mse=constant_mse,
                final_to_initial_ratio=final_mse / initial_mse, final_to_constant_ratio=final_mse / constant_mse,
                learning_passed=final_mse <= initial_mse * tiny.max_initial_mse_ratio
                                and final_mse <= constant_mse * tiny.max_constant_mse_ratio,
                label_statistics={"mean": float(np.mean(target)), "std": float(np.std(target)),
                                  "min": float(np.min(target)), "max": float(np.max(target))})
        else:
            scored = predictions.merge(eval_targets[KEYS], on=KEYS, validate="one_to_one")
            evaluation, daily = evaluate(scored, eval_targets, config.evaluation)
            initial_scored = validation.keys.assign(score=initial).merge(eval_targets[KEYS], on=KEYS,
                                                                         validate="one_to_one")
            _, initial_daily = evaluate(initial_scored, eval_targets, config.evaluation)
            metrics["evaluation"] = evaluation
            daily.to_csv(run_dir / "daily.csv", index=False)
            initial_daily.to_csv(run_dir / "initial_daily.csv", index=False)
        recorder.event("evaluation", "completed", metrics=metrics)
        if not save_checks(run_dir, checks):
            raise ValueError(f"Fit behavior checks failed: {run_dir}")
        write_json(run_dir / "metrics.json", metrics)
        finish(run_dir, "completed", 0, resources=resource_usage())
        return run_dir, model, metrics
    except BaseException as exc:
        recorder.failure("diagnostic_fit", exc)
        finish(run_dir, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", 1, error=str(exc))
        raise


def _small_sample(plan, config, destination, recorder):
    bundle, train, targets, _, _, _ = _prepare(config, destination / "tiny_data", recorder, training_only=True)
    full = train.observed[..., :train.history_feature_count].all(axis=2).sum(axis=1) == config.data.lookback
    eligible = np.flatnonzero(full)
    if len(eligible) < plan.small_sample.samples:
        raise ValueError("Insufficient fully observed real samples for fixed-batch diagnostic")
    indices = np.sort(np.random.default_rng(plan.small_sample.seed).choice(
        eligible, size=plan.small_sample.samples, replace=False))
    batch = subset_batch(train, indices)
    labels = targets.target.to_numpy()[indices]
    batch.keys.assign(target=labels).to_parquet(destination / "tiny_samples.parquet", index=False)
    np.savez_compressed(destination / "tiny_inputs.npz", x=batch.x, observed=batch.observed,
                        history_dates=batch.history_dates, target=labels)
    snapshot = config.snapshot()
    snapshot.update(experiment_id=plan.experiment_id, seed=plan.small_sample.seed)
    snapshot["model"].update(epochs=plan.small_sample.epochs, batch_size=plan.small_sample.samples,
                              log_every_steps=plan.small_sample.record_every)
    effective = RunConfig.model_validate(snapshot)
    path, model, metrics = _fit(effective, batch, labels, np.ones(len(labels), dtype=bool), batch, None,
                               destination / "tiny_fit", destination / "source.zip", tiny=plan.small_sample)
    repeated_path, repeated, _ = _fit(effective, batch, labels, np.ones(len(labels), dtype=bool), batch, None,
        destination / "tiny_reproduction", destination / "source.zip", tiny=plan.small_sample)
    difference = float(np.max(np.abs(model.predict(batch) - repeated.predict(batch))))
    reproducible = np.allclose(model.predict(batch), repeated.predict(batch),
                              atol=config.acceptance.repro_atol, rtol=config.acceptance.repro_rtol)
    metrics["reproduction"] = {"run": str(repeated_path), "passed": bool(reproducible),
                               "max_abs_prediction_difference": difference,
                               "atol": config.acceptance.repro_atol, "rtol": config.acceptance.repro_rtol}
    checks = prediction_checks(model, batch, bundle.bars, config.data,
                               config.acceptance.repro_atol, config.acceptance.repro_rtol)
    checks.append(check("fixed_batch_reproduction", reproducible, max_abs_difference=difference))
    write_json(destination / "tiny_behavior.json", {"passed": all(item["passed"] for item in checks),
                                                   "checks": checks})
    if not all(item["passed"] for item in checks):
        raise ValueError("Tiny-sample causal or label-free behavior failed")
    return {"run": str(path), **metrics}


def summarize_controls(rows, config):
    rng = np.random.default_rng(config.bootstrap_seed)
    results = []
    for period, section in rows.groupby("period", sort=True):
        seeds = sorted(section.seed.unique())
        dates = sorted(section.date.unique())
        if set(seeds) != set(config.seeds) or len(dates) < 2 * config.block_days:
            results.append({"period": period, "status": "inconclusive", "reason": "insufficient seeds/day blocks"})
            continue
        fields = ["rank_ic_correct", "rank_ic_shuffled", "rank_ic_initial", "ic_correct", "ic_shuffled",
                  "long_short_return_correct", "long_short_return_shuffled"]
        arrays = {field: section.pivot(index="date", columns="seed", values=field).reindex(
                  index=dates, columns=seeds).to_numpy(dtype=float) for field in fields}
        if not all(np.isfinite(array).all() for array in arrays.values()):
            results.append({"period": period, "status": "inconclusive", "reason": "undefined correlations"})
            continue
        arrays["rank_ic_correct_minus_shuffled"] = arrays["rank_ic_correct"] - arrays["rank_ic_shuffled"]
        arrays["rank_ic_shuffled_minus_initial"] = arrays["rank_ic_shuffled"] - arrays["rank_ic_initial"]
        draws = {field: [] for field in arrays}
        for _ in range(config.bootstrap_samples):
            indices = _block_indices(len(dates), config.block_days, rng)
            for field, array in arrays.items():
                draws[field].append(float(array[indices].mean()))
        alpha = (1 - config.confidence) / 2
        estimates = {field: {"mean": float(array.mean()), "interval": np.quantile(
                     draws[field], [alpha, 1 - alpha]).tolist(),
                     "seed_mean_std": float(array.mean(axis=0).std(ddof=1)),
                     "daily_seed_mean_std": float(array.mean(axis=1).std(ddof=1))}
                     for field, array in arrays.items()}
        positive_seeds = int((arrays["rank_ic_shuffled"].mean(axis=0) > 0).sum())
        results.append({"period": period, "status": "evaluated", "days": len(dates), "seeds": len(seeds),
                        "estimates": estimates, "positive_shuffled_seeds": positive_seeds,
                        "suspicious_positive_control": positive_seeds == len(seeds)
                        and estimates["rank_ic_shuffled"]["interval"][0] > 0})
    evaluated = [result for result in results if result["status"] == "evaluated"]
    repeated = len(evaluated) >= 2 and all(result["suspicious_positive_control"] for result in evaluated)
    return {"periods": results, "repeated_positive_shuffled_advantage": repeated,
            "status": "investigate" if repeated else "inconclusive" if len(evaluated) != len(config.periods)
                      else "no_repeated_positive_control_detected",
            "method": "moving consecutive trading-day blocks; shared day draws across seeds and paired arms",
            "confidence": config.confidence, "block_days": config.block_days,
            "bootstrap_samples": config.bootstrap_samples, "bootstrap_seed": config.bootstrap_seed,
            "limits": "Few dates and permutations; diagnostic flag only, not a formal permutation p-value. "
                      "No flag does not rule out leakage. Positive controls may also retain marginal/time effects."}


def execute(destination):
    destination = Path(destination)
    plan = DiagnosticPlan.model_validate(read_json(destination / "plan.json"))
    base = RunConfig.model_validate(read_json(destination / "base_config.json"))
    controls = [(item["name"], RunConfig.model_validate(item["config"]))
                for item in read_json(destination / "control_configs.json")]
    recorder = Recorder(destination, plan.experiment_id)
    recorder.event("run", "worker_started")
    resume = read_json(destination / "resume.json") if (destination / "resume.json").exists() else None
    previous = Path(resume["reference"]) if resume else None
    existing = {(item["period"], item["seed"], item["arm"]): Path(item["run"])
                for item in read_json(previous / "attempts.json")} if previous else {}
    small = read_json(previous / "small_sample.json") if previous else _small_sample(plan, base, destination, recorder)
    write_json(destination / "small_sample.json", small)
    recorder.event("validation", "small_sample_completed", learning_passed=small["learning_passed"],
                   final_to_constant_ratio=small["final_to_constant_ratio"])
    paired, attempts = [], []
    for name, config in controls:
        bundle, train, targets, mask, validation, eval_targets = _prepare(
            config, destination / (name + "_data"), recorder)
        if previous and read_json(previous / (name + "_data") / "data.json")["data_sha256"] != bundle.metadata["data_sha256"]:
            raise ValueError("Resume source data changed")
        train = subset_batch(train, np.flatnonzero(mask))
        labels = targets.target.to_numpy()[mask]
        mask = np.ones(len(labels), dtype=bool)
        for seed in plan.negative_control.seeds:
            shuffled, mapping, sections = permute_training_targets(train.keys, labels, seed)
            if not any(item["changed_targets"] for item in sections):
                raise ValueError("Permutation did not alter any training targets")
            permutation_path = destination / f"{name}_{seed}_permutation.parquet"
            mapping.to_parquet(permutation_path, index=False)
            if previous:
                original_mapping = pd.read_parquet(previous / permutation_path.name)
                if not mapping.equals(original_mapping):
                    raise ValueError("Resume permutation, keys or targets changed")
            write_json(destination / f"{name}_{seed}_permutation.json", {"seed": seed, "sections": sections,
                        "scope": "eligible training samples only, fixed once per run; evaluation untouched"})
            snapshot = config.snapshot()
            snapshot.update(seed=seed, experiment_id=plan.experiment_id)
            effective = RunConfig.model_validate(snapshot)
            arms = {}
            for arm, values in (("correct", labels), ("shuffled", shuffled)):
                path = existing.get((name, seed, arm))
                reused = path is not None
                if reused:
                    if read_json(path / "config.json") != effective.snapshot():
                        raise ValueError("Resume training configuration changed")
                    recorder.event("training", "archived_run_reused", period=name, seed=seed, arm=arm, run=str(path))
                else:
                    recorder.event("training", "control_started", period=name, seed=seed, arm=arm)
                    path, model, metrics = _fit(effective, train, values, mask, validation, eval_targets,
                        destination / f"{name}_{seed}_{arm}", destination / "source.zip")
                attempts.append({"period": name, "seed": seed, "arm": arm, "run": str(path)})
                write_json(destination / "attempts.json", attempts)
                arms[arm] = path
                if seed == plan.negative_control.seeds[0]:
                    behavior_path = destination / f"{name}_{arm}_behavior.json"
                    if reused:
                        if not read_json(previous / behavior_path.name)["passed"]:
                            raise ValueError("Archived causal behavior failed")
                        shutil.copyfile(previous / behavior_path.name, behavior_path)
                    else:
                        checks = prediction_checks(model, validation, bundle.bars, effective.data,
                            effective.acceptance.repro_atol, effective.acceptance.repro_rtol)
                        if not all(item["passed"] for item in checks):
                            raise ValueError("Control causal behavior failed")
                        write_json(behavior_path, {"checks": checks, "passed": True})
            correct = read_json(arms["correct"] / "metrics.json")["evaluation"]
            sham = read_json(arms["shuffled"] / "metrics.json")["evaluation"]
            if any(correct[key] != sham[key] for key in ("sample_keys_sha256", "targets_sha256")):
                raise AssertionError("Evaluation inputs differ between control arms")
            initial_left = pd.read_parquet(arms["correct"] / "initial_predictions.parquet")
            initial_right = pd.read_parquet(arms["shuffled"] / "initial_predictions.parquet")
            if not initial_left[KEYS].equals(initial_right[KEYS]) or not np.allclose(
                    initial_left.score, initial_right.score, atol=effective.acceptance.repro_atol,
                    rtol=effective.acceptance.repro_rtol):
                raise AssertionError("Control arms did not start from matching predictions")
            daily = pd.read_csv(arms["correct"] / "daily.csv").merge(
                pd.read_csv(arms["shuffled"] / "daily.csv"), on="date", suffixes=("_correct", "_shuffled"),
                validate="one_to_one")
            daily = daily.merge(pd.read_csv(arms["correct"] / "initial_daily.csv")[["date", "rank_ic"]].rename(
                columns={"rank_ic": "rank_ic_initial"}), on="date", validate="one_to_one")
            daily["period"], daily["seed"] = name, seed
            paired.append(daily)
            recorder.event("validation", "control_pair_completed", period=name, seed=seed,
                correct_rank_ic=correct["rank_ic_mean"], shuffled_rank_ic=sham["rank_ic_mean"])
        del bundle, train, validation
    rows = pd.concat(paired, ignore_index=True)
    rows.to_csv(destination / "paired_daily.csv", index=False)
    negative = summarize_controls(rows, plan.negative_control)
    write_json(destination / "negative_control.json", negative)
    checks = [check("artifact_integrity", verify_artifacts(Path(item["run"]))["passed"], run=item["run"])
              for item in attempts + [{"run": small["run"]}, {"run": small["reproduction"]["run"]}]]
    if not save_checks(destination, checks):
        raise ValueError("Archive integrity verification failed")
    result = {"schema_version": 1, "execution_completed": True, "small_sample": small,
              "negative_control": negative, "final_views": 0, "model_promoted": False,
              "evidence_scope": "Real development slices; raw official returns; local approximate evaluation only",
              "conclusion": "Both diagnostics are necessary clues; neither establishes model validity"}
    write_json(destination / "metrics.json", result)
    (destination / "report.txt").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_resume(reference, plan):
    reference = Path(reference).resolve()
    if not verify_artifacts(reference)["passed"] or read_json(reference / "plan.json") != plan.model_dump(mode="json"):
        raise ValueError("Resume requires intact artifacts and the exact frozen plan")
    small = read_json(reference / "small_sample.json")
    if not read_json(reference / "tiny_behavior.json")["passed"] or not small["reproduction"]["passed"]:
        raise ValueError("Resume requires completed tiny-sample behavior and reproduction checks")
    runs = [Path(item["run"]) for item in read_json(reference / "attempts.json")]
    runs += [Path(small["run"]), Path(small["reproduction"]["run"])]
    if any(read_json(path / "manifest.json")["status"] != "completed" or not verify_artifacts(path)["passed"]
           for path in runs):
        raise ValueError("Archived fit integrity/completion failed")
    project = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(reference / "source.zip") as archive:
        for name in archive.namelist():
            core = name.startswith("src/jinyiquant/") or name in {
                "modified_tlob/adapter.py", "modified_tlob/model.py", "modified_tlob/normalization.py",
                "modified_tlob/device.py", "modified_tlob/schema.py"}
            if core and name.endswith(".py") and archive.read(name) != (project / name).read_bytes():
                raise ValueError(f"Training/data/evaluation source changed: {name}")
        original = ast.parse(archive.read("modified_tlob/diagnostics.py").decode("utf-8-sig"))
        current = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        functions = {"_fit", "_prepare", "_small_sample", "permute_training_targets", "summarize_controls"}
        def extract(module):
            return {node.name: ast.dump(node) for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name in functions}
        if extract(original) != extract(current):
            raise ValueError("Diagnostic fit/selection/summary behavior changed")
    return {"reference": str(reference), "reason": "complete remaining fixed-budget diagnostics after interruption",
            "validated_complete_runs": len(runs), "plan_unchanged": True, "core_source_unchanged": True}


def run_diagnostics(plan_path, output, *, resume_from=None):
    plan, config, controls = load_plan(plan_path)
    resume = validate_resume(resume_from, plan) if resume_from else None
    if resume and read_json(Path(resume["reference"]) / "base_config.json") != config.snapshot():
        raise ValueError("Resume requires the exact frozen base configuration")
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    recorder = Recorder(destination, plan.experiment_id)
    write_json(destination / "plan.json", plan.model_dump(mode="json"))
    if resume:
        write_json(destination / "resume.json", resume)
        shutil.copyfile(Path(resume["reference"]) / "tiny_behavior.json", destination / "tiny_behavior.json")
    write_json(destination / "base_config.json", config.snapshot())
    write_json(destination / "control_configs.json", [{"name": name, "config": cfg.snapshot()}
                                                     for name, cfg in controls])
    write_json(destination / "manifest.json", {"schema_version": 1, "run_id": destination.name,
        "experiment_id": plan.experiment_id, "hypothesis": plan.rationale, "status": "running",
        "started_at": utc_now(), "plan_sha256": digest(plan.model_dump(mode="json")),
        "budget": plan.budget.model_dump(), "code": capture_source(destination), "environment": environment(),
        "references": ["https://karpathy.github.io/2019/04/25/recipe/",
                       "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html"],
        "determinism": {"algorithms": True, "device": config.model.device,
                        "cpu_threads": config.model.cpu_threads, "acceptance": config.acceptance.model_dump()}})
    recorder.event("run", "started", rationale=plan.rationale, budget=plan.budget.model_dump())
    status, exit_code, peak, error, process = "failed", 1, 0.0, None, None
    try:
        project = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONIOENCODING": "utf-8",
               "PYTHONPATH": os.pathsep.join([str(project / "src"), str(project), os.environ.get("PYTHONPATH", "")])}
        with (destination / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "modified_tlob.diagnostics", "--worker", str(destination)],
                env=env, stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            while process.poll() is None:
                try:
                    child = psutil.Process(process.pid)
                    peak = max(peak, sum(item.memory_info().rss for item in [child, *child.children(recursive=True)]) / 2**20)
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic() - started > plan.budget.wall_seconds or peak > plan.budget.max_rss_mb:
                    _kill_tree(process)
                    status, exit_code, error = "budget_exceeded", 124, "Diagnostic wall-time or memory budget exceeded"
                    recorder.event("budget", "exceeded", level="ERROR", reason=error, recovery_state="restart_as_new_run")
                    break
                time.sleep(0.1)
            else:
                exit_code = process.returncode
                status = "completed" if exit_code == 0 else "failed"
    except BaseException as exc:
        if process is not None and process.poll() is None:
            _kill_tree(process)
        recorder.failure("diagnostics", exc)
        error = str(exc)
        status, exit_code = ("interrupted", 130) if isinstance(exc, KeyboardInterrupt) else ("failed", 1)
    if exit_code and error is None:
        error = read_json(destination / "failure.json") if (destination / "failure.json").exists() else "Worker failed"
    finish(destination, status, exit_code, resources={"wall_seconds": time.monotonic() - started,
           "peak_rss_mb": peak, "monitor_interval_seconds": 0.1}, error=error)
    return destination, exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/learning_diagnostics.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            execute(args.worker)
        except BaseException as exc:
            Recorder(args.worker, read_json(args.worker / "plan.json")["experiment_id"]).failure("worker", exc)
            write_json(args.worker / "failure.json", {"type": type(exc).__name__, "reason": str(exc)})
            raise
        return 0
    if args.output is None:
        parser.error("--output is required")
    destination, code = run_diagnostics(args.plan, args.output, resume_from=args.resume_from)
    print(json.dumps({"archive": str(destination), "exit_code": code}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
