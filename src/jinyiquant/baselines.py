"""Frozen forward-date Mean/Ridge/TLOB baselines, with paired day-block uncertainty.

Run in the optional TLOB environment: python -m jinyiquant.baselines --plan ... --output ...
No hyperparameter search, threshold selection, final access or model promotion.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from pydantic import Field, model_validator

from .checks import check, verify_artifacts
from .comparison import _block_indices
from .config import Period, RunConfig, StrictModel, load_config
from .evaluation import EVALUATION_VERSION
from .io import digest, file_sha256, read_json, utc_now, write_json
from .runner import _kill_tree, require_contract_metrics, run, save_checks
from .tracking import Recorder, capture_source, environment, finish

MODELS = ("mean", "ridge", "modified_tlob")
KEYS = ["date", "instrument"]
SUMMARY_METRICS = ("mse", "mae", "ic_mean", "rank_ic_mean", "long_return_mean", "short_return_mean",
                   "long_short_return_mean", "turnover_mean", "score_change_mae", "rank_persistence_mean")


class ResearchPeriod(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    train: Period
    validation: Period


class BaselinePlan(StrictModel):
    schema_version: int = Field(ge=1, le=1)
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    base_config: Path
    pilot_run: Path
    seeds: tuple[int, ...]
    periods: tuple[ResearchPeriod, ...]
    total_wall_seconds: float = Field(gt=0)
    per_run_wall_seconds: float = Field(gt=0)
    max_rss_mb: float = Field(gt=0)
    max_total_rss_mb: float = Field(gt=0)
    parallel_runs: int = Field(default=1, ge=1, le=2)
    max_train_samples: int = Field(ge=1)
    block_days: int = Field(ge=1)
    bootstrap_samples: int = Field(ge=100)
    confidence: float = Field(gt=0, lt=1)
    bootstrap_seed: int = Field(ge=0)
    reproduce_period: str
    rationale: str = Field(min_length=1)
    speed_estimate: dict

    @model_validator(mode="after")
    def independent_windows(self):
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("At least three distinct seeds are required")
        if any(seed < 0 or seed > 2**32 - 1 for seed in self.seeds):
            raise ValueError("Seeds must be uint32")
        if len(self.periods) < 2 or len({period.name for period in self.periods}) != len(self.periods):
            raise ValueError("At least two uniquely named research periods are required")
        ordered = sorted(self.periods, key=lambda period: period.validation.start)
        if any(p.train.end > p.validation.start for p in ordered):
            raise ValueError("Training labels must precede validation")
        if any(a.validation.end > b.validation.start for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Research validation periods must not overlap")
        if self.reproduce_period not in {p.name for p in self.periods}:
            raise ValueError("Reproduction must use a declared research period")
        return self


def load_plan(path):
    path = Path(path).resolve()
    raw = read_json(path)
    for field in ("base_config", "pilot_run"):
        raw[field] = str((path.parent / raw[field]).resolve())
    plan = BaselinePlan.model_validate(raw)
    config = load_config(plan.base_config)
    if config.model.kind != "modified_tlob" or config.evaluation_split != "validation":
        raise ValueError("Baseline source must be current TLOB, research validation only")
    for period in plan.periods:
        effective = variant(config, plan, period, plan.seeds[0], "modified_tlob", config.output_root)
        start = max(effective.data.allowed_start,
                    pd.Timestamp(period.train.start) - pd.Timedelta(days=effective.data.history_days))
        if start < config.splits.final.end and period.validation.end > config.splits.final.start:
            raise ValueError("Data request including warmup would open the frozen final interval")
    return plan, config


def variant(base, plan, period, seed, model, output):
    raw = base.snapshot()
    raw.update(seed=seed, experiment_id=plan.experiment_id, hypothesis=plan.rationale,
               evaluation_split="validation", output_root=str(output))
    raw["splits"].update(train=period.train.model_dump(mode="json"),
                         validation=period.validation.model_dump(mode="json"))
    raw["model"]["kind"] = model
    raw["budget"].update(wall_seconds=plan.per_run_wall_seconds, max_rss_mb=plan.max_rss_mb,
                          max_train_samples=plan.max_train_samples)
    raw["logging"]["learning_curves"] = True
    raw["evaluation"]["constant_portfolios"] = "undefined"
    return RunConfig.model_validate(raw)


def measure_pilot(path, base):
    path = Path(path)
    if read_json(path / "manifest.json")["status"] != "completed" or not verify_artifacts(path)["passed"]:
        raise ValueError("Speed pilot must have completed with intact artifacts")
    cfg = read_json(path / "config.json")
    if cfg["model"] != base.snapshot()["model"] or cfg["data"] != base.snapshot()["data"]:
        raise ValueError("Pilot must use the current TLOB architecture, optimizer, data recipe and device")
    if cfg["evaluation_split"] != "validation" or cfg["splits"]["final"] != base.snapshot()["splits"]["final"]:
        raise ValueError("Pilot must preserve the sealed final interval")
    events = [json.loads(line) for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    training = [e["context"] for e in events if e["stage"] == "training" and e["event_type"] == "progress"
                and "optimizer" in e["context"]][-1]
    manifest = read_json(path / "manifest.json")
    return {"run": str(path.resolve()), "manifest_sha256": file_sha256(path / "manifest.json"),
            "training_samples": training["samples"], "epochs": cfg["model"]["epochs"],
            "training_including_validation_seconds": training["elapsed_seconds"],
            "effective_training_samples_per_second": training["samples"] * cfg["model"]["epochs"] /
                                                     training["elapsed_seconds"],
            "pipeline_wall_seconds": manifest["resources"]["wall_seconds"],
            "peak_rss_mb": manifest["resources"]["peak_rss_mb"],
            "note": "Short CPU measurement; includes validation overhead; extrapolation is approximate"}


def coverage_rows(path, config):
    rows = []
    checks = read_json(path / "checks.json")["checks"]
    for split, filename in (("train", "train_history.parquet"), ("validation", "evaluation_history.parquet")):
        quality = pd.read_parquet(path / filename)
        quality["insufficient_history"] = quality.effective_history_length < config.data.min_history
        quality["incomplete_window"] = ~quality.full_observed_window
        quality["any_missing"] = quality.missing_fraction > 0
        quality["no_history"] = quality.effective_history_length == 0
        name = "train_expected_universe_coverage" if split == "train" else "expected_universe_coverage"
        answer = next(item["details"] for item in checks if item["name"] == name)
        answer_by_section = {pd.Timestamp(item["date"]): item["coverage"] for item in answer["by_section"]}
        for date, group in quality.groupby("date", sort=True):
            rows.append({"split": split, "timestamp": date.isoformat(), "date": date.strftime("%Y-%m-%d"),
                "history_sample_count": len(group), "missing_answer_fraction": 1 - answer_by_section[date],
                "insufficient_history_fraction": float(group.insufficient_history.mean()),
                "incomplete_window_fraction": float(group.incomplete_window.mean()),
                "no_history_fraction": float(group.no_history.mean()),
                "any_missing_feature_fraction": float(group.any_missing.mean()),
                "missing_feature_slot_fraction": float(group.missing_fraction.mean()),
                "source_missing_slot_fraction": float(group.source_missing_fraction.mean()),
                "history_population": "labeled training samples" if split == "train" else "independent full prediction universe"})
    return rows


def paired_estimates(rows, plan):
    results = []
    for period in plan.periods:
        data = rows[rows.period == period.name]
        for reference in ("mean", "ridge"):
            for metric in ("mse", "ic", "rank_ic", "long_short_return", "turnover"):
                left = data[data.model == "modified_tlob"][["seed", "date", metric]]
                right = data[data.model == reference][["seed", "date", metric]]
                paired = left.merge(right, on=["seed", "date"], how="outer", validate="one_to_one",
                                    suffixes=("_tlob", "_reference"), indicator=True)
                item = {"period": period.name, "reference": reference, "metric": metric,
                        "difference": "TLOB minus reference", "status": "undefined"}
                expected_days = data[data.model == "modified_tlob"].date.nunique()
                if (paired._merge != "both").any() or len(paired) != expected_days * len(plan.seeds):
                    raise ValueError("Incomplete paired date/seed grid")
                if expected_days < max(2, plan.block_days):
                    item.update(status="insufficient_days", reason="Too few independent dates for declared day blocks")
                elif not np.isfinite(paired[[metric + "_tlob", metric + "_reference"]].to_numpy()).all():
                    item["reason"] = "Undefined observations retained; constant reference cannot rank stocks"
                else:
                    paired["delta"] = paired[metric + "_tlob"] - paired[metric + "_reference"]
                    day = paired.groupby("date", sort=True).delta.mean().to_numpy()
                    seed = paired.groupby("seed").delta.mean().to_numpy()
                    rng = np.random.default_rng(plan.bootstrap_seed)
                    draws = [float(day[_block_indices(len(day), plan.block_days, rng)].mean())
                             for _ in range(plan.bootstrap_samples)]
                    alpha = (1 - plan.confidence) / 2
                    item.update(status="estimated", mean=float(day.mean()),
                                interval=np.quantile(draws, [alpha, 1-alpha]).tolist(), days=len(day),
                                seed_mean_std=float(seed.std(ddof=1)))
                results.append(item)
    return {"comparisons": results, "evaluation_version": EVALUATION_VERSION,
            "confidence": plan.confidence, "block_days": plan.block_days,
            "bootstrap_samples": plan.bootstrap_samples, "bootstrap_seed": plan.bootstrap_seed,
            "method": "TLOB-reference paired by date/seed; average seeds within day, resample consecutive day blocks",
            "limits": "Few dates, same-market stocks and overlapping return horizons; seed repetitions are not independent days. "
                      "No promotion or practical improvement threshold is inferred from these baseline observations."}


def summarize(destination, plan):
    attempts = read_json(destination / "attempts.json")
    daily_rows, timestamp_rows, coverage, summary, checks = [], [], [], [], []
    comparison = {}
    for item in attempts:
        path = Path(item["run"])
        config = RunConfig.model_validate(read_json(path / "config.json"))
        metrics = require_contract_metrics(read_json(path / "metrics.json"))
        manifest = read_json(path / "manifest.json")
        metadata = {field: item[field] for field in ("period", "seed", "model")}
        daily_rows.append(pd.read_csv(path / "daily.csv").assign(**metadata))
        timestamp_rows.append(pd.read_csv(path / "by_timestamp.csv").assign(**metadata))
        coverage.extend([{**row, **metadata} for row in coverage_rows(path, config)])
        summary.append({**metadata, **{field: metrics[field] for field in SUMMARY_METRICS},
                        "evaluation_version": metrics["evaluation_version"],
                        "economic_metrics": metrics.get("economic_metrics"),
                        "resources": manifest["resources"], "run": str(path),
                        "training_samples": read_json(path / "coverage.json")["training_samples_used"]})
        checks.append(check("run_integrity", manifest["status"] == "completed" and verify_artifacts(path)["passed"], **metadata))
        train = pd.read_parquet(path / "train_history.parquet")
        signature = {field: config.snapshot()[field] for field in ("data", "splits", "budget", "evaluation", "acceptance")}
        signature.update(data_snapshot=read_json(path / "data.json"), sample_keys=metrics["sample_keys_sha256"],
                         targets=metrics["targets_sha256"],
                         code=manifest["code"]["content_sha256"],
                         training_keys=digest(train[KEYS].assign(date=train.date.astype(str)).to_dict("records")),
                         training_samples=summary[-1]["training_samples"])
        if item["period"] in comparison:
            checks.append(check("fair_data_samples_budget", comparison[item["period"]] == signature, **metadata))
        else:
            comparison[item["period"]] = signature
        if item["model"] == "mean":
            checks.append(check("constant_ranking_is_undefined", metrics["ic_mean"] is None
                and metrics["rank_ic_mean"] is None and metrics["long_short_return_mean"] is None
                and metrics["ranking_defined_section_count"] == 0, **metadata))
    rows = pd.concat(daily_rows, ignore_index=True)
    rows.to_csv(destination / "daily.csv", index=False)
    pd.concat(timestamp_rows, ignore_index=True).to_csv(destination / "by_timestamp.csv", index=False)
    pd.DataFrame(coverage).to_csv(destination / "coverage_by_timestamp.csv", index=False)
    uncertainty = paired_estimates(rows, plan)
    write_json(destination / "paired_comparison.json", uncertainty)
    write_json(destination / "summary.json", summary)
    pd.DataFrame([{**{key: value for key, value in row.items()
                      if key not in ("resources", "economic_metrics")},
                   **row["resources"]} for row in summary]).to_csv(destination / "summary.csv", index=False)
    expected = {(p.name, s, m) for p in plan.periods for s in plan.seeds for m in MODELS}
    checks.append(check("complete_grid", {(r["period"], r["seed"], r["model"]) for r in attempts} == expected
                        and len(attempts) == len(expected)))
    reproduction = read_json(destination / "reproduction.json")
    checks.append(check("same_seed_reproduction", reproduction["passed"]))
    if not save_checks(destination, checks):
        raise ValueError("Baseline suite acceptance failed")
    variation = []
    for period in plan.periods:
        for model in MODELS:
            cells = [row for row in summary if row["period"] == period.name and row["model"] == model]
            variation.append({"period": period.name, "model": model,
                "metrics": {field: {"mean": float(np.mean(v)) if v else None,
                                     "seed_std": float(np.std(v, ddof=1)) if len(v) > 1 else None}
                            for field in SUMMARY_METRICS for v in [[row[field] for row in cells if row[field] is not None]]}})
    result = {"schema_version": 1, "execution_completed": True, "final_views": 0, "model_promoted": False,
              "evaluation_version": EVALUATION_VERSION,
              "economic_metrics": {row["run"]: row.pop("economic_metrics") for row in summary},
              "grid_runs": len(attempts), "variation": variation,
              "scope": "Forward dates within development data; local approximate metrics; no official score",
              "selection": "Current TLOB remains the comparison reference; no hyperparameter or checkpoint selection",
              "deterministic_references": "Mean/Ridge ignore seed; their repetitions measure reproducibility, not independent evidence"}
    ratios = []
    for period in plan.periods:
        for seed in plan.seeds:
            cells = {row["model"]: row["mse"] for row in summary
                     if row["period"] == period.name and row["seed"] == seed}
            ratios.append({"period": period.name, "seed": seed,
                           "tlob_to_mean_mse_ratio": cells["modified_tlob"] / cells["mean"] if cells["mean"] > 0 else None})
    result["mse_vs_mean"] = ratios
    from .research_plots import plot_baseline_summary
    plot_baseline_summary(variation, summary, destination / "baseline_overview.png")
    write_json(destination / "metrics.json", result)
    (destination / "report.txt").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def execute(destination):
    destination = Path(destination)
    plan = BaselinePlan.model_validate(read_json(destination / "plan.json"))
    base = RunConfig.model_validate(read_json(destination / "base_config.json"))
    rec = Recorder(destination, plan.experiment_id)
    attempts = []
    cells = iter((p, s, m) for p in plan.periods for s in plan.seeds for m in MODELS)
    pending = {}
    with ThreadPoolExecutor(max_workers=plan.parallel_runs) as executor:
        def submit_next():
            cell = next(cells, None)
            if cell is None:
                return
            period, seed, model = cell
            config = variant(base, plan, period, seed, model, destination / "fits")
            rec.event("training", "cell_started", period=period.name, seed=seed, model=model)
            pending[executor.submit(run, config)] = cell
        for _ in range(plan.parallel_runs):
            submit_next()
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                period, seed, model = pending.pop(future)
                path, code = future.result()
                attempts.append({"period": period.name, "seed": seed, "model": model, "run": str(path), "exit_code": code})
                write_json(destination / "attempts.json", attempts)
                if code:
                    raise RuntimeError(f"Baseline cell failed ({code}): {path}")
                rec.event("evaluation", "cell_completed", **attempts[-1])
            for _ in completed:
                submit_next()
    original = next(row for row in attempts if row["period"] == plan.reproduce_period
                    and row["seed"] == plan.seeds[0] and row["model"] == "modified_tlob")
    config = RunConfig.model_validate(read_json(Path(original["run"]) / "config.json"))
    rec.event("validation", "reproduction_started", reference=original["run"], purpose="predeclared tolerance check")
    repeated, code = run(config, reference=Path(original["run"]))
    write_json(destination / "reproduction.json", {"reference": original["run"], "run": str(repeated), "passed": code == 0})
    if code:
        raise RuntimeError(f"Reproduction failed: {repeated}")
    summarize(destination, plan)


def run_suite(plan_path, output):
    plan, base = load_plan(plan_path)
    speed = measure_pilot(plan.pilot_run, base)
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    rec = Recorder(destination, plan.experiment_id)
    write_json(destination / "plan.json", plan.model_dump(mode="json"))
    write_json(destination / "base_config.json", base.snapshot())
    write_json(destination / "pilot_measurement.json", speed)
    write_json(destination / "manifest.json", {"schema_version": 1, "run_id": destination.name,
        "experiment_id": plan.experiment_id, "hypothesis": plan.rationale, "status": "running", "started_at": utc_now(),
        "plan_sha256": digest(plan.model_dump(mode="json")), "code": capture_source(destination), "environment": environment(),
        "budget": {"wall_seconds": plan.total_wall_seconds, "max_rss_mb": plan.max_total_rss_mb,
                   "parallel_runs": plan.parallel_runs, "per_fit_max_rss_mb": plan.max_rss_mb},
        "references": ["https://github.com/LeonardoBerti00/TLOB",
                       "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html"],
        "determinism": {"algorithms": True, "device": base.model.device, "cpu_threads": base.model.cpu_threads,
                        "acceptance": base.acceptance.model_dump()}})
    rec.event("run", "started", budget_seconds=plan.total_wall_seconds, final_access=False)
    started, peak, process = time.monotonic(), 0., None
    status, code, error = "failed", 1, None
    try:
        project = Path(__file__).resolve().parents[2]
        env = {**os.environ, "PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OMP_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONIOENCODING": "utf-8",
               "PYTHONPATH": os.pathsep.join([str(project / "src"), str(project), os.environ.get("PYTHONPATH", "")])}
        with (destination / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "jinyiquant.baselines", "--worker", str(destination)],
                env=env, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            last_log = 0.
            while process.poll() is None:
                try:
                    child = psutil.Process(process.pid)
                    processes = [child, *child.children(recursive=True)]
                    rss = sum(p.memory_info().rss for p in processes) / 2**20
                    peak = max(peak, rss)
                    if time.monotonic() - last_log >= base.logging.resource_interval_seconds:
                        rec.event("resources", "sample", rss_mb=rss,
                                  cpu_seconds=sum(sum(p.cpu_times()[:2]) for p in processes))
                        last_log = time.monotonic()
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic() - started > plan.total_wall_seconds or peak > plan.max_total_rss_mb:
                    _kill_tree(process)
                    status, code, error = "budget_exceeded", 124, "Baseline suite time or memory budget exceeded"
                    rec.event("budget", "exceeded", level="ERROR", reason=error, recovery_state="new_run_required")
                    break
                time.sleep(.1)
            else:
                code = process.returncode
                status = "completed" if code == 0 else "failed"
    except BaseException as exc:
        if process is not None and process.poll() is None:
            _kill_tree(process)
        rec.failure("suite", exc)
        error = str(exc)
        status, code = ("interrupted", 130) if isinstance(exc, KeyboardInterrupt) else ("failed", 1)
    if code and error is None:
        error = read_json(destination / "failure.json") if (destination / "failure.json").exists() else "Worker failed"
    finish(destination, status, code, resources={"wall_seconds": time.monotonic()-started,
           "peak_rss_mb": peak, "monitor_interval_seconds": .1}, error=error)
    return destination, code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
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
    if args.plan is None or args.output is None:
        parser.error("--plan and --output are required")
    path, code = run_suite(args.plan, args.output)
    print(json.dumps({"archive": str(path), "exit_code": code}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
