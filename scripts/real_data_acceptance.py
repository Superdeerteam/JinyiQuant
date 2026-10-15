"""Freeze and verify one real development slice; never open the final split or select a model."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.checks import check, verify_artifacts
from jinyiquant.config import load_config
from jinyiquant.data import KEYS, _read, select_targets, validate_keys
from jinyiquant.io import file_sha256, read_json, utc_now, write_json
from jinyiquant.runner import run
from jinyiquant.tracking import Recorder, capture_source, environment, finish


def _label_free(reference, destination):
    """Fresh interpreter: block answer paths/fields before entering the production inference API."""
    from jinyiquant import data
    from modified_tlob.infer import infer

    config = load_config(reference / "config.json")
    forbidden = config.data.labels_path.resolve()
    reads = []
    original_read = data._read

    def guarded_read(path, columns, *args, **kwargs):
        resolved = Path(path).resolve()
        if resolved == forbidden or resolved.is_relative_to(forbidden):
            raise AssertionError("Inference attempted to open an answer source")
        if {"target", "vwap_return", "label_end", "end_price", "vwap"}.intersection(columns):
            raise AssertionError("Inference requested answer fields")
        reads.append({"path": str(resolved), "columns": columns})
        return original_read(path, columns, *args, **kwargs)

    data._read = guarded_read

    def audit(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            resolved = Path(os.fsdecode(args[0])).resolve()
            if resolved == forbidden or resolved.is_relative_to(forbidden):
                raise AssertionError("Answer file access is forbidden")

    sys.addaudithook(audit)
    path = infer(reference / "model.pt", config.data.bars_path, destination / "independent_keys.parquet",
                 destination / "inference", device="cpu", instrument_map=config.data.instrument_map_path,
                 start_date=config.splits.validation.start, end_date=config.splits.validation.end,
                 min_coverage=config.acceptance.min_coverage)
    write_json(destination / "label_free.json", {"run": str(path), "answer_access_blocked": True,
                                                "reads": reads, "process_id": os.getpid()})


def _alignment(config, run_dir, destination):
    """Independently compare keys and original official returns without reading final answers."""
    first, last = config.splits.train.start, config.splits.validation.end
    universe, _ = _read(config.data.instruments_path, KEYS, first, last)
    universe = validate_keys(universe, "official universe")
    labels, sources = _read(config.data.labels_path, [*KEYS, "vwap_return", "label_end"], first, last)
    labels = validate_keys(labels, "official answers").rename(columns={"vwap_return": "target"})
    mapping = pd.read_csv(config.data.instrument_map_path)
    if mapping.isna().any().any() or mapping.instrument_id.duplicated().any() or mapping.instrument.duplicated().any():
        raise ValueError("Ambiguous/non-null one-to-one mapping requirement failed")
    bars, bar_sources = _read(config.data.bars_path, ["date", "instrument_id"], first, last)
    if not bars.instrument_id.isin(mapping.instrument_id).all():
        raise ValueError("Unmapped real bar IDs")
    if bars.duplicated(["date", "instrument_id"]).any():
        raise ValueError("Duplicate real bar keys")
    keys = select_targets(universe.assign(target=0.), config.splits.validation,
                          config.data.label_horizon_minutes, sample_times=config.data.sample_times)[KEYS]
    keys.to_parquet(destination / "independent_keys.parquet", index=False)
    actual = pd.read_parquet(run_dir / "predictions.parquet")
    targets = pd.read_parquet(run_dir / "targets.parquet")
    selected = select_targets(labels, config.splits.validation, config.data.label_horizon_minutes,
                              sample_times=config.data.sample_times)
    selected = selected.merge(keys, on=KEYS, how="inner", validate="one_to_one")
    joined = targets.merge(selected[[*KEYS, "target"]], on=KEYS, how="outer", suffixes=("", "_official"),
                           validate="one_to_one", indicator=True)
    aligned = joined._merge.eq("both").all() and np.array_equal(joined.target, joined.target_official)
    evidence = {"mapping_rows": len(mapping), "mapping_sha256": file_sha256(config.data.instrument_map_path),
                "mapping_provenance": "user-supplied explicit mapping; cross-file checks, no inferred security IDs",
                "bar_rows": len(bars), "bar_sources": bar_sources, "answer_sources": sources,
                "raw_answers": len(labels), "universe_rows": len(universe), "expected_eval_keys": len(keys),
                "scored_eval_keys": len(targets), "coverage": read_json(run_dir / "coverage.json"),
                "duplicate_keys": 0, "unmapped_bar_ids": 0,
                "prediction_keys_equal": actual[KEYS].equals(keys), "official_values_unchanged": bool(aligned)}
    write_json(destination / "alignment.json", evidence)
    return evidence


def acceptance(config_path, destination):
    config = load_config(config_path)
    if config.evaluation_split != "validation" or config.model.kind != "modified_tlob":
        raise ValueError("This engineering acceptance only permits TLOB research validation")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rec = Recorder(destination, config.experiment_id)
    write_json(destination / "manifest.json", {"schema_version": 1, "run_id": destination.name,
               "experiment_id": config.experiment_id, "status": "running", "started_at": utc_now()})
    rec.event("run", "started")
    code = capture_source(destination)
    write_json(destination / "freeze.json", {"schema_version": 1, "version": config.experiment_id,
        "frozen_at_utc": utc_now(), "code": code, "config": config.snapshot(), "config_sha256": config.checksum,
        "environment": environment(), "budget_seconds_total": 900,
        "predeclared_rules": {"data_and_keys": "identical source checksums and exact keys",
            "prediction_and_metrics": config.acceptance.model_dump(),
            "history": "report each key before filling, including short/all-missing/stale windows; no full-history claim",
            "final": "unopened; regression test requires load after checkpoint and no epoch feedback",
            "quality": "no improvement threshold, no promotion, local approximate metrics only"},
        "public_reference": "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage"})
    attempts, checks = [], []
    status, exit_code = "failed", 1
    try:
        for reference in (None, "first"):
            path, result = run(config, reference=Path(attempts[0]) if reference else None)
            attempts.append(str(path))
            write_json(destination / "attempts.json", attempts)
            if result:
                raise RuntimeError(f"Training/reproduction failed, exit {result}: {path}")
            checks.append(check("run_artifacts_" + str(len(attempts)), verify_artifacts(path)["passed"]))
            rec.event("validation", "training_run_verified", run=str(path), rerun=bool(reference))
        first, second = map(Path, attempts)
        alignment = _alignment(config, first, destination)
        checks.append(check("official_alignment", alignment["prediction_keys_equal"]
                            and alignment["official_values_unchanged"]))
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(__file__).resolve().parents[1] / "src"),
                                                        str(Path(__file__).resolve().parents[1])])}
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--label-free-reference", str(first),
                                "--output", str(destination)], env=env, capture_output=True, text=True,
                               encoding="utf-8", timeout=300)
        (destination / "label_free.log").write_text(child.stdout + child.stderr, encoding="utf-8")
        if child.returncode:
            raise RuntimeError("Independent label-free inference failed; see label_free.log")
        label_free = read_json(destination / "label_free.json")
        actual = pd.read_parquet(Path(label_free["run"]) / "predictions.parquet")
        expected = pd.read_parquet(first / "predictions.parquet")
        delta = float(np.max(np.abs(actual.factor.to_numpy() - expected.score.to_numpy())))
        checks.append(check("label_free_process", actual[KEYS].equals(expected[KEYS]) and
                            np.allclose(actual.factor, expected.score, atol=config.acceptance.repro_atol,
                                        rtol=config.acceptance.repro_rtol), max_abs_difference=delta,
                            answer_access_blocked=label_free["answer_access_blocked"]))
        checks.append(check("inference_artifacts", verify_artifacts(Path(label_free["run"]))["passed"]))
        reproducibility = read_json(second / "manifest.json")["reproducibility"]
        checks.append(check("same_config_rerun", reproducibility["passed"], evidence=reproducibility))
        for index, path in enumerate((first, second)):
            data = read_json(path / "data.json")
            checks.append(check("final_unopened_" + str(index),
                                pd.Timestamp(data["requested_end"]) <= config.splits.final.start))
            checks.append(check("behavior_checks_" + str(index), read_json(path / "checks.json")["passed"]))
        checks.append(check("total_budget", time.monotonic() - started <= 900, budget_seconds=900))
        passed = all(c["passed"] for c in checks)
        status, exit_code = ("completed", 0) if passed else ("failed", 1)
        report = {"schema_version": 1, "version": config.experiment_id, "passed": passed, "checks": checks,
                  "runs": attempts, "label_free_run": label_free["run"], "reproducibility": reproducibility,
                  "resources": [read_json(p / "manifest.json")["resources"] for p in (first, second)],
                  "history_quality": {name: alignment["coverage"][name]["history_quality"]
                                      for name in ("train", "evaluation")},
                  "wall_seconds": time.monotonic() - started, "final_views": 0,
                  "gpu": "CPU only; GPU not used or accepted", "platform": "not tested",
                  "conclusion": "Real-slice engineering only; no alpha/quality improvement claim or model promotion"}
    except BaseException as exc:
        rec.failure("acceptance", exc)
        report = {"schema_version": 1, "version": config.experiment_id, "passed": False, "runs": attempts,
                  "checks": checks, "reason": str(exc), "exception_type": type(exc).__name__,
                  "wall_seconds": time.monotonic() - started}
        if isinstance(exc, KeyboardInterrupt):
            exit_code = 130
    write_json(destination / "acceptance.json", report)
    finish(destination, status, exit_code, resources={"wall_seconds": time.monotonic() - started})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/frozen_tlob_real_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label-free-reference", type=Path)
    args = parser.parse_args()
    if args.label_free_reference:
        _label_free(args.label_free_reference, args.output)
    else:
        result = acceptance(args.config, args.output)
        print(json.dumps({"passed": result["passed"], "archive": str(args.output.resolve())}))
        raise SystemExit(0 if result["passed"] else 1)
