"""Local inference from saved weights and market/key files; never trains or reads labels."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Support both `python -m modified_tlob.infer` and direct execution of this file.
if __package__ in {None, ""}:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    if (root / "src").is_dir():
        sys.path.insert(0, str(root / "src"))

import numpy as np
import pandas as pd

from jinyiquant.checks import check
from jinyiquant.data import build_batch, history_diagnostics, load_data
from jinyiquant.io import file_sha256, read_json, utc_now, write_json
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir
from modified_tlob.adapter import TLOBAdapter
from modified_tlob.device import device_resources


def infer(weights, bars, keys, output_root, *, device="auto", price_scale=None,
          instrument_map=None, start_date=None, end_date=None, min_coverage=.6):
    """Write an isolated inference archive and return its path. Dates are [start, end)."""
    run_dir = new_run_dir(Path(output_root).resolve())
    rec = Recorder(run_dir, "modified-tlob-local-inference")
    started = time.monotonic()
    write_json(run_dir / "manifest.json", {"schema_version": 1, "run_id": run_dir.name,
        "experiment_id": rec.experiment_id, "started_at": utc_now(), "status": "running",
        "scope": "local_inference_only"})
    rec.event("run", "started", label_input=False, training=False)
    try:
        if not 0 < min_coverage <= 1:
            raise ValueError("min_coverage must be in (0, 1]")
        first = None if start_date is None else pd.Timestamp(start_date)
        last = None if end_date is None else pd.Timestamp(end_date)
        if any(t is not None and (pd.isna(t) or t.tzinfo) for t in (first, last)):
            raise ValueError("Dates must be naive exchange-local timestamps")
        if first is not None and last is not None and first >= last:
            raise ValueError("Require start_date < end_date (half-open)")
        weights = Path(weights).resolve()
        rec.event("loading", "started", path=str(weights))
        model = TLOBAdapter.load(weights, device=device, recorder=rec)
        rec.event("loading", "model_loaded", device=str(model.device), training=False)
        preprocessing = dict(model.preprocessing)
        if price_scale is not None:
            if not 0 < price_scale < float("inf"):
                raise ValueError("price_scale must be positive and finite")
            preprocessing["price_scale"] = price_scale
        config = {**preprocessing, "bars_path": str(Path(bars).resolve()), "labels_path": None,
                  "instruments_path": str(Path(keys).resolve()),
                  "instrument_map_path": None if instrument_map is None else str(Path(instrument_map).resolve()),
                  "source": "local_inference_market_and_keys_only", "version": "1"}
        write_json(run_dir / "config.json", {"schema_version": 1, "weights": str(weights),
            "data": config, "device": device, "selected_device": str(model.device),
            "start_date": None if first is None else str(first), "end_date": None if last is None else str(last),
            "min_coverage": min_coverage, "training": False, "label_input": False})
        manifest = read_json(run_dir / "manifest.json")
        manifest.update(code=capture_source(run_dir), environment=environment(), model_sha256=file_sha256(weights))
        manifest["environment"]["determinism"].update(device=str(model.device),
            cpu_threads=model.config["cpu_threads"], deterministic_algorithms=True,
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
        write_json(run_dir / "manifest.json", manifest)
        rec.event("data", "loading", label_input=False)
        bundle = load_data(config, start=first, end=last)
        write_json(run_dir / "data.json", bundle.metadata)
        prediction_keys = bundle.universe
        if prediction_keys is None or prediction_keys.empty:
            raise ValueError("Prediction keys must be nonempty")
        rec.event("data", "validated", rows=len(bundle.bars), prediction_keys=len(prediction_keys))
        rec.event("inference", "started", training=False, label_input=False)
        predictions, coverage, history = [], [], []
        for date, section in prediction_keys.groupby("date", sort=True):
            batch = build_batch(bundle.bars, section, config)
            history.append(history_diagnostics(batch))
            coverage.append(batch.coverage)
            write_json(run_dir / "coverage.json", coverage)
            if batch.coverage["min_section_coverage"] < min_coverage:
                raise ValueError(f"Insufficient history coverage at {date}")
            rec.event("preprocessing", "completed", date=str(date), coverage=batch.coverage)
            predictions.append(batch.keys.assign(factor=model.predict(batch)))
        result = pd.concat(predictions, ignore_index=True)[["date", "instrument", "factor"]]
        if not result[["date", "instrument"]].equals(prediction_keys.reset_index(drop=True)[["date", "instrument"]]):
            raise ValueError("Prediction keys do not match requested universe")
        checks = [check("finite_predictions", np.isfinite(result.factor).all()),
                  check("exact_prediction_keys", result[["date", "instrument"]].equals(
                      prediction_keys.reset_index(drop=True)[["date", "instrument"]])),
                  check("history_coverage", all(c["min_section_coverage"] >= min_coverage for c in coverage)),
                  check("label_free_inputs", config["labels_path"] is None)]
        passed = all(c["passed"] for c in checks)
        write_json(run_dir / "checks.json", {"schema_version": 1, "passed": passed, "checks": checks})
        if not passed:
            raise ValueError("Inference acceptance checks failed")
        result.to_parquet(run_dir / "predictions.parquet", index=False)
        pd.concat(history, ignore_index=True).to_parquet(run_dir / "evaluation_history.parquet", index=False)
        rec.event("saving", "predictions_saved", path="predictions.parquet", samples=len(result))
        rec.event("inference", "completed", samples=len(result), resources=device_resources(model.device))
        finish(run_dir, "completed", 0, resources={"wall_seconds": time.monotonic() - started,
                                                  **device_resources(model.device)})
        return run_dir
    except BaseException as exc:
        rec.failure("inference", exc)
        finish(run_dir, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
               130 if isinstance(exc, KeyboardInterrupt) else 1,
               resources={"wall_seconds": time.monotonic() - started},
               error={"type": type(exc).__name__, "reason": str(exc)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="Saved model.pt or JSON text weights")
    parser.add_argument("--bars", required=True, type=Path, help="Raw market file/directory")
    parser.add_argument("--keys", required=True, type=Path, help="Prediction date/instrument table")
    parser.add_argument("--output-root", required=True, type=Path, help="Parent for unique inference archives")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--price-scale", type=float, help="Source price unit divisor; defaults to saved value")
    parser.add_argument("--instrument-map", type=Path, help="Explicit mapping if market uses instrument_id")
    parser.add_argument("--start-date", help="Inclusive prediction start, Asia/Shanghai naive")
    parser.add_argument("--end-date", help="Exclusive prediction end")
    parser.add_argument("--min-coverage", type=float, default=.6)
    args = parser.parse_args(argv)
    try:
        path = infer(**vars(args))
    except (ValueError, OSError, RuntimeError) as exc:
        from jinyiquant.io import redact
        print(json.dumps({"exit_code": 1, "error_type": type(exc).__name__, "reason": redact(str(exc))}))
        return 1
    print(json.dumps({"exit_code": 0, "output_dir": str(path), "predictions": str(path / "predictions.parquet")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
