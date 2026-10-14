"""Platform inference: historical bars and point-in-time pool, never answers or training."""

import hashlib
import re
import time
from pathlib import Path

import pandas as pd

from jinyiquant.data import build_batch, preprocess_bars, validate_keys
from jinyiquant.io import file_sha256, read_json, utc_now, write_json
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir

from .adapter import TLOBAdapter


def table_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError("Data source must be a simple SQL table identifier")
    return value


def query_frame(query, table, columns, start, end):
    return query(f"SELECT {', '.join(columns)} FROM {table_name(table)}",
                 filters={"date": [str(start), str(end)]}).df()


def predict_range(datasources, start_date, end_date, *, package_root=None, query=None):
    root = Path(package_root) if package_root else Path(__file__).resolve().parents[1]
    run_dir = new_run_dir(root / "inference_runs")
    rec = Recorder(run_dir, "modified-tlob-platform-inference")
    started = time.monotonic()
    selected = "unknown"
    write_json(run_dir / "manifest.json", {"schema_version": 1, "run_id": run_dir.name,
        "experiment_id": rec.experiment_id, "started_at": utc_now(), "status": "running",
        "scope": "platform_inference_only", "start_date": str(start_date), "end_date": str(end_date)})
    rec.event("run", "started", start_date=str(start_date), end_date=str(end_date), label_input=False)
    try:
        if query is None:
            import dai
            query = dai.query
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        if start.tzinfo or end.tzinfo or start > end:
            raise ValueError("Expected naive exchange-local start <= end")
        weights = root / "modified_tlob_weights.json"
        contract = read_json(root / "submission_config.json")
        if contract["weights_sha256"] != file_sha256(weights):
            raise ValueError("Submission weight checksum mismatch")
        manifest = read_json(run_dir / "manifest.json")
        manifest.update(code=capture_source(run_dir), environment=environment(),
                        model_sha256=contract["weights_sha256"], contract=contract)
        write_json(run_dir / "manifest.json", manifest)
        rec.event("loading", "started", path="modified_tlob_weights.json")
        model = TLOBAdapter.load(weights, device=contract["device"], recorder=rec)
        selected = str(model.device)
        manifest["environment"]["determinism"].update(device=selected, cpu_threads=model.config["cpu_threads"],
            torch_version=manifest["environment"]["dependencies"].get("torch"), deterministic_algorithms=True)
        write_json(run_dir / "manifest.json", manifest)
        rec.event("loading", "model_loaded", path="modified_tlob_weights.json")
        data = dict(model.preprocessing)
        # DAI quotes are yuan; offline download quotes may have been cents.
        # Both pass through the same conversion to canonical yuan windows.
        data["price_scale"] = contract["platform_price_scale"]
        warm_start = start - pd.Timedelta(days=data["history_days"])
        columns = ["date", "instrument", *data["feature_columns"]]
        rec.event("data", "loading", source=table_name(datasources["bar1m"]),
                  feature_columns=data["feature_columns"], warm_start=str(warm_start))
        bars = query_frame(query, datasources["bar1m"], columns, warm_start, end)
        bars["date"] = pd.to_datetime(bars["date"])
        bars = bars.loc[(bars.date >= warm_start) & (bars.date <= end)]
        bars = preprocess_bars(bars, data)
        pool = query_frame(query, datasources.get("instruments", "cpt_jyc_2026_instruments"),
                           ["date", "instrument"], start, end)
        pool = validate_keys(pool, "platform universe")
        keys = pool.loc[(pool.date >= start) & (pool.date <= end) &
                        pool.date.dt.strftime("%H:%M").isin(data["sample_times"])]
        if keys.empty:
            raise ValueError("No eligible platform prediction keys")
        rec.event("data", "validated", bars=len(bars), keys=len(keys), label_input=False)
        write_json(run_dir / "data.json", {
            "schema_version": 1, "source": "platform_DAI", "bar_table": datasources["bar1m"],
            "universe_table": datasources.get("instruments", "cpt_jyc_2026_instruments"),
            "start": str(start), "end": str(end), "warm_start": str(warm_start),
            "preprocessed_bars_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(bars, index=False).to_numpy().tobytes()).hexdigest(),
            "keys_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(keys, index=False).to_numpy().tobytes()).hexdigest(),
            "feature_columns": data["feature_columns"], "platform_price_scale": data["price_scale"],
            "rows": len(bars), "prediction_keys": len(keys), "label_input": False})
        rec.event("inference", "started", label_input=False)
        results = []
        # Build windows per prediction timestamp to bound the dominant tensor allocation.
        for date, group in keys.groupby("date", sort=True):
            if time.monotonic() - started > contract["wall_seconds"]:
                raise TimeoutError("Platform inference wall-time budget exceeded")
            batch = build_batch(bars, group, data)
            if batch.coverage["min_section_coverage"] < contract["min_coverage"]:
                raise ValueError(f"Insufficient history coverage at {date}")
            rec.event("preprocessing", "completed", date=str(date), coverage=batch.coverage)
            result = batch.keys.assign(factor=model.predict(batch))
            results.append(result)
        result = pd.concat(results, ignore_index=True)[["date", "instrument", "factor"]]
        if not result[["date", "instrument"]].equals(keys.reset_index(drop=True)[["date", "instrument"]]):
            raise ValueError("Prediction universe alignment failed")
        rec.event("inference", "completed", samples=len(result), label_input=False)
        elapsed = time.monotonic() - started
        if elapsed > contract["wall_seconds"]:
            raise TimeoutError("Platform inference wall-time budget exceeded")
        finish(run_dir, "completed", 0, resources={"wall_seconds": elapsed, "device": selected})
        return result
    except BaseException as exc:
        rec.failure("inference", exc)
        finish(run_dir, "failed", 1, resources={"wall_seconds": time.monotonic() - started, "device": selected},
               error={"exception_type": type(exc).__name__, "reason": str(exc)})
        raise


def main(datasources, start_date, end_date):
    return predict_range(datasources, start_date, end_date)
