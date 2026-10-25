"""Platform inference: historical bars and point-in-time pool, never answers or training."""

import hashlib
import re
import time
from pathlib import Path

import pandas as pd

from jinyiquant.data import build_batch, preprocess_bars, validate_keys
from jinyiquant.io import canonical_json, file_sha256, read_json, utc_now, write_json
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir

from .adapter import TLOBAdapter
from .device import device_resources


def table_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError("Data source must be a simple SQL table identifier")
    return value


def query_frame(query, table, columns, start, end):
    return query(f"SELECT {', '.join(columns)} FROM {table_name(table)}",
                 filters={"date": [str(start), str(end)]}).df()


def iter_bar_days(query, table, warm_start, end, data):
    """Retain the same trailing observations as one full-range read, in bounded daily blocks."""
    cursor, end = pd.Timestamp(warm_start), pd.Timestamp(end)
    columns = ["date", "instrument", *data["feature_columns"]]
    carry = None
    lookback = int(data["lookback"])
    while cursor <= end:
        next_day = cursor.normalize() + pd.Timedelta(days=1)
        upper = min(next_day, end)
        final_block = end < next_day
        raw = query_frame(query, table, columns, cursor, upper)
        raw["date"] = pd.to_datetime(raw["date"])
        # DAI date filters include both ends. Give midnight to the following
        # block, except when it is the exact final requested observation.
        within_end = raw.date <= upper if final_block else raw.date < upper
        raw = raw.loc[(raw.date >= cursor) & within_end]
        current = preprocess_bars(raw, data)
        del raw
        metadata = {
            "start": str(cursor), "end": str(upper), "end_inclusive": final_block,
            "rows": len(current),
            "preprocessed_bars_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(current, index=False).to_numpy().tobytes()).hexdigest(),
        }
        window = current if carry is None or carry.empty else pd.concat([carry, current], ignore_index=True)
        del current
        # Keep ALL securities and even all-missing rows. A security entering
        # tomorrow's pool may need old observations from before a suspension.
        # Only new rows are preprocessed; carrying canonical values avoids
        # applying the source price scale twice.
        carry = window.groupby("instrument", sort=False, observed=True).tail(lookback).copy()
        metadata["retained_rows"] = len(carry)
        yield window, metadata
        del window
        if final_block:
            break
        cursor = next_day


def section_history(bars, keys, lookback):
    """Prune irrelevant rows before the shared window builder copies/sorts them."""
    latest = keys["date"].max()
    selected = bars.loc[(bars.date <= latest) & bars.instrument.isin(keys.instrument)]
    return selected.groupby("instrument", sort=False, observed=True).tail(lookback)


class HistoryRecovery:
    """Cache bounded, strictly earlier quotes only for insufficient windows."""

    def __init__(self, query, table, data, warm_start, policy, deadline, recorder):
        self.query, self.table, self.data = query, table_name(table), data
        self.warm_start, self.deadline, self.recorder = warm_start, deadline, recorder
        self.max_days = policy.get("max_backfill_days", 366)
        self.group_size = policy.get("instrument_batch_size", 50)
        if (not isinstance(self.max_days, int) or not 1 <= self.max_days <= 366
                or not isinstance(self.group_size, int) or not 1 <= self.group_size <= 50):
            raise ValueError("History recovery requires 1..366 days and 1..50 instruments per query")
        self.cache, self.attempted = {}, set()

    def recover(self, history, batch):
        missing = batch.effective_rows.sum(axis=1) < self.data["min_history"]
        if not missing.any():
            return batch
        row_counts = (~pd.isna(batch.history_dates)).sum(axis=1)
        names = batch.keys.loc[missing, "instrument"].tolist()
        # Older observations cannot enter a window already occupied by L rows,
        # even when those L rows are all missing. Preserve the trained semantics.
        pending = [name for name, count in zip(batch.keys.loc[missing, "instrument"], row_counts[missing])
                   if name not in self.attempted and count < self.data["lookback"]]
        cutoff = batch.keys.date.max()
        lower = cutoff - pd.Timedelta(days=self.max_days)
        columns = ["date", "instrument", *self.data["feature_columns"]]
        for offset in range(0, len(pending), self.group_size):
            group = pending[offset:offset + self.group_size]
            if time.monotonic() > self.deadline:
                raise TimeoutError("Platform inference budget exceeded during history recovery")
            if lower < self.warm_start:
                # Date predicates precede ROW_NUMBER so later observations can
                # never displace the required earlier history. Limit in SQL,
                # before materializing pandas, to preserve the memory bound.
                sql = (f"SELECT {', '.join(columns)} FROM {self.table} "
                       f"WHERE date >= TIMESTAMP '{lower}' AND date < TIMESTAMP '{self.warm_start}' "
                       "QUALIFY ROW_NUMBER() OVER (PARTITION BY instrument ORDER BY date DESC) "
                       f"<= {int(self.data['lookback'])}")
                raw = self.query(sql, filters={"date": [str(lower), str(self.warm_start)],
                                               "instrument": group}).df()
                raw["date"] = pd.to_datetime(raw["date"])
                raw = raw.loc[raw.instrument.isin(group) & raw.date.ge(lower) & raw.date.lt(self.warm_start)]
                recovered = preprocess_bars(raw, self.data)
                if recovered.groupby("instrument", observed=True).size().gt(self.data["lookback"]).any():
                    raise ValueError("History recovery query exceeded its per-instrument row limit")
                for name, frame in recovered.groupby("instrument", sort=False, observed=True):
                    self.cache[name] = frame.copy()
                self.recorder.event("data", "history_backfilled", date=str(cutoff),
                                    start=str(lower), end_exclusive=str(self.warm_start),
                                    requested_instruments=len(group), rows=len(recovered),
                                    rows_per_instrument_limit=self.data["lookback"],
                                    instruments_sha256=hashlib.sha256(canonical_json(group).encode()).hexdigest(),
                                    preprocessed_bars_sha256=hashlib.sha256(pd.util.hash_pandas_object(
                                        recovered, index=False).to_numpy().tobytes()).hexdigest())
            self.attempted.update(group)
        supplements = [self.cache[name] for name in names if name in self.cache]
        if not supplements:
            return batch
        # Cache is applied only to originally insufficient instruments. Windows
        # and batch positions of all already-usable samples remain unchanged.
        combined = pd.concat([*supplements, history], ignore_index=True)
        return build_batch(section_history(combined, batch.keys, self.data["lookback"]), batch.keys, self.data)


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
        if pd.isna(start) or pd.isna(end) or start.tzinfo or end.tzinfo or start > end:
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
        rec.event("loading", "model_loaded", path="modified_tlob_weights.json",
                  open_residual=None if model.open_residual_head is None else {
                      key: model.open_residual_metadata[key]
                      for key in ("method", "sample_time", "feature_dim")})
        data = dict(model.preprocessing)
        # DAI quotes are yuan; offline download quotes may have been cents.
        # Both pass through the same conversion to canonical yuan windows.
        data["price_scale"] = contract["platform_price_scale"]
        warm_start = start - pd.Timedelta(days=data["history_days"])
        bar_table = table_name(datasources["bar1m"])
        history_policy = contract.get("history_policy", {"mode": "strict"})
        if history_policy.get("mode") not in {"strict", "recover_then_mean"}:
            raise ValueError("Unknown platform history policy")
        recovery = (HistoryRecovery(query, bar_table, data, warm_start, history_policy,
                                    started + contract["wall_seconds"], rec)
                    if history_policy["mode"] == "recover_then_mean" else None)
        rec.event("data", "loading", source=table_name(datasources["bar1m"]),
                  feature_columns=data["feature_columns"], warm_start=str(warm_start),
                  strategy="daily_with_per_instrument_tail", resources=device_resources(model.device))
        pool = query_frame(query, datasources.get("instruments", "cpt_jyc_2026_instruments"),
                           ["date", "instrument"], start, end)
        pool = validate_keys(pool, "platform universe")
        keys = pool.loc[(pool.date >= start) & (pool.date <= end) &
                        pool.date.dt.strftime("%H:%M").isin(data["sample_times"])]
        del pool
        if keys.empty:
            raise ValueError("No eligible platform prediction keys")
        rec.event("data", "validated", keys=len(keys), label_input=False)
        metadata = {
            "schema_version": 2, "source": "platform_DAI", "bar_table": bar_table,
            "universe_table": datasources.get("instruments", "cpt_jyc_2026_instruments"),
            "start": str(start), "end": str(end), "warm_start": str(warm_start),
            "strategy": "daily_with_per_instrument_tail", "retained_rows_per_instrument": data["lookback"],
            "keys_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(keys, index=False).to_numpy().tobytes()).hexdigest(),
            "feature_columns": data["feature_columns"], "platform_price_scale": data["price_scale"],
            "rows": 0, "prediction_keys": len(keys), "label_input": False,
            "history_policy": history_policy, "chunks": []}
        write_json(run_dir / "data.json", metadata)
        rec.event("inference", "started", label_input=False)
        results = []
        fallback_samples, recovered_samples, usable_samples = 0, 0, 0
        warned_days = set()
        key_days = keys.date.dt.normalize()
        for bars, chunk in iter_bar_days(query, bar_table, warm_start, end, data):
            if time.monotonic() - started > contract["wall_seconds"]:
                raise TimeoutError("Platform inference wall-time budget exceeded")
            metadata["rows"] += chunk["rows"]
            metadata["chunks"].append(chunk)
            write_json(run_dir / "data.json", metadata)
            rec.event("data", "chunk_loaded", **chunk, window_rows=len(bars),
                      resources=device_resources(model.device))
            day_keys = keys.loc[key_days.eq(pd.Timestamp(chunk["start"]).normalize())]
            for date, group in day_keys.groupby("date", sort=True):
                if time.monotonic() - started > contract["wall_seconds"]:
                    raise TimeoutError("Platform inference wall-time budget exceeded")
                history = section_history(bars, group, data["lookback"])
                batch = build_batch(history, group, data)
                before = batch.coverage["usable_samples"]
                if recovery is not None:
                    batch = recovery.recover(history, batch)
                del history
                expected, usable = batch.coverage["expected_samples"], batch.coverage["usable_samples"]
                insufficient = expected - usable
                rec.event("preprocessing", "history_checked", date=str(date),
                          coverage=batch.coverage, usable_before_recovery=before,
                          recovered_samples=usable - before, history_policy=history_policy["mode"])
                if batch.coverage["min_section_coverage"] < contract["min_coverage"] and recovery is None:
                    raise ValueError(f"Insufficient history coverage at {date}: usable={usable}/{expected}, "
                                     f"required={contract['min_coverage']}, min_history={data['min_history']}, "
                                     f"warm_start={warm_start}")
                if insufficient and recovery is not None:
                    detail = {"date": str(date), "expected": expected, "usable": usable,
                              "fallback_samples": insufficient, "fallback": "saved_training_mean",
                              "coverage": batch.coverage["min_section_coverage"],
                              "warning_threshold": contract["min_coverage"]}
                    rec.event("inference", "history_fallback", level="WARNING", **detail)
                    if date.normalize() not in warned_days:
                        print(canonical_json({"event_type": "history_fallback", **detail}), flush=True)
                        warned_days.add(date.normalize())
                fallback_samples += insufficient
                recovered_samples += usable - before
                usable_samples += usable
                rec.event("preprocessing", "completed", date=str(date), coverage=batch.coverage)
                results.append(batch.keys.assign(factor=model.predict(batch)))
                del batch
            del bars
        result = pd.concat(results, ignore_index=True)[["date", "instrument", "factor"]]
        if not result[["date", "instrument"]].equals(keys.reset_index(drop=True)[["date", "instrument"]]):
            raise ValueError("Prediction universe alignment failed")
        manifest = read_json(run_dir / "manifest.json")
        manifest["history_handling"] = {"policy": history_policy, "usable_samples": usable_samples,
                                        "recovered_samples": recovered_samples,
                                        "fallback_samples": fallback_samples,
                                        "data_quality": "fallback_used" if fallback_samples else "all_usable"}
        write_json(run_dir / "manifest.json", manifest)
        if recovery is not None and usable_samples == 0:
            raise ValueError("No usable historical quotes in the entire prediction range, even after backfill; "
                             "check the bar datasource, instrument identifiers and available dates")
        rec.event("inference", "completed", samples=len(result), label_input=False)
        elapsed = time.monotonic() - started
        if elapsed > contract["wall_seconds"]:
            raise TimeoutError("Platform inference wall-time budget exceeded")
        finish(run_dir, "completed", 0, resources={"wall_seconds": elapsed, **device_resources(model.device)})
        return result
    except BaseException as exc:
        rec.failure("inference", exc)
        finish(run_dir, "failed", 1, resources={"wall_seconds": time.monotonic() - started, "device": selected},
               error={"exception_type": type(exc).__name__, "reason": str(exc)})
        raise


def main(datasources, start_date, end_date):
    return predict_range(datasources, start_date, end_date)
