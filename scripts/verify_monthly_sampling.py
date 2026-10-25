"""Verify a monthly-sampled training read on real data and record machine-readable evidence."""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from jinyiquant.config import load_config
from jinyiquant.data import build_batch, history_diagnostics, load_data, select_targets
from jinyiquant.io import utc_now, write_json
from modified_tlob.adapter import timestamp_zscore_train_rms_matched


def rss_mb():
    return round(psutil.Process().memory_info().rss / 2 ** 20, 1)


def verify(config, output):
    sampling = config.data.monthly_sampling
    if sampling is None:
        raise ValueError("This verification requires data.monthly_sampling")
    start, end = config.splits.train.start, config.splits.validation.end
    began, rss_before = time.monotonic(), rss_mb()
    bundle = load_data(config.data, start=start, end=end)
    cold_seconds, cold_rss = time.monotonic() - began, rss_mb()
    audit, digest = bundle.metadata["monthly_sampling"], bundle.metadata["data_sha256"]
    selected = pd.DatetimeIndex(audit["selected_days"])
    months = len(pd.date_range(sampling.start, sampling.end, freq="MS", inclusive="left"))
    label_days = pd.DatetimeIndex(bundle.labels.date.dt.normalize().unique())
    universe_days = pd.DatetimeIndex(bundle.instruments.date.dt.normalize().unique())
    bar_days = pd.DatetimeIndex(bundle.bars.date.dt.normalize().unique())
    def inside(days):
        return days[(days >= pd.Timestamp(sampling.start)) & (days < pd.Timestamp(sampling.end))]
    history_end = pd.Timestamp(audit["history_end"])
    per_day = bundle.labels.groupby(bundle.labels.date.dt.normalize())["date"].nunique()
    checks = {
        "selected_days": audit["days"] == sampling.days_per_month * months == len(selected),
        "month_audit": len(audit["months"]) == months,
        "labels_restricted_to_selected_days": set(inside(label_days).strftime("%Y-%m-%d")) == set(audit["selected_days"]),
        "universe_restricted_to_selected_days": set(inside(universe_days).strftime("%Y-%m-%d")) == set(audit["selected_days"]),
        "eight_sample_times_per_day": bool((per_day.reindex(selected) == len(config.data.sample_times)).all()),
        "bars_keep_selected_days_and_context": set(bar_days[(bar_days >= pd.Timestamp(sampling.start))
                                                           & (bar_days < history_end)].strftime("%Y-%m-%d")) == {
            day for day in set(audit["selected_days"]) | set(audit["context_days"]) if pd.Timestamp(day) < history_end},
        "bars_keep_post_range_warmup": set(universe_days[(universe_days >= history_end)
                                                         & (universe_days < pd.Timestamp(end))].strftime("%Y-%m-%d"))
        <= set(bar_days[bar_days >= history_end].strftime("%Y-%m-%d")),
    }
    if "09:30" not in config.data.sample_times:
        raise ValueError("Opening-window verification requires the 09:30 sampling time")
    opening_days = selected[::sampling.days_per_month]
    opening = bundle.labels.loc[bundle.labels.date.dt.normalize().isin(opening_days)
                                & bundle.labels.date.dt.strftime("%H:%M").eq("09:30")]
    batch = build_batch(bundle.bars, opening[["date", "instrument"]], config.data)
    diagnostics = history_diagnostics(batch)
    history = {"opening_days": len(opening_days), "keys": int(len(batch.keys)),
               "lookback": int(config.data.lookback),
               "history_rows_min": int(diagnostics.history_row_count.min()),
               "history_rows_median": float(diagnostics.history_row_count.median()),
               "full_lookback_fraction": float((diagnostics.history_row_count >= config.data.lookback).mean()),
               "usable_samples": int(batch.coverage["usable_samples"]),
               "min_section_coverage": float(batch.coverage["min_section_coverage"])}
    checks["opening_day_coverage"] = history["min_section_coverage"] >= config.acceptance.min_coverage
    targets = select_targets(bundle.labels, config.splits.train, config.data.label_horizon_minutes,
                             sample_times=config.data.training_sample_times or config.data.sample_times)
    labels = targets["target"].to_numpy(dtype=float)
    transformed, csz_stats = timestamp_zscore_train_rms_matched(targets[["date"]], labels, np.isfinite(labels))
    csz = {"rows": csz_stats["rows"], "timestamps": csz_stats["timestamps"],
           "centered_std": csz_stats["centered_std"],
           "min_rows_per_timestamp": csz_stats["min_rows_per_timestamp"],
           "zero_variance_timestamps": csz_stats["zero_variance_timestamps"],
           "raw_std": csz_stats["raw_std"], "transformed_std": csz_stats["transformed_std"],
           "c": csz_stats["c"], "finite": bool(np.isfinite(transformed).all())}
    evidence = {
        "schema_version": 1, "created_at_utc": utc_now(), "experiment_id": config.experiment_id,
        "sampling": {"days": audit["days"], "months": len(audit["months"]),
                     "days_per_month": audit["days_per_month"], "selected": audit["selected_days"],
                     "context_days": len(audit["context_days"]), "history_end": audit["history_end"]},
        "rows": {"bars": len(bundle.bars), "bar_days": int(len(bar_days)), "labels": len(bundle.labels),
                 "universe": len(bundle.instruments), "training_samples": len(targets)},
        "opening_history": history, "csz": csz,
        "cache": {"cold_events": len(bundle.cache_events),
                  "written": sum(event["event"] == "cache_written" for event in bundle.cache_events)},
        "resources": {"cold_seconds": round(cold_seconds, 1), "rss_before_mb": rss_before,
                      "rss_after_read_mb": cold_rss},
    }
    del targets, transformed, batch, diagnostics, bundle
    gc.collect()
    warm_began = time.monotonic()
    warm = load_data(config.data, start=start, end=end)
    evidence["resources"].update(warm_seconds=round(time.monotonic() - warm_began, 1), rss_warm_mb=rss_mb())
    checks["warm_read_all_cache_hits"] = all(event["event"] == "cache_hit" for event in warm.cache_events)
    checks["warm_read_identical_metadata"] = warm.metadata["data_sha256"] == digest
    evidence["checks"] = checks
    evidence["passed"] = all(checks.values())
    write_json(output, evidence)
    print(f"passed={evidence['passed']} days={audit['days']} months={months} samples={len(warm.labels)} "
          f"cold_s={cold_seconds:.1f} warm_s={evidence['resources']['warm_seconds']:.1f} "
          f"rss_mb={cold_rss}", flush=True)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    return 0 if verify(load_config(args.config), args.output)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
