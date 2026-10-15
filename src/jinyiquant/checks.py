"""Behavioral acceptance checks used by actual runs and standalone tests."""

import numpy as np
import pandas as pd

from .data import build_batch
from .io import file_sha256, read_json


def check(name, passed, **details):
    return {"name": name, "passed": bool(passed), "details": details}


def batch_checks(batch, targets, split, minimum_coverage):
    timestamps = batch.history_dates.astype("datetime64[ns]")
    cutoffs = batch.keys["date"].to_numpy(dtype="datetime64[ns]")[:, None]
    valid = ~np.isnat(timestamps)
    observed_rows = batch.observed.any(axis=2).any(axis=1)
    coverage_by_date = pd.DataFrame({"date": batch.keys["date"], "covered": observed_rows}).groupby("date").covered.mean()
    return [check("finite_features", np.isfinite(batch.x).all()),
            check("causal_history", np.all(~valid | (timestamps <= cutoffs))),
            check("label_boundary", ((targets.date >= split.start) & (targets.date < split.end)
                                      & (targets.label_end < split.end)
                                      & (targets.label_end > targets.date)).all()),
            check("sample_coverage", len(coverage_by_date) > 0 and coverage_by_date.min() >= minimum_coverage
                  and batch.coverage["min_section_coverage"] >= minimum_coverage,
                  minimum_cross_section_coverage=float(coverage_by_date.min()) if len(coverage_by_date) else 0,
                  minimum_history_coverage=batch.coverage["min_section_coverage"],
                  required=minimum_coverage)]


def prediction_checks(model, batch, bars, data_config, atol, rtol):
    scores = model.predict(batch)
    checks = [check("finite_predictions", scores.shape == (len(batch.keys),) and np.isfinite(scores).all()),
              check("unique_prediction_keys", not batch.keys.duplicated(["date", "instrument"]).any())]
    # Evaluate the same earlier keys once with all data and once with changed future data.
    cutoff = batch.keys.date.sort_values().iloc[len(batch.keys) // 2]
    earlier_keys = batch.keys.loc[batch.keys.date <= cutoff].reset_index(drop=True)
    altered = bars.copy()
    future = altered.date > cutoff
    altered.loc[future, list(data_config.feature_columns)] = 1e9
    injected_rows = 0
    if not future.any():
        # Even a single-section run must exercise the future-data boundary.
        injected = earlier_keys.drop_duplicates("instrument").copy()
        injected["date"] = cutoff + pd.Timedelta(minutes=1)
        for column in data_config.feature_columns:
            injected[column] = 1e9
        altered = pd.concat([altered, injected], ignore_index=True)
        injected_rows = len(injected)
    original = build_batch(bars, earlier_keys, data_config)
    changed = build_batch(altered, earlier_keys, data_config)
    inputs_unchanged = (np.array_equal(original.x, changed.x)
                        and np.array_equal(original.observed, changed.observed)
                        and np.array_equal(original.history_dates, changed.history_dates, equal_nan=True))
    checks.append(check("future_market_invariance",
                        inputs_unchanged and np.allclose(model.predict(original), model.predict(changed),
                                                         atol=atol, rtol=rtol),
                        cutoff=str(cutoff), altered_future_rows=int(future.sum()) + injected_rows,
                        injected_future_rows=injected_rows, input_windows_unchanged=inputs_unchanged))
    # A fresh batch built solely from bars and keys proves the inference path has no target dependency.
    clean = build_batch(bars, batch.keys[["date", "instrument"]], data_config)
    checks.append(check("inference_without_labels",
                        np.allclose(scores, model.predict(clean), atol=atol, rtol=rtol),
                        inputs=["bars", "date/instrument keys", "saved model"]))
    return checks


def numerical_equal(left, right, atol, rtol):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(numerical_equal(left[k], right[k], atol, rtol) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(numerical_equal(a, b, atol, rtol) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(left, right, atol=atol, rtol=rtol, equal_nan=False))
    return left == right


def verify_artifacts(run_dir):
    manifest = read_json(run_dir / "manifest.json")
    results = []
    for artifact in manifest["artifacts"]:
        path = (run_dir / artifact["path"]).resolve()
        within = path.is_relative_to(run_dir.resolve())
        results.append(check("artifact:" + artifact["path"], within and path.is_file()
                             and file_sha256(path) == artifact["sha256"]))
    return {"schema_version": 1, "passed": bool(results) and all(c["passed"] for c in results), "checks": results}
