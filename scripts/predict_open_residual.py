"""Score the frozen V2 trunk plus its trained 09:30 residual head on the development months.

The V3-open candidate must be *identical* to the frozen V2-CSZ baseline everywhere except
the 09:30 section.  This entry point therefore re-generates both weight sets on exactly the
archived replay keys and asserts, per day chunk,

* the V2 trunk scores reproduce the archived replay predictions bit for bit,
* every non-09:30 candidate score is bit-identical to the reproduced V2 score,
* every changed row is a 09:30 row, and
* splicing the candidate 09:30 rows into the archived V2 table yields the candidate table
  bit for bit (so the other seven sections really are the V2 predictions reused verbatim).

One prediction archive is written per seed/checkpoint under ``--output-root`` (default
``runs/v3-open-20261024/seed<seed>``) with per-window ``predictions.parquet`` and the
combined full-day table.  No label value is read by the prediction path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jinyiquant.data import build_batch, load_data  # noqa: E402
from jinyiquant.io import file_sha256, utc_now, write_json  # noqa: E402
from modified_tlob.adapter import OPEN_SAMPLE_TIME, TLOBAdapter, open_market_mask  # noqa: E402
from modified_tlob.device import device_resources  # noqa: E402

DATA_ROOT = Path("D:/进益资本杯/数据")
PLAN_PATH = ROOT / "configs" / "label_contrast_20261024_plan.json"
REPLAY_RUN = Path("D:/进益资本杯/runs/label-contrast-eval-20261024/20261024T171905.318633Z_ad9cc8a7400e")
BASELINE_WEIGHTS = Path("D:/进益资本杯/runs/model-versions/v2-cloud-20261023/60d/modified_tlob_weights.json")
BASELINE_WEIGHTS_SHA256 = "3e44451670320c282f7421b77d773bc2af8190cbdf02a62c2a1f7e6b9284c0aa"
DEVELOPMENT_WINDOWS = ("nov2023", "jan2024", "mar2024")
KEYS = ["date", "instrument"]


def replay_contract() -> tuple[dict, list[dict]]:
    """Replay data contract with worktree-safe absolute local paths."""
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    plan["data"].update({
        "bars_path": str(DATA_ROOT / "cpt_jyc_2026_e2e_bar1m"),
        "labels_path": str(DATA_ROOT / "cpt_jyc_2026_vwap"),
        "instruments_path": str(DATA_ROOT / "cpt_jyc_2026_instruments(1).parquet"),
        "instrument_map_path": str(DATA_ROOT / "cpt_jyc_2026_stock_bar1m_instrument_mapping.csv"),
        "cache_dir": None,
    })
    windows = [window for window in plan["windows"] if window["name"] in DEVELOPMENT_WINDOWS]
    if [window["name"] for window in windows] != list(DEVELOPMENT_WINDOWS):
        raise ValueError("development windows are missing from the replay plan")
    return plan, windows


def archived(window: str) -> Path:
    return REPLAY_RUN / "windows" / window / "v2_60d_csz"


def predict_window(models: dict, bundle, config: dict, keys: pd.DataFrame, chunk_days: int):
    """Day-chunked predictions on the archived key order (same batches as the replay)."""
    days = np.sort(pd.unique(keys.date.dt.normalize()))
    scores = {name: [] for name in models}
    chunks = []
    for offset in range(0, len(days), chunk_days):
        chunk_set = set(pd.to_datetime(days[offset:offset + chunk_days]))
        chunk_keys = keys.loc[keys.date.dt.normalize().isin(chunk_set)].reset_index(drop=True)
        started = time.monotonic()
        batch = build_batch(bundle.bars, chunk_keys, config)
        for name, model in models.items():
            scores[name].append(model.predict(batch))
        chunks.append({"days": len(chunk_set), "keys": len(batch.keys),
                       "clock_counts": {clock: int(count) for clock, count in
                                        batch.keys.date.dt.strftime("%H:%M").value_counts()
                                        .sort_index().items()},
                       "seconds": float(time.monotonic() - started)})
        del batch
    return {name: np.concatenate(values) for name, values in scores.items()}, chunks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="V3-open checkpoint archive (model.pt)")
    parser.add_argument("--base-weights", type=Path, default=BASELINE_WEIGHTS)
    parser.add_argument("--output-root", type=Path, help="Default: runs/v3-open-20261024/seed<seed>")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--chunk-days", type=int, default=11)
    parser.add_argument("--windows", nargs="+", choices=DEVELOPMENT_WINDOWS,
                        help="Development windows to score (default: all three)")
    args = parser.parse_args(argv)

    plan, windows = replay_contract()
    if args.windows:
        plan["windows"] = [window for window in plan["windows"] if window["name"] in args.windows]
        windows = [window for window in windows if window["name"] in args.windows]
    weights = args.weights.resolve()
    base_weights = args.base_weights.resolve()
    base_sha = file_sha256(base_weights)
    if base_sha != BASELINE_WEIGHTS_SHA256:
        raise ValueError("Frozen V2 weights do not match the declared sha256")
    candidate = TLOBAdapter.load(weights, device=args.device)
    if candidate.open_residual_head is None:
        raise ValueError("The candidate checkpoint carries no 09:30 residual head")
    metadata = candidate.open_residual_metadata
    if metadata.get("base_state_sha256") is None:
        raise ValueError("The candidate checkpoint does not carry its trunk provenance")
    head_seed = metadata.get("init_seed", metadata.get("seed"))
    if head_seed is None or int(metadata.get("shuffle_seed", head_seed)) != int(head_seed):
        raise ValueError("The candidate checkpoint does not identify a single head seed")
    seed = int(head_seed)
    output_root = (args.output_root or ROOT / "runs" / "v3-open-20261024" / f"seed{seed}").resolve()
    (output_root / "predictions").mkdir(parents=True, exist_ok=True)

    batch_size = int(plan.get("predict_batch_size", 32))
    candidate.config["batch_size"] = batch_size
    baseline = TLOBAdapter.load(base_weights, device=args.device)
    baseline.config["batch_size"] = batch_size
    if baseline.open_residual_head is not None:
        raise ValueError("The frozen baseline checkpoint unexpectedly carries a residual head")

    started = time.monotonic()
    window_records, combined, checks = [], [], []
    for window in windows:
        name = window["name"]
        start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        bundle = load_data(plan["data"], start, end)
        keys = pd.read_parquet(REPLAY_RUN / "windows" / name / "keys.parquet")
        if list(keys.columns) != KEYS or keys.duplicated(KEYS).any():
            raise ValueError(f"{name}: archived prediction keys are malformed")
        archived_table = pd.read_parquet(archived(name) / "predictions.parquet")
        if len(archived_table) != len(keys) or set(archived_table.columns) != {"date", "instrument", "score"}:
            raise ValueError(f"{name}: archived baseline predictions are malformed")
        reference = archived_table["score"].to_numpy(dtype=np.float64)
        scores, chunks = predict_window({"v2": baseline, "v3": candidate}, bundle, plan["data"],
                                        keys, args.chunk_days)
        v2, v3 = scores["v2"], scores["v3"]
        opening = open_market_mask(keys, OPEN_SAMPLE_TIME)
        non_opening = ~opening
        changed = v3 != v2
        spliced = reference.copy()
        spliced[opening] = v3[opening]
        window_checks = [
            {"id": "baseline_reproduced_bitwise", "passed": bool(np.array_equal(v2, reference)),
             "rows": int(len(v2)), "max_abs_diff": float(np.max(np.abs(v2 - reference)))},
            {"id": "non_opening_bitwise_equal_to_v2",
             "passed": bool(np.array_equal(v3[non_opening], v2[non_opening])),
             "rows": int(non_opening.sum()),
             "max_abs_diff": float(np.max(np.abs(v3[non_opening] - v2[non_opening])))},
            {"id": "changes_confined_to_opening", "passed": bool(changed[non_opening].sum() == 0),
             "changed_rows": int(changed.sum()), "non_opening_changed": int(changed[non_opening].sum())},
            {"id": "opening_splice_equals_candidate", "passed": bool(np.array_equal(spliced, v3)),
             "max_abs_diff": float(np.max(np.abs(spliced - v3)))},
            {"id": "opening_differs_from_v2", "passed": bool(changed[opening].sum() > 0),
             "opening_rows": int(opening.sum()), "opening_changed": int(changed[opening].sum()),
             "opening_unchanged_rows": int((opening & ~changed).sum())},
        ]
        checks.extend({"window": name, **item} for item in window_checks)
        if not all(item["passed"] for item in window_checks):
            raise ValueError(f"{name}: prediction invariance checks failed: "
                             f"{[item['id'] for item in window_checks if not item['passed']]}")

        table = keys.assign(window=name, score=v3.astype(np.float64))
        out = output_root / "predictions" / name
        out.mkdir(parents=True, exist_ok=True)
        table[KEYS + ["score"]].to_parquet(out / "predictions.parquet", index=False)
        reloaded = pd.read_parquet(out / "predictions.parquet")
        if not np.array_equal(reloaded["score"].to_numpy(dtype=np.float64), v3):
            raise ValueError(f"{name}: parquet round-trip changed the predictions")
        combined.append(table)
        coverage = json.loads((REPLAY_RUN / "windows" / name / "coverage.json").read_text(encoding="utf-8"))
        opening_delta = v3[opening] - v2[opening]
        window_records.append({
            "window": name, "start": str(start)[:10], "end": str(end)[:10],
            "days": int(pd.unique(keys.date.dt.normalize()).size),
            "prediction_keys": int(len(keys)), "target_keys": int(coverage["targets"]),
            "sections": int(len(chunks)), "chunks": chunks,
            "archived_baseline_predictions_sha256": file_sha256(archived(name) / "predictions.parquet"),
            "archived_keys_sha256": file_sha256(REPLAY_RUN / "windows" / name / "keys.parquet"),
            "changed_rows": int(changed.sum()), "opening_rows": int(opening.sum()),
            "opening_unchanged_rows": int((opening & ~changed).sum()),
            "opening_delta_abs_mean": float(np.abs(opening_delta).mean()),
            "opening_delta_abs_max": float(np.abs(opening_delta).max()),
            "checks": window_checks,
        })
        print(json.dumps({"window": name, "keys": int(len(keys)), "changed": int(changed.sum()),
                          "opening_changed": int(changed[opening].sum())}), flush=True)
        del bundle, keys, archived_table, reference, scores, v2, v3, spliced, table, reloaded

    full = pd.concat(combined, ignore_index=True)[["window", "date", "instrument", "score"]]
    full.to_parquet(output_root / "predictions_full_development.parquet", index=False)
    reloaded = pd.read_parquet(output_root / "predictions_full_development.parquet")
    if not np.array_equal(reloaded["score"].to_numpy(dtype=np.float64),
                          full["score"].to_numpy(dtype=np.float64)):
        raise ValueError("combined parquet round-trip changed the predictions")

    evidence = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "experiment_id": "v3-open-20261024",
        "stage": f"seed{seed}_open_residual_predictions",
        "seed": seed,
        "candidate": {
            "checkpoint": str(weights), "checkpoint_sha256": file_sha256(weights),
            "trunk_seed": int(candidate.seed), "head_seed": seed,
            "open_residual": metadata,
            "predict_batch_size": batch_size,
        },
        "frozen_baseline": {
            "weights": str(base_weights), "weights_sha256": base_sha,
            "replay_run": str(REPLAY_RUN),
            "state_sha256": candidate.open_residual_metadata.get("base_state_sha256"),
        },
        "protocol": {
            "windows": [record["window"] for record in window_records],
            "keys": "archived replay keys, identical row order and day chunking (chunk_days=11)",
            "sections": list(plan["data"]["sample_times"]),
            "reuse_rule": "candidate scores outside 09:30 are the frozen V2 replay scores; the 09:30 "
                          "section is trunk + residual head in the checkpoint target space",
            "labels_read": False,
            "local_approximation": True,
        },
        "windows": window_records,
        "combined": {"path": str(output_root / "predictions_full_development.parquet"),
                     "rows": int(len(full)),
                     "sha256": file_sha256(output_root / "predictions_full_development.parquet")},
        "checks": checks,
        "all_passed": bool(all(item["passed"] for item in checks)),
        "resources": {"wall_seconds": float(time.monotonic() - started),
                      **device_resources(candidate.device)},
    }
    if not evidence["all_passed"]:
        raise ValueError("prediction evidence checks failed")
    write_json(output_root / "prediction_checks.json", evidence)
    print(json.dumps({"exit_code": 0, "output_root": str(output_root), "seed": seed,
                      "all_passed": evidence["all_passed"], "combined_rows": int(len(full))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
