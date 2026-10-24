"""Fingerprint the frozen weight sets and prove which training window each one encodes.

Identity is established from two independent sources: the checkpoint metadata (saved
target scale/mean, label scheme, preprocessing) and a byte-level tensor comparison.
The target-scale fingerprint is recomputed from the official labels for the candidate
training windows, so a checkpoint cannot silently claim a training window it did not use.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jinyiquant.data import select_targets  # noqa: E402
from jinyiquant.io import file_sha256, read_json  # noqa: E402


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def checkpoint_tensors(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tensors = {}
    for key, value in payload["state_dict"].items():
        dtype = np.float32 if value.get("dtype", "float32").endswith("32") else np.float64
        tensors[key] = np.asarray(value["data"], dtype=dtype)
    return tensors


def tensor_digest(tensors: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(tensors):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(tensors[key]).tobytes())
    return digest.hexdigest()


def label_months(labels_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for file in sorted(labels_path.glob("*.parquet")):
        stamp = pd.Timestamp(file.stem.split("_")[-1] + "-01")
        if stamp < end and stamp + pd.offsets.MonthBegin(1) > start:
            frames.append(pd.read_parquet(file, columns=["date", "instrument", "vwap_return"]))
    if not frames:
        raise ValueError(f"No label months overlap {start}..{end}")
    labels = pd.concat(frames, ignore_index=True)
    labels["date"] = pd.to_datetime(labels["date"])
    return labels.rename(columns={"vwap_return": "target"})


def fingerprint(labels_path: Path, instruments_path: Path, window: dict, sample_times: list[str],
                horizon: int) -> dict:
    """Recompute the fitted raw-target moments exactly as the training pool selected them."""
    start, end = pd.Timestamp(window["train_start"]), pd.Timestamp(window["train_end"])
    labels = label_months(labels_path, start, end)
    universe = pd.read_parquet(instruments_path, columns=["date", "instrument"])
    universe = universe.loc[(universe.date >= start) & (universe.date < end)]
    labels = labels.merge(universe, on=["date", "instrument"], how="inner")
    selected = select_targets(labels, {"start": start, "end": end}, horizon, sample_times=sample_times)
    values = np.asarray(selected.target, dtype=np.float32).astype(np.float64)
    return {"window": window["name"], "samples": int(len(values)),
            "target_std": float(values.std()), "target_mean": float(values.mean()),
            "selection": selected.attrs.get("selection", {})}


def relative_error(value: float, expected: float) -> float:
    return abs(value - expected) / max(abs(expected), 1e-12)


def describe(name: str, path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model_config", {})
    preprocessing = payload.get("preprocessing", {})
    tensors = checkpoint_tensors(path)
    params = sum(int(tensor.size) for tensor in tensors.values())
    stats = payload.get("target_centering_stats") or {}
    return {
        "name": name, "path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size,
        "schema_version": payload.get("schema_version"), "label": payload.get("label"),
        "prediction_units": payload.get("prediction_units"), "target_centering": payload.get("target_centering"),
        "target_scale": payload.get("target_scale"), "training_mean": payload.get("training_mean"),
        "seed": payload.get("seed"), "torch_version": payload.get("torch_version"),
        "max_steps": model.get("max_steps"), "target_scaling": model.get("target_scaling"),
        "price_scale": preprocessing.get("price_scale"), "lookback": preprocessing.get("lookback"),
        "history_days": preprocessing.get("history_days"), "sample_times": preprocessing.get("sample_times"),
        "centering_method": stats.get("method"), "transformed_std": stats.get("transformed_std"),
        "parameters": params, "tensor_digest": tensor_digest(tensors), "tensors": len(tensors),
    }


def package_claim(path: Path, entry: dict) -> dict:
    directory = path.parent
    claim = {"package_dir": str(directory)}
    contract = directory / "submission_config.json"
    if contract.is_file():
        value = read_json(contract)
        claim["weights_sha256"] = value.get("weights_sha256")
        claim["weights_sha256_match"] = value.get("weights_sha256") == entry["sha256"]
        claim["platform_price_scale"] = value.get("platform_price_scale")
        claim["history_policy"] = value.get("history_policy")
    provenance = directory / "provenance.json"
    if provenance.is_file():
        value = read_json(provenance)
        flat = value.get("flat_package", {})
        claim["provenance_training_run_id"] = flat.get("training_run_id")
        claim["provenance_source_commit"] = flat.get("source_commit")
        claim["provenance_weights_changed"] = flat.get("weights_changed")
        claim["provenance_created_at"] = flat.get("created_at")
        recorded = flat.get("payload_sha256", {})
        observed = {name: file_sha256(directory / name) if (directory / name).is_file() else None
                    for name in recorded}
        claim["payload_matches"] = sum(observed[name] == digest for name, digest in recorded.items())
        claim["payload_total"] = len(recorded)
        claim["payload_mismatches"] = [name for name, digest in recorded.items()
                                       if observed[name] != digest]
    return claim


def run_dir_claims(path: Path) -> dict:
    run_dir = path.parent if path.name.startswith("modified_tlob_weights") else path
    config = run_dir / "config.json"
    if not config.is_file():
        return {}
    value = read_json(config)
    return {"run_dir": str(run_dir), "experiment_id": value.get("experiment_id"),
            "seed": value.get("seed"), "train": value.get("splits", {}).get("train"),
            "validation": value.get("splits", {}).get("validation"),
            "model": {key: value.get("model", {}).get(key) for key in
                      ("max_steps", "target_scaling", "target_centering")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "modified_tlob/model_version_identity_20261023.json")
    parser.add_argument("--table", type=Path,
                        default=ROOT / "modified_tlob/model_version_identity_20261023.csv")
    args = parser.parse_args(argv)
    plan = read_json(args.plan)
    base = args.plan.resolve().parent
    data = plan["data"]
    labels_path = resolve(base, data["labels_path"])
    windows = plan["training_windows"]
    instruments_path = resolve(base, data["instruments_path"])
    prints = {window["name"]: fingerprint(labels_path, instruments_path, window, data["sample_times"],
                                          data["label_horizon_minutes"]) for window in windows}
    entries, held = [], {}
    for candidate in plan["identity_candidates"]:
        path = resolve(base, candidate["path"])
        entry = describe(candidate["name"], path)
        tensors = checkpoint_tensors(path)
        held[candidate["name"]] = tensors
        best = min(windows, key=lambda window: relative_error(
            entry["target_scale"], prints[window["name"]]["target_std"]))
        fingerprint_entry = prints[best["name"]]
        entry["matched_training_window"] = best["name"]
        entry["matched_target_std"] = fingerprint_entry["target_std"]
        entry["std_relative_error"] = relative_error(entry["target_scale"], fingerprint_entry["target_std"])
        entry["matched_target_mean"] = fingerprint_entry["target_mean"]
        entry["mean_error"] = abs(entry["training_mean"] - fingerprint_entry["target_mean"])
        entry["fingerprint_accepted"] = entry["std_relative_error"] < 1e-6
        entry["mean_matches_raw_label"] = (entry["target_centering"] == "none"
                                           and entry["mean_error"] < 1e-9)
        entry["package"] = package_claim(path, entry)
        entry["run"] = run_dir_claims(path)
        entries.append(entry)
    matrix = {}
    for left in held:
        for right in held:
            if left >= right:
                continue
            keys_equal = set(held[left]) == set(held[right])
            if not keys_equal:
                matrix[f"{left}|{right}"] = {"keys_equal": False}
                continue
            maximum, different = 0.0, 0
            for key in held[left]:
                if held[left][key].shape != held[right][key].shape:
                    keys_equal = False
                    break
                difference = np.abs(held[left][key].astype(np.float64) - held[right][key].astype(np.float64))
                maximum = max(maximum, float(difference.max()) if difference.size else 0.0)
                different += int(np.count_nonzero(difference))
            matrix[f"{left}|{right}"] = {"keys_equal": keys_equal, "max_abs_difference": maximum,
                                         "different_values": different, "identical": maximum == 0.0}
    progression_path = ROOT / "modified_tlob/training_progression_20261015.json"
    progression = read_json(progression_path)
    chosen = next((run for run in progression["archive_audit"]["actual_runs"]
                   if run["run_id"] == "20261015T104057.017178Z_41fc2c056943"), None)
    submitted = next(entry for entry in entries if entry["name"] == "submit_v0_package")
    record = {
        "schema_version": 1,
        "experiment_id": "model-version-identity-20261023",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "fingerprints": prints,
        "entries": entries,
        "tensor_pairs": matrix,
        "submitted_identity": {
            "weights_sha256": submitted["sha256"],
            "target_scale": submitted["target_scale"],
            "training_mean": submitted["training_mean"],
            "matched_training_window": submitted["matched_training_window"],
            "progression_record": progression_path.relative_to(ROOT).as_posix(),
            "progression_run": chosen,
            "progression_config": {
                key: progression.get("selected_research_candidate", {}).get("config", {}).get(key)
                for key in ("experiment_id", "seed", "splits", "model")},
            "user_reported_first_public_score": 0.18503,
            "score_verified_locally": False,
            "claim": ("The seven-file package received on 2026-10-23 (identical weight sha256) and the "
                      "dist/v0-submission directory both carry this checkpoint; the recorded run id points to "
                      "the step3c July_25_28, ten-day, eight-time, 1500-step seed-11 run whose training split is "
                      "2023-07-11..2023-07-25. That run directory itself is no longer on disk, so provenance "
                      "rests on the recorded run id, the saved target scale/mean fingerprint, and the received copy."),
        },
    }
    record["submitted_identity"]["progression_run_matches_split"] = bool(
        chosen and chosen.get("path", "").find("july_25_28_s11_d10_t8") >= 0)
    repair = ROOT / "modified_tlob/v0_package_repair_20261023.json"
    if repair.is_file():
        record["package_repairs"] = read_json(repair)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    table = pd.DataFrame(entries)
    table["payload_matches"] = [entry["package"].get("payload_matches") for entry in entries]
    table["payload_total"] = [entry["package"].get("payload_total") for entry in entries]
    columns = ["name", "sha256", "bytes", "schema_version", "label", "target_centering", "max_steps",
               "target_scale", "training_mean", "matched_training_window", "std_relative_error", "mean_error",
               "fingerprint_accepted", "payload_matches", "payload_total", "torch_version", "parameters", "path"]
    table[columns].to_csv(args.table, index=False, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
