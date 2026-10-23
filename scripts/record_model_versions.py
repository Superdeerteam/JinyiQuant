"""Record V0/V1 model versions from two completed runs and their paired daily evidence.

V0 keeps the raw official training label; V1 only enables the existing full-timestamp
label cross-sectional z-score rescaled to the centered training-pool variance. Network,
inputs, training dates, steps, seed and budget are compared before anything is recorded,
the trained networks are checked against the archived seed-11 arms, and the already
recorded style-residual replay supplies the style-adjusted comparison without rerunning it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.io import file_sha256, read_json, utc_now, write_json

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = {"experiment_id", "hypothesis", "output_root"}
SHARED = ("data", "splits", "budget", "acceptance", "evaluation", "logging", "seed", "evaluation_split")
CSZ = "timestamp_zscore_train_rms_matched"
ARCHIVE = ROOT / "modified_tlob/label_cszscore_20261018.json"
RISK = ROOT / "modified_tlob/risk_neutralization_20261018.json"
ARCHIVED_ARMS = {
    "v0": ROOT / "runs/fixed-seed-ensemble-20261017/20261017T064944.073834Z_193b03ead7ff/fits/"
                "research_a_s11/modified_tlob_weights.json",
    "v1": ROOT / "runs/label-cszscore-20261018/20261018T104425.466761Z_1b8039d5f4e1/fits/"
                "research_a_s11/modified_tlob_weights.json"}
METRICS = ("rank_ic_mean", "ic_mean", "rank_ic_ir", "long_short_return_mean", "long_short_sharpe",
           "turnover_mean", "min_daily_long_short_return", "day_count", "sample_count")
STYLE_METRICS = ("rank_ic_mean", "rank_ic_ir", "long_short_return_mean", "long_short_sharpe",
                 "turnover_mean", "min_daily_long_short_return")
PAIRED = ("rank_ic", "long_short_return", "turnover")


def assert_only_label_transform_changes(v0, v1):
    if {key for key in v0 if v0[key] != v1.get(key)} != IDENTITY | {"model"}:
        raise ValueError("V0/V1 differ outside experiment identity and the model block")
    model_keys = {key for key in v0["model"] if v0["model"][key] != v1["model"].get(key)}
    if model_keys != {"target_centering"}:
        raise ValueError(f"Only model.target_centering may change, saw {sorted(model_keys)}")
    if (v0["model"]["target_centering"], v1["model"]["target_centering"]) != ("none", CSZ):
        raise ValueError("Version labels are not none -> timestamp_zscore_train_rms_matched")
    for key in SHARED:
        if v0[key] != v1[key]:
            raise ValueError(f"V0/V1 must keep {key} identical")


def numeric_tensors(path):
    out = {}

    def walk(value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, prefix + "/" + key)
        elif isinstance(value, list) and value:
            try:
                out[prefix] = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                return
    walk(read_json(path))
    return out


def reproduction(label, run_dir):
    archived, current = numeric_tensors(ARCHIVED_ARMS[label]), numeric_tensors(
        run_dir / "modified_tlob_weights.json")
    shared = [key for key in set(archived) & set(current) if archived[key].shape == current[key].shape]
    if not shared:
        raise ValueError(f"{label}: no comparable numeric tensors with the archived arm")
    difference = max(float(np.max(np.abs(archived[key] - current[key]))) for key in shared)
    return {"archived_arm": ARCHIVED_ARMS[label].relative_to(ROOT).as_posix(),
            "archived_arm_sha256": file_sha256(ARCHIVED_ARMS[label]),
            "network_weights_bit_identical": difference == 0.0, "max_abs_difference": difference,
            "tensors": len(shared)}


def style_adjusted():
    recorded = read_json(RISK)
    if recorded["experiment_id"] != "risk-neutralization-20261018" or recorded["seeds"] != [11, 29, 47]:
        raise ValueError("Style-residual replay must be the frozen three-seed record")
    windows = {}
    for name in ("research_a", "research_b"):
        seed_mean = recorded["windows"][name]["seed_mean"]
        entry = {}
        for arm, scheme in (("v0", "ref10"), ("v1", "cand_csz")):
            for version in ("raw", "processed"):
                entry[f"{arm}_{version}"] = {key: seed_mean[scheme][version][key] for key in STYLE_METRICS}
            raw_rank_ic = seed_mean[scheme]["raw"]["rank_ic_mean"]
            entry[f"{arm}_rank_ic_retained"] = (abs(seed_mean[scheme]["processed"]["rank_ic_mean"])
                                                / abs(raw_rank_ic)) if raw_rank_ic else None
            entry[f"{arm}_raw_rank_ic_positive"] = raw_rank_ic > 0
        delta = recorded["windows"][name]["cross_scheme"]["cand_csz_minus_ref10"]["processed"]["seed_mean_delta"]
        entry["v1_minus_v0_processed"] = {key: delta.get(key) for key in
                                          ("rank_ic_mean", "rank_ic_ir", "long_short_return_mean",
                                           "long_short_sharpe", "turnover_mean")}
        windows[name] = entry
    return {"source": {"path": RISK.relative_to(ROOT).as_posix(), "sha256": file_sha256(RISK),
                       "method": recorded["risk_model"]["exposure_date_rule"],
                       "scope": recorded["scope"], "seeds": recorded["seeds"]},
            "windows": windows,
            "note": "Already-recorded three-seed style-residual replay; V0 is ref10 and V1 is cand_csz. "
                    "It is reused, not recomputed; the trained networks reproduce those archived arms."}


def training_events(run_dir):
    events = [json.loads(line) for line in
              (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    summaries = [event["context"] for event in events
                 if event["stage"] == "training" and event["event_type"] == "progress"
                 and "steps" in event["context"]]
    if len(summaries) != 1:
        raise ValueError(f"Expected one training summary event, saw {len(summaries)}")
    fitted = [event["context"] for event in events if event["event_type"] == "target_centering_fitted"]
    return summaries[0], (fitted[-1] if fitted else None)


def version_entry(label, config_path, run_dir):
    config = read_json(run_dir / "config.json")
    manifest = read_json(run_dir / "manifest.json")
    if manifest["status"] != "completed":
        raise ValueError(f"{label} run is not completed: {manifest['status']}")
    progress, fitted = training_events(run_dir)
    metrics = {key: read_json(run_dir / "metrics.json")[key] for key in METRICS}
    return {
        "config": {"path": config_path.relative_to(ROOT).as_posix(), "sha256": file_sha256(config_path)},
        "run": run_dir.relative_to(ROOT).as_posix(), "run_id": run_dir.name,
        "weights_sha256": file_sha256(run_dir / "modified_tlob_weights.json"),
        "label_semantics": {"target_centering": config["model"]["target_centering"],
                            "target_scaling": config["model"]["target_scaling"],
                            "prediction_units": "relative_return_score"
                            if config["model"]["target_centering"] != "none" else "original_label_return"},
        "training": {"steps": progress["steps"], "samples_seen": progress["samples_seen"],
                     "samples": progress["samples"], "loss": progress["loss"],
                     "target_scale": progress["target_scale"],
                     "target_centering_stats": None if fitted is None else {
                         key: fitted[key] for key in
                         ("method", "c", "centered_std", "transformed_std", "fallback_mean",
                          "timestamps", "rows", "zero_variance_timestamps")}},
        "metrics": metrics,
    }


def paired_daily(v0_run, v1_run):
    left = pd.read_csv(v0_run / "daily.csv", parse_dates=["date"]).set_index("date")
    right = pd.read_csv(v1_run / "daily.csv", parse_dates=["date"]).set_index("date")
    common = left.index.intersection(right.index)
    if len(common) < 2:
        raise ValueError("Paired comparison needs at least two common evaluation days")
    rows = []
    for date in common:
        row = {"date": date.strftime("%Y-%m-%d")}
        for metric in PAIRED:
            row[metric + "_delta"] = (None if pd.isna(left.loc[date, metric]) or pd.isna(right.loc[date, metric])
                                      else float(right.loc[date, metric] - left.loc[date, metric]))
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("date")
    mean = {metric: (None if frame[metric + "_delta"].isna().all()
                     else float(frame[metric + "_delta"].mean())) for metric in PAIRED}
    worst = {}
    for metric in ("long_short_return", "rank_ic"):
        delta = frame[metric + "_delta"].dropna()
        worst[metric] = None if delta.empty else {"date": str(delta.idxmin()), "delta": float(delta.min())}
    return {"days": len(common), "daily": rows, "mean_delta": mean, "worst_day_delta": worst}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v0-run", type=Path, required=True)
    parser.add_argument("--v1-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v0_run, v1_run = args.v0_run.resolve(), args.v1_run.resolve()
    assert_only_label_transform_changes(*(read_json(run / "config.json") for run in (v0_run, v1_run)))
    v0 = version_entry("v0", ROOT / "configs/model_v0.json", v0_run)
    v1 = version_entry("v1", ROOT / "configs/model_v1.json", v1_run)
    archive = read_json(ARCHIVE)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite a recorded version file: {args.output}")
    record = {
        "schema_version": 1, "recorded_at_utc": utc_now(), "experiment_id": "model-versions-20261023",
        "policy": "V0 and V1 share network, inputs, training dates, steps, seed and budget; only "
                  "model.target_centering changes, so this comparison isolates the training label.",
        "versions": {"v0": v0, "v1": v1}, "comparison": paired_daily(v0_run, v1_run),
        "style_adjusted": style_adjusted(),
        "reproduction": {"v0": reproduction("v0", v0_run), "v1": reproduction("v1", v1_run)},
        "archived_multi_seed_evidence": {
            "path": ARCHIVE.relative_to(ROOT).as_posix(), "sha256": file_sha256(ARCHIVE),
            "conclusion": archive["conclusion"], "candidate_status": archive["candidate_status"],
            "note": "Archived three-seed/two-window CSZ evidence is the uncertainty reference; this "
                    "single-seed pair adds a same-pipeline version record, not independent confirmation."},
        "limits": [
            "One seed and one evaluation window in the new pair; paired daily differences are not "
            "independent confirmation.",
            "Local RankIC/long-short/turnover are approximations, not the official platform score.",
            "The archived CSZ round failed its worst-day guardrail on raw scores; V1 is formed by user "
            "direction and is not recorded as a gate pass.",
            "Style residualization is the frozen local proxy, not the official M.jyc_eval preprocessing.",
            "Confirmation, locked and legacy final ranges were not opened; platform limits remain unverified."],
    }
    write_json(args.output, record)
    print(json.dumps({"output": str(args.output), "mean_delta": record["comparison"]["mean_delta"],
                      "reproduction": {k: v["network_weights_bit_identical"]
                                       for k, v in record["reproduction"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
