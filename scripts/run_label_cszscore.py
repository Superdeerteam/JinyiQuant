"""Fit the 10-day/1500-step recipe on full-timestamp z-score labels rescaled to the centered-pool variance.

The only training-target change this round is ``y* = c * q`` where ``q`` is the
cross-sectional z-score of the official label within every complete training timestamp
and ``c`` is one training-pool constant matching the centered labels' standard
deviation. The target scale stays the raw-label standard deviation, inference emits
``network(x) * target_scale`` without adding ``c`` back or reading labels, and every
evaluation keeps the official raw labels. The formal comparison is candidate minus the
archived raw-label 10-day reference under the unchanged gates. The same-seed contrast
against the timestamp-centered control is reported separately and decides nothing.
"""

import argparse
import ast
import copy
import difflib
import hashlib
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.checks import check
from jinyiquant.config import EvaluationConfig, RunConfig
from jinyiquant.evaluation import evaluate
from jinyiquant.io import digest, file_sha256, read_json, utc_now, write_json
from jinyiquant.runner import _answer_coverage_check, save_checks
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir
from modified_tlob.adapter import timestamp_zscore_train_rms_matched
from modified_tlob.coverage import DELTA_METRICS, arm_exposure, coverage_decision
from modified_tlob.ensemble import SEEDS
from scripts.run_coverage_step_scaling import analysis, arm_means
from scripts.run_label_centering import (
    QLIB_PRECEDENT,
    assert_only_centering_changes,
    reference_arm,
)
from scripts.run_seed_ensemble import claim_training, complete_correlations, frame_digest
from scripts.run_training_coverage_arms import (
    TRAINING_IDENTITY_FILES,
    BudgetExceeded,
    fit_worker,
    infer_worker,
    score_predictions,
    supervise,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "label-cszscore-20261018"
KEYS = ["date", "instrument"]
WINDOWS = ("research_a", "research_b")
REFERENCE, CENTERED, CANDIDATE = "ref10", "cand_centered", "cand_csz"
STEPS = 1500
BATCH = 32
MODE = "timestamp_zscore_train_rms_matched"
FULL_SEEDS = tuple(SEEDS)
PROGRESS_SEEDS = (11,)
FROZEN = ROOT / "configs/training_coverage_20261017_v4_frozen.json"
REFERENCE_RUN = ROOT / "runs/fixed-seed-ensemble-20261017/20261017T064944.073834Z_193b03ead7ff"
SCORING_RUN = ROOT / "runs/fixed-seed-ensemble-20261017/analysis-recovery/20261017T091219.300500Z_7c40b5b5d26d"
CENTERING_PLAN = ROOT / "configs/label_centering_20261018_plan.json"
CENTERING_RECORD = ROOT / "modified_tlob/label_centering_20261018.json"
PLAN = ROOT / "configs/label_cszscore_20261018_plan.json"
RUNS = ROOT / "runs" / EXPERIMENT
TREATMENT_FILES = ("modified_tlob/adapter.py", "src/jinyiquant/config.py")
SCORING_METRICS = ("rank_ic", "long_short_return", "turnover", "long_return", "short_return")
QLIB_NOTE = ("Qlib applies CSZScoreNorm per datetime; this round only borrows that public cross-sectional "
             "z-score template. Matching the centered pool variance with one constant c is this project's "
             "control design, not a copied Qlib configuration.")


def job_recorder(output):
    rec = Recorder(output, EXPERIMENT)
    rec.run_id = output.name + "_" + digest(str(output.resolve()))[:12]
    return rec


def fit_dir(root, name, seed):
    return Path(root) / "fits" / f"{name}_s{seed}"


def evaluation_config(window):
    return EvaluationConfig.model_validate(window["arms"][REFERENCE]["config"]["evaluation"])


# ------------------------------------------------------------------ target transform contract


def assert_only_target_transform_changes(reference, candidate, label):
    """The candidate differs from the raw-label recipe only in the training target mode."""
    differing = {key for key in set(reference) | set(candidate) if reference.get(key) != candidate.get(key)}
    if differing - {"experiment_id", "hypothesis", "output_root", "budget", "model"}:
        raise ValueError(f"{label}: unexpected recipe differences")
    model_keys = sorted(key for key in set(reference["model"]) | set(candidate["model"])
                        if reference["model"].get(key) != candidate["model"].get(key))
    if model_keys != ["target_centering"]:
        raise ValueError(f"{label}: only model.target_centering may change, saw {model_keys}")
    if (reference["model"].get("target_centering", "none"), candidate["model"].get("target_centering")) != (
            "none", MODE):
        raise ValueError(f"{label}: target transform is not none -> {MODE}")
    if candidate["model"]["target_scaling"] != "train_std" or candidate["model"]["max_steps"] != STEPS:
        raise ValueError(f"{label}: raw-label train_std scaling and {STEPS} steps must stay fixed")
    if set(reference["budget"]) != set(candidate["budget"]):
        raise ValueError(f"{label}: budget schema must not change")


def csz_label_stats(frame, expected):
    """Planned transform statistics from the frozen training-target artifact."""
    keys = frame[KEYS].reset_index(drop=True)
    target = frame["target"].to_numpy(dtype=np.float32)
    selected, stats = timestamp_zscore_train_rms_matched(keys, target, np.ones(len(target), dtype=bool))
    if len(selected) != len(target):
        raise ValueError("Planned transform dropped training rows")
    if stats["timestamps"] != 80 or stats["rows"] != len(target) or stats["mask_excluded_rows"] != 0:
        raise ValueError("Planned transform does not cover every full training timestamp")
    if stats["timestamp_means_sha256"] != expected["timestamp_means_sha256"]:
        raise ValueError("Planned timestamp means differ from the centered control")
    if not np.isclose(stats["raw_std"], expected["raw_std"], atol=1e-14, rtol=1e-10):
        raise ValueError("Planned raw target standard deviation differs from the centered control")
    if not np.isclose(stats["centered_std"], expected["centered_std"], atol=1e-14, rtol=1e-10):
        raise ValueError("Planned centered target standard deviation differs from the centered control")
    if not np.isclose(stats["transformed_std"], stats["centered_std"], atol=1e-15, rtol=1e-10):
        raise ValueError("Planned transformed variance does not match the centered pool variance")
    return stats


# ------------------------------------------------------------------ reused arms


def control_arm(centering_plan, record, reference, name):
    """Prove the six completed centered fits and predictions are the reused control."""
    execution = Path(record["execution"]["run_dir"])
    recovery = record["execution"].get("recovery") or {}
    source = Path(recovery["source_run"]) if recovery else execution
    declared = next(w for w in centering_plan["windows"] if w["name"] == name)["arms"][CENTERED]
    expected = centering_plan["label_centering"]["per_window"][name]
    rows = {(row["window"], int(row["seed"])): row for row in record["execution"]["fits"]}
    entry = {"days": 10, "steps": STEPS, "samples": declared["samples"],
             "frozen_identity": dict(declared["frozen_identity"]), "artifact": declared["artifact"],
             "artifact_sha256": declared["artifact_sha256"], "train_dates": list(declared["train_dates"]),
             "train_end_exclusive": declared["train_end_exclusive"], "config": copy.deepcopy(declared["config"]),
             "per_seed": {}}
    reference_predictions = reference["per_seed"]
    for seed in FULL_SEEDS:
        fitting = fit_dir(source, name, seed)
        row = rows.get((name, seed))
        if row is None or Path(row["run"]).resolve() != fitting.resolve():
            raise ValueError(f"{name} seed {seed}: centered fit is not the one recorded by the completed run")
        weights = fitting / "model.pt"
        if file_sha256(weights) != row["weights_sha256"]:
            raise ValueError(f"{name} seed {seed}: centered weights changed")
        identity = read_json(fitting / "training_identity.json")
        frozen = {**entry["frozen_identity"], "dates": entry["train_dates"]}
        if {key: identity.get(key) for key in frozen} != frozen:
            raise ValueError(f"{name} seed {seed}: centered fit used different training keys or labels")
        archived = read_json(fitting / "config.json")
        differing = {key for key in set(archived) | set(entry["config"]) if archived.get(key) != entry["config"].get(key)}
        if differing - {"seed"}:
            raise ValueError(f"{name} seed {seed}: centered recipe differs from the frozen control")
        guarded = copy.deepcopy(archived)
        guarded["seed"] = entry["config"]["seed"]
        assert_only_centering_changes(reference["config"], guarded, name)
        training = read_json(fitting / "training.json")
        stats = training.get("target_centering_stats")
        if training.get("target_centering") != "timestamp_mean_train" or not isinstance(stats, dict):
            raise ValueError(f"{name} seed {seed}: centered fit did not record timestamp centering")
        if stats.get("method") != "timestamp_mean_train":
            raise ValueError(f"{name} seed {seed}: centered statistics have the wrong method")
        if training["steps"] != STEPS or training["samples"] != entry["samples"]:
            raise ValueError(f"{name} seed {seed}: centered fit did not complete the declared budget")
        if stats["timestamps"] != 80 or stats["rows"] != entry["samples"] or stats["mask_excluded_rows"] != 0:
            raise ValueError(f"{name} seed {seed}: centered fit did not use every row and timestamp")
        if not np.isclose(training["target_scale"], expected["raw_std"], atol=1e-14, rtol=1e-10):
            raise ValueError(f"{name} seed {seed}: centered fit target scale is not the raw-label std")
        if not np.isclose(training["centered_target_std"], expected["centered_std"], atol=1e-14, rtol=1e-10):
            raise ValueError(f"{name} seed {seed}: centered fit statistics differ from the frozen control")
        predictions = source / "inference_full" / name / f"seed{seed}_predictions.parquet"
        if not predictions.is_file():
            raise FileNotFoundError(predictions)
        frame = pd.read_parquet(predictions)
        if frame_digest(frame[KEYS]) != frame_digest(pd.read_parquet(reference_predictions[str(seed)]["predictions"])[KEYS]):
            raise ValueError(f"{name} seed {seed}: centered predictions use different evaluation keys")
        entry["per_seed"][str(seed)] = {
            "fit_dir": str(fitting), "weights": str(weights), "weights_sha256": file_sha256(weights),
            "predictions": str(predictions), "predictions_sha256": file_sha256(predictions),
            "training": {key: training[key] for key in
                         ("loss", "target_scale", "training_std", "centered_target_std", "steps", "samples",
                          "samples_seen", "loss_units", "prediction_units")},
            "centering": {key: stats[key] for key in
                          ("timestamps", "rows", "min_rows_per_timestamp", "mask_excluded_rows", "raw_std",
                           "centered_std", "max_abs_centered_timestamp_mean")}}
    return entry


def source_identity(raw_archives, control_archives, delta_path):
    """Prove the reused fits share the training code and the new branch is the only addition."""
    result, non_treatment = {}, {}
    for archive_path in sorted({str(path) for path in [*raw_archives, *control_archives]}):
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            rows = {}
            for name in TRAINING_IDENTITY_FILES:
                current = ROOT / name
                if name not in names or not current.is_file():
                    raise ValueError(f"Training source snapshot cannot be compared: {name}")
                archived = hashlib.sha256(archive.read(name)).hexdigest()
                rows[name] = {"archived_sha256": archived, "current_sha256": file_sha256(current),
                              "identical": archived == file_sha256(current),
                              "treatment_file": name in TREATMENT_FILES}
                if name not in TREATMENT_FILES and archived != rows[name]["current_sha256"]:
                    raise ValueError(f"Training code differs from an archived snapshot: {name} in {archive_path}")
        signatures = {tuple((name, row["archived_sha256"]) for name, row in rows.items()
                            if name not in TREATMENT_FILES)}
        if len(signatures) != 1:
            raise ValueError("The reused snapshots do not share one non-treatment training source")
        non_treatment.update({name: row for name, row in rows.items() if name not in TREATMENT_FILES})
        result[archive_path] = {"archive_sha256": file_sha256(Path(archive_path)), "files": rows}
    with zipfile.ZipFile(control_archives[0]) as archive:
        old_adapter = archive.read(TREATMENT_FILES[0]).decode("utf-8").replace("\r\n", "\n")
        old_config = archive.read(TREATMENT_FILES[1]).decode("utf-8").replace("\r\n", "\n")
    current = {name: (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")
               for name in TREATMENT_FILES}
    assert_csz_only_addition(old_adapter, current[TREATMENT_FILES[0]])
    assert_config_only_addition(old_config, current[TREATMENT_FILES[1]])
    diff = list(difflib.unified_diff(old_adapter.splitlines(), current[TREATMENT_FILES[0]].splitlines(),
                                     fromfile="centered/" + TREATMENT_FILES[0],
                                     tofile="current/" + TREATMENT_FILES[0], lineterm=""))
    diff += list(difflib.unified_diff(old_config.splitlines(), current[TREATMENT_FILES[1]].splitlines(),
                                      fromfile="centered/" + TREATMENT_FILES[1],
                                      tofile="current/" + TREATMENT_FILES[1], lineterm=""))
    write_json(delta_path, {
        "schema_version": 1, "treatment_files": list(TREATMENT_FILES),
        "archived_adapter_sha256": hashlib.sha256(old_adapter.encode()).hexdigest(),
        "current_adapter_sha256": file_sha256(ROOT / TREATMENT_FILES[0]),
        "archived_config_sha256": hashlib.sha256(old_config.encode()).hexdigest(),
        "current_config_sha256": file_sha256(ROOT / TREATMENT_FILES[1]),
        "diff_sha256": hashlib.sha256("\n".join(diff).encode()).hexdigest(), "diff": diff,
        "note": "The six raw-label and six centered fits share every non-treatment training file. The current "
                "adapter adds the declared CSZ transform as a new method and generalizes metadata/load dispatch; "
                "the centered transform block is byte-identical. time_loss_weighting=none keeps other branches "
                "inactive."})
    return result, non_treatment


def _methods(text):
    tree = ast.parse(text)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    return {node.name: ast.get_source_segment(text, node) for node in cls.body if isinstance(node, ast.FunctionDef)}


def assert_csz_only_addition(archived, current):
    """The centered transform is unchanged; only the new CSZ method branch is added."""
    marker = "        self.target_centering_stats = None\n"
    end_marker = "        return centered[mask].astype(np.float32), stats"
    start = archived.index(marker, archived.index("    def _training_targets"))
    end = archived.index(end_marker, start) + len(end_marker)
    block = archived[start:end]
    current_start = current.index(marker, current.index("    def _training_targets"))
    current_end = current.index(end_marker, current_start) + len(end_marker)
    if current[current_start:current_end] != block:
        raise ValueError("The centered training-target transform changed")
    old, new = _methods(archived), _methods(current)
    if set(new) - set(old) != {"_csz_training_targets"} or set(old) - set(new):
        raise ValueError("Unexpected method added to or removed from the adapter")
    allowed = {"__init__", "_training_targets", "fit", "_payload", "load"}
    changed = {name for name in old if name not in allowed and old[name] != new[name]}
    if changed:
        raise ValueError(f"Unexpected adapter changes outside the declared branch: {sorted(changed)}")
    if "timestamp_zscore_train_rms_matched" not in current or "timestamp_zscore_train_rms_matched" in archived:
        raise ValueError("The archived snapshot already has the CSZ branch or the current one is missing it")
    if "TARGET_CENTERING_MODES" not in current or "TARGET_CENTERING_MODES" in archived:
        raise ValueError("The adapter does not declare the new target-centering mode set")


def assert_config_only_addition(archived, current):
    removed = [line for line in difflib.unified_diff(archived.splitlines(), current.splitlines(), lineterm="")
               if line.startswith("-") and not line.startswith("---")]
    if len(removed) != 1 or "timestamp_mean_train" not in removed[0]:
        raise ValueError("Configuration change is not the declared target-centering enum extension")
    if MODE not in current:
        raise ValueError("The configuration enum is missing the CSZ mode")


# ------------------------------------------------------------------ plan freeze


def prepare(output_path=PLAN, *, frozen_path=FROZEN, output_root=RUNS,
            centering_plan_path=CENTERING_PLAN, centering_record_path=CENTERING_RECORD):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Frozen plans are immutable; select a new output name")
    frozen_path = Path(frozen_path).resolve()
    frozen = read_json(frozen_path)
    centering_plan_path = Path(centering_plan_path).resolve()
    centering_plan = read_json(centering_plan_path)
    centering_record = read_json(centering_record_path)
    if centering_record["plan"]["sha256"] != file_sha256(centering_plan_path):
        raise ValueError("The centered record was not produced by this frozen centering plan")
    if file_sha256(frozen["registry_path"]) != frozen["registry_sha256"]:
        raise ValueError("Shared data-role registration changed")
    if read_json(REFERENCE_RUN / "manifest.json")["status"] == "running":
        raise ValueError("The raw-label reference run is still running")
    run = new_run_dir(Path(output_root))
    rec = Recorder(run, EXPERIMENT)
    source = capture_source(run)
    write_json(run / "manifest.json", {"schema_version": 1, "run_id": run.name, "experiment_id": EXPERIMENT,
        "started_at": utc_now(), "status": "running", "new_training": 0, "final_views": 0, "mode": "prepare",
        "code": source, "environment": environment()})
    try:
        rec.event("run", "started", mode="prepare", steps=STEPS, seeds=list(FULL_SEEDS))
        limits = frozen["measured_budget"]["limits"]
        policy = copy.deepcopy(frozen["policy"])
        if (limits["max_rss_mb"], limits["wall_seconds"]) != (9472.0, 900.0):
            raise ValueError("The reused measured budget is not the frozen one")
        windows, inventory, planned = [], {}, {}
        raw_archives, control_archives = [], []
        for name in WINDOWS:
            centering_window = next(w for w in centering_plan["windows"] if w["name"] == name)
            scope = {"train_start": centering_window["train_start"],
                     "train_end_exclusive": centering_window["train_end_exclusive"],
                     "evaluation_start": centering_window["evaluation_start"],
                     "evaluation_end_exclusive": centering_window["evaluation_end_exclusive"],
                     "evaluation_dates": centering_window["evaluation_dates"]}
            reference = reference_arm(frozen, name)
            if reference["artifact"] != centering_window["arms"][REFERENCE]["artifact"]:
                raise ValueError(f"{name}: centering plan used a different training artifact")
            targets = Path(centering_window["targets"]["path"])
            if not targets.is_file() or file_sha256(targets) != centering_window["targets"]["sha256"]:
                raise ValueError(f"{name}: frozen evaluation targets changed")
            inventory[str(targets)] = centering_window["targets"]["sha256"]
            for row in reference["per_seed"].values():
                inventory[row["predictions"]] = row["predictions_sha256"]
                inventory[row["weights"]] = row["weights_sha256"]
                raw_archives.append(Path(row["fit_dir"]) / "source.zip")
            artifact = Path(reference["artifact"])
            if file_sha256(artifact) != reference["artifact_sha256"]:
                raise ValueError(f"{name}: frozen training target artifact changed")
            inventory[reference["artifact"]] = reference["artifact_sha256"]
            centering_stats = centering_plan["label_centering"]["per_window"][name]
            stats = csz_label_stats(pd.read_parquet(artifact), centering_stats)
            planned[name] = stats
            control = control_arm(centering_plan, centering_record, reference, name)
            control_archives.extend(Path(row["fit_dir"]) / "source.zip" for row in control["per_seed"].values())
            for row in control["per_seed"].values():
                inventory[row["predictions"]] = row["predictions_sha256"]
                inventory[row["weights"]] = row["weights_sha256"]
            assert_only_centering_changes(reference["config"], control["config"], name)
            candidate = {"days": 10, "steps": STEPS, "samples": reference["samples"],
                         "frozen_identity": dict(reference["frozen_identity"]),
                         "artifact": reference["artifact"], "artifact_sha256": reference["artifact_sha256"],
                         "train_dates": list(reference["train_dates"]),
                         "train_end_exclusive": reference["train_end_exclusive"],
                         "config": copy.deepcopy(reference["config"])}
            candidate["config"]["experiment_id"] = EXPERIMENT
            candidate["config"]["hypothesis"] = (
                "Cross-sectional timestamp z-scores rescaled by one training-pool constant to the centered "
                "labels' variance reduce cross-regime target-scale instability and improve out-of-sample ranking "
                "against the raw-label 10-day reference under the unchanged gates.")
            candidate["config"]["output_root"] = str(RUNS / "fits")
            candidate["config"]["model"]["target_centering"] = MODE
            candidate["config"]["budget"] = {"max_train_samples": reference["samples"],
                                             "max_rss_mb": limits["max_rss_mb"],
                                             "wall_seconds": limits["wall_seconds"]}
            assert_only_target_transform_changes(reference["config"], candidate["config"], name)
            RunConfig.model_validate(candidate["config"])
            arms = {REFERENCE: reference, CENTERED: control, CANDIDATE: candidate}
            windows.append({
                "name": name, **scope, "arms": arms,
                "exposure": {"batch_size": BATCH, "steps": STEPS,
                             "arms": {arm: arm_exposure(spec["samples"], steps=STEPS, batch_size=BATCH)
                                      for arm, spec in arms.items()},
                             "interpretation": "All three arms use 1500 x 32 = 48,000 sample exposures on the "
                                               "same training dates and seeds; only the training target differs."},
                "targets": {"path": str(targets), "sha256": inventory[str(targets)]},
                "fit_once_before_evaluation": True})
        identity_rows, non_treatment = source_identity(raw_archives, control_archives, run / "source_delta.json")
        inventory[str(run / "source_delta.json")] = file_sha256(run / "source_delta.json")
        inventory[str(centering_plan_path)] = file_sha256(centering_plan_path)
        inventory[str(Path(centering_record_path).resolve())] = file_sha256(Path(centering_record_path).resolve())
        plan = {
            "schema_version": 1, "experiment_id": EXPERIMENT, "created_at_utc": utc_now(),
            "hypothesis": "Standardizing the official label within each full training timestamp, then restoring "
                          "the centered training-pool variance with one constant, makes the target scale comparable "
                          "across market states and improves out-of-sample ranking against the raw-label 10-day "
                          "reference under the unchanged gates. Passing that comparison would not by itself show "
                          "the new standardization beats the centered control.",
            "only_factor": "training target raw official return -> float32(c q), where q is the full-timestamp "
                           "z-score and c = std(centered labels)/std(q), both ddof=0 over the complete training "
                           "pool; target_scale stays the raw-label std; network, inputs, steps, batch order, MSE, "
                           "seeds and dates fixed",
            "source_frozen": {"path": str(run / "source.zip"), "sha256": source["snapshot_sha256"],
                              "content_sha256": source["content_sha256"],
                              "files": {name: source["files"][name] for name in
                                        (*TRAINING_IDENTITY_FILES, *TREATMENT_FILES)}},
            "reference_source": {
                "run": str(REFERENCE_RUN), "manifest_status": read_json(REFERENCE_RUN / "manifest.json")["status"],
                "scoring_run": str(SCORING_RUN),
                "centering_plan": {"path": str(centering_plan_path), "sha256": file_sha256(centering_plan_path)},
                "centering_record": {"path": str(Path(centering_record_path).resolve()),
                                     "sha256": file_sha256(Path(centering_record_path).resolve())},
                "centering_run": centering_record["execution"]["run_dir"],
                "centering_fit_source": str(Path(centering_record["execution"]["recovery"]["source_run"])
                                            if centering_record["execution"].get("recovery") else
                                            Path(centering_record["execution"]["run_dir"])),
                "note": "The six raw-label fits and the six centered fits are reused, never refitted. Their "
                        "weights, predictions, training keys, labels and configs are checked before the freeze."},
            "recipe_fixed": {"raw12": True, "sample_times": 8, "lookback": 128, "loss": "MSE",
                             "target_scaling": "train_std",
                             "target_std_fit": "raw official selected training labels, full 10-day pool",
                             "learning_rate": 1e-4, "batch_size": BATCH, "steps": STEPS,
                             "sampler": "torch.randperm(seed) unchanged", "smoothing": "none",
                             "time_loss_weighting": "none"},
            "label_transform": {
                "mode": MODE,
                "formula": "r = float64(float32(y)); z = r - mean_t(r); q = z / std_t(r) if std_t(r) > 0 else 0; "
                           "c = std(z) / std(q); y* = float32(c q)",
                "cross_section": "one complete timestamp of the frozen 10-day training pool, all stocks and all "
                                 "eight sample times; never a random mini-batch",
                "ddof": 0,
                "pool_constant": "c is computed once from every selected training row before mini-batches and "
                                 "before the min-history mask",
                "zero_variance": "zero-variance timestamps keep all rows with q=0; the whole-pool std(q)=0 case "
                                 "fails before training",
                "target_scale": "raw official training-label standard deviation, unchanged from the reference",
                "inference": "network(x) * target_scale; c is never multiplied back and no future timestamp mean "
                             "or standard deviation is added",
                "fallback": "timestamp-centered training-label mean over the selected rows, identical to the "
                            "centered control; not re-estimated from y*",
                "evaluation": "official raw labels for every ranking, long-short, turnover and raw-label error; "
                              "centered-label error and the optimized transformed-target loss are named separately",
                "timestamps_per_window": {name: planned[name]["timestamps"] for name in WINDOWS},
                "per_window": planned,
                "public_precedent": {"url": QLIB_PRECEDENT[0], "use": QLIB_NOTE},
                "assumption_status": "minimal unverified assumption for this project; the public cross-sectional "
                                     "z-score precedent does not establish that variance matching improves this "
                                     "model or beats the centered control"},
            "windows": windows, "training_source_identity": identity_rows,
            "non_treatment_source": non_treatment,
            "source_delta": {"path": str(run / "source_delta.json"),
                             "sha256": inventory[str(run / "source_delta.json")],
                             "note": "Only adapter.py and config.py differ from the centered snapshot. The centered "
                                     "transform block is byte-identical; the CSZ method is the new branch."},
            "policy": policy, "calibration": frozen["calibration"], "statistics": policy["statistics"],
            "progress": {"seeds": list(PROGRESS_SEEDS),
                         "rule": "Seed 11 is printed and archived only as progress after all six fits; no decision "
                                 "is made until all three seeds and both windows are complete.",
                         "mandatory_full_replay": True},
            "steps": {"reference": STEPS, "centered": STEPS, "candidate": STEPS},
            "seeds": {"progress": list(PROGRESS_SEEDS), "full": list(FULL_SEEDS)},
            "deployment_seed": 11,
            "budget": {"effective_candidate_training": 6, "fit_attempts_max": 12,
                       "retry_policy": "At most one same-configuration, same-seed technical retry per incomplete "
                                       "fit; a completed fit is never rerun because of its score.",
                       "local_peak_rss_cap_mib": limits["max_rss_mb"],
                       "per_fit_wall_cap_seconds": limits["wall_seconds"],
                       "fit_stage_wall_cap_seconds": 9000.0,
                       "inference_chunk_days": 5, "inference_windows": len(WINDOWS),
                       "round_scope": "The 9000-second cap covers preparation, six fits, inference and both "
                                      "analyses; CUDA fits run one at a time."},
            "measured_budget": {"limits": limits, "source": str(frozen_path)},
            "registry_path": frozen["registry_path"], "registry_sha256": frozen["registry_sha256"],
            "inputs_sha256": inventory, "final_views": 0,
            "access_policy": "research_a/research_b are already-used research data; confirmation, locked and "
                             "legacy final ranges stay closed this round.",
            "notes": "Six mandatory CSZ fits. The six raw-label and six centered arms are reused, never refitted. "
                     "No other label transform, standardization strength, step count, seed or loss is searched.",
            "retained_version": "10-day raw-label reference (unchanged)",
        }
        checks = [
            check("reference_reuses_three_seeds_per_window",
                  all(len(window["arms"][REFERENCE]["per_seed"]) == len(FULL_SEEDS) for window in windows)),
            check("centered_control_reuses_three_seeds_per_window",
                  all(len(window["arms"][CENTERED]["per_seed"]) == len(FULL_SEEDS) for window in windows)),
            check("centered_control_uses_every_training_row_and_all_eight_times",
                  all(row["centering"]["mask_excluded_rows"] == 0 and row["centering"]["timestamps"] == 80
                      and row["centering"]["rows"] == window["arms"][CENTERED]["samples"]
                      for window in windows for row in window["arms"][CENTERED]["per_seed"].values())),
            check("candidate_differs_only_in_target_transform",
                  all(assert_only_target_transform_changes(
                      window["arms"][REFERENCE]["config"], window["arms"][CANDIDATE]["config"],
                      window["name"]) is None for window in windows)),
            check("planned_csz_statistics_cover_every_timestamp",
                  all(planned[window["name"]]["timestamps"] == 80
                      and planned[window["name"]]["rows"] == window["arms"][CANDIDATE]["samples"]
                      and planned[window["name"]]["mask_excluded_rows"] == 0 for window in windows)),
            check("planned_csz_variance_matches_centered_pool",
                  all(np.isclose(row["transformed_std"], row["centered_std"], atol=1e-15, rtol=1e-10)
                      and row["q_std"] > 0 and np.isfinite(row["c"]) for row in planned.values())),
            check("planned_csz_uses_float64_statistics_and_zero_variance_rule",
                  all(row["ddof"] == 0 and isinstance(row["per_timestamp"], list)
                      and len(row["per_timestamp"]) == 80
                      and all(isinstance(item["samples"], int) for item in row["per_timestamp"])
                      for row in planned.values())),
            check("unchanged_original_gain_floor_and_worst_day_tolerance",
                  np.isclose(policy["minimum_mean_gain"], 0.01164665094206816, atol=1e-15, rtol=0)
                  and np.isclose(policy["noise"]["min_daily_long_short_return"],
                                 0.00013896761043335348, atol=1e-18, rtol=0)),
            check("budget_freezes_six_effective_fits_and_twelve_starts",
                  (6, 12, 9000.0, 5) == (6, plan["budget"]["fit_attempts_max"],
                                         plan["budget"]["fit_stage_wall_cap_seconds"],
                                         plan["budget"]["inference_chunk_days"])),
            check("source_diff_is_the_declared_csz_branch_only",
                  file_sha256(run / "source_delta.json") == inventory[str(run / "source_delta.json")]),
        ]
        if not save_checks(run, checks):
            raise ValueError("Plan freeze checks failed")
        write_json(output_path, plan)
        write_json(run / "plan.json", plan)
        rec.event("prepare", "completed", plan=str(output_path), plan_sha256=file_sha256(output_path),
                  mandatory_fits=6, progress_seeds=list(PROGRESS_SEEDS))
        finish(run, "completed", 0, resources={"wall_seconds": 0.0})
    except BaseException as exc:
        rec.failure("prepare", exc)
        finish(run, "failed", 1, error={"reason": str(exc)})
        raise
    return run


# ------------------------------------------------------------------ worker supervision


def prepare_job_output(output, kind, *, allow_existing=False):
    """Create a worker directory; only an incomplete fit may be resumed in place."""
    output = Path(output)
    if output.exists():
        if not allow_existing or kind != "fit":
            raise FileExistsError(f"Worker output already exists: {output}")
        if (output / "model.pt").exists() or (output / "training.json").exists():
            raise ValueError("A completed fit cannot be rerun")
    else:
        output.mkdir(parents=True)


def launch(plan_path, output, kind, window_name, *, seed=None, paths=None, cap=None, allow_existing=False):
    prepare_job_output(output, kind, allow_existing=allow_existing)
    plan = read_json(plan_path)
    window = next(w for w in plan["windows"] if w["name"] == window_name)
    config = copy.deepcopy(window["arms"][CANDIDATE]["config"])
    if seed is not None:
        config["seed"] = seed
    write_json(output / "config.json", config)
    write_json(output / "job.json", {"kind": kind, "window": window_name, "arm": CANDIDATE, "seed": seed,
                                     "paths": paths or {}})
    rec = job_recorder(output)
    started = time.monotonic()
    write_json(output / "manifest.json", {"schema_version": 1, "run_id": rec.run_id, "experiment_id": EXPERIMENT,
        "started_at": utc_now(), "status": "running", "kind": kind, "window": window_name, "arm": CANDIDATE,
        "seed": seed, "plan_sha256": file_sha256(plan_path), "code": capture_source(output),
        "environment": environment(), "new_training": int(kind == "fit")})
    rec.event("run", "started", kind=kind, window=window_name, arm=CANDIDATE, seed=seed)
    caps = cap or {"wall_seconds": plan["budget"]["per_fit_wall_cap_seconds"],
                   "rss_mb": plan["budget"]["local_peak_rss_cap_mib"]}
    command = [sys.executable, "-m", "scripts.run_label_cszscore", "worker",
               str(Path(plan_path).resolve()), "--output", str(output)]
    try:
        result = supervise(command, output, cap_rss_mb=caps["rss_mb"], cap_seconds=caps["wall_seconds"],
                           rec=rec, seed=seed or FULL_SEEDS[0])
    except BudgetExceeded as exc:
        finish(output, "budget_exceeded", 124, resources={"wall_seconds": time.monotonic() - started},
               error={"reason": str(exc), "recovery": "stop; no silent budget extension or extra fits"})
        raise
    finish(output, "completed" if result["returncode"] == 0 else "failed", result["returncode"],
           resources={"wall_seconds": result["wall_seconds"], "peak_rss_mb": result["peak_rss_mb"]})
    if result["returncode"]:
        raise RuntimeError(f"Worker failed; see {output / 'worker.log'}")
    print(f"{kind} {window_name} seed={seed}: {result['wall_seconds']:.1f}s "
          f"peak={result['peak_rss_mb']:.0f}MiB", flush=True)
    return result


def worker(plan_path, output):
    rec = job_recorder(output)
    try:
        plan, job = read_json(plan_path), read_json(output / "job.json")
        if file_sha256(plan_path) != read_json(output / "manifest.json")["plan_sha256"]:
            raise ValueError("Frozen plan changed")
        config = RunConfig.model_validate(read_json(output / "config.json"))
        window = next(w for w in plan["windows"] if w["name"] == job["window"])
        if job["kind"] == "fit":
            fit_worker(plan, window, job["arm"], config, output, rec)
        else:
            infer_worker(plan, window, job["arm"], config,
                         {int(seed): Path(path) for seed, path in job["paths"].items()}, output, rec)
        rec.event("run", "completed", kind=job["kind"])
    except BaseException as exc:
        rec.failure("worker", exc)
        raise


# ------------------------------------------------------------------ execution


def prior_claims(stem):
    rows = []
    for claim in sorted((RUNS / "training_claims").glob(stem + "*.json")):
        record = read_json(claim)
        archive = Path(record["run"])
        rows.append({"claim": str(claim), "archive": archive,
                     "claimed_at_utc": record.get("claimed_at_utc"),
                     "plan_sha256": record.get("plan_sha256"),
                     "complete": (archive / "model.pt").is_file() and (archive / "training.json").is_file()})
    return rows


def claim_fit(plan_path, name, seed, path):
    stem = f"{name}_s{seed}"
    prior = prior_claims(stem)
    if any(row["complete"] for row in prior):
        raise ValueError("Training claim already spent; no additional fits allowed")
    if len(prior) >= 2:
        raise ValueError("At most one technical retry is allowed per incomplete fit")
    if any(row["plan_sha256"] != file_sha256(plan_path) for row in prior):
        raise ValueError("An earlier claim belongs to a different frozen plan")
    attempt = len(prior) + 1
    claim = RUNS / "training_claims" / (f"{stem}.json" if attempt == 1 else f"{stem}_attempt{attempt}.json")
    claim_training(claim, path, name, seed, plan_path)
    return attempt, prior


def verify_artifact(path, expected, label):
    if file_sha256(path) != expected:
        raise ValueError(f"{label}: artifact identity changed")
    return True


def require_complete_attempts(attempts):
    completed = [row for row in attempts if row.get("status") == "completed"]
    if len(completed) != 6:
        raise ValueError(f"Six completed candidate fits are required, saw {len(completed)}")
    return completed


def check_recovery_source(plan, source, plan_path=PLAN):
    """A completed six-fit archive can be re-scored without opening a training path."""
    source = Path(source).resolve()
    manifest = read_json(source / "manifest.json")
    if manifest.get("plan_sha256") != file_sha256(plan_path):
        raise ValueError("Recovery source was produced by a different frozen plan")
    if manifest.get("status") not in ("failed", "completed"):
        raise ValueError("Recovery source is not a finished archive")
    attempts = read_json(source / "attempts.json")
    completed = require_complete_attempts(attempts)
    for window in plan["windows"]:
        for seed in FULL_SEEDS:
            fitting = fit_dir(source, window["name"], seed)
            if not all((fitting / filename).is_file()
                       for filename in ("model.pt", "training.json", "training_identity.json")):
                raise FileNotFoundError(f"Recovery source is missing fit artifacts: {fitting}")
            if not (source / "inference" / window["name"] / f"seed{seed}_predictions.parquet").is_file():
                raise FileNotFoundError(f"Recovery source is missing predictions for {window['name']} seed {seed}")
    return completed


def verify_frozen_source(plan, plan_path):
    snapshot = Path(plan["source_frozen"]["path"])
    if file_sha256(snapshot) != plan["source_frozen"]["sha256"]:
        raise ValueError("Frozen source snapshot changed")
    for name, expected in plan["source_frozen"]["files"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"Training source changed after freeze: {name}")


def fit_candidate(plan_path, root, plan, window, seed, rec, attempts, started):
    name = window["name"]
    if len(attempts) >= plan["budget"]["fit_attempts_max"]:
        raise RuntimeError("Training attempt cap exhausted")
    path = fit_dir(root, name, seed)
    attempt, prior = claim_fit(plan_path, name, seed, path)
    if prior:
        write_json(root / "recoveries.json", {
            "interrupted_archives": [{"archive": str(row["archive"]), "claim": row["claim"]} for row in prior],
            "reason": "an earlier attempt at this seed was interrupted before it saved a model, so it left no "
                      "result to select on"})
        rec.event("training", "retry", window=name, seed=seed, attempt=attempt,
                  interrupted_archives=[str(row["archive"]) for row in prior])
    attempts.append({"window": name, "arm": CANDIDATE, "seed": seed, "steps": STEPS,
                     "attempt": attempt, "path": str(path), "status": "started"})
    write_json(root / "attempts.json", attempts)
    remaining = plan["budget"]["fit_stage_wall_cap_seconds"] - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("Fit stage budget exhausted")
    launch(plan_path, path, "fit", name, seed=seed,
           cap={"wall_seconds": min(plan["budget"]["per_fit_wall_cap_seconds"], remaining),
                "rss_mb": plan["budget"]["local_peak_rss_cap_mib"]},
           allow_existing=bool(prior))
    training = read_json(path / "training.json")
    stats = training.get("target_centering_stats")
    expected = plan["label_transform"]["per_window"][name]
    if training["steps"] != STEPS or training["samples"] != window["arms"][CANDIDATE]["samples"]:
        raise ValueError("CSZ fit did not complete the declared steps and samples")
    if training.get("target_centering") != MODE or not isinstance(stats, dict):
        raise ValueError("CSZ fit did not record its target transform")
    if stats["method"] != MODE or stats["timestamps"] != 80 or stats["mask_excluded_rows"] != 0:
        raise ValueError("CSZ fit did not use every full training timestamp")
    if not np.isclose(training["target_scale"], expected["raw_std"], atol=1e-14, rtol=1e-10):
        raise ValueError("CSZ fit target scale is not the raw-label standard deviation")
    if not np.isclose(training["centered_target_std"], expected["centered_std"], atol=1e-14, rtol=1e-10):
        raise ValueError("CSZ fit centered statistics differ from the control")
    if not np.isclose(training["transformed_target_std"], expected["float32_transformed_std"],
                      atol=1e-12, rtol=1e-5):
        raise ValueError("CSZ fit transformed statistics differ from the frozen plan")
    if not np.isclose(stats["c"], expected["c"], atol=1e-15, rtol=1e-10):
        raise ValueError("CSZ fit pool constant differs from the frozen plan")
    if not np.isclose(stats["fallback_mean"], expected["fallback_mean"], atol=1e-14):
        raise ValueError("CSZ fallback is not the centered-control fallback")
    attempts[-1]["status"] = "completed"
    attempts[-1]["training"] = {key: training[key] for key in
                                ("loss", "target_scale", "raw_target_std", "centered_target_std",
                                 "transformed_target_std", "steps", "samples", "samples_seen", "loss_units",
                                 "prediction_units")}
    attempts[-1]["transform"] = {key: stats[key] for key in
                                 ("timestamps", "rows", "min_rows_per_timestamp", "mask_excluded_rows", "raw_std",
                                  "centered_std", "q_std", "c", "transformed_std", "float32_transformed_std",
                                  "fallback_mean", "zero_variance_timestamps")}
    write_json(root / "attempts.json", attempts)


def fit_checks(plan, fit_root):
    rows = []
    for window in plan["windows"]:
        name = window["name"]
        expected = plan["label_transform"]["per_window"][name]
        for seed in FULL_SEEDS:
            fitting = fit_dir(fit_root, name, seed)
            training = read_json(fitting / "training.json")
            stats = training["target_centering_stats"]
            identity = read_json(fitting / "training_identity.json")
            frozen = {**window["arms"][CANDIDATE]["frozen_identity"], "dates": window["arms"][CANDIDATE]["train_dates"]}
            if {key: identity.get(key) for key in frozen} != frozen:
                raise ValueError(f"{name} seed {seed}: CSZ fit used different training keys or labels")
            rows.append({
                "window": name, "seed": seed, "steps": training["steps"], "samples": training["samples"],
                "target_scale": training["target_scale"], "raw_target_std": training["raw_target_std"],
                "centered_target_std": training["centered_target_std"],
                "transformed_target_std": training["transformed_target_std"],
                "prediction_units": training["prediction_units"], "timestamps": stats["timestamps"],
                "rows": stats["rows"], "mask_excluded_rows": stats["mask_excluded_rows"],
                "q_std": stats["q_std"], "c": stats["c"], "transformed_std": stats["transformed_std"],
                "float32_transformed_std": stats["float32_transformed_std"],
                "fallback_mean": stats["fallback_mean"], "zero_variance_timestamps": stats["zero_variance_timestamps"],
                "target_scale_matches_plan_raw_std": bool(np.isclose(
                    training["target_scale"], expected["raw_std"], atol=1e-14, rtol=1e-10)),
                "centered_std_matches_plan": bool(np.isclose(
                    training["centered_target_std"], expected["centered_std"], atol=1e-14, rtol=1e-10)),
                "transformed_std_matches_plan": bool(np.isclose(
                    training["transformed_target_std"], expected["float32_transformed_std"],
                    atol=1e-12, rtol=1e-5)),
                "pool_variance_matched": bool(np.isclose(
                    stats["transformed_std"], stats["centered_std"], atol=1e-15, rtol=1e-10)),
                "fallback_matches_control": bool(np.isclose(
                    stats["fallback_mean"], expected["fallback_mean"], atol=1e-14))})
    return [
        check("candidate_runs_all_six_declared_fits", len(rows) == 6, rows=rows),
        check("csz_fits_complete_1500_steps", all(row["steps"] == STEPS for row in rows), rows=rows),
        check("csz_fits_cover_every_full_timestamp",
              all(row["timestamps"] == 80 and row["rows"] == row["samples"] and row["mask_excluded_rows"] == 0
                  for row in rows), rows=rows),
        check("target_std_is_raw_label_std_not_transformed",
              all(row["target_scale_matches_plan_raw_std"] and row["centered_std_matches_plan"] for row in rows),
              rows=rows),
        check("csz_variance_matches_centered_pool",
              all(row["pool_variance_matched"] and row["transformed_std_matches_plan"] for row in rows), rows=rows),
        check("csz_fallback_stays_the_centered_control_fallback",
              all(row["fallback_matches_control"] for row in rows), rows=rows),
        check("candidate_predictions_are_relative_scores",
              all(row["prediction_units"] == "relative_return_score" for row in rows), rows=rows),
    ]


def centered_error(predictions, targets, evaluation):
    """Diagnostic MSE against per-timestamp-centered official labels, in float64."""
    scored = predictions.merge(targets[KEYS], on=KEYS, validate="one_to_one")
    aligned = scored.merge(targets, on=KEYS, validate="one_to_one")
    aligned["target"] = aligned["target"].astype(np.float64)
    aligned["target"] = aligned["target"] - aligned.groupby("date")["target"].transform("mean")
    return evaluate(scored, aligned, evaluation, return_sections=True, undefined_constant_portfolios=True)


def error_report(plan, fit_root, paths):
    report = {}
    for window in plan["windows"]:
        name = window["name"]
        targets = pd.read_parquet(window["targets"]["path"])
        evaluation = evaluation_config(window)
        rows = {}
        for arm, by_seed in paths[name].items():
            rows[arm] = {}
            for seed, path in sorted(by_seed.items()):
                predictions = pd.read_parquet(path)
                centered, daily, _ = centered_error(predictions, targets, evaluation)
                official = evaluate(predictions.merge(targets[KEYS], on=KEYS, validate="one_to_one"), targets,
                                    evaluation, return_sections=True, undefined_constant_portfolios=True)
                official_metrics, official_daily, _ = official
                for column in ("rank_ic", "long_short_return", "turnover"):
                    if not np.allclose(official_daily[column], daily[column], atol=1e-10, rtol=1e-9,
                                       equal_nan=True):
                        raise ValueError(f"{name} {arm} seed {seed}: float64 centered labels changed {column}")
                if arm == CANDIDATE:
                    training = read_json(fit_dir(fit_root, name, seed) / "training.json")
                    stats = training["target_centering_stats"]
                    extra = {"training_transformed_label_loss": training.get("loss"),
                             "training_loss_units": training.get("loss_units"),
                             "training_optimization_loss": training.get("optimization_loss"),
                             "transform_c": stats["c"], "transform_q_std": stats["q_std"]}
                elif arm == CENTERED:
                    centered_fit = Path(window["arms"][CENTERED]["per_seed"][str(seed)]["fit_dir"])
                    training = read_json(centered_fit / "training.json")
                    extra = {"training_centered_label_loss": training.get("loss"),
                             "training_loss_units": training.get("loss_units"),
                             "training_optimization_loss": training.get("optimization_loss")}
                else:
                    training = window["arms"][REFERENCE]["per_seed"][str(seed)]["training"]
                    extra = {"training_raw_label_loss": training.get("loss"),
                             "training_loss_units": "raw_label_mse",
                             "training_optimization_loss": None}
                rows[arm][str(seed)] = {
                    "official_mse_raw_label": official_metrics["mse"],
                    "diagnostic_mse_centered_label": centered["mse"],
                    "official_mae_raw_label": official_metrics["mae"],
                    "diagnostic_mae_centered_label": centered["mae"],
                    "official_ic_raw_label": official_metrics["ic_mean"],
                    "diagnostic_ic_centered_label": centered["ic_mean"],
                    "target_scale_raw_label_std": training.get("target_scale"),
                    "centered_target_std": training.get("centered_target_std"),
                    "transformed_target_std": training.get("transformed_target_std"),
                    **extra}
        report[name] = rows
    return report


def score_all(plan, paths, checks):
    scored = {}
    for window in plan["windows"]:
        name = window["name"]
        if file_sha256(window["targets"]["path"]) != window["targets"]["sha256"]:
            raise ValueError("Frozen evaluation targets changed")
        targets = pd.read_parquet(window["targets"]["path"])
        evaluation = evaluation_config(window)
        daily, metrics, population = {}, {}, {}
        for arm, by_seed in paths[name].items():
            for seed, path in sorted(by_seed.items()):
                predictions = pd.read_parquet(path)
                values, frame, sections = score_predictions(predictions, targets, evaluation)
                label = f"{name}_{arm}_s{seed}"
                checks.append(_answer_coverage_check(predictions[KEYS], targets, 0.99,
                                                     label + "_answer_coverage", "instruments"))
                checks.append(check(label + "_complete_correlation_coverage",
                                    complete_correlations(values, 80)))
                checks.append(check(label + "_exact_ten_day_eight_time_grid",
                                    len(sections) == 80 and sections.groupby("date").size().eq(8).all()
                                    and len(frame) == len(window["evaluation_dates"])))
                daily[(arm, seed)] = frame.set_index("date")[list(SCORING_METRICS)]
                metrics[(arm, seed)] = values
                population.setdefault(arm, set()).add(values["sample_keys_sha256"] + values["targets_sha256"])
        if any(len(value) != 1 for value in population.values()):
            raise ValueError(f"{name}: seeds were scored on different keys")
        if len({next(iter(value)) for value in population.values()}) != 1:
            raise ValueError(f"{name}: arms were scored on different keys or answers")
        scored[name] = {"daily": daily, "metrics": metrics, "targets": targets}
    return scored


def side_contrast(scored_window, name, high, low, seeds):
    means = {arm: arm_means(scored_window["daily"], arm, seeds) for arm in (high, low)}
    delta = means[high] - means[low]
    columns = ("long_return", "short_return")
    return {
        "period": name,
        "candidate": {column: float(means[high][column].mean()) for column in columns},
        "centered_control": {column: float(means[low][column].mean()) for column in columns},
        "delta": {column: float(delta[column].mean()) for column in columns},
        "worst_day_delta": {column: float(means[high][column].min() - means[low][column].min())
                            for column in columns},
        "per_seed_delta": {
            str(seed): {column: float(scored_window["daily"][(high, seed)][column].mean()
                                      - scored_window["daily"][(low, seed)][column].mean())
                        for column in (*DELTA_METRICS, *columns)} for seed in seeds}}


def control_contrast(paired, intervals, summaries, scored, seeds):
    return {
        "schema_version": 1, "comparison": f"{CANDIDATE} minus {CENTERED}",
        "scope": "same_seed_descriptive_contrast_no_gates", "seeds": list(seeds),
        "mean_rank_ic_delta": float(paired.rank_ic.mean()),
        "period_mean_deltas": {str(key): float(value) for key, value in paired.groupby("period").rank_ic.mean().items()},
        "gross_mean_delta": float(paired.long_short_return.mean()),
        "turnover_mean_delta": float(paired.turnover.mean()),
        "per_window": {row["period"]: {"worst_day_delta": row["worst_day_delta"],
                                       "arm_window_means": row["arm_window_means"],
                                       "per_seed_mean_delta": row["per_seed_mean_delta"]}
                       for row in summaries},
        "long_short_sides": {name: side_contrast(scored[name], name, CANDIDATE, CENTERED, seeds)
                             for name in WINDOWS},
        "intervals": intervals,
        "note": "Reported only. Passing the raw-label comparison does not show the new standardization beats "
                "the centered control, and no post-hoc gate is added to this contrast."}


def run_progress(plan, root, candidate_paths, rec):
    checks = []
    paths = {window["name"]: {
        REFERENCE: {11: window["arms"][REFERENCE]["per_seed"]["11"]["predictions"]},
        CENTERED: {11: window["arms"][CENTERED]["per_seed"]["11"]["predictions"]},
        CANDIDATE: candidate_paths[window["name"]][CANDIDATE]} for window in plan["windows"]}
    scored = score_all(plan, paths, checks)
    paired, intervals, summaries = analysis(plan, scored, CANDIDATE, REFERENCE, PROGRESS_SEEDS)
    paired.to_csv(root / "progress_seed11_paired_daily.csv", index=False)
    progress = {
        "schema_version": 1, "descriptive_only": True, "seeds": list(PROGRESS_SEEDS),
        "note": "Progress display only after all six fits. It cannot accept, reject or stop the experiment.",
        "mean_rank_ic_delta": float(paired.rank_ic.mean()),
        "period_mean_deltas": {str(key): float(value) for key, value in paired.groupby("period").rank_ic.mean().items()},
        "worst_day_delta": {row["period"]: row["worst_day_delta"] for row in summaries},
        "gross_mean_delta": float(paired.long_short_return.mean()),
        "turnover_mean_delta": float(paired.turnover.mean()),
        "intervals": intervals, "per_window": summaries,
        "checks": {row["name"]: row["passed"] for row in checks}}
    write_json(root / "progress_seed11.json", progress)
    if not save_checks(root, checks):
        raise ValueError("Seed-11 progress scoring checks failed")
    rec.event("evaluation", "progress_seed11", descriptive_only=True,
              mean_rank_ic_delta=progress["mean_rank_ic_delta"],
              period_mean_deltas=progress["period_mean_deltas"])
    print("progress seed11 (not a decision)", progress["mean_rank_ic_delta"],
          progress["period_mean_deltas"], flush=True)
    return progress


def execute(plan_path, root):
    plan = read_json(plan_path)
    started = time.monotonic()
    rec = Recorder(root, EXPERIMENT)
    write_json(root / "manifest.json", {"schema_version": 1, "run_id": root.name, "experiment_id": EXPERIMENT,
        "started_at": utc_now(), "status": "running", "plan_sha256": file_sha256(plan_path),
        "code": capture_source(root), "environment": environment(), "final_views": 0})
    write_json(root / "preregistration.json", plan)
    attempts = []
    try:
        rec.event("run", "started", fits_planned=6, progress_seeds=list(PROGRESS_SEEDS),
                  full_seeds=list(FULL_SEEDS))
        verify_frozen_source(plan, plan_path)
        if file_sha256(plan["registry_path"]) != plan["registry_sha256"]:
            raise ValueError("Shared data-role registration changed")
        for path, checksum in plan["inputs_sha256"].items():
            if file_sha256(path) != checksum:
                raise ValueError(f"Frozen input changed: {path}")
        for window in plan["windows"]:
            for seed in FULL_SEEDS:
                fit_candidate(plan_path, root, plan, window, seed, rec, attempts, started)
        require_complete_attempts(attempts)
        candidate_paths = {}
        for window in plan["windows"]:
            name = window["name"]
            inference = root / "inference" / name
            paths = {str(seed): str(fit_dir(root, name, seed) / "model.pt") for seed in FULL_SEEDS}
            remaining = plan["budget"]["fit_stage_wall_cap_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Round budget exhausted before inference")
            launch(plan_path, inference, "infer", name, paths=paths,
                   cap={"wall_seconds": min(plan["budget"]["per_fit_wall_cap_seconds"], remaining),
                        "rss_mb": plan["budget"]["local_peak_rss_cap_mib"]})
            candidate_paths[name] = {CANDIDATE: {
                seed: str(inference / f"seed{seed}_predictions.parquet") for seed in FULL_SEEDS}}
        progress = run_progress(plan, root, candidate_paths, rec)
        run_full(plan, root, candidate_paths, progress, rec, attempts)
        if time.monotonic() - started > plan["budget"]["fit_stage_wall_cap_seconds"]:
            raise TimeoutError("Round wall cap exhausted during analysis")
        rec.event("run", "completed", training_attempts=len(attempts), final_views=0)
        finish(root, "completed", 0, resources={"wall_seconds": time.monotonic() - started})
    except BaseException as exc:
        rec.failure("experiment", exc)
        write_json(root / "decision.json", {"result": "inconclusive", "action": "retain_10_day_reference",
            "reason": str(exc), "training_attempts": len(attempts), "final_views": 0, "model_promoted": False})
        finish(root, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", 1,
               resources={"wall_seconds": time.monotonic() - started}, error={"reason": str(exc)})
        raise


def run_full(plan, root, candidate_paths, progress, rec, attempts, *, fit_root=None, inference_root=None):
    fit_root, inference_root = Path(fit_root or root), Path(inference_root or root)
    checks = fit_checks(plan, fit_root)
    paths = {}
    for window in plan["windows"]:
        name = window["name"]
        paths[name] = {
            REFERENCE: {int(seed): row["predictions"] for seed, row in
                        window["arms"][REFERENCE]["per_seed"].items()},
            CENTERED: {int(seed): row["predictions"] for seed, row in
                       window["arms"][CENTERED]["per_seed"].items()},
            CANDIDATE: candidate_paths[name][CANDIDATE]}
    scored = score_all(plan, paths, checks)
    errors = error_report(plan, fit_root, paths)
    costs = {window["name"]: read_json(inference_root / "inference" / window["name"] / "inference_cost.json")
             for window in plan["windows"]}
    for window in plan["windows"]:
        name = window["name"]
        chunks = costs[name]["chunks"]
        checks.append(check(f"{name}_inference_uses_five_day_chunks",
                            len(chunks) == 2 and all(len(chunk["dates"]) == 5 for chunk in chunks),
                            chunks=[len(chunk["dates"]) for chunk in chunks]))
        checks.append(check(f"{name}_inference_has_three_seed_predictions",
                            all((inference_root / "inference" / name / f"seed{seed}_predictions.parquet").is_file()
                                for seed in FULL_SEEDS)))
    paired, intervals, summaries = analysis(plan, scored, CANDIDATE, REFERENCE, FULL_SEEDS)
    control_paired, control_intervals, control_summaries = analysis(
        plan, scored, CANDIDATE, CENTERED, FULL_SEEDS)
    paired.to_csv(root / "paired_daily.csv", index=False)
    control_paired.to_csv(root / "control_paired_daily.csv", index=False)
    write_json(root / "paired_intervals.json", intervals)
    write_json(root / "control_paired_intervals.json", control_intervals)
    exposure = {window["name"]: window["exposure"] for window in plan["windows"]}
    decision = coverage_decision(
        paired, summaries, intervals, plan["policy"], cost_measured=bool(costs), exposure=exposure,
        only_factor=plan["only_factor"],
        interpretation="A non-accept result rejects the claim that this variance-matched timestamp z-score "
                       "target improves the retained raw-label 10-day reference under the unchanged gates. It "
                       "does not show that every cross-sectional transform or target scale is uninformative.",
        limitations="Two research windows and three seeds are selection data; local gross metrics exclude fees "
                    "and slippage; the centered-control contrast is descriptive only; the platform time and "
                    "memory limits remain unverified.")
    decision["action"] = ("retain_csz_candidate_for_confirmation"
                          if decision["result"] == "accept" else "retain_10_day_reference")
    decision["stage"] = "full"
    decision["progress_seed11"] = {"descriptive_only": True,
                                   "mean_rank_ic_delta": progress["mean_rank_ic_delta"],
                                   "period_mean_deltas": progress["period_mean_deltas"],
                                   "worst_day_delta": progress["worst_day_delta"]}
    control = control_contrast(control_paired, control_intervals, control_summaries, scored, FULL_SEEDS)
    decision["comparisons"] = {
        "candidate_vs_raw_label_reference": {"result": decision["result"], "action": decision["action"],
                                             "checks": decision["checks"],
                                             "mean_rank_ic_delta": decision["mean_rank_ic_delta"],
                                             "period_mean_deltas": decision["period_mean_deltas"]},
        "candidate_vs_centered_control": {key: control[key] for key in
                                          ("scope", "mean_rank_ic_delta", "period_mean_deltas", "gross_mean_delta",
                                           "turnover_mean_delta")}}
    decision["note"] = ("The formal decision uses candidate minus raw-label reference only. The same-seed "
                        "centered-control contrast is reported to separate 'beats the raw reference' from "
                        "'new standardization beats centering'.")
    write_json(root / "decision.json", decision)
    write_json(root / "metrics.json", {
        "label_transform": plan["label_transform"],
        "windows": {row["period"]: {
            "reference_comparison": row,
            "control_comparison": next(item for item in control_summaries if item["period"] == row["period"]),
            "long_short_sides": control["long_short_sides"][row["period"]],
            "error_report": errors[row["period"]]} for row in summaries},
        "control_comparison": control,
        "progress_seed11": decision["progress_seed11"],
        "inference_cost": costs, "exposure": exposure,
        "calibration": plan["calibration"]["noise"], "minimum_mean_gain": plan["policy"]["minimum_mean_gain"],
        "costs": {"effective_candidate_training": plan["budget"]["effective_candidate_training"],
                  "fit_attempts_max": plan["budget"]["fit_attempts_max"],
                  "per_fit_wall_cap_seconds": plan["budget"]["per_fit_wall_cap_seconds"],
                  "round_wall_cap_seconds": plan["budget"]["fit_stage_wall_cap_seconds"],
                  "local_peak_rss_cap_mib": plan["budget"]["local_peak_rss_cap_mib"],
                  "fits": attempts}})
    if not save_checks(root, checks):
        raise ValueError("Full scoring checks failed")
    rec.event("evaluation", "decision", result=decision["result"], action=decision["action"],
              mean_rank_ic_delta=decision["mean_rank_ic_delta"],
              period_mean_deltas=decision["period_mean_deltas"], checks=decision["checks"])
    print("decision", decision["result"], decision["action"], decision["mean_rank_ic_delta"],
          decision["period_mean_deltas"], flush=True)


# ------------------------------------------------------------------ recovery and record


def recover(plan_path, source, output_root=RUNS):
    """Re-score a six-fit archive without opening any training path."""
    plan = read_json(plan_path)
    source = Path(source).resolve()
    attempts = check_recovery_source(plan, source, plan_path)
    run = new_run_dir(Path(output_root))
    rec = Recorder(run, EXPERIMENT)
    write_json(run / "manifest.json", {"schema_version": 1, "run_id": run.name, "experiment_id": EXPERIMENT,
        "started_at": utc_now(), "status": "running", "mode": "recover-analysis", "new_training": 0,
        "source_run": str(source), "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "plan_sha256": file_sha256(plan_path), "code": capture_source(run), "environment": environment(),
        "final_views": 0})
    write_json(run / "preregistration.json", plan)
    write_json(run / "attempts.json", attempts)
    try:
        rec.event("run", "started", mode="recover-analysis", source_run=str(source), new_training=0)
        write_json(run / "recovery.json", {
            "source_run": str(source), "source_manifest_sha256": file_sha256(source / "manifest.json"),
            "reason": "The six fits and no-label inferences completed, but analysis or serialization failed. "
                      "The repair reuses the existing complete weights and predictions; no model is retrained.",
            "reused_attempts": attempts, "new_training": 0})
        candidate_paths = {window["name"]: {CANDIDATE: {
            seed: str(source / "inference" / window["name"] / f"seed{seed}_predictions.parquet")
            for seed in FULL_SEEDS}} for window in plan["windows"]}
        progress = run_progress(plan, run, candidate_paths, rec)
        run_full(plan, run, candidate_paths, progress, rec, attempts, fit_root=source, inference_root=source)
        rec.event("run", "completed", mode="recover-analysis", source_run=str(source), new_training=0)
        finish(run, "completed", 0, resources={"wall_seconds": 0.0})
    except BaseException as exc:
        rec.failure("recovery", exc)
        write_json(run / "decision.json", {"result": "inconclusive", "action": "retain_10_day_reference",
            "reason": str(exc), "training_attempts": 0, "final_views": 0, "model_promoted": False})
        finish(run, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", 1,
               resources={"wall_seconds": 0.0}, error={"reason": str(exc)})
        raise
    return run


def recorded_fits(root):
    rows = []
    if not (Path(root) / "attempts.json").is_file():
        return rows
    for item in read_json(Path(root) / "attempts.json"):
        if item["status"] != "completed":
            continue
        rows.append({"window": item["window"], "seed": item["seed"], "steps": item["steps"],
                     "run": item["path"], "attempt": item.get("attempt"),
                     "resources": read_json(Path(item["path"]) / "manifest.json").get("resources"),
                     "weights_sha256": file_sha256(Path(item["path"]) / "model.pt"),
                     "training": item.get("training"), "transform": item.get("transform")})
    return rows


def conclusion_text(decision, metrics, progress):
    failed = [key for key, passed in decision["checks"].items() if not passed]
    per_window = ", ".join(f"{key}: {value:+.6f}" for key, value in decision["period_mean_deltas"].items())
    control = metrics["control_comparison"]
    control_windows = ", ".join(f"{key}: {value:+.6f}" for key, value in control["period_mean_deltas"].items())
    mse = []
    for name, rows in metrics["windows"].items():
        arms = rows["error_report"]
        seed = sorted(arms[CANDIDATE])[0]
        mse.append(f"{name}: raw-label MSE {arms[CANDIDATE][seed]['official_mse_raw_label']:.6e}, "
                   f"centered-label MSE {arms[CANDIDATE][seed]['diagnostic_mse_centered_label']:.6e}, "
                   f"optimized transformed-target loss {arms[CANDIDATE][seed]['training_transformed_label_loss']:.6e} "
                   f"(seed {seed})")
    head = (f"With only the training target changed (raw official return -> full-timestamp z-score rescaled by "
            f"one pool constant to the centered-pool variance, raw-label target_std kept), the seed-mean paired "
            f"daily all-eight-time RankIC difference against the raw-label 10-day/1500 reference was "
            f"{decision['mean_rank_ic_delta']:+.6f} against the unchanged floor of "
            f"{decision['minimum_mean_gain']:.6f}; per window {per_window}. Seed 11 progress was "
            f"{progress['mean_rank_ic_delta']:+.6f} and is not part of the decision.")
    if decision["result"] == "accept":
        tail = (" The candidate clears the unchanged gain, replication, guardrail, interval and cost gates, so it "
                "may be registered for a separately frozen confirmation stage. It is not a platform-verified "
                "submission, and this acceptance does not show the variance-matched standardization beats the "
                "centered control.")
    else:
        tail = (f" The candidate fails the predeclared gates {', '.join(failed) or 'unspecified'}, so the "
                "raw-label 10-day reference is retained. The three seeds are complete; this is not evidence that "
                "every cross-sectional label transform is uninformative.")
    return (head + " Error reporting, kept separate from ranking: " + "; ".join(mse) + ". Same-seed contrast "
            "against the centered control (report only): RankIC " f"{control['mean_rank_ic_delta']:+.6f} "
            f"(per window {control_windows}), gross {control['gross_mean_delta']:+.9f}, turnover "
            f"{control['turnover_mean_delta']:+.6f}." + tail)


def record(plan_path, execution, output_path):
    plan = read_json(plan_path)
    execution = Path(execution).resolve()
    if read_json(execution / "manifest.json")["status"] != "completed":
        raise ValueError("A research record requires a completed execution archive")
    decision = read_json(execution / "decision.json")
    metrics = read_json(execution / "metrics.json")
    progress = read_json(execution / "progress_seed11.json")
    fits = recorded_fits(execution)
    document = {
        "schema_version": 1, "experiment_id": EXPERIMENT, "recorded_at_utc": utc_now(),
        "hypothesis": plan["hypothesis"], "only_factor": plan["only_factor"],
        "recipe_fixed": plan["recipe_fixed"], "label_transform": plan["label_transform"],
        "steps": plan["steps"], "seeds": plan["seeds"], "deployment_seed": plan["deployment_seed"],
        "plan": {"path": str(Path(plan_path).resolve()), "sha256": file_sha256(plan_path)},
        "reference_source": plan["reference_source"],
        "gates": {"policy": plan["policy"], "minimum_mean_gain": plan["policy"]["minimum_mean_gain"],
                  "worst_day_tolerance": plan["policy"]["noise"]["min_daily_long_short_return"],
                  "thresholds_status": "copied unchanged from the frozen 20261017 v4 calibration"},
        "reused_arms": {
            "raw_label_reference": {window["name"]: {seed: {key: row[key] for key in
                                                             ("weights_sha256", "predictions_sha256", "training")}
                                                      for seed, row in window["arms"][REFERENCE]["per_seed"].items()}
                                    for window in plan["windows"]},
            "centered_control": {window["name"]: {seed: {key: row[key] for key in
                                                           ("weights_sha256", "predictions_sha256", "training")}
                                                    for seed, row in window["arms"][CENTERED]["per_seed"].items()}
                                 for window in plan["windows"]}},
        "execution": {"run_dir": str(execution), "manifest_sha256": file_sha256(execution / "manifest.json"),
                      "new_training_fits": len(fits), "fits": fits, "progress_seed11": progress,
                      "retries": (read_json(execution / "recoveries.json")
                                  if (execution / "recoveries.json").is_file() else None),
                      "recovery": (read_json(execution / "recovery.json")
                                   if (execution / "recovery.json").is_file() else None)},
        "exposure": {window["name"]: window["exposure"] for window in plan["windows"]},
        "results": {"decision": decision, "windows": metrics["windows"],
                    "control_comparison": metrics["control_comparison"],
                    "inference_cost": metrics["inference_cost"],
                    "intervals": read_json(execution / "paired_intervals.json"),
                    "control_intervals": read_json(execution / "control_paired_intervals.json")},
        "conclusion": conclusion_text(decision, metrics, progress),
        "retained_version": ("raw-label 10-day reference (unchanged)" if decision["result"] != "accept"
                             else "variance-matched timestamp-zscore 10-day candidate (research stage only)"),
        "candidate_status": ("not promoted" if decision["result"] != "accept"
                             else "promoted to a separately frozen confirmation stage"),
        "model_promoted": False, "final_views": 0, "platform_verified": False,
        "limitations": [
            "Two research windows and three seeds are selection data, not independent confirmation.",
            "The CSZ template has a public cross-sectional precedent, but matching the centered pool variance "
            "with one constant is this project's control design, not a validated Qlib recipe.",
            "The target_std remains the raw-label standard deviation by design, so the optimized target is "
            "y*/target_std rather than the dimensionless q.",
            "Raw-label MSE, centered-label MSE and the optimized transformed-target loss measure different "
            "targets; none replaces the official competition score.",
            "The centered-control contrast is descriptive and cannot promote or reject the candidate.",
            "Confirmation, locked and legacy final ranges were not opened; platform time and memory limits "
            "remain unverified."],
        "next_steps": [
            "If rejected, form a new pre-registration before testing another standardization strength or loss; "
            "do not retune this round after seeing its result.",
            "If accepted, freeze the complete candidate and a separate confirmation plan before opening any "
            "untouched range; do not overwrite the current reference weights."],
    }
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Research records are append-only; select a new output name")
    write_json(output_path, document)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "worker", "run", "recover", "record"))
    parser.add_argument("path", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--centering-plan", type=Path)
    parser.add_argument("--centering-record", type=Path)
    parser.add_argument("--execution", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(str(prepare(args.output or PLAN, frozen_path=args.frozen or FROZEN,
                          output_root=args.output_root or RUNS,
                          centering_plan_path=args.centering_plan or CENTERING_PLAN,
                          centering_record_path=args.centering_record or CENTERING_RECORD)), flush=True)
    elif args.command == "worker":
        worker(args.path.resolve(), args.output.resolve())
    elif args.command == "run":
        root = new_run_dir(Path(args.output_root or RUNS).resolve())
        print(str(root), flush=True)
        execute(args.path.resolve(), root)
    elif args.command == "recover":
        if args.execution is None:
            parser.error("recover requires --execution <six-fit archive>")
        print(str(recover(args.path.resolve(), args.execution, args.output_root or RUNS)), flush=True)
    else:
        if args.execution is None or args.output is None:
            parser.error("record requires --execution <completed run> and --output <record.json>")
        print(str(record(args.path.resolve(), args.execution, args.output)), flush=True)


if __name__ == "__main__":
    main()
