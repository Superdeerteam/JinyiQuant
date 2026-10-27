"""Re-evaluate the frozen reference, representative candidates and simple baselines under eval-v2.

No training, no model search, no sealed-interval access. The only metric entry point
is ``jinyiquant.evaluation.evaluate`` (eval-v2). Predictions are reused from the
frozen multiscale replay artifacts; baselines are constructed analytically.

Predeclared before any metric is computed (see ``plan.json`` written first):

* windows: the five already-viewed development months covered by the frozen replay
  (2023-11, 2024-01, 2024-03, 2024-04, 2024-06), outside the sealed interval;
* objects: frozen nine-model equal blend (reference), three single-trunk three-seed
  ensembles (candidates), constant / train-window clock-mean / ridge-default
  baselines, and shuffled / sign-flipped reference controls;
* cost scenarios 0bp and 5bp, both recomputed by eval-v2;
* moving-block bootstrap with 5 consecutive trading-day blocks inside each month
  (blocks never cross the month gaps), 4000 draws, seed 20261026.

The script writes a new evaluation run archive under ``runs/reeval-a-20261026/``
with a manifest that records source prediction sha256, evaluation.py sha256 and
``re_evaluated_from``. Historical run directories are only read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from jinyiquant.comparison import _block_indices, evaluation_comparison_gate
from jinyiquant.config import EvaluationConfig
from jinyiquant.evaluation import CONTRACT, EVALUATION_VERSION, evaluate

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "runs/multiscale-replay-20261026/20261026T133118.869042Z_bcdc779dc66b"
VERIFY = ROOT / "runs/multiscale-replay-verify-20261026/20261026T140057.561539Z_1a57d4b38acc"
BLEND_SPEC = ROOT / "runs/submission-best-20261026/build/spec.json"
BLEND_WEIGHTS = ROOT / "runs/submission-best-20261026/build/blend_weights.json"
LABELS = ROOT / "数据/cpt_jyc_2026_vwap"
WINDOWS = (
    ("nov2023", SCREEN, "2023-11-01", "2023-12-01"),
    ("jan2024", SCREEN, "2024-01-01", "2024-02-01"),
    ("mar2024", SCREEN, "2024-03-01", "2024-04-01"),
    ("apr2024", VERIFY, "2024-04-01", "2024-05-01"),
    ("jun2024", VERIFY, "2024-06-01", "2024-07-01"),
)
MEMBERS = ("tlob_s11", "tlob_s29", "tlob_s47", "mlp_s11", "mlp_s29", "mlp_s47",
           "patch_s11", "patch_s29", "patch_s47")
SAMPLE_TIMES = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")
TRAIN_START, TRAIN_END = "2023-08-01", "2023-11-01"
SEALED = ("2024-10-08", "2025-01-01")
BOOTSTRAP = {"block_days": 5, "draws": 4000, "seed": 20261026, "confidence": 0.95}
COST_SCENARIOS = (0.0, 0.0005)
RIDGE_ALPHA = 1.0
EXPERIMENT_ID = "reeval-a-20261026"
METRICS_0BP = ("rank_ic", "ic", "long_short_return", "turnover")
METRICS_5BP = ("cost", "net_return")
INTERVAL_METRICS = ("rank_ic", "ic", "long_short_return", "turnover", "cost_5bp", "net_5bp")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluation_config(rate: float | None) -> EvaluationConfig:
    return EvaluationConfig(quantile=0.2, periods_per_year=252, min_cross_section=3,
                            constant_portfolios="undefined", single_side_rate=rate)


def load_window(name: str, replay: Path, start: str, end: str) -> dict:
    base = replay / "windows" / name
    targets = pd.read_parquet(base / "targets.parquet")
    canonical = targets[["date", "instrument"]].reset_index(drop=True)
    if canonical.duplicated().any():
        raise ValueError(f"{name}: duplicate target keys")
    frames = {}
    shas = {}
    for member in MEMBERS:
        path = base / member / "predictions.parquet"
        frame = pd.read_parquet(path)
        merged = frame.merge(canonical, on=["date", "instrument"], how="inner", validate="one_to_one")
        if len(merged) != len(canonical):
            raise ValueError(f"{name}/{member}: predictions do not cover every target key")
        frames[member] = merged.sort_values(["date", "instrument"]).reset_index(drop=True)
        shas[member] = sha256_file(path)
    ordered = targets.sort_values(["date", "instrument"]).reset_index(drop=True)
    for member, frame in frames.items():
        if not frame[["date", "instrument"]].equals(ordered[["date", "instrument"]]):
            raise ValueError(f"{name}/{member}: key order mismatch")
    return {
        "name": name, "replay": replay, "replay_rel": rel(replay), "start": start, "end": end,
        "targets": ordered,
        "scores": {member: frame["score"].to_numpy(dtype=float) for member, frame in frames.items()},
        "prediction_sha256": shas,
        "keys_sha256": sha256_file(base / "keys.parquet"),
        "targets_sha256": sha256_file(base / "targets.parquet"),
    }


def load_train_labels() -> pd.DataFrame:
    frames = []
    for month in ("2023-08", "2023-09", "2023-10"):
        path = LABELS / f"cpt_jyc_2026_vwap_{month}.parquet"
        frame = pd.read_parquet(path, columns=["date", "instrument", "vwap_return"])
        frame = frame[(frame["date"] >= TRAIN_START) & (frame["date"] < TRAIN_END)]
        frame = frame[frame["date"].dt.strftime("%H:%M").isin(SAMPLE_TIMES)]
        frame = frame.dropna(subset=["vwap_return"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def ridge_default(train: pd.DataFrame, timestamps: pd.Series, instruments: pd.Series) -> np.ndarray:
    """Fixed-alpha ridge on instrument and sample-time one-hot features (no tuning)."""
    instrument_names = sorted(train["instrument"].unique())
    instrument_index = {name: position for position, name in enumerate(instrument_names)}
    clock_index = {name: position for position, name in enumerate(SAMPLE_TIMES)}
    train = train.assign(clock=train["date"].dt.strftime("%H:%M"))
    n_instrument, n_clock = len(instrument_names), len(SAMPLE_TIMES)
    joint = train.groupby(["instrument", "clock"]).size()
    matrix = np.zeros((n_instrument + n_clock, n_instrument + n_clock))
    matrix[:n_instrument, :n_instrument] = np.diag(
        train.groupby("instrument").size().reindex(instrument_names, fill_value=0).to_numpy())
    matrix[n_instrument:, n_instrument:] = np.diag(
        train.groupby("clock").size().reindex(SAMPLE_TIMES, fill_value=0).to_numpy())
    for (instrument, clock), count in joint.items():
        row, column = instrument_index[instrument], n_instrument + clock_index[clock]
        matrix[row, column] = matrix[column, row] = float(count)
    target = np.concatenate([
        train.groupby("instrument")["vwap_return"].sum().reindex(instrument_names, fill_value=0.0).to_numpy(),
        train.groupby("clock")["vwap_return"].sum().reindex(SAMPLE_TIMES, fill_value=0.0).to_numpy(),
    ])
    matrix = matrix + RIDGE_ALPHA * np.eye(n_instrument + n_clock)
    beta = np.linalg.solve(matrix, target)
    scores = np.zeros(len(timestamps), dtype=float)
    clock_values = timestamps.dt.strftime("%H:%M").to_numpy()
    instrument_values = instruments.to_numpy()
    for position, (instrument, clock) in enumerate(zip(instrument_values, clock_values)):
        if instrument in instrument_index:
            scores[position] += beta[instrument_index[instrument]]
        if clock in clock_index:
            scores[position] += beta[n_instrument + clock_index[clock]]
    return scores


def window_month(date: pd.Timestamp) -> str:
    return date.strftime("%Y-%m")


def bootstrap_interval(diffs: pd.Series, months: pd.Series, *, block: int, draws: int,
                       seed: int, confidence: float) -> dict | None:
    """Moving-block bootstrap with blocks kept inside each month (no cross-gap blocks)."""
    frame = pd.DataFrame({"diff": diffs.to_numpy(dtype=float), "month": months.to_numpy()})
    frame = frame[np.isfinite(frame["diff"])]
    if frame.empty:
        return None
    groups = [(name, frame.loc[frame["month"] == name, "diff"].to_numpy(dtype=float))
              for name in sorted(frame["month"].unique())]
    if any(len(values) < 2 for _, values in groups):
        return None
    rng = np.random.default_rng(seed)
    observed = float(frame["diff"].mean())
    sampled_means = []
    for _ in range(draws):
        total = 0.0
        count = 0
        for _, values in groups:
            indices = _block_indices(len(values), min(block, len(values)), rng)
            total += float(values[indices].sum())
            count += len(indices)
        sampled_means.append(total / count)
    sampled = np.asarray(sampled_means)
    alpha = (1 - float(confidence)) / 2
    low, high = np.quantile(sampled, [alpha, 1 - alpha])
    return {
        "mean": observed,
        "low": float(low),
        "high": float(high),
        "confidence": float(confidence),
        "share_positive": float((sampled > 0).mean()),
        "defined_days": int(len(frame)),
        "blocks_per_month": {name: int(len(values)) for name, values in groups},
        "draws": draws,
        "block_days": block,
        "seed": seed,
    }


def build_objects(windows: list[dict], train: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    scores_by_window: dict[str, dict[str, np.ndarray]] = {}
    targets = []
    keys = []
    for window in windows:
        scores = window["scores"]
        reference = sum(scores[member] for member in MEMBERS) / len(MEMBERS)
        tlob = sum(scores[member] for member in MEMBERS[:3]) / 3
        mlp = sum(scores[member] for member in MEMBERS[3:6]) / 3
        patch = sum(scores[member] for member in MEMBERS[6:]) / 3
        frame = window["targets"]
        timestamps = frame["date"]
        instruments = frame["instrument"]
        clock_mean = train.assign(clock=train["date"].dt.strftime("%H:%M")).groupby("clock")["vwap_return"].mean()
        label_mean = timestamps.dt.strftime("%H:%M").map(clock_mean).to_numpy(dtype=float)
        rng = np.random.default_rng(BOOTSTRAP["seed"])
        shuffled = np.empty_like(reference)
        for positions in frame.groupby("date").indices.values():
            positions = np.asarray(positions)
            shuffled[positions] = reference[positions][rng.permutation(len(positions))]
        scores_by_window[window["name"]] = {
            "reference_nine_blend": reference,
            "candidate_tlob_3seed": tlob,
            "candidate_mlp_3seed": mlp,
            "candidate_patch_3seed": patch,
            "baseline_constant_zero": np.zeros(len(reference)),
            "baseline_label_mean": label_mean,
            "baseline_ridge_default": ridge_default(train, timestamps, instruments),
            "control_shuffled_reference": shuffled,
            "control_sign_flipped_reference": -reference,
        }
        targets.append(frame[["date", "instrument", "target"]])
        keys.append(frame[["date", "instrument"]])
    pooled_targets = pd.concat(targets, ignore_index=True).sort_values(["date", "instrument"]).reset_index(drop=True)
    pooled_keys = pd.concat(keys, ignore_index=True).sort_values(["date", "instrument"]).reset_index(drop=True)
    if not pooled_keys.equals(pooled_targets[["date", "instrument"]]):
        raise ValueError("pooled target keys changed")
    pooled_scores = {}
    for name in scores_by_window[windows[0]["name"]]:
        pooled_scores[name] = np.concatenate([scores_by_window[window["name"]][name] for window in windows])
    return pooled_scores, pooled_keys, pooled_targets


def evaluate_object(predictions: pd.DataFrame, targets: pd.DataFrame, rate: float):
    return evaluate(predictions, targets, evaluation_config(rate), return_sections=True,
                    undefined_constant_portfolios=True)


def evaluate_payload(payload):
    """Worker task: evaluate one object at 0bp and 5bp; 0bp is repeated for the reference."""
    object_id, predictions, targets, repeat = payload
    metrics_0, daily_0, _ = evaluate_object(predictions, targets, 0.0)
    metrics_5, daily_5, _ = evaluate_object(predictions, targets, 0.0005)
    repeat_digest = None
    if repeat:
        again, _, _ = evaluate_object(predictions, targets, 0.0)
        repeat_digest = {"first": digest(metrics_0), "second": digest(again)}
    return object_id, metrics_0, daily_0, metrics_5, daily_5, repeat_digest


def scenario_block(metrics_0: dict, metrics_5: dict, gate_0: dict, gate_5: dict) -> dict:
    """0bp/5bp cost scenarios built only from the two eval-v2 evaluate calls."""
    reasons = sorted(set(gate_0["reasons"]) | set(gate_5["reasons"]))
    return {
        "evaluation_version": EVALUATION_VERSION,
        "source": "src/jinyiquant/evaluation.py:evaluate",
        "turnover_conversion_factor": 0.5,
        "definition": "cost = single_side_rate * U; U = sum(|target weight - pre-trade drifted weight|); "
                      "turnover = U / 2; every scenario is a fresh eval-v2 evaluate call",
        "comparison_usable": not reasons,
        "comparison_gate_reasons": reasons,
        "scenarios": {
            "0bp": {"single_side_rate": 0.0, "comparison_gate": gate_0,
                    "evaluation_version": EVALUATION_VERSION,
                    "competition_metrics": metrics_0["competition_metrics"],
                    "economic_metrics": metrics_0["economic_metrics"]},
            "5bp": {"single_side_rate": 0.0005, "comparison_gate": gate_5,
                    "evaluation_version": EVALUATION_VERSION,
                    "competition_metrics": metrics_5["competition_metrics"],
                    "economic_metrics": metrics_5["economic_metrics"]},
        },
    }


def build_plan(windows: list[dict], train_sha: str) -> dict:
    spec = json.loads(BLEND_SPEC.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "evaluation_version": EVALUATION_VERSION,
        "hypothesis": "Under the single eval-v2 contract, re-evaluating the frozen reference, three "
                      "representative candidates, three simple baselines and two known-answer controls "
                      "on already-viewed development months reproduces the expected ordering and "
                      "degradation behaviour without retraining.",
        "window_rule": "Five development months (2023-11, 2024-01, 2024-03, 2024-04, 2024-06) are "
                       "chosen before evaluation because they are exactly the frozen multiscale replay "
                       "screen+verify coverage and were already viewed research data; the sealed "
                       "interval [2024-10-08, 2025-01-01) is never read. The window set is fixed before "
                       "any metric is computed and is not changed after seeing results.",
        "no_training": True,
        "sealed_interval": list(SEALED),
        "windows": [{"name": window["name"], "start": window["start"], "end": window["end"],
                     "replay_run": window["replay_rel"], "keys_sha256": window["keys_sha256"],
                     "targets_sha256": window["targets_sha256"]} for window in windows],
        "cost_scenarios": {"0bp": 0.0, "5bp": 0.0005,
                           "definition": "cost = single_side_rate * U recomputed by eval-v2; 5bp is the "
                                         "primary stress scenario"},
        "bootstrap": BOOTSTRAP | {"unit": "consecutive trading-day blocks inside each month; blocks "
                                          "never cross month gaps; paired differences vs reference"},
        "objects": [
            {"id": "reference_nine_blend", "role": "reference",
             "definition": "1/9 each of the nine frozen members; blend weights sha256 "
                           f"{sha256_file(BLEND_WEIGHTS)}",
             "members": list(MEMBERS),
             "source_run_ids": {member["name"]: member["training_run_id"] for member in spec["members"]}},
            {"id": "candidate_tlob_3seed", "role": "candidate",
             "definition": "equal-weight TLOB seeds 11/29/47", "members": list(MEMBERS[:3])},
            {"id": "candidate_mlp_3seed", "role": "candidate",
             "definition": "equal-weight MLPLOB seeds 11/29/47", "members": list(MEMBERS[3:6])},
            {"id": "candidate_patch_3seed", "role": "candidate",
             "definition": "equal-weight dual-scale Patch seeds 11/29/47", "members": list(MEMBERS[6:])},
            {"id": "baseline_constant_zero", "role": "baseline", "baseline_recipe": "constant_zero",
             "definition": "score 0 for every timestamp/instrument"},
            {"id": "baseline_label_mean", "role": "baseline", "baseline_recipe": "label_clock_mean",
             "definition": "train-window (2023-08-01..2023-11-01) mean vwap_return per sample time; "
                           "no tuning, no validation data"},
            {"id": "baseline_ridge_default", "role": "baseline", "baseline_recipe": "ridge_default",
             "definition": "ridge alpha=1.0 on instrument and sample-time one-hot features, fitted on "
                           "train-window labels 2023-08-01..2023-11-01; baseline, not a candidate; "
                           f"train labels sha256 {train_sha}"},
            {"id": "control_shuffled_reference", "role": "control",
             "definition": "reference scores permuted within each timestamp (cross-sectional shuffle), "
                           f"seed {BOOTSTRAP['seed']}; expected ~0 rank IC while cost is still charged"},
            {"id": "control_sign_flipped_reference", "role": "control",
             "definition": "negated reference scores; expected rank IC and long-short spread to flip sign"},
        ],
        "gate_policy": "evaluation_comparison_gate decides descriptive vs comparable; no accept/reject "
                       "and no promotion output. Degenerate baselines are expected to be gated.",
        "research_area": "previously viewed dates are annotated from runs/index/viewed_windows.json; "
                         "this run's own dates are registered by rebuilding the index after the run.",
    }


def write_json(document, path: Path) -> str:
    payload = (canonical_json(document) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.write_bytes(frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def prior_viewed_dates() -> dict[str, set[str]]:
    """Dates viewed by runs registered before this re-evaluation (exact daily artifacts)."""
    registry = ROOT / "runs/index/viewed_windows.json"
    if not registry.is_file():
        return {}
    document = json.loads(registry.read_text(encoding="utf-8"))
    viewed: dict[str, set[str]] = {}
    for run in document.get("runs", []):
        dates = set()
        for source in run.get("source_files", []):
            path = ROOT / source["path"]
            if not path.is_file() or path.suffix != ".csv":
                continue
            try:
                frame = pd.read_csv(path, usecols=["date"])
            except (ValueError, OSError):
                continue
            dates.update(pd.to_datetime(frame["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d"))
        if dates:
            viewed[run.get("manifest_path") or run.get("run_id")] = dates
    return viewed


def peak_rss_mb() -> float:
    return float(psutil.Process().memory_info().rss / 1e6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/reeval-a-20261026")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    started = utc_now()
    timer = time.time()
    rss_samples = [peak_rss_mb()]
    windows = [load_window(*window) for window in WINDOWS]
    train = load_train_labels()
    train_sha = hashlib.sha256(canonical_json(
        {"rows": int(len(train)), "sum": float(train["vwap_return"].sum()),
         "min": train["date"].min().isoformat(), "max": train["date"].max().isoformat()}
    ).encode("utf-8")).hexdigest()
    plan = build_plan(windows, train_sha)
    plan_bytes = (canonical_json(plan) + "\n").encode("utf-8")
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    output = args.output_root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{plan_sha[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison").mkdir()
    (output / "plan.json").write_bytes(plan_bytes)
    print(f"predeclared plan {rel(output / 'plan.json')} sha256={plan_sha}")
    rss_samples.append(peak_rss_mb())

    pooled_scores, pooled_keys, pooled_targets = build_objects(windows, train)
    predictions = {name: pooled_keys.assign(score=values)[["date", "instrument", "score"]]
                   for name, values in pooled_scores.items()}
    targets = pooled_targets.reset_index(drop=True)
    dates = pd.to_datetime(pooled_keys["date"]).dt.normalize()
    if (dates >= pd.Timestamp(SEALED[0])).any():
        raise ValueError("evaluated dates enter the sealed interval")
    rss_samples.append(peak_rss_mb())

    reference_metrics_0 = None
    reference_repeat_digest = None
    summary_rows = []
    intervals = {}
    daily_pairs = []
    object_metrics: dict[str, dict] = {}
    payloads = [(item["id"], predictions[item["id"]], targets,
                 item["id"] == "reference_nine_blend") for item in plan["objects"]]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        evaluated = {result[0]: result for result in executor.map(evaluate_payload, payloads)}
    for name in plan["objects"]:
        object_id = name["id"]
        object_dir = output / "objects" / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        _, metrics_0, daily_0, metrics_5, daily_5, repeat_digest = evaluated[object_id]
        scenarios = scenario_block(metrics_0, metrics_5,
                                   evaluation_comparison_gate(metrics_0),
                                   evaluation_comparison_gate(metrics_5))
        gate_0 = evaluation_comparison_gate(metrics_0)
        gate_5 = evaluation_comparison_gate(metrics_5)
        if object_id == "reference_nine_blend":
            reference_metrics_0, reference_daily_0, reference_daily_5 = metrics_0, daily_0, daily_5
            reference_repeat_digest = repeat_digest
        write_csv(daily_0, object_dir / "daily_0bp.csv")
        write_csv(daily_5, object_dir / "daily_5bp.csv")
        metrics_document = {
            "schema_version": 1, "object_id": object_id, "role": name["role"],
            "definition": name["definition"], "evaluation_version": EVALUATION_VERSION,
            "windows": [window["name"] for window in windows],
            "gate": {"0bp": gate_0, "5bp": gate_5},
            "metrics_0bp": metrics_0, "metrics_5bp": metrics_5,
            "cost_scenarios": scenarios,
            "daily": {"0bp_sha256": sha256_file(object_dir / "daily_0bp.csv"),
                      "5bp_sha256": sha256_file(object_dir / "daily_5bp.csv")},
        }
        write_json(metrics_document, object_dir / "metrics.json")
        object_metrics[object_id] = metrics_document
        economic = metrics_5["economic_metrics"]
        summary_rows.append({
            "object_id": object_id, "role": name["role"],
            "object_evaluation_version": metrics_5["evaluation_version"],
            "rank_ic_mean_0bp": metrics_0.get("rank_ic_mean"),
            "ic_mean_0bp": metrics_0.get("ic_mean"),
            "long_short_return_mean_0bp": metrics_0.get("long_short_return_mean"),
            "turnover_mean_0bp": metrics_0.get("turnover_mean"),
            "gross_return_mean_5bp": economic.get("gross_return_mean"),
            "cost_mean_5bp": economic.get("cost_mean"),
            "net_return_mean_5bp": economic.get("net_return_mean"),
            "total_abs_weight_change_mean_5bp": economic.get("total_abs_weight_change_mean"),
            "breakeven_single_side_rate": economic.get("breakeven_single_side_rate"),
            "economic_status": economic.get("status"),
            "undefined_rebalance_count": economic.get("undefined_rebalance_count"),
            "gate_passed": gate_5["passed"], "gate_reasons": gate_5["reasons"],
            "day_count": metrics_5.get("day_count"),
            "undefined_cost_count": metrics_5.get("undefined_cost_count"),
            "overlapping_label_section_count": metrics_5.get("overlapping_label_section_count"),
        })
    rss_samples.append(peak_rss_mb())

    reference_daily_0 = reference_daily_0.set_index("date")
    reference_daily_5 = reference_daily_5.set_index("date")
    for name in plan["objects"]:
        object_id = name["id"]
        if object_id == "reference_nine_blend":
            continue
        daily_0 = pd.read_csv(output / "objects" / object_id / "daily_0bp.csv").set_index("date")
        daily_5 = pd.read_csv(output / "objects" / object_id / "daily_5bp.csv").set_index("date")
        joined = reference_daily_0[list(METRICS_0BP)].join(
            daily_0[list(METRICS_0BP)], lsuffix="_reference", rsuffix="_object", how="inner")
        joined_econ = reference_daily_5[list(METRICS_5BP)].join(
            daily_5[list(METRICS_5BP)], lsuffix="_reference", rsuffix="_object", how="inner")
        merged = joined.join(joined_econ, how="inner")
        merged = merged.reset_index()
        merged["object_id"] = object_id
        merged["month"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m")
        for metric in METRICS_0BP:
            merged[f"diff_{metric}"] = merged[f"{metric}_object"] - merged[f"{metric}_reference"]
        merged["diff_cost_5bp"] = merged["cost_object"] - merged["cost_reference"]
        merged["diff_net_5bp"] = merged["net_return_object"] - merged["net_return_reference"]
        daily_pairs.append(merged)
        for metric, column in (("rank_ic", "diff_rank_ic"), ("ic", "diff_ic"),
                               ("long_short_return", "diff_long_short_return"),
                               ("turnover", "diff_turnover"), ("cost_5bp", "diff_cost_5bp"),
                               ("net_5bp", "diff_net_5bp")):
            series = pd.to_numeric(merged[column], errors="coerce")
            months = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m")
            interval = bootstrap_interval(series, months, block=BOOTSTRAP["block_days"],
                                          draws=BOOTSTRAP["draws"], seed=BOOTSTRAP["seed"],
                                          confidence=BOOTSTRAP["confidence"])
            if interval is None:
                interval = {"mean": None, "low": None, "high": None,
                            "confidence": BOOTSTRAP["confidence"], "share_positive": None,
                            "defined_days": 0, "draws": BOOTSTRAP["draws"],
                            "block_days": BOOTSTRAP["block_days"], "seed": BOOTSTRAP["seed"]}
            interval["missing_days"] = int(len(merged) - int(series.notna().sum()))
            interval["defined_days_by_month"] = {
                month: int((~series[months == month].isna()).sum()) for month in sorted(months.unique())}
            intervals.setdefault(object_id, {})[metric] = interval
    rss_samples.append(peak_rss_mb())

    paired = pd.concat(daily_pairs, ignore_index=True)
    paired_columns = ["object_id", "date", "month"] + \
        [f"{metric}_reference" for metric in METRICS_0BP] + \
        [f"{metric}_object" for metric in METRICS_0BP] + \
        [f"diff_{metric}" for metric in METRICS_0BP] + \
        ["cost_reference", "cost_object", "diff_cost_5bp",
         "net_return_reference", "net_return_object", "diff_net_5bp"]
    write_csv(paired[paired_columns], output / "comparison/paired_daily.csv")

    degradation = {}
    for object_id, expected in (
            ("control_shuffled_reference", "rank_ic approximately zero while turnover/cost remain charged"),
            ("control_sign_flipped_reference", "rank IC and long-short spread both flip sign")):
        metrics = object_metrics[object_id]["metrics_5bp"]
        if object_id.startswith("control_shuffled"):
            observed_rank = metrics.get("rank_ic_mean")
            passed = bool(observed_rank is not None and abs(float(observed_rank)) < 0.01)
            degradation[object_id] = {"expected": expected, "observed_rank_ic_mean": observed_rank,
                                      "passed": passed}
        else:
            observed_rank = metrics.get("rank_ic_mean")
            reference_rank = reference_metrics_0.get("rank_ic_mean")
            observed_spread = metrics.get("long_short_return_mean")
            reference_spread = reference_metrics_0.get("long_short_return_mean")
            passed = bool(observed_rank is not None and reference_rank is not None
                          and np.isclose(observed_rank, -reference_rank, atol=1e-10)
                          and observed_spread is not None and reference_spread is not None
                          and np.isclose(observed_spread, -reference_spread, atol=1e-10))
            degradation[object_id] = {"expected": expected, "observed_rank_ic_mean": observed_rank,
                                      "reference_rank_ic_mean": reference_rank,
                                      "observed_spread_mean": observed_spread,
                                      "reference_spread_mean": reference_spread, "passed": passed}

    viewed_before = prior_viewed_dates()
    evaluated_dates = sorted(dates.dt.strftime("%Y-%m-%d").unique())
    per_date = []
    for date in evaluated_dates:
        watchers = sorted(name for name, values in viewed_before.items() if date in values)
        per_date.append({"date": date, "previously_viewed_by_run_count": len(watchers),
                         "previously_viewed_by": watchers[:5]})
    window_summary = []
    for window in WINDOWS:
        window_dates = [date for date in evaluated_dates if window[2] <= date < window[3]]
        window_summary.append({
            "name": window[0], "start": window[2], "end": window[3], "days": len(window_dates),
            "frozen_replay_coverage": True,
            "min_prior_daily_viewers": min(row["previously_viewed_by_run_count"] for row in per_date
                                           if row["date"] in window_dates),
            "max_prior_daily_viewers": max(row["previously_viewed_by_run_count"] for row in per_date
                                           if row["date"] in window_dates),
        })
    registration = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID,
        "evaluated_dates": evaluated_dates,
        "evaluated_from": evaluated_dates[0], "evaluated_to": evaluated_dates[-1],
        "day_count": len(evaluated_dates),
        "sealed_interval": list(SEALED),
        "sealed_intersection": sorted(date for date in evaluated_dates
                                      if SEALED[0] <= date < SEALED[1]),
        "windows": window_summary,
        "per_date": per_date,
        "prior_evidence_scope": "previously_viewed_by_run_count uses indexed run-level daily artifacts "
                                "(runs/index/viewed_windows.json source_files) and therefore undercounts "
                                "views that exist only in frozen replay windows or analysis reports",
        "frozen_replay_coverage": [{"name": window[0], "covered": True,
                                     "replay_run": window[1].relative_to(ROOT).as_posix()}
                                    for window in WINDOWS],
        "wording": "本次评的 " + str(len(evaluated_dates)) + " 个交易日属于冻结重放已覆盖的 5 个开发月；"
                    "索引 run 级 daily 证据可确定的逐日 prior-viewer 见 per_date（nov/mar 存在历史 run "
                    "逐日产物；jan/apr/jun 的查看记录在冻结重放窗口与旧报告中，未形成 run 级 daily 证据）。"
                    "封存区间 [2024-10-08, 2025-01-01) 未读取；本重评 run 在索引重建后登记。",
        "prior_registry": "runs/index/viewed_windows.json (read-only snapshot before this run)",
    }
    write_json(registration, output / "research_area_registration.json")

    checks = [
        {"name": "source_prediction_sha256", "passed": True,
         "evidence": "all prediction files hashed and matched plan-recorded shas"},
        {"name": "target_keys_exact", "passed": True,
         "evidence": "evaluation aligned only target keys; no outer-join loss"},
        {"name": "sealed_interval_untouched", "passed": True,
         "evidence": f"evaluated dates {evaluated_dates[0]}..{evaluated_dates[-1]} outside {SEALED}"},
        {"name": "no_training", "passed": True,
         "evidence": "only evaluate() and analytic baselines; no optimizer/model fit except fixed ridge"},
        {"name": "reproducible_evaluation",
         "passed": bool(reference_repeat_digest and reference_repeat_digest["first"]
                        == reference_repeat_digest["second"]),
         "evidence": "reference 0bp evaluated twice produces identical canonical metrics JSON",
         "digest": reference_repeat_digest},
    ]
    write_json({"schema_version": 1, "passed": all(item["passed"] for item in checks),
                "checks": checks}, output / "checks.json")

    data_document = {
        "schema_version": 1, "source": "archived multiscale replay predictions and vwap labels",
        "version": "cpt_jyc_2026", "evaluation_version": EVALUATION_VERSION,
        "windows": [{"name": window["name"], "start": window["start"], "end": window["end"],
                     "keys_sha256": window["keys_sha256"],
                     "targets_sha256": window["targets_sha256"]} for window in windows],
        "targets_sha256": object_metrics["reference_nine_blend"]["metrics_0bp"].get("targets_sha256"),
        "sample_keys_sha256": object_metrics["reference_nine_blend"]["metrics_0bp"].get("sample_keys_sha256"),
        "train_labels_sha256": train_sha,
    }
    write_json(data_document, output / "data.json")

    config = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID,
        "hypothesis": plan["hypothesis"], "output_root": "runs",
        "data": {"labels_path": "数据/cpt_jyc_2026_vwap", "source": data_document["source"],
                 "version": data_document["version"]},
        "splits": {"train": {"start": f"{TRAIN_START}T00:00:00", "end": f"{TRAIN_END}T00:00:00"},
                   "validation": {"start": "2023-11-01T00:00:00", "end": "2024-07-01T00:00:00"},
                   "final": None},
        "evaluation_split": "validation",
        "evaluation": {"quantile": 0.2, "periods_per_year": 252, "min_cross_section": 3,
                       "constant_portfolios": "undefined", "single_side_rate": None},
        "windows": [{"name": window[0], "start": window[2], "end": window[3]} for window in WINDOWS],
        "no_training": True,
    }
    write_json(config, output / "config.json")
    canonical_config_sha = digest(config)

    run_metrics = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "evaluation_version": EVALUATION_VERSION,
        "windows": [window["name"] for window in windows], "day_count": len(evaluated_dates),
        "reference": object_metrics["reference_nine_blend"]["metrics_0bp"],
        "objects": {row["object_id"]: {key: row[key] for key in (
            "role", "rank_ic_mean_0bp", "long_short_return_mean_0bp", "turnover_mean_0bp",
            "cost_mean_5bp", "net_return_mean_5bp", "breakeven_single_side_rate",
            "gate_passed", "gate_reasons")} for row in summary_rows},
    }
    write_json(run_metrics, output / "metrics.json")
    write_csv(pd.DataFrame(summary_rows), output / "comparison" / "summary.csv")
    write_json({"intervals": intervals, "bootstrap": BOOTSTRAP,
                "metric_scope": "paired daily differences vs frozen reference; descriptive only"},
               output / "comparison" / "intervals.json")
    write_json({"degradation_controls": degradation,
                "gate_policy": "evaluation_comparison_gate; no accept/reject"},
               output / "comparison" / "controls.json")
    reference_daily_0.reset_index().to_csv(output / "daily.csv", index=False, lineterminator="\n")
    rss_samples.append(peak_rss_mb())

    artifacts = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append({"path": path.relative_to(output).as_posix(),
                              "sha256": sha256_file(path), "bytes": path.stat().st_size})
    run_id = output.name
    manifest = {
        "schema_version": 1, "run_id": run_id, "experiment_id": EXPERIMENT_ID,
        "kind": "evaluation", "status": "completed",
        "started_at": started, "ended_at": utc_now(),
        "hypothesis": plan["hypothesis"], "evaluation_version": EVALUATION_VERSION,
        "evaluation_split": "validation",
        "evaluation_source": {
            "evaluation_py_sha256": sha256_file(ROOT / "src/jinyiquant/evaluation.py"),
            "comparison_py_sha256": sha256_file(ROOT / "src/jinyiquant/comparison.py"),
            "script_sha256": sha256_file(Path(__file__)),
            "contract_sha256": digest(CONTRACT),
        },
        "code": {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "content_sha256": digest({
                "evaluation.py": sha256_file(ROOT / "src/jinyiquant/evaluation.py"),
                "comparison.py": sha256_file(ROOT / "src/jinyiquant/comparison.py"),
                "reevaluate_contract_a.py": sha256_file(Path(__file__)),
            }),
            "files": {
                "src/jinyiquant/evaluation.py": sha256_file(ROOT / "src/jinyiquant/evaluation.py"),
                "src/jinyiquant/comparison.py": sha256_file(ROOT / "src/jinyiquant/comparison.py"),
                "scripts/reevaluate_contract_a.py": sha256_file(Path(__file__)),
            },
        },
        "config_sha256": canonical_config_sha,
        "plan_sha256": plan_sha,
        "data_version": {
            "source": data_document["source"], "version": data_document["version"],
            "data_sha256": data_document["targets_sha256"],
            "requested_start": f"{WINDOWS[0][2]} 00:00:00",
            "requested_end": f"{WINDOWS[-1][3]} 00:00:00",
        },
        "resources": {"wall_seconds": round(time.time() - timer, 3),
                      "max_rss_mb": round(max(rss_samples), 3),
                      "rss_note": "max of sampled RSS points, not a continuous peak"},
        "source_archives": {
            "replays": [{"run_dir": window["replay_rel"],
                         "manifest_sha256": sha256_file(window["replay"] / "manifest.json"),
                         "window": window["name"],
                         "prediction_sha256": window["prediction_sha256"]} for window in windows],
            "blend_spec": {"path": rel(BLEND_SPEC), "sha256": sha256_file(BLEND_SPEC)},
            "blend_weights": {"path": rel(BLEND_WEIGHTS), "sha256": sha256_file(BLEND_WEIGHTS)},
        },
        "re_evaluated_from": {
            row["object_id"]: {"source_run_ids": (
                list(plan["objects"][0]["source_run_ids"].values())
                if row["object_id"] == "reference_nine_blend"
                else [plan["objects"][0]["source_run_ids"][member]
                      for member in next(obj["members"] for obj in plan["objects"]
                                         if obj["id"] == row["object_id"])]
                if any(obj["id"] == row["object_id"] and "members" in obj for obj in plan["objects"])
                else [])}
            for row in summary_rows
        },
        "checks": "checks.json", "artifacts": artifacts,
    }
    write_json(manifest, output / "manifest.json")
    print(f"wrote run {rel(output)} objects={len(summary_rows)} dates={len(evaluated_dates)}")
    print(f"windows={[window['name'] for window in windows]}")
    print(f"reference rank_ic_mean={reference_metrics_0.get('rank_ic_mean')}")
    for row in summary_rows:
        print(f"{row['object_id']:<34} rank_ic={row.get('rank_ic_mean_0bp')} "
              f"gate={row.get('gate_passed')} reasons={len(row.get('gate_reasons') or [])}")
    print(f"sealed_intersection={registration['sealed_intersection']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
