"""Freeze 10/40-day dates and purge training targets; never fit or open retained labels."""

import argparse
import copy
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from jinyiquant.config import Period, RunConfig
from jinyiquant.data import select_targets, validate_keys
from jinyiquant.io import digest, file_sha256, read_json, utc_now, write_json
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir
from modified_tlob.schema import lob_columns

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [11, 29, 47]
TIMES = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]
RESERVED = [("2023-08-03", "2023-08-05"), ("2024-07-01", "2024-07-27"),
            ("2024-10-08", "2025-01-01")]
EXP = "training-coverage-20261017"


def freeze_windows(prior, calendar):
    """Keep registered cutoffs, evaluation dates and recipe; vary only training start."""
    calendar = pd.DatetimeIndex(calendar).sort_values().unique()
    if calendar.tz is not None or calendar.hasnans or not calendar.equals(calendar.normalize()):
        raise ValueError("Calendar must contain exchange-local dates")
    windows = []
    for old in prior["protocol"]["candidate_windows"]:
        name, cutoff = old["name"], old["train_end_exclusive"]
        evaluation = calendar[(calendar >= old["evaluation_start"]) &
                              (calendar < old["evaluation_end_exclusive"])].strftime("%Y-%m-%d").tolist()
        if evaluation != old["evaluation_dates"]:
            raise ValueError("Evaluation dates differ from existing registration")
        arms = {}
        for count in (10, 40):
            days = calendar[calendar < cutoff][-count:].strftime("%Y-%m-%d").tolist()
            if len(days) != count or (count == 10 and days != old["train_dates"]):
                raise ValueError("Training dates differ from official last-N-day rule")
            if any(lo <= day < hi for day in days for lo, hi in RESERVED):
                raise ValueError("Training dates intersect a retained evaluation period")
            cfg = copy.deepcopy(prior["configs"]["research_a"])
            cfg.update(experiment_id=EXP, seed=11,
                       hypothesis="40 training dates improve subsequent ranking and long-short stability")
            cfg["splits"]["train"] = {"start": days[0], "end": cutoff}
            if name == "locked":
                # Structural validation split is the exchange holiday gap; never scored.
                cfg["splits"]["validation"] = {"start": cutoff, "end": old["evaluation_start"]}
                cfg["evaluation_split"] = "final"
                cfg["final_access"] = {"max_views": 1, "purpose": "Only after frozen confirmation passes"}
            else:
                cfg["splits"]["validation"] = {"start": old["evaluation_start"],
                                                "end": old["evaluation_end_exclusive"]}
            valid = RunConfig.model_validate(cfg)
            if (tuple(valid.data.feature_columns) != lob_columns(3) or valid.data.lookback != 128
                    or list(valid.data.sample_times) != TIMES or list(valid.data.training_sample_times) != TIMES
                    or valid.model.max_steps != 1500 or valid.model.batch_size != 32
                    or valid.model.learning_rate != 1e-4 or valid.model.target_scaling != "train_std"
                    or valid.model.time_loss_weighting != "none" or valid.model.book_representation != "raw"):
                raise ValueError("Inherited recipe differs from requested recipe")
            arms[str(count)] = {"train_dates": days, "train_end_exclusive": cutoff, "config": valid.snapshot()}
        left, right = [copy.deepcopy(arms[str(n)]["config"]) for n in (10, 40)]
        left["splits"]["train"].pop("start")
        right["splits"]["train"].pop("start")
        if left != right:
            raise ValueError("Undeclared recipe difference beyond training start")
        windows.append({"name": name, "role": "research_used" if name.startswith("research_") else name,
                        "evaluation_start": old["evaluation_start"],
                        "evaluation_end_exclusive": old["evaluation_end_exclusive"],
                        "evaluation_dates": evaluation, "arms": arms,
                        "fit_once_before_evaluation": True, "evaluation_label_views_this_round": 0})
    return windows


def purge_training(labels, universe, days, cutoff):
    """Use actual label_end if present, and retain only frozen official-universe keys."""
    labels = validate_keys(labels, "training labels")
    labels = labels.loc[labels.date.dt.strftime("%Y-%m-%d").isin(days)].copy()
    universe = validate_keys(universe, "training universe")
    membership = labels.merge(universe[["date", "instrument"]], on=["date", "instrument"],
                              how="left", indicator=True, validate="one_to_one")
    outside = int(membership._merge.ne("both").sum())
    labels = membership.loc[membership._merge.eq("both")].drop(columns="_merge")
    selected = select_targets(labels, Period(start=days[0], end=cutoff), sample_times=TIMES)
    checks = selected.attrs["selection"].copy()
    checks.update(outside_official_universe=outside, actual_label_end_available="label_end" in labels,
                  max_selected_label_end=None if selected.empty else selected.label_end.max().isoformat(),
                  selected_key_label_sha256=digest(selected.assign(
                      date=selected.date.astype(str), label_end=selected.label_end.astype(str)
                  ).to_dict("records")))
    # Official targets exclude overnight returns. With midnight role boundaries,
    # the same-day exchange close is a conservative bound even if the lunch
    # clock is unknown; do not mislabel this bound as the actual endpoint.
    bound = selected.label_end if "label_end" in labels else selected.date.dt.normalize() + pd.Timedelta(hours=15)
    checks.update(boundary_proof="source_label_end" if "label_end" in labels else "official_no_overnight_upper_bound_1500",
                  max_conservative_label_end=None if selected.empty else bound.max().isoformat())
    if selected.empty or not selected.label_end.lt(pd.Timestamp(cutoff)).all():
        raise ValueError("Empty training targets or endpoints crossing frozen cutoff")
    if not bound.lt(pd.Timestamp(cutoff)).all():
        raise ValueError("Cannot prove training labels end before the cutoff")
    for lo, hi in RESERVED:
        if ((selected.date < hi) & (bound >= lo)).any():
            raise ValueError("Training label intervals intersect a retained period")
    return selected, checks


def read_training_labels(config, days, rec):
    folder = Path(config["data"]["labels_path"])
    frames, sources = [], {}
    for month in sorted({day[:7] for day in days}):
        paths = list(folder.glob(f"*{month}*.parquet"))
        if len(paths) != 1:
            raise ValueError(f"Expected one training label shard for {month}")
        path = paths[0]
        names = pq.read_schema(path).names
        cols = [k for k in ["date", "instrument", "target", "vwap_return", "label_end"] if k in names]
        frame = pd.read_parquet(path, columns=cols,
                                filters=[("date", ">=", pd.Timestamp(days[0])),
                                         ("date", "<", pd.Timestamp(days[-1]) + pd.Timedelta(days=1))])
        frame = frame.loc[frame.date.dt.strftime("%Y-%m-%d").isin(days)]
        if any(((frame.date >= lo) & (frame.date < hi)).any() for lo, hi in RESERVED):
            raise ValueError("Reader decoded retained evaluation labels")
        frames.append(frame)
        sources[str(path.resolve())] = file_sha256(path)
        rec.event("data_validation", "training_labels_read", month=month, rows=len(frame))
    return pd.concat(frames, ignore_index=True), sources


def freeze(output, output_root, prior_path):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Frozen plans are immutable; select a new output name")
    prior = read_json(prior_path)
    run = new_run_dir(Path(output_root))
    rec = Recorder(run, EXP)
    write_json(run / "manifest.json", {"status": "running", "run_id": run.name, "experiment_id": EXP,
               "started_at": utc_now(), "code": capture_source(run), "environment": environment()})
    rec.event("startup", "started", new_training=0)
    try:
        path = Path(prior["configs"]["research_a"]["data"]["instruments_path"])
        universe = pd.read_parquet(path, columns=["date", "instrument"])
        calendar = pd.DatetimeIndex(pd.to_datetime(universe.date).dt.normalize().unique()).sort_values()
        windows = freeze_windows(prior, calendar)
        inputs, audits = {str(path): file_sha256(path)}, []
        for document in [ROOT / "比赛说明/比赛说明.txt", ROOT / "比赛说明/可调用数据说明.txt"]:
            inputs[str(document)] = file_sha256(document)
        for window in windows:
            broad = window["arms"]["40"]
            labels, sources = read_training_labels(broad["config"], broad["train_dates"], rec)
            inputs.update(sources)
            for count, arm in window["arms"].items():
                selected, checks = purge_training(labels, universe, arm["train_dates"], arm["train_end_exclusive"])
                target = run / f"{window['name']}_{count}_training_targets.parquet"
                selected.to_parquet(target, index=False)
                audit = {"window": window["name"], "training_days": int(count), **checks,
                         "artifact": str(target.resolve()), "sha256": file_sha256(target)}
                audits.append(audit)
                rec.event("data_validation", "purge_completed", **audit)
        if file_sha256(prior["registry_path"]) != prior["registry_sha256"]:
            raise ValueError("Original shared role registry changed")
        plan = {"schema_version": 2, "experiment_id": EXP, "created_at_utc": utc_now(),
                "hypothesis": "Expanding training dates from 10 to 40 reduces recent-window sensitivity and improves subsequent ranking and long-short stability.",
                "research_seeds": SEEDS, "deployment_seed": 11, "loss": "MSE", "smoothing": "none",
                "fit_policy": "Once per origin, arm and prescribed seed; fixed weights afterward; no score-based retries.",
                "windows": windows, "old_final": {"start": "2023-08-03", "end_exclusive": "2023-08-05", "role": "reserved"},
                "access_policy": {"research": "A/B already used: screening/calibration/replication only",
                                  "confirmation": "One opening after complete recipe and thresholds freeze",
                                  "locked": "One opening after confirmation passes without recipe changes",
                                  "evaluation_label_views_this_round": 0},
                "calendar_source": str(path), "calendar_sha256": file_sha256(path),
                "protocol_source": str(prior_path), "protocol_sha256": file_sha256(prior_path),
                "registry_path": prior["registry_path"], "registry_sha256": prior["registry_sha256"],
                "training_selection": audits, "inputs_sha256": inputs,
                "purge_policy": "Source label_end preferred; endpoint < training cutoff; frozen date membership. Without label_end, official no-overnight labels are conservatively bounded by same-day 15:00, strictly before the midnight cutoff. Existing intraday selection clock (11:30 -> 13:30) remains an unverified assumption, not actual endpoint verification.",
                "execution_status": "frozen_not_trained",
                "acceptance": {"primary": "paired daily RankIC", "secondary": ["long_short_return", "turnover", "worst_day"],
                               "thresholds_status": "not calibrated in this task; candidate training requires separate baseline-only threshold freeze",
                               "replication_seeds": SEEDS, "seed_selection": "prohibited", "model_promoted": False},
                "diagnostic_budget": {"new_training": 0, "additional_weight_month_inferences_max": 6},
                "literature": {"backbone": "https://arxiv.org/html/2502.15757v3",
                               "distribution_shift": "https://arxiv.org/abs/2306.09862",
                               "restriction": "Classification scores do not establish returns; no causal weight-age claim."},
                "run_dir": str(run.resolve())}
        write_json(output, plan)
        write_json(run / "plan.json", plan)
        checks = {"passed": True, "new_training": 0, "reserved_label_views": 0,
                  "official_calendar_and_original_cutoffs_match": True,
                  "both_arm_configs_valid": True, "purged_training_sets": len(audits),
                  "training_label_boundaries_proven": True,
                  "actual_endpoint_verification_complete": all(a["actual_label_end_available"] for a in audits),
                  "label_clock_limitation": "Source lacks label_end; trading-clock assumption explicit.",
                  "plan_sha256": file_sha256(output)}
        write_json(run / "checks.json", checks)
        write_json(run / "metrics.json", {"training_selection": audits, "new_training": 0})
        rec.event("shutdown", "completed", **checks)
        finish(run, "completed", 0)
    except BaseException as exc:
        rec.failure("freeze", exc)
        finish(run, "failed", 1, error={"reason": str(exc)})
        raise
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/training_coverage_20261017_v3_frozen.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/training-coverage-freeze")
    parser.add_argument("--prior", type=Path, default=ROOT / "configs/fixed_seed_ensemble_20261017_plan.json")
    args = parser.parse_args()
    print(freeze(args.output, args.output_root, args.prior))
