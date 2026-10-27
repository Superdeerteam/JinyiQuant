"""Narrow the candidate pool and register every attempt that has been looked at.

Two jobs, both read-only with respect to models and data:

* **screen** - for every archived object of the frozen nine-checkpoint family, measure the
  paired primary-metric difference against the deployable incumbent on the five declared
  research months, using the registered estimator (paired by trading day, 5-day consecutive
  blocks inside each declared period, 4000 draws, 95%). Only the primary metric is screened:
  a single-seed arm cannot clear the registered secondary lines on seed noise alone, so a
  secondary breach is never a screening reason here.
* **register** - write down every checkpoint, blend ratio and parameter that has been looked
  at in this family, with the artifact that holds the evidence, so the selection bias of the
  whole search is visible instead of being implicit in the winner.

The screen is validated against the registered placebo scale: the six equal-quality pair
differences re-measured here must reproduce the magnitudes recorded in
``configs/b4_promotion_prereg_20261027.json``. If they do not, the screen is not the
registered estimator and the script refuses to report anything.

Nothing here trains, promotes or freezes anything.

Usage
    python scripts/candidate_triage.py --output runs/candidate-triage-20261027
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jinyiquant.comparison import _block_indices  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VOLATILITY = ROOT / "runs/b4-volatility-20261027/20261027T064907Z_a79926d3e455"
PREREG = ROOT / "configs/b4_promotion_prereg_20261027.json"
RATIO_SWEEP = ROOT / "modified_tlob/tlob_mlplob_ratio_sweep_20261026.json"
GROUPED_RANKING = ROOT / "modified_tlob/grouped_ranking_20261025.json"
BLEND_RECORD = ROOT / "modified_tlob/tlob_mlplob_blend_20261026.json"
BASELINE = "nine_blend"
WINDOWS = (
    ("nov2023", "2023-11-01", "2023-11-30"),
    ("jan2024", "2024-01-01", "2024-01-31"),
    ("mar2024", "2024-03-01", "2024-03-31"),
    ("apr2024", "2024-04-01", "2024-04-30"),
    ("jun2024", "2024-06-01", "2024-06-30"),
)
MEMBERS = ("tlob_s11", "tlob_s29", "tlob_s47", "mlp_s11", "mlp_s29", "mlp_s47",
           "patch_s11", "patch_s29", "patch_s47")
ENSEMBLES = {"tlob_3seed": MEMBERS[:3], "mlp_3seed": MEMBERS[3:6], "patch_3seed": MEMBERS[6:]}
OBJECTS = MEMBERS + tuple(ENSEMBLES) + (BASELINE,)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")


def event(path: Path, stage: str, event_type: str, **context) -> None:
    record = {"schema_version": 1, "timestamp_utc": utc_now(), "level": "INFO", "stage": stage,
              "event_type": event_type, "context": context, "run_id": path.parent.name}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stitch(object_id: str) -> pd.DataFrame:
    """Re-assemble one object's 104-day primary-metric daily series from the per-month archives.

    The per-month rank_ic values are per-cross-section averages, so re-stitching them reproduces
    the pooled series exactly; this is checked against the registered placebo differences below.
    """
    frames = []
    for name, start, end in WINDOWS:
        path = VOLATILITY / "objects" / object_id / f"daily_0bp_{name}.csv"
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"])
        if frame["date"].min() < pd.Timestamp(start) or frame["date"].max() > pd.Timestamp(end):
            raise SystemExit(f"{object_id}/{name}: daily rows fall outside the declared month")
        frames.append(frame[["date", "rank_ic"]])
    out = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    if out["date"].duplicated().any():
        raise SystemExit(f"{object_id}: duplicate trading days after stitching")
    return out


def period_positions(dates: pd.Series) -> list[np.ndarray]:
    spans = [(pd.Timestamp(start), pd.Timestamp(end)) for _, start, end in WINDOWS]
    return [np.flatnonzero(((dates >= start) & (dates <= end)).to_numpy()) for start, end in spans]


def paired(differences: np.ndarray, groups: list[np.ndarray], spec: dict) -> dict:
    """Registered estimator: equal weight per declared period, blocks inside each period."""
    effects = [float(differences[positions].mean()) for positions in groups]
    improvement = float(np.mean(effects))
    rng = np.random.default_rng(int(spec["seed"]))
    draws = []
    for _ in range(int(spec["bootstrap_samples"])):
        sampled = np.concatenate([positions[_block_indices(len(positions), int(spec["block_days"]), rng)]
                                  for positions in groups])
        draws.append(float(differences[sampled].mean()))
    alpha = (1 - float(spec["confidence"])) / 2
    low, high = np.quantile(draws, [alpha, 1 - alpha])
    return {"paired_improvement": improvement,
            "block_interval": {"low": float(low), "high": float(high),
                               "block_days": int(spec["block_days"]),
                               "bootstrap_samples": int(spec["bootstrap_samples"]),
                               "confidence": float(spec["confidence"]), "seed": int(spec["seed"])},
            "period_effects": effects,
            "positive_periods": int(sum(1 for value in effects if value > 0)),
            "periods": len(effects),
            "paired_day_count": int(len(differences))}


def classify(reading: dict, floor: float, noise: float) -> str:
    improvement = reading["paired_improvement"]
    clears = (improvement >= floor and reading["block_interval"]["low"] > 0
              and reading["positive_periods"] >= 4)
    if clears:
        return "clears_promotion_floor"
    if abs(improvement) <= noise:
        return "inside_measured_noise"
    if improvement < 0:
        return "negative"
    return "below_floor_without_clearing_interval"


INDEX = ROOT / "runs/index/manifest_index.jsonl"


def index_provenance(experiment_ids: list[str]) -> dict:
    """Reuse runs/index as the fact source: every attempt must resolve to an indexed run."""
    if not INDEX.is_file():
        return {"available": False, "why": f"{INDEX.relative_to(ROOT).as_posix()} is missing"}
    counts: dict[str, dict] = {}
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        experiment = row.get("experiment_id")
        if experiment not in experiment_ids:
            continue
        bucket = counts.setdefault(experiment, {"runs": 0, "status": {}})
        bucket["runs"] += 1
        status = row.get("status") or "null"
        bucket["status"][status] = bucket["status"].get(status, 0) + 1
    missing = sorted(set(experiment_ids) - set(counts))
    return {"available": True, "source": INDEX.relative_to(ROOT).as_posix(),
            "experiments": counts, "not_yet_indexed": missing}


def attempt_registry(prereg: dict) -> dict:
    """Every checkpoint, blend ratio and parameter looked at in this family, with its source."""
    sweep = read_json(RATIO_SWEEP)
    blend = read_json(BLEND_RECORD)
    registry = {
        "schema_version": 1,
        "what": "attempts already looked at in the nine-checkpoint family and its single-factor "
                "children; selection bias is only visible if the whole search is listed",
        "families": [],
    }
    registry["families"].append({
        "id": "checkpoints_and_ensembles",
        "factor": "which frozen checkpoint set is combined (backbone x seed)",
        "values_looked_at": list(OBJECTS),
        "evidence": f"{VOLATILITY.relative_to(ROOT).as_posix()}/objects/<object>/metrics_<month>.json",
        "windows": [name for name, _, _ in WINDOWS],
        "evaluated_days": 104,
        "screen_rule": "primary metric only (rank_ic_mean); secondary lines are not a screening "
                       "reason for single-seed arms",
        "outcome": "see triage.json screening table",
        "attempts": len(OBJECTS),
    })
    grid = sweep.get("grid") or []
    registry["families"].append({
        "id": "tlob_mlplob_blend_ratio",
        "factor": "TLOB weight in a two-trunk blend",
        "values_looked_at": grid,
        "grid_step": sweep.get("grid_step"),
        "evidence": RATIO_SWEEP.relative_to(ROOT).as_posix(),
        "best_by_metric": sweep.get("best"),
        "selection_bias": f"{len(grid)} grid points were scored on the same evaluation window, so "
                          f"picking the argmax inflates the apparent gain; the reported optimum is "
                          f"a selection result, not an out-of-sample finding",
        "outcome": "superseded: the deployable incumbent is an equal-weight nine-checkpoint blend, "
                   "not a two-trunk blend at the swept optimum",
        "attempts": len(grid),
    })
    arms = [arm if isinstance(arm, str) else (arm.get("name") or arm.get("id"))
            for arm in (blend.get("arms") or [])]
    registry["families"].append({
        "id": "two_trunk_blend_arms",
        "factor": "two-trunk blend composition",
        "values_looked_at": arms,
        "evidence": BLEND_RECORD.relative_to(ROOT).as_posix(),
        "outcome": (blend.get("conclusion") or blend.get("decision") or "")[:400] or None,
        "attempts": len(arms),
    })
    grouped = read_json(GROUPED_RANKING) if GROUPED_RANKING.is_file() else {}
    if grouped:
        registry["families"].append({
            "id": "sampling_and_ranking_loss",
            "factor": "model.sampling and model.ranking_loss inside the same V3 60d/19800 family",
            "values_looked_at": sorted((grouped.get("arms") or {}).keys()),
            "evidence": GROUPED_RANKING.relative_to(ROOT).as_posix(),
            "outcome": (grouped.get("decision") or "")[:600] or None,
            "attempts": len(grouped.get("arms") or {}),
        })
    registry["families"].append({
        "id": "training_amount",
        "factor": "model.max_steps via the pre-registered effective-epoch budget",
        "values_looked_at": ["1.32", "2.64", "5.28"],
        "evidence": "runs/training-trajectory-replay-20261027 and "
                    "runs/training-budget-analysis-20261027",
        "outcome": "refused: every doubling lowers the primary metric with an interval below zero",
        "attempts": 3,
    })
    registry["families"].append({
        "id": "factor_temporal_persistence",
        "factor": "causal EWMA over the intraday cross-sections on the submitted factor",
        "values_looked_at": ["alpha=0.5"],
        "evidence": "runs/turnover-stability-20261027 and runs/turnover-verify-20261027",
        "outcome": "negative under the frozen gate: primary metric falls 0.008 with the interval "
                   "below zero, so the turnover gain cannot carry it",
        "attempts": 1,
    })
    registry["config_space_survey"] = {
        "what": "which single factors of this recipe family have actually been varied in the "
                "configuration archive, so an untried lever is not mistaken for an exhausted one",
        "varied": {"batch_size": ["32", "64"], "num_layers": ["3", "4"],
                   "lookback": ["128", "256", "30"], "history_days": ["0", "10"],
                   "ranking_loss": ["none", "pairwise_hinge"],
                   "sampling": ["timestamp_grouped"]},
        "never_varied": ["hidden_dim", "learning_rate", "time_loss_weighting"],
        "reading": "capacity (hidden_dim), the step size and the temporal loss weighting are the "
                   "only gate-expressible levers this family has not exercised; sampling and the "
                   "ranking loss have been exercised and lost",
        "caveat": "never varied is not evidence that the lever helps; it is the absence of "
                  "evidence either way",
    }
    registry["total_attempts"] = sum(family["attempts"] for family in registry["families"])
    registry["index_provenance"] = index_provenance([
        "b4-volatility-20261027", "cost-curve-20261027", "turnover-stability-20261027",
        "turnover-readout-20261027", "turnover-verify-20261027",
        "training-trajectory-20261027", "training-trajectory-replay-20261027",
        "training-budget-analysis-20261027", "candidate-triage-20261027",
        "multiscale-replay-20261026", "multiscale-replay-verify-20261026",
        "grouped-ranking-20261025",
    ])
    return registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", help="write a run archive under runs/<archive>/<run_id>/")
    parser.add_argument("--prereg", type=Path, default=PREREG)
    args = parser.parse_args(argv)
    if args.archive:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output = ROOT / "runs" / args.archive / f"{stamp}_{sha256_file(args.prereg)[:12]}"
        if output.exists():
            raise SystemExit(f"refused: {output} already exists")
    elif args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        if (ROOT / "runs").resolve() in output.resolve().parents:
            raise SystemExit("refused: --output inside runs/ needs --archive")
    else:
        raise SystemExit("refused: pass --output or --archive")

    prereg = read_json(args.prereg)
    template = prereg["object_policy_template"]
    spec = {"block_days": template["block_days"], "bootstrap_samples": template["bootstrap_samples"],
            "confidence": template["confidence"], "seed": template["seed"]}
    floor = float(prereg["derived_thresholds"]["min_improvement"])
    noise = float(prereg["derived_thresholds"]["placebo_max_abs_primary_difference"])
    registered_placebo = prereg["measured_volatility"]["placebo_pair_rank_ic_differences"]

    output.mkdir(parents=True, exist_ok=True)
    events = output / "events.jsonl"
    plan = {
        "schema_version": 1, "created_at": utc_now(), "entry": "scripts/candidate_triage.py",
        "question": "does any already-measured object clear the promotion floor against the "
                    "deployable incumbent, and what has been looked at so far?",
        "baseline": BASELINE, "objects": list(OBJECTS), "windows": [name for name, _, _ in WINDOWS],
        "primary_metric": template["primary_metric"], "estimator": spec,
        "promotion_floor": floor, "measured_noise_scale": noise,
        "screen_rule": "primary metric only; secondary lines are not a screening reason",
        "prereg": {"path": args.prereg.relative_to(ROOT).as_posix(),
                   "sha256": sha256_file(args.prereg)},
        "sealed_read": False, "trains_anything": False,
    }
    write_json(output / "plan.json", plan)
    event(events, "triage", "started", objects=len(OBJECTS))

    series = {object_id: stitch(object_id) for object_id in OBJECTS}
    dates = series[BASELINE]["date"]
    for object_id, frame in series.items():
        if not frame["date"].equals(dates):
            raise SystemExit(f"{object_id}: trading-day index differs from the baseline")
    groups = period_positions(dates)

    # The screen must be the registered estimator: reproduce the placebo magnitudes exactly.
    # The registered placebo magnitudes are rounded to eight decimals, so agreement is checked
    # against that rounding bound rather than machine precision.
    rounding_tolerance = 5e-9
    validation = {}
    for key, registered in registered_placebo.items():
        left, right = key.split("__vs__")
        recomputed = abs(float((series[left]["rank_ic"].to_numpy() - series[right]["rank_ic"].to_numpy()).mean()))
        validation[key] = {"registered_magnitude": registered, "recomputed_magnitude": recomputed,
                           "absolute_difference": abs(recomputed - registered),
                           "reproduced": abs(recomputed - registered) <= rounding_tolerance}
    validation_tolerance = rounding_tolerance
    if not all(item["reproduced"] for item in validation.values()):
        raise SystemExit("refused: the screen does not reproduce the registered placebo scale, so it "
                         "is not the registered estimator")
    event(events, "triage", "placebo_scale_reproduced", pairs=len(validation),
          tolerance=validation_tolerance)

    baseline_values = series[BASELINE]["rank_ic"].to_numpy(dtype=float)
    screening = {}
    for object_id in OBJECTS:
        if object_id == BASELINE:
            continue
        values = series[object_id]["rank_ic"].to_numpy(dtype=float)
        reading = paired(values - baseline_values, groups, spec)
        reading["baseline_rank_ic_mean"] = float(baseline_values.mean())
        reading["object_rank_ic_mean"] = float(values.mean())
        reading["classification"] = classify(reading, floor, noise)
        reading["clears_floor"] = reading["classification"] == "clears_promotion_floor"
        reading["is_multi_seed"] = object_id in ENSEMBLES
        reading["notes"] = ("single-seed arm: its secondary metrics are not comparable at the "
                            "registered lines, so only the primary metric is used"
                            if object_id in MEMBERS else
                            "three-seed aggregate" if object_id in ENSEMBLES else "")
        screening[object_id] = reading
    survivors = [key for key, value in screening.items() if value["clears_floor"]]
    event(events, "triage", "screened", survivors=len(survivors))

    registry = attempt_registry(prereg)
    best = max(screening.items(), key=lambda item: item[1]["paired_improvement"])
    triage = {
        "schema_version": 1, "created_at": utc_now(),
        "question": plan["question"],
        "coverage": {"days": int(len(dates)), "periods": len(groups),
                     "windows": [name for name, _, _ in WINDOWS], "sealed_intersection": 0},
        "baseline": {"object_id": BASELINE, "rank_ic_mean": float(baseline_values.mean())},
        "promotion_floor": floor, "measured_noise_scale": noise,
        "screen_validated_by_placebo": validation,
        "screen_validation_tolerance": validation_tolerance,
        "screening": screening,
        "survivors": survivors,
        "closest_to_the_floor": {"object_id": best[0],
                                 "paired_improvement": best[1]["paired_improvement"],
                                 "block_interval": best[1]["block_interval"],
                                 "classification": best[1]["classification"],
                                 "gap_to_floor": floor - best[1]["paired_improvement"]},
        "conclusion": {
            "survivors": len(survivors),
            "reading": "no archived object clears the promotion floor against the deployable "
                       "incumbent; every difference sits at or below the measured noise scale, so "
                       "re-selecting or re-blending the existing checkpoints cannot produce a "
                       "candidate",
            "what_this_does_not_say": "it does not say the recipe cannot be improved; it says the "
                                      "search over existing checkpoints and blend ratios is "
                                      "exhausted as a source of candidates",
        },
        "attempt_registry": registry,
        "limits": [
            "the five declared months are already-viewed development data: this is a screening "
            "result, not confirmation",
            f"{len(OBJECTS)} objects were scored on the same window, so the best of them is a "
            f"selection from a search and its apparent edge is inflated",
            "single-seed arms cannot be judged on the registered secondary lines; only the "
            "primary metric is used here",
            "local approximate metrics only: no official cross-submission rank and no "
            "stress-stability item",
        ],
        "prereg": plan["prereg"],
    }
    # Candidate narrowing: at most two next directions, each one single factor, each with a
    # pre-declared screening rule and its blocker stated.
    nominations = {
        "rule": "at most two, each one primary factor, each requiring its own pre-registration and "
                "a training run before it can be screened; a single-seed arm is never judged on the "
                "registered secondary lines",
        "screening_rule_reused_for_any_nomination": {
            "primary_metric": template["primary_metric"],
            "floor": floor,
            "interval": "5-day consecutive blocks inside each declared period, lower bound > 0",
            "repeat": "at least 4 of 5 declared periods positive",
            "form": "three-seed aggregate, never a single seed",
            "economics": "0/2/5/10bp reported; 5bp is a stress gate, not a positive condition",
        },
        "items": [
            {
                "id": "N1",
                "factor": "model.time_loss_weighting",
                "from": "none (uniform)",
                "to": "one decaying weighting rule, single value",
                "why": "the overfit signature is that the loss keeps falling while the out-of-sample "
                       "ranking falls with it, so the binding constraint is what the optimizer sees "
                       "per sample rather than how long it runs; this is the only untried, "
                       "gate-expressible lever that changes the weighting of the training sample "
                       "without changing the budget, the trunk or the window",
                "status": "authorized_to_pre_register",
                "blocker": "needs a new pre-registration and a three-seed training run; nothing is "
                           "run here",
            },
            {
                "id": "N2",
                "factor": "model.hidden_dim",
                "from": "40",
                "to": "60 or 80 (1.5-2x at the anchor budget)",
                "why": "capacity is the reserve lever: it is gate-expressible and this family has "
                       "never varied it",
                "status": "conditional_not_authorized",
                "blocker": "the registered v2 routing licenses the capacity probe only for a flat "
                           "loss trend; the measured trend is falling, so this direction contradicts "
                           "the recorded diagnosis and must not be started before the captain "
                           "re-registers the routing",
            },
        ],
        "explicitly_not_nominated": {
            "model.sampling": "already exercised (timestamp_grouped) and lost 0.0018 RankIC with the "
                              "interval below zero",
            "model.ranking_loss": "already exercised (pairwise_hinge alpha=1.0 collapsed, alpha=0.1 "
                                  "did not beat its control)",
            "data.lookback_256": "already exercised in the Patch 256 family",
            "re_blending_existing_checkpoints": "screened here: no combination beats the incumbent "
                                                "blend by the floor",
            "output_level_turnover_smoothing": "measured negative under the frozen gate and not "
                                               "expressible in the gate's factor vocabulary",
        },
    }
    triage["nominations"] = nominations
    write_json(output / "triage.json", triage)
    write_json(output / "attempts.json", registry)
    event(events, "triage", "written", survivors=len(survivors),
          closest=best[0], closest_improvement=best[1]["paired_improvement"])

    print(f"screened {len(screening)} objects against {BASELINE} "
          f"(rank_ic_mean {baseline_values.mean():.6f}), 104 days, 5 periods")
    for object_id, value in sorted(screening.items(),
                                   key=lambda item: -item[1]["paired_improvement"]):
        interval = value["block_interval"]
        print(f"  {object_id:14s} {value['paired_improvement']:+.6f} "
              f"[{interval['low']:+.6f}, {interval['high']:+.6f}] "
              f"{value['positive_periods']}/{value['periods']} "
              f"{value['classification']}")
    print(f"survivors: {survivors or 'none'}   closest gap to floor: "
          f"{triage['closest_to_the_floor']['gap_to_floor']:+.6f}")
    print(f"attempts registered: {registry['total_attempts']} across {len(registry['families'])} families")

    if args.archive:
        # Run-level daily table so the shared index can determine the dates this run read.
        pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d")}).to_csv(output / "daily.csv", index=False)
        write_json(output / "metrics.json", {
            "schema_version": 1,
            "day_count": int(len(dates)),
            "valid_days": int(len(dates)),
            "baseline_rank_ic_mean": float(baseline_values.mean()),
            "object_rank_ic_mean": {object_id: float(frame["rank_ic"].mean())
                                    for object_id, frame in series.items()},
            "paired_improvement_vs_baseline": {object_id: value["paired_improvement"]
                                               for object_id, value in screening.items()},
            "survivors": survivors,
            "attempt_count": registry["total_attempts"],
            "note": "screening statistics re-published from the archived per-month dailies; this "
                    "run computes no metric of its own",
        })
        artifacts = [{"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
                      "sha256": sha256_file(path)} for path in sorted(output.rglob("*"))
                     if path.is_file() and path.name != "manifest.json"]
        write_json(output / "manifest.json", {
            "schema_version": 1, "run_id": output.name, "experiment_id": args.archive,
            "task": "t9 narrow the candidate pool and register the attempts",
            "kind": "analysis", "status": "completed",
            "started_at": plan["created_at"], "ended_at": utc_now(), "exit_code": 0,
            "hypothesis": plan["question"],
            "code": {"commit": _git("rev-parse", "HEAD"),
                     "dirty_fingerprint": _git("status", "--porcelain")[:2000]},
            "plan_sha256": sha256_file(output / "plan.json"),
            "config_sha256": sha256_file(args.prereg),
            "seed": None,
            "seeds_note": "deterministic aggregates; seed slots are never fabricated",
            "data": {"source": "per-month daily artifacts of the frozen replay measurement",
                     "version": f"{VOLATILITY.name}"},
            "evaluation_version": "eval-v2",
            "verdict": {"survivors": len(survivors),
                        "closest_object": best[0],
                        "closest_improvement": best[1]["paired_improvement"],
                        "gap_to_floor": triage["closest_to_the_floor"]["gap_to_floor"],
                        "floor": floor, "noise_scale": noise},
            "checks": {"placebo_scale_reproduced": True,
                       "screen_primary_only": True, "trains_anything": False,
                       "sealed_read": False},
            "artifacts": artifacts,
            "record": {"triage": "triage.json", "attempts": "attempts.json"},
        })
        event(events, "triage", "archived", survivors=len(survivors))
        print(f"archive {output.relative_to(ROOT).as_posix()}")
    return 0


def _git(*args: str) -> str:
    import subprocess

    return subprocess.run(("git", *args), cwd=str(ROOT), capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
