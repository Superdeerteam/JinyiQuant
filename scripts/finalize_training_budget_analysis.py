"""Assemble the training-amount analysis into a run archive and one conclusion document.

Reads what the analysis and the decision rule already produced and never recomputes a metric:

* ``tmp`` analysis artifacts: ``plan.json``, ``analysis.json``, the per-budget-point objects,
  and the ``decision.json`` written by ``scripts/budget_decision_prereg.py --apply``;
* the trajectory's own per-budget-point loss records (``budget_points/*.json``) and the
  per-epoch loss events (``events.jsonl``).

It writes one archive under ``runs/<experiment>/<run_id>/`` with a manifest and an event log,
so the reading is traceable in the shared experiment index.

The conclusion is reported on two clearly separated layers. The **registered verdict** is the
one the pre-registration licenses. The **descriptive reading** reports the paired numbers as
they are, and is labelled as outside the registered state space so it can never be mistaken
for a verdict.

Usage
    python scripts/finalize_training_budget_analysis.py \
        --analysis tmp/t5-run/analysis.json --decision tmp/t5-run/decision.json \
        --losses tmp/t5-run-losses.json --trajectory <training run dir> --experiment training-budget-analysis-20261027
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "configs/b4_budget_decision_prereg_20261027.json"
REGISTRATION = ROOT / "configs/b4_budget_decision_prereg_20261027.registration.json"
MAIN_PREREG = ROOT / "configs/b4_promotion_prereg_20261027.json"
PRIMARY_PAIR = ("double-2.64", "double-5.28")
FIRST_PAIR = ("anchor-1.32", "double-2.64")


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
    record = {"schema_version": 1, "timestamp_utc": utc_now(), "level": "INFO",
              "stage": stage, "event_type": event_type, "context": context,
              "run_id": path.parent.name}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def git_state() -> dict:
    def capture(*args: str) -> str:
        return subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True).stdout.strip()

    return {"commit": capture("git", "rev-parse", "HEAD"),
            "dirty_fingerprint": capture("git", "status", "--porcelain")[:2000]}


def pair_view(analysis: dict, left: str, right: str) -> dict:
    pair = next((item for item in analysis["pairs"]
                 if item["baseline"] == left and item["candidate"] == right), None)
    if pair is None:
        raise SystemExit(f"pair {left} -> {right} is missing from the analysis")
    effects = [value for value in (pair.get("period_effects") or []) if value is not None]
    interval = pair["block_interval"]
    return {
        "baseline": left,
        "candidate": right,
        "paired_improvement_rank_ic": pair["paired_improvement"],
        "block_interval": {"low": interval["low"], "high": interval["high"],
                           "block_days": interval["block_days"],
                           "bootstrap_samples": interval["bootstrap_samples"]},
        "interval_excludes_zero": not (interval["low"] <= 0 <= interval["high"]),
        "interval_entirely_below_zero": interval["high"] < 0,
        "period_effects": pair.get("period_effects"),
        "positive_periods": sum(1 for value in effects if value > 0),
        "periods": len(effects),
        "paired_day_count": pair["paired_day_count"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--losses", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--experiment", default="training-budget-analysis-20261027")
    args = parser.parse_args(argv)

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    losses = json.loads(args.losses.read_text(encoding="utf-8"))

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}_{sha256_file(args.analysis)[:12]}"
    run_dir = ROOT / "runs" / args.experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    events = run_dir / "events.jsonl"
    event(events, "run", "started", experiment_id=args.experiment, entry=__file__,
          analysis=args.analysis.as_posix())

    objects = analysis["objects"]
    levels = {label: objects[label]["metrics_0bp"]["rank_ic_mean"] for label in objects}
    primary = pair_view(analysis, *PRIMARY_PAIR)
    first = pair_view(analysis, *FIRST_PAIR)

    registered_verdict = {
        "state": decision["outcome"]["state"],
        "label": decision["outcome"]["state_label"],
        "why": decision["validation"]["why"],
        "table_cell": decision["outcome"]["table_cell"],
        "rule": {
            "prereg": decision["prereg"]["path"],
            "prereg_sha256": decision["prereg"]["sha256"],
            "registered_at": decision["prereg"]["registered_at"],
            "improvement_floor": decision["pair"]["improvement_floor"],
            "still_improving": "improvement >= floor AND interval low > 0 AND >= 4/5 periods positive",
            "saturated": "improvement <= 0 AND interval includes zero",
            "everything_else": "inconclusive",
        },
        "thresholds_modified": False,
        "no_threshold_relaxation": True,
    }

    # The registered rule classifies; the action question is decided separately and
    # cannot be left vague: "keep training" requires an interval lower bound above zero.
    answer = {
        "question": "是否还有其他预算点值得追加 / 是否继续增加训练量",
        "decision": "refuse_further_training_amount",
        "decision_label": "拒绝继续增加训练量",
        "why": (
            "the 'still improving' branch requires the 5-day block interval lower bound to clear "
            f"zero. The primary pair's interval is [{primary['block_interval']['low']:.6f}, "
            f"{primary['block_interval']['high']:.6f}], entirely below zero, which decisively "
            "excludes 'more training would be better'. Both other pairs are negative with "
            "intervals below zero as well"),
        "what_the_inconclusive_state_actually_covers": (
            "only the classification of the downgrade (saturated versus significant regression). "
            "It does not leave the action undetermined"),
        "what_would_have_to_hold_to_add_another_point": (
            "a paired improvement of at least the 0.0017 floor whose 5-day block interval lower "
            "bound is above zero, with at least 4 of 5 declared periods positive"),
        "registered_verdict_unchanged": True,
        "rule_bytes_edited": False,
    }

    metrics_0bp = {label: objects[label]["metrics_0bp"] for label in objects}
    trade_off = {
        "finding": (
            "the 2.64 point is not uniformly worse: IC_IR rises "
            f"{metrics_0bp['anchor-1.32']['ic_ir']:.3f} -> {metrics_0bp['double-2.64']['ic_ir']:.3f} "
            f"({metrics_0bp['double-2.64']['ic_ir'] / metrics_0bp['anchor-1.32']['ic_ir'] - 1:+.1%}) "
            f"while RankIC falls only {abs(first['paired_improvement_rank_ic']):.5f}"),
        "ic_ir_by_point": {label: metrics_0bp[label]["ic_ir"] for label in objects},
        "rank_ic_mean_0bp_by_point": {label: metrics_0bp[label]["rank_ic_mean"] for label in objects},
        "platform_weight": "the published composite weights IC_IR at 20%, so a stability gain is "
                           "not nothing even when the level drops",
        "reading": "between 1.32 and 2.64 the extra budget buys stability at the cost of level; "
                   "at 5.28 both collapse, so the trade is not monotone in the budget",
        "does_not_change_the_judgment": True,
        "why_not": "promotion is keyed to the primary metric rank_ic_mean against min_improvement "
                   "0.0034; every point here is negative on that metric and below the registered "
                   "baseline level, so a secondary stability gain cannot carry a promotion",
    }

    evidence_strength = {
        "strong": [
            "three budget points from one continuous trajectory, one recipe, only the optimizer "
            "budget differing",
            "identical 104-trading-day paired index across all three arms, five declared periods",
            "every exposure report reconciles against the measured pool (coverage 1.000000, steps "
            "19777/39553/79106 recomputed from the declared effective epochs)",
            "all three merged objects load unchanged in the pre-registered gate",
        ],
        "weak_or_caveated": [
            "the five declared months are already-viewed development data: they measure a budget "
            "effect and cannot confirm a candidate",
            "one trajectory, one seed: a single-seed arm cannot clear the registered ic_mean "
            "secondary line on noise alone, so nothing here is a promotion candidate",
            "the registered rule has no state for a significant regression, so its verdict is "
            "capped at 'inconclusive' even for a consistent downgrade",
            "local metrics only: no official cross-submission rank and no stress-stability item",
        ],
    }

    descriptive = {
        "what": "the paired numbers as measured, outside the registered state space: the rule has "
                "no state for a statistically clear regression, so it can only return "
                "'inconclusive' and this layer must never be quoted as a verdict",
        "primary_pair": primary,
        "first_pair": first,
        "rank_ic_mean_0bp_by_point": levels,
        "direction": ("both doublings move the primary metric down and every 5-day block interval "
                      "stays below zero; the primary pair loses "
                      f"{abs(primary['paired_improvement_rank_ic']):.6f} RankIC with "
                      f"{primary['positive_periods']}/{primary['periods']} positive periods"),
    }

    missing_evidence = {
        "why_the_verdict_is_inconclusive": (
            "the registered rule admits only 'still improving' (>= floor, interval above zero, "
            ">=4/5 periods positive) and 'saturated' (<= 0 AND interval covering zero). The "
            "observed reading is <= 0 with an interval entirely below zero, which matches "
            "neither branch, so the rule returns the third state by construction"),
        "not_a_data_gap": (
            "no measurement is missing: the arms, the 104 paired days, the five declared periods "
            "and both exposure reconciliations are all present. What is missing is a registered "
            "state for a significant regression"),
        "consequence_if_nothing_changes": (
            "without a newly registered state, every future reading of this shape is also capped "
            "at 'inconclusive', no matter how large or how consistent the regression is"),
    }

    probe = {
        "licensed_by": "configs/b4_budget_decision_prereg_20261027.json probe_routing",
        "route": decision["outcome"].get("route"),
        "started": False,
        "why_not_started": "a probe needs its own pre-registration and the captain's approval; "
                           "this analysis does not start one",
        "caution": ("the literal route is the one for 'inconclusive with a falling loss'. The "
                    "descriptive layer already shows a consistent regression across all five "
                    "periods, which is what the 2x2 table would read as the second column "
                    "('validation flat or falling'). The conflict is between the registered "
                    "three-state rule and the table's two-valued column, and resolving it "
                    "requires a new registration, not an edit of this one"),
    }

    economics = {
        label: objects[label]["cost_scenarios"] for label in objects
    }
    economic_reading = {
        "scenarios": economics,
        "primary_stress": "5bp",
        "note": "5bp is a locked local stress gate and is never a positive promotion condition; "
                "the official score contains no cost term. The local turnover proxy is direction "
                "evidence only and is never substituted for the official turnover item.",
        "net_5bp_by_point_bp": {label: economics[label]["5bp"]["net_return_bp"] for label in objects},
        "all_points_negative_at_5bp": all(economics[label]["5bp"]["net_return_bp"] < 0
                                          for label in objects),
    }

    conclusion = {
        "schema_version": 1,
        "created_at": utc_now(),
        "experiment_id": args.experiment,
        "question": analysis["question"],
        "single_factor": analysis["single_factor"],
        "coverage": analysis["coverage"],
        "registered_verdict": registered_verdict,
        "answer": answer,
        "metric_trade_off": trade_off,
        "evidence_strength": evidence_strength,
        "descriptive_reading": descriptive,
        "missing_evidence": missing_evidence,
        "probe": probe,
        "loss_trend": decision["loss_trend"],
        "loss_records": {
            "per_budget_point": losses["per_budget_point"],
            "epoch_series": losses["epoch_series"],
            "estimator": losses["estimator"],
            "source": losses["source"],
        },
        "economics": economic_reading,
        "exposure": analysis["exposure"],
        "gate_format": analysis["gate_format"],
        "promotion": ("nothing is promoted here. A budget point that cleared this screening rule "
                      "would still need the registered promotion gate: min_improvement 0.0034, an "
                      "interval excluding zero, 4/5 positive periods, no secondary or net5 "
                      "regression beyond the measured limits, resource compliance on the inference "
                      "path, and a multi-seed aggregate"),
        "limits": analysis["limits"],
        "inputs": {
            "analysis": {"path": args.analysis.as_posix(), "sha256": sha256_file(args.analysis)},
            "decision": {"path": args.decision.as_posix(), "sha256": sha256_file(args.decision)},
            "losses": {"path": args.losses.as_posix(), "sha256": sha256_file(args.losses)},
            "trajectory_run": args.trajectory.as_posix(),
            "prereg": {"path": decision["prereg"]["path"],
                       "sha256": decision["prereg"]["sha256"],
                       "registered_at": decision["prereg"]["registered_at"],
                       "registration": REGISTRATION.relative_to(ROOT).as_posix(),
                       "registration_sha256": sha256_file(REGISTRATION)},
            "main_prereg_sha256": sha256_file(MAIN_PREREG),
        },
    }
    write_json(run_dir / "conclusion.json", conclusion)
    event(events, "analysis", "conclusion_written",
          verdict=registered_verdict["state"], label=registered_verdict["label"],
          primary_improvement=primary["paired_improvement_rank_ic"],
          primary_interval=[primary["block_interval"]["low"], primary["block_interval"]["high"]],
          loss_trend=decision["loss_trend"]["trend"], probe=probe["route"]["first"] if probe["route"] else None)

    for name in ("plan.json", "analysis.json"):
        shutil.copy2(Path(args.analysis).parent / name, run_dir / name)
    shutil.copy2(args.decision, run_dir / "decision.json")
    shutil.copy2(args.losses, run_dir / "losses.json")
    shutil.copytree(Path(args.analysis).parent / "objects", run_dir / "objects")
    event(events, "analysis", "artifacts_copied", objects=sorted(os_.name for os_ in
                                                                 (run_dir / "objects").iterdir()))

    # Run-level daily table so the shared index can determine which dates this run read.
    # It re-publishes the per-object daily artifacts; nothing is recomputed here.
    wide = None
    for label in objects:
        frame = pd.read_csv(run_dir / "objects" / label / "daily_0bp.csv")[["date", "rank_ic"]]
        frame = frame.rename(columns={"rank_ic": f"rank_ic_{label}"})
        wide = frame if wide is None else wide.merge(frame, on="date", how="outer", validate="one_to_one")
    wide = wide.sort_values("date").reset_index(drop=True)
    wide.to_csv(run_dir / "daily.csv", index=False)
    write_json(run_dir / "metrics.json", {
        "schema_version": 1,
        "day_count": int(len(wide)),
        "valid_days": int(len(wide)),
        "objects": {label: {"metrics_0bp": objects[label]["metrics_0bp"],
                            "metrics_5bp": objects[label]["metrics_5bp"],
                            "cost_scenarios": objects[label]["cost_scenarios"]}
                    for label in objects},
        "note": "headline numbers re-published from the per-object eval-v2 archives; this run "
                "computes no metric of its own",
    })
    event(events, "analysis", "daily_and_metrics_written", days=int(len(wide)))

    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append({"path": path.relative_to(run_dir).as_posix(),
                              "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": args.experiment,
        "task": "t5 analyse the training-amount budget points",
        "kind": "analysis",
        "status": "completed",
        "started_at": analysis["created_at"],
        "ended_at": utc_now(),
        "exit_code": 0,
        "hypothesis": analysis["question"],
        "code": git_state(),
        "plan_sha256": sha256_file(run_dir / "plan.json"),
        "config_sha256": sha256_file(PREREG),
        "seed": None,
        "seeds_note": "deterministic aggregates: no seed replication, seed slots are never fabricated",
        "data": {"source": "frozen replay of one continuous trajectory on the five declared months",
                 "version": args.trajectory.name},
        "splits": {"evaluation_split": "the five declared research months",
                   "final": {"start": "2024-10-08", "end": "2025-01-01", "read": False}},
        "evaluation_version": analysis["evaluation_version"],
        "verdict": {"state": registered_verdict["state"], "label": registered_verdict["label"],
                    "rule": registered_verdict["rule"]["prereg"],
                    "registered_at": registered_verdict["rule"]["registered_at"],
                    "thresholds_modified": False},
        "metrics": {"rank_ic_mean_0bp": levels,
                    "primary_pair_improvement": primary["paired_improvement_rank_ic"],
                    "primary_pair_interval": [primary["block_interval"]["low"],
                                              primary["block_interval"]["high"]],
                    "net_5bp_bp": economic_reading["net_5bp_by_point_bp"]},
        "checks": {"segments": analysis["segments"],
                   "exposure_reconciled": all(segment["status"] != "missing"
                                              for segment in analysis["exposure"]["segments"].values()),
                   "gate_format_loadable": all(check["loadable"]
                                               for check in analysis["gate_format"].values())},
        "artifacts": artifacts,
        "resources": {"note": "analysis only: no training and no platform inference in this run"},
        "record": {"prereg": decision["prereg"]["path"], "decision": "decision.json",
                   "conclusion": "conclusion.json"},
    }
    write_json(run_dir / "manifest.json", manifest)
    event(events, "run", "ended", status="completed")

    print(f"archive {run_dir.relative_to(ROOT).as_posix()}")
    print(f"  conclusion { (run_dir / 'conclusion.json').relative_to(ROOT).as_posix()}")
    print(f"  verdict {registered_verdict['state']} / {registered_verdict['label']}")
    print(f"  primary pair {primary['baseline']} -> {primary['candidate']}: "
          f"{primary['paired_improvement_rank_ic']:.6f} "
          f"[{primary['block_interval']['low']:.6f}, {primary['block_interval']['high']:.6f}]")
    print(f"  loss trend {decision['loss_trend']['trend']} {decision['loss_trend']['relative_changes']}")
    print(f"  probe (not started) {probe['route']['first'] if probe['route'] else None}")
    print(f"  answer: {answer['decision_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
