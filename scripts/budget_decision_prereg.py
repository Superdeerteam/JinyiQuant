"""Pre-register the undertrained-versus-saturated decision rule for the training-amount experiment.

The rule must exist before any budget-point replay result is read, so that "keep training",
"the sample is the limit" and "the budget is not the limit" cannot be chosen after the fact.
This module lands the pre-registration, stamps it with a timestamp and a hash, verifies that
it inherits the registered threshold basis without touching any promotion threshold, and can
later apply it mechanically to an analysis artifact.

Three-state by construction: a reading either clears the still-improving rule, or clears the
saturated rule, or is *inconclusive* and routes to a named probe. There is no default side.

Usage
    python scripts/budget_decision_prereg.py --check
    python scripts/budget_decision_prereg.py --register
    python scripts/budget_decision_prereg.py --apply tmp/t5-dryrun/analysis.json \
        [--losses tmp/losses.json] [--output tmp/decision.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.promotion_gate import (  # noqa: E402  (one source for json/hash helpers)
    canonical_digest,
    rel,
    sha256_file,
    utc_now,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "configs/b4_budget_decision_prereg_20261027.json"
REGISTRATION = ROOT / "configs/b4_budget_decision_prereg_20261027.registration.json"
MAIN_PREREG = ROOT / "configs/b4_promotion_prereg_20261027.json"
MAIN_REGISTRATION = ROOT / "configs/b4_promotion_prereg_20261027.registration.json"
GRID = 0.0001


class Refused(Exception):
    """The document or the reading cannot be used as declared."""


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def round_up(grid: float, value: float) -> float:
    return float(np.ceil(round(value / grid, 9)) * grid)


def verify_prereg(document: dict, main: dict) -> dict:
    """Structural and inheritance checks; every number is reproduced, not trusted."""
    required = ("schema_version", "registration_id", "primary_pair", "arms", "windows",
                "estimator", "thresholds", "verdicts", "decision_table", "probe_routing")
    missing = [key for key in required if key not in document]
    if missing:
        raise Refused(f"the pre-registration is missing {missing}")
    if document["schema_version"] != 1:
        raise Refused("unsupported schema_version")

    # 1) The improvement floor must be the half floor on the registered grid, nothing else.
    main_thresholds = main["derived_thresholds"]
    min_improvement = float(main_thresholds["min_improvement"])
    grid = float(main_thresholds["grids"]["primary"])
    if abs(grid - GRID) > 1e-15:
        raise Refused(f"the registered primary grid is {grid}, this document assumes {GRID}")
    expected = round_up(grid, min_improvement / 2)
    floor = document["thresholds"]["improvement_floor"]
    if abs(float(floor["value"]) - expected) > 1e-12:
        raise Refused(f"improvement floor {floor['value']} is not min_improvement/2 rounded up on "
                      f"the grid ({expected})")
    derived = floor["derivation"]
    if abs(float(derived["min_improvement"]) - min_improvement) > 1e-12:
        raise Refused("the floor derivation does not cite the registered min_improvement")
    if derived["source_sha256"] != sha256_file(MAIN_PREREG):
        raise Refused("the floor derivation cites bytes that are not the registered pre-registration")
    if not floor.get("not_a_promotion_threshold"):
        raise Refused("the floor must declare that it is not a promotion threshold")

    # 2) The windows and the repeat rule must be the registered ones.
    declared = [(str(start), str(end)) for start, end in document["windows"]["declared"]]
    registered = [(str(start), str(end)) for start, end in main["periods"]["declared"]]
    if declared != registered:
        raise Refused(f"declared periods {declared} differ from the registered {registered}")
    if document["windows"]["names"] != main["periods"]["names"]:
        raise Refused("period names differ from the registered pre-registration")
    if document["windows"].get("sealed_read"):
        raise Refused("this rule may not read the sealed interval")
    rules = main["evidence_rules"]
    repeat = document["estimator"]["period_repeat"]
    required_positive = int(np.ceil(float(rules["required_positive_fraction"]) * len(registered) - 1e-12))
    if int(repeat["required_positive"]) != required_positive or int(repeat["periods"]) != len(registered):
        raise Refused(f"the repeat rule must be {required_positive} of {len(registered)} as registered")
    interval = document["estimator"]["interval"]
    if int(interval["block_days"]) != int(main["object_policy_template"]["block_days"]):
        raise Refused("block_days differs from the registered object policy")
    if int(interval["bootstrap_samples"]) != int(main["object_policy_template"]["bootstrap_samples"]):
        raise Refused("bootstrap_samples differs from the registered object policy")
    if abs(float(interval["confidence"]) - float(main["object_policy_template"]["confidence"])) > 1e-12:
        raise Refused("confidence differs from the registered object policy")

    # 3) Three states at minimum, no default side, and the undeclared cell must not be invented.
    #    A later revision may add exactly one further determined state (a significant
    #    regression), which the registered v1 could not express.
    states = list(document["verdicts"]["states"])
    required_states = ["still_improving", "saturated", "inconclusive"]
    if not set(required_states) <= set(states) or states[-1] != "inconclusive":
        raise Refused(f"unexpected verdict states {states}")
    extra = [state for state in states if state not in required_states]
    if extra and extra != ["significantly_degraded"]:
        raise Refused(f"only one further state is admissible, got {extra}")
    mandatory = next((value for key, value in document["verdicts"].items()
                      if key.endswith("_state_is_mandatory") and value), None)
    if not mandatory:
        raise Refused(f"the {len(states)}-state requirement must be declared mandatory")
    if not document["verdicts"].get("prohibited_claims"):
        raise Refused("the prohibited claims must be listed")
    cells = {(cell["loss_trend"], cell["validation_verdict"]): cell
             for cell in document["decision_table"]["cells"]}
    if len(cells) != len(document["decision_table"]["cells"]):
        raise Refused("the decision table declares the same cell twice")
    axis = document["verdicts"].get("axis") or {}
    admissible = set(states) | set(axis.values())
    stray = [key for key in cells if key[1] not in admissible]
    if stray:
        raise Refused(f"decision cells reference undeclared validation values {stray}")
    if ("falling", "still_improving") not in cells and not (
            axis.get("still_improving") and ("falling", axis["still_improving"]) in cells):
        raise Refused("no cell covers 'loss falling and validation improving'")
    reserved = document["decision_table"].get("unregistered_combination") or {}
    if (reserved.get("loss_trend"), reserved.get("validation_verdict")) in cells:
        raise Refused("the undeclared combination must not also be a declared cell")
    if reserved.get("conclusion") != "inconclusive":
        raise Refused("an undeclared combination must resolve to inconclusive")

    # 4) Nothing in the registered gate may be touched.
    relation = document["relation_to_registered_gate"]
    if not relation.get("promotion_threshold_unchanged"):
        raise Refused("this document must declare the promotion thresholds unchanged")
    for metric, limit in main_thresholds["secondary_limits"].items():
        if metric not in ("net_return_mean_5bp",) and metric not in main["object_policy_template"]["max_metric_regressions"]:
            continue
        if abs(float(limit) - float(main["object_policy_template"]["max_metric_regressions"][metric])) > 1e-12:
            raise Refused(f"secondary limit {metric} is inconsistent with the registered policy")
    if abs(float(main_thresholds["net5_limit"]) - float(main["net5_limit"])) > 1e-12:
        raise Refused("net5 limit is inconsistent with the registered pre-registration")

    return {
        "min_improvement": min_improvement,
        "improvement_floor": expected,
        "periods": len(registered),
        "required_positive": required_positive,
        "block_days": int(interval["block_days"]),
        "bootstrap_samples": int(interval["bootstrap_samples"]),
        "confidence": float(interval["confidence"]),
        "secondary_lines_untouched": True,
        "sealed_read": False,
    }


def register(document: dict, checks: dict, prereg_path: Path | None = None) -> dict:
    """One-shot registration: stamp the bytes, never overwrite an existing record."""
    prereg_path = Path(prereg_path or PREREG)
    registration_path = registration_for(prereg_path)
    if registration_path.exists():
        raise Refused(f"{rel(registration_path)} already exists; the pre-registration is one-shot")
    record = {
        "schema_version": 1,
        "registration_id": document["registration_id"],
        "registered_at": utc_now(),
        "author_role": document["owner_role"],
        "prereg_path": rel(prereg_path),
        "prereg_sha256": sha256_file(prereg_path),
        "inherits": {
            "path": rel(MAIN_PREREG),
            "sha256": sha256_file(MAIN_PREREG),
            "registration_path": rel(MAIN_REGISTRATION),
            "registration_sha256": sha256_file(MAIN_REGISTRATION),
        },
        "checks": checks,
        "thresholds_modified": False,
        "promotion_threshold_untouched": True,
        "sealed_read": False,
        "candidate_state_at_registration": (
            "no budget-point replay metric was read before this registration; the author had not "
            "opened any analysis artifact, comparison artifact or budget-point object for this "
            "experiment, and the thresholds come only from the registered b4 basis and this "
            "document"),
        "one_shot": True,
    }
    record["canonical_digest"] = canonical_digest(record)
    write_json(record, registration_path)
    return record


def registration_for(prereg_path: Path) -> Path:
    """Every pre-registration carries its record beside it: <name>.registration.json."""
    prereg_path = Path(prereg_path)
    return prereg_path.with_suffix(".registration.json")


def verify_registration(document: dict, main: dict, prereg_path: Path | None = None) -> dict:
    prereg_path = Path(prereg_path or PREREG)
    registration_path = registration_for(prereg_path)
    if not registration_path.is_file():
        raise Refused(f"{rel(registration_path)} is missing; run --register before applying the rule")
    record = read_json(registration_path)
    if record["prereg_sha256"] != sha256_file(prereg_path):
        raise Refused("the registered bytes of the pre-registration have changed")
    if record["registered_at"].startswith("0000"):
        raise Refused("registration timestamp is not set")
    checks = verify_prereg(document, main)
    if record["checks"]["improvement_floor"] != checks["improvement_floor"]:
        raise Refused("the recorded floor disagrees with the document")
    stored = dict(record)
    digest = stored.pop("canonical_digest", None)
    if digest != canonical_digest(stored):
        raise Refused("the registration record digest does not reproduce")
    return record


def loss_trend(document: dict, losses: dict[str, float] | None) -> dict:
    """Classify the per-epoch mean training loss against the pre-declared epsilon."""
    spec = document["thresholds"]["loss_trend"]
    epsilon = float(spec["relative_epsilon"])
    order = [arm["label"] for arm in document["arms"]]
    if not losses:
        return {"trend": "undetermined", "why": "no training-loss record was supplied", "values": {}}
    missing = [label for label in order if label not in losses]
    if missing:
        return {"trend": "undetermined", "why": f"training loss missing for {missing}", "values": {}}
    values = [float(losses[label]) for label in order]
    if not all(np.isfinite(values)) or any(value <= 0 for value in values):
        return {"trend": "undetermined", "why": "training loss must be positive and finite",
                "values": dict(zip(order, values))}
    first = (values[1] - values[0]) / values[0]
    second = (values[2] - values[1]) / values[1]
    falling = first < -epsilon and second < -epsilon
    plateaued = abs(first) <= epsilon and abs(second) <= epsilon
    trend = "falling" if falling else ("plateaued" if plateaued else "undetermined")
    return {"trend": trend, "relative_changes": [float(first), float(second)],
            "epsilon": epsilon, "values": dict(zip(order, values))}


def verdict(document: dict, reading: dict) -> dict:
    """Map one paired reading to exactly one of the three states."""
    floor = float(document["thresholds"]["improvement_floor"]["value"])
    band = document["thresholds"]["saturation_band"]
    improvement = reading["paired_improvement"]
    low = reading["block_interval"]["low"]
    high = reading["block_interval"]["high"]
    effects = [value for value in (reading.get("period_effects") or []) if value is not None]
    positive = sum(1 for value in effects if value > 0)
    need = int(document["estimator"]["period_repeat"]["required_positive"])
    includes_zero = low is not None and high is not None and low <= 0 <= high
    if improvement is None or low is None or high is None:
        return {"state": "inconclusive", "why": "the paired reading is incomplete",
                "positive_periods": positive, "required_positive": need}
    if ("significantly_degraded" in (document["verdicts"]["states"] or [])
            and improvement <= float(band["improvement_at_or_below"]) and high < 0):
        return {"state": "significantly_degraded",
                "why": f"improvement {improvement:.6f} <= 0 and the interval [{low:.6f}, "
                       f"{high:.6f}] lies entirely below zero, which is a determined downgrade "
                       f"rather than an absence of evidence",
                "positive_periods": positive, "required_positive": need}
    if improvement >= floor and low > 0 and positive >= need:
        return {"state": "still_improving",
                "why": f"improvement {improvement:.6f} >= {floor}, interval low {low:.6f} > 0, "
                       f"{positive}/{need} required periods positive",
                "positive_periods": positive, "required_positive": need}
    if improvement <= float(band["improvement_at_or_below"]) and includes_zero and band[
            "interval_must_include_zero"]:
        return {"state": "saturated",
                "why": f"improvement {improvement:.6f} <= 0 and interval [{low:.6f}, {high:.6f}] "
                       f"covers zero", "positive_periods": positive, "required_positive": need}
    reasons = []
    if improvement < floor:
        reasons.append(f"improvement {improvement:.6f} < floor {floor}")
    if low <= 0:
        reasons.append(f"interval low {low:.6f} <= 0")
    if positive < need:
        reasons.append(f"{positive} positive periods < required {need}")
    if improvement <= 0 and not includes_zero:
        reasons.append("improvement <= 0 but the interval excludes zero, which matches neither rule")
    return {"state": "inconclusive", "why": "; ".join(reasons),
            "positive_periods": positive, "required_positive": need}


def decision(document: dict, state: dict, trend: dict) -> dict:
    """Apply the declared 2 x 2 table; an uncovered combination stays inconclusive."""
    cells = {(cell["loss_trend"], cell["validation_verdict"]): cell
             for cell in document["decision_table"]["cells"]}
    reserved = document["decision_table"]["unregistered_combination"]
    label = dict(document["verdicts"]["labels"])
    axis = document["verdicts"].get("axis") or {}
    axis_value = axis.get(state["state"], state["state"])
    determined = {value for _, value in cells}
    if trend["trend"] == "undetermined" or axis_value not in determined:
        return {"state": "inconclusive", "state_label": label["inconclusive"],
                "table_cell": None, "validation_axis": axis_value,
                "route": _route(document, state, trend),
                "why": f"loss trend {trend['trend']}, validation {state['state']} "
                       f"(axis {axis_value}): the table cannot be applied and neither side may "
                       f"be claimed"}
    cell = cells.get((trend["trend"], axis_value))
    if cell is None:
        return {"state": "inconclusive", "state_label": label["inconclusive"],
                "table_cell": reserved["id"], "route": None,
                "why": reserved["why"], "first_action": reserved["first_action"]}
    return {"state": "determined", "state_label": cell["conclusion_label"],
            "conclusion": cell["conclusion"], "table_cell": cell["id"],
            "validation_axis": axis_value,
            "next_step": cell["next_step"], "next_step_detail": cell["next_step_detail"],
            "route": None, "why": None}


def _route(document: dict, state: dict, trend: dict) -> dict | None:
    routes = document["probe_routing"]["routes"]
    if trend["trend"] == "undetermined":
        match = next((item for item in routes if item["when"] == "loss trend undetermined"), None)
        return match
    key = "falling" if trend["trend"] == "falling" else "plateaued"
    return next((item for item in routes if item["when"] == f"verdict inconclusive and loss trend {key}"),
                None)


def apply(document: dict, record: dict, analysis_path: Path, losses_path: Path | None,
          baseline: str | None, candidate: str | None, prereg_path: Path | None = None) -> dict:
    analysis = read_json(analysis_path)
    created = str(analysis.get("created_at") or "")
    if not created:
        raise Refused(f"{rel(analysis_path)} has no created_at; the ordering cannot be proven")
    if created < record["registered_at"]:
        raise Refused(f"the analysis artifact was created at {created}, before the registration at "
                      f"{record['registered_at']}; the rule was not registered in advance")
    coverage = analysis.get("coverage") or {}
    names = list(document["windows"]["names"])
    if list(coverage.get("periods") or []) != names:
        raise Refused(f"the analysis covers {coverage.get('periods')}, expected the declared {names}")
    if int(coverage.get("days") or 0) != int(document["windows"]["days_expected"]):
        raise Refused(f"the analysis covers {coverage.get('days')} days, expected "
                      f"{document['windows']['days_expected']}")
    primary = document["primary_pair"]
    left = baseline or primary["baseline"]
    right = candidate or primary["candidate"]
    pair = next((item for item in analysis.get("pairs") or []
                 if item["baseline"] == left and item["candidate"] == right), None)
    if pair is None:
        raise Refused(f"the analysis has no pair {left} -> {right}; available: "
                      f"{[(p['baseline'], p['candidate']) for p in analysis.get('pairs') or []]}")
    losses = read_json(losses_path) if losses_path else None
    trend = loss_trend(document, losses)
    state = verdict(document, pair)
    outcome = decision(document, state, trend)
    return {
        "schema_version": 1,
        "evaluated_at": utc_now(),
        "prereg": {"path": rel(Path(prereg_path or PREREG)), "sha256": record["prereg_sha256"],
                   "registered_at": record["registered_at"]},
        "analysis": {"path": rel(analysis_path), "created_at": created},
        "pair": {"baseline": left, "candidate": right,
                 "paired_improvement": pair["paired_improvement"],
                 "block_interval": pair["block_interval"],
                 "period_effects": pair.get("period_effects"),
                 "improvement_floor": document["thresholds"]["improvement_floor"]["value"]},
        "loss_trend": trend,
        "validation": state,
        "outcome": outcome,
        "no_threshold_relaxation": True,
        "promotion": "not decided here: promotion stays with scripts/promotion_gate.py and "
                     "min_improvement 0.0034",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prereg", type=Path, default=PREREG,
                        help="pre-registration document; its record is <name>.registration.json")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--apply", type=Path)
    parser.add_argument("--losses", type=Path)
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    prereg_path = args.prereg if args.prereg.is_absolute() else ROOT / args.prereg
    registration_path = registration_for(prereg_path)
    document = read_json(prereg_path)
    main = read_json(MAIN_PREREG)
    try:
        if args.register:
            record = register(document, verify_prereg(document, main), prereg_path)
            print(f"registered {record['registration_id']} at {record['registered_at']}")
            print(f"  prereg {rel(prereg_path)} sha256={record['prereg_sha256'][:16]}...")
            print(f"  floor={record['checks']['improvement_floor']} "
                  f"(min_improvement {record['checks']['min_improvement']}/2 on grid {GRID}), "
                  f"repeat {record['checks']['required_positive']}/{record['checks']['periods']}, "
                  f"blocks {record['checks']['block_days']}x{record['checks']['bootstrap_samples']}")
            print(f"  record {rel(registration_path)}")
            return 0
        checks = verify_prereg(document, main)
        record = verify_registration(document, main, prereg_path) if registration_path.is_file() else None
        if args.apply:
            if record is None:
                raise Refused("the rule is not registered yet; run --register before applying it")
            outcome = apply(document, record, args.apply, args.losses, args.baseline,
                            args.candidate, prereg_path)
            if args.output:
                write_json(outcome, args.output)
            print(f"registered_at={record['registered_at']} analysis_created_at="
                  f"{outcome['analysis']['created_at']} prereg_before_analysis="
                  f"{record['registered_at'] <= outcome['analysis']['created_at']}")
            print(f"  pair {outcome['pair']['baseline']} -> {outcome['pair']['candidate']}: "
                  f"improvement={outcome['pair']['paired_improvement']} "
                  f"interval=[{outcome['pair']['block_interval']['low']}, "
                  f"{outcome['pair']['block_interval']['high']}] "
                  f"floor={outcome['pair']['improvement_floor']}")
            print(f"  loss trend: {outcome['loss_trend']['trend']} "
                  f"({outcome['loss_trend'].get('why') or outcome['loss_trend'].get('relative_changes')})")
            print(f"  validation: {outcome['validation']['state']} ({outcome['validation']['why']})")
            print(f"  decision: {outcome['outcome']['state']} / {outcome['outcome']['state_label']}"
                  + (f" cell={outcome['outcome']['table_cell']}" if outcome['outcome']['table_cell'] else "")
                  + (f" next={outcome['outcome'].get('next_step')}" if outcome['outcome'].get("next_step") else ""))
            if outcome["outcome"].get("route"):
                route = outcome["outcome"]["route"]
                print(f"  probe: {route.get('first')} ({route.get('detail') or route.get('why')})")
            return 0
        print(f"prereg ok: floor={checks['improvement_floor']} (min_improvement "
              f"{checks['min_improvement']}/2), periods={checks['periods']}, repeat "
              f"{checks['required_positive']}/{checks['periods']}, blocks "
              f"{checks['block_days']}x{checks['bootstrap_samples']}, secondary lines untouched")
        if record is None:
            print("not registered yet: run --register to stamp the bytes and the timestamp")
        else:
            print(f"registered_at={record['registered_at']} prereg_sha256="
                  f"{record['prereg_sha256'][:16]}...")
        return 0
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
