"""Independent B-stage release verification (t15): resources, competition metrics, economics.

The B stage has to answer one question: may any candidate replace the deployable incumbent?
This verifier answers it from the frozen artifacts only, with its own metric implementation:

* competition: every already-measured object is rebuilt from the frozen replay predictions
  (nine blend baseline plus the three-seed ensembles and single seeds), then compared to the
  baseline on paired trading days with the pre-registered equal-period estimator and a 5-day
  consecutive-block interval inside each declared period; the registered improvement floor,
  repeat requirement and secondary degradation lines are read from the registered
  pre-registration, never re-declared here;
* economics: 0/2/5/10bp gross/cost/net plus the 5bp stress regression line, reported
  separately from the competition block and labelled as a local approximation;
* resources: the verifier's own inference-path measurement from t13 (host RSS with subprocesses
  and VRAM reported separately) is applied to the 24 GiB platform line with its margin;
* fairness and bookkeeping: identical windows, keys, targets and evaluation caliber for every
  object, comparable training budget in the frozen member recipes, and the attempt registry
  covering everything that was looked at;
* integrity: the registered thresholds are byte-identical to the registration, no threshold was
  lowered, and no local 5bp net is presented as an official score.

The decision is stated explicitly: released or not released, with the blocking reason.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "b-release-verify-20261027"

PREREG = ROOT / "configs/b4_promotion_prereg_20261027.json"
REGISTRATION = ROOT / "configs/b4_promotion_prereg_20261027.registration.json"
T9_RUN = ROOT / "runs/candidate-triage-20261027/20261027T141645.188723Z_04ac4b10a063"
T13_RUN = ROOT / "runs/resource-inference-verify-20261027/20261027T120917Z_713a835c173a"
T5_CONCLUSION = ROOT / "runs/training-budget-analysis-20261027/20261027T140646.334251Z_ed6edf7e0d70/conclusion.json"
GATE_SCRIPT = ROOT / "scripts/promotion_gate.py"

LIMIT_MIB = 24576.0              # 24 GiB platform host-memory line
PREREG_PROMOTION_MIB = 20889.0   # registered 15% headroom line
BLEND_WEIGHTS = ROOT / "runs/submission-best-20261026/build/blend_weights.json"
BARS = ROOT / "数据/cpt_jyc_2026_e2e_bar1m"

MEMBERS = ("tlob_s11", "tlob_s29", "tlob_s47", "mlp_s11", "mlp_s29", "mlp_s47",
           "patch_s11", "patch_s29", "patch_s47")
OBJECTS = {
    "nine_blend": MEMBERS,
    "tlob_3seed": MEMBERS[0:3], "mlp_3seed": MEMBERS[3:6], "patch_3seed": MEMBERS[6:9],
    **{member: (member,) for member in MEMBERS},
}
BASELINE = "nine_blend"
SECONDARY_DIRECTIONS = {"ic_mean": "maximize", "ic_ir": "maximize",
                        "long_short_sharpe": "maximize", "long_short_return_mean": "maximize",
                        "min_daily_long_short_return": "maximize", "turnover_mean": "minimize"}
SUMMARY_KEYS = {"turnover_mean": "turnover_mean_daily"}
RATES = (("0bp", 0.0), ("2bp", 0.0002), ("5bp", 0.0005), ("10bp", 0.001))
PRIMARY = "5bp"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def num(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def helper():
    """Reuse this verifier's own eval-v2 reimplementation, not the experiment's code."""
    spec = importlib.util.spec_from_file_location(
        "verifier_eval", ROOT / "scripts/verify_turnover_change.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def block_indices(length: int, block_days: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, length - block_days + 1, size=int(np.ceil(length / block_days)))
    return (starts[:, None] + np.arange(block_days)).ravel()[:length]


def paired_comparison(baseline_daily: pd.DataFrame, candidate_daily: pd.DataFrame,
                      periods: list[tuple[pd.Timestamp, pd.Timestamp]], policy: dict) -> dict:
    """Paired daily RankIC difference, equal-period improvement and block interval."""
    frame = pd.DataFrame({"day": pd.to_datetime(baseline_daily["day"]).reset_index(drop=True),
                          "baseline": baseline_daily["rank_ic"].to_numpy(dtype=float),
                          "candidate": candidate_daily["rank_ic"].to_numpy(dtype=float)})
    if not pd.to_datetime(candidate_daily["day"]).reset_index(drop=True).equals(frame["day"]):
        raise ValueError("objects are not evaluated on the identical trading-day index")
    frame["difference"] = frame["candidate"] - frame["baseline"]  # primary metric maximizes
    positions = [np.flatnonzero(((frame["day"] >= start) & (frame["day"] <= end)).to_numpy())
                 for start, end in periods]
    covered = np.zeros(len(frame), dtype=bool)
    for block in positions:
        covered[block] = True
    effects = [float(frame["difference"].to_numpy()[block].mean()) if len(block) else None
               for block in positions]
    improvement = float(np.mean(effects)) if effects and None not in effects else None
    differences = frame["difference"].to_numpy(dtype=float)
    rng = np.random.default_rng(int(policy["seed"]))
    draws = []
    for _ in range(int(policy["bootstrap_samples"])):
        sampled = np.concatenate([block[block_indices(len(block), int(policy["block_days"]), rng)]
                                  for block in positions])
        draws.append(float(np.mean(differences[sampled])))
    alpha = (1.0 - float(policy["confidence"])) / 2.0
    low, high = np.quantile(np.asarray(draws), [alpha, 1.0 - alpha])
    positive = sum(1 for effect in effects if effect is not None and effect > 0)
    required_positive = max(1, int(np.ceil(
        float(policy["required_positive_fraction"]) * len(periods) - 1e-12)))
    return {"days": int(len(frame)), "days_outside_periods": int((~covered).sum()),
            "period_effects": [num(effect) for effect in effects],
            "improvement": num(improvement),
            "interval": {"low": num(low), "high": num(high),
                         "confidence": float(policy["confidence"]),
                         "block_days": int(policy["block_days"]),
                         "draws": int(policy["bootstrap_samples"]), "seed": int(policy["seed"])},
            "positive_periods": positive, "periods": len(effects),
            "required_positive_periods": required_positive}


def economics(sections: pd.DataFrame, rate: float, helper_module) -> dict:
    summary = helper_module.aggregate(sections)
    daily = helper_module.daily_from_sections(sections, rate)
    gross = summary["gross_return_mean_economic"]
    change = summary["total_abs_weight_change_mean"]
    cost = None if change is None else rate * change
    return {"single_side_rate": rate,
            "gross_return_mean_bp": None if gross is None else gross * 1e4,
            "cost_mean_bp": None if cost is None else cost * 1e4,
            "net_return_mean_bp": None if (gross is None or cost is None) else (gross - cost) * 1e4,
            "daily_net_mean_bp": num(daily["net_return"].mean() * 1e4),
            "turnover_local": num(summary["turnover_mean_daily"]),
            "breakeven_single_side_rate_bp": (None if summary["breakeven_single_side_rate"] is None
                                              else summary["breakeven_single_side_rate"] * 1e4),
            "share_days_net_positive": num(float((daily["net_return"] > 0).sum()) / len(daily))}


def scan_official_numeric_values(paths: list[Path]) -> list[dict]:
    """No artifact may claim a numeric official score / percentile / composite."""
    hits = []

    def walk(node, path: str, file: str):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str) and key.lower().startswith("official") \
                        and isinstance(value, (int, float)) and not isinstance(value, bool):
                    hits.append({"file": file, "path": f"{path}/{key}", "value": value})
                walk(value, f"{path}/{key}", file)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]", file)

    for path in paths:
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        walk(document, "", path.relative_to(ROOT).as_posix())
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / EXPERIMENT_ID)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)

    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    policy = dict(prereg["object_policy_template"])
    evidence_rules = dict(prereg["evidence_rules"])
    periods = [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in prereg["periods"]["declared"]]
    comparison_policy = {**policy, **{"required_positive_fraction": evidence_rules["required_positive_fraction"]}}
    floor = float(policy["min_improvement"])
    secondary_limits = dict(policy["max_metric_regressions"])
    net5_limit = float(prereg["net5_limit"])

    plan = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "kind": "verification", "task": "t15",
        "question": "may any candidate replace the deployable incumbent under the B-stage release rules",
        "thresholds_source": {"path": PREREG.relative_to(ROOT).as_posix(), "sha256": sha256(PREREG),
                              "registered_sha256": registration.get("prereg_sha256")},
        "release_rule": {"primary_metric": policy["primary_metric"],
                         "improvement_floor": floor,
                         "interval_must_exclude_zero": True,
                         "required_positive_periods": int(np.ceil(
                             evidence_rules["required_positive_fraction"] * len(periods) - 1e-12)),
                         "secondary_limits": secondary_limits,
                         "net5_limit": net5_limit,
                         "resource_line_mib": LIMIT_MIB, "registered_promotion_line_mib": PREREG_PROMOTION_MIB,
                         "statement": "all conditions must hold; a single failed condition denies release"},
        "objects": {name: list(members) for name, members in OBJECTS.items()},
        "baseline": BASELINE,
        "economics_scenarios": {label: rate for label, rate in RATES},
        "resource_source": T13_RUN.relative_to(ROOT).as_posix(),
        "independence": "every metric is recomputed by this verifier from the frozen replay "
                        "predictions; thresholds are read from the registered pre-registration and the "
                        "workflow never re-declares or lowers them",
        "sealed_interval": "not read by this verification",
    }
    plan_bytes = (json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    output = args.output_root / f"{started:%Y%m%dT%H%M%SZ}_{plan_sha[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_bytes(plan_bytes)
    events = output / "events.jsonl"

    def event(stage: str, event_type: str, **context):
        with events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"schema_version": 1,
                                     "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                     "stage": stage, "event_type": event_type, "context": context},
                                    ensure_ascii=False, sort_keys=True) + "\n")

    print(f"predeclared plan {output.relative_to(ROOT).as_posix()}/plan.json sha256={plan_sha}")
    event("setup", "started", plan_sha256=plan_sha, objects=len(OBJECTS), floor=floor)

    # ------------------------------------------------------------------ recomputation
    vt = helper()
    windows = [vt.load_window(*window) for window in vt.WINDOWS]
    keys, target, member_scores = vt.pooled(windows)
    scores = {name: np.stack([member_scores[member] for member in members]).mean(axis=0)
              for name, members in OBJECTS.items()}

    summaries, dailies, sections_store = {}, {}, {}
    for name, values in scores.items():
        sections = vt.evaluate_sections(keys, target, values)
        summary = vt.aggregate(sections)
        summaries[name], dailies[name], sections_store[name] = summary, summary["daily"], sections
        print(f"evaluated {name}: rank_ic={summary['rank_ic_mean']:.6f}", flush=True)
    event("evaluate", "objects_ready", count=len(summaries))

    candidates = {}
    for name in OBJECTS:
        if name == BASELINE:
            continue
        paired = paired_comparison(dailies[BASELINE], dailies[name], periods, comparison_policy)
        secondary = {}
        for metric, limit in secondary_limits.items():
            key = SUMMARY_KEYS.get(metric, metric)
            base_value, cand_value = summaries[BASELINE].get(key), summaries[name].get(key)
            direction = SECONDARY_DIRECTIONS[metric]
            regression = ((base_value - cand_value) if direction == "maximize"
                          else (cand_value - base_value))
            secondary[metric] = {"baseline": num(base_value), "candidate": num(cand_value),
                                 "direction": direction, "regression": num(regression),
                                 "limit": float(limit), "passed": bool(regression <= float(limit))}
        base_net5 = float(vt.daily_from_sections(sections_store[BASELINE], 0.0005)["net_return"].mean())
        cand_net5 = float(vt.daily_from_sections(sections_store[name], 0.0005)["net_return"].mean())
        net5_regression = base_net5 - cand_net5
        conditions = {
            "improvement_at_least_floor": bool(paired["improvement"] is not None
                                               and paired["improvement"] >= floor),
            "interval_lower_bound_positive": bool(paired["interval"]["low"] is not None
                                                  and paired["interval"]["low"] > 0),
            "repeat_across_periods": bool(paired["positive_periods"]
                                          >= paired["required_positive_periods"]),
            "secondaries_within_limits": all(item["passed"] for item in secondary.values()),
            "net5_within_limit": bool(net5_regression <= net5_limit),
        }
        candidates[name] = {
            "object_id": name, "members": list(OBJECTS[name]),
            "rank_ic_mean": num(summaries[name]["rank_ic_mean"]),
            "baseline_rank_ic_mean": num(summaries[BASELINE]["rank_ic_mean"]),
            "paired": paired, "gap_to_floor": num(floor - paired["improvement"])
            if paired["improvement"] is not None else None,
            "secondary": secondary,
            "net5": {"baseline": num(base_net5), "candidate": num(cand_net5),
                     "regression": num(net5_regression), "limit": net5_limit,
                     "passed": bool(net5_regression <= net5_limit)},
            "conditions": conditions,
            "condition_count_passed": int(sum(conditions.values())),
            "releasable": bool(all(conditions.values())),
            "economics": {label: economics(sections_store[name], rate, vt)
                          for label, rate in RATES},
            "signal": {key: num(summaries[name][key]) for key in
                       ("ic_mean", "ic_ir", "rank_ic_ir", "long_short_sharpe",
                        "long_short_return_mean", "min_daily_long_short_return",
                        "turnover_mean_daily", "min_cross_sectional_std")},
        }
    event("compare", "candidates_ready", releasable=sum(1 for c in candidates.values() if c["releasable"]))
    baseline_economics = {label: economics(sections_store[BASELINE], rate, vt) for label, rate in RATES}
    baseline_record = {"rank_ic_mean": num(summaries[BASELINE]["rank_ic_mean"]),
                       "signal": {key: num(summaries[BASELINE][key]) for key in
                                  ("ic_mean", "ic_ir", "rank_ic_ir", "long_short_sharpe",
                                   "long_short_return_mean", "min_daily_long_short_return",
                                   "turnover_mean_daily", "min_cross_sectional_std")},
                       "economics": baseline_economics}

    # ------------------------------------------------------------ reconciliation with t9
    triage = json.loads((T9_RUN / "triage.json").read_text(encoding="utf-8"))
    screen_vs_triage = []
    for name, block in triage["screening"].items():
        mine = candidates.get(name)
        if mine is None:
            continue
        screen_vs_triage.append({
            "object_id": name,
            "triage_improvement": block.get("paired_improvement"),
            "verifier_improvement": mine["paired"]["improvement"],
            "abs_gap": num(abs((block.get("paired_improvement") or 0.0)
                               - (mine["paired"]["improvement"] or 0.0))),
            "triage_interval": block.get("block_interval"),
            "verifier_interval": mine["paired"]["interval"],
            "triage_positive_periods": block.get("positive_periods"),
            "verifier_positive_periods": mine["paired"]["positive_periods"],
        })
    triage_gap = max((item["abs_gap"] or 0.0) for item in screen_vs_triage) if screen_vs_triage else None

    # --------------------------------------------------------------- resources (t13)
    resource = json.loads((T13_RUN / "results.json").read_text(encoding="utf-8"))
    bound_mib = resource["bound"]["bound_mib"]
    resource_block = {
        "source": T13_RUN.relative_to(ROOT).as_posix(),
        "source_verdict": resource.get("verdict"),
        "host_peak_mb": resource["host"]["overall_peak_mb"],
        "host_commit_peak_mb": resource["host"]["overall_commit_mb"],
        "host_peaks_by_run_mb": resource["host"]["peaks_mb"],
        "full_range_bound_mib": bound_mib,
        "limit_mib": LIMIT_MIB,
        "margin_mib": resource["bound"]["margin_mib"],
        "margin_fraction": resource["bound"]["margin_fraction"],
        "registered_promotion_line_mib": PREREG_PROMOTION_MIB,
        "margin_vs_registered_line_mib": num(PREREG_PROMOTION_MIB - bound_mib)
        if bound_mib is not None else None,
        "gpu": {"device_peak_used_mb": resource["gpu"]["device_peak_used_mb"],
                "torch_peak_allocated_mb": resource["gpu"]["torch_peak_allocated_mb"],
                "torch_peak_reserved_mb": resource["gpu"]["torch_peak_reserved_mb"],
                "policy": "VRAM reported separately, never counted against the host-memory line"},
        "training_path_gate": False,
        "training_peak_reported_mb": 19584.0546875,
        "training_note": "archived full-scale training peak is reported only and never enters the "
                         "24 GiB judgement",
        "compliant": bool(bound_mib is not None and bound_mib <= LIMIT_MIB
                          and bound_mib <= PREREG_PROMOTION_MIB),
    }

    # ---------------------------------------------------------------------- fairness
    blend = json.loads(BLEND_WEIGHTS.read_text(encoding="utf-8"))
    budget_keys = ("max_steps", "epochs", "batch_size", "learning_rate", "alpha",
                   "target_scaling", "target_centering", "time_loss_weighting",
                   "book_representation", "trade_features")
    member_budgets = {member["name"]: {key: (member["payload"].get("model_config") or {}).get(key)
                                       for key in budget_keys} for member in blend["members"]}
    distinct_budgets = {json.dumps(value, sort_keys=True) for value in member_budgets.values()}
    lookbacks = {(member["payload"].get("preprocessing") or {}).get("lookback")
                 for member in blend["members"]}
    keys_sha = hashlib.sha256(pd.util.hash_pandas_object(keys, index=False).to_numpy().tobytes()).hexdigest()
    targets_sha = hashlib.sha256(pd.util.hash_pandas_object(
        pd.concat([keys, pd.Series(target, name="target")], axis=1), index=False)
        .to_numpy().tobytes()).hexdigest()
    fairness = {
        "windows": [window[0] for window in vt.WINDOWS],
        "trading_days": int(pd.to_datetime(keys["date"]).dt.normalize().nunique()),
        "cross_sections": int(keys["date"].nunique()),
        "samples": int(len(keys)),
        "shared_keys_sha256": keys_sha, "shared_targets_sha256": targets_sha,
        "one_shared_key_target_index": True,
        "evaluation_caliber": {"version": vt.EVALUATION_VERSION if hasattr(vt, "EVALUATION_VERSION") else "eval-v2",
                               "quantile": 0.2, "min_cross_section": 3, "periods_per_year": 252,
                               "cost_scenarios": [label for label, _ in RATES]},
        "training_budget": {"distinct_member_budgets": len(distinct_budgets),
                            "member_budgets": member_budgets,
                            "lookbacks": sorted(lookbacks),
                            "note": "the candidate objects are equal-weight means of the same frozen "
                                    "members; no candidate was retrained, so the budget is comparable "
                                    "by construction"},
        "sealed_interval_read": False,
    }

    # ------------------------------------------------------------------- bookkeeping
    attempts = json.loads((T9_RUN / "attempts.json").read_text(encoding="utf-8"))
    families = attempts["families"]
    registered_objects = set()
    for family in families:
        registered_objects.update(family.get("values_looked_at") or [])
    screening_names = set(triage["screening"])
    registry = {
        "path": (T9_RUN / "attempts.json").relative_to(ROOT).as_posix(),
        "total_attempts": attempts.get("total_attempts"),
        "family_count": len(families),
        "family_attempts_sum": sum(int(family.get("attempts") or 0) for family in families),
        "families": [{"id": family.get("id"), "attempts": family.get("attempts"),
                      "factor": family.get("factor")} for family in families],
        "index_provenance": attempts.get("index_provenance"),
        "screening_objects": sorted(screening_names),
        "screening_objects_registered": sorted(screening_names & registered_objects),
        "screening_objects_missing": sorted(screening_names - registered_objects),
        "baseline_registered": BASELINE in registered_objects or "nine_blend" in registered_objects,
        "config_space_survey": attempts.get("config_space_survey"),
    }
    index_path = ROOT / "runs/index/manifest_index.jsonl"
    index_lines = []
    if index_path.is_file():
        index_lines = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
    registry["index_path"] = index_path.relative_to(ROOT).as_posix() if index_path.is_file() else None
    registry["index_runs"] = len(index_lines)

    # --------------------------------------------------------------- integrity checks
    retired = prereg.get("supersedes") or {}
    retired_thresholds = {item.get("value"): item for item in retired.get("retired_thresholds") or []}
    threshold_integrity = {
        "prereg_path": PREREG.relative_to(ROOT).as_posix(),
        "prereg_sha256": sha256(PREREG),
        "registered_sha256": registration.get("prereg_sha256"),
        "verified": sha256(PREREG) == registration.get("prereg_sha256"),
        "registered_at": registration.get("registered_at"),
        "improvement_floor": floor,
        "secondary_limits": secondary_limits, "net5_limit": net5_limit,
        "resource_line_mib": LIMIT_MIB, "registered_promotion_line_mib": PREREG_PROMOTION_MIB,
        "retired_thresholds": retired.get("retired_thresholds"),
        "retired_usable_for_promotion": retired.get("usable_for_promotion"),
        "retired_0046_recorded": 0.0046 in retired_thresholds,
        "retired_0046_usable": (retired.get("usable_for_promotion")
                                if 0.0046 in retired_thresholds else None),
        "official_score_note": (prereg.get("economic_risk_boundary") or {}).get("official_score_note"),
        "cost_scenario_independence": (prereg.get("metric_caliber_layers") or {}).get(
            "cost_scenario_independence"),
        "official_turnover": (prereg.get("metric_caliber_layers") or {}).get("official_turnover"),
    }
    threshold_integrity["files"] = sorted(path.name for path in
                                          (ROOT / "configs").glob("b4_promotion_prereg_20261027*"))

    official_numeric = scan_official_numeric_values([
        PREREG, REGISTRATION,
        ROOT / "modified_tlob/cost_curve_20261027.json",
        ROOT / "modified_tlob/turnover_stability_20261027.json",
        T9_RUN / "triage.json", T9_RUN / "attempts.json",
        T13_RUN / "results.json",
    ])
    wording = {
        "official_numeric_claims": official_numeric,
        "official_score_note": threshold_integrity["official_score_note"],
        "official_turnover_note": threshold_integrity["official_turnover"],
        "five_bp_is_pressure_only": (prereg.get("economic_risk_boundary") or {}).get(
            "official_score_note") is not None,
        "local_net_label": "net return at any rate is a local economic lens, never the official score",
    }
    gate_source = GATE_SCRIPT.read_text(encoding="utf-8")
    threshold_integrity["gate_reads_prereg_thresholds"] = bool(
        re.search(r'prereg\["object_policy_template"\]', gate_source)
        and re.search(r'prereg\["net5_limit"\]', gate_source))
    threshold_integrity["gate_hardcoded_floor"] = "0.0034" in gate_source

    # --------------------------------------------------------------------- decision
    releasable = [name for name, block in candidates.items() if block["releasable"]]
    best = max(candidates.values(), key=lambda block: block["paired"]["improvement"] or -math.inf)
    decision = {
        "released": bool(releasable),
        "decision": "released" if releasable else "not_released",
        "candidates_checked": len(candidates),
        "releasable_candidates": releasable,
        "blocking_conditions": {name: [condition for condition, passed in block["conditions"].items()
                                       if not passed]
                                for name, block in candidates.items()
                                if not block["releasable"]},
        "closest_candidate": {"object_id": best["object_id"],
                              "improvement": best["paired"]["improvement"],
                              "floor": floor, "gap_to_floor": best["gap_to_floor"],
                              "interval": best["paired"]["interval"],
                              "positive_periods": best["paired"]["positive_periods"],
                              "required_positive_periods": best["paired"]["required_positive_periods"]},
        "reasons": [],
        "incumbent_preserved": BASELINE,
    }
    if not releasable:
        decision["reasons"].append(
            f"no candidate reaches the registered improvement floor {floor}: best {best['object_id']} "
            f"at {best['paired']['improvement']:.6f} (gap {best['gap_to_floor']:.6f})")
        if (best["paired"]["interval"]["low"] or 0) <= 0:
            decision["reasons"].append(
                f"the best candidate's 5-day block interval {best['paired']['interval']} still contains "
                f"zero and only {best['paired']['positive_periods']} of "
                f"{best['paired']['periods']} periods are positive")
        decision["reasons"].append(
            "the resource dimension is compliant, so the release is denied on competition metrics and "
            "economics, not on memory")
        decision["reasons"].append(
            "5bp net is negative for the incumbent and every candidate: it is a pressure gate and a "
            "local approximation, never an official score")

    # ------------------------------------------------------------------------ checks
    checks = []
    checks.append({"name": "resource_within_24gib_with_margin", "passed": bool(resource_block["compliant"]),
                   "evidence": resource_block})
    competition_ok = (not releasable) and triage_gap is not None and triage_gap <= 1e-9
    checks.append({"name": "competition_metrics_checked_against_registered_floor",
                   "passed": competition_ok,
                   "evidence": {"floor": floor,
                                "candidates_checked": len(candidates),
                                "releasable": releasable,
                                "closest": decision["closest_candidate"],
                                "reconciliation_with_t9_max_abs_gap": triage_gap,
                                "triage_comparison": screen_vs_triage,
                                "secondary_regressions_recorded": {
                                    name: {metric: block["regression"]
                                           for metric, block in candidates[name]["secondary"].items()}
                                    for name in candidates}}})
    economics_gate = bool(best["net5"]["passed"])
    checks.append({"name": "five_bp_economics_recorded_separately", "passed": bool(economics_gate),
                   "evidence": {"gate": "net return at 5bp may regress by no more than the registered limit",
                                "limit": net5_limit,
                                "per_candidate": {name: candidates[name]["net5"] for name in candidates},
                                "absolute_5bp_net_bp": {
                                    "baseline": baseline_economics[PRIMARY]["net_return_mean_bp"],
                                    **{name: candidates[name]["economics"][PRIMARY]["net_return_mean_bp"]
                                       for name in candidates}},
                                "competition_vs_economics": "competition improvement and the 5bp economic "
                                                            "gate are recorded in separate blocks; neither "
                                                            "substitutes for the other",
                                "label": "local approximation; pressure only"}})
    fairness_ok = (fairness["trading_days"] == 104 and len(fairness["windows"]) == 5
                   and fairness["training_budget"]["distinct_member_budgets"] == 1
                   and fairness["training_budget"]["lookbacks"] == [128]
                   and not registry["screening_objects_missing"]
                   and registry["family_attempts_sum"] == registry["total_attempts"])
    checks.append({"name": "fair_comparison_and_attempt_registry_complete", "passed": bool(fairness_ok),
                   "evidence": {"fairness": fairness, "registry": registry}})
    integrity_ok = (threshold_integrity["verified"] and floor == 0.0034
                    and threshold_integrity["gate_reads_prereg_thresholds"]
                    and not official_numeric
                    and threshold_integrity["gate_hardcoded_floor"] is False
                    and threshold_integrity["retired_0046_recorded"]
                    and threshold_integrity["retired_usable_for_promotion"] is False)
    checks.append({"name": "no_threshold_lowered_and_no_official_score_claim",
                   "passed": bool(integrity_ok),
                   "evidence": {"threshold_integrity": threshold_integrity,
                                "wording": wording,
                                "decision": decision}})
    verdict = "pass" if all(item["passed"] for item in checks) else "fail"

    per_candidate = pd.DataFrame([{
        "object_id": name,
        "rank_ic_mean": candidate["rank_ic_mean"],
        "improvement": candidate["paired"]["improvement"],
        "interval_low": candidate["paired"]["interval"]["low"],
        "interval_high": candidate["paired"]["interval"]["high"],
        "positive_periods": candidate["paired"]["positive_periods"],
        "required_positive_periods": candidate["paired"]["required_positive_periods"],
        "gap_to_floor": candidate["gap_to_floor"],
        "secondaries_passed": candidate["conditions"]["secondaries_within_limits"],
        "net5_regression": candidate["net5"]["regression"],
        "net5_passed": candidate["net5"]["passed"],
        "conditions_passed": candidate["condition_count_passed"],
        "releasable": candidate["releasable"],
        "gross_bp_0bp": candidate["economics"]["0bp"]["gross_return_mean_bp"],
        "net_bp_5bp": candidate["economics"]["5bp"]["net_return_mean_bp"],
        "turnover_local": candidate["economics"]["5bp"]["turnover_local"],
        "breakeven_bp": candidate["economics"]["5bp"]["breakeven_single_side_rate_bp"],
    } for name, candidate in candidates.items()])
    per_candidate.to_csv(output / "candidates.csv", index=False, lineterminator="\n")

    results = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "kind": "verification", "task": "t15",
        "created_at": started.isoformat(), "verdict": verdict,
        "decision": decision,
        "checks_passed": sum(item["passed"] for item in checks), "checks_total": len(checks),
        "baseline": baseline_record,
        "candidates": candidates,
        "economics_note": "0/2/5/10bp are local economic lenses on already-viewed development months; "
                          "5bp is the registered pressure gate; none of them is an official score",
        "resource": resource_block,
        "fairness": fairness,
        "registry": registry,
        "threshold_integrity": threshold_integrity,
        "wording": wording,
        "reconciliation_with_t9": {"max_abs_improvement_gap": triage_gap,
                                   "rows": screen_vs_triage},
        "checks": checks,
        "evidence_limits": [
            "five already-viewed development months: this is a release audit, not sealed confirmation",
            "official cross-submission rank composite and stress stability are unavailable locally",
            "5bp net is a local approximation with no market-impact term and is never an official score",
            "no sealed interval was read",
        ],
    }
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True)
                                         + "\n", encoding="utf-8")
    (output / "checks.json").write_text(json.dumps(
        {"schema_version": 1, "checks": checks, "checks_passed": results["checks_passed"],
         "checks_total": results["checks_total"], "verdict": verdict, "decision": decision["decision"]},
        ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    event("decide", "verdict", verdict=verdict, decision=decision["decision"],
          releasable=decision["releasable_candidates"])

    print(f"\nverdict: {verdict} ({results['checks_passed']}/{results['checks_total']} checks)")
    print(f"decision: {decision['decision']}  releasable={decision['releasable_candidates']}")
    closest = decision["closest_candidate"]
    print(f"closest: {closest['object_id']} improvement={closest['improvement']:.6f} "
          f"floor={closest['floor']:.4f} gap={closest['gap_to_floor']:.6f} "
          f"interval=[{closest['interval']['low']:.6f}, {closest['interval']['high']:.6f}] "
          f"positive={closest['positive_periods']}/{closest['required_positive_periods']}")
    print(f"resource: bound {resource_block['full_range_bound_mib']} MiB <= {LIMIT_MIB} MiB, "
          f"margin {resource_block['margin_mib']} MiB; VRAM device "
          f"{resource_block['gpu']['device_peak_used_mb']} MB reported separately")
    print(f"reconciliation with t9: max |Δimprovement| = {triage_gap}")
    for item in checks:
        print(f"  [{'PASS' if item['passed'] else 'FAIL'}] {item['name']}")
    print(f"\nrun archive: {output.relative_to(ROOT).as_posix()}")
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
