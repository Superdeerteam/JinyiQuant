"""Apply the pre-registered capacity-20261024 gates and record the decision.

The gates, thresholds, orientation and selection order are read from
``configs/capacity_20261024/experiments.json`` (frozen before training) and every
observed value is read from the replay comparison artifact
``modified_tlob/capacity_20261024.json``. Nothing here recomputes a metric: the script
only compares recorded numbers to recorded thresholds, so the decision can be
re-derived from the two inputs alone.

The decision is void if the comparison's evidence chain is not intact (weight hashes,
identical evaluation scope per window, agreement with the frozen replay report,
pre-registration match); that check runs before any gate is evaluated.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jinyiquant.io import read_json, utc_now, write_json  # noqa: E402

COMPARISON = "modified_tlob/capacity_20261024.json"
POOLED_POINTER = "#/paired_deltas/pooled[pair={pair},metric={metric}]"
WORST_POINTER = "#/pooled_metrics/{arm}/min_daily_long_short_return"
# Transcribed from decision_rule.selection ("within 0.002 (the V2 tie-break band)"), which was
# written before training; no threshold here is invented after seeing results.
TIE_BREAK_BAND = 0.002


def compare(value: float, op: str, threshold: float) -> bool:
    if op == "gte":
        return value >= threshold
    if op == "lte":
        return value <= threshold
    raise ValueError(f"unsupported gate operator {op}")


def pooled_row(comparison: dict, arm: str, reference: str, metric: str) -> dict:
    pair = f"{arm}-minus-{reference}"
    for row in comparison["paired_deltas"]["pooled"]:
        if row["pair"] == pair and row["metric"] == metric:
            return row
    raise KeyError(f"comparison artifact has no pooled row for {pair} / {metric}")


def bootstrap_gates(comparison: dict, arm: str, reference: str, bootstrap: dict) -> list[dict]:
    """The frozen rule: the 5-day block interval must exclude 0 on the favourable side."""
    results = []
    for metric in bootstrap["applies_to"]:
        row = pooled_row(comparison, arm, reference, metric)
        low, high = row["block5_ci_low"], row["block5_ci_high"]
        passed = bool(high < 0) if metric == "turnover" else bool(low > 0)
        results.append({
            "metric": metric, "days": row["days"], "ci_low": low, "ci_high": high,
            "excludes_zero": bool(low > 0 or high < 0),
            "required_side": "below_zero" if metric == "turnover" else "above_zero",
            "passed": passed,
            "evidence": COMPARISON + POOLED_POINTER.format(pair=f"{arm}-minus-{reference}", metric=metric)})
    return results


def numeric_gates(comparison: dict, arm: str, reference: str, gates: list[dict]) -> list[dict]:
    results = []
    for gate in gates:
        if gate["id"] == "worst_day_long_short":
            pooled = comparison["pooled_metrics"][arm]
            observed = pooled["min_daily_long_short_return"]
            evidence = COMPARISON + WORST_POINTER.format(arm=arm)
            extra = {"date": pooled["worst_day"], "window": pooled["worst_day_window"]}
        else:
            row = pooled_row(comparison, arm, reference, gate["metric"])
            observed = row["mean_delta"]
            evidence = COMPARISON + POOLED_POINTER.format(
                pair=f"{arm}-minus-{reference}", metric=gate["metric"]) + "/mean_delta"
            extra = {"ci_low": row["block5_ci_low"], "ci_high": row["block5_ci_high"], "days": row["days"]}
        results.append({
            "id": gate["id"], "metric": gate["metric"], "statistic": gate["statistic"],
            "op": gate["op"], "threshold": gate["value"], "observed": observed,
            "passed": compare(observed, gate["op"], gate["value"]), "evidence": evidence, **extra})
    return results


def evidence_chain(comparison: dict) -> dict:
    verification = comparison["verification"]
    weights_ok = all(item["match"] for item in verification["weights_sha256"])
    scope_ok = all(item["all_versions_identical"]
                   for item in verification["identical_evaluation_scope_per_window"])
    report_ok = bool(verification["report_agreement"]["match"])
    prereg_ok = bool(verification["preregistration"]["pairs_match"]
                     and verification["preregistration"]["windows_match"])
    return {
        "weights_sha256_match": weights_ok,
        "identical_evaluation_scope_per_window": scope_ok,
        "replay_report_agreement": report_ok,
        "preregistration_match": prereg_ok,
        "intact": weights_ok and scope_ok and report_ok and prereg_ok,
        "evidence": COMPARISON + "#/verification",
    }


def changed_factors(arm: dict, reference: dict) -> list[str]:
    """What the arm changes relative to V2: exposure, capacity, or both."""
    factors = []
    exposure_differs = arm.get("max_steps") != reference.get("steps")
    share = arm.get("expected_epoch_share")
    if share is not None and reference.get("epochs") is not None and share != reference["epochs"]:
        exposure_differs = True
    if exposure_differs:
        factors.append("training_epochs")
    if arm.get("hidden_dim") != reference.get("hidden_dim"):
        factors.append("capacity")
    return factors or ["none"]


def arm_decision(comparison: dict, prereg: dict, arm: dict, reference: dict) -> dict:
    name, reference_name = arm["arm"], reference["version"]
    numeric = numeric_gates(comparison, name, reference_name, prereg["pre_registered_gates"]["gates"])
    bootstrap = bootstrap_gates(comparison, name, reference_name,
                                prereg["pre_registered_gates"]["bootstrap_gate"])
    rank_ic_delta = pooled_row(comparison, name, reference_name, "rank_ic")["mean_delta"]
    long_short_delta = pooled_row(comparison, name, reference_name, "long_short_return")["mean_delta"]
    failed = [gate["id"] for gate in numeric if not gate["passed"]] + \
             [f"bootstrap_{gate['metric']}" for gate in bootstrap if not gate["passed"]]
    return {
        "arm": name, "group": arm["group"], "hidden_dim": arm.get("hidden_dim"),
        "max_steps": arm.get("max_steps"), "expected_samples_seen": arm.get("expected_samples_seen"),
        "expected_epoch_share": arm.get("expected_epoch_share"),
        "changed_vs_v2": changed_factors(arm, reference),
        "numeric_gates": numeric, "bootstrap_gates": bootstrap,
        "all_numeric_passed": all(gate["passed"] for gate in numeric),
        "all_bootstrap_passed": all(gate["passed"] for gate in bootstrap),
        "passed": all(gate["passed"] for gate in numeric) and all(gate["passed"] for gate in bootstrap),
        "failed_gates": failed,
        "exit_rule_applies": bool(rank_ic_delta <= 0 and long_short_delta <= 0),
        "exit_rule_reason": "pooled RankIC delta <= 0 and pooled long-short delta <= 0: a training-loss "
                            "movement is not evidence, the version exits the submission-candidate pool",
        "pooled_rank_ic_delta": rank_ic_delta, "pooled_long_short_delta": long_short_delta,
    }


def turnover_of(arm: dict) -> float:
    for gate in arm["numeric_gates"]:
        if gate["id"] == "turnover_degradation":
            return gate["observed"]
    raise KeyError("turnover gate missing")


def select_winner(arms: list[dict], band: float = TIE_BREAK_BAND) -> dict:
    passing = [arm for arm in arms if arm["passed"]]
    ranked = sorted(arms, key=lambda arm: arm["pooled_rank_ic_delta"], reverse=True)
    if not passing:
        return {"strongest_passing_version": None, "tie_break_applied": False,
                "ranked_passing": [],
                "closest_non_passing": ranked[0]["arm"],
                "reason": "no version passed every pre-registered gate, so the frozen selection order has "
                          "no candidate to rank"}
    passing.sort(key=lambda arm: arm["pooled_rank_ic_delta"], reverse=True)
    winner, tie_break = passing[0], False
    if len(passing) > 1 and abs(winner["pooled_rank_ic_delta"] - passing[1]["pooled_rank_ic_delta"]) <= band:
        tie_break = True
        winner = sorted(passing[:2], key=lambda arm: (arm["pooled_long_short_delta"],
                                                      -turnover_of(arm), -arm["hidden_dim"]),
                        reverse=True)[0]
    return {"strongest_passing_version": winner["arm"], "tie_break_applied": tie_break,
            "ranked_passing": [{"arm": arm["arm"], "pooled_rank_ic_delta": arm["pooled_rank_ic_delta"]}
                               for arm in passing],
            "reason": "highest pooled RankIC delta among versions passing every gate"}


def question(group_arms: list[dict], hypothesis: str) -> dict:
    passed = [arm for arm in group_arms if arm["passed"]]
    failed = {arm["arm"]: arm["failed_gates"] for arm in group_arms if not arm["passed"]}
    return {
        "arms": [arm["arm"] for arm in group_arms],
        "hypothesis": hypothesis,
        "answer": "yes" if passed else "no",
        "passing": [arm["arm"] for arm in passed],
        "failed_gates": failed,
        "reason": (f"passed every gate: {', '.join(arm['arm'] for arm in passed)}" if passed
                   else "no arm in this group reached the pre-registered RankIC improvement margin; "
                        "failed gates: " + "; ".join(f"{name}: {', '.join(gates)}"
                                                     for name, gates in failed.items())),
    }


def step3_decision(capacity_arms: list[dict], winner: dict) -> dict:
    """step3 is triggered only by a capacity winner, which then needs a 2-epoch confirmation."""
    capacity_winner = (winner["strongest_passing_version"]
                       if winner["strongest_passing_version"] in {arm["arm"] for arm in capacity_arms}
                       else None)
    return {
        "triggered": capacity_winner is not None,
        "trigger_condition": "a capacity-group version (w80_9000 or w160_9000) passes every pre-registered gate",
        "action_if_triggered": "retrain that width at the training-volume arm's 2-epoch exposure (29966 updates, "
                               "958864 samples) and repeat the identical three-window paired comparison before any "
                               "submission, so width and exposure are not confounded",
        "winner_if_triggered": capacity_winner,
        "reason": ("no capacity version passed, so no width result needs a 2-epoch confirmation run"
                   if capacity_winner is None else
                   "the capacity winner's gain must be separated from exposure before submission"),
    }


def one_line(questions: dict, winner: str | None, step3: dict) -> str:
    volume = "有效" if questions["training_volume"]["answer"] == "yes" else "无效"
    capacity = "有效" if questions["capacity"]["answer"] == "yes" else "无效"
    step3_text = "触发" if step3["triggered"] else "不触发"
    if winner is None:
        return (f"两个问题均为否：增加训练曝光（0.6→2 轮）{volume}、扩大容量（40→80/160）{capacity}，"
                f"三开发月均未达到预登记 RankIC 门槛；本分支无提交候选，名额交回标签分支，step3 {step3_text}。")
    return (f"问题1（训练曝光）{volume}、问题2（容量）{capacity}；最强通过版本 {winner}，step3 {step3_text}。")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=ROOT / COMPARISON)
    parser.add_argument("--preregistration", type=Path,
                        default=ROOT / "configs/capacity_20261024/experiments.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "modified_tlob/capacity_20261024_decision.json")
    args = parser.parse_args(argv)

    comparison = read_json(args.comparison)
    prereg = read_json(args.preregistration)
    chain = evidence_chain(comparison)
    if not chain["intact"]:
        raise ValueError(f"evidence chain is not intact, decision refused: {chain}")

    reference = {**prereg["reference"], "hidden_dim": prereg["reused_arm"]["hidden_dim"]}
    arms = [arm_decision(comparison, prereg, arm, reference) for arm in prereg["arms"]]
    volume_arms = [arm for arm in arms if arm["group"] == "training_volume"]
    capacity_arms = [arm for arm in arms if arm["group"] == "capacity"]
    questions = {
        "training_volume": question(volume_arms, prereg["hypotheses"]["training_volume"]),
        "capacity": question(capacity_arms, prereg["hypotheses"]["capacity"]),
    }

    winner = select_winner(arms)
    step3 = step3_decision(capacity_arms, winner)
    both_negative = questions["training_volume"]["answer"] == "no" and questions["capacity"]["answer"] == "no"
    handback = {
        "candidate_selected": winner["strongest_passing_version"] is not None,
        "submission_candidate": winner["strongest_passing_version"],
        "slot": "returns_to_label_branch" if winner["strongest_passing_version"] is None else "capacity_branch",
        "rule": prereg["decision_rule"]["handback"],
        "negative_result": ("both pre-registered questions are negative: extra training exposure and extra width at "
                            "the fixed 9000-update budget do not improve the three development months over V2"
                            if both_negative else "not both questions are negative; see questions"),
        "exited_arms": [arm["arm"] for arm in arms if arm["exit_rule_applies"]],
    }

    decision = {
        "schema_version": 1,
        "kind": "capacity_study_decision",
        "study_id": prereg["study_id"],
        "generated_utc": utc_now(),
        "decision_inputs": {
            "comparison": COMPARISON,
            "comparison_replay_run_id": comparison["replay"]["run_id"],
            "comparison_code_commit": comparison["replay"]["code_commit"],
            "comparison_plan_seed": comparison["plan"]["seed"],
            "preregistration": "configs/capacity_20261024/experiments.json",
            "gates_frozen": prereg["pre_registered_gates"]["frozen"],
            "gates_source": prereg["pre_registered_gates"]["source"],
            "selection_rule": prereg["decision_rule"]["selection"],
            "tie_break_band": TIE_BREAK_BAND,
            "no_post_hoc": prereg["decision_rule"]["no_post_hoc"],
            "no_checkpoint_selection": prereg["decision_rule"]["no_checkpoint_selection"],
        },
        "evidence_chain": chain,
        "reference": {"version": prereg["reference"]["version"], "steps": prereg["reference"]["steps"],
                      "epochs": prereg["reference"]["epochs"],
                      "weights_sha256": prereg["reference"]["weights_sha256"]},
        "change_registration": {arm["arm"]: {"changed_vs_v2": arm["changed_vs_v2"],
                                             "hidden_dim": arm["hidden_dim"],
                                             "max_steps": arm["max_steps"],
                                             "expected_samples_seen": arm["expected_samples_seen"],
                                             "expected_epoch_share": arm["expected_epoch_share"]}
                                for arm in arms},
        "arms": arms,
        "questions": questions,
        "selection": winner,
        "step3": step3,
        "handback": handback,
        "training_loss_note": {
            "matched_step_loss": comparison["curves"]["contrasts"]["matched_step_loss"],
            "continuity_max_abs_loss_difference":
                comparison["verification"]["training_continuity"]["max_abs_loss_difference"],
            "statement": "the two-epoch arm reproduces the reused V2 run's 100-step loss rows exactly and its extra "
                         "epochs do not raise development RankIC; per the frozen shared_guardrail a training-loss "
                         "movement is not evidence of improvement",
            "evidence": COMPARISON + "#/curves/contrasts/matched_step_loss",
        },
        "boundaries": [
            "single seed (11) per arm and the three development months are already-used development data, so this "
            "is a screening decision, not independent confirmation",
            "local approximate metric (gross forward label returns, no fees or slippage), not the official score",
            "the capacity contrast changes width and final-head depth together, so a width conclusion carries "
            "that caveat",
            "final split 2024-10-08..2025-01-01 and the locked windows stay unopened",
        ],
    }
    decision["one_line_conclusion"] = one_line(questions, winner["strongest_passing_version"], step3)
    write_json(args.out, decision)
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
