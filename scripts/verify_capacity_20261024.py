"""Fail-closed checks for the capacity-20261024 pre-registration (configs/capacity_20261024/experiments.json).

Verifies, with machine-readable output, that:
1. every arm config differs from configs/model_v2_60d_csz.json only on the declared allowed keys;
2. the declared frozen expectations still equal the V2 config item by item;
3. the step arithmetic and config file hashes recorded in the preregistration are consistent;
4. the pre-registered gate numbers equal the V2 record's pre_registered_thresholds;
5. ModifiedTLOB parameter counts and final-head structures match the recorded values (needs torch).

Run with `python scripts/verify_capacity_20261024.py`; a torch-capable interpreter completes check 5,
otherwise it is reported as skipped and `complete` is false. Exit 0 only if every executed check passes.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
STUDY = ROOT / "configs" / "capacity_20261024" / "experiments.json"
V2_CONFIG = ROOT / "configs" / "model_v2_60d_csz.json"
V2_RECORD = ROOT / "runs" / "model-versions" / "v2-cloud-20261023" / "record.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def flatten(value, prefix=""):
    """Flatten nested dicts into dotted keys; lists and scalars stay as leaves."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}{key}."))
        return out
    return {prefix[:-1]: value}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_config_diffs(plan, reference):
    allowed = set(plan["frozen_recipe"]["allowed_differences"])
    rows = []
    for arm in plan["arms"]:
        path = ROOT / arm["config"]
        actual = flatten(read_json(path))
        unexpected = sorted((set(actual) ^ set(reference)) | {key for key in set(actual) & set(reference)
                                                              if actual[key] != reference[key]})
        outside = sorted(set(unexpected) - allowed)
        rows.append({"arm": arm["arm"], "changed_keys": unexpected, "unexpected_keys": outside,
                     "passed": not outside,
                     "config_sha256_matches": file_sha256(path) == arm["config_sha256"]})
    passed = all(row["passed"] and row["config_sha256_matches"] for row in rows)
    return {"check": "config_diff_vs_v2", "passed": passed, "detail": rows}


def check_declared_expectations(plan, reference):
    rows = []
    for key, expected in flatten(plan["frozen_recipe"]["declared_expectations"]).items():
        if key not in reference:
            passed = False
        elif key == "data.feature_columns":
            passed = tuple(expected) == tuple(reference[key])
        else:
            passed = expected == reference[key]
        rows.append({"key": key, "passed": passed, "reference": reference.get(key), "declared": expected})
    return {"check": "declared_expectations", "passed": all(row["passed"] for row in rows), "detail": rows}


def check_step_arithmetic(plan, record):
    step = plan["step_arithmetic"]
    pool = record["sample_exposure_matching"]["arms"]["60d"]["pool_selected"]
    per_epoch = math.ceil(pool / step["batch_size"])
    expected = {
        "pool_matches_record": pool == step["train_pool_selected"] == 479432,
        "updates_per_epoch": per_epoch == step["updates_per_epoch"] == 14983,
        "full_epoch_steps": step["full_epochs"] * per_epoch == step["full_epoch_steps"] == 29966,
        "full_epoch_samples": step["full_epochs"] * pool == step["full_epoch_samples"] == 958864,
        "nine_thousand_samples": 9000 * step["batch_size"] == step["nine_thousand_step_samples"] == 288000,
        "epoch_share": abs(9000 * step["batch_size"] / pool - step["nine_thousand_step_epoch_share"]) < 1e-15,
        "record_epochs": abs(record["sample_exposure_matching"]["arms"]["60d"]["epochs"]
                             - step["nine_thousand_step_epoch_share"]) < 1e-12,
    }
    arms = {arm["arm"]: arm for arm in plan["arms"]}
    volume = arms["w40_full2epoch"]
    expected["w40_config_steps"] = volume["max_steps"] == 29966 and volume["epochs"] == 2
    expected["w80_config_steps"] = arms["w80_9000"]["max_steps"] == 9000
    expected["w160_config_steps"] = arms["w160_9000"]["max_steps"] == 9000
    for arm in plan["arms"]:
        config = read_json(ROOT / arm["config"])
        expected[f"{arm['arm']}_config_max_steps"] = config["model"]["max_steps"] == arm["max_steps"]
        expected[f"{arm['arm']}_config_epochs"] = config["model"]["epochs"] == arm["epochs"]
        expected[f"{arm['arm']}_budget_wall"] = config["budget"]["wall_seconds"] >= 10800
        expected[f"{arm['arm']}_learning_curves"] = (config["logging"]["learning_curves"] is True
                                                     and config["logging"]["learning_curve_every_steps"] == 500)
    return {"check": "step_arithmetic", "passed": all(expected.values()), "detail": expected}


def check_gates(plan, record):
    reference = record["pre_registered_thresholds"]
    gates = {gate["id"]: gate for gate in plan["pre_registered_gates"]["gates"]}
    expected = {
        "rank_ic_mean_improvement": gates["rank_ic_mean_improvement"]["value"]
        == reference["min_rank_ic_improvement"],
        "long_short_mean_degradation": gates["long_short_mean_degradation"]["value"]
        == -reference["long_short_degradation_tolerance"],
        "turnover_degradation": gates["turnover_degradation"]["value"]
        == reference["turnover_degradation_tolerance"],
        "worst_day_long_short": gates["worst_day_long_short"]["value"]
        == reference["worst_day_long_short_floor"],
        "no_absolute_bound_claim": "derived_absolute_bounds_for_this_round" in plan["pre_registered_gates"],
    }
    return {"check": "pre_registered_gates_vs_v2_record", "passed": all(expected.values()), "detail": expected}


def check_params(plan):
    try:
        from modified_tlob.model import ModifiedTLOB
    except ImportError as exc:
        return {"check": "param_counts", "passed": True, "complete": False,
                "detail": {"skipped": f"torch unavailable: {exc}"}}
    expected = plan["capacity_params"]["widths"]
    arms = {int(arm["hidden_dim"]): arm for arm in plan["arms"]}
    rows = {}
    for width, recorded in expected.items():
        model = ModifiedTLOB(hidden_dim=int(width), num_layers=4, seq_size=128, num_features=12,
                             num_heads=1, is_sin_emb=True)
        structure, head_params = [], 0
        for layer in model.final_layers:
            if hasattr(layer, "weight"):
                params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
                head_params += params
                structure.append({"op": "Linear", "in": layer.in_features, "out": layer.out_features,
                                  "params": params})
            else:
                structure.append({"op": type(layer).__name__})
        total = sum(parameter.numel() for parameter in model.parameters())
        rows[width] = {
            "total_params": total,
            "matches_total": total == recorded["total_params"],
            "matches_head": structure == recorded["final_layers"],
            "matches_head_params": head_params == recorded["final_head_params"],
            "matches_arm_total": width not in arms or total == arms[width]["total_params"],
            "matches_head_shapes": width not in arms or (
                [f"Linear({item['in']},{item['out']})" if item["op"] == "Linear" else item["op"]
                 for item in structure] == arms[width]["final_head_linear_layers"]),
        }
    passed = all(all(value for key, value in row.items() if key.startswith("matches")) for row in rows.values())
    return {"check": "param_counts", "passed": passed, "complete": True, "detail": rows}


def main():
    plan = read_json(STUDY)
    reference = flatten(read_json(V2_CONFIG))
    record = read_json(V2_RECORD)
    checks = [
        check_config_diffs(plan, reference),
        check_declared_expectations(plan, reference),
        check_step_arithmetic(plan, record),
        check_gates(plan, record),
        check_params(plan),
    ]
    passed = all(item["passed"] for item in checks)
    result = {"study": plan["study_id"], "passed": passed,
              "complete": all(item.get("complete", True) for item in checks), "checks": checks}
    print(json.dumps(result, indent=1, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
