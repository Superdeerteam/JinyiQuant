"""Generate, register and check the feature-normalisation experiment on the 132-day rung.

Two arms share one recipe except ``data.feature_transform``: ``none`` (control) versus
``price_rel_vol_log`` (candidate). The candidate turns every price channel into
``(p - ref) / ref`` with ``ref`` the mid price at the window's own last step and every volume
channel into ``log1p(max(v, 0))`` (``src/jinyiquant/data.py::relative_price_log_volume``). Three
seeds, two budget points, both arms on the same host.

Chain without circular hashes: recipes -> prereg (pins recipe sha256) -> plans (reference the
prereg sha256) -> registration (pins the prereg sha256 and every plan sha256).

Usage:
    python scripts/feature_norm_prereg.py --generate
    python scripts/feature_norm_prereg.py --register
    python scripts/feature_norm_prereg.py --check
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

EXPERIMENT = "featnorm-132d-20261028"
PREREG = CONFIGS / f"b4_feature_norm_prereg_20261028.json"
REGISTRATION = CONFIGS / f"b4_feature_norm_prereg_20261028.registration.json"
TRANSFORM = "price_rel_vol_log"

SEEDS = (11, 29, 47)
# Per-seed control recipe (frozen 132-day data-probe recipe) and the candidate recipe written
# from it. Seed 11 is the primary; 29 and 47 are support seeds.
CONTROL = {
    11: "configs/mlplob_v3_132d_csz_19800.json",
    29: "configs/mlplob_v3_132d_csz_19800_seed29.json",
    47: "configs/mlplob_v3_132d_csz_19800_seed47.json",
}
CANDIDATE = {seed: f"configs/featnorm_132d_csz_seed{seed}.json" for seed in SEEDS}
PLAN_TEMPLATE = {seed: f"configs/data_probe_132d_20261027_seed{seed}_plan.json" for seed in SEEDS}

HOST = ("hz-t3.matpool.com:26155 (36 cores / 192 GB / 3x Tesla P100-SXM2-16GB, "
        "driver 550.163.01 / CUDA 12.2)")
STEPS = {"data-1.32": 43488, "data-2.64": 86976}

FROZEN = [
    "model.kind", "model.hidden_dim", "model.num_layers", "data.feature_columns",
    "data.lookback", "data.sample_times", "data.label_horizon_minutes",
    "data.outside_universe_labels", "data.price_scale", "model.target_centering",
    "model.target_scaling", "model.time_loss_weighting", "model.batch_size",
    "model.learning_rate", "evaluation_split", "evaluation", "acceptance",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: dict) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def plan_path(seed: int, arm: str) -> Path:
    return CONFIGS / f"featnorm_132d_seed{seed}_{arm}_plan.json"


def write_candidate_recipes() -> None:
    for seed in SEEDS:
        recipe = load(ROOT / CONTROL[seed])
        recipe["data"]["feature_transform"] = TRANSFORM
        dump(ROOT / CANDIDATE[seed], recipe)


def write_prereg() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT,
        "frozen_at_utc": now(),
        "author": "captain",
        "question_zh": (
            "固定 MLPLOB v3 CSZ 132 日配方与 1.32/2.64 两个预算点，只把输入特征换成"
            "价格相对化 + 成交量 log1p，样本外排序表现是否改善"),
        "hypothesis_zh": (
            "标签已按时间戳横截面 z-score，而特征侧是原始盘口价乘 100 加原始股数，"
            "同一时间戳跨股票排序时分数尺度不可比。把价格相对化到窗口自身的 mid、"
            "成交量取 log1p，应减少模型在 40 维隐层里同时学绝对价位不变性所需的工作量，"
            "从而提高 RankIC。"),
        "factor": {
            "changed": "data.feature_transform",
            "from": "none",
            "to": TRANSFORM,
            "definition": {
                "prices": "(p - ref) / ref, ref = (ask_price1 + bid_price1) / 2 at the window's own last step, read after forward fill",
                "volumes": "log1p(max(v, 0))",
                "invalid_ref": "a missing or non-positive ref zeroes that window's price channels",
                "causality": "per (instrument, window); no future row, no cross-window or dataset statistic",
                "code": "src/jinyiquant/data.py::relative_price_log_volume, called from window_arrays only",
            },
            "default_path_unchanged": (
                "with the key absent the window bytes are unchanged "
                "(X_SHA 21feb1ac535dea5cee5cd43666880586aeb513760b83f84a2b0d069b8222ce43)"),
        },
        "arms": {
            "control": {"data.feature_transform": "none", "recipes": [CONTROL[s] for s in SEEDS]},
            "candidate": {"data.feature_transform": TRANSFORM, "recipes": [CANDIDATE[s] for s in SEEDS]},
        },
        "rung": {
            "train_trading_days": 132,
            "train_window": ["2023-04-14", "2023-10-31"],
            "validation": ["2023-11-01", "2023-11-15"],
            "note_zh": "沿用已注册 132 日配方的时间切分，只改特征变换一个因素",
        },
        "seeds": list(SEEDS),
        "primary_seed": 11,
        "budget_points": [
            {"name": "data-1.32", "effective_epochs": 1.32, "steps": STEPS["data-1.32"]},
            {"name": "data-2.64", "effective_epochs": 2.64, "steps": STEPS["data-2.64"]},
        ],
        "primary": {
            "metric": "rank_ic",
            "point": "data-1.32",
            "caliber": ("configs/training_trajectory_replay_20261027_plan.json: 5 windows "
                        "(2023-11, 2024-01, 2024-03, 2024-04, 2024-06), 104 trading days, eval-v2"),
            "pairing": "candidate - control within (seed, trading day), then mean over the three seeds",
            "uncertainty": "5-trading-day moving block bootstrap, 20000 draws",
        },
        "thresholds": {
            "min_improvement": 0.003,
            "ci_must_exclude_zero": True,
            "min_positive_windows": 4,
            "windows_total": 5,
            "note_zh": "+0.003 约为 132 日 9 种子标准差 0.00247 的 1.2 倍",
        },
        "secondary": {
            "point_2.64": "descriptive at the same caliber; reported, not gated",
            "turnover_degradation_max": 0.0318,
            "net5": "reported at 5bp for both arms",
        },
        "decision_rule_zh": (
            "当且仅当三种子配对平均 RankIC 改善 >= 0.003、其 95% 区间下界 > 0、"
            "且 5 个窗口中至少 4 个为正时，判定特征归一化值得进入配方；否则判定不通过并停止该线。"),
        "cost_cap": {"runs": 6, "steps_per_run": STEPS["data-2.64"], "wall_hours": 3},
        "no_threshold_change": True,
        "homogeneity": {
            "host": HOST,
            "torch": "2.6.0+cu124",
            "threads_per_process": 1,
            "one_process_per_gpu": True,
            "note_zh": "两臂同机同 torch；控制臂同时是跨机复现检查，但跨机一致性只作描述，不作判据",
        },
        "ordering": {
            "registered_before_any_feature_transform_training_started": True,
            "note_zh": "本文件落盘时该线一次训练都没有跑过，也没有任何 RankIC 被读取",
        },
        "scope_limits_zh": [
            "132 日档的结论不外推到 441/927 档，后者需要单独预登记",
            "若通过，927 三种子的配方作废：不能与旧配方模型混编，也不能直接当提交候选",
            "本轮不追求晋升，只判定特征归一化是否值得进入配方",
        ],
    }


def write_plan(seed: int, arm: str, prereg_sha: str) -> Path:
    plan = copy.deepcopy(load(ROOT / PLAN_TEMPLATE[seed]))
    recipe = CONTROL[seed] if arm == "control" else CANDIDATE[seed]
    path = plan_path(seed, arm)
    plan["experiment_id"] = f"{EXPERIMENT}-seed{seed}-{arm}"
    plan["created_at_utc"] = now()
    plan["status"] = "execution_plan_of_registered_prereg"
    plan["registration"] = {
        "path": relative(PREREG),
        "sha256": prereg_sha,
        "registration_path": relative(REGISTRATION),
        "role": "primary" if seed == 11 else "support",
        "note": ("the primary seed is the only escalation seed; support seeds are descriptive "
                 "stability references"),
    }
    plan["question_zh"] = (
        f"种子 {seed}、{'对照臂（特征不变）' if arm == 'control' else '候选臂（价格相对化 + 成交量 log1p）'}："
        "132 日配方、1.32/2.64 两个预算点上的样本外排序表现")
    plan["hypothesis"] = (
        "Holding the frozen 132-day recipe and the two effective-epoch budget points fixed, "
        + ("the control arm reproduces the registered baseline on this host."
           if arm == "control" else
           "replacing the raw price/volume channels with window-relative prices and log volumes "
           "raises out-of-sample RankIC.")
    )
    if arm == "control":
        plan["factor"] = {
            "changed": "control.identity",
            "note": "control arm: the recipe is the registered 132-day data-probe recipe, unchanged",
        }
    else:
        plan["factor"] = {
            "changed": "data.feature_transform",
            "from": "none",
            "to": TRANSFORM,
            "note": "the only factor that differs from the control arm",
        }
    plan["budget_points"] = [
        {"name": "data-1.32", "effective_epochs": 1.32, "steps": STEPS["data-1.32"],
         "role": "primary point of the registered comparison"},
        {"name": "data-2.64", "effective_epochs": 2.64, "steps": STEPS["data-2.64"],
         "role": "descriptive second point; reported, not gated"},
    ]
    plan["recipe_config"] = recipe
    plan["recipe_sha256"] = sha256(ROOT / recipe)
    plan["recipe_frozen"] = FROZEN + ["data.feature_transform"]
    plan["budget_only_overrides"] = dict(plan.get("budget_only_overrides") or {})
    plan["budget_only_overrides"]["note"] = (
        "feature_transform is the registered factor, not a budget override; any other deviation "
        "from the frozen recipe must fail the run guard")
    plan["cloud"] = {
        "host": HOST,
        "gpu_index": SEEDS.index(seed),
        "concurrency": {
            "processes": 3,
            "gpus": 3,
            "one_process_per_gpu": True,
            "threads_per_process": 1,
            "sharing": "one seed per card; both arms run in separate waves",
        },
    }
    dump(path, plan)
    return path


def generate() -> None:
    write_candidate_recipes()
    dump(PREREG, write_prereg())
    prereg_sha = sha256(PREREG)
    for seed in SEEDS:
        for arm in ("control", "pricerel"):
            write_plan(seed, arm, prereg_sha)
    print(f"prereg  {relative(PREREG)} {prereg_sha}")
    for seed in SEEDS:
        print(f"recipe  {CANDIDATE[seed]} {sha256(ROOT / CANDIDATE[seed])}")


def register() -> None:
    plans = {}
    for seed in SEEDS:
        for arm in ("control", "pricerel"):
            path = plan_path(seed, arm)
            plans[relative(path)] = sha256(path)
    value = {
        "schema_version": 1,
        "prereg_path": relative(PREREG),
        "prereg_sha256": sha256(PREREG),
        "registered_at_utc": now(),
        "host": HOST,
        "recipe_pins": {COND: sha256(ROOT / COND) for COND in list(CONTROL.values()) + list(CANDIDATE.values())},
        "plan_pins": plans,
        "ordering": {
            "registered_before_any_feature_transform_training_started": True,
            "note_zh": "登记时 hz-t3 上尚未启动任何一条该线训练",
        },
    }
    dump(REGISTRATION, value)
    print(f"registration {relative(REGISTRATION)} {sha256(REGISTRATION)}")


def check() -> int:
    failures = []
    registration = load(REGISTRATION)
    if registration["prereg_sha256"] != sha256(PREREG):
        failures.append("prereg sha256 does not match the registration")
    for path, pinned in registration["recipe_pins"].items():
        if sha256(ROOT / path) != pinned:
            failures.append(f"recipe sha256 drift: {path}")
    for path, pinned in registration["plan_pins"].items():
        if sha256(ROOT / path) != pinned:
            failures.append(f"plan sha256 drift: {path}")
    for path in registration["plan_pins"]:
        plan = load(ROOT / path)
        if plan["registration"]["sha256"] != sha256(PREREG):
            failures.append(f"plan points at a stale prereg: {path}")
        if plan["recipe_sha256"] != sha256(ROOT / plan["recipe_config"]):
            failures.append(f"plan recipe_sha256 drift: {path}")
    for recipe in list(CONTROL.values()) + list(CANDIDATE.values()):
        transform = load(ROOT / recipe)["data"].get("feature_transform", "none")
        want = "none" if recipe in CONTROL.values() else TRANSFORM
        if transform != want:
            failures.append(f"recipe transform mismatch: {recipe} has {transform!r}")
    for line in failures:
        print(f"FAIL {line}")
    print("PASS" if not failures else f"{len(failures)} failure(s)")
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.generate:
        generate()
    if args.register:
        register()
    if args.check:
        return check()
    if not (args.generate or args.register or args.check):
        parser.error("one of --generate/--register/--check is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
