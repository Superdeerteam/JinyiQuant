"""C 阶段冻结：把单一冻结基线（九模型等权）做成可复核的冻结包与哈希清单。

产出（runs/c-freeze-20261027/<run_id>/）：

* ``package-full/``  —— 完整推理包（模块树 + 九成员权重 + 平台契约），由既有全包模板复制后替换；
* ``package-flat/``  —— 七文件上传包（权重超过 49 MB 时按既有规则分片），由 ``package_flat_submission.py`` 生成；
* ``freeze.json``    —— 冻结对象：权重/预处理/评价契约/费用情景/门槛/资源线/证据边界的哈希与取值；
* ``sha256_index.json`` —— 上述输入与产物的逐文件摘要；
* ``manifest.json``、``events.jsonl`` —— 运行档案。

只读输入，不修改历史记录；封存区间不读取（无 final split 读取证据，本脚本只读索引与已公开档案）。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "runs/submission-best-20261026/build"
TEMPLATE = ROOT / "dist/tlob-mlplob-submission-20261026-package"
PREREG = ROOT / "configs/b4_promotion_prereg_20261027.json"
REGISTRATION = ROOT / "configs/b4_promotion_prereg_20261027.registration.json"
B_VERDICT = ROOT / "runs/b-release-verify-20261027"
T15_DIR = ROOT / "runs/candidate-triage-20261027"
EXPECTED_WEIGHTS_SHA = "93df1dfb89eb187033f8e0f0225f39ea9ce86a5078d12514af1f7c098393ae5a"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def event(path: Path, stage: str, event_type: str, **context) -> None:
    record = {"schema_version": 1, "timestamp_utc": utc_now(), "stage": stage,
              "event_type": event_type, "context": context}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def child_env() -> dict:
    return {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"}


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def evaluation_contract() -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from jinyiquant import evaluation

    spec = json.dumps(evaluation.METRIC_SPEC, sort_keys=True, separators=(",", ":")).encode()
    return {"version": evaluation.EVALUATION_VERSION,
            "metric_spec_sha256": hashlib.sha256(spec).hexdigest(),
            "module_sha256": sha256_file(ROOT / "src/jinyiquant/evaluation.py"),
            "metric_spec_keys": sorted(evaluation.METRIC_SPEC)}


def viewed_windows() -> dict:
    path = ROOT / "runs/index/viewed_windows.json"
    if not path.exists():
        return {"path": None, "sha256": None}
    document = json.loads(path.read_text(encoding="utf-8"))
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
            "sealed_interval": document.get("sealed_interval") or document.get("sealed"),
            "not_read": document.get("not_read")}


def final_reads() -> list[str]:
    """Any archived run that claims the sealed split would invalidate the freeze."""
    offenders = []
    for manifest in ROOT.glob("runs/*/*/manifest.json"):
        try:
            document = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if document.get("evaluation_split") == "final":
            offenders.append(str(manifest.relative_to(ROOT)))
    return offenders


def assemble_package(run_dir: Path, blend: dict, spec: dict) -> Path:
    """Build the full inference package from the current repo sources + the frozen nine-member blend."""
    package = run_dir / "package-full"
    if package.exists():
        shutil.rmtree(package)
    sources = {
        "jinyiquant/__init__.py": ROOT / "src/jinyiquant/__init__.py",
        "jinyiquant/data.py": ROOT / "src/jinyiquant/data.py",
        "jinyiquant/io.py": ROOT / "src/jinyiquant/io.py",
        "jinyiquant/tracking.py": ROOT / "src/jinyiquant/tracking.py",
        "jinyiquant/trajectory.py": ROOT / "src/jinyiquant/trajectory.py",
        "modified_tlob/LICENSE": ROOT / "modified_tlob/LICENSE",
        "modified_tlob/__init__.py": ROOT / "modified_tlob/__init__.py",
        "modified_tlob/schema.py": ROOT / "modified_tlob/schema.py",
        "modified_tlob/device.py": ROOT / "modified_tlob/device.py",
        "modified_tlob/normalization.py": ROOT / "modified_tlob/normalization.py",
        "modified_tlob/model.py": ROOT / "modified_tlob/model.py",
        "modified_tlob/adapter.py": ROOT / "modified_tlob/adapter.py",
        "modified_tlob/submission.py": ROOT / "modified_tlob/submission.py",
        "modified_tlob/provenance.json": ROOT / "modified_tlob/provenance.json",
        "requirements-cpu.lock.txt": ROOT / "modified_tlob/requirements-cpu.lock.txt",
        "inference.ipynb": TEMPLATE / "inference.ipynb",
    }
    for relative, source in sources.items():
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    weights = package / "modified_tlob_weights.json"
    try:
        os.link(BUILD / "blend_weights.json", weights)
    except OSError:
        shutil.copy2(BUILD / "blend_weights.json", weights)
    weights_sha = sha256_file(weights)
    contract = {"schema_version": 1, **{key: spec["submission"][key] for key in
                                        ("device", "history_policy", "min_coverage",
                                         "platform_price_scale", "wall_seconds")},
                "weights_sha256": weights_sha}
    write_json(package / "submission_config.json", contract)
    provenance = {
        "schema_version": 1, "experiment_id": spec["experiment_id"], "commit": git_commit(),
        "repository": "jinyiquant C-stage frozen submission",
        "evidence_boundary": ("B 阶段放行判定为不放行，维持本冻结基线；指标均为本地 eval-v2 近似口径，"
                              "不是官方综合分；5bp 情景只作压力门"),
        "members": [{"name": member["name"], "weight": member["weight"], "sha256": member["sha256"],
                     "training_run_id": member.get("training_run_id"),
                     "training_steps": (member.get("training") or {}).get("steps")}
                    for member in spec["members"]],
        "blend": {"method": "score_weighted_mean", "weights": "equal over 9 members",
                  "units": "member_target_scale_restored_return_score"},
        "preprocessing": blend["members"][0]["payload"]["preprocessing"] if blend.get("members") else None,
        "flat_package": {"source_commit": git_commit(), "builder": "scripts/package_flat_submission.py"},
        "freeze": {"stage": "C", "created_at_utc": utc_now(),
                   "b_verdict": "not_released; nine_blend maintained"},
    }
    write_json(package / "provenance.json", provenance)
    # The full package owns the frozen member training configs: the offline flat check replays
    # the synthetic DAI inputs from members[0]["config"]["data"], and the package must describe
    # the nine-member blend it actually carries.
    training_config = {
        "schema_version": 1,
        "kind": blend.get("kind"),
        "trunk": "mixed",
        "experiment_id": spec["experiment_id"],
        "blend_weights_sha256": weights_sha,
        "created_at_utc": utc_now(),
        "code": {"commit": git_commit()},
        "member_kinds": sorted({(member.get("training") or {}).get("kind")
                                for member in spec["members"]
                                if (member.get("training") or {}).get("kind")}),
        "members": [],
        "blend": blend.get("blend") or {
            "method": "score_weighted_mean",
            "weights": {member["name"]: member["weight"] for member in spec["members"]}},
        "note": "Frozen equal-weight blend; member training configs are embedded for the offline flat check.",
    }
    for member in spec["members"]:
        config_path = Path(member["training_config"])
        training_config["members"].append({
            "name": member["name"], "weight": member["weight"], "sha256": member["sha256"],
            "weights_sha256": member["sha256"], "training_run_id": member.get("training_run_id"),
            "training": member.get("training"), "training_config": str(config_path),
            "training_config_sha256": sha256_file(config_path),
            "config": json.loads(config_path.read_text(encoding="utf-8")),
        })
    write_json(package / "training_config.json", training_config)
    files = {path.relative_to(package).as_posix(): sha256_file(path)
             for path in sorted(package.rglob("*"))
             if path.is_file() and path.name != "package_manifest.json"}
    write_json(package / "package_manifest.json", {
        "schema_version": 1, "files": files, "platform_acceptance": "not_verified",
        "scope": "development", "training_run_id": spec["experiment_id"],
        "weights_sha256": weights_sha, "kind": blend.get("kind"),
        "members": len(blend.get("members") or [])})
    return package


def build_flat(run_dir: Path, package: Path, events: Path) -> dict:
    output = run_dir / "package-flat"
    if output.exists():
        shutil.rmtree(output)
    archive = output.with_suffix(".zip")
    if archive.exists():
        archive.unlink()
    command = [sys.executable, str(ROOT / "scripts/package_flat_submission.py"), str(package), str(output)]
    event(events, "flat", "started", command=" ".join(command))
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=child_env())
    event(events, "flat", "completed" if result.returncode == 0 else "failed",
          returncode=result.returncode, stdout=result.stdout[-1500:], stderr=result.stderr[-1500:])
    files = sorted(path.name for path in output.iterdir()) if output.exists() else []
    return {"returncode": result.returncode, "files": files,
            "stdout_tail": result.stdout[-800:], "stderr_tail": result.stderr[-800:]}


def check_flat(run_dir: Path, package: Path, flat: Path, events: Path) -> dict:
    # package_flat_submission writes the upload archive beside the flat directory.
    archive = run_dir / "package-flat.zip"
    if not archive.exists():
        archive = next((path for path in (run_dir / "package-flat").glob("*.zip")), flat)
    output = run_dir / "flat-check"
    command = [sys.executable, str(ROOT / "scripts/check_flat_submission.py"), str(package),
               str(archive), "--output", str(output)]
    event(events, "flat_check", "started", command=" ".join(command))
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=child_env())
    checks_path = output / "checks.json"
    payload = json.loads(checks_path.read_text(encoding="utf-8")) if checks_path.exists() else {}
    event(events, "flat_check", "completed" if result.returncode == 0 else "failed",
          returncode=result.returncode, checks_sha256=sha256_file(checks_path) if checks_path.exists() else None)
    return {"returncode": result.returncode, "checks_sha256":
            sha256_file(checks_path) if checks_path.exists() else None,
            "checks": payload, "stdout_tail": result.stdout[-800:], "stderr_tail": result.stderr[-800:]}


def main() -> int:
    run_dir = ROOT / "runs/c-freeze-20261027" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_"
                                                  + os.urandom(4).hex())
    run_dir.mkdir(parents=True, exist_ok=True)
    events = run_dir / "events.jsonl"
    event(events, "run", "started", phase="C-freeze", git=git_commit())

    weights_path = BUILD / "blend_weights.json"
    blend = json.loads(weights_path.read_text(encoding="utf-8"))
    spec = json.loads((BUILD / "spec.json").read_text(encoding="utf-8"))
    weights_sha = sha256_file(weights_path)
    if weights_sha != EXPECTED_WEIGHTS_SHA or spec["blend_weights_sha256"] != EXPECTED_WEIGHTS_SHA:
        raise SystemExit(f"Frozen blend weights drifted: {weights_sha}")

    package = assemble_package(run_dir, blend, spec)
    flat = build_flat(run_dir, package, events)
    checked = check_flat(run_dir, package, run_dir / "package-flat", events) if flat["returncode"] == 0 else {
        "returncode": None, "checks": {}, "skipped": "flat build failed"}

    note = json.loads(PREREG.read_text(encoding="utf-8")) if PREREG.exists() else {}
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8")) if REGISTRATION.exists() else {}
    members = blend.get("members") or []
    preprocessing = [json.dumps(member["payload"]["preprocessing"], sort_keys=True) for member in members]
    index = {}
    for path in (weights_path, BUILD / "spec.json", PREREG, REGISTRATION,
                 ROOT / "src/jinyiquant/evaluation.py", ROOT / "scripts/promotion_gate.py",
                 package / "submission_config.json", package / "provenance.json",
                 package / "modified_tlob_weights.json"):
        if path.exists():
            index[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = sha256_file(path)
    for path in sorted((run_dir / "package-flat").glob("*")):
        if path.is_file():
            index[str(path.relative_to(run_dir))] = sha256_file(path)

    freeze = {
        "schema_version": 1, "stage": "C", "generated_at_utc": utc_now(),
        "single_candidate": None,
        "candidate_policy": "B 阶段无候选放行（t15 not_released）；冻结的单一对象是维持中的基线 nine_blend",
        "baseline": {
            "experiment_id": spec["experiment_id"],
            "members": [{"name": member["name"], "weight": member["weight"], "sha256": member["sha256"],
                         "training_run_id": member.get("training_run_id"),
                         "steps": (member.get("training") or {}).get("steps"),
                         "kind": (member.get("training") or {}).get("kind")} for member in members],
            "blend_weights_path": str(weights_path.relative_to(ROOT)),
            "blend_weights_sha256": weights_sha,
            "blend_kind": blend.get("kind"), "blend_method": blend.get("blend", {}).get("method"),
            "equal_weights": all(abs(member["weight"] - 1 / 9) < 1e-12 for member in members),
        },
        "preprocessing": {
            "shared_identical_across_members": len(set(preprocessing)) == 1,
            "block": members[0]["payload"]["preprocessing"] if members else None,
            "block_sha256": hashlib.sha256(preprocessing[0].encode()).hexdigest() if preprocessing else None,
        },
        "evaluation_contract": evaluation_contract(),
        "cost_scenarios": {"bps": [0, 2, 5, 10], "primary": 5,
                           "definition": "单边费率 r 的线性成本 r·U；5bp 为主压力情景，官方换手项本地不可算",
                           "reference_record": "modified_tlob/cost_curve_20261027.json"},
        "promotion_thresholds": {
            "prereg_path": str(PREREG.relative_to(ROOT)), "prereg_sha256": sha256_file(PREREG),
            "registration_path": str(REGISTRATION.relative_to(ROOT)),
            "registration_sha256": sha256_file(REGISTRATION),
            "registration_record": registration,
            "derived_thresholds": note.get("derived_thresholds"),
            "retired_thresholds": (note.get("supersedes") or {}).get("retired_thresholds"),
            "net5_limit": note.get("net5_limit"),
            "evaluation_version": note.get("evaluation_version"),
            "declared_periods": (note.get("periods") or {}).get("declared"),
            "gate_script": "scripts/promotion_gate.py",
            "gate_script_sha256": sha256_file(ROOT / "scripts/promotion_gate.py")
            if (ROOT / "scripts/promotion_gate.py").exists() else None,
            "evidence_rules": note.get("evidence_rules") or note.get("evidence"),
            "resource_budget": note.get("resource_budget") or note.get("resource"),
        },
        "submission_package": {
            "full_package": str(package.relative_to(ROOT)),
            "flat_build": flat, "offline_check": {"returncode": checked.get("returncode"),
                                                  "checks_sha256": checked.get("checks_sha256"),
                                                  "passed": all(bool(value.get("passed")) for key, value
                                                                in (checked.get("checks") or {}).items()
                                                                if isinstance(value, dict) and "passed" in value)
                                                  if checked.get("checks") else None},
        },
        "evidence_boundary": {
            "viewed_windows": viewed_windows(),
            "final_split_reads": final_reads(),
            "confirmation_region": ("没有未使用的独立确认区：5 个开发月均已看，封存区间 [2024-10-08, 2025-01-01) "
                                    "仍封闭，只能由 t11 的 FinalAccessClaim 一次性打开"),
            "sealed_interval": ["2024-10-08", "2025-01-01"],
        },
        "b_verdict": {"stage_b": "not_released", "reason": "比赛指标无候选达标；资源与经济性维度分别记录",
                      "candidate_triage_dir": str(T15_DIR.relative_to(ROOT)) if T15_DIR.exists() else None,
                      "b_release_verify_dir": str(B_VERDICT.relative_to(ROOT)) if B_VERDICT.exists() else None},
        "sha256_index": dict(sorted(index.items())),
        "open_items": ["t11 封存终检需追加 periods=封存窗、minimum_periods=1 的小扩展后一次性读取",
                       "t12 真实平台流程：用本冻结包按 scripts/replay_platform_submission.py 路径执行并记录内存/时限/覆盖"],
    }
    write_json(run_dir / "freeze.json", freeze)
    write_json(run_dir / "sha256_index.json", dict(sorted(index.items())))
    write_json(run_dir / "manifest.json", {
        "schema_version": 1, "run_id": run_dir.name, "stage": "C", "generated_at_utc": utc_now(),
        "git": git_commit(), "freeze_sha256": sha256_file(run_dir / "freeze.json"),
        "weights_sha256": weights_sha, "members": len(members),
        "flat_files": flat["files"], "flat_check_returncode": checked.get("returncode"),
        "environment": {"python": sys.version.split()[0], "executable": sys.executable},
    })
    event(events, "run", "completed", weights_sha256=weights_sha, flat_files=len(flat["files"]),
          flat_check_returncode=checked.get("returncode"))
    print(json.dumps({"run_dir": str(run_dir), "weights_sha256": weights_sha,
                      "flat_returncode": flat["returncode"], "flat_files": flat["files"],
                      "check_returncode": checked.get("returncode"),
                      "check_passed": freeze["submission_package"]["offline_check"]["passed"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
