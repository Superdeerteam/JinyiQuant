"""Sealed final-check pre-registration extension: template, instantiation and validation.

The template is a landable pre-registration: it inherits the measured threshold
basis from ``configs/b4_promotion_prereg_20261027.json`` and changes exactly two
things - the evaluation periods become the single sealed window
``[2024-10-08, 2025-01-01)`` and the repeat condition drops to one period.

Nothing here reads the sealed interval. The only file touched outside the
pre-registration is ``runs/index/viewed_windows.json``, and only its
determination fields (``determination``, ``intersects_sealed`` and the
``counts.intersects_sealed_*`` tallies); the guard fails closed if any indexed
run already intersects the sealed window.

Usage:
    python scripts/sealed_final_prereg.py --check
    python scripts/sealed_final_prereg.py --register --baseline <dir> --candidate <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.promotion_gate import (
    SEALED_INTERVAL,
    SEALED_PERIOD,
    Refused,
    sealed_index_guard,
    sha256_file,
    validate_sealed_scope,
    verify_registration,
)

ROOT = Path(__file__).resolve().parents[1]
MAIN_PREREG = ROOT / "configs/b4_promotion_prereg_20261027.json"
TEMPLATE = ROOT / "configs/b4_sealed_final_prereg_template_20261027.json"
DEFAULT_OUTPUT = ROOT / "configs/b4_sealed_final_prereg_20261027.json"
DEFAULT_REGISTRATION = ROOT / "configs/b4_sealed_final_prereg_20261027.registration.json"


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(document, path: Path) -> str:
    payload = (canonical_json(document) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def registered_main() -> tuple[dict, str]:
    """Load the main pre-registration exactly as registered (bytes and hash)."""
    registration_path = ROOT / "configs/b4_promotion_prereg_20261027.registration.json"
    registration = verify_registration(MAIN_PREREG, registration_path)
    return registration["prereg"], registration["prereg_sha256"]


def template_document() -> dict:
    """Deterministic template content; ``--check`` compares the file with these bytes."""
    main, main_sha = registered_main()
    evidence = dict(main["evidence_rules"])
    evidence["minimum_periods"] = 1
    evidence["note"] = "the sealed interval is one period; no further months are required because " \
                       "the interval is read exactly once"
    return {
        "schema_version": 1,
        "registration_id": "b4-sealed-final-20261027",
        "phase": "C",
        "kind": "sealed_final_extension",
        "template": True,
        "owner_role": main["owner_role"],
        "evaluation_version": main["evaluation_version"],
        "purpose": "one-shot sealed final check of the frozen baseline and the single frozen candidate "
                   "on [2024-10-08, 2025-01-01); land it with --register immediately before reading the "
                   "sealed interval and never re-open that interval",
        "extends": {
            "path": MAIN_PREREG.relative_to(ROOT).as_posix(),
            "sha256": main_sha,
            "relationship": "same threshold_basis, same estimator, same secondary degradation lines, "
                            "same resource budget; only the periods and the repeat condition change",
            "inherited_blocks": ["economic_risk_boundary", "metric_caliber_layers",
                                 "pre_registration_completeness", "resource_budget", "supersedes"],
        },
        "instantiate_with": "PYTHONPATH=src;. .venv/Scripts/python.exe scripts/sealed_final_prereg.py "
                            "--register --baseline <sealed baseline object> --candidate <sealed "
                            "candidate object>",
        "placeholders": {
            "registered_at": "UTC timestamp written at instantiation; must be later than every "
                             "candidate artifact start",
            "baseline_object": "sealed-window re-evaluation of the incumbent baseline; recorded by path "
                               "at registration without reading its metrics",
            "candidate_object": "the single frozen candidate; recorded by path at registration without "
                                "reading its metrics",
        },
        "primary_metric": main["primary_metric"],
        "object_policy_template": dict(main["object_policy_template"]),
        "net5_limit": main["net5_limit"],
        "derived_thresholds": dict(main["derived_thresholds"]),
        "periods": {
            "declared": [[SEALED_PERIOD[0], SEALED_PERIOD[1]]],
            "names": ["sealed_2024q4"],
            "sealed": True,
            "sealed_interval": [SEALED_INTERVAL[0], SEALED_INTERVAL[1]],
            "single_period": True,
            "no_more_months_requirement": True,
            "rule": "the sealed interval is evaluated exactly once as one period; splitting it into "
                    "segments, shortening it, or reopening it is a repeated peek and is refused",
        },
        "evidence_rules": evidence,
        "verdict": {
            "states": ["accept", "reject", "inconclusive"],
            "labels": {"accept": "通过", "reject": "未通过", "inconclusive": "证据不足"},
            "no_threshold_relaxation": True,
            "decision_authority": "scripts/promotion_gate.py produces the verdict from this frozen "
                                  "registration; there is no manual override and no post-hoc edit",
            "conclusion_templates": {
                "accept": "封存终检通过：RankIC 配对日块增益 {improvement} >= 预登记门槛 "
                          "{min_improvement}，95% 区间 [{low}, {high}] 不含零，资源与次要/5bp 门均通过。",
                "reject": "封存终检未通过：{failed_gates}。门槛与情景未调整，维持冻结基线；不得据该结果"
                          "调整后再次评估同一区间。",
                "inconclusive": "封存终检证据不足：{missing_evidence}。区间只读一次，不重复取样；若需更多"
                                "证据只能另立方案并使用新的未用数据区。",
            },
        },
        "prohibitions": [
            "no lowering min_improvement, any secondary limit, or the 5bp stress scenario after seeing "
            "the sealed result",
            "no splitting the sealed interval into several periods and reporting the best one",
            "no re-running the sealed evaluation with a different candidate or a tuned variant",
            "no substituting research-window evidence for the sealed verdict",
            "no reporting local 5bp net return as an official score",
        ],
        "index_guard": {
            "path": "runs/index/viewed_windows.json",
            "fields_read": ["determination", "intersects_sealed", "counts.intersects_sealed_*"],
            "require": "intersects_sealed_true == 0",
            "rule": "if any indexed run already intersects the sealed interval the one-shot check is "
                    "compromised and the extension is refused",
        },
    }


def portable_path(path: Path) -> str:
    """Workspace-relative POSIX path when possible, so registrations survive a clone."""
    absolute = path.resolve()
    try:
        return absolute.relative_to(ROOT).as_posix()
    except ValueError:
        return absolute.as_posix()


def register(baseline: Path, candidate: Path, output: Path, registration_path: Path) -> int:
    output = output if output.is_absolute() else ROOT / output
    registration_path = registration_path if registration_path.is_absolute() else ROOT / registration_path
    if output.exists() or registration_path.exists():
        raise Refused(f"{output.name} already exists: the sealed final check is registered once and "
                      "never re-registered")
    for label, path in (("baseline", baseline), ("candidate", candidate)):
        if not path.is_dir():
            raise Refused(f"{label} artifact directory not found: {path}")
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    if template != template_document():
        raise Refused("template bytes differ from the builder; regenerate it before registering")
    validate_sealed_scope(template, require_registered=False)
    guard = sealed_index_guard()
    if guard["intersects_sealed_true"]:
        raise Refused("an indexed run already intersects the sealed interval; the one-shot check "
                      "is compromised")
    document = dict(template)
    document["template"] = False
    document["registered_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    document["baseline_object"] = {"object_id": baseline.name, "path": portable_path(baseline)}
    document["candidate_object"] = {"object_id": candidate.name, "path": portable_path(candidate)}
    document["registered_by"] = "scripts/sealed_final_prereg.py --register"
    validate_sealed_scope(document, require_registered=True)
    sha = write_json(document, output)
    write_json({"schema_version": 1, "registration_id": document["registration_id"],
                "registered_at": document["registered_at"], "prereg_sha256": sha,
                "prereg_path": portable_path(output),
                "extends": document["extends"], "index_guard": guard,
                "candidate_state_at_registration": "no metric from the sealed interval had been read "
                                                   "at registration; artifacts were recorded by path"},
               registration_path)
    print(f"registered {output.relative_to(ROOT).as_posix()} sha256={sha[:16]} "
          f"registered_at={document['registered_at']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="verify the template file matches the builder and the sealed scope rules")
    parser.add_argument("--register", action="store_true",
                        help="instantiate the template into a concrete registration (once)")
    parser.add_argument("--baseline", type=Path, help="sealed-window baseline object directory")
    parser.add_argument("--candidate", type=Path, help="sealed-window candidate object directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    args = parser.parse_args(argv)
    try:
        if args.check:
            on_disk = json.loads(TEMPLATE.read_text(encoding="utf-8"))
            built = template_document()
            if on_disk != built:
                raise Refused("template file differs from the builder output")
            validate_sealed_scope(on_disk, require_registered=False)
            guard = sealed_index_guard()
            print(f"template ok: thresholds inherited, periods=[[{SEALED_PERIOD[0]}, "
                  f"{SEALED_PERIOD[1]}]], sealed_interval={list(SEALED_INTERVAL)}")
            print(f"index guard: intersects_sealed_true={guard['intersects_sealed_true']} "
                  f"false={guard['intersects_sealed_false']} unknown={guard['intersects_sealed_unknown']}")
            print(f"template sha256={sha256_file(TEMPLATE)[:16]}")
            return 0
        if args.register:
            if args.baseline is None or args.candidate is None:
                raise Refused("--register needs --baseline and --candidate artifact directories")
            return register(args.baseline, args.candidate, args.output, args.registration)
        parser.print_help()
        return 2
    except (Refused, ValueError, KeyError, OSError, TypeError) as exc:
        print(f"REFUSED: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
