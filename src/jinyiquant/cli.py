"""Unified CLI; every run prints a machine-readable location and exit code."""

import argparse
from pathlib import Path

from .checks import verify_artifacts
from .config import load_config
from .demo import create_demo
from .experiments import calibrate, save_comparison, suite
from .io import canonical_json, read_json, redact
from .runner import failed_config, run, worker


def _check_input_paths(config):
    """Check input locations without opening data files or traversing directories."""
    checks = []
    for name in ("bars_path", "labels_path", "instruments_path", "instrument_map_path"):
        path = getattr(config.data, name)
        if path is None:
            continue
        kind = "file" if path.is_file() else "directory" if path.is_dir() else "missing_or_other"
        checks.append({"name": name, "path": str(path), "kind": kind,
                       "passed": kind in {"file", "directory"}})
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jinyi")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Create a synthetic engineering fixture without overwriting")
    demo.add_argument("--output", required=True, type=Path)
    validate = commands.add_parser("validate-config")
    validate.add_argument("config", type=Path)
    validate.add_argument("--check-inputs", action="store_true",
                          help="Check input paths exist as files/directories without reading their contents")
    execute = commands.add_parser("run")
    execute.add_argument("config", type=Path)
    rerun = commands.add_parser("rerun")
    rerun.add_argument("run_dir", type=Path)
    check = commands.add_parser("check", help="Verify persisted acceptance results and artifact integrity")
    check.add_argument("run_dir", type=Path)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("config", type=Path)
    calibration.add_argument("--seeds", type=int, nargs="+", required=True)
    calibration.add_argument("--periods", type=Path, required=True, help="JSON array of validation start/end periods")
    experiment = commands.add_parser("suite")
    experiment.add_argument("plan", type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("--baseline", nargs="+", type=Path, required=True)
    compare.add_argument("--candidate", nargs="+", type=Path, required=True)
    compare.add_argument("--policy", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare-data")
    prepare.add_argument("archive", type=Path)
    prepare.add_argument("--member", required=True)
    prepare.add_argument("--output", required=True, type=Path)
    hidden = commands.add_parser("_worker", help=argparse.SUPPRESS)
    hidden.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "_worker":
            return worker(args.run_dir)
        if args.command == "demo":
            print(canonical_json({"config": str(create_demo(args.output)), "scope": "synthetic engineering fixture"}))
            return 0
        if args.command in {"run", "validate-config", "calibrate"}:
            try:
                config = load_config(args.config)
            except Exception as exc:
                path, code = failed_config(args.config, exc)
                print(canonical_json({"run_dir": str(path), "exit_code": code, "status": "invalid_config"}))
                return code
            if args.command == "validate-config":
                result = {"valid": True, "config_sha256": config.checksum}
                if args.check_inputs:
                    checks = _check_input_paths(config)
                    result.update(inputs_present=all(check["passed"] for check in checks), checks=checks)
                print(canonical_json(result))
                return 0 if result.get("inputs_present", True) else 2
            if args.command == "run":
                path, code = run(config)
            else:
                path, code = calibrate(config, args.seeds, read_json(args.periods))
        elif args.command == "rerun":
            manifest = read_json(args.run_dir / "manifest.json")
            registration = {k: manifest[k] for k in ("comparison_policy_sha256", "comparison_policy_registered_at") if k in manifest}
            path, code = run(load_config(args.run_dir / "config.json"), reference=args.run_dir,
                             policy_registration=registration)
        elif args.command == "check":
            result = verify_artifacts(args.run_dir)
            acceptance = read_json(args.run_dir / "checks.json")
            result["passed"] = result["passed"] and acceptance["passed"]
            print(canonical_json(result))
            return 0 if result["passed"] else 3
        elif args.command == "compare":
            path, code = save_comparison(args.baseline, args.candidate, read_json(args.policy), args.output)
        elif args.command == "suite":
            path, code = suite(args.plan)
        elif args.command == "prepare-data":
            from .preparation import prepare_data
            path = prepare_data(args.archive, args.member, args.output)
            code = 0
        print(canonical_json({"output_dir": str(path.resolve()), "exit_code": code}))
        return code
    except (ValueError, OSError, RuntimeError) as exc:
        print(canonical_json({"exit_code": 2, "error_type": type(exc).__name__, "reason": redact(str(exc))}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
