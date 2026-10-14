"""Bounded CPU train/rerun/package/inference acceptance, with persistent JSON evidence."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tlob_fixture import create_fixture

from jinyiquant.checks import verify_artifacts
from jinyiquant.config import load_config
from jinyiquant.io import read_json, write_json
from jinyiquant.runner import run
from modified_tlob.package import build_package


def acceptance(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(destination / "plan.json", {"schema_version": 1, "hypothesis": "Engineering contracts only",
        "device": "cpu", "budget_seconds": 300, "repro_atol": 1e-6, "repro_rtol": 1e-5,
        "checks": ["train_save_load", "text_weights", "causality", "no_labels_in_inference",
                   "archive_integrity", "rerun", "isolated_offline_notebook"], "final_access": False})
    config = load_config(create_fixture(destination / "fixture"))
    attempts = []
    try:
        first, code = run(config)
        attempts.append(str(first))
        if code:
            raise RuntimeError(f"Initial CPU run failed: {first}")
        second, code = run(config, reference=first)
        attempts.append(str(second))
        if code:
            raise RuntimeError(f"CPU rerun failed: {second}")
        assert verify_artifacts(first)["passed"] and verify_artifacts(second)["passed"]
        package = build_package(first, destination / "upload")
        shutil.copyfile(first / "predictions.parquet", destination / "fixture/expected_predictions.parquet")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        smoke = destination / "package_smoke.json"
        result = subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("tlob_package_smoke.py")),
            str(package), str(destination / "fixture"), "--output", str(smoke)], cwd=package, env=env,
            capture_output=True, text=True, encoding="utf-8", timeout=120)
        (destination / "package_smoke.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError("Isolated package failed; see package_smoke.log")
        elapsed = time.monotonic() - started
        report = {"schema_version": 1, "passed": elapsed <= 300, "runs": attempts, "package": str(package),
                  "wall_seconds": elapsed, "budget_seconds": 300, "package_smoke": read_json(smoke),
                  "reproducibility": read_json(second / "manifest.json")["reproducibility"],
                  "real_data": "not_available", "gpu": "not_verified", "platform": "not_verified",
                  "conclusion": "Engineering evidence only; no prediction-quality improvement claim"}
        write_json(destination / "acceptance.json", report)
        return report
    except BaseException as exc:
        write_json(destination / "acceptance.json", {"schema_version": 1, "passed": False, "runs": attempts,
            "exception_type": type(exc).__name__, "reason": str(exc), "wall_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report = acceptance(parser.parse_args().output)
    print(json.dumps(report, ensure_ascii=True))
    raise SystemExit(0 if report["passed"] else 1)
