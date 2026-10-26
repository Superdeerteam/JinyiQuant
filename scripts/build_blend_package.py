"""Build a frozen-weight package source for a fixed score blend.

The package carries the member checkpoints inside one schema-4 weights file plus the
exact member training configs, run ids and code, so the seven-file platform bundle can
be flattened and timed without touching the original training projects. Members may
use different torch trunks; the training config records the trunk composition.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jinyiquant.io import file_sha256, utc_now, write_json  # noqa: E402
from modified_tlob.adapter import BACKBONES, BLEND_KIND, BLEND_SCHEMA_VERSION  # noqa: E402

JINYI_FILES = ("__init__.py", "io.py", "data.py", "tracking.py")
TLOB_FILES = ("__init__.py", "adapter.py", "device.py", "model.py", "normalization.py",
              "schema.py", "submission.py", "provenance.json", "LICENSE")


def resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def check_config_matches(member, payload, config):
    model = config.get("model", {})
    stored = payload.get("model_config", {})
    ignored = {"device", "cpu_threads", "log_every_steps"}
    mismatched = {key: (value, stored.get(key)) for key, value in model.items()
                  if key not in ignored and stored.get(key) != value}
    if mismatched:
        raise ValueError(f"{member['name']}: training config differs from checkpoint: {mismatched}")


def build_package(spec_path, destination):
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    base = spec_path.parent
    blend_path = resolve(base, spec["blend_weights"])
    blend_sha = file_sha256(blend_path)
    if blend_sha != spec["blend_weights_sha256"]:
        raise ValueError("Blend checkpoint checksum changed")
    blend = json.loads(blend_path.read_text(encoding="utf-8"))
    if blend.get("kind") != BLEND_KIND or blend.get("schema_version") != BLEND_SCHEMA_VERSION:
        raise ValueError("Blend checkpoint must be the schema-4 fixed score blend")
    names = [entry["name"] for entry in blend["members"]]
    weights = {entry["name"]: float(entry["weight"]) for entry in blend["members"]}
    kinds = {entry["name"]: entry["payload"]["kind"] for entry in blend["members"]}
    trunk = next(iter(set(kinds.values()))) if len(set(kinds.values())) == 1 else "mixed"
    declared = blend.get("blend", {}).get("kind")
    if declared is not None and declared != trunk:
        raise ValueError(f"Blend trunk label {declared!r} disagrees with its members ({trunk!r})")
    if names != [member["name"] for member in spec["members"]]:
        raise ValueError(f"Member order/name mismatch: checkpoint {names}")
    if weights != {member["name"]: float(member["weight"]) for member in spec["members"]}:
        raise ValueError("Member weights differ from the package spec")
    member_records = []
    for member, entry in zip(spec["members"], blend["members"]):
        if entry["payload"].get("kind") not in BACKBONES:
            raise ValueError(f"{member['name']}: unsupported member kind")
        weights_path = resolve(base, member["weights"])
        digest = file_sha256(weights_path)
        if digest != member["sha256"] or digest != entry["sha256"]:
            raise ValueError(f"{member['name']}: member checksum changed")
        config_path = resolve(base, member["training_config"])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        check_config_matches(member, entry["payload"], config)
        member_records.append({
            "name": member["name"], "weight": float(member["weight"]),
            "training_run_id": member["training_run_id"],
            "training_config": str(config_path), "training_config_sha256": file_sha256(config_path),
            "weights_sha256": digest, "training": member.get("training"),
            "config": config,
        })
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Use a new package directory: {destination}")
    destination.mkdir(parents=True)
    for name in JINYI_FILES:
        target = destination / "jinyiquant" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "src" / "jinyiquant" / name, target)
    for name in TLOB_FILES:
        target = destination / "modified_tlob" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "modified_tlob" / name, target)
    shutil.copyfile(ROOT / "modified_tlob" / "inference.ipynb", destination / "inference.ipynb")
    shutil.copyfile(ROOT / "modified_tlob" / "requirements-cpu.lock.txt",
                    destination / "requirements-cpu.lock.txt")
    shutil.copyfile(blend_path, destination / "modified_tlob_weights.json")
    write_json(destination / "training_config.json", {
        "schema_version": 1, "kind": BLEND_KIND, "experiment_id": spec["experiment_id"],
        "trunk": trunk, "member_kinds": kinds,
        "blend": {"method": "score_weighted_mean", "weights": weights,
                  "units": "member_target_scale_restored_return_score"},
        "blend_weights_sha256": blend_sha, "members": member_records,
        "created_at_utc": utc_now(), "code": {"commit": git_commit()},
        "note": "Frozen-weight blend inference only; members were trained in separate projects."})
    contract = dict(spec["submission"])
    contract.update({"schema_version": 1, "weights_sha256": blend_sha})
    write_json(destination / "submission_config.json", contract)
    inventory = {path.relative_to(destination).as_posix(): file_sha256(path)
                 for path in destination.rglob("*") if path.is_file()}
    write_json(destination / "package_manifest.json", {
        "schema_version": 1, "files": inventory,
        "training_run_id": spec.get("training_run_id", spec["experiment_id"]),
        "platform_acceptance": "not_verified", "scope": "development"})
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(build_package(args.spec, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
