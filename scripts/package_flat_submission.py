"""Flatten the checked platform inference dependency graph into seven upload files."""

import argparse
import ast
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from jinyiquant.io import file_sha256, utc_now, write_json

UPLOAD_FILES = {
    "modified_tlob_weights.json", "provenance.json", "requirements.txt", "submission_config.json",
    "modified_tlob_flat.py", "jinyiquant_flat.py", "inference.ipynb",
}
JINYI_MODULES = ("io", "data", "tracking")
TLOB_MODULES = ("schema", "device", "normalization", "model", "adapter", "submission")

FLAT_CAPTURE = '''
def capture_source(run_dir):
    names = ("jinyiquant_flat.py", "modified_tlob_flat.py", "inference.ipynb",
             "requirements.txt", "provenance.json", "submission_config.json")
    paths = [PROJECT_ROOT / name for name in names]
    inventory = {p.name: file_sha256(p) for p in paths}
    with zipfile.ZipFile(run_dir / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.name)
    provenance = read_json(PROJECT_ROOT / "provenance.json")
    return {"commit": provenance["flat_package"]["source_commit"],
            "working_tree_status": None, "snapshot": "source.zip",
            "snapshot_sha256": file_sha256(run_dir / "source.zip"),
            "content_sha256": digest(inventory), "files": inventory,
            "layout": "seven_flat_files"}
'''

SCHEMA_FORWARD = '''
def lob_trade_columns():
    # Resolve only when called, after both flat modules have finished importing.
    from modified_tlob_flat import lob_trade_columns as columns
    return columns()
'''


class FlatImports(ast.NodeTransformer):
    def __init__(self, package, module):
        self.package, self.module = package, module

    def visit_ImportFrom(self, node):
        if node.module == "__future__":
            return None
        if self.package == "jinyiquant":
            if node.level and node.module == "io":
                return None
            if node.module == "modified_tlob.schema":
                if [alias.name for alias in node.names] != ["lob_trade_columns"]:
                    raise ValueError("Unexpected cross-package schema dependency")
                return ast.parse(SCHEMA_FORWARD).body[0]
            if node.module == "modified_tlob.device":
                node.module = "modified_tlob_flat"
        elif self.package == "modified_tlob":
            if node.level:
                if node.module not in TLOB_MODULES:
                    raise ValueError(f"Unexpected internal model dependency: {node.module}")
                return None
            if node.module and node.module.startswith("jinyiquant."):
                if node.module.split(".")[1] not in JINYI_MODULES:
                    raise ValueError(f"Unexpected inference dependency: {node.module}")
                node.module = "jinyiquant_flat"
        return node

    def visit_Assign(self, node):
        if self.module == "tracking" and any(
                isinstance(target, ast.Name) and target.id == "PROJECT_ROOT" for target in node.targets):
            return ast.parse("PROJECT_ROOT = Path(__file__).resolve().parent").body[0]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if self.module == "tracking" and node.name == "capture_source":
            return ast.parse(FLAT_CAPTURE).body[0]
        return self.generic_visit(node)


def merged_source(package_dir, package, modules, license_text=""):
    sections, definitions = [], set()
    for module in modules:
        path = package_dir / package / f"{module}.py"
        source = path.read_text(encoding="utf-8")
        if module == "submission":
            source = source.replace("Path(__file__).resolve().parents[1]", "Path(__file__).resolve().parent")
        tree = FlatImports(package, module).visit(ast.parse(source))
        ast.fix_missing_locations(tree)
        for node in tree.body:
            names = []
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [item.id for target in targets for item in ast.walk(target) if isinstance(item, ast.Name)]
            for name in names:
                if name in definitions:
                    raise ValueError(f"Flat module name collision: {package}.{name}")
                definitions.add(name)
        sections.append(f"# Source: {package}/{module}.py\n" + ast.unparse(tree))
    header = f'"""Generated flat platform inference module: {package}.\n{license_text}"""\n'
    result = header + "from __future__ import annotations\n\n" + "\n\n".join(sections) + "\n"
    compile(result, package + "_flat.py", "exec")
    return result


def build_flat_package(package_dir, destination):
    package_dir, destination = Path(package_dir).resolve(), Path(destination).resolve()
    archive_path = destination.with_suffix(".zip")
    if destination.exists() or archive_path.exists():
        raise FileExistsError("Use a new output directory and archive")
    manifest_path = package_dir / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = package_dir / name
        if not path.resolve().is_relative_to(package_dir) or file_sha256(path) != expected:
            raise ValueError(f"Package checksum mismatch: {name}")
    contract = json.loads((package_dir / "submission_config.json").read_text(encoding="utf-8"))
    if contract["weights_sha256"] != manifest["files"]["modified_tlob_weights.json"]:
        raise ValueError("Submission weight checksum mismatch")
    license_text = (package_dir / "modified_tlob/LICENSE").read_text(encoding="utf-8")
    sources = {
        "jinyiquant_flat.py": merged_source(package_dir, "jinyiquant", JINYI_MODULES),
        "modified_tlob_flat.py": merged_source(package_dir, "modified_tlob", TLOB_MODULES, license_text),
    }
    destination.mkdir(parents=True, exist_ok=False)
    for name, text in sources.items():
        (destination / name).write_text(text, encoding="utf-8", newline="\n")
    for name in ("modified_tlob_weights.json", "submission_config.json"):
        shutil.copyfile(package_dir / name, destination / name)
    notebook = json.loads((package_dir / "inference.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = [line.replace("from modified_tlob.submission import", "from modified_tlob_flat import")
                              for line in cell["source"]]
            cell["outputs"], cell["execution_count"] = [], None
    write_json(destination / "inference.ipynb", notebook)
    # Use one public Torch version so an installed CPU or CUDA local build can
    # satisfy the same requirement. Torch resolves its own OS/GPU dependencies.
    requirements = ["# Python 3.11. BigQuant provides proprietary dai; do not install an unrelated PyPI package.",
                    "# Existing torch 2.5.0 CPU/CUDA local builds both satisfy torch==2.5.0."]
    for line in (package_dir / "requirements-cpu.lock.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", "--")):
            requirements.append("torch==2.5.0" if line.startswith("torch==") else line)
    (destination / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    provenance = json.loads((package_dir / "modified_tlob/provenance.json").read_text(encoding="utf-8"))
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    provenance["flat_package"] = {
        "schema_version": 1, "created_at": utc_now(), "source_commit": code_commit,
        "source_package_manifest_sha256": file_sha256(manifest_path),
        "training_run_id": manifest["training_run_id"],
        "weights_sha256": contract["weights_sha256"], "weights_changed": False,
        "history_policy": contract.get("history_policy", {"mode": "strict"}),
        "scope": "Frozen-weight platform inference; original training project remains separate.",
        "source_modules": {f"{package}/{module}.py": manifest["files"][f"{package}/{module}.py"]
                           for package, modules in (("jinyiquant", JINYI_MODULES), ("modified_tlob", TLOB_MODULES))
                           for module in modules},
        "layout_changes": ["Merge inference dependencies with imports adjusted to the two flat files.",
                           "Resolve artifact paths beside flat files; capture their source in runtime archives.",
                           "Preserve daily streaming, model arithmetic, preprocessing and inference ordering."],
        "license_included_in": "modified_tlob_flat.py module docstring",
        "platform_acceptance": "not_verified",
        "payload_sha256": {path.name: file_sha256(path) for path in sorted(destination.iterdir())},
    }
    write_json(destination / "provenance.json", provenance)
    assert {p.name for p in destination.iterdir()} == UPLOAD_FILES
    assert file_sha256(destination / "modified_tlob_weights.json") == contract["weights_sha256"]
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(UPLOAD_FILES):
            archive.write(destination / name, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None and set(archive.namelist()) == UPLOAD_FILES
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(build_flat_package(args.package_dir, args.destination))


if __name__ == "__main__":
    main()
