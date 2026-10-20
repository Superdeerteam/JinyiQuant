"""Compare the seven-file upload archive with its frozen-weight source package offline."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

EXPECTED_FILES = {
    "modified_tlob_weights.json", "provenance.json", "requirements.txt", "submission_config.json",
    "modified_tlob_flat.py", "jinyiquant_flat.py", "inference.ipynb",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(package, baseline_config, output, *, flat):
    import importlib.abc
    import socket

    package = package.resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(package))
    if flat:
        class BlockOriginalPackages(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"jinyiquant", "modified_tlob"}:
                    raise AssertionError(f"Flat package tried to import an original folder: {fullname}")
        sys.meta_path.insert(0, BlockOriginalPackages())

    def no_network(*_args, **_kwargs):
        raise AssertionError("Upload verification must not use the network")

    socket.socket.connect = no_network
    socket.create_connection = no_network
    import numpy as np
    import pandas as pd

    config = json.loads(baseline_config.read_text(encoding="utf-8"))["data"]
    features = config["feature_columns"]
    instruments = [f"S{i:03d}" for i in range(16)]
    dates = []
    for day in pd.date_range("2022-01-03", "2022-01-05"):
        dates.extend(pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=121, freq="min"))
        dates.extend(pd.date_range(day + pd.Timedelta(hours=13), periods=91, freq="min"))
    dates = pd.DatetimeIndex(dates)
    count = len(dates) * len(instruments)
    bars = pd.DataFrame({"date": np.repeat(dates, len(instruments)),
                         "instrument": np.tile(instruments, len(dates))})
    rng = np.random.default_rng(20)
    for index, name in enumerate(features):
        bars[name] = (12 + index * .01 + rng.normal(0, .01, count)
                      if "price" in name else rng.integers(10, 3000, count).astype(float))
    bars.loc[np.arange(count) % 139 == 0, features[0]] = -1
    start, end = pd.Timestamp("2022-01-04 09:30"), pd.Timestamp("2022-01-05 14:30")
    pool = bars.loc[bars.date.between(start, end) &
                    bars.date.dt.strftime("%H:%M").isin(config["sample_times"]),
                    ["date", "instrument"]].copy()
    queries = []

    def query(sql, *, filters):
        assert not any(name in sql.lower() for name in ("vwap", "target", "label"))
        columns = sql.split(" FROM ")[0].removeprefix("SELECT ").split(", ")
        table = pool if "instruments" in sql else bars
        lower, upper = map(pd.Timestamp, filters["date"])
        frame = table.loc[table.date.between(lower, upper), columns].copy()
        queries.append({"rows": len(frame), "start": str(lower), "end": str(upper),
                        "role": "pool" if table is pool else "bars"})
        return SimpleNamespace(df=lambda: frame)

    sys.modules["dai"] = SimpleNamespace(query=query)
    scope = {}
    for cell in json.loads((package / "inference.ipynb").read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "inference.ipynb", "exec"), scope)
    actual = scope["main"]({"bar1m": "bars", "instruments": "instruments"}, str(start), str(end))
    assert actual[["date", "instrument"]].equals(pool.reset_index(drop=True))
    assert np.isfinite(actual.factor).all()
    actual.to_parquet(output.with_suffix(".parquet"), index=False)
    result = {"passed": True, "rows": len(actual), "queries": queries,
              "prediction_sha256": hashlib.sha256(actual.factor.to_numpy().tobytes()).hexdigest(),
              "notebook_executed": True, "network": "blocked", "flat": flat}
    if flat:
        from importlib.metadata import version

        from packaging.requirements import Requirement

        dependencies = {}
        for line in (package / "requirements.txt").read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                requirement = Requirement(line)
                installed = version(requirement.name)
                assert requirement.specifier.contains(installed), (line, installed)
                dependencies[requirement.name] = installed
        archives = list((package / "inference_runs").glob("*/source.zip"))
        assert len(archives) == 1
        with zipfile.ZipFile(archives[0]) as archive:
            assert set(archive.namelist()) == EXPECTED_FILES - {"modified_tlob_weights.json"}
        assert not (package / "jinyiquant").exists() and not (package / "modified_tlob").exists()
        assert not any(n.split(".")[0] in {"jinyiquant", "modified_tlob"} for n in sys.modules)
        result.update(requirements_satisfied=dependencies, original_packages_imported=False,
                      flat_source_snapshot_verified=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def verify(baseline, archive_path, output_root):
    output_root.mkdir(parents=True, exist_ok=False)
    reference, flat = output_root / "reference", output_root / "flat"
    manifest = json.loads((baseline / "package_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        source = baseline / name
        assert source.resolve().is_relative_to(baseline.resolve()) and sha256(source) == expected
        target = reference / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(baseline / "package_manifest.json", reference / "package_manifest.json")
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None and set(archive.namelist()) == EXPECTED_FILES
        flat.mkdir()
        for name in sorted(EXPECTED_FILES):
            (flat / name).write_bytes(archive.read(name))
    weights_hash = sha256(flat / "modified_tlob_weights.json")
    assert weights_hash == manifest["files"]["modified_tlob_weights.json"]
    assert sha256(flat / "submission_config.json") == manifest["files"]["submission_config.json"]
    provenance = json.loads((flat / "provenance.json").read_text(encoding="utf-8"))
    for name, expected in provenance["flat_package"]["payload_sha256"].items():
        assert sha256(flat / name) == expected
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    for mode, package in (("reference", reference), ("flat", flat)):
        completed = subprocess.run([
            sys.executable, "-I", "-B", str(Path(__file__).resolve()), str(package),
            str(reference / "training_config.json"), "--output", str(output_root / f"{mode}.json"),
            "--worker", mode,
        ], cwd=output_root, env=environment, text=True, capture_output=True, timeout=120)
        (output_root / f"{mode}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"Isolated {mode} check failed: {output_root / (mode + '.log')}")
    import numpy as np
    import pandas as pd

    previous = pd.read_parquet(output_root / "reference.parquet")
    current = pd.read_parquet(output_root / "flat.parquet")
    pd.testing.assert_frame_equal(current, previous, check_exact=True)
    worker_result = json.loads((output_root / "flat.json").read_text(encoding="utf-8"))
    result = {"schema_version": 1, "passed": True, "scope": "frozen_real_weights_synthetic_DAI_inputs",
              "files": sorted(EXPECTED_FILES), "archive": str(archive_path),
              "archive_bytes": archive_path.stat().st_size, "archive_sha256": sha256(archive_path),
              "weights_sha256": weights_hash, "weights_byte_identical": True,
              "submission_config_byte_identical": True, "prediction_rows": len(current),
              "max_abs_prediction_difference": float(np.max(np.abs(current.factor - previous.factor))),
              "original_packages_imported": False, "notebook_executed": True,
              "requirements_satisfied": worker_result["requirements_satisfied"],
              "source_snapshot_verified": True, "network": "blocked", "platform_acceptance": "not_verified"}
    (output_root / "checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("reference", "flat"))
    args = parser.parse_args()
    if args.worker:
        worker(args.baseline, args.archive, args.output, flat=args.worker == "flat")
    else:
        print(json.dumps(verify(args.baseline.resolve(), args.archive.resolve(), args.output.resolve())))


if __name__ == "__main__":
    main()
