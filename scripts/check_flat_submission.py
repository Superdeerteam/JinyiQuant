"""Compare flat upload files, including optional weight shards, with their frozen source."""

import argparse
import hashlib
import json
import os
import re
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
SHARDED_BLEND_KIND = "modified_tlob_sharded_blend"
UPLOAD_FILE_LIMIT = 50_000_000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_inventory(archive):
    """Resolve only explicitly listed, flat text shards before extracting any file."""
    names = archive.namelist()
    if len(names) != len(set(names)) or not EXPECTED_FILES.issubset(names):
        raise ValueError("Upload archive has missing or duplicate files")
    if any(info.file_size >= UPLOAD_FILE_LIMIT for info in archive.infolist()):
        raise ValueError("Each upload file must be below 50,000,000 bytes")
    payload = json.loads(archive.read("modified_tlob_weights.json"))
    shards = set()
    if payload.get("kind") == SHARDED_BLEND_KIND:
        for entry in payload["members"]:
            filename = entry["file"]
            if not isinstance(filename, str) or not re.fullmatch(r"modified_tlob_weights_\d{3,}\.json", filename):
                raise ValueError("Unsafe weight shard filename")
            if filename in shards:
                raise ValueError("Duplicate weight shard filename")
            shards.add(filename)
    expected = EXPECTED_FILES | shards
    if set(names) != expected:
        raise ValueError("Upload archive differs from its declared flat file inventory")
    return expected


def verify_weight_identity(reference, uploaded):
    """Check every stored tensor value and member weight, independent of prediction tests."""
    if sha256(reference) == sha256(uploaded):
        return True
    with Path(uploaded).open(encoding="utf-8") as source:
        manifest = json.load(source)
    if (manifest.get("kind") != SHARDED_BLEND_KIND
            or manifest.get("source_weights_sha256") != sha256(reference)):
        raise ValueError("Upload weights differ from their frozen source")
    with Path(reference).open(encoding="utf-8") as source:
        payload = json.load(source)
    if len(manifest["members"]) != len(payload["members"]):
        raise ValueError("Weight shard member count differs from source")
    for stored, original in zip(manifest["members"], payload["members"]):
        filename = stored["file"]
        if not re.fullmatch(r"modified_tlob_weights_\d{3,}\.json", filename):
            raise ValueError("Unsafe weight shard filename")
        shard = Path(uploaded).parent / filename
        if (stored["name"] != original["name"] or stored["weight"] != original["weight"]
                or shard.stat().st_size != stored["bytes"] or sha256(shard) != stored["sha256"]):
            raise ValueError("Weight shard member identity differs from source")
        expected = json.dumps(original["payload"], ensure_ascii=False, allow_nan=False,
                              separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(expected).hexdigest() != stored["sha256"]:
            raise ValueError("Weight shard tensor values differ from source")
    return False


def worker(package, baseline_config, output, *, flat, scenario="normal"):
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

    training = json.loads(baseline_config.read_text(encoding="utf-8"))
    # Single-model packages carry data at the top level; blend packages carry the
    # frozen member configs, whose first member defines the same preprocessing.
    config = training.get("data") or training["members"][0]["config"]["data"]
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
    if scenario == "opening-gap":
        # The pool keeps both securities. S000 has valid quotes only before the
        # original ten-day prefetch at the opening; S001 has no quotes at all.
        bars = bars.loc[~bars.instrument.eq("S001") &
                        ~(bars.instrument.eq("S000") & bars.date.le(start))].copy()
        prior = pd.DataFrame({"date": pd.date_range("2021-12-24 13:00", periods=config["lookback"], freq="min"),
                              "instrument": "S000"})
        for index, name in enumerate(features):
            prior[name] = 12 + index * .01 + np.arange(len(prior)) * .0001 if "price" in name else 100 + index
        bars = pd.concat([prior, bars], ignore_index=True)
    queries = []

    def query(sql, *, filters):
        assert not any(name in sql.lower() for name in ("vwap", "target", "label"))
        columns = sql.split(" FROM ")[0].removeprefix("SELECT ").split(", ")
        table_name = re.search(r"FROM (\w+)", sql).group(1)
        table = pool if table_name == "instruments" else bars
        lower, upper = map(pd.Timestamp, filters["date"])
        frame = table.loc[table.date.between(lower, upper), columns].copy()
        if "QUALIFY" in sql:
            assert "WHERE date >= TIMESTAMP" in sql and "AND date < TIMESTAMP" in sql
            assert len(filters["instrument"]) <= 50
            limit = int(re.search(r"<= (\d+)$", sql).group(1))
            frame = frame.loc[frame.date.lt(upper) & frame.instrument.isin(filters["instrument"])]
            frame = frame.sort_values("date").groupby("instrument", sort=False).tail(limit)
        queries.append({"rows": len(frame), "start": str(lower), "end": str(upper),
                        "role": "pool" if table is pool else "backfill" if "QUALIFY" in sql else "bars"})
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
              "notebook_executed": True, "network": "blocked", "flat": flat, "scenario": scenario}
    if scenario == "opening-gap":
        if flat:
            from jinyiquant_flat import build_batch, preprocess_bars
            from modified_tlob_flat import TLOBAdapter
        else:
            from jinyiquant.data import build_batch, preprocess_bars
            from modified_tlob.adapter import TLOBAdapter
        model = TLOBAdapter.load(package / "modified_tlob_weights.json", device="cpu")
        contract = json.loads((package / "submission_config.json").read_text(encoding="utf-8"))
        data = {**model.preprocessing, "price_scale": contract["platform_price_scale"]}
        prepared = preprocess_bars(bars, data)
        original = prepared.loc[prepared.date.ge(start - pd.Timedelta(days=data["history_days"]))]
        expected, originally_usable = [], []
        for _, group in pool.groupby("date", sort=True):
            base = build_batch(original, group, data)
            complete = build_batch(prepared, group, data)
            missing = base.effective_rows.sum(axis=1) < data["min_history"]
            # Independent oracle: leave every usable original window untouched;
            # only missing windows use all available earlier observations.
            for name in ("x", "observed", "history_dates"):
                getattr(base, name)[missing] = getattr(complete, name)[missing]
            expected.extend(model.predict(base))
            originally_usable.extend(~missing)
        np.testing.assert_array_equal(actual.factor.to_numpy(), expected)
        assert actual.loc[actual.instrument.eq("S001"), "factor"].eq(model.intercept).all()
        assert any(query["role"] == "backfill" for query in queries)
        result.update(independent_window_oracle_equal=True, saved_mean_fallback_verified=True,
                      originally_usable_samples=int(np.sum(originally_usable)))
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


def verify(baseline, archive_path, output_root, *, scenario="normal"):
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
        assert archive.testzip() is None
        upload_files = upload_inventory(archive)
        flat.mkdir()
        for name in sorted(upload_files):
            (flat / name).write_bytes(archive.read(name))
    weights_hash = sha256(flat / "modified_tlob_weights.json")
    weights_byte_identical = verify_weight_identity(reference / "modified_tlob_weights.json",
                                                    flat / "modified_tlob_weights.json")
    source_contract = json.loads((reference / "submission_config.json").read_text(encoding="utf-8"))
    upload_contract = json.loads((flat / "submission_config.json").read_text(encoding="utf-8"))
    assert upload_contract.pop("weights_sha256") == weights_hash
    source_contract.pop("weights_sha256")
    assert source_contract == upload_contract
    provenance = json.loads((flat / "provenance.json").read_text(encoding="utf-8"))
    for name, expected in provenance["flat_package"]["payload_sha256"].items():
        assert sha256(flat / name) == expected
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    for mode, package in (("reference", reference), ("flat", flat)):
        completed = subprocess.run([
            sys.executable, "-I", "-B", str(Path(__file__).resolve()), str(package),
            str(reference / "training_config.json"), "--output", str(output_root / f"{mode}.json"),
            "--worker", mode, "--scenario", scenario,
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
              "files": sorted(upload_files), "archive": str(archive_path),
              "archive_bytes": archive_path.stat().st_size, "archive_sha256": sha256(archive_path),
              "weights_sha256": weights_hash, "weights_byte_identical": weights_byte_identical,
              "weight_values_identical": True, "upload_file_limit_bytes": UPLOAD_FILE_LIMIT,
              "upload_file_bytes": {name: (flat / name).stat().st_size for name in sorted(upload_files)},
              "submission_config_byte_identical": sha256(flat / "submission_config.json")
                  == manifest["files"]["submission_config.json"], "prediction_rows": len(current),
              "max_abs_prediction_difference": float(np.max(np.abs(current.factor - previous.factor))),
              "original_packages_imported": False, "notebook_executed": True,
              "requirements_satisfied": worker_result["requirements_satisfied"],
              "source_snapshot_verified": True, "network": "blocked", "platform_acceptance": "not_verified"}
    result["scenario"] = scenario
    if scenario == "opening-gap":
        result.update(independent_window_oracle_equal=worker_result["independent_window_oracle_equal"],
                      saved_mean_fallback_verified=worker_result["saved_mean_fallback_verified"],
                      originally_usable_samples=worker_result["originally_usable_samples"])
    (output_root / "checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("reference", "flat"))
    parser.add_argument("--scenario", choices=("normal", "opening-gap"), default="normal")
    args = parser.parse_args()
    if args.worker:
        worker(args.baseline, args.archive, args.output, flat=args.worker == "flat", scenario=args.scenario)
    else:
        print(json.dumps(verify(args.baseline.resolve(), args.archive.resolve(), args.output.resolve(),
                                scenario=args.scenario)))


if __name__ == "__main__":
    main()
