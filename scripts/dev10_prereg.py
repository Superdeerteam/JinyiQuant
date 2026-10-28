"""Register and check the dev10-postprocess preregistration (zero-training extension).

Chain: plan -> prereg (pins plan sha) -> registration (pins prereg sha, plan sha,
every checkpoint weight and every data month read by the replay or the analysis).

    python scripts/dev10_prereg.py --register
    python scripts/dev10_prereg.py --check

--register refuses to overwrite an existing registration. --check re-hashes every
pinned file and exits non-zero on any drift, so the protocol can be re-verified
before each reading of results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "dev10_replay_20261028_plan.json"
PREREG = ROOT / "configs" / "dev10_postprocess_prereg_20261028.json"
REGISTRATION = ROOT / "configs" / "dev10_postprocess_registration_20261028.json"

BARS_DIR = ROOT / "数据" / "cpt_jyc_2026_e2e_bar1m"
VWAP_DIR = ROOT / "数据" / "cpt_jyc_2026_vwap"
EXPOSURE_DIR = ROOT / "数据" / "cpt_jyc_2026_exposure"
STATIC_FILES = {
    "instruments": ROOT / "数据" / "cpt_jyc_2026_instruments(1).parquet",
    "instrument_map": ROOT / "数据" / "cpt_jyc_2026_stock_bar1m_instrument_mapping.csv",
    "exposure_2023": EXPOSURE_DIR / "cpt_jyc_2026_exposure_2023_part_0001.parquet",
    "exposure_2024": EXPOSURE_DIR / "cpt_jyc_2026_exposure_2024_part_0001.parquet",
}

# Months the replay reads: each window needs bars from the previous month as
# history and its own month; the analysis needs labels from one month before the
# first window (volatility lookback and the 70-day stress baseline).
BAR_MONTHS = ["2023-09", "2023-10"] + [f"2024-{m:02d}" for m in range(1, 10)]
VWAP_MONTHS = ["2023-08", "2023-09", "2023-10"] + [f"2024-{m:02d}" for m in range(1, 10)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path_text: str) -> Path:
    # Plan paths are resolved against the plan file directory, like the replay runner.
    path = Path(path_text)
    return path if path.is_absolute() else (PLAN.parent / path).resolve()


def single(pattern: Path) -> Path:
    hits = sorted(ROOT.glob(str(pattern.relative_to(ROOT))))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one file for {pattern}, found {len(hits)}")
    return hits[0]


def bar_file(month: str) -> Path:
    stamp = month.replace("-", "")
    return single(BARS_DIR / f"{stamp}.*feather")


def vwap_file(month: str) -> Path:
    path = VWAP_DIR / f"cpt_jyc_2026_vwap_{month}.parquet"
    if not path.is_file():
        raise SystemExit(f"missing label file {path}")
    return path


def pins(plan: dict) -> dict:
    weights = {}
    for version in plan["versions"]:
        path = resolve(version["weights"])
        weights[version["name"]] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)}
    return {
        "weights": weights,
        "bars": {month: {"path": str(bar_file(month).relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(bar_file(month))} for month in BAR_MONTHS},
        "vwap": {month: {"path": str(vwap_file(month).relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(vwap_file(month))} for month in VWAP_MONTHS},
        "static": {
            name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)}
            for name, path in STATIC_FILES.items()
        },
    }


def declared_versions(prereg: dict) -> set[str]:
    families = prereg["replay"]["families"]
    return set(families["submission_incumbents"]) | set(families["dose_rungs"])


def validate(prereg: dict, plan: dict) -> None:
    names = [window["name"] for window in plan["windows"]]
    if names != prereg["windows"]["all_ten"]:
        raise SystemExit(f"plan windows {names} differ from prereg all_ten {prereg['windows']['all_ten']}")
    versions = {version["name"] for version in plan["versions"]}
    if versions != declared_versions(prereg):
        raise SystemExit(f"plan versions differ from prereg: {sorted(versions ^ declared_versions(prereg))}")
    evaluation = plan["evaluation"]
    expected = {"quantile": 0.2, "periods_per_year": 252, "min_cross_section": 3, "constant_portfolios": "undefined"}
    if evaluation != expected:
        raise SystemExit(f"plan evaluation {evaluation} differs from eval-v2 defaults {expected}")
    if prereg["replay"]["plan_sha256"] != sha256(PLAN):
        raise SystemExit("prereg pins a different plan sha256 than the plan on disk")


def register() -> None:
    if REGISTRATION.exists():
        raise SystemExit(f"{REGISTRATION} exists; refusing to overwrite a registration")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    validate(prereg, plan)
    record = {
        "schema_version": 1,
        "registration_id": "dev10-postprocess-20261028",
        "prereg_path": str(PREREG.relative_to(ROOT)).replace("\\", "/"),
        "prereg_sha256": sha256(PREREG),
        "plan_path": str(PLAN.relative_to(ROOT)).replace("\\", "/"),
        "plan_sha256": sha256(PLAN),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "registered_by": "captain",
        "host": "local Windows box, NVIDIA RTX 2060 6GB, python 3.13.12, torch 2.11.0+cu128",
        "ordering": {
            "registered_before_replay_started": True,
            "registered_before_any_new_month_score": True,
            "windows_and_versions_fixed_in_plan": True,
        },
        "windows": prereg["windows"]["all_ten"],
        "total_days": prereg["windows"]["total_days"],
        "pins": pins(plan),
    }
    REGISTRATION.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"registered {REGISTRATION.relative_to(ROOT)} prereg_sha256={record['prereg_sha256']}")


def check() -> None:
    if not REGISTRATION.exists():
        raise SystemExit(f"missing {REGISTRATION}; run --register first")
    record = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    errors: list[str] = []

    def expect(name: str, value: str) -> None:
        if record.get(name) != value:
            errors.append(f"{name}: registration {record.get(name)} != current {value}")

    expect("prereg_sha256", sha256(PREREG))
    expect("plan_sha256", sha256(PLAN))
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    try:
        validate(prereg, plan)
    except SystemExit as error:
        errors.append(str(error))

    current = pins(plan)
    for group in ("weights", "bars", "vwap", "static"):
        registered = record["pins"][group]
        if set(registered) != set(current[group]):
            errors.append(f"pins/{group} key set changed")
            continue
        for key, entry in registered.items():
            if entry["path"] != current[group][key]["path"]:
                errors.append(f"pins/{group}/{key} path changed")
            elif entry["sha256"] != current[group][key]["sha256"]:
                errors.append(f"pins/{group}/{key} sha256 drifted")

    if errors:
        print("registration check FAILED:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print(f"registration check OK: {len(current['weights'])} weights, {len(current['bars'])} bar months, "
          f"{len(current['vwap'])} label months, {len(current['static'])} static files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--register", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.register:
        register()
    else:
        check()


if __name__ == "__main__":
    sys.exit(main())
