"""Verify an explicit security map against a bounded real-data slice."""

import argparse
from pathlib import Path

import pandas as pd

from jinyiquant.config import load_config
from jinyiquant.data import _read, _read_file, load_data
from jinyiquant.io import file_sha256, utc_now, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    raw = config.snapshot()
    raw["data"]["instrument_map_path"] = str(args.mapping.resolve())
    raw["output_root"] = str(output / "runs")
    raw["experiment_id"] = "real-data-mapping-check"
    raw["hypothesis"] = "Explicit user-supplied map resolves the missing security-ID dependency on the declared real slice"
    write_json(output / "config.json", raw)
    config = load_config(output / "config.json")
    mapping = pd.read_csv(args.mapping, dtype={"instrument": "string"})
    report = {"schema_version": 1, "timestamp_utc": utc_now(), "mapping_path": str(args.mapping.resolve()),
              "mapping_sha256": file_sha256(args.mapping), "mapping_rows": len(mapping),
              "columns": list(mapping.columns), "provenance": "user supplied; structural and cross-file verification",
              "checks": []}

    def check(name, passed, **details):
        report["checks"].append({"name": name, "passed": bool(passed), "details": details})

    check("required_columns", set(mapping.columns) == {"instrument_id", "instrument"})
    if report["checks"][-1]["passed"]:
        check("nonempty_nonnull", len(mapping) > 0 and not mapping.isna().any().any())
        check("one_to_one", not mapping.instrument_id.duplicated().any() and not mapping.instrument.duplicated().any())
        check("integer_ids", pd.api.types.is_integer_dtype(mapping.instrument_id.dtype))
        check("security_format", mapping.instrument.str.fullmatch(r"\d{6}\.(SH|SZ|BJ)").all())
        start, end = config.splits.train.start, config.splits.validation.end
        bars = _read_file(config.data.bars_path, ["date", "instrument_id"], start, end)
        missing = sorted(set(bars.instrument_id) - set(mapping.instrument_id))
        check("real_bar_id_coverage", not missing, bar_rows=len(bars), bar_unique_ids=int(bars.instrument_id.nunique()),
              unmapped_ids=missing, start=str(start), end=str(end))
        universe = _read_file(config.data.instruments_path, ["date", "instrument"], start, end)
        missing = sorted(set(universe.instrument) - set(mapping.instrument))
        coverage = universe.assign(mapped=universe.instrument.isin(mapping.instrument)).groupby("date").mapped.mean()
        check("dated_universe_coverage", len(coverage) > 0 and coverage.min() >= config.acceptance.min_coverage,
              universe_rows=len(universe), universe_unique_instruments=int(universe.instrument.nunique()),
              unmapped_instruments=missing, minimum_section_coverage=float(coverage.min()) if len(coverage) else 0,
              required=config.acceptance.min_coverage, tolerated_missing=bool(missing))
        labels, records = _read(config.data.labels_path, ["date", "instrument", "vwap_return", "target"], start, end)
        joined = labels.merge(universe.assign(in_universe=True), on=["date", "instrument"], how="left")
        extra = joined.loc[joined.in_universe.isna()]
        report["label_universe_diagnostic"] = {
            "label_rows": len(labels), "matched_rows": int(joined.in_universe.notna().sum()),
            "outside_universe_rows": len(extra), "outside_universe_unique_codes": int(extra.instrument.nunique()),
            "outside_universe_counts": {str(k): int(v) for k, v in extra.instrument.value_counts().head(20).items()},
            "source_records": records,
            "note": "Diagnostic only; no labels or universe memberships are modified",
        }
        try:
            bundle = load_data(config.data, start=start, end=end)
            write_json(output / "data.json", bundle.metadata)
            aligned = bundle.labels.merge(bundle.bars[["date", "instrument"]],
                                          on=["date", "instrument"], how="left", indicator=True)
            check("real_data_interface_load", True, bar_rows=len(bundle.bars), label_rows=len(bundle.labels),
                  universe_rows=len(bundle.universe), exact_timestamp_bar_matches=int((aligned._merge == "both").sum()))
        except (ValueError, OSError) as exc:
            check("real_data_interface_load", False, exception_type=type(exc).__name__, reason=str(exc))
    report["mapping_confirmed"] = all(c["passed"] for c in report["checks"]
                                      if c["name"] not in {"real_data_interface_load", "dated_universe_coverage"})
    report["passed"] = all(c["passed"] for c in report["checks"])
    write_json(output / "mapping_check.json", report)
    print(output / "mapping_check.json", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
