"""Isolated worker: run one seven-file flat submission package over local official bars.

The DAI data source is replaced with an in-memory reader that serves the same tables
(``bars`` and ``instruments``) with platform units: quotes are converted from the
downloaded cent integers to yuan, and the contract keeps ``platform_price_scale=1``.
Original ``jinyiquant``/``modified_tlob`` packages are blocked so only the flat code
under test can execute.
"""

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow.feather as feather

FEATURES = [
    "ask_price1", "ask_volume1", "bid_price1", "bid_volume1",
    "ask_price2", "ask_volume2", "bid_price2", "bid_volume2",
    "ask_price3", "ask_volume3", "bid_price3", "bid_volume3",
]
PRICE_COLUMNS = [name for name in FEATURES if name.endswith(("price1", "price2", "price3"))]
MONTH = re.compile(r"(20\d{2})(0[1-9]|1[0-2])")


def read_bars(directory: Path, warm: pd.Timestamp, end: pd.Timestamp, mapping: dict[int, str]) -> pd.DataFrame:
    frames = []
    for file in sorted(directory.glob("*.feather")):
        match = MONTH.search(file.stem)
        if not match:
            continue
        month = pd.Timestamp(f"{match.group(1)}-{match.group(2)}-01")
        if month >= end or month + pd.offsets.MonthBegin(1) <= warm:
            continue
        table = feather.read_table(file, columns=["date", "instrument_id", *FEATURES]).to_pandas()
        table["date"] = pd.to_datetime(table["date"])
        table = table.loc[(table.date >= warm) & (table.date <= end)].copy()
        table["instrument"] = table["instrument_id"].map(mapping)
        if table.instrument.isna().any():
            raise ValueError(f"unmapped instrument_id in {file.name}")
        table = table.drop(columns="instrument_id")
        table[PRICE_COLUMNS] = table[PRICE_COLUMNS].astype("float64") / 100.0
        frames.append(table)
    if not frames:
        raise ValueError("No bar files overlap the requested range")
    bars = pd.concat(frames, ignore_index=True)
    return bars.sort_values(["date", "instrument"], ignore_index=True)[["date", "instrument", *FEATURES]]


def main():
    (package, start, end, bars_path, instruments_path, mapping_path, output, device) = sys.argv[1:9]
    package, output = Path(package).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(package))
    import importlib.abc

    class BlockOriginal(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"jinyiquant", "modified_tlob"}:
                raise AssertionError(f"Flat package imported original module: {fullname}")

    sys.meta_path.insert(0, BlockOriginal())
    mapping_frame = pd.read_csv(mapping_path)
    mapping = dict(zip(mapping_frame.instrument_id, mapping_frame.instrument))
    pool = pd.read_parquet(instruments_path, columns=["date", "instrument"])
    pool["date"] = pd.to_datetime(pool["date"])
    pool = pool.loc[(pool.date >= start) & (pool.date <= end)].reset_index(drop=True)
    scope = {}
    for cell in json.loads((package / "inference.ipynb").read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "inference.ipynb", "exec"), scope)
    contract = json.loads((package / "submission_config.json").read_text(encoding="utf-8"))
    history_days = 10
    warm = start - pd.Timedelta(days=history_days)
    bars = read_bars(Path(bars_path), warm, end, mapping)
    queries = []

    def query(sql, *, filters):
        columns = sql.split(" FROM ")[0].removeprefix("SELECT ").split(", ")
        table_name = re.search(r"FROM (\w+)", sql).group(1)
        frame = pool if table_name == "instruments" else bars
        lower, upper = map(pd.Timestamp, filters["date"])
        selected = frame.loc[frame.date.between(lower, upper), columns].copy()
        backfill = "QUALIFY" in sql
        if backfill:
            assert "WHERE date >= TIMESTAMP" in sql and "AND date < TIMESTAMP" in sql
            limit = int(re.search(r"<= (\d+)$", sql).group(1))
            selected = selected.loc[selected.date.lt(upper) & selected.instrument.isin(filters["instrument"])]
            selected = selected.sort_values("date").groupby("instrument", sort=False).tail(limit)
        queries.append({"table": table_name, "rows": len(selected), "backfill": backfill,
                        "start": str(lower), "end": str(upper)})
        return SimpleNamespace(df=lambda: selected)

    sys.modules["dai"] = SimpleNamespace(query=query)
    started = time.monotonic()
    result = scope["main"]({"bar1m": "bars", "instruments": "instruments"}, str(start), str(end))
    elapsed = time.monotonic() - started
    result.to_parquet(output / "predictions.parquet", index=False)
    run_root = package / "inference_runs"
    runs = sorted(path for path in run_root.glob("*") if path.is_dir()) if run_root.is_dir() else []
    run_manifest = json.loads((runs[-1] / "manifest.json").read_text(encoding="utf-8")) if runs else {}
    summary = {
        "package": str(package), "start": str(start), "end": str(end), "device": device,
        "rows": len(result), "elapsed_seconds": elapsed, "queries": len(queries),
        "backfill_queries": sum(1 for entry in queries if entry["backfill"]),
        "served_rows": sum(entry["rows"] for entry in queries),
        "history_handling": run_manifest.get("history_handling"),
        "prediction_sha256": hashlib.sha256(result.factor.to_numpy().tobytes()).hexdigest(),
        "platform_price_scale": contract.get("platform_price_scale"),
        "weights_sha256": contract.get("weights_sha256"),
        "original_module_imported": [name for name in sys.modules
                                     if name.split(".")[0] in {"jinyiquant", "modified_tlob"}],
        "rows_per_instrument": {key: float(value) for key, value in
                                result.groupby("instrument").size().describe().to_dict().items()},
    }
    (output / "worker.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"rows": len(result), "elapsed_seconds": elapsed, "fallback": summary["history_handling"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
