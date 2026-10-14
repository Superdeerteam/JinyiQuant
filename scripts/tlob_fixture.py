"""Deterministic LOB fixture and effective config; never model-quality evidence."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.io import read_json, write_json
from modified_tlob.schema import lob_columns


def create_fixture(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(314159)
    rows, labels = [], []
    for day in pd.bdate_range("2020-01-02", "2020-01-09"):
        date = day + pd.Timedelta(hours=10)
        for stock in range(5):
            instrument = f"{stock + 1:06d}.SZ"
            for lag in range(8):
                row = {"date": date - pd.Timedelta(minutes=7 - lag), "instrument": instrument}
                for level in range(1, 6):
                    row.update({f"ask_price{level}": 10 + stock + .01 * level + rng.normal(0, .001),
                                f"bid_price{level}": 10 + stock - .01 * level + rng.normal(0, .001),
                                f"ask_volume{level}": float(rng.integers(1, 1000)),
                                f"bid_volume{level}": float(rng.integers(1, 1000))})
                rows.append(row)
            labels.append({"date": date, "instrument": instrument, "vwap_return": rng.normal(0, .001),
                           "label_end": date + pd.Timedelta(minutes=30)})
    bars = pd.DataFrame(rows)
    bars.loc[1, "ask_price1"] = -1
    bars.loc[2, "ask_volume1"] = np.nan
    bars.to_parquet(destination / "bars.parquet", index=False)
    labels = pd.DataFrame(labels)
    labels.to_parquet(destination / "labels.parquet", index=False)
    labels[["date", "instrument"]].to_parquet(destination / "instruments.parquet", index=False)
    raw = read_json(Path(__file__).resolve().parents[1] / "configs/modified_tlob_cpu.json")
    raw["data"].update(bars_path="bars.parquet", labels_path="labels.parquet", instruments_path="instruments.parquet",
                       instrument_map_path=None, cache_dir=None,
                       feature_columns=list(lob_columns(3)), source="synthetic_LOB_engineering_fixture", version="1",
                       lookback=8, min_history=2, price_scale=1, history_days=1, sample_times=["10:00"])
    raw["splits"] = {"train": {"start": "2020-01-02", "end": "2020-01-06"},
                     "validation": {"start": "2020-01-06", "end": "2020-01-08"},
                     "final": {"start": "2020-01-08", "end": "2020-01-10"}}
    raw["model"].update(hidden_dim=8, num_layers=2, epochs=2, batch_size=3, log_every_steps=2)
    raw["output_root"] = "runs"
    write_json(destination / "config.json", raw)
    return destination / "config.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(create_fixture(parser.parse_args().output))
