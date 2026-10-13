"""Deterministic fixture generator; generated results are engineering evidence only."""

from pathlib import Path

import numpy as np
import pandas as pd

from .io import write_json


def create_demo(destination: Path):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(314159)
    rows, labels = [], []
    for day in pd.bdate_range("2020-01-02", "2020-02-14"):
        for time in ("10:00", "10:30", "14:00", "14:30"):
            date = day + pd.Timedelta(hours=int(time[:2]), minutes=int(time[3:]))
            for stock in range(8):
                signal = rng.normal()
                instrument = f"{stock + 1:06d}.SZ"
                for lag in range(5, -1, -1):
                    rows.append({"date": date - pd.Timedelta(minutes=lag), "instrument": instrument,
                                 "close": 10 + stock + signal + rng.normal(scale=.1),
                                 "volume": max(0, 1000 + 100 * signal + rng.normal(scale=20))})
                labels.append({"date": date, "instrument": instrument,
                               "target": .001 * signal + rng.normal(scale=.001),
                               "label_end": date + pd.Timedelta(minutes=30)})
    bars = pd.DataFrame(rows)
    # Exercise causal missing-value processing without dropping expected prediction keys.
    bars.loc[bars.index % 137 == 0, "close"] = -1
    bars.to_parquet(destination / "bars.parquet", index=False)
    targets = pd.DataFrame(labels)
    targets.to_parquet(destination / "labels.parquet", index=False)
    targets[["date", "instrument"]].to_parquet(destination / "instruments.parquet", index=False)
    config = {"schema_version": 1, "experiment_id": "engineering-smoke",
              "hypothesis": "Engineering acceptance only: causal inputs, complete archives and deterministic reruns",
              "output_root": "runs", "data": {"bars_path": "bars.parquet", "labels_path": "labels.parquet",
              "instruments_path": "instruments.parquet", "source": "synthetic_engineering_fixture",
              "version": "1", "lookback": 5, "min_history": 3,
              "sample_times": ["10:00", "10:30", "14:00", "14:30"]},
              "splits": {"train": {"start": "2020-01-02", "end": "2020-01-20"},
                         "validation": {"start": "2020-01-20", "end": "2020-02-03"},
                         "final": {"start": "2020-02-03", "end": "2020-02-15"}},
              "model": {"kind": "ridge", "alpha": 1.0},
              "budget": {"wall_seconds": 60, "max_rss_mb": 1024, "max_train_samples": 10000}}
    write_json(destination / "config.json", config)
    return destination / "config.json"
