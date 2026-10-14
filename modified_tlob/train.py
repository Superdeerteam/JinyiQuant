"""Explicit training entry; public inference never calls this module."""

import argparse
import time
from pathlib import Path

import pandas as pd

from jinyiquant.config import load_config
from jinyiquant.io import file_sha256, write_json
from jinyiquant.runner import run
from jinyiquant.tracking import Recorder, new_run_dir

from .submission import query_frame


def train_and_save(datasources, config_path=None, *, output_root=None, query=None):
    """Private-platform retraining on frozen development splits, with normal archives."""
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(config_path or root / "training_config.json")
    if cfg.model.kind != "modified_tlob" or cfg.evaluation_split != "validation":
        raise ValueError("Platform training requires modified_tlob and development validation only")
    if query is None:
        import dai
        query = dai.query
    archive = new_run_dir(Path(output_root) if output_root else root / "training_runs")
    rec = Recorder(archive, cfg.experiment_id)
    started = time.monotonic()
    rec.event("run", "started", config_sha256=cfg.checksum, purpose="frozen_platform_training_data_fetch")
    start = max(pd.Timestamp(cfg.data.allowed_start), pd.Timestamp(cfg.splits.train.start) -
                pd.Timedelta(days=cfg.data.history_days))
    end = cfg.splits.validation.end
    try:
        requests = [("bars", datasources["bar1m"], ["date", "instrument", *cfg.data.feature_columns], start),
                    ("instruments", datasources.get("instruments", "cpt_jyc_2026_instruments"),
                     ["date", "instrument"], cfg.splits.train.start),
                    ("labels", datasources.get("vwap", "cpt_jyc_2026_vwap"),
                     ["date", "instrument", "vwap_return"], cfg.splits.train.start)]
        for role, table, columns, first in requests:
            rec.event("data", "loading", role=role, source=table, start=str(first), end=str(end))
            path = archive / f"{role}.parquet"
            query_frame(query, table, columns, first, end).to_parquet(path, index=False)
            rec.event("data", "fetched", role=role, sha256=file_sha256(path))
            if time.monotonic() - started > cfg.budget.wall_seconds:
                raise TimeoutError("Platform training preparation budget exceeded")
        raw = cfg.snapshot()
        raw["data"].update(bars_path="bars.parquet", labels_path="labels.parquet",
                           instruments_path="instruments.parquet", instrument_map_path=None, cache_dir=None,
                           price_scale=1, source="official_DAI_frozen_development_splits")
        raw["output_root"] = "runs"
        raw["budget"]["wall_seconds"] -= time.monotonic() - started
        write_json(archive / "config.json", raw)
        run_dir, code = run(load_config(archive / "config.json"))
        if code:
            raise RuntimeError(f"Training failed ({code}); archive: {run_dir}")
        rec.event("run", "ended", status="completed", training_run_id=run_dir.name,
                  wall_seconds=time.monotonic() - started)
        return run_dir / "modified_tlob_weights.json"
    except BaseException as exc:
        rec.failure("platform_training", exc)
        rec.event("run", "ended", status="failed", wall_seconds=time.monotonic() - started)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    path, code = run(load_config(args.config))
    print(path)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
