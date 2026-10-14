"""Run only uploaded code in an isolated process, using a label-denying local DAI double."""

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def verify(package, fixture):
    import numpy as np
    import pandas as pd

    package, fixture = Path(package).resolve(), Path(fixture).resolve()
    sys.path.insert(0, str(package))
    started = time.monotonic()
    queries = []
    def no_network(*args, **kwargs):
        raise RuntimeError("External network disabled by package smoke test")
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    socket.create_connection = no_network
    bars = pd.read_parquet(fixture / "bars.parquet")
    pool = pd.read_parquet(fixture / "instruments.parquet")
    def query(sql, *, filters):
        queries.append({"sql": sql, "filters": filters})
        if any(term in sql.lower() for term in ("vwap", "target", "label", "end_price")):
            raise AssertionError("Inference queried future answers")
        frame = pool if "instruments" in sql else bars
        start, end = map(pd.Timestamp, filters["date"])
        columns = [s.strip() for s in sql.split("FROM")[0].removeprefix("SELECT ").split(",")]
        result = frame.loc[(frame.date >= start) & (frame.date <= end), columns].copy()
        return SimpleNamespace(df=lambda: result)
    sys.modules["dai"] = SimpleNamespace(query=query)
    notebook = json.loads((package / "inference.ipynb").read_text(encoding="utf-8"))
    scope = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "inference.ipynb", "exec"), scope)
    cfg = json.loads((package / "training_config.json").read_text(encoding="utf-8"))
    start = cfg["splits"]["validation"]["start"]
    end = str(pd.Timestamp(cfg["splits"]["validation"]["end"]) - pd.Timedelta(nanoseconds=1))
    result = scope["main"]({"bar1m": "cpt_jyc_2026_stock_bar1m"}, start, end)
    import jinyiquant.data as data
    import modified_tlob.submission as submission
    assert Path(submission.__file__).is_relative_to(package)
    assert Path(data.__file__).is_relative_to(package)
    assert list(result) == ["date", "instrument", "factor"]
    assert np.isfinite(result.factor).all()
    expected = pd.read_parquet(fixture / "expected_predictions.parquet")
    assert result[["date", "instrument"]].equals(expected[["date", "instrument"]])
    atol, rtol = cfg["acceptance"]["repro_atol"], cfg["acceptance"]["repro_rtol"]
    np.testing.assert_allclose(result.factor, expected.score, atol=atol, rtol=rtol)
    # Perturb only later market observations and call main for the earlier date.
    cutoff = result.date.min()
    earlier = scope["main"]({"bar1m": "cpt_jyc_2026_stock_bar1m"}, start, str(cutoff))
    features = cfg["data"]["feature_columns"]
    bars.loc[bars.date > cutoff, features] = 1e9
    later_changed = scope["main"]({"bar1m": "cpt_jyc_2026_stock_bar1m"}, start, str(cutoff))
    np.testing.assert_allclose(later_changed.factor, earlier.factor, atol=atol, rtol=rtol)
    # Separately exercise the explicit private retraining entry. It has its own
    # query double and can consult labels only in the frozen development splits.
    from modified_tlob.train import train_and_save
    labels = pd.read_parquet(fixture / "labels.parquet")
    original_bars = pd.read_parquet(fixture / "bars.parquet")
    training_queries = []
    def training_query(sql, *, filters):
        training_queries.append({"sql": sql, "filters": filters})
        begin, finish = map(pd.Timestamp, filters["date"])
        assert finish <= pd.Timestamp(cfg["splits"]["validation"]["end"])
        frame = labels if "vwap" in sql else pool if "instruments" in sql else original_bars
        columns = [s.strip() for s in sql.split("FROM")[0].removeprefix("SELECT ").split(",")]
        result = frame.loc[(frame.date >= begin) & (frame.date <= finish), columns].copy()
        return SimpleNamespace(df=lambda: result)
    new_weights = train_and_save({"bar1m": "cpt_jyc_2026_stock_bar1m"}, query=training_query)
    assert new_weights.is_file() and new_weights.is_relative_to(package)
    from jinyiquant.data import build_batch, preprocess_bars
    from modified_tlob.adapter import TLOBAdapter
    retrained = TLOBAdapter.load(new_weights, device="cpu")
    prepared = preprocess_bars(original_bars, retrained.preprocessing)
    np.testing.assert_allclose(retrained.predict(build_batch(prepared, expected[["date", "instrument"]],
                              retrained.preprocessing)), expected.score, atol=atol, rtol=rtol)
    return {"schema_version": 1, "passed": True, "rows": len(result), "queries": queries,
            "private_training_queries": training_queries, "private_training_entry": "passed_local_double",
            "max_abs_prediction_difference": float(np.max(np.abs(result.factor.to_numpy() - expected.score))),
            "atol": atol, "rtol": rtol, "wall_seconds": time.monotonic() - started,
            "network": "blocked", "repo_import_path": "absent", "notebook_executed": True,
            "platform_acceptance": "not_verified", "scope": "local synthetic DAI double only"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.package, args.fixture)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
