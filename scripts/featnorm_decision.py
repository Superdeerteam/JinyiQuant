"""Turn a feature-transform replay run into the pre-registered decision.

Reads ``windows/<window>/<version>/daily.csv`` written by ``replay_model_versions.py``,
forms the candidate-minus-control delta paired by trading day inside each seed, averages
the three seeds, and reports the three registered criteria:

1. pooled mean RankIC delta >= the registered improvement threshold,
2. the 5-trading-day moving-block bootstrap 95% interval excludes zero,
3. at least four of the five declared windows have a positive mean delta.

Nothing here re-derives evaluation: it only differences two stored daily series.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_blend_replay import block_interval  # noqa: E402  (house bootstrap)

SEEDS = (11, 29, 47)
POINTS = (("132", "data-1.32"), ("264", "data-2.64"))
CANDIDATE = "p-s{seed}-{point}"
CONTROL = "c-s{seed}-{point}"
WINDOWS = ("nov2023", "jan2024", "mar2024", "apr2024", "jun2024")
DAILY_COLUMNS = ("rank_ic", "ic", "long_short_return", "turnover")


def read_daily(run_dir: Path, window: str, version: str, extra_dirs=()) -> pd.DataFrame:
    candidates = [run_dir, *extra_dirs]
    for base in candidates:
        path = base / "windows" / window / version / "daily.csv"
        if path.exists():
            break
    else:
        raise FileNotFoundError(candidates[0] / "windows" / window / version / "daily.csv")
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date").sort_index()


def seed_delta(run_dir: Path, window: str, point: str, column: str,
               candidate_template: str = CANDIDATE, control_template: str = CONTROL,
               seeds: tuple = SEEDS, extra_dirs=()) -> pd.Series:
    """candidate minus control on the same trading days, one seed at a time."""
    out = []
    for seed in seeds:
        candidate = read_daily(run_dir, window, candidate_template.format(seed=seed, point=point), extra_dirs)[column]
        control = read_daily(run_dir, window, control_template.format(seed=seed, point=point), extra_dirs)[column]
        joined = pd.concat({"candidate": candidate, "control": control}, axis=1).dropna()
        out.append((joined["candidate"] - joined["control"]).rename(seed))
    return pd.concat(out, axis=1).mean(axis=1)


def block_bootstrap(values: np.ndarray, block: int, draws: int, seed: int) -> dict:
    """House 5-day moving-block bootstrap, shared with every other paired comparison."""
    interval = block_interval(values, block=block, draws=draws, seed=seed)
    return dict(interval, draws=draws, block_days=block)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--prereg", type=Path,
                        default=Path("configs/b4_feature_norm_prereg_20261028.json"),
                        help="registered decision contract; thresholds come from this file")
    parser.add_argument("--min-improvement", type=float, default=None,
                        help="override the registered threshold (e.g. 0.0 for a sign rule)")
    parser.add_argument("--min-positive-windows", type=int, default=None)
    parser.add_argument("--block-days", type=int, default=5)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261028)
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                        help="comma separated seed list")
    parser.add_argument("--candidate-template", default=CANDIDATE,
                        help="version name template; {seed} and {point} are filled in")
    parser.add_argument("--control-template", default=CONTROL)
    parser.add_argument("--extra-run", type=Path, action="append", default=[],
                        help="additional run dir searched when a version is missing (repeatable)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    seeds = tuple(int(part) for part in str(args.seeds).split(",") if part.strip())

    prereg = json.loads(args.prereg.read_text(encoding="utf-8")) if args.prereg.exists() else {}
    if args.min_improvement is not None:
        threshold = float(args.min_improvement)
    elif prereg:
        threshold = float(prereg["thresholds"]["min_improvement"])
    else:
        raise SystemExit("no --prereg and no --min-improvement: refusing to invent a threshold")
    if args.min_positive_windows is not None:
        min_positive = int(args.min_positive_windows)
    elif prereg:
        min_positive = int(prereg["thresholds"]["min_positive_windows"])
    else:
        min_positive = 0

    report = {"run_dir": str(args.run_dir), "prereg": str(args.prereg), "seeds": list(seeds),
              "threshold": threshold, "min_positive_windows": min_positive, "points": {}}
    for point, label in POINTS:
        windows, pooled = {}, []
        for window in WINDOWS:
            delta = seed_delta(args.run_dir, window, point, "rank_ic",
                               args.candidate_template, args.control_template, seeds,
                               tuple(args.extra_run))
            windows[window] = {"days": int(len(delta)), "mean": float(delta.mean())}
            pooled.append(delta)
        series = pd.concat(pooled).sort_index()
        stats = block_bootstrap(series.to_numpy(), args.block_days, args.draws, args.bootstrap_seed)
        positive = sum(1 for w in windows.values() if w["mean"] > 0)
        criteria = {
            "improvement": stats["mean"] >= threshold,
            "interval_excludes_zero": stats["ci_low"] > 0.0,
            "positive_windows": positive >= min_positive,
        }
        report["points"][label] = {
            "windows": windows, "pooled": stats, "positive_windows": positive,
            "criteria": criteria, "passed": all(criteria.values()),
        }

    primary = report["points"]["data-1.32"]
    report["verdict"] = "adopt" if primary["passed"] else "stop"
    report["verdict_basis"] = "data-1.32 is the registered primary point"
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
