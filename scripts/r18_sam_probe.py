"""R18: SAM optimisation probe - score, pair and account the two budget-matched arms.

Both arms are trained by ``scripts/run_training_plan.py`` from the frozen MLPLOB 441d CSZ recipe
(``configs/r18_{adam,sam}_441d_seed11.json``): identical architecture, data, targets, seed and
learning rate, identical 73,000 training iterations. The only difference is the optimiser - plain
Adam for the control, single-step SAM (rho fixed ex ante at 0.05) for the treatment, injected with
``tmp/r18/r18_sam.py`` through a ``sitecustomize`` patch so the treatment costs the same
forward/backward budget and halves the number of updates.

This module reuses the row-level extraction and evaluation machinery of
``scripts/probe_r15_r16_heads.py``: each arm's trunk scores for the eight screening windows are
extracted by that script's ``extract`` stage into ``cache/<arm>/<window>`` and turned into daily and
pooled metrics here.
"""

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_probe():
    spec = importlib.util.spec_from_file_location("probe_r15_r16_heads",
                                                  ROOT / "scripts" / "probe_r15_r16_heads.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = _load_probe()
WINDOWS = P.WINDOWS
EVALUATION = P.EVALUATION
ARMS = ("adam", "sam")
DEFAULT_CACHE = ROOT / "runs" / "r18-sam-20261028" / "cache"
DEFAULT_OUT = ROOT / "artifacts" / "r18-sam-20261028"
RHO = 0.05
LOOP_STEPS = 73000
BATCH_SIZE = 32
# Fixed before any score was read: the treatment is worth a follow-up only when the pooled paired
# rank-IC interval excludes zero and long/short and turnover do not degrade beyond the gate the
# evaluator applies to candidate blends (0.0005 rank-IC is a blend gate, not a solo-arm gate).
PREREG = {
    "schema_version": 1,
    "experiment_id": "r18-sam-probe-20261028",
    "hypothesis": ("SAM's sharpness-aware update should reach a flatter region of the same frozen "
                   "trunk recipe; if the flatness buys generalisation, the treatment must beat the "
                   "pure-Adam control at matched forward/backward budget on the eight screening "
                   "windows, not merely differ from it."),
    "primary_metric": "pooled daily rank_ic over the 8 screening windows",
    "decision_rule": {
        "increment": "block-bootstrap interval of the paired daily rank_ic difference excludes 0 and "
                     "the point estimate is positive",
        "cost_gate": "wall seconds per arm reported; the treatment must not need more "
                     "forward/backward passes than the control",
        "side_effects": "long_short_return_mean and turnover_mean deltas reported with intervals; a "
                        "rank-IC win that costs turnover above the control by more than 10% is a "
                        "partial result, not a promotion",
    },
    "frozen_choices": {
        "rho": RHO, "perturbation_steps": 1, "batch_size": BATCH_SIZE, "loop_steps_per_arm": LOOP_STEPS,
        "forward_backward_per_arm": LOOP_STEPS, "updates": {"adam": LOOP_STEPS, "sam": LOOP_STEPS // 2},
        "seed": 11, "learning_rate": 1e-4, "windows": list(WINDOWS),
        "reference_arm": ("frozen production trunk scaling-441d-seed11 @1.32 (artifacts/"
                          "r15r16-20261028/per_window.csv s_base) reported as context only"),
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------------------- score stage
def score_stage(args):
    from jinyiquant.evaluation import evaluate

    out = Path(args.out)
    for arm in ARMS:
        rows = []
        for unit in WINDOWS:
            cache_dir = Path(args.cache) / arm / unit
            if not (cache_dir / "meta.json").exists():
                raise FileNotFoundError(f"{arm}/{unit}: extraction cache missing at {cache_dir}")
            cached = P.read_cache(cache_dir)
            scale = float(cached.meta["target_scale"])
            keys = cached.keys.reset_index(drop=True)
            targets = keys.assign(target=cached.target).dropna(subset=["target"])
            scored = (keys.assign(score=cached.base * scale).loc[cached.valid]
                      .merge(targets[["date", "instrument"]], on=["date", "instrument"], how="inner"))
            metrics, daily, _ = evaluate(scored, targets, EVALUATION, return_sections=True)
            window_dir = out / "windows" / unit
            window_dir.mkdir(parents=True, exist_ok=True)
            scored.to_parquet(window_dir / f"{arm}.parquet", index=False)
            daily.to_csv(window_dir / f"{arm}.daily.csv", index=False)
            daily.assign(arm=arm, window=unit).to_csv(out / "daily" / f"{arm}_{unit}.csv", index=False) \
                if (out / "daily").mkdir(parents=True, exist_ok=True) is None else None
            metrics.update(window=unit, arm=arm, rows=int(len(scored)),
                           weights_sha256=cached.meta["weights_sha256"],
                           weights=str(cached.meta["weights"]))
            rows.append(metrics)
            P.log(f"{arm} {unit}: rank_ic {P.fmt(metrics['rank_ic_mean'])} "
                  f"long_short {P.fmt(metrics['long_short_return_mean'])} "
                  f"turnover {P.fmt(metrics['turnover_mean'], 4)}")
        frame = pd.DataFrame(rows)
        frame.to_csv(out / f"metrics_{arm}.csv", index=False)
        P.log(f"{arm}: pooled rank_ic {P.fmt(frame.rank_ic_mean.mean())} over {len(frame)} windows")
    return 0


# ------------------------------------------------------------------------------- cost stage
def cost_stage(args):
    out = Path(args.out)
    rows = []
    for arm in ARMS:
        run_dirs = sorted(Path(args.runs).glob("*/manifest.json"))
        matches = []
        for manifest_path in run_dirs:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("experiment_id") == f"r18-{arm}-441d-seed11":
                matches.append(manifest_path.parent)
        if len(matches) != 1:
            raise ValueError(f"{arm}: expected exactly one run dir, found {matches}")
        run_dir = matches[0]
        events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
                  if line.strip()]
        progress = [event for event in events if event.get("event_type") == "progress"]
        ended = [event for event in events if event.get("event_type") == "ended"]
        final = progress[-1]["context"] if progress else {}
        started = [event for event in events if event.get("event_type") == "started"
                   and event.get("stage") == "training"]
        wall = ended[-1]["context"]["resources"]["wall_seconds"] if ended else None
        steps = int(final.get("steps", 0))
        samples = int(final.get("samples_seen", 0))
        training_seconds = None
        if started and progress:
            training_seconds = (pd.Timestamp(progress[-1]["timestamp_utc"]) -
                                pd.Timestamp(started[-1]["timestamp_utc"])).total_seconds()
        rows.append({
            "arm": arm, "run_dir": str(run_dir),
            "config_sha256": (run_dir / "config.sha256").read_text(encoding="utf-8").strip(),
            "steps": steps, "samples_seen": samples,
            "forward_backward_passes": steps,
            "updates": steps if arm == "adam" else steps // 2,
            "wall_seconds": wall, "training_seconds": training_seconds,
            "milliseconds_per_iteration": (1000 * training_seconds / steps) if training_seconds and steps else None,
            "final_loss": final.get("loss"), "learning_rate": final.get("learning_rate"),
            "checkpoint": str(run_dir / "model.pt"), "checkpoint_sha256": sha256(run_dir / "model.pt"),
            "events_sha256": sha256(run_dir / "events.jsonl"),
        })
        if arm == "sam":
            rho_recorded = [event for event in events if event.get("event_type") == "started"]
            rows[-1]["rho"] = RHO
            rows[-1]["sam_marker_events"] = len(rho_recorded) > 0
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "cost.csv", index=False)
    P.log("cost: " + "; ".join(f"{row['arm']} steps={row['steps']} updates={row['updates']} "
                               f"wall={row['wall_seconds']}" for row in rows))
    return rows


# ------------------------------------------------------------------------------- report stage
def report_stage(args):
    out = Path(args.out)
    pooled = []
    per_window = []
    daily = {}
    for arm in ARMS:
        frame = pd.read_csv(out / f"metrics_{arm}.csv")
        pooled.append({
            "arm": arm, "windows": len(frame),
            "rank_ic_mean": float(frame.rank_ic_mean.mean()),
            "rank_ic_std_across_windows": float(frame.rank_ic_mean.std(ddof=1)),
            "long_short_return_mean": float(frame.long_short_return_mean.mean()),
            "turnover_mean": float(frame.turnover_mean.mean()),
            "rank_ic_min": float(frame.rank_ic_mean.min()), "rank_ic_max": float(frame.rank_ic_mean.max()),
        })
        per_window.append(frame.assign(arm=arm))
        series = {}
        for unit in WINDOWS:
            path = out / "windows" / unit / f"{arm}.daily.csv"
            series[unit] = pd.read_csv(path)
        daily[arm] = series
    pooled_frame = pd.DataFrame(pooled)
    per_window_frame = pd.concat(per_window, ignore_index=True)
    pooled_frame.to_csv(out / "pooled.csv", index=False)
    per_window_frame.to_csv(out / "per_window.csv", index=False)

    def pooled_daily(arm):
        values = []
        for unit in WINDOWS:
            frame = daily[arm][unit]
            values.append(frame.assign(window=unit))
        return pd.concat(values, ignore_index=True)

    result = {"pooled": pooled_frame.to_dict(orient="records"),
              "per_window": per_window_frame.to_dict(orient="records"), "paired": [], "per_window_delta": []}
    left, right = pooled_daily("sam"), pooled_daily("adam")
    for value in ("rank_ic", "long_short_return", "turnover"):
        paired = left[["date", "window", value]].merge(
            right[["date", "window", value]], on=["date", "window"], suffixes=("_sam", "_adam"))
        delta = (paired[f"{value}_sam"] - paired[f"{value}_adam"]).to_numpy(dtype=float)
        interval = P.block_interval(delta)
        result["paired"].append({
            "value": value, "days": interval["days"],
            "mean_sam_minus_adam": interval["mean"],
            "ci_low": interval["ci_low"], "ci_high": interval["ci_high"],
            "sam_days_above_adam": int((delta > 0).sum()),
        })
        P.log(f"paired {value}: sam-adam {P.fmt(interval['mean'])} "
              f"[{P.fmt(interval['ci_low'])},{P.fmt(interval['ci_high'])}] "
              f"days_won {int((delta > 0).sum())}/{interval['days']}")
    # Per-window deltas: same five-window style pairing the evaluator uses for blends.
    for unit in WINDOWS:
        sam_daily = daily["sam"][unit].set_index("date")
        adam_daily = daily["adam"][unit].set_index("date")
        joined = sam_daily[["rank_ic"]].join(adam_daily[["rank_ic"]], lsuffix="_sam", rsuffix="_adam",
                                             how="inner")
        delta = (joined.rank_ic_sam - joined.rank_ic_adam).dropna()
        result["per_window_delta"].append({
            "window": unit, "days": int(len(delta)),
            "mean_delta": float(delta.mean()) if len(delta) else None,
            "days_sam_wins": int((delta > 0).sum()),
        })
    pd.DataFrame(result["paired"]).to_csv(out / "paired.csv", index=False)
    pd.DataFrame(result["per_window_delta"]).to_csv(out / "per_window_delta.csv", index=False)
    result["prereg"] = PREREG
    cost = pd.read_csv(out / "cost.csv")
    result["cost"] = cost.to_dict(orient="records")
    P.write_json_safe(out / "summary.json", result)
    return 0


# ------------------------------------------------------------------------------- selfcheck
def selfcheck_stage(args):
    out = Path(args.out)
    prereg_path = out / "prereg.json"
    P.write_json_safe(prereg_path, PREREG)
    checks = {"prereg_written": str(prereg_path), "arms": {}}
    for arm in ARMS:
        cache_meta = [json.loads((Path(args.cache) / arm / unit / "meta.json").read_text())
                      for unit in WINDOWS]
        hashes = {meta["weights_sha256"] for meta in cache_meta}
        checks["arms"][arm] = {
            "windows": len(cache_meta),
            "weights": cache_meta[0]["weights"], "weights_sha256": sorted(hashes),
            "weights_sha256_consistent": len(hashes) == 1,
            "rows": int(sum(meta["rows"] for meta in cache_meta)),
            "sealed_clear": all(pd.Timestamp(meta["end"]) <= pd.Timestamp("2024-09-01")
                                for meta in cache_meta),
        }
    ad = pd.read_csv(out / "metrics_adam.csv").set_index("window")
    sa = pd.read_csv(out / "metrics_sam.csv").set_index("window")
    if list(ad.index) != list(sa.index):
        raise ValueError("arm metrics cover different windows")
    checks["same_windows_rows"] = bool((ad.rows == sa.rows).all())
    checks["window_rows"] = {unit: int(ad.loc[unit, "rows"]) for unit in ad.index}
    checks["prereg_frozen_choices"] = PREREG["frozen_choices"]
    weight_dir = out / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        source = Path(checks["arms"][arm]["weights"])
        target = weight_dir / f"{arm}.pt"
        shutil.copy2(source, target)
        checks["arms"][arm]["copied"] = str(target)
        checks["arms"][arm]["copied_sha256"] = sha256(target)
    P.write_json_safe(out / "selfcheck.json", checks)
    P.log("selfcheck: " + json.dumps({arm: {k: v for k, v in checks["arms"][arm].items()
                                            if k in ("weights_sha256_consistent", "sealed_clear", "rows")}
                                     for arm in ARMS}))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("score", "cost", "report", "selfcheck"))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--runs", default="/mnt/work/r18/runs")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {"score": score_stage, "cost": cost_stage, "report": report_stage,
            "selfcheck": selfcheck_stage}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
