"""R15/R16 small-head probes on the frozen 441d seed11 @1.32 MLPLOB representation.

R15 hypothesis: the same 128-minute book window means different things for a high- versus
low-volatility, wide- versus narrow-spread, active- versus quiet stock; the modulation is
visible only in per-stock state that the trunk's per-sample normalisation removes. The gate
reads point-in-time state only.

R16 hypothesis: mean regression ignores tails and skew; predicting three fixed quantiles of the
same 30-minute label carries that information. The output rule is fixed BEFORE any result is
read (see ``SCORE_RULES``), and quantiles are never combined into an expected return.

Both families are probes on a frozen trunk: the trunk is loaded read-only, its 80-d
representation ``h`` and raw score ``s_base`` are cached once per unit, and every head is trained
on that cache with a parameter-matched control.

Stages (each resumable, each writing machine-readable files):

``extract``  Cache ``h``, ``s_base``, the official label and the valid mask for one unit
             (``train`` or one development window).
``stats``    Point-in-time per-stock state (trailing 20 trading days of *already realised* label
             rows only) for one cached unit.
``train``    Train every head on the ``train`` cache: per-family control plus treatment.
``score``    Score all heads on one cached development window.
``report``   Per-window metrics, paired daily deltas with a moving-block bootstrap, and the
             summary the captain's gate reads.

The representation and window machinery is imported from ``scripts/market_context.py`` (the
audited frozen-trunk cache) instead of being re-implemented here.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy import special

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from jinyiquant.config import load_config  # noqa: E402
from jinyiquant.data import load_data, select_targets  # noqa: E402
from jinyiquant.io import write_json  # noqa: E402


def fmt(value, digits=5):
    return "nan" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def json_safe(value):
    """write_json refuses NaN/Inf; metric frames carry NaN on the first rebalance."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    return value


def write_json_safe(path, value):
    write_json(path, json_safe(value))
from jinyiquant.pool import training_pool  # noqa: E402
from jinyiquant.runner import inference_keys  # noqa: E402
from modified_tlob.adapter import TLOBAdapter  # noqa: E402

_MC_SPEC = importlib.util.spec_from_file_location("market_context", ROOT / "scripts" / "market_context.py")
market_context = importlib.util.module_from_spec(_MC_SPEC)
_MC_SPEC.loader.exec_module(market_context)
trunk_forward = market_context.trunk_forward
file_sha256 = market_context.file_sha256
keys_sha256 = market_context.keys_sha256

DEFAULT_CONFIG = "configs/scaling_441d_csz_seed11.json"
DEFAULT_WEIGHTS = ("runs/container-weights-20261028/old-hz4/runs/scaling-441d-seed11/"
                   "20261028T041026.425164Z_18e6e2c65c08/model-scaling-1.32.pt")
DEFAULT_CACHE = "runs/r15r16-probe-20261028/cache"
DEFAULT_OUT = "artifacts/r15r16-probe-20261028"

# The eight declared screening windows. jun2024 and sep2024 are confirmation blocks and stay out.
WINDOWS = {
    "nov2023": ("2023-11-01", "2023-12-01"),
    "jan2024": ("2024-01-01", "2024-02-01"),
    "feb2024": ("2024-02-01", "2024-03-01"),
    "mar2024": ("2024-03-01", "2024-04-01"),
    "apr2024": ("2024-04-01", "2024-05-01"),
    "may2024": ("2024-05-01", "2024-06-01"),
    "jul2024": ("2024-07-01", "2024-08-01"),
    "aug2024": ("2024-08-01", "2024-09-01"),
}
TRAIN_UNIT = ("2022-01-04", "2023-11-01")
EVALUATION = {"quantile": 0.2, "periods_per_year": 252, "min_cross_section": 3}
QUANTILES = (0.2, 0.5, 0.8)

# PREREGISTERED OUTPUT RULES (fixed before any label is scored).
#   r16: the ranking score is the predicted MEDIAN q_hat(0.5) of the 30-minute label, in trunk
#        optimisation units. q_hat(0.2)/q_hat(0.8) are reported as tail diagnostics only and are
#        never averaged into a mean or an expected return.
#   r15: the gated head's own scalar output (same units as the control).
SCORE_RULES = {
    "r16_quantile": {"score": "q_hat(0.5)", "combine_quantiles": False,
                     "interpretation": "median of the label, NOT an expected return"},
    "r15_gate": {"score": "gated scalar output", "gate_inputs": "point-in-time stock state only"},
}

ARMS = ("rep_linear", "rep_mlp", "stats_linear", "r15_concat", "r15_gate", "r16_quantile")
CONTROL_OF = {"r15_gate": "r15_concat", "r16_quantile": "rep_mlp", "r15_concat": "rep_mlp",
              "stats_linear": "rep_linear"}

STATE_NAMES = ("vol20", "logamt20", "illiq20", "logprice")
STATE_LOOKBACK_DAYS = 20


# ------------------------------------------------------------------------------------ helpers
def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def dataset(kind, *, remote):
    """One dataset root: the four paths the recipe needs, per host.

    ``primary`` is the frozen 2020-01..2023-11 dataset. ``extended`` is the GPU host's uploaded
    2023-12..2024-08 view; that mount cannot hold symlinks, so the two roots are kept separate
    and each unit names the one whose months cover it. Both share the instruments table and the
    instrument map, which never change.
    """
    if remote:
        tables = {"instruments": "/root/data/cpt_jyc_2026_instruments(1).parquet",
                  "map": "/root/data/cpt_jyc_2026_stock_bar1m_instrument_mapping.csv"}
        if kind == "primary":
            return {"bars": "/root/data/cpt_jyc_2026_e2e_bar1m",
                    "labels": "/root/data/cpt_jyc_2026_vwap", **tables}
        return {"bars": "/mnt/work/r15r16/data/bars", "labels": "/mnt/work/r15r16/data/vwap",
                **tables}
    base = ROOT / "数据"
    return {"bars": str(base / "cpt_jyc_2026_e2e_bar1m"),
            "labels": str(base / "cpt_jyc_2026_vwap"),
            "instruments": str(base / "cpt_jyc_2026_instruments(1).parquet"),
            "map": str(base / "cpt_jyc_2026_stock_bar1m_instrument_mapping.csv")}


def data_kind_for(name):
    """The dataset that covers one unit: primary for <=2023-11, extended for the 2024 windows."""
    return "primary" if name in ("train", "nov2023") else "extended"


def rooted_data(data, paths):
    return data.model_copy(update={
        "bars_path": paths["bars"], "labels_path": paths["labels"],
        "instruments_path": paths["instruments"], "instrument_map_path": paths["map"],
        "cache_dir": None})


def unit_split(start, end, limit_days=None):
    if isinstance(start, str):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
    if limit_days:
        end = min(end, start + pd.Timedelta(days=int(limit_days)))
    return SimpleNamespace(start=start, end=end)


def make_adapter(weights, device):
    adapter = TLOBAdapter.load(Path(weights), device=device)
    if adapter.config.get("kind") != "mlplob":
        raise ValueError("the probe needs the MLPLOB trunk")
    if adapter.network.representation_dim != 80:
        raise ValueError(f"trunk representation is {adapter.network.representation_dim}-d, expected 80")
    adapter.network.eval()
    return adapter


def extract_rows(adapter, pool, *, out, key, part_rows, slice_rows):
    """Representation (80-d) and raw trunk score for every pool row, checkpointed in parts."""
    total = len(pool.keys)
    hidden = np.empty((total, 80), dtype=np.float32)
    raw = np.empty(total, dtype=np.float64)
    parts = Path(out) / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for start in range(0, total, part_rows):
        stop = min(start + part_rows, total)
        path = parts / f"{key}_{start:07d}_{stop:07d}.npz"
        if path.exists():
            saved = np.load(path)
            hidden[start:stop], raw[start:stop] = saved["h"], saved["base"]
            continue
        for low in range(start, stop, 16384):
            high = min(low + 16384, stop)
            hidden[low:high], raw[low:high] = trunk_forward(
                adapter, pool.batch(np.arange(low, high)), slice_rows=slice_rows)
        np.savez(path, h=hidden[start:stop], base=raw[start:stop])
        log(f"  extracted rows {start}..{stop}/{total}")
    return hidden, raw


def read_cache(path):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    cached = SimpleNamespace(meta=meta, keys=pd.read_parquet(path / "keys.parquet"))
    for name in ("h", "base", "target", "y_opt", "valid"):
        if (path / f"{name}.npy").exists():
            setattr(cached, name, np.load(path / f"{name}.npy"))
    if (path / "state.npy").exists():
        cached.state = np.load(path / "state.npy")
    if keys_sha256(cached.keys) != meta["keys_sha256"]:
        raise ValueError(f"{path}: cached keys disagree with the manifest")
    return cached


# ------------------------------------------------------------------------------- extract stage
def extract_unit(config, data, adapter, *, name, start, end, out, weights, part_rows, slice_rows,
                 stride, limit_days=None):
    started = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    split = unit_split(start, end, limit_days)
    bundle = load_data(data, start=split.start, end=split.end, stream_bars=True)
    labels = select_targets(bundle.labels, split, config.data.label_horizon_minutes,
                           sample_times=config.data.sample_times)
    train_unit = name == "train"
    if train_unit and stride > 1:
        days = np.sort(pd.unique(labels["date"].dt.normalize()))
        keep = set(pd.to_datetime(days[::stride]))
        labels = labels.loc[labels["date"].dt.normalize().isin(keep)].reset_index(drop=True)
    # The train unit is cached on the label keys only: no cross-section summary is needed, so a
    # second pass over the universe would only duplicate rows.
    context_keys = (labels[["date", "instrument"]].reset_index(drop=True) if train_unit
                    else inference_keys(bundle, split, config.data).reset_index(drop=True))
    rows_keys = context_keys
    log(f"{name}: {len(rows_keys)} rows, {len(context_keys)} universe keys, "
        f"{rows_keys.date.dt.normalize().nunique()} days")

    union = pd.concat([rows_keys[["date", "instrument"]].assign(_rows=1),
                       context_keys[["date", "instrument"]].assign(_ctx=1)], ignore_index=True)
    union = (union.groupby(["date", "instrument"], as_index=False)
             .agg(_rows=("_rows", "sum"), _ctx=("_ctx", "sum"))
             .sort_values(["date", "instrument"], kind="stable").reset_index(drop=True))
    pool = training_pool(None, union[["date", "instrument"]], config.data, streaming=True,
                         arrays=bundle.arrays)
    usable = np.asarray(pool.effective_count) >= int(config.data.min_history)
    log(f"{name}: pool {len(union)} rows ({int(usable.sum())} usable)")
    hidden, raw = extract_rows(adapter, pool, out=out, key=name, part_rows=part_rows,
                               slice_rows=slice_rows)
    del bundle, pool
    gc.collect()

    position = union[["date", "instrument"]].assign(_pos=np.arange(len(union)))
    mapping = rows_keys[["date", "instrument"]].merge(position, on=["date", "instrument"],
                                                      how="left", validate="one_to_one")
    if mapping._pos.isna().any():
        raise ValueError("cache rows missing from the extraction pool")
    row_pos = mapping._pos.to_numpy(dtype=np.int64)
    keys = rows_keys
    aligned = keys.merge(labels, on=["date", "instrument"], how="left", validate="one_to_one")
    target = aligned.target.to_numpy(dtype=np.float64)
    hidden_rows = hidden[row_pos]
    raw_rows = raw[row_pos]
    valid = usable[row_pos]
    del hidden, raw, usable, aligned, labels, union, position, mapping
    gc.collect()

    if train_unit:
        rows = np.flatnonzero(valid)
        if not np.isfinite(target[rows]).all():
            raise ValueError("training rows must all carry a finite label")
        frozen = copy.deepcopy(adapter.target_centering_stats)
        optimization, stats = adapter._training_targets(keys, target, np.ones(len(keys), bool))
        # A strided smoke window necessarily refits centering, so there the comparison is skipped.
        worst = None if stride > 1 else market_context.compare_stats(stats, frozen)
        adapter.target_centering_stats = frozen
        if worst is not None and worst > 1e-12:
            raise ValueError(f"target centering moved by {worst:g}: the cache does not reproduce "
                             f"the frozen labels")
        y_opt = np.full(len(keys), np.nan, dtype=np.float32)
        y_opt[rows] = optimization[rows] / float(adapter.target_scale)
        hidden_rows, raw_rows, target, valid = hidden_rows[rows], raw_rows[rows], target[rows], valid[rows]
        log(f"{name}: target-centering self check "
            f"{'skipped (stride>1)' if worst is None else f'max_abs_diff={worst:g}'}")
    else:
        y_opt = np.full(len(keys), np.nan, dtype=np.float32)

    np.save(out / "h.npy", hidden_rows)
    np.save(out / "base.npy", raw_rows)
    np.save(out / "target.npy", target)
    np.save(out / "y_opt.npy", y_opt)
    np.save(out / "valid.npy", valid)
    keys.reset_index(drop=True).to_parquet(out / "keys.parquet", index=False)
    meta = {"name": name, "start": str(split.start), "end": str(split.end),
            "rows": int(len(keys)), "keys_sha256": keys_sha256(keys),
            "days": int(keys.date.dt.normalize().nunique()),
            "timestamps": int(keys.date.nunique()),
            "weights": str(weights), "weights_sha256": file_sha256(weights),
            "target_scale": float(adapter.target_scale),
            "train_stride": int(stride), "valid_rows": int(valid.sum()),
            "labels_finite": int(np.isfinite(target).sum()),
            "h_sha256": hashlib.sha256(np.ascontiguousarray(hidden_rows).tobytes()).hexdigest(),
            "seconds": round(time.monotonic() - started, 1)}
    write_json(out / "meta.json", meta)
    write_json(Path(out).parent / f"{name}.json", meta)
    for part in sorted((Path(out) / "parts").glob("*.npz")):
        part.unlink()
    (Path(out) / "parts").rmdir()
    log(f"{name}: cached {meta['rows']} rows ({meta['valid_rows']} valid) in {meta['seconds']}s")
    return meta


# --------------------------------------------------------------------------------- stats stage
def label_history(labels_dirs, start, end, horizon_days):
    """Daily per-stock aggregates from the official label table, one monthly file at a time.

    Only rows strictly *before* a sample are ever used by :func:`state_features`, so a row whose
    amount/volume describe its own forward interval still enters a later day's state after that
    interval is realised.
    """
    files = sorted({path for directory in labels_dirs for path in Path(directory).glob("*.parquet")})
    first = pd.Timestamp(start) - pd.Timedelta(days=int(horizon_days * 2.2) + 20)
    frames = []
    for path in files:
        stem = path.stem
        month = pd.Timestamp(stem[-7:] + "-01")
        if month + pd.offsets.MonthBegin(1) <= first or month >= pd.Timestamp(end) + pd.Timedelta(days=40):
            continue
        frame = pd.read_parquet(path, columns=["date", "instrument", "vwap_return", "vwap",
                                               "amount", "num_trades"])
        frames.append(frame)
    if not frames:
        raise ValueError("no label history found for the requested range")
    history = pd.concat(frames, ignore_index=True)
    history["day"] = pd.to_datetime(history["date"]).dt.normalize()
    history = history.dropna(subset=["vwap_return"])
    history["absret"] = history["vwap_return"].abs()
    daily = (history.groupby(["instrument", "day"], sort=True)
             .agg(vol=("vwap_return", "std"), absret=("absret", "mean"),
                  amount=("amount", "mean"), num_trades=("num_trades", "mean"),
                  vwap=("vwap", "last"))
             .reset_index())
    daily["illiq"] = daily["absret"] / (daily["amount"].clip(lower=1.0) / 1e8)
    return daily


def state_features(daily, keys, *, lookback=STATE_LOOKBACK_DAYS):
    """Trailing state per row, then a per-date cross-sectional z-score of each component."""
    frame = daily.sort_values(["instrument", "day"], kind="stable")
    rolled = frame.groupby("instrument", sort=True)[["vol", "amount", "illiq"]].transform(
        lambda s: s.rolling(lookback, min_periods=max(5, lookback // 2)).mean().shift(1))
    rolled["logprice"] = np.log(frame.groupby("instrument", sort=True)["vwap"]
                                .transform(lambda s: s.ffill().shift(1)).clip(lower=0.01))
    rolled["day"] = frame["day"]
    rolled["instrument"] = frame["instrument"]
    values = pd.DataFrame({"vol20": rolled["vol"],
                           "logamt20": np.log1p(rolled["amount"].clip(lower=0.0)),
                           "illiq20": np.log1p(rolled["illiq"].clip(lower=0.0) * 1e4),
                           "logprice": rolled["logprice"],
                           "day": rolled["day"], "instrument": rolled["instrument"]})
    query = keys[["date", "instrument"]].copy()
    query["day"] = pd.to_datetime(query["date"]).dt.normalize()
    merged = query.merge(values, on=["day", "instrument"], how="left", validate="many_to_one")
    state = merged[list(STATE_NAMES)].to_numpy(dtype=np.float64)
    # Cross-sectional standardisation per sample date keeps the state comparable over time; rows
    # without enough history stay NaN and are imputed to the cross-sectional mean (zero).
    dates = merged["day"].to_numpy()
    for column in range(state.shape[1]):
        series = pd.Series(state[:, column])
        grouped = series.groupby(dates)
        mean = grouped.transform("mean")
        std = grouped.transform("std")
        z = ((series - mean) / std.replace(0.0, np.nan)).to_numpy()
        state[:, column] = z
    state = np.clip(np.nan_to_num(state, nan=0.0), -3.0, 3.0).astype(np.float32)
    coverage = float(np.isfinite(merged["vol20"].to_numpy(dtype=np.float64)).mean())
    return state, coverage


def stats_stage(args):
    torch.set_num_threads(1)
    config = load_config(Path(args.config))
    if args.labels_dirs:
        labels_dirs = [Path(part) for part in args.labels_dirs.split(",")]
    else:
        labels_dirs = sorted({Path(dataset("primary", remote=args.remote_data)["labels"]),
                              Path(dataset("extended", remote=args.remote_data)["labels"])})
    log(f"stats: label history from {[str(path) for path in labels_dirs]}")
    names = [args.unit] if args.unit else ["train", *WINDOWS]
    summary = {}
    for name in names:
        path = Path(args.cache) / name
        cached = read_cache(path)
        start, end = (TRAIN_UNIT if name == "train" else WINDOWS[name])
        split = unit_split(start, end, args.limit_days)
        daily = label_history(labels_dirs, split.start, split.end, STATE_LOOKBACK_DAYS)
        state, coverage = state_features(daily, cached.keys)
        np.save(path / "state.npy", state)
        meta = dict(cached.meta)
        meta["state_names"] = list(STATE_NAMES)
        meta["state_lookback_days"] = STATE_LOOKBACK_DAYS
        meta["state_coverage"] = coverage
        write_json(path / "meta.json", meta)
        summary[name] = {"rows": int(len(state)), "coverage": coverage,
                         "mean_abs": {n: float(np.abs(state[:, i]).mean())
                                      for i, n in enumerate(STATE_NAMES)}}
        log(f"{name}: state {state.shape} coverage={coverage:.4f}")
        del cached, daily
        gc.collect()
    write_json(Path(args.out) / "state.json", summary)
    return summary


# --------------------------------------------------------------------------------- head models
class RepLinear(torch.nn.Module):
    """Reference: one linear score on the frozen representation."""

    def __init__(self, width=80):
        super().__init__()
        self.out = torch.nn.Linear(width, 1)

    def forward(self, h, s):
        return self.out(h).squeeze(-1)


class RepMLP(torch.nn.Module):
    """Control backbone: ``h -> relu(hidden) -> score``."""

    def __init__(self, width=80, hidden=64):
        super().__init__()
        self.body = torch.nn.Linear(width, hidden)
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, h, s):
        return self.out(torch.relu(self.body(h))).squeeze(-1)


class StatsLinear(torch.nn.Module):
    """Statistics bypass head: point-in-time stock state only, no representation."""

    def __init__(self, state_dim):
        super().__init__()
        self.out = torch.nn.Linear(state_dim, 1)

    def forward(self, h, s):
        return self.out(s).squeeze(-1)


class ConcatMLP(torch.nn.Module):
    """R15 control: representation and state enter the same hidden layer (no gating)."""

    def __init__(self, width, state_dim, hidden=64):
        super().__init__()
        self.body = torch.nn.Linear(width + state_dim, hidden)
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, h, s):
        return self.out(torch.relu(self.body(torch.cat([h, s], dim=-1)))).squeeze(-1)


class GateHead(torch.nn.Module):
    """R15 treatment: the state gates the representation's hidden reading.

    ``g = sigmoid(W_s s + b_s)`` is a per-unit gate over the 64-d reading of ``h``; the state
    never enters the score additively, so the only way state can act is by choosing how the same
    book window is read.
    """

    def __init__(self, width, state_dim, hidden=64):
        super().__init__()
        self.body = torch.nn.Linear(width, hidden)
        self.gate = torch.nn.Linear(state_dim, hidden)
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, h, s):
        z = torch.relu(self.body(h))
        g = torch.sigmoid(self.gate(s))
        return self.out(g * z).squeeze(-1)

    def gate_profile(self, s):
        return torch.sigmoid(self.gate(s))


class QuantileHead(torch.nn.Module):
    """R16: three fixed quantiles of the same label; the score is the median (preregistered)."""

    def __init__(self, width=80, hidden=64, quantiles=QUANTILES):
        super().__init__()
        self.quantiles = tuple(quantiles)
        self.body = torch.nn.Linear(width, hidden)
        self.out = torch.nn.Linear(hidden, len(self.quantiles))

    def forward(self, h, s):
        return self.out(torch.relu(self.body(h)))


def parameter_count(module):
    return int(sum(p.numel() for p in module.parameters()))


def build_arm(arm, state_dim):
    if arm == "rep_linear":
        return RepLinear()
    if arm == "rep_mlp":
        return RepMLP()
    if arm == "stats_linear":
        return StatsLinear(state_dim)
    if arm == "r15_concat":
        return ConcatMLP(80, state_dim)
    if arm == "r15_gate":
        return GateHead(80, state_dim)
    if arm == "r16_quantile":
        return QuantileHead()
    raise ValueError(f"unknown arm {arm!r}")


def pinball(prediction, target, quantiles):
    losses = []
    for index, q in enumerate(quantiles):
        error = target - prediction[:, index]
        losses.append(torch.maximum(q * error, (q - 1.0) * error))
    return torch.stack(losses, dim=-1).mean()


def train_arm(arm, cached, *, steps, batch_size, lr, seed, device, log_every, center=False):
    torch.manual_seed(seed)
    state_dim = int(cached.state.shape[1])
    model = build_arm(arm, state_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hidden = torch.from_numpy(np.ascontiguousarray(cached.h))
    state = torch.from_numpy(np.ascontiguousarray(cached.state))
    target = torch.from_numpy(np.ascontiguousarray(cached.y_opt.astype(np.float32)))
    # The pinball gradient has a fixed magnitude (tau or 1-tau) whatever the error, so the raw
    # optimisation target (~0.9 s.d. with fat tails) gives a signal far weaker than MSE's 2*error.
    # Standardising the target for that arm only equalises the effective step size; the exported
    # head is rescaled back to optimisation units so downstream code stays unchanged.
    target_std = 1.0
    if arm == "r16_quantile":
        target_std = float(np.std(target.numpy())) or 1.0
        target = target / target_std
        if center:
            # Ranking happens inside a timestamp, so the market-wide component of the label is pure
            # nuisance for a quantile head; removing it keeps the pinball signal cross-sectional.
            dates = pd.to_datetime(cached.keys["date"]).to_numpy()
            frame = pd.DataFrame({"date": dates, "y": target.numpy()})
            target = torch.from_numpy((frame.y - frame.groupby("date").y.transform("mean"))
                                      .to_numpy(dtype=np.float32))
    if arm == "r16_quantile":
        # Start the three outputs at the unconditional quantiles so the level costs no steps.
        with torch.no_grad():
            model.out.bias.copy_(torch.from_numpy(
                np.quantile(target.numpy(), QUANTILES).astype(np.float32)))
    total = len(target)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(total, generator=generator)
    cursor, step, running, ticks, trace = 0, 0, 0.0, 0, []
    first = None
    started = time.monotonic()
    while step < steps:
        if cursor >= total:
            order = torch.randperm(total, generator=generator)
            cursor = 0
        take = min(batch_size, total - cursor)
        index = order[cursor:cursor + take]
        cursor += take
        prediction = model(hidden[index].to(device), state[index].to(device))
        y = target[index].to(device)
        if arm == "r16_quantile":
            loss = pinball(prediction, y.unsqueeze(-1), QUANTILES)
        else:
            loss = torch.mean((prediction - y) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1
        value = float(loss.detach())
        first = value if first is None else first
        running += value
        ticks += 1
        if step % log_every == 0 or step == steps:
            if not np.isfinite(running):
                raise FloatingPointError(f"arm {arm} diverged at step {step}")
            trace.append({"step": step, "loss": running / ticks})
            log(f"  {arm} step {step}/{steps}: loss {running / ticks:.6g} "
                f"({time.monotonic() - started:.0f}s)")
            running, ticks = 0.0, 0
    if target_std != 1.0:
        with torch.no_grad():
            model.out.weight.mul_(target_std)
            model.out.bias.mul_(target_std)
    return model.eval(), {"arm": arm, "steps": steps, "batch_size": batch_size, "lr": lr,
                          "target_std": target_std, "center_target": bool(center),
                          "seed": seed, "parameters": parameter_count(model),
                          "first_loss": first, "final_loss": trace[-1]["loss"], "trace": trace,
                          "seconds": round(time.monotonic() - started, 1)}


def export_weights(model, path):
    payload = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    np.savez(path, **payload)
    return {name: list(value.shape) for name, value in payload.items()}


def load_weights(arm, path, state_dim):
    model = build_arm(arm, state_dim)
    saved = np.load(path)
    model.load_state_dict({name: torch.from_numpy(saved[name]) for name in saved.files})
    return model.eval()


def train_stage(args):
    torch.set_num_threads(args.threads)
    config = load_config(Path(args.config))
    cache = Path(args.cache)
    cached = read_cache(cache / "train")
    usable = cached.valid.astype(bool) & np.isfinite(cached.y_opt)
    for name in ("h", "state", "y_opt"):
        setattr(cached, name, getattr(cached, name)[usable])
    log(f"train: {len(cached.y_opt)} rows, state_dim={cached.state.shape[1]}, "
        f"weights={cached.meta['weights_sha256'][:12]}")
    out = Path(args.out)
    (out / "weights").mkdir(parents=True, exist_ok=True)
    summary = {"cache": str(cache), "rows": int(len(cached.y_opt)),
               "keys_sha256": cached.meta["keys_sha256"], "state_names": list(STATE_NAMES),
               "recipe": {"optimizer": "Adam", "lr": args.lr, "batch_size": args.batch_size,
                          "steps": args.steps, "seed": args.seed,
                          "objective": {"r16_quantile": "pinball(0.2,0.5,0.8)",
                                        "other": "mse on trunk optimisation target"}},
               "score_rules": SCORE_RULES, "control_of": CONTROL_OF, "arms": {},
               "generated_utc": pd.Timestamp.utcnow().isoformat()}
    for arm in args.arms.split(","):
        arm = arm.strip()
        model, record = train_arm(arm, cached, steps=args.steps, batch_size=args.batch_size,
                                  lr=args.lr, seed=args.seed, device=args.device,
                                  log_every=args.log_every, center=args.center_target)
        weight_path = out / "weights" / f"{arm}.npz"
        record["shapes"] = export_weights(model, weight_path)
        record["weights"] = str(weight_path)
        record["weights_sha256"] = file_sha256(weight_path)
        summary["arms"][arm] = record
        log(f"{arm}: {record['parameters']} parameters, exported {weight_path}")
    counts = {arm: summary["arms"][arm]["parameters"] for arm in summary["arms"]}
    for treatment, control in CONTROL_OF.items():
        if treatment in counts and control in counts:
            delta = (counts[treatment] - counts[control]) / counts[control]
            summary.setdefault("parameter_match", {})[f"{treatment}_vs_{control}"] = {
                "treatment": counts[treatment], "control": counts[control],
                "relative": float(delta)}
    write_json_safe(out / "training.json", summary)
    return summary


# -------------------------------------------------------------------------------- score stage
def score_arms(cached, arms, weights_dir, device):
    """Scores for every arm on one cached unit, in the cached row order."""
    hidden = torch.from_numpy(np.ascontiguousarray(cached.h))
    state = torch.from_numpy(np.ascontiguousarray(cached.state))
    scores = {}
    for arm in arms:
        model = load_weights(arm, Path(weights_dir) / f"{arm}.npz", int(cached.state.shape[1]))
        with torch.inference_mode():
            output = model(hidden, state)
        if arm == "r16_quantile":
            quantiles = output.numpy().astype(np.float64)
            scores[arm] = {"score": quantiles[:, 1], "quantiles": quantiles}
        else:
            scores[arm] = {"score": output.numpy().astype(np.float64)}
    return scores


def score_stage(args):
    torch.set_num_threads(args.threads)
    from jinyiquant.evaluation import evaluate
    cache = Path(args.cache)
    cached = read_cache(cache / args.unit)
    arms = [arm.strip() for arm in args.arms.split(",")]
    scores = score_arms(cached, arms, Path(args.out) / "weights", args.device)
    # Heads regress the trunk optimisation target (label / target_scale); rescale to label units so
    # every arm, including the frozen base score, is expressed in the same units downstream.
    for arm in scores:
        scores[arm]["score"] = scores[arm]["score"] * cached.meta["target_scale"]
        if "quantiles" in scores[arm]:
            scores[arm]["quantiles"] = scores[arm]["quantiles"] * cached.meta["target_scale"]
    keys = cached.keys.reset_index(drop=True)
    targets = keys.assign(target=cached.target).dropna(subset=["target"])
    base = cached.base * cached.meta["target_scale"]
    rows = []
    window_dir = Path(args.out) / "windows" / args.unit
    window_dir.mkdir(parents=True, exist_ok=True)
    keys.assign(score=base).loc[cached.valid].to_parquet(window_dir / "s_base.parquet", index=False)
    for arm in arms:
        values = scores[arm]["score"]
        scored = keys.assign(score=values).loc[cached.valid]
        scored = scored.merge(targets[["date", "instrument"]], on=["date", "instrument"], how="inner")
        metrics, daily, sections = evaluate(scored, targets, EVALUATION, return_sections=True)
        scored.to_parquet(window_dir / f"{arm}.parquet", index=False)
        daily.to_csv(window_dir / f"{arm}.daily.csv", index=False)
        metrics["window"] = args.unit
        metrics["arm"] = arm
        metrics["rows"] = int(len(scored))
        if arm == "r16_quantile":
            quantiles = scores[arm]["quantiles"]
            frame = keys.assign(q20=quantiles[:, 0], q50=quantiles[:, 1], q80=quantiles[:, 2])
            frame.loc[cached.valid].to_parquet(window_dir / "r16_quantiles.parquet", index=False)
            metrics["tail"] = {"mean_width": float(np.mean(quantiles[cached.valid, 2] -
                                                          quantiles[cached.valid, 0])),
                               "mean_skew": float(np.mean(quantiles[cached.valid, 2] +
                                                          quantiles[cached.valid, 0] -
                                                          2 * quantiles[cached.valid, 1]))}
        rows.append(metrics)
        log(f"{args.unit} {arm}: rank_ic {fmt(metrics['rank_ic_mean'])} "
            f"long_short {fmt(metrics['long_short_return_mean'])} "
            f"turnover {fmt(metrics['turnover_mean'], 4)} "
            f"ranking_defined_sections {metrics.get('ranking_defined_section_count')}/"
            f"{metrics.get('cross_section_count')}")
    base_scored = keys.assign(score=base).loc[cached.valid].merge(
        targets[["date", "instrument"]], on=["date", "instrument"], how="inner")
    metrics, daily, _ = evaluate(base_scored, targets, EVALUATION, return_sections=True)
    daily.to_csv(window_dir / "s_base.daily.csv", index=False)
    metrics.update(window=args.unit, arm="s_base", rows=int(len(base_scored)))
    rows.append(metrics)
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(window_dir / "metrics.csv", index=False)
    log(f"{args.unit}: freeze-trunk base rank_ic {fmt(metrics['rank_ic_mean'])}")
    return rows


# ------------------------------------------------------------------------------- report stage
def block_interval(values, *, block=5, draws=20000, seed=20261028):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {"days": int(len(values)), "mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), size=(draws,))
    offsets = np.arange(block)[None, :]
    indices = (starts[:, None] + offsets) % len(values)
    means = values[indices].mean(axis=1)
    return {"days": int(len(values)), "mean": float(values.mean()),
            "ci_low": float(np.quantile(means, 0.025)),
            "ci_high": float(np.quantile(means, 0.975))}


def daily_series(frames):
    """One row per trading day per arm: the eight sample times are averaged before pairing."""
    series = {}
    for arm, frame in frames.items():
        frame = frame.copy()
        frame["day"] = pd.to_datetime(frame["date"]).dt.normalize()
        series[arm] = (frame.groupby("day", sort=True)[["rank_ic", "long_short_return", "turnover"]]
                       .mean().reset_index())
    return series


def paired(treatment, control, *, value="rank_ic"):
    joined = treatment[["day", value]].merge(control[["day", value]], on="day",
                                             suffixes=("", "_control"), how="inner",
                                             validate="one_to_one")
    return block_interval(joined[value] - joined[f"{value}_control"])


def report_stage(args):
    out = Path(args.out)
    windows = [w.strip() for w in args.windows.split(",")] if args.windows else list(WINDOWS)
    metrics = []
    for window in windows:
        path = out / "windows" / window / "metrics.csv"
        if not path.exists():
            log(f"{window}: no metrics yet, skipped")
            continue
        metrics.append(pd.read_csv(path).assign(window=window))
    if not metrics:
        raise SystemExit("no windows scored yet")
    metrics = pd.concat(metrics, ignore_index=True)
    metrics.to_csv(out / "metrics_by_window.csv", index=False)
    frames = {}
    for window in windows:
        for path in sorted((out / "windows" / window).glob("*.daily.csv")):
            arm = path.name[: -len(".daily.csv")]
            frames.setdefault(arm, []).append(pd.read_csv(path).assign(window=window))
    frames = {arm: pd.concat(parts, ignore_index=True) for arm, parts in frames.items()}
    series = daily_series(frames)
    base = series.get("s_base")
    pooled, per_window = [], []
    for arm, days in series.items():
        entry = {"arm": arm, "days": int(len(days)),
                 "rank_ic_mean": float(days.rank_ic.mean()),
                 "long_short_return_mean": float(days.long_short_return.mean()),
                 "turnover_mean": float(days.turnover.mean())}
        if base is not None and arm != "s_base":
            entry["delta_rank_ic_vs_base"] = paired(days, base)
            entry["delta_long_short_vs_base"] = paired(days, base, value="long_short_return")
        pooled.append(entry)
        for window in windows:
            path = out / "windows" / window / f"{arm}.daily.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame["day"] = pd.to_datetime(frame["date"]).dt.normalize()
            day_means = frame.groupby("day")[["rank_ic", "long_short_return", "turnover"]].mean()
            base_path = out / "windows" / window / "s_base.daily.csv"
            delta = None
            if base_path.exists():
                base_frame = pd.read_csv(base_path)
                base_frame["day"] = pd.to_datetime(base_frame["date"]).dt.normalize()
                base_day = base_frame.groupby("day")["rank_ic"].mean()
                joined = day_means["rank_ic"].to_frame("rank_ic").join(
                    base_day.to_frame("rank_ic_base"), how="inner")
                delta = float((joined.rank_ic - joined.rank_ic_base).mean())
            per_window.append({"window": window, "arm": arm, "days": int(len(day_means)),
                               "rank_ic_mean": float(day_means.rank_ic.mean()),
                               "long_short_return_mean": float(day_means.long_short_return.mean()),
                               "turnover_mean": float(day_means.turnover.mean()),
                               "delta_rank_ic_vs_base": delta})
    pooled = pd.DataFrame(pooled).sort_values("rank_ic_mean", ascending=False)
    pooled.to_csv(out / "pooled.csv", index=False)
    per_window = pd.DataFrame(per_window)
    per_window.to_csv(out / "per_window.csv", index=False)
    pairs = []
    for treatment, control in CONTROL_OF.items():
        if treatment not in series or control not in series:
            continue
        pairs.append({"treatment": treatment, "control": control,
                      "delta_rank_ic": paired(series[treatment], series[control]),
                      "delta_long_short": paired(series[treatment], series[control],
                                                 value="long_short_return")})
    write_json_safe(out / "summary.json", {
        "generated_utc": pd.Timestamp.utcnow().isoformat(), "windows": windows,
        "score_rules": SCORE_RULES, "control_of": CONTROL_OF,
        "pooled": pooled.to_dict(orient="records"), "paired": pairs,
        "block_bootstrap": {"block": 5, "draws": 20000, "seed": 20261028,
                            "unit": "trading day, eight sample times averaged"}})
    log(pooled.to_string(index=False))
    for pair in pairs:
        log(f"{pair['treatment']} - {pair['control']}: delta rank_ic "
            f"{pair['delta_rank_ic']['mean']:+.5f} "
            f"[{pair['delta_rank_ic']['ci_low']:+.5f},{pair['delta_rank_ic']['ci_high']:+.5f}] "
            f"days={pair['delta_rank_ic']['days']}")
    return {"pooled": pooled.to_dict(orient="records"), "paired": pairs}


def selfcheck_stage(args):
    """Behavioural checks: no sealed read, point-in-time state, and the pre-registered score rule."""
    config = load_config(Path(args.config))
    remote = args.remote_data
    from jinyiquant.data import SEALED_END, SEALED_START

    sealed_lo, sealed_hi = pd.Timestamp(SEALED_START), pd.Timestamp(SEALED_END)
    report = {"units": {}, "state_point_in_time": {}, "quantile_rule": {}}
    for name, (start, end) in [("train", TRAIN_UNIT), *WINDOWS.items()]:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if start < sealed_hi and end > sealed_lo:
            raise AssertionError(f"{name} intersects the sealed block {sealed_lo}..{sealed_hi}")
        report["units"][name] = {"start": str(start), "end": str(end), "sealed_clear": True}

    cache = Path(args.cache)
    unit = args.unit or "nov2023"
    cached = read_cache(cache / unit)
    if not (cache / unit / "state.npy").exists():
        log(f"selfcheck: {unit} has no state cache, state checks skipped")
    else:
        labels_dirs = sorted({Path(dataset("primary", remote=remote)["labels"]),
                              Path(dataset("extended", remote=remote)["labels"])})
        start, end = (pd.Timestamp(value) for value in WINDOWS[unit])
        days = np.sort(pd.unique(pd.to_datetime(cached.keys["date"]).dt.normalize()))[:4]
        checked = {}
        for day in days:
            # Recompute the state using only labels strictly before this day and compare.
            cached_rows = np.asarray(cached.state, dtype=np.float32)
            mask = pd.to_datetime(cached.keys["date"]).dt.normalize().to_numpy() == pd.Timestamp(day)
            if not mask.any():
                continue
            # Recompute from the same cross section, with labels truncated to days < day.
            daily = label_history(labels_dirs, pd.Timestamp(start.strftime("%Y-%m-01")),
                                  pd.Timestamp(day), STATE_LOOKBACK_DAYS)
            recomputed, _ = state_features(daily, cached.keys.loc[mask, ["date", "instrument"]])
            reference = np.asarray(recomputed, dtype=np.float64)
            diff = float(np.max(np.abs(reference - cached_rows[mask].astype(np.float64))))
            checked[str(pd.Timestamp(day).date())] = diff
        worst = max(checked.values()) if checked else None
        report["state_point_in_time"] = {"unit": unit, "days": checked, "max_abs_diff": worst}
        if worst is not None and worst > 1e-9:
            raise AssertionError(f"state cache is not point in time: max_abs_diff={worst:g}")

    quantiles_path = Path(args.out) / "windows" / unit / "r16_quantiles.parquet"
    if quantiles_path.exists():
        frame = pd.read_parquet(quantiles_path)
        score = pd.read_parquet(Path(args.out) / "windows" / unit / "r16_quantile.parquet")
        merged = frame.merge(score, on=["date", "instrument"], how="inner")
        diff = float(np.max(np.abs(merged["q50"] - merged["score"])))
        report["quantile_rule"] = {"rows": int(len(merged)), "max_abs_diff": diff,
                                   "rule": SCORE_RULES["r16_quantile"]}
        if diff > 1e-6:
            raise AssertionError(f"R16 score is not the median: max_abs_diff={diff:g}")
    write_json_safe(Path(args.out) / "selfcheck.json", report)
    log(f"selfcheck: sealed clear for {len(report['units'])} units; "
        f"state max_abs_diff {report['state_point_in_time'].get('max_abs_diff')}; "
        f"quantile rule {report['quantile_rule']}")
    return report


def spearman(frame, value_column):
    """Mean per-timestamp cross-sectional rank correlation of one score column with the label."""
    correlations = []
    for _, section in frame.groupby("date", sort=True):
        if len(section) < 3:
            continue
        left = pd.Series(section[value_column]).rank().to_numpy()
        right = pd.Series(section["target"]).rank().to_numpy()
        if left.std() == 0 or right.std() == 0:
            correlations.append(0.0)
            continue
        correlations.append(float(np.corrcoef(left, right)[0, 1]))
    return float(np.mean(correlations)) if correlations else None


def diag_stage(args):
    """Why did an arm fail? Training-objective fit, tail behaviour and score dispersion."""
    torch.set_num_threads(args.threads)
    cache, out = Path(args.cache), Path(args.out)
    units = [args.unit] if args.unit else list(WINDOWS)
    arms = [arm.strip() for arm in args.arms.split(",")]
    rows, gate_stats = [], []
    for unit in units:
        cached = read_cache(cache / unit)
        valid = cached.valid.astype(bool)
        scores = score_arms(cached, arms, out / "weights", args.device)
        keys = cached.keys.reset_index(drop=True)
        target = keys.assign(target=cached.target * 1.0)
        scale = cached.meta["target_scale"]
        y = cached.y_opt.astype(np.float64)
        # Trivial reference: per-timestamp empirical quantiles of the very same target (in sample,
        # so it is optimistic for the baseline and conservative for the trained head).
        frame = keys.assign(y=y, target=cached.target)
        reference = frame.dropna(subset=["target"])
        quantile_table = reference.groupby("date")["y"].quantile(QUANTILES).unstack()
        for arm in arms:
            values = scores[arm]["score"] / scale  # back to optimisation units for the losses
            scored = keys.assign(score=values, target=cached.target).loc[valid].dropna(subset=["target"])
            dispersion = (scored.groupby("date")["score"].std().mean())
            entry = {"unit": unit, "arm": arm, "cross_section_std": float(dispersion),
                     "rank_ic": spearman(scored, "score")}
            if "quantiles" in scores[arm]:
                quantiles = scores[arm]["quantiles"] / scale
                scored_q = keys.loc[valid].assign(q20=quantiles[valid, 0], q50=quantiles[valid, 1],
                                                  q80=quantiles[valid, 2], target=cached.target[valid],
                                                  y=y[valid])
                scored_q = scored_q.dropna(subset=["target"])
                entry["pinball"] = float(pinball_loss(scored_q, ("q20", "q50", "q80")))
                entry["pinball_trivial"] = float(pinball_loss(scored_q, None, quantile_table))
                entry["crossing_rate"] = float(((scored_q.q20 > scored_q.q50) |
                                                (scored_q.q50 > scored_q.q80)).mean())
                entry["ic_q20"] = spearman(scored_q, "q20")
                entry["ic_q50"] = spearman(scored_q, "q50")
                entry["ic_q80"] = spearman(scored_q, "q80")
                entry["mean_width"] = float((scored_q.q80 - scored_q.q20).mean())
                entry["skew_proxy"] = float((scored_q.q80 + scored_q.q20 - 2 * scored_q.q50).mean())
            if arm == "r15_gate" and (out / "weights" / "r15_gate.npz").exists():
                model = load_weights("r15_gate", out / "weights" / "r15_gate.npz",
                                     int(cached.state.shape[1]))
                with torch.inference_mode():
                    gate = model.gate_profile(torch.from_numpy(cached.state[valid])).numpy()
                state = cached.state[valid]
                gate_stats.append({"unit": unit, "mean_gate": float(gate.mean()),
                                   "std_gate": float(gate.std()),
                                   "state_gate_corr": [float(np.corrcoef(state[:, i], gate.mean(axis=1))[0, 1])
                                                       for i in range(state.shape[1])]})
            rows.append(entry)
            log(f"{unit} {arm}: ic {entry['rank_ic']} std {entry['cross_section_std']:.2e}"
                + (f" pinball {entry['pinball']:.5f} vs trivial {entry['pinball_trivial']:.5f} "
                   f"crossing {entry['crossing_rate']:.3f}" if "pinball" in entry else ""))
    table = pd.DataFrame(rows)
    table.to_csv(out / "diag.csv", index=False)
    write_json_safe(out / "diag.json", {"rows": rows, "gate": gate_stats,
                                        "state_names": list(STATE_NAMES)})
    log(table.groupby("arm")[["rank_ic", "cross_section_std"]].mean().to_string())
    return rows


def pinball_loss(scored, columns, quantile_table=None):
    """Pinball loss in optimisation units; ``columns`` names arm quantiles, else the trivial one."""
    error = []
    for index, q in enumerate(QUANTILES):
        if columns is not None:
            prediction = scored[columns[index]].to_numpy(dtype=np.float64)
        else:
            prediction = scored.merge(quantile_table[[q]].rename(columns={q: "reference"}),
                                      left_on="date", right_index=True,
                                      how="left")["reference"].to_numpy(dtype=np.float64)
        target = scored["y"].to_numpy(dtype=np.float64)
        error.append(np.maximum(q * (target - prediction), (q - 1.0) * (target - prediction)))
    return float(np.mean(np.concatenate(error)))


def rank_normal(values):
    """Cross-sectional rank mapped to a standard normal score (0 mean, unit variance per timestamp)."""
    ranks = pd.Series(values).rank(method="average").to_numpy()
    count = len(ranks)
    if count < 3:
        return np.zeros(count)
    return special.ndtri((ranks - 0.5) / count)


def fusion_stage(args):
    """Out-of-sample fusion increment: does adding an arm to the frozen base score help per day?"""
    out = Path(args.out)
    units = [args.unit] if args.unit else list(WINDOWS)
    arms = [arm.strip() for arm in args.arms.split(",")]
    weights = [float(value) for value in args.blend_weights.split(",")]
    rows = []
    for unit in units:
        window = out / "windows" / unit
        base = pd.read_parquet(window / "s_base.parquet").rename(columns={"score": "base"})
        for arm in arms:
            path = window / f"{arm}.parquet"
            if not path.exists():
                continue
            frame = base.merge(pd.read_parquet(path).rename(columns={"score": arm}),
                               on=["date", "instrument"], how="inner")
            target = pd.read_parquet(window / "targets.parquet") if (window / "targets.parquet").exists() else None
            if target is None:
                # targets are not materialised per window; rebuild them from the cache instead.
                cached = read_cache(Path(args.cache) / unit)
                target = cached.keys.assign(target=cached.target).dropna(subset=["target"])
            frame = frame.merge(target[["date", "instrument", "target"]], on=["date", "instrument"],
                                how="inner")
            sections = []
            for _, section in frame.groupby("date", sort=True):
                if len(section) < 3:
                    continue
                label = section["target"].to_numpy(dtype=np.float64)
                base_z = rank_normal(section["base"].to_numpy(dtype=np.float64))
                arm_z = rank_normal(section[arm].to_numpy(dtype=np.float64))
                row = {"date": section["date"].iloc[0], "unit": unit,
                       "base_ic": spearman_section(base_z, label)}
                for weight in weights:
                    blended = (1.0 - weight) * base_z + weight * arm_z
                    row[f"ic_{weight:g}"] = spearman_section(blended, label)
                sections.append(row)
            daily = pd.DataFrame(sections)
            daily.to_csv(window / f"fusion_{arm}.daily.csv", index=False)
            for weight in [0.0, *weights]:
                column = "base_ic" if weight == 0 else f"ic_{weight:g}"
                rows.append({"unit": unit, "arm": arm, "weight": weight,
                             "ic": float(daily[column].mean()), "days": int(len(daily)),
                             "series": daily[column].to_numpy(dtype=float)})
    table = pd.DataFrame([{key: value for key, value in row.items() if key != "series"} for row in rows])
    table.to_csv(out / "fusion.csv", index=False)
    summary = {}
    for arm in sorted({row["arm"] for row in rows}):
        reference = next((row for row in rows if row["arm"] == arm and row["weight"] == 0.0), None)
        summary[arm] = {}
        for weight in weights:
            row = next((item for item in rows if item["arm"] == arm and item["weight"] == weight), None)
            if row is None or reference is None:
                continue
            delta = row["series"] - reference["series"]
            summary[arm][f"{weight:g}"] = {"ic": row["ic"], "base_ic": reference["ic"],
                                           "delta": float(np.mean(delta)),
                                           "interval": block_interval(delta),
                                           "days": int(len(delta))}
            log(f"fusion {arm} w={weight:g}: ic {row['ic']:.5f} "
                f"(base {reference['ic']:.5f}) delta {np.mean(delta):+.5f} "
                f"[{summary[arm][f'{weight:g}']['interval']['ci_low']:+.5f},"
                f"{summary[arm][f'{weight:g}']['interval']['ci_high']:+.5f}]")
    write_json_safe(out / "fusion.json", {"weights": weights, "arms": summary})
    return summary


def spearman_section(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.corrcoef(pd.Series(left).rank(), pd.Series(right).rank())[0, 1])


# ------------------------------------------------------------------------------------- CLI
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "stats", "train", "score", "report",
                                          "selfcheck", "diag", "fusion"))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--unit", default=None)
    parser.add_argument("--windows", default=None)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--blend-weights", default="0.1,0.2,0.3")
    parser.add_argument("--center-target", action="store_true",
                        help="quantile arm only: remove the per-timestamp cross-sectional mean")
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--part-rows", type=int, default=500000)
    parser.add_argument("--slice-rows", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=1, help="train day stride")
    parser.add_argument("--limit-days", type=int, default=None, help="smoke: first N calendar days")
    parser.add_argument("--remote-data", action="store_true")
    parser.add_argument("--dataset", default=None, choices=("primary", "extended"),
                        help="override which dataset root every unit reads")
    parser.add_argument("--labels-dirs", default=None,
                        help="comma-separated label directories for the state history")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.stage == "extract":
        config = load_config(Path(args.config))
        adapter = make_adapter(args.weights, args.device)
        units = [args.unit] if args.unit else ["train", *WINDOWS]
        metas = {}
        for name in units:
            start, end = TRAIN_UNIT if name == "train" else WINDOWS[name]
            paths = args.dataset and dataset(args.dataset, remote=args.remote_data) or dataset(
                data_kind_for(name), remote=args.remote_data)
            data = rooted_data(config.data, paths)
            log(f"{name}: bars {paths['bars']} labels {paths['labels']}")
            metas[name] = extract_unit(config, data, adapter, name=name, start=start, end=end,
                                       out=Path(args.cache) / name, weights=args.weights,
                                       part_rows=args.part_rows, slice_rows=args.slice_rows,
                                       stride=args.stride, limit_days=args.limit_days)
        write_json_safe(Path(args.cache) / "extraction.json", metas)
        return metas
    if args.stage == "stats":
        return stats_stage(args)
    if args.stage == "train":
        return train_stage(args)
    if args.stage == "score":
        if not args.unit:
            raise SystemExit("score needs --unit")
        return score_stage(args)
    if args.stage == "selfcheck":
        return selfcheck_stage(args)
    if args.stage == "diag":
        return diag_stage(args)
    if args.stage == "fusion":
        return fusion_stage(args)
    return report_stage(args)


if __name__ == "__main__":
    main()
