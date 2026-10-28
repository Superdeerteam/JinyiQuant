"""R11: neighbour lead-lag correction head on the frozen 441-day seed-11 trunk.

Hypothesis (registered): a related stock's recent move carries information this stock has
not yet reflected, and every stock listens to its *own* leaders -- unlike the rejected
whole-market summary head, which gave every stock the same context vector.

Stages
1. ``select``  -- neighbour map from the training window only (2022-01-04 .. 2023-11-01).
   Statistic: corr(neighbour 15-minute return / imbalance change over [k-15, k], own forward
   30-minute return / imbalance change over [k, k+30]) evaluated on every same-session
   minute pair of the training window (~166 rows/day).  No label, no post-training data.
   The map and the frozen random permutation are hashed before anything downstream runs.
2. ``train``   -- score the frozen trunk (r441, seed 11, 441d @1.32) on the training window,
   build neighbour features, fit two residual heads on identical rows and capacity:
   ``real`` (frozen map) and ``rand`` (frozen permutation of the same neighbour sets).
3. ``predict`` -- dev windows 2023-11 .. 2024-09 (unsealed); base score is the archived
   dev10 replay r441_s11 prediction, the head only adds a per-timestamp standardised
   correction, so the fusion weight is scale-free.
4. ``evaluate``-- per-day paired differences, long/short, turnover, inference cost; the
   fusion weight is frozen on the early dev block and judged on the later blocks.

Red lines: neighbours come from training history only and are frozen before the first dev
read; the sealed interval is never touched (``load_data`` refuses it); nothing is committed.
"""

from __future__ import annotations

import argparse
import gc
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jinyiquant.data import build_batch, load_data, select_targets  # noqa: E402
from jinyiquant.evaluation import evaluate  # noqa: E402
from jinyiquant.io import file_sha256, read_json, utc_now, write_json  # noqa: E402
from jinyiquant.tracking import Recorder, capture_source, finish, new_run_dir  # noqa: E402
from modified_tlob.adapter import TLOBAdapter  # noqa: E402

PLAN = ROOT / "configs" / "dev10_replay_20261028_plan.json"
TRUNK = ROOT / "runs" / "dose-replay-20261027" / "weights" / "441d-seed11" / \
    "modified_tlob_weights-scaling-1.32.json"
TRUNK_SHA = "f3466c6d207e6377c3b66e304a1acc026fe9d10963007373550a0e79848ee96a"
REPLAY_ROOT = ROOT / "runs" / "dev10-replay-20261028" / "20261028T094439.842764Z_0854d2d5ca3d"

TRAIN_START, TRAIN_END = pd.Timestamp("2022-01-04"), pd.Timestamp("2023-11-01")
CLOCKS = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")
LEAD_MINUTES, FOLLOW_MINUTES = 15, 30
SESSION_BOUNDS = ((0, 134), (135, 255))


def dense_lead_slots() -> np.ndarray:
    """Slots k whose leader window [k-15, k] and follower window [k, k+30] stay in one session."""
    slots = []
    for low, high in SESSION_BOUNDS:
        slots.extend(slot for slot in range(max(low, LEAD_MINUTES), high + 1)
                     if slot + FOLLOW_MINUTES <= high and slot - LEAD_MINUTES >= low)
    return np.asarray(slots, dtype=np.int64)


DENSE_SLOTS = dense_lead_slots()
HORIZONS = (1, 5, 15)
K_NEIGHBOURS = 5
NEIGHBOUR_SEED = 20261028
SELECT_FIELDS = ("bid_price1", "ask_price1", "bid_volume1", "bid_volume2", "bid_volume3",
                 "ask_volume1", "ask_volume2", "ask_volume3")
N_FEATURES = K_NEIGHBOURS * len(HORIZONS) * 3
FFILL_LIMIT = 5
MORNING = (9 * 60 + 16, 11 * 60 + 30)
AFTERNOON = (13 * 60, 15 * 60)
N_SLOTS = (MORNING[1] - MORNING[0] + 1) + (AFTERNOON[1] - AFTERNOON[0] + 1)
CLOCK_SLOT = {"09:30": 14, "10:00": 44, "10:30": 74, "11:00": 104, "11:30": 134,
              "13:30": 165, "14:00": 195, "14:30": 225}
WEIGHT_GRID = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)
EARLY_BLOCK = ("nov2023", "jan2024", "feb2024")
HIDDEN = 32


def data_config(fields=None) -> dict:
    plan = read_json(PLAN)
    data = dict(plan["data"])
    base = PLAN.parent
    for name in ("bars_path", "labels_path", "instruments_path", "instrument_map_path"):
        data[name] = str((base / data[name]).resolve())
    data["cache_dir"] = None
    if fields is not None:
        data["feature_columns"] = list(fields)
    return data


# ---------------------------------------------------------------- bar geometry
def minute_slot(minutes: np.ndarray) -> np.ndarray:
    slot = np.full(minutes.shape, -1, dtype=np.int32)
    morning = (minutes >= MORNING[0]) & (minutes <= MORNING[1])
    slot[morning] = minutes[morning] - MORNING[0]
    afternoon = (minutes >= AFTERNOON[0]) & (minutes <= AFTERNOON[1])
    slot[afternoon] = (MORNING[1] - MORNING[0] + 1) + minutes[afternoon] - AFTERNOON[0]
    return slot


def ffill_slots(array: np.ndarray, limit: int = FFILL_LIMIT) -> np.ndarray:
    """Forward fill along the session-minute axis, at most ``limit`` slots."""
    out = array.copy()
    for step in range(1, limit + 1):
        shifted = np.full_like(out, np.nan)
        shifted[:, step:] = out[:, :-step]
        out = np.where(np.isnan(out), shifted, out)
    return out


def day_fields(frame: pd.DataFrame, codes: pd.Series, n_codes: int) -> dict:
    """One day of (n_codes, N_SLOTS) forward-filled mid / depth imbalance / spread."""
    minutes = (frame["date"].dt.hour * 60 + frame["date"].dt.minute).to_numpy()
    slot = minute_slot(minutes)
    code = codes.reindex(frame["instrument"]).to_numpy()
    keep = (slot >= 0) & pd.notna(code)
    if not keep.any():
        return {}
    code = code[keep].astype(np.int64)
    flat = code * N_SLOTS + slot[keep]

    def stack(values: np.ndarray) -> np.ndarray:
        out = np.full(n_codes * N_SLOTS, np.nan)
        out[flat] = values
        return out.reshape(n_codes, N_SLOTS)

    bid_price = frame["bid_price1"].to_numpy(dtype=np.float64)
    ask_price = frame["ask_price1"].to_numpy(dtype=np.float64)
    mid_value = (bid_price + ask_price) / 2.0
    bid_volume = np.nansum(np.vstack([frame[f"bid_volume{i}"].to_numpy(dtype=np.float64)
                                      for i in (1, 2, 3)]), axis=0)
    ask_volume = np.nansum(np.vstack([frame[f"ask_volume{i}"].to_numpy(dtype=np.float64)
                                      for i in (1, 2, 3)]), axis=0)
    total = bid_volume + ask_volume
    with np.errstate(divide="ignore", invalid="ignore"):
        imbalance = np.where(total > 0, (bid_volume - ask_volume) / total, np.nan)
        spread = np.where(mid_value > 0, (ask_price - bid_price) / mid_value, np.nan)
    return {"mid": ffill_slots(stack(mid_value[keep])),
            "imbalance": ffill_slots(stack(imbalance[keep])),
            "spread": ffill_slots(stack(spread[keep]))}


def clock_deltas(fields: dict, clock: str) -> list[np.ndarray]:
    """Change blocks at one clock: per horizon (return, imbalance change, spread change)."""
    slot = CLOCK_SLOT[clock]
    blocks = []
    for horizon in HORIZONS:
        back = max(0, slot - horizon)
        mid, imbalance, spread = fields["mid"], fields["imbalance"], fields["spread"]
        with np.errstate(divide="ignore", invalid="ignore"):
            blocks.extend([mid[:, slot] / mid[:, back] - 1.0,
                           imbalance[:, slot] - imbalance[:, back],
                           spread[:, slot] - spread[:, back]])
    return blocks


def cross_section_z(values: np.ndarray) -> np.ndarray:
    """Per-timestamp cross-sectional standardisation; missing -> 0; clip to +-4."""
    import warnings

    finite = np.where(np.isfinite(values), values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(finite, axis=0)
        std = np.nanstd(finite, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    z = (values - mean) / std
    z = np.where(np.isfinite(z), z, 0.0)
    return np.clip(z, -4.0, 4.0)


def build_field_cache(bars: pd.DataFrame, instruments: pd.Index) -> dict:
    """Per-day (mid, imbalance, spread) matrices, computed once for every neighbour map."""
    codes = pd.Series(np.arange(len(instruments)), index=instruments)
    cache = {}
    for current_day, day_frame in bars.groupby(bars["date"].dt.normalize(), sort=True):
        fields = day_fields(day_frame, codes, len(instruments))
        if fields:
            cache[np.datetime64(current_day, "ns").astype(np.int64)] = fields
    return cache


def neighbour_features(keys: pd.DataFrame, neighbour_codes: np.ndarray, instruments: pd.Index,
                       field_cache: dict) -> np.ndarray:
    """Neighbour features for ``keys`` rows, cross-sectionally z-scored at every timestamp."""
    codes = pd.Series(np.arange(len(instruments)), index=instruments)
    key_code = codes.reindex(keys["instrument"]).to_numpy()
    output = np.zeros((len(keys), N_FEATURES), dtype=np.float32)
    stamp = keys["date"].dt.strftime("%H:%M").to_numpy()
    day_key = keys["date"].dt.normalize().to_numpy().astype("datetime64[ns]").astype(np.int64)
    for day_int, fields in field_cache.items():
        in_day = day_key == day_int
        if not in_day.any():
            continue
        for clock in CLOCKS:
            rows = np.where(in_day & (stamp == clock))[0]
            if not len(rows):
                continue
            valid = np.isfinite(key_code[rows])
            blocks = clock_deltas(fields, clock)
            gather = np.where(valid[:, None], neighbour_codes[np.where(valid, key_code[rows], 0).astype(np.int64)],
                              -1)
            columns = []
            for block in blocks:
                values = np.where(gather >= 0, block[np.maximum(gather, 0)], np.nan)
                columns.append(values)
            output[rows] = cross_section_z(np.column_stack(columns))
    return output


# ---------------------------------------------------------------- selection statistics
def lag_correlation(leaders: np.ndarray, followers: np.ndarray) -> np.ndarray:
    """corr(leaders[:, j], followers[:, i]) with coverage correction for missing rows."""
    import warnings

    leaders = np.ascontiguousarray(leaders, dtype=np.float32)
    followers = np.ascontiguousarray(followers, dtype=np.float32)

    mask_a, mask_b = np.isfinite(leaders), np.isfinite(followers)
    rows = leaders.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_a = np.nanmean(np.where(mask_a, leaders, np.nan), axis=0)
        mean_b = np.nanmean(np.where(mask_b, followers, np.nan), axis=0)
        std_a = np.nanstd(np.where(mask_a, leaders, np.nan), axis=0)
        std_b = np.nanstd(np.where(mask_b, followers, np.nan), axis=0)
    std_a = np.where(np.isfinite(std_a) & (std_a > 0), std_a, np.nan)
    std_b = np.where(np.isfinite(std_b) & (std_b > 0), std_b, np.nan)
    za = np.where(mask_a, (leaders - np.nan_to_num(mean_a)) / np.nan_to_num(std_a, nan=1.0), 0.0)
    zb = np.where(mask_b, (followers - np.nan_to_num(mean_b)) / np.nan_to_num(std_b, nan=1.0), 0.0)
    product = (za.T @ zb) / rows
    pair = (mask_a.astype(np.float32).T @ mask_b.astype(np.float32)) / rows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr = np.where(pair >= 0.5, product / np.where(pair > 0, pair, np.nan), np.nan)
    return np.clip(corr, -1.0, 1.0)


def select_map(args, run_dir: Path, recorder: Recorder) -> dict:
    config = data_config(fields=SELECT_FIELDS)
    leader_ret, follower_ret, leader_imb, follower_imb = [], [], [], []
    instruments, codes = None, None
    select_start = max(PD_TRAIN_START, TRAIN_END - pd.DateOffset(months=args.select_months))
    pairs = list(select_chunks(select_start, TRAIN_END, args.chunk_days))
    if args.limit_months:
        months = list(pd.period_range(select_start, TRAIN_END - pd.Timedelta(days=1), freq="M"))
        pairs = [pair for pair in pairs if pair[0] in set(months[: args.limit_months])]
    recorder.event("select", "started", chunks=len(pairs), start=str(select_start.date()),
                   end=str(TRAIN_END.date()), k=K_NEIGHBOURS,
                   statistic="corr(neighbour 15-min change at k, own forward 30-min change at k) "
                             "on return and imbalance, computed per minute over the training window")
    for month, chunk_start, chunk_end in pairs:
        began = time.monotonic()
        bundle = load_data(config, chunk_start, chunk_end)
        bars = bundle.bars
        if instruments is None:
            instruments = pd.Index(sorted(bars["instrument"].unique()))
            codes = pd.Series(np.arange(len(instruments)), index=instruments)
        for _, frame in bars.groupby(bars["date"].dt.normalize(), sort=True):
            fields = day_fields(frame, codes, len(instruments))
            if not fields:
                continue
            mid, imb = fields["mid"], fields["imbalance"]
            near, far = DENSE_SLOTS - LEAD_MINUTES, DENSE_SLOTS + FOLLOW_MINUTES
            with np.errstate(divide="ignore", invalid="ignore"):
                lead_ret = (mid[:, DENSE_SLOTS] / mid[:, near] - 1.0).astype(np.float32)
                foll_ret = (mid[:, far] / mid[:, DENSE_SLOTS] - 1.0).astype(np.float32)
            lead_imb = (imb[:, DENSE_SLOTS] - imb[:, near]).astype(np.float32)
            foll_imb = (imb[:, far] - imb[:, DENSE_SLOTS]).astype(np.float32)
            lead_ret, foll_ret = lead_ret.T, foll_ret.T
            lead_imb, foll_imb = lead_imb.T, foll_imb.T
            leader_ret.append(lead_ret)
            follower_ret.append(foll_ret)
            leader_imb.append(lead_imb)
            follower_imb.append(foll_imb)
        recorder.event("select", "month_done", month=str(month), chunk=str(chunk_start.date()),
                       lag_rows=len(leader_ret), seconds=round(time.monotonic() - began, 1))
        del bundle, bars
        gc.collect()
    leader_ret, follower_ret = np.vstack(leader_ret), np.vstack(follower_ret)
    leader_imb, follower_imb = np.vstack(leader_imb), np.vstack(follower_imb)
    corr_ret = lag_correlation(leader_ret, follower_ret)
    corr_imb = lag_correlation(leader_imb, follower_imb)
    corr_same = lag_correlation(leader_ret, leader_ret)  # 15-min co-movement, documentation only
    score = 0.5 * (corr_ret + corr_imb)
    np.fill_diagonal(score, np.nan)
    coverage = np.isfinite(leader_ret).mean(axis=0) * np.isfinite(follower_ret).mean(axis=0)
    recorder.event("select", "correlation_done", lag_rows=int(len(leader_ret)),
                   instruments=int(len(instruments)), eligible=int((coverage >= 0.5).sum()),
                   coverage_median=float(np.median(coverage)),
                   corr_ret_median=float(np.nanmedian(corr_ret)),
                   corr_imb_median=float(np.nanmedian(corr_imb)))
    order = np.argsort(-np.where(np.isfinite(score), score, -np.inf), axis=1)
    neighbour = np.full((len(instruments), K_NEIGHBOURS), -1, dtype=np.int64)
    for row in range(len(instruments)):
        picked = [j for j in order[row] if j != row and np.isfinite(score[row, j])][:K_NEIGHBOURS]
        if len(picked) < K_NEIGHBOURS:
            picked += [j for j in range(len(instruments)) if j != row and j not in picked][:K_NEIGHBOURS - len(picked)]
        neighbour[row] = picked
    same_order = np.argsort(-np.where(np.isfinite(corr_same), corr_same, -np.inf), axis=1)
    top_contemporaneous = [set(row[1:21]) for row in same_order]
    overlap = float(np.mean([len(set(neighbour[row]).intersection(top_contemporaneous[row])) / K_NEIGHBOURS
                             for row in range(len(instruments))]))
    rng = np.random.default_rng(NEIGHBOUR_SEED)
    permutation = rng.permutation(len(instruments))
    neighbour_random = neighbour[permutation].copy()
    for row in range(len(instruments)):
        clashes = neighbour_random[row] == row
        if clashes.any():
            forbidden = set(neighbour_random[row].tolist()) | {row}
            pool = [j for j in range(len(instruments)) if j not in forbidden][::-1]
            neighbour_random[row, clashes] = pool[: int(clashes.sum())]

    def map_frame(matrix: np.ndarray, scores: np.ndarray | None) -> pd.DataFrame:
        frames = []
        for rank in range(K_NEIGHBOURS):
            entry = {"instrument": instruments, "rank": rank + 1,
                     "neighbour": instruments[matrix[:, rank]]}
            if scores is not None:
                entry["leadlag_score"] = scores[:, rank]
            frames.append(pd.DataFrame(entry))
        return pd.concat(frames, ignore_index=True)

    real = map_frame(neighbour, score[np.arange(len(instruments))[:, None], neighbour])
    rand = map_frame(neighbour_random, None)
    real.to_csv(run_dir / "neighbour_map.csv", index=False)
    rand.to_csv(run_dir / "neighbour_map_random.csv", index=False)
    np.save(run_dir / "leadlag_score.npy", score.astype(np.float32))
    np.save(run_dir / "contemporaneous_corr.npy", corr_same.astype(np.float32))
    pd.Series(instruments, name="instrument").to_csv(run_dir / "neighbour_universe.csv", index=False)
    frozen = {
        "schema_version": 1, "created_at": utc_now(),
        "statistic": "0.5 * (corr(ret_j[k-15,k], ret_i[k,k+30]) + corr(d_imb_j[k-15,k], d_imb_i[k,k+30]))",
        "clocks": list(CLOCKS),
        "windows": {"leader_minutes": LEAD_MINUTES, "follower_minutes": FOLLOW_MINUTES,
                    "slots_per_day": int(len(DENSE_SLOTS)),
                    "first_slots": [int(x) for x in DENSE_SLOTS[:3]],
                    "session_bounds": [list(b) for b in SESSION_BOUNDS]},
        "training_history_only": [str(TRAIN_START.date()), str(TRAIN_END.date())],
        "statistic_window": [str(select_start.date()), str(TRAIN_END.date())],
        "lag_rows": int(len(leader_ret)), "instruments": int(len(instruments)),
        "eligible_instruments": int((coverage >= 0.5).sum()), "k_neighbours": K_NEIGHBOURS,
        "permutation_seed": NEIGHBOUR_SEED, "labels_used": False,
        "post_training_data_used": False,
        "neighbour_map": {"path": "neighbour_map.csv", "sha256": file_sha256(run_dir / "neighbour_map.csv")},
        "neighbour_map_random": {"path": "neighbour_map_random.csv",
                                 "sha256": file_sha256(run_dir / "neighbour_map_random.csv")},
        "universe": {"path": "neighbour_universe.csv",
                     "sha256": file_sha256(run_dir / "neighbour_universe.csv")},
        "score_matrix": {"path": "leadlag_score.npy", "sha256": file_sha256(run_dir / "leadlag_score.npy")},
        "diagnostics": {"chosen_neighbour_score_median": float(np.nanmedian(
            score[np.arange(len(instruments))[:, None], neighbour])),
            "contemporaneous_corr_median": float(np.nanmedian(corr_same)),
            "leadlag_vs_contemporaneous_topk_overlap": float(overlap),
            "note": "the contemporaneous correlation and the overlap are documentation only; "
                    "the frozen map comes from the lead-lag statistic alone"},
    }
    write_json(run_dir / "frozen_neighbours.json", frozen)
    recorder.event("select", "frozen", sha256=frozen["neighbour_map"]["sha256"],
                   random_sha256=frozen["neighbour_map_random"]["sha256"])
    return frozen


def select_chunks(start: pd.Timestamp, end: pd.Timestamp, days_per_chunk: int):
    """Month-aligned slices of at most ``days_per_chunk`` days (memory bounded selection pass)."""
    for month, chunk_start, chunk_end in month_chunks(start, end):
        cursor = chunk_start
        while cursor < chunk_end:
            stop = min(cursor + pd.Timedelta(days=days_per_chunk), chunk_end)
            yield month, cursor, stop
            cursor = stop


def month_chunks(start: pd.Timestamp, end: pd.Timestamp):
    for month in pd.period_range(start, end - pd.Timedelta(days=1), freq="M"):
        chunk_start = max(pd.Timestamp(month.start_time), start)
        chunk_end = min(pd.Timestamp(month.end_time) + pd.Timedelta(seconds=1), end)
        if chunk_start < chunk_end:
            yield month, chunk_start, chunk_end


PD_TRAIN_START = TRAIN_START


def load_map(path: Path) -> tuple[np.ndarray, pd.Index]:
    frame = pd.read_csv(path)
    universe = pd.Index(sorted(frame["instrument"].unique()))
    codes = pd.Series(np.arange(len(universe)), index=universe)
    matrix = np.full((len(universe), K_NEIGHBOURS), -1, dtype=np.int64)
    for row in frame.itertuples(index=False):
        matrix[codes[row.instrument], row.rank - 1] = codes[row.neighbour]
    return matrix, universe


# ---------------------------------------------------------------- head
class ResidualHead:
    """45 -> 32 (GELU) -> 1 with a zero-initialised output layer (step 0 == frozen trunk)."""

    def __init__(self, n_features: int):
        import torch
        from torch import nn

        self.net = nn.Sequential(nn.Linear(n_features, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)
        self.torch = torch

    def __call__(self, x):
        return self.net(x).squeeze(-1)

    def parameters(self):
        return self.net.parameters()

    def state(self) -> dict:
        return {name: value.detach().cpu().numpy().tolist() for name, value in self.net.state_dict().items()}

    def load(self, state: dict):
        self.net.load_state_dict({name: self.torch.tensor(value, dtype=self.torch.float32)
                                  for name, value in state.items()})
        self.net.eval()

    def to(self, device):
        self.net.to(device)
        return self

    def eval(self):
        self.net.eval()
        return self

    def train(self):
        self.net.train()
        return self


def mean_rank_ic(prediction: np.ndarray, target: np.ndarray, stamps) -> float:
    frame = pd.DataFrame({"stamp": pd.to_datetime(stamps).to_numpy(), "p": prediction, "y": target})
    values = []
    for _, group in frame.groupby("stamp"):
        if len(group) < 20:
            continue
        rank_p, rank_y = group["p"].rank(), group["y"].rank()
        if rank_p.std() == 0 or rank_y.std() == 0:
            continue
        values.append(float(np.corrcoef(rank_p, rank_y)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def feature_rank_ic(features: np.ndarray, residual: np.ndarray, stamps: np.ndarray,
                    max_stamps: int = 400) -> np.ndarray:
    """Mean per-timestamp rank IC of every feature against the trunk residual (diagnostic only)."""
    frame = pd.DataFrame({"stamp": pd.to_datetime(stamps), "y": residual})
    unique = np.sort(pd.unique(frame["stamp"]))
    if len(unique) > max_stamps:
        unique = unique[np.linspace(0, len(unique) - 1, max_stamps).astype(int)]
    stamp_codes = pd.Categorical(frame["stamp"], categories=unique).codes
    values = np.zeros(len(unique))
    accum = np.zeros(features.shape[1])
    for index, stamp in enumerate(unique):
        rows = np.where(stamp_codes == index)[0]
        if len(rows) < 50:
            continue
        y_rank = pd.Series(residual[rows]).rank().to_numpy(dtype=np.float64)
        y_rank -= y_rank.mean()
        y_norm = np.sqrt((y_rank**2).sum())
        if y_norm == 0:
            continue
        block = pd.DataFrame(features[rows].astype(np.float64))
        rank = block.rank().to_numpy()
        rank -= rank.mean(axis=0)
        norm = np.sqrt((rank**2).sum(axis=0))
        norm[norm == 0] = np.nan
        accum += (rank.T @ y_rank) / (norm * y_norm)
        values[index] = 1.0
    count = max(values.sum(), 1.0)
    return accum / count


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


# ---------------------------------------------------------------- stage: train
def finite_or_none(value) -> float | None:
    """JSON keeps no NaN, so diagnostics report None instead of a non finite statistic."""
    value = float(value)
    return value if np.isfinite(value) else None


def save_cache(path: Path, features_real, features_rand, base, target, stamp, instrument) -> None:
    """Persist the frozen-trunk base scores and both neighbour feature matrices."""
    np.savez(path, features_real=features_real, features_rand=features_rand,
             base=np.asarray(base, dtype=np.float64), target=np.asarray(target, dtype=np.float64),
             stamp=pd.to_datetime(stamp).to_numpy().astype("datetime64[s]").astype(np.int64),
             instrument=np.asarray(instrument, dtype="U16"))


def load_cache(path: Path):
    """Reload a saved feature cache so a later stage never repeats the expensive pass."""
    with np.load(path) as payload:
        return (payload["features_real"], payload["features_rand"], payload["base"],
                payload["target"], pd.Series(pd.to_datetime(payload["stamp"], unit="s")),
                payload["instrument"])


def stage_train(args, run_dir: Path, recorder: Recorder) -> dict:
    torch = __import__("torch")
    frozen = read_json(run_dir / "frozen_neighbours.json")
    if file_sha256(run_dir / "neighbour_map.csv") != frozen["neighbour_map"]["sha256"]:
        raise ValueError("neighbour map changed after it was frozen")
    if file_sha256(TRUNK) != TRUNK_SHA:
        raise ValueError("frozen trunk changed")
    neighbour_real, instruments = load_map(run_dir / "neighbour_map.csv")
    neighbour_rand, universe_random = load_map(run_dir / "neighbour_map_random.csv")
    if not instruments.equals(universe_random):
        raise ValueError("random map universe differs from the real map universe")
    config = data_config()
    device = "cuda" if cuda_available() else "cpu"
    model = TLOBAdapter.load(TRUNK, device=device)
    model.config["batch_size"] = 256
    features_real, features_rand, base, target, stamps, names = [], [], [], [], [], []
    pairs = list(month_chunks(TRAIN_START, TRAIN_END))
    if args.limit_months:
        pairs = pairs[: args.limit_months]
    cached = load_cache(run_dir / "train_cache.npz") if args.reuse_cache else None
    for month, chunk_start, chunk_end in ([] if cached else pairs):
        began = time.monotonic()
        bundle = load_data(config, chunk_start, chunk_end)
        rows = select_targets(bundle.labels, {"start": TRAIN_START, "end": TRAIN_END},
                              config["label_horizon_minutes"], sample_times=config["sample_times"])
        rows = rows.loc[(rows["date"] >= chunk_start) & (rows["date"] < chunk_end)].reset_index(drop=True)
        if rows.empty:
            continue
        field_cache = build_field_cache(bundle.bars, instruments)
        features_begin = time.monotonic()
        days = np.sort(pd.unique(rows["date"].dt.normalize()))
        for offset in range(0, len(days), 6):
            block = set(pd.to_datetime(days[offset:offset + 6]))
            part = rows.loc[rows["date"].dt.normalize().isin(block)].reset_index(drop=True)
            keys = part[["date", "instrument"]]
            batch = build_batch(bundle.bars, keys, config)
            scores = model.predict(batch)
            del batch
            features_real.append(neighbour_features(keys, neighbour_real, instruments, field_cache))
            features_rand.append(neighbour_features(keys, neighbour_rand, instruments, field_cache))
            base.append(scores)
            target.append(part["target"].to_numpy(dtype=np.float64))
            stamps.append(part["date"])
            names.append(part[["date", "instrument", "target"]])
            gc.collect()
        recorder.event("train", "month_done", month=str(month), rows=int(len(rows)),
                       fields_seconds=round(features_begin - began, 1),
                       features_seconds=round(time.monotonic() - features_begin, 1),
                       seconds=round(time.monotonic() - began, 1))
        del bundle
        gc.collect()
    if cached:
        features_real, features_rand, base, target, stamp, instruments_cached = cached
        recorder.event("train", "cache_reused", rows=int(len(target)),
                       features=int(features_real.shape[1]))
        del instruments_cached
    else:
        features_real = np.vstack(features_real).astype(np.float16)
        features_rand = np.vstack(features_rand).astype(np.float16)
        base = np.concatenate(base)
        target = np.concatenate(target)
        stamp = pd.concat(stamps, ignore_index=True)
    keys = pd.concat(names, ignore_index=True).assign(base=base) if names else None
    if keys is not None:
        keys.to_parquet(run_dir / "train_keys.parquet", index=False)
        save_cache(run_dir / "train_cache.npz", features_real, features_rand, base, target, stamp,
                   keys["instrument"].to_numpy())
        recorder.event("train", "cache_built", rows=len(keys), features=int(features_real.shape[1]),
                       base_std=float(np.std(base)), target_std=float(np.std(target)))
        del keys

    day = pd.to_datetime(stamp).dt.normalize().to_numpy()
    unique_days = np.sort(np.unique(day))
    holdout = set(pd.to_datetime(unique_days[-max(1, len(unique_days) // 10):]))
    is_holdout = pd.Series(day).isin(holdout).to_numpy()
    residual = target - base
    # The features are cross-sectionally demeaned per timestamp, so the head can only speak to
    # the cross-sectional part of the residual: train on the per-timestamp demeaned residual
    # (rank IC, the quantity the evaluation rewards, is invariant to that centring).
    residual = residual - pd.DataFrame({"s": pd.to_datetime(stamp).to_numpy(), "r": residual})         .groupby("s")["r"].transform("mean").to_numpy()
    sigma = float(np.std(residual[~is_holdout]))
    y = (residual / sigma).astype(np.float32)
    train_rows = np.where(~is_holdout)[0]
    valid_rows = np.where(is_holdout)[0]
    stamp_np = pd.to_datetime(stamp).to_numpy()
    diagnostics = {}
    for arm, matrix in (("real", features_real), ("rand", features_rand)):
        ic = feature_rank_ic(matrix, residual, stamp_np)
        diagnostics[arm] = {"feature_rank_ic": ic.tolist(),
                            "feature_rank_ic_mean": float(np.mean(ic)),
                            "feature_rank_ic_absmax": float(np.max(np.abs(ic)))}
        recorder.event("train", "feature_ic", arm=arm, mean=finite_or_none(np.mean(ic)),
                       absmax=finite_or_none(np.max(np.abs(ic))),
                       undefined_features=int(np.sum(~np.isfinite(ic))))
    results, histories = {}, {}
    for arm, matrix in (("real", features_real), ("rand", features_rand)):
        torch.manual_seed(args.seed)
        head = ResidualHead(matrix.shape[1]).to(device)
        optimiser = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        x_all = torch.from_numpy(np.ascontiguousarray(matrix))
        y_all = torch.from_numpy(y)
        history, best = [], None
        for epoch in range(args.epochs):
            head.train()
            order = np.random.default_rng(args.seed + epoch).permutation(len(train_rows))
            total, seen = 0.0, 0
            for offset in range(0, len(order), args.batch_size):
                index = train_rows[order[offset:offset + args.batch_size]]
                xb = x_all[index].to(device).float()
                yb = y_all[torch.from_numpy(index)].to(device)
                loss = torch.mean((head(xb) - yb) ** 2)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                total += float(loss) * len(index)
                seen += len(index)
            head.eval()
            with torch.no_grad():
                valid_pred = head(x_all[torch.from_numpy(valid_rows)].to(device).float()).cpu().numpy()
            valid_mse = float(np.mean((valid_pred - y[valid_rows]) ** 2))
            valid_ic = mean_rank_ic(valid_pred, residual[valid_rows], stamp_np[valid_rows])
            entry = {"epoch": epoch + 1, "train_mse": total / max(seen, 1), "valid_mse": valid_mse,
                     "valid_rank_ic": valid_ic}
            history.append(entry)
            recorder.event("train", "epoch", arm=arm, **entry)
            if best is None or valid_mse < best["valid_mse"]:
                best = {**entry, "state": head.state()}
        head.load(best["state"])
        with torch.no_grad():
            full = head(x_all.to(device).float()).cpu().numpy()
        in_sample = mean_rank_ic(full[train_rows], residual[train_rows], stamp_np[train_rows])
        payload = {"schema_version": 1, "arm": arm, "features": int(matrix.shape[1]), "hidden": HIDDEN,
                   "seed": args.seed, "residual_sigma": sigma, "state": best["state"],
                   "best_epoch": best["epoch"], "valid_mse": best["valid_mse"],
                   "valid_rank_ic": best["valid_rank_ic"], "in_sample_rank_ic": in_sample,
                   "train_rows": int(len(train_rows)), "valid_rows": int(len(valid_rows)),
                   "target": "(y - frozen trunk score) demeaned per timestamp, divided by "
                             "sigma of the held-out training days",
                   "created_at": utc_now()}
        path = run_dir / f"head_{arm}.json"
        write_json(path, payload)
        results[arm] = {"path": path.name, "sha256": file_sha256(path), "best_epoch": best["epoch"],
                        "valid_mse": best["valid_mse"], "valid_rank_ic": best["valid_rank_ic"],
                        "in_sample_rank_ic": in_sample, "residual_sigma": sigma}
        histories[arm] = history
        del x_all, head
        gc.collect()
    write_json(run_dir / "head_training.json", {
        "arms": results, "histories": histories, "seed": args.seed, "epochs": args.epochs,
        "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "holdout_days": int(len(holdout)), "rows": int(len(target)),
        "device": device, "diagnostics": diagnostics})
    return results


# ---------------------------------------------------------------- stage: predict
def stage_predict(args, run_dir: Path, recorder: Recorder) -> dict:
    torch = __import__("torch")
    frozen = read_json(run_dir / "frozen_neighbours.json")
    if file_sha256(run_dir / "neighbour_map.csv") != frozen["neighbour_map"]["sha256"]:
        raise ValueError("neighbour map changed after it was frozen")
    training = read_json(run_dir / "head_training.json")
    config = data_config()
    neighbour_real, instruments = load_map(run_dir / "neighbour_map.csv")
    neighbour_rand, _ = load_map(run_dir / "neighbour_map_random.csv")
    heads = {}
    for arm in ("real", "rand"):
        payload = read_json(run_dir / f"head_{arm}.json")
        if file_sha256(run_dir / f"head_{arm}.json") != training["arms"][arm]["sha256"]:
            raise ValueError(f"head_{arm}.json changed after training")
        head = ResidualHead(payload["features"])
        head.load(payload["state"])
        head.eval()
        heads[arm] = (head, payload)
    (run_dir / "predictions").mkdir(exist_ok=True)
    windows = []
    for window in sorted(REPLAY_ROOT.glob("windows/*")):
        name = window.name
        if args.windows and name not in args.windows:
            continue
        keys = pd.read_parquet(window / "keys.parquet")
        base = pd.read_parquet(window / "r441_s11" / "predictions.parquet")
        frame = keys.merge(base, on=["date", "instrument"], how="left", validate="one_to_one")
        if frame["score"].isna().any():
            raise ValueError(f"{name}: base prediction missing for some dev keys")
        began = time.monotonic()
        start = pd.Timestamp(frame["date"].min()).normalize() - pd.Timedelta(days=1)
        end = pd.Timestamp(frame["date"].max()).normalize() + pd.Timedelta(days=1)
        bundle = load_data(config, start, end)
        loaded = time.monotonic()
        field_cache = build_field_cache(bundle.bars, instruments)
        delta = {}
        for arm, matrix in (("real", neighbour_real), ("rand", neighbour_rand)):
            features = neighbour_features(frame[["date", "instrument"]], matrix, instruments, field_cache)
            with torch.no_grad():
                out = head_call(heads[arm][0], torch, features)
            delta[arm] = stamp_z(out, frame["date"])
        scored = time.monotonic()
        del bundle
        gc.collect()
        store = frame[["date", "instrument"]].assign(
            base=frame["score"].to_numpy(), base_z=stamp_z(frame["score"].to_numpy(), frame["date"]),
            delta_real_z=delta["real"], delta_rand_z=delta["rand"])
        store.to_parquet(run_dir / "predictions" / f"{name}.parquet", index=False)
        windows.append({"window": name, "rows": int(len(store)),
                        "days": int(pd.to_datetime(store["date"]).dt.normalize().nunique()),
                        "load_seconds": round(loaded - began, 1),
                        "feature_seconds": round(scored - loaded, 1),
                        "rows_per_second": round(len(store) / max(scored - loaded, 1e-6), 0)})
        recorder.event("predict", "window_done", **windows[-1])
    write_json(run_dir / "predict_timing.json", {"windows": windows,
                                                 "note": "base score is the archived dev10 replay "
                                                         "r441_s11 prediction; only feature extraction "
                                                         "and the head forward are timed here"})
    return {"windows": windows}


def head_call(head, torch, features: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(features)).float()
    return head(tensor).cpu().numpy().astype(np.float64)


def stamp_z(values: np.ndarray, stamps: pd.Series) -> np.ndarray:
    frame = pd.DataFrame({"stamp": pd.to_datetime(stamps).to_numpy(), "value": values})
    grouped = frame.groupby("stamp")["value"]
    mean = grouped.transform("mean").to_numpy()
    std = grouped.transform("std").to_numpy()
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    return np.clip((values - mean) / std, -5.0, 5.0)


# ---------------------------------------------------------------- stage: evaluate
def stage_evaluate(args, run_dir: Path, recorder: Recorder) -> dict:
    evaluation = read_json(PLAN)["evaluation"]
    windows = [path.stem for path in sorted((run_dir / "predictions").glob("*.parquet"))]
    daily_rows, metric_rows = [], []
    for name in windows:
        frame = pd.read_parquet(run_dir / "predictions" / f"{name}.parquet")
        targets = pd.read_parquet(REPLAY_ROOT / "windows" / name / "targets.parquet")
        arms = {"base": frame["base_z"].to_numpy()}
        for weight in WEIGHT_GRID:
            arms[f"real_w{weight:.2f}"] = frame["base_z"].to_numpy() + weight * frame["delta_real_z"].to_numpy()
            arms[f"rand_w{weight:.2f}"] = frame["base_z"].to_numpy() + weight * frame["delta_rand_z"].to_numpy()
        for arm, score in arms.items():
            scored = frame[["date", "instrument"]].assign(score=score)
            scored = scored.merge(targets[["date", "instrument"]], on=["date", "instrument"],
                                  how="inner", validate="one_to_one")
            metrics, daily, _ = evaluate(
                scored, targets, evaluation, return_sections=True,
                undefined_constant_portfolios=evaluation.get("constant_portfolios") == "undefined")
            metrics = {key: value for key, value in metrics.items() if np.isscalar(value)}
            metrics.update({"window": name, "arm": arm})
            metric_rows.append(metrics)
            daily_rows.append(daily.assign(window=name, arm=arm))
    stamps = pd.concat(daily_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    stamps.to_csv(run_dir / "timestamp_by_arm.csv", index=False)
    daily = (stamps.groupby(["window", "arm", "date"], as_index=False)
             [["rank_ic", "long_short_return", "turnover"]].mean())
    daily.to_csv(run_dir / "daily_by_arm.csv", index=False)
    metrics.to_csv(run_dir / "window_metrics.csv", index=False)
    base_daily = daily.loc[daily.arm == "base"].set_index(["window", "date"])
    paired = []
    for arm in [f"{kind}_w{weight:.2f}" for kind in ("real", "rand") for weight in WEIGHT_GRID]:
        subject = daily.loc[daily.arm == arm].set_index(["window", "date"])
        joined = subject.join(base_daily, how="inner", rsuffix="_base")
        in_early = np.array([window in EARLY_BLOCK for window in joined.index.get_level_values(0)])
        for metric in ("rank_ic", "long_short_return", "turnover"):
            diff = (joined[metric] - joined[f"{metric}_base"]).to_numpy(dtype=float)
            paired.append({"arm": arm, "metric": metric, "days": int(np.isfinite(diff).sum()),
                           "all_mean": _safe_mean(diff), "early_mean": _safe_mean(diff[in_early]),
                           "late_mean": _safe_mean(diff[~in_early]),
                           "late_t": _block_t(diff[~in_early], 5)})
    paired = pd.DataFrame(paired)
    paired.to_csv(run_dir / "paired_increments.csv", index=False)
    candidates = paired.loc[(paired.arm.str.startswith("real")) & (paired.metric == "rank_ic")]
    chosen = candidates.sort_values("early_mean", ascending=False).iloc[0]
    weight = float(chosen.arm.split("_w")[1])
    late = paired.loc[paired.metric.isin(["rank_ic", "long_short_return", "turnover"]) &
                      paired.arm.isin([f"real_w{weight:.2f}", f"rand_w{weight:.2f}"])]
    summary = {"frozen_weight": {"rule": "argmax early-block mean daily paired rank_ic gain",
                                 "early_block": list(EARLY_BLOCK), "weight": weight,
                                 "early_rank_ic_gain": float(chosen.early_mean)},
               "late_blocks": late.to_dict(orient="records"),
               "paired_table": paired.to_dict(orient="records"),
               "windows": windows}
    write_json(run_dir / "results.json", summary)
    recorder.event("evaluate", "paired", weight=weight,
                   late=late.to_dict(orient="records"))
    return summary


def _safe_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def _block_t(values: np.ndarray, block: int) -> float:
    """t-statistic of the mean over contiguous day blocks (day-correlation aware, coarse)."""
    values = values[np.isfinite(values)]
    if len(values) < 2 * block:
        return float("nan")
    n_blocks = len(values) // block
    trimmed = values[: n_blocks * block].reshape(n_blocks, block).mean(axis=1)
    if trimmed.std(ddof=1) == 0:
        return float("nan")
    return float(trimmed.mean() / (trimmed.std(ddof=1) / np.sqrt(n_blocks)))


def prepare_multistage_run(run_dir: Path) -> None:
    """Let several stages share one run directory.

    The audit registry keys ``runs`` by directory name, so a second stage in the same
    directory would fail to register. Archive the earlier manifest and release its row;
    every artifact of the earlier stage stays in place next to the archived manifest.
    """
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        return
    previous = read_json(manifest)
    archived = run_dir / f"manifest_stage_{previous.get('stages', ['unknown'])[-1]}.json"
    manifest.replace(archived)
    registry = ROOT / "runs" / "audit.sqlite3"
    if registry.exists():
        with sqlite3.connect(registry) as connection:
            connection.execute("DELETE FROM runs WHERE run_id = ?", (run_dir.name,))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=["select", "train", "predict", "evaluate", "all"])
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--output-root", default=str(ROOT / "runs" / "r11-neighbours-20261028"))
    parser.add_argument("--limit-months", type=int, default=0)
    parser.add_argument("--select-months", type=int, default=12,
                        help="trailing months of the training window used for the neighbour statistic")
    parser.add_argument("--chunk-days", type=int, default=10,
                        help="selection pass chunk length in days (bounds peak memory)")
    parser.add_argument("--windows", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--reuse-cache", action="store_true",
                        help="train from an existing train_cache.npz instead of re-extracting")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else new_run_dir(Path(args.output_root))
    run_dir.mkdir(parents=True, exist_ok=True)
    prepare_multistage_run(run_dir)
    recorder = Recorder(run_dir, "r11-neighbours")
    recorder.event("run", "started", requested=args.stage, run_dir=str(run_dir), limit_months=args.limit_months)
    started = time.monotonic()
    stages = ["select", "train", "predict", "evaluate"] if args.stage == "all" else [args.stage]
    for stage in stages:
        recorder.event("run", "stage_started", name=stage)
        {"select": select_map, "train": stage_train, "predict": stage_predict,
         "evaluate": stage_evaluate}[stage](args, run_dir, recorder)
        recorder.event("run", "stage_completed", name=stage)
    write_json(run_dir / "manifest.json", {
        "schema_version": 1, "experiment_id": "r11-neighbours", "run_id": run_dir.name,
        "stages": stages, "created_at": utc_now(),
        "trunk": {"path": str(TRUNK.relative_to(ROOT)), "sha256": TRUNK_SHA},
        "base_predictions": str(REPLAY_ROOT.relative_to(ROOT)),
        "source": capture_source(run_dir)})
    finish(run_dir, "completed", 0, resources={"wall_seconds": time.monotonic() - started})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
