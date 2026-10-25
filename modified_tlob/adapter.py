"""Batch training/inference and portable checkpoints with frozen preprocessing."""

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .device import device_resources, select_device
from .model import ModifiedTLOB, OpenResidualHead
from .schema import validate_input_schema

PREPROCESS_KEYS = ("feature_columns", "lookback", "min_history", "missing_policy", "max_ffill_steps",
                   "missing_sentinel", "price_scale", "history_days", "sample_times", "label_horizon_minutes")
DATE_DECAY_HALF_LIFE_TRADING_DAYS = 10.0
TARGET_CENTERING_MODES = ("none", "timestamp_mean_train", "timestamp_zscore_train_rms_matched")
TARGET_LABELS = {
    "none": "official_vwap_return",
    "timestamp_mean_train": "official_vwap_return_minus_training_timestamp_mean",
    "timestamp_zscore_train_rms_matched": "official_vwap_return_timestamp_zscore_train_rms_matched",
}
PREDICTION_UNITS = {"none": "original_label_return", "timestamp_mean_train": "relative_return_score",
                    "timestamp_zscore_train_rms_matched": "relative_return_score"}
TRAINING_LOSS_UNITS = {
    "none": "original_return_squared",
    "timestamp_mean_train": "centered_return_squared",
    "timestamp_zscore_train_rms_matched": "rms_matched_transformed_return_squared",
}
PROGRESS_LOSS_UNITS = {"none": "raw_label_mse", "timestamp_mean_train": "centered_label_mse",
                       "timestamp_zscore_train_rms_matched": "rms_matched_transformed_label_mse"}


def timestamp_zscore_train_rms_matched(keys, target, mask):
    """Cross-sectional z-score labels rescaled to the centered training-pool variance.

    For every complete training timestamp ``t``, with ``r = float64(float32(y))``:
    ``z = r - mean_t(r)``, ``sigma_t = sqrt(mean_t(z**2))`` (ddof=0), and
    ``q = z / sigma_t`` (or ``0`` when ``sigma_t == 0``). One pool constant
    ``c = std(z) / std(q)`` (ddof=0) makes ``y* = float32(c q)`` share the centered
    labels' pool standard deviation. The fallback is the centered-label mean, matching
    the centered control; it is never re-estimated from the transformed targets.
    """
    raw64 = np.asarray(target, dtype=np.float32).astype(np.float64)
    if len(keys) != len(raw64):
        raise ValueError("Target-transform keys and targets are not aligned")
    if not np.isfinite(raw64).all():
        raise ValueError("Target transform requires finite official labels")
    timestamps = pd.to_datetime(keys["date"])
    if timestamps.isna().any():
        raise ValueError("Target transform requires a timestamp for every training row")
    frame = pd.DataFrame({"timestamp": timestamps.to_numpy(), "raw": raw64})
    grouped = frame.groupby("timestamp", sort=False)["raw"]
    counts = grouped.size()
    means = grouped.transform("mean").to_numpy(dtype=np.float64)
    centered = raw64 - means
    timestamp_means = grouped.mean()
    squares = pd.DataFrame({"timestamp": frame["timestamp"], "square": centered ** 2}).groupby(
        "timestamp", sort=False)["square"].sum().reindex(counts.index)
    stds = np.sqrt(squares.to_numpy(dtype=np.float64) / counts.to_numpy(dtype=np.float64))
    std_rows = pd.Series(stds, index=counts.index).reindex(frame["timestamp"]).to_numpy(dtype=np.float64)
    q = np.divide(centered, std_rows, out=np.zeros_like(centered), where=std_rows > 0)
    centered_std = float(centered.std())
    q_std = float(q.std())
    if not np.isfinite([centered_std, q_std]).all() or q_std <= 0:
        raise ValueError("Timestamp-zscore training labels have zero or nonfinite pool q standard deviation")
    c = centered_std / q_std
    transformed64 = c * q
    transformed = transformed64.astype(np.float32)
    if not np.isfinite(transformed).all():
        raise FloatingPointError("Nonfinite rms-matched training targets")
    centered_means = pd.DataFrame({"timestamp": frame["timestamp"], "centered": centered}).groupby(
        "timestamp", sort=False)["centered"].mean()
    zero_variance = [str(timestamp) for timestamp, std in zip(counts.index, stds) if std == 0]
    rows = [{"timestamp": str(timestamp), "samples": int(counts.loc[timestamp]),
             "mean": float(timestamp_means.loc[timestamp]), "std": float(std)}
            for timestamp, std in zip(counts.index, stds)]
    selected = np.ones(len(raw64), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    stats = {
        "method": "timestamp_zscore_train_rms_matched",
        "formula": "q = (r - mean_t(r)) / std_t(r); y* = c q with c = std(z) / std(q), all ddof=0",
        "label_source": "official training labels only",
        "scope": "every row of each full training timestamp in the selected training pool; computed once before "
                 "mini-batches and before the min-history mask",
        "ddof": 0,
        "timestamps": int(len(counts)), "rows": int(len(raw64)),
        "min_rows_per_timestamp": int(counts.min()),
        "mask_excluded_rows": int((~selected).sum()),
        "raw_mean": float(raw64.mean()), "raw_std": float(raw64.std()),
        "centered_mean": float(centered.mean()), "centered_std": centered_std,
        "q_mean": float(q.mean()), "q_std": q_std,
        "c": float(c),
        "c_formula": "std(centered training labels, ddof=0) / std(q, ddof=0) over the complete training pool",
        "transformed_mean": float(transformed64.mean()), "transformed_std": float(c * q_std),
        "float32_transformed_mean": float(transformed.astype(np.float64).mean()),
        "float32_transformed_std": float(transformed.astype(np.float64).std()),
        "selected_centered_mean": float(centered[selected].mean()),
        "selected_centered_std": float(centered[selected].std()),
        "selected_transformed_std": float(transformed[selected].astype(np.float64).std()),
        "fallback_mean": float(centered[selected].astype(np.float32).astype(np.float64).mean()),
        "fallback_basis": "timestamp-centered training-label mean in the same float32 representation used by "
                          "the centered control; never re-estimated from the transformed targets",
        "zero_variance_timestamps": int(len(zero_variance)),
        "zero_variance_timestamp_list": zero_variance,
        "max_abs_centered_timestamp_mean": float(np.abs(centered_means.to_numpy()).max()),
        "timestamp_means_sha256": hashlib.sha256(json.dumps(
            [{"timestamp": str(timestamp), "mean": float(value)}
             for timestamp, value in timestamp_means.items()], sort_keys=True, ensure_ascii=False,
            separators=(",", ":")).encode("utf-8")).hexdigest(),
        "per_timestamp_sha256": hashlib.sha256(json.dumps(
            rows, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "per_timestamp": rows,
    }
    if stats["max_abs_centered_timestamp_mean"] > 1e-12:
        raise ValueError("Timestamp-centered training labels are not zero-mean within a timestamp")
    if abs(stats["transformed_std"] - centered_std) > 1e-12 * max(centered_std, 1.0):
        raise ValueError("RMS-matched training labels do not match the centered pool standard deviation")
    return transformed[selected], stats



OPEN_RESIDUAL_METHOD = "opening_0930_additive_residual"
OPEN_SAMPLE_TIME = "09:30"


def open_market_mask(keys, sample_time=OPEN_SAMPLE_TIME):
    """Boolean opening-section marker parsed from the exchange-local prediction keys."""
    if len(sample_time) != 5 or sample_time[2] != ":":
        raise ValueError("Opening marker requires an HH:MM sample time")
    timestamps = pd.to_datetime(keys["date"])
    if timestamps.isna().any():
        raise ValueError("Opening marker requires a timestamp for every prediction key")
    if isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError("Opening marker requires timezone-naive exchange-local timestamps")
    return timestamps.dt.strftime("%H:%M").eq(sample_time).to_numpy()


def frozen_stats_sha256(stats):
    digest = hashlib.sha256()
    digest.update(json.dumps({"c": float(stats["c"]), "per_timestamp": stats["per_timestamp"]},
                             sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def apply_frozen_timestamp_zscore(keys, target, stats):
    """Map official labels into a checkpoint's frozen cross-sectional target space.

    Reuses the saved per-timestamp means/stds and pool constant ``c`` verbatim:
    ``r = float64(float32(y))``, ``q = (r - mean_t) / std_t`` (0 where ``std_t == 0``)
    and ``y* = float32(c q)``. No statistic is re-estimated, so the residual head
    learns in exactly the units of the trunk whose output it corrects.
    """
    if not isinstance(stats, dict) or stats.get("method") != "timestamp_zscore_train_rms_matched":
        raise ValueError("Frozen target transform requires saved cross-sectional z-score statistics")
    rows = stats.get("per_timestamp")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Frozen target transform is missing its per-timestamp statistics")
    c = float(stats["c"])
    if not np.isfinite(c) or c <= 0:
        raise ValueError("Frozen target transform requires a positive finite pool constant")
    raw64 = np.asarray(target, dtype=np.float32).astype(np.float64)
    if len(keys) != len(raw64):
        raise ValueError("Frozen target-transform keys and targets are not aligned")
    if not np.isfinite(raw64).all():
        raise ValueError("Frozen target transform requires finite official labels")
    timestamps = pd.to_datetime(keys["date"])
    if timestamps.isna().any() or isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError("Frozen target transform requires timezone-naive timestamps for every row")
    table = {pd.Timestamp(row["timestamp"]): row for row in rows}
    means, stds = np.empty(len(raw64)), np.empty(len(raw64))
    for index, timestamp in enumerate(timestamps):
        row = table.get(timestamp)
        if row is None:
            raise ValueError(f"Frozen target transform has no statistics for {timestamp}")
        means[index], stds[index] = float(row["mean"]), float(row["std"])
    centered = raw64 - means
    q = np.divide(centered, stds, out=np.zeros_like(centered), where=stds > 0)
    transformed = (c * q).astype(np.float32)
    if not np.isfinite(centered).all() or not np.isfinite(transformed).all():
        raise FloatingPointError("Nonfinite frozen rms-matched training targets")
    return transformed, {
        "method": "frozen_timestamp_zscore_train_rms_matched",
        "source": "checkpoint target_centering_stats reused verbatim; no statistic re-estimated",
        "c": c, "rows": int(len(raw64)), "timestamps": int(timestamps.nunique()),
        "zero_variance_timestamps": int((stds == 0).sum()),
        "stats_sha256": frozen_stats_sha256(stats),
        "checkpoint_per_timestamp_sha256": stats.get("per_timestamp_sha256"),
        "transformed_mean": float(transformed.astype(np.float64).mean()),
        "transformed_std": float(transformed.astype(np.float64).std()),
    }


def state_dict_sha256(state):
    """Stable digest of a tensor state dict, used to prove frozen weights are unchanged."""
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(state[name].detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


class TLOBAdapter:
    def __init__(self, model_config, data_config, *, seed=42, recorder=None):
        self.config = model_config.model_dump() if hasattr(model_config, "model_dump") else dict(model_config)
        self.config.setdefault("target_scaling", "none")
        self.config.setdefault("target_centering", "none")
        self.config.setdefault("max_steps", None)
        self.config.setdefault("time_loss_weighting", "none")
        self.config.setdefault("book_representation", "raw")
        self.config.setdefault("trade_features", "zero")
        if self.config["target_scaling"] not in ("none", "train_std"):
            raise ValueError("Unsupported target scaling")
        if self.config["target_centering"] not in TARGET_CENTERING_MODES:
            raise ValueError("Unsupported target centering")
        if self.config["time_loss_weighting"] not in ("none", "train_variance", "date_decay_10d"):
            raise ValueError("Unsupported loss weighting")
        data = data_config.model_dump(mode="json") if hasattr(data_config, "model_dump") else dict(data_config)
        self.preprocessing = {k: data[k] for k in PREPROCESS_KEYS}
        self.training_sample_times = tuple(data.get("training_sample_times") or data["sample_times"])
        fields = tuple(self.preprocessing["feature_columns"])
        self.lob_feature_count = validate_input_schema(
            fields, self.config["book_representation"], self.config["trade_features"])
        self.new_input_contract = self.config["book_representation"] != "raw" or len(fields) == 15
        self.seed, self.recorder = seed, recorder
        self.device = select_device(self.config["device"], recorder)
        torch.set_num_threads(self.config["cpu_threads"])
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        self.architecture = {k: self.config[k] for k in ("hidden_dim", "num_layers", "num_heads", "is_sin_emb")}
        self.architecture.update(seq_size=data["lookback"], num_features=len(fields))
        self.network = ModifiedTLOB(**self.architecture).to(self.device)
        self.intercept = None
        self.target_scale = 1.0
        self.target_centering_stats = None
        self.time_loss_weighting_stats = None
        self.open_residual_head = None
        self.open_residual_metadata = None
        self.trained = False
        if recorder:
            recorder.event("training", "determinism", seed=seed, algorithms=True, cpu_threads=torch.get_num_threads(),
                           device=str(self.device),
                           cuda_matmul_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                           note="GPU tolerance-based reproducibility; cross-device rounding may differ")

    def _input(self, batch, *, sample_mask=None):
        x = np.asarray(batch.x, dtype=np.float32)
        expected = (len(batch.keys), self.architecture["seq_size"], self.architecture["num_features"])
        if x.shape != expected or batch.observed.shape != x.shape or not np.isfinite(x).all():
            raise ValueError("Batch requires aligned finite preprocessed LOB windows")
        if self.architecture["num_features"] == 15 and (x[..., 12:] < 0).any():
            raise ValueError("Trade flow inputs must be nonnegative after preprocessing")
        # Validate the entire pool before selecting. Boolean indexing already owns
        # its result, so raw-book trade training needs only one window allocation.
        select_first = (sample_mask is not None and self.architecture["num_features"] == 15
                        and self.config["book_representation"] == "raw")
        if select_first:
            x = x[sample_mask]
        if self.new_input_contract:
            if not select_first:
                x = x.copy()
            if self.config["book_representation"] == "relative_mid_bps":
                # One anchor per trailing window preserves its relative price path.
                prices = x[..., :self.lob_feature_count:2].astype(np.float64)
                best = prices[:, -1, :2]
                mid = best.mean(axis=1)[:, None, None]
                valid = (prices > 0) & (best > 0).all(axis=1)[:, None, None]
                relative = np.zeros_like(prices)
                np.divide(prices, mid, out=relative, where=valid)
                x[..., :self.lob_feature_count:2] = np.where(valid, (relative - 1) * 10000, 0)
            if self.architecture["num_features"] == 15:
                flows = x[..., 12:]
                if self.config["trade_features"] == "log1p":
                    np.log1p(flows, out=flows)
                else:
                    flows.fill(0)
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite transformed LOB inputs")
        return x if sample_mask is None or select_first else x[sample_mask]

    def _event(self, stage, event, **context):
        if self.recorder:
            self.recorder.event(stage, event, **context)

    def _fit_date_decay_weights(self, keys):
        """Exponential recency weights by training-day age, normalized to mean one.

        Age zero is the newest training date; one age unit is one ordered date in the
        selected training pool. The half-life is fixed at ten trading days.
        """
        dates = pd.to_datetime(keys.date).dt.normalize()
        counts = dates.value_counts().sort_index()
        ordered = counts.index
        ages = np.arange(len(ordered) - 1, -1, -1, dtype=np.float64)
        day_weights = 2.0 ** (-ages / DATE_DECAY_HALF_LIFE_TRADING_DAYS)
        raw = dates.map(pd.Series(day_weights, index=ordered)).to_numpy(dtype=np.float64)
        sample_mean = float(raw.mean())
        if not np.isfinite(raw).all() or not np.isfinite(sample_mean) or sample_mean <= 0:
            raise ValueError("Invalid date-decay loss weights")
        weights = (raw / sample_mean).astype(np.float32)
        total = float(weights.sum(dtype=np.float64))
        by_date = []
        for position, date in enumerate(ordered):
            selected = (dates == date).to_numpy()
            weight = float(weights[selected][0])
            by_date.append({"date": str(date.date()),
                            "age_trading_days": int(len(ordered) - 1 - position),
                            "samples": int(selected.sum()), "weight": weight,
                            "weight_share": weight * int(selected.sum()) / total})
        recent = [row for row in by_date if row["age_trading_days"] < 10]
        self.time_loss_weighting_stats = {
            "method": "date_decay_10d",
            "half_life_trading_days": DATE_DECAY_HALF_LIFE_TRADING_DAYS,
            "normalization": "mean_one_over_selected_training_samples",
            "samples": int(len(raw)), "days": int(len(ordered)),
            "oldest_age_trading_days": int(len(ordered) - 1),
            "mean_weight": float(weights.mean(dtype=np.float64)), "weight_sum": total,
            "recent_days": 10,
            "recent_weight_share": float(sum(row["weight_share"] for row in recent)),
            "by_date": by_date}
        self._event("training", "time_loss_weighting_fitted", **self.time_loss_weighting_stats)
        return weights

    def _fit_time_loss_weights(self, keys, target):
        """Freeze loss weights using only the selected training rows."""
        self.time_loss_weighting_stats = None
        mode = self.config["time_loss_weighting"]
        if mode == "none":
            return None
        if mode == "date_decay_10d":
            return self._fit_date_decay_weights(keys)
        times = keys.date.dt.strftime("%H:%M").to_numpy()
        if set(times) != set(self.training_sample_times):
            raise ValueError("Time loss weighting requires every configured training sample time and no other times")
        target = target.astype(np.float64)
        variance = float(target.var())
        epsilon = .05 * variance
        by_time = {}
        for sample_time in self.training_sample_times:
            selected = times == sample_time
            slot_variance = float(target[selected].var())
            by_time[sample_time] = {
                "samples": int(selected.sum()), "days": int(keys.date[selected].dt.normalize().nunique()),
                "variance": slot_variance, "weight": 1.0 if variance == 0 else 1 / (slot_variance + epsilon)}
        mean_weight = sum(row["weight"] * row["samples"] for row in by_time.values()) / len(target)
        for row in by_time.values():
            row["weight"] /= mean_weight
        self.time_loss_weighting_stats = {
            "method": "train_variance", "samples": len(target), "days": int(keys.date.dt.normalize().nunique()),
            "variance": variance, "epsilon_factor": .05, "epsilon": epsilon,
            "constant_target_fallback": variance == 0, "by_time": by_time}
        self._event("training", "time_loss_weighting_fitted", **self.time_loss_weighting_stats)
        return np.asarray([by_time[sample_time]["weight"] for sample_time in times], dtype=np.float32)

    def _csz_training_targets(self, keys, raw64, mask):
        """Fit the frozen cross-sectional z-score transform and expose its statistics."""
        selected, stats = timestamp_zscore_train_rms_matched(keys, raw64, mask)
        self.target_centering_stats = stats
        self._event("training", "target_centering_fitted", **stats)
        return selected, stats

    def _training_targets(self, keys, target, mask):
        """Center official labels within each full training timestamp.

        The mean is computed once from every row of the selected training pool at a
        timestamp, before mini-batches and before the min-history mask. The mask only
        chooses which centered labels enter the optimizer. The standard deviation that
        drives ``target_std`` remains the raw label standard deviation, so centering
        does not also change the target scale. Inference never sees this label vector,
        the timestamp means, or any future cross-section.
        """
        raw = np.asarray(target, dtype=np.float32)
        raw64 = raw.astype(np.float64)
        if self.config["target_centering"] == "timestamp_zscore_train_rms_matched":
            return self._csz_training_targets(keys, raw64, mask)
        self.target_centering_stats = None
        if self.config["target_centering"] == "none":
            return raw[mask], None
        if len(keys) != len(raw64):
            raise ValueError("Centering keys and targets are not aligned")
        timestamps = pd.to_datetime(keys["date"])
        if timestamps.isna().any():
            raise ValueError("Centering requires a timestamp for every training row")
        frame = pd.DataFrame({"timestamp": timestamps.to_numpy(), "target": raw64})
        grouped = frame.groupby("timestamp", sort=False)["target"]
        means = grouped.transform("mean").to_numpy()
        centered = raw64 - means
        timestamp_means = grouped.mean()
        centered_means = pd.DataFrame({"timestamp": frame["timestamp"], "target": centered}).groupby(
            "timestamp", sort=False)["target"].mean()
        counts = grouped.size()
        if not np.isfinite(means).all() or not np.isfinite(centered).all():
            raise FloatingPointError("Nonfinite timestamp-centered training targets")
        values = timestamp_means.to_numpy(dtype=np.float64)
        stats = {
            "method": "timestamp_mean_train",
            "label_source": "official training labels only",
            "scope": "every row of each full timestamp in the selected training pool; computed once before "
                     "mini-batches and before the min-history mask",
            "timestamps": int(len(timestamp_means)), "rows": int(len(raw64)),
            "min_rows_per_timestamp": int(counts.min()),
            "mask_excluded_rows": int((~np.asarray(mask, dtype=bool)).sum()),
            "raw_mean": float(raw64.mean()), "raw_std": float(raw64.std()),
            "centered_mean": float(centered.mean()), "centered_std": float(centered.std()),
            "selected_centered_mean": float(centered[mask].mean()),
            "selected_centered_std": float(centered[mask].std()),
            "mean_abs_timestamp_mean": float(np.abs(values).mean()),
            "rms_timestamp_mean": float(np.sqrt(np.mean(values ** 2))),
            "max_abs_timestamp_mean": float(np.abs(values).max()),
            "max_abs_centered_timestamp_mean": float(np.abs(centered_means.to_numpy()).max()),
            "timestamp_means_sha256": hashlib.sha256(json.dumps(
                [{"timestamp": str(timestamp), "mean": float(value)}
                 for timestamp, value in timestamp_means.items()], sort_keys=True).encode("utf-8")).hexdigest(),
        }
        if stats["max_abs_centered_timestamp_mean"] > 1e-12:
            raise ValueError("Timestamp-centered training labels are not zero-mean within a timestamp")
        self.target_centering_stats = stats
        self._event("training", "target_centering_fitted", **stats)
        return centered[mask].astype(np.float32), stats

    def fit(self, batch, target, *, sample_mask=None, validation=None, observer=None,
            observer_every_steps=None):
        if observer_every_steps is not None and (type(observer_every_steps) is not int or observer_every_steps < 1):
            raise ValueError("observer_every_steps must be a positive integer or None")
        y = np.asarray(target, dtype=np.float32)
        if y.shape != (len(batch.keys),) or not len(y) or not np.isfinite(y).all():
            raise ValueError("Training requires nonempty aligned finite targets")
        mask = np.ones(len(y), dtype=bool) if sample_mask is None else np.asarray(sample_mask)
        if mask.dtype != bool or mask.shape != y.shape or not mask.any():
            raise ValueError("Training sample mask must select aligned samples")
        x = self._input(batch, sample_mask=mask)
        y, centering_stats = self._training_targets(batch.keys, y, mask)
        time_weights = self._fit_time_loss_weights(batch.keys.loc[mask], y)
        if validation is not None and self.config["target_centering"] != "none":
            raise ValueError("Timestamp-mean training requires timestamp-centered validation targets")
        validation_inputs = None
        if validation is not None:
            vb, vy = validation
            vx, vy = self._input(vb), np.asarray(vy, dtype=np.float32)
            if vy.shape != (len(vx),) or not len(vy) or not np.isfinite(vy).all():
                raise ValueError("Validation requires nonempty aligned finite targets")
            validation_inputs = vx, vy
        raw_selected = np.asarray(target, dtype=np.float32)[mask].astype(np.float64)
        training_std = float(raw_selected.std())
        self.intercept = (float(centering_stats["fallback_mean"]) if centering_stats is not None
                          and "fallback_mean" in centering_stats else float(y.astype(np.float64).mean()))
        self.target_scale = training_std if self.config["target_scaling"] == "train_std" and training_std > 0 else 1.0
        optimization_targets = (y if self.target_scale == 1.0 else
                                (y.astype(np.float64) / self.target_scale).astype(np.float32))
        if not np.isfinite(optimization_targets).all():
            raise FloatingPointError("Nonfinite scaled training targets")
        self._event("training", "target_scaling_fitted", method=self.config["target_scaling"],
                    target_scale=self.target_scale, training_std=training_std, samples=len(y),
                    constant_target_fallback=self.config["target_scaling"] == "train_std" and training_std == 0,
                    centering=self.config["target_centering"],
                    centered_training_std=(None if centering_stats is None
                                           else centering_stats["selected_centered_std"]),
                    centering_stats=centering_stats,
                    loss_units=("centered_return_squared" if centering_stats is not None
                                else "original_return_squared"))
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config["learning_rate"])
        generator = torch.Generator().manual_seed(self.seed)
        started, steps, samples_seen, epochs_completed = time.monotonic(), 0, 0, 0
        last_observed_step = None

        def observe(epoch):
            nonlocal last_observed_step
            if observer is None or last_observed_step == steps:
                return
            modes = [(module, module.training) for module in self.network.modules()]
            numpy_state, python_state = np.random.get_state(), random.getstate()
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()
                       ] if self.device.type == "cuda" else []
            try:
                with torch.random.fork_rng(devices=devices):
                    observer(self, epoch, steps)
            finally:
                for module, training in modes:
                    module.training = training
                np.random.set_state(numpy_state)
                random.setstate(python_state)
            last_observed_step = steps

        observe(0)
        for epoch in range(self.config["epochs"]):
            self.network.train()
            order = torch.randperm(len(y), generator=generator).numpy()
            total = 0.0
            weighted_total = 0.0
            optimization_total = 0.0
            epoch_samples = 0
            for offset in range(0, len(y), self.config["batch_size"]):
                indices = order[offset:offset + self.config["batch_size"]]
                xb = torch.from_numpy(x[indices]).to(self.device)
                yb = torch.from_numpy(optimization_targets[indices]).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                predictions = self.network(xb)
                unweighted_loss = torch.nn.functional.mse_loss(predictions, yb)
                if time_weights is None:
                    loss = unweighted_loss
                else:
                    weights = torch.from_numpy(time_weights[indices]).to(self.device)
                    loss = (weights * (predictions - yb).square()).sum() / weights.sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                gradients = [p.grad for p in self.network.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("Nonfinite gradients")
                optimizer.step()
                steps += 1
                epoch_samples += len(indices)
                samples_seen += len(indices)
                optimization_loss = loss.item()
                raw_loss = unweighted_loss.item() * self.target_scale ** 2
                weighted_loss = optimization_loss * self.target_scale ** 2
                total += raw_loss * len(indices)
                weighted_total += weighted_loss * len(indices)
                optimization_total += optimization_loss * len(indices)
                if steps % self.config["log_every_steps"] == 0:
                    gradient_norms = {
                        group: float(torch.stack([parameter.grad.detach().double().square().sum()
                            for name, parameter in self.network.named_parameters()
                            if name.split(".")[0] == group and parameter.grad is not None]).sum().sqrt())
                        for group in ("norm_layer", "emb_layer", "layers", "final_layers")}
                    self._event("training", "progress", epoch=epoch + 1, step=steps, loss=raw_loss,
                                loss_units=PROGRESS_LOSS_UNITS[self.config["target_centering"]],
                                weighted_loss=weighted_loss,
                                optimization_loss=optimization_loss,
                                gradient_norms=gradient_norms,
                                learning_rate=optimizer.param_groups[0]["lr"],
                                elapsed_seconds=time.monotonic() - started,
                                samples_seen=samples_seen,
                                throughput_samples_per_second=samples_seen /
                                max(time.monotonic() - started, 1e-9), resources=device_resources(self.device))
                if observer_every_steps is not None and steps % observer_every_steps == 0:
                    observe(epoch + epoch_samples / len(y))
                if self.config["max_steps"] is not None and steps >= self.config["max_steps"]:
                    break
            complete_epoch = epoch_samples == len(y)
            epochs_completed += int(complete_epoch)
            actual_epochs = epoch + epoch_samples / len(y)
            self._event("training", "epoch_completed" if complete_epoch else "epoch_stopped",
                        epoch=epoch + 1, epochs=actual_epochs, step=steps, loss=total / epoch_samples,
                        weighted_loss=weighted_total / epoch_samples,
                        optimization_loss=optimization_total / epoch_samples,
                        epoch_samples=epoch_samples, samples_seen=samples_seen, complete_epoch=complete_epoch,
                        learning_rate=optimizer.param_groups[0]["lr"], resources=device_resources(self.device))
            observe(epoch + 1 if complete_epoch else actual_epochs)
            if validation_inputs is not None:
                vx, vy = validation_inputs
                self.network.eval()
                squared_error = 0.0
                with torch.inference_mode():
                    for offset in range(0, len(vy), self.config["batch_size"]):
                        xb = torch.from_numpy(vx[offset:offset + self.config["batch_size"]]).to(self.device)
                        yb = torch.from_numpy(vy[offset:offset + self.config["batch_size"]]).to(self.device)
                        squared_error += torch.nn.functional.mse_loss(
                            self.network(xb) * self.target_scale, yb, reduction="sum").item()
                self._event("validation", "epoch_completed", epoch=epoch + 1, loss=squared_error / len(vy),
                            samples=len(vy), checkpoint_selection=False, resources=device_resources(self.device))
            if self.config["max_steps"] is not None and steps >= self.config["max_steps"]:
                break
        self.trained = True
        return {"loss": total / epoch_samples, "learning_rate": self.config["learning_rate"],
                "weighted_loss": weighted_total / epoch_samples,
                "optimization_loss": optimization_total / epoch_samples, "target_scale": self.target_scale,
                "target_scaling": self.config["target_scaling"],
                "target_centering": self.config["target_centering"],
                "raw_target_std": training_std,
                "centered_target_std": (None if centering_stats is None
                                        else centering_stats["selected_centered_std"]),
                "transformed_target_std": (None if centering_stats is None
                                           else centering_stats.get("selected_transformed_std")),
                "mask_excluded": int((~mask).sum()),
                "loss_units": TRAINING_LOSS_UNITS[self.config["target_centering"]],
                "time_loss_weighting": self.config["time_loss_weighting"],
                "optimizer": {"none": "Adam_MSE", "train_variance": "Adam_time_weighted_MSE",
                              "date_decay_10d": "Adam_date_weighted_MSE"}[self.config["time_loss_weighting"]],
                "steps": steps, "samples": len(y), "samples_seen": samples_seen,
                "epochs": actual_epochs, "epochs_completed": epochs_completed,
                "stop_reason": "max_steps" if steps == self.config["max_steps"] else "epochs"}

    def predict(self, batch):
        if not self.trained:
            raise ValueError("Model is not fitted")
        return self._predict(batch)

    def _missing_history(self, batch):
        return (batch.observed[..., :self.lob_feature_count].any(axis=2).sum(axis=1)
                < self.preprocessing["min_history"])

    def _trunk_features(self, x):
        """Frozen trunk representation of prepared inputs; the residual head's only input."""
        features = np.empty((len(x), self.network.feature_dim), dtype=np.float32)
        self.network.eval()
        with torch.inference_mode():
            for offset in range(0, len(x), self.config["batch_size"]):
                xb = torch.from_numpy(x[offset:offset + self.config["batch_size"]]).to(self.device)
                features[offset:offset + len(xb)] = self.network.forward_features(xb).cpu().numpy()
        return features

    def _output_scores(self, x, open_rows=None):
        """Unscaled output-head scores; ``open_rows`` adds the trained 09:30 correction.

        Rows outside ``open_rows`` keep the frozen trunk score bit for bit, so a
        residual head can only ever change the section it was declared for.
        """
        head = None if open_rows is None else self.open_residual_head
        scores = np.empty(len(x), dtype=np.float64)
        self.network.eval()
        if head is not None:
            head.eval()
        with torch.inference_mode():
            for offset in range(0, len(x), self.config["batch_size"]):
                xb = torch.from_numpy(x[offset:offset + self.config["batch_size"]]).to(self.device)
                trunk = self.network.forward_features(xb)
                output = self.network.forward_output(trunk)
                if head is not None:
                    selected = np.nonzero(open_rows[offset:offset + len(xb)])[0]
                    if len(selected):
                        index = torch.from_numpy(selected).to(self.device)
                        output[index] = output[index] + head(trunk[index]).to(output.dtype)
                scores[offset:offset + len(xb)] = output.cpu().numpy()
        return scores

    def _predict(self, batch):
        """Fixed-weight scores for epoch diagnostics; public inference still requires a fitted model."""
        x = self._input(batch)
        open_rows = None
        if self.open_residual_head is not None:
            open_rows = open_market_mask(batch.keys, self.open_residual_metadata.get(
                "sample_time", OPEN_SAMPLE_TIME)) & ~self._missing_history(batch)
        scores = self._output_scores(x, open_rows) * self.target_scale
        missing = self._missing_history(batch)
        # Fallback is the fitted training-target mean. For timestamp-centered training
        # this is a relative score; no future timestamp mean or evaluation label exists here.
        scores[missing] = self.intercept
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite predictions")
        return scores

    def attach_open_residual(self, *, hidden_dim=32, seed=None):
        """Freeze the trunk and add a zero-initialised 09:30 correction head on top of it.

        The trunk parameters keep their checkpoint ``requires_grad`` flag on purpose: flipping
        it makes oneDNN select a packed inference kernel whose rounding differs from the frozen
        V2 predictions. Freezing is enforced instead by optimising only the head on trunk
        outputs computed under ``inference_mode``, so no trunk gradient is ever created.
        """
        if self.open_residual_head is not None:
            raise ValueError("The open residual head is already attached")
        head = OpenResidualHead(self.network.feature_dim, hidden_dim,
                                seed=self.seed if seed is None else seed).to(self.device)
        self.open_residual_head = head
        self.open_residual_metadata = {
            "method": OPEN_RESIDUAL_METHOD, "sample_time": OPEN_SAMPLE_TIME,
            "feature_dim": head.feature_dim, "hidden_dim": head.hidden_dim,
            "gating": "correction added only where the prediction key exchange time equals sample_time "
                      "and min_history is satisfied; every other row keeps the frozen trunk score",
            "target_space": "checkpoint standardized target units, scaled by target_scale at inference",
            "target_centering": self.config["target_centering"],
            "init_seed": int(self.seed if seed is None else seed),
            "trunk_parameters": int(sum(p.numel() for p in self.network.parameters())),
            "head_parameters": int(sum(p.numel() for p in head.parameters())),
            "optimizer_scope": "head parameters only; trunk outputs are detached frozen features",
        }
        self._event("training", "open_residual_attached", **self.open_residual_metadata)
        return head

    def fit_open_residual(self, batch, target, *, sample_mask=None, max_steps=1500,
                          learning_rate=1e-4, batch_size=32):
        """Fit only the 09:30 correction head on frozen trunk outputs.

        ``target`` is the training target in optimization units (the checkpoint's
        standardized space, already divided by ``target_scale``). The head regresses
        ``target - frozen trunk output`` for history-valid 09:30 samples; the outer
        ``target_scale`` is applied at prediction time, not here.
        """
        head = self.open_residual_head
        if head is None:
            raise ValueError("Attach the open residual head before fitting it")
        if type(max_steps) is not int or max_steps < 1 or type(batch_size) is not int or batch_size < 1:
            raise ValueError("max_steps and batch_size must be positive integers")
        y = np.asarray(target, dtype=np.float32)
        if y.shape != (len(batch.keys),) or not len(y) or not np.isfinite(y).all():
            raise ValueError("Open residual training requires nonempty aligned finite targets")
        eligible = open_market_mask(batch.keys, self.open_residual_metadata.get(
            "sample_time", OPEN_SAMPLE_TIME)) & ~self._missing_history(batch)
        if sample_mask is not None:
            mask = np.asarray(sample_mask)
            if mask.dtype != bool or mask.shape != y.shape:
                raise ValueError("Open residual sample mask must be aligned booleans")
            if (mask & ~eligible).any():
                raise ValueError("Open residual training may select only history-valid 09:30 samples")
            eligible &= mask
        if not eligible.any():
            raise ValueError("Open residual training requires a history-valid 09:30 sample")
        x = self._input(batch, sample_mask=eligible)
        features = self._trunk_features(x)
        residual = (y[eligible].astype(np.float64) - self._output_scores(x)).astype(np.float32)
        initial_mse = float(np.square(residual.astype(np.float64)).mean())
        # Clear stale trunk gradients left by any earlier full-trunk fit so the freeze check
        # below observes this fit only; no optimizer owns those parameters.
        self.network.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        head.train()
        optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(self.seed)
        started, steps = time.monotonic(), 0
        while steps < max_steps:
            order = torch.randperm(len(residual), generator=generator).numpy()
            for offset in range(0, len(order), batch_size):
                indices = order[offset:offset + batch_size]
                xb = torch.from_numpy(features[indices]).to(self.device)
                yb = torch.from_numpy(residual[indices]).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(head(xb), yb)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite open residual loss")
                loss.backward()
                gradients = [p.grad for p in head.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("Nonfinite open residual gradients")
                optimizer.step()
                steps += 1
                if steps % 100 == 0 or steps == max_steps:
                    self._event("training", "open_residual_progress", step=steps, loss=loss.item(),
                                initial_mse=initial_mse, elapsed_seconds=time.monotonic() - started)
                if steps >= max_steps:
                    break
        head.eval()
        if any(parameter.grad is not None for parameter in self.network.parameters()):
            raise RuntimeError("Frozen trunk received gradients during open residual training")
        with torch.inference_mode():
            correction = np.concatenate([
                head(torch.from_numpy(features[offset:offset + batch_size]).to(self.device)).cpu().numpy()
                for offset in range(0, len(features), batch_size)])
        final_mse = float(np.square(correction.astype(np.float64) - residual).mean())
        self.open_residual_metadata.update({
            "samples": int(eligible.sum()), "steps": steps, "learning_rate": learning_rate,
            "batch_size": batch_size, "shuffle_seed": self.seed, "optimizer": "Adam_MSE",
            "initial_mse": initial_mse, "final_mse": final_mse,
            "mse_reduction": None if initial_mse <= 0 else 1 - final_mse / initial_mse,
            "residual_std": float(residual.astype(np.float64).std()),
            "correction_mean": float(correction.astype(np.float64).mean()),
            "correction_std": float(correction.astype(np.float64).std()),
            "correction_min": float(correction.min()), "correction_max": float(correction.max()),
            "wall_seconds": time.monotonic() - started,
            "in_sample": True,
        })
        summary = {k: v for k, v in self.open_residual_metadata.items()
                   if k not in {"gating", "target_space"}}
        self._event("training", "open_residual_fitted", **summary)
        return summary

    def validate_training_preprocessing(self, batch, sample_mask):
        self._input(batch)
        return {"passed": True, "normalization": "per_sample_BiN_no_population_fit",
                "input_features": self.preprocessing["feature_columns"], "mask_appended": False}

    def _payload(self):
        if not self.trained:
            raise ValueError("Cannot save an unfitted model")
        payload = {"schema_version": 3 if self.new_input_contract else (2 if self.config["target_scaling"] == "train_std" else 1),
                   "kind": "modified_tlob", "architecture": self.architecture,
                   "model_config": self.config, "preprocessing": self.preprocessing, "seed": self.seed,
                   "training_mean": self.intercept,
                   "label": TARGET_LABELS[self.config["target_centering"]],
                   "prediction_units": PREDICTION_UNITS[self.config["target_centering"]],
                   "loss": {"none": "MSE", "train_variance": "time_weighted_MSE",
                            "date_decay_10d": "date_weighted_MSE"}[self.config["time_loss_weighting"]],
                   "target_scale": self.target_scale,
                   "target_centering": self.config["target_centering"],
                   "target_centering_stats": self.target_centering_stats,
                   "time_loss_weighting_stats": self.time_loss_weighting_stats,
                   "torch_version": str(torch.__version__),
                   "state_dict": {k: v.detach().cpu() for k, v in self.network.state_dict().items()}}
        if self.open_residual_head is not None:
            # The trunk tensors above stay byte-identical to the frozen checkpoint; only the
            # separate head state and its training metadata are appended.
            payload["schema_version"] = 4
            payload["open_residual"] = {
                **self.open_residual_metadata,
                "method": OPEN_RESIDUAL_METHOD, "sample_time": self.open_residual_metadata["sample_time"],
                "feature_dim": self.open_residual_head.feature_dim,
                "hidden_dim": self.open_residual_head.hidden_dim,
                "base_state_sha256": state_dict_sha256(payload["state_dict"]),
                "state_dict": {k: v.detach().cpu() for k, v in self.open_residual_head.state_dict().items()},
            }
        return payload

    def save(self, path):
        torch.save(self._payload(), Path(path))

    def export_text(self, path):
        payload = self._payload()
        states = [payload["state_dict"]]
        if "open_residual" in payload:
            states.append(payload["open_residual"]["state_dict"])
        for state in states:
            state.update({k: {"dtype": str(t.dtype).removeprefix("torch."), "shape": list(t.shape),
                              "data": t.reshape(-1).tolist()} for k, t in state.items()})
        Path(path).write_text(json.dumps(payload, allow_nan=False, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path, *, device="cpu", recorder=None):
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else torch.load(
            path, map_location="cpu", weights_only=True)
        if payload["schema_version"] not in (1, 2, 3, 4) or payload["kind"] != "modified_tlob":
            raise ValueError("Unsupported checkpoint format")
        if (payload["schema_version"] == 4) != ("open_residual" in payload):
            raise ValueError("Open residual checkpoints require schema_version=4 and a head payload")
        config = {**payload["model_config"], "device": device}
        if payload["schema_version"] == 3 and not {"book_representation", "trade_features"}.issubset(config):
            raise ValueError("Checkpoint is missing its input transformation contract")
        result = cls(config, payload["preprocessing"], seed=payload["seed"], recorder=recorder)
        if payload.get("target_centering", "none") != result.config["target_centering"]:
            raise ValueError("Checkpoint target-centering contract mismatch")
        if result.new_input_contract and payload["schema_version"] < 3:
            raise ValueError("LOB input transformations require checkpoint schema_version=3")
        if payload["schema_version"] == 1 and result.config["target_scaling"] != "none":
            raise ValueError("Target scaling requires checkpoint schema_version=2")
        if result.architecture != payload["architecture"]:
            raise ValueError("Checkpoint architecture/preprocessing mismatch")
        state = payload["state_dict"]
        if path.suffix == ".json":
            state = {k: torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"])).reshape(meta["shape"])
                     for k, meta in state.items()}
        if not all(torch.isfinite(t).all() for t in state.values()):
            raise ValueError("Nonfinite checkpoint weights")
        result.network.load_state_dict(state, strict=True)
        result.intercept = float(payload["training_mean"])
        if not np.isfinite(result.intercept):
            raise ValueError("Invalid training fallback")
        if "target_scale" not in payload and (payload["schema_version"] == 2
                                               or result.config["target_scaling"] == "train_std"):
            raise ValueError("Scaled checkpoint is missing target_scale")
        result.target_scale = float(payload.get("target_scale", 1.0))
        if not np.isfinite(result.target_scale) or result.target_scale <= 0:
            raise ValueError("Invalid target scale")
        if result.config["target_scaling"] == "none" and result.target_scale != 1.0:
            raise ValueError("Unscaled checkpoint requires target_scale=1")
        result.target_centering_stats = payload.get("target_centering_stats")
        expected_centering = result.config["target_centering"]
        if expected_centering != "none" and (
                not isinstance(result.target_centering_stats, dict)
                or result.target_centering_stats.get("method") != expected_centering):
            raise ValueError("Centered checkpoint is missing its target-transform statistics")
        result.time_loss_weighting_stats = payload.get("time_loss_weighting_stats")
        if result.config["time_loss_weighting"] == "train_variance":
            if not isinstance(result.time_loss_weighting_stats, dict):
                raise ValueError("Weighted checkpoint is missing training weight statistics")
            result.training_sample_times = tuple(result.time_loss_weighting_stats["by_time"])
        elif result.config["time_loss_weighting"] == "date_decay_10d":
            if (not isinstance(result.time_loss_weighting_stats, dict)
                    or result.time_loss_weighting_stats.get("method") != "date_decay_10d"):
                raise ValueError("Date-weighted checkpoint is missing its training weight statistics")
        if "open_residual" in payload:
            result._load_open_residual(payload["open_residual"], path)
        result.trained = True
        return result

    def _load_open_residual(self, payload, path):
        if not isinstance(payload, dict) or payload.get("method") != OPEN_RESIDUAL_METHOD:
            raise ValueError("Unsupported open residual head payload")
        if payload.get("feature_dim") != self.network.feature_dim:
            raise ValueError("Open residual head feature width does not match the trunk")
        if payload.get("hidden_dim", 0) < 1:
            raise ValueError("Invalid open residual head width")
        head = OpenResidualHead(self.network.feature_dim, payload["hidden_dim"]).to(self.device)
        state = payload.get("state_dict")
        if not isinstance(state, dict) or set(head.state_dict()) != set(state):
            raise ValueError("Open residual head state does not match its architecture")
        if path.suffix == ".json":
            state = {k: torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"])).reshape(meta["shape"])
                     for k, meta in state.items()}
        if not all(torch.isfinite(t).all() for t in state.values()):
            raise ValueError("Nonfinite open residual head weights")
        if payload.get("base_state_sha256") != state_dict_sha256(
                {k: v.detach().cpu() for k, v in self.network.state_dict().items()}):
            raise ValueError("Open residual head does not belong to this frozen trunk")
        head.load_state_dict(state, strict=True)
        head.eval()
        self.open_residual_head = head
        self.open_residual_metadata = {k: v for k, v in payload.items() if k != "state_dict"}
