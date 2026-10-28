"""Batch training/inference and portable checkpoints with frozen preprocessing."""

import hashlib
import importlib
import json
import os
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jinyiquant.io import write_json

from .device import device_resources, select_device
from .market_head import (MIN_CROSS_SECTION, cross_section_floor, head_from_payload, head_payload,
                          market_context, text_state as _text_state, timestamp_codes)
from .model import MLPLOB, DualPatchTLOB, ModifiedTLOB, configure_finite_check, parameter_groups
from .schema import validate_input_schema

BACKBONES = {"modified_tlob": ModifiedTLOB, "mlplob": MLPLOB, "dual_patch": DualPatchTLOB}
TRAINING_ONLY_MODULE = "jinyiquant.trajectory"
STREAMING_POOL_MODULE = "jinyiquant.pool"
# Requested compile mode -> torch.compile keyword arguments; {} keeps the eager path.
# "aot-eager" is the only backend that runs without Triton, so it is the diagnostic
# option on platforms where inductor is unavailable (Windows).
COMPILE_OPTIONS = {"eager": {}, "default": {"mode": "default"},
                   "reduce-overhead": {"mode": "reduce-overhead"},
                   "aot-eager": {"backend": "aot_eager"}}
# A compiled trunk must reproduce the eager forward bit-for-bit within this bound.
COMPILE_EQUIVALENCE_ATOL = 1e-6
# Key chunk a streaming pool is scored in: its windows are built per chunk, never pooled.
PREDICT_CHUNK_ROWS = 8192


def _trajectory_helpers():
    """Training-only helpers from ``jinyiquant.trajectory``, imported lazily.

    Inference and the flat submission package never need the trajectory module: the flat
    builder whitelists only inference dependencies, so a module-level import here would make
    the package unbuildable. Training paths (fit/save/resume/exposure) still resolve it.
    """
    return importlib.import_module(TRAINING_ONLY_MODULE)


def _streaming_pool(batch):
    """``batch`` itself when it is a streaming pool, else ``None`` for a materialised batch.

    Only ``jinyiquant.pool.StreamingPool`` takes the streamed-window path; the type is resolved
    lazily for the same reason as the trajectory helpers (the flat inference package carries
    neither module), so a materialised ``Batch`` keeps its exact training path. A build without
    the pool module can hold no pool - the class is defined there - so an absent module means
    "materialised" rather than an error on the inference path.
    """
    try:
        pool_type = importlib.import_module(STREAMING_POOL_MODULE).StreamingPool
    except ModuleNotFoundError:
        return None
    return batch if isinstance(batch, pool_type) else None


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
BLEND_KIND = "modified_tlob_blend"
BLEND_SCHEMA_VERSION = 4
SHARDED_BLEND_KIND = "modified_tlob_sharded_blend"
UPLOAD_WEIGHT_LIMIT = 50_000_000
BLEND_SHARED_KEYS = ("target_scaling", "target_centering", "time_loss_weighting", "book_representation",
                     "trade_features")
BACKBONE_SHARED_KEYS = {"modified_tlob": ("num_heads", "is_sin_emb"), "mlplob": (),
                        "dual_patch": ("num_heads", "patch_counts")}
TRAINING_STATE_SCHEMA_VERSION = 1


def _canonical_sha256(value):
    """Stable digest of a JSON-serialisable structure, or None for an absent one."""
    if value is None:
        return None
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _batch_indices_sha256(batches):
    """Digest of one epoch's batch composition, used to verify a resumed sampler position."""
    digest = hashlib.sha256()
    for indices in batches:
        array = np.ascontiguousarray(np.asarray(indices, dtype="<i8"))
        digest.update(len(array).to_bytes(8, "little"))
        digest.update(array.tobytes())
    return digest.hexdigest()


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



def timestamp_groups(keys, mask):
    """Usable sample positions grouped by their full timestamp (date and slot time).

    Grouped sampling replaces the global random shuffle with one cross-section per
    optimizer step, so every pairwise ranking comparison is same-timestamp only.
    """
    dates = pd.to_datetime(keys["date"])
    if dates.isna().any():
        raise ValueError("Grouped sampling requires a timestamp for every training row")
    selected = np.flatnonzero(np.asarray(mask, dtype=bool))
    frame = pd.DataFrame({"timestamp": dates.to_numpy()[selected], "index": selected})
    groups = [values.to_numpy() for _, values in frame.groupby("timestamp", sort=True)["index"]]
    if len(groups) < 2:
        raise ValueError("Grouped sampling requires at least two training timestamps")
    return groups


def grouped_epoch_batches(groups, batch_size, generator):
    """One shuffled pass over timestamps, split into exactly-sized same-timestamp batches.

    Each timestamp's usable rows are shuffled and chunked; the remainder that does not
    fill a batch is dropped so every batch holds exactly ``batch_size`` same-timestamp
    rows. All other rows are used exactly once per epoch.
    """
    batches = []
    for position in torch.randperm(len(groups), generator=generator).tolist():
        indices = groups[position]
        order = torch.randperm(len(indices), generator=generator).numpy()
        shuffled = indices[order]
        for offset in range(0, len(shuffled) - batch_size + 1, batch_size):
            batches.append(shuffled[offset:offset + batch_size])
    if not batches:
        raise ValueError("Grouped sampling produced no complete same-timestamp batch")
    return batches


def pairwise_hinge_ranking_loss(scores, targets):
    """Mean pairwise hinge loss over one same-timestamp group.

    ``relu(-(s_i - s_j) * (y_i - y_j))`` over all ordered pairs, divided by ``n**2``,
    matching the reference ``tf.reduce_mean`` with an all-ones mask
    (Feng et al., Temporal Relational Ranking, AAAI 2018). ``targets`` must be the
    same batch targets, so ordering is the same-timestamp label ordering.
    """
    score_gap = scores.unsqueeze(1) - scores.unsqueeze(0)
    target_gap = targets.unsqueeze(1) - targets.unsqueeze(0)
    return torch.clamp(-(score_gap * target_gap), min=0).mean()


def _gradients_are_finite(parameters):
    """Single stream drain for the whole gradient set.

    The per-tensor form used to read one host scalar per parameter tensor, which on
    an asynchronous accelerator drained the stream once per tensor instead of once
    per check.
    """
    flat = [grad.reshape(-1) for grad in (p.grad for p in parameters) if grad is not None]
    return bool(flat) and bool(torch.isfinite(torch.cat(flat)).all())


def _architecture_with_patch_counts(architecture):
    """Canonicalize the dual-scale patch counts for checkpoint comparisons.

    ``torch.save`` preserves the tuple while JSON text export stores a list, so the
    loader compares tuple forms instead of the serialized container.
    """
    architecture = dict(architecture)
    if "patch_counts" in architecture:
        architecture["patch_counts"] = tuple(architecture["patch_counts"])
    return architecture


class TLOBAdapter:
    """Shared regression training/inference loop for the torch LOB backbones.

    ``model.kind`` selects the upstream trunk (TLOB attention, MLPLOB dual-axis MLP
    or the project's dual-scale patch encoder); preprocessing, labels, loss,
    checkpoints and evaluation stay identical.
    """

    def __init__(self, model_config, data_config, *, seed=42, recorder=None):
        self.config = model_config.model_dump() if hasattr(model_config, "model_dump") else dict(model_config)
        self.config.setdefault("kind", "modified_tlob")
        if self.config["kind"] not in BACKBONES:
            raise ValueError(f"Unsupported torch model kind: {self.config['kind']!r}")
        if self.config["kind"] == "dual_patch":
            self.config["patch_counts"] = tuple(self.config.get("patch_counts", (4, 16)))
        self.config.setdefault("target_scaling", "none")
        self.config.setdefault("target_centering", "none")
        self.config.setdefault("max_steps", None)
        self.config.setdefault("time_loss_weighting", "none")
        self.config.setdefault("sampling", "shuffle")
        self.config.setdefault("lr_schedule", "constant")
        self.config.setdefault("ranking_loss", "none")
        self.config.setdefault("ranking_loss_weight", 0.0)
        self.config.setdefault("book_representation", "raw")
        self.config.setdefault("trade_features", "zero")
        self.config.setdefault("finite_check_every", 50)
        self.config.setdefault("compile_mode", "eager")
        if self.config["target_scaling"] not in ("none", "train_std"):
            raise ValueError("Unsupported target scaling")
        if self.config["target_centering"] not in TARGET_CENTERING_MODES:
            raise ValueError("Unsupported target centering")
        if self.config["time_loss_weighting"] not in ("none", "train_variance", "date_decay_10d"):
            raise ValueError("Unsupported loss weighting")
        if self.config["sampling"] not in ("shuffle", "timestamp_grouped"):
            raise ValueError("Unsupported sampling mode")
        if self.config["lr_schedule"] != "constant":
            # The frozen rule only supports a constant rate; the helper module stays lazy so
            # inference packaging never resolves the training-only trajectory module.
            if self.config["lr_schedule"] not in _trajectory_helpers().LR_SCHEDULES:
                raise ValueError("Unsupported learning-rate schedule")
        if self.config["ranking_loss"] not in ("none", "pairwise_hinge"):
            raise ValueError("Unsupported ranking loss")
        self.ranking_loss_weight = float(self.config["ranking_loss_weight"])
        if not np.isfinite(self.ranking_loss_weight) or self.ranking_loss_weight < 0:
            raise ValueError("Unsupported ranking loss weight")
        if (self.config["ranking_loss"] == "none") != (self.ranking_loss_weight == 0):
            raise ValueError("Ranking loss and its weight must be enabled together")
        data = data_config.model_dump(mode="json") if hasattr(data_config, "model_dump") else dict(data_config)
        self.preprocessing = {k: data[k] for k in PREPROCESS_KEYS}
        # Default-off: only a non-default transform joins the frozen preprocessing contract, so
        # existing checkpoint preprocessing dicts keep their exact identity.
        if data.get("feature_transform", "none") != "none":
            self.preprocessing["feature_transform"] = data["feature_transform"]
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
        self.architecture = {k: self.config[k] for k in ("hidden_dim", "num_layers")}
        if self.config["kind"] == "modified_tlob":
            self.architecture.update({k: self.config[k] for k in ("num_heads", "is_sin_emb")})
        elif self.config["kind"] == "dual_patch":
            self.architecture.update({k: self.config[k] for k in ("num_heads", "patch_counts")})
        self.architecture.update(seq_size=data["lookback"], num_features=len(fields))
        self.network = BACKBONES[self.config["kind"]](**self.architecture).to(self.device)
        self.finite_check_every = self.config["finite_check_every"]
        configure_finite_check(self.network, self.finite_check_every)
        self.parameter_groups = parameter_groups(self.network)
        self.intercept = None
        self.target_scale = 1.0
        self.target_centering_stats = None
        self.time_loss_weighting_stats = None
        self.trained = False
        # Default-off: a checkpoint without a market-context head keeps the frozen prediction
        # path byte for byte, including the batch-size slicing of the device forward.
        self.market_head = None
        # Set to True only to measure the damage of scoring a fragment: a market-context
        # head needs the whole same-timestamp cross-section in one call (see _market_residual).
        self.allow_cross_section_fragment = False
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

    def _learning_rate(self, step: int = 0) -> float:
        return _trajectory_helpers().learning_rate_at(self.config["lr_schedule"],
                                                      self.config["learning_rate"], step)

    def _training_fingerprint(self, keys, optimization_targets, used_positions):
        """Everything besides the parameters that a resumed update depends on."""
        canonical = {
            "preprocessing": self.preprocessing,
            "architecture": {key: (list(value) if isinstance(value, tuple) else value)
                             for key, value in self.architecture.items()},
            "model_config": {key: self.config[key] for key in (
                "batch_size", "sampling", "ranking_loss", "ranking_loss_weight", "target_scaling",
                "target_centering", "time_loss_weighting", "learning_rate", "lr_schedule")},
            "seed": self.seed,
            "fitted": {
                "intercept": None if self.intercept is None else float(self.intercept),
                "target_scale": float(self.target_scale),
                "target_centering_stats_sha256": _canonical_sha256(self.target_centering_stats),
                "time_loss_weighting_stats_sha256": _canonical_sha256(self.time_loss_weighting_stats),
            },
        }
        config_digest = _canonical_sha256(canonical)
        timestamps = pd.to_datetime(keys["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        data_digest = hashlib.sha256(
            np.ascontiguousarray(np.asarray(optimization_targets, dtype=np.float32)).tobytes()
            + np.asarray(used_positions, dtype=np.uint8).tobytes()
            + "\n".join(timestamps.tolist()).encode("utf-8")).hexdigest()
        return {"config_sha256": config_digest, "data_sha256": data_digest,
                "combined_sha256": hashlib.sha256((config_digest + data_digest).encode()).hexdigest()}

    def _save_training_state(self, directory, *, optimizer, generator, fingerprint, position,
                             generator_before_epoch, steps_per_epoch, exposure_counts):
        """Write a resumable training state: parameters, optimizer, RNG and sampler position."""
        trajectory = _trajectory_helpers()
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        numpy_state = np.random.get_state()
        payload = {
            "schema_version": TRAINING_STATE_SCHEMA_VERSION,
            "kind": self.config["kind"],
            "architecture": self.architecture,
            "model_config": self.config,
            "preprocessing": self.preprocessing,
            "seed": self.seed,
            "device": str(self.device),
            "torch_version": str(torch.__version__),
            "lr_schedule": self.config["lr_schedule"],
            "learning_rate": self._learning_rate(int(position["steps"])),
            "fingerprint": fingerprint,
            "position": {**{key: (value if key == "epoch_batches_sha256" else int(value))
                            for key, value in position.items()},
                         "steps_per_epoch": {str(key): int(value)
                                             for key, value in steps_per_epoch.items()}},
            "generator_before_epoch": generator_before_epoch,
            "rng": {"torch": torch.get_rng_state(), "generator": generator.get_state(),
                    # Store the uint32 RNG state as raw bytes: a torch.uint32 tensor cannot be
                    # loaded by the torch 2.5 runtime pinned for submission.
                    "numpy": (numpy_state[0], numpy_state[1].tobytes(),
                              numpy_state[2], numpy_state[3], numpy_state[4]),
                    "python": random.getstate(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
            "network_state_dict": {key: value.detach().cpu()
                                   for key, value in self.network.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "exposure_counts": (None if exposure_counts is None
                                else torch.from_numpy(np.asarray(exposure_counts).copy())),
        }
        torch.save(payload, target / trajectory.CHECKPOINT_FILENAME)
        optimizer_state = payload["optimizer_state_dict"]
        exposure = payload["exposure_counts"]
        metadata = {key: payload[key] for key in (
            "schema_version", "kind", "seed", "device", "torch_version", "lr_schedule",
            "learning_rate", "preprocessing", "position", "fingerprint")}
        metadata.update({
            "optimizer": {"state_entries": len(optimizer_state["state"]),
                          "step": max((int(state["step"]) for state in optimizer_state["state"].values()),
                                      default=0)},
            "rng_streams": sorted(payload["rng"]),
            "sampler": {"epoch_index": int(position["epoch_index"]),
                        "batch_index": int(position["batch_index"]),
                        "epoch_batches_sha256": position.get("epoch_batches_sha256"),
                        "generator_state_saved": True},
            "exposure_counters": None if exposure is None else {
                "rows": int(exposure.numel()), "samples_seen": int(exposure.sum()),
                "sha256": hashlib.sha256(exposure.numpy().astype("<i8").tobytes()).hexdigest()},
        })
        write_json(target / trajectory.CHECKPOINT_METADATA_FILENAME, metadata)
        self._event("training", "training_state_saved", directory=str(target),
                    steps=int(position["steps"]), epoch_index=int(position["epoch_index"]),
                    batch_index=int(position["batch_index"]), exposure=exposure_counts is not None)
        return metadata

    def _restore_training_state(self, checkpoint, *, optimizer, generator, fingerprint, usable):
        """Load a training state and fail closed if its recipe, pool or position drifted."""
        if checkpoint.get("schema_version") != TRAINING_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported training-state checkpoint schema")
        if (checkpoint["kind"] != self.config["kind"]
                or checkpoint["preprocessing"] != self.preprocessing
                or _architecture_with_patch_counts(checkpoint["architecture"]) != self.architecture):
            raise ValueError("Training-state checkpoint architecture/preprocessing mismatch")
        if checkpoint["fingerprint"]["combined_sha256"] != fingerprint["combined_sha256"]:
            raise ValueError("Training data or recipe changed since the checkpoint; "
                             "a continuous trajectory requires the identical pool")
        if checkpoint["lr_schedule"] != self.config["lr_schedule"]:
            raise ValueError("Learning-rate schedule changed since the checkpoint")
        position = dict(checkpoint["position"])
        if int(position["usable_samples"]) != usable:
            raise ValueError("Training-state checkpoint was written for a different usable sample count")
        if int(position["steps"]) < 1 or int(position["batch_index"]) < 0:
            raise ValueError("Malformed training-state position")
        self.network.load_state_dict(checkpoint["network_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        rng = checkpoint["rng"]
        torch.set_rng_state(rng["torch"])
        numpy_state = rng["numpy"]
        encoded = numpy_state[1]
        if isinstance(encoded, torch.Tensor):  # legacy checkpoints stored a uint32 tensor
            state_array = encoded.numpy().astype(np.uint32)
        else:
            state_array = np.frombuffer(bytes(encoded), dtype=np.uint32).copy()
        np.random.set_state((numpy_state[0], state_array,
                             numpy_state[2], numpy_state[3], numpy_state[4]))
        random.setstate(rng["python"])
        if str(checkpoint["device"]).startswith("cuda"):
            if not torch.cuda.is_available():
                raise ValueError("A CUDA training-state checkpoint cannot be resumed without CUDA")
            torch.cuda.set_rng_state_all(rng["cuda"])
        if int(position["batch_index"]) > 0:
            generator.set_state(checkpoint["generator_before_epoch"])
        exposure = checkpoint.get("exposure_counts")
        return position, None if exposure is None else exposure.numpy().astype(np.int64)

    def _training_forward(self, probe):
        """Forward callable for the optimization loop, with the compile fallback contract.

        ``report["status"]`` is ``eager`` (not requested), ``compiled`` or ``fallback``, and
        every fallback emits a warning event, so a silent degradation is impossible. A
        compiled trunk must reproduce the eager forward on one real batch within
        ``COMPILE_EQUIVALENCE_ATOL`` before it is allowed to drive the trajectory; only the
        validated branch-free ``_forward_impl`` math is compiled, never the input guards.
        """
        requested = self.config["compile_mode"]
        report = {"requested": requested, "status": "eager", "fallback": None,
                  "equivalence_atol": COMPILE_EQUIVALENCE_ATOL,
                  "finite_check_every": self.finite_check_every, "probe_shape": list(probe.shape)}
        options = COMPILE_OPTIONS[requested]
        if not options:
            self._event("training", "kernel_path", **report)
            return self.network, report
        report["options"] = options
        try:
            compile_fn = getattr(torch, "compile", None)
            if compile_fn is None:
                raise RuntimeError("torch.compile is unavailable in this torch build")
            compiled = compile_fn(self.network._forward_impl, **options)
            with torch.enable_grad():
                expected, actual = self.network(probe), compiled(probe)
            difference = float((expected - actual).abs().max().detach())
            if not bool(torch.isfinite(actual).all()):
                raise RuntimeError("compiled forward returned non-finite values")
            if difference > COMPILE_EQUIVALENCE_ATOL:
                raise RuntimeError(f"compiled forward differs from eager by {difference:.3g} "
                                   f"> {COMPILE_EQUIVALENCE_ATOL:g}")
        except Exception as exc:  # any compile/validate failure must fall back to eager
            report.update(status="fallback", reason="compile_failed",
                          fallback=f"{type(exc).__name__}: {exc}"[:600])
            self._event("training", "compile_fallback", level="WARNING", **report)
            return self.network, report

        def predict(x):
            # The amortized input guard stays outside the graph, so compiled and eager
            # trajectories keep exactly the same host-visible checks.
            self.network.finite_check(x)
            return compiled(x)

        report.update(status="compiled", max_abs_forward_difference=difference)
        self._event("training", "compile_enabled", **report)
        return predict, report

    def fit(self, batch, target, *, sample_mask=None, validation=None, observer=None,
            observer_every_steps=None, resume=None, step_budget=None, checkpoint_at_steps=None,
            checkpoint_dir=None, track_exposure=False):
        """One continuous optimization trajectory, optionally resumed and checkpointed.

        ``resume`` is a loaded training-state payload; the pool, recipe, fitted transforms and
        sampler position must reproduce the saved run bit for bit or the fit fails closed.
        ``step_budget`` stops at an absolute step count, ``checkpoint_at_steps`` writes a
        resumable state at those absolute steps, and ``track_exposure`` accumulates per-row
        sample exposure for the budget accounting.

        ``batch`` is either a materialised :class:`jinyiquant.data.Batch`, whose window tensor
        is prefetched once and indexed per step, or a :class:`jinyiquant.pool.StreamingPool`,
        whose windows are built per draw (``pool.batch``) and never materialised. The switch is
        the pool type alone, so the default path is unchanged; the sampler draws in masked
        space and the pool positions of a draw are ``np.flatnonzero(sample_mask)[indices]``.
        """
        if observer_every_steps is not None and (type(observer_every_steps) is not int
                                                 or observer_every_steps < 1):
            raise ValueError("observer_every_steps must be a positive integer or None")
        checkpoint_set = sorted(set(checkpoint_at_steps or ()))
        if any(type(value) is not int or value < 1 for value in checkpoint_set):
            raise ValueError("checkpoint_at_steps must be positive absolute step numbers")
        if checkpoint_set and checkpoint_dir is None:
            raise ValueError("checkpoint_at_steps requires checkpoint_dir")
        if step_budget is not None and (type(step_budget) is not int or step_budget < 1):
            raise ValueError("step_budget must be a positive integer or None")
        if resume is not None and step_budget is not None \
                and int(resume["position"]["steps"]) >= step_budget:
            raise ValueError("step_budget must exceed the resumed step count")
        y = np.asarray(target, dtype=np.float32)
        if y.shape != (len(batch.keys),) or not len(y) or not np.isfinite(y).all():
            raise ValueError("Training requires nonempty aligned finite targets")
        mask = np.ones(len(y), dtype=bool) if sample_mask is None else np.asarray(sample_mask)
        if mask.dtype != bool or mask.shape != y.shape or not mask.any():
            raise ValueError("Training sample mask must select aligned samples")
        # The sampler draws in masked space; a streaming pool needs the pool position of every
        # selected row, and builds only the windows of the draw instead of the whole pool.
        pool = _streaming_pool(batch)
        selected = None if pool is None else np.flatnonzero(mask)
        x = None if pool is not None else self._input(batch, sample_mask=mask)

        def training_input(rows):
            """Preprocessed windows of the masked rows ``rows``, in sampler coordinates.

            The materialised path indexes the prefetched window tensor. The streaming path
            rebuilds exactly those windows from the pool's bars, so the optimizer sees the
            same bytes either way; nothing else in this loop depends on which one ran.
            """
            if pool is None:
                return x[rows]
            return self._input(pool.batch(selected[rows]))

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
        self.target_scale = (training_std if self.config["target_scaling"] == "train_std"
                             and training_std > 0 else 1.0)
        optimization_targets = (y if self.target_scale == 1.0 else
                                (y.astype(np.float64) / self.target_scale).astype(np.float32))
        if not np.isfinite(optimization_targets).all():
            raise FloatingPointError("Nonfinite scaled training targets")
        self._event("training", "target_scaling_fitted", method=self.config["target_scaling"],
                    target_scale=self.target_scale, training_std=training_std, samples=len(y),
                    constant_target_fallback=(self.config["target_scaling"] == "train_std"
                                              and training_std == 0),
                    centering=self.config["target_centering"],
                    centered_training_std=(None if centering_stats is None
                                           else centering_stats["selected_centered_std"]),
                    centering_stats=centering_stats,
                    loss_units=("centered_return_squared" if centering_stats is not None
                                else "original_return_squared"))
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self._learning_rate(0))
        generator = torch.Generator().manual_seed(self.seed)
        grouped = self.config["sampling"] == "timestamp_grouped"
        groups = timestamp_groups(batch.keys, mask) if grouped else None
        batch_size = self.config["batch_size"]
        used_positions = np.ones(len(y), dtype=bool)
        if grouped:
            used_positions[:] = False
            for group in groups:
                used_positions[group[:(len(group) // batch_size) * batch_size]] = True
            self._event("training", "grouped_sampling_fitted", groups=len(groups),
                        group_key="full_timestamp", batch_size=batch_size, samples=len(y),
                        samples_used_per_epoch=int(used_positions.sum()),
                        dropped_tail_per_epoch=int(len(y) - int(used_positions.sum())),
                        ranking_loss=self.config["ranking_loss"],
                        ranking_loss_weight=self.ranking_loss_weight)
        usable = int(used_positions.sum())
        self.training_keys = batch.keys.loc[mask]
        self.used_positions = used_positions
        fingerprint = self._training_fingerprint(self.training_keys, optimization_targets, used_positions)
        # Compile after the pool is fixed so the probe batch has the real training shape.
        self.network.train()
        probe_rows = slice(0, min(batch_size, len(y)))
        probe = torch.from_numpy(training_input(probe_rows)).to(self.device)
        predict_forward, compile_report = self._training_forward(probe)
        terms_dim = 4 if self.ranking_loss_weight else 3
        exposure_counts = np.zeros(len(y), dtype=np.int64) if track_exposure else None
        steps, samples_seen, epochs_completed = 0, 0, 0
        start_epoch, start_batch, steps_per_epoch = 0, 0, {}
        resume_epoch_batches_sha256 = None
        if resume is not None:
            position, restored_exposure = self._restore_training_state(
                resume, optimizer=optimizer, generator=generator, fingerprint=fingerprint, usable=usable)
            steps, samples_seen = int(position["steps"]), int(position["samples_seen"])
            epochs_completed = int(position["epochs_completed"])
            start_epoch, start_batch = int(position["epoch_index"]), int(position["batch_index"])
            steps_per_epoch = {int(key): int(value) for key, value in position["steps_per_epoch"].items()}
            resume_epoch_batches_sha256 = position.get("epoch_batches_sha256")
            if restored_exposure is not None:
                if exposure_counts is None:
                    raise ValueError("The checkpoint carries exposure counters; resume with track_exposure=True")
                exposure_counts = restored_exposure
            self._event("training", "training_state_resumed", steps=steps, samples_seen=samples_seen,
                        epoch_index=start_epoch, batch_index=start_batch,
                        learning_rate=self._learning_rate(steps))
        started = time.monotonic()
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

        if resume is None:
            observe(0)
        stop_reason = "epochs"
        actual_epochs, final_loss = 0.0, None
        for epoch in range(start_epoch, self.config["epochs"]):
            self.network.train()
            generator_before_epoch = generator.get_state()
            if grouped:
                epoch_batches = grouped_epoch_batches(groups, batch_size, generator)
            else:
                order = torch.randperm(len(y), generator=generator).numpy()
                epoch_batches = [order[offset:offset + batch_size]
                                 for offset in range(0, len(y), batch_size)]
            epoch_batches_sha256 = _batch_indices_sha256(epoch_batches)
            offset = start_batch if epoch == start_epoch else 0
            if offset:
                if offset >= len(epoch_batches):
                    raise ValueError("Checkpoint batch position exceeds the rebuilt epoch")
                if epoch_batches_sha256 != resume_epoch_batches_sha256:
                    raise ValueError("Checkpoint sampling position does not reproduce the saved batch order")
            epoch_samples = int(sum(len(indices) for indices in epoch_batches[:offset]))
            total = weighted_total = optimization_total = objective_total = ranking_total = 0.0
            # Host-visible step statistics are accumulated on the device and read once per
            # log interval and once per epoch, instead of three stream drains per step.
            epoch_totals = torch.zeros(terms_dim, dtype=torch.float64, device=self.device)
            for position in range(offset, len(epoch_batches)):
                indices = epoch_batches[position]
                xb = torch.from_numpy(training_input(indices)).to(self.device)
                yb = torch.from_numpy(optimization_targets[indices]).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                predictions = predict_forward(xb)
                unweighted_loss = torch.nn.functional.mse_loss(predictions, yb)
                if time_weights is None:
                    mse_term = unweighted_loss
                else:
                    weights = torch.from_numpy(time_weights[indices]).to(self.device)
                    mse_term = (weights * (predictions - yb).square()).sum() / weights.sum()
                ranking_loss = None
                if self.ranking_loss_weight:
                    ranking_loss = pairwise_hinge_ranking_loss(predictions, yb)
                    loss = mse_term + self.ranking_loss_weight * ranking_loss
                else:
                    loss = mse_term
                # Sampling keeps the guards fail-closed while amortizing their stream drains;
                # one is still paid on the first step and every finite_check_every steps.
                checking = steps % self.finite_check_every == 0
                if checking and not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                if checking and not _gradients_are_finite(self.network.parameters()):
                    raise FloatingPointError("Nonfinite gradients")
                optimizer.step()
                steps += 1
                steps_per_epoch[epoch] = position + 1
                epoch_samples += len(indices)
                samples_seen += len(indices)
                if exposure_counts is not None:
                    exposure_counts[indices] += 1
                terms = torch.stack([unweighted_loss.detach(), mse_term.detach(), loss.detach()]
                                    + ([] if ranking_loss is None else [ranking_loss.detach()]))
                epoch_totals += terms.double() * len(indices)
                if steps % self.config["log_every_steps"] == 0:
                    step_terms = (terms.double() * self.target_scale ** 2).tolist()
                    raw_loss, weighted_loss, objective_loss = step_terms[:3]
                    optimization_loss = step_terms[1] / self.target_scale ** 2
                    ranking_value = step_terms[3] if ranking_loss is not None else None
                    gradient_norms = {
                        group: float(torch.stack([parameter.grad.detach().double().square().sum()
                            for name, parameter in self.network.named_parameters()
                            if name.split(".")[0] == group and parameter.grad is not None]).sum().sqrt())
                        for group in self.parameter_groups}
                    self._event("training", "progress", epoch=epoch + 1, step=steps, loss=raw_loss,
                                loss_units=PROGRESS_LOSS_UNITS[self.config["target_centering"]],
                                weighted_loss=weighted_loss, optimization_loss=optimization_loss,
                                objective_loss=objective_loss, ranking_loss=ranking_value,
                                ranking_loss_weight=self.ranking_loss_weight,
                                sampling=self.config["sampling"], gradient_norms=gradient_norms,
                                learning_rate=self._learning_rate(steps),
                                elapsed_seconds=time.monotonic() - started,
                                samples_seen=samples_seen,
                                throughput_samples_per_second=samples_seen /
                                max(time.monotonic() - started, 1e-9),
                                resources=device_resources(self.device))
                if observer_every_steps is not None and steps % observer_every_steps == 0:
                    observe(epoch + epoch_samples / len(y))
                if steps in checkpoint_set:
                    at_epoch_end = position + 1 >= len(epoch_batches)
                    self._save_training_state(
                        checkpoint_dir, optimizer=optimizer, generator=generator, fingerprint=fingerprint,
                        position={"steps": steps, "samples_seen": samples_seen,
                                  "epochs_completed": epochs_completed,
                                  "epoch_index": epoch + 1 if at_epoch_end else epoch,
                                  "batch_index": 0 if at_epoch_end else position + 1,
                                  "epoch_batches_sha256": None if at_epoch_end else epoch_batches_sha256,
                                  "usable_samples": usable},
                        generator_before_epoch=(generator.get_state() if at_epoch_end
                                                else generator_before_epoch),
                        steps_per_epoch=steps_per_epoch, exposure_counts=exposure_counts)
                if step_budget is not None and steps >= step_budget:
                    stop_reason = "step_budget"
                    break
                if self.config["max_steps"] is not None and steps >= self.config["max_steps"]:
                    stop_reason = "max_steps"
                    break
            if epoch_samples:
                complete_epoch = epoch_samples == usable
                epochs_completed += int(complete_epoch)
                actual_epochs = epoch + epoch_samples / len(y)
                epoch_readout = epoch_totals.tolist()
                total = epoch_readout[0] * self.target_scale ** 2
                weighted_total = epoch_readout[1] * self.target_scale ** 2
                optimization_total, objective_total = epoch_readout[1], epoch_readout[2]
                if self.ranking_loss_weight:
                    ranking_total = epoch_readout[3]
                final_loss = total / epoch_samples
                self._event("training", "epoch_completed" if complete_epoch else "epoch_stopped",
                            epoch=epoch + 1, epochs=actual_epochs, step=steps, loss=final_loss,
                            weighted_loss=weighted_total / epoch_samples,
                            optimization_loss=optimization_total / epoch_samples,
                            objective_loss=objective_total / epoch_samples,
                            ranking_loss=(ranking_total / epoch_samples if self.ranking_loss_weight else None),
                            epoch_samples=epoch_samples, samples_seen=samples_seen,
                            complete_epoch=complete_epoch, learning_rate=self._learning_rate(steps),
                            resources=device_resources(self.device))
                observe(epoch + 1 if complete_epoch else actual_epochs)
            if validation_inputs is not None:
                vx, vy = validation_inputs
                self.network.eval()
                squared_error = 0.0
                with torch.inference_mode():
                    for offset_validation in range(0, len(vy), self.config["batch_size"]):
                        xb = torch.from_numpy(
                            vx[offset_validation:offset_validation + self.config["batch_size"]]).to(self.device)
                        yb = torch.from_numpy(
                            vy[offset_validation:offset_validation + self.config["batch_size"]]).to(self.device)
                        squared_error += torch.nn.functional.mse_loss(
                            self.network(xb) * self.target_scale, yb, reduction="sum").item()
                self._event("validation", "epoch_completed", epoch=epoch + 1,
                            loss=squared_error / len(vy), samples=len(vy), checkpoint_selection=False,
                            resources=device_resources(self.device))
            if stop_reason != "epochs":
                break
        self.trained = True
        self.exposure_counts = exposure_counts
        self.steps_per_epoch = dict(steps_per_epoch)
        optimizer_name = {"none": "Adam_MSE", "train_variance": "Adam_time_weighted_MSE",
                          "date_decay_10d": "Adam_date_weighted_MSE"}[self.config["time_loss_weighting"]]
        if self.ranking_loss_weight:
            optimizer_name = f"{optimizer_name}+{self.ranking_loss_weight:g}x_pairwise_hinge"
        return {"loss": final_loss, "learning_rate": self._learning_rate(steps),
                "lr_schedule": self.config["lr_schedule"],
                "weighted_loss": (weighted_total / epoch_samples if epoch_samples else None),
                "optimization_loss": (optimization_total / epoch_samples if epoch_samples else None),
                "objective_loss": (objective_total / epoch_samples if epoch_samples else None),
                "ranking_loss": (ranking_total / epoch_samples if self.ranking_loss_weight
                                 and epoch_samples else None),
                "ranking_loss_weight": self.ranking_loss_weight,
                "ranking_loss_kind": self.config["ranking_loss"],
                "sampling": self.config["sampling"],
                "target_scale": self.target_scale,
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
                "optimizer": optimizer_name,
                "steps": steps, "samples": len(y), "samples_seen": samples_seen,
                "epochs": actual_epochs, "epochs_completed": epochs_completed,
                "resumed": resume is not None, "track_exposure": exposure_counts is not None,
                "usable_samples": usable, "stop_reason": stop_reason,
                "finite_check_every": self.finite_check_every, "compile": compile_report}

    def exposure_report(self, *, declared: dict | None = None) -> dict:
        """Per-day and per-epoch sample exposure for the trajectory that just ran."""
        if self.exposure_counts is None:
            raise ValueError("Exposure tracking was not enabled for this fit")
        return _trajectory_helpers().exposure_report(
            self.training_keys, self.used_positions,
            self.exposure_counts[self.used_positions],
            batch_size=self.config["batch_size"],
            steps=sum(self.steps_per_epoch.values()),
            per_epoch_steps=self.steps_per_epoch, declared=declared)


    def predict(self, batch):
        if not self.trained:
            raise ValueError("Model is not fitted")
        return self._predict(batch)

    def _predict(self, batch):
        """Fixed-weight scores for epoch diagnostics; public inference still requires a fitted model.

        ``batch`` is a materialised :class:`jinyiquant.data.Batch` or a
        :class:`jinyiquant.pool.StreamingPool`. The pool has no window tensor to score in one
        pass, so a bounded key chunk is windowed and scored at a time, and the chunk is a whole
        number of forward slices: a row's score comes out of a device kernel whose tiling depends
        on how many rows it is fed, so scoring the pool in ``batch_size`` slices aligned to the
        pool start is what keeps the two paths bitwise equal.
        """
        pool = _streaming_pool(batch)
        if pool is None:
            return self._score(self._input(batch), batch.observed, batch.keys)
        if self.market_head is not None:
            raise ValueError(
                "A market-context head needs one materialised Batch per same-timestamp "
                "cross-section: a streaming pool chunk is not a cross-section, so its market "
                "summaries would be computed on an arbitrary subset. Score whole "
                "cross-section batches (build_batch per day chunk) instead of the pool.")
        width = self.config["batch_size"]
        chunk_rows = max(width, -(-PREDICT_CHUNK_ROWS // width) * width)
        scores = np.empty(len(pool), dtype=np.float64)
        for start in range(0, len(pool), chunk_rows):
            rows = np.arange(start, min(start + chunk_rows, len(pool)))
            batch_chunk = pool.batch(rows)
            scores[rows] = self._score(self._input(batch_chunk), batch_chunk.observed)
        return scores

    def _score(self, x, observed, keys=None):
        """Device forward in batch-size slices, then the missing-history fallback score."""
        raw = np.empty(len(x), dtype=np.float64)
        hidden = (np.empty((len(x), self.network.representation_dim), dtype=np.float32)
                  if self.market_head is not None else None)
        if hidden is not None and keys is None:
            raise ValueError("A market-context head needs the batch keys for its timestamps")
        self.network.eval()
        with torch.inference_mode():
            for offset in range(0, len(x), self.config["batch_size"]):
                xb = torch.from_numpy(x[offset:offset + self.config["batch_size"]]).to(self.device)
                if hidden is None:
                    raw[offset:offset + len(xb)] = self.network(xb).cpu().numpy()
                else:
                    score, state = self.network.forward_with_representation(xb)
                    raw[offset:offset + len(xb)] = score.cpu().numpy()
                    hidden[offset:offset + len(xb)] = state.cpu().numpy()
        missing = (observed[..., :self.lob_feature_count].any(axis=2).sum(axis=1)
                   < self.preprocessing["min_history"])
        scores = raw if hidden is None else raw + self._market_residual(hidden, keys, missing)
        scores *= self.target_scale
        # Fallback is the fitted training-target mean. For timestamp-centered training
        # this is a relative score; no future timestamp mean or evaluation label exists here.
        scores[missing] = self.intercept
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite predictions")
        return scores

    def _market_residual(self, hidden, keys, missing):
        """Market-context residual per row, evaluated once for the whole cross-section.

        Rows are put in canonical ``(date, instrument)`` order before aggregating, so a
        permutation of the batch cannot change a timestamp's summary, and a row's residual
        depends only on its own representation and its own timestamp's summary -- never on how
        the device forward was sliced or on any label.
        """
        head = self.market_head
        order, codes, groups, counts = timestamp_codes(keys)
        if not getattr(self, "allow_cross_section_fragment", False):
            smallest = cross_section_floor(codes, groups)
            if smallest < MIN_CROSS_SECTION:
                raise ValueError(
                    f"predict() got {len(codes)} rows whose smallest same-timestamp cross-section "
                    f"is {smallest} rows; a market-context head must see the whole cross-section in "
                    f"one call (score each replay chunk whole, never re-chunk it). Set "
                    f"adapter.allow_cross_section_fragment=True only to measure that damage.")
        summary = market_context(hidden[order], codes, valid=~np.asarray(missing, dtype=bool)[order],
                                 counts=counts, n_groups=groups)
        with torch.inference_mode():
            state = torch.from_numpy(np.ascontiguousarray(hidden[order])).to(self.device)
            if head.arm == "C":
                per_row = torch.from_numpy(np.ascontiguousarray(summary[codes])).to(self.device)
                residual = head.delta(state, per_row)
            else:
                residual = head.delta(state)
        result = np.empty(len(hidden), dtype=np.float64)
        result[order] = residual.cpu().numpy().astype(np.float64)
        return result

    def validate_training_preprocessing(self, batch, sample_mask):
        self._input(batch)
        return {"passed": True, "normalization": "per_sample_BiN_no_population_fit",
                "input_features": self.preprocessing["feature_columns"], "mask_appended": False}

    def _loss_label(self):
        label = {"none": "MSE", "train_variance": "time_weighted_MSE",
                 "date_decay_10d": "date_weighted_MSE"}[self.config["time_loss_weighting"]]
        if self.ranking_loss_weight:
            label = f"{label}+{self.ranking_loss_weight:g}x_{self.config['ranking_loss']}"
        return label

    def _payload(self):
        if not self.trained:
            raise ValueError("Cannot save an unfitted model")
        payload = {"schema_version": 3 if self.new_input_contract else (2 if self.config["target_scaling"] == "train_std" else 1),
                "kind": self.config["kind"], "architecture": self.architecture,
                "model_config": self.config, "preprocessing": self.preprocessing, "seed": self.seed,
                "training_mean": self.intercept,
                "label": TARGET_LABELS[self.config["target_centering"]],
                "prediction_units": PREDICTION_UNITS[self.config["target_centering"]],
                "loss": self._loss_label(),
                "sampling": self.config["sampling"],
                "target_scale": self.target_scale,
                "target_centering": self.config["target_centering"],
                "target_centering_stats": self.target_centering_stats,
                "time_loss_weighting_stats": self.time_loss_weighting_stats,
                "torch_version": str(torch.__version__),
                "state_dict": {k: v.detach().cpu() for k, v in self.network.state_dict().items()}}
        # Default-off: only a model carrying a market-context head writes this block, so every
        # existing checkpoint keeps its exact payload layout.
        if self.market_head is not None:
            payload["market_head"] = head_payload(self.market_head)
        return payload

    def save(self, path):
        torch.save(self._payload(), Path(path))

    def export_text(self, path):
        payload = self._payload()
        payload["state_dict"] = _text_state(payload["state_dict"])
        if payload.get("market_head") is not None:
            payload["market_head"]["state_dict"] = _text_state(payload["market_head"]["state_dict"])
        Path(path).write_text(json.dumps(payload, allow_nan=False, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path, *, device="cpu", recorder=None):
        path = Path(path)
        if path.suffix == ".json":
            with path.open(encoding="utf-8") as source:
                payload = json.load(source)
            if payload.get("kind") == SHARDED_BLEND_KIND:
                return _load_sharded_blend(cls, path, payload, device=device, recorder=recorder)
        else:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls.from_payload(payload, device=device, recorder=recorder, text_state=path.suffix == ".json")

    @classmethod
    def from_payload(cls, payload, *, device="cpu", recorder=None, text_state=False):
        if payload.get("kind") == BLEND_KIND:
            return _blend_from_payload(cls, payload, device=device, recorder=recorder)
        if payload["schema_version"] not in (1, 2, 3) or payload["kind"] not in BACKBONES:
            raise ValueError("Unsupported checkpoint format")
        config = {**payload["model_config"], "device": device}
        config.setdefault("kind", payload["kind"])
        if config["kind"] != payload["kind"]:
            raise ValueError("Checkpoint kind and model configuration disagree")
        if payload["schema_version"] == 3 and not {"book_representation", "trade_features"}.issubset(config):
            raise ValueError("Checkpoint is missing its input transformation contract")
        result = cls(config, payload["preprocessing"], seed=payload["seed"], recorder=recorder)
        if payload.get("target_centering", "none") != result.config["target_centering"]:
            raise ValueError("Checkpoint target-centering contract mismatch")
        if result.new_input_contract and payload["schema_version"] < 3:
            raise ValueError("LOB input transformations require checkpoint schema_version=3")
        if payload["schema_version"] == 1 and result.config["target_scaling"] != "none":
            raise ValueError("Target scaling requires checkpoint schema_version=2")
        if result.architecture != _architecture_with_patch_counts(payload["architecture"]):
            raise ValueError("Checkpoint architecture/preprocessing mismatch")
        state = payload["state_dict"]
        if text_state:
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
        if payload.get("market_head") is not None:
            # Additive block: an attached residual head is rebuilt and loaded strictly, and its
            # rows are scored through the market summary of their own timestamp.
            result.market_head = head_from_payload(payload["market_head"],
                                                   text_state_weights=text_state).to(result.device)
            result.market_head.eval()
        result.trained = True
        return result


def _load_sharded_blend(adapter_cls, path, payload, *, device, recorder):
    """Read one text checkpoint at a time, retaining only loaded model tensors."""
    entries = payload.get("members")
    if payload.get("schema_version") != 1 or not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("Sharded blend requires schema 1 and at least two members")
    members, weights, names, files = [], [], [], set()
    root = path.resolve().parent
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Malformed sharded blend member")
        name, filename, weight = entry.get("name"), entry.get("file"), entry.get("weight")
        if (not isinstance(name, str) or not name or name in names
                or not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight < 0):
            raise ValueError("Sharded blend requires unique names and finite nonnegative weights")
        if (not isinstance(filename, str) or not filename.startswith("modified_tlob_weights_")
                or not filename.endswith(".json") or any(char in filename for char in "/\\:")
                or filename in files):
            raise ValueError("Unsafe or duplicate weight shard filename")
        shard = root / filename
        if shard.is_symlink() or shard.resolve().parent != root:
            raise ValueError("Weight shard must be a regular file beside its manifest")
        size = shard.stat().st_size
        if size != entry.get("bytes") or size >= UPLOAD_WEIGHT_LIMIT:
            raise ValueError(f"Weight shard size mismatch or upload limit exceeded: {filename}")
        digest = hashlib.sha256()
        with shard.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry.get("sha256"):
            raise ValueError(f"Weight shard checksum mismatch: {filename}")
        with shard.open(encoding="utf-8") as source:
            member_payload = json.load(source)
        if member_payload.get("kind") not in BACKBONES:
            raise ValueError("Weight shards must contain plain torch backbone checkpoints")
        members.append(adapter_cls.from_payload(member_payload, device=device, recorder=recorder,
                                                text_state=True))
        del member_payload
        names.append(name)
        weights.append(float(weight))
        files.add(filename)
        if recorder is not None:
            recorder.event("loading", "weight_shard_loaded", member=name, path=filename, bytes=size)
    return TLOBBlend(members, weights, names=names)


def _blend_from_payload(adapter_cls, payload, *, device, recorder):
    if payload.get("schema_version") != BLEND_SCHEMA_VERSION:
        raise ValueError("Unsupported blend checkpoint schema")
    entries = payload.get("members")
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("Blend checkpoint requires at least two members")
    members, weights, names = [], [], []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
            raise ValueError("Blend member entry is malformed")
        member_payload = entry["payload"]
        if member_payload.get("kind") not in BACKBONES:
            raise ValueError("Blend members must be plain torch backbone checkpoints")
        name = entry.get("name", f"member_{index}")
        weight = entry.get("weight")
        if not isinstance(name, str) or not name or not isinstance(weight, (int, float)):
            raise ValueError("Blend member requires a string name and numeric weight")
        names.append(name)
        weights.append(float(weight))
        members.append(adapter_cls.from_payload(member_payload, device=device, recorder=recorder,
                                                text_state=True))
    return TLOBBlend(members, weights, names=names)


def _blend_shared_preprocessing(member):
    """Member preprocessing minus the lookback, which the blend slices per member."""
    return {key: value for key, value in member.preprocessing.items() if key != "lookback"}


class TLOBBlend:
    """Fixed-weight average of frozen checkpoints, scored in the shared return units.

    Member scores restore their own fitted ``target_scale`` (including the
    missing-history fallback), so the weighted sum is already in comparable units
    and no refitting, rank transform or normalization is applied at prediction time.
    Members may differ in trunk, capacity and lookback; every member must share the
    rest of preprocessing, prediction scale and the common recipe keys, and
    trunk-specific keys must match within each trunk group. The batch is built once
    at the longest lookback and each member scores its trailing rows, so a 128-bar
    checkpoint keeps consuming exactly the 128 bars it was trained on while a
    256-bar arm sees the longer shared history. ``kinds`` records the trunk
    composition in member order.
    """

    kind = BLEND_KIND

    def __init__(self, members, weights, *, names=None):
        if len(members) < 2:
            raise ValueError("Blend requires at least two members")
        if len(weights) != len(members):
            raise ValueError("Blend weights must align with members")
        weights = tuple(float(weight) for weight in weights)
        if any(not np.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Blend weights must be finite and nonnegative")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("Blend weights must sum to one")
        reference = members[0]
        groups, lookbacks = {}, []
        for index, member in enumerate(members):
            kind = member.config.get("kind")
            if kind not in BACKBONES:
                raise ValueError(f"Blend member {index} has an unsupported trunk kind")
            if not member.trained:
                raise ValueError("Blend members must be fitted checkpoints")
            if _blend_shared_preprocessing(member) != _blend_shared_preprocessing(reference):
                raise ValueError(f"Blend member {index} preprocessing differs from member 0")
            lookback = member.preprocessing.get("lookback")
            if type(lookback) is not int or lookback < 4:
                raise ValueError(f"Blend member {index} requires an integer lookback of at least 4")
            lookbacks.append(lookback)
            if member.target_scale != reference.target_scale:
                raise ValueError("Blend members must share one prediction scale")
            for key in BLEND_SHARED_KEYS:
                if member.config.get(key) != reference.config.get(key):
                    raise ValueError(f"Blend member {index} {key} differs from member 0")
            groups.setdefault(kind, []).append((index, member))
        for kind, group in groups.items():
            first_index, first = group[0]
            for index, member in group[1:]:
                for key in BACKBONE_SHARED_KEYS[kind]:
                    if member.config.get(key) != first.config.get(key):
                        raise ValueError(f"Blend member {index} {key} differs from member {first_index}")
        self.members = tuple(members)
        self.weights = weights
        self.names = tuple(names) if names else tuple(str(index) for index in range(len(members)))
        if len(self.names) != len(self.members) or len(set(self.names)) != len(self.members):
            raise ValueError("Blend member names must be unique")
        self.kinds = tuple(groups)
        longest = max(lookbacks)
        self.preprocessing = {**_blend_shared_preprocessing(reference), "lookback": longest}
        self.config = dict(reference.config)
        self.architecture = {**reference.architecture, "seq_size": longest}
        self.device = reference.device
        self.target_scale = reference.target_scale
        self.intercept = float(sum(weight * member.intercept for weight, member in zip(weights, self.members)))
        self.trained = True

    @staticmethod
    def _member_batch(member, batch):
        """Trailing rows reproduce exactly the window the member was trained on."""
        lookback = member.preprocessing["lookback"]
        if batch.x.shape[1] < lookback:
            raise ValueError(f"Blend batch carries {batch.x.shape[1]} bars but member requires {lookback}")
        if batch.x.shape[1] == lookback:
            return batch
        return replace(batch, x=batch.x[:, -lookback:], observed=batch.observed[:, -lookback:],
                       history_dates=batch.history_dates[:, -lookback:])

    def predict(self, batch):
        # Keep member mini-batches aligned with the blend contract after loading.
        batch_size = int(self.config.get("batch_size", 32))
        scores = None
        for weight, member in zip(self.weights, self.members):
            member.config["batch_size"] = batch_size
            values = np.asarray(member.predict(self._member_batch(member, batch)), dtype=np.float64)
            scores = weight * values if scores is None else scores + weight * values
        if scores is None or not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite blend predictions")
        return scores
