"""R14: masked-segment self-supervision on non-label minutes, then a label fine-tune.

The supervised recipe samples eight timestamp columns a day; every other minute of the same
trading history is unused by the labels. This module trains the MLPLOB trunk to reconstruct a
randomly masked contiguous span of a 128-bar window that ends at one of those non-label
minutes, then hands the trunk (everything but the scalar score head) to a fresh supervised
adapter. The reconstruction objective is fixed in advance and never consults a label.

Three properties are enforced rather than assumed:

* the pretraining keys and the pretraining bar window never reach the validation split
  (``assert_train_window`` / ``bar_window_guard``);
* pretraining minutes are disjoint from the supervised sample times (``non_label_minutes``);
* the transferred parameters are exactly the trunk, so the score head starts fresh and the
  fine-tune is an initialisation, not a continuation of the reconstruction objective
  (``apply_checkpoint``).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TASK = "masked_segment_reconstruction_v1"
STRIDE_MINUTES = 5
GRADIENT_CLIP = 1.0
# The recipe's allowed timestamp columns: 09:30-11:00, 11:30, 13:00-14:30.
ALLOWED_MINUTES = tuple(range(570, 661)) + (690,) + tuple(range(780, 871))


def clock_minutes(value: str) -> int:
    """Minutes after midnight for an ``HH:MM`` timestamp column."""
    hours, minutes = str(value).split(":")
    return int(hours) * 60 + int(minutes)


def clock_label(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def non_label_minutes(label_times, *, stride: int = STRIDE_MINUTES) -> list[str]:
    """Every allowed timestamp column the supervised recipe does not sample, on a fixed grid.

    The grid is fixed (``stride`` minutes) so the pretraining pool cannot drift with a
    hyper-parameter search, and the label columns are removed exactly, not approximately.
    """
    labels = {clock_minutes(value) for value in label_times}
    allowed = set(ALLOWED_MINUTES)
    selected = [value for value in range(min(ALLOWED_MINUTES), max(ALLOWED_MINUTES) + 1)
                if value in allowed and value % stride == 0 and value not in labels]
    if not selected:
        raise ValueError("The recipe leaves no non-label minute for pretraining")
    overlap = labels.intersection(selected)
    if overlap:  # pragma: no cover - the filter above already removes them
        raise ValueError(f"Pretraining minutes overlap the supervised columns: {sorted(overlap)}")
    return [clock_label(value) for value in selected]


def assert_train_window(keys: pd.DataFrame, *, train_start, train_end) -> dict:
    """Fail closed unless every pretraining key sits inside the training split."""
    if len(keys) == 0:
        raise ValueError("Pretraining needs at least one key")
    dates = pd.to_datetime(keys["date"])
    start, end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    earliest, latest = dates.min(), dates.max()
    if earliest < start:
        raise ValueError(f"Pretraining key {earliest} precedes the training split {start}")
    if latest >= end:
        raise ValueError(f"Pretraining key {latest} reaches the training split end {end}")
    return {"keys": int(len(keys)), "earliest": earliest.isoformat(), "latest": latest.isoformat(),
            "train_start": start.isoformat(), "train_end": end.isoformat()}


def bar_window_guard(dates, *, train_end, validation_start, sealed_start=None) -> dict:
    """Fail closed unless the pretraining bar window stops before the validation split.

    ``dates`` is the bar matrix's datetime64 column (tens of millions of rows), so the check
    reads extrema instead of materialising a pandas view of it.
    """
    values = np.asarray(dates)
    if len(values) == 0:
        raise ValueError("Pretraining bar window is empty")
    earliest, latest = values.min(), values.max()
    if latest >= np.datetime64(pd.Timestamp(train_end)):
        raise ValueError(f"Pretraining bars reach {pd.Timestamp(latest)}, at or past the "
                         "training split end")
    if latest >= np.datetime64(pd.Timestamp(validation_start)):
        raise ValueError(f"Pretraining bars reach {pd.Timestamp(latest)}, at the validation split")
    if sealed_start is not None and latest >= np.datetime64(pd.Timestamp(sealed_start)):
        raise ValueError(f"Pretraining bars reach the sealed interval at {pd.Timestamp(latest)}")
    return {"bars": int(len(values)), "earliest": pd.Timestamp(earliest).isoformat(),
            "latest": pd.Timestamp(latest).isoformat(), "validation_start": str(validation_start)}


MONTHS_FORBIDDEN_FROM = "2024-01-01"


def month_inventory(dates, *, forbidden_from: str = MONTHS_FORBIDDEN_FROM) -> dict:
    """The calendar months the training read touched, and a guard against sealed months.

    The training split ends on 2023-11-01, so a month at or past ``forbidden_from`` means the
    read reached data that the arm must never see.  ``dates`` is the streamed bar matrix, so the
    months come from a single unique pass over the datetime64 column, not a pandas frame.
    """
    values = np.asarray(dates)
    if len(values) == 0:
        raise ValueError("Training bar window is empty")
    months = np.unique(values.astype("datetime64[M]"))
    labels = [str(month) for month in months]
    permitted = forbidden_from[:7]
    forbidden = [label for label in labels if label >= permitted]
    if forbidden:
        raise ValueError(f"The training read reached months at or past {permitted}: {forbidden}")
    payload = chr(10).join(labels)
    return {"months": labels, "first": labels[0], "last": labels[-1],
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "forbidden_from": forbidden_from, "bars": int(len(values))}


def pretrain_keys(universe: pd.DataFrame, *, train_start, train_end, minutes, per_slot: int,
                  seed: int) -> pd.DataFrame:
    """One window per (trading day, non-label minute, sampled instrument).

    The instrument universe is the recipe's own independent instrument table, restricted to
    the training split; ``per_slot`` instruments are drawn without replacement per timestamp
    column with a generator seeded apart from the training seed, so the sample is part of the
    pre-registered task rather than a side effect of the model seed.
    """
    if per_slot < 1:
        raise ValueError("per_slot must be positive")
    frame = pd.DataFrame({"date": pd.to_datetime(universe["date"]),
                          "instrument": universe["instrument"].astype(str)})
    start, end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    frame = frame.loc[(frame.date >= start) & (frame.date < end)]
    if frame.empty:
        raise ValueError("The instrument universe holds no row inside the training split")
    by_day = {day.normalize(): sorted(set(group["instrument"]))
              for day, group in frame.groupby(frame.date.dt.normalize(), sort=True)}
    rng = np.random.default_rng(np.random.PCG64(np.uint64(seed) * np.uint64(0x9E3779B1)
                                               + np.uint64(0x5F356495)))
    columns = [clock_minutes(value) for value in minutes]
    rows = []
    for day, instruments in by_day.items():
        count = min(per_slot, len(instruments))
        for minute in columns:
            picked = rng.choice(len(instruments), size=count, replace=False)
            stamp = day + pd.Timedelta(minutes=int(minute))
            rows.append(pd.DataFrame({"date": stamp, "instrument": np.asarray(instruments)[picked]}))
    keys = pd.concat(rows, ignore_index=True).sort_values(
        ["date", "instrument"], kind="stable").reset_index(drop=True)
    return keys


TARGET_CLIP = 5.0


def mask_windows(x: np.ndarray, observed: np.ndarray, *, mask_length: int, generator,
                 clip: float = TARGET_CLIP) -> dict:
    """Replace one contiguous span per window with the observed context mean.

    The masked span carries no information about itself: every feature of every masked bar is
    overwritten by the mean of that feature over the window's observed, unmasked bars. The
    reconstruction target is the same span standardised per window and feature by the observed
    context, so the loss does not depend on price level or instrument volatility. Features
    without two observed context bars carry no target and are dropped from the loss.

    The standardised target is clipped to ``+-clip``: a context whose deviation sits just above
    the usability floor would otherwise turn a one-tick move into a target of hundreds, and one
    such window dominates the batch. Clipping keeps the reconstruction target on the scale of
    the context deviation, which is what the trunk is asked to model.
    """
    values = np.asarray(x, dtype=np.float32)
    seen = np.asarray(observed, dtype=bool)
    if values.ndim != 3 or seen.shape != values.shape:
        raise ValueError("Masking needs aligned [window, bars, features] tensors")
    windows, bars, features = values.shape
    if not 0 < mask_length < bars:
        raise ValueError(f"mask_length must be in (0, {bars})")
    starts = generator.integers(0, bars - mask_length + 1, size=windows)
    index = np.arange(bars)[None, :]
    masked = (index >= starts[:, None]) & (index < (starts + mask_length)[:, None])
    context = seen & ~masked[:, :, None]
    counts = context.sum(axis=1)                                    # [windows, features]
    totals = np.where(context, values, 0.0).sum(axis=1)
    means = totals / np.maximum(counts, 1)
    centred = np.where(context, values - means[:, None, :], 0.0)
    variance = (centred ** 2).sum(axis=1) / np.maximum(counts - 1, 1)
    deviation = np.sqrt(np.maximum(variance, 0.0))
    usable = (counts >= 2) & (deviation > 1e-3)
    scale = np.where(usable, np.maximum(deviation, 1e-3), 1.0)
    inputs = np.where(masked[:, :, None], means[:, None, :], values).astype(np.float32)
    standardised = (values - means[:, None, :]) / scale[:, None, :]
    targets = np.clip(standardised, -abs(clip), abs(clip)).astype(np.float32)
    weight = masked[:, :, None] & usable[:, None, :]
    if not weight.any():
        raise ValueError("Masked batch carries no reconstruction target")
    return {"inputs": inputs, "targets": targets, "weight": weight,
            "masked_fraction": float(masked.mean()),
            "target_fraction": float(weight.mean())}


def score_head_keys(network) -> tuple[str, ...]:
    """State-dict keys of the scalar score head, which pretraining never transfers."""
    layers = getattr(network, "final_layers", None)
    if layers is None:
        raise ValueError("The trunk has no final_layers; the score head cannot be identified")
    prefix = f"final_layers.{len(layers) - 1}."
    keys = tuple(key for key in network.state_dict() if key.startswith(prefix))
    if not keys:
        raise ValueError("Score head keys are empty; refusing an ambiguous transfer")
    return keys


def encoder_state(network) -> dict:
    head = set(score_head_keys(network))
    return {key: value.detach().cpu().clone() for key, value in network.state_dict().items()
            if key not in head}


def state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.numpy() if hasattr(tensor, "numpy")
                                           else np.asarray(tensor), dtype=np.float32).tobytes())
    return digest.hexdigest()


def save_checkpoint(path: Path, network, decoder, spec: "PretrainSpec", *, metadata: dict) -> dict:
    """Persist the pretrained trunk, the reconstruction decoder and the task description."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = encoder_state(network)
    record = {"schema_version": 1, "task": TASK, "spec": spec.as_mapping(),
              "encoder_keys": len(encoder), "encoder_sha256": state_digest(encoder),
              "head_keys": list(score_head_keys(network)), "metadata": metadata}
    torch.save({"schema_version": 1, "task": TASK, "encoder": encoder,
                "decoder": {key: value.detach().cpu()
                for key, value in (decoder.state_dict() if hasattr(decoder, "state_dict")
                                   else decoder).items()},
                "spec": spec.as_mapping(), "metadata": metadata}, path)
    return record


def apply_checkpoint(adapter, checkpoint_path: Path, *, recorder=None) -> dict:
    """Initialise an adapter's trunk from a pretraining checkpoint and leave the head fresh.

    The checkpoint must hold exactly one tensor per non-head parameter; a missing or extra key
    fails the run instead of silently loading a partially pretrained trunk.
    """
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    encoder = payload.get("encoder") or {}
    target = adapter.network.state_dict()
    head = set(score_head_keys(adapter.network))
    expected = {key for key in target if key not in head}
    if set(encoder) != expected:
        missing = sorted(expected - set(encoder))[:5]
        extra = sorted(set(encoder) - expected)[:5]
        raise ValueError(f"Pretraining checkpoint does not match the trunk "
                         f"(missing={missing}, extra={extra})")
    fresh = {key: target[key].detach().cpu().clone() for key in head}
    merged = {key: target[key] for key in target}
    merged.update({key: value.to(target[key].dtype) for key, value in encoder.items()})
    adapter.network.load_state_dict(merged, strict=True)
    after = adapter.network.state_dict()
    for key, value in fresh.items():
        if not torch.equal(value, after[key].cpu()):
            raise ValueError(f"Score head {key!r} changed during the trunk transfer")
    record = {"checkpoint": Path(checkpoint_path).as_posix(),
              "encoder_keys": len(encoder), "encoder_sha256": payload.get("metadata", {})
              .get("encoder_sha256", state_digest(encoder)),
              "head_keys": sorted(head), "head_source": "fresh_reinitialised",
              "task": payload.get("task")}
    if recorder is not None:
        recorder.event("pretrain", "trunk_transferred", **record)
    return record


@dataclass(frozen=True)
class PretrainSpec:
    """The pre-registered pretraining task: fixed steps, fixed mask, fixed minute grid."""

    steps: int
    batch_size: int
    learning_rate: float
    mask_length: int
    instruments_per_slot: int
    seed: int
    minutes: tuple[str, ...]

    @classmethod
    def from_mapping(cls, mapping) -> "PretrainSpec":
        missing = [key for key in ("steps", "batch_size", "learning_rate", "mask_length",
                                   "instruments_per_slot", "seed", "minutes")
                   if key not in mapping]
        if missing:
            raise ValueError(f"Pretraining plan is missing {missing}")
        spec = cls(steps=int(mapping["steps"]), batch_size=int(mapping["batch_size"]),
                   learning_rate=float(mapping["learning_rate"]),
                   mask_length=int(mapping["mask_length"]),
                   instruments_per_slot=int(mapping["instruments_per_slot"]),
                   seed=int(mapping["seed"]),
                   minutes=tuple(str(value) for value in mapping["minutes"]))
        if spec.steps < 1 or spec.batch_size < 1 or spec.instruments_per_slot < 1:
            raise ValueError("Pretraining steps, batch size and instruments must be positive")
        if not 0 < spec.learning_rate:
            raise ValueError("Pretraining learning rate must be positive")
        if not 0 < spec.mask_length < 128:
            raise ValueError("mask_length must be inside the window")
        if len(set(spec.minutes)) != len(spec.minutes):
            raise ValueError("Pretraining minutes must be unique")
        return spec

    def as_mapping(self) -> dict:
        return {"task": TASK, "steps": self.steps, "batch_size": self.batch_size,
                "learning_rate": self.learning_rate, "mask_length": self.mask_length,
                "instruments_per_slot": self.instruments_per_slot, "seed": self.seed,
                "minutes": list(self.minutes)}


def epoch_batches(count: int, batch_size: int, generator) -> list[np.ndarray]:
    """One shuffled pass over ``count`` usable rows, split into full and short batches."""
    order = generator.permutation(count)
    return [order[offset:offset + batch_size] for offset in range(0, count, batch_size)]


def run_pretraining(network, pool, spec: PretrainSpec, *, device, recorder=None,
                    log_every_steps: int = 100) -> dict:
    """Train the trunk and a reconstruction decoder on masked non-label windows.

    The loop is deliberately separate from :meth:`TLOBAdapter.fit`: no label, no target
    transform and no ranking loss exist here, and the step counter is the pre-registered
    pretraining budget, so the fine-tune can be measured against a fixed share of the total.
    """
    positions = np.flatnonzero(pool.fit_mask)
    if len(positions) < spec.batch_size:
        raise ValueError("Pretraining pool holds fewer usable rows than one batch")
    bars = int(pool.settings["lookback"])
    features = len(pool.features)
    decoder = torch.nn.Linear(network.representation_dim, bars * features).to(device)
    parameters = list(network.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=spec.learning_rate)
    generator = np.random.default_rng(np.random.PCG64(np.uint64(spec.seed) * np.uint64(0x2545F491)
                                                      + np.uint64(0x1F123BB5)))
    network.train()
    started = time.monotonic()
    steps = samples_seen = epochs = 0
    first_loss = None
    loss_curve: list[dict] = []
    last_loss = None
    while steps < spec.steps:
        batches = epoch_batches(len(positions), spec.batch_size, generator)
        for draw in batches:
            batch = pool.batch(positions[draw])
            values = np.asarray(batch.x, dtype=np.float32)
            observed = np.asarray(batch.observed, dtype=bool)
            if values.shape != (len(draw), bars, features):
                raise ValueError(f"Pretraining batch shape {values.shape} does not match "
                                 f"({len(draw)}, {bars}, {features})")
            masked = mask_windows(values, observed, mask_length=spec.mask_length,
                                 generator=generator)
            inputs = torch.from_numpy(masked["inputs"]).to(device)
            targets = torch.from_numpy(masked["targets"]).to(device)
            weight = torch.from_numpy(masked["weight"]).to(device)
            optimizer.zero_grad(set_to_none=True)
            representation = network.forward_with_representation(inputs)[1]
            prediction = decoder(representation).view(len(draw), bars, features)
            # SmoothL1 rather than squared error: the arm runs one fixed objective for a fixed
            # budget, so a single outlier window must not be able to set the trunk's scale.
            residual = torch.nn.functional.smooth_l1_loss(prediction, targets, beta=1.0,
                                                          reduction="none")
            loss = (residual * weight).sum() / weight.sum().clamp_min(1.0)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite pretraining loss")
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP))
            optimizer.step()
            steps += 1
            samples_seen += int(len(draw))
            last_loss = float(loss.detach().cpu())
            if first_loss is None:
                first_loss = last_loss
            if steps % log_every_steps == 0 or steps == spec.steps:
                entry = {"step": steps, "loss": last_loss,
                         "gradient_norm": gradient_norm,
                         "elapsed_seconds": time.monotonic() - started,
                         "samples_seen": samples_seen,
                         "throughput_samples_per_second": samples_seen /
                         max(time.monotonic() - started, 1e-9)}
                loss_curve.append(entry)
                if recorder is not None:
                    recorder.event("pretrain", "progress", level="INFO", **{
                        **entry, "mask_length": spec.mask_length, "learning_rate": spec.learning_rate,
                        "masked_fraction": masked["masked_fraction"],
                        "target_fraction": masked["target_fraction"]})
            if steps >= spec.steps:
                break
        epochs += 1
    wall = time.monotonic() - started
    decoder_state = {key: value.detach().cpu() for key, value in decoder.state_dict().items()}
    return {"schema_version": 1, "task": TASK, "steps": steps, "samples_seen": samples_seen,
            "epochs": epochs, "usable_windows": int(len(positions)),
            "batches_per_epoch": int(-(-len(positions) // spec.batch_size)),
            "first_loss": first_loss, "final_loss": last_loss,
            "wall_seconds": wall, "throughput_steps_per_second": steps / max(wall, 1e-9),
            "parameters": int(sum(p.numel() for p in parameters)),
            "decoder_parameters": int(sum(p.numel() for p in decoder.parameters())),
            "loss_curve": loss_curve, "spec": spec.as_mapping(), "decoder_state": decoder_state}
