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
from .model import ModifiedTLOB
from .schema import validate_input_schema

PREPROCESS_KEYS = ("feature_columns", "lookback", "min_history", "missing_policy", "max_ffill_steps",
                   "missing_sentinel", "price_scale", "history_days", "sample_times", "label_horizon_minutes")
DATE_DECAY_HALF_LIFE_TRADING_DAYS = 10.0


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
        if self.config["target_centering"] not in ("none", "timestamp_mean_train"):
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
        self.intercept = float(y.astype(np.float64).mean())
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
                                loss_units=("centered_label_mse" if centering_stats is not None
                                            else "raw_label_mse"),
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
                "mask_excluded": int((~mask).sum()),
                "loss_units": ("centered_return_squared" if centering_stats is not None
                               else "original_return_squared"),
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

    def _predict(self, batch):
        """Fixed-weight scores for epoch diagnostics; public inference still requires a fitted model."""
        x = self._input(batch)
        scores = np.empty(len(x), dtype=np.float64)
        self.network.eval()
        with torch.inference_mode():
            for offset in range(0, len(x), self.config["batch_size"]):
                xb = torch.from_numpy(x[offset:offset + self.config["batch_size"]]).to(self.device)
                scores[offset:offset + len(xb)] = self.network(xb).cpu().numpy()
        scores *= self.target_scale
        missing = (batch.observed[..., :self.lob_feature_count].any(axis=2).sum(axis=1)
                   < self.preprocessing["min_history"])
        # Fallback is the fitted training-target mean. For timestamp-centered training
        # this is a relative score; no future timestamp mean or evaluation label exists here.
        scores[missing] = self.intercept
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite predictions")
        return scores

    def validate_training_preprocessing(self, batch, sample_mask):
        self._input(batch)
        return {"passed": True, "normalization": "per_sample_BiN_no_population_fit",
                "input_features": self.preprocessing["feature_columns"], "mask_appended": False}

    def _payload(self):
        if not self.trained:
            raise ValueError("Cannot save an unfitted model")
        return {"schema_version": 3 if self.new_input_contract else (2 if self.config["target_scaling"] == "train_std" else 1),
                "kind": "modified_tlob", "architecture": self.architecture,
                "model_config": self.config, "preprocessing": self.preprocessing, "seed": self.seed,
                "training_mean": self.intercept,
                "label": ("official_vwap_return_minus_training_timestamp_mean"
                          if self.config["target_centering"] != "none" else "official_vwap_return"),
                "prediction_units": ("relative_return_score" if self.config["target_centering"] != "none"
                                     else "original_label_return"),
                "loss": {"none": "MSE", "train_variance": "time_weighted_MSE",
                         "date_decay_10d": "date_weighted_MSE"}[self.config["time_loss_weighting"]],
                "target_scale": self.target_scale,
                "target_centering": self.config["target_centering"],
                "target_centering_stats": self.target_centering_stats,
                "time_loss_weighting_stats": self.time_loss_weighting_stats,
                "torch_version": str(torch.__version__),
                "state_dict": {k: v.detach().cpu() for k, v in self.network.state_dict().items()}}

    def save(self, path):
        torch.save(self._payload(), Path(path))

    def export_text(self, path):
        payload = self._payload()
        payload["state_dict"] = {k: {"dtype": str(t.dtype).removeprefix("torch."), "shape": list(t.shape),
                                     "data": t.reshape(-1).tolist()} for k, t in payload["state_dict"].items()}
        Path(path).write_text(json.dumps(payload, allow_nan=False, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path, *, device="cpu", recorder=None):
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else torch.load(
            path, map_location="cpu", weights_only=True)
        if payload["schema_version"] not in (1, 2, 3) or payload["kind"] != "modified_tlob":
            raise ValueError("Unsupported checkpoint format")
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
        if result.config["target_centering"] != "none" and (
                not isinstance(result.target_centering_stats, dict)
                or result.target_centering_stats.get("method") != "timestamp_mean_train"):
            raise ValueError("Timestamp-centered checkpoint is missing its centering statistics")
        result.time_loss_weighting_stats = payload.get("time_loss_weighting_stats")
        if result.config["time_loss_weighting"] == "train_variance":
            if not isinstance(result.time_loss_weighting_stats, dict):
                raise ValueError("Weighted checkpoint is missing training weight statistics")
            result.training_sample_times = tuple(result.time_loss_weighting_stats["by_time"])
        elif result.config["time_loss_weighting"] == "date_decay_10d":
            if (not isinstance(result.time_loss_weighting_stats, dict)
                    or result.time_loss_weighting_stats.get("method") != "date_decay_10d"):
                raise ValueError("Date-weighted checkpoint is missing its training weight statistics")
        result.trained = True
        return result
