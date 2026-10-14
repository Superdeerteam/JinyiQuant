"""Batch training/inference and portable checkpoints with frozen preprocessing."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .device import device_resources, select_device
from .model import ModifiedTLOB
from .schema import lob_columns

PREPROCESS_KEYS = ("feature_columns", "lookback", "min_history", "missing_policy", "max_ffill_steps",
                   "missing_sentinel", "price_scale", "history_days", "sample_times", "label_horizon_minutes")


class TLOBAdapter:
    def __init__(self, model_config, data_config, *, seed=42, recorder=None):
        self.config = model_config.model_dump() if hasattr(model_config, "model_dump") else dict(model_config)
        data = data_config.model_dump(mode="json") if hasattr(data_config, "model_dump") else dict(data_config)
        self.preprocessing = {k: data[k] for k in PREPROCESS_KEYS}
        fields = tuple(self.preprocessing["feature_columns"])
        if fields not in (lob_columns(3), lob_columns(5)):
            raise ValueError("Unexpected LOB field order")
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
        self.trained = False
        if recorder:
            recorder.event("training", "determinism", seed=seed, algorithms=True, cpu_threads=torch.get_num_threads(),
                           device=str(self.device), cuda_matmul_workspace="CUBLAS_WORKSPACE_CONFIG=:4096:8",
                           note="GPU tolerance-based reproducibility; cross-device rounding may differ")

    def _input(self, batch):
        x = np.asarray(batch.x, dtype=np.float32)
        expected = (len(batch.keys), self.architecture["seq_size"], self.architecture["num_features"])
        if x.shape != expected or batch.observed.shape != x.shape or not np.isfinite(x).all():
            raise ValueError("Batch requires aligned finite preprocessed LOB windows")
        return x

    def _event(self, stage, event, **context):
        if self.recorder:
            self.recorder.event(stage, event, **context)

    def fit(self, batch, target, *, sample_mask=None, validation=None):
        x, y = self._input(batch), np.asarray(target, dtype=np.float32)
        if y.shape != (len(x),) or not len(y) or not np.isfinite(y).all():
            raise ValueError("Training requires nonempty aligned finite targets")
        mask = np.ones(len(y), dtype=bool) if sample_mask is None else np.asarray(sample_mask)
        if mask.dtype != bool or mask.shape != y.shape or not mask.any():
            raise ValueError("Training sample mask must select aligned samples")
        x, y = x[mask], y[mask]
        validation_inputs = None
        if validation is not None:
            vb, vy = validation
            vx, vy = self._input(vb), np.asarray(vy, dtype=np.float32)
            if vy.shape != (len(vx),) or not len(vy) or not np.isfinite(vy).all():
                raise ValueError("Validation requires nonempty aligned finite targets")
            validation_inputs = vx, vy
        self.intercept = float(y.astype(np.float64).mean())
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config["learning_rate"])
        generator = torch.Generator().manual_seed(self.seed)
        started, steps = time.monotonic(), 0
        for epoch in range(self.config["epochs"]):
            self.network.train()
            order = torch.randperm(len(y), generator=generator).numpy()
            total = 0.0
            for offset in range(0, len(y), self.config["batch_size"]):
                indices = order[offset:offset + self.config["batch_size"]]
                xb = torch.from_numpy(x[indices]).to(self.device)
                yb = torch.from_numpy(y[indices]).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(self.network(xb), yb)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                gradients = [p.grad for p in self.network.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("Nonfinite gradients")
                optimizer.step()
                steps += 1
                total += loss.item() * len(indices)
                if steps % self.config["log_every_steps"] == 0:
                    self._event("training", "progress", epoch=epoch + 1, step=steps, loss=loss.item(),
                                learning_rate=optimizer.param_groups[0]["lr"],
                                elapsed_seconds=time.monotonic() - started,
                                throughput_samples_per_second=(epoch * len(y) + offset + len(indices)) /
                                max(time.monotonic() - started, 1e-9), resources=device_resources(self.device))
            self._event("training", "epoch_completed", epoch=epoch + 1, step=steps, loss=total / len(y),
                        learning_rate=optimizer.param_groups[0]["lr"], resources=device_resources(self.device))
            if validation_inputs is not None:
                vx, vy = validation_inputs
                self.network.eval()
                squared_error = 0.0
                with torch.inference_mode():
                    for offset in range(0, len(vy), self.config["batch_size"]):
                        xb = torch.from_numpy(vx[offset:offset + self.config["batch_size"]]).to(self.device)
                        yb = torch.from_numpy(vy[offset:offset + self.config["batch_size"]]).to(self.device)
                        squared_error += torch.nn.functional.mse_loss(self.network(xb), yb, reduction="sum").item()
                self._event("validation", "epoch_completed", epoch=epoch + 1, loss=squared_error / len(vy),
                            samples=len(vy), checkpoint_selection=False, resources=device_resources(self.device))
        self.trained = True
        return {"loss": total / len(y), "learning_rate": self.config["learning_rate"],
                "optimizer": "Adam_MSE", "steps": steps, "samples": len(y)}

    def predict(self, batch):
        if not self.trained:
            raise ValueError("Model is not fitted")
        x = self._input(batch)
        scores = np.empty(len(x), dtype=np.float64)
        self.network.eval()
        with torch.inference_mode():
            for offset in range(0, len(x), self.config["batch_size"]):
                xb = torch.from_numpy(x[offset:offset + self.config["batch_size"]]).to(self.device)
                scores[offset:offset + len(xb)] = self.network(xb).cpu().numpy()
        missing = batch.observed.any(axis=2).sum(axis=1) < self.preprocessing["min_history"]
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
        return {"schema_version": 1, "kind": "modified_tlob", "architecture": self.architecture,
                "model_config": self.config, "preprocessing": self.preprocessing, "seed": self.seed,
                "training_mean": self.intercept, "label": "official_vwap_return", "loss": "MSE",
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
        if payload["schema_version"] != 1 or payload["kind"] != "modified_tlob":
            raise ValueError("Unsupported checkpoint format")
        config = {**payload["model_config"], "device": device}
        result = cls(config, payload["preprocessing"], seed=payload["seed"], recorder=recorder)
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
        result.trained = True
        return result
