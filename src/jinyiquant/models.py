"""Small CPU baseline: preprocessing fitted on training data, saved with weights."""

from pathlib import Path

import numpy as np


class LinearModel:
    def __init__(self, kind="ridge", alpha=1.0):
        self.kind, self.alpha = kind, alpha
        self.mean = self.scale = self.weights = None
        self.intercept = None

    @staticmethod
    def features(batch):
        # Missingness is an explicit input; no target object enters prediction.
        return np.concatenate([batch.x.reshape(len(batch.x), -1),
                               batch.observed.reshape(len(batch.x), -1).astype(float)], axis=1)

    def fit(self, batch, target, *, sample_mask=None):
        x = self.features(batch)
        y = np.asarray(target, dtype=float)
        if len(y) == 0 or y.shape != (len(x),) or not np.isfinite(y).all() or not np.isfinite(x).all():
            raise ValueError("Training requires nonempty aligned finite features and targets")
        if sample_mask is not None:
            mask = np.asarray(sample_mask)
            if mask.dtype != bool or mask.shape != (len(y),) or not mask.any():
                raise ValueError("Training sample mask must select at least one aligned sample")
            x, y = x[mask], y[mask]
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        z = (x - self.mean) / self.scale
        self.intercept = float(y.mean())
        if self.kind == "mean":
            self.weights = np.zeros(x.shape[1])
        else:
            self.weights = np.linalg.solve(z.T @ z + self.alpha * np.eye(z.shape[1]),
                                           z.T @ (y - self.intercept))
        prediction = (x - self.mean) / self.scale @ self.weights + self.intercept
        return {"loss": float(np.mean((prediction - y) ** 2)), "learning_rate": None,
                "optimizer": "closed_form_ridge", "steps": 1, "samples": len(y)}

    def predict(self, batch):
        if self.weights is None:
            raise ValueError("Model is not fitted")
        prediction = (self.features(batch) - self.mean) / self.scale @ self.weights + self.intercept
        # Missing all history is not a zero-price observation. Use only the
        # training target mean; never consult future labels to pick a fallback.
        missing = ~batch.observed.any(axis=(1, 2))
        prediction[missing] = self.intercept
        return prediction

    def save(self, path: Path):
        np.savez(path, mean=self.mean, scale=self.scale, weights=self.weights,
                 intercept=self.intercept, kind=self.kind, alpha=self.alpha)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as state:
            model = cls(str(state["kind"]), float(state["alpha"]))
            model.mean, model.scale, model.weights = (state[k].copy() for k in ("mean", "scale", "weights"))
            model.intercept = float(state["intercept"])
        return model
