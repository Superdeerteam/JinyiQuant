"""Upstream BiN with stable zero-variance gradients and persistent parameters (MIT)."""

import torch
from torch import nn


class BiN(nn.Module):
    def __init__(self, d1, t1):
        super().__init__()
        if d1 < 2 or t1 < 2:
            raise ValueError("BiN requires at least two features and observations")
        self.d1, self.t1 = d1, t1
        self.B1 = nn.Parameter(torch.zeros(t1, 1))
        self.l1 = nn.Parameter(torch.empty(t1, 1))
        nn.init.xavier_normal_(self.l1)
        self.B2 = nn.Parameter(torch.zeros(d1, 1))
        self.l2 = nn.Parameter(torch.empty(d1, 1))
        nn.init.xavier_normal_(self.l2)
        self.y1 = nn.Parameter(torch.full((1,), .5))
        self.y2 = nn.Parameter(torch.full((1,), .5))

    @staticmethod
    def _standardize(x, dim):
        # Keep the upstream std kernel (sqrt(var) has different rounding).
        # Native std has finite zero gradients for constant slices; guard both
        # axes before division, without an in-place autograd mutation.
        std = x.std(dim=dim, keepdim=True, correction=1)
        denominator = torch.where(std < 1e-4, torch.ones_like(std), std)
        return (x - x.mean(dim=dim, keepdim=True)) / denominator

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.d1, self.t1):
            raise ValueError(f"BiN expects [batch, {self.d1}, {self.t1}]")
        # Upstream resets negative mixing weights to .01. Preserve that behavior
        # without replacing parameters and orphaning optimizer state.
        with torch.no_grad():
            for weight in (self.y1, self.y2):
                weight.copy_(torch.where(weight < 0, torch.full_like(weight, .01), weight))
        temporal = self.l2[None] * self._standardize(x, 2) + self.B2[None]
        feature = self.l1.T[None] * self._standardize(x, 1) + self.B1.T[None]
        return self.y1 * feature + self.y2 * temporal
