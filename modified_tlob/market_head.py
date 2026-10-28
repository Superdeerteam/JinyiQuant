"""Market-context interaction residual head (arm C) and its context-free control (arm B).

The frozen MLPLOB trunk maps a 128x12 LOB window to an 80-d representation ``h`` (the last
hidden activation) and then to a scalar score ``s_base``. Only ``h`` is visible to the scalar
head, so ``h`` is the natural interface for anything added on top of the frozen candidate.

Two heads are trained on top of that frozen pair ``(h, s_base)``:

* arm ``B`` -- context-free control: ``ds = MLP_B(h)``
* arm ``C`` -- market interaction: ``ds = w^T (P h  .  G(m_t))``

with ``m_t = [mean_j h_jt, std_j h_jt, coverage_t]`` summarising the same-timestamp
cross-section of the stock pool. Both heads zero-initialise their output layer, so at step 0
``s_base + ds == s_base`` bit for bit and the training start is the frozen candidate.

Contract the acceptance checks rely on
--------------------------------------
1. ``market_context`` is a pure function of one timestamp's rows: float64 accumulation over
   rows the caller has sorted canonically, no label and no score enters it, and a row without
   enough history is excluded from the summary while still counting in the coverage denominator
   (the frozen model scores such rows with its fallback, never with the head).
2. ``delta`` never reduces across rows: row ``i`` uses ``h[i]`` and ``m[group[i]]`` only. The
   caller therefore evaluates the residual for the whole cross-section in one call, which keeps
   a row's residual independent of how the trunk forward was sliced.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch import nn

REPRESENTATION_DIM = 80
CONTEXT_WIDTH = 16
CONTEXT_HIDDEN = 64
SCALAR_HIDDEN = 156
CONTEXT_DIM = 2 * REPRESENTATION_DIM + 1
CONTEXT_PARTS = ("mean", "std", "coverage")
ARMS = ("B", "C")

# A same-timestamp cross-section of a real prediction universe holds ~1e3 instruments. A batch
# whose smallest timestamp group is far below that has almost certainly been re-chunked, which
# would summarise m_t over a fragment instead of the cross-section; adapters refuse it unless the
# caller asks for the damage explicitly.
MIN_CROSS_SECTION = 64


# --------------------------------------------------------------------------- market summary
def market_context(h, group, *, valid=None, counts=None, n_groups=None):
    """Same-timestamp cross-section summary of the representation.

    Parameters
    ----------
    h : array [rows, D] float
        Representation of every row of the batch, in the caller's canonical order.
    group : array [rows] int
        Timestamp code in ``[0, n_groups)`` for every row.
    valid : array [rows] bool, optional
        Rows that enter the mean/std and the coverage numerator. Defaults to all rows. A row
        without enough history is ``False``; it still counts in the coverage denominator,
        because the denominator is the size of the same-timestamp cross-section, not of the
        subset the head can score. A row whose representation is not finite is dropped from
        the summary as well.
    counts : array [n_groups] int, optional
        Total rows per timestamp (the coverage denominator). Defaults to the number of rows
        carrying each code, which assumes the batch *is* the whole cross-section; a caller that
        scores a subset must pass ``counts`` or coverage degenerates to one. Every caller in
        this repository passes it.
    n_groups : int, optional
        Number of timestamps. Defaults to ``group.max() + 1``.

    Returns
    -------
    context : array [n_groups, 2 * D + 1] float64
        ``[mean | std | coverage]``. A timestamp with no valid row gets mean/std 0 and
        coverage 0; the head is never evaluated on its rows (they take the frozen fallback).
    """
    h = np.asarray(h, dtype=np.float64)
    group = np.asarray(group)
    if h.ndim != 2 or not len(h):
        raise ValueError("market_context needs a nonempty [rows, dim] representation")
    if group.shape != (len(h),) or group.dtype.kind not in "iu":
        raise ValueError("market_context needs an integer timestamp code per row")
    if len(h) and (group.min() < 0 or (n_groups is not None and group.max() >= n_groups)):
        raise ValueError("market_context timestamp codes must lie in [0, n_groups)")
    if valid is None:
        valid = np.ones(len(h), dtype=bool)
    valid = np.asarray(valid)
    if valid.dtype != bool or valid.shape != (len(h),):
        raise ValueError("market_context valid mask must be a bool array aligned with h")
    finite = np.isfinite(h).all(axis=1)
    if not finite.all():
        # A non-finite representation cannot be summarised; drop it from the mean/std rather
        # than poisoning its whole timestamp. The row keeps its own residual (computed from its
        # own representation) and is caught by the caller's non-finite prediction guard.
        valid = valid & finite
    n_groups = int(group.max()) + 1 if n_groups is None else int(n_groups)
    total = (np.bincount(group, minlength=n_groups) if counts is None
             else np.asarray(counts, dtype=np.float64))
    if total.shape != (n_groups,):
        raise ValueError("market_context counts must have one entry per timestamp")
    width = h.shape[1]
    # Masking by multiplication would turn a NaN row into NaN*0 = NaN and poison the whole
    # timestamp, so the excluded rows enter the sums as exact zeros instead.
    weighted = np.where(valid[:, None], h, 0.0)
    seen = np.bincount(group, weights=valid.astype(np.float64), minlength=n_groups)
    context = np.zeros((n_groups, 2 * width + 1), dtype=np.float64)
    sums = np.zeros((n_groups, width), dtype=np.float64)
    squares = np.zeros((n_groups, width), dtype=np.float64)
    np.add.at(sums, group, weighted)
    np.add.at(squares, group, weighted * weighted)
    present = seen > 0
    mean = np.zeros_like(sums)
    np.divide(sums, seen[:, None], out=mean, where=present[:, None])
    mean_square = np.zeros_like(squares)
    np.divide(squares, seen[:, None], out=mean_square, where=present[:, None])
    variance = np.maximum(mean_square - mean * mean, 0.0)
    context[:, :width] = mean
    context[:, width:2 * width] = np.sqrt(variance)
    context[:, 2 * width] = np.divide(seen, total, out=np.zeros(n_groups), where=total > 0)
    if not np.isfinite(context).all():
        raise FloatingPointError("Nonfinite market context")
    return context


def cross_section_floor(codes, n_groups: int) -> int:
    """Fewest rows any timestamp of this batch contributes to its own summary."""
    per_group = np.bincount(np.asarray(codes), minlength=n_groups)
    return int(per_group.min()) if len(per_group) else 0


def timestamp_codes(keys, *, instrument_column="instrument", date_column="date"):
    """Canonical timestamp codes for a key table.

    Rows are ordered by ``(date, instrument)`` before the codes are handed out, so the summary
    of a timestamp does not depend on the order the caller happens to hold its rows in. Returns
    ``(order, codes, n_groups, counts)`` where ``order`` maps canonical position to input row.
    """
    import pandas as pd

    dates = pd.to_datetime(keys[date_column]).to_numpy()
    instruments = keys[instrument_column].astype(str).to_numpy()
    order = np.lexsort((instruments, dates))
    _, codes = np.unique(dates[order], return_inverse=True)
    counts = np.bincount(codes, minlength=int(codes.max()) + 1 if len(codes) else 0)
    return order, codes, len(counts), counts


# --------------------------------------------------------------------------------- heads
def _cast(value, reference):
    return value.to(device=reference.device, dtype=reference.dtype)


class MarketInteractionHead(nn.Module):
    """Arm C: multiplicative interaction between the stock and its same-timestamp market."""

    arm = "C"

    def __init__(self, *, represent_dim=REPRESENTATION_DIM, width=CONTEXT_WIDTH,
                 context_hidden=CONTEXT_HIDDEN):
        super().__init__()
        self.spec = {"arm": self.arm, "represent_dim": int(represent_dim), "width": int(width),
                     "context_hidden": int(context_hidden)}
        self.project = nn.Linear(self.spec["represent_dim"], self.spec["width"], bias=False)
        self.context = nn.Sequential(nn.Linear(CONTEXT_DIM, self.spec["context_hidden"]), nn.GELU(),
                                     nn.Linear(self.spec["context_hidden"], self.spec["width"]))
        self.mix = nn.Linear(self.spec["width"], 1, bias=False)
        nn.init.zeros_(self.mix.weight)

    def delta(self, h, m):
        """Residual score [rows] from representations [rows, D] and per-row contexts [rows, 161]."""
        if h.ndim != 2 or m.ndim != 2 or m.shape[0] != h.shape[0]:
            raise ValueError("delta needs aligned [rows, D] and [rows, context] inputs")
        return self.mix(self.project(_cast(h, self.project.weight))
                        * self.context(_cast(m, self.context[0].weight))).reshape(-1)


class ScalarResidualHead(nn.Module):
    """Arm B: context-free residual on the same representation, comparable parameter count."""

    arm = "B"

    def __init__(self, *, represent_dim=REPRESENTATION_DIM, scalar_hidden=SCALAR_HIDDEN):
        super().__init__()
        self.spec = {"arm": self.arm, "represent_dim": int(represent_dim),
                     "scalar_hidden": int(scalar_hidden)}
        self.layers = nn.Sequential(nn.Linear(self.spec["represent_dim"], self.spec["scalar_hidden"]),
                                    nn.GELU(),
                                    nn.Linear(self.spec["scalar_hidden"], 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def delta(self, h, m=None):
        """Residual score [rows]; ``m`` is accepted and ignored so both arms share one call."""
        if h.ndim != 2:
            raise ValueError("delta needs a [rows, D] representation")
        return self.layers(_cast(h, self.layers[0].weight)).reshape(-1)


def build_head(arm, *, represent_dim=REPRESENTATION_DIM, width=CONTEXT_WIDTH,
               context_hidden=CONTEXT_HIDDEN, scalar_hidden=SCALAR_HIDDEN, hidden=None,
               ctx_dim=None):
    """Build one arm. ``hidden``/``ctx_dim`` are accepted aliases used by the audit tooling."""
    if hidden is not None:
        represent_dim = hidden
    if ctx_dim is not None and int(ctx_dim) != CONTEXT_DIM:
        raise ValueError(f"ctx_dim must be {CONTEXT_DIM} ([mean | std | coverage] of 80 dims)")
    key = str(arm).upper()
    if key == "C":
        return MarketInteractionHead(represent_dim=represent_dim, width=width,
                                     context_hidden=context_hidden)
    if key == "B":
        return ScalarResidualHead(represent_dim=represent_dim, scalar_hidden=scalar_hidden)
    raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")


def parameter_count(head):
    return sum(parameter.numel() for parameter in head.parameters())


def head_payload(head):
    """Checkpoint block for a head: arm, construction spec and weights."""
    return {"arm": head.arm, "spec": dict(head.spec),
            "state_dict": {name: value.detach().cpu() for name, value in head.state_dict().items()}}


def text_state(state):
    """JSON-safe form of a state dict, matching ``TLOBAdapter.export_text``."""
    return {name: {"dtype": str(value.dtype).removeprefix("torch."), "shape": list(value.shape),
                   "data": value.reshape(-1).tolist()} for name, value in state.items()}


def decode_state(state: Mapping, text: bool):
    if not text:
        return {name: torch.as_tensor(value) for name, value in state.items()}
    return {name: torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"])).reshape(meta["shape"])
            for name, meta in state.items()}


def head_from_payload(payload, *, text_state_weights: bool = False):
    """Rebuild a head from its checkpoint block and load its weights strictly."""
    spec = dict(payload.get("spec") or {})
    head = build_head(payload["arm"], represent_dim=spec.get("represent_dim", REPRESENTATION_DIM),
                      width=spec.get("width", CONTEXT_WIDTH),
                      context_hidden=spec.get("context_hidden", CONTEXT_HIDDEN),
                      scalar_hidden=spec.get("scalar_hidden", SCALAR_HIDDEN))
    state = decode_state(payload["state_dict"], text_state_weights)
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite market-head weights")
    head.load_state_dict(state, strict=True)
    return head.eval()
