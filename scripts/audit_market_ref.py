"""Auditor-owned reference model of the market-context head contract (T3, read-only audit).

Nothing in this module imports the candidate implementation. It exists for two reasons:

1. **Control.** A known-correct implementation of the T1 contract, so the six invariance
   checks can be shown to pass on something. If a check fails on the candidate but also
   fails here, the check itself is wrong (measuring instrument noise), not the candidate.
2. **Mutants.** Deliberately broken variants, each violating exactly one invariance, so
   every check can be shown to *detect* that violation at a magnitude far above 1e-6.
   An audit without this evidence cannot distinguish "passes" from "does not look".

The contract under audit (from the team task): for a timestamp group ``t = (date, sample_time)``

    m_t      = [mean_j h_jt , std_j h_jt , effective_coverage_t]          (80 + 80 + 1 = 161)
    u        = P h                        (80 -> 16)
    c        = G(m_t)                     (161 -> 16, two-layer MLP)
    delta_s  = w^T (u * c)                (w zero-initialised)
    s_new    = s_base + delta_s

with fallback samples (no bar history) excluded from the pool entirely, labels never
read during representation/market/score computation, and no information from any
timestamp after ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

HIDDEN = 80
CTX_RAW = 2 * HIDDEN + 1
PROJ = 16
POSITIONS = (
    "09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30",
)


# --------------------------------------------------------------------------------------
# synthetic panel
# --------------------------------------------------------------------------------------
@dataclass
class Panel:
    """One synthetic research panel, row = (date, sample_time, instrument).

    ``h`` mimics the frozen trunk's 80-dim representation (GELU output of the last
    Linear(320, 80), hence bounded below by ~-0.17 and right-skewed). ``eligible`` is
    True where the instrument has bar history; ineligible rows are the fallback samples
    that must not enter the market pool. ``y`` carries labels with NaNs where missing.
    """

    h: np.ndarray                 # [N, 80] float32
    group: np.ndarray             # [N] int, group id (0..n_groups-1)
    instrument: np.ndarray        # [N] object, instrument code
    date: np.ndarray              # [N] object, YYYY-MM-DD
    time: np.ndarray              # [N] object, HH:MM
    eligible: np.ndarray          # [N] bool
    y: np.ndarray                 # [N] float32, NaN = missing label
    base: np.ndarray              # [N] float32, frozen baseline score
    pool: np.ndarray | None = None  # [n_groups] cross-section size (coverage denominator)
    keys: list = field(default_factory=list)   # group id -> (date, time)

    def __post_init__(self):
        if self.pool is None:
            n = int(self.group.max()) + 1 if self.group.size else 0
            self.pool = np.bincount(self.group, minlength=n).astype(np.int64)
        self.pool = np.asarray(self.pool, dtype=np.int64)

    @property
    def rows(self) -> int:
        return int(self.h.shape[0])

    def copy(self) -> "Panel":
        return Panel(self.h.copy(), self.group.copy(), self.instrument.copy(),
                     self.date.copy(), self.time.copy(), self.eligible.copy(),
                     self.y.copy(), self.base.copy(), self.pool.copy(), list(self.keys))


def make_panel(*, dates=("2023-11-01", "2023-11-02", "2023-11-03"),
               times=POSITIONS, instruments=1000, fallback_fraction=0.1,
               missing_label_fraction=0.0, missing_h_fraction=0.0, seed=20261028,
               shock_future=None) -> Panel:
    """Build a synthetic panel with a realistic pool size (the official universe is 1000/date)."""
    rng = np.random.default_rng(seed)
    keys, rows_h, group, instr, date_col, time_col = [], [], [], [], [], []
    codes = [f"{i:06d}.SZ" for i in range(instruments)]
    for d in dates:
        for t in times:
            keys.append((d, t))
    for gid, (d, t) in enumerate(keys):
        # A common market factor plus idiosyncratic noise, then GELU, so the 80-dim rows
        # share cross-sectional structure the way frozen representations do.
        factor = rng.normal(0.0, 1.0, size=HIDDEN)
        z = factor[None, :] * 0.6 + rng.normal(0.0, 0.8, size=(instruments, HIDDEN))
        batch = torch.from_numpy(z.astype(np.float32))
        h = torch.nn.functional.gelu(batch).numpy()
        rows_h.append(h)
        group.append(np.full(instruments, gid, dtype=np.int64))
        instr.append(np.asarray(codes, dtype=object))
        date_col.append(np.full(instruments, d, dtype=object))
        time_col.append(np.full(instruments, t, dtype=object))
    h = np.concatenate(rows_h)
    group_arr = np.concatenate(group)
    panel = Panel(h=h.astype(np.float32), group=group_arr,
                  instrument=np.concatenate(instr), date=np.concatenate(date_col),
                  time=np.concatenate(time_col), eligible=np.ones(group_arr.size, bool),
                  y=np.zeros(group_arr.size, np.float32), base=np.zeros(group_arr.size, np.float32),
                  keys=keys)
    n = panel.rows
    if fallback_fraction:
        pick = rng.choice(n, int(round(n * fallback_fraction)), replace=False)
        panel.eligible[pick] = False
    if missing_h_fraction:
        # Non-finite representations only ever appear on rows the pool excludes (a fallback
        # sample has no representation); they must not be able to reach the summary.
        pool = np.flatnonzero(~panel.eligible)
        pick = rng.choice(pool, int(round(pool.size * missing_h_fraction)), replace=False)
        panel.h[pick] = np.nan
    panel.y = (rng.normal(0.0, 1.0, n) * 0.01).astype(np.float32)
    panel.base = rng.normal(0.0, 0.05, n).astype(np.float32)
    if missing_label_fraction:
        pick = rng.choice(n, int(round(n * missing_label_fraction)), replace=False)
        panel.y[pick] = np.nan
    if shock_future is not None:
        # Replace every row after `shock_future` (a group id) with fresh random content.
        after = panel.group > shock_future
        rng2 = np.random.default_rng(seed + 7)
        panel.h[after] = torch.nn.functional.gelu(torch.from_numpy(
            rng2.normal(0.0, 2.0, size=(int(after.sum()), HIDDEN)).astype(np.float32))).numpy()
        panel.y[after] = (rng2.normal(0.0, 1.0, size=int(after.sum())) * 5).astype(np.float32)
        panel.base[after] = rng2.normal(0.0, 0.5, size=int(after.sum())).astype(np.float32)
    return panel


def perms(n, reads, seed=0):
    """Row permutations of a panel, resampled independently each read."""
    rng = np.random.default_rng(seed)
    return [rng.permutation(n) for _ in range(reads)]


# --------------------------------------------------------------------------------------
# market summary
# --------------------------------------------------------------------------------------
def market_summary(h, group, eligible, *, accumulate="float64", include_fallback=False,
                   std_scope="group", std_unbiased=False, pool=None):
    """``m_t`` and ``coverage_t`` per row, computed only from that row's own timestamp group.

    ``accumulate``:
        float64            - order-independent reference (sums in float64)
        float32_pairwise   - torch float32 reductions (the candidate's likely default)
        float32_loop       - naive sequential float32 accumulation (order-dependent mutant)
    ``include_fallback``: mutant that lets no-history rows into the pool (and denominator).
    ``std_scope``: ``group`` (correct) or ``global`` (mutant that reads every timestamp,
    i.e. a look-ahead leak).
    """
    h = np.asarray(h, dtype=np.float32)
    group = np.asarray(group)
    eligible = np.asarray(eligible, bool)
    pool_mask = np.ones(group.size, bool) if include_fallback else eligible
    finite = np.isfinite(h).all(axis=1)   # a row without a usable representation is dropped
    valid = pool_mask & finite
    n_groups = int(group.max()) + 1 if group.size else 0
    if pool is None:
        n_pool = np.bincount(group[pool_mask], minlength=n_groups).astype(np.float64)
    else:
        # The coverage denominator is the same-timestamp cross-section size, which the caller
        # knows from the universe even for rows it never materialised.
        n_pool = np.asarray(pool, dtype=np.float64)
    n_valid = np.bincount(group[valid], minlength=n_groups).astype(np.float64)
    if accumulate == "float64":
        num = np.zeros((n_groups, h.shape[1]), np.float64)
        np.add.at(num, group[valid], h[valid].astype(np.float64))
        mean = np.divide(num, n_valid[:, None], out=np.zeros_like(num), where=n_valid[:, None] > 0)
        sq = np.zeros((n_groups, h.shape[1]), np.float64)
        np.add.at(sq, group[valid], (h[valid].astype(np.float64) - mean[group[valid]]) ** 2)
        denom = np.maximum(n_valid - (1.0 if std_unbiased else 0.0), 1.0)   # ddof=0 by default
        std = np.sqrt(np.divide(sq, denom[:, None], out=np.zeros_like(sq),
                                where=n_valid[:, None] > 0))
    elif accumulate == "float32_pairwise":
        mean, std = _torch_stats(h, group, valid, n_groups, std_unbiased)
    elif accumulate == "float32_loop":
        mean, std = _loop_stats(h, group, valid, n_groups)
    else:
        raise ValueError(f"unknown accumulate={accumulate!r}")
    if std_scope == "global":
        # Mutant: the standard deviation is taken over the whole panel, so a timestamp's
        # context depends on timestamps after it.
        g_mean = np.nanmean(h[valid], axis=0).astype(np.float32)
        g_std = np.nanstd(h[valid], axis=0).astype(np.float32)
        mean = np.broadcast_to(g_mean, mean.shape).copy()
        std = np.broadcast_to(g_std, std.shape).copy()
    coverage = np.divide(n_valid, n_pool, out=np.zeros(n_groups), where=n_pool > 0)
    m_group = np.concatenate([mean, std, coverage[:, None]], axis=1).astype(np.float32)
    return m_group[group], coverage[group]


def panel_take(panel: "Panel", idx, *, keep_pool=False) -> "Panel":
    """Row subset. ``keep_pool`` keeps the coverage denominator of the surviving groups.

    Dropping rows the caller never materialised (fallback samples) must not change any
    surviving row's context, so the denominator stays the universe cross-section size;
    dropping real pool members shrinks the cross-section, so the denominator follows.
    """
    idx = np.asarray(idx)
    keep_groups = np.unique(panel.group[idx])
    remap = {int(g): i for i, g in enumerate(keep_groups)}
    sub = Panel(panel.h[idx], np.asarray([remap[int(g)] for g in panel.group[idx]], np.int64),
                panel.instrument[idx], panel.date[idx], panel.time[idx], panel.eligible[idx],
                panel.y[idx], panel.base[idx],
                panel.pool[keep_groups] if keep_pool
                else np.bincount(np.asarray([remap[int(g)] for g in panel.group[idx]], np.int64),
                                 minlength=len(keep_groups)).astype(np.int64),
                [panel.keys[int(g)] for g in keep_groups])
    return sub


def _torch_stats(h, group, valid, n_groups, std_unbiased):
    t = torch.from_numpy(np.nan_to_num(h, nan=0.0))
    g = torch.from_numpy(group)
    v = torch.from_numpy(valid)
    num = torch.zeros(n_groups, h.shape[1], dtype=torch.float32)
    num.index_add_(0, g[v], t[v])
    n = torch.zeros(n_groups, dtype=torch.float32)
    n.index_add_(0, g[v], torch.ones(int(v.sum()), dtype=torch.float32))
    mean = num / n.clamp_min(1.0)[:, None]
    dev = torch.zeros_like(num)
    dev.index_add_(0, g[v], (t[v] - mean[g[v]]) ** 2)
    denom = (n - (1.0 if std_unbiased else 0.0)).clamp_min(1.0)
    std = torch.sqrt(dev / denom[:, None])
    return mean.numpy(), std.numpy()


def _loop_stats(h, group, valid, n_groups):
    """Naive sequential float32 accumulation; summation order depends on row order."""
    mean = np.zeros((n_groups, h.shape[1]), np.float32)
    counts = np.zeros(n_groups, np.float32)
    for i in range(group.size):
        if not valid[i]:
            continue
        mean[group[i]] += h[i]
        counts[group[i]] += 1.0
    mean = mean / np.maximum(counts, 1.0)[:, None]
    var = np.zeros((n_groups, h.shape[1]), np.float32)
    for i in range(group.size):
        if not valid[i]:
            continue
        var[group[i]] += (h[i] - mean[group[i]]) ** 2
    var = var / np.maximum(counts - 1.0, 1.0)[:, None]
    return mean, np.sqrt(var)


def ctx_dim() -> int:
    return CTX_RAW


# --------------------------------------------------------------------------------------
# head
# --------------------------------------------------------------------------------------
class MarketHead(nn.Module):
    """C arm (market interaction) or B arm (context-free scalar head)."""

    def __init__(self, arm="C", *, hidden=HIDDEN, ctx_raw=CTX_RAW, proj=PROJ, width=32,
                 zero_init_w=True, freeze_w=False, seed=11):
        super().__init__()
        if arm not in {"B", "C"}:
            raise ValueError("arm must be 'B' or 'C'")
        self.arm = arm
        gen = torch.Generator().manual_seed(seed)
        self.P = nn.Linear(hidden, proj)
        if arm == "C":
            self.G = nn.Sequential(nn.Linear(ctx_raw, width), nn.GELU(), nn.Linear(width, proj))
            w = torch.zeros(proj) if zero_init_w else torch.randn(proj, generator=gen) * 0.02
            self.w = nn.Parameter(w)
            if freeze_w:
                self.w.requires_grad_(False)
        else:
            # Parameter budget comparable to the C arm's projection + gate MLP.
            self.M = nn.Sequential(nn.Linear(hidden, proj), nn.GELU(), nn.Linear(proj, 1))

    def delta(self, h, m=None):
        """Residual score [rows]; mirrors the candidate head's ``delta`` entry point."""
        return self.forward(h, m)

    def forward(self, h, m=None):
        if self.arm == "B":
            return self.M(h).squeeze(-1)
        u = self.P(h)
        c = self.G(m)
        return (u * c * self.w).sum(-1)


# --------------------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------------------
class ReferenceTarget:
    """Known-correct control; every invariance check must pass here."""

    name = "reference"

    def __init__(self, *, accumulate="float64", include_fallback=False, std_scope="group",
                 zero_init_w=True, freeze_w=False, label_dropna=False, std_unbiased=False,
                 seed=11):
        torch.set_num_threads(1)
        self._init = dict(accumulate=accumulate, include_fallback=include_fallback,
                          std_scope=std_scope, zero_init_w=zero_init_w, freeze_w=freeze_w,
                          label_dropna=label_dropna, std_unbiased=std_unbiased, seed=seed)
        self.kw = dict(accumulate=accumulate, include_fallback=include_fallback,
                       std_scope=std_scope, std_unbiased=std_unbiased)
        self.head_c = MarketHead("C", zero_init_w=zero_init_w, freeze_w=freeze_w, seed=seed)
        self.head_b = MarketHead("B", seed=seed)
        self.label_dropna = label_dropna

    # adapter surface -------------------------------------------------------------
    def summarize(self, panel: Panel):
        if self.label_dropna:
            # Mutant: labels are inner-joined before aggregation, so a missing label
            # silently removes a stock from the market pool.
            keep = np.isfinite(panel.y)
            m, cov = market_summary(panel.h[keep], panel.group[keep], panel.eligible[keep],
                                    pool=panel.pool, **self.kw)
            full_m = np.full((panel.rows, CTX_RAW), np.nan, np.float32)
            full_cov = np.full(panel.rows, np.nan, np.float32)
            idx = np.flatnonzero(keep)
            full_m[idx] = m
            full_cov[idx] = cov
            return full_m, full_cov
        return market_summary(panel.h, panel.group, panel.eligible, pool=panel.pool,
                              **self.kw)

    def module(self, arm="C"):
        return self.head_c if arm == "C" else self.head_b

    def delta(self, panel: Panel, m, *, arm="C", h=None, base=None) -> np.ndarray:
        head = self.module(arm)
        hh = torch.from_numpy(np.nan_to_num(panel.h if h is None else h, nan=0.0))
        with torch.no_grad():
            if arm == "B":
                out = head(hh).numpy()
            else:
                mm = torch.from_numpy(np.nan_to_num(m, nan=0.0))
                out = head(hh, mm).numpy()
        return out

    def residual_weight(self):
        return self.head_c.w.detach().numpy()

    # arming ------------------------------------------------------------------------
    def arm(self, *, scale=0.02, seed=11):
        """Install a non-zero residual weight on the C head (in place, returns self).

        At step 0 a zero-initialised residual is identically zero, so *no* invariance
        check can distinguish a correct head from a broken one.  Checks 1-5 therefore run
        on an armed head (or, better, on the trained checkpoint); the zero-init property
        itself is audited separately, on a freshly built head.
        """
        w = self.head_c.w.detach()
        if float(w.abs().max()) == 0.0:
            gen = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                w.copy_(torch.randn(w.shape, generator=gen) * scale)
        return self

    def fresh(self):
        """A freshly built target of the same flavour (zero-initialised where applicable)."""
        return ReferenceTarget(**self._init)

    def describe(self):
        return {"name": self.name, **{k: v for k, v in self.kw.items()},
                "label_dropna": self.label_dropna}


def mutants(seed=11):
    """Name -> target, each breaking exactly one invariance."""
    return {
        "per_chunk_m": ChunkedTarget(seed=seed),
        "label_dropna": ReferenceTarget(label_dropna=True, seed=seed),
        "fallback_in_pool": ReferenceTarget(include_fallback=True, seed=seed),
        "global_std": ReferenceTarget(std_scope="global", seed=seed),
        "float32_loop_order": ReferenceTarget(accumulate="float32_loop", seed=seed),
        "nonzero_init_w": ReferenceTarget(zero_init_w=False, seed=seed),
        "std_unbiased": ReferenceTarget(std_unbiased=True, seed=seed),
        "dead_residual": ReferenceTarget(freeze_w=True, seed=seed),
    }


class ChunkedTarget(ReferenceTarget):
    """Mutant: each inference chunk builds its own market summary from its own rows.

    This is the failure mode the batch-size check exists to catch: a correct design has to
    compute ``m_t`` on the *whole* timestamp pool once and then reuse it for every chunk.
    """

    name = "mutant:per_chunk_m"

    def delta(self, panel, m, *, arm="C", h=None, base=None):
        # The bug: the market summary is rebuilt from whatever rows this call happens to
        # see, so a stock's score depends on which inference batch it landed in.
        own_m, _ = market_summary(panel.h if h is None else h, panel.group, panel.eligible,
                                  accumulate="float64")
        return super().delta(panel, own_m, arm=arm, h=h, base=base)


def build_reference(**kw) -> ReferenceTarget:
    return ReferenceTarget(**kw)
