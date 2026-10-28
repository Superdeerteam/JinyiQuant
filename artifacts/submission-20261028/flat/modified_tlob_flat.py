"""Generated flat platform inference module: modified_tlob.
MIT License

Copyright (c) 2026 Leonardo Berti

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

# Source: modified_tlob/schema.py
"""Model-layer re-export of the shared input contract, which the data layer owns."""
from jinyiquant_flat import TRADE_COLUMNS, lob_columns, lob_trade_columns, validate_input_schema
__all__ = ['TRADE_COLUMNS', 'lob_columns', 'lob_trade_columns', 'validate_input_schema']

# Source: modified_tlob/device.py
"""Device preflight only; runtime OOM/errors propagate to the run recorder."""
import os
import torch

def _configure_cuda_workspace():
    """PyTorch 2.5 requires one of these values before the first cuBLAS call."""
    name = 'CUBLAS_WORKSPACE_CONFIG'
    value = os.environ.get(name)
    if value is None:
        if torch.cuda.is_initialized():
            raise RuntimeError(f'Set {name} before CUDA initialization; restart in a fresh process')
        value = ':4096:8'
        os.environ[name] = value
    if value not in {':4096:8', ':16:8'}:
        raise ValueError(f"{name} must be ':4096:8' or ':16:8'; got {value!r}")
    return value

def _cuda_preflight():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable (CPU build, missing GPU or driver)')
    x = torch.ones((2, 2), device='cuda', requires_grad=True)
    (x @ x).sum().backward()
    torch.cuda.synchronize()

def select_device(requested='auto', recorder=None):
    if requested not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('device must be auto/cpu/cuda')
    selected, reason = ('cpu', 'explicit_cpu')
    if requested != 'cpu':
        _configure_cuda_workspace()
        try:
            _cuda_preflight()
        except (RuntimeError, AssertionError, OSError) as exc:
            if requested == 'cuda':
                raise RuntimeError(f'Explicit CUDA preflight failed: {exc}') from exc
            reason = f'cuda_preflight_failed: {type(exc).__name__}: {exc}'
        else:
            selected, reason = ('cuda', 'cuda_compute_preflight_passed')
    if recorder:
        recorder.event('device', 'selected', requested=requested, selected=selected, reason=reason, torch_version=torch.__version__, cuda_build=torch.version.cuda, cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'), resources=device_resources(torch.device(selected)))
    return torch.device(selected)

def device_resources(device):
    import psutil
    p = psutil.Process()
    return {'rss_mb': p.memory_info().rss / 2 ** 20, 'cpu_seconds': sum(p.cpu_times()[:2]), 'gpu_memory_mb': torch.cuda.memory_allocated(device) / 2 ** 20 if device.type == 'cuda' else None, 'gpu_peak_memory_mb': torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == 'cuda' else None, 'device': str(device)}

# Source: modified_tlob/normalization.py
"""Upstream BiN with stable zero-variance gradients and persistent parameters (MIT)."""
import torch
from torch import nn

class BiN(nn.Module):

    def __init__(self, d1, t1):
        super().__init__()
        if d1 < 2 or t1 < 2:
            raise ValueError('BiN requires at least two features and observations')
        self.d1, self.t1 = (d1, t1)
        self.B1 = nn.Parameter(torch.zeros(t1, 1))
        self.l1 = nn.Parameter(torch.empty(t1, 1))
        nn.init.xavier_normal_(self.l1)
        self.B2 = nn.Parameter(torch.zeros(d1, 1))
        self.l2 = nn.Parameter(torch.empty(d1, 1))
        nn.init.xavier_normal_(self.l2)
        self.y1 = nn.Parameter(torch.full((1,), 0.5))
        self.y2 = nn.Parameter(torch.full((1,), 0.5))

    @staticmethod
    def _standardize(x, dim):
        std = x.std(dim=dim, keepdim=True, correction=1)
        denominator = torch.where(std < 0.0001, torch.ones_like(std), std)
        return (x - x.mean(dim=dim, keepdim=True)) / denominator

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.d1, self.t1):
            raise ValueError(f'BiN expects [batch, {self.d1}, {self.t1}]')
        with torch.no_grad():
            for weight in (self.y1, self.y2):
                weight.copy_(torch.where(weight < 0, torch.full_like(weight, 0.01), weight))
        temporal = self.l2[None] * self._standardize(x, 2) + self.B2[None]
        feature = self.l1.T[None] * self._standardize(x, 1) + self.B1.T[None]
        return self.y1 * feature + self.y2 * temporal

# Source: modified_tlob/model.py
"""Fixed upstream TLOB/MLPLOB trunks adapted to scalar regression. See provenance.json (MIT).

``DualPatchTLOB`` is a project extension: it keeps the raw LOB input, BiN and scalar
head contract but replaces the trunk with two channel-independent PatchTST-style
encoders over non-overlapping coarse and fine scales (Nie et al., PatchTST, ICLR 2023).
The patch length defines each scale, so extending the lookback adds patches instead of
stretching a patch: the 128-bar recipe uses 4x32 and 16x8, and the 256-bar arm uses
8x32 and 32x8.
"""
import torch
from torch import nn
FINITE_CHECK_EVERY_DEFAULT = 50

class InputFiniteCheck:
    """Amortized "input must be finite" guard for the trunk entry points.

    ``torch.isfinite(x).all()`` is a device reduction whose scalar result is read on
    the host, so it drains the stream on every call and prevents CUDA-graph capture.
    The guard therefore stays unconditional wherever that drain is free (CPU) and, on
    CUDA, runs on the first call plus every ``every_n``-th call afterwards.
    ``every_n=1`` restores the original unconditional per-call check.
    """

    def __init__(self, every_n=FINITE_CHECK_EVERY_DEFAULT):
        self.every_n = FINITE_CHECK_EVERY_DEFAULT
        self.calls = 0
        self.set_every_n(every_n)

    def set_every_n(self, every_n):
        if type(every_n) is not int or every_n < 1:
            raise ValueError('finite_check_every must be a positive integer')
        self.every_n = every_n
        return self

    def reset(self):
        self.calls = 0
        return self

    def __call__(self, x):
        self.calls += 1
        if self.every_n > 1 and x.device.type == 'cuda' and (self.calls - 1) % self.every_n:
            return
        if not torch.isfinite(x).all():
            raise ValueError('Input must be finite after shared missing-value preprocessing')

def configure_finite_check(network, every_n):
    """Point a built backbone's input guard at the configured sampling interval."""
    check = getattr(network, 'finite_check', None)
    if not isinstance(check, InputFiniteCheck):
        raise ValueError('Backbone is missing its input finiteness guard')
    return check.reset().set_every_n(every_n)

class ComputeQKV(nn.Module):

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim * num_heads)

    def forward(self, x):
        return (self.q(x), self.k(x), self.v(x))

class MLP(nn.Module):

    def __init__(self, start_dim, hidden_dim, final_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(final_dim)
        self.fc = nn.Linear(start_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, final_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.fc2(self.gelu(self.fc(x)))
        if x.shape[-1] == residual.shape[-1]:
            x = x + residual
        return self.gelu(self.layer_norm(x))

class TransformerLayer(nn.Module):

    def __init__(self, hidden_dim, num_heads, final_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = ComputeQKV(hidden_dim, num_heads)
        self.attention = nn.MultiheadAttention(hidden_dim * num_heads, num_heads, batch_first=True)
        self.mlp = MLP(hidden_dim, hidden_dim * 4, final_dim)
        self.w0 = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, x):
        residual = x
        q, k, v = self.qkv(x)
        x, _ = self.attention(q, k, v, average_attn_weights=False, need_weights=True)
        x = self.mlp(self.norm(self.w0(x) + residual))
        return x + residual if x.shape[-1] == residual.shape[-1] else x

def sinusoidal_positional_embedding(seq_size, hidden_dim):
    if hidden_dim % 2:
        raise ValueError('Sinusoidal embedding requires even hidden_dim')
    positions = torch.arange(seq_size).unsqueeze(1)
    denominators = torch.pow(10000.0, 2 * torch.arange(hidden_dim // 2) / hidden_dim)
    result = torch.zeros(seq_size, hidden_dim)
    result[:, 0::2] = torch.sin(positions / denominators)
    result[:, 1::2] = torch.cos(positions / denominators)
    return result

class ModifiedTLOB(nn.Module):

    def __init__(self, *, hidden_dim=40, num_layers=4, seq_size=128, num_features=12, num_heads=1, is_sin_emb=True):
        super().__init__()
        dimensions = (hidden_dim, num_layers, seq_size, num_features, num_heads)
        if any((type(d) is not int or d <= 0 for d in dimensions)):
            raise ValueError('Model dimensions must be positive integers')
        if hidden_dim < 4 or seq_size < 4 or num_features < 2:
            raise ValueError('Quarter compression requires hidden_dim/seq_size >= 4 and features >= 2')
        self.seq_size, self.num_features = (seq_size, num_features)
        self.norm_layer = BiN(num_features, seq_size)
        self.emb_layer = nn.Linear(num_features, hidden_dim)
        if is_sin_emb:
            self.register_buffer('pos_encoder', sinusoidal_positional_embedding(seq_size, hidden_dim))
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, seq_size, hidden_dim))
        self.finite_check = InputFiniteCheck()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            last = i == num_layers - 1
            self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim // 4 if last else hidden_dim))
            self.layers.append(TransformerLayer(seq_size, num_heads, seq_size // 4 if last else seq_size))
        total_dim = hidden_dim // 4 * (seq_size // 4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.extend([nn.Linear(total_dim, total_dim // 4), nn.GELU()])
            total_dim //= 4
        self.final_layers.append(nn.Linear(total_dim, 1))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.seq_size, self.num_features):
            raise ValueError(f'Expected [batch, {self.seq_size}, {self.num_features}]')
        self.finite_check(x)
        return self._forward_impl(x)

    def _forward_impl(self, x):
        """Validated, branch-free trunk math: the only region that may be compiled."""
        x = self.norm_layer(x.transpose(1, 2)).transpose(1, 2)
        x = self.emb_layer(x) + self.pos_encoder
        for layer in self.layers:
            x = layer(x).transpose(1, 2)
        x = x.transpose(1, 2).reshape(x.shape[0], -1)
        for layer in self.final_layers:
            x = layer(x)
        return x[:, 0]

class MLPLOB(nn.Module):
    """Upstream dual-axis MLP trunk with a scalar return head.

    Follows upstream ``models/mlplob.py`` for non-LOBSTER inputs: per-sample BiN,
    a feature projection, then alternating feature-axis and time-axis MLP blocks
    with the same permutation after every component, and the same quarter
    compression before the head. Only the three-class head is replaced by one
    regression output, so no attention or positional embedding is used.
    """

    def __init__(self, *, hidden_dim=40, num_layers=3, seq_size=128, num_features=12):
        super().__init__()
        dimensions = (hidden_dim, num_layers, seq_size, num_features)
        if any((type(d) is not int or d <= 0 for d in dimensions)):
            raise ValueError('Model dimensions must be positive integers')
        if hidden_dim < 4 or seq_size < 4 or num_features < 2:
            raise ValueError('Quarter compression requires hidden_dim/seq_size >= 4 and features >= 2')
        self.seq_size, self.num_features = (seq_size, num_features)
        self.norm_layer = BiN(num_features, seq_size)
        self.finite_check = InputFiniteCheck()
        self.layers = nn.ModuleList([nn.Linear(num_features, hidden_dim), nn.GELU()])
        for i in range(num_layers):
            last = i == num_layers - 1
            wide = 2 if last else 4
            self.layers.append(MLP(hidden_dim, hidden_dim * wide, hidden_dim // 4 if last else hidden_dim))
            self.layers.append(MLP(seq_size, seq_size * wide, seq_size // 4 if last else seq_size))
        total_dim = hidden_dim // 4 * (seq_size // 4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.extend([nn.Linear(total_dim, total_dim // 4), nn.GELU()])
            total_dim //= 4
        self.final_layers.append(nn.Linear(total_dim, 1))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.seq_size, self.num_features):
            raise ValueError(f'Expected [batch, {self.seq_size}, {self.num_features}]')
        self.finite_check(x)
        return self._forward_impl(x)

    def _forward_impl(self, x):
        """Validated, branch-free trunk math: the only region that may be compiled."""
        x = self.norm_layer(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            x = layer(x).transpose(1, 2)
        x = x.reshape(x.shape[0], -1)
        for layer in self.final_layers:
            x = layer(x)
        return x[:, 0]
DUAL_PATCH_CHANNEL_DIM = 8

class PatchEncoder(nn.Module):
    """Channel-independent PatchTST-style encoder for one temporal scale.

    The lookback is cut into non-overlapping patches; every feature channel is
    tokenized separately with the shared patch embedding, positional encoding and
    transformer, matching the channel-independent design of PatchTST.
    """

    def __init__(self, *, seq_size, patch_count, hidden_dim, num_layers, num_heads):
        super().__init__()
        if type(patch_count) is not int or patch_count < 2 or patch_count > seq_size or seq_size % patch_count:
            raise ValueError('patch_count must be an integer >= 2 that divides the lookback exactly')
        if type(hidden_dim) is not int or type(num_heads) is not int or hidden_dim % num_heads:
            raise ValueError('Patch attention requires hidden_dim divisible by num_heads')
        self.patch_count = patch_count
        self.patch_length = seq_size // patch_count
        self.embed = nn.Linear(self.patch_length, hidden_dim)
        self.position = nn.Parameter(torch.empty(1, patch_count, hidden_dim))
        nn.init.uniform_(self.position, -0.02, 0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError('PatchEncoder expects [batch, features, lookback]')
        batch, features, seq_size = x.shape
        if seq_size != self.patch_count * self.patch_length:
            raise ValueError('PatchEncoder lookback differs from its patch contract')
        patches = x.reshape(batch * features, self.patch_count, self.patch_length)
        tokens = self.embed(patches) + self.position
        return self.norm(self.encoder(tokens))

class DualPatchTLOB(nn.Module):
    """Dual-scale patch trunk over the raw order-book window.

    Two channel-independent patch encoders summarize non-overlapping 32-bar (coarse,
    slower structure) and 8-bar (fine, fast changes) scales, one patch count per
    lookback: 4x32/16x8 at 128 bars, 8x32/32x8 at 256 bars. Each branch projects every
    channel's patch tokens to a short embedding, the branch embeddings are
    concatenated and a scalar regression head fuses them. BiN and the scalar output
    contract are shared with the other trunks.
    """

    def __init__(self, *, hidden_dim=40, num_layers=3, seq_size=128, num_features=12, num_heads=4, patch_counts=(4, 16)):
        super().__init__()
        dimensions = (hidden_dim, num_layers, seq_size, num_features, num_heads)
        if any((type(d) is not int or d <= 0 for d in dimensions)):
            raise ValueError('Model dimensions must be positive integers')
        if hidden_dim < 4 or seq_size < 4 or num_features < 2:
            raise ValueError('Dual patch requires hidden_dim/seq_size >= 4 and features >= 2')
        counts = tuple(patch_counts)
        if len(counts) != 2 or any((type(c) is not int for c in counts)) or len(set(counts)) != 2:
            raise ValueError('patch_counts requires two distinct integer scales')
        if any((count < 2 or seq_size % count for count in counts)):
            raise ValueError('patch_counts must be >= 2 and divide the lookback exactly')
        if counts[0] > counts[1]:
            raise ValueError('patch_counts must be ordered coarse-to-fine (fewer patches first)')
        self.seq_size, self.num_features = (seq_size, num_features)
        self.patch_counts = counts
        self.channel_dim = DUAL_PATCH_CHANNEL_DIM
        self.norm_layer = BiN(num_features, seq_size)
        self.finite_check = InputFiniteCheck()
        self.branches = nn.ModuleList([PatchEncoder(seq_size=seq_size, patch_count=count, hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads) for count in counts])
        self.channel_heads = nn.ModuleList([nn.Linear(count * hidden_dim, self.channel_dim) for count in counts])
        fused = len(counts) * num_features * self.channel_dim
        self.head = nn.Sequential(nn.Linear(fused, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.seq_size, self.num_features):
            raise ValueError(f'Expected [batch, {self.seq_size}, {self.num_features}]')
        self.finite_check(x)
        return self._forward_impl(x)

    def _forward_impl(self, x):
        """Validated, branch-free trunk math: the only region that may be compiled."""
        normalized = self.norm_layer(x.transpose(1, 2))
        batch = normalized.shape[0]
        embeddings = []
        for encoder, channel_head in zip(self.branches, self.channel_heads):
            tokens = encoder(normalized)
            channel = tokens.reshape(batch, self.num_features, -1)
            embeddings.append(channel_head(channel).reshape(batch, -1))
        return self.head(torch.cat(embeddings, dim=1))[:, 0]

def parameter_groups(network):
    """Top-level module names owning parameters, in registration order.

    Training logs and diagnostics report gradient norms per group; upstream MLPLOB
    has no ``emb_layer``, so the groups are read from the built network instead of
    a hard-coded TLOB list.
    """
    return tuple(dict.fromkeys((name.split('.')[0] for name, _ in network.named_parameters())))

# Source: modified_tlob/adapter.py
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
from jinyiquant_flat import write_json
BACKBONES = {'modified_tlob': ModifiedTLOB, 'mlplob': MLPLOB, 'dual_patch': DualPatchTLOB}
TRAINING_ONLY_MODULE = 'jinyiquant.trajectory'
COMPILE_MODES = {'eager': None, 'default': 'default', 'reduce-overhead': 'reduce-overhead'}
COMPILE_EQUIVALENCE_ATOL = 1e-06

def _trajectory_helpers():
    """Training-only helpers from ``jinyiquant.trajectory``, imported lazily.

    Inference and the flat submission package never need the trajectory module: the flat
    builder whitelists only inference dependencies, so a module-level import here would make
    the package unbuildable. Training paths (fit/save/resume/exposure) still resolve it.
    """
    return importlib.import_module(TRAINING_ONLY_MODULE)
PREPROCESS_KEYS = ('feature_columns', 'lookback', 'min_history', 'missing_policy', 'max_ffill_steps', 'missing_sentinel', 'price_scale', 'history_days', 'sample_times', 'label_horizon_minutes')
DATE_DECAY_HALF_LIFE_TRADING_DAYS = 10.0
TARGET_CENTERING_MODES = ('none', 'timestamp_mean_train', 'timestamp_zscore_train_rms_matched')
TARGET_LABELS = {'none': 'official_vwap_return', 'timestamp_mean_train': 'official_vwap_return_minus_training_timestamp_mean', 'timestamp_zscore_train_rms_matched': 'official_vwap_return_timestamp_zscore_train_rms_matched'}
PREDICTION_UNITS = {'none': 'original_label_return', 'timestamp_mean_train': 'relative_return_score', 'timestamp_zscore_train_rms_matched': 'relative_return_score'}
TRAINING_LOSS_UNITS = {'none': 'original_return_squared', 'timestamp_mean_train': 'centered_return_squared', 'timestamp_zscore_train_rms_matched': 'rms_matched_transformed_return_squared'}
PROGRESS_LOSS_UNITS = {'none': 'raw_label_mse', 'timestamp_mean_train': 'centered_label_mse', 'timestamp_zscore_train_rms_matched': 'rms_matched_transformed_label_mse'}
BLEND_KIND = 'modified_tlob_blend'
BLEND_SCHEMA_VERSION = 4
SHARDED_BLEND_KIND = 'modified_tlob_sharded_blend'
UPLOAD_WEIGHT_LIMIT = 50000000
BLEND_SHARED_KEYS = ('target_scaling', 'target_centering', 'time_loss_weighting', 'book_representation', 'trade_features')
BACKBONE_SHARED_KEYS = {'modified_tlob': ('num_heads', 'is_sin_emb'), 'mlplob': (), 'dual_patch': ('num_heads', 'patch_counts')}
TRAINING_STATE_SCHEMA_VERSION = 1

def _canonical_sha256(value):
    """Stable digest of a JSON-serialisable structure, or None for an absent one."""
    if value is None:
        return None
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def _batch_indices_sha256(batches):
    """Digest of one epoch's batch composition, used to verify a resumed sampler position."""
    digest = hashlib.sha256()
    for indices in batches:
        array = np.ascontiguousarray(np.asarray(indices, dtype='<i8'))
        digest.update(len(array).to_bytes(8, 'little'))
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
        raise ValueError('Target-transform keys and targets are not aligned')
    if not np.isfinite(raw64).all():
        raise ValueError('Target transform requires finite official labels')
    timestamps = pd.to_datetime(keys['date'])
    if timestamps.isna().any():
        raise ValueError('Target transform requires a timestamp for every training row')
    frame = pd.DataFrame({'timestamp': timestamps.to_numpy(), 'raw': raw64})
    grouped = frame.groupby('timestamp', sort=False)['raw']
    counts = grouped.size()
    means = grouped.transform('mean').to_numpy(dtype=np.float64)
    centered = raw64 - means
    timestamp_means = grouped.mean()
    squares = pd.DataFrame({'timestamp': frame['timestamp'], 'square': centered ** 2}).groupby('timestamp', sort=False)['square'].sum().reindex(counts.index)
    stds = np.sqrt(squares.to_numpy(dtype=np.float64) / counts.to_numpy(dtype=np.float64))
    std_rows = pd.Series(stds, index=counts.index).reindex(frame['timestamp']).to_numpy(dtype=np.float64)
    q = np.divide(centered, std_rows, out=np.zeros_like(centered), where=std_rows > 0)
    centered_std = float(centered.std())
    q_std = float(q.std())
    if not np.isfinite([centered_std, q_std]).all() or q_std <= 0:
        raise ValueError('Timestamp-zscore training labels have zero or nonfinite pool q standard deviation')
    c = centered_std / q_std
    transformed64 = c * q
    transformed = transformed64.astype(np.float32)
    if not np.isfinite(transformed).all():
        raise FloatingPointError('Nonfinite rms-matched training targets')
    centered_means = pd.DataFrame({'timestamp': frame['timestamp'], 'centered': centered}).groupby('timestamp', sort=False)['centered'].mean()
    zero_variance = [str(timestamp) for timestamp, std in zip(counts.index, stds) if std == 0]
    rows = [{'timestamp': str(timestamp), 'samples': int(counts.loc[timestamp]), 'mean': float(timestamp_means.loc[timestamp]), 'std': float(std)} for timestamp, std in zip(counts.index, stds)]
    selected = np.ones(len(raw64), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    stats = {'method': 'timestamp_zscore_train_rms_matched', 'formula': 'q = (r - mean_t(r)) / std_t(r); y* = c q with c = std(z) / std(q), all ddof=0', 'label_source': 'official training labels only', 'scope': 'every row of each full training timestamp in the selected training pool; computed once before mini-batches and before the min-history mask', 'ddof': 0, 'timestamps': int(len(counts)), 'rows': int(len(raw64)), 'min_rows_per_timestamp': int(counts.min()), 'mask_excluded_rows': int((~selected).sum()), 'raw_mean': float(raw64.mean()), 'raw_std': float(raw64.std()), 'centered_mean': float(centered.mean()), 'centered_std': centered_std, 'q_mean': float(q.mean()), 'q_std': q_std, 'c': float(c), 'c_formula': 'std(centered training labels, ddof=0) / std(q, ddof=0) over the complete training pool', 'transformed_mean': float(transformed64.mean()), 'transformed_std': float(c * q_std), 'float32_transformed_mean': float(transformed.astype(np.float64).mean()), 'float32_transformed_std': float(transformed.astype(np.float64).std()), 'selected_centered_mean': float(centered[selected].mean()), 'selected_centered_std': float(centered[selected].std()), 'selected_transformed_std': float(transformed[selected].astype(np.float64).std()), 'fallback_mean': float(centered[selected].astype(np.float32).astype(np.float64).mean()), 'fallback_basis': 'timestamp-centered training-label mean in the same float32 representation used by the centered control; never re-estimated from the transformed targets', 'zero_variance_timestamps': int(len(zero_variance)), 'zero_variance_timestamp_list': zero_variance, 'max_abs_centered_timestamp_mean': float(np.abs(centered_means.to_numpy()).max()), 'timestamp_means_sha256': hashlib.sha256(json.dumps([{'timestamp': str(timestamp), 'mean': float(value)} for timestamp, value in timestamp_means.items()], sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest(), 'per_timestamp_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest(), 'per_timestamp': rows}
    if stats['max_abs_centered_timestamp_mean'] > 1e-12:
        raise ValueError('Timestamp-centered training labels are not zero-mean within a timestamp')
    if abs(stats['transformed_std'] - centered_std) > 1e-12 * max(centered_std, 1.0):
        raise ValueError('RMS-matched training labels do not match the centered pool standard deviation')
    return (transformed[selected], stats)

def timestamp_groups(keys, mask):
    """Usable sample positions grouped by their full timestamp (date and slot time).

    Grouped sampling replaces the global random shuffle with one cross-section per
    optimizer step, so every pairwise ranking comparison is same-timestamp only.
    """
    dates = pd.to_datetime(keys['date'])
    if dates.isna().any():
        raise ValueError('Grouped sampling requires a timestamp for every training row')
    selected = np.flatnonzero(np.asarray(mask, dtype=bool))
    frame = pd.DataFrame({'timestamp': dates.to_numpy()[selected], 'index': selected})
    groups = [values.to_numpy() for _, values in frame.groupby('timestamp', sort=True)['index']]
    if len(groups) < 2:
        raise ValueError('Grouped sampling requires at least two training timestamps')
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
        raise ValueError('Grouped sampling produced no complete same-timestamp batch')
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
    if 'patch_counts' in architecture:
        architecture['patch_counts'] = tuple(architecture['patch_counts'])
    return architecture

class TLOBAdapter:
    """Shared regression training/inference loop for the torch LOB backbones.

    ``model.kind`` selects the upstream trunk (TLOB attention, MLPLOB dual-axis MLP
    or the project's dual-scale patch encoder); preprocessing, labels, loss,
    checkpoints and evaluation stay identical.
    """

    def __init__(self, model_config, data_config, *, seed=42, recorder=None):
        self.config = model_config.model_dump() if hasattr(model_config, 'model_dump') else dict(model_config)
        self.config.setdefault('kind', 'modified_tlob')
        if self.config['kind'] not in BACKBONES:
            raise ValueError(f"Unsupported torch model kind: {self.config['kind']!r}")
        if self.config['kind'] == 'dual_patch':
            self.config['patch_counts'] = tuple(self.config.get('patch_counts', (4, 16)))
        self.config.setdefault('target_scaling', 'none')
        self.config.setdefault('target_centering', 'none')
        self.config.setdefault('max_steps', None)
        self.config.setdefault('time_loss_weighting', 'none')
        self.config.setdefault('sampling', 'shuffle')
        self.config.setdefault('lr_schedule', 'constant')
        self.config.setdefault('ranking_loss', 'none')
        self.config.setdefault('ranking_loss_weight', 0.0)
        self.config.setdefault('book_representation', 'raw')
        self.config.setdefault('trade_features', 'zero')
        self.config.setdefault('finite_check_every', 50)
        self.config.setdefault('compile_mode', 'eager')
        if self.config['target_scaling'] not in ('none', 'train_std'):
            raise ValueError('Unsupported target scaling')
        if self.config['target_centering'] not in TARGET_CENTERING_MODES:
            raise ValueError('Unsupported target centering')
        if self.config['time_loss_weighting'] not in ('none', 'train_variance', 'date_decay_10d'):
            raise ValueError('Unsupported loss weighting')
        if self.config['sampling'] not in ('shuffle', 'timestamp_grouped'):
            raise ValueError('Unsupported sampling mode')
        if self.config['lr_schedule'] != 'constant':
            if self.config['lr_schedule'] not in _trajectory_helpers().LR_SCHEDULES:
                raise ValueError('Unsupported learning-rate schedule')
        if self.config['ranking_loss'] not in ('none', 'pairwise_hinge'):
            raise ValueError('Unsupported ranking loss')
        self.ranking_loss_weight = float(self.config['ranking_loss_weight'])
        if not np.isfinite(self.ranking_loss_weight) or self.ranking_loss_weight < 0:
            raise ValueError('Unsupported ranking loss weight')
        if (self.config['ranking_loss'] == 'none') != (self.ranking_loss_weight == 0):
            raise ValueError('Ranking loss and its weight must be enabled together')
        data = data_config.model_dump(mode='json') if hasattr(data_config, 'model_dump') else dict(data_config)
        self.preprocessing = {k: data[k] for k in PREPROCESS_KEYS}
        self.training_sample_times = tuple(data.get('training_sample_times') or data['sample_times'])
        fields = tuple(self.preprocessing['feature_columns'])
        self.lob_feature_count = validate_input_schema(fields, self.config['book_representation'], self.config['trade_features'])
        self.new_input_contract = self.config['book_representation'] != 'raw' or len(fields) == 15
        self.seed, self.recorder = (seed, recorder)
        self.device = select_device(self.config['device'], recorder)
        torch.set_num_threads(self.config['cpu_threads'])
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        self.architecture = {k: self.config[k] for k in ('hidden_dim', 'num_layers')}
        if self.config['kind'] == 'modified_tlob':
            self.architecture.update({k: self.config[k] for k in ('num_heads', 'is_sin_emb')})
        elif self.config['kind'] == 'dual_patch':
            self.architecture.update({k: self.config[k] for k in ('num_heads', 'patch_counts')})
        self.architecture.update(seq_size=data['lookback'], num_features=len(fields))
        self.network = BACKBONES[self.config['kind']](**self.architecture).to(self.device)
        self.finite_check_every = self.config['finite_check_every']
        configure_finite_check(self.network, self.finite_check_every)
        self.parameter_groups = parameter_groups(self.network)
        self.intercept = None
        self.target_scale = 1.0
        self.target_centering_stats = None
        self.time_loss_weighting_stats = None
        self.trained = False
        if recorder:
            recorder.event('training', 'determinism', seed=seed, algorithms=True, cpu_threads=torch.get_num_threads(), device=str(self.device), cuda_matmul_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'), note='GPU tolerance-based reproducibility; cross-device rounding may differ')

    def _input(self, batch, *, sample_mask=None):
        x = np.asarray(batch.x, dtype=np.float32)
        expected = (len(batch.keys), self.architecture['seq_size'], self.architecture['num_features'])
        if x.shape != expected or batch.observed.shape != x.shape or (not np.isfinite(x).all()):
            raise ValueError('Batch requires aligned finite preprocessed LOB windows')
        if self.architecture['num_features'] == 15 and (x[..., 12:] < 0).any():
            raise ValueError('Trade flow inputs must be nonnegative after preprocessing')
        select_first = sample_mask is not None and self.architecture['num_features'] == 15 and (self.config['book_representation'] == 'raw')
        if select_first:
            x = x[sample_mask]
        if self.new_input_contract:
            if not select_first:
                x = x.copy()
            if self.config['book_representation'] == 'relative_mid_bps':
                prices = x[..., :self.lob_feature_count:2].astype(np.float64)
                best = prices[:, -1, :2]
                mid = best.mean(axis=1)[:, None, None]
                valid = (prices > 0) & (best > 0).all(axis=1)[:, None, None]
                relative = np.zeros_like(prices)
                np.divide(prices, mid, out=relative, where=valid)
                x[..., :self.lob_feature_count:2] = np.where(valid, (relative - 1) * 10000, 0)
            if self.architecture['num_features'] == 15:
                flows = x[..., 12:]
                if self.config['trade_features'] == 'log1p':
                    np.log1p(flows, out=flows)
                else:
                    flows.fill(0)
            if not np.isfinite(x).all():
                raise ValueError('Nonfinite transformed LOB inputs')
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
            raise ValueError('Invalid date-decay loss weights')
        weights = (raw / sample_mean).astype(np.float32)
        total = float(weights.sum(dtype=np.float64))
        by_date = []
        for position, date in enumerate(ordered):
            selected = (dates == date).to_numpy()
            weight = float(weights[selected][0])
            by_date.append({'date': str(date.date()), 'age_trading_days': int(len(ordered) - 1 - position), 'samples': int(selected.sum()), 'weight': weight, 'weight_share': weight * int(selected.sum()) / total})
        recent = [row for row in by_date if row['age_trading_days'] < 10]
        self.time_loss_weighting_stats = {'method': 'date_decay_10d', 'half_life_trading_days': DATE_DECAY_HALF_LIFE_TRADING_DAYS, 'normalization': 'mean_one_over_selected_training_samples', 'samples': int(len(raw)), 'days': int(len(ordered)), 'oldest_age_trading_days': int(len(ordered) - 1), 'mean_weight': float(weights.mean(dtype=np.float64)), 'weight_sum': total, 'recent_days': 10, 'recent_weight_share': float(sum((row['weight_share'] for row in recent))), 'by_date': by_date}
        self._event('training', 'time_loss_weighting_fitted', **self.time_loss_weighting_stats)
        return weights

    def _fit_time_loss_weights(self, keys, target):
        """Freeze loss weights using only the selected training rows."""
        self.time_loss_weighting_stats = None
        mode = self.config['time_loss_weighting']
        if mode == 'none':
            return None
        if mode == 'date_decay_10d':
            return self._fit_date_decay_weights(keys)
        times = keys.date.dt.strftime('%H:%M').to_numpy()
        if set(times) != set(self.training_sample_times):
            raise ValueError('Time loss weighting requires every configured training sample time and no other times')
        target = target.astype(np.float64)
        variance = float(target.var())
        epsilon = 0.05 * variance
        by_time = {}
        for sample_time in self.training_sample_times:
            selected = times == sample_time
            slot_variance = float(target[selected].var())
            by_time[sample_time] = {'samples': int(selected.sum()), 'days': int(keys.date[selected].dt.normalize().nunique()), 'variance': slot_variance, 'weight': 1.0 if variance == 0 else 1 / (slot_variance + epsilon)}
        mean_weight = sum((row['weight'] * row['samples'] for row in by_time.values())) / len(target)
        for row in by_time.values():
            row['weight'] /= mean_weight
        self.time_loss_weighting_stats = {'method': 'train_variance', 'samples': len(target), 'days': int(keys.date.dt.normalize().nunique()), 'variance': variance, 'epsilon_factor': 0.05, 'epsilon': epsilon, 'constant_target_fallback': variance == 0, 'by_time': by_time}
        self._event('training', 'time_loss_weighting_fitted', **self.time_loss_weighting_stats)
        return np.asarray([by_time[sample_time]['weight'] for sample_time in times], dtype=np.float32)

    def _csz_training_targets(self, keys, raw64, mask):
        """Fit the frozen cross-sectional z-score transform and expose its statistics."""
        selected, stats = timestamp_zscore_train_rms_matched(keys, raw64, mask)
        self.target_centering_stats = stats
        self._event('training', 'target_centering_fitted', **stats)
        return (selected, stats)

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
        if self.config['target_centering'] == 'timestamp_zscore_train_rms_matched':
            return self._csz_training_targets(keys, raw64, mask)
        self.target_centering_stats = None
        if self.config['target_centering'] == 'none':
            return (raw[mask], None)
        if len(keys) != len(raw64):
            raise ValueError('Centering keys and targets are not aligned')
        timestamps = pd.to_datetime(keys['date'])
        if timestamps.isna().any():
            raise ValueError('Centering requires a timestamp for every training row')
        frame = pd.DataFrame({'timestamp': timestamps.to_numpy(), 'target': raw64})
        grouped = frame.groupby('timestamp', sort=False)['target']
        means = grouped.transform('mean').to_numpy()
        centered = raw64 - means
        timestamp_means = grouped.mean()
        centered_means = pd.DataFrame({'timestamp': frame['timestamp'], 'target': centered}).groupby('timestamp', sort=False)['target'].mean()
        counts = grouped.size()
        if not np.isfinite(means).all() or not np.isfinite(centered).all():
            raise FloatingPointError('Nonfinite timestamp-centered training targets')
        values = timestamp_means.to_numpy(dtype=np.float64)
        stats = {'method': 'timestamp_mean_train', 'label_source': 'official training labels only', 'scope': 'every row of each full timestamp in the selected training pool; computed once before mini-batches and before the min-history mask', 'timestamps': int(len(timestamp_means)), 'rows': int(len(raw64)), 'min_rows_per_timestamp': int(counts.min()), 'mask_excluded_rows': int((~np.asarray(mask, dtype=bool)).sum()), 'raw_mean': float(raw64.mean()), 'raw_std': float(raw64.std()), 'centered_mean': float(centered.mean()), 'centered_std': float(centered.std()), 'selected_centered_mean': float(centered[mask].mean()), 'selected_centered_std': float(centered[mask].std()), 'mean_abs_timestamp_mean': float(np.abs(values).mean()), 'rms_timestamp_mean': float(np.sqrt(np.mean(values ** 2))), 'max_abs_timestamp_mean': float(np.abs(values).max()), 'max_abs_centered_timestamp_mean': float(np.abs(centered_means.to_numpy()).max()), 'timestamp_means_sha256': hashlib.sha256(json.dumps([{'timestamp': str(timestamp), 'mean': float(value)} for timestamp, value in timestamp_means.items()], sort_keys=True).encode('utf-8')).hexdigest()}
        if stats['max_abs_centered_timestamp_mean'] > 1e-12:
            raise ValueError('Timestamp-centered training labels are not zero-mean within a timestamp')
        self.target_centering_stats = stats
        self._event('training', 'target_centering_fitted', **stats)
        return (centered[mask].astype(np.float32), stats)

    def _learning_rate(self, step: int=0) -> float:
        return _trajectory_helpers().learning_rate_at(self.config['lr_schedule'], self.config['learning_rate'], step)

    def _training_fingerprint(self, keys, optimization_targets, used_positions):
        """Everything besides the parameters that a resumed update depends on."""
        canonical = {'preprocessing': self.preprocessing, 'architecture': {key: list(value) if isinstance(value, tuple) else value for key, value in self.architecture.items()}, 'model_config': {key: self.config[key] for key in ('batch_size', 'sampling', 'ranking_loss', 'ranking_loss_weight', 'target_scaling', 'target_centering', 'time_loss_weighting', 'learning_rate', 'lr_schedule')}, 'seed': self.seed, 'fitted': {'intercept': None if self.intercept is None else float(self.intercept), 'target_scale': float(self.target_scale), 'target_centering_stats_sha256': _canonical_sha256(self.target_centering_stats), 'time_loss_weighting_stats_sha256': _canonical_sha256(self.time_loss_weighting_stats)}}
        config_digest = _canonical_sha256(canonical)
        timestamps = pd.to_datetime(keys['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        data_digest = hashlib.sha256(np.ascontiguousarray(np.asarray(optimization_targets, dtype=np.float32)).tobytes() + np.asarray(used_positions, dtype=np.uint8).tobytes() + '\n'.join(timestamps.tolist()).encode('utf-8')).hexdigest()
        return {'config_sha256': config_digest, 'data_sha256': data_digest, 'combined_sha256': hashlib.sha256((config_digest + data_digest).encode()).hexdigest()}

    def _save_training_state(self, directory, *, optimizer, generator, fingerprint, position, generator_before_epoch, steps_per_epoch, exposure_counts):
        """Write a resumable training state: parameters, optimizer, RNG and sampler position."""
        trajectory = _trajectory_helpers()
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        numpy_state = np.random.get_state()
        payload = {'schema_version': TRAINING_STATE_SCHEMA_VERSION, 'kind': self.config['kind'], 'architecture': self.architecture, 'model_config': self.config, 'preprocessing': self.preprocessing, 'seed': self.seed, 'device': str(self.device), 'torch_version': str(torch.__version__), 'lr_schedule': self.config['lr_schedule'], 'learning_rate': self._learning_rate(int(position['steps'])), 'fingerprint': fingerprint, 'position': {**{key: value if key == 'epoch_batches_sha256' else int(value) for key, value in position.items()}, 'steps_per_epoch': {str(key): int(value) for key, value in steps_per_epoch.items()}}, 'generator_before_epoch': generator_before_epoch, 'rng': {'torch': torch.get_rng_state(), 'generator': generator.get_state(), 'numpy': (numpy_state[0], numpy_state[1].tobytes(), numpy_state[2], numpy_state[3], numpy_state[4]), 'python': random.getstate(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, 'network_state_dict': {key: value.detach().cpu() for key, value in self.network.state_dict().items()}, 'optimizer_state_dict': optimizer.state_dict(), 'exposure_counts': None if exposure_counts is None else torch.from_numpy(np.asarray(exposure_counts).copy())}
        torch.save(payload, target / trajectory.CHECKPOINT_FILENAME)
        optimizer_state = payload['optimizer_state_dict']
        exposure = payload['exposure_counts']
        metadata = {key: payload[key] for key in ('schema_version', 'kind', 'seed', 'device', 'torch_version', 'lr_schedule', 'learning_rate', 'preprocessing', 'position', 'fingerprint')}
        metadata.update({'optimizer': {'state_entries': len(optimizer_state['state']), 'step': max((int(state['step']) for state in optimizer_state['state'].values()), default=0)}, 'rng_streams': sorted(payload['rng']), 'sampler': {'epoch_index': int(position['epoch_index']), 'batch_index': int(position['batch_index']), 'epoch_batches_sha256': position.get('epoch_batches_sha256'), 'generator_state_saved': True}, 'exposure_counters': None if exposure is None else {'rows': int(exposure.numel()), 'samples_seen': int(exposure.sum()), 'sha256': hashlib.sha256(exposure.numpy().astype('<i8').tobytes()).hexdigest()}})
        write_json(target / trajectory.CHECKPOINT_METADATA_FILENAME, metadata)
        self._event('training', 'training_state_saved', directory=str(target), steps=int(position['steps']), epoch_index=int(position['epoch_index']), batch_index=int(position['batch_index']), exposure=exposure_counts is not None)
        return metadata

    def _restore_training_state(self, checkpoint, *, optimizer, generator, fingerprint, usable):
        """Load a training state and fail closed if its recipe, pool or position drifted."""
        if checkpoint.get('schema_version') != TRAINING_STATE_SCHEMA_VERSION:
            raise ValueError('Unsupported training-state checkpoint schema')
        if checkpoint['kind'] != self.config['kind'] or checkpoint['preprocessing'] != self.preprocessing or _architecture_with_patch_counts(checkpoint['architecture']) != self.architecture:
            raise ValueError('Training-state checkpoint architecture/preprocessing mismatch')
        if checkpoint['fingerprint']['combined_sha256'] != fingerprint['combined_sha256']:
            raise ValueError('Training data or recipe changed since the checkpoint; a continuous trajectory requires the identical pool')
        if checkpoint['lr_schedule'] != self.config['lr_schedule']:
            raise ValueError('Learning-rate schedule changed since the checkpoint')
        position = dict(checkpoint['position'])
        if int(position['usable_samples']) != usable:
            raise ValueError('Training-state checkpoint was written for a different usable sample count')
        if int(position['steps']) < 1 or int(position['batch_index']) < 0:
            raise ValueError('Malformed training-state position')
        self.network.load_state_dict(checkpoint['network_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        rng = checkpoint['rng']
        torch.set_rng_state(rng['torch'])
        numpy_state = rng['numpy']
        encoded = numpy_state[1]
        if isinstance(encoded, torch.Tensor):
            state_array = encoded.numpy().astype(np.uint32)
        else:
            state_array = np.frombuffer(bytes(encoded), dtype=np.uint32).copy()
        np.random.set_state((numpy_state[0], state_array, numpy_state[2], numpy_state[3], numpy_state[4]))
        random.setstate(rng['python'])
        if str(checkpoint['device']).startswith('cuda'):
            if not torch.cuda.is_available():
                raise ValueError('A CUDA training-state checkpoint cannot be resumed without CUDA')
            torch.cuda.set_rng_state_all(rng['cuda'])
        if int(position['batch_index']) > 0:
            generator.set_state(checkpoint['generator_before_epoch'])
        exposure = checkpoint.get('exposure_counts')
        return (position, None if exposure is None else exposure.numpy().astype(np.int64))

    def _training_forward(self, probe):
        """Forward callable for the optimization loop, with the compile fallback contract.

        ``report["status"]`` is ``eager`` (not requested), ``compiled`` or ``fallback``, and
        every fallback emits a warning event, so a silent degradation is impossible. A
        compiled trunk must reproduce the eager forward on one real batch within
        ``COMPILE_EQUIVALENCE_ATOL`` before it is allowed to drive the trajectory; only the
        validated branch-free ``_forward_impl`` math is compiled, never the input guards.
        """
        requested = self.config['compile_mode']
        report = {'requested': requested, 'status': 'eager', 'fallback': None, 'equivalence_atol': COMPILE_EQUIVALENCE_ATOL, 'finite_check_every': self.finite_check_every, 'probe_shape': list(probe.shape)}
        mode = COMPILE_MODES[requested]
        if mode is None:
            self._event('training', 'kernel_path', **report)
            return (self.network._forward_impl, report)
        try:
            compile_fn = getattr(torch, 'compile', None)
            if compile_fn is None:
                raise RuntimeError('torch.compile is unavailable in this torch build')
            compiled = compile_fn(self.network._forward_impl, mode=mode)
            with torch.enable_grad():
                expected, actual = (self.network._forward_impl(probe), compiled(probe))
            difference = float((expected - actual).abs().max())
            if not bool(torch.isfinite(actual).all()):
                raise RuntimeError('compiled forward returned non-finite values')
            if difference > COMPILE_EQUIVALENCE_ATOL:
                raise RuntimeError(f'compiled forward differs from eager by {difference:.3g} > {COMPILE_EQUIVALENCE_ATOL:g}')
        except Exception as exc:
            report.update(status='fallback', reason='compile_failed', fallback=f'{type(exc).__name__}: {exc}'[:600])
            self._event('training', 'compile_fallback', level='WARNING', **report)
            return (self.network._forward_impl, report)
        report.update(status='compiled', mode=mode, max_abs_forward_difference=difference)
        self._event('training', 'compile_enabled', **report)
        return (compiled, report)

    def fit(self, batch, target, *, sample_mask=None, validation=None, observer=None, observer_every_steps=None, resume=None, step_budget=None, checkpoint_at_steps=None, checkpoint_dir=None, track_exposure=False):
        """One continuous optimization trajectory, optionally resumed and checkpointed.

        ``resume`` is a loaded training-state payload; the pool, recipe, fitted transforms and
        sampler position must reproduce the saved run bit for bit or the fit fails closed.
        ``step_budget`` stops at an absolute step count, ``checkpoint_at_steps`` writes a
        resumable state at those absolute steps, and ``track_exposure`` accumulates per-row
        sample exposure for the budget accounting.
        """
        if observer_every_steps is not None and (type(observer_every_steps) is not int or observer_every_steps < 1):
            raise ValueError('observer_every_steps must be a positive integer or None')
        checkpoint_set = sorted(set(checkpoint_at_steps or ()))
        if any((type(value) is not int or value < 1 for value in checkpoint_set)):
            raise ValueError('checkpoint_at_steps must be positive absolute step numbers')
        if checkpoint_set and checkpoint_dir is None:
            raise ValueError('checkpoint_at_steps requires checkpoint_dir')
        if step_budget is not None and (type(step_budget) is not int or step_budget < 1):
            raise ValueError('step_budget must be a positive integer or None')
        if resume is not None and step_budget is not None and (int(resume['position']['steps']) >= step_budget):
            raise ValueError('step_budget must exceed the resumed step count')
        y = np.asarray(target, dtype=np.float32)
        if y.shape != (len(batch.keys),) or not len(y) or (not np.isfinite(y).all()):
            raise ValueError('Training requires nonempty aligned finite targets')
        mask = np.ones(len(y), dtype=bool) if sample_mask is None else np.asarray(sample_mask)
        if mask.dtype != bool or mask.shape != y.shape or (not mask.any()):
            raise ValueError('Training sample mask must select aligned samples')
        x = self._input(batch, sample_mask=mask)
        y, centering_stats = self._training_targets(batch.keys, y, mask)
        time_weights = self._fit_time_loss_weights(batch.keys.loc[mask], y)
        if validation is not None and self.config['target_centering'] != 'none':
            raise ValueError('Timestamp-mean training requires timestamp-centered validation targets')
        validation_inputs = None
        if validation is not None:
            vb, vy = validation
            vx, vy = (self._input(vb), np.asarray(vy, dtype=np.float32))
            if vy.shape != (len(vx),) or not len(vy) or (not np.isfinite(vy).all()):
                raise ValueError('Validation requires nonempty aligned finite targets')
            validation_inputs = (vx, vy)
        raw_selected = np.asarray(target, dtype=np.float32)[mask].astype(np.float64)
        training_std = float(raw_selected.std())
        self.intercept = float(centering_stats['fallback_mean']) if centering_stats is not None and 'fallback_mean' in centering_stats else float(y.astype(np.float64).mean())
        self.target_scale = training_std if self.config['target_scaling'] == 'train_std' and training_std > 0 else 1.0
        optimization_targets = y if self.target_scale == 1.0 else (y.astype(np.float64) / self.target_scale).astype(np.float32)
        if not np.isfinite(optimization_targets).all():
            raise FloatingPointError('Nonfinite scaled training targets')
        self._event('training', 'target_scaling_fitted', method=self.config['target_scaling'], target_scale=self.target_scale, training_std=training_std, samples=len(y), constant_target_fallback=self.config['target_scaling'] == 'train_std' and training_std == 0, centering=self.config['target_centering'], centered_training_std=None if centering_stats is None else centering_stats['selected_centered_std'], centering_stats=centering_stats, loss_units='centered_return_squared' if centering_stats is not None else 'original_return_squared')
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self._learning_rate(0))
        generator = torch.Generator().manual_seed(self.seed)
        grouped = self.config['sampling'] == 'timestamp_grouped'
        groups = timestamp_groups(batch.keys, mask) if grouped else None
        batch_size = self.config['batch_size']
        used_positions = np.ones(len(y), dtype=bool)
        if grouped:
            used_positions[:] = False
            for group in groups:
                used_positions[group[:len(group) // batch_size * batch_size]] = True
            self._event('training', 'grouped_sampling_fitted', groups=len(groups), group_key='full_timestamp', batch_size=batch_size, samples=len(y), samples_used_per_epoch=int(used_positions.sum()), dropped_tail_per_epoch=int(len(y) - int(used_positions.sum())), ranking_loss=self.config['ranking_loss'], ranking_loss_weight=self.ranking_loss_weight)
        usable = int(used_positions.sum())
        self.training_keys = batch.keys.loc[mask]
        self.used_positions = used_positions
        fingerprint = self._training_fingerprint(self.training_keys, optimization_targets, used_positions)
        exposure_counts = np.zeros(len(y), dtype=np.int64) if track_exposure else None
        steps, samples_seen, epochs_completed = (0, 0, 0)
        start_epoch, start_batch, steps_per_epoch = (0, 0, {})
        resume_epoch_batches_sha256 = None
        if resume is not None:
            position, restored_exposure = self._restore_training_state(resume, optimizer=optimizer, generator=generator, fingerprint=fingerprint, usable=usable)
            steps, samples_seen = (int(position['steps']), int(position['samples_seen']))
            epochs_completed = int(position['epochs_completed'])
            start_epoch, start_batch = (int(position['epoch_index']), int(position['batch_index']))
            steps_per_epoch = {int(key): int(value) for key, value in position['steps_per_epoch'].items()}
            resume_epoch_batches_sha256 = position.get('epoch_batches_sha256')
            if restored_exposure is not None:
                if exposure_counts is None:
                    raise ValueError('The checkpoint carries exposure counters; resume with track_exposure=True')
                exposure_counts = restored_exposure
            self._event('training', 'training_state_resumed', steps=steps, samples_seen=samples_seen, epoch_index=start_epoch, batch_index=start_batch, learning_rate=self._learning_rate(steps))
        started = time.monotonic()
        last_observed_step = None

        def observe(epoch):
            nonlocal last_observed_step
            if observer is None or last_observed_step == steps:
                return
            modes = [(module, module.training) for module in self.network.modules()]
            numpy_state, python_state = (np.random.get_state(), random.getstate())
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == 'cuda' else []
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
        stop_reason = 'epochs'
        actual_epochs, final_loss = (0.0, None)
        for epoch in range(start_epoch, self.config['epochs']):
            self.network.train()
            generator_before_epoch = generator.get_state()
            if grouped:
                epoch_batches = grouped_epoch_batches(groups, batch_size, generator)
            else:
                order = torch.randperm(len(y), generator=generator).numpy()
                epoch_batches = [order[offset:offset + batch_size] for offset in range(0, len(y), batch_size)]
            epoch_batches_sha256 = _batch_indices_sha256(epoch_batches)
            offset = start_batch if epoch == start_epoch else 0
            if offset:
                if offset >= len(epoch_batches):
                    raise ValueError('Checkpoint batch position exceeds the rebuilt epoch')
                if epoch_batches_sha256 != resume_epoch_batches_sha256:
                    raise ValueError('Checkpoint sampling position does not reproduce the saved batch order')
            epoch_samples = int(sum((len(indices) for indices in epoch_batches[:offset])))
            total = weighted_total = optimization_total = objective_total = ranking_total = 0.0
            for position in range(offset, len(epoch_batches)):
                indices = epoch_batches[position]
                xb = torch.from_numpy(x[indices]).to(self.device)
                yb = torch.from_numpy(optimization_targets[indices]).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                predictions = self.network(xb)
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
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                gradients = [p.grad for p in self.network.parameters() if p.grad is not None]
                if not gradients or not all((torch.isfinite(g).all() for g in gradients)):
                    raise FloatingPointError('Nonfinite gradients')
                optimizer.step()
                steps += 1
                steps_per_epoch[epoch] = position + 1
                epoch_samples += len(indices)
                samples_seen += len(indices)
                if exposure_counts is not None:
                    exposure_counts[indices] += 1
                optimization_loss = mse_term.item()
                objective_loss = loss.item()
                raw_loss = unweighted_loss.item() * self.target_scale ** 2
                weighted_loss = optimization_loss * self.target_scale ** 2
                total += raw_loss * len(indices)
                weighted_total += weighted_loss * len(indices)
                optimization_total += optimization_loss * len(indices)
                objective_total += objective_loss * len(indices)
                ranking_value = None if ranking_loss is None else float(ranking_loss.detach())
                if ranking_value is not None:
                    ranking_total += ranking_value * len(indices)
                if steps % self.config['log_every_steps'] == 0:
                    gradient_norms = {group: float(torch.stack([parameter.grad.detach().double().square().sum() for name, parameter in self.network.named_parameters() if name.split('.')[0] == group and parameter.grad is not None]).sum().sqrt()) for group in self.parameter_groups}
                    self._event('training', 'progress', epoch=epoch + 1, step=steps, loss=raw_loss, loss_units=PROGRESS_LOSS_UNITS[self.config['target_centering']], weighted_loss=weighted_loss, optimization_loss=optimization_loss, objective_loss=objective_loss, ranking_loss=ranking_value, ranking_loss_weight=self.ranking_loss_weight, sampling=self.config['sampling'], gradient_norms=gradient_norms, learning_rate=self._learning_rate(steps), elapsed_seconds=time.monotonic() - started, samples_seen=samples_seen, throughput_samples_per_second=samples_seen / max(time.monotonic() - started, 1e-09), resources=device_resources(self.device))
                if observer_every_steps is not None and steps % observer_every_steps == 0:
                    observe(epoch + epoch_samples / len(y))
                if steps in checkpoint_set:
                    at_epoch_end = position + 1 >= len(epoch_batches)
                    self._save_training_state(checkpoint_dir, optimizer=optimizer, generator=generator, fingerprint=fingerprint, position={'steps': steps, 'samples_seen': samples_seen, 'epochs_completed': epochs_completed, 'epoch_index': epoch + 1 if at_epoch_end else epoch, 'batch_index': 0 if at_epoch_end else position + 1, 'epoch_batches_sha256': None if at_epoch_end else epoch_batches_sha256, 'usable_samples': usable}, generator_before_epoch=generator.get_state() if at_epoch_end else generator_before_epoch, steps_per_epoch=steps_per_epoch, exposure_counts=exposure_counts)
                if step_budget is not None and steps >= step_budget:
                    stop_reason = 'step_budget'
                    break
                if self.config['max_steps'] is not None and steps >= self.config['max_steps']:
                    stop_reason = 'max_steps'
                    break
            if epoch_samples:
                complete_epoch = epoch_samples == usable
                epochs_completed += int(complete_epoch)
                actual_epochs = epoch + epoch_samples / len(y)
                final_loss = total / epoch_samples
                self._event('training', 'epoch_completed' if complete_epoch else 'epoch_stopped', epoch=epoch + 1, epochs=actual_epochs, step=steps, loss=final_loss, weighted_loss=weighted_total / epoch_samples, optimization_loss=optimization_total / epoch_samples, objective_loss=objective_total / epoch_samples, ranking_loss=ranking_total / epoch_samples if self.ranking_loss_weight else None, epoch_samples=epoch_samples, samples_seen=samples_seen, complete_epoch=complete_epoch, learning_rate=self._learning_rate(steps), resources=device_resources(self.device))
                observe(epoch + 1 if complete_epoch else actual_epochs)
            if validation_inputs is not None:
                vx, vy = validation_inputs
                self.network.eval()
                squared_error = 0.0
                with torch.inference_mode():
                    for offset_validation in range(0, len(vy), self.config['batch_size']):
                        xb = torch.from_numpy(vx[offset_validation:offset_validation + self.config['batch_size']]).to(self.device)
                        yb = torch.from_numpy(vy[offset_validation:offset_validation + self.config['batch_size']]).to(self.device)
                        squared_error += torch.nn.functional.mse_loss(self.network(xb) * self.target_scale, yb, reduction='sum').item()
                self._event('validation', 'epoch_completed', epoch=epoch + 1, loss=squared_error / len(vy), samples=len(vy), checkpoint_selection=False, resources=device_resources(self.device))
            if stop_reason != 'epochs':
                break
        self.trained = True
        self.exposure_counts = exposure_counts
        self.steps_per_epoch = dict(steps_per_epoch)
        optimizer_name = {'none': 'Adam_MSE', 'train_variance': 'Adam_time_weighted_MSE', 'date_decay_10d': 'Adam_date_weighted_MSE'}[self.config['time_loss_weighting']]
        if self.ranking_loss_weight:
            optimizer_name = f'{optimizer_name}+{self.ranking_loss_weight:g}x_pairwise_hinge'
        return {'loss': final_loss, 'learning_rate': self._learning_rate(steps), 'lr_schedule': self.config['lr_schedule'], 'weighted_loss': weighted_total / epoch_samples if epoch_samples else None, 'optimization_loss': optimization_total / epoch_samples if epoch_samples else None, 'objective_loss': objective_total / epoch_samples if epoch_samples else None, 'ranking_loss': ranking_total / epoch_samples if self.ranking_loss_weight and epoch_samples else None, 'ranking_loss_weight': self.ranking_loss_weight, 'ranking_loss_kind': self.config['ranking_loss'], 'sampling': self.config['sampling'], 'target_scale': self.target_scale, 'target_scaling': self.config['target_scaling'], 'target_centering': self.config['target_centering'], 'raw_target_std': training_std, 'centered_target_std': None if centering_stats is None else centering_stats['selected_centered_std'], 'transformed_target_std': None if centering_stats is None else centering_stats.get('selected_transformed_std'), 'mask_excluded': int((~mask).sum()), 'loss_units': TRAINING_LOSS_UNITS[self.config['target_centering']], 'time_loss_weighting': self.config['time_loss_weighting'], 'optimizer': optimizer_name, 'steps': steps, 'samples': len(y), 'samples_seen': samples_seen, 'epochs': actual_epochs, 'epochs_completed': epochs_completed, 'resumed': resume is not None, 'track_exposure': exposure_counts is not None, 'usable_samples': usable, 'stop_reason': stop_reason}

    def exposure_report(self, *, declared: dict | None=None) -> dict:
        """Per-day and per-epoch sample exposure for the trajectory that just ran."""
        if self.exposure_counts is None:
            raise ValueError('Exposure tracking was not enabled for this fit')
        return _trajectory_helpers().exposure_report(self.training_keys, self.used_positions, self.exposure_counts[self.used_positions], batch_size=self.config['batch_size'], steps=sum(self.steps_per_epoch.values()), per_epoch_steps=self.steps_per_epoch, declared=declared)

    def predict(self, batch):
        if not self.trained:
            raise ValueError('Model is not fitted')
        return self._predict(batch)

    def _predict(self, batch):
        """Fixed-weight scores for epoch diagnostics; public inference still requires a fitted model."""
        x = self._input(batch)
        scores = np.empty(len(x), dtype=np.float64)
        self.network.eval()
        with torch.inference_mode():
            for offset in range(0, len(x), self.config['batch_size']):
                xb = torch.from_numpy(x[offset:offset + self.config['batch_size']]).to(self.device)
                scores[offset:offset + len(xb)] = self.network(xb).cpu().numpy()
        scores *= self.target_scale
        missing = batch.observed[..., :self.lob_feature_count].any(axis=2).sum(axis=1) < self.preprocessing['min_history']
        scores[missing] = self.intercept
        if not np.isfinite(scores).all():
            raise FloatingPointError('Nonfinite predictions')
        return scores

    def validate_training_preprocessing(self, batch, sample_mask):
        self._input(batch)
        return {'passed': True, 'normalization': 'per_sample_BiN_no_population_fit', 'input_features': self.preprocessing['feature_columns'], 'mask_appended': False}

    def _loss_label(self):
        label = {'none': 'MSE', 'train_variance': 'time_weighted_MSE', 'date_decay_10d': 'date_weighted_MSE'}[self.config['time_loss_weighting']]
        if self.ranking_loss_weight:
            label = f"{label}+{self.ranking_loss_weight:g}x_{self.config['ranking_loss']}"
        return label

    def _payload(self):
        if not self.trained:
            raise ValueError('Cannot save an unfitted model')
        return {'schema_version': 3 if self.new_input_contract else 2 if self.config['target_scaling'] == 'train_std' else 1, 'kind': self.config['kind'], 'architecture': self.architecture, 'model_config': self.config, 'preprocessing': self.preprocessing, 'seed': self.seed, 'training_mean': self.intercept, 'label': TARGET_LABELS[self.config['target_centering']], 'prediction_units': PREDICTION_UNITS[self.config['target_centering']], 'loss': self._loss_label(), 'sampling': self.config['sampling'], 'target_scale': self.target_scale, 'target_centering': self.config['target_centering'], 'target_centering_stats': self.target_centering_stats, 'time_loss_weighting_stats': self.time_loss_weighting_stats, 'torch_version': str(torch.__version__), 'state_dict': {k: v.detach().cpu() for k, v in self.network.state_dict().items()}}

    def save(self, path):
        torch.save(self._payload(), Path(path))

    def export_text(self, path):
        payload = self._payload()
        payload['state_dict'] = {k: {'dtype': str(t.dtype).removeprefix('torch.'), 'shape': list(t.shape), 'data': t.reshape(-1).tolist()} for k, t in payload['state_dict'].items()}
        Path(path).write_text(json.dumps(payload, allow_nan=False, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, path, *, device='cpu', recorder=None):
        path = Path(path)
        if path.suffix == '.json':
            with path.open(encoding='utf-8') as source:
                payload = json.load(source)
            if payload.get('kind') == SHARDED_BLEND_KIND:
                return _load_sharded_blend(cls, path, payload, device=device, recorder=recorder)
        else:
            payload = torch.load(path, map_location='cpu', weights_only=True)
        return cls.from_payload(payload, device=device, recorder=recorder, text_state=path.suffix == '.json')

    @classmethod
    def from_payload(cls, payload, *, device='cpu', recorder=None, text_state=False):
        if payload.get('kind') == BLEND_KIND:
            return _blend_from_payload(cls, payload, device=device, recorder=recorder)
        if payload['schema_version'] not in (1, 2, 3) or payload['kind'] not in BACKBONES:
            raise ValueError('Unsupported checkpoint format')
        config = {**payload['model_config'], 'device': device}
        config.setdefault('kind', payload['kind'])
        if config['kind'] != payload['kind']:
            raise ValueError('Checkpoint kind and model configuration disagree')
        if payload['schema_version'] == 3 and (not {'book_representation', 'trade_features'}.issubset(config)):
            raise ValueError('Checkpoint is missing its input transformation contract')
        result = cls(config, payload['preprocessing'], seed=payload['seed'], recorder=recorder)
        if payload.get('target_centering', 'none') != result.config['target_centering']:
            raise ValueError('Checkpoint target-centering contract mismatch')
        if result.new_input_contract and payload['schema_version'] < 3:
            raise ValueError('LOB input transformations require checkpoint schema_version=3')
        if payload['schema_version'] == 1 and result.config['target_scaling'] != 'none':
            raise ValueError('Target scaling requires checkpoint schema_version=2')
        if result.architecture != _architecture_with_patch_counts(payload['architecture']):
            raise ValueError('Checkpoint architecture/preprocessing mismatch')
        state = payload['state_dict']
        if text_state:
            state = {k: torch.tensor(meta['data'], dtype=getattr(torch, meta['dtype'])).reshape(meta['shape']) for k, meta in state.items()}
        if not all((torch.isfinite(t).all() for t in state.values())):
            raise ValueError('Nonfinite checkpoint weights')
        result.network.load_state_dict(state, strict=True)
        result.intercept = float(payload['training_mean'])
        if not np.isfinite(result.intercept):
            raise ValueError('Invalid training fallback')
        if 'target_scale' not in payload and (payload['schema_version'] == 2 or result.config['target_scaling'] == 'train_std'):
            raise ValueError('Scaled checkpoint is missing target_scale')
        result.target_scale = float(payload.get('target_scale', 1.0))
        if not np.isfinite(result.target_scale) or result.target_scale <= 0:
            raise ValueError('Invalid target scale')
        if result.config['target_scaling'] == 'none' and result.target_scale != 1.0:
            raise ValueError('Unscaled checkpoint requires target_scale=1')
        result.target_centering_stats = payload.get('target_centering_stats')
        expected_centering = result.config['target_centering']
        if expected_centering != 'none' and (not isinstance(result.target_centering_stats, dict) or result.target_centering_stats.get('method') != expected_centering):
            raise ValueError('Centered checkpoint is missing its target-transform statistics')
        result.time_loss_weighting_stats = payload.get('time_loss_weighting_stats')
        if result.config['time_loss_weighting'] == 'train_variance':
            if not isinstance(result.time_loss_weighting_stats, dict):
                raise ValueError('Weighted checkpoint is missing training weight statistics')
            result.training_sample_times = tuple(result.time_loss_weighting_stats['by_time'])
        elif result.config['time_loss_weighting'] == 'date_decay_10d':
            if not isinstance(result.time_loss_weighting_stats, dict) or result.time_loss_weighting_stats.get('method') != 'date_decay_10d':
                raise ValueError('Date-weighted checkpoint is missing its training weight statistics')
        result.trained = True
        return result
BLEND_SCORE_METHODS = ('score_weighted_mean', 'score_mean')

def reduce_member_scores(method, weights, values):
    """Combine aligned member score arrays: the single reduction point for inference.

    ``score_weighted_mean`` is the historical fixed-weight sum. ``score_mean`` reproduces
    ``np.mean`` over the stacked member scores (sum first, then one division by the member
    count) so an equal-weight blend is bit-identical to that arithmetic.
    """
    if method not in BLEND_SCORE_METHODS:
        raise ValueError(f'Unsupported blend score method: {method!r}')
    if not values:
        raise ValueError('Blend reduction requires at least one member score array')
    if method == 'score_mean':
        total = values[0]
        for value in values[1:]:
            total = total + value
        return total / len(values)
    scores = None
    for weight, value in zip(weights, values):
        scores = weight * value if scores is None else scores + weight * value
    return scores

def _load_sharded_blend(adapter_cls, path, payload, *, device, recorder):
    """Read one text checkpoint at a time, retaining only loaded model tensors."""
    entries = payload.get('members')
    if payload.get('schema_version') != 1 or not isinstance(entries, list) or len(entries) < 2:
        raise ValueError('Sharded blend requires schema 1 and at least two members')
    members, weights, names, files = ([], [], [], set())
    root = path.resolve().parent
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Malformed sharded blend member')
        name, filename, weight = (entry.get('name'), entry.get('file'), entry.get('weight'))
        if not isinstance(name, str) or not name or name in names or (not isinstance(weight, (int, float))) or (not np.isfinite(weight)) or (weight < 0):
            raise ValueError('Sharded blend requires unique names and finite nonnegative weights')
        if not isinstance(filename, str) or not filename.startswith('modified_tlob_weights_') or (not filename.endswith('.json')) or any((char in filename for char in '/\\:')) or (filename in files):
            raise ValueError('Unsafe or duplicate weight shard filename')
        shard = root / filename
        if shard.is_symlink() or shard.resolve().parent != root:
            raise ValueError('Weight shard must be a regular file beside its manifest')
        size = shard.stat().st_size
        if size != entry.get('bytes') or size >= UPLOAD_WEIGHT_LIMIT:
            raise ValueError(f'Weight shard size mismatch or upload limit exceeded: {filename}')
        digest = hashlib.sha256()
        with shard.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != entry.get('sha256'):
            raise ValueError(f'Weight shard checksum mismatch: {filename}')
        with shard.open(encoding='utf-8') as source:
            member_payload = json.load(source)
        if member_payload.get('kind') not in BACKBONES:
            raise ValueError('Weight shards must contain plain torch backbone checkpoints')
        members.append(adapter_cls.from_payload(member_payload, device=device, recorder=recorder, text_state=True))
        del member_payload
        names.append(name)
        weights.append(float(weight))
        files.add(filename)
        if recorder is not None:
            recorder.event('loading', 'weight_shard_loaded', member=name, path=filename, bytes=size)
    return TLOBBlend(members, weights, names=names, method=(payload.get('blend') or {}).get('method', 'score_weighted_mean'))

def _blend_from_payload(adapter_cls, payload, *, device, recorder):
    if payload.get('schema_version') != BLEND_SCHEMA_VERSION:
        raise ValueError('Unsupported blend checkpoint schema')
    entries = payload.get('members')
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError('Blend checkpoint requires at least two members')
    members, weights, names = ([], [], [])
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get('payload'), dict):
            raise ValueError('Blend member entry is malformed')
        member_payload = entry['payload']
        if member_payload.get('kind') not in BACKBONES:
            raise ValueError('Blend members must be plain torch backbone checkpoints')
        name = entry.get('name', f'member_{index}')
        weight = entry.get('weight')
        if not isinstance(name, str) or not name or (not isinstance(weight, (int, float))):
            raise ValueError('Blend member requires a string name and numeric weight')
        names.append(name)
        weights.append(float(weight))
        members.append(adapter_cls.from_payload(member_payload, device=device, recorder=recorder, text_state=True))
    return TLOBBlend(members, weights, names=names, method=(payload.get('blend') or {}).get('method', 'score_weighted_mean'))

def _blend_shared_preprocessing(member):
    """Member preprocessing minus the lookback, which the blend slices per member."""
    return {key: value for key, value in member.preprocessing.items() if key != 'lookback'}

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

    def __init__(self, members, weights, *, names=None, method='score_weighted_mean'):
        if len(members) < 2:
            raise ValueError('Blend requires at least two members')
        if len(weights) != len(members):
            raise ValueError('Blend weights must align with members')
        weights = tuple((float(weight) for weight in weights))
        if any((not np.isfinite(weight) or weight < 0 for weight in weights)):
            raise ValueError('Blend weights must be finite and nonnegative')
        if abs(sum(weights) - 1.0) > 1e-09:
            raise ValueError('Blend weights must sum to one')
        if method not in BLEND_SCORE_METHODS:
            raise ValueError(f'Unsupported blend score method: {method!r}')
        if method == 'score_mean' and any((weight != 1.0 / len(members) for weight in weights)):
            raise ValueError('score_mean requires equal member weights')
        self.method = method
        reference = members[0]
        groups, lookbacks = ({}, [])
        for index, member in enumerate(members):
            kind = member.config.get('kind')
            if kind not in BACKBONES:
                raise ValueError(f'Blend member {index} has an unsupported trunk kind')
            if not member.trained:
                raise ValueError('Blend members must be fitted checkpoints')
            if _blend_shared_preprocessing(member) != _blend_shared_preprocessing(reference):
                raise ValueError(f'Blend member {index} preprocessing differs from member 0')
            lookback = member.preprocessing.get('lookback')
            if type(lookback) is not int or lookback < 4:
                raise ValueError(f'Blend member {index} requires an integer lookback of at least 4')
            lookbacks.append(lookback)
            if member.target_scale != reference.target_scale:
                raise ValueError('Blend members must share one prediction scale')
            for key in BLEND_SHARED_KEYS:
                if member.config.get(key) != reference.config.get(key):
                    raise ValueError(f'Blend member {index} {key} differs from member 0')
            groups.setdefault(kind, []).append((index, member))
        for kind, group in groups.items():
            first_index, first = group[0]
            for index, member in group[1:]:
                for key in BACKBONE_SHARED_KEYS[kind]:
                    if member.config.get(key) != first.config.get(key):
                        raise ValueError(f'Blend member {index} {key} differs from member {first_index}')
        self.members = tuple(members)
        self.weights = weights
        self.names = tuple(names) if names else tuple((str(index) for index in range(len(members))))
        if len(self.names) != len(self.members) or len(set(self.names)) != len(self.members):
            raise ValueError('Blend member names must be unique')
        self.kinds = tuple(groups)
        longest = max(lookbacks)
        self.preprocessing = {**_blend_shared_preprocessing(reference), 'lookback': longest}
        self.config = dict(reference.config)
        self.architecture = {**reference.architecture, 'seq_size': longest}
        self.device = reference.device
        self.target_scale = reference.target_scale
        self.intercept = float(sum((weight * member.intercept for weight, member in zip(weights, self.members))))
        self.trained = True

    @staticmethod
    def _member_batch(member, batch):
        """Trailing rows reproduce exactly the window the member was trained on."""
        lookback = member.preprocessing['lookback']
        if batch.x.shape[1] < lookback:
            raise ValueError(f'Blend batch carries {batch.x.shape[1]} bars but member requires {lookback}')
        if batch.x.shape[1] == lookback:
            return batch
        return replace(batch, x=batch.x[:, -lookback:], observed=batch.observed[:, -lookback:], history_dates=batch.history_dates[:, -lookback:])

    def predict(self, batch):
        batch_size = int(self.config.get('batch_size', 32))
        values = []
        for member in self.members:
            member.config['batch_size'] = batch_size
            values.append(np.asarray(member.predict(self._member_batch(member, batch)), dtype=np.float64))
        scores = reduce_member_scores(self.method, self.weights, values)
        if scores is None or not np.isfinite(scores).all():
            raise FloatingPointError('Nonfinite blend predictions')
        return scores

# Source: modified_tlob/submission.py
"""Platform inference: historical bars and point-in-time pool, never answers or training."""
import hashlib
import re
import time
from pathlib import Path
import pandas as pd
from jinyiquant_flat import build_batch, preprocess_bars, validate_keys
from jinyiquant_flat import canonical_json, file_sha256, read_json, utc_now, write_json
from jinyiquant_flat import Recorder, capture_source, environment, finish, new_run_dir
import numpy as np
SMOOTHING_KINDS = {'none': None, 'causal_ewma_turnover_matched': 'causal_ewma'}
SMOOTHING_STATE_RESET = 'trading_day'

def cross_sectional_z(keys, scores):
    """Population z-score per timestamp (the caliber the frozen operator measured)."""
    values = np.asarray(scores, dtype=float)
    frame = pd.DataFrame({'date': keys['date'].to_numpy(), 'score': values})
    grouped = frame.groupby('date')['score']
    mean = grouped.transform('mean').to_numpy(dtype=float)
    scale = grouped.transform('std', ddof=0).to_numpy(dtype=float)
    safe = np.where(scale > 0, scale, 1.0)
    return np.where(scale > 0, (values - mean) / safe, 0.0)

class CausalSmoother:
    """Registered causal EWMA over the cross-sectional z-score of the blended score.

    ``y_t = alpha * z_t + (1 - alpha) * y_(t-1)`` per instrument; sections are walked in
    ascending timestamp order inside one trading day and the state resets at every new day.
    Only rows at timestamps ``<= t`` are ever read, so no future row can enter a value and
    no whole-sample statistic is used.
    """

    def __init__(self, settings):
        if not isinstance(settings, dict):
            raise ValueError('Smoothing settings must be an object')
        identifier = settings.get('id', 'none')
        if identifier not in SMOOTHING_KINDS:
            raise ValueError(f'Unsupported smoothing id: {identifier!r}')
        if settings.get('state_reset', SMOOTHING_STATE_RESET) != SMOOTHING_STATE_RESET:
            raise ValueError('The causal EWMA resets its state on every trading day')
        self.identifier = identifier
        self.alpha = settings.get('alpha')
        if identifier != 'none':
            alpha = self.alpha
            if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
                raise ValueError('Causal EWMA requires a numeric alpha')
            if not 0.0 < float(alpha) <= 1.0:
                raise ValueError('Causal EWMA requires alpha in (0, 1]')
            self.alpha = float(alpha)
            self.beta = 1.0 - self.alpha
        self.day = None
        self.state = {}

    @property
    def active(self):
        return self.identifier != 'none'

    def step(self, keys, scores):
        """Smooth exactly one timestamp; the day's state carries into the next section."""
        values = np.asarray(scores, dtype=float)
        if len(keys) != len(values):
            raise ValueError('Smoothing keys and scores are not aligned')
        if not len(values):
            return values
        if keys.duplicated().any():
            raise ValueError('Smoothing keys must be unique per timestamp')
        timestamps = pd.to_datetime(keys['date'])
        days = timestamps.dt.normalize()
        if days.nunique() != 1:
            raise ValueError('One smoothing step covers exactly one timestamp')
        day = days.iloc[0]
        if self.day is None or day != self.day:
            self.state = {}
            self.day = day
        z = cross_sectional_z(keys, values)
        smoothed = np.empty(len(values), dtype=float)
        state = self.state
        alpha, beta = (self.alpha, self.beta)
        for index, name in enumerate(keys['instrument'].to_numpy()):
            previous = state.get(name)
            value = z[index] if previous is None else alpha * z[index] + beta * previous
            state[name] = value
            smoothed[index] = value
        return smoothed

def smooth_scores(keys, scores, settings):
    """Apply the smoother to a whole key frame, sections in ascending timestamp order."""
    values = np.asarray(scores, dtype=float)
    if len(keys) != len(values):
        raise ValueError('Smoothing keys and scores are not aligned')
    smoother = CausalSmoother(settings)
    if not smoother.active:
        return values
    order = np.argsort(pd.to_datetime(keys['date']).to_numpy(), kind='stable')
    frame = keys.iloc[order].reset_index(drop=True)
    ordered = values[order]
    stamps = pd.to_datetime(frame['date']).to_numpy()
    output = np.empty(len(values), dtype=float)
    start = 0
    while start < len(frame):
        stop = start + 1
        while stop < len(frame) and stamps[stop] == stamps[start]:
            stop += 1
        output[order[start:stop]] = smoother.step(frame.iloc[start:stop], ordered[start:stop])
        start = stop
    return output

def table_name(value):
    if not isinstance(value, str) or not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', value):
        raise ValueError('Data source must be a simple SQL table identifier')
    return value

def query_frame(query, table, columns, start, end):
    return query(f"SELECT {', '.join(columns)} FROM {table_name(table)}", filters={'date': [str(start), str(end)]}).df()

def iter_bar_days(query, table, warm_start, end, data):
    """Retain the same trailing observations as one full-range read, in bounded daily blocks."""
    cursor, end = (pd.Timestamp(warm_start), pd.Timestamp(end))
    columns = ['date', 'instrument', *data['feature_columns']]
    carry = None
    lookback = int(data['lookback'])
    while cursor <= end:
        next_day = cursor.normalize() + pd.Timedelta(days=1)
        upper = min(next_day, end)
        final_block = end < next_day
        raw = query_frame(query, table, columns, cursor, upper)
        raw['date'] = pd.to_datetime(raw['date'])
        within_end = raw.date <= upper if final_block else raw.date < upper
        raw = raw.loc[(raw.date >= cursor) & within_end]
        current = preprocess_bars(raw, data)
        del raw
        metadata = {'start': str(cursor), 'end': str(upper), 'end_inclusive': final_block, 'rows': len(current), 'preprocessed_bars_sha256': hashlib.sha256(pd.util.hash_pandas_object(current, index=False).to_numpy().tobytes()).hexdigest()}
        window = current if carry is None or carry.empty else pd.concat([carry, current], ignore_index=True)
        del current
        carry = window.groupby('instrument', sort=False, observed=True).tail(lookback).copy()
        metadata['retained_rows'] = len(carry)
        yield (window, metadata)
        del window
        if final_block:
            break
        cursor = next_day

def section_history(bars, keys, lookback):
    """Prune irrelevant rows before the shared window builder copies/sorts them."""
    latest = keys['date'].max()
    selected = bars.loc[(bars.date <= latest) & bars.instrument.isin(keys.instrument)]
    return selected.groupby('instrument', sort=False, observed=True).tail(lookback)

class HistoryRecovery:
    """Cache bounded, strictly earlier quotes only for insufficient windows."""

    def __init__(self, query, table, data, warm_start, policy, deadline, recorder):
        self.query, self.table, self.data = (query, table_name(table), data)
        self.warm_start, self.deadline, self.recorder = (warm_start, deadline, recorder)
        self.max_days = policy.get('max_backfill_days', 366)
        self.group_size = policy.get('instrument_batch_size', 50)
        if not isinstance(self.max_days, int) or not 1 <= self.max_days <= 366 or (not isinstance(self.group_size, int)) or (not 1 <= self.group_size <= 50):
            raise ValueError('History recovery requires 1..366 days and 1..50 instruments per query')
        self.cache, self.attempted = ({}, set())

    def recover(self, history, batch):
        missing = batch.effective_rows.sum(axis=1) < self.data['min_history']
        if not missing.any():
            return batch
        row_counts = (~pd.isna(batch.history_dates)).sum(axis=1)
        names = batch.keys.loc[missing, 'instrument'].tolist()
        pending = [name for name, count in zip(batch.keys.loc[missing, 'instrument'], row_counts[missing]) if name not in self.attempted and count < self.data['lookback']]
        cutoff = batch.keys.date.max()
        lower = cutoff - pd.Timedelta(days=self.max_days)
        columns = ['date', 'instrument', *self.data['feature_columns']]
        for offset in range(0, len(pending), self.group_size):
            group = pending[offset:offset + self.group_size]
            if time.monotonic() > self.deadline:
                raise TimeoutError('Platform inference budget exceeded during history recovery')
            if lower < self.warm_start:
                sql = f"SELECT {', '.join(columns)} FROM {self.table} WHERE date >= TIMESTAMP '{lower}' AND date < TIMESTAMP '{self.warm_start}' QUALIFY ROW_NUMBER() OVER (PARTITION BY instrument ORDER BY date DESC) <= {int(self.data['lookback'])}"
                raw = self.query(sql, filters={'date': [str(lower), str(self.warm_start)], 'instrument': group}).df()
                raw['date'] = pd.to_datetime(raw['date'])
                raw = raw.loc[raw.instrument.isin(group) & raw.date.ge(lower) & raw.date.lt(self.warm_start)]
                recovered = preprocess_bars(raw, self.data)
                if recovered.groupby('instrument', observed=True).size().gt(self.data['lookback']).any():
                    raise ValueError('History recovery query exceeded its per-instrument row limit')
                for name, frame in recovered.groupby('instrument', sort=False, observed=True):
                    self.cache[name] = frame.copy()
                self.recorder.event('data', 'history_backfilled', date=str(cutoff), start=str(lower), end_exclusive=str(self.warm_start), requested_instruments=len(group), rows=len(recovered), rows_per_instrument_limit=self.data['lookback'], instruments_sha256=hashlib.sha256(canonical_json(group).encode()).hexdigest(), preprocessed_bars_sha256=hashlib.sha256(pd.util.hash_pandas_object(recovered, index=False).to_numpy().tobytes()).hexdigest())
            self.attempted.update(group)
        supplements = [self.cache[name] for name in names if name in self.cache]
        if not supplements:
            return batch
        combined = pd.concat([*supplements, history], ignore_index=True)
        return build_batch(section_history(combined, batch.keys, self.data['lookback']), batch.keys, self.data)

def predict_range(datasources, start_date, end_date, *, package_root=None, query=None):
    root = Path(package_root) if package_root else Path(__file__).resolve().parent
    run_dir = new_run_dir(root / 'inference_runs')
    rec = Recorder(run_dir, 'modified-tlob-platform-inference')
    started = time.monotonic()
    selected = 'unknown'
    write_json(run_dir / 'manifest.json', {'schema_version': 1, 'run_id': run_dir.name, 'experiment_id': rec.experiment_id, 'started_at': utc_now(), 'status': 'running', 'scope': 'platform_inference_only', 'start_date': str(start_date), 'end_date': str(end_date)})
    rec.event('run', 'started', start_date=str(start_date), end_date=str(end_date), label_input=False)
    try:
        if query is None:
            import dai
            query = dai.query
        start, end = (pd.Timestamp(start_date), pd.Timestamp(end_date))
        if pd.isna(start) or pd.isna(end) or start.tzinfo or end.tzinfo or (start > end):
            raise ValueError('Expected naive exchange-local start <= end')
        weights = root / 'modified_tlob_weights.json'
        contract = read_json(root / 'submission_config.json')
        if contract['weights_sha256'] != file_sha256(weights):
            raise ValueError('Submission weight checksum mismatch')
        smoother = CausalSmoother(contract.get('smoothing') or {'id': 'none'})
        manifest = read_json(run_dir / 'manifest.json')
        manifest.update(code=capture_source(run_dir), environment=environment(), model_sha256=contract['weights_sha256'], contract=contract)
        write_json(run_dir / 'manifest.json', manifest)
        rec.event('loading', 'started', path='modified_tlob_weights.json')
        model = TLOBAdapter.load(weights, device=contract['device'], recorder=rec)
        selected = str(model.device)
        manifest['environment']['determinism'].update(device=selected, cpu_threads=model.config['cpu_threads'], torch_version=manifest['environment']['dependencies'].get('torch'), deterministic_algorithms=True)
        write_json(run_dir / 'manifest.json', manifest)
        rec.event('loading', 'model_loaded', path='modified_tlob_weights.json')
        data = dict(model.preprocessing)
        data['price_scale'] = contract['platform_price_scale']
        warm_start = start - pd.Timedelta(days=data['history_days'])
        bar_table = table_name(datasources['bar1m'])
        history_policy = contract.get('history_policy', {'mode': 'strict'})
        if history_policy.get('mode') not in {'strict', 'recover_then_mean'}:
            raise ValueError('Unknown platform history policy')
        recovery = HistoryRecovery(query, bar_table, data, warm_start, history_policy, started + contract['wall_seconds'], rec) if history_policy['mode'] == 'recover_then_mean' else None
        rec.event('data', 'loading', source=table_name(datasources['bar1m']), feature_columns=data['feature_columns'], warm_start=str(warm_start), strategy='daily_with_per_instrument_tail', resources=device_resources(model.device))
        pool = query_frame(query, datasources.get('instruments', 'cpt_jyc_2026_instruments'), ['date', 'instrument'], start, end)
        pool = validate_keys(pool, 'platform universe')
        keys = pool.loc[(pool.date >= start) & (pool.date <= end) & pool.date.dt.strftime('%H:%M').isin(data['sample_times'])]
        del pool
        if keys.empty:
            raise ValueError('No eligible platform prediction keys')
        rec.event('data', 'validated', keys=len(keys), label_input=False)
        metadata = {'schema_version': 2, 'source': 'platform_DAI', 'bar_table': bar_table, 'universe_table': datasources.get('instruments', 'cpt_jyc_2026_instruments'), 'start': str(start), 'end': str(end), 'warm_start': str(warm_start), 'strategy': 'daily_with_per_instrument_tail', 'retained_rows_per_instrument': data['lookback'], 'keys_sha256': hashlib.sha256(pd.util.hash_pandas_object(keys, index=False).to_numpy().tobytes()).hexdigest(), 'feature_columns': data['feature_columns'], 'platform_price_scale': data['price_scale'], 'rows': 0, 'prediction_keys': len(keys), 'label_input': False, 'history_policy': history_policy, 'chunks': []}
        write_json(run_dir / 'data.json', metadata)
        rec.event('inference', 'started', label_input=False)
        results = []
        fallback_samples, recovered_samples, usable_samples = (0, 0, 0)
        smoothed_sections = 0
        warned_days = set()
        key_days = keys.date.dt.normalize()
        for bars, chunk in iter_bar_days(query, bar_table, warm_start, end, data):
            if time.monotonic() - started > contract['wall_seconds']:
                raise TimeoutError('Platform inference wall-time budget exceeded')
            metadata['rows'] += chunk['rows']
            metadata['chunks'].append(chunk)
            write_json(run_dir / 'data.json', metadata)
            rec.event('data', 'chunk_loaded', **chunk, window_rows=len(bars), resources=device_resources(model.device))
            day_keys = keys.loc[key_days.eq(pd.Timestamp(chunk['start']).normalize())]
            for date, group in day_keys.groupby('date', sort=True):
                if time.monotonic() - started > contract['wall_seconds']:
                    raise TimeoutError('Platform inference wall-time budget exceeded')
                history = section_history(bars, group, data['lookback'])
                batch = build_batch(history, group, data)
                before = batch.coverage['usable_samples']
                if recovery is not None:
                    batch = recovery.recover(history, batch)
                del history
                expected, usable = (batch.coverage['expected_samples'], batch.coverage['usable_samples'])
                insufficient = expected - usable
                rec.event('preprocessing', 'history_checked', date=str(date), coverage=batch.coverage, usable_before_recovery=before, recovered_samples=usable - before, history_policy=history_policy['mode'])
                if batch.coverage['min_section_coverage'] < contract['min_coverage'] and recovery is None:
                    raise ValueError(f"Insufficient history coverage at {date}: usable={usable}/{expected}, required={contract['min_coverage']}, min_history={data['min_history']}, warm_start={warm_start}")
                if insufficient and recovery is not None:
                    detail = {'date': str(date), 'expected': expected, 'usable': usable, 'fallback_samples': insufficient, 'fallback': 'saved_training_mean', 'coverage': batch.coverage['min_section_coverage'], 'warning_threshold': contract['min_coverage']}
                    rec.event('inference', 'history_fallback', level='WARNING', **detail)
                    if date.normalize() not in warned_days:
                        print(canonical_json({'event_type': 'history_fallback', **detail}), flush=True)
                        warned_days.add(date.normalize())
                fallback_samples += insufficient
                recovered_samples += usable - before
                usable_samples += usable
                rec.event('preprocessing', 'completed', date=str(date), coverage=batch.coverage)
                scores = model.predict(batch)
                if smoother.active:
                    scores = smoother.step(batch.keys, scores)
                    smoothed_sections += 1
                results.append(batch.keys.assign(factor=scores))
                del batch
            del bars
        result = pd.concat(results, ignore_index=True)[['date', 'instrument', 'factor']]
        if not result[['date', 'instrument']].equals(keys.reset_index(drop=True)[['date', 'instrument']]):
            raise ValueError('Prediction universe alignment failed')
        manifest = read_json(run_dir / 'manifest.json')
        manifest['history_handling'] = {'policy': history_policy, 'usable_samples': usable_samples, 'recovered_samples': recovered_samples, 'fallback_samples': fallback_samples, 'data_quality': 'fallback_used' if fallback_samples else 'all_usable'}
        manifest['smoothing'] = {'id': smoother.identifier, 'alpha': smoother.alpha if smoother.active else None, 'sections_smoothed': smoothed_sections, 'state_reset': SMOOTHING_STATE_RESET, 'causality_rule': 'a value at timestamp t reads only rows at timestamps <= t of the same trading day', 'operator': 'cross-sectional population z-score, then causal per-day EWMA'}
        write_json(run_dir / 'manifest.json', manifest)
        if recovery is not None and usable_samples == 0:
            raise ValueError('No usable historical quotes in the entire prediction range, even after backfill; check the bar datasource, instrument identifiers and available dates')
        rec.event('inference', 'completed', samples=len(result), label_input=False)
        elapsed = time.monotonic() - started
        if elapsed > contract['wall_seconds']:
            raise TimeoutError('Platform inference wall-time budget exceeded')
        finish(run_dir, 'completed', 0, resources={'wall_seconds': elapsed, **device_resources(model.device)})
        return result
    except BaseException as exc:
        rec.failure('inference', exc)
        finish(run_dir, 'failed', 1, resources={'wall_seconds': time.monotonic() - started, 'device': selected}, error={'exception_type': type(exc).__name__, 'reason': str(exc)})
        raise

def main(datasources, start_date, end_date):
    return predict_range(datasources, start_date, end_date)
