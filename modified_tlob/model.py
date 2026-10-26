"""Fixed upstream TLOB/MLPLOB trunks adapted to scalar regression. See provenance.json (MIT).

``DualPatchTLOB`` is a project extension: it keeps the same 128-bar raw LOB input,
BiN and scalar head contract but replaces the trunk with two channel-independent
PatchTST-style encoders over non-overlapping 4-patch (32-bar) and 16-patch (8-bar)
scales (Nie et al., PatchTST, ICLR 2023).
"""

import torch
from torch import nn

from .normalization import BiN


class ComputeQKV(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim * num_heads)

    def forward(self, x):
        return self.q(x), self.k(x), self.v(x)


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
        # Keep MHA's own in-projection in addition to upstream's external Q/K/V.
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
        raise ValueError("Sinusoidal embedding requires even hidden_dim")
    positions = torch.arange(seq_size).unsqueeze(1)
    denominators = torch.pow(10000., 2 * torch.arange(hidden_dim // 2) / hidden_dim)
    result = torch.zeros(seq_size, hidden_dim)
    result[:, 0::2] = torch.sin(positions / denominators)
    result[:, 1::2] = torch.cos(positions / denominators)
    return result


class ModifiedTLOB(nn.Module):
    def __init__(self, *, hidden_dim=40, num_layers=4, seq_size=128, num_features=12,
                 num_heads=1, is_sin_emb=True):
        super().__init__()
        dimensions = (hidden_dim, num_layers, seq_size, num_features, num_heads)
        if any(type(d) is not int or d <= 0 for d in dimensions):
            raise ValueError("Model dimensions must be positive integers")
        if hidden_dim < 4 or seq_size < 4 or num_features < 2:
            raise ValueError("Quarter compression requires hidden_dim/seq_size >= 4 and features >= 2")
        self.seq_size, self.num_features = seq_size, num_features
        self.norm_layer = BiN(num_features, seq_size)
        self.emb_layer = nn.Linear(num_features, hidden_dim)
        if is_sin_emb:
            self.register_buffer("pos_encoder", sinusoidal_positional_embedding(seq_size, hidden_dim))
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, seq_size, hidden_dim))
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            last = i == num_layers - 1
            self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim // 4 if last else hidden_dim))
            self.layers.append(TransformerLayer(seq_size, num_heads, seq_size // 4 if last else seq_size))
        total_dim = (hidden_dim // 4) * (seq_size // 4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.extend([nn.Linear(total_dim, total_dim // 4), nn.GELU()])
            total_dim //= 4
        self.final_layers.append(nn.Linear(total_dim, 1))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.seq_size, self.num_features):
            raise ValueError(f"Expected [batch, {self.seq_size}, {self.num_features}]")
        if not torch.isfinite(x).all():
            raise ValueError("Input must be finite after shared missing-value preprocessing")
        x = self.norm_layer(x.transpose(1, 2)).transpose(1, 2)
        x = self.emb_layer(x) + self.pos_encoder
        for layer in self.layers:
            x = layer(x).transpose(1, 2)
        # Upstream rearrange 'b s f -> b (f s) 1', not a direct [s,f] flatten.
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
        if any(type(d) is not int or d <= 0 for d in dimensions):
            raise ValueError("Model dimensions must be positive integers")
        if hidden_dim < 4 or seq_size < 4 or num_features < 2:
            raise ValueError("Quarter compression requires hidden_dim/seq_size >= 4 and features >= 2")
        self.seq_size, self.num_features = seq_size, num_features
        self.norm_layer = BiN(num_features, seq_size)
        self.layers = nn.ModuleList([nn.Linear(num_features, hidden_dim), nn.GELU()])
        for i in range(num_layers):
            last = i == num_layers - 1
            wide = 2 if last else 4  # Upstream doubles the last block instead of the 4x interior width.
            self.layers.append(MLP(hidden_dim, hidden_dim * wide,
                                   hidden_dim // 4 if last else hidden_dim))
            self.layers.append(MLP(seq_size, seq_size * wide,
                                   seq_size // 4 if last else seq_size))
        total_dim = (hidden_dim // 4) * (seq_size // 4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.extend([nn.Linear(total_dim, total_dim // 4), nn.GELU()])
            total_dim //= 4
        self.final_layers.append(nn.Linear(total_dim, 1))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.seq_size, self.num_features):
            raise ValueError(f"Expected [batch, {self.seq_size}, {self.num_features}]")
        if not torch.isfinite(x).all():
            raise ValueError("Input must be finite after shared missing-value preprocessing")
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
            raise ValueError("patch_count must be an integer >= 2 that divides the lookback exactly")
        if type(hidden_dim) is not int or type(num_heads) is not int or hidden_dim % num_heads:
            raise ValueError("Patch attention requires hidden_dim divisible by num_heads")
        self.patch_count = patch_count
        self.patch_length = seq_size // patch_count
        self.embed = nn.Linear(self.patch_length, hidden_dim)
        self.position = nn.Parameter(torch.empty(1, patch_count, hidden_dim))
        nn.init.uniform_(self.position, -0.02, 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("PatchEncoder expects [batch, features, lookback]")
        batch, features, seq_size = x.shape
        if seq_size != self.patch_count * self.patch_length:
            raise ValueError("PatchEncoder lookback differs from its patch contract")
        patches = x.reshape(batch * features, self.patch_count, self.patch_length)
        tokens = self.embed(patches) + self.position
        return self.norm(self.encoder(tokens))


class DualPatchTLOB(nn.Module):
    """Dual-scale patch trunk over the unchanged 128x12 raw order-book window.

    Two channel-independent patch encoders summarize non-overlapping 32-bar
    (4-patch, slower structure) and 8-bar (16-patch, fast changes) scales. Each
    branch projects every channel's patch tokens to a short embedding, the branch
    embeddings are concatenated and a scalar regression head fuses them. BiN and
    the scalar output contract are shared with the other trunks.
    """

    def __init__(self, *, hidden_dim=40, num_layers=3, seq_size=128, num_features=12,
                 num_heads=4, patch_counts=(4, 16)):
        super().__init__()
        dimensions = (hidden_dim, num_layers, seq_size, num_features, num_heads)
        if any(type(d) is not int or d <= 0 for d in dimensions):
            raise ValueError("Model dimensions must be positive integers")
        if hidden_dim < 4 or seq_size < 4 or num_features < 2:
            raise ValueError("Dual patch requires hidden_dim/seq_size >= 4 and features >= 2")
        counts = tuple(patch_counts)
        if len(counts) != 2 or any(type(c) is not int for c in counts) or len(set(counts)) != 2:
            raise ValueError("patch_counts requires two distinct integer scales")
        if any(count < 2 or seq_size % count for count in counts):
            raise ValueError("patch_counts must be >= 2 and divide the lookback exactly")
        if counts[0] > counts[1]:
            raise ValueError("patch_counts must be ordered coarse-to-fine (fewer patches first)")
        self.seq_size, self.num_features = seq_size, num_features
        self.patch_counts = counts
        self.channel_dim = DUAL_PATCH_CHANNEL_DIM
        self.norm_layer = BiN(num_features, seq_size)
        self.branches = nn.ModuleList([
            PatchEncoder(seq_size=seq_size, patch_count=count, hidden_dim=hidden_dim,
                         num_layers=num_layers, num_heads=num_heads) for count in counts])
        self.channel_heads = nn.ModuleList([
            nn.Linear(count * hidden_dim, self.channel_dim) for count in counts])
        fused = len(counts) * num_features * self.channel_dim
        self.head = nn.Sequential(nn.Linear(fused, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.seq_size, self.num_features):
            raise ValueError(f"Expected [batch, {self.seq_size}, {self.num_features}]")
        if not torch.isfinite(x).all():
            raise ValueError("Input must be finite after shared missing-value preprocessing")
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
    return tuple(dict.fromkeys(name.split(".")[0] for name, _ in network.named_parameters()))
