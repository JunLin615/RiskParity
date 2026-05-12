"""
dual_transformer_mode_v7.py

Dual-Transformer V7 for cross-sectional ranking.

Purpose
-------
This version keeps the V5 architecture direction, but fixes the diagnosed
collapse inside the intra-stock factor compression module.

Key changes
-----------
1. temporal_compressed_len default is 64 and is not derived from legacy
   model_dim/temporal_channels compatibility fields.

2. Activation can be Softsign. Default activation is now "softsign".
   This is intended to avoid turning oscillatory financial time-series into
   sparse positive-peak "spectra" through zero-mean normalization + GELU/ReLU-like
   nonlinearities.

3. Factor self-attention output gets an explicit LayerNorm at the end.

4. Reduce cross-attention receives a gated residual injection from the
   self-attention output:
       reduced = reduce_cross(query, self_attn_out)
       residual = projection(pool(self_attn_out))
       reduced = reduced + gate * residual
       reduced = LayerNorm(reduced)

   The gate is learnable and initialized small, so the model can explore the
   residual strength instead of forcing a bypass.

5. Scalar factors are still ignored by default:
       use_scalar_factors = False

6. Stock aggregation/broadcast residual switches are retained and default True
   in this file, because your latest manual experiment indicates the collapse
   source is not primarily there. You can still set them False if needed.

Compatibility
-------------
Public API remains:
    DualTransformerRanker.from_feature_counts(...)
    model(x_ts, x_scalar=None, stock_valid_mask=None, return_dict=False)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
from torch import nn


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class DualTransformerConfig:
    num_ts_factors: int
    num_scalar_factors: int
    seq_len: int = 128

    # Stage 1: independent factor denoising.
    depthwise_layers: int = 3
    depthwise_kernel_size: int = 5

    # Stage 2: time compression. Keep at 64 by default.
    temporal_compressed_len: int = 64

    # Stage 3: cross-factor mixing and channel reduction.
    mixed_channels_1: int = 64
    mixed_channels_2: int = 32
    temporal_out_channels: int = 16
    mixed_kernel_size: int = 5

    # Normalization and regularization.
    group_norm_groups: int = 32
    dropout: float = 0.1
    activation: str = "softsign"
    input_nan_to_num: bool = True

    # Intra-stock factor attention.
    factor_num_layers: int = 1
    factor_num_heads: int = 4
    factor_ff_dim: int = 256
    factor_reduce_tokens: int = 8
    factor_use_positional_encoding: bool = True

    # New anti-collapse controls.
    factor_self_output_norm: bool = True
    reduce_residual_pool: str = "adaptive"  # "adaptive" or "mean"
    reduce_residual_gate_init: float = -2.0  # sigmoid(-2) ≈ 0.119
    reduce_residual_projection: bool = True

    # Cross-stock aggregation-broadcast attention.
    stock_aggregate_tokens: int = 32
    cross_num_heads: int = 4
    cross_ff_dim: int = 256
    aggregate_residual_query: bool = True
    broadcast_residual_query: bool = True

    # Scalar anti-overfitting switch.
    use_scalar_factors: bool = False

    # Score head.
    score_hidden_dim: Optional[int] = 128
    score_head_layers: int = 2

    # Compatibility fields accepted from older configs.
    # They are not used to derive V7 dimensions.
    temporal_hidden_channels: Optional[int] = None
    temporal_channels: Optional[int] = None
    temporal_kernel_size: Optional[int] = None
    temporal_conv_layers: Optional[int] = None
    model_dim: Optional[int] = None
    cross_num_layers: int = 1
    norm_first: bool = True
    output_squeeze_batch_if_unbatched: bool = True

    def token_dim(self) -> int:
        return int(self.temporal_compressed_len)

    def temporal_token_count(self) -> int:
        return int(self.temporal_out_channels)

    def scalar_token_count(self) -> int:
        return 1 if (bool(self.use_scalar_factors) and int(self.num_scalar_factors) > 0) else 0

    def factor_token_count(self) -> int:
        return self.temporal_token_count() + self.scalar_token_count()

    def stock_embedding_dim(self) -> int:
        return int(self.factor_reduce_tokens) * self.token_dim()

    def resolved_model_dim(self) -> int:
        return self.stock_embedding_dim()


# ============================================================
# Utilities
# ============================================================

def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "softsign":
        return nn.Softsign()
    if name == "tanh":
        return nn.Tanh()
    if name in {"identity", "linear", "none"}:
        return nn.Identity()
    raise ValueError(f"unsupported activation: {name!r}")


def _largest_valid_group_count(num_channels: int, requested_groups: int) -> int:
    c = int(num_channels)
    g = min(max(1, int(requested_groups)), c)
    while g > 1:
        if c % g == 0:
            return g
        g -= 1
    return 1


def _make_group_norm(num_channels: int, requested_groups: int) -> nn.GroupNorm:
    groups = _largest_valid_group_count(num_channels, requested_groups)
    return nn.GroupNorm(num_groups=groups, num_channels=int(num_channels))


def _make_depthwise_norm(num_channels: int) -> nn.GroupNorm:
    # Per-factor normalization over time, no cross-factor statistic mixing.
    return nn.GroupNorm(num_groups=int(num_channels), num_channels=int(num_channels))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ============================================================
# Attention blocks
# ============================================================

class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention block with configurable activation FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        activation: str = "softsign",
    ) -> None:
        super().__init__()
        if int(dim) % int(num_heads) != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.norm1 = nn.LayerNorm(int(dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=int(dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.drop1 = nn.Dropout(float(dropout))

        self.norm2 = nn.LayerNorm(int(dim))
        self.ff = nn.Sequential(
            nn.Linear(int(dim), int(ff_dim)),
            _activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(int(ff_dim), int(dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.ff(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block with configurable activation FFN."""

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        activation: str = "softsign",
    ) -> None:
        super().__init__()
        if int(query_dim) != int(kv_dim):
            raise ValueError("CrossAttentionBlock expects query_dim == kv_dim")
        if int(query_dim) % int(num_heads) != 0:
            raise ValueError(f"dim={query_dim} must be divisible by num_heads={num_heads}")

        self.dim = int(query_dim)
        self.q_norm = nn.LayerNorm(self.dim)
        self.kv_norm = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.drop = nn.Dropout(float(dropout))

        self.ff_norm = nn.LayerNorm(self.dim)
        self.ff = nn.Sequential(
            nn.Linear(self.dim, int(ff_dim)),
            _activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(int(ff_dim), self.dim),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        residual_query: bool = True,
    ) -> torch.Tensor:
        q = self.q_norm(query)
        kv = self.kv_norm(key_value)
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        if residual_query:
            x = query + self.drop(out)
        else:
            x = self.drop(out)
        x = x + self.ff(self.ff_norm(x))
        return x


# ============================================================
# Temporal encoder
# ============================================================

class DepthwiseThenMixedTemporalEncoder(nn.Module):
    """
    Temporal encoder.

    Input:
        x_ts: [B, N, F_ts, T]

    Output:
        temporal_tokens: [B, N, temporal_out_channels, temporal_compressed_len]
    """

    def __init__(self, config: DualTransformerConfig) -> None:
        super().__init__()
        self.config = config

        f_ts = int(config.num_ts_factors)
        if f_ts <= 0:
            raise ValueError("num_ts_factors must be positive")
        if int(config.depthwise_layers) < 1:
            raise ValueError("depthwise_layers must be >= 1")
        if int(config.temporal_compressed_len) <= 0:
            raise ValueError("temporal_compressed_len must be positive")

        dw_k = int(config.depthwise_kernel_size)
        dw_pad = dw_k // 2
        mix_k = int(config.mixed_kernel_size)
        mix_pad = mix_k // 2

        depthwise_layers: list[nn.Module] = []
        for _ in range(int(config.depthwise_layers)):
            depthwise_layers += [
                nn.Conv1d(f_ts, f_ts, kernel_size=dw_k, padding=dw_pad, groups=f_ts),
                _make_depthwise_norm(f_ts),
                _activation(config.activation),
                nn.Dropout(float(config.dropout)),
            ]
        self.depthwise_denoise = nn.Sequential(*depthwise_layers)

        self.pool = nn.AdaptiveAvgPool1d(int(config.temporal_compressed_len))

        c1 = int(config.mixed_channels_1)
        c2 = int(config.mixed_channels_2)
        c3 = int(config.temporal_out_channels)

        self.mixed_reduce = nn.Sequential(
            nn.Conv1d(f_ts, c1, kernel_size=mix_k, padding=mix_pad),
            _make_group_norm(c1, int(config.group_norm_groups)),
            _activation(config.activation),
            nn.Dropout(float(config.dropout)),

            nn.Conv1d(c1, c2, kernel_size=mix_k, padding=mix_pad),
            _make_group_norm(c2, int(config.group_norm_groups)),
            _activation(config.activation),
            nn.Dropout(float(config.dropout)),

            nn.Conv1d(c2, c3, kernel_size=mix_k, padding=mix_pad),
            _make_group_norm(c3, int(config.group_norm_groups)),
            _activation(config.activation),
            nn.Dropout(float(config.dropout)),
        )

    def forward(self, x_ts: torch.Tensor) -> torch.Tensor:
        if x_ts.ndim != 4:
            raise ValueError(f"x_ts must have shape [B,N,F_ts,T], got {tuple(x_ts.shape)}")

        b, n, f_ts, t = x_ts.shape
        if f_ts != int(self.config.num_ts_factors):
            raise ValueError(f"expected F_ts={self.config.num_ts_factors}, got {f_ts}")
        if t != int(self.config.seq_len):
            raise ValueError(f"expected T={self.config.seq_len}, got {t}")

        if self.config.input_nan_to_num:
            x_ts = torch.nan_to_num(x_ts, nan=0.0, posinf=0.0, neginf=0.0)

        x = x_ts.reshape(b * n, f_ts, t)
        x = self.depthwise_denoise(x)
        x = self.pool(x)
        x = self.mixed_reduce(x)
        return x.reshape(
            b,
            n,
            int(self.config.temporal_out_channels),
            int(self.config.temporal_compressed_len),
        )


# ============================================================
# Scalar encoder
# ============================================================

class ScalarVectorTokenEncoder(nn.Module):
    """Map scalar factors to one token, disabled by default through config."""

    def __init__(
        self,
        num_scalar_factors: int,
        token_dim: int,
        dropout: float = 0.1,
        activation: str = "softsign",
    ) -> None:
        super().__init__()
        self.num_scalar_factors = int(num_scalar_factors)
        self.token_dim = int(token_dim)

        if self.num_scalar_factors > 0:
            hidden = max(self.token_dim, self.num_scalar_factors * 4)
            self.net = nn.Sequential(
                nn.LayerNorm(self.num_scalar_factors),
                nn.Linear(self.num_scalar_factors, hidden),
                _activation(activation),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, self.token_dim),
                nn.LayerNorm(self.token_dim),
                nn.Dropout(float(dropout)),
            )
        else:
            self.net = None

    def forward(
        self,
        x_scalar: Optional[torch.Tensor],
        batch_size: int,
        num_stocks: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.num_scalar_factors == 0:
            return torch.empty(batch_size, num_stocks, 0, self.token_dim, device=device)

        if x_scalar is None:
            raise ValueError("x_scalar is required because num_scalar_factors > 0")
        if x_scalar.ndim != 3:
            raise ValueError(f"x_scalar must have shape [B,N,F_sc], got {tuple(x_scalar.shape)}")

        b, n, f_sc = x_scalar.shape
        if b != batch_size or n != num_stocks or f_sc != self.num_scalar_factors:
            raise ValueError(f"x_scalar shape mismatch, got {tuple(x_scalar.shape)}")

        x = torch.nan_to_num(x_scalar, nan=0.0, posinf=0.0, neginf=0.0)
        assert self.net is not None
        return self.net(x).unsqueeze(2)


# ============================================================
# Intra-stock compressor
# ============================================================

class IntraStockTokenCompressor(nn.Module):
    """
    Stock-local token processing.

    Input:
        tokens [B, N, F_total, D]

    Output:
        reduced_tokens [B, N, R, D]
        stock_embedding [B, N, R*D]

    Main anti-collapse mechanism:
        self-attention output is normalized;
        reduce cross-attention receives a gated residual from pooled self-attn tokens.
    """

    def __init__(self, config: DualTransformerConfig) -> None:
        super().__init__()
        self.config = config
        d = config.token_dim()
        f_total = config.factor_token_count()
        r = int(config.factor_reduce_tokens)

        if f_total <= 0:
            raise ValueError("factor_token_count must be positive")

        if bool(config.factor_use_positional_encoding):
            self.source_pos_embedding = nn.Parameter(torch.zeros(1, f_total, d))
            self.query_pos_embedding = nn.Parameter(torch.zeros(1, r, d))
        else:
            self.register_parameter("source_pos_embedding", None)
            self.register_parameter("query_pos_embedding", None)

        self.factor_self_blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=d,
                num_heads=int(config.factor_num_heads),
                ff_dim=int(config.factor_ff_dim),
                dropout=float(config.dropout),
                activation=config.activation,
            )
            for _ in range(int(config.factor_num_layers))
        ])

        self.self_output_norm = nn.LayerNorm(d) if bool(config.factor_self_output_norm) else nn.Identity()

        self.reduce_queries = nn.Parameter(torch.zeros(1, r, d))
        self.reduce_cross = CrossAttentionBlock(
            query_dim=d,
            kv_dim=d,
            num_heads=int(config.factor_num_heads),
            ff_dim=int(config.factor_ff_dim),
            dropout=float(config.dropout),
            activation=config.activation,
        )

        if str(config.reduce_residual_pool).lower() == "adaptive":
            self.residual_pool = nn.AdaptiveAvgPool1d(r)
        elif str(config.reduce_residual_pool).lower() == "mean":
            self.residual_pool = None
        else:
            raise ValueError("reduce_residual_pool must be 'adaptive' or 'mean'")

        if bool(config.reduce_residual_projection):
            self.reduce_residual_proj = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d),
            )
        else:
            self.reduce_residual_proj = nn.Identity()

        self.reduce_residual_gate_logit = nn.Parameter(
            torch.tensor(float(config.reduce_residual_gate_init), dtype=torch.float32)
        )

        self.out_norm = nn.LayerNorm(d)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.reduce_queries, mean=0.0, std=0.02)
        if self.source_pos_embedding is not None:
            nn.init.normal_(self.source_pos_embedding, mean=0.0, std=0.02)
        if self.query_pos_embedding is not None:
            nn.init.normal_(self.query_pos_embedding, mean=0.0, std=0.02)

    def _make_reduce_residual(self, x: torch.Tensor, r: int) -> torch.Tensor:
        """
        x: [B*N, F_total, D]
        return: [B*N, R, D]
        """
        if self.residual_pool is None:
            pooled = x.mean(dim=1, keepdim=True).expand(-1, r, -1)
        else:
            # Pool along token dimension: [BN,F,D] -> [BN,D,F] -> [BN,D,R] -> [BN,R,D]
            pooled = self.residual_pool(x.transpose(1, 2)).transpose(1, 2)
        return self.reduce_residual_proj(pooled)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 4:
            raise ValueError(f"tokens must have shape [B,N,F,D], got {tuple(tokens.shape)}")

        b, n, f_total, d = tokens.shape
        r = int(self.config.factor_reduce_tokens)
        x = tokens.reshape(b * n, f_total, d)

        if self.source_pos_embedding is not None:
            if f_total > self.source_pos_embedding.shape[1]:
                raise ValueError("token count exceeds source positional embedding length")
            x = x + self.source_pos_embedding[:, :f_total, :]

        for block in self.factor_self_blocks:
            x = block(x)

        x = self.self_output_norm(x)

        q = self.reduce_queries.expand(b * n, -1, -1)
        if self.query_pos_embedding is not None:
            q = q + self.query_pos_embedding

        reduced = self.reduce_cross(q, x, residual_query=True)

        residual = self._make_reduce_residual(x, r)
        gate = torch.sigmoid(self.reduce_residual_gate_logit)
        reduced = reduced + gate * residual

        reduced = self.out_norm(reduced)
        reduced = reduced.reshape(b, n, r, d)
        flat = reduced.flatten(start_dim=2)
        return reduced, flat


# ============================================================
# Cross-stock aggregation-broadcast
# ============================================================

class StockAggregationBroadcast(nn.Module):
    def __init__(self, config: DualTransformerConfig) -> None:
        super().__init__()
        self.config = config
        d = config.stock_embedding_dim()
        k = int(config.stock_aggregate_tokens)

        self.aggregate_queries = nn.Parameter(torch.zeros(1, k, d))
        self.aggregate_query_pos = nn.Parameter(torch.zeros(1, k, d))

        self.aggregate_cross = CrossAttentionBlock(
            query_dim=d,
            kv_dim=d,
            num_heads=int(config.cross_num_heads),
            ff_dim=int(config.cross_ff_dim),
            dropout=float(config.dropout),
            activation=config.activation,
        )
        self.broadcast_cross = CrossAttentionBlock(
            query_dim=d,
            kv_dim=d,
            num_heads=int(config.cross_num_heads),
            ff_dim=int(config.cross_ff_dim),
            dropout=float(config.dropout),
            activation=config.activation,
        )

        self.broadcast_norm = nn.LayerNorm(d)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.aggregate_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.aggregate_query_pos, mean=0.0, std=0.02)

    def forward(
        self,
        stock_embeddings: torch.Tensor,
        stock_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if stock_embeddings.ndim != 3:
            raise ValueError(f"stock_embeddings must have shape [B,N,D], got {tuple(stock_embeddings.shape)}")

        b, n, d = stock_embeddings.shape
        key_padding_mask = None
        if stock_valid_mask is not None:
            if stock_valid_mask.shape != stock_embeddings.shape[:2]:
                raise ValueError("stock_valid_mask shape mismatch")
            key_padding_mask = ~stock_valid_mask.bool()

        q = self.aggregate_queries.expand(b, -1, -1) + self.aggregate_query_pos
        latent = self.aggregate_cross(
            q,
            stock_embeddings,
            key_padding_mask=key_padding_mask,
            residual_query=bool(self.config.aggregate_residual_query),
        )
        broadcast = self.broadcast_cross(
            stock_embeddings,
            latent,
            key_padding_mask=None,
            residual_query=bool(self.config.broadcast_residual_query),
        )

        # If broadcast_residual_query=True, broadcast already contains stock residual.
        # If False, this stays as a pure bottleneck output.
        return self.broadcast_norm(broadcast)


# ============================================================
# Score head
# ============================================================

class ScoreHead(nn.Module):
    def __init__(
        self,
        model_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "softsign",
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("score_head_layers must be >= 1")

        hidden = int(hidden_dim or model_dim)
        layers: list[nn.Module] = [nn.LayerNorm(int(model_dim))]

        if int(num_layers) == 1:
            layers.append(nn.Linear(int(model_dim), 1))
        else:
            in_dim = int(model_dim)
            for _ in range(int(num_layers) - 1):
                layers += [
                    nn.Linear(in_dim, hidden),
                    _activation(activation),
                    nn.Dropout(float(dropout)),
                ]
                in_dim = hidden
            layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ============================================================
# Full model
# ============================================================

class DualTransformerRanker(nn.Module):
    def __init__(self, config: DualTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.model_dim = config.stock_embedding_dim()

        self.temporal_encoder = DepthwiseThenMixedTemporalEncoder(config)

        effective_scalar_factors = int(config.num_scalar_factors) if bool(config.use_scalar_factors) else 0
        self.scalar_encoder = ScalarVectorTokenEncoder(
            effective_scalar_factors,
            config.token_dim(),
            float(config.dropout),
            config.activation,
        )

        self.intra_stock_compressor = IntraStockTokenCompressor(config)
        self.stock_aggregator = StockAggregationBroadcast(config)
        self.score_head = ScoreHead(
            config.stock_embedding_dim(),
            config.score_hidden_dim,
            int(config.score_head_layers),
            float(config.dropout),
            config.activation,
        )

    @classmethod
    def from_feature_counts(
        cls,
        num_ts_factors: int,
        num_scalar_factors: int,
        seq_len: int = 128,
        **kwargs: Any,
    ) -> "DualTransformerRanker":
        config = DualTransformerConfig(
            num_ts_factors=int(num_ts_factors),
            num_scalar_factors=int(num_scalar_factors),
            seq_len=int(seq_len),
            **kwargs,
        )
        return cls(config)

    def _ensure_batched(
        self,
        x_ts: torch.Tensor,
        x_scalar: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        was_unbatched = False

        if x_ts.ndim == 3:
            x_ts = x_ts.unsqueeze(0)
            was_unbatched = True
        elif x_ts.ndim != 4:
            raise ValueError(f"x_ts must have shape [B,N,F_ts,T] or [N,F_ts,T], got {tuple(x_ts.shape)}")

        if x_scalar is not None:
            if x_scalar.ndim == 2:
                x_scalar = x_scalar.unsqueeze(0)
            elif x_scalar.ndim != 3:
                raise ValueError(f"x_scalar must have shape [B,N,F_sc] or [N,F_sc], got {tuple(x_scalar.shape)}")

        return x_ts, x_scalar, was_unbatched

    def forward(
        self,
        x_ts: torch.Tensor,
        x_scalar: Optional[torch.Tensor] = None,
        stock_valid_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        x_ts, x_scalar, was_unbatched = self._ensure_batched(x_ts, x_scalar)

        b, n, f_ts, t = x_ts.shape
        if f_ts != int(self.config.num_ts_factors):
            raise ValueError(f"expected F_ts={self.config.num_ts_factors}, got {f_ts}")
        if t != int(self.config.seq_len):
            raise ValueError(f"expected T={self.config.seq_len}, got {t}")

        if stock_valid_mask is not None and stock_valid_mask.ndim == 1:
            stock_valid_mask = stock_valid_mask.unsqueeze(0)

        temporal_tokens = self.temporal_encoder(x_ts)
        scalar_token = self.scalar_encoder(x_scalar, b, n, x_ts.device)
        tokens = torch.cat([temporal_tokens, scalar_token], dim=2)

        reduced_tokens, stock_embedding = self.intra_stock_compressor(tokens)
        cross_embedding = self.stock_aggregator(stock_embedding, stock_valid_mask=stock_valid_mask)
        scores = self.score_head(cross_embedding)

        if stock_valid_mask is not None:
            scores = scores.masked_fill(~stock_valid_mask.bool(), 0.0)

        if was_unbatched and self.config.output_squeeze_batch_if_unbatched:
            scores_out = scores.squeeze(0)
            stock_out = stock_embedding.squeeze(0)
            cross_out = cross_embedding.squeeze(0)
            temporal_out = temporal_tokens.squeeze(0)
            reduced_out = reduced_tokens.squeeze(0)
        else:
            scores_out = scores
            stock_out = stock_embedding
            cross_out = cross_embedding
            temporal_out = temporal_tokens
            reduced_out = reduced_tokens

        if not return_dict:
            return scores_out

        return {
            "scores": scores_out,
            "stock_embedding": stock_out,
            "cross_embedding": cross_out,
            "temporal_tokens": temporal_out,
            "reduced_tokens": reduced_out,
        }

    def predict_scores(
        self,
        x_ts: torch.Tensor,
        x_scalar: Optional[torch.Tensor] = None,
        stock_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x_ts, x_scalar, stock_valid_mask, return_dict=False)

    def get_config_dict(self) -> dict[str, Any]:
        return asdict(self.config)


def make_tiny_model_for_test(
    num_ts_factors: int = 6,
    num_scalar_factors: int = 2,
    seq_len: int = 32,
) -> DualTransformerRanker:
    return DualTransformerRanker.from_feature_counts(
        num_ts_factors=num_ts_factors,
        num_scalar_factors=num_scalar_factors,
        seq_len=seq_len,
        depthwise_layers=2,
        temporal_compressed_len=16,
        mixed_channels_1=16,
        mixed_channels_2=8,
        temporal_out_channels=4,
        factor_reduce_tokens=2,
        factor_num_heads=4,
        factor_ff_dim=32,
        stock_aggregate_tokens=4,
        cross_num_heads=4,
        cross_ff_dim=64,
        score_hidden_dim=32,
        dropout=0.05,
        group_norm_groups=4,
        activation="softsign",
    )


__all__ = [
    "DualTransformerConfig",
    "DepthwiseThenMixedTemporalEncoder",
    "ScalarVectorTokenEncoder",
    "IntraStockTokenCompressor",
    "StockAggregationBroadcast",
    "ScoreHead",
    "DualTransformerRanker",
    "count_parameters",
    "make_tiny_model_for_test",
]
