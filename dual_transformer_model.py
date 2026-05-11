"""
dual_transformer_model.py

Dual-Transformer V4 for cross-sectional ranking.

Revision target
---------------
This model implements the 2026-05-10 redesign:
1. Temporal factors are first denoised independently by depthwise Conv1d.
2. The time axis is compressed from seq_len to temporal_compressed_len, default 64.
3. Cross-factor mixing starts only after denoising and pooling, then channels are
   reduced F_ts -> 64 -> 32 -> 16.
4. All scalar factors are mapped jointly to one length-64 vector.
5. The stock-local token matrix is [16 temporal tokens + 1 scalar token, 64].
6. One factor self-attention block with positional embedding is followed by
   cross-attention compression to 8 tokens.
7. The 8 x 64 tokens are flattened to a 512-dimensional stock embedding.
8. Stock-level interaction uses aggregation-broadcast cross-attention:
   N stocks -> K=32 latent market tokens -> N stocks, with residual and LayerNorm.

The public API remains compatible with earlier training scripts:
    DualTransformerRanker.from_feature_counts(...)
    model(x_ts, x_scalar)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
from torch import nn


@dataclass(frozen=True)
class DualTransformerConfig:
    num_ts_factors: int
    num_scalar_factors: int
    seq_len: int = 128

    # Independent per-factor temporal denoising.
    depthwise_layers: int = 3
    depthwise_kernel_size: int = 5

    # Time compression. User target: 128 -> 64.
    temporal_compressed_len: int = 64

    # Cross-factor mixing after depthwise denoising.
    mixed_channels_1: int = 64
    mixed_channels_2: int = 32
    temporal_out_channels: int = 16
    mixed_kernel_size: int = 5

    group_norm_groups: int = 32
    dropout: float = 0.1
    activation: str = "gelu"
    input_nan_to_num: bool = True

    # Intra-stock attention.
    factor_num_layers: int = 1
    factor_num_heads: int = 4
    factor_ff_dim: int = 256
    factor_reduce_tokens: int = 8
    factor_use_positional_encoding: bool = True

    # Stock aggregation-broadcast attention.
    stock_aggregate_tokens: int = 32
    cross_num_heads: int = 8
    cross_ff_dim: int = 1024

    # Score head.
    score_hidden_dim: Optional[int] = None
    score_head_layers: int = 2

    # Backward-compatible fields accepted from old JSON/presets.
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

    def scalar_token_count(self) -> int:
        return 1 if int(self.num_scalar_factors) > 0 else 0

    def factor_token_count(self) -> int:
        return int(self.temporal_out_channels) + self.scalar_token_count()

    def stock_embedding_dim(self) -> int:
        return int(self.factor_reduce_tokens) * self.token_dim()

    def resolved_model_dim(self) -> int:
        return self.stock_embedding_dim()


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
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
    # num_groups=num_channels gives one group per factor channel.
    return nn.GroupNorm(num_groups=int(num_channels), num_channels=int(num_channels))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float, activation: str = "gelu") -> None:
        super().__init__()
        if int(dim) % int(num_heads) != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.norm1 = nn.LayerNorm(int(dim))
        self.attn = nn.MultiheadAttention(int(dim), int(num_heads), dropout=float(dropout), batch_first=True)
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
        out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(out)
        x = x + self.ff(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float, activation: str = "gelu") -> None:
        super().__init__()
        if int(dim) % int(num_heads) != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.q_norm = nn.LayerNorm(self.dim)
        self.kv_norm = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(self.dim, int(num_heads), dropout=float(dropout), batch_first=True)
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
        x = query + self.drop(out) if residual_query else self.drop(out)
        x = x + self.ff(self.ff_norm(x))
        return x


class DepthwiseThenMixedTemporalEncoder(nn.Module):
    """[B,N,F_ts,T] -> [B,N,temporal_out_channels,temporal_compressed_len]."""

    def __init__(self, config: DualTransformerConfig) -> None:
        super().__init__()
        self.config = config
        f_ts = int(config.num_ts_factors)
        if f_ts <= 0:
            raise ValueError("num_ts_factors must be positive")

        dw_k = int(config.depthwise_kernel_size)
        dw_pad = dw_k // 2
        mix_k = int(config.mixed_kernel_size)
        mix_pad = mix_k // 2

        depthwise: list[nn.Module] = []
        for _ in range(int(config.depthwise_layers)):
            depthwise += [
                nn.Conv1d(f_ts, f_ts, kernel_size=dw_k, padding=dw_pad, groups=f_ts),
                _make_depthwise_norm(f_ts),
                _activation(config.activation),
                nn.Dropout(float(config.dropout)),
            ]
        self.depthwise_denoise = nn.Sequential(*depthwise)
        self.pool = nn.AdaptiveAvgPool1d(int(config.temporal_compressed_len))

        c1 = int(config.mixed_channels_1)
        c2 = int(config.mixed_channels_2)
        c3 = int(config.temporal_out_channels)
        groups = int(config.group_norm_groups)

        self.mixed_reduce = nn.Sequential(
            nn.Conv1d(f_ts, c1, kernel_size=mix_k, padding=mix_pad),
            _make_group_norm(c1, groups),
            _activation(config.activation),
            nn.Dropout(float(config.dropout)),
            nn.Conv1d(c1, c2, kernel_size=mix_k, padding=mix_pad),
            _make_group_norm(c2, groups),
            _activation(config.activation),
            nn.Dropout(float(config.dropout)),
            nn.Conv1d(c2, c3, kernel_size=mix_k, padding=mix_pad),
            _make_group_norm(c3, groups),
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
        return x.reshape(b, n, int(self.config.temporal_out_channels), int(self.config.temporal_compressed_len))


class ScalarVectorTokenEncoder(nn.Module):
    """[B,N,F_sc] -> [B,N,1,token_dim]."""

    def __init__(self, num_scalar_factors: int, token_dim: int, dropout: float = 0.1, activation: str = "gelu") -> None:
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

    def forward(self, x_scalar: Optional[torch.Tensor], batch_size: int, num_stocks: int, device: torch.device) -> torch.Tensor:
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


class IntraStockTokenCompressor(nn.Module):
    """[B,N,17,64] -> [B,N,8,64] and flattened [B,N,512]."""

    def __init__(self, config: DualTransformerConfig) -> None:
        super().__init__()
        self.config = config
        d = config.token_dim()
        f_total = config.factor_token_count()
        if f_total <= 0:
            raise ValueError("factor_token_count must be positive")

        if bool(config.factor_use_positional_encoding):
            self.source_pos_embedding = nn.Parameter(torch.zeros(1, f_total, d))
            self.query_pos_embedding = nn.Parameter(torch.zeros(1, int(config.factor_reduce_tokens), d))
        else:
            self.register_parameter("source_pos_embedding", None)
            self.register_parameter("query_pos_embedding", None)

        self.factor_self_blocks = nn.ModuleList([
            SelfAttentionBlock(d, int(config.factor_num_heads), int(config.factor_ff_dim), float(config.dropout), config.activation)
            for _ in range(int(config.factor_num_layers))
        ])
        self.reduce_queries = nn.Parameter(torch.zeros(1, int(config.factor_reduce_tokens), d))
        self.reduce_cross = CrossAttentionBlock(d, int(config.factor_num_heads), int(config.factor_ff_dim), float(config.dropout), config.activation)
        self.out_norm = nn.LayerNorm(d)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.reduce_queries, mean=0.0, std=0.02)
        if self.source_pos_embedding is not None:
            nn.init.normal_(self.source_pos_embedding, mean=0.0, std=0.02)
        if self.query_pos_embedding is not None:
            nn.init.normal_(self.query_pos_embedding, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 4:
            raise ValueError(f"tokens must have shape [B,N,F,D], got {tuple(tokens.shape)}")
        b, n, f_total, d = tokens.shape
        x = tokens.reshape(b * n, f_total, d)
        if self.source_pos_embedding is not None:
            if f_total > self.source_pos_embedding.shape[1]:
                raise ValueError("token count exceeds source positional embedding length")
            x = x + self.source_pos_embedding[:, :f_total, :]
        for block in self.factor_self_blocks:
            x = block(x)
        q = self.reduce_queries.expand(b * n, -1, -1)
        if self.query_pos_embedding is not None:
            q = q + self.query_pos_embedding
        reduced = self.reduce_cross(q, x, residual_query=True)
        reduced = self.out_norm(reduced)
        reduced = reduced.reshape(b, n, int(self.config.factor_reduce_tokens), d)
        return reduced, reduced.flatten(start_dim=2)


class StockAggregationBroadcast(nn.Module):
    """[B,N,512] -> aggregate K market tokens -> broadcast back to [B,N,512]."""

    def __init__(self, config: DualTransformerConfig) -> None:
        super().__init__()
        d = config.stock_embedding_dim()
        k = int(config.stock_aggregate_tokens)
        self.aggregate_queries = nn.Parameter(torch.zeros(1, k, d))
        self.aggregate_query_pos = nn.Parameter(torch.zeros(1, k, d))
        self.aggregate_cross = CrossAttentionBlock(d, int(config.cross_num_heads), int(config.cross_ff_dim), float(config.dropout), config.activation)
        self.broadcast_cross = CrossAttentionBlock(d, int(config.cross_num_heads), int(config.cross_ff_dim), float(config.dropout), config.activation)
        self.broadcast_norm = nn.LayerNorm(d)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.aggregate_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.aggregate_query_pos, mean=0.0, std=0.02)

    def forward(self, stock_embeddings: torch.Tensor, stock_valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if stock_embeddings.ndim != 3:
            raise ValueError(f"stock_embeddings must have shape [B,N,D], got {tuple(stock_embeddings.shape)}")
        b, n, d = stock_embeddings.shape
        key_padding_mask = None
        if stock_valid_mask is not None:
            if stock_valid_mask.shape != stock_embeddings.shape[:2]:
                raise ValueError("stock_valid_mask shape mismatch")
            key_padding_mask = ~stock_valid_mask.bool()
        q = self.aggregate_queries.expand(b, -1, -1) + self.aggregate_query_pos
        latent = self.aggregate_cross(q, stock_embeddings, key_padding_mask=key_padding_mask, residual_query=True)

        # Decoder-style broadcast:
        # query = each stock's own embedding; key/value = aggregated market latent tokens.
        #
        # residual_query=True makes broadcast_cross internally do:
        #   x = stock_embeddings + cross_attn(stock_embeddings, latent)
        #   x = x + FFN(x)
        # so the FFN sees the fused "own stock feature + market context".
        #
        # With residual_query=False, the FFN only sees the pure broadcasted macro
        # feature, and the original stock feature is added back only after FFN.
        broadcast = self.broadcast_cross(stock_embeddings, latent, key_padding_mask=None, residual_query=True)

        # Do not add stock_embeddings again here; broadcast already contains the
        # residual stock feature inside broadcast_cross.
        return self.broadcast_norm(broadcast)


class ScoreHead(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: Optional[int] = None, num_layers: int = 2, dropout: float = 0.1, activation: str = "gelu") -> None:
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
                layers += [nn.Linear(in_dim, hidden), _activation(activation), nn.Dropout(float(dropout))]
                in_dim = hidden
            layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DualTransformerRanker(nn.Module):
    def __init__(self, config: DualTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.model_dim = config.stock_embedding_dim()
        self.temporal_encoder = DepthwiseThenMixedTemporalEncoder(config)
        self.scalar_encoder = ScalarVectorTokenEncoder(int(config.num_scalar_factors), config.token_dim(), float(config.dropout), config.activation)
        self.intra_stock_compressor = IntraStockTokenCompressor(config)
        self.stock_aggregator = StockAggregationBroadcast(config)
        self.score_head = ScoreHead(config.stock_embedding_dim(), config.score_hidden_dim, int(config.score_head_layers), float(config.dropout), config.activation)

    @classmethod
    def from_feature_counts(cls, num_ts_factors: int, num_scalar_factors: int, seq_len: int = 128, **kwargs: Any) -> "DualTransformerRanker":
        config = DualTransformerConfig(int(num_ts_factors), int(num_scalar_factors), int(seq_len), **kwargs)
        return cls(config)

    def _ensure_batched(self, x_ts: torch.Tensor, x_scalar: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor], bool]:
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

    def forward(self, x_ts: torch.Tensor, x_scalar: Optional[torch.Tensor] = None, stock_valid_mask: Optional[torch.Tensor] = None, return_dict: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
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
        return {"scores": scores_out, "stock_embedding": stock_out, "cross_embedding": cross_out, "temporal_tokens": temporal_out, "reduced_tokens": reduced_out}

    def predict_scores(self, x_ts: torch.Tensor, x_scalar: Optional[torch.Tensor] = None, stock_valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x_ts, x_scalar, stock_valid_mask, return_dict=False)

    def get_config_dict(self) -> dict[str, Any]:
        return asdict(self.config)


def make_tiny_model_for_test(num_ts_factors: int = 6, num_scalar_factors: int = 2, seq_len: int = 32) -> DualTransformerRanker:
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
