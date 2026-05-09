"""
dual_transformer_model_v2.py

Dual-Transformer V2 for cross-sectional ranking.

Changes vs V1:
1. CNN no longer flattens [stock, factor]. Stocks are batch items and time-series
   factors are Conv1d input channels: [B,N,F_ts,T] -> [B*N,F_ts,T].
2. BatchNorm is removed; CNN uses GroupNorm.
3. CNN channels are hidden temporal-factor tokens: [B,N,C_out,T_comp].
4. T_comp is the raw token encoding. Optional Linear(T_comp -> model_dim) is
   used when model_dim is set.
5. Factor-level attention uses learnable positional embeddings.
6. Cross-sectional stock attention remains permutation-equivariant, no position.
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

    # V2 temporal CNN. Defaults follow the user's proposed structure.
    temporal_hidden_channels: int = 512
    temporal_out_channels: int = 256
    temporal_compressed_len: int = 32
    temporal_kernel_size: int = 5
    temporal_conv_layers: int = 2
    group_norm_groups: int = 32

    # Deprecated V1 compatibility. Accepted to avoid breaking old config files.
    temporal_channels: Optional[int] = None

    # If None, attention dim D = temporal_compressed_len.
    # If set, raw T_comp token encodings are projected to model_dim.
    model_dim: Optional[int] = None

    factor_num_layers: int = 1
    factor_num_heads: int = 4
    factor_ff_dim: int = 256
    factor_use_positional_encoding: bool = True

    cross_num_layers: int = 1
    cross_num_heads: int = 4
    cross_ff_dim: int = 256

    dropout: float = 0.1
    activation: str = "gelu"
    norm_first: bool = True

    score_hidden_dim: Optional[int] = None
    score_head_layers: int = 2

    input_nan_to_num: bool = True
    output_squeeze_batch_if_unbatched: bool = True

    def resolved_model_dim(self) -> int:
        return int(self.temporal_compressed_len if self.model_dim is None else self.model_dim)

    def factor_token_count(self) -> int:
        return int(self.temporal_out_channels) + int(self.num_scalar_factors)


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


def _make_encoder_layer(
    model_dim: int,
    num_heads: int,
    ff_dim: int,
    dropout: float,
    activation: str,
    norm_first: bool,
) -> nn.TransformerEncoderLayer:
    if int(model_dim) % int(num_heads) != 0:
        raise ValueError(f"model_dim={model_dim} must be divisible by num_heads={num_heads}")
    return nn.TransformerEncoderLayer(
        d_model=int(model_dim),
        nhead=int(num_heads),
        dim_feedforward=int(ff_dim),
        dropout=float(dropout),
        activation=str(activation),
        batch_first=True,
        norm_first=bool(norm_first),
    )


def _make_transformer_encoder(layer: nn.TransformerEncoderLayer, num_layers: int) -> nn.TransformerEncoder:
    try:
        return nn.TransformerEncoder(layer, num_layers=int(num_layers), enable_nested_tensor=False)
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=int(num_layers))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


class CrossFactorTemporalEncoder(nn.Module):
    """
    Treat factors as Conv1d input channels and stocks as batch items.

    Input:  x_ts [B,N,F_ts,T]
    Output: raw tokens [B,N,C_out,T_comp]
    """

    def __init__(self, config: DualTransformerConfig) -> None:
        super().__init__()
        self.config = config
        if config.num_ts_factors <= 0:
            raise ValueError("num_ts_factors must be positive")
        if config.temporal_hidden_channels <= 0 or config.temporal_out_channels <= 0:
            raise ValueError("temporal channels must be positive")
        if config.temporal_compressed_len <= 0:
            raise ValueError("temporal_compressed_len must be positive")
        if config.temporal_conv_layers < 1:
            raise ValueError("temporal_conv_layers must be >= 1")

        k = int(config.temporal_kernel_size)
        pad = k // 2
        hidden = int(config.temporal_hidden_channels)
        out_ch = int(config.temporal_out_channels)
        groups = int(config.group_norm_groups)

        layers: list[nn.Module] = [
            nn.Conv1d(int(config.num_ts_factors), hidden, kernel_size=k, padding=pad),
            _make_group_norm(hidden, groups),
            _activation(config.activation),
            nn.Dropout(float(config.dropout)),
        ]
        for _ in range(int(config.temporal_conv_layers) - 1):
            layers += [
                nn.Conv1d(hidden, hidden, kernel_size=k, padding=pad),
                _make_group_norm(hidden, groups),
                _activation(config.activation),
                nn.Dropout(float(config.dropout)),
            ]
        layers += [
            nn.Conv1d(hidden, out_ch, kernel_size=1),
            _make_group_norm(out_ch, groups),
            _activation(config.activation),
            nn.Dropout(float(config.dropout)),
        ]
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(int(config.temporal_compressed_len))

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
        x = self.pool(self.conv(x))
        return x.reshape(b, n, int(self.config.temporal_out_channels), int(self.config.temporal_compressed_len))


class TemporalTokenProjection(nn.Module):
    """Project raw T_comp token encodings to model_dim if needed."""

    def __init__(self, compressed_len: int, model_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Identity() if int(compressed_len) == int(model_dim) else nn.Linear(int(compressed_len), int(model_dim))
        self.norm = nn.LayerNorm(int(model_dim))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, raw_tokens: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(self.proj(raw_tokens)))


class ScalarFactorEmbedding(nn.Module):
    """Embed scalar factors as scalar tokens aligned to model_dim."""

    def __init__(self, num_scalar_factors: int, model_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_scalar_factors = int(num_scalar_factors)
        self.model_dim = int(model_dim)
        if self.num_scalar_factors > 0:
            self.weight = nn.Parameter(torch.empty(self.num_scalar_factors, self.model_dim))
            self.bias = nn.Parameter(torch.zeros(self.num_scalar_factors, self.model_dim))
            self.norm = nn.LayerNorm(self.model_dim)
            self.dropout = nn.Dropout(float(dropout))
            self.reset_parameters()
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            self.norm = nn.Identity()
            self.dropout = nn.Identity()

    def reset_parameters(self) -> None:
        if self.num_scalar_factors > 0:
            nn.init.xavier_uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x_scalar: Optional[torch.Tensor], batch_size: int, num_stocks: int, device: torch.device) -> torch.Tensor:
        if self.num_scalar_factors == 0:
            return torch.empty(batch_size, num_stocks, 0, self.model_dim, device=device)
        if x_scalar is None:
            raise ValueError("x_scalar is required because num_scalar_factors > 0")
        if x_scalar.ndim != 3:
            raise ValueError(f"x_scalar must have shape [B,N,F_sc], got {tuple(x_scalar.shape)}")
        b, n, f_sc = x_scalar.shape
        if b != batch_size or n != num_stocks or f_sc != self.num_scalar_factors:
            raise ValueError(f"x_scalar shape mismatch, got {tuple(x_scalar.shape)}")
        x = torch.nan_to_num(x_scalar, nan=0.0, posinf=0.0, neginf=0.0)
        tokens = x.unsqueeze(-1) * self.weight.view(1, 1, f_sc, self.model_dim)
        tokens = tokens + self.bias.view(1, 1, f_sc, self.model_dim)
        return self.dropout(self.norm(tokens))


class FactorLevelTransformer(nn.Module):
    """Factor attention inside each stock, with optional positional embeddings."""

    def __init__(self, config: DualTransformerConfig) -> None:
        super().__init__()
        self.config = config
        d = config.resolved_model_dim()
        token_count = 1 + config.factor_token_count()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        if config.factor_use_positional_encoding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, token_count, d))
        else:
            self.register_parameter("pos_embedding", None)
        layer = _make_encoder_layer(d, int(config.factor_num_heads), int(config.factor_ff_dim), float(config.dropout), config.activation, bool(config.norm_first))
        self.encoder = _make_transformer_encoder(layer, int(config.factor_num_layers))
        self.out_norm = nn.LayerNorm(d)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        if self.pos_embedding is not None:
            nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, factor_tokens: torch.Tensor) -> torch.Tensor:
        if factor_tokens.ndim != 4:
            raise ValueError(f"factor_tokens must have shape [B,N,F,D], got {tuple(factor_tokens.shape)}")
        b, n, f_total, d = factor_tokens.shape
        x = factor_tokens.reshape(b * n, f_total, d)
        cls = self.cls_token.expand(b * n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if self.pos_embedding is not None:
            if x.shape[1] > self.pos_embedding.shape[1]:
                raise ValueError("token count exceeds positional embedding length")
            x = x + self.pos_embedding[:, : x.shape[1], :]
        x = self.encoder(x)
        return self.out_norm(x[:, 0, :]).reshape(b, n, d)


class CrossSectionalTransformer(nn.Module):
    """Stock-level cross-sectional attention, without positional encoding."""

    def __init__(self, config: DualTransformerConfig) -> None:
        super().__init__()
        d = config.resolved_model_dim()
        layer = _make_encoder_layer(d, int(config.cross_num_heads), int(config.cross_ff_dim), float(config.dropout), config.activation, bool(config.norm_first))
        self.encoder = _make_transformer_encoder(layer, int(config.cross_num_layers))
        self.out_norm = nn.LayerNorm(d)

    def forward(self, stock_embeddings: torch.Tensor, stock_valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if stock_embeddings.ndim != 3:
            raise ValueError(f"stock_embeddings must have shape [B,N,D], got {tuple(stock_embeddings.shape)}")
        key_padding_mask = None
        if stock_valid_mask is not None:
            if stock_valid_mask.shape != stock_embeddings.shape[:2]:
                raise ValueError("stock_valid_mask shape mismatch")
            key_padding_mask = ~stock_valid_mask.bool()
        return self.out_norm(self.encoder(stock_embeddings, src_key_padding_mask=key_padding_mask))


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
        self.model_dim = config.resolved_model_dim()
        self.temporal_encoder = CrossFactorTemporalEncoder(config)
        self.temporal_token_projection = TemporalTokenProjection(config.temporal_compressed_len, self.model_dim, config.dropout)
        self.scalar_embedding = ScalarFactorEmbedding(config.num_scalar_factors, self.model_dim, config.dropout)
        self.factor_transformer = FactorLevelTransformer(config)
        self.cross_transformer = CrossSectionalTransformer(config)
        self.score_head = ScoreHead(self.model_dim, config.score_hidden_dim, config.score_head_layers, config.dropout, config.activation)

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

        raw_ts_tokens = self.temporal_encoder(x_ts)
        ts_tokens = self.temporal_token_projection(raw_ts_tokens)
        scalar_tokens = self.scalar_embedding(x_scalar, b, n, x_ts.device)
        factor_tokens = torch.cat([ts_tokens, scalar_tokens], dim=2)
        stock_embedding = self.factor_transformer(factor_tokens)
        cross_embedding = self.cross_transformer(stock_embedding, stock_valid_mask)
        scores = self.score_head(cross_embedding)
        if stock_valid_mask is not None:
            scores = scores.masked_fill(~stock_valid_mask.bool(), 0.0)

        if was_unbatched and self.config.output_squeeze_batch_if_unbatched:
            scores_out = scores.squeeze(0)
            stock_out = stock_embedding.squeeze(0)
            cross_out = cross_embedding.squeeze(0)
            raw_out = raw_ts_tokens.squeeze(0)
        else:
            scores_out = scores
            stock_out = stock_embedding
            cross_out = cross_embedding
            raw_out = raw_ts_tokens
        if not return_dict:
            return scores_out
        return {"scores": scores_out, "stock_embedding": stock_out, "cross_embedding": cross_out, "raw_ts_tokens": raw_out}

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
        temporal_hidden_channels=32,
        temporal_out_channels=8,
        temporal_compressed_len=8,
        group_norm_groups=4,
        model_dim=32,
        factor_num_layers=1,
        factor_num_heads=4,
        factor_ff_dim=64,
        cross_num_layers=1,
        cross_num_heads=4,
        cross_ff_dim=64,
        score_hidden_dim=32,
        dropout=0.05,
    )


__all__ = [
    "DualTransformerConfig",
    "CrossFactorTemporalEncoder",
    "TemporalTokenProjection",
    "ScalarFactorEmbedding",
    "FactorLevelTransformer",
    "CrossSectionalTransformer",
    "ScoreHead",
    "DualTransformerRanker",
    "count_parameters",
    "make_tiny_model_for_test",
]
