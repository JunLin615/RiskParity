"""
dual_transformer_model.py

Dual-Transformer architecture for cross-sectional ranking.

Architecture
------------
Input tensors from rank_dataset.py:

    x_ts:     [B, N, F_ts, T] or [N, F_ts, T]
    x_scalar: [B, N, F_sc]    or [N, F_sc]

where:
    B    = outer DataLoader batch size
    N    = number of stocks in one cross-section, e.g. 512
    F_ts = number of time-series factors
    F_sc = number of scalar factors
    T    = sequence length, e.g. 128

Main stages:
1. Temporal factor encoder:
    [B, N, F_ts, T]
        -> reshape [B*N*F_ts, 1, T]
        -> 1D conv + adaptive pooling
        -> [B, N, F_ts, D]

2. Scalar factor embedding:
    [B, N, F_sc]
        -> independent scalar token embeddings
        -> [B, N, F_sc, D]

3. Factor-level transformer:
    concat time-series and scalar factor tokens:
        [B, N, F_ts + F_sc, D]
    add CLS token per stock:
        [B*N, 1 + F_ts + F_sc, D]
    no positional encoding
    output stock fingerprint:
        [B, N, D]

4. Cross-sectional transformer:
    [B, N, D]
    no positional encoding
    output contextualized stock embedding:
        [B, N, D]

5. Score head:
    [B, N, D] -> [B, N]

Notes
-----
This file intentionally does not implement the ranking loss. Put losses such as
Spearman/SoftSort/Pairwise loss in rank_loss.py.
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
    """Model hyperparameters."""

    num_ts_factors: int
    num_scalar_factors: int
    seq_len: int = 128

    temporal_channels: int = 16
    temporal_compressed_len: int = 32
    temporal_kernel_size: int = 5
    temporal_conv_layers: int = 2

    model_dim: Optional[int] = None

    factor_num_layers: int = 1
    factor_num_heads: int = 8
    factor_ff_dim: int = 1024

    cross_num_layers: int = 1
    cross_num_heads: int = 8
    cross_ff_dim: int = 1024

    dropout: float = 0.1
    activation: str = "gelu"
    norm_first: bool = True

    score_hidden_dim: Optional[int] = None
    score_head_layers: int = 2

    input_nan_to_num: bool = True
    output_squeeze_batch_if_unbatched: bool = True

    def resolved_model_dim(self) -> int:
        """Return D. By default D = temporal_channels * temporal_compressed_len."""
        if self.model_dim is not None:
            return int(self.model_dim)
        return int(self.temporal_channels) * int(self.temporal_compressed_len)

    def temporal_flat_dim(self) -> int:
        """Return temporal encoder flat dimension before optional projection."""
        return int(self.temporal_channels) * int(self.temporal_compressed_len)


# ============================================================
# Small utilities
# ============================================================

def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    raise ValueError(f"unsupported activation: {name!r}")


def _validate_positive(name: str, value: int) -> None:
    if int(value) < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def _make_encoder_layer(
    model_dim: int,
    num_heads: int,
    ff_dim: int,
    dropout: float,
    activation: str,
    norm_first: bool,
) -> nn.TransformerEncoderLayer:
    if model_dim % num_heads != 0:
        raise ValueError(f"model_dim={model_dim} must be divisible by num_heads={num_heads}")

    return nn.TransformerEncoderLayer(
        d_model=model_dim,
        nhead=num_heads,
        dim_feedforward=ff_dim,
        dropout=dropout,
        activation=activation,
        batch_first=True,
        norm_first=norm_first,
    )


def _make_transformer_encoder(
    encoder_layer: nn.TransformerEncoderLayer,
    num_layers: int,
) -> nn.TransformerEncoder:
    """
    Construct TransformerEncoder while avoiding nested-tensor warnings in
    PyTorch versions that support enable_nested_tensor.
    """
    try:
        return nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )
    except TypeError:
        return nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))


# ============================================================
# Temporal encoder
# ============================================================

class TemporalFactorEncoder(nn.Module):
    """
    Encode each time-series factor independently.

    Input:
        x_ts: [B, N, F_ts, T]

    Output:
        tokens: [B, N, F_ts, D]
    """

    def __init__(self, config: DualTransformerConfig) -> None:
        super().__init__()

        _validate_positive("num_ts_factors", config.num_ts_factors)
        if config.num_ts_factors <= 0:
            raise ValueError("num_ts_factors must be positive for TemporalFactorEncoder")
        if config.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if config.temporal_channels <= 0:
            raise ValueError("temporal_channels must be positive")
        if config.temporal_compressed_len <= 0:
            raise ValueError("temporal_compressed_len must be positive")
        if config.temporal_conv_layers <= 0:
            raise ValueError("temporal_conv_layers must be positive")

        self.config = config
        c = int(config.temporal_channels)
        k = int(config.temporal_kernel_size)
        padding = k // 2

        layers: list[nn.Module] = []

        # First layer maps one single-factor channel to C channels.
        layers.extend([
            nn.Conv1d(1, c, kernel_size=k, padding=padding),
            nn.BatchNorm1d(c),
            _activation(config.activation),
            nn.Dropout(config.dropout),
        ])

        # Additional depthwise-separable style blocks.
        for _ in range(int(config.temporal_conv_layers) - 1):
            layers.extend([
                nn.Conv1d(c, c, kernel_size=k, padding=padding, groups=c),
                nn.Conv1d(c, c, kernel_size=1),
                nn.BatchNorm1d(c),
                _activation(config.activation),
                nn.Dropout(config.dropout),
            ])

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(int(config.temporal_compressed_len))

        flat_dim = config.temporal_flat_dim()
        model_dim = config.resolved_model_dim()

        if flat_dim == model_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(flat_dim, model_dim)

        self.out_norm = nn.LayerNorm(model_dim)

    def forward(self, x_ts: torch.Tensor) -> torch.Tensor:
        if x_ts.ndim != 4:
            raise ValueError(f"x_ts must have shape [B, N, F_ts, T], got {tuple(x_ts.shape)}")

        b, n, f_ts, t = x_ts.shape
        if f_ts != self.config.num_ts_factors:
            raise ValueError(f"expected F_ts={self.config.num_ts_factors}, got {f_ts}")
        if t != self.config.seq_len:
            raise ValueError(f"expected T={self.config.seq_len}, got {t}")

        if self.config.input_nan_to_num:
            x_ts = torch.nan_to_num(x_ts, nan=0.0, posinf=0.0, neginf=0.0)

        x = x_ts.reshape(b * n * f_ts, 1, t)
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.proj(x)
        x = self.out_norm(x)
        return x.reshape(b, n, f_ts, -1)


# ============================================================
# Scalar embedding
# ============================================================

class ScalarFactorEmbedding(nn.Module):
    """
    Independently embed scalar factors into factor tokens.

    For each scalar factor j:
        token_j = value_j * weight_j + bias_j

    Input:
        x_scalar: [B, N, F_sc]

    Output:
        tokens: [B, N, F_sc, D]
    """

    def __init__(self, num_scalar_factors: int, model_dim: int, dropout: float = 0.1) -> None:
        super().__init__()

        _validate_positive("num_scalar_factors", num_scalar_factors)
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")

        self.num_scalar_factors = int(num_scalar_factors)
        self.model_dim = int(model_dim)

        if self.num_scalar_factors > 0:
            self.weight = nn.Parameter(torch.empty(self.num_scalar_factors, self.model_dim))
            self.bias = nn.Parameter(torch.zeros(self.num_scalar_factors, self.model_dim))
            self.norm = nn.LayerNorm(self.model_dim)
            self.dropout = nn.Dropout(dropout)
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
            raise ValueError(f"x_scalar must have shape [B, N, F_sc], got {tuple(x_scalar.shape)}")

        b, n, f_sc = x_scalar.shape
        if b != batch_size or n != num_stocks:
            raise ValueError(
                f"x_scalar batch/stock dims {tuple(x_scalar.shape[:2])} do not match "
                f"x_ts dims {(batch_size, num_stocks)}"
            )
        if f_sc != self.num_scalar_factors:
            raise ValueError(f"expected F_sc={self.num_scalar_factors}, got {f_sc}")

        x = torch.nan_to_num(x_scalar, nan=0.0, posinf=0.0, neginf=0.0)
        tokens = x.unsqueeze(-1) * self.weight.view(1, 1, f_sc, self.model_dim)
        tokens = tokens + self.bias.view(1, 1, f_sc, self.model_dim)
        tokens = self.norm(tokens)
        tokens = self.dropout(tokens)
        return tokens


# ============================================================
# Factor-level attention
# ============================================================

class FactorLevelTransformer(nn.Module):
    """
    Factor-level self-attention inside each stock.

    Input:
        factor_tokens: [B, N, F_total, D]

    Output:
        stock_embedding: [B, N, D]
    """

    def __init__(self, config: DualTransformerConfig) -> None:
        super().__init__()
        model_dim = config.resolved_model_dim()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        encoder_layer = _make_encoder_layer(
            model_dim=model_dim,
            num_heads=int(config.factor_num_heads),
            ff_dim=int(config.factor_ff_dim),
            dropout=float(config.dropout),
            activation=config.activation,
            norm_first=bool(config.norm_first),
        )
        self.encoder = _make_transformer_encoder(encoder_layer, num_layers=int(config.factor_num_layers))
        self.out_norm = nn.LayerNorm(model_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, factor_tokens: torch.Tensor) -> torch.Tensor:
        if factor_tokens.ndim != 4:
            raise ValueError(f"factor_tokens must have shape [B, N, F, D], got {tuple(factor_tokens.shape)}")

        b, n, f_total, d = factor_tokens.shape
        x = factor_tokens.reshape(b * n, f_total, d)

        cls = self.cls_token.expand(b * n, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.encoder(x)
        stock_embedding = x[:, 0, :]
        stock_embedding = self.out_norm(stock_embedding)
        return stock_embedding.reshape(b, n, d)


# ============================================================
# Cross-sectional attention
# ============================================================

class CrossSectionalTransformer(nn.Module):
    """
    Cross-sectional self-attention among stocks.

    Input:
        stock_embeddings: [B, N, D]

    Optional:
        stock_valid_mask: [B, N], True for valid stocks, False for padded/invalid.

    Output:
        contextualized: [B, N, D]
    """

    def __init__(self, config: DualTransformerConfig) -> None:
        super().__init__()
        model_dim = config.resolved_model_dim()

        encoder_layer = _make_encoder_layer(
            model_dim=model_dim,
            num_heads=int(config.cross_num_heads),
            ff_dim=int(config.cross_ff_dim),
            dropout=float(config.dropout),
            activation=config.activation,
            norm_first=bool(config.norm_first),
        )
        self.encoder = _make_transformer_encoder(encoder_layer, num_layers=int(config.cross_num_layers))
        self.out_norm = nn.LayerNorm(model_dim)

    def forward(self, stock_embeddings: torch.Tensor, stock_valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if stock_embeddings.ndim != 3:
            raise ValueError(
                f"stock_embeddings must have shape [B, N, D], got {tuple(stock_embeddings.shape)}"
            )

        src_key_padding_mask = None
        if stock_valid_mask is not None:
            if stock_valid_mask.shape != stock_embeddings.shape[:2]:
                raise ValueError(
                    f"stock_valid_mask shape {tuple(stock_valid_mask.shape)} does not match "
                    f"[B, N]={tuple(stock_embeddings.shape[:2])}"
                )
            # PyTorch expects True for positions to ignore.
            src_key_padding_mask = ~stock_valid_mask.bool()

        x = self.encoder(stock_embeddings, src_key_padding_mask=src_key_padding_mask)
        x = self.out_norm(x)
        return x


# ============================================================
# Score head
# ============================================================

class ScoreHead(nn.Module):
    """Map contextualized stock embeddings [B, N, D] to scores [B, N]."""

    def __init__(
        self,
        model_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("score_head_layers must be >= 1")

        hidden_dim = int(hidden_dim or model_dim)

        layers: list[nn.Module] = [nn.LayerNorm(model_dim)]

        if num_layers == 1:
            layers.append(nn.Linear(model_dim, 1))
        else:
            in_dim = model_dim
            for _ in range(num_layers - 1):
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    _activation(activation),
                    nn.Dropout(dropout),
                ])
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.net(x).squeeze(-1)
        return score


# ============================================================
# Full model
# ============================================================

class DualTransformerRanker(nn.Module):
    """
    Dual-Transformer ranker.

    Forward input:
        x_ts:     [B, N, F_ts, T] or [N, F_ts, T]
        x_scalar: [B, N, F_sc]    or [N, F_sc]

    Forward output:
        scores: [B, N] or [N] if input was unbatched and squeeze is enabled.
    """

    def __init__(self, config: DualTransformerConfig) -> None:
        super().__init__()

        if config.num_ts_factors <= 0:
            raise ValueError("num_ts_factors must be positive")
        if config.num_scalar_factors < 0:
            raise ValueError("num_scalar_factors must be non-negative")

        self.config = config
        self.model_dim = config.resolved_model_dim()

        self.temporal_encoder = TemporalFactorEncoder(config)
        self.scalar_embedding = ScalarFactorEmbedding(
            num_scalar_factors=int(config.num_scalar_factors),
            model_dim=self.model_dim,
            dropout=float(config.dropout),
        )
        self.factor_transformer = FactorLevelTransformer(config)
        self.cross_transformer = CrossSectionalTransformer(config)
        self.score_head = ScoreHead(
            model_dim=self.model_dim,
            hidden_dim=config.score_hidden_dim,
            num_layers=int(config.score_head_layers),
            dropout=float(config.dropout),
            activation=config.activation,
        )

    @classmethod
    def from_feature_counts(
        cls,
        num_ts_factors: int,
        num_scalar_factors: int,
        seq_len: int = 128,
        **kwargs: Any,
    ) -> "DualTransformerRanker":
        """Convenience constructor from feature counts."""
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
        """Convert unbatched inputs to batched inputs."""
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
                raise ValueError(
                    f"x_scalar must have shape [B,N,F_sc] or [N,F_sc], got {tuple(x_scalar.shape)}"
                )

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
        if f_ts != self.config.num_ts_factors:
            raise ValueError(f"expected F_ts={self.config.num_ts_factors}, got {f_ts}")
        if t != self.config.seq_len:
            raise ValueError(f"expected T={self.config.seq_len}, got {t}")

        if stock_valid_mask is not None and stock_valid_mask.ndim == 1:
            stock_valid_mask = stock_valid_mask.unsqueeze(0)

        ts_tokens = self.temporal_encoder(x_ts)
        scalar_tokens = self.scalar_embedding(
            x_scalar=x_scalar,
            batch_size=b,
            num_stocks=n,
            device=x_ts.device,
        )

        factor_tokens = torch.cat([ts_tokens, scalar_tokens], dim=2)
        stock_embedding = self.factor_transformer(factor_tokens)
        cross_embedding = self.cross_transformer(stock_embedding, stock_valid_mask=stock_valid_mask)
        scores = self.score_head(cross_embedding)

        if stock_valid_mask is not None:
            scores = scores.masked_fill(~stock_valid_mask.bool(), 0.0)

        if was_unbatched and self.config.output_squeeze_batch_if_unbatched:
            scores_out = scores.squeeze(0)
            stock_embedding_out = stock_embedding.squeeze(0)
            cross_embedding_out = cross_embedding.squeeze(0)
        else:
            scores_out = scores
            stock_embedding_out = stock_embedding
            cross_embedding_out = cross_embedding

        if not return_dict:
            return scores_out

        return {
            "scores": scores_out,
            "stock_embedding": stock_embedding_out,
            "cross_embedding": cross_embedding_out,
        }

    def predict_scores(
        self,
        x_ts: torch.Tensor,
        x_scalar: Optional[torch.Tensor] = None,
        stock_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference helper."""
        self.eval()
        with torch.no_grad():
            return self.forward(x_ts=x_ts, x_scalar=x_scalar, stock_valid_mask=stock_valid_mask, return_dict=False)

    def get_config_dict(self) -> dict[str, Any]:
        """Return config as a plain dictionary."""
        return asdict(self.config)


# ============================================================
# Smoke-test helper
# ============================================================

def make_tiny_model_for_test(
    num_ts_factors: int = 6,
    num_scalar_factors: int = 2,
    seq_len: int = 32,
) -> DualTransformerRanker:
    """Create a small model for quick tests."""
    return DualTransformerRanker.from_feature_counts(
        num_ts_factors=num_ts_factors,
        num_scalar_factors=num_scalar_factors,
        seq_len=seq_len,
        temporal_channels=4,
        temporal_compressed_len=8,
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
    "TemporalFactorEncoder",
    "ScalarFactorEmbedding",
    "FactorLevelTransformer",
    "CrossSectionalTransformer",
    "ScoreHead",
    "DualTransformerRanker",
    "count_parameters",
    "make_tiny_model_for_test",
]
