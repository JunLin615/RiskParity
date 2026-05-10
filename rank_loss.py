"""
rank_loss.py

Ranking losses and metrics for the Dual-Transformer cross-sectional ranker.

This module intentionally avoids optional dependencies such as torchsort.
It provides:
1. Pearson correlation loss
2. Spearman-style soft-rank correlation loss
3. Pairwise logistic ranking loss
4. RankIC / IC metrics
5. Temperature scheduler for soft-rank losses

Tensor convention
-----------------
pred:
    [B, N] or [N], model scores. Higher score means more preferred.

target:
    [B, N] or [N].
    Usually local rank_pct from rank_dataset.py:
        lowest future return -> close to 1/N
        highest future return -> 1.0

mask:
    Optional bool tensor with same shape as pred/target.
    True means valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


EPS = 1e-8


@dataclass(frozen=True)
class RankLossConfig:
    """Ranking loss configuration."""

    loss_type: str = "spearman"
    temperature: float = 1.0
    eps: float = EPS
    pairwise_margin: float = 0.0
    pairwise_max_pairs: Optional[int] = None

    # NDCG-style top-heavy pairwise loss options.
    ndcg_temperature: float = 1.0
    ndcg_gain_power: float = 1.0
    ndcg_max_pairs: Optional[int] = None


@dataclass(frozen=True)
class TemperatureSchedule:
    """
    Exponential temperature schedule.

    Example:
        tau = schedule(epoch)
    """

    start: float = 1.0
    end: float = 0.1
    decay_epochs: int = 50

    def __call__(self, epoch: int) -> float:
        if self.decay_epochs <= 0:
            return float(self.end)
        t = min(max(float(epoch), 0.0), float(self.decay_epochs))
        ratio = t / float(self.decay_epochs)
        return float(self.start * ((self.end / self.start) ** ratio))


def _ensure_2d(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return [B, N] tensor and whether input was originally 1D."""
    if x.ndim == 1:
        return x.unsqueeze(0), True
    if x.ndim == 2:
        return x, False
    raise ValueError(f"expected tensor with shape [N] or [B,N], got {tuple(x.shape)}")


def _restore_dim(x: torch.Tensor, was_1d: bool) -> torch.Tensor:
    return x.squeeze(0) if was_1d else x


def make_valid_mask(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build validity mask from finite pred/target and optional mask."""
    valid = torch.isfinite(pred) & torch.isfinite(target)
    if mask is not None:
        valid = valid & mask.bool()
    return valid


def masked_center(
    x: torch.Tensor,
    mask: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Center x along N using mask."""
    m = mask.to(dtype=x.dtype)
    count = m.sum(dim=-1, keepdim=True).clamp_min(eps)
    mean = (x * m).sum(dim=-1, keepdim=True) / count
    return (x - mean) * m


def masked_pearson_corr(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Masked Pearson correlation per batch item.

    Returns:
        corr: [B] or scalar if input was 1D.
    """
    x2, was_1d = _ensure_2d(x)
    y2, _ = _ensure_2d(y)

    if x2.shape != y2.shape:
        raise ValueError(f"x and y shape mismatch: {tuple(x2.shape)} vs {tuple(y2.shape)}")

    if mask is not None:
        mask2, _ = _ensure_2d(mask.bool())
    else:
        mask2 = None

    valid = make_valid_mask(x2, y2, mask2)
    m = valid.to(dtype=x2.dtype)

    xc = masked_center(x2, valid, eps=eps)
    yc = masked_center(y2, valid, eps=eps)

    cov = (xc * yc * m).sum(dim=-1)
    vx = (xc.square() * m).sum(dim=-1).clamp_min(eps)
    vy = (yc.square() * m).sum(dim=-1).clamp_min(eps)

    corr = cov / torch.sqrt(vx * vy)
    corr = torch.clamp(corr, -1.0, 1.0)

    # If fewer than 2 valid elements, set corr to 0.
    valid_count = valid.sum(dim=-1)
    corr = torch.where(valid_count >= 2, corr, torch.zeros_like(corr))

    return _restore_dim(corr, was_1d)


def pearson_corr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Loss = 1 - mean(masked Pearson correlation)."""
    corr = masked_pearson_corr(pred, target, mask=mask, eps=eps)
    return 1.0 - corr.mean()


def soft_rank_pairwise(
    scores: torch.Tensor,
    temperature: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    regularized: bool = True,
) -> torch.Tensor:
    """
    Differentiable pairwise-sigmoid soft rank.

    Higher score -> higher rank.

    Approximation:
        rank_i = 1 + sum_j sigmoid((score_i - score_j) / tau)

    With this convention, the largest score has rank near N and the smallest
    score has rank near 1. If regularized=True, subtract 0.5 self-comparison
    contribution so the expected rank range is closer to [1, N].
    """
    x, was_1d = _ensure_2d(scores)
    tau = max(float(temperature), EPS)

    if mask is not None:
        m, _ = _ensure_2d(mask.bool())
    else:
        m = torch.ones_like(x, dtype=torch.bool)

    # [B, N, N], diff[i, j] = score_i - score_j
    diff = x.unsqueeze(-1) - x.unsqueeze(-2)
    pair = torch.sigmoid(diff / tau)

    valid_pair = m.unsqueeze(-1) & m.unsqueeze(-2)
    pair = pair * valid_pair.to(dtype=pair.dtype)

    rank = 1.0 + pair.sum(dim=-1)

    if regularized:
        # Remove sigmoid(0)=0.5 self contribution for valid positions.
        rank = rank - 0.5 * m.to(dtype=rank.dtype)

    rank = rank * m.to(dtype=rank.dtype)
    return _restore_dim(rank, was_1d)


def rank_to_pct(rank: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = EPS) -> torch.Tensor:
    """Convert rank to percentile rank by valid count."""
    r, was_1d = _ensure_2d(rank)

    if mask is not None:
        m, _ = _ensure_2d(mask.bool())
    else:
        m = torch.isfinite(r)

    count = m.sum(dim=-1, keepdim=True).to(dtype=r.dtype).clamp_min(eps)
    out = r / count
    out = out * m.to(dtype=out.dtype)
    return _restore_dim(out, was_1d)


def spearman_soft_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    target_is_rank_like: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Spearman-style loss using differentiable soft rank for predictions.

    pred:
        raw model scores.

    target:
        usually local rank_pct or rank_centered labels.
        If target_is_rank_like=False, target is also converted to soft rank.
    """
    p2, was_1d = _ensure_2d(pred)
    t2, _ = _ensure_2d(target)

    if p2.shape != t2.shape:
        raise ValueError(f"pred and target shape mismatch: {tuple(p2.shape)} vs {tuple(t2.shape)}")

    if mask is not None:
        m2, _ = _ensure_2d(mask.bool())
    else:
        m2 = None

    valid = make_valid_mask(p2, t2, m2)

    pred_rank = soft_rank_pairwise(p2, temperature=temperature, mask=valid)
    pred_rank_pct = rank_to_pct(pred_rank, mask=valid, eps=eps)

    if target_is_rank_like:
        target_rank_like = t2
    else:
        target_rank = soft_rank_pairwise(t2, temperature=temperature, mask=valid)
        target_rank_like = rank_to_pct(target_rank, mask=valid, eps=eps)

    return pearson_corr_loss(pred_rank_pct, target_rank_like, mask=valid, eps=eps)


def pairwise_logistic_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    margin: float = 0.0,
    max_pairs: Optional[int] = None,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Pairwise logistic ranking loss.

    For pairs where target_i > target_j, encourage pred_i > pred_j.

    Loss:
        softplus(-(pred_i - pred_j - margin))

    max_pairs:
        Optional random pair subsampling per batch item for memory control.
    """
    p, _ = _ensure_2d(pred)
    t, _ = _ensure_2d(target)

    if p.shape != t.shape:
        raise ValueError(f"pred and target shape mismatch: {tuple(p.shape)} vs {tuple(t.shape)}")

    if mask is not None:
        m, _ = _ensure_2d(mask.bool())
    else:
        m = make_valid_mask(p, t)

    losses = []

    for b in range(p.shape[0]):
        valid = m[b] & torch.isfinite(p[b]) & torch.isfinite(t[b])
        pb = p[b, valid]
        tb = t[b, valid]

        if pb.numel() < 2:
            continue

        # diff[i, j] = value_i - value_j.
        # For target_i > target_j, encourage pred_i > pred_j.
        target_diff = tb.unsqueeze(1) - tb.unsqueeze(0)
        pair_mask = target_diff > 0

        if not pair_mask.any():
            continue

        pred_diff = pb.unsqueeze(1) - pb.unsqueeze(0)
        pair_losses = F.softplus(-(pred_diff[pair_mask] - float(margin)))

        if max_pairs is not None and pair_losses.numel() > int(max_pairs):
            idx = torch.randperm(pair_losses.numel(), device=pair_losses.device)[: int(max_pairs)]
            pair_losses = pair_losses[idx]

        losses.append(pair_losses.mean())

    if not losses:
        return pred.sum() * 0.0

    return torch.stack(losses).mean()


def _rank_positions_desc(values: torch.Tensor) -> torch.Tensor:
    """
    Return 1-based ordinal positions after sorting values descending.

    Highest value gets position 1.
    """
    order = torch.argsort(values, descending=True)
    positions = torch.empty_like(order, dtype=torch.float32)
    positions[order] = torch.arange(1, values.numel() + 1, device=values.device, dtype=torch.float32)
    return positions


def ndcg_pairwise_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    gain_power: float = 1.0,
    max_pairs: Optional[int] = None,
    eps: float = EPS,
) -> torch.Tensor:
    """
    NDCG-style top-heavy pairwise logistic loss.

    Motivation
    ----------
    The strategy cares much more about the top-ranked stocks than the whole
    cross-section. This loss keeps the differentiable pairwise logistic form but
    weights pairs by a target-side NDCG-like discount:

        high true-rank item at ideal position 1 receives the largest weight;
        lower true-rank items receive smaller weights according to
        1 / log2(position + 1).

    For each pair where target_i > target_j, encourage:

        pred_i > pred_j

    Loss per pair:
        weight_ij * softplus(-(pred_i - pred_j) / temperature)

    Notes
    -----
    This is differentiable w.r.t. pred. The neural network training problem is
    still non-convex, but the per-pair logistic surrogate is smooth and stable.
    """
    p, _ = _ensure_2d(pred)
    t, _ = _ensure_2d(target)

    if p.shape != t.shape:
        raise ValueError(f"pred and target shape mismatch: {tuple(p.shape)} vs {tuple(t.shape)}")

    if mask is not None:
        m, _ = _ensure_2d(mask.bool())
    else:
        m = make_valid_mask(p, t)

    tau = max(float(temperature), eps)
    gain_power = max(float(gain_power), eps)

    losses = []

    for b in range(p.shape[0]):
        valid = m[b] & torch.isfinite(p[b]) & torch.isfinite(t[b])
        pb = p[b, valid].float()
        tb = t[b, valid].float()

        n = pb.numel()
        if n < 2:
            continue

        # Convert arbitrary target values to rank-percentile-like relevance in [0, 1].
        # This makes the loss robust for target_mode=rank_pct, rank_centered,
        # zscore, or raw returns.
        target_pos = _rank_positions_desc(tb)  # top target -> 1
        rel = 1.0 - (target_pos - 1.0) / max(float(n - 1), 1.0)
        rel = rel.clamp(0.0, 1.0)

        # NDCG-like gain and ideal discount.
        gain = torch.pow(torch.tensor(2.0, device=pb.device), gain_power * rel) - 1.0
        discount = 1.0 / torch.log2(target_pos + 1.0)

        # Pair i,j: if target_i > target_j, i should be ranked above j.
        target_diff = tb.unsqueeze(1) - tb.unsqueeze(0)
        pair_mask = target_diff > 0

        if not pair_mask.any():
            continue

        pred_diff = pb.unsqueeze(1) - pb.unsqueeze(0)

        gain_diff = (gain.unsqueeze(1) - gain.unsqueeze(0)).abs()
        # Emphasize pairs whose better member should be near the top.
        top_discount = torch.maximum(discount.unsqueeze(1), discount.unsqueeze(0))
        pair_weight = gain_diff * top_discount
        pair_weight = pair_weight[pair_mask].clamp_min(eps)

        pair_losses = F.softplus(-pred_diff[pair_mask] / tau)
        weighted = pair_losses * pair_weight

        if max_pairs is not None and weighted.numel() > int(max_pairs):
            idx = torch.randperm(weighted.numel(), device=weighted.device)[: int(max_pairs)]
            weighted = weighted[idx]
            pair_weight = pair_weight[idx]

        losses.append(weighted.sum() / pair_weight.sum().clamp_min(eps))

    if not losses:
        return pred.sum() * 0.0

    return torch.stack(losses).mean()


def rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    config: RankLossConfig = RankLossConfig(),
) -> torch.Tensor:
    """Dispatch ranking loss by config.loss_type."""
    loss_type = str(config.loss_type).lower()

    if loss_type in {"pearson", "ic", "corr"}:
        return pearson_corr_loss(pred, target, mask=mask, eps=config.eps)

    if loss_type in {"spearman", "soft_spearman", "soft_rank"}:
        return spearman_soft_rank_loss(
            pred,
            target,
            mask=mask,
            temperature=config.temperature,
            target_is_rank_like=True,
            eps=config.eps,
        )

    if loss_type in {"pairwise", "pairwise_logistic"}:
        return pairwise_logistic_loss(
            pred,
            target,
            mask=mask,
            margin=config.pairwise_margin,
            max_pairs=config.pairwise_max_pairs,
            eps=config.eps,
        )

    if loss_type in {"ndcg", "ndcg_pairwise", "topheavy_pairwise"}:
        return ndcg_pairwise_loss(
            pred,
            target,
            mask=mask,
            temperature=config.ndcg_temperature,
            gain_power=config.ndcg_gain_power,
            max_pairs=config.ndcg_max_pairs,
            eps=config.eps,
        )

    raise ValueError(f"unsupported loss_type: {config.loss_type!r}")


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def rank_ic(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Spearman-like RankIC using hard ranks, returned per batch item.

    Uses float32 internally to avoid fp16 overflow under AMP.
    """
    p, was_1d = _ensure_2d(pred.detach().float())
    t, _ = _ensure_2d(target.detach().float())

    if mask is not None:
        m, _ = _ensure_2d(mask.bool())
    else:
        m = make_valid_mask(p, t)

    out = []

    for b in range(p.shape[0]):
        valid = m[b] & torch.isfinite(p[b]) & torch.isfinite(t[b])
        if valid.sum() < 2:
            out.append(torch.tensor(0.0, device=p.device, dtype=torch.float32))
            continue

        pr = torch.argsort(torch.argsort(p[b, valid])).float()
        tr = torch.argsort(torch.argsort(t[b, valid])).float()
        corr = masked_pearson_corr(pr, tr, eps=eps).float()
        out.append(corr)

    result = torch.stack(out)
    return _restore_dim(result, was_1d)


@torch.no_grad()
def information_coefficient(
    pred: torch.Tensor,
    forward_return: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Pearson IC between model scores and raw forward returns."""
    return masked_pearson_corr(pred, forward_return, mask=mask, eps=eps)


@torch.no_grad()
def topk_mean_return(
    pred: torch.Tensor,
    forward_return: torch.Tensor,
    k: int = 20,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean raw forward return of top-k scores, per batch item."""
    p, was_1d = _ensure_2d(pred)
    r, _ = _ensure_2d(forward_return)

    if mask is not None:
        m, _ = _ensure_2d(mask.bool())
    else:
        m = make_valid_mask(p, r)

    vals = []
    for b in range(p.shape[0]):
        valid = m[b] & torch.isfinite(p[b]) & torch.isfinite(r[b])
        if valid.sum() == 0:
            vals.append(torch.tensor(0.0, device=p.device, dtype=p.dtype))
            continue
        kk = min(int(k), int(valid.sum().item()))
        valid_scores = p[b, valid]
        valid_returns = r[b, valid]
        idx = torch.topk(valid_scores, k=kk, largest=True).indices
        vals.append(valid_returns[idx].mean())

    result = torch.stack(vals)
    return _restore_dim(result, was_1d)


@torch.no_grad()
def topk_excess_return(
    pred: torch.Tensor,
    forward_return: torch.Tensor,
    k: int = 20,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Top-k mean return minus full valid-universe mean return."""
    p, was_1d = _ensure_2d(pred.detach().float())
    r, _ = _ensure_2d(forward_return.detach().float())

    if mask is not None:
        m, _ = _ensure_2d(mask.bool())
    else:
        m = make_valid_mask(p, r)

    vals = []
    for b in range(p.shape[0]):
        valid = m[b] & torch.isfinite(p[b]) & torch.isfinite(r[b])
        if valid.sum() == 0:
            vals.append(torch.tensor(0.0, device=p.device, dtype=torch.float32))
            continue

        kk = min(int(k), int(valid.sum().item()))
        valid_scores = p[b, valid]
        valid_returns = r[b, valid]
        idx = torch.topk(valid_scores, k=kk, largest=True).indices
        vals.append(valid_returns[idx].mean() - valid_returns.mean())

    result = torch.stack(vals)
    return _restore_dim(result, was_1d)


@torch.no_grad()
def long_short_spread_return(
    pred: torch.Tensor,
    forward_return: torch.Tensor,
    k: int = 20,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Top-k predicted mean return minus bottom-k predicted mean return."""
    p, was_1d = _ensure_2d(pred.detach().float())
    r, _ = _ensure_2d(forward_return.detach().float())

    if mask is not None:
        m, _ = _ensure_2d(mask.bool())
    else:
        m = make_valid_mask(p, r)

    vals = []
    for b in range(p.shape[0]):
        valid = m[b] & torch.isfinite(p[b]) & torch.isfinite(r[b])
        if valid.sum() < 2:
            vals.append(torch.tensor(0.0, device=p.device, dtype=torch.float32))
            continue

        kk = min(int(k), int(valid.sum().item()) // 2)
        if kk <= 0:
            vals.append(torch.tensor(0.0, device=p.device, dtype=torch.float32))
            continue

        valid_scores = p[b, valid]
        valid_returns = r[b, valid]
        top_idx = torch.topk(valid_scores, k=kk, largest=True).indices
        bot_idx = torch.topk(valid_scores, k=kk, largest=False).indices
        vals.append(valid_returns[top_idx].mean() - valid_returns[bot_idx].mean())

    result = torch.stack(vals)
    return _restore_dim(result, was_1d)


@torch.no_grad()
def topk_hit_rate(
    pred: torch.Tensor,
    target: torch.Tensor,
    k: int = 20,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Predicted Top-K and true Top-K overlap ratio.

    Formula:
        hit_rate@K = |PredTopK intersection TrueTopK| / K

    Since both sets have size K, this is also Precision@K and Recall@K.
    """
    p, was_1d = _ensure_2d(pred.detach().float())
    t, _ = _ensure_2d(target.detach().float())

    if mask is not None:
        m, _ = _ensure_2d(mask.bool())
    else:
        m = make_valid_mask(p, t)

    vals = []
    for b in range(p.shape[0]):
        valid = m[b] & torch.isfinite(p[b]) & torch.isfinite(t[b])
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            vals.append(torch.tensor(0.0, device=p.device, dtype=torch.float32))
            continue

        kk = min(int(k), n_valid)
        valid_scores = p[b, valid]
        valid_target = t[b, valid]

        pred_idx = torch.topk(valid_scores, k=kk, largest=True).indices
        true_idx = torch.topk(valid_target, k=kk, largest=True).indices

        pred_mask = torch.zeros(n_valid, dtype=torch.bool, device=p.device)
        true_mask = torch.zeros(n_valid, dtype=torch.bool, device=p.device)
        pred_mask[pred_idx] = True
        true_mask[true_idx] = True

        hit = (pred_mask & true_mask).sum().float() / float(kk)
        vals.append(hit)

    result = torch.stack(vals)
    return _restore_dim(result, was_1d)


@torch.no_grad()
def ndcg_at_k(
    pred: torch.Tensor,
    target: torch.Tensor,
    k: int = 20,
    mask: Optional[torch.Tensor] = None,
    gain_power: float = 1.0,
    eps: float = EPS,
) -> torch.Tensor:
    """
    NDCG@K computed from target-side relevance.

    This is a monitoring metric, not the training loss. It is robust to target
    scale by converting target values to rank-percentile relevance inside each
    cross-section.
    """
    p, was_1d = _ensure_2d(pred.detach().float())
    t, _ = _ensure_2d(target.detach().float())

    if mask is not None:
        m, _ = _ensure_2d(mask.bool())
    else:
        m = make_valid_mask(p, t)

    vals = []
    gain_power = max(float(gain_power), eps)

    for b in range(p.shape[0]):
        valid = m[b] & torch.isfinite(p[b]) & torch.isfinite(t[b])
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            vals.append(torch.tensor(0.0, device=p.device, dtype=torch.float32))
            continue

        kk = min(int(k), n_valid)
        valid_scores = p[b, valid]
        valid_target = t[b, valid]

        target_pos = _rank_positions_desc(valid_target)
        if n_valid > 1:
            rel = 1.0 - (target_pos - 1.0) / float(n_valid - 1)
        else:
            rel = torch.ones_like(target_pos)
        rel = rel.clamp(0.0, 1.0)

        gains = torch.pow(torch.tensor(2.0, device=p.device), gain_power * rel) - 1.0
        discounts = 1.0 / torch.log2(torch.arange(2, kk + 2, device=p.device, dtype=torch.float32))

        pred_idx = torch.topk(valid_scores, k=kk, largest=True).indices
        ideal_idx = torch.topk(valid_target, k=kk, largest=True).indices

        dcg = (gains[pred_idx] * discounts).sum()
        idcg = (gains[ideal_idx] * discounts).sum().clamp_min(eps)
        vals.append(dcg / idcg)

    result = torch.stack(vals)
    return _restore_dim(result, was_1d)


@torch.no_grad()
def topk_true_rank_mean(
    pred: torch.Tensor,
    target: torch.Tensor,
    k: int = 20,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Mean true rank position of predicted Top-K names.

    Rank convention:
        true best stock has rank position 1.
        true worst stock has rank position N.

    For K=20:
        ideal value is (1 + 20) / 2 = 10.5.
        random expectation is approximately (N + 1) / 2.

    Lower is better.
    """
    p, was_1d = _ensure_2d(pred.detach().float())
    t, _ = _ensure_2d(target.detach().float())

    if mask is not None:
        m, _ = _ensure_2d(mask.bool())
    else:
        m = make_valid_mask(p, t)

    vals = []
    for b in range(p.shape[0]):
        valid = m[b] & torch.isfinite(p[b]) & torch.isfinite(t[b])
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            vals.append(torch.tensor(float("nan"), device=p.device, dtype=torch.float32))
            continue

        kk = min(int(k), n_valid)
        valid_scores = p[b, valid]
        valid_target = t[b, valid]

        true_rank_pos = _rank_positions_desc(valid_target)  # best target -> 1
        pred_top_idx = torch.topk(valid_scores, k=kk, largest=True).indices
        vals.append(true_rank_pos[pred_top_idx].mean())

    result = torch.stack(vals)
    return _restore_dim(result, was_1d)


@torch.no_grad()
def topk_true_rank_normalized_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    k: int = 20,
    mask: Optional[torch.Tensor] = None,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Normalize topk_true_rank_mean to an interpretable score.

    Score convention:
        1.0 = ideal predicted Top-K are exactly true Top-K.
        0.0 = random expectation.
        <0  = worse than random.

    This is useful for dashboards. Higher is better.
    """
    p, was_1d = _ensure_2d(pred.detach().float())
    t, _ = _ensure_2d(target.detach().float())

    if mask is not None:
        m, _ = _ensure_2d(mask.bool())
    else:
        m = make_valid_mask(p, t)

    vals = []
    for b in range(p.shape[0]):
        valid = m[b] & torch.isfinite(p[b]) & torch.isfinite(t[b])
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            vals.append(torch.tensor(float("nan"), device=p.device, dtype=torch.float32))
            continue

        kk = min(int(k), n_valid)
        rank_mean = topk_true_rank_mean(p[b], t[b], k=kk, mask=valid).float()

        ideal = (1.0 + float(kk)) / 2.0
        random_mean = (1.0 + float(n_valid)) / 2.0
        denom = max(random_mean - ideal, eps)
        score = (random_mean - rank_mean) / denom
        vals.append(score)

    result = torch.stack(vals)
    return _restore_dim(result, was_1d)


__all__ = [
    "EPS",
    "RankLossConfig",
    "TemperatureSchedule",
    "make_valid_mask",
    "masked_pearson_corr",
    "pearson_corr_loss",
    "soft_rank_pairwise",
    "rank_to_pct",
    "spearman_soft_rank_loss",
    "pairwise_logistic_loss",
    "ndcg_pairwise_loss",
    "rank_loss",
    "rank_ic",
    "information_coefficient",
    "topk_mean_return",
    "topk_excess_return",
    "long_short_spread_return",
    "topk_hit_rate",
    "ndcg_at_k",
    "topk_true_rank_mean",
    "topk_true_rank_normalized_score",
]
