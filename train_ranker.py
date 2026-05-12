"""
train_ranker.py

Training utilities for the Dual-Transformer cross-sectional ranker.

This file is intentionally a reusable training module rather than a hard-coded
project script. It can be imported from notebooks or used as a reference for a
CLI training script.

Expected flow
-------------
1. Build or load a FeatureLabelBundle using factor_pipeline.py.
2. Build train/valid datasets using rank_dataset.py.
3. Create a DualTransformerRanker from dual_transformer_model.py.
4. Call fit_model().

Minimal example
---------------
import factor_pipeline as fp
import rank_dataset as rd
import dual_transformer_model as dtm
import train_ranker as tr

bundle = fp.load_bundle("data/cache/stage1_factor_label_bundle.pkl")

ds_config = rd.CrossSectionDatasetConfig(
    sample_size=512,
    seq_len=128,
    samples_per_date=4,
    label_name="label_ret_t1_t6",
    target_mode="rank_pct",
)

train_ds, valid_ds = rd.make_train_valid_datasets_from_bundle(
    bundle,
    config=ds_config,
    train_end="20231231",
)

model = dtm.DualTransformerRanker.from_feature_counts(
    num_ts_factors=len(bundle.metadata["ts_factor_names"]),
    num_scalar_factors=len(bundle.metadata["scalar_factor_names"]),
    seq_len=128,
)

train_config = tr.TrainConfig(max_epochs=50, checkpoint_dir="checkpoints/ranker")
history = tr.fit_model(model, train_ds, valid_ds, train_config)
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # tensorboard may be missing in a fresh environment
    SummaryWriter = None  # type: ignore

import dual_transformer_model as dtm
import rank_dataset as rd
import rank_loss as rl


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class TrainConfig:
    """Training configuration."""

    max_epochs: int = 50
    batch_size: int = 1
    num_workers: int = 0

    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: Optional[float] = 1.0

    loss_type: str = "spearman"
    tau_start: float = 1.0
    tau_end: float = 0.1
    tau_decay_epochs: int = 50

    pairwise_margin: float = 0.0
    pairwise_max_pairs: Optional[int] = 20000

    # NDCG-style top-heavy pairwise loss.
    ndcg_temperature: float = 1.0
    ndcg_gain_power: float = 1.0
    ndcg_max_pairs: Optional[int] = 50000

    # Explicit L2 penalty. AdamW weight_decay is still decoupled optimizer
    # regularization; this term is added to the training objective directly.
    l2_lambda: float = 0.0
    l2_exclude_bias_norm: bool = True

    topk_metric_k: int = 20

    device: str = "auto"
    seed: int = 42

    # Experiment output.
    # checkpoint_dir is treated as the experiment root by default.
    # Each fit_model() call creates checkpoint_dir/<timestamp_or_run_name>/.
    checkpoint_dir: Optional[str] = None
    create_timestamp_run_dir: bool = True
    run_name: Optional[str] = None
    timestamp_format: str = "%Y%m%d_%H%M%S"
    save_best: bool = True
    save_last: bool = True
    save_history_each_epoch: bool = True
    save_dataset_summary: bool = True
    save_model_config: bool = True

    early_stopping_patience: Optional[int] = 10
    metric_for_best: str = "valid_rank_ic"
    maximize_metric: bool = True

    # Console / TensorBoard logging.
    # Set log_every_steps=0 to suppress per-step console logs.
    log_every_steps: int = 0
    use_amp: bool = False

    use_tensorboard: bool = True
    tensorboard_dirname: str = "tensorboard"
    tb_log_every_steps: int = 50
    tb_log_memory_every_steps: int = 10
    tb_log_grad_norm: bool = True
    tb_log_epoch_metrics: bool = True
    tb_log_config_text: bool = True
    tb_log_histograms: bool = False
    tb_histogram_every_epochs: int = 5


# ============================================================
# Setup helpers
# ============================================================

def set_global_seed(seed: int) -> None:
    """Set common random seeds."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve training device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_dataloader(
    dataset: rd.CrossSectionRankDataset,
    config: TrainConfig,
    shuffle: bool,
) -> DataLoader:
    """Create DataLoader for CrossSectionRankDataset."""
    return DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=bool(shuffle),
        num_workers=int(config.num_workers),
        collate_fn=rd.cross_section_collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a batch to device."""
    out = dict(batch)
    for key in ("x_ts", "x_scalar", "y", "y_raw"):
        if key in out and torch.is_tensor(out[key]):
            out[key] = out[key].to(device, non_blocking=True)
    return out


def build_loss_config(train_config: TrainConfig, epoch: int) -> rl.RankLossConfig:
    """Build RankLossConfig for current epoch."""
    tau_schedule = rl.TemperatureSchedule(
        start=float(train_config.tau_start),
        end=float(train_config.tau_end),
        decay_epochs=int(train_config.tau_decay_epochs),
    )
    tau = tau_schedule(epoch)
    return rl.RankLossConfig(
        loss_type=train_config.loss_type,
        temperature=tau,
        pairwise_margin=float(train_config.pairwise_margin),
        pairwise_max_pairs=train_config.pairwise_max_pairs,
        ndcg_temperature=float(train_config.ndcg_temperature),
        ndcg_gain_power=float(train_config.ndcg_gain_power),
        ndcg_max_pairs=train_config.ndcg_max_pairs,
    )


def create_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Create AdamW optimizer."""
    return AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )


def l2_regularization_loss(
    model: nn.Module,
    exclude_bias_norm: bool = True,
) -> torch.Tensor:
    """
    Explicit L2 penalty added to the loss.

    By default, bias vectors and normalization parameters are excluded. This is
    usually more stable for Transformer / GroupNorm / LayerNorm models.
    """
    penalty: Optional[torch.Tensor] = None

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if exclude_bias_norm:
            lname = name.lower()
            if lname.endswith(".bias") or "norm" in lname:
                continue

        term = param.float().pow(2).sum()
        penalty = term if penalty is None else penalty + term

    if penalty is None:
        first = next(model.parameters())
        return first.sum() * 0.0

    return penalty



def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return learning rate of the first optimizer param group."""
    if not optimizer.param_groups:
        return float("nan")
    return float(optimizer.param_groups[0].get("lr", float("nan")))


def get_gpu_memory_stats(device: torch.device) -> dict[str, float]:
    """
    Return CUDA memory stats in MB.

    Values are 0 when device is not CUDA.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
            "max_reserved_mb": 0.0,
        }

    return {
        "allocated_mb": float(torch.cuda.memory_allocated(device) / 1024**2),
        "reserved_mb": float(torch.cuda.memory_reserved(device) / 1024**2),
        "max_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "max_reserved_mb": float(torch.cuda.max_memory_reserved(device) / 1024**2),
    }


def compute_total_grad_norm(model: nn.Module) -> float:
    """
    Compute global L2 gradient norm.

    This is used when grad clipping is disabled but TensorBoard grad logging is
    requested.
    """
    total_sq = 0.0
    has_grad = False
    for p in model.parameters():
        if p.grad is None:
            continue
        grad = p.grad.detach()
        if not torch.isfinite(grad).all():
            continue
        param_norm = grad.float().norm(2).item()
        total_sq += param_norm * param_norm
        has_grad = True
    return float(total_sq ** 0.5) if has_grad else float("nan")


def create_tensorboard_writer(
    run_dir: Optional[Path],
    train_config: TrainConfig,
) -> Optional[Any]:
    """
    Create TensorBoard SummaryWriter under run_dir/tensorboard.

    Returns None when disabled, run_dir is None, or tensorboard is not installed.
    """
    if not bool(train_config.use_tensorboard):
        return None

    if run_dir is None:
        print("TensorBoard disabled because run_dir is None. Set checkpoint_dir to enable it.")
        return None

    if SummaryWriter is None:
        print("TensorBoard is not installed. Install with: pip install tensorboard")
        return None

    tb_dir = run_dir / str(train_config.tensorboard_dirname)
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tb_dir), flush_secs=30)


def tb_add_scalar(writer: Optional[Any], tag: str, value: Any, step: int) -> None:
    """Safely write one scalar to TensorBoard."""
    if writer is None:
        return
    try:
        if value is None:
            return
        value_f = float(value)
        if not np.isfinite(value_f):
            return
        writer.add_scalar(tag, value_f, int(step))
    except Exception:
        return


def tb_log_metrics(
    writer: Optional[Any],
    prefix: str,
    metrics: dict[str, float],
    step: int,
) -> None:
    """Write a metrics dictionary to TensorBoard."""
    if writer is None:
        return
    for k, v in metrics.items():
        tb_add_scalar(writer, f"{prefix}/{k}", v, step)


def tb_log_memory(
    writer: Optional[Any],
    device: torch.device,
    step: int,
    prefix: str = "memory",
) -> None:
    """Write CUDA memory stats to TensorBoard."""
    if writer is None:
        return
    stats = get_gpu_memory_stats(device)
    for k, v in stats.items():
        tb_add_scalar(writer, f"{prefix}/{k}", v, step)


def tb_log_grad_and_param_norms(
    writer: Optional[Any],
    model: nn.Module,
    step: int,
    prefix: str = "norms",
) -> None:
    """Write global parameter and gradient norms to TensorBoard."""
    if writer is None:
        return

    total_param_sq = 0.0
    total_grad_sq = 0.0
    has_grad = False

    for p in model.parameters():
        with torch.no_grad():
            param_norm = p.detach().float().norm(2).item()
            total_param_sq += param_norm * param_norm

            if p.grad is not None:
                grad = p.grad.detach()
                if torch.isfinite(grad).all():
                    grad_norm = grad.float().norm(2).item()
                    total_grad_sq += grad_norm * grad_norm
                    has_grad = True

    tb_add_scalar(writer, f"{prefix}/param_l2", total_param_sq ** 0.5, step)
    if has_grad:
        tb_add_scalar(writer, f"{prefix}/grad_l2", total_grad_sq ** 0.5, step)


def tb_log_model_histograms(
    writer: Optional[Any],
    model: nn.Module,
    epoch: int,
) -> None:
    """Optionally write parameter histograms."""
    if writer is None:
        return
    for name, param in model.named_parameters():
        try:
            writer.add_histogram(f"parameters/{name}", param.detach().float().cpu(), epoch)
            if param.grad is not None:
                writer.add_histogram(f"gradients/{name}", param.grad.detach().float().cpu(), epoch)
        except Exception:
            continue


# ============================================================
# Metrics
# ============================================================


def _safe_mean(values: list[float]) -> float:
    """
    Mean that does not warn when all values are NaN.

    Returning NaN here is intentional; the caller can decide whether this metric
    is unavailable. This avoids RuntimeWarning: Mean of empty slice.
    """
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


@torch.no_grad()
def compute_batch_metrics(
    scores: torch.Tensor,
    y: torch.Tensor,
    y_raw: torch.Tensor,
    topk: int = 20,
) -> dict[str, float]:
    """
    Compute metrics for one batch.

    Metrics are computed in float32 even when AMP produces float16 scores.
    """
    scores_f = scores.detach().float()
    y_f = y.detach().float()
    y_raw_f = y_raw.detach().float()

    rank_ic = rl.rank_ic(scores_f, y_f).detach().float().mean().item()
    ic = rl.information_coefficient(scores_f, y_raw_f).detach().float().mean().item()

    topk_ret = rl.topk_mean_return(scores_f, y_raw_f, k=int(topk)).detach().float().mean().item()
    topk_excess_ret = rl.topk_excess_return(scores_f, y_raw_f, k=int(topk)).detach().float().mean().item()
    long_short_spread = rl.long_short_spread_return(scores_f, y_raw_f, k=int(topk)).detach().float().mean().item()

    topk_hit = rl.topk_hit_rate(scores_f, y_f, k=int(topk)).detach().float().mean().item()
    ndcg_k = rl.ndcg_at_k(scores_f, y_f, k=int(topk)).detach().float().mean().item()
    topk_rank_mean = rl.topk_true_rank_mean(scores_f, y_f, k=int(topk)).detach().float().mean().item()
    topk_rank_score = rl.topk_true_rank_normalized_score(scores_f, y_f, k=int(topk)).detach().float().mean().item()

    # Score distribution diagnostics.
    # If score_std and score_range are near zero on validation, the model is
    # giving almost identical scores to every stock in the cross-section.
    score_std = scores_f.std(dim=-1, unbiased=False).mean().item()
    score_range = (scores_f.max(dim=-1).values - scores_f.min(dim=-1).values).mean().item()
    score_abs_mean = scores_f.abs().mean().item()
    score_mean = scores_f.mean().item()

    return {
        "rank_ic": float(rank_ic),
        "ic": float(ic),
        "topk_ret": float(topk_ret),
        "topk_excess_ret": float(topk_excess_ret),
        "long_short_spread": float(long_short_spread),
        "topk_hit_rate": float(topk_hit),
        "ndcg_at_k": float(ndcg_k),
        "topk_true_rank_mean": float(topk_rank_mean),
        "topk_true_rank_score": float(topk_rank_score),
        "score_std": float(score_std),
        "score_range": float(score_range),
        "score_abs_mean": float(score_abs_mean),
        "score_mean": float(score_mean),
    }


# ============================================================
# Train / validation loops
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_config: TrainConfig,
    epoch: int,
    writer: Optional[Any] = None,
    global_step: int = 0,
) -> tuple[dict[str, float], int]:
    """Train one epoch."""
    model.train()

    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    epoch_start = time.perf_counter()
    loss_config = build_loss_config(train_config, epoch)

    use_amp = bool(train_config.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp and hasattr(torch, "amp") else None

    losses: list[float] = []
    objective_losses: list[float] = []
    l2_losses: list[float] = []
    rank_ics: list[float] = []
    ics: list[float] = []
    topk_rets: list[float] = []
    topk_excess_rets: list[float] = []
    long_short_spreads: list[float] = []
    topk_hit_rates: list[float] = []
    ndcg_at_ks: list[float] = []
    topk_true_rank_means: list[float] = []
    topk_true_rank_scores: list[float] = []
    score_stds: list[float] = []
    score_ranges: list[float] = []
    score_abs_means: list[float] = []
    score_means: list[float] = []
    grad_norms: list[float] = []

    for step, batch in enumerate(loader, start=1):
        global_step += 1
        step_start = time.perf_counter()
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        grad_norm = float("nan")

        if use_amp:
            with torch.amp.autocast("cuda"):
                scores = model(batch["x_ts"], batch["x_scalar"])
                rank_objective_loss = rl.rank_loss(scores, batch["y"], config=loss_config)
                l2_loss = l2_regularization_loss(
                    model,
                    exclude_bias_norm=bool(train_config.l2_exclude_bias_norm),
                )
                loss = rank_objective_loss + float(train_config.l2_lambda) * l2_loss

            assert scaler is not None
            scaler.scale(loss).backward()

            if train_config.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(train_config.grad_clip_norm),
                )
                grad_norm = float(grad_norm_tensor.detach().float().cpu().item())
            elif train_config.tb_log_grad_norm:
                scaler.unscale_(optimizer)
                grad_norm = compute_total_grad_norm(model)

            scaler.step(optimizer)
            scaler.update()
        else:
            scores = model(batch["x_ts"], batch["x_scalar"])
            rank_objective_loss = rl.rank_loss(scores, batch["y"], config=loss_config)
            l2_loss = l2_regularization_loss(
                model,
                exclude_bias_norm=bool(train_config.l2_exclude_bias_norm),
            )
            loss = rank_objective_loss + float(train_config.l2_lambda) * l2_loss
            loss.backward()

            if train_config.grad_clip_norm is not None:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(train_config.grad_clip_norm),
                )
                grad_norm = float(grad_norm_tensor.detach().float().cpu().item())
            elif train_config.tb_log_grad_norm:
                grad_norm = compute_total_grad_norm(model)

            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        objective_losses.append(float(rank_objective_loss.detach().cpu().item()))
        l2_losses.append(float(l2_loss.detach().cpu().item()))
        if np.isfinite(grad_norm):
            grad_norms.append(float(grad_norm))

        metrics = compute_batch_metrics(
            scores.detach(),
            batch["y"].detach(),
            batch["y_raw"].detach(),
            topk=train_config.topk_metric_k,
        )
        rank_ics.append(metrics["rank_ic"])
        ics.append(metrics["ic"])
        topk_rets.append(metrics["topk_ret"])
        topk_excess_rets.append(metrics["topk_excess_ret"])
        long_short_spreads.append(metrics["long_short_spread"])
        topk_hit_rates.append(metrics["topk_hit_rate"])
        ndcg_at_ks.append(metrics["ndcg_at_k"])
        topk_true_rank_means.append(metrics["topk_true_rank_mean"])
        topk_true_rank_scores.append(metrics["topk_true_rank_score"])
        score_stds.append(metrics["score_std"])
        score_ranges.append(metrics["score_range"])
        score_abs_means.append(metrics["score_abs_mean"])
        score_means.append(metrics["score_mean"])

        # TensorBoard step-level logging.
        if train_config.tb_log_every_steps and global_step % int(train_config.tb_log_every_steps) == 0:
            tb_add_scalar(writer, "train_step/loss", float(loss.detach().cpu().item()), global_step)
            tb_add_scalar(writer, "train_step/objective_loss", float(rank_objective_loss.detach().cpu().item()), global_step)
            tb_add_scalar(writer, "train_step/l2_loss_raw", float(l2_loss.detach().cpu().item()), global_step)
            tb_add_scalar(writer, "train_step/l2_loss_weighted", float(train_config.l2_lambda) * float(l2_loss.detach().cpu().item()), global_step)
            tb_add_scalar(writer, "train_step/rank_ic", metrics["rank_ic"], global_step)
            tb_add_scalar(writer, "train_step/ic", metrics["ic"], global_step)
            tb_add_scalar(writer, "train_step/topk_ret", metrics["topk_ret"], global_step)
            tb_add_scalar(writer, "train_step/topk_excess_ret", metrics["topk_excess_ret"], global_step)
            tb_add_scalar(writer, "train_step/long_short_spread", metrics["long_short_spread"], global_step)
            tb_add_scalar(writer, "train_step/topk_hit_rate", metrics["topk_hit_rate"], global_step)
            tb_add_scalar(writer, "train_step/ndcg_at_k", metrics["ndcg_at_k"], global_step)
            tb_add_scalar(writer, "train_step/topk_true_rank_mean", metrics["topk_true_rank_mean"], global_step)
            tb_add_scalar(writer, "train_step/topk_true_rank_score", metrics["topk_true_rank_score"], global_step)
            tb_add_scalar(writer, "train_step/score_std", metrics["score_std"], global_step)
            tb_add_scalar(writer, "train_step/score_range", metrics["score_range"], global_step)
            tb_add_scalar(writer, "train_step/score_abs_mean", metrics["score_abs_mean"], global_step)
            tb_add_scalar(writer, "train_step/score_mean", metrics["score_mean"], global_step)
            tb_add_scalar(writer, "train_step/temperature", loss_config.temperature, global_step)
            tb_add_scalar(writer, "train_step/lr", get_current_lr(optimizer), global_step)
            tb_add_scalar(writer, "train_step/seconds_per_step", time.perf_counter() - step_start, global_step)
            if np.isfinite(grad_norm):
                tb_add_scalar(writer, "train_step/grad_norm", grad_norm, global_step)

        if train_config.tb_log_memory_every_steps and global_step % int(train_config.tb_log_memory_every_steps) == 0:
            tb_log_memory(writer, device, global_step, prefix="memory_step")

        if train_config.log_every_steps and step % int(train_config.log_every_steps) == 0:
            print(
                f"epoch={epoch:03d} step={step:05d} "
                f"loss={_safe_mean(losses):.6f} "
                f"rank_ic={_safe_mean(rank_ics):.4f} "
                f"ic={_safe_mean(ics):.4f} "
                f"topk_ret={_safe_mean(topk_rets):.6f} "
                f"tau={loss_config.temperature:.4f}"
            )

    elapsed = time.perf_counter() - epoch_start
    memory = get_gpu_memory_stats(device)

    metrics_out = {
        "train_loss": _safe_mean(losses),
        "train_objective_loss": _safe_mean(objective_losses),
        "train_l2_loss_raw": _safe_mean(l2_losses),
        "train_l2_loss_weighted": float(train_config.l2_lambda) * _safe_mean(l2_losses),
        "train_rank_ic": _safe_mean(rank_ics),
        "train_ic": _safe_mean(ics),
        "train_topk_ret": _safe_mean(topk_rets),
        "train_topk_excess_ret": _safe_mean(topk_excess_rets),
        "train_long_short_spread": _safe_mean(long_short_spreads),
        "train_topk_hit_rate": _safe_mean(topk_hit_rates),
        "train_ndcg_at_k": _safe_mean(ndcg_at_ks),
        "train_topk_true_rank_mean": _safe_mean(topk_true_rank_means),
        "train_topk_true_rank_score": _safe_mean(topk_true_rank_scores),
        "train_score_std": _safe_mean(score_stds),
        "train_score_range": _safe_mean(score_ranges),
        "train_score_abs_mean": _safe_mean(score_abs_means),
        "train_score_mean": _safe_mean(score_means),
        "train_grad_norm": _safe_mean(grad_norms),
        "train_epoch_seconds": float(elapsed),
        "train_steps_per_second": float(len(loader) / elapsed) if elapsed > 0 else float("nan"),
        "temperature": float(loss_config.temperature),
        "lr": get_current_lr(optimizer),
        "gpu_allocated_mb": memory["allocated_mb"],
        "gpu_reserved_mb": memory["reserved_mb"],
        "gpu_max_allocated_mb": memory["max_allocated_mb"],
        "gpu_max_reserved_mb": memory["max_reserved_mb"],
    }

    return metrics_out, global_step



@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    train_config: TrainConfig,
    epoch: int,
    writer: Optional[Any] = None,
    global_step: Optional[int] = None,
) -> dict[str, float]:
    """Validate one epoch."""
    model.eval()

    val_start = time.perf_counter()
    loss_config = build_loss_config(train_config, epoch)

    losses: list[float] = []
    rank_ics: list[float] = []
    ics: list[float] = []
    topk_rets: list[float] = []
    topk_excess_rets: list[float] = []
    long_short_spreads: list[float] = []
    topk_hit_rates: list[float] = []
    ndcg_at_ks: list[float] = []
    topk_true_rank_means: list[float] = []
    topk_true_rank_scores: list[float] = []
    score_stds: list[float] = []
    score_ranges: list[float] = []
    score_abs_means: list[float] = []
    score_means: list[float] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        scores = model(batch["x_ts"], batch["x_scalar"])
        loss = rl.rank_loss(scores, batch["y"], config=loss_config)

        losses.append(float(loss.detach().cpu().item()))

        metrics = compute_batch_metrics(
            scores.detach(),
            batch["y"].detach(),
            batch["y_raw"].detach(),
            topk=train_config.topk_metric_k,
        )
        rank_ics.append(metrics["rank_ic"])
        ics.append(metrics["ic"])
        topk_rets.append(metrics["topk_ret"])
        topk_excess_rets.append(metrics["topk_excess_ret"])
        long_short_spreads.append(metrics["long_short_spread"])
        topk_hit_rates.append(metrics["topk_hit_rate"])
        ndcg_at_ks.append(metrics["ndcg_at_k"])
        topk_true_rank_means.append(metrics["topk_true_rank_mean"])
        topk_true_rank_scores.append(metrics["topk_true_rank_score"])
        score_stds.append(metrics["score_std"])
        score_ranges.append(metrics["score_range"])
        score_abs_means.append(metrics["score_abs_mean"])
        score_means.append(metrics["score_mean"])

    elapsed = time.perf_counter() - val_start
    out = {
        "valid_loss": _safe_mean(losses),
        "valid_rank_ic": _safe_mean(rank_ics),
        "valid_ic": _safe_mean(ics),
        "valid_topk_ret": _safe_mean(topk_rets),
        "valid_topk_excess_ret": _safe_mean(topk_excess_rets),
        "valid_long_short_spread": _safe_mean(long_short_spreads),
        "valid_topk_hit_rate": _safe_mean(topk_hit_rates),
        "valid_ndcg_at_k": _safe_mean(ndcg_at_ks),
        "valid_topk_true_rank_mean": _safe_mean(topk_true_rank_means),
        "valid_topk_true_rank_score": _safe_mean(topk_true_rank_scores),
        "valid_score_std": _safe_mean(score_stds),
        "valid_score_range": _safe_mean(score_ranges),
        "valid_score_abs_mean": _safe_mean(score_abs_means),
        "valid_score_mean": _safe_mean(score_means),
        "valid_epoch_seconds": float(elapsed),
        "valid_steps_per_second": float(len(loader) / elapsed) if elapsed > 0 else float("nan"),
        "temperature": float(loss_config.temperature),
    }

    if writer is not None and global_step is not None:
        tb_log_metrics(writer, "valid_epoch", out, int(global_step))

    return out



# ============================================================
# Checkpointing
# ============================================================

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    train_config: TrainConfig,
    history: list[dict[str, float]],
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Save model checkpoint."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "train_config": asdict(train_config),
        "history": history,
        "model_config": model.get_config_dict() if hasattr(model, "get_config_dict") else None,
        "extra": extra or {},
    }

    torch.save(payload, p)
    return p


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load checkpoint into model and optionally optimizer."""
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def _is_better(value: float, best: Optional[float], maximize: bool) -> bool:
    if best is None:
        return True
    if not np.isfinite(value):
        return False
    return value > best if maximize else value < best


def _json_safe(obj: Any) -> Any:
    """Convert common Python/numpy/pandas objects to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    """Write JSON with UTF-8 encoding."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, ensure_ascii=False, indent=2)
    return p


def make_run_dir(train_config: TrainConfig) -> Optional[Path]:
    """
    Resolve run output directory.

    If checkpoint_dir is None, no files are written.

    Default layout:
        checkpoint_dir/YYYYMMDD_HHMMSS/

    If run_name is provided:
        checkpoint_dir/run_name/

    If the target folder already exists, suffixes _001, _002, ... are appended.
    """
    if train_config.checkpoint_dir is None:
        return None

    root = Path(train_config.checkpoint_dir)

    if train_config.create_timestamp_run_dir:
        run_name = train_config.run_name or datetime.now().strftime(train_config.timestamp_format)
        base = root / run_name
    else:
        base = root

    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base

    if not train_config.create_timestamp_run_dir:
        base.mkdir(parents=True, exist_ok=True)
        return base

    for i in range(1, 1000):
        candidate = Path(f"{base}_{i:03d}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate

    raise RuntimeError(f"could not create unique run directory under {root}")


def dataset_summary_dict(dataset: Optional[rd.CrossSectionRankDataset]) -> Optional[dict[str, Any]]:
    """Return JSON-safe dataset summary if available."""
    if dataset is None:
        return None
    if hasattr(dataset, "summary"):
        summary = dataset.summary()
        if hasattr(summary, "to_dict"):
            return _json_safe(summary.to_dict())
    return {
        "length": len(dataset),
    }


def save_history_csv(history: list[dict[str, float]], path: str | Path) -> Path:
    """Save training history as CSV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(p, index=False)
    return p


# ============================================================
# Full training
# ============================================================

def fit_model(
    model: nn.Module,
    train_dataset: rd.CrossSectionRankDataset,
    valid_dataset: Optional[rd.CrossSectionRankDataset],
    train_config: TrainConfig = TrainConfig(),
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> pd.DataFrame:
    """
    Fit model and return training history as DataFrame.

    Output layout when train_config.checkpoint_dir is set:
        checkpoint_dir/
          20260503_153012/
            train_config.json
            model_config.json
            dataset_summary.json
            history.csv
            best.pt
            last.pt
            run_summary.json
            tensorboard/
    """
    set_global_seed(train_config.seed)

    device = resolve_device(train_config.device)
    model.to(device)

    optimizer = optimizer or create_optimizer(model, train_config)

    train_loader = make_dataloader(train_dataset, train_config, shuffle=True)
    valid_loader = make_dataloader(valid_dataset, train_config, shuffle=False) if valid_dataset is not None else None

    run_dir = make_run_dir(train_config)
    writer = create_tensorboard_writer(run_dir, train_config)

    if run_dir is not None:
        write_json(run_dir / "train_config.json", asdict(train_config))

        if train_config.save_model_config:
            model_config = model.get_config_dict() if hasattr(model, "get_config_dict") else {}
            write_json(run_dir / "model_config.json", model_config)

        if train_config.save_dataset_summary:
            write_json(
                run_dir / "dataset_summary.json",
                {
                    "train": dataset_summary_dict(train_dataset),
                    "valid": dataset_summary_dict(valid_dataset),
                },
            )

    if writer is not None and train_config.tb_log_config_text:
        try:
            writer.add_text("config/train_config", json.dumps(_json_safe(asdict(train_config)), ensure_ascii=False, indent=2))
            if hasattr(model, "get_config_dict"):
                writer.add_text("config/model_config", json.dumps(_json_safe(model.get_config_dict()), ensure_ascii=False, indent=2))
            writer.add_text(
                "config/dataset_summary",
                json.dumps(
                    _json_safe({
                        "train": dataset_summary_dict(train_dataset),
                        "valid": dataset_summary_dict(valid_dataset),
                    }),
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        except Exception as exc:
            print(f"Warning: failed to write TensorBoard config text: {exc}")

    history: list[dict[str, float]] = []
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    bad_epochs = 0
    stopped_early = False
    global_step = 0

    parameter_count = dtm.count_parameters(model) if hasattr(dtm, "count_parameters") else None

    print(f"device={device}")
    print(f"train_samples={len(train_dataset)}")
    if valid_dataset is not None:
        print(f"valid_samples={len(valid_dataset)}")
    print(f"parameters={parameter_count if parameter_count is not None else 'unknown'}")
    if run_dir is not None:
        print(f"run_dir={run_dir}")
    if writer is not None:
        print(f"tensorboard_dir={run_dir / train_config.tensorboard_dirname}")

    tb_add_scalar(writer, "meta/parameter_count", parameter_count, 0)
    tb_log_memory(writer, device, 0, prefix="memory_epoch")

    for epoch in range(1, int(train_config.max_epochs) + 1):
        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train_config=train_config,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
        )

        if valid_loader is not None:
            valid_metrics = validate_one_epoch(
                model=model,
                loader=valid_loader,
                device=device,
                train_config=train_config,
                epoch=epoch,
                writer=writer,
                global_step=global_step,
            )
        else:
            valid_metrics = {}

        row = {
            "epoch": float(epoch),
            "global_step": float(global_step),
            **train_metrics,
            **valid_metrics,
        }
        history.append(row)

        if writer is not None and train_config.tb_log_epoch_metrics:
            tb_log_metrics(writer, "train_epoch", {k.replace("train_", ""): v for k, v in train_metrics.items()}, epoch)
            tb_log_metrics(writer, "valid_epoch", {k.replace("valid_", ""): v for k, v in valid_metrics.items()}, epoch)
            tb_add_scalar(writer, "epoch/global_step", global_step, epoch)
            tb_log_memory(writer, device, epoch, prefix="memory_epoch")
            if train_config.tb_log_grad_norm:
                tb_log_grad_and_param_norms(writer, model, epoch, prefix="norms_epoch")
            if train_config.tb_log_histograms and epoch % int(train_config.tb_histogram_every_epochs) == 0:
                tb_log_model_histograms(writer, model, epoch)
            writer.flush()

        metric_name = train_config.metric_for_best
        metric_value = row.get(metric_name)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={row.get('train_loss', float('nan')):.6f} "
            f"train_rank_ic={row.get('train_rank_ic', float('nan')):.4f} "
            f"valid_loss={row.get('valid_loss', float('nan')):.6f} "
            f"valid_rank_ic={row.get('valid_rank_ic', float('nan')):.4f} "
            f"valid_ic={row.get('valid_ic', float('nan')):.4f} "
            f"valid_topk_ret={row.get('valid_topk_ret', float('nan')):.6f} "
            f"valid_hit@k={row.get('valid_topk_hit_rate', float('nan')):.3f} "
            f"valid_ndcg@k={row.get('valid_ndcg_at_k', float('nan')):.3f} "
            f"valid_rankmean@k={row.get('valid_topk_true_rank_mean', float('nan')):.1f} "
            f"valid_score_std={row.get('valid_score_std', float('nan')):.4g} "
            f"gpu_max_alloc={row.get('gpu_max_allocated_mb', float('nan')):.0f}MB "
            f"tau={row.get('temperature', float('nan')):.4f}"
        )

        if run_dir is not None and train_config.save_history_each_epoch:
            save_history_csv(history, run_dir / "history.csv")

        if run_dir is not None and train_config.save_last:
            save_checkpoint(
                run_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_config=train_config,
                history=history,
                extra={"run_dir": str(run_dir)},
            )

        if metric_value is not None and train_config.save_best:
            if _is_better(float(metric_value), best_metric, bool(train_config.maximize_metric)):
                best_metric = float(metric_value)
                best_epoch = int(epoch)
                bad_epochs = 0
                if run_dir is not None:
                    save_checkpoint(
                        run_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        train_config=train_config,
                        history=history,
                        extra={
                            "best_metric": best_metric,
                            "best_epoch": best_epoch,
                            "metric_name": metric_name,
                            "run_dir": str(run_dir),
                        },
                    )
            else:
                bad_epochs += 1

        if train_config.early_stopping_patience is not None and valid_loader is not None:
            if bad_epochs >= int(train_config.early_stopping_patience):
                print(f"early stopping at epoch={epoch}, bad_epochs={bad_epochs}")
                stopped_early = True
                break

    history_df = pd.DataFrame(history)

    if run_dir is not None:
        save_history_csv(history, run_dir / "history.csv")

        write_json(
            run_dir / "run_summary.json",
            {
                "run_dir": str(run_dir),
                "tensorboard_dir": str(run_dir / train_config.tensorboard_dirname),
                "completed_epochs": len(history),
                "completed_global_steps": global_step,
                "stopped_early": stopped_early,
                "metric_for_best": train_config.metric_for_best,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "final_metrics": history[-1] if history else {},
                "files": {
                    "train_config": "train_config.json",
                    "model_config": "model_config.json",
                    "dataset_summary": "dataset_summary.json",
                    "history": "history.csv",
                    "best_checkpoint": "best.pt",
                    "last_checkpoint": "last.pt",
                    "tensorboard": train_config.tensorboard_dirname,
                },
            },
        )
        history_df.attrs["run_dir"] = str(run_dir)
        history_df.attrs["tensorboard_dir"] = str(run_dir / train_config.tensorboard_dirname)

    if writer is not None:
        writer.flush()
        writer.close()

    return history_df




# ============================================================
# Convenience constructors
# ============================================================

def build_model_from_bundle(
    bundle: Any,
    seq_len: int = 128,
    **model_kwargs: Any,
) -> dtm.DualTransformerRanker:
    """Create DualTransformerRanker from FeatureLabelBundle metadata."""
    if not hasattr(bundle, "metadata"):
        raise TypeError("bundle must have metadata")
    ts_names = list(bundle.metadata["ts_factor_names"])
    scalar_names = list(bundle.metadata["scalar_factor_names"])

    return dtm.DualTransformerRanker.from_feature_counts(
        num_ts_factors=len(ts_names),
        num_scalar_factors=len(scalar_names),
        seq_len=int(seq_len),
        **model_kwargs,
    )


def build_datasets_from_bundle(
    bundle: Any,
    ds_config: rd.CrossSectionDatasetConfig,
    train_end: Optional[Any] = None,
    valid_end: Optional[Any] = None,
    valid_ratio: float = 0.2,
) -> tuple[rd.CrossSectionRankDataset, rd.CrossSectionRankDataset]:
    """Create train/valid datasets from bundle."""
    return rd.make_train_valid_datasets_from_bundle(
        bundle=bundle,
        config=ds_config,
        train_end=train_end,
        valid_end=valid_end,
        valid_ratio=valid_ratio,
    )


__all__ = [
    "TrainConfig",
    "set_global_seed",
    "resolve_device",
    "make_dataloader",
    "move_batch_to_device",
    "build_loss_config",
    "create_optimizer",
    "compute_batch_metrics",
    "train_one_epoch",
    "validate_one_epoch",
    "save_checkpoint",
    "load_checkpoint",
    "fit_model",
    "build_model_from_bundle",
    "build_datasets_from_bundle",
]
