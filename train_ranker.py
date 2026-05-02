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

    log_every_steps: int = 50
    use_amp: bool = False


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
    )


def create_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Create AdamW optimizer."""
    return AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )


# ============================================================
# Metrics
# ============================================================

def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


@torch.no_grad()
def compute_batch_metrics(
    scores: torch.Tensor,
    y: torch.Tensor,
    y_raw: torch.Tensor,
    topk: int = 20,
) -> dict[str, float]:
    """Compute metrics for one batch."""
    rank_ic = rl.rank_ic(scores, y).detach().float().mean().item()
    ic = rl.information_coefficient(scores, y_raw).detach().float().mean().item()
    topk_ret = rl.topk_mean_return(scores, y_raw, k=int(topk)).detach().float().mean().item()

    return {
        "rank_ic": float(rank_ic),
        "ic": float(ic),
        "topk_ret": float(topk_ret),
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
) -> dict[str, float]:
    """Train one epoch."""
    model.train()

    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)

    loss_config = build_loss_config(train_config, epoch)

    use_amp = bool(train_config.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp and hasattr(torch, "amp") else None

    losses: list[float] = []
    rank_ics: list[float] = []
    ics: list[float] = []
    topk_rets: list[float] = []

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                scores = model(batch["x_ts"], batch["x_scalar"])
                loss = rl.rank_loss(scores, batch["y"], config=loss_config)

            assert scaler is not None
            scaler.scale(loss).backward()

            if train_config.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.grad_clip_norm))

            scaler.step(optimizer)
            scaler.update()
        else:
            scores = model(batch["x_ts"], batch["x_scalar"])
            loss = rl.rank_loss(scores, batch["y"], config=loss_config)
            loss.backward()

            if train_config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.grad_clip_norm))

            optimizer.step()

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

        if train_config.log_every_steps and step % int(train_config.log_every_steps) == 0:
            print(
                f"epoch={epoch:03d} step={step:05d} "
                f"loss={_safe_mean(losses):.6f} "
                f"rank_ic={_safe_mean(rank_ics):.4f} "
                f"ic={_safe_mean(ics):.4f} "
                f"topk_ret={_safe_mean(topk_rets):.6f} "
                f"tau={loss_config.temperature:.4f}"
            )

    return {
        "train_loss": _safe_mean(losses),
        "train_rank_ic": _safe_mean(rank_ics),
        "train_ic": _safe_mean(ics),
        "train_topk_ret": _safe_mean(topk_rets),
        "temperature": float(loss_config.temperature),
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    train_config: TrainConfig,
    epoch: int,
) -> dict[str, float]:
    """Validate one epoch."""
    model.eval()

    loss_config = build_loss_config(train_config, epoch)

    losses: list[float] = []
    rank_ics: list[float] = []
    ics: list[float] = []
    topk_rets: list[float] = []

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

    return {
        "valid_loss": _safe_mean(losses),
        "valid_rank_ic": _safe_mean(rank_ics),
        "valid_ic": _safe_mean(ics),
        "valid_topk_ret": _safe_mean(topk_rets),
        "temperature": float(loss_config.temperature),
    }


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

    If train_config.run_name is provided, it is used instead of timestamp.
    """
    set_global_seed(train_config.seed)

    device = resolve_device(train_config.device)
    model.to(device)

    optimizer = optimizer or create_optimizer(model, train_config)

    train_loader = make_dataloader(train_dataset, train_config, shuffle=True)
    valid_loader = make_dataloader(valid_dataset, train_config, shuffle=False) if valid_dataset is not None else None

    run_dir = make_run_dir(train_config)

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

    history: list[dict[str, float]] = []
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    bad_epochs = 0
    stopped_early = False

    print(f"device={device}")
    print(f"train_samples={len(train_dataset)}")
    if valid_dataset is not None:
        print(f"valid_samples={len(valid_dataset)}")
    print(f"parameters={dtm.count_parameters(model) if hasattr(dtm, 'count_parameters') else 'unknown'}")
    if run_dir is not None:
        print(f"run_dir={run_dir}")

    for epoch in range(1, int(train_config.max_epochs) + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train_config=train_config,
            epoch=epoch,
        )

        if valid_loader is not None:
            valid_metrics = validate_one_epoch(
                model=model,
                loader=valid_loader,
                device=device,
                train_config=train_config,
                epoch=epoch,
            )
        else:
            valid_metrics = {}

        row = {
            "epoch": float(epoch),
            **train_metrics,
            **valid_metrics,
        }
        history.append(row)

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
        # Ensure final history exists even if save_history_each_epoch=False.
        save_history_csv(history, run_dir / "history.csv")

        write_json(
            run_dir / "run_summary.json",
            {
                "run_dir": str(run_dir),
                "completed_epochs": len(history),
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
                },
            },
        )
        history_df.attrs["run_dir"] = str(run_dir)

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
