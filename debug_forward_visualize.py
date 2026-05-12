"""
debug_forward_visualize_attention.py

Forward visualization debugger with intra-stock attention hooks.

This version extends debug_forward_visualize_ompfix.py by capturing internal
states inside IntraStockTokenCompressor:

    x_ts
    temporal_tokens
    intra_stock_input_tokens          # before positional encoding/self-attention
    factor_self_block_0_out           # after self-attention block
    reduce_cross_out_pre_norm         # after reduce cross-attention, before out_norm
    reduce_out_norm_input             # alias/checkpoint before out_norm
    reduce_out_norm_output            # after out_norm, before reshape
    reduced_tokens                    # final [B,N,R,D]
    stock_embedding
    aggregate_cross_out               # optional, if module exists
    broadcast_cross_out               # optional, if module exists
    cross_embedding
    scores

It plots individual stock heatmaps and pairwise stock-difference heatmaps.

Windows/Conda OpenMP workaround is included because matplotlib + torch + MKL can
trigger "OMP: Error #15: Initializing libiomp5md.dll..." in some environments.

Example
-------
python debug_forward_visualize_attention.py ^
  --config train_run_config_example.json ^
  --checkpoint checkpoints/dual_transformer_ranker/20260511_215509/last.pt ^
  --split valid ^
  --batch-index 0 ^
  --num-stocks 2 ^
  --mode both
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

import factor_pipeline as fp
import rank_dataset as rd
import train_ranker as tr


# ============================================================
# Loading helpers
# ============================================================

def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return p


def pick(cfg: dict[str, Any], name: str, default: Any) -> Any:
    return cfg[name] if name in cfg else default


def load_checkpoint_payload(path: Optional[str]) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        return payload
    return {"model_state_dict": payload}


def cleaned_model_config_from_checkpoint(payload: Optional[dict[str, Any]]) -> dict[str, Any]:
    if payload is None:
        return {}
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        return {}
    out = dict(model_config)
    for key in ["num_ts_factors", "num_scalar_factors", "seq_len"]:
        out.pop(key, None)
    return out


def model_kwargs_from_json(cfg: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "dropout", "activation",
        "depthwise_layers", "depthwise_kernel_size", "temporal_compressed_len",
        "mixed_channels_1", "mixed_channels_2", "temporal_out_channels",
        "mixed_kernel_size", "group_norm_groups",
        "factor_num_layers", "factor_num_heads", "factor_ff_dim",
        "factor_reduce_tokens", "factor_use_positional_encoding",
        "stock_aggregate_tokens", "cross_num_heads", "cross_ff_dim",
        "aggregate_residual_query", "broadcast_residual_query",
        "use_scalar_factors",
        "score_hidden_dim", "score_head_layers",
        "model_dim", "cross_num_layers",
        "temporal_hidden_channels", "temporal_channels",
    ]
    return {k: cfg[k] for k in keys if k in cfg}


def make_datasets(cfg: dict[str, Any]):
    bundle_path = pick(cfg, "bundle_path", "data/cache/stage1_factor_label_bundle.pkl")
    bundle = fp.load_bundle(bundle_path)

    ds_config = rd.CrossSectionDatasetConfig(
        sample_size=int(pick(cfg, "sample_size", 512)),
        seq_len=int(pick(cfg, "seq_len", 128)),
        samples_per_date=int(pick(cfg, "samples_per_date", 1)),
        label_name=str(pick(cfg, "label_name", "label_ret_t1_t6")),
        label_valid_name=pick(cfg, "label_valid_name", "label_valid_t1_t6"),
        target_mode=str(pick(cfg, "target_mode", "rank_pct")),
        require_full_history=bool(pick(cfg, "require_full_history", True)),
        allow_smaller_sample=bool(pick(cfg, "allow_smaller_sample", False)),
        return_tensors="torch",
        random_seed=int(pick(cfg, "seed", 42)),
        candidate_pool_size=pick(cfg, "candidate_pool_size", None),
    )

    train_ds, valid_ds = tr.build_datasets_from_bundle(
        bundle,
        ds_config=ds_config,
        train_end=pick(cfg, "train_end", None),
        valid_end=pick(cfg, "valid_end", None),
        valid_ratio=float(pick(cfg, "valid_ratio", 0.2)),
    )
    return bundle, train_ds, valid_ds


def load_model(cfg: dict[str, Any], bundle: Any, checkpoint: Optional[str]) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = load_checkpoint_payload(checkpoint)
    ckpt_kwargs = cleaned_model_config_from_checkpoint(payload)
    if ckpt_kwargs:
        model_kwargs = ckpt_kwargs
        print("Using model_config from checkpoint to build model.")
    else:
        model_kwargs = model_kwargs_from_json(cfg)
        print("Checkpoint has no model_config; using model kwargs from JSON config.")

    print(json.dumps(model_kwargs, ensure_ascii=False, indent=2)[:5000])

    model = tr.build_model_from_bundle(
        bundle,
        seq_len=int(pick(cfg, "seq_len", 128)),
        **model_kwargs,
    )

    if payload is not None:
        state = payload.get("model_state_dict", payload)
        model.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint: {checkpoint}")

    return model, model_kwargs


# ============================================================
# Forward hook capture
# ============================================================

class ForwardCapture:
    """Capture internal activations with forward hooks."""

    def __init__(self, model: torch.nn.Module, batch_size: int, num_stocks: int) -> None:
        self.model = model
        self.batch_size = int(batch_size)
        self.num_stocks = int(num_stocks)
        self.cache: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []

    def _detach(self, x: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(x):
            return x.detach().float().cpu()
        if isinstance(x, (tuple, list)) and x and torch.is_tensor(x[0]):
            return x[0].detach().float().cpu()
        return None

    def _reshape_bn(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [B*N, ...] back to [B,N,...] when possible."""
        b, n = self.batch_size, self.num_stocks
        if x.ndim >= 2 and x.shape[0] == b * n:
            return x.reshape(b, n, *x.shape[1:])
        return x

    def _store(self, name: str, x: Any, reshape_bn: bool = False) -> None:
        t = self._detach(x)
        if t is None:
            return
        if reshape_bn:
            t = self._reshape_bn(t)
        self.cache[name] = t

    def register(self) -> None:
        comp = getattr(self.model, "intra_stock_compressor", None)
        if comp is not None:
            # Before any positional encoding / self-attention inside compressor.
            self.handles.append(
                comp.register_forward_pre_hook(
                    lambda module, inputs: self._store("intra_stock_input_tokens", inputs[0], reshape_bn=False)
                )
            )

            blocks = getattr(comp, "factor_self_blocks", None)
            if blocks is not None:
                for i, block in enumerate(blocks):
                    self.handles.append(
                        block.register_forward_hook(
                            lambda module, inputs, output, i=i: self._store(
                                f"factor_self_block_{i}_out",
                                output,
                                reshape_bn=True,
                            )
                        )
                    )

            reduce_cross = getattr(comp, "reduce_cross", None)
            if reduce_cross is not None:
                self.handles.append(
                    reduce_cross.register_forward_hook(
                        lambda module, inputs, output: self._store(
                            "reduce_cross_out_pre_norm",
                            output,
                            reshape_bn=True,
                        )
                    )
                )

            out_norm = getattr(comp, "out_norm", None)
            if out_norm is not None:
                self.handles.append(
                    out_norm.register_forward_pre_hook(
                        lambda module, inputs: self._store(
                            "reduce_out_norm_input",
                            inputs[0],
                            reshape_bn=True,
                        )
                    )
                )
                self.handles.append(
                    out_norm.register_forward_hook(
                        lambda module, inputs, output: self._store(
                            "reduce_out_norm_output",
                            output,
                            reshape_bn=True,
                        )
                    )
                )

        agg = getattr(self.model, "stock_aggregator", None)
        if agg is not None:
            aggregate_cross = getattr(agg, "aggregate_cross", None)
            if aggregate_cross is not None:
                self.handles.append(
                    aggregate_cross.register_forward_hook(
                        lambda module, inputs, output: self._store("aggregate_cross_out", output, reshape_bn=False)
                    )
                )
            broadcast_cross = getattr(agg, "broadcast_cross", None)
            if broadcast_cross is not None:
                self.handles.append(
                    broadcast_cross.register_forward_hook(
                        lambda module, inputs, output: self._store("broadcast_cross_out", output, reshape_bn=False)
                    )
                )

    def remove(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()


# ============================================================
# Stats helpers
# ============================================================

def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def cross_stock_stats(name: str, x: torch.Tensor) -> dict[str, float | str]:
    x = x.detach().float().cpu()
    if x.ndim < 2:
        arr = x.reshape(1, -1)
    elif x.ndim == 2:
        arr = x
    else:
        arr = x.reshape(x.shape[0], x.shape[1], -1)

    if arr.ndim == 2:
        std_by_b = arr.std(dim=1, unbiased=False)
        range_by_b = arr.max(dim=1).values - arr.min(dim=1).values
        abs_mean = arr.abs().mean()
        mean = arr.mean()
    else:
        std_by_b = arr.std(dim=1, unbiased=False).mean(dim=1)
        range_by_b = (arr.max(dim=1).values - arr.min(dim=1).values).mean(dim=1)
        abs_mean = arr.abs().mean()
        mean = arr.mean()

    return {
        f"{name}_shape": str(tuple(x.shape)),
        f"{name}_cross_stock_std_mean": float(std_by_b.mean().item()),
        f"{name}_cross_stock_std_min": float(std_by_b.min().item()),
        f"{name}_cross_stock_std_max": float(std_by_b.max().item()),
        f"{name}_cross_stock_range_mean": float(range_by_b.mean().item()),
        f"{name}_abs_mean": float(abs_mean.item()),
        f"{name}_mean": float(mean.item()),
        f"{name}_finite_ratio": float(torch.isfinite(x).float().mean().item()),
    }


def print_stats(stats: dict[str, float | str]) -> None:
    for k, v in stats.items():
        if isinstance(v, str):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.8g}")


def infer_token_grid_from_vector(vec: torch.Tensor, ref_tokens: Optional[torch.Tensor]) -> np.ndarray:
    arr = tensor_to_numpy(vec)
    if ref_tokens is not None and ref_tokens.ndim == 2:
        r, d = int(ref_tokens.shape[0]), int(ref_tokens.shape[1])
        if arr.size == r * d:
            return arr.reshape(r, d)
    return arr.reshape(1, -1)


# ============================================================
# Plot helpers
# ============================================================

def save_heatmap(
    arr: np.ndarray,
    path: Path,
    title: str,
    xlabel: str = "time / dim",
    ylabel: str = "channel / token",
    center_zero: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4))
    if center_zero:
        vmax = np.nanmax(np.abs(arr))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        vmin = -vmax
    else:
        vmin = np.nanpercentile(arr, 1)
        vmax = np.nanpercentile(arr, 99)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = None, None

    im = ax.imshow(arr, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_line(arr: np.ndarray, path: Path, title: str, xlabel: str = "index", ylabel: str = "value") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr, dtype=float).reshape(-1)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(arr)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_score_distribution(scores: torch.Tensor, stock_indices: list[int], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor_to_numpy(scores).reshape(-1)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(arr, marker=".", linewidth=0.8, markersize=2)
    for idx in stock_indices:
        ax.axvline(idx, linestyle="--", linewidth=1)
        ax.text(idx, arr[idx], f" s{idx}", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("stock index within sampled cross-section")
    ax.set_ylabel("score")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_layer_std_bar(stats_by_layer: dict[str, dict[str, float | str]], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names, vals = [], []
    for name, stats in stats_by_layer.items():
        key = f"{name}_cross_stock_std_mean"
        if key in stats and isinstance(stats[key], float):
            names.append(name)
            vals.append(max(float(stats[key]), 1e-12))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(names, vals)
    ax.set_title(title)
    ax.set_ylabel("cross-stock std mean")
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def layer_tensor_for_stock(t: torch.Tensor, b_idx: int, stock_idx: int, reduced_ref: Optional[torch.Tensor]) -> np.ndarray:
    """
    Convert a layer tensor to a 2D grid for one stock.

    Supported:
        [B,N,A,D] -> [A,D]
        [B,N,D] -> reshape to [R,D] if possible, else [1,D]
        [B,K,D] without stock dimension -> not called for stock-specific plots
    """
    if t.ndim == 4:
        return tensor_to_numpy(t[b_idx, stock_idx])
    if t.ndim == 3:
        return infer_token_grid_from_vector(t[b_idx, stock_idx], reduced_ref)
    if t.ndim == 2:
        return tensor_to_numpy(t[b_idx, stock_idx]).reshape(1, -1)
    return tensor_to_numpy(t).reshape(1, -1)


def plot_stock_layers(
    out_dir: Path,
    mode: str,
    b_idx: int,
    stock_idx: int,
    batch: dict[str, Any],
    layer_map: dict[str, torch.Tensor],
    ordered_layers: list[str],
) -> list[str]:
    saved = []
    prefix = out_dir / mode / f"sample_{b_idx:02d}" / f"stock_{stock_idx:04d}"

    # x_ts from batch.
    path = prefix / "00_x_ts_heatmap.png"
    save_heatmap(
        tensor_to_numpy(batch["x_ts"][b_idx, stock_idx]),
        path,
        title=f"{mode} sample={b_idx} stock={stock_idx} x_ts [F_ts,T]",
        xlabel="time",
        ylabel="raw ts factor",
    )
    saved.append(str(path))

    reduced_ref = layer_map.get("reduced_tokens", None)
    reduced_one = reduced_ref[b_idx, stock_idx] if reduced_ref is not None and reduced_ref.ndim >= 4 else None

    for idx, key in enumerate(ordered_layers, start=1):
        if key not in layer_map:
            continue
        t = layer_map[key]
        # Skip non-stock latent layers for individual stock heatmaps.
        if t.ndim >= 3 and t.shape[1] != batch["x_ts"].shape[1]:
            continue

        arr = layer_tensor_for_stock(t, b_idx, stock_idx, reduced_one)
        path = prefix / f"{idx:02d}_{key}_heatmap.png"
        save_heatmap(arr, path, title=f"{mode} sample={b_idx} stock={stock_idx} {key}")
        saved.append(str(path))

        if arr.size > 64:
            path_line = prefix / f"{idx:02d}_{key}_line.png"
            save_line(arr.reshape(-1), path_line, title=f"{mode} sample={b_idx} stock={stock_idx} {key} flat")
            saved.append(str(path_line))

    return saved


def plot_pair_differences(
    out_dir: Path,
    mode: str,
    b_idx: int,
    stock_a: int,
    stock_b: int,
    batch: dict[str, Any],
    layer_map: dict[str, torch.Tensor],
    ordered_layers: list[str],
) -> list[str]:
    saved = []
    prefix = out_dir / mode / f"sample_{b_idx:02d}" / f"diff_stock_{stock_a:04d}_minus_{stock_b:04d}"

    path = prefix / "00_x_ts_diff_heatmap.png"
    save_heatmap(
        tensor_to_numpy(batch["x_ts"][b_idx, stock_a] - batch["x_ts"][b_idx, stock_b]),
        path,
        title=f"{mode} sample={b_idx} stock {stock_a}-{stock_b} x_ts diff",
        center_zero=True,
    )
    saved.append(str(path))

    reduced_ref = layer_map.get("reduced_tokens", None)
    reduced_one = reduced_ref[b_idx, stock_a] if reduced_ref is not None and reduced_ref.ndim >= 4 else None

    for idx, key in enumerate(ordered_layers, start=1):
        if key not in layer_map:
            continue
        t = layer_map[key]
        if t.ndim >= 3 and t.shape[1] != batch["x_ts"].shape[1]:
            continue

        if t.ndim == 4:
            diff = t[b_idx, stock_a] - t[b_idx, stock_b]
            arr = tensor_to_numpy(diff)
        elif t.ndim == 3:
            diff = t[b_idx, stock_a] - t[b_idx, stock_b]
            arr = infer_token_grid_from_vector(diff, reduced_one)
        elif t.ndim == 2:
            diff = t[b_idx, stock_a] - t[b_idx, stock_b]
            arr = tensor_to_numpy(diff).reshape(1, -1)
        else:
            continue

        path = prefix / f"{idx:02d}_{key}_diff_heatmap.png"
        save_heatmap(arr, path, title=f"{mode} sample={b_idx} stock {stock_a}-{stock_b} {key} diff", center_zero=True)
        saved.append(str(path))

        if arr.size > 64:
            path_line = prefix / f"{idx:02d}_{key}_diff_line.png"
            save_line(arr.reshape(-1), path_line, title=f"{mode} sample={b_idx} stock {stock_a}-{stock_b} {key} diff flat", ylabel="diff")
            saved.append(str(path_line))

    return saved


# ============================================================
# Main
# ============================================================

def choose_indices(total: int, count: int, seed: int, explicit: Optional[str]) -> list[int]:
    if explicit:
        vals = [int(x.strip()) for x in explicit.split(",") if x.strip()]
        return [v for v in vals if 0 <= v < total]
    rng = random.Random(seed)
    count = min(int(count), int(total))
    return sorted(rng.sample(range(total), count))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="valid", choices=["train", "valid"])
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", default="both", choices=["eval", "train", "both"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-stocks", type=int, default=2)
    parser.add_argument("--stock-indices", default=None, help="Comma-separated stock indices, e.g. 3,17")
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_json(args.config)
    bundle, train_ds, valid_ds = make_datasets(cfg)
    ds = valid_ds if args.split == "valid" else train_ds

    loader = tr.make_dataloader(
        ds,
        tr.TrainConfig(
            batch_size=int(pick(cfg, "batch_size", 1)),
            num_workers=0,
            device=args.device,
            topk_metric_k=int(pick(cfg, "topk_metric_k", 20)),
        ),
        shuffle=False,
    )

    batch = None
    for i, b in enumerate(loader):
        if i == int(args.batch_index):
            batch = b
            break
    if batch is None:
        raise RuntimeError(f"batch-index {args.batch_index} not found")

    device = torch.device(args.device)
    model, model_kwargs = load_model(cfg, bundle, args.checkpoint)
    model = model.to(device)
    batch = tr.move_batch_to_device(batch, device)

    bsz, n_stocks = int(batch["x_ts"].shape[0]), int(batch["x_ts"].shape[1])

    if args.sample_index is None:
        rng = random.Random(args.seed)
        b_idx = rng.randrange(bsz)
    else:
        b_idx = int(args.sample_index)
        if not (0 <= b_idx < bsz):
            raise ValueError(f"sample-index must be in [0, {bsz}), got {b_idx}")

    stock_indices = choose_indices(
        total=n_stocks,
        count=int(args.num_stocks),
        seed=int(args.seed) + 1009,
        explicit=args.stock_indices,
    )
    if len(stock_indices) < 1:
        raise RuntimeError("no valid stock indices selected")

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif args.checkpoint is not None:
        out_dir = Path(args.checkpoint).resolve().parent / "forward_visual_debug_attention"
    else:
        out_dir = Path("forward_visual_debug_attention") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Visual debug selection ===")
    print("split:", args.split)
    print("batch_index:", args.batch_index)
    print("sample_index within batch:", b_idx)
    print("stock_indices:", stock_indices)
    print("out_dir:", out_dir)

    print("\n=== Dataset / target diagnostics ===")
    for name in ["x_ts", "x_scalar", "y", "y_raw"]:
        if name in batch and torch.is_tensor(batch[name]):
            print(f"\n-- {name} --")
            print_stats(cross_stock_stats(name, batch[name]))

    ordered_layers = [
        "temporal_tokens",
        "intra_stock_input_tokens",
        "factor_self_block_0_out",
        "factor_self_block_1_out",
        "factor_self_block_2_out",
        "reduce_cross_out_pre_norm",
        "reduce_out_norm_input",
        "reduce_out_norm_output",
        "reduced_tokens",
        "stock_embedding",
        "aggregate_cross_out",
        "broadcast_cross_out",
        "cross_embedding",
        "final_embedding",
        "scores",
    ]

    modes = ["eval", "train"] if args.mode == "both" else [args.mode]
    manifest: dict[str, Any] = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "batch_index": args.batch_index,
        "sample_index": b_idx,
        "stock_indices": stock_indices,
        "model_kwargs": model_kwargs,
        "outputs": {},
    }

    for mode in modes:
        print(f"\n=== Forward diagnostics and plots: model.{mode}() ===")
        if mode == "eval":
            model.eval()
        else:
            model.train()
            torch.manual_seed(args.seed)

        capture = ForwardCapture(model, batch_size=bsz, num_stocks=n_stocks)
        capture.register()
        try:
            with torch.no_grad():
                out = model(batch["x_ts"], batch.get("x_scalar"), return_dict=True)
        finally:
            capture.remove()

        # Merge hook captures and model return_dict. Return_dict wins only if same key already exists.
        layer_map: dict[str, torch.Tensor] = {}
        layer_map.update(capture.cache)
        layer_map.update({k: v.detach().float().cpu() for k, v in out.items() if torch.is_tensor(v)})

        stats_by_layer: dict[str, dict[str, float | str]] = {}
        for key in ordered_layers:
            if key in layer_map:
                stats_by_layer[key] = cross_stock_stats(key, layer_map[key])
                print(f"\n-- {key} --")
                print_stats(stats_by_layer[key])

        mode_outputs = []

        path = out_dir / mode / f"sample_{b_idx:02d}" / "layer_cross_stock_std_bar.png"
        save_layer_std_bar(stats_by_layer, path, title=f"{mode} layer cross-stock std mean")
        mode_outputs.append(str(path))

        if "scores" in layer_map:
            path = out_dir / mode / f"sample_{b_idx:02d}" / "score_distribution.png"
            save_score_distribution(
                layer_map["scores"][b_idx],
                stock_indices=stock_indices,
                path=path,
                title=f"{mode} sample={b_idx} score distribution across stocks",
            )
            mode_outputs.append(str(path))

        for stock_idx in stock_indices:
            mode_outputs.extend(plot_stock_layers(out_dir, mode, b_idx, stock_idx, batch, layer_map, ordered_layers))

        if len(stock_indices) >= 2:
            mode_outputs.extend(
                plot_pair_differences(out_dir, mode, b_idx, stock_indices[0], stock_indices[1], batch, layer_map, ordered_layers)
            )

        if "scores" in layer_map:
            metrics = tr.compute_batch_metrics(
                layer_map["scores"].to(device),
                batch["y"].detach().float(),
                batch["y_raw"].detach().float(),
                topk=int(pick(cfg, "topk_metric_k", 20)),
            )
        else:
            metrics = {}

        print("\nMetrics:")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

        manifest["outputs"][mode] = {
            "stats_by_layer": stats_by_layer,
            "metrics": metrics,
            "files": mode_outputs,
        }

    manifest_path = save_json(out_dir / "manifest.json", manifest)

    md_lines = [
        "# Forward visual debug with attention internals",
        "",
        f"- split: `{args.split}`",
        f"- batch_index: `{args.batch_index}`",
        f"- sample_index: `{b_idx}`",
        f"- stock_indices: `{stock_indices}`",
        f"- checkpoint: `{args.checkpoint}`",
        "",
        "## Files",
        "",
    ]
    for mode, info in manifest["outputs"].items():
        md_lines.append(f"### {mode}")
        md_lines.append("")
        for f in info["files"]:
            rel = Path(f).relative_to(out_dir)
            md_lines.append(f"- `{rel}`")
        md_lines.append("")

    index_path = out_dir / "index.md"
    index_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("\nSaved visual debug outputs:")
    print("manifest:", manifest_path)
    print("index:", index_path)


if __name__ == "__main__":
    main()
