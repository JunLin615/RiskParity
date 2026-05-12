"""
debug_forward_variance_v2.py

Layer-wise forward variance debugger for the cross-sectional ranker.

V2 fix
------
The previous version built the model from the JSON config before loading the
checkpoint. If the JSON did not contain all model fields used in a past run,
the newly built model could have different shapes from the checkpoint.

This version first reads checkpoint["model_config"] and uses it to instantiate
the model, then loads the checkpoint. This guarantees architecture consistency
for saved runs.

Example
-------
python debug_forward_variance_v2.py \
  --config build_and_train_config_earlystop_example.json \
  --checkpoint checkpoints/dual_transformer_ranker/20260511_213652/last.pt \
  --split valid \
  --batch-index 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch

import factor_pipeline as fp
import rank_dataset as rd
import train_ranker as tr


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def pick(cfg: dict[str, Any], name: str, default: Any) -> Any:
    return cfg[name] if name in cfg else default


def cross_stock_stats(name: str, x: torch.Tensor) -> dict[str, float | str]:
    """
    Compute cross-sectional statistics.

    x can be:
        [B, N]
        [B, N, ...]
    """
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


def load_checkpoint_payload(path: Optional[str]) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        return payload
    return {"model_state_dict": payload}


def cleaned_model_config_from_checkpoint(payload: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Extract model_config from checkpoint and remove constructor-supplied keys.

    build_model_from_bundle() already supplies:
        num_ts_factors
        num_scalar_factors
        seq_len
    so these keys must not be passed again as **kwargs.
    """
    if payload is None:
        return {}

    model_config = payload.get("model_config")
    if model_config is None:
        return {}

    if not isinstance(model_config, dict):
        return {}

    cfg = dict(model_config)
    for key in ["num_ts_factors", "num_scalar_factors", "seq_len"]:
        cfg.pop(key, None)
    return cfg


def model_kwargs_from_json(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Fallback model kwargs from JSON when no checkpoint model_config exists.
    """
    keys = [
        "dropout",
        "activation",
        "depthwise_layers",
        "depthwise_kernel_size",
        "temporal_compressed_len",
        "mixed_channels_1",
        "mixed_channels_2",
        "temporal_out_channels",
        "mixed_kernel_size",
        "group_norm_groups",
        "factor_num_layers",
        "factor_num_heads",
        "factor_ff_dim",
        "factor_reduce_tokens",
        "factor_use_positional_encoding",
        "stock_aggregate_tokens",
        "cross_num_heads",
        "cross_ff_dim",
        "aggregate_residual_query",
        "broadcast_residual_query",
        "use_scalar_factors",
        "score_hidden_dim",
        "score_head_layers",
        "model_dim",
        "cross_num_layers",
        "temporal_hidden_channels",
        "temporal_channels",
    ]
    return {k: cfg[k] for k in keys if k in cfg}


def load_model(
    cfg: dict[str, Any],
    bundle: Any,
    checkpoint: Optional[str],
) -> torch.nn.Module:
    payload = load_checkpoint_payload(checkpoint)

    ckpt_kwargs = cleaned_model_config_from_checkpoint(payload)
    if ckpt_kwargs:
        model_kwargs = ckpt_kwargs
        print("Using model_config from checkpoint to build model.")
        print(json.dumps(model_kwargs, ensure_ascii=False, indent=2)[:5000])
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

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="valid", choices=["train", "valid"])
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
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
    model = load_model(cfg, bundle, args.checkpoint).to(device)
    batch = tr.move_batch_to_device(batch, device)

    print("\n=== Dataset / target diagnostics ===")
    print("split:", args.split)
    print("batch_index:", args.batch_index)
    print_stats(cross_stock_stats("x_ts", batch["x_ts"]))
    if "x_scalar" in batch and torch.is_tensor(batch["x_scalar"]):
        print_stats(cross_stock_stats("x_scalar", batch["x_scalar"]))
    print_stats(cross_stock_stats("y", batch["y"]))
    print_stats(cross_stock_stats("y_raw", batch["y_raw"]))

    for mode in ["eval", "train"]:
        print(f"\n=== Forward diagnostics: model.{mode}() ===")
        if mode == "eval":
            model.eval()
        else:
            model.train()

        with torch.no_grad():
            out = model(batch["x_ts"], batch.get("x_scalar"), return_dict=True)

        for key in [
            "temporal_tokens",
            "reduced_tokens",
            "stock_embedding",
            "cross_embedding",
            "scores",
        ]:
            if key in out:
                print(f"\n-- {key} --")
                print_stats(cross_stock_stats(key, out[key]))

        scores = out["scores"].detach().float()
        print("\nScore first row first 10:")
        print(scores[0, :10].cpu().numpy())

        metrics = tr.compute_batch_metrics(
            scores,
            batch["y"].detach().float(),
            batch["y_raw"].detach().float(),
            topk=int(pick(cfg, "topk_metric_k", 20)),
        )
        print("\nMetrics:")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
