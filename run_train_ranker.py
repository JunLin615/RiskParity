"""
run_train_ranker.py

Command-line entrypoint for training the Dual-Transformer ranker.

This is the "run this file to start training" layer.
The underlying reusable training functions remain in train_ranker.py.

Example
-------
python run_train_ranker.py \
  --bundle-path data/cache/stage1_factor_label_bundle.pkl \
  --checkpoint-dir checkpoints/dual_transformer_ranker \
  --train-end 20231231 \
  --valid-end 20241231 \
  --model-preset small \
  --max-epochs 10

You can also use a JSON config:

python run_train_ranker.py --config train_run_config_example.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Optional

import factor_pipeline as fp
import rank_dataset as rd
import train_ranker as tr


# ============================================================
# Config helpers
# ============================================================

def load_json_config(path: Optional[str]) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return p


def pick(args: argparse.Namespace, cfg: dict[str, Any], name: str, default: Any) -> Any:
    """CLI value has priority over JSON config; otherwise default."""
    v = getattr(args, name, None)
    if v is not None:
        return v
    if name in cfg:
        return cfg[name]
    return default


def model_preset_kwargs(preset: str) -> dict[str, Any]:
    """
    Model presets.

    small:
        For RTX 4060 8G first tests.

    medium:
        A reasonable local-GPU setting if small is stable.

    base:
        The originally proposed 512-dim model, better for larger GPUs.
    """
    preset = str(preset).lower()

    if preset == "small":
        return {
            "temporal_channels": 8,
            "temporal_compressed_len": 16,
            "model_dim": 128,
            "factor_num_layers": 1,
            "factor_num_heads": 4,
            "factor_ff_dim": 256,
            "cross_num_layers": 1,
            "cross_num_heads": 4,
            "cross_ff_dim": 256,
            "score_hidden_dim": 128,
            "dropout": 0.1,
        }

    if preset == "medium":
        return {
            "temporal_channels": 8,
            "temporal_compressed_len": 32,
            "model_dim": 256,
            "factor_num_layers": 1,
            "factor_num_heads": 8,
            "factor_ff_dim": 512,
            "cross_num_layers": 1,
            "cross_num_heads": 8,
            "cross_ff_dim": 512,
            "score_hidden_dim": 256,
            "dropout": 0.1,
        }

    if preset == "base":
        return {
            "temporal_channels": 16,
            "temporal_compressed_len": 32,
            "model_dim": 512,
            "factor_num_layers": 1,
            "factor_num_heads": 8,
            "factor_ff_dim": 1024,
            "cross_num_layers": 1,
            "cross_num_heads": 8,
            "cross_ff_dim": 1024,
            "score_hidden_dim": 512,
            "dropout": 0.1,
        }

    raise ValueError("model_preset must be one of: small, medium, base")


def override_model_kwargs(base: dict[str, Any], args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    """Override preset model kwargs with explicit CLI/JSON values."""
    model_keys = [
        "temporal_channels",
        "temporal_compressed_len",
        "model_dim",
        "factor_num_layers",
        "factor_num_heads",
        "factor_ff_dim",
        "cross_num_layers",
        "cross_num_heads",
        "cross_ff_dim",
        "score_hidden_dim",
        "dropout",
    ]
    out = dict(base)
    for key in model_keys:
        v = getattr(args, key, None)
        if v is not None:
            out[key] = v
        elif key in cfg:
            out[key] = cfg[key]
    return out


# ============================================================
# Argparse
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Dual-Transformer ranker.")

    parser.add_argument("--config", type=str, default=None, help="Optional JSON config path.")

    # Data
    parser.add_argument("--bundle-path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)

    # Split
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--valid-end", type=str, default=None)
    parser.add_argument("--valid-ratio", type=float, default=None)

    # Dataset
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--samples-per-date", type=int, default=None)
    parser.add_argument("--label-name", type=str, default=None)
    parser.add_argument("--label-valid-name", type=str, default=None)
    parser.add_argument("--target-mode", type=str, default=None)
    parser.add_argument("--candidate-pool-size", type=int, default=None)
    parser.add_argument("--allow-smaller-sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--require-full-history", action=argparse.BooleanOptionalAction, default=None)

    # Model
    parser.add_argument("--model-preset", type=str, default=None, choices=["small", "medium", "base"])
    parser.add_argument("--temporal-channels", type=int, default=None)
    parser.add_argument("--temporal-compressed-len", type=int, default=None)
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--factor-num-layers", type=int, default=None)
    parser.add_argument("--factor-num-heads", type=int, default=None)
    parser.add_argument("--factor-ff-dim", type=int, default=None)
    parser.add_argument("--cross-num-layers", type=int, default=None)
    parser.add_argument("--cross-num-heads", type=int, default=None)
    parser.add_argument("--cross-ff-dim", type=int, default=None)
    parser.add_argument("--score-hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    # Train
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--loss-type", type=str, default=None, choices=["spearman", "pearson", "pairwise", "ndcg", "ndcg_pairwise"])
    parser.add_argument("--tau-start", type=float, default=None)
    parser.add_argument("--tau-end", type=float, default=None)
    parser.add_argument("--tau-decay-epochs", type=int, default=None)
    parser.add_argument("--ndcg-temperature", type=float, default=None)
    parser.add_argument("--ndcg-gain-power", type=float, default=None)
    parser.add_argument("--ndcg-max-pairs", type=int, default=None)
    parser.add_argument("--l2-lambda", type=float, default=None)
    parser.add_argument("--l2-exclude-bias-norm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--topk-metric-k", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--log-every-steps", type=int, default=None)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=None)

    # TensorBoard
    parser.add_argument("--use-tensorboard", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tensorboard-dirname", type=str, default=None)
    parser.add_argument("--tb-log-every-steps", type=int, default=None)
    parser.add_argument("--tb-log-memory-every-steps", type=int, default=None)
    parser.add_argument("--tb-log-grad-norm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tb-log-epoch-metrics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tb-log-config-text", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tb-log-histograms", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tb-histogram-every-epochs", type=int, default=None)

    return parser


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_json_config(args.config)

    bundle_path = pick(args, cfg, "bundle_path", "data/cache/stage1_factor_label_bundle.pkl")
    checkpoint_dir = pick(args, cfg, "checkpoint_dir", "checkpoints/dual_transformer_ranker")
    run_name = pick(args, cfg, "run_name", None)

    train_end = pick(args, cfg, "train_end", None)
    valid_end = pick(args, cfg, "valid_end", None)
    valid_ratio = pick(args, cfg, "valid_ratio", 0.2)

    sample_size = pick(args, cfg, "sample_size", 512)
    seq_len = pick(args, cfg, "seq_len", 128)
    samples_per_date = pick(args, cfg, "samples_per_date", 1)
    label_name = pick(args, cfg, "label_name", "label_ret_t1_t6")
    label_valid_name = pick(args, cfg, "label_valid_name", "label_valid_t1_t6")
    target_mode = pick(args, cfg, "target_mode", "rank_pct")
    candidate_pool_size = pick(args, cfg, "candidate_pool_size", None)
    allow_smaller_sample = pick(args, cfg, "allow_smaller_sample", False)
    require_full_history = pick(args, cfg, "require_full_history", True)

    model_preset = pick(args, cfg, "model_preset", "small")
    model_kwargs = override_model_kwargs(model_preset_kwargs(model_preset), args, cfg)

    train_params = {
        "max_epochs": pick(args, cfg, "max_epochs", 10),
        "batch_size": pick(args, cfg, "batch_size", 1),
        "num_workers": pick(args, cfg, "num_workers", 0),
        "lr": pick(args, cfg, "lr", 1e-4),
        "weight_decay": pick(args, cfg, "weight_decay", 1e-4),
        "grad_clip_norm": pick(args, cfg, "grad_clip_norm", 1.0),
        "loss_type": pick(args, cfg, "loss_type", "spearman"),
        "tau_start": pick(args, cfg, "tau_start", 1.0),
        "tau_end": pick(args, cfg, "tau_end", 0.1),
        "tau_decay_epochs": pick(args, cfg, "tau_decay_epochs", 50),
        "ndcg_temperature": pick(args, cfg, "ndcg_temperature", 1.0),
        "ndcg_gain_power": pick(args, cfg, "ndcg_gain_power", 1.0),
        "ndcg_max_pairs": pick(args, cfg, "ndcg_max_pairs", 50000),
        "l2_lambda": pick(args, cfg, "l2_lambda", 0.0),
        "l2_exclude_bias_norm": pick(args, cfg, "l2_exclude_bias_norm", True),
        "topk_metric_k": pick(args, cfg, "topk_metric_k", 20),
        "device": pick(args, cfg, "device", "auto"),
        "seed": pick(args, cfg, "seed", 42),
        "early_stopping_patience": pick(args, cfg, "early_stopping_patience", 10),
        "log_every_steps": pick(args, cfg, "log_every_steps", 50),
        "use_amp": pick(args, cfg, "use_amp", False),
        "use_tensorboard": pick(args, cfg, "use_tensorboard", True),
        "tensorboard_dirname": pick(args, cfg, "tensorboard_dirname", "tensorboard"),
        "tb_log_every_steps": pick(args, cfg, "tb_log_every_steps", 50),
        "tb_log_memory_every_steps": pick(args, cfg, "tb_log_memory_every_steps", 10),
        "tb_log_grad_norm": pick(args, cfg, "tb_log_grad_norm", True),
        "tb_log_epoch_metrics": pick(args, cfg, "tb_log_epoch_metrics", True),
        "tb_log_config_text": pick(args, cfg, "tb_log_config_text", True),
        "tb_log_histograms": pick(args, cfg, "tb_log_histograms", False),
        "tb_histogram_every_epochs": pick(args, cfg, "tb_histogram_every_epochs", 5),
    }

    full_run_config = {
        "bundle_path": bundle_path,
        "checkpoint_dir": checkpoint_dir,
        "run_name": run_name,
        "train_end": train_end,
        "valid_end": valid_end,
        "valid_ratio": valid_ratio,
        "dataset": {
            "sample_size": sample_size,
            "seq_len": seq_len,
            "samples_per_date": samples_per_date,
            "label_name": label_name,
            "label_valid_name": label_valid_name,
            "target_mode": target_mode,
            "candidate_pool_size": candidate_pool_size,
            "allow_smaller_sample": allow_smaller_sample,
            "require_full_history": require_full_history,
        },
        "model_preset": model_preset,
        "model": model_kwargs,
        "train": train_params,
    }

    print("Loading bundle:", bundle_path)
    bundle = fp.load_bundle(bundle_path)

    ds_config = rd.CrossSectionDatasetConfig(
        sample_size=int(sample_size),
        seq_len=int(seq_len),
        samples_per_date=int(samples_per_date),
        label_name=str(label_name),
        label_valid_name=None if label_valid_name in (None, "None", "none", "") else str(label_valid_name),
        target_mode=str(target_mode),
        require_full_history=bool(require_full_history),
        allow_smaller_sample=bool(allow_smaller_sample),
        return_tensors="torch",
        random_seed=int(train_params["seed"]),
        candidate_pool_size=candidate_pool_size,
    )

    print("Building train/valid datasets...")
    train_ds, valid_ds = tr.build_datasets_from_bundle(
        bundle,
        ds_config=ds_config,
        train_end=train_end,
        valid_end=valid_end,
        valid_ratio=float(valid_ratio),
    )

    print("Train dataset summary:")
    print(train_ds.summary())
    print("Valid dataset summary:")
    print(valid_ds.summary())

    print("Building model...")
    model = tr.build_model_from_bundle(
        bundle,
        seq_len=int(seq_len),
        **model_kwargs,
    )

    train_config = tr.TrainConfig(
        checkpoint_dir=str(checkpoint_dir),
        run_name=run_name,
        max_epochs=int(train_params["max_epochs"]),
        batch_size=int(train_params["batch_size"]),
        num_workers=int(train_params["num_workers"]),
        lr=float(train_params["lr"]),
        weight_decay=float(train_params["weight_decay"]),
        grad_clip_norm=None if train_params["grad_clip_norm"] is None else float(train_params["grad_clip_norm"]),
        loss_type=str(train_params["loss_type"]),
        tau_start=float(train_params["tau_start"]),
        tau_end=float(train_params["tau_end"]),
        tau_decay_epochs=int(train_params["tau_decay_epochs"]),
        ndcg_temperature=float(train_params["ndcg_temperature"]),
        ndcg_gain_power=float(train_params["ndcg_gain_power"]),
        ndcg_max_pairs=train_params["ndcg_max_pairs"],
        l2_lambda=float(train_params["l2_lambda"]),
        l2_exclude_bias_norm=bool(train_params["l2_exclude_bias_norm"]),
        topk_metric_k=int(train_params["topk_metric_k"]),
        device=str(train_params["device"]),
        seed=int(train_params["seed"]),
        early_stopping_patience=train_params["early_stopping_patience"],
        log_every_steps=int(train_params["log_every_steps"]),
        use_amp=bool(train_params["use_amp"]),
        use_tensorboard=bool(train_params["use_tensorboard"]),
        tensorboard_dirname=str(train_params["tensorboard_dirname"]),
        tb_log_every_steps=int(train_params["tb_log_every_steps"]),
        tb_log_memory_every_steps=int(train_params["tb_log_memory_every_steps"]),
        tb_log_grad_norm=bool(train_params["tb_log_grad_norm"]),
        tb_log_epoch_metrics=bool(train_params["tb_log_epoch_metrics"]),
        tb_log_config_text=bool(train_params["tb_log_config_text"]),
        tb_log_histograms=bool(train_params["tb_log_histograms"]),
        tb_histogram_every_epochs=int(train_params["tb_histogram_every_epochs"]),
    )

    print("Starting training...")
    history = tr.fit_model(
        model=model,
        train_dataset=train_ds,
        valid_dataset=valid_ds,
        train_config=train_config,
    )

    run_dir = history.attrs.get("run_dir")
    if run_dir:
        run_dir_path = Path(run_dir)
        save_json(run_dir_path / "launcher_config.json", full_run_config)

        if args.config is not None:
            try:
                shutil.copy2(args.config, run_dir_path / "input_config.json")
            except Exception as exc:
                print(f"Warning: failed to copy input config: {exc}")

        print("Training finished.")
        print("Run directory:", run_dir_path)
    else:
        print("Training finished. No run_dir because checkpoint_dir=None.")


if __name__ == "__main__":
    main()
