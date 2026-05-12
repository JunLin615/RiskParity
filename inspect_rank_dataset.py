"""
inspect_rank_dataset.py

Print basic information about the currently used factor/label bundle and
CrossSectionRankDataset split.

Purpose
-------
After multiple code versions, this script helps confirm what dataset range,
date split, factor counts, label names, sample sizes, and batch shapes are
actually being used by the current training configuration.

It intentionally reuses the same project modules as training:
    factor_pipeline.py
    rank_dataset.py
    train_ranker.py or a specified train module

Usage
-----
python inspect_rank_dataset.py --config train_run_config_example.json

If your runner currently imports a custom training module, specify it:
python inspect_rank_dataset.py --config train_run_config_example.json --train-module train_ranker_earlystop

Optional batch inspection:
python inspect_rank_dataset.py --config train_run_config_example.json --inspect-batches 2

Optional date detail:
python inspect_rank_dataset.py --config train_run_config_example.json --show-dates 10
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch

import factor_pipeline as fp
import rank_dataset as rd


# ============================================================
# Generic helpers
# ============================================================

def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def pick(cfg: dict[str, Any], name: str, default: Any) -> Any:
    return cfg[name] if name in cfg else default


def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def safe_len(x: Any) -> Optional[int]:
    try:
        return len(x)
    except Exception:
        return None


def as_list(x: Any, max_items: Optional[int] = None) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        out = list(x)
    elif isinstance(x, np.ndarray):
        out = x.tolist()
    elif isinstance(x, pd.Index):
        out = x.tolist()
    else:
        try:
            out = list(x)
        except Exception:
            return []
    if max_items is not None:
        return out[:max_items]
    return out


def try_get(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                pass
        if isinstance(obj, dict) and name in obj:
            return obj[name]
    return default


def to_date_str(x: Any) -> str:
    if x is None:
        return "None"
    s = str(x)
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def describe_array(values: Any, name: str = "values") -> dict[str, Any]:
    arr = np.asarray(as_list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{name}_count": 0,
            f"{name}_min": None,
            f"{name}_p25": None,
            f"{name}_median": None,
            f"{name}_mean": None,
            f"{name}_p75": None,
            f"{name}_max": None,
        }
    return {
        f"{name}_count": int(arr.size),
        f"{name}_min": float(np.min(arr)),
        f"{name}_p25": float(np.percentile(arr, 25)),
        f"{name}_median": float(np.median(arr)),
        f"{name}_mean": float(np.mean(arr)),
        f"{name}_p75": float(np.percentile(arr, 75)),
        f"{name}_max": float(np.max(arr)),
    }


def print_dict(d: dict[str, Any], indent: int = 2) -> None:
    for k, v in d.items():
        print(" " * indent + f"{k}: {v}")


# ============================================================
# Bundle inspection
# ============================================================

def get_panel_shape(panel: Any) -> Any:
    if panel is None:
        return None
    if hasattr(panel, "shape"):
        return tuple(panel.shape)
    return None


def get_panel_dates(panel: Any) -> list[Any]:
    """
    Try to infer dates from common panel layouts.

    Possible layouts:
        MultiIndex index with level trade_date / date
        columns with trade_date
        index directly date-like
    """
    if panel is None:
        return []

    if isinstance(panel, pd.DataFrame):
        if isinstance(panel.index, pd.MultiIndex):
            names = list(panel.index.names)
            for candidate in ["trade_date", "date", "signal_date"]:
                if candidate in names:
                    return sorted(panel.index.get_level_values(candidate).unique().tolist())
            # fallback first level
            return sorted(panel.index.get_level_values(0).unique().tolist())

        for col in ["trade_date", "date", "signal_date"]:
            if col in panel.columns:
                return sorted(panel[col].dropna().unique().tolist())

        return sorted(panel.index.unique().tolist())

    return []


def print_bundle_info(bundle: Any) -> None:
    print_section("Bundle basic info")

    metadata = getattr(bundle, "metadata", None)
    if metadata is None and isinstance(bundle, dict):
        metadata = bundle.get("metadata")
    metadata = metadata or {}

    print("Bundle class:", type(bundle).__name__)

    for key in [
        "factor_count",
        "label_count",
        "feature_panel_shape",
        "label_panel_shape",
        "factor_names",
        "label_names",
        "start_date",
        "end_date",
        "created_at",
    ]:
        if key in metadata:
            val = metadata[key]
            if isinstance(val, list):
                print(f"{key}: count={len(val)} first10={val[:10]}")
            else:
                print(f"{key}: {val}")

    feature_panel = getattr(bundle, "feature_panel", None)
    label_panel = getattr(bundle, "label_panel", None)
    scalar_panel = getattr(bundle, "scalar_panel", None)

    if feature_panel is not None:
        dates = get_panel_dates(feature_panel)
        print("feature_panel shape:", get_panel_shape(feature_panel))
        if dates:
            print("feature_panel date range:", to_date_str(min(dates)), "→", to_date_str(max(dates)), f"(n_dates={len(dates)})")

    if label_panel is not None:
        dates = get_panel_dates(label_panel)
        print("label_panel shape:", get_panel_shape(label_panel))
        if dates:
            print("label_panel date range:", to_date_str(min(dates)), "→", to_date_str(max(dates)), f"(n_dates={len(dates)})")

    if scalar_panel is not None:
        print("scalar_panel shape:", get_panel_shape(scalar_panel))


# ============================================================
# Dataset construction and inspection
# ============================================================

def make_dataset_config(cfg: dict[str, Any]) -> rd.CrossSectionDatasetConfig:
    return rd.CrossSectionDatasetConfig(
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


def load_train_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise RuntimeError(f"Failed to import train module {name!r}: {exc}") from exc


def build_datasets(bundle: Any, cfg: dict[str, Any], train_module_name: str):
    train_mod = load_train_module(train_module_name)
    ds_config = make_dataset_config(cfg)

    if not hasattr(train_mod, "build_datasets_from_bundle"):
        raise RuntimeError(
            f"{train_module_name} does not have build_datasets_from_bundle(). "
            "Please pass the same train module used by your runner."
        )

    train_ds, valid_ds = train_mod.build_datasets_from_bundle(
        bundle,
        ds_config=ds_config,
        train_end=pick(cfg, "train_end", None),
        valid_end=pick(cfg, "valid_end", None),
        valid_ratio=float(pick(cfg, "valid_ratio", 0.2)),
    )
    return train_mod, ds_config, train_ds, valid_ds


def infer_dataset_dates(ds: Any) -> list[Any]:
    candidates = [
        "dates",
        "sample_dates",
        "signal_dates",
        "trade_dates",
        "available_dates",
        "date_list",
    ]
    for name in candidates:
        val = try_get(ds, [name])
        out = as_list(val)
        if out:
            return out

    # Some datasets keep samples list of tuples/dicts.
    samples = try_get(ds, ["samples", "index", "items"])
    if samples is not None:
        out = []
        for s in as_list(samples):
            if isinstance(s, dict):
                for k in ["date", "trade_date", "signal_date"]:
                    if k in s:
                        out.append(s[k])
                        break
            elif isinstance(s, (tuple, list)) and len(s) > 0:
                out.append(s[0])
        if out:
            return out

    return []


def infer_pool_sizes(ds: Any) -> list[int]:
    candidates = [
        "pool_sizes",
        "candidate_counts",
        "date_pool_sizes",
        "valid_counts",
        "num_candidates_by_date",
    ]
    for name in candidates:
        val = try_get(ds, [name])
        if isinstance(val, dict):
            return [int(v) for v in val.values()]
        out = as_list(val)
        if out:
            try:
                return [int(x) for x in out]
            except Exception:
                pass

    # Try common date_to_codes/candidates maps.
    for name in ["date_to_codes", "date_to_candidates", "candidates_by_date", "valid_codes_by_date"]:
        val = try_get(ds, [name])
        if isinstance(val, dict):
            return [len(v) for v in val.values()]

    return []


def inspect_one_dataset(ds: Any, name: str, show_dates: int = 0) -> None:
    print_section(f"{name} dataset")

    print("class:", type(ds).__name__)
    print("len(dataset):", safe_len(ds))

    config = try_get(ds, ["config", "ds_config"])
    if config is not None:
        print("dataset config:", config)

    dates = infer_dataset_dates(ds)
    if dates:
        unique_dates = sorted(set(dates))
        print("date range:", to_date_str(min(unique_dates)), "→", to_date_str(max(unique_dates)))
        print("n_sample_dates:", len(dates))
        print("n_unique_dates:", len(unique_dates))
        if show_dates:
            print(f"first {show_dates} dates:", [to_date_str(x) for x in unique_dates[:show_dates]])
            print(f"last  {show_dates} dates:", [to_date_str(x) for x in unique_dates[-show_dates:]])
    else:
        print("date range: unavailable from dataset attributes")

    pool_sizes = infer_pool_sizes(ds)
    if pool_sizes:
        print("candidate/effective pool size summary:")
        print_dict(describe_array(pool_sizes, "pool_size"), indent=2)
    else:
        print("candidate/effective pool sizes: unavailable from dataset attributes")

    # Try one sample.
    try:
        sample = ds[0]
        print("\nFirst sample keys:", list(sample.keys()) if isinstance(sample, dict) else type(sample))
        if isinstance(sample, dict):
            for key, value in sample.items():
                if torch.is_tensor(value):
                    print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype} finite={torch.isfinite(value).float().mean().item():.4f}")
                elif isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape} dtype={value.dtype}")
                else:
                    print(f"  {key}: {value}")
    except Exception as exc:
        print("Failed to inspect ds[0]:", repr(exc))


def make_loader(train_mod: Any, ds: Any, cfg: dict[str, Any], shuffle: bool):
    if hasattr(train_mod, "TrainConfig") and hasattr(train_mod, "make_dataloader"):
        tc = train_mod.TrainConfig(
            batch_size=int(pick(cfg, "batch_size", 1)),
            num_workers=int(pick(cfg, "num_workers", 0)),
            device=str(pick(cfg, "device", "auto")),
            topk_metric_k=int(pick(cfg, "topk_metric_k", 20)),
        )
        return train_mod.make_dataloader(ds, tc, shuffle=shuffle)

    from torch.utils.data import DataLoader
    return DataLoader(
        ds,
        batch_size=int(pick(cfg, "batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(pick(cfg, "num_workers", 0)),
    )


def inspect_batches(train_mod: Any, train_ds: Any, valid_ds: Any, cfg: dict[str, Any], n_batches: int) -> None:
    if n_batches <= 0:
        return

    print_section("Batch inspection")

    for split_name, ds, shuffle in [
        ("train", train_ds, False),
        ("valid", valid_ds, False),
    ]:
        print(f"\n[{split_name}]")
        loader = make_loader(train_mod, ds, cfg, shuffle=shuffle)
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            print(f"batch {i}:")
            if isinstance(batch, dict):
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        finite = torch.isfinite(value).float().mean().item() if value.is_floating_point() else 1.0
                        msg = f"  {key}: shape={tuple(value.shape)} dtype={value.dtype} finite={finite:.4f}"
                        if key in {"y", "y_raw"} and value.is_floating_point():
                            msg += f" mean={value.float().mean().item():.6g} std={value.float().std(unbiased=False).item():.6g}"
                        print(msg)
                    else:
                        print(f"  {key}: {type(value).__name__}")
            else:
                print(type(batch))


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training JSON config path.")
    parser.add_argument(
        "--train-module",
        default="train_ranker",
        help="Training module to reuse, e.g. train_ranker, train_ranker_earlystop, train_ranker_scoredebug.",
    )
    parser.add_argument("--inspect-batches", type=int, default=1, help="Number of train/valid batches to inspect.")
    parser.add_argument("--show-dates", type=int, default=5, help="Show first/last N dates for each split.")
    args = parser.parse_args()

    cfg = load_json(args.config)
    bundle_path = pick(cfg, "bundle_path", "data/cache/stage1_factor_label_bundle.pkl")

    print_section("Config summary")
    print("config:", args.config)
    print("train_module:", args.train_module)
    for key in [
        "bundle_path",
        "data_start_date",
        "data_end_date",
        "train_end",
        "valid_end",
        "valid_ratio",
        "sample_size",
        "seq_len",
        "samples_per_date",
        "batch_size",
        "label_name",
        "label_valid_name",
        "target_mode",
        "candidate_pool_size",
        "allow_smaller_sample",
        "require_full_history",
        "seed",
    ]:
        if key in cfg:
            print(f"{key}: {cfg[key]}")

    bundle = fp.load_bundle(bundle_path)
    print_bundle_info(bundle)

    train_mod, ds_config, train_ds, valid_ds = build_datasets(bundle, cfg, args.train_module)

    print_section("Effective CrossSectionDatasetConfig")
    print(ds_config)

    inspect_one_dataset(train_ds, "Train", show_dates=int(args.show_dates))
    inspect_one_dataset(valid_ds, "Valid", show_dates=int(args.show_dates))

    inspect_batches(train_mod, train_ds, valid_ds, cfg, int(args.inspect_batches))

    print_section("Interpretation hints")
    print("1. If train_end is set, split should be time-based:")
    print("   train dates <= train_end; valid dates > train_end and <= valid_end.")
    print("2. valid_ratio is usually only used when train_end is null.")
    print("3. Training dataset may resample stocks each epoch; validation dataset is usually deterministic.")
    print("4. Confirm actual date range above, not only JSON date fields.")


if __name__ == "__main__":
    main()
