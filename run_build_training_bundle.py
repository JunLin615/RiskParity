"""
run_build_training_bundle.py

Command-line entrypoint for building the training bundle used by run_train_ranker.py.

This is the "run this file before training" layer.

It reads a JSON config, loads the local stock database through stock_data.py,
runs factor_pipeline.py, and saves a FeatureLabelBundle to bundle_path.

Recommended workflow
--------------------
1. Build bundle:
    python run_build_training_bundle.py --config train_run_config_example.json

2. Train model:
    python run_train_ranker.py --config train_run_config_example.json

The same config file can be used by both scripts. This builder uses only the
data-related keys and ignores training/model-only keys.

Important
---------
If you want rank_dataset.py to have full seq_len history for early training
dates, set data_start_date earlier than the first intended training signal date.
For example, if training starts in 2020 and seq_len=128, set data_start_date
around 2019-06-01 or earlier.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import factor_pipeline as fp
from stock_data import create_stock_manager, load_tushare_token


# ============================================================
# Helpers
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


def pick_any(args: argparse.Namespace, cfg: dict[str, Any], names: list[str], default: Any) -> Any:
    """Pick first available value among multiple possible key names."""
    for name in names:
        v = getattr(args, name, None)
        if v is not None:
            return v
    for name in names:
        if name in cfg:
            return cfg[name]
    return default


def normalize_yyyymmdd(date: Optional[str]) -> Optional[str]:
    if date is None:
        return None
    x = str(date).replace("-", "").strip()
    if len(x) != 8 or not x.isdigit():
        raise ValueError(f"date must be YYYYMMDD or YYYY-MM-DD, got {date!r}")
    return x


def shift_yyyymmdd(date: Optional[str], days: int) -> Optional[str]:
    """Shift YYYYMMDD date by calendar days."""
    if date is None:
        return None
    x = normalize_yyyymmdd(date)
    dt = datetime.strptime(x, "%Y%m%d")
    return (dt + timedelta(days=int(days))).strftime("%Y%m%d")


def bool_from_value(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in {"1", "true", "yes", "y", "on"}
    return bool(x)


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build factor/label training bundle.")

    parser.add_argument("--config", type=str, default=None, help="Optional JSON config path.")

    # stock_data.py / database
    parser.add_argument("--token-path", type=str, default=None)
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--default-start-date", type=str, default=None)

    # output
    parser.add_argument("--bundle-path", type=str, default=None)
    parser.add_argument("--save-panels-dir", type=str, default=None)

    # final signal-date range kept in feature_panel / label_panel
    parser.add_argument("--data-start-date", type=str, default=None)
    parser.add_argument("--data-end-date", type=str, default=None)

    # raw data load range from DB
    parser.add_argument("--load-start-date", type=str, default=None)
    parser.add_argument("--load-end-date", type=str, default=None)
    parser.add_argument("--warmup-days", type=int, default=None)
    parser.add_argument("--forward-days-buffer", type=int, default=None)

    # factor/label options
    parser.add_argument("--feature-adjust-type", type=str, default=None, choices=["raw", "qfq", "hfq"])
    parser.add_argument("--label-adjust-type", type=str, default=None, choices=["raw", "qfq", "hfq"])
    parser.add_argument("--execution-adjust-type", type=str, default=None, choices=["raw", "qfq", "hfq"])

    parser.add_argument("--standardize-features", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--winsorize-features", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--winsor-lower-q", type=float, default=None)
    parser.add_argument("--winsor-upper-q", type=float, default=None)

    parser.add_argument("--include-close-to-close-label", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-execution-label", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fee-rate-buy", type=float, default=None)
    parser.add_argument("--fee-rate-sell", type=float, default=None)

    parser.add_argument("--mask-labels-by-universe", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mask-labels-by-trade-status", action=argparse.BooleanOptionalAction, default=None)

    # offsets / windows
    parser.add_argument("--buy-offset", type=int, default=None)
    parser.add_argument("--sell-offset", type=int, default=None)
    parser.add_argument("--yz-window", type=int, default=None)
    parser.add_argument("--amount-unit-scale", type=float, default=None)

    return parser


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_json_config(args.config)

    # Shared config keys with run_train_ranker.py.
    bundle_path = pick(args, cfg, "bundle_path", "data/cache/stage1_factor_label_bundle.pkl")

    # DB config.
    token_path = pick(args, cfg, "token_path", "tushare_token.txt")
    db_path = pick(args, cfg, "db_path", "data/db/stock_data.db")
    default_start_date = pick(args, cfg, "default_start_date", "20180101")

    # Date config.
    # Prefer data_start_date/data_end_date for bundle generation.
    # Fall back to start_date/end_date if present.
    # Fall back to train_end/valid_end only for end date, because train_end is not a start date.
    data_start_date = pick_any(
        args,
        cfg,
        ["data_start_date", "start_date"],
        default_start_date,
    )
    data_end_date = pick_any(
        args,
        cfg,
        ["data_end_date", "end_date", "valid_end"],
        None,
    )

    warmup_days = int(pick(args, cfg, "warmup_days", 450))
    forward_days_buffer = int(pick(args, cfg, "forward_days_buffer", 30))

    load_start_date = pick_any(
        args,
        cfg,
        ["load_start_date", "data_load_start_date"],
        shift_yyyymmdd(data_start_date, -warmup_days),
    )
    load_end_date = pick_any(
        args,
        cfg,
        ["load_end_date", "data_load_end_date"],
        shift_yyyymmdd(data_end_date, forward_days_buffer) if data_end_date is not None else None,
    )

    # Pipeline options.
    feature_adjust_type = pick(args, cfg, "feature_adjust_type", "qfq")
    label_adjust_type = pick(args, cfg, "label_adjust_type", "qfq")
    execution_adjust_type = pick(args, cfg, "execution_adjust_type", "raw")

    standardize_features = bool_from_value(pick(args, cfg, "standardize_features", True))
    winsorize_features = bool_from_value(pick(args, cfg, "winsorize_features", True))
    winsor_lower_q = float(pick(args, cfg, "winsor_lower_q", 0.01))
    winsor_upper_q = float(pick(args, cfg, "winsor_upper_q", 0.99))

    include_close_to_close_label = bool_from_value(pick(args, cfg, "include_close_to_close_label", True))
    include_execution_label = bool_from_value(pick(args, cfg, "include_execution_label", True))

    fee_rate_buy = float(pick(args, cfg, "fee_rate_buy", 0.0015))
    fee_rate_sell = float(pick(args, cfg, "fee_rate_sell", 0.0015))

    mask_labels_by_universe = bool_from_value(pick(args, cfg, "mask_labels_by_universe", True))
    mask_labels_by_trade_status = bool_from_value(pick(args, cfg, "mask_labels_by_trade_status", True))

    buy_offset = int(pick(args, cfg, "buy_offset", 1))
    sell_offset = int(pick(args, cfg, "sell_offset", 6))
    yz_window = int(pick(args, cfg, "yz_window", 30))
    amount_unit_scale = float(pick(args, cfg, "amount_unit_scale", 1000.0))

    save_panels_dir = pick(args, cfg, "save_panels_dir", None)

    build_config_used = {
        "config_file": args.config,
        "bundle_path": bundle_path,
        "token_path": token_path,
        "db_path": db_path,
        "default_start_date": default_start_date,
        "data_start_date": normalize_yyyymmdd(data_start_date),
        "data_end_date": normalize_yyyymmdd(data_end_date),
        "load_start_date": normalize_yyyymmdd(load_start_date),
        "load_end_date": normalize_yyyymmdd(load_end_date),
        "warmup_days": warmup_days,
        "forward_days_buffer": forward_days_buffer,
        "feature_adjust_type": feature_adjust_type,
        "label_adjust_type": label_adjust_type,
        "execution_adjust_type": execution_adjust_type,
        "standardize_features": standardize_features,
        "winsorize_features": winsorize_features,
        "winsor_lower_q": winsor_lower_q,
        "winsor_upper_q": winsor_upper_q,
        "include_close_to_close_label": include_close_to_close_label,
        "include_execution_label": include_execution_label,
        "fee_rate_buy": fee_rate_buy,
        "fee_rate_sell": fee_rate_sell,
        "mask_labels_by_universe": mask_labels_by_universe,
        "mask_labels_by_trade_status": mask_labels_by_trade_status,
        "buy_offset": buy_offset,
        "sell_offset": sell_offset,
        "yz_window": yz_window,
        "amount_unit_scale": amount_unit_scale,
        "save_panels_dir": save_panels_dir,
    }

    print("Building training bundle with config:")
    print(json.dumps(build_config_used, ensure_ascii=False, indent=2))

    token = load_tushare_token(token_path)
    manager = create_stock_manager(
        tushare_token=token,
        db_path=db_path,
        default_start_date=default_start_date,
    )

    pipe_config = fp.FactorPipelineConfig(
        start_date=normalize_yyyymmdd(data_start_date),
        end_date=normalize_yyyymmdd(data_end_date),
        load_start_date=normalize_yyyymmdd(load_start_date),
        load_end_date=normalize_yyyymmdd(load_end_date),
        feature_adjust_type=str(feature_adjust_type),
        label_adjust_type=str(label_adjust_type),
        execution_adjust_type=str(execution_adjust_type),
        yz_window=int(yz_window),
        amount_unit_scale=float(amount_unit_scale),
        buy_offset=int(buy_offset),
        sell_offset=int(sell_offset),
        include_close_to_close_label=include_close_to_close_label,
        include_execution_label=include_execution_label,
        fee_rate_buy=float(fee_rate_buy),
        fee_rate_sell=float(fee_rate_sell),
        mask_labels_by_universe=mask_labels_by_universe,
        mask_labels_by_trade_status=mask_labels_by_trade_status,
        standardize_features=standardize_features,
        winsorize_features=winsorize_features,
        winsor_lower_q=winsor_lower_q,
        winsor_upper_q=winsor_upper_q,
    )

    bundle = fp.build_bundle_from_manager(
        manager,
        config=pipe_config,
        keep_raw_matrices=False,
    )

    bundle_path_obj = fp.save_bundle(bundle, bundle_path)
    print("Saved bundle:", bundle_path_obj)

    # Save sidecar metadata next to the bundle.
    sidecar_dir = bundle_path_obj.parent
    stem = bundle_path_obj.stem

    save_json(sidecar_dir / f"{stem}.build_config.json", build_config_used)
    save_json(sidecar_dir / f"{stem}.metadata.json", bundle.metadata)

    if args.config is not None:
        try:
            shutil.copy2(args.config, sidecar_dir / f"{stem}.input_config.json")
        except Exception as exc:
            print(f"Warning: failed to copy input config: {exc}")

    if save_panels_dir:
        paths = fp.save_panels(bundle, save_panels_dir)
        print("Saved separate panels:")
        for k, v in paths.items():
            print(f"  {k}: {v}")

    print("Bundle summary:")
    print(json.dumps({
        "factor_count": bundle.metadata.get("factor_count"),
        "label_count": bundle.metadata.get("label_count"),
        "feature_panel_shape": bundle.metadata.get("feature_panel_shape"),
        "label_panel_shape": bundle.metadata.get("label_panel_shape"),
        "factor_names_first10": bundle.metadata.get("factor_names", [])[:10],
        "label_names": bundle.metadata.get("label_names", []),
    }, ensure_ascii=False, indent=2))

    print("Done.")


if __name__ == "__main__":
    main()
