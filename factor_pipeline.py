"""
factor_pipeline.py

Feature/label pipeline for the cross-sectional ranking model.

Role
----
This module is the preparation layer before rank_dataset.py / DataLoader.

It does:
1. Load wide matrices from StockDataManager.
2. Compute stage-1 factors using factor_library.py.
3. Build t+1 to t+6 labels using label_builder.py.
4. Apply universe/trade-status masks.
5. Optionally cross-sectionally winsorize and z-score features.
6. Convert factor/label dictionaries into MultiIndex panels.
7. Save/load pipeline bundles as cache files.

It does NOT:
1. Randomly sample 512 stocks.
2. Implement PyTorch Dataset or DataLoader.
3. Train models.
4. Run backtests.

Matrix convention
-----------------
All raw matrices:
    index   = trade_date
    columns = ts_code

Stacked panels:
    index   = MultiIndex [trade_date, ts_code]
    columns = factor names or label names
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import factor_library as fl
import label_builder as lb


# ============================================================
# Configuration and bundle
# ============================================================

SCALAR_FACTOR_NAMES = (
    "free_float_ratio",
    "log_total_mv",
    "log_circ_mv",
    "log_pb",
    "log_ps_ttm",
    "ep_positive",
    "ep_negative",
    "ep_is_loss",
)

BINARY_FACTOR_NAMES = (
    "ep_is_loss",
)

DEFAULT_REQUIRED_MARKET_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "vol",
    "amount",
)

DEFAULT_BASIC_FIELDS = (
    "turnover_rate_f",
    "free_share",
    "total_share",
    "total_mv",
    "circ_mv",
    "pb",
    "ps_ttm",
    "pe_ttm",
)

DEFAULT_MONEYFLOW_FIELDS = (
    "net_mf_amount",
    "buy_lg_amount",
    "sell_lg_amount",
    "buy_elg_amount",
    "sell_elg_amount",
)

DEFAULT_ELIGIBILITY_FIELDS = (
    "is_eligible",
    "can_buy",
    "can_sell",
    "is_suspended",
    "is_limit_up",
    "is_limit_down",
)


@dataclass(frozen=True)
class FactorPipelineConfig:
    """
    Pipeline configuration.

    Date fields
    -----------
    start_date / end_date:
        Final signal-date range to keep in output panels.

    load_start_date / load_end_date:
        Data range to load from DB. Use load_start_date earlier than start_date
        if you need rolling lookback warm-up. Use load_end_date later than
        end_date if you need future labels near end_date.
    """

    start_date: Optional[str] = None
    end_date: Optional[str] = None
    load_start_date: Optional[str] = None
    load_end_date: Optional[str] = None

    feature_adjust_type: str = "qfq"
    label_adjust_type: str = "qfq"
    execution_adjust_type: str = "raw"

    windows: tuple[int, ...] = (5, 10, 20, 60)
    yz_window: int = 30
    annualization: int = 252
    min_periods_ratio: float = 0.7
    amount_unit_scale: float = 1000.0
    eps: float = 1e-12

    buy_offset: int = 1
    sell_offset: int = 6

    include_close_to_close_label: bool = True
    include_execution_label: bool = True

    fee_rate_buy: float = 0.0015
    fee_rate_sell: float = 0.0015

    mask_features_by_universe: bool = False
    mask_labels_by_universe: bool = True
    mask_labels_by_trade_status: bool = True

    standardize_features: bool = True
    winsorize_features: bool = True
    winsor_lower_q: float = 0.01
    winsor_upper_q: float = 0.99
    preserve_binary_factors: bool = True

    drop_all_nan_feature_rows: bool = False
    drop_all_nan_label_rows: bool = False


@dataclass
class FeatureLabelBundle:
    """
    Container returned by the pipeline.

    factors / labels:
        Dict of wide matrices.

    feature_panel / label_panel:
        MultiIndex panels with [trade_date, ts_code].

    metadata:
        Dict with config, factor names, label names, and shape summaries.
    """

    factors: dict[str, pd.DataFrame]
    labels: dict[str, pd.DataFrame]
    feature_panel: pd.DataFrame
    label_panel: pd.DataFrame
    market: dict[str, pd.DataFrame] = field(default_factory=dict)
    basic: dict[str, pd.DataFrame] = field(default_factory=dict)
    moneyflow: dict[str, pd.DataFrame] = field(default_factory=dict)
    eligibility: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Date helpers
# ============================================================

def normalize_yyyymmdd(date: Optional[str]) -> Optional[str]:
    """Normalize date string to YYYYMMDD."""
    if date is None:
        return None
    x = str(date).replace("-", "").strip()
    if len(x) != 8 or not x.isdigit():
        raise ValueError(f"date must be YYYYMMDD or YYYY-MM-DD, got {date!r}")
    return x


def to_datetime_index(values: pd.Index | Sequence[Any]) -> pd.DatetimeIndex:
    """Convert date-like values to sorted DatetimeIndex."""
    return pd.DatetimeIndex(pd.to_datetime(values)).sort_values()


def filter_matrix_by_signal_dates(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Filter a wide matrix by final signal-date range."""
    out = df.copy()
    if start_date is not None:
        out = out.loc[out.index >= pd.to_datetime(normalize_yyyymmdd(start_date), format="%Y%m%d")]
    if end_date is not None:
        out = out.loc[out.index <= pd.to_datetime(normalize_yyyymmdd(end_date), format="%Y%m%d")]
    return out


def filter_dict_by_signal_dates(
    data: Mapping[str, pd.DataFrame],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Filter all matrices in a dictionary by signal-date range."""
    return {
        k: filter_matrix_by_signal_dates(v, start_date=start_date, end_date=end_date)
        for k, v in data.items()
    }


# ============================================================
# Long table to wide matrix utilities
# ============================================================

def build_field_matrix(
    data: pd.DataFrame,
    field: str,
    date_col: str = "trade_date",
    code_col: str = "ts_code",
    date_format: Optional[str] = "%Y%m%d",
) -> pd.DataFrame:
    """Build one wide matrix from a long table."""
    if data.empty:
        return pd.DataFrame()

    if field not in data.columns:
        raise KeyError(f"field {field!r} not found in data columns")

    df = data[[date_col, code_col, field]].copy()

    if date_format is None:
        df[date_col] = pd.to_datetime(df[date_col])
    else:
        df[date_col] = pd.to_datetime(df[date_col].astype(str), format=date_format, errors="coerce")

    df = df.dropna(subset=[date_col, code_col])
    mat = (
        df.pivot(index=date_col, columns=code_col, values=field)
          .sort_index()
          .sort_index(axis=1)
    )
    return mat


def build_matrices_from_long_table(
    data: pd.DataFrame,
    fields: Iterable[str],
    date_col: str = "trade_date",
    code_col: str = "ts_code",
    date_format: Optional[str] = "%Y%m%d",
    missing_ok: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build multiple wide matrices from a long table."""
    out: dict[str, pd.DataFrame] = {}
    for field in fields:
        if field not in data.columns:
            if missing_ok:
                continue
            raise KeyError(f"field {field!r} not found in data columns")
        out[field] = build_field_matrix(
            data=data,
            field=field,
            date_col=date_col,
            code_col=code_col,
            date_format=date_format,
        )
    return out


def bool_matrix_from_numeric(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Convert numeric matrix to boolean matrix by > threshold."""
    x = fl.as_float_frame(df)
    return (x > threshold).fillna(False)


# ============================================================
# Loading from StockDataManager
# ============================================================

def _resolve_load_dates(config: FactorPipelineConfig) -> tuple[Optional[str], Optional[str]]:
    """Resolve database loading date range."""
    load_start = normalize_yyyymmdd(config.load_start_date or config.start_date)
    load_end = normalize_yyyymmdd(config.load_end_date or config.end_date)
    return load_start, load_end


def load_market_matrices_from_manager(
    manager: Any,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adjust_type: str = "qfq",
    fields: Iterable[str] = DEFAULT_REQUIRED_MARKET_FIELDS,
) -> dict[str, pd.DataFrame]:
    """
    Load OHLCV matrices from StockDataManager.

    Expected manager API:
        manager.get_prices(start_date=..., end_date=..., adjust_type=...)
    """
    df = manager.get_prices(
        start_date=normalize_yyyymmdd(start_date),
        end_date=normalize_yyyymmdd(end_date),
        adjust_type=adjust_type,
    )
    return build_matrices_from_long_table(df, fields=fields, missing_ok=True)


def load_basic_matrices_from_manager(
    manager: Any,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fields: Iterable[str] = DEFAULT_BASIC_FIELDS,
) -> dict[str, pd.DataFrame]:
    """
    Load daily_basic matrices from StockDataManager.

    Expected manager API:
        manager.store.get_stock_daily_basic(start_date=..., end_date=...)
    """
    df = manager.store.get_stock_daily_basic(
        start_date=normalize_yyyymmdd(start_date),
        end_date=normalize_yyyymmdd(end_date),
    )
    return build_matrices_from_long_table(df, fields=fields, missing_ok=True)


def load_moneyflow_matrices_from_manager(
    manager: Any,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fields: Iterable[str] = DEFAULT_MONEYFLOW_FIELDS,
) -> dict[str, pd.DataFrame]:
    """
    Load moneyflow matrices from StockDataManager.

    Expected manager API:
        manager.store.get_stock_moneyflow(start_date=..., end_date=...)
    """
    df = manager.store.get_stock_moneyflow(
        start_date=normalize_yyyymmdd(start_date),
        end_date=normalize_yyyymmdd(end_date),
    )
    return build_matrices_from_long_table(df, fields=fields, missing_ok=True)


def load_eligibility_matrices_from_manager(
    manager: Any,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fields: Iterable[str] = DEFAULT_ELIGIBILITY_FIELDS,
) -> dict[str, pd.DataFrame]:
    """
    Load eligibility/trade-status matrices from StockDataManager.

    Expected manager API:
        manager.store.get_stock_eligibility_daily(start_date=..., end_date=...)
    """
    df = manager.store.get_stock_eligibility_daily(
        start_date=normalize_yyyymmdd(start_date),
        end_date=normalize_yyyymmdd(end_date),
    )
    mats = build_matrices_from_long_table(df, fields=fields, missing_ok=True)

    # These fields are logically boolean, but keep them as bool matrices for masks.
    for key in ("is_eligible", "can_buy", "can_sell", "is_suspended", "is_limit_up", "is_limit_down"):
        if key in mats:
            mats[key] = bool_matrix_from_numeric(mats[key])

    return mats


def load_all_matrices_from_manager(
    manager: Any,
    config: FactorPipelineConfig,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    """
    Load all raw matrices needed by the pipeline.

    Returns
    -------
    feature_market:
        OHLCV matrices using config.feature_adjust_type.

    label_market:
        OHLCV matrices using config.label_adjust_type.

    execution_market:
        OHLCV matrices using config.execution_adjust_type.

    basic:
        daily_basic matrices.

    moneyflow:
        moneyflow matrices.

    eligibility:
        eligibility/trade-status matrices.
    """
    load_start, load_end = _resolve_load_dates(config)

    feature_market = load_market_matrices_from_manager(
        manager,
        start_date=load_start,
        end_date=load_end,
        adjust_type=config.feature_adjust_type,
    )

    if config.label_adjust_type == config.feature_adjust_type:
        label_market = feature_market
    else:
        label_market = load_market_matrices_from_manager(
            manager,
            start_date=load_start,
            end_date=load_end,
            adjust_type=config.label_adjust_type,
        )

    if config.execution_adjust_type == config.feature_adjust_type:
        execution_market = feature_market
    elif config.execution_adjust_type == config.label_adjust_type:
        execution_market = label_market
    else:
        execution_market = load_market_matrices_from_manager(
            manager,
            start_date=load_start,
            end_date=load_end,
            adjust_type=config.execution_adjust_type,
        )

    basic = load_basic_matrices_from_manager(manager, start_date=load_start, end_date=load_end)
    moneyflow = load_moneyflow_matrices_from_manager(manager, start_date=load_start, end_date=load_end)
    eligibility = load_eligibility_matrices_from_manager(manager, start_date=load_start, end_date=load_end)

    return feature_market, label_market, execution_market, basic, moneyflow, eligibility


# ============================================================
# Feature building and normalization
# ============================================================

def get_universe_mask(eligibility: Optional[Mapping[str, pd.DataFrame]]) -> Optional[pd.DataFrame]:
    """Extract is_eligible mask from eligibility dict if available."""
    if eligibility is None:
        return None
    if "is_eligible" not in eligibility:
        return None
    return eligibility["is_eligible"].astype(bool)


def get_buyable_mask(eligibility: Optional[Mapping[str, pd.DataFrame]]) -> Optional[pd.DataFrame]:
    """Extract can_buy mask from eligibility dict if available."""
    if eligibility is None:
        return None
    if "can_buy" not in eligibility:
        return None
    return eligibility["can_buy"].astype(bool)


def get_sellable_mask(eligibility: Optional[Mapping[str, pd.DataFrame]]) -> Optional[pd.DataFrame]:
    """Extract can_sell mask from eligibility dict if available."""
    if eligibility is None:
        return None
    if "can_sell" not in eligibility:
        return None
    return eligibility["can_sell"].astype(bool)


def apply_mask_to_factor_dict(
    factors: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Set factor values to NaN where mask is False."""
    out: dict[str, pd.DataFrame] = {}
    m = mask.astype(bool)
    for name, df in factors.items():
        aligned, mask_aligned = fl.align_frames(df, m.astype(float))
        out[name] = aligned.where(mask_aligned.astype(bool))
    return out


def normalize_factor_dict(
    factors: Mapping[str, pd.DataFrame],
    *,
    winsorize: bool = True,
    standardize: bool = True,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    preserve_binary_factors: bool = True,
    binary_factor_names: Iterable[str] = BINARY_FACTOR_NAMES,
) -> dict[str, pd.DataFrame]:
    """
    Normalize factors by date cross-section.

    Default:
        winsorize 1%/99%, then z-score.

    Binary factor names such as ep_is_loss can be preserved as 0/1.
    """
    binary_set = set(binary_factor_names)
    out: dict[str, pd.DataFrame] = {}

    for name, df in factors.items():
        x = fl.clean_inf(fl.as_float_frame(df))

        if preserve_binary_factors and name in binary_set:
            out[name] = x
            continue

        if winsorize:
            x = fl.winsorize_cross_section(x, lower_q=lower_q, upper_q=upper_q)
        if standardize:
            x = fl.zscore_cross_section(x)

        out[name] = x

    return out


def build_feature_factors_from_matrices(
    market: Mapping[str, pd.DataFrame],
    basic: Optional[Mapping[str, pd.DataFrame]] = None,
    moneyflow: Optional[Mapping[str, pd.DataFrame]] = None,
    eligibility: Optional[Mapping[str, pd.DataFrame]] = None,
    config: FactorPipelineConfig = FactorPipelineConfig(),
) -> dict[str, pd.DataFrame]:
    """Compute feature factors from raw matrices."""
    factor_config = fl.FactorConfig(
        windows=tuple(config.windows),
        yz_window=int(config.yz_window),
        annualization=int(config.annualization),
        min_periods_ratio=float(config.min_periods_ratio),
        amount_unit_scale=float(config.amount_unit_scale),
        eps=float(config.eps),
    )

    factors = fl.build_all_factors(
        market=market,
        basic=basic,
        moneyflow=moneyflow,
        config=factor_config,
    )

    universe_mask = get_universe_mask(eligibility)
    if config.mask_features_by_universe and universe_mask is not None:
        factors = apply_mask_to_factor_dict(factors, universe_mask)

    if config.winsorize_features or config.standardize_features:
        factors = normalize_factor_dict(
            factors,
            winsorize=config.winsorize_features,
            standardize=config.standardize_features,
            lower_q=config.winsor_lower_q,
            upper_q=config.winsor_upper_q,
            preserve_binary_factors=config.preserve_binary_factors,
        )

    return filter_dict_by_signal_dates(
        factors,
        start_date=config.start_date,
        end_date=config.end_date,
    )


# ============================================================
# Label building
# ============================================================

def build_label_matrices_from_matrices(
    label_market: Mapping[str, pd.DataFrame],
    execution_market: Optional[Mapping[str, pd.DataFrame]] = None,
    eligibility: Optional[Mapping[str, pd.DataFrame]] = None,
    config: FactorPipelineConfig = FactorPipelineConfig(),
) -> dict[str, pd.DataFrame]:
    """
    Build label matrices from market and eligibility/trade-status matrices.

    close-to-close labels use label_market["close"].
    execution labels use execution_market OHLC if provided.
    """
    lb_config = lb.LabelConfig(
        buy_offset=int(config.buy_offset),
        sell_offset=int(config.sell_offset),
        eps=float(config.eps),
    )

    universe_mask = get_universe_mask(eligibility) if config.mask_labels_by_universe else None
    buyable_mask = get_buyable_mask(eligibility) if config.mask_labels_by_trade_status else None
    sellable_mask = get_sellable_mask(eligibility) if config.mask_labels_by_trade_status else None

    labels: dict[str, pd.DataFrame] = {}

    if config.include_close_to_close_label:
        labels.update(lb.build_close_to_close_labels(
            close=label_market["close"],
            universe_mask=universe_mask,
            buyable_mask=buyable_mask,
            sellable_mask=sellable_mask,
            config=lb_config,
            prefix=f"t{config.buy_offset}_t{config.sell_offset}",
        ))

    if config.include_execution_label:
        if execution_market is None:
            execution_market = label_market

        required = ("open", "high", "low", "close")
        missing = [k for k in required if k not in execution_market]
        if missing:
            raise KeyError(f"execution_market missing required keys: {missing}")

        buy_price = lb.calc_buy_exec_price_max_close_avg(
            execution_market["open"],
            execution_market["high"],
            execution_market["low"],
            execution_market["close"],
        )
        sell_price = lb.calc_sell_exec_price_min_close_avg(
            execution_market["open"],
            execution_market["high"],
            execution_market["low"],
            execution_market["close"],
        )

        labels.update(lb.build_execution_labels(
            buy_price=buy_price,
            sell_price=sell_price,
            universe_mask=universe_mask,
            buyable_mask=buyable_mask,
            sellable_mask=sellable_mask,
            fee_rate_buy=float(config.fee_rate_buy),
            fee_rate_sell=float(config.fee_rate_sell),
            config=lb_config,
            prefix=f"exec_t{config.buy_offset}_t{config.sell_offset}",
        ))

    return filter_dict_by_signal_dates(
        labels,
        start_date=config.start_date,
        end_date=config.end_date,
    )


# ============================================================
# Panel alignment and dataset-style helpers
# ============================================================

def build_feature_panel(
    factors: Mapping[str, pd.DataFrame],
    config: FactorPipelineConfig = FactorPipelineConfig(),
) -> pd.DataFrame:
    """Stack factor matrices into a MultiIndex feature panel."""
    return fl.stack_factor_dict(
        factors,
        drop_all_nan_rows=config.drop_all_nan_feature_rows,
        sort_index=True,
    )


def build_label_panel(
    labels: Mapping[str, pd.DataFrame],
    config: FactorPipelineConfig = FactorPipelineConfig(),
) -> pd.DataFrame:
    """Stack label matrices into a MultiIndex label panel."""
    return lb.stack_label_dict(
        labels,
        drop_all_nan_rows=config.drop_all_nan_label_rows,
        sort_index=True,
    )


def join_feature_label_panels(
    feature_panel: pd.DataFrame,
    label_panel: pd.DataFrame,
    how: str = "inner",
    label_prefix: str = "",
) -> pd.DataFrame:
    """
    Join feature and label panels on [trade_date, ts_code].

    label_prefix can be used to avoid column name collisions.
    """
    if not isinstance(feature_panel.index, pd.MultiIndex):
        raise TypeError("feature_panel.index must be MultiIndex [trade_date, ts_code]")
    if not isinstance(label_panel.index, pd.MultiIndex):
        raise TypeError("label_panel.index must be MultiIndex [trade_date, ts_code]")

    labels = label_panel.add_prefix(label_prefix) if label_prefix else label_panel
    return feature_panel.join(labels, how=how)


def get_available_signal_dates(
    feature_panel: pd.DataFrame,
    label_panel: Optional[pd.DataFrame] = None,
    label_name: Optional[str] = None,
    min_stock_count: int = 512,
) -> list[pd.Timestamp]:
    """
    List signal dates with at least min_stock_count usable rows.

    If label_panel and label_name are provided, label must be non-null.
    Otherwise only feature rows are counted.
    """
    if not isinstance(feature_panel.index, pd.MultiIndex):
        raise TypeError("feature_panel.index must be MultiIndex [trade_date, ts_code]")

    usable = feature_panel.notna().any(axis=1)

    if label_panel is not None and label_name is not None:
        label_series = label_panel[label_name].reindex(feature_panel.index)
        usable &= label_series.notna()

    counts = usable.groupby(level="trade_date").sum()
    return [pd.Timestamp(dt) for dt, cnt in counts.items() if int(cnt) >= int(min_stock_count)]


def get_codes_on_date(
    panel: pd.DataFrame,
    trade_date: Any,
    require_non_na_any: bool = True,
) -> list[str]:
    """Return codes available on a given signal date from a panel."""
    if not isinstance(panel.index, pd.MultiIndex):
        raise TypeError("panel.index must be MultiIndex [trade_date, ts_code]")

    dt = pd.Timestamp(trade_date)
    try:
        sub = panel.xs(dt, level="trade_date")
    except KeyError:
        return []

    if require_non_na_any:
        sub = sub[sub.notna().any(axis=1)]

    return sub.index.astype(str).tolist()


def split_factor_names(
    factor_names: Iterable[str],
    scalar_factor_names: Iterable[str] = SCALAR_FACTOR_NAMES,
) -> tuple[list[str], list[str]]:
    """
    Split factor names into time-series and scalar groups.

    Scalar group uses explicit names. All other factors are treated as time-series.
    """
    scalar_set = set(scalar_factor_names)
    factor_names = list(factor_names)
    scalar = [x for x in factor_names if x in scalar_set]
    ts = [x for x in factor_names if x not in scalar_set]
    return ts, scalar


def make_model_input_arrays_for_date(
    factor_panel: pd.DataFrame,
    trade_date: Any,
    codes: Sequence[str],
    ts_factor_names: Sequence[str],
    scalar_factor_names: Sequence[str],
    seq_len: int = 128,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build numpy arrays for one date and a fixed code list.

    Returns
    -------
    x_ts:
        Shape [N, F_ts, T].
        The sequence ends at trade_date and uses the previous seq_len dates.

    x_scalar:
        Shape [N, F_scalar].
        Scalar factors are taken at trade_date only.

    codes_out:
        Codes actually returned, same order as input codes.

    Notes
    -----
    This is a lightweight bridge helper. A full PyTorch Dataset should be built
    in rank_dataset.py.
    """
    if not isinstance(factor_panel.index, pd.MultiIndex):
        raise TypeError("factor_panel.index must be MultiIndex [trade_date, ts_code]")

    dt = pd.Timestamp(trade_date)
    all_dates = pd.DatetimeIndex(factor_panel.index.get_level_values("trade_date").unique()).sort_values()
    if dt not in all_dates:
        raise KeyError(f"trade_date not found in factor_panel: {dt}")

    end_pos = all_dates.get_loc(dt)
    start_pos = max(0, end_pos - int(seq_len) + 1)
    seq_dates = all_dates[start_pos:end_pos + 1]

    # Left pad if history is shorter than seq_len.
    pad_len = int(seq_len) - len(seq_dates)

    codes_out = [str(c) for c in codes]
    ts_arrays = []

    for code in codes_out:
        idx = pd.MultiIndex.from_product([seq_dates, [code]], names=["trade_date", "ts_code"])
        sub = factor_panel.reindex(idx)

        ts_values = sub[list(ts_factor_names)].to_numpy(dtype=float)  # [T_actual, F_ts]
        if pad_len > 0:
            pad = np.full((pad_len, len(ts_factor_names)), np.nan, dtype=float)
            ts_values = np.vstack([pad, ts_values])

        ts_arrays.append(ts_values.T)  # [F_ts, T]

    x_ts = np.stack(ts_arrays, axis=0) if ts_arrays else np.empty((0, len(ts_factor_names), seq_len))

    scalar_idx = pd.MultiIndex.from_product([[dt], codes_out], names=["trade_date", "ts_code"])
    scalar_values = factor_panel.reindex(scalar_idx)[list(scalar_factor_names)].to_numpy(dtype=float)
    x_scalar = scalar_values.reshape(len(codes_out), len(scalar_factor_names))

    x_ts = np.nan_to_num(x_ts, nan=fill_value, posinf=fill_value, neginf=fill_value)
    x_scalar = np.nan_to_num(x_scalar, nan=fill_value, posinf=fill_value, neginf=fill_value)

    return x_ts.astype(np.float32), x_scalar.astype(np.float32), codes_out


# ============================================================
# High-level builders
# ============================================================

def build_bundle_from_matrices(
    feature_market: Mapping[str, pd.DataFrame],
    label_market: Optional[Mapping[str, pd.DataFrame]] = None,
    execution_market: Optional[Mapping[str, pd.DataFrame]] = None,
    basic: Optional[Mapping[str, pd.DataFrame]] = None,
    moneyflow: Optional[Mapping[str, pd.DataFrame]] = None,
    eligibility: Optional[Mapping[str, pd.DataFrame]] = None,
    config: FactorPipelineConfig = FactorPipelineConfig(),
) -> FeatureLabelBundle:
    """
    Build a FeatureLabelBundle from already prepared matrices.
    """
    if label_market is None:
        label_market = feature_market
    if execution_market is None:
        execution_market = label_market

    factors = build_feature_factors_from_matrices(
        market=feature_market,
        basic=basic,
        moneyflow=moneyflow,
        eligibility=eligibility,
        config=config,
    )

    labels = build_label_matrices_from_matrices(
        label_market=label_market,
        execution_market=execution_market,
        eligibility=eligibility,
        config=config,
    )

    feature_panel = build_feature_panel(factors, config=config)
    label_panel = build_label_panel(labels, config=config)

    ts_names, scalar_names = split_factor_names(factors.keys())

    metadata = {
        "config": asdict(config),
        "factor_names": list(factors.keys()),
        "label_names": list(labels.keys()),
        "ts_factor_names": ts_names,
        "scalar_factor_names": scalar_names,
        "feature_panel_shape": tuple(feature_panel.shape),
        "label_panel_shape": tuple(label_panel.shape),
        "factor_count": len(factors),
        "label_count": len(labels),
    }

    return FeatureLabelBundle(
        factors=dict(factors),
        labels=dict(labels),
        feature_panel=feature_panel,
        label_panel=label_panel,
        market=dict(feature_market),
        basic=dict(basic or {}),
        moneyflow=dict(moneyflow or {}),
        eligibility=dict(eligibility or {}),
        metadata=metadata,
    )


def build_bundle_from_manager(
    manager: Any,
    config: FactorPipelineConfig = FactorPipelineConfig(),
    keep_raw_matrices: bool = False,
) -> FeatureLabelBundle:
    """
    Build a FeatureLabelBundle directly from StockDataManager.
    """
    feature_market, label_market, execution_market, basic, moneyflow, eligibility = load_all_matrices_from_manager(
        manager=manager,
        config=config,
    )

    bundle = build_bundle_from_matrices(
        feature_market=feature_market,
        label_market=label_market,
        execution_market=execution_market,
        basic=basic,
        moneyflow=moneyflow,
        eligibility=eligibility,
        config=config,
    )

    if keep_raw_matrices:
        bundle.market = {
            "feature_open": feature_market.get("open", pd.DataFrame()),
            "feature_high": feature_market.get("high", pd.DataFrame()),
            "feature_low": feature_market.get("low", pd.DataFrame()),
            "feature_close": feature_market.get("close", pd.DataFrame()),
            "label_close": label_market.get("close", pd.DataFrame()),
            "execution_close": execution_market.get("close", pd.DataFrame()),
        }
    else:
        bundle.market = {}

    return bundle


# ============================================================
# Cache I/O
# ============================================================

def save_bundle(bundle: FeatureLabelBundle, path: str | Path) -> Path:
    """
    Save a FeatureLabelBundle to pickle.

    Pickle is used because it preserves DataFrame indexes, metadata, and dicts
    without requiring pyarrow/fastparquet.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def load_bundle(path: str | Path) -> FeatureLabelBundle:
    """Load a FeatureLabelBundle from pickle."""
    p = Path(path)
    with p.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, FeatureLabelBundle):
        raise TypeError(f"cache object is not FeatureLabelBundle: {type(obj)!r}")
    return obj


def save_panels(
    bundle: FeatureLabelBundle,
    directory: str | Path,
    feature_name: str = "feature_panel.pkl",
    label_name: str = "label_panel.pkl",
    metadata_name: str = "metadata.pkl",
) -> dict[str, Path]:
    """
    Save feature panel, label panel, and metadata separately as pickle files.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)

    feature_path = d / feature_name
    label_path = d / label_name
    metadata_path = d / metadata_name

    bundle.feature_panel.to_pickle(feature_path)
    bundle.label_panel.to_pickle(label_path)

    with metadata_path.open("wb") as f:
        pickle.dump(bundle.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "feature_panel": feature_path,
        "label_panel": label_path,
        "metadata": metadata_path,
    }


def load_panels(
    directory: str | Path,
    feature_name: str = "feature_panel.pkl",
    label_name: str = "label_panel.pkl",
    metadata_name: str = "metadata.pkl",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load feature panel, label panel, and metadata saved by save_panels."""
    d = Path(directory)
    feature_panel = pd.read_pickle(d / feature_name)
    label_panel = pd.read_pickle(d / label_name)

    with (d / metadata_name).open("rb") as f:
        metadata = pickle.load(f)

    return feature_panel, label_panel, metadata


__all__ = [
    "SCALAR_FACTOR_NAMES",
    "BINARY_FACTOR_NAMES",
    "DEFAULT_REQUIRED_MARKET_FIELDS",
    "DEFAULT_BASIC_FIELDS",
    "DEFAULT_MONEYFLOW_FIELDS",
    "DEFAULT_ELIGIBILITY_FIELDS",
    "FactorPipelineConfig",
    "FeatureLabelBundle",
    "normalize_yyyymmdd",
    "build_field_matrix",
    "build_matrices_from_long_table",
    "bool_matrix_from_numeric",
    "load_market_matrices_from_manager",
    "load_basic_matrices_from_manager",
    "load_moneyflow_matrices_from_manager",
    "load_eligibility_matrices_from_manager",
    "load_all_matrices_from_manager",
    "get_universe_mask",
    "get_buyable_mask",
    "get_sellable_mask",
    "apply_mask_to_factor_dict",
    "normalize_factor_dict",
    "build_feature_factors_from_matrices",
    "build_label_matrices_from_matrices",
    "build_feature_panel",
    "build_label_panel",
    "join_feature_label_panels",
    "get_available_signal_dates",
    "get_codes_on_date",
    "split_factor_names",
    "make_model_input_arrays_for_date",
    "build_bundle_from_matrices",
    "build_bundle_from_manager",
    "save_bundle",
    "load_bundle",
    "save_panels",
    "load_panels",
]
