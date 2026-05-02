"""
label_builder.py

Label construction utilities for the cross-sectional ranking model.

Design
------
1. Pure functions only: no database reads/writes and no model dependencies.
2. Matrix convention:
   - index: trade_date
   - columns: ts_code
   - values: field values
3. Main supervised target:
   signal date t uses information up to t.
   label at t is the forward log return from t+1 to t+6:

       label_ret_t1_t6[t] = log(price[t+6] / price[t+1])

4. The returned label matrices are meant for training only. They must never be
   merged into features for dates <= t during inference/backtesting.

Recommended usage
-----------------
labels = build_close_to_close_labels(adj_close)

label_ret = labels["label_ret_t1_t6"]
label_rank_pct = labels["label_rank_pct_t1_t6"]
label_valid = labels["label_valid_t1_t6"]

For a sampled 512-stock training batch on one date:
local = make_local_rank_labels(label_ret.loc[date, sampled_codes])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class LabelConfig:
    """Default label configuration."""

    buy_offset: int = 1
    sell_offset: int = 6
    eps: float = EPS
    rank_ascending: bool = True  # low return -> low rank, high return -> high rank
    min_valid_count: int = 2


# ============================================================
# Basic helpers
# ============================================================

def as_float_frame(x: pd.DataFrame | pd.Series, name: Optional[str] = None) -> pd.DataFrame:
    """Convert a Series/DataFrame to a float DataFrame."""
    if isinstance(x, pd.Series):
        x = x.to_frame(name=name or x.name)
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"expected pandas DataFrame or Series, got {type(x)!r}")
    out = x.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.astype(float)


def align_frames(*frames: pd.DataFrame, join: str = "outer") -> tuple[pd.DataFrame, ...]:
    """Align multiple wide matrices by index and columns."""
    dfs = [as_float_frame(f) for f in frames]
    if len(dfs) == 0:
        return tuple()

    if join not in {"outer", "inner"}:
        raise ValueError("join must be 'outer' or 'inner'")

    idx = dfs[0].index
    cols = dfs[0].columns
    for df in dfs[1:]:
        if join == "outer":
            idx = idx.union(df.index)
            cols = cols.union(df.columns)
        else:
            idx = idx.intersection(df.index)
            cols = cols.intersection(df.columns)

    idx = pd.Index(idx).sort_values()
    cols = pd.Index(cols).sort_values()
    return tuple(df.reindex(index=idx, columns=cols) for df in dfs)


def clean_inf(df: pd.DataFrame) -> pd.DataFrame:
    """Replace positive/negative infinity with NaN."""
    return df.replace([np.inf, -np.inf], np.nan)


def require_positive_price(price: pd.DataFrame) -> pd.DataFrame:
    """Return price with non-positive values set to NaN."""
    x = as_float_frame(price)
    return x.where(x > 0.0)


def safe_log_ratio(numer: pd.DataFrame, denom: pd.DataFrame) -> pd.DataFrame:
    """Safe log(numer / denom); invalid non-positive entries become NaN."""
    n, d = align_frames(numer, denom)
    valid = (n > 0.0) & (d > 0.0)
    out = pd.DataFrame(np.nan, index=n.index, columns=n.columns, dtype=float)
    out[valid] = np.log(n[valid] / d[valid])
    return clean_inf(out)


def make_boolean_frame(mask: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Convert numeric/bool matrix to bool DataFrame while preserving NaNs as False."""
    if isinstance(mask, pd.Series):
        mask = mask.to_frame(name=mask.name)
    if not isinstance(mask, pd.DataFrame):
        raise TypeError(f"expected DataFrame or Series, got {type(mask)!r}")
    return mask.fillna(False).astype(bool)


def _validate_offsets(buy_offset: int, sell_offset: int) -> None:
    if buy_offset < 0 or sell_offset < 0:
        raise ValueError("buy_offset and sell_offset must be non-negative")
    if sell_offset <= buy_offset:
        raise ValueError("sell_offset must be greater than buy_offset")


# ============================================================
# Forward return labels
# ============================================================

def calc_forward_log_return(
    price: pd.DataFrame,
    buy_offset: int = 1,
    sell_offset: int = 6,
) -> pd.DataFrame:
    """
    Forward log return from t+buy_offset to t+sell_offset.

    For the default weekly strategy:
        label[t] = log(price[t+6] / price[t+1])
    """
    _validate_offsets(buy_offset, sell_offset)
    px = require_positive_price(price)
    buy_px = px.shift(-buy_offset)
    sell_px = px.shift(-sell_offset)
    return safe_log_ratio(sell_px, buy_px)


def calc_forward_simple_return(
    price: pd.DataFrame,
    buy_offset: int = 1,
    sell_offset: int = 6,
) -> pd.DataFrame:
    """
    Forward simple return from t+buy_offset to t+sell_offset.

    label[t] = price[t+sell_offset] / price[t+buy_offset] - 1
    """
    _validate_offsets(buy_offset, sell_offset)
    px = require_positive_price(price)
    buy_px = px.shift(-buy_offset)
    sell_px = px.shift(-sell_offset)
    out = sell_px / buy_px - 1.0
    return clean_inf(out)


def calc_forward_execution_log_return(
    buy_price: pd.DataFrame,
    sell_price: pd.DataFrame,
    buy_offset: int = 1,
    sell_offset: int = 6,
    fee_rate_buy: float = 0.0,
    fee_rate_sell: float = 0.0,
) -> pd.DataFrame:
    """
    Conservative execution-return label.

    buy_price should already be the execution buy matrix, for example:
        max(close, avg)

    sell_price should already be the execution sell matrix, for example:
        min(close, avg)

    label[t] = log(
        sell_price[t+sell_offset] * (1 - fee_rate_sell)
        /
        buy_price[t+buy_offset] * (1 + fee_rate_buy)
    )
    """
    _validate_offsets(buy_offset, sell_offset)
    bp, sp = align_frames(buy_price, sell_price)
    bp = require_positive_price(bp).shift(-buy_offset)
    sp = require_positive_price(sp).shift(-sell_offset)

    adjusted_buy = bp * (1.0 + float(fee_rate_buy))
    adjusted_sell = sp * (1.0 - float(fee_rate_sell))
    return safe_log_ratio(adjusted_sell, adjusted_buy)


def calc_buy_exec_price_max_close_avg(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    """
    Buy execution price helper:
        buy_price = max(close, avg)
        avg = (open + high + low + close) / 4
    """
    o, h, l, c = align_frames(open_, high, low, close)
    avg = (o + h + l + c) / 4.0
    return c.where(c >= avg, avg)


def calc_sell_exec_price_min_close_avg(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sell execution price helper:
        sell_price = min(close, avg)
        avg = (open + high + low + close) / 4
    """
    o, h, l, c = align_frames(open_, high, low, close)
    avg = (o + h + l + c) / 4.0
    return c.where(c <= avg, avg)


# ============================================================
# Validity masks
# ============================================================

def calc_label_valid_mask(
    label_ret: pd.DataFrame,
    universe_mask: Optional[pd.DataFrame] = None,
    buyable_mask: Optional[pd.DataFrame] = None,
    sellable_mask: Optional[pd.DataFrame] = None,
    buy_offset: int = 1,
    sell_offset: int = 6,
) -> pd.DataFrame:
    """
    Build a label-valid mask.

    Parameters
    ----------
    label_ret:
        Forward return label at signal date t.
    universe_mask:
        Optional mask at signal date t, such as is_eligible[t].
    buyable_mask:
        Optional can_buy mask at real buy date. If passed as a daily matrix,
        this function shifts it to signal date t:
            buyable_mask_at_t = can_buy[t+buy_offset]
    sellable_mask:
        Optional can_sell mask at real sell date. If passed as a daily matrix,
        this function shifts it to signal date t:
            sellable_mask_at_t = can_sell[t+sell_offset]

    Notes
    -----
    For pure model training, you may choose not to pass buyable/sellable masks
    and only require future returns to be finite. For execution-aware labels,
    passing them is stricter.
    """
    _validate_offsets(buy_offset, sell_offset)

    ret = as_float_frame(label_ret)
    valid = ret.notna() & np.isfinite(ret)

    if universe_mask is not None:
        uni = make_boolean_frame(universe_mask).reindex(index=ret.index, columns=ret.columns).fillna(False)
        valid &= uni

    if buyable_mask is not None:
        buyable = make_boolean_frame(buyable_mask).reindex(index=ret.index, columns=ret.columns)
        buyable_at_signal = buyable.shift(-buy_offset).fillna(False)
        valid &= buyable_at_signal

    if sellable_mask is not None:
        sellable = make_boolean_frame(sellable_mask).reindex(index=ret.index, columns=ret.columns)
        sellable_at_signal = sellable.shift(-sell_offset).fillna(False)
        valid &= sellable_at_signal

    return valid.astype(bool)


def mask_labels(label: pd.DataFrame, valid_mask: pd.DataFrame) -> pd.DataFrame:
    """Set labels to NaN where valid_mask is False."""
    y, m = align_frames(label, valid_mask.astype(float))
    return y.where(m.astype(bool))


# ============================================================
# Cross-sectional rank labels
# ============================================================

def calc_cross_section_rank(
    label_ret: pd.DataFrame,
    valid_mask: Optional[pd.DataFrame] = None,
    ascending: bool = True,
    method: str = "average",
    min_valid_count: int = 2,
) -> pd.DataFrame:
    """
    Cross-sectional rank by date.

    With ascending=True:
        lowest return -> rank 1
        highest return -> rank N
    """
    y = as_float_frame(label_ret)
    if valid_mask is not None:
        y = mask_labels(y, valid_mask)

    rank = y.rank(axis=1, ascending=ascending, method=method, na_option="keep")

    count = y.notna().sum(axis=1)
    bad_dates = count < int(min_valid_count)
    if bad_dates.any():
        rank.loc[bad_dates, :] = np.nan

    return rank


def calc_cross_section_rank_pct(
    label_ret: pd.DataFrame,
    valid_mask: Optional[pd.DataFrame] = None,
    ascending: bool = True,
    method: str = "average",
    min_valid_count: int = 2,
) -> pd.DataFrame:
    """
    Cross-sectional percentile rank by date.

    With ascending=True:
        lowest return -> close to 1/N
        highest return -> 1.0
    """
    y = as_float_frame(label_ret)
    if valid_mask is not None:
        y = mask_labels(y, valid_mask)

    rank_pct = y.rank(axis=1, ascending=ascending, method=method, pct=True, na_option="keep")

    count = y.notna().sum(axis=1)
    bad_dates = count < int(min_valid_count)
    if bad_dates.any():
        rank_pct.loc[bad_dates, :] = np.nan

    return rank_pct


def calc_cross_section_rank_centered(
    label_ret: pd.DataFrame,
    valid_mask: Optional[pd.DataFrame] = None,
    ascending: bool = True,
    method: str = "average",
    min_valid_count: int = 2,
) -> pd.DataFrame:
    """
    Cross-sectional rank transformed to approximately [-1, 1].

    Formula:
        centered = 2 * rank_pct - 1

    Highest return is near +1 when ascending=True.
    """
    pct = calc_cross_section_rank_pct(
        label_ret=label_ret,
        valid_mask=valid_mask,
        ascending=ascending,
        method=method,
        min_valid_count=min_valid_count,
    )
    return 2.0 * pct - 1.0


def make_local_rank_labels(
    returns: pd.Series,
    valid_mask: Optional[pd.Series] = None,
    ascending: bool = True,
    method: str = "average",
    output: str = "rank_pct",
) -> pd.Series:
    """
    Create local rank labels for one sampled cross-section.

    This is intended for stochastic 512-stock training batches.

    Parameters
    ----------
    returns:
        Future returns for sampled stocks on one signal date.
    valid_mask:
        Optional boolean mask for the sampled stocks.
    output:
        - "rank": raw local rank, 1..N
        - "rank_pct": percentile rank
        - "rank_centered": 2 * rank_pct - 1
        - "zscore": return z-score inside the local sample
        - "raw": raw returns with invalid positions as NaN
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    y = pd.to_numeric(returns, errors="coerce").astype(float)
    y = y.replace([np.inf, -np.inf], np.nan)

    if valid_mask is not None:
        m = valid_mask.reindex(y.index).fillna(False).astype(bool)
        y = y.where(m)

    if output == "raw":
        return y

    if y.notna().sum() < 2:
        return pd.Series(np.nan, index=y.index, name=output)

    if output == "rank":
        return y.rank(ascending=ascending, method=method, na_option="keep").rename(output)

    rank_pct = y.rank(ascending=ascending, method=method, pct=True, na_option="keep")

    if output == "rank_pct":
        return rank_pct.rename(output)

    if output == "rank_centered":
        return (2.0 * rank_pct - 1.0).rename(output)

    if output == "zscore":
        mu = y.mean()
        sigma = y.std(ddof=0)
        if not np.isfinite(sigma) or sigma <= EPS:
            return pd.Series(np.nan, index=y.index, name=output)
        return ((y - mu) / sigma).rename(output)

    raise ValueError("output must be one of: rank, rank_pct, rank_centered, zscore, raw")


# ============================================================
# Top-k diagnostic labels
# ============================================================

def calc_topk_mask(
    score_or_return: pd.DataFrame,
    k: int,
    valid_mask: Optional[pd.DataFrame] = None,
    largest: bool = True,
) -> pd.DataFrame:
    """
    Mark top-k names by date.

    This is mostly useful for label diagnostics, e.g. true future top-k.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    x = as_float_frame(score_or_return)
    if valid_mask is not None:
        x = mask_labels(x, valid_mask)

    out = pd.DataFrame(False, index=x.index, columns=x.columns)

    for dt, row in x.iterrows():
        s = row.dropna()
        if s.empty:
            continue
        selected = s.nlargest(k).index if largest else s.nsmallest(k).index
        out.loc[dt, selected] = True

    return out


def calc_forward_return_quantile_bucket(
    label_ret: pd.DataFrame,
    valid_mask: Optional[pd.DataFrame] = None,
    n_bins: int = 10,
    ascending: bool = True,
    min_valid_count: int = 10,
) -> pd.DataFrame:
    """
    Convert forward returns into per-date quantile buckets 0..n_bins-1.

    Higher bucket means higher return when ascending=True.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    y = as_float_frame(label_ret)
    if valid_mask is not None:
        y = mask_labels(y, valid_mask)

    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns, dtype=float)

    for dt, row in y.iterrows():
        s = row.dropna()
        if len(s) < max(min_valid_count, n_bins):
            continue

        rank_pct = s.rank(ascending=ascending, method="first", pct=True)
        bucket = np.ceil(rank_pct * n_bins).astype(int) - 1
        bucket = bucket.clip(0, n_bins - 1)
        out.loc[dt, bucket.index] = bucket.astype(float)

    return out


# ============================================================
# High-level builders
# ============================================================

def build_close_to_close_labels(
    close: pd.DataFrame,
    universe_mask: Optional[pd.DataFrame] = None,
    buyable_mask: Optional[pd.DataFrame] = None,
    sellable_mask: Optional[pd.DataFrame] = None,
    config: LabelConfig = LabelConfig(),
    prefix: str = "t1_t6",
) -> dict[str, pd.DataFrame]:
    """
    Build close-to-close label matrices.

    Default:
        label_ret_t1_t6[t] = log(close[t+6] / close[t+1])
    """
    label_ret = calc_forward_log_return(
        close,
        buy_offset=config.buy_offset,
        sell_offset=config.sell_offset,
    )

    valid = calc_label_valid_mask(
        label_ret=label_ret,
        universe_mask=universe_mask,
        buyable_mask=buyable_mask,
        sellable_mask=sellable_mask,
        buy_offset=config.buy_offset,
        sell_offset=config.sell_offset,
    )

    label_ret_masked = mask_labels(label_ret, valid)
    rank = calc_cross_section_rank(
        label_ret_masked,
        valid_mask=valid,
        ascending=config.rank_ascending,
        min_valid_count=config.min_valid_count,
    )
    rank_pct = calc_cross_section_rank_pct(
        label_ret_masked,
        valid_mask=valid,
        ascending=config.rank_ascending,
        min_valid_count=config.min_valid_count,
    )
    rank_centered = 2.0 * rank_pct - 1.0

    return {
        f"label_ret_{prefix}": label_ret_masked,
        f"label_rank_{prefix}": rank,
        f"label_rank_pct_{prefix}": rank_pct,
        f"label_rank_centered_{prefix}": rank_centered,
        f"label_valid_{prefix}": valid,
    }


def build_execution_labels(
    buy_price: pd.DataFrame,
    sell_price: pd.DataFrame,
    universe_mask: Optional[pd.DataFrame] = None,
    buyable_mask: Optional[pd.DataFrame] = None,
    sellable_mask: Optional[pd.DataFrame] = None,
    fee_rate_buy: float = 0.0,
    fee_rate_sell: float = 0.0,
    config: LabelConfig = LabelConfig(),
    prefix: str = "exec_t1_t6",
) -> dict[str, pd.DataFrame]:
    """
    Build execution-aware label matrices.

    Default:
        label_exec_t1_t6[t] =
            log(sell_price[t+6] * (1 - fee_sell)
                / buy_price[t+1] * (1 + fee_buy))
    """
    label_ret = calc_forward_execution_log_return(
        buy_price=buy_price,
        sell_price=sell_price,
        buy_offset=config.buy_offset,
        sell_offset=config.sell_offset,
        fee_rate_buy=fee_rate_buy,
        fee_rate_sell=fee_rate_sell,
    )

    valid = calc_label_valid_mask(
        label_ret=label_ret,
        universe_mask=universe_mask,
        buyable_mask=buyable_mask,
        sellable_mask=sellable_mask,
        buy_offset=config.buy_offset,
        sell_offset=config.sell_offset,
    )

    label_ret_masked = mask_labels(label_ret, valid)
    rank = calc_cross_section_rank(
        label_ret_masked,
        valid_mask=valid,
        ascending=config.rank_ascending,
        min_valid_count=config.min_valid_count,
    )
    rank_pct = calc_cross_section_rank_pct(
        label_ret_masked,
        valid_mask=valid,
        ascending=config.rank_ascending,
        min_valid_count=config.min_valid_count,
    )
    rank_centered = 2.0 * rank_pct - 1.0

    return {
        f"label_ret_{prefix}": label_ret_masked,
        f"label_rank_{prefix}": rank,
        f"label_rank_pct_{prefix}": rank_pct,
        f"label_rank_centered_{prefix}": rank_centered,
        f"label_valid_{prefix}": valid,
    }


def build_labels_from_ohlc(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    universe_mask: Optional[pd.DataFrame] = None,
    buyable_mask: Optional[pd.DataFrame] = None,
    sellable_mask: Optional[pd.DataFrame] = None,
    fee_rate_buy: float = 0.0,
    fee_rate_sell: float = 0.0,
    config: LabelConfig = LabelConfig(),
    include_close_to_close: bool = True,
    include_execution: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Build both close-to-close and execution-aware labels from OHLC matrices.

    Execution price convention:
        buy_price = max(close, avg)
        sell_price = min(close, avg)
        avg = (open + high + low + close) / 4
    """
    labels: dict[str, pd.DataFrame] = {}

    if include_close_to_close:
        labels.update(build_close_to_close_labels(
            close=close,
            universe_mask=universe_mask,
            buyable_mask=buyable_mask,
            sellable_mask=sellable_mask,
            config=config,
            prefix=f"t{config.buy_offset}_t{config.sell_offset}",
        ))

    if include_execution:
        buy_price = calc_buy_exec_price_max_close_avg(open_, high, low, close)
        sell_price = calc_sell_exec_price_min_close_avg(open_, high, low, close)
        labels.update(build_execution_labels(
            buy_price=buy_price,
            sell_price=sell_price,
            universe_mask=universe_mask,
            buyable_mask=buyable_mask,
            sellable_mask=sellable_mask,
            fee_rate_buy=fee_rate_buy,
            fee_rate_sell=fee_rate_sell,
            config=config,
            prefix=f"exec_t{config.buy_offset}_t{config.sell_offset}",
        ))

    return labels


# ============================================================
# Panel utilities
# ============================================================

def _stack_keep_na(df: pd.DataFrame) -> pd.Series:
    """Stack a wide DataFrame while preserving NaNs."""
    x = as_float_frame(df)
    try:
        return x.stack(future_stack=True)
    except TypeError:
        return x.stack(dropna=False)


def stack_label_dict(
    labels: Mapping[str, pd.DataFrame],
    *,
    drop_all_nan_rows: bool = False,
    sort_index: bool = True,
) -> pd.DataFrame:
    """
    Convert a label dictionary into a MultiIndex panel.

    Output index:
        [trade_date, ts_code]

    Output columns:
        label names
    """
    series_list = []
    for name, df in labels.items():
        if df.dtypes.apply(lambda x: x == bool).all():
            s = _stack_keep_na(df.astype(float)).rename(name)
        else:
            s = _stack_keep_na(df).rename(name)
        series_list.append(s)

    if not series_list:
        return pd.DataFrame()

    panel = pd.concat(series_list, axis=1)
    panel.index = panel.index.set_names(["trade_date", "ts_code"])

    if drop_all_nan_rows:
        panel = panel.dropna(how="all")
    if sort_index:
        panel = panel.sort_index()

    return panel


def unstack_label_panel(panel: pd.DataFrame, label_name: str) -> pd.DataFrame:
    """Recover one wide label matrix from a stacked panel."""
    if label_name not in panel.columns:
        raise KeyError(f"label_name not in panel columns: {label_name}")
    if not isinstance(panel.index, pd.MultiIndex):
        raise TypeError("panel.index must be MultiIndex [trade_date, ts_code]")
    return panel[label_name].unstack("ts_code")


def list_expected_label_names(
    buy_offset: int = 1,
    sell_offset: int = 6,
    include_close_to_close: bool = True,
    include_execution: bool = True,
) -> list[str]:
    """List expected label names under the default weekly specification."""
    names: list[str] = []

    if include_close_to_close:
        prefix = f"t{buy_offset}_t{sell_offset}"
        names.extend([
            f"label_ret_{prefix}",
            f"label_rank_{prefix}",
            f"label_rank_pct_{prefix}",
            f"label_rank_centered_{prefix}",
            f"label_valid_{prefix}",
        ])

    if include_execution:
        prefix = f"exec_t{buy_offset}_t{sell_offset}"
        names.extend([
            f"label_ret_{prefix}",
            f"label_rank_{prefix}",
            f"label_rank_pct_{prefix}",
            f"label_rank_centered_{prefix}",
            f"label_valid_{prefix}",
        ])

    return names


__all__ = [
    "EPS",
    "LabelConfig",
    "as_float_frame",
    "align_frames",
    "clean_inf",
    "safe_log_ratio",
    "calc_forward_log_return",
    "calc_forward_simple_return",
    "calc_forward_execution_log_return",
    "calc_buy_exec_price_max_close_avg",
    "calc_sell_exec_price_min_close_avg",
    "calc_label_valid_mask",
    "mask_labels",
    "calc_cross_section_rank",
    "calc_cross_section_rank_pct",
    "calc_cross_section_rank_centered",
    "make_local_rank_labels",
    "calc_topk_mask",
    "calc_forward_return_quantile_bucket",
    "build_close_to_close_labels",
    "build_execution_labels",
    "build_labels_from_ohlc",
    "stack_label_dict",
    "unstack_label_panel",
    "list_expected_label_names",
]
