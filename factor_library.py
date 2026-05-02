"""
factor_library.py

Stage-1 factor library for a cross-sectional ranking model.

Design
------
1. Pure functions only: no database reads/writes and no model dependencies.
2. Matrix convention:
   - index: trade_date
   - columns: ts_code
   - values: field values
3. All factor functions return wide pandas DataFrames.
4. This module provides:
   - single-factor functions
   - grouped batch builders
   - factor-dict to MultiIndex panel utilities

Recommended data convention
---------------------------
Use adjusted prices for return/training factors.
Use raw prices for execution price, limit-up/limit-down, and trade-status logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12
DEFAULT_WINDOWS = (5, 10, 20, 60)
DEFAULT_YZ_WINDOW = 30
DEFAULT_ANNUALIZATION = 252


@dataclass(frozen=True)
class FactorConfig:
    windows: tuple[int, ...] = DEFAULT_WINDOWS
    yz_window: int = DEFAULT_YZ_WINDOW
    annualization: int = DEFAULT_ANNUALIZATION
    min_periods_ratio: float = 0.7
    amount_unit_scale: float = 1000.0
    eps: float = EPS


# ============================================================
# Helpers
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


def _floor_denominator(denom: pd.DataFrame, eps: float = EPS) -> pd.DataFrame:
    """Keep NaN as NaN and replace tiny non-null denominators by eps."""
    d = denom.copy()
    mask = d.notna() & (d.abs() < eps)
    return d.mask(mask, eps)


def safe_divide(numer: pd.DataFrame, denom: pd.DataFrame, eps: float = EPS) -> pd.DataFrame:
    """Safe divide: numer / denom, with small denominators floored by eps."""
    n, d = align_frames(numer, denom)
    d = _floor_denominator(d, eps=eps)
    return clean_inf(n / d)


def safe_log_ratio(numer: pd.DataFrame, denom: pd.DataFrame) -> pd.DataFrame:
    """Safe log(numer / denom); invalid non-positive entries become NaN."""
    n, d = align_frames(numer, denom)
    valid = (n > 0) & (d > 0)
    out = pd.DataFrame(np.nan, index=n.index, columns=n.columns, dtype=float)
    out[valid] = np.log(n[valid] / d[valid])
    return clean_inf(out)


def clip_frame(df: pd.DataFrame, lower: Optional[float] = None, upper: Optional[float] = None) -> pd.DataFrame:
    """Clip a frame after coercing it to float."""
    return as_float_frame(df).clip(lower=lower, upper=upper)


def rolling_min_periods(window: int, min_periods: Optional[int] = None, min_periods_ratio: float = 0.7) -> int:
    """Resolve min_periods from a window and a ratio."""
    if min_periods is not None:
        return int(min_periods)
    return max(1, int(np.ceil(window * min_periods_ratio)))


def winsorize_cross_section(
    df: pd.DataFrame,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """Winsorize each date's cross-section."""
    x = as_float_frame(df)
    lower = x.quantile(lower_q, axis=1)
    upper = x.quantile(upper_q, axis=1)
    return x.clip(lower=lower, upper=upper, axis=0)


def zscore_cross_section(df: pd.DataFrame, eps: float = EPS) -> pd.DataFrame:
    """Cross-sectional z-score by date."""
    x = as_float_frame(df)
    mu = x.mean(axis=1)
    sigma = x.std(axis=1, ddof=0)
    sigma = sigma.mask(sigma.abs() < eps, np.nan)
    return clean_inf(x.sub(mu, axis=0).div(sigma, axis=0))


def rank_pct_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank by date."""
    return as_float_frame(df).rank(axis=1, pct=True)


def require_keys(data: Mapping[str, pd.DataFrame], keys: Sequence[str], group_name: str = "data") -> None:
    """Validate required mapping keys."""
    missing = [k for k in keys if k not in data or data[k] is None]
    if missing:
        raise KeyError(f"{group_name} missing required keys: {missing}")


# ============================================================
# Pure time-series factors
# ============================================================

def calc_log_return(close: pd.DataFrame) -> pd.DataFrame:
    """ret_1 = log(close / close.shift(1))."""
    c = as_float_frame(close)
    return safe_log_ratio(c, c.shift(1))


def calc_high_open_ret(high: pd.DataFrame, open_: pd.DataFrame) -> pd.DataFrame:
    """high_open_ret = log(high / open)."""
    return safe_log_ratio(*align_frames(high, open_))


def calc_overnight_ret(open_: pd.DataFrame, pre_close: pd.DataFrame) -> pd.DataFrame:
    """overnight_ret = log(open / pre_close)."""
    return safe_log_ratio(*align_frames(open_, pre_close))


def calc_intraday_ret(close: pd.DataFrame, open_: pd.DataFrame) -> pd.DataFrame:
    """intraday_ret = log(close / open)."""
    return safe_log_ratio(*align_frames(close, open_))


def calc_turnover_log(turnover_rate_f: pd.DataFrame) -> pd.DataFrame:
    """log1p(turnover_rate_f), with negative values set to 0 first."""
    x = as_float_frame(turnover_rate_f).clip(lower=0.0)
    return clean_inf(np.log1p(x))


def calc_main_net_buy_ratio(
    net_mf_amount: pd.DataFrame,
    amount: pd.DataFrame,
    amount_unit_scale: float = 1000.0,
    eps: float = EPS,
) -> pd.DataFrame:
    """net_mf_amount / max(amount * amount_unit_scale, eps), clipped to [-1, 1]."""
    net, amt = align_frames(net_mf_amount, amount)
    out = safe_divide(net, amt * float(amount_unit_scale), eps=eps)
    return clip_frame(out, -1.0, 1.0)


def calc_large_order_imbalance(
    buy_lg_amount: pd.DataFrame,
    sell_lg_amount: pd.DataFrame,
    eps: float = EPS,
) -> pd.DataFrame:
    """(buy_lg_amount - sell_lg_amount) / max(buy_lg_amount + sell_lg_amount, eps)."""
    buy, sell = align_frames(buy_lg_amount, sell_lg_amount)
    out = safe_divide(buy - sell, buy + sell, eps=eps)
    return clip_frame(out, -1.0, 1.0)


def calc_elg_participation(
    buy_elg_amount: pd.DataFrame,
    sell_elg_amount: pd.DataFrame,
    amount: pd.DataFrame,
    amount_unit_scale: float = 1000.0,
    eps: float = EPS,
) -> pd.DataFrame:
    """(buy_elg_amount + sell_elg_amount) / max(amount * amount_unit_scale, eps), clipped to [0, 1]."""
    buy, sell, amt = align_frames(buy_elg_amount, sell_elg_amount, amount)
    out = safe_divide(buy + sell, amt * float(amount_unit_scale), eps=eps)
    return clip_frame(out, 0.0, 1.0)


def calc_volume_log_change(vol: pd.DataFrame) -> pd.DataFrame:
    """log1p(vol) - log1p(vol.shift(1))."""
    v = as_float_frame(vol).clip(lower=0.0)
    lv = np.log1p(v)
    return clean_inf(lv - lv.shift(1))


def calc_upper_shadow_ratio(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    eps: float = EPS,
    clip: bool = True,
) -> pd.DataFrame:
    """(high - max(open, close)) / max(high - low, eps)."""
    o, h, l, c = align_frames(open_, high, low, close)
    oc_max = o.where(o >= c, c)
    out = safe_divide(h - oc_max, h - l, eps=eps)
    return clip_frame(out, 0.0, 1.0) if clip else out


def calc_lower_shadow_ratio(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    eps: float = EPS,
    clip: bool = True,
) -> pd.DataFrame:
    """(min(open, close) - low) / max(high - low, eps)."""
    o, h, l, c = align_frames(open_, high, low, close)
    oc_min = o.where(o <= c, c)
    out = safe_divide(oc_min - l, h - l, eps=eps)
    return clip_frame(out, 0.0, 1.0) if clip else out


def calc_body_ratio(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    eps: float = EPS,
    clip: bool = True,
) -> pd.DataFrame:
    """abs(close - open) / max(high - low, eps)."""
    o, h, l, c = align_frames(open_, high, low, close)
    out = safe_divide((c - o).abs(), h - l, eps=eps)
    return clip_frame(out, 0.0, 1.0) if clip else out


def calc_candle_factors(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    eps: float = EPS,
) -> dict[str, pd.DataFrame]:
    """Compute upper_shadow, lower_shadow, and body_ratio together."""
    return {
        "upper_shadow": calc_upper_shadow_ratio(open_, high, low, close, eps=eps),
        "lower_shadow": calc_lower_shadow_ratio(open_, high, low, close, eps=eps),
        "body_ratio": calc_body_ratio(open_, high, low, close, eps=eps),
    }


def calc_yang_zhang_volatility(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    pre_close: pd.DataFrame,
    window: int = DEFAULT_YZ_WINDOW,
    annualization: int = DEFAULT_ANNUALIZATION,
    min_periods: Optional[int] = None,
    min_periods_ratio: float = 0.7,
    log_transform: bool = True,
) -> pd.DataFrame:
    """
    Yang-Zhang volatility.

    Default output is log1p(yz_vol * sqrt(annualization)).
    """
    o, h, l, c, pc = align_frames(open_, high, low, close, pre_close)
    mp = rolling_min_periods(window, min_periods, min_periods_ratio=min_periods_ratio)

    overnight = safe_log_ratio(o, pc)
    open_close = safe_log_ratio(c, o)

    log_ho = safe_log_ratio(h, o)
    log_lo = safe_log_ratio(l, o)
    log_hc = safe_log_ratio(h, c)
    log_lc = safe_log_ratio(l, c)
    rs = log_ho * log_hc + log_lo * log_lc

    overnight_var = overnight.rolling(window, min_periods=mp).var(ddof=1)
    open_close_var = open_close.rolling(window, min_periods=mp).var(ddof=1)
    rs_var = rs.rolling(window, min_periods=mp).mean()

    if window <= 1:
        k = 0.34
    else:
        k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))

    yz_var = overnight_var + k * open_close_var + (1.0 - k) * rs_var
    yz_var = yz_var.where(yz_var >= 0.0, 0.0)
    yz_vol = np.sqrt(yz_var)

    if not log_transform:
        return clean_inf(yz_vol)

    return clean_inf(np.log1p(yz_vol * np.sqrt(float(annualization))))


# ============================================================
# Rolling time-series factors
# ============================================================

def calc_ma_slope(
    close: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    min_periods_ratio: float = 0.7,
) -> pd.DataFrame:
    """ma_slope_N = log(MA_N / MA_N.shift(1))."""
    c = as_float_frame(close)
    mp = rolling_min_periods(window, min_periods, min_periods_ratio=min_periods_ratio)
    ma = c.rolling(window, min_periods=mp).mean()
    return safe_log_ratio(ma, ma.shift(1))


def calc_ma_bias(
    close: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    min_periods_ratio: float = 0.7,
) -> pd.DataFrame:
    """ma_bias_N = log(close / MA_N)."""
    c = as_float_frame(close)
    mp = rolling_min_periods(window, min_periods, min_periods_ratio=min_periods_ratio)
    ma = c.rolling(window, min_periods=mp).mean()
    return safe_log_ratio(c, ma)


def calc_rolling_volatility(
    ret_1: pd.DataFrame,
    window: int,
    annualization: int = DEFAULT_ANNUALIZATION,
    min_periods: Optional[int] = None,
    min_periods_ratio: float = 0.7,
    log_transform: bool = True,
) -> pd.DataFrame:
    """rolling_std(ret_1, N), default transformed as log1p(vol * sqrt(annualization))."""
    r = as_float_frame(ret_1)
    mp = rolling_min_periods(window, min_periods, min_periods_ratio=min_periods_ratio)
    vol = r.rolling(window, min_periods=mp).std(ddof=1)
    if not log_transform:
        return clean_inf(vol)
    return clean_inf(np.log1p(vol * np.sqrt(float(annualization))))


def calc_ret_volume_corr(
    ret_1: pd.DataFrame,
    vol_log_change: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    min_periods_ratio: float = 0.7,
) -> pd.DataFrame:
    """Rolling correlation between ret_1 and vol_log_change."""
    r, v = align_frames(ret_1, vol_log_change)
    mp = rolling_min_periods(window, min_periods, min_periods_ratio=min_periods_ratio)
    return clean_inf(r.rolling(window, min_periods=mp).corr(v)).clip(-1.0, 1.0)


def calc_rolling_max_ret(
    ret_1: pd.DataFrame,
    window: int,
    clip_bounds: tuple[float, float] = (-0.3, 0.3),
    min_periods: Optional[int] = None,
    min_periods_ratio: float = 0.7,
) -> pd.DataFrame:
    """rolling_max(ret_1, N), clipped to [-0.3, 0.3] by default."""
    r = as_float_frame(ret_1)
    mp = rolling_min_periods(window, min_periods, min_periods_ratio=min_periods_ratio)
    return clip_frame(r.rolling(window, min_periods=mp).max(), clip_bounds[0], clip_bounds[1])


def calc_rolling_min_ret(
    ret_1: pd.DataFrame,
    window: int,
    clip_bounds: tuple[float, float] = (-0.3, 0.3),
    min_periods: Optional[int] = None,
    min_periods_ratio: float = 0.7,
) -> pd.DataFrame:
    """rolling_min(ret_1, N), clipped to [-0.3, 0.3] by default."""
    r = as_float_frame(ret_1)
    mp = rolling_min_periods(window, min_periods, min_periods_ratio=min_periods_ratio)
    return clip_frame(r.rolling(window, min_periods=mp).min(), clip_bounds[0], clip_bounds[1])


def calc_rolling_sharpe(
    ret_1: pd.DataFrame,
    window: int,
    eps: float = EPS,
    clip_bounds: tuple[float, float] = (-5.0, 5.0),
    min_periods: Optional[int] = None,
    min_periods_ratio: float = 0.7,
) -> pd.DataFrame:
    """rolling_mean(ret_1, N) / max(rolling_std(ret_1, N), eps), clipped to [-5, 5]."""
    r = as_float_frame(ret_1)
    mp = rolling_min_periods(window, min_periods, min_periods_ratio=min_periods_ratio)
    mean = r.rolling(window, min_periods=mp).mean()
    std = r.rolling(window, min_periods=mp).std(ddof=1)
    out = safe_divide(mean, std, eps=eps)
    return clip_frame(out, clip_bounds[0], clip_bounds[1])


def calc_range_position(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    window: int,
    eps: float = EPS,
    min_periods: Optional[int] = None,
    min_periods_ratio: float = 0.7,
    clip: bool = True,
) -> pd.DataFrame:
    """(close - rolling_low_N) / max(rolling_high_N - rolling_low_N, eps)."""
    c, h, l = align_frames(close, high, low)
    mp = rolling_min_periods(window, min_periods, min_periods_ratio=min_periods_ratio)
    rolling_high = h.rolling(window, min_periods=mp).max()
    rolling_low = l.rolling(window, min_periods=mp).min()
    out = safe_divide(c - rolling_low, rolling_high - rolling_low, eps=eps)
    return clip_frame(out, 0.0, 1.0) if clip else out


def calc_night_day_spread(
    overnight_ret: pd.DataFrame,
    intraday_ret: pd.DataFrame,
    window: int,
    clip_bounds: tuple[float, float] = (-0.5, 0.5),
    min_periods: Optional[int] = None,
    min_periods_ratio: float = 0.7,
) -> pd.DataFrame:
    """rolling_sum(overnight_ret, N) - rolling_sum(intraday_ret, N)."""
    night, day = align_frames(overnight_ret, intraday_ret)
    mp = rolling_min_periods(window, min_periods, min_periods_ratio=min_periods_ratio)
    out = night.rolling(window, min_periods=mp).sum() - day.rolling(window, min_periods=mp).sum()
    return clip_frame(out, clip_bounds[0], clip_bounds[1])


def calc_log_rvol(
    vol: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    min_periods_ratio: float = 0.7,
) -> pd.DataFrame:
    """log((vol + 1) / (rolling_mean(vol.shift(1), N) + 1))."""
    v = as_float_frame(vol).clip(lower=0.0)
    mp = rolling_min_periods(window, min_periods, min_periods_ratio=min_periods_ratio)
    base = v.shift(1).rolling(window, min_periods=mp).mean()
    return clean_inf(np.log((v + 1.0) / (base + 1.0)))


def calc_price_efficiency(
    close: pd.DataFrame,
    ret_1: pd.DataFrame,
    window: int,
    eps: float = EPS,
    min_periods: Optional[int] = None,
    min_periods_ratio: float = 0.7,
) -> pd.DataFrame:
    """abs(log(close_t / close_{t-N})) / max(rolling_sum(abs(ret_1), N), eps), clipped to [0, 1]."""
    c, r = align_frames(close, ret_1)
    mp = rolling_min_periods(window, min_periods, min_periods_ratio=min_periods_ratio)
    straight = safe_log_ratio(c, c.shift(window)).abs()
    path = r.abs().rolling(window, min_periods=mp).sum()
    out = safe_divide(straight, path, eps=eps)
    return clip_frame(out, 0.0, 1.0)


def calc_boll_z(
    close: pd.DataFrame,
    window: int,
    eps: float = EPS,
    clip_bounds: tuple[float, float] = (-5.0, 5.0),
    min_periods: Optional[int] = None,
    min_periods_ratio: float = 0.7,
) -> pd.DataFrame:
    """(close - MA_N) / max(rolling_std(close, N), eps), clipped to [-5, 5]."""
    c = as_float_frame(close)
    mp = rolling_min_periods(window, min_periods, min_periods_ratio=min_periods_ratio)
    ma = c.rolling(window, min_periods=mp).mean()
    std = c.rolling(window, min_periods=mp).std(ddof=1)
    out = safe_divide(c - ma, std, eps=eps)
    return clip_frame(out, clip_bounds[0], clip_bounds[1])


# ============================================================
# Scalar factors
# ============================================================

def calc_free_float_ratio(
    free_share: pd.DataFrame,
    total_share: pd.DataFrame,
    eps: float = EPS,
) -> pd.DataFrame:
    """free_share / total_share, clipped to [0, 1]."""
    free, total = align_frames(free_share, total_share)
    return clip_frame(safe_divide(free, total, eps=eps), 0.0, 1.0)


def calc_log_total_mv(total_mv: pd.DataFrame) -> pd.DataFrame:
    """log1p(total_mv), with negative values treated as NaN."""
    x = as_float_frame(total_mv)
    x = x.where(x >= 0.0)
    return clean_inf(np.log1p(x))


def calc_log_circ_mv(circ_mv: pd.DataFrame) -> pd.DataFrame:
    """log1p(circ_mv), with negative values treated as NaN."""
    x = as_float_frame(circ_mv)
    x = x.where(x >= 0.0)
    return clean_inf(np.log1p(x))


def calc_log_pb(
    pb: pd.DataFrame,
    winsorize: bool = True,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """log1p(pb), optionally winsorized by date."""
    x = as_float_frame(pb)
    x = x.where(x >= 0.0)
    out = clean_inf(np.log1p(x))
    return winsorize_cross_section(out, lower_q=lower_q, upper_q=upper_q) if winsorize else out


def calc_log_ps_ttm(
    ps_ttm: pd.DataFrame,
    winsorize: bool = True,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """log1p(ps_ttm), optionally winsorized by date."""
    x = as_float_frame(ps_ttm)
    x = x.where(x >= 0.0)
    out = clean_inf(np.log1p(x))
    return winsorize_cross_section(out, lower_q=lower_q, upper_q=upper_q) if winsorize else out


def calc_ep_factors(pe_ttm: pd.DataFrame, eps: float = EPS) -> dict[str, pd.DataFrame]:
    """
    EP factor triplet:
    - ep_positive: 1 / pe_ttm when pe_ttm > 0, else 0
    - ep_negative: 1 / pe_ttm when pe_ttm < 0, else 0
    - ep_is_loss: 1 if pe_ttm < 0, 0 if pe_ttm >= 0, NaN if missing
    """
    pe = as_float_frame(pe_ttm)
    pe_safe = pe.mask(pe.abs() < eps, np.nan)
    ep = clean_inf(1.0 / pe_safe)

    ep_positive = ep.where(pe > 0.0, 0.0)
    ep_negative = ep.where(pe < 0.0, 0.0)

    ep_is_loss = pd.DataFrame(np.nan, index=pe.index, columns=pe.columns, dtype=float)
    ep_is_loss[pe.notna()] = (pe[pe.notna()] < 0.0).astype(float)

    return {
        "ep_positive": clean_inf(ep_positive),
        "ep_negative": clean_inf(ep_negative),
        "ep_is_loss": ep_is_loss,
    }


# ============================================================
# Batch builders
# ============================================================

def calc_base_time_series_factors(
    market: Mapping[str, pd.DataFrame],
    basic: Optional[Mapping[str, pd.DataFrame]] = None,
    moneyflow: Optional[Mapping[str, pd.DataFrame]] = None,
    amount_unit_scale: float = 1000.0,
    eps: float = EPS,
    include_yang_zhang: bool = True,
    yz_window: int = DEFAULT_YZ_WINDOW,
    annualization: int = DEFAULT_ANNUALIZATION,
    min_periods_ratio: float = 0.7,
) -> dict[str, pd.DataFrame]:
    """
    Batch compute pure time-series factors.

    Required market keys:
        open, high, low, close, pre_close, vol, amount

    Optional basic key:
        turnover_rate_f

    Optional moneyflow keys:
        net_mf_amount, buy_lg_amount, sell_lg_amount,
        buy_elg_amount, sell_elg_amount
    """
    require_keys(market, ["open", "high", "low", "close", "pre_close", "vol", "amount"], group_name="market")

    open_ = market["open"]
    high = market["high"]
    low = market["low"]
    close = market["close"]
    pre_close = market["pre_close"]
    vol = market["vol"]
    amount = market["amount"]

    factors: dict[str, pd.DataFrame] = {
        "ret_1": calc_log_return(close),
        "high_open_ret": calc_high_open_ret(high, open_),
        "overnight_ret": calc_overnight_ret(open_, pre_close),
        "intraday_ret": calc_intraday_ret(close, open_),
        "vol_log_change": calc_volume_log_change(vol),
    }

    factors.update(calc_candle_factors(open_, high, low, close, eps=eps))

    if basic is not None and "turnover_rate_f" in basic:
        factors["turnover_log"] = calc_turnover_log(basic["turnover_rate_f"])

    if moneyflow is not None:
        if "net_mf_amount" in moneyflow:
            factors["main_net_buy_ratio"] = calc_main_net_buy_ratio(
                moneyflow["net_mf_amount"], amount,
                amount_unit_scale=amount_unit_scale, eps=eps,
            )
        if "buy_lg_amount" in moneyflow and "sell_lg_amount" in moneyflow:
            factors["large_order_imbalance"] = calc_large_order_imbalance(
                moneyflow["buy_lg_amount"], moneyflow["sell_lg_amount"], eps=eps,
            )
        if "buy_elg_amount" in moneyflow and "sell_elg_amount" in moneyflow:
            factors["elg_participation"] = calc_elg_participation(
                moneyflow["buy_elg_amount"], moneyflow["sell_elg_amount"], amount,
                amount_unit_scale=amount_unit_scale, eps=eps,
            )

    if include_yang_zhang:
        factors[f"yang_zhang_vol_{yz_window}"] = calc_yang_zhang_volatility(
            open_, high, low, close, pre_close,
            window=yz_window,
            annualization=annualization,
            min_periods_ratio=min_periods_ratio,
            log_transform=True,
        )

    return factors


def calc_window_time_series_factors(
    market: Mapping[str, pd.DataFrame],
    base_factors: Optional[Mapping[str, pd.DataFrame]] = None,
    windows: Iterable[int] = DEFAULT_WINDOWS,
    night_day_extra_windows: Iterable[int] = (1,),
    annualization: int = DEFAULT_ANNUALIZATION,
    min_periods_ratio: float = 0.7,
    eps: float = EPS,
) -> dict[str, pd.DataFrame]:
    """
    Batch compute rolling time-series factors.

    Required market keys:
        open, high, low, close, pre_close, vol
    """
    require_keys(market, ["open", "high", "low", "close", "pre_close", "vol"], group_name="market")

    close = market["close"]
    high = market["high"]
    low = market["low"]
    open_ = market["open"]
    pre_close = market["pre_close"]
    vol = market["vol"]

    ret_1 = base_factors["ret_1"] if base_factors is not None and "ret_1" in base_factors else calc_log_return(close)
    vol_log_change = (
        base_factors["vol_log_change"]
        if base_factors is not None and "vol_log_change" in base_factors
        else calc_volume_log_change(vol)
    )
    overnight_ret = (
        base_factors["overnight_ret"]
        if base_factors is not None and "overnight_ret" in base_factors
        else calc_overnight_ret(open_, pre_close)
    )
    intraday_ret = (
        base_factors["intraday_ret"]
        if base_factors is not None and "intraday_ret" in base_factors
        else calc_intraday_ret(close, open_)
    )

    factors: dict[str, pd.DataFrame] = {}
    windows = tuple(int(w) for w in windows)

    for w in windows:
        factors[f"ma_slope_{w}"] = calc_ma_slope(close, w, min_periods_ratio=min_periods_ratio)
        factors[f"ma_bias_{w}"] = calc_ma_bias(close, w, min_periods_ratio=min_periods_ratio)
        factors[f"rolling_vol_{w}"] = calc_rolling_volatility(
            ret_1, w, annualization=annualization,
            min_periods_ratio=min_periods_ratio, log_transform=True,
        )
        factors[f"ret_vol_corr_{w}"] = calc_ret_volume_corr(
            ret_1, vol_log_change, w, min_periods_ratio=min_periods_ratio,
        )
        factors[f"rolling_max_ret_{w}"] = calc_rolling_max_ret(
            ret_1, w, min_periods_ratio=min_periods_ratio,
        )
        factors[f"rolling_min_ret_{w}"] = calc_rolling_min_ret(
            ret_1, w, min_periods_ratio=min_periods_ratio,
        )
        factors[f"rolling_sharpe_{w}"] = calc_rolling_sharpe(
            ret_1, w, eps=eps, min_periods_ratio=min_periods_ratio,
        )
        factors[f"range_position_{w}"] = calc_range_position(
            close, high, low, w, eps=eps, min_periods_ratio=min_periods_ratio,
        )
        factors[f"log_rvol_{w}"] = calc_log_rvol(
            vol, w, min_periods_ratio=min_periods_ratio,
        )
        factors[f"price_efficiency_{w}"] = calc_price_efficiency(
            close, ret_1, w, eps=eps, min_periods_ratio=min_periods_ratio,
        )
        factors[f"boll_z_{w}"] = calc_boll_z(
            close, w, eps=eps, min_periods_ratio=min_periods_ratio,
        )

    night_windows = tuple(dict.fromkeys([int(w) for w in tuple(night_day_extra_windows) + windows]))
    for w in night_windows:
        factors[f"night_day_spread_{w}"] = calc_night_day_spread(
            overnight_ret, intraday_ret, w, min_periods_ratio=min_periods_ratio,
        )

    return factors


def calc_scalar_factors(
    basic: Mapping[str, pd.DataFrame],
    eps: float = EPS,
    winsorize_pb_ps: bool = True,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> dict[str, pd.DataFrame]:
    """Batch compute scalar factors from daily_basic-style matrices."""
    factors: dict[str, pd.DataFrame] = {}

    if "free_share" in basic and "total_share" in basic:
        factors["free_float_ratio"] = calc_free_float_ratio(basic["free_share"], basic["total_share"], eps=eps)
    if "total_mv" in basic:
        factors["log_total_mv"] = calc_log_total_mv(basic["total_mv"])
    if "circ_mv" in basic:
        factors["log_circ_mv"] = calc_log_circ_mv(basic["circ_mv"])
    if "pb" in basic:
        factors["log_pb"] = calc_log_pb(basic["pb"], winsorize=winsorize_pb_ps, lower_q=lower_q, upper_q=upper_q)
    if "ps_ttm" in basic:
        factors["log_ps_ttm"] = calc_log_ps_ttm(basic["ps_ttm"], winsorize=winsorize_pb_ps, lower_q=lower_q, upper_q=upper_q)
    if "pe_ttm" in basic:
        factors.update(calc_ep_factors(basic["pe_ttm"], eps=eps))

    return factors


def build_all_factors(
    market: Mapping[str, pd.DataFrame],
    basic: Optional[Mapping[str, pd.DataFrame]] = None,
    moneyflow: Optional[Mapping[str, pd.DataFrame]] = None,
    config: FactorConfig = FactorConfig(),
    include_base: bool = True,
    include_window: bool = True,
    include_scalar: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build all stage-1 factors."""
    factors: dict[str, pd.DataFrame] = {}

    base_factors: dict[str, pd.DataFrame] = {}
    if include_base:
        print("[进度 1/3] 计算基础量价因子...")
        base_factors = calc_base_time_series_factors(
            market=market,
            basic=basic,
            moneyflow=moneyflow,
            amount_unit_scale=config.amount_unit_scale,
            eps=config.eps,
            include_yang_zhang=True,
            yz_window=config.yz_window,
            annualization=config.annualization,
            min_periods_ratio=config.min_periods_ratio,
        )
        factors.update(base_factors)
        print("[进度 1/3] ✅ 基础因子计算完成")

    if include_window:
        print("[进度 2/3] 计算滚动窗口因子（耗时较长）...")
        factors.update(calc_window_time_series_factors(
            market=market,
            base_factors=base_factors,
            windows=config.windows,
            night_day_extra_windows=(1,),
            annualization=config.annualization,
            min_periods_ratio=config.min_periods_ratio,
            eps=config.eps,
        ))
        print("[进度 2/3] ✅ 滚动因子计算完成")

    if include_scalar and basic is not None:
        print("[进度 3/3] 计算估值&市值因子...")
        factors.update(calc_scalar_factors(basic=basic, eps=config.eps, winsorize_pb_ps=True))
        print("[进度 3/3] ✅ 估值因子计算完成")

    return factors


# ============================================================
# Output utilities
# ============================================================

def _stack_keep_na(df: pd.DataFrame) -> pd.Series:
    """
    Stack a wide DataFrame while preserving NaNs.

    pandas >= 2.1 recommends future_stack=True; older versions do not support it.
    """
    x = as_float_frame(df)
    try:
        return x.stack(future_stack=True)
    except TypeError:
        return x.stack(dropna=False)


def stack_factor_dict(
    factors: Mapping[str, pd.DataFrame],
    *,
    drop_all_nan_rows: bool = False,
    sort_index: bool = True,
) -> pd.DataFrame:
    """
    Convert a factor dictionary into a MultiIndex panel.

    Output index: [trade_date, ts_code]
    Output columns: factor names
    """
    series_list = []
    for name, df in factors.items():
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


def unstack_factor_panel(panel: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    """Recover one wide factor matrix from a stacked panel."""
    if factor_name not in panel.columns:
        raise KeyError(f"factor_name not in panel columns: {factor_name}")
    if not isinstance(panel.index, pd.MultiIndex):
        raise TypeError("panel.index must be MultiIndex [trade_date, ts_code]")
    return panel[factor_name].unstack("ts_code")


def list_expected_factor_names(
    windows: Iterable[int] = DEFAULT_WINDOWS,
    yz_window: int = DEFAULT_YZ_WINDOW,
) -> list[str]:
    """List expected factor names under the default stage-1 specification."""
    names = [
        "ret_1",
        "high_open_ret",
        "overnight_ret",
        "intraday_ret",
        "vol_log_change",
        "upper_shadow",
        "lower_shadow",
        "body_ratio",
        "turnover_log",
        "main_net_buy_ratio",
        "large_order_imbalance",
        "elg_participation",
        f"yang_zhang_vol_{yz_window}",
    ]

    windows = tuple(int(w) for w in windows)
    for w in windows:
        names.extend([
            f"ma_slope_{w}",
            f"ma_bias_{w}",
            f"rolling_vol_{w}",
            f"ret_vol_corr_{w}",
            f"rolling_max_ret_{w}",
            f"rolling_min_ret_{w}",
            f"rolling_sharpe_{w}",
            f"range_position_{w}",
            f"log_rvol_{w}",
            f"price_efficiency_{w}",
            f"boll_z_{w}",
        ])

    for w in tuple(dict.fromkeys([1] + list(windows))):
        names.append(f"night_day_spread_{w}")

    names.extend([
        "free_float_ratio",
        "log_total_mv",
        "log_circ_mv",
        "log_pb",
        "log_ps_ttm",
        "ep_positive",
        "ep_negative",
        "ep_is_loss",
    ])

    return names


__all__ = [
    "EPS",
    "DEFAULT_WINDOWS",
    "DEFAULT_YZ_WINDOW",
    "DEFAULT_ANNUALIZATION",
    "FactorConfig",
    "as_float_frame",
    "align_frames",
    "clean_inf",
    "safe_divide",
    "safe_log_ratio",
    "clip_frame",
    "rolling_min_periods",
    "winsorize_cross_section",
    "zscore_cross_section",
    "rank_pct_cross_section",
    "calc_log_return",
    "calc_high_open_ret",
    "calc_overnight_ret",
    "calc_intraday_ret",
    "calc_turnover_log",
    "calc_main_net_buy_ratio",
    "calc_large_order_imbalance",
    "calc_elg_participation",
    "calc_volume_log_change",
    "calc_upper_shadow_ratio",
    "calc_lower_shadow_ratio",
    "calc_body_ratio",
    "calc_candle_factors",
    "calc_yang_zhang_volatility",
    "calc_ma_slope",
    "calc_ma_bias",
    "calc_rolling_volatility",
    "calc_ret_volume_corr",
    "calc_rolling_max_ret",
    "calc_rolling_min_ret",
    "calc_rolling_sharpe",
    "calc_range_position",
    "calc_night_day_spread",
    "calc_log_rvol",
    "calc_price_efficiency",
    "calc_boll_z",
    "calc_free_float_ratio",
    "calc_log_total_mv",
    "calc_log_circ_mv",
    "calc_log_pb",
    "calc_log_ps_ttm",
    "calc_ep_factors",
    "calc_base_time_series_factors",
    "calc_window_time_series_factors",
    "calc_scalar_factors",
    "build_all_factors",
    "stack_factor_dict",
    "unstack_factor_panel",
    "list_expected_factor_names",
]
