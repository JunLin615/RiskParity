
from __future__ import annotations

"""
tsmom_backtest.py

纯函数版 TSMOM 成交级回测函数库。

依赖：
- pandas
- numpy
- backtest.py
- tsmom_utils_v2.py

设计目标：
1. 与 IRP / RP 使用同一套成交级回测框架，严格控制变量；
2. 使用 shares / cash / pending_signal 的离散交易状态机；
3. 严格整手成交、次日按指定 execution_price_type 成交；
4. 区分估值价格（有限前向填充）与交易价格（不填充，遇停牌顺延）；
5. 固定权重基准不复用策略过滤器，避免过滤器污染基准。
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from backtest import (
    build_field_matrix,
    calc_actual_weights,
    calc_asset_correlation_matrix,
    calc_avg_pairwise_correlation,
    calc_portfolio_value,
    calc_turnover_from_weights,
    ensure_same_index_columns,
    get_execution_price_matrix,
    get_rebalance_dates,
    next_trading_date,
    normalize_weights as normalize_series_weights,
    performance_summary,
    rebalance_to_target_weights,
    should_rebalance,
)
from backtest_utils import summarize_weight_statistics
from risk_parity import ensure_datetime_index
from tsmom_utils import (
    build_tsmom_positions,
    prepare_price_matrix,
)


@dataclass(frozen=True)
class TSMOMSignalSnapshot:
    """某个 signal_date 上的 TSMOM 截面快照。"""

    signal_date: pd.Timestamp
    raw_signal: pd.Series
    signal: pd.Series
    volatility: pd.Series
    target_weight: pd.Series


# ============================================================
# 基础工具
# ============================================================


def _coerce_int_sequence(value: int | Sequence[int], name: str) -> list[int]:
    if np.isscalar(value):
        out = [int(value)]
    else:
        out = [int(x) for x in value]

    if len(out) == 0:
        raise ValueError(f"{name} cannot be empty")
    return out


def _broadcast_param_sequence(
    value: int | Sequence[int],
    base: Sequence[int],
    name: str,
) -> list[int]:
    if np.isscalar(value):
        out = [int(value)] * len(base)
    else:
        out = [int(x) for x in value]
        if len(out) != len(base):
            raise ValueError(f"{name} length must match lookback length")
    return out



def _normalize_side(side: str) -> str:
    x = str(side).strip().lower()
    mapping = {
        "both": "long_short",
        "long_short": "long_short",
        "ls": "long_short",
        "long_only": "long_only",
        "long": "long_only",
        "short_only": "short_only",
        "short": "short_only",
    }
    if x not in mapping:
        raise ValueError("side must be 'long_short', 'long_only', or 'short_only'")
    return mapping[x]


def calc_ma_filter_mask(
    price_df: pd.DataFrame,
    ma_window: int = 120,
    side: str = "long_short",
) -> pd.DataFrame:
    """
    计算 MA 过滤掩码。

    - long_short: 返回 {1, -1, 0} 状态矩阵
    - long_only : 返回 {1, 0}
    - short_only: 返回 {1, 0}
    """
    if ma_window <= 0:
        raise ValueError("ma_window must be positive")

    px = ensure_datetime_index(price_df).astype(float)
    ma = px.rolling(window=ma_window, min_periods=ma_window).mean()
    side = _normalize_side(side)

    if side == "long_short":
        mask = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        mask = mask.mask(np.isfinite(px) & np.isfinite(ma) & (px >= ma), 1.0)
        mask = mask.mask(np.isfinite(px) & np.isfinite(ma) & (px < ma), -1.0)
        return mask

    if side == "long_only":
        return (np.isfinite(px) & np.isfinite(ma) & (px >= ma)).astype(float)

    return (np.isfinite(px) & np.isfinite(ma) & (px < ma)).astype(float)


def apply_side_rules(
    position_df: pd.DataFrame,
    side: str = "long_only",
) -> pd.DataFrame:
    pos = ensure_datetime_index(position_df).astype(float)
    side = _normalize_side(side)

    if side == "long_short":
        return pos
    if side == "long_only":
        return pos.clip(lower=0.0)
    return pos.clip(upper=0.0)


def apply_signal_threshold(
    position_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    threshold: float = 0.0,
) -> pd.DataFrame:
    pos = ensure_datetime_index(position_df).astype(float)
    sig = ensure_datetime_index(signal_df).astype(float)
    sig = sig.reindex(index=pos.index, columns=pos.columns)

    if threshold <= 0:
        return pos

    mask = sig.abs() >= float(threshold)
    return pos.where(mask, 0.0)


def apply_eligibility_mask(
    position_df: pd.DataFrame,
    eligibility_mask: Optional[pd.DataFrame | pd.Series] = None,
) -> pd.DataFrame:
    pos = ensure_datetime_index(position_df).astype(float)
    if eligibility_mask is None:
        return pos

    if isinstance(eligibility_mask, pd.Series):
        col_mask = eligibility_mask.reindex(pos.columns).fillna(False).astype(bool)
        return pos.where(col_mask, 0.0)

    mask = ensure_datetime_index(eligibility_mask).reindex(index=pos.index, columns=pos.columns)
    mask = mask.fillna(False).astype(bool)
    return pos.where(mask, 0.0)


def apply_ma_filter_to_positions(
    position_df: pd.DataFrame,
    price_df: pd.DataFrame,
    ma_filter_window: Optional[int] = None,
    side: str = "long_only",
) -> pd.DataFrame:
    pos = ensure_datetime_index(position_df).astype(float)
    if ma_filter_window is None:
        return pos

    px = ensure_datetime_index(price_df).astype(float).reindex(index=pos.index, columns=pos.columns)
    ma_mask = calc_ma_filter_mask(px, ma_window=ma_filter_window, side=side)
    side = _normalize_side(side)

    if side == "long_short":
        out = pos.copy()
        long_ok = ma_mask == 1.0
        short_ok = ma_mask == -1.0
        return out.where(((out >= 0.0) & long_ok) | ((out <= 0.0) & short_ok), 0.0)

    return pos.where(ma_mask.astype(bool), 0.0)


def normalize_weight_matrix_to_target_gross(
    weight_df: pd.DataFrame,
    target_gross: float = 1.0,
) -> pd.DataFrame:
    w = ensure_datetime_index(weight_df).astype(float)
    gross = w.abs().sum(axis=1)
    scale = pd.Series(0.0, index=w.index, dtype=float)
    valid = gross > 0
    scale.loc[valid] = float(target_gross) / gross.loc[valid]
    return w.mul(scale, axis=0)


def _convert_to_tradeable_long_only_weights(
    weight_like_df: pd.DataFrame,
    normalize_trade_weights: bool = True,
) -> pd.DataFrame:
    """
    将信号/理论仓位转换为成交引擎可执行的 long-only 权重：
    1. 剪去所有负权重；
    2. 若 normalize_trade_weights=True，则把正权重归一化到和为 1；
       若某日全为 0，则保留现金。
    """
    w = ensure_datetime_index(weight_like_df).astype(float).clip(lower=0.0)
    if not normalize_trade_weights:
        return w

    out = []
    for dt, row in w.iterrows():
        out.append(normalize_series_weights(row).rename(dt))
    return pd.DataFrame(out)


def summarize_exposure(weight_df: pd.DataFrame) -> pd.DataFrame:
    w = ensure_datetime_index(weight_df).fillna(0.0)
    gross = w.abs().sum(axis=1)
    net = w.sum(axis=1)
    long_exposure = w.clip(lower=0.0).sum(axis=1)
    short_exposure = (-w.clip(upper=0.0)).sum(axis=1)
    active_count = (w != 0).sum(axis=1)

    rows = {
        "start": w.index.min(),
        "end": w.index.max(),
        "n_obs": len(w),
        "avg_gross": float(gross.mean()) if len(gross) > 0 else np.nan,
        "avg_net": float(net.mean()) if len(net) > 0 else np.nan,
        "avg_long_exposure": float(long_exposure.mean()) if len(long_exposure) > 0 else np.nan,
        "avg_short_exposure": float(short_exposure.mean()) if len(short_exposure) > 0 else np.nan,
        "avg_active_assets": float(active_count.mean()) if len(active_count) > 0 else np.nan,
        "max_gross": float(gross.max()) if len(gross) > 0 else np.nan,
        "max_abs_net": float(net.abs().max()) if len(net) > 0 else np.nan,
    }
    return pd.DataFrame([rows])


def _minimum_history_rows(
    lookback: int | Sequence[int],
    skip_recent: int | Sequence[int],
    vol_method: str,
    vol_window: int,
    ma_filter_window: Optional[int],
) -> int:
    lookbacks = _coerce_int_sequence(lookback, name="lookback")
    skips = _broadcast_param_sequence(skip_recent, base=lookbacks, name="skip_recent")
    signal_need = max(lb + sk + 1 for lb, sk in zip(lookbacks, skips))
    vol_need = max(20, vol_window) if vol_method == "ewma" else vol_window
    return max(signal_need, vol_need + 1, int(ma_filter_window or 0) + 1)


def _build_processed_weight_matrix(
    prepared_price: pd.DataFrame,
    lookback: int | Sequence[int],
    skip_recent: int | Sequence[int],
    signal_type: str,
    use_excess_returns: bool,
    rf: Optional[pd.Series | float],
    annual_rf: Optional[float],
    return_type: str,
    vol_method: str,
    vol_window: int,
    ewma_lambda: float,
    annualization: int,
    target_vol: float,
    max_abs_position: Optional[float],
    side: str,
    signal_threshold: float,
    ma_filter_window: Optional[int],
    eligibility_mask: Optional[pd.DataFrame | pd.Series],
    normalize_to_gross: Optional[float],
    vol_floor: float,
    normalize_trade_weights: bool,
    combination_method: str,
    horizon_weights: Optional[Sequence[float]],
    signal_clip: Optional[float],
    zero_to_nan: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    outputs = build_tsmom_positions(
        price_df=prepared_price,
        lookback=lookback,
        skip_recent=skip_recent,
        signal_type=signal_type,
        use_excess_returns=use_excess_returns,
        rf=rf,
        annual_rf=annual_rf,
        return_type=return_type,
        vol_method=vol_method,
        vol_window=vol_window,
        ewma_lambda=ewma_lambda,
        annualization=annualization,
        target_vol=target_vol,
        max_abs_position=max_abs_position,
        execution_lag=0,
        normalize_to_gross=normalize_to_gross,
        vol_floor=vol_floor,
        combination_method=combination_method,
        horizon_weights=horizon_weights,
        signal_clip=signal_clip,
        zero_to_nan=zero_to_nan,
    )

    theoretical = ensure_datetime_index(outputs["final_position"]).astype(float)
    theoretical = apply_side_rules(theoretical, side=side)
    theoretical = apply_signal_threshold(theoretical, signal_df=outputs["signal"], threshold=signal_threshold)
    theoretical = apply_ma_filter_to_positions(
        theoretical,
        price_df=prepared_price,
        ma_filter_window=ma_filter_window,
        side=side,
    )
    theoretical = apply_eligibility_mask(theoretical, eligibility_mask=eligibility_mask)

    if normalize_to_gross is not None:
        theoretical = normalize_weight_matrix_to_target_gross(theoretical, target_gross=normalize_to_gross)

    tradeable = _convert_to_tradeable_long_only_weights(
        theoretical,
        normalize_trade_weights=normalize_trade_weights,
    )
    return outputs, tradeable


# ============================================================
# 点时权重计算
# ============================================================

def compute_tsmom_target_weights_on_date(
    close_price_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    price_prepare_kwargs: Optional[dict] = None,
    lookback: int | Sequence[int] = 252,
    skip_recent: int | Sequence[int] = 0,
    signal_type: str = "sign",
    use_excess_returns: bool = True,
    rf: Optional[pd.Series | float] = None,
    annual_rf: Optional[float] = None,
    return_type: str = "simple",
    vol_method: str = "ewma",
    vol_window: int = 63,
    ewma_lambda: float = 0.94,
    annualization: int = 252,
    target_vol: float = 0.40,
    max_abs_position: Optional[float] = None,
    side: str = "long_only",
    signal_threshold: float = 0.0,
    ma_filter_window: Optional[int] = None,
    eligibility_mask: Optional[pd.DataFrame | pd.Series] = None,
    normalize_to_gross: Optional[float] = 1.0,
    vol_floor: float = 1e-8,
    normalize_trade_weights: bool = True,
    combination_method: str = "mean",
    horizon_weights: Optional[Sequence[float]] = None,
    signal_clip: Optional[float] = 3.0,
    zero_to_nan: bool = False,
) -> pd.Series:
    """
    在 signal_date 当日收盘后，使用截至 signal_date 的历史价格计算“次日待执行”的目标权重。

    注意：
    - 这里返回的是成交引擎可执行的 long-only 目标权重；
    - 若 side='long_short'，会先按多空信号构造理论仓位，但最终执行层仍会把负权重裁剪为 0；
    - 若当日无可交易多头信号，则返回全 0（组合保持现金）。
    """
    price = ensure_datetime_index(close_price_df).astype(float)
    signal_date = pd.Timestamp(signal_date)
    hist_price = price.loc[:signal_date].copy()

    if hist_price.empty:
        return pd.Series(0.0, index=price.columns, name="weight")

    prepare_kwargs = dict(price_prepare_kwargs or {})
    prepared_price = prepare_price_matrix(hist_price, **prepare_kwargs)
    if prepared_price.empty:
        return pd.Series(0.0, index=price.columns, name="weight")

    _, tradeable = _build_processed_weight_matrix(
        prepared_price=prepared_price,
        lookback=lookback,
        skip_recent=skip_recent,
        signal_type=signal_type,
        use_excess_returns=use_excess_returns,
        rf=rf,
        annual_rf=annual_rf,
        return_type=return_type,
        vol_method=vol_method,
        vol_window=vol_window,
        ewma_lambda=ewma_lambda,
        annualization=annualization,
        target_vol=target_vol,
        max_abs_position=max_abs_position,
        side=side,
        signal_threshold=signal_threshold,
        ma_filter_window=ma_filter_window,
        eligibility_mask=eligibility_mask,
        normalize_to_gross=normalize_to_gross,
        vol_floor=vol_floor,
        normalize_trade_weights=normalize_trade_weights,
        combination_method=combination_method,
        horizon_weights=horizon_weights,
        signal_clip=signal_clip,
        zero_to_nan=zero_to_nan,
    )

    weight = tradeable.iloc[-1].reindex(price.columns).fillna(0.0)
    weight.name = "weight"
    return weight


compute_target_weights_on_date = compute_tsmom_target_weights_on_date


def make_tsmom_signal_snapshot(
    close_price_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    price_prepare_kwargs: Optional[dict] = None,
    lookback: int | Sequence[int] = 252,
    skip_recent: int | Sequence[int] = 0,
    signal_type: str = "sign",
    use_excess_returns: bool = True,
    rf: Optional[pd.Series | float] = None,
    annual_rf: Optional[float] = None,
    return_type: str = "simple",
    vol_method: str = "ewma",
    vol_window: int = 63,
    ewma_lambda: float = 0.94,
    annualization: int = 252,
    target_vol: float = 0.40,
    max_abs_position: Optional[float] = None,
    side: str = "long_only",
    signal_threshold: float = 0.0,
    ma_filter_window: Optional[int] = None,
    eligibility_mask: Optional[pd.DataFrame | pd.Series] = None,
    normalize_to_gross: Optional[float] = 1.0,
    vol_floor: float = 1e-8,
    normalize_trade_weights: bool = True,
    combination_method: str = "mean",
    horizon_weights: Optional[Sequence[float]] = None,
    signal_clip: Optional[float] = 3.0,
    zero_to_nan: bool = False,
) -> TSMOMSignalSnapshot:
    price = ensure_datetime_index(close_price_df).astype(float)
    signal_date = pd.Timestamp(signal_date)
    hist_price = price.loc[:signal_date].copy()
    prepare_kwargs = dict(price_prepare_kwargs or {})
    prepared_price = prepare_price_matrix(hist_price, **prepare_kwargs)

    outputs, tradeable = _build_processed_weight_matrix(
        prepared_price=prepared_price,
        lookback=lookback,
        skip_recent=skip_recent,
        signal_type=signal_type,
        use_excess_returns=use_excess_returns,
        rf=rf,
        annual_rf=annual_rf,
        return_type=return_type,
        vol_method=vol_method,
        vol_window=vol_window,
        ewma_lambda=ewma_lambda,
        annualization=annualization,
        target_vol=target_vol,
        max_abs_position=max_abs_position,
        side=side,
        signal_threshold=signal_threshold,
        ma_filter_window=ma_filter_window,
        eligibility_mask=eligibility_mask,
        normalize_to_gross=normalize_to_gross,
        vol_floor=vol_floor,
        normalize_trade_weights=normalize_trade_weights,
        combination_method=combination_method,
        horizon_weights=horizon_weights,
        signal_clip=signal_clip,
        zero_to_nan=zero_to_nan,
    )

    raw_signal = outputs["raw_signal"].iloc[-1].reindex(price.columns).astype(float).fillna(0.0)
    signal_last = outputs["signal"].iloc[-1].reindex(price.columns).astype(float).fillna(0.0)
    vol_last = outputs["vol"].iloc[-1].reindex(price.columns).astype(float).fillna(np.nan)
    weight_last = tradeable.iloc[-1].reindex(price.columns).astype(float).fillna(0.0)

    return TSMOMSignalSnapshot(
        signal_date=signal_date,
        raw_signal=raw_signal.rename("raw_signal"),
        signal=signal_last.rename("signal"),
        volatility=vol_last.rename("volatility"),
        target_weight=weight_last.rename("weight"),
    )


# ============================================================
# 主回测函数
# ============================================================

def simulate_tsmom_backtest(
    market: dict[str, pd.DataFrame],
    initial_cash: float = 1_000_000.0,
    lookback: int | Sequence[int] = 252,
    skip_recent: int | Sequence[int] = 0,
    signal_type: str = "sign",
    use_excess_returns: bool = True,
    rf: Optional[pd.Series | float] = None,
    annual_rf: Optional[float] = None,
    return_type: str = "simple",
    vol_method: str = "ewma",
    vol_window: int = 63,
    ewma_lambda: float = 0.94,
    annualization: int = 252,
    target_vol: float = 0.40,
    max_abs_position: Optional[float] = None,
    rebalance_freq: str = "M",
    execution_price_type: str = "avg",
    valuation_ffill_limit: int = 5,
    fee_rate_buy: float = 0.0005,
    fee_rate_sell: float = 0.0005,
    lot_size: int | dict[str, int] = 100,
    max_trade_amount_ratio: Optional[float] = 0.05,
    amount_unit_scale: float = 1000.0,
    use_drift_trigger: bool = False,
    drift_threshold: float = 0.05,
    price_prepare_kwargs: Optional[dict] = None,
    side: str = "long_only",
    signal_threshold: float = 0.0,
    ma_filter_window: Optional[int] = None,
    eligibility_mask: Optional[pd.DataFrame | pd.Series] = None,
    normalize_to_gross: Optional[float] = 1.0,
    vol_floor: float = 1e-8,
    normalize_trade_weights: bool = True,
    risk_free_rate: float = 0.0,
    combination_method: str = "mean",
    horizon_weights: Optional[Sequence[float]] = None,
    signal_clip: Optional[float] = 3.0,
    zero_to_nan: bool = False,
) -> dict[str, object]:
    """
    TSMOM 成交级回测主入口。

    支持单周期与多周期 TSMOM 信号：
    - lookback 可为 int 或 Sequence[int]
    - skip_recent 可为 int 或 Sequence[int]
    - 组合参数通过 combination_method / horizon_weights / signal_clip / zero_to_nan 控制

    与 IRP / RP 对齐的地方：
    - t 日收盘后生成目标权重；
    - t+1 按 execution_price_type 成交；
    - 严格整手；
    - 区分估值价与成交价；
    - 支持成交额占比约束、停牌顺延、偏离触发调仓。
    """
    side = _normalize_side(side)
    market = ensure_same_index_columns(market)
    close_px = ensure_datetime_index(market["close"]).astype(float)

    val_px = close_px.ffill(limit=valuation_ffill_limit)
    exec_px = get_execution_price_matrix(market, execution_price_type=execution_price_type)
    amount_px = market.get("amount", None)

    codes = close_px.columns
    dates = close_px.index
    scheduled_dates = get_rebalance_dates(dates, freq=rebalance_freq)
    min_hist_rows = _minimum_history_rows(
        lookback=lookback,
        skip_recent=skip_recent,
        vol_method=vol_method,
        vol_window=vol_window,
        ma_filter_window=ma_filter_window,
    )

    shares = pd.Series(0, index=codes, dtype=int)
    cash = float(initial_cash)
    pending_signal = None

    nav_records = []
    return_records = []
    position_records = []
    weight_records = []
    target_weight_records = []
    trade_records = []
    rebalance_logs = []

    prev_nav = initial_cash
    price_prepare_kwargs = dict(price_prepare_kwargs or {})

    for i, dt in enumerate(dates):
        val_today = val_px.loc[dt]
        exec_today = exec_px.loc[dt]
        amount_today = amount_px.loc[dt] if amount_px is not None else None

        # 1) 执行前一日信号
        if pending_signal is not None:
            target_weights = pending_signal["target_weights"].reindex(codes).fillna(0.0)
            signal_date = pending_signal["signal_date"]
            reason = pending_signal["reason"]
            drift_value = pending_signal["drift_value"]

            involved_assets = shares[shares > 0].index.union(target_weights[target_weights > 0].index)
            involved_exec_px = exec_today.reindex(involved_assets)
            invalid_px = ~(np.isfinite(involved_exec_px) & (involved_exec_px > 0))

            if not invalid_px.any():
                before_weights = calc_actual_weights(shares, val_today, cash).reindex(codes).fillna(0.0)

                new_shares, new_cash, trades_df, after_weights = rebalance_to_target_weights(
                    current_shares=shares,
                    cash=cash,
                    target_weights=target_weights,
                    exec_prices=exec_today,
                    val_prices=val_today,
                    amount_series=amount_today,
                    fee_rate_buy=fee_rate_buy,
                    fee_rate_sell=fee_rate_sell,
                    lot_size=lot_size,
                    max_trade_amount_ratio=max_trade_amount_ratio,
                    amount_unit_scale=amount_unit_scale,
                    trade_date=dt,
                )

                shares = new_shares
                cash = new_cash

                if len(trades_df) > 0:
                    trade_records.append(trades_df)

                turnover = calc_turnover_from_weights(before_weights, after_weights)
                rebalance_logs.append(
                    {
                        "signal_date": signal_date,
                        "trade_date": dt,
                        "reason": reason,
                        "drift_value": drift_value,
                        "turnover": turnover,
                        "cash_after_trade": cash,
                        "traded": int(len(trades_df) > 0),
                        "trade_count": int(len(trades_df)),
                    }
                )
                pending_signal = None

        # 2) 记录当日收盘状态
        nav_today = calc_portfolio_value(shares, val_today, cash)
        ret_today = nav_today / prev_nav - 1.0 if i > 0 else 0.0

        nav_records.append({"trade_date": dt, "nav": nav_today, "cash": cash})
        return_records.append({"trade_date": dt, "return": ret_today})
        position_records.append(pd.DataFrame([shares.values], index=[dt], columns=codes))
        weight_records.append(
            pd.DataFrame(
                [calc_actual_weights(shares, val_today, cash).reindex(codes).fillna(0.0).values],
                index=[dt],
                columns=codes,
            )
        )

        prev_nav = nav_today

        # 3) 当日收盘后生成明日信号
        next_dt = next_trading_date(dates, dt)
        if next_dt is None:
            continue

        if close_px.loc[:dt].shape[0] < min_hist_rows:
            continue

        target_weights_today = compute_tsmom_target_weights_on_date(
            close_price_df=close_px,
            signal_date=dt,
            price_prepare_kwargs=price_prepare_kwargs,
            lookback=lookback,
            skip_recent=skip_recent,
            signal_type=signal_type,
            use_excess_returns=use_excess_returns,
            rf=rf,
            annual_rf=annual_rf,
            return_type=return_type,
            vol_method=vol_method,
            vol_window=vol_window,
            ewma_lambda=ewma_lambda,
            annualization=annualization,
            target_vol=target_vol,
            max_abs_position=max_abs_position,
            side=side,
            signal_threshold=signal_threshold,
            ma_filter_window=ma_filter_window,
            eligibility_mask=eligibility_mask,
            normalize_to_gross=normalize_to_gross,
            vol_floor=vol_floor,
            normalize_trade_weights=normalize_trade_weights,
            combination_method=combination_method,
            horizon_weights=horizon_weights,
            signal_clip=signal_clip,
            zero_to_nan=zero_to_nan,
        )

        if len(target_weights_today) == 0:
            continue

        target_weight_records.append(
            pd.DataFrame([target_weights_today.reindex(codes).fillna(0.0).values], index=[dt], columns=codes)
        )

        current_actual_weights = calc_actual_weights(shares, val_today, cash).reindex(codes).fillna(0.0)

        if pending_signal is None:
            do_rebalance, reason, drift_value = should_rebalance(
                signal_date=dt,
                scheduled_rebalance_dates=scheduled_dates,
                current_actual_weights=current_actual_weights,
                target_weights_today=target_weights_today,
                use_drift_trigger=use_drift_trigger,
                drift_threshold=drift_threshold,
            )
            if do_rebalance:
                pending_signal = {
                    "signal_date": dt,
                    "target_weights": target_weights_today,
                    "reason": reason,
                    "drift_value": drift_value,
                }

    nav_df = pd.DataFrame(nav_records).set_index("trade_date")
    returns = pd.DataFrame(return_records).set_index("trade_date")["return"]
    positions_df = pd.concat(position_records, axis=0).sort_index() if position_records else pd.DataFrame(columns=codes)
    weights_df = pd.concat(weight_records, axis=0).sort_index() if weight_records else pd.DataFrame(columns=codes)
    target_weights_df = (
        pd.concat(target_weight_records, axis=0).sort_index() if target_weight_records else pd.DataFrame(columns=codes)
    )

    if trade_records:
        trades_df = pd.concat(trade_records, axis=0, ignore_index=True)
    else:
        trades_df = pd.DataFrame(columns=["trade_date", "ts_code", "side", "price", "shares", "trade_value", "cost"])

    rebalance_log_df = pd.DataFrame(rebalance_logs)
    if len(rebalance_log_df) > 0:
        rebalance_log_df["signal_date"] = pd.to_datetime(rebalance_log_df["signal_date"])
        rebalance_log_df["trade_date"] = pd.to_datetime(rebalance_log_df["trade_date"])

    summary = performance_summary(
        nav=nav_df["nav"],
        returns=returns,
        risk_free_rate=risk_free_rate,
        annualization=annualization,
    )

    corr_matrix = calc_asset_correlation_matrix(val_px, return_type="log")
    summary["avg_asset_correlation"] = calc_avg_pairwise_correlation(corr_matrix)

    # 诊断矩阵：仅用于分析与制图，不参与交易执行
    prepared_close_full = prepare_price_matrix(close_px, **price_prepare_kwargs)
    diag_outputs, model_target_weights_df = _build_processed_weight_matrix(
        prepared_price=prepared_close_full,
        lookback=lookback,
        skip_recent=skip_recent,
        signal_type=signal_type,
        use_excess_returns=use_excess_returns,
        rf=rf,
        annual_rf=annual_rf,
        return_type=return_type,
        vol_method=vol_method,
        vol_window=vol_window,
        ewma_lambda=ewma_lambda,
        annualization=annualization,
        target_vol=target_vol,
        max_abs_position=max_abs_position,
        side=side,
        signal_threshold=signal_threshold,
        ma_filter_window=ma_filter_window,
        eligibility_mask=eligibility_mask,
        normalize_to_gross=normalize_to_gross,
        vol_floor=vol_floor,
        normalize_trade_weights=normalize_trade_weights,
        combination_method=combination_method,
        horizon_weights=horizon_weights,
        signal_clip=signal_clip,
        zero_to_nan=zero_to_nan,
    )

    weight_stats = summarize_weight_statistics(weights_df)
    exposure_stats = summarize_exposure(weights_df)

    return {
        "nav_df": nav_df,
        "returns": returns,
        "positions_df": positions_df,
        "weights_df": weights_df,
        "target_weights_df": target_weights_df,
        "trades_df": trades_df,
        "rebalance_log_df": rebalance_log_df,
        "asset_corr_matrix": corr_matrix,
        "summary": summary,
        "raw_signal_df": diag_outputs["raw_signal"],
        "signal_df": diag_outputs["signal"],
        "execution_signal_df": diag_outputs.get("execution_signal"),
        "vol_df": diag_outputs["vol"],
        "execution_vol_df": diag_outputs.get("execution_vol"),
        "model_target_weights_df": model_target_weights_df,
        "prepared_close_df": prepared_close_full,
        "horizons": diag_outputs.get("horizons"),
        "horizon_weights": diag_outputs.get("horizon_weights"),
        "horizon_raw_signal_dict": diag_outputs.get("horizon_raw_signal_dict"),
        "horizon_signal_dict": diag_outputs.get("horizon_signal_dict"),
        "horizon_raw_signal_panel": diag_outputs.get("horizon_raw_signal_panel"),
        "horizon_signal_panel": diag_outputs.get("horizon_signal_panel"),
        "weight_stats": weight_stats,
        "exposure_stats": exposure_stats,
    }


__all__ = [
    "TSMOMSignalSnapshot",
    "build_field_matrix",
    "compute_tsmom_target_weights_on_date",
    "compute_target_weights_on_date",
    "make_tsmom_signal_snapshot",
    "simulate_tsmom_backtest",
    "summarize_exposure",
]
