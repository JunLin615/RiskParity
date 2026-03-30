from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from backtest import (
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
from risk_parity import ensure_datetime_index, prepare_price_matrix, solve_risk_parity_weights


@dataclass(frozen=True)
class IntrinsicFeatureWindow:
    """本征风险平价在某个 signal_date 上使用到的标准化特征窗口。"""

    price_returns: pd.DataFrame
    activity_log_ratio: pd.DataFrame
    zr: pd.DataFrame
    zv: pd.DataFrame
    combined_signal: pd.DataFrame


@dataclass(frozen=True)
class IntrinsicCovarianceResult:
    """本征风险平价协方差估计结果。"""

    covariance: pd.DataFrame
    feature_window: IntrinsicFeatureWindow
    lookback_window: int
    long_lookback_window: int
    annualization: int


# ============================================================
# 基础工具
# ============================================================


def _safe_column_zscore(df: pd.DataFrame, ddof: int = 1, eps: float = 1e-12) -> pd.DataFrame:
    """按列做时间序列 z-score；标准差过小的列会被置为 NaN。"""
    if df.empty:
        return df.copy()

    mean = df.mean(axis=0)
    std = df.std(axis=0, ddof=ddof)
    std = std.mask(std.abs() <= eps)
    return (df - mean) / std



def _calc_returns(price_df: pd.DataFrame, return_type: str = "log") -> pd.DataFrame:
    px = ensure_datetime_index(price_df)
    if return_type == "log":
        ret = np.log(px / px.shift(1))
    elif return_type == "simple":
        ret = px.pct_change()
    else:
        raise ValueError("return_type must be 'log' or 'simple'")
    return ret.dropna(how="all")



def _calc_log_activity_ratio(activity_df: pd.DataFrame) -> pd.DataFrame:
    act = ensure_datetime_index(activity_df)
    ratio = np.log(act / act.shift(1))
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    return ratio.dropna(how="all")



def _build_activity_prepare_kwargs(
    price_prepare_kwargs: Optional[dict],
    activity_prepare_kwargs: Optional[dict],
) -> dict:
    if activity_prepare_kwargs is not None:
        return dict(activity_prepare_kwargs)

    base = dict(price_prepare_kwargs or {})
    # 价格可有限前向填充，成交量/成交额默认不前向填充，避免制造虚假活跃度。
    base.setdefault("ffill", False)
    return base



def _slice_long_window(df: pd.DataFrame, signal_date: Optional[pd.Timestamp], long_lookback_window: int) -> pd.DataFrame:
    x = ensure_datetime_index(df)
    if signal_date is not None:
        x = x.loc[:pd.Timestamp(signal_date)]
    # 需要前一日才能计算收益率/比值，因此取 long_lookback_window + 1 行原始数据。
    return x.iloc[-(long_lookback_window + 1):].copy()



def build_intrinsic_feature_window(
    close_price_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    signal_date: Optional[pd.Timestamp] = None,
    long_lookback_window: int = 252,
    price_prepare_kwargs: Optional[dict] = None,
    activity_prepare_kwargs: Optional[dict] = None,
    return_type: str = "log",
    zscore_ddof: int = 1,
    zscore_eps: float = 1e-12,
) -> IntrinsicFeatureWindow:
    """
    构造本征风险平价使用的标准化特征窗口。

    逻辑：
    1. 在 long_lookback_window 上分别计算价格收益率与活跃度比值；
    2. 分别按列做 z-score，得到 zr 与 zv；
    3. 构造 combined_signal = zr + zv。

    说明：
    - 这里的 activity_df 可以是 amount、vol 或任何非负活跃度代理变量矩阵；
    - z-score 是沿时间维度、对每个资产分别计算。
    """
    if long_lookback_window < 2:
        raise ValueError("long_lookback_window must be >= 2")

    price_prepare_kwargs = dict(price_prepare_kwargs or {})
    activity_prepare_kwargs = _build_activity_prepare_kwargs(price_prepare_kwargs, activity_prepare_kwargs)

    raw_price = _slice_long_window(close_price_df, signal_date, long_lookback_window)
    raw_activity = _slice_long_window(activity_df, signal_date, long_lookback_window)

    if len(raw_price) < long_lookback_window + 1 or len(raw_activity) < long_lookback_window + 1:
        empty = pd.DataFrame()
        return IntrinsicFeatureWindow(empty, empty, empty, empty, empty)

    prepared_price = prepare_price_matrix(raw_price, **price_prepare_kwargs)
    prepared_activity = prepare_price_matrix(raw_activity, **activity_prepare_kwargs)

    common_cols = prepared_price.columns.intersection(prepared_activity.columns)
    common_idx = prepared_price.index.intersection(prepared_activity.index)

    if len(common_cols) == 0 or len(common_idx) < 2:
        empty = pd.DataFrame()
        return IntrinsicFeatureWindow(empty, empty, empty, empty, empty)

    prepared_price = prepared_price.reindex(index=common_idx, columns=common_cols)
    prepared_activity = prepared_activity.reindex(index=common_idx, columns=common_cols)

    price_returns = _calc_returns(prepared_price, return_type=return_type)
    activity_log_ratio = _calc_log_activity_ratio(prepared_activity)

    common_cols = price_returns.columns.intersection(activity_log_ratio.columns)
    common_idx = price_returns.index.intersection(activity_log_ratio.index)

    if len(common_cols) == 0 or len(common_idx) == 0:
        empty = pd.DataFrame()
        return IntrinsicFeatureWindow(empty, empty, empty, empty, empty)

    price_returns = price_returns.reindex(index=common_idx, columns=common_cols)
    activity_log_ratio = activity_log_ratio.reindex(index=common_idx, columns=common_cols)

    zr = _safe_column_zscore(price_returns, ddof=zscore_ddof, eps=zscore_eps)
    zv = _safe_column_zscore(activity_log_ratio, ddof=zscore_ddof, eps=zscore_eps)
    combined_signal = zr + zv

    valid_cols = (
        zr.notna().any(axis=0)
        & zv.notna().any(axis=0)
        & combined_signal.notna().any(axis=0)
    )
    keep_cols = valid_cols[valid_cols].index

    if len(keep_cols) == 0:
        empty = pd.DataFrame()
        return IntrinsicFeatureWindow(empty, empty, empty, empty, empty)

    price_returns = price_returns[keep_cols]
    activity_log_ratio = activity_log_ratio[keep_cols]
    zr = zr[keep_cols]
    zv = zv[keep_cols]
    combined_signal = combined_signal[keep_cols]

    return IntrinsicFeatureWindow(
        price_returns=price_returns,
        activity_log_ratio=activity_log_ratio,
        zr=zr,
        zv=zv,
        combined_signal=combined_signal,
    )



def estimate_intrinsic_covariance_from_features(
    feature_window: IntrinsicFeatureWindow,
    lookback_window: int = 120,
    annualization: int = 252,
    drop_any_na: bool = True,
    min_periods: Optional[int] = None,
    min_diag_var: float = 1e-12,
    jitter: float = 0.0,
) -> pd.DataFrame:
    """
    用改造后的本征信号构造协方差矩阵。

    对每个资产定义 x_i = zr_i + zv_i，则：
    - 对角线 Var(x_i) = Var(zr_i) + Var(zv_i) + 2 Cov(zr_i, zv_i)
    - 非对角线 Cov(x_i, x_j)
      = Cov(zr_i, zr_j) + Cov(zr_i, zv_j) + Cov(zv_i, zr_j) + Cov(zv_i, zv_j)

    因而直接对 combined_signal = zr + zv 求样本协方差即可，
    该构造天然保持对称，且样本协方差矩阵在数值上是半正定的。
    """
    if lookback_window < 2:
        raise ValueError("lookback_window must be >= 2")

    x = feature_window.combined_signal.copy()
    if x.empty:
        return pd.DataFrame()

    x = x.iloc[-lookback_window:].copy()
    if drop_any_na:
        x = x.dropna(how="any")

    if min_periods is None:
        min_periods = 2

    if x.shape[0] < 2 or x.shape[1] == 0:
        return pd.DataFrame(columns=x.columns, index=x.columns, dtype=float)

    cov = x.cov(min_periods=min_periods) * annualization
    cov = cov.replace([np.inf, -np.inf], np.nan)

    valid_diag = pd.Series(np.diag(cov), index=cov.index)
    keep = valid_diag[(valid_diag.notna()) & (valid_diag > min_diag_var)].index
    cov = cov.loc[keep, keep]

    if cov.empty:
        return cov

    cov = (cov + cov.T) / 2.0
    if jitter > 0:
        cov = cov + np.eye(len(cov)) * float(jitter)

    return cov



def estimate_intrinsic_covariance_from_matrices(
    close_price_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    signal_date: Optional[pd.Timestamp] = None,
    lookback_window: int = 120,
    long_lookback_window: int = 252,
    price_prepare_kwargs: Optional[dict] = None,
    activity_prepare_kwargs: Optional[dict] = None,
    return_type: str = "log",
    annualization: int = 252,
    drop_any_na: bool = True,
    min_periods: Optional[int] = None,
    min_diag_var: float = 1e-12,
    jitter: float = 0.0,
    zscore_ddof: int = 1,
    zscore_eps: float = 1e-12,
) -> IntrinsicCovarianceResult:
    features = build_intrinsic_feature_window(
        close_price_df=close_price_df,
        activity_df=activity_df,
        signal_date=signal_date,
        long_lookback_window=long_lookback_window,
        price_prepare_kwargs=price_prepare_kwargs,
        activity_prepare_kwargs=activity_prepare_kwargs,
        return_type=return_type,
        zscore_ddof=zscore_ddof,
        zscore_eps=zscore_eps,
    )

    cov = estimate_intrinsic_covariance_from_features(
        feature_window=features,
        lookback_window=lookback_window,
        annualization=annualization,
        drop_any_na=drop_any_na,
        min_periods=min_periods,
        min_diag_var=min_diag_var,
        jitter=jitter,
    )

    return IntrinsicCovarianceResult(
        covariance=cov,
        feature_window=features,
        lookback_window=lookback_window,
        long_lookback_window=long_lookback_window,
        annualization=annualization,
    )



def calc_intrinsic_risk_parity_weights_from_covariance(
    cov_matrix: pd.DataFrame,
    target_risk_budget: Optional[np.ndarray | list[float]] = None,
    initial_weights: Optional[np.ndarray | list[float]] = None,
    long_only: bool = True,
    weight_bounds: Optional[list[tuple[float, float]]] = None,
    tol: float = 1e-12,
    maxiter: int = 10_000,
) -> pd.Series:
    if cov_matrix.empty:
        return pd.Series(dtype=float)

    weights = solve_risk_parity_weights(
        cov_matrix=cov_matrix,
        target_risk_budget=target_risk_budget,
        initial_weights=initial_weights,
        long_only=long_only,
        weight_bounds=weight_bounds,
        tol=tol,
        maxiter=maxiter,
    )
    return pd.Series(weights, index=cov_matrix.index, name="weight")



def _extract_solver_kwargs(weight_kwargs: Optional[dict]) -> dict:
    weight_kwargs = dict(weight_kwargs or {})
    allowed = {
        "target_risk_budget",
        "initial_weights",
        "long_only",
        "weight_bounds",
        "tol",
        "maxiter",
    }
    return {k: v for k, v in weight_kwargs.items() if k in allowed}



def _extract_covariance_kwargs(weight_kwargs: Optional[dict]) -> dict:
    weight_kwargs = dict(weight_kwargs or {})
    allowed = {
        "annualization",
        "drop_any_na",
        "min_periods",
        "min_diag_var",
        "jitter",
        "zscore_ddof",
        "zscore_eps",
        "return_type",
    }
    return {k: v for k, v in weight_kwargs.items() if k in allowed}



def calc_intrinsic_risk_parity_weights_from_matrices(
    close_price_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    signal_date: Optional[pd.Timestamp] = None,
    lookback_window: int = 120,
    long_lookback_window: int = 252,
    price_prepare_kwargs: Optional[dict] = None,
    activity_prepare_kwargs: Optional[dict] = None,
    irp_weight_kwargs: Optional[dict] = None,
) -> pd.Series:
    """
    从价格矩阵 + 活跃度矩阵直接计算本征风险平价权重。

    兼容原 risk_parity/backtest 的参数风格：
    - annualization / drop_any_na 等协方差相关参数可放在 irp_weight_kwargs 中；
    - long_only / weight_bounds / tol / maxiter 等求解参数也可放在 irp_weight_kwargs 中。
    """
    cov_kwargs = _extract_covariance_kwargs(irp_weight_kwargs)
    solver_kwargs = _extract_solver_kwargs(irp_weight_kwargs)

    cov_result = estimate_intrinsic_covariance_from_matrices(
        close_price_df=close_price_df,
        activity_df=activity_df,
        signal_date=signal_date,
        lookback_window=lookback_window,
        long_lookback_window=long_lookback_window,
        price_prepare_kwargs=price_prepare_kwargs,
        activity_prepare_kwargs=activity_prepare_kwargs,
        annualization=cov_kwargs.pop("annualization", 252),
        drop_any_na=cov_kwargs.pop("drop_any_na", True),
        min_periods=cov_kwargs.pop("min_periods", None),
        min_diag_var=cov_kwargs.pop("min_diag_var", 1e-12),
        jitter=cov_kwargs.pop("jitter", 0.0),
        return_type=cov_kwargs.pop("return_type", "log"),
        zscore_ddof=cov_kwargs.pop("zscore_ddof", 1),
        zscore_eps=cov_kwargs.pop("zscore_eps", 1e-12),
    )

    weights = calc_intrinsic_risk_parity_weights_from_covariance(
        cov_matrix=cov_result.covariance,
        target_risk_budget=solver_kwargs.get("target_risk_budget"),
        initial_weights=solver_kwargs.get("initial_weights"),
        long_only=solver_kwargs.get("long_only", True),
        weight_bounds=solver_kwargs.get("weight_bounds"),
        tol=solver_kwargs.get("tol", 1e-12),
        maxiter=solver_kwargs.get("maxiter", 10_000),
    )

    full_index = ensure_datetime_index(close_price_df).columns
    return normalize_series_weights(weights.reindex(full_index).fillna(0.0))



def compute_intrinsic_target_weights_on_date(
    market: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    lookback_window: int = 120,
    long_lookback_window: int = 252,
    activity_field: str = "amount",
    irp_prepare_kwargs: Optional[dict] = None,
    activity_prepare_kwargs: Optional[dict] = None,
    irp_weight_kwargs: Optional[dict] = None,
) -> pd.Series:
    market = ensure_same_index_columns(market)

    if activity_field not in market:
        raise ValueError(f"market missing '{activity_field}' for intrinsic risk parity")

    close_px = ensure_datetime_index(market["close"])
    activity_df = ensure_datetime_index(market[activity_field])

    return calc_intrinsic_risk_parity_weights_from_matrices(
        close_price_df=close_px,
        activity_df=activity_df,
        signal_date=signal_date,
        lookback_window=lookback_window,
        long_lookback_window=long_lookback_window,
        price_prepare_kwargs=irp_prepare_kwargs,
        activity_prepare_kwargs=activity_prepare_kwargs,
        irp_weight_kwargs=irp_weight_kwargs,
    )



def calc_intrinsic_risk_contribution(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
) -> pd.Series:
    idx = weights.index.intersection(cov_matrix.index)
    if len(idx) == 0:
        return pd.Series(dtype=float)

    w = weights.reindex(idx).fillna(0.0).values
    cov = cov_matrix.loc[idx, idx].values

    port_var = float(w.T @ cov @ w)
    if port_var <= 0:
        return pd.Series(0.0, index=idx)

    port_vol = np.sqrt(port_var)
    mrc = (cov @ w) / port_vol
    rc = w * mrc
    prc = rc / port_vol
    return pd.Series(prc, index=idx)



def calc_historical_intrinsic_risk_contributions(
    weights_df: pd.DataFrame,
    close_price_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    lookback_window: int = 120,
    long_lookback_window: int = 252,
    irp_prepare_kwargs: Optional[dict] = None,
    activity_prepare_kwargs: Optional[dict] = None,
    irp_weight_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    records = []
    cov_kwargs = _extract_covariance_kwargs(irp_weight_kwargs)

    for dt, weights in weights_df.iterrows():
        cov_result = estimate_intrinsic_covariance_from_matrices(
            close_price_df=close_price_df,
            activity_df=activity_df,
            signal_date=dt,
            lookback_window=lookback_window,
            long_lookback_window=long_lookback_window,
            price_prepare_kwargs=irp_prepare_kwargs,
            activity_prepare_kwargs=activity_prepare_kwargs,
            annualization=cov_kwargs.get("annualization", 252),
            drop_any_na=cov_kwargs.get("drop_any_na", True),
            min_periods=cov_kwargs.get("min_periods"),
            min_diag_var=cov_kwargs.get("min_diag_var", 1e-12),
            jitter=cov_kwargs.get("jitter", 0.0),
            return_type=cov_kwargs.get("return_type", "log"),
            zscore_ddof=cov_kwargs.get("zscore_ddof", 1),
            zscore_eps=cov_kwargs.get("zscore_eps", 1e-12),
        )

        if cov_result.covariance.empty:
            records.append(pd.Series(0.0, index=weights.index, name=dt))
            continue

        rc = calc_intrinsic_risk_contribution(weights, cov_result.covariance)
        rc = rc.reindex(weights.index).fillna(0.0)
        rc.name = dt
        records.append(rc)

    if not records:
        return pd.DataFrame(columns=weights_df.columns)

    return pd.DataFrame(records)


# ============================================================
# 主回测函数
# ============================================================


def simulate_intrinsic_risk_parity_backtest(
    market: dict[str, pd.DataFrame],
    initial_cash: float = 1_000_000.0,
    lookback_window: int = 120,
    long_lookback_window: int = 252,
    activity_field: str = "amount",
    rebalance_freq: str = "Q",
    execution_price_type: str = "avg",
    valuation_ffill_limit: int = 5,
    fee_rate_buy: float = 0.0005,
    fee_rate_sell: float = 0.0005,
    lot_size: int | dict[str, int] = 100,
    max_trade_amount_ratio: Optional[float] = 0.05,
    amount_unit_scale: float = 1000.0,
    use_drift_trigger: bool = False,
    drift_threshold: float = 0.05,
    irp_prepare_kwargs: Optional[dict] = None,
    activity_prepare_kwargs: Optional[dict] = None,
    irp_weight_kwargs: Optional[dict] = None,
    risk_free_rate: float = 0.0,
    annualization: int = 252,
) -> dict[str, object]:
    """
    本征风险平价回测主入口。

    与原 simulate_risk_parity_backtest 保持同样的交易/回测框架：
    - 当日收盘后生成信号；
    - 下一交易日按 execution_price 执行；
    - 支持成交额占比约束、整手交易、停牌顺延、偏离触发调仓。

    主要差异：
    - 权重由“价格收益率 + 活跃度比值”的标准化组合构造；
    - 只有在历史原始数据长度满足 long_lookback_window + 1 后，资产才会进入权重计算。
    """
    if long_lookback_window < lookback_window:
        raise ValueError("long_lookback_window must be >= lookback_window")

    irp_prepare_kwargs = dict(irp_prepare_kwargs or {})
    irp_weight_kwargs = dict(irp_weight_kwargs or {})

    market = ensure_same_index_columns(market)
    if activity_field not in market:
        raise ValueError(f"market missing '{activity_field}'")

    close_px = ensure_datetime_index(market["close"])
    val_px = close_px.ffill(limit=valuation_ffill_limit)
    exec_px = get_execution_price_matrix(market, execution_price_type=execution_price_type)
    amount_px = market.get("amount", None)
    activity_df = ensure_datetime_index(market[activity_field])

    dates = close_px.index
    codes = close_px.columns
    scheduled_dates = get_rebalance_dates(dates, freq=rebalance_freq)

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

    for i, dt in enumerate(dates):
        val_today = val_px.loc[dt]
        exec_today = exec_px.loc[dt]
        amount_today = amount_px.loc[dt] if amount_px is not None else None

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

        next_dt = next_trading_date(dates, dt)
        if next_dt is None:
            continue

        # 至少要有 long_lookback_window + 1 个原始交易日，才能生成 long_lookback_window 个收益率观测。
        if close_px.loc[:dt].shape[0] < long_lookback_window + 1:
            continue
        if activity_df.loc[:dt].shape[0] < long_lookback_window + 1:
            continue

        target_weights_today = compute_intrinsic_target_weights_on_date(
            market=market,
            signal_date=dt,
            lookback_window=lookback_window,
            long_lookback_window=long_lookback_window,
            activity_field=activity_field,
            irp_prepare_kwargs=irp_prepare_kwargs,
            activity_prepare_kwargs=activity_prepare_kwargs,
            irp_weight_kwargs=irp_weight_kwargs,
        )

        if len(target_weights_today) == 0 or target_weights_today.sum() <= 0:
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
    avg_corr = calc_avg_pairwise_correlation(corr_matrix)
    summary["avg_asset_correlation"] = avg_corr

    risk_contribution_df = calc_historical_intrinsic_risk_contributions(
        weights_df=weights_df,
        close_price_df=close_px,
        activity_df=activity_df,
        lookback_window=lookback_window,
        long_lookback_window=long_lookback_window,
        irp_prepare_kwargs=irp_prepare_kwargs,
        activity_prepare_kwargs=activity_prepare_kwargs,
        irp_weight_kwargs=irp_weight_kwargs,
    )

    return {
        "nav_df": nav_df,
        "returns": returns,
        "positions_df": positions_df,
        "weights_df": weights_df,
        "target_weights_df": target_weights_df,
        "trades_df": trades_df,
        "rebalance_log_df": rebalance_log_df,
        "asset_corr_matrix": corr_matrix,
        "risk_contribution_df": risk_contribution_df,
        "summary": summary,
    }


__all__ = [
    "IntrinsicFeatureWindow",
    "IntrinsicCovarianceResult",
    "build_intrinsic_feature_window",
    "estimate_intrinsic_covariance_from_features",
    "estimate_intrinsic_covariance_from_matrices",
    "calc_intrinsic_risk_parity_weights_from_covariance",
    "calc_intrinsic_risk_parity_weights_from_matrices",
    "compute_intrinsic_target_weights_on_date",
    "calc_intrinsic_risk_contribution",
    "calc_historical_intrinsic_risk_contributions",
    "simulate_intrinsic_risk_parity_backtest",
]
