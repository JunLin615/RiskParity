from __future__ import annotations

"""
intrinsic_backtest.py

纯函数版回测函数库（本征风险平价策略）
依赖：
- pandas
- numpy
- backtest.py
- intrinsic_risk_parity.py
- tsmom_utils.py

设计目标：
1. 尽量复用原 risk parity 回测库的交易/估值/调仓框架
2. 将“权重生成器”替换为 intrinsic_risk_parity.py 中的本征风险平价权重逻辑
3. 保持纯函数，不与数据库、notebook、实盘模块耦合
4. 严格避免未来函数：
   - t 日收盘后用截至 t 的历史数据生成信号
   - t+1 按指定成交价口径成交
5. 支持多周期 TSMOM 门控：
   - 使用 t 日收盘后可得信息生成 {-1, 0, 1} 门控信号
   - 仅保留门控为 +1 的资产进入当次 IRP 资产池
   - 若当次资产池为空，则目标权重全为 0，组合持有现金
"""

from typing import Optional, Sequence

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
    performance_summary,
    rebalance_to_target_weights,
    should_rebalance,
)
from intrinsic_risk_parity import (
    calc_historical_intrinsic_risk_contributions,
    compute_intrinsic_target_weights_on_date as _compute_intrinsic_target_weights_on_date,
)
from risk_parity import ensure_datetime_index
from tsmom_utils import build_tsmom_gate_bundle


# ============================================================
# 门控工具
# ============================================================


def _normalize_multi_period_gate_params(
    multi_period_gate_params: Optional[dict],
) -> Optional[dict]:
    if multi_period_gate_params is None:
        return None

    params = dict(multi_period_gate_params)
    params.setdefault("execution_lag", 0)
    params.setdefault("gate_type", "directional")
    return params



def build_multi_period_gate_signal_matrix(
    market: dict[str, pd.DataFrame],
    gate_price_field: str = "close",
    multi_period_gate_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    预先构造整段回测区间上的多周期门控矩阵。

    返回值恒为与 market[gate_price_field] 同 shape 的 DataFrame，
    元素严格离散到 {-1, 0, 1}。

    若 multi_period_gate_params 为 None，则返回全 1 矩阵，表示
    “不过滤任何资产”。
    """
    market = ensure_same_index_columns(market)
    price_df = ensure_datetime_index(market[gate_price_field])

    if multi_period_gate_params is None:
        return pd.DataFrame(1.0, index=price_df.index, columns=price_df.columns)

    params = _normalize_multi_period_gate_params(multi_period_gate_params)
    gate_bundle = build_tsmom_gate_bundle(
        price_df=price_df,
        **params,
    )

    gate_df = ensure_datetime_index(gate_bundle["gate"])
    gate_df = gate_df.reindex(index=price_df.index, columns=price_df.columns)
    gate_df = gate_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gate_df = np.sign(gate_df).astype(float)
    return gate_df


# ============================================================
# 权重计算封装
# ============================================================


def compute_intrinsic_target_weights_on_date(
    market: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    lookback_window: int = 120,
    long_lookback_window: int = 252,
    activity_field: str = "amount",
    irp_prepare_kwargs: Optional[dict] = None,
    activity_prepare_kwargs: Optional[dict] = None,
    irp_weight_kwargs: Optional[dict] = None,
    time_momentum_filter_window: Optional[int] = None,
    eligible_assets: Optional[Sequence[str]] = None,
) -> pd.Series:
    """
    在 signal_date 当日收盘后，用 intrinsic_risk_parity.py 中的逻辑
    计算目标本征风险平价权重。

    若提供 eligible_assets，则先把资产池裁切为该子集，再做 IRP。
    若裁切后资产池为空，则返回全 0 权重（持有现金）。
    """
    market = ensure_same_index_columns(market)
    full_index = ensure_datetime_index(market["close"]).columns

    filtered_market = market
    if eligible_assets is not None:
        eligible_cols = full_index.intersection(pd.Index(list(eligible_assets)))
        if len(eligible_cols) == 0:
            return pd.Series(0.0, index=full_index, name="weight")
        filtered_market = {k: v.loc[:, eligible_cols].copy() for k, v in market.items()}

    weights = _compute_intrinsic_target_weights_on_date(
        market=filtered_market,
        signal_date=signal_date,
        lookback_window=lookback_window,
        long_lookback_window=long_lookback_window,
        activity_field=activity_field,
        irp_prepare_kwargs=irp_prepare_kwargs,
        activity_prepare_kwargs=activity_prepare_kwargs,
        irp_weight_kwargs=irp_weight_kwargs,
        time_momentum_filter_window=time_momentum_filter_window,
    )
    return weights.reindex(full_index).fillna(0.0)


# 为了尽量贴近原 backtest.py 的接口风格，给一个同名别名。
compute_target_weights_on_date = compute_intrinsic_target_weights_on_date


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
    time_momentum_filter_window: Optional[int] = None,
    multi_period_gate_params: Optional[dict] = None,
    multi_period_gate_price_field: str = "close",
    risk_free_rate: float = 0.0,
    annualization: int = 252,
) -> dict[str, object]:
    """
    本征风险平价回测主入口。

    与原 simulate_risk_parity_backtest 基本保持一致，唯一核心差异是：
    目标权重由 intrinsic_risk_parity.py 负责生成。

    新增参数
    --------
    multi_period_gate_params : dict | None
        若不为 None，则使用 tsmom_utils.py 中的多周期门控逻辑生成门控矩阵，
        每次在计算 IRP 调仓目标前，仅保留门控为 +1 的资产；门控为 -1 / 0
        的资产暂时剔除出资产池。
    multi_period_gate_price_field : str
        构造门控时使用的价格字段，默认 close。
    """
    if long_lookback_window < lookback_window:
        raise ValueError("long_lookback_window must be >= lookback_window")

    irp_prepare_kwargs = dict(irp_prepare_kwargs or {})
    activity_prepare_kwargs = dict(activity_prepare_kwargs or {}) if activity_prepare_kwargs is not None else None
    irp_weight_kwargs = dict(irp_weight_kwargs or {})

    market = ensure_same_index_columns(market)
    if activity_field not in market:
        raise ValueError(f"market missing '{activity_field}'")
    if multi_period_gate_price_field not in market:
        raise ValueError(f"market missing '{multi_period_gate_price_field}'")

    close_px = ensure_datetime_index(market["close"])

    # 1. 估值价格：针对 close 做有限前向填充，用于计算市值和盘后权重
    val_px = close_px.ffill(limit=valuation_ffill_limit)

    # 2. 交易价格：保持原始口径不填充，遇到 NaN 视为当日不可交易
    exec_px = get_execution_price_matrix(market, execution_price_type=execution_price_type)
    amount_px = market.get("amount", None)
    activity_df = ensure_datetime_index(market[activity_field])

    dates = close_px.index
    codes = close_px.columns
    scheduled_dates = get_rebalance_dates(dates, freq=rebalance_freq)

    gate_signal_df = build_multi_period_gate_signal_matrix(
        market=market,
        gate_price_field=multi_period_gate_price_field,
        multi_period_gate_params=multi_period_gate_params,
    ).reindex(index=dates, columns=codes).fillna(0.0)
    gate_signal_df = np.sign(gate_signal_df).astype(float)
    gate_eligible_count = (gate_signal_df > 0).sum(axis=1).rename("eligible_asset_count")

    # 组合状态
    shares = pd.Series(0, index=codes, dtype=int)
    cash = float(initial_cash)

    # 待执行信号：在 t 日生成，在 t+1 执行
    pending_signal = None

    # 记录容器
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

        # --------------------------------------------------
        # 1. 如果昨天有信号，今天执行调仓
        # --------------------------------------------------
        if pending_signal is not None:
            target_weights = pending_signal["target_weights"].reindex(codes).fillna(0.0)
            signal_date = pending_signal["signal_date"]
            reason = pending_signal["reason"]
            drift_value = pending_signal["drift_value"]
            gate_eligible_assets = pending_signal["gate_eligible_assets"]

            involved_assets = shares[shares > 0].index.union(target_weights[target_weights > 0].index)
            involved_exec_px = exec_today.reindex(involved_assets)

            # 若相关资产任一今日不可交易，则顺延
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
                        "gate_eligible_assets": int(gate_eligible_assets),
                    }
                )
                pending_signal = None

        # --------------------------------------------------
        # 2. 当日收盘后记录净值 / 持仓 / 实际权重
        # --------------------------------------------------
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

        # --------------------------------------------------
        # 3. 用截至今日收盘的数据生成“明天执行”的调仓信号
        # --------------------------------------------------
        next_dt = next_trading_date(dates, dt)
        if next_dt is None:
            continue

        # 至少需要 llbw + 1 个原始交易日，才能形成 llbw 个收益率/活跃度比值观测。
        if close_px.loc[:dt].shape[0] < long_lookback_window + 1:
            continue
        if activity_df.loc[:dt].shape[0] < long_lookback_window + 1:
            continue

        gate_signal_today = gate_signal_df.loc[dt].reindex(codes).fillna(0.0).astype(float)
        eligible_assets_today = gate_signal_today[gate_signal_today > 0].index

        target_weights_today = compute_intrinsic_target_weights_on_date(
            market=market,
            signal_date=dt,
            lookback_window=lookback_window,
            long_lookback_window=long_lookback_window,
            activity_field=activity_field,
            irp_prepare_kwargs=irp_prepare_kwargs,
            activity_prepare_kwargs=activity_prepare_kwargs,
            irp_weight_kwargs=irp_weight_kwargs,
            time_momentum_filter_window=time_momentum_filter_window,
            eligible_assets=eligible_assets_today,
        )

        target_weight_records.append(
            pd.DataFrame([target_weights_today.reindex(codes).fillna(0.0).values], index=[dt], columns=codes)
        )

        current_actual_weights = calc_actual_weights(shares, val_today, cash).reindex(codes).fillna(0.0)

        # 若已有待执行顺延信号，则不覆盖
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
                    "gate_eligible_assets": int((gate_signal_today > 0).sum()),
                }

    # ------------------------------------------------------
    # 汇总输出
    # ------------------------------------------------------
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
    summary["avg_gate_eligible_assets"] = float(gate_eligible_count.mean()) if len(gate_eligible_count) > 0 else np.nan

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
        "gate_signal_df": gate_signal_df,
        "gate_eligible_count": gate_eligible_count,
        "gate_enabled": multi_period_gate_params is not None,
        "summary": summary,
    }


__all__ = [
    "build_multi_period_gate_signal_matrix",
    "compute_intrinsic_target_weights_on_date",
    "compute_target_weights_on_date",
    "simulate_intrinsic_risk_parity_backtest",
]


if __name__ == "__main__":
    dates = pd.date_range("2024-01-02", periods=260, freq="B")
    codes = ["510300.SH", "511010.SH", "518880.SH"]

    rng = np.random.default_rng(42)

    def make_price(start: float) -> pd.Series:
        r = rng.normal(0, 0.01, len(dates))
        px = start * np.exp(np.cumsum(r))
        return pd.Series(px, index=dates)

    close = pd.DataFrame({
        "510300.SH": make_price(3.5),
        "511010.SH": make_price(112.0),
        "518880.SH": make_price(4.8),
    }, index=dates)

    open_ = close * (1 + rng.normal(0, 0.002, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, close.shape)))
    amount = pd.DataFrame(2e8, index=dates, columns=codes)

    market = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "amount": amount,
    }

    result = simulate_intrinsic_risk_parity_backtest(
        market=market,
        initial_cash=1_000_000.0,
        lookback_window=60,
        long_lookback_window=120,
        activity_field="amount",
        rebalance_freq="M",
        execution_price_type="avg",
        valuation_ffill_limit=5,
        fee_rate_buy=0.0005,
        fee_rate_sell=0.0005,
        lot_size=100,
        max_trade_amount_ratio=0.05,
        amount_unit_scale=1000.0,
        use_drift_trigger=True,
        drift_threshold=0.08,
        irp_prepare_kwargs={
            "calendar": close.index,
            "ffill": True,
            "ffill_limit": 5,
            "min_non_na_ratio": 0.8,
            "drop_all_na_dates": True,
        },
        activity_prepare_kwargs={
            "calendar": close.index,
            "ffill": False,
            "min_non_na_ratio": 0.8,
            "drop_all_na_dates": True,
        },
        irp_weight_kwargs={
            "annualization": 252,
            "drop_any_na": True,
            "long_only": True,
        },
        multi_period_gate_params={
            "lookback": [21, 63, 126],
            "signal_type": "sign",
            "use_excess_returns": False,
            "combination_method": "vote",
            "gate_type": "directional",
            "gate_threshold": 0.0,
            "execution_lag": 0,
        },
    )

    print(result["summary"])
    print(result["nav_df"].tail())
    print(result["trades_df"].head())
