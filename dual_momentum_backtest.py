
"""
dual_momentum_backtest.py

纯函数版双动量（Dual Momentum）回测函数库
依赖：
- pandas
- numpy
- dual_momentum_fixed.py

特点：
1. 不和 notebook、数据库、实盘模块耦合
2. 支持整手交易
3. long-only，无杠杆，不允许负现金
4. 支持交易成本
5. 支持固定调仓周期 + 偏离触发调仓
6. 严格避免未来函数：
   - 在 t 日收盘后用截至 t 的历史数据生成双动量目标权重
   - 在下一交易日 t+1 按指定成交价口径成交
7. 支持成交额占比限制，避免单日成交额占比过高
8. 区分估值价格（有限前向填充）与交易价格（不填充，默认 best-effort 处理停牌）
9. 支持显式现金仓位：当双动量目标权重未满配时，剩余仓位保留现金

说明：
- 默认成交价 execution_price_type = "avg"
- 这里的 avg 默认定义为 (open + high + low + close) / 4
- 如果你想改成别的“均价”定义，可以自行改 get_execution_price_matrix
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from dual_momentum import (
    calc_dual_momentum_snapshot,
    ensure_datetime_index,
    prepare_price_matrix,
)


# ============================================================
# 基础工具
# ============================================================

def normalize_weights(weights: pd.Series) -> pd.Series:
    w = weights.astype(float).copy()
    s = w.sum()
    if s <= 0:
        return pd.Series(0.0, index=w.index)
    return w / s


def ensure_same_index_columns(df_dict: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    确保多个矩阵的 index / columns 一致，以 close 为主基准。
    """
    if "close" not in df_dict:
        raise ValueError("df_dict must contain 'close'")

    base = ensure_datetime_index(df_dict["close"])
    idx = base.index
    cols = base.columns

    out = {}
    for k, df in df_dict.items():
        x = ensure_datetime_index(df)
        x = x.reindex(index=idx, columns=cols)
        out[k] = x
    return out


def build_field_matrix(
    data: pd.DataFrame,
    field: str,
    date_col: str = "trade_date",
    code_col: str = "ts_code",
    date_format: Optional[str] = "%Y%m%d",
) -> pd.DataFrame:
    """
    从长表构建单个字段矩阵。
    """
    df = data[[date_col, code_col, field]].copy()
    if date_format is None:
        df[date_col] = pd.to_datetime(df[date_col])
    else:
        df[date_col] = pd.to_datetime(df[date_col], format=date_format)

    mat = (
        df.pivot(index=date_col, columns=code_col, values=field)
          .sort_index()
          .sort_index(axis=1)
    )
    return mat


def build_market_matrices(
    data: pd.DataFrame,
    fields: Sequence[str] = ("open", "high", "low", "close", "amount"),
    date_col: str = "trade_date",
    code_col: str = "ts_code",
    date_format: Optional[str] = "%Y%m%d",
) -> dict[str, pd.DataFrame]:
    """
    从长表构建多个市场字段矩阵。
    """
    mats = {}
    for field in fields:
        mats[field] = build_field_matrix(
            data=data,
            field=field,
            date_col=date_col,
            code_col=code_col,
            date_format=date_format,
        )
    return ensure_same_index_columns(mats)


def get_execution_price_matrix(
    market: dict[str, pd.DataFrame],
    execution_price_type: str = "avg",
) -> pd.DataFrame:
    """
    生成成交价矩阵。原始口径，不填充。
    支持：
    - open
    - close
    - high
    - low
    - avg  -> (open + high + low + close) / 4
    """
    market = ensure_same_index_columns(market)

    if execution_price_type in {"open", "close", "high", "low"}:
        return market[execution_price_type].copy()

    if execution_price_type == "avg":
        needed = ["open", "high", "low", "close"]
        for k in needed:
            if k not in market:
                raise ValueError(f"market missing '{k}' for execution_price_type='avg'")
        return (market["open"] + market["high"] + market["low"] + market["close"]) / 4.0

    raise ValueError("execution_price_type must be one of: open, close, high, low, avg")


def get_rebalance_dates(
    trading_dates: Sequence,
    freq: str = "Q",
) -> pd.DatetimeIndex:
    """
    获取固定调仓日列表。
    """
    idx = pd.DatetimeIndex(pd.to_datetime(trading_dates)).sort_values().unique()
    s = pd.Series(idx, index=idx)

    if freq == "D":
        return idx
    if freq == "W":
        return pd.DatetimeIndex(s.groupby(idx.to_period("W")).last().values)
    if freq == "M":
        return pd.DatetimeIndex(s.groupby(idx.to_period("M")).last().values)
    if freq == "Q":
        return pd.DatetimeIndex(s.groupby(idx.to_period("Q")).last().values)
    if freq == "Y":
        return pd.DatetimeIndex(s.groupby(idx.to_period("Y")).last().values)

    raise ValueError("freq must be one of: D, W, M, Q, Y")


def next_trading_date(
    trading_index: pd.DatetimeIndex,
    date: pd.Timestamp,
) -> Optional[pd.Timestamp]:
    """
    返回给定交易日的下一交易日。
    """
    pos = trading_index.get_indexer([pd.Timestamp(date)])[0]
    if pos < 0 or pos >= len(trading_index) - 1:
        return None
    return trading_index[pos + 1]


def calc_portfolio_value(
    shares: pd.Series,
    prices: pd.Series,
    cash: float,
) -> float:
    """
    组合总资产 = 持仓市值 + 现金
    （通常使用 valuation price）
    """
    aligned_prices = prices.reindex(shares.index)
    return float((shares * aligned_prices).fillna(0.0).sum() + cash)


def calc_actual_weights(
    shares: pd.Series,
    prices: pd.Series,
    cash: float = 0.0,
) -> pd.Series:
    """
    用给定价格计算当前持仓对应的实际权重（不含现金权重）。
    （通常使用 valuation price）
    """
    mv = (shares * prices.reindex(shares.index)).fillna(0.0)
    total = mv.sum() + cash
    if total <= 0:
        return pd.Series(0.0, index=shares.index)
    return mv / total


def calc_cash_weight(
    shares: pd.Series,
    prices: pd.Series,
    cash: float = 0.0,
) -> float:
    """
    用给定价格计算当前现金权重。
    """
    mv = (shares * prices.reindex(shares.index)).fillna(0.0).sum()
    total = float(mv + cash)
    if total <= 0:
        return 0.0
    return float(cash / total)


def combine_weights_with_cash(
    asset_weights: pd.Series,
    cash_weight: float,
    cash_column_name: str = "CASH",
) -> pd.Series:
    """
    将资产权重与现金权重合并成一个 Series。
    """
    w = asset_weights.astype(float).copy()
    w.loc[cash_column_name] = float(cash_weight)
    return w


def calc_weight_drift(
    actual_weights: pd.Series,
    target_weights: pd.Series,
) -> float:
    """
    权重偏离度：sum(abs(actual - target))
    """
    actual = actual_weights.reindex(target_weights.index).fillna(0.0)
    target = target_weights.reindex(actual_weights.index).fillna(0.0)
    union_idx = actual.index.union(target.index)
    actual = actual.reindex(union_idx).fillna(0.0)
    target = target.reindex(union_idx).fillna(0.0)
    return float((actual - target).abs().sum())


def calc_turnover_from_weights(
    old_weights: pd.Series,
    new_weights: pd.Series,
) -> float:
    """
    权重口径换手率：sum(abs(new-old))
    注：这里通常使用资产权重（不含现金），避免现金项导致双重计数。
    """
    union_idx = old_weights.index.union(new_weights.index)
    ow = old_weights.reindex(union_idx).fillna(0.0)
    nw = new_weights.reindex(union_idx).fillna(0.0)
    return float((nw - ow).abs().sum())


def get_trade_value_cap(
    amount_series: Optional[pd.Series],
    max_trade_amount_ratio: Optional[float],
    amount_unit_scale: float = 1000.0,  # tushare 的 amount 往往是千元
) -> pd.Series:
    """
    根据当日成交额生成每个标的允许的最大成交额。
    """
    if amount_series is None or max_trade_amount_ratio is None:
        if amount_series is None:
            return pd.Series(dtype=float)
        return pd.Series(np.inf, index=amount_series.index)

    cap = amount_series.astype(float) * float(amount_unit_scale) * float(max_trade_amount_ratio)
    return cap


def round_shares_to_lot(
    shares: float,
    lot_size: int = 100,
) -> int:
    """
    向下取整到整手。
    """
    if shares <= 0:
        return 0
    return int(np.floor(shares / lot_size) * lot_size)


def compute_target_weights_on_date(
    close_price_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    dm_prepare_kwargs: Optional[dict[str, Any]] = None,
    dm_snapshot_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    在 signal_date 当日收盘后，用截至 signal_date 的历史价格计算双动量目标权重。
    返回 snapshot 字典，至少包含：
    - target_weights
    - cash_weight
    - selected_assets
    - defensive_assets_used
    - detail
    - market_regime_on
    """
    dm_prepare_kwargs = dm_prepare_kwargs or {}
    dm_snapshot_kwargs = dm_snapshot_kwargs or {}

    px = ensure_datetime_index(close_price_df)
    hist = px.loc[:signal_date].copy()

    prepared = prepare_price_matrix(hist, **dm_prepare_kwargs)

    invalid_snapshot = {
        "as_of_date": pd.Timestamp(signal_date),
        "selected_assets": [],
        "defensive_assets_used": [],
        "target_weights": pd.Series(dtype=float),
        "cash_weight": 1.0,
        "detail": pd.DataFrame(),
        "market_regime_on": False,
        "signal_valid": False,
    }

    if prepared.shape[0] < 2 or prepared.shape[1] < 1:
        return invalid_snapshot

    try:
        snap = calc_dual_momentum_snapshot(
            price_df=prepared,
            as_of_date=signal_date,
            **dm_snapshot_kwargs,
        )
    except Exception as e:
        empty_w = pd.Series(0.0, index=close_price_df.columns, name="target_weight")
        return {
            "signal_valid": False,
            "signal_error": str(e),
            "as_of_date": signal_date,
            "selected_assets": [],
            "defensive_assets_used": [],
            "target_weights": empty_w,
            "cash_weight": 1.0,
            "detail": pd.DataFrame(),
            "market_regime_on": False,
        }

    snap["target_weights"] = snap["target_weights"].reindex(px.columns).fillna(0.0)
    snap["cash_weight"] = float(snap.get("cash_weight", max(0.0, 1.0 - float(snap["target_weights"].sum()))))
    snap["signal_valid"] = True
    return snap


def should_rebalance(
    signal_date: pd.Timestamp,
    scheduled_rebalance_dates: pd.DatetimeIndex,
    current_actual_weights: pd.Series,
    target_weights_today: Optional[pd.Series],
    use_drift_trigger: bool = False,
    drift_threshold: float = 0.05,
) -> tuple[bool, str, float]:
    """
    判断 signal_date 是否触发调仓。
    """
    is_scheduled = pd.Timestamp(signal_date) in pd.DatetimeIndex(scheduled_rebalance_dates)

    drift_value = np.nan
    drift_hit = False

    if use_drift_trigger and target_weights_today is not None and len(target_weights_today) > 0:
        drift_value = calc_weight_drift(current_actual_weights, target_weights_today)
        drift_hit = drift_value >= drift_threshold

    if is_scheduled and drift_hit:
        return True, "schedule+drift", drift_value
    if is_scheduled:
        return True, "schedule", drift_value
    if drift_hit:
        return True, "drift", drift_value

    return False, "", drift_value


# ============================================================
# 离散调仓与交易成本
# ============================================================

def _sell_step(
    current_shares: pd.Series,
    target_shares: pd.Series,
    exec_prices: pd.Series,
    fee_rate_buy: float,
    fee_rate_sell: float,
    trade_value_caps: pd.Series,
    lot_size_map: pd.Series,
    cash: float,
    trade_date: pd.Timestamp,
) -> tuple[pd.Series, float, list[dict]]:
    """
    先卖出，释放现金。使用 execution_price 成交。
    """
    shares = current_shares.copy().astype(int)
    trades = []

    for code in shares.index:
        cur = int(shares.get(code, 0))
        tgt = int(target_shares.get(code, 0))
        if cur <= tgt:
            continue

        px = float(exec_prices.get(code, np.nan))
        if not np.isfinite(px) or px <= 0:
            continue

        lot = int(lot_size_map.get(code, 100))
        cap_value = float(trade_value_caps.get(code, np.inf))

        desired_sell = cur - tgt
        desired_sell = round_shares_to_lot(desired_sell, lot)

        max_sell_by_cap = round_shares_to_lot(cap_value / px, lot) if np.isfinite(cap_value) else desired_sell
        sell_shares = min(desired_sell, cur, max_sell_by_cap)
        sell_shares = round_shares_to_lot(sell_shares, lot)

        if sell_shares <= 0:
            continue

        trade_value = sell_shares * px
        cost = trade_value * fee_rate_sell

        shares[code] = cur - sell_shares
        cash += trade_value - cost

        trades.append({
            "trade_date": trade_date,
            "ts_code": code,
            "side": "SELL",
            "price": px,
            "shares": int(sell_shares),
            "trade_value": float(trade_value),
            "cost": float(cost),
        })

    return shares, cash, trades


def _buy_step_iterative(
    current_shares: pd.Series,
    target_weights: Optional[pd.Series] = None,
    exec_prices: Optional[pd.Series] = None,
    val_prices: Optional[pd.Series] = None,
    fee_rate_buy: float = 0.0005,
    trade_value_caps: Optional[pd.Series] = None,
    lot_size_map: Optional[pd.Series] = None,
    cash: float = 0.0,
    trade_date: Optional[pd.Timestamp] = None,
    target_values: Optional[pd.Series] = None,
    residual_fill: bool = True,
) -> tuple[pd.Series, float, list[dict]]:
    """
    保留原函数名以兼容旧代码，但实现改为：

    1. 先按“目标金额 / 成交价”一步到位计算目标买入股数；
    2. 若受手续费、整手、成交额上限影响仍有剩余现金，再做少量残差补齐。

    推荐传入 target_values（由信号日冻结的目标金额）。
    若 target_values 为 None，则回退到旧口径：使用 val_prices 估组合价值，再由 target_weights 生成目标金额。
    """
    if trade_date is None:
        trade_date = pd.NaT
    if exec_prices is None:
        raise ValueError("exec_prices must not be None")

    shares = current_shares.copy().astype(int)
    idx = shares.index.union(exec_prices.index)
    shares = shares.reindex(idx).fillna(0).astype(int)
    exec_prices = exec_prices.reindex(idx)

    if trade_value_caps is None:
        trade_value_caps = pd.Series(np.inf, index=idx, dtype=float)
    else:
        trade_value_caps = trade_value_caps.reindex(idx).fillna(np.inf).astype(float)

    if lot_size_map is None:
        lot_size_map = pd.Series(100, index=idx, dtype=int)
    else:
        lot_size_map = lot_size_map.reindex(idx).fillna(100).astype(int)

    if target_values is None:
        if target_weights is None or val_prices is None:
            raise ValueError("either target_values or (target_weights and val_prices) must be provided")
        val_prices = val_prices.reindex(idx)
        portfolio_value = calc_portfolio_value(shares, val_prices, cash)
        target_weights = target_weights.reindex(idx).fillna(0.0).clip(lower=0.0)
        if float(target_weights.sum()) > 1.0 + 1e-10:
            raise ValueError("target_weights sum must be <= 1")
        target_values = target_weights * float(portfolio_value)
    else:
        target_values = target_values.reindex(idx).fillna(0.0).clip(lower=0.0)

    trades: list[dict] = []
    remaining_cap = trade_value_caps.copy()

    # ------------------------------
    # 第一步：直接计算目标买入股数
    # ------------------------------
    current_exec_value = (shares * exec_prices.reindex(idx)).fillna(0.0)
    desired_buy_value = (target_values - current_exec_value).clip(lower=0.0)
    buy_order = desired_buy_value.sort_values(ascending=False).index.tolist()

    tentative_shares = pd.Series(0, index=idx, dtype=int)
    tentative_costs = pd.Series(0.0, index=idx, dtype=float)

    for code in buy_order:
        px = float(exec_prices.get(code, np.nan))
        if not np.isfinite(px) or px <= 0:
            continue

        lot = int(lot_size_map.get(code, 100))
        cap_val = float(remaining_cap.get(code, np.inf))
        max_by_cap = round_shares_to_lot(cap_val / px, lot) if np.isfinite(cap_val) else np.iinfo(np.int32).max
        raw_target = float(desired_buy_value.get(code, 0.0)) / px
        target_buy_shares = round_shares_to_lot(raw_target, lot)
        target_buy_shares = int(min(target_buy_shares, max_by_cap))
        if target_buy_shares <= 0:
            continue

        trade_value = target_buy_shares * px
        cost = trade_value * fee_rate_buy
        tentative_shares.loc[code] = target_buy_shares
        tentative_costs.loc[code] = trade_value + cost

    total_cash_need = float(tentative_costs.sum())

    if total_cash_need > cash and total_cash_need > 0:
        scale = float(cash / total_cash_need)
        tentative_shares.loc[:] = 0
        tentative_costs.loc[:] = 0.0

        for code in buy_order:
            px = float(exec_prices.get(code, np.nan))
            if not np.isfinite(px) or px <= 0:
                continue

            lot = int(lot_size_map.get(code, 100))
            cap_val = float(remaining_cap.get(code, np.inf))
            max_by_cap = round_shares_to_lot(cap_val / px, lot) if np.isfinite(cap_val) else np.iinfo(np.int32).max
            scaled_value = float(desired_buy_value.get(code, 0.0)) * scale
            target_buy_shares = round_shares_to_lot(scaled_value / px, lot)
            target_buy_shares = int(min(target_buy_shares, max_by_cap))
            if target_buy_shares <= 0:
                continue

            trade_value = target_buy_shares * px
            cost = trade_value * fee_rate_buy
            tentative_shares.loc[code] = target_buy_shares
            tentative_costs.loc[code] = trade_value + cost

    # 执行第一步买入
    for code in buy_order:
        buy_shares = int(tentative_shares.get(code, 0))
        if buy_shares <= 0:
            continue

        px = float(exec_prices.get(code, np.nan))
        trade_value = float(buy_shares * px)
        cost = float(trade_value * fee_rate_buy)
        cash_need = trade_value + cost
        if cash_need > cash + 1e-12:
            continue

        shares.loc[code] = int(shares.get(code, 0)) + buy_shares
        cash -= cash_need

        cap_val = float(remaining_cap.get(code, np.inf))
        if np.isfinite(cap_val):
            remaining_cap.loc[code] = max(0.0, cap_val - trade_value)

        trades.append({
            "trade_date": trade_date,
            "ts_code": code,
            "side": "BUY",
            "price": px,
            "shares": int(buy_shares),
            "trade_value": trade_value,
            "cost": cost,
        })

    # ------------------------------
    # 第二步：残差补齐（逐手）
    # ------------------------------
    if residual_fill:
        blocked = pd.Series(False, index=idx)

        while True:
            current_exec_value = (shares * exec_prices.reindex(idx)).fillna(0.0)
            residual_value = (target_values - current_exec_value).clip(lower=0.0)
            candidates = residual_value[(residual_value > 0) & (~blocked)]
            if len(candidates) == 0:
                break

            code = candidates.idxmax()
            px = float(exec_prices.get(code, np.nan))
            if not np.isfinite(px) or px <= 0:
                blocked.loc[code] = True
                continue

            lot = int(lot_size_map.get(code, 100))
            one_lot_value = lot * px
            one_lot_cost = one_lot_value * fee_rate_buy
            one_lot_cash_need = one_lot_value + one_lot_cost

            cap_val = float(remaining_cap.get(code, np.inf))
            if np.isfinite(cap_val) and cap_val < one_lot_value - 1e-12:
                blocked.loc[code] = True
                continue

            if cash < one_lot_cash_need - 1e-12:
                blocked.loc[code] = True
                continue

            # 若再买一手会明显超出目标金额，则停止该标的补齐
            if one_lot_value > float(residual_value.get(code, 0.0)) + 1e-12:
                blocked.loc[code] = True
                continue

            shares.loc[code] = int(shares.get(code, 0)) + lot
            cash -= one_lot_cash_need

            if np.isfinite(cap_val):
                remaining_cap.loc[code] = max(0.0, cap_val - one_lot_value)

            trades.append({
                "trade_date": trade_date,
                "ts_code": code,
                "side": "BUY",
                "price": px,
                "shares": int(lot),
                "trade_value": float(one_lot_value),
                "cost": float(one_lot_cost),
            })

    return shares.astype(int), float(cash), trades

def _adjust_target_for_best_effort(
    current_shares: pd.Series,
    signal_prices: pd.Series,
    signal_nav: float,
    target_weights: pd.Series,
    target_cash_weight: float,
    tradable_assets: Sequence[str],
    blocked_assets: Sequence[str],
) -> tuple[pd.Series, float]:
    """
    对不可交易资产采用“冻结原仓位”，并将其余可投资预算按 tradable 目标权重比例重新分配。
    """
    idx = current_shares.index.union(target_weights.index).union(pd.Index(tradable_assets)).union(pd.Index(blocked_assets))
    shares = current_shares.reindex(idx).fillna(0).astype(int)
    signal_prices = signal_prices.reindex(idx)
    target_weights = target_weights.reindex(idx).fillna(0.0).clip(lower=0.0)

    adjusted = pd.Series(0.0, index=idx, dtype=float)
    if signal_nav <= 0:
        return adjusted, 1.0

    blocked = [x for x in blocked_assets if x in idx]
    tradable = [x for x in tradable_assets if x in idx]

    blocked_values = (shares.reindex(blocked).astype(float) * signal_prices.reindex(blocked)).fillna(0.0)
    blocked_weights = blocked_values / float(signal_nav)
    if len(blocked) > 0:
        adjusted.loc[blocked] = blocked_weights.values

    available_budget = max(0.0, 1.0 - float(target_cash_weight) - float(blocked_weights.sum()))
    tradable_raw = target_weights.reindex(tradable).fillna(0.0).clip(lower=0.0)
    tradable_sum = float(tradable_raw.sum())

    if tradable_sum > 0 and available_budget > 0:
        adjusted.loc[tradable] = tradable_raw / tradable_sum * available_budget

    adjusted_cash_weight = max(0.0, 1.0 - float(adjusted.sum()))
    return adjusted, float(adjusted_cash_weight)


def rebalance_to_target_weights(
    current_shares: pd.Series,
    cash: float,
    target_weights: pd.Series,
    exec_prices: pd.Series,
    val_prices: Optional[pd.Series] = None,
    amount_series: Optional[pd.Series] = None,
    fee_rate_buy: float = 0.0005,
    fee_rate_sell: float = 0.0005,
    lot_size: int | dict[str, int] = 100,
    max_trade_amount_ratio: Optional[float] = None,
    amount_unit_scale: float = 1000.0,
    trade_date: Optional[pd.Timestamp] = None,
    signal_nav: Optional[float] = None,
    target_cash_weight: Optional[float] = None,
    weight_calc_prices: Optional[pd.Series] = None,
) -> tuple[pd.Series, float, pd.DataFrame, pd.Series, float]:
    """
    将当前持仓调仓到目标权重附近。

    当前推荐口径：
    - 目标金额由 signal_nav（通常为信号日收盘组合净值）冻结；
    - 交易股数按 trade_date 的 exec_prices 一次性换算；
    - 若受手续费 / 整手 / 成交额上限影响，再做少量残差补齐。

    兼容旧口径：
    - 若 signal_nav is None，则退回到基于 val_prices 的旧逻辑。
    """
    if trade_date is None:
        trade_date = pd.NaT

    idx = current_shares.index.union(target_weights.index).union(exec_prices.index)
    if val_prices is not None:
        idx = idx.union(val_prices.index)
    if weight_calc_prices is not None:
        idx = idx.union(weight_calc_prices.index)

    shares = current_shares.reindex(idx).fillna(0).astype(int)
    target_weights = target_weights.reindex(idx).fillna(0.0).clip(lower=0.0)
    if float(target_weights.sum()) > 1.0 + 1e-10:
        raise ValueError("target_weights sum must be <= 1")

    exec_prices = exec_prices.reindex(idx)
    if val_prices is not None:
        val_prices = val_prices.reindex(idx)

    if weight_calc_prices is None:
        weight_calc_prices = exec_prices.copy()
        if val_prices is not None:
            weight_calc_prices = weight_calc_prices.where(np.isfinite(weight_calc_prices) & (weight_calc_prices > 0), val_prices)
    else:
        weight_calc_prices = weight_calc_prices.reindex(idx)

    if amount_series is not None:
        amount_series = amount_series.reindex(idx)

    if isinstance(lot_size, dict):
        lot_size_map = pd.Series(lot_size).reindex(idx).fillna(100).astype(int)
    else:
        lot_size_map = pd.Series(int(lot_size), index=idx)

    trade_value_caps = get_trade_value_cap(
        amount_series=amount_series,
        max_trade_amount_ratio=max_trade_amount_ratio,
        amount_unit_scale=amount_unit_scale,
    ).reindex(idx)

    if max_trade_amount_ratio is not None:
        trade_value_caps = trade_value_caps.fillna(0.0)
    else:
        trade_value_caps = trade_value_caps.fillna(np.inf)

    if signal_nav is None:
        if val_prices is None:
            raise ValueError("val_prices must not be None when signal_nav is None")
        portfolio_value = calc_portfolio_value(shares, val_prices, cash)
        signal_nav = float(portfolio_value)

    target_cash_weight = max(0.0, float(1.0 - target_weights.sum())) if target_cash_weight is None else float(target_cash_weight)
    target_values = target_weights * float(signal_nav)

    # 先按目标金额换算目标股数，用于卖出阶段
    target_shares = pd.Series(0, index=idx, dtype=int)
    for code in idx:
        px = float(exec_prices.get(code, np.nan))
        if not np.isfinite(px) or px <= 0:
            continue
        lot = int(lot_size_map.get(code, 100))
        raw_shares = float(target_values.get(code, 0.0)) / px
        target_shares.loc[code] = round_shares_to_lot(raw_shares, lot)

    shares_after_sell, cash_after_sell, sell_trades = _sell_step(
        current_shares=shares,
        target_shares=target_shares,
        exec_prices=exec_prices,
        fee_rate_buy=fee_rate_buy,
        fee_rate_sell=fee_rate_sell,
        trade_value_caps=trade_value_caps,
        lot_size_map=lot_size_map,
        cash=cash,
        trade_date=trade_date,
    )

    shares_after_buy, cash_after_buy, buy_trades = _buy_step_iterative(
        current_shares=shares_after_sell,
        exec_prices=exec_prices,
        fee_rate_buy=fee_rate_buy,
        trade_value_caps=trade_value_caps,
        lot_size_map=lot_size_map,
        cash=cash_after_sell,
        trade_date=trade_date,
        target_values=target_values,
        residual_fill=True,
    )

    trades = sell_trades + buy_trades
    trades_df = pd.DataFrame(trades)

    actual_weights = calc_actual_weights(shares_after_buy, weight_calc_prices, cash_after_buy).reindex(idx).fillna(0.0)
    actual_cash_weight = calc_cash_weight(shares_after_buy, weight_calc_prices, cash_after_buy)
    return shares_after_buy.astype(int), float(cash_after_buy), trades_df, actual_weights, float(actual_cash_weight)

def calc_nav_from_returns(returns: pd.Series, initial_nav: float = 1.0) -> pd.Series:
    nav = (1.0 + returns.fillna(0.0)).cumprod() * initial_nav
    nav.name = "nav"
    return nav


def calc_drawdown(nav: pd.Series) -> pd.Series:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    dd.name = "drawdown"
    return dd


def calc_total_return(nav: pd.Series) -> float:
    if len(nav) == 0:
        return np.nan
    return float(nav.iloc[-1] / nav.iloc[0] - 1.0)


def calc_annual_return(nav: pd.Series, annualization: int = 252) -> float:
    if len(nav) < 2:
        return np.nan
    n = len(nav) - 1
    return float((nav.iloc[-1] / nav.iloc[0]) ** (annualization / n) - 1.0)


def calc_annual_volatility(returns: pd.Series, annualization: int = 252) -> float:
    if len(returns.dropna()) < 2:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(annualization))


def calc_excess_return(nav: pd.Series, annualization: int = 252, risk_free_rate: float = 0.0) -> float:
    if len(nav) < 2:
        return np.nan
    n = len(nav) - 1
    return float((nav.iloc[-1] / nav.iloc[0]) ** (annualization / n) - 1.0 - risk_free_rate)


def calc_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualization: int = 252) -> float:
    ann_ret = float(returns.mean() * annualization)
    ann_vol = calc_annual_volatility(returns, annualization=annualization)
    if not np.isfinite(ann_vol) or ann_vol == 0:
        return np.nan
    return (ann_ret - risk_free_rate) / ann_vol


def calc_max_drawdown(nav: pd.Series) -> float:
    dd = calc_drawdown(nav)
    if len(dd) == 0:
        return np.nan
    return float(dd.min())


def calc_calmar_ratio(nav: pd.Series, annualization: int = 252) -> float:
    max_drawdown = calc_max_drawdown(nav)
    if not np.isfinite(max_drawdown) or len(nav) < 2 or max_drawdown == 0:
        return np.nan
    n = len(nav) - 1
    return float(((nav.iloc[-1] / nav.iloc[0]) ** (annualization / n) - 1.0) / abs(max_drawdown))


def performance_summary(
    nav: pd.Series,
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization: int = 252,
) -> pd.Series:
    return pd.Series({
        "total_return": calc_total_return(nav),
        "annual_return": calc_annual_return(nav, annualization=annualization),
        "excess_return": calc_excess_return(nav, annualization=annualization, risk_free_rate=risk_free_rate),
        "annual_volatility": calc_annual_volatility(returns, annualization=annualization),
        "sharpe_ratio": calc_sharpe_ratio(returns, risk_free_rate=risk_free_rate, annualization=annualization),
        "max_drawdown": calc_max_drawdown(nav),
        "calmar_ratio": calc_calmar_ratio(nav, annualization=annualization),
    })


# ============================================================
# 主回测函数
# ============================================================

def simulate_dual_momentum_backtest(
    market: dict[str, pd.DataFrame],
    initial_cash: float = 1_000_000.0,
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
    dm_prepare_kwargs: Optional[dict[str, Any]] = None,
    dm_snapshot_kwargs: Optional[dict[str, Any]] = None,
    risk_free_rate: float = 0.0,
    annualization: int = 252,
    include_cash_column: bool = True,
    cash_column_name: str = "CASH",
    store_snapshot_details: bool = False,
    execution_policy: str = "best_effort",
) -> dict[str, object]:
    """
    双动量回测主入口。

    execution_policy:
    - "best_effort": 对不可交易资产冻结原仓位，对可交易资产尽力成交；当日结束清空 pending_signal
    - "strict": 任一参与资产不可交易则整单顺延，pending_signal 保留
    """
    dm_prepare_kwargs = dm_prepare_kwargs or {}
    dm_snapshot_kwargs = dm_snapshot_kwargs or {}

    execution_policy = str(execution_policy).lower()
    if execution_policy not in {"best_effort", "strict"}:
        raise ValueError("execution_policy must be one of {'best_effort', 'strict'}")

    market = ensure_same_index_columns(market)
    close_px = ensure_datetime_index(market["close"])

    val_px = close_px.ffill(limit=valuation_ffill_limit)
    exec_px = get_execution_price_matrix(market, execution_price_type=execution_price_type)
    amount_px = market.get("amount", None)

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
    snapshot_logs = []
    snapshot_details: dict[pd.Timestamp, pd.DataFrame] = {}

    prev_nav = initial_cash

    for i, dt in enumerate(dates):
        val_today = val_px.loc[dt]
        exec_today = exec_px.loc[dt]
        amount_today = amount_px.loc[dt] if amount_px is not None else None

        # --------------------------------------------------
        # 1. 如果昨天有信号，今天执行调仓
        # --------------------------------------------------
        if pending_signal is not None:
            original_target_weights = pending_signal["target_weights"].reindex(codes).fillna(0.0)
            signal_date = pending_signal["signal_date"]
            signal_nav = float(pending_signal["signal_nav"])
            signal_prices = pending_signal["signal_prices"].reindex(codes)
            reason = pending_signal["reason"]
            drift_value = pending_signal["drift_value"]
            original_target_cash_weight = float(pending_signal.get("cash_weight", max(0.0, 1.0 - float(original_target_weights.sum()))))

            involved_assets = shares[shares > 0].index.union(original_target_weights[original_target_weights > 0].index)
            involved_exec_px = exec_today.reindex(involved_assets)
            valid_mask = np.isfinite(involved_exec_px) & (involved_exec_px > 0)
            tradable_assets = involved_assets[valid_mask].tolist()
            blocked_assets = involved_assets[~valid_mask].tolist()

            if execution_policy == "strict" and len(blocked_assets) > 0:
                rebalance_logs.append({
                    "signal_date": signal_date,
                    "trade_date": dt,
                    "reason": reason,
                    "drift_value": drift_value,
                    "turnover": 0.0,
                    "target_cash_weight": original_target_cash_weight,
                    "cash_weight_before_trade": calc_cash_weight(shares, signal_prices, cash),
                    "cash_weight_after_trade": calc_cash_weight(shares, signal_prices, cash),
                    "cash_after_trade": cash,
                    "traded": 0,
                    "trade_count": 0,
                    "execution_policy": execution_policy,
                    "blocked_assets": ",".join(blocked_assets),
                    "partial_execution": 0,
                    "signal_cleared": 0,
                })
            else:
                if execution_policy == "best_effort" and len(blocked_assets) > 0:
                    exec_target_weights, exec_target_cash_weight = _adjust_target_for_best_effort(
                        current_shares=shares,
                        signal_prices=signal_prices,
                        signal_nav=signal_nav,
                        target_weights=original_target_weights,
                        target_cash_weight=original_target_cash_weight,
                        tradable_assets=tradable_assets,
                        blocked_assets=blocked_assets,
                    )
                    partial_execution = 1
                else:
                    exec_target_weights = original_target_weights.copy()
                    exec_target_cash_weight = original_target_cash_weight
                    partial_execution = 0

                before_weights = calc_actual_weights(shares, signal_prices, cash).reindex(codes).fillna(0.0)
                before_cash_weight = calc_cash_weight(shares, signal_prices, cash)

                new_shares, new_cash, trades_df, after_weights, after_cash_weight = rebalance_to_target_weights(
                    current_shares=shares,
                    cash=cash,
                    target_weights=exec_target_weights,
                    exec_prices=exec_today,
                    val_prices=val_today,
                    amount_series=amount_today,
                    fee_rate_buy=fee_rate_buy,
                    fee_rate_sell=fee_rate_sell,
                    lot_size=lot_size,
                    max_trade_amount_ratio=max_trade_amount_ratio,
                    amount_unit_scale=amount_unit_scale,
                    trade_date=dt,
                    signal_nav=signal_nav,
                    target_cash_weight=exec_target_cash_weight,
                    weight_calc_prices=signal_prices,
                )

                shares = new_shares
                cash = new_cash

                if len(trades_df) > 0:
                    trade_records.append(trades_df)

                turnover = calc_turnover_from_weights(before_weights, after_weights)

                rebalance_logs.append({
                    "signal_date": signal_date,
                    "trade_date": dt,
                    "reason": reason,
                    "drift_value": drift_value,
                    "turnover": turnover,
                    "target_cash_weight": exec_target_cash_weight,
                    "cash_weight_before_trade": before_cash_weight,
                    "cash_weight_after_trade": after_cash_weight,
                    "cash_after_trade": cash,
                    "traded": int(len(trades_df) > 0),
                    "trade_count": int(len(trades_df)),
                    "execution_policy": execution_policy,
                    "blocked_assets": ",".join(blocked_assets),
                    "partial_execution": partial_execution,
                    "signal_cleared": 1,
                })

                pending_signal = None

        # --------------------------------------------------
        # 2. 当日收盘后记录净值 / 持仓 / 实际权重
        # --------------------------------------------------
        nav_today = calc_portfolio_value(shares, val_today, cash)
        ret_today = nav_today / prev_nav - 1.0 if i > 0 else 0.0
        cash_weight_today = calc_cash_weight(shares, val_today, cash)

        nav_records.append({
            "trade_date": dt,
            "nav": nav_today,
            "cash": cash,
            "cash_weight": cash_weight_today,
        })
        return_records.append({"trade_date": dt, "return": ret_today})

        pos_row = pd.DataFrame([shares.values], index=[dt], columns=codes)

        actual_weights_today = calc_actual_weights(shares, val_today, cash).reindex(codes).fillna(0.0)
        if include_cash_column:
            wt_cols = list(codes) + [cash_column_name]
            wt_vals = list(actual_weights_today.values) + [cash_weight_today]
            wt_row = pd.DataFrame([wt_vals], index=[dt], columns=wt_cols)
        else:
            wt_row = pd.DataFrame([actual_weights_today.values], index=[dt], columns=codes)

        position_records.append(pos_row)
        weight_records.append(wt_row)

        prev_nav = nav_today

        # --------------------------------------------------
        # 3. 用截至今日收盘的数据生成“明天执行”的调仓信号
        # --------------------------------------------------
        next_dt = next_trading_date(dates, dt)
        if next_dt is None:
            continue

        signal_snapshot = compute_target_weights_on_date(
            close_price_df=close_px,
            signal_date=dt,
            dm_prepare_kwargs=dm_prepare_kwargs,
            dm_snapshot_kwargs=dm_snapshot_kwargs,
        )

        if not bool(signal_snapshot.get("signal_valid", True)):
            continue

        target_weights_today = signal_snapshot["target_weights"].reindex(codes).fillna(0.0)
        target_cash_weight_today = float(signal_snapshot.get("cash_weight", max(0.0, 1.0 - float(target_weights_today.sum()))))

        if include_cash_column:
            tw_cols = list(codes) + [cash_column_name]
            tw_vals = list(target_weights_today.values) + [target_cash_weight_today]
            tw_row = pd.DataFrame([tw_vals], index=[dt], columns=tw_cols)
        else:
            tw_row = pd.DataFrame([target_weights_today.values], index=[dt], columns=codes)
        target_weight_records.append(tw_row)

        snapshot_logs.append({
            "signal_date": dt,
            "selected_assets": ",".join(signal_snapshot.get("selected_assets", [])),
            "defensive_assets_used": ",".join(signal_snapshot.get("defensive_assets_used", [])),
            "selected_count": len(signal_snapshot.get("selected_assets", [])),
            "target_cash_weight": target_cash_weight_today,
            "market_regime_on": bool(signal_snapshot.get("market_regime_on", True)),
        })

        if store_snapshot_details:
            detail = signal_snapshot.get("detail", None)
            if isinstance(detail, pd.DataFrame):
                snapshot_details[pd.Timestamp(dt)] = detail.copy()

        current_actual_weights = actual_weights_today.reindex(codes).fillna(0.0)

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
                    "signal_nav": float(nav_today),
                    "signal_prices": val_today.copy(),
                    "target_weights": target_weights_today,
                    "cash_weight": target_cash_weight_today,
                    "reason": reason,
                    "drift_value": drift_value,
                }

    nav_df = pd.DataFrame(nav_records).set_index("trade_date")
    returns = pd.DataFrame(return_records).set_index("trade_date")["return"]
    positions_df = pd.concat(position_records, axis=0).sort_index() if position_records else pd.DataFrame(columns=codes)
    weights_df = pd.concat(weight_records, axis=0).sort_index() if weight_records else pd.DataFrame(columns=list(codes) + ([cash_column_name] if include_cash_column else []))
    target_weights_df = pd.concat(target_weight_records, axis=0).sort_index() if target_weight_records else pd.DataFrame(columns=list(codes) + ([cash_column_name] if include_cash_column else []))

    if trade_records:
        trades_df = pd.concat(trade_records, axis=0, ignore_index=True)
    else:
        trades_df = pd.DataFrame(columns=["trade_date", "ts_code", "side", "price", "shares", "trade_value", "cost"])

    rebalance_log_df = pd.DataFrame(rebalance_logs)
    if len(rebalance_log_df) > 0:
        rebalance_log_df["signal_date"] = pd.to_datetime(rebalance_log_df["signal_date"])
        rebalance_log_df["trade_date"] = pd.to_datetime(rebalance_log_df["trade_date"])

    snapshot_log_df = pd.DataFrame(snapshot_logs)
    if len(snapshot_log_df) > 0:
        snapshot_log_df["signal_date"] = pd.to_datetime(snapshot_log_df["signal_date"])

    summary = performance_summary(
        nav=nav_df["nav"],
        returns=returns,
        risk_free_rate=risk_free_rate,
        annualization=annualization,
    )

    result = {
        "nav_df": nav_df,
        "returns": returns,
        "positions_df": positions_df,
        "weights_df": weights_df,
        "target_weights_df": target_weights_df,
        "trades_df": trades_df,
        "rebalance_log_df": rebalance_log_df,
        "snapshot_log_df": snapshot_log_df,
        "summary": summary,
    }
    if store_snapshot_details:
        result["snapshot_details"] = snapshot_details

    return result

def simulate_dual_momentum_backtest_from_long_data(
    data: pd.DataFrame,
    fields: Sequence[str] = ("open", "high", "low", "close", "amount"),
    date_col: str = "trade_date",
    code_col: str = "ts_code",
    date_format: Optional[str] = "%Y%m%d",
    **backtest_kwargs,
) -> dict[str, object]:
    """
    从长表一站式构建市场矩阵并运行双动量回测。
    """
    market = build_market_matrices(
        data=data,
        fields=fields,
        date_col=date_col,
        code_col=code_col,
        date_format=date_format,
    )
    return simulate_dual_momentum_backtest(
        market=market,
        **backtest_kwargs,
    )


# ============================================================
# 示例
# ============================================================

if __name__ == "__main__":
    dates = pd.date_range("2024-01-02", periods=260, freq="B")
    codes = ["510300.SH", "512100.SH", "511010.SH"]

    rng = np.random.default_rng(42)

    def make_price(start, drift=0.0, vol=0.01):
        r = rng.normal(drift, vol, len(dates))
        px = start * np.exp(np.cumsum(r))
        return pd.Series(px, index=dates)

    close = pd.DataFrame({
        "510300.SH": make_price(3.5, drift=0.0006, vol=0.012),
        "512100.SH": make_price(1.2, drift=0.0009, vol=0.018),
        "511010.SH": make_price(110.0, drift=0.0001, vol=0.003),
    }, index=dates)

    open_ = close * (1 + rng.normal(0, 0.002, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, close.shape)))
    amount = pd.DataFrame(2e8, index=dates, columns=codes)

    open_.loc[dates[100], "510300.SH"] = np.nan
    close.loc[dates[100], "510300.SH"] = np.nan

    market = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "amount": amount,
    }

    result = simulate_dual_momentum_backtest(
        market=market,
        initial_cash=1_000_000.0,
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
        dm_prepare_kwargs={
            "calendar": close.index,
            "ffill": True,
            "ffill_limit": 5,
            "min_non_na_ratio": 0.8,
            "drop_all_na_dates": True,
        },
        dm_snapshot_kwargs={
            "candidate_assets": ["510300.SH", "512100.SH"],
            "defensive_assets": ["511010.SH"],
            "abs_lookback": 60,
            "abs_threshold": 0.0,
            "rel_lookbacks": [20, 60, 120],
            "rel_weights": [0.2, 0.3, 0.5],
            "top_k": 1,
            "weighting": "equal",
            "fill_unallocated_to_defensive": True,
            "min_history": 120,
        },
        include_cash_column=True,
        cash_column_name="CASH",
    )

    print(result["summary"])
    print(result["nav_df"].tail())
    print(result["trades_df"].head())
