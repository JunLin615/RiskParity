"""
backtest.py

纯函数版回测函数库
依赖：
- pandas
- numpy
- risk_parity.py

特点：
1. 不和 notebook、数据库、实盘模块耦合
2. 支持整手交易
3. long-only，无杠杆，不允许负现金
4. 支持交易成本
5. 支持固定调仓周期 + 偏离触发调仓
6. 严格避免未来函数：
   - 在 t 日收盘后用截至 t 的历史数据生成信号
   - 在下一交易日 t+1 按指定成交价口径成交
7. 支持成交额占比限制，避免单日成交额占比过高
8. 区分估值价格（有限前向填充）与交易价格（不填充，遇停牌顺延）

说明：
- 默认成交价 execution_price_type = "avg"
- 这里的 avg 默认定义为 (open + high + low + close) / 4
- 如果你想改成别的“均价”定义，可以自行改 get_execution_price_matrix
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from risk_parity import (
    calc_risk_parity_weights_from_prices,
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
    """
    union_idx = old_weights.index.union(new_weights.index)
    ow = old_weights.reindex(union_idx).fillna(0.0)
    nw = new_weights.reindex(union_idx).fillna(0.0)
    return float((nw - ow).abs().sum())


def get_trade_value_cap(
    amount_series: Optional[pd.Series],
    max_trade_amount_ratio: Optional[float],
    amount_unit_scale: float = 1000.0,#tushare数据的单位似乎是1000
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
    lookback_window: int = 120,
    rp_prepare_kwargs: Optional[dict] = None,
    rp_weight_kwargs: Optional[dict] = None,
) -> pd.Series:
    """
    在 signal_date 当日收盘后，用截至 signal_date 的历史价格计算目标风险平价权重。
    """
    rp_prepare_kwargs = rp_prepare_kwargs or {}
    rp_weight_kwargs = rp_weight_kwargs or {}

    px = ensure_datetime_index(close_price_df)
    hist = px.loc[:signal_date].copy()
    hist = hist.iloc[-lookback_window:].copy()

    prepared = prepare_price_matrix(hist, **rp_prepare_kwargs)
    prepared = prepared.dropna(how="any")

    if prepared.shape[0] < 2 or prepared.shape[1] < 1:
        return pd.Series(dtype=float)

    w = calc_risk_parity_weights_from_prices(prepared, **rp_weight_kwargs)
    w = w.reindex(px.columns).fillna(0.0)
    return normalize_weights(w)


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
    target_weights: pd.Series,
    exec_prices: pd.Series,
    val_prices: pd.Series,
    fee_rate_buy: float,
    trade_value_caps: pd.Series,
    lot_size_map: pd.Series,
    cash: float,
    trade_date: pd.Timestamp,
) -> tuple[pd.Series, float, list[dict]]:
    """
    迭代买入：
    按 valuation_price 评估当前真实权重，按 execution_price 买入扣除现金。
    """
    shares = current_shares.copy().astype(int)
    trades = []

    remaining_cap = trade_value_caps.copy().astype(float)
    target_weights = normalize_weights(target_weights.reindex(shares.index).fillna(0.0))
    blocked = pd.Series(False, index=shares.index)

    while True:
        # 权重偏离估算基于估值价（valuation price）
        portfolio_value = calc_portfolio_value(shares, val_prices, cash)
        if portfolio_value <= 0:
            break

        actual_weights = calc_actual_weights(shares, val_prices, cash=cash).reindex(shares.index).fillna(0.0)
        gap = (target_weights - actual_weights).fillna(0.0)

        candidates = gap[(gap > 0) & (~blocked)]
        if len(candidates) == 0:
            break

        code = candidates.idxmax()
        
        # 实际交易扣除基于交易价（execution price）
        px = float(exec_prices.get(code, np.nan))
        if not np.isfinite(px) or px <= 0:
            blocked.loc[code] = True
            continue

        lot = int(lot_size_map.get(code, 100))
        one_lot_shares = lot
        one_lot_value = one_lot_shares * px
        one_lot_cost = one_lot_value * fee_rate_buy
        one_lot_cash_need = one_lot_value + one_lot_cost

        cap_val = float(remaining_cap.get(code, np.inf))
        if np.isfinite(cap_val) and cap_val < one_lot_value:
            blocked.loc[code] = True
            continue

        if cash < one_lot_cash_need:
            blocked.loc[code] = True
            continue

        # 成交
        shares.loc[code] = int(shares.get(code, 0)) + one_lot_shares
        cash -= one_lot_cash_need

        if np.isfinite(cap_val):
            remaining_cap.loc[code] = cap_val - one_lot_value

        trades.append({
            "trade_date": trade_date,
            "ts_code": code,
            "side": "BUY",
            "price": px,
            "shares": int(one_lot_shares),
            "trade_value": float(one_lot_value),
            "cost": float(one_lot_cost),
        })

    return shares, cash, trades


def rebalance_to_target_weights(
    current_shares: pd.Series,
    cash: float,
    target_weights: pd.Series,
    exec_prices: pd.Series,
    val_prices: pd.Series,
    amount_series: Optional[pd.Series] = None,
    fee_rate_buy: float = 0.0005,
    fee_rate_sell: float = 0.0005,
    lot_size: int | dict[str, int] = 100,
    max_trade_amount_ratio: Optional[float] = None,
    amount_unit_scale: float = 1000.0,
    trade_date: Optional[pd.Timestamp] = None,
) -> tuple[pd.Series, float, pd.DataFrame, pd.Series]:
    """
    将当前持仓调仓到目标权重附近。
    """
    if trade_date is None:
        trade_date = pd.NaT

    idx = current_shares.index.union(target_weights.index).union(exec_prices.index)
    shares = current_shares.reindex(idx).fillna(0).astype(int)
    target_weights = normalize_weights(target_weights.reindex(idx).fillna(0.0))
    exec_prices = exec_prices.reindex(idx)
    val_prices = val_prices.reindex(idx)
    
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

    # 用【估值价】估算组合整体目标金额
    portfolio_value = calc_portfolio_value(shares, val_prices, cash)
    target_value = target_weights * portfolio_value

    # 用【交易价】反推需要挂单的目标股数
    target_shares = pd.Series(0, index=idx, dtype=int)
    for code in idx:
        px = float(exec_prices.get(code, np.nan))
        if not np.isfinite(px) or px <= 0:
            continue
        lot = int(lot_size_map.get(code, 100))
        raw_shares = float(target_value.get(code, 0.0)) / px
        target_shares.loc[code] = round_shares_to_lot(raw_shares, lot)

    # 先卖后买
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

    # 买入阶段按目标权重逐手逼近
    shares_after_buy, cash_after_buy, buy_trades = _buy_step_iterative(
        current_shares=shares_after_sell,
        target_weights=target_weights,
        exec_prices=exec_prices,
        val_prices=val_prices,
        fee_rate_buy=fee_rate_buy,
        trade_value_caps=trade_value_caps,
        lot_size_map=lot_size_map,
        cash=cash_after_sell,
        trade_date=trade_date,
    )

    trades = sell_trades + buy_trades
    trades_df = pd.DataFrame(trades)

    # 交易完成后的权重按估值价展示
    actual_weights = calc_actual_weights(shares_after_buy, val_prices, cash_after_buy).reindex(idx).fillna(0.0)
    return shares_after_buy.astype(int), float(cash_after_buy), trades_df, actual_weights


# ============================================================
# 绩效统计
# ============================================================
# 保持原逻辑不变，为节约篇幅保留了方法签名
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
    if len(nav) == 0: return np.nan
    return float(nav.iloc[-1] / nav.iloc[0] - 1.0)

def calc_annual_return(nav: pd.Series, annualization: int = 252) -> float:
    if len(nav) < 2: return np.nan
    n = len(nav) - 1
    return float((nav.iloc[-1] / nav.iloc[0]) ** (annualization / n) - 1.0)

def calc_annual_volatility(returns: pd.Series, annualization: int = 252) -> float:
    if len(returns.dropna()) < 2: return np.nan
    return float(returns.std(ddof=1) * np.sqrt(annualization))
def calc_excess_return(nav: pd.Series, annualization: int = 252, risk_free_rate: float = 0.0) -> float:
    if len(nav) < 2: return np.nan
    n = len(nav) - 1
    return float((nav.iloc[-1] / nav.iloc[0]) ** (annualization / n) - 1.0-risk_free_rate)

def calc_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualization: int = 252) -> float:
    ann_ret = float(returns.mean() * annualization)
    ann_vol = calc_annual_volatility(returns, annualization=annualization)
    if not np.isfinite(ann_vol) or ann_vol == 0: return np.nan
    return (ann_ret - risk_free_rate) / ann_vol

def calc_max_drawdown(nav: pd.Series) -> float:
    dd = calc_drawdown(nav)
    if len(dd) == 0: return np.nan
    return float(dd.min())
def calc_calmar_ratio(nav: pd.Series, annualization: int = 252) -> float:
    max_drawdown = calc_max_drawdown(nav)
    if not np.isfinite(max_drawdown) or len(nav) < 2: return np.nan
    n = len(nav) - 1
    return float(((nav.iloc[-1] / nav.iloc[0]) ** (annualization / n) - 1.0)/max_drawdown)
def performance_summary(nav: pd.Series, returns: pd.Series, risk_free_rate: float = 0.0, annualization: int = 252) -> pd.Series:
    return pd.Series({
        "total_return": calc_total_return(nav),
        "annual_return": calc_annual_return(nav, annualization=annualization),
        "excess_return":calc_excess_return(nav, annualization=annualization, risk_free_rate=risk_free_rate),
        "annual_volatility": calc_annual_volatility(returns, annualization=annualization),
        "sharpe_ratio": calc_sharpe_ratio(returns, risk_free_rate=risk_free_rate, annualization=annualization),
        "max_drawdown": calc_max_drawdown(nav),
        "calmar_ratio": calc_calmar_ratio(nav, annualization=annualization),
    })
def calc_asset_correlation_matrix(
    prices: pd.DataFrame,
    return_type: str = "log",
    method: str = "pearson"
) -> pd.DataFrame:
    """
    计算资产收益率的相关性矩阵。
    
    参数：
    prices: 资产价格矩阵，index 为时间，columns 为资产代码
    return_type: "simple" (简单收益率) 或 "log" (对数收益率，资产配置通常更推荐使用 log)
    method: "pearson" (线性相关), "kendall", "spearman" (秩相关)
    """
    # 必须先进行前向填充，避免不同资产因停牌日不同导致计算 pct_change 时产生错位空洞
    px = prices.ffill() 
    
    if return_type == "log":
        rets = np.log(px / px.shift(1))
    else:
        rets = px.pct_change()
        
    return rets.corr(method=method)


def calc_avg_pairwise_correlation(corr_matrix: pd.DataFrame) -> float:
    """
    计算平均两两相关系数（剔除对角线上的自身相关性 1.0）。
    用于评估整个资产池的天然分散度。数值越低，说明资产互补性越强。
    """
    mat = corr_matrix.values
    n = mat.shape[0]
    if n < 2:
        return np.nan
    
    # 提取上三角矩阵中的非对角线元素并求均值
    upper_tri_elements = mat[np.triu_indices(n, k=1)]
    return float(np.nanmean(upper_tri_elements))

def calc_covariance_matrix(
    prices: pd.DataFrame,
    return_type: str = "log",
    annualization: int = 252
    ) -> pd.DataFrame:
    """
    计算基于历史价格的年化协方差矩阵。
    """
    px = prices.ffill()
    if return_type == "log":
        rets = np.log(px / px.shift(1))
    else:
        rets = px.pct_change()
    
    # 计算协方差并年化
    return rets.cov() * annualization


def calc_risk_contribution(
    weights: pd.Series,
    cov_matrix: pd.DataFrame
    ) -> pd.Series:
    """
    给定当前资产权重和协方差矩阵，计算各资产的百分比风险贡献度 (%RC)。
    所有资产的 %RC 之和理论上等于 1.0 (或 100%)。
    """
    # 提取共有标的并统一顺序
    idx = weights.index.intersection(cov_matrix.index)
    if len(idx) == 0:
        return pd.Series(dtype=float)

    w = weights.reindex(idx).fillna(0.0).values
    cov = cov_matrix.loc[idx, idx].values

    # 1. 组合总方差与波动率
    port_var = float(w.T @ cov @ w)
    if port_var <= 0:
        return pd.Series(0.0, index=idx)
    port_vol = np.sqrt(port_var)

    # 2. 边际风险贡献 (MRC) = (Sigma * w) / port_vol
    mrc = (cov @ w) / port_vol

    # 3. 绝对风险贡献 (RC) = w * MRC
    rc = w * mrc

    # 4. 百分比风险贡献 (%RC) = RC / port_vol
    prc = rc / port_vol

    return pd.Series(prc, index=idx)


def calc_historical_risk_contributions(
    weights_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    lookback_window: int = 120,
    return_type: str = "log",
    annualization: int = 252
    ) -> pd.DataFrame:
    """
    批量计算回测期间每一天（或每一个调仓点）的实际风险贡献度。
    用于后续绘制“风险贡献堆叠面积图”，检验风险平价是否破裂。
    """
    rc_records = []
    
    # 为了避免价格数据透支未来，计算某日的风险贡献时，只能用该日之前的历史价格
    for dt, weights in weights_df.iterrows():
        # 截取历史窗口数据
        hist_px = prices_df.loc[:dt].iloc[-lookback_window:]
        
        if len(hist_px) < 2:
            rc_records.append(pd.Series(0.0, index=weights.index, name=dt))
            continue
            
        # 计算协方差矩阵
        cov_mat = calc_covariance_matrix(
            prices=hist_px, 
            return_type=return_type, 
            annualization=annualization
        )
        
        # 计算当期的风险贡献
        rc_series = calc_risk_contribution(weights, cov_mat)
        rc_series.name = dt
        rc_records.append(rc_series)
        
    return pd.DataFrame(rc_records)

# ============================================================
# 主回测函数
# ============================================================

def simulate_risk_parity_backtest(
    market: dict[str, pd.DataFrame],
    initial_cash: float = 1_000_000.0,
    lookback_window: int = 120,
    rebalance_freq: str = "Q",
    execution_price_type: str = "avg",
    valuation_ffill_limit: int = 5,   # 新增：控制估值价格前向填充的上限
    fee_rate_buy: float = 0.0005,
    fee_rate_sell: float = 0.0005,
    lot_size: int | dict[str, int] = 100,
    max_trade_amount_ratio: Optional[float] = 0.05,
    amount_unit_scale: float = 1000.0,
    use_drift_trigger: bool = False,
    drift_threshold: float = 0.05,
    rp_prepare_kwargs: Optional[dict] = None,
    rp_weight_kwargs: Optional[dict] = None,
    risk_free_rate: float = 0.0,
    annualization: int = 252,
) -> dict[str, object]:
    """
    风险平价回测主入口。
    """
    rp_prepare_kwargs = rp_prepare_kwargs or {}
    rp_weight_kwargs = rp_weight_kwargs or {}

    market = ensure_same_index_columns(market)
    close_px = ensure_datetime_index(market["close"])
    
    # 1. 估值价格：针对 close 价格做有限前向填充，用于计算市值和盘后权重
    val_px = close_px.ffill(limit=valuation_ffill_limit)
    
    # 2. 交易价格：保持原始口径不填充，遇到 NaN 视为当日停牌/不可交易
    exec_px = get_execution_price_matrix(market, execution_price_type=execution_price_type)
    amount_px = market.get("amount", None)

    dates = close_px.index
    codes = close_px.columns
    scheduled_dates = get_rebalance_dates(dates, freq=rebalance_freq)

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
        # close_today 用于历史样本收集（生成信号时）
        val_today = val_px.loc[dt]       # 估值专属
        exec_today = exec_px.loc[dt]     # 交易专属
        amount_today = amount_px.loc[dt] if amount_px is not None else None

        # --------------------------------------------------
        # 1. 如果昨天有信号，今天执行调仓
        # --------------------------------------------------
        if pending_signal is not None:
            target_weights = pending_signal["target_weights"].reindex(codes).fillna(0.0)
            signal_date = pending_signal["signal_date"]
            reason = pending_signal["reason"]
            drift_value = pending_signal["drift_value"]

            # 评估“调整目标标的”：持有需要被清/降仓的 + 目标要求买入建仓的
            involved_assets = shares[shares > 0].index.union(target_weights[target_weights > 0].index)
            involved_exec_px = exec_today.reindex(involved_assets)

            # 如果任何相关的调整目标标的当日不可交易（NaN 或 <=0），顺延至全可交易日
            invalid_px = ~(np.isfinite(involved_exec_px) & (involved_exec_px > 0))
            if invalid_px.any():
                # 停牌触发顺延，今日跳过交易动作，保留 pending_signal 等待明日重试
                pass 
            else:
                # 正常调仓
                before_weights = calc_actual_weights(shares, val_today, cash).reindex(codes).fillna(0.0)

                new_shares, new_cash, trades_df, after_weights = rebalance_to_target_weights(
                    current_shares=shares,
                    cash=cash,
                    target_weights=target_weights,
                    exec_prices=exec_today,
                    val_prices=val_today,  # 传入估值价格配合目标计算
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

                rebalance_logs.append({
                    "signal_date": signal_date,
                    "trade_date": dt,
                    "reason": reason,
                    "drift_value": drift_value,
                    "turnover": turnover,
                    "cash_after_trade": cash,
                    "traded": int(len(trades_df) > 0),
                    "trade_count": int(len(trades_df)),
                })

                pending_signal = None

        # --------------------------------------------------
        # 2. 当日收盘后记录净值 / 持仓 / 实际权重（使用估值价格 val_today）
        # --------------------------------------------------
        nav_today = calc_portfolio_value(shares, val_today, cash)
        ret_today = nav_today / prev_nav - 1.0 if i > 0 else 0.0

        nav_records.append({"trade_date": dt, "nav": nav_today, "cash": cash})
        return_records.append({"trade_date": dt, "return": ret_today})

        pos_row = pd.DataFrame([shares.values], index=[dt], columns=codes)
        wt_row = pd.DataFrame([calc_actual_weights(shares, val_today, cash).reindex(codes).fillna(0.0).values],
                              index=[dt], columns=codes)
        position_records.append(pos_row)
        weight_records.append(wt_row)

        prev_nav = nav_today

        # --------------------------------------------------
        # 3. 用截至今日收盘的数据生成“明天执行”的调仓信号
        # --------------------------------------------------
        next_dt = next_trading_date(dates, dt)
        if next_dt is None:
            continue

        hist = close_px.loc[:dt].copy()

        # 样本不足，不生成信号
        if hist.shape[0] < lookback_window:
            continue

        # 生成目标权重（依赖组件内已做了自己的 nan 数据清洗）
        target_weights_today = compute_target_weights_on_date(
            close_price_df=close_px,
            signal_date=dt,
            lookback_window=lookback_window,
            rp_prepare_kwargs=rp_prepare_kwargs,
            rp_weight_kwargs=rp_weight_kwargs,
        )

        if len(target_weights_today) == 0:
            continue

        target_weight_records.append(
            pd.DataFrame([target_weights_today.reindex(codes).fillna(0.0).values], index=[dt], columns=codes)
        )

        # 偏离测算使用最新的估值价格
        current_actual_weights = calc_actual_weights(shares, val_today, cash).reindex(codes).fillna(0.0)

        # 注意：如果 pending_signal 不为空，说明正在顺延中，此时不应覆盖老的调仓目标信号
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

    # ------------------------------------------------------
    # 汇总输出
    # ------------------------------------------------------
    nav_df = pd.DataFrame(nav_records).set_index("trade_date")
    returns = pd.DataFrame(return_records).set_index("trade_date")["return"]
    positions_df = pd.concat(position_records, axis=0).sort_index() if position_records else pd.DataFrame(columns=codes)
    weights_df = pd.concat(weight_records, axis=0).sort_index() if weight_records else pd.DataFrame(columns=codes)
    target_weights_df = pd.concat(target_weight_records, axis=0).sort_index() if target_weight_records else pd.DataFrame(columns=codes)

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

    risk_contribution_df = calc_historical_risk_contributions(
        weights_df=weights_df,
        prices_df=val_px,
        lookback_window=lookback_window,
        return_type=rp_weight_kwargs.get("return_type", "log"),
        annualization=annualization
    )

    return {
        "nav_df": nav_df,
        "returns": returns,
        "positions_df": positions_df,
        "weights_df": weights_df,
        "target_weights_df": target_weights_df,
        "trades_df": trades_df,
        "rebalance_log_df": rebalance_log_df,
        "asset_corr_matrix": corr_matrix,  # 将相关性矩阵一并输出
        "risk_contribution_df": risk_contribution_df, # <--- 输出风险贡献矩阵
        "summary": summary,
    }


# ============================================================
# 示例
# ============================================================

if __name__ == "__main__":
    dates = pd.date_range("2024-01-02", periods=260, freq="B")
    codes = ["510300.SH", "511010.SH", "518880.SH"]

    rng = np.random.default_rng(42)

    def make_price(start):
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

    # 人为制造某一天 510300.SH 的交易价缺失来测试顺延逻辑
    open_.loc[dates[100], "510300.SH"] = np.nan
    close.loc[dates[100], "510300.SH"] = np.nan

    market = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "amount": amount,
    }

    result = simulate_risk_parity_backtest(
        market=market,
        initial_cash=1_000_000.0,
        lookback_window=60,
        rebalance_freq="M",
        execution_price_type="avg",
        valuation_ffill_limit=5,    # 开启估值价格的前推平补
        fee_rate_buy=0.0005,
        fee_rate_sell=0.0005,
        lot_size=100,
        max_trade_amount_ratio=0.05,
        amount_unit_scale=1000.0,
        use_drift_trigger=True,
        drift_threshold=0.08,
        rp_prepare_kwargs={
            "calendar": close.index,
            "ffill": True,
            "ffill_limit": 5,
            "min_non_na_ratio": 0.8,
            "drop_all_na_dates": True,
        },
        rp_weight_kwargs={
            "method": "sample",
            "return_type": "log",
            "annualization": 252,
            "long_only": True,
            "drop_any_na": True,
        },
    )

    print(result["summary"])
    print(result["nav_df"].tail())
    print(result["trades_df"].head())