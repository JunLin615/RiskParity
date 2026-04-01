"""
backtest_utils.py

通用策略回测辅助函数库（纯函数版）
不依赖数据库、下单模块与具体策略逻辑。

主要提供：
1. 价格矩阵整理与对齐
2. 收益率 / 净值 / 超额表现基础计算
3. 权重矩阵回测（适用于任意策略给出的目标权重）
4. 固定比例组合基准（买入后不动 / 周期再平衡）
5. 常用绩效评价与基准对比函数

设计原则：
- 输入输出尽量使用 pandas / numpy
- 尽量兼容 market_data.py 风格的长表数据
- 尽量适配“策略信号 -> 权重 -> 回测”的常见工作流
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


# ============================================================
# 基础清洗与矩阵整理
# ============================================================


def ensure_datetime_index(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    确保对象的 index 为 DatetimeIndex，并按时间升序排列。
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out



def build_price_matrix(
    data: pd.DataFrame,
    date_col: str = "trade_date",
    code_col: str = "ts_code",
    price_col: str = "close",
    date_format: Optional[str] = "%Y%m%d",
) -> pd.DataFrame:
    """
    从长表构建价格矩阵。

    返回
    ----
    DataFrame
        index = DatetimeIndex
        columns = ts_code
        values = 价格
    """
    df = data[[date_col, code_col, price_col]].copy()
    if date_format is None:
        df[date_col] = pd.to_datetime(df[date_col])
    else:
        df[date_col] = pd.to_datetime(df[date_col], format=date_format)

    price = (
        df.pivot(index=date_col, columns=code_col, values=price_col)
        .sort_index()
        .sort_index(axis=1)
    )
    return price



def align_price_matrix(
    price_df: pd.DataFrame,
    calendar: Optional[Sequence] = None,
) -> pd.DataFrame:
    """
    按统一日期轴对齐价格矩阵。
    """
    price = ensure_datetime_index(price_df)
    if calendar is None:
        return price

    calendar_index = pd.to_datetime(pd.Index(calendar))
    calendar_index = pd.DatetimeIndex(calendar_index).sort_values().unique()
    return price.reindex(calendar_index)



def forward_fill_prices(
    price_df: pd.DataFrame,
    limit: Optional[int] = 5,
) -> pd.DataFrame:
    """
    对价格矩阵做前向填充。
    只填中间缺口，不会填序列开头的缺失。
    """
    price = ensure_datetime_index(price_df)
    return price.ffill(limit=limit)



def prepare_price_matrix(
    price_df: pd.DataFrame,
    calendar: Optional[Sequence] = None,
    ffill: bool = True,
    ffill_limit: Optional[int] = 5,
    min_non_na_ratio: float = 0.8,
    drop_all_na_dates: bool = True,
) -> pd.DataFrame:
    """
    价格矩阵预处理总入口：
    1. 统一日期轴
    2. 有限前向填充
    3. 剔除缺失过多的资产
    4. 视情况删除全空日期
    """
    price = align_price_matrix(price_df, calendar=calendar)

    if ffill:
        price = forward_fill_prices(price, limit=ffill_limit)

    if drop_all_na_dates:
        price = price.dropna(how="all")

    valid_ratio = price.notna().mean(axis=0)
    keep_cols = valid_ratio[valid_ratio >= min_non_na_ratio].index.tolist()
    return price[keep_cols]



def drop_assets_by_missing_ratio(
    df: pd.DataFrame,
    max_missing_ratio: float = 0.2,
) -> pd.DataFrame:
    """
    删除缺失比例过高的资产列。
    """
    x = df.copy()
    missing_ratio = x.isna().mean(axis=0)
    keep_cols = missing_ratio[missing_ratio <= max_missing_ratio].index.tolist()
    return x[keep_cols]



def drop_dates_with_any_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    删除仍然存在任意缺失值的日期。
    """
    return df.dropna(how="any")



def select_last_n_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    取最后 n 行。
    """
    return df.iloc[-n:].copy()


# ============================================================
# 收益率、净值与对齐
# ============================================================


def calc_simple_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算简单收益率：
        r_t = P_t / P_{t-1} - 1
    """
    price = ensure_datetime_index(price_df)
    ret = price.pct_change(fill_method=None)
    return ret.dropna(how="all")



def calc_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算对数收益率：
        r_t = ln(P_t / P_{t-1})
    """
    price = ensure_datetime_index(price_df)
    ret = np.log(price / price.shift(1))
    return ret.dropna(how="all")



def nav_to_returns(nav: pd.Series) -> pd.Series:
    """
    净值序列转简单收益率序列。
    """
    x = ensure_datetime_index(nav).astype(float)
    ret = x.pct_change(fill_method=None)
    return ret.dropna()



def calc_nav_from_returns(
    returns: pd.Series,
    initial_nav: float = 1.0,
) -> pd.Series:
    """
    从简单收益率序列构建净值序列。
    """
    ret = ensure_datetime_index(returns).fillna(0.0).astype(float)
    nav = (1.0 + ret).cumprod() * float(initial_nav)
    nav.name = "nav"
    return nav



def normalize_nav(nav: pd.Series, start_value: float = 1.0) -> pd.Series:
    """
    将净值序列归一化到给定起点。
    """
    x = ensure_datetime_index(nav).astype(float)
    if len(x) == 0:
        return x
    out = x / float(x.iloc[0]) * float(start_value)
    out.name = getattr(nav, "name", "nav")
    return out



def align_nav_series(
    left_nav: pd.Series,
    right_nav: pd.Series,
    join: str = "inner",
) -> tuple[pd.Series, pd.Series]:
    """
    对齐两条净值曲线。
    """
    a = ensure_datetime_index(left_nav).astype(float)
    b = ensure_datetime_index(right_nav).astype(float)
    idx = a.index.join(b.index, how=join)
    return a.reindex(idx), b.reindex(idx)



def annual_rate_to_period_return(
    annual_rate: float,
    annualization: int = 252,
    continuous: bool = False,
) -> float:
    """
    将年化利率转换为单期收益率。
    """
    if continuous:
        return float(np.exp(annual_rate / annualization) - 1.0)
    return float((1.0 + annual_rate) ** (1.0 / annualization) - 1.0)



def build_constant_cash_return_series(
    index: Sequence,
    annual_rate: float = 0.0,
    annualization: int = 252,
    continuous: bool = False,
    name: str = "cash_return",
) -> pd.Series:
    """
    构建常数现金收益率序列（日频）。
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(index)))
    period_ret = annual_rate_to_period_return(
        annual_rate=annual_rate,
        annualization=annualization,
        continuous=continuous,
    )
    return pd.Series(period_ret, index=idx, name=name)


# ============================================================
# 权重与再平衡辅助
# ============================================================


def normalize_weights(
    weights: Sequence[float] | pd.Series,
    target_sum: float = 1.0,
) -> np.ndarray | pd.Series:
    """
    将权重缩放到指定总和。
    """
    if isinstance(weights, pd.Series):
        total = float(weights.sum())
        if np.isclose(total, 0.0):
            raise ValueError("weights sum is zero, cannot normalize")
        return weights / total * float(target_sum)

    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if np.isclose(total, 0.0):
        raise ValueError("weights sum is zero, cannot normalize")
    return w / total * float(target_sum)



def equal_weight_series(
    asset_names: Sequence[str],
    total_weight: float = 1.0,
    name: str = "weight",
) -> pd.Series:
    """
    生成等权重序列。
    """
    n = len(asset_names)
    if n == 0:
        raise ValueError("asset_names cannot be empty")
    w = np.ones(n, dtype=float) / n * float(total_weight)
    return pd.Series(w, index=list(asset_names), name=name)



def weights_dict_to_series(
    weights: Mapping[str, float] | pd.Series,
    asset_names: Sequence[str],
    fill_missing: float = 0.0,
    normalize: bool = False,
    total_weight: float = 1.0,
    name: str = "weight",
) -> pd.Series:
    """
    将 dict / Series 权重转换为与资产列表对齐的 Series。
    """
    if isinstance(weights, pd.Series):
        s = weights.copy()
    else:
        s = pd.Series(dict(weights), dtype=float)

    out = s.reindex(list(asset_names)).fillna(fill_missing).astype(float)
    out.name = name
    if normalize:
        out = normalize_weights(out, target_sum=total_weight)
        out.name = name
    return out



def build_constant_weight_matrix(
    index: Sequence,
    asset_names: Sequence[str],
    weights: Mapping[str, float] | Sequence[float] | pd.Series,
    normalize: bool = False,
    total_weight: float = 1.0,
) -> pd.DataFrame:
    """
    构建常数目标权重矩阵。
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(index)))
    cols = list(asset_names)

    if isinstance(weights, Mapping) or isinstance(weights, pd.Series):
        w = weights_dict_to_series(
            weights=weights,
            asset_names=cols,
            fill_missing=0.0,
            normalize=normalize,
            total_weight=total_weight,
        )
    else:
        w = pd.Series(np.asarray(weights, dtype=float), index=cols, name="weight")
        if normalize:
            w = normalize_weights(w, target_sum=total_weight)
            w.name = "weight"

    mat = pd.DataFrame(np.tile(w.values, (len(idx), 1)), index=idx, columns=cols)
    return mat



def infer_rebalance_dates(
    index: Sequence,
    freq: Optional[str] = "M",
    custom_dates: Optional[Sequence] = None,
    include_first: bool = True,
) -> pd.DatetimeIndex:
    """
    生成再平衡日期。

    参数
    ----
    freq : str | None
        None / "none" 表示不做周期再平衡（只保留首日，若 include_first=True）
        支持: D/W/M/Q/Y
    custom_dates : Sequence | None
        自定义再平衡日期
    include_first : bool
        是否强制将首个交易日加入再平衡日期中
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(index))).sort_values().unique()
    if len(idx) == 0:
        return idx

    selected = pd.DatetimeIndex([])

    if custom_dates is not None:
        custom_idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(custom_dates))).sort_values().unique()
        selected = idx.intersection(custom_idx)
    else:
        if freq is None or str(freq).lower() in {"none", "never", "buy_and_hold", "hold"}:
            selected = pd.DatetimeIndex([])
        else:
            f = str(freq).upper()
            if f == "D":
                selected = idx
            elif f in {"W", "M", "Q", "Y"}:
                periods = idx.to_period(f)
                selected = idx.to_series().groupby(periods).last().values
                selected = pd.DatetimeIndex(selected)
            else:
                raise ValueError("freq must be one of None/D/W/M/Q/Y or custom_dates")

    if include_first and idx[0] not in selected:
        selected = selected.union(pd.DatetimeIndex([idx[0]]))

    return pd.DatetimeIndex(selected).sort_values().unique()



def build_rebalance_flag(
    index: Sequence,
    freq: Optional[str] = "M",
    custom_dates: Optional[Sequence] = None,
    include_first: bool = True,
    name: str = "rebalance_flag",
) -> pd.Series:
    """
    构建再平衡标记序列。
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(index))).sort_values().unique()
    rebalance_dates = infer_rebalance_dates(
        index=idx,
        freq=freq,
        custom_dates=custom_dates,
        include_first=include_first,
    )
    flag = pd.Series(False, index=idx, name=name)
    flag.loc[rebalance_dates] = True
    return flag



def lag_weights(
    weights_df: pd.DataFrame,
    periods: int = 1,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    对权重矩阵整体滞后。
    常用于避免未来函数：t 日信号在 t+1 日生效。
    """
    w = ensure_datetime_index(weights_df)
    return w.shift(periods).fillna(fill_value)



def calc_turnover_from_weights(
    weights_df: pd.DataFrame,
    one_way: bool = True,
) -> pd.Series:
    """
    基于权重矩阵计算换手率。

    说明
    ----
    这里的 turnover = sum(|w_t - w_{t-1}|)
    若 one_way=True，则再除以 2，更接近常见单边换手率定义。
    """
    w = ensure_datetime_index(weights_df).fillna(0.0)
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    if one_way:
        turnover = turnover / 2.0
    turnover.name = "turnover"
    return turnover



def calc_effective_n_assets(weights_df: pd.DataFrame) -> pd.Series:
    """
    计算有效持仓数量：1 / sum(w_i^2)
    权重越分散，有效持仓数越高。
    """
    w = ensure_datetime_index(weights_df).fillna(0.0)
    eff_n = 1.0 / w.pow(2).sum(axis=1).replace(0.0, np.nan)
    eff_n.name = "effective_n_assets"
    return eff_n


# ============================================================
# 任意权重矩阵回测
# ============================================================


def backtest_from_weight_matrix(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    initial_nav: float = 1.0,
    fee_rate: float = 0.0,
    weight_lag: int = 0,
    cash_return: float | pd.Series = 0.0,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    对任意目标权重矩阵进行简单回测。

    假设
    ----
    1. returns_df 为简单收益率矩阵
    2. weights_df 表示每期“持有中的权重”
    3. 若需要用 t 日信号在 t+1 日生效，可设置 weight_lag=1
    4. 权重和可小于 1，剩余部分视为现金，按 cash_return 计息
    5. 交易成本按权重变动的绝对值求和乘以 fee_rate 扣除

    返回
    ----
    dict:
        portfolio_return, gross_return, net_return, nav,
        weights, turnover, trade_cost, risky_weight_sum, cash_weight
    """
    ret = ensure_datetime_index(returns_df).astype(float)
    w = ensure_datetime_index(weights_df).astype(float)

    ret, w = ret.align(w, join="inner", axis=0)
    w = w.reindex(columns=ret.columns).fillna(0.0)

    if weight_lag != 0:
        w = lag_weights(w, periods=weight_lag, fill_value=0.0)

    if isinstance(cash_return, pd.Series):
        cash_ret = ensure_datetime_index(cash_return).astype(float)
        cash_ret = cash_ret.reindex(ret.index).fillna(0.0)
    else:
        cash_ret = pd.Series(float(cash_return), index=ret.index, name="cash_return")

    risky_weight_sum = w.sum(axis=1)
    cash_weight = 1.0 - risky_weight_sum

    asset_gross_return = (w * ret).sum(axis=1)
    cash_leg_return = cash_weight * cash_ret
    gross_return = asset_gross_return + cash_leg_return

    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    trade_cost = turnover * float(fee_rate)

    net_return = gross_return - trade_cost
    nav = calc_nav_from_returns(net_return, initial_nav=initial_nav)

    asset_contribution = w.mul(ret, axis=0)
    asset_contribution.columns = [f"contrib_{c}" for c in asset_contribution.columns]

    return {
        "portfolio_return": net_return.rename("portfolio_return"),
        "gross_return": gross_return.rename("gross_return"),
        "net_return": net_return.rename("net_return"),
        "nav": nav.rename("nav"),
        "weights": w,
        "turnover": turnover.rename("turnover"),
        "trade_cost": trade_cost.rename("trade_cost"),
        "risky_weight_sum": risky_weight_sum.rename("risky_weight_sum"),
        "cash_weight": cash_weight.rename("cash_weight"),
        "asset_contribution": asset_contribution,
    }


# ============================================================
# 固定比例组合基准（基于价格、按份额精确模拟）
# ============================================================


def _prepare_price_and_target_weights(
    price_df: pd.DataFrame,
    target_weights: Mapping[str, float] | Sequence[float] | pd.Series,
    normalize: bool = True,
    total_weight: float = 1.0,
    drop_any_na: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    price = ensure_datetime_index(price_df).astype(float)

    if isinstance(target_weights, Mapping) or isinstance(target_weights, pd.Series):
        w = weights_dict_to_series(
            weights=target_weights,
            asset_names=price.columns.tolist(),
            fill_missing=0.0,
            normalize=normalize,
            total_weight=total_weight,
        )
    else:
        arr = np.asarray(target_weights, dtype=float)
        if len(arr) != price.shape[1]:
            raise ValueError("target_weights length does not match price_df columns")
        w = pd.Series(arr, index=price.columns, name="target_weight")
        if normalize:
            w = normalize_weights(w, target_sum=total_weight)
            w.name = "target_weight"

    keep_cols = w[w != 0.0].index.tolist()
    if len(keep_cols) == 0:
        raise ValueError("all target weights are zero")

    price = price[keep_cols].copy()
    w = w[keep_cols].astype(float)

    if drop_any_na:
        price = price.dropna(how="any")
    else:
        price = price.dropna(how="all")

    if price.empty:
        raise ValueError("price_df is empty after NA handling")

    return price, w



def _simulate_constant_weight_portfolio(
    price_df: pd.DataFrame,
    target_weights: Mapping[str, float] | Sequence[float] | pd.Series,
    rebalance_flag: pd.Series,
    initial_capital: float = 1.0,
    fee_rate: float = 0.0,
    normalize: bool = True,
    total_weight: float = 1.0,
    drop_any_na: bool = True,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    固定目标权重组合的份额级模拟引擎。
    """
    price, target_w = _prepare_price_and_target_weights(
        price_df=price_df,
        target_weights=target_weights,
        normalize=normalize,
        total_weight=total_weight,
        drop_any_na=drop_any_na,
    )

    rebalance_flag = ensure_datetime_index(rebalance_flag).reindex(price.index).fillna(False).astype(bool)
    if len(rebalance_flag) == 0 or not bool(rebalance_flag.iloc[0]):
        rebalance_flag.iloc[0] = True

    cols = price.columns.tolist()
    n = len(price)

    position_values = pd.DataFrame(index=price.index, columns=cols, dtype=float)
    weights = pd.DataFrame(index=price.index, columns=cols, dtype=float)
    shares = pd.DataFrame(index=price.index, columns=cols, dtype=float)
    turnover = pd.Series(0.0, index=price.index, name="turnover")
    trade_cost = pd.Series(0.0, index=price.index, name="trade_cost")
    cash_value = pd.Series(0.0, index=price.index, name="cash_value")
    total_value = pd.Series(np.nan, index=price.index, name="portfolio_value")

    target_risky_weight = float(target_w.sum())

    current_shares = pd.Series(0.0, index=cols, dtype=float)
    current_cash = float(initial_capital)

    for i, dt in enumerate(price.index):
        px = price.loc[dt].astype(float)

        holdings_value = current_shares * px
        pre_total_value = float(holdings_value.sum() + current_cash)

        if bool(rebalance_flag.loc[dt]):
            pre_weights = holdings_value / pre_total_value if pre_total_value != 0 else holdings_value * np.nan
            pre_weights = pre_weights.fillna(0.0)
            pre_cash_weight = max(0.0, 1.0 - float(pre_weights.sum()))

            target_full_weights = target_w.copy()
            if target_risky_weight < 1.0:
                # 未投资部分保持现金
                pass
            elif target_risky_weight > 1.0+1e-12:
                raise ValueError("target risky weight sum exceeds 1.0; leverage is not supported in this function")

            full_pre = pd.concat([pre_weights, pd.Series({"__cash__": pre_cash_weight})])
            full_target = pd.concat([target_full_weights, pd.Series({"__cash__": 1.0 - target_risky_weight})])
            this_turnover = float((full_target - full_pre).abs().sum())
            this_trade_cost = pre_total_value * float(fee_rate) * this_turnover

            post_total_value = pre_total_value - this_trade_cost
            target_values = post_total_value * target_w
            current_shares = target_values / px
            current_cash = post_total_value * (1.0 - target_risky_weight)

            holdings_value = current_shares * px
            total_val = float(holdings_value.sum() + current_cash)

            turnover.loc[dt] = this_turnover
            trade_cost.loc[dt] = this_trade_cost
        else:
            total_val = pre_total_value

        position_values.loc[dt] = holdings_value.values
        weights.loc[dt] = (holdings_value / total_val).values if total_val != 0 else np.nan
        shares.loc[dt] = current_shares.values
        cash_value.loc[dt] = current_cash
        total_value.loc[dt] = total_val

    nav = total_value / float(initial_capital)
    nav.name = "nav"
    portfolio_return = nav.pct_change(fill_method=None).fillna(0.0)
    portfolio_return.name = "portfolio_return"

    actual_rebalance_flag = rebalance_flag.reindex(price.index).fillna(False).astype(bool)
    actual_rebalance_flag.name = "rebalance_flag"

    return {
        "nav": nav,
        "portfolio_return": portfolio_return,
        "weights": weights,
        "position_values": position_values,
        "shares": shares,
        "cash_value": cash_value,
        "portfolio_value": total_value,
        "turnover": turnover,
        "trade_cost": trade_cost,
        "rebalance_flag": actual_rebalance_flag,
        "target_weight": target_w.rename("target_weight"),
    }



def backtest_buy_and_hold_portfolio(
    price_df: pd.DataFrame,
    target_weights: Mapping[str, float] | Sequence[float] | pd.Series,
    initial_capital: float = 1.0,
    fee_rate: float = 0.0,
    normalize: bool = True,
    total_weight: float = 1.0,
    drop_any_na: bool = True,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    固定比例买入后不动。
    首日按目标权重买入，之后仅随价格波动自然漂移。
    """
    price = ensure_datetime_index(price_df)
    flag = build_rebalance_flag(price.index, freq=None, include_first=True)
    return _simulate_constant_weight_portfolio(
        price_df=price,
        target_weights=target_weights,
        rebalance_flag=flag,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        normalize=normalize,
        total_weight=total_weight,
        drop_any_na=drop_any_na,
    )



def backtest_periodic_rebalance_portfolio(
    price_df: pd.DataFrame,
    target_weights: Mapping[str, float] | Sequence[float] | pd.Series,
    rebalance_freq: Optional[str] = "M",
    custom_rebalance_dates: Optional[Sequence] = None,
    initial_capital: float = 1.0,
    fee_rate: float = 0.0,
    normalize: bool = True,
    total_weight: float = 1.0,
    drop_any_na: bool = True,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    固定比例组合按周期再平衡。
    """
    price = ensure_datetime_index(price_df)
    flag = build_rebalance_flag(
        price.index,
        freq=rebalance_freq,
        custom_dates=custom_rebalance_dates,
        include_first=True,
    )
    return _simulate_constant_weight_portfolio(
        price_df=price,
        target_weights=target_weights,
        rebalance_flag=flag,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        normalize=normalize,
        total_weight=total_weight,
        drop_any_na=drop_any_na,
    )



def compare_fixed_weight_portfolios(
    price_df: pd.DataFrame,
    target_weights: Mapping[str, float] | Sequence[float] | pd.Series,
    rebalance_freq: Optional[str] = "M",
    custom_rebalance_dates: Optional[Sequence] = None,
    initial_capital: float = 1.0,
    fee_rate: float = 0.0,
    normalize: bool = True,
    total_weight: float = 1.0,
    drop_any_na: bool = True,
) -> dict[str, object]:
    """
    同时返回：
    1. 买入后不动结果
    2. 周期再平衡结果
    3. 净值对比表
    4. 简单绩效汇总表
    """
    buy_hold = backtest_buy_and_hold_portfolio(
        price_df=price_df,
        target_weights=target_weights,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        normalize=normalize,
        total_weight=total_weight,
        drop_any_na=drop_any_na,
    )

    rebalanced = backtest_periodic_rebalance_portfolio(
        price_df=price_df,
        target_weights=target_weights,
        rebalance_freq=rebalance_freq,
        custom_rebalance_dates=custom_rebalance_dates,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        normalize=normalize,
        total_weight=total_weight,
        drop_any_na=drop_any_na,
    )

    nav_table = pd.concat(
        [
            buy_hold["nav"].rename("buy_and_hold"),
            rebalanced["nav"].rename("periodic_rebalance"),
        ],
        axis=1,
    )

    summary = pd.concat(
        [
            summarize_performance(buy_hold["nav"]).rename("buy_and_hold"),
            summarize_performance(rebalanced["nav"]).rename("periodic_rebalance"),
        ],
        axis=1,
    ).T

    return {
        "buy_and_hold": buy_hold,
        "periodic_rebalance": rebalanced,
        "nav_table": nav_table,
        "summary": summary,
    }


# ============================================================
# 绩效评价与基准对比
# ============================================================


def calc_drawdown_series(nav: pd.Series) -> pd.Series:
    """
    计算回撤序列。
    """
    x = ensure_datetime_index(nav).astype(float)
    running_max = x.cummax()
    dd = x / running_max - 1.0
    dd.name = "drawdown"
    return dd



def calc_max_drawdown(nav: pd.Series) -> float:
    """
    计算最大回撤。
    """
    dd = calc_drawdown_series(nav)
    return float(dd.min())



def calc_annualized_return(
    nav: pd.Series,
    annualization: int = 252,
) -> float:
    """
    计算年化收益率（CAGR 近似）。
    """
    x = ensure_datetime_index(nav).astype(float).dropna()
    if len(x) < 2:
        return np.nan
    total_return = float(x.iloc[-1] / x.iloc[0])
    n_periods = len(x) - 1
    if total_return <= 0 or n_periods <= 0:
        return np.nan
    return float(total_return ** (annualization / n_periods) - 1.0)



def calc_annualized_volatility(
    nav: pd.Series | None = None,
    returns: Optional[pd.Series] = None,
    annualization: int = 252,
) -> float:
    """
    计算年化波动率。
    """
    if returns is None:
        if nav is None:
            raise ValueError("either nav or returns must be provided")
        ret = nav_to_returns(nav)
    else:
        ret = ensure_datetime_index(returns).astype(float).dropna()

    if len(ret) < 2:
        return np.nan
    return float(ret.std(ddof=1) * np.sqrt(annualization))



def calc_sharpe_ratio(
    nav: pd.Series,
    risk_free_annual: float = 0.0,
    annualization: int = 252,
) -> float:
    """
    计算夏普比率。
    """
    ret = nav_to_returns(nav)
    if len(ret) < 2:
        return np.nan
    rf = annual_rate_to_period_return(risk_free_annual, annualization=annualization)
    excess = ret - rf
    vol = excess.std(ddof=1)
    if np.isclose(vol, 0.0):
        return np.nan
    return float(excess.mean() / vol * np.sqrt(annualization))



def calc_sortino_ratio(
    nav: pd.Series,
    risk_free_annual: float = 0.0,
    annualization: int = 252,
) -> float:
    """
    计算 Sortino 比率。
    """
    ret = nav_to_returns(nav)
    if len(ret) < 2:
        return np.nan
    rf = annual_rate_to_period_return(risk_free_annual, annualization=annualization)
    excess = ret - rf
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1)
    if downside.empty or np.isclose(downside_std, 0.0):
        return np.nan
    return float(excess.mean() / downside_std * np.sqrt(annualization))



def calc_calmar_ratio(
    nav: pd.Series,
    annualization: int = 252,
) -> float:
    """
    计算 Calmar 比率。
    """
    ann_ret = calc_annualized_return(nav, annualization=annualization)
    mdd = calc_max_drawdown(nav)
    if np.isnan(ann_ret) or np.isclose(mdd, 0.0):
        return np.nan
    return float(ann_ret / abs(mdd))



def calc_win_rate(nav: pd.Series) -> float:
    """
    计算收益率为正的期数占比。
    """
    ret = nav_to_returns(nav)
    if len(ret) == 0:
        return np.nan
    return float((ret > 0).mean())



def calc_tracking_error(
    strategy_nav: pd.Series,
    benchmark_nav: pd.Series,
    annualization: int = 252,
) -> float:
    """
    计算年化跟踪误差。
    """
    s, b = align_nav_series(strategy_nav, benchmark_nav, join="inner")
    active_ret = nav_to_returns(s) - nav_to_returns(b)
    if len(active_ret) < 2:
        return np.nan
    return float(active_ret.std(ddof=1) * np.sqrt(annualization))



def calc_information_ratio(
    strategy_nav: pd.Series,
    benchmark_nav: pd.Series,
    annualization: int = 252,
) -> float:
    """
    计算信息比率。
    """
    s, b = align_nav_series(strategy_nav, benchmark_nav, join="inner")
    active_ret = nav_to_returns(s) - nav_to_returns(b)
    if len(active_ret) < 2:
        return np.nan
    te = active_ret.std(ddof=1)
    if np.isclose(te, 0.0):
        return np.nan
    return float(active_ret.mean() / te * np.sqrt(annualization))



def calc_beta_alpha(
    strategy_nav: pd.Series,
    benchmark_nav: pd.Series,
    risk_free_annual: float = 0.0,
    annualization: int = 252,
) -> tuple[float, float]:
    """
    用简化 CAPM 方式计算 beta 与年化 alpha。
    """
    s, b = align_nav_series(strategy_nav, benchmark_nav, join="inner")
    sr = nav_to_returns(s)
    br = nav_to_returns(b)
    idx = sr.index.intersection(br.index)
    sr = sr.reindex(idx).dropna()
    br = br.reindex(idx).dropna()
    idx = sr.index.intersection(br.index)
    sr = sr.reindex(idx)
    br = br.reindex(idx)

    if len(sr) < 2:
        return np.nan, np.nan

    rf = annual_rate_to_period_return(risk_free_annual, annualization=annualization)
    s_ex = sr - rf
    b_ex = br - rf

    var_b = float(b_ex.var(ddof=1))
    if np.isclose(var_b, 0.0):
        return np.nan, np.nan

    beta = float(np.cov(s_ex, b_ex, ddof=1)[0, 1] / var_b)
    alpha_period = float(s_ex.mean() - beta * b_ex.mean())
    alpha_annual = float((1.0 + alpha_period) ** annualization - 1.0)
    return beta, alpha_annual



def summarize_performance(
    nav: pd.Series,
    risk_free_annual: float = 0.0,
    annualization: int = 252,
) -> pd.Series:
    """
    汇总单条净值曲线的常用绩效指标。
    """
    x = ensure_datetime_index(nav).astype(float).dropna()
    ret = nav_to_returns(x)

    summary = pd.Series(
        {
            "start": x.index.min(),
            "end": x.index.max(),
            "n_obs": len(x),
            "total_return": float(x.iloc[-1] / x.iloc[0] - 1.0) if len(x) > 0 else np.nan,
            "annual_return": calc_annualized_return(x, annualization=annualization),
            "annual_vol": calc_annualized_volatility(x, annualization=annualization),
            "sharpe": calc_sharpe_ratio(x, risk_free_annual=risk_free_annual, annualization=annualization),
            "sortino": calc_sortino_ratio(x, risk_free_annual=risk_free_annual, annualization=annualization),
            "calmar": calc_calmar_ratio(x, annualization=annualization),
            "max_drawdown": calc_max_drawdown(x),
            "win_rate": calc_win_rate(x),
            "avg_period_return": float(ret.mean()) if len(ret) > 0 else np.nan,
        },
        name="performance",
    )
    return summary



def summarize_relative_performance(
    strategy_nav: pd.Series,
    benchmark_nav: pd.Series,
    risk_free_annual: float = 0.0,
    annualization: int = 252,
) -> pd.Series:
    """
    汇总策略相对基准的常用对比指标。
    """
    s, b = align_nav_series(strategy_nav, benchmark_nav, join="inner")
    s = s.dropna()
    b = b.dropna()
    idx = s.index.intersection(b.index)
    s = s.reindex(idx)
    b = b.reindex(idx)

    if len(s) < 2:
        return pd.Series(dtype=float, name="relative_performance")

    strategy_total = float(s.iloc[-1] / s.iloc[0] - 1.0)
    benchmark_total = float(b.iloc[-1] / b.iloc[0] - 1.0)
    beta, alpha = calc_beta_alpha(
        strategy_nav=s,
        benchmark_nav=b,
        risk_free_annual=risk_free_annual,
        annualization=annualization,
    )

    excess_nav = normalize_nav(s / b, start_value=1.0)

    summary = pd.Series(
        {
            "strategy_total_return": strategy_total,
            "benchmark_total_return": benchmark_total,
            "excess_total_return": strategy_total - benchmark_total,
            "tracking_error": calc_tracking_error(s, b, annualization=annualization),
            "information_ratio": calc_information_ratio(s, b, annualization=annualization),
            "beta": beta,
            "alpha_annual": alpha,
            "excess_max_drawdown": calc_max_drawdown(excess_nav),
            "correlation": float(nav_to_returns(s).corr(nav_to_returns(b))),
        },
        name="relative_performance",
    )
    return summary



def summarize_weight_statistics(weights_df: pd.DataFrame) -> pd.DataFrame:
    """
    汇总权重矩阵的一些通用特征。
    """
    w = ensure_datetime_index(weights_df).fillna(0.0)
    turnover = calc_turnover_from_weights(w, one_way=False)
    eff_n = calc_effective_n_assets(w)

    rows = {
        "start": w.index.min(),
        "end": w.index.max(),
        "n_obs": len(w),
        "avg_gross_weight": float(w.abs().sum(axis=1).mean()),
        "avg_net_weight": float(w.sum(axis=1).mean()),
        "avg_max_single_weight": float(w.abs().max(axis=1).mean()),
        "avg_turnover": float(turnover.mean()),
        "max_turnover": float(turnover.max()),
        "avg_effective_n_assets": float(eff_n.mean()),
    }
    return pd.DataFrame([rows])


# ============================================================
# 示例
# ============================================================

if __name__ == "__main__":
    # 示例：三资产固定权重组合
    df = pd.DataFrame(
        {
            "trade_date": [
                "20240102", "20240103", "20240104", "20240105",
                "20240102", "20240103", "20240104", "20240105",
                "20240102", "20240103", "20240104", "20240105",
            ],
            "ts_code": [
                "510300.SH", "510300.SH", "510300.SH", "510300.SH",
                "511010.SH", "511010.SH", "511010.SH", "511010.SH",
                "518880.SH", "518880.SH", "518880.SH", "518880.SH",
            ],
            "close": [
                3.50, 3.52, 3.49, 3.55,
                112.0, 112.1, 112.0, 112.2,
                4.85, 4.90, 4.88, 4.95,
            ],
        }
    )

    price = build_price_matrix(df)
    weights = {"510300.SH": 0.5, "511010.SH": 0.3, "518880.SH": 0.2}

    result = compare_fixed_weight_portfolios(
        price_df=price,
        target_weights=weights,
        rebalance_freq="M",
        fee_rate=0.0,
    )

    print("nav table")
    print(result["nav_table"])
    print()
    print("summary")
    print(result["summary"])
