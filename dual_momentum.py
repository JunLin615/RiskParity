"""
dual_momentum.py

纯函数版双动量（Dual Momentum）函数库。
不依赖数据库、回测撮合、资金管理模块。
主要提供：
1. 价格矩阵整理与缺失处理
2. 动量、波动率、均线过滤等信号计算
3. 绝对动量 + 相对动量的资产筛选
4. 目标权重生成（不含成交执行）
5. 调仓时点下的信号与权重明细输出

设计原则：
- 输入输出尽量使用 pandas / numpy
- 不和数据库、回测、下单逻辑耦合
- 优先适配日线级别 ETF / LOF / 指数类资产
- 函数命名与 risk_parity.py 尽量保持一致
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    import talib
    HAS_TALIB = True
except Exception:
    talib = None
    HAS_TALIB = False


# ============================================================
# 基础清洗与矩阵整理
# ============================================================


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保 DataFrame 的 index 为 DatetimeIndex，并按时间升序排列。
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
    out = price.reindex(calendar_index)
    return out



def forward_fill_prices(
    price_df: pd.DataFrame,
    limit: Optional[int] = 5,
) -> pd.DataFrame:
    """
    对价格矩阵做前向填充。只填中间缺口，不会填序列开头的缺失。
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
    price = price[keep_cols]
    return price



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
# 收益率、波动率与过滤器
# ============================================================


def calc_simple_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算简单收益率：r_t = P_t / P_{t-1} - 1
    """
    price = ensure_datetime_index(price_df)
    ret = price.pct_change()
    return ret.dropna(how="all")



def calc_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算对数收益率：r_t = ln(P_t / P_{t-1})
    """
    price = ensure_datetime_index(price_df)
    ret = np.log(price / price.shift(1))
    return ret.dropna(how="all")



def calc_trailing_returns(
    price_df: pd.DataFrame,
    lookback: int,
    skip_recent: int = 0,
    return_type: str = "simple",
) -> pd.DataFrame:
    """
    计算滚动回看收益率。

    参数
    ----
    lookback : int
        回看窗口长度。例如 60 表示过去 60 个交易日收益率。
    skip_recent : int
        跳过最近若干个交易日。例如 21 可实现近似 12-1 动量。
    return_type : {"simple", "log"}
        简单收益率或对数收益率。
    """
    price = ensure_datetime_index(price_df)

    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if skip_recent < 0:
        raise ValueError("skip_recent must be >= 0")

    p_end = price.shift(skip_recent)
    p_start = price.shift(lookback + skip_recent)

    if return_type == "simple":
        out = p_end / p_start - 1.0
    elif return_type == "log":
        out = np.log(p_end / p_start)
    else:
        raise ValueError("return_type must be 'simple' or 'log'")

    return out



def calc_multi_lookback_momentum(
    price_df: pd.DataFrame,
    lookbacks: Sequence[int],
    weights: Optional[Sequence[float]] = None,
    skip_recent: int = 0,
    return_type: str = "simple",
) -> pd.DataFrame:
    """
    计算多周期加权动量得分。

    例如：20/60/120 日动量加权求和。
    """
    if len(lookbacks) == 0:
        raise ValueError("lookbacks must not be empty")

    lbs = [int(x) for x in lookbacks]
    if any(x <= 0 for x in lbs):
        raise ValueError("all lookbacks must be positive")

    if weights is None:
        w = np.ones(len(lbs), dtype=float) / len(lbs)
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(lbs):
            raise ValueError("weights length must match lookbacks length")
        if np.isclose(w.sum(), 0.0):
            raise ValueError("weights sum must not be 0")
        w = w / w.sum()

    score = None
    for lb, wi in zip(lbs, w):
        r = calc_trailing_returns(
            price_df=price_df,
            lookback=lb,
            skip_recent=skip_recent,
            return_type=return_type,
        )
        term = r * wi
        score = term if score is None else score.add(term, fill_value=np.nan)

    return score



def calc_annualized_volatility(
    returns_df: pd.DataFrame,
    annualization: int = 252,
    rolling_window: Optional[int] = None,
) -> pd.DataFrame | pd.Series:
    """
    计算年化波动率。

    - rolling_window 为 None：返回各列整体波动率 Series
    - rolling_window 为整数：返回滚动年化波动率 DataFrame
    """
    ret = returns_df.copy()

    if rolling_window is None:
        return ret.std(ddof=1) * np.sqrt(annualization)

    return ret.rolling(rolling_window).std(ddof=1) * np.sqrt(annualization)



def calc_moving_average(
    price_df: pd.DataFrame,
    window: int,
    ma_type: str = "sma",
    use_talib: bool = False,
) -> pd.DataFrame:
    """
    计算移动平均线。

    参数
    ----
    ma_type : {"sma", "ema"}
        均线类型。
    use_talib : bool
        若为 True 且安装了 TA-Lib，则优先使用 TA-Lib。
    """
    price = ensure_datetime_index(price_df)

    if window <= 0:
        raise ValueError("window must be positive")

    ma_type = ma_type.lower()
    if ma_type not in {"sma", "ema"}:
        raise ValueError("ma_type must be 'sma' or 'ema'")

    if use_talib and HAS_TALIB:
        out = pd.DataFrame(index=price.index, columns=price.columns, dtype=float)
        for col in price.columns:
            arr = price[col].astype(float).values
            if ma_type == "sma":
                out[col] = talib.SMA(arr, timeperiod=window)
            else:
                out[col] = talib.EMA(arr, timeperiod=window)
        return out

    if ma_type == "sma":
        return price.rolling(window).mean()
    return price.ewm(span=window, adjust=False).mean()



def calc_price_above_ma_filter(
    price_df: pd.DataFrame,
    ma_window: int,
    ma_type: str = "sma",
    use_talib: bool = False,
) -> pd.DataFrame:
    """
    价格是否位于均线之上。
    """
    price = ensure_datetime_index(price_df)
    ma = calc_moving_average(price, window=ma_window, ma_type=ma_type, use_talib=use_talib)
    return price > ma



def calc_market_regime_filter(
    price_df: pd.DataFrame,
    market_asset: str,
    lookback: int = 60,
    threshold: float = 0.0,
    skip_recent: int = 0,
    ma_window: Optional[int] = None,
    ma_type: str = "sma",
    use_talib: bool = False,
) -> pd.Series:
    """
    计算市场总开关。

    规则：
    1. 市场资产的绝对动量 > threshold
    2. 若指定 ma_window，则还要求价格 > MA
    """
    price = ensure_datetime_index(price_df)
    if market_asset not in price.columns:
        raise KeyError(f"market_asset '{market_asset}' not found in columns")

    market_px = price[[market_asset]]
    abs_ret = calc_trailing_returns(
        market_px,
        lookback=lookback,
        skip_recent=skip_recent,
        return_type="simple",
    )[market_asset]
    regime = abs_ret > threshold

    if ma_window is not None:
        ma_filter = calc_price_above_ma_filter(
            market_px,
            ma_window=ma_window,
            ma_type=ma_type,
            use_talib=use_talib,
        )[market_asset]
        regime = regime & ma_filter

    return regime.astype(bool)


# ============================================================
# 动量信号
# ============================================================


def calc_absolute_momentum(
    price_df: pd.DataFrame,
    lookback: int = 60,
    threshold: float = 0.0,
    skip_recent: int = 0,
    return_type: str = "simple",
    ma_window: Optional[int] = None,
    ma_type: str = "sma",
    use_talib: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算绝对动量得分与通过过滤的布尔矩阵。

    返回
    ----
    scores : DataFrame
        绝对动量得分（通常为过去 N 日收益率）
    passed : DataFrame
        是否通过绝对动量过滤
    """
    price = ensure_datetime_index(price_df)
    scores = calc_trailing_returns(
        price_df=price,
        lookback=lookback,
        skip_recent=skip_recent,
        return_type=return_type,
    )
    passed = scores > threshold

    if ma_window is not None:
        ma_filter = calc_price_above_ma_filter(
            price_df=price,
            ma_window=ma_window,
            ma_type=ma_type,
            use_talib=use_talib,
        )
        passed = passed & ma_filter

    return scores, passed.astype(bool)



def calc_relative_momentum(
    price_df: pd.DataFrame,
    lookback: Optional[int] = 60,
    skip_recent: int = 0,
    return_type: str = "simple",
    lookbacks: Optional[Sequence[int]] = None,
    weights: Optional[Sequence[float]] = None,
    risk_adjusted: bool = False,
    vol_lookback: int = 20,
    annualization: int = 252,
) -> pd.DataFrame:
    """
    计算相对动量得分。

    支持两种方式：
    1. 单窗口动量（lookback）
    2. 多窗口加权动量（lookbacks + weights）

    可选做风险调整：score / rolling_vol
    """
    price = ensure_datetime_index(price_df)

    if lookbacks is not None:
        score = calc_multi_lookback_momentum(
            price_df=price,
            lookbacks=lookbacks,
            weights=weights,
            skip_recent=skip_recent,
            return_type=return_type,
        )
    else:
        if lookback is None:
            raise ValueError("lookback must not be None when lookbacks is None")
        score = calc_trailing_returns(
            price_df=price,
            lookback=lookback,
            skip_recent=skip_recent,
            return_type=return_type,
        )

    if risk_adjusted:
        ret = calc_simple_returns(price)
        rolling_vol = calc_annualized_volatility(
            ret,
            annualization=annualization,
            rolling_window=vol_lookback,
        )
        score = score / rolling_vol

    return score



def calc_cross_sectional_rank(
    score_df: pd.DataFrame,
    ascending: bool = False,
    method: str = "dense",
) -> pd.DataFrame:
    """
    计算横截面排名。

    默认 descending rank：分数越高，排名越靠前（1 为最强）。
    """
    score = score_df.copy()
    return score.rank(axis=1, ascending=ascending, method=method)



def mask_relative_momentum_by_absolute(
    relative_score_df: pd.DataFrame,
    absolute_pass_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    用绝对动量过滤相对动量得分。
    未通过绝对动量过滤的资产记为 NaN。
    """
    rel = relative_score_df.copy()
    mask = absolute_pass_df.reindex_like(rel)
    return rel.where(mask)


# ============================================================
# 权重分配
# ============================================================


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """
    归一化权重，使其和为 1。
    """
    w = np.asarray(weights, dtype=float)
    total = np.nansum(w)
    if np.isclose(total, 0.0):
        raise ValueError("weights sum is 0, cannot normalize")
    return w / total



def equal_weights(n_assets: int) -> np.ndarray:
    """
    等权重。
    """
    if n_assets <= 0:
        raise ValueError("n_assets must be positive")
    return np.ones(n_assets, dtype=float) / n_assets



def rank_weights(ranks: Sequence[float], descending_best: bool = True) -> np.ndarray:
    """
    根据排名分配权重。

    参数
    ----
    ranks : Sequence[float]
        排名，通常 1 为最好。
    descending_best : bool
        True 表示排名越小越好。
    """
    r = np.asarray(ranks, dtype=float)
    if descending_best:
        score = 1.0 / r
    else:
        score = r
    return normalize_weights(score)



def score_weights(scores: Sequence[float], non_negative: bool = True) -> np.ndarray:
    """
    根据分数分配权重。

    若 non_negative=True，则会将负分数截断为 0。
    """
    s = np.asarray(scores, dtype=float)
    if non_negative:
        s = np.clip(s, a_min=0.0, a_max=None)
    if np.isclose(np.nansum(s), 0.0):
        return equal_weights(len(s))
    return normalize_weights(s)



def inverse_volatility_weights(
    vol: pd.Series | np.ndarray,
) -> pd.Series | np.ndarray:
    """
    逆波动率权重。
    """
    if isinstance(vol, pd.Series):
        inv = 1.0 / vol
        inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
        w = inv / inv.sum()
        return w

    vol_arr = np.asarray(vol, dtype=float)
    inv = 1.0 / vol_arr
    inv[np.isinf(inv)] = np.nan
    inv = np.nan_to_num(inv, nan=0.0)
    return normalize_weights(inv)



def calc_inverse_vol_weights_from_prices(
    price_df: pd.DataFrame,
    vol_lookback: int = 20,
    annualization: int = 252,
) -> pd.Series:
    """
    从价格矩阵直接计算逆波动率权重。
    """
    price = ensure_datetime_index(price_df)
    ret = calc_simple_returns(price)
    rolling_vol = calc_annualized_volatility(
        ret,
        annualization=annualization,
        rolling_window=vol_lookback,
    )
    vol = rolling_vol.iloc[-1].dropna()
    w = inverse_volatility_weights(vol)
    return pd.Series(w, index=vol.index, name="weight")



def cap_weights(
    weights: pd.Series,
    max_weight: float,
    target_sum: Optional[float] = None,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> pd.Series:
    """
    对权重设置单资产上限，并尽量保持总投资权重不超过 target_sum。

    说明
    ----
    - 若 target_sum 为 None，则默认保持原权重和不变。
    - 若在 max_weight 限制下无法分配满 target_sum，则会保留剩余未分配权重，
      由调用方决定是否计入现金仓位。
    """
    if max_weight <= 0 or max_weight > 1:
        raise ValueError("max_weight must be in (0, 1]")

    w = weights.copy().astype(float)
    current_sum = float(w.sum())
    if np.isclose(current_sum, 0.0):
        return w * 0.0

    if target_sum is None:
        target_sum = current_sum
    target_sum = float(target_sum)

    if target_sum < 0:
        raise ValueError("target_sum must be >= 0")
    if np.isclose(target_sum, 0.0):
        return w * 0.0

    # 将输入权重缩放到目标总投资权重
    w = w / current_sum * target_sum

    for _ in range(max_iter):
        over = w > max_weight + tol
        if not over.any():
            break

        fixed = w.where(over, 0.0).clip(upper=max_weight)
        free = w.where(~over, 0.0)
        free_sum = float(free.sum())
        remaining = target_sum - float(fixed.sum())

        if remaining <= tol:
            w = fixed
            break

        if free_sum <= tol:
            # 无法继续分配，保留剩余未分配权重
            w = fixed
            break

        w = fixed + free * (remaining / free_sum)

    w = w.clip(lower=0.0)
    return w


# ============================================================
# 双动量核心：单期快照
# ============================================================


def _select_available_assets(
    price_df: pd.DataFrame,
    candidate_assets: Optional[Sequence[str]] = None,
    defensive_assets: Optional[Sequence[str]] = None,
) -> tuple[list[str], list[str]]:
    """
    解析风险资产池与防御资产池。
    """
    cols = price_df.columns.tolist()
    defensive = [] if defensive_assets is None else [x for x in defensive_assets if x in cols]

    if candidate_assets is None:
        candidate = [x for x in cols if x not in defensive]
    else:
        candidate = [x for x in candidate_assets if x in cols]

    if len(candidate) == 0:
        raise ValueError("candidate assets are empty after filtering by available columns")

    return candidate, defensive



def _apply_min_history_filter(
    price_window: pd.DataFrame,
    assets: Sequence[str],
    min_history: Optional[int] = None,
) -> list[str]:
    """
    保留历史数据长度充足的资产。
    """
    if min_history is None:
        return list(assets)

    keep = []
    for asset in assets:
        s = price_window[asset].dropna()
        if len(s) >= min_history:
            keep.append(asset)
    return keep



def calc_dual_momentum_snapshot(
    price_df: pd.DataFrame,
    as_of_date: Optional[str | pd.Timestamp] = None,
    candidate_assets: Optional[Sequence[str]] = None,
    defensive_assets: Optional[Sequence[str]] = None,
    abs_lookback: int = 60,
    abs_threshold: float = 0.0,
    abs_skip_recent: int = 0,
    abs_return_type: str = "simple",
    abs_ma_window: Optional[int] = None,
    abs_ma_type: str = "sma",
    rel_lookback: Optional[int] = 60,
    rel_lookbacks: Optional[Sequence[int]] = None,
    rel_weights: Optional[Sequence[float]] = None,
    rel_skip_recent: int = 0,
    rel_return_type: str = "simple",
    rel_risk_adjusted: bool = False,
    rel_vol_lookback: int = 20,
    top_k: int = 3,
    weighting: str = "equal",
    fill_unallocated_to_defensive: bool = True,
    min_history: Optional[int] = None,
    annualization: int = 252,
    market_asset: Optional[str] = None,
    market_lookback: Optional[int] = None,
    market_threshold: float = 0.0,
    market_skip_recent: int = 0,
    market_ma_window: Optional[int] = None,
    market_ma_type: str = "sma",
    use_talib: bool = False,
    max_single_weight: Optional[float] = None,
) -> dict:
    """
    计算某一个观察时点的双动量信号与目标权重。

    说明
    ----
    本函数只生成信号与目标权重，不进行成交撮合。

    返回
    ----
    dict
        {
            "as_of_date": Timestamp,
            "selected_assets": list[str],
            "defensive_assets_used": list[str],
            "target_weights": Series,
            "cash_weight": float,
            "detail": DataFrame,
            "market_regime_on": bool,
        }
    """
    price = ensure_datetime_index(price_df)
    if as_of_date is not None:
        as_of = pd.Timestamp(as_of_date)
        price = price.loc[:as_of]
    if price.empty:
        raise ValueError("price_df is empty after slicing by as_of_date")

    candidate, defensive = _select_available_assets(
        price_df=price,
        candidate_assets=candidate_assets,
        defensive_assets=defensive_assets,
    )

    need_windows = [abs_lookback + abs_skip_recent]
    if rel_lookbacks is not None:
        need_windows.extend([int(x) + rel_skip_recent for x in rel_lookbacks])
    elif rel_lookback is not None:
        need_windows.append(rel_lookback + rel_skip_recent)
    if abs_ma_window is not None:
        need_windows.append(abs_ma_window)
    if rel_risk_adjusted:
        need_windows.append(rel_vol_lookback + 1)
    if market_lookback is not None:
        need_windows.append(market_lookback + market_skip_recent)
    if market_ma_window is not None:
        need_windows.append(market_ma_window)
    
    need_n = max(need_windows) + 1

    # 先用完整历史做 min_history 准入过滤
    candidate = _apply_min_history_filter(price, candidate, min_history=min_history)

    if len(candidate) == 0:
        raise ValueError("no candidate assets left after min_history filter")

    # EMA 对初值更敏感；若本期快照使用 EMA 过滤，则优先使用全历史，
    # 避免因截断窗口导致最后一期 EMA 明显失真。
    use_full_history = (
        abs_ma_window is not None
        and abs_ma_type.lower() == "ema"
    ) or (
        market_ma_window is not None
        and market_ma_type.lower() == "ema"
    )

    if use_full_history:
        price_window = price.copy()
    else:
        price_window = select_last_n_rows(price, min(len(price), need_n + 5))
    # 市场总开关
    market_regime_on = True
    if market_asset is not None:
        mk_lb = abs_lookback if market_lookback is None else market_lookback
        market_regime = calc_market_regime_filter(
            price_df=price_window,
            market_asset=market_asset,
            lookback=mk_lb,
            threshold=market_threshold,
            skip_recent=market_skip_recent,
            ma_window=market_ma_window,
            ma_type=market_ma_type,
            use_talib=use_talib,
        )
        market_regime_on = bool(market_regime.iloc[-1]) if len(market_regime) > 0 else False

    candidate_price = price_window[candidate]

    abs_score_df, abs_pass_df = calc_absolute_momentum(
        price_df=candidate_price,
        lookback=abs_lookback,
        threshold=abs_threshold,
        skip_recent=abs_skip_recent,
        return_type=abs_return_type,
        ma_window=abs_ma_window,
        ma_type=abs_ma_type,
        use_talib=use_talib,
    )
    rel_score_df = calc_relative_momentum(
        price_df=candidate_price,
        lookback=rel_lookback,
        skip_recent=rel_skip_recent,
        return_type=rel_return_type,
        lookbacks=rel_lookbacks,
        weights=rel_weights,
        risk_adjusted=rel_risk_adjusted,
        vol_lookback=rel_vol_lookback,
        annualization=annualization,
    )
    masked_rel_score = mask_relative_momentum_by_absolute(rel_score_df, abs_pass_df)
    rank_df = calc_cross_sectional_rank(masked_rel_score, ascending=False, method="dense")

    abs_score = abs_score_df.iloc[-1].reindex(candidate)
    abs_pass = abs_pass_df.iloc[-1].reindex(candidate).fillna(False)
    rel_score = masked_rel_score.iloc[-1].reindex(candidate)
    rel_rank = rank_df.iloc[-1].reindex(candidate)

    if not market_regime_on:
        abs_pass[:] = False
        rel_score[:] = np.nan
        rel_rank[:] = np.nan

    eligible = rel_score.dropna().sort_values(ascending=False)
    selected_assets = eligible.head(top_k).index.tolist()

    weights = pd.Series(0.0, index=price.columns, name="target_weight")
    defensive_used: list[str] = []
    cash_weight = 0.0

    if len(selected_assets) > 0:
        if weighting == "equal":
            risk_w = pd.Series(equal_weights(len(selected_assets)), index=selected_assets)
        elif weighting == "score":
            risk_w = pd.Series(score_weights(rel_score[selected_assets].values), index=selected_assets)
        elif weighting == "rank":
            risk_w = pd.Series(rank_weights(rel_rank[selected_assets].values), index=selected_assets)
        elif weighting == "inv_vol":
            sub_px = candidate_price[selected_assets]
            risk_w = calc_inverse_vol_weights_from_prices(
                sub_px,
                vol_lookback=rel_vol_lookback,
                annualization=annualization,
            )
            risk_w = risk_w.reindex(selected_assets).fillna(0.0)
        else:
            raise ValueError("weighting must be one of {'equal', 'score', 'rank', 'inv_vol'}")

        if fill_unallocated_to_defensive and len(selected_assets) < top_k:
            risk_alloc = len(selected_assets) / top_k
            risk_w = risk_w * risk_alloc
            leftover = max(0.0, 1.0 - float(risk_w.sum()))
            if len(defensive) > 0 and leftover > 0:
                defensive_used = defensive.copy()
                defensive_w = pd.Series(equal_weights(len(defensive_used)), index=defensive_used) * leftover
                weights.loc[defensive_w.index] = defensive_w.values
            else:
                cash_weight = leftover

        weights.loc[risk_w.index] = risk_w.values
    else:
        if len(defensive) > 0:
            defensive_used = defensive.copy()
            weights.loc[defensive_used] = equal_weights(len(defensive_used))
        else:
            cash_weight = 1.0

    if max_single_weight is not None and weights.sum() > 0:
        non_zero = weights[weights > 0]
        pre_cap_invested = float(non_zero.sum())
        capped = cap_weights(
            non_zero,
            max_weight=max_single_weight,
            target_sum=pre_cap_invested,
        )
        residual_cash = max(0.0, pre_cap_invested - float(capped.sum()))
        weights.loc[:] = 0.0
        weights.loc[capped.index] = capped.values
        cash_weight += residual_cash

    weights = weights.clip(lower=0.0)
    cash_weight = max(0.0, float(cash_weight))

    total_alloc = float(weights.sum()) + cash_weight
    if total_alloc > 1.0 + 1e-10:
        scale = 1.0 / total_alloc
        weights = weights * scale
        cash_weight = cash_weight * scale

    detail = pd.DataFrame({
        "close": price_window.iloc[-1].reindex(candidate),
        "abs_score": abs_score,
        "abs_pass": abs_pass.astype(bool),
        "rel_score": rel_score,
        "rel_rank": rel_rank,
        "selected": False,
        "target_weight": weights.reindex(candidate).fillna(0.0),
    }, index=candidate)
    if len(selected_assets) > 0:
        detail.loc[selected_assets, "selected"] = True
    detail.index.name = "asset"

    return {
        "as_of_date": price_window.index[-1],
        "selected_assets": selected_assets,
        "defensive_assets_used": defensive_used,
        "target_weights": weights,
        "cash_weight": cash_weight,
        "detail": detail.sort_values(["selected", "rel_score"], ascending=[False, False]),
        "market_regime_on": market_regime_on,
    }


# ============================================================
# 调仓辅助与时间序列权重
# ============================================================


def get_rebalance_dates(
    price_df: pd.DataFrame,
    freq: str = "M",
) -> pd.DatetimeIndex:
    """
    生成调仓日期。

    参数
    ----
    freq : str
        例如：
        - "M"  月末
        - "W-FRI" 每周最后一个周五
        - "Q"  季末

    返回
    ----
    DatetimeIndex
        从已有交易日中抽取的调仓日。
    """
    price = ensure_datetime_index(price_df)
    if price.empty:
        return pd.DatetimeIndex([])

    freq_alias = {
        "M": "ME",
        "Q": "QE",
        "Y": "YE",
        "A": "YE",
    }.get(freq, freq)

    marker = pd.Series(price.index, index=price.index)
    rebalance = marker.resample(freq_alias).last().dropna()
    return pd.DatetimeIndex(rebalance.values)



def calc_dual_momentum_weights_over_time(
    price_df: pd.DataFrame,
    rebalance_dates: Optional[Sequence] = None,
    freq: str = "M",
    warmup: Optional[int] = None,
    include_cash: bool = False,
    cash_column_name: str = "CASH",
    **snapshot_kwargs,
) -> pd.DataFrame:
    """
    按调仓时点批量计算双动量目标权重。

    说明
    ----
    只输出每个调仓日的目标权重矩阵，不做持仓推进与收益计算。
    """
    price = ensure_datetime_index(price_df)
    if rebalance_dates is None:
        dates = get_rebalance_dates(price, freq=freq)
    else:
        dates = pd.to_datetime(pd.Index(rebalance_dates))
        dates = pd.DatetimeIndex([d for d in dates if d in price.index])

    if warmup is not None:
        dates = dates[dates >= price.index[min(len(price.index) - 1, warmup - 1)]]

    rows = []
    for dt in dates:
        snap = calc_dual_momentum_snapshot(
            price_df=price.loc[:dt],
            as_of_date=dt,
            **snapshot_kwargs,
        )
        w = snap["target_weights"].copy()
        if include_cash:
            w.loc[cash_column_name] = float(snap.get("cash_weight", 0.0))
        w.name = pd.Timestamp(dt)
        rows.append(w)

    if len(rows) == 0:
        cols = list(price.columns)
        if include_cash and cash_column_name not in cols:
            cols.append(cash_column_name)
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=cols, dtype=float)

    weight_df = pd.DataFrame(rows)
    weight_df.index.name = "rebalance_date"

    cols = list(price.columns)
    if include_cash and cash_column_name not in cols:
        cols.append(cash_column_name)

    weight_df = weight_df.reindex(columns=cols).fillna(0.0)
    return weight_df


# ============================================================
# 一站式入口：从长表到双动量目标权重
# ============================================================


def calc_dual_momentum_snapshot_from_long_data(
    data: pd.DataFrame,
    date_col: str = "trade_date",
    code_col: str = "ts_code",
    price_col: str = "close",
    date_format: Optional[str] = "%Y%m%d",
    calendar: Optional[Sequence] = None,
    ffill: bool = True,
    ffill_limit: Optional[int] = 5,
    min_non_na_ratio: float = 0.8,
    drop_all_na_dates: bool = True,
    **snapshot_kwargs,
) -> dict:
    """
    从长表直接计算某一时点的双动量快照。
    """
    price = build_price_matrix(
        data=data,
        date_col=date_col,
        code_col=code_col,
        price_col=price_col,
        date_format=date_format,
    )
    price = prepare_price_matrix(
        price_df=price,
        calendar=calendar,
        ffill=ffill,
        ffill_limit=ffill_limit,
        min_non_na_ratio=min_non_na_ratio,
        drop_all_na_dates=drop_all_na_dates,
    )
    return calc_dual_momentum_snapshot(price, **snapshot_kwargs)



def calc_dual_momentum_weights_from_long_data(
    data: pd.DataFrame,
    date_col: str = "trade_date",
    code_col: str = "ts_code",
    price_col: str = "close",
    date_format: Optional[str] = "%Y%m%d",
    calendar: Optional[Sequence] = None,
    ffill: bool = True,
    ffill_limit: Optional[int] = 5,
    min_non_na_ratio: float = 0.8,
    drop_all_na_dates: bool = True,
    rebalance_dates: Optional[Sequence] = None,
    freq: str = "M",
    warmup: Optional[int] = None,
    include_cash: bool = False,
    cash_column_name: str = "CASH",
    **snapshot_kwargs,
) -> pd.DataFrame:
    """
    从长表直接批量计算双动量目标权重矩阵。
    """
    price = build_price_matrix(
        data=data,
        date_col=date_col,
        code_col=code_col,
        price_col=price_col,
        date_format=date_format,
    )
    price = prepare_price_matrix(
        price_df=price,
        calendar=calendar,
        ffill=ffill,
        ffill_limit=ffill_limit,
        min_non_na_ratio=min_non_na_ratio,
        drop_all_na_dates=drop_all_na_dates,
    )
    return calc_dual_momentum_weights_over_time(
        price_df=price,
        rebalance_dates=rebalance_dates,
        freq=freq,
        warmup=warmup,
        include_cash=include_cash,
        cash_column_name=cash_column_name,
        **snapshot_kwargs,
    )


# ============================================================
# 分析输出
# ============================================================


def make_dual_momentum_report(
    snapshot_result: dict,
) -> pd.DataFrame:
    """
    将 calc_dual_momentum_snapshot 的 detail 表整理输出。
    """
    detail = snapshot_result["detail"].copy()
    detail.insert(0, "as_of_date", snapshot_result["as_of_date"])
    detail["market_regime_on"] = bool(snapshot_result["market_regime_on"])
    detail["cash_weight"] = float(snapshot_result.get("cash_weight", 0.0))
    return detail



def summarize_price_preparation(
    raw_price_df: pd.DataFrame,
    prepared_price_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    汇总价格矩阵整理前后的信息。
    """
    raw = ensure_datetime_index(raw_price_df)
    prep = ensure_datetime_index(prepared_price_df)

    rows = {
        "raw_rows": raw.shape[0],
        "raw_assets": raw.shape[1],
        "prepared_rows": prep.shape[0],
        "prepared_assets": prep.shape[1],
        "raw_missing_ratio": raw.isna().mean().mean(),
        "prepared_missing_ratio": prep.isna().mean().mean(),
        "raw_start": raw.index.min(),
        "raw_end": raw.index.max(),
        "prepared_start": prep.index.min(),
        "prepared_end": prep.index.max(),
    }
    return pd.DataFrame([rows])


# ============================================================
# 示例
# ============================================================

if __name__ == "__main__":
    rng = pd.date_range("2024-01-01", periods=220, freq="B")
    df = pd.DataFrame({
        "trade_date": list(rng.strftime("%Y%m%d")) * 4,
        "ts_code": ["510300.SH"] * len(rng) + ["159915.SZ"] * len(rng) + ["512880.SH"] * len(rng) + ["511010.SH"] * len(rng),
        "close": np.concatenate([
            np.linspace(3.5, 4.3, len(rng)),
            np.linspace(1.5, 2.4, len(rng)) * (1 + 0.02 * np.sin(np.arange(len(rng)) / 8)),
            np.linspace(0.9, 1.1, len(rng)) * (1 + 0.03 * np.sin(np.arange(len(rng)) / 5)),
            np.linspace(10.0, 10.3, len(rng)),
        ])
    })

    price = build_price_matrix(df)
    snap = calc_dual_momentum_snapshot(
        price_df=price,
        candidate_assets=["510300.SH", "159915.SZ", "512880.SH"],
        defensive_assets=["511010.SH"],
        abs_lookback=60,
        rel_lookback=60,
        top_k=2,
        weighting="equal",
    )
    print(snap["detail"])
    print(snap["target_weights"][snap["target_weights"] > 0])
