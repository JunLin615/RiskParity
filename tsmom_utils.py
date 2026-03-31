from __future__ import annotations

from typing import Optional, Sequence
import warnings

import numpy as np
import pandas as pd


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
    从 market_data.py 风格的长表构建价格矩阵。

    参数
    ----
    data : DataFrame
        至少包含 date_col, code_col, price_col
    date_col : str
        日期列名
    code_col : str
        资产代码列名
    price_col : str
        价格列名
    date_format : str | None
        日期解析格式；如果为 None，则交给 pandas 自动解析

    返回
    ----
    DataFrame
        index = DatetimeIndex
        columns = ts_code
        values = price
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
# 收益率、风险自由利率与超额收益
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



def annual_rate_to_daily_return(
    annual_rate: float,
    annualization: int = 252,
    continuous: bool = False,
) -> float:
    """
    将年化利率转换为单日收益率。

    参数
    ----
    annual_rate : float
        年化利率，例如 0.02 表示 2%
    annualization : int
        年化天数，默认 252
    continuous : bool
        是否将 annual_rate 视为连续复利年化利率
    """
    if continuous:
        return float(np.exp(annual_rate / annualization) - 1.0)
    return float((1.0 + annual_rate) ** (1.0 / annualization) - 1.0)



def build_constant_risk_free_series(
    index: Sequence,
    annual_rate: float = 0.0,
    annualization: int = 252,
    continuous: bool = False,
    name: str = "rf",
) -> pd.Series:
    """
    构建常数风险自由利率序列（日频）。
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(index)))
    daily_rf = annual_rate_to_daily_return(
        annual_rate=annual_rate,
        annualization=annualization,
        continuous=continuous,
    )
    return pd.Series(daily_rf, index=idx, name=name, dtype=float)



def align_risk_free_series(
    rf: pd.Series,
    index: Sequence,
    fill_method: str = "ffill",
) -> pd.Series:
    """
    将风险自由利率序列对齐到目标日期轴。
    """
    target_index = pd.DatetimeIndex(pd.to_datetime(pd.Index(index)))
    out = rf.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index().reindex(target_index)

    if fill_method == "ffill":
        out = out.ffill()
    elif fill_method == "bfill":
        out = out.bfill()
    elif fill_method == "none":
        pass
    else:
        raise ValueError("fill_method must be 'ffill', 'bfill', or 'none'")

    return out



def calc_excess_returns(
    returns_df: pd.DataFrame,
    rf: Optional[pd.Series | float] = None,
    annual_rf: Optional[float] = None,
    annualization: int = 252,
    rf_continuous: bool = False,
) -> pd.DataFrame:
    """
    计算超额收益率。

    参数
    ----
    returns_df : DataFrame
        资产收益率矩阵（日频）
    rf : Series | float | None
        若为 Series，则视为日频风险自由利率序列；若为 float，则视为日频常数利率
    annual_rf : float | None
        若给定，则将其视为年化无风险利率并转换为日频
    annualization : int
        年化天数
    rf_continuous : bool
        annual_rf 是否按连续复利处理
    """
    ret = ensure_datetime_index(returns_df)

    if rf is not None and annual_rf is not None:
        raise ValueError("rf and annual_rf cannot both be provided")

    if annual_rf is not None:
        rf_series = build_constant_risk_free_series(
            index=ret.index,
            annual_rate=annual_rf,
            annualization=annualization,
            continuous=rf_continuous,
        )
    elif rf is None:
        rf_series = pd.Series(0.0, index=ret.index, name="rf")
    elif np.isscalar(rf):
        rf_series = pd.Series(float(rf), index=ret.index, name="rf")
    else:
        rf_series = align_risk_free_series(rf, ret.index, fill_method="ffill")

    return ret.sub(rf_series, axis=0)


# ============================================================
# 动量信号计算
# ============================================================


def _prepare_returns_for_momentum(
    price_df: Optional[pd.DataFrame] = None,
    returns_df: Optional[pd.DataFrame] = None,
    return_type: str = "simple",
) -> pd.DataFrame:
    if price_df is None and returns_df is None:
        raise ValueError("price_df and returns_df cannot both be None")

    if returns_df is not None:
        ret = ensure_datetime_index(returns_df)
    else:
        price = ensure_datetime_index(price_df)  # type: ignore[arg-type]
        if return_type == "simple":
            ret = calc_simple_returns(price)
        elif return_type == "log":
            ret = calc_log_returns(price)
        else:
            raise ValueError("return_type must be 'simple' or 'log'")
    return ret



def calc_past_returns(
    price_df: Optional[pd.DataFrame] = None,
    returns_df: Optional[pd.DataFrame] = None,
    lookback: int = 252,
    skip_recent: int = 0,
    return_type: str = "simple",
    output_type: str = "simple",
) -> pd.DataFrame:
    """
    计算过去 lookback 个交易日的累计收益率。

    参数
    ----
    price_df : DataFrame | None
        价格矩阵
    returns_df : DataFrame | None
        收益率矩阵；若提供，则优先使用
    lookback : int
        动量观察窗口，例如 252 表示近 12 个月交易日
    skip_recent : int
        跳过最近若干日，例如 21 表示做 12-1 动量
    return_type : {"simple", "log"}
        当只提供 price_df 时，使用哪种方式生成收益率矩阵
    output_type : {"simple", "log"}
        输出 simple 累计收益率或 log 累计收益率
    """
    ret = _prepare_returns_for_momentum(price_df=price_df, returns_df=returns_df, return_type=return_type)

    if return_type == "simple":
        cum_input = np.log1p(ret)
    else:
        cum_input = ret.copy()

    if skip_recent > 0:
        cum_input = cum_input.shift(skip_recent)

    cum_log = cum_input.rolling(window=lookback, min_periods=lookback).sum()

    if output_type == "log":
        return cum_log
    if output_type == "simple":
        return np.expm1(cum_log)
    raise ValueError("output_type must be 'simple' or 'log'")



def calc_excess_past_returns(
    price_df: Optional[pd.DataFrame] = None,
    returns_df: Optional[pd.DataFrame] = None,
    lookback: int = 252,
    skip_recent: int = 0,
    return_type: str = "simple",
    output_type: str = "simple",
    rf: Optional[pd.Series | float] = None,
    annual_rf: Optional[float] = None,
    annualization: int = 252,
    rf_continuous: bool = False,
) -> pd.DataFrame:
    """
    计算过去 lookback 个交易日的累计超额收益率。
    """
    ret = _prepare_returns_for_momentum(price_df=price_df, returns_df=returns_df, return_type=return_type)
    excess = calc_excess_returns(
        ret,
        rf=rf,
        annual_rf=annual_rf,
        annualization=annualization,
        rf_continuous=rf_continuous,
    )
    return calc_past_returns(
        returns_df=excess,
        lookback=lookback,
        skip_recent=skip_recent,
        return_type="simple",
        output_type=output_type,
    )



def calc_tsmom_raw_signal(
    price_df: Optional[pd.DataFrame] = None,
    returns_df: Optional[pd.DataFrame] = None,
    lookback: int = 252,
    skip_recent: int = 0,
    return_type: str = "simple",
    use_excess_returns: bool = True,
    rf: Optional[pd.Series | float] = None,
    annual_rf: Optional[float] = None,
    annualization: int = 252,
    rf_continuous: bool = False,
) -> pd.DataFrame:
    """
    计算 TSMOM 原始信号（累计收益率）。

    默认使用累计超额收益率，更贴近 AQR / TSMOM 文献中的思路；
    若不需要超额收益，可设置 use_excess_returns=False。
    """
    if use_excess_returns:
        return calc_excess_past_returns(
            price_df=price_df,
            returns_df=returns_df,
            lookback=lookback,
            skip_recent=skip_recent,
            return_type=return_type,
            output_type="simple",
            rf=rf,
            annual_rf=annual_rf,
            annualization=annualization,
            rf_continuous=rf_continuous,
        )

    return calc_past_returns(
        price_df=price_df,
        returns_df=returns_df,
        lookback=lookback,
        skip_recent=skip_recent,
        return_type=return_type,
        output_type="simple",
    )



def calc_tsmom_sign_signal(
    price_df: Optional[pd.DataFrame] = None,
    returns_df: Optional[pd.DataFrame] = None,
    lookback: int = 252,
    skip_recent: int = 0,
    return_type: str = "simple",
    use_excess_returns: bool = True,
    rf: Optional[pd.Series | float] = None,
    annual_rf: Optional[float] = None,
    annualization: int = 252,
    rf_continuous: bool = False,
    zero_to_nan: bool = False,
) -> pd.DataFrame:
    """
    计算符号型 TSMOM 信号：
        signal = sign(past return)
    """
    raw_signal = calc_tsmom_raw_signal(
        price_df=price_df,
        returns_df=returns_df,
        lookback=lookback,
        skip_recent=skip_recent,
        return_type=return_type,
        use_excess_returns=use_excess_returns,
        rf=rf,
        annual_rf=annual_rf,
        annualization=annualization,
        rf_continuous=rf_continuous,
    )

    signal = np.sign(raw_signal)
    if zero_to_nan:
        signal = signal.where(signal != 0)
    return signal.astype(float)



def calc_tsmom_standardized_signal(
    raw_signal_df: pd.DataFrame,
    method: str = "cross_sectional_zscore",
    clip: Optional[float] = 3.0,
) -> pd.DataFrame:
    """
    对原始动量信号进行标准化。

    参数
    ----
    raw_signal_df : DataFrame
        一般为累计收益率矩阵
    method : {"cross_sectional_zscore", "time_series_zscore", "rank"}
        标准化方式
    clip : float | None
        若不为 None，则对标准化后的信号截断
    """
    x = ensure_datetime_index(raw_signal_df)

    if method == "cross_sectional_zscore":
        mean = x.mean(axis=1)
        std = x.std(axis=1, ddof=1).replace(0.0, np.nan)
        out = x.sub(mean, axis=0).div(std, axis=0)
    elif method == "time_series_zscore":
        mean = x.expanding(min_periods=20).mean()
        std = x.expanding(min_periods=20).std(ddof=1).replace(0.0, np.nan)
        out = (x - mean) / std
    elif method == "rank":
        rank = x.rank(axis=1, method="average", pct=True)
        out = rank * 2.0 - 1.0
    else:
        raise ValueError("method must be 'cross_sectional_zscore', 'time_series_zscore', or 'rank'")

    if clip is not None:
        out = out.clip(lower=-abs(clip), upper=abs(clip))
    return out


# ============================================================
# 多周期信号组合
# ============================================================


def _coerce_horizons(lookback: int | Sequence[int]) -> list[int]:
    if np.isscalar(lookback):
        horizons = [int(lookback)]
    else:
        horizons = [int(x) for x in lookback]

    if len(horizons) == 0:
        raise ValueError("lookback cannot be empty")
    if any(x <= 0 for x in horizons):
        raise ValueError("all lookback values must be positive")
    return horizons


def _broadcast_int_param(
    value: int | Sequence[int],
    horizons: Sequence[int],
    name: str,
) -> list[int]:
    if np.isscalar(value):
        out = [int(value)] * len(horizons)
    else:
        out = [int(x) for x in value]
        if len(out) != len(horizons):
            raise ValueError(f"{name} length must match lookback length")

    if any(x < 0 for x in out):
        raise ValueError(f"all {name} values must be >= 0")
    return out


def _normalize_horizon_weights(
    horizons: Sequence[int],
    horizon_weights: Optional[Sequence[float]] = None,
) -> pd.Series:
    idx = pd.Index([int(x) for x in horizons], name="lookback")

    if horizon_weights is None:
        w = pd.Series(1.0, index=idx, dtype=float)
    else:
        w = pd.Series([float(x) for x in horizon_weights], index=idx, dtype=float)
        if len(w) != len(idx):
            raise ValueError("horizon_weights length must match lookback length")

    total = float(w.abs().sum())
    if np.isclose(total, 0.0):
        raise ValueError("horizon_weights absolute sum is zero")
    return w / total


def _transform_raw_signal(
    raw_signal_df: pd.DataFrame,
    signal_type: str = "sign",
    clip: Optional[float] = 3.0,
    zero_to_nan: bool = False,
) -> pd.DataFrame:
    if signal_type == "sign":
        out = np.sign(raw_signal_df)
        if zero_to_nan:
            out = out.where(out != 0)
        return out.astype(float)

    if signal_type == "raw":
        return raw_signal_df.copy().astype(float)

    if signal_type == "rank":
        return calc_tsmom_standardized_signal(raw_signal_df, method="rank", clip=None).astype(float)

    if signal_type == "cross_sectional_zscore":
        return calc_tsmom_standardized_signal(raw_signal_df, method="cross_sectional_zscore", clip=clip).astype(float)

    if signal_type == "time_series_zscore":
        return calc_tsmom_standardized_signal(raw_signal_df, method="time_series_zscore", clip=clip).astype(float)

    raise ValueError(
        "signal_type must be 'sign', 'raw', 'rank', 'cross_sectional_zscore', or 'time_series_zscore'"
    )


def concat_horizon_frames(
    frames_by_horizon: dict[int, pd.DataFrame],
    level_name: str = "lookback",
) -> pd.DataFrame:
    if len(frames_by_horizon) == 0:
        return pd.DataFrame()
    return pd.concat(frames_by_horizon, axis=1, names=[level_name, "ts_code"])


def combine_horizon_frames(
    frames_by_horizon: dict[int, pd.DataFrame],
    method: str = "mean",
    horizon_weights: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    将多个 horizon 的 DataFrame 组合为一个综合信号矩阵。

    参数
    ----
    method : {"mean", "weighted_mean", "sum", "median", "vote"}
        - mean / weighted_mean: 对有效 horizon 按权重平均
        - sum: 按权重求和，不做分母归一化
        - median: 逐格取中位数
        - vote: 对各 horizon 的 sign 做加权投票，输出 -1/0/1
    """
    if len(frames_by_horizon) == 0:
        return pd.DataFrame()

    horizons = list(frames_by_horizon.keys())
    weights = _normalize_horizon_weights(horizons, horizon_weights=horizon_weights)

    aligned = {int(h): ensure_datetime_index(df).astype(float) for h, df in frames_by_horizon.items()}
    union_index = pd.DatetimeIndex([])
    union_columns = pd.Index([])
    for h in horizons:
        union_index = union_index.union(aligned[h].index)
        union_columns = union_columns.union(aligned[h].columns)

    arr_list = []
    for h in horizons:
        arr_list.append(aligned[h].reindex(index=union_index, columns=union_columns).values.astype(float))

    panel = np.stack(arr_list, axis=0)
    valid = np.isfinite(panel)
    w = weights.reindex(horizons).values.astype(float)[:, None, None]

    if method in {"mean", "weighted_mean"}:
        numerator = np.nansum(panel * w, axis=0)
        denominator = np.nansum(valid * w, axis=0)
        out = np.full_like(numerator, np.nan, dtype=float)
        np.divide(numerator, denominator, out=out, where=denominator > 0)
    elif method == "sum":
        out = np.nansum(panel * w, axis=0)
        out[~np.any(valid, axis=0)] = np.nan
    elif method == "median":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out = np.nanmedian(panel, axis=0)
        out[~np.any(valid, axis=0)] = np.nan
    elif method == "vote":
        vote_panel = np.sign(panel)
        numerator = np.nansum(vote_panel * w, axis=0)
        denominator = np.nansum(valid * w, axis=0)
        avg_vote = np.full_like(numerator, np.nan, dtype=float)
        np.divide(numerator, denominator, out=avg_vote, where=denominator > 0)
        out = np.sign(avg_vote)
    else:
        raise ValueError("method must be 'mean', 'weighted_mean', 'sum', 'median', or 'vote'")

    return pd.DataFrame(out, index=union_index, columns=union_columns)


def calc_multi_horizon_tsmom_raw_signals(
    price_df: Optional[pd.DataFrame] = None,
    returns_df: Optional[pd.DataFrame] = None,
    lookback: int | Sequence[int] = 252,
    skip_recent: int | Sequence[int] = 0,
    return_type: str = "simple",
    use_excess_returns: bool = True,
    rf: Optional[pd.Series | float] = None,
    annual_rf: Optional[float] = None,
    annualization: int = 252,
    rf_continuous: bool = False,
) -> dict[int, pd.DataFrame]:
    horizons = _coerce_horizons(lookback)
    skips = _broadcast_int_param(skip_recent, horizons, name="skip_recent")

    out: dict[int, pd.DataFrame] = {}
    for h, s in zip(horizons, skips):
        out[int(h)] = calc_tsmom_raw_signal(
            price_df=price_df,
            returns_df=returns_df,
            lookback=int(h),
            skip_recent=int(s),
            return_type=return_type,
            use_excess_returns=use_excess_returns,
            rf=rf,
            annual_rf=annual_rf,
            annualization=annualization,
            rf_continuous=rf_continuous,
        )
    return out


def calc_multi_horizon_tsmom_signals(
    price_df: Optional[pd.DataFrame] = None,
    returns_df: Optional[pd.DataFrame] = None,
    lookback: int | Sequence[int] = 252,
    skip_recent: int | Sequence[int] = 0,
    return_type: str = "simple",
    signal_type: str = "sign",
    use_excess_returns: bool = True,
    rf: Optional[pd.Series | float] = None,
    annual_rf: Optional[float] = None,
    annualization: int = 252,
    rf_continuous: bool = False,
    signal_clip: Optional[float] = 3.0,
    zero_to_nan: bool = False,
) -> dict[int, pd.DataFrame]:
    raw_dict = calc_multi_horizon_tsmom_raw_signals(
        price_df=price_df,
        returns_df=returns_df,
        lookback=lookback,
        skip_recent=skip_recent,
        return_type=return_type,
        use_excess_returns=use_excess_returns,
        rf=rf,
        annual_rf=annual_rf,
        annualization=annualization,
        rf_continuous=rf_continuous,
    )

    out: dict[int, pd.DataFrame] = {}
    for h, raw_df in raw_dict.items():
        out[int(h)] = _transform_raw_signal(
            raw_signal_df=raw_df,
            signal_type=signal_type,
            clip=signal_clip,
            zero_to_nan=zero_to_nan,
        )
    return out


# ============================================================
# 波动率估计
# ============================================================


def calc_rolling_volatility(
    returns_df: pd.DataFrame,
    window: int = 63,
    annualization: int = 252,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    计算滚动年化波动率矩阵。
    """
    ret = ensure_datetime_index(returns_df)
    if min_periods is None:
        min_periods = window
    vol = ret.rolling(window=window, min_periods=min_periods).std(ddof=1)
    return vol * np.sqrt(annualization)



def calc_ewma_volatility(
    returns_df: pd.DataFrame,
    lam: float = 0.94,
    annualization: int = 252,
    min_periods: int = 20,
) -> pd.DataFrame:
    """
    计算 EWMA 年化波动率矩阵。

    说明
    ----
    这里采用 pandas ewm 实现：
        sigma_t = std_ewm(r_t)
    对日度收益率做指数加权，再乘以 sqrt(annualization) 年化。
    """
    ret = ensure_datetime_index(returns_df)
    alpha = 1.0 - lam
    vol = ret.ewm(alpha=alpha, adjust=False, min_periods=min_periods).std(bias=False)
    return vol * np.sqrt(annualization)



def apply_vol_floor(
    vol_df: pd.DataFrame,
    vol_floor: Optional[float] = 1e-8,
) -> pd.DataFrame:
    """
    对波动率设置下限，避免仓位缩放时出现极端杠杆。
    """
    vol = vol_df.copy()
    if vol_floor is not None:
        vol = vol.clip(lower=vol_floor)
    return vol


# ============================================================
# 仓位生成与约束
# ============================================================


def lag_signal(
    signal_df: pd.DataFrame,
    periods: int = 1,
) -> pd.DataFrame:
    """
    将信号滞后若干期，避免前视偏差。
    """
    signal = ensure_datetime_index(signal_df)
    return signal.shift(periods)



def scale_signal_to_target_vol(
    signal_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    target_vol: float = 0.40,
    max_abs_position: Optional[float] = None,
    min_vol: float = 1e-8,
) -> pd.DataFrame:
    """
    将信号按单资产目标波动率缩放：
        position = signal * target_vol / vol

    这一步只做“每个资产自身”的波动率目标控制，
    不做组合层面的再归一化。
    """
    signal = ensure_datetime_index(signal_df)
    vol = ensure_datetime_index(vol_df)
    vol = vol.reindex_like(signal)
    vol = vol.clip(lower=min_vol)

    position = signal * (target_vol / vol)

    if max_abs_position is not None:
        position = position.clip(lower=-abs(max_abs_position), upper=abs(max_abs_position))

    return position



def apply_abs_position_cap(
    position_df: pd.DataFrame,
    cap: float = 1.0,
) -> pd.DataFrame:
    """
    对仓位矩阵施加绝对值上限。
    """
    x = position_df.copy()
    return x.clip(lower=-abs(cap), upper=abs(cap))



def normalize_positions_to_target_gross(
    position_df: pd.DataFrame,
    target_gross: float = 1.0,
    min_active_assets: int = 1,
) -> pd.DataFrame:
    """
    按日期将仓位矩阵归一化到目标总杠杆：
        gross_t = sum_i |w_{t,i}|

    当某日有效资产数小于 min_active_assets 时，该日保持 NaN。
    """
    pos = ensure_datetime_index(position_df)
    gross = pos.abs().sum(axis=1)
    active_assets = pos.notna().sum(axis=1)

    scale = target_gross / gross.replace(0.0, np.nan)
    scale = scale.where(active_assets >= min_active_assets)
    return pos.mul(scale, axis=0)



def mask_positions_by_data_availability(
    position_df: pd.DataFrame,
    price_df: pd.DataFrame,
    require_current_price: bool = True,
) -> pd.DataFrame:
    """
    根据价格矩阵可用性屏蔽不可交易资产仓位。
    """
    pos = ensure_datetime_index(position_df)
    price = ensure_datetime_index(price_df).reindex_like(pos)

    if require_current_price:
        mask = price.notna()
    else:
        mask = price.notna() | price.shift(1).notna()

    return pos.where(mask)



def calc_position_turnover(
    position_df: pd.DataFrame,
    one_way: bool = True,
) -> pd.Series:
    """
    计算仓位换手率。

    参数
    ----
    one_way : bool
        True  -> 0.5 * sum(|dw|)
        False -> sum(|dw|)
    """
    pos = ensure_datetime_index(position_df)
    delta = pos.diff().abs().sum(axis=1)
    if one_way:
        delta = delta * 0.5
    return delta.rename("turnover")


# ============================================================
# 一站式 TSMOM 信号 / 仓位 / 门控构建
# ============================================================


def _normalize_side(side: str = "long_short") -> str:
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


def apply_side_rules(
    signal_or_position_df: pd.DataFrame,
    side: str = "long_short",
) -> pd.DataFrame:
    x = ensure_datetime_index(signal_or_position_df).astype(float)
    side = _normalize_side(side)

    if side == "long_short":
        return x
    if side == "long_only":
        return x.clip(lower=0.0)
    return x.clip(upper=0.0)


def apply_signal_threshold(
    signal_df: pd.DataFrame,
    threshold: float = 0.0,
) -> pd.DataFrame:
    x = ensure_datetime_index(signal_df).astype(float)
    if threshold <= 0:
        return x
    return x.where(x.abs() >= float(threshold), 0.0)


def build_tsmom_signal_bundle(
    price_df: pd.DataFrame,
    lookback: int | Sequence[int] = 252,
    skip_recent: int | Sequence[int] = 0,
    signal_type: str = "sign",
    use_excess_returns: bool = True,
    rf: Optional[pd.Series | float] = None,
    annual_rf: Optional[float] = None,
    return_type: str = "simple",
    annualization: int = 252,
    execution_lag: int = 1,
    combination_method: str = "mean",
    horizon_weights: Optional[Sequence[float]] = None,
    signal_clip: Optional[float] = 3.0,
    zero_to_nan: bool = False,
) -> dict[str, object]:
    """
    构建 TSMOM 信号总入口，兼容单周期与多周期。

    推荐用法
    --------
    - 单周期 baseline：lookback=252, signal_type="sign"
    - 多周期组合：lookback=[63, 126, 252], signal_type="sign", combination_method="mean"

    返回
    ----
    dict，包含：
    - price / returns
    - raw_signal: 综合原始信号
    - signal: 综合后的最终信号
    - execution_signal: 滞后后的可执行信号
    - horizon_raw_signal_dict / horizon_signal_dict
    - horizon_raw_signal_panel / horizon_signal_panel
    - horizons / horizon_weights
    """
    price = ensure_datetime_index(price_df)

    if return_type == "simple":
        returns = calc_simple_returns(price)
    elif return_type == "log":
        returns = calc_log_returns(price)
    else:
        raise ValueError("return_type must be 'simple' or 'log'")

    horizons = _coerce_horizons(lookback)
    horizon_weight_series = _normalize_horizon_weights(horizons, horizon_weights=horizon_weights)

    horizon_raw_signal_dict = calc_multi_horizon_tsmom_raw_signals(
        returns_df=returns,
        lookback=horizons,
        skip_recent=skip_recent,
        return_type="simple",
        use_excess_returns=use_excess_returns,
        rf=rf,
        annual_rf=annual_rf,
        annualization=annualization,
    )
    horizon_signal_dict = {
        int(h): _transform_raw_signal(
            raw_signal_df=raw_df,
            signal_type=signal_type,
            clip=signal_clip,
            zero_to_nan=zero_to_nan,
        )
        for h, raw_df in horizon_raw_signal_dict.items()
    }

    horizon_raw_signal_panel = concat_horizon_frames(horizon_raw_signal_dict)
    horizon_signal_panel = concat_horizon_frames(horizon_signal_dict)

    raw_signal = combine_horizon_frames(
        horizon_raw_signal_dict,
        method=combination_method,
        horizon_weights=horizon_weight_series.values,
    )
    signal = combine_horizon_frames(
        horizon_signal_dict,
        method=combination_method,
        horizon_weights=horizon_weight_series.values,
    )

    if execution_lag > 0:
        execution_signal = lag_signal(signal, periods=execution_lag)
    else:
        execution_signal = signal.copy()

    return {
        "price": price,
        "returns": returns,
        "raw_signal": raw_signal,
        "signal": signal,
        "execution_signal": execution_signal,
        "horizons": horizons,
        "horizon_weights": horizon_weight_series,
        "horizon_raw_signal_dict": horizon_raw_signal_dict,
        "horizon_signal_dict": horizon_signal_dict,
        "horizon_raw_signal_panel": horizon_raw_signal_panel,
        "horizon_signal_panel": horizon_signal_panel,
    }


def build_tsmom_gate_matrix(
    signal_df: pd.DataFrame,
    gate_type: str = "directional",
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    将 TSMOM 信号转换成门控矩阵，方便用于其它策略。

    gate_type
    ---------
    directional         -> {-1, 0, 1}
    long_only           -> {0, 1}
    short_only          -> {0, 1}
    continuous          -> 保留原连续信号，小信号归零
    continuous_long_only -> 保留正向连续信号，小信号归零
    """
    x = ensure_datetime_index(signal_df).astype(float)
    x = apply_signal_threshold(x, threshold=threshold)

    if gate_type == "directional":
        return np.sign(x).astype(float)
    if gate_type == "long_only":
        return (x > 0).astype(float)
    if gate_type == "short_only":
        return (x < 0).astype(float)
    if gate_type == "continuous":
        return x
    if gate_type == "continuous_long_only":
        return x.clip(lower=0.0)

    raise ValueError(
        "gate_type must be 'directional', 'long_only', 'short_only', 'continuous', or 'continuous_long_only'"
    )


def apply_gate_to_positions(
    position_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    mode: str = "multiply",
) -> pd.DataFrame:
    """
    将门控矩阵应用到任意仓位/权重矩阵。

    mode
    ----
    multiply           : 直接逐格相乘
    filter             : gate != 0 时保留，否则置 0
    directional_filter : 仅保留与 gate 方向一致的仓位
    """
    pos = ensure_datetime_index(position_df).astype(float)
    gate = ensure_datetime_index(gate_df).reindex(index=pos.index, columns=pos.columns).fillna(0.0).astype(float)

    if mode == "multiply":
        return pos * gate

    if mode == "filter":
        return pos.where(gate != 0.0, 0.0)

    if mode == "directional_filter":
        gate_sign = np.sign(gate)
        keep = ((pos > 0) & (gate_sign > 0)) | ((pos < 0) & (gate_sign < 0))
        return pos.where(keep, 0.0)

    raise ValueError("mode must be 'multiply', 'filter', or 'directional_filter'")


def build_tsmom_signal_only_positions(
    price_df: pd.DataFrame,
    lookback: int | Sequence[int] = 252,
    skip_recent: int | Sequence[int] = 0,
    signal_type: str = "sign",
    use_excess_returns: bool = True,
    rf: Optional[pd.Series | float] = None,
    annual_rf: Optional[float] = None,
    return_type: str = "simple",
    annualization: int = 252,
    execution_lag: int = 1,
    combination_method: str = "mean",
    horizon_weights: Optional[Sequence[float]] = None,
    signal_clip: Optional[float] = 3.0,
    zero_to_nan: bool = False,
    side: str = "long_short",
    signal_threshold: float = 0.0,
    max_abs_position: Optional[float] = None,
    normalize_to_gross: Optional[float] = None,
    use_price_mask: bool = True,
) -> dict[str, object]:
    """
    只做 TSMOM 信号处理，不做波动率缩放。

    适用场景
    --------
    1. 直接把 TSMOM 信号当作轻量仓位模板；
    2. 给其它策略做门控（gate）；
    3. 对比“纯信号版”与“波动率缩放版”的差异。
    """
    bundle = build_tsmom_signal_bundle(
        price_df=price_df,
        lookback=lookback,
        skip_recent=skip_recent,
        signal_type=signal_type,
        use_excess_returns=use_excess_returns,
        rf=rf,
        annual_rf=annual_rf,
        return_type=return_type,
        annualization=annualization,
        execution_lag=execution_lag,
        combination_method=combination_method,
        horizon_weights=horizon_weights,
        signal_clip=signal_clip,
        zero_to_nan=zero_to_nan,
    )

    signal_position = ensure_datetime_index(bundle["execution_signal"]).astype(float)
    signal_position = apply_signal_threshold(signal_position, threshold=signal_threshold)
    signal_position = apply_side_rules(signal_position, side=side)

    if max_abs_position is not None:
        signal_position = apply_abs_position_cap(signal_position, cap=max_abs_position)

    final_position = signal_position.copy()
    if use_price_mask:
        final_position = mask_positions_by_data_availability(final_position, ensure_datetime_index(bundle["price"]))

    if normalize_to_gross is not None:
        final_position = normalize_positions_to_target_gross(final_position, target_gross=normalize_to_gross)

    out = dict(bundle)
    out["signal_position"] = signal_position
    out["scaled_position"] = signal_position.copy()
    out["final_position"] = final_position
    return out


def build_tsmom_gate_bundle(
    price_df: pd.DataFrame,
    lookback: int | Sequence[int] = 252,
    skip_recent: int | Sequence[int] = 0,
    signal_type: str = "sign",
    use_excess_returns: bool = True,
    rf: Optional[pd.Series | float] = None,
    annual_rf: Optional[float] = None,
    return_type: str = "simple",
    annualization: int = 252,
    execution_lag: int = 1,
    combination_method: str = "mean",
    horizon_weights: Optional[Sequence[float]] = None,
    signal_clip: Optional[float] = 3.0,
    zero_to_nan: bool = False,
    gate_type: str = "directional",
    gate_threshold: float = 0.0,
) -> dict[str, object]:
    bundle = build_tsmom_signal_bundle(
        price_df=price_df,
        lookback=lookback,
        skip_recent=skip_recent,
        signal_type=signal_type,
        use_excess_returns=use_excess_returns,
        rf=rf,
        annual_rf=annual_rf,
        return_type=return_type,
        annualization=annualization,
        execution_lag=execution_lag,
        combination_method=combination_method,
        horizon_weights=horizon_weights,
        signal_clip=signal_clip,
        zero_to_nan=zero_to_nan,
    )

    gate = build_tsmom_gate_matrix(
        signal_df=ensure_datetime_index(bundle["execution_signal"]),
        gate_type=gate_type,
        threshold=gate_threshold,
    )

    out = dict(bundle)
    out["gate"] = gate
    out["final_position"] = gate
    return out


def build_tsmom_positions(
    price_df: pd.DataFrame,
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
    execution_lag: int = 1,
    normalize_to_gross: Optional[float] = None,
    vol_floor: float = 1e-8,
    combination_method: str = "mean",
    horizon_weights: Optional[Sequence[float]] = None,
    signal_clip: Optional[float] = 3.0,
    zero_to_nan: bool = False,
) -> dict[str, object]:
    """
    从价格矩阵构建 TSMOM 相关核心输出。

    兼容性说明
    ----------
    - 当 lookback 为单个整数时，行为与旧版基本一致；
    - 当 lookback 为多个周期时，会先对每个周期分别生成信号，再做组合；
    - 推荐多周期 baseline：lookback=[63, 126, 252], signal_type="sign", combination_method="mean"。
    """
    bundle = build_tsmom_signal_bundle(
        price_df=price_df,
        lookback=lookback,
        skip_recent=skip_recent,
        signal_type=signal_type,
        use_excess_returns=use_excess_returns,
        rf=rf,
        annual_rf=annual_rf,
        return_type=return_type,
        annualization=annualization,
        execution_lag=execution_lag,
        combination_method=combination_method,
        horizon_weights=horizon_weights,
        signal_clip=signal_clip,
        zero_to_nan=zero_to_nan,
    )

    price = ensure_datetime_index(bundle["price"])
    returns = ensure_datetime_index(bundle["returns"])
    exec_signal = ensure_datetime_index(bundle["execution_signal"])

    if vol_method == "rolling":
        vol = calc_rolling_volatility(returns, window=vol_window, annualization=annualization)
    elif vol_method == "ewma":
        vol = calc_ewma_volatility(returns, lam=ewma_lambda, annualization=annualization)
    else:
        raise ValueError("vol_method must be 'rolling' or 'ewma'")

    vol = apply_vol_floor(vol, vol_floor=vol_floor)

    if execution_lag > 0:
        exec_vol = lag_signal(vol, periods=execution_lag)
    else:
        exec_vol = vol.copy()

    scaled_position = scale_signal_to_target_vol(
        signal_df=exec_signal,
        vol_df=exec_vol,
        target_vol=target_vol,
        max_abs_position=max_abs_position,
        min_vol=vol_floor,
    )

    final_position = mask_positions_by_data_availability(scaled_position, price)
    if normalize_to_gross is not None:
        final_position = normalize_positions_to_target_gross(final_position, target_gross=normalize_to_gross)

    out = dict(bundle)
    out["vol"] = vol
    out["execution_vol"] = exec_vol
    out["scaled_position"] = scaled_position
    out["final_position"] = final_position
    return out


# ============================================================
# 分析输出
# ============================================================


def summarize_signal_distribution(
    signal_df: pd.DataFrame,
    asof: Optional[str | pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    汇总某一日（或最后一日）信号分布。
    """
    signal = ensure_datetime_index(signal_df)
    if signal.empty:
        return pd.DataFrame(columns=["value"])

    if asof is None:
        s = signal.iloc[-1]
        dt = signal.index[-1]
    else:
        dt = pd.Timestamp(asof)
        s = signal.loc[dt]

    s = s.dropna()
    if len(s) == 0:
        return pd.DataFrame(columns=["value"])

    out = pd.DataFrame(
        {
            "stat": [
                "date",
                "count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
                "positive_count",
                "negative_count",
                "zero_count",
            ],
            "value": [
                dt,
                int(s.count()),
                float(s.mean()),
                float(s.std(ddof=1)) if s.count() > 1 else np.nan,
                float(s.min()),
                float(s.quantile(0.25)),
                float(s.quantile(0.50)),
                float(s.quantile(0.75)),
                float(s.max()),
                int((s > 0).sum()),
                int((s < 0).sum()),
                int((s == 0).sum()),
            ],
        }
    )
    return out.set_index("stat")



def summarize_position_matrix(
    position_df: pd.DataFrame,
    asof: Optional[str | pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    汇总某一日（或最后一日）仓位分布。
    """
    pos = ensure_datetime_index(position_df)
    if pos.empty:
        return pd.DataFrame(columns=["value"])

    if asof is None:
        s = pos.iloc[-1]
        dt = pos.index[-1]
    else:
        dt = pd.Timestamp(asof)
        s = pos.loc[dt]

    s = s.dropna()
    if len(s) == 0:
        return pd.DataFrame(columns=["value"])

    gross = float(s.abs().sum())
    net = float(s.sum())
    long_count = int((s > 0).sum())
    short_count = int((s < 0).sum())

    out = pd.DataFrame(
        {
            "stat": [
                "date",
                "count",
                "gross",
                "net",
                "mean",
                "std",
                "min",
                "max",
                "long_count",
                "short_count",
            ],
            "value": [
                dt,
                int(s.count()),
                gross,
                net,
                float(s.mean()),
                float(s.std(ddof=1)) if s.count() > 1 else np.nan,
                float(s.min()),
                float(s.max()),
                long_count,
                short_count,
            ],
        }
    )
    return out.set_index("stat")



def summarize_tsmom_pipeline(
    outputs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    汇总 build_tsmom_positions 输出的关键维度信息。
    """
    price = ensure_datetime_index(outputs["price"])
    ret = ensure_datetime_index(outputs["returns"])
    raw_signal = ensure_datetime_index(outputs["raw_signal"])
    final_position = ensure_datetime_index(outputs["final_position"])

    rows = {
        "price_rows": price.shape[0],
        "price_assets": price.shape[1],
        "return_rows": ret.shape[0],
        "signal_rows": raw_signal.shape[0],
        "position_rows": final_position.shape[0],
        "position_assets": final_position.shape[1],
        "price_start": price.index.min(),
        "price_end": price.index.max(),
        "position_start": final_position.index.min(),
        "position_end": final_position.index.max(),
        "price_missing_ratio": float(price.isna().mean().mean()),
        "position_missing_ratio": float(final_position.isna().mean().mean()),
    }
    return pd.DataFrame([rows])


__all__ = [
    "ensure_datetime_index",
    "build_price_matrix",
    "align_price_matrix",
    "forward_fill_prices",
    "prepare_price_matrix",
    "drop_assets_by_missing_ratio",
    "drop_dates_with_any_na",
    "select_last_n_rows",
    "calc_simple_returns",
    "calc_log_returns",
    "annual_rate_to_daily_return",
    "build_constant_risk_free_series",
    "align_risk_free_series",
    "calc_excess_returns",
    "calc_past_returns",
    "calc_excess_past_returns",
    "calc_tsmom_raw_signal",
    "calc_tsmom_sign_signal",
    "calc_tsmom_standardized_signal",
    "_coerce_horizons",
    "concat_horizon_frames",
    "combine_horizon_frames",
    "calc_multi_horizon_tsmom_raw_signals",
    "calc_multi_horizon_tsmom_signals",
    "calc_rolling_volatility",
    "calc_ewma_volatility",
    "apply_vol_floor",
    "lag_signal",
    "scale_signal_to_target_vol",
    "apply_abs_position_cap",
    "normalize_positions_to_target_gross",
    "mask_positions_by_data_availability",
    "calc_position_turnover",
    "apply_side_rules",
    "apply_signal_threshold",
    "build_tsmom_signal_bundle",
    "build_tsmom_gate_matrix",
    "apply_gate_to_positions",
    "build_tsmom_signal_only_positions",
    "build_tsmom_gate_bundle",
    "build_tsmom_positions",
    "summarize_signal_distribution",
    "summarize_position_matrix",
    "summarize_tsmom_pipeline",
]


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "trade_date": [
                "20240102", "20240103", "20240104", "20240105", "20240108",
                "20240102", "20240103", "20240104", "20240105", "20240108",
                "20240102", "20240103", "20240104", "20240105", "20240108",
            ],
            "ts_code": [
                "510300.SH", "510300.SH", "510300.SH", "510300.SH", "510300.SH",
                "511010.SH", "511010.SH", "511010.SH", "511010.SH", "511010.SH",
                "518880.SH", "518880.SH", "518880.SH", "518880.SH", "518880.SH",
            ],
            "close": [
                3.50, 3.52, 3.49, 3.56, 3.58,
                112.0, 112.1, 112.0, 112.2, 112.4,
                4.85, 4.90, 4.88, 4.95, 4.99,
            ],
        }
    )

    price = build_price_matrix(df)
    price = prepare_price_matrix(price, calendar=price.index)

    outputs = build_tsmom_positions(
        price_df=price,
        lookback=[2, 3],
        signal_type="sign",
        combination_method="mean",
        use_excess_returns=False,
        vol_method="rolling",
        vol_window=3,
        target_vol=0.2,
        execution_lag=1,
        normalize_to_gross=1.0,
    )

    signal_only = build_tsmom_signal_only_positions(
        price_df=price,
        lookback=[2, 3],
        signal_type="sign",
        combination_method="mean",
        use_excess_returns=False,
        execution_lag=1,
        normalize_to_gross=1.0,
    )

    print("summary")
    print(summarize_tsmom_pipeline(outputs))
    print()
    print("latest vol-scaled positions")
    print(outputs["final_position"].tail())
    print()
    print("latest signal-only positions")
    print(signal_only["final_position"].tail())
