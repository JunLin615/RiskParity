"""
risk_parity.py

纯函数版风险平价函数库
不依赖回测模块
主要提供：
1. 价格矩阵整理与缺失处理
2. 收益率、波动率、协方差计算
3. 逆波动率权重
4. 风险平价权重求解
5. 风险贡献分析

设计原则：
- 输入输出尽量使用 pandas / numpy
- 不和数据库、回测、下单逻辑耦合
- 优先适配日线级别 ETF / LOF / 指数类资产
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


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

    参数
    ----
    price_df : DataFrame
        价格矩阵，index 为日期
    calendar : Sequence | None
        统一日期轴；可传 DatetimeIndex/list/Series
        若为 None，则使用原 price_df.index

    返回
    ----
    DataFrame
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
    对价格矩阵做前向填充。
    只填中间缺口，不会填序列开头的缺失。

    参数
    ----
    price_df : DataFrame
        价格矩阵
    limit : int | None
        最多连续填充天数；None 表示不限制

    返回
    ----
    DataFrame
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

    参数
    ----
    price_df : DataFrame
        原始价格矩阵
    calendar : Sequence | None
        对齐日期轴
    ffill : bool
        是否前向填充
    ffill_limit : int | None
        前向填充最大连续天数
    min_non_na_ratio : float
        列保留阈值。资产非空比例低于该值则剔除
    drop_all_na_dates : bool
        是否删除所有资产都为空的日期

    返回
    ----
    DataFrame
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
# 收益率与风险估计
# ============================================================

def calc_simple_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算简单收益率：
        r_t = P_t / P_{t-1} - 1
    """
    price = ensure_datetime_index(price_df)
    ret = price.pct_change()
    return ret.dropna(how="all")


def calc_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算对数收益率：
        r_t = ln(P_t / P_{t-1})
    """
    price = ensure_datetime_index(price_df)
    ret = np.log(price / price.shift(1))
    return ret.dropna(how="all")


def winsorize_returns(
    returns_df: pd.DataFrame,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """
    对收益率按列做分位数截尾，降低极端值影响。
    """
    ret = returns_df.copy()
    for col in ret.columns:
        s = ret[col]
        lo = s.quantile(lower_q)
        hi = s.quantile(upper_q)
        ret[col] = s.clip(lower=lo, upper=hi)
    return ret


def calc_annualized_volatility(
    returns_df: pd.DataFrame,
    annualization: int = 252,
) -> pd.Series:
    """
    计算年化波动率。
    """
    ret = returns_df.copy()
    return ret.std(ddof=1) * np.sqrt(annualization)


def calc_covariance_matrix(
    returns_df: pd.DataFrame,
    annualization: int = 252,
) -> pd.DataFrame:
    """
    计算年化协方差矩阵。
    """
    ret = returns_df.copy()
    return ret.cov() * annualization


def calc_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算相关系数矩阵。
    """
    ret = returns_df.copy()
    return ret.corr()


def calc_ewma_variance(
    return_series: pd.Series,
    lam: float = 0.94,
    annualization: int = 252,
) -> float:
    """
    计算单资产 EWMA 年化方差。
    """
    x = return_series.dropna().values
    if len(x) == 0:
        return np.nan

    var = np.var(x[: min(20, len(x))], ddof=1) if len(x) > 1 else x[0] ** 2
    for r in x:
        var = lam * var + (1.0 - lam) * (r ** 2)
    return var * annualization


def calc_ewma_volatility(
    returns_df: pd.DataFrame,
    lam: float = 0.94,
    annualization: int = 252,
) -> pd.Series:
    """
    计算各资产 EWMA 年化波动率。
    """
    vals = {}
    for col in returns_df.columns:
        vals[col] = np.sqrt(calc_ewma_variance(returns_df[col], lam=lam, annualization=annualization))
    return pd.Series(vals)


def calc_ewma_covariance_matrix(
    returns_df: pd.DataFrame,
    lam: float = 0.94,
    annualization: int = 252,
) -> pd.DataFrame:
    """
    计算 EWMA 年化协方差矩阵。

    说明
    ----
    对每个时点使用：
        S_t = lam * S_{t-1} + (1-lam) * r_t r_t^T
    """
    ret = returns_df.dropna(how="any").copy()
    cols = ret.columns.tolist()
    x = ret.values

    if x.shape[0] == 0:
        return pd.DataFrame(np.nan, index=cols, columns=cols)

    if x.shape[0] == 1:
        outer = np.outer(x[0], x[0]) * annualization
        return pd.DataFrame(outer, index=cols, columns=cols)

    s = np.cov(x.T, ddof=1)
    for row in x:
        outer = np.outer(row, row)
        s = lam * s + (1.0 - lam) * outer

    s = s * annualization
    return pd.DataFrame(s, index=cols, columns=cols)


# ============================================================
# 风险平价核心计算
# ============================================================

def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """
    归一化权重，使其和为 1。
    """
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    return w / total


def equal_weights(n_assets: int) -> np.ndarray:
    """
    等权重。
    """
    return np.ones(n_assets, dtype=float) / n_assets


def inverse_volatility_weights(
    vol: pd.Series | np.ndarray,
) -> pd.Series | np.ndarray:
    """
    逆波动率权重。
    """
    if isinstance(vol, pd.Series):
        inv = 1.0 / vol
        w = inv / inv.sum()
        return w

    vol_arr = np.asarray(vol, dtype=float)
    inv = 1.0 / vol_arr
    return inv / inv.sum()


def calc_portfolio_variance(
    weights: Sequence[float],
    cov_matrix: pd.DataFrame | np.ndarray,
) -> float:
    """
    组合方差：
        w^T Σ w
    """
    w = np.asarray(weights, dtype=float)
    cov = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else np.asarray(cov_matrix, dtype=float)
    return float(w @ cov @ w)


def calc_portfolio_volatility(
    weights: Sequence[float],
    cov_matrix: pd.DataFrame | np.ndarray,
) -> float:
    """
    组合波动率：
        sqrt(w^T Σ w)
    """
    return float(np.sqrt(calc_portfolio_variance(weights, cov_matrix)))


def calc_marginal_risk_contribution(
    weights: Sequence[float],
    cov_matrix: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """
    边际风险贡献 MRC：
        MRC_i = (Σw)_i / sqrt(w^T Σ w)
    """
    w = np.asarray(weights, dtype=float)
    cov = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else np.asarray(cov_matrix, dtype=float)

    port_vol = calc_portfolio_volatility(w, cov)
    return (cov @ w) / port_vol


def calc_risk_contribution(
    weights: Sequence[float],
    cov_matrix: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """
    风险贡献 RC：
        RC_i = w_i * MRC_i
    """
    w = np.asarray(weights, dtype=float)
    mrc = calc_marginal_risk_contribution(w, cov_matrix)
    return w * mrc


def calc_relative_risk_contribution(
    weights: Sequence[float],
    cov_matrix: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """
    相对风险贡献 RRC：
        RRC_i = RC_i / sigma_p
    """
    rc = calc_risk_contribution(weights, cov_matrix)
    port_vol = calc_portfolio_volatility(weights, cov_matrix)
    return rc / port_vol


def risk_budget_objective(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    target_risk_budget: np.ndarray,
) -> float:
    """
    风险预算目标函数。
    最小化：
        sum((RC_i - b_i * sigma_p)^2)
    """
    rc = calc_risk_contribution(weights, cov_matrix)
    port_vol = calc_portfolio_volatility(weights, cov_matrix)
    target_rc = target_risk_budget * port_vol
    return float(np.sum((rc - target_rc) ** 2))


def solve_risk_parity_weights(
    cov_matrix: pd.DataFrame | np.ndarray,
    target_risk_budget: Optional[Sequence[float]] = None,
    initial_weights: Optional[Sequence[float]] = None,
    long_only: bool = True,
    weight_bounds: Optional[Sequence[tuple[float, float]]] = None,
    tol: float = 1e-12,
    maxiter: int = 10_000,
) -> np.ndarray:
    """
    求解风险平价 / 风险预算权重。

    参数
    ----
    cov_matrix : DataFrame | ndarray
        协方差矩阵
    target_risk_budget : Sequence[float] | None
        风险预算；None 表示等风险预算
    initial_weights : Sequence[float] | None
        初始权重；None 时默认使用逆波动率权重作为初值
    long_only : bool
        是否限制权重非负
    weight_bounds : Sequence[tuple[float, float]] | None
        自定义权重范围
    tol : float
        优化容忍度
    maxiter : int
        最大迭代次数

    返回
    ----
    ndarray
        最优权重
    """
    cov = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else np.asarray(cov_matrix, dtype=float)
    n = cov.shape[0]

    if target_risk_budget is None:
        b = np.ones(n, dtype=float) / n
    else:
        b = normalize_weights(target_risk_budget)

    if initial_weights is None:
        vol0 = np.sqrt(np.diag(cov))
        x0 = inverse_volatility_weights(vol0)
    else:
        x0 = normalize_weights(initial_weights)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if weight_bounds is not None:
        bounds = weight_bounds
    elif long_only:
        bounds = [(0.0, 1.0)] * n
    else:
        bounds = [(-1.0, 1.0)] * n

    result = minimize(
        fun=risk_budget_objective,
        x0=np.asarray(x0, dtype=float),
        args=(cov, b),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        tol=tol,
        options={"maxiter": maxiter, "disp": False},
    )

    w = normalize_weights(result.x)
    return w


# ============================================================
# 从价格到权重的一站式入口
# ============================================================

def estimate_covariance_from_prices(
    price_df: pd.DataFrame,
    method: str = "sample",
    return_type: str = "log",
    annualization: int = 252,
    winsorize: bool = False,
    winsor_lower_q: float = 0.01,
    winsor_upper_q: float = 0.99,
    ewma_lambda: float = 0.94,
    drop_any_na: bool = True,
) -> pd.DataFrame:
    """
    从价格矩阵估计协方差矩阵。

    参数
    ----
    method : {"sample", "ewma"}
        协方差估计方法
    return_type : {"log", "simple"}
        收益率类型
    """
    price = ensure_datetime_index(price_df)

    if return_type == "log":
        ret = calc_log_returns(price)
    elif return_type == "simple":
        ret = calc_simple_returns(price)
    else:
        raise ValueError("return_type must be 'log' or 'simple'")

    if winsorize:
        ret = winsorize_returns(ret, lower_q=winsor_lower_q, upper_q=winsor_upper_q)

    if drop_any_na:
        ret = ret.dropna(how="any")

    if method == "sample":
        return calc_covariance_matrix(ret, annualization=annualization)
    elif method == "ewma":
        return calc_ewma_covariance_matrix(ret, lam=ewma_lambda, annualization=annualization)
    else:
        raise ValueError("method must be 'sample' or 'ewma'")


def calc_inverse_vol_weights_from_prices(
    price_df: pd.DataFrame,
    return_type: str = "log",
    annualization: int = 252,
    use_ewma: bool = False,
    ewma_lambda: float = 0.94,
    drop_any_na: bool = True,
) -> pd.Series:
    """
    从价格矩阵直接计算逆波动率权重。
    """
    price = ensure_datetime_index(price_df)

    if return_type == "log":
        ret = calc_log_returns(price)
    elif return_type == "simple":
        ret = calc_simple_returns(price)
    else:
        raise ValueError("return_type must be 'log' or 'simple'")

    if drop_any_na:
        ret = ret.dropna(how="any")

    if use_ewma:
        vol = calc_ewma_volatility(ret, lam=ewma_lambda, annualization=annualization)
    else:
        vol = calc_annualized_volatility(ret, annualization=annualization)

    w = inverse_volatility_weights(vol)
    return pd.Series(w, index=vol.index, name="weight")


def calc_risk_parity_weights_from_prices(
    price_df: pd.DataFrame,
    method: str = "sample",
    return_type: str = "log",
    annualization: int = 252,
    target_risk_budget: Optional[Sequence[float]] = None,
    winsorize: bool = False,
    winsor_lower_q: float = 0.01,
    winsor_upper_q: float = 0.99,
    ewma_lambda: float = 0.94,
    long_only: bool = True,
    weight_bounds: Optional[Sequence[tuple[float, float]]] = None,
    drop_any_na: bool = True,
) -> pd.Series:
    """
    从价格矩阵直接计算风险平价权重。
    """
    cov = estimate_covariance_from_prices(
        price_df=price_df,
        method=method,
        return_type=return_type,
        annualization=annualization,
        winsorize=winsorize,
        winsor_lower_q=winsor_lower_q,
        winsor_upper_q=winsor_upper_q,
        ewma_lambda=ewma_lambda,
        drop_any_na=drop_any_na,
    )

    weights = solve_risk_parity_weights(
        cov_matrix=cov,
        target_risk_budget=target_risk_budget,
        long_only=long_only,
        weight_bounds=weight_bounds,
    )

    return pd.Series(weights, index=cov.index, name="weight")


# ============================================================
# 分析输出
# ============================================================

def make_risk_contribution_report(
    weights: Sequence[float] | pd.Series,
    cov_matrix: pd.DataFrame | np.ndarray,
    asset_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    生成风险贡献分析表。
    """
    if isinstance(cov_matrix, pd.DataFrame):
        names = cov_matrix.index.tolist()
    else:
        names = list(asset_names) if asset_names is not None else [f"asset_{i}" for i in range(len(weights))]

    w = np.asarray(weights, dtype=float)
    rc = calc_risk_contribution(w, cov_matrix)
    rrc = calc_relative_risk_contribution(w, cov_matrix)
    mrc = calc_marginal_risk_contribution(w, cov_matrix)

    report = pd.DataFrame({
        "weight": w,
        "marginal_risk_contribution": mrc,
        "risk_contribution": rc,
        "relative_risk_contribution": rrc,
    }, index=names)

    report.index.name = "asset"
    return report


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
    # 假设你已经从数据库里读取了长表 daily_prices
    df = pd.DataFrame({
        "trade_date": [
            "20240102", "20240103", "20240104",
            "20240102", "20240103", "20240104",
            "20240102", "20240103", "20240104",
        ],
        "ts_code": [
            "510300.SH", "510300.SH", "510300.SH",
            "511010.SH", "511010.SH", "511010.SH",
            "518880.SH", "518880.SH", "518880.SH",
        ],
        "close": [
            3.50, 3.52, 3.49,
            112.0, 112.1, 112.0,
            4.85, 4.90, 4.88,
        ]
    })

    # 1. 构建价格矩阵
    price = build_price_matrix(df, date_col="trade_date", code_col="ts_code", price_col="close")

    # 2. 预处理
    price_prepared = prepare_price_matrix(
        price,
        calendar=price.index,
        ffill=True,
        ffill_limit=5,
        min_non_na_ratio=0.8,
        drop_all_na_dates=True,
    )

    # 3. 计算风险平价权重
    weights = calc_risk_parity_weights_from_prices(
        price_prepared,
        method="sample",       # 或 "ewma"
        return_type="log",
        annualization=252,
        long_only=True,
        drop_any_na=True,
    )

    # 4. 协方差和风险贡献
    cov = estimate_covariance_from_prices(price_prepared)
    report = make_risk_contribution_report(weights, cov)

    print("weights")
    print(weights)
    print()
    print("risk contribution report")
    print(report)