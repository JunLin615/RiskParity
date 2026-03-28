"""
market_state_lib.py

基于现有 market_data.py 输出的数据格式，提供：
1. 对数收益率分布
2. 成交活跃度（默认 vol，可替换为换手率列）比值对数分布
3. Yang-Zhang 波动率序列与分布
4. 9 状态 / 4 状态映射、状态概率、状态转移概率矩阵
5. 任意时间粒度 delta_t 与相位 phase 的聚合与相位平均

设计原则：
- 直接接收 market_data.py 的日线 DataFrame（open/high/low/close/vol/amount/trade_date/ts_code）
- 不依赖回测框架，只做“可复用计算函数库”
- 时间粒度 delta_t 使用“交易日根数”做聚合，而不是自然日
- 支持指定相位 phase；也支持遍历全部相位后做统计
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

import numpy as np
import pandas as pd


PHASE_ALL = "all"
PHASE_MODE = int | Literal["all"]

# 9 状态：价格为主序，活跃度为次序
STATE9_ORDER: list[tuple[int, int]] = [
    (1, 1),
    (1, 0),
    (1, -1),
    (0, 1),
    (0, 0),
    (0, -1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
]
STATE9_LABELS: list[str] = [
    "(+1,+1)",
    "(+1,0)",
    "(+1,-1)",
    "(0,+1)",
    "(0,0)",
    "(0,-1)",
    "(-1,+1)",
    "(-1,0)",
    "(-1,-1)",
]
STATE9_NAME_MAP: dict[tuple[int, int], str] = dict(zip(STATE9_ORDER, STATE9_LABELS))
STATE9_CODE_MAP: dict[tuple[int, int], int] = {state: i + 1 for i, state in enumerate(STATE9_ORDER)}

# 4 状态：只保留 +/-1 组合
STATE4_ORDER: list[tuple[int, int]] = [
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
]
STATE4_LABELS: list[str] = [
    "(+1,+1)",
    "(+1,-1)",
    "(-1,+1)",
    "(-1,-1)",
]
STATE4_CODE_MAP: dict[tuple[int, int], int] = {state: i + 1 for i, state in enumerate(STATE4_ORDER)}


@dataclass(frozen=True)
class DistributionResult:
    """用于承载某个指标在最近 lbw 个观测上的样本分布。"""

    values: pd.Series
    metric_name: str
    delta_t: int
    phase: PHASE_MODE
    lbw: Optional[int]


@dataclass(frozen=True)
class StateAnalysisResult:
    """状态分析的统一返回对象。"""

    state_frame: pd.DataFrame
    state_prob_3x3: pd.DataFrame
    state_prob_9: pd.Series
    transition_9x9: pd.DataFrame
    transition_4x4: pd.DataFrame
    delta_t: int
    phase: PHASE_MODE
    price_threshold: float
    activity_threshold: float
    lbw: Optional[int]


# =========================
# 基础预处理
# =========================


def _ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")



def _ensure_ts_code(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_code" in df.columns:
        return df.copy()
    out = df.copy()
    out["ts_code"] = "SINGLE_ASSET"
    return out



def _prepare_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    _ensure_required_columns(df, ["trade_date", "open", "high", "low", "close"])
    out = _ensure_ts_code(df)
    out = out.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    if out["trade_date"].isna().any():
        # 兼容已经是 yyyy-mm-dd 的情况
        mask = out["trade_date"].isna()
        reparsed = pd.to_datetime(df.loc[mask, "trade_date"].astype(str), errors="coerce")
        out.loc[mask, "trade_date"] = reparsed

    numeric_cols = [c for c in ["open", "high", "low", "close", "vol", "amount"] if c in out.columns]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return out



def load_prices_from_store(
    store_like: Any,
    ts_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    兼容 market_data.py 的 MarketDataStore / MarketDataManager.store 接口。

    参数
    ----
    store_like:
        需要提供 get_daily_prices(ts_codes=[...], start_date=..., end_date=...) 方法。
    """
    if not hasattr(store_like, "get_daily_prices"):
        raise TypeError("store_like 需要提供 get_daily_prices 方法")
    return store_like.get_daily_prices(ts_codes=[ts_code], start_date=start_date, end_date=end_date)


# =========================
# 时间粒度聚合
# =========================


def _aggregate_single_asset(
    df: pd.DataFrame,
    delta_t: int = 1,
    phase: int = 0,
    extra_sum_cols: Optional[list[str]] = None,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    if delta_t < 1:
        raise ValueError("delta_t 必须 >= 1")
    if phase < 0 or phase >= delta_t:
        raise ValueError("phase 必须满足 0 <= phase < delta_t")

    g = df.sort_values("trade_date").reset_index(drop=True).copy()
    if g.empty:
        return g

    if phase > 0:
        g = g.iloc[phase:].reset_index(drop=True)
    if g.empty:
        return g

    group_id = np.arange(len(g)) // delta_t
    g["_group_id"] = group_id

    if drop_incomplete:
        group_sizes = g.groupby("_group_id").size()
        valid_ids = group_sizes[group_sizes == delta_t].index
        g = g[g["_group_id"].isin(valid_ids)].copy()
        if g.empty:
            return g.drop(columns=["_group_id"], errors="ignore")

    sum_cols = [c for c in ["vol", "amount"] if c in g.columns]
    if extra_sum_cols:
        for c in extra_sum_cols:
            if c in g.columns and c not in sum_cols:
                sum_cols.append(c)

    agg_dict: dict[str, Any] = {
        "ts_code": "first",
        "trade_date": "last",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "pre_close" in g.columns:
        agg_dict["pre_close"] = "first"
    for c in sum_cols:
        agg_dict[c] = "sum"

    out = g.groupby("_group_id", as_index=False).agg(agg_dict)

    start_dates = g.groupby("_group_id")["trade_date"].first().rename("start_trade_date")
    end_dates = g.groupby("_group_id")["trade_date"].last().rename("end_trade_date")
    out = out.join(start_dates, on="_group_id").join(end_dates, on="_group_id")
    out = out.drop(columns=["_group_id"], errors="ignore")
    out = out.rename(columns={"trade_date": "trade_date"})
    return out.reset_index(drop=True)



def aggregate_bars(
    df: pd.DataFrame,
    delta_t: int = 1,
    phase: int = 0,
    activity_col: str = "vol",
    extra_sum_cols: Optional[list[str]] = None,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    按交易日根数聚合成更粗粒度 K 线。

    - open: 组内第一根 open
    - close: 组内最后一根 close
    - high: 组内最高价
    - low: 组内最低价
    - vol/amount/activity_col: 组内求和
    - trade_date: 组内最后一个交易日
    - start_trade_date/end_trade_date: 组的起止交易日
    """
    out = _prepare_price_frame(df)

    extra_cols = [] if extra_sum_cols is None else list(extra_sum_cols)
    if activity_col not in extra_cols and activity_col not in {"vol", "amount"}:
        extra_cols.append(activity_col)

    frames: list[pd.DataFrame] = []
    for _, g in out.groupby("ts_code", sort=False):
        agg = _aggregate_single_asset(
            g,
            delta_t=delta_t,
            phase=phase,
            extra_sum_cols=extra_cols,
            drop_incomplete=drop_incomplete,
        )
        if not agg.empty:
            frames.append(agg)

    if not frames:
        return pd.DataFrame(columns=out.columns.tolist())

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return result



def aggregate_bars_all_phases(
    df: pd.DataFrame,
    delta_t: int,
    activity_col: str = "vol",
    extra_sum_cols: Optional[list[str]] = None,
    drop_incomplete: bool = True,
) -> dict[int, pd.DataFrame]:
    """返回所有相位下的聚合结果。"""
    if delta_t < 1:
        raise ValueError("delta_t 必须 >= 1")
    return {
        phase: aggregate_bars(
            df=df,
            delta_t=delta_t,
            phase=phase,
            activity_col=activity_col,
            extra_sum_cols=extra_sum_cols,
            drop_incomplete=drop_incomplete,
        )
        for phase in range(delta_t)
    }


# =========================
# 指标计算
# =========================


def compute_log_return_series(df: pd.DataFrame) -> pd.Series:
    out = _prepare_price_frame(df)
    prev_close = out.groupby("ts_code")["close"].shift(1)
    ret = np.log(out["close"] / prev_close)
    ret.name = "log_return"
    return ret



def compute_log_activity_ratio_series(
    df: pd.DataFrame,
    activity_col: str = "vol",
) -> pd.Series:
    out = _prepare_price_frame(df)
    if activity_col not in out.columns:
        raise ValueError(f"找不到 activity_col={activity_col!r}，可改成换手率列或成交量列")

    activity = pd.to_numeric(out[activity_col], errors="coerce")
    prev_activity = activity.groupby(out["ts_code"]).shift(1)
    ratio = np.log(activity / prev_activity)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    ratio.name = f"log_{activity_col}_ratio"
    return ratio



def _rolling_sample_variance(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).var(ddof=1)



def _yang_zhang_k(window: int) -> float:
    if window <= 1:
        raise ValueError("Yang-Zhang 窗口必须 > 1")
    return 0.34 / (1.34 + (window + 1) / (window - 1))



def compute_yang_zhang_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualize: bool = False,
    trading_periods: int = 252,
    delta_t: int = 1,
) -> pd.Series:
    """
    计算 Yang-Zhang 波动率。

    返回的是“每根 bar 对应一个滚动波动率估计值”的序列。
    若 annualize=True，则乘以 sqrt(trading_periods / delta_t)。
    """
    if window <= 1:
        raise ValueError("window 必须 > 1")

    out = _prepare_price_frame(df)
    results = []
    k = _yang_zhang_k(window)

    for _, g in out.groupby("ts_code", sort=False):
        g = g.sort_values("trade_date").reset_index(drop=True)

        prev_close = g["close"].shift(1)
        log_oc = np.log(g["close"] / g["open"])
        log_ov = np.log(g["open"] / prev_close)

        # Rogers-Satchell 项
        rs = (
            np.log(g["high"] / g["open"]) * np.log(g["high"] / g["close"])
            + np.log(g["low"] / g["open"]) * np.log(g["low"] / g["close"])
        )

        sigma_o2 = _rolling_sample_variance(log_ov, window)
        sigma_c2 = _rolling_sample_variance(log_oc, window)
        sigma_rs = rs.rolling(window=window, min_periods=window).mean()

        yz_var = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs
        yz_var = yz_var.clip(lower=0)
        yz_vol = np.sqrt(yz_var)

        if annualize:
            yz_vol = yz_vol * np.sqrt(trading_periods / max(delta_t, 1))

        yz_vol.index = g.index
        results.append(yz_vol)

    if not results:
        return pd.Series(dtype=float, name="yz_volatility")

    merged = pd.concat(results).sort_index()
    merged.name = "yz_volatility"
    return merged


# =========================
# 分布提取
# =========================


def _tail_distribution(series: pd.Series, lbw: Optional[int]) -> pd.Series:
    s = pd.Series(series).dropna().reset_index(drop=True)
    if lbw is not None:
        if lbw <= 0:
            raise ValueError("lbw 必须 > 0")
        s = s.tail(lbw).reset_index(drop=True)
    return s



def _compute_distribution_single_phase(
    df: pd.DataFrame,
    metric: Literal["log_return", "log_activity_ratio", "yz_volatility"],
    lbw: Optional[int],
    delta_t: int,
    phase: int,
    activity_col: str = "vol",
    yz_window: int = 20,
    annualize_yz: bool = False,
) -> DistributionResult:
    bars = aggregate_bars(df, delta_t=delta_t, phase=phase, activity_col=activity_col)

    if metric == "log_return":
        values = compute_log_return_series(bars)
    elif metric == "log_activity_ratio":
        values = compute_log_activity_ratio_series(bars, activity_col=activity_col)
    elif metric == "yz_volatility":
        values = compute_yang_zhang_volatility(
            bars,
            window=yz_window,
            annualize=annualize_yz,
            delta_t=delta_t,
        )
    else:
        raise ValueError(f"不支持的 metric: {metric}")

    dist = _tail_distribution(values, lbw=lbw)
    return DistributionResult(
        values=dist,
        metric_name=metric,
        delta_t=delta_t,
        phase=phase,
        lbw=lbw,
    )



def get_log_return_distribution(
    df: pd.DataFrame,
    lbw: Optional[int] = None,
    delta_t: int = 1,
    phase: PHASE_MODE = 0,
) -> DistributionResult | dict[int, DistributionResult]:
    if phase == PHASE_ALL:
        return {
            p: _compute_distribution_single_phase(
                df=df,
                metric="log_return",
                lbw=lbw,
                delta_t=delta_t,
                phase=p,
            )
            for p in range(delta_t)
        }
    return _compute_distribution_single_phase(
        df=df,
        metric="log_return",
        lbw=lbw,
        delta_t=delta_t,
        phase=phase,
    )



def get_log_activity_ratio_distribution(
    df: pd.DataFrame,
    lbw: Optional[int] = None,
    delta_t: int = 1,
    phase: PHASE_MODE = 0,
    activity_col: str = "vol",
) -> DistributionResult | dict[int, DistributionResult]:
    if phase == PHASE_ALL:
        return {
            p: _compute_distribution_single_phase(
                df=df,
                metric="log_activity_ratio",
                lbw=lbw,
                delta_t=delta_t,
                phase=p,
                activity_col=activity_col,
            )
            for p in range(delta_t)
        }
    return _compute_distribution_single_phase(
        df=df,
        metric="log_activity_ratio",
        lbw=lbw,
        delta_t=delta_t,
        phase=phase,
        activity_col=activity_col,
    )



def get_yang_zhang_vol_distribution(
    df: pd.DataFrame,
    lbw: Optional[int] = None,
    yz_window: int = 20,
    delta_t: int = 1,
    phase: PHASE_MODE = 0,
    annualize: bool = False,
) -> DistributionResult | dict[int, DistributionResult]:
    if phase == PHASE_ALL:
        return {
            p: _compute_distribution_single_phase(
                df=df,
                metric="yz_volatility",
                lbw=lbw,
                delta_t=delta_t,
                phase=p,
                yz_window=yz_window,
                annualize_yz=annualize,
            )
            for p in range(delta_t)
        }
    return _compute_distribution_single_phase(
        df=df,
        metric="yz_volatility",
        lbw=lbw,
        delta_t=delta_t,
        phase=phase,
        yz_window=yz_window,
        annualize_yz=annualize,
    )



def combine_phase_distributions(
    dist_map: dict[int, DistributionResult],
) -> pd.DataFrame:
    """把 all phase 的结果堆叠起来，便于后续统一画图或做核密度估计。"""
    rows = []
    for phase, res in dist_map.items():
        if res.values.empty:
            continue
        tmp = pd.DataFrame({
            "phase": phase,
            "value": res.values.values,
            "metric": res.metric_name,
        })
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["phase", "value", "metric"])
    return pd.concat(rows, ignore_index=True)


# =========================
# 状态定义与编码
# =========================


def value_to_tristate(values: pd.Series, threshold: float) -> pd.Series:
    if threshold < 0:
        raise ValueError("threshold 必须 >= 0")
    vals = pd.to_numeric(values, errors="coerce")
    state = pd.Series(np.where(vals > threshold, 1, np.where(vals < -threshold, -1, 0)), index=vals.index)
    state.name = "state"
    return state



def encode_state9(price_state: pd.Series, activity_state: pd.Series) -> pd.Series:
    pairs = list(zip(price_state.astype("Int64"), activity_state.astype("Int64")))
    codes = pd.Series([STATE9_CODE_MAP.get((int(p), int(a))) if pd.notna(p) and pd.notna(a) else np.nan for p, a in pairs], index=price_state.index)
    codes.name = "state9_code"
    return codes



def encode_state4(price_state: pd.Series, activity_state: pd.Series) -> pd.Series:
    pairs = list(zip(price_state.astype("Int64"), activity_state.astype("Int64")))
    codes = pd.Series([STATE4_CODE_MAP.get((int(p), int(a))) if pd.notna(p) and pd.notna(a) else np.nan for p, a in pairs], index=price_state.index)
    codes.name = "state4_code"
    return codes



def build_state_frame(
    df: pd.DataFrame,
    price_threshold: float,
    activity_threshold: float,
    delta_t: int = 1,
    phase: int = 0,
    lbw: Optional[int] = None,
    activity_col: str = "vol",
) -> pd.DataFrame:
    bars = aggregate_bars(df, delta_t=delta_t, phase=phase, activity_col=activity_col)
    bars = _prepare_price_frame(bars)

    bars["log_return"] = compute_log_return_series(bars)
    bars["log_activity_ratio"] = compute_log_activity_ratio_series(bars, activity_col=activity_col)
    bars["price_state"] = value_to_tristate(bars["log_return"], threshold=price_threshold)
    bars["activity_state"] = value_to_tristate(
        bars["log_activity_ratio"], threshold=activity_threshold
    )
    bars["state_tuple"] = list(zip(bars["price_state"], bars["activity_state"]))
    bars["state9_code"] = encode_state9(bars["price_state"], bars["activity_state"])
    bars["state4_code"] = encode_state4(bars["price_state"], bars["activity_state"])
    bars["state9_label"] = bars["state_tuple"].map(STATE9_NAME_MAP)

    bars = bars.dropna(subset=["log_return", "log_activity_ratio", "state9_code"]).reset_index(drop=True)
    if lbw is not None:
        if lbw <= 0:
            raise ValueError("lbw 必须 > 0")
        bars = bars.groupby("ts_code", group_keys=False).tail(lbw).reset_index(drop=True)
    return bars


# =========================
# 状态概率与转移矩阵
# =========================


def compute_state_probability_3x3(state_frame: pd.DataFrame) -> pd.DataFrame:
    _ensure_required_columns(state_frame, ["price_state", "activity_state"])
    table = pd.crosstab(
        state_frame["price_state"],
        state_frame["activity_state"],
        normalize="all",
        dropna=False,
    )
    table = table.reindex(index=[1, 0, -1], columns=[1, 0, -1], fill_value=0.0)
    table.index.name = "price_state"
    table.columns.name = "activity_state"
    return table



def compute_state_probability_9(state_frame: pd.DataFrame) -> pd.Series:
    _ensure_required_columns(state_frame, ["state9_code"])
    prob = state_frame["state9_code"].value_counts(normalize=True).sort_index()
    prob = prob.reindex(range(1, 10), fill_value=0.0)
    prob.index = pd.Index(STATE9_LABELS, name="state")
    prob.name = "probability"
    return prob



def _transition_matrix_from_codes(
    codes: pd.Series,
    states: list[int],
) -> pd.DataFrame:
    clean = pd.Series(codes).dropna().astype(int).reset_index(drop=True)
    if len(clean) < 2:
        labels = [str(s) for s in states]
        return pd.DataFrame(0.0, index=labels, columns=labels)

    current = clean.iloc[:-1]
    nxt = clean.iloc[1:]
    mat = pd.crosstab(current, nxt, normalize="index")
    mat = mat.reindex(index=states, columns=states, fill_value=0.0)
    mat.index = pd.Index([str(s) for s in states], name="from_state")
    mat.columns = pd.Index([str(s) for s in states], name="to_state")
    return mat



def compute_transition_matrix_9x9(state_frame: pd.DataFrame) -> pd.DataFrame:
    _ensure_required_columns(state_frame, ["state9_code"])
    frames = []
    for _, g in state_frame.groupby("ts_code", sort=False):
        codes = g["state9_code"].dropna().astype(int).reset_index(drop=True)
        if len(codes) < 2:
            continue
        current = codes.iloc[:-1].reset_index(drop=True)
        nxt = codes.iloc[1:].reset_index(drop=True)
        frames.append(pd.DataFrame({"from": current, "to": nxt}))

    if not frames:
        return pd.DataFrame(0.0, index=STATE9_LABELS, columns=STATE9_LABELS)

    pairs = pd.concat(frames, ignore_index=True)
    mat = pd.crosstab(pairs["from"], pairs["to"], normalize="all")
    mat = mat.reindex(index=range(1, 10), columns=range(1, 10), fill_value=0.0)
    mat.index = pd.Index(STATE9_LABELS, name="from_state")
    mat.columns = pd.Index(STATE9_LABELS, name="to_state")
    return mat



def compute_transition_matrix_4x4(state_frame: pd.DataFrame) -> pd.DataFrame:
    _ensure_required_columns(state_frame, ["state4_code"])
    frames = []
    for _, g in state_frame.groupby("ts_code", sort=False):
        codes = g["state4_code"].dropna().astype(int).reset_index(drop=True)
        if len(codes) < 2:
            continue
        current = codes.iloc[:-1].reset_index(drop=True)
        nxt = codes.iloc[1:].reset_index(drop=True)
        frames.append(pd.DataFrame({"from": current, "to": nxt}))

    if not frames:
        return pd.DataFrame(0.0, index=STATE4_LABELS, columns=STATE4_LABELS)

    pairs = pd.concat(frames, ignore_index=True)
    mat = pd.crosstab(pairs["from"], pairs["to"], normalize="all")
    mat = mat.reindex(index=range(1, 5), columns=range(1, 5), fill_value=0.0)
    mat.index = pd.Index(STATE4_LABELS, name="from_state")
    mat.columns = pd.Index(STATE4_LABELS, name="to_state")
    return mat



def analyze_states(
    df: pd.DataFrame,
    price_threshold: float,
    activity_threshold: float,
    delta_t: int = 1,
    phase: PHASE_MODE = 0,
    lbw: Optional[int] = None,
    activity_col: str = "vol",
) -> StateAnalysisResult | dict[int, StateAnalysisResult]:
    if phase == PHASE_ALL:
        return {
            p: analyze_states(
                df=df,
                price_threshold=price_threshold,
                activity_threshold=activity_threshold,
                delta_t=delta_t,
                phase=p,
                lbw=lbw,
                activity_col=activity_col,
            )
            for p in range(delta_t)
        }

    state_frame = build_state_frame(
        df=df,
        price_threshold = price_threshold,
        activity_threshold = activity_threshold,
        delta_t=delta_t,
        phase=phase,
        lbw=lbw,
        activity_col=activity_col,
    )
    return StateAnalysisResult(
        state_frame=state_frame,
        state_prob_3x3=compute_state_probability_3x3(state_frame),
        state_prob_9=compute_state_probability_9(state_frame),
        transition_9x9=compute_transition_matrix_9x9(state_frame),
        transition_4x4=compute_transition_matrix_4x4(state_frame),
        delta_t=delta_t,
        phase=phase,
        price_threshold=price_threshold,
        activity_threshold=activity_threshold,
        lbw=lbw,
    )



def average_transition_matrices(
    analysis_map: dict[int, StateAnalysisResult],
    matrix_type: Literal["9x9", "4x4"] = "9x9",
) -> pd.DataFrame:
    if not analysis_map:
        raise ValueError("analysis_map 不能为空")

    mats = []
    for res in analysis_map.values():
        mats.append(res.transition_9x9 if matrix_type == "9x9" else res.transition_4x4)

    total = mats[0].copy().astype(float)
    for mat in mats[1:]:
        total = total.add(mat.astype(float), fill_value=0.0)
    return total / len(mats)



def average_state_probabilities(
    analysis_map: dict[int, StateAnalysisResult],
    matrix_type: Literal["3x3", "9"] = "3x3",
) -> pd.DataFrame | pd.Series:
    if not analysis_map:
        raise ValueError("analysis_map 不能为空")

    first = next(iter(analysis_map.values()))
    total: pd.DataFrame | pd.Series
    total = first.state_prob_3x3.copy().astype(float) if matrix_type == "3x3" else first.state_prob_9.copy().astype(float)

    for res in list(analysis_map.values())[1:]:
        add_obj = res.state_prob_3x3 if matrix_type == "3x3" else res.state_prob_9
        total = total.add(add_obj.astype(float), fill_value=0.0)
    return total / len(analysis_map)


# =========================
# 便捷总入口
# =========================


def run_full_market_state_analysis(
    df: pd.DataFrame,
    price_threshold: float,
    activity_threshold: float,
    lbw: Optional[int] = None,
    delta_t: int = 1,
    phase: PHASE_MODE = 0,
    activity_col: str = "vol",
    yz_window: int = 20,
    annualize_yz: bool = False,
) -> dict[str, Any]:
    """
    一次性产出常用对象，便于后续单独写脚本做制图、比较和策略分析。
    """
    log_ret_dist = get_log_return_distribution(df, lbw=lbw, delta_t=delta_t, phase=phase)
    log_act_dist = get_log_activity_ratio_distribution(
        df,
        lbw=lbw,
        delta_t=delta_t,
        phase=phase,
        activity_col=activity_col,
    )
    yz_dist = get_yang_zhang_vol_distribution(
        df,
        lbw=lbw,
        yz_window=yz_window,
        delta_t=delta_t,
        phase=phase,
        annualize=annualize_yz,
    )
    state_res = analyze_states(
        df,
        price_threshold=price_threshold,
        activity_threshold=activity_threshold,
        delta_t=delta_t,
        phase=phase,
        lbw=lbw,
        activity_col=activity_col,
    )

    result = {
        "log_return_distribution": log_ret_dist,
        "log_activity_ratio_distribution": log_act_dist,
        "yang_zhang_vol_distribution": yz_dist,
        "state_analysis": state_res,
    }

    if phase == PHASE_ALL:
        result["phase_avg_transition_9x9"] = average_transition_matrices(state_res, matrix_type="9x9")
        result["phase_avg_transition_4x4"] = average_transition_matrices(state_res, matrix_type="4x4")
        result["phase_avg_state_prob_3x3"] = average_state_probabilities(state_res, matrix_type="3x3")
        result["phase_avg_state_prob_9"] = average_state_probabilities(state_res, matrix_type="9")
        result["log_return_distribution_all_phases"] = combine_phase_distributions(log_ret_dist)
        result["log_activity_ratio_distribution_all_phases"] = combine_phase_distributions(log_act_dist)
        result["yang_zhang_vol_distribution_all_phases"] = combine_phase_distributions(yz_dist)

    return result


__all__ = [
    "PHASE_ALL",
    "STATE9_ORDER",
    "STATE9_LABELS",
    "STATE9_CODE_MAP",
    "STATE4_ORDER",
    "STATE4_LABELS",
    "STATE4_CODE_MAP",
    "DistributionResult",
    "StateAnalysisResult",
    "load_prices_from_store",
    "aggregate_bars",
    "aggregate_bars_all_phases",
    "compute_log_return_series",
    "compute_log_activity_ratio_series",
    "compute_yang_zhang_volatility",
    "get_log_return_distribution",
    "get_log_activity_ratio_distribution",
    "get_yang_zhang_vol_distribution",
    "combine_phase_distributions",
    "value_to_tristate",
    "encode_state9",
    "encode_state4",
    "build_state_frame",
    "compute_state_probability_3x3",
    "compute_state_probability_9",
    "compute_transition_matrix_9x9",
    "compute_transition_matrix_4x4",
    "analyze_states",
    "average_transition_matrices",
    "average_state_probabilities",
    "run_full_market_state_analysis",
]
