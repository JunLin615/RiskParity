"""
dual_momentum_params.py

纯参数版双动量（Dual Momentum）策略配置库。
不依赖数据库、不依赖回测执行模块。
主要提供：
1. 数据字段与数据库表名参数
2. 资产池、绝对动量、相对动量、调仓、仓位等配置对象
3. 参数合法性校验与标准化
4. 参数序列化 / 反序列化
5. 常用参数模板（适合日线 ETF / LOF / 指数类资产）

设计原则：
- 只管理“参数与配置”，不实现具体回测逻辑
- 输入输出尽量使用标准库对象，便于被任意回测框架消费
- 默认兼容 risk_parity.py 使用的数据字段命名：trade_date / ts_code / close
- 优先适配日线级别 ETF / LOF / 指数轮动策略
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal, Mapping, Optional, Sequence
import json


# ============================================================
# 通用工具
# ============================================================

WeightingMethod = Literal["equal", "score", "rank", "inverse_vol"]
RebalanceFrequency = Literal["daily", "weekly", "monthly"]
TradeTiming = Literal["next_open", "next_close", "same_close"]
AbsoluteFilterMode = Literal["off", "candidate", "market"]
AbsolutePassRule = Literal["all", "any", "weighted"]
RankingMethod = Literal["total_return", "risk_adjusted_return", "weighted_multi_window"]
DefensiveMode = Literal["cash", "single_asset", "equal_defensive_basket"]
UniverseSelectionMode = Literal["manual_list", "db_tag", "db_query"]
SignalPriceField = Literal["open", "high", "low", "close", "adj_close", "pre_close"]


def _ensure_unique_str_list(values: Sequence[str], field_name: str) -> list[str]:
    out = [str(v) for v in values]
    if len(out) != len(set(out)):
        raise ValueError(f"{field_name} contains duplicated values")
    return out



def normalize_weights(weights: Sequence[float], *, field_name: str = "weights") -> list[float]:
    vals = [float(x) for x in weights]
    total = sum(vals)
    if total <= 0:
        raise ValueError(f"{field_name} sum must be > 0")
    return [x / total for x in vals]



def ensure_same_length(*seqs: Sequence[Any], field_names: Optional[Sequence[str]] = None) -> None:
    lengths = [len(x) for x in seqs]
    if len(set(lengths)) != 1:
        if field_names is None:
            raise ValueError(f"sequence lengths are inconsistent: {lengths}")
        detail = ", ".join(f"{name}={length}" for name, length in zip(field_names, lengths))
        raise ValueError(f"sequence lengths are inconsistent: {detail}")





def _merge_mapping_into_dataclass(default_obj: Any, updates: Optional[Mapping[str, Any]]) -> Any:
    """
    将 updates 合并到 default_obj 上，未提供的字段保留默认值。

    只做浅层字段级别合并；嵌套 dataclass 由外层显式处理。
    """
    if updates is None:
        return default_obj
    merged = {name: getattr(default_obj, name) for name in default_obj.__dataclass_fields__.keys()}
    merged.update(dict(updates))
    return type(default_obj)(**merged)

def _dataclass_to_plain_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _dataclass_to_plain_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _dataclass_to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_plain_dict(v) for v in obj]
    return obj


# ============================================================
# 数据与存储参数
# ============================================================

@dataclass(slots=True)
class DataSchemaConfig:
    """
    行情数据字段配置。
    默认兼容 risk_parity.py 中 build_price_matrix 的字段命名。
    """

    date_col: str = "trade_date"
    code_col: str = "ts_code"
    close_col: str = "close"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    pre_close_col: str = "pre_close"
    volume_col: str = "vol"
    amount_col: str = "amount"
    adj_factor_col: str = "adj_factor"
    date_format: Optional[str] = "%Y%m%d"
    signal_price_field: SignalPriceField = "close"
    execution_price_field: SignalPriceField = "open"
    calendar_source: Literal["price_index", "db_calendar", "custom"] = "price_index"

    def validate(self) -> None:
        if not self.date_col:
            raise ValueError("date_col must not be empty")
        if not self.code_col:
            raise ValueError("code_col must not be empty")
        if not self.close_col:
            raise ValueError("close_col must not be empty")


@dataclass(slots=True)
class StorageConfig:
    """
    仅定义数据库表名 / 字段名约定，不执行任何数据库写入。
    便于与风险平价策略共用数据库。
    """

    price_table: str = "fund_daily"
    universe_table: str = "strategy_universe"
    signal_table: str = "dual_momentum_signal"
    target_weight_table: str = "dual_momentum_target_weight"
    parameter_table: str = "strategy_parameter_snapshot"
    strategy_name: str = "dual_momentum"
    strategy_version: str = "v1"

    signal_date_col: str = "signal_date"
    rebalance_date_col: str = "rebalance_date"
    weight_col: str = "target_weight"
    selected_col: str = "is_selected"
    abs_score_col: str = "absolute_score"
    rel_score_col: str = "relative_score"
    rank_col: str = "relative_rank"
    reason_col: str = "selection_reason"

    def validate(self) -> None:
        required = [
            self.price_table,
            self.universe_table,
            self.signal_table,
            self.target_weight_table,
            self.parameter_table,
            self.strategy_name,
            self.strategy_version,
        ]
        if any(not x for x in required):
            raise ValueError("storage config contains empty table / strategy names")


# ============================================================
# 策略参数主体
# ============================================================

@dataclass(slots=True)
class UniverseConfig:
    """
    候选资产池配置。
    """

    selection_mode: UniverseSelectionMode = "manual_list"
    asset_codes: list[str] = field(default_factory=list)
    defensive_asset_codes: list[str] = field(default_factory=list)
    benchmark_codes: list[str] = field(default_factory=list)
    market_proxy_codes: list[str] = field(default_factory=list)
    universe_tag: Optional[str] = None
    db_query_name: Optional[str] = None
    min_listed_days: int = 120
    min_non_na_ratio: float = 0.9
    min_turnover: Optional[float] = None
    min_amount: Optional[float] = None
    exclude_recent_n_days: int = 0

    def validate(self) -> None:
        self.asset_codes = _ensure_unique_str_list(self.asset_codes, "asset_codes")
        self.defensive_asset_codes = _ensure_unique_str_list(self.defensive_asset_codes, "defensive_asset_codes")
        self.benchmark_codes = _ensure_unique_str_list(self.benchmark_codes, "benchmark_codes")
        self.market_proxy_codes = _ensure_unique_str_list(self.market_proxy_codes, "market_proxy_codes")

        if self.selection_mode == "manual_list" and not self.asset_codes:
            raise ValueError("asset_codes must not be empty when selection_mode='manual_list'")
        if self.selection_mode == "db_tag" and not self.universe_tag:
            raise ValueError("universe_tag must not be empty when selection_mode='db_tag'")
        if self.selection_mode == "db_query" and not self.db_query_name:
            raise ValueError("db_query_name must not be empty when selection_mode='db_query'")
        if self.min_listed_days < 0:
            raise ValueError("min_listed_days must be >= 0")
        if not (0 < self.min_non_na_ratio <= 1):
            raise ValueError("min_non_na_ratio must be in (0, 1]")
        if self.exclude_recent_n_days < 0:
            raise ValueError("exclude_recent_n_days must be >= 0")


@dataclass(slots=True)
class MomentumWindowConfig:
    """
    动量窗口配置。

    lookbacks:
        例如 [20, 60, 120]
    weights:
        当 score_method 需要多窗口加权时使用；若为空则默认等权。
    """

    lookbacks: list[int] = field(default_factory=lambda: [60])
    weights: list[float] = field(default_factory=list)

    def validate(self, *, field_name: str = "momentum_window") -> None:
        if not self.lookbacks:
            raise ValueError(f"{field_name}.lookbacks must not be empty")
        if any(int(x) <= 0 for x in self.lookbacks):
            raise ValueError(f"{field_name}.lookbacks must all be positive integers")
        self.lookbacks = [int(x) for x in self.lookbacks]
        if len(set(self.lookbacks)) != len(self.lookbacks):
            raise ValueError(f"{field_name}.lookbacks must not contain duplicates")

        if self.weights:
            ensure_same_length(self.lookbacks, self.weights, field_names=[f"{field_name}.lookbacks", f"{field_name}.weights"])
            self.weights = normalize_weights(self.weights, field_name=f"{field_name}.weights")
        else:
            n = len(self.lookbacks)
            self.weights = [1.0 / n] * n


@dataclass(slots=True)
class AbsoluteMomentumConfig:
    """
    绝对动量过滤配置。

    mode:
        - off: 不启用绝对动量
        - candidate: 对候选资产自身做绝对动量过滤
        - market: 对市场代理资产做绝对动量总开关过滤
    pass_rule:
        - all: 多窗口全部通过
        - any: 多窗口任一通过
        - weighted: 按多窗口加权得分是否大于阈值
    threshold:
        默认 0.0，表示收益率大于 0 才通过
    """

    mode: AbsoluteFilterMode = "candidate"
    pass_rule: AbsolutePassRule = "weighted"
    windows: MomentumWindowConfig = field(default_factory=lambda: MomentumWindowConfig([20, 60, 120], [0.2, 0.3, 0.5]))
    threshold: float = 0.0
    use_ma_filter: bool = False
    ma_window: int = 120
    require_price_above_ma: bool = False
    market_proxy_code: Optional[str] = None

    def validate(self) -> None:
        self.windows.validate(field_name="absolute_momentum.windows")
        if self.use_ma_filter:
            if self.ma_window <= 0:
                raise ValueError("absolute_momentum.ma_window must be > 0")
        if self.mode == "market" and not self.market_proxy_code:
            raise ValueError("market_proxy_code must not be empty when absolute_momentum.mode='market'")


@dataclass(slots=True)
class RelativeMomentumConfig:
    """
    相对动量排序配置。
    """

    ranking_method: RankingMethod = "weighted_multi_window"
    windows: MomentumWindowConfig = field(default_factory=lambda: MomentumWindowConfig([20, 60, 120], [0.2, 0.3, 0.5]))
    top_k: int = 3
    buffer_k: int = 0
    min_score: Optional[float] = None
    tie_breaker: Literal["code", "short_window", "long_window"] = "long_window"
    risk_adjust_lookback: int = 20
    risk_adjust_floor: float = 1e-8

    def validate(self) -> None:
        self.windows.validate(field_name="relative_momentum.windows")
        if self.top_k <= 0:
            raise ValueError("relative_momentum.top_k must be > 0")
        if self.buffer_k < 0:
            raise ValueError("relative_momentum.buffer_k must be >= 0")
        if self.risk_adjust_lookback <= 0:
            raise ValueError("relative_momentum.risk_adjust_lookback must be > 0")
        if self.risk_adjust_floor <= 0:
            raise ValueError("relative_momentum.risk_adjust_floor must be > 0")


@dataclass(slots=True)
class RebalanceConfig:
    """
    调仓参数。
    只定义规则，不实现实际调仓。
    """

    frequency: RebalanceFrequency = "monthly"
    trade_timing: TradeTiming = "next_open"
    weekday: int = 0
    monthday_rule: Literal["first", "last", "nth"] = "last"
    nth_weekday_of_month: Optional[int] = None
    warmup_bars: int = 120
    skip_if_insufficient_history: bool = True

    def validate(self) -> None:
        if not (0 <= self.weekday <= 4):
            raise ValueError("rebalance.weekday must be in [0, 4]")
        if self.monthday_rule == "nth":
            if self.nth_weekday_of_month is None:
                raise ValueError("nth_weekday_of_month must be provided when monthday_rule='nth'")
            if self.nth_weekday_of_month <= 0:
                raise ValueError("nth_weekday_of_month must be > 0")
        if self.warmup_bars <= 0:
            raise ValueError("warmup_bars must be > 0")


@dataclass(slots=True)
class AllocationConfig:
    """
    持仓分配参数。
    """

    weighting_method: WeightingMethod = "equal"
    max_single_weight: float = 1.0
    min_single_weight: float = 0.0
    score_power: float = 1.0
    use_selected_count_as_denominator: bool = True
    defensive_mode: DefensiveMode = "single_asset"
    defensive_asset_code: Optional[str] = None
    allow_partial_defensive_fill: bool = True
    normalize_final_weights: bool = True

    def validate(self, universe: UniverseConfig) -> None:
        if not (0 < self.max_single_weight <= 1):
            raise ValueError("allocation.max_single_weight must be in (0, 1]")
        if not (0 <= self.min_single_weight <= 1):
            raise ValueError("allocation.min_single_weight must be in [0, 1]")
        if self.min_single_weight > self.max_single_weight:
            raise ValueError("allocation.min_single_weight must be <= allocation.max_single_weight")
        if self.score_power <= 0:
            raise ValueError("allocation.score_power must be > 0")

        if self.defensive_mode == "single_asset":
            if not self.defensive_asset_code:
                if len(universe.defensive_asset_codes) != 1:
                    raise ValueError(
                        "defensive_asset_code must be set, or universe.defensive_asset_codes must contain exactly one asset "
                        "when defensive_mode='single_asset'"
                    )
        elif self.defensive_mode == "equal_defensive_basket":
            if not universe.defensive_asset_codes:
                raise ValueError(
                    "universe.defensive_asset_codes must not be empty when defensive_mode='equal_defensive_basket'"
                )


@dataclass(slots=True)
class RiskControlConfig:
    """
    风控类参数。
    只定义阈值与开关，不实现风控逻辑。
    """

    enable_market_gate: bool = False
    market_gate_code: Optional[str] = None
    market_gate_lookback: int = 120
    market_gate_threshold: float = 0.0

    enable_turnover_limit: bool = False
    max_turnover_per_rebalance: Optional[float] = None

    enable_volatility_filter: bool = False
    volatility_lookback: int = 20
    max_annualized_volatility: Optional[float] = None

    enable_drawdown_filter: bool = False
    drawdown_lookback: int = 60
    max_drawdown_threshold: Optional[float] = None

    def validate(self) -> None:
        if self.enable_market_gate:
            if not self.market_gate_code:
                raise ValueError("market_gate_code must not be empty when enable_market_gate=True")
            if self.market_gate_lookback <= 0:
                raise ValueError("market_gate_lookback must be > 0")

        if self.enable_turnover_limit:
            if self.max_turnover_per_rebalance is None:
                raise ValueError("max_turnover_per_rebalance must not be None when enable_turnover_limit=True")
            if not (0 <= self.max_turnover_per_rebalance <= 2):
                raise ValueError("max_turnover_per_rebalance must be in [0, 2]")

        if self.enable_volatility_filter:
            if self.volatility_lookback <= 0:
                raise ValueError("volatility_lookback must be > 0")
            if self.max_annualized_volatility is None or self.max_annualized_volatility <= 0:
                raise ValueError("max_annualized_volatility must be > 0 when enable_volatility_filter=True")

        if self.enable_drawdown_filter:
            if self.drawdown_lookback <= 0:
                raise ValueError("drawdown_lookback must be > 0")
            if self.max_drawdown_threshold is None or self.max_drawdown_threshold <= 0:
                raise ValueError("max_drawdown_threshold must be > 0 when enable_drawdown_filter=True")


@dataclass(slots=True)
class CostConfig:
    """
    成本参数，仅作为回测 / 实盘模块的输入，不在本库中执行。
    """

    commission_rate: float = 0.0003
    slippage_rate: float = 0.0005
    stamp_duty_rate: float = 0.0
    min_commission: float = 0.0

    def validate(self) -> None:
        for name, value in {
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "stamp_duty_rate": self.stamp_duty_rate,
            "min_commission": self.min_commission,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass(slots=True)
class DualMomentumParams:
    """
    双动量策略总参数对象。

    说明
    ----
    本对象只定义参数，不实现：
    - 数据库读取
    - 指标计算
    - 调仓执行
    - 回测撮合
    """

    data_schema: DataSchemaConfig = field(default_factory=DataSchemaConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    absolute_momentum: AbsoluteMomentumConfig = field(default_factory=AbsoluteMomentumConfig)
    relative_momentum: RelativeMomentumConfig = field(default_factory=RelativeMomentumConfig)
    rebalance: RebalanceConfig = field(default_factory=RebalanceConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    risk_control: RiskControlConfig = field(default_factory=RiskControlConfig)
    cost: CostConfig = field(default_factory=CostConfig)

    def validate(self) -> None:
        self.data_schema.validate()
        self.storage.validate()
        self.universe.validate()
        self.absolute_momentum.validate()
        self.relative_momentum.validate()
        self.rebalance.validate()
        self.allocation.validate(self.universe)
        self.risk_control.validate()
        self.cost.validate()

        max_lookback = max(
            max(self.absolute_momentum.windows.lookbacks),
            max(self.relative_momentum.windows.lookbacks),
            self.absolute_momentum.ma_window if self.absolute_momentum.use_ma_filter else 0,
            self.relative_momentum.risk_adjust_lookback,
            self.risk_control.market_gate_lookback if self.risk_control.enable_market_gate else 0,
            self.risk_control.volatility_lookback if self.risk_control.enable_volatility_filter else 0,
            self.risk_control.drawdown_lookback if self.risk_control.enable_drawdown_filter else 0,
        )
        if self.rebalance.warmup_bars < max_lookback:
            raise ValueError(
                f"rebalance.warmup_bars ({self.rebalance.warmup_bars}) must be >= max required lookback ({max_lookback})"
            )

        if self.relative_momentum.top_k > 0 and self.universe.asset_codes:
            if self.relative_momentum.top_k > len(self.universe.asset_codes):
                raise ValueError("relative_momentum.top_k must be <= number of asset_codes")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _dataclass_to_plain_dict(self)

    def to_json(self, *, ensure_ascii: bool = False, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)


# ============================================================
# 反序列化与模板
# ============================================================

def dual_momentum_params_from_dict(data: Mapping[str, Any]) -> DualMomentumParams:
    """
    从字典恢复参数对象。
    适合从 JSON / YAML / 数据库参数快照中恢复。

    设计原则
    --------
    - 未提供的字段保留 dataclass 自身默认值
    - 嵌套对象（例如 absolute_momentum.windows）也保留各自默认值
    - 不会因为缺失某个 key 而意外回退到别的 dataclass 的默认窗口
    """
    data = dict(data)

    abs_default = AbsoluteMomentumConfig()
    abs_data = dict(data.get("absolute_momentum", {}))
    abs_windows = _merge_mapping_into_dataclass(abs_default.windows, abs_data.get("windows"))
    abs_cfg = _merge_mapping_into_dataclass(
        abs_default,
        {**abs_data, "windows": abs_windows} if "windows" in abs_data else abs_data,
    )

    rel_default = RelativeMomentumConfig()
    rel_data = dict(data.get("relative_momentum", {}))
    rel_windows = _merge_mapping_into_dataclass(rel_default.windows, rel_data.get("windows"))
    rel_cfg = _merge_mapping_into_dataclass(
        rel_default,
        {**rel_data, "windows": rel_windows} if "windows" in rel_data else rel_data,
    )

    params = DualMomentumParams(
        data_schema=_merge_mapping_into_dataclass(DataSchemaConfig(), data.get("data_schema")),
        storage=_merge_mapping_into_dataclass(StorageConfig(), data.get("storage")),
        universe=_merge_mapping_into_dataclass(UniverseConfig(), data.get("universe")),
        absolute_momentum=abs_cfg,
        relative_momentum=rel_cfg,
        rebalance=_merge_mapping_into_dataclass(RebalanceConfig(), data.get("rebalance")),
        allocation=_merge_mapping_into_dataclass(AllocationConfig(), data.get("allocation")),
        risk_control=_merge_mapping_into_dataclass(RiskControlConfig(), data.get("risk_control")),
        cost=_merge_mapping_into_dataclass(CostConfig(), data.get("cost")),
    )
    params.validate()
    return params


def make_basic_dual_momentum_params(
    asset_codes: Sequence[str],
    defensive_asset_code: str,
) -> DualMomentumParams:
    """
    最小可用双动量参数模板：
    - 候选池由手工传入
    - 绝对动量：20/60/120 加权，大于 0 通过
    - 相对动量：20/60/120 加权，选前 3
    - 月度调仓
    - 等权持仓
    - 剩余仓位放入单一防御资产
    """
    params = DualMomentumParams(
        universe=UniverseConfig(
            selection_mode="manual_list",
            asset_codes=list(asset_codes),
            defensive_asset_codes=[defensive_asset_code],
            min_listed_days=120,
            min_non_na_ratio=0.9,
        ),
        absolute_momentum=AbsoluteMomentumConfig(
            mode="candidate",
            pass_rule="weighted",
            windows=MomentumWindowConfig([20, 60, 120], [0.2, 0.3, 0.5]),
            threshold=0.0,
            use_ma_filter=False,
        ),
        relative_momentum=RelativeMomentumConfig(
            ranking_method="weighted_multi_window",
            windows=MomentumWindowConfig([20, 60, 120], [0.2, 0.3, 0.5]),
            top_k=min(3, len(asset_codes)),
            buffer_k=1,
        ),
        rebalance=RebalanceConfig(
            frequency="monthly",
            trade_timing="next_open",
            monthday_rule="last",
            warmup_bars=120,
        ),
        allocation=AllocationConfig(
            weighting_method="equal",
            defensive_mode="single_asset",
            defensive_asset_code=defensive_asset_code,
            max_single_weight=1.0,
        ),
    )
    params.validate()
    return params



def make_a_share_etf_dual_momentum_params() -> DualMomentumParams:
    """
    A 股 ETF 常用模板。

    默认池子示例：
    - 510300.SH  沪深300ETF
    - 510500.SH  中证500ETF
    - 159915.SZ  创业板ETF
    - 588000.SH  科创50ETF
    - 515180.SH  红利ETF
    - 512880.SH  证券ETF
    防御资产：
    - 511010.SH  国债ETF（示例）
    """
    asset_codes = [
        "510300.SH",
        "510500.SH",
        "159915.SZ",
        "588000.SH",
        "515180.SH",
        "512880.SH",
    ]
    defensive = "511010.SH"

    params = make_basic_dual_momentum_params(asset_codes=asset_codes, defensive_asset_code=defensive)
    params.storage.price_table = "fund_daily"
    params.storage.signal_table = "dual_momentum_signal"
    params.storage.target_weight_table = "dual_momentum_target_weight"
    params.universe.market_proxy_codes = ["510300.SH"]
    params.risk_control.enable_market_gate = False
    params.validate()
    return params



def make_weekly_fast_dual_momentum_params(
    asset_codes: Sequence[str],
    defensive_asset_code: str,
) -> DualMomentumParams:
    """
    偏快节奏的周频双动量模板。
    适合行业 ETF / 风格 ETF 的高频一点的轮动研究。
    """
    params = DualMomentumParams(
        universe=UniverseConfig(
            selection_mode="manual_list",
            asset_codes=list(asset_codes),
            defensive_asset_codes=[defensive_asset_code],
            min_listed_days=60,
            min_non_na_ratio=0.9,
        ),
        absolute_momentum=AbsoluteMomentumConfig(
            mode="candidate",
            pass_rule="weighted",
            windows=MomentumWindowConfig([10, 20, 60], [0.2, 0.3, 0.5]),
            threshold=0.0,
        ),
        relative_momentum=RelativeMomentumConfig(
            ranking_method="weighted_multi_window",
            windows=MomentumWindowConfig([10, 20, 60], [0.2, 0.3, 0.5]),
            top_k=min(2, len(asset_codes)),
            buffer_k=1,
        ),
        rebalance=RebalanceConfig(
            frequency="weekly",
            trade_timing="next_open",
            weekday=0,
            warmup_bars=60,
        ),
        allocation=AllocationConfig(
            weighting_method="equal",
            defensive_mode="single_asset",
            defensive_asset_code=defensive_asset_code,
        ),
    )
    params.validate()
    return params



def summarize_dual_momentum_params(params: DualMomentumParams) -> dict[str, Any]:
    """
    输出一份更紧凑的参数摘要，便于日志、前端或回测报告使用。
    """
    params.validate()
    defensive_code = params.allocation.defensive_asset_code
    if not defensive_code and len(params.universe.defensive_asset_codes) == 1:
        defensive_code = params.universe.defensive_asset_codes[0]

    return {
        "strategy_name": params.storage.strategy_name,
        "strategy_version": params.storage.strategy_version,
        "n_assets": len(params.universe.asset_codes),
        "asset_codes": list(params.universe.asset_codes),
        "defensive_asset_codes": list(params.universe.defensive_asset_codes),
        "absolute_mode": params.absolute_momentum.mode,
        "absolute_lookbacks": list(params.absolute_momentum.windows.lookbacks),
        "absolute_weights": list(params.absolute_momentum.windows.weights),
        "absolute_threshold": params.absolute_momentum.threshold,
        "relative_ranking_method": params.relative_momentum.ranking_method,
        "relative_lookbacks": list(params.relative_momentum.windows.lookbacks),
        "relative_weights": list(params.relative_momentum.windows.weights),
        "top_k": params.relative_momentum.top_k,
        "buffer_k": params.relative_momentum.buffer_k,
        "rebalance_frequency": params.rebalance.frequency,
        "trade_timing": params.rebalance.trade_timing,
        "weighting_method": params.allocation.weighting_method,
        "defensive_mode": params.allocation.defensive_mode,
        "defensive_asset_code": defensive_code,
        "warmup_bars": params.rebalance.warmup_bars,
    }


# ============================================================
# 示例
# ============================================================

if __name__ == "__main__":
    params = make_a_share_etf_dual_momentum_params()

    print("dual momentum params summary")
    print(json.dumps(summarize_dual_momentum_params(params), ensure_ascii=False, indent=2))
    print()
    print("full params json")
    print(params.to_json())
