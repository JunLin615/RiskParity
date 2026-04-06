
"""
stock_data.py

股票数据管理库（面向 A 股日频宽截面研究）

设计目标
--------
1. 与 market_data.py 保持风格一致，便于理解和复用。
2. 重点支持 A 股个股数据库的首次建库与日常维护。
3. 以“按交易日抓取全市场横截面”为主，而非逐股票拉历史。
4. 保留原始事实层（raw / adj_factor / daily_basic / moneyflow / st）。
5. 可选接入 stk_factor_pro 作为研究加速层。
6. 支持进度条、限流、重试、断点续传、横截面友好索引。
7. 不包含训练、回测、组合权重、收益归因等研究层逻辑。
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
try:
    import tushare as ts
except Exception:  # pragma: no cover
    ts = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from market_data import (
        load_tushare_token,
        now_str,
        today_str,
        ensure_parent_dir,
        chunked,
    )
except Exception:
    def load_tushare_token(token_path="tushare_token.txt") -> str:
        return Path(token_path).read_text(encoding="utf-8").strip()

    def now_str() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def today_str() -> str:
        return datetime.now().strftime("%Y%m%d")

    def ensure_parent_dir(file_path: str | Path) -> None:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    def chunked(seq: list[str], size: int):
        for i in range(0, len(seq), size):
            yield seq[i:i + size]


# =========================
# 基础工具
# =========================

DEFAULT_STOCK_BASIC_FIELDS = ",".join([
    "ts_code", "symbol", "name", "area", "industry", "fullname", "enname",
    "cnspell", "market", "exchange", "curr_type", "list_status", "list_date",
    "delist_date", "is_hs", "act_name", "act_ent_type"
])

DEFAULT_DAILY_BASIC_FIELDS = ",".join([
    "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f",
    "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio",
    "dv_ttm", "total_share", "float_share", "free_share", "total_mv",
    "circ_mv"
])

DEFAULT_ST_FIELDS = ",".join([
    "ts_code", "name", "pub_date", "imp_date", "st_tpye", "st_reason",
    "st_explain"
])


def str_or_none(x: Any) -> Optional[str]:
    if x is None:
        return None
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None


def normalize_adjust_type(adjust_type: Optional[str]) -> str:
    if adjust_type is None:
        return "raw"
    x = str(adjust_type).strip().lower()
    if x in {"raw", "none", "noadj", "unadjusted"}:
        return "raw"
    if x in {"qfq", "forward", "pre"}:
        return "qfq"
    if x in {"hfq", "backward", "post"}:
        return "hfq"
    raise ValueError(f"不支持的 adjust_type: {adjust_type!r}")


def ensure_yyyymmdd(date_str: str) -> str:
    x = str(date_str).replace("-", "").strip()
    if len(x) != 8 or not x.isdigit():
        raise ValueError(f"日期格式应为 YYYYMMDD，收到: {date_str!r}")
    return x


def iter_with_progress(
    seq: Iterable[Any],
    *,
    show_progress: bool = True,
    desc: str = "",
    total: Optional[int] = None,
):
    if show_progress and tqdm is not None:
        return tqdm(seq, total=total, desc=desc)
    return seq


# =========================
# 配置
# =========================

@dataclass
class RateLimitConfig:
    per_minute: int
    sleep_buffer: float = 0.02


@dataclass
class StockDataConfig:
    tushare_token: str
    db_path: str = "data/db/stock_data.db"
    default_start_date: str = "20100101"
    default_exchange: str = "SSE"
    retry_times: int = 3
    retry_sleep: float = 1.0
    use_wal: bool = True
    endpoint_limits: dict[str, RateLimitConfig] = field(default_factory=lambda: {
        "stock_basic": RateLimitConfig(per_minute=50),
        "trade_cal": RateLimitConfig(per_minute=50),
        "daily": RateLimitConfig(per_minute=500),
        "adj_factor": RateLimitConfig(per_minute=200),
        "daily_basic": RateLimitConfig(per_minute=120),
        "moneyflow": RateLimitConfig(per_minute=120),
        "stock_st": RateLimitConfig(per_minute=60),
        "st": RateLimitConfig(per_minute=60),
        "stk_factor_pro": RateLimitConfig(per_minute=500),
    })


# =========================
# 限流器
# =========================

class EndpointRateLimiter:
    def __init__(self, limits: dict[str, RateLimitConfig]):
        self.limits = limits
        self.events: dict[str, deque[float]] = defaultdict(deque)

    def acquire(self, endpoint: str) -> None:
        cfg = self.limits.get(endpoint)
        if cfg is None or cfg.per_minute <= 0:
            return

        dq = self.events[endpoint]
        now = time.time()
        while dq and now - dq[0] >= 60:
            dq.popleft()

        if len(dq) >= cfg.per_minute:
            wait = 60 - (now - dq[0]) + cfg.sleep_buffer
            if wait > 0:
                time.sleep(wait)

        now = time.time()
        while dq and now - dq[0] >= 60:
            dq.popleft()
        dq.append(now)


# =========================
# Tushare 客户端
# =========================

class StockTushareClient:
    def __init__(self, config: StockDataConfig):
        self.config = config
        if ts is None:
            raise ImportError("未安装 tushare，请先在目标环境中安装 tushare。")
        ts.set_token(config.tushare_token)
        self.pro = ts.pro_api(config.tushare_token)
        self.rate_limiter = EndpointRateLimiter(config.endpoint_limits)

    def _call_with_retry(
        self,
        endpoint: str,
        func: Callable[..., pd.DataFrame],
        **kwargs,
    ) -> pd.DataFrame:
        last_exc = None
        for attempt in range(1, self.config.retry_times + 1):
            try:
                self.rate_limiter.acquire(endpoint)
                df = func(**kwargs)
                if df is None:
                    return pd.DataFrame()
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                return df
            except Exception as exc:
                last_exc = exc
                if attempt >= self.config.retry_times:
                    break
                time.sleep(self.config.retry_sleep * attempt)
        raise RuntimeError(
            f"Tushare 调用失败 endpoint={endpoint}, kwargs={kwargs}, error={last_exc}"
        ) from last_exc

    def fetch_stock_basic(
        self,
        list_status: str = "L",
        exchange: str = "",
        is_hs: Optional[str] = None,
        fields: str = DEFAULT_STOCK_BASIC_FIELDS,
    ) -> pd.DataFrame:
        kwargs = {
            "exchange": exchange,
            "list_status": list_status,
            "fields": fields,
        }
        if is_hs is not None:
            kwargs["is_hs"] = is_hs

        df = self._call_with_retry("stock_basic", self.pro.stock_basic, **kwargs)
        if df.empty:
            return df

        df["created_at"] = now_str()
        df["updated_at"] = now_str()
        return df

    def fetch_trade_calendar(
        self,
        exchange: str = "SSE",
        start_date: str = "20100101",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = today_str()

        df = self._call_with_retry(
            "trade_cal",
            self.pro.trade_cal,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
        )
        if df.empty:
            return df
        df["created_at"] = now_str()
        df["updated_at"] = now_str()
        return df

    def fetch_daily_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        trade_date = ensure_yyyymmdd(trade_date)
        df = self._call_with_retry("daily", self.pro.daily, trade_date=trade_date)
        if df.empty:
            return df
        df["source"] = "tushare_daily"
        df["created_at"] = now_str()
        df["updated_at"] = now_str()
        return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    def fetch_adj_factor_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        trade_date = ensure_yyyymmdd(trade_date)
        df = self._call_with_retry("adj_factor", self.pro.adj_factor, trade_date=trade_date)
        if df.empty:
            return df
        df["source"] = "tushare_adj_factor"
        df["created_at"] = now_str()
        df["updated_at"] = now_str()
        return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    def fetch_daily_basic_by_trade_date(
        self,
        trade_date: str,
        fields: str = DEFAULT_DAILY_BASIC_FIELDS,
    ) -> pd.DataFrame:
        trade_date = ensure_yyyymmdd(trade_date)
        df = self._call_with_retry(
            "daily_basic",
            self.pro.daily_basic,
            trade_date=trade_date,
            fields=fields,
        )
        if df.empty:
            return df
        df["source"] = "tushare_daily_basic"
        df["created_at"] = now_str()
        df["updated_at"] = now_str()
        return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    def fetch_moneyflow_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        trade_date = ensure_yyyymmdd(trade_date)
        df = self._call_with_retry("moneyflow", self.pro.moneyflow, trade_date=trade_date)
        if df.empty:
            return df
        df["source"] = "tushare_moneyflow"
        df["created_at"] = now_str()
        df["updated_at"] = now_str()
        return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    def fetch_stock_st_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        trade_date = ensure_yyyymmdd(trade_date)
        df = self._call_with_retry("stock_st", self.pro.stock_st, trade_date=trade_date)
        if df.empty:
            return df
        df["source"] = "tushare_stock_st"
        df["created_at"] = now_str()
        df["updated_at"] = now_str()
        return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    def fetch_st_events_by_ts_code(
        self,
        ts_code: str,
        fields: str = DEFAULT_ST_FIELDS,
    ) -> pd.DataFrame:
        df = self._call_with_retry("st", self.pro.st, ts_code=ts_code, fields=fields)
        if df.empty:
            return df
        df["source"] = "tushare_st"
        df["created_at"] = now_str()
        df["updated_at"] = now_str()
        return df.sort_values(["ts_code", "imp_date", "pub_date"]).reset_index(drop=True)

    def fetch_factor_pro_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        trade_date = ensure_yyyymmdd(trade_date)
        df = self._call_with_retry("stk_factor_pro", self.pro.stk_factor_pro, trade_date=trade_date)
        if df.empty:
            return df
        df["source"] = "tushare_stk_factor_pro"
        df["created_at"] = now_str()
        df["updated_at"] = now_str()
        return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


# =========================
# SQLite 存储
# =========================

class StockDataStore:
    def __init__(self, db_path: str, use_wal: bool = True):
        self.db_path = str(db_path)
        self.use_wal = use_wal
        ensure_parent_dir(self.db_path)
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        if self.use_wal:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA cache_size = -200000;")
        return conn

    def _table_exists(self, conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _get_table_columns(self, conn: sqlite3.Connection, table_name: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {r[1] for r in rows}

    def _ensure_columns(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        columns_sql: dict[str, str],
    ) -> None:
        if not self._table_exists(conn, table_name):
            return
        existing = self._get_table_columns(conn, table_name)
        for col, col_sql in columns_sql.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_sql}")

    def _init_db(self) -> None:
        with self.connect() as conn:
            cur = conn.cursor()

            cur.execute("""
            CREATE TABLE IF NOT EXISTS stocks (
                ts_code TEXT PRIMARY KEY,
                symbol TEXT,
                name TEXT,
                area TEXT,
                industry TEXT,
                fullname TEXT,
                enname TEXT,
                cnspell TEXT,
                market TEXT,
                exchange TEXT,
                curr_type TEXT,
                list_status TEXT,
                list_date TEXT,
                delist_date TEXT,
                is_hs TEXT,
                act_name TEXT,
                act_ent_type TEXT,
                source TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS trade_calendar (
                exchange TEXT NOT NULL,
                cal_date TEXT NOT NULL,
                is_open INTEGER,
                pretrade_date TEXT,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (exchange, cal_date)
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_daily_raw (
                ts_code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                pre_close REAL,
                change REAL,
                pct_chg REAL,
                vol REAL,
                amount REAL,
                source TEXT,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (ts_code, trade_date)
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_adj_factor (
                ts_code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                adj_factor REAL,
                source TEXT,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (ts_code, trade_date)
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_daily_basic (
                ts_code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                close REAL,
                turnover_rate REAL,
                turnover_rate_f REAL,
                volume_ratio REAL,
                pe REAL,
                pe_ttm REAL,
                pb REAL,
                ps REAL,
                ps_ttm REAL,
                dv_ratio REAL,
                dv_ttm REAL,
                total_share REAL,
                float_share REAL,
                free_share REAL,
                total_mv REAL,
                circ_mv REAL,
                source TEXT,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (ts_code, trade_date)
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_moneyflow (
                ts_code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                buy_sm_vol REAL,
                buy_sm_amount REAL,
                sell_sm_vol REAL,
                sell_sm_amount REAL,
                buy_md_vol REAL,
                buy_md_amount REAL,
                sell_md_vol REAL,
                sell_md_amount REAL,
                buy_lg_vol REAL,
                buy_lg_amount REAL,
                sell_lg_vol REAL,
                sell_lg_amount REAL,
                buy_elg_vol REAL,
                buy_elg_amount REAL,
                sell_elg_vol REAL,
                sell_elg_amount REAL,
                net_mf_vol REAL,
                net_mf_amount REAL,
                source TEXT,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (ts_code, trade_date)
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_st_daily (
                ts_code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                name TEXT,
                type TEXT,
                type_name TEXT,
                source TEXT,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (ts_code, trade_date)
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_st_event (
                ts_code TEXT NOT NULL,
                pub_date TEXT NOT NULL,
                imp_date TEXT,
                name TEXT,
                st_tpye TEXT,
                st_reason TEXT,
                st_explain TEXT,
                source TEXT,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (ts_code, pub_date, imp_date)
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_factor_pro (
                ts_code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                data_json TEXT,
                source TEXT,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (ts_code, trade_date)
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_eligibility_daily (
                ts_code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                is_listed INTEGER,
                is_st INTEGER,
                is_suspended INTEGER,
                has_daily INTEGER,
                has_daily_basic INTEGER,
                has_moneyflow INTEGER,
                days_since_list INTEGER,
                total_mv REAL,
                circ_mv REAL,
                amount REAL,
                close REAL,
                is_eligible INTEGER,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (ts_code, trade_date)
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_state (
                table_name TEXT NOT NULL,
                key_type TEXT NOT NULL,
                key_value TEXT NOT NULL,
                status TEXT,
                row_count INTEGER,
                message TEXT,
                updated_at TEXT,
                PRIMARY KEY (table_name, key_type, key_value)
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS update_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                key_type TEXT,
                key_value TEXT,
                row_count INTEGER,
                status TEXT,
                message TEXT,
                run_time TEXT,
                created_at TEXT
            );
            """)

            self._ensure_indexes(conn)
            conn.commit()

    def _ensure_indexes(self, conn: sqlite3.Connection) -> None:
        index_sqls = [
            "CREATE INDEX IF NOT EXISTS idx_trade_calendar_open ON trade_calendar(exchange, is_open, cal_date);",
            "CREATE INDEX IF NOT EXISTS idx_stock_daily_raw_trade_date ON stock_daily_raw(trade_date, ts_code);",
            "CREATE INDEX IF NOT EXISTS idx_stock_adj_factor_trade_date ON stock_adj_factor(trade_date, ts_code);",
            "CREATE INDEX IF NOT EXISTS idx_stock_daily_basic_trade_date ON stock_daily_basic(trade_date, ts_code);",
            "CREATE INDEX IF NOT EXISTS idx_stock_moneyflow_trade_date ON stock_moneyflow(trade_date, ts_code);",
            "CREATE INDEX IF NOT EXISTS idx_stock_st_daily_trade_date ON stock_st_daily(trade_date, ts_code);",
            "CREATE INDEX IF NOT EXISTS idx_stock_factor_pro_trade_date ON stock_factor_pro(trade_date, ts_code);",
            "CREATE INDEX IF NOT EXISTS idx_stock_eligibility_daily_trade_date ON stock_eligibility_daily(trade_date, ts_code);",
            "CREATE INDEX IF NOT EXISTS idx_update_log_table_key ON update_log(table_name, key_type, key_value);",
        ]
        for sql in index_sqls:
            conn.execute(sql)

    # -------- 通用日志 --------

    def log_update(
        self,
        table_name: str,
        key_type: Optional[str],
        key_value: Optional[str],
        row_count: int,
        status: str,
        message: str = "",
    ) -> None:
        with self.connect() as conn:
            conn.execute("""
                INSERT INTO update_log (
                    table_name, key_type, key_value, row_count,
                    status, message, run_time, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                table_name, key_type, key_value, row_count,
                status, message, now_str(), now_str()
            ))
            conn.commit()

    def set_ingestion_state(
        self,
        table_name: str,
        key_type: str,
        key_value: str,
        status: str,
        row_count: int,
        message: str = "",
    ) -> None:
        with self.connect() as conn:
            conn.execute("""
                INSERT INTO ingestion_state (
                    table_name, key_type, key_value, status,
                    row_count, message, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(table_name, key_type, key_value) DO UPDATE SET
                    status=excluded.status,
                    row_count=excluded.row_count,
                    message=excluded.message,
                    updated_at=excluded.updated_at
            """, (
                table_name, key_type, key_value, status,
                row_count, message, now_str()
            ))
            conn.commit()

    # -------- 保存基础信息 --------

    def upsert_stocks(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        x = df.copy()
        if "source" not in x.columns:
            x["source"] = "tushare_stock_basic"
        if "created_at" not in x.columns:
            x["created_at"] = now_str()
        x["updated_at"] = now_str()

        keep_cols = [
            "ts_code", "symbol", "name", "area", "industry", "fullname", "enname",
            "cnspell", "market", "exchange", "curr_type", "list_status",
            "list_date", "delist_date", "is_hs", "act_name", "act_ent_type",
            "source", "created_at", "updated_at"
        ]
        for c in keep_cols:
            if c not in x.columns:
                x[c] = None
        x = x[keep_cols]

        sql = """
        INSERT INTO stocks (
            ts_code, symbol, name, area, industry, fullname, enname,
            cnspell, market, exchange, curr_type, list_status, list_date,
            delist_date, is_hs, act_name, act_ent_type, source, created_at, updated_at
        ) VALUES (
            :ts_code, :symbol, :name, :area, :industry, :fullname, :enname,
            :cnspell, :market, :exchange, :curr_type, :list_status, :list_date,
            :delist_date, :is_hs, :act_name, :act_ent_type, :source, :created_at, :updated_at
        )
        ON CONFLICT(ts_code) DO UPDATE SET
            symbol=excluded.symbol,
            name=excluded.name,
            area=excluded.area,
            industry=excluded.industry,
            fullname=excluded.fullname,
            enname=excluded.enname,
            cnspell=excluded.cnspell,
            market=excluded.market,
            exchange=excluded.exchange,
            curr_type=excluded.curr_type,
            list_status=excluded.list_status,
            list_date=excluded.list_date,
            delist_date=excluded.delist_date,
            is_hs=excluded.is_hs,
            act_name=excluded.act_name,
            act_ent_type=excluded.act_ent_type,
            source=excluded.source,
            updated_at=excluded.updated_at
        ;
        """
        with self.connect() as conn:
            conn.executemany(sql, x.to_dict("records"))
            conn.commit()

    def upsert_trade_calendar(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        x = df.copy()
        if "created_at" not in x.columns:
            x["created_at"] = now_str()
        x["updated_at"] = now_str()
        keep_cols = ["exchange", "cal_date", "is_open", "pretrade_date", "created_at", "updated_at"]
        for c in keep_cols:
            if c not in x.columns:
                x[c] = None
        x = x[keep_cols]
        sql = """
        INSERT INTO trade_calendar (
            exchange, cal_date, is_open, pretrade_date, created_at, updated_at
        ) VALUES (
            :exchange, :cal_date, :is_open, :pretrade_date, :created_at, :updated_at
        )
        ON CONFLICT(exchange, cal_date) DO UPDATE SET
            is_open=excluded.is_open,
            pretrade_date=excluded.pretrade_date,
            updated_at=excluded.updated_at
        ;
        """
        with self.connect() as conn:
            conn.executemany(sql, x.to_dict("records"))
            conn.commit()

    # -------- 通用表保存 --------

    def _normalize_records(
        self,
        df: pd.DataFrame,
        keep_cols: list[str],
        source_default: str,
    ) -> pd.DataFrame:
        x = df.copy()
        if "source" not in x.columns:
            x["source"] = source_default
        if "created_at" not in x.columns:
            x["created_at"] = now_str()
        x["updated_at"] = now_str()
        for c in keep_cols:
            if c not in x.columns:
                x[c] = None
        return x[keep_cols]

    def upsert_stock_daily_raw(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        keep_cols = [
            "ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "change", "pct_chg", "vol", "amount",
            "source", "created_at", "updated_at"
        ]
        x = self._normalize_records(df, keep_cols, "tushare_daily")
        sql = """
        INSERT INTO stock_daily_raw (
            ts_code, trade_date, open, high, low, close,
            pre_close, change, pct_chg, vol, amount,
            source, created_at, updated_at
        ) VALUES (
            :ts_code, :trade_date, :open, :high, :low, :close,
            :pre_close, :change, :pct_chg, :vol, :amount,
            :source, :created_at, :updated_at
        )
        ON CONFLICT(ts_code, trade_date) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            pre_close=excluded.pre_close,
            change=excluded.change,
            pct_chg=excluded.pct_chg,
            vol=excluded.vol,
            amount=excluded.amount,
            source=excluded.source,
            updated_at=excluded.updated_at
        ;
        """
        with self.connect() as conn:
            conn.executemany(sql, x.to_dict("records"))
            conn.commit()

    def upsert_stock_adj_factor(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        keep_cols = ["ts_code", "trade_date", "adj_factor", "source", "created_at", "updated_at"]
        x = self._normalize_records(df, keep_cols, "tushare_adj_factor")
        sql = """
        INSERT INTO stock_adj_factor (
            ts_code, trade_date, adj_factor, source, created_at, updated_at
        ) VALUES (
            :ts_code, :trade_date, :adj_factor, :source, :created_at, :updated_at
        )
        ON CONFLICT(ts_code, trade_date) DO UPDATE SET
            adj_factor=excluded.adj_factor,
            source=excluded.source,
            updated_at=excluded.updated_at
        ;
        """
        with self.connect() as conn:
            conn.executemany(sql, x.to_dict("records"))
            conn.commit()

    def upsert_stock_daily_basic(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        keep_cols = [
            "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f",
            "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio",
            "dv_ttm", "total_share", "float_share", "free_share", "total_mv",
            "circ_mv", "source", "created_at", "updated_at"
        ]
        x = self._normalize_records(df, keep_cols, "tushare_daily_basic")
        sql = """
        INSERT INTO stock_daily_basic (
            ts_code, trade_date, close, turnover_rate, turnover_rate_f,
            volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm,
            total_share, float_share, free_share, total_mv, circ_mv,
            source, created_at, updated_at
        ) VALUES (
            :ts_code, :trade_date, :close, :turnover_rate, :turnover_rate_f,
            :volume_ratio, :pe, :pe_ttm, :pb, :ps, :ps_ttm, :dv_ratio, :dv_ttm,
            :total_share, :float_share, :free_share, :total_mv, :circ_mv,
            :source, :created_at, :updated_at
        )
        ON CONFLICT(ts_code, trade_date) DO UPDATE SET
            close=excluded.close,
            turnover_rate=excluded.turnover_rate,
            turnover_rate_f=excluded.turnover_rate_f,
            volume_ratio=excluded.volume_ratio,
            pe=excluded.pe,
            pe_ttm=excluded.pe_ttm,
            pb=excluded.pb,
            ps=excluded.ps,
            ps_ttm=excluded.ps_ttm,
            dv_ratio=excluded.dv_ratio,
            dv_ttm=excluded.dv_ttm,
            total_share=excluded.total_share,
            float_share=excluded.float_share,
            free_share=excluded.free_share,
            total_mv=excluded.total_mv,
            circ_mv=excluded.circ_mv,
            source=excluded.source,
            updated_at=excluded.updated_at
        ;
        """
        with self.connect() as conn:
            conn.executemany(sql, x.to_dict("records"))
            conn.commit()

    def upsert_stock_moneyflow(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        keep_cols = [
            "ts_code", "trade_date",
            "buy_sm_vol", "buy_sm_amount", "sell_sm_vol", "sell_sm_amount",
            "buy_md_vol", "buy_md_amount", "sell_md_vol", "sell_md_amount",
            "buy_lg_vol", "buy_lg_amount", "sell_lg_vol", "sell_lg_amount",
            "buy_elg_vol", "buy_elg_amount", "sell_elg_vol", "sell_elg_amount",
            "net_mf_vol", "net_mf_amount", "source", "created_at", "updated_at"
        ]
        x = self._normalize_records(df, keep_cols, "tushare_moneyflow")
        sql = """
        INSERT INTO stock_moneyflow (
            ts_code, trade_date,
            buy_sm_vol, buy_sm_amount, sell_sm_vol, sell_sm_amount,
            buy_md_vol, buy_md_amount, sell_md_vol, sell_md_amount,
            buy_lg_vol, buy_lg_amount, sell_lg_vol, sell_lg_amount,
            buy_elg_vol, buy_elg_amount, sell_elg_vol, sell_elg_amount,
            net_mf_vol, net_mf_amount, source, created_at, updated_at
        ) VALUES (
            :ts_code, :trade_date,
            :buy_sm_vol, :buy_sm_amount, :sell_sm_vol, :sell_sm_amount,
            :buy_md_vol, :buy_md_amount, :sell_md_vol, :sell_md_amount,
            :buy_lg_vol, :buy_lg_amount, :sell_lg_vol, :sell_lg_amount,
            :buy_elg_vol, :buy_elg_amount, :sell_elg_vol, :sell_elg_amount,
            :net_mf_vol, :net_mf_amount, :source, :created_at, :updated_at
        )
        ON CONFLICT(ts_code, trade_date) DO UPDATE SET
            buy_sm_vol=excluded.buy_sm_vol,
            buy_sm_amount=excluded.buy_sm_amount,
            sell_sm_vol=excluded.sell_sm_vol,
            sell_sm_amount=excluded.sell_sm_amount,
            buy_md_vol=excluded.buy_md_vol,
            buy_md_amount=excluded.buy_md_amount,
            sell_md_vol=excluded.sell_md_vol,
            sell_md_amount=excluded.sell_md_amount,
            buy_lg_vol=excluded.buy_lg_vol,
            buy_lg_amount=excluded.buy_lg_amount,
            sell_lg_vol=excluded.sell_lg_vol,
            sell_lg_amount=excluded.sell_lg_amount,
            buy_elg_vol=excluded.buy_elg_vol,
            buy_elg_amount=excluded.buy_elg_amount,
            sell_elg_vol=excluded.sell_elg_vol,
            sell_elg_amount=excluded.sell_elg_amount,
            net_mf_vol=excluded.net_mf_vol,
            net_mf_amount=excluded.net_mf_amount,
            source=excluded.source,
            updated_at=excluded.updated_at
        ;
        """
        with self.connect() as conn:
            conn.executemany(sql, x.to_dict("records"))
            conn.commit()

    def upsert_stock_st_daily(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        keep_cols = ["ts_code", "trade_date", "name", "type", "type_name", "source", "created_at", "updated_at"]
        x = self._normalize_records(df, keep_cols, "tushare_stock_st")
        sql = """
        INSERT INTO stock_st_daily (
            ts_code, trade_date, name, type, type_name, source, created_at, updated_at
        ) VALUES (
            :ts_code, :trade_date, :name, :type, :type_name, :source, :created_at, :updated_at
        )
        ON CONFLICT(ts_code, trade_date) DO UPDATE SET
            name=excluded.name,
            type=excluded.type,
            type_name=excluded.type_name,
            source=excluded.source,
            updated_at=excluded.updated_at
        ;
        """
        with self.connect() as conn:
            conn.executemany(sql, x.to_dict("records"))
            conn.commit()

    def upsert_stock_st_event(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        keep_cols = [
            "ts_code", "pub_date", "imp_date", "name", "st_tpye", "st_reason",
            "st_explain", "source", "created_at", "updated_at"
        ]
        x = self._normalize_records(df, keep_cols, "tushare_st")
        sql = """
        INSERT INTO stock_st_event (
            ts_code, pub_date, imp_date, name, st_tpye, st_reason,
            st_explain, source, created_at, updated_at
        ) VALUES (
            :ts_code, :pub_date, :imp_date, :name, :st_tpye, :st_reason,
            :st_explain, :source, :created_at, :updated_at
        )
        ON CONFLICT(ts_code, pub_date, imp_date) DO UPDATE SET
            name=excluded.name,
            st_tpye=excluded.st_tpye,
            st_reason=excluded.st_reason,
            st_explain=excluded.st_explain,
            source=excluded.source,
            updated_at=excluded.updated_at
        ;
        """
        with self.connect() as conn:
            conn.executemany(sql, x.to_dict("records"))
            conn.commit()

    def upsert_stock_factor_pro(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        x = df.copy()
        meta_cols = {"ts_code", "trade_date", "source", "created_at", "updated_at"}
        if "source" not in x.columns:
            x["source"] = "tushare_stk_factor_pro"
        if "created_at" not in x.columns:
            x["created_at"] = now_str()
        x["updated_at"] = now_str()

        factor_cols = [c for c in x.columns if c not in meta_cols]
        payload = x[factor_cols].apply(
            lambda row: pd.Series(row).dropna().to_json(force_ascii=False),
            axis=1,
        )
        out = pd.DataFrame({
            "ts_code": x["ts_code"],
            "trade_date": x["trade_date"],
            "data_json": payload,
            "source": x["source"],
            "created_at": x["created_at"],
            "updated_at": x["updated_at"],
        })
        sql = """
        INSERT INTO stock_factor_pro (
            ts_code, trade_date, data_json, source, created_at, updated_at
        ) VALUES (
            :ts_code, :trade_date, :data_json, :source, :created_at, :updated_at
        )
        ON CONFLICT(ts_code, trade_date) DO UPDATE SET
            data_json=excluded.data_json,
            source=excluded.source,
            updated_at=excluded.updated_at
        ;
        """
        with self.connect() as conn:
            conn.executemany(sql, out.to_dict("records"))
            conn.commit()

    def upsert_stock_eligibility_daily(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        keep_cols = [
            "ts_code", "trade_date", "is_listed", "is_st", "is_suspended",
            "has_daily", "has_daily_basic", "has_moneyflow", "days_since_list",
            "total_mv", "circ_mv", "amount", "close", "is_eligible",
            "created_at", "updated_at"
        ]
        x = self._normalize_records(df, keep_cols, "derived")
        sql = """
        INSERT INTO stock_eligibility_daily (
            ts_code, trade_date, is_listed, is_st, is_suspended,
            has_daily, has_daily_basic, has_moneyflow, days_since_list,
            total_mv, circ_mv, amount, close, is_eligible,
            created_at, updated_at
        ) VALUES (
            :ts_code, :trade_date, :is_listed, :is_st, :is_suspended,
            :has_daily, :has_daily_basic, :has_moneyflow, :days_since_list,
            :total_mv, :circ_mv, :amount, :close, :is_eligible,
            :created_at, :updated_at
        )
        ON CONFLICT(ts_code, trade_date) DO UPDATE SET
            is_listed=excluded.is_listed,
            is_st=excluded.is_st,
            is_suspended=excluded.is_suspended,
            has_daily=excluded.has_daily,
            has_daily_basic=excluded.has_daily_basic,
            has_moneyflow=excluded.has_moneyflow,
            days_since_list=excluded.days_since_list,
            total_mv=excluded.total_mv,
            circ_mv=excluded.circ_mv,
            amount=excluded.amount,
            close=excluded.close,
            is_eligible=excluded.is_eligible,
            updated_at=excluded.updated_at
        ;
        """
        with self.connect() as conn:
            conn.executemany(sql, x.to_dict("records"))
            conn.commit()

    # -------- 查询 --------

    def get_trade_calendar(
        self,
        exchange: str = "SSE",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        is_open: Optional[int] = None,
    ) -> pd.DataFrame:
        where = ["exchange = ?"]
        params: list[Any] = [exchange]
        if start_date:
            where.append("cal_date >= ?")
            params.append(start_date)
        if end_date:
            where.append("cal_date <= ?")
            params.append(end_date)
        if is_open is not None:
            where.append("is_open = ?")
            params.append(is_open)
        sql = f"""
        SELECT * FROM trade_calendar
        WHERE {" AND ".join(where)}
        ORDER BY cal_date
        """
        with self.connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def get_open_trade_dates(
        self,
        exchange: str = "SSE",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[str]:
        df = self.get_trade_calendar(
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
            is_open=1,
        )
        if df.empty:
            return []
        return df["cal_date"].astype(str).tolist()

    def get_stocks(
        self,
        list_statuses: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        sql = "SELECT * FROM stocks"
        params: list[Any] = []
        if list_statuses:
            xs = list(list_statuses)
            placeholders = ",".join("?" for _ in xs)
            sql += f" WHERE list_status IN ({placeholders})"
            params.extend(xs)
        sql += " ORDER BY ts_code"
        with self.connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def get_ts_codes(
        self,
        list_statuses: Optional[Iterable[str]] = None,
    ) -> list[str]:
        df = self.get_stocks(list_statuses=list_statuses)
        if df.empty:
            return []
        return df["ts_code"].astype(str).tolist()

    def get_table_trade_dates(
        self,
        table_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[str]:
        where = []
        params: list[Any] = []
        if start_date:
            where.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            where.append("trade_date <= ?")
            params.append(end_date)

        sql = f"SELECT DISTINCT trade_date FROM {table_name}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY trade_date"

        with self.connect() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        if df.empty:
            return []
        return df["trade_date"].astype(str).tolist()

    def get_missing_trade_dates(
        self,
        table_name: str,
        expected_trade_dates: list[str],
    ) -> list[str]:
        existing = set(self.get_table_trade_dates(table_name))
        return [d for d in expected_trade_dates if d not in existing]

    def get_latest_trade_date(self, table_name: str) -> Optional[str]:
        with self.connect() as conn:
            row = conn.execute(f"SELECT MAX(trade_date) FROM {table_name}").fetchone()
        return str_or_none(row[0]) if row else None

    def get_table(
        self,
        table_name: str,
        ts_codes: Optional[list[str]] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        where = []
        params: list[Any] = []
        if ts_codes:
            placeholders = ",".join("?" for _ in ts_codes)
            where.append(f"ts_code IN ({placeholders})")
            params.extend(ts_codes)
        if trade_date:
            where.append("trade_date = ?")
            params.append(trade_date)
        if start_date:
            where.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            where.append("trade_date <= ?")
            params.append(end_date)

        sql = f"SELECT * FROM {table_name}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY trade_date, ts_code"

        with self.connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def get_stock_daily_raw(
        self,
        ts_codes: Optional[list[str]] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.get_table("stock_daily_raw", ts_codes, trade_date, start_date, end_date)

    def get_stock_adj_factor(
        self,
        ts_codes: Optional[list[str]] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.get_table("stock_adj_factor", ts_codes, trade_date, start_date, end_date)

    def get_stock_daily_basic(
        self,
        ts_codes: Optional[list[str]] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.get_table("stock_daily_basic", ts_codes, trade_date, start_date, end_date)

    def get_stock_moneyflow(
        self,
        ts_codes: Optional[list[str]] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.get_table("stock_moneyflow", ts_codes, trade_date, start_date, end_date)

    def get_stock_st_daily(
        self,
        ts_codes: Optional[list[str]] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.get_table("stock_st_daily", ts_codes, trade_date, start_date, end_date)

    def get_stock_st_event(
        self,
        ts_codes: Optional[list[str]] = None,
        pub_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        where = []
        params: list[Any] = []
        if ts_codes:
            placeholders = ",".join("?" for _ in ts_codes)
            where.append(f"ts_code IN ({placeholders})")
            params.extend(ts_codes)
        if pub_date:
            where.append("pub_date = ?")
            params.append(pub_date)
        if start_date:
            where.append("pub_date >= ?")
            params.append(start_date)
        if end_date:
            where.append("pub_date <= ?")
            params.append(end_date)
        sql = "SELECT * FROM stock_st_event"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts_code, pub_date, imp_date"
        with self.connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def get_stock_factor_pro(
        self,
        ts_codes: Optional[list[str]] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        expand_json: bool = False,
    ) -> pd.DataFrame:
        df = self.get_table("stock_factor_pro", ts_codes, trade_date, start_date, end_date)
        if df.empty or not expand_json:
            return df
        expanded = []
        for item in df["data_json"].fillna("{}"):
            try:
                expanded.append(json.loads(item))
            except Exception:
                expanded.append({})
        return pd.concat([df.drop(columns=["data_json"]).reset_index(drop=True), pd.DataFrame(expanded)], axis=1)

    def get_stock_eligibility_daily(
        self,
        ts_codes: Optional[list[str]] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.get_table("stock_eligibility_daily", ts_codes, trade_date, start_date, end_date)

    def get_update_log(
        self,
        table_name: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        where = []
        params: list[Any] = []
        if table_name:
            where.append("table_name = ?")
            params.append(table_name)
        sql = "SELECT * FROM update_log"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)


# =========================
# 数据管理
# =========================

class StockDataManager:
    def __init__(self, config: StockDataConfig):
        self.config = config
        self.client = StockTushareClient(config)
        self.store = StockDataStore(config.db_path, use_wal=config.use_wal)

    # -------- 初始化 --------

    def initialize_trade_calendar(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> pd.DataFrame:
        start_date = start_date or self.config.default_start_date
        exchange = exchange or self.config.default_exchange
        df = self.client.fetch_trade_calendar(
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
        )
        if not df.empty:
            self.store.upsert_trade_calendar(df)
        self.store.log_update(
            table_name="trade_calendar",
            key_type="exchange",
            key_value=exchange,
            row_count=len(df),
            status="success",
            message=f"start={start_date}, end={end_date or today_str()}",
        )
        return df

    def initialize_stock_basic(
        self,
        list_statuses: Iterable[str] = ("L", "D", "P"),
        exchange: str = "",
        is_hs: Optional[str] = None,
    ) -> pd.DataFrame:
        frames = []
        for status in list_statuses:
            df = self.client.fetch_stock_basic(
                list_status=status,
                exchange=exchange,
                is_hs=is_hs,
            )
            if not df.empty:
                frames.append(df)
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not out.empty:
            out = out.sort_values(["ts_code", "list_status"]).drop_duplicates("ts_code", keep="last")
            self.store.upsert_stocks(out)
        self.store.log_update(
            table_name="stocks",
            key_type="full_refresh",
            key_value=",".join(list(list_statuses)),
            row_count=len(out),
            status="success",
            message=f"exchange={exchange}, is_hs={is_hs}",
        )
        return out

    # -------- 按日期更新 --------

    def _run_trade_date_backfill(
        self,
        *,
        table_name: str,
        fetcher: Callable[[str], pd.DataFrame],
        saver: Callable[[pd.DataFrame], None],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        show_progress: bool = True,
        skip_existing: bool = True,
        empty_is_success: bool = True,
    ) -> dict[str, int]:
        start_date = start_date or self.config.default_start_date
        end_date = end_date or today_str()

        open_dates = self.store.get_open_trade_dates(
            exchange=self.config.default_exchange,
            start_date=start_date,
            end_date=end_date,
        )
        if not open_dates:
            raise ValueError("交易日历为空，请先 initialize_trade_calendar().")

        target_dates = open_dates
        if skip_existing:
            target_dates = self.store.get_missing_trade_dates(table_name, target_dates)

        summary = {
            "date_count": len(target_dates),
            "success_count": 0,
            "inserted_rows": 0,
        }

        iterator = iter_with_progress(
            target_dates,
            show_progress=show_progress,
            desc=f"{table_name} backfill",
            total=len(target_dates),
        )
        for trade_date in iterator:
            try:
                df = fetcher(trade_date)
                if not df.empty:
                    saver(df)
                    row_count = len(df)
                    status = "success"
                    message = ""
                else:
                    row_count = 0
                    status = "success" if empty_is_success else "warning"
                    message = "empty result"

                self.store.set_ingestion_state(
                    table_name=table_name,
                    key_type="trade_date",
                    key_value=trade_date,
                    status=status,
                    row_count=row_count,
                    message=message,
                )
                self.store.log_update(
                    table_name=table_name,
                    key_type="trade_date",
                    key_value=trade_date,
                    row_count=row_count,
                    status=status,
                    message=message,
                )
                summary["success_count"] += 1
                summary["inserted_rows"] += row_count
            except Exception as exc:
                self.store.set_ingestion_state(
                    table_name=table_name,
                    key_type="trade_date",
                    key_value=trade_date,
                    status="error",
                    row_count=0,
                    message=str(exc),
                )
                self.store.log_update(
                    table_name=table_name,
                    key_type="trade_date",
                    key_value=trade_date,
                    row_count=0,
                    status="error",
                    message=str(exc),
                )
                if show_progress and tqdm is None:
                    print(f"[WARN] {table_name} {trade_date} failed: {exc}")

        return summary

    def backfill_stock_daily(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        show_progress: bool = True,
        skip_existing: bool = True,
    ) -> dict[str, int]:
        return self._run_trade_date_backfill(
            table_name="stock_daily_raw",
            fetcher=self.client.fetch_daily_by_trade_date,
            saver=self.store.upsert_stock_daily_raw,
            start_date=start_date,
            end_date=end_date,
            show_progress=show_progress,
            skip_existing=skip_existing,
            empty_is_success=True,
        )

    def backfill_stock_adj_factor(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        show_progress: bool = True,
        skip_existing: bool = True,
    ) -> dict[str, int]:
        return self._run_trade_date_backfill(
            table_name="stock_adj_factor",
            fetcher=self.client.fetch_adj_factor_by_trade_date,
            saver=self.store.upsert_stock_adj_factor,
            start_date=start_date,
            end_date=end_date,
            show_progress=show_progress,
            skip_existing=skip_existing,
            empty_is_success=True,
        )

    def backfill_stock_daily_basic(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        show_progress: bool = True,
        skip_existing: bool = True,
    ) -> dict[str, int]:
        return self._run_trade_date_backfill(
            table_name="stock_daily_basic",
            fetcher=self.client.fetch_daily_basic_by_trade_date,
            saver=self.store.upsert_stock_daily_basic,
            start_date=start_date,
            end_date=end_date,
            show_progress=show_progress,
            skip_existing=skip_existing,
            empty_is_success=True,
        )

    def backfill_stock_moneyflow(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        show_progress: bool = True,
        skip_existing: bool = True,
    ) -> dict[str, int]:
        return self._run_trade_date_backfill(
            table_name="stock_moneyflow",
            fetcher=self.client.fetch_moneyflow_by_trade_date,
            saver=self.store.upsert_stock_moneyflow,
            start_date=start_date,
            end_date=end_date,
            show_progress=show_progress,
            skip_existing=skip_existing,
            empty_is_success=True,
        )

    def backfill_stock_st_daily(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        show_progress: bool = True,
        skip_existing: bool = True,
    ) -> dict[str, int]:
        return self._run_trade_date_backfill(
            table_name="stock_st_daily",
            fetcher=self.client.fetch_stock_st_by_trade_date,
            saver=self.store.upsert_stock_st_daily,
            start_date=start_date,
            end_date=end_date,
            show_progress=show_progress,
            skip_existing=skip_existing,
            empty_is_success=True,
        )

    def backfill_stock_factor_pro(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        show_progress: bool = True,
        skip_existing: bool = True,
    ) -> dict[str, int]:
        return self._run_trade_date_backfill(
            table_name="stock_factor_pro",
            fetcher=self.client.fetch_factor_pro_by_trade_date,
            saver=self.store.upsert_stock_factor_pro,
            start_date=start_date,
            end_date=end_date,
            show_progress=show_progress,
            skip_existing=skip_existing,
            empty_is_success=True,
        )

    def backfill_stock_st_events(
        self,
        ts_codes: Optional[list[str]] = None,
        list_statuses: Iterable[str] = ("L", "D", "P"),
        show_progress: bool = True,
        skip_existing: bool = False,
    ) -> dict[str, int]:
        if ts_codes is None:
            ts_codes = self.store.get_ts_codes(list_statuses=list_statuses)

        summary = {"stock_count": len(ts_codes), "success_count": 0, "inserted_rows": 0}
        iterator = iter_with_progress(
            ts_codes,
            show_progress=show_progress,
            desc="stock_st_event backfill",
            total=len(ts_codes),
        )
        for ts_code in iterator:
            if skip_existing:
                existing = self.store.get_stock_st_event(ts_codes=[ts_code])
                if not existing.empty:
                    summary["success_count"] += 1
                    continue
            try:
                df = self.client.fetch_st_events_by_ts_code(ts_code)
                if not df.empty:
                    self.store.upsert_stock_st_event(df)
                self.store.log_update(
                    table_name="stock_st_event",
                    key_type="ts_code",
                    key_value=ts_code,
                    row_count=len(df),
                    status="success",
                    message="",
                )
                summary["success_count"] += 1
                summary["inserted_rows"] += len(df)
            except Exception as exc:
                self.store.log_update(
                    table_name="stock_st_event",
                    key_type="ts_code",
                    key_value=ts_code,
                    row_count=0,
                    status="error",
                    message=str(exc),
                )
        return summary

    # -------- 最新维护 --------

    def _update_latest_trade_date_table(
        self,
        *,
        table_name: str,
        fetcher: Callable[[str], pd.DataFrame],
        saver: Callable[[pd.DataFrame], None],
        trade_date: Optional[str] = None,
    ) -> dict[str, int]:
        if trade_date is None:
            open_dates = self.store.get_open_trade_dates(
                exchange=self.config.default_exchange,
                start_date=self.config.default_start_date,
                end_date=today_str(),
            )
            if not open_dates:
                raise ValueError("交易日历为空，请先初始化。")
            trade_date = open_dates[-1]
        df = fetcher(trade_date)
        if not df.empty:
            saver(df)
        self.store.set_ingestion_state(
            table_name=table_name,
            key_type="trade_date",
            key_value=trade_date,
            status="success",
            row_count=len(df),
            message="",
        )
        self.store.log_update(
            table_name=table_name,
            key_type="trade_date",
            key_value=trade_date,
            row_count=len(df),
            status="success",
            message="",
        )
        return {"trade_date": trade_date, "inserted_rows": len(df)}

    def update_latest_stock_daily(self, trade_date: Optional[str] = None) -> dict[str, int]:
        return self._update_latest_trade_date_table(
            table_name="stock_daily_raw",
            fetcher=self.client.fetch_daily_by_trade_date,
            saver=self.store.upsert_stock_daily_raw,
            trade_date=trade_date,
        )

    def update_latest_stock_adj_factor(self, trade_date: Optional[str] = None) -> dict[str, int]:
        return self._update_latest_trade_date_table(
            table_name="stock_adj_factor",
            fetcher=self.client.fetch_adj_factor_by_trade_date,
            saver=self.store.upsert_stock_adj_factor,
            trade_date=trade_date,
        )

    def update_latest_stock_daily_basic(self, trade_date: Optional[str] = None) -> dict[str, int]:
        return self._update_latest_trade_date_table(
            table_name="stock_daily_basic",
            fetcher=self.client.fetch_daily_basic_by_trade_date,
            saver=self.store.upsert_stock_daily_basic,
            trade_date=trade_date,
        )

    def update_latest_stock_moneyflow(self, trade_date: Optional[str] = None) -> dict[str, int]:
        return self._update_latest_trade_date_table(
            table_name="stock_moneyflow",
            fetcher=self.client.fetch_moneyflow_by_trade_date,
            saver=self.store.upsert_stock_moneyflow,
            trade_date=trade_date,
        )

    def update_latest_stock_st_daily(self, trade_date: Optional[str] = None) -> dict[str, int]:
        return self._update_latest_trade_date_table(
            table_name="stock_st_daily",
            fetcher=self.client.fetch_stock_st_by_trade_date,
            saver=self.store.upsert_stock_st_daily,
            trade_date=trade_date,
        )

    def update_latest_stock_factor_pro(self, trade_date: Optional[str] = None) -> dict[str, int]:
        return self._update_latest_trade_date_table(
            table_name="stock_factor_pro",
            fetcher=self.client.fetch_factor_pro_by_trade_date,
            saver=self.store.upsert_stock_factor_pro,
            trade_date=trade_date,
        )

    def update_latest_all(
        self,
        trade_date: Optional[str] = None,
        include_factor_pro: bool = True,
        include_moneyflow: bool = True,
        include_st_daily: bool = True,
    ) -> dict[str, dict[str, int]]:
        out = {
            "stock_daily_raw": self.update_latest_stock_daily(trade_date=trade_date),
            "stock_adj_factor": self.update_latest_stock_adj_factor(trade_date=trade_date),
            "stock_daily_basic": self.update_latest_stock_daily_basic(trade_date=trade_date),
        }
        if include_moneyflow:
            out["stock_moneyflow"] = self.update_latest_stock_moneyflow(trade_date=trade_date)
        if include_st_daily:
            out["stock_st_daily"] = self.update_latest_stock_st_daily(trade_date=trade_date)
        if include_factor_pro:
            out["stock_factor_pro"] = self.update_latest_stock_factor_pro(trade_date=trade_date)
        return out

    # -------- 价格口径 --------

    def get_prices(
        self,
        ts_codes: Optional[list[str]] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust_type: str = "raw",
    ) -> pd.DataFrame:
        adjust_type = normalize_adjust_type(adjust_type)
        raw = self.store.get_stock_daily_raw(
            ts_codes=ts_codes,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
        )
        if raw.empty or adjust_type == "raw":
            if not raw.empty:
                raw["adjust_type"] = "raw"
            return raw

        fac = self.store.get_stock_adj_factor(
            ts_codes=ts_codes,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
        )
        if fac.empty:
            raise ValueError("缺少复权因子，无法计算 qfq/hfq。")

        raw = raw.copy()
        fac = fac.copy()
        raw["trade_date"] = raw["trade_date"].astype(str)
        fac["trade_date"] = fac["trade_date"].astype(str)

        merged = raw.merge(
            fac[["ts_code", "trade_date", "adj_factor"]],
            on=["ts_code", "trade_date"],
            how="left",
        )
        merged = merged.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        merged["adj_factor"] = pd.to_numeric(merged["adj_factor"], errors="coerce")
        merged["adj_factor"] = merged.groupby("ts_code")["adj_factor"].ffill().bfill()

        if merged["adj_factor"].isna().any():
            bad = merged.loc[merged["adj_factor"].isna(), "ts_code"].astype(str).unique().tolist()[:5]
            raise ValueError(f"存在缺失复权因子的股票，示例: {bad}")

        if adjust_type == "hfq":
            ratio = merged["adj_factor"]
        else:  # qfq
            latest = merged.groupby("ts_code")["adj_factor"].transform("last")
            ratio = merged["adj_factor"] / latest

        for col in ["open", "high", "low", "close"]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce") * ratio

        merged = merged.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        merged["pre_close"] = merged.groupby("ts_code")["close"].shift(1)
        merged["change"] = merged["close"] - merged["pre_close"]
        merged["pct_chg"] = np.where(
            pd.notna(merged["pre_close"]) & (merged["pre_close"] != 0),
            merged["change"] / merged["pre_close"] * 100.0,
            np.nan,
        )
        merged["adjust_type"] = adjust_type
        return merged.drop(columns=["adj_factor"])

    # -------- eligibility / 资产池辅助 --------

    def build_eligibility_daily(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_list_days: int = 60,
        min_amount: Optional[float] = None,
        min_total_mv: Optional[float] = None,
        max_total_mv: Optional[float] = None,
        require_daily_basic: bool = True,
        require_moneyflow: bool = False,
        exclude_st: bool = True,
        show_progress: bool = True,
        skip_existing: bool = False,
    ) -> dict[str, int]:
        start_date = start_date or self.config.default_start_date
        end_date = end_date or today_str()

        trade_dates = self.store.get_open_trade_dates(
            exchange=self.config.default_exchange,
            start_date=start_date,
            end_date=end_date,
        )
        if skip_existing:
            trade_dates = self.store.get_missing_trade_dates("stock_eligibility_daily", trade_dates)

        stocks = self.store.get_stocks()
        if stocks.empty:
            raise ValueError("stocks 为空，请先 initialize_stock_basic().")

        stocks = stocks.copy()
        stocks["list_date"] = pd.to_datetime(stocks["list_date"], format="%Y%m%d", errors="coerce")
        stocks["delist_date"] = pd.to_datetime(stocks["delist_date"], format="%Y%m%d", errors="coerce")

        summary = {"date_count": len(trade_dates), "success_count": 0, "inserted_rows": 0}
        iterator = iter_with_progress(
            trade_dates,
            show_progress=show_progress,
            desc="eligibility build",
            total=len(trade_dates),
        )

        for trade_date in iterator:
            dt = pd.to_datetime(trade_date, format="%Y%m%d")
            base = stocks[["ts_code", "list_date", "delist_date"]].copy()
            base["trade_date"] = trade_date
            base["is_listed"] = (
                (base["list_date"].notna()) &
                (base["list_date"] <= dt) &
                (base["delist_date"].isna() | (base["delist_date"] >= dt))
            ).astype(int)
            base["days_since_list"] = np.where(
                base["list_date"].notna(),
                (dt - base["list_date"]).dt.days,
                np.nan,
            )

            daily = self.store.get_stock_daily_raw(trade_date=trade_date)
            basic = self.store.get_stock_daily_basic(trade_date=trade_date)
            mf = self.store.get_stock_moneyflow(trade_date=trade_date)
            st_df = self.store.get_stock_st_daily(trade_date=trade_date)

            base = base.merge(
                daily[["ts_code", "amount", "close"]].assign(has_daily=1),
                on="ts_code",
                how="left",
            )
            base = base.merge(
                basic[["ts_code", "total_mv", "circ_mv"]].assign(has_daily_basic=1),
                on="ts_code",
                how="left",
            )
            if not mf.empty:
                base = base.merge(
                    mf[["ts_code"]].drop_duplicates().assign(has_moneyflow=1),
                    on="ts_code",
                    how="left",
                )
            else:
                base["has_moneyflow"] = np.nan

            if not st_df.empty:
                base = base.merge(
                    st_df[["ts_code"]].drop_duplicates().assign(is_st=1),
                    on="ts_code",
                    how="left",
                )
            else:
                base["is_st"] = np.nan

            base["has_daily"] = base["has_daily"].fillna(0).astype(int)
            base["has_daily_basic"] = base["has_daily_basic"].fillna(0).astype(int)
            base["has_moneyflow"] = base["has_moneyflow"].fillna(0).astype(int)
            base["is_st"] = base["is_st"].fillna(0).astype(int)
            base["is_suspended"] = np.where(base["is_listed"].eq(1) & base["has_daily"].eq(0), 1, 0)

            eligible = base["is_listed"].eq(1) & base["days_since_list"].fillna(-1).ge(min_list_days) & base["has_daily"].eq(1)
            if require_daily_basic:
                eligible &= base["has_daily_basic"].eq(1)
            if require_moneyflow:
                eligible &= base["has_moneyflow"].eq(1)
            if exclude_st:
                eligible &= base["is_st"].eq(0)
            if min_amount is not None:
                eligible &= pd.to_numeric(base["amount"], errors="coerce").fillna(-np.inf).ge(min_amount)
            if min_total_mv is not None:
                eligible &= pd.to_numeric(base["total_mv"], errors="coerce").fillna(-np.inf).ge(min_total_mv)
            if max_total_mv is not None:
                eligible &= pd.to_numeric(base["total_mv"], errors="coerce").fillna(np.inf).le(max_total_mv)

            base["is_eligible"] = eligible.astype(int)
            base["created_at"] = now_str()
            base["updated_at"] = now_str()

            out_cols = [
                "ts_code", "trade_date", "is_listed", "is_st", "is_suspended",
                "has_daily", "has_daily_basic", "has_moneyflow", "days_since_list",
                "total_mv", "circ_mv", "amount", "close", "is_eligible",
                "created_at", "updated_at"
            ]
            out = base[out_cols].copy()
            self.store.upsert_stock_eligibility_daily(out)
            self.store.log_update(
                table_name="stock_eligibility_daily",
                key_type="trade_date",
                key_value=trade_date,
                row_count=len(out),
                status="success",
                message=f"eligible={int(out['is_eligible'].sum())}",
            )
            summary["success_count"] += 1
            summary["inserted_rows"] += len(out)

        return summary

    # -------- 横截面读取 --------

    def get_cross_section_snapshot(
        self,
        trade_date: str,
        *,
        include_daily: bool = True,
        include_basic: bool = True,
        include_moneyflow: bool = False,
        include_factor_pro: bool = False,
        include_st: bool = True,
        include_eligibility: bool = True,
        adjust_type: str = "qfq",
        eligible_only: bool = False,
        min_total_mv: Optional[float] = None,
        max_total_mv: Optional[float] = None,
    ) -> pd.DataFrame:
        trade_date = ensure_yyyymmdd(trade_date)

        universe = self.store.get_stocks()
        if universe.empty:
            return pd.DataFrame()
        out = universe[[
            "ts_code", "symbol", "name", "area", "industry", "market", "exchange",
            "list_status", "list_date", "delist_date"
        ]].copy()

        if include_daily:
            price_df = self.get_prices(trade_date=trade_date, adjust_type=adjust_type)
            if not price_df.empty:
                price_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
                out = out.merge(price_df[price_cols], on="ts_code", how="left")

        if include_basic:
            basic = self.store.get_stock_daily_basic(trade_date=trade_date)
            if not basic.empty:
                out = out.merge(
                    basic.drop(columns=["source", "created_at", "updated_at"], errors="ignore"),
                    on=["ts_code", "trade_date"] if "trade_date" in out.columns else ["ts_code"],
                    how="left",
                )

        if include_moneyflow:
            mf = self.store.get_stock_moneyflow(trade_date=trade_date)
            if not mf.empty:
                out = out.merge(
                    mf.drop(columns=["source", "created_at", "updated_at"], errors="ignore"),
                    on=["ts_code", "trade_date"] if "trade_date" in out.columns else ["ts_code"],
                    how="left",
                )

        if include_st:
            st_df = self.store.get_stock_st_daily(trade_date=trade_date)
            if not st_df.empty:
                st_df = st_df[["ts_code", "type", "type_name"]].copy()
                st_df["is_st"] = 1
                out = out.merge(st_df, on="ts_code", how="left")
            else:
                out["is_st"] = np.nan

        if include_eligibility:
            elig = self.store.get_stock_eligibility_daily(trade_date=trade_date)
            if not elig.empty:
                out = out.merge(
                    elig.drop(columns=["created_at", "updated_at"], errors="ignore"),
                    on=["ts_code", "trade_date"] if "trade_date" in out.columns else ["ts_code"],
                    how="left",
                )

        if include_factor_pro:
            factor_df = self.store.get_stock_factor_pro(trade_date=trade_date, expand_json=True)
            if not factor_df.empty:
                factor_df = factor_df.drop(columns=["source", "created_at", "updated_at"], errors="ignore")
                out = out.merge(
                    factor_df,
                    on=["ts_code", "trade_date"] if "trade_date" in out.columns else ["ts_code"],
                    how="left",
                )

        if "trade_date" not in out.columns:
            out["trade_date"] = trade_date

        if eligible_only and "is_eligible" in out.columns:
            out = out[out["is_eligible"] == 1].copy()

        if min_total_mv is not None and "total_mv" in out.columns:
            out = out[pd.to_numeric(out["total_mv"], errors="coerce") >= min_total_mv].copy()

        if max_total_mv is not None and "total_mv" in out.columns:
            out = out[pd.to_numeric(out["total_mv"], errors="coerce") <= max_total_mv].copy()

        return out.sort_values("ts_code").reset_index(drop=True)

    # -------- 一键首建 --------

    def initialize_database(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_factor_pro: bool = True,
        include_moneyflow: bool = True,
        include_st_daily: bool = True,
        build_eligibility: bool = True,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        start_date = start_date or self.config.default_start_date
        end_date = end_date or today_str()

        result: dict[str, Any] = {}
        result["trade_calendar"] = {
            "row_count": len(self.initialize_trade_calendar(start_date=start_date, end_date=end_date))
        }
        result["stocks"] = {
            "row_count": len(self.initialize_stock_basic())
        }
        result["stock_daily_raw"] = self.backfill_stock_daily(
            start_date=start_date,
            end_date=end_date,
            show_progress=show_progress,
            skip_existing=True,
        )
        result["stock_adj_factor"] = self.backfill_stock_adj_factor(
            start_date=start_date,
            end_date=end_date,
            show_progress=show_progress,
            skip_existing=True,
        )
        result["stock_daily_basic"] = self.backfill_stock_daily_basic(
            start_date=start_date,
            end_date=end_date,
            show_progress=show_progress,
            skip_existing=True,
        )
        if include_moneyflow:
            result["stock_moneyflow"] = self.backfill_stock_moneyflow(
                start_date=start_date,
                end_date=end_date,
                show_progress=show_progress,
                skip_existing=True,
            )
        if include_st_daily:
            result["stock_st_daily"] = self.backfill_stock_st_daily(
                start_date=start_date,
                end_date=end_date,
                show_progress=show_progress,
                skip_existing=True,
            )
        if include_factor_pro:
            result["stock_factor_pro"] = self.backfill_stock_factor_pro(
                start_date=start_date,
                end_date=end_date,
                show_progress=show_progress,
                skip_existing=True,
            )
        if build_eligibility:
            result["stock_eligibility_daily"] = self.build_eligibility_daily(
                start_date=start_date,
                end_date=end_date,
                show_progress=show_progress,
                skip_existing=True,
            )
        return result


# =========================
# 便捷函数
# =========================

def create_stock_manager(
    tushare_token: str,
    db_path: str = "data/db/stock_data.db",
    default_start_date: str = "20100101",
    default_exchange: str = "SSE",
    retry_times: int = 3,
    retry_sleep: float = 1.0,
) -> StockDataManager:
    config = StockDataConfig(
        tushare_token=tushare_token,
        db_path=db_path,
        default_start_date=default_start_date,
        default_exchange=default_exchange,
        retry_times=retry_times,
        retry_sleep=retry_sleep,
    )
    return StockDataManager(config)


# =========================
# 使用示例
# =========================

if __name__ == "__main__":
    TOKEN = load_tushare_token("tushare_token.txt")
    manager = create_stock_manager(
        tushare_token=TOKEN,
        db_path="data/db/stock_data.db",
        default_start_date="20180101",
        default_exchange="SSE",
    )

    # 1. 初始化基础信息
    manager.initialize_trade_calendar(start_date="20180101")
    manager.initialize_stock_basic(list_statuses=("L", "D", "P"))

    # 2. 首建主要事实表
    # manager.backfill_stock_daily(start_date="20180101", show_progress=True)
    # manager.backfill_stock_adj_factor(start_date="20180101", show_progress=True)
    # manager.backfill_stock_daily_basic(start_date="20180101", show_progress=True)
    # manager.backfill_stock_moneyflow(start_date="20180101", show_progress=True)
    # manager.backfill_stock_st_daily(start_date="20180101", show_progress=True)
    # manager.backfill_stock_factor_pro(start_date="20180101", show_progress=True)

    # 3. 生成 eligibility 日表
    # manager.build_eligibility_daily(
    #     start_date="20180101",
    #     min_list_days=60,
    #     min_amount=10000,
    #     exclude_st=True,
    #     show_progress=True,
    # )

    # 4. 更新最新交易日
    # print(manager.update_latest_all())

    # 5. 读取某日横截面
    # snap = manager.get_cross_section_snapshot(
    #     trade_date="20251231",
    #     include_moneyflow=True,
    #     include_factor_pro=False,
    #     adjust_type="qfq",
    #     eligible_only=False,
    # )
    # print(snap.head())
