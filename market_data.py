
"""
market_data.py

基于 Tushare + SQLite 的市场数据函数库
只包含市场数据的获取、存储、检查、更新
不包含任何回测、策略、权重、收益率等后计算数据

兼容性说明
----------
1. 尽量保持原有对外接口不变：
   - create_manager
   - MarketDataManager.update_one_daily_price
   - MarketDataManager.update_daily_prices
   - MarketDataStore.get_daily_prices
   等接口仍可继续使用。
2. 新增了“raw + 复权因子 + 物化日线”的架构：
   - daily_prices_raw     : 永久保留未复权原始日线
   - asset_adj_factors    : 统一复权因子表（股票/ETF/基金）
   - daily_prices         : 对下游暴露的默认口径日线（raw/qfq/hfq 之一）
3. 默认价格口径由系统元数据控制。初始默认为 raw。
   调用迁移/物化函数后，可以把 daily_prices 切换为 qfq/hfq，
   后续 update_one_daily_price / update_daily_prices 会自动维持该口径。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import tushare as ts


# =========================
# 基础工具
# =========================
def load_tushare_token(token_path="tushare_token.txt") -> str:
    """
    项目目录下放置tushare_token.txt，里面写一行token。
    """
    token = Path(token_path).read_text(encoding="utf-8").strip()
    return token


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def ensure_parent_dir(file_path: str | Path) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def chunked(seq: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def normalize_asset_type(asset_type: Optional[str]) -> Optional[str]:
    """
    统一资产类型标识：
    - stock: A股股票
    - fund : 场内基金 / ETF / LOF 等

    允许传入：
    - stock / equity / E
    - fund / etf / lof / FD
    """
    if asset_type is None:
        return None

    x = str(asset_type).strip().lower()
    if x in {"stock", "equity", "e", "a_share", "ashare"}:
        return "stock"
    if x in {"fund", "etf", "lof", "fd", "fund_etf"}:
        return "fund"
    raise ValueError(f"不支持的 asset_type: {asset_type!r}")


def infer_asset_type_from_ts_code(ts_code: str) -> str:
    """
    当数据库中还没有该资产的基础信息时，尽量通过代码做启发式判断。

    规则并非完美，但对常见 A 股/ETF 场景通常足够：
    - 000/001/002/003/300/301/600/601/603/605/688/689 等 => stock
    - 1xx/5xx/16x/18x/50x/51x/56x/58x 等基金/ETF 常见前缀 => fund
    """
    code = str(ts_code).split(".")[0]

    fund_prefixes = (
        "1", "5",
        "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169",
        "500", "501", "502", "503", "505", "506", "508",
        "510", "511", "512", "513", "514", "515", "516", "517", "518", "519",
        "560", "561", "562", "563", "564", "565", "566", "567", "568", "569",
        "580", "581", "582", "583", "584", "585", "586", "587", "588", "589",
        "160", "161", "162", "163", "164", "165", "166", "167", "168", "169",
        "184",
    )
    stock_prefixes = (
        "000", "001", "002", "003",
        "300", "301",
        "600", "601", "603", "605",
        "688", "689",
        "430", "831", "832", "833", "834", "835", "836", "837", "838", "839",
        "870", "871", "872", "873", "874", "875", "876", "877", "878", "879",
        "920",
    )

    if code.startswith(stock_prefixes):
        return "stock"
    if code.startswith(fund_prefixes):
        return "fund"

    # 最后的兜底：1/5 开头大概率是基金，其他常见股票前缀走股票
    if code.startswith(("1", "5")):
        return "fund"
    return "stock"


def normalize_adjust_type(adjust_type: Optional[str]) -> str:
    """
    统一复权口径：
    - raw
    - qfq
    - hfq
    """
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


# =========================
# 配置
# =========================

@dataclass
class MarketDataConfig:
    tushare_token: str
    db_path: str = "data/db/market_data.db"
    default_start_date: str = "20100101"
    default_exchange: str = "SSE"


# =========================
# Tushare 客户端
# =========================

class TushareClient:
    def __init__(self, token: str):
        self.token = token
        ts.set_token(token)
        self.pro = ts.pro_api(token)

    # -------- 基础信息 --------

    def fetch_fund_basic(
        self,
        market: str = "E",
        status: str = "L",
    ) -> pd.DataFrame:
        """
        获取场内基金基础信息
        market='E' 常用于场内基金
        status='L' 表示上市状态
        """
        df = self.pro.fund_basic(market=market, status=status)
        if df.empty:
            return df

        rename_map = {
            "ts_code": "ts_code",
            "name": "name",
            "management": "management",
            "custodian": "custodian",
            "fund_type": "fund_type",
            "found_date": "found_date",
            "due_date": "due_date",
            "list_date": "list_date",
            "issue_date": "issue_date",
            "delist_date": "delist_date",
            "issue_amount": "issue_amount",
            "m_fee": "m_fee",
            "c_fee": "c_fee",
            "duration_year": "duration_year",
            "p_value": "p_value",
            "min_amount": "min_amount",
            "exp_return": "exp_return",
            "benchmark": "benchmark",
            "status": "status",
            "invest_type": "invest_type",
            "type": "type",
            "trustee": "trustee",
            "purc_startdate": "purc_startdate",
            "redm_startdate": "redm_startdate",
            "market": "market",
        }
        df = df.rename(columns=rename_map)

        keep_cols = [
            "ts_code", "name", "management", "custodian", "fund_type",
            "found_date", "due_date", "list_date", "issue_date", "delist_date",
            "issue_amount", "m_fee", "c_fee", "duration_year", "p_value",
            "min_amount", "exp_return", "benchmark", "status", "invest_type",
            "type", "trustee", "purc_startdate", "redm_startdate", "market"
        ]
        df = df[[c for c in keep_cols if c in df.columns]].copy()
        df["asset_type"] = "fund"
        df["list_status"] = df.get("status", "L")
        df["source_table"] = "fund_basic"
        df["created_at"] = now_str()
        df["updated_at"] = now_str()
        return df

    def fetch_stock_basic(
        self,
        exchange: str = "",
        list_status: str = "L",
        is_hs: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取 A 股基础信息。
        """
        kwargs = {
            "exchange": exchange,
            "list_status": list_status,
        }
        if is_hs is not None:
            kwargs["is_hs"] = is_hs

        df = self.pro.stock_basic(**kwargs)
        if df.empty:
            return df

        rename_map = {
            "ts_code": "ts_code",
            "symbol": "symbol",
            "name": "name",
            "area": "area",
            "industry": "industry",
            "fullname": "fullname",
            "cnspell": "cnspell",
            "market": "market",
            "exchange": "exchange",
            "list_status": "list_status",
            "list_date": "list_date",
            "delist_date": "delist_date",
            "is_hs": "is_hs",
        }
        df = df.rename(columns=rename_map)
        keep_cols = [c for c in rename_map.values() if c in df.columns]
        df = df[keep_cols].copy()
        df["asset_type"] = "stock"
        df["status"] = df.get("list_status", "L")
        df["source_table"] = "stock_basic"
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

        df = self.pro.trade_cal(
            exchange=exchange,
            start_date=start_date,
            end_date=end_date
        )
        if df.empty:
            return df

        df["created_at"] = now_str()
        df["updated_at"] = now_str()
        return df

    # -------- 日线数据 --------

    def fetch_fund_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取基金/ETF/LOF 日线数据（未复权原始日线）。
        """
        if end_date is None:
            end_date = today_str()

        df = self.pro.fund_daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        if df.empty:
            return df

        df["asset_type"] = "fund"
        df["source"] = "tushare"
        df["created_at"] = now_str()
        df["updated_at"] = now_str()
        return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    def fetch_stock_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取 A 股日线数据（未复权原始日线）。
        """
        if end_date is None:
            end_date = today_str()

        df = self.pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        if df.empty:
            return df

        df["asset_type"] = "stock"
        df["source"] = "tushare"
        df["created_at"] = now_str()
        df["updated_at"] = now_str()
        return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    def fetch_asset_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: Optional[str] = None,
        asset_type: Optional[str] = None,
    ) -> pd.DataFrame:
        asset_type = normalize_asset_type(asset_type) if asset_type is not None else infer_asset_type_from_ts_code(ts_code)
        if asset_type == "fund":
            return self.fetch_fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if asset_type == "stock":
            return self.fetch_stock_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        raise ValueError(f"不支持的 asset_type: {asset_type!r}")

    def fetch_fund_daily_batch(
        self,
        ts_codes: list[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        frames = []
        for ts_code in ts_codes:
            df = self.fetch_fund_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def fetch_stock_daily_batch(
        self,
        ts_codes: list[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        frames = []
        for ts_code in ts_codes:
            df = self.fetch_stock_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    # -------- 复权因子 --------

    def fetch_stock_adj_factor(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = today_str()

        df = self.pro.adj_factor(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
        if df.empty:
            return df

        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        df["asset_type"] = "stock"
        df["source"] = "tushare_adj_factor"
        df["created_at"] = now_str()
        df["updated_at"] = now_str()
        return df

    def fetch_fund_adj_factor(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = today_str()

        df = self.pro.fund_adj(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
        if df.empty:
            return df

        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        df["asset_type"] = "fund"
        df["source"] = "tushare_fund_adj"
        df["created_at"] = now_str()
        df["updated_at"] = now_str()
        return df

    def fetch_asset_adj_factor(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        asset_type: Optional[str] = None,
    ) -> pd.DataFrame:
        asset_type = normalize_asset_type(asset_type) if asset_type is not None else infer_asset_type_from_ts_code(ts_code)
        if asset_type == "fund":
            return self.fetch_fund_adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if asset_type == "stock":
            return self.fetch_stock_adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
        raise ValueError(f"不支持的 asset_type: {asset_type!r}")


# =========================
# SQLite 存储
# =========================

class MarketDataStore:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        ensure_parent_dir(self.db_path)
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    # -------- schema migration helpers --------

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
            cursor = conn.cursor()

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS instruments (
                ts_code TEXT PRIMARY KEY,
                name TEXT,
                management TEXT,
                custodian TEXT,
                fund_type TEXT,
                found_date TEXT,
                due_date TEXT,
                list_date TEXT,
                issue_date TEXT,
                delist_date TEXT,
                issue_amount REAL,
                m_fee REAL,
                c_fee REAL,
                duration_year REAL,
                p_value REAL,
                min_amount REAL,
                exp_return REAL,
                benchmark TEXT,
                status TEXT,
                invest_type TEXT,
                type TEXT,
                trustee TEXT,
                purc_startdate TEXT,
                redm_startdate TEXT,
                market TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            """)

            # 给老表做 schema 增量迁移
            self._ensure_columns(conn, "instruments", {
                "asset_type": "TEXT",
                "symbol": "TEXT",
                "exchange": "TEXT",
                "list_status": "TEXT",
                "area": "TEXT",
                "industry": "TEXT",
                "fullname": "TEXT",
                "cnspell": "TEXT",
                "is_hs": "TEXT",
                "source_table": "TEXT",
            })

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_prices (
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

            self._ensure_columns(conn, "daily_prices", {
                "asset_type": "TEXT",
                "adjust_type": "TEXT",
            })

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_prices_raw (
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
                asset_type TEXT,
                adjust_type TEXT,
                PRIMARY KEY (ts_code, trade_date)
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS asset_adj_factors (
                ts_code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                adj_factor REAL,
                asset_type TEXT,
                source TEXT,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (ts_code, trade_date)
            );
            """)

            cursor.execute("""
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

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS update_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                ts_code TEXT,
                start_date TEXT,
                end_date TEXT,
                row_count INTEGER,
                status TEXT,
                message TEXT,
                run_time TEXT,
                created_at TEXT
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_meta (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            );
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_daily_prices_trade_date
            ON daily_prices (trade_date);
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_daily_prices_raw_trade_date
            ON daily_prices_raw (trade_date);
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_adj_factors_trade_date
            ON asset_adj_factors (trade_date);
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trade_calendar_is_open
            ON trade_calendar (exchange, is_open, cal_date);
            """)

            conn.commit()

        # 初始化默认口径
        if self.get_meta("default_adjust_type") is None:
            self.set_meta("default_adjust_type", "raw")

    # -------- 系统元数据 --------

    def set_meta(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute("""
                INSERT INTO system_meta (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=excluded.updated_at
            """, (key, value, now_str()))
            conn.commit()

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT value FROM system_meta WHERE key = ?",
                (key,),
            ).fetchone()
        return row[0] if row else default

    def get_default_adjust_type(self) -> str:
        return normalize_adjust_type(self.get_meta("default_adjust_type", "raw"))

    def set_default_adjust_type(self, adjust_type: str) -> None:
        self.set_meta("default_adjust_type", normalize_adjust_type(adjust_type))

    # -------- 通用日志 --------

    def log_update(
        self,
        table_name: str,
        ts_code: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        row_count: int,
        status: str,
        message: str = "",
    ) -> None:
        with self.connect() as conn:
            conn.execute("""
                INSERT INTO update_log (
                    table_name, ts_code, start_date, end_date,
                    row_count, status, message, run_time, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                table_name, ts_code, start_date, end_date,
                row_count, status, message, now_str(), now_str()
            ))
            conn.commit()

    # -------- 保存基础信息 --------

    def upsert_instruments(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        all_cols = [
            "ts_code", "symbol", "name", "area", "industry", "fullname", "cnspell",
            "management", "custodian", "fund_type",
            "found_date", "due_date", "list_date", "issue_date", "delist_date",
            "issue_amount", "m_fee", "c_fee", "duration_year", "p_value",
            "min_amount", "exp_return", "benchmark",
            "status", "list_status", "invest_type", "type", "trustee",
            "purc_startdate", "redm_startdate", "market", "exchange", "is_hs",
            "asset_type", "source_table",
            "created_at", "updated_at",
        ]

        x = df.copy()
        if "created_at" not in x.columns:
            x["created_at"] = now_str()
        x["updated_at"] = now_str()

        # fund_basic 里通常只有 status；stock_basic 里通常只有 list_status
        if "status" not in x.columns and "list_status" in x.columns:
            x["status"] = x["list_status"]
        if "list_status" not in x.columns and "status" in x.columns:
            x["list_status"] = x["status"]

        for col in all_cols:
            if col not in x.columns:
                x[col] = None

        x = x[all_cols]
        records = x.to_dict("records")

        sql = """
        INSERT INTO instruments (
            ts_code, symbol, name, area, industry, fullname, cnspell,
            management, custodian, fund_type,
            found_date, due_date, list_date, issue_date, delist_date,
            issue_amount, m_fee, c_fee, duration_year, p_value,
            min_amount, exp_return, benchmark,
            status, list_status, invest_type, type, trustee,
            purc_startdate, redm_startdate, market, exchange, is_hs,
            asset_type, source_table, created_at, updated_at
        ) VALUES (
            :ts_code, :symbol, :name, :area, :industry, :fullname, :cnspell,
            :management, :custodian, :fund_type,
            :found_date, :due_date, :list_date, :issue_date, :delist_date,
            :issue_amount, :m_fee, :c_fee, :duration_year, :p_value,
            :min_amount, :exp_return, :benchmark,
            :status, :list_status, :invest_type, :type, :trustee,
            :purc_startdate, :redm_startdate, :market, :exchange, :is_hs,
            :asset_type, :source_table, :created_at, :updated_at
        )
        ON CONFLICT(ts_code) DO UPDATE SET
            symbol=excluded.symbol,
            name=excluded.name,
            area=excluded.area,
            industry=excluded.industry,
            fullname=excluded.fullname,
            cnspell=excluded.cnspell,
            management=excluded.management,
            custodian=excluded.custodian,
            fund_type=excluded.fund_type,
            found_date=excluded.found_date,
            due_date=excluded.due_date,
            list_date=excluded.list_date,
            issue_date=excluded.issue_date,
            delist_date=excluded.delist_date,
            issue_amount=excluded.issue_amount,
            m_fee=excluded.m_fee,
            c_fee=excluded.c_fee,
            duration_year=excluded.duration_year,
            p_value=excluded.p_value,
            min_amount=excluded.min_amount,
            exp_return=excluded.exp_return,
            benchmark=excluded.benchmark,
            status=excluded.status,
            list_status=excluded.list_status,
            invest_type=excluded.invest_type,
            type=excluded.type,
            trustee=excluded.trustee,
            purc_startdate=excluded.purc_startdate,
            redm_startdate=excluded.redm_startdate,
            market=excluded.market,
            exchange=excluded.exchange,
            is_hs=excluded.is_hs,
            asset_type=excluded.asset_type,
            source_table=excluded.source_table,
            updated_at=excluded.updated_at
        ;
        """

        with self.connect() as conn:
            conn.executemany(sql, records)
            conn.commit()

    # -------- 保存日线 --------

    def _normalize_price_records(
        self,
        df: pd.DataFrame,
        default_adjust_type: str,
    ) -> pd.DataFrame:
        x = df.copy()
        if "source" not in x.columns:
            x["source"] = "tushare"
        if "created_at" not in x.columns:
            x["created_at"] = now_str()
        x["updated_at"] = now_str()
        if "asset_type" not in x.columns:
            x["asset_type"] = None
        if "adjust_type" not in x.columns:
            x["adjust_type"] = default_adjust_type

        keep_cols = [
            "ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "change", "pct_chg", "vol", "amount",
            "source", "created_at", "updated_at", "asset_type", "adjust_type"
        ]
        for c in keep_cols:
            if c not in x.columns:
                x[c] = None
        return x[keep_cols]

    def upsert_daily_prices(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        x = self._normalize_price_records(df, default_adjust_type="raw")
        records = x.to_dict("records")

        sql = """
        INSERT INTO daily_prices (
            ts_code, trade_date, open, high, low, close,
            pre_close, change, pct_chg, vol, amount,
            source, created_at, updated_at, asset_type, adjust_type
        ) VALUES (
            :ts_code, :trade_date, :open, :high, :low, :close,
            :pre_close, :change, :pct_chg, :vol, :amount,
            :source, :created_at, :updated_at, :asset_type, :adjust_type
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
            asset_type=excluded.asset_type,
            adjust_type=excluded.adjust_type,
            updated_at=excluded.updated_at
        ;
        """

        with self.connect() as conn:
            conn.executemany(sql, records)
            conn.commit()

    def upsert_daily_prices_raw(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        x = self._normalize_price_records(df, default_adjust_type="raw")
        x["adjust_type"] = "raw"
        records = x.to_dict("records")

        sql = """
        INSERT INTO daily_prices_raw (
            ts_code, trade_date, open, high, low, close,
            pre_close, change, pct_chg, vol, amount,
            source, created_at, updated_at, asset_type, adjust_type
        ) VALUES (
            :ts_code, :trade_date, :open, :high, :low, :close,
            :pre_close, :change, :pct_chg, :vol, :amount,
            :source, :created_at, :updated_at, :asset_type, :adjust_type
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
            asset_type=excluded.asset_type,
            adjust_type=excluded.adjust_type,
            updated_at=excluded.updated_at
        ;
        """

        with self.connect() as conn:
            conn.executemany(sql, records)
            conn.commit()

    def replace_daily_prices_for_ts_code(self, ts_code: str, df: pd.DataFrame) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM daily_prices WHERE ts_code = ?", (ts_code,))
            conn.commit()
        if not df.empty:
            self.upsert_daily_prices(df)

    def backup_daily_prices_to_raw(
        self,
        ts_codes: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """
        将现有 daily_prices 备份到 daily_prices_raw。
        使用 upsert 方式，不会重复插入。
        """
        df = self.get_daily_prices(ts_codes=ts_codes, start_date=start_date, end_date=end_date)
        if df.empty:
            return 0

        df = df.copy()
        if "adjust_type" in df.columns:
            df["adjust_type"] = "raw"
        else:
            df["adjust_type"] = "raw"
        self.upsert_daily_prices_raw(df)
        return len(df)

    # -------- 保存复权因子 --------

    def upsert_adj_factors(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        x = df.copy()
        if "created_at" not in x.columns:
            x["created_at"] = now_str()
        x["updated_at"] = now_str()
        if "asset_type" not in x.columns:
            x["asset_type"] = None
        if "source" not in x.columns:
            x["source"] = "tushare_adj_factor"

        keep_cols = ["ts_code", "trade_date", "adj_factor", "asset_type", "source", "created_at", "updated_at"]
        for c in keep_cols:
            if c not in x.columns:
                x[c] = None
        x = x[keep_cols]
        records = x.to_dict("records")

        sql = """
        INSERT INTO asset_adj_factors (
            ts_code, trade_date, adj_factor, asset_type, source, created_at, updated_at
        ) VALUES (
            :ts_code, :trade_date, :adj_factor, :asset_type, :source, :created_at, :updated_at
        )
        ON CONFLICT(ts_code, trade_date) DO UPDATE SET
            adj_factor=excluded.adj_factor,
            asset_type=excluded.asset_type,
            source=excluded.source,
            updated_at=excluded.updated_at
        ;
        """

        with self.connect() as conn:
            conn.executemany(sql, records)
            conn.commit()

    # -------- 保存交易日历 --------

    def upsert_trade_calendar(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        df = df.copy()
        if "created_at" not in df.columns:
            df["created_at"] = now_str()
        df["updated_at"] = now_str()

        keep_cols = [
            "exchange", "cal_date", "is_open", "pretrade_date",
            "created_at", "updated_at"
        ]
        df = df[keep_cols]
        records = df.to_dict("records")

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
            conn.executemany(sql, records)
            conn.commit()

    # -------- 查询 --------

    def get_instruments(self, listed_only: bool = True) -> pd.DataFrame:
        sql = "SELECT * FROM instruments"
        if listed_only:
            sql += " WHERE COALESCE(list_status, status) = 'L'"
        sql += " ORDER BY ts_code"

        with self.connect() as conn:
            return pd.read_sql_query(sql, conn)

    def get_ts_codes(self, listed_only: bool = True) -> list[str]:
        df = self.get_instruments(listed_only=listed_only)
        if df.empty:
            return []
        return df["ts_code"].tolist()

    def get_instrument_asset_type(self, ts_code: str) -> Optional[str]:
        sql = "SELECT asset_type FROM instruments WHERE ts_code = ?"
        with self.connect() as conn:
            row = conn.execute(sql, (ts_code,)).fetchone()
        return row[0] if row and row[0] else None

    def get_latest_trade_date(self, ts_code: str) -> Optional[str]:
        sql = """
        SELECT MAX(trade_date) AS latest_trade_date
        FROM daily_prices
        WHERE ts_code = ?
        """
        with self.connect() as conn:
            row = conn.execute(sql, (ts_code,)).fetchone()
        return row[0] if row and row[0] else None

    def get_raw_latest_trade_date(self, ts_code: str) -> Optional[str]:
        sql = """
        SELECT MAX(trade_date) AS latest_trade_date
        FROM daily_prices_raw
        WHERE ts_code = ?
        """
        with self.connect() as conn:
            row = conn.execute(sql, (ts_code,)).fetchone()
        return row[0] if row and row[0] else None

    def get_price_date_range(self, ts_code: str) -> tuple[Optional[str], Optional[str]]:
        sql = """
        SELECT MIN(trade_date) AS min_date, MAX(trade_date) AS max_date
        FROM daily_prices
        WHERE ts_code = ?
        """
        with self.connect() as conn:
            row = conn.execute(sql, (ts_code,)).fetchone()
        if not row:
            return None, None
        return row[0], row[1]

    def get_raw_price_date_range(self, ts_code: str) -> tuple[Optional[str], Optional[str]]:
        sql = """
        SELECT MIN(trade_date) AS min_date, MAX(trade_date) AS max_date
        FROM daily_prices_raw
        WHERE ts_code = ?
        """
        with self.connect() as conn:
            row = conn.execute(sql, (ts_code,)).fetchone()
        if not row:
            return None, None
        return row[0], row[1]

    def get_adj_factor_date_range(self, ts_code: str) -> tuple[Optional[str], Optional[str]]:
        sql = """
        SELECT MIN(trade_date) AS min_date, MAX(trade_date) AS max_date
        FROM asset_adj_factors
        WHERE ts_code = ?
        """
        with self.connect() as conn:
            row = conn.execute(sql, (ts_code,)).fetchone()
        if not row:
            return None, None
        return row[0], row[1]

    def get_daily_prices(
        self,
        ts_codes: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        where = []
        params: list = []

        if ts_codes:
            placeholders = ",".join(["?"] * len(ts_codes))
            where.append(f"ts_code IN ({placeholders})")
            params.extend(ts_codes)

        if start_date:
            where.append("trade_date >= ?")
            params.append(start_date)

        if end_date:
            where.append("trade_date <= ?")
            params.append(end_date)

        sql = "SELECT * FROM daily_prices"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts_code, trade_date"

        with self.connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def get_raw_daily_prices(
        self,
        ts_codes: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        where = []
        params: list = []

        if ts_codes:
            placeholders = ",".join(["?"] * len(ts_codes))
            where.append(f"ts_code IN ({placeholders})")
            params.extend(ts_codes)

        if start_date:
            where.append("trade_date >= ?")
            params.append(start_date)

        if end_date:
            where.append("trade_date <= ?")
            params.append(end_date)

        sql = "SELECT * FROM daily_prices_raw"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts_code, trade_date"

        with self.connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def get_adj_factors(
        self,
        ts_codes: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        where = []
        params: list = []

        if ts_codes:
            placeholders = ",".join(["?"] * len(ts_codes))
            where.append(f"ts_code IN ({placeholders})")
            params.extend(ts_codes)

        if start_date:
            where.append("trade_date >= ?")
            params.append(start_date)

        if end_date:
            where.append("trade_date <= ?")
            params.append(end_date)

        sql = "SELECT * FROM asset_adj_factors"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts_code, trade_date"

        with self.connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def get_trade_calendar(
        self,
        exchange: str = "SSE",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        is_open: Optional[int] = None,
    ) -> pd.DataFrame:
        where = ["exchange = ?"]
        params: list = [exchange]

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
        return df["cal_date"].tolist()

    def get_missing_ts_codes(self, ts_codes: Optional[list[str]] = None) -> list[str]:
        if ts_codes is None:
            ts_codes = self.get_ts_codes()

        missing = []
        for ts_code in ts_codes:
            latest = self.get_latest_trade_date(ts_code)
            if latest is None:
                missing.append(ts_code)
        return missing

    def get_empty_or_stale_ts_codes(
        self,
        ts_codes: Optional[list[str]] = None,
        expected_latest_trade_date: Optional[str] = None,
    ) -> list[str]:
        if ts_codes is None:
            ts_codes = self.get_ts_codes()

        stale = []
        for ts_code in ts_codes:
            latest = self.get_latest_trade_date(ts_code)
            if latest is None:
                stale.append(ts_code)
            elif expected_latest_trade_date and latest < expected_latest_trade_date:
                stale.append(ts_code)
        return stale

    def get_update_log(
        self,
        table_name: Optional[str] = None,
        ts_code: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        where = []
        params: list = []

        if table_name:
            where.append("table_name = ?")
            params.append(table_name)

        if ts_code:
            where.append("ts_code = ?")
            params.append(ts_code)

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

class MarketDataManager:
    def __init__(self, config: MarketDataConfig):
        self.config = config
        self.client = TushareClient(config.tushare_token)
        self.store = MarketDataStore(config.db_path)

    # -------- 内部辅助 --------

    def resolve_asset_type(
        self,
        ts_code: str,
        asset_type: Optional[str] = None,
    ) -> str:
        if asset_type is not None:
            return normalize_asset_type(asset_type)

        db_type = self.store.get_instrument_asset_type(ts_code)
        if db_type:
            return normalize_asset_type(db_type)

        return infer_asset_type_from_ts_code(ts_code)

    def _materialize_prices_for_asset(
        self,
        ts_code: str,
        adjust_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        用 raw + adj_factor 计算某资产的默认日线口径。
        """
        adjust_type = normalize_adjust_type(adjust_type or self.store.get_default_adjust_type())
        raw = self.store.get_raw_daily_prices(ts_codes=[ts_code])

        if raw.empty:
            return pd.DataFrame()

        raw = raw.copy()
        raw["trade_date"] = raw["trade_date"].astype(str)
        raw = raw.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        if adjust_type == "raw":
            out = raw.copy()
            out["adjust_type"] = "raw"
            return out

        fac = self.store.get_adj_factors(ts_codes=[ts_code])
        fac = fac.copy()

        if fac.empty:
            raise ValueError(f"{ts_code} 缺少复权因子，无法物化 {adjust_type} 日线。")

        fac["trade_date"] = fac["trade_date"].astype(str)
        fac = fac.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        merged = raw.merge(
            fac[["ts_code", "trade_date", "adj_factor"]],
            on=["ts_code", "trade_date"],
            how="left",
        )

        # 一般 Tushare 因子是全交易日覆盖；这里加上稳健性处理
        merged["adj_factor"] = merged["adj_factor"].astype(float)
        merged["adj_factor"] = merged["adj_factor"].ffill().bfill()

        if merged["adj_factor"].isna().any():
            raise ValueError(f"{ts_code} 复权因子仍存在缺失，无法继续计算。")

        if adjust_type == "hfq":
            ratio = merged["adj_factor"]
        elif adjust_type == "qfq":
            latest_factor = float(merged["adj_factor"].iloc[-1])
            if latest_factor == 0:
                raise ValueError(f"{ts_code} 最新复权因子为 0，无法计算 qfq。")
            ratio = merged["adj_factor"] / latest_factor
        else:
            raise ValueError(f"不支持的 adjust_type: {adjust_type!r}")

        for col in ["open", "high", "low", "close"]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce") * ratio

        # 让 pre_close / change / pct_chg 与调整后的 close 保持内部一致
        merged = merged.sort_values("trade_date").reset_index(drop=True)
        adj_close = merged["close"].astype(float)
        adj_pre_close = adj_close.shift(1)

        # 第一天尽量保留一个可解释的 pre_close
        first_ratio = float(ratio.iloc[0])
        first_raw_pre_close = pd.to_numeric(merged.loc[0, "pre_close"], errors="coerce")
        if pd.notna(first_raw_pre_close):
            adj_pre_close.iloc[0] = float(first_raw_pre_close) * first_ratio

        merged["pre_close"] = adj_pre_close
        merged["change"] = merged["close"] - merged["pre_close"]
        merged["pct_chg"] = np.where(
            pd.notna(merged["pre_close"]) & (merged["pre_close"] != 0),
            merged["change"] / merged["pre_close"] * 100.0,
            np.nan,
        )

        merged["adjust_type"] = adjust_type
        merged["updated_at"] = now_str()
        if "created_at" not in merged.columns:
            merged["created_at"] = now_str()

        return merged.drop(columns=["adj_factor"], errors="ignore")

    def ensure_adj_factors_for_asset(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        asset_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        确保某资产在指定时间范围内存在复权因子。
        如果本地没有，就按资产类型自动拉取。
        """
        asset_type = self.resolve_asset_type(ts_code, asset_type=asset_type)

        if start_date is None or end_date is None:
            raw_start, raw_end = self.store.get_raw_price_date_range(ts_code)
            if start_date is None:
                start_date = raw_start
            if end_date is None:
                end_date = raw_end

        if not start_date or not end_date:
            return pd.DataFrame()

        existing = self.store.get_adj_factors(
            ts_codes=[ts_code],
            start_date=start_date,
            end_date=end_date,
        )
        existing_dates = set(existing["trade_date"].astype(str).tolist()) if len(existing) > 0 else set()

        open_dates = set(
            self.store.get_open_trade_dates(
                exchange=self.config.default_exchange,
                start_date=start_date,
                end_date=end_date,
            )
        )
        # 对基金和股票，在这里统一使用交易所交易日历做覆盖判定
        expected_dates = sorted(d for d in open_dates if start_date <= d <= end_date)

        if expected_dates and len(existing_dates) >= len(expected_dates):
            return existing

        fetched = self.client.fetch_asset_adj_factor(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            asset_type=asset_type,
        )
        if not fetched.empty:
            fetched["asset_type"] = asset_type
            self.store.upsert_adj_factors(fetched)
            self.store.log_update(
                table_name="asset_adj_factors",
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                row_count=len(fetched),
                status="success",
                message=f"asset_type={asset_type}",
            )
        else:
            self.store.log_update(
                table_name="asset_adj_factors",
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                row_count=0,
                status="warning",
                message=f"empty result, asset_type={asset_type}",
            )

        return self.store.get_adj_factors(
            ts_codes=[ts_code],
            start_date=start_date,
            end_date=end_date,
        )

    def rematerialize_one_asset(
        self,
        ts_code: str,
        adjust_type: Optional[str] = None,
    ) -> pd.DataFrame:
        adjust_type = normalize_adjust_type(adjust_type or self.store.get_default_adjust_type())

        if adjust_type != "raw":
            self.ensure_adj_factors_for_asset(ts_code)

        materialized = self._materialize_prices_for_asset(ts_code, adjust_type=adjust_type)
        if not materialized.empty:
            self.store.replace_daily_prices_for_ts_code(ts_code, materialized)
        return materialized

    def rematerialize_daily_prices(
        self,
        ts_codes: Optional[list[str]] = None,
        adjust_type: Optional[str] = None,
        listed_only: bool = True,
    ) -> dict[str, int]:
        adjust_type = normalize_adjust_type(adjust_type or self.store.get_default_adjust_type())
        if ts_codes is None:
            raw_df = self.store.get_raw_daily_prices()
            if raw_df.empty:
                ts_codes = self.store.get_ts_codes(listed_only=listed_only)
            else:
                ts_codes = sorted(raw_df["ts_code"].astype(str).unique().tolist())

        updated_count = 0
        replaced_rows = 0

        for ts_code in ts_codes:
            df = self.rematerialize_one_asset(ts_code=ts_code, adjust_type=adjust_type)
            if not df.empty:
                updated_count += 1
                replaced_rows += len(df)

        self.store.set_default_adjust_type(adjust_type)
        return {
            "adjust_type": adjust_type,
            "asset_count": len(ts_codes),
            "updated_count": updated_count,
            "replaced_rows": replaced_rows,
        }

    # -------- 初始化 --------

    def initialize_basic_data(
        self,
        fund_market: str = "E",
        fund_status: str = "L",
        cal_start_date: str = "20100101",
        cal_end_date: Optional[str] = None,
        exchange: Optional[str] = None,
        include_stocks: bool = False,
        stock_exchange: str = "",
        stock_list_status: str = "L",
    ) -> None:
        """
        保持原接口语义：
        默认仍初始化基金基础信息 + 交易日历。
        如需同时初始化股票基础信息，可设置 include_stocks=True。
        """
        self.update_instruments(market=fund_market, status=fund_status)
        if include_stocks:
            self.update_stock_instruments(exchange=stock_exchange, list_status=stock_list_status)
        self.update_trade_calendar(
            exchange=exchange or self.config.default_exchange,
            start_date=cal_start_date,
            end_date=cal_end_date,
        )

    # -------- 更新基础信息 --------

    def update_instruments(
        self,
        market: str = "E",
        status: str = "L",
    ) -> pd.DataFrame:
        """
        保持向后兼容：仍表示更新基金/ETF 基础信息。
        """
        df = self.client.fetch_fund_basic(market=market, status=status)
        if not df.empty:
            self.store.upsert_instruments(df)

        self.store.log_update(
            table_name="instruments",
            ts_code=None,
            start_date=None,
            end_date=None,
            row_count=len(df),
            status="success",
            message=f"market={market}, status={status}, asset_type=fund",
        )
        return df

    def update_stock_instruments(
        self,
        exchange: str = "",
        list_status: str = "L",
        is_hs: Optional[str] = None,
    ) -> pd.DataFrame:
        df = self.client.fetch_stock_basic(
            exchange=exchange,
            list_status=list_status,
            is_hs=is_hs,
        )
        if not df.empty:
            self.store.upsert_instruments(df)

        self.store.log_update(
            table_name="instruments",
            ts_code=None,
            start_date=None,
            end_date=None,
            row_count=len(df),
            status="success",
            message=f"exchange={exchange}, list_status={list_status}, asset_type=stock",
        )
        return df

    def update_trade_calendar(
        self,
        exchange: str = "SSE",
        start_date: str = "20100101",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        df = self.client.fetch_trade_calendar(
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
        )
        if not df.empty:
            self.store.upsert_trade_calendar(df)

        self.store.log_update(
            table_name="trade_calendar",
            ts_code=None,
            start_date=start_date,
            end_date=end_date or today_str(),
            row_count=len(df),
            status="success",
            message=f"exchange={exchange}",
        )
        return df

    # -------- 更新日线 --------

    def update_one_daily_price(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        asset_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        向后兼容的单资产日线更新入口。

        新逻辑：
        1. 原始日线始终落到 daily_prices_raw
        2. 再按当前默认口径（raw/qfq/hfq）物化到 daily_prices
        """
        if end_date is None:
            end_date = today_str()

        asset_type = self.resolve_asset_type(ts_code, asset_type=asset_type)

        if start_date is None:
            latest_raw = self.store.get_raw_latest_trade_date(ts_code)
            if latest_raw is None:
                start_date = self.config.default_start_date
            else:
                dt = datetime.strptime(latest_raw, "%Y%m%d") + timedelta(days=1)
                start_date = dt.strftime("%Y%m%d")

        if start_date > end_date:
            self.store.log_update(
                table_name="daily_prices_raw",
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                row_count=0,
                status="skipped",
                message="already up to date",
            )
            return pd.DataFrame()

        raw_df = self.client.fetch_asset_daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            asset_type=asset_type,
        )

        if not raw_df.empty:
            raw_df["asset_type"] = asset_type
            raw_df["adjust_type"] = "raw"
            self.store.upsert_daily_prices_raw(raw_df)

        self.store.log_update(
            table_name="daily_prices_raw",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            row_count=len(raw_df),
            status="success",
            message=f"asset_type={asset_type}",
        )

        default_adjust_type = self.store.get_default_adjust_type()
        if default_adjust_type == "raw":
            # raw 口径下直接同步 materialized 日线
            if not raw_df.empty:
                materialized = raw_df.copy()
                materialized["adjust_type"] = "raw"
                self.store.upsert_daily_prices(materialized)
            else:
                materialized = pd.DataFrame()
        else:
            if not raw_df.empty:
                # 注意：qfq 随最新因子变化，需要重算该资产的全历史 materialized 日线
                self.ensure_adj_factors_for_asset(ts_code=ts_code, asset_type=asset_type)
                materialized = self.rematerialize_one_asset(ts_code=ts_code, adjust_type=default_adjust_type)
            else:
                materialized = pd.DataFrame()

        return materialized

    def update_one_stock_daily_price(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.update_one_daily_price(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            asset_type="stock",
        )

    def update_daily_prices(
        self,
        ts_codes: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        listed_only: bool = True,
        asset_type_map: Optional[dict[str, str]] = None,
    ) -> dict[str, int]:
        if end_date is None:
            end_date = today_str()

        if ts_codes is None:
            ts_codes = self.store.get_ts_codes(listed_only=listed_only)

        summary = {
            "instrument_count": 0,
            "updated_count": 0,
            "inserted_rows": 0,
        }

        asset_type_map = asset_type_map or {}

        for ts_code in ts_codes:
            summary["instrument_count"] += 1
            df = self.update_one_daily_price(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                asset_type=asset_type_map.get(ts_code),
            )
            if not df.empty:
                summary["updated_count"] += 1
                summary["inserted_rows"] += len(df)

        return summary

    def update_stock_daily_prices(
        self,
        ts_codes: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, int]:
        return self.update_daily_prices(
            ts_codes=ts_codes,
            start_date=start_date,
            end_date=end_date,
            listed_only=False,
            asset_type_map={ts_code: "stock" for ts_code in ts_codes},
        )

    def refresh_recent_daily_prices(
        self,
        ts_codes: Optional[list[str]] = None,
        lookback_days: int = 30,
        end_date: Optional[str] = None,
        listed_only: bool = True,
        asset_type_map: Optional[dict[str, str]] = None,
    ) -> dict[str, int]:
        if end_date is None:
            end_date = today_str()

        start_date = (
            datetime.strptime(end_date, "%Y%m%d") - timedelta(days=lookback_days)
        ).strftime("%Y%m%d")

        return self.update_daily_prices(
            ts_codes=ts_codes,
            start_date=start_date,
            end_date=end_date,
            listed_only=listed_only,
            asset_type_map=asset_type_map,
        )

    # -------- 复权迁移 / 物化 --------

    def migrate_to_materialized_adjusted_prices(
        self,
        ts_codes: Optional[list[str]] = None,
        adjust_type: str = "qfq",
        listed_only: bool = False,
        backup_existing_daily: bool = True,
        asset_type_map: Optional[dict[str, str]] = None,
    ) -> dict[str, int]:
        """
        将旧库升级为：
        daily_prices_raw + asset_adj_factors + materialized daily_prices

        步骤：
        1. 若 daily_prices_raw 里还没有原始行情，先把现有 daily_prices 备份进去
        2. 根据 raw 日线时间范围补齐复权因子
        3. 按指定口径（raw/qfq/hfq）重建 daily_prices
        4. 记录默认口径，后续 update_one_daily_price / update_daily_prices 会自动维持该口径
        """
        adjust_type = normalize_adjust_type(adjust_type)
        asset_type_map = asset_type_map or {}

        if ts_codes is None:
            raw_df = self.store.get_raw_daily_prices()
            if raw_df.empty:
                ts_codes = self.store.get_ts_codes(listed_only=listed_only)
            else:
                ts_codes = sorted(set(raw_df["ts_code"].astype(str).tolist()) | set(self.store.get_ts_codes(listed_only=listed_only)))

        backed_up_rows = 0
        if backup_existing_daily:
            backed_up_rows = self.store.backup_daily_prices_to_raw(ts_codes=ts_codes)

        factor_rows = 0
        updated_assets = 0
        replaced_rows = 0

        for ts_code in ts_codes:
            asset_type = self.resolve_asset_type(ts_code, asset_type=asset_type_map.get(ts_code))
            raw_start, raw_end = self.store.get_raw_price_date_range(ts_code)

            if not raw_start or not raw_end:
                # 如果 raw 里还没有，但 daily_prices 里有，backup 后这里通常会有；否则跳过
                continue

            if adjust_type != "raw":
                fac_df = self.ensure_adj_factors_for_asset(
                    ts_code=ts_code,
                    start_date=raw_start,
                    end_date=raw_end,
                    asset_type=asset_type,
                )
                factor_rows += len(fac_df)

            materialized = self.rematerialize_one_asset(
                ts_code=ts_code,
                adjust_type=adjust_type,
            )
            if not materialized.empty:
                updated_assets += 1
                replaced_rows += len(materialized)

        self.store.set_default_adjust_type(adjust_type)

        return {
            "adjust_type": adjust_type,
            "asset_count": len(ts_codes),
            "backed_up_rows": backed_up_rows,
            "updated_assets": updated_assets,
            "replaced_rows": replaced_rows,
            "factor_rows_snapshot": factor_rows,
        }

    # -------- 检查 --------

    def check_price_coverage(
        self,
        ts_codes: Optional[list[str]] = None,
        listed_only: bool = True,
    ) -> pd.DataFrame:
        if ts_codes is None:
            ts_codes = self.store.get_ts_codes(listed_only=listed_only)

        rows = []
        for ts_code in ts_codes:
            min_date, max_date = self.store.get_price_date_range(ts_code)
            rows.append({
                "ts_code": ts_code,
                "start_date": min_date,
                "end_date": max_date,
                "has_data": int(max_date is not None),
            })

        return pd.DataFrame(rows)

    def check_latest_price_status(
        self,
        ts_codes: Optional[list[str]] = None,
        exchange: Optional[str] = None,
        listed_only: bool = True,
    ) -> pd.DataFrame:
        if exchange is None:
            exchange = self.config.default_exchange

        if ts_codes is None:
            ts_codes = self.store.get_ts_codes(listed_only=listed_only)

        cal = self.store.get_trade_calendar(exchange=exchange, is_open=1)
        expected_latest = None if cal.empty else cal["cal_date"].max()

        rows = []
        for ts_code in ts_codes:
            latest = self.store.get_latest_trade_date(ts_code)
            rows.append({
                "ts_code": ts_code,
                "latest_trade_date": latest,
                "expected_latest_trade_date": expected_latest,
                "is_up_to_date": int(latest == expected_latest) if latest and expected_latest else 0,
            })

        return pd.DataFrame(rows)

    def check_adjustment_status(
        self,
        ts_codes: Optional[list[str]] = None,
        listed_only: bool = False,
    ) -> pd.DataFrame:
        if ts_codes is None:
            ts_codes = self.store.get_ts_codes(listed_only=listed_only)

        rows = []
        default_adjust_type = self.store.get_default_adjust_type()

        for ts_code in ts_codes:
            raw_start, raw_end = self.store.get_raw_price_date_range(ts_code)
            fac_start, fac_end = self.store.get_adj_factor_date_range(ts_code)
            mat_start, mat_end = self.store.get_price_date_range(ts_code)
            rows.append({
                "ts_code": ts_code,
                "default_adjust_type": default_adjust_type,
                "raw_start": raw_start,
                "raw_end": raw_end,
                "adj_factor_start": fac_start,
                "adj_factor_end": fac_end,
                "materialized_start": mat_start,
                "materialized_end": mat_end,
            })

        return pd.DataFrame(rows)


# =========================
# 便捷函数
# =========================

def create_manager(
    tushare_token: str,
    db_path: str = "data/db/market_data.db",
    default_start_date: str = "20100101",
    default_exchange: str = "SSE",
) -> MarketDataManager:
    config = MarketDataConfig(
        tushare_token=tushare_token,
        db_path=db_path,
        default_start_date=default_start_date,
        default_exchange=default_exchange,
    )
    return MarketDataManager(config)


# =========================
# 使用示例
# =========================

if __name__ == "__main__":
    TOKEN = "YOUR_TUSHARE_TOKEN"
    manager = create_manager(
        tushare_token=TOKEN,
        db_path="data/db/market_data.db",
        default_start_date="20150101",
        default_exchange="SSE",
    )

    # 1. 初始化基金基础数据 + 交易日历（保持旧用法）
    manager.initialize_basic_data(
        fund_market="E",
        fund_status="L",
        cal_start_date="20150101",
    )

    # 2. 如有需要，可补充股票基础信息
    # manager.update_stock_instruments(exchange="", list_status="L")

    # 3. 全量/增量更新资产日线（raw -> materialized）
    summary = manager.update_daily_prices()
    print("update summary:", summary)

    # 4. 一次性把旧库升级成前复权口径
    # migrate_summary = manager.migrate_to_materialized_adjusted_prices(adjust_type="qfq")
    # print("migrate summary:", migrate_summary)

    # 5. 只更新最近 30 天
    recent_summary = manager.refresh_recent_daily_prices(lookback_days=30)
    print("recent update summary:", recent_summary)

    # 6. 检查覆盖情况
    coverage = manager.check_price_coverage()
    print(coverage.head())

    # 7. 读取部分行情（这里拿到的是当前 default_adjust_type 口径的 daily_prices）
    ts_codes = manager.store.get_ts_codes()[:5]
    prices = manager.store.get_daily_prices(
        ts_codes=ts_codes,
        start_date="20240101",
        end_date=today_str(),
    )
    print(prices.head())
