"""
market_data.py

基于 Tushare + SQLite 的市场数据函数库
只包含市场数据的获取、存储、检查、更新
不包含任何回测、策略、权重、收益率等后计算数据
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

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
        获取基金/ETF/LOF 日线数据
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

        df["source"] = "tushare"
        df["created_at"] = now_str()
        df["updated_at"] = now_str()
        return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

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
            CREATE INDEX IF NOT EXISTS idx_daily_prices_trade_date
            ON daily_prices (trade_date);
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trade_calendar_is_open
            ON trade_calendar (exchange, is_open, cal_date);
            """)

            conn.commit()

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

        df = df.copy()
        if "created_at" not in df.columns:
            df["created_at"] = now_str()
        df["updated_at"] = now_str()

        records = df.to_dict("records")

        sql = """
        INSERT INTO instruments (
            ts_code, name, management, custodian, fund_type,
            found_date, due_date, list_date, issue_date, delist_date,
            issue_amount, m_fee, c_fee, duration_year, p_value,
            min_amount, exp_return, benchmark, status, invest_type,
            type, trustee, purc_startdate, redm_startdate, market,
            created_at, updated_at
        ) VALUES (
            :ts_code, :name, :management, :custodian, :fund_type,
            :found_date, :due_date, :list_date, :issue_date, :delist_date,
            :issue_amount, :m_fee, :c_fee, :duration_year, :p_value,
            :min_amount, :exp_return, :benchmark, :status, :invest_type,
            :type, :trustee, :purc_startdate, :redm_startdate, :market,
            :created_at, :updated_at
        )
        ON CONFLICT(ts_code) DO UPDATE SET
            name=excluded.name,
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
            invest_type=excluded.invest_type,
            type=excluded.type,
            trustee=excluded.trustee,
            purc_startdate=excluded.purc_startdate,
            redm_startdate=excluded.redm_startdate,
            market=excluded.market,
            updated_at=excluded.updated_at
        ;
        """

        with self.connect() as conn:
            conn.executemany(sql, records)
            conn.commit()

    # -------- 保存日线 --------

    def upsert_daily_prices(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        df = df.copy()
        if "source" not in df.columns:
            df["source"] = "tushare"
        if "created_at" not in df.columns:
            df["created_at"] = now_str()
        df["updated_at"] = now_str()

        keep_cols = [
            "ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "change", "pct_chg", "vol", "amount",
            "source", "created_at", "updated_at"
        ]
        df = df[keep_cols]
        records = df.to_dict("records")

        sql = """
        INSERT INTO daily_prices (
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
            sql += " WHERE status = 'L'"
        sql += " ORDER BY ts_code"

        with self.connect() as conn:
            return pd.read_sql_query(sql, conn)

    def get_ts_codes(self, listed_only: bool = True) -> list[str]:
        df = self.get_instruments(listed_only=listed_only)
        if df.empty:
            return []
        return df["ts_code"].tolist()

    def get_latest_trade_date(self, ts_code: str) -> Optional[str]:
        sql = """
        SELECT MAX(trade_date) AS latest_trade_date
        FROM daily_prices
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

    # -------- 初始化 --------

    def initialize_basic_data(
        self,
        fund_market: str = "E",
        fund_status: str = "L",
        cal_start_date: str = "20100101",
        cal_end_date: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> None:
        self.update_instruments(market=fund_market, status=fund_status)
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
        df = self.client.fetch_fund_basic(market=market, status=status)
        if not df.empty:
            df["created_at"] = now_str()
            df["updated_at"] = now_str()
            self.store.upsert_instruments(df)

        self.store.log_update(
            table_name="instruments",
            ts_code=None,
            start_date=None,
            end_date=None,
            row_count=len(df),
            status="success",
            message=f"market={market}, status={status}",
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
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = today_str()

        if start_date is None:
            latest = self.store.get_latest_trade_date(ts_code)
            if latest is None:
                start_date = self.config.default_start_date
            else:
                dt = datetime.strptime(latest, "%Y%m%d") + timedelta(days=1)
                start_date = dt.strftime("%Y%m%d")

        if start_date > end_date:
            self.store.log_update(
                table_name="daily_prices",
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                row_count=0,
                status="skipped",
                message="already up to date",
            )
            return pd.DataFrame()

        df = self.client.fetch_fund_daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
        if not df.empty:
            self.store.upsert_daily_prices(df)

        self.store.log_update(
            table_name="daily_prices",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            row_count=len(df),
            status="success",
            message="",
        )
        return df

    def update_daily_prices(
        self,
        ts_codes: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        listed_only: bool = True,
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

        for ts_code in ts_codes:
            summary["instrument_count"] += 1
            df = self.update_one_daily_price(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            if not df.empty:
                summary["updated_count"] += 1
                summary["inserted_rows"] += len(df)

        return summary

    def refresh_recent_daily_prices(
        self,
        ts_codes: Optional[list[str]] = None,
        lookback_days: int = 30,
        end_date: Optional[str] = None,
        listed_only: bool = True,
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
        )

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

    # 1. 初始化基础数据
    manager.initialize_basic_data(
        fund_market="E",
        fund_status="L",
        cal_start_date="20150101",
    )

    # 2. 查看当前资产列表
    instruments = manager.store.get_instruments(listed_only=True)
    print("instruments:", instruments.shape)

    # 3. 全量/增量更新全部基金日线
    summary = manager.update_daily_prices()
    print("update summary:", summary)

    # 4. 只更新最近 30 天
    recent_summary = manager.refresh_recent_daily_prices(lookback_days=30)
    print("recent update summary:", recent_summary)

    # 5. 检查覆盖情况
    coverage = manager.check_price_coverage()
    print(coverage.head())

    # 6. 读取部分行情
    ts_codes = manager.store.get_ts_codes()[:5]
    prices = manager.store.get_daily_prices(
        ts_codes=ts_codes,
        start_date="20240101",
        end_date=today_str(),
    )
    print(prices.head())