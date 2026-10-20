"""行情数据层:多源 failover + 本地磁盘缓存。

数据源优先级(东财在国内服务器上最全最快;沙箱/海外网络下自动落到新浪/腾讯):
- A股快照:东财 stock_zh_a_spot_em -> 新浪 stock_zh_a_spot
- A股日K :东财 stock_zh_a_hist -> 腾讯 stock_zh_a_hist_tx -> 新浪 stock_zh_a_daily
- ETF日K :东财 fund_etf_hist_em -> 腾讯(代码当作股票拉)
- 港股   :实时统一走腾讯 qt.gtimg.cn(免费且海外可达);日K 走新浪 stock_hk_daily
- 概念板块:东财(排名/成分/历史),不可用时由 heat.py 用全A快照自聚合兜底
- 交易日历:新浪 tool_trade_date_hist_sina;失败则退化为"周一~周五"近似

健壮性原则:单个接口失败只影响该次数据,抛 DataFeedError 或返回空表,
绝不让一次网络抖动炸掉整个引擎。
"""
from __future__ import annotations

import logging
import pickle
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import akshare as ak
import pandas as pd

from .config import Section
from .models import Market, full_id, split_symbol

log = logging.getLogger("qtrader.data")

_A_CODE = re.compile(r"^\d{6}$")
_HK_CODE = re.compile(r"^\d{5}$")


class DataFeedError(Exception):
    pass


def _norm_df(df: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """把中文列名重命名为内部标准列名;缺列则报错交由上层切换数据源。"""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataFeedError(f"数据列缺失: {missing}, 实际: {list(df.columns)[:12]}")
    return df.rename(columns=cols)


class DataFeed:
    def __init__(self, cfg: Section, state_dir: Path | str) -> None:
        self.cfg = cfg
        state_dir = Path(state_dir)
        self.interval = float(cfg["data"].get("request_interval_sec", 0.35))
        cache_cfg = cfg["data"]
        self.cache_dir = Path(cache_cfg.get("cache_dir", state_dir / "cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "hist").mkdir(exist_ok=True)
        self.spot_cache_seconds = int(cache_cfg.get("spot_cache_seconds", 120))
        self.hist_cache_hours = int(cache_cfg.get("hist_cache_hours", 12))
        self.val_cache_hours = int(cache_cfg.get("val_cache_hours", 12))
        self._spot_ts = 0.0
        self._spot_df: Optional[pd.DataFrame] = None
        self._calendar: Optional[list[date]] = None
        self._calendar_day = ""

    # ---------------- 基础工具 ----------------
    def _pace(self) -> None:
        time.sleep(self.interval)

    def _try(self, name: str, fn, *args, **kwargs):
        """单源尝试:失败/空返回记 warning 返回 None,由调用方切下一源。"""
        try:
            self._pace()
            out = fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 单个数据源故障不应击穿整个系统
            log.warning("数据源失败 %s: %s", name, e)
            return None
        if out is None or (hasattr(out, "empty") and out.empty):
            log.warning("数据源空返回 %s", name)
            return None
        return out

    @staticmethod
    def _is_etf(symbol: str) -> bool:
        mkt, code = split_symbol(symbol)
        return mkt in (Market.SH, Market.SZ) and code[:3] in ("510", "511", "512", "513", "515", "516", "518", "159", "560", "561", "562", "563", "588")

    # ---------------- A股全市场快照 ----------------
    def spot_all(self, force: bool = False) -> pd.DataFrame:
        """全A实时快照,列: symbol/code/name/price/pct_chg/open/high/low/preclose/volume/amount/turnover_rate"""
        now = time.time()
        if (not force and self._spot_df is not None
                and now - self._spot_ts < self.spot_cache_seconds):
            return self._spot_df

        pkl = self.cache_dir / "spot_all.pkl"
        if not force and pkl.exists() and now - pkl.stat().st_mtime < self.spot_cache_seconds:
            df = pickle.loads(pkl.read_bytes())
            self._spot_df, self._spot_ts = df, pkl.stat().st_mtime
            return df

        df = self._spot_em() or self._spot_sina()
        if df is None:
            if self._spot_df is not None:
                log.error("全A快照所有数据源失败,退回上次缓存(可能过期)")
                return self._spot_df
            raise DataFeedError("全A快照: 所有数据源均不可用")
        self._spot_df, self._spot_ts = df, now
        pkl.write_bytes(pickle.dumps(df))
        return df

    def _spot_em(self) -> Optional[pd.DataFrame]:
        raw = self._try("em.spot", ak.stock_zh_a_spot_em)
        if raw is None:
            return None
        df = _norm_df(raw, {"代码": "code", "名称": "name", "最新价": "price",
                            "涨跌幅": "pct_chg", "今开": "open", "最高": "high",
                            "最低": "low", "昨收": "preclose", "成交量": "volume",
                            "成交额": "amount", "换手率": "turnover_rate"})
        df["code"] = df["code"].astype(str).str.zfill(6)
        df["symbol"] = df["code"].map(
            lambda c: full_id(c, Market.SH if c.startswith(("6", "9", "5")) else Market.SZ))
        df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce") / 100.0
        return self._clean_spot(df)

    def _spot_sina(self) -> Optional[pd.DataFrame]:
        raw = self._try("sina.spot", ak.stock_zh_a_spot)
        if raw is None:
            return None
        df = _norm_df(raw, {"代码": "raw", "名称": "name", "最新价": "price",
                            "涨跌幅": "pct_chg", "今开": "open", "最高": "high",
                            "最低": "low", "昨收": "preclose", "成交量": "volume",
                            "成交额": "amount"})
        def split_raw(s: str) -> str:
            s = str(s)
            return full_id(s[2:], Market(s[:2].upper()))
        df["symbol"] = df["raw"].map(split_raw)
        df["code"] = df["symbol"].str[2:]
        df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce") / 100.0
        df["turnover_rate"] = float("nan")
        return self._clean_spot(df)

    @staticmethod
    def _clean_spot(df: pd.DataFrame) -> pd.DataFrame:
        for c in ("price", "open", "high", "low", "preclose", "volume", "amount"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["price"]).reset_index(drop=True)

    # ---------------- 少量标的实时报价(A+HK,腾讯) ----------------
    def quote(self, symbols: list[str]) -> pd.DataFrame:
        """腾讯批量实时:支持 A股(SH600519) 与 港股(HK00700) 混合,列同 spot_all。"""
        import requests
        out: list[dict] = []
        batch = 60
        for i in range(0, len(symbols), batch):
            chunk = symbols[i:i + batch]
            qs = ",".join((s[:2].lower() + s[2:]) for s in chunk)
            try:
                self._pace()
                r = requests.get(f"https://qt.gtimg.cn/q={qs}", timeout=8)
                r.encoding = "gbk"
                for sym, line in zip(chunk, r.text.strip().split(";")):
                    if "=" not in line:
                        continue
                    fields = line.split("=")[1].strip('"').split("~")
                    if len(fields) < 40:
                        continue
                    is_hk = sym.startswith("HK")
                    try:
                        row = {
                            "symbol": sym,
                            "name": fields[1],
                            "price": float(fields[3]),
                            "preclose": float(fields[4]),
                            "open": float(fields[5]),
                            "high": float(fields[33]),
                            "low": float(fields[34]),
                            # A股 idx36=股数 idx37=万元;港股 idx36=股数 idx37=元
                            "volume": float(fields[36]) if fields[36] else 0.0,
                            "amount": float(fields[37]) * (1.0 if is_hk else 1e4)
                                      if fields[37] else 0.0,
                            "pct_chg": (float(fields[3]) / float(fields[4]) - 1.0
                                        if float(fields[4]) else 0.0),
                            "turnover_rate": float("nan"),
                        }
                    except (ValueError, ZeroDivisionError):
                        continue
                    out.append(row)
            except Exception as e:  # noqa: BLE001
                log.warning("腾讯实时报价失败(部分): %s", e)
        return pd.DataFrame(out)

    # ---------------- 日K ----------------
    def hist(self, symbol: str, start: str, end: str, force: bool = False) -> pd.DataFrame:
        """日K,列: dt(date)/open/close/high/low/volume/amount。带磁盘缓存。"""
        cache = self.cache_dir / "hist" / f"{symbol}.csv"
        if cache.exists() and not force:
            mtime = cache.stat().st_mtime
            df = pd.read_csv(cache, parse_dates=["dt"])
            covered = (df["dt"].min() <= pd.Timestamp(start)
                       and df["dt"].max() >= pd.Timestamp(end) - timedelta(days=4))
            if covered and time.time() - mtime < self.hist_cache_hours * 3600:
                return df[(df["dt"] >= start) & (df["dt"] <= end)].reset_index(drop=True)

        df = self._fetch_hist(symbol, start, end)
        if df is None:
            if cache.exists():  # 数据源临时不可用时,允许用过期缓存(并明确记日志)
                log.warning("%s 历史数据源不可用,使用过期缓存", symbol)
                old = pd.read_csv(cache, parse_dates=["dt"])
                return old[(old["dt"] >= start) & (old["dt"] <= end)].reset_index(drop=True)
            raise DataFeedError(f"{symbol} 日K: 所有数据源均不可用")

        # 合并缓存(增量更新)
        if cache.exists():
            old = pd.read_csv(cache, parse_dates=["dt"])
            df = pd.concat([old, df]).drop_duplicates("dt").sort_values("dt")
        df.to_csv(cache, index=False)
        return df[(df["dt"] >= start) & (df["dt"] <= end)].reset_index(drop=True)

    def _fetch_hist(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        mkt, code = split_symbol(symbol)
        s, e = start.replace("-", ""), end.replace("-", "")
        if mkt == Market.HK:
            raw = self._try("sina.hk_hist", ak.stock_hk_daily, symbol=code)
            if raw is None:
                return None
            # 新浪港股:amount 列即成交额(不同版本可能叫 turnover)
            cols = {"date": "dt", "open": "open", "high": "high",
                    "low": "low", "close": "close", "volume": "volume"}
            if "turnover" in raw.columns and "amount" not in raw.columns:
                cols["turnover"] = "amount"
            elif "amount" in raw.columns:
                cols["amount"] = "amount"
            df = _norm_df(raw, cols)
            if "amount" not in df.columns:
                df["amount"] = 0.0
            df["dt"] = pd.to_datetime(df["dt"])
            return self._clean_hist(df)

        if mkt == Market.BJ:
            # 北交所:50万开户门槛实盘无权限,腾讯/新浪日K也不可靠,源头放弃
            return None
        if self._is_etf(symbol):
            raw = (self._try("em.etf_hist", ak.fund_etf_hist_em, symbol=code,
                             period="daily", start_date=s, end_date=e, adjust="qfq")
                   or self._try("sina.etf_hist", ak.fund_etf_hist_sina,
                                symbol=mkt.lower() + code))
            if raw is None:
                return None
        else:
            raw = (self._try("em.hist", ak.stock_zh_a_hist, symbol=code, period="daily",
                             start_date=s, end_date=e, adjust="qfq")
                   or self._try("tx.hist", ak.stock_zh_a_hist_tx,
                                symbol=mkt.lower() + code, start_date=s, end_date=e)
                   or self._try("sina.a_hist", ak.stock_zh_a_daily,
                                symbol=mkt.lower() + code, adjust="qfq"))
            if raw is None:
                return None
        df = self._map_kline(raw)
        return self._clean_hist(df)

    @staticmethod
    def _map_kline(raw: pd.DataFrame) -> pd.DataFrame:
        if "日期" in raw.columns:  # 东财(成交量单位:手 -> 统一为股)
            df = _norm_df(raw, {"日期": "dt", "开盘": "open", "收盘": "close",
                                "最高": "high", "最低": "low", "成交量": "volume",
                                "成交额": "amount"})
            df["dt"] = pd.to_datetime(df["dt"])
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100
            return df
        if "date" in raw.columns:  # 腾讯/新浪(成交量单位:股,直接可用)
            df = raw.rename(columns={"date": "dt", "open": "open", "close": "close",
                                      "high": "high", "low": "low"})
            if "volume" not in df.columns and "amount" in df.columns:
                df = df.rename(columns={"amount": "volume"})
            for c in ("volume", "amount"):
                if c not in df.columns:
                    df[c] = 0.0
            keep = ["dt", "open", "close", "high", "low", "volume", "amount"]
            df = df[[c for c in keep if c in df.columns]]
            df["dt"] = pd.to_datetime(df["dt"])
            return df
        raise DataFeedError(f"未知K线列: {list(raw.columns)[:10]}")

    @staticmethod
    def _clean_hist(df: pd.DataFrame) -> pd.DataFrame:
        for c in ("open", "close", "high", "low", "volume", "amount"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return (df.dropna(subset=["close"])
                  .sort_values("dt")
                  .reset_index(drop=True))

    # ---------------- 大盘风向标(指数日K) ----------------
    def regime_hist(self, index_symbol: str, start: str, end: str) -> pd.DataFrame:
        _, code = split_symbol(index_symbol)
        raw = (self._try("sina.index_daily", ak.stock_zh_index_daily,
                         symbol=f"sh{code}" if index_symbol.startswith("SH") else f"sz{code}")
               or self._try("em.index_hist", ak.index_zh_a_hist, symbol=code,
                            period="daily", start_date=start.replace("-", ""),
                            end_date=end.replace("-", "")))
        if raw is None:
            raise DataFeedError("指数历史不可用")
        cols = {"date": "dt", "close": "close", "open": "open"} \
            if "date" in raw.columns else {"日期": "dt", "收盘": "close", "开盘": "open"}
        df = raw.rename(columns=cols)[["dt", "close"]]
        df["dt"] = pd.to_datetime(df["dt"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        return df.dropna().sort_values("dt").reset_index(drop=True)

    # ---------------- 市值批量(腾讯,fields[45]=总市值亿元) ----------------
    def market_caps(self, symbols: list[str]) -> dict[str, float]:
        """返回 {symbol: 总市值(元)};腾讯批量,60只/请求。"""
        import requests
        out: dict[str, float] = {}
        batch = 60
        for i in range(0, len(symbols), batch):
            chunk = [s for s in symbols[i:i + batch] if not s.startswith("HK")]
            if not chunk:
                continue
            qs = ",".join((s[:2].lower() + s[2:]) for s in chunk)
            try:
                self._pace()
                r = requests.get(f"https://qt.gtimg.cn/q={qs}", timeout=8)
                r.encoding = "gbk"
                for sym, line in zip(chunk, r.text.strip().split(";")):
                    if "=" not in line:
                        continue
                    f = line.split("=")[1].strip('"').split("~")
                    if len(f) <= 45 or not f[45]:
                        continue
                    try:
                        out[sym] = float(f[45]) * 1e8
                    except ValueError:
                        continue
            except Exception as e:  # noqa: BLE001
                log.warning("腾讯市值批量失败(部分): %s", e)
        return out

    # ---------------- 个股估值历史(PE-TTM / PB,百度) ----------------
    def stock_val_hist(self, symbol: str, indicator: str = "市盈率(TTM)") -> Optional[pd.DataFrame]:
        """个股估值历史,列: date/value。indicator: 市盈率(TTM) | 市净率。
        带缓存(val_cache_hours);失败返回 None(上层降级:视为估值不可知,不买)。"""
        _, code = split_symbol(symbol)
        cache = self.cache_dir / "val" / f"{symbol}_{indicator}.csv"
        cache.parent.mkdir(exist_ok=True)
        if cache.exists() and time.time() - cache.stat().st_mtime < self.val_cache_hours * 3600:
            df = pd.read_csv(cache, parse_dates=["date"])
            return df if len(df) else None
        raw = self._try("baidu.val", ak.stock_zh_valuation_baidu,
                        symbol=code, indicator=indicator, period="近五年")
        if raw is None:
            return None
        df = raw.rename(columns={"date": "date", "value": "value"})
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        if df.empty:
            return None
        df.to_csv(cache, index=False)
        return df

    # ---------------- 指数估值历史(乐咕,周频) ----------------
    def index_pe_hist(self, index_name: str = "沪深300") -> Optional[pd.DataFrame]:
        """指数PE历史(滚动市盈率),列: date/pe。数据源返回约5年周频,
        准则要求10年窗口——数据不足时按可得窗口计算并如实标注。"""
        cache = self.cache_dir / "val" / f"index_{index_name}.csv"
        cache.parent.mkdir(exist_ok=True)
        if cache.exists() and time.time() - cache.stat().st_mtime < self.val_cache_hours * 3600:
            df = pd.read_csv(cache, parse_dates=["date"])
            return df if len(df) else None
        raw = self._try("lg.index_pe", ak.stock_index_pe_lg, symbol=index_name)
        if raw is None:
            return None
        col = "滚动市盈率" if "滚动市盈率" in raw.columns else raw.columns[2]
        df = pd.DataFrame({
            "date": pd.to_datetime(raw["日期"]),
            "pe": pd.to_numeric(raw[col], errors="coerce"),
        }).dropna().sort_values("date").reset_index(drop=True)
        if df.empty:
            return None
        df.to_csv(cache, index=False)
        return df

    # ---------------- 个股财务(新浪,年频指标) ----------------
    def fundamentals(self, symbol: str, years: int = 4) -> Optional[pd.DataFrame]:
        """新浪财务指标(报告期行宽表): 日期/ROE/资产负债率 等。
        供 pool.py 做 P1 硬条件(ROE/负债率/现金流三表单拉)。"""
        _, code = split_symbol(symbol)
        cache = self.cache_dir / "fin" / f"{symbol}_indicator.csv"
        cache.parent.mkdir(exist_ok=True)
        if cache.exists() and time.time() - cache.stat().st_mtime < 7 * 86400:
            return pd.read_csv(cache, dtype={"日期": str})
        start_year = str(date.today().year - years)
        raw = self._try("sina.fin_indicator", ak.stock_financial_analysis_indicator,
                        symbol=code, start_year=start_year)
        if raw is None:
            return None
        raw.to_csv(cache, index=False)
        return raw

    def fina_report(self, symbol: str, table: str) -> Optional[pd.DataFrame]:
        """新浪三大报表(宽表:行=报告日,列=科目)。table: 现金流量表|利润表|资产负债表。"""
        mkt, code = split_symbol(symbol)
        cache = self.cache_dir / "fin" / f"{symbol}_{table}.csv"
        cache.parent.mkdir(exist_ok=True)
        if cache.exists() and time.time() - cache.stat().st_mtime < 7 * 86400:
            return pd.read_csv(cache, dtype={"报告日": str})
        raw = self._try(f"sina.{table}", ak.stock_financial_report_sina,
                        stock=f"{mkt.lower()}{code}", symbol=table)
        if raw is None:
            return None
        raw.to_csv(cache, index=False)
        return raw

    # ---------------- 分红历史(巨潮) ----------------
    def dividends(self, symbol: str) -> Optional[pd.DataFrame]:
        _, code = split_symbol(symbol)
        cache = self.cache_dir / "fin" / f"{symbol}_dividend.csv"
        cache.parent.mkdir(exist_ok=True)
        if cache.exists() and time.time() - cache.stat().st_mtime < 7 * 86400:
            return pd.read_csv(cache, dtype=str)
        raw = self._try("cninfo.dividend", ak.stock_dividend_cninfo, symbol=code)
        if raw is None:
            return None
        raw.to_csv(cache, index=False)
        return raw

    # ---------------- 财报预约披露日历(东财,尽力而为) ----------------
    def disclosure_dates(self, period: str) -> dict[str, str]:
        """返回 {code: 预约披露日}。东财接口,沙箱/海外环境不可用时返回空 dict,
        G5 检查自动降级为"仅日志提醒"。period: '20260930' 等报告期末。"""
        cache = self.cache_dir / "fin" / f"disclosure_{period}.csv"
        cache.parent.mkdir(exist_ok=True)
        if cache.exists() and time.time() - cache.stat().st_mtime < 7 * 86400:
            df = pd.read_csv(cache, dtype=str)
        else:
            raw = self._try("em.disclosure", ak.stock_yysj_em,
                            symbol="沪深A股", date=period)
            if raw is None:
                return {}
            df = raw
            df.to_csv(cache, index=False)
        if df.empty:
            return {}
        code_col = next((c for c in ("代码", "股票代码") if c in df.columns), None)
        date_col = next((c for c in df.columns if "预约" in str(c) and "时间" in str(c)), None)
        if not code_col or not date_col:
            return {}
        return dict(zip(df[code_col].astype(str), df[date_col].astype(str)))

    # ---------------- 概念板块(东财) ----------------
    def concept_rank(self) -> Optional[pd.DataFrame]:
        """返回列: name/day_chg/amount;不可用返回 None(由热度引擎自聚合兜底)。"""
        raw = self._try("em.concepts", ak.stock_board_concept_name_em)
        if raw is None:
            return None
        cols = {"板块名称": "name", "涨跌幅": "day_chg"}
        if "总市值" in raw.columns:
            cols["总市值"] = "amount"
        elif "成交额" in raw.columns:
            cols["成交额"] = "amount"
        df = raw.rename(columns=cols)
        if "amount" not in df.columns:
            df["amount"] = 0.0
        df["day_chg"] = pd.to_numeric(df["day_chg"], errors="coerce") / 100.0
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        return df.dropna(subset=["name"])[["name", "day_chg", "amount"]]

    def concept_cons(self, board_name: str) -> pd.DataFrame:
        raw = self._try("em.concept_cons", ak.stock_board_concept_cons_em,
                        symbol=board_name)
        if raw is None:
            return pd.DataFrame()
        cols = {"代码": "code", "名称": "name"}
        if "代码" not in raw.columns:
            return pd.DataFrame()
        df = raw.rename(columns=cols)[["code", "name"]]
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df

    def concept_hist(self, board_name: str, start: str, end: str) -> Optional[pd.DataFrame]:
        raw = self._try("em.concept_hist", ak.stock_board_concept_hist_em,
                        symbol=board_name, period="daily", start_date=start.replace("-", ""),
                        end_date=end.replace("-", ""), adjust="")
        if raw is None:
            return None
        df = _norm_df(raw, {"日期": "dt", "收盘": "close"})
        df["dt"] = pd.to_datetime(df["dt"])
        return df.sort_values("dt").reset_index(drop=True)

    # ---------------- 交易日历 ----------------
    def trade_dates(self) -> list[date]:
        today_str = date.today().isoformat()
        if self._calendar and self._calendar_day == today_str:
            return self._calendar
        raw = self._try("sina.calendar", ak.tool_trade_date_hist_sina)
        if raw is None:
            log.warning("交易日历不可用,退化为周一~周五近似(节假日会空转,不影响安全)")
            return []
        if "trade_date" in raw.columns:
            dates = pd.to_datetime(raw["trade_date"]).dt.date.tolist()
        else:
            dates = [d for d in pd.to_datetime(raw.iloc[:, 0]).dt.date.tolist()]
        self._calendar, self._calendar_day = dates, today_str
        return dates

    def is_trading_day(self, d: Optional[date] = None) -> bool:
        d = d or date.today()
        cal = self.trade_dates()
        if not cal:
            return d.weekday() < 5
        return d in cal
