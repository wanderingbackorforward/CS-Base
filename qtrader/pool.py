"""股票池引擎:P1 硬条件筛选 / P5 估值分位 / 池子加载。

职责划分(对齐 A股投资准则 v2.0):
- P1 硬条件:机器自动筛(市值/成交额/上市年限/ST + 尽力拉取的财务指标),
  拿不到稳定数据源的项目(审计意见/立案调查/质押比例)进入"人工确认清单";
- P2/P4 人工判断与理由卡:写在 pool.yaml,机器不越权;
- P5 估值分位:PE-TTM 近5年百分位;周期行业(银行/券商/钢铁/化工/有色/保险)改用 PB;
- P6 重筛:4/8/10 月底跑 pool 命令,产出新进/移出清单等人工确认。

诚实性原则:估值数据拿不到时返回 None,上层一律按"估值不可知 -> 不买"处理,
绝不拿缺失数据当"估值便宜"。
"""
from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from .config import Section
from .datafeed import DataFeed

log = logging.getLogger("qtrader.pool")


class PoolEntry(dict):
    """pool.yaml 里的一只股票(理由卡 P4):
    symbol/industry/thesis/how_it_makes_money/invalid_signals/max_wait_years"""


class PoolManager:
    def __init__(self, cfg: Section, feed: DataFeed) -> None:
        self.cfg = cfg
        self.feed = feed
        p = cfg["pool"]
        self.pool_file = Path(p.get("file", "./pool.yaml"))
        hf = p["hard_filters"]
        self.min_list_years = int(hf.get("min_list_years", 3))
        self.min_mcap = float(hf.get("min_market_cap", 3e11))
        self.min_amount = float(hf.get("min_avg_amount_60d", 1e8))
        self.min_roe = float(hf.get("min_roe_3y", 0.10))
        self.min_ocf_ratio = float(hf.get("min_ocf_ratio_3y", 0.80))
        self.max_debt = float(hf.get("max_debt_ratio", 0.60))
        self.max_goodwill = float(hf.get("max_goodwill_pct", 0.20))
        self.require_div = bool(hf.get("require_dividend_3y", True))
        self.debt_exempt = set(hf.get("exempt_debt_industries", ["银行", "保险", "券商"]))
        v = p["valuation"]
        self.val_years = int(v.get("window_years", 5))
        self.cyclic = set(v.get("cyclic_industries", ["银行", "券商", "钢铁", "化工", "有色", "保险"]))
        self.cache_dir = Path(cfg["data"].get("cache_dir", "./data/cache"))

    # ---------------- 生效池子(pool.yaml = 人工确认后的理由卡) ----------------
    def load_pool(self) -> list[PoolEntry]:
        if not self.pool_file.exists():
            log.warning("池子文件 %s 不存在:先跑 `python main.py pool` 生成候选,"
                        "人工确认(P2/P4)后写入 pool.yaml,value_dip 才会开仓", self.pool_file)
            return []
        with open(self.pool_file, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        stocks = data.get("stocks", [])
        out = []
        for s in stocks:
            if not isinstance(s, dict) or not s.get("symbol"):
                continue
            s.setdefault("industry", "")
            out.append(PoolEntry(s))
        log.info("生效池子: %d 只(%s)", len(out), self.pool_file.name)
        return out

    def industries(self, pool: list[PoolEntry]) -> dict[str, str]:
        return {e["symbol"]: str(e.get("industry", "")) for e in pool}

    def industry_of(self, symbol: str, pool: list[PoolEntry]) -> str:
        for e in pool:
            if e["symbol"] == symbol:
                return str(e.get("industry", ""))
        return ""

    # ---------------- P5 估值分位 ----------------
    def val_quantile(self, symbol: str, industry: str = "") -> float | None:
        """个股估值分位(0~1)。周期行业用 PB;PE 拿不到时尝试 PB 兜底;
        都拿不到返回 None(G4 闸门:None -> 不买,估值不可知不做猜)。"""
        q = self._quantile_from_hist(
            lambda: self.feed.stock_val_hist(symbol, "市盈率(TTM)"))
        if q is not None and industry not in self.cyclic:
            return q
        pb = self._quantile_from_hist(
            lambda: self.feed.stock_val_hist(symbol, "市净率"))
        if pb is not None and (q is None or industry in self.cyclic):
            return pb
        return q

    def index_quantile(self, index_name: str = "沪深300") -> float | None:
        """指数 PE 分位(E2 宽基闸门)。数据源约5年周频,按可得窗口计算。"""
        df = self.feed.index_pe_hist(index_name)
        if df is None or df.empty:
            return None
        pe = float(df["pe"].iloc[-1])
        if pe <= 0:
            return None
        return float((df["pe"] < pe).mean())

    @staticmethod
    def _quantile_from_hist(fetch) -> float | None:
        try:
            df = fetch()
        except Exception as e:  # noqa: BLE001
            log.warning("估值历史拉取失败: %s", e)
            return None
        if df is None or df.empty:
            return None
        v = float(df["value"].iloc[-1])
        if v <= 0:
            return None
        return float((df["value"] < v).mean())

    # ---------------- G5 财报前不买(尽力而为) ----------------
    def earnings_blackout(self, symbols: list[str], gap_days: int = 3,
                          asof: date | None = None) -> set[str]:
        """返回处于"财报/业绩预告公布前 gap_days 个交易日"内的标的。
        披露日历数据源(东财)不可用时返回空集并日志提醒——这是保护性规则,
        放行前请人工核对财报日历。"""
        asof = asof or date.today()
        period = self._current_report_period(asof)
        if not period:
            return set()
        cal = self.feed.disclosure_dates(period)
        if not cal:
            log.info("G5:披露日历不可用(数据源限制),本周期该检查降级为人工核对")
            return set()
        dates = sorted(self.feed.trade_dates())
        if not dates:
            return set()
        if asof in dates:
            window = set(dates[dates.index(asof): dates.index(asof) + gap_days + 1])
        else:
            upcoming = [d for d in dates if d > asof]
            window = set(upcoming[:gap_days]) | {asof}
        out = set()
        for s in symbols:
            try:
                d = date.fromisoformat(str(cal.get(s[2:], "")))
            except ValueError:
                continue
            if d in window:
                out.add(s)
        return out

    @staticmethod
    def _current_report_period(asof: date) -> str:
        """当前最临近的财报季预约披露期(季报期末):1-4月看上年年报,4-4月底看一季报……"""
        y = asof.year
        for m, p in ((1, f"{y-1}1231"), (2, f"{y-1}1231"), (3, f"{y-1}1231"),
                     (4, f"{y}0331"), (5, f"{y}0331"), (6, f"{y}0331"),
                     (7, f"{y}0630"), (8, f"{y}0630"), (9, f"{y}0630"),
                     (10, f"{y}0930"), (11, f"{y}0930"), (12, f"{y}0930")):
            if asof.month == m:
                return p
        return ""

    # ---------------- P1 硬条件自动筛 ----------------
    def run_hard_filter(self, max_stocks: int = 400) -> pd.DataFrame:
        """跑一遍 P1:粗筛(spot+市值) -> 逐只财务验证。
        产出 data/pool_candidates.csv,供人工 P2 判断后写入 pool.yaml。"""
        spot = self.feed.spot_all(force=True)
        df = spot[~spot["symbol"].astype(str).str.startswith("BJ")].copy()
        name = df["name"].astype(str)
        df = df[~(name.str.upper().str.contains("ST", na=False) | name.str.contains("退", na=False))]
        df = df[df["amount"] >= self.min_amount]
        syms = df["symbol"].tolist()[: max_stocks * 3]
        caps = self.feed.market_caps(syms)
        cap_series = pd.Series(caps)
        qualified = [s for s in syms if caps.get(s, 0) >= self.min_mcap]
        log.info("P1 粗筛:全A %d -> 成交额达标 %d -> 市值>=300亿 %d",
                 len(spot), len(syms), len(qualified))
        rows = []
        for i, s in enumerate(qualified[:max_stocks], 1):
            info = self._fundamentals_row(s)
            info["symbol"] = s
            info["name"] = str(df[df["symbol"] == s].iloc[0]["name"])
            info["market_cap"] = caps.get(s, 0)
            rows.append(info)
            if i % 25 == 0:
                log.info("P1 财务核验进度: %d/%d", i, min(len(qualified), max_stocks))
        out = pd.DataFrame(rows)
        cols = ["symbol", "name", "market_cap", "roe_min_3y", "debt_ratio",
                "ocf_profit_ratio", "goodwill_equity", "dividend_years",
                "list_years", "hard_pass", "manual_check"]
        for c in cols:
            if c not in out.columns:
                out[c] = ""
        out = out[cols]
        path = Path("data/pool_candidates.csv")
        path.parent.mkdir(exist_ok=True)
        out.to_csv(path, index=False, encoding="utf-8-sig")
        log.info("P1 候选清单已导出: %s(%d 只,hard_pass=全部硬条件可验证通过;"
                 "manual_check 列为数据源拿不到、需人工确认的项目)", path, len(out))
        return out

    def _fundamentals_row(self, symbol: str) -> dict:
        """逐只拉财务,返回指标行。拿不到的项标 None -> 归入 manual_check。"""
        row: dict = {"roe_min_3y": None, "debt_ratio": None, "ocf_profit_ratio": None,
                     "goodwill_equity": None, "dividend_years": None, "list_years": None}
        manual: list[str] = []
        ind = self.feed.fundamentals(symbol, years=4)
        if ind is not None and len(ind):
            years_rows = ind[ind["日期"].astype(str).str.endswith("12-31")]
            if len(years_rows) >= 3:
                recent = years_rows.head(3) if str(years_rows.iloc[0]["日期"]) >= str(years_rows.iloc[-1]["日期"]) else years_rows.tail(3)
                roe_col = next((c for c in recent.columns if "净资产收益率" in c and "加权" not in c), None)
                debt_col = next((c for c in recent.columns if "资产负债率" in c), None)
                if roe_col:
                    roe = pd.to_numeric(recent[roe_col], errors="coerce").dropna()
                    if len(roe):
                        row["roe_min_3y"] = round(float(roe.min()) / 100, 4)
                if debt_col:
                    d = pd.to_numeric(recent[debt_col], errors="coerce").dropna()
                    if len(d):
                        row["debt_ratio"] = round(float(d.iloc[-1]) / 100, 4)
        # 现金流质量:3年经营现金流合计 / 3年净利润合计
        cf = self.feed.fina_report(symbol, "现金流量表")
        lr = self.feed.fina_report(symbol, "利润表")
        if cf is not None and lr is not None:
            ocf = self._annual_value(cf, "经营活动产生的现金流量")
            profit = self._annual_value(lr, "净利润")
            if ocf is not None and profit is not None:
                row["ocf_profit_ratio"] = round(ocf / profit, 4) if profit else None
        # 商誉/净资产
        bs = self.feed.fina_report(symbol, "资产负债表")
        if bs is not None:
            gw = self._annual_value(bs, "商誉", positive_only=True)
            eq = self._annual_value(bs, "所有者权益(或股东权益)合计",
                                    positive_only=True)
            if gw is not None and eq:
                row["goodwill_equity"] = round(gw / eq, 4) if eq else None
        # 分红年数
        div = self.feed.dividends(symbol)
        if div is not None and len(div):
            d_col = next((c for c in div.columns if "派息" in str(c)), None)
            if d_col:
                this_year = date.today().year
                years_with_div = set()
                for _, r in div.iterrows():
                    try:
                        y = int(str(r.get("实施方案公告日期", ""))[:4])
                    except (ValueError, TypeError):
                        continue
                    if y >= this_year - 2 and pd.notna(r[d_col]) and str(r[d_col]) not in ("0", "0.0", ""):
                        years_with_div.add(y)
                row["dividend_years"] = len(years_with_div)
        # 上市年限(尽力)
        row["list_years"] = self._list_years(symbol)
        # 硬条件判定(None 项不判过 -> 人工)
        checks = {
            "ROE>=10%": row["roe_min_3y"] is not None and row["roe_min_3y"] >= self.min_roe,
            "负债率<=60%": row["debt_ratio"] is None or row["debt_ratio"] <= self.max_debt,
            "现金流>=80%": row["ocf_profit_ratio"] is None
                            or row["ocf_profit_ratio"] >= self.min_ocf_ratio,
            "商誉<=20%": row["goodwill_equity"] is None
                          or row["goodwill_equity"] <= self.max_goodwill,
            "近3年分红": not self.require_div
                          or (row["dividend_years"] is not None and row["dividend_years"] >= 3),
            "上市满3年": row["list_years"] is None or row["list_years"] >= self.min_list_years,
        }
        row["hard_pass"] = "PASS" if all(checks.values()) else "FAIL:" + ",".join(
            k for k, ok in checks.items() if not ok)
        manual = [k for k, v in (("ROE", row["roe_min_3y"]), ("负债率", row["debt_ratio"]),
                                 ("现金流", row["ocf_profit_ratio"]),
                                 ("商誉", row["goodwill_equity"]),
                                 ("分红", row["dividend_years"]),
                                 ("上市年限", row["list_years"])) if v is None]
        manual += ["质押比例", "审计意见", "立案调查"]  # 无稳定免费数据源,恒人工
        row["manual_check"] = "|".join(manual)
        return row

    @staticmethod
    def _annual_value(df: pd.DataFrame, col_prefix: str,
                      positive_only: bool = False) -> float | None:
        """从三大报表宽表取"最近年报"的某科目值(亿元->元)。"""
        if df is None or df.empty or col_prefix not in df.columns:
            return None
        sub = df[df["报告日"].astype(str).str.endswith("12-31")]
        if sub.empty:
            return None
        v = pd.to_numeric(sub[col_prefix], errors="coerce").dropna()
        if not len(v):
            return None
        val = float(v.iloc[-1])
        if positive_only and val <= 0:
            return None
        return val

    def _list_years(self, symbol: str) -> int | None:
        """上市年限:交易所名录缓存月更。拿不到返回 None(人工)。"""
        mkt, code = symbol[:2], symbol[2:]
        cache = self.cache_dir / "fin" / "list_dates.csv"
        if not (cache.exists() and time.time() - cache.stat().st_mtime < 30 * 86400):
            frames = []
            for fn, name in ((self.feed._try("sh.info", lambda: __import__("akshare", fromlist=["ak"]).stock_info_sh_name_code()), "sh"),
                             (self.feed._try("sz.info", lambda: __import__("akshare", fromlist=["ak"]).stock_info_sz_name_code()), "sz")):
                if fn is not None:
                    frames.append(fn)
            if not frames:
                return None
            all_df = pd.concat(frames, ignore_index=True)
            all_df.to_csv(cache, index=False)
        df = pd.read_csv(cache, dtype=str)
        code_col = next((c for c in df.columns if "代码" in c), None)
        date_col = next((c for c in df.columns if "上市" in c and "日期" in c), None)
        if code_col is None or date_col is None:
            return None
        rows = df[df[code_col] == code]
        if rows.empty:
            return None
        try:
            ipo = pd.to_datetime(rows.iloc[0][date_col]).date()
        except Exception:  # noqa: BLE001
            return None
        return round((date.today() - ipo).days / 365.25, 1)
