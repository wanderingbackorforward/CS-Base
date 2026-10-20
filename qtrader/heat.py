"""热度引擎:每天找出市场最热的概念板块,并给出候选股票池。

主路:东财概念板块涨幅/成交额排名(国内服务器上又快又全);
兜底:东财不可用时,直接用全A实时快照自聚合——今天全市场最强的股票本身就是热点,
     涨幅榜+成交额榜双过滤,逻辑上与"打板"同源。

你点名的关注方向(幻方->AI/算力/DeepSeek,孙宇晨->稳定币/数字货币)通过
config.heat.watch_keywords 标注,在报告与看板中高亮,方便你肉眼校验机器的判断。
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from .config import Section
from .datafeed import DataFeed
from .models import HeatItem, HeatReport

log = logging.getLogger("qtrader.heat")


class HeatEngine:
    def __init__(self, cfg: Section, feed: DataFeed) -> None:
        self.cfg = cfg
        self.feed = feed
        self.top_n = int(cfg["heat"].get("top_n", 6))
        self.min_amount = float(cfg["heat"].get("min_board_amount", 2e9))
        self.keywords: list[str] = list(cfg["heat"].get("watch_keywords", []))
        self.show_stocks = int(cfg["heat"].get("show_top_stocks", 8))

    # ---------------- 对外主入口 ----------------
    def build(self) -> HeatReport:
        try:
            boards = self.feed.concept_rank()
        except Exception as e:  # noqa: BLE001
            boards = None
            log.warning("概念板块数据不可用: %s", e)
        spot = None
        try:
            spot = self.feed.spot_all()
        except Exception as e:  # noqa: BLE001
            log.warning("全A快照不可用: %s", e)
        if boards is not None and len(boards) > 0:
            report = self._from_boards(boards)
        elif spot is not None and len(spot) > 0:
            report = self._from_spot(spot)
        else:
            report = HeatReport(ts=datetime.now(), source="empty", items=[])
        report.source = report.source or "?"
        log.info("热度报告: %s | top: %s", report.source,
                 ",".join(i.name for i in report.items[:5]))
        return report

    # ---------------- 主路:东财概念板块 ----------------
    def _from_boards(self, boards: pd.DataFrame) -> HeatReport:
        # 有成交额就用"涨幅60% + 成交额40%"打综合分,否则纯涨幅
        has_amt = boards["amount"].sum() > 0
        df = boards.copy()
        df = df[df["day_chg"].notna()]
        df["chg_rank"] = df["day_chg"].rank(ascending=False)
        df["score"] = 0.0
        if has_amt:
            df["amt_rank"] = df["amount"].rank(ascending=False)
            df["score"] = 0.6 / df["chg_rank"] + 0.4 / df["amt_rank"]
        else:
            df["score"] = 1.0 / df["chg_rank"]
        if (df["amount"] > 0).any():
            df = df[df["amount"] >= self.min_amount]
        df = df.sort_values("score", ascending=False).head(self.top_n)

        items: list[HeatItem] = []
        cons_cache: dict[str, pd.DataFrame] = {}
        for i, row in enumerate(df.itertuples(index=False), start=1):
            members = self.feed.concept_cons(row.name)
            cons_cache[row.name] = members
            top = self._top_members(members)
            items.append(HeatItem(
                name=row.name, day_chg=float(row.day_chg), amount=float(row.amount),
                score=float(row.score), rank=i,
                is_watch=self._hit(row.name), top_stocks=top,
            ))
        # 关注方向命中的板块保证进入视野(即便排名靠后)
        for row in boards.itertuples(index=False):
            if self._hit(row.name) and row.name not in cons_cache:
                members = self.feed.concept_cons(row.name)
                top = self._top_members(members)
                items.append(HeatItem(
                    name=row.name, day_chg=float(row.day_chg), amount=float(row.amount),
                    score=0.0, rank=99, is_watch=True, top_stocks=top,
                ))
        return HeatReport(ts=datetime.now(), source="em_concepts", items=items)

    def _top_members(self, members: pd.DataFrame) -> list[str]:
        """板块内按今日涨幅排序展示头部成分(名称+涨幅),供看板展示。"""
        if members is None or members.empty:
            return []
        try:
            spot = self.feed.spot_all()
            m = members.merge(spot[["code", "name", "pct_chg"]],
                              on="code", suffixes=("", "_spot"), how="inner")
            m = m.sort_values("pct_chg", ascending=False).head(self.show_stocks)
            return [f"{r['name']} {r['pct_chg']:+.1f}%"
                    for _, r in m.iterrows() if pd.notna(r["pct_chg"])]
        except Exception:  # noqa: BLE001
            return list(members["name"].head(self.show_stocks))

    # ---------------- 兜底:全A自聚合(沙箱/数据源故障时同样可用) ----------------
    def _from_spot(self, spot: pd.DataFrame) -> HeatReport:
        df = spot[spot["amount"] >= 3e8].copy()  # 只看日成交额 3 亿以上的活跃票
        df = df[df["pct_chg"].notna()].sort_values("pct_chg", ascending=False)
        strong = df[df["pct_chg"] >= 0.02].head(30)
        items = [
            HeatItem(
                name="全市场强势股聚合(数据源兜底模式)", day_chg=float(strong["pct_chg"].mean())
                if len(strong) else 0.0,
                amount=float(strong["amount"].sum()) if len(strong) else 0.0,
                score=1.0, rank=1, is_watch=False,
                top_stocks=[f"{r['name']} {r['pct_chg']:+.1%}"
                            for _, r in strong.head(self.show_stocks).iterrows()],
            )
        ]
        return HeatReport(ts=datetime.now(), source="spot_aggregate", items=items)

    # ---------------- 候选股票池(交给策略) ----------------
    def candidates(self, report: HeatReport) -> list[str]:
        """热门板块成分股并集;兜底模式则返回全A强势票。"""
        if report.source == "spot_aggregate":
            try:
                spot = self.feed.spot_all()
                df = spot[(spot["amount"] >= 3e8) & (spot["pct_chg"] >= 0.02)
                          & (spot["pct_chg"] <= 0.09)]
                df = self._tradable(df)
                return df.nlargest(40, "pct_chg")["symbol"].tolist()
            except Exception:  # noqa: BLE001
                return []
        syms: list[str] = []
        for item in report.items:
            try:
                members = self.feed.concept_cons(item.name)
                if members.empty:
                    continue
                for c in members["code"]:
                    c = str(c)
                    if c.startswith(("4", "8", "9")):
                        continue  # 北交所(43/83/87/92)/沪B(900):实盘无对应权限
                    if c.startswith(("6", "5")):
                        syms.append(f"SH{c}")
                    elif c.startswith(("0", "3")):
                        syms.append(f"SZ{c}")
            except Exception as e:  # noqa: BLE001
                log.warning("板块 %s 成分获取失败: %s", item.name, e)
        return list(dict.fromkeys(syms))  # 去重保序

    @staticmethod
    def _tradable(df: pd.DataFrame) -> pd.DataFrame:
        """可交易性过滤:北交所(50万开户门槛)与 ST/退市/停牌票不可进候选池。

        纸面在无实盘权限的市场上赚钱毫无意义,这里在源头掐掉。
        """
        df = df[~df["symbol"].astype(str).str.startswith("BJ")]
        name = df["name"].astype(str).str.upper()
        df = df[~(name.str.contains("ST", na=False) | name.str.contains("退", na=False))]
        if "price" in df.columns:
            df = df[df["price"].notna() & (df["price"] > 0)]
        return df

    def _hit(self, name: str) -> bool:
        return any(k.lower() in str(name).lower() for k in self.keywords)
