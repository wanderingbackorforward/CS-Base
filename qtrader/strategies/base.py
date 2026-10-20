"""策略基类与决策上下文。

约定(对齐 A股投资准则 v2.0 的 X1 执行模式):
- 策略只在收盘后跑一遍,产出 PendingSignal(明日计划),不直接下单;
- 引擎次日开盘统一撮合 pending 信号,成交后回调 strategy.on_fill;
- 策略可测、可回测、可替换,不碰现金与订单簿。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ..models import Account, PendingSignal, Position


@dataclass
class Context:
    """决策上下文:引擎在收盘后信号周期组装,喂给所有策略。"""
    today: str                       # "2026-10-20"
    spot: pd.DataFrame = field(default_factory=pd.DataFrame)   # 全A快照
    quote: pd.DataFrame = field(default_factory=pd.DataFrame)  # 持仓/池子实时报价
    hist: dict[str, pd.DataFrame] = field(default_factory=dict)  # 日K {symbol: df}
    account: Optional[Account] = None
    positions: dict[str, Position] = field(default_factory=dict)
    market_value: float = 0.0
    # ---- 准则 v2.0 新增 ----
    pool: list = field(default_factory=list)      # 生效池子(pool.yaml, PoolEntry)
    pending: list = field(default_factory=list)   # 未成交信号(含当日新生成)
    pool_val_q: Optional[dict] = None             # 估值分位缓存 {symbol: q|None}
    blackout: set = field(default_factory=set)    # G5 财报披露前不买的标的
    name_map: dict = field(default_factory=dict)  # symbol -> 证券简称(G10 ST 检测)
    industry_map: dict = field(default_factory=dict)  # symbol -> 行业(G2 行业上限)
    month_end: bool = False                       # H5第一步/H8:今日是否月末交易日
    quarter_day: bool = False                     # C2/H5第二步/H9/P5:季度调仓日

    def equity(self) -> float:
        if self.account is not None:
            return self.account.equity(self.market_value)
        return self.market_value

    def price(self, symbol: str) -> float:
        """优先实时报价,回退快照。"""
        if self.quote is not None and len(self.quote):
            row = self.quote[self.quote["symbol"] == symbol]
            if len(row):
                return float(row.iloc[0]["price"])
        if self.spot is not None and len(self.spot):
            row = self.spot[self.spot["symbol"] == symbol]
            if len(row):
                return float(row.iloc[0]["price"])
        return 0.0

    def pct_chg(self, symbol: str) -> float:
        if self.spot is not None and len(self.spot):
            row = self.spot[self.spot["symbol"] == symbol]
            if len(row):
                return float(row.iloc[0]["pct_chg"])
        return 0.0

    def name_of(self, symbol: str) -> str:
        return self.name_map.get(symbol, "")

    def industry_of(self, symbol: str) -> str:
        return self.industry_map.get(symbol, "")

    def hist_row(self, symbol: str) -> Optional[pd.Series]:
        df = self.hist.get(symbol)
        if df is None or df.empty:
            return None
        return df.iloc[-1]


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def entry_signals(self, ctx: Context) -> list[PendingSignal]:
        """买入信号(收盘后生成,次日开盘撮合)。"""

    @abstractmethod
    def manage(self, ctx: Context) -> list[PendingSignal]:
        """卖出信号:G6/G7/G8/G10 等持仓管理。"""

    def on_fill(self, sig: PendingSignal, filled_qty: int, ok: bool,
                price: float = 0.0) -> None:
        """成交回调:引擎撮合后调用,策略用它推进档位状态机。默认无状态。"""
