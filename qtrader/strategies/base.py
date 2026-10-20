"""策略基类与决策上下文。

约定:策略只负责"产生意图(订单+理由)与持仓管理",不负责仓位校验与成交;
仓位大小由风控统一计算,成交由 Broker 统一执行。策略可测、可回测、可替换。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ..models import Account, Order, Position


@dataclass
class Context:
    """决策上下文:引擎在每个决策点组装,喂给所有策略。"""
    today: str                       # "2026-10-20"
    spot: pd.DataFrame               # 全A快照(列见 datafeed)
    quote: pd.DataFrame              # 少量标的实时报价(A+HK,腾讯)
    hist: dict[str, pd.DataFrame]   # 候选标的日K {symbol: df}
    regime_close: Optional[pd.Series] = None  # 大盘指数收盘序列(截至今日)
    account: Optional[Account] = None
    positions: dict[str, Position] = field(default_factory=dict)
    market_value: float = 0.0
    candidates: list[str] = field(default_factory=list)  # 热点候选池

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

    def hist_row(self, symbol: str) -> Optional[pd.Series]:
        df = self.hist.get(symbol)
        if df is None or df.empty:
            return None
        return df.iloc[-1]

    def pct_chg(self, symbol: str) -> float:
        """今日涨幅(快照口径)。"""
        if self.spot is not None and len(self.spot):
            row = self.spot[self.spot["symbol"] == symbol]
            if len(row):
                return float(row.iloc[0]["pct_chg"])
        return 0.0


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def entry_signals(self, ctx: Context) -> list[Order]:
        """新开仓信号(qty 留 0,由风控定仓;引擎负责补齐数量)。"""

    @abstractmethod
    def manage(self, ctx: Context) -> list[Order]:
        """持仓管理:止盈/止损/死线/时间止损,返回卖出订单(qty=全部可卖,引擎补齐)。"""
