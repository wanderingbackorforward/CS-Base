"""策略一:热点动量(追热点,大涨必跑)。

入场(尾盘决策,规避 T+1 当日无法止损的敞口):
- 标的在热度引擎给出的热门板块候选池内;
- 放量突破 N 日新高(量 >= 5日均量 * vol_ratio_min);
- 当日涨幅 2%~7%:强,但不追涨停、不追 20cm 一日情;
- 收盘(以决策时现价近似)位于当日振幅前 80%,承接有力;
- 站上 MA20 且 MA20 向上;近 3 日累计涨幅不过热(<15%,防止追在山顶)。

出场纪律(你的原话:"死线是大涨后必须及时跑路"):
- 死线一:昨日涨幅 >= 7%(含涨停) -> 今日无条件跑;
- 死线二:浮盈 >= 8% -> 落袋,不幻想;
- 浮盈触及 5% 后止损线上移至保本+1%,并随高点回撤 7% 移动;
- 硬止损 -6%(T+1 决定买入当天跑不了,只能吃隔夜风险,仓位管理兜底);
- 4 个交易日不创新高 -> 撤(资金效率)。
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from ..config import Section
from ..indicators import range_pos, sma, vol_ma
from ..models import Order, Position, Side
from .base import Context, Strategy

log = logging.getLogger("qtrader.hot")


class HotMomentum(Strategy):
    name = "hot_momentum"

    def __init__(self, cfg: Section) -> None:
        self.a_market = cfg["market"]["a_share"]  # 股票池过滤配置(引擎市场闸门)
        s = cfg["strategy"][self.name]
        self.enabled = bool(s.get("enabled", True))
        self.lookback = int(s.get("breakout_lookback", 20))
        self.vol_ratio = float(s.get("vol_ratio_min", 1.8))
        self.chg_min = float(s.get("day_change_min", 0.02))
        self.chg_max = float(s.get("day_change_max", 0.07))
        self.range_pos_min = float(s.get("close_range_pos_min", 0.80))
        self.take_profit = float(s.get("take_profit", 0.08))
        self.pump_exit = float(s.get("pump_exit_pct", 0.07))
        self.trail_arm = float(s.get("trailing_arm", 0.05))
        self.trail_gap = 0.07          # 高点回撤 7% 触发移动止盈
        self.hard_stop = float(s.get("hard_stop", 0.06))
        self.time_stop = int(s.get("time_stop_days", 4))
        self.max_positions = int(s.get("max_positions", 3))

    # ---------------- 入场 ----------------
    def entry_signals(self, ctx: Context) -> list[Order]:
        if not self.enabled:
            return []
        orders: list[Order] = []
        mine = [p for p in ctx.positions.values() if p.strategy == self.name]
        if len(mine) >= self.max_positions:
            return []
        for sym in ctx.candidates:
            if len(orders) + len(mine) >= self.max_positions:
                break
            if sym in ctx.positions:
                continue
            row = ctx.spot[ctx.spot["symbol"] == sym] if len(ctx.spot) else []
            if not len(row):
                continue
            r = row.iloc[0]
            if not self._pass_universe(r):
                continue
            hist = ctx.hist.get(sym)
            if hist is None or len(hist) < self.lookback + 5:
                continue
            why = self._entry_ok(sym, hist, r)
            if why:
                orders.append(Order(symbol=sym, side=Side.BUY, qty=0,
                                    ts=datetime.now(), strategy=self.name, reason=why))
        return orders

    def _pass_universe(self, r: pd.Series) -> bool:
        a = self.a_market
        name = str(r.get("name", ""))
        code = str(r.get("code", ""))
        if a.get("exclude_st", True) and ("ST" in name.upper() or "退" in name):
            return False
        if a.get("exclude_star", True) and code.startswith("688"):
            return False
        if code.startswith(("8", "4", "92")):
            return False  # 北交所
        if code.startswith("300") and not a.get("include_chinext", False):
            return False
        if not (float(a.get("min_price", 3)) <= float(r["price"]) <= float(a.get("max_price", 150))):
            return False
        return float(r["amount"]) >= float(a.get("min_amount", 1e8))

    def _entry_ok(self, sym: str, hist: pd.DataFrame, spot_row: pd.Series) -> str:
        today = hist.iloc[-1]
        past = hist.iloc[:-1]
        chg = float(spot_row["pct_chg"])
        if not (self.chg_min <= chg <= self.chg_max):
            return ""
        # 放量突破 N 日新高(成交量用近端数据,盘中为进行时,量阈值放宽一档)
        vol_ma5_prev = float(vol_ma(past["volume"], 5).iloc[-1])
        if vol_ma5_prev <= 0 or float(today["volume"]) < vol_ma5_prev * self.vol_ratio * 0.6:
            return ""
        hh = float(past["high"].iloc[-self.lookback:].max())
        price = float(spot_row["price"])
        if price <= hh:
            return ""
        if range_pos(today if today["high"] > today["low"] else spot_row) < self.range_pos_min:
            return ""
        closes = hist["close"]
        ma20 = sma(closes, 20)
        if price < float(ma20.iloc[-1]) or float(ma20.iloc[-1]) < float(ma20.iloc[-2]):
            return ""
        # 近 3 日累计涨幅不过热(以昨收基准,今日涨幅单列)
        if len(closes) >= 4:
            r3 = float(closes.iloc[-1]) / float(closes.iloc[-4]) - 1 - chg
            if r3 > 0.15:
                return ""
        return (f"热点突破:{spot_row['name']} 放量({float(today['volume'])/vol_ma5_prev:.1f}倍)"
                f"创{self.lookback}日新高,现涨{chg:.1%}")

    # ---------------- 持仓管理 ----------------
    def manage(self, ctx: Context) -> list[Order]:
        sells: list[Order] = []
        for sym, p in list(ctx.positions.items()):
            if p.strategy != self.name:
                continue
            price = ctx.price(sym)
            if price <= 0:
                continue
            gain = price / p.avg_cost - 1
            p.high_water = max(p.high_water, price)
            why = ""
            # 死线一:昨日已大涨(>=7% 或涨停),今日无条件跑
            hist = ctx.hist.get(sym)
            if hist is not None and len(hist) >= 2:
                y_close = float(hist["close"].iloc[-2])
                y_prev = float(hist["close"].iloc[-3]) if len(hist) >= 3 else y_close
                if y_prev > 0 and y_close / y_prev - 1 >= self.pump_exit:
                    why = f"死线:昨日大涨{y_close / y_prev - 1:.1%},今天必须跑"
            if not why and gain >= self.take_profit:
                why = f"落袋:浮盈{gain:.1%} >= {self.take_profit:.0%},大涨后及时跑路"
            if not why and price <= p.avg_cost * (1 - self.hard_stop):
                why = f"硬止损:{gain:.1%} 触及-{self.hard_stop:.0%}"
            if not why:
                if gain >= self.trail_arm:
                    p.meta["armed"] = True
                if p.meta.get("armed"):
                    p.stop_price = max(
                        p.stop_price,
                        p.avg_cost * 1.01,
                        p.high_water * (1 - self.trail_gap),
                    )
                if p.stop_price > 0 and price <= p.stop_price:
                    why = (f"移动止盈触发:高点{p.high_water:.2f}回撤,"
                           f"止损线{p.stop_price:.2f}")
            if not why and p.days_held >= self.time_stop and gain < 0.02:
                why = f"时间止损:持有{p.days_held}日未创新高({gain:.1%})"
            if why:
                sells.append(Order(symbol=sym, side=Side.SELL, qty=0, ts=datetime.now(),
                                   strategy=self.name, reason=why))
        return sells
