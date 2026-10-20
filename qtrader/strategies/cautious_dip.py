"""策略二:谨慎抄底(ETF/老登股分批接飞刀)。

对标你说的"大跌后谨慎后可以抄底"、以及段永平式的"买跌出来的价值"——
但用机械纪律替代感觉:
- 入场观察:自 60 日高点回撤 >= 12%,且 RSI14 超卖后开始回升,且当日收阳、
  收盘位于振幅上半区(有承接,不是继续崩);
- 分批:最多 3 批,每再跌 5% 补一批,单批仓位 = 单票上限/3,不一次打满;
- 死线自保护:持仓单日暴跌 > 8% 暂停补仓 3 天(不接下落的刀);
- 出场:+10% 落袋 / 收复 MA20(趋势修复)清仓 / 均价 -10% 认错 / 15 日时间止损。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from ..config import Section
from ..indicators import drawdown_from_high, range_pos, rsi, sma
from ..models import Order, Side
from .base import Context, Strategy

log = logging.getLogger("qtrader.dip")


class CautiousDip(Strategy):
    name = "cautious_dip"

    def __init__(self, cfg: Section) -> None:
        s = cfg["strategy"][self.name]
        self.enabled = bool(s.get("enabled", True))
        self.targets: list[str] = [t["symbol"] for t in cfg["universe"]["dip_targets"]]
        self.look_high = int(s.get("lookback_high", 60))
        self.dd_trigger = float(s.get("drawdown_trigger", 0.12))
        self.rsi_win = int(s.get("rsi_window", 14))
        self.rsi_os = float(s.get("rsi_oversold", 32))
        self.tranches = int(s.get("tranche_count", 3))
        self.step = float(s.get("tranche_step", 0.05))
        self.target = float(s.get("target_profit", 0.10))
        self.time_stop = int(s.get("time_stop_days", 15))
        self.hard_stop = float(s.get("hard_stop", 0.10))
        self.max_positions = int(s.get("max_positions", 3))
        self.size_hint = float(cfg["risk"].get("max_position_pct", 0.30)) / self.tranches

    # ---------------- 入场(含分批加仓) ----------------
    def entry_signals(self, ctx: Context) -> list[Order]:
        if not self.enabled:
            return []
        orders: list[Order] = []
        held = [p for p in ctx.positions.values() if p.strategy == self.name]
        for sym in self.targets:
            pos = ctx.positions.get(sym)
            if pos is not None:
                if pos.strategy == self.name:
                    order = self._add_on(ctx, sym)
                    if order:
                        orders.append(order)
                continue  # 别的策略持有的标的,不碰
            if len(held) + len(orders) >= self.max_positions:
                break
            hist = ctx.hist.get(sym)
            if hist is None or len(hist) < self.look_high:
                continue
            why = self._entry_ok(sym, hist)
            if why:
                orders.append(Order(symbol=sym, side=Side.BUY, qty=0, ts=datetime.now(),
                                    strategy=self.name, reason=why,
                                    size_hint_pct=self.size_hint))
        return orders

    def _entry_ok(self, sym: str, hist) -> str:
        closes = hist["close"]
        dd = float(drawdown_from_high(closes, self.look_high).iloc[-1])
        if dd < self.dd_trigger:
            return ""
        r = rsi(closes, self.rsi_win)
        if not (r.iloc[-1] > r.iloc[-2] and r.iloc[-1] < self.rsi_os + 8):
            return ""  # 超卖后刚回升,但不追深V之后太远
        today = hist.iloc[-1]
        if float(today["close"]) <= float(today["open"]):
            return ""  # 收阴不接
        if range_pos(today) < 0.4:
            return ""  # 收盘趴在低位,承接不足
        return (f"谨慎抄底:自{self.look_high}日高点回撤{dd:.1%},"
                f"RSI{r.iloc[-1]:.0f}超卖回升,当日收阳")

    def _add_on(self, ctx: Context, sym: str) -> Order | None:
        p = ctx.positions[sym]
        price = ctx.price(sym)
        if price <= 0:
            return None
        n = int(p.meta.get("tranche", 1))
        last = float(p.meta.get("last_price", p.avg_cost))
        until = p.meta.get("no_add_until")
        if until and datetime.now().strftime("%Y-%m-%d") <= str(until):
            return None
        if n >= self.tranches:
            return None
        if price <= last * (1 - self.step):
            p.meta["tranche"] = n + 1
            p.meta["last_price"] = price
            return Order(symbol=sym, side=Side.BUY, qty=0, ts=datetime.now(),
                         strategy=self.name,
                         reason=f"第{n + 1}批接飞刀:较上批价再跌{self.step:.0%}",
                         size_hint_pct=self.size_hint)
        return None

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
            # 单日暴跌:暂停加仓 3 天(死线自保护)
            hist = ctx.hist.get(sym)
            if hist is not None and len(hist) >= 2:
                y = float(hist["close"].iloc[-2])
                if y > 0 and price / y - 1 <= -0.08:
                    p.meta["no_add_until"] = (
                        datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
            why = ""
            if gain >= self.target:
                why = f"抄底落袋:+{gain:.1%} 达标"
            if not why and hist is not None and len(hist) >= 20:
                ma20 = sma(hist["close"], 20).iloc[-1]
                if float(ma20) > 0 and price > float(ma20) and gain > 0:
                    why = f"趋势修复:收回MA20({gain:.1%}),还给别人涨"
            if not why and gain <= -self.hard_stop:
                why = f"认错离场:{gain:.1%} 触及-{self.hard_stop:.0%}底线"
            if not why and p.days_held >= self.time_stop:
                why = f"时间止损:持有{p.days_held}日无起色({gain:.1%})"
            if why:
                sells.append(Order(symbol=sym, side=Side.SELL, qty=0, ts=datetime.now(),
                                   strategy=self.name, reason=why))
        return sells
