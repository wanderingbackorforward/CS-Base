"""纸面撮合 broker:按 A股/港股真实规则模拟成交与费用。

A股:佣金万2.5(最低5元)+ 印花税0.05%(卖出)+ 过户费0.001%(双边)+ T+1 + 100股整手;
港股(纸面):佣金同上 + 印花税0.1%(双边,取保守值)+ T+0 + 按整手简化为100股;
滑点:成交价 = 参考价 * (1 ± slippage),买入上浮、卖出下打——纸面必须按"对我不利"的方向模拟,
这是无数纸面回测虚高的根源,我们不自己骗自己。

实盘预留:Broker 是接口,实盘时实现 QmtBroker(miniQMT)或 IbbkrBroker(港美)即可,
Engine/策略层零改动。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime

from .config import Section
from .models import Account, Market, Order, OrderStatus, Position, Side, split_symbol
from .store import Store

log = logging.getLogger("qtrader.broker")


class Broker(ABC):
    @abstractmethod
    def execute(self, order: Order, ref_price: float, account: Account,
                positions: dict[str, Position]) -> Order:
        """就地执行订单,更新 account/positions 并持久化,返回终态订单。"""


class PaperBroker(Broker):
    def __init__(self, cfg: Section, store: Store) -> None:
        self.store = store
        p = cfg["broker"]["paper"]
        self.commission_rate = float(p.get("commission_rate", 0.00025))
        self.commission_min = float(p.get("commission_min", 5.0))
        self.stamp_a = float(p.get("stamp_tax", 0.0005))
        self.stamp_hk = 0.001           # 港股印花双边 0.1%(保守)
        self.transfer_fee = float(p.get("transfer_fee", 0.00001))
        self.slippage = float(p.get("slippage", 0.001))
        self.lot = 100

    # ---------- 费用 ----------
    def fee(self, market: Market, side: Side, amount: float) -> float:
        commission = max(amount * self.commission_rate, self.commission_min)
        fee = commission
        if market == Market.HK:
            fee += amount * self.stamp_hk
        elif side == Side.SELL:
            fee += amount * self.stamp_a
        if market != Market.HK:
            fee += amount * self.transfer_fee
        return fee

    # ---------- 成交 ----------
    def execute(self, order: Order, ref_price: float, account: Account,
                positions: dict[str, Position]) -> Order:
        market, _ = split_symbol(order.symbol)
        if not ref_price or ref_price <= 0 or order.qty <= 0:
            order.status = OrderStatus.REJECTED
            order.reason = (order.reason + " | 拒单:价格无效或数量非法").strip(" |")
            self.store.save_order(order)
            return order
        if market != Market.HK and order.qty % self.lot != 0:
            order.status = OrderStatus.REJECTED
            order.reason += " | 拒单:A股须整手(100股)"
            self.store.save_order(order)
            return order

        if order.side == Side.BUY:
            price = ref_price * (1 + self.slippage)
            amount = price * order.qty
            fee = self.fee(market, Side.BUY, amount)
            if amount + fee > account.cash + 1e-6:
                order.status = OrderStatus.REJECTED
                order.reason += f" | 拒单:现金不足(需 {amount + fee:.0f})"
                self.store.save_order(order)
                return order
            account.cash -= amount + fee
            pos = positions.get(order.symbol)
            if pos is None:
                pos = Position(symbol=order.symbol, strategy=order.strategy,
                               opened_at=datetime.now())
                positions[order.symbol] = pos
            # 含费摊入成本
            total_cost = pos.avg_cost * pos.qty + amount + fee
            pos.qty += order.qty
            pos.avg_cost = total_cost / pos.qty
            pos.high_water = max(pos.high_water, price)
            order.status = OrderStatus.FILLED
            order.filled_price, order.filled_qty, order.fee = price, order.qty, fee
        else:
            pos = positions.get(order.symbol)
            sellable = pos.available_qty if pos else 0
            qty = min(order.qty, pos.qty if pos else 0)
            if qty <= 0 or sellable < qty:
                order.status = OrderStatus.REJECTED
                order.reason += (f" | 拒单:T+1可卖不足"
                                 f"(可卖 {sellable}, 想卖 {order.qty})")
                self.store.save_order(order)
                return order
            price = ref_price * (1 - self.slippage)
            amount = price * qty
            fee = self.fee(market, Side.SELL, amount)
            account.cash += amount - fee
            pos.qty -= qty
            pos.available_qty = max(0, pos.available_qty - qty)
            order.status = OrderStatus.FILLED
            order.filled_price, order.filled_qty, order.fee = price, qty, fee
            if pos.qty <= 0:
                self.store.drop_position(order.symbol)
                del positions[order.symbol]

        self.store.save_position(pos) if order.side == Side.BUY and pos.qty > 0 else None
        if order.side == Side.SELL and order.symbol in positions and positions[order.symbol].qty > 0:
            self.store.save_position(positions[order.symbol])
        self.store.save_account(account)
        self.store.save_order(order)
        log.info("成交 %s %s x%d @%.3f 费%.2f | %s", order.side.value, order.symbol,
                 order.filled_qty, order.filled_price, order.fee, order.reason)
        return order

    # ---------- T+1:新交易日开盘,昨日买入解锁可卖 ----------
    def promote_new_day(self, positions: dict[str, Position]) -> None:
        for p in positions.values():
            if p.available_qty < p.qty:
                p.available_qty = p.qty
                self.store.save_position(p)
                log.info("T+1 解锁: %s 可卖 %d", p.symbol, p.available_qty)
