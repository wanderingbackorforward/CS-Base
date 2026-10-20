"""纸面撮合 broker:按 A股/港股/美股真实规则模拟成交与费用。

A股:佣金万2.5(最低5元)+ 印花税0.05%(卖出)+ 过户费0.001%(双边)+ T+1 + 100股整手;
港股(纸面):佣金同上 + 印花税0.1%(双边,取保守值)+ T+0 + 按整手简化为100股;
美股(纸面):无印花/过户费;佣金 max(交易额*费率, 每单最低18美元,汇丰标准) + T+0 + 零股可交易;
滑点:成交价 = 参考价 * (1 ± slippage),买入上浮、卖出下打——纸面必须按"对我不利"的方向模拟,
这是无数纸面回测虚高的根源,我们不自己骗自己。

实盘预留:Broker 是接口,实盘时实现 QmtBroker(miniQMT)或 IbbkrBroker(港美)即可,
Engine/策略层零改动。
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from .config import Section
from .models import (Account, Market, Order, OrderStatus, PendingSignal,
                     Position, Side, split_symbol)
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

    @staticmethod
    def _is_etf(symbol: str) -> bool:
        """场内ETF(沪市51x/56x/58x开头,深市15x/159开头):免印花税、免过户费。"""
        mkt, code = split_symbol(symbol)
        return str(mkt) in ("Market.SH", "SH") and \
            code[:3] in ("510", "511", "512", "513", "515", "516", "518",
                         "560", "561", "562", "563", "588") or \
            str(mkt) in ("Market.SZ", "SZ") and code[:3] in ("159",)

    # ---------- 费用 ----------
    def fee(self, market: Market, side: Side, amount: float,
            symbol: str = "") -> float:
        if market == Market.US:
            # 美股:无印花税/过户费;佣金按汇丰模型 max(交易额*费率, 每单最低)
            return max(amount * self.commission_rate, self.commission_min)
        commission = max(amount * self.commission_rate, self.commission_min)
        fee = commission
        if market == Market.HK:
            fee += amount * self.stamp_hk
        elif symbol and self._is_etf(symbol):
            pass                      # ETF:免印花税/过户费(准则:ETF交易不收)
        elif side == Side.SELL:
            fee += amount * self.stamp_a
        if market != Market.HK and not (symbol and self._is_etf(symbol)):
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
        if market not in (Market.HK, Market.US) and order.qty % self.lot != 0:
            order.status = OrderStatus.REJECTED
            order.reason += " | 拒单:A股须整手(100股)"
            self.store.save_order(order)
            return order

        if order.side == Side.BUY:
            price = ref_price * (1 + self.slippage)
            amount = price * order.qty
            fee = self.fee(market, Side.BUY, amount, order.symbol)
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
            fee = self.fee(market, Side.SELL, amount, order.symbol)
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


class ManualBroker:
    """实盘人工下单通道(live 模式)。

    汇丰没有面向个人的下单 API,所以"下单"先以人工通道落地:
    - submit(): 信号转人工订单(PENDING 登记),系统发提醒,不动账户/持仓;
    - 你在券商 App 里真实下单成交;
    - settle(): 把实际成交价/数量/费用回填,系统按真实资金流记账(账户/持仓/订单)。

    将来券商开放 API 时实现 ApiBroker(submit 直连下单,成交自动回填 settle),
    引擎/策略层零改动——这就是"先写接口"的含义。

    与 PaperBroker 的分工:PaperBroker 用于 paper 模式自动撮合(模拟盘),
    ManualBroker 用于 live 模式实盘记账,二者互不替代。
    """

    def __init__(self, store: Store) -> None:
        self.store = store

    # ---------- 登记:信号 -> 待人工执行订单(幂等) ----------
    def submit(self, sig: PendingSignal, qty: int = 0, price: float = 0.0,
               ts: datetime | None = None) -> Order:
        o = Order(symbol=sig.symbol, side=Side(sig.side), qty=max(0, int(qty)),
                  ts=ts or datetime.now(), strategy=sig.strategy,
                  reason=f"{sig.rule} | {sig.reason}")
        o.reason = f"{o.reason} | {sig.signal_id}"
        self.store.save_order(o)
        self.store.set_meta(f"manual_order:{sig.signal_id}", o.order_id)
        return o

    def find(self, sig: PendingSignal) -> Optional[Order]:
        """找回该信号已登记的人工订单(无则 None)。"""
        oid = self.store.get_meta(f"manual_order:{sig.signal_id}")
        if not oid:
            return None
        for o in self.store.recent_orders(500):
            if o.order_id == str(oid):
                return o
        return None

    # ---------- 结算:按用户回填的真实成交记账 ----------
    def settle(self, order: Order, price: float, qty: int, fee: float,
               account: Account, positions: dict[str, Position],
               t_plus_1: bool) -> Order:
        """按券商 App 的真实成交回填:现金/持仓/订单就地结算并持久化。

        信任事实:价格/数量/费用以用户所见为准(系统不猜);
        现金允许记负(先成交后补入金记录的常见时序),记 warning 提醒对账。
        """
        if price <= 0 or qty <= 0:
            order.status = OrderStatus.REJECTED
            order.reason += " | 回填失败:价格或数量非法"
            self.store.save_order(order)
            return order

        if order.side == Side.BUY:
            amount = price * qty
            account.cash -= amount + fee
            if account.cash < 0:
                log.warning("实盘回填后现金为负(%.2f):真实账户应有未记录的入金,"
                            "请用 cash 命令/看板入金对账", account.cash)
            pos = positions.get(order.symbol)
            if pos is None:
                pos = Position(symbol=order.symbol, strategy=order.strategy,
                                opened_at=datetime.now())
                positions[order.symbol] = pos
            total_cost = pos.avg_cost * pos.qty + amount + fee
            pos.qty += qty
            pos.avg_cost = total_cost / pos.qty
            pos.high_water = max(pos.high_water, price)
            if not t_plus_1:                    # 美股 T+0 随买随卖
                pos.available_qty = pos.qty
            order.status = OrderStatus.FILLED
            order.filled_price, order.filled_qty, order.fee = price, qty, fee
            self.store.save_position(pos)
        else:
            pos = positions.get(order.symbol)
            if pos is None or qty > pos.qty:
                order.status = OrderStatus.REJECTED
                order.reason += (f" | 回填失败:卖出{qty}股超过系统持仓"
                                 f"({pos.qty if pos else 0}股),先核对账户")
                self.store.save_order(order)
                return order
            amount = price * qty
            account.cash += amount - fee
            pos.qty -= qty
            pos.available_qty = max(0, pos.available_qty - qty)
            order.status = OrderStatus.FILLED
            order.filled_price, order.filled_qty, order.fee = price, qty, fee
            if pos.qty <= 0:
                self.store.drop_position(order.symbol)
                del positions[order.symbol]
            else:
                self.store.save_position(pos)

        self.store.save_account(account)
        self.store.save_order(order)
        log.info("实盘成交(人工回填) %s %s x%d @%.4g 费%.2g | %s",
                 order.side.value, order.symbol, qty, price, fee, order.reason)
        return order


class ApiBrokerError(Exception):
    pass


class ApiBroker(Broker):
    """券商 API 直连下单(live 模式,当前实现 Alpaca 美股)。

    与 PaperBroker 同签名:execute(order, ref_price, account, positions),
    引擎撮合分支零改动——live+API 时把 self.broker 换成本类即可。

    语义:
    - 下券商市价单 -> 轮询订单状态直到 filled / 拒绝 / 超时;
    - 成交价、数量、费用一律以券商回报为准(系统不猜);
    - 结算复用 ManualBroker.settle(真实资金流记账,美股 T+0);
    - 超时绝不再补单(防重复成交):订单留 PENDING,信号关闭并注明
      券商单号,由人工核实后 fill 补记或跳过。

    密钥配置(config us.broker.api,环境变量优先):
      key_id / secret,或环境变量 ALPACA_KEY_ID / ALPACA_SECRET。
    base_url 默认 paper(paper-api.alpaca.markets),实盘切
      https://api.alpaca.markets(准则:先跑通模拟盘再动真钱)。

    其他券商(富途OpenAPI/老虎/长桥/IBKR/QMT)接入:实现 _submit/_poll
    两个方法 + configured(),execute() 状态机通用。
    """

    def __init__(self, cfg: Section, store: Store) -> None:
        import requests  # 延迟导入:离线环境不影响其他 broker
        self._requests = requests
        self.store = store
        api = cfg["broker"]["api"]
        self.provider = str(api.get("provider", "alpaca"))
        self.base_url = str(api.get("base_url",
                                    "https://paper-api.alpaca.markets")).rstrip("/")
        self.key_id = os.environ.get("ALPACA_KEY_ID") or str(api.get("key_id", "") or "")
        self.secret = os.environ.get("ALPACA_SECRET") or str(api.get("secret", "") or "")
        self.poll_seconds = float(api.get("poll_seconds", 1.0))
        self.timeout_sec = float(api.get("timeout_sec", 30))
        self.zero_commission = bool(api.get("zero_commission", True))

    def configured(self) -> bool:
        return bool(self.key_id and self.secret)

    # ---------- Alpaca REST ----------
    def _headers(self) -> dict:
        return {"APCA-API-KEY-ID": self.key_id,
                "APCA-API-SECRET-KEY": self.secret}

    @staticmethod
    def _strip(symbol: str) -> str:
        return symbol[2:] if symbol[:2] == "US" else symbol

    def _submit(self, order: Order) -> str:
        r = self._requests.post(
            f"{self.base_url}/v2/orders", headers=self._headers(),
            json={"symbol": self._strip(order.symbol), "qty": str(order.qty),
                  "side": order.side.value.lower(), "type": "market",
                  "time_in_force": "day"}, timeout=10)
        if r.status_code not in (200, 201, 207):
            raise ApiBrokerError(f"HTTP{r.status_code}: {r.text[:200]}")
        return str(r.json()["id"])

    def _poll(self, oid: str) -> dict:
        r = self._requests.get(f"{self.base_url}/v2/orders/{oid}",
                               headers=self._headers(), timeout=10)
        if r.status_code != 200:
            raise ApiBrokerError(f"查询失败 HTTP{r.status_code}: {r.text[:200]}")
        return r.json()

    def account_info(self) -> dict:
        """连通性/对账:券商账户概要。"""
        r = self._requests.get(f"{self.base_url}/v2/account",
                               headers=self._headers(), timeout=10)
        if r.status_code != 200:
            raise ApiBrokerError(f"HTTP{r.status_code}: {r.text[:200]}")
        d = r.json()
        return {"status": d.get("status"), "currency": d.get("currency"),
                "cash": float(d.get("cash", 0) or 0),
                "equity": float(d.get("equity", 0) or 0),
                "buying_power": float(d.get("buying_power", 0) or 0)}

    # ---------- 执行(与 PaperBroker 同签名) ----------
    def execute(self, order: Order, ref_price: float, account: Account,
                 positions: dict[str, Position]) -> Order:
        if not self.configured():
            order.status = OrderStatus.REJECTED
            order.reason += " | 拒单:券商API密钥未配置(us.broker.api 或 ALPACA_KEY_ID/ALPACA_SECRET)"
            self.store.save_order(order)
            return order
        if order.qty <= 0:
            order.status = OrderStatus.REJECTED
            order.reason += " | 拒单:数量非法"
            self.store.save_order(order)
            return order

        try:
            oid = self._submit(order)
            log.info("券商API市价单已提交 %s %s x%d -> 券商单号 %s",
                     order.side.value, order.symbol, order.qty, oid)
        except Exception as e:  # noqa: BLE001 网络与拒单统一按拒单处理
            order.status = OrderStatus.REJECTED
            order.reason += f" | 券商拒单:{e}"
            self.store.save_order(order)
            log.warning("券商下单失败 %s: %s", order.symbol, e)
            return order

        deadline = time.time() + self.timeout_sec
        while time.time() < deadline:
            try:
                d = self._poll(oid)
            except Exception as e:  # noqa: BLE001 查询抖动:继续等
                log.warning("券商订单查询抖动 %s: %s", oid, e)
                time.sleep(self.poll_seconds)
                continue
            st = str(d.get("status", ""))
            if st == "filled":
                px = float(d.get("filled_avg_price") or 0)
                qty = int(float(d.get("filled_qty") or 0))
                fee = 0.0 if self.zero_commission else float(d.get("filled_fee") or 0)
                if px > 0 and qty > 0:
                    order.reason += f" | 券商单{oid}"
                    order = ManualBroker(self.store).settle(
                        order, px, qty, fee, account, positions, t_plus_1=False)
                    return order
                break                     # 成交但价格异常,按超时路径处理
            if st in ("canceled", "expired", "rejected", "pending_cancel"):
                order.status = OrderStatus.REJECTED
                order.reason += (f" | 券商状态{st}:"
                                 f"{d.get('cancel_reason') or d.get('reject_reason') or ''}")
                self.store.save_order(order)
                return order
            time.sleep(self.poll_seconds)

        # 超时:券商单已提交,绝不自动补单(防重复成交)——留人工核实
        order.status = OrderStatus.PENDING
        order.reason += (f" | 券商单{oid}超时{self.timeout_sec}s未回报:"
                         f"到券商App核实;已成交则 fill 补记,未成交则不管")
        self.store.save_order(order)
        log.warning("券商单超时未回报 %s(%s),保留人工核实", oid, order.symbol)
        return order
