"""实盘人工通道编排层(live 模式)。

流程:系统算出信号 -> submit_manual() 登记待执行订单并发提醒 ->
你在券商 App(汇丰等)真实下单 -> fill_signal() 回填实际成交价/数量/费用
-> 系统按真实资金流记账(账户/持仓/订单/交易日志 D2)并推进档位状态机。

账户资金自己补:cash_adjust() 入金/出金/绝对校准,让系统现金与真实账户对齐。

引擎鸭子类型:Engine(A股)或 UsEngine(美股)均可,需要
store / account / strategies / capital / _market_value 属性。
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from .broker import ManualBroker
from .models import Order, OrderStatus, PendingSignal, Side

log = logging.getLogger("qtrader.fill")


# ---------------- 引擎适配 ----------------
def _is_us_engine(engine) -> bool:
    from .us_engine import UsEngine
    return isinstance(engine, UsEngine)


def find_signal(store, signal_id: str) -> PendingSignal | None:
    """精确匹配或唯一前缀匹配待执行信号。"""
    pend = store.pending_signals()
    for s in pend:
        if s.signal_id == signal_id:
            return s
    prefix_hits = [s for s in pend if s.signal_id.startswith(signal_id)]
    return prefix_hits[0] if len(prefix_hits) == 1 else None


def resolve_engine(engines: list, signal_id: str):
    """双引擎里找到持有该信号的引擎(找不到返回 None)。"""
    for eng in engines:
        if eng is None:
            continue
        if find_signal(eng.store, signal_id) is not None:
            return eng
    return None


# ---------------- live 模式:信号 -> 人工订单 ----------------
def submit_manual(engine, signals: list[PendingSignal], positions: dict,
                  quote: pd.DataFrame | None, ts: datetime | None = None) -> int:
    """live 模式的"下单"动作:信号转人工订单(幂等)并打提醒日志。

    不撮合、不动账户——真正的一击在你券商 App 里完成。
    返回本次新登记的订单数。
    """
    broker = ManualBroker(engine.store)
    qrow: dict = {}
    if quote is not None and len(quote):
        qrow = {str(r["symbol"]): r for _, r in quote.iterrows()}
    try:
        mv = engine._market_value(positions, quote=quote)
    except Exception:  # noqa: BLE001 估值失败不影响登记
        mv = 0.0
    equity = engine.account.equity(mv)

    new_count = 0
    for sig in signals:
        if engine.store.get_meta(f"manual_order:{sig.signal_id}"):
            continue                      # 已登记过,幂等
        row = qrow.get(sig.symbol)
        px = float(row["price"]) if row is not None and row.get("price") else 0.0
        if sig.side == Side.BUY:
            budget = sig.size_hint_pct * equity
            qty = int(budget // (px * 1.002)) if px > 0 else 0
        else:
            pos = positions.get(sig.symbol)
            hold = pos.qty if pos else 0
            qty = sig.qty if sig.qty > 0 else int(hold * sig.sell_ratio)
        est = px * qty if px > 0 else 0.0
        o = broker.submit(sig, qty=qty, price=px, ts=ts)
        new_count += 1
        side_txt = sig.side.value if hasattr(sig.side, "value") else str(sig.side)
        log.info("待人工下单 [%s] %s %s 建议%d股 预估%.0f | 券商App成交后回填: "
                 "python main.py fill %s --price <成交价> --qty <股数> --fee <费用>",
                 sig.rule, side_txt, sig.symbol, qty, est, sig.signal_id)
    if signals:
        log.info("live 实盘模式:%d 条信号待你在券商App下单,成交后回填"
                 "(看板信号卡片「回填成交」按钮,或 fill 命令)", len(signals))
    return new_count


# ---------------- 回填:真实成交入账 ----------------
def fill_signal(engine, signal_id: str, price: float, qty: int,
                fee: float = 0.0, mood: int = 0,
                t_plus_1: bool | None = None) -> dict:
    """成交回填:你在券商 App 里的真实成交,按事实记账并推进状态机。

    mood:下单时情绪 1-5(D2 纪律字段,可选)。
    返回 {"ok": bool, "msg": str}。
    """
    sig = find_signal(engine.store, signal_id)
    if sig is None:
        return {"ok": False, "msg": f"找不到待执行信号: {signal_id}"}
    if price is None or price <= 0 or not qty or qty <= 0:
        return {"ok": False, "msg": "成交价与数量必须为正(以券商App成交回报为准)"}

    if t_plus_1 is None:
        t_plus_1 = not _is_us_engine(engine)
    broker = ManualBroker(engine.store)
    order = broker.find(sig) or broker.submit(sig, qty=qty, price=price)
    positions = engine.store.all_positions()
    order = broker.settle(order, float(price), int(qty), float(fee),
                          engine.account, positions, t_plus_1=t_plus_1)
    if order.status != OrderStatus.FILLED:
        return {"ok": False, "msg": order.reason}

    sig.status = "EXECUTED"
    sig.executed_date = datetime.now().strftime("%Y-%m-%d")
    sig.filled_price, sig.filled_qty = float(price), int(qty)
    note = f"实盘回填 @{price:.4g} x{qty} 费{fee:.2f}"
    if mood:
        note += f" 情绪{mood}/5"
    sig.note = note
    engine.store.save_signal(sig)

    st = next((s for s in engine.strategies if s.name == sig.strategy), None)
    if st is not None:
        st.on_fill(sig, int(qty), True)

    engine.store.export_snapshot()
    cur = "USD" if not t_plus_1 else "CNY"
    side_txt = sig.side.value if hasattr(sig.side, "value") else str(sig.side)
    log.info("D2 交易日志 [%s] %s %s x%d @%.4g 费%.2g %s|%s",
             sig.rule, side_txt, sig.symbol, qty, price, fee, cur, note)
    return {"ok": True,
            "msg": f"已入账:{sig.rule} {side_txt} {sig.symbol} "
                   f"x{qty} @ {price:.4g} 费 {fee:.2g} {cur}",
            "order_id": order.order_id}


# ---------------- 放弃:不下单了 ----------------
def skip_signal(engine, signal_id: str, note: str = "") -> dict:
    """放弃信号:订单作废、档位释放,等下一次触发条件重新评估。"""
    sig = find_signal(engine.store, signal_id)
    if sig is None:
        return {"ok": False, "msg": f"找不到待执行信号: {signal_id}"}
    broker = ManualBroker(engine.store)
    order = broker.find(sig)
    if order is not None and order.status == OrderStatus.PENDING:
        order.status = OrderStatus.REJECTED
        order.reason += " | 人工放弃(未在券商App下单)"
        engine.store.save_order(order)
    sig.status = "CANCELLED"
    sig.note = note or "人工放弃,档位已释放,待条件重新触发"
    engine.store.save_signal(sig)
    st = next((s for s in engine.strategies if s.name == sig.strategy), None)
    if st is not None:
        st.on_fill(sig, 0, False)
    log.info("人工放弃信号 %s %s(%s)", sig.rule, sig.symbol, sig.signal_id)
    return {"ok": True, "msg": f"已放弃:{sig.rule} {sig.symbol}(档位已释放)"}


# ---------------- 账户资金自己补 ----------------
def cash_adjust(engine, delta: float = 0.0, set_to: float | None = None,
                note: str = "") -> dict:
    """入金/出金/现金校准:让系统现金与真实账户对齐。

    - delta > 0 入金 / delta < 0 出金(出金不得超过系统现金);
    - set_to 直接校准到绝对值(以券商App显示的可用资金为准);
    - 本金基准(engine.capital)随 delta 同步,保证"总盈亏=净值-本金"口径不虚增。
    """
    acc = engine.account
    old = acc.cash
    if set_to is not None:
        if set_to < 0:
            return {"ok": False, "msg": "校准目标不能为负"}
        delta = set_to - acc.cash
    if abs(delta) < 1e-9:
        return {"ok": True, "msg": f"现金无变化({old:.0f})"}
    if delta < 0 and acc.cash + delta < -1e-6:
        return {"ok": False,
                "msg": f"出金 {abs(delta):.0f} 超过系统现金 {acc.cash:.0f}"}
    acc.cash += delta
    engine.capital += delta            # 成本基准同步,盈亏口径不失真
    engine.store.save_account(acc)
    engine.store.export_snapshot()
    action = "入金" if delta > 0 else "出金"
    log.info("账户%s %.2f(现金 %.2f -> %.2f)| %s",
             action, abs(delta), old, acc.cash, note or "无备注")
    return {"ok": True,
            "msg": f"已{action} {abs(delta):.2g}:现金 {old:.0f} -> {acc.cash:.0f}"
                   f"(本金基准同步调整为 {engine.capital:.0f})"}
