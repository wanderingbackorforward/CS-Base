"""实盘人工通道单测(全部离线,不碰网络):ManualBroker 登记结算 / fill / skip / cash。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qtrader import fill as F                               # noqa: E402
from qtrader.broker import ManualBroker                     # noqa: E402
from qtrader.models import (Account, OrderStatus,           # noqa: E402
                            PendingSignal, Side)


class FakeStrategy:
    def __init__(self, name: str) -> None:
        self.name = name
        self.fills = []          # (signal, qty, ok)

    def on_fill(self, sig, qty: int, ok: bool) -> None:
        self.fills.append((sig.signal_id, qty, ok))


class FakeEngine:
    """duck-typing 引擎:满足 fill 模块所需的最小接口。"""

    def __init__(self, tmp: Path, capital: float = 10000.0,
                 t_plus_1: bool = True) -> None:
        from qtrader.store import Store
        self.store = Store(tmp)
        self.account = Account(cash=capital, day_start_equity=capital,
                               peak_equity=capital)
        self.capital = capital
        self.strategies: list = []
        self.t_plus_1 = t_plus_1

    def _market_value(self, positions, quote=None):
        return 0.0


def make_signal(symbol="USQQQM", side=Side.BUY, rule="C4", date="2026-10-20",
                strategy="us_core", **kw) -> PendingSignal:
    return PendingSignal(symbol=symbol, side=side, signal_date=date,
                         strategy=strategy, rule=rule,
                         reason="底仓建立", **kw)


@pytest.fixture()
def us_engine(tmp_path):
    e = FakeEngine(tmp_path / "us", capital=10000.0, t_plus_1=False)
    st = FakeStrategy("us_core")
    e.strategies.append(st)
    return e


@pytest.fixture()
def cn_engine(tmp_path):
    e = FakeEngine(tmp_path / "cn", capital=10000.0, t_plus_1=True)
    st = FakeStrategy("value_dip")
    e.strategies.append(st)
    return e


# ---------------- ManualBroker:登记 + 结算 ----------------
def test_manual_broker_submit_and_find(us_engine):
    sig = make_signal()
    us_engine.store.save_signal(sig)
    b = ManualBroker(us_engine.store)
    o = b.submit(sig, qty=9, price=296.0)
    assert o.status == OrderStatus.PENDING
    found = b.find(sig)
    assert found is not None and found.order_id == o.order_id


def test_settle_buy_updates_account_position(us_engine):
    sig = make_signal()
    us_engine.store.save_signal(sig)
    b = ManualBroker(us_engine.store)
    o = b.submit(sig, qty=9, price=296.0)
    o = b.settle(o, price=296.0, qty=9, fee=18.0,
                 account=us_engine.account,
                 positions=us_engine.store.all_positions(),
                 t_plus_1=False)
    assert o.status == OrderStatus.FILLED
    assert us_engine.account.cash == pytest.approx(10000 - 296 * 9 - 18)
    pos = us_engine.store.all_positions()["USQQQM"]
    assert pos.qty == 9
    assert pos.avg_cost == pytest.approx((296 * 9 + 18) / 9)
    assert pos.available_qty == 9          # 美股 T+0 立即可卖


def test_settle_buy_cn_t_plus_1(cn_engine):
    sig = make_signal(symbol="SH600036", rule="G3-A", strategy="value_dip")
    cn_engine.store.save_signal(sig)
    b = ManualBroker(cn_engine.store)
    o = b.submit(sig, qty=100, price=10.0)
    o = b.settle(o, price=10.0, qty=100, fee=5.0,
                 account=cn_engine.account,
                 positions=cn_engine.store.all_positions(),
                 t_plus_1=True)
    pos = cn_engine.store.all_positions()["SH600036"]
    assert pos.qty == 100
    assert pos.available_qty == 0          # A股 T+1 当日不可卖


def test_settle_sell_rejects_over_position(us_engine):
    sig = make_signal(side=Side.SELL, rule="C8")
    us_engine.store.save_signal(sig)
    b = ManualBroker(us_engine.store)
    o = b.submit(sig, qty=5, price=300.0)
    o = b.settle(o, price=300.0, qty=5, fee=0.0,
                 account=us_engine.account,
                 positions=us_engine.store.all_positions(),
                 t_plus_1=False)
    assert o.status == OrderStatus.REJECTED  # 系统无持仓,拒绝回填


# ---------------- fill_signal 编排 ----------------
def test_fill_signal_executes_and_advances_state(us_engine):
    sig = make_signal()
    us_engine.store.save_signal(sig)
    r = F.fill_signal(us_engine, sig.signal_id, price=296.0, qty=9,
                      fee=18.0, mood=3)
    assert r["ok"], r["msg"]
    # 信号 EXECUTED + 回填信息
    [s] = us_engine.store.pending_signals(statuses=("EXECUTED",))
    assert s.filled_price == 296.0 and s.filled_qty == 9
    assert "情绪3/5" in s.note
    # 策略状态机被推进(ok=True)
    st = us_engine.strategies[0]
    assert st.fills and st.fills[0] == (sig.signal_id, 9, True)
    # 订单 FILLED 带规则编号(D2)
    o = us_engine.store.recent_orders(5)[0]
    assert o.status == OrderStatus.FILLED and "C4" in o.reason


def test_fill_signal_prefix_match(us_engine):
    us_engine.store.save_signal(make_signal())
    r = F.fill_signal(us_engine, "S20261020", price=296.0, qty=9)
    assert r["ok"]


def test_fill_signal_bad_args(us_engine):
    us_engine.store.save_signal(make_signal())
    assert not F.fill_signal(us_engine, "S20261020", price=0, qty=9)["ok"]
    assert not F.fill_signal(us_engine, "not-exist", price=1, qty=1)["ok"]


def test_fill_signal_sell(us_engine):
    # 先买入建仓,再卖出回填
    sig_buy = make_signal()
    us_engine.store.save_signal(sig_buy)
    F.fill_signal(us_engine, sig_buy.signal_id, price=296.0, qty=9, fee=18.0)
    sig_sell = make_signal(side=Side.SELL, rule="C8", qty=9, sell_ratio=1.0)
    us_engine.store.save_signal(sig_sell)
    cash_before = us_engine.account.cash
    r = F.fill_signal(us_engine, sig_sell.signal_id, price=310.0, qty=9, fee=18.0)
    assert r["ok"]
    assert us_engine.account.cash == pytest.approx(cash_before + 310 * 9 - 18)
    assert "USQQQM" not in us_engine.store.all_positions()


# ---------------- skip / cash ----------------
def test_skip_signal_releases_tranche(us_engine):
    sig = make_signal(rule="C5-A")
    us_engine.store.save_signal(sig)
    ManualBroker(us_engine.store).submit(sig, qty=5, price=290.0)
    r = F.skip_signal(us_engine, sig.signal_id, "不想要了")
    assert r["ok"]
    [s] = us_engine.store.pending_signals(statuses=("CANCELLED",))
    assert s.note == "不想要了"
    st = us_engine.strategies[0]
    assert st.fills and st.fills[0] == (sig.signal_id, 0, False)  # 释放档位


def test_cash_adjust_delta_and_set(us_engine):
    r = F.cash_adjust(us_engine, delta=5000, note="入金")
    assert r["ok"]
    assert us_engine.account.cash == 15000
    assert us_engine.capital == 15000      # 本金基准同步,盈亏口径不失真
    r = F.cash_adjust(us_engine, delta=-2000, note="出金")
    assert r["ok"] and us_engine.account.cash == 13000
    r = F.cash_adjust(us_engine, set_to=9999, note="对账校准")
    assert r["ok"] and us_engine.account.cash == 9999
    assert not F.cash_adjust(us_engine, delta=-99999)["ok"]  # 出金超现金拒绝


def test_resolve_engine_finds_right_market(tmp_path):
    cn = FakeEngine(tmp_path / "a", t_plus_1=True)
    us = FakeEngine(tmp_path / "b", t_plus_1=False)
    cn.store.save_signal(make_signal(symbol="SH600036", rule="G3-A",
                                      strategy="value_dip"))
    us.store.save_signal(make_signal(rule="C5-B"))
    assert F.resolve_engine([cn, us], "S20261020-SH600036") is cn
    assert F.resolve_engine([cn, us], "S20261020-USQQQM") is us
    assert F.resolve_engine([cn, us], "zzz") is None
