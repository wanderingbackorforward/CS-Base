"""ApiBroker(券商API直连)状态机单测:mock _submit/_poll,不打真网络。

覆盖:成交结算 / 券商拒单 / 回报超时(绝不补单) / 未配置密钥。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qtrader.config import Section                      # noqa: E402
from qtrader.models import Account, Order, OrderStatus, Side   # noqa: E402
from qtrader.broker import ApiBroker                    # noqa: E402
from qtrader.store import Store                         # noqa: E402


def make_broker(tmp_path: Path, **api_over) -> tuple[ApiBroker, Store, Account]:
    store = Store(tmp_path)
    api = {"key_id": "test-key", "secret": "test-secret",
           "timeout_sec": 0.3, "poll_seconds": 0.01,
           "base_url": "https://paper-api.alpaca.markets"}
    api.update(api_over)
    cfg = Section({"broker": {"api": api}})
    acc = Account(cash=10000.0, day_start_equity=10000.0, peak_equity=10000.0)
    return ApiBroker(cfg, store), store, acc


def make_order(symbol="USQQQM", side=Side.BUY, qty=9) -> Order:
    from datetime import datetime
    return Order(symbol=symbol, side=side, qty=qty, ts=datetime.now(),
                 strategy="us_core", reason="C4 | 底仓建立")


def test_filled_settles_at_broker_price(tmp_path):
    b, store, acc = make_broker(tmp_path)
    b._submit = lambda order: "OID-1"
    b._poll = lambda oid: {"status": "filled",
                           "filled_avg_price": "296.20", "filled_qty": "9"}
    o = b.execute(make_order(), 295.0, acc, store.all_positions())
    assert o.status == OrderStatus.FILLED
    assert o.filled_price == pytest.approx(296.20)      # 券商回报价,非参考价
    assert acc.cash == pytest.approx(10000 - 296.20 * 9)   # 零佣金
    assert store.all_positions()["USQQQM"].qty == 9
    assert "OID-1" in o.reason


def test_rejected_by_broker(tmp_path):
    b, store, acc = make_broker(tmp_path)
    b._submit = lambda order: "OID-2"
    b._poll = lambda oid: {"status": "rejected", "reject_reason": "insufficient buying power"}
    o = b.execute(make_order(), 295.0, acc, store.all_positions())
    assert o.status == OrderStatus.REJECTED
    assert "券商状态rejected" in o.reason
    assert acc.cash == 10000.0                          # 拒单不动账


def test_submit_network_error_rejected(tmp_path):
    b, store, acc = make_broker(tmp_path)

    def _boom(order):
        raise RuntimeError("connection refused")
    b._submit = _boom
    o = b.execute(make_order(), 295.0, acc, store.all_positions())
    assert o.status == OrderStatus.REJECTED
    assert "券商拒单" in o.reason


def test_timeout_never_refills(tmp_path):
    """回报超时:订单留 PENDING(券商单已提交),绝不再自动补单。"""
    b, store, acc = make_broker(tmp_path)
    b._submit = lambda order: "OID-3"
    b._poll = lambda oid: {"status": "accepted"}       # 一直不成交也不拒
    t0 = time.time()
    o = b.execute(make_order(), 295.0, acc, store.all_positions())
    assert time.time() - t0 < 3                          # 0.3s 超时及时返回
    assert o.status == OrderStatus.PENDING
    assert "OID-3" in o.reason and "超时" in o.reason
    assert acc.cash == 10000.0                          # 未结算
    assert store.all_positions() == {}                  # 未建仓


def test_not_configured_rejected(tmp_path):
    b, store, acc = make_broker(tmp_path, key_id="", secret="")
    assert not b.configured()
    o = b.execute(make_order(), 295.0, acc, store.all_positions())
    assert o.status == OrderStatus.REJECTED
    assert "未配置" in o.reason


def test_poll_jitter_tolerated(tmp_path):
    """查询抖动一次后成交:仍能正常结算。"""
    b, store, acc = make_broker(tmp_path)
    b._submit = lambda order: "OID-4"
    calls = {"n": 0}

    def flaky_poll(oid):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("socket timeout")
        return {"status": "filled", "filled_avg_price": "100.5", "filled_qty": "10"}
    b._poll = flaky_poll
    o = b.execute(make_order(qty=10), 100.0, acc, store.all_positions())
    assert o.status == OrderStatus.FILLED
    assert calls["n"] == 2


def test_sell_settles_cash(tmp_path):
    b, store, acc = make_broker(tmp_path)
    # 先买入建仓(走人工通道结算)
    from qtrader.broker import ManualBroker
    sig_like = make_order(side=Side.BUY, qty=9)
    ManualBroker(store).settle(sig_like, 296.0, 9, 18.0, acc,
                               store.all_positions(), t_plus_1=False)
    b._submit = lambda order: "OID-5"
    b._poll = lambda oid: {"status": "filled",
                           "filled_avg_price": "310.00", "filled_qty": "9"}
    o = b.execute(make_order(side=Side.SELL, qty=9), 308.0, acc,
                  store.all_positions())
    assert o.status == OrderStatus.FILLED
    assert acc.cash == pytest.approx(10000 - 296 * 9 - 18 + 310 * 9)
    assert "USQQQM" not in store.all_positions()
