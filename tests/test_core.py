"""核心逻辑单测(全部离线合成数据,不碰网络):费用/T+1/风控/策略信号/持久化。"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qtrader.broker import PaperBroker          # noqa: E402
from qtrader.config import Section              # noqa: E402
from qtrader.indicators import drawdown_from_high, rsi, sma  # noqa: E402
from qtrader.models import (Account, Order, Position, Side, OrderStatus)  # noqa: E402
from qtrader.risk import RiskManager            # noqa: E402
from qtrader.store import Store                 # noqa: E402
from qtrader.strategies.base import Context    # noqa: E402
from qtrader.strategies.cautious_dip import CautiousDip   # noqa: E402
from qtrader.strategies.hot_momentum import HotMomentum    # noqa: E402


def make_cfg() -> Section:
    return Section({
        "app": {"initial_capital": 10000, "state_dir": "./data"},
        "market": {"a_share": {"exclude_st": True, "exclude_star": True,
                               "exclude_bj": True, "include_chinext": False,
                               "min_price": 3, "max_price": 150,
                               "min_amount": 1e8}},
        "universe": {"dip_targets": [{"symbol": "SH600519", "name": "M"}]},
        "heat": {"top_n": 6, "watch_keywords": ["AI"], "show_top_stocks": 8},
        "strategy": {
            "hot_momentum": {"enabled": True, "breakout_lookback": 20,
                             "vol_ratio_min": 1.8, "day_change_min": 0.02,
                             "day_change_max": 0.07, "close_range_pos_min": 0.8,
                             "take_profit": 0.08, "pump_exit_pct": 0.07,
                             "trailing_arm": 0.05, "hard_stop": 0.06,
                             "time_stop_days": 4, "max_positions": 3},
            "cautious_dip": {"enabled": True, "lookback_high": 60,
                             "drawdown_trigger": 0.12, "rsi_window": 14,
                             "rsi_oversold": 32, "tranche_count": 3,
                             "tranche_step": 0.05, "target_profit": 0.10,
                             "time_stop_days": 15, "hard_stop": 0.10,
                             "max_positions": 3},
        },
        "risk": {"max_position_pct": 0.30, "max_total_exposure": 0.85,
                 "regime_ma": 20, "regime_low_exposure": 0.40,
                 "daily_loss_limit_pct": -0.03, "cooldown_days": 2,
                 "max_drawdown_kill": -0.15},
        "broker": {"paper": {"commission_rate": 0.00025, "commission_min": 5.0,
                             "stamp_tax": 0.0005, "transfer_fee": 0.00001,
                             "slippage": 0.001}},
    })


def make_hist(closes: list[float], volumes: list[float] | None = None,
              dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    closes = pd.Series(closes, dtype=float)
    n = len(closes)
    dates = dates or pd.bdate_range(end=datetime.now().date(), periods=n)
    vols = volumes if volumes is not None else [1e6] * n
    df = pd.DataFrame({
        "dt": dates[:n],
        "open": closes.shift(1).fillna(closes.iloc[0]),
        "close": closes,
        "volume": vols,
        "amount": closes * pd.Series(vols) * 100,
    })
    # 振幅收窄:收盘贴近全天高点(尾盘强势语义)
    df["high"] = df[["open", "close"]].max(axis=1) * 1.003
    df["low"] = df[["open", "close"]].min(axis=1) * 0.997
    return df


# ---------------- 指标 ----------------
def test_indicators() -> None:
    s = pd.Series([float(i) for i in range(1, 31)])
    assert sma(s, 5).iloc[-1] == pytest.approx(28.0)
    assert rsi(pd.Series([10.0] * 5 + [20.0])).iloc[-1] > 80  # 连涨应超买
    assert rsi(pd.Series([20.0] * 5 + [10.0])).iloc[-1] < 20  # 连跌应超卖
    dd = drawdown_from_high(pd.Series([100, 90, 80.0]), 3)
    assert dd.iloc[-1] == pytest.approx(0.2)


# ---------------- 纸面撮合:费用与 T+1 ----------------
def test_paper_broker_fees_and_t1(tmp_path) -> None:
    cfg = make_cfg()
    store = Store(tmp_path)
    broker = PaperBroker(cfg, store)
    acc = Account(cash=10000, day_start_equity=10000, peak_equity=10000)
    positions: dict = {}
    o = Order(symbol="SH600519", side=Side.BUY, qty=100, ts=datetime.now())
    o = broker.execute(o, 10.0, acc, positions)
    assert o.status == OrderStatus.FILLED
    # 金额 1000 -> 佣金=max(0.25,5)=5 + 过户 0.01 + 滑点抬价
    assert o.fee == pytest.approx(5 + 100 * 10.0 * 1.001 * 0.00001, rel=1e-3)
    assert o.filled_price == pytest.approx(10.01)
    assert acc.cash < 10000 - 100 * 10.0  # 现金已扣
    # T+1:当日买入不可卖
    sell = Order(symbol="SH600519", side=Side.SELL, qty=100, ts=datetime.now())
    sell = broker.execute(sell, 10.5, acc, positions)
    assert sell.status == OrderStatus.REJECTED
    # 次日解锁后可卖,卖出收印花
    broker.promote_new_day(positions)
    sell2 = Order(symbol="SH600519", side=Side.SELL, qty=100, ts=datetime.now())
    sell2 = broker.execute(sell2, 10.5, acc, positions)
    assert sell2.status == OrderStatus.FILLED
    assert sell2.fee > 5  # 佣金5 + 印花~0.53 + 过户
    assert "SH600519" not in positions


def test_paper_broker_lot_and_cash(tmp_path) -> None:
    cfg = make_cfg()
    store = Store(tmp_path)
    broker = PaperBroker(cfg, store)
    acc = Account(cash=10000, day_start_equity=10000, peak_equity=10000)
    positions: dict = {}
    bad = Order(symbol="SZ000001", side=Side.BUY, qty=150, ts=datetime.now())
    assert broker.execute(bad, 10, acc, positions).status == OrderStatus.REJECTED
    too_rich = Order(symbol="SZ000001", side=Side.BUY, qty=10000, ts=datetime.now())
    assert broker.execute(too_rich, 10, acc, positions).status == OrderStatus.REJECTED


# ---------------- 风控 ----------------
def test_risk_kill_switch(tmp_path) -> None:
    cfg = make_cfg()
    store = Store(tmp_path)
    risk = RiskManager(cfg, store)
    acc = Account(cash=8400, day_start_equity=10000, peak_equity=10000)
    notes = risk.settle_eod(acc, 0.0)  # equity 8400,回撤 16%
    assert acc.killed and notes
    assert not risk.can_open(acc, {}, 0)[0]


# ---------------- 策略:追热点入场 ----------------
def test_hot_momentum_entry() -> None:
    cfg = make_cfg()
    st = HotMomentum(cfg)
    # 60 日横盘 10 元,今日放量 +5% 突破
    closes = [10.0] * 59 + [10.55]
    hist = make_hist(closes, volumes=[1e6] * 59 + [5e6])
    spot = pd.DataFrame([{"symbol": "SH600000", "code": "600000", "name": "浦发",
                         "price": 10.55, "pct_chg": 0.05, "amount": 3e8}])
    ctx = Context(today="2026-10-20", spot=spot, quote=spot,
                  hist={"SH600000": hist}, positions={}, candidates=["SH600000"])
    orders = st.entry_signals(ctx)
    assert len(orders) == 1 and orders[0].side == Side.BUY
    assert "突破" in orders[0].reason
    # 涨停级(>7%)不追
    spot2 = spot.copy()
    spot2["pct_chg"] = 0.095
    ctx2 = Context(today="2026-10-20", spot=spot2, quote=spot2,
                   hist={"SH600000": hist}, positions={})
    assert st.entry_signals(ctx2) == []


def test_hot_momentum_pump_exit() -> None:
    cfg = make_cfg()
    st = HotMomentum(cfg)
    pos = Position(symbol="SH600000", qty=100, available_qty=100,
                   avg_cost=10.0, strategy="hot_momentum", days_held=2)
    # hist[-1]=今日(进行时现价),[-2]=昨日收 10.75,[-3]=9.95 -> 昨日 +8% 触发死线
    closes = [9.0, 9.2, 9.5, 9.95, 10.75, 10.8]
    hist = make_hist(closes)
    spot = pd.DataFrame([{"symbol": "SH600000", "price": 10.8, "pct_chg": 0.005}])
    ctx = Context(today="2026-10-20", spot=spot, quote=spot,
                  hist={"SH600000": hist}, positions={"SH600000": pos})
    sells = st.manage(ctx)
    assert len(sells) == 1 and "死线" in sells[0].reason


# ---------------- 策略:谨慎抄底 ----------------
def test_cautious_dip_entry_and_stop() -> None:
    cfg = make_cfg()
    st = CautiousDip(cfg)
    # 从 100 一路阴跌到 85(-15%),最近一日缩量收阳回升(需 >=60 根)
    closes = [100 - i * 0.28 for i in range(59)] + [84.5, 83.4, 85.2]
    hist = make_hist(closes)
    hist.iloc[-1, hist.columns.get_loc("open")] = 83.8  # 今收阳
    spot = pd.DataFrame([{"symbol": "SH600519", "price": 85.2, "pct_chg": 0.02}])
    ctx = Context(today="2026-10-20", spot=spot, quote=spot,
                  hist={"SH600519": hist}, positions={})
    orders = st.entry_signals(ctx)
    assert len(orders) == 1
    assert orders[0].size_hint_pct > 0  # 分批:首批只占 1/3
    # 持仓 -11% 应认错离场(成本 96 vs 现价 85.2)
    pos = Position(symbol="SH600519", qty=100, available_qty=100, avg_cost=96.0,
                   strategy="cautious_dip", days_held=3,
                   meta={"tranche": 3, "last_price": 80.0})
    ctx2 = Context(today="2026-10-20", spot=spot, quote=spot,
                   hist={"SH600519": hist}, positions={"SH600519": pos})
    sells = st.manage(ctx2)
    assert len(sells) == 1 and "认错" in sells[0].reason


# ---------------- 持久化 ----------------
def test_store_roundtrip(tmp_path) -> None:
    store = Store(tmp_path)
    acc = Account(cash=1234.5, day_start_equity=1000, peak_equity=1000)
    store.save_account(acc)
    assert store.load_account().cash == 1234.5
    p = Position(symbol="SH600519", qty=200, available_qty=100, avg_cost=10.5,
                 strategy="hot_momentum")
    store.save_position(p)
    assert store.all_positions()["SH600519"].qty == 200
    o = Order(symbol="SH600519", side=Side.BUY, qty=200, ts=datetime.now(),
              status=OrderStatus.FILLED, filled_price=10.5, filled_qty=200, fee=5.0)
    store.save_order(o)
    assert store.recent_orders(5)[0].filled_price == 10.5
    store.append_equity(datetime.now(), 1000, 500, 500)
    assert len(store.equity_curve()) == 1
