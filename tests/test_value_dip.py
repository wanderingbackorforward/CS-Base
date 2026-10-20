"""value_dip 档位状态机单元测试(纯本地,不依赖网络)。

覆盖:G3 触发与单日单档、G4 估值闸门(含不可知)、G5 档位占用、
G6 涨回前高卖2/3、G7 移动止盈、G8 高估值清仓、G10 风险事件、G12 深跌复核、
信号持久化 round-trip。

运行:python -m unittest tests.test_value_dip -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrader.config import load_config
from qtrader.models import Account, PendingSignal, Position, Side
from qtrader.store import Store
from qtrader.strategies.base import Context
from qtrader.strategies.value_dip import ValueDip

SYM = "SH600000"
CFG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class FakePM:
    """替身 PoolManager:估值分位可控。"""

    def __init__(self, q=None):
        self.q = q

    def val_quantile(self, symbol, industry=""):
        return self.q


def make_hist(prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": prices})


def make_ctx(hist_prices, today="2026-10-20", q_pm=None, positions=None,
             pending=None, name="浦发银行", cash=10000.0, mv=0.0,
             month_end=False, rebalance_day=False):
    hist = {SYM: make_hist(hist_prices)} if hist_prices else {}
    pos = {SYM: Position(symbol=SYM, qty=300, available_qty=300, avg_cost=8.0,
                         strategy="value_dip")} if positions else {}
    return Context(
        today=today,
        spot=pd.DataFrame(),
        quote=pd.DataFrame(),
        hist=hist,
        account=Account(cash=cash, day_start_equity=cash, peak_equity=cash),
        positions=pos or {},
        market_value=mv,
        pool=[{"symbol": SYM, "industry": "银行", "name": name}],
        pending=pending or [],
        pool_val_q={},
        blackout=set(),
        name_map={SYM: name},
        industry_map={SYM: "银行"},
        month_end=month_end,
        rebalance_day=rebalance_day,
    ), q_pm


class ValueDipTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(self.tmp.name)
        self.cfg = load_config(str(CFG_PATH))

    def tearDown(self):
        self.tmp.cleanup()

    def _strategy(self, q=None):
        return ValueDip(self.cfg, FakePM(q), self.store)

    # -------- G3:回撤档位触发 --------
    def test_g3_buy_signal_and_tranche_occupation(self):
        """-21%回撤 + 估值分位30% -> G3-A 买1份;档位变 pending;
        再跑一遍不重复触发(G5 档位占用);单日单档不触发B。"""
        st = self._strategy(q=0.30)
        flat = [10.0] * 260
        dip = flat + [7.9]                    # 前高10,收7.9 -> -21%
        ctx, _ = make_ctx(dip)
        sigs = st.entry_signals(ctx)
        self.assertEqual(len(sigs), 1)
        sig = sigs[0]
        self.assertEqual(sig.rule, "G3-A")
        self.assertEqual(sig.tranche, "A")
        self.assertAlmostEqual(sig.trigger_high, 10.0)
        self.assertEqual(sig.size_hint_pct, 0.04)
        stA = self.store.pool_state(SYM)["tranches"]["A"]
        self.assertEqual(stA["state"], "pending")
        # 同日重跑:pending 档位不可再触发
        ctx2, _ = make_ctx(dip)
        self.assertEqual(st.entry_signals(ctx2), [])
        # 次日更浅价不再触发A(占用中),-21%未到B线 -> 无信号
        ctx3, _ = make_ctx(dip)
        self.assertEqual(st.entry_signals(ctx3), [])

    def test_g3_b_fires_after_deeper_drop(self):
        """-36% -> B档触发(A、B同跌但 one_per_day 只取最浅可用档)。"""
        st = self._strategy(q=0.30)
        dip = [10.0] * 260 + [6.4]            # -36%
        ctx, _ = make_ctx(dip)
        sigs = st.entry_signals(ctx)
        self.assertEqual([s.tranche for s in sigs], ["A"])  # 单日单档取最浅
        ctx2, _ = make_ctx(dip)
        sigs2 = st.entry_signals(ctx2)
        self.assertEqual([s.tranche for s in sigs2], ["B"])

    # -------- G4:估值闸门 --------
    def test_g4_valuation_gate(self):
        st = self._strategy(q=0.55)
        dip = [10.0] * 260 + [7.9]
        ctx, _ = make_ctx(dip)
        self.assertEqual(st.entry_signals(ctx), [])   # 分位>50% 不买
        st2 = self._strategy(q=None)
        ctx2, _ = make_ctx(dip)
        self.assertEqual(st2.entry_signals(ctx2), [])  # 估值不可知 不买

    # -------- G5:财报黑盒 --------
    def test_g5_earnings_blackout(self):
        st = self._strategy(q=0.30)
        dip = [10.0] * 260 + [7.9]
        ctx, _ = make_ctx(dip)
        ctx.blackout = {SYM}
        self.assertEqual(st.entry_signals(ctx), [])

    # -------- G6:涨回前高卖2/3 --------
    def test_g6_sell_two_thirds_at_prev_high(self):
        st = self._strategy(q=0.30)
        dip = [10.0] * 260 + [7.9]
        ctx, _ = make_ctx(dip)
        sig = st.entry_signals(ctx)[0]
        st.on_fill(sig, 300, True)                    # A档成交300股 -> held
        self.assertEqual(self.store.pool_state(SYM)["tranches"]["A"]["qty"], 300)
        # 涨回前高10 -> G6 卖 2/3
        back = [10.0] * 260 + [7.9, 10.0]
        ctx2, _ = make_ctx(back, positions=True, q_pm=0.30)
        sells = st.manage(ctx2)
        g6 = [s for s in sells if s.rule == "G6-A"]
        self.assertEqual(len(g6), 1)
        self.assertEqual(g6[0].qty, 200)               # int(300*0.667) 取整手
        self.assertAlmostEqual(g6[0].sell_ratio, 0.667)
        st.on_fill(g6[0], 200, True)                  # -> partial
        self.assertEqual(self.store.pool_state(SYM)["tranches"]["A"]["state"],
                         "partial")

    # -------- G7:移动止盈 --------
    def test_g7_trailing_exit(self):
        st = self._strategy(q=0.30)
        dip = [10.0] * 260 + [7.9]
        ctx, _ = make_ctx(dip)
        sig = st.entry_signals(ctx)[0]
        st.on_fill(sig, 300, True)
        back = [10.0] * 260 + [7.9, 10.0]
        ctx2, _ = make_ctx(back, positions=True, q_pm=0.30)
        g6 = [s for s in st.manage(ctx2) if s.rule == "G6-A"][0]
        st.on_fill(g6, 200, True)
        # 冲高12,回落到9(=12*0.75) -> G7 清尾
        up = [10.0] * 260 + [7.9, 10.0, 12.0]
        ctx3, _ = make_ctx(up, positions=True, q_pm=0.30)
        st.manage(ctx3)
        self.assertEqual(self.store.pool_state(SYM)["tranches"]["A"]["trailing_high"],
                         12.0)
        down = [10.0] * 260 + [7.9, 10.0, 12.0, 9.0]
        ctx4, _ = make_ctx(down, positions=True, q_pm=0.30)
        sells = st.manage(ctx4)
        g7 = [s for s in sells if s.rule == "G7-A"]
        self.assertEqual(len(g7), 1)
        self.assertEqual(g7[0].sell_ratio, 1.0)
        st.on_fill(g7[0], 100, True)                  # -> available,可再触发
        self.assertEqual(self.store.pool_state(SYM)["tranches"]["A"]["state"],
                         "available")

    # -------- G8:高估值清仓 --------
    def test_g8_overvaluation_clear_all(self):
        st = self._strategy(q=0.30)
        dip = [10.0] * 260 + [7.9]
        ctx, _ = make_ctx(dip)
        sig = st.entry_signals(ctx)[0]
        st.on_fill(sig, 300, True)
        hot = [10.0] * 260 + [7.9, 10.0]              # 涨回前高但估值分位95%
        ctx2, _ = make_ctx(hot, positions=True, q_pm=0.95)
        # G8 在 G6 之前触发,全清优先
        st_pm = self._strategy(q=0.95)
        sells = st_pm.manage(ctx2)
        g8 = [s for s in sells if s.rule == "G8"]
        self.assertEqual(len(g8), 1)
        self.assertEqual(g8[0].sell_ratio, 1.0)
        st_pm.on_fill(g8[0], 300, True)
        stA = self.store.pool_state(SYM)["tranches"]["A"]
        self.assertEqual(stA["state"], "available")
        self.assertEqual(stA["qty"], 0)

    # -------- G10:风险事件 --------
    def test_g10_risk_event_st_all(self):
        st = self._strategy(q=0.30)
        dip = [10.0] * 260 + [7.9]
        ctx, _ = make_ctx(dip)
        sig = st.entry_signals(ctx)[0]
        st.on_fill(sig, 300, True)
        st_named = self._strategy(q=0.30)
        ctx2, _ = make_ctx(dip, positions=True, name="ST浦发", q_pm=0.30)
        sells = st_named.manage(ctx2)
        g10 = [s for s in sells if s.rule == "G10"]
        self.assertEqual(len(g10), 1)
        self.assertEqual(g10[0].sell_ratio, 1.0)
        self.assertTrue(self.store.pool_state(SYM).get("risk_exit"))
        # 成交后 status=risk_exit,不再有新信号
        st_named.on_fill(g10[0], 300, True)
        ctx3, _ = make_ctx(dip, name="ST浦发", q_pm=0.30)
        self.assertEqual(st_named.entry_signals(ctx3), [])

    # -------- G12:深回撤复核 --------
    def test_g12_deep_drawdown_review(self):
        st = self._strategy(q=0.30)
        crash = [10.0] * 260 + [3.4]                  # -66%
        ctx, _ = make_ctx(crash)
        self.assertEqual(st.entry_signals(ctx), [])
        state = self.store.pool_state(SYM)
        self.assertTrue(state.get("needs_review"))
        # 复核期间不再触发
        ctx2, _ = make_ctx(crash)
        self.assertEqual(st.entry_signals(ctx2), [])
        # 人工复核通过 -> 复位
        st.reset_symbol(SYM, "复核通过")
        self.assertFalse(self.store.pool_state(SYM).get("needs_review"))

    # -------- G1:持仓家数上限 --------
    def test_g1_max_holdings(self):
        st = self._strategy(q=0.30)
        dip = [10.0] * 260 + [7.9]
        # 已持8只 -> 不再开新仓(本标的未持有)
        pos = {}
        for i in range(8):
            s = f"SH60010{i}"
            pos[s] = Position(symbol=s, qty=100, strategy="value_dip")
        ctx, _ = make_ctx(dip, positions=True)
        ctx.positions = pos
        ctx.market_value = 8000.0
        self.assertEqual(st.entry_signals(ctx), [])

    # -------- 信号持久化 round-trip --------
    def test_signal_store_roundtrip(self):
        sig = PendingSignal(symbol=SYM, side=Side.BUY, signal_date="2026-10-20",
                            strategy="value_dip", rule="G3-A", tranche="A",
                            trigger_high=10.0, size_hint_pct=0.04,
                            reason="测试")
        self.store.save_signal(sig)
        loaded = self.store.pending_signals()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].rule, "G3-A")
        self.assertEqual(loaded[0].tranche, "A")
        self.assertAlmostEqual(loaded[0].trigger_high, 10.0)
        self.assertEqual(loaded[0].side, Side.BUY)
        # 状态更新后从待办里消失
        sig.status = "EXECUTED"
        sig.filled_qty = 100
        self.store.save_signal(sig)
        self.assertEqual(self.store.pending_signals(), [])


if __name__ == "__main__":
    unittest.main()
