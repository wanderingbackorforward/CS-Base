"""策略:美股核心仓(美股投资准则 v1.1 C1-C9,纳指100 ETF)。

核心仓 70% 分 5 份:2 份底仓长期持有(C4/C7),3 份留给回调档位(C5);
涨回触发当日前高就把这一份全部卖回待命(C8,注意与 A股 G6 的 2/3 不同);
待命资金全部停泊在短期国债 ETF(SGOV)里赚利息(C9)。

档位状态机(per 核心ETF,存 store.pool_states[symbol]):
    available --C5触发--> pending --成交--> held --C8涨回前高--> selling
                                    ^                          |
                                    +------卖回成交后回到 available+

核心纪律:
- C1 只用每日收盘价;C2 前高 = 252 个交易日最高收盘;C3 回撤 = close/前高 - 1;
- C4 启动买 2 份底仓(28%);C7 底仓不卖,只受 R1 年度再平衡调整;
- C5 档位 -10%/-20%/-30% 各买 1 份(14%);C6 每日最多触发一档,占用未卖回不重复触发;
- C8 每份单独记触发日的前高,收盘回到该价位 -> 整份卖出(100%,非 2/3);
- C9 现金(含未触发份额)停泊 SGOV:现金超缓冲线即扫入;需要现金时由引擎卖 SGOV 补。
"""
from __future__ import annotations

import logging

from ..config import Section
from ..models import PendingSignal, Side
from .base import Context, Strategy

log = logging.getLogger("qtrader.us_core")

_TRANCHE_DEFAULT = {"state": "available", "trigger_high": 0.0,
                    "qty": 0, "signal_date": ""}


class UsCore(Strategy):
    name = "us_core"

    def __init__(self, us_cfg: Section, store) -> None:
        self.store = store
        s = us_cfg["core"]
        self.enabled = bool(us_cfg.get("enabled", True))
        self.symbol = str(s.get("symbol", "USQQQM"))
        c = us_cfg["cash"]
        self.park_symbol = str(c.get("symbol", "USSGOV"))
        self.total_pct = float(s.get("total_pct", 0.70))
        self.parts = max(int(s.get("parts", 5)), 1)
        self.base_parts = max(int(s.get("base_parts", 2)), 0)
        self.base_pct = self.total_pct * self.base_parts / self.parts   # C4: 28%
        self.tranche_pct = self.total_pct / self.parts                  # 每份 14%
        self.high_window = int(s.get("high_window", 252))
        self.tranches = [(str(t["key"]), float(t["dd"]))
                         for t in s.get("tranches", [{"key": "A", "dd": -0.10}])]
        self.sweep_buffer = float(c.get("sweep_buffer", 0.02))
        self.drift = float(us_cfg.get("rebalance_drift", 0.05))

    # ---------------- 状态读写 ----------------
    def _state(self) -> dict:
        st = self.store.pool_state(self.symbol, default=None)
        if not st or "tranches" not in st:
            st = {"base": {"qty": 0, "deployed": False},
                  "tranches": {}, "needs_review": False, "review_note": ""}
        for k in (t[0] for t in self.tranches):
            st["tranches"].setdefault(k, dict(_TRANCHE_DEFAULT))
        return st

    def _save(self, st: dict) -> None:
        self.store.save_pool_state(self.symbol, st)

    # ---------------- 收盘后信号(C1:只用收盘价) ----------------
    def entry_signals(self, ctx: Context) -> list[PendingSignal]:
        if not self.enabled:
            return []
        out: list[PendingSignal] = []
        today = ctx.today
        equity = ctx.equity()
        if equity <= 0:
            return out
        pending_keys = {(s.symbol, s.rule.split("-")[0]) for s in ctx.pending}
        st = self._state()
        pos = ctx.positions.get(self.symbol)
        price = ctx.price(self.symbol)
        hist = ctx.hist.get(self.symbol)
        close = float(hist["close"].iloc[-1]) if hist is not None and len(hist) else price

        # ---- C4:启动买入底仓(2/5 = 28%),一次买完 ----
        base = st["base"]
        if not base.get("deployed") and base.get("qty", 0) <= 0 \
                and (self.symbol, "C4") not in pending_keys:
            out.append(PendingSignal(
                symbol=self.symbol, side=Side.BUY, signal_date=today,
                strategy=self.name, rule="C4", size_hint_pct=round(self.base_pct, 4),
                reason=(f"C4 启动底仓:核心仓{self.total_pct:.0%}的"
                        f"{self.base_parts}/{self.parts}={self.base_pct:.0%},一次买完")))
        # ---- R1:年度再平衡日,底仓偏离 >5pp 才调(C7:底仓只在此时动) ----
        elif ctx.rebalance_day and base.get("qty", 0) > 0 and price > 0:
            cur = base["qty"] * price / equity
            gap = self.base_pct - cur
            if abs(gap) > self.drift and (self.symbol, "R1") not in pending_keys:
                if gap > 0:
                    out.append(PendingSignal(
                        symbol=self.symbol, side=Side.BUY, signal_date=today,
                        strategy=self.name, rule="R1", size_hint_pct=round(gap, 4),
                        reason=(f"R1 年度再平衡:底仓{cur:.1%} -> 目标"
                                f"{self.base_pct:.1%},补买{gap:.1%}")))
                else:
                    qty = int(abs(gap) * equity / price)
                    if qty > 0:
                        out.append(PendingSignal(
                            symbol=self.symbol, side=Side.SELL, signal_date=today,
                            strategy=self.name, rule="R1", qty=qty,
                            sell_ratio=round(min(1.0, qty / max(pos.qty, 1)), 4)
                            if pos else 1.0,
                            reason=(f"R1 年度再平衡:底仓{cur:.1%} 超过目标"
                                    f"{self.base_pct:.1%},减回{qty}股")))

        # ---- C5/C6:回调档位(前高=252日最高收盘;每日最多触发一档) ----
        if hist is not None and len(hist) >= self.high_window // 2 and base.get("deployed"):
            closes = hist["close"].reset_index(drop=True)
            window = closes.iloc[-self.high_window:]
            prev_high = float(window.max())
            if prev_high > 0:
                dd = close / prev_high - 1
                for key, dd_th in self.tranches:  # 浅档优先,命中即停(C6)
                    tr = st["tranches"][key]
                    if tr["state"] != "available":
                        continue  # C6:占用未卖回,不重复触发
                    if dd > dd_th:
                        break
                    held_pct = ((pos.qty * close / equity) if pos else 0.0)
                    if held_pct + self.tranche_pct > self.total_pct + 0.02:
                        log.info("C5:%s 已到核心仓上限 %.0f%%,不加仓",
                                 self.symbol, self.total_pct * 100)
                        break
                    out.append(PendingSignal(
                        symbol=self.symbol, side=Side.BUY, signal_date=today,
                        strategy=self.name, rule=f"C5-{key}", tranche=key,
                        trigger_high=prev_high, size_hint_pct=round(self.tranche_pct, 4),
                        reason=(f"C5-{key}档:自252日前高{prev_high:.2f}回撤{dd:.1%}"
                                f"(触发线{dd_th:.0%}),买1份({self.tranche_pct:.0%})")))
                    tr.update({"state": "pending", "trigger_high": prev_high,
                               "qty": 0, "signal_date": today})
                    log.info("信号 C5-%s %s 回撤 %.1f%%", key, self.symbol, dd * 100)
                    break  # C6:每个交易日最多触发一个档位

        # ---- C9:现金停泊 SGOV(含现金池10% + 未触发份额) ----
        out += self._sweep_signal(ctx, today, equity, pending_keys)
        if st.get("needs_review"):
            st["needs_review"] = False  # 核心仓无理由卡,复核标记仅提示用
        self._save(st)
        for s in out:
            log.info("信号 %s", s.reason)
        return out

    def _sweep_signal(self, ctx: Context, today: str, equity: float,
                      pending_keys: set) -> list[PendingSignal]:
        """C9:现金超过缓冲线 -> 扫入 SGOV 计息(未触发份额与现金池的统一停泊处)。"""
        cash_pct = ctx.account.cash / equity if equity > 0 else 0.0
        idle = cash_pct - self.sweep_buffer
        if idle <= 0.005 or (self.park_symbol, "C9") in pending_keys:
            return []
        return [PendingSignal(
            symbol=self.park_symbol, side=Side.BUY, signal_date=today,
            strategy=self.name, rule="C9", size_hint_pct=round(idle, 4),
            reason=(f"C9 待命资金停泊:现金{cash_pct:.1%} 超过缓冲"
                    f"{self.sweep_buffer:.0%},扫入短期国债ETF计息"))]

    def manage(self, ctx: Context) -> list[PendingSignal]:
        """C8:每份涨回触发日前高 -> 整份卖回(100%),资金回到待命(SGOV)。"""
        if not self.enabled:
            return []
        out: list[PendingSignal] = []
        st = self._state()
        pos = ctx.positions.get(self.symbol)
        hist = ctx.hist.get(self.symbol)
        if hist is None or not len(hist) or pos is None:
            return out
        close = float(hist["close"].iloc[-1])
        pending_keys = {(s.symbol, s.rule) for s in ctx.pending}
        for key, tr in st["tranches"].items():
            if tr["state"] != "held" or tr["qty"] <= 0:
                continue
            if (self.symbol, f"C8-{key}") in pending_keys:
                continue
            if close >= tr["trigger_high"] > 0:
                qty = min(tr["qty"], pos.qty)
                if qty > 0:
                    out.append(PendingSignal(
                        symbol=self.symbol, side=Side.SELL, signal_date=ctx.today,
                        strategy=self.name, rule=f"C8-{key}", tranche=key,
                        qty=qty, sell_ratio=round(min(1.0, qty / pos.qty), 4),
                        reason=(f"C8-{key}档收盘{close:.2f}涨回触发日前高"
                                f"{tr['trigger_high']:.2f},整份卖出{qty}股回待命")))
                    tr["state"] = "selling"
                    log.info("信号 C8-%s %s 涨回前高,整份卖回 %d 股",
                             key, self.symbol, qty)
        self._save(st)
        return out

    # ---------------- 成交回调 ----------------
    def on_fill(self, sig: PendingSignal, filled_qty: int, ok: bool) -> None:
        if sig.strategy != self.name:
            return
        st = self._state()
        if sig.side == Side.BUY:
            if not ok:
                if sig.tranche and sig.tranche in st["tranches"]:
                    st["tranches"][sig.tranche].update(
                        {"state": "available", "qty": 0, "trigger_high": 0.0})
                elif sig.rule == "C4":
                    pass  # 底仓未成交,明日再试(状态未推进)
            else:
                if sig.rule == "C4":
                    st["base"]["deployed"] = True
                    st["base"]["qty"] = st["base"].get("qty", 0) + filled_qty
                elif sig.rule == "R1":
                    st["base"]["qty"] = max(0, st["base"].get("qty", 0) + filled_qty)
                elif sig.tranche and sig.tranche in st["tranches"]:
                    st["tranches"][sig.tranche].update(
                        {"state": "held", "qty": filled_qty})
        else:  # SELL
            if not ok:
                if sig.tranche and sig.tranche in st["tranches"]:
                    st["tranches"][sig.tranche]["state"] = "held"
            else:
                if sig.tranche and sig.tranche in st["tranches"] \
                        and sig.rule.startswith("C8"):
                    st["tranches"][sig.tranche].update(
                        {"state": "available", "qty": 0, "trigger_high": 0.0})
                elif sig.rule == "R1":
                    st["base"]["qty"] = max(0, st["base"].get("qty", 0) - filled_qty)
        self._save(st)
