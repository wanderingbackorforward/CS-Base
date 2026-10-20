"""策略:大底储备(投资系统准则 v3 Z1-Z4,只在大熊市动用)。

储备专门留给 2000/2008 那种大熊市,可能十年才用一次;正是这笔钱
让你在最恐慌的时候敢买。这里用历史最高点而不是一年前高,避免在
长期阴跌里被"前高"带着走。

- Z1 美股:纳指100 从历史最高收盘价跌超 35%/45%/55%,各投储备的1/3
     买纳指100ETF(window_years=0 -> 全历史最高);
- Z2 A股:沪深300 从近3年最高收盘价跌超 40%/50%/60%,各投1/3,
     买沪深300或中证A500 ETF;
- Z3 纳指回到当时的历史最高点 / 沪深300回到当时的3年最高点,
     卖出这部分,钱回到储备;
- Z4 储备平时放短期美债ETF或货币基金,不挪作他用;现金超过缓冲线
     即扫入停泊(交易层没用到的钱也停泊在这里,买入时再变现)。

档位状态机(存 store.pool_states["__reserve__"]):
    T1/T2/T3: available -> pending(信号日) -> held -> selling(Z3) -> available
"""
from __future__ import annotations

import logging

from ..config import Section
from ..models import PendingSignal, Side
from .base import Context, Strategy

log = logging.getLogger("qtrader.reserve")

_STATE_KEY = "__reserve__"
_TRANCHE_KEYS = ("T1", "T2", "T3")
_DEFAULT = {"state": "available", "high": 0.0, "qty": 0, "entry_date": ""}


class Reserve(Strategy):
    name = "reserve"

    def __init__(self, section: Section, store, *, market: str = "cn",
                 pool_manager=None) -> None:
        self.store = store
        self.market = market               # "cn" | "us"
        self.enabled = True
        r = section["reserve"]
        self.total_pct = float(r.get("total_pct", 0.15))
        self.park_symbol = str(r.get("park_symbol", "SH511990" if market == "cn"
                                     else "USSGOV"))
        self.park_min_amount = float(r.get("park_min_amount", 10000))
        # Z1 用 track(QQQ 长历史代表纳指);Z2 用 index_symbol(沪深300指数)
        self.track_symbol = str(r.get("track_symbol", "") or
                                r.get("index_symbol", "SH000300"))
        self.buy_symbol = str(r.get("buy_symbol", "SH510300" if market == "cn"
                                    else "USQQQM"))
        self.window_years = int(r.get("window_years", 3))   # 0=历史最高(美股)
        self.thresholds = [float(x) for x in
                           (r.get("thresholds") or [-0.40, -0.50, -0.60])]
        self.sweep_buffer = float(r.get("sweep_buffer", 0.02))
        # 规则编号:美股买入走 Z1,A股买入走 Z2,卖出 Z3,停泊 Z4(D2 可追溯)
        self.buy_rule = "Z1" if market == "us" else "Z2"
        while len(self.thresholds) < len(_TRANCHE_KEYS):
            self.thresholds.append(self.thresholds[-1] - 0.10)
        self.thresholds = self.thresholds[:len(_TRANCHE_KEYS)]
        self.tranche_pct = round(self.total_pct / len(_TRANCHE_KEYS), 5)

    # ---------------- 状态 ----------------
    def _state(self) -> dict:
        st = self.store.pool_state(_STATE_KEY, default=None) or {}
        tr = st.setdefault("tranches", {})
        for k in _TRANCHE_KEYS:
            tr.setdefault(k, dict(_DEFAULT))
        st.setdefault("park_built", False)
        st.setdefault("last_dd", None)
        st.setdefault("last_high", None)
        return st

    def _save(self, st: dict) -> None:
        self.store.save_pool_state(_STATE_KEY, st)

    # ---------------- 参考高点与回撤 ----------------
    def _window_bars(self) -> int:
        """近3年 ≈ 756 个交易日;window_years=0 表示全历史(美股 Z1)。"""
        return 0 if self.window_years <= 0 else self.window_years * 252

    def drawdown(self, hist) -> tuple[float, float] | None:
        """返回 (窗口最高收盘, 当前回撤);数据不足返回 None,绝不猜。"""
        if hist is None or hist.empty:
            return None
        closes = hist["close"].astype(float)
        if len(closes) < 60:
            return None
        bars = self._window_bars()
        window = closes if bars <= 0 else closes.iloc[-bars:]
        high = float(window.max())
        if high <= 0:
            return None
        return high, float(closes.iloc[-1]) / high - 1

    # ---------------- 收盘后买入信号(Z1/Z2 三档 + Z4 停泊扫入) ----------------
    def entry_signals(self, ctx: Context) -> list[PendingSignal]:
        if not self.enabled:
            return []
        out: list[PendingSignal] = []
        today = ctx.today
        equity = ctx.equity()
        if equity <= 0:
            return out
        st = self._state()
        pending_rules = {s.rule for s in ctx.pending
                         if s.strategy == self.name}
        pending_syms = {s.symbol for s in ctx.pending}

        # ---- Z1/Z2:指数大底三档 ----
        hist = ctx.hist.get(self.track_symbol)
        dd_info = self.drawdown(hist)
        if dd_info is None:
            log.info("%s:%s 指数历史不可用,本周期不判定大底",
                     self.buy_rule, self.track_symbol)
        else:
            high, dd = dd_info
            st["last_dd"] = round(dd, 4)
            st["last_high"] = round(high, 3)
            for i, (key, th) in enumerate(zip(_TRANCHE_KEYS, self.thresholds)):
                rule = f"{self.buy_rule}-{i + 1}"
                tr = st["tranches"][key]
                if tr["state"] != "available" or rule in pending_rules:
                    continue
                if dd > th:
                    continue
                out.append(PendingSignal(
                    symbol=self.buy_symbol, side=Side.BUY, signal_date=today,
                    strategy=self.name, rule=rule, tranche=key,
                    trigger_high=high,
                    size_hint_pct=self.tranche_pct,
                    reason=(f"{rule}(第{i + 1}档):{self.track_symbol} 自"
                            + ("历史最高" if self.window_years <= 0
                               else f"近{self.window_years}年最高")
                            + f"收盘 {high:.2f} 回撤 {dd:.1%},触发线 {th:.0%},"
                              f"投入储备的1/3(账户{self.tranche_pct:.1%})买"
                              f"{self.buy_symbol};Z3 回到该高点即卖出")))
                tr.update({"state": "pending", "high": high, "qty": 0,
                           "entry_date": today})
                log.warning("大底信号 %s %s 回撤 %.1f%%(触发线%.0f%%)",
                            rule, self.track_symbol, dd * 100, abs(th) * 100)
            self._save(st)

        # ---- Z4:闲置现金扫入停泊(货基/短债ETF),买入时再变现 ----
        out += self._sweep_signal(ctx, st, equity, today, pending_syms)
        return out

    def _sweep_signal(self, ctx: Context, st: dict, equity: float,
                      today: str, pending_syms: set) -> list[PendingSignal]:
        if not ctx.account or self.park_symbol in pending_syms:
            return []
        cash_pct = ctx.account.cash / equity if equity > 0 else 0.0
        excess_pct = cash_pct - self.sweep_buffer
        if excess_pct <= 0:
            return []
        amount = excess_pct * equity
        # A股场内货基一手约1万元:不足一手按现金停泊(不硬买)
        if self.market == "cn" and amount < self.park_min_amount:
            return []
        # 已有足够停泊仓位就不重复扫(只在现金明显堆积时补)
        park_pos = ctx.positions.get(self.park_symbol)
        if park_pos is not None and park_pos.strategy == self.name \
                and cash_pct <= self.sweep_buffer * 2:
            return []
        out = [PendingSignal(
            symbol=self.park_symbol, side=Side.BUY, signal_date=today,
            strategy=self.name, rule="Z4",
            size_hint_pct=round(excess_pct, 4),
            reason=(f"Z4 停泊:闲置现金 {cash_pct:.1%} 超过缓冲"
                    f"{self.sweep_buffer:.0%},扫入 {self.park_symbol} "
                    f"等待(储备/交易层未用资金,买入时自动变现)"))]
        log.info("Z4 停泊信号:现金%.0f%%超缓冲,扫入%s",
                cash_pct * 100, self.park_symbol)
        return out

    # ---------------- 收盘后持仓管理(Z3 卖出) ----------------
    def manage(self, ctx: Context) -> list[PendingSignal]:
        if not self.enabled:
            return []
        out: list[PendingSignal] = []
        today = ctx.today
        st = self._state()
        pending_rules = {s.rule for s in ctx.pending
                         if s.strategy == self.name}
        pos = ctx.positions.get(self.buy_symbol)
        held = pos is not None and pos.strategy == self.name
        hist = ctx.hist.get(self.track_symbol)
        idx_close = float(hist["close"].iloc[-1]) if hist is not None \
            and len(hist) else ctx.price(self.track_symbol)

        for i, key in enumerate(_TRANCHE_KEYS):
            rule = f"Z3-{i + 1}"
            tr = st["tranches"][key]
            if tr["state"] != "held" or rule in pending_rules:
                continue
            if not held or tr["qty"] <= 0:
                # 持仓已不在(人工清仓/历史遗留):复位档位
                tr.update({"state": "available", "qty": 0, "high": 0.0})
                self._save(st)
                continue
            if tr["high"] <= 0 or idx_close < tr["high"]:
                continue
            qty = min(tr["qty"], pos.available_qty if self.market == "cn"
                      else pos.qty)
            if self.market == "cn":
                qty -= qty % 100
            if qty <= 0:
                log.warning("Z3 %s 第%s档触发但可卖为0(T+1),次日再试",
                            self.buy_symbol, key)
                continue
            out.append(PendingSignal(
                symbol=self.buy_symbol, side=Side.SELL, signal_date=today,
                strategy=self.name, rule=rule, tranche=key, qty=qty,
                sell_ratio=round(min(1.0, qty / max(pos.qty, 1)), 4),
                reason=(f"Z3-{i + 1}:{self.track_symbol} 收盘 {idx_close:.2f} "
                        f"回到当时高点 {tr['high']:.2f},卖出第{i + 1}档"
                        f"({qty}股),钱回到储备停泊")))
            tr["state"] = "selling"
            log.warning("Z3 信号:%s 回到高点,卖出第%s档 %d 股",
                        self.track_symbol, key, qty)
        self._save(st)
        return out

    # ---------------- 成交回调 ----------------
    def on_fill(self, sig: PendingSignal, filled_qty: int, ok: bool) -> None:
        if sig.strategy != self.name:
            return
        st = self._state()
        tr = st["tranches"].get(sig.tranche) if sig.tranche else None
        if sig.side == Side.BUY:
            if sig.rule.startswith(self.buy_rule) and tr is not None:
                if ok:
                    tr.update({"state": "held", "qty": filled_qty})
                else:
                    tr.update({"state": "available", "qty": 0, "high": 0.0})
            elif sig.rule == "Z4" and ok:
                st["park_built"] = True
        else:
            if not ok:
                if sig.rule.startswith("Z3") and tr is not None:
                    tr["state"] = "held"      # 失败回退,信号次日重试(F4同理)
            else:
                if sig.rule.startswith("Z3") and tr is not None:
                    tr.update({"state": "available", "qty": 0, "high": 0.0})
        self._save(st)

    # ---------------- 展示辅助 ----------------
    def snapshot(self) -> dict:
        """看板/状态展示:三档状态 + 最近回撤。"""
        st = self._state()
        return {
            "track_symbol": self.track_symbol,
            "buy_symbol": self.buy_symbol,
            "park_symbol": self.park_symbol,
            "last_high": st.get("last_high"),
            "last_dd": st.get("last_dd"),
            "thresholds": self.thresholds,
            "tranches": {k: dict(st["tranches"][k]) for k in _TRANCHE_KEYS},
        }
