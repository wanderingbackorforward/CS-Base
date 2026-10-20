"""策略:个股档位抄底(A股投资准则 G1-G12,只交易 pool.yaml 池内股票)。

档位状态机(per symbol,存 store.pool_states):
    available --G3触发--> held --G6涨回前高--> partial --G7回落25%--> available
                    |                                  |--G8高估值--> available
                    |--G10风险事件--------------------------> 清仓+available

核心纪律(每条信号都带规则编号,对齐 D3"没有规则编号的交易算违规"):
- G3 回撤档位:前高=252日最高收盘;-20%/-35%/-50% 各买 1 份(每份净值4%);
- G4 买入闸门:估值分位<=50%,估值数据缺失视为"不可知 -> 不买";
- G5 档位占用:用掉后卖回之前不重复触发;财报前3个交易日不买;
- G6 涨回该份触发日的前高 -> 卖这份的 2/3;
- G7 剩 1/3 移动止盈:从 G6 后最高收盘回落 25% 才卖;
- G8 估值分位 > 90% -> 全部清仓;
- G10 ST/风险事件 -> 次日全卖(跌停挂单顺延,X3);
- G12 回撤 > 65% -> 停止加仓,标记强制复核(不卖,人工);
- G1/G2 仓位:单只12%(3份),同行业<=25%,个股合计<=60%,最多8只。

信号在收盘后生成(X1),次日开盘撮合;本策略不产生盘中动作。
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from ..config import Section
from ..models import PendingSignal, Side
from .base import Context, Strategy

log = logging.getLogger("qtrader.value")

_TRANCHE_DEFAULT = {"state": "available", "trigger_high": 0.0, "qty": 0,
                    "avg_cost": 0.0, "trailing_high": 0.0, "signal_date": ""}


class ValueDip(Strategy):
    name = "value_dip"

    def __init__(self, cfg: Section, pool_manager, store) -> None:
        s = cfg["strategy"][self.name]
        self.enabled = bool(s.get("enabled", True))
        self.pm = pool_manager          # qtrader.pool.PoolManager
        self.store = store
        self.high_window = int(s.get("high_window", 252))
        self.tranches = [(str(t["key"]), float(t["dd"]))
                         for t in s.get("tranches", [{"key": "A", "dd": -0.20}])]
        self.tranche_pct = float(s.get("tranche_pct", 0.04))
        self.per_stock_cap = float(s.get("per_stock_cap", 0.12))
        self.max_holdings = int(s.get("max_holdings", 8))
        self.industry_cap = float(s.get("industry_cap", 0.25))
        self.stocks_cap = float(s.get("stocks_cap", 0.60))
        self.buy_q_max = float(s.get("buy_val_quantile_max", 0.50))
        self.sell_q_min = float(s.get("sell_val_quantile_min", 0.90))
        self.sell_back_ratio = float(s.get("sell_back_ratio", 0.667))
        self.trail_fallback = float(s.get("trailing_fallback", 0.25))
        self.deep_dd = float(s.get("deep_dd_review", 0.65))
        self.one_per_day = bool(s.get("one_tranche_per_day", True))

    # ---------------- 档位状态读写 ----------------
    def _state(self, symbol: str) -> dict:
        st = self.store.pool_state(symbol, default=None)
        if not st:
            st = {"status": "active", "tranches": {}, "pe_q": None,
                  "needs_review": False, "review_note": ""}
        for k in (t[0] for t in self.tranches):
            st["tranches"].setdefault(k, dict(_TRANCHE_DEFAULT))
        return st

    def _save_state(self, symbol: str, st: dict) -> None:
        self.store.save_pool_state(symbol, st)

    # ---------------- 收盘后主入口(X1) ----------------
    def entry_signals(self, ctx: Context) -> list[PendingSignal]:
        """G3 买入信号:档位触发 + G4 估值闸门 + G5 占用/财报/风控事件检查。"""
        if not self.enabled:
            return []
        pool = ctx.pool
        if not pool:
            log.info("value_dip:池子为空,不产生买入信号(先配 pool.yaml)")
            return []
        out: list[PendingSignal] = []
        today = ctx.today
        equity = ctx.equity()
        held = [s for s, p in ctx.positions.items() if p.strategy == self.name]
        blackout = ctx.blackout  # G5 财报前不买(尽力而为)
        pending_syms = {s.symbol for s in ctx.pending}

        for e in pool:
            sym = e["symbol"]
            st = self._state(sym)
            if st.get("status") != "active" or st.get("needs_review"):
                continue  # G10 已退池 / G12 待复核:停新仓
            if st.get("risk_exit"):
                continue
            hist = ctx.hist.get(sym)
            if hist is None or len(hist) < self.high_window // 2:
                continue
            closes = hist["close"].reset_index(drop=True)
            window = closes.iloc[-self.high_window:]
            if len(window) < self.high_window // 2:
                continue
            prev_high = float(window.max())
            if prev_high <= 0:
                continue
            close = float(closes.iloc[-1])
            dd = close / prev_high - 1

            # G12 深回撤:不再加仓,标记复核(不卖,人工判断理由卡)
            if dd <= -self.deep_dd and not st.get("needs_review"):
                st["needs_review"] = True
                st["review_note"] = f"回撤{dd:.1%}触及G12复核线,人工复核理由卡"
                self._save_state(sym, st)
                log.warning("G12:%s 回撤 %.1f%% 超过 %.0f%%,停止加仓待人工复核",
                            sym, dd * 100, self.deep_dd * 100)
                continue
            if st.get("needs_review"):
                continue

            # G1 持仓家数上限
            if sym not in held and len(held) >= self.max_holdings:
                continue
            # G5 财报前不买
            if sym in blackout:
                log.info("G5:%s 财报披露临近,本单不买", sym)
                continue

            # G3 档位触发:找最浅的一个未占用档位(单只单日最多1个)
            fired = None
            for key, dd_th in self.tranches:
                tr = st["tranches"][key]
                if tr["state"] != "available":
                    continue  # G5 档位占用:卖回之前不重复触发
                if dd <= dd_th:
                    fired = (key, dd_th)
                    if self.one_per_day:
                        break
            if not fired:
                continue
            key, dd_th = fired

            # G4 估值闸门:分位 <= 50%;估值不可知 -> 不买
            industry = str(e.get("industry", ""))
            if ctx.pool_val_q is not None and sym in ctx.pool_val_q:
                q = ctx.pool_val_q[sym]
            else:
                q = self.pm.val_quantile(sym, industry)
                if ctx.pool_val_q is not None:
                    ctx.pool_val_q[sym] = q
            if q is None:
                log.info("G4:%s 估值分位不可知,不买(数据缺失按保守处理)", sym)
                continue
            if q > self.buy_q_max:
                continue

            # G2 仓位闸门:单只/行业/个股合计
            pos = ctx.positions.get(sym)
            held_pct = (pos.qty * close / equity) if pos else 0.0
            if held_pct + self.tranche_pct > self.per_stock_cap + 1e-9:
                continue
            same_industry_pct = self._industry_exposure(ctx, industry, equity) \
                if industry else 0.0
            if industry and sym not in held \
                    and same_industry_pct + self.tranche_pct > self.industry_cap + 1e-9:
                log.info("G2:%s 行业(%s)占比 %.0f%% 将超上限 %.0f%%",
                         sym, industry, same_industry_pct * 100, self.industry_cap * 100)
                continue
            stocks_pct = self._stocks_exposure(ctx, equity)
            if sym not in held and stocks_pct + self.tranche_pct > self.stocks_cap + 1e-9:
                log.info("G2:个股仓 %.0f%% + 新份 %.0f%% 超上限 %.0f%%",
                         stocks_pct * 100, self.tranche_pct * 100, self.stocks_cap * 100)
                continue
            if sym in pending_syms:
                continue  # 当日已有该票信号(防同日多条)

            sig = PendingSignal(
                symbol=sym, side=Side.BUY, signal_date=today, strategy=self.name,
                rule=f"G3-{key}", tranche=key, trigger_high=prev_high,
                size_hint_pct=self.tranche_pct,
                reason=(f"G3-{key}档:自252日前高{prev_high:.2f}回撤{dd:.1%}"
                        f"(触发线{dd_th:.0%}),估值分位{q:.0%}<=50%,买1份"
                        f"({self.tranche_pct:.0%}净值)"))
            # 信号日即占用档位(防次日重复生成;执行失败由回调释放)
            st["tranches"][key] = {"state": "pending", "trigger_high": prev_high,
                                   "qty": 0, "avg_cost": 0.0, "trailing_high": 0.0,
                                   "signal_date": today}
            st["pe_q"] = q
            self._save_state(sym, st)
            out.append(sig)
            log.info("信号 %s", sig.reason)
        return out

    def manage(self, ctx: Context) -> list[PendingSignal]:
        """卖出信号:G6 涨回前高卖2/3 / G7 回落25%清尾 / G8 高估值清仓 / G10 风险事件。"""
        if not self.enabled:
            return []
        out: list[PendingSignal] = []
        equity = ctx.equity()
        pending_syms = {(s.symbol, s.rule) for s in ctx.pending}
        pool_syms = {e["symbol"] for e in ctx.pool}
        for sym, pos in list(ctx.positions.items()):
            if pos.strategy != self.name:
                continue
            st = self._state(sym)
            if st.get("risk_exit"):
                continue
            # G11:持仓已移出生效池 -> 强制复核(人工确认失效才卖,系统不越权)
            if sym not in pool_syms and not st.get("needs_review"):
                st["needs_review"] = True
                st["review_note"] = ("G11:已不在生效池(P6重筛移出?),"
                                     "人工复核理由卡;确认失效请清仓后运行 reset_symbol")
                self._save_state(sym, st)
                log.warning("G11 %s:已移出生效池,停止加仓,待人工复核", sym)
                continue
            hist = ctx.hist.get(sym)
            close = float(hist["close"].iloc[-1]) if hist is not None and len(hist) else 0.0
            if close <= 0:
                continue

            # G10 风险事件:ST/退市(名称检测;立案/造假靠人工,系统尽力)
            name = ctx.name_of(sym)
            if name and ("ST" in name.upper() or "退" in name):
                out.append(PendingSignal(
                    symbol=sym, side=Side.SELL, signal_date=ctx.today,
                    strategy=self.name, rule="G10", sell_ratio=1.0,
                    reason=f"G10 风险事件:证券简称变为[{name}],次日全部卖出"
                           f"(跌停则按X3逐日挂单)"))
                st["risk_exit"] = True
                self._save_state(sym, st)
                log.warning("G10 风险事件 %s: %s -> 全部卖出", sym, name)
                continue

            # G8 高估值清仓:分位 > 90%
            industry = ctx.industry_of(sym)
            q = (ctx.pool_val_q or {}).get(sym)
            if q is None:
                q = self.pm.val_quantile(sym, industry)
                if ctx.pool_val_q is not None:
                    ctx.pool_val_q[sym] = q
            st["pe_q"] = q
            if q is not None and q > self.sell_q_min:
                if (sym, "G8") not in pending_syms:
                    out.append(PendingSignal(
                        symbol=sym, side=Side.SELL, signal_date=ctx.today,
                        strategy=self.name, rule="G8", sell_ratio=1.0,
                        reason=f"G8 高估值:分位{q:.0%}>90%,全部清仓"))
                    for tr in st["tranches"].values():
                        tr.update({"state": "selling_all"})
                    self._save_state(sym, st)
                    log.info("G8 %s 估值分位 %.0f%% 清仓", sym, q * 100)
                continue

            # G6/G7 逐份检查
            for key, tr in st["tranches"].items():
                if (sym, f"G6-{key}") in pending_syms or (sym, f"G7-{key}") in pending_syms:
                    continue
                if tr["state"] == "held" and close >= tr["trigger_high"] > 0:
                    qty = int(tr["qty"] * self.sell_back_ratio)
                    qty -= qty % 100
                    if qty >= 100:
                        out.append(PendingSignal(
                            symbol=sym, side=Side.SELL, signal_date=ctx.today,
                            strategy=self.name, rule=f"G6-{key}", tranche=key,
                            qty=qty, sell_ratio=self.sell_back_ratio,
                            reason=(f"G6-{key}档涨回触发日前高{tr['trigger_high']:.2f}"
                                    f"(现{close:.2f}),卖出该份的2/3")))
                        tr["state"] = "selling_23"
                        tr["trailing_high"] = close
                        self._save_state(sym, st)
                        log.info("G6 %s %s档:涨回前高,卖2/3(%d股)", sym, key, qty)
                elif tr["state"] == "partial":
                    tr["trailing_high"] = max(tr["trailing_high"], close)
                    if tr["trailing_high"] > 0 and close <= tr["trailing_high"] * (1 - self.trail_fallback):
                        out.append(PendingSignal(
                            symbol=sym, side=Side.SELL, signal_date=ctx.today,
                            strategy=self.name, rule=f"G7-{key}", tranche=key,
                            sell_ratio=1.0,
                            reason=(f"G7-{key}档移动止盈:自高点{tr['trailing_high']:.2f}"
                                    f"回落{self.trail_fallback:.0%},卖出剩余1/3")))
                        tr["state"] = "selling_last"
                        self._save_state(sym, st)
                    else:
                        self._save_state(sym, st)
            self._save_state(sym, st)
        return out

    # ---------------- 撮合回调(engine 执行成交后调用) ----------------
    def on_fill(self, sig: PendingSignal, filled_qty: int, ok: bool) -> None:
        """档位状态机的成交确认:买入成功->held;失败->释放档位;
        G6 卖2/3成交->partial;G7/G8/G10 清仓成交->恢复 available。
        卖出失败(跌停挂起)保持中间态,信号仍 PENDING 次日重试(X3)。"""
        if sig.strategy != self.name:
            return
        st = self._state(sig.symbol)
        if sig.side == Side.BUY:
            if not sig.tranche:
                return
            tr = st["tranches"].get(sig.tranche)
            if tr is None:
                return
            if ok:
                tr.update({"state": "held", "qty": filled_qty})
            else:
                tr.update({"state": "available", "qty": 0, "trigger_high": 0.0})
        elif sig.side == Side.SELL:
            if not ok:  # 失败回退中间态,等次日重试或人工处理
                if sig.rule.startswith("G6") and sig.tranche:
                    st["tranches"][sig.tranche]["state"] = "held"
                elif sig.rule.startswith("G7") and sig.tranche:
                    st["tranches"][sig.tranche]["state"] = "partial"
            else:
                if sig.rule.startswith("G6") and sig.tranche:
                    st["tranches"][sig.tranche]["state"] = "partial"
                elif sig.rule.startswith("G7") and sig.tranche:
                    tr = st["tranches"][sig.tranche]
                    tr.update({"state": "available", "qty": 0, "trigger_high": 0.0,
                               "trailing_high": 0.0})
                elif sig.rule in ("G8", "G10"):
                    for t in st["tranches"].values():
                        t.update({"state": "available", "qty": 0, "trigger_high": 0.0,
                                  "trailing_high": 0.0})
                    if sig.rule == "G10":
                        st["status"] = "risk_exit"
        self._save_state(sig.symbol, st)

    def reset_symbol(self, symbol: str, note: str = "") -> None:
        """人工复核通过后复位状态(P6/G12)。"""
        st = self._state(symbol)
        st["needs_review"] = False
        st["risk_exit"] = False
        if note:
            st["review_note"] = note
        st["status"] = "active"
        self._save_state(symbol, st)

    # ---------------- 仓位统计 ----------------
    def _industry_exposure(self, ctx: Context, industry: str, equity: float) -> float:
        total = 0.0
        for s, p in ctx.positions.items():
            if p.strategy == self.name and ctx.industry_of(s) == industry:
                total += p.qty * ctx.price(s)
        return total / equity if equity > 0 else 1.0

    def _stocks_exposure(self, ctx: Context, equity: float) -> float:
        total = sum(p.qty * ctx.price(s) for s, p in ctx.positions.items()
                   if p.strategy == self.name)
        return total / equity if equity > 0 else 1.0
