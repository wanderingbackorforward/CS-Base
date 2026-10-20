"""策略:底层仓(投资系统准则 v3 L1-L5,只持有不择时)。

底层仓不止盈、不抄底、不追涨——"卖得太早"和"踏空"在这一层不会发生。
- L1 美股:纳指100 ETF 40% + 标普500 ETF 35%,分3个月、每月等额买完,
     之后不择时;
- L2 A股:红利低波ETF 20% + 老登股3~5只(准入 U2,来自 pool.yaml kind=old)
     等权合计 20%;
- L3 每年1月第一个交易日再平衡,偏离目标超过5个百分点才调整;
- L4 唯一例外:A股老登股触发硬失效(F2)时卖出,钱换入红利低波ETF;
- L5 A股个股持有超1年股息免税 -> 底层老登股不频繁换股。

账本纪律:同一标的同时只归一个策略管——交易层已持有的股票不选入底层,
底层已持有的股票交易层也不再买(持仓按 strategy 归属,一层一账)。
"""
from __future__ import annotations

import logging

from ..config import Section
from ..models import PendingSignal, Side
from .base import Context, Strategy

log = logging.getLogger("qtrader.base")

_STATE_KEY = "__base_layer__"


class BaseLayer(Strategy):
    name = "base_layer"

    def __init__(self, section: Section, store, *, market: str = "cn",
                 pool_manager=None) -> None:
        self.store = store
        self.market = market
        self.pm = pool_manager
        self.enabled = True
        b = section["base"]
        alloc = section.get("allocation", {}) or {}
        if market == "cn":
            # L2:红利低波ETF + 老登股等权
            self.etf_symbol = str(b["dividend_etf"]["symbol"])
            self.etf_target = float(b["dividend_etf"]["target_pct"])
            self.old_target = float(b["old_stocks"]["target_pct"])
            self.old_min = int(b["old_stocks"].get("min_count", 3))
            self.old_max = int(b["old_stocks"].get("max_count", 5))
            self.etfs = [(self.etf_symbol, self.etf_target)]
            self.months = 1                      # A股一次性建仓
        else:
            # L1:两只宽基ETF,分3个月每月等额
            self.etfs = [(str(e["symbol"]), float(e["target_pct"]))
                         for e in b.get("etfs", [])]
            self.months = max(int(b.get("months", 3)), 1)
        self.drift = float(section.get("rebalance_drift",
                                       alloc.get("rebalance_drift", 0.05)))

    # ---------------- 状态 ----------------
    def _state(self) -> dict:
        st = self.store.pool_state(_STATE_KEY, default=None) or {}
        st.setdefault("etf_built", False)        # A股:红利ETF底仓已建
        st.setdefault("months_bought", [])        # US:已扣款的月份 YYYY-MM
        st.setdefault("old_targets", {})          # A股:{老登股: 目标占比}
        return st

    def _save(self, st: dict) -> None:
        self.store.save_pool_state(_STATE_KEY, st)

    # ---------------- 收盘后:建仓/DCA/再平衡 ----------------
    def entry_signals(self, ctx: Context) -> list[PendingSignal]:
        if not self.enabled:
            return []
        out: list[PendingSignal] = []
        today = ctx.today
        equity = ctx.equity()
        if equity <= 0:
            return out
        st = self._state()
        pending_keys = {(s.symbol, s.rule.split("-")[0]) for s in ctx.pending}

        if self.market == "cn":
            out += self._cn_build(ctx, st, equity, today, pending_keys)
        else:
            out += self._us_dca(ctx, st, equity, today, pending_keys)

        # L3:年度再平衡,偏离 > 5pp 才调
        if ctx.rebalance_day:
            out += self._rebalance(ctx, st, equity, today, pending_keys)
        return out

    # ---------------- A股 L2 建仓 ----------------
    def _cn_build(self, ctx: Context, st: dict, equity: float, today: str,
                  pending_keys: set) -> list[PendingSignal]:
        out: list[PendingSignal] = []
        # 红利低波ETF底仓(一次买完)
        if not st["etf_built"] and (self.etf_symbol, "L2") not in pending_keys \
                and self.etf_symbol not in ctx.positions:
            out.append(PendingSignal(
                symbol=self.etf_symbol, side=Side.BUY, signal_date=today,
                strategy=self.name, rule="L2",
                size_hint_pct=round(self.etf_target, 4),
                reason=(f"L2 底层仓建仓:红利低波ETF {self.etf_symbol} "
                        f"目标{self.etf_target:.0%},一次性买完,之后不择时")))
        # 老登股:pool kind=old 等权;账本纪律——交易层已持有的不选
        if not st["old_targets"]:
            olds = [e for e in ctx.pool
                    if str(e.get("kind", "")) == "old" and not e.get("hard_fail")]
            picked = [e for e in olds
                      if e["symbol"] not in ctx.positions
                      or ctx.positions[e["symbol"]].strategy == self.name]
            picked = picked[: self.old_max]
            if 0 < len(picked) < self.old_min:
                log.warning("L2:老登股候选仅 %d 只(<%d),空缺额度暂留现金,"
                            "下次 U4 重筛后补齐", len(picked), self.old_min)
            if picked:
                per = round(self.old_target / len(picked), 4)
                st["old_targets"] = {str(e["symbol"]): per for e in picked}
                self._save(st)
                log.info("L2:老登股 %d 只等权入选(每只%.1f%%):%s",
                         len(picked), per * 100,
                         ",".join(str(e["symbol"]) for e in picked))
        for sym, tgt in list(st["old_targets"].items()):
            pos = ctx.positions.get(sym)
            if pos is not None and pos.strategy == self.name:
                continue
            if (sym, "L2") in pending_keys:
                continue
            entry = next((e for e in ctx.pool
                          if str(e.get("symbol")) == sym), None)
            if entry is None or entry.get("hard_fail"):
                continue     # 建仓途中被移出/失效:不买,等重筛
            out.append(PendingSignal(
                symbol=sym, side=Side.BUY, signal_date=today,
                strategy=self.name, rule="L2", size_hint_pct=tgt,
                reason=(f"L2 底层仓:老登股 {sym} 等权{tgt:.1%}"
                        f"(准入U2,{entry.get('name', '')}),"
                        f"持有超1年股息免税不频繁换股(L5)")))
        return out

    # ---------------- 美股 L1:分3个月每月等额 ----------------
    def _us_dca(self, ctx: Context, st: dict, equity: float, today: str,
                pending_keys: set) -> list[PendingSignal]:
        out: list[PendingSignal] = []
        if len(st["months_bought"]) >= self.months:
            return out
        month = today[:7]
        if month in st["months_bought"]:
            return out
        per = round(1.0 / self.months, 4)
        for sym, tgt in self.etfs:
            if (sym, "L1") in pending_keys:
                continue
            out.append(PendingSignal(
                symbol=sym, side=Side.BUY, signal_date=today,
                strategy=self.name, rule="L1",
                size_hint_pct=round(tgt * per, 4),
                reason=(f"L1 底层仓定投:{sym} 目标{tgt:.0%}的第"
                        f"{len(st['months_bought']) + 1}/{self.months}个月"
                        f"({tgt * per:.1%}),买完不择时")))
        return out

    # ---------------- L4:老登股硬失效换红利ETF ----------------
    def manage(self, ctx: Context) -> list[PendingSignal]:
        if not self.enabled or self.market != "cn":
            return []
        out: list[PendingSignal] = []
        today = ctx.today
        equity = ctx.equity()
        st = self._state()
        pool = {str(e["symbol"]): e for e in ctx.pool}
        pending_keys = {(s.symbol, s.rule.split("-")[0]) for s in ctx.pending}
        for sym, tgt in list(st["old_targets"].items()):
            pos = ctx.positions.get(sym)
            if pos is None or pos.strategy != self.name:
                continue
            entry = pool.get(sym)
            name = ctx.name_of(sym)
            reason = ""
            if entry is None:
                reason = "被移出准入名单(U4重筛)"
            elif entry.get("hard_fail"):
                reason = str(entry.get("hard_fail_reason",
                                       "财报硬条件失效(F2)"))
            elif name and ("ST" in name.upper() or "退" in name):
                reason = f"证券简称变为[{name}]"
            if not reason:
                continue
            if (sym, "L4") in pending_keys:
                continue
            px = ctx.price(sym) or pos.avg_cost
            value_pct = round(pos.qty * px / equity, 4) if equity > 0 else 0.0
            out.append(PendingSignal(
                symbol=sym, side=Side.SELL, signal_date=today,
                strategy=self.name, rule="L4", sell_ratio=1.0,
                reason=f"L4 老登股硬失效:{reason};卖出后钱换入红利低波ETF"))
            out.append(PendingSignal(
                symbol=self.etf_symbol, side=Side.BUY, signal_date=today,
                strategy=self.name, rule="L4",
                size_hint_pct=value_pct,
                reason=(f"L4 换仓:{sym} 硬失效卖出(约{value_pct:.1%}净值),"
                        f"等额换入红利低波ETF {self.etf_symbol}")))
            st["old_targets"].pop(sym, None)
            st.setdefault("exited_olds", {})[sym] = reason
            self._save(st)
            log.warning("L4 %s 硬失效(%s),卖出换入红利ETF", sym, reason)
        return out

    # ---------------- L3:年度再平衡(偏离>5pp) ----------------
    def _rebalance(self, ctx: Context, st: dict, equity: float, today: str,
                   pending_keys: set) -> list[PendingSignal]:
        out: list[PendingSignal] = []
        if self.market == "cn":
            targets = dict(st.get("old_targets") or {})
            if st.get("etf_built") or self.etf_symbol in ctx.positions:
                targets[self.etf_symbol] = self.etf_target
        else:
            targets = dict(self.etfs)
        for sym, tgt in targets.items():
            pos = ctx.positions.get(sym)
            if pos is None or pos.strategy != self.name:
                continue
            px = ctx.price(sym)
            if px <= 0:
                continue
            cur = pos.qty * px / equity if equity > 0 else 0.0
            gap = tgt - cur
            if abs(gap) <= self.drift:
                continue
            if (sym, "L3") in pending_keys:
                continue
            if gap > 0:
                out.append(PendingSignal(
                    symbol=sym, side=Side.BUY, signal_date=today,
                    strategy=self.name, rule="L3", size_hint_pct=round(gap, 4),
                    reason=(f"L3 年度再平衡:{sym} 现{cur:.1%} -> 目标"
                            f"{tgt:.1%},补买{gap:.1%}")))
            else:
                qty = int(abs(gap) * equity / px)
                if qty > 0:
                    out.append(PendingSignal(
                        symbol=sym, side=Side.SELL, signal_date=today,
                        strategy=self.name, rule="L3", qty=qty,
                        sell_ratio=round(min(1.0, qty / max(pos.qty, 1)), 4),
                        reason=(f"L3 年度再平衡:{sym} 现{cur:.1%} 超过目标"
                                f"{tgt:.1%},减回{qty}股")))
        if out:
            log.info("L3 年度再平衡:%d 条调仓信号(其余偏离均在5pp内)",
                     len(out))
        return out

    # ---------------- 成交回调 ----------------
    def on_fill(self, sig: PendingSignal, filled_qty: int, ok: bool) -> None:
        if sig.strategy != self.name:
            return
        st = self._state()
        if not ok:
            return
        if sig.rule in ("L1",):
            month = sig.signal_date[:7]
            if month not in st["months_bought"]:
                st["months_bought"].append(month)
                st["months_bought"].sort()
        elif sig.rule == "L2" and sig.symbol == self.etf_symbol:
            st["etf_built"] = True
        elif sig.rule == "L4" and sig.side == Side.SELL:
            st["old_targets"].pop(sig.symbol, None)
        self._save(st)

    # ---------------- 展示辅助 ----------------
    def targets(self) -> dict[str, float]:
        st = self._state()
        out = {}
        if self.market == "cn":
            if st.get("etf_built") or True:
                out[self.etf_symbol] = self.etf_target
            out.update(st.get("old_targets") or {})
        else:
            out.update(dict(self.etfs))
        return out
