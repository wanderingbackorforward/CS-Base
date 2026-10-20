"""策略:美股卫星仓(美股投资准则 v1.1 S1-S13 + R2/R3,优质个股档位抄底)。

个股由"理由卡"决定买卖(us_pool.yaml,必填):
- S6 大跌分3档抄底:-15%/-30%/-45%(前高=252日最高收盘;每份=单只额度10%的1/3);
- S7 买入前确认理由卡仍成立;财报公布前3个交易日不买(读 next_earnings 字段);
- S9 涨回触发日前高 -> 卖这一份的 2/3;S10 剩 1/3 移动止盈,普通股回落25%、
     高波动个股(vol_class: high)回落30%,专门用来吃台积电/AMD 涨过前高的大行情;
- S11 理由失效 -> 人工确认后全部卖出(系统标记 needs_review,不越权);
- S12 "涨太多了""感觉要跌"不是卖出理由 -> 代码里根本没有这条路径;
- S13 回撤超60% -> 停止加仓,强制复核(不设固定止损,这是你选择承担的风险);
- S5 过去5年最大回撤 <= 70% 才允许新买(数据驱动闸门);
- R2 单只上限10%,涨超15%减回10%;R3 同行业合计 <= 15%;R4 卫星仓合计 <= 20%。

档位状态机(同 A股 value_dip,存 store.pool_states):
    available --S6触发--> pending --成交--> held --S9涨回前高--> partial
                             ^                                 |
                             +--S10回落25%/30%卖出后回到 available+
"""
from __future__ import annotations

import logging
from datetime import date

from ..config import Section
from ..models import PendingSignal, Side
from .base import Context, Strategy

log = logging.getLogger("qtrader.us_sat")

_TRANCHE_DEFAULT = {"state": "available", "trigger_high": 0.0, "qty": 0,
                    "trailing_high": 0.0, "signal_date": ""}


class UsSatellite(Strategy):
    name = "us_satellite"

    def __init__(self, us_cfg: Section, store) -> None:
        self.store = store
        s = us_cfg["satellite"]
        self.enabled = bool(us_cfg.get("enabled", True))
        self.high_window = int(s.get("high_window", 252))
        self.tranches = [(str(t["key"]), float(t["dd"]))
                         for t in s.get("tranches", [{"key": "A", "dd": -0.15}])]
        self.per_stock_pct = float(s.get("per_stock_pct", 0.10))
        self.tranche_pct = round(self.per_stock_pct / len(self.tranches), 4)
        self.trim_trigger = float(s.get("trim_trigger", 0.15))
        self.industry_cap = float(s.get("industry_cap", 0.15))
        self.sat_cap = float(s.get("total_pct", 0.20))
        self.max_stocks = int(s.get("max_stocks", 3))
        self.max_dd_5y = float(s.get("max_dd_5y", 0.70))
        self.earnings_gap = int(s.get("earnings_gap_days", 3))
        self.sell_back_ratio = float(s.get("sell_back_ratio", 0.667))
        self.trail_normal = float(s.get("trailing_normal", 0.25))
        self.trail_high = float(s.get("trailing_high", 0.30))
        self.deep_dd = float(s.get("deep_dd_review", 0.60))

    # ---------------- 状态读写 ----------------
    def _state(self, symbol: str) -> dict:
        st = self.store.pool_state(symbol, default=None)
        if not st or "tranches" not in st:
            st = {"status": "active", "tranches": {}, "needs_review": False,
                  "review_note": "", "vol_class": "normal"}
        for k in (t[0] for t in self.tranches):
            st["tranches"].setdefault(k, dict(_TRANCHE_DEFAULT))
        return st

    def _save(self, symbol: str, st: dict) -> None:
        self.store.save_pool_state(symbol, st)

    # ---------------- 收盘后买入信号 ----------------
    def entry_signals(self, ctx: Context) -> list[PendingSignal]:
        if not self.enabled:
            return []
        out: list[PendingSignal] = []
        today = ctx.today
        equity = ctx.equity()
        if equity <= 0 or not ctx.pool:
            return out
        pool = {str(e["symbol"]): e for e in ctx.pool}
        held = [s for s, p in ctx.positions.items()
                if p.strategy == self.name and s in pool]
        pending_syms = {s.symbol for s in ctx.pending}

        for sym, e in pool.items():
            st = self._state(sym)
            if st.get("status") != "active" or st.get("needs_review"):
                continue  # S13/S11 待复核:停新仓
            hist = ctx.hist.get(sym)
            if hist is None or len(hist) < self.high_window // 2:
                continue
            closes = hist["close"].reset_index(drop=True)
            window = closes.iloc[-self.high_window:]
            prev_high = float(window.max())
            if prev_high <= 0:
                continue
            close = float(closes.iloc[-1])
            dd = close / prev_high - 1

            # S5:过去5年最大回撤 <= 70%(数据驱动闸门,拿不到足够历史则放行由人工兜底)
            fivey = closes.iloc[-(252 * 5):] if len(closes) >= 252 * 5 else closes
            if len(fivey) >= 252:
                cummax = fivey.cummax()
                max_dd = float((fivey / cummax - 1).min())
                if max_dd < -self.max_dd_5y:
                    log.info("S5:%s 近5年最大回撤 %.0f%% 超上限 %.0f%%,不买",
                             sym, abs(max_dd) * 100, self.max_dd_5y * 100)
                    continue

            # S13:档位用完之后回撤超60% -> 停止加仓,强制复核理由卡
            if dd <= -self.deep_dd and not st.get("needs_review"):
                st["needs_review"] = True
                st["review_note"] = (f"回撤{dd:.1%}触及S13复核线,人工复核理由卡;"
                                     f"成立则继续持有等回前高,失效则按S11清仓")
                self._save(sym, st)
                log.warning("S13:%s 回撤 %.1f%% 超 %.0f%%,停止加仓待人工复核",
                            sym, dd * 100, self.deep_dd * 100)
                continue

            # S7:理由卡必填(thesis/invalid_signals 缺失 = 理由不成立 -> 不买)
            thesis = e.get("thesis") or []
            if not thesis or not (e.get("invalid_signals") or []):
                log.info("S7:%s 理由卡未填全,不买", sym)
                continue
            # S7:财报公布前 N 个交易日不买(人工维护 next_earnings;自然日近似交易日)
            nxt = str(e.get("next_earnings", "") or "")
            if nxt:
                try:
                    gap = (date.fromisoformat(nxt[:10])
                           - date.fromisoformat(today)).days
                    if 0 <= gap <= self.earnings_gap:
                        log.info("S7:%s 财报日%s临近(%d天),本单不买", sym, nxt, gap)
                        continue
                except ValueError:
                    pass

            # R4:最多3只 / 卫星仓合计 <= 20%
            if sym not in held and len(held) >= self.max_stocks:
                continue
            if sym not in pending_syms:
                sat_pct = self._sat_exposure(ctx, equity)
                if sym not in held and sat_pct + self.tranche_pct > self.sat_cap + 1e-9:
                    log.info("R4:卫星仓 %.0f%% + 新份 %.1f%% 超上限 %.0f%%",
                             sat_pct * 100, self.tranche_pct * 100, self.sat_cap * 100)
                    continue

            # S6:档位触发(浅档优先;单只单日最多1个)
            fired = None
            for key, dd_th in self.tranches:
                tr = st["tranches"][key]
                if tr["state"] != "available":
                    continue  # S8:用掉后卖回之前不重复触发
                if dd <= dd_th:
                    fired = (key, dd_th)
                    break
            if not fired:
                continue
            key, dd_th = fired

            # R2:单只上限 10%(3份打满)
            pos = ctx.positions.get(sym)
            held_pct = (pos.qty * close / equity) if pos else 0.0
            if held_pct + self.tranche_pct > self.per_stock_pct + 1e-9:
                continue
            # R3:同行业合计 <= 15%
            industry = str(e.get("industry", ""))
            if industry:
                ind_pct = self._industry_exposure(ctx, industry, equity)
                if sym not in held and ind_pct + self.tranche_pct > self.industry_cap + 1e-9:
                    log.info("R3:%s 行业(%s)占比 %.0f%% 将超上限 %.0f%%",
                             sym, industry, ind_pct * 100, self.industry_cap * 100)
                    continue
            if sym in pending_syms:
                continue

            out.append(PendingSignal(
                symbol=sym, side=Side.BUY, signal_date=today, strategy=self.name,
                rule=f"S6-{key}", tranche=key, trigger_high=prev_high,
                size_hint_pct=self.tranche_pct,
                reason=(f"S6-{key}档:自252日前高{prev_high:.2f}回撤{dd:.1%}"
                        f"(触发线{dd_th:.0%}),理由卡成立,买1份"
                        f"({self.tranche_pct:.1%})")))
            st["tranches"][key] = {"state": "pending", "trigger_high": prev_high,
                                   "qty": 0, "trailing_high": 0.0,
                                   "signal_date": today}
            st["vol_class"] = str(e.get("vol_class", "normal"))
            self._save(sym, st)
            log.info("信号 S6-%s %s 回撤 %.1f%%", key, sym, dd * 100)
        return out

    # ---------------- 持仓管理(S9/S10/S11/R2) ----------------
    def manage(self, ctx: Context) -> list[PendingSignal]:
        if not self.enabled:
            return []
        out: list[PendingSignal] = []
        equity = ctx.equity()
        pool_syms = {str(e["symbol"]) for e in ctx.pool}
        pending_keys = {(s.symbol, s.rule) for s in ctx.pending}
        for sym, pos in list(ctx.positions.items()):
            if pos.strategy != self.name:
                continue
            st = self._state(sym)
            # S11:已不在理由卡池 -> 强制复核(人工确认失效才卖,系统不越权)
            if sym not in pool_syms and not st.get("needs_review"):
                st["needs_review"] = True
                st["review_note"] = ("S11:已不在 us_pool.yaml,人工复核理由卡;"
                                     "确认失效请清仓后 reset_symbol")
                self._save(sym, st)
                log.warning("S11 %s:已移出生效池,停止加仓,待人工复核", sym)
                continue
            hist = ctx.hist.get(sym)
            if hist is None or not len(hist):
                continue
            close = float(hist["close"].iloc[-1])
            if close <= 0:
                continue

            # R2:涨超15% -> 减回10%(唯一主动减仓条款)
            pos_pct = pos.qty * close / equity if equity > 0 else 0.0
            if pos_pct > self.trim_trigger and (sym, "R2") not in pending_keys:
                keep_qty = int(self.per_stock_pct * equity / close)
                sell_qty = pos.qty - keep_qty
                if sell_qty > 0:
                    out.append(PendingSignal(
                        symbol=sym, side=Side.SELL, signal_date=ctx.today,
                        strategy=self.name, rule="R2", qty=sell_qty,
                        sell_ratio=round(sell_qty / pos.qty, 4),
                        reason=(f"R2 单只占比{pos_pct:.1%}超过{self.trim_trigger:.0%},"
                                f"减回{self.per_stock_pct:.0%}(卖{sell_qty}股)")))
                    log.info("信号 R2 %s 占比 %.1f%% 减仓 %d 股",
                             sym, pos_pct * 100, sell_qty)
                    continue

            # S9/S10 逐份检查
            trail = self.trail_high if st.get("vol_class") == "high" \
                else self.trail_normal
            for key, tr in st["tranches"].items():
                if (sym, f"S9-{key}") in pending_keys or (sym, f"S10-{key}") in pending_keys:
                    continue
                if tr["state"] == "held" and close >= tr["trigger_high"] > 0:
                    qty = int(tr["qty"] * self.sell_back_ratio)
                    if qty > 0:
                        out.append(PendingSignal(
                            symbol=sym, side=Side.SELL, signal_date=ctx.today,
                            strategy=self.name, rule=f"S9-{key}", tranche=key,
                            qty=qty, sell_ratio=round(min(1.0, qty / pos.qty), 4),
                            reason=(f"S9-{key}档收盘{close:.2f}涨回触发日前高"
                                    f"{tr['trigger_high']:.2f},卖出该份2/3({qty}股)")))
                        tr["state"] = "selling_23"
                        tr["trailing_high"] = close
                        log.info("信号 S9-%s %s 涨回前高,卖2/3(%d股)", key, sym, qty)
                elif tr["state"] == "partial":
                    tr["trailing_high"] = max(tr["trailing_high"], close)
                    if tr["trailing_high"] > 0 \
                            and close <= tr["trailing_high"] * (1 - trail):
                        out.append(PendingSignal(
                            symbol=sym, side=Side.SELL, signal_date=ctx.today,
                            strategy=self.name, rule=f"S10-{key}", tranche=key,
                            qty=tr["qty"], sell_ratio=round(
                                min(1.0, tr["qty"] / max(pos.qty, 1)), 4),
                            reason=(f"S10-{key}档移动止盈:自高点"
                                    f"{tr['trailing_high']:.2f}回落{trail:.0%},"
                                    f"卖出剩余1/3({tr['qty']}股)")))
                        tr["state"] = "selling_last"
                        log.info("信号 S10-%s %s 自高点回落 %.0f%%", key, sym, trail * 100)
            self._save(sym, st)
        return out

    # ---------------- 成交回调 ----------------
    def on_fill(self, sig: PendingSignal, filled_qty: int, ok: bool) -> None:
        if sig.strategy != self.name:
            return
        st = self._state(sig.symbol)
        if sig.side == Side.BUY:
            if not sig.tranche or sig.tranche not in st["tranches"]:
                return
            if ok:
                st["tranches"][sig.tranche].update({"state": "held", "qty": filled_qty})
            else:
                st["tranches"][sig.tranche].update(
                    {"state": "available", "qty": 0, "trigger_high": 0.0})
        else:
            if not ok:  # 失败回退中间态,信号保留 PENDING 次日重试
                if sig.rule.startswith("S9") and sig.tranche in st["tranches"]:
                    st["tranches"][sig.tranche]["state"] = "held"
                elif sig.rule.startswith("S10") and sig.tranche in st["tranches"]:
                    st["tranches"][sig.tranche]["state"] = "partial"
            else:
                if sig.rule.startswith("S9") and sig.tranche in st["tranches"]:
                    st["tranches"][sig.tranche]["state"] = "partial"
                elif sig.rule.startswith("S10") and sig.tranche in st["tranches"]:
                    tr = st["tranches"][sig.tranche]
                    tr.update({"state": "available", "qty": 0, "trigger_high": 0.0,
                               "trailing_high": 0.0})
                elif sig.rule in ("S11",):
                    for t in st["tranches"].values():
                        t.update({"state": "available", "qty": 0,
                                  "trigger_high": 0.0, "trailing_high": 0.0})
                    st["status"] = "risk_exit"
        self._save(sig.symbol, st)

    def reset_symbol(self, symbol: str, note: str = "") -> None:
        """人工复核通过后复位状态(S13/S11)。"""
        st = self._state(symbol)
        st["needs_review"] = False
        if note:
            st["review_note"] = note
        st["status"] = "active"
        self._save(symbol, st)

    # ---------------- 仓位统计 ----------------
    def _industry_exposure(self, ctx: Context, industry: str, equity: float) -> float:
        total = 0.0
        for s, p in ctx.positions.items():
            if p.strategy == self.name:
                e = next((x for x in ctx.pool
                          if str(x.get("symbol")) == s), {})
                if str(e.get("industry", "")) == industry:
                    total += p.qty * ctx.price(s)
        return total / equity if equity > 0 else 1.0

    def _sat_exposure(self, ctx: Context, equity: float) -> float:
        total = sum(p.qty * ctx.price(s) for s, p in ctx.positions.items()
                    if p.strategy == self.name)
        return total / equity if equity > 0 else 1.0
