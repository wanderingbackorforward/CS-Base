"""策略:交易层通用引擎(投资系统准则 v3 E1-E7,美股/A股共用)。

引擎公式两边一样,参数按每只股票自己的波动率σ自动缩放:
波动大的档位深、止盈宽;波动小的档位浅、止盈窄。不按板块手写参数。

- E1 σ = 252日日收益率std × √252(年化),每月最后交易日更新;
- E2 前高 = 252日最高收盘;回撤 = 收盘/前高 - 1;只用收盘价;
- E3 抄底:额度分3份,回撤达 a1/a2/a3 倍σ 各买1份(a3·σ 最深不超60%),
     A股还须估值分位 <= 50%(不可知=不买);财报前不开新仓;
- E4 每份记触发日前高,涨回卖这份的 2/3;剩余1/3自之后最高收盘回落 b·σ 卖;
- E5 追涨:收盘创252日新高且50日线>200日线,买1份追涨份;自买入后
     最高收盘回落 c·σ 卖;A股买入后15个交易日没再创新高就卖(时间止损);
- E6 档位卖回前不重复触发;追涨份不摊平、不转抄底份;
- E7 抄底份/追涨份分开标记与统计(pool_states 状态机);
- F1 单只 <= 账户5%,同行业上限(A股15%/美股10%);
- F2 硬失效(季度亏损/ST/立案/移出准入名单)次日全卖(机器可判部分自动,
     其余由 pool.yaml hard_fail 人工标注);
- F3 交易层整体自最高点回撤超25% -> 停止新开仓1个月(持仓管理继续)。

档位状态机(per symbol,存 store.pool_states):
    dip.A/B/C: available -> pending(信号日) -> held -> selling_23 -> partial
               -> selling_last -> available
    chase:      none -> pending -> held -> selling -> none
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from ..config import Section
from ..models import PendingSignal, Side
from .base import Context, Strategy

log = logging.getLogger("qtrader.trade")

_DIP_KEYS = ("A", "B", "C")
_DIP_DEFAULT = {"state": "available", "trigger_high": 0.0, "qty": 0,
                "trailing_high": 0.0, "signal_date": ""}
_CHASE_DEFAULT = {"state": "none", "entry_high": 0.0, "trailing_high": 0.0,
                  "qty": 0, "bars_held": 0, "signal_date": ""}

_META_PEAK = "trade:peak_mv"
_META_HALT_UNTIL = "trade:halt_until"
_META_HALT_NOTE = "trade:halt_note"


class TradeEngine(Strategy):
    name = "trade_engine"

    def __init__(self, section: Section, store, *, market: str = "cn",
                 pool_manager=None) -> None:
        self.store = store
        self.market = market              # "cn" | "us"
        self.pm = pool_manager             # A股估值分位(美股为 None)
        s = section["trade"]
        self.enabled = True
        # E1/E2
        self.sigma_window = int(s.get("sigma_window", 252))
        self.high_window = int(s.get("high_window", 252))
        # E3
        self.a1 = float(s.get("a1", 0.5))
        self.a2 = float(s.get("a2", 0.9))
        self.a3 = float(s.get("a3", 1.3))
        self.deep_cap = float(s.get("deep_cap", 0.60))
        # E4/E5
        self.b = float(s.get("b", 0.6))
        self.c = float(s.get("c", 0.4))
        self.sell_back_ratio = float(s.get("sell_back_ratio", 0.667))
        # F1
        self.per_stock_cap = float(s.get("per_stock_cap", 0.05))
        self.industry_cap = float(s.get("industry_cap", 0.15))
        self.max_holdings = int(s.get("max_holdings", 8))
        self.tranche_pct = round(self.per_stock_cap / 3.0, 5)   # 每份 = 额度/3
        # A股估值条件(美股不设)
        self.val_gate = market == "cn"
        self.buy_q_max = float(s.get("buy_val_quantile_max", 0.50))
        self.sell_q_min = float(s.get("sell_val_quantile_min", 0.90))
        # 财报前不开新仓
        self.earnings_gap = int(s.get("earnings_gap_days", 3))
        # 追涨(E5)
        ch = s.get("chase", {}) or {}
        self.chase_enabled = bool(ch.get("enabled", True))
        self.chase_tech_only = bool(ch.get("tech_only", market == "cn"))
        self.chase_cap_pct = float(ch.get("cap_pct", 0.05))
        self.chase_per_month = int(ch.get("per_month_max", 1))
        self.chase_time_stop = int(ch.get("time_stop_days", 0))   # 0=不设(美股)
        self.chase_day_max = float(ch.get("day_chg_max", 0.0))     # 0=不设
        self.chase_days5_max = float(ch.get("days5_chg_max", 0.0))  # 0=不设
        self.chase_limit_premium = float(ch.get("limit_premium_max", 0.0))
        # U1(美股):过去5年最大回撤 <= 70%
        self.max_dd_5y = float(s.get("max_dd_5y", 0.70))
        # F3(交易层熔断参数在 risk 段)
        rk = section.get("risk", {}) or {}
        self.trade_halt_dd = float(rk.get("trade_halt_dd", -0.25))
        self.trade_halt_days = int(rk.get("trade_halt_days", 30))

    # ---------------- 状态读写 ----------------
    def _state(self, symbol: str) -> dict:
        st = self.store.pool_state(symbol, default=None)
        if not st or "dip" not in st:
            st = {"status": "active", "kind": "", "sigma": 0.0,
                  "sigma_date": "", "dip": {}, "chase": dict(_CHASE_DEFAULT)}
        for k in _DIP_KEYS:
            st["dip"].setdefault(k, dict(_DIP_DEFAULT))
        ch = st.setdefault("chase", dict(_CHASE_DEFAULT))
        for k, v in _CHASE_DEFAULT.items():
            ch.setdefault(k, v)
        return st

    def _save(self, symbol: str, st: dict) -> None:
        self.store.save_pool_state(symbol, st)

    # ---------------- E1:波动率 ----------------
    def _update_sigma(self, st: dict, hist: pd.DataFrame, today: str,
                      month_end: bool) -> float:
        """σ = 252日日收益率std×√252;每月最后交易日更新,缺失立即补算。"""
        need = (st.get("sigma", 0.0) <= 0.0) or (month_end and
                                                 st.get("sigma_date") != today)
        if not need:
            return st.get("sigma", 0.0)
        if hist is None or len(hist) < self.sigma_window // 2:
            return st.get("sigma", 0.0)
        closes = hist["close"].astype(float).iloc[-(self.sigma_window + 1):]
        ret = closes.pct_change().dropna()
        if len(ret) < self.sigma_window // 2:
            return st.get("sigma", 0.0)
        sigma = float(ret.std() * (len(ret) ** 0.5))
        if sigma > 0:
            st["sigma"] = round(sigma, 4)
            st["sigma_date"] = today
        return st.get("sigma", 0.0)

    def _dip_thresholds(self, sigma: float) -> list[tuple[str, float]]:
        """三档抄底线(-a·σ),a3·σ 最深不超过 deep_cap。"""
        a3 = min(self.a3 * sigma, self.deep_cap)
        return [("A", -self.a1 * sigma), ("B", -self.a2 * sigma),
                ("C", -a3)]

    # ---------------- F3:交易层熔断 ----------------
    def _halt_check(self, ctx: Context, trade_mv: float) -> None:
        if trade_mv <= 0:
            return
        peak = float(self.store.get_meta(_META_PEAK, 0.0) or 0.0)
        if trade_mv > peak:
            peak = trade_mv
            self.store.set_meta(_META_PEAK, peak)
        until = str(self.store.get_meta(_META_HALT_UNTIL, "") or "")
        if until and until >= ctx.today:
            return
        dd = trade_mv / peak - 1 if peak > 0 else 0.0
        if dd <= self.trade_halt_dd:
            until = (date.fromisoformat(ctx.today)
                     + timedelta(days=self.trade_halt_days)).isoformat()
            self.store.set_meta(_META_HALT_UNTIL, until)
            self.store.set_meta(
                _META_HALT_NOTE,
                f"F3 交易层自最高点回撤{dd:.1%},停止新开仓1个月(至{until}),强制复盘")
            log.critical("F3 交易层熔断:自峰值%.0f回撤%.1f%%,停新开仓至 %s",
                         peak, dd * 100, until)

    def is_halted(self, today: str) -> bool:
        until = str(self.store.get_meta(_META_HALT_UNTIL, "") or "")
        return bool(until) and until >= today

    def halt_note(self) -> str:
        return str(self.store.get_meta(_META_HALT_NOTE, "") or "")

    def _trade_mv(self, ctx: Context) -> float:
        return sum(p.qty * ctx.price(s) for s, p in ctx.positions.items()
                   if p.strategy == self.name)

    # ---------------- 收盘后买入信号(E3 抄底 + E5 追涨) ----------------
    def entry_signals(self, ctx: Context) -> list[PendingSignal]:
        if not self.enabled or not ctx.pool:
            return []
        if self.is_halted(ctx.today):
            log.warning("F3 熔断中(至 %s),本周期不产生新开仓信号",
                        self.store.get_meta(_META_HALT_UNTIL, ""))
            return []
        out: list[PendingSignal] = []
        today = ctx.today
        equity = ctx.equity()
        if equity <= 0:
            return out
        pool = {str(e["symbol"]): e for e in ctx.pool}
        held_syms = [s for s, p in ctx.positions.items()
                     if p.strategy == self.name]
        pending_syms = {s.symbol for s in ctx.pending}
        held_count = len(held_syms) + len(
            {s.symbol for s in ctx.pending if s.strategy == self.name
             and s.symbol not in held_syms})

        for sym, e in pool.items():
            st = self._state(sym)
            if st.get("status") != "active":
                continue
            if e.get("hard_fail"):
                continue  # F2 标注由 manage() 当晚处理
            # 账本纪律:同一标的同时只归一个策略管(底层仓持有的不进交易层)
            pos_other = ctx.positions.get(sym)
            if pos_other is not None and pos_other.strategy != self.name:
                continue
            hist = ctx.hist.get(sym)
            if hist is None or len(hist) < self.high_window // 2:
                continue
            closes = hist["close"].astype(float).reset_index(drop=True)
            close = float(closes.iloc[-1])
            if close <= 0:
                continue
            st["kind"] = str(e.get("kind", "tech" if self.market == "us" else ""))
            sigma = self._update_sigma(st, hist, today, ctx.month_end)
            if sigma <= 0:
                self._save(sym, st)
                continue  # 波动率不可知:不猜
            self._save(sym, st)

            # ---- 公共闸门 ----
            if sym in pending_syms:
                continue  # 当日已有信号
            if sym in ctx.blackout:
                log.info("E3:%s 财报前%d个交易日不开新仓", sym, self.earnings_gap)
                continue
            if self.market == "us":
                if not self._u1_ok(sym, closes):
                    continue
                if not self._us_earnings_ok(e, today):
                    continue
            if sym not in held_syms and held_count >= self.max_holdings:
                continue  # 交易层持股数上限

            # ---- E3 抄底档位 ----
            out += self._dip_entry(ctx, sym, e, st, closes, close, sigma,
                                  equity, today)
            # ---- E5 追涨份 ----
            out += self._chase_entry(ctx, sym, e, st, closes, close, sigma,
                                     equity, today)
        return out

    # ---------------- E3:三档抄底 ----------------
    def _dip_entry(self, ctx: Context, sym: str, e: dict, st: dict,
                   closes: pd.Series, close: float, sigma: float,
                   equity: float, today: str) -> list[PendingSignal]:
        window = closes.iloc[-self.high_window:]
        prev_high = float(window.max())
        if prev_high <= 0:
            return []
        dd = close / prev_high - 1
        fired = None
        for key, th in self._dip_thresholds(sigma):
            tr = st["dip"][key]
            if tr["state"] != "available":
                continue  # E6:卖回前不重复触发
            if dd <= th:
                fired = (key, th, prev_high)
                break
        if fired is None:
            return []
        key, th, prev_high = fired

        # A股估值条件:分位 <= 50%;不可知 -> 不买
        q = None
        if self.val_gate:
            q = (ctx.pool_val_q or {}).get(sym)
            if q is None and self.pm is not None:
                q = self.pm.val_quantile(sym, str(e.get("industry", "")))
                if ctx.pool_val_q is not None:
                    ctx.pool_val_q[sym] = q
            if q is None:
                log.info("E3:%s 估值分位不可知,不买", sym)
                return []
            if q > self.buy_q_max:
                return []

        # F1:单只/行业上限
        if not self._cap_ok(ctx, sym, e, close, equity, self.tranche_pct,
                            new_holding=(sym not in
                                         {s for s, p in ctx.positions.items()
                                          if p.strategy == self.name})):
            return []

        ths = ", ".join(f"{k}={v:.1%}" for k, v in self._dip_thresholds(sigma))
        sig = PendingSignal(
            symbol=sym, side=Side.BUY, signal_date=today, strategy=self.name,
            rule=f"E3-{key}", tranche=key, trigger_high=prev_high,
            size_hint_pct=self.tranche_pct,
            reason=(f"E3-{key}档:σ{sigma:.0%},自252日前高{prev_high:.2f}回撤"
                    f"{dd:.1%}达触发线{th:.1%}(档位[{ths}]),"
                    + (f"估值分位{q:.0%}<=50%," if self.val_gate else "")
                    + f"买1份(账户{self.tranche_pct:.1%})"))
        st["dip"][key] = {"state": "pending", "trigger_high": prev_high,
                          "qty": 0, "trailing_high": 0.0, "signal_date": today}
        if self.val_gate:
            st["pe_q"] = q
        self._save(sym, st)
        log.info("信号 %s", sig.reason)
        return [sig]

    # ---------------- E5:追涨份 ----------------
    def _chase_entry(self, ctx: Context, sym: str, e: dict, st: dict,
                     closes: pd.Series, close: float, sigma: float,
                     equity: float, today: str) -> list[PendingSignal]:
        if not self.chase_enabled or st["chase"]["state"] != "none":
            return []                       # E6:追涨份持有中不重复、不摊平
        if self.chase_tech_only and st.get("kind") != "tech":
            return []                       # A股:老登股不追涨
        if len(closes) < self.high_window or len(closes) < 200:
            return []
        window = closes.iloc[-self.high_window:]
        if close < float(window.max()):
            return []                       # 未创252日新高
        ma50 = float(closes.iloc[-50:].mean())
        ma200 = float(closes.iloc[-200:].mean())
        if ma50 <= ma200:
            return []                       # 均线形态不成立
        # 每月笔数上限
        mkey = f"trade:chase:{today[:7]}"
        used = int(self.store.get_meta(mkey, 0) or 0)
        if used >= self.chase_per_month:
            log.info("E5:%s 本月追涨已达%d笔上限", sym, self.chase_per_month)
            return []
        # A股:当天涨幅/近5日涨幅过滤
        if self.chase_day_max > 0:
            day_chg = ctx.pct_chg(sym)
            if day_chg >= self.chase_day_max:
                log.info("E5:%s 当天涨幅%.1f%%>=%.0f%%,不追", sym,
                         day_chg * 100, self.chase_day_max * 100)
                return []
        if self.chase_days5_max > 0 and len(closes) >= 6:
            d5 = close / float(closes.iloc[-6]) - 1
            if d5 >= self.chase_days5_max:
                log.info("E5:%s 近5日涨幅%.1f%%>=%.0f%%,不追", sym,
                         d5 * 100, self.chase_days5_max * 100)
                return []
        # 追涨资金上限(全部追涨份合计)
        chase_mv = self._chase_mv(ctx)
        if chase_mv / equity + self.tranche_pct > self.chase_cap_pct + 1e-9:
            log.info("E5:追涨份合计%.1f%% + 新份将超上限%.0f%%",
                     chase_mv / equity * 100, self.chase_cap_pct * 100)
            return []
        # F1:单只/行业上限
        if not self._cap_ok(ctx, sym, e, close, equity, self.tranche_pct,
                            new_holding=(sym not in
                                         {s for s, p in ctx.positions.items()
                                          if p.strategy == self.name})):
            return []

        sig = PendingSignal(
            symbol=sym, side=Side.BUY, signal_date=today, strategy=self.name,
            rule="E5", tranche="E5", trigger_high=close,
            size_hint_pct=self.tranche_pct,
            reason=(f"E5 追涨:收盘{close:.2f}创252日新高,50日线{ma50:.2f}>"
                    f"200日线{ma200:.2f},买1份追涨份(账户{self.tranche_pct:.1%};"
                    f"自最高收盘回落{self.c * sigma:.1%}卖出"
                    + (f";{self.chase_time_stop}个交易日没再创新高就卖"
                       if self.chase_time_stop else "") + ")"))
        st["chase"] = {"state": "pending", "entry_high": close,
                       "trailing_high": close, "qty": 0, "bars_held": 0,
                       "signal_date": today}
        self._save(sym, st)
        self.store.set_meta(mkey, used + 1)
        log.info("信号 %s", sig.reason)
        return [sig]

    # ---------------- 收盘后持仓管理(E4/E5卖出 + F2 + 高估值 + F3) ----------------
    def manage(self, ctx: Context) -> list[PendingSignal]:
        if not self.enabled:
            return []
        out: list[PendingSignal] = []
        today = ctx.today
        equity = ctx.equity()
        pool = {str(e["symbol"]): e for e in ctx.pool}
        pending_keys = {(s.symbol, s.rule) for s in ctx.pending}
        self._halt_check(ctx, self._trade_mv(ctx))

        for sym, pos in list(ctx.positions.items()):
            if pos.strategy != self.name:
                continue
            st = self._state(sym)
            hist = ctx.hist.get(sym)
            close = float(hist["close"].iloc[-1]) if hist is not None \
                and len(hist) else ctx.price(sym)
            if close <= 0:
                continue
            entry = pool.get(sym)

            # ---- F2 硬失效:次日全部卖出(机器可判) ----
            name = ctx.name_of(sym)
            reason_f2 = self._f2_reason(sym, entry, name)
            if reason_f2 and (sym, "F2") not in pending_keys:
                out.append(PendingSignal(
                    symbol=sym, side=Side.SELL, signal_date=today,
                    strategy=self.name, rule="F2", sell_ratio=1.0,
                    reason=f"F2 硬失效:{reason_f2};次日全部卖出"
                           f"(跌停则按F4逐日挂单)"))
                st["status"] = "risk_exit"
                st["f2_note"] = reason_f2
                for tr in st["dip"].values():
                    tr["state"] = "selling_all"
                st["chase"]["state"] = "selling"
                self._save(sym, st)
                log.warning("F2 %s 硬失效:%s -> 全部卖出", sym, reason_f2)
                continue

            # ---- A股高估值清仓:分位 > 90% ----
            if self.val_gate and (sym, "V90") not in pending_keys:
                q = (ctx.pool_val_q or {}).get(sym)
                if q is None and self.pm is not None:
                    q = self.pm.val_quantile(sym, ctx.industry_of(sym))
                    if ctx.pool_val_q is not None:
                        ctx.pool_val_q[sym] = q
                if q is not None:
                    st["pe_q"] = q
                    if q > self.sell_q_min:
                        out.append(PendingSignal(
                            symbol=sym, side=Side.SELL, signal_date=today,
                            strategy=self.name, rule="V90", sell_ratio=1.0,
                            reason=f"高估值清仓:估值分位{q:.0%}>90%,"
                                   f"卖出剩余全部(A股市场参数)"))
                        for tr in st["dip"].values():
                            tr["state"] = "selling_all"
                        st["chase"]["state"] = "selling"
                        self._save(sym, st)
                        log.info("V90 %s 估值分位 %.0f%% 清仓", sym, q * 100)
                        continue

            # ---- E4:抄底份(涨回前高卖2/3 / 剩1/3回落b·σ卖) ----
            for key, tr in st["dip"].items():
                rule_s23 = f"E4-{key}"
                rule_last = f"E4R-{key}"
                if (sym, rule_s23) in pending_keys \
                        or (sym, rule_last) in pending_keys:
                    continue
                if tr["state"] == "held" and tr["trigger_high"] > 0 \
                        and close >= tr["trigger_high"]:
                    qty = int(tr["qty"] * self.sell_back_ratio)
                    qty -= qty % 100 if self.market == "cn" else 0
                    if qty > 0:
                        out.append(PendingSignal(
                            symbol=sym, side=Side.SELL, signal_date=today,
                            strategy=self.name, rule=rule_s23, tranche=key,
                            qty=qty, sell_ratio=round(
                                min(1.0, qty / max(pos.qty, 1)), 4),
                            reason=(f"E4-{key}档:收盘{close:.2f}涨回触发日前高"
                                    f"{tr['trigger_high']:.2f},卖出这份的2/3"
                                    f"({qty}股);剩1/3自最高收盘回落"
                                    f"{self.b * st['sigma']:.1%}卖")))
                        tr["state"] = "selling_23"
                        tr["trailing_high"] = close
                        log.info("E4-%s %s 涨回前高,卖2/3(%d股)", key, sym, qty)
                elif tr["state"] == "partial":
                    tr["trailing_high"] = max(tr["trailing_high"], close)
                    trail_th = tr["trailing_high"] * (1 - self.b * st["sigma"])
                    if tr["trailing_high"] > 0 and close <= trail_th:
                        out.append(PendingSignal(
                            symbol=sym, side=Side.SELL, signal_date=today,
                            strategy=self.name, rule=rule_last, tranche=key,
                            qty=tr["qty"], sell_ratio=round(
                                min(1.0, tr["qty"] / max(pos.qty, 1)), 4),
                            reason=(f"E4R-{key}档:自最高收盘"
                                    f"{tr['trailing_high']:.2f}回落"
                                    f"{self.b * st['sigma']:.1%}(σ"
                                    f"{st['sigma']:.0%}),卖出剩余1/3")))
                        tr["state"] = "selling_last"
                        log.info("E4R-%s %s 剩余1/3回落卖出", key, sym)

            # ---- E5:追涨份管理(回落c·σ卖 / 时间止损 / 新高计数) ----
            ch = st["chase"]
            if ch["state"] == "held" and (sym, "E5S") not in pending_keys \
                    and (sym, "E5T") not in pending_keys:
                if close > ch["trailing_high"]:
                    ch["trailing_high"] = close
                    ch["bars_held"] = 0            # 又创新高:时间止损重新计数
                else:
                    ch["bars_held"] += 1
                    if self.chase_time_stop > 0 \
                            and ch["bars_held"] >= self.chase_time_stop:
                        out.append(PendingSignal(
                            symbol=sym, side=Side.SELL, signal_date=today,
                            strategy=self.name, rule="E5T", tranche="E5",
                            qty=ch["qty"], sell_ratio=round(
                                min(1.0, ch["qty"] / max(pos.qty, 1)), 4),
                            reason=(f"E5T 追涨时间止损:买入后"
                                    f"{ch['bars_held']}个交易日没再创新高"
                                    f"(限{self.chase_time_stop}),卖出追涨份")))
                        ch["state"] = "selling"
                        log.info("E5T %s 追涨时间止损(%d日无新高)",
                                 sym, ch["bars_held"])
                        self._save(sym, st)
                        continue
                    if ch["trailing_high"] > 0 and close <= ch["trailing_high"] \
                            * (1 - self.c * st["sigma"]):
                        out.append(PendingSignal(
                            symbol=sym, side=Side.SELL, signal_date=today,
                            strategy=self.name, rule="E5S", tranche="E5",
                            qty=ch["qty"], sell_ratio=round(
                                min(1.0, ch["qty"] / max(pos.qty, 1)), 4),
                            reason=(f"E5S 追涨份卖出:自买入后最高收盘"
                                    f"{ch['trailing_high']:.2f}回落"
                                    f"{self.c * st['sigma']:.1%}"
                                    f"(σ{st['sigma']:.0%})")))
                        ch["state"] = "selling"
                        log.info("E5S %s 追涨份回落卖出", sym)
            self._save(sym, st)
        return out

    # ---------------- 成交回调 ----------------
    def on_fill(self, sig: PendingSignal, filled_qty: int, ok: bool) -> None:
        if sig.strategy != self.name:
            return
        st = self._state(sig.symbol)
        dip_tr = st["dip"].get(sig.tranche) if sig.tranche else None
        ch = st["chase"]
        if sig.side == Side.BUY:
            if sig.rule.startswith("E3") and dip_tr is not None:
                if ok:
                    dip_tr.update({"state": "held", "qty": filled_qty})
                else:
                    dip_tr.update({"state": "available", "qty": 0,
                                   "trigger_high": 0.0})
            elif sig.rule == "E5":
                if ok:
                    ch.update({"state": "held", "qty": filled_qty})
                else:
                    st["chase"] = dict(_CHASE_DEFAULT)
        else:
            if not ok:   # 失败回退中间态,信号保留次日重试(F4 跌停同理)
                if sig.rule.startswith("E4-") and dip_tr is not None:
                    dip_tr["state"] = "held"
                elif sig.rule.startswith("E4R-") and dip_tr is not None:
                    dip_tr["state"] = "partial"
                elif sig.rule in ("E5S", "E5T"):
                    ch["state"] = "held"
                elif sig.rule == "F2":
                    st["status"] = "risk_exit"
                    for tr in st["dip"].values():
                        tr["state"] = "selling_all"
                elif sig.rule == "V90":
                    for tr in st["dip"].values():
                        tr["state"] = "selling_all"
            else:
                if sig.rule.startswith("E4-") and dip_tr is not None:
                    dip_tr["state"] = "partial"
                elif sig.rule.startswith("E4R-") and dip_tr is not None:
                    dip_tr.update({"state": "available", "qty": 0,
                                   "trigger_high": 0.0, "trailing_high": 0.0})
                elif sig.rule in ("E5S", "E5T"):
                    st["chase"] = dict(_CHASE_DEFAULT)
                elif sig.rule in ("F2", "V90"):
                    for tr in st["dip"].values():
                        tr.update({"state": "available", "qty": 0,
                                   "trigger_high": 0.0, "trailing_high": 0.0})
                    st["chase"] = dict(_CHASE_DEFAULT)
                    if sig.rule == "F2":
                        st["status"] = "risk_exit"   # 回池重筛前不再入场
        self._save(sig.symbol, st)

    def reset_symbol(self, symbol: str, note: str = "") -> None:
        """人工复位:重筛后重新入池/复核通过时调用。"""
        st = self._state(symbol)
        st["status"] = "active"
        st.pop("f2_note", None)
        if note:
            st["review_note"] = note
        self._save(symbol, st)

    def reset_halt(self, note: str = "人工复盘后提前解除") -> None:
        self.store.set_meta(_META_HALT_UNTIL, "")
        self.store.set_meta(_META_HALT_NOTE, note)

    # ---------------- 闸门/统计 ----------------
    def _f2_reason(self, sym: str, entry: dict | None, name: str) -> str:
        """F2 硬失效判定:机器可判的自动判,其余由 pool.yaml hard_fail 标注。"""
        if entry is None:
            return "被移出准入名单(U4重筛)"
        if entry.get("hard_fail"):
            return str(entry.get("hard_fail_reason", "财报硬条件失效(人工标注)"))
        if name and ("ST" in name.upper() or "退" in name):
            return f"证券简称变为[{name}](ST/退市风险)"
        return ""

    def _u1_ok(self, sym: str, closes: pd.Series) -> bool:
        """U1:过去5年最大回撤 <= 70%(历史不足则放行,由准入名单人工兜底)。"""
        fivey = closes.iloc[-(252 * 5):]
        if len(fivey) < 252:
            return True
        max_dd = float((fivey / fivey.cummax() - 1).min())
        if max_dd < -self.max_dd_5y:
            log.info("U1:%s 近5年最大回撤%.0f%%超上限%.0f%%,不买",
                     sym, abs(max_dd) * 100, self.max_dd_5y * 100)
            return False
        return True

    def _us_earnings_ok(self, e: dict, today: str) -> bool:
        nxt = str(e.get("next_earnings", "") or "")
        if not nxt:
            return True
        try:
            gap = (date.fromisoformat(nxt[:10]) - date.fromisoformat(today)).days
        except ValueError:
            return True
        return not (0 <= gap <= self.earnings_gap)

    def _cap_ok(self, ctx: Context, sym: str, e: dict, close: float,
                equity: float, add_pct: float, new_holding: bool) -> bool:
        pos = ctx.positions.get(sym)
        held_pct = (pos.qty * close / equity) if pos is not None else 0.0
        if held_pct + add_pct > self.per_stock_cap + 1e-9:
            log.info("F1:%s 单只%.1f%% + 新份%.1f%% 超上限%.0f%%", sym,
                     held_pct * 100, add_pct * 100, self.per_stock_cap * 100)
            return False
        industry = ctx.industry_of(sym) or str(e.get("industry", ""))
        if industry:
            ind_pct = self._industry_pct(ctx, industry, equity)
            if new_holding and ind_pct + add_pct > self.industry_cap + 1e-9:
                log.info("F1:%s 行业(%s)%.1f%% + 新份将超上限%.0f%%", sym,
                         industry, ind_pct * 100, self.industry_cap * 100)
                return False
        return True

    def _industry_pct(self, ctx: Context, industry: str, equity: float) -> float:
        total = sum(p.qty * ctx.price(s) for s, p in ctx.positions.items()
                   if p.strategy == self.name
                   and (ctx.industry_of(s) or "") == industry)
        return total / equity if equity > 0 else 1.0

    def _chase_mv(self, ctx: Context) -> float:
        """全部追涨份市值(按档位状态机里 chase.held 的 qty 估值)。"""
        total = 0.0
        for sym, st in self.store.all_pool_states().items():
            ch = st.get("chase") or {}
            if ch.get("state") in ("held", "selling") and ch.get("qty"):
                total += ch["qty"] * ctx.price(sym)
        return total
