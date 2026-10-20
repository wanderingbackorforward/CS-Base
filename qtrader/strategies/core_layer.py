"""策略:核心层(投资系统准则 v5 C1-C3)。一直持有,只做再平衡。

核心保证长期至少拿到大部分指数收益:
- C1 首次建仓:分3个月、每月等额买完(月末信号次日执行);
- C2 再平衡带:只在核心和储备之间做,按目标比例(美股40:10 → 核心80%,
      A股55:15 → 核心约79%),季度调仓日偏离 ≥ 5个百分点才调回;进攻仓不参与;
- C3 新钱先用:每月存入的钱先放现金;调仓日优先补低配最多的部分;闲置现金
      超过缓冲线即扫入停泊(货基/短债ETF)——储备平时不挪作他用。

核心不止盈、不择时、不抄底;"低买高卖"完全交给再平衡带。
美股核心只买一只标普500 ETF(进攻仓已重仓科技,再配纳指=同方向押两次)。
"""
from __future__ import annotations

import logging

from ..config import Section
from ..models import PendingSignal, Side
from .base import Context, Strategy

log = logging.getLogger("qtrader.core")


class CoreLayer(Strategy):
    name = "core"

    def __init__(self, section: Section, store, *, market: str = "cn",
                 pool_manager=None) -> None:
        self.store = store
        self.market = market               # "cn" | "us"
        self.enabled = True
        c = section["core"]
        alloc = section["allocation"]
        self.etfs = list(c.get("etfs") or [])
        self.dca_months = max(1, int(c.get("dca_months", 3)))
        self.park_symbol = str(c.get("park_symbol",
                                     "SH511990" if market == "cn" else "USSGOV"))
        self.park_min_amount = float(c.get("park_min_amount", 10000))
        self.sweep_buffer = float(c.get("sweep_buffer", 0.02))
        self.core_pct = float(alloc.get("core_pct", 0.40))
        self.reserve_pct = float(alloc.get("reserve_pct", 0.15))
        self.drift = float(alloc.get("rebalance_drift", 0.05))
        # 核心占"核心+储备"的目标份额(美股 40/(40+10)=80%)
        self.target_share = self.core_pct / max(self.core_pct + self.reserve_pct,
                                                1e-9)
        if not self.etfs:
            log.warning("core 层无 ETF 配置,只做停泊扫入")

    # ---------------- 状态 ----------------
    def _dca(self) -> dict:
        st = self.store.get_meta("core_dca", default=None) or {}
        st.setdefault("done", {})          # {etf_symbol: 已买月数}
        return st

    def _save_dca(self, st: dict) -> None:
        self.store.set_meta("core_dca", st)

    def _dca_built(self, sym: str) -> bool:
        return self._dca()["done"].get(sym, 0) >= self.dca_months

    def _core_mv(self, ctx: Context) -> tuple[float, dict]:
        """核心层市值与各ETF市值。"""
        per: dict[str, float] = {}
        total = 0.0
        for e in self.etfs:
            sym = e["symbol"]
            pos = ctx.positions.get(sym)
            mv = pos.qty * ctx.price(sym) if pos is not None else 0.0
            per[sym] = mv
            total += mv
        return total, per

    # ---------------- 收盘后买入信号 ----------------
    def entry_signals(self, ctx: Context) -> list[PendingSignal]:
        out: list[PendingSignal] = []
        today = ctx.today
        equity = ctx.equity()
        if equity <= 0:
            return out
        pending_syms = {(s.symbol, s.rule) for s in ctx.pending}

        # ---- C3:闲置现金扫入停泊(新钱先留 buffer 现金,其余停泊等待) ----
        out += self._sweep(ctx, equity, today, pending_syms)

        # ---- C1:首次建仓,分3个月、每月等额(月末信号) ----
        if ctx.month_end:
            dca = self._dca()
            for e in self.etfs:
                sym = e["symbol"]
                tgt = float(e.get("target_pct", 0.0))
                if dca["done"].get(sym, 0) >= self.dca_months:
                    continue
                if (sym, "C1") in pending_syms:
                    continue
                # 已有持仓但没有建仓记录(人工预置)-> 视为已建完,交给 C2
                if sym in ctx.positions and sym not in dca["done"]:
                    dca["done"][sym] = self.dca_months
                    continue
                month_pct = round(tgt / self.dca_months, 5)
                out.append(PendingSignal(
                    symbol=sym, side=Side.BUY, signal_date=today,
                    strategy=self.name, rule="C1",
                    size_hint_pct=month_pct,
                    reason=(f"C1 核心建仓:{self.name_of(sym)} 目标{tgt:.0%},"
                            f"分{self.dca_months}个月每月等额,本月买{month_pct:.2%}"
                            f"(第{dca['done'].get(sym, 0) + 1}/{self.dca_months}期)")))
            self._save_dca(dca)

        # ---- C2:再平衡带,季度调仓日,只在核心与储备之间 ----
        if ctx.quarter_day:
            out += self._rebalance(ctx, equity, today, pending_syms)
        return out

    def _sweep(self, ctx: Context, equity: float, today: str,
               pending_syms: set) -> list[PendingSignal]:
        """C3:现金超过缓冲即停泊。储备/进攻仓未用资金都在这里等。"""
        if (self.park_symbol, "C3") in pending_syms:
            return []
        cash_pct = ctx.account.cash / equity if equity > 0 else 0.0
        excess = cash_pct - self.sweep_buffer
        if excess <= 0:
            return []
        amount = excess * equity
        if self.market == "cn" and amount < self.park_min_amount:
            return []                       # 一手货基都买不起,按现金停泊
        pos = ctx.positions.get(self.park_symbol)
        if pos is not None and cash_pct <= self.sweep_buffer * 2:
            return []                       # 已有停泊仓,现金只是正常波动
        log.info("C3 停泊:现金%.0f%%超缓冲%.0f%%,扫入%s",
                 cash_pct * 100, self.sweep_buffer * 100, self.park_symbol)
        return [PendingSignal(
            symbol=self.park_symbol, side=Side.BUY, signal_date=today,
            strategy=self.name, rule="C3", size_hint_pct=round(excess, 5),
            reason=(f"C3 新钱先放现金:闲置现金{cash_pct:.1%}超缓冲"
                    f"{self.sweep_buffer:.0%},扫入停泊{self.park_symbol}等待;"
                    f"买入时自动变现(储备平时不挪作他用))")]

    def _rebalance(self, ctx: Context, equity: float, today: str,
                   pending_syms: set) -> list[PendingSignal]:
        """C2:核心 vs 储备的再平衡带。

        口径:核心份额 = 核心市值 /(账户市值 - 进攻仓已部署市值)。
        进攻仓未部署的钱也停泊在储备里,但不算核心份额的分母会随进攻仓
        部署自动收缩,季度再收敛一次;进攻仓本身绝不参与 C2 交易。
        """
        core_mv, per_mv = self._core_mv(ctx)
        if any(not self._dca_built(e["symbol"]) for e in self.etfs):
            log.info("C2 跳过:核心 C1 建仓未完成")
            return []
        attack_mv = sum(p.qty * ctx.price(s) for s, p in ctx.positions.items()
                        if p.strategy == "attack")
        envelope = max(ctx.equity() - attack_mv, 1e-9)
        share = core_mv / envelope
        gap = share - self.target_share
        if abs(gap) < self.drift:
            log.info("C2 核心份额 %.1f%%(目标%.1f%%),偏离<%.0fpp,不动",
                     share * 100, self.target_share * 100, self.drift * 100)
            return []
        out: list[PendingSignal] = []
        delta_mv = gap * envelope            # >0 核心超配(卖出),<0 低配(买入)
        if delta_mv > 0:
            # 超配:按各ETF市值比例卖出,钱回停泊(下个周期 C3 扫走)
            for sym, mv in sorted(per_mv.items(), key=lambda kv: -kv[1]):
                if mv <= 0 or (sym, "C2b") in pending_syms:
                    continue
                part = mv / max(core_mv, 1e-9)
                px = ctx.price(sym)
                qty = int(delta_mv * part / px)
                if self.market == "cn":
                    qty -= qty % 100
                if qty <= 0:
                    continue
                out.append(PendingSignal(
                    symbol=sym, side=Side.SELL, signal_date=today,
                    strategy=self.name, rule="C2b", qty=qty,
                    sell_ratio=round(min(1.0, qty / 1000), 5),
                    reason=(f"C2 再平衡:核心份额{share:.1%}超目标"
                            f"{self.target_share:.1%}逾{self.drift:.0%},"
                            f"卖{qty}股回储备停泊")))
        else:
            # 低配:C3 先用新钱/现金,不够引擎自动变现停泊(E3→C3资金链)
            for e in self.etfs:
                sym = e["symbol"]
                if (sym, "C2a") in pending_syms:
                    continue
                tgt = float(e.get("target_pct", 0.0))
                etf_target_mv = (tgt / max(self.core_pct, 1e-9)) \
                    * self.target_share * envelope
                need = etf_target_mv - per_mv.get(sym, 0.0)
                if need <= 0:
                    continue
                out.append(PendingSignal(
                    symbol=sym, side=Side.BUY, signal_date=today,
                    strategy=self.name, rule="C2a",
                    size_hint_pct=round(need / equity, 5),
                    reason=(f"C2 再平衡:核心份额{share:.1%}低于目标"
                            f"{self.target_share:.1%}逾{self.drift:.0%},"
                            f"补{self.name_of(sym)}至目标(C3 先用现金)")))
        log.info("C2 触发:核心份额%.1f%% vs 目标%.1f%%,%s %.0f 元",
                 share * 100, self.target_share * 100,
                 "卖出" if delta_mv > 0 else "买入", abs(delta_mv))
        return out

    def manage(self, ctx: Context) -> list[PendingSignal]:
        return []            # 核心:不止盈、不择时,卖出只发生在 C2 超配

    def on_fill(self, sig: PendingSignal, filled_qty: int, ok: bool,
                price: float = 0.0) -> None:
        if sig.strategy != self.name or not ok:
            return
        if sig.rule == "C1":
            dca = self._dca()
            dca["done"][sig.symbol] = dca["done"].get(sig.symbol, 0) + 1
            self._save_dca(dca)
            log.info("C1 %s 第%d/%d期完成", sig.symbol,
                     dca["done"][sig.symbol], self.dca_months)

    # ---------------- 展示 ----------------
    def snapshot(self, ctx: Context | None = None) -> dict:
        st = self._dca()
        return {"etfs": self.etfs, "dca_months": self.dca_months,
                "dca_done": st.get("done", {}),
                "park_symbol": self.park_symbol,
                "target_share": self.target_share}
