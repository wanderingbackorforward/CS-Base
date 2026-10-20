"""策略:ETF压舱仓(A股投资准则 E1-E4)。

ETF 只做两件事:降低整体波动 + 没有个股机会时存放资金,规则刻意保持简单:
- E1 红利低波ETF 占10%:启动即买入,长期持有,只在年度再平衡(R1)时调整;
     分红到账买回同一只——纸面模式拿不到分红事件,回测/模拟盘按"长期持有"处理;
- E2 宽基ETF(沪深300/中证A500 二选一)占10%:按指数PE近10年分位分三档,
     <30% 全持有 / 30%-70% 持一半 / >70% 不持有;每月最后一个交易日收盘后检查;
     分位数据拿不到 -> 保持现状不动(保守);
- E3 个股抄底缺钱、现金池不够时,先卖宽基ETF补上(引擎执行层实现);
- E4 ETF 分红暂不征个税,纸面无感,仅记录。

信号同为 PendingSignal(X1:收盘生成,次日开盘撮合),买卖均带规则编号。
"""
from __future__ import annotations

import logging

from ..config import Section
from ..models import PendingSignal, Side
from .base import Context, Strategy

log = logging.getLogger("qtrader.etf")


class EtfCore(Strategy):
    name = "etf_core"

    def __init__(self, cfg: Section, pool_manager, store) -> None:
        s = cfg["strategy"][self.name]
        self.enabled = bool(s.get("enabled", True))
        self.pm = pool_manager
        self.store = store
        d = s["dividend_etf"]
        self.div_symbol = str(d.get("symbol", "SH512890"))
        self.div_target = float(d.get("target_pct", 0.10))
        b = s["broad_etf"]
        self.broad_symbol = str(b.get("symbol", "SH510300"))
        self.broad_target = float(b.get("target_pct", 0.10))
        self.low_q = float(b.get("low_q", 0.30))
        self.high_q = float(b.get("high_q", 0.70))
        self.index_val_years = int(s.get("index_val_window_years", 10))
        self.monthly = bool(s.get("monthly_check", True))
        self.drift = float(cfg["allocation"].get("rebalance_drift", 0.05))
        self.cash_floor = float(cfg["allocation"].get("cash_floor", 0.05))
        # 指数名(乐咕按指数名查询,沪深300ETF -> 沪深300)
        self.index_name = str(s.get("index_name", "沪深300"))
        self._last_index_q: float | None = None

    # ---------------- E2:目标持有比例 ----------------
    def broad_target_pct(self) -> tuple[float | None, float | None]:
        """按指数估值分位返回宽基ETF目标占比。
        返回 (target, q);q=None 表示数据不可知 -> 保持现状。"""
        q = self.pm.index_quantile(self.index_name)
        self._last_index_q = q
        if q is None:
            return None, None
        if q < self.low_q:
            return self.broad_target, q
        if q <= self.high_q:
            return self.broad_target / 2, q
        return 0.0, q

    # ---------------- 收盘后信号 ----------------
    def entry_signals(self, ctx: Context) -> list[PendingSignal]:
        """E1 启动买入 + E2 月度检查(月末/R1再平衡日)调仓。"""
        if not self.enabled:
            return []
        out: list[PendingSignal] = []
        today = ctx.today
        equity = ctx.equity()
        if equity <= 0:
            return out
        pending_keys = {(s.symbol, s.rule) for s in ctx.pending}

        # ---- E1 红利低波:启动即买足;此后只在年度再平衡日(R1)校正缺口 ----
        div_pos = ctx.positions.get(self.div_symbol)
        if div_pos is None or ctx.rebalance_day:
            out += self._adjust_to_target(
                ctx, self.div_symbol, self.div_target, "E1", today,
                equity, pending_keys, allow_sell=False,
                note="红利低波ETF,长期持有;启动买入/年度再平衡校正")

        # ---- E2 宽基:每月最后交易日 / 年度再平衡日检查 ----
        if self.monthly and (ctx.month_end or ctx.rebalance_day):
            target, q = self.broad_target_pct()
            if target is None:
                log.info("E2:指数估值分位不可知,保持现状(数据源限制)")
            else:
                band = f"分位{q:.0%}:" + (
                    "低位全持有" if target >= self.broad_target
                    else ("中位持一半" if target > 0 else "高位不持有"))
                out += self._adjust_to_target(
                    ctx, self.broad_symbol, target, "E2", today,
                    equity, pending_keys, allow_sell=True, note=band)
        return out

    def manage(self, ctx: Context) -> list[PendingSignal]:
        """ETF 无日常卖出管理(E2 减仓在 entry 月度检查里生成)。"""
        return []

    # ---------------- 仓位调整信号生成 ----------------
    def _adjust_to_target(self, ctx: Context, symbol: str, target_pct: float,
                          rule: str, today: str, equity: float,
                          pending_keys: set, allow_sell: bool,
                          note: str) -> list[PendingSignal]:
        out = []
        if (symbol, rule) in pending_keys:
            return out
        pos = ctx.positions.get(symbol)
        price = ctx.price(symbol)
        cur_pct = (pos.qty * price / equity) if (pos and price > 0) else 0.0
        gap = target_pct - cur_pct
        if abs(gap) <= self.drift:
            return out
        if gap > 0:
            out.append(PendingSignal(
                symbol=symbol, side=Side.BUY, signal_date=today,
                strategy=self.name, rule=rule, size_hint_pct=round(gap, 4),
                reason=f"{rule} {note}:现{cur_pct:.1%} -> 目标{target_pct:.1%},"
                       f"补买{gap:.1%}净值"))
        elif allow_sell and pos is not None:
            sell_ratio = min(1.0, -gap / cur_pct) if cur_pct > 0 else 1.0
            out.append(PendingSignal(
                symbol=symbol, side=Side.SELL, signal_date=today,
                strategy=self.name, rule=rule, sell_ratio=round(sell_ratio, 4),
                reason=f"{rule} {note}:现{cur_pct:.1%} -> 目标{target_pct:.1%},"
                       f"减仓{sell_ratio:.0%}"))
        for s in out:
            log.info("信号 %s", s.reason)
        return out

    # ---------------- E3:个股买入缺钱时,先卖宽基补现金(engine 调用) ----------------
    def fund_need_from_broad(self) -> str | None:
        """返回宽基ETF代码(供引擎在现金不足时优先变现)。"""
        return self.broad_symbol if self.enabled else None
