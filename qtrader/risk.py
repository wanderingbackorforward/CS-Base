"""风控:仓位/冷却/熔断。风控不是装饰,是这个系统里唯一不许打折的模块。

规则(config 可调):
- 单票最大占比 / 总仓位上限;
- 大盘风向标(指数收于 MA20 之下)时,总仓位上限减半——趋势不对,少动;
- 单日净值亏到阈值 -> 冷却 N 天不开新仓(防上头);
- 自峰值回撤到阈值 -> 一键清仓熔断(killed),只能人工复位。
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import pandas as pd

from .config import Section
from .models import Account, Position
from .store import Store

log = logging.getLogger("qtrader.risk")


class RiskManager:
    def __init__(self, cfg: Section, store: Store) -> None:
        self.cfg = cfg
        self.store = store
        r = cfg["risk"]
        self.max_position_pct = float(r.get("max_position_pct", 0.30))
        self.max_total_exposure = float(r.get("max_total_exposure", 0.85))
        self.regime_ma = int(r.get("regime_ma", 20))
        self.regime_low_exposure = float(r.get("regime_low_exposure", 0.40))
        self.daily_loss_limit = float(r.get("daily_loss_limit_pct", -0.03))
        self.cooldown_days = int(r.get("cooldown_days", 2))
        self.kill_drawdown = float(r.get("max_drawdown_kill", -0.15))
        # 准则 v2.0:逆向分档买入体系,大盘走弱恰是买点 -> 趋势降仓默认关闭
        self.regime_enabled = bool(r.get("regime_enabled", False))

    # ---------- 仓位约束 ----------
    def exposure_cap(self, regime_close: pd.Series | None) -> float:
        """根据大盘趋势返回当前允许的总仓位上限(0~1)。"""
        if not self.regime_enabled:
            return self.max_total_exposure
        if regime_close is None or len(regime_close) < self.regime_ma:
            return min(self.max_total_exposure, 0.5)  # 数据不足,保守半仓
        ma = float(regime_close.iloc[-self.regime_ma:].mean())
        last = float(regime_close.iloc[-1])
        return self.max_total_exposure if last >= ma else self.regime_low_exposure

    def can_open(self, account: Account, positions: dict[str, Position],
                 market_value: float, asof: date | None = None) -> tuple[bool, str]:
        if account.killed:
            return False, "熔断已触发,人工复位前禁止开仓"
        today = (asof or date.today()).isoformat()
        cooldown_until = self.store.get_meta("cooldown_until")
        if cooldown_until and today <= str(cooldown_until):
            return False, f"冷却期至 {cooldown_until}(单日亏损超限)"
        if not positions:
            return True, ""
        cap = self.exposure_cap(None)
        # 注意:总仓位检查在 size 阶段用 equity 校验,这里先做持仓家数上限
        return True, ""

    def check_open(self, account: Account, positions: dict[str, Position],
                   symbol: str, qty: int, price: float, market_value: float,
                   regime_close: pd.Series | None,
                   asof: date | None = None) -> tuple[bool, str]:
        """开仓前综合校验(引擎在下单前调用)。"""
        ok, why = self.can_open(account, positions, market_value, asof)
        if not ok:
            return False, why
        equity = account.equity(market_value)
        cap = self.exposure_cap(regime_close)
        cur_exposure = market_value / equity if equity > 0 else 1.0
        add_value = qty * price
        if (cur_exposure * equity + add_value) > cap * equity:
            return False, (f"总仓位超限: 现有{cur_exposure:.0%} + 新增"
                           f"{add_value / equity:.0%} > 上限{cap:.0%}")
        same = positions.get(symbol)
        if same is not None:
            held = same.qty * price
            if (held + add_value) / equity > self.max_position_pct:
                return False, f"单票超限: {symbol} > {self.max_position_pct:.0%}"
        elif add_value / equity > self.max_position_pct:
            return False, (f"单票超限: 目标 {add_value / equity:.0%}"
                           f" > {self.max_position_pct:.0%}")
        return True, ""

    def size(self, account: Account, market_value: float, price: float,
             lot: int = 100) -> int:
        """按"单票上限与可用现金取小"计算买入数量,向下取整到整手。"""
        equity = account.equity(market_value)
        budget = min(self.max_position_pct * equity, account.cash)
        qty = int(budget // (price * 1.001))  # 1.001 预留滑点
        qty -= qty % lot
        return qty if qty >= lot else 0

    # ---------- 盘后结算:日亏冷却 / 回撤熔断 ----------
    def settle_eod(self, account: Account, market_value: float,
                   asof: date | None = None) -> list[str]:
        notes: list[str] = []
        equity = account.equity(market_value)
        account.peak_equity = max(account.peak_equity, equity)
        # 单日亏损冷却(阈值 >= 0 视为禁用:美股准则大跌恰是买点,不能因单日亏损禁买)
        if account.day_start_equity > 0 and self.daily_loss_limit < 0:
            day_pnl = equity / account.day_start_equity - 1
            if day_pnl <= self.daily_loss_limit:
                until = ((asof or date.today())
                         + timedelta(days=self.cooldown_days)).isoformat()
                self.store.set_meta("cooldown_until", until)
                notes.append(f"单日亏损 {day_pnl:.1%},冷却至 {until} 禁止开新仓")
        # 回撤熔断(阈值 >= 0 视为禁用)
        if account.peak_equity > 0 and self.kill_drawdown < 0:
            dd = equity / account.peak_equity - 1
            if dd <= self.kill_drawdown and not account.killed:
                account.killed = True
                notes.append(f"净值回撤 {dd:.1%} 触及熔断线,清仓并暂停一切新开仓;"
                             f"复盘后用 `python main.py reset` 人工复位")
                log.critical("KILL SWITCH: dd=%.1f%%", dd * 100)
        return notes
