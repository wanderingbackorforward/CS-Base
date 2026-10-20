"""调度引擎:交易日按时间点驱动「盯市 / 尾盘决策 / 盘后结算」。

节奏(北京时间,均为交易日):
- 09:20  T+1 解锁可卖、天数+1、当日盈亏基准重置
- 盘中各时点(mark_times)轻量盯市:只用持仓的实时报价跑一遍持仓管理(盘中止损)
- 14:40  主决策:热度引擎 -> 候选池 -> 策略信号 -> 风控 -> 模拟撮合
- 15:10  盘后:盯市结算、日亏冷却检查、回撤熔断检查、导出人可读快照

鲁棒性:tick 里任何异常只记日志,绝不退出——量化系统跑不满一个月,
多半不是策略烂,是进程没扛住一次网络抖动。
"""
from __future__ import annotations

import logging
import time as _time
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from .broker import PaperBroker
from .config import Section
from .datafeed import DataFeed
from .heat import HeatEngine
from .models import Account, Order, Side, split_symbol
from .risk import RiskManager
from .store import Store
from .strategies import build_strategies
from .strategies.base import Context

log = logging.getLogger("qtrader.engine")

TZ = ZoneInfo("Asia/Shanghai")
EMPTY_DF = pd.DataFrame()


class Engine:
    def __init__(self, cfg: Section) -> None:
        self.cfg = cfg
        self.state_dir = cfg["app"].get("state_dir", "./data")
        self.store = Store(self.state_dir)
        self.feed = DataFeed(cfg, self.store.dir)
        self.heat = HeatEngine(cfg, self.feed)
        self.risk = RiskManager(cfg, self.store)
        self.broker = PaperBroker(cfg, self.store)
        self.strategies = build_strategies(cfg)
        self.tick_seconds = int(cfg["schedule"].get("tick_seconds", 30))
        self.decision_time = str(cfg["schedule"].get("decision_time", "14:40"))
        self.eod_time = str(cfg["schedule"].get("eod_time", "15:10"))
        self.promote_time = str(cfg["schedule"].get("morning_promote", "09:20"))
        self.mark_times = [str(x) for x in cfg["schedule"].get(
            "mark_times", ["09:35", "10:30", "11:30", "13:30", "14:40", "15:05"])]
        self.capital = float(cfg["app"].get("initial_capital", 10000))
        self.regime_index = str(cfg["risk"].get("regime_index", "SH000001"))

        acc = self.store.load_account()
        if acc is None:
            acc = Account(cash=self.capital, day_start_equity=self.capital,
                          peak_equity=self.capital)
            self.store.save_account(acc)
            log.info("新账户初始化,本金 %.0f", self.capital)
        self.account = acc

    # ---------------- 时间工具 ----------------
    def now(self) -> datetime:
        return datetime.now(TZ)

    def _hhmm(self, t: str) -> int:
        h, m = t.split(":")
        return int(h) * 60 + int(m)

    def _job_done(self, name: str) -> bool:
        return bool(self.store.get_meta(f"job:{name}:{date.today().isoformat()}"))

    def _mark_job(self, name: str) -> None:
        self.store.set_meta(f"job:{name}:{date.today().isoformat()}", True)

    # ---------------- 主循环 ----------------
    def serve_forever(self) -> None:
        log.info("引擎启动,循环间隔 %ds,决策点 %s", self.tick_seconds, self.decision_time)
        while True:
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                log.exception("tick 异常(已吞掉,继续循环)")
            _time.sleep(self.tick_seconds)

    def tick(self) -> None:
        now = self.now()
        hhmm = now.hour * 60 + now.minute
        if not self.feed.is_trading_day(now.date()):
            return
        if hhmm >= self._hhmm(self.promote_time) and not self._job_done("promote"):
            self._promote()
            self._mark_job("promote")
        for mt in self.mark_times:
            if hhmm >= self._hhmm(mt) and not self._job_done(f"mark{mt}"):
                self._mark_cycle()
                self._mark_job(f"mark{mt}")
                break
        if hhmm >= self._hhmm(self.decision_time) and not self._job_done("decision"):
            self.decision_cycle()
            self._mark_job("decision")
        if hhmm >= self._hhmm(self.eod_time) and not self._job_done("eod"):
            self._eod_cycle()
            self._mark_job("eod")

    # ---------------- 每日开盘准备 ----------------
    def _promote(self) -> None:
        positions = self.store.all_positions()
        self.broker.promote_new_day(positions)
        for p in positions.values():
            p.days_held += 1
            self.store.save_position(p)
        mv = self._market_value(positions)
        self.account.day_start_equity = self.account.equity(mv)
        self.store.save_account(self.account)
        log.info("开盘准备:可卖解锁完成,今日盈亏基准 %.0f", self.account.day_start_equity)
        self._flatten_if_killed(positions)

    # ---------------- 市值 ----------------
    def _market_value(self, positions: dict, quote: Optional[pd.DataFrame] = None,
                      spot: Optional[pd.DataFrame] = None) -> float:
        if not positions:
            return 0.0
        syms = list(positions)
        px = {}
        if quote is not None and len(quote):
            for _, r in quote[quote["symbol"].isin(syms)].iterrows():
                px[r["symbol"]] = float(r["price"])
        if spot is not None and len(spot):
            missing = [s for s in syms if s not in px]
            if missing:
                rows = spot[spot["symbol"].isin(missing)]
                for _, r in rows.iterrows():
                    px.setdefault(r["symbol"], float(r["price"]))
        mv = 0.0
        for s, p in positions.items():
            price = px.get(s)
            if not price or price <= 0:
                price = p.avg_cost  # 数据缺失按成本计,不让净值凭空消失
            mv += p.qty * price
        return mv

    def _quotes_for(self, symbols: list[str]) -> pd.DataFrame:
        syms = [s for s in dict.fromkeys(symbols) if s]
        if not syms:
            return EMPTY_DF
        try:
            return self.feed.quote(syms)
        except Exception as e:  # noqa: BLE001
            log.warning("实时报价失败: %s", e)
            return EMPTY_DF

    # ---------------- 盘中轻量盯市 ----------------
    def _mark_cycle(self) -> None:
        positions = self.store.all_positions()
        if not positions:
            return
        quote = self._quotes_for(list(positions))
        if quote.empty:
            return
        mv = self._market_value(positions, quote=quote)
        ctx = Context(today=self.now().date().isoformat(), spot=EMPTY_DF, quote=quote,
                      hist={}, positions=positions, account=self.account,
                      market_value=mv)
        for st in self.strategies:
            for o in st.manage(ctx):
                self._execute_sell(o, positions, quote)
        mv = self._market_value(positions, quote=quote)
        self.store.append_equity(self.now(), self.account.equity(mv),
                                 self.account.cash, mv)

    # ---------------- 主决策周期 ----------------
    def decision_cycle(self) -> None:
        t0 = self.now()
        log.info("========== 尾盘决策开始 %s ==========", t0.strftime("%H:%M"))
        spot = EMPTY_DF
        try:
            spot = self.feed.spot_all(force=True)
        except Exception as e:  # noqa: BLE001
            log.error("全A快照不可用: %s", e)
        quote = EMPTY_DF
        positions = self.store.all_positions()
        quote = self._quotes_for(list(positions) + [t["symbol"] for t in
                                                    self.cfg["universe"]["dip_targets"]])
        mv = self._market_value(positions, quote=quote, spot=spot)

        regime_close: Optional[pd.Series] = None
        try:
            end = t0.date().isoformat()
            start = (t0.date() - timedelta(days=120)).isoformat()
            regime_close = self.feed.regime_hist(self.regime_index, start, end)["close"]
        except Exception as e:  # noqa: BLE001
            log.warning("大盘风向标数据不可用,仓位上限自动收紧: %s", e)

        report = self.heat.build()
        self.store.save_heat(report)
        candidates = self.heat.candidates(report)[:60]

        need_hist = list(dict.fromkeys(
            list(positions.keys())
            + [t["symbol"] for t in self.cfg["universe"]["dip_targets"]]
            + candidates))
        start = (t0.date() - timedelta(days=280)).isoformat()
        end = t0.date().isoformat()
        hist: dict[str, pd.DataFrame] = {}
        for s in need_hist:
            try:
                h = self.feed.hist(s, start, end)
                if h is not None and not h.empty:
                    hist[s] = h
            except Exception as e:  # noqa: BLE001
                log.debug("hist %s 失败: %s", s, e)

        ctx = Context(today=end, spot=spot, quote=quote, hist=hist,
                      regime_close=regime_close, account=self.account,
                      positions=positions, market_value=mv, candidates=candidates)

        for st in self.strategies:
            try:
                entries = st.entry_signals(ctx)
            except Exception:  # noqa: BLE001
                log.exception("策略 %s 入场信号异常", st.name)
                entries = []
            for o in entries:
                self._execute_entry(o, ctx)
            try:
                sells = st.manage(ctx)
            except Exception:  # noqa: BLE001
                log.exception("策略 %s 持仓管理异常", st.name)
                sells = []
            for o in sells:
                self._execute_sell(o, positions, quote, spot)

        mv = self._market_value(positions, quote=quote, spot=spot)
        self.store.append_equity(self.now(), self.account.equity(mv),
                                 self.account.cash, mv)
        snap = self.store.export_snapshot()
        log.info("决策完成,净值 %.0f,快照 %s", self.account.equity(mv), snap)

    def _execute_entry(self, o: Order, ctx: Context) -> None:
        positions = ctx.positions
        price = ctx.price(o.symbol)
        if price <= 0:
            log.warning("跳过 %s:无有效价格", o.symbol)
            return
        hint = o.size_hint_pct or float(self.cfg["risk"].get("max_position_pct", 0.30))
        equity = self.account.equity(ctx.market_value)
        budget = min(hint * equity, self.account.cash)
        lot = 100
        qty = int(budget // (price * 1.001))
        qty -= qty % lot
        if qty < lot:
            log.info("跳过 %s:预算不足一手(现金 %.0f)", o.symbol, self.account.cash)
            return
        ok, why = self.risk.check_open(self.account, positions, o.symbol, qty,
                                       price, ctx.market_value, ctx.regime_close,
                                       asof=self.now().date())
        if not ok:
            log.info("风控拦截 %s: %s", o.symbol, why)
            from .models import OrderStatus
            o.status = OrderStatus.REJECTED
            o.reason += f" | 风控拦截:{why}"
            self.store.save_order(o)
            return
        o.qty = qty
        self.broker.execute(o, price, self.account, positions)

    def _execute_sell(self, o: Order, positions: dict, quote: pd.DataFrame,
                       spot: pd.DataFrame = EMPTY_DF) -> None:
        p = positions.get(o.symbol)
        if p is None:
            return
        price = 0.0
        if len(quote):
            row = quote[quote["symbol"] == o.symbol]
            if len(row):
                price = float(row.iloc[0]["price"])
        if price <= 0 and len(spot):
            row = spot[spot["symbol"] == o.symbol]
            if len(row):
                price = float(row.iloc[0]["price"])
        if price <= 0:
            price = p.avg_cost
        o.qty = p.available_qty
        if o.qty <= 0:
            log.info("%s 想卖但 T+1 可卖为 0(明日自动再评估)", o.symbol)
            return
        self.broker.execute(o, price, self.account, positions)

    # ---------------- 盘后结算 ----------------
    def _eod_cycle(self) -> None:
        positions = self.store.all_positions()
        quote = self._quotes_for(list(positions))
        spot = EMPTY_DF
        try:
            spot = self.feed.spot_all()
        except Exception:  # noqa: BLE001
            pass
        self._flatten_if_killed(positions, quote=quote, spot=spot)
        mv = self._market_value(positions, quote=quote, spot=spot)
        notes = self.risk.settle_eod(self.account, mv, asof=self.now().date())
        self.store.save_account(self.account)
        for n in notes:
            log.warning("风控: %s", n)
        self.store.append_equity(self.now(), self.account.equity(mv),
                                 self.account.cash, mv)
        self.store.export_snapshot()
        log.info("盘后结算完成:净值 %.0f / 现金 %.0f / 市值 %.0f",
                 self.account.equity(mv), self.account.cash, mv)

    def _flatten_if_killed(self, positions: dict, quote: pd.DataFrame = EMPTY_DF,
                           spot: pd.DataFrame = EMPTY_DF) -> None:
        """熔断或冷却触发后:全部持仓清仓(尊重 T+1 可卖约束)。"""
        if not self.account.killed or not positions:
            return
        log.critical("熔断状态:清仓所有可卖持仓")
        for sym, p in positions.items():
            o = Order(symbol=sym, side=Side.SELL, qty=p.available_qty,
                      ts=self.now(), strategy=p.strategy,
                      reason="熔断清仓:净值回撤触及熔断线")
            self._execute_sell(o, positions, quote, spot)

    # ---------------- 运维入口 ----------------
    def run_once(self) -> None:
        """立即执行一轮完整决策(不限时间,用于首次验证/手动补跑)。
        执行后打上当日 decision 标记,避免随后的 serve 模式当天重复决策。"""
        self.decision_cycle()
        self._mark_job("decision")

    def reset(self) -> None:
        """人工复位熔断与冷却。"""
        self.account.killed = False
        self.store.save_account(self.account)
        self.store.set_meta("cooldown_until", "")
        log.info("已复位熔断与冷却,恢复开仓资格")
