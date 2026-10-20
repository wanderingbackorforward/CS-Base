"""调度引擎(准则 v2.0 X1 模式):收盘后算信号,次日开盘撮合。

节奏(北京时间,交易日):
- 09:20  开盘准备:T+1 解锁可卖、持有天数+1、当日盈亏基准、熔断清仓、R1 判定
- 09:25  执行昨日 pending 信号(卖单优先;X3 跌停挂起次日再试 / X4 涨停顺延;
         G3 买入按开盘价定股数,E3 缺钱先卖宽基ETF,R2 现金池保护)
- 盘中各时点(mark_times)仅记录净值曲线,不产生任何交易决策
- 15:10  收盘后信号周期:组装 Context -> 策略产出 PendingSignal 入库(G10 风险
         事件当晚即可在看板/日志看到) -> 旧策略持仓清退 -> 导出快照
- 15:35  盘后结算:净值、单日亏冷却、回撤熔断、快照

鲁棒性:tick 里任何异常只记日志,绝不退出——量化系统跑不满一个月,
多半不是策略烂,是进程没扛住一次网络抖动。
"""
from __future__ import annotations

import logging
import time as _time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .broker import PaperBroker
from .config import Section
from .datafeed import DataFeed
from .models import Account, Order, PendingSignal, Side
from .pool import PoolManager
from .risk import RiskManager
from .store import Store
from .strategies import build_strategies
from .strategies.base import Context

log = logging.getLogger("qtrader.engine")

EMPTY_DF = pd.DataFrame()


class Engine:
    def __init__(self, cfg: Section) -> None:
        self.cfg = cfg
        self.state_dir = Path(cfg["app"].get("state_dir", "./data"))
        self.store = Store(self.state_dir)
        self.feed = DataFeed(cfg, self.store.dir)
        self.pool_manager = PoolManager(cfg, self.feed)
        self.risk = RiskManager(cfg, self.store)
        self.broker = PaperBroker(cfg, self.store)
        self.strategies = build_strategies(cfg, self.pool_manager, self.store)
        self.active_names = {s.name for s in self.strategies}

        sch = cfg["schedule"]
        self.tick_seconds = int(sch.get("tick_seconds", 30))
        self.promote_time = str(sch.get("promote_time", "09:20"))
        self.execute_time = str(sch.get("execute_time", "09:25"))
        self.signal_time = str(sch.get("signal_time", "15:10"))
        self.eod_time = str(sch.get("eod_time", "15:35"))
        self.mark_times = [str(x) for x in sch.get("mark_times", ["10:30", "14:00"])]
        self.rebalance_month = int(sch.get("rebalance_month", 1))
        self.stale_signal_days = int(sch.get("stale_signal_days", 5))

        alloc = cfg["allocation"]
        self.stocks_pct = float(alloc.get("stocks_pct", 0.60))
        self.etf_pct = float(alloc.get("etf_pct", 0.20))
        self.cash_pct = float(alloc.get("cash_pct", 0.20))
        self.rebalance_drift = float(alloc.get("rebalance_drift", 0.05))
        self.cash_floor = float(alloc.get("cash_floor", 0.05))

        b = cfg["broker"]["paper"]
        self.limit_up_pct = float(b.get("limit_up_pct", 0.098))
        self.limit_down_pct = float(b.get("limit_down_pct", -0.098))

        self.capital = float(cfg["app"].get("initial_capital", 10000))
        # paper=纸面自动撮合(模拟盘) / live=实盘人工通道(下单接口,成交回填记账)
        self.mode = str(cfg["app"].get("mode", "paper")).lower()
        if self.mode not in ("paper", "live"):
            self.mode = "paper"
        acc = self.store.load_account()
        if acc is None:
            acc = Account(cash=self.capital, day_start_equity=self.capital,
                          peak_equity=self.capital)
            self.store.save_account(acc)
            log.info("新账户初始化,本金 %.0f", self.capital)
        self.account = acc
        # ETF 压舱仓实例引用(E3 融资来源)
        self._etf_core = next((s for s in self.strategies if s.name == "etf_core"), None)

    # ---------------- 时间工具 ----------------
    def now(self) -> datetime:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Shanghai"))

    def _hhmm(self, t: str) -> int:
        h, m = t.split(":")
        return int(h) * 60 + int(m)

    def _job_done(self, name: str) -> bool:
        return bool(self.store.get_meta(f"job:{name}:{date.today().isoformat()}"))

    def _mark_job(self, name: str) -> None:
        self.store.set_meta(f"job:{name}:{date.today().isoformat()}", True)

    def _trading_dates(self) -> list[date]:
        try:
            return sorted(self.feed.trade_dates())
        except Exception:  # noqa: BLE001
            return []

    def _is_month_end(self, d: date) -> bool:
        """d 是否当月最后一个交易日(E2 月度检查日)。"""
        dates = self._trading_dates()
        after = [x for x in dates if x >= d]
        if not after:
            return d.weekday() == 4  # 日历拿不到:近似按周五
        nxt = after[0]
        return nxt.month != d.month

    def _is_rebalance_day(self, d: date) -> bool:
        """R1:年度再平衡日 = 再平衡月的第一个交易日。"""
        if d.month != self.rebalance_month:
            return False
        dates = [x for x in self._trading_dates() if x.year == d.year
                 and x.month == self.rebalance_month]
        return bool(dates) and d == dates[0]

    # ---------------- 主循环 ----------------
    def serve_forever(self) -> None:
        log.info("引擎启动(X1模式):信号 %s / 撮合 %s / 结算 %s",
                 self.signal_time, self.execute_time, self.eod_time)
        while True:
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                log.exception("tick 异常(已吞掉,继续循环)")
            _time.sleep(self.tick_seconds)

    def tick(self) -> None:
        now = self.now()
        if not self.feed.is_trading_day(now.date()):
            return
        hhmm = now.hour * 60 + now.minute
        if hhmm >= self._hhmm(self.promote_time) and not self._job_done("promote"):
            self._promote()
            self._mark_job("promote")
        if hhmm >= self._hhmm(self.execute_time) and not self._job_done("execute"):
            self._execute_cycle()
            self._mark_job("execute")
        for mt in self.mark_times:
            if hhmm >= self._hhmm(mt) and not self._job_done(f"mark{mt}"):
                self._mark_cycle()
                self._mark_job(f"mark{mt}")
                break
        if hhmm >= self._hhmm(self.signal_time) and not self._job_done("signal"):
            self.signal_cycle()
            self._mark_job("signal")
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
        log.info("开盘准备:T+1解锁完成,今日盈亏基准 %.0f,净值 %.0f",
                 self.account.day_start_equity, self.account.equity(mv))
        self._flatten_if_killed(positions)

    # ---------------- 市值/报价 ----------------
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
            for _, r in spot[spot["symbol"].isin(syms)].iterrows():
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

    # ---------------- 池子闸门(实盘权限对齐:1万元无北交所/科创板/创业板权限) ----------------
    def _gate_pool(self, pool: list, name_map: dict) -> list:
        a = self.cfg["market"]["a_share"]
        out = []
        for e in pool:
            sym = str(e["symbol"])
            name = str(name_map.get(sym, e.get("name", "")))
            if a.get("exclude_bj", True) and sym.startswith("BJ"):
                continue
            if a.get("exclude_star", True) and sym.startswith("SH688"):
                continue
            if not a.get("include_chinext", False) and sym.startswith("SZ30"):
                continue
            if a.get("exclude_st", True) and ("ST" in name.upper() or "退" in name):
                continue
            out.append(e)
        dropped = len(pool) - len(out)
        if dropped:
            log.info("市场闸门:池子 %d 只中 %d 只被过滤(无权限/ST/板块)",
                     len(pool), dropped)
        return out

    # ---------------- 盘中盯市:仅记净值 ----------------
    def _mark_cycle(self) -> None:
        positions = self.store.all_positions()
        quote = self._quotes_for(list(positions))
        mv = self._market_value(positions, quote=quote)
        self.store.append_equity(self.now(), self.account.equity(mv),
                                 self.account.cash, mv)

    # ---------------- 收盘后信号周期(X1 核心) ----------------
    def signal_cycle(self) -> None:
        t0 = self.now()
        today = t0.date()
        log.info("========== 收盘后信号周期 %s ==========", today.isoformat())
        pool: list = []
        try:
            pool = self.pool_manager.load_pool()
        except Exception:  # noqa: BLE001
            log.exception("池子加载失败,本周期不产生个股信号")

        spot = EMPTY_DF
        try:
            spot = self.feed.spot_all(force=True)
        except Exception as e:  # noqa: BLE001
            log.error("全A快照不可用: %s", e)
        name_map: dict[str, str] = {}
        if len(spot):
            for _, r in spot.iterrows():
                name_map[str(r["symbol"])] = str(r["name"])
        pool = self._gate_pool(pool, name_map)
        pool_syms = [str(e["symbol"]) for e in pool]

        positions = self.store.all_positions()
        etf_syms = []
        if self._etf_core is not None:
            etf_syms = [self._etf_core.div_symbol, self._etf_core.broad_symbol]
        quote = self._quotes_for(list(positions) + pool_syms + etf_syms)
        mv = self._market_value(positions, quote=quote, spot=spot)

        need_hist = list(dict.fromkeys(list(positions) + pool_syms))
        start = (today - timedelta(days=420)).isoformat()  # 覆盖252个交易日
        hist: dict[str, pd.DataFrame] = {}
        for s in need_hist:
            try:
                h = self.feed.hist(s, start, today.isoformat())
                if h is not None and not h.empty:
                    hist[s] = h
            except Exception as e:  # noqa: BLE001
                log.debug("hist %s 失败: %s", s, e)

        blackout: set[str] = set()
        try:
            gap = int(self.cfg["pool"].get("earnings_gap_days", 3))
            watch = [s for s, p in positions.items()
                     if p.strategy == "value_dip"] + pool_syms
            blackout = self.pool_manager.earnings_blackout(watch, gap_days=gap,
                                                           asof=today)
        except Exception:  # noqa: BLE001
            log.warning("G5 财报黑盒计算失败,本周期跳过该检查")

        pending = self.store.pending_signals()
        ctx = Context(
            today=today.isoformat(), spot=spot, quote=quote, hist=hist,
            account=self.account, positions=positions, market_value=mv,
            pool=pool, pending=pending, pool_val_q={}, blackout=blackout,
            name_map=name_map,
            industry_map={str(e["symbol"]): str(e.get("industry", "")) for e in pool},
            month_end=self._is_month_end(today),
            rebalance_day=self._is_rebalance_day(today),
        )

        new_count = 0
        for st in self.strategies:
            try:
                sigs = st.entry_signals(ctx) + st.manage(ctx)
            except Exception:  # noqa: BLE001
                log.exception("策略 %s 信号异常", st.name)
                sigs = []
            for sig in sigs:
                try:
                    self.store.save_signal(sig)
                    ctx.pending.append(sig)
                    new_count += 1
                    log.info("信号入库 %s %s %s | %s", sig.rule, sig.side,
                             sig.symbol, sig.reason)
                except Exception:  # noqa: BLE001
                    log.exception("信号保存失败 %s", sig.signal_id)

        # 旧策略持仓清退(切换准则 v2.0 后的存量,规则编号 legacy)
        for sym, p in positions.items():
            if p.strategy in self.active_names:
                continue
            if any(s.symbol == sym for s in ctx.pending):
                continue
            sig = PendingSignal(
                symbol=sym, side=Side.SELL, signal_date=today.isoformat(),
                strategy=p.strategy, rule="legacy", sell_ratio=1.0,
                reason="legacy:旧策略停用,存量持仓按准则切换清退")
            self.store.save_signal(sig)
            ctx.pending.append(sig)
            new_count += 1
            log.info("信号入库 legacy SELL %s(旧策略 %s 存量清退)", sym, p.strategy)

        # R1 再平衡检查:偏离 >5pp 的仓位给提示(ETF 由 etf_core 调,个股靠 G2 上限
        # 与卖出规则自然回归,准则未授权强制交易个股)
        self._rebalance_notes(ctx)

        mv = self._market_value(positions, quote=quote, spot=spot)
        self.store.append_equity(t0, self.account.equity(mv),
                                 self.account.cash, mv)
        snap = self.store.export_snapshot()
        log.info("信号周期完成:新增 %d 条信号,待执行 %d 条,净值 %.0f,快照 %s",
                 new_count, len(self.store.pending_signals()),
                 self.account.equity(mv), snap)

    def _rebalance_notes(self, ctx: Context) -> None:
        """R1:报告三仓占比与目标偏离;年度再平衡日打醒目日志。"""
        equity = ctx.equity()
        if equity <= 0:
            return
        stocks_mv = sum(p.qty * ctx.price(s) for s, p in ctx.positions.items()
                        if p.strategy == "value_dip")
        etf_mv = sum(p.qty * ctx.price(s) for s, p in ctx.positions.items()
                     if p.strategy == "etf_core")
        cash_pct = self.account.cash / equity
        rows = [
            ("个股", stocks_mv / equity, self.stocks_pct),
            ("ETF", etf_mv / equity, self.etf_pct),
            ("现金", cash_pct, self.cash_pct),
        ]
        drifts = [(n, cur, tgt, cur - tgt) for n, cur, tgt in rows
                  if abs(cur - tgt) > self.rebalance_drift]
        if ctx.rebalance_day:
            log.info("R1 年度再平衡日:当前 个股%.0f%% / ETF%.0f%% / 现金%.0f%%"
                     "(目标 60/20/20)", *(r[1] * 100 for r in rows))
        for n, cur, tgt, d in drifts:
            log.warning("R1:%s仓偏离目标 %.0f%%(现 %.0f%% / 目标 %.0f%%)"
                        "%s", n, abs(d) * 100, cur * 100, tgt * 100,
                        " -> 由 etf_core 生成调仓信号" if n == "ETF"
                        else " -> 个股靠G2上限与卖出规则自然回归;现金待机会")
        # R2:现金池低于下限,提示卖出回款优先留存现金
        if cash_pct < self.cash_floor:
            log.warning("R2:现金池 %.0f%% 低于 %.0f%%,下一笔卖出回款优先补足现金池,"
                        "暂缓新买入", cash_pct * 100, self.cash_floor * 100)

    # ---------------- 次日开盘撮合周期(X1 执行侧) ----------------
    def _execute_cycle(self) -> None:
        signals = self.store.pending_signals()
        if not signals:
            return
        t0 = self.now()
        today = t0.date()
        if self.mode == "live":
            # 实盘模式:系统不下单——信号转人工订单发提醒,你在券商App
            # 下单成交后回填(fill 命令/看板),系统按真实资金流记账。
            from .fill import submit_manual
            positions = self.store.all_positions()
            quote = self._quotes_for(list({s.symbol for s in signals}))
            submit_manual(self, signals, positions, quote, ts=t0)
            return
        log.info("========== 开盘撮合 %s:%d 条待执行信号 ==========",
                 today.isoformat(), len(signals))
        positions = self.store.all_positions()
        quote = self._quotes_for(list({s.symbol for s in signals}))
        if quote.empty:
            try:
                quote = self.feed.spot_all()
            except Exception:  # noqa: BLE001
                log.error("撮合取消:开盘报价不可用,信号保留明日重试")
                return
        qrow = {str(r["symbol"]): r for _, r in quote.iterrows()}

        sells = [s for s in signals if s.side == Side.SELL]
        buys = [s for s in signals if s.side == Side.BUY]
        for sig in sells + buys:  # 先卖后买:腾出现金再买
            try:
                self._execute_one(sig, qrow, positions, today)
            except Exception:  # noqa: BLE001
                log.exception("信号 %s 执行异常(保留重试)", sig.signal_id)

        mv = self._market_value(positions, quote=quote)
        self.store.append_equity(t0, self.account.equity(mv),
                                 self.account.cash, mv)
        self.store.export_snapshot()
        log.info("撮合完成:净值 %.0f / 现金 %.0f / 市值 %.0f",
                 self.account.equity(mv), self.account.cash, mv)

    # ---------------- 单信号执行 ----------------
    def _execute_one(self, sig: PendingSignal, qrow: dict,
                     positions: dict, today: date) -> None:
        def _strategy_of(name: str):
            return next((s for s in self.strategies if s.name == name), None)

        def _finish(status: str, note: str, ok: bool, filled_qty: int = 0) -> None:
            sig.status = status
            sig.note = note
            sig.executed_date = today.isoformat()
            self.store.save_signal(sig)
            st = _strategy_of(sig.strategy)
            if st is not None:
                st.on_fill(sig, filled_qty, ok)

        # ---- 过期作废:非风险信号超过 stale 天仍未成交,释放档位重新评估 ----
        age = self._signal_age_days(sig, today)
        if age is not None and age >= self.stale_signal_days \
                and sig.rule not in ("G10", "G8"):
            _finish("CANCELLED", f"挂起{age}个交易日未成交,作废重评(档位已释放)",
                    ok=False)
            log.info("信号作废 %s %s(%s)", sig.rule, sig.symbol, sig.signal_id)
            return

        row = qrow.get(sig.symbol)
        if row is None:
            sig.note = "开盘报价缺失,保留明日重试"
            self.store.save_signal(sig)
            return
        price = float(row["price"])
        preclose = float(row["preclose"]) if row["preclose"] else 0.0
        if price <= 0:
            sig.note = "开盘价无效,保留明日重试"
            self.store.save_signal(sig)
            return
        open_pct = price / preclose - 1 if preclose > 0 else 0.0

        if sig.side == Side.SELL:
            # ---- X3:跌停挂起,次日集合竞价再挂,直到成交 ----
            if open_pct <= self.limit_down_pct:
                sig.note = f"X3 跌停({open_pct:+.1%})挂起,明日开盘再试"
                self.store.save_signal(sig)
                log.warning("X3 %s 跌停,卖单挂起: %s", sig.symbol, sig.reason)
                return
            pos = positions.get(sig.symbol)
            if pos is None:
                _finish("CANCELLED", "已无持仓,信号取消", ok=True)
                return
            if sig.sell_ratio >= 0.999:
                qty = pos.qty          # 清仓类:允许零股一次性卖出
            else:
                qty = int(pos.qty * sig.sell_ratio)
                if sig.qty > 0:
                    qty = min(qty, sig.qty)
                qty -= qty % 100
            qty = min(qty, pos.available_qty)
            if qty <= 0:
                _finish("FAILED", "T+1 可卖为0(明日解锁后请重跑 exec)", ok=False)
                return
            o = Order(symbol=sig.symbol, side=Side.SELL, qty=qty, ts=self.now(),
                     strategy=sig.strategy, reason=f"{sig.rule} | {sig.reason}")
            o = self.broker.execute(o, price, self.account, positions)
            if o.status.value == "FILLED":
                _finish("EXECUTED", f"开盘价{price:.2f}成交{o.filled_qty}股",
                        ok=True, filled_qty=o.filled_qty)
            else:
                _finish("FAILED", f"拒单:{o.reason}", ok=False)
            return

        # ---- 买入 ----
        # X4:开盘涨停,顺延下一交易日,不追涨停
        if open_pct >= self.limit_up_pct:
            sig.note = f"X4 开盘涨停({open_pct:+.1%}),顺延下一交易日"
            self.store.save_signal(sig)
            log.info("X4 %s 涨停,买入顺延", sig.symbol)
            return
        mv = self._market_value(positions, quote=None, spot=None)
        equity = self.account.equity(mv)
        budget = sig.size_hint_pct * equity
        # R2:现金池已低于下限 -> 缓买,等卖出回款补足现金池(信号保留,5日内补足仍可成交)
        if sig.strategy == "value_dip" and equity > 0 \
                and self.account.cash / equity < self.cash_floor:
            sig.note = (f"R2 现金池保护:现金{self.account.cash:.0f}已低于"
                        f"{self.cash_floor:.0%}下限,缓买等回款")
            self.store.save_signal(sig)
            log.info("R2 %s 缓买: %s", sig.symbol, sig.note)
            return
        # E3:现金不够这份预算 -> 先卖宽基ETF补上
        if self.account.cash < budget * 1.002:
            self._fund_from_broad(budget * 1.002 - self.account.cash,
                                  positions, qrow, sig)
        budget = min(budget, self.account.cash)
        lot = 100
        qty = int(budget // (price * 1.002))
        qty -= qty % lot
        if qty < lot:
            need = lot * price * 1.002
            _finish("FAILED",
                    f"每份预算{budget:.0f}元不足一手(约需{need:.0f}元,股价{price:.2f});"
                    f"1万本金下高价股无法按4%分档,考虑追加本金或按D1调整参数",
                    ok=False)
            log.warning("G3 %s 买入失败:份额预算不足一手(本金适配问题)", sig.symbol)
            return
        ok, why = self.risk.check_open(self.account, positions, sig.symbol, qty,
                                       price, mv, None, asof=today)
        if not ok:
            _finish("FAILED", f"风控拦截:{why}", ok=False)
            log.info("风控拦截 %s: %s", sig.symbol, why)
            return
        o = Order(symbol=sig.symbol, side=Side.BUY, qty=qty, ts=self.now(),
                 strategy=sig.strategy, reason=f"{sig.rule} | {sig.reason}")
        o = self.broker.execute(o, price, self.account, positions)
        if o.status.value == "FILLED":
            _finish("EXECUTED", f"开盘价{price:.2f}成交{o.filled_qty}股",
                    ok=True, filled_qty=o.filled_qty)
        else:
            _finish("FAILED", f"拒单:{o.reason}", ok=False)

    # ---------------- E3:个股买入缺钱,先卖宽基ETF补现金 ----------------
    def _fund_from_broad(self, need: float, positions: dict, qrow: dict,
                         sig: PendingSignal) -> bool:
        """现金不足时按 E3 先变现宽基ETF。返回是否成功补足。"""
        if need <= 0 or self._etf_core is None:
            return need <= 0
        sym = self._etf_core.fund_need_from_broad()
        pos = positions.get(sym) if sym else None
        if pos is None or pos.available_qty <= 0:
            return False
        row = qrow.get(sym)
        px = float(row["price"]) if row is not None else 0.0
        if px <= 0:
            return False
        qty = int(need / px // 100 + 1) * 100
        qty = min(qty - qty % 100, pos.available_qty)
        if qty <= 0:
            return False
        o = Order(symbol=sym, side=Side.SELL, qty=qty, ts=self.now(),
                  strategy="etf_core",
                  reason=f"E3 | 个股买入缺现金{need:.0f}元,先卖宽基ETF{qty}份")
        o = self.broker.execute(o, px, self.account, positions)
        if o.status.value == "FILLED":
            log.info("E3 变现 %s x%d @%.3f(补个股买入现金)", sym, o.filled_qty,
                     o.filled_price)
            return True
        return False

    def _signal_age_days(self, sig: PendingSignal, today: date) -> Optional[int]:
        """信号已挂起的交易日数(信号日次日计1)。"""
        try:
            d0 = date.fromisoformat(sig.signal_date)
        except ValueError:
            return None
        dates = [x for x in self._trading_dates() if d0 < x <= today]
        return len(dates)

    # ---------------- 盘后结算 ----------------
    def _eod_cycle(self) -> None:
        positions = self.store.all_positions()
        quote = self._quotes_for(list(positions))
        mv = self._market_value(positions, quote=quote)
        notes = self.risk.settle_eod(self.account, mv, asof=self.now().date())
        self.store.save_account(self.account)
        for n in notes:
            log.warning("风控: %s", n)
        self.store.append_equity(self.now(), self.account.equity(mv),
                                 self.account.cash, mv)
        self.store.export_snapshot()
        log.info("盘后结算完成:净值 %.0f / 现金 %.0f / 市值 %.0f",
                 self.account.equity(mv), self.account.cash, mv)

    # ---------------- 熔断清仓(系统安全网,绕过 X1 立即执行) ----------------
    def _flatten_if_killed(self, positions: dict,
                           quote: Optional[pd.DataFrame] = None) -> None:
        if not self.account.killed or not positions:
            return
        log.critical("熔断状态:清仓所有可卖持仓")
        if quote is None or quote.empty:
            quote = self._quotes_for(list(positions))
        qrow = {str(r["symbol"]): r for _, r in quote.iterrows()} if len(quote) else {}
        for sym, p in positions.items():
            qty = p.available_qty
            if qty <= 0:
                continue
            px = float(qrow[sym]["price"]) if sym in qrow else p.avg_cost
            o = Order(symbol=sym, side=Side.SELL, qty=qty, ts=self.now(),
                      strategy=p.strategy, reason="熔断清仓:净值回撤触及熔断线")
            self.broker.execute(o, px, self.account, positions)

    # ---------------- 运维入口 ----------------
    def run_once(self) -> None:
        """立即跑一轮收盘后信号周期(验证/手动补跑)。
        执行后打上当日 signal 标记,避免 serve 模式当天重复算信号。"""
        self.signal_cycle()
        self._mark_job("signal")

    def run_execute(self) -> None:
        """立即执行待办信号(模拟盘验证/手动补跑撮合)。
        会先做当日开盘准备(T+1 解锁),再走完整撮合流程。"""
        if not self._job_done("promote"):
            self._promote()
            self._mark_job("promote")
        self._execute_cycle()
        self._mark_job("execute")

    def reset(self) -> None:
        """人工复位熔断与冷却。"""
        self.account.killed = False
        self.store.save_account(self.account)
        self.store.set_meta("cooldown_until", "")
        log.info("已复位熔断与冷却,恢复开仓资格")
