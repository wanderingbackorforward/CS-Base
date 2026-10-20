"""调度引擎(投资系统准则 v3,X1 模式):收盘后算信号,次日开盘撮合。

节奏(北京时间,交易日):
- 09:20  开盘准备:T+1 解锁可卖、持有天数+1、当日盈亏基准、熔断清仓
- 09:25  执行昨日 pending 信号(卖单优先;F4 跌停挂起次日再试 / X4 涨停顺延;
         买入缺现金先卖停泊货基变现;E5 追涨限价不超昨收102%)
- 盘中各时点(mark_times)仅记录净值曲线,不产生任何交易决策
- 15:10  收盘后信号周期:组装 Context(注入沪深300指数历史供 Z2 判定)
         -> v3 三层策略(底层L/交易层E/储备Z)产出 PendingSignal 入库
         -> 旧策略持仓清退(legacy)-> 三层占比核对 -> 导出快照
- 15:35  盘后结算:净值、回撤熔断(默认禁用)、快照

v3 三层结构:底层仓只持有不择时;交易层 E1-E7 按σ自动缩放;
大底储备 Z1-Z4 只在大熊市动用,平时停泊货基。

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

        # v3 三层结构:A股 底层40% / 交易层45% / 储备15%
        alloc = cfg["allocation"]
        self.base_pct = float(alloc.get("base_pct", 0.40))
        self.trade_pct = float(alloc.get("trade_pct", 0.45))
        self.reserve_pct = float(alloc.get("reserve_pct", 0.15))
        self.rebalance_drift = float(alloc.get("rebalance_drift", 0.05))

        b = cfg["broker"]["paper"]
        self.limit_up_pct = float(b.get("limit_up_pct", 0.098))
        self.limit_down_pct = float(b.get("limit_down_pct", -0.098))

        self.capital = float(cfg["app"].get("initial_capital", 10000))
        # paper=纸面自动撮合(模拟盘) / live=实盘(券商API自动下单,缺密钥降级人工通道)
        self.mode = str(cfg["app"].get("mode", "paper")).lower()
        if self.mode not in ("paper", "live"):
            self.mode = "paper"
        # A股券商API(QMT/miniQMT)接入位:
        # Windows + 券商开通 miniQMT + pip install xtquant 后,在此构造
        # QmtBroker 替换 PaperBroker 并置 live_api=True,撮合分支零改动。
        # 未接入时 live 模式走人工通道(信号->你在券商App下单->fill 回填记账)。
        self.live_api = False
        acc = self.store.load_account()
        if acc is None:
            acc = Account(cash=self.capital, day_start_equity=self.capital,
                          peak_equity=self.capital)
            self.store.save_account(acc)
            log.info("新账户初始化,本金 %.0f", self.capital)
        self.account = acc
        # v3 三层策略实例引用(停泊变现/限价约束等引擎侧逻辑用)
        self._base_layer = next((s for s in self.strategies
                                if s.name == "base_layer"), None)
        self._trade = next((s for s in self.strategies
                            if s.name == "trade_engine"), None)
        self._reserve = next((s for s in self.strategies
                              if s.name == "reserve"), None)

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
        # v3:报价覆盖持仓 + 池子 + 储备标的(指数ETF/停泊货基)
        reserve_syms = []
        if self._reserve is not None:
            reserve_syms = [self._reserve.buy_symbol, self._reserve.park_symbol]
        quote = self._quotes_for(list(positions) + pool_syms + reserve_syms)
        mv = self._market_value(positions, quote=quote, spot=spot)

        need_hist = list(dict.fromkeys(list(positions) + pool_syms
                                       + reserve_syms))
        start = (today - timedelta(days=420)).isoformat()  # 覆盖252个交易日
        hist: dict[str, pd.DataFrame] = {}
        for s in need_hist:
            try:
                h = self.feed.hist(s, start, today.isoformat())
                if h is not None and not h.empty:
                    hist[s] = h
            except Exception as e:  # noqa: BLE001
                log.debug("hist %s 失败: %s", s, e)
        # Z2:注入沪深300指数历史(近3年最高收盘判定,指数走 regime_hist 通道)
        if self._reserve is not None:
            idx = self._reserve.track_symbol
            try:
                istart = (today - timedelta(days=1150)).isoformat()
                h = self.feed.regime_hist(idx, istart, today.isoformat())
                if h is not None and len(h):
                    hist[idx] = h
            except Exception as e:  # noqa: BLE001
                log.warning("Z2 指数历史(%s)不可用,本周期不判定大底: %s", idx, e)

        # 财报前不开新仓(E3):A股 3 个交易日
        blackout: set[str] = set()
        try:
            gap = int(self.cfg["trade"].get("earnings_gap_days", 3))
            watch = [s for s, p in positions.items()
                     if p.strategy == "trade_engine"] + pool_syms
            blackout = self.pool_manager.earnings_blackout(watch, gap_days=gap,
                                                           asof=today)
        except Exception:  # noqa: BLE001
            log.warning("财报黑盒计算失败,本周期跳过该检查")

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

        # L3/R1:三层占比核对(偏离>5pp 醒目提示)
        self._allocation_notes(ctx)

        mv = self._market_value(positions, quote=quote, spot=spot)
        self.store.append_equity(t0, self.account.equity(mv),
                                 self.account.cash, mv)
        snap = self.store.export_snapshot()
        log.info("信号周期完成:新增 %d 条信号,待执行 %d 条,净值 %.0f,快照 %s",
                 new_count, len(self.store.pending_signals()),
                 self.account.equity(mv), snap)

    def _allocation_notes(self, ctx: Context) -> None:
        """L3:报告三层占比与目标偏离;年度再平衡日打醒目日志。"""
        equity = ctx.equity()
        if equity <= 0:
            return
        a = self.allocation()
        rows = [
            ("底层仓", a["base_deployed_pct"], self.base_pct),
            ("交易层", a["trade_pct"], self.trade_pct),
            ("储备", a["reserve_pct"], self.reserve_pct),
        ]
        if ctx.rebalance_day:
            log.info("L3 年度再平衡日:三层 底层%.0f%% / 交易层%.0f%% / 储备%.0f%%"
                     "(目标 %.0f%%/%.0f%%/%.0f%%;底层仓偏离>5pp 才调整)",
                     a["base_deployed_pct"] * 100, a["trade_pct"] * 100,
                     a["reserve_pct"] * 100, self.base_pct * 100,
                     self.trade_pct * 100, self.reserve_pct * 100)
        for n, cur, tgt in rows:
            if abs(cur - tgt) > self.rebalance_drift:
                log.warning("L3:%s 现 %.0f%% / 目标 %.0f%%,偏离超 %.0fpp;"
                            "底层仓由 L3 调仓信号收敛;交易层靠 E 档位自然回归;"
                            "储备只在大熊市动用,平时停泊不算偏离",
                            n, cur * 100, tgt * 100, self.rebalance_drift * 100)
        # F3 熔断状态提示(交易层)
        if self._trade is not None and self._trade.is_halted(ctx.today):
            log.warning("F3 交易层熔断中:%s", self._trade.halt_note())

    # ---------------- 三层占比(v3 结构:持有与交易彻底分开) ----------------
    def allocation(self) -> dict:
        """三层占比:底层/交易层/储备。停泊(货基)与现金先归属储备
        (15%额度),超出部分记交易层待命(交易层没用到的钱也停泊在那里)。"""
        positions = self.store.all_positions()
        quote = self._quotes_for(list(positions))
        mv_total = self._market_value(positions, quote=quote)
        equity = self.account.equity(mv_total)
        eq = equity if equity > 0 else 1.0
        park_sym = self._reserve.park_symbol if self._reserve else ""
        base_mv = trade_mv = res_mv = park_mv = 0.0
        for s, p in positions.items():
            px = 0.0
            if len(quote):
                r = quote[quote["symbol"] == s]
                if len(r):
                    px = float(r.iloc[0]["price"])
            px = px or p.avg_cost
            v = p.qty * px
            if p.strategy == "base_layer":
                base_mv += v
            elif p.strategy == "trade_engine":
                trade_mv += v
            elif s == park_sym:
                park_mv += v
            else:
                res_mv += v
        idle = park_mv + self.account.cash
        res_target = self.reserve_pct * eq
        reserve_total = res_mv + min(idle, res_target)
        trade_wait = max(0.0, idle - res_target)
        return {
            "base_pct": base_mv / eq,
            "trade_pct": (trade_mv + trade_wait) / eq,
            "reserve_pct": reserve_total / eq,
            "base_deployed_pct": base_mv / eq,
            "trade_deployed_pct": trade_mv / eq,
            "reserve_deployed_pct": res_mv / eq,
            "park_pct": park_mv / eq,
            "cash_pct": self.account.cash / eq,
            "equity": equity,
        }

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

        # ---- 过期作废:非风险信号超过 stale 天仍未成交,释放档位重新评估
        #      (F2 硬失效除外:F4 要求跌停也逐日挂单直到成交) ----
        age = self._signal_age_days(sig, today)
        if age is not None and age >= self.stale_signal_days \
                and not sig.rule.startswith("F2"):
            _finish("CANCELLED", f"挂起{age}个交易日未成交,作废重评(档位已释放)",
                    ok=False)
            log.info("信号作废 %s %s(%s)", sig.rule, sig.symbol, sig.signal_id)
            return

        def _timeout(order) -> None:
            # 券商单已提交但回报超时:不猜成交与否、绝不自动补单;
            # 信号关闭并注明券商单号,人工核实后 fill 补记 / skip 释放档位
            sig.status = "CANCELLED"
            sig.note = (f"券商单回报超时,勿重复下单!到券商App核实:"
                        f"已成交则 fill {sig.signal_id} 补记,未成交则跳过")
            sig.executed_date = today.isoformat()
            self.store.save_signal(sig)
            log.warning("券商单超时,信号转人工核实 %s %s(%s)",
                        sig.rule, sig.symbol, order.reason)

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
            elif o.status.value == "PENDING":
                _timeout(o)
            else:
                _finish("FAILED", f"拒单:{o.reason}", ok=False)
            return

        # ---- 买入 ----
        # X4:开盘涨停,顺延下一交易日,不追涨停(F5:不追涨停)
        if open_pct >= self.limit_up_pct:
            sig.note = f"X4 开盘涨停({open_pct:+.1%}),顺延下一交易日"
            self.store.save_signal(sig)
            log.info("X4 %s 涨停,买入顺延", sig.symbol)
            return
        # E5 追涨限价约束:次日限价不超过前收盘价的102%,开盘更高则顺延
        if sig.rule == "E5" and self._trade is not None \
                and self._trade.chase_limit_premium > 0 and preclose > 0 \
                and price > preclose * (1 + self._trade.chase_limit_premium):
            sig.note = (f"E5 限价约束:开盘{price:.2f}超昨收{preclose:.2f}的"
                        f"{self._trade.chase_limit_premium:.0%},限价单挂不到,"
                        f"顺延下一交易日")
            self.store.save_signal(sig)
            log.info("E5 %s 限价约束,买入顺延", sig.symbol)
            return
        mv = self._market_value(positions, quote=None, spot=None)
        equity = self.account.equity(mv)
        budget = sig.size_hint_pct * equity
        # 现金不够这份预算 -> 先卖停泊货基变现(Z4:停泊资金买入时回巢)
        if self.account.cash < budget * 1.002:
            self._fund_from_park(budget * 1.002 - self.account.cash,
                                 positions, qrow, sig)
        budget = min(budget, self.account.cash)
        lot = 100
        qty = int(budget // (price * 1.002))
        qty -= qty % lot
        if qty < lot:
            need = lot * price * 1.002
            _finish("FAILED",
                    f"每份预算{budget:.0f}元不足一手(约需{need:.0f}元,股价{price:.2f});"
                    f"1万本金下高价股按5%单只上限分3份每份约167元,买不进一手属正常,"
                    f"考虑追加本金或按D1调整参数",
                    ok=False)
            log.warning("%s %s 买入失败:份额预算不足一手(本金适配问题)",
                        sig.rule, sig.symbol)
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
        elif o.status.value == "PENDING":
            _timeout(o)
        else:
            _finish("FAILED", f"拒单:{o.reason}", ok=False)

    # ---------------- 买入缺钱:先卖停泊货基变现(Z4 停泊资金回巢) ----------------
    def _fund_from_park(self, need: float, positions: dict, qrow: dict,
                        sig: PendingSignal) -> bool:
        """现金不足时先卖停泊货基(SH511990)。底层仓/储备指数ETF绝不动——
        只动停泊仓:它本来就是"等待中的钱"。返回是否成功补足。"""
        if need <= 0 or self._reserve is None:
            return need <= 0
        sym = self._reserve.park_symbol
        pos = positions.get(sym)
        if pos is None or pos.available_qty <= 0:
            return False
        row = qrow.get(sym)
        px = float(row["price"]) if row is not None else 0.0
        if px <= 0:
            px = pos.avg_cost
        if px <= 0:
            return False
        qty = int(need / px // 100 + 1) * 100
        qty = min(qty - qty % 100, pos.available_qty)
        if qty <= 0:
            return False
        o = Order(symbol=sym, side=Side.SELL, qty=qty, ts=self.now(),
                  strategy=self._reserve.name,
                  reason=f"Z4 | {sig.rule}买{sig.symbol}缺现金{need:.0f}元,"
                         f"先变现停泊货基{qty}份")
        o = self.broker.execute(o, px, self.account, positions)
        if o.status.value == "FILLED":
            log.info("Z4 变现 %s x%d @%.3f(补 %s 买入现金)", sym, o.filled_qty,
                     o.filled_price, sig.symbol)
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
