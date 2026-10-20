"""美股调度引擎(美股投资准则 v1.1,X1 美股版):收盘后算信号,次日开盘撮合。

与 A股引擎(engine.py)的关键差异:
- 独立账户/独立状态目录(./data_us),美元计价,与A股账户分开管理;
- 调度为北京时间:06:10 算信号(美股收盘约 04:00-05:00 北京),
  22:40 开盘撮合(美股开盘约 21:30-22:30 北京);周五收盘的信号在北京周一
  早上生成、周一晚开盘执行——恰好就是"下一个美股交易日开盘",语义不丢;
- 无 T+1(T+0 随买随卖)、无涨跌停、零股可交易;
- C1 只用每日收盘价:信号一律以"刚收盘的美股交易日"为信号日;
- 节假日防陈旧成交:撮合前核对报价 preclose 与信号日收盘一致,不一致说明
  今天不是真实交易日(报价是旧session的),整批顺延次日;
- 现金停泊 C9:买入缺现金先卖 SGOV 补上(引擎执行层,同 A股 E3 思路);
- 下单由人在汇丰App人工确认,系统只算信号/发提醒/记日志(纸面撮合用于模拟盘)。

鲁棒性:任何异常只记日志绝不退出;数据源失败 -> 本周期不产生信号,绝不瞎猜。
"""
from __future__ import annotations

import logging
import math
import time as _time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from .broker import PaperBroker
from .config import Section
from .datafeed import DataFeed
from .models import Account, Market, Order, PendingSignal, Side
from .risk import RiskManager
from .store import Store
from .strategies import build_us_strategies
from .strategies.base import Context

log = logging.getLogger("qtrader.us_engine")

EMPTY_DF = pd.DataFrame()


class UsEngine:
    def __init__(self, cfg: Section) -> None:
        self.cfg = cfg
        us = cfg["us"]
        self.enabled = bool(us.get("enabled", True))
        self.state_dir = Path(us.get("state_dir", "./data_us"))
        self.store = Store(self.state_dir)

        # 组件专用子配置:broker/risk 用美股参数,数据缓存落在美股目录
        comp_cfg = Section({
            "broker": {"paper": dict(us.get("broker", {}))},
            "risk": dict(us.get("risk", {})),
            "data": {
                "request_interval_sec": cfg["data"].get("request_interval_sec", 0.35),
                "cache_dir": str(self.state_dir / "cache"),
                "spot_cache_seconds": cfg["data"].get("spot_cache_seconds", 120),
                "hist_cache_hours": cfg["data"].get("hist_cache_hours", 12),
                "val_cache_hours": cfg["data"].get("val_cache_hours", 12),
            },
        })
        self.feed = DataFeed(comp_cfg, self.state_dir)
        self.broker = PaperBroker(comp_cfg, self.store)
        self.risk = RiskManager(comp_cfg, self.store)
        self.strategies = build_us_strategies(cfg, self.store)
        self._us_core = next((s for s in self.strategies
                              if s.name == "us_core"), None)
        self._us_sat = next((s for s in self.strategies
                             if s.name == "us_satellite"), None)

        sch = us["schedule"]
        self.tick_seconds = int(sch.get("tick_seconds", 30))
        self.signal_time = str(sch.get("signal_time", "06:10"))
        self.execute_time = str(sch.get("execute_time", "22:40"))
        self.stale_signal_days = int(sch.get("stale_signal_days", 5))

        self.capital = float(us.get("initial_capital", 10000))
        # paper=纸面自动撮合(模拟盘) / live=实盘人工通道(下单接口,成交回填记账)
        self.mode = str(us.get("mode", "paper")).lower()
        if self.mode not in ("paper", "live"):
            self.mode = "paper"
        acc = self.store.load_account()
        if acc is None:
            acc = Account(cash=self.capital, day_start_equity=self.capital,
                          peak_equity=self.capital)
            self.store.save_account(acc)
            log.info("美股新账户初始化,本金 %.0f USD", self.capital)
        self.account = acc

        self.pool_file = Path(us.get("satellite", {})
                              .get("pool_file", "./us_pool.yaml"))
        self._pool_cache: tuple[float, list] = (0.0, [])

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

    @staticmethod
    def _us_weekday(d: date) -> bool:
        """美股交易日近似:周一~周五;节假日由"数据不变/报价陈旧"防线兜住。"""
        return d.weekday() < 5

    def usd_cny(self) -> Optional[float]:
        try:
            return self.feed.usd_cny()
        except Exception:  # noqa: BLE001
            return None

    # ---------------- 卫星池(us_pool.yaml,理由卡) ----------------
    def load_pool(self) -> list[dict]:
        """加载卫星仓理由卡池(mtime 缓存;S1-S5 人工把关,系统只做字段校验)。"""
        if not self.pool_file.exists():
            return []
        mtime = self.pool_file.stat().st_mtime
        if self._pool_cache[0] == mtime:
            return self._pool_cache[1]
        try:
            data = yaml.safe_load(self.pool_file.read_text(encoding="utf-8")) or {}
            stocks = data.get("stocks") or []
            out = []
            for e in stocks:
                sym = str(e.get("symbol", ""))
                if not sym.startswith("US"):
                    log.warning("us_pool:%s 缺 US 前缀,跳过(应为 USTSM 形式)", sym)
                    continue
                out.append(e)
            self._pool_cache = (mtime, out)
            return out
        except Exception:  # noqa: BLE001
            log.exception("us_pool.yaml 解析失败,本周期按空池处理")
            return []

    # ---------------- 主循环 ----------------
    def serve_forever(self) -> None:
        log.info("美股引擎启动:信号 %s / 撮合 %s(北京时间)",
                 self.signal_time, self.execute_time)
        while True:
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                log.exception("美股 tick 异常(已吞掉,继续循环)")
            _time.sleep(self.tick_seconds)

    def tick(self) -> None:
        if not self.enabled:
            return
        now = self.now()
        if not self._us_weekday(now.date()):
            return
        hhmm = now.hour * 60 + now.minute
        if hhmm >= self._hhmm(self.signal_time) and not self._job_done("us_signal"):
            self.signal_cycle()
            self._mark_job("us_signal")
        if hhmm >= self._hhmm(self.execute_time) and not self._job_done("us_execute"):
            self.execute_cycle()
            self._mark_job("us_execute")

    # ---------------- 行情工具 ----------------
    def _quotes_for(self, symbols: list[str]) -> pd.DataFrame:
        syms = [s for s in dict.fromkeys(symbols) if s and s.startswith("US")]
        if not syms:
            return EMPTY_DF
        try:
            return self.feed.quote(syms)
        except Exception as e:  # noqa: BLE001
            log.warning("美股实时报价失败: %s", e)
            return EMPTY_DF

    def _market_value(self, positions: dict,
                      quote: Optional[pd.DataFrame] = None) -> float:
        if not positions:
            return 0.0
        px = {}
        if quote is not None and len(quote):
            for _, r in quote[quote["symbol"].isin(positions)].iterrows():
                px[str(r["symbol"])] = float(r["price"])
        mv = 0.0
        for s, p in positions.items():
            price = px.get(s)
            if not price or price <= 0:
                price = p.avg_cost
            mv += p.qty * price
        return mv

    def _hist_for(self, symbols: list[str], session: str) -> dict[str, pd.DataFrame]:
        """拉日线到 session 日(含),窗口覆盖 252 个交易日 + 5年回撤检查。"""
        out: dict[str, pd.DataFrame] = {}
        start = (date.fromisoformat(session)
                 - timedelta(days=1400)).isoformat()  # ~5年半自然日
        for s in symbols:
            try:
                h = self.feed.hist(s, start, session)
                if h is not None and not h.empty:
                    out[s] = h
            except Exception as e:  # noqa: BLE001
                log.debug("美股 hist %s 失败: %s", s, e)
        return out

    # ---------------- 收盘后信号周期(C1:只用收盘价) ----------------
    def signal_cycle(self) -> None:
        t0 = self.now()
        log.info("========== 美股收盘后信号周期(北京 %s)==========", t0.date())
        pool = self.load_pool()
        core_sym = self._us_core.symbol if self._us_core else "USQQQM"

        hist = self._hist_for(
            list({core_sym} | {str(e["symbol"]) for e in pool}), t0.date().isoformat())
        core_hist = hist.get(core_sym)
        if core_hist is None or core_hist.empty:
            log.warning("美股核心ETF日线不可用,本周期不产生信号(绝不瞎猜)")
            return
        session = core_hist["dt"].iloc[-1].date()   # 刚收盘的美股交易日
        session_close = float(core_hist["close"].iloc[-1])
        self.store.set_meta("us:session", session.isoformat())
        self.store.set_meta(f"us:last_close:{core_sym}", session_close)
        log.info("信号基准日:美股 %s 收盘(QQQM %.2f)", session, session_close)

        positions = self.store.all_positions()
        quote = self._quotes_for(list(positions) + [core_sym]
                                 + [str(e["symbol"]) for e in pool])
        mv = self._market_value(positions, quote=quote)
        self.account.day_start_equity = self.account.equity(mv)

        rebalance_day = self._is_us_rebalance_day(core_hist, session)
        ctx = Context(
            today=session.isoformat(), quote=quote, hist=hist,
            account=self.account, positions=positions, market_value=mv,
            pool=pool, pending=self.store.pending_signals(),
            rebalance_day=rebalance_day,
        )

        new_count = 0
        for st in self.strategies:
            try:
                sigs = st.entry_signals(ctx) + st.manage(ctx)
            except Exception:  # noqa: BLE001
                log.exception("美股策略 %s 信号异常", st.name)
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

        # 提醒语态:美股准则要求"推送写明规则编号/标的/股数/预估金额" -> 打进日志与看板
        for sig in self.store.pending_signals():
            log.info("待执行提醒 [%s] %s %s(%s) %s", sig.rule, sig.side,
                     sig.symbol, session, sig.reason)

        notes = self.risk.settle_eod(self.account, mv, asof=session)
        for n in notes:
            log.warning("风控: %s", n)
        self.store.save_account(self.account)
        self.store.append_equity(t0, self.account.equity(mv),
                                 self.account.cash, mv)
        self.store.export_snapshot()
        log.info("美股信号周期完成:新增 %d,待执行 %d,净值 %.0f USD",
                 new_count, len(self.store.pending_signals()),
                 self.account.equity(mv))

    def _is_us_rebalance_day(self, core_hist: pd.DataFrame, session: date) -> bool:
        """R1:每年1月的第一个美股交易日(以核心ETF历史为准)。"""
        if session.month != 1:
            return False
        dt_dates = pd.to_datetime(core_hist["dt"]).dt.date
        jan = [d for d in dt_dates if d.year == session.year and d.month == 1]
        return bool(jan) and session == jan[0]

    # ---------------- 开盘撮合周期 ----------------
    def execute_cycle(self) -> None:
        signals = self.store.pending_signals()
        if not signals:
            return
        t0 = self.now()
        if self.mode == "live":
            # 实盘模式:系统不下单——信号转人工订单发提醒,你在券商App
            # 下单成交后回填(fill 命令/看板),系统按真实资金流记账。
            from .fill import submit_manual
            positions = self.store.all_positions()
            quote = self._quotes_for(list({s.symbol for s in signals})
                                     + list(positions))
            submit_manual(self, signals, positions, quote, ts=self.now())
            return
        log.info("========== 美股开盘撮合(北京 %s):%d 条待执行 ==========",
                 t0.date(), len(signals))
        positions = self.store.all_positions()
        quote = self._quotes_for(list({s.symbol for s in signals})
                                 + list(positions))
        if quote.empty:
            log.error("撮合取消:美股报价不可用,信号保留次日重试")
            return
        qrow = {str(r["symbol"]): r for _, r in quote.iterrows()}

        # 节假日防陈旧成交:核心ETF报价的 preclose 应等于信号日收盘
        core_sym = self._us_core.symbol if self._us_core else "USQQQM"
        last_close = self.store.get_meta(f"us:last_close:{core_sym}")
        crow = qrow.get(core_sym)
        if last_close and crow is not None:
            preclose = float(crow["preclose"]) if crow["preclose"] else 0.0
            if preclose <= 0 or abs(preclose / last_close - 1) > 0.005:
                log.warning("报价陈旧(核心ETF昨收%.2f != 信号日收盘%.2f),"
                            "今天多半是美股节假日,整批顺延次日",
                            preclose, last_close)
                return

        today_us = (t0 - timedelta(hours=13)).date()  # 北京22:40 = 美股当日盘中
        sells = [s for s in signals if s.side == Side.SELL]
        buys = [s for s in signals if s.side == Side.BUY]
        for sig in sells + buys:  # 先卖后买:腾出现金
            try:
                self._execute_one(sig, qrow, positions, today_us)
            except Exception:  # noqa: BLE001
                log.exception("美股信号 %s 执行异常(保留重试)", sig.signal_id)

        mv = self._market_value(positions, quote=quote)
        self.store.append_equity(t0, self.account.equity(mv),
                                 self.account.cash, mv)
        self.store.export_snapshot()
        log.info("美股撮合完成:净值 %.0f / 现金 %.0f / 市值 %.0f USD",
                 self.account.equity(mv), self.account.cash, mv)

    def _execute_one(self, sig: PendingSignal, qrow: dict,
                     positions: dict, today: date) -> None:
        def _finish(status: str, note: str, ok: bool, filled_qty: int = 0) -> None:
            sig.status = status
            sig.note = note
            sig.executed_date = today.isoformat()
            self.store.save_signal(sig)
            st = next((s for s in self.strategies if s.name == sig.strategy), None)
            if st is not None:
                st.on_fill(sig, filled_qty, ok)

        age = self._signal_age_days(sig, today)
        if age is not None and age >= self.stale_signal_days:
            _finish("CANCELLED", f"挂起{age}个交易日未成交,作废重评(档位已释放)",
                    ok=False)
            log.info("美股信号作废 %s %s", sig.rule, sig.symbol)
            return

        row = qrow.get(sig.symbol)
        if row is None:
            sig.note = "开盘报价缺失,保留次日重试"
            self.store.save_signal(sig)
            return
        # X1 美股版:按本session开盘价撮合;拿不到开盘价则用现价并注明
        price = float(row["open"]) if row["open"] else float(row["price"])
        if price <= 0:
            sig.note = "开盘价无效,保留次日重试"
            self.store.save_signal(sig)
            return

        if sig.side == Side.SELL:
            pos = positions.get(sig.symbol)
            if pos is None:
                _finish("CANCELLED", "已无持仓,信号取消", ok=True)
                return
            qty = sig.qty if sig.qty > 0 else int(pos.qty * sig.sell_ratio)
            qty = min(qty, pos.qty)
            if qty <= 0:
                _finish("FAILED", "可卖数量为0", ok=False)
                return
            o = Order(symbol=sig.symbol, side=Side.SELL, qty=qty, ts=self.now(),
                      strategy=sig.strategy, reason=f"{sig.rule} | {sig.reason}")
            o = self.broker.execute(o, price, self.account, positions)
            if o.status.value == "FILLED":
                self._sync_t0(positions, sig.symbol)
                _finish("EXECUTED", f"开盘价{price:.2f}成交{o.filled_qty}股",
                        ok=True, filled_qty=o.filled_qty)
            else:
                _finish("FAILED", f"拒单:{o.reason}", ok=False)
            return

        # ---- 买入 ----
        mv = self._market_value(positions, quote=None)
        equity = self.account.equity(mv)
        budget = sig.size_hint_pct * equity
        # C9:现金不够 -> 先卖 SGOV 变现补上(停泊资金回巢)
        if self.account.cash < budget * 1.002:
            self._fund_from_park(budget * 1.002 - self.account.cash,
                                 positions, qrow, sig)
        budget = min(budget, self.account.cash)
        qty = int(budget // (price * 1.002))  # 美股零股可交易,无整手约束
        if qty < 1:
            _finish("FAILED",
                    f"每份预算{budget:.0f}USD不足1股(股价{price:.2f});"
                    f"追加本金或按D1调整参数", ok=False)
            log.warning("买入失败:%s 预算不足1股(%s)", sig.symbol, sig.rule)
            return
        o = Order(symbol=sig.symbol, side=Side.BUY, qty=qty, ts=self.now(),
                  strategy=sig.strategy, reason=f"{sig.rule} | {sig.reason}")
        o = self.broker.execute(o, price, self.account, positions)
        if o.status.value == "FILLED":
            self._sync_t0(positions, sig.symbol)
            _finish("EXECUTED", f"开盘价{price:.2f}成交{o.filled_qty}股",
                    ok=True, filled_qty=o.filled_qty)
        else:
            _finish("FAILED", f"拒单:{o.reason}", ok=False)

    def _sync_t0(self, positions: dict, symbol: str) -> None:
        """美股 T+0:成交后立刻解锁全部可卖。"""
        pos = positions.get(symbol)
        if pos is not None and pos.available_qty < pos.qty:
            pos.available_qty = pos.qty
            self.store.save_position(pos)

    def _fund_from_park(self, need: float, positions: dict, qrow: dict,
                        sig: PendingSignal) -> bool:
        """C9 停泊资金回巢:买入缺现金先卖 SGOV。"""
        if need <= 0 or self._us_core is None:
            return need <= 0
        sym = self._us_core.park_symbol
        pos = positions.get(sym)
        if pos is None or pos.qty <= 0:
            return False
        row = qrow.get(sym)
        px = float(row["price"]) if row is not None else 0.0
        if px <= 0:
            return False
        qty = min(pos.qty, math.ceil(need / px))
        if qty <= 0:
            return False
        o = Order(symbol=sym, side=Side.SELL, qty=qty, ts=self.now(),
                  strategy="us_core",
                  reason=f"C9 | {sig.rule}买{sig.symbol}缺现金{need:.0f}USD,"
                         f"先变现SGOV{qty}股")
        o = self.broker.execute(o, px, self.account, positions)
        if o.status.value == "FILLED":
            self._sync_t0(positions, sym)
            log.info("C9 变现 %s x%d @%.2f(补 %s 买入现金)",
                     sym, o.filled_qty, o.filled_price, sig.symbol)
            return True
        return False

    def _signal_age_days(self, sig: PendingSignal, today: date) -> Optional[int]:
        try:
            d0 = date.fromisoformat(sig.signal_date)
        except ValueError:
            return None
        days = (today - d0).days
        if days <= 0:
            return 0
        # 近似:自然日折算交易日(5/7),陈旧作废是保护性的,粗一点无妨
        return int(days * 5 / 7)

    # ---------------- 三仓占比(R1/R4/R5 展示) ----------------
    def allocation(self) -> dict:
        """三仓占比。准则:未触发份额停泊 SGOV,但记账上属于其归属仓位
        (核心加仓份额记核心仓、个股未触发额度记卫星仓),现金池封顶10%。"""
        positions = self.store.all_positions()
        quote = self._quotes_for(list(positions))
        mv_total = self._market_value(positions, quote=quote)
        equity = self.account.equity(mv_total)
        park_sym = self._us_core.park_symbol if self._us_core else "USSGOV"
        core_mv = sat_mv = park_mv = 0.0
        for s, p in positions.items():
            row = quote[quote["symbol"] == s] if len(quote) else []
            px = float(row.iloc[0]["price"]) if len(row) else p.avg_cost
            v = p.qty * (px or p.avg_cost)
            if p.strategy == "us_satellite":
                sat_mv += v
            elif s != park_sym:
                core_mv += v
            else:  # SGOV 停泊仓(C9):现金池 + 未触发份额
                park_mv += v
        eq = equity if equity > 0 else 1.0
        cash_total = park_mv + self.account.cash
        cash_target = float(self.cfg["us"]["cash"].get("total_pct", 0.10)) * eq
        cash_mv = min(cash_total, cash_target)
        waiting = max(0.0, cash_total - cash_mv)  # 待命份额(归属核心/卫星)
        core_quota = max(0.0, (self._us_core.total_pct if self._us_core else 0.70)
                         * eq - core_mv)
        sat_quota = max(0.0, (self._us_sat.sat_cap if self._us_sat else 0.20)
                        * eq - sat_mv)
        quota = core_quota + sat_quota
        core_wait = waiting * (core_quota / quota) if quota > 0 else 0.0
        sat_wait = waiting - core_wait
        return {
            "core_pct": (core_mv + core_wait) / eq,
            "sat_pct": (sat_mv + sat_wait) / eq,
            "cash_pct": cash_mv / eq,
            "core_deployed_pct": core_mv / eq,
            "sat_deployed_pct": sat_mv / eq,
            "park_waiting_pct": waiting / eq,
            "equity": equity,
        }

    # ---------------- 运维入口 ----------------
    def run_once(self) -> None:
        """立即跑一轮收盘后信号周期(验证/手动补跑)。"""
        self.signal_cycle()
        self._mark_job("us_signal")

    def run_execute(self) -> None:
        """立即撮合待办信号(模拟盘验证/手动补跑)。"""
        self.execute_cycle()
        self._mark_job("us_execute")

    def reset(self) -> None:
        self.account.killed = False
        self.store.save_account(self.account)
        self.store.set_meta("cooldown_until", "")
        log.info("美股账户已复位(冷却/熔断),恢复开仓资格")
