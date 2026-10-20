"""事件驱动回测器(日频):与实盘/纸面共用同一套策略代码与费用模型——
回测里能赚钱的逻辑,就是实盘里跑的逻辑;不做两套会互相骗人的代码。

诚实性声明(必须写在前面的三个偏差):
1) 幸存者偏差:股票池按"当前"成交额头部取样本,退市/ST 票天然被排除;
2) 尾盘成交近似:决策与成交均按当日收盘价(+滑点),真实尾盘单与收盘价有差异;
3) 盘中止损近似:跳空低开按开盘价成交(悲观正确),盘中触发按当日收盘(近似)。
日频回测无法验证的东西(涨停排板成交、日内路径)本回测一律不做美化,结论仅作方向参考。

用法:python main.py backtest [--start 2024-06-01 --end 2026-10-01 --size 60 --heat-mode proxy]
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from qtrader.broker import PaperBroker
from qtrader.config import Section
from qtrader.datafeed import DataFeed, DataFeedError
from qtrader.engine import EMPTY_DF
from qtrader.models import Account, Market, Order, OrderStatus, Side, full_id
from qtrader.risk import RiskManager
from qtrader.store import Store
from qtrader.strategies import build_strategies
from qtrader.strategies.base import Context

log = logging.getLogger("qtrader.backtest")


class InMemoryStore(Store):
    """与生产 Store 同接口的内存版(回测绝不写生产库)。"""

    def __init__(self) -> None:  # 不调用父类的文件初始化
        self._orders: list[dict] = []
        self._positions: dict[str, Any] = {}
        self._account: dict | None = None
        self._equity: list[dict] = []
        self._meta: dict[str, Any] = {}

    def set_meta(self, key, value): self._meta[key] = value
    def get_meta(self, key, default=None): return self._meta.get(key, default)
    def save_account(self, acc): self._account = acc.to_dict()
    def load_account(self):
        from qtrader.models import Account as A
        return A.from_dict(dict(self._account)) if self._account else None
    def save_order(self, o): self._orders.append(o.to_dict())
    def recent_orders(self, limit=50):
        from qtrader.models import Order as O
        return [O.from_dict(dict(d)) for d in self._orders[-limit:]]
    def all_positions(self):
        from qtrader.models import Position
        return {k: Position.from_dict(dict(v)) for k, v in self._positions.items()}
    def save_position(self, p): self._positions[p.symbol] = p.to_dict()
    def drop_position(self, s): self._positions.pop(s, None)
    def append_equity(self, ts, equity, cash, mv):
        self._equity.append({"ts": ts.strftime("%Y-%m-%d %H:%M:%S"),
                             "equity": equity})
    def equity_curve(self, limit=5000): return self._equity[:limit]
    def export_snapshot(self): return Path(".")


def _load_positions_mut(store: InMemoryStore) -> dict:
    """直接取内存字典本体(回测内部用,减少序列化开销)。"""
    return store._positions  # noqa: SLF001


def run_backtest(cfg: Section, start: str | None = None, end: str | None = None,
                 universe_size: int | None = None, heat_mode: str | None = None) -> None:
    bt = cfg["backtest"]
    start = start or bt.get("start", "2024-06-01")
    end = end or bt.get("end", "2026-10-01")
    size = universe_size or int(bt.get("universe_size", 60))
    heat_mode = heat_mode or bt.get("heat_mode", "proxy")

    state_dir = Path(cfg["app"].get("state_dir", "./data"))
    feed = DataFeed(cfg, state_dir)
    log.info("回测区间 %s ~ %s | 池大小 %d | 热度模式 %s", start, end, size, heat_mode)

    # ---------- 交易日序列 ----------
    cal = feed.trade_dates()
    sd, ed = date.fromisoformat(start), date.fromisoformat(end)
    if cal:
        days = [d for d in cal if sd <= d <= ed]
    else:
        days = [d.date() for d in pd.bdate_range(sd, ed)]
    if not days:
        log.error("回测区间内没有交易日")
        return
    look_start = (days[0] - timedelta(days=320)).isoformat()

    # ---------- 股票池 ----------
    a_cfg = cfg["market"]["a_share"]
    symbols: list[str] = []
    try:
        spot = feed.spot_all(force=True)
        df = spot[spot["amount"] > 0]
        def _ok(r) -> bool:
            code, name = str(r["code"]), str(r["name"])
            if a_cfg.get("exclude_st", True) and ("ST" in name.upper() or "退" in name):
                return False
            if code.startswith(("8", "4", "92", "688")):
                return False
            if code.startswith("300") and not a_cfg.get("include_chinext", False):
                return False
            return True
        df = df[df.apply(_ok, axis=1)].nlargest(size, "amount")
        symbols = df["symbol"].tolist()
    except Exception as e:  # noqa: BLE001
        log.error("股票池构建失败(行情不可达): %s", e)
        return
    dip_targets = [t["symbol"] for t in cfg["universe"]["dip_targets"]]
    all_syms = list(dict.fromkeys(symbols + dip_targets))
    log.info("股票池 %d 只(+抄底池 %d 只)。注意:按当前成交额取样,存在幸存者偏差。",
             len(symbols), len(dip_targets))

    # ---------- 预取日K ----------
    hist: dict[str, pd.DataFrame] = {}
    for i, s in enumerate(all_syms, 1):
        try:
            h = feed.hist(s, look_start, end)
            if h is not None and not h.empty:
                hist[s] = h
        except Exception as e:  # noqa: BLE001
            log.debug("hist %s 失败: %s", s, e)
        if i % 20 == 0:
            log.info("预取日K %d/%d", i, len(all_syms))
    log.info("日K预取完成: %d/%d 成功", len(hist), len(all_syms))

    regime_index = str(cfg["risk"].get("regime_index", "SH000001"))
    try:
        regime = feed.regime_hist(regime_index, look_start, end)
    except Exception as e:  # noqa: BLE001
        regime = None
        log.warning("指数数据不可用,仓位闸门按保守模式: %s", e)

    # ---------- 热度 boards 模式数据 ----------
    boards: dict[str, tuple[pd.Series, pd.Series, list[str]]] = {}
    if heat_mode == "boards":
        rank = None
        try:
            rank = feed.concept_rank()
        except Exception:  # noqa: BLE001
            pass
        if rank is None or len(rank) == 0:
            log.warning("板块数据不可达,自动降级为 proxy 热度模式")
            heat_mode = "proxy"
        else:
            top_names = rank.nlargest(int(bt.get("boards_top_n", 30)),
                                      "day_chg")["name"].tolist()
            for i, bname in enumerate(top_names, 1):
                try:
                    h = feed.concept_hist(bname, look_start, end)
                    cons = feed.concept_cons(bname)
                    if h is not None and len(h):
                        h = h.set_index("dt")["close"]
                        pct = h.pct_change().fillna(0)
                        cons_syms = [full_id(str(c),
                                             Market.SH if str(c).startswith(("6", "9", "5"))
                                             else Market.SZ)
                                     for c in cons["code"]]
                        boards[bname] = (h, pct, cons_syms)
                except Exception as e:  # noqa: BLE001
                    log.debug("板块 %s 失败: %s", bname, e)
                if i % 10 == 0:
                    log.info("板块预取 %d/%d", i, len(top_names))
            log.info("板块数据就绪 %d 个", len(boards))

    # ---------- 收益率矩阵(proxy 热度用) ----------
    closes = pd.DataFrame({s: h.set_index("dt")["close"] for s, h in hist.items()
                           if len(h)})
    ret3 = closes.pct_change(3)

    # ---------- 回测核心 ----------
    store = InMemoryStore()
    capital = float(cfg["app"].get("initial_capital", 10000))
    account = Account(cash=capital, day_start_equity=capital, peak_equity=capital)
    store.save_account(account)
    broker = PaperBroker(cfg, store)
    risk = RiskManager(cfg, store)
    strategies = build_strategies(cfg)
    positions: dict[str, Any] = {}

    equity_pts: list[tuple[str, float]] = []
    trades: list[dict] = []
    d0 = days[0]
    pre_close: dict[str, float] = {}
    hint_pct = float(cfg["risk"].get("max_position_pct", 0.30))

    total = len(days)
    for idx, d in enumerate(days):
        d_str = d.isoformat()
        # --- 开盘 promote ---
        for p in positions.values():
            p.available_qty = p.qty
            p.days_held += 1
        if d != d0:
            account.day_start_equity = account.equity(_mv(positions, pre_close))
        # --- 熔断:全清并跳过策略 ---
        if account.killed:
            _flatten(positions, broker, account, d, hist, trades)
        # --- 开盘跳空止损 ---
        for sym, p in list(positions.items()):
            h = hist.get(sym)
            if h is None or p.stop_price <= 0:
                continue
            row = h[h["dt"] == d_str]
            if not len(row):
                continue
            op = float(row.iloc[0]["open"])
            if op <= p.stop_price:
                pre_cost = p.avg_cost
                o = Order(symbol=sym, side=Side.SELL, qty=p.available_qty,
                          ts=datetime(d.year, d.month, d.day, 9, 30),
                          strategy=p.strategy,
                          reason=f"跳空低开 {op:.2f} 击穿止损 {p.stop_price:.2f}")
                o = broker.execute(o, op, account, positions)
                _record(trades, o, pre_cost, d)
        # --- 14:40 尾盘决策 ---
        # 合成当日 spot(只包含需要的行)
        need = list(positions) + dip_targets
        cands = _candidates(d, heat_mode, ret3, boards, closes)
        need += cands
        rows = []
        for s in dict.fromkeys(need):
            h = hist.get(s)
            if h is None:
                continue
            r = h[h["dt"] == d_str]
            if not len(r):
                continue
            r = r.iloc[0]
            o_, c_, hi_, lo_ = (float(r["open"]), float(r["close"]),
                                float(r["high"]), float(r["low"]))
            amt = float(r["amount"]) if "amount" in r and pd.notna(r["amount"]) \
                and float(r["amount"]) > 0 else c_ * float(r["volume"]) * 100
            rows.append({"symbol": s, "code": s[2:], "name": s, "price": c_,
                         "pct_chg": (c_ / pre_close[s] - 1) if s in pre_close else 0.0,
                         "open": o_, "high": hi_, "low": lo_,
                         "volume": float(r["volume"]), "amount": amt})
        spot_df = pd.DataFrame(rows)
        if not len(spot_df):
            continue
        histslice = {s: hist[s][hist[s]["dt"] <= d_str].reset_index(drop=True)
                     for s in set(need) & set(hist)}
        regime_close = None
        if regime is not None:
            rc = regime[regime["dt"] <= d_str]["close"]
            regime_close = rc if len(rc) else None
        mv = _mv(positions, {r["symbol"]: r["price"] for _, r in spot_df.iterrows()})
        ctx = Context(today=d_str, spot=spot_df, quote=spot_df, hist=histslice,
                      regime_close=regime_close, account=account, positions=positions,
                      market_value=mv, candidates=cands)
        for st in strategies:
            try:
                entries = st.entry_signals(ctx)
            except Exception:  # noqa: BLE001
                log.exception("策略 %s 入场异常 @%s", st.name, d_str)
                entries = []
            for o in entries:
                price = ctx.price(o.symbol)
                if price <= 0:
                    continue
                hint = o.size_hint_pct or hint_pct
                budget = min(hint * account.equity(mv), account.cash)
                qty = int(budget // (price * 1.001))
                qty -= qty % 100
                if qty < 100:
                    continue
                ok, why = risk.check_open(account, positions, o.symbol, qty,
                                          price, mv, regime_close, asof=d)
                if not ok:
                    continue
                o.qty = qty
                o = broker.execute(o, price, account, positions)
                _record(trades, o, 0.0, d)
            try:
                sells = st.manage(ctx)
            except Exception:  # noqa: BLE001
                log.exception("策略 %s 管理异常 @%s", st.name, d_str)
                sells = []
            for o in sells:
                p = positions.get(o.symbol)
                if not p or p.available_qty <= 0:
                    continue
                pre_cost = p.avg_cost
                o.qty = p.available_qty
                o = broker.execute(o, ctx.price(o.symbol), account, positions)
                _record(trades, o, pre_cost, d)
        # --- 更新 pre_close / 净值 ---
        for _, r in spot_df.iterrows():
            pre_close[r["symbol"]] = float(r["price"])
        mv = _mv(positions, {r["symbol"]: r["price"] for _, r in spot_df.iterrows()})
        notes = risk.settle_eod(account, mv)
        for n in notes:
            log.info("%s | %s", d_str, n)
        if account.killed:
            _flatten(positions, broker, account, d, hist, trades)
        eq = account.equity(mv)
        equity_pts.append((d_str, eq))
        if (idx + 1) % max(total // 10, 1) == 0:
            log.info("回测进度 %d/%d 净值 %.0f", idx + 1, total, eq)

    # ---------- 统计 ----------
    _summary(equity_pts, trades, capital, regime, cfg, d0.isoformat(), d.isoformat())


def _mv(positions: dict, px: dict) -> float:
    mv = 0.0
    for s, p in positions.items():
        price = px.get(s) or p.avg_cost
        mv += p.qty * price
    return mv


def _flatten(positions, broker, account, d, hist, trades) -> None:
    for sym, p in list(positions.items()):
        if p.available_qty <= 0:
            continue
        h = hist.get(sym)
        if h is None:
            continue
        r = h[h["dt"] == d.isoformat()]
        if not len(r):
            continue
        price = float(r.iloc[0]["close"])
        pre_cost = p.avg_cost
        o = Order(symbol=sym, side=Side.SELL, qty=p.available_qty,
                  ts=datetime(d.year, d.month, d.day, 9, 30), strategy=p.strategy,
                  reason="熔断清仓")
        o = broker.execute(o, price, account, positions)
        _record(trades, o, pre_cost, d)


def _record(trades: list, o: Order, pre_cost: float, d) -> None:
    if o.status != OrderStatus.FILLED:
        return
    realized = 0.0 if o.side == Side.BUY else \
        (o.filled_price - pre_cost) * o.filled_qty - o.fee
    trades.append({"date": d.isoformat(), "symbol": o.symbol, "side": o.side.value,
                   "qty": o.filled_qty, "price": o.filled_price, "fee": o.fee,
                   "realized": round(realized, 2), "strategy": o.strategy,
                   "reason": o.reason})


def _candidates(d, heat_mode, ret3, boards, closes) -> list[str]:
    """当日热度候选池:proxy=3日动量榜;boards=当日板块涨幅榜成分并集。"""
    d_str = d.isoformat()
    if heat_mode == "boards" and boards:
        ranked: list[str] = []
        scored = []
        for name, (_, pct, cons) in boards.items():
            v = pct.get(d_str)
            if pd.notna(v):
                scored.append((float(v), name))
        scored.sort(reverse=True)
        for _, name in scored[:6]:
            ranked += boards[name][2]
        in_pool = [s for s in dict.fromkeys(ranked) if s in closes.columns]
        return in_pool[:80]
    if d_str in ret3.index:
        col = ret3.loc[d_str].dropna().sort_values(ascending=False)
        return col.head(40).index.tolist()
    return []


def _summary(equity_pts, trades, capital, regime, cfg, start, end) -> None:
    if not equity_pts:
        log.error("无净值数据")
        return
    eq = pd.Series({d: v for d, v in equity_pts})
    final = float(eq.iloc[-1])
    total_ret = final / capital - 1
    days = (date.fromisoformat(eq.index[-1]) - date.fromisoformat(eq.index[0])).days
    ann = (1 + total_ret) ** (365 / max(days, 1)) - 1 if days > 0 else 0.0
    roll_max = eq.cummax()
    dd = (eq / roll_max - 1).min()
    closed = [t for t in trades if t["side"] == "SELL"]
    wins = [t for t in closed if t["realized"] > 0]
    losses = [t for t in closed if t["realized"] <= 0]
    gross_win = sum(t["realized"] for t in wins)
    gross_loss = abs(sum(t["realized"] for t in losses))
    fees = sum(t["fee"] for t in trades)
    buys = [t for t in trades if t["side"] == "BUY"]

    bench = ""
    if regime is not None and len(regime):
        seg = regime[(regime["dt"] >= start) & (regime["dt"] <= end)]
        if len(seg) >= 2:
            b = float(seg["close"].iloc[-1]) / float(seg["close"].iloc[0]) - 1
            bench = f"{b:+.1%}"

    print("\n" + "=" * 62)
    print(f"回测结果 {start} ~ {end}(本金 {capital:.0f},基准指数同期 {bench or 'N/A'})")
    print("=" * 62)
    print(f"期末净值   {final:>12,.0f}   总收益 {total_ret:+.1%}   年化 {ann:+.1%}")
    print(f"最大回撤   {dd:>12.1%}   交易 {len(trades)} 笔(买 {len(buys)} / 卖 {len(closed)})")
    if closed:
        wr = len(wins) / len(closed)
        pf = gross_win / gross_loss if gross_loss else float("inf")
        print(f"胜率       {wr:>12.1%}   盈亏比 {pf:.2f}"
              f"(赢 {len(wins)} 亏 {len(losses)})")
    print(f"总费用     {fees:>12,.0f}(佣金+印花+过户+滑点已计入净值)")
    for name in dict.fromkeys(t["strategy"] for t in trades):
        sub = [t for t in trades if t["strategy"] == name]
        pnl = sum(t["realized"] for t in sub if t["side"] == "SELL")
        print(f"  策略 {name:<16} 交易 {len(sub):>3} 笔   已实现盈亏 {pnl:+,.0f}")
    if closed:
        hold = 0
        cnt = 0
        by_sym: dict[str, list] = {}
        for t in closed:
            by_sym.setdefault(t["symbol"], []).append(t)
        print("最近 5 笔平仓:")
        for t in closed[-5:]:
            print(f"  {t['date']} {t['symbol']:<10} {t['strategy']:<14} "
                  f"{t['realized']:+8.0f}  {t['reason'][:30]}")
    print("=" * 62)

    out = Path(cfg["app"].get("state_dir", "./data")) / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(trades).to_csv(out / f"trades_{stamp}.csv", index=False)
    eq.rename("equity").to_csv(out / f"equity_{stamp}.csv")
    print(f"明细已保存: {out}/trades_{stamp}.csv / equity_{stamp}.csv")
    print("提示:样本池有幸存者偏差,尾盘成交按收盘近似;结果只作方向参考,")
    print("      纸面交易 1-3 个月的数据才是实盘前的最终考试。")
