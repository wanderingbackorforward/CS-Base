"""QTrader CLI 入口(A股准则 v2.0 + 美股准则 v1.1 双引擎)。

常用命令(X1 流程:收盘后算信号 -> 次日开盘撮合):
  python main.py serve      # 常驻:A股+美股双引擎 + 看板(:8000,密码登录)
  python main.py once       # A股:立即跑一轮收盘后信号周期
  python main.py exec       # A股:立即撮合当前待执行信号(paper)/转人工订单(live)
  python main.py pool       # A股:P1 硬条件自动筛,产出 data/pool_candidates.csv
  python main.py status     # A股:账户/持仓/待执行信号/档位状态
  python main.py us-once    # 美股:立即跑一轮收盘后信号周期(北京时间06:10同款)
  python main.py us-exec    # 美股:立即撮合待执行信号(开盘价撮合)
  python main.py us-status  # 美股:账户/三仓占比/信号/档位状态/汇率
  python main.py reset      # A股:人工复位熔断/冷却
  python main.py us-reset   # 美股:人工复位

live 实盘人工通道(下单在券商App,系统记账):
  python main.py pending    # 双账户列出待人工执行信号
  python main.py fill <signal_id> --price 296.3 --qty 9 --fee 18 [--mood 3]
  python main.py skip <signal_id> --note 理由
  python main.py cash --market us --delta 5000 --note 入金
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
import time
from pathlib import Path

from qtrader.config import load_config
from qtrader.engine import Engine
from qtrader.us_engine import UsEngine


def setup_logging(state_dir: Path, level: str = "INFO") -> None:
    log_dir = state_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "qtrader.log", maxBytes=10 * 1024 * 1024, backupCount=5,
        encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def cmd_pool(cfg) -> None:
    """P1 硬条件自动筛:产出 data/pool_candidates.csv,人工 P2/P4 后写 pool.yaml。"""
    from qtrader.datafeed import DataFeed
    from qtrader.pool import PoolManager
    feed = DataFeed(cfg, cfg["app"].get("state_dir", "./data"))
    pm = PoolManager(cfg, feed)
    print("P1 硬条件筛选开始(全A快照 -> 市值/成交额粗筛 -> 逐只财务核验)……")
    df = pm.run_hard_filter()
    if df is not None and len(df):
        passed = df[df["hard_pass"] == "PASS"]
        print(f"候选 {len(df)} 只,其中硬条件全过(PASS) {len(passed)} 只;")
        print("已导出 data/pool_candidates.csv -> 人工判断(P2)+写理由卡(P4)"
              "-> 拷入 pool.yaml 生效")
        name_col = "name" if "name" in passed.columns else None
        for _, r in passed.head(20).iterrows():
            print(f"  {r['symbol']} {r.get(name_col, '') if name_col else ''} "
                  f"市值{float(r['market_cap'])/1e8:.0f}亿")
    else:
        print("P1 未产出候选(数据源不可用?),检查日志后重试")


def cmd_us_status(us: UsEngine) -> None:
    acc = us.account
    store = us.store
    positions = store.all_positions()
    alloc = us.allocation()
    quote = us._quotes_for(list(positions))
    mv = us._market_value(positions, quote=quote)
    eq = acc.equity(mv)
    fx = us.usd_cny()
    print(f"[美股账户 USD] 净值 {eq:,.0f} | 现金 {acc.cash:,.0f} | 市值 {mv:,.0f}"
          f" | 总盈亏 {eq - us.capital:+,.0f}")
    print(f"三仓占比(R1/R4/R5):核心 {alloc['core_pct']:.0%} / "
          f"卫星 {alloc['sat_pct']:.0%} / 现金池 {alloc['cash_pct']:.0%}"
          f"(目标 70%/20%/10%)")
    if fx:
        print(f"报税参考:USD/CNY ≈ {fx:.3f},净值折人民币约 {eq * fx:,.0f} 元"
              f"(境外所得次年3.1-6.30自行申报,盈亏同年内可互抵)")
    else:
        print("USD/CNY 汇率暂不可用(报税参考,不影响交易)")
    if positions:
        print(f"持仓 {len(positions)} 只:")
        for s, p in positions.items():
            print(f"  {s} [{p.strategy}] x{p.qty} @ {p.avg_cost:.2f} USD")
    else:
        print("持仓 0 只(启动后 C4 会先买 QQQM 底仓,C9 把闲置现金停泊 SGOV)")
    signals = store.pending_signals()
    print(f"待执行信号 {len(signals)} 条:")
    for s in signals:
        print(f"  [{s.rule}] {s.side} {s.symbol} (信号日 {s.signal_date}) "
              f"{s.reason[:70]}")
    states = store.all_pool_states()
    active = {k: v for k, v in states.items()
              if any(t.get("state") not in ("available",)
                     for t in (v.get("tranches") or {}).values())
              or v.get("needs_review")
              or (v.get("base") or {}).get("deployed")}
    if active:
        print("档位状态(底仓/非空闲/待复核):")
        for sym, st in active.items():
            if st.get("needs_review"):
                print(f"  {sym}: 需人工复核 -> {st.get('review_note', '')}")
                continue
            base = st.get("base") or {}
            if base.get("deployed"):
                print(f"  {sym} 底仓: {base.get('qty', 0)} 股已部署(C4)")
            for k, tr in st.get("tranches", {}).items():
                if tr.get("state") != "available":
                    print(f"  {sym} {k}档: {tr['state']} "
                          f"触发前高{tr.get('trigger_high', 0):.2f} "
                          f"数量{tr.get('qty', 0)}")


def cmd_manual(engine, us_engine, args) -> None:
    """live 人工通道:pending / fill / skip / cash。"""
    from qtrader import fill as F

    engines = [engine] + ([us_engine] if us_engine else [])
    names = {id(engine): "A股", id(us_engine): "美股"}

    if args.command == "pending":
        total = 0
        for eng in engines:
            sigs = eng.store.pending_signals()
            cur = "USD" if eng is us_engine else "CNY"
            print(f"[{names[id(eng)]} {cur} {eng.mode}模式] 待人工执行信号 {len(sigs)} 条")
            for s in sigs:
                oid = eng.store.get_meta(f"manual_order:{s.signal_id}")
                mark = "已登记订单" if oid else "未登记"
                print(f"  {s.signal_id}")
                print(f"    [{s.rule}] {s.side} {s.symbol} "
                      f"(信号日 {s.signal_date}) {mark}")
                print(f"    {s.reason[:76]}")
            total += len(sigs)
        print(f"共 {total} 条;成交后回填: python main.py fill <signal_id> "
              f"--price <成交价> --qty <股数> --fee <费用>")
        return

    if args.command == "fill":
        if not args.price or not args.qty:
            print("用法: fill <signal_id> --price 成交价 --qty 股数 [--fee 费用] "
                  "[--mood 1-5](以券商App成交回报为准)")
            return
        eng = F.resolve_engine(engines, args.signal_id)
        if eng is None:
            print(f"找不到待执行信号: {args.signal_id}(先跑 pending 查看)")
            return
        r = F.fill_signal(eng, args.signal_id, args.price, args.qty,
                          args.fee or 0.0, args.mood or 0)
        print(("OK " if r["ok"] else "FAIL ") + r["msg"])
        return

    if args.command == "skip":
        eng = F.resolve_engine(engines, args.signal_id)
        if eng is None:
            print(f"找不到待执行信号: {args.signal_id}")
            return
        r = F.skip_signal(eng, args.signal_id, args.note or "")
        print(("OK " if r["ok"] else "FAIL ") + r["msg"])
        return

    if args.command == "cash":
        if args.market not in ("cn", "us"):
            print("用法: cash --market cn|us (--delta 金额 | --set 绝对值) "
                  "[--note 备注];delta>0入金,<0出金")
            return
        eng = us_engine if args.market == "us" else engine
        if eng is None:
            print("该市场引擎未启用")
            return
        if args.delta is None and args.set_to is None:
            print(f"[{names[id(eng)]}] 当前系统现金 {eng.account.cash:,.2f} "
                  f"(本金基准 {eng.capital:,.0f})")
            return
        r = F.cash_adjust(eng, delta=args.delta or 0.0, set_to=args.set_to,
                          note=args.note or "")
        print(("OK " if r["ok"] else "FAIL ") + r["msg"])
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QTrader 量化交易系统(A股准则 v2.0 + 美股准则 v1.1)")
    parser.add_argument("command", choices=["serve", "once", "exec", "status",
                                            "reset", "pool",
                                            "us-once", "us-exec", "us-status",
                                            "us-reset",
                                            "pending", "fill", "skip", "cash"])
    parser.add_argument("signal_id", nargs="?", default=None,
                        help="fill/skip 用:信号ID(pending 可查,支持唯一前缀)")
    parser.add_argument("--price", type=float, default=None,
                        help="fill 用:券商App实际成交价")
    parser.add_argument("--qty", type=int, default=None,
                        help="fill 用:实际成交股数")
    parser.add_argument("--fee", type=float, default=0.0,
                        help="fill 用:券商实际收取的费用(佣金等)")
    parser.add_argument("--mood", type=int, default=None, choices=[1, 2, 3, 4, 5],
                        help="fill 用:下单时情绪 1-5(D2 纪律字段)")
    parser.add_argument("--note", default=None, help="skip/cash 用:备注")
    parser.add_argument("--market", default=None, choices=["cn", "us"],
                        help="cash 用:目标账户")
    parser.add_argument("--delta", type=float, default=None,
                        help="cash 用:入金(+)出金(-)金额")
    parser.add_argument("--set", dest="set_to", type=float, default=None,
                        help="cash 用:直接校准到该绝对现金值")
    parser.add_argument("-c", "--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    state_dir = Path(cfg["app"].get("state_dir", "./data"))
    setup_logging(state_dir, cfg["app"].get("log_level", "INFO"))

    if args.command == "pool":
        cmd_pool(cfg)
        return

    engine = Engine(cfg)
    us_enabled = bool(cfg["us"].get("enabled", False))
    us_engine = UsEngine(cfg) if us_enabled else None

    if args.command in ("pending", "fill", "skip", "cash"):
        cmd_manual(engine, us_engine, args)
        return

    if args.command == "serve":
        import threading
        import uvicorn
        from qtrader.dashboard import create_app

        app = create_app(engine, us_engine)
        host = cfg["server"].get("host", "0.0.0.0")
        port = int(cfg["server"].get("port", 8000))
        t = threading.Thread(
            target=lambda: uvicorn.run(app, host=host, port=port,
                                       log_level="warning"),
            daemon=True)
        t.start()
        print(f"看板已启动: http://<服务器IP>:{port} (登录密码见 config.yaml server.password)")
        engines = [engine] + ([us_engine] if us_engine else [])
        print("双引擎常驻运行(A股 15:10信号/09:25撮合;美股 06:10信号/22:40撮合,北京时间)")
        tick = min(int(e.tick_seconds if hasattr(e, "tick_seconds") else 30)
                   for e in engines)
        while True:
            for e in engines:
                try:
                    e.tick()
                except Exception:  # noqa: BLE001
                    logging.getLogger("qtrader.main").exception(
                        "%s tick 异常(已吞掉)", type(e).__name__)
            time.sleep(tick)
    elif args.command == "once":
        engine.run_once()
    elif args.command == "exec":
        engine.run_execute()
    elif args.command == "reset":
        engine.reset()
    elif args.command == "us-once":
        if us_engine:
            us_engine.run_once()
    elif args.command == "us-exec":
        if us_engine:
            us_engine.run_execute()
    elif args.command == "us-reset":
        if us_engine:
            us_engine.reset()
    elif args.command == "us-status":
        if us_engine:
            cmd_us_status(us_engine)
    elif args.command == "status":
        acc = engine.account
        store = engine.store
        positions = store.all_positions()
        mv = engine._market_value(positions)
        eq = acc.equity(mv)
        print(f"净值 {eq:.0f} | 现金 {acc.cash:.0f} | 市值 {mv:.0f} "
              f"| 总盈亏 {eq - engine.capital:+.0f} "
              f"| 熔断 {'是' if acc.killed else '否'}")
        cash_pct = acc.cash / eq if eq > 0 else 0
        if cash_pct < engine.cash_floor:
            print(f"! R2 警告:现金池 {cash_pct:.0%} 低于下限 {engine.cash_floor:.0%}")
        tranche_pct = float(cfg["strategy"]["value_dip"].get("tranche_pct", 0.04))
        lot_budget = tranche_pct * eq
        if lot_budget < 1000:
            print(f"! 本金适配:每份预算 {lot_budget:.0f} 元,"
                  f"股价超过 {lot_budget/100:.1f} 元的池内股将买不进一手"
                  f"(G3 会如实拒单;追加本金或按 D1 调 tranche_pct)")
        print(f"持仓 {len(positions)} 只:")
        for s, p in positions.items():
            print(f"  {s} [{p.strategy}] x{p.qty} @ {p.avg_cost:.2f} "
                  f"可卖 {p.available_qty} 持有 {p.days_held}日")
        signals = store.pending_signals()
        print(f"待执行信号 {len(signals)} 条:")
        for s in signals:
            print(f"  [{s.rule}] {s.side} {s.symbol} (信号日 {s.signal_date}) "
                  f"{s.reason[:60]}")
        states = store.all_pool_states()
        active = {k: v for k, v in states.items()
                  if any(t.get("state") not in ("available",)
                         for t in (v.get("tranches") or {}).values())
                  or v.get("needs_review")}
        if active:
            print("档位状态(非空闲/待复核):")
            for sym, st in active.items():
                if st.get("needs_review"):
                    print(f"  {sym}: 需人工复核 -> {st.get('review_note', '')}")
                    continue
                for k, tr in st.get("tranches", {}).items():
                    if tr.get("state") != "available":
                        print(f"  {sym} {k}档: {tr['state']} "
                              f"触发前高{tr.get('trigger_high', 0):.2f} "
                              f"数量{tr.get('qty', 0)}")


if __name__ == "__main__":
    main()
