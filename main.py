"""QTrader CLI 入口。

常用命令(准则 v2.0 X1 流程):
  python main.py serve     # 常驻运行:收盘后算信号 + 次日开盘撮合 + 看板(:8000)
  python main.py once      # 立即跑一轮收盘后信号周期(验证/手动补跑)
  python main.py exec      # 立即撮合当前待执行信号(模拟盘验证/手动补跑)
  python main.py pool      # 跑 P1 硬条件自动筛,产出 data/pool_candidates.csv
  python main.py status    # 查看账户/持仓/待执行信号/档位状态
  python main.py reset     # 人工复位熔断/冷却
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
from pathlib import Path

from qtrader.config import load_config
from qtrader.engine import Engine


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


def main() -> None:
    parser = argparse.ArgumentParser(description="QTrader 量化交易系统(A股准则 v2.0)")
    parser.add_argument("command", choices=["serve", "once", "exec", "status",
                                            "reset", "pool"])
    parser.add_argument("-c", "--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    state_dir = Path(cfg["app"].get("state_dir", "./data"))
    setup_logging(state_dir, cfg["app"].get("log_level", "INFO"))

    if args.command == "pool":
        cmd_pool(cfg)
        return

    engine = Engine(cfg)

    if args.command == "serve":
        import threading
        import uvicorn
        from qtrader.dashboard import create_app

        app = create_app(engine)
        host = cfg["server"].get("host", "0.0.0.0")
        port = int(cfg["server"].get("port", 8000))
        t = threading.Thread(
            target=lambda: uvicorn.run(app, host=host, port=port,
                                       log_level="warning"),
            daemon=True)
        t.start()
        print(f"看板已启动: http://<服务器IP>:{port}")
        engine.serve_forever()
    elif args.command == "once":
        engine.run_once()
    elif args.command == "exec":
        engine.run_execute()
    elif args.command == "reset":
        engine.reset()
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
