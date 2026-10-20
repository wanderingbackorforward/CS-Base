"""QTrader CLI 入口。

常用命令:
  python main.py serve     # 常驻运行:引擎调度 + 看板(默认 :8000)
  python main.py once      # 立即跑一轮完整决策(验证/手动补跑)
  python main.py status    # 查看账户/持仓/净值
  python main.py backtest --start 2024-06-01 --end 2026-10-01 --size 60
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


def main() -> None:
    parser = argparse.ArgumentParser(description="QTrader 量化交易系统")
    parser.add_argument("command", choices=["serve", "once", "status", "reset",
                                            "backtest"])
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("--start", default=None, help="回测开始日期")
    parser.add_argument("--end", default=None, help="回测结束日期")
    parser.add_argument("--size", type=int, default=None, help="回测股票池大小")
    parser.add_argument("--heat-mode", default=None, choices=["proxy", "boards"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    state_dir = Path(cfg["app"].get("state_dir", "./data"))
    setup_logging(state_dir, cfg["app"].get("log_level", "INFO"))
    log = logging.getLogger("qtrader.main")

    if args.command == "backtest":
        from backtest import run_backtest
        run_backtest(cfg, start=args.start, end=args.end,
                     universe_size=args.size, heat_mode=args.heat_mode)
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
        log.info("看板已启动: http://<服务器IP>:%d", port)
        engine.serve_forever()
    elif args.command == "once":
        engine.run_once()
    elif args.command == "status":
        acc = engine.account
        positions = engine.store.all_positions()
        mv = engine._market_value(positions)
        eq = acc.equity(mv)
        print(f"净值 {eq:.0f} | 现金 {acc.cash:.0f} | 市值 {mv:.0f} "
              f"| 总盈亏 {eq - engine.capital:+.0f} "
              f"| 熔断 {'是' if acc.killed else '否'}")
        for s, p in positions.items():
            print(f"  {s} [{p.strategy}] x{p.qty} @ {p.avg_cost:.2f} "
                  f"止损 {p.stop_price or '-'} 持有 {p.days_held}日")
    elif args.command == "reset":
        engine.reset()


if __name__ == "__main__":
    main()
