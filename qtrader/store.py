"""SQLite 状态持久化:账户/订单/持仓/净值曲线/热度报告/熔断标志。

设计要点:
- 所有状态唯一落盘于这一个库文件(docker 挂载 ./data 即可保留全部状态);
- 订单表是"事实源",持仓/账户可由订单+成交重放推导,此处直接存最新快照并保留订单流水;
- 线程安全:引擎线程写、看板线程读,统一走 threading.Lock + 短事务。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import Account, HeatReport, Order, PendingSignal, Position

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT
);
CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY, ts TEXT, symbol TEXT, side TEXT, qty INTEGER,
    status TEXT, filled_price REAL, filled_qty INTEGER, fee REAL,
    strategy TEXT, reason TEXT
);
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY, data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS equity_curve (
    ts TEXT, equity REAL, cash REAL, market_value REAL
);
CREATE TABLE IF NOT EXISTS heat_reports (
    ts TEXT, source TEXT, data TEXT
);
CREATE TABLE IF NOT EXISTS pending_signals (
    signal_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, signal_date TEXT,
    strategy TEXT, rule TEXT, reason TEXT, size_hint_pct REAL, sell_ratio REAL,
    qty INTEGER, tranche TEXT, trigger_high REAL, status TEXT,
    executed_date TEXT, filled_price REAL, filled_qty INTEGER, note TEXT
);
CREATE TABLE IF NOT EXISTS pool_states (
    symbol TEXT PRIMARY KEY, data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(ts);
CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_curve(ts);
CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_signals(status);
"""


class Store:
    def __init__(self, state_dir: str | Path) -> None:
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "cache").mkdir(exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        self.db_path = self.dir / "qtrader.db"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---------- 基础 ----------
    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ---------- meta(冷却截止日/熔断等开关) ----------
    def set_meta(self, key: str, value: Any) -> None:
        self._exec(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )

    def get_meta(self, key: str, default: Any = None) -> Any:
        rows = self._query("SELECT value FROM meta WHERE key=?", (key,))
        return json.loads(rows[0][0]) if rows else default

    # ---------- 账户 ----------
    def save_account(self, acc: Account) -> None:
        self._exec(
            "INSERT INTO account(id,data) VALUES(1,?) "
            "ON CONFLICT(id) DO UPDATE SET data=excluded.data",
            (json.dumps(acc.to_dict(), ensure_ascii=False),),
        )

    def load_account(self) -> Account | None:
        rows = self._query("SELECT data FROM account WHERE id=1")
        return Account.from_dict(json.loads(rows[0][0])) if rows else None

    # ---------- 订单 ----------
    def save_order(self, o: Order) -> None:
        self._exec(
            "INSERT OR REPLACE INTO orders(order_id,ts,symbol,side,qty,status,"
            "filled_price,filled_qty,fee,strategy,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (o.order_id, o.ts.strftime("%Y-%m-%d %H:%M:%S"), o.symbol, o.side.value,
             o.qty, o.status.value, o.filled_price, o.filled_qty, o.fee,
             o.strategy, o.reason),
        )

    def recent_orders(self, limit: int = 50) -> list[Order]:
        rows = self._query(
            "SELECT order_id,ts,symbol,side,qty,status,filled_price,filled_qty,fee,"
            "strategy,reason FROM orders ORDER BY ts DESC,order_id DESC LIMIT ?",
            (limit,),
        )
        return [
            Order(
                order_id=r[0], ts=datetime.strptime(r[1], "%Y-%m-%d %H:%M:%S"),
                symbol=r[2], side=_side(r[3]), qty=r[4],
                status=_status(r[5]), filled_price=r[6] or 0.0, filled_qty=r[7] or 0,
                fee=r[8] or 0.0, strategy=r[9] or "", reason=r[10] or "",
            )
            for r in rows
        ]

    # ---------- 持仓 ----------
    def all_positions(self) -> dict[str, Position]:
        rows = self._query("SELECT symbol,data FROM positions")
        return {r[0]: Position.from_dict(json.loads(r[1])) for r in rows}

    def save_position(self, p: Position) -> None:
        self._exec(
            "INSERT OR REPLACE INTO positions(symbol,data) VALUES(?,?)",
            (p.symbol, json.dumps(p.to_dict(), ensure_ascii=False)),
        )

    def drop_position(self, symbol: str) -> None:
        self._exec("DELETE FROM positions WHERE symbol=?", (symbol,))

    # ---------- 净值曲线 ----------
    def append_equity(self, ts: datetime, equity: float, cash: float, mv: float) -> None:
        self._exec(
            "INSERT INTO equity_curve(ts,equity,cash,market_value) VALUES(?,?,?,?)",
            (ts.strftime("%Y-%m-%d %H:%M:%S"), equity, cash, mv),
        )

    def equity_curve(self, limit: int = 5000) -> list[dict]:
        rows = self._query(
            "SELECT ts,equity FROM (SELECT ts,equity FROM equity_curve "
            "ORDER BY ts DESC LIMIT ?) ORDER BY ts", (limit,),
        )
        return [{"ts": r[0], "equity": r[1]} for r in rows]

    # ---------- 热度报告 ----------
    def save_heat(self, report: HeatReport) -> None:
        self._exec(
            "INSERT INTO heat_reports(ts,source,data) VALUES(?,?,?)",
            (report.ts.strftime("%Y-%m-%d %H:%M:%S"), report.source, report.to_json()),
        )

    def latest_heat(self) -> HeatReport | None:
        rows = self._query(
            "SELECT data FROM heat_reports ORDER BY ts DESC LIMIT 1")
        return HeatReport.from_json(rows[0][0]) if rows else None

    # ---------- 待执行信号(X1:收盘后算,次日开盘撮合) ----------
    _SIG_COLS = ("signal_id,symbol,side,signal_date,strategy,rule,reason,"
                 "size_hint_pct,sell_ratio,qty,tranche,trigger_high,status,"
                 "executed_date,filled_price,filled_qty,note")

    def save_signal(self, s: PendingSignal) -> None:
        d = s.to_dict()
        self._exec(
            "INSERT OR REPLACE INTO pending_signals("
            f"{self._SIG_COLS}) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            tuple(d[c] for c in self._SIG_COLS.split(",")),
        )

    def pending_signals(self, statuses: tuple[str, ...] = ("PENDING",)) -> list[PendingSignal]:
        marks = ",".join("?" * len(statuses))
        rows = self._query(
            f"SELECT {self._SIG_COLS} FROM pending_signals "
            f"WHERE status IN ({marks}) ORDER BY signal_date, signal_id",
            statuses)
        out = []
        for r in rows:
            d = dict(zip(self._SIG_COLS.split(","), r))
            out.append(PendingSignal.from_dict(d))
        return out

    def recent_signals(self, limit: int = 50) -> list[PendingSignal]:
        rows = self._query(
            f"SELECT {self._SIG_COLS} FROM pending_signals "
            "ORDER BY signal_date DESC, signal_id DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            d = dict(zip(self._SIG_COLS.split(","), r))
            out.append(PendingSignal.from_dict(d))
        return out

    # ---------- 池子档位状态(value_dip 策略状态机) ----------
    def save_pool_state(self, symbol: str, data: dict) -> None:
        self._exec(
            "INSERT OR REPLACE INTO pool_states(symbol,data) VALUES(?,?)",
            (symbol, json.dumps(data, ensure_ascii=False)),
        )

    def pool_state(self, symbol: str, default: dict | None = None) -> dict:
        rows = self._query("SELECT data FROM pool_states WHERE symbol=?", (symbol,))
        return json.loads(rows[0][0]) if rows else (default or {})

    def all_pool_states(self) -> dict[str, dict]:
        rows = self._query("SELECT symbol,data FROM pool_states")
        return {r[0]: json.loads(r[1]) for r in rows}

    # ---------- 人工可读快照 ----------
    def export_snapshot(self) -> Path:
        """每次决策后导出一份人可读 JSON(方便肉眼检查/迁移/提交私有仓库)。"""
        acc = self.load_account()
        snap = {
            "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "account": acc.to_dict() if acc else None,
            "positions": [p.to_dict() for p in self.all_positions().values()],
            "orders_recent": [o.to_dict() for o in self.recent_orders(30)],
            "signals_recent": [s.to_dict() for s in self.recent_signals(30)],
        }
        path = self.dir / "snapshot.json"
        path.write_text(
            json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def _side(v: str):
    from .models import Side
    return Side(v)


def _status(v: str):
    from .models import OrderStatus
    return OrderStatus(v)
