"""核心数据模型:订单/成交/持仓/账户/热度条目。

约定:
- 系统内标的统一使用带市场前缀的 full_id,如 "SH600519" / "SZ000001" / "HK00700";
- 所有时间均为北京时间(Asia/Shanghai)字符串或 datetime;
- 所有金额均为 CNY(港股纸面亦按 CNY 计,忽略汇率,简化处理)。
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional

_ids = itertools.count(1)


class Market(str, Enum):
    SH = "SH"
    SZ = "SZ"
    HK = "HK"
    US = "US"
    BJ = "BJ"   # 北交所(默认股票池排除,但行情需要能解析)


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"


def full_id(code: str, market: Market) -> str:
    return f"{market.value}{code}"


def split_symbol(symbol: str) -> tuple[Market, str]:
    """'SH600519' -> (Market.SH, '600519')"""
    mkt, code = symbol[:2], symbol[2:]
    if mkt not in Market.__members__:
        raise ValueError(f"无法识别的市场前缀: {symbol}")
    return Market(mkt), code


@dataclass
class Order:
    symbol: str
    side: Side
    qty: int
    ts: datetime
    strategy: str = ""
    reason: str = ""                # 人话理由,审计用(看板/日志直接展示)
    size_hint_pct: float = 0.0      # 策略建议仓位占净值比例(0=用风控默认单票上限)
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float = 0.0
    filled_qty: int = 0
    fee: float = 0.0                # 总费用(佣金+印花+过户)

    def __post_init__(self) -> None:
        if not self.order_id:
            self.order_id = f"O{datetime.now().strftime('%y%m%d%H%M%S')}{next(_ids)}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["side"] = self.side.value
        d["status"] = self.status.value
        d["ts"] = self.ts.strftime("%Y-%m-%d %H:%M:%S")
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Order":
        d = dict(d)
        d["side"] = Side(d["side"])
        d["status"] = OrderStatus(d["status"])
        d["ts"] = datetime.strptime(d["ts"], "%Y-%m-%d %H:%M:%S")
        return Order(**d)


@dataclass
class Position:
    symbol: str
    qty: int = 0
    available_qty: int = 0           # T+1:当日买入不可卖
    avg_cost: float = 0.0           # 含费买入均价
    stop_price: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)
    days_held: int = 0               # 持有交易日数,每日开盘 +1
    strategy: str = ""
    high_water: float = 0.0         # 持有期最高收盘价(移动止盈用)
    meta: dict[str, Any] = field(default_factory=dict)  # 策略私有状态(分批/触发价等)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["opened_at"] = self.opened_at.strftime("%Y-%m-%d %H:%M:%S")
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Position":
        d = dict(d)
        d["opened_at"] = datetime.strptime(d["opened_at"], "%Y-%m-%d %H:%M:%S")
        return Position(**d)


@dataclass
class Account:
    cash: float
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    killed: bool = False             # 回撤熔断标志,人工复位

    def equity(self, market_value: float) -> float:
        return self.cash + market_value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Account":
        return Account(**d)


@dataclass
class HeatItem:
    """热度引擎产出:一个概念/行业板块的热度条目"""
    name: str
    day_chg: float = 0.0            # 板块今日涨幅
    amount: float = 0.0             # 板块成交额(元)
    score: float = 0.0
    rank: int = 0
    is_watch: bool = False          # 是否命中你的关注关键词(幻方/孙宇晨映射)
    top_stocks: list[str] = field(default_factory=list)  # 板块内今日最强的几只


@dataclass
class HeatReport:
    ts: datetime
    source: str = ""                 # em=东财概念板块 / spot=全A自聚合
    items: list[HeatItem] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({
            "ts": self.ts.strftime("%Y-%m-%d %H:%M:%S"),
            "source": self.source,
            "items": [asdict(i) for i in self.items],
        }, ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "HeatReport":
        d = json.loads(s)
        return HeatReport(
            ts=datetime.strptime(d["ts"], "%Y-%m-%d %H:%M:%S"),
            source=d["source"],
            items=[HeatItem(**i) for i in d["items"]],
        )
