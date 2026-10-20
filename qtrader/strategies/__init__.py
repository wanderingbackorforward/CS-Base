"""策略注册表:按配置实例化启用的策略(顺序即执行优先级)。

投资系统准则 v5 生效策略(双市场同一套三层结构):
- core_layer 核心层 C1-C3(持有+再平衡带;美股只买一只标普500 ETF)
- attack     进攻仓 H1-H10(重仓赢家/早走输家;人物名单 P1-P5 + 论点卡 H3)

v3 的 trade_engine/base_layer/reserve 与 v2 的 value_dip/etf_core、
v4 的 us_core/us_satellite 均已废除(旧持仓由引擎按 legacy 规则清退)。
"""
from __future__ import annotations

import logging

from ..config import Section
from .attack import Attack
from .core_layer import CoreLayer

log = logging.getLogger("qtrader.strategies")

_V5 = (CoreLayer, Attack)


def build_strategies(cfg: Section, pool_manager=None, store=None) -> list:
    """A股策略(v5 两层):core_layer / attack。A股 attack 需要 pool_manager
    (估值分位),缺则该策略跳过并警告。"""
    out = []
    if store is None:
        log.warning("v5 策略需要 store,本次不加载任何策略")
        return out
    for cls in _V5:
        try:
            inst = cls(cfg, store, market="cn", pool_manager=pool_manager)
        except Exception:  # noqa: BLE001 配置缺失等:跳过而不是崩引擎
            log.exception("策略 %s 初始化失败,跳过", cls.name)
            continue
        if getattr(inst, "enabled", True):
            out.append(inst)
    return out


def build_us_strategies(cfg: Section, us_store) -> list:
    """美股策略(v5 两层):配置在 cfg["us"],独立 store,美元计价。"""
    out = []
    for cls in _V5:
        try:
            inst = cls(cfg["us"], us_store, market="us")
        except Exception:  # noqa: BLE001
            log.exception("美股策略 %s 初始化失败,跳过", cls.name)
            continue
        if getattr(inst, "enabled", True):
            out.append(inst)
    return out


def loaded_strategy_names() -> set[str]:
    return {cls.name for cls in _V5}
