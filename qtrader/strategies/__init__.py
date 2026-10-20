"""策略注册表:按配置实例化启用的策略(顺序即执行优先级)。

投资系统准则 v3 生效策略(双市场同一套三层结构):
- base_layer   底层仓 L1-L5(只持有不择时,每年1月再平衡)
- trade_engine 交易层通用引擎 E1-E7(σ 自适应档位/追涨,F1-F3 风控)
- reserve      大底储备 Z1-Z4(历史高/三年高三档,停泊 Z4)

旧策略 value_dip/etf_core/us_core/us_satellite 已停用(准则 v1.1/v2.2),
保留代码供对照;旧策略持仓由引擎按 legacy 规则清退。
"""
from __future__ import annotations

import logging

from ..config import Section
from .base_layer import BaseLayer
from .reserve import Reserve
from .trade_engine import TradeEngine

log = logging.getLogger("qtrader.strategies")

_V3 = (BaseLayer, TradeEngine, Reserve)


def build_strategies(cfg: Section, pool_manager=None, store=None) -> list:
    """A股策略(v3 三层):base_layer / trade_engine / reserve。
    三者都读 pool(老登股/科技股池)与 store;缺任一(如旧回测器调用)
    则全部禁用并警告。"""
    out = []
    if pool_manager is None or store is None:
        log.warning("v3 策略需要 pool_manager/store,本次不加载任何策略")
        return out
    for cls in _V3:
        try:
            inst = cls(cfg, store, market="cn", pool_manager=pool_manager)
        except Exception:  # noqa: BLE001 配置缺失等:跳过而不是崩引擎
            log.exception("策略 %s 初始化失败,跳过", cls.name)
            continue
        if getattr(inst, "enabled", True):
            out.append(inst)
    return out


def build_us_strategies(cfg: Section, us_store) -> list:
    """美股策略(v3 三层):配置在 cfg["us"],独立 store,美元计价。"""
    out = []
    for cls in _V3:
        try:
            inst = cls(cfg["us"], us_store, market="us")
        except Exception:  # noqa: BLE001
            log.exception("美股策略 %s 初始化失败,跳过", cls.name)
            continue
        if getattr(inst, "enabled", True):
            out.append(inst)
    return out


def loaded_strategy_names() -> set[str]:
    return {cls.name for cls in _V3}
