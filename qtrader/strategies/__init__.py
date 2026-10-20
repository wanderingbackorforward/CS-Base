"""策略注册表:按配置实例化启用的策略(顺序即执行优先级)。

A股准则 v2.0 生效策略:value_dip(个股档位抄底 G1-G12)+ etf_core(压舱仓 E1-E4);
美股准则 v1.1 生效策略:us_core(核心仓 C1-C9)+ us_satellite(卫星仓 S1-S13);
旧策略 hot_momentum/cautious_dip 已停用,保留代码供对照。
"""
from __future__ import annotations

import logging

from ..config import Section
from .etf_core import EtfCore
from .us_core import UsCore
from .us_satellite import UsSatellite
from .value_dip import ValueDip

log = logging.getLogger("qtrader.strategies")


def build_strategies(cfg: Section, pool_manager=None, store=None) -> list:
    """A股策略。value_dip/etf_core 需要 pool_manager 与 store;
    缺任一(如旧回测器调用)则该策略禁用并警告。"""
    out = []
    for cls in (ValueDip, EtfCore):
        if pool_manager is None or store is None:
            log.warning("策略 %s 需要 pool_manager/store,本次不加载", cls.name)
            continue
        try:
            inst = cls(cfg, pool_manager, store)
        except Exception:  # noqa: BLE001 配置缺失等:跳过而不是崩引擎
            log.exception("策略 %s 初始化失败,跳过", cls.name)
            continue
        if getattr(inst, "enabled", True):
            out.append(inst)
    return out


def build_us_strategies(cfg: Section, us_store) -> list:
    """美股策略(准则 v1.1):独立 store,配置在 cfg["us"]。"""
    out = []
    for cls in (UsCore, UsSatellite):
        try:
            inst = cls(cfg["us"], us_store)
        except Exception:  # noqa: BLE001
            log.exception("美股策略 %s 初始化失败,跳过", cls.name)
            continue
        if getattr(inst, "enabled", True):
            out.append(inst)
    return out


def loaded_strategy_names() -> set[str]:
    return {ValueDip.name, EtfCore.name, UsCore.name, UsSatellite.name}
