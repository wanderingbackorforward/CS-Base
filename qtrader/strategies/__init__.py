"""策略注册表:新增策略在此登记即可被引擎加载。"""
from __future__ import annotations

from ..config import Section
from .cautious_dip import CautiousDip
from .hot_momentum import HotMomentum


def build_strategies(cfg: Section) -> list:
    """按配置实例化启用的策略(顺序即执行优先级)。"""
    out = []
    for cls in (HotMomentum, CautiousDip):
        inst = cls(cfg)
        if getattr(inst, "enabled", True):
            out.append(inst)
    return out
