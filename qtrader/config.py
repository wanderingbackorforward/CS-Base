"""配置加载:yaml -> 类型宽松的 Config 对象(嵌套 dict 访问 + 默认值)。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class Section:
    """用 cfg["risk"]["kill"] 的写法访问 yaml 配置,键缺失时返回默认值而不是抛错。"""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        v = self._data.get(key)
        return Section(v) if isinstance(v, dict) else v

    def get(self, key: str, default: Any = None) -> Any:
        v = self._data.get(key)
        return default if v is None else v

    def __contains__(self, key: str) -> bool:
        return key in self._data


def load_config(path: str | None = None) -> Section:
    path = path or os.environ.get("QTRADER_CONFIG", "config.yaml")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p.resolve()}")
    with open(p, encoding="utf-8") as f:
        return Section(yaml.safe_load(f))
