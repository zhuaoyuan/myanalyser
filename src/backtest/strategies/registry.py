"""策略包注册表。"""

from __future__ import annotations

from typing import Dict

from .base import StrategyBundle
from .low_risk_debt import build_bundle as build_low_risk_debt


_STRATEGY_BUILDERS = {
    "low_risk_debt": build_low_risk_debt,
}


def get_strategy_bundle(name: str) -> StrategyBundle:
    key = name.strip().lower()
    if key not in _STRATEGY_BUILDERS:
        available = ", ".join(sorted(_STRATEGY_BUILDERS.keys()))
        raise ValueError(f"未知策略包: {name}，可用: {available}")
    return _STRATEGY_BUILDERS[key]()


def list_strategy_names() -> list[str]:
    return sorted(_STRATEGY_BUILDERS.keys())
