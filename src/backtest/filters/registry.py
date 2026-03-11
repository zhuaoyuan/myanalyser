"""过滤器注册表。通过环境变量 FUND_BACKTEST_FILTERS 指定链式过滤器名（逗号分隔）。"""

from __future__ import annotations

import os

from .base import FundFilter
from .filtered_candidates_csv import FilteredCandidatesCsvFilter
from .max_funds import MaxFundsFilter

REGISTRY: dict[str, type[FundFilter]] = {
    "filtered_candidates": FilteredCandidatesCsvFilter,
    "max_funds": MaxFundsFilter,
}

ENV_VAR = "FUND_BACKTEST_FILTERS"


def get_filter_chain() -> list[FundFilter]:
    """从环境变量 FUND_BACKTEST_FILTERS 解析过滤器名，按顺序返回实例列表。"""
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return []
    names = [n.strip() for n in raw.split(",") if n.strip()]
    chain: list[FundFilter] = []
    for name in names:
        key = name.lower()
        if key not in REGISTRY:
            available = ", ".join(sorted(REGISTRY.keys()))
            raise ValueError(f"未知过滤器: {name}，可用: {available}")
        chain.append(REGISTRY[key]())
    return chain


def apply_filter_chain(candidates: set[str], filters: list[FundFilter]) -> set[str]:
    """对候选集依次应用过滤器链（每步取交集/子集）。"""
    result = candidates
    for f in filters:
        result = f.filter(result)
        if not result:
            break
    return result
