"""基金过滤器抽象。"""

from __future__ import annotations

from typing import Protocol


class FundFilter(Protocol):
    """基金过滤器协议。主流程链式调用，不感知具体实现。"""

    def filter(self, candidates: set[str]) -> set[str]:
        """输入候选基金编码集合，返回过滤后的子集。"""
        ...
