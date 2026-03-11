"""基金过滤器抽象。

本目录统一存放两类过滤器实现：
- FundFilter：加载前过滤器链，仅接收基金编码集合，用于 FILTERED_FUND_CANDIDATES_CSV、max_funds 等
- FilterStrategy：策略包内筛选，接收 data/as_of_date/universe，可基于净值动态计算（如 MostStableFilterStrategy）
"""

from __future__ import annotations

from typing import Protocol


class FundFilter(Protocol):
    """加载前过滤器链协议。在 load_fund_nav_data 之前执行，仅基于基金编码。"""

    def filter(self, candidates: set[str]) -> set[str]:
        """输入候选基金编码集合，返回过滤后的子集。"""
        ...
