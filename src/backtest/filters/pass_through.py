"""直通过滤器（FilterStrategy）。原样返回 universe，用于策略包中不做筛选的场景。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data import BacktestData
from ..strategies.base import FilterStrategy


@dataclass(frozen=True)
class PassThroughFilter(FilterStrategy):
    name: str = "pass_through"

    def filter_symbols(
        self,
        data: BacktestData,
        as_of_date: pd.Timestamp,
        universe: list[str],
    ) -> list[str]:
        return universe
