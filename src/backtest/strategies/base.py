"""回测策略接口定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from ..data import BacktestData


class FilterStrategy(Protocol):
    name: str

    def filter_symbols(
        self,
        data: BacktestData,
        as_of_date: pd.Timestamp,
        universe: list[str],
    ) -> list[str]:
        ...


class ScoreStrategy(Protocol):
    name: str

    def score(
        self,
        data: BacktestData,
        as_of_date: pd.Timestamp,
        symbols: list[str],
    ) -> pd.DataFrame:
        """返回包含 symbol、综合得分、综合排名 的 DataFrame。"""
        ...


class PositionStrategy(Protocol):
    name: str

    def target_weights(
        self,
        scored: pd.DataFrame,
        top_n: int,
    ) -> dict[str, float]:
        ...


@dataclass(frozen=True)
class StrategyBundle:
    name: str
    filter_strategy: FilterStrategy
    score_strategy: ScoreStrategy
    position_strategy: PositionStrategy
