"""verify_e2e 固定选基策略包。用于 verify step10，从 ClickHouse 预取选基后每调仓日均持同一组合。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data import BacktestData
from ..strategies.base import FilterStrategy, ScoreStrategy, StrategyBundle
from .low_risk_debt import EqualWeightPosition


@dataclass(frozen=True)
class FixedSelectionFilterStrategy(FilterStrategy):
    """仅保留 allowed_symbols 中的基金。"""

    name: str = "fixed_selection"
    allowed_symbols: tuple[str, ...] = ()

    def filter_symbols(
        self,
        data: BacktestData,
        as_of_date: pd.Timestamp,
        universe: list[str],
    ) -> list[str]:
        allowed = set(self.allowed_symbols)
        return [s for s in universe if s in allowed]


@dataclass(frozen=True)
class FixedSelectionScoreStrategy(ScoreStrategy):
    """按 allowed_symbols 顺序返回打分表（综合得分=1,2,3...）。"""

    name: str = "fixed_selection_score"
    allowed_symbols: tuple[str, ...] = ()

    def score(
        self,
        data: BacktestData,
        as_of_date: pd.Timestamp,
        symbols: list[str],
    ) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame(columns=["symbol", "综合得分", "综合排名"])
        order = {s: i for i, s in enumerate(self.allowed_symbols) if s in symbols}
        sorted_symbols = sorted(symbols, key=lambda s: order.get(s, 9999))
        rows = [
            {"symbol": s, "综合得分": float(len(sorted_symbols) - i), "综合排名": i + 1}
            for i, s in enumerate(sorted_symbols)
        ]
        return pd.DataFrame(rows)


def build_bundle_verify_e2e(allowed_symbols: list[str]) -> StrategyBundle:
    """构建 verify_e2e 策略包，每调仓日均持有 allowed_symbols 的等权组合。"""
    # 仅保留纯数字基金代码（6~8 位），避免非预期格式
    valid = [
        s.zfill(6)
        for s in (str(x).strip() for x in allowed_symbols if x)
        if s.isdigit() and 1 <= len(s) <= 8
    ]
    t = tuple(valid)
    return StrategyBundle(
        name="verify_e2e_top5",
        filter_strategy=FixedSelectionFilterStrategy(allowed_symbols=t),
        score_strategy=FixedSelectionScoreStrategy(allowed_symbols=t),
        position_strategy=EqualWeightPosition(),
    )
