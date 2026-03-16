"""稳健型（低波动偏债）策略包。

依据 docs/参考/分类型的硬约束和主次目标.md 稳健型定义：
- 硬约束：最大回撤 ≥ -8%、年化收益 ≥ 5%、夏普比率 ≥ 0.5
- 主目标：卡玛比率（越大越好）
- 次目标 1：夏普比率（越大越好）
- 次目标 2：最大回撤（越浅越好）
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from compute_fund_composite_score import compute_composite_score

from ..data import BacktestData
from ..filters import SteadyDebtFilterStrategy
from ..metrics import compute_low_risk_debt_metrics
from .base import PositionStrategy, ScoreStrategy, StrategyBundle
from .low_risk_debt import EqualWeightPosition

# 稳健型打分：主目标卡玛 60%，次目标1夏普 25%，次目标2回撤 15%（回撤越浅越好=asc）
_STEADY_DEBT_SCORE_GROUPS = [
    (
        "稳健得分",
        1.0,
        [
            ("近3年卡玛比率", 0.6, "desc"),
            ("近3年夏普比率", 0.25, "desc"),
            ("近3年最大回撤率", 0.15, "asc"),
        ],
    ),
]


@dataclass(frozen=True)
class SteadyDebtScoreStrategy(ScoreStrategy):
    """稳健型评分：主目标卡玛，次目标夏普、回撤。"""

    name: str = "steady_debt_score"

    def score(
        self,
        data: BacktestData,
        as_of_date: pd.Timestamp,
        symbols: list[str],
    ) -> pd.DataFrame:
        rows: list[dict] = []
        as_of_date = pd.Timestamp(as_of_date)

        for symbol in symbols:
            df_symbol = data.by_symbol.get(symbol)
            if df_symbol is None or df_symbol.empty:
                continue
            mask = df_symbol["date"] <= as_of_date
            df_hist = df_symbol.loc[mask]
            if df_hist.empty:
                continue

            dates = df_hist["date"].to_numpy(dtype="datetime64[D]")
            prices = df_hist["close"].to_numpy(dtype=float)
            metrics = compute_low_risk_debt_metrics(dates, prices)
            rows.append({"symbol": symbol, **metrics})

        if not rows:
            return pd.DataFrame(columns=["symbol", "综合得分", "综合排名"])

        df = pd.DataFrame(rows)
        scored = compute_composite_score(df, secondary_groups=_STEADY_DEBT_SCORE_GROUPS)
        scored = scored.sort_values("综合得分", ascending=False).reset_index(drop=True)
        return scored


def build_bundle_steady_debt() -> StrategyBundle:
    """稳健型策略包：稳健型硬约束 + 卡玛主目标打分。"""
    return StrategyBundle(
        name="steady_debt",
        filter_strategy=SteadyDebtFilterStrategy(),
        score_strategy=SteadyDebtScoreStrategy(),
        position_strategy=EqualWeightPosition(),
    )
