"""低风险偏债策略包（筛选 + 评分 + 仓位）。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data import BacktestData
from ..metrics import compute_low_risk_debt_metrics
from ...compute_fund_composite_score import compute_composite_score
from .base import FilterStrategy, PositionStrategy, ScoreStrategy, StrategyBundle


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


@dataclass(frozen=True)
class LowRiskDebtScoreStrategy(ScoreStrategy):
    name: str = "low_risk_debt_score"

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
        scored = compute_composite_score(df)
        scored = scored.sort_values("综合得分", ascending=False).reset_index(drop=True)
        return scored


@dataclass(frozen=True)
class EqualWeightPosition(PositionStrategy):
    name: str = "equal_weight"

    def target_weights(self, scored: pd.DataFrame, top_n: int) -> dict[str, float]:
        if scored.empty:
            return {}
        if top_n <= 0:
            return {}
        top = scored.head(top_n)
        symbols = top["symbol"].tolist()
        if not symbols:
            return {}
        weight = 1.0 / len(symbols)
        return {s: weight for s in symbols}


def build_bundle() -> StrategyBundle:
    return StrategyBundle(
        name="low_risk_debt",
        filter_strategy=PassThroughFilter(),
        score_strategy=LowRiskDebtScoreStrategy(),
        position_strategy=EqualWeightPosition(),
    )
