"""稳健型（低波动偏债）过滤策略（FilterStrategy）。

依据 docs/参考/分类型的硬约束和主次目标.md 稳健型定义：
- 硬约束：最大回撤 ≥ -8%、年化收益 ≥ 5%、夏普比率 ≥ 0.5
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fund_metrics_core import WindowConfig
from steady_debt_logic import filter_one

from ..data import BacktestData
from ..metrics import compute_low_risk_debt_metrics
from ..strategies.base import FilterStrategy


def _compute_steady_debt_metrics(
    df_hist: pd.DataFrame,
) -> dict[str, float | None]:
    """从净值历史计算稳健型 filter 所需指标（小数形式）。

    调用方应保证 df_hist 已过滤为 date <= as_of_date。
    指标为 compute_low_risk_debt_metrics 返回的原始小数形式。
    """
    if df_hist.empty or len(df_hist) < 2:
        return {}

    cfg = WindowConfig()
    min_3y = cfg.trading_days_per_year * 3
    if len(df_hist) < min_3y:
        return {}

    dates = df_hist["date"].to_numpy(dtype="datetime64[D]")
    prices = df_hist["close"].to_numpy(dtype=float)
    core = compute_low_risk_debt_metrics(dates, prices)

    return {
        "近3年最大回撤率": core.get("近3年最大回撤率"),
        "近3年年化收益率": core.get("近3年年化收益率"),
        "近3年夏普比率": core.get("近3年夏普比率"),
    }


@dataclass(frozen=True)
class SteadyDebtFilterStrategy(FilterStrategy):
    """稳健型筛选：基于目标日期前的净值动态计算指标并应用稳健型硬约束。"""

    name: str = "steady_debt"

    def filter_symbols(
        self,
        data: BacktestData,
        as_of_date: pd.Timestamp,
        universe: list[str],
    ) -> list[str]:
        result: list[str] = []
        as_of_ts = pd.Timestamp(as_of_date)
        for symbol in universe:
            df_symbol = data.by_symbol.get(symbol)
            if df_symbol is None or df_symbol.empty:
                continue
            mask = df_symbol["date"] <= as_of_ts
            df_hist = df_symbol.loc[mask]
            if df_hist.empty:
                continue
            row = _compute_steady_debt_metrics(df_hist)
            if not row:
                continue
            is_filtered, _ = filter_one(row)
            if not is_filtered:
                result.append(symbol)
        return result
