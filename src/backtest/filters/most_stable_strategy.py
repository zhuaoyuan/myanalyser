"""最稳健原则过滤策略（FilterStrategy）。

复用 filter_score 的 most_stable 规则，指标由目标日期前的净值数据动态计算
（与 low_risk_debt 的评分逻辑类似，基于 data.by_symbol + as_of_date）。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from most_stable_logic import filter_one
from fund_metrics_core import WindowConfig
from scoreboard_metrics import window_metrics

from ..data import BacktestData
from ..metrics import compute_low_risk_debt_metrics
from ..strategies.base import FilterStrategy


def _compute_most_stable_metrics(
    df_hist: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> dict[str, float | None]:
    """从净值历史计算 most_stable 所需指标（中文列名，百分比需 *100）。

    调用方应保证 df_hist 已过滤为 date <= as_of_date。
    """
    if df_hist.empty or len(df_hist) < 2:
        return {}

    nav_df = df_hist.rename(columns={"date": "净值日期", "close": "复权净值"})[
        ["净值日期", "复权净值"]
    ].copy()
    nav_df = nav_df.sort_values("净值日期")

    if nav_df.empty or len(nav_df) < 2:
        return {}

    cfg = WindowConfig()
    min_3y = cfg.trading_days_per_year * 3
    if len(nav_df) < min_3y:
        return {}

    dates = nav_df["净值日期"].to_numpy(dtype="datetime64[D]")
    prices = nav_df["复权净值"].to_numpy(dtype=float)
    core = compute_low_risk_debt_metrics(dates, prices)
    w3 = window_metrics(nav_df, as_of_date, years=3)

    ann_3y = core.get("近3年年化收益率")
    ann_1y = core.get("近1年年化收益率")
    up_month_3y = core.get("近3年上涨月份比例")
    up_quarter_3y = w3.get("up_quarter_ratio_3y")
    month_std_3y = w3.get("month_return_std_3y")
    sharpe_1y = core.get("近1年夏普比率")
    sharpe_3y = core.get("近3年夏普比率")
    calmar_1y = core.get("近1年卡玛比率")
    calmar_3y = core.get("近3年卡玛比率")

    def _pct(v: float | None) -> float | None:
        return None if v is None else v * 100.0

    return {
        "近3年年化收益率": _pct(ann_3y),
        "近1年年化收益率": _pct(ann_1y),
        "近3年上涨季度比例": _pct(up_quarter_3y),
        "近3年上涨月份比例": _pct(up_month_3y),
        "近3年月涨跌幅标准差": _pct(month_std_3y),
        "近1年夏普比率": sharpe_1y,
        "近3年夏普比率": sharpe_3y,
        "近1年卡玛比率": calmar_1y,
        "近3年卡玛比率": calmar_3y,
    }


@dataclass(frozen=True)
class MostStableFilterStrategy(FilterStrategy):
    """最稳健原则筛选：基于目标日期前的净值动态计算指标并应用 most_stable 规则。"""

    name: str = "most_stable"

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
            row = _compute_most_stable_metrics(df_hist, as_of_ts)
            if not row:
                continue
            is_filtered, _ = filter_one(row)
            if not is_filtered:
                result.append(symbol)
        return result
