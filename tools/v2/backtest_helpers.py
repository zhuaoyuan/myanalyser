"""v2 回测相关共享辅助函数。

位置: myanalyser/tools/v2/backtest_helpers.py
供 multi_t_backtest、verify_filter_flow_report 等复用；通过 sys.path 插入 tools/v2 目录后 import。

协议约定（详见 docs/参考/v2日期与区间协议约定.md）：
- hold_days / t-step: 交易日
- lookback: 1 年 = 243 交易日（A 股口径）
- 日期区间: [start, end] 双闭
"""
from __future__ import annotations

from bisect import bisect_left

import pandas as pd

# A 股口径，与 fund_metrics_core.WindowConfig 一致
TRADING_DAYS_PER_YEAR = 243


def compute_start_from_lookback(
    as_of_date: pd.Timestamp,
    lookback_years: int,
    trading_days: list[pd.Timestamp],
    *,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Timestamp:
    """按交易日历计算 lookback 起始日（1 年 = trading_days_per_year 个交易日）。

    Args:
        as_of_date: T 日，须为交易日
        lookback_years: 回看年数
        trading_days: 交易日历列表（已排序）
        trading_days_per_year: 每年交易日数，默认 243（A 股）

    Returns:
        起始日（T 往前第 lookback_years * trading_days_per_year 个交易日）

    Raises:
        ValueError: 起始索引 < 0（交易日历不足以覆盖 lookback）
    """
    t_index = bisect_left(trading_days, as_of_date)
    if t_index >= len(trading_days):
        t_index = len(trading_days) - 1
    lookback_days = lookback_years * trading_days_per_year
    start_index = t_index - lookback_days
    if start_index < 0:
        raise ValueError(
            f"lookback ({lookback_years} 年 = {lookback_days} 交易日) 超出交易日历范围："
            f"T={as_of_date.date()} 前仅 {t_index} 个交易日"
        )
    return trading_days[start_index]


def compute_end_extended_str(
    as_of_date: pd.Timestamp,
    hold_days: int,
    trading_days: list[pd.Timestamp],
) -> str:
    """按交易日历计算 end_date + hold_days 对应的日期字符串。

    as_of_date 必须为 trading_days 中的交易日（通常由 _resolve_trade_day 得到），
    bisect_left 可正确得到其索引。

    Args:
        as_of_date: 窗口结束日（T 日），须为交易日
        hold_days: 持仓天数
        trading_days: 交易日历列表（已排序）

    Returns:
        YYYY-MM-DD 格式的日期字符串

    Raises:
        ValueError: hold_days 超出交易日历范围
    """
    t_index = bisect_left(trading_days, as_of_date)
    end_index = t_index + hold_days
    if end_index >= len(trading_days):
        remaining = len(trading_days) - t_index - 1  # T 后（不含 T 当日）的交易日数
        raise ValueError(
            f"hold-days ({hold_days}) 超出交易日历范围：T={as_of_date.date()} 后 "
            f"仅剩 {remaining} 个交易日，多 T 回测将中断"
        )
    return trading_days[end_index].strftime("%Y-%m-%d")
