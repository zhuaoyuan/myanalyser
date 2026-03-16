"""v2 回测相关共享辅助函数。

位置: myanalyser/tools/v2/backtest_helpers.py
供 multi_t_backtest、verify_filter_flow_report 等复用；通过 sys.path 插入 tools/v2 目录后 import。
"""
from __future__ import annotations

from bisect import bisect_left

import pandas as pd


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
