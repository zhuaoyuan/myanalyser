# -*- coding: utf-8 -*-
"""v2 backtest_helpers 单元测试。

需求来源：20260316 基金过滤逻辑与预期差异调整 - compute_end_extended_str 抽取。
"""
from __future__ import annotations

import pandas as pd
import pytest


class TestComputeEndExtendedStr:
    """compute_end_extended_str 函数测试"""

    def test_normal_within_calendar(self) -> None:
        """正常：hold_days 在交易日历范围内，返回正确延伸日期"""
        from backtest_helpers import compute_end_extended_str

        trading_days = [
            pd.Timestamp("2024-12-27"),  # index 0
            pd.Timestamp("2024-12-30"),  # index 1 (T)
            pd.Timestamp("2024-12-31"),  # index 2
            pd.Timestamp("2025-01-02"),  # index 3
        ]
        as_of = pd.Timestamp("2024-12-30")  # T 日，index 1
        result = compute_end_extended_str(as_of, hold_days=2, trading_days=trading_days)
        assert result == "2025-01-02"  # index 1+2 = 3

    def test_hold_days_zero_returns_t_date(self) -> None:
        """边界：hold_days=0，返回 T 日本身"""
        from backtest_helpers import compute_end_extended_str

        trading_days = [
            pd.Timestamp("2024-12-30"),
            pd.Timestamp("2024-12-31"),
        ]
        as_of = pd.Timestamp("2024-12-30")
        result = compute_end_extended_str(as_of, hold_days=0, trading_days=trading_days)
        assert result == "2024-12-30"

    def test_exceeds_calendar_raises(self) -> None:
        """异常：hold_days 超出交易日历范围，抛出 ValueError"""
        from backtest_helpers import compute_end_extended_str

        trading_days = [
            pd.Timestamp("2024-12-30"),  # T
            pd.Timestamp("2024-12-31"),  # T+1，仅剩 1 个交易日
        ]
        as_of = pd.Timestamp("2024-12-30")
        with pytest.raises(ValueError, match="hold-days.*超出交易日历范围"):
            compute_end_extended_str(as_of, hold_days=5, trading_days=trading_days)

    def test_exactly_last_day_raises(self) -> None:
        """边界：end_index 等于 len，即刚好超出，应抛出"""
        from backtest_helpers import compute_end_extended_str

        trading_days = [pd.Timestamp("2024-12-30"), pd.Timestamp("2024-12-31")]
        # as_of 为最后一日时，hold_days=1 需取下一日，超出日历
        as_of = pd.Timestamp("2024-12-31")  # index 1
        # hold_days=1 -> end_index=2, len=2 -> 2>=2 成立，抛出
        with pytest.raises(ValueError, match="超出交易日历范围"):
            compute_end_extended_str(as_of, hold_days=1, trading_days=trading_days)

    def test_as_of_first_in_calendar(self) -> None:
        """正常：as_of 为日历首日"""
        from backtest_helpers import compute_end_extended_str

        trading_days = [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-04"),
        ]
        as_of = pd.Timestamp("2024-01-02")
        result = compute_end_extended_str(as_of, hold_days=1, trading_days=trading_days)
        assert result == "2024-01-03"
