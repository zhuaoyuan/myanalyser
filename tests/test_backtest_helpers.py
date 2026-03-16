# -*- coding: utf-8 -*-
"""v2 backtest_helpers 单元测试。

需求来源：20260316 基金过滤逻辑与预期差异调整 - compute_end_extended_str 抽取。

Import 说明：backtest_helpers 位于 tools/v2，需 conftest 将 tools/v2 加入 sys.path 后方可导入，
故各 test 方法内延迟 import，与 conftest 的路径注入时机一致。
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


class TestComputeStartFromLookback:
    """compute_start_from_lookback 函数测试（1 年 = 243 交易日）"""

    def test_normal_one_year(self) -> None:
        """正常：1 年 lookback，返回 T 前 243 个交易日"""
        from backtest_helpers import compute_start_from_lookback

        # 构造至少 244 个交易日（index 0..243），T 在 index 243
        trading_days = [pd.Timestamp("2023-01-03") + pd.Timedelta(days=i) for i in range(400)]
        trading_days = [d for d in trading_days if d.dayofweek < 5][:300]
        assert len(trading_days) >= 244, f"need 244+ trading days, got {len(trading_days)}"
        as_of = trading_days[243]  # T 日
        result = compute_start_from_lookback(as_of, lookback_years=1, trading_days=trading_days)
        assert result == trading_days[0]  # index 243 - 243 = 0

    def test_three_years(self) -> None:
        """正常：3 年 lookback"""
        from backtest_helpers import compute_start_from_lookback

        trading_days = [pd.Timestamp("2020-01-02") + pd.Timedelta(days=i) for i in range(1200)]
        trading_days = [d for d in trading_days if d.dayofweek < 5][:800]
        as_of = trading_days[729]  # T = 第 730 个交易日
        result = compute_start_from_lookback(as_of, lookback_years=3, trading_days=trading_days)
        # 729 - 3*243 = 729 - 729 = 0
        assert result == trading_days[0]

    def test_exceeds_calendar_raises(self) -> None:
        """异常：lookback 超出交易日历范围，抛出 ValueError"""
        from backtest_helpers import compute_start_from_lookback

        trading_days = [
            pd.Timestamp("2024-12-27"),
            pd.Timestamp("2024-12-30"),  # T
            pd.Timestamp("2024-12-31"),
        ]
        as_of = pd.Timestamp("2024-12-30")  # index 1，前仅 1 个交易日
        with pytest.raises(ValueError, match="lookback.*超出交易日历范围"):
            compute_start_from_lookback(as_of, lookback_years=1, trading_days=trading_days)

    def test_zero_years_returns_same_day(self) -> None:
        """边界：lookback 0 年，返回 T 日本身"""
        from backtest_helpers import compute_start_from_lookback

        trading_days = [
            pd.Timestamp("2024-12-30"),
            pd.Timestamp("2024-12-31"),
        ]
        as_of = pd.Timestamp("2024-12-31")
        result = compute_start_from_lookback(as_of, lookback_years=0, trading_days=trading_days)
        assert result == as_of
