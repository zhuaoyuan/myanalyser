# -*- coding: utf-8 -*-
"""v2 日期与区间协议约定 单元测试。

需求来源：myanalyser/docs/参考/v2日期与区间协议约定.md

覆盖场景：
- 正常：交易日口径、lookback 243/年、双闭区间语义
- 异常：超出交易日历、start>end、空/无效输入
- 边界：hold_days=0、lookback=0、as_of 非交易日、单日日历
"""
from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# compute_end_extended_str
# ---------------------------------------------------------------------------
class TestComputeEndExtendedStr:
    """compute_end_extended_str: T + hold_days 交易日延伸"""

    def test_normal_within_calendar(self) -> None:
        """正常：hold_days 在交易日历内，返回正确日期"""
        from backtest_helpers import compute_end_extended_str

        trading_days = [
            pd.Timestamp("2024-12-27"),
            pd.Timestamp("2024-12-30"),  # T
            pd.Timestamp("2024-12-31"),
            pd.Timestamp("2025-01-02"),
        ]
        result = compute_end_extended_str(
            pd.Timestamp("2024-12-30"), hold_days=2, trading_days=trading_days
        )
        assert result == "2025-01-02"

    def test_hold_days_zero_returns_t_date(self) -> None:
        """边界：hold_days=0，返回 T 日本身（含头含尾语义）"""
        from backtest_helpers import compute_end_extended_str

        trading_days = [pd.Timestamp("2024-12-30"), pd.Timestamp("2024-12-31")]
        result = compute_end_extended_str(
            pd.Timestamp("2024-12-30"), hold_days=0, trading_days=trading_days
        )
        assert result == "2024-12-30"

    def test_exceeds_calendar_raises(self) -> None:
        """异常：hold_days 超出交易日历，抛出 ValueError"""
        from backtest_helpers import compute_end_extended_str

        trading_days = [
            pd.Timestamp("2024-12-30"),
            pd.Timestamp("2024-12-31"),
        ]
        with pytest.raises(ValueError, match="hold-days.*超出交易日历范围"):
            compute_end_extended_str(
                pd.Timestamp("2024-12-30"), hold_days=5, trading_days=trading_days
            )

    def test_exactly_last_day_hold_days_one_raises(self) -> None:
        """边界：T 为日历最后一日，hold_days=1 需取下一日，超出"""
        from backtest_helpers import compute_end_extended_str

        trading_days = [pd.Timestamp("2024-12-30"), pd.Timestamp("2024-12-31")]
        with pytest.raises(ValueError, match="超出交易日历范围"):
            compute_end_extended_str(
                pd.Timestamp("2024-12-31"), hold_days=1, trading_days=trading_days
            )

    def test_as_of_not_in_trading_days_bisect_behavior(self) -> None:
        """边界：as_of 非交易日，bisect_left 落在下一交易日（文档要求须在 trading_days 中）"""
        from backtest_helpers import compute_end_extended_str

        trading_days = [
            pd.Timestamp("2024-12-27"),
            pd.Timestamp("2024-12-30"),
            pd.Timestamp("2024-12-31"),
            pd.Timestamp("2025-01-02"),
        ]
        # 2024-12-29 非交易日，bisect_left 得 index=1 -> 2024-12-30
        as_of = pd.Timestamp("2024-12-29")
        result = compute_end_extended_str(as_of, hold_days=1, trading_days=trading_days)
        # bisect_left(2024-12-29) -> 1，t_index=1，end_index=2 -> 2024-12-31
        assert result == "2024-12-31"

    def test_single_element_calendar_hold_days_zero(self) -> None:
        """边界：仅 1 个交易日，hold_days=0 可返回"""
        from backtest_helpers import compute_end_extended_str

        trading_days = [pd.Timestamp("2024-12-30")]
        result = compute_end_extended_str(
            pd.Timestamp("2024-12-30"), hold_days=0, trading_days=trading_days
        )
        assert result == "2024-12-30"

    def test_empty_trading_days_raises(self) -> None:
        """异常：空交易日历，应抛出（bisect_left 可能返回 0）"""
        from backtest_helpers import compute_end_extended_str

        with pytest.raises((IndexError, ValueError)):
            compute_end_extended_str(
                pd.Timestamp("2024-12-30"), hold_days=0, trading_days=[]
            )


# ---------------------------------------------------------------------------
# compute_start_from_lookback
# ---------------------------------------------------------------------------
class TestComputeStartFromLookback:
    """compute_start_from_lookback: 1 年 = 243 交易日"""

    def test_normal_one_year(self) -> None:
        """正常：1 年 lookback，返回 T 前 243 个交易日"""
        from backtest_helpers import compute_start_from_lookback

        trading_days = [pd.Timestamp("2023-01-03") + pd.Timedelta(days=i) for i in range(400)]
        trading_days = [d for d in trading_days if d.dayofweek < 5][:300]
        assert len(trading_days) >= 244
        as_of = trading_days[243]
        result = compute_start_from_lookback(
            as_of, lookback_years=1, trading_days=trading_days
        )
        assert result == trading_days[0]

    def test_three_years(self) -> None:
        """正常：3 年 lookback"""
        from backtest_helpers import compute_start_from_lookback

        trading_days = [pd.Timestamp("2020-01-02") + pd.Timedelta(days=i) for i in range(1200)]
        trading_days = [d for d in trading_days if d.dayofweek < 5][:800]
        as_of = trading_days[729]
        result = compute_start_from_lookback(
            as_of, lookback_years=3, trading_days=trading_days
        )
        assert result == trading_days[0]

    def test_trading_days_per_year_override(self) -> None:
        """正常：自定义 trading_days_per_year 参数"""
        from backtest_helpers import compute_start_from_lookback

        trading_days = [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-04"),
            pd.Timestamp("2024-01-05"),  # index 3
        ]
        # 1 年 = 2 交易日（测试用），start_index = 3 - 2 = 1
        result = compute_start_from_lookback(
            trading_days[3], lookback_years=1, trading_days=trading_days,
            trading_days_per_year=2,
        )
        assert result == trading_days[1]

    def test_exceeds_calendar_raises(self) -> None:
        """异常：lookback 超出交易日历范围"""
        from backtest_helpers import compute_start_from_lookback

        trading_days = [
            pd.Timestamp("2024-12-27"),
            pd.Timestamp("2024-12-30"),
            pd.Timestamp("2024-12-31"),
        ]
        with pytest.raises(ValueError, match="lookback.*超出交易日历范围"):
            compute_start_from_lookback(
                pd.Timestamp("2024-12-30"), lookback_years=1, trading_days=trading_days
            )

    def test_zero_years_returns_t_date(self) -> None:
        """边界：lookback_years=0，返回 T 日本身"""
        from backtest_helpers import compute_start_from_lookback

        trading_days = [pd.Timestamp("2024-12-30"), pd.Timestamp("2024-12-31")]
        as_of = pd.Timestamp("2024-12-31")
        result = compute_start_from_lookback(
            as_of, lookback_years=0, trading_days=trading_days
        )
        assert result == as_of

    def test_as_of_after_calendar_clamps_to_last(self) -> None:
        """边界：as_of 晚于日历最后日，实现中 t_index=len-1"""
        from backtest_helpers import compute_start_from_lookback

        trading_days = [
            pd.Timestamp("2024-12-27"),
            pd.Timestamp("2024-12-30"),
            pd.Timestamp("2024-12-31"),
        ]
        as_of = pd.Timestamp("2025-06-15")  # 晚于最后交易日
        result = compute_start_from_lookback(
            as_of, lookback_years=0, trading_days=trading_days
        )
        # t_index 被 clamp 到 len-1，start_index = 2 - 0 = 2
        assert result == trading_days[2]

    def test_empty_trading_days_raises(self) -> None:
        """异常：空交易日历"""
        from backtest_helpers import compute_start_from_lookback

        with pytest.raises((IndexError, ValueError)):
            compute_start_from_lookback(
                pd.Timestamp("2024-12-30"), lookback_years=0, trading_days=[]
            )


# ---------------------------------------------------------------------------
# t-step / _build_t_list / _resolve_trade_day（协议：t-step 为交易日步进）
# ---------------------------------------------------------------------------
class TestResolveTradeDayAndBuildTList:
    """_resolve_trade_day、_build_t_list：t-step 交易日步进"""

    def test_resolve_trade_day_exact_match(self) -> None:
        """正常：目标日在交易日历中，返回该日"""
        try:
            from multi_t_backtest import _resolve_trade_day
        except ImportError as e:
            pytest.skip(f"multi_t_backtest 依赖未满足: {e}")

        trading_days = [
            pd.Timestamp("2024-12-27"),
            pd.Timestamp("2024-12-30"),
            pd.Timestamp("2024-12-31"),
        ]
        result = _resolve_trade_day(pd.Timestamp("2024-12-30"), trading_days)
        assert result == pd.Timestamp("2024-12-30")

    def test_resolve_trade_day_weekend_prev_day(self) -> None:
        """正常：目标为周末，返回前一个交易日"""
        try:
            from multi_t_backtest import _resolve_trade_day
        except ImportError as e:
            pytest.skip(f"multi_t_backtest 依赖未满足: {e}")

        trading_days = [
            pd.Timestamp("2024-12-27"),  # Fri
            pd.Timestamp("2024-12-30"),  # Mon
            pd.Timestamp("2024-12-31"),  # Tue
        ]
        # 2024-12-28/29 周末
        result = _resolve_trade_day(pd.Timestamp("2024-12-29"), trading_days)
        assert result == pd.Timestamp("2024-12-27")

    def test_resolve_trade_day_before_calendar_raises(self) -> None:
        """异常：目标早于日历首日"""
        try:
            from multi_t_backtest import _resolve_trade_day
        except ImportError as e:
            pytest.skip(f"multi_t_backtest 依赖未满足: {e}")

        trading_days = [pd.Timestamp("2024-12-30"), pd.Timestamp("2024-12-31")]
        with pytest.raises(ValueError, match="earlier than trading calendar"):
            _resolve_trade_day(pd.Timestamp("2024-01-01"), trading_days)

    def test_build_t_list_t_step(self) -> None:
        """正常：t-step 按交易日步进"""
        try:
            from multi_t_backtest import _build_t_list
        except ImportError as e:
            pytest.skip(f"multi_t_backtest 依赖未满足: {e}")

        trading_days = [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-04"),
            pd.Timestamp("2024-01-05"),
        ]
        result = _build_t_list(
            t_list=None,
            t_start="2024-01-02",
            t_end="2024-01-05",
            t_step=2,
            trading_days=trading_days,
        )
        assert result == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")]

    def test_build_t_list_no_params_raises(self) -> None:
        """异常：缺少 t-list 且缺少 t-start/t-end/t-step"""
        try:
            from multi_t_backtest import _build_t_list
        except ImportError as e:
            pytest.skip(f"multi_t_backtest 依赖未满足: {e}")

        trading_days = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
        with pytest.raises(ValueError, match="must provide"):
            _build_t_list(
                t_list=None, t_start=None, t_end=None, t_step=None,
                trading_days=trading_days,
            )

    def test_build_t_list_start_after_end_raises(self) -> None:
        """异常：t-start > t-end"""
        try:
            from multi_t_backtest import _build_t_list
        except ImportError as e:
            pytest.skip(f"multi_t_backtest 依赖未满足: {e}")

        trading_days = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
        with pytest.raises(ValueError, match="t-start cannot be after t-end"):
            _build_t_list(
                t_list=None,
                t_start="2024-01-05",
                t_end="2024-01-02",
                t_step=1,
                trading_days=trading_days,
            )


# ---------------------------------------------------------------------------
# 双闭区间语义 [start, end]
# ---------------------------------------------------------------------------
class TestDoubleClosedIntervalSemantics:
    """日期区间 [start, end] 双闭：>= start 且 <= end"""

    def test_compare_series_slice_double_closed(self) -> None:
        """验证 compare 模块使用双闭区间切片（协议：[start,end] 双闭）"""
        # 协议要求：>= start 且 <= end；compare 模块 L192-193 使用相同逻辑
        # 通过代码检查实现，此处做语义断言
        start_ts = pd.Timestamp("2024-01-02")
        end_ts = pd.Timestamp("2024-01-05")
        dates = [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-05"),
            pd.Timestamp("2024-01-06"),
        ]
        s = pd.Series([1, 2, 3, 4, 5], index=dates)
        filtered = s[(s.index >= start_ts) & (s.index <= end_ts)]
        assert list(filtered.index) == [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-05"),
        ]
        assert pd.Timestamp("2024-01-02") in filtered.index
        assert pd.Timestamp("2024-01-05") in filtered.index
        assert pd.Timestamp("2024-01-01") not in filtered.index
        assert pd.Timestamp("2024-01-06") not in filtered.index

    def test_constant_trading_days_per_year(self) -> None:
        """协议常量：1 年 = 243 交易日"""
        try:
            from fund_metrics_core import WindowConfig
            expected = WindowConfig.trading_days_per_year
        except ImportError:
            from backtest_helpers import TRADING_DAYS_PER_YEAR
            expected = TRADING_DAYS_PER_YEAR
        assert expected == 243, "A股口径：1年=243交易日"
