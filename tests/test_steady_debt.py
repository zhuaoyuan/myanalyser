"""steady_debt 稳健型策略包 - 单元测试（正常/异常/边界场景）。

依据 docs/需求日志/20260316_steady_debt_稳健型策略包.md 及
docs/参考/分类型的硬约束和主次目标.md 第二节「稳健型」。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ws = Path(__file__).resolve().parents[2]
if str(_ws) not in sys.path:
    sys.path.insert(0, str(_ws))
_src = _ws / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# ==================== 场景清单 ====================
# 正常场景：
#   N1. filter_one 三条硬约束全满足 -> 通过（is_filtered=False）
#   N2. registry 可获取 steady_debt 且使用 SteadyDebtFilterStrategy
#   N3. build_bundle 返回完整 StrategyBundle
#   N4. SteadyDebtScoreStrategy 对通过 filter 的 symbols 正确算分
#   N5. SteadyDebtFilterStrategy 对满足条件的 universe 正确过滤
#
# 异常场景：
#   E1. filter_one 输入空 dict -> 三个约束均缺，应过滤
#   E2. filter_one 输入 None 值 -> 缺失 reason
#   E3. filter_one 输入 NaN -> 按缺失处理
#   E4. filter_one 输入非数值（字符串）-> 按缺失处理
#   E5. 未知策略名 -> ValueError
#   E6. SteadyDebtFilterStrategy 对 data.by_symbol 中无数据/空数据 symbol 跳过
#
# 边界条件：
#   B1. 最大回撤临界 -0.08：刚好等于通过，-0.0801 过滤
#   B2. 年化临界 0.05：刚好等于通过，0.0499 过滤
#   B3. 夏普临界 0.5：刚好等于通过，0.49 过滤
#   B4. 多条件同时不满足 -> 合并 reason
#   B5. 极大值/极小值（max_dd=0, ann=1.0, sharpe=10）-> 通过
#   B6. 窗口不足 3 年 -> _compute_steady_debt_metrics 返回空，symbol 被跳过


# ==================== 正常场景 ====================


class TestSteadyDebtNormal:
    """正常场景。"""

    def test_filter_one_all_constraints_met_passes(self) -> None:
        """N1: 三条硬约束全满足 -> 通过。"""
        from steady_debt_logic import filter_one

        is_filtered, reason = filter_one({
            "近3年最大回撤率": -0.05,
            "近3年年化收益率": 0.06,
            "近3年夏普比率": 0.8,
        })
        assert is_filtered is False
        assert reason == ""

    def test_registry_steady_debt_uses_steady_debt_filter(self) -> None:
        """N2: registry 可获取 steady_debt 且使用 SteadyDebtFilterStrategy。"""
        from myanalyser.src.backtest.filters import SteadyDebtFilterStrategy
        from myanalyser.src.backtest.strategies.registry import (
            get_strategy_bundle,
            list_strategy_names,
        )

        bundle = get_strategy_bundle("steady_debt")
        assert bundle.name == "steady_debt"
        assert isinstance(bundle.filter_strategy, SteadyDebtFilterStrategy)
        assert "steady_debt" in list_strategy_names()

    def test_build_bundle_returns_complete_bundle(self) -> None:
        """N3: build_bundle 返回完整 StrategyBundle。"""
        from myanalyser.src.backtest.filters import SteadyDebtFilterStrategy
        from myanalyser.src.backtest.strategies.steady_debt import (
            SteadyDebtScoreStrategy,
            build_bundle_steady_debt,
        )
        from myanalyser.src.backtest.strategies.low_risk_debt import EqualWeightPosition

        bundle = build_bundle_steady_debt()
        assert bundle.name == "steady_debt"
        assert isinstance(bundle.filter_strategy, SteadyDebtFilterStrategy)
        assert isinstance(bundle.score_strategy, SteadyDebtScoreStrategy)
        assert isinstance(bundle.position_strategy, EqualWeightPosition)

    def test_score_strategy_produces_ranking(self) -> None:
        """N4: SteadyDebtScoreStrategy 对 symbols 正确算分并排序。"""
        from myanalyser.src.backtest.data import BacktestData
        from myanalyser.src.backtest.strategies.steady_debt import SteadyDebtScoreStrategy

        # 生成足够 3 年窗口的净值（729 交易日）
        cfg = 243 * 3
        dates = pd.date_range("2020-01-01", periods=cfg, freq="B")
        nav_up = 1.0 + np.linspace(0, 0.15, len(dates))
        df = pd.DataFrame({"date": dates, "close": nav_up})
        data = BacktestData(
            long_df=df.assign(symbol="000001", open=df["close"], high=df["close"], low=df["close"]),
            by_symbol={"000001": df},
            trading_dates=list(dates),
        )
        strategy = SteadyDebtScoreStrategy()
        scored = strategy.score(data, dates[-1], ["000001"])
        assert not scored.empty
        assert "综合得分" in scored.columns
        assert "综合排名" in scored.columns
        assert scored.iloc[0]["symbol"] == "000001"

    def test_filter_strategy_passes_qualified_symbols(self) -> None:
        """N5: SteadyDebtFilterStrategy 对满足条件的 symbol 通过。"""
        from unittest.mock import patch

        from myanalyser.src.backtest.data import BacktestData
        from myanalyser.src.backtest.filters import SteadyDebtFilterStrategy

        dates = pd.date_range("2020-01-01", periods=800, freq="B")
        df = pd.DataFrame({"date": dates, "close": 1.0 + np.linspace(0, 0.2, len(dates))})
        data = BacktestData(
            long_df=df.assign(symbol="000001", open=df["close"], high=df["close"], low=df["close"]),
            by_symbol={"000001": df},
            trading_dates=list(dates),
        )
        # 正常计算应满足稳健型约束（上升趋势足够）
        f = SteadyDebtFilterStrategy()
        out = f.filter_symbols(data, dates[-1], ["000001"])
        assert "000001" in out


# ==================== 异常场景 ====================


class TestSteadyDebtException:
    """异常场景。"""

    def test_filter_one_empty_dict_filtered(self) -> None:
        """E1: 空 dict -> 三个约束均缺，应过滤。"""
        from steady_debt_logic import filter_one

        is_filtered, reason = filter_one({})
        assert is_filtered is True
        assert "回撤" in reason or "年化" in reason or "夏普" in reason

    def test_filter_one_none_values_filtered(self) -> None:
        """E2: 关键指标为 None -> 按缺失过滤。"""
        from steady_debt_logic import filter_one

        is_filtered, reason = filter_one({
            "近3年最大回撤率": None,
            "近3年年化收益率": None,
            "近3年夏普比率": None,
        })
        assert is_filtered is True
        assert "缺失" in reason or "年化" in reason or "夏普" in reason

    def test_filter_one_nan_treated_as_missing(self) -> None:
        """E3: NaN 按缺失处理。"""
        from steady_debt_logic import filter_one

        is_filtered, _ = filter_one({
            "近3年最大回撤率": float("nan"),
            "近3年年化收益率": 0.06,
            "近3年夏普比率": 0.8,
        })
        assert is_filtered is True

    def test_filter_one_non_numeric_treated_as_missing(self) -> None:
        """E4: 非数值（无法转换的字符串）按缺失处理。"""
        from steady_debt_logic import filter_one

        is_filtered, _ = filter_one({
            "近3年最大回撤率": "N/A",
            "近3年年化收益率": "invalid",
            "近3年夏普比率": 0.8,
        })
        # "N/A"/"invalid" 无法转 float，_to_float 返回 None，应被过滤
        assert is_filtered is True

    def test_registry_unknown_strategy_raises(self) -> None:
        """E5: 未知策略名 -> ValueError。"""
        from myanalyser.src.backtest.strategies.registry import get_strategy_bundle

        with pytest.raises(ValueError, match="未知策略包"):
            get_strategy_bundle("nonexistent_steady")

    def test_filter_strategy_skips_missing_symbol_data(self) -> None:
        """E6: 无数据/空数据 symbol 被跳过。"""
        from myanalyser.src.backtest.data import BacktestData
        from myanalyser.src.backtest.filters import SteadyDebtFilterStrategy

        data = BacktestData(
            long_df=pd.DataFrame(),
            by_symbol={},
            trading_dates=[],
        )
        f = SteadyDebtFilterStrategy()
        out = f.filter_symbols(data, pd.Timestamp("2024-01-01"), ["000001", "000002"])
        assert out == []


# ==================== 边界条件 ====================


class TestSteadyDebtBoundary:
    """边界条件。"""

    def test_max_dd_boundary_minus_0_08_passes(self) -> None:
        """B1: max_dd=-0.08 临界通过，-0.0801 过滤。"""
        from steady_debt_logic import filter_one

        passed, _ = filter_one({
            "近3年最大回撤率": -0.08,
            "近3年年化收益率": 0.05,
            "近3年夏普比率": 0.5,
        })
        assert passed is False, "临界值 -0.08 应通过"

        failed, _ = filter_one({
            "近3年最大回撤率": -0.0801,
            "近3年年化收益率": 0.05,
            "近3年夏普比率": 0.5,
        })
        assert failed is True, "-0.0801 应被过滤"

    def test_ann_return_boundary_0_05_passes(self) -> None:
        """B2: 年化 0.05 临界通过，0.0499 过滤。"""
        from steady_debt_logic import filter_one

        passed, _ = filter_one({
            "近3年最大回撤率": -0.05,
            "近3年年化收益率": 0.05,
            "近3年夏普比率": 0.6,
        })
        assert passed is False, "年化 0.05 临界应通过"

        failed, _ = filter_one({
            "近3年最大回撤率": -0.05,
            "近3年年化收益率": 0.0499,
            "近3年夏普比率": 0.6,
        })
        assert failed is True, "年化 0.0499 应被过滤"

    def test_sharpe_boundary_0_5_passes(self) -> None:
        """B3: 夏普 0.5 临界通过，0.49 过滤。"""
        from steady_debt_logic import filter_one

        passed, _ = filter_one({
            "近3年最大回撤率": -0.05,
            "近3年年化收益率": 0.06,
            "近3年夏普比率": 0.5,
        })
        assert passed is False, "夏普 0.5 临界应通过"

        failed, _ = filter_one({
            "近3年最大回撤率": -0.05,
            "近3年年化收益率": 0.06,
            "近3年夏普比率": 0.49,
        })
        assert failed is True, "夏普 0.49 应被过滤"

    def test_multiple_reasons_combined(self) -> None:
        """B4: 多条件同时不满足 -> 合并 reason。"""
        from steady_debt_logic import filter_one

        is_filtered, reason = filter_one({
            "近3年最大回撤率": -0.12,
            "近3年年化收益率": 0.03,
            "近3年夏普比率": 0.3,
        })
        assert is_filtered is True
        assert "回撤" in reason
        assert "年化" in reason
        assert "夏普" in reason

    def test_extreme_values_pass(self) -> None:
        """B5: 极大值（优质）应通过。"""
        from steady_debt_logic import filter_one

        passed, _ = filter_one({
            "近3年最大回撤率": 0.0,
            "近3年年化收益率": 1.0,
            "近3年夏普比率": 10.0,
        })
        assert passed is False, "优质基金应通过"

    def test_score_empty_symbols_returns_empty_df(self) -> None:
        """B6: 空 symbols 返回空 DataFrame。"""
        from myanalyser.src.backtest.data import BacktestData
        from myanalyser.src.backtest.strategies.steady_debt import SteadyDebtScoreStrategy

        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        df = pd.DataFrame({"date": dates, "close": 1.0 + np.linspace(0, 0.05, len(dates))})
        data = BacktestData(
            long_df=df.assign(symbol="000001", open=df["close"], high=df["close"], low=df["close"]),
            by_symbol={"000001": df},
            trading_dates=list(dates),
        )
        strategy = SteadyDebtScoreStrategy()
        scored = strategy.score(data, dates[-1], [])
        assert scored.empty
        assert "综合得分" in scored.columns
