"""fund_metrics_core 单测：持仓期指标及卡玛/溃疡绩效指数 12 月门槛。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fund_metrics_core import WindowConfig, compute_holding_period_metrics


def _make_dates(n: int, base: str = "2025-01-01") -> np.ndarray:
    return np.array(
        [np.datetime64(base) + np.timedelta64(i, "D") for i in range(n)],
        dtype="datetime64[D]",
    )


class TestComputeHoldingPeriodMetricsCalmar(unittest.TestCase):
    """卡玛比率 12 个月门槛：持仓期不足 12 月时返回说明字符串。"""

    def test_calmar_short_period_returns_note(self) -> None:
        """持仓 < 12 月：卡玛比率为「样本不足，不计算卡玛」。"""
        n = 40
        dates = _make_dates(n)
        prices = np.linspace(1.0, 1.01, n)
        prices[20:25] *= 0.995  # 小回撤
        out = compute_holding_period_metrics(dates, prices, config=WindowConfig(trading_days_per_year=243))
        self.assertEqual(out["卡玛比率"], "样本不足，不计算卡玛")

    def test_calmar_long_period_returns_float(self) -> None:
        """持仓 >= 12 月：卡玛比率为浮点数。"""
        n = 250
        dates = _make_dates(n, "2024-01-01")
        prices = np.linspace(1.0, 1.05, n)
        prices[100:120] *= 0.98  # 约 -2% 回撤
        out = compute_holding_period_metrics(dates, prices, config=WindowConfig(trading_days_per_year=243))
        self.assertIsInstance(out["卡玛比率"], (int, float))
        self.assertIsNotNone(out["卡玛比率"])
        self.assertGreater(out["卡玛比率"], 0)


class TestComputeHoldingPeriodMetricsUPI(unittest.TestCase):
    """溃疡绩效指数 12 个月门槛：持仓期不足 12 月时返回说明字符串。"""

    def test_upi_short_period_returns_note(self) -> None:
        """持仓 < 12 月：溃疡绩效指数为「样本不足，不计算溃疡绩效指数」。"""
        n = 40
        dates = _make_dates(n)
        prices = np.linspace(1.0, 1.01, n)
        prices[20:25] *= 0.995  # 有回撤才有溃疡指数
        out = compute_holding_period_metrics(dates, prices, config=WindowConfig(trading_days_per_year=243))
        self.assertEqual(out["溃疡绩效指数"], "样本不足，不计算溃疡绩效指数")

    def test_upi_long_period_returns_float(self) -> None:
        """持仓 >= 12 月：溃疡绩效指数为浮点数。"""
        n = 250
        dates = _make_dates(n, "2024-01-01")
        prices = np.linspace(1.0, 1.05, n)
        prices[100:120] *= 0.98  # 有回撤才有溃疡指数
        out = compute_holding_period_metrics(dates, prices, config=WindowConfig(trading_days_per_year=243))
        self.assertIsInstance(out["溃疡绩效指数"], (int, float))
        self.assertIsNotNone(out["溃疡绩效指数"])
        self.assertGreater(out["溃疡绩效指数"], 0)
