"""scoreboard_metrics 共享模块单测。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scoreboard_metrics import (
    METRIC_DIRECTIONS,
    compute_metrics,
    load_nav_df,
    safe_code,
    window_metrics,
)


class ScoreboardMetricsTest(unittest.TestCase):
    def test_safe_code(self) -> None:
        self.assertEqual(safe_code("1"), "000001")
        self.assertEqual(safe_code(163402), "163402")
        self.assertEqual(safe_code("  163402  "), "163402")

    def test_load_nav_df_empty_missing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(load_nav_df(Path(d) / "nonexistent.csv").empty)

    def test_load_nav_df_valid(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "000001.csv"
            pd.DataFrame(
                [{"基金代码": "000001", "净值日期": "2024-01-01", "复权净值": 1.0}],
            ).to_csv(p, index=False, encoding="utf-8-sig")
            df = load_nav_df(p)
            self.assertEqual(len(df), 1)
            self.assertIn("净值日期", df.columns)
            self.assertIn("复权净值", df.columns)

    def test_compute_metrics_single_point(self) -> None:
        df = pd.DataFrame([{"净值日期": pd.Timestamp("2024-01-01"), "复权净值": 1.0}])
        out = compute_metrics(df, pd.Timestamp("2024-01-01"))
        self.assertIsNone(out["annual_return"])
        # 单点 max_drawdown 为 0（cummax=nav，1-nav/roll_max=0）
        self.assertIn(out["max_drawdown"], (None, 0.0))

    def test_compute_metrics_two_points(self) -> None:
        df = pd.DataFrame([
            {"净值日期": pd.Timestamp("2024-01-01"), "复权净值": 1.0},
            {"净值日期": pd.Timestamp("2024-12-31"), "复权净值": 1.1},
        ])
        out = compute_metrics(df, pd.Timestamp("2024-12-31"))
        self.assertIsNotNone(out["annual_return"])
        self.assertIsNotNone(out["max_drawdown"])
        self.assertIn("max_drawdown_recovery_days", out)
        self.assertIn("max_single_day_drop", out)

    def test_compute_metrics_max_single_day_drop(self) -> None:
        """最大单日跌幅为日收益率最小值（最负值）。"""
        df = pd.DataFrame([
            {"净值日期": pd.Timestamp("2024-01-01"), "复权净值": 1.0},
            {"净值日期": pd.Timestamp("2024-01-02"), "复权净值": 0.95},
            {"净值日期": pd.Timestamp("2024-01-03"), "复权净值": 0.98},
        ])
        out = compute_metrics(df, pd.Timestamp("2024-01-03"))
        self.assertIsNotNone(out["max_single_day_drop"])
        self.assertLess(out["max_single_day_drop"], 0)
        # 1->0.95: -5%, 0.95->0.98: +3.16%
        self.assertAlmostEqual(out["max_single_day_drop"], -0.05, places=4)

    def test_window_metrics_empty(self) -> None:
        df = pd.DataFrame(columns=["净值日期", "复权净值"])
        out = window_metrics(df, pd.Timestamp("2024-12-31"), years=3)
        self.assertEqual(out, {})

    def test_metric_directions_keys(self) -> None:
        self.assertIn("annual_return", METRIC_DIRECTIONS)
        self.assertIn("sharpe_ratio_1y", METRIC_DIRECTIONS)
        self.assertEqual(METRIC_DIRECTIONS["max_drawdown"], "asc")
        self.assertEqual(METRIC_DIRECTIONS["annual_return"], "desc")
