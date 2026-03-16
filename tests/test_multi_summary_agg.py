# -*- coding: utf-8 -*-
"""multi_summary_agg.csv 生成逻辑单元测试。

需求来源：20260316_multi_summary_agg 需求文案
测试覆盖：_write_multi_summary_agg 与 df.describe() 结果一致性、t_count、边界情况。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest


class TestWriteMultiSummaryAgg:
    """_write_multi_summary_agg 单元测试"""

    def test_agg_values_match_describe_tolerance_1e6(self) -> None:
        """汇总指标与 df.describe() 等结果一致，容差 1e-6"""
        from multi_t_backtest import _write_multi_summary_agg

        summary_df = pd.DataFrame({
            "as_of_date": ["2023-01-01", "2023-07-01", "2024-01-02"],
            "filter_start": ["2020-01-01"] * 3,
            "filter_end": ["2023-01-01", "2023-07-01", "2024-01-02"],
            "backtest_start": ["2023-01-01"] * 3,
            "backtest_end": ["2023-02-01", "2023-08-01", "2024-02-01"],
            "allowed_funds": [100, 105, 98],
            "年化收益率": ["0.05", "0.08", "-0.02"],
            "最大回撤率": ["-0.03", "-0.025", "-0.04"],
            "夏普比率": ["1.2", "1.5", "0.9"],
        })
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            _write_multi_summary_agg(summary_df, out)
            agg_path = out / "multi_summary_agg.csv"
            assert agg_path.exists()
            agg_df = pd.read_csv(agg_path, encoding="utf-8-sig")

            numeric_df = summary_df[["年化收益率", "最大回撤率", "夏普比率"]].apply(
                pd.to_numeric, errors="coerce"
            )
            for stat in ("mean", "std", "min", "max"):
                agg_row = agg_df[agg_df["stat_type"] == stat]
                if agg_row.empty:
                    continue
                for col in ["年化收益率", "最大回撤率", "夏普比率"]:
                    actual = agg_row[col].iloc[0]
                    if actual == "" or (isinstance(actual, str) and not actual):
                        expected = numeric_df[col].describe().get(stat)
                        assert pd.isna(expected) or numeric_df[col].isna().all()
                    else:
                        expected = numeric_df[col].describe()[stat]
                        assert abs(float(actual) - float(expected)) < 1e-6, (
                            f"{stat} {col}: {actual} vs {expected}"
                        )

    def test_t_count_equals_summary_rows(self) -> None:
        """t_count 等于 multi_summary.csv 行数"""
        from multi_t_backtest import _write_multi_summary_agg

        summary_df = pd.DataFrame({
            "as_of_date": ["2023-01-01", "2023-07-01"],
            "filter_start": ["2020-01-01"] * 2,
            "filter_end": ["2023-01-01", "2023-07-01"],
            "backtest_start": ["2023-01-01"] * 2,
            "backtest_end": ["2023-02-01", "2023-08-01"],
            "allowed_funds": [100, 105],
            "年化收益率": ["0.05", "0.08"],
        })
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            _write_multi_summary_agg(summary_df, out)
            agg_df = pd.read_csv(out / "multi_summary_agg.csv", encoding="utf-8-sig")
            tc_row = agg_df[agg_df["stat_type"] == "t_count"]
            assert not tc_row.empty
            assert int(tc_row["年化收益率"].iloc[0]) == 2

    def test_empty_summary_skips_agg(self) -> None:
        """summary_df 为空时不生成 agg 文件"""
        from multi_t_backtest import _write_multi_summary_agg

        summary_df = pd.DataFrame()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            _write_multi_summary_agg(summary_df, out)
            assert not (out / "multi_summary_agg.csv").exists()

    def test_single_t_std_empty_or_nan(self) -> None:
        """只有 1 个 T 日时 std 为空或 NaN"""
        from multi_t_backtest import _write_multi_summary_agg

        summary_df = pd.DataFrame({
            "as_of_date": ["2023-01-01"],
            "filter_start": ["2020-01-01"],
            "filter_end": ["2023-01-01"],
            "backtest_start": ["2023-01-01"],
            "backtest_end": ["2023-02-01"],
            "allowed_funds": [100],
            "年化收益率": ["0.05"],
        })
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            _write_multi_summary_agg(summary_df, out)
            agg_df = pd.read_csv(out / "multi_summary_agg.csv", encoding="utf-8-sig")
            std_row = agg_df[agg_df["stat_type"] == "std"]
            assert not std_row.empty
            val = std_row["年化收益率"].iloc[0]
            assert val == "" or (isinstance(val, float) and (val != val or val == 0))

    def test_win_rate_annual_return(self) -> None:
        """win_rate 为年化收益率 > 0 的比例"""
        from multi_t_backtest import _write_multi_summary_agg

        summary_df = pd.DataFrame({
            "as_of_date": ["2023-01-01", "2023-07-01", "2024-01-02"],
            "filter_start": ["2020-01-01"] * 3,
            "filter_end": ["2023-01-01", "2023-07-01", "2024-01-02"],
            "backtest_start": ["2023-01-01"] * 3,
            "backtest_end": ["2023-02-01", "2023-08-01", "2024-02-01"],
            "allowed_funds": [100] * 3,
            "年化收益率": ["0.05", "0.08", "-0.02"],
        })
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            _write_multi_summary_agg(summary_df, out)
            agg_df = pd.read_csv(out / "multi_summary_agg.csv", encoding="utf-8-sig")
            wr_row = agg_df[agg_df["stat_type"] == "win_rate"]
            assert not wr_row.empty
            win_rate = float(wr_row["年化收益率"].iloc[0])
            assert abs(win_rate - 2 / 3) < 1e-6
