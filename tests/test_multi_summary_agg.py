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

from multi_t_backtest import _write_multi_summary_agg


class TestWriteMultiSummaryAgg:
    """_write_multi_summary_agg 单元测试"""

    def test_agg_values_match_describe_tolerance_1e6(self) -> None:
        """汇总指标与 df.describe() 等结果一致，容差 1e-6"""
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
            _stat_map = {"p25": "25%", "median": "50%", "p75": "75%"}
            for stat in ("mean", "median", "std", "min", "max", "p25", "p75", "count"):
                agg_row = agg_df[agg_df["stat_type"] == stat]
                if agg_row.empty:
                    continue
                describe_key = _stat_map.get(stat, stat)
                for col in ["年化收益率", "最大回撤率", "夏普比率"]:
                    actual = agg_row[col].iloc[0]
                    expected = numeric_df[col].describe().get(describe_key, float("nan"))
                    if actual == "" or pd.isna(actual) or (isinstance(actual, str) and not actual):
                        assert pd.isna(expected) or numeric_df[col].isna().all()
                    else:
                        assert abs(float(actual) - float(expected)) < 1e-6, (
                            f"{stat} {col}: {actual} vs {expected}"
                        )

    def test_t_count_equals_summary_rows(self) -> None:
        """t_count 等于 multi_summary.csv 行数"""
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
        summary_df = pd.DataFrame()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            _write_multi_summary_agg(summary_df, out)
            assert not (out / "multi_summary_agg.csv").exists()

    def test_single_t_std_empty_or_nan(self) -> None:
        """只有 1 个 T 日时 std 为空或 NaN"""
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

    def test_empty_string_and_non_numeric_treated_as_nan(self) -> None:
        """空串、非数值按 NaN 处理，不参与 mean/std 等计算"""
        summary_df = pd.DataFrame({
            "as_of_date": ["2023-01-01", "2023-07-01", "2024-01-02"],
            "年化收益率": ["0.05", "", "invalid"],  # 第2行空串、第3行非数值
        })
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            _write_multi_summary_agg(summary_df, out)
            agg_df = pd.read_csv(out / "multi_summary_agg.csv", encoding="utf-8-sig")
            # 仅第1行 0.05 有效，mean=0.05, count=1, std=空
            mean_row = agg_df[agg_df["stat_type"] == "mean"]
            assert abs(float(mean_row["年化收益率"].iloc[0]) - 0.05) < 1e-6
            count_row = agg_df[agg_df["stat_type"] == "count"]
            assert int(count_row["年化收益率"].iloc[0]) == 1
            std_row = agg_df[agg_df["stat_type"] == "std"]
            std_val = std_row["年化收益率"].iloc[0]
            # CSV 写入空串，读回时可能为 NaN（pandas 推断列类型）
            assert std_val == "" or (isinstance(std_val, float) and (pd.isna(std_val) or std_val == 0))

    def test_column_all_nan_agg_empty(self) -> None:
        """某指标列全为 NaN 时，该列 agg 统计为空"""
        summary_df = pd.DataFrame({
            "as_of_date": ["2023-01-01", "2023-07-01"],
            "年化收益率": ["0.05", "0.08"],
            "最大回撤率": [float("nan"), float("nan")],  # 全 NaN
        })
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            _write_multi_summary_agg(summary_df, out)
            agg_df = pd.read_csv(out / "multi_summary_agg.csv", encoding="utf-8-sig")
            mean_row = agg_df[agg_df["stat_type"] == "mean"]
            assert mean_row["年化收益率"].iloc[0] != "" and mean_row["年化收益率"].iloc[0] != "nan"
            # 最大回撤率全 NaN，mean 应为空
            val = mean_row["最大回撤率"].iloc[0]
            assert val == "" or (isinstance(val, float) and pd.isna(val))

    def test_no_metric_columns_skips_agg(self) -> None:
        """summary 仅有 as_of_date 等非 metric 列时，不生成 agg 文件"""
        summary_df = pd.DataFrame({
            "as_of_date": ["2023-01-01", "2023-07-01"],
            "filter_start": ["2020-01-01"] * 2,
            "allowed_funds": [100, 105],
        })
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            _write_multi_summary_agg(summary_df, out)
            assert not (out / "multi_summary_agg.csv").exists()

    def test_win_rate_all_positive(self) -> None:
        """win_rate 全正时结果为 1.0"""
        summary_df = pd.DataFrame({
            "as_of_date": ["2023-01-01", "2023-07-01"],
            "年化收益率": ["0.05", "0.08"],
        })
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            _write_multi_summary_agg(summary_df, out)
            agg_df = pd.read_csv(out / "multi_summary_agg.csv", encoding="utf-8-sig")
            wr_row = agg_df[agg_df["stat_type"] == "win_rate"]
            assert abs(float(wr_row["年化收益率"].iloc[0]) - 1.0) < 1e-6

    def test_win_rate_all_negative(self) -> None:
        """win_rate 全负时结果为 0.0"""
        summary_df = pd.DataFrame({
            "as_of_date": ["2023-01-01", "2023-07-01"],
            "年化收益率": ["-0.05", "-0.08"],
        })
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            _write_multi_summary_agg(summary_df, out)
            agg_df = pd.read_csv(out / "multi_summary_agg.csv", encoding="utf-8-sig")
            wr_row = agg_df[agg_df["stat_type"] == "win_rate"]
            assert abs(float(wr_row["年化收益率"].iloc[0]) - 0.0) < 1e-6

    def test_win_rate_with_zero_boundary(self) -> None:
        """win_rate 边界：年化收益率=0 不计入正收益"""
        summary_df = pd.DataFrame({
            "as_of_date": ["2023-01-01", "2023-07-01", "2024-01-02"],
            "年化收益率": ["0.05", "0", "-0.02"],  # 1 正 1 零 1 负
        })
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            _write_multi_summary_agg(summary_df, out)
            agg_df = pd.read_csv(out / "multi_summary_agg.csv", encoding="utf-8-sig")
            wr_row = agg_df[agg_df["stat_type"] == "win_rate"]
            # (s > 0).sum() = 1, n = 3 -> 1/3
            assert abs(float(wr_row["年化收益率"].iloc[0]) - 1 / 3) < 1e-6

    def test_extreme_values_min_max(self) -> None:
        """极大值、极小值正确计入 min/max"""
        summary_df = pd.DataFrame({
            "as_of_date": ["2023-01-01", "2023-07-01", "2024-01-02"],
            "年化收益率": ["1e10", "0", "-1e10"],
        })
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            _write_multi_summary_agg(summary_df, out)
            agg_df = pd.read_csv(out / "multi_summary_agg.csv", encoding="utf-8-sig")
            min_row = agg_df[agg_df["stat_type"] == "min"]
            max_row = agg_df[agg_df["stat_type"] == "max"]
            assert abs(float(min_row["年化收益率"].iloc[0]) - (-1e10)) < 1e-6
            assert abs(float(max_row["年化收益率"].iloc[0]) - 1e10) < 1e-6

    def test_stat_type_and_column_order(self) -> None:
        """stat_type 与列顺序符合需求"""
        summary_df = pd.DataFrame({
            "as_of_date": ["2023-01-01", "2023-07-01"],
            "年化收益率": ["0.05", "0.08"],
        })
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            _write_multi_summary_agg(summary_df, out)
            agg_df = pd.read_csv(out / "multi_summary_agg.csv", encoding="utf-8-sig")
            expected_stats = ["mean", "median", "std", "min", "max", "p25", "p75", "count", "win_rate", "t_count"]
            actual = agg_df["stat_type"].tolist()
            assert actual == expected_stats
            assert agg_df.columns[0] == "stat_type"

    def test_output_root_must_exist(self) -> None:
        """output_root 必须存在（非 mkdir 场景下会失败）"""
        summary_df = pd.DataFrame({
            "as_of_date": ["2023-01-01"],
            "年化收益率": ["0.05"],
        })
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "nonexistent_subdir"
            assert not out.exists()
            with pytest.raises((FileNotFoundError, OSError)):
                _write_multi_summary_agg(summary_df, out)

