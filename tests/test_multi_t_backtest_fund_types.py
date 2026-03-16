# -*- coding: utf-8 -*-
"""multi_t_backtest --fund-types 参数单元测试。

需求来源：20260316_multi_t_backtest_基金类型筛选参数.md
测试覆盖：_load_type_filtered_codes、_build_purchase_csv_for_filter、argparse --fund-types
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pandas as pd
import pytest


# 场景清单（供总结用）
# 正常场景：N01-N06
# 异常场景：E01-E08
# 边界条件：B01-B05


class TestLoadTypeFilteredCodes:
    """_load_type_filtered_codes 单元测试"""

    def test_N01_fee_csv_exists_types_match_returns_codes(self) -> None:
        """正常：CSV 存在，指定类型有匹配，返回基金编码集合"""
        from multi_t_backtest import _load_type_filtered_codes

        with tempfile.TemporaryDirectory() as d:
            prep = Path(d)
            fee_csv = prep / "fund_fee_filtered.csv"
            pd.DataFrame(
                {
                    "类型": ["A类730天", "A类730天", "C类30天"],
                    "基金编码": ["100016", "100035", "110007"],
                    "申购费率": ["0.15%", "0.08%", "0%"],
                    "赎回费率": ["0%", "0%", "0%"],
                }
            ).to_csv(fee_csv, index=False, encoding="utf-8-sig")

            result = _load_type_filtered_codes(prep, ["A类730天"])
            assert result == {"100016", "100035"}

    def test_N02_multiple_types_all_match(self) -> None:
        """正常：多种类型均有匹配"""
        from multi_t_backtest import _load_type_filtered_codes

        with tempfile.TemporaryDirectory() as d:
            prep = Path(d)
            fee_csv = prep / "fund_fee_filtered.csv"
            pd.DataFrame(
                {
                    "类型": ["A类730天", "C类30天", "A类180天"],
                    "基金编码": ["100016", "110007", "110019"],
                    "申购费率": ["0.15%", "0%", "0.12%"],
                    "赎回费率": ["0%", "0%", "0%"],
                }
            ).to_csv(fee_csv, index=False, encoding="utf-8-sig")

            result = _load_type_filtered_codes(prep, ["A类730天", "C类30天"])
            assert result == {"100016", "110007"}

    def test_E01_fee_csv_not_exists_raises(self) -> None:
        """异常：fund_fee_filtered.csv 不存在 -> FileNotFoundError"""
        from multi_t_backtest import _load_type_filtered_codes

        with tempfile.TemporaryDirectory() as d:
            prep = Path(d)
            with pytest.raises(FileNotFoundError, match="fund_fee_filtered.csv"):
                _load_type_filtered_codes(prep, ["A类730天"])

    def test_E02_fee_csv_missing_type_column_raises(self) -> None:
        """异常：CSV 缺少「类型」列 -> ValueError"""
        from multi_t_backtest import _load_type_filtered_codes

        with tempfile.TemporaryDirectory() as d:
            prep = Path(d)
            fee_csv = prep / "fund_fee_filtered.csv"
            pd.DataFrame({"基金编码": ["100016"], "申购费率": ["0.15%"]}).to_csv(
                fee_csv, index=False, encoding="utf-8-sig"
            )
            with pytest.raises(ValueError, match="缺少 类型 或 基金编码"):
                _load_type_filtered_codes(prep, ["A类730天"])

    def test_E03_fee_csv_missing_code_column_raises(self) -> None:
        """异常：CSV 缺少「基金编码」列 -> ValueError"""
        from multi_t_backtest import _load_type_filtered_codes

        with tempfile.TemporaryDirectory() as d:
            prep = Path(d)
            fee_csv = prep / "fund_fee_filtered.csv"
            pd.DataFrame({"类型": ["A类730天"], "申购费率": ["0.15%"]}).to_csv(
                fee_csv, index=False, encoding="utf-8-sig"
            )
            with pytest.raises(ValueError, match="缺少 类型 或 基金编码"):
                _load_type_filtered_codes(prep, ["A类730天"])

    def test_B01_type_with_whitespace_stripped(self) -> None:
        """边界：类型列有前后空格，应能匹配"""
        from multi_t_backtest import _load_type_filtered_codes

        with tempfile.TemporaryDirectory() as d:
            prep = Path(d)
            fee_csv = prep / "fund_fee_filtered.csv"
            pd.DataFrame(
                {
                    "类型": [" A类730天 ", "C类30天"],
                    "基金编码": ["100016", "110007"],
                    "申购费率": ["0.15%", "0%"],
                    "赎回费率": ["0%", "0%"],
                }
            ).to_csv(fee_csv, index=False, encoding="utf-8-sig")

            result = _load_type_filtered_codes(prep, ["A类730天"])
            assert result == {"100016"}

    def test_B02_codes_normalized_to_zfill6(self) -> None:
        """边界：基金编码应统一为 6 位（zfill）"""
        from multi_t_backtest import _load_type_filtered_codes

        with tempfile.TemporaryDirectory() as d:
            prep = Path(d)
            fee_csv = prep / "fund_fee_filtered.csv"
            pd.DataFrame(
                {
                    "类型": ["A类730天", "A类730天"],
                    "基金编码": ["16", "100035"],  # 16 -> 000016
                    "申购费率": ["0.15%", "0.08%"],
                    "赎回费率": ["0%", "0%"],
                }
            ).to_csv(fee_csv, index=False, encoding="utf-8-sig")

            result = _load_type_filtered_codes(prep, ["A类730天"])
            assert "000016" in result
            assert "100035" in result


class TestBuildPurchaseCsvForFilter:
    """_build_purchase_csv_for_filter 单元测试"""

    def test_N03_fund_types_empty_returns_eligible_csv(self) -> None:
        """正常：fund_types 为空时直接返回 eligible_csv"""
        from multi_t_backtest import _build_purchase_csv_for_filter

        with tempfile.TemporaryDirectory() as d:
            eligible_csv = Path(d) / "eligible.csv"
            filter_dir = Path(d) / "filter"
            pd.DataFrame({"基金编码": ["100016"], "其他": ["x"]}).to_csv(
                eligible_csv, index=False, encoding="utf-8-sig"
            )
            result = _build_purchase_csv_for_filter(
                eligible_csv=eligible_csv,
                filter_dir=filter_dir,
                fund_types=[],
                type_allowed_codes=None,
                logger=logging.getLogger("test"),
            )
            assert result == eligible_csv

    def test_N04_fund_types_with_intersection_writes_csv(self) -> None:
        """正常：fund_types 非空，与 eligible 有交集，写入 filter_dir 并返回路径"""
        from multi_t_backtest import _build_purchase_csv_for_filter

        with tempfile.TemporaryDirectory() as d:
            eligible_csv = Path(d) / "eligible.csv"
            filter_dir = Path(d) / "filter"
            pd.DataFrame(
                {"基金编码": ["100016", "100035", "999999"], "其他": ["a", "b", "c"]}
            ).to_csv(eligible_csv, index=False, encoding="utf-8-sig")

            result = _build_purchase_csv_for_filter(
                eligible_csv=eligible_csv,
                filter_dir=filter_dir,
                fund_types=["A类730天"],
                type_allowed_codes={"100016", "100035"},
                logger=logging.getLogger("test"),
            )
            out_csv = filter_dir / "eligible_by_type.csv"
            assert result == out_csv
            assert out_csv.exists()
            df = pd.read_csv(out_csv, dtype=str, encoding="utf-8-sig")
            assert set(df["基金编码"].tolist()) == {"100016", "100035"}
            assert len(df) == 2

    def test_N05_eligible_uses_基金代码_column(self) -> None:
        """正常：eligible 使用「基金代码」列（无基金编码）"""
        from multi_t_backtest import _build_purchase_csv_for_filter

        with tempfile.TemporaryDirectory() as d:
            eligible_csv = Path(d) / "eligible.csv"
            filter_dir = Path(d) / "filter"
            pd.DataFrame(
                {"基金代码": ["100016", "100035"], "其他": ["a", "b"]}
            ).to_csv(eligible_csv, index=False, encoding="utf-8-sig")

            result = _build_purchase_csv_for_filter(
                eligible_csv=eligible_csv,
                filter_dir=filter_dir,
                fund_types=["A类730天"],
                type_allowed_codes={"100016", "100035"},
                logger=logging.getLogger("test"),
            )
            df = pd.read_csv(result, dtype=str, encoding="utf-8-sig")
            assert "基金代码" in df.columns
            assert len(df) == 2

    def test_E04_type_allowed_codes_empty_raises(self) -> None:
        """异常：指定类型在 CSV 中无匹配（type_allowed_codes 为空）-> ValueError"""
        from multi_t_backtest import _build_purchase_csv_for_filter

        with tempfile.TemporaryDirectory() as d:
            eligible_csv = Path(d) / "eligible.csv"
            filter_dir = Path(d) / "filter"
            pd.DataFrame({"基金编码": ["100016"]}).to_csv(
                eligible_csv, index=False, encoding="utf-8-sig"
            )
            with pytest.raises(ValueError, match="无匹配基金"):
                _build_purchase_csv_for_filter(
                    eligible_csv=eligible_csv,
                    filter_dir=filter_dir,
                    fund_types=["不存在类型"],
                    type_allowed_codes=set(),
                    logger=logging.getLogger("test"),
                )

    def test_E05_eligible_intersection_empty_raises(self) -> None:
        """异常：eligible 与 type_allowed_codes 取交后为空 -> ValueError"""
        from multi_t_backtest import _build_purchase_csv_for_filter

        with tempfile.TemporaryDirectory() as d:
            eligible_csv = Path(d) / "eligible.csv"
            filter_dir = Path(d) / "filter"
            pd.DataFrame({"基金编码": ["100016", "100035"]}).to_csv(
                eligible_csv, index=False, encoding="utf-8-sig"
            )
            with pytest.raises(ValueError, match="取交后为空"):
                _build_purchase_csv_for_filter(
                    eligible_csv=eligible_csv,
                    filter_dir=filter_dir,
                    fund_types=["A类730天"],
                    type_allowed_codes={"999999", "888888"},  # 与 eligible 无交集
                    logger=logging.getLogger("test"),
                )

    def test_E06_eligible_missing_code_columns_raises(self) -> None:
        """异常：eligible 既无「基金编码」也无「基金代码」-> ValueError"""
        from multi_t_backtest import _build_purchase_csv_for_filter

        with tempfile.TemporaryDirectory() as d:
            eligible_csv = Path(d) / "eligible.csv"
            filter_dir = Path(d) / "filter"
            pd.DataFrame({"其他列": ["x"]}).to_csv(
                eligible_csv, index=False, encoding="utf-8-sig"
            )
            with pytest.raises(ValueError, match="缺少 基金编码 或 基金代码"):
                _build_purchase_csv_for_filter(
                    eligible_csv=eligible_csv,
                    filter_dir=filter_dir,
                    fund_types=["A类730天"],
                    type_allowed_codes={"100016"},
                    logger=logging.getLogger("test"),
                )

    def test_B03_eligible_code_with_leading_zeros_normalized(self) -> None:
        """边界：eligible 中基金编码格式不一，应 zfill(6) 后匹配"""
        from multi_t_backtest import _build_purchase_csv_for_filter

        with tempfile.TemporaryDirectory() as d:
            eligible_csv = Path(d) / "eligible.csv"
            filter_dir = Path(d) / "filter"
            pd.DataFrame(
                {"基金编码": ["16", "  100035  "], "其他": ["a", "b"]}
            ).to_csv(eligible_csv, index=False, encoding="utf-8-sig")

            result = _build_purchase_csv_for_filter(
                eligible_csv=eligible_csv,
                filter_dir=filter_dir,
                fund_types=["A类730天"],
                type_allowed_codes={"000016", "100035"},
                logger=logging.getLogger("test"),
            )
            df = pd.read_csv(result, dtype=str, encoding="utf-8-sig")
            assert len(df) == 2

    def test_B04_type_allowed_codes_none_with_empty_fund_types_returns_eligible(
        self,
    ) -> None:
        """边界：type_allowed_codes=None 且 fund_types 非空（理论上不应出现，实现上会返回 eligible）"""
        from multi_t_backtest import _build_purchase_csv_for_filter

        with tempfile.TemporaryDirectory() as d:
            eligible_csv = Path(d) / "eligible.csv"
            filter_dir = Path(d) / "filter"
            pd.DataFrame({"基金编码": ["100016"]}).to_csv(
                eligible_csv, index=False, encoding="utf-8-sig"
            )
            result = _build_purchase_csv_for_filter(
                eligible_csv=eligible_csv,
                filter_dir=filter_dir,
                fund_types=["A类730天"],
                type_allowed_codes=None,
                logger=logging.getLogger("test"),
            )
            assert result == eligible_csv


class TestArgparseFundTypes:
    """argparse --fund-types 参数测试"""

    def test_N06_cli_fund_types_option_exists(self) -> None:
        """正常：--fund-types 选项存在且帮助信息正确"""
        import subprocess

        result = subprocess.run(
            [
                "python",
                "-m",
                "multi_t_backtest",
                "--help",
            ],
            cwd=str(Path(__file__).resolve().parent.parent / "tools" / "v2"),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--fund-types" in result.stdout
        assert "基金类型" in result.stdout or "fund" in result.stdout.lower()
