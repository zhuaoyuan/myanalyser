# -*- coding: utf-8 -*-
"""
基金规模变动抓取模块单元测试（pytest）

覆盖：正常场景、异常场景、边界条件。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from fund_gmbd import (
    _normalize_code,
    _parse_num,
    _parse_gmbd_from_content,
    _to_yiyuan,
    _to_pct,
    fetch_gmbd_api,
    fund_gmbd_em,
)


# ============= 1. 正常场景 =============


class TestNormalizeCode:
    """基金编码标准化 - 正常场景"""

    def test_short_code_padded(self):
        assert _normalize_code("198") == "000198"

    def test_full_six_digit(self):
        assert _normalize_code("000198") == "000198"
        assert _normalize_code("519674") == "519674"

    def test_whitespace_stripped(self):
        assert _normalize_code("  110011  ") == "110011"


class TestParseNum:
    """数值解析 - 正常场景"""

    def test_thousand_separator(self):
        assert _parse_num("12,392.31") == 12392.31
        assert _parse_num("7,646.28") == 7646.28

    def test_plain_number(self):
        assert _parse_num("12345.67") == 12345.67
        assert _parse_num("0") == 0.0


class TestToYiyuan:
    """元转亿元 - 正常场景"""

    def test_valid_conversion(self):
        assert _to_yiyuan(1239230971212.3) == 12392.31
        assert _to_yiyuan(764628495753.39) == 7646.28

    def test_zero(self):
        assert _to_yiyuan(0) == 0.0


class TestToPct:
    """净资产变动率格式化 - 正常场景"""

    def test_negative_pct(self):
        assert _to_pct(-3.39) == "-3.39%"

    def test_positive_pct(self):
        assert _to_pct(3.67) == "3.67%"


class TestParseGmbdFromContent:
    """HTML 解析 - 正常场景"""

    def test_empty_html_returns_empty_df(self):
        df = _parse_gmbd_from_content("")
        assert df.empty
        df = _parse_gmbd_from_content("<div>no table</div>")
        assert df.empty

    def test_valid_table_parsed(self):
        html = """
        <table class="gmbd">
        <thead><tr><th>日期</th><th>期间申购（亿份）</th><th>期间赎回（亿份）</th><th>期末总份额（亿份）</th><th>期末净资产（亿元）</th><th>净资产变动率</th></tr></thead>
        <tbody>
        <tr><td>2025-12-31</td><td>12,392.31</td><td>12,660.77</td><td>7,646.28</td><td>7,646.28</td><td>-3.39%</td></tr>
        <tr><td>2019-06-28</td><td>---</td><td>---</td><td>---</td><td>10,334.35</td><td>-0.17%</td></tr>
        </tbody>
        </table>
        """
        df = _parse_gmbd_from_content(html)
        assert len(df) == 2
        assert list(df.columns) == [
            "日期", "期间申购（亿份）", "期间赎回（亿份）",
            "期末总份额（亿份）", "期末净资产（亿元）", "净资产变动率",
        ]
        assert df.iloc[0]["日期"] == "2025-12-31"
        assert df.iloc[0]["期间申购（亿份）"] == 12392.31
        assert pd.isna(df.iloc[1]["期间申购（亿份）"])
        assert df.iloc[1]["期末净资产（亿元）"] == 10334.35

    def test_table_without_gmbd_class_fallback(self):
        html = """
        <table>
        <tr><th>日期</th><th>期末净资产（亿元）</th></tr>
        <tr><td>2025-01-01</td><td>1,234.56</td></tr>
        </table>
        """
        df = _parse_gmbd_from_content(html)
        assert not df.empty
        assert len(df) == 1
        assert df.iloc[0]["期末净资产（亿元）"] == 1234.56


class TestFundGmbdEmFromData:
    """fund_gmbd_em - API data 分支正常场景"""

    @patch("fund_gmbd.fetch_gmbd_api")
    def test_from_data_mock(self, mock_fetch):
        mock_fetch.return_value = {
            "content": "",
            "summary": "",
            "data": [
                {
                    "FSRQ": "2025-12-31",
                    "QJSG": 1239230971212.3,
                    "QJSH": 1266076674667.53,
                    "QMJZC": 764628495753.39,
                    "QMZFE": 764628495753.39,
                    "ZFEBDL": -3.39,
                },
            ],
        }
        df = fund_gmbd_em("000198")
        assert not df.empty
        assert len(df) == 1
        assert df.iloc[0]["日期"] == "2025-12-31"
        assert df.iloc[0]["期间申购（亿份）"] == 12392.31
        assert df.iloc[0]["期末净资产（亿元）"] == 7646.28
        assert "-3.39%" in str(df.iloc[0]["净资产变动率"])

    @patch("fund_gmbd.fetch_gmbd_api")
    def test_from_content_mock(self, mock_fetch):
        table_html = """
        <table class="gmbd">
        <thead><tr><th>日期</th><th>期间申购（亿份）</th><th>期间赎回（亿份）</th><th>期末总份额（亿份）</th><th>期末净资产（亿元）</th><th>净资产变动率</th></tr></thead>
        <tbody>
        <tr><td>2025-12-31</td><td>12,392.31</td><td>12,660.77</td><td>7,646.28</td><td>7,646.28</td><td>-3.39%</td></tr>
        </tbody>
        </table>
        """
        mock_fetch.return_value = {"content": table_html, "summary": "", "data": []}
        df = fund_gmbd_em("000198")
        assert not df.empty
        assert len(df) == 1
        assert df.iloc[0]["期末净资产（亿元）"] == 7646.28


class TestLocalExample:
    """本地 HTML 样例校验"""

    def test_parse_gmbd_example_html(self):
        example_path = Path(__file__).resolve().parents[1] / "tmp" / "gmbd_example.html"
        if not example_path.exists():
            pytest.skip("本地样例 gmbd_example.html 不存在")
        from bs4 import BeautifulSoup

        html = example_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")
        div = soup.find("div", id="gmbdtable")
        if not div or not div.find("table"):
            pytest.skip("gmbd_example.html 中未找到 gmbdtable 表格")
        table_html = str(div.find("table"))
        df = _parse_gmbd_from_content(table_html)
        assert not df.empty
        assert "日期" in df.columns
        assert len(df) >= 50
        assert df.iloc[0]["日期"] == "2025-12-31"
        assert df.iloc[0]["期末净资产（亿元）"] == 7646.28


# ============= 2. 异常场景 =============


class TestParseNumException:
    """_parse_num 异常/空值"""

    def test_dash_returns_none(self):
        assert _parse_num("---") is None

    def test_empty_string(self):
        assert _parse_num("") is None
        assert _parse_num("   ") is None

    def test_none_input(self):
        assert _parse_num(None) is None

    def test_invalid_string(self):
        assert _parse_num("abc") is None
        assert _parse_num("N/A") is None


class TestToYiyuanException:
    """_to_yiyuan 异常输入"""

    def test_empty_and_dash(self):
        assert _to_yiyuan(None) is None
        assert _to_yiyuan("") is None
        assert _to_yiyuan("---") is None

    def test_invalid_type(self):
        assert _to_yiyuan("abc") is None


class TestToPctException:
    """_to_pct 异常输入"""

    def test_empty(self):
        assert _to_pct(None) is None
        assert _to_pct("") is None


class TestFundGmbdEmException:
    """fund_gmbd_em 异常场景"""

    @patch("fund_gmbd.fetch_gmbd_api")
    def test_empty_api_response(self, mock_fetch):
        mock_fetch.return_value = {"content": "", "summary": "", "data": []}
        df = fund_gmbd_em("000000")
        assert df.empty
        assert list(df.columns) == [
            "日期", "期间申购（亿份）", "期间赎回（亿份）",
            "期末总份额（亿份）", "期末净资产（亿元）", "净资产变动率",
        ]

    @patch("fund_gmbd.fetch_gmbd_api")
    def test_network_error_propagates(self, mock_fetch):
        mock_fetch.side_effect = Exception("Connection timeout")
        with pytest.raises(Exception, match="Connection timeout"):
            fund_gmbd_em("000198")


class TestFetchGmbdApiException:
    """fetch_gmbd_api 异常/边界"""

    def test_invalid_code_returns_empty(self):
        result = fetch_gmbd_api("")
        assert result == {"content": "", "summary": "", "data": []}

    def test_non_digit_code_returns_empty(self):
        result = fetch_gmbd_api("abc")
        assert result == {"content": "", "summary": "", "data": []}


# ============= 3. 边界条件 =============


class TestNormalizeCodeBoundary:
    """编码标准化边界"""

    def test_empty_string(self):
        assert _normalize_code("") == ""

    def test_non_digit_preserved(self):
        # 非纯数字不补零，只 strip
        assert _normalize_code("abc") == "abc"
        assert _normalize_code("  xyz  ") == "xyz"

    def test_numeric_string_from_int(self):
        # code 可能为 int（如从 CSV 读入）
        assert _normalize_code(198) == "000198"  # type: ignore


class TestParseNumBoundary:
    """_parse_num 边界"""

    def test_zero(self):
        assert _parse_num("0") == 0.0

    def test_negative(self):
        assert _parse_num("-123.45") == -123.45

    def test_large_number(self):
        assert _parse_num("12,345,678,901.23") == 12345678901.23


class TestToYiyuanBoundary:
    """_to_yiyuan 边界"""

    def test_very_small(self):
        # 小于 1e8 的数值，round 后可能为 0.01
        assert _to_yiyuan(1000000) == 0.01

    def test_string_numeric(self):
        # API 可能返回字符串形式的数字
        assert _to_yiyuan("1239230971212.3") == 12392.31


class TestFundGmbdEmBoundary:
    """fund_gmbd_em 边界条件"""

    @patch("fund_gmbd.fetch_gmbd_api")
    def test_invalid_code_empty_columns(self, mock_fetch):
        df = fund_gmbd_em("")
        assert df.empty
        assert list(df.columns) == [
            "日期", "期间申购（亿份）", "期间赎回（亿份）",
            "期末总份额（亿份）", "期末净资产（亿元）", "净资产变动率",
        ]
        mock_fetch.assert_not_called()

    @patch("fund_gmbd.fetch_gmbd_api")
    def test_money_fund_qmjzc_fallback(self, mock_fetch):
        """货币基金：QMJZC 为空时用 QMZFE 填充期末总份额"""
        mock_fetch.return_value = {
            "content": "",
            "summary": "",
            "data": [
                {
                    "FSRQ": "2025-12-31",
                    "QJSG": 100000000000,
                    "QJSH": 100000000000,
                    "QMJZC": None,  # 空
                    "QMZFE": 1033435000000,  # 仅净资产
                    "ZFEBDL": -0.17,
                },
            ],
        }
        df = fund_gmbd_em("000198")
        assert not df.empty
        assert df.iloc[0]["期末总份额（亿份）"] == 10334.35
        assert df.iloc[0]["期末净资产（亿元）"] == 10334.35

    @patch("fund_gmbd.fetch_gmbd_api")
    def test_change_fallback_when_zfebdl_missing(self, mock_fetch):
        """ZFEBDL 缺失时使用 CHANGE 字段"""
        mock_fetch.return_value = {
            "content": "",
            "summary": "",
            "data": [
                {
                    "FSRQ": "2025-12-31",
                    "QJSG": 100000000000,
                    "QJSH": 100000000000,
                    "QMJZC": 100000000000,
                    "QMZFE": 100000000000,
                    "ZFEBDL": None,
                    "CHANGE": "-1.5%",
                },
            ],
        }
        df = fund_gmbd_em("000198")
        assert df.iloc[0]["净资产变动率"] == "-1.5%"

    @patch("fund_gmbd.fetch_gmbd_api")
    def test_partial_null_in_row(self, mock_fetch):
        """行内部分字段为 null"""
        mock_fetch.return_value = {
            "content": "",
            "summary": "",
            "data": [
                {
                    "FSRQ": "2025-12-31",
                    "QJSG": None,
                    "QJSH": None,
                    "QMJZC": 764628495753.39,
                    "QMZFE": 764628495753.39,
                    "ZFEBDL": -3.39,
                },
            ],
        }
        df = fund_gmbd_em("000198")
        assert not df.empty
        assert pd.isna(df.iloc[0]["期间申购（亿份）"])
        assert pd.isna(df.iloc[0]["期间赎回（亿份）"])
        assert df.iloc[0]["期末净资产（亿元）"] == 7646.28


class TestParseGmbdContentBoundary:
    """HTML 解析边界"""

    def test_table_row_cell_count_mismatch_skipped(self):
        """td 数量与 th 不一致的行应被跳过"""
        html = """
        <table class="gmbd">
        <tr><th>A</th><th>B</th></tr>
        <tr><td>1</td><td>2</td><td>3</td></tr>
        <tr><td>4</td></tr>
        <tr><td>5</td><td>6</td></tr>
        </table>
        """
        df = _parse_gmbd_from_content(html)
        assert len(df) == 1
        assert df.iloc[0]["A"] == "5"
        assert df.iloc[0]["B"] == "6"


# ============= 4. CLI 测试 =============


class TestFetchFundGmbdCLI:
    """fetch_fund_gmbd CLI 单元测试"""

    def test_load_codes_from_csv_success(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8-sig"
        ) as f:
            f.write("基金代码,名称\n000198,余额宝\n110011,易方达\n")
            csv_path = Path(f.name)
        try:
            from fetch_fund_gmbd import _load_codes_from_csv

            codes = _load_codes_from_csv(csv_path)
            assert "000198" in codes
            assert "110011" in codes
            assert len(codes) == 2
        finally:
            csv_path.unlink()

    def test_load_codes_from_csv_missing_column_raises(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8-sig"
        ) as f:
            f.write("code,name\n000198,a\n")
            csv_path = Path(f.name)
        try:
            from fetch_fund_gmbd import _load_codes_from_csv

            with pytest.raises(ValueError, match="缺少列"):
                _load_codes_from_csv(csv_path, code_col="基金代码")
        finally:
            csv_path.unlink()

    def test_load_codes_from_csv_dedup(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8-sig"
        ) as f:
            f.write("基金代码\n000198\n000198\n110011\n")
            csv_path = Path(f.name)
        try:
            from fetch_fund_gmbd import _load_codes_from_csv

            codes = _load_codes_from_csv(csv_path)
            assert codes == ["000198", "110011"]
        finally:
            csv_path.unlink()
