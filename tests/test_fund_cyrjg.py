# -*- coding: utf-8 -*-
"""
基金持有人结构抓取模块单元测试（pytest/unittest）

覆盖：正常场景、异常场景、边界条件。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

_root = Path(__file__).resolve().parent.parent
_src = _root / "src"
_tools = _root / "tools"
for p in (_src, _tools):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fund_cyrjg import (
    _normalize_code,
    _parse_num,
    _parse_cyrjg_from_content,
    _parse_cyrjg_from_html,
    _parse_cyrjg_from_markdown,
    fetch_cyrjg_api,
    fund_cyrjg_em,
)


# ============= 1. 正常场景 =============


class TestNormalizeCode:
    """基金编码标准化"""

    def test_short_code_padded(self):
        assert _normalize_code("15") == "000015"

    def test_full_six_digit(self):
        assert _normalize_code("000015") == "000015"
        assert _normalize_code("000198") == "000198"

    def test_whitespace_stripped(self):
        assert _normalize_code("  000015  ") == "000015"


class TestParseNum:
    """数值解析"""

    def test_thousand_separator(self):
        assert _parse_num("7,932.19") == 7932.19
        assert _parse_num("68.84") == 68.84

    def test_plain_number(self):
        assert _parse_num("46.34") == 46.34
        assert _parse_num("0") == 0.0


class TestParseCyrjgFromContent:
    """Markdown 表格解析"""

    def test_empty_content_returns_empty_df(self):
        assert _parse_cyrjg_from_content("").empty
        assert _parse_cyrjg_from_content("   ").empty

    def test_valid_markdown_table_parsed(self):
        content = """
| 公告日期 | 机构持有比例 | 个人持有比例 | 内部持有比例 | 总份额（亿份） |
| --- | --- | --- | --- | --- |
| 2025-06-30 | 63.45% | 36.55% | 0.01% | 68.84 |
| 2024-12-31 | 53.39% | 46.61% | 0.01% | 46.34 |
| 2024-06-30 | --- | 100.00% | 0.00% | 7,932.19 |
"""
        df = _parse_cyrjg_from_content(content)
        assert len(df) == 3
        assert list(df.columns) == [
            "日期",
            "机构持有比例",
            "个人持有比例",
            "内部持有比例",
            "总份额（亿份）",
        ]
        assert df.iloc[0]["日期"] == "2025-06-30"
        assert df.iloc[0]["机构持有比例"] == "63.45%"
        assert df.iloc[0]["总份额（亿份）"] == 68.84
        assert pd.isna(df.iloc[2]["机构持有比例"]) is False  # --- 在原列保留
        assert df.iloc[2]["机构持有比例"] == "---"
        assert df.iloc[2]["总份额（亿份）"] == 7932.19

    def test_minimal_two_rows(self):
        content = """
| 公告日期 | 机构持有比例 | 总份额（亿份） |
| --- | --- | --- |
| 2025-06-30 | 50.00% | 100.00 |
"""
        df = _parse_cyrjg_from_content(content)
        assert len(df) == 1
        assert df.iloc[0]["日期"] == "2025-06-30"
        assert df.iloc[0]["总份额（亿份）"] == 100.0

    def test_html_table_parsed(self):
        """API 实际返回 HTML 表格，非 Markdown"""
        html = """
<table class="w782 comm cyrjg">
<thead><tr><th class="first">公告日期</th><th>机构持有比例</th><th>个人持有比例</th><th>内部持有比例</th><th class="last">总份额（亿份）</th></tr></thead>
<tbody>
<tr><td>2025-06-30</td><td>63.45%</td><td>36.55%</td><td>0.01%</td><td>68.84</td></tr>
<tr><td>2024-12-31</td><td>53.39%</td><td>46.61%</td><td>0.01%</td><td>46.34</td></tr>
</tbody>
</table>
"""
        df = _parse_cyrjg_from_html(html)
        assert len(df) == 2
        assert df.iloc[0]["日期"] == "2025-06-30"
        assert df.iloc[0]["机构持有比例"] == "63.45%"
        assert df.iloc[0]["总份额（亿份）"] == 68.84


class TestFundCyrjgEm:
    """fund_cyrjg_em - API mock"""

    @patch("fund_cyrjg.fetch_cyrjg_api")
    def test_from_content_mock(self, mock_fetch):
        content = """
| 公告日期 | 机构持有比例 | 个人持有比例 | 内部持有比例 | 总份额（亿份） |
| --- | --- | --- | --- | --- |
| 2025-06-30 | 63.45% | 36.55% | 0.01% | 68.84 |
"""
        mock_fetch.return_value = {"content": content, "summary": "test"}
        df = fund_cyrjg_em("000015")
        assert not df.empty
        assert len(df) == 1
        assert df.iloc[0]["日期"] == "2025-06-30"
        assert df.iloc[0]["机构持有比例"] == "63.45%"
        assert df.iloc[0]["总份额（亿份）"] == 68.84

    @patch("fund_cyrjg.fetch_cyrjg_api")
    def test_empty_content_returns_empty_df(self, mock_fetch):
        mock_fetch.return_value = {"content": "", "summary": ""}
        df = fund_cyrjg_em("000015")
        assert df.empty
        assert list(df.columns) == [
            "日期",
            "机构持有比例",
            "个人持有比例",
            "内部持有比例",
            "总份额（亿份）",
        ]


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


class TestFundCyrjgEmException:
    """fund_cyrjg_em 异常场景"""

    @patch("fund_cyrjg.fetch_cyrjg_api")
    def test_network_error_propagates(self, mock_fetch):
        mock_fetch.side_effect = Exception("Connection timeout")
        with pytest.raises(Exception, match="Connection timeout"):
            fund_cyrjg_em("000015")


class TestFetchCyrjgApiException:
    """fetch_cyrjg_api 异常/边界"""

    def test_invalid_code_returns_empty(self):
        result = fetch_cyrjg_api("")
        assert result == {"content": "", "summary": ""}

    def test_non_digit_code_returns_empty(self):
        result = fetch_cyrjg_api("abc")
        assert result == {"content": "", "summary": ""}


# ============= 3. 边界条件 =============


class TestFundCyrjgEmBoundary:
    """fund_cyrjg_em 边界"""

    @patch("fund_cyrjg.fetch_cyrjg_api")
    def test_invalid_code_empty_columns(self, mock_fetch):
        df = fund_cyrjg_em("")
        assert df.empty
        assert list(df.columns) == [
            "日期",
            "机构持有比例",
            "个人持有比例",
            "内部持有比例",
            "总份额（亿份）",
        ]
        mock_fetch.assert_not_called()


class TestParseCyrjgContentBoundary:
    """Markdown 解析边界"""

    def test_separator_line_skipped(self):
        content = """
| A | B |
| --- | --- |
| 1 | 2 |
"""
        df = _parse_cyrjg_from_content(content)
        assert len(df) == 1
        assert df.iloc[0]["A"] == "1"

    def test_row_cell_count_mismatch_skipped(self):
        content = """
| A | B |
| --- | --- |
| 1 |
| 2 | 3 |
"""
        df = _parse_cyrjg_from_content(content)
        assert len(df) == 1
        assert df.iloc[0]["A"] == "2"
        assert df.iloc[0]["B"] == "3"


# ============= 4. CLI 测试 =============


class TestFetchFundCyrjgCLI:
    """fetch_fund_cyrjg CLI 单元测试"""

    def test_load_codes_from_csv_success(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8-sig"
        ) as f:
            f.write("基金代码,名称\n000015,华夏纯债\n000198,余额宝\n")
            csv_path = Path(f.name)
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "fetch_fund_cyrjg",
                Path(__file__).resolve().parent.parent / "tools" / "prep" / "fetch_fund_cyrjg.py",
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            codes = mod._load_codes_from_csv(csv_path)
            assert "000015" in codes
            assert "000198" in codes
            assert len(codes) == 2
        finally:
            csv_path.unlink()

    def test_load_codes_from_csv_missing_column_raises(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8-sig"
        ) as f:
            f.write("code,name\n000015,a\n")
            csv_path = Path(f.name)
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "fetch_fund_cyrjg",
                Path(__file__).resolve().parent.parent / "tools" / "prep" / "fetch_fund_cyrjg.py",
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            with pytest.raises(ValueError, match="缺少列"):
                mod._load_codes_from_csv(csv_path, code_col="基金代码")
        finally:
            csv_path.unlink()
