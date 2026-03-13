# -*- coding: utf-8 -*-
"""fund_gmbd 模块单元测试（解析逻辑 + 本地 HTML + mock）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from fund_gmbd import (
    _normalize_code,
    _parse_num,
    _parse_gmbd_from_content,
    fund_gmbd_em,
)


class TestFundGmbd(unittest.TestCase):
    def test_normalize_code(self) -> None:
        self.assertEqual(_normalize_code("198"), "000198")
        self.assertEqual(_normalize_code("000198"), "000198")
        self.assertEqual(_normalize_code("  110011  "), "110011")
        self.assertEqual(_normalize_code("519674"), "519674")

    def test_parse_num(self) -> None:
        self.assertEqual(_parse_num("12,392.31"), 12392.31)
        self.assertEqual(_parse_num("7,646.28"), 7646.28)
        self.assertIsNone(_parse_num("---"))
        self.assertIsNone(_parse_num(""))
        self.assertIsNone(_parse_num(None))

    def test_parse_gmbd_from_content_empty(self) -> None:
        df = _parse_gmbd_from_content("")
        self.assertTrue(df.empty)
        df = _parse_gmbd_from_content("<div>no table</div>")
        self.assertTrue(df.empty)

    def test_parse_gmbd_from_content_with_table(self) -> None:
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
        self.assertEqual(len(df), 2)
        self.assertEqual(
            list(df.columns),
            ["日期", "期间申购（亿份）", "期间赎回（亿份）", "期末总份额（亿份）", "期末净资产（亿元）", "净资产变动率"],
        )
        self.assertEqual(df.iloc[0]["日期"], "2025-12-31")
        self.assertEqual(df.iloc[0]["期间申购（亿份）"], 12392.31)
        self.assertTrue(pd.isna(df.iloc[1]["期间申购（亿份）"]))
        self.assertEqual(df.iloc[1]["期末净资产（亿元）"], 10334.35)

    def test_parse_gmbd_from_local_example(self) -> None:
        """使用项目内 gmbd_example.html 校验解析。"""
        example_path = Path(__file__).resolve().parents[1] / "tmp" / "gmbd_example.html"
        if not example_path.exists():
            self.skipTest("本地样例 gmbd_example.html 不存在")
        html = example_path.read_text(encoding="utf-8")
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        div = soup.find("div", id="gmbdtable")
        if not div or not div.find("table"):
            self.skipTest("gmbd_example.html 中未找到 gmbdtable 表格")
        table_html = str(div.find("table"))
        df = _parse_gmbd_from_content(table_html)
        self.assertFalse(df.empty)
        self.assertIn("日期", df.columns)
        self.assertGreaterEqual(len(df), 50)
        self.assertEqual(df.iloc[0]["日期"], "2025-12-31")
        self.assertEqual(df.iloc[0]["期末净资产（亿元）"], 7646.28)

    @patch("fund_gmbd.fetch_gmbd_api")
    def test_fund_gmbd_em_from_content_mock(self, mock_fetch) -> None:
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
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["日期"], "2025-12-31")
        self.assertEqual(df.iloc[0]["期末净资产（亿元）"], 7646.28)
        mock_fetch.assert_called_once_with("000198", timeout=15)

    @patch("fund_gmbd.fetch_gmbd_api")
    def test_fund_gmbd_em_from_data_mock(self, mock_fetch) -> None:
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
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["日期"], "2025-12-31")
        self.assertEqual(df.iloc[0]["期间申购（亿份）"], 12392.31)
        self.assertEqual(df.iloc[0]["期末净资产（亿元）"], 7646.28)
        self.assertIn("-3.39%", str(df.iloc[0]["净资产变动率"]))

    @patch("fund_gmbd.fetch_gmbd_api")
    def test_fund_gmbd_em_empty(self, mock_fetch) -> None:
        mock_fetch.return_value = {"content": "", "summary": "", "data": []}
        df = fund_gmbd_em("000000")
        self.assertTrue(df.empty)
        self.assertEqual(
            list(df.columns),
            [
                "日期", "期间申购（亿份）", "期间赎回（亿份）",
                "期末总份额（亿份）", "期末净资产（亿元）", "净资产变动率",
            ],
        )
