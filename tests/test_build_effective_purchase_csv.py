"""测试 build_effective_purchase_csv：从 fund_purchase 剔除黑名单生成 fund_purchase_effective。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transforms.build_effective_purchase_csv import build_effective_purchase_csv


class BuildEffectivePurchaseCsvTest(unittest.TestCase):
    def test_no_blacklist_returns_full_purchase(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            purchase = Path(d) / "fund_purchase.csv"
            pd.DataFrame(
                [
                    {"基金代码": "000001", "基金简称": "A", "申购状态": "开放", "赎回状态": "开放", "购买起点": "10", "日累计限定金额": "1e9", "手续费": "0.1"},
                    {"基金代码": "163402", "基金简称": "B", "申购状态": "开放", "赎回状态": "开放", "购买起点": "10", "日累计限定金额": "1e9", "手续费": "0.15"},
                ],
                columns=["基金代码", "基金简称", "申购状态", "赎回状态", "购买起点", "日累计限定金额", "手续费"],
            ).to_csv(purchase, index=False, encoding="utf-8-sig")
            output = Path(d) / "fund_purchase_effective.csv"

            result = build_effective_purchase_csv(
                purchase_csv=purchase,
                blacklist_csv=None,
                output_csv=output,
            )
            self.assertEqual(result["original_count"], 2)
            self.assertEqual(result["blacklist_removed"], 0)
            self.assertEqual(result["effective_count"], 2)
            out_df = pd.read_csv(output, dtype=str, encoding="utf-8-sig")
            self.assertEqual(len(out_df), 2)

    def test_blacklist_excludes_funds(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            purchase = Path(d) / "fund_purchase.csv"
            pd.DataFrame(
                [
                    {"基金代码": "000001", "基金简称": "A", "申购状态": "开放", "赎回状态": "开放", "购买起点": "10", "日累计限定金额": "1e9", "手续费": "0.1"},
                    {"基金代码": "163402", "基金简称": "B", "申购状态": "开放", "赎回状态": "开放", "购买起点": "10", "日累计限定金额": "1e9", "手续费": "0.15"},
                    {"基金代码": "000003", "基金简称": "C", "申购状态": "开放", "赎回状态": "开放", "购买起点": "10", "日累计限定金额": "1e9", "手续费": "0.08"},
                ],
                columns=["基金代码", "基金简称", "申购状态", "赎回状态", "购买起点", "日累计限定金额", "手续费"],
            ).to_csv(purchase, index=False, encoding="utf-8-sig")
            blacklist = Path(d) / "fund_blacklist.csv"
            pd.DataFrame([{"基金代码": "163402"}]).to_csv(blacklist, index=False, encoding="utf-8-sig")
            output = Path(d) / "fund_purchase_effective.csv"

            result = build_effective_purchase_csv(
                purchase_csv=purchase,
                blacklist_csv=blacklist,
                output_csv=output,
            )
            self.assertEqual(result["original_count"], 3)
            self.assertEqual(result["blacklist_removed"], 1)
            self.assertEqual(result["effective_count"], 2)
            out_df = pd.read_csv(output, dtype=str, encoding="utf-8-sig")
            codes = set(out_df["基金代码"].str.strip().str.zfill(6).tolist())
            self.assertIn("000001", codes)
            self.assertIn("000003", codes)
            self.assertNotIn("163402", codes)


if __name__ == "__main__":
    unittest.main()
