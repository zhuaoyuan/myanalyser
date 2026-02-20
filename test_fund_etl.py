from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from fund_etl import (
    RetryConfig,
    run_step1_purchase,
    run_step2_overview,
    run_step3_nav,
    verify_interfaces,
)


class FundEtlTest(unittest.TestCase):
    def test_verify_interfaces_maps_asset_scale(self) -> None:
        purchase_df = pd.DataFrame(
            columns=[
                "基金代码",
                "基金简称",
                "申购状态",
                "赎回状态",
                "下一开放日",
                "购买起点",
                "日累计限定金额",
                "手续费",
            ]
        )
        overview_df = pd.DataFrame(columns=["基金代码", "基金简称", "净资产规模"])
        nav_df = pd.DataFrame(columns=["净值日期", "累计净值"])

        with patch("fund_etl.ak.fund_purchase_em", return_value=purchase_df), patch(
            "fund_etl.ak.fund_overview_em", return_value=overview_df
        ), patch("fund_etl.ak.fund_open_fund_info_em", return_value=nav_df):
            report = verify_interfaces(sample_code="000001", nav_code="000001")

        self.assertTrue(report["fund_purchase_em"]["required_columns_present"])
        self.assertEqual(report["fund_overview_em"]["asset_scale_source_column"], "净资产规模")
        self.assertTrue(report["fund_open_fund_info_em"]["required_columns_present"])

    def test_step1_writes_expected_columns(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "基金代码": "1",
                    "基金简称": "A",
                    "申购状态": "开放申购",
                    "赎回状态": "开放赎回",
                    "下一开放日": "",
                    "购买起点": 10,
                    "日累计限定金额": 100,
                    "手续费": 0.1,
                    "无关字段": "x",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "purchase.csv"
            with patch("fund_etl.ak.fund_purchase_em", return_value=raw):
                df = run_step1_purchase(out)
            self.assertTrue(out.exists())
            self.assertEqual(df.columns.tolist(), [
                "基金代码",
                "基金简称",
                "申购状态",
                "赎回状态",
                "下一开放日",
                "购买起点",
                "日累计限定金额",
                "手续费",
            ])
            self.assertEqual(df.iloc[0]["基金代码"], "000001")

    def test_step2_skip_done_and_log_failures(self) -> None:
        purchase = pd.DataFrame(
            [
                {"基金代码": "000001"},
                {"基金代码": "000002"},
            ]
        )
        overview_existing = pd.DataFrame(
            [
                {
                    "基金代码": "000001",
                    "基金简称": "x",
                    "基金全称": "x",
                    "基金类型": "x",
                    "发行日期": "x",
                    "成立日期/规模": "x",
                    "资产规模": "x",
                    "份额规模": "x",
                    "基金管理人": "x",
                    "基金托管人": "x",
                    "基金经理人": "x",
                    "成立来分红": "x",
                    "管理费率": "x",
                    "托管费率": "x",
                    "销售服务费率": "x",
                    "最高认购费率": "x",
                    "业绩比较基准": "x",
                    "跟踪标的": "x",
                }
            ]
        )

        def overview_side_effect(symbol: str) -> pd.DataFrame:
            if symbol == "000002":
                raise RuntimeError("boom")
            return pd.DataFrame([{"基金简称": "ok", "净资产规模": "1亿元"}])

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            purchase_csv = base / "purchase.csv"
            overview_csv = base / "overview.csv"
            fail_log = base / "failed.jsonl"
            purchase.to_csv(purchase_csv, index=False, encoding="utf-8-sig")
            overview_existing.to_csv(overview_csv, index=False, encoding="utf-8-sig")

            with patch("fund_etl.ak.fund_overview_em", side_effect=overview_side_effect):
                summary = run_step2_overview(
                    purchase_csv=purchase_csv,
                    overview_csv=overview_csv,
                    fail_log=fail_log,
                    retry_cfg=RetryConfig(max_retries=2, retry_sleep_seconds=0),
                )

            self.assertEqual(summary["already_done"], 1)
            self.assertEqual(summary["fetched"], 0)
            self.assertEqual(summary["failed"], 1)
            self.assertTrue(fail_log.exists())
            lines = fail_log.read_text(encoding="utf-8").strip().splitlines()
            rec = json.loads(lines[-1])
            self.assertEqual(rec["code"], "000002")

    def test_step3_write_per_code_csv(self) -> None:
        purchase = pd.DataFrame([{"基金代码": "000009"}])
        nav = pd.DataFrame(
            [
                {"净值日期": "2024-01-01", "累计净值": 1.0},
                {"净值日期": "2024-01-02", "累计净值": 1.1},
            ]
        )

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            purchase_csv = base / "purchase.csv"
            nav_dir = base / "nav"
            fail_log = base / "failed_nav.jsonl"
            purchase.to_csv(purchase_csv, index=False, encoding="utf-8-sig")

            with patch("fund_etl.ak.fund_open_fund_info_em", return_value=nav):
                summary = run_step3_nav(
                    purchase_csv=purchase_csv,
                    nav_dir=nav_dir,
                    fail_log=fail_log,
                    retry_cfg=RetryConfig(max_retries=1, retry_sleep_seconds=0),
                )

            out = nav_dir / "000009.csv"
            self.assertTrue(out.exists())
            written = pd.read_csv(out, dtype={"基金代码": str})
            self.assertEqual(written.columns.tolist(), ["基金代码", "净值日期", "累计净值"])
            self.assertEqual(len(written), 2)
            self.assertEqual(summary["rows_written"], 2)
            self.assertFalse(fail_log.exists())


if __name__ == "__main__":
    unittest.main()
