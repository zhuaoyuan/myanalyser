from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fund_etl import (
    ProgressConfig,
    RetryConfig,
    run_step1_purchase,
    run_step2_overview,
    run_step3_nav,
    run_step4_bonus,
    run_step5_split,
    run_step6_personnel,
    run_step7_cum_return,
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
        nav_df = pd.DataFrame(columns=["净值日期", "单位净值", "日增长率"])
        bonus_df = pd.DataFrame(columns=["年份", "权益登记日", "除息日", "每份分红", "分红发放日"])
        split_df = pd.DataFrame(columns=["年份", "拆分折算日", "拆分类型", "拆分折算比例"])
        cum_return_df = pd.DataFrame(columns=["日期", "累计收益率"])
        personnel_df = pd.DataFrame(columns=["基金代码", "公告标题", "基金名称", "公告日期", "报告ID"])

        with patch("fund_etl.ak.fund_purchase_em", return_value=purchase_df), patch(
            "fund_etl.ak.fund_overview_em", return_value=overview_df
        ), patch("fund_etl.ak.fund_open_fund_info_em", side_effect=[nav_df, bonus_df, split_df, cum_return_df]), patch(
            "fund_etl.ak.fund_announcement_personnel_em", return_value=personnel_df
        ):
            report = verify_interfaces(sample_code="000001", nav_code="000001")

        self.assertTrue(report["fund_purchase_em"]["required_columns_present"])
        self.assertEqual(report["fund_overview_em"]["asset_scale_source_column"], "净资产规模")
        self.assertTrue(report["fund_open_fund_info_em"]["required_columns_present"])
        self.assertTrue(report["fund_announcement_personnel_em"]["required_columns_present"])

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

    def test_step2_multithread_incremental_persist_and_progress(self) -> None:
        purchase = pd.DataFrame([{"基金代码": "000001"}, {"基金代码": "000002"}])

        def overview_side_effect(symbol: str) -> pd.DataFrame:
            if symbol == "000002":
                raise RuntimeError("boom")
            return pd.DataFrame(
                [
                    {
                        "基金简称": "ok",
                        "基金全称": "ok",
                        "基金类型": "混合",
                        "发行日期": "2020-01-01",
                        "成立日期/规模": "x",
                        "净资产规模": "1亿元",
                        "份额规模": "1亿份",
                    }
                ]
            )

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            purchase_csv = base / "purchase.csv"
            overview_csv = base / "overview.csv"
            fail_log = base / "failed.jsonl"
            purchase.to_csv(purchase_csv, index=False, encoding="utf-8-sig")

            with patch("fund_etl.ak.fund_overview_em", side_effect=overview_side_effect), patch("builtins.print") as mock_print:
                summary = run_step2_overview(
                    purchase_csv=purchase_csv,
                    overview_csv=overview_csv,
                    fail_log=fail_log,
                    retry_cfg=RetryConfig(max_retries=1, retry_sleep_seconds=0),
                    max_workers=2,
                    progress_cfg=ProgressConfig(print_interval_seconds=0),
                )

            written = pd.read_csv(overview_csv, dtype={"基金代码": str})
            self.assertEqual(summary["fetched"], 1)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(len(written), 1)
            self.assertEqual(written.iloc[0]["基金代码"], "000001")
            self.assertGreaterEqual(mock_print.call_count, 2)

    def test_step3_write_per_code_csv(self) -> None:
        purchase = pd.DataFrame([{"基金代码": "000009"}])
        nav = pd.DataFrame(
            [
                {"净值日期": "2024-01-01", "单位净值": 1.0, "日增长率": 0.1},
                {"净值日期": "2024-01-02", "单位净值": 1.1, "日增长率": 0.2},
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
            self.assertEqual(written.columns.tolist(), ["基金代码", "净值日期", "单位净值", "日增长率"])
            self.assertEqual(len(written), 2)
            self.assertEqual(summary["rows_written"], 2)
            self.assertFalse(fail_log.exists())

    def test_step3_multithread_progress_and_partial_success(self) -> None:
        purchase = pd.DataFrame([{"基金代码": "000009"}, {"基金代码": "000010"}])

        def nav_side_effect(symbol: str, indicator: str) -> pd.DataFrame:
            if symbol == "000010":
                raise RuntimeError("boom")
            return pd.DataFrame([{"净值日期": "2024-01-01", "单位净值": 1.0, "日增长率": 0.1}])

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            purchase_csv = base / "purchase.csv"
            nav_dir = base / "nav"
            fail_log = base / "failed_nav.jsonl"
            purchase.to_csv(purchase_csv, index=False, encoding="utf-8-sig")

            with patch("fund_etl.ak.fund_open_fund_info_em", side_effect=nav_side_effect), patch(
                "builtins.print"
            ) as mock_print:
                summary = run_step3_nav(
                    purchase_csv=purchase_csv,
                    nav_dir=nav_dir,
                    fail_log=fail_log,
                    retry_cfg=RetryConfig(max_retries=1, retry_sleep_seconds=0),
                    max_workers=2,
                    progress_cfg=ProgressConfig(print_interval_seconds=0),
                )

            self.assertEqual(summary["fetched"], 1)
            self.assertEqual(summary["failed"], 1)
            self.assertTrue((nav_dir / "000009.csv").exists())
            self.assertFalse((nav_dir / "000010.csv").exists())
            self.assertGreaterEqual(mock_print.call_count, 2)

    def test_step4_bonus_write_empty_file_when_no_data(self) -> None:
        purchase = pd.DataFrame([{"基金代码": "000011"}])
        bonus = pd.DataFrame(columns=["年份", "权益登记日", "除息日", "每份分红", "分红发放日"])

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            purchase_csv = base / "purchase.csv"
            bonus_dir = base / "bonus"
            fail_log = base / "failed_bonus.jsonl"
            purchase.to_csv(purchase_csv, index=False, encoding="utf-8-sig")

            with patch("fund_etl.ak.fund_open_fund_info_em", return_value=bonus):
                summary = run_step4_bonus(
                    purchase_csv=purchase_csv,
                    bonus_dir=bonus_dir,
                    fail_log=fail_log,
                    retry_cfg=RetryConfig(max_retries=1, retry_sleep_seconds=0),
                )

            out = bonus_dir / "000011.csv"
            self.assertTrue(out.exists())
            written = pd.read_csv(out, dtype={"基金代码": str})
            self.assertEqual(written.columns.tolist(), ["基金代码", "年份", "权益登记日", "除息日", "每份分红", "分红发放日"])
            self.assertEqual(len(written), 0)
            self.assertEqual(summary["rows_written"], 0)
            self.assertFalse(fail_log.exists())

    def test_step5_split_write_rows(self) -> None:
        purchase = pd.DataFrame([{"基金代码": "000012"}])
        split = pd.DataFrame(
            [
                {"年份": "2007年", "拆分折算日": "2007-05-11", "拆分类型": "份额折算", "拆分折算比例": "1:3.9939"}
            ]
        )

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            purchase_csv = base / "purchase.csv"
            split_dir = base / "split"
            fail_log = base / "failed_split.jsonl"
            purchase.to_csv(purchase_csv, index=False, encoding="utf-8-sig")

            with patch("fund_etl.ak.fund_open_fund_info_em", return_value=split):
                summary = run_step5_split(
                    purchase_csv=purchase_csv,
                    split_dir=split_dir,
                    fail_log=fail_log,
                    retry_cfg=RetryConfig(max_retries=1, retry_sleep_seconds=0),
                )

            out = split_dir / "000012.csv"
            self.assertTrue(out.exists())
            written = pd.read_csv(out, dtype={"基金代码": str})
            self.assertEqual(written.columns.tolist(), ["基金代码", "年份", "拆分折算日", "拆分类型", "拆分折算比例"])
            self.assertEqual(len(written), 1)
            self.assertEqual(summary["rows_written"], 1)
            self.assertFalse(fail_log.exists())

    def test_step6_personnel_write_empty_file_when_no_data(self) -> None:
        purchase = pd.DataFrame([{"基金代码": "000013"}])
        personnel = pd.DataFrame(columns=["基金代码", "公告标题", "基金名称", "公告日期", "报告ID"])

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            purchase_csv = base / "purchase.csv"
            personnel_dir = base / "personnel"
            fail_log = base / "failed_personnel.jsonl"
            purchase.to_csv(purchase_csv, index=False, encoding="utf-8-sig")

            with patch("fund_etl.ak.fund_announcement_personnel_em", return_value=personnel):
                summary = run_step6_personnel(
                    purchase_csv=purchase_csv,
                    personnel_dir=personnel_dir,
                    fail_log=fail_log,
                    retry_cfg=RetryConfig(max_retries=1, retry_sleep_seconds=0),
                )

            out = personnel_dir / "000013.csv"
            self.assertTrue(out.exists())
            written = pd.read_csv(out, dtype={"基金代码": str})
            self.assertEqual(written.columns.tolist(), ["基金代码", "公告标题", "基金名称", "公告日期", "报告ID"])
            self.assertEqual(len(written), 0)
            self.assertEqual(summary["rows_written"], 0)
            self.assertFalse(fail_log.exists())

    def test_step6_personnel_length_mismatch_treated_as_empty(self) -> None:
        purchase = pd.DataFrame([{"基金代码": "161005"}])

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            purchase_csv = base / "purchase.csv"
            personnel_dir = base / "personnel"
            fail_log = base / "failed_personnel.jsonl"
            purchase.to_csv(purchase_csv, index=False, encoding="utf-8-sig")

            with patch(
                "fund_etl.ak.fund_announcement_personnel_em",
                side_effect=ValueError("Length mismatch: Expected axis has 0 elements, new values have 8 elements"),
            ):
                summary = run_step6_personnel(
                    purchase_csv=purchase_csv,
                    personnel_dir=personnel_dir,
                    fail_log=fail_log,
                    retry_cfg=RetryConfig(max_retries=1, retry_sleep_seconds=0),
                )

            out = personnel_dir / "161005.csv"
            self.assertTrue(out.exists())
            written = pd.read_csv(out, dtype={"基金代码": str})
            self.assertEqual(written.columns.tolist(), ["基金代码", "公告标题", "基金名称", "公告日期", "报告ID"])
            self.assertEqual(len(written), 0)
            self.assertEqual(summary["rows_written"], 0)
            self.assertEqual(summary["failed"], 0)
            self.assertFalse(fail_log.exists())

    def test_step7_cum_return_write_rows(self) -> None:
        purchase = pd.DataFrame([{"基金代码": "000014"}])
        cum_return = pd.DataFrame(
            [
                {"日期": "2024-01-01", "累计收益率": 1.2},
                {"日期": "2024-01-02", "累计收益率": 1.5},
            ]
        )

        def cum_return_side_effect(symbol: str, indicator: str, period: str) -> pd.DataFrame:
            self.assertEqual(symbol, "000014")
            self.assertEqual(indicator, "累计收益率走势")
            self.assertEqual(period, "成立来")
            return cum_return

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            purchase_csv = base / "purchase.csv"
            cum_return_dir = base / "cum_return"
            fail_log = base / "failed_cum_return.jsonl"
            purchase.to_csv(purchase_csv, index=False, encoding="utf-8-sig")

            with patch("fund_etl.ak.fund_open_fund_info_em", side_effect=cum_return_side_effect):
                summary = run_step7_cum_return(
                    purchase_csv=purchase_csv,
                    cum_return_dir=cum_return_dir,
                    fail_log=fail_log,
                    retry_cfg=RetryConfig(max_retries=1, retry_sleep_seconds=0),
                )

            out = cum_return_dir / "000014.csv"
            self.assertTrue(out.exists())
            written = pd.read_csv(out, dtype={"基金代码": str})
            self.assertEqual(written.columns.tolist(), ["基金代码", "日期", "累计收益率"])
            self.assertEqual(len(written), 2)
            self.assertEqual(summary["rows_written"], 2)
            self.assertFalse(fail_log.exists())

    def test_closed_loop_all_steps(self) -> None:
        purchase_raw = pd.DataFrame(
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
                }
            ]
        )
        overview = pd.DataFrame([{"基金简称": "A", "基金全称": "A基金", "净资产规模": "1亿元"}])
        nav = pd.DataFrame([{"净值日期": "2024-01-01", "单位净值": 1.0, "日增长率": 0.1}])
        bonus = pd.DataFrame([{"年份": "2021年", "权益登记日": "2021-12-16", "除息日": "2021-12-16", "每份分红": "每份派现金0.1584元", "分红发放日": "2021-12-20"}])
        split = pd.DataFrame([{"年份": "2007年", "拆分折算日": "2007-05-11", "拆分类型": "份额折算", "拆分折算比例": "1:3.9939"}])
        personnel = pd.DataFrame(
            [{"基金代码": "000001", "公告标题": "基金经理调整公告", "基金名称": "A基金", "公告日期": "2024-01-01", "报告ID": "AN1"}]
        )
        cum_return = pd.DataFrame([{"日期": "2024-01-01", "累计收益率": 1.2}])

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            purchase_csv = base / "purchase.csv"
            overview_csv = base / "overview.csv"
            nav_dir = base / "nav"
            fail_overview = base / "failed_overview.jsonl"
            fail_nav = base / "failed_nav.jsonl"
            fail_bonus = base / "failed_bonus.jsonl"
            fail_split = base / "failed_split.jsonl"
            fail_personnel = base / "failed_personnel.jsonl"
            fail_cum_return = base / "failed_cum_return.jsonl"

            with patch("fund_etl.ak.fund_purchase_em", return_value=purchase_raw):
                step1_df = run_step1_purchase(purchase_csv)
            self.assertEqual(len(step1_df), 1)

            with patch("fund_etl.ak.fund_overview_em", return_value=overview):
                step2 = run_step2_overview(
                    purchase_csv=purchase_csv,
                    overview_csv=overview_csv,
                    fail_log=fail_overview,
                    retry_cfg=RetryConfig(max_retries=1, retry_sleep_seconds=0),
                    max_workers=2,
                    progress_cfg=ProgressConfig(print_interval_seconds=999),
                )
            self.assertEqual(step2["fetched"], 1)
            self.assertTrue(overview_csv.exists())
            self.assertFalse(fail_overview.exists())

            with patch("fund_etl.ak.fund_open_fund_info_em", return_value=nav):
                step3 = run_step3_nav(
                    purchase_csv=purchase_csv,
                    nav_dir=nav_dir,
                    fail_log=fail_nav,
                    retry_cfg=RetryConfig(max_retries=1, retry_sleep_seconds=0),
                    max_workers=2,
                    progress_cfg=ProgressConfig(print_interval_seconds=999),
                )
            self.assertEqual(step3["fetched"], 1)
            self.assertTrue((nav_dir / "000001.csv").exists())
            self.assertFalse(fail_nav.exists())

            with patch("fund_etl.ak.fund_open_fund_info_em", return_value=bonus):
                step4 = run_step4_bonus(
                    purchase_csv=purchase_csv,
                    bonus_dir=base / "bonus",
                    fail_log=fail_bonus,
                    retry_cfg=RetryConfig(max_retries=1, retry_sleep_seconds=0),
                    progress_cfg=ProgressConfig(print_interval_seconds=999),
                )
            self.assertEqual(step4["fetched"], 1)
            self.assertTrue((base / "bonus" / "000001.csv").exists())
            self.assertFalse(fail_bonus.exists())

            with patch("fund_etl.ak.fund_open_fund_info_em", return_value=split):
                step5 = run_step5_split(
                    purchase_csv=purchase_csv,
                    split_dir=base / "split",
                    fail_log=fail_split,
                    retry_cfg=RetryConfig(max_retries=1, retry_sleep_seconds=0),
                    progress_cfg=ProgressConfig(print_interval_seconds=999),
                )
            self.assertEqual(step5["fetched"], 1)
            self.assertTrue((base / "split" / "000001.csv").exists())
            self.assertFalse(fail_split.exists())

            with patch("fund_etl.ak.fund_announcement_personnel_em", return_value=personnel):
                step6 = run_step6_personnel(
                    purchase_csv=purchase_csv,
                    personnel_dir=base / "personnel",
                    fail_log=fail_personnel,
                    retry_cfg=RetryConfig(max_retries=1, retry_sleep_seconds=0),
                    progress_cfg=ProgressConfig(print_interval_seconds=999),
                )
            self.assertEqual(step6["fetched"], 1)
            self.assertTrue((base / "personnel" / "000001.csv").exists())
            self.assertFalse(fail_personnel.exists())

            with patch("fund_etl.ak.fund_open_fund_info_em", return_value=cum_return):
                step7 = run_step7_cum_return(
                    purchase_csv=purchase_csv,
                    cum_return_dir=base / "cum_return",
                    fail_log=fail_cum_return,
                    retry_cfg=RetryConfig(max_retries=1, retry_sleep_seconds=0),
                    progress_cfg=ProgressConfig(print_interval_seconds=999),
                )
            self.assertEqual(step7["fetched"], 1)
            self.assertTrue((base / "cum_return" / "000001.csv").exists())
            self.assertFalse(fail_cum_return.exists())


if __name__ == "__main__":
    unittest.main()
