from __future__ import annotations

import shutil
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import backtest_portfolio
import check_trade_day_data_integrity
import compare_adjusted_nav_and_cum_return
import fund_etl
import pipeline_scoreboard
from pipeline_scoreboard import _build_scoreboard


def _test_run_id(tag: str) -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{tag}"


class CoreCliIntegrationTest(unittest.TestCase):
    def test_fund_etl_cli_run_id_layout(self) -> None:
        run_id = _test_run_id("fund_etl")
        project_root = Path(__file__).resolve().parent.parent
        version_root = project_root / "data" / "versions" / run_id
        try:
            purchase_df = pd.DataFrame(
                [
                    {
                        "基金代码": "1",
                        "基金简称": "A",
                        "申购状态": "开放申购",
                        "赎回状态": "开放赎回",
                        "下一开放日": "",
                        "购买起点": 1,
                        "日累计限定金额": 100,
                        "手续费": 0.1,
                    }
                ]
            )
            with patch("fund_etl.ak.fund_purchase_em", return_value=purchase_df), patch.object(
                sys,
                "argv",
                ["fund_etl.py", "--mode", "step1", "--run-id", run_id],
            ):
                fund_etl.main()

            self.assertTrue((version_root / "fund_etl" / "fund_purchase.csv").exists())
        finally:
            if version_root.exists():
                shutil.rmtree(version_root, ignore_errors=True)

    def test_compare_cli_with_run_id_layout(self) -> None:
        run_id = _test_run_id("compare")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base_dir = root / "data" / "versions" / run_id / "fund_etl"
            adjusted = base_dir / "fund_adjusted_nav_by_code"
            cum = base_dir / "fund_cum_return_by_code"
            adjusted.mkdir(parents=True, exist_ok=True)
            cum.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {"基金代码": "000001", "净值日期": "2024-01-01", "复权净值": 1.0},
                    {"基金代码": "000001", "净值日期": "2024-01-02", "复权净值": 1.1},
                ]
            ).to_csv(adjusted / "000001.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {"基金代码": "000001", "日期": "2024-01-01", "累计收益率": 0.0},
                    {"基金代码": "000001", "日期": "2024-01-02", "累计收益率": 0.1},
                ]
            ).to_csv(cum / "000001.csv", index=False, encoding="utf-8-sig")

            result = compare_adjusted_nav_and_cum_return.compare_adjusted_nav_and_cum_return_with_error_log(
                base_dir=base_dir,
                output_dir=base_dir / "fund_return_compare",
                error_log_path=root / "data" / "versions" / run_id / "logs" / "compare_adjusted_nav_cum_return_errors.jsonl",
            )
            self.assertTrue(result["summary_csv"].exists())
            self.assertTrue(result["error_jsonl"].parent.name == "logs")

    def test_check_trade_day_integrity_cli_with_run_id_layout(self) -> None:
        run_id = _test_run_id("integrity")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base_dir = root / "data" / "versions" / run_id / "fund_etl"
            nav_dir = base_dir / "fund_adjusted_nav_by_code"
            nav_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"基金代码": "000001", "净值日期": "2024-01-01", "复权净值": 1.0},
                    {"基金代码": "000001", "净值日期": "2024-01-02", "复权净值": 1.1},
                ]
            ).to_csv(nav_dir / "000001.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [{"基金代码": "000001", "成立日期/规模": "2010-01-01"}]
            ).to_csv(base_dir / "fund_overview.csv", index=False, encoding="utf-8-sig")
            trade_dates = root / "data" / "common" / "trade_dates.csv"
            trade_dates.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"trade_date": ["2024-01-01", "2024-01-02"]}).to_csv(
                trade_dates, index=False, encoding="utf-8-sig"
            )

            with patch.object(
                sys,
                "argv",
                [
                    "check_trade_day_data_integrity.py",
                    "--base-dir",
                    str(base_dir),
                    "--start-date",
                    "2024-01-01",
                    "--end-date",
                    "2024-01-02",
                    "--trade-dates-csv",
                    str(trade_dates),
                    "--output-dir",
                    str(base_dir / "trade_day_integrity_reports"),
                ],
            ):
                check_trade_day_data_integrity.main()
            self.assertTrue((base_dir / "trade_day_integrity_reports").exists())

    def test_pipeline_cli_smoke_skip_sinks_with_run_id_layout(self) -> None:
        run_id = _test_run_id("scoreboard")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base_dir = root / "data" / "versions" / run_id / "fund_etl"
            base_dir.mkdir(parents=True, exist_ok=True)
            (base_dir / "fund_personnel_by_code").mkdir(parents=True, exist_ok=True)
            (base_dir / "fund_adjusted_nav_by_code").mkdir(parents=True, exist_ok=True)
            out_dir = root / "artifacts" / f"scoreboard_{run_id}"

            pd.DataFrame(
                [
                    {
                        "基金代码": "000001",
                        "基金简称": "A",
                        "申购状态": "开放申购",
                        "赎回状态": "开放赎回",
                        "下一开放日": "",
                        "购买起点": 1,
                        "日累计限定金额": 100,
                        "手续费": "1.0%",
                    }
                ]
            ).to_csv(base_dir / "fund_purchase.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {
                        "基金代码": "000001",
                        "基金简称": "A",
                        "基金全称": "A基金",
                        "基金类型": "混合",
                        "发行日期": "2010-01-01",
                        "成立日期/规模": "2010-01-01",
                        "资产规模": "10亿元",
                        "份额规模": "10亿份",
                        "基金管理人": "X",
                        "基金托管人": "Y",
                        "基金经理人": "Z",
                        "成立来分红": "",
                        "管理费率": "1.0%",
                        "托管费率": "0.2%",
                        "销售服务费率": "0.1%",
                        "最高认购费率": "1.2%",
                        "业绩比较基准": "",
                        "跟踪标的": "",
                    }
                ]
            ).to_csv(base_dir / "fund_overview.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [{"基金代码": "000001", "公告标题": "x", "基金名称": "A", "公告日期": "2024-01-01", "报告ID": "1"}]
            ).to_csv(
                base_dir / "fund_personnel_by_code" / "000001.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame(
                [
                    {"基金代码": "000001", "净值日期": "2024-01-01", "单位净值": 1.0, "复权净值": 1.0, "cumulative_factor": 1.0},
                    {"基金代码": "000001", "净值日期": "2024-02-01", "单位净值": 1.1, "复权净值": 1.1, "cumulative_factor": 1.0},
                ]
            ).to_csv(
                base_dir / "fund_adjusted_nav_by_code" / "000001.csv",
                index=False,
                encoding="utf-8-sig",
            )

            args = pipeline_scoreboard.build_parser().parse_args(
                [
                    "--purchase-csv",
                    str(base_dir / "fund_purchase.csv"),
                    "--overview-csv",
                    str(base_dir / "fund_overview.csv"),
                    "--personnel-dir",
                    str(base_dir / "fund_personnel_by_code"),
                    "--nav-dir",
                    str(base_dir / "fund_adjusted_nav_by_code"),
                    "--output-dir",
                    str(out_dir),
                    "--data-version",
                    run_id,
                    "--as-of-date",
                    "2024-02-01",
                    "--skip-sinks",
                ]
            )
            pipeline_scoreboard.run_pipeline(args)
            self.assertTrue((out_dir / f"fund_scoreboard_{run_id}.csv").exists())

            # --resume: 验证 checkpoint 存在时跳过计算、产出一致
            checkpoint_dir = out_dir / ".checkpoint" / run_id
            self.assertTrue(checkpoint_dir.exists(), "checkpoint dir should exist after first run")
            scoreboard_csv = out_dir / f"fund_scoreboard_{run_id}.csv"
            first_content = scoreboard_csv.read_text(encoding="utf-8-sig")
            args_resume = pipeline_scoreboard.build_parser().parse_args(
                [
                    "--purchase-csv",
                    str(base_dir / "fund_purchase.csv"),
                    "--overview-csv",
                    str(base_dir / "fund_overview.csv"),
                    "--personnel-dir",
                    str(base_dir / "fund_personnel_by_code"),
                    "--nav-dir",
                    str(base_dir / "fund_adjusted_nav_by_code"),
                    "--output-dir",
                    str(out_dir),
                    "--data-version",
                    run_id,
                    "--as-of-date",
                    "2024-02-01",
                    "--skip-sinks",
                    "--resume",
                ]
            )
            pipeline_scoreboard.run_pipeline(args_resume)
            second_content = scoreboard_csv.read_text(encoding="utf-8-sig")
            self.assertEqual(first_content, second_content, "resume should produce identical output")

    def test_build_scoreboard_exclusion_detail_no_duplicate_primary_key(self) -> None:
        """多指标为 null 的基金仅产生一条 metric_null 记录，保证主键 (data_version, fund_code, reason_code) 唯一"""
        import numpy as np

        dim_base = pd.DataFrame(
            [{"fund_code": "003184", "fund_name": "X"}],
        )
        # 构造一只基金有多个 null 指标的 metric_df
        metric_row = {"fund_code": "003184", "stale_nav_excluded": False}
        for k in pipeline_scoreboard.METRIC_DIRECTIONS:
            metric_row[k] = np.nan
        metric_df = pd.DataFrame([metric_row])

        _, exclusion_detail, _ = _build_scoreboard(
            dim_base_df=dim_base,
            metric_df=metric_df,
            data_version="test_v1",
            as_of_date=pd.Timestamp("2026-02-26"),
        )
        pk_cols = ["data_version", "fund_code", "reason_code"]
        dup = exclusion_detail.duplicated(subset=pk_cols)
        self.assertFalse(dup.any(), f"exclusion_detail 存在重复主键: {exclusion_detail[dup]}")
        metric_null_rows = exclusion_detail[exclusion_detail["reason_code"] == "metric_null"]
        self.assertEqual(len(metric_null_rows), 1)
        self.assertIn("annual_return", str(metric_null_rows.iloc[0]["reason_detail"]))

    def test_backtest_cli_smoke_with_run_id_layout(self) -> None:
        run_id = _test_run_id("backtest")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            trade_dates_csv = root / "data" / "common" / "trade_dates.csv"
            out_dir = root / "artifacts" / f"backtest_{run_id}"
            trade_dates_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"trade_date": pd.date_range("2024-01-01", "2024-03-31", freq="D").strftime("%Y-%m-%d")}).to_csv(
                trade_dates_csv,
                index=False,
                encoding="utf-8-sig",
            )

            nav_dates = pd.date_range("2024-01-02", "2024-03-31", freq="D")
            nav_rows: list[dict[str, object]] = []
            for i, dt in enumerate(nav_dates):
                nav_rows.append({"fund_code": "000001", "nav_date": dt.strftime("%Y-%m-%d"), "adjusted_nav": 1.0 + 0.01 * i})
                nav_rows.append({"fund_code": "000002", "nav_date": dt.strftime("%Y-%m-%d"), "adjusted_nav": 1.0})
            nav_df = pd.DataFrame(nav_rows)

            def fake_query(query: str, container_name: str) -> pd.DataFrame:
                if "FROM fund_analysis.fact_fund_scoreboard_snapshot GROUP BY data_version" in query:
                    return pd.DataFrame([{"data_version": "202401", "as_of_date": "2024-01-01"}])
                if "FROM fund_analysis.fact_fund_scoreboard_snapshot" in query and "SELECT fund_code" in query:
                    return pd.DataFrame([{"fund_code": "000001"}, {"fund_code": "000002"}])
                if "FROM fund_analysis.fact_fund_nav_daily GROUP BY data_version" in query:
                    return pd.DataFrame([{"data_version": "nav_v1", "as_of_date": "2024-03-31"}])
                if "FROM fund_analysis.fact_fund_nav_daily" in query and "SELECT fund_code, nav_date, adjusted_nav" in query:
                    return nav_df.copy()
                raise AssertionError(f"unexpected query: {query}")

            args = Namespace(
                start_date="2024-01-01",
                end_date="2024-01-31",
                output_dir=out_dir,
                trade_dates_csv=str(trade_dates_csv),
                rebalance_interval_days=15,
                holding_period_days=30,
                selection_rule_id=f"test_rule_{run_id}",
                selection_data_version=None,
                selection_where="1",
                selection_order_by="annual_return_rank ASC, fund_code ASC",
                selection_limit=2,
                exclude_subscribe_status="暂停申购,封闭期",
                exclude_redeem_status="暂停赎回,封闭期",
                nav_data_version=None,
                clickhouse_db="fund_analysis",
                clickhouse_container="fund_clickhouse",
            )
            with patch("backtest_portfolio._run_clickhouse_query", side_effect=fake_query):
                backtest_portfolio.run_backtest(args)
            self.assertTrue((out_dir / "backtest_run_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
