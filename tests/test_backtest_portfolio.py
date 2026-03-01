from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtest_portfolio import _assign_selection_versions, _build_windows, run_backtest


class BacktestPortfolioTest(unittest.TestCase):
    def test_build_windows_with_fixed_interval_and_holding_period(self) -> None:
        trade_days = pd.date_range("2024-01-01", "2024-03-31", freq="D").tolist()
        windows = _build_windows(
            start_date="2024-01-01",
            end_date="2024-01-31",
            trade_days=trade_days,
            rebalance_interval_days=15,
            holding_period_days=30,
        )

        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0].d1_anchor_date.strftime("%Y-%m-%d"), "2024-01-01")
        self.assertEqual(windows[0].d2_last_hist_trade_date.strftime("%Y-%m-%d"), "2024-01-01")
        self.assertEqual(windows[0].d3_buy_date.strftime("%Y-%m-%d"), "2024-01-02")
        self.assertEqual(windows[0].d4_sell_date.strftime("%Y-%m-%d"), "2024-02-01")
        self.assertEqual(windows[1].d1_anchor_date.strftime("%Y-%m-%d"), "2024-01-16")
        self.assertEqual(windows[2].d1_anchor_date.strftime("%Y-%m-%d"), "2024-01-31")

    def test_assign_selection_versions_auto_by_d2(self) -> None:
        trade_days = pd.date_range("2024-01-01", "2024-04-30", freq="D").tolist()
        windows = _build_windows(
            start_date="2024-01-01",
            end_date="2024-02-20",
            trade_days=trade_days,
            rebalance_interval_days=15,
            holding_period_days=30,
        )
        # 人工调整一个窗口 d2，验证 earliest 边界会被 skip
        windows[0].d2_last_hist_trade_date = pd.Timestamp("2023-12-31")
        version_dates = pd.DataFrame(
            [
                {"data_version": "202401", "as_of_date": pd.Timestamp("2024-01-10")},
                {"data_version": "202402", "as_of_date": pd.Timestamp("2024-02-01")},
            ]
        )

        _assign_selection_versions(windows, version_dates, fixed_data_version=None)

        self.assertEqual(windows[0].status, "skipped")
        self.assertEqual(windows[0].skip_reason, "no_data_version_before_d2")
        self.assertEqual(windows[1].selection_data_version, "202401")
        self.assertEqual(windows[2].selection_data_version, "202401")
        self.assertEqual(windows[3].selection_data_version, "202402")

    def test_run_backtest_end_to_end_with_mock_clickhouse(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            trade_dates_csv = base / "trade_dates.csv"
            output_dir = base / "out"
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
                output_dir=output_dir,
                trade_dates_csv=str(trade_dates_csv),
                rebalance_interval_days=15,
                holding_period_days=30,
                selection_rule_id="test_rule",
                selection_data_version=None,
                selection_where="1",
                selection_order_by="annual_return DESC, fund_code ASC",
                selection_limit=2,
                exclude_subscribe_status="暂停申购,封闭期",
                exclude_redeem_status="暂停赎回,封闭期",
                nav_data_version=None,
                clickhouse_db="fund_analysis",
                clickhouse_container="fund_clickhouse",
            )

            with patch("backtest_portfolio._run_clickhouse_query", side_effect=fake_query):
                run_backtest(args)

            run_summary = pd.read_csv(output_dir / "backtest_run_summary.csv")
            window_detail = pd.read_csv(output_dir / "backtest_window_detail.csv")
            window_positions = pd.read_csv(output_dir / "backtest_window_positions.csv", dtype={"fund_code": str})
            aggregate = pd.read_csv(output_dir / "backtest_aggregate_summary.csv")
            report_text = (output_dir / "backtest_report.md").read_text(encoding="utf-8")

            self.assertEqual(run_summary.iloc[0]["selection_rule_id"], "test_rule")
            self.assertEqual(run_summary.iloc[0]["nav_data_version"], "nav_v1")

            self.assertEqual(len(window_detail), 3)
            self.assertTrue((window_detail["status"] == "ok").all())
            self.assertAlmostEqual(float(window_detail.iloc[0]["portfolio_return"]), 0.15, places=9)

            self.assertEqual(len(window_positions), 6)
            self.assertSetEqual(set(window_positions["fund_code"].tolist()), {"000001", "000002"})

            self.assertEqual(int(aggregate.iloc[0]["total_windows"]), 3)
            self.assertEqual(int(aggregate.iloc[0]["valid_windows"]), 3)
            self.assertIn("基金组合回测报告", report_text)

    def test_run_backtest_all_windows_skipped_should_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            trade_dates_csv = base / "trade_dates.csv"
            output_dir = base / "out"
            pd.DataFrame({"trade_date": pd.date_range("2024-01-01", "2024-03-31", freq="D").strftime("%Y-%m-%d")}).to_csv(
                trade_dates_csv,
                index=False,
                encoding="utf-8-sig",
            )

            def fake_query(query: str, container_name: str) -> pd.DataFrame:
                if "FROM fund_analysis.fact_fund_scoreboard_snapshot GROUP BY data_version" in query:
                    return pd.DataFrame([{"data_version": "202401", "as_of_date": "2024-01-01"}])
                if "FROM fund_analysis.fact_fund_scoreboard_snapshot" in query and "SELECT fund_code" in query:
                    return pd.DataFrame(columns=["fund_code"])
                if "FROM fund_analysis.fact_fund_nav_daily GROUP BY data_version" in query:
                    return pd.DataFrame([{"data_version": "nav_v1", "as_of_date": "2024-03-31"}])
                if "FROM fund_analysis.fact_fund_nav_daily" in query and "SELECT fund_code, nav_date, adjusted_nav" in query:
                    return pd.DataFrame(columns=["fund_code", "nav_date", "adjusted_nav"])
                raise AssertionError(f"unexpected query: {query}")

            args = Namespace(
                start_date="2024-01-01",
                end_date="2024-01-31",
                output_dir=output_dir,
                trade_dates_csv=str(trade_dates_csv),
                rebalance_interval_days=15,
                holding_period_days=30,
                selection_rule_id="test_rule",
                selection_data_version=None,
                selection_where="1",
                selection_order_by="annual_return DESC, fund_code ASC",
                selection_limit=2,
                exclude_subscribe_status="暂停申购,封闭期",
                exclude_redeem_status="暂停赎回,封闭期",
                nav_data_version=None,
                clickhouse_db="fund_analysis",
                clickhouse_container="fund_clickhouse",
            )

            with patch("backtest_portfolio._run_clickhouse_query", side_effect=fake_query):
                run_backtest(args)

            positions = pd.read_csv(output_dir / "backtest_window_positions.csv")
            detail = pd.read_csv(output_dir / "backtest_window_detail.csv")
            agg = pd.read_csv(output_dir / "backtest_aggregate_summary.csv")

            self.assertTrue(positions.empty)
            self.assertEqual(int((detail["status"] == "ok").sum()), 0)
            self.assertEqual(int(agg.iloc[0]["valid_windows"]), 0)


if __name__ == "__main__":
    unittest.main()
