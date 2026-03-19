"""V2 流程 CLI 最小基线集成测试。

与 docs/V2完整流程说明.md 对齐，确保流程中用到的脚本/环境在 verify.sh 的 step1(单元测试) 与
step2(CLI smoke) 中均有覆盖。

覆盖的 V2 流程组件：
- compare_adjusted_nav_and_cum_return_window (v2 窗口化 compare)
- compare_backtest_curves
- fetch_fund_index_sw (mock akshare)
- benchmark_portfolio_backtest
- prep_data_workflow (v2)
- filter_funds_for_next_step (v2)
- run_filter_and_score
- adjusted_nav_tool (通过 test_adjusted_nav_tool 单测)
- build_filtered_purchase_csv (通过 test_pipeline_regression 单测)
- prep_eligible_window (通过 test_v2_phase0_2 单测)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
_TOOLS = _PROJECT_ROOT / "tools"
_TOOLS_PREP = _TOOLS / "prep"
_TOOLS_V2 = _TOOLS / "v2"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class V2FlowCliSmokeTest(unittest.TestCase):
    """V2 流程各脚本 CLI 最小基线 smoke，不依赖网络与 fund_infra。"""

    def test_compare_window_cli_smoke(self) -> None:
        """v2 compare_adjusted_nav_and_cum_return_window 可正常执行。"""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = root / "fund_etl"
            adj = base / "fund_adjusted_nav_by_code"
            cum = base / "fund_cum_return_by_code"
            adj.mkdir(parents=True)
            cum.mkdir(parents=True)
            out = root / "compare_out"
            out.mkdir()

            pd.DataFrame([
                {"基金代码": "000001", "净值日期": "2024-01-01", "复权净值": 1.0},
                {"基金代码": "000001", "净值日期": "2024-01-02", "复权净值": 1.01},
            ]).to_csv(adj / "000001.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([
                {"基金代码": "000001", "日期": "2024-01-01", "累计收益率": 0.0},
                {"基金代码": "000001", "日期": "2024-01-02", "累计收益率": 0.01},
            ]).to_csv(cum / "000001.csv", index=False, encoding="utf-8-sig")

            mod = __import__("v2.compare.compare_adjusted_nav_and_cum_return_window", fromlist=["main"])
            with patch.object(sys, "argv", [
                "compare_window.py",
                "--base-dir", str(base),
                "--start-date", "2024-01-01",
                "--end-date", "2024-01-02",
                "--output-dir", str(out),
            ]):
                mod.main()

            self.assertTrue((out / "summary.csv").exists())
            self.assertTrue((out / "details").is_dir())

    def test_compare_backtest_curves_cli_smoke(self) -> None:
        """compare_backtest_curves 可正常执行。"""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dir_a = root / "strategy_a"
            dir_b = root / "baseline_b"
            dir_a.mkdir()
            dir_b.mkdir()
            out = root / "compare_out"
            pd.DataFrame([
                {"date": "2024-01-01", "equity": 1.0, "cumulative_return": 0.0},
                {"date": "2024-01-02", "equity": 1.01, "cumulative_return": 0.01},
            ]).to_csv(dir_a / "equity_curve.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([
                {"date": "2024-01-01", "equity": 1.0, "cumulative_return": 0.0},
                {"date": "2024-01-02", "equity": 1.005, "cumulative_return": 0.005},
            ]).to_csv(dir_b / "equity_curve.csv", index=False, encoding="utf-8-sig")

            cmd = [
                sys.executable,
                str(_TOOLS / "compare_backtest_curves.py"),
                "--backtest-dir", str(dir_a),
                "--base-dir", str(dir_b),
                "--output-dir", str(out),
            ]
            ret = subprocess.run(cmd, cwd=str(_PROJECT_ROOT.parent), capture_output=True, text=True)
            self.assertEqual(ret.returncode, 0, f"stderr: {ret.stderr}")
            self.assertTrue((out / "backtest_curves.html").exists())

    def test_fetch_fund_index_sw_cli_smoke(self) -> None:
        """fetch_fund_index_sw 可正常执行（mock akshare，同 test_fetch_fund_index_sw）。"""
        # 需在进程内 mock，subprocess 无法继承 patch
        mock_df = pd.DataFrame({
            "日期": ["2024-01-02", "2024-01-03"],
            "收盘指数": [100.0, 101.0],
            "开盘指数": [99.0, 100.0],
            "最高指数": [101.0, 102.0],
            "最低指数": [98.0, 99.0],
            "涨跌幅": [0.0, 1.0],
        })
        if str(_TOOLS_PREP) not in sys.path:
            sys.path.insert(0, str(_TOOLS_PREP))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch("akshare.index_hist_fund_sw", return_value=mock_df):
                from fetch_fund_index_sw import run
                result = run(
                    output_root=root,
                    run_id="test_smoke",
                    start_date="2024-01-01",
                    end_date="2024-01-10",
                    request_delay=0,
                )
            self.assertGreater(len(result), 0)
            out_sub = list(root.glob("*/807*"))
            self.assertGreater(len(out_sub), 0, "应产出至少一个指数子目录")

    def test_benchmark_portfolio_backtest_cli_smoke(self) -> None:
        """benchmark_portfolio_backtest 可正常执行。"""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            etl = root / "fund_etl"
            nav_dir = etl / "fund_adjusted_nav_by_code"
            cum_dir = etl / "fund_cum_return_by_code"
            nav_dir.mkdir(parents=True)
            cum_dir.mkdir(parents=True)
            out_dir = root / "backtest_out"
            trade_csv = root / "trade_dates.csv"
            trade_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"trade_date": ["2024-01-02", "2024-01-03", "2024-01-04"]}).to_csv(
                trade_csv, index=False, encoding="utf-8-sig"
            )

            for code in ("161119", "510300"):
                pd.DataFrame([
                    {"基金代码": code, "净值日期": "2024-01-02", "复权净值": 1.0},
                    {"基金代码": code, "净值日期": "2024-01-03", "复权净值": 1.01},
                    {"基金代码": code, "净值日期": "2024-01-04", "复权净值": 1.02},
                ]).to_csv(nav_dir / f"{code}.csv", index=False, encoding="utf-8-sig")
                # 累计收益率以百分点计（1=1%），与 benchmark 的 _CUM_RETURN_BASE=100 一致
                pd.DataFrame([
                    {"基金代码": code, "日期": "2024-01-02", "累计收益率": 0.0},
                    {"基金代码": code, "日期": "2024-01-03", "累计收益率": 1.0},
                    {"基金代码": code, "日期": "2024-01-04", "累计收益率": 2.0},
                ]).to_csv(cum_dir / f"{code}.csv", index=False, encoding="utf-8-sig")

            cmd = [
                sys.executable,
                str(_TOOLS_V2 / "benchmark_portfolio_backtest.py"),
                "--fund-etl-dir", str(etl),
                "--start-date", "2024-01-02",
                "--end-date", "2024-01-04",
                "--rebalance-interval", "1",
                "--portfolio", "161119:0.70,510300:0.30",
                "--output-dir", str(out_dir),
                "--trade-dates-csv", str(trade_csv),
            ]
            env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT / "src")}
            ret = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), capture_output=True, text=True, env=env)
            self.assertEqual(ret.returncode, 0, f"stderr: {ret.stderr}")
            self.assertTrue((out_dir / "equity_curve.csv").exists())
            self.assertTrue((out_dir / "summary.csv").exists())

    def test_prep_data_workflow_v2_cli_help(self) -> None:
        """prep_data_workflow (v2) CLI 可解析参数。"""
        cmd = [
            sys.executable,
            str(_TOOLS_V2 / "prep_data_workflow.py"),
            "-h",
        ]
        ret = subprocess.run(cmd, cwd=str(_PROJECT_ROOT.parent), capture_output=True, text=True)
        self.assertEqual(ret.returncode, 0)
        self.assertIn("--work-dir", ret.stdout)

    def test_filter_funds_for_next_step_cli_smoke(self) -> None:
        """filter_funds_for_next_step (v2) 可正常执行。"""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = root / "fund_etl"
            base.mkdir()
            (base / "fund_nav_by_code").mkdir()
            (base / "fund_adjusted_nav_by_code").mkdir()
            compare_details = root / "compare_details"
            compare_details.mkdir()
            integrity_details = root / "integrity_details"
            integrity_details.mkdir()

            pd.DataFrame([{
                "基金代码": "000001", "基金简称": "A",
                "申购状态": "开放申购", "赎回状态": "开放赎回",
                "购买起点": "1", "日累计限定金额": "50000", "手续费": "0.1",
            }]).to_csv(base / "fund_purchase.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([{"基金代码": "000001", "成立日期/规模": "2010-01-01"}]).to_csv(
                base / "fund_overview.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame([
                {"基金代码": "000001", "净值日期": "2024-01-02", "复权净值": 1.0},
            ]).to_csv(base / "fund_adjusted_nav_by_code" / "000001.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([
                {"基金代码": "000001", "净值日期": "2024-01-02", "单位净值": 1.0},
            ]).to_csv(base / "fund_nav_by_code" / "000001.csv", index=False, encoding="utf-8-sig")
            # compare detail: 000001.csv，偏差 < 2% 通过规则4
            pd.DataFrame([
                {"期初日期": "2024-01-02", "期末日期": "2024-01-03", "本地远程收益率偏差": "0.01"},
            ]).to_csv(compare_details / "000001.csv", index=False, encoding="utf-8-sig")
            # integrity detail: 通过规则5
            pd.DataFrame([
                {"交易日日期": "2024-01-02", "该日期数据是否存在": "是"},
            ]).to_csv(integrity_details / "000001_2024-01-01_2024-12-31.csv", index=False, encoding="utf-8-sig")

            out_csv = root / "filter_result.csv"
            cmd = [
                sys.executable,
                str(_PROJECT_ROOT / "src" / "v2" / "filters" / "filter_funds_for_next_step.py"),
                "--base-dir", str(base),
                "--compare-details-dir", str(compare_details),
                "--integrity-details-dir", str(integrity_details),
                "--start-date", "2024-01-01",
                "--end-date", "2024-12-31",
                "--output-csv", str(out_csv),
            ]
            env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT / "src")}
            ret = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), capture_output=True, text=True, env=env)
            self.assertEqual(ret.returncode, 0, f"stderr: {ret.stderr}")
            self.assertTrue(out_csv.exists())
            df = pd.read_csv(out_csv, dtype=str, encoding="utf-8-sig")
            self.assertFalse(df.empty)
            self.assertIn("基金编码", df.columns)

    def test_run_filter_and_score_cli_smoke(self) -> None:
        """run_filter_and_score 可正常执行。"""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            scoreboard = root / "scoreboard.csv"
            work_dir = root / "filter_score_work"
            work_dir.mkdir()

            pd.DataFrame([
                {"基金代码": "000001", "基金名称": "A", "近3年年化收益率": "5", "近1年年化收益率": "4",
                 "近3年上涨季度比例": "85", "近3年上涨月份比例": "75", "近3年月涨跌幅标准差": "1.0",
                 "近1年夏普比率": "1.2", "近3年夏普比率": "1.1", "近1年卡玛比率": "1.5", "近3年卡玛比率": "1.3"},
            ]).to_csv(scoreboard, index=False, encoding="utf-8-sig")

            env = {"PYTHONPATH": str(_PROJECT_ROOT.parent)}
            cmd = [
                sys.executable, "-m", "myanalyser.src.filter_score.filter_and_score_main",
                "-i", str(scoreboard),
                "-w", str(work_dir),
                "-f", str(_PROJECT_ROOT / "src" / "filter_score" / "filters" / "most_stable.py"),
                "-s", str(_PROJECT_ROOT / "src" / "filter_score" / "scores" / "low_risk_debt.py"),
            ]
            ret = subprocess.run(cmd, cwd=str(_PROJECT_ROOT.parent), capture_output=True, text=True, env={**__import__("os").environ, **env})
            self.assertEqual(ret.returncode, 0, f"stderr: {ret.stderr}")
            self.assertTrue((work_dir / "filter_result.csv").exists())
            self.assertTrue((work_dir / "scored_result.csv").exists())


if __name__ == "__main__":
    unittest.main()
