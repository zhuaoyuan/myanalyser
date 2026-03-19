"""V2 最小基线完整集成测试：固定输入预跑全流程，回归时逐环节校验产物与预期一致。

与 docs/V2完整流程说明.md 对齐，覆盖 step5(复权净值)~step10(筛选打分+重算)。
输入：tests/baseline/mini_case_v2/input/
预期：tests/baseline/mini_case_v2/expected/default/
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
BASELINE_ROOT = PROJECT / "tests" / "baseline" / "mini_case_v2"
INPUT_ROOT = BASELINE_ROOT / "input"
DEFAULT_EXPECTED = BASELINE_ROOT / "expected" / "default"


def _expected_dir() -> Path:
    env_path = os.getenv("MYANALYSER_BASELINE_V2_EXPECTED_DIR", "").strip()
    return Path(env_path).resolve() if env_path else DEFAULT_EXPECTED.resolve()


def _read_csv(path: Path, code_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if code_col and code_col in df.columns:
        df[code_col] = df[code_col].astype(str).str.strip().str.zfill(6)
        df = df.sort_values(code_col).reset_index(drop=True)
    return df


def _assert_csv_equal(actual: Path, expected: Path, code_col: str | None = None) -> None:
    a = _read_csv(actual, code_col=code_col)
    e = _read_csv(expected, code_col=code_col)
    pd.testing.assert_frame_equal(a, e, check_dtype=False)


class V2BaselineRegressionTest(unittest.TestCase):
    """V2 全流程回归：固定输入 → 跑 step5~10 → 逐环节对比 expected。"""

    def test_v2_baseline_full_flow_regression(self) -> None:
        expected_dir = _expected_dir()
        self.assertTrue(expected_dir.exists(), f"expected_dir not found: {expected_dir}")
        self.assertTrue(INPUT_ROOT.exists(), f"input not found: {INPUT_ROOT}")

        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            input_copy = work / "input"
            output_root = work / "output"
            shutil.copytree(INPUT_ROOT, input_copy)
            output_root.mkdir(parents=True)
            fund_etl = input_copy / "fund_etl"
            artifacts = output_root / "artifacts"
            artifacts.mkdir(parents=True)
            trade_dates = PROJECT / "data" / "common" / "trade_dates.csv"
            env = {**os.environ, "PYTHONPATH": str(PROJECT / "src")}

            # Step5: adjusted_nav
            subprocess.run(
                [sys.executable, "src/adjusted_nav_tool.py",
                 "--nav-dir", str(fund_etl / "fund_nav_by_code"),
                 "--bonus-dir", str(fund_etl / "fund_bonus_by_code"),
                 "--split-dir", str(fund_etl / "fund_split_by_code"),
                 "--output-dir", str(fund_etl / "fund_adjusted_nav_by_code"),
                 "--allow-missing-event-until", "2020-12-31"],
                check=True, capture_output=True, cwd=str(PROJECT),
            )
            # Create cum_return from adjusted_nav
            for p in (fund_etl / "fund_adjusted_nav_by_code").glob("*.csv"):
                df = pd.read_csv(p, dtype=str)
                if "净值日期" not in df.columns or "复权净值" not in df.columns:
                    continue
                df["净值日期"] = pd.to_datetime(df["净值日期"], errors="coerce")
                df["复权净值"] = pd.to_numeric(df["复权净值"], errors="coerce")
                df = df.dropna(subset=["净值日期", "复权净值"]).sort_values("净值日期")
                if df.empty or len(df) < 2:
                    continue
                code = p.stem.zfill(6)
                base = float(df["复权净值"].iloc[0])
                if base <= 0:
                    continue
                cum = (df["复权净值"] / base - 1) * 100
                out = fund_etl / "fund_cum_return_by_code" / f"{code}.csv"
                out.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({
                    "基金代码": code,
                    "日期": df["净值日期"].dt.strftime("%Y-%m-%d"),
                    "累计收益率": cum.round(6).astype(str),
                }).to_csv(out, index=False, encoding="utf-8-sig")

            # Step6: integrity
            subprocess.run(
                [sys.executable, "src/check_trade_day_data_integrity.py",
                 "--base-dir", str(fund_etl),
                 "--start-date", "2025-01-01", "--end-date", "2025-12-31",
                 "--trade-dates-csv", str(trade_dates),
                 "--output-dir", str(artifacts / "trade_day_integrity_reports")],
                check=True, capture_output=True, cwd=str(PROJECT),
            )
            # Step7: compare
            subprocess.run(
                [sys.executable, "-m", "v2.compare.compare_adjusted_nav_and_cum_return_window",
                 "--base-dir", str(fund_etl),
                 "--start-date", "2025-01-01", "--end-date", "2025-12-31",
                 "--output-dir", str(artifacts / "fund_return_compare")],
                check=True, capture_output=True, cwd=str(PROJECT), env=env,
            )
            # Step8: filter + build_filtered
            purchase_csv = fund_etl / "fund_purchase_effective.csv" if (fund_etl / "fund_purchase_effective.csv").exists() else fund_etl / "fund_purchase.csv"
            filter_csv = artifacts / "filtered_fund_candidates.csv"
            subprocess.run(
                [sys.executable, str(PROJECT / "src" / "v2" / "filters" / "filter_funds_for_next_step.py"),
                 "--base-dir", str(fund_etl), "--purchase-csv", str(purchase_csv),
                 "--compare-details-dir", str(artifacts / "fund_return_compare" / "details"),
                 "--integrity-details-dir", str(artifacts / "trade_day_integrity_reports" / "details_2025-01-01_2025-12-31"),
                 "--start-date", "2023-01-01", "--end-date", "2025-12-31", "--max-abs-deviation", "0.02",
                 "--output-csv", str(filter_csv)],
                check=True, capture_output=True, cwd=str(PROJECT), env=env,
            )
            filtered_purchase = artifacts / "fund_purchase_for_step10_filtered.csv"
            subprocess.run(
                [sys.executable, "src/transforms/build_filtered_purchase_csv.py",
                 "--purchase-csv", str(purchase_csv), "--filter-csv", str(filter_csv),
                 "--output-csv", str(filtered_purchase)],
                check=True, capture_output=True, cwd=str(PROJECT),
            )
            # Step9: scoreboard
            nav_dir = fund_etl / "fund_adjusted_nav_by_code"
            max_date = None
            for p in nav_dir.glob("*.csv"):
                df = pd.read_csv(p, dtype={"净值日期": str}, encoding="utf-8-sig")
                if "净值日期" not in df.columns:
                    continue
                ds = pd.to_datetime(df["净值日期"], errors="coerce").dropna()
                if not ds.empty and (max_date is None or ds.max() > max_date):
                    max_date = ds.max()
            as_of = max_date.strftime("%Y-%m-%d") if max_date is not None else "2025-12-31"
            scoreboard_dir = artifacts / "scoreboard"
            scoreboard_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, "src/pipeline_scoreboard.py",
                 "--purchase-csv", str(filtered_purchase),
                 "--overview-csv", str(fund_etl / "fund_overview.csv"),
                 "--personnel-dir", str(fund_etl / "fund_personnel_by_code"),
                 "--nav-dir", str(nav_dir),
                 "--output-dir", str(scoreboard_dir),
                 "--data-version", "baseline_v2",
                 "--as-of-date", as_of,
                 "--stale-max-days", "3650",
                 "--skip-sinks", "--formal-only"],
                check=True, capture_output=True, cwd=str(PROJECT),
            )
            # Step10: filter_score + recalc
            filter_score_dir = artifacts / "filter_score"
            subprocess.run(
                ["bash", "tools/run_filter_and_score.sh",
                 "-i", str(scoreboard_dir / "fund_scoreboard_baseline_v2.csv"),
                 "-w", str(filter_score_dir),
                 "-f", "src/filter_score/filters/most_stable.py",
                 "-s", "src/filter_score/scores/low_risk_debt.py"],
                check=True, capture_output=True, cwd=str(PROJECT),
            )
            subprocess.run(
                [sys.executable, "src/verify_scoreboard_recalc.py",
                 "--scoreboard-csv", str(scoreboard_dir / "fund_scoreboard_baseline_v2.csv"),
                 "--fund-etl-dir", str(fund_etl),
                 "--output-dir", str(artifacts / "scoreboard_recheck"),
                 "--max-input-rows", "50"],
                check=True, capture_output=True, cwd=str(PROJECT),
            )

            # Regression: compare each step's output to expected
            exp_adj = expected_dir / "fund_adjusted_nav_by_code"
            for p in exp_adj.glob("*.csv"):
                _assert_csv_equal(fund_etl / "fund_adjusted_nav_by_code" / p.name, p)

            exp_int_summary = list((expected_dir / "trade_day_integrity_reports").glob("trade_day_integrity_summary_*.csv"))
            if exp_int_summary:
                act_summary = list((artifacts / "trade_day_integrity_reports").glob("trade_day_integrity_summary_*.csv"))
                self.assertEqual(len(act_summary), len(exp_int_summary))
                for e in exp_int_summary:
                    a = artifacts / "trade_day_integrity_reports" / e.name
                    self.assertTrue(a.exists(), f"missing {a}")
                    _assert_csv_equal(a, e)

            _assert_csv_equal(filter_csv, expected_dir / "filtered_fund_candidates.csv", code_col="基金编码")
            _assert_csv_equal(filtered_purchase, expected_dir / "fund_purchase_for_step10_filtered.csv", code_col="基金代码")
            _assert_csv_equal(
                scoreboard_dir / "fund_scoreboard_baseline_v2.csv",
                expected_dir / "fund_scoreboard_baseline_v2.csv",
                code_col="基金代码",
            )
            _assert_csv_equal(filter_score_dir / "filter_result.csv", expected_dir / "filter_result.csv", code_col="基金代码")
            _assert_csv_equal(filter_score_dir / "scored_result.csv", expected_dir / "scored_result.csv", code_col="基金代码")
            _assert_csv_equal(
                artifacts / "scoreboard_recheck" / "summary.csv",
                expected_dir / "scoreboard_recheck" / "summary.csv",
                code_col="基金代码",
            )


if __name__ == "__main__":
    unittest.main()
