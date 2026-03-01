from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pipeline_scoreboard
import verify_scoreboard_recalc
from filter_funds_for_next_step import filter_funds_for_next_step
from transforms.build_filtered_purchase_csv import build_filtered_purchase_csv


BASELINE_ROOT = Path(__file__).resolve().parent / "baseline" / "mini_case"
DEFAULT_EXPECTED_DIR = BASELINE_ROOT / "expected" / "default"


def _expected_dir() -> Path:
    env_path = os.getenv("MYANALYSER_BASELINE_EXPECTED_DIR", "").strip()
    return Path(env_path).resolve() if env_path else DEFAULT_EXPECTED_DIR.resolve()


def _read_csv(path: Path, code_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if code_col and code_col in df.columns:
        df[code_col] = df[code_col].astype(str).str.strip().str.zfill(6)
        df = df.sort_values(code_col).reset_index(drop=True)
    return df


class PipelineRegressionBaselineTest(unittest.TestCase):
    def test_baseline_filter_scoreboard_and_recalc(self) -> None:
        expected_dir = _expected_dir()
        self.assertTrue(expected_dir.exists(), f"expected_dir not found: {expected_dir}")

        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            input_root = work / "input"
            output_root = work / "output"
            shutil.copytree(BASELINE_ROOT / "input", input_root)
            output_root.mkdir(parents=True, exist_ok=True)

            fund_etl_dir = input_root / "fund_etl"
            purchase_csv = fund_etl_dir / "fund_purchase_effective.csv"
            if not purchase_csv.exists():
                purchase_csv = fund_etl_dir / "fund_purchase.csv"
            filter_df = filter_funds_for_next_step(
                purchase_csv=purchase_csv,
                overview_csv=fund_etl_dir / "fund_overview.csv",
                nav_dir=fund_etl_dir / "fund_nav_by_code",
                adjusted_nav_dir=fund_etl_dir / "fund_adjusted_nav_by_code",
                compare_details_dir=input_root / "artifacts" / "fund_return_compare" / "details",
                integrity_details_dir=input_root
                / "artifacts"
                / "trade_day_integrity_reports"
                / "details_2025-01-01_2025-12-31",
                start_date="2023-01-01",
                max_abs_deviation=0.02,
            )
            filter_csv = output_root / "filtered_fund_candidates.csv"
            filter_df.to_csv(filter_csv, index=False, encoding="utf-8-sig")

            expected_filter = _read_csv(expected_dir / "filtered_fund_candidates.csv", code_col="基金编码")
            actual_filter = _read_csv(filter_csv, code_col="基金编码")
            pd.testing.assert_frame_equal(actual_filter, expected_filter, check_dtype=False)

            filtered_purchase_csv = output_root / "fund_purchase_for_step10_filtered.csv"
            build_filtered_purchase_csv(
                purchase_csv=purchase_csv,
                filter_csv=filter_csv,
                output_csv=filtered_purchase_csv,
            )
            expected_purchase = _read_csv(expected_dir / "fund_purchase_for_step10_filtered.csv", code_col="基金代码")
            actual_purchase = _read_csv(filtered_purchase_csv, code_col="基金代码")
            pd.testing.assert_frame_equal(actual_purchase, expected_purchase, check_dtype=False)

            args = pipeline_scoreboard.build_parser().parse_args(
                [
                    "--purchase-csv",
                    str(filtered_purchase_csv),
                    "--overview-csv",
                    str(fund_etl_dir / "fund_overview.csv"),
                    "--personnel-dir",
                    str(fund_etl_dir / "fund_personnel_by_code"),
                    "--nav-dir",
                    str(fund_etl_dir / "fund_adjusted_nav_by_code"),
                    "--output-dir",
                    str(output_root),
                    "--data-version",
                    "baseline_v1",
                    "--as-of-date",
                    "2026-02-27",
                    "--stale-max-days",
                    "3650",
                    "--formal-only",
                ]
            )
            pipeline_scoreboard.run_pipeline(args)

            expected_scoreboard = _read_csv(expected_dir / "fund_scoreboard_baseline_v1.csv", code_col="基金代码")
            actual_scoreboard = _read_csv(output_root / "fund_scoreboard_baseline_v1.csv", code_col="基金代码")
            pd.testing.assert_frame_equal(actual_scoreboard, expected_scoreboard, check_dtype=False)

            verify_out = output_root / "scoreboard_recheck"
            verify_scoreboard_recalc.run_verification(
                scoreboard_csv=output_root / "fund_scoreboard_baseline_v1.csv",
                fund_etl_dir=fund_etl_dir,
                output_dir=verify_out,
                max_input_rows=50,
            )
            expected_recheck = _read_csv(expected_dir / "scoreboard_recheck" / "summary.csv", code_col="基金代码")
            actual_recheck = _read_csv(verify_out / "summary.csv", code_col="基金代码")
            pd.testing.assert_frame_equal(actual_recheck, expected_recheck, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
