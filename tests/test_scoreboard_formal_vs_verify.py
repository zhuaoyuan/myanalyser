"""formal 与 pipeline 核验路径的回测对比测试。使用 verify 数据（若存在）验证两者结果一致。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pipeline_scoreboard
from verify_scoreboard_recalc import run_verification


# 可选的 verify 数据路径，用于回测
VERIFY_RUN_ID = "20260227_094810_verify_e2e"
VERIFY_DATA_VERSION = f"{VERIFY_RUN_ID}_db"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERIFY_FUND_ETL = PROJECT_ROOT / "data" / "versions" / VERIFY_RUN_ID / "fund_etl"
VERIFY_ARTIFACTS = PROJECT_ROOT / "artifacts" / f"verify_{VERIFY_RUN_ID}"
VERIFY_SCOREBOARD_ORIG = VERIFY_ARTIFACTS / "scoreboard" / f"fund_scoreboard_{VERIFY_DATA_VERSION}.csv"
VERIFY_FILTERED_PURCHASE = VERIFY_FUND_ETL / "fund_purchase_for_step10_filtered.csv"


class ScoreboardFormalVsVerifyTest(unittest.TestCase):
    @unittest.skipUnless(
        VERIFY_FUND_ETL.exists()
        and VERIFY_SCOREBOARD_ORIG.exists()
        and VERIFY_FILTERED_PURCHASE.exists(),
        f"需要 verify 数据: {VERIFY_FUND_ETL}, {VERIFY_SCOREBOARD_ORIG}, {VERIFY_FILTERED_PURCHASE}",
    )
    def test_formal_output_matches_verify_recalc(self) -> None:
        """使用 verify 数据：formal 产出经 verify_scoreboard_recalc 核验应全部通过。"""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out_formal = root / "scoreboard_formal"
            out_formal.mkdir(parents=True)
            recheck_dir = root / "scoreboard_recheck"

            overview_csv = VERIFY_FUND_ETL / "fund_overview.csv"
            nav_dir = VERIFY_FUND_ETL / "fund_adjusted_nav_by_code"
            personnel_dir = VERIFY_FUND_ETL / "fund_personnel_by_code"
            # 从 nav 推断 as_of_date
            max_date = None
            for p in nav_dir.glob("*.csv"):
                df = pd.read_csv(p, dtype={"净值日期": str}, encoding="utf-8-sig")
                if "净值日期" in df.columns:
                    ds = pd.to_datetime(df["净值日期"], errors="coerce").dropna()
                    if not ds.empty:
                        m = ds.max()
                        if max_date is None or m > max_date:
                            max_date = m
            self.assertIsNotNone(max_date, "nav 目录应有有效净值")
            as_of_str = max_date.strftime("%Y-%m-%d")

            args = pipeline_scoreboard.build_parser().parse_args([
                "--purchase-csv", str(VERIFY_FILTERED_PURCHASE),
                "--overview-csv", str(overview_csv),
                "--personnel-dir", str(personnel_dir),
                "--nav-dir", str(nav_dir),
                "--output-dir", str(out_formal),
                "--data-version", "test_formal",
                "--as-of-date", as_of_str,
                "--stale-max-days", "3650",
                "--formal-only",
            ])
            pipeline_scoreboard.run_pipeline(args)

            formal_csv = out_formal / "fund_scoreboard_test_formal.csv"
            self.assertTrue(formal_csv.exists(), "formal 应产出 scoreboard CSV")

            result = run_verification(
                scoreboard_csv=formal_csv,
                fund_etl_dir=VERIFY_FUND_ETL,
                output_dir=recheck_dir,
                max_input_rows=200,
            )
            summary_df = pd.read_csv(result["summary_csv"], dtype=str, encoding="utf-8-sig")
            failed = summary_df[summary_df["待核验字段是否全部核验通过"] != "是"]
            self.assertTrue(
                failed.empty,
                f"formal 产出核验应全部通过，失败: {failed[['基金代码', '未通过字段名']].to_dict('records')}",
            )
