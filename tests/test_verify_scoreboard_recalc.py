from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verify_scoreboard_recalc import (
    VERIFY_FIELDS,
    _build_recalc_metrics_with_latest_nav_date,
    _to_display_value,
    run_verification,
)


class VerifyScoreboardRecalcTest(unittest.TestCase):
    def test_run_verification_outputs_summary_and_details(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fund_etl_dir = root / "fund_etl"
            nav_dir = fund_etl_dir / "fund_adjusted_nav_by_code"
            nav_dir.mkdir(parents=True, exist_ok=True)
            output_dir = root / "recheck"
            scoreboard_csv = root / "fund_scoreboard.csv"

            pd.DataFrame(
                [
                    {"基金代码": "000001", "净值日期": "2024-01-01", "复权净值": 1.0},
                ]
            ).to_csv(nav_dir / "000001.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {"基金代码": "000002", "净值日期": "2024-01-01", "复权净值": 1.0},
                ]
            ).to_csv(nav_dir / "000002.csv", index=False, encoding="utf-8-sig")

            base_df = pd.DataFrame([{"基金代码": "000001"}, {"基金代码": "000002"}], dtype=str)
            recalc_df = _build_recalc_metrics_with_latest_nav_date(base_df, nav_dir, latest_nav_date=None)
            merged = base_df.merge(recalc_df, on="基金代码", how="left")

            rows: list[dict[str, object]] = []
            for _, row in merged.iterrows():
                one = {"基金代码": row["基金代码"]}
                for field_name, internal_name, style in VERIFY_FIELDS:
                    one[field_name] = _to_display_value(row.get(internal_name), style=style)
                rows.append(one)
            pd.DataFrame(rows).to_csv(scoreboard_csv, index=False, encoding="utf-8-sig")

            result = run_verification(
                scoreboard_csv=scoreboard_csv,
                fund_etl_dir=fund_etl_dir,
                output_dir=output_dir,
                max_input_rows=200,
            )
            self.assertTrue(result["summary_csv"].exists())
            self.assertTrue(result["details_dir"].exists())
            self.assertTrue(result["metrics_recalc_sample_csv"].exists())

            summary_df = pd.read_csv(result["summary_csv"], dtype=str, encoding="utf-8-sig")
            self.assertEqual(summary_df.shape[0], 2)
            self.assertSetEqual(
                set(summary_df["待核验字段是否全部核验通过"].tolist()),
                {"是"},
            )

            detail_1 = pd.read_csv(result["details_dir"] / "000001.csv", dtype=str, encoding="utf-8-sig")
            self.assertEqual(detail_1.shape[0], len(VERIFY_FIELDS))
            self.assertSetEqual(set(detail_1["核验是否通过"].tolist()), {"是"})

    def test_run_verification_raise_when_rows_over_limit(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fund_etl_dir = root / "fund_etl"
            nav_dir = fund_etl_dir / "fund_adjusted_nav_by_code"
            nav_dir.mkdir(parents=True, exist_ok=True)
            output_dir = root / "recheck"
            scoreboard_csv = root / "fund_scoreboard.csv"

            rows: list[dict[str, object]] = []
            for i in range(201):
                row = {"基金代码": str(i + 1).zfill(6)}
                for field_name, _, _ in VERIFY_FIELDS:
                    row[field_name] = ""
                rows.append(row)
            pd.DataFrame(rows).to_csv(scoreboard_csv, index=False, encoding="utf-8-sig")

            with self.assertRaises(ValueError):
                run_verification(
                    scoreboard_csv=scoreboard_csv,
                    fund_etl_dir=fund_etl_dir,
                    output_dir=output_dir,
                    max_input_rows=200,
                )


if __name__ == "__main__":
    unittest.main()
