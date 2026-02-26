from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adjusted_nav_tool import calculate_adjusted_nav, process_all_funds


class AdjustedNavToolTest(unittest.TestCase):
    def test_calculate_adjusted_nav_with_dividend_and_split(self) -> None:
        nav = pd.DataFrame(
            [
                {"净值日期": "2024-01-01", "单位净值": 1.0},
                {"净值日期": "2024-01-02", "单位净值": 2.0},
                {"净值日期": "2024-01-03", "单位净值": 1.0},
            ]
        )
        bonus = pd.DataFrame([{"除息日": "2024-01-02", "每份分红": "每份派现金0.5元"}])
        split = pd.DataFrame([{"拆分折算日": "2024-01-03", "拆分折算比例": "1:3"}])

        out = calculate_adjusted_nav(df_nav=nav, df_dividend=bonus, df_split=split)

        self.assertEqual(out["净值日期"].tolist(), ["2024-01-01", "2024-01-02", "2024-01-03"])
        self.assertAlmostEqual(out.loc[0, "cumulative_factor"], 1.0)
        self.assertAlmostEqual(out.loc[1, "cumulative_factor"], 1.25)
        self.assertAlmostEqual(out.loc[2, "cumulative_factor"], 3.75)
        self.assertAlmostEqual(out.loc[2, "复权净值"], 3.75)

    def test_process_sample_163402(self) -> None:
        base = Path(__file__).resolve().parent.parent / "data" / "samples"
        with tempfile.TemporaryDirectory() as d:
            output_dir = Path(d) / "adjusted"
            summary = process_all_funds(
                nav_dir=base / "fund_nav_by_code",
                bonus_dir=base / "fund_bonus_by_code",
                split_dir=base / "fund_split_by_code",
                output_dir=output_dir,
                codes=["163402"],
            )

            self.assertEqual(summary["funds"], 1)
            out_csv = output_dir / "163402.csv"
            self.assertTrue(out_csv.exists())

            out_df = pd.read_csv(out_csv, dtype={"基金代码": str})
            self.assertFalse(out_df.empty)
            self.assertIn("复权净值", out_df.columns.tolist())
            self.assertIn("cumulative_factor", out_df.columns.tolist())

            split_row = out_df[out_df["净值日期"] == "2007-05-11"]
            self.assertEqual(len(split_row), 1)
            self.assertAlmostEqual(float(split_row.iloc[0]["cumulative_factor"]), 4.3884424059, places=6)

    def test_skip_and_log_when_bonus_and_split_both_missing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            nav_dir = base / "nav"
            bonus_dir = base / "bonus"
            split_dir = base / "split"
            output_dir = base / "out"
            nav_dir.mkdir(parents=True, exist_ok=True)
            bonus_dir.mkdir(parents=True, exist_ok=True)
            split_dir.mkdir(parents=True, exist_ok=True)

            nav_df = pd.DataFrame([{"基金代码": "000001", "净值日期": "2024-01-01", "单位净值": 1.0}])
            nav_df.to_csv(nav_dir / "000001.csv", index=False, encoding="utf-8-sig")

            summary = process_all_funds(
                nav_dir=nav_dir,
                bonus_dir=bonus_dir,
                split_dir=split_dir,
                output_dir=output_dir,
                codes=["000001"],
            )

            self.assertEqual(summary["funds"], 0)
            self.assertEqual(summary["failed"], 1)
            self.assertFalse((output_dir / "000001.csv").exists())

            fail_log = output_dir / "failed_adjusted_nav.jsonl"
            self.assertTrue(fail_log.exists())
            rec = json.loads(fail_log.read_text(encoding="utf-8").strip().splitlines()[-1])
            self.assertEqual(rec["code"], "000001")
            self.assertTrue(
                "missing bonus files" in rec["error"] or "missing split files" in rec["error"]
            )

    def test_skip_when_output_already_exists(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            nav_dir = base / "nav"
            bonus_dir = base / "bonus"
            split_dir = base / "split"
            output_dir = base / "out"
            nav_dir.mkdir(parents=True, exist_ok=True)
            bonus_dir.mkdir(parents=True, exist_ok=True)
            split_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            nav_df = pd.DataFrame([{"基金代码": "000001", "净值日期": "2024-01-01", "单位净值": 1.0}])
            bonus_df = pd.DataFrame(columns=["基金代码", "年份", "权益登记日", "除息日", "每份分红", "分红发放日"])
            split_df = pd.DataFrame(columns=["基金代码", "年份", "拆分折算日", "拆分类型", "拆分折算比例"])
            nav_df.to_csv(nav_dir / "000001.csv", index=False, encoding="utf-8-sig")
            bonus_df.to_csv(bonus_dir / "000001.csv", index=False, encoding="utf-8-sig")
            split_df.to_csv(split_dir / "000001.csv", index=False, encoding="utf-8-sig")
            (output_dir / "000001.csv").write_text("基金代码,净值日期,单位净值,复权净值,cumulative_factor\n", encoding="utf-8")

            summary = process_all_funds(
                nav_dir=nav_dir,
                bonus_dir=bonus_dir,
                split_dir=split_dir,
                output_dir=output_dir,
                codes=["000001"],
            )

            self.assertEqual(summary["total"], 1)
            self.assertEqual(summary["funds"], 0)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["skipped"], 1)

    def test_fail_when_dividend_date_not_in_nav_calendar(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            nav_dir = base / "nav"
            bonus_dir = base / "bonus"
            split_dir = base / "split"
            output_dir = base / "out"
            nav_dir.mkdir(parents=True, exist_ok=True)
            bonus_dir.mkdir(parents=True, exist_ok=True)
            split_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            nav_df = pd.DataFrame(
                [
                    {"基金代码": "000001", "净值日期": "2024-01-01", "单位净值": 1.0},
                    {"基金代码": "000001", "净值日期": "2024-01-08", "单位净值": 1.1},
                ]
            )
            bonus_df = pd.DataFrame(
                [
                    {
                        "基金代码": "000001",
                        "年份": "2024年",
                        "权益登记日": "2024-01-03",
                        "除息日": "2024-01-03",
                        "每份分红": "每份派现金0.0500元",
                        "分红发放日": "2024-01-05",
                    }
                ]
            )
            split_df = pd.DataFrame(columns=["基金代码", "年份", "拆分折算日", "拆分类型", "拆分折算比例"])
            nav_df.to_csv(nav_dir / "000001.csv", index=False, encoding="utf-8-sig")
            bonus_df.to_csv(bonus_dir / "000001.csv", index=False, encoding="utf-8-sig")
            split_df.to_csv(split_dir / "000001.csv", index=False, encoding="utf-8-sig")

            summary = process_all_funds(
                nav_dir=nav_dir,
                bonus_dir=bonus_dir,
                split_dir=split_dir,
                output_dir=output_dir,
                codes=["000001"],
            )

            self.assertEqual(summary["funds"], 0)
            self.assertEqual(summary["failed"], 1)
            self.assertFalse((output_dir / "000001.csv").exists())
            fail_log = output_dir / "failed_adjusted_nav.jsonl"
            self.assertTrue(fail_log.exists())
            rec = json.loads(fail_log.read_text(encoding="utf-8").strip().splitlines()[-1])
            self.assertEqual(rec["code"], "000001")
            self.assertIn("missing nav data on dividend dates", rec["error"])
            self.assertIn("2024-01-03", rec["error"])

    def test_allow_missing_dividend_date_when_before_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            nav_dir = base / "nav"
            bonus_dir = base / "bonus"
            split_dir = base / "split"
            output_dir = base / "out"
            nav_dir.mkdir(parents=True, exist_ok=True)
            bonus_dir.mkdir(parents=True, exist_ok=True)
            split_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            nav_df = pd.DataFrame(
                [
                    {"基金代码": "000001", "净值日期": "2024-01-01", "单位净值": 1.0},
                    {"基金代码": "000001", "净值日期": "2024-01-08", "单位净值": 1.1},
                ]
            )
            bonus_df = pd.DataFrame(
                [
                    {
                        "基金代码": "000001",
                        "年份": "2024年",
                        "权益登记日": "2024-01-03",
                        "除息日": "2024-01-03",
                        "每份分红": "每份派现金0.0500元",
                        "分红发放日": "2024-01-05",
                    }
                ]
            )
            split_df = pd.DataFrame(columns=["基金代码", "年份", "拆分折算日", "拆分类型", "拆分折算比例"])
            nav_df.to_csv(nav_dir / "000001.csv", index=False, encoding="utf-8-sig")
            bonus_df.to_csv(bonus_dir / "000001.csv", index=False, encoding="utf-8-sig")
            split_df.to_csv(split_dir / "000001.csv", index=False, encoding="utf-8-sig")

            summary = process_all_funds(
                nav_dir=nav_dir,
                bonus_dir=bonus_dir,
                split_dir=split_dir,
                output_dir=output_dir,
                codes=["000001"],
                allow_missing_event_until=pd.Timestamp("2024-01-03"),
            )

            self.assertEqual(summary["funds"], 1)
            self.assertEqual(summary["failed"], 0)
            self.assertTrue((output_dir / "000001.csv").exists())

    def test_still_fail_when_missing_dividend_date_after_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            nav_dir = base / "nav"
            bonus_dir = base / "bonus"
            split_dir = base / "split"
            output_dir = base / "out"
            nav_dir.mkdir(parents=True, exist_ok=True)
            bonus_dir.mkdir(parents=True, exist_ok=True)
            split_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            nav_df = pd.DataFrame(
                [
                    {"基金代码": "000001", "净值日期": "2024-01-01", "单位净值": 1.0},
                    {"基金代码": "000001", "净值日期": "2024-01-08", "单位净值": 1.1},
                ]
            )
            bonus_df = pd.DataFrame(
                [
                    {
                        "基金代码": "000001",
                        "年份": "2024年",
                        "权益登记日": "2024-01-03",
                        "除息日": "2024-01-03",
                        "每份分红": "每份派现金0.0500元",
                        "分红发放日": "2024-01-05",
                    }
                ]
            )
            split_df = pd.DataFrame(columns=["基金代码", "年份", "拆分折算日", "拆分类型", "拆分折算比例"])
            nav_df.to_csv(nav_dir / "000001.csv", index=False, encoding="utf-8-sig")
            bonus_df.to_csv(bonus_dir / "000001.csv", index=False, encoding="utf-8-sig")
            split_df.to_csv(split_dir / "000001.csv", index=False, encoding="utf-8-sig")

            summary = process_all_funds(
                nav_dir=nav_dir,
                bonus_dir=bonus_dir,
                split_dir=split_dir,
                output_dir=output_dir,
                codes=["000001"],
                allow_missing_event_until=pd.Timestamp("2024-01-02"),
            )

            self.assertEqual(summary["funds"], 0)
            self.assertEqual(summary["failed"], 1)
            self.assertFalse((output_dir / "000001.csv").exists())


if __name__ == "__main__":
    unittest.main()
