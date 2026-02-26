from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compare_adjusted_nav_and_cum_return import compare_adjusted_nav_and_cum_return


class CompareAdjustedNavAndCumReturnTest(unittest.TestCase):
    def test_missing_fund_file_logged(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            adjusted = base / "fund_adjusted_nav_by_code"
            cum = base / "fund_cum_return_by_code"
            adjusted.mkdir(parents=True, exist_ok=True)
            cum.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [{"基金代码": "000001", "净值日期": "2024-01-01", "复权净值": 1.0}]
            ).to_csv(adjusted / "000001.csv", index=False, encoding="utf-8-sig")

            result = compare_adjusted_nav_and_cum_return(base_dir=base)

            summary = pd.read_csv(result["summary_csv"], dtype={"基金代码": str})
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary.iloc[0]["基金代码"], "000001")
            self.assertEqual(summary.iloc[0]["数据是否缺失"], "是")
            self.assertEqual(int(summary.iloc[0]["参与比对收益率的天数"]), 0)

            error_lines = Path(result["error_jsonl"]).read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(error_lines), 1)
            rec = json.loads(error_lines[0])
            self.assertEqual(rec["code"], "000001")
            self.assertIn("fund file missing", rec["error"])

    def test_no_common_date_logged(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            adjusted = base / "fund_adjusted_nav_by_code"
            cum = base / "fund_cum_return_by_code"
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
                    {"基金代码": "000001", "日期": "2024-01-03", "累计收益率": 0.1},
                    {"基金代码": "000001", "日期": "2024-01-04", "累计收益率": 0.2},
                ]
            ).to_csv(cum / "000001.csv", index=False, encoding="utf-8-sig")

            result = compare_adjusted_nav_and_cum_return(base_dir=base)

            summary = pd.read_csv(result["summary_csv"], dtype={"基金代码": str})
            self.assertEqual(summary.iloc[0]["数据是否缺失"], "是")

            error_lines = Path(result["error_jsonl"]).read_text(encoding="utf-8").strip().splitlines()
            rec = json.loads(error_lines[0])
            self.assertIn("no common date found", rec["error"])

    def test_detail_and_summary_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            adjusted = base / "fund_adjusted_nav_by_code"
            cum = base / "fund_cum_return_by_code"
            adjusted.mkdir(parents=True, exist_ok=True)
            cum.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {"基金代码": "000001", "净值日期": "2024-01-01", "复权净值": 1.00},
                    {"基金代码": "000001", "净值日期": "2024-01-02", "复权净值": 1.06},
                    {"基金代码": "000001", "净值日期": "2024-01-03", "复权净值": 1.05},
                    {"基金代码": "000001", "净值日期": "2024-01-04", "复权净值": 1.08},
                ]
            ).to_csv(adjusted / "000001.csv", index=False, encoding="utf-8-sig")

            pd.DataFrame(
                [
                    {"基金代码": "000001", "日期": "2024-01-01", "累计收益率": 0.00},
                    {"基金代码": "000001", "日期": "2024-01-03", "累计收益率": 1.05},
                    {"基金代码": "000001", "日期": "2024-01-04", "累计收益率": 1.08},
                ]
            ).to_csv(cum / "000001.csv", index=False, encoding="utf-8-sig")

            result = compare_adjusted_nav_and_cum_return(base_dir=base)

            detail = pd.read_csv(Path(result["detail_dir"]) / "000001.csv")
            self.assertEqual(len(detail), 2)
            self.assertEqual(detail.iloc[0]["偏差类型"], "10%以上")
            self.assertEqual(detail.iloc[1]["偏差类型"], "10%以上")

            summary = pd.read_csv(result["summary_csv"], dtype={"基金代码": str})
            row = summary.iloc[0]
            self.assertEqual(row["数据是否缺失"], "否")
            self.assertEqual(int(row["参与比对收益率的天数"]), 2)
            self.assertEqual(int(row["因日期数据缺失跳过的天数"]), 1)
            self.assertEqual(row["<1%偏差占比"], "0.00%")
            self.assertEqual(row["10%以上偏差占比"], "100.00%")


if __name__ == "__main__":
    unittest.main()
