from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from filter_funds_for_next_step import filter_funds_for_next_step


class FilterFundsForNextStepTest(unittest.TestCase):
    def test_apply_all_filter_rules(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "fund_etl"
            base.mkdir(parents=True, exist_ok=True)
            compare_details = Path(d) / "fund_return_compare" / "details"
            compare_details.mkdir(parents=True, exist_ok=True)
            integrity_details = Path(d) / "trade_day_integrity_reports" / "details_2025-01-01_2025-12-31"
            integrity_details.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                {"基金代码": ["000001", "000002", "000003", "000004", "000005", "000006", "000007"]}
            ).to_csv(base / "fund_purchase.csv", index=False, encoding="utf-8-sig")

            pd.DataFrame(
                {"基金代码": ["000001", "000003", "000004", "000005", "000006", "000007"]}
            ).to_csv(base / "fund_overview.csv", index=False, encoding="utf-8-sig")

            nav_dir = base / "fund_nav_by_code"
            adjusted_dir = base / "fund_adjusted_nav_by_code"
            nav_dir.mkdir(parents=True, exist_ok=True)
            adjusted_dir.mkdir(parents=True, exist_ok=True)

            for code in ["000001", "000002", "000004", "000005", "000006", "000007"]:
                pd.DataFrame({"基金代码": [code], "净值日期": ["2025-01-02"], "单位净值": [1.0]}).to_csv(
                    nav_dir / f"{code}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            for code in ["000001", "000002", "000003", "000005", "000006", "000007"]:
                pd.DataFrame({"基金代码": [code], "净值日期": ["2025-01-02"], "复权净值": [1.0]}).to_csv(
                    adjusted_dir / f"{code}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            pd.DataFrame(
                {
                    "期初日期": ["2025-01-02"],
                    "本地远程收益率偏差": ["0.0100"],
                }
            ).to_csv(compare_details / "000001.csv", index=False, encoding="utf-8-sig")

            pd.DataFrame(
                {
                    "期初日期": ["2022-12-30"],
                    "本地远程收益率偏差": ["0.0100"],
                }
            ).to_csv(compare_details / "000006.csv", index=False, encoding="utf-8-sig")

            pd.DataFrame(
                {
                    "期初日期": ["2025-01-02"],
                    "本地远程收益率偏差": ["0.0300"],
                }
            ).to_csv(compare_details / "000007.csv", index=False, encoding="utf-8-sig")

            pd.DataFrame({"交易日日期": ["2025-01-02"], "该日期数据是否存在": ["是"]}).to_csv(
                integrity_details / "000001_2025-01-01_2025-12-31.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame({"交易日日期": ["2025-01-02"], "该日期数据是否存在": ["否"]}).to_csv(
                integrity_details / "000007_2025-01-01_2025-12-31.csv",
                index=False,
                encoding="utf-8-sig",
            )

            out = filter_funds_for_next_step(
                purchase_csv=base / "fund_purchase.csv",
                overview_csv=base / "fund_overview.csv",
                nav_dir=nav_dir,
                adjusted_nav_dir=adjusted_dir,
                compare_details_dir=compare_details,
                integrity_details_dir=integrity_details,
                start_date="2023-01-01",
                max_abs_deviation=0.02,
            )

            self.assertEqual(list(out["基金编码"]), ["000001", "000002", "000003", "000004", "000005", "000006", "000007"])
            by_code = {row["基金编码"]: row for row in out.to_dict("records")}

            self.assertEqual(by_code["000001"]["是否过滤"], "否")
            self.assertEqual(by_code["000001"]["过滤原因"], "")

            self.assertEqual(by_code["000002"]["是否过滤"], "是")
            self.assertIn("规则1", by_code["000002"]["过滤原因"])

            self.assertEqual(by_code["000003"]["是否过滤"], "是")
            self.assertIn("规则2", by_code["000003"]["过滤原因"])

            self.assertEqual(by_code["000004"]["是否过滤"], "是")
            self.assertIn("规则3", by_code["000004"]["过滤原因"])

            self.assertEqual(by_code["000005"]["是否过滤"], "是")
            self.assertIn("规则4", by_code["000005"]["过滤原因"])

            self.assertEqual(by_code["000006"]["是否过滤"], "是")
            self.assertIn("规则4", by_code["000006"]["过滤原因"])

            self.assertEqual(by_code["000007"]["是否过滤"], "是")
            self.assertIn("规则4", by_code["000007"]["过滤原因"])
            self.assertIn("规则5", by_code["000007"]["过滤原因"])


if __name__ == "__main__":
    unittest.main()
