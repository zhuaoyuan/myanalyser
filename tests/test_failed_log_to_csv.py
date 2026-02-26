from __future__ import annotations

import tempfile
import unittest
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from failed_log_to_csv import jsonl_to_csv


class FailedLogToCsvTest(unittest.TestCase):
    def test_jsonl_to_csv(self) -> None:
        content = "\n".join(
            [
                '{"ts":"2026-02-21T01:55:54","stage":"step3_nav","code":"000009","error":"err1"}',
                '{"ts":"2026-02-21T01:55:58","stage":"step3_nav","code":"000010","error":"err2","extra":"x"}',
            ]
        )

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            input_path = base / "failed_nav.jsonl"
            output_path = base / "failed_nav.csv"
            input_path.write_text(content, encoding="utf-8")

            rows = jsonl_to_csv(input_path=input_path, output_path=output_path)

            self.assertEqual(rows, 2)
            self.assertTrue(output_path.exists())
            with output_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, ["ts", "stage", "code", "error", "extra"])
                rows = list(reader)

            self.assertEqual(rows[0]["code"], "000009")
            self.assertEqual(rows[0]["extra"], "")
            self.assertEqual(rows[1]["extra"], "x")


if __name__ == "__main__":
    unittest.main()
