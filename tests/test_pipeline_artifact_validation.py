from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from validators.validate_pipeline_artifacts import validate_stage


class PipelineArtifactValidationTest(unittest.TestCase):
    def test_stage_validation_ok_on_baseline_purchase(self) -> None:
        purchase_csv = Path(__file__).resolve().parent / "baseline" / "mini_case" / "input" / "fund_etl" / "fund_purchase.csv"
        errors = validate_stage("fund_etl_step2_input", {"purchase_csv": purchase_csv})
        self.assertEqual(errors, [])

    def test_stage_validation_ok_on_baseline_effective_purchase(self) -> None:
        effective_csv = Path(__file__).resolve().parent / "baseline" / "mini_case" / "input" / "fund_etl" / "fund_purchase_effective.csv"
        if not effective_csv.exists():
            effective_csv = Path(__file__).resolve().parent / "baseline" / "mini_case" / "input" / "fund_etl" / "fund_purchase.csv"
        errors = validate_stage("fund_etl_step2_input_effective", {"purchase_csv": effective_csv})
        self.assertEqual(errors, [])

    def test_stage_validation_fails_on_missing_required_column(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fund_purchase.csv"
            pd.DataFrame([{"基金简称": "A"}]).to_csv(path, index=False, encoding="utf-8-sig")
            errors = validate_stage("fund_etl_step2_input", {"purchase_csv": path})
            self.assertTrue(errors)
            self.assertTrue(any("missing required columns" in e for e in errors))

    def test_stage_validation_fails_on_duplicate_code(self) -> None:
        src = Path(__file__).resolve().parent / "baseline" / "mini_case" / "input" / "fund_etl" / "fund_purchase.csv"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fund_purchase.csv"
            shutil.copy2(src, path)
            df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
            df = pd.concat([df, df.head(1)], ignore_index=True)
            df.to_csv(path, index=False, encoding="utf-8-sig")
            errors = validate_stage("fund_etl_step2_input", {"purchase_csv": path})
            self.assertTrue(any("unique key violated" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
