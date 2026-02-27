from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline_scoreboard import _resolve_clickhouse_scope_tables, _resolve_clickhouse_write_config


class ClickHouseWriteProfileTest(unittest.TestCase):
    def test_auto_small_dataset_uses_fast_profile(self) -> None:
        cfg = _resolve_clickhouse_write_config(
            profile="auto",
            target_fund_count=20,
            nav_row_count=50_000,
            period_row_count=40_000,
            small_data_threshold_funds=200,
        )
        self.assertEqual(cfg.profile, "fast")
        self.assertEqual(cfg.wait_mutation_timeout_sec, 0)
        self.assertEqual(cfg.nav_sleep_between_chunks_sec, 0.0)
        self.assertEqual(cfg.insert_workers, 4)

    def test_auto_large_dataset_falls_back_to_safe_profile(self) -> None:
        cfg = _resolve_clickhouse_write_config(
            profile="auto",
            target_fund_count=500,
            nav_row_count=1_000_000,
            period_row_count=900_000,
            small_data_threshold_funds=200,
        )
        self.assertEqual(cfg.profile, "safe")
        self.assertEqual(cfg.wait_mutation_timeout_sec, 120)
        self.assertEqual(cfg.nav_chunk_rows, 5_000)

    def test_safe_profile_keeps_conservative_defaults(self) -> None:
        cfg = _resolve_clickhouse_write_config(
            profile="safe",
            target_fund_count=10,
            nav_row_count=1,
            period_row_count=1,
            small_data_threshold_funds=200,
        )
        self.assertEqual(cfg.profile, "safe")
        self.assertEqual(cfg.max_partition_groups_per_insert, 1)
        self.assertEqual(cfg.insert_workers, 1)

    def test_verify_minimal_scope_only_keeps_nav_and_scoreboard(self) -> None:
        tables = _resolve_clickhouse_scope_tables("verify_minimal")
        self.assertEqual(tables, ["fact_fund_nav_daily", "fact_fund_scoreboard_snapshot"])


if __name__ == "__main__":
    unittest.main()
