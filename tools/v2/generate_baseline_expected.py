#!/usr/bin/env python3
"""Generate mini_case_v2 expected outputs. Run once to bootstrap baseline.

PROJECT = myanalyser 根目录，脚本位于 myanalyser/tools/v2/
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent  # myanalyser
sys.path.insert(0, str(PROJECT / "src"))
os.chdir(PROJECT)
input_root = PROJECT / "tests" / "baseline" / "mini_case_v2" / "input"
output_root = PROJECT / "tests" / "baseline" / "mini_case_v2" / "_run_output"
expected_dir = PROJECT / "tests" / "baseline" / "mini_case_v2" / "expected" / "default"
fund_etl = input_root / "fund_etl"
trade_dates = PROJECT / "data" / "common" / "trade_dates.csv"

if expected_dir.exists() and not os.getenv("OVERWRITE_BASELINE"):
    sys.exit("expected 已存在，设置 OVERWRITE_BASELINE=1 覆盖")

shutil.rmtree(output_root, ignore_errors=True)
output_root.mkdir(parents=True)
artifacts = output_root / "artifacts"
artifacts.mkdir(parents=True)
# Step5: adjusted_nav
subprocess.run([sys.executable, "src/adjusted_nav_tool.py",
    "--nav-dir", str(fund_etl / "fund_nav_by_code"),
    "--bonus-dir", str(fund_etl / "fund_bonus_by_code"),
    "--split-dir", str(fund_etl / "fund_split_by_code"),
    "--output-dir", str(fund_etl / "fund_adjusted_nav_by_code"),
    "--allow-missing-event-until", "2020-12-31",
], check=True, capture_output=True)
# Create cum_return from adjusted_nav for step7 compare (共享逻辑见 _baseline_helpers)
from _baseline_helpers import build_cum_return_from_adjusted_nav

build_cum_return_from_adjusted_nav(
    fund_etl / "fund_adjusted_nav_by_code",
    fund_etl / "fund_cum_return_by_code",
)
# Copy fund_etl to work dir for steps 6-10 (steps modify in place or use output)
work_etl = output_root / "fund_etl"
shutil.copytree(fund_etl, work_etl)
# Step6: integrity
subprocess.run([sys.executable, "src/check_trade_day_data_integrity.py",
    "--base-dir", str(work_etl),
    "--start-date", "2025-01-01", "--end-date", "2025-12-31",
    "--trade-dates-csv", str(trade_dates),
    "--output-dir", str(artifacts / "trade_day_integrity_reports"),
], check=True, capture_output=True)
# Step7: compare
subprocess.run([sys.executable, "-m", "v2.compare.compare_adjusted_nav_and_cum_return_window",
    "--base-dir", str(work_etl),
    "--start-date", "2025-01-01", "--end-date", "2025-12-31",
    "--output-dir", str(artifacts / "fund_return_compare"),
], check=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(PROJECT / "src")})
# Step8: filter + build_filtered
purchase_csv = work_etl / "fund_purchase_effective.csv" if (work_etl / "fund_purchase_effective.csv").exists() else work_etl / "fund_purchase.csv"
filter_csv = artifacts / "filtered_fund_candidates.csv"
subprocess.run([sys.executable, str(PROJECT / "src" / "v2" / "filters" / "filter_funds_for_next_step.py"),
    "--base-dir", str(work_etl), "--purchase-csv", str(purchase_csv),
    "--compare-details-dir", str(artifacts / "fund_return_compare" / "details"),
    "--integrity-details-dir", str(artifacts / "trade_day_integrity_reports" / "details_2025-01-01_2025-12-31"),
    "--start-date", "2023-01-01", "--end-date", "2025-12-31", "--max-abs-deviation", "0.02",
    "--output-csv", str(filter_csv),
], check=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(PROJECT / "src")})
filtered_purchase = artifacts / "fund_purchase_for_step10_filtered.csv"
subprocess.run([sys.executable, "src/transforms/build_filtered_purchase_csv.py",
    "--purchase-csv", str(purchase_csv), "--filter-csv", str(filter_csv),
    "--output-csv", str(filtered_purchase),
], check=True, capture_output=True)
# Step9: scoreboard
nav_dir = work_etl / "fund_adjusted_nav_by_code"
max_date = None
for p in nav_dir.glob("*.csv"):
    try:
        df = pd.read_csv(p, usecols=["净值日期"], dtype=str, encoding="utf-8-sig")
    except (KeyError, ValueError):
        continue
    ds = pd.to_datetime(df["净值日期"], errors="coerce").dropna()
    if not ds.empty and (max_date is None or ds.max() > max_date):
        max_date = ds.max()
as_of = max_date.strftime("%Y-%m-%d") if max_date is not None else "2025-12-31"
scoreboard_dir = artifacts / "scoreboard"
scoreboard_dir.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "src/pipeline_scoreboard.py",
    "--purchase-csv", str(filtered_purchase),
    "--overview-csv", str(work_etl / "fund_overview.csv"),
    "--personnel-dir", str(work_etl / "fund_personnel_by_code"),
    "--nav-dir", str(nav_dir),
    "--output-dir", str(scoreboard_dir),
    "--data-version", "baseline_v2",
    "--as-of-date", as_of,
    "--stale-max-days", "3650",
    "--skip-sinks", "--formal-only",
], check=True, capture_output=True)
# Step10: filter_score + recalc
filter_score_dir = artifacts / "filter_score"
subprocess.run(["bash", "tools/run_filter_and_score.sh",
    "-i", str(scoreboard_dir / "fund_scoreboard_baseline_v2.csv"),
    "-w", str(filter_score_dir),
    "-f", "src/filter_score/filters/most_stable.py",
    "-s", "src/filter_score/scores/low_risk_debt.py",
], check=True, capture_output=True, cwd=str(PROJECT))
recheck_dir = artifacts / "scoreboard_recheck"
subprocess.run([sys.executable, "src/verify_scoreboard_recalc.py",
    "--scoreboard-csv", str(scoreboard_dir / "fund_scoreboard_baseline_v2.csv"),
    "--fund-etl-dir", str(work_etl),
    "--output-dir", str(recheck_dir),
    "--max-input-rows", "50",
], check=True, capture_output=True)
# Copy to expected
shutil.rmtree(expected_dir, ignore_errors=True)
expected_dir.mkdir(parents=True)
shutil.copytree(work_etl / "fund_adjusted_nav_by_code", expected_dir / "fund_adjusted_nav_by_code")
shutil.copytree(artifacts / "trade_day_integrity_reports", expected_dir / "trade_day_integrity_reports")
shutil.copytree(artifacts / "fund_return_compare", expected_dir / "fund_return_compare")
shutil.copy(filter_csv, expected_dir / "filtered_fund_candidates.csv")
shutil.copy(filtered_purchase, expected_dir / "fund_purchase_for_step10_filtered.csv")
shutil.copytree(scoreboard_dir, expected_dir / "scoreboard")
shutil.copy(filter_score_dir / "filter_result.csv", expected_dir / "filter_result.csv")
shutil.copy(filter_score_dir / "scored_result.csv", expected_dir / "scored_result.csv")
shutil.copytree(recheck_dir, expected_dir / "scoreboard_recheck")
# Also copy scoreboard CSV with fixed name for regression test
shutil.copy(scoreboard_dir / "fund_scoreboard_baseline_v2.csv", expected_dir / "fund_scoreboard_baseline_v2.csv")
print("All steps done. Expected saved to", expected_dir)
