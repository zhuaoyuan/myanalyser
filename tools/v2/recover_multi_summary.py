#!/usr/bin/env python3
"""从已有 T 日子目录的 summary.csv 重建 multi_summary.csv 和 multi_summary_agg.csv。

用于补救：multi_t_backtest 在写完所有 T 日后、汇总前被中断，导致只有日期子目录而无 multi_summary 的情形。

用法:
  python myanalyser/tools/v2/recover_multi_summary.py --output-root myanalyser/artifacts/backtest_multi/RUN_ID/RULESET_VERSION
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_MYANALYSER_ROOT = _SCRIPT_DIR.parent.parent
_SRC = _MYANALYSER_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from multi_t_backtest import _extract_metrics, _write_multi_summary_agg

logger = logging.getLogger(__name__)


def recover(output_root: Path) -> None:
    output_root = Path(output_root).resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"output_root 不存在: {output_root}")

    # 扫描所有日期子目录（格式 YYYY-MM-DD）
    date_dirs = sorted(
        d for d in output_root.iterdir()
        if d.is_dir() and len(d.name) == 10 and d.name[4] == "-" and d.name[7] == "-"
    )
    if not date_dirs:
        raise FileNotFoundError(f"未找到日期子目录: {output_root}")

    summary_rows = []
    for d in date_dirs:
        as_of_str = d.name
        summary_csv = d / "summary.csv"
        if not summary_csv.exists():
            logger.warning("[%s] 缺少 summary.csv，跳过", as_of_str)
            continue
        metrics = _extract_metrics(summary_csv)
        detail_csv = d / "period_detail.csv"
        report_md = d / "backtest_report.md"
        summary_rows.append({
            "as_of_date": as_of_str,
            "filter_start": "",
            "filter_end": "",
            "backtest_start": as_of_str,
            "backtest_end": "",
            "allowed_funds": "",
            "compare_summary": "",
            "integrity_summary": "",
            "eligible_csv": "",
            "filter_csv": "",
            "scoreboard_csv": "",
            "summary_csv": str(summary_csv),
            "detail_csv": str(detail_csv),
            "report_md": str(report_md),
            **metrics,
        })

    if not summary_rows:
        raise ValueError("无有效 T 日数据可恢复")

    summary_df = pd.DataFrame(summary_rows)
    summary_out = output_root / "multi_summary.csv"
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")
    logger.info("[recover] multi_summary -> %s (%d rows)", summary_out, len(summary_df))

    _write_multi_summary_agg(summary_df, output_root)
    logger.info("[recover] multi_summary_agg -> %s", output_root / "multi_summary_agg.csv")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="从 T 日子目录重建 multi_summary 与 multi_summary_agg")
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    recover(args.output_root)
    print("OK")


if __name__ == "__main__":
    main()
