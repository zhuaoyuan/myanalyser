#!/usr/bin/env python3
"""批量运行比较基准组合回测：遍历比较基准.md 中 8 种组合，生成各自的回测产物。

用法:
  python myanalyser/tools/v2/run_benchmark_portfolios.py \
    --fund-etl-dir myanalyser/data/versions/20260315_123456_full_run_v2/fund_etl

  # 自定义日期 / 再平衡间隔
  python myanalyser/tools/v2/run_benchmark_portfolios.py \
    --fund-etl-dir myanalyser/data/versions/RUN_ID/fund_etl \
    --start-date 2020-01-02 --end-date 2024-12-31 \
    --rebalance-interval 121

产物目录: artifacts/backtest_base/{run_id}/
"""
from __future__ import annotations

import argparse
import datetime
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_MYANALYSER_ROOT = _SCRIPT_DIR.parent.parent

logger = logging.getLogger(__name__)

# 组合定义来源: docs/参考/比较基准.md（汇总一览表）。
# 此处硬编码以保证可靠性；若 Markdown 有变更，需同步更新本列表。
BENCHMARKS: list[tuple[str, str]] = [
    ("保守型_A", "161119:1.00"),
    ("保守型_B", "161119:0.90,510050:0.10"),
    ("稳健型_A", "161119:0.70,510300:0.30"),
    ("稳健型_B", "161119:0.75,510300:0.15,510500:0.10"),
    ("均衡型_A", "161119:0.50,510300:0.50"),
    ("均衡型_B", "161119:0.45,510300:0.35,510500:0.20"),
    ("进攻型_A", "161119:0.20,510300:0.80"),
    ("进攻型_B", "161119:0.15,510300:0.50,510500:0.20,159915:0.15"),
]

DEFAULT_START = "2015-02-27"
DEFAULT_END = datetime.date.today().isoformat()
DEFAULT_REBALANCE = 243


def _derive_run_id(fund_etl_dir: Path) -> str:
    """从 fund_etl_dir 路径推导 run_id。"""
    if fund_etl_dir.name == "fund_etl":
        return fund_etl_dir.parent.name
    return fund_etl_dir.name


def _collect_summary(output_root: Path) -> pd.DataFrame | None:
    """汇总各组合的 summary.csv 到一张表。"""
    rows: list[dict[str, str]] = []
    for name, _ in BENCHMARKS:
        summary_csv = output_root / name / "summary.csv"
        if not summary_csv.exists():
            continue
        df = pd.read_csv(summary_csv, dtype=str, encoding="utf-8-sig")
        metrics: dict[str, str] = {"组合名称": name}
        for _, row in df.iterrows():
            sec = str(row.get("section", "")).strip()
            nm = str(row.get("name", "")).strip()
            val = str(row.get("value", "")).strip()
            if sec in ("metrics_holding", "metrics_pybroker") and nm:
                metrics[nm] = val
        rows.append(metrics)
    return pd.DataFrame(rows) if rows else None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="批量运行比较基准组合回测（比较基准.md 中 8 种组合）",
    )
    parser.add_argument("--fund-etl-dir", required=True, type=Path,
                        help="fund_etl 数据目录（强制传参）")
    parser.add_argument("--start-date", default=DEFAULT_START,
                        help=f"起始日期（默认 {DEFAULT_START}）")
    parser.add_argument("--end-date", default=DEFAULT_END,
                        help=f"结束日期（默认 {DEFAULT_END}）")
    parser.add_argument("--rebalance-interval", type=int,
                        default=DEFAULT_REBALANCE,
                        help=f"再平衡间隔交易日数（默认 {DEFAULT_REBALANCE}）")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="产物根目录（默认 artifacts/backtest_base/{run_id}）")
    parser.add_argument("--integrity-threshold", type=float, default=0.95)
    parser.add_argument("--compare-threshold", type=float, default=0.80)
    args = parser.parse_args()

    fund_etl_dir = args.fund_etl_dir.resolve()
    if not fund_etl_dir.is_dir():
        print(f"[ERROR] fund_etl_dir 不存在: {fund_etl_dir}")
        sys.exit(1)

    run_id = _derive_run_id(fund_etl_dir)
    output_root = (
        args.output_root
        or _MYANALYSER_ROOT / "artifacts" / "backtest_base" / run_id
    )
    output_root.mkdir(parents=True, exist_ok=True)

    script_path = _SCRIPT_DIR / "benchmark_portfolio_backtest.py"
    python_bin = sys.executable

    print(f"[batch] fund_etl_dir: {fund_etl_dir}")
    print(f"[batch] run_id:       {run_id}")
    print(f"[batch] output_root:  {output_root}")
    print(f"[batch] 日期:         {args.start_date} ~ {args.end_date}")
    print(f"[batch] 再平衡:       每 {args.rebalance_interval} 交易日")
    print(f"[batch] 组合数量:     {len(BENCHMARKS)}")

    results: list[dict[str, str | int]] = []
    for name, portfolio in BENCHMARKS:
        output_dir = output_root / name
        print(f"\n{'=' * 60}")
        print(f"[batch] 运行: {name} ({portfolio})")
        print(f"{'=' * 60}")

        cmd = [
            python_bin, str(script_path),
            "--fund-etl-dir", str(fund_etl_dir),
            "--start-date", args.start_date,
            "--end-date", args.end_date,
            "--rebalance-interval", str(args.rebalance_interval),
            "--portfolio", portfolio,
            "--output-dir", str(output_dir),
            "--integrity-threshold", str(args.integrity_threshold),
            "--compare-threshold", str(args.compare_threshold),
        ]
        proc = subprocess.run(cmd)
        status = "成功" if proc.returncode == 0 else "失败"
        results.append({"组合": name, "状态": status, "返回码": proc.returncode})

        if proc.returncode != 0:
            print(f"[batch] {name} 失败，返回码 {proc.returncode}")

    # 汇总
    agg = _collect_summary(output_root)
    if agg is not None:
        agg_path = output_root / "benchmark_summary.csv"
        agg.to_csv(agg_path, index=False, encoding="utf-8-sig")
        print(f"\n[batch] 汇总 -> {agg_path}")

    # 结果表
    print(f"\n{'=' * 60}")
    print("[batch] 批量运行结果:")
    for r in results:
        mark = "✓" if r["返回码"] == 0 else "✗"
        print(f"  {mark} {r['组合']}: {r['状态']}")

    failed = [r for r in results if r["返回码"] != 0]
    if failed:
        print(f"\n[batch] {len(failed)}/{len(results)} 个组合运行失败")
        sys.exit(1)
    print(f"\n[batch] 全部 {len(results)} 个组合运行成功")
    print(f"[batch] 产物根目录: {output_root}")


if __name__ == "__main__":
    main()
