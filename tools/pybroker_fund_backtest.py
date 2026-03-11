#!/usr/bin/env python3
"""PyBroker 基金回测（策略包模式）。

示例：
  python tools/pybroker_fund_backtest.py \
    --nav-dir finance-runs/run_20260310_191534/data \
    --strategy low_risk_debt \
    --start-date 2023-01-01 --end-date 2025-12-31 \
    --rebalance 20 --top-n 3 \
    --output-dir myanalyser/output/pybroker_backtest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_MYANALYSER_ROOT = _SCRIPT_DIR.parent
_SRC = _MYANALYSER_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from project_paths import project_root

from backtest import load_fund_nav_data, run_backtest
from backtest.engine import write_reports
from backtest.strategies.registry import get_strategy_bundle, list_strategy_names

_DEFAULT_NAV_DIR = (
    project_root().parent
    / "finance-runs"
    / "run_20260310_191534"
    / "data"
    / "versions"
    / "20260310_191534"
    / "fund_etl"
    / "fund_adjusted_nav_by_code"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="PyBroker 基金回测（策略包模式）")
    parser.add_argument(
        "--nav-dir",
        type=Path,
        default=_DEFAULT_NAV_DIR,
        help=f"复权净值目录或 run data 目录，默认 {_DEFAULT_NAV_DIR}",
    )
    parser.add_argument(
        "--strategy",
        default="low_risk_debt",
        help=f"策略包名称，可选: {', '.join(list_strategy_names())}",
    )
    parser.add_argument("--max-funds", type=int, default=200, help="最多加载基金数量")
    parser.add_argument("--start-date", default="2023-01-01", help="回测起始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", default="2025-12-31", help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--rebalance", type=int, default=20, help="调仓周期（交易日数）")
    parser.add_argument("--top-n", type=int, default=3, help="持仓基金数量")
    parser.add_argument("--warmup", type=int, default=252, help="策略预热 bar 数")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_MYANALYSER_ROOT / "output" / "pybroker_backtest",
        help="输出目录",
    )
    args = parser.parse_args()

    print(f"[pybroker_backtest] 加载数据: {args.nav_dir}")
    data = load_fund_nav_data(
        args.nav_dir,
        max_funds=args.max_funds,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if data.trading_dates:
        print(
            f"[pybroker_backtest] 基金数: {len(data.by_symbol)}, 日期: "
            f"{data.trading_dates[0].date()} ~ {data.trading_dates[-1].date()}",
        )

    bundle = get_strategy_bundle(args.strategy)
    print(f"[pybroker_backtest] 策略包: {bundle.name}")

    print("[pybroker_backtest] 运行回测...")
    backtest_result = run_backtest(
        data,
        bundle,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        rebalance_period=args.rebalance,
        warmup=args.warmup,
    )

    reports = write_reports(args.output_dir, backtest_result, data)
    print(f"[pybroker_backtest] 汇总报告: {reports['summary']}")
    print(f"[pybroker_backtest] 明细报告: {reports['detail']}")


if __name__ == "__main__":
    main()
