#!/usr/bin/env python3
"""PyBroker 基金回测（策略包模式）。

支持多过滤器链：通过环境变量 FUND_BACKTEST_FILTERS 指定（逗号分隔），如：
  FUND_BACKTEST_FILTERS=filtered_candidates,max_funds
  FILTERED_FUND_CANDIDATES_CSV=path/to/filtered_fund_candidates.csv
  FUND_BACKTEST_MAX_FUNDS=50

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

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_MYANALYSER_ROOT = _SCRIPT_DIR.parent
_SRC = _MYANALYSER_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from project_paths import project_root

from backtest import (
    apply_filter_chain,
    get_available_symbols,
    get_filter_chain,
    load_fund_nav_data,
    run_backtest,
)
from backtest.engine import BacktestConfig
from backtest.engine import write_reports
from backtest.strategies.registry import get_strategy_bundle, list_strategy_names

def _guess_latest_run_data_dir() -> Path | None:
    runs_dir = project_root().parent / "finance-runs"
    if not runs_dir.is_dir():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not candidates:
        return None
    latest = sorted(candidates, key=lambda p: p.name)[-1]
    return latest / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="PyBroker 基金回测（策略包模式）")
    default_nav_dir = _guess_latest_run_data_dir()
    parser.add_argument(
        "--nav-dir",
        type=Path,
        default=default_nav_dir,
        help="复权净值目录或 run data 目录，默认自动选择最新 run",
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
    parser.add_argument("--initial-cash", type=float, default=100_000, help="初始资金")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_MYANALYSER_ROOT / "output" / "pybroker_backtest",
        help="输出目录",
    )
    args = parser.parse_args()

    if args.nav_dir is None:
        raise SystemExit("未找到可用的默认数据目录，请显式传入 --nav-dir")

    allowed_codes: set[str] | None = None
    filters = get_filter_chain()
    if filters:
        candidates = get_available_symbols(args.nav_dir)
        allowed_codes = apply_filter_chain(candidates, filters)
        if not allowed_codes:
            raise SystemExit(
                "过滤器链将基金池缩减为空，请检查 FUND_BACKTEST_FILTERS 及各过滤器配置"
            )
        print(f"[pybroker_backtest] 过滤器链: {len(allowed_codes)} 只基金")

    print(f"[pybroker_backtest] 加载数据: {args.nav_dir}")
    data = load_fund_nav_data(
        args.nav_dir,
        max_funds=args.max_funds,
        start_date=args.start_date,
        end_date=args.end_date,
        allowed_codes=allowed_codes,
    )
    if data.trading_dates:
        print(
            f"[pybroker_backtest] 基金数: {len(data.by_symbol)}, 日期: "
            f"{data.trading_dates[0].date()} ~ {data.trading_dates[-1].date()}",
        )

    bundle = get_strategy_bundle(args.strategy)
    print(f"[pybroker_backtest] 策略包: {bundle.name}")

    print("[pybroker_backtest] 运行回测...")
    config = BacktestConfig(initial_cash=args.initial_cash)
    backtest_result = run_backtest(
        data,
        bundle,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        rebalance_period=args.rebalance,
        warmup=args.warmup,
        config=config,
    )

    run_config = {
        "strategy": args.strategy,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "rebalance": args.rebalance,
        "top_n": args.top_n,
        "warmup": args.warmup,
        "initial_cash": args.initial_cash,
    }
    reports = write_reports(
        args.output_dir,
        backtest_result,
        data,
        run_config=run_config,
        initial_cash=config.initial_cash,
    )

    # 控制台核心指标摘要
    summary_df = pd.read_csv(reports["summary"], encoding="utf-8-sig")
    metrics = summary_df[summary_df["section"] == "metrics"]
    if not metrics.empty:
        key_names = ["近1年年化收益率", "近3年年化收益率", "近1年最大回撤率", "近1年夏普比率", "近1年卡玛比率"]
        parts = []
        for _, row in metrics.iterrows():
            name = str(row.get("name", ""))
            if name in key_names:
                val = row.get("value")
                if val is not None and str(val) != "nan":
                    try:
                        fval = float(val)
                        parts.append(f"{name}: {fval:.2%}" if "率" in name or "收益" in name else f"{name}: {fval:.2f}")
                    except (TypeError, ValueError):
                        parts.append(f"{name}: {val}")
        if parts:
            print(f"[pybroker_backtest] {' | '.join(parts)}")

    print(f"[pybroker_backtest] 汇总: {reports['summary']}")
    print(f"[pybroker_backtest] 明细: {reports['detail']}")
    print(f"[pybroker_backtest] 净值曲线: {reports['equity_curve']}")
    print(f"[pybroker_backtest] 报告: {reports['report_md']}")
    if "curves_html" in reports:
        print(f"[pybroker_backtest] 收益曲线图: {reports['curves_html']}")


if __name__ == "__main__":
    main()
