"""verify step10 回测适配层：从 ClickHouse 取选基 + 本地净值 + 新 backtest 引擎。

替代 backtest_portfolio.py，供 verify.sh step10 调用。产出与新 backtest 一致
（period_detail.csv、backtest_report.md 等）。
"""

from __future__ import annotations

import argparse
import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd

from project_paths import project_root

from backtest import load_fund_nav_data, run_backtest
from backtest.engine import BacktestConfig, write_reports
from backtest.strategies.verify_e2e_top5 import build_bundle_verify_e2e


def _quote_sql(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _run_clickhouse_query(query: str, container_name: str) -> pd.DataFrame:
    cmd = [
        "docker",
        "exec",
        container_name,
        "clickhouse-client",
        "--format",
        "CSVWithNames",
        "--query",
        query,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    text = result.stdout.strip()
    if not text:
        return pd.DataFrame()
    return pd.read_csv(StringIO(text))


def _build_status_filter(col_name: str, excluded_values: list[str]) -> str:
    if not excluded_values:
        return "1"
    values = ", ".join(_quote_sql(v) for v in excluded_values)
    return f"({col_name} NOT IN ({values}) OR {col_name} IS NULL)"


def _fetch_fund_selection(
    clickhouse_db: str,
    clickhouse_container: str,
    data_version: str,
    selection_where: str,
    selection_order_by: str,
    selection_limit: int,
    exclude_subscribe_status: list[str],
    exclude_redeem_status: list[str],
) -> pd.DataFrame:
    subscribe_filter = _build_status_filter("subscribe_status", exclude_subscribe_status)
    redeem_filter = _build_status_filter("redeem_status", exclude_redeem_status)
    query = (
        "SELECT fund_code "
        f"FROM {clickhouse_db}.fact_fund_scoreboard_snapshot "
        f"WHERE data_version={_quote_sql(data_version)} "
        f"AND ({selection_where}) "
        f"AND {subscribe_filter} "
        f"AND {redeem_filter} "
        f"ORDER BY {selection_order_by} "
        f"LIMIT {int(selection_limit)}"
    )
    df = _run_clickhouse_query(query, clickhouse_container)
    if df.empty or "fund_code" not in df.columns:
        return pd.DataFrame(columns=["fund_code", "weight"])
    df["fund_code"] = df["fund_code"].astype(str).str.strip().str.zfill(6)
    df = df.drop_duplicates(subset=["fund_code"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["fund_code", "weight"])
    df["weight"] = 1.0 / float(len(df))
    return df[["fund_code", "weight"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="verify step10 回测适配层：ClickHouse 选基 + 本地净值 + 新 backtest"
    )
    root = Path(project_root())
    parser.add_argument("--start-date", required=True, help="回测起始 YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="回测结束 YYYY-MM-DD")
    parser.add_argument("--nav-dir", required=True, type=Path, help="复权净值目录（fund_adjusted_nav_by_code）")
    parser.add_argument("--output-dir", required=True, type=Path, help="输出目录")
    parser.add_argument(
        "--trade-dates-csv",
        default=str(root / "data" / "common" / "trade_dates.csv"),
        help="交易日历 CSV",
    )
    parser.add_argument("--rebalance", type=int, default=15, help="调仓周期（交易日数）")
    parser.add_argument("--top-n", type=int, default=5, help="持仓基金数量")
    parser.add_argument("--selection-rule-id", default="verify_e2e_top5")
    parser.add_argument("--selection-data-version", required=True, help="选基 data_version")
    parser.add_argument("--selection-where", default="1")
    parser.add_argument(
        "--selection-order-by",
        default="annual_return DESC, fund_code ASC",
    )
    parser.add_argument("--selection-limit", type=int, default=5)
    parser.add_argument("--exclude-subscribe-status", default="暂停申购,封闭期")
    parser.add_argument("--exclude-redeem-status", default="暂停赎回,封闭期")
    parser.add_argument("--clickhouse-db", default="fund_analysis")
    parser.add_argument("--clickhouse-container", default="fund_clickhouse")
    parser.add_argument("--warmup", type=int, default=243)
    parser.add_argument("--initial-cash", type=float, default=100_000)
    args = parser.parse_args()

    exclude_sub = [s.strip() for s in args.exclude_subscribe_status.split(",") if s.strip()]
    exclude_red = [s.strip() for s in args.exclude_redeem_status.split(",") if s.strip()]

    sel_df = _fetch_fund_selection(
        clickhouse_db=args.clickhouse_db,
        clickhouse_container=args.clickhouse_container,
        data_version=args.selection_data_version,
        selection_where=args.selection_where,
        selection_order_by=args.selection_order_by,
        selection_limit=args.selection_limit,
        exclude_subscribe_status=exclude_sub,
        exclude_redeem_status=exclude_red,
    )
    if sel_df.empty:
        raise SystemExit("从 ClickHouse 获取选基为空，请检查 scoreboard 是否已入库")

    allowed_codes = set(sel_df["fund_code"].astype(str).tolist())
    allowed_list = sel_df["fund_code"].astype(str).tolist()

    data = load_fund_nav_data(
        args.nav_dir,
        max_funds=500,
        start_date=args.start_date,
        end_date=args.end_date,
        allowed_codes=allowed_codes,
    )
    if not data.by_symbol:
        raise SystemExit("未加载到任何有效净值数据，请检查 nav-dir 与日期范围")

    bundle = build_bundle_verify_e2e(allowed_list)
    config = BacktestConfig(initial_cash=args.initial_cash)
    result = run_backtest(
        data,
        bundle,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=min(args.top_n, len(allowed_list)),
        rebalance_period=args.rebalance,
        warmup=args.warmup,
        config=config,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "strategy": "verify_e2e_top5",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "rebalance": args.rebalance,
        "top_n": args.top_n,
        "selection_rule_id": args.selection_rule_id,
        "selection_data_version": args.selection_data_version,
        "nav_dir": str(args.nav_dir.resolve()),
        "warmup": args.warmup,
        "initial_cash": args.initial_cash,
    }
    reports = write_reports(
        args.output_dir,
        result,
        data,
        run_config=run_config,
        initial_cash=config.initial_cash,
    )

    print(f"[backtest_verify_e2e] period_detail={reports['detail']}")
    print(f"[backtest_verify_e2e] backtest_report={reports['report_md']}")
