from __future__ import annotations

import argparse
import bisect
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from project_paths import project_root


@dataclass
class Window:
    window_id: int
    d1_anchor_date: pd.Timestamp
    d2_last_hist_trade_date: pd.Timestamp
    d3_buy_date: pd.Timestamp
    d4_sell_date: pd.Timestamp
    selection_data_version: str | None = None
    status: str = "pending"
    skip_reason: str = ""


def _quote_sql(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _run_clickhouse_query(query: str, container_name: str) -> pd.DataFrame:
    cmd = ["docker", "exec", container_name, "clickhouse-client", "--format", "CSVWithNames", "--query", query]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    text = result.stdout.strip()
    if not text:
        return pd.DataFrame()
    return pd.read_csv(StringIO(text))


def _load_trade_days(trade_dates_csv: Path) -> list[pd.Timestamp]:
    trade_df = pd.read_csv(trade_dates_csv, dtype={"trade_date": str}, encoding="utf-8-sig")
    if "trade_date" not in trade_df.columns:
        raise ValueError(f"trade_dates.csv 缺少 trade_date 列: {trade_dates_csv}")
    days = (
        pd.to_datetime(trade_df["trade_date"], errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not days:
        raise ValueError(f"交易日历为空: {trade_dates_csv}")
    return days


def _build_windows(
    start_date: str,
    end_date: str,
    trade_days: list[pd.Timestamp],
    rebalance_interval_days: int,
    holding_period_days: int,
) -> list[Window]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date 不能晚于 end_date")

    windows: list[Window] = []
    anchor = start
    wid = 1
    while anchor <= end:
        idx2 = bisect.bisect_left(trade_days, anchor)
        if idx2 >= len(trade_days):
            break
        d2 = trade_days[idx2]
        if idx2 + 1 >= len(trade_days):
            break
        d3 = trade_days[idx2 + 1]
        d4_target = d3 + pd.Timedelta(days=holding_period_days)
        idx4 = bisect.bisect_left(trade_days, d4_target)
        if idx4 >= len(trade_days):
            break
        d4 = trade_days[idx4]
        windows.append(
            Window(
                window_id=wid,
                d1_anchor_date=anchor,
                d2_last_hist_trade_date=d2,
                d3_buy_date=d3,
                d4_sell_date=d4,
            )
        )
        wid += 1
        anchor = anchor + pd.Timedelta(days=rebalance_interval_days)
    return windows


def _fetch_selection_version_dates(clickhouse_db: str, clickhouse_container: str) -> pd.DataFrame:
    query = (
        f"SELECT data_version, toDate(max(as_of_date)) AS as_of_date "
        f"FROM {clickhouse_db}.fact_fund_scoreboard_snapshot "
        "GROUP BY data_version ORDER BY as_of_date, data_version"
    )
    df = _run_clickhouse_query(query, clickhouse_container)
    if df.empty:
        return pd.DataFrame(columns=["data_version", "as_of_date"])
    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["as_of_date"]).sort_values(["as_of_date", "data_version"]).reset_index(drop=True)
    return df


def _assign_selection_versions(windows: list[Window], version_dates: pd.DataFrame, fixed_data_version: str | None) -> None:
    if fixed_data_version:
        for w in windows:
            w.selection_data_version = fixed_data_version
        return
    if version_dates.empty:
        for w in windows:
            w.status = "skipped"
            w.skip_reason = "no_scoreboard_data_version"
        return
    as_of_list = version_dates["as_of_date"].tolist()
    version_list = version_dates["data_version"].astype(str).tolist()
    for w in windows:
        idx = bisect.bisect_right(as_of_list, w.d2_last_hist_trade_date) - 1
        if idx < 0:
            w.status = "skipped"
            w.skip_reason = "no_data_version_before_d2"
            continue
        w.selection_data_version = version_list[idx]


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


def _fetch_latest_nav_data_version(clickhouse_db: str, clickhouse_container: str) -> str:
    query = (
        f"SELECT data_version, max(as_of_date) AS as_of_date FROM {clickhouse_db}.fact_fund_nav_daily "
        "GROUP BY data_version ORDER BY as_of_date DESC, data_version DESC LIMIT 1"
    )
    df = _run_clickhouse_query(query, clickhouse_container)
    if df.empty:
        raise ValueError("fact_fund_nav_daily 中没有可用 data_version")
    return str(df.iloc[0]["data_version"])


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _fetch_nav_data(
    clickhouse_db: str,
    clickhouse_container: str,
    nav_data_version: str,
    fund_codes: list[str],
    nav_start: pd.Timestamp,
    nav_end: pd.Timestamp,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for codes in _chunks(sorted(set(fund_codes)), 500):
        in_list = ", ".join(_quote_sql(code) for code in codes)
        query = (
            "SELECT fund_code, nav_date, adjusted_nav "
            f"FROM {clickhouse_db}.fact_fund_nav_daily "
            f"WHERE data_version={_quote_sql(nav_data_version)} "
            f"AND fund_code IN ({in_list}) "
            f"AND nav_date >= {_quote_sql(nav_start.strftime('%Y-%m-%d'))} "
            f"AND nav_date <= {_quote_sql(nav_end.strftime('%Y-%m-%d'))} "
            "ORDER BY fund_code, nav_date"
        )
        part = _run_clickhouse_query(query, clickhouse_container)
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["fund_code", "nav_date", "adjusted_nav"])
    nav_df = pd.concat(frames, ignore_index=True)
    nav_df["fund_code"] = nav_df["fund_code"].astype(str).str.strip().str.zfill(6)
    nav_df["nav_date"] = pd.to_datetime(nav_df["nav_date"], errors="coerce").dt.normalize()
    nav_df["adjusted_nav"] = pd.to_numeric(nav_df["adjusted_nav"], errors="coerce")
    nav_df = nav_df.dropna(subset=["nav_date", "adjusted_nav"])
    return nav_df.sort_values(["fund_code", "nav_date"]).reset_index(drop=True)


def _max_drawdown(nav_series: pd.Series) -> float:
    if nav_series.empty:
        return float("nan")
    roll_max = nav_series.cummax()
    drawdown = 1.0 - nav_series / roll_max
    return float(drawdown.max())


def _build_window_dates_index(trade_days: list[pd.Timestamp], start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    l = bisect.bisect_left(trade_days, start)
    r = bisect.bisect_right(trade_days, end)
    return trade_days[l:r]


def _render_markdown_report(
    run_summary: pd.DataFrame,
    aggregate_summary: pd.DataFrame,
    window_detail: pd.DataFrame,
    output_dir: Path,
) -> str:
    run = run_summary.iloc[0].to_dict()
    agg = aggregate_summary.iloc[0].to_dict()
    lines = [
        "# 基金组合回测报告",
        "",
        "## 运行参数",
        f"- run_id: {run['run_id']}",
        f"- run_time: {run['run_time']}",
        f"- 回测区间: {run['start_date']} ~ {run['end_date']}",
        f"- 调仓间隔(天): {run['rebalance_interval_days']}",
        f"- 持有周期(自然日): {run['holding_period_days']}",
        f"- 选基规则ID: {run['selection_rule_id']}",
        f"- 选基数据版本模式: {run['selection_data_version_mode']}",
        f"- 净值数据版本: {run['nav_data_version']}",
        "",
        "## 汇总结果",
        f"- 总窗口数: {int(agg['total_windows'])}",
        f"- 有效窗口数: {int(agg['valid_windows'])}",
        f"- 跳过窗口数: {int(agg['skipped_windows'])}",
        f"- 胜率: {agg['win_rate']:.4f}",
        f"- 平均收益: {agg['mean_return']:.4f}",
        f"- 中位数收益: {agg['median_return']:.4f}",
        f"- P10/P50/P90: {agg['p10_return']:.4f} / {agg['p50_return']:.4f} / {agg['p90_return']:.4f}",
        f"- 最佳/最差窗口收益: {agg['best_window_return']:.4f} / {agg['worst_window_return']:.4f}",
        "",
        "## 输出文件",
        "- backtest_run_summary.csv",
        "- backtest_window_detail.csv",
        "- backtest_window_positions.csv",
        "- backtest_aggregate_summary.csv",
    ]

    top = window_detail[window_detail["status"] == "ok"].sort_values("portfolio_return", ascending=False).head(3)
    if not top.empty:
        lines.extend(["", "## Top 3 窗口", "|window_id|d3|d4|return|", "|---:|---|---|---:|"])
        for _, row in top.iterrows():
            lines.append(
                f"|{int(row['window_id'])}|{row['d3_buy_date']}|{row['d4_sell_date']}|{float(row['portfolio_return']):.4f}|"
            )

    report = "\n".join(lines) + "\n"
    (output_dir / "backtest_report.md").write_text(report, encoding="utf-8")
    return report


def run_backtest(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trade_dates_csv = Path(args.trade_dates_csv).resolve()

    trade_days = _load_trade_days(trade_dates_csv)
    windows = _build_windows(
        start_date=args.start_date,
        end_date=args.end_date,
        trade_days=trade_days,
        rebalance_interval_days=args.rebalance_interval_days,
        holding_period_days=args.holding_period_days,
    )
    if not windows:
        raise ValueError("未生成任何可回测窗口，请检查日期范围和交易日历")

    version_dates = _fetch_selection_version_dates(args.clickhouse_db, args.clickhouse_container)
    _assign_selection_versions(windows, version_dates, args.selection_data_version)

    selection_cache: dict[str, pd.DataFrame] = {}
    exclude_subscribe = [s.strip() for s in args.exclude_subscribe_status.split(",") if s.strip()]
    exclude_redeem = [s.strip() for s in args.exclude_redeem_status.split(",") if s.strip()]
    for w in windows:
        if w.status == "skipped":
            continue
        assert w.selection_data_version is not None
        if w.selection_data_version not in selection_cache:
            selection_cache[w.selection_data_version] = _fetch_fund_selection(
                clickhouse_db=args.clickhouse_db,
                clickhouse_container=args.clickhouse_container,
                data_version=w.selection_data_version,
                selection_where=args.selection_where,
                selection_order_by=args.selection_order_by,
                selection_limit=args.selection_limit,
                exclude_subscribe_status=exclude_subscribe,
                exclude_redeem_status=exclude_redeem,
            )
        if selection_cache[w.selection_data_version].empty:
            w.status = "skipped"
            w.skip_reason = "empty_selection"

    nav_data_version = args.nav_data_version or _fetch_latest_nav_data_version(args.clickhouse_db, args.clickhouse_container)

    active_windows = [w for w in windows if w.status != "skipped"]
    selected_funds: set[str] = set()
    for w in active_windows:
        assert w.selection_data_version is not None
        selected_funds.update(selection_cache[w.selection_data_version]["fund_code"].tolist())

    nav_df = pd.DataFrame(columns=["fund_code", "nav_date", "adjusted_nav"])
    if selected_funds and active_windows:
        nav_start = min(w.d3_buy_date for w in active_windows)
        nav_end = max(w.d4_sell_date for w in active_windows)
        nav_df = _fetch_nav_data(
            clickhouse_db=args.clickhouse_db,
            clickhouse_container=args.clickhouse_container,
            nav_data_version=nav_data_version,
            fund_codes=sorted(selected_funds),
            nav_start=nav_start,
            nav_end=nav_end,
        )

    nav_map: dict[str, pd.Series] = {
        code: grp.set_index("nav_date")["adjusted_nav"].sort_index() for code, grp in nav_df.groupby("fund_code")
    }

    window_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    for w in windows:
        if w.status == "skipped":
            window_rows.append(
                {
                    "window_id": w.window_id,
                    "d1_anchor_date": w.d1_anchor_date.strftime("%Y-%m-%d"),
                    "d2_last_hist_trade_date": w.d2_last_hist_trade_date.strftime("%Y-%m-%d"),
                    "d3_buy_date": w.d3_buy_date.strftime("%Y-%m-%d"),
                    "d4_sell_date": w.d4_sell_date.strftime("%Y-%m-%d"),
                    "calendar_holding_days": int((w.d4_sell_date - w.d3_buy_date).days),
                    "trading_holding_days": len(_build_window_dates_index(trade_days, w.d3_buy_date, w.d4_sell_date)),
                    "selection_data_version": w.selection_data_version,
                    "selected_fund_count": 0,
                    "weight_sum": 0.0,
                    "portfolio_return": np.nan,
                    "portfolio_nav_start": np.nan,
                    "portfolio_nav_end": np.nan,
                    "max_drawdown_in_window": np.nan,
                    "volatility_in_window": np.nan,
                    "status": "skipped",
                    "skip_reason": w.skip_reason,
                }
            )
            continue

        assert w.selection_data_version is not None
        selected = selection_cache[w.selection_data_version].copy()
        selected["buy_nav_at_d3"] = np.nan
        selected["sell_nav_at_d4"] = np.nan
        interval_dates = _build_window_dates_index(trade_days, w.d3_buy_date, w.d4_sell_date)
        if not interval_dates:
            w.status = "skipped"
            w.skip_reason = "no_trade_days_in_window"

        usable_rows: list[dict[str, object]] = []
        portfolio_nav_df = pd.DataFrame(index=interval_dates)
        for _, row in selected.iterrows():
            code = row["fund_code"]
            s = nav_map.get(code)
            if s is None:
                continue
            buy_nav = s.get(w.d3_buy_date, np.nan)
            sell_nav = s.get(w.d4_sell_date, np.nan)
            if pd.isna(buy_nav) or pd.isna(sell_nav) or float(buy_nav) <= 0:
                continue
            nav_path = s.reindex(interval_dates)
            if nav_path.isna().any():
                continue
            portfolio_nav_df[code] = nav_path / float(buy_nav)
            usable_rows.append(
                {
                    "fund_code": code,
                    "weight": float(row["weight"]),
                    "buy_nav_at_d3": float(buy_nav),
                    "sell_nav_at_d4": float(sell_nav),
                }
            )

        if not usable_rows:
            w.status = "skipped"
            w.skip_reason = "no_fund_with_valid_nav_on_d3_d4"
            window_rows.append(
                {
                    "window_id": w.window_id,
                    "d1_anchor_date": w.d1_anchor_date.strftime("%Y-%m-%d"),
                    "d2_last_hist_trade_date": w.d2_last_hist_trade_date.strftime("%Y-%m-%d"),
                    "d3_buy_date": w.d3_buy_date.strftime("%Y-%m-%d"),
                    "d4_sell_date": w.d4_sell_date.strftime("%Y-%m-%d"),
                    "calendar_holding_days": int((w.d4_sell_date - w.d3_buy_date).days),
                    "trading_holding_days": len(interval_dates),
                    "selection_data_version": w.selection_data_version,
                    "selected_fund_count": 0,
                    "weight_sum": 0.0,
                    "portfolio_return": np.nan,
                    "portfolio_nav_start": np.nan,
                    "portfolio_nav_end": np.nan,
                    "max_drawdown_in_window": np.nan,
                    "volatility_in_window": np.nan,
                    "status": "skipped",
                    "skip_reason": w.skip_reason,
                }
            )
            continue

        usable_df = pd.DataFrame(usable_rows)
        usable_df["weight"] = usable_df["weight"] / usable_df["weight"].sum()
        weight_map = dict(zip(usable_df["fund_code"], usable_df["weight"]))
        portfolio_nav = sum(portfolio_nav_df[code] * float(weight_map[code]) for code in portfolio_nav_df.columns)
        portfolio_nav = portfolio_nav.sort_index()
        portfolio_return = float(portfolio_nav.iloc[-1] - 1.0)
        daily_return = portfolio_nav.pct_change().dropna()
        volatility = float(daily_return.std(ddof=1)) if len(daily_return) > 1 else 0.0
        mdd = _max_drawdown(portfolio_nav)

        for _, row in usable_df.iterrows():
            fund_return = float(row["sell_nav_at_d4"] / row["buy_nav_at_d3"] - 1.0)
            contribution = float(row["weight"] * fund_return)
            position_rows.append(
                {
                    "window_id": w.window_id,
                    "fund_code": row["fund_code"],
                    "weight": float(row["weight"]),
                    "buy_nav_at_d3": float(row["buy_nav_at_d3"]),
                    "sell_nav_at_d4": float(row["sell_nav_at_d4"]),
                    "fund_return": fund_return,
                    "contribution": contribution,
                }
            )

        window_rows.append(
            {
                "window_id": w.window_id,
                "d1_anchor_date": w.d1_anchor_date.strftime("%Y-%m-%d"),
                "d2_last_hist_trade_date": w.d2_last_hist_trade_date.strftime("%Y-%m-%d"),
                "d3_buy_date": w.d3_buy_date.strftime("%Y-%m-%d"),
                "d4_sell_date": w.d4_sell_date.strftime("%Y-%m-%d"),
                "calendar_holding_days": int((w.d4_sell_date - w.d3_buy_date).days),
                "trading_holding_days": len(interval_dates),
                "selection_data_version": w.selection_data_version,
                "selected_fund_count": int(len(usable_df)),
                "weight_sum": float(usable_df["weight"].sum()),
                "portfolio_return": portfolio_return,
                "portfolio_nav_start": float(portfolio_nav.iloc[0]),
                "portfolio_nav_end": float(portfolio_nav.iloc[-1]),
                "max_drawdown_in_window": float(mdd),
                "volatility_in_window": volatility,
                "status": "ok",
                "skip_reason": "",
            }
        )

    window_detail = pd.DataFrame(window_rows).sort_values("window_id").reset_index(drop=True)
    position_cols = ["window_id", "fund_code", "weight", "buy_nav_at_d3", "sell_nav_at_d4", "fund_return", "contribution"]
    if position_rows:
        window_positions = pd.DataFrame(position_rows).sort_values(["window_id", "fund_code"]).reset_index(drop=True)
    else:
        window_positions = pd.DataFrame(columns=position_cols)

    ok_df = window_detail[window_detail["status"] == "ok"].copy()
    ret = ok_df["portfolio_return"].dropna()
    if ret.empty:
        agg = {
            "total_windows": int(len(window_detail)),
            "valid_windows": 0,
            "skipped_windows": int((window_detail["status"] != "ok").sum()),
            "win_rate": 0.0,
            "mean_return": np.nan,
            "median_return": np.nan,
            "std_return": np.nan,
            "p10_return": np.nan,
            "p50_return": np.nan,
            "p90_return": np.nan,
            "best_window_return": np.nan,
            "worst_window_return": np.nan,
        }
    else:
        agg = {
            "total_windows": int(len(window_detail)),
            "valid_windows": int(len(ok_df)),
            "skipped_windows": int((window_detail["status"] != "ok").sum()),
            "win_rate": float((ret > 0).mean()),
            "mean_return": float(ret.mean()),
            "median_return": float(ret.median()),
            "std_return": float(ret.std(ddof=1)) if len(ret) > 1 else 0.0,
            "p10_return": float(ret.quantile(0.10)),
            "p50_return": float(ret.quantile(0.50)),
            "p90_return": float(ret.quantile(0.90)),
            "best_window_return": float(ret.max()),
            "worst_window_return": float(ret.min()),
        }
    aggregate_summary = pd.DataFrame([agg])

    run_summary = pd.DataFrame(
        [
            {
                "run_id": pd.Timestamp.now().strftime("%Y%m%d%H%M%S"),
                "run_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "rebalance_interval_days": args.rebalance_interval_days,
                "holding_period_days": args.holding_period_days,
                "trade_calendar": str(trade_dates_csv),
                "fee_mode": "ignore_subscribe_redeem_conversion_fee",
                "overlap_mode": "independent_windows",
                "selection_rule_id": args.selection_rule_id,
                "selection_data_version_mode": args.selection_data_version or "auto_by_d2",
                "nav_data_version": nav_data_version,
                "selection_limit": args.selection_limit,
                "selection_where": args.selection_where,
                "selection_order_by": args.selection_order_by,
            }
        ]
    )

    run_summary.to_csv(output_dir / "backtest_run_summary.csv", index=False, encoding="utf-8-sig")
    window_detail.to_csv(output_dir / "backtest_window_detail.csv", index=False, encoding="utf-8-sig")
    window_positions.to_csv(output_dir / "backtest_window_positions.csv", index=False, encoding="utf-8-sig")
    aggregate_summary.to_csv(output_dir / "backtest_aggregate_summary.csv", index=False, encoding="utf-8-sig")
    _render_markdown_report(run_summary, aggregate_summary, window_detail, output_dir)

    print(f"run_summary={output_dir / 'backtest_run_summary.csv'}")
    print(f"window_detail={output_dir / 'backtest_window_detail.csv'}")
    print(f"window_positions={output_dir / 'backtest_window_positions.csv'}")
    print(f"aggregate_summary={output_dir / 'backtest_aggregate_summary.csv'}")
    print(f"report={output_dir / 'backtest_report.md'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="回测脚本：基于 ClickHouse 持久化基金数据按规则选基并回测")
    parser.add_argument("--start-date", required=True, help="回测起始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="回测结束日期，格式 YYYY-MM-DD")
    parser.add_argument("--output-dir", required=True, type=Path, help="输出目录")
    parser.add_argument(
        "--trade-dates-csv",
        default=str(project_root() / "data" / "common" / "trade_dates.csv"),
        help="交易日历CSV，默认 data/common/trade_dates.csv",
    )

    parser.add_argument("--rebalance-interval-days", type=int, default=15, help="窗口间隔天数")
    parser.add_argument("--holding-period-days", type=int, default=30, help="持有自然日")

    parser.add_argument("--selection-rule-id", default="default_rule_v1", help="选基规则ID")
    parser.add_argument("--selection-data-version", default=None, help="固定选基 data_version，默认按 d2 自动选择")
    parser.add_argument("--selection-where", default="1", help="选基 WHERE 条件（作用于 scoreboard 快照）")
    parser.add_argument(
        "--selection-order-by",
        default="annual_return_rank ASC, fund_code ASC",
        help="选基 ORDER BY 表达式",
    )
    parser.add_argument("--selection-limit", type=int, default=10, help="选基金数量上限")
    parser.add_argument("--exclude-subscribe-status", default="暂停申购,封闭期", help="排除的申购状态，逗号分隔")
    parser.add_argument("--exclude-redeem-status", default="暂停赎回,封闭期", help="排除的赎回状态，逗号分隔")

    parser.add_argument("--nav-data-version", default=None, help="净值 data_version，默认自动取最新")

    parser.add_argument("--clickhouse-db", default="fund_analysis")
    parser.add_argument("--clickhouse-container", default="fund_clickhouse")
    return parser


if __name__ == "__main__":
    run_backtest(build_parser().parse_args())
