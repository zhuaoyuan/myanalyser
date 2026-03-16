#!/usr/bin/env python3
"""v2 多 T 日回测主入口。

协议约定（详见 docs/参考/v2日期与区间协议约定.md）：
- lookback：1 年 = 243 交易日（backtest_helpers.compute_start_from_lookback）
- hold_days / t-step：交易日
- 日期区间：[start, end] 双闭；filter/compare/integrity 延伸至 end_extended
"""
from __future__ import annotations

# python myanalyser/tools/v2/multi_t_backtest.py \
#   --run-id "20260315_123456_full_run_v2" \
#   --ruleset-version "20260315_v1" \
#   --t-list "2023-01-03,2023-07-03,2024-01-02,2024-07-01" \
#   --trading-calendar-csv "myanalyser/data/common/trade_dates.csv" \
#   --prep-work-dir "myanalyser/tmp/prep_work_v2" \
#   --lookback-years 3 \
#   --hold-days 21 \
#   --strategy "low_risk_debt_most_stable" \
#   --max-funds 5000 \
#   --rebalance 0 \
#   --top-n 5 \
#   --warmup 0 \
#   --initial-cash 100000    

# python myanalyser/tools/v2/multi_t_backtest.py \
#   --run-id "20260315_123456_full_run_v2" \
#   --ruleset-version "20260315_v2" \
#   --t-start 2010-01-01 --t-end 2026-03-01 --t-step 25 \
#   --trading-calendar-csv "myanalyser/data/common/trade_dates.csv" \
#   --prep-work-dir "myanalyser/tmp/prep_work_v2" \
#   --lookback-years 3 \
#   --hold-days 42 \
#   --strategy "low_risk_debt_most_stable" \
#   --max-funds 5000 \
#   --rebalance 0 \
#   --top-n 5 \
#   --warmup 0 \
#   --initial-cash 100000    


import argparse
import logging
import os
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_MYANALYSER_ROOT = _SCRIPT_DIR.parent.parent  # tools/v2 -> tools -> myanalyser
_SRC = _MYANALYSER_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from project_paths import project_root

from backtest import load_fund_nav_data, run_backtest
from backtest.engine import BacktestConfig, write_reports
from backtest.strategies.registry import get_strategy_bundle, list_strategy_names
from pipeline_scoreboard import run_pipeline as run_scoreboard
from transforms.build_filtered_purchase_csv import build_filtered_purchase_csv
from v2.compare.compare_adjusted_nav_and_cum_return_window import (
    compare_adjusted_nav_and_cum_return_window,
)
from backtest_helpers import compute_end_extended_str, compute_start_from_lookback
from v2.filters.filter_funds_for_next_step import filter_funds_for_next_step
from v2.filters.prep_eligible_window import (
    compute_personnel_excluded_and_merge,
    run as run_prep_eligible_window,
)
from check_trade_day_data_integrity import (
    load_trade_days,
    load_eligible_fund_codes,
    compute_integrity_for_fund,
)

try:
    from fund_metrics_core import HOLDING_METRIC_NAMES
except ImportError:
    logging.warning("fund_metrics_core 未安装，multi_summary_agg 将跳过")
    HOLDING_METRIC_NAMES = ()


logger = logging.getLogger(__name__)

# win_rate 针对的列名，与 fund_metrics_core.HOLDING_METRIC_NAMES 对齐
_ANN_RETURN_COL = "年化收益率"


def _load_trade_calendar(trading_calendar_csv: Path) -> list[pd.Timestamp]:
    trade_df = pd.read_csv(trading_calendar_csv, dtype={"trade_date": str}, encoding="utf-8-sig")
    if "trade_date" not in trade_df.columns:
        raise ValueError(f"交易日历缺少 trade_date 列: {trading_calendar_csv}")
    trade_df["trade_date"] = pd.to_datetime(trade_df["trade_date"], errors="coerce")
    dates = trade_df.dropna(subset=["trade_date"])["trade_date"].dt.normalize().unique().tolist()
    return sorted(pd.Timestamp(d) for d in dates)


def _resolve_trade_day(target: pd.Timestamp, trading_days: list[pd.Timestamp]) -> pd.Timestamp:
    target = pd.Timestamp(target).normalize()
    idx = bisect_right(trading_days, target) - 1
    if idx < 0:
        raise ValueError(f"date {target.date()} is earlier than trading calendar start")
    return trading_days[idx]


def _parse_t_list(raw: str) -> list[pd.Timestamp]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        return []
    return [pd.to_datetime(x) for x in items]


def _build_t_list(
    *,
    t_list: str | None,
    t_start: str | None,
    t_end: str | None,
    t_step: int | None,
    trading_days: list[pd.Timestamp],
) -> list[pd.Timestamp]:
    if t_list:
        raw_list = _parse_t_list(t_list)
        resolved = {_resolve_trade_day(ts, trading_days) for ts in raw_list}
        return sorted(resolved)

    if not (t_start and t_end and t_step):
        raise ValueError("must provide --t-list or (--t-start, --t-end, --t-step)")

    start_ts = _resolve_trade_day(pd.to_datetime(t_start), trading_days)
    end_ts = _resolve_trade_day(pd.to_datetime(t_end), trading_days)
    if start_ts > end_ts:
        raise ValueError(f"t-start cannot be after t-end: {t_start} > {t_end}")

    start_idx = bisect_left(trading_days, start_ts)
    end_idx = bisect_right(trading_days, end_ts) - 1
    if start_idx < 0 or end_idx < 0:
        raise ValueError("invalid trading calendar range for t-start/t-end")

    indices = list(range(start_idx, end_idx + 1, int(t_step)))
    return [trading_days[i] for i in indices]


def _cache_dir(root: Path, *parts: str) -> Path:
    return root.joinpath(*parts)


def _ensure_compare_cache(
    *,
    base_dir: Path,
    cache_dir: Path,
    start_date: str,
    end_date: str,
) -> tuple[Path, Path]:
    summary_csv = cache_dir / "summary.csv"
    details_dir = cache_dir / "details"
    if summary_csv.exists() and details_dir.exists():
        logger.info("[compare] cache hit: %s", cache_dir)
        return summary_csv, details_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    result = compare_adjusted_nav_and_cum_return_window(
        base_dir=base_dir,
        start_date=start_date,
        end_date=end_date,
        output_dir=cache_dir,
        error_log_path=cache_dir / "errors.jsonl",
    )
    return result["summary_csv"], result["detail_dir"]


def _run_integrity_window(
    *,
    base_dir: Path,
    trade_dates_csv: Path,
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    fund_dir = base_dir / "fund_adjusted_nav_by_code"
    overview_csv = base_dir / "fund_overview.csv"
    trade_dates_csv = trade_dates_csv.resolve()

    if not fund_dir.is_dir():
        raise FileNotFoundError(f"未找到目录: {fund_dir}")
    if not overview_csv.is_file():
        raise FileNotFoundError(f"未找到文件: {overview_csv}")
    if not trade_dates_csv.is_file():
        raise FileNotFoundError(f"未找到交易日历文件: {trade_dates_csv}")

    trade_days = load_trade_days(trade_dates_csv, start_date, end_date)
    eligible_codes = load_eligible_fund_codes(overview_csv, start_date)

    summary_output = output_dir / f"trade_day_integrity_summary_{start_date}_{end_date}.csv"
    details_dir = output_dir / f"details_{start_date}_{end_date}"
    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []
    fund_files = sorted(fund_dir.glob("*.csv"))
    processed_count = 0

    for fund_csv in fund_files:
        if fund_csv.stem not in eligible_codes:
            continue
        fund_code, ratio, detail_df = compute_integrity_for_fund(fund_csv, trade_days)
        summary_records.append({"基金编码": fund_code, "数据完整比例": f"{ratio:.6f}"})
        processed_count += 1

        detail_output = details_dir / f"{fund_code}_{start_date}_{end_date}.csv"
        detail_df.to_csv(detail_output, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summary_records, columns=["基金编码", "数据完整比例"])
    summary_df.to_csv(summary_output, index=False, encoding="utf-8-sig")

    logger.info(
        "[integrity] trade_days=%d funds=%d included=%d -> %s",
        len(trade_days),
        len(fund_files),
        processed_count,
        summary_output,
    )
    return summary_output, details_dir


def _ensure_integrity_cache(
    *,
    base_dir: Path,
    cache_dir: Path,
    trade_dates_csv: Path,
    start_date: str,
    end_date: str,
) -> tuple[Path, Path]:
    summary_csv = cache_dir / f"trade_day_integrity_summary_{start_date}_{end_date}.csv"
    details_dir = cache_dir / f"details_{start_date}_{end_date}"
    if summary_csv.exists() and details_dir.exists():
        logger.info("[integrity] cache hit: %s", cache_dir)
        return summary_csv, details_dir

    return _run_integrity_window(
        base_dir=base_dir,
        trade_dates_csv=trade_dates_csv,
        start_date=start_date,
        end_date=end_date,
        output_dir=cache_dir,
    )


def _build_purchase_csv_for_filter(
    eligible_csv: Path,
    filter_dir: Path,
    fund_types: list[str],
    type_allowed_codes: set[str] | None,
    logger: logging.Logger,
) -> Path:
    """按基金类型筛选 eligible，返回用于 filter 步骤的 purchase CSV 路径。

    当 fund_types 为空时返回 eligible_csv；否则与 type_allowed_codes 取交集并写入 filter_dir。
    """
    if not fund_types or type_allowed_codes is None:
        return eligible_csv

    if not type_allowed_codes:
        raise ValueError(
            f"--fund-types {fund_types} 在 fund_fee_filtered.csv 中无匹配基金，请检查类型拼写"
        )

    eligible_df = pd.read_csv(eligible_csv, dtype=str, encoding="utf-8-sig")
    if "基金编码" in eligible_df.columns:
        code_col = "基金编码"
    elif "基金代码" in eligible_df.columns:
        code_col = "基金代码"
    else:
        raise ValueError(
            f"eligible CSV 缺少 基金编码 或 基金代码 列: {list(eligible_df.columns)}"
        )

    eligible_df["_code"] = eligible_df[code_col].astype(str).str.strip().str.zfill(6)
    filtered_df = eligible_df[eligible_df["_code"].isin(type_allowed_codes)].drop(
        columns=["_code"]
    )
    if filtered_df.empty:
        raise ValueError(
            f"eligible 与 fund-types {fund_types} 取交后为空，请检查 prep 与类型配置"
        )

    filter_dir.mkdir(parents=True, exist_ok=True)
    output_path = filter_dir / "eligible_by_type.csv"
    filtered_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(
        "[fund-types] %s -> %d 基金（原 %d）",
        ",".join(fund_types),
        len(filtered_df),
        len(eligible_df),
    )
    return output_path


def _load_type_filtered_codes(prep_work_dir: Path, fund_types: list[str]) -> set[str]:
    """从 prep-work-dir/fund_fee_filtered.csv 筛选指定类型的基金编码。"""
    fee_csv = prep_work_dir / "fund_fee_filtered.csv"
    if not fee_csv.exists():
        raise FileNotFoundError(f"基金类型筛选需要 {fee_csv}，请先运行 prep 生成")
    df = pd.read_csv(fee_csv, dtype={"基金编码": str, "类型": str}, encoding="utf-8-sig")
    if "类型" not in df.columns or "基金编码" not in df.columns:
        raise ValueError(f"{fee_csv} 缺少 类型 或 基金编码 列")
    types_set = {t.strip() for t in fund_types}
    filtered = df[df["类型"].astype(str).str.strip().isin(types_set)]
    return {str(c).strip().zfill(6) for c in filtered["基金编码"].dropna().tolist() if c}


def _read_allowed_codes(filter_csv: Path) -> set[str]:
    df = pd.read_csv(filter_csv, dtype={"基金编码": str}, encoding="utf-8-sig")
    if df.empty or "基金编码" not in df.columns:
        return set()
    if "是否过滤" in df.columns:
        allowed_df = df[df["是否过滤"].astype(str).str.strip() == "否"]
    else:
        allowed_df = df
    return {str(v).strip().zfill(6) for v in allowed_df["基金编码"].dropna().tolist()}


def _write_multi_summary_agg(summary_df: pd.DataFrame, output_root: Path) -> None:
    """基于 summary_df 生成 multi_summary_agg.csv，含跨 T 的汇总统计。

    仅对 multi_summary 中存在的数值型 metrics 列（与 fund_metrics_core.HOLDING_METRIC_NAMES 白名单
    取交）计算 mean/median/std/min/max/p25/p75/count、win_rate（_ANN_RETURN_COL>0 比例）、t_count。
    t_count 写入首列 metric_cols[0]，下游解析时以 stat_type=t_count 行、首指标列读取。
    """
    if summary_df.empty:
        logger.info("[multi] agg summary skipped: summary_df empty")
        return

    known_metrics = set(HOLDING_METRIC_NAMES) if HOLDING_METRIC_NAMES else set()
    metric_cols = [c for c in summary_df.columns if c in known_metrics]
    if not metric_cols:
        logger.info("[multi] agg summary skipped: no metric columns")
        return

    numeric_df = summary_df[metric_cols].apply(pd.to_numeric, errors="coerce")

    t_count = len(summary_df)
    agg_rows: list[dict[str, object]] = []

    def _fmt(v: float) -> str | float:
        if pd.isna(v):
            return ""
        return v

    def _std_fn(s: pd.Series) -> float:
        return s.std() if s.notna().sum() > 1 else float("nan")

    _STAT_FNS: dict[str, object] = {
        "mean": lambda s: s.mean(),
        "median": lambda s: s.median(),
        "std": _std_fn,
        "min": lambda s: s.min(),
        "max": lambda s: s.max(),
    }
    for stat, fn in _STAT_FNS.items():
        row: dict[str, object] = {"stat_type": stat}
        for c in metric_cols:
            row[c] = _fmt(fn(numeric_df[c]))
        agg_rows.append(row)

    for q_name, q_val in [("p25", 0.25), ("p75", 0.75)]:
        row = {"stat_type": q_name}
        for c in metric_cols:
            row[c] = _fmt(numeric_df[c].quantile(q_val))
        agg_rows.append(row)

    row = {"stat_type": "count"}
    for c in metric_cols:
        row[c] = int(numeric_df[c].notna().sum())
    agg_rows.append(row)

    ann_ret_col = _ANN_RETURN_COL if _ANN_RETURN_COL in metric_cols else None
    row = {"stat_type": "win_rate"}
    for c in metric_cols:
        if c == ann_ret_col:
            s = numeric_df[c].dropna()
            n = len(s)
            row[c] = round((s > 0).sum() / n, 6) if n > 0 else ""
        else:
            row[c] = ""
    agg_rows.append(row)

    # t_count 元信息
    row = {"stat_type": "t_count"}
    for c in metric_cols:
        row[c] = "" if c != metric_cols[0] else t_count
    agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows, columns=["stat_type"] + metric_cols)
    agg_path = output_root / "multi_summary_agg.csv"
    agg_df.to_csv(agg_path, index=False, encoding="utf-8-sig")
    logger.info("[multi] agg summary -> %s", agg_path)


def _extract_metrics(summary_csv: Path) -> dict[str, str]:
    if not summary_csv.exists():
        return {}
    df = pd.read_csv(summary_csv, dtype=str, encoding="utf-8-sig")
    metrics = {}
    if df.empty:
        return metrics
    metric_df = df[df["section"] == "metrics_holding"] if "section" in df.columns else pd.DataFrame()
    for _, row in metric_df.iterrows():
        name = str(row.get("name", "")).strip()
        val = str(row.get("value", "")).strip()
        if name:
            metrics[name] = val
    return metrics


def _run_scoreboard(
    *,
    purchase_csv: Path,
    overview_csv: Path,
    personnel_dir: Path,
    nav_dir: Path,
    output_dir: Path,
    data_version: str,
    as_of_date: str,
    latest_nav_date: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        purchase_csv=purchase_csv,
        overview_csv=overview_csv,
        personnel_dir=personnel_dir,
        nav_dir=nav_dir,
        output_dir=output_dir,
        data_version=data_version,
        as_of_date=as_of_date,
        latest_nav_date=latest_nav_date,
        stale_max_days=2,
        code_limit=None,
        skip_sinks=True,
        formal_only=True,
        resume=False,
        apply_ddl=False,
        # fund_db_infra 约定与 myanalyser 平级（workspace 根下）；仓库结构调整时需同步修改
        mysql_ddl=project_root().parent / "fund_db_infra" / "sql" / "mysql_schema.sql",
        clickhouse_ddl=project_root().parent / "fund_db_infra" / "sql" / "clickhouse_schema.sql",
        mysql_host="127.0.0.1",
        mysql_port=3306,
        mysql_user="root",
        mysql_password=os.environ.get("MYSQL_PASSWORD", ""),
        mysql_db="fund_analysis",
        clickhouse_host="127.0.0.1",
        clickhouse_port=8123,
        clickhouse_user="default",
        clickhouse_password="",
        clickhouse_db="fund_analysis",
        clickhouse_container="fund_clickhouse",
        clickhouse_write_profile="auto",
        small_data_threshold_funds=200,
        clickhouse_write_scope="full",
    )
    run_scoreboard(args)
    scoreboard_csv = output_dir / f"fund_scoreboard_{data_version}.csv"
    if not scoreboard_csv.exists():
        raise FileNotFoundError(f"scoreboard output not found: {scoreboard_csv}")
    return scoreboard_csv


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="v2 multi-T backtest runner (windowed cache)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ruleset-version", required=True)
    parser.add_argument("--t-list", default=None)
    parser.add_argument("--t-start", default=None)
    parser.add_argument("--t-end", default=None)
    parser.add_argument("--t-step", type=int, default=None, help="step in trading days")
    parser.add_argument("--lookback-years", type=int, default=3)
    parser.add_argument("--hold-days", type=int, default=42)
    parser.add_argument("--trading-calendar-csv", type=Path, required=True)
    parser.add_argument("--prep-work-dir", type=Path, required=True)
    parser.add_argument("--strategy", default="low_risk_debt", help=f"strategy name: {', '.join(list_strategy_names())}")
    parser.add_argument("--max-funds", type=int, default=200)
    parser.add_argument("--rebalance", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=243)
    parser.add_argument("--initial-cash", type=float, default=100_000)
    parser.add_argument(
        "--fund-types",
        nargs="*",
        default=None,
        help="基金类型列表，从 prep-work-dir/fund_fee_filtered.csv 筛选。"
        "可空格或逗号分隔，如 --fund-types A类730天 C类30天",
    )
    args = parser.parse_args()

    run_id = args.run_id
    ruleset_version = args.ruleset_version
    # 支持空格分隔与逗号分隔，如 ["A类730天", "C类30天"] 或 ["A类730天,C类30天"] -> ["A类730天", "C类30天"]
    fund_types = [
        x.strip()
        for v in (args.fund_types or [])
        for x in str(v).split(",")
        if x.strip()
    ]

    workspace_root = project_root()
    data_root = workspace_root / "data" / "versions" / run_id
    fund_etl_dir = data_root / "fund_etl"
    if not fund_etl_dir.is_dir():
        raise FileNotFoundError(f"fund_etl dir not found: {fund_etl_dir}")

    trading_calendar_csv = args.trading_calendar_csv.resolve()
    trading_days = _load_trade_calendar(trading_calendar_csv)
    t_list = _build_t_list(
        t_list=args.t_list,
        t_start=args.t_start,
        t_end=args.t_end,
        t_step=args.t_step,
        trading_days=trading_days,
    )
    if not t_list:
        raise ValueError("empty T list")

    cache_root = data_root / "cache" / "v2"
    output_root = workspace_root / "artifacts" / "backtest_multi" / run_id / ruleset_version
    output_root.mkdir(parents=True, exist_ok=True)

    prep_work_dir = args.prep_work_dir.resolve()
    if not prep_work_dir.is_dir():
        raise FileNotFoundError(f"prep work dir not found: {prep_work_dir}")

    type_allowed_codes = _load_type_filtered_codes(prep_work_dir, fund_types) if fund_types else None

    summary_rows: list[dict[str, object]] = []

    for t in t_list:
        as_of_date = _resolve_trade_day(t, trading_days)
        as_of_str = as_of_date.strftime("%Y-%m-%d")
        start_date = compute_start_from_lookback(
            as_of_date, args.lookback_years, trading_days
        )
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = as_of_str

        end_extended_str = compute_end_extended_str(as_of_date, args.hold_days, trading_days)

        cache_key = f"{start_str}_{end_str}"
        cache_key_filter = f"{start_str}_{end_extended_str}"
        compare_dir = _cache_dir(cache_root, "compare", ruleset_version, cache_key_filter)
        integrity_dir = _cache_dir(cache_root, "integrity", ruleset_version, cache_key_filter)
        eligible_dir = _cache_dir(cache_root, "prep_eligible", ruleset_version, cache_key)
        filter_dir = _cache_dir(cache_root, "filter", ruleset_version, cache_key_filter)
        scoreboard_dir = _cache_dir(cache_root, "scoreboard", ruleset_version, as_of_str, cache_key)

        logger.info("[T=%s] start window %s -> %s, filter/compare/integrity 延伸至 %s", as_of_str, start_str, end_str, end_extended_str)

        compare_summary, compare_details = _ensure_compare_cache(
            base_dir=fund_etl_dir,
            cache_dir=compare_dir,
            start_date=start_str,
            end_date=end_extended_str,
        )

        integrity_summary, integrity_details = _ensure_integrity_cache(
            base_dir=fund_etl_dir,
            cache_dir=integrity_dir,
            trade_dates_csv=trading_calendar_csv,
            start_date=start_str,
            end_date=end_extended_str,
        )

        eligible_csv = eligible_dir / "eligible_fund_candidates.csv"
        base_path = eligible_dir / f"eligible_base_{start_str}_{end_str}.csv"
        personnel_excluded_path = eligible_dir / f"personnel_excluded_{start_str}_{end_str}.csv"
        personnel_dir_path = fund_etl_dir / "fund_personnel_by_code"

        if eligible_csv.exists():
            logger.info("[eligible] cache hit: %s", eligible_csv)
        elif base_path.exists():
            eligible_dir.mkdir(parents=True, exist_ok=True)
            if not personnel_dir_path.is_dir():
                base_df = pd.read_csv(base_path, dtype=str, encoding="utf-8-sig")
                base_df.to_csv(eligible_csv, index=False, encoding="utf-8-sig")
                logger.info("[eligible] cache hit (base only, personnel dir absent): %s", eligible_csv)
            elif personnel_excluded_path.exists():
                base_df = pd.read_csv(base_path, dtype=str, encoding="utf-8-sig")
                excl_df = pd.read_csv(personnel_excluded_path, dtype=str, encoding="utf-8-sig")
                excl_col = "基金编码" if "基金编码" in excl_df.columns else "基金代码"
                excluded = {str(c).strip().zfill(6) for c in excl_df[excl_col].dropna().tolist() if c}
                base_df["_code"] = base_df["基金代码"].astype(str).str.strip().str.zfill(6)
                final_df = base_df[~base_df["_code"].isin(excluded)].drop(columns=["_code"])
                final_df.to_csv(eligible_csv, index=False, encoding="utf-8-sig")
                logger.info("[eligible] cache hit (base + personnel merge): %s", eligible_csv)
            else:
                # base 存在、personnel 不存在：仅计算 personnel 并合并，避免重复跑 c.1+a+b+e
                compute_personnel_excluded_and_merge(
                    base_path=base_path,
                    personnel_dir=personnel_dir_path,
                    personnel_excluded_path=personnel_excluded_path,
                    output_path=eligible_csv,
                    start_date=start_str,
                    end_date=end_str,
                    logger=logger,
                )
        else:
            eligible_dir.mkdir(parents=True, exist_ok=True)
            run_prep_eligible_window(
                work_dir=prep_work_dir,
                start_date=start_str,
                end_date=end_str,
                personnel_dir=personnel_dir_path,
                output_path=eligible_csv,
                logger=logger,
            )

        purchase_csv_for_filter = _build_purchase_csv_for_filter(
            eligible_csv=eligible_csv,
            filter_dir=filter_dir,
            fund_types=fund_types,
            type_allowed_codes=type_allowed_codes,
            logger=logger,
        )

        filter_csv = filter_dir / "filtered_fund_candidates.csv"
        # fund_types 非空时 filter 不缓存（便于调整筛选条件）；否则使用 ruleset_version 区分缓存
        if fund_types:
            filter_dir.mkdir(parents=True, exist_ok=True)
            filter_df = filter_funds_for_next_step(
                purchase_csv=purchase_csv_for_filter,
                overview_csv=fund_etl_dir / "fund_overview.csv",
                nav_dir=fund_etl_dir / "fund_nav_by_code",
                adjusted_nav_dir=fund_etl_dir / "fund_adjusted_nav_by_code",
                compare_details_dir=compare_details,
                integrity_details_dir=integrity_details,
                start_date=start_str,
                end_date=end_extended_str,
            )
            filter_df.to_csv(filter_csv, index=False, encoding="utf-8-sig")
            logger.info("[filter] write %s (fund-types 指定，未使用缓存)", filter_csv)
        elif filter_csv.exists():
            logger.info("[filter] cache hit: %s", filter_csv)
        else:
            filter_dir.mkdir(parents=True, exist_ok=True)
            filter_df = filter_funds_for_next_step(
                purchase_csv=purchase_csv_for_filter,
                overview_csv=fund_etl_dir / "fund_overview.csv",
                nav_dir=fund_etl_dir / "fund_nav_by_code",
                adjusted_nav_dir=fund_etl_dir / "fund_adjusted_nav_by_code",
                compare_details_dir=compare_details,
                integrity_details_dir=integrity_details,
                start_date=start_str,
                end_date=end_extended_str,
            )
            filter_df.to_csv(filter_csv, index=False, encoding="utf-8-sig")
            logger.info("[filter] write %s", filter_csv)

        allowed_codes = _read_allowed_codes(filter_csv)
        if not allowed_codes:
            raise ValueError(f"filtered candidates empty for T={as_of_str}")

        purchase_filtered_csv = filter_dir / "fund_purchase_for_step10_filtered.csv"
        if not purchase_filtered_csv.exists():
            build_filtered_purchase_csv(
                purchase_csv=purchase_csv_for_filter,
                filter_csv=filter_csv,
                output_csv=purchase_filtered_csv,
            )
            if not purchase_filtered_csv.exists():
                raise FileNotFoundError(
                    f"build_filtered_purchase_csv 未生成 {purchase_filtered_csv}"
                )

        scoreboard_csv = scoreboard_dir / f"fund_scoreboard_{run_id}_{ruleset_version}_{as_of_str}.csv"
        if scoreboard_csv.exists():
            logger.info("[scoreboard] cache hit: %s", scoreboard_csv)
        else:
            scoreboard_dir.mkdir(parents=True, exist_ok=True)
            scoreboard_csv = _run_scoreboard(
                purchase_csv=purchase_filtered_csv,
                overview_csv=fund_etl_dir / "fund_overview.csv",
                personnel_dir=fund_etl_dir / "fund_personnel_by_code",
                nav_dir=fund_etl_dir / "fund_adjusted_nav_by_code",
                output_dir=scoreboard_dir,
                data_version=f"{run_id}_{ruleset_version}_{as_of_str}",
                as_of_date=as_of_str,
                latest_nav_date=as_of_str,
            )

        t_index = bisect_left(trading_days, as_of_date)
        end_index = t_index + args.hold_days
        if end_index >= len(trading_days):
            raise ValueError(f"hold-days exceeds trading calendar end for T={as_of_str}")
        backtest_end = trading_days[end_index]
        backtest_end_str = backtest_end.strftime("%Y-%m-%d")

        # 使用 lookback 起始日，使 MostStableFilterStrategy 等策略能在 T 日有足够历史数据计算 3 年指标。
        # 注意：max_funds 较大时加载窗口变长会增加内存占用。
        data = load_fund_nav_data(
            fund_etl_dir / "fund_adjusted_nav_by_code",
            max_funds=args.max_funds,
            start_date=start_str,
            end_date=backtest_end_str,
            allowed_codes=allowed_codes,
        )
        if "date" in data.long_df.columns:
            min_avail = data.long_df["date"].min()
            start_ts = pd.to_datetime(start_str)
            if min_avail is not None and pd.notna(min_avail) and min_avail > start_ts:
                logger.warning(
                    "[T=%s] 数据实际起始晚于 lookback 起始，策略可能缺少足够历史: %s > %s",
                    as_of_str,
                    min_avail,
                    start_str,
                )

        bundle = get_strategy_bundle(args.strategy)
        config = BacktestConfig(initial_cash=args.initial_cash)
        backtest_result = run_backtest(
            data,
            bundle,
            start_date=as_of_str,
            end_date=backtest_end_str,
            top_n=args.top_n,
            rebalance_period=args.rebalance,
            warmup=args.warmup,
            config=config,
        )

        t_output_dir = output_root / as_of_str
        reports = write_reports(
            t_output_dir,
            backtest_result,
            data,
            run_config={
                "strategy": args.strategy,
                "start_date": as_of_str,
                "end_date": backtest_end_str,
                "rebalance": args.rebalance,
                "top_n": args.top_n,
                "warmup": args.warmup,
                "initial_cash": args.initial_cash,
                "nav_dir": str((fund_etl_dir / "fund_adjusted_nav_by_code").resolve()),
                "max_funds": args.max_funds,
                "run_id": run_id,
                "ruleset_version": ruleset_version,
                "filter_start": start_str,
                "filter_end": end_str,
            },
            initial_cash=config.initial_cash,
        )

        metrics = _extract_metrics(reports["summary"])
        summary_rows.append(
            {
                "as_of_date": as_of_str,
                "filter_start": start_str,
                "filter_end": end_str,
                "backtest_start": as_of_str,
                "backtest_end": backtest_end_str,
                "allowed_funds": len(allowed_codes),
                "compare_summary": str(compare_summary),
                "integrity_summary": str(integrity_summary),
                "eligible_csv": str(eligible_csv),
                "filter_csv": str(filter_csv),
                "scoreboard_csv": str(scoreboard_csv),
                "summary_csv": str(reports["summary"]),
                "detail_csv": str(reports["detail"]),
                "report_md": str(reports["report_md"]),
                **metrics,
            }
        )

        logger.info("[T=%s] done -> %s", as_of_str, t_output_dir)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_root / "multi_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    logger.info("[multi] summary -> %s", summary_csv)

    _write_multi_summary_agg(summary_df, output_root)


if __name__ == "__main__":
    main()
