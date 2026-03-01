from __future__ import annotations

import argparse
import pickle
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql
from validators.validate_pipeline_artifacts import validate_stage_or_raise

from scoreboard_metrics import (
    METRIC_DIRECTIONS,
    compute_metrics as _metrics_compute,
    load_nav_df as _metrics_load_nav,
    window_metrics as _metrics_window,
)

# 并发读取 CSV 的线程数，控制 I/O 并发与资源占用
MAX_CSV_WORKERS = 8

# 计算阶段 checkpoint 文件名（用于 --resume 断点续传）
CHECKPOINT_KEYS = [
    "dim_base",
    "metric_df",
    "nav_df",
    "period_df",
    "scoreboard",
    "exclusion_detail",
    "exclusion_summary",
]

CH_METRICS_COLS = [
    "data_version",
    "as_of_date",
    "fund_code",
    "nav_start_date",
    "nav_end_date",
    "annual_return",
    "up_quarter_ratio",
    "up_month_ratio",
    "up_week_ratio",
    "quarter_return_std",
    "month_return_std",
    "week_return_std",
    "max_drawdown",
    "annual_return_3y",
    "up_quarter_ratio_3y",
    "up_month_ratio_3y",
    "up_week_ratio_3y",
    "quarter_return_std_3y",
    "month_return_std_3y",
    "week_return_std_3y",
    "max_drawdown_3y",
    "annual_return_1y",
    "up_month_ratio_1y",
    "up_week_ratio_1y",
    "month_return_std_1y",
    "week_return_std_1y",
    "max_drawdown_1y",
    "recent_month_return",
    "sharpe_ratio_1y",
    "sharpe_ratio_3y",
    "calmar_ratio_1y",
    "calmar_ratio_3y",
    "max_drawdown_recovery_days",
    "max_drawdown_recovery_days_1y",
    "max_drawdown_recovery_days_3y",
    "max_single_day_drop",
    "max_single_day_drop_1y",
    "max_single_day_drop_3y",
]

CH_SCOREBOARD_COLS = [
    "data_version",
    "as_of_date",
    "fund_code",
    "fund_name",
    "nav_start_date",
    "nav_end_date",
    "scale_billion",
    "inception_years",
    "annual_return",
    "up_quarter_ratio",
    "up_month_ratio",
    "up_week_ratio",
    "quarter_return_std",
    "month_return_std",
    "week_return_std",
    "max_drawdown",
    "annual_return_3y",
    "up_quarter_ratio_3y",
    "up_month_ratio_3y",
    "up_week_ratio_3y",
    "quarter_return_std_3y",
    "month_return_std_3y",
    "week_return_std_3y",
    "max_drawdown_3y",
    "annual_return_1y",
    "up_month_ratio_1y",
    "up_week_ratio_1y",
    "month_return_std_1y",
    "week_return_std_1y",
    "max_drawdown_1y",
    "recent_month_return",
    "sharpe_ratio_1y",
    "sharpe_ratio_3y",
    "calmar_ratio_1y",
    "calmar_ratio_3y",
    "max_drawdown_recovery_days",
    "max_drawdown_recovery_days_1y",
    "max_drawdown_recovery_days_3y",
    "max_single_day_drop",
    "max_single_day_drop_1y",
    "max_single_day_drop_3y",
    "fund_type",
    "subscribe_status",
    "redeem_status",
    "next_open_day",
    "purchase_min_amount",
    "daily_limit_amount",
    "management_fee_rate",
    "custodian_fee_rate",
    "sales_service_fee_rate",
    "max_subscribe_fee_rate",
    "purchase_fee_rate",
    "fee_source_max_subscribe",
    "fee_source_purchase",
    "last_updated_date",
    "last_personnel_change_date",
]

EXPORT_COLUMN_SPECS = [
    ("fund_code", "基金代码", "raw"),
    ("fund_name", "基金名称", "raw"),
    ("nav_start_date", "期初日期", "date"),
    ("nav_end_date", "期末日期", "date"),
    ("scale_billion", "规模-亿元", "round2"),
    ("inception_years", "成立年数", "round2"),
    ("annual_return", "年化收益率", "percent2"),
    ("up_quarter_ratio", "上涨季度比例", "percent0"),
    ("up_month_ratio", "上涨月份比例", "percent0"),
    ("up_week_ratio", "上涨星期比例", "percent0"),
    ("quarter_return_std", "季涨跌幅标准差", "percent2"),
    ("month_return_std", "月涨跌幅标准差", "percent2"),
    ("week_return_std", "周涨跌幅标准差", "percent2"),
    ("max_drawdown", "最大回撤率", "percent2"),
    ("annual_return_3y", "近3年年化收益率", "percent2"),
    ("up_quarter_ratio_3y", "近3年上涨季度比例", "percent0"),
    ("up_month_ratio_3y", "近3年上涨月份比例", "percent0"),
    ("up_week_ratio_3y", "近3年上涨星期比例", "percent0"),
    ("quarter_return_std_3y", "近3年季涨跌幅标准差", "percent2"),
    ("month_return_std_3y", "近3年月涨跌幅标准差", "percent2"),
    ("week_return_std_3y", "近3年周涨跌幅标准差", "percent2"),
    ("max_drawdown_3y", "近3年最大回撤率", "percent2"),
    ("annual_return_1y", "近1年年化收益率", "percent2"),
    ("up_month_ratio_1y", "近1年上涨月份比例", "percent0"),
    ("up_week_ratio_1y", "近1年上涨星期比例", "percent0"),
    ("month_return_std_1y", "近1年月涨跌幅标准差", "percent2"),
    ("week_return_std_1y", "近1年周涨跌幅标准差", "percent2"),
    ("max_drawdown_1y", "近1年最大回撤率", "percent2"),
    ("recent_month_return", "最近一个月涨跌幅", "percent2"),
    ("sharpe_ratio_1y", "近1年夏普比率", "round2"),
    ("sharpe_ratio_3y", "近3年夏普比率", "round2"),
    ("calmar_ratio_1y", "近1年卡玛比率", "round2"),
    ("calmar_ratio_3y", "近3年卡玛比率", "round2"),
    ("max_drawdown_recovery_days", "全期最长回撤修复天数", "int"),
    ("max_drawdown_recovery_days_1y", "近1年最长回撤修复天数", "int"),
    ("max_drawdown_recovery_days_3y", "近3年最长回撤修复天数", "int"),
    ("max_single_day_drop", "全期最大单日跌幅", "percent2"),
    ("max_single_day_drop_1y", "近1年最大单日跌幅", "percent2"),
    ("max_single_day_drop_3y", "近3年最大单日跌幅", "percent2"),
    ("fund_type", "基金类型", "raw"),
    ("subscribe_status", "申购状态", "raw"),
    ("redeem_status", "赎回状态", "raw"),
    ("next_open_day", "下一开放日", "date"),
    ("purchase_min_amount", "购买起点", "round2"),
    ("daily_limit_amount", "日累计限定金额", "round2"),
    ("management_fee_rate", "管理费率", "percent2"),
    ("custodian_fee_rate", "托管费率", "percent2"),
    ("sales_service_fee_rate", "销售服务费率", "percent2"),
    ("max_subscribe_fee_rate", "最高认购费率", "percent2"),
    ("fee_source_max_subscribe", "最高认购费率来源", "raw"),
    ("purchase_fee_rate", "手续费", "percent2"),
    ("fee_source_purchase", "手续费来源", "raw"),
    ("last_updated_date", "最近更新日期", "date"),
    ("last_personnel_change_date", "最近人事变动日期", "date"),
]


@dataclass
class DbConfig:
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_db: str
    clickhouse_host: str
    clickhouse_port: int
    clickhouse_user: str
    clickhouse_password: str
    clickhouse_db: str


@dataclass(frozen=True)
class ClickHouseWriteConfig:
    profile: str
    wait_mutation_timeout_sec: int
    wait_mutation_poll_sec: float
    nav_chunk_rows: int
    period_chunk_rows: int
    max_partition_groups_per_insert: int
    nav_sleep_between_chunks_sec: float
    insert_workers: int


def _resolve_clickhouse_write_config(
    profile: str,
    target_fund_count: int,
    nav_row_count: int,
    period_row_count: int,
    small_data_threshold_funds: int,
) -> ClickHouseWriteConfig:
    selected = profile
    if profile == "auto":
        is_small = (
            target_fund_count <= small_data_threshold_funds
            and nav_row_count <= 200_000
            and period_row_count <= 200_000
        )
        selected = "fast" if is_small else "safe"

    if selected == "fast":
        return ClickHouseWriteConfig(
            profile="fast",
            wait_mutation_timeout_sec=0,
            wait_mutation_poll_sec=1.0,
            nav_chunk_rows=0,
            period_chunk_rows=0,
            max_partition_groups_per_insert=16,
            nav_sleep_between_chunks_sec=0.0,
            insert_workers=4,
        )

    return ClickHouseWriteConfig(
        profile="safe",
        wait_mutation_timeout_sec=120,
        wait_mutation_poll_sec=5.0,
        nav_chunk_rows=5_000,
        period_chunk_rows=5_000,
        max_partition_groups_per_insert=1,
        nav_sleep_between_chunks_sec=0.5,
        insert_workers=1,
    )


def _resolve_clickhouse_scope_tables(scope: str) -> list[str]:
    if scope == "verify_minimal":
        return ["fact_fund_nav_daily", "fact_fund_scoreboard_snapshot"]
    return [
        "fact_fund_nav_daily",
        "fact_fund_return_period",
        "fact_fund_metrics_snapshot",
        "fact_fund_scoreboard_snapshot",
    ]


def _format_value(value: object, style: str) -> object:
    if pd.isna(value):
        return None
    if style == "raw":
        return value
    if style == "date":
        dt = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(dt) else dt.strftime("%Y-%m-%d")
    if style == "int":
        return int(round(float(value)))
    if style == "round2":
        return round(float(value), 2)
    if style == "percent2":
        return round(float(value) * 100.0, 2)
    if style == "percent0":
        return int(round(float(value) * 100.0))
    return value


def _build_scoreboard_export_df(scoreboard: pd.DataFrame) -> pd.DataFrame:
    export_df = pd.DataFrame()
    for src, target, style in EXPORT_COLUMN_SPECS:
        series = scoreboard[src] if src in scoreboard.columns else pd.Series([None] * len(scoreboard))
        export_df[target] = series.map(lambda v: _format_value(v, style))
    return export_df


def _safe_code(value: object) -> str:
    return str(value).strip().zfill(6)


def _extract_first_float(text: object) -> float | None:
    if text is None:
        return None
    s = str(text)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _parse_rate_to_ratio(text: object) -> float | None:
    if text is None:
        return None
    s = str(text).strip()
    if s == "" or s == "---":
        return None
    val = _extract_first_float(s)
    if val is None:
        return None
    return val / 100.0


# MySQL DECIMAL(20,6) 最大值；Python float 无法精确表示 99999999999999.999999 会变成 1e14 导致溢出
_DECIMAL_20_6_MAX = 99999999999999.99


def _clamp_decimal_20_6(value: float | None) -> float | None:
    """将数值截断到 MySQL DECIMAL(20,6) 范围内，避免 float 精度导致入库溢出。"""
    if value is None or (isinstance(value, float) and (value != value or value == float("inf") or value == float("-inf"))):
        return None
    try:
        v = float(value)
        if v > _DECIMAL_20_6_MAX:
            return _DECIMAL_20_6_MAX
        if v < -_DECIMAL_20_6_MAX:
            return -_DECIMAL_20_6_MAX
        return v
    except (TypeError, ValueError):
        return None


def _parse_number(text: object) -> float | None:
    if text is None:
        return None
    s = str(text).strip()
    if s == "" or s == "---":
        return None
    return _extract_first_float(s)


def _parse_date(text: object) -> pd.Timestamp | pd.NaT:
    if text is None:
        return pd.NaT
    s = str(text).strip()
    if s == "" or s == "---":
        return pd.NaT
    date_match = re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", s)
    if date_match:
        return pd.to_datetime(date_match.group(0), errors="coerce")
    zh_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", s)
    if zh_match:
        y, m, d = zh_match.groups()
        return pd.to_datetime(f"{int(y):04d}-{int(m):02d}-{int(d):02d}", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _load_one_personnel(
    path: Path,
    latest_allowed_date: pd.Timestamp | None = None,
) -> tuple[str, pd.Timestamp | pd.NaT]:
    """单文件读取，供并发调用。返回 (code, latest_date)。"""
    code = _safe_code(path.stem)
    if not path.exists():
        return (code, pd.NaT)
    try:
        df = pd.read_csv(path, dtype={"基金代码": str})
    except Exception:
        return (code, pd.NaT)
    if "公告日期" not in df.columns or df.empty:
        return (code, pd.NaT)
    ds = pd.to_datetime(df["公告日期"], errors="coerce").dropna()
    if latest_allowed_date is not None:
        ds = ds[ds <= latest_allowed_date]
    return (code, ds.max() if not ds.empty else pd.NaT)


def _load_personnel_latest_date(
    personnel_dir: Path,
    target_codes: set[str] | None = None,
    latest_allowed_date: pd.Timestamp | None = None,
) -> dict[str, pd.Timestamp | pd.NaT]:
    if target_codes is not None:
        paths = [personnel_dir / f"{code}.csv" for code in sorted(target_codes)]
    else:
        paths = list(personnel_dir.glob("*.csv"))

    out: dict[str, pd.Timestamp | pd.NaT] = {}
    with ThreadPoolExecutor(max_workers=MAX_CSV_WORKERS) as executor:
        futures = {
            executor.submit(_load_one_personnel, p, latest_allowed_date): p
            for p in paths
        }
        for future in as_completed(futures):
            code, val = future.result()
            out[code] = val
    return out


def _build_dim_base(
    purchase_df: pd.DataFrame,
    overview_df: pd.DataFrame,
    personnel_latest: dict[str, pd.Timestamp | pd.NaT],
    data_version: str,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    purchase = purchase_df.copy()
    overview = overview_df.copy()
    purchase["基金代码"] = purchase["基金代码"].map(_safe_code)
    overview["基金代码"] = overview["基金代码"].map(_safe_code)
    purchase = purchase.drop_duplicates(subset=["基金代码"], keep="first")
    overview = overview.drop_duplicates(subset=["基金代码"], keep="first")

    merged = purchase.merge(overview, on="基金代码", how="inner", suffixes=("_purchase", "_overview"))

    merged["fund_code"] = merged["基金代码"]
    merged["fund_name"] = merged["基金简称_overview"].fillna(merged["基金简称_purchase"])
    merged["fund_full_name"] = merged.get("基金全称", pd.Series(dtype=str))
    merged["fund_type"] = merged.get("基金类型", pd.Series(dtype=str))
    merged["inception_date"] = merged.get("成立日期/规模", pd.Series(dtype=str)).map(_parse_date)
    merged["inception_years"] = (as_of_date - merged["inception_date"]).dt.days / 365.25
    merged["scale_billion"] = merged.get("资产规模", pd.Series(dtype=str)).map(_parse_number).map(_clamp_decimal_20_6)

    merged["subscribe_status"] = merged.get("申购状态", pd.Series(dtype=str))
    merged["redeem_status"] = merged.get("赎回状态", pd.Series(dtype=str))
    merged["next_open_day"] = merged.get("下一开放日", pd.Series(dtype=str)).map(_parse_date)
    merged["purchase_min_amount"] = merged.get("购买起点", pd.Series(dtype=str)).map(_parse_number).map(_clamp_decimal_20_6)
    merged["daily_limit_amount"] = merged.get("日累计限定金额", pd.Series(dtype=str)).map(_parse_number).map(_clamp_decimal_20_6)

    merged["management_fee_rate"] = merged.get("管理费率", pd.Series(dtype=str)).map(_parse_rate_to_ratio)
    merged["custodian_fee_rate"] = merged.get("托管费率", pd.Series(dtype=str)).map(_parse_rate_to_ratio)
    merged["sales_service_fee_rate"] = merged.get("销售服务费率", pd.Series(dtype=str)).map(_parse_rate_to_ratio)
    merged["max_subscribe_fee_rate"] = merged.get("最高认购费率", pd.Series(dtype=str)).map(_parse_rate_to_ratio)
    merged["purchase_fee_rate"] = merged.get("手续费", pd.Series(dtype=str)).map(_parse_rate_to_ratio)

    merged["fee_source_max_subscribe"] = "fund_overview.最高认购费率"
    merged["fee_source_purchase"] = "fund_purchase.手续费"
    merged["last_updated_date"] = as_of_date
    merged["last_personnel_change_date"] = merged["fund_code"].map(personnel_latest)
    merged["data_version"] = data_version
    merged["as_of_date"] = as_of_date

    cols = [
        "data_version",
        "as_of_date",
        "fund_code",
        "fund_name",
        "fund_full_name",
        "fund_type",
        "inception_date",
        "inception_years",
        "scale_billion",
        "subscribe_status",
        "redeem_status",
        "next_open_day",
        "purchase_min_amount",
        "daily_limit_amount",
        "management_fee_rate",
        "custodian_fee_rate",
        "sales_service_fee_rate",
        "max_subscribe_fee_rate",
        "purchase_fee_rate",
        "fee_source_max_subscribe",
        "fee_source_purchase",
        "last_updated_date",
        "last_personnel_change_date",
    ]
    return merged[cols].copy()


def _process_one_nav_formal(
    path: Path,
    as_of_date: pd.Timestamp,
    stale_max_days: int,
    latest_nav_date: pd.Timestamp | None = None,
) -> dict[str, object] | None:
    """正式计算模式：仅计算指标，不构建 nav/period 明细。返回 metric_row 或 None。"""
    if not path.exists():
        return None
    code = _safe_code(path.stem)
    df = _metrics_load_nav(path)
    if latest_nav_date is not None:
        df = df[df["净值日期"] <= latest_nav_date].reset_index(drop=True)
    if df.empty:
        return None

    end_date = df["净值日期"].iloc[-1]
    if (as_of_date - end_date).days > stale_max_days:
        return {
            "fund_code": code,
            "nav_start_date": df["净值日期"].iloc[0],
            "nav_end_date": end_date,
            "stale_nav_excluded": True,
        }

    base = _metrics_compute(df, end_date)
    m3 = _metrics_window(df, end_date, years=3)
    m1 = _metrics_window(df, end_date, years=1)
    metric_row: dict[str, object] = {
        "fund_code": code,
        "nav_start_date": df["净值日期"].iloc[0],
        "nav_end_date": end_date,
        "stale_nav_excluded": False,
        **base,
        **m3,
        **m1,
    }
    for k in METRIC_DIRECTIONS:
        metric_row.setdefault(k, None)
    return metric_row


def _process_one_nav(
    path: Path,
    as_of_date: pd.Timestamp,
    stale_max_days: int,
    latest_nav_date: pd.Timestamp | None = None,
) -> tuple[dict[str, object] | None, pd.DataFrame | None, list[pd.DataFrame]]:
    """单文件处理，供并发调用。返回 (metric_row, nav_part, period_parts)。"""
    if not path.exists():
        return (None, None, [])
    code = _safe_code(path.stem)
    try:
        df = pd.read_csv(path, dtype={"基金代码": str})
    except Exception:
        return (None, None, [])
    if df.empty or "净值日期" not in df.columns or "复权净值" not in df.columns:
        return (None, None, [])
    df["净值日期"] = pd.to_datetime(df["净值日期"], errors="coerce")
    df["复权净值"] = pd.to_numeric(df["复权净值"], errors="coerce")
    df["单位净值"] = pd.to_numeric(df.get("单位净值"), errors="coerce")
    df = df.dropna(subset=["净值日期", "复权净值"]).sort_values("净值日期").reset_index(drop=True)
    if latest_nav_date is not None:
        df = df[df["净值日期"] <= latest_nav_date].reset_index(drop=True)
    if df.empty:
        return (None, None, [])

    end_date = df["净值日期"].iloc[-1]
    if (as_of_date - end_date).days > stale_max_days:
        metric_row = {
            "fund_code": code,
            "nav_start_date": df["净值日期"].iloc[0],
            "nav_end_date": end_date,
            "stale_nav_excluded": True,
        }
        return (metric_row, None, [])

    df["daily_return"] = df["复权净值"].pct_change()
    nav_part = pd.DataFrame(
        {
            "fund_code": code,
            "nav_date": df["净值日期"].values,
            "unit_nav": df["单位净值"].values,
            "adjusted_nav": df["复权净值"].values,
            "daily_return": df["daily_return"].values,
            "latest_nav_date": end_date,
        }
    )
    period_parts: list[pd.DataFrame] = []
    for ptype, freq in [("W", "W-FRI"), ("M", "ME"), ("Q", "QE")]:
        s = df.set_index("净值日期")["复权净值"].resample(freq).last().dropna()
        r = s.pct_change().dropna()
        if r.empty:
            continue
        starts = s.index.to_series().shift(1).reindex(r.index)
        period_parts.append(
            pd.DataFrame(
                {
                    "fund_code": code,
                    "period_type": ptype,
                    "period_key": r.index.strftime("%Y-%m-%d"),
                    "period_start": starts.values,
                    "period_end": r.index.values,
                    "period_return": r.values.astype(float),
                }
            )
        )

    base = _metrics_compute(df, end_date)
    m3 = _metrics_window(df, end_date, years=3)
    m1 = _metrics_window(df, end_date, years=1)
    metric_row = {
        "fund_code": code,
        "nav_start_date": df["净值日期"].iloc[0],
        "nav_end_date": end_date,
        "stale_nav_excluded": False,
        **base,
        **m3,
        **m1,
    }
    for k in METRIC_DIRECTIONS:
        metric_row.setdefault(k, None)
    return (metric_row, nav_part, period_parts)


def _calc_all_metrics(
    nav_dir: Path,
    as_of_date: pd.Timestamp,
    stale_max_days: int,
    code_limit: int | None,
    latest_nav_date: pd.Timestamp | None = None,
    target_codes: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if target_codes is not None:
        files = [nav_dir / f"{code}.csv" for code in sorted(target_codes)]
    else:
        files = sorted(nav_dir.glob("*.csv"))
    if code_limit:
        files = files[:code_limit]

    print(f"[scoreboard] _calc_all_metrics start: {len(files)} files, workers={MAX_CSV_WORKERS}")
    metric_rows: list[dict[str, object]] = []
    nav_parts: list[pd.DataFrame] = []
    period_parts: list[pd.DataFrame] = []
    total = len(files)

    with ThreadPoolExecutor(max_workers=MAX_CSV_WORKERS) as executor:
        futures = [
            executor.submit(_process_one_nav, p, as_of_date, stale_max_days, latest_nav_date) for p in files
        ]
        done = 0
        for future in as_completed(futures):
            mrow, npart, pparts = future.result()
            if mrow is not None:
                metric_rows.append(mrow)
            if npart is not None:
                nav_parts.append(npart)
            period_parts.extend(pparts)
            done += 1
            if done % 50 == 0 or done == total:
                print(f"[scoreboard] _calc_all_metrics progress: {done}/{total} files")

    nav_cols = ["fund_code", "nav_date", "unit_nav", "adjusted_nav", "daily_return", "latest_nav_date"]
    period_cols = ["fund_code", "period_type", "period_key", "period_start", "period_end", "period_return"]
    metric_df = pd.DataFrame(metric_rows)
    nav_df = pd.concat(nav_parts, ignore_index=True) if nav_parts else pd.DataFrame(columns=nav_cols)
    period_df = pd.concat(period_parts, ignore_index=True) if period_parts else pd.DataFrame(columns=period_cols)
    return metric_df, nav_df, period_df


def _calc_all_metrics_formal(
    nav_dir: Path,
    as_of_date: pd.Timestamp,
    stale_max_days: int,
    code_limit: int | None,
    latest_nav_date: pd.Timestamp | None = None,
    target_codes: set[str] | None = None,
) -> pd.DataFrame:
    """正式计算模式：仅计算指标，不构建 nav_df/period_df，显著提速。"""
    if target_codes is not None:
        files = [nav_dir / f"{code}.csv" for code in sorted(target_codes)]
    else:
        files = sorted(nav_dir.glob("*.csv"))
    if code_limit:
        files = files[:code_limit]

    print(f"[scoreboard] _calc_all_metrics_formal start: {len(files)} files, workers={MAX_CSV_WORKERS}")
    metric_rows: list[dict[str, object]] = []
    total = len(files)

    with ThreadPoolExecutor(max_workers=MAX_CSV_WORKERS) as executor:
        futures = [
            executor.submit(_process_one_nav_formal, p, as_of_date, stale_max_days, latest_nav_date) for p in files
        ]
        done = 0
        for future in as_completed(futures):
            mrow = future.result()
            if mrow is not None:
                metric_rows.append(mrow)
            done += 1
            if done % 50 == 0 or done == total:
                print(f"[scoreboard] _calc_all_metrics_formal progress: {done}/{total} files")

    return pd.DataFrame(metric_rows)


def _build_scoreboard(
    dim_base_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    data_version: str,
    as_of_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exclusions: list[dict[str, object]] = []

    # exclude stale nav rows first
    stale_codes = set(metric_df.loc[metric_df["stale_nav_excluded"] == True, "fund_code"].astype(str).tolist())
    for code in stale_codes:
        exclusions.append(
            {
                "data_version": data_version,
                "as_of_date": as_of_date.date(),
                "fund_code": code,
                "reason_code": "stale_nav_date",
                "reason_detail": "latest nav date older than as_of_date-2",
            }
        )

    metric_ok = metric_df[metric_df["stale_nav_excluded"] == False].copy()
    merged = metric_ok.merge(dim_base_df, on="fund_code", how="inner")

    missing_nav_codes = set(dim_base_df["fund_code"]) - set(metric_ok["fund_code"])
    for code in sorted(missing_nav_codes):
        exclusions.append(
            {
                "data_version": data_version,
                "as_of_date": as_of_date.date(),
                "fund_code": code,
                "reason_code": "missing_nav",
                "reason_detail": "no adjusted nav data",
            }
        )

    # any null in performance metrics => exclude（向量化替代 iterrows）
    # 每只基金仅保留一条 metric_null 记录（主键 data_version+fund_code+reason_code 唯一）
    metric_cols = [k for k in METRIC_DIRECTIONS if k in merged.columns]
    has_null = merged[metric_cols].isna().any(axis=1)
    keep_mask = ~has_null
    excluded = merged[has_null]
    if not excluded.empty:
        na_mask = excluded[metric_cols].isna()
        for i in range(len(excluded)):
            row = excluded.iloc[i]
            null_metrics = [metric_cols[j] for j in range(len(metric_cols)) if na_mask.iloc[i, j]]
            exclusions.append(
                {
                    "data_version": data_version,
                    "as_of_date": as_of_date.date(),
                    "fund_code": row["fund_code"],
                    "reason_code": "metric_null",
                    "reason_detail": ", ".join(sorted(null_metrics)),
                }
            )

    kept = merged[keep_mask].copy().reset_index(drop=True)

    summary_counter = Counter(item["reason_code"] for item in exclusions)
    exclusion_summary = pd.DataFrame(
        [
            {
                "data_version": data_version,
                "as_of_date": as_of_date.date(),
                "reason_code": k,
                "fund_count": int(v),
            }
            for k, v in sorted(summary_counter.items())
        ]
    )

    exclusion_detail = pd.DataFrame(exclusions)
    if exclusion_detail.empty:
        exclusion_detail = pd.DataFrame(
            columns=["data_version", "as_of_date", "fund_code", "reason_code", "reason_detail"]
        )

    return kept, exclusion_detail, exclusion_summary


def _to_records(df: pd.DataFrame, cols: list[str]) -> list[list[object]]:
    tmp = df[cols].copy()
    for c in tmp.columns:
        if pd.api.types.is_datetime64_any_dtype(tmp[c]):
            tmp[c] = tmp[c].dt.date
    tmp = tmp.replace({np.nan: None, pd.NaT: None})
    return tmp.values.tolist()


def apply_mysql_schema(db: DbConfig, ddl_path: Path) -> None:
    sql = ddl_path.read_text(encoding="utf-8")
    conn = pymysql.connect(
        host=db.mysql_host,
        port=db.mysql_port,
        user=db.mysql_user,
        password=db.mysql_password,
        autocommit=True,
        charset="utf8mb4",
        local_infile=True,
    )
    try:
        with conn.cursor() as cur:
            for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
                cur.execute(stmt)
    finally:
        conn.close()


def apply_clickhouse_schema(db: DbConfig, ddl_path: Path) -> None:
    sql = ddl_path.read_text(encoding="utf-8")
    for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
        _run_clickhouse_query_via_docker(stmt, "fund_clickhouse")


def _run_clickhouse_query_via_docker(query: str, container_name: str) -> None:
    subprocess.run(
        ["docker", "exec", container_name, "clickhouse-client", "--query", query],
        check=True,
        capture_output=True,
        text=True,
    )


def _query_clickhouse_text_via_docker(query: str, container_name: str) -> str:
    proc = subprocess.run(
        ["docker", "exec", container_name, "clickhouse-client", "--query", query],
        check=True,
        capture_output=True,
        text=True,
    )
    return (proc.stdout or "").strip()


def _wait_clickhouse_mutations(
    container_name: str, database: str, timeout_sec: int = 120, poll_interval_sec: float = 5.0
) -> None:
    """等待指定库下所有 mutation 完成，避免 DELETE 与 INSERT 并发导致 OOM。"""
    if timeout_sec <= 0:
        print(f"[clickhouse] mutation wait skipped for {database} (timeout_sec={timeout_sec})")
        return
    escaped_db = database.replace("\\", "\\\\").replace("'", "\\'")
    deadline = time.perf_counter() + timeout_sec
    while time.perf_counter() < deadline:
        query = (
            f"SELECT count() FROM system.mutations "
            f"WHERE database = '{escaped_db}' AND is_done = 0"
        )
        try:
            text = _query_clickhouse_text_via_docker(query, container_name)
            pending = int(text) if text else 0
        except subprocess.CalledProcessError:
            pending = 1  # 查询失败则假定有 pending，继续等待
        if pending == 0:
            print(f"[clickhouse] mutations done for {database}")
            return
        print(f"[clickhouse] waiting for {pending} mutations in {database}...")
        time.sleep(poll_interval_sec)
    print(f"[clickhouse] mutation wait timeout ({timeout_sec}s), proceeding anyway")


def _insert_clickhouse_csv(
    df: pd.DataFrame,
    table: str,
    columns: list[str],
    container_name: str,
    partition_group_cols: list[str] | None = None,
    chunk_rows: int = 200_000,
    max_partition_groups_per_insert: int = 1,
    sleep_between_chunks_sec: float = 0.0,
    insert_workers: int = 1,
) -> None:
    if df.empty:
        return

    payload_df = df[columns].copy()
    if partition_group_cols:
        key_cols: list[str] = []
        for col in partition_group_cols:
            key_col = f"_partition_key_{col}"
            if col in payload_df.columns and pd.api.types.is_datetime64_any_dtype(payload_df[col]):
                payload_df[key_col] = payload_df[col].dt.strftime("%Y-%m")
            elif col in payload_df.columns:
                payload_df[key_col] = payload_df[col].astype(str)
            else:
                payload_df[key_col] = ""
            key_cols.append(key_col)
        parts = [g.drop(columns=key_cols) for _, g in payload_df.groupby(key_cols, dropna=False)]
    else:
        parts = [payload_df]

    bundled_parts: list[pd.DataFrame] = []
    if partition_group_cols and max_partition_groups_per_insert > 1:
        current_batch: list[pd.DataFrame] = []
        for part in parts:
            current_batch.append(part)
            if len(current_batch) >= max_partition_groups_per_insert:
                bundled_parts.append(pd.concat(current_batch, ignore_index=True))
                current_batch = []
        if current_batch:
            bundled_parts.append(pd.concat(current_batch, ignore_index=True))
    else:
        bundled_parts = parts

    query = f"INSERT INTO {table} ({','.join(columns)}) FORMAT CSV"
    chunk_plan: list[tuple[pd.DataFrame, int, int]] = []
    total_chunks = 0
    for part_df in bundled_parts:
        if part_df.empty:
            continue
        effective_chunk_rows = chunk_rows if chunk_rows > 0 else len(part_df)
        n_chunks = max(1, (len(part_df) + effective_chunk_rows - 1) // effective_chunk_rows) if effective_chunk_rows > 0 else 1
        total_chunks += n_chunks
        for col in part_df.columns:
            if pd.api.types.is_datetime64_any_dtype(part_df[col]):
                part_df[col] = part_df[col].dt.strftime("%Y-%m-%d")
        part_df = part_df.replace({np.nan: None, pd.NaT: None})
        if effective_chunk_rows <= 0:
            effective_chunk_rows = len(part_df)
        for start in range(0, len(part_df), effective_chunk_rows):
            chunk_df = part_df.iloc[start : start + effective_chunk_rows]
            chunk_plan.append((chunk_df, start, start + len(chunk_df)))

    done_counter = 0

    def _insert_one(chunk_df: pd.DataFrame, start: int, end: int) -> None:
        nonlocal done_counter
        csv_text = chunk_df.to_csv(index=False, header=False, na_rep="\\N")
        try:
            subprocess.run(
                ["docker", "exec", "-i", container_name, "clickhouse-client", "--query", query],
                input=csv_text,
                text=True,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as err:
            stderr = (err.stderr or "").strip()
            snippet = stderr[-1000:] if stderr else "<empty stderr>"
            raise RuntimeError(
                f"ClickHouse insert failed for {table}, rows={len(chunk_df)}, "
                f"chunk_range=[{start},{end}), stderr={snippet}"
            ) from err
        done_counter += 1
        if total_chunks > 50 and done_counter % 100 == 0:
            print(f"[clickhouse] {table} insert progress: {done_counter}/{total_chunks} chunks")
        if sleep_between_chunks_sec > 0:
            time.sleep(sleep_between_chunks_sec)

    workers = max(1, int(insert_workers))
    if workers == 1 or len(chunk_plan) <= 1:
        for chunk_df, start, end in chunk_plan:
            _insert_one(chunk_df, start, end)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_insert_one, cdf, s, e) for cdf, s, e in chunk_plan]
            for future in as_completed(futures):
                future.result()


def _get_checkpoint_dir(output_dir: Path, data_version: str) -> Path:
    """返回计算阶段 checkpoint 目录。"""
    return output_dir / ".checkpoint" / data_version


def _checkpoint_complete(checkpoint_dir: Path) -> bool:
    """检查 checkpoint 是否完整（所有必需文件存在）。"""
    if not checkpoint_dir.exists():
        return False
    for key in CHECKPOINT_KEYS:
        if not (checkpoint_dir / f"{key}.pkl").exists():
            return False
    return True


def _save_checkpoint(
    checkpoint_dir: Path,
    dim_base: pd.DataFrame,
    metric_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    period_df: pd.DataFrame,
    scoreboard: pd.DataFrame,
    exclusion_detail: pd.DataFrame,
    exclusion_summary: pd.DataFrame,
) -> None:
    """将计算阶段产物写入 checkpoint 目录（pickle 格式）。"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "dim_base": dim_base,
        "metric_df": metric_df,
        "nav_df": nav_df,
        "period_df": period_df,
        "scoreboard": scoreboard,
        "exclusion_detail": exclusion_detail,
        "exclusion_summary": exclusion_summary,
    }
    for key, df in data.items():
        path = checkpoint_dir / f"{key}.pkl"
        with open(path, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[scoreboard] checkpoint saved: {checkpoint_dir}")


def _load_checkpoint(checkpoint_dir: Path) -> dict[str, pd.DataFrame]:
    """从 checkpoint 目录加载计算阶段产物。"""
    data: dict[str, pd.DataFrame] = {}
    for key in CHECKPOINT_KEYS:
        path = checkpoint_dir / f"{key}.pkl"
        with open(path, "rb") as f:
            data[key] = pickle.load(f)
    print(f"[scoreboard] checkpoint loaded: {checkpoint_dir}")
    return data


def run_pipeline(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    as_of = pd.to_datetime(args.as_of_date)
    latest_nav_date = pd.to_datetime(args.latest_nav_date) if args.latest_nav_date else None
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_stage_or_raise(
        "scoreboard_input",
        purchase_csv=args.purchase_csv,
        overview_csv=args.overview_csv,
        personnel_dir=args.personnel_dir,
        nav_dir=args.nav_dir,
    )
    checkpoint_dir = _get_checkpoint_dir(output_dir, args.data_version)

    # --formal-only: 不兼容 --resume（正式计算快速路径，无需断点续传）
    formal_only = getattr(args, "formal_only", False)
    if formal_only:
        args.skip_sinks = True
    if latest_nav_date is not None and args.resume:
        print("[scoreboard] --latest-nav-date is set, --resume disabled for this run")
        args.resume = False

    # --resume: 若 checkpoint 完整则加载，跳过 personnel/dim_base/metrics/scoreboard 计算（formal_only 时不使用）
    if not formal_only and args.resume and _checkpoint_complete(checkpoint_dir):
        t_load = time.perf_counter()
        ck = _load_checkpoint(checkpoint_dir)
        dim_base = ck["dim_base"]
        metric_df = ck["metric_df"]
        nav_df = ck["nav_df"]
        period_df = ck["period_df"]
        scoreboard = ck["scoreboard"]
        exclusion_detail = ck["exclusion_detail"]
        exclusion_summary = ck["exclusion_summary"]
        print(
            f"[scoreboard] resume from checkpoint: dim_base={len(dim_base)}, metrics={len(metric_df)}, "
            f"nav_rows={len(nav_df)}, period_rows={len(period_df)}, scoreboard={len(scoreboard)}, "
            f"elapsed={time.perf_counter() - t_load:.2f}s"
        )
    else:
        if args.resume and not _checkpoint_complete(checkpoint_dir):
            print(f"[scoreboard] --resume specified but checkpoint incomplete, running full computation")
        purchase_df = pd.read_csv(args.purchase_csv, dtype={"基金代码": str})
        overview_df = pd.read_csv(args.overview_csv, dtype={"基金代码": str})
        target_codes = {_safe_code(code) for code in purchase_df.get("基金代码", pd.Series(dtype=str)).dropna().tolist()}
        print(f"[scoreboard] target_fund_count={len(target_codes)}")

        t_load = time.perf_counter()
        personnel_cutoff_date = latest_nav_date if latest_nav_date is not None else None
        personnel_latest = _load_personnel_latest_date(
            args.personnel_dir,
            target_codes=target_codes,
            latest_allowed_date=personnel_cutoff_date,
        )
        print(f"[scoreboard] personnel_load done: {len(personnel_latest)} codes, elapsed={time.perf_counter() - t_load:.2f}s")

        t_dim = time.perf_counter()
        dim_base = _build_dim_base(
            purchase_df=purchase_df,
            overview_df=overview_df,
            personnel_latest=personnel_latest,
            data_version=args.data_version,
            as_of_date=as_of,
        )
        print(f"[scoreboard] dim_base done: rows={len(dim_base)}, elapsed={time.perf_counter() - t_dim:.2f}s")

        t_metrics = time.perf_counter()
        if formal_only:
            metric_df = _calc_all_metrics_formal(
                nav_dir=args.nav_dir,
                as_of_date=as_of,
                stale_max_days=args.stale_max_days,
                code_limit=args.code_limit,
                latest_nav_date=latest_nav_date,
                target_codes=target_codes,
            )
            nav_cols = ["fund_code", "nav_date", "unit_nav", "adjusted_nav", "daily_return", "latest_nav_date"]
            period_cols = ["fund_code", "period_type", "period_key", "period_start", "period_end", "period_return"]
            nav_df = pd.DataFrame(columns=nav_cols)
            period_df = pd.DataFrame(columns=period_cols)
            print(
                f"[scoreboard] _calc_all_metrics_formal done: metrics={len(metric_df)}, "
                f"elapsed={time.perf_counter() - t_metrics:.2f}s"
            )
        else:
            metric_df, nav_df, period_df = _calc_all_metrics(
                nav_dir=args.nav_dir,
                as_of_date=as_of,
                stale_max_days=args.stale_max_days,
                code_limit=args.code_limit,
                latest_nav_date=latest_nav_date,
                target_codes=target_codes,
            )
            print(
                f"[scoreboard] _calc_all_metrics done: metrics={len(metric_df)}, nav_rows={len(nav_df)}, "
                f"period_rows={len(period_df)}, elapsed={time.perf_counter() - t_metrics:.2f}s"
            )

        t_scoreboard = time.perf_counter()
        scoreboard, exclusion_detail, exclusion_summary = _build_scoreboard(
            dim_base_df=dim_base,
            metric_df=metric_df,
            data_version=args.data_version,
            as_of_date=as_of,
        )
        print(
            f"[scoreboard] _build_scoreboard done: scoreboard={len(scoreboard)}, "
            f"exclusions={len(exclusion_detail)}, elapsed={time.perf_counter() - t_scoreboard:.2f}s"
        )

        # 计算完成后写入 checkpoint，供后续 --resume 使用（formal_only 不写 checkpoint）
        if not formal_only:
            _save_checkpoint(
                checkpoint_dir,
                dim_base=dim_base,
                metric_df=metric_df,
                nav_df=nav_df,
                period_df=period_df,
                scoreboard=scoreboard,
                exclusion_detail=exclusion_detail,
                exclusion_summary=exclusion_summary,
            )

    # 以下流程：CSV 导出 + DB 写入（resume 与正常路径共用）
    output_dir.mkdir(parents=True, exist_ok=True)

    # export csv
    scoreboard_csv = output_dir / f"fund_scoreboard_{args.data_version}.csv"
    exclusion_detail_csv = output_dir / f"fund_exclusion_detail_{args.data_version}.csv"
    exclusion_summary_csv = output_dir / f"fund_exclusion_summary_{args.data_version}.csv"

    t_csv = time.perf_counter()
    scoreboard_export_df = _build_scoreboard_export_df(scoreboard)
    scoreboard_export_df.to_csv(scoreboard_csv, index=False, encoding="utf-8-sig")
    exclusion_detail.to_csv(exclusion_detail_csv, index=False, encoding="utf-8-sig")
    exclusion_summary.to_csv(exclusion_summary_csv, index=False, encoding="utf-8-sig")
    validate_stage_or_raise("scoreboard_output", scoreboard_csv=scoreboard_csv)
    print(f"[scoreboard] CSV export done: elapsed={time.perf_counter() - t_csv:.2f}s")

    if not args.skip_sinks:
        db = DbConfig(
            mysql_host=args.mysql_host,
            mysql_port=args.mysql_port,
            mysql_user=args.mysql_user,
            mysql_password=args.mysql_password,
            mysql_db=args.mysql_db,
            clickhouse_host=args.clickhouse_host,
            clickhouse_port=args.clickhouse_port,
            clickhouse_user=args.clickhouse_user,
            clickhouse_password=args.clickhouse_password,
            clickhouse_db=args.clickhouse_db,
        )

        if args.apply_ddl:
            t_ddl = time.perf_counter()
            apply_mysql_schema(db, args.mysql_ddl)
            apply_clickhouse_schema(db, args.clickhouse_ddl)
            print(f"[scoreboard] DDL apply done: elapsed={time.perf_counter() - t_ddl:.2f}s")

        # write mysql
        t_mysql = time.perf_counter()
        mysql_conn = pymysql.connect(
            host=db.mysql_host,
            port=db.mysql_port,
            user=db.mysql_user,
            password=db.mysql_password,
            database=db.mysql_db,
            autocommit=True,
            charset="utf8mb4",
            local_infile=True,
        )
        try:
            with mysql_conn.cursor() as cur:
                cur.execute("DELETE FROM dim_fund_base WHERE data_version=%s", (args.data_version,))
                cur.execute("DELETE FROM dim_fund_exclusion_detail WHERE data_version=%s", (args.data_version,))
                cur.execute("DELETE FROM dim_fund_exclusion_summary WHERE data_version=%s", (args.data_version,))

                base_cols = [
                    "data_version",
                    "as_of_date",
                    "fund_code",
                    "fund_name",
                    "fund_full_name",
                    "fund_type",
                    "inception_date",
                    "inception_years",
                    "scale_billion",
                    "subscribe_status",
                    "redeem_status",
                    "next_open_day",
                    "purchase_min_amount",
                    "daily_limit_amount",
                    "management_fee_rate",
                    "custodian_fee_rate",
                    "sales_service_fee_rate",
                    "max_subscribe_fee_rate",
                    "purchase_fee_rate",
                    "fee_source_max_subscribe",
                    "fee_source_purchase",
                    "last_updated_date",
                    "last_personnel_change_date",
                ]
                if not dim_base.empty:
                    sql = (
                        "INSERT INTO dim_fund_base ("
                        + ",".join(base_cols)
                        + ") VALUES ("
                        + ",".join(["%s"] * len(base_cols))
                        + ")"
                    )
                    cur.executemany(sql, _to_records(dim_base, base_cols))

                if not exclusion_detail.empty:
                    # 按主键去重，保证同一 run_id 重复执行幂等（INSERT 前已 DELETE）
                    detail_dedup = exclusion_detail.drop_duplicates(
                        subset=["data_version", "fund_code", "reason_code"], keep="first"
                    )
                    detail_cols = ["data_version", "as_of_date", "fund_code", "reason_code", "reason_detail"]
                    sql = (
                        "INSERT INTO dim_fund_exclusion_detail ("
                        + ",".join(detail_cols)
                        + ") VALUES ("
                        + ",".join(["%s"] * len(detail_cols))
                        + ")"
                    )
                    cur.executemany(sql, _to_records(detail_dedup, detail_cols))

                if not exclusion_summary.empty:
                    summary_cols = ["data_version", "as_of_date", "reason_code", "fund_count"]
                    sql = (
                        "INSERT INTO dim_fund_exclusion_summary ("
                        + ",".join(summary_cols)
                        + ") VALUES ("
                        + ",".join(["%s"] * len(summary_cols))
                        + ")"
                    )
                    cur.executemany(sql, _to_records(exclusion_summary, summary_cols))
        finally:
            mysql_conn.close()
        print(f"[scoreboard] MySQL write done: elapsed={time.perf_counter() - t_mysql:.2f}s")

        # write clickhouse
        t_clickhouse = time.perf_counter()
        ch_write_cfg = _resolve_clickhouse_write_config(
            profile=args.clickhouse_write_profile,
            target_fund_count=len(dim_base),
            nav_row_count=len(nav_df),
            period_row_count=len(period_df),
            small_data_threshold_funds=args.small_data_threshold_funds,
        )
        ch_tables_to_write = _resolve_clickhouse_scope_tables(args.clickhouse_write_scope)
        print(
            "[clickhouse] write profile="
            f"{ch_write_cfg.profile} "
            f"(requested={args.clickhouse_write_profile}, funds={len(dim_base)}, "
            f"nav_rows={len(nav_df)}, period_rows={len(period_df)})"
        )
        print(f"[clickhouse] write scope={args.clickhouse_write_scope}, tables={','.join(ch_tables_to_write)}")
        escaped_data_version = args.data_version.replace("\\", "\\\\").replace("'", "\\'")
        for table in ch_tables_to_write:
            full_table = f"{db.clickhouse_db}.{table}"
            count_text = _query_clickhouse_text_via_docker(
                f"SELECT count() FROM {full_table} WHERE data_version = '{escaped_data_version}'",
                args.clickhouse_container,
            )
            existing = int(count_text) if count_text else 0
            if existing > 0:
                print(f"[clickhouse] cleanup table={full_table} existing_rows={existing}")
                _run_clickhouse_query_via_docker(
                    f"ALTER TABLE {full_table} DELETE WHERE data_version = '{escaped_data_version}'",
                    args.clickhouse_container,
                )
            else:
                print(f"[clickhouse] cleanup skipped table={full_table} (no existing rows)")

        # DELETE 触发 mutation，需等待完成后再 INSERT，否则易 OOM
        _wait_clickhouse_mutations(
            args.clickhouse_container,
            args.clickhouse_db,
            timeout_sec=ch_write_cfg.wait_mutation_timeout_sec,
            poll_interval_sec=ch_write_cfg.wait_mutation_poll_sec,
        )

        if "fact_fund_nav_daily" in ch_tables_to_write and not nav_df.empty:
            nav_df = nav_df.copy()
            nav_df.insert(0, "data_version", args.data_version)
            nav_df.insert(1, "as_of_date", as_of.date())
            _insert_clickhouse_csv(
                nav_df,
                f"{db.clickhouse_db}.fact_fund_nav_daily",
                [
                    "data_version",
                    "as_of_date",
                    "fund_code",
                    "nav_date",
                    "unit_nav",
                    "adjusted_nav",
                    "daily_return",
                    "latest_nav_date",
                ],
                args.clickhouse_container,
                partition_group_cols=["nav_date"],
                chunk_rows=ch_write_cfg.nav_chunk_rows,
                max_partition_groups_per_insert=ch_write_cfg.max_partition_groups_per_insert,
                sleep_between_chunks_sec=ch_write_cfg.nav_sleep_between_chunks_sec,
                insert_workers=ch_write_cfg.insert_workers,
            )

        if "fact_fund_return_period" in ch_tables_to_write and not period_df.empty:
            period_df = period_df.copy()
            period_df.insert(0, "data_version", args.data_version)
            period_df.insert(1, "as_of_date", as_of.date())
            _insert_clickhouse_csv(
                period_df,
                f"{db.clickhouse_db}.fact_fund_return_period",
                [
                    "data_version",
                    "as_of_date",
                    "fund_code",
                    "period_type",
                    "period_key",
                    "period_start",
                    "period_end",
                    "period_return",
                ],
                args.clickhouse_container,
                partition_group_cols=["period_type", "period_end"],
                chunk_rows=ch_write_cfg.period_chunk_rows,
                max_partition_groups_per_insert=ch_write_cfg.max_partition_groups_per_insert,
                insert_workers=ch_write_cfg.insert_workers,
            )

        if "fact_fund_metrics_snapshot" in ch_tables_to_write and not metric_df.empty:
            metric_insert = metric_df[metric_df["stale_nav_excluded"] == False].copy()
            metric_insert = metric_insert.drop(columns=["stale_nav_excluded"])
            metric_insert.insert(0, "data_version", args.data_version)
            metric_insert.insert(1, "as_of_date", as_of.date())
            for col in CH_METRICS_COLS:
                if col not in metric_insert.columns:
                    metric_insert[col] = None
            _insert_clickhouse_csv(
                metric_insert,
                f"{db.clickhouse_db}.fact_fund_metrics_snapshot",
                CH_METRICS_COLS,
                args.clickhouse_container,
                insert_workers=ch_write_cfg.insert_workers,
            )

        if "fact_fund_scoreboard_snapshot" in ch_tables_to_write and not scoreboard.empty:
            for col in CH_SCOREBOARD_COLS:
                if col not in scoreboard.columns:
                    scoreboard[col] = None
            _insert_clickhouse_csv(
                scoreboard,
                f"{db.clickhouse_db}.fact_fund_scoreboard_snapshot",
                CH_SCOREBOARD_COLS,
                args.clickhouse_container,
                insert_workers=ch_write_cfg.insert_workers,
            )
        print(f"[scoreboard] ClickHouse write done: elapsed={time.perf_counter() - t_clickhouse:.2f}s")
    else:
        print("skip_sinks=True: only CSV artifacts generated, DB writes skipped")

    print(f"scoreboard_rows={len(scoreboard)}")
    print(f"exclusion_detail_rows={len(exclusion_detail)}")
    print(f"exclusion_summary_rows={len(exclusion_summary)}")
    print(f"scoreboard_csv={scoreboard_csv}")
    print(f"exclusion_detail_csv={exclusion_detail_csv}")
    print(f"exclusion_summary_csv={exclusion_summary_csv}")
    print(f"[scoreboard] elapsed_seconds={time.perf_counter() - t0:.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fund scoreboard pipeline with MySQL/ClickHouse sinks")
    parser.add_argument("--purchase-csv", type=Path, required=True)
    parser.add_argument("--overview-csv", type=Path, required=True)
    parser.add_argument("--personnel-dir", type=Path, required=True)
    parser.add_argument("--nav-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-version", type=str, required=True)
    parser.add_argument("--as-of-date", type=str, required=True)
    parser.add_argument(
        "--latest-nav-date",
        type=str,
        default=None,
        help="可选，按净值日期截断输入数据（仅保留 <= 该日期的净值）后计算指标，格式 YYYY-MM-DD",
    )
    parser.add_argument("--stale-max-days", type=int, default=2)
    parser.add_argument("--code-limit", type=int, default=None)
    parser.add_argument("--skip-sinks", action="store_true", help="Only generate CSV outputs, skip MySQL/ClickHouse writes")
    parser.add_argument(
        "--formal-only",
        action="store_true",
        help="正式计算模式：仅用 Python 计算指标，不构建 nav/period 明细，不写 DB，显著提速",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint: skip personnel/dim_base/metrics/scoreboard if checkpoint exists",
    )

    parser.add_argument("--apply-ddl", action="store_true")
    parser.add_argument("--mysql-ddl", type=Path, default=Path("fund_db_infra/sql/mysql_schema.sql"))
    parser.add_argument("--clickhouse-ddl", type=Path, default=Path("fund_db_infra/sql/clickhouse_schema.sql"))

    parser.add_argument("--mysql-host", type=str, default="127.0.0.1")
    parser.add_argument("--mysql-port", type=int, default=3306)
    parser.add_argument("--mysql-user", type=str, default="root")
    parser.add_argument("--mysql-password", type=str, default="your_strong_password")
    parser.add_argument("--mysql-db", type=str, default="fund_analysis")

    parser.add_argument("--clickhouse-host", type=str, default="127.0.0.1")
    parser.add_argument("--clickhouse-port", type=int, default=8123)
    parser.add_argument("--clickhouse-user", type=str, default="default")
    parser.add_argument("--clickhouse-password", type=str, default="")
    parser.add_argument("--clickhouse-db", type=str, default="fund_analysis")
    parser.add_argument("--clickhouse-container", type=str, default="fund_clickhouse")
    parser.add_argument(
        "--clickhouse-write-profile",
        type=str,
        choices=["auto", "safe", "fast"],
        default="auto",
        help="ClickHouse 写入策略: auto(按数据量自动), safe(保守限速), fast(放开限速)",
    )
    parser.add_argument(
        "--small-data-threshold-funds",
        type=int,
        default=200,
        help="auto 模式下判定小数据量的基金数量阈值",
    )
    parser.add_argument(
        "--clickhouse-write-scope",
        type=str,
        choices=["full", "verify_minimal"],
        default="full",
        help="ClickHouse 写入表范围: full(全部事实表) / verify_minimal(仅 nav_daily + scoreboard)",
    )
    return parser


if __name__ == "__main__":
    run_pipeline(build_parser().parse_args())
