from __future__ import annotations

import argparse
import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql

RF_ANNUAL = 0.015

METRIC_DIRECTIONS = {
    "annual_return": "desc",
    "up_quarter_ratio": "desc",
    "up_month_ratio": "desc",
    "up_week_ratio": "desc",
    "quarter_return_std": "asc",
    "month_return_std": "asc",
    "week_return_std": "asc",
    "max_drawdown": "asc",
    "annual_return_3y": "desc",
    "up_quarter_ratio_3y": "desc",
    "up_month_ratio_3y": "desc",
    "up_week_ratio_3y": "desc",
    "quarter_return_std_3y": "asc",
    "month_return_std_3y": "asc",
    "week_return_std_3y": "asc",
    "max_drawdown_3y": "asc",
    "annual_return_1y": "desc",
    "up_month_ratio_1y": "desc",
    "up_week_ratio_1y": "desc",
    "month_return_std_1y": "asc",
    "week_return_std_1y": "asc",
    "max_drawdown_1y": "asc",
    "prev_month_return": "desc",
    "curr_month_return": "desc",
    "sharpe_ratio": "desc",
    "calmar_ratio": "desc",
}

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
    "prev_month_return",
    "curr_month_return",
    "sharpe_ratio",
    "calmar_ratio",
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
    "prev_month_return",
    "curr_month_return",
    "sharpe_ratio",
    "calmar_ratio",
    "annual_return_rank",
    "up_quarter_ratio_rank",
    "up_month_ratio_rank",
    "up_week_ratio_rank",
    "quarter_return_std_rank",
    "month_return_std_rank",
    "week_return_std_rank",
    "max_drawdown_rank",
    "annual_return_3y_rank",
    "up_quarter_ratio_3y_rank",
    "up_month_ratio_3y_rank",
    "up_week_ratio_3y_rank",
    "quarter_return_std_3y_rank",
    "month_return_std_3y_rank",
    "week_return_std_3y_rank",
    "max_drawdown_3y_rank",
    "annual_return_1y_rank",
    "up_month_ratio_1y_rank",
    "up_week_ratio_1y_rank",
    "month_return_std_1y_rank",
    "week_return_std_1y_rank",
    "max_drawdown_1y_rank",
    "prev_month_return_rank",
    "curr_month_return_rank",
    "sharpe_ratio_rank",
    "calmar_ratio_rank",
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
    ("annual_return_rank", "年化收益率排名", "int"),
    ("up_quarter_ratio", "上涨季度比例", "percent0"),
    ("up_quarter_ratio_rank", "上涨季度比例排名", "int"),
    ("up_month_ratio", "上涨月份比例", "percent0"),
    ("up_month_ratio_rank", "上涨月份比例排名", "int"),
    ("up_week_ratio", "上涨星期比例", "percent0"),
    ("up_week_ratio_rank", "上涨星期比例排名", "int"),
    ("quarter_return_std", "季涨跌幅标准差", "percent2"),
    ("quarter_return_std_rank", "季涨跌幅标准差排名", "int"),
    ("month_return_std", "月涨跌幅标准差", "percent2"),
    ("month_return_std_rank", "月涨跌幅标准差排名", "int"),
    ("week_return_std", "周涨跌幅标准差", "percent2"),
    ("week_return_std_rank", "周涨跌幅标准差排名", "int"),
    ("max_drawdown", "最大回撤率", "percent2"),
    ("max_drawdown_rank", "最大回撤率排名", "int"),
    ("annual_return_3y", "近3年年化收益率", "percent2"),
    ("annual_return_3y_rank", "近3年年化收益率排名", "int"),
    ("up_quarter_ratio_3y", "近3年上涨季度比例", "percent0"),
    ("up_quarter_ratio_3y_rank", "近3年上涨季度比例排名", "int"),
    ("up_month_ratio_3y", "近3年上涨月份比例", "percent0"),
    ("up_month_ratio_3y_rank", "近3年上涨月份比例排名", "int"),
    ("up_week_ratio_3y", "近3年上涨星期比例", "percent0"),
    ("up_week_ratio_3y_rank", "近3年上涨星期比例排名", "int"),
    ("quarter_return_std_3y", "近3年季涨跌幅标准差", "percent2"),
    ("quarter_return_std_3y_rank", "近3年季涨跌幅标准差排名", "int"),
    ("month_return_std_3y", "近3年月涨跌幅标准差", "percent2"),
    ("month_return_std_3y_rank", "近3年月涨跌幅标准差排名", "int"),
    ("week_return_std_3y", "近3年周涨跌幅标准差", "percent2"),
    ("week_return_std_3y_rank", "近3年周涨跌幅标准差排名", "int"),
    ("max_drawdown_3y", "近3年最大回撤率", "percent2"),
    ("max_drawdown_3y_rank", "近3年最大回撤率排名", "int"),
    ("annual_return_1y", "近1年年化收益率", "percent2"),
    ("annual_return_1y_rank", "近1年年化收益率排名", "int"),
    ("up_month_ratio_1y", "近1年上涨月份比例", "percent0"),
    ("up_month_ratio_1y_rank", "近1年上涨月份比例排名", "int"),
    ("up_week_ratio_1y", "近1年上涨星期比例", "percent0"),
    ("up_week_ratio_1y_rank", "近1年上涨星期比例排名", "int"),
    ("month_return_std_1y", "近1年月涨跌幅标准差", "percent2"),
    ("month_return_std_1y_rank", "近1年月涨跌幅标准差排名", "int"),
    ("week_return_std_1y", "近1年周涨跌幅标准差", "percent2"),
    ("week_return_std_1y_rank", "近1年周涨跌幅标准差排名", "int"),
    ("max_drawdown_1y", "近1年最大回撤率", "percent2"),
    ("max_drawdown_1y_rank", "近1年最大回撤率排名", "int"),
    ("prev_month_return", "前1月涨跌幅", "percent2"),
    ("prev_month_return_rank", "前1月涨跌幅排名", "int"),
    ("curr_month_return", "月涨跌幅", "percent2"),
    ("curr_month_return_rank", "月涨跌幅排名", "int"),
    ("sharpe_ratio", "夏普比率", "round2"),
    ("sharpe_ratio_rank", "夏普比率排名", "int"),
    ("calmar_ratio", "卡玛比率", "round2"),
    ("calmar_ratio_rank", "卡玛比率排名", "int"),
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


def _max_drawdown(nav: pd.Series) -> float | None:
    if nav.empty:
        return None
    roll_max = nav.cummax()
    dd = 1.0 - nav / roll_max
    if dd.empty:
        return None
    return float(dd.max())


def _period_returns(nav_df: pd.DataFrame, freq: str) -> pd.Series:
    s = nav_df.set_index("净值日期")["复权净值"].sort_index()
    period_nav = s.resample(freq).last().dropna()
    return period_nav.pct_change().dropna()


def _annual_return(nav_df: pd.DataFrame) -> float | None:
    if nav_df.shape[0] < 2:
        return None
    start_val = float(nav_df["复权净值"].iloc[0])
    end_val = float(nav_df["复权净值"].iloc[-1])
    days = int((nav_df["净值日期"].iloc[-1] - nav_df["净值日期"].iloc[0]).days)
    if start_val <= 0 or end_val <= 0 or days <= 0:
        return None
    return float((end_val / start_val) ** (365.0 / days) - 1.0)


def _up_ratio(returns: pd.Series) -> float | None:
    if returns.empty:
        return None
    return float((returns > 0).mean())


def _std(returns: pd.Series) -> float | None:
    if returns.empty:
        return None
    return float(returns.std(ddof=1)) if returns.shape[0] > 1 else 0.0


def _compute_metrics(nav_df: pd.DataFrame, end_date: pd.Timestamp) -> dict[str, float | None]:
    nav_df = nav_df.sort_values("净值日期").copy()
    w_ret = _period_returns(nav_df, "W-FRI")
    m_ret = _period_returns(nav_df, "ME")
    q_ret = _period_returns(nav_df, "QE")

    annual_return = _annual_return(nav_df)
    max_dd = _max_drawdown(nav_df["复权净值"])

    # last full month and current month MTD
    curr_month_start = pd.Timestamp(end_date.year, end_date.month, 1)
    prev_month_end = curr_month_start - pd.Timedelta(days=1)
    prev_month_start = pd.Timestamp(prev_month_end.year, prev_month_end.month, 1)

    curr_month_df = nav_df[nav_df["净值日期"] >= curr_month_start]
    prev_month_df = nav_df[(nav_df["净值日期"] >= prev_month_start) & (nav_df["净值日期"] <= prev_month_end)]

    curr_month_return = None
    if curr_month_df.shape[0] >= 2:
        curr_month_return = float(curr_month_df["复权净值"].iloc[-1] / curr_month_df["复权净值"].iloc[0] - 1.0)

    prev_month_return = None
    if prev_month_df.shape[0] >= 2:
        prev_month_return = float(prev_month_df["复权净值"].iloc[-1] / prev_month_df["复权净值"].iloc[0] - 1.0)

    sharpe = None
    if w_ret.shape[0] > 1:
        weekly_mean = float(w_ret.mean())
        weekly_std = float(w_ret.std(ddof=1))
        if weekly_std > 0:
            sharpe = ((weekly_mean * 52.0) - RF_ANNUAL) / (weekly_std * math.sqrt(52.0))

    calmar = None
    if annual_return is not None and max_dd is not None and max_dd > 0:
        calmar = annual_return / max_dd

    return {
        "annual_return": annual_return,
        "up_quarter_ratio": _up_ratio(q_ret),
        "up_month_ratio": _up_ratio(m_ret),
        "up_week_ratio": _up_ratio(w_ret),
        "quarter_return_std": _std(q_ret),
        "month_return_std": _std(m_ret),
        "week_return_std": _std(w_ret),
        "max_drawdown": max_dd,
        "prev_month_return": prev_month_return,
        "curr_month_return": curr_month_return,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
    }


def _window_metrics(nav_df: pd.DataFrame, end_date: pd.Timestamp, years: int) -> dict[str, float | None]:
    start = end_date - pd.DateOffset(years=years)
    win = nav_df[nav_df["净值日期"] >= start].copy()
    if win.empty:
        return {}

    w_ret = _period_returns(win, "W-FRI")
    m_ret = _period_returns(win, "ME")
    q_ret = _period_returns(win, "QE")

    prefix = f"{years}y"
    out: dict[str, float | None] = {
        f"annual_return_{prefix}": _annual_return(win),
        f"up_month_ratio_{prefix}": _up_ratio(m_ret),
        f"up_week_ratio_{prefix}": _up_ratio(w_ret),
        f"month_return_std_{prefix}": _std(m_ret),
        f"week_return_std_{prefix}": _std(w_ret),
        f"max_drawdown_{prefix}": _max_drawdown(win["复权净值"]),
    }
    if years == 3:
        out.update(
            {
                "up_quarter_ratio_3y": _up_ratio(q_ret),
                "quarter_return_std_3y": _std(q_ret),
            }
        )
    return out


def _load_personnel_latest_date(personnel_dir: Path) -> dict[str, pd.Timestamp | pd.NaT]:
    out: dict[str, pd.Timestamp | pd.NaT] = {}
    for path in personnel_dir.glob("*.csv"):
        code = _safe_code(path.stem)
        try:
            df = pd.read_csv(path, dtype={"基金代码": str})
        except Exception:
            out[code] = pd.NaT
            continue
        if "公告日期" not in df.columns or df.empty:
            out[code] = pd.NaT
            continue
        ds = pd.to_datetime(df["公告日期"], errors="coerce").dropna()
        out[code] = ds.max() if not ds.empty else pd.NaT
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
    merged["scale_billion"] = merged.get("资产规模", pd.Series(dtype=str)).map(_parse_number)

    merged["subscribe_status"] = merged.get("申购状态", pd.Series(dtype=str))
    merged["redeem_status"] = merged.get("赎回状态", pd.Series(dtype=str))
    merged["next_open_day"] = merged.get("下一开放日", pd.Series(dtype=str)).map(_parse_date)
    merged["purchase_min_amount"] = merged.get("购买起点", pd.Series(dtype=str)).map(_parse_number)
    merged["daily_limit_amount"] = merged.get("日累计限定金额", pd.Series(dtype=str)).map(_parse_number)

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


def _calc_all_metrics(nav_dir: Path, as_of_date: pd.Timestamp, stale_max_days: int, code_limit: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    nav_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []

    files = sorted(nav_dir.glob("*.csv"))
    if code_limit:
        files = files[:code_limit]

    for path in files:
        code = _safe_code(path.stem)
        df = pd.read_csv(path, dtype={"基金代码": str})
        if df.empty or "净值日期" not in df.columns or "复权净值" not in df.columns:
            continue
        df["净值日期"] = pd.to_datetime(df["净值日期"], errors="coerce")
        df["复权净值"] = pd.to_numeric(df["复权净值"], errors="coerce")
        df["单位净值"] = pd.to_numeric(df.get("单位净值"), errors="coerce")
        df = df.dropna(subset=["净值日期", "复权净值"]).sort_values("净值日期").reset_index(drop=True)
        if df.empty:
            continue

        end_date = df["净值日期"].iloc[-1]
        if (as_of_date - end_date).days > stale_max_days:
            metric_rows.append(
                {
                    "fund_code": code,
                    "nav_start_date": df["净值日期"].iloc[0],
                    "nav_end_date": end_date,
                    "stale_nav_excluded": True,
                }
            )
            continue

        df["daily_return"] = df["复权净值"].pct_change()
        for _, row in df.iterrows():
            nav_rows.append(
                {
                    "fund_code": code,
                    "nav_date": row["净值日期"],
                    "unit_nav": row.get("单位净值"),
                    "adjusted_nav": row["复权净值"],
                    "daily_return": row.get("daily_return"),
                    "latest_nav_date": end_date,
                }
            )

        for ptype, freq in [("W", "W-FRI"), ("M", "ME"), ("Q", "QE")]:
            s = df.set_index("净值日期")["复权净值"].resample(freq).last().dropna()
            r = s.pct_change().dropna()
            for idx, val in r.items():
                start_idx = s.index[s.index.get_loc(idx) - 1]
                period_rows.append(
                    {
                        "fund_code": code,
                        "period_type": ptype,
                        "period_key": idx.strftime("%Y-%m-%d"),
                        "period_start": start_idx,
                        "period_end": idx,
                        "period_return": float(val),
                    }
                )

        base = _compute_metrics(df, end_date)
        m3 = _window_metrics(df, end_date, years=3)
        m1 = _window_metrics(df, end_date, years=1)

        metric_row: dict[str, object] = {
            "fund_code": code,
            "nav_start_date": df["净值日期"].iloc[0],
            "nav_end_date": end_date,
            "stale_nav_excluded": False,
            **base,
            **m3,
            **m1,
        }
        # fill missing keys for schema stability
        for k in METRIC_DIRECTIONS:
            metric_row.setdefault(k, None)
        metric_rows.append(metric_row)

    return pd.DataFrame(metric_rows), pd.DataFrame(nav_rows), pd.DataFrame(period_rows)


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

    # any null in performance metrics => exclude
    keep_mask = np.ones(len(merged), dtype=bool)
    for idx, row in merged.iterrows():
        null_metrics = [k for k in METRIC_DIRECTIONS if pd.isna(row.get(k))]
        if null_metrics:
            keep_mask[idx] = False
            for m in null_metrics:
                exclusions.append(
                    {
                        "data_version": data_version,
                        "as_of_date": as_of_date.date(),
                        "fund_code": row["fund_code"],
                        "reason_code": "metric_null",
                        "reason_detail": m,
                    }
                )

    kept = merged[keep_mask].copy().reset_index(drop=True)

    for metric, direction in METRIC_DIRECTIONS.items():
        asc = direction == "asc"
        kept[f"{metric}_rank"] = kept[metric].rank(method="min", ascending=asc).astype("Int64")

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


def _insert_clickhouse_csv(
    df: pd.DataFrame,
    table: str,
    columns: list[str],
    container_name: str,
    partition_group_cols: list[str] | None = None,
) -> None:
    if df.empty:
        return
    payload_df = df[columns].copy()
    if partition_group_cols:
        key_cols: list[str] = []
        for col in partition_group_cols:
            key_col = f"_partition_key_{col}"
            if col in payload_df.columns:
                if pd.api.types.is_datetime64_any_dtype(payload_df[col]):
                    payload_df[key_col] = payload_df[col].dt.strftime("%Y-%m")
                else:
                    as_dt = pd.to_datetime(payload_df[col], errors="coerce")
                    if as_dt.notna().all():
                        payload_df[key_col] = as_dt.dt.strftime("%Y-%m")
                    else:
                        payload_df[key_col] = payload_df[col].astype(str)
            else:
                payload_df[key_col] = ""
            key_cols.append(key_col)
        parts = [g.drop(columns=key_cols) for _, g in payload_df.groupby(key_cols, dropna=False)]
    else:
        parts = [payload_df]

    query = f"INSERT INTO {table} ({','.join(columns)}) FORMAT CSV"
    for part_df in parts:
        for col in part_df.columns:
            if pd.api.types.is_datetime64_any_dtype(part_df[col]):
                part_df[col] = part_df[col].dt.strftime("%Y-%m-%d")
        part_df = part_df.replace({np.nan: None, pd.NaT: None})
        csv_text = part_df.to_csv(index=False, header=False, na_rep="\\N")
        subprocess.run(
            ["docker", "exec", "-i", container_name, "clickhouse-client", "--query", query],
            input=csv_text,
            text=True,
            check=True,
            capture_output=True,
        )


def run_pipeline(args: argparse.Namespace) -> None:
    as_of = pd.to_datetime(args.as_of_date)

    purchase_df = pd.read_csv(args.purchase_csv, dtype={"基金代码": str})
    overview_df = pd.read_csv(args.overview_csv, dtype={"基金代码": str})
    personnel_latest = _load_personnel_latest_date(args.personnel_dir)

    dim_base = _build_dim_base(
        purchase_df=purchase_df,
        overview_df=overview_df,
        personnel_latest=personnel_latest,
        data_version=args.data_version,
        as_of_date=as_of,
    )

    metric_df, nav_df, period_df = _calc_all_metrics(
        nav_dir=args.nav_dir,
        as_of_date=as_of,
        stale_max_days=args.stale_max_days,
        code_limit=args.code_limit,
    )

    scoreboard, exclusion_detail, exclusion_summary = _build_scoreboard(
        dim_base_df=dim_base,
        metric_df=metric_df,
        data_version=args.data_version,
        as_of_date=as_of,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # export csv
    scoreboard_csv = output_dir / f"fund_scoreboard_{args.data_version}.csv"
    exclusion_detail_csv = output_dir / f"fund_exclusion_detail_{args.data_version}.csv"
    exclusion_summary_csv = output_dir / f"fund_exclusion_summary_{args.data_version}.csv"

    scoreboard_export_df = _build_scoreboard_export_df(scoreboard)
    scoreboard_export_df.to_csv(scoreboard_csv, index=False, encoding="utf-8-sig")
    exclusion_detail.to_csv(exclusion_detail_csv, index=False, encoding="utf-8-sig")
    exclusion_summary.to_csv(exclusion_summary_csv, index=False, encoding="utf-8-sig")

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
        apply_mysql_schema(db, args.mysql_ddl)
        apply_clickhouse_schema(db, args.clickhouse_ddl)

    # write mysql
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
                detail_cols = ["data_version", "as_of_date", "fund_code", "reason_code", "reason_detail"]
                sql = (
                    "INSERT INTO dim_fund_exclusion_detail ("
                    + ",".join(detail_cols)
                    + ") VALUES ("
                    + ",".join(["%s"] * len(detail_cols))
                    + ")"
                )
                cur.executemany(sql, _to_records(exclusion_detail, detail_cols))

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

    # write clickhouse
    _run_clickhouse_query_via_docker(
        f"ALTER TABLE {db.clickhouse_db}.fact_fund_nav_daily DELETE WHERE data_version = '{args.data_version}'",
        args.clickhouse_container,
    )
    _run_clickhouse_query_via_docker(
        f"ALTER TABLE {db.clickhouse_db}.fact_fund_return_period DELETE WHERE data_version = '{args.data_version}'",
        args.clickhouse_container,
    )
    _run_clickhouse_query_via_docker(
        f"ALTER TABLE {db.clickhouse_db}.fact_fund_metrics_snapshot DELETE WHERE data_version = '{args.data_version}'",
        args.clickhouse_container,
    )
    _run_clickhouse_query_via_docker(
        f"ALTER TABLE {db.clickhouse_db}.fact_fund_scoreboard_snapshot DELETE WHERE data_version = '{args.data_version}'",
        args.clickhouse_container,
    )

    if not nav_df.empty:
        nav_df = nav_df.copy()
        nav_df.insert(0, "data_version", args.data_version)
        nav_df.insert(1, "as_of_date", as_of.date())
        _insert_clickhouse_csv(
            nav_df,
            f"{db.clickhouse_db}.fact_fund_nav_daily",
            ["data_version", "as_of_date", "fund_code", "nav_date", "unit_nav", "adjusted_nav", "daily_return", "latest_nav_date"],
            args.clickhouse_container,
            partition_group_cols=["nav_date"],
        )

    if not period_df.empty:
        period_df = period_df.copy()
        period_df.insert(0, "data_version", args.data_version)
        period_df.insert(1, "as_of_date", as_of.date())
        _insert_clickhouse_csv(
            period_df,
            f"{db.clickhouse_db}.fact_fund_return_period",
            ["data_version", "as_of_date", "fund_code", "period_type", "period_key", "period_start", "period_end", "period_return"],
            args.clickhouse_container,
            partition_group_cols=["period_type", "period_end"],
        )

    if not metric_df.empty:
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
        )

    if not scoreboard.empty:
        for col in CH_SCOREBOARD_COLS:
            if col not in scoreboard.columns:
                scoreboard[col] = None
        _insert_clickhouse_csv(
            scoreboard,
            f"{db.clickhouse_db}.fact_fund_scoreboard_snapshot",
            CH_SCOREBOARD_COLS,
            args.clickhouse_container,
        )

    print(f"scoreboard_rows={len(scoreboard)}")
    print(f"exclusion_detail_rows={len(exclusion_detail)}")
    print(f"exclusion_summary_rows={len(exclusion_summary)}")
    print(f"scoreboard_csv={scoreboard_csv}")
    print(f"exclusion_detail_csv={exclusion_detail_csv}")
    print(f"exclusion_summary_csv={exclusion_summary_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fund scoreboard pipeline with MySQL/ClickHouse sinks")
    parser.add_argument("--purchase-csv", type=Path, required=True)
    parser.add_argument("--overview-csv", type=Path, required=True)
    parser.add_argument("--personnel-dir", type=Path, required=True)
    parser.add_argument("--nav-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-version", type=str, required=True)
    parser.add_argument("--as-of-date", type=str, required=True)
    parser.add_argument("--stale-max-days", type=int, default=2)
    parser.add_argument("--code-limit", type=int, default=None)

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
    return parser


if __name__ == "__main__":
    run_pipeline(build_parser().parse_args())
