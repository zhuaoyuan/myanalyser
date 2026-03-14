#!/usr/bin/env python3
"""临时脚本：从 scored_result.csv 提取基金指标，基于 fund_etl 原始数据重算并比对。

独立实现验算逻辑，不复用 scoreboard_metrics。

用法：
  cd myanalyser && python tools/verify_scored_result_metrics.py

输出：myanalyser/artifacts/filter_score_run_2/metric_verification_detail.csv
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

RF_ANNUAL = 0.015  # 无风险年化收益率，用于夏普比率


def _safe_code(value: object) -> str:
    return str(value).strip().zfill(6)


def _load_nav_df(nav_csv: Path) -> pd.DataFrame:
    """从单基金 CSV 加载净值 DataFrame。"""
    if not nav_csv.exists():
        return pd.DataFrame(columns=["净值日期", "复权净值"])
    df = pd.read_csv(nav_csv, dtype={"基金代码": str}, encoding="utf-8-sig")
    if "净值日期" not in df.columns or "复权净值" not in df.columns:
        return pd.DataFrame(columns=["净值日期", "复权净值"])
    df["净值日期"] = pd.to_datetime(df["净值日期"], errors="coerce")
    df["复权净值"] = pd.to_numeric(df["复权净值"], errors="coerce")
    return df.dropna(subset=["净值日期", "复权净值"]).sort_values("净值日期").reset_index(drop=True)


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


def _max_drawdown(nav: pd.Series) -> float | None:
    if nav.empty:
        return None
    roll_max = nav.cummax()
    dd = 1.0 - nav / roll_max
    if dd.empty:
        return None
    return float(dd.max())


def _max_drawdown_recovery_days(nav_df: pd.DataFrame) -> float | None:
    """最长回撤修复天数：从回撤谷底到收复前高所需的最长自然日天数。"""
    if nav_df.empty or nav_df.shape[0] < 2:
        return None
    nav = nav_df.set_index("净值日期")["复权净值"].sort_index()
    dates = nav.index
    recovery_days_list: list[int] = []
    i = 0
    while i < len(nav):
        peak_val = float(nav.iloc[i])
        j = i + 1
        trough_val = peak_val
        trough_date = dates[i]
        while j < len(nav) and float(nav.iloc[j]) < peak_val:
            v = float(nav.iloc[j])
            if v < trough_val:
                trough_val = v
                trough_date = dates[j]
            j += 1
        if j < len(nav):
            recovery_days_list.append((dates[j] - trough_date).days)
        i = j if j > i + 1 else i + 1
    return float(max(recovery_days_list)) if recovery_days_list else None


def _max_single_day_drop(nav_df: pd.DataFrame) -> float | None:
    """最大单日跌幅：区间内日收益率的最小值。"""
    if nav_df.empty or nav_df.shape[0] < 2:
        return None
    ret = nav_df["复权净值"].pct_change().dropna()
    if ret.empty:
        return None
    return float(ret.min())


def _compute_metrics(nav_df: pd.DataFrame, end_date: pd.Timestamp) -> dict[str, float | None]:
    """全样本指标：年化、胜率、波动、回撤、最近一个月涨跌幅、最长回撤修复天数、最大单日跌幅。"""
    nav_df = nav_df.sort_values("净值日期").copy()
    w_ret = _period_returns(nav_df, "W-FRI")
    m_ret = _period_returns(nav_df, "ME")
    q_ret = _period_returns(nav_df, "QE")

    annual_return = _annual_return(nav_df)
    max_dd = _max_drawdown(nav_df["复权净值"])

    curr_month_start = pd.Timestamp(end_date.year, end_date.month, 1)
    recent_month_end = curr_month_start - pd.Timedelta(days=1)
    recent_month_start = pd.Timestamp(recent_month_end.year, recent_month_end.month, 1)
    recent_month_df = nav_df[
        (nav_df["净值日期"] >= recent_month_start) & (nav_df["净值日期"] <= recent_month_end)
    ]
    recent_month_return = None
    if recent_month_df.shape[0] >= 2:
        recent_month_return = float(
            recent_month_df["复权净值"].iloc[-1] / recent_month_df["复权净值"].iloc[0] - 1.0
        )

    return {
        "annual_return": annual_return,
        "up_quarter_ratio": _up_ratio(q_ret),
        "up_month_ratio": _up_ratio(m_ret),
        "up_week_ratio": _up_ratio(w_ret),
        "quarter_return_std": _std(q_ret),
        "month_return_std": _std(m_ret),
        "week_return_std": _std(w_ret),
        "max_drawdown": max_dd,
        "recent_month_return": recent_month_return,
        "max_drawdown_recovery_days": _max_drawdown_recovery_days(nav_df),
        "max_single_day_drop": _max_single_day_drop(nav_df),
    }


def _window_metrics(nav_df: pd.DataFrame, end_date: pd.Timestamp, years: int) -> dict[str, float | None]:
    """近 N 年窗口指标。"""
    start = end_date - pd.DateOffset(years=years)
    win = nav_df[nav_df["净值日期"] >= start].copy()
    if win.empty:
        return {}

    w_ret = _period_returns(win, "W-FRI")
    m_ret = _period_returns(win, "ME")
    q_ret = _period_returns(win, "QE")
    prefix = f"{years}y"

    annual = _annual_return(win)
    max_dd = _max_drawdown(win["复权净值"])
    sharpe = None
    if w_ret.shape[0] > 1:
        weekly_mean = float(w_ret.mean())
        weekly_std = float(w_ret.std(ddof=1))
        if weekly_std > 0:
            sharpe = ((weekly_mean * 52.0) - RF_ANNUAL) / (weekly_std * math.sqrt(52.0))
    calmar = None
    if annual is not None and max_dd is not None and max_dd > 0:
        calmar = annual / max_dd

    out: dict[str, float | None] = {
        f"annual_return_{prefix}": annual,
        f"up_month_ratio_{prefix}": _up_ratio(m_ret),
        f"up_week_ratio_{prefix}": _up_ratio(w_ret),
        f"month_return_std_{prefix}": _std(m_ret),
        f"week_return_std_{prefix}": _std(w_ret),
        f"max_drawdown_{prefix}": max_dd,
        f"sharpe_ratio_{prefix}": sharpe,
        f"calmar_ratio_{prefix}": calmar,
        f"max_drawdown_recovery_days_{prefix}": _max_drawdown_recovery_days(win),
        f"max_single_day_drop_{prefix}": _max_single_day_drop(win),
    }
    if years == 3:
        out.update({"up_quarter_ratio_3y": _up_ratio(q_ret), "quarter_return_std_3y": _std(q_ret)})
    return out


# 指标定义：(中文字段名, 内部字段名, 展示格式)
# 与 pipeline_scoreboard.EXPORT_COLUMN_SPECS / verify_scoreboard_recalc 对齐
VERIFY_FIELDS: list[tuple[str, str, str]] = [
    ("规模-亿元", "scale_billion", "round2"),
    ("成立年数", "inception_years", "round2"),
    ("年化收益率", "annual_return", "percent2"),
    ("上涨季度比例", "up_quarter_ratio", "percent0"),
    ("上涨月份比例", "up_month_ratio", "percent0"),
    ("上涨星期比例", "up_week_ratio", "percent0"),
    ("季涨跌幅标准差", "quarter_return_std", "percent2"),
    ("月涨跌幅标准差", "month_return_std", "percent2"),
    ("周涨跌幅标准差", "week_return_std", "percent2"),
    ("最大回撤率", "max_drawdown", "percent2"),
    ("近3年年化收益率", "annual_return_3y", "percent2"),
    ("近3年上涨季度比例", "up_quarter_ratio_3y", "percent0"),
    ("近3年上涨月份比例", "up_month_ratio_3y", "percent0"),
    ("近3年上涨星期比例", "up_week_ratio_3y", "percent0"),
    ("近3年季涨跌幅标准差", "quarter_return_std_3y", "percent2"),
    ("近3年月涨跌幅标准差", "month_return_std_3y", "percent2"),
    ("近3年周涨跌幅标准差", "week_return_std_3y", "percent2"),
    ("近3年最大回撤率", "max_drawdown_3y", "percent2"),
    ("近1年年化收益率", "annual_return_1y", "percent2"),
    ("近1年上涨月份比例", "up_month_ratio_1y", "percent0"),
    ("近1年上涨星期比例", "up_week_ratio_1y", "percent0"),
    ("近1年月涨跌幅标准差", "month_return_std_1y", "percent2"),
    ("近1年周涨跌幅标准差", "week_return_std_1y", "percent2"),
    ("近1年最大回撤率", "max_drawdown_1y", "percent2"),
    ("最近一个月涨跌幅", "recent_month_return", "percent2"),
    ("近1年夏普比率", "sharpe_ratio_1y", "round2"),
    ("近3年夏普比率", "sharpe_ratio_3y", "round2"),
    ("近1年卡玛比率", "calmar_ratio_1y", "round2"),
    ("近3年卡玛比率", "calmar_ratio_3y", "round2"),
    ("全期最长回撤修复天数", "max_drawdown_recovery_days", "int"),
    ("近1年最长回撤修复天数", "max_drawdown_recovery_days_1y", "int"),
    ("近3年最长回撤修复天数", "max_drawdown_recovery_days_3y", "int"),
    ("全期最大单日跌幅", "max_single_day_drop", "percent2"),
    ("近1年最大单日跌幅", "max_single_day_drop_1y", "percent2"),
    ("近3年最大单日跌幅", "max_single_day_drop_3y", "percent2"),
]


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


def _parse_number(text: object) -> float | None:
    if text is None:
        return None
    s = str(text).strip()
    if s == "" or s == "---":
        return None
    return _extract_first_float(s)


def _parse_date(text: object) -> pd.Timestamp | None:
    if text is None:
        return None
    s = str(text).strip()
    if s == "" or s == "---":
        return None
    date_match = re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", s)
    if date_match:
        dt = pd.to_datetime(date_match.group(0), errors="coerce")
        return None if pd.isna(dt) else dt
    zh_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", s)
    if zh_match:
        y, m, d = zh_match.groups()
        dt = pd.to_datetime(f"{int(y):04d}-{int(m):02d}-{int(d):02d}", errors="coerce")
        return None if pd.isna(dt) else dt
    return None


def _to_display_value(value: object, style: str) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    if style == "int":
        return int(round(float(value)))
    if style == "round2":
        return round(float(value), 2)
    if style == "percent2":
        return round(float(value) * 100.0, 2)
    if style == "percent0":
        return int(round(float(value) * 100.0))
    return float(value)


def _parse_original_value(value: object, style: str) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    if style in {"int", "percent0"}:
        return int(round(float(text)))
    return round(float(text), 2)


def _is_equal(left: int | float | None, right: int | float | None, style: str) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if style in {"int", "percent0"}:
        return int(left) == int(right)
    return abs(float(left) - float(right)) <= 1e-6


def _load_overview_scale_and_inception(overview_csv: Path) -> dict[str, tuple[float | None, float | None]]:
    """加载 fund_overview.csv，返回 {code: (scale_billion, inception_date)}"""
    if not overview_csv.exists():
        return {}
    df = pd.read_csv(overview_csv, dtype={"基金代码": str}, encoding="utf-8-sig")
    df["基金代码"] = df["基金代码"].map(_safe_code)
    out: dict[str, tuple[float | None, float | None]] = {}
    for _, row in df.iterrows():
        code = row["基金代码"]
        scale = _parse_number(row.get("资产规模"))
        inc_str = row.get("成立日期/规模")
        inc_dt = _parse_date(inc_str)
        out[code] = (scale, inc_dt)
    return out


def run(
    scored_csv: Path,
    fund_etl_dir: Path,
    output_csv: Path,
) -> None:
    nav_dir = fund_etl_dir / "fund_adjusted_nav_by_code"
    overview_csv = fund_etl_dir / "fund_overview.csv"

    df = pd.read_csv(scored_csv, dtype={"基金代码": str}, encoding="utf-8-sig")
    df["基金代码"] = df["基金代码"].map(_safe_code)

    # 期末日期作为 end_date
    df["期末日期"] = pd.to_datetime(df["期末日期"], errors="coerce")

    overview_data = _load_overview_scale_and_inception(overview_csv)

    detail_rows: list[dict] = []

    for _, row in df.iterrows():
        code = row["基金代码"]
        fund_name = row.get("基金名称", "")
        end_date = row["期末日期"]

        recalc: dict[str, object] = {}

        # 规模、成立年数：从 fund_overview
        scale, inception_dt = overview_data.get(code, (None, None))
        if scale is not None:
            recalc["scale_billion"] = scale
        if inception_dt is not None and not pd.isna(end_date):
            recalc["inception_years"] = (end_date - pd.Timestamp(inception_dt)).days / 365.25

        # NAV 相关指标：从 fund_adjusted_nav_by_code，独立验算
        nav_path = nav_dir / f"{code}.csv"
        nav_df = _load_nav_df(nav_path)
        if not nav_df.empty:
            nav_df = nav_df[nav_df["净值日期"] <= end_date].reset_index(drop=True)
        if not nav_df.empty:
            actual_end = nav_df["净值日期"].iloc[-1]
            recalc.update(_compute_metrics(nav_df=nav_df, end_date=actual_end))
            recalc.update(_window_metrics(nav_df=nav_df, end_date=actual_end, years=3))
            recalc.update(_window_metrics(nav_df=nav_df, end_date=actual_end, years=1))

        for cn_name, internal_name, style in VERIFY_FIELDS:
            orig_raw = row.get(cn_name)
            orig_display = _parse_original_value(orig_raw, style=style)
            recalc_raw = recalc.get(internal_name)
            recalc_display = _to_display_value(recalc_raw, style=style)
            passed = _is_equal(orig_display, recalc_display, style=style)

            detail_rows.append(
                {
                    "基金代码": code,
                    "基金名称": fund_name,
                    "指标名称": cn_name,
                    "原结果": orig_display,
                    "验算结果": recalc_display,
                    "是否一致": "是" if passed else "否",
                }
            )

    out_df = pd.DataFrame(
        detail_rows,
        columns=["基金代码", "基金名称", "指标名称", "原结果", "验算结果", "是否一致"],
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"已写入: {output_csv}")
    print(f"总比对行数: {len(out_df)}")
    mismatch = out_df[out_df["是否一致"] == "否"]
    if not mismatch.empty:
        print(f"不一致数量: {len(mismatch)}")
        print(mismatch.groupby("指标名称").size().to_string())


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    scored_csv = base / "artifacts" / "filter_score_run_2" / "scored_result.csv"
    fund_etl_dir = base / "data" / "versions" / "20260228_1_formal" / "fund_etl"
    output_csv = base / "artifacts" / "filter_score_run_2" / "metric_verification_detail.csv"

    run(
        scored_csv=scored_csv,
        fund_etl_dir=fund_etl_dir,
        output_csv=output_csv,
    )


if __name__ == "__main__":
    main()
