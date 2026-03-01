"""评分榜指标计算共享模块。供 pipeline_scoreboard 正式计算与 verify_scoreboard_recalc 核验共用。"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

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
    "recent_month_return": "desc",
    "sharpe_ratio_1y": "desc",
    "sharpe_ratio_3y": "desc",
    "calmar_ratio_1y": "desc",
    "calmar_ratio_3y": "desc",
    "max_drawdown_recovery_days": "asc",
    "max_drawdown_recovery_days_1y": "asc",
    "max_drawdown_recovery_days_3y": "asc",
    "max_single_day_drop": "asc",
    "max_single_day_drop_1y": "asc",
    "max_single_day_drop_3y": "asc",
}


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
    """计算最长回撤修复天数：目标时间段内，从回撤谷底到收复前高所需的最长自然日天数。"""
    if nav_df.empty or nav_df.shape[0] < 2:
        return None
    nav = nav_df.set_index("净值日期")["复权净值"].sort_index()
    dates = nav.index
    recovery_days_list: list[int] = []
    i = 0
    while i < len(nav):
        peak_val = float(nav.iloc[i])
        peak_date = dates[i]
        j = i + 1
        trough_val = peak_val
        trough_date = peak_date
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
    """计算最大单日跌幅：区间内日收益率的最小值（最负值）。"""
    if nav_df.empty or nav_df.shape[0] < 2:
        return None
    ret = nav_df["复权净值"].pct_change().dropna()
    if ret.empty:
        return None
    return float(ret.min())


def compute_metrics(nav_df: pd.DataFrame, end_date: pd.Timestamp) -> dict[str, float | None]:
    """计算全样本指标（年化、胜率、波动、回撤、最近一个月涨跌幅、最长回撤修复天数、最大单日跌幅）。"""
    nav_df = nav_df.sort_values("净值日期").copy()
    w_ret = _period_returns(nav_df, "W-FRI")
    m_ret = _period_returns(nav_df, "ME")
    q_ret = _period_returns(nav_df, "QE")

    annual_return = _annual_return(nav_df)
    max_dd = _max_drawdown(nav_df["复权净值"])

    # 最近一个完整自然月的涨跌幅
    curr_month_start = pd.Timestamp(end_date.year, end_date.month, 1)
    recent_month_end = curr_month_start - pd.Timedelta(days=1)
    recent_month_start = pd.Timestamp(recent_month_end.year, recent_month_end.month, 1)
    recent_month_df = nav_df[(nav_df["净值日期"] >= recent_month_start) & (nav_df["净值日期"] <= recent_month_end)]

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


def window_metrics(nav_df: pd.DataFrame, end_date: pd.Timestamp, years: int) -> dict[str, float | None]:
    """计算近 N 年窗口指标。"""
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


def load_nav_df(nav_csv: Path) -> pd.DataFrame:
    """从单基金 CSV 加载净值 DataFrame。"""
    if not nav_csv.exists():
        return pd.DataFrame(columns=["净值日期", "复权净值"])
    df = pd.read_csv(nav_csv, dtype={"基金代码": str}, encoding="utf-8-sig")
    if "净值日期" not in df.columns or "复权净值" not in df.columns:
        return pd.DataFrame(columns=["净值日期", "复权净值"])
    df["净值日期"] = pd.to_datetime(df["净值日期"], errors="coerce")
    df["复权净值"] = pd.to_numeric(df["复权净值"], errors="coerce")
    return df.dropna(subset=["净值日期", "复权净值"]).sort_values("净值日期").reset_index(drop=True)


def safe_code(value: object) -> str:
    """标准化基金代码为 6 位字符串。"""
    return str(value).strip().zfill(6)
