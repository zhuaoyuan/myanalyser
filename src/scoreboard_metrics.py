"""评分榜指标计算共享模块。供 pipeline_scoreboard 正式计算与 verify_scoreboard_recalc 核验共用。

与 Backtest 重叠的指标统一使用 fund_metrics_core 计算逻辑，保证口径一致。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fund_metrics_core import (
    CN_TO_EN_LOW_RISK,
    cagr,
    compute_low_risk_debt_metrics,
    longest_recovery_days,
    max_drawdown as _core_max_drawdown,
    return_over_period,
    WindowConfig,
)

RF_ANNUAL = 0.015
_config = WindowConfig()

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


def _up_ratio(returns: pd.Series) -> float | None:
    if returns.empty:
        return None
    return float((returns > 0).mean())


def _std(returns: pd.Series) -> float | None:
    if returns.empty:
        return None
    return float(returns.std(ddof=1)) if returns.shape[0] > 1 else 0.0


def _max_single_day_drop(nav_df: pd.DataFrame) -> float | None:
    """计算最大单日跌幅：区间内日收益率的最小值（最负值）。"""
    if nav_df.empty or nav_df.shape[0] < 2:
        return None
    ret = nav_df["复权净值"].pct_change().dropna()
    if ret.empty:
        return None
    return float(ret.min())


def compute_metrics(nav_df: pd.DataFrame, end_date: pd.Timestamp) -> dict[str, float | None]:
    """计算全样本指标。重叠指标使用 fund_metrics_core 与 Backtest 一致。"""
    nav_df = nav_df.sort_values("净值日期").copy()
    w_ret = _period_returns(nav_df, "W-FRI")
    m_ret = _period_returns(nav_df, "ME")
    q_ret = _period_returns(nav_df, "QE")

    prices = nav_df["复权净值"].to_numpy(dtype=float)
    dates = nav_df["净值日期"].to_numpy(dtype="datetime64[D]")
    cfg = WindowConfig()

    annual_return = cagr(prices, cfg.trading_days_per_year)
    max_dd = _core_max_drawdown(prices)
    max_drawdown_recovery_days = longest_recovery_days(dates, prices)

    # 最近一个月 = 最近 20 个交易日（与 Backtest 一致，A 股平均月交易日）
    win_1m = _config.trading_days_per_month
    prices_1m = prices[-win_1m:] if len(prices) >= win_1m else prices
    recent_month_return = return_over_period(prices_1m) if len(prices_1m) >= 2 else None

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
        "max_drawdown_recovery_days": max_drawdown_recovery_days,
        "max_single_day_drop": _max_single_day_drop(nav_df),
    }


def window_metrics(nav_df: pd.DataFrame, end_date: pd.Timestamp, years: int) -> dict[str, float | None]:
    """计算近 N 年窗口指标。与 Backtest 重叠的指标使用 fund_metrics_core，窗口为最近 N 个交易日。仅支持 years=1 或 3。"""
    if years not in (1, 3):
        raise ValueError(f"window_metrics 仅支持 years=1 或 3，当前为 {years}")
    n_rows = _config.trading_days_per_year if years == 1 else _config.trading_days_per_year * 3
    win = nav_df.tail(n_rows).copy()
    if win.empty or len(win) < 2:
        return {}

    dates = win["净值日期"].to_numpy(dtype="datetime64[D]")
    prices = win["复权净值"].to_numpy(dtype=float)
    prefix = f"{years}y"

    core_out = compute_low_risk_debt_metrics(dates, prices)

    out: dict[str, float | None] = {}
    for cn, en in CN_TO_EN_LOW_RISK.items():
        if cn not in core_out:
            continue
        if en.endswith("_1y") and years == 1:
            out[en] = core_out[cn]
        elif en.endswith("_3y") and years == 3:
            out[en] = core_out[cn]
        elif en == "recent_month_return" and years == 1:
            out[en] = core_out[cn]

    w_ret = _period_returns(win, "W-FRI")
    m_ret = _period_returns(win, "ME")
    q_ret = _period_returns(win, "QE")

    if years == 1:
        out["up_month_ratio_1y"] = _up_ratio(m_ret)
    else:
        out["up_week_ratio_3y"] = _up_ratio(w_ret)
    out[f"month_return_std_{prefix}"] = _std(m_ret)
    if years == 1:
        out["week_return_std_1y"] = core_out.get("近1年周涨跌幅标准差")
    else:
        out["week_return_std_3y"] = _std(w_ret)
    out[f"max_single_day_drop_{prefix}"] = _max_single_day_drop(win)
    if years == 3:
        out["up_quarter_ratio_3y"] = _up_ratio(q_ret)
        out["quarter_return_std_3y"] = _std(q_ret)
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
