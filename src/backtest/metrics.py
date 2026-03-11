"""回测所需指标计算。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowConfig:
    trading_days_per_year: int = 252
    trading_days_per_month: int = 21
    trading_days_per_week: int = 5


def _safe_slice(arr: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=float)
    if len(arr) <= length:
        return arr
    return arr[-length:]


def _cagr(prices: np.ndarray, trading_days_per_year: int) -> float | None:
    if len(prices) < 2:
        return None
    start = prices[0]
    end = prices[-1]
    if start <= 0 or end <= 0:
        return None
    years = (len(prices) - 1) / trading_days_per_year
    if years <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def _max_drawdown(prices: np.ndarray) -> float | None:
    if len(prices) < 2:
        return None
    running_max = np.maximum.accumulate(prices)
    drawdown = prices / running_max - 1.0
    return float(np.nanmin(drawdown))


def _longest_recovery_days(dates: np.ndarray, prices: np.ndarray) -> float | None:
    if len(prices) < 2:
        return None
    peak_idx = 0
    in_drawdown = False
    max_days = 0.0
    for i in range(1, len(prices)):
        if prices[i] >= prices[peak_idx]:
            if in_drawdown:
                days = (dates[i] - dates[peak_idx]).astype("timedelta64[D]").astype(int)
                max_days = max(max_days, float(days))
                in_drawdown = False
            peak_idx = i
        else:
            if not in_drawdown:
                in_drawdown = True
    if in_drawdown:
        days = (dates[-1] - dates[peak_idx]).astype("timedelta64[D]").astype(int)
        max_days = max(max_days, float(days))
    return max_days


def _return_over_period(prices: np.ndarray) -> float | None:
    if len(prices) < 2:
        return None
    start = prices[0]
    end = prices[-1]
    if start <= 0:
        return None
    return end / start - 1


def _rolling_returns(prices: np.ndarray, step: int) -> np.ndarray:
    if len(prices) <= step:
        return np.array([], dtype=float)
    return prices[step:] / prices[:-step] - 1.0


def _sharpe_ratio(daily_returns: np.ndarray, trading_days_per_year: int) -> float | None:
    if len(daily_returns) < 2:
        return None
    mean = float(np.nanmean(daily_returns))
    std = float(np.nanstd(daily_returns, ddof=1))
    if std <= 0:
        return None
    return mean / std * math.sqrt(trading_days_per_year)


def compute_low_risk_debt_metrics(
    dates: np.ndarray,
    prices: np.ndarray,
    config: WindowConfig | None = None,
) -> dict[str, float | None]:
    """计算低风险偏债策略所需指标。

    Args:
        dates: np.datetime64[D] 数组。
        prices: float 数组。
    """
    cfg = config or WindowConfig()

    prices = np.asarray(prices, dtype=float)
    dates = np.asarray(dates)

    # 近1年 / 近3年窗口
    win_1y = cfg.trading_days_per_year
    win_3y = cfg.trading_days_per_year * 3
    win_1m = cfg.trading_days_per_month
    win_1w = cfg.trading_days_per_week

    prices_1y = _safe_slice(prices, win_1y)
    dates_1y = _safe_slice(dates, win_1y)
    prices_3y = _safe_slice(prices, win_3y)
    dates_3y = _safe_slice(dates, win_3y)

    max_dd_1y = _max_drawdown(prices_1y)
    max_dd_3y = _max_drawdown(prices_3y)
    rec_days_3y = _longest_recovery_days(dates_3y, prices_3y)

    ann_return_1y = _cagr(prices_1y, cfg.trading_days_per_year)
    ann_return_3y = _cagr(prices_3y, cfg.trading_days_per_year)

    calmar_1y = None
    if ann_return_1y is not None and max_dd_1y is not None and max_dd_1y != 0:
        calmar_1y = ann_return_1y / abs(max_dd_1y)

    calmar_3y = None
    if ann_return_3y is not None and max_dd_3y is not None and max_dd_3y != 0:
        calmar_3y = ann_return_3y / abs(max_dd_3y)

    # 最近一个月涨跌幅
    prices_1m = _safe_slice(prices, win_1m)
    ret_1m = _return_over_period(prices_1m)

    # 近1年上涨星期比例（用 5 日收益近似周）
    weekly_returns_1y = _rolling_returns(prices_1y, win_1w)
    weekly_up_ratio_1y = None
    if len(weekly_returns_1y) > 0:
        weekly_up_ratio_1y = float(np.nanmean(weekly_returns_1y > 0))

    # 近3年上涨月份比例（用 21 日收益近似月）
    monthly_returns_3y = _rolling_returns(prices_3y, win_1m)
    monthly_up_ratio_3y = None
    if len(monthly_returns_3y) > 0:
        monthly_up_ratio_3y = float(np.nanmean(monthly_returns_3y > 0))

    # 近1年周涨跌幅标准差
    weekly_returns_1y_std = None
    if len(weekly_returns_1y) > 1:
        weekly_returns_1y_std = float(np.nanstd(weekly_returns_1y, ddof=1))

    # 近3年夏普比率（按日收益）
    daily_returns_3y = _rolling_returns(prices_3y, 1)
    sharpe_3y = _sharpe_ratio(daily_returns_3y, cfg.trading_days_per_year)

    return {
        "近1年最大回撤率": max_dd_1y,
        "近3年最长回撤修复天数": rec_days_3y,
        "近3年最大回撤率": max_dd_3y,
        "近1年卡玛比率": calmar_1y,
        "近1年年化收益率": ann_return_1y,
        "最近一个月涨跌幅": ret_1m,
        "近1年上涨星期比例": weekly_up_ratio_1y,
        "近3年上涨月份比例": monthly_up_ratio_3y,
        "近1年周涨跌幅标准差": weekly_returns_1y_std,
        "近3年卡玛比率": calmar_3y,
        "近3年年化收益率": ann_return_3y,
        "近3年夏普比率": sharpe_3y,
    }
