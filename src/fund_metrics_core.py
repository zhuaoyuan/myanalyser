"""基金指标计算核心逻辑（Backtest 与 Scoreboard 共用）。

统一口径（A 股）：
- 年化收益：243 交易日/年（A 股近 10 年平均约 242–244 天）
- 最近一个月：最近 20 个交易日（剔除长假干扰后的平均月交易日）
- 回撤修复天数：从峰顶到收复峰顶的全程时长（含下跌+回升）
- 周/月：5 日收益近似周、20 日收益近似月
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowConfig:
    trading_days_per_year: int = 243  # A 股近 10 年平均交易日约 242–244 天
    trading_days_per_month: int = 20  # 剔除长假干扰后的平均月交易日
    trading_days_per_week: int = 5   # 维持不变，A 股极少周六开盘（即便调休）


def safe_slice(arr: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=arr.dtype)
    if len(arr) <= length:
        return arr
    return arr[-length:]


def cagr(prices: np.ndarray, trading_days_per_year: int) -> float | None:
    """年化收益率，基准 243 交易日/年（A 股）。"""
    if len(prices) < 2:
        return None
    start = float(prices[0])
    end = float(prices[-1])
    if start <= 0 or end <= 0:
        return None
    years = (len(prices) - 1) / trading_days_per_year
    if years <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def max_drawdown(prices: np.ndarray) -> float | None:
    """最大回撤率（负值，如 -0.1 表示 10% 回撤）。含 NaN 时返回 None。"""
    if len(prices) < 2:
        return None
    arr = np.asarray(prices, dtype=float)
    if np.any(np.isnan(arr)):
        return None
    running_max = np.maximum.accumulate(arr)
    drawdown = arr / running_max - 1.0
    result = float(np.nanmin(drawdown))
    return None if np.isnan(result) else result


def longest_recovery_days(dates: np.ndarray, prices: np.ndarray) -> float | None:
    """最长回撤修复天数：从峰顶到收复峰顶的全程时长（含下跌+回升），自然日。"""
    if len(prices) < 2:
        return None
    dates = np.asarray(dates)
    prices = np.asarray(prices, dtype=float)
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


def return_over_period(prices: np.ndarray) -> float | None:
    """区间涨跌幅 (end/start - 1)。"""
    if len(prices) < 2:
        return None
    start = float(prices[0])
    end = float(prices[-1])
    if start <= 0:
        return None
    return end / start - 1


def rolling_returns(prices: np.ndarray, step: int) -> np.ndarray:
    """step 日滚动收益。"""
    if len(prices) <= step:
        return np.array([], dtype=float)
    return np.asarray(prices[step:], dtype=float) / np.asarray(prices[:-step], dtype=float) - 1.0


def sharpe_ratio(daily_returns: np.ndarray, trading_days_per_year: int) -> float | None:
    """夏普比率，按日收益年化。"""
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
    """计算低风险偏债策略所需指标（中文列名，Backtest/Scoreboard 共用）。

    Args:
        dates: np.datetime64[D] 数组。
        prices: float 数组。
    """
    cfg = config or WindowConfig()
    prices = np.asarray(prices, dtype=float)
    dates = np.asarray(dates)

    win_1y = cfg.trading_days_per_year
    win_3y = cfg.trading_days_per_year * 3
    win_1m = cfg.trading_days_per_month
    win_1w = cfg.trading_days_per_week

    # 窗口不足时返回 None，避免用短样本冠以「近N年」造成误导
    has_1y = len(prices) >= win_1y
    has_3y = len(prices) >= win_3y

    prices_1y = safe_slice(prices, win_1y) if has_1y else np.array([], dtype=float)
    dates_1y = safe_slice(dates, win_1y) if has_1y else np.array([], dtype=dates.dtype)
    prices_3y = safe_slice(prices, win_3y) if has_3y else np.array([], dtype=float)
    dates_3y = safe_slice(dates, win_3y) if has_3y else np.array([], dtype=dates.dtype)

    max_dd_1y = max_drawdown(prices_1y) if has_1y else None
    max_dd_3y = max_drawdown(prices_3y) if has_3y else None
    rec_days_1y = longest_recovery_days(dates_1y, prices_1y) if has_1y else None
    rec_days_3y = longest_recovery_days(dates_3y, prices_3y) if has_3y else None

    ann_return_1y = cagr(prices_1y, cfg.trading_days_per_year) if has_1y else None
    ann_return_3y = cagr(prices_3y, cfg.trading_days_per_year) if has_3y else None

    calmar_1y = None
    if has_1y and ann_return_1y is not None and max_dd_1y is not None and max_dd_1y != 0:
        calmar_1y = ann_return_1y / abs(max_dd_1y)
    calmar_3y = None
    if has_3y and ann_return_3y is not None and max_dd_3y is not None and max_dd_3y != 0:
        calmar_3y = ann_return_3y / abs(max_dd_3y)

    prices_1m = safe_slice(prices, win_1m)
    ret_1m = return_over_period(prices_1m)

    weekly_returns_1y = rolling_returns(prices_1y, win_1w) if has_1y else np.array([], dtype=float)
    weekly_up_ratio_1y = float(np.nanmean(weekly_returns_1y > 0)) if len(weekly_returns_1y) > 0 else None

    monthly_returns_3y = rolling_returns(prices_3y, win_1m) if has_3y else np.array([], dtype=float)
    monthly_up_ratio_3y = float(np.nanmean(monthly_returns_3y > 0)) if len(monthly_returns_3y) > 0 else None

    weekly_returns_1y_std = None
    if has_1y and len(weekly_returns_1y) > 1:
        weekly_returns_1y_std = float(np.nanstd(weekly_returns_1y, ddof=1))

    daily_returns_1y = rolling_returns(prices_1y, 1) if has_1y else np.array([], dtype=float)
    daily_returns_3y = rolling_returns(prices_3y, 1) if has_3y else np.array([], dtype=float)
    sharpe_1y = sharpe_ratio(daily_returns_1y, cfg.trading_days_per_year) if has_1y else None
    sharpe_3y = sharpe_ratio(daily_returns_3y, cfg.trading_days_per_year) if has_3y else None

    return {
        "近1年最大回撤率": max_dd_1y,
        "近1年最长回撤修复天数": rec_days_1y,
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
        "近1年夏普比率": sharpe_1y,
    }


# Scoreboard 用英文列名时的映射（仅重叠指标）
CN_TO_EN_LOW_RISK = {
    "近1年最大回撤率": "max_drawdown_1y",
    "近1年最长回撤修复天数": "max_drawdown_recovery_days_1y",
    "近3年最长回撤修复天数": "max_drawdown_recovery_days_3y",
    "近3年最大回撤率": "max_drawdown_3y",
    "近1年卡玛比率": "calmar_ratio_1y",
    "近1年年化收益率": "annual_return_1y",
    "最近一个月涨跌幅": "recent_month_return",
    "近1年上涨星期比例": "up_week_ratio_1y",
    "近3年上涨月份比例": "up_month_ratio_3y",
    "近1年周涨跌幅标准差": "week_return_std_1y",
    "近3年卡玛比率": "calmar_ratio_3y",
    "近3年年化收益率": "annual_return_3y",
    "近3年夏普比率": "sharpe_ratio_3y",
    "近1年夏普比率": "sharpe_ratio_1y",
}
