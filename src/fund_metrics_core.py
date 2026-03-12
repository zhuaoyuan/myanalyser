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


def sortino_ratio(daily_returns: np.ndarray, trading_days_per_year: int) -> float | None:
    """索提诺比率，仅用下行波动率年化。"""
    if len(daily_returns) < 2:
        return None
    downside = daily_returns[daily_returns < 0]
    if len(downside) < 2:
        return None  # 无负收益或样本不足时无法定义下行波动
    std_down = float(np.nanstd(downside, ddof=1))
    if std_down <= 0:
        return None
    mean = float(np.nanmean(daily_returns))
    return mean / std_down * math.sqrt(trading_days_per_year)


def profit_factor(prices: np.ndarray) -> float | None:
    """盈利因子：毛利/毛损（基于净值变化）。

    无任何亏损日（losses==0）时返回 None，因分母未定义；调用方需自行处理。
    """
    if len(prices) < 2:
        return None
    changes = np.diff(np.asarray(prices, dtype=float))
    wins = np.sum(changes[changes > 0])
    losses = np.sum(changes[changes < 0])
    if losses >= 0:
        return None
    denom = -losses
    if denom <= np.finfo(float).eps:
        return None
    return float(wins / denom)


def ulcer_index(prices: np.ndarray) -> float | None:
    """溃疡指数：回撤百分比的均方根。"""
    if len(prices) < 2:
        return None
    arr = np.asarray(prices, dtype=float)
    if np.any(arr <= 0):
        return None
    running_max = np.maximum.accumulate(arr)
    drawdown_pct = (arr / running_max - 1.0) * 100
    return float(np.sqrt(np.nanmean(drawdown_pct**2)))


def ulcer_performance_index(
    prices: np.ndarray,
    trading_days_per_year: int,
    risk_free_rate: float = 0.0,
) -> float | None:
    """溃疡绩效指数 UPI = (年化收益率 - 无风险利率) / 溃疡指数。"""
    ui = ulcer_index(prices)
    if ui is None or ui <= 0:
        return None
    ann_ret = cagr(prices, trading_days_per_year)
    if ann_ret is None:
        return None
    return (ann_ret - risk_free_rate) / ui


def _linear_trend_residuals(y: np.ndarray) -> tuple[float, float, int] | None:
    """线性趋势拟合，返回 ss_tot, ss_res, n。y 长度 < 3 时返回 None。"""
    if len(y) < 3:
        return None
    y = np.asarray(y, dtype=float)
    n = len(y)
    x = np.arange(n, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot <= 0:
        return None
    x_var = np.sum((x - x_mean) ** 2)
    if x_var <= 0:
        return None
    b = np.sum((x - x_mean) * (y - y_mean)) / x_var
    a = y_mean - b * x_mean
    ss_res = np.sum((y - (a + b * x)) ** 2)
    return (float(ss_tot), float(ss_res), n)


def equity_r_squared(prices: np.ndarray) -> float | None:
    """净值序列相对于线性趋势的 R 方。"""
    out = _linear_trend_residuals(np.asarray(prices, dtype=float))
    if out is None:
        return None
    ss_tot, ss_res, _ = out
    return float(1 - ss_res / ss_tot)


def regression_std_error(prices: np.ndarray) -> float | None:
    """净值相对线性趋势的残差标准误差。"""
    out = _linear_trend_residuals(np.asarray(prices, dtype=float))
    if out is None:
        return None
    ss_tot, ss_res, n = out
    return float(np.sqrt(ss_res / (n - 2)))


def annual_volatility(daily_returns: np.ndarray, trading_days_per_year: int) -> float | None:
    """年化波动率（小数形式，如 0.1 表示 10%）。"""
    if len(daily_returns) < 2:
        return None
    std = float(np.nanstd(daily_returns, ddof=1))
    return std * math.sqrt(trading_days_per_year)


# metrics_holding 产出指标（持仓期间全样本）
HOLDING_METRIC_NAMES = (
    "年化收益率",
    "夏普比率",
    "索提诺比率",
    "卡玛比率",
    "盈利因子",
    "溃疡指数",
    "溃疡绩效指数",
    "净值R方",
    "标准误差",
    "上涨星期比例",
    "上涨月份比例",
    "最大回撤率",
    "最长回撤修复天数",
    "年化波动率",
)


def compute_holding_period_metrics(
    dates: np.ndarray,
    prices: np.ndarray,
    config: WindowConfig | None = None,
) -> dict[str, float | None]:
    """计算持仓期间全样本指标（无 1y/3y 窗口），用于 metrics_holding。

    Args:
        dates: np.datetime64[D] 数组。
        prices: float 数组。
    """
    cfg = config or WindowConfig()
    prices = np.asarray(prices, dtype=float)
    dates = np.asarray(dates)

    if len(prices) < 2:
        return {k: None for k in HOLDING_METRIC_NAMES}

    win_1m = cfg.trading_days_per_month
    win_1w = cfg.trading_days_per_week
    daily_returns = rolling_returns(prices, 1)
    weekly_returns = rolling_returns(prices, win_1w)
    monthly_returns = rolling_returns(prices, win_1m)

    ann_return = cagr(prices, cfg.trading_days_per_year)
    sharpe = sharpe_ratio(daily_returns, cfg.trading_days_per_year)
    sortino = sortino_ratio(daily_returns, cfg.trading_days_per_year)
    max_dd = max_drawdown(prices)
    calmar = None
    if ann_return is not None and max_dd is not None and max_dd != 0:
        calmar = ann_return / abs(max_dd)
    pf = profit_factor(prices)
    ui = ulcer_index(prices)
    upi = ulcer_performance_index(prices, cfg.trading_days_per_year)
    r2 = equity_r_squared(prices)
    std_err = regression_std_error(prices)
    weekly_up = float(np.nanmean(weekly_returns > 0)) if len(weekly_returns) > 0 else None
    monthly_up = float(np.nanmean(monthly_returns > 0)) if len(monthly_returns) > 0 else None
    rec_days = longest_recovery_days(dates, prices)
    ann_vol = annual_volatility(daily_returns, cfg.trading_days_per_year)

    return {
        "年化收益率": ann_return,
        "夏普比率": sharpe,
        "索提诺比率": sortino,
        "卡玛比率": calmar,
        "盈利因子": pf,
        "溃疡指数": ui,
        "溃疡绩效指数": upi,
        "净值R方": r2,
        "标准误差": std_err,
        "上涨星期比例": weekly_up,
        "上涨月份比例": monthly_up,
        "最大回撤率": max_dd,
        "最长回撤修复天数": rec_days,
        "年化波动率": ann_vol,
    }


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
