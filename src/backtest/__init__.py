"""回测框架与策略实现入口。"""

from .data import BacktestData, load_fund_nav_data
from .engine import run_backtest
from .strategies.registry import get_strategy_bundle

__all__ = [
    "BacktestData",
    "load_fund_nav_data",
    "run_backtest",
    "get_strategy_bundle",
]
