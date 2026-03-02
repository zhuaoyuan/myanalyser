"""筛选与打分的抽象接口与加载逻辑。

- 过滤策略：实现 filter_one(row) -> (是否被过滤, 过滤原因)
- 算分策略：实现 compute_score(df) -> df，返回带 综合得分 及 计算过程数据 的 DataFrame
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FilterStrategy(Protocol):
    """过滤策略协议。实现模块需定义 STRATEGY_NAME 并实现 filter_one。"""

    STRATEGY_NAME: str

    def filter_one(self, row: dict) -> tuple[bool, str]:
        """对单行数据判断是否过滤。

        Returns:
            (是否被过滤, 过滤细节原因)
            - (True, "原因") 表示该基金应被过滤掉
            - (False, "") 表示通过
        """
        ...


@runtime_checkable
class ScoreStrategy(Protocol):
    """算分策略协议。实现模块需定义 STRATEGY_NAME 并实现 compute_score。"""

    STRATEGY_NAME: str

    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """对 DataFrame 计算得分，返回带新列的 DataFrame。

        必须包含列：综合得分。
        计算过程数据以额外列展示（如 得分_风险控制、得分_短期业绩 等）。
        """
        ...


def load_filter_strategy(script_path: Path) -> FilterStrategy:
    """从 Python 脚本路径加载过滤策略。脚本需定义 STRATEGY_NAME 和 filter_one（或类实例）。"""
    spec = importlib.util.spec_from_file_location("filter_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "STRATEGY_NAME"):
        raise ValueError(f"过滤脚本必须定义 STRATEGY_NAME: {script_path}")
    if not hasattr(mod, "filter_one"):
        raise ValueError(f"过滤脚本必须定义 filter_one 函数: {script_path}")

    # 支持函数或类实例
    if callable(mod.filter_one):
        # 函数形式：包装为简单对象
        class FuncFilter:
            STRATEGY_NAME = mod.STRATEGY_NAME

            def filter_one(self, row: dict) -> tuple[bool, str]:
                return mod.filter_one(row)

        return FuncFilter()
    raise ValueError(f"filter_one 必须是可调用对象: {script_path}")


def load_score_strategy(script_path: Path) -> ScoreStrategy:
    """从 Python 脚本路径加载算分策略。脚本需定义 STRATEGY_NAME 和 compute_score。"""
    spec = importlib.util.spec_from_file_location("score_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "STRATEGY_NAME"):
        raise ValueError(f"算分脚本必须定义 STRATEGY_NAME: {script_path}")
    if not hasattr(mod, "compute_score"):
        raise ValueError(f"算分脚本必须定义 compute_score 函数: {script_path}")

    if callable(mod.compute_score):
        class FuncScore:
            STRATEGY_NAME = mod.STRATEGY_NAME

            def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
                return mod.compute_score(df)

        return FuncScore()
    raise ValueError(f"compute_score 必须是可调用对象: {script_path}")
