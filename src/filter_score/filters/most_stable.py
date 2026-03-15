"""最稳健原则过滤策略（re-export 自共享模块）。

供 step10b 动态加载，保持 filter_score 流水线不变。
规则详见 most_stable_logic 模块。

导入约定：优先 from myanalyser.src.most_stable_logic（workspace 根在 PYTHONPATH 时），
否则 from most_stable_logic（myanalyser/src 在 PYTHONPATH 时）。run_filter_and_score 使用前者。
"""

from __future__ import annotations

# 从共享模块 re-export，保证 load_filter_strategy 按路径加载时仍可用
try:
    from myanalyser.src.most_stable_logic import STRATEGY_NAME, filter_one
except ImportError:
    from most_stable_logic import STRATEGY_NAME, filter_one
