"""稳健型（低波动偏债）过滤逻辑（共享模块）。

依据 docs/参考/分类型的硬约束和主次目标.md 第二节「稳健型」。
硬约束（全部满足才通过）：
- 最大回撤 ≥ -8%（即 近3年最大回撤率 >= -0.08，小数）
- 年化收益 ≥ 5%（即 近3年年化收益率 >= 0.05，小数）
- 夏普比率 ≥ 0.5（即 近3年夏普比率 >= 0.5）
"""

from __future__ import annotations

from typing import Any

STRATEGY_NAME = "稳健型（低波动偏债）"

# 硬约束：指标为 fund_metrics_core 输出的小数形式
_MAX_DD_THRESHOLD = -0.08   # -8%
_ANN_RETURN_THRESHOLD = 0.05  # 5%
_SHARPE_THRESHOLD = 0.5


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, float) and val != val:  # NaN check
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def filter_one(row: dict) -> tuple[bool, str]:
    """对单行判断是否过滤。返回 (是否被过滤, 过滤原因)。

    期望 row 含：近3年最大回撤率、近3年年化收益率、近3年夏普比率（均为小数）。
    """
    reasons = []
    max_dd = _to_float(row.get("近3年最大回撤率"))
    ann = _to_float(row.get("近3年年化收益率"))
    sharpe = _to_float(row.get("近3年夏普比率"))

    if max_dd is None:
        reasons.append("近3年最大回撤率缺失")
    elif max_dd < _MAX_DD_THRESHOLD:
        reasons.append("近3年最大回撤率<-8%")
    if ann is None or ann < _ANN_RETURN_THRESHOLD:
        reasons.append("近3年年化收益率<5%")
    if sharpe is None or sharpe < _SHARPE_THRESHOLD:
        reasons.append("近3年夏普比率<0.5")

    if not reasons:
        return False, ""
    return True, "; ".join(reasons)
