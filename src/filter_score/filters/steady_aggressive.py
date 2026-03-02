"""偏稳进取原则过滤策略。

规则（全部满足才通过，任一条不满足即过滤）：
- 近1年年化收益率 > 4
- 近3年年化收益率 > 4
- 近3年上涨季度比例 > 80
- 近3年上涨月份比例 > 60
- 近3年最大回撤率 < 10
"""

from __future__ import annotations

STRATEGY_NAME = "偏稳进取原则"

_RULES = [
    ("近1年年化收益率", lambda v: v is not None and v > 4, "近1年年化收益率≤4"),
    ("近3年年化收益率", lambda v: v is not None and v > 4, "近3年年化收益率≤4"),
    ("近3年上涨季度比例", lambda v: v is not None and v > 80, "近3年上涨季度比例≤80"),
    ("近3年上涨月份比例", lambda v: v is not None and v > 60, "近3年上涨月份比例≤60"),
    ("近3年最大回撤率", lambda v: v is not None and v < 10, "近3年最大回撤率≥10"),
]


def _to_float(val) -> float | None:
    if val is None:
        return None
    if isinstance(val, float) and val != val:  # NaN check
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def filter_one(row: dict) -> tuple[bool, str]:
    """对单行判断是否过滤。返回 (是否被过滤, 过滤原因)。"""
    reasons = []
    for col, pred, msg in _RULES:
        val = row.get(col)
        v = _to_float(val)
        if not pred(v):
            reasons.append(msg)
    if not reasons:
        return False, ""
    return True, "; ".join(reasons)
