"""最稳健原则过滤逻辑（共享模块）。

供 filter_score 流水线与 backtest 共用，实现 filter_one 规则判断。
规则（全部满足才通过，任一条不满足即过滤）：
- 近3年年化收益率 > 3
- 近1年年化收益率 > 3
- 近3年上涨季度比例 > 80
- 近3年上涨月份比例 > 70
- 近3年月涨跌幅标准差 < 1.5
- 近1年夏普比率 > 1
- 近3年夏普比率 > 1
- 近1年卡玛比率 > 1
- 近3年卡玛比率 > 1
"""

from __future__ import annotations

STRATEGY_NAME = "最稳健原则"

_RULES = [
    ("近3年年化收益率", lambda v: v is not None and v > 3, "近3年年化收益率≤3"),
    ("近1年年化收益率", lambda v: v is not None and v > 3, "近1年年化收益率≤3"),
    ("近3年上涨季度比例", lambda v: v is not None and v > 80, "近3年上涨季度比例≤80"),
    ("近3年上涨月份比例", lambda v: v is not None and v > 70, "近3年上涨月份比例≤70"),
    ("近3年月涨跌幅标准差", lambda v: v is not None and v < 1.5, "近3年月涨跌幅标准差≥1.5"),
    ("近1年夏普比率", lambda v: v is not None and v > 1, "近1年夏普比率≤1"),
    ("近3年夏普比率", lambda v: v is not None and v > 1, "近3年夏普比率≤1"),
    ("近1年卡玛比率", lambda v: v is not None and v > 1, "近1年卡玛比率≤1"),
    ("近3年卡玛比率", lambda v: v is not None and v > 1, "近3年卡玛比率≤1"),
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
