"""非A类&不限申购赎回过滤策略。

满足下列条件之一即过滤（任一条命中即过滤）：
- 基金名称最后是"A"
- 申购状态为"暂停申购"
- 赎回状态为"暂停赎回"
- 申购状态为"限大额"且日累计限定金额<200000
"""

from __future__ import annotations

STRATEGY_NAME = "非A类&不限申购赎回"

_THRESHOLD_DAILY_LIMIT = 200000


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

    # 1. 基金名称最后是"A"
    name = row.get("基金名称")
    if name is not None and str(name).strip().endswith("A"):
        reasons.append("基金名称以A结尾")

    # 2. 申购状态为"暂停申购"
    purchase_status = row.get("申购状态")
    if purchase_status == "暂停申购":
        reasons.append("申购状态为暂停申购")

    # 3. 赎回状态为"暂停赎回"
    redeem_status = row.get("赎回状态")
    if redeem_status == "暂停赎回":
        reasons.append("赎回状态为暂停赎回")

    # 4. 申购状态为"限大额"且日累计限定金额<200000
    if purchase_status == "限大额":
        daily_limit = _to_float(row.get("日累计限定金额"))
        if daily_limit is not None and daily_limit < _THRESHOLD_DAILY_LIMIT:
            reasons.append(f"限大额且日累计限定金额{daily_limit}<{_THRESHOLD_DAILY_LIMIT}")

    if not reasons:
        return False, ""
    return True, "; ".join(reasons)
