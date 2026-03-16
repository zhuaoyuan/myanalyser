"""v2 公共工具函数。

基金代码归一化等，供 filter/prep/compare 等模块共用。
"""
from __future__ import annotations


def safe_fund_code(value: object) -> str:
    """将输入规范化为 6 位基金代码，非法值返回空串。

    规则：非 6 位纯数字、超长、空值、非数字字符均返回空串。
    """
    if value is None:
        return ""
    if isinstance(value, int):
        if 0 <= value <= 999999:
            return f"{value:06d}"
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if value.is_integer() and 0 <= value <= 999999:
            return f"{int(value):06d}"
        return ""
    s = str(value).strip()
    if not s or s == "---":
        return ""
    if not s.isdigit() or len(s) > 6:
        return ""
    return s.zfill(6)
