# -*- coding: utf-8 -*-
"""v2.utils.safe_fund_code 单元测试。

基于需求 20260316_v2协议规范性检查报告_优化实施 中 2.7 迁移注意：safe_fund_code 行为变化。

场景分类：
- 正常场景：6 位纯数字、短数字补零、带前后空格、int/float 合法范围
- 异常场景：TBD/---/12a34 等非纯数字、负数、>=1000000、混合字符
- 边界条件：None、空串、空格串、pd.NA、NaN、0、999999、7 位及以上、非整数 float
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from v2.utils import safe_fund_code


# ============= 正常场景 =============


def test_safe_fund_code_normal_6_digits() -> None:
    """正常：6 位纯数字字符串。"""
    assert safe_fund_code("000001") == "000001"
    assert safe_fund_code("163402") == "163402"
    assert safe_fund_code("999999") == "999999"


def test_safe_fund_code_short_digit_zero_fill() -> None:
    """正常：短数字左侧补零至 6 位。"""
    assert safe_fund_code("1") == "000001"
    assert safe_fund_code("15") == "000015"
    assert safe_fund_code("123") == "000123"


def test_safe_fund_code_with_whitespace() -> None:
    """正常：前后空格 strip 后解析。"""
    assert safe_fund_code("  000001  ") == "000001"
    assert safe_fund_code("  15  ") == "000015"
    assert safe_fund_code("\t163402\n") == "163402"


def test_safe_fund_code_int_valid_range() -> None:
    """正常：int 在 0~999999 范围内。"""
    assert safe_fund_code(0) == "000000"
    assert safe_fund_code(1) == "000001"
    assert safe_fund_code(163402) == "163402"
    assert safe_fund_code(999999) == "999999"


def test_safe_fund_code_float_integer_valid() -> None:
    """正常：float 为整数值且在 0~999999 范围内。"""
    assert safe_fund_code(1.0) == "000001"
    assert safe_fund_code(163402.0) == "163402"
    assert safe_fund_code(999999.0) == "999999"


# ============= 异常场景（需求 2.7 迁移注意） =============


def test_safe_fund_code_non_numeric_tbd() -> None:
    """异常：TBD 等占位符非纯数字，返回空串（旧版 zfill 会得 000tbd）。"""
    assert safe_fund_code("TBD") == ""
    assert safe_fund_code("tbd") == ""


def test_safe_fund_code_dash_placeholder() -> None:
    """异常：--- 占位符返回空串。"""
    assert safe_fund_code("---") == ""


def test_safe_fund_code_mixed_alphanumeric() -> None:
    """异常：12a34 等混合字符非纯数字，返回空串。"""
    assert safe_fund_code("12a34") == ""
    assert safe_fund_code("abc") == ""
    assert safe_fund_code("00000a") == ""
    assert safe_fund_code("16A402") == ""


def test_safe_fund_code_negative_int() -> None:
    """异常：负整数返回空串（旧 prep 可能产生 -00001 等）。"""
    assert safe_fund_code(-1) == ""
    assert safe_fund_code(-999) == ""


def test_safe_fund_code_overflow_int() -> None:
    """异常：>=1000000 返回空串。"""
    assert safe_fund_code(1000000) == ""
    assert safe_fund_code(9999999) == ""
    assert safe_fund_code(10000000) == ""


def test_safe_fund_code_overflow_str() -> None:
    """异常：7 位及以上数字字符串返回空串。"""
    assert safe_fund_code("1234567") == ""
    assert safe_fund_code("1000000") == ""


# ============= 边界条件 =============


def test_safe_fund_code_none() -> None:
    """边界：None 返回空串。"""
    assert safe_fund_code(None) == ""


def test_safe_fund_code_empty_string() -> None:
    """边界：空串返回空串。"""
    assert safe_fund_code("") == ""


def test_safe_fund_code_whitespace_only() -> None:
    """边界：纯空格 strip 后为空返回空串。"""
    assert safe_fund_code("   ") == ""
    assert safe_fund_code("\t\n") == ""


def test_safe_fund_code_pd_na() -> None:
    """边界：pd.NA 返回空串。"""
    assert safe_fund_code(pd.NA) == ""


def test_safe_fund_code_nan_float() -> None:
    """边界：NaN float 返回空串。"""
    assert safe_fund_code(float("nan")) == ""
    assert safe_fund_code(math.nan) == ""


def test_safe_fund_code_non_integer_float() -> None:
    """边界：非整数 float 如 3.14 返回空串。"""
    assert safe_fund_code(3.14) == ""
    assert safe_fund_code(12.5) == ""


def test_safe_fund_code_boundary_zero() -> None:
    """边界：0 和 000000 合法。"""
    assert safe_fund_code(0) == "000000"
    assert safe_fund_code("0") == "000000"
    assert safe_fund_code("000000") == "000000"


def test_safe_fund_code_boundary_max() -> None:
    """边界：999999 为合法上界。"""
    assert safe_fund_code(999999) == "999999"
    assert safe_fund_code("999999") == "999999"


def test_safe_fund_code_boundary_exact_6_chars() -> None:
    """边界：恰好 6 位数字。"""
    assert safe_fund_code("123456") == "123456"


# ============= 并发/类型安全（可选） =============


def test_safe_fund_code_object_types() -> None:
    """边界：传入其他类型如 bool 等。"""
    # bool 会走 str() 路径，str(True)="True" -> 非数字 -> ""
    assert safe_fund_code(True) == ""
    assert safe_fund_code(False) == ""


def test_safe_fund_code_negative_float_integer() -> None:
    """边界：-1.0 为整数 float 但值为负，应返回空串。"""
    # 当前实现：float 分支检查 value.is_integer() and 0 <= value <= 999999
    # -1.0.is_integer()=True 但 0 <= -1.0 为 False，返回 ""
    assert safe_fund_code(-1.0) == ""
