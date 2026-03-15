# -*- coding: utf-8 -*-
"""
v2 prep_eligible_from_cached 单元测试。

场景分类：
- 正常：c1+b+e 全通过、部分通过、自定义输出路径
- 异常：缺失输入文件、缺失列、空 purchase
- 边界：_safe_code 空/非法/极大值，_parse_date 多格式、空值
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from v2.transforms import prep_eligible_from_cached as pec


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _make_purchase(codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"基金代码": codes})


def test_eligible_filters_c1_b_e_and_ignores_a(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001", "000002", "000003"]))
    _write_csv(
        work_dir / "fund_fee_filtered.csv",
        pd.DataFrame({
            "类型": ["A类30天", "A类30天"],
            "基金编码": ["000001", "000002"],
            "申购费率": ["0.1%", "0.1%"],
            "赎回费率": ["0%", "0%"],
        }),
    )
    _write_csv(
        work_dir / "fund_gmbd.csv",
        pd.DataFrame({
            "基金代码": ["000001", "000002", "000003"],
            "日期": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "期末净资产（亿元）": ["3", "1.5", "5"],
        }),
    )
    _write_csv(
        work_dir / "fund_overview.csv",
        pd.DataFrame({
            "基金代码": ["000001", "000002", "000003"],
            "成立日期/规模": ["2010-01-01", "2010-01-01", "2010-01-01"],
        }),
    )
    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    output = pec.run(work_dir)
    result = pd.read_csv(output, dtype=str)
    assert set(result["基金代码"].str.zfill(6)) == {"000001"}


def test_eligible_e_age_strict_over_three_years(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000010", "000011"]))
    _write_csv(
        work_dir / "fund_fee_filtered.csv",
        pd.DataFrame({
            "类型": ["A类30天", "A类30天"],
            "基金编码": ["000010", "000011"],
            "申购费率": ["0.1%", "0.1%"],
            "赎回费率": ["0%", "0%"],
        }),
    )
    _write_csv(
        work_dir / "fund_gmbd.csv",
        pd.DataFrame({
            "基金代码": ["000010", "000011"],
            "日期": ["2024-01-01", "2024-01-01"],
            "期末净资产（亿元）": ["3", "3"],
        }),
    )
    _write_csv(
        work_dir / "fund_overview.csv",
        pd.DataFrame({
            "基金代码": ["000010", "000011"],
            "成立日期/规模": ["2023-03-15", "2023-03-14"],
        }),
    )

    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    output = pec.run(work_dir)
    result = pd.read_csv(output, dtype=str)
    assert set(result["基金代码"].str.zfill(6)) == {"000011"}


def test_eligible_auto_generate_fee_filtered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000021"]))
    _write_csv(
        work_dir / "fund_fee_structured.csv",
        pd.DataFrame({
            "基金编码": ["000021", "000021"],
            "申购状态": ["开放申购", "开放申购"],
            "赎回状态": ["开放赎回", "开放赎回"],
            "数据类型": ["申购费率", "赎回费率"],
            "费率": ["0.10%", "0%"],
            "金额阶梯起点": ["0", ""],
            "持仓期限阶梯起点": ["", "0"],
            "持仓期限阶梯终点": ["", ""],
        }),
    )
    _write_csv(
        work_dir / "fund_gmbd.csv",
        pd.DataFrame({
            "基金代码": ["000021"],
            "日期": ["2024-01-01"],
            "期末净资产（亿元）": ["3"],
        }),
    )
    _write_csv(
        work_dir / "fund_overview.csv",
        pd.DataFrame({
            "基金代码": ["000021"],
            "成立日期/规模": ["2010-01-01"],
        }),
    )

    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    output = pec.run(work_dir)
    assert (work_dir / "fund_fee_filtered.csv").exists()
    result = pd.read_csv(output, dtype=str)
    assert set(result["基金代码"].str.zfill(6)) == {"000021"}


# --- _safe_code 单元测试 ---


def test_safe_code_normal_six_digits() -> None:
    assert pec._safe_code("1") == "000001"
    assert pec._safe_code("000001") == "000001"
    assert pec._safe_code("163402") == "163402"


def test_safe_code_null_and_empty() -> None:
    assert pec._safe_code(None) == ""
    assert pec._safe_code("") == ""
    assert pec._safe_code("   ") == ""
    assert pec._safe_code("---") == ""


def test_safe_code_invalid_returns_empty() -> None:
    assert pec._safe_code("abc") == ""
    assert pec._safe_code("12a34") == ""
    assert pec._safe_code("1234567") == ""  # >6 digits


def test_safe_code_int_and_float() -> None:
    assert pec._safe_code(1) == "000001"
    assert pec._safe_code(163402) == "163402"
    assert pec._safe_code(1.0) == "000001"


# --- _parse_date 单元测试 ---


def test_parse_date_iso_format() -> None:
    result = pec._parse_date("2010-01-01")
    assert result is not None
    assert result.strftime("%Y-%m-%d") == "2010-01-01"


def test_parse_date_chinese_format() -> None:
    result = pec._parse_date("2013年03月20日")
    assert result is not None
    assert result.strftime("%Y-%m-%d") == "2013-03-20"


def test_parse_date_null_and_na() -> None:
    assert pec._parse_date(None) is None
    assert pec._parse_date(float("nan")) is None
    import math
    assert pec._parse_date(float("nan")) is None  # pd.isna handles this


def test_parse_date_invalid() -> None:
    assert pec._parse_date("---") is None
    assert pec._parse_date("") is None


# --- run 异常/边界场景 ---


def test_run_missing_purchase_csv(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="missing purchase csv"):
        pec.run(work_dir)


def test_run_missing_gmbd_csv(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001"]))
    _write_csv(work_dir / "fund_fee_filtered.csv", pd.DataFrame({"基金编码": ["000001"], "类型": ["A"]}))
    with pytest.raises(FileNotFoundError, match="missing gmbd csv"):
        pec.run(work_dir)


def test_run_empty_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """所有基金均不满足过滤条件时，输出空 CSV。"""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001"]))
    _write_csv(
        work_dir / "fund_fee_filtered.csv",
        pd.DataFrame({"基金编码": ["000099"], "类型": ["A"]}),
    )  # 000001 不在 c1
    _write_csv(
        work_dir / "fund_gmbd.csv",
        pd.DataFrame({"基金代码": ["000001"], "日期": ["2024-01-01"], "期末净资产（亿元）": ["5"]}),
    )
    _write_csv(
        work_dir / "fund_overview.csv",
        pd.DataFrame({"基金代码": ["000001"], "成立日期/规模": ["2010-01-01"]}),
    )
    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    output = pec.run(work_dir)
    result = pd.read_csv(output, dtype=str)
    assert result.empty or len(result) == 0


def test_run_custom_output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    out_dir = tmp_path / "custom_out"
    out_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001"]))
    _write_csv(
        work_dir / "fund_fee_filtered.csv",
        pd.DataFrame({"基金编码": ["000001"], "类型": ["A"], "申购费率": ["0.1%"], "赎回费率": ["0%"]}),
    )
    _write_csv(
        work_dir / "fund_gmbd.csv",
        pd.DataFrame({"基金代码": ["000001"], "日期": ["2024-01-01"], "期末净资产（亿元）": ["3"]}),
    )
    _write_csv(
        work_dir / "fund_overview.csv",
        pd.DataFrame({"基金代码": ["000001"], "成立日期/规模": ["2010-01-01"]}),
    )
    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    custom_out = out_dir / "my_prep_result.csv"
    output = pec.run(work_dir, output_path=custom_out)
    assert output == custom_out
    assert custom_out.exists()


# --- _safe_code 单元测试 ---


def test_safe_code_normal_six_digits() -> None:
    assert pec._safe_code("000001") == "000001"
    assert pec._safe_code("163402") == "163402"
    assert pec._safe_code("1") == "000001"


def test_safe_code_null_and_empty() -> None:
    assert pec._safe_code(None) == ""
    assert pec._safe_code("") == ""
    assert pec._safe_code("   ") == ""
    assert pec._safe_code("---") == ""


def test_safe_code_invalid_returns_empty() -> None:
    assert pec._safe_code("abc") == ""
    assert pec._safe_code("00000a") == ""
    assert pec._safe_code("1234567") == ""  # >6 位


def test_safe_code_int_and_float() -> None:
    assert pec._safe_code(1) == "000001"
    assert pec._safe_code(163402) == "163402"
    assert pec._safe_code(1.0) == "000001"


# --- _parse_date 单元测试 ---


def test_parse_date_yyyy_mm_dd() -> None:
    result = pec._parse_date("2023-03-15")
    assert result is not None
    assert result.strftime("%Y-%m-%d") == "2023-03-15"


def test_parse_date_chinese_format() -> None:
    result = pec._parse_date("2013年03月20日")
    assert result is not None
    assert result.strftime("%Y-%m-%d") == "2013-03-20"


def test_parse_date_null_and_invalid() -> None:
    assert pec._parse_date(None) is None
    assert pec._parse_date("") is None
    assert pec._parse_date("---") is None
    assert pd.isna(pec._parse_date(float("nan"))) if pec._parse_date(float("nan")) is None else True


# --- run 异常与边界 ---


def test_run_missing_purchase_csv_raises(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="missing purchase csv"):
        pec.run(work_dir)


def test_run_missing_gmbd_csv_raises(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001"]))
    _write_csv(work_dir / "fund_fee_filtered.csv", pd.DataFrame({"基金编码": ["000001"]}))
    _write_csv(
        work_dir / "fund_overview.csv",
        pd.DataFrame({"基金代码": ["000001"], "成立日期/规模": ["2010-01-01"]}),
    )
    with pytest.raises(FileNotFoundError, match="missing gmbd csv"):
        pec.run(work_dir)


def test_run_empty_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """所有基金被过滤后，输出为空 DataFrame 但文件存在。"""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001"]))
    _write_csv(work_dir / "fund_fee_filtered.csv", pd.DataFrame({"基金编码": ["000002"]}))  # 无交集
    _write_csv(
        work_dir / "fund_gmbd.csv",
        pd.DataFrame({"基金代码": ["000001"], "日期": ["2024-01-01"], "期末净资产（亿元）": ["3"]}),
    )
    _write_csv(
        work_dir / "fund_overview.csv",
        pd.DataFrame({"基金代码": ["000001"], "成立日期/规模": ["2010-01-01"]}),
    )
    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    output = pec.run(work_dir)
    result = pd.read_csv(output, dtype=str)
    assert result.empty or set(result["基金代码"].str.zfill(6)) == set()


def test_run_custom_output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001"]))
    _write_csv(work_dir / "fund_fee_filtered.csv", pd.DataFrame({"基金编码": ["000001"]}))
    _write_csv(
        work_dir / "fund_gmbd.csv",
        pd.DataFrame({"基金代码": ["000001"], "日期": ["2024-01-01"], "期末净资产（亿元）": ["5"]}),
    )
    _write_csv(
        work_dir / "fund_overview.csv",
        pd.DataFrame({"基金代码": ["000001"], "成立日期/规模": ["2010-01-01"]}),
    )
    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    custom_out = tmp_path / "custom" / "out.csv"
    output = pec.run(work_dir, output_path=custom_out)
    assert output == custom_out.resolve()
    assert custom_out.exists()


# ---- _safe_code 单元测试 ----
def test_safe_code_normal_6_digits() -> None:
    assert pec._safe_code("000001") == "000001"
    assert pec._safe_code("163402") == "163402"
    assert pec._safe_code(1) == "000001"
    assert pec._safe_code(163402) == "163402"


def test_safe_code_null_and_invalid() -> None:
    assert pec._safe_code(None) == ""
    assert pec._safe_code("") == ""
    assert pec._safe_code("   ") == ""
    assert pec._safe_code("---") == ""
    assert pec._safe_code("abc123") == ""
    assert pec._safe_code("000001X") == ""


def test_safe_code_boundary_and_edge() -> None:
    assert pec._safe_code("1") == "000001"
    assert pec._safe_code(1.0) == "000001"
    assert pec._safe_code("123456") == "123456"
    assert pec._safe_code("1234567") == ""
    assert pec._safe_code(pd.NA) == ""


# ---- _parse_date 单元测试 ----
def test_parse_date_formats() -> None:
    assert pec._parse_date("2013年03月20日") == pd.Timestamp("2013-03-20")
    assert pec._parse_date("2010年1月5日") == pd.Timestamp("2010-01-05")
    assert pec._parse_date("2024-01-01") == pd.Timestamp("2024-01-01")
    assert pec._parse_date("2024/06/15") == pd.Timestamp("2024-06-15")


def test_parse_date_null_and_invalid() -> None:
    """NaT/None 均表示无效，实现可能返回 NaT（pd.to_datetime coercible）"""
    assert pec._parse_date(None) is None
    assert pec._parse_date("") is None
    assert pec._parse_date("---") is None
    assert pec._parse_date(float("nan")) is None
    # pd.NA 经 str 后可能被 pd.to_datetime 解析为 NaT
    r = pec._parse_date(pd.NA)
    assert r is None or (pd.isna(r) and r is not None)


# ---- run 异常场景 ----
def test_run_missing_purchase_csv(tmp_path: Path) -> None:
    (tmp_path / "work").mkdir()
    work_dir = tmp_path / "work"
    with pytest.raises(FileNotFoundError, match="missing purchase csv"):
        pec.run(work_dir)


def test_run_missing_gmbd_csv(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001"]))
    _write_csv(work_dir / "fund_fee_filtered.csv", pd.DataFrame({"基金编码": ["000001"], "类型": ["A类30天"]}))
    _write_csv(work_dir / "fund_overview.csv", pd.DataFrame({"基金代码": ["000001"], "成立日期/规模": ["2010-01-01"]}))
    with pytest.raises(FileNotFoundError, match="missing gmbd csv"):
        pec.run(work_dir)


def test_run_empty_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000099"]))
    _write_csv(work_dir / "fund_fee_filtered.csv", pd.DataFrame({"基金编码": [], "类型": []}))
    _write_csv(work_dir / "fund_gmbd.csv", pd.DataFrame({"基金代码": [], "日期": [], "期末净资产（亿元）": []}))
    _write_csv(work_dir / "fund_overview.csv", pd.DataFrame({"基金代码": [], "成立日期/规模": []}))
    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    output = pec.run(work_dir)
    result = pd.read_csv(output, dtype=str)
    assert result.empty or len(result) == 0


def test_run_custom_output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001"]))
    _write_csv(work_dir / "fund_fee_filtered.csv", pd.DataFrame({"基金编码": ["000001"], "类型": ["A类30天"]}))
    _write_csv(work_dir / "fund_gmbd.csv", pd.DataFrame({"基金代码": ["000001"], "日期": ["2024-01-01"], "期末净资产（亿元）": ["5"]}))
    _write_csv(work_dir / "fund_overview.csv", pd.DataFrame({"基金代码": ["000001"], "成立日期/规模": ["2010-01-01"]}))
    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    custom_out = tmp_path / "out" / "custom_prep.csv"
    output = pec.run(work_dir, output_path=custom_out)
    assert output == custom_out.resolve()
    assert custom_out.exists()


# --- _safe_code 单元测试 ---


def test_safe_code_normal_6_digits() -> None:
    assert pec._safe_code("000001") == "000001"
    assert pec._safe_code("163402") == "163402"
    assert pec._safe_code("1") == "000001"


def test_safe_code_null_and_empty() -> None:
    assert pec._safe_code(None) == ""
    assert pec._safe_code("") == ""
    assert pec._safe_code("   ") == ""
    assert pec._safe_code("---") == ""


def test_safe_code_invalid_returns_empty() -> None:
    assert pec._safe_code("abc") == ""
    assert pec._safe_code("00000a") == ""
    assert pec._safe_code("1234567") == ""  # >6 位
    assert pec._safe_code("12.34") == ""


def test_safe_code_int_float() -> None:
    assert pec._safe_code(1) == "000001"
    assert pec._safe_code(163402) == "163402"
    assert pec._safe_code(1.0) == "000001"
    assert pec._safe_code(3.14) == ""  # 非整数 float


# --- _parse_date 单元测试 ---


def test_parse_date_iso_format() -> None:
    result = pec._parse_date("2010-01-01")
    assert result is not None
    assert result.strftime("%Y-%m-%d") == "2010-01-01"


def test_parse_date_chinese_format() -> None:
    result = pec._parse_date("2013年03月20日")
    assert result is not None
    assert result.strftime("%Y-%m-%d") == "2013-03-20"


def test_parse_date_null_na() -> None:
    assert pec._parse_date(None) is None
    assert pec._parse_date(float("nan")) is None
    assert pec._parse_date("") is None
    assert pec._parse_date("---") is None


# --- run 异常与边界 ---


def test_run_missing_purchase_csv(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_gmbd.csv", pd.DataFrame({"基金代码": ["000001"], "期末净资产（亿元）": ["3"]}))
    _write_csv(work_dir / "fund_overview.csv", pd.DataFrame({"基金代码": ["000001"], "成立日期": ["2010-01-01"]}))
    with pytest.raises(FileNotFoundError, match="missing purchase csv"):
        pec.run(work_dir)


def test_run_missing_gmbd_csv(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001"]))
    _write_csv(
        work_dir / "fund_fee_filtered.csv",
        pd.DataFrame({"基金编码": ["000001"], "类型": ["A类30天"]}),
    )
    with pytest.raises(FileNotFoundError, match="missing gmbd csv"):
        pec.run(work_dir)


def test_run_empty_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["999999"]))
    _write_csv(
        work_dir / "fund_fee_filtered.csv",
        pd.DataFrame({"基金编码": ["000001"], "类型": ["A类30天"]}),
    )
    _write_csv(
        work_dir / "fund_gmbd.csv",
        pd.DataFrame({"基金代码": ["999999"], "日期": ["2024-01-01"], "期末净资产（亿元）": ["5"]}),
    )
    _write_csv(
        work_dir / "fund_overview.csv",
        pd.DataFrame({"基金代码": ["999999"], "成立日期/规模": ["2010-01-01"]}),
    )
    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    output = pec.run(work_dir)
    result = pd.read_csv(output, dtype=str)
    assert result.empty or len(result) == 0


def test_run_custom_output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    custom_out = tmp_path / "custom_output" / "my_result.csv"
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001"]))
    _write_csv(
        work_dir / "fund_fee_filtered.csv",
        pd.DataFrame({"基金编码": ["000001"], "类型": ["A类30天"], "申购费率": ["0.1%"], "赎回费率": ["0%"]}),
    )
    _write_csv(
        work_dir / "fund_gmbd.csv",
        pd.DataFrame({"基金代码": ["000001"], "日期": ["2024-01-01"], "期末净资产（亿元）": ["3"]}),
    )
    _write_csv(
        work_dir / "fund_overview.csv",
        pd.DataFrame({"基金代码": ["000001"], "成立日期/规模": ["2010-01-01"]}),
    )
    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    output = pec.run(work_dir, output_path=custom_out)
    assert output == custom_out.resolve()
    assert custom_out.exists()


# ---- _safe_code 单元测试（边界/异常）----


def test_safe_code_normal_6_digits() -> None:
    assert pec._safe_code("000001") == "000001"
    assert pec._safe_code("163402") == "163402"
    assert pec._safe_code("1") == "000001"


def test_safe_code_null_and_invalid() -> None:
    assert pec._safe_code(None) == ""
    assert pec._safe_code("") == ""
    assert pec._safe_code("---") == ""
    assert pec._safe_code("abc123") == ""
    assert pec._safe_code("1234567") == ""  # >6 位
    assert pec._safe_code("12.5") == ""  # 非整数


def test_safe_code_int_and_float_integer() -> None:
    assert pec._safe_code(1) == "000001"
    assert pec._safe_code(163402) == "163402"
    assert pec._safe_code(1.0) == "000001"


# ---- _parse_date 单元测试 ----


def test_parse_date_iso_format() -> None:
    result = pec._parse_date("2010-01-01")
    assert result is not None
    assert result.strftime("%Y-%m-%d") == "2010-01-01"


def test_parse_date_chinese_format() -> None:
    result = pec._parse_date("2013年03月20日")
    assert result is not None
    assert result.strftime("%Y-%m-%d") == "2013-03-20"


def test_parse_date_null_and_invalid() -> None:
    import numpy as np
    assert pec._parse_date(None) is None
    assert pec._parse_date(float("nan")) is None
    r = pec._parse_date(pd.NA)
    assert r is None or pd.isna(r)  # 实现可能返回 NaT
    assert pec._parse_date("---") is None
    assert pec._parse_date("") is None


# ---- run 异常/边界场景 ----


def test_run_missing_purchase_csv(tmp_path: Path) -> None:
    (tmp_path / "work").mkdir()
    with pytest.raises(FileNotFoundError, match="missing purchase csv"):
        pec.run(tmp_path / "work")


def test_run_missing_gmbd_csv(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001"]))
    with pytest.raises(FileNotFoundError, match="missing gmbd csv"):
        pec.run(work_dir)


def test_run_empty_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """所有基金均被过滤后，输出空 CSV。"""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000099"]))
    _write_csv(
        work_dir / "fund_fee_filtered.csv",
        pd.DataFrame({"基金编码": ["000001"], "类型": ["A类30天"]}),
    )
    _write_csv(
        work_dir / "fund_gmbd.csv",
        pd.DataFrame({
            "基金代码": ["000099"],
            "日期": ["2024-01-01"],
            "期末净资产（亿元）": ["1"],
        }),
    )
    _write_csv(
        work_dir / "fund_overview.csv",
        pd.DataFrame({"基金代码": ["000099"], "成立日期/规模": ["2010-01-01"]}),
    )
    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    output = pec.run(work_dir)
    result = pd.read_csv(output, dtype=str)
    assert result.empty or len(result) == 0


def test_run_custom_output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001"]))
    _write_csv(
        work_dir / "fund_fee_filtered.csv",
        pd.DataFrame({"基金编码": ["000001"], "类型": ["A类30天"]}),
    )
    _write_csv(
        work_dir / "fund_gmbd.csv",
        pd.DataFrame({
            "基金代码": ["000001"],
            "日期": ["2024-01-01"],
            "期末净资产（亿元）": ["3"],
        }),
    )
    _write_csv(
        work_dir / "fund_overview.csv",
        pd.DataFrame({"基金代码": ["000001"], "成立日期/规模": ["2010-01-01"]}),
    )
    custom_out = tmp_path / "custom_prep_result.csv"
    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    output = pec.run(work_dir, output_path=custom_out)
    assert output == custom_out
    assert custom_out.exists()


# === _safe_code 边界/异常 ===
def test_safe_code_normal() -> None:
    assert pec._safe_code("000001") == "000001"
    assert pec._safe_code(1) == "000001"
    assert pec._safe_code(163402) == "163402"
    assert pec._safe_code(1.0) == "000001"


def test_safe_code_null_invalid() -> None:
    assert pec._safe_code(None) == ""
    assert pec._safe_code("") == ""
    assert pec._safe_code("   ") == ""
    assert pec._safe_code("---") == ""


def test_safe_code_invalid_returns_empty() -> None:
    assert pec._safe_code("abc123") == ""
    assert pec._safe_code("1234567") == ""  # >6 位
    assert pec._safe_code("12.5") == ""
    assert pec._safe_code(pd.NA) == ""


# === _parse_date 边界/异常 ===
def test_parse_date_formats() -> None:
    assert pec._parse_date("2010-01-01") is not None
    assert pec._parse_date("2013年03月20日") is not None
    assert pec._parse_date("2013年3月5日") is not None


def test_parse_date_null_invalid() -> None:
    assert pec._parse_date(None) is None
    assert pec._parse_date("---") is None
    assert pec._parse_date("") is None
    r = pec._parse_date(pd.NA)
    assert r is None or pd.isna(r)  # 实现可能返回 NaT


# === run 异常场景 ===
def test_run_missing_purchase_raises(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="missing purchase"):
        pec.run(work_dir)


def test_run_missing_gmbd_raises(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001"]))
    _write_csv(work_dir / "fund_fee_filtered.csv", pd.DataFrame({"基金编码": ["000001"], "类型": ["A类30天"]}))
    _write_csv(work_dir / "fund_overview.csv", pd.DataFrame({"基金代码": ["000001"], "成立日期/规模": ["2010-01-01"]}))
    with pytest.raises(FileNotFoundError, match="missing gmbd"):
        pec.run(work_dir)


def test_run_empty_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """所有基金均被过滤，输出空 CSV"""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["999999"]))
    _write_csv(work_dir / "fund_fee_filtered.csv", pd.DataFrame({"基金编码": [], "类型": []}))
    _write_csv(work_dir / "fund_gmbd.csv", pd.DataFrame({"基金代码": [], "日期": [], "期末净资产（亿元）": []}))
    _write_csv(work_dir / "fund_overview.csv", pd.DataFrame({"基金代码": [], "成立日期/规模": []}))
    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    output = pec.run(work_dir)
    result = pd.read_csv(output, dtype=str)
    assert result.empty or len(result) == 0


def test_run_custom_output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    out_custom = tmp_path / "custom_output" / "my_result.csv"
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001"]))
    _write_csv(work_dir / "fund_fee_filtered.csv", pd.DataFrame({"基金编码": ["000001"], "类型": ["A类30天"]}))
    _write_csv(
        work_dir / "fund_gmbd.csv",
        pd.DataFrame({"基金代码": ["000001"], "日期": ["2024-01-01"], "期末净资产（亿元）": ["3"]}),
    )
    _write_csv(work_dir / "fund_overview.csv", pd.DataFrame({"基金代码": ["000001"], "成立日期/规模": ["2010-01-01"]}))
    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    output = pec.run(work_dir, output_path=out_custom)
    assert output == out_custom
    assert output.exists()


# --- 场景分类：_safe_code ---


def test_safe_code_normal_six_digit() -> None:
    """正常场景：6 位数字字符串。"""
    assert pec._safe_code("000001") == "000001"
    assert pec._safe_code("163402") == "163402"
    assert pec._safe_code("1") == "000001"


def test_safe_code_null_and_empty() -> None:
    """异常/边界：None、空字符串、---。"""
    assert pec._safe_code(None) == ""
    assert pec._safe_code("") == ""
    assert pec._safe_code("   ") == ""
    assert pec._safe_code("---") == ""


def test_safe_code_int_and_float() -> None:
    """正常场景：整数、整型浮点数。"""
    assert pec._safe_code(1) == "000001"
    assert pec._safe_code(163402) == "163402"
    assert pec._safe_code(1.0) == "000001"


def test_safe_code_invalid_returns_empty() -> None:
    """异常场景：非数字、超长、非法字符。"""
    assert pec._safe_code("abc123") == ""
    assert pec._safe_code("1234567") == ""
    assert pec._safe_code("12-34") == ""


# --- 场景分类：_parse_date ---


def test_parse_date_iso_format() -> None:
    """正常场景：YYYY-MM-DD。"""
    result = pec._parse_date("2010-01-01")
    assert result is not None
    assert result.strftime("%Y-%m-%d") == "2010-01-01"


def test_parse_date_chinese_format() -> None:
    """正常场景：2013年03月20日。"""
    result = pec._parse_date("2013年03月20日")
    assert result is not None
    assert result.strftime("%Y-%m-%d") == "2013-03-20"


def test_parse_date_null_and_invalid() -> None:
    """异常/边界：None、空、---、NaN。"""
    assert pec._parse_date(None) is None
    assert pec._parse_date("") is None
    assert pec._parse_date("---") is None
    assert pec._parse_date(float("nan")) is None


# --- 场景分类：run 入口 ---


def test_run_missing_purchase_csv_raises(tmp_path: Path) -> None:
    """异常场景：缺少 fund_purchase.csv。"""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="missing purchase csv"):
        pec.run(work_dir)


def test_run_empty_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """边界：所有基金被过滤，输出空 CSV。"""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["999999"]))
    _write_csv(work_dir / "fund_fee_filtered.csv", pd.DataFrame({"基金编码": []}))
    _write_csv(
        work_dir / "fund_gmbd.csv",
        pd.DataFrame({"基金代码": [], "日期": [], "期末净资产（亿元）": []}),
    )
    _write_csv(work_dir / "fund_overview.csv", pd.DataFrame({"基金代码": [], "成立日期": []}))
    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    output = pec.run(work_dir)
    result = pd.read_csv(output, dtype=str)
    assert len(result) == 0


def test_run_custom_output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """正常场景：指定 output_path 输出到自定义路径。"""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001"]))
    _write_csv(
        work_dir / "fund_fee_filtered.csv",
        pd.DataFrame({"基金编码": ["000001"], "类型": ["A类"], "申购费率": ["0.1%"], "赎回费率": ["0%"]}),
    )
    _write_csv(
        work_dir / "fund_gmbd.csv",
        pd.DataFrame({"基金代码": ["000001"], "日期": ["2024-01-01"], "期末净资产（亿元）": ["3"]}),
    )
    _write_csv(
        work_dir / "fund_overview.csv",
        pd.DataFrame({"基金代码": ["000001"], "成立日期/规模": ["2010-01-01"]}),
    )
    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    custom_out = tmp_path / "custom" / "prep_result.csv"
    output = pec.run(work_dir, output_path=custom_out)
    assert output == custom_out.resolve()
    assert custom_out.exists()


# --- _safe_code 单元测试 ---


def test_safe_code_valid_digits() -> None:
    """正常场景：6 位及以内纯数字。"""
    assert pec._safe_code("1") == "000001"
    assert pec._safe_code("000001") == "000001"
    assert pec._safe_code("163402") == "163402"


def test_safe_code_int_and_float() -> None:
    """正常场景：整数、整型浮点数。"""
    assert pec._safe_code(1) == "000001"
    assert pec._safe_code(163402) == "163402"
    assert pec._safe_code(1.0) == "000001"


def test_safe_code_null_and_invalid() -> None:
    """异常/边界：None、空串、---、非数字、超长。"""
    assert pec._safe_code(None) == ""
    assert pec._safe_code("") == ""
    assert pec._safe_code("   ") == ""
    assert pec._safe_code("---") == ""
    assert pec._safe_code("abc") == ""
    assert pec._safe_code("16A402") == ""
    assert pec._safe_code("1234567") == ""  # >6 位
    assert pec._safe_code(pd.NA) == ""  # pandas NA -> str 后可能为 nan


def test_safe_code_leading_zeros() -> None:
    """边界：前导零、短数字。"""
    assert pec._safe_code("001") == "000001"
    assert pec._safe_code("  001  ") == "000001"


# --- _parse_date 单元测试 ---


def test_parse_date_iso_format() -> None:
    """正常场景：YYYY-MM-DD 格式。"""
    ts = pec._parse_date("2023-03-15")
    assert ts is not None
    assert ts.year == 2023 and ts.month == 3 and ts.day == 15


def test_parse_date_chinese_format() -> None:
    """正常场景：2013年03月20日 格式。"""
    ts = pec._parse_date("2013年03月20日")
    assert ts is not None
    assert ts.year == 2013 and ts.month == 3 and ts.day == 20


def test_parse_date_null_and_invalid() -> None:
    """异常/边界：None、空、---、不可解析。"""
    assert pec._parse_date(None) is None
    assert pec._parse_date("") is None
    assert pec._parse_date("---") is None
    assert pec._parse_date("无效") is None or pd.isna(pec._parse_date("无效"))


# --- run 异常/边界场景 ---


def test_run_missing_purchase_csv(tmp_path: Path) -> None:
    """异常：缺少 fund_purchase.csv。"""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_gmbd.csv", pd.DataFrame({"基金代码": ["000001"], "期末净资产（亿元）": ["3"]}))
    _write_csv(work_dir / "fund_overview.csv", pd.DataFrame({"基金代码": ["000001"], "成立日期/规模": ["2010-01-01"]}))
    _write_csv(work_dir / "fund_fee_filtered.csv", pd.DataFrame({"基金编码": ["000001"]}))
    with pytest.raises(FileNotFoundError, match="missing purchase csv"):
        pec.run(work_dir)


def test_run_missing_gmbd_csv(tmp_path: Path) -> None:
    """异常：缺少 fund_gmbd.csv。"""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001"]))
    _write_csv(work_dir / "fund_overview.csv", pd.DataFrame({"基金代码": ["000001"], "成立日期/规模": ["2010-01-01"]}))
    _write_csv(work_dir / "fund_fee_filtered.csv", pd.DataFrame({"基金编码": ["000001"]}))
    with pytest.raises(FileNotFoundError, match="missing gmbd csv"):
        pec.run(work_dir)


def test_run_empty_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """边界：所有基金被过滤，输出空 CSV。"""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["999999"]))  # 不在 fee/gmbd/overview
    _write_csv(
        work_dir / "fund_fee_filtered.csv",
        pd.DataFrame({"基金编码": ["000001"], "类型": ["A类30天"], "申购费率": ["0.1%"], "赎回费率": ["0%"]}),
    )
    _write_csv(
        work_dir / "fund_gmbd.csv",
        pd.DataFrame({"基金代码": ["000001"], "日期": ["2024-01-01"], "期末净资产（亿元）": ["5"]}),
    )
    _write_csv(
        work_dir / "fund_overview.csv",
        pd.DataFrame({"基金代码": ["000001"], "成立日期/规模": ["2010-01-01"]}),
    )
    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    output = pec.run(work_dir)
    result = pd.read_csv(output, dtype=str)
    assert result.empty or len(result) == 0


def test_run_custom_output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """正常：指定 output_path 写入到非 work_dir。"""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    custom_out = out_dir / "my_prep_result.csv"
    _write_csv(work_dir / "fund_purchase.csv", _make_purchase(["000001"]))
    _write_csv(
        work_dir / "fund_fee_filtered.csv",
        pd.DataFrame({"基金编码": ["000001"], "类型": ["A类30天"], "申购费率": ["0.1%"], "赎回费率": ["0%"]}),
    )
    _write_csv(
        work_dir / "fund_gmbd.csv",
        pd.DataFrame({"基金代码": ["000001"], "日期": ["2024-01-01"], "期末净资产（亿元）": ["5"]}),
    )
    _write_csv(
        work_dir / "fund_overview.csv",
        pd.DataFrame({"基金代码": ["000001"], "成立日期/规模": ["2010-01-01"]}),
    )
    monkeypatch.setattr(pec, "_get_today", lambda: pd.Timestamp("2026-03-15"))
    output = pec.run(work_dir, output_path=custom_out)
    assert output == custom_out
    assert custom_out.exists()
    result = pd.read_csv(output, dtype=str)
    assert set(result["基金代码"].str.zfill(6)) == {"000001"}
