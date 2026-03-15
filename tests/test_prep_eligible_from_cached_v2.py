# -*- coding: utf-8 -*-
"""
v2 prep_eligible_from_cached 单元测试。
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
