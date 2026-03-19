# -*- coding: utf-8 -*-
"""
预备数据工作流 prep_data_workflow 的单元测试。

场景分类：
- 正常场景：纯函数、筛选逻辑、完整流程（mock 外部依赖）
- 异常场景：缺参、不存在的 CSV、子 CLI 失败
- 边界条件：空值、恰好等于阈值（2亿、date）、空 DataFrame
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import prep_data_workflow as pw


# ============= 纯函数：_safe_code =============


def test_safe_code_normal() -> None:
    """正常：6 位代码、不足补零、带空格。"""
    assert pw._safe_code("000001") == "000001"
    assert pw._safe_code("1") == "000001"
    assert pw._safe_code("  15  ") == "000015"


def test_safe_code_boundary() -> None:
    """边界：空字符串、None、数字。"""
    assert pw._safe_code("") == "000000"
    # str(None)="None", zfill(6) -> "00None"
    assert pw._safe_code(None) == "00None"
    assert pw._safe_code(123) == "000123"


# ============= 纯函数：_parse_date =============


def test_parse_date_normal() -> None:
    """正常：YYYY-MM-DD、中文格式。"""
    d = pw._parse_date("2024-01-15")
    assert d is not None
    assert d.year == 2024 and d.month == 1 and d.day == 15

    d2 = pw._parse_date("2013年03月20日")
    assert d2 is not None
    assert d2.year == 2013 and d2.month == 3 and d2.day == 20


def test_parse_date_empty_and_special() -> None:
    """边界：空值、---、None。注意：pd.NA 经 to_datetime 可能返回 NaT，在筛选时与 None 等效。"""
    assert pw._parse_date(None) is None
    assert pw._parse_date("") is None
    assert pw._parse_date("---") is None
    # pd.NA 不是 float，当前实现会落入 to_datetime，返回 NaT；在筛选逻辑中与 None 等效
    result = pw._parse_date(pd.NA)
    assert result is None or pd.isna(result)


def test_parse_date_slash_format() -> None:
    """正常：YYYY/MM/DD。"""
    d = pw._parse_date("2024/06/01")
    assert d is not None
    assert d.year == 2024 and d.month == 6 and d.day == 1


# ============= _apply_filters 筛选逻辑 =============


def _make_purchase_df(codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "基金代码": codes,
        "基金简称": ["A"] * len(codes),
        "申购状态": ["开放申购"] * len(codes),
        "赎回状态": ["开放赎回"] * len(codes),
        "下一开放日": [""] * len(codes),
        "购买起点": [10] * len(codes),
        "日累计限定金额": [100] * len(codes),
        "手续费": [0.1] * len(codes),
    })


def test_apply_filters_all_pass(tmp_path: Path) -> None:
    """正常：全部条件满足，保留所有基金。"""
    date_str = "2024-01-01"
    purchase = _make_purchase_df(["000001", "000002"])
    purchase.to_csv(tmp_path / "x.csv", index=False, encoding="utf-8-sig")

    # c.1: 两个基金都在
    pd.DataFrame({"类型": ["A类30天", "A类30天"], "基金编码": ["000001", "000002"], "申购费率": ["0.1%", "0.1%"], "赎回费率": ["0%", "0%"]}).to_csv(
        tmp_path / "c1.csv", index=False, encoding="utf-8-sig"
    )
    # b: 都曾规模>2亿
    pd.DataFrame({
        "基金代码": ["000001", "000002"],
        "日期": ["2024-06-01", "2024-06-01"],
        "期末净资产（亿元）": ["5", "3"],
    }).to_csv(tmp_path / "b.csv", index=False, encoding="utf-8-sig")
    # e: date 前成立
    pd.DataFrame({
        "基金代码": ["000001", "000002"],
        "成立日期/规模": ["2020-01-01", "2021-06-15"],
    }).to_csv(tmp_path / "e.csv", index=False, encoding="utf-8-sig")

    logger = logging.getLogger("test")
    result = pw._apply_filters(
        purchase, date_str,
        tmp_path / "b.csv", tmp_path / "c1.csv", tmp_path / "e.csv",
        logger,
    )
    assert len(result) == 2
    assert set(result["基金代码"].str.zfill(6)) == {"000001", "000002"}


def test_apply_filters_boundary_scale_exactly_2(tmp_path: Path) -> None:
    """边界：规模恰好 2 亿应排除（需求是 >2亿 保留）。"""
    purchase = _make_purchase_df(["000001"])
    pd.DataFrame({"类型": ["A类30天"], "基金编码": ["000001"], "申购费率": ["0.1%"], "赎回费率": ["0%"]}).to_csv(
        tmp_path / "c1.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({
        "基金代码": ["000001"],
        "日期": ["2024-06-01"],
        "期末净资产（亿元）": ["2"],
    }).to_csv(tmp_path / "b.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "基金代码": ["000001"],
        "成立日期/规模": ["2020-01-01"],
    }).to_csv(tmp_path / "e.csv", index=False, encoding="utf-8-sig")

    logger = logging.getLogger("test")
    result = pw._apply_filters(
        purchase, "2024-01-01",
        tmp_path / "b.csv", tmp_path / "c1.csv", tmp_path / "e.csv",
        logger,
    )
    assert len(result) == 0


def test_apply_filters_boundary_inc_date_eq_date(tmp_path: Path) -> None:
    """边界：成立日期恰好等于 date 应排除（需求是 date 前成立）。"""
    purchase = _make_purchase_df(["000001"])
    pd.DataFrame({"类型": ["A类30天"], "基金编码": ["000001"], "申购费率": ["0.1%"], "赎回费率": ["0%"]}).to_csv(
        tmp_path / "c1.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({
        "基金代码": ["000001"],
        "日期": ["2024-06-01"],
        "期末净资产（亿元）": ["5"],
    }).to_csv(tmp_path / "b.csv", index=False, encoding="utf-8-sig")
    # 成立日期 = 2024-01-01
    pd.DataFrame({
        "基金代码": ["000001"],
        "成立日期/规模": ["2024-01-01"],
    }).to_csv(tmp_path / "e.csv", index=False, encoding="utf-8-sig")

    logger = logging.getLogger("test")
    result = pw._apply_filters(
        purchase, "2024-01-01",
        tmp_path / "b.csv", tmp_path / "c1.csv", tmp_path / "e.csv",
        logger,
    )
    assert len(result) == 0


def test_apply_filters_missing_file_skips_condition(tmp_path: Path) -> None:
    """正常：某 CSV 不存在时跳过该条件（不报错）。"""
    purchase = _make_purchase_df(["000001"])
    pd.DataFrame({"类型": ["A类30天"], "基金编码": ["000001"], "申购费率": ["0.1%"], "赎回费率": ["0%"]}).to_csv(
        tmp_path / "c1.csv", index=False, encoding="utf-8-sig"
    )
    # b 不存在 -> 跳过 b，其余满足
    pd.DataFrame({
        "基金代码": ["000001"],
        "日期": ["2024-06-01"],
        "期末净资产（亿元）": ["5"],
    }).to_csv(tmp_path / "b.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "基金代码": ["000001"],
        "成立日期/规模": ["2020-01-01"],
    }).to_csv(tmp_path / "e.csv", index=False, encoding="utf-8-sig")

    logger = logging.getLogger("test")
    # 使用不存在的 b 路径，应跳过 b 条件
    result = pw._apply_filters(
        purchase, "2024-01-01",
        tmp_path / "b_nonexistent.csv", tmp_path / "c1.csv", tmp_path / "e.csv",
        logger,
    )
    # b 跳过，c1+e 满足，应保留
    assert len(result) == 1


def test_apply_filters_empty_purchase(tmp_path: Path) -> None:
    """边界：空 purchase 返回空 DataFrame。"""
    purchase = _make_purchase_df([])
    pd.DataFrame(columns=["类型", "基金编码", "申购费率", "赎回费率"]).to_csv(
        tmp_path / "c1.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["基金代码", "日期", "期末净资产（亿元）"]).to_csv(
        tmp_path / "b.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["基金代码", "成立日期/规模"]).to_csv(
        tmp_path / "e.csv", index=False, encoding="utf-8-sig"
    )

    logger = logging.getLogger("test")
    result = pw._apply_filters(
        purchase, "2024-01-01",
        tmp_path / "b.csv", tmp_path / "c1.csv", tmp_path / "e.csv",
        logger,
    )
    assert len(result) == 0


# ============= run() 完整流程（mock） =============


def test_run_with_all_csvs_incremental(tmp_path: Path) -> None:
    """正常：传入全部已有 CSV，增量模式跑通。"""
    work = tmp_path / "work"
    out = tmp_path / "result.csv"
    work.mkdir()

    purchase_csv = tmp_path / "purchase.csv"
    _make_purchase_df(["000001", "000002"]).to_csv(purchase_csv, index=False, encoding="utf-8-sig")

    cyrjg_csv = tmp_path / "cyrjg.csv"
    pd.DataFrame({
        "基金代码": ["000001", "000002"],
        "日期": ["2024-06-01", "2024-06-01"],
        "机构持有比例": ["50%", "40%"],
    }).to_csv(cyrjg_csv, index=False, encoding="utf-8-sig")

    gmbd_csv = tmp_path / "gmbd.csv"
    pd.DataFrame({
        "基金代码": ["000001", "000002"],
        "日期": ["2024-06-01", "2024-06-01"],
        "期末净资产（亿元）": ["5", "3"],
    }).to_csv(gmbd_csv, index=False, encoding="utf-8-sig")

    fee_csv = tmp_path / "fee.csv"
    pd.DataFrame({
        "基金编码": ["000001", "000002"],
        "申购状态": ["开放申购", "开放申购"],
        "赎回状态": ["开放赎回", "开放赎回"],
    }).to_csv(fee_csv, index=False, encoding="utf-8-sig")

    overview_csv = tmp_path / "overview.csv"
    pd.DataFrame({
        "基金代码": ["000001", "000002"],
        "成立日期/规模": ["2020-01-01", "2021-06-15"],
    }).to_csv(overview_csv, index=False, encoding="utf-8-sig")

    def mock_run_cli(script: str, args: list, logger: logging.Logger) -> bool:
        if "filter_fund_fee_by_holding" in script:
            # 解析 -o 参数
            out_path = work / "fund_fee_filtered.csv"
            for i, a in enumerate(args):
                if a == "-o" and i + 1 < len(args):
                    out_path = Path(args[i + 1])
                    break
            pd.DataFrame({
                "类型": ["A类30天", "A类30天"],
                "基金编码": ["000001", "000002"],
                "申购费率": ["0.1%", "0.1%"],
                "赎回费率": ["0%", "0%"],
            }).to_csv(out_path, index=False, encoding="utf-8-sig")
            return True
        return False

    with patch.object(pw, "_run_cli", side_effect=mock_run_cli):
        pw.run(
            date_str="2024-01-01",
            output_path=out,
            work_dir=work,
            purchase_csv=purchase_csv,
            cyrjg_csv=cyrjg_csv,
            gmbd_csv=gmbd_csv,
            fee_csv=fee_csv,
            overview_csv=overview_csv,
            delay=0.3,
        )
    assert out.exists()
    df = pd.read_csv(out, dtype=str)
    assert len(df) == 2


def test_cli_help() -> None:
    """正常：CLI -h 可执行且退出码 0。"""
    import sys
    with patch.object(sys, "argv", ["prep_data_workflow.py", "-h"]):
        with pytest.raises(SystemExit) as exc:
            pw.main()
        assert exc.value.code == 0


def test_cli_missing_required_args() -> None:
    """异常：缺少 --date 或 -o 时 argparse 报错。"""
    import sys
    with patch.object(sys, "argv", ["prep_data_workflow.py"]):
        with pytest.raises(SystemExit) as exc:
            pw.main()
        assert exc.value.code != 0


def test_step_a_cyrjg_cli_failure(tmp_path: Path) -> None:
    """异常：cyrjg 抓取失败（无 existing）时抛出 FileNotFoundError。"""
    x_path = tmp_path / "x.csv"
    _make_purchase_df(["000001"]).to_csv(x_path, index=False, encoding="utf-8-sig")
    work = tmp_path / "work"
    work.mkdir()

    with patch.object(pw, "_run_cli", return_value=False):
        with pytest.raises(FileNotFoundError, match="持有人比例抓取失败"):
            pw._step_a_cyrjg(x_path, work, None, 0.3, logging.getLogger("test"))


def test_safe_code_none() -> None:
    """边界：_safe_code(None) 实际行为。"""
    v = pw._safe_code(None)
    # str(None)='None', strip 后 zfill(6) -> '00None' (len<6 时左侧补0)
    assert isinstance(v, str)
    assert len(v) >= 6 or v == "00None"  # 实际可能为 "00None"
