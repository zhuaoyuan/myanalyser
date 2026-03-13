# -*- coding: utf-8 -*-
"""fetch_fund_fee 脚本的单元测试（解析逻辑 + 主流程 mock）。"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from fetch_fund_fee import (
    _parse_amount_tier,
    _parse_period_tier,
    _parse_fee_value,
    _choose_min_fee,
    _load_fund_records,
    run,
)


def test_parse_amount_tier() -> None:
    assert _parse_amount_tier("小于100万元") == (0.0, 100.0)
    assert _parse_amount_tier("大于等于100万元，小于300万元") == (100.0, 300.0)
    assert _parse_amount_tier("大于等于500万元") == (500.0, None)
    assert _parse_amount_tier("---") == (None, None)
    assert _parse_amount_tier("") == (None, None)
    assert _parse_amount_tier("大于等于1000万元") == (1000.0, None)


def test_parse_period_tier() -> None:
    assert _parse_period_tier("小于7天") == (0.0, 7.0)
    assert _parse_period_tier("大于等于7天") == (7.0, None)
    assert _parse_period_tier("大于等于7天，小于365天") == (7.0, 365.0)
    assert _parse_period_tier("大于等于1年") == (365.0, None)
    assert _parse_period_tier("小于1年") == (0.0, 365.0)
    assert _parse_period_tier("---") == (None, None)
    assert _parse_period_tier("大于等于2年，小于3年") == (730.0, 1095.0)


def test_parse_fee_value() -> None:
    assert _parse_fee_value("0.30%") == ("0.30%", 0.30)
    assert _parse_fee_value("0.03%") == ("0.03%", 0.03)
    assert _parse_fee_value("每笔1000元") == ("每笔1000元", None)
    assert _parse_fee_value("") == ("", None)
    assert _parse_fee_value("1.50%") == ("1.50%", 1.50)


def test_choose_min_fee() -> None:
    row = pd.Series({
        "原费率": "0.30%",
        "天天基金优惠费率-银行卡购买": "0.03%",
        "天天基金优惠费率-活期宝购买": "0.03%",
    })
    fee_cols = ["原费率", "天天基金优惠费率-银行卡购买", "天天基金优惠费率-活期宝购买"]
    assert _choose_min_fee(row, fee_cols) == "0.03%"

    row2 = pd.Series({"原费率": "每笔1000元", "天天基金优惠费率-活期宝购买": "每笔1000元"})
    assert _choose_min_fee(row2, fee_cols) == "每笔1000元"


def test_load_fund_records(tmp_path: Path) -> None:
    csv_path = tmp_path / "purchase.csv"
    pd.DataFrame({
        "基金代码": ["000001", "000002", "000001"],
        "申购状态": ["开放申购", "开放申购", "开放申购"],
        "赎回状态": ["开放赎回", "开放赎回", "开放赎回"],
    }).to_csv(csv_path, index=False, encoding="utf-8-sig")
    records = _load_fund_records(csv_path)
    assert [r["基金编码"] for r in records] == ["000001", "000002"]
    assert records[0]["申购状态"] == "开放申购" and records[0]["赎回状态"] == "开放赎回"


def test_run_with_mocked_akshare(tmp_path: Path) -> None:
    purchase_csv = tmp_path / "fund_purchase.csv"
    purchase_csv.write_text(
        "基金代码,基金简称,申购状态,赎回状态\n000306,某基金,开放申购,开放赎回\n",
        encoding="utf-8-sig",
    )
    output_csv = tmp_path / "fund_fee_structured.csv"
    exception_log = tmp_path / "fund_fee_exceptions.csv"

    purchase_df = pd.DataFrame({
        "适用金额": ["小于100万元", "大于等于500万元"],
        "适用期限": ["---", "---"],
        "原费率": ["0.30%", "每笔1000元"],
        "天天基金优惠费率-活期宝购买": ["0.03%", "每笔1000元"],
    })
    redemption_df = pd.DataFrame({
        "适用金额": ["---", "---"],
        "适用期限": ["小于7天", "大于等于7天"],
        "赎回费率": ["1.50%", "0.00%"],
    })

    logger = logging.getLogger("test")

    def mock_fund_fee_em(symbol: str, indicator: str):
        if "申购" in indicator:
            return purchase_df
        return redemption_df

    with patch("akshare.fund_fee_em", side_effect=mock_fund_fee_em):
        run(purchase_csv, output_csv, exception_log, logger, request_delay=0)

    result = pd.read_csv(output_csv, dtype=str, encoding="utf-8-sig")
    assert list(result.columns) == [
        "基金编码", "申购状态", "赎回状态", "数据类型", "费率", "金额阶梯起点", "金额阶梯终点",
        "持仓期限阶梯起点", "持仓期限阶梯终点",
    ]
    assert len(result) == 4
    assert set(result["基金编码"]) == {"000306"}
    assert set(result["数据类型"]) == {"申购费率", "赎回费率"}
    fees = result["费率"].tolist()
    assert "0.03%" in fees
    assert "每笔1000元" in fees
    assert "1.50%" in fees
    assert "0.00%" in fees


@pytest.mark.parametrize("csv_content", [
    "基金代码,基金简称,申购状态,赎回状态\n000999,暂停基金,暂停申购,开放赎回\n",
    "基金代码,申购状态,赎回状态\n000999,暂停申购,开放赎回\n",
])
def test_run_skip_fee_query_when_not_open(tmp_path: Path, csv_content: str) -> None:
    """申购/赎回非开放时不查费率，输出空费率行；且不调用 akshare（有无基金简称列皆可）。"""
    purchase_csv = tmp_path / "fund_purchase.csv"
    purchase_csv.write_text(csv_content, encoding="utf-8-sig")
    output_csv = tmp_path / "fund_fee_structured.csv"
    exception_log = tmp_path / "fund_fee_exceptions.csv"
    logger = logging.getLogger("test")

    call_count = 0

    def track_calls(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return pd.DataFrame()

    with patch("akshare.fund_fee_em", side_effect=track_calls):
        run(purchase_csv, output_csv, exception_log, logger, request_delay=0)

    assert call_count == 0, "非开放基金不应调用 akshare"
    result = pd.read_csv(output_csv, dtype=str, encoding="utf-8-sig")
    assert len(result) == 2  # 申购费率、赎回费率 各一行
    assert all(result["基金编码"] == "000999")
    assert all(result["申购状态"] == "暂停申购") and all(result["赎回状态"] == "开放赎回")
    assert all(result["费率"].fillna("") == "")
    assert all(result["金额阶梯起点"].fillna("") == "")


def test_run_no_status_columns_queries_all(tmp_path: Path) -> None:
    """缺失申购状态/赎回状态列时，视为未知，全部查询（兼容旧版）。"""
    purchase_csv = tmp_path / "fund_purchase.csv"
    purchase_csv.write_text(
        "基金代码,基金简称\n000306,某基金\n",
        encoding="utf-8-sig",
    )
    output_csv = tmp_path / "fund_fee_structured.csv"
    exception_log = tmp_path / "fund_fee_exceptions.csv"
    logger = logging.getLogger("test")

    purchase_df = pd.DataFrame({
        "适用金额": ["小于100万元"],
        "适用期限": ["---"],
        "原费率": ["0.15%"],
    })
    redemption_df = pd.DataFrame({
        "适用金额": ["---", "---"],
        "适用期限": ["小于7天", "大于等于7天"],
        "赎回费率": ["1.50%", "0.00%"],
    })

    call_count = 0

    def mock_fund_fee_em(symbol: str, indicator: str):
        nonlocal call_count
        call_count += 1
        return purchase_df if "申购" in indicator else redemption_df

    with patch("akshare.fund_fee_em", side_effect=mock_fund_fee_em):
        run(purchase_csv, output_csv, exception_log, logger, request_delay=0)

    assert call_count == 2, "无状态列时应查询申购+赎回各 1 次"
    result = pd.read_csv(output_csv, dtype=str, encoding="utf-8-sig")
    assert len(result) >= 3
    assert all(result["基金编码"] == "000306")
    assert all(result["申购状态"].fillna("") == "") and all(result["赎回状态"].fillna("") == "")
