# -*- coding: utf-8 -*-
"""fetch_fund_fee 脚本的单元测试（解析逻辑 + 主流程 mock）。"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from fetch_fund_fee import (
    _parse_amount_tier,
    _parse_period_tier,
    _parse_fee_value,
    _choose_min_fee,
    _load_fund_codes,
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


def test_load_fund_codes(tmp_path: Path) -> None:
    csv_path = tmp_path / "purchase.csv"
    pd.DataFrame({"基金代码": ["000001", "000002", "000001"]}).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )
    codes = _load_fund_codes(csv_path)
    assert codes == ["000001", "000002"]


def test_run_with_mocked_akshare(tmp_path: Path) -> None:
    purchase_csv = tmp_path / "fund_purchase.csv"
    purchase_csv.write_text(
        "基金代码,基金简称,申购状态,赎回状态\n000306,某基金,开放,开放\n",
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

    with patch("fetch_fund_fee.ak") as mock_ak:
        mock_ak.fund_fee_em = mock_fund_fee_em
        run(purchase_csv, output_csv, exception_log, logger, request_delay=0)

    result = pd.read_csv(output_csv, dtype=str, encoding="utf-8-sig")
    assert list(result.columns) == [
        "基金编码", "数据类型", "费率", "金额阶梯起点", "金额阶梯终点",
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
