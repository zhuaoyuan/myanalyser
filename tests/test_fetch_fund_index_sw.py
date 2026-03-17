# -*- coding: utf-8 -*-
"""fetch_fund_index_sw 脚本的单元测试（mock akshare 避免网络请求）。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from fetch_fund_index_sw import fetch_index_hist, run


def _mock_index_hist(symbol: str, period: str = "day") -> pd.DataFrame:
    """模拟 AKShare index_hist_fund_sw 返回。"""
    return pd.DataFrame({
        "日期": ["2020-01-02", "2020-01-03", "2020-01-06"],
        "收盘指数": [100.0, 101.5, 99.8],
        "开盘指数": [0, 0, 0],
        "最高指数": [0, 0, 0],
        "最低指数": [0, 0, 0],
        "涨跌幅": [0.0, 1.5, -1.67],
    })


def test_run_with_mocked_akshare(tmp_path: Path) -> None:
    """run 正常返回时写入 CSV，列名正确。"""
    with patch("akshare.index_hist_fund_sw", side_effect=_mock_index_hist):
        result = run(output_dir=tmp_path, request_delay=0)

    assert len(result) == 3
    assert "807100" in result
    assert "807200" in result
    assert "807300" in result

    csv_path = result["807100"]
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    assert list(df.columns) == ["date", "close", "open", "high", "low", "pct_chg", "symbol", "name"]
    assert len(df) == 3
    assert str(df["symbol"].iloc[0]) == "807100"
    assert df["close"].iloc[0] == 100.0
    assert df["date"].iloc[0] == "2020-01-02"


def test_run_empty_result_raises(tmp_path: Path) -> None:
    """run 当某指数返回空时跳过该指数，但继续处理其他。"""
    call_count = 0

    def mock_sometimes_empty(symbol: str, period: str = "day"):
        nonlocal call_count
        call_count += 1
        if symbol == "807200":
            return pd.DataFrame()
        return _mock_index_hist(symbol, period)

    with patch("akshare.index_hist_fund_sw", side_effect=mock_sometimes_empty):
        result = run(output_dir=tmp_path, request_delay=0)

    assert "807200" not in result
    assert "807100" in result
    assert "807300" in result
