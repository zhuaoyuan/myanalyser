# -*- coding: utf-8 -*-
"""fetch_fund_index_sw 脚本的单元测试（mock akshare 避免网络请求）。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from fetch_fund_index_sw import _filter_by_date_range, fetch_index_hist, run


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


def test_filter_by_date_range() -> None:
    """_filter_by_date_range 双闭区间过滤正确。"""
    df = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"],
        "close": [1, 2, 3, 4],
    })
    out = _filter_by_date_range(df, "2020-01-02", "2020-01-03")
    assert list(out["date"]) == ["2020-01-02", "2020-01-03"]
    assert len(_filter_by_date_range(df, None, None)) == 4
    assert len(_filter_by_date_range(df, "2020-01-01", None)) == 4
    assert len(_filter_by_date_range(df, None, "2020-01-06")) == 4


def test_run_with_mocked_akshare(tmp_path: Path) -> None:
    """run 正常返回时每个指数写入独立子目录，产出与 backtest_base 一致的四类文件。"""
    with patch("akshare.index_hist_fund_sw", side_effect=_mock_index_hist):
        result = run(
            output_root=tmp_path,
            run_id="test_run",
            start_date="2020-01-01",
            end_date="2020-01-10",
            request_delay=0,
        )

    assert len(result) == 3
    assert "807100" in result
    subdir = result["807100"].parent
    assert "807100_申万权益" in str(subdir)
    assert (subdir / "equity_curve.csv").exists()
    assert (subdir / "summary.csv").exists()
    assert (subdir / "backtest_report.md").exists()
    assert (subdir / "backtest_curves.html").exists()

    df = pd.read_csv(subdir / "equity_curve.csv", encoding="utf-8-sig")
    assert list(df.columns) == ["date", "equity", "cumulative_return"]
    assert len(df) == 3
    assert df["date"].iloc[0] == "2020-01-02"
    assert df["equity"].iloc[0] == pytest.approx(1.0)


def test_run_empty_result_skips(tmp_path: Path) -> None:
    """run 当某指数返回空时跳过该指数，但继续处理其他。"""
    call_count = 0

    def mock_sometimes_empty(symbol: str, period: str = "day"):
        nonlocal call_count
        call_count += 1
        if symbol == "807200":
            return pd.DataFrame()
        return _mock_index_hist(symbol, period)

    with patch("akshare.index_hist_fund_sw", side_effect=mock_sometimes_empty):
        result = run(
            output_root=tmp_path,
            run_id="test_run",
            request_delay=0,
        )

    assert "807200" not in result
    assert "807100" in result
    assert "807300" in result
