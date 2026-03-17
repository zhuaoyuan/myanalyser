# -*- coding: utf-8 -*-
"""chain_multi_t_backtest 工具脚本单测。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# 通过 conftest 或 pytest path 确保 tools/v2 在路径中
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "tools" / "v2"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from chain_multi_t_backtest import (
    _has_buy_orders,
    _list_t_dirs,
    _read_summary_metrics,
    chain,
)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _make_summary(start: str, end: str, initial_mv: float, end_mv: float) -> pd.DataFrame:
    return pd.DataFrame({
        "section": ["config", "config", "config", "metrics_pybroker", "metrics_pybroker"],
        "name": ["起始日期", "结束日期", "初始资金", "初始市值", "期末市值"],
        "value": [start, end, "100000", str(initial_mv), str(end_mv)],
    })


def _make_orders_with_buys() -> pd.DataFrame:
    return pd.DataFrame({
        "type": ["buy", "buy"],
        "symbol": ["000001", "000002"],
        "fill_date": ["2024-01-02", "2024-01-02"],
    })


def _make_orders_empty_buys() -> pd.DataFrame:
    return pd.DataFrame({"type": [], "symbol": [], "fill_date": []})


def _make_equity_curve(start: str, end: str, initial: float, end_val: float) -> pd.DataFrame:
    return pd.DataFrame({
        "date": [start, end],
        "equity": [initial, end_val],
        "cumulative_return": [0.0, end_val / initial - 1.0],
    })


def _make_period_detail() -> pd.DataFrame:
    return pd.DataFrame({
        "stat_date": ["2024-01-01"],
        "fill_date": ["2024-01-02"],
        "period_return": [0.01],
    })


def _make_positions_flat() -> pd.DataFrame:
    return pd.DataFrame({
        "stat_date": ["2024-01-01"],
        "symbol": ["000001"],
        "weight": [0.5],
        "rank": [1],
    })


def test_chain_two_periods(tmp_path: Path) -> None:
    """两期链式：T1 期末 110000 作为 T2 期初，缩放后连续。"""
    # T1: 2024-01-01 -> 2024-02-01, 100000 -> 110000
    t1 = tmp_path / "2024-01-01"
    t1.mkdir()
    _write_csv(t1 / "summary.csv", _make_summary("2024-01-01", "2024-02-01", 100000, 110000))
    _write_csv(t1 / "orders.csv", _make_orders_with_buys())
    _write_csv(t1 / "equity_curve.csv", _make_equity_curve("2024-01-01", "2024-02-01", 100000, 110000))
    _write_csv(t1 / "period_detail.csv", _make_period_detail())
    _write_csv(t1 / "positions_flat.csv", _make_positions_flat())

    # T2: 2024-02-01 -> 2024-03-01, 100000 -> 105000（期初应按 110000 缩放）
    t2 = tmp_path / "2024-02-01"
    t2.mkdir()
    _write_csv(t2 / "summary.csv", _make_summary("2024-02-01", "2024-03-01", 100000, 105000))
    _write_csv(t2 / "orders.csv", _make_orders_with_buys())
    _write_csv(t2 / "equity_curve.csv", _make_equity_curve("2024-02-01", "2024-03-01", 100000, 105000))
    _write_csv(t2 / "period_detail.csv", _make_period_detail())
    _write_csv(t2 / "positions_flat.csv", _make_positions_flat())

    out_dir = chain(tmp_path, "chain")

    assert out_dir.exists()
    assert (out_dir / "equity_curve.csv").exists()
    assert (out_dir / "summary.csv").exists()
    eq = pd.read_csv(out_dir / "equity_curve.csv")
    assert len(eq) >= 2
    # 首日 100000，末日应为 105000 * (110000/100000) = 115500
    first_eq = float(eq["equity"].iloc[0])
    last_eq = float(eq["equity"].iloc[-1])
    assert first_eq == 100000.0
    assert abs(last_eq - 115500) < 1.0  # T2 缩放后 105000 * 1.1 = 115500


def test_skip_t_without_buys(tmp_path: Path) -> None:
    """无买入的 T 被跳过。"""
    t1 = tmp_path / "2024-01-01"
    t1.mkdir()
    _write_csv(t1 / "summary.csv", _make_summary("2024-01-01", "2024-02-01", 100000, 110000))
    _write_csv(t1 / "orders.csv", _make_orders_empty_buys())  # 无买入
    _write_csv(t1 / "equity_curve.csv", _make_equity_curve("2024-01-01", "2024-02-01", 100000, 110000))

    with pytest.raises(ValueError, match="无有效 T 日可链式"):
        chain(tmp_path, "chain")


def test_has_buy_orders(tmp_path: Path) -> None:
    """_has_buy_orders 正确识别有/无买入。"""
    p = tmp_path / "orders.csv"
    _write_csv(p, _make_orders_with_buys())
    assert _has_buy_orders(p) is True
    _write_csv(p, _make_orders_empty_buys())
    assert _has_buy_orders(p) is False


def test_read_summary_metrics(tmp_path: Path) -> None:
    """_read_summary_metrics 提取 config 与 metrics。"""
    p = tmp_path / "summary.csv"
    _write_csv(p, _make_summary("2024-01-01", "2024-02-01", 100000, 110000))
    m = _read_summary_metrics(p)
    assert m.get("起始日期") == "2024-01-01"
    assert m.get("结束日期") == "2024-02-01"
    assert m.get("初始市值") == 100000.0
    assert m.get("期末市值") == 110000.0


def test_list_t_dirs(tmp_path: Path) -> None:
    """_list_t_dirs 按日期排序。"""
    (tmp_path / "2024-02-01").mkdir()
    (tmp_path / "2024-01-01").mkdir()
    (tmp_path / "not-date").mkdir()
    pairs = _list_t_dirs(tmp_path)
    assert len(pairs) == 2
    assert pairs[0][1] == "2024-01-01"
    assert pairs[1][1] == "2024-02-01"
