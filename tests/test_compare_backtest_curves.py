# -*- coding: utf-8 -*-
"""compare_backtest_curves 工具脚本单测。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import sys

_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "tools"
_MYANALYSER_ROOT = _SCRIPT_DIR.parent
_SRC = _MYANALYSER_ROOT / "src"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from compare_backtest_curves import (
    _load_equity_curve,
    _get_label,
    _compute_intersection_dates,
    _reindex_to_dates,
    run,
)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _make_equity_curve(dates: list[str], equities: list[float]) -> pd.DataFrame:
    base = equities[0] if equities[0] > 0 else 1.0
    cum_ret = [e / base - 1.0 for e in equities]
    return pd.DataFrame({
        "date": dates,
        "equity": equities,
        "cumulative_return": cum_ret,
    })


def _make_summary() -> pd.DataFrame:
    return pd.DataFrame({
        "section": ["config", "config", "metrics_holding"],
        "name": ["起始日期", "结束日期", "年化收益率"],
        "value": ["2024-01-02", "2024-01-10", "0.05"],
    })


def test_load_equity_curve(tmp_path: Path) -> None:
    """能加载 equity_curve.csv。"""
    eq = _make_equity_curve(["2024-01-02", "2024-01-03"], [100000.0, 101000.0])
    _write_csv(tmp_path / "equity_curve.csv", eq)
    df = _load_equity_curve(tmp_path)
    assert df is not None
    assert len(df) == 2
    assert "date" in df.columns and "equity" in df.columns


def test_load_equity_curve_missing(tmp_path: Path) -> None:
    """缺少 equity_curve.csv 返回 None。"""
    assert _load_equity_curve(tmp_path) is None


def test_get_label() -> None:
    """从路径提取标签。"""
    assert _get_label(Path("/a/b/chain")) == "chain"
    assert _get_label(Path("/a/b/807200_申万债券")) == "807200_申万债券"
    assert _get_label(Path("/a/b/保守型_A")) == "保守型_A"


def test_compute_intersection_dates() -> None:
    """起止对齐、过程中间取并集：范围 [max(start), min(end)]，范围内任一曲线有值即保留。"""
    df1 = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])})
    df2 = pd.DataFrame({"date": pd.to_datetime(["2024-01-03", "2024-01-04"])})
    curves = [("a", df1), ("b", df2)]
    common = _compute_intersection_dates(curves)
    assert common is not None
    assert len(common) == 2
    assert common[0].strftime("%Y-%m-%d") == "2024-01-03"
    assert common[1].strftime("%Y-%m-%d") == "2024-01-04"


def test_compute_intersection_dates_gap_in_one_curve() -> None:
    """某组合中间缺日：取并集后该日仍保留，缺值曲线用 ffill，不影响其他组合。"""
    df1 = pd.DataFrame({"date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"])})
    df2 = pd.DataFrame({"date": pd.to_datetime(["2024-01-03", "2024-01-05"])})  # 缺 01-04
    curves = [("a", df1), ("b", df2)]
    common = _compute_intersection_dates(curves)
    assert common is not None
    assert len(common) == 3  # 并集：01-03, 01-04, 01-05
    assert "2024-01-04" in [d.strftime("%Y-%m-%d") for d in common]


def test_reindex_to_dates() -> None:
    """对齐到目标日期。"""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "equity": [100000.0, 101000.0, 102000.0],
    })
    target = pd.DatetimeIndex(["2024-01-03", "2024-01-04"])
    out = _reindex_to_dates(df, target)
    assert len(out) == 2
    assert out.index[0].strftime("%Y-%m-%d") == "2024-01-03"
    assert float(out["equity"].iloc[0]) == 101000.0


def test_run_two_dirs(tmp_path: Path) -> None:
    """两目录对比：生成单一 backtest_curves.html，含 Summary 表与曲线图。"""
    # 目录 A：2024-01-02 ~ 2024-01-10
    dir_a = tmp_path / "chain"
    dir_a.mkdir()
    dates_a = [f"2024-01-{i:02d}" for i in range(2, 11)]
    eq_a = _make_equity_curve(dates_a, [100000.0 + i * 1000 for i in range(9)])
    _write_csv(dir_a / "equity_curve.csv", eq_a)
    _write_csv(dir_a / "summary.csv", _make_summary())

    # 目录 B：2024-01-03 ~ 2024-01-08（交集 2024-01-03 ~ 2024-01-08）
    dir_b = tmp_path / "保守型_A"
    dir_b.mkdir()
    dates_b = [f"2024-01-{i:02d}" for i in range(3, 9)]
    eq_b = _make_equity_curve(dates_b, [100000.0 + i * 500 for i in range(6)])
    _write_csv(dir_b / "equity_curve.csv", eq_b)
    _write_csv(dir_b / "summary.csv", _make_summary())

    out_dir = tmp_path / "output"
    result = run(
        backtest_dirs=[dir_a],
        base_dirs=[dir_b],
        output_dir=out_dir,
    )

    assert (out_dir / "backtest_curves.html").exists()
    html_content = (out_dir / "backtest_curves.html").read_text(encoding="utf-8")
    assert "Summary 对比表" in html_content
    assert "净值曲线" in html_content
    assert "chain" in html_content or "保守型_A" in html_content
    assert result["output"] == out_dir / "backtest_curves.html"


def test_run_empty_raises(tmp_path: Path) -> None:
    """无有效目录时抛出 ValueError。"""
    with pytest.raises(ValueError, match="无有效输入目录"):
        run(backtest_dirs=[], base_dirs=[], output_dir=tmp_path)


def test_run_no_intersection_raises(tmp_path: Path) -> None:
    """无日期交集时抛出 ValueError。"""
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    _write_csv(dir_a / "equity_curve.csv", _make_equity_curve(
        ["2024-01-02", "2024-01-03"], [100000.0, 101000.0]
    ))
    _write_csv(dir_a / "summary.csv", _make_summary())

    dir_b = tmp_path / "b"
    dir_b.mkdir()
    _write_csv(dir_b / "equity_curve.csv", _make_equity_curve(
        ["2024-02-01", "2024-02-02"], [100000.0, 101000.0]
    ))
    _write_csv(dir_b / "summary.csv", _make_summary())

    with pytest.raises(ValueError, match="无交集"):
        run(backtest_dirs=[dir_a], base_dirs=[dir_b], output_dir=tmp_path / "out")
