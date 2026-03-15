# -*- coding: utf-8 -*-
"""
compare_fund_etl_runs 单元测试。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# 需将 tools 加入 path 以导入 compare_fund_etl_runs
import sys
_root = Path(__file__).resolve().parent.parent
if str(_root / "tools") not in sys.path:
    sys.path.insert(0, str(_root / "tools"))

from compare_fund_etl_runs import (
    compare_csv_content,
    compare_overlap_time_series,
    file_hash,
    get_fund_codes,
)


def test_get_fund_codes_empty_dir(tmp_path: Path) -> None:
    assert get_fund_codes(tmp_path) == set()


def test_get_fund_codes_nonexistent(tmp_path: Path) -> None:
    assert get_fund_codes(tmp_path / "not_exists") == set()


def test_get_fund_codes_extracts_stem(tmp_path: Path) -> None:
    (tmp_path / "000001.csv").write_text("a,b\n1,2")
    (tmp_path / "163402.csv").write_text("x,y\n3,4")
    assert get_fund_codes(tmp_path) == {"000001", "163402"}


def test_file_hash_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "a.csv"
    f.write_text("x,y\n1,2")
    h1 = file_hash(f)
    h2 = file_hash(f)
    assert h1 == h2
    assert len(h1) == 64


def test_file_hash_differs_on_content(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("x\n1")
    (tmp_path / "b.csv").write_text("x\n2")
    assert file_hash(tmp_path / "a.csv") != file_hash(tmp_path / "b.csv")


def test_compare_csv_content_identical(tmp_path: Path) -> None:
    content = "基金代码,日期\n000001,2024-01-01"
    (tmp_path / "a.csv").write_text(content)
    (tmp_path / "b.csv").write_text(content)
    ok, msg = compare_csv_content(tmp_path / "a.csv", tmp_path / "b.csv")
    assert ok is True
    assert msg == "一致"


def test_compare_csv_content_shape_diff(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("a,b\n1,2")
    (tmp_path / "b.csv").write_text("a,b\n1,2\n3,4")
    ok, msg = compare_csv_content(tmp_path / "a.csv", tmp_path / "b.csv")
    assert ok is False
    assert "shape" in msg


def test_compare_csv_content_column_diff(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("a,b\n1,2")
    (tmp_path / "b.csv").write_text("a,c\n1,2")
    ok, msg = compare_csv_content(tmp_path / "a.csv", tmp_path / "b.csv")
    assert ok is False
    assert "列名" in msg


def test_compare_overlap_time_series_no_common_dates(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("日期,净值\n2024-01-01,1.0")
    (tmp_path / "b.csv").write_text("日期,净值\n2024-06-01,1.1")
    ok, msg = compare_overlap_time_series(
        tmp_path / "a.csv", tmp_path / "b.csv", "日期", "日期"
    )
    assert ok is False
    assert "无共同日期" in msg


def test_compare_overlap_time_series_identical_overlap(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("日期,净值\n2024-01-01,1.0\n2024-01-02,1.01")
    (tmp_path / "b.csv").write_text("日期,净值\n2024-01-01,1.0\n2024-01-02,1.01\n2024-06-01,1.5")
    ok, msg = compare_overlap_time_series(
        tmp_path / "a.csv", tmp_path / "b.csv", "日期", "日期"
    )
    assert ok is True
    assert "一致" in msg
