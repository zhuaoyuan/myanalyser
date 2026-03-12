"""回测引擎并行逻辑单元测试（_split_chunks、_filter_symbols_with_parallel）。

覆盖需求：20260312 most_stable 回测多线程优化与并行逻辑上移。
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

_ws = Path(__file__).resolve().parents[2]
if str(_ws) not in sys.path:
    sys.path.insert(0, str(_ws))

_src = _ws / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import pytest

from myanalyser.src.backtest.data import BacktestData
from myanalyser.src.backtest.engine import (
    UNIVERSE_PARALLEL_THRESHOLD,
    _filter_symbols_with_parallel,
    _split_chunks,
)
from myanalyser.src.backtest.filters import MostStableFilterStrategy, PassThroughFilter


def _make_backtest_data(
    n_symbols: int = 2, n_days: int = 800, trend_up: float = 0.05
) -> BacktestData:
    """构造回测数据。"""
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    rows = []
    by_symbol: dict[str, pd.DataFrame] = {}
    for i in range(n_symbols):
        symbol = str(i).zfill(6)
        nav = 1.0 + np.linspace(0, trend_up * (i + 1), len(dates))
        df_sym = pd.DataFrame({"date": dates, "close": nav})
        by_symbol[symbol] = df_sym
        chunk = df_sym.assign(
            symbol=symbol,
            open=df_sym["close"],
            high=df_sym["close"],
            low=df_sym["close"],
        )
        rows.append(chunk[["symbol", "date", "open", "high", "low", "close"]])
    long_df = pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    trading_dates = sorted(
        pd.Series(long_df["date"].unique()).dropna().map(lambda d: pd.Timestamp(d).normalize()).tolist()
    )
    return BacktestData(long_df=long_df, by_symbol=by_symbol, trading_dates=trading_dates)


# ==================== _split_chunks ====================


class TestSplitChunks:
    """_split_chunks 正常/边界/异常场景。"""

    # --- 正常场景 ---
    def test_split_chunks_normal_even_division(self) -> None:
        """正常：12 个元素均分 4 份，每份 3 个。"""
        lst = [f"s{i}" for i in range(12)]
        chunks = _split_chunks(lst, 4)
        assert len(chunks) == 4
        assert sum(len(c) for c in chunks) == 12
        for c in chunks:
            assert len(c) == 3

    def test_split_chunks_normal_uneven(self) -> None:
        """正常：10 个元素分 3 份，前两份 4 个、后一份 2 个。"""
        lst = [f"s{i}" for i in range(10)]
        chunks = _split_chunks(lst, 3)
        assert len(chunks) == 3
        assert sum(len(c) for c in chunks) == 10
        assert len(chunks[0]) == 4
        assert len(chunks[1]) == 4
        assert len(chunks[2]) == 2

    def test_split_chunks_normal_n_greater_than_len(self) -> None:
        """正常：5 个元素分 10 份，实际得 5 份。"""
        lst = [f"s{i}" for i in range(5)]
        chunks = _split_chunks(lst, 10)
        assert len(chunks) == 5
        assert [len(c) for c in chunks] == [1, 1, 1, 1, 1]

    def test_split_chunks_normal_single_element_per_chunk(self) -> None:
        """正常：4 个元素分 4 份，每份 1 个。"""
        lst = ["a", "b", "c", "d"]
        chunks = _split_chunks(lst, 4)
        assert chunks == [["a"], ["b"], ["c"], ["d"]]

    # --- 边界条件 ---
    def test_split_chunks_boundary_n_one_returns_whole(self) -> None:
        """边界：n=1 返回整列表为单块。"""
        lst = ["a", "b", "c"]
        chunks = _split_chunks(lst, 1)
        assert chunks == [["a", "b", "c"]]

    def test_split_chunks_boundary_n_zero_returns_whole(self) -> None:
        """边界：n=0 返回整列表为单块。"""
        lst = ["a", "b"]
        chunks = _split_chunks(lst, 0)
        assert chunks == [["a", "b"]]

    def test_split_chunks_boundary_empty_list_n_gt_1(self) -> None:
        """边界：空列表、n>1 返回空块列表。"""
        chunks = _split_chunks([], 4)
        assert chunks == []

    def test_split_chunks_boundary_empty_list_n_one(self) -> None:
        """边界：空列表、n<=1 返回 [[]]。"""
        chunks = _split_chunks([], 1)
        assert chunks == [[]]

    def test_split_chunks_boundary_single_element(self) -> None:
        """边界：单元素列表分多份。"""
        chunks = _split_chunks(["x"], 5)
        assert chunks == [["x"]]


# ==================== _filter_symbols_with_parallel ====================


class TestFilterSymbolsWithParallel:
    """_filter_symbols_with_parallel 正常/异常/边界场景。"""

    # --- 正常场景 ---
    def test_serial_path_universe_below_threshold(self) -> None:
        """正常：universe ≤ threshold 走串行路径，结果与直接调用 filter_symbols 一致。"""
        data = _make_backtest_data(10, 400)
        universe = sorted(data.by_symbol.keys())
        f = PassThroughFilter()
        as_of = data.trading_dates[-1]

        out = _filter_symbols_with_parallel(f, data, as_of, universe, threshold=100)
        direct = f.filter_symbols(data, as_of, universe)
        assert sorted(out) == sorted(direct)

    def test_parallel_path_universe_above_threshold(self) -> None:
        """正常：universe > threshold 走并行路径，结果与串行一致（顺序经 sorted 保证）。"""
        # 构造 150 只基金以触发并行
        data = _make_backtest_data(n_symbols=150, n_days=800)
        universe = sorted(data.by_symbol.keys())
        assert len(universe) > UNIVERSE_PARALLEL_THRESHOLD
        f = PassThroughFilter()
        as_of = data.trading_dates[-1]

        out = _filter_symbols_with_parallel(f, data, as_of, universe)
        direct = f.filter_symbols(data, as_of, universe)
        assert sorted(out) == sorted(direct)
        assert out == sorted(out)  # 输出已排序

    def test_parallel_most_stable_filter_consistency(self) -> None:
        """正常：MostStableFilterStrategy 在并行与串行下结果一致。"""
        data = _make_backtest_data(n_symbols=120, n_days=800, trend_up=0.12)
        universe = sorted(data.by_symbol.keys())
        f = MostStableFilterStrategy()
        as_of = data.trading_dates[-1]

        out_parallel = _filter_symbols_with_parallel(f, data, as_of, universe)
        out_serial = _filter_symbols_with_parallel(
            f, data, as_of, universe, threshold=9999
        )
        assert sorted(out_parallel) == sorted(out_serial)

    # --- 边界条件 ---
    def test_boundary_universe_exactly_threshold_serial(self) -> None:
        """边界：universe 长度等于 threshold 走串行。"""
        data = _make_backtest_data(n_symbols=UNIVERSE_PARALLEL_THRESHOLD, n_days=400)
        universe = sorted(data.by_symbol.keys())
        f = PassThroughFilter()
        as_of = data.trading_dates[-1]

        out = _filter_symbols_with_parallel(
            f, data, as_of, universe, threshold=UNIVERSE_PARALLEL_THRESHOLD
        )
        assert out == universe

    def test_boundary_universe_threshold_plus_one_parallel(self) -> None:
        """边界：universe 长度 = threshold + 1 走并行。"""
        data = _make_backtest_data(
            n_symbols=UNIVERSE_PARALLEL_THRESHOLD + 1, n_days=400
        )
        universe = sorted(data.by_symbol.keys())
        f = PassThroughFilter()
        as_of = data.trading_dates[-1]

        out = _filter_symbols_with_parallel(
            f, data, as_of, universe, threshold=UNIVERSE_PARALLEL_THRESHOLD
        )
        assert sorted(out) == universe

    def test_boundary_empty_universe(self) -> None:
        """边界：空 universe 返回空列表。"""
        data = _make_backtest_data(2, 100)
        f = PassThroughFilter()
        as_of = data.trading_dates[-1]

        out = _filter_symbols_with_parallel(f, data, as_of, [])
        assert out == []

    def test_boundary_max_workers_one(self) -> None:
        """边界：max_workers=1 仍可完成（单线程并行）。"""
        data = _make_backtest_data(n_symbols=150, n_days=400)
        universe = sorted(data.by_symbol.keys())
        f = PassThroughFilter()
        as_of = data.trading_dates[-1]

        out = _filter_symbols_with_parallel(
            f, data, as_of, universe, max_workers=1
        )
        assert sorted(out) == universe

    # --- 异常场景 ---
    def test_exception_filter_raises_propagates_runtime_error(self) -> None:
        """异常：filter_symbols 抛异常时，封装为带 chunk 信息的 RuntimeError。"""
        data = _make_backtest_data(150, 400)
        universe = sorted(data.by_symbol.keys())
        as_of = data.trading_dates[-1]

        mock_filter = MagicMock()
        mock_filter.filter_symbols.side_effect = ValueError("mock error")

        with pytest.raises(RuntimeError) as exc_info:
            _filter_symbols_with_parallel(
                mock_filter, data, as_of, universe,
                threshold=50,  # 触发并行
            )
        err = exc_info.value
        assert "filter_symbols failed" in str(err)
        assert "chunk" in str(err).lower()
        assert isinstance(err.__cause__, ValueError)
        assert "mock error" in str(err.__cause__)

    def test_exception_filter_raises_in_serial_path(self) -> None:
        """异常：串行路径下 filter_symbols 抛异常时直接传播。"""
        data = _make_backtest_data(5, 100)
        universe = sorted(data.by_symbol.keys())
        as_of = data.trading_dates[-1]

        mock_filter = MagicMock()
        mock_filter.filter_symbols.side_effect = ValueError("serial error")

        with pytest.raises(ValueError) as exc_info:
            _filter_symbols_with_parallel(
                mock_filter, data, as_of, universe,
                threshold=100,
            )
        assert "serial error" in str(exc_info.value)


# ==================== 并发/顺序确定性 ---

class TestParallelDeterminism:
    """并行结果顺序与确定性。"""

    def test_output_sorted_regardless_of_completion_order(self) -> None:
        """并行结果经 sorted 处理，与完成顺序无关。"""
        data = _make_backtest_data(n_symbols=200, n_days=400)
        universe = sorted(data.by_symbol.keys())
        f = PassThroughFilter()
        as_of = data.trading_dates[-1]

        out1 = _filter_symbols_with_parallel(f, data, as_of, universe)
        out2 = _filter_symbols_with_parallel(f, data, as_of, universe)
        assert out1 == out2
        assert out1 == sorted(out1)
