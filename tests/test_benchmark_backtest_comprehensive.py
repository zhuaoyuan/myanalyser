"""benchmark_portfolio_backtest 补充测试：覆盖边界条件、异常场景和额外正常路径。

目标：与 test_benchmark_portfolio_backtest.py（26 用例）互补，
覆盖需求日志中所有功能点的正常/异常/边界场景。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_MYANALYSER_ROOT = Path(__file__).resolve().parent.parent
_SRC = _MYANALYSER_ROOT / "src"
_TOOLS_V2 = _MYANALYSER_ROOT / "tools" / "v2"
for p in (_SRC, _TOOLS_V2):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmark_portfolio_backtest import (
    _per_fund_compare,
    _to_series,
    load_adjusted_nav,
    load_trade_calendar,
    parse_portfolio,
    simulate_portfolio,
    validate_compare,
    validate_integrity,
    write_outputs,
)
from run_benchmark_portfolios import (
    BENCHMARKS,
    _collect_summary,
    _derive_run_id,
)


def _make_nav(dates: list[pd.Timestamp], prices: list[float]) -> pd.Series:
    return pd.Series(prices, index=dates, dtype=float)


def _write_adj_csv(path: Path, code: str, dates: list[str], values: list[float]):
    rows = [
        {"基金代码": code, "净值日期": d, "单位净值": str(v),
         "复权净值": str(v), "cumulative_factor": "1.0"}
        for d, v in zip(dates, values)
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _write_cum_csv(path: Path, dates: list[str], values: list[float]):
    rows = [{"日期": d, "累计收益率": str(v)} for d, v in zip(dates, values)]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


# ============================================================================
# parse_portfolio — 边界条件
# ============================================================================

class TestParsePortfolioBoundary:
    def test_weight_at_exact_tolerance_pass(self):
        """权重和 1.0 + 5e-7 在容差 1e-6 内 → 应通过。"""
        result = parse_portfolio("161119:0.5000005,510300:0.5")
        assert len(result) == 2

    def test_weight_just_beyond_tolerance_fail(self):
        """权重和 1.0 + 2e-6 超出容差 → 应拒绝。"""
        with pytest.raises(ValueError, match="权重之和"):
            parse_portfolio("161119:0.500001,510300:0.500001")

    def test_weight_zero_for_one_fund(self):
        """单只基金权重为 0，总和仍为 1 → 格式合法。"""
        result = parse_portfolio("161119:0.0,510300:1.0")
        assert result[0][1] == 0.0
        assert result[1][1] == 1.0

    def test_non_numeric_weight_raises(self):
        """非数值权重 → ValueError。"""
        with pytest.raises(ValueError):
            parse_portfolio("161119:abc,510300:0.5")

    def test_many_funds(self):
        """10 只基金均分权重 → 应正确解析。"""
        parts = [f"{100000 + i}:0.10" for i in range(10)]
        result = parse_portfolio(",".join(parts))
        assert len(result) == 10
        total = sum(w for _, w in result)
        assert abs(total - 1.0) < 1e-6


# ============================================================================
# simulate_portfolio — 边界与异常
# ============================================================================

class TestSimulatePortfolioBoundaryAndError:
    @pytest.fixture()
    def dates_5(self) -> list[pd.Timestamp]:
        return pd.bdate_range("2024-01-01", periods=5).tolist()

    def test_rebalance_every_day(self, dates_5):
        """rebalance_interval=1 → 每天再平衡，不应报错。"""
        nav = {
            "000001": _make_nav(dates_5, [1.0, 1.2, 1.1, 1.3, 1.4]),
            "000002": _make_nav(dates_5, [2.0, 2.0, 2.0, 2.0, 2.0]),
        }
        eq = simulate_portfolio(
            nav, [("000001", 0.5), ("000002", 0.5)], dates_5, 1, 1000.0,
        )
        assert len(eq) == 5
        assert float(eq["equity"].iloc[0]) == pytest.approx(1000.0)

    def test_negative_price_first_day_raises(self, dates_5):
        """首日价格为负 → ValueError。"""
        nav = {"000001": _make_nav(dates_5, [-1.0, 1.0, 2.0, 3.0, 4.0])}
        with pytest.raises(ValueError, match="首日净值为零或负值"):
            simulate_portfolio(nav, [("000001", 1.0)], dates_5, 9999, 100.0)

    def test_single_day(self):
        """仅 1 个交易日 → 正常返回 1 行，cumulative_return = 0。"""
        dates = pd.bdate_range("2024-01-01", periods=1).tolist()
        nav = {"000001": _make_nav(dates, [10.0])}
        eq = simulate_portfolio(nav, [("000001", 1.0)], dates, 9999, 100.0)
        assert len(eq) == 1
        assert float(eq["cumulative_return"].iloc[0]) == pytest.approx(0.0)

    def test_large_capital(self):
        """极大初始资金（1e15）不应产生浮点溢出。"""
        dates = pd.bdate_range("2024-01-01", periods=5).tolist()
        nav = {"000001": _make_nav(dates, [1.0, 1.1, 1.2, 1.3, 1.4])}
        eq = simulate_portfolio(nav, [("000001", 1.0)], dates, 9999, 1e15)
        assert len(eq) == 5
        assert np.isfinite(eq["equity"].iloc[-1])

    def test_all_nan_fund_raises(self):
        """某基金在日期范围内全为 NaN → ValueError。"""
        dates = pd.bdate_range("2024-01-01", periods=5).tolist()
        nav = {"000001": _make_nav(dates, [float("nan")] * 5)}
        with pytest.raises(ValueError, match="无任何净值数据"):
            simulate_portfolio(nav, [("000001", 1.0)], dates, 9999, 100.0)

    def test_rebalance_interval_zero_raises(self):
        """rebalance_interval=0 → ZeroDivisionError（缺乏参数校验）。"""
        dates = pd.bdate_range("2024-01-01", periods=5).tolist()
        nav = {"000001": _make_nav(dates, [1.0] * 5)}
        with pytest.raises(ZeroDivisionError):
            simulate_portfolio(nav, [("000001", 1.0)], dates, 0, 100.0)

    def test_ffill_partial_coverage(self, dates_5):
        """仅首尾有数据，中间缺失 → ffill 后连续。"""
        nav_a = _make_nav([dates_5[0], dates_5[4]], [1.0, 2.0])
        nav_b = _make_nav(dates_5, [10.0] * 5)
        nav = {"000001": nav_a, "000002": nav_b}
        eq = simulate_portfolio(
            nav, [("000001", 0.5), ("000002", 0.5)], dates_5, 9999, 100.0,
        )
        assert len(eq) == 5
        assert not eq["equity"].isna().any()


# ============================================================================
# _to_series — 正常与边界
# ============================================================================

class TestToSeries:
    def test_normal(self):
        """正常数据 → 返回有值的 Series。"""
        df = pd.DataFrame({"日期": ["2024-01-02", "2024-01-03"], "值": ["1.5", "2.5"]})
        s = _to_series(df, "日期", "值", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
        assert len(s) == 2
        assert float(s.iloc[0]) == pytest.approx(1.5)

    def test_all_non_numeric_returns_empty(self):
        """所有值为非数值字符串 → coerce 后全 NaN → 空 Series。"""
        df = pd.DataFrame({"日期": ["2024-01-02", "2024-01-03"], "值": ["abc", "def"]})
        s = _to_series(df, "日期", "值", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
        assert s.empty

    def test_dates_outside_range(self):
        """所有日期在查询范围外 → 空 Series。"""
        df = pd.DataFrame({"日期": ["2025-06-01", "2025-06-02"], "值": ["1.0", "2.0"]})
        s = _to_series(df, "日期", "值", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
        assert s.empty


# ============================================================================
# _per_fund_compare — 边界条件
# ============================================================================

class TestPerFundCompareBoundary:
    def test_exactly_two_common_dates(self, tmp_path: Path):
        """恰好 2 个公共日期（最小可比对长度）→ 返回有效 ratio。"""
        dates = ["2024-01-02", "2024-01-03"]
        adj_csv = tmp_path / "adj.csv"
        cum_csv = tmp_path / "cum.csv"
        _write_adj_csv(adj_csv, "000001", dates, [1.0, 1.01])
        _write_cum_csv(cum_csv, dates, [0.0, 1.0])
        ratio = _per_fund_compare(
            adj_csv, cum_csv,
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"),
        )
        assert ratio is not None

    def test_single_common_date_returns_none(self, tmp_path: Path):
        """仅 1 个公共日期（< 2）→ 返回 None。"""
        adj_csv = tmp_path / "adj.csv"
        cum_csv = tmp_path / "cum.csv"
        _write_adj_csv(adj_csv, "000001", ["2024-01-02"], [1.0])
        _write_cum_csv(cum_csv, ["2024-01-02"], [0.0])
        ratio = _per_fund_compare(
            adj_csv, cum_csv,
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"),
        )
        assert ratio is None

    def test_local_s_zero_all_skipped(self, tmp_path: Path):
        """所有公共日期 local_s <= 0 → total=0 → None。"""
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        adj_csv = tmp_path / "adj.csv"
        cum_csv = tmp_path / "cum.csv"
        _write_adj_csv(adj_csv, "000001", dates, [0.0, 0.0, 1.0])
        _write_cum_csv(cum_csv, dates, [0.0, 0.0, 1.0])
        ratio = _per_fund_compare(
            adj_csv, cum_csv,
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"),
        )
        assert ratio is None

    def test_denom_near_zero_skipped(self, tmp_path: Path):
        """remote_s + 100 ≈ 0（remote_s ≈ -100）→ 该点跳过，total=0 → None。"""
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        adj_csv = tmp_path / "adj.csv"
        cum_csv = tmp_path / "cum.csv"
        _write_adj_csv(adj_csv, "000001", dates, [1.0, 1.0, 1.01])
        _write_cum_csv(cum_csv, dates, [-100.0, -100.0, 0.0])
        ratio = _per_fund_compare(
            adj_csv, cum_csv,
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"),
        )
        assert ratio is None

    def test_both_returns_near_zero_good(self, tmp_path: Path):
        """两端收益率都接近零 → 计为 good → ratio = 1.0。"""
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        adj_csv = tmp_path / "adj.csv"
        cum_csv = tmp_path / "cum.csv"
        _write_adj_csv(adj_csv, "000001", dates, [1.0, 1.0, 1.0])
        _write_cum_csv(cum_csv, dates, [0.0, 0.0, 0.0])
        ratio = _per_fund_compare(
            adj_csv, cum_csv,
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"),
        )
        assert ratio is not None
        assert ratio == pytest.approx(1.0)


# ============================================================================
# validate_compare — 额外异常场景
# ============================================================================

class TestValidateCompareExtra:
    def test_adj_missing_cum_exists(self, tmp_path: Path):
        """复权净值文件不存在但 cum_return 存在 → 错误列表包含 '不存在'。"""
        nav_dir = tmp_path / "fund_adjusted_nav_by_code"
        nav_dir.mkdir()
        cum_dir = tmp_path / "fund_cum_return_by_code"
        cum_dir.mkdir()
        _write_cum_csv(cum_dir / "000001.csv", ["2024-01-02"], [0.0])
        errs = validate_compare(tmp_path, ["000001"], "2024-01-02", "2024-01-05", 0.80)
        assert len(errs) == 1
        assert "不存在" in errs[0]


# ============================================================================
# validate_integrity — 额外异常场景
# ============================================================================

class TestValidateIntegrityExtra:
    def _write_trade_dates(self, path: Path, dates: list[str]):
        pd.DataFrame({"trade_date": dates}).to_csv(path, index=False, encoding="utf-8-sig")

    def test_no_trade_days_in_range(self, tmp_path: Path):
        """交易日历在给定范围内无交易日 → 返回错误。"""
        td_csv = tmp_path / "trade_dates.csv"
        self._write_trade_dates(td_csv, ["2025-06-01", "2025-06-02"])
        nav_dir = tmp_path / "fund_adjusted_nav_by_code"
        nav_dir.mkdir()
        errs = validate_integrity(
            tmp_path, ["000001"], "2024-01-01", "2024-12-31", 0.95, td_csv,
        )
        assert len(errs) == 1
        assert "无交易日" in errs[0]


# ============================================================================
# load_adjusted_nav — 边界
# ============================================================================

class TestLoadAdjustedNav:
    def _write_nav(self, path: Path, code: str, dates: list[str], values: list[float]):
        rows = [
            {"基金代码": code, "净值日期": d, "单位净值": str(v),
             "复权净值": str(v), "cumulative_factor": "1.0"}
            for d, v in zip(dates, values)
        ]
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

    def test_duplicate_dates_keep_last(self, tmp_path: Path):
        """同一日期有多条记录 → 保留最后一条。"""
        nav_dir = tmp_path / "fund_adjusted_nav_by_code"
        nav_dir.mkdir()
        self._write_nav(
            nav_dir / "000001.csv", "000001",
            ["2024-01-02", "2024-01-02", "2024-01-03"],
            [1.0, 1.5, 2.0],
        )
        result = load_adjusted_nav(tmp_path, ["000001"], "2024-01-01", "2024-12-31")
        assert "000001" in result
        s = result["000001"]
        assert len(s) == 2
        assert float(s.iloc[0]) == pytest.approx(1.5)


# ============================================================================
# load_trade_calendar — 边界
# ============================================================================

class TestLoadTradeCalendar:
    def test_empty_range(self, tmp_path: Path):
        """日期范围内无交易日 → 空列表。"""
        td_csv = tmp_path / "trade_dates.csv"
        pd.DataFrame({"trade_date": ["2025-01-02"]}).to_csv(
            td_csv, index=False, encoding="utf-8-sig",
        )
        result = load_trade_calendar(td_csv, "2024-01-01", "2024-12-31")
        assert result == []

    def test_normal_range(self, tmp_path: Path):
        """正常范围 → 返回匹配的交易日。"""
        td_csv = tmp_path / "trade_dates.csv"
        pd.DataFrame({"trade_date": ["2024-01-02", "2024-01-03", "2024-01-04"]}).to_csv(
            td_csv, index=False, encoding="utf-8-sig",
        )
        result = load_trade_calendar(td_csv, "2024-01-02", "2024-01-03")
        assert len(result) == 2


# ============================================================================
# write_outputs — 边界
# ============================================================================

class TestWriteOutputsEdge:
    def test_minimal_two_rows(self, tmp_path: Path):
        """最小有效 equity curve（2 行） → 正常生成全部产物。"""
        dates = pd.bdate_range("2024-01-01", periods=2).tolist()
        eq = pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "equity": [100.0, 110.0],
            "cumulative_return": [0.0, 0.1],
        })
        write_outputs(
            eq, [("000001", 1.0)], tmp_path,
            "2024-01-01", "2024-01-02", 243, 100.0,
        )
        assert (tmp_path / "equity_curve.csv").exists()
        assert (tmp_path / "summary.csv").exists()
        assert (tmp_path / "backtest_report.md").exists()


# ============================================================================
# run_benchmark_portfolios 辅助函数
# ============================================================================

class TestRunBenchmarkPortfoliosHelpers:
    def test_derive_run_id_fund_etl(self, tmp_path: Path):
        """fund_etl 叶目录 → 取父目录名。"""
        p = tmp_path / "20260315_123456_full_run" / "fund_etl"
        p.mkdir(parents=True)
        assert _derive_run_id(p) == "20260315_123456_full_run"

    def test_derive_run_id_other(self, tmp_path: Path):
        """非 fund_etl 叶目录 → 取当前目录名。"""
        p = tmp_path / "my_data"
        p.mkdir()
        assert _derive_run_id(p) == "my_data"

    def test_all_benchmarks_parse_correctly(self):
        """8 个硬编码组合定义全部能被 parse_portfolio 正确解析。"""
        assert len(BENCHMARKS) == 8
        for name, portfolio_str in BENCHMARKS:
            result = parse_portfolio(portfolio_str)
            total = sum(w for _, w in result)
            assert abs(total - 1.0) < 1e-6, f"{name} 权重和 != 1.0"

    def test_collect_summary_empty(self, tmp_path: Path):
        """无 summary.csv → 返回 None。"""
        result = _collect_summary(tmp_path)
        assert result is None

    def test_collect_summary_with_data(self, tmp_path: Path):
        """有 summary.csv → 返回 DataFrame。"""
        for name, _ in BENCHMARKS[:2]:
            d = tmp_path / name
            d.mkdir()
            pd.DataFrame([
                {"section": "metrics_holding", "name": "年化收益率", "value": "0.05"},
                {"section": "config", "name": "组合", "value": name},
            ]).to_csv(d / "summary.csv", index=False, encoding="utf-8-sig")
        result = _collect_summary(tmp_path)
        assert result is not None
        assert len(result) == 2
