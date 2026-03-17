"""benchmark_portfolio_backtest 单元测试。"""
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
    parse_portfolio,
    simulate_portfolio,
    validate_compare,
    validate_integrity,
    write_outputs,
)


# ---------------------------------------------------------------------------
# parse_portfolio
# ---------------------------------------------------------------------------

class TestParsePortfolio:
    def test_single_fund(self):
        result = parse_portfolio("161119:1.00")
        assert result == [("161119", 1.0)]

    def test_two_funds(self):
        result = parse_portfolio("161119:0.70, 510300:0.30")
        assert len(result) == 2
        assert result[0] == ("161119", 0.70)
        assert result[1] == ("510300", 0.30)

    def test_four_funds(self):
        result = parse_portfolio("161119:0.15,510300:0.50,510500:0.20,159915:0.15")
        assert len(result) == 4
        total = sum(w for _, w in result)
        assert abs(total - 1.0) < 0.01

    def test_zero_padded_code(self):
        result = parse_portfolio("1234:0.50, 5678:0.50")
        assert result[0][0] == "001234"
        assert result[1][0] == "005678"

    def test_weight_not_sum_to_one(self):
        with pytest.raises(ValueError, match="权重之和"):
            parse_portfolio("161119:0.50,510300:0.30")

    def test_duplicate_code(self):
        with pytest.raises(ValueError, match="重复基金代码"):
            parse_portfolio("161119:0.50,161119:0.50")

    def test_empty(self):
        with pytest.raises(ValueError, match="组合为空"):
            parse_portfolio("")

    def test_bad_format(self):
        with pytest.raises(ValueError, match="格式错误"):
            parse_portfolio("161119-0.50")


# ---------------------------------------------------------------------------
# simulate_portfolio
# ---------------------------------------------------------------------------

def _make_nav(dates: list[pd.Timestamp], prices: list[float]) -> pd.Series:
    return pd.Series(prices, index=dates, dtype=float)


class TestSimulatePortfolio:
    @pytest.fixture()
    def dates_10(self) -> list[pd.Timestamp]:
        return pd.bdate_range("2024-01-01", periods=10).tolist()

    def test_no_rebalance_single_fund(self, dates_10):
        """单只基金、不再平衡 → equity = price * initial_shares。"""
        prices = [float(i + 1) for i in range(10)]
        nav = {"000001": _make_nav(dates_10, prices)}
        portfolio = [("000001", 1.0)]
        eq = simulate_portfolio(nav, portfolio, dates_10, 9999, 100.0)
        assert len(eq) == 10
        assert float(eq["equity"].iloc[0]) == pytest.approx(100.0)
        assert float(eq["equity"].iloc[-1]) == pytest.approx(100.0 * 10.0 / 1.0)

    def test_no_rebalance_two_funds(self, dates_10):
        """两只基金、不再平衡 → 各自按初始份额跟踪。"""
        prices_a = [1.0 + 0.1 * i for i in range(10)]  # 1.0 → 1.9
        prices_b = [2.0] * 10                            # flat
        nav = {
            "000001": _make_nav(dates_10, prices_a),
            "000002": _make_nav(dates_10, prices_b),
        }
        portfolio = [("000001", 0.5), ("000002", 0.5)]
        eq = simulate_portfolio(nav, portfolio, dates_10, 9999, 100.0)
        shares_a = 50.0 / 1.0
        shares_b = 50.0 / 2.0
        expected_last = shares_a * 1.9 + shares_b * 2.0
        assert float(eq["equity"].iloc[-1]) == pytest.approx(expected_last, rel=1e-6)

    def test_rebalance_effect(self, dates_10):
        """验证再平衡实际改变了份额分配。"""
        prices_a = [1.0 + 0.2 * i for i in range(10)]  # trending up
        prices_b = [10.0] * 10
        nav = {
            "000001": _make_nav(dates_10, prices_a),
            "000002": _make_nav(dates_10, prices_b),
        }
        portfolio = [("000001", 0.5), ("000002", 0.5)]

        eq_no_rb = simulate_portfolio(nav, portfolio, dates_10, 9999, 1000.0)
        eq_rb_5 = simulate_portfolio(nav, portfolio, dates_10, 5, 1000.0)

        last_no_rb = float(eq_no_rb["equity"].iloc[-1])
        last_rb = float(eq_rb_5["equity"].iloc[-1])
        assert last_no_rb != pytest.approx(last_rb, rel=1e-4)

    def test_cumulative_return(self, dates_10):
        nav = {"000001": _make_nav(dates_10, [1.0] * 5 + [2.0] * 5)}
        eq = simulate_portfolio(nav, [("000001", 1.0)], dates_10, 9999, 100.0)
        assert float(eq["cumulative_return"].iloc[0]) == pytest.approx(0.0)
        assert float(eq["cumulative_return"].iloc[-1]) == pytest.approx(1.0)

    def test_forward_fill(self):
        """基金数据有缺失时，前向填充应保证模拟连续。"""
        dates = pd.bdate_range("2024-01-01", periods=5).tolist()
        nav_a = _make_nav([dates[0], dates[2], dates[4]], [1.0, 1.5, 2.0])
        nav_b = _make_nav(dates, [10.0] * 5)
        nav = {"000001": nav_a, "000002": nav_b}
        portfolio = [("000001", 0.5), ("000002", 0.5)]
        eq = simulate_portfolio(nav, portfolio, dates, 9999, 100.0)
        assert len(eq) == 5
        assert not eq["equity"].isna().any()


# ---------------------------------------------------------------------------
# validate_integrity (with synthetic data)
# ---------------------------------------------------------------------------

class TestValidateIntegrity:
    def _write_nav_csv(self, path: Path, code: str, dates: list[str]):
        rows = [{"基金代码": code, "净值日期": d, "单位净值": "1.0",
                 "复权净值": "1.0", "cumulative_factor": "1.0"} for d in dates]
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

    def _write_trade_dates(self, path: Path, dates: list[str]):
        pd.DataFrame({"trade_date": dates}).to_csv(path, index=False, encoding="utf-8-sig")

    def test_pass(self, tmp_path: Path):
        trade_dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        nav_dir = tmp_path / "fund_adjusted_nav_by_code"
        nav_dir.mkdir()
        self._write_nav_csv(nav_dir / "000001.csv", "000001", trade_dates)
        td_csv = tmp_path / "trade_dates.csv"
        self._write_trade_dates(td_csv, trade_dates)

        errs = validate_integrity(
            tmp_path, ["000001"], "2024-01-02", "2024-01-05", 0.95, td_csv,
        )
        assert errs == []

    def test_fail_missing_data(self, tmp_path: Path):
        trade_dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        nav_dir = tmp_path / "fund_adjusted_nav_by_code"
        nav_dir.mkdir()
        self._write_nav_csv(nav_dir / "000001.csv", "000001", trade_dates[:2])
        td_csv = tmp_path / "trade_dates.csv"
        self._write_trade_dates(td_csv, trade_dates)

        errs = validate_integrity(
            tmp_path, ["000001"], "2024-01-02", "2024-01-05", 0.95, td_csv,
        )
        assert len(errs) == 1
        assert "数据完整比例" in errs[0]

    def test_fail_missing_file(self, tmp_path: Path):
        nav_dir = tmp_path / "fund_adjusted_nav_by_code"
        nav_dir.mkdir()
        td_csv = tmp_path / "trade_dates.csv"
        self._write_trade_dates(td_csv, ["2024-01-02"])

        errs = validate_integrity(
            tmp_path, ["000001"], "2024-01-02", "2024-01-02", 0.95, td_csv,
        )
        assert len(errs) == 1
        assert "不存在" in errs[0]


# ---------------------------------------------------------------------------
# validate_compare
# ---------------------------------------------------------------------------

class TestValidateCompare:
    def test_skip_when_no_cum_return_dir(self, tmp_path: Path):
        """无 cum_return 目录 → 降级跳过，返回空错误列表。"""
        nav_dir = tmp_path / "fund_adjusted_nav_by_code"
        nav_dir.mkdir()
        errs = validate_compare(tmp_path, ["000001"], "2024-01-02", "2024-01-05", 0.80)
        assert errs == []

    def test_skip_when_fund_has_no_cum_csv(self, tmp_path: Path):
        nav_dir = tmp_path / "fund_adjusted_nav_by_code"
        nav_dir.mkdir()
        cum_dir = tmp_path / "fund_cum_return_by_code"
        cum_dir.mkdir()
        errs = validate_compare(tmp_path, ["000001"], "2024-01-02", "2024-01-05", 0.80)
        assert errs == []


# ---------------------------------------------------------------------------
# write_outputs
# ---------------------------------------------------------------------------

class TestWriteOutputs:
    def test_files_created(self, tmp_path: Path):
        dates = pd.bdate_range("2024-01-01", periods=30).tolist()
        eq = pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "equity": np.linspace(100, 120, 30),
            "cumulative_return": np.linspace(0.0, 0.2, 30),
        })
        portfolio = [("000001", 0.6), ("000002", 0.4)]
        write_outputs(eq, portfolio, tmp_path, "2024-01-01", "2024-02-09", 20, 100.0)

        assert (tmp_path / "equity_curve.csv").exists()
        assert (tmp_path / "summary.csv").exists()
        assert (tmp_path / "backtest_report.md").exists()

        summary = pd.read_csv(tmp_path / "summary.csv", dtype=str, encoding="utf-8-sig")
        assert "section" in summary.columns
        assert "metrics_holding" in summary["section"].values
