"""pybroker 回测策略包化 - 综合单元测试（正常/异常/边界场景）。"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_ws = Path(__file__).resolve().parents[2]
if str(_ws) not in sys.path:
    sys.path.insert(0, str(_ws))

# 确保 src 在 path 中（与 CLI 一致）
_src = _ws / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from myanalyser.src.backtest.data import BacktestData, load_fund_nav_data
from myanalyser.src.backtest.filters import (
    apply_filter_chain,
    get_filter_chain,
)
from myanalyser.src.backtest.filters.filtered_candidates_csv import (
    ENV_VAR as FILTERED_ENV,
    FilteredCandidatesCsvFilter,
)
from myanalyser.src.backtest.filters.max_funds import (
    ENV_VAR as MAX_FUNDS_ENV,
    MaxFundsFilter,
)
from myanalyser.src.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
    write_reports,
)
from myanalyser.src.backtest.metrics import (
    WindowConfig,
    compute_low_risk_debt_metrics,
)
from myanalyser.src.backtest.filters import PassThroughFilter
from myanalyser.src.backtest.strategies.low_risk_debt import (
    EqualWeightPosition,
    LowRiskDebtScoreStrategy,
    build_bundle,
)
from myanalyser.src.backtest.strategies.registry import (
    get_strategy_bundle,
    list_strategy_names,
)


def _make_fund_nav_csv(path: Path, code: str, dates: list[str], navs: list[float]) -> None:
    rows = [{"基金代码": code, "净值日期": d, "复权净值": v} for d, v in zip(dates, navs)]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _make_backtest_data(
    n_symbols: int = 2, n_days: int = 800, trend_up: float = 0.05
) -> BacktestData:
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


# ==================== 正常场景 ====================


class TestNormalScenarios(unittest.TestCase):
    """正常场景：标准输入下的预期行为。"""

    def test_score_strategy_prefers_stronger_trend(self) -> None:
        """正常：上升趋势更强的基金得分更高、排名靠前。"""
        dates = pd.date_range("2020-01-01", periods=800, freq="B")
        nav_up = 1.0 + np.linspace(0, 1.0, len(dates))
        nav_flat = 1.0 + np.linspace(0, 0.05, len(dates))
        df_up = pd.DataFrame({"date": dates, "close": nav_up})
        df_flat = pd.DataFrame({"date": dates, "close": nav_flat})
        long_rows = []
        for symbol, df in [("000001", df_up), ("000002", df_flat)]:
            for _, r in df.iterrows():
                long_rows.append({"symbol": symbol, "date": r["date"], "open": r["close"], "high": r["close"], "low": r["close"], "close": r["close"]})
        long_df = pd.DataFrame(long_rows)
        data = BacktestData(long_df=long_df, by_symbol={"000001": df_up, "000002": df_flat}, trading_dates=list(dates))
        strategy = LowRiskDebtScoreStrategy()
        scored = strategy.score(data, dates[-1], ["000001", "000002"])
        self.assertFalse(scored.empty)
        self.assertIn("综合得分", scored.columns)
        self.assertIn("综合排名", scored.columns)
        self.assertEqual(scored.iloc[0]["symbol"], "000001")

    def test_filter_pass_through_returns_universe(self) -> None:
        """正常：PassThroughFilter 原样返回 universe。"""
        data = _make_backtest_data(2, 100)
        f = PassThroughFilter()
        out = f.filter_symbols(data, data.trading_dates[-1], ["000000", "000001"])
        self.assertEqual(out, ["000000", "000001"])

    def test_position_equal_weight_splits_top_n(self) -> None:
        """正常：等权仓位策略按 top_n 均分。"""
        scored = pd.DataFrame({
            "symbol": ["A", "B", "C"],
            "综合得分": [0.9, 0.7, 0.5],
            "综合排名": [1, 2, 3],
        })
        pos = EqualWeightPosition()
        w = pos.target_weights(scored, 2)
        self.assertEqual(len(w), 2)
        self.assertAlmostEqual(w["A"], 0.5)
        self.assertAlmostEqual(w["B"], 0.5)

    def test_registry_get_low_risk_debt_bundle(self) -> None:
        """正常：注册表可获取 low_risk_debt 策略包。"""
        bundle = get_strategy_bundle("low_risk_debt")
        self.assertEqual(bundle.name, "low_risk_debt")
        self.assertIn("low_risk_debt", list_strategy_names())

    def test_registry_case_insensitive(self) -> None:
        """正常：策略名大小写不敏感。"""
        b1 = get_strategy_bundle("LOW_RISK_DEBT")
        b2 = get_strategy_bundle("low_risk_debt")
        self.assertEqual(b1.name, b2.name)

    def test_registry_low_risk_debt_most_stable_uses_most_stable_filter(self) -> None:
        """正常：low_risk_debt_most_stable 使用 MostStableFilterStrategy。"""
        from myanalyser.src.backtest.filters import MostStableFilterStrategy

        bundle = get_strategy_bundle("low_risk_debt_most_stable")
        self.assertEqual(bundle.name, "low_risk_debt_most_stable")
        self.assertIsInstance(bundle.filter_strategy, MostStableFilterStrategy)

    def test_registry_steady_debt_uses_steady_debt_filter(self) -> None:
        """正常：steady_debt 使用 SteadyDebtFilterStrategy 且符合稳健型硬约束。"""
        from myanalyser.src.backtest.filters import SteadyDebtFilterStrategy

        bundle = get_strategy_bundle("steady_debt")
        self.assertEqual(bundle.name, "steady_debt")
        self.assertIsInstance(bundle.filter_strategy, SteadyDebtFilterStrategy)
        self.assertIn("steady_debt", list_strategy_names())

    def test_steady_debt_logic_filter_one(self) -> None:
        """正常：稳健型 filter_one 硬约束（最大回撤≥-8%、年化≥5%、夏普≥0.5）。"""
        from myanalyser.src.steady_debt_logic import filter_one

        # 通过：满足全部约束
        ok, _ = filter_one({
            "近3年最大回撤率": -0.05,
            "近3年年化收益率": 0.06,
            "近3年夏普比率": 0.8,
        })
        self.assertFalse(ok, "应通过")

        # 过滤：回撤过深
        fail, msg = filter_one({
            "近3年最大回撤率": -0.12,
            "近3年年化收益率": 0.06,
            "近3年夏普比率": 0.8,
        })
        self.assertTrue(fail)
        self.assertIn("回撤", msg)

        # 过滤：年化不足
        fail, _ = filter_one({
            "近3年最大回撤率": -0.05,
            "近3年年化收益率": 0.03,
            "近3年夏普比率": 0.8,
        })
        self.assertTrue(fail)

        # 过滤：夏普不足
        fail, _ = filter_one({
            "近3年最大回撤率": -0.05,
            "近3年年化收益率": 0.06,
            "近3年夏普比率": 0.3,
        })
        self.assertTrue(fail)

    def test_filter_chain_empty_when_env_not_set(self) -> None:
        """正常：未设置 FUND_BACKTEST_FILTERS 时过滤器链为空。"""
        old = os.environ.pop("FUND_BACKTEST_FILTERS", None)
        try:
            self.assertEqual(get_filter_chain(), [])
        finally:
            if old is not None:
                os.environ["FUND_BACKTEST_FILTERS"] = old

    def test_filtered_candidates_csv_filter_intersects(self) -> None:
        """正常：FilteredCandidatesCsvFilter 取 CSV 白名单与 candidates 交集。"""
        with tempfile.TemporaryDirectory() as d:
            csv_path = Path(d) / "filtered.csv"
            pd.DataFrame(
                {"基金编码": ["000001", "000003"], "是否过滤": ["否", "否"], "过滤原因": ["", ""]}
            ).to_csv(csv_path, index=False, encoding="utf-8-sig")
            old = os.environ.pop(FILTERED_ENV, None)
            try:
                os.environ[FILTERED_ENV] = str(csv_path)
                f = FilteredCandidatesCsvFilter()
                out = f.filter({"000001", "000002", "000003"})
                self.assertEqual(out, {"000001", "000003"})
            finally:
                if old is not None:
                    os.environ[FILTERED_ENV] = old
                else:
                    os.environ.pop(FILTERED_ENV, None)

    def test_max_funds_filter_caps(self) -> None:
        """正常：MaxFundsFilter 按数量截断。"""
        old = os.environ.pop(MAX_FUNDS_ENV, None)
        try:
            os.environ[MAX_FUNDS_ENV] = "2"
            f = MaxFundsFilter()
            out = f.filter({"000003", "000001", "000002"})
            self.assertEqual(out, {"000001", "000002"})
        finally:
            if old is not None:
                os.environ[MAX_FUNDS_ENV] = old
            else:
                os.environ.pop(MAX_FUNDS_ENV, None)

    def test_most_stable_filter_symbols_iterates_and_respects_filter_one(self) -> None:
        """MostStableFilterStrategy 正确遍历 universe 并遵从 filter_one 判定。"""
        from unittest.mock import patch

        from myanalyser.src.backtest.filters import MostStableFilterStrategy

        QUALIFIED = {
            "近3年年化收益率": 5.0, "近1年年化收益率": 4.0,
            "近3年上涨季度比例": 85, "近3年上涨月份比例": 75, "近3年月涨跌幅标准差": 1.0,
            "近1年夏普比率": 1.5, "近3年夏普比率": 1.2, "近1年卡玛比率": 2.0, "近3年卡玛比率": 1.5,
        }
        UNQUALIFIED = dict(QUALIFIED, 近3年年化收益率=2.5)

        def _mock_by_symbol(symbol_order: list[str]):
            idx = [0]

            def _side_effect(df, _as_of):
                sym = symbol_order[idx[0] % len(symbol_order)]
                idx[0] += 1
                return QUALIFIED if sym == "000001" else UNQUALIFIED

            return _side_effect

        data = _make_backtest_data(n_symbols=2, n_days=800, trend_up=0.12)
        syms = sorted(data.by_symbol.keys())

        with patch("myanalyser.src.backtest.filters.most_stable_strategy._compute_most_stable_metrics", side_effect=_mock_by_symbol(syms)):
            f = MostStableFilterStrategy()
            out = f.filter_symbols(data, data.trading_dates[-1], syms)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0], "000001")

    def test_most_stable_compute_metrics_integration(self) -> None:
        """_compute_most_stable_metrics 与 filter_one 对接：真实指标计算输出可被正确消费。"""
        from myanalyser.src.backtest.filters.most_stable_strategy import _compute_most_stable_metrics
        from myanalyser.src.most_stable_logic import filter_one

        data = _make_backtest_data(n_symbols=2, n_days=800, trend_up=0.12)
        df = data.by_symbol["000001"]
        as_of = data.trading_dates[-1]

        row = _compute_most_stable_metrics(df, as_of)
        self.assertGreater(len(row), 0)
        for k in ["近3年年化收益率", "近1年年化收益率", "近1年夏普比率", "近3年卡玛比率"]:
            self.assertIn(k, row)

        is_filtered, reason = filter_one(row)
        self.assertIsInstance(is_filtered, bool)
        self.assertIsInstance(reason, str)

    def test_load_fund_nav_data_with_allowed_codes(self) -> None:
        """正常：allowed_codes 限制加载的基金集合。"""
        with tempfile.TemporaryDirectory() as d:
            nav_dir = Path(d) / "nav"
            nav_dir.mkdir()
            dates = pd.date_range("2023-01-01", periods=100, freq="B").strftime("%Y-%m-%d")
            navs = [1.0 + 0.001 * i for i in range(100)]
            _make_fund_nav_csv(nav_dir / "000001.csv", "000001", dates.tolist(), navs)
            _make_fund_nav_csv(nav_dir / "000002.csv", "000002", dates.tolist(), [1.0] * 100)
            _make_fund_nav_csv(nav_dir / "000003.csv", "000003", dates.tolist(), navs)
            data = load_fund_nav_data(
                nav_dir, max_funds=10, allowed_codes={"000001", "000003"}
            )
            self.assertEqual(set(data.by_symbol.keys()), {"000001", "000003"})

    def test_load_fund_nav_data_allowed_codes_normalizes_stem(self) -> None:
        """正常：allowed_codes 与文件 stem 比较时做相同归一化（如 1.csv -> 000001）。"""
        with tempfile.TemporaryDirectory() as d:
            nav_dir = Path(d) / "nav"
            nav_dir.mkdir()
            dates = pd.date_range("2023-01-01", periods=100, freq="B").strftime("%Y-%m-%d")
            navs = [1.0 + 0.001 * i for i in range(100)]
            _make_fund_nav_csv(nav_dir / "1.csv", "1", dates.tolist(), navs)
            _make_fund_nav_csv(nav_dir / "2.csv", "2", dates.tolist(), [1.0] * 100)
            _make_fund_nav_csv(nav_dir / "000003.csv", "000003", dates.tolist(), navs)
            data = load_fund_nav_data(
                nav_dir, max_funds=10, allowed_codes={"000001", "000002"}
            )
            self.assertEqual(set(data.by_symbol.keys()), {"000001", "000002"})

    def test_load_fund_nav_data_valid_dir(self) -> None:
        """正常：有效净值目录加载成功。"""
        with tempfile.TemporaryDirectory() as d:
            nav_dir = Path(d) / "nav"
            nav_dir.mkdir()
            dates = pd.date_range("2023-01-01", periods=100, freq="B").strftime("%Y-%m-%d")
            navs = [1.0 + 0.001 * i for i in range(100)]
            _make_fund_nav_csv(nav_dir / "000001.csv", "000001", dates.tolist(), navs)
            _make_fund_nav_csv(nav_dir / "000002.csv", "000002", dates.tolist(), [1.0] * 100)
            data = load_fund_nav_data(nav_dir, max_funds=10)
            self.assertEqual(len(data.by_symbol), 2)
            self.assertFalse(data.long_df.empty)
            self.assertTrue(len(data.trading_dates) > 0)

    def test_metrics_compute_returns_dict(self) -> None:
        """正常：指标计算返回包含预期键的字典。"""
        dates = np.arange("2020-01-01", "2023-06-01", dtype="datetime64[D]")
        prices = 1.0 + np.linspace(0, 0.3, len(dates))
        out = compute_low_risk_debt_metrics(dates, prices)
        for k in ["近1年最大回撤率", "近3年年化收益率", "近1年卡玛比率"]:
            self.assertIn(k, out)

    def test_run_backtest_produces_period_log(self) -> None:
        """正常：回测引擎产出 period_log。"""
        data = _make_backtest_data(3, 400, 0.2)
        bundle = build_bundle()
        result = run_backtest(
            data, bundle,
            start_date="2021-01-01", end_date="2022-12-31",
            top_n=2, rebalance_period=20, warmup=60,
        )
        self.assertIsInstance(result, BacktestResult)
        self.assertIsInstance(result.period_log, list)

    def test_write_reports_creates_summary_and_detail(self) -> None:
        """正常：报表写入生成 summary、detail、equity_curve、orders、positions_flat、report_md。"""
        data = _make_backtest_data(2, 300)
        bundle = build_bundle()
        result = run_backtest(
            data, bundle,
            start_date="2021-01-01", end_date="2022-06-30",
            top_n=2, rebalance_period=40, warmup=60,
        )
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            paths = write_reports(out_dir, result, data)
            self.assertIn("summary", paths)
            self.assertIn("detail", paths)
            self.assertIn("equity_curve", paths)
            self.assertIn("orders", paths)
            self.assertIn("positions_flat", paths)
            self.assertIn("report_md", paths)
            self.assertTrue(paths["summary"].exists())
            self.assertTrue(paths["detail"].exists())
            self.assertTrue(paths["equity_curve"].exists())
            self.assertTrue(paths["orders"].exists())
            self.assertTrue(paths["positions_flat"].exists())
            self.assertTrue(paths["report_md"].exists())
            summary_df = pd.read_csv(paths["summary"], encoding="utf-8-sig")
            self.assertFalse(summary_df.empty)
            holding = summary_df[summary_df["section"] == "metrics_holding"]
            if not holding.empty:
                expected_names = {
                    "年化收益率", "夏普比率", "索提诺比率", "卡玛比率",
                    "盈利因子", "溃疡指数", "溃疡绩效指数", "净值R方",
                    "标准误差", "上涨星期比例", "上涨月份比例", "最大回撤率",
                    "最长回撤修复天数", "年化波动率",
                }
                actual_names = set(holding["name"].dropna().unique())
                self.assertTrue(
                    expected_names.issubset(actual_names),
                    f"metrics_holding 应有完整指标，got: {actual_names}",
                )
            detail_df = pd.read_csv(paths["detail"], encoding="utf-8-sig")
            self.assertIn("period_return", detail_df.columns)


# ==================== 异常场景 ====================


class TestExceptionScenarios(unittest.TestCase):
    """异常场景：非法输入、缺失资源等。"""

    def test_registry_unknown_strategy_raises(self) -> None:
        """异常：未知策略名抛出 ValueError。"""
        with self.assertRaises(ValueError) as ctx:
            get_strategy_bundle("nonexistent_strategy")
        self.assertIn("未知策略包", str(ctx.exception))

    def test_filter_registry_unknown_raises(self) -> None:
        """异常：未知过滤器名抛出 ValueError。"""
        old = os.environ.pop("FUND_BACKTEST_FILTERS", None)
        try:
            os.environ["FUND_BACKTEST_FILTERS"] = "unknown_filter"
            with self.assertRaises(ValueError) as ctx:
                get_filter_chain()
            self.assertIn("未知过滤器", str(ctx.exception))
        finally:
            if old is not None:
                os.environ["FUND_BACKTEST_FILTERS"] = old
            else:
                os.environ.pop("FUND_BACKTEST_FILTERS", None)

    def test_load_nav_dir_not_exists_raises(self) -> None:
        """异常：目录不存在抛出 FileNotFoundError。"""
        with self.assertRaises(FileNotFoundError):
            load_fund_nav_data(Path("/nonexistent/path/xyz"))

    def test_load_nav_empty_dir_raises(self) -> None:
        """异常：无 CSV 的目录抛出 FileNotFoundError。"""
        with tempfile.TemporaryDirectory() as d:
            empty = Path(d) / "empty"
            empty.mkdir()
            with self.assertRaises(FileNotFoundError):
                load_fund_nav_data(empty)

    def test_score_empty_symbols_returns_empty_df(self) -> None:
        """异常：空标的列表返回空 DataFrame。"""
        data = _make_backtest_data(2, 100)
        strategy = LowRiskDebtScoreStrategy()
        scored = strategy.score(data, data.trading_dates[-1], [])
        self.assertTrue(scored.empty)
        self.assertIn("综合得分", scored.columns)

    def test_position_top_n_zero_returns_empty(self) -> None:
        """异常：top_n=0 返回空权重。"""
        scored = pd.DataFrame({"symbol": ["A"], "综合得分": [0.5], "综合排名": [1]})
        pos = EqualWeightPosition()
        w = pos.target_weights(scored, 0)
        self.assertEqual(w, {})


# ==================== 边界条件 ====================


class TestBoundaryConditions(unittest.TestCase):
    """边界条件：空值、极值、临界参数。"""

    def test_score_single_symbol(self) -> None:
        """边界：单基金评分。"""
        data = _make_backtest_data(1, 300)
        strategy = LowRiskDebtScoreStrategy()
        scored = strategy.score(data, data.trading_dates[-1], ["000000"])
        self.assertEqual(len(scored), 1)
        self.assertEqual(scored.iloc[0]["symbol"], "000000")

    def test_metrics_empty_prices_handled(self) -> None:
        """边界：极短价格序列。"""
        dates = np.array(["2020-01-01"], dtype="datetime64[D]")
        prices = np.array([1.0])
        out = compute_low_risk_debt_metrics(dates, prices)
        self.assertIsInstance(out, dict)
        # 多数指标在数据不足时为 None
        self.assertIn("近1年最大回撤率", out)

    def test_metrics_window_insufficient_returns_none(self) -> None:
        """边界：窗口不足时近N年指标返回 None，避免用短样本冠以「近N年」造成误导。"""
        cfg = WindowConfig()
        win_1y, win_3y = cfg.trading_days_per_year, cfg.trading_days_per_year * 3

        def _run(n: int) -> dict:
            dates = np.arange("2024-01-01", n, dtype="datetime64[D]")
            prices = np.ones(n) * 1.0
            return compute_low_risk_debt_metrics(dates, prices, config=cfg)

        # 不足 1 年：近 1 年、近 3 年 均为 None
        out_short = _run(100)
        for k in ["近1年最大回撤率", "近3年最大回撤率", "近1年年化收益率", "近3年年化收益率"]:
            self.assertIsNone(out_short[k], msg=f"{k} 应在窗口不足时为 None")

        # 不足 3 年但够 1 年：近 1 年 有值，近 3 年 为 None
        out_mid = _run(win_1y + 10)
        self.assertIsNotNone(out_mid["近1年最大回撤率"], "近1年应有值")
        self.assertIsNone(out_mid["近3年最大回撤率"], "近3年应为 None")

        # 足够 3 年：均有值
        out_long = _run(win_3y + 10)
        self.assertIsNotNone(out_long["近1年最大回撤率"])
        self.assertIsNotNone(out_long["近3年最大回撤率"])

    def test_metrics_zero_price_handled(self) -> None:
        """边界：含 0 或负价格（可能产生 inf/nan）。"""
        dates = np.arange("2020-01-01", "2022-01-01", dtype="datetime64[D]")[:300]
        prices = np.ones(len(dates)) * 0.5
        prices[-1] = 0.0  # 末尾为 0
        out = compute_low_risk_debt_metrics(dates, prices)
        self.assertIsInstance(out, dict)

    def test_position_scored_empty_returns_empty(self) -> None:
        """边界：scored 为空返回空权重。"""
        pos = EqualWeightPosition()
        w = pos.target_weights(pd.DataFrame(columns=["symbol", "综合得分", "综合排名"]), 3)
        self.assertEqual(w, {})

    def test_rebalance_period_zero(self) -> None:
        """边界：rebalance_period=0 仅首日调仓。"""
        # 需要足够交易日覆盖回测窗口，否则 pybroker DataSource 为空
        data = _make_backtest_data(2, 600)
        bundle = build_bundle()
        result = run_backtest(
            data, bundle,
            start_date="2021-01-01", end_date="2022-12-31",
            top_n=2, rebalance_period=0, warmup=50,
        )
        self.assertIsInstance(result, BacktestResult)
        self.assertIsInstance(result.period_log, list)

    def test_top_n_exceeds_symbols(self) -> None:
        """边界：top_n 大于标的数时自动缩小。"""
        # 需要足够交易日覆盖回测窗口
        data = _make_backtest_data(2, 600)
        bundle = build_bundle()
        result = run_backtest(
            data, bundle,
            start_date="2021-01-01", end_date="2022-06-30",
            top_n=100, rebalance_period=20, warmup=50,
        )
        self.assertIsInstance(result, BacktestResult)
        if result.period_log:
            first = result.period_log[0]
            self.assertLessEqual(len(first["selected_symbols"]), 2)

    def test_load_nav_with_date_filter_empty_result_raises(self) -> None:
        """边界：日期过滤后无数据应抛出。"""
        with tempfile.TemporaryDirectory() as d:
            nav_dir = Path(d) / "nav"
            nav_dir.mkdir()
            _make_fund_nav_csv(
                nav_dir / "000001.csv",
                "000001",
                ["2023-01-01", "2023-01-02"],
                [1.0, 1.01],
            )
            with self.assertRaises(ValueError):
                load_fund_nav_data(nav_dir, start_date="2030-01-01", end_date="2030-12-31")


if __name__ == "__main__":
    unittest.main()
