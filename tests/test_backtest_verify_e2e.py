"""backtest_verify_e2e 与 verify_e2e_top5 策略的单元测试（正常/异常/边界）。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import backtest_verify_e2e
from backtest.strategies.verify_e2e_top5 import (
    FixedSelectionFilterStrategy,
    FixedSelectionScoreStrategy,
    build_bundle_verify_e2e,
)


def _make_nav_csv(path: Path, code: str, dates: list[str], navs: list[float]) -> None:
    rows = [{"基金代码": code, "净值日期": d, "复权净值": v} for d, v in zip(dates, navs)]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


# ==================== 正常场景 ====================


class TestNormalScenarios(unittest.TestCase):
    """正常场景：标准输入下的预期行为。"""

    def test_quote_sql_normal(self) -> None:
        """正常：普通字符串正确加引号。"""
        self.assertEqual(backtest_verify_e2e._quote_sql("202401"), "'202401'")

    def test_quote_sql_escape_single_quote(self) -> None:
        """正常：单引号被转义。"""
        self.assertEqual(backtest_verify_e2e._quote_sql("a'b"), "'a\\'b'")

    def test_quote_sql_escape_backslash(self) -> None:
        """正常：反斜杠被转义。"""
        self.assertEqual(backtest_verify_e2e._quote_sql("a\\b"), "'a\\\\b'")

    def test_build_status_filter_empty(self) -> None:
        """正常：空排除列表返回 1。"""
        self.assertEqual(backtest_verify_e2e._build_status_filter("col", []), "1")

    def test_build_status_filter_single(self) -> None:
        """正常：单值过滤正确构建。"""
        out = backtest_verify_e2e._build_status_filter("subscribe_status", ["暂停申购"])
        self.assertIn("subscribe_status", out)
        self.assertIn("暂停申购", out)

    def test_build_status_filter_multiple(self) -> None:
        """正常：多值过滤正确构建。"""
        out = backtest_verify_e2e._build_status_filter(
            "redeem_status", ["暂停赎回", "封闭期"]
        )
        self.assertIn("redeem_status", out)
        self.assertIn("暂停赎回", out)
        self.assertIn("封闭期", out)

    def test_fetch_fund_selection_normal(self) -> None:
        """正常：ClickHouse 返回多基金，格式正确。"""
        df_in = pd.DataFrame([{"fund_code": "000001"}, {"fund_code": "000002"}])

        def fake_query(q: str, c: str) -> pd.DataFrame:
            if "SELECT fund_code" in q and "fact_fund_scoreboard_snapshot" in q:
                return df_in.copy()
            raise AssertionError(f"unexpected query: {q}")

        with patch("backtest_verify_e2e._run_clickhouse_query", side_effect=fake_query):
            out = backtest_verify_e2e._fetch_fund_selection(
                clickhouse_db="fund_analysis",
                clickhouse_container="fund_clickhouse",
                data_version="202401",
                selection_where="1",
                selection_order_by="annual_return DESC",
                selection_limit=5,
                exclude_subscribe_status=["暂停申购"],
                exclude_redeem_status=["暂停赎回"],
            )
        self.assertFalse(out.empty)
        self.assertIn("fund_code", out.columns)
        self.assertIn("weight", out.columns)
        self.assertEqual(set(out["fund_code"].tolist()), {"000001", "000002"})
        self.assertAlmostEqual(out["weight"].sum(), 1.0, places=9)

    def test_main_e2e_success(self) -> None:
        """正常：main 端到端成功，产出 period_detail.csv、backtest_report.md。"""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            nav_dir = root / "fund_adjusted_nav_by_code"
            nav_dir.mkdir(parents=True, exist_ok=True)
            out_dir = root / "artifacts" / "backtest_test"
            nav_dates = pd.date_range("2024-01-02", "2024-03-31", freq="D")
            for code in ("000001", "000002"):
                rows = []
                for i, dt in enumerate(nav_dates):
                    adj = 1.0 + 0.01 * i if code == "000001" else 1.0
                    rows.append({"基金代码": code, "净值日期": dt.strftime("%Y-%m-%d"), "复权净值": adj})
                pd.DataFrame(rows).to_csv(nav_dir / f"{code}.csv", index=False, encoding="utf-8-sig")

            def fake_query(q: str, c: str) -> pd.DataFrame:
                if "fact_fund_scoreboard_snapshot" in q and "SELECT fund_code" in q:
                    return pd.DataFrame([{"fund_code": "000001"}, {"fund_code": "000002"}])
                raise AssertionError(f"unexpected query: {q}")

            argv = [
                "backtest_verify_e2e",
                "--start-date", "2024-01-01",
                "--end-date", "2024-01-31",
                "--nav-dir", str(nav_dir),
                "--output-dir", str(out_dir),
                "--selection-data-version", "test_v1",
                "--warmup", "5",
            ]
            with patch("backtest_verify_e2e._run_clickhouse_query", side_effect=fake_query), patch.object(
                sys, "argv", argv
            ):
                backtest_verify_e2e.main()

            self.assertTrue((out_dir / "period_detail.csv").exists())
            self.assertTrue((out_dir / "backtest_report.md").exists())
            detail = pd.read_csv(out_dir / "period_detail.csv", encoding="utf-8-sig")
            self.assertFalse(detail.empty)
            self.assertIn("period_return", detail.columns)

    def test_fixed_selection_filter_all_hit(self) -> None:
        """正常：universe 全部在 allowed 中。"""
        strat = FixedSelectionFilterStrategy(allowed_symbols=("000001", "000002"))
        # 需要 BacktestData，但 filter 仅用 universe，可传简单 mock
        from backtest.data import BacktestData
        data = BacktestData(long_df=pd.DataFrame(), by_symbol={}, trading_dates=[])
        out = strat.filter_symbols(data, pd.Timestamp("2024-01-15"), ["000001", "000002"])
        self.assertEqual(sorted(out), ["000001", "000002"])

    def test_fixed_selection_filter_partial_hit(self) -> None:
        """正常：universe 部分在 allowed 中。"""
        strat = FixedSelectionFilterStrategy(allowed_symbols=("000001",))
        from backtest.data import BacktestData
        data = BacktestData(long_df=pd.DataFrame(), by_symbol={}, trading_dates=[])
        out = strat.filter_symbols(data, pd.Timestamp("2024-01-15"), ["000001", "000003"])
        self.assertEqual(out, ["000001"])

    def test_fixed_selection_score_empty(self) -> None:
        """正常：symbols 为空返回空 DataFrame。"""
        strat = FixedSelectionScoreStrategy(allowed_symbols=("000001", "000002"))
        from backtest.data import BacktestData
        data = BacktestData(long_df=pd.DataFrame(), by_symbol={}, trading_dates=[])
        out = strat.score(data, pd.Timestamp("2024-01-15"), [])
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), ["symbol", "综合得分", "综合排名"])

    def test_fixed_selection_score_order_preserved(self) -> None:
        """正常：按 allowed_symbols 顺序打分。"""
        strat = FixedSelectionScoreStrategy(allowed_symbols=("000002", "000001"))
        from backtest.data import BacktestData
        data = BacktestData(long_df=pd.DataFrame(), by_symbol={}, trading_dates=[])
        out = strat.score(data, pd.Timestamp("2024-01-15"), ["000001", "000002"])
        self.assertEqual(len(out), 2)
        # 000002 在前，得分应高于 000001
        r2 = out[out["symbol"] == "000002"].iloc[0]
        r1 = out[out["symbol"] == "000001"].iloc[0]
        self.assertGreater(float(r2["综合得分"]), float(r1["综合得分"]))

    def test_build_bundle_verify_e2e_normal(self) -> None:
        """正常：构建策略包。"""
        bundle = build_bundle_verify_e2e(["000001", "000002"])
        self.assertEqual(bundle.name, "verify_e2e_top5")
        self.assertIsNotNone(bundle.filter_strategy)
        self.assertIsNotNone(bundle.score_strategy)
        self.assertIsNotNone(bundle.position_strategy)


# ==================== 异常场景 ====================


class TestExceptionScenarios(unittest.TestCase):
    """异常场景：非法输入、缺失资源等。"""

    def test_main_empty_selection_exits(self) -> None:
        """异常：选基为空应 SystemExit。"""
        def fake_query(q: str, c: str) -> pd.DataFrame:
            if "fact_fund_scoreboard_snapshot" in q and "SELECT fund_code" in q:
                return pd.DataFrame(columns=["fund_code"])
            raise AssertionError(f"unexpected query: {q}")

        argv = [
            "backtest_verify_e2e",
            "--start-date", "2024-01-01",
            "--end-date", "2024-01-31",
            "--nav-dir", "/tmp/nav",
            "--output-dir", "/tmp/out",
            "--selection-data-version", "empty",
        ]
        with patch("backtest_verify_e2e._run_clickhouse_query", side_effect=fake_query), patch.object(
            sys, "argv", argv
        ):
            with self.assertRaises(SystemExit):
                backtest_verify_e2e.main()

    def test_main_no_nav_data_exits(self) -> None:
        """异常：净值目录无有效数据应退出。load_fund_nav_data 抛 ValueError。"""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            nav_dir = root / "nav_empty"
            nav_dir.mkdir()
            # 格式错误的 CSV（缺净值日期/复权净值列），load_fund_nav_data 会 skip 后抛 ValueError
            (nav_dir / "000001.csv").write_text("bad,columns\n1,2\n", encoding="utf-8")
            out_dir = root / "out"

            def fake_query(q: str, c: str) -> pd.DataFrame:
                if "fact_fund_scoreboard_snapshot" in q and "SELECT fund_code" in q:
                    return pd.DataFrame([{"fund_code": "000001"}])
                raise AssertionError(f"unexpected query: {q}")

            argv = [
                "backtest_verify_e2e",
                "--start-date", "2024-01-01",
                "--end-date", "2024-01-31",
                "--nav-dir", str(nav_dir),
                "--output-dir", str(out_dir),
                "--selection-data-version", "v1",
            ]
            with patch("backtest_verify_e2e._run_clickhouse_query", side_effect=fake_query), patch.object(
                sys, "argv", argv
            ):
                with self.assertRaises(ValueError):
                    backtest_verify_e2e.main()

    def test_fetch_fund_selection_empty_result(self) -> None:
        """异常：查询返回空。"""
        def fake_query(q: str, c: str) -> pd.DataFrame:
            return pd.DataFrame()

        with patch("backtest_verify_e2e._run_clickhouse_query", side_effect=fake_query):
            out = backtest_verify_e2e._fetch_fund_selection(
                clickhouse_db="fund_analysis",
                clickhouse_container="fund_clickhouse",
                data_version="202401",
                selection_where="1",
                selection_order_by="annual_return DESC",
                selection_limit=5,
                exclude_subscribe_status=[],
                exclude_redeem_status=[],
            )
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), ["fund_code", "weight"])


# ==================== 边界条件 ====================


class TestBoundaryConditions(unittest.TestCase):
    """边界条件：空值、极值、单元素等。"""

    def test_build_bundle_empty_allowed(self) -> None:
        """边界：allowed_symbols 为空列表。"""
        bundle = build_bundle_verify_e2e([])
        self.assertEqual(bundle.filter_strategy.allowed_symbols, ())

    def test_build_bundle_code_padding(self) -> None:
        """边界：基金代码自动补齐为 6 位。"""
        bundle = build_bundle_verify_e2e(["1", "000002"])
        self.assertEqual(bundle.filter_strategy.allowed_symbols, ("000001", "000002"))

    def test_build_bundle_strips_whitespace(self) -> None:
        """边界：去除前后空格。"""
        bundle = build_bundle_verify_e2e(["  000001  ", "000002"])
        self.assertEqual(bundle.filter_strategy.allowed_symbols, ("000001", "000002"))

    def test_build_bundle_skips_empty_strings(self) -> None:
        """边界：空字符串被过滤。"""
        bundle = build_bundle_verify_e2e(["000001", "", "000002"])
        self.assertEqual(bundle.filter_strategy.allowed_symbols, ("000001", "000002"))

    def test_build_bundle_rejects_non_digit(self) -> None:
        """边界：非数字代码被剔除，仅保留纯数字。"""
        bundle = build_bundle_verify_e2e(["ABC123", "000001"])
        self.assertEqual(bundle.filter_strategy.allowed_symbols, ("000001",))

    def test_build_bundle_rejects_all_zeros(self) -> None:
        """边界：全零代码（如 000000）被剔除。"""
        bundle = build_bundle_verify_e2e(["000000", "000001"])
        self.assertEqual(bundle.filter_strategy.allowed_symbols, ("000001",))

    def test_fixed_selection_filter_empty_universe(self) -> None:
        """边界：universe 为空。"""
        strat = FixedSelectionFilterStrategy(allowed_symbols=("000001",))
        from backtest.data import BacktestData
        data = BacktestData(long_df=pd.DataFrame(), by_symbol={}, trading_dates=[])
        out = strat.filter_symbols(data, pd.Timestamp("2024-01-15"), [])
        self.assertEqual(out, [])

    def test_fixed_selection_filter_empty_allowed(self) -> None:
        """边界：allowed 为空，universe 有值。"""
        strat = FixedSelectionFilterStrategy(allowed_symbols=())
        from backtest.data import BacktestData
        data = BacktestData(long_df=pd.DataFrame(), by_symbol={}, trading_dates=[])
        out = strat.filter_symbols(data, pd.Timestamp("2024-01-15"), ["000001"])
        self.assertEqual(out, [])

    def test_main_single_fund(self) -> None:
        """边界：单只基金。"""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            nav_dir = root / "nav"
            nav_dir.mkdir()
            out_dir = root / "out"
            nav_dates = pd.date_range("2024-01-02", "2024-03-31", freq="D")
            rows = [{"基金代码": "000001", "净值日期": dt.strftime("%Y-%m-%d"), "复权净值": 1.0 + 0.001 * i} for i, dt in enumerate(nav_dates)]
            pd.DataFrame(rows).to_csv(nav_dir / "000001.csv", index=False, encoding="utf-8-sig")

            def fake_query(q: str, c: str) -> pd.DataFrame:
                if "fact_fund_scoreboard_snapshot" in q and "SELECT fund_code" in q:
                    return pd.DataFrame([{"fund_code": "000001"}])
                raise AssertionError(f"unexpected query: {q}")

            argv = [
                "backtest_verify_e2e",
                "--start-date", "2024-01-01",
                "--end-date", "2024-01-31",
                "--nav-dir", str(nav_dir),
                "--output-dir", str(out_dir),
                "--selection-data-version", "v1",
                "--top-n", "1",
                "--warmup", "5",
            ]
            with patch("backtest_verify_e2e._run_clickhouse_query", side_effect=fake_query), patch.object(
                sys, "argv", argv
            ):
                backtest_verify_e2e.main()

            self.assertTrue((out_dir / "period_detail.csv").exists())
            detail = pd.read_csv(out_dir / "period_detail.csv", encoding="utf-8-sig")
            self.assertFalse(detail.empty)

    def test_main_top_n_exceeds_selection(self) -> None:
        """边界：top_n 大于选基数量，应取 min。"""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            nav_dir = root / "nav"
            nav_dir.mkdir()
            out_dir = root / "out"
            nav_dates = pd.date_range("2024-01-02", "2024-03-31", freq="D")
            for code in ("000001", "000002"):
                rows = [{"基金代码": code, "净值日期": dt.strftime("%Y-%m-%d"), "复权净值": 1.0} for dt in nav_dates]
                pd.DataFrame(rows).to_csv(nav_dir / f"{code}.csv", index=False, encoding="utf-8-sig")

            def fake_query(q: str, c: str) -> pd.DataFrame:
                if "fact_fund_scoreboard_snapshot" in q and "SELECT fund_code" in q:
                    return pd.DataFrame([{"fund_code": "000001"}, {"fund_code": "000002"}])
                raise AssertionError(f"unexpected query: {q}")

            argv = [
                "backtest_verify_e2e",
                "--start-date", "2024-01-01",
                "--end-date", "2024-01-31",
                "--nav-dir", str(nav_dir),
                "--output-dir", str(out_dir),
                "--selection-data-version", "v1",
                "--top-n", "10",
                "--warmup", "5",
            ]
            with patch("backtest_verify_e2e._run_clickhouse_query", side_effect=fake_query), patch.object(
                sys, "argv", argv
            ):
                backtest_verify_e2e.main()

            self.assertTrue((out_dir / "period_detail.csv").exists())

    def test_exclude_status_empty_string(self) -> None:
        """边界：exclude 为空字符串，解析后为空列表。"""
        def fake_query(q: str, c: str) -> pd.DataFrame:
            if "fact_fund_scoreboard_snapshot" in q and "SELECT fund_code" in q:
                return pd.DataFrame([{"fund_code": "000001"}])
            raise AssertionError(f"unexpected query: {q}")

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            nav_dir = root / "nav"
            nav_dir.mkdir()
            out_dir = root / "out"
            nav_dates = pd.date_range("2024-01-02", "2024-03-31", freq="D")
            rows = [{"基金代码": "000001", "净值日期": dt.strftime("%Y-%m-%d"), "复权净值": 1.0} for dt in nav_dates]
            pd.DataFrame(rows).to_csv(nav_dir / "000001.csv", index=False, encoding="utf-8-sig")

            argv = [
                "backtest_verify_e2e",
                "--start-date", "2024-01-01",
                "--end-date", "2024-01-31",
                "--nav-dir", str(nav_dir),
                "--output-dir", str(out_dir),
                "--selection-data-version", "v1",
                "--exclude-subscribe-status", "",
                "--exclude-redeem-status", "",
                "--warmup", "5",
            ]
            with patch("backtest_verify_e2e._run_clickhouse_query", side_effect=fake_query), patch.object(
                sys, "argv", argv
            ):
                backtest_verify_e2e.main()

            self.assertTrue((out_dir / "period_detail.csv").exists())


if __name__ == "__main__":
    unittest.main()
