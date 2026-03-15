"""filter_score 模块单元测试与 CLI 冒烟。"""

import sys
from pathlib import Path

# 支持 myanalyser 包导入（从 myanalyser 或 finance 运行）
_ws = Path(__file__).resolve().parents[2]
if str(_ws) not in sys.path:
    sys.path.insert(0, str(_ws))

import unittest
from unittest.mock import patch

import pandas as pd

from myanalyser.src.filter_score.base import load_filter_strategy, load_score_strategy
from myanalyser.src.filter_score.filter_and_score_main import main, run_pipeline
from myanalyser.src.most_stable_logic import filter_one as filter_most_stable
from myanalyser.src.filter_score.filters.non_a_unlimited_purchase import (
    filter_one as filter_non_a_unlimited,
)
from myanalyser.src.filter_score.filters.steady_aggressive import (
    filter_one as filter_steady_aggressive,
)


class FilterScoreTest(unittest.TestCase):
    def test_most_stable_filter_pass(self) -> None:
        """满足最稳健原则的行应通过。"""
        row = {
            "近3年年化收益率": 5.0,
            "近1年年化收益率": 4.0,
            "近3年上涨季度比例": 85,
            "近3年上涨月份比例": 75,
            "近3年月涨跌幅标准差": 1.0,
            "近1年夏普比率": 1.5,
            "近3年夏普比率": 1.2,
            "近1年卡玛比率": 2.0,
            "近3年卡玛比率": 1.5,
        }
        is_filtered, reason = filter_most_stable(row)
        self.assertFalse(is_filtered)
        self.assertEqual(reason, "")

    def test_most_stable_filter_fail_low_return(self) -> None:
        """近3年年化收益率<=3 应被过滤。"""
        row = {
            "近3年年化收益率": 2.5,
            "近1年年化收益率": 4.0,
            "近3年上涨季度比例": 85,
            "近3年上涨月份比例": 75,
            "近3年月涨跌幅标准差": 1.0,
            "近1年夏普比率": 1.5,
            "近3年夏普比率": 1.2,
            "近1年卡玛比率": 2.0,
            "近3年卡玛比率": 1.5,
        }
        is_filtered, reason = filter_most_stable(row)
        self.assertTrue(is_filtered)
        self.assertIn("近3年年化收益率", reason)

    def test_most_stable_filter_fail_high_std(self) -> None:
        """近3年月涨跌幅标准差>=1.5 应被过滤。"""
        row = {
            "近3年年化收益率": 5.0,
            "近1年年化收益率": 4.0,
            "近3年上涨季度比例": 85,
            "近3年上涨月份比例": 75,
            "近3年月涨跌幅标准差": 2.0,
            "近1年夏普比率": 1.5,
            "近3年夏普比率": 1.2,
            "近1年卡玛比率": 2.0,
            "近3年卡玛比率": 1.5,
        }
        is_filtered, reason = filter_most_stable(row)
        self.assertTrue(is_filtered)
        self.assertIn("近3年月涨跌幅标准差", reason)

    def test_non_a_unlimited_filter_pass(self) -> None:
        """非A类、开放申购赎回、无限大额或限大额>=20万应通过。"""
        row = {
            "基金名称": "华夏成长混合",  # 不以A结尾
            "申购状态": "开放申购",
            "赎回状态": "开放赎回",
            "日累计限定金额": 100000000000.0,
        }
        is_filtered, reason = filter_non_a_unlimited(row)
        self.assertFalse(is_filtered)
        self.assertEqual(reason, "")

    def test_non_a_unlimited_filter_name_ends_with_a(self) -> None:
        """基金名称以A结尾应被过滤。"""
        row = {
            "基金名称": "华夏聚利债券A",
            "申购状态": "开放申购",
            "赎回状态": "开放赎回",
        }
        is_filtered, reason = filter_non_a_unlimited(row)
        self.assertTrue(is_filtered)
        self.assertIn("A", reason)

    def test_non_a_unlimited_filter_suspend_purchase(self) -> None:
        """申购状态为暂停申购应被过滤。"""
        row = {
            "基金名称": "华夏成长混合",
            "申购状态": "暂停申购",
            "赎回状态": "开放赎回",
        }
        is_filtered, reason = filter_non_a_unlimited(row)
        self.assertTrue(is_filtered)
        self.assertIn("暂停申购", reason)

    def test_non_a_unlimited_filter_suspend_redeem(self) -> None:
        """赎回状态为暂停赎回应被过滤。"""
        row = {
            "基金名称": "华夏成长混合",
            "申购状态": "开放申购",
            "赎回状态": "暂停赎回",
        }
        is_filtered, reason = filter_non_a_unlimited(row)
        self.assertTrue(is_filtered)
        self.assertIn("暂停赎回", reason)

    def test_non_a_unlimited_filter_limit_low(self) -> None:
        """限大额且日累计限定金额<20万应被过滤。"""
        row = {
            "基金名称": "广发某某债券C",
            "申购状态": "限大额",
            "赎回状态": "开放赎回",
            "日累计限定金额": 100000,  # 10万 < 20万
        }
        is_filtered, reason = filter_non_a_unlimited(row)
        self.assertTrue(is_filtered)
        self.assertIn("限大额", reason)

    def test_non_a_unlimited_filter_limit_high_pass(self) -> None:
        """限大额且日累计限定金额>=20万应通过。"""
        row = {
            "基金名称": "广发某某债券C",
            "申购状态": "限大额",
            "赎回状态": "开放赎回",
            "日累计限定金额": 3000000.0,  # 300万 >= 20万
        }
        is_filtered, reason = filter_non_a_unlimited(row)
        self.assertFalse(is_filtered)
        self.assertEqual(reason, "")

    def test_steady_aggressive_filter_pass(self) -> None:
        """满足偏稳进取原则的行应通过。"""
        row = {
            "近1年年化收益率": 5.0,
            "近3年年化收益率": 6.0,
            "近3年上涨季度比例": 85,
            "近3年上涨月份比例": 65,
            "近3年最大回撤率": 8.0,
        }
        is_filtered, reason = filter_steady_aggressive(row)
        self.assertFalse(is_filtered)
        self.assertEqual(reason, "")

    def test_steady_aggressive_filter_fail_low_return(self) -> None:
        """近1年年化收益率<=4 应被过滤。"""
        row = {
            "近1年年化收益率": 3.5,
            "近3年年化收益率": 6.0,
            "近3年上涨季度比例": 85,
            "近3年上涨月份比例": 65,
            "近3年最大回撤率": 8.0,
        }
        is_filtered, reason = filter_steady_aggressive(row)
        self.assertTrue(is_filtered)
        self.assertIn("近1年年化收益率", reason)

    def test_steady_aggressive_filter_fail_low_month_ratio(self) -> None:
        """近3年上涨月份比例<=60 应被过滤。"""
        row = {
            "近1年年化收益率": 5.0,
            "近3年年化收益率": 6.0,
            "近3年上涨季度比例": 85,
            "近3年上涨月份比例": 55,
            "近3年最大回撤率": 8.0,
        }
        is_filtered, reason = filter_steady_aggressive(row)
        self.assertTrue(is_filtered)
        self.assertIn("近3年上涨月份比例", reason)

    def test_steady_aggressive_filter_fail_high_drawdown(self) -> None:
        """近3年最大回撤率>=10 应被过滤。"""
        row = {
            "近1年年化收益率": 5.0,
            "近3年年化收益率": 6.0,
            "近3年上涨季度比例": 85,
            "近3年上涨月份比例": 65,
            "近3年最大回撤率": 12.0,
        }
        is_filtered, reason = filter_steady_aggressive(row)
        self.assertTrue(is_filtered)
        self.assertIn("近3年最大回撤率", reason)

    def test_load_steady_aggressive_filter_strategy(self) -> None:
        """应能加载 steady_aggressive 过滤脚本。"""
        path = Path(__file__).resolve().parents[1] / "src/filter_score/filters/steady_aggressive.py"
        self.assertTrue(path.exists())
        strategy = load_filter_strategy(path)
        self.assertEqual(strategy.STRATEGY_NAME, "偏稳进取原则")
        is_f, _ = strategy.filter_one({"近3年年化收益率": 3})
        self.assertTrue(is_f)

    def test_load_non_a_unlimited_filter_strategy(self) -> None:
        """应能加载 non_a_unlimited_purchase 过滤脚本。"""
        path = Path(__file__).resolve().parents[1] / "src/filter_score/filters/non_a_unlimited_purchase.py"
        self.assertTrue(path.exists())
        strategy = load_filter_strategy(path)
        self.assertEqual(strategy.STRATEGY_NAME, "非A类&不限申购赎回")
        is_f, _ = strategy.filter_one({"基金名称": "xxA"})
        self.assertTrue(is_f)

    def test_load_filter_strategy(self) -> None:
        """应能加载 most_stable 过滤脚本。"""
        path = Path(__file__).resolve().parents[1] / "src/filter_score/filters/most_stable.py"
        self.assertTrue(path.exists())
        strategy = load_filter_strategy(path)
        self.assertEqual(strategy.STRATEGY_NAME, "最稳健原则")
        is_f, _ = strategy.filter_one({"近3年年化收益率": 2})
        self.assertTrue(is_f)

    def test_load_score_strategy(self) -> None:
        """应能加载 low_risk_debt 算分脚本。"""
        path = Path(__file__).resolve().parents[1] / "src/filter_score/scores/low_risk_debt.py"
        self.assertTrue(path.exists())
        strategy = load_score_strategy(path)
        self.assertEqual(strategy.STRATEGY_NAME, "低风险偏债得分")
        df = pd.DataFrame({
            "基金代码": ["001"],
            "基金名称": ["A"],
            "近1年最大回撤率": [0.05],
            "近3年最长回撤修复天数": [50],
            "近3年最大回撤率": [0.10],
            "近1年卡玛比率": [6.0],
            "近1年年化收益率": [4.0],
            "最近一个月涨跌幅": [0.3],
            "近1年上涨星期比例": [75],
            "近3年上涨月份比例": [80],
            "近1年周涨跌幅标准差": [0.15],
            "近3年卡玛比率": [5.0],
            "近3年年化收益率": [4.5],
            "近3年夏普比率": [1.8],
        })
        out = strategy.compute_score(df)
        self.assertIn("综合得分", out.columns)
        self.assertIn("得分_风险控制", out.columns)

    def test_load_score_strategy_steady_profit_priority(self) -> None:
        """应能加载 偏稳收益优先 算分脚本，使用自定义权重。"""
        path = Path(__file__).resolve().parents[1] / "src/filter_score/scores/steady_profit_priority.py"
        self.assertTrue(path.exists())
        strategy = load_score_strategy(path)
        self.assertEqual(strategy.STRATEGY_NAME, "偏稳收益优先")
        df = pd.DataFrame({
            "基金代码": ["001"],
            "基金名称": ["A"],
            "近1年最大回撤率": [0.05],
            "近3年最长回撤修复天数": [50],
            "近3年最大回撤率": [0.10],
            "近1年卡玛比率": [6.0],
            "近1年年化收益率": [4.0],
            "最近一个月涨跌幅": [0.3],
            "近1年上涨星期比例": [75],
            "近3年上涨月份比例": [80],
            "近1年周涨跌幅标准差": [0.15],
            "近3年卡玛比率": [5.0],
            "近3年年化收益率": [4.5],
            "近3年夏普比率": [1.8],
        })
        out = strategy.compute_score(df)
        self.assertIn("综合得分", out.columns)
        self.assertIn("得分_风险控制", out.columns)
        self.assertIn("得分_短期业绩", out.columns)

    def test_run_pipeline_smoke(self) -> None:
        """端到端：小样本 CSV + 最稳健过滤 + 低风险偏债算分。"""
        import tempfile

        project = Path(__file__).resolve().parents[1]
        real_input = project / "result_example/fund_scoreboard_20260301_1_formal_retry_step4_rerun_db.csv"
        if real_input.exists():
            input_csv = real_input
        else:
            # 构造最小样例（含过滤与算分所需列）
            sample = pd.DataFrame({
                "基金代码": ["001", "002"],
                "基金名称": ["A", "B"],
                "近3年年化收益率": [5.0, 2.0],
                "近1年年化收益率": [4.0, 3.5],
                "近3年上涨季度比例": [85, 90],
                "近3年上涨月份比例": [75, 80],
                "近3年月涨跌幅标准差": [1.0, 1.2],
                "近1年夏普比率": [1.5, 1.2],
                "近3年夏普比率": [1.2, 1.1],
                "近1年卡玛比率": [2.0, 1.5],
                "近3年卡玛比率": [1.5, 1.2],
                "近1年最大回撤率": [0.05, 0.06],
                "近3年最长回撤修复天数": [50, 60],
                "近3年最大回撤率": [0.10, 0.12],
                "最近一个月涨跌幅": [0.3, 0.2],
                "近1年上涨星期比例": [75, 70],
                "近3年上涨月份比例": [75, 80],
                "近1年周涨跌幅标准差": [0.15, 0.2],
            })
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                input_csv = Path(f.name)
            sample.to_csv(input_csv, index=False, encoding="utf-8-sig")

        work_dir = project / "artifacts/filter_score_smoke"
        filter_script = project / "src/filter_score/filters/most_stable.py"
        score_script = project / "src/filter_score/scores/low_risk_debt.py"

        code = run_pipeline(
            input_csv=input_csv,
            work_dir=work_dir,
            filter_scripts=[filter_script],
            score_script=score_script,
            progress_interval=0,
        )
        self.assertEqual(code, 0)
        self.assertTrue((work_dir / "filter_result.csv").exists())
        self.assertTrue((work_dir / "scored_result.csv").exists())
        filter_df = pd.read_csv(work_dir / "filter_result.csv", dtype={"基金代码": str}, encoding="utf-8-sig")
        scored_df = pd.read_csv(work_dir / "scored_result.csv", dtype={"基金代码": str}, encoding="utf-8-sig")
        self.assertIn("最稳健原则_是否被过滤", filter_df.columns)
        self.assertIn("综合得分", scored_df.columns)
        self.assertIn("计算策略名称", scored_df.columns)

    def test_cli_smoke(self) -> None:
        """CLI 应能接受参数并完成流程。"""
        project = Path(__file__).resolve().parents[1]
        inp = project / "result_example/fund_scoreboard_20260301_1_formal_retry_step4_rerun_db.csv"
        if not inp.exists():
            self.skipTest("样例 CSV 不存在")
        work = project / "artifacts/filter_score_cli_smoke"
        work.mkdir(parents=True, exist_ok=True)
        filter_script = project / "src/filter_score/filters/most_stable.py"
        score_script = project / "src/filter_score/scores/low_risk_debt.py"
        with patch("sys.argv", [
            "filter_and_score_main",
            "-i", str(inp),
            "-w", str(work),
            "-f", str(filter_script),
            "-s", str(score_script),
            "--progress-interval", "0",
        ]):
            code = main()
        self.assertEqual(code, 0)
        self.assertTrue((work / "scored_result.csv").exists())
