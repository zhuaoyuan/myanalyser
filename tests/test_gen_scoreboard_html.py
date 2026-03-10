"""gen_scoreboard_html 单元测试。"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from myanalyser.tools.gen_scoreboard_html import (
    build_html,
    load_and_prepare,
    load_fund_etl_data,
)


class TestGenScoreboardHtml:
    """gen_scoreboard_html 测试。"""

    def test_load_and_prepare_returns_all_columns(self) -> None:
        """load_and_prepare 应返回完整列。"""
        project = Path(__file__).resolve().parents[1]
        inp = project / "result_example/composite_score_output_0301.csv"
        if not inp.exists():
            pytest.skip("样例 CSV 不存在")
        rows, columns, meta = load_and_prepare(inp)
        assert len(rows) > 0
        assert "基金代码" in columns
        assert "综合得分" in columns
        assert "近1年年化收益率" in columns
        assert "近1年最大回撤率" in columns
        assert set(rows[0].keys()) == set(columns)
        assert meta["total"] == len(rows)

    def test_build_html_without_fund_etl(self) -> None:
        """无 fund_etl 时应生成不含 NAV 图表的 HTML。"""
        rows, columns, meta = load_and_prepare(
            Path(__file__).resolve().parents[1] / "result_example/composite_score_output_0301.csv"
        )
        if not rows:
            pytest.skip("无样例数据")
        html = build_html(rows[:5], columns, meta, "测试看板")
        assert "测试看板" in html
        assert "chartScatter" in html
        assert "近1年最大回撤率" in html
        assert "近1年年化收益率" in html
        assert "chartHist" not in html  # 已移除综合得分分布
        assert "COLUMNS" in html
        assert "colCheckboxes" in html

    def test_cli_smoke(self) -> None:
        """CLI 应能完成基本流程。"""
        project = Path(__file__).resolve().parents[1]
        inp = project / "result_example/composite_score_output_0301.csv"
        if not inp.exists():
            pytest.skip("样例 CSV 不存在")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "scoreboard.html"
            import subprocess
            result = subprocess.run(
                ["python", str(project / "tools/gen_scoreboard_html.py"), "-i", str(inp), "-o", str(out)],
                capture_output=True,
                text=True,
                cwd=project.parent,
            )
            assert result.returncode == 0, result.stderr
            assert out.exists()
            content = out.read_text(encoding="utf-8")
            assert "RAW_DATA" in content

    def test_load_fund_etl_data(self) -> None:
        """load_fund_etl_data 应能加载净值和人事数据。"""
        project = Path(__file__).resolve().parents[1]
        fund_etl = project / "tests/baseline/mini_case/input/fund_etl"
        if not (fund_etl / "fund_adjusted_nav_by_code/000006.csv").exists():
            pytest.skip("mini_case fund_etl 不存在")
        nav_by_code, personnel_by_code = load_fund_etl_data(fund_etl, ["000006", "000003"])
        assert "000006" in nav_by_code
        assert len(nav_by_code["000006"]) > 0
        assert "000006" in personnel_by_code
        assert personnel_by_code["000006"].startswith("2024")
