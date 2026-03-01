"""compute_fund_composite_score 单元测试。"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from myanalyser.src.compute_fund_composite_score import (
    compute_composite_score,
    main,
)


def test_compute_composite_score_adds_columns() -> None:
    """应添加得分_*、综合得分、综合排名列。"""
    import pandas as pd

    df = pd.DataFrame({
        "基金代码": ["001", "002"],
        "基金名称": ["A", "B"],
        "近1年最大回撤率": [0.05, 0.10],
        "近3年最长回撤修复天数": [30, 60],
        "近3年最大回撤率": [0.08, 0.12],
        "近1年卡玛比率": [5.0, 8.0],
        "近1年年化收益率": [3.0, 5.0],
        "最近一个月涨跌幅": [0.2, 0.5],
        "近1年上涨星期比例": [70, 80],
        "近3年上涨月份比例": [75, 85],
        "近1年周涨跌幅标准差": [0.2, 0.1],
        "近3年卡玛比率": [4.0, 6.0],
        "近3年年化收益率": [4.0, 5.5],
        "近3年夏普比率": [1.5, 2.0],
    })
    result = compute_composite_score(df)
    for col in ["得分_风险控制", "得分_短期业绩", "得分_持有体验", "得分_长期业绩", "综合得分", "综合排名"]:
        assert col in result.columns, f"缺少列: {col}"
    assert len(result) == 2


def test_compute_composite_score_ranking() -> None:
    """综合得分越高排名越靠前（数字越小）。"""
    import pandas as pd

    df = pd.DataFrame({
        "基金代码": ["001", "002", "003"],
        "基金名称": ["A", "B", "C"],
        "近1年最大回撤率": [0.02, 0.05, 0.10],
        "近3年最长回撤修复天数": [20, 40, 80],
        "近3年最大回撤率": [0.03, 0.06, 0.12],
        "近1年卡玛比率": [10.0, 6.0, 3.0],
        "近1年年化收益率": [6.0, 4.0, 2.0],
        "最近一个月涨跌幅": [0.5, 0.3, 0.1],
        "近1年上涨星期比例": [85, 75, 65],
        "近3年上涨月份比例": [90, 80, 70],
        "近1年周涨跌幅标准差": [0.05, 0.15, 0.30],
        "近3年卡玛比率": [8.0, 5.0, 2.0],
        "近3年年化收益率": [5.5, 4.0, 2.5],
        "近3年夏普比率": [2.5, 1.5, 0.5],
    })
    result = compute_composite_score(df)
    # 001 各项指标最佳，应排名第 1
    rank_001 = result.loc[result["基金代码"] == "001", "综合排名"].iloc[0]
    assert rank_001 == 1


def test_cli_smoke() -> None:
    """CLI 可从输入 CSV 生成输出 CSV。"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        inp = tmp_path / "in.csv"
        out = tmp_path / "out.csv"
        df = pd.DataFrame({
            "基金代码": ["001"],
            "基金名称": ["测试基金"],
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
        df.to_csv(inp, index=False, encoding="utf-8-sig")

        with patch("sys.argv", ["compute_fund_composite_score", "-i", str(inp), "-o", str(out)]):
            exit_code = main()
        assert exit_code == 0
        assert out.exists()
        out_df = pd.read_csv(out, dtype={"基金代码": str})
        assert "综合得分" in out_df.columns
        assert "综合排名" in out_df.columns


def test_cli_missing_input_returns_nonzero() -> None:
    """输入文件不存在时应返回非 0。"""
    with patch("sys.argv", ["compute_fund_composite_score", "-i", "/nonexistent/path.csv", "-o", "/tmp/out.csv"]):
        code = main()
    assert code != 0
