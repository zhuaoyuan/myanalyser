"""低风险偏债得分算分策略。

规则同 compute_fund_composite_score：
  A. 风险控制 (35%): 近1年最大回撤40% + 近3年最长回撤修复天数40% + 近3年最大回撤20%
  B. 短期业绩 (30%): 近1年卡玛比率50% + 近1年年化收益30% + 最近一个月涨跌幅20%
  C. 持有体验 (20%): 近1年上涨星期比例30% + 近3年上涨月比例30% + 近1年周标准差40%
  D. 长期业绩 (15%): 近3年卡玛比率50% + 近3年年化收益30% + 近3年夏普比率20%
"""

from __future__ import annotations

import sys
from pathlib import Path

# 运行前需保证 PYTHONPATH 含工作区根（或由 filter_and_score_main 已注入）
try:
    from myanalyser.src.compute_fund_composite_score import compute_composite_score
except ModuleNotFoundError:
    _root = Path(__file__).resolve().parents[4]  # 工作区根 (finance)
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from myanalyser.src.compute_fund_composite_score import compute_composite_score

STRATEGY_NAME = "低风险偏债得分"


def compute_score(df):
    """对 DataFrame 计算综合得分及二级指标。"""
    import pandas as pd

    return compute_composite_score(df)
