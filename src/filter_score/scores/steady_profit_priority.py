"""偏稳收益优先得分算分策略。

规则（权重调整）：
  A. 风险控制 (25%): 近1年最大回撤40% + 近3年最长回撤修复天数40% + 近3年最大回撤20%
  B. 短期业绩 (35%): 近1年卡玛比率30% + 近1年年化收益50% + 最近一个月涨跌幅20%
  C. 持有体验 (15%): 近1年上涨星期比例30% + 近3年上涨月比例30% + 近1年周标准差40%
  D. 长期业绩 (25%): 近3年卡玛比率30% + 近3年年化收益50% + 近3年夏普比率20%
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from myanalyser.src.compute_fund_composite_score import compute_composite_score
except ModuleNotFoundError:
    _root = Path(__file__).resolve().parents[4]  # 工作区根 (finance)
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from myanalyser.src.compute_fund_composite_score import compute_composite_score

STRATEGY_NAME = "偏稳收益优先"

# 偏稳收益优先：组权重与组内指标权重
_RISK_METRICS = [
    ("近1年最大回撤率", 0.4, "asc"),
    ("近3年最长回撤修复天数", 0.4, "asc"),
    ("近3年最大回撤率", 0.2, "asc"),
]
_SHORT_TERM_METRICS = [
    ("近1年卡玛比率", 0.3, "desc"),
    ("近1年年化收益率", 0.5, "desc"),
    ("最近一个月涨跌幅", 0.2, "desc"),
]
_HOLDING_METRICS = [
    ("近1年上涨星期比例", 0.3, "desc"),
    ("近3年上涨月份比例", 0.3, "desc"),
    ("近1年周涨跌幅标准差", 0.4, "asc"),
]
_LONG_TERM_METRICS = [
    ("近3年卡玛比率", 0.3, "desc"),
    ("近3年年化收益率", 0.5, "desc"),
    ("近3年夏普比率", 0.2, "desc"),
]

STEADY_PROFIT_GROUPS = [
    ("风险控制", 0.25, _RISK_METRICS),
    ("短期业绩", 0.35, _SHORT_TERM_METRICS),
    ("持有体验", 0.15, _HOLDING_METRICS),
    ("长期业绩", 0.25, _LONG_TERM_METRICS),
]


def compute_score(df):
    """对 DataFrame 计算综合得分及二级指标（偏稳收益优先权重）。"""
    return compute_composite_score(df, secondary_groups=STEADY_PROFIT_GROUPS)
