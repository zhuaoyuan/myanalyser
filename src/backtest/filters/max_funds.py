"""按数量截断的过滤器。

读取环境变量 FUND_BACKTEST_MAX_FUNDS，若为正整数则保留前 N 个（按编码排序以保证确定性）。
未设置或非正整数时原样返回 candidates。
"""

from __future__ import annotations

import os


ENV_VAR = "FUND_BACKTEST_MAX_FUNDS"


class MaxFundsFilter:
    def filter(self, candidates: set[str]) -> set[str]:
        raw = os.environ.get(ENV_VAR)
        if not raw:
            return candidates
        try:
            n = int(raw)
        except ValueError:
            return candidates
        if n <= 0:
            return candidates
        sorted_codes = sorted(candidates)
        return set(sorted_codes[:n])
