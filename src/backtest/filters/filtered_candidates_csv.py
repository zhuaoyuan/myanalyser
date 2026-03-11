"""从 filtered_fund_candidates.csv 取白名单的过滤器。

读取环境变量 FILTERED_FUND_CANDIDATES_CSV，若指向的 CSV 存在则解析：
  列：基金编码、是否过滤、过滤原因
  取 是否过滤=否 的基金编码与 candidates 取交集。

未设置或文件不存在时原样返回 candidates。
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ENV_VAR = "FILTERED_FUND_CANDIDATES_CSV"


def _safe_code(value: object) -> str:
    return str(value).strip().zfill(6)


class FilteredCandidatesCsvFilter:
    def filter(self, candidates: set[str]) -> set[str]:
        path = os.environ.get(ENV_VAR)
        if not path:
            return candidates
        p = Path(path).resolve()
        if not p.exists() or not p.is_file():
            return candidates

        df = pd.read_csv(p, dtype={"基金编码": str}, encoding="utf-8-sig")
        if df.empty or "基金编码" not in df.columns:
            return candidates
        if "是否过滤" in df.columns:
            allowed_df = df[df["是否过滤"].astype(str).str.strip() == "否"]
        else:
            allowed_df = df
        allowed = {_safe_code(c) for c in allowed_df["基金编码"].dropna().tolist()}
        return candidates & allowed
