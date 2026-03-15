#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析 fund_fee_complete.csv 中业务主键重复的记录，并抓取 akshare 原始数据找出未解析的文本格式。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PREP_DIR = Path(__file__).resolve().parent.parent / "prep"
if str(PREP_DIR) not in sys.path:
    sys.path.insert(0, str(PREP_DIR))

from fetch_fund_fee import _parse_amount_tier, _parse_period_tier

CSV_PATH = Path(
    "/Users/zhuaoyuan/cursor-workspace/finance/finance-runs/run_20260310_191534"
    "/data/versions/20260310_191534/fund_etl/fee/fund_fee_complete.csv"
)


def main() -> None:
    df = pd.read_csv(CSV_PATH, dtype=str, encoding="utf-8-sig")
    key_cols = ["数据类型", "金额阶梯起点", "金额阶梯终点", "持仓期限阶梯起点", "持仓期限阶梯终点"]
    if not all(c in df.columns for c in key_cols):
        print("缺少列:", [c for c in key_cols if c not in df.columns])
        return

    # 业务主键重复组
    dup = df.groupby(key_cols, dropna=False).filter(lambda g: len(g) > 1)
    if dup.empty:
        print("无重复")
        return

    # 每组取基金编码与费率
    dup_groups = dup.groupby(key_cols, dropna=False)
    print(f"重复组数: {len(dup_groups)}")
    print()

    # 抽样：取前5组，展示基金编码和费率差异
    for i, (key, grp) in enumerate(dup_groups):
        if i >= 5:
            break
        codes = grp["基金编码"].unique().tolist()
        fees = grp["费率"].unique().tolist()
        print(f"组{i+1} 主键: {key}")
        print(f"  基金数: {len(codes)}, 费率: {fees[:5]}{'...' if len(fees)>5 else ''}")
        print()

    # 抓取 akshare 原始数据，找出未解析的 适用金额/适用期限
    try:
        import akshare as ak
    except ImportError:
        print("需要 akshare，跳过原始数据抓取")
        return

    # 取若干有重复的基金编码
    sample_codes = dup["基金编码"].drop_duplicates().head(20).tolist()
    unparsed_amt: set[str] = set()
    unparsed_period: set[str] = set()
    all_amt: set[str] = set()
    all_period: set[str] = set()

    for code in sample_codes:
        for indicator, amt_col, period_col in [
            ("申购费率（前端）", True, True),
            ("赎回费率", False, True),
        ]:
            try:
                d = ak.fund_fee_em(symbol=code, indicator=indicator)
            except Exception:
                continue
            if d is None or d.empty:
                continue
            ac = "适用金额" if "适用金额" in d.columns else (d.columns[0] if len(d.columns) > 0 else None)
            pc = "适用期限" if "适用期限" in d.columns else (d.columns[1] if len(d.columns) > 1 else None)
            for _, row in d.iterrows():
                amt_t = str(row.get(ac, "")).strip() if ac else ""
                per_t = str(row.get(pc, "")).strip() if pc else ""
                if amt_t and amt_t not in ("---", "—", "nan"):
                    all_amt.add(amt_t)
                    if _parse_amount_tier(amt_t) == (None, None):
                        unparsed_amt.add(amt_t)
                if per_t and per_t not in ("---", "—", "nan"):
                    all_period.add(per_t)
                    if _parse_period_tier(per_t) == (None, None):
                        unparsed_period.add(per_t)

    print("=== 适用金额：未解析的原始文本 ===")
    for t in sorted(unparsed_amt):
        print(f"  {repr(t)}")
    print()

    print("=== 适用期限：未解析的原始文本 ===")
    for t in sorted(unparsed_period):
        print(f"  {repr(t)}")
    print()

    if not unparsed_amt and not unparsed_period:
        print("当前抽样中所有文本均已解析。重复可能源于：1) akshare 多行同阶梯；2) 其他基金存在未覆盖格式。")


if __name__ == "__main__":
    main()
