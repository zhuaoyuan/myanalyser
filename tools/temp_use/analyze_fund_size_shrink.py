#!/usr/bin/env python3
"""分析：2024年市值>2亿的基金，在最新数据中变成<2亿的基金。"""

import re
import pandas as pd

BASE = "/Users/zhuaoyuan/cursor-workspace/finance/finance-runs/run_20260310_191534/data/versions/20260310_191534/fund_etl"
GMBD = f"{BASE}/fund_gmbd.csv"
OVERVIEW = f"{BASE}/fund_overview.csv"


def parse_overview_asset(s: str) -> float | None:
    """从 资产规模 列解析出数值（亿元）。如 '59.36亿元（截止至：...）' -> 59.36"""
    if pd.isna(s) or not s:
        return None
    m = re.search(r"([\d.]+)\s*亿元", str(s))
    return float(m.group(1)) if m else None


def main():
    # 1. fund_gmbd: 2024年 期末净资产（亿元）> 2
    gmbd = pd.read_csv(GMBD)
    # 选列：优先 期末净资产（亿元），若无则用 期末净资产（亿）
    col_na = [c for c in gmbd.columns if "期末净资产" in c and "亿" in c]
    if not col_na:
        raise SystemExit("fund_gmbd 中未找到期末净资产列")
    na_col = col_na[0]
    gmbd["日期"] = pd.to_datetime(gmbd["日期"], errors="coerce")
    gmbd_2024 = gmbd[gmbd["日期"].dt.year == 2024].copy()
    gmbd_2024["净值_float"] = pd.to_numeric(gmbd_2024[na_col], errors="coerce")
    above_2 = gmbd_2024[gmbd_2024["净值_float"] > 2]
    codes_above_2 = set(above_2["基金代码"].astype(str).str.strip().unique())
    # 记录 2024 年该基金的最大规模
    max_na = gmbd_2024.groupby("基金代码")["净值_float"].max().to_dict()

    # 2. fund_overview: 资产规模 < 2
    overview = pd.read_csv(OVERVIEW)
    overview["基金代码"] = overview["基金代码"].astype(str).str.strip()
    overview["资产_float"] = overview["资产规模"].apply(parse_overview_asset)
    below_2 = overview[overview["资产_float"].notna() & (overview["资产_float"] < 2)]
    codes_below_2 = set(below_2["基金代码"].unique())

    # 3. 交集：2024年>2亿，最新<2亿
    shrink = codes_above_2 & codes_below_2
    shrink_df = overview[overview["基金代码"].isin(shrink)].copy()
    shrink_df["2024_max_亿"] = shrink_df["基金代码"].map(
        lambda c: max_na.get(c, max_na.get(c.strip()))
    )
    shrink_df["最新_亿"] = shrink_df["资产_float"]
    shrink_df = shrink_df.sort_values("最新_亿", ascending=True)

    print("=" * 60)
    print("2024年市值>2亿 → 最新<2亿 的基金（规模萎缩）")
    print("=" * 60)
    print(f"符合条件的基金数：{len(shrink_df)}")
    print()
    cols = ["基金代码", "基金简称", "2024_max_亿", "最新_亿", "资产规模"]
    print(shrink_df[cols].to_string(index=False))
    return shrink_df


if __name__ == "__main__":
    main()
