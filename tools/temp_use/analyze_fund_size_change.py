#!/usr/bin/env python3
"""
分析：2024年市值>2亿的基金，在最新数据中变成<2亿的有哪些
"""
import re
import pandas as pd
from pathlib import Path

BASE = Path("/Users/zhuaoyuan/cursor-workspace/finance")

# 输入文件
GMBD_CSV = BASE / "finance-runs/run_20260310_191534/data/versions/20260310_191534/fund_etl/fund_gmbd.csv"
OVERVIEW_CSV = BASE / "finance-runs/run_20260301_1_formal_retry_step4_rerun/data/versions/20260301_1_formal_retry_step4_rerun/fund_etl/fund_overview.csv"


def parse_overview_asset(s: str) -> float | None:
    """从 资产规模 列解析出亿元数值，例如 '0.53亿元（截止至：...）' -> 0.53"""
    if pd.isna(s) or not isinstance(s, str):
        return None
    m = re.search(r"([\d.]+)\s*亿元", s)
    return float(m.group(1)) if m else None


def main():
    # 1. 读取 fund_gmbd，筛选 2024 年 期末净资产（亿元）> 2 的基金
    gmbd = pd.read_csv(GMBD_CSV)
    gmbd["日期"] = pd.to_datetime(gmbd["日期"])
    gmbd_2024 = gmbd[gmbd["日期"].dt.year == 2024].copy()

    # 期末净资产（亿元）列名
    col_na = "期末净资产（亿元）"
    if col_na not in gmbd_2024.columns:
        print("可用列:", list(gmbd_2024.columns))
        return

    gmbd_2024[col_na] = pd.to_numeric(gmbd_2024[col_na], errors="coerce")
    funds_2024_gt2 = set(gmbd_2024[gmbd_2024[col_na] > 2]["基金代码"].astype(str).str.zfill(6))

    print(f"2024年市值>2亿的基金数量: {len(funds_2024_gt2)}")

    # 2. 读取 fund_overview，解析资产规模
    overview = pd.read_csv(OVERVIEW_CSV)
    overview["基金代码"] = overview["基金代码"].astype(str).str.zfill(6)
    overview["资产规模_亿"] = overview["资产规模"].apply(parse_overview_asset)

    # 最新数据中 < 2 亿的基金
    overview_lt2 = overview[overview["资产规模_亿"].notna() & (overview["资产规模_亿"] < 2)]
    funds_latest_lt2 = set(overview_lt2["基金代码"].tolist())

    print(f"最新数据中市值<2亿的基金数量: {len(funds_latest_lt2)}")

    # 3. 交集：2024>2亿 且 最新<2亿
    downgraded = funds_2024_gt2 & funds_latest_lt2
    print(f"\n2024年>2亿、最新<2亿的基金数量: {len(downgraded)}")

    # 输出明细：基金代码、基金简称、2024年规模、最新规模
    result_rows = []
    for code in sorted(downgraded):
        # 2024 年该基金的最大规模
        g = gmbd_2024[gmbd_2024["基金代码"].astype(str).str.zfill(6) == code]
        na_2024 = g[col_na].max() if len(g) else None

        ov = overview[overview["基金代码"] == code]
        na_latest = ov["资产规模_亿"].iloc[0] if len(ov) else None
        name = ov["基金简称"].iloc[0] if len(ov) and "基金简称" in ov.columns else ""

        result_rows.append({
            "基金代码": code,
            "基金简称": name,
            "2024年最大规模(亿)": round(na_2024, 2) if pd.notna(na_2024) else None,
            "最新规模(亿)": round(na_latest, 2) if pd.notna(na_latest) else None,
        })

    df = pd.DataFrame(result_rows)
    df = df.sort_values("2024年最大规模(亿)", ascending=False)

    # 导出到 myanalyser/tmp
    out_dir = BASE / "myanalyser/tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fund_downgraded_2024gt2_to_latestlt2.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n已导出: {out_path}")

    print("\n基金明细（按2024年规模降序）：")
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
