#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 fund_gmbd 筛选 2025 年清盘基金，并查看其净值情况。

用法:
  python tools/analyze_liquidated_funds_2025.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# 数据路径
RUN_ROOT = Path(__file__).resolve().parents[1].parent / "finance-runs" / "run_20260310_191534"
DATA = RUN_ROOT / "data" / "versions" / "20260310_191534" / "fund_etl"
GMBD_CSV = DATA / "fund_gmbd.csv"
NAV_DIR = DATA / "fund_adjusted_nav_by_code"


def main() -> None:
    if not GMBD_CSV.exists():
        print(f"未找到: {GMBD_CSV}")
        sys.exit(1)

    df = pd.read_csv(GMBD_CSV, low_memory=False)
    # 规范列名
    nav_col = "期末净资产（亿元）"
    if nav_col not in df.columns:
        print("列名:", df.columns.tolist())
        sys.exit(1)

    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df["基金代码"] = df["基金代码"].astype(str).str.zfill(6)

    # 2025 年 期末净资产 = 0 的基金
    df_2025 = df[df["日期"].dt.year == 2025].copy()
    zero_2025 = df_2025[df_2025[nav_col] == 0]
    codes_zero = zero_2025["基金代码"].unique().tolist()

    print("=" * 60)
    print("2025 年期末净资产=0 的基金（视为清盘）")
    print("=" * 60)
    print(f"基金数量: {len(codes_zero)}")
    print(f"示例: {codes_zero[:15]}")
    print()

    # 哪些有净值文件
    nav_files = {f.stem: f for f in NAV_DIR.glob("*.csv")} if NAV_DIR.exists() else {}
    has_nav = [c for c in codes_zero if c in nav_files]
    no_nav = [c for c in codes_zero if c not in nav_files]

    print("有调整后净值文件的:", len(has_nav), "只")
    print("无调整后净值文件的:", len(no_nav), "只")
    print()

    if not has_nav:
        print("没有同时满足「2025年净资产=0」且「有净值文件」的基金。")
        return

    print("=" * 60)
    print("有净值文件的 2025 年清盘基金详情")
    print("=" * 60)

    for code in has_nav[:15]:  # 最多展示 15 只
        nav_path = nav_files[code]
        ndf = pd.read_csv(nav_path)
        ndf["净值日期"] = pd.to_datetime(ndf["净值日期"], errors="coerce")
        ndf = ndf.dropna(subset=["净值日期"]).sort_values("净值日期")

        if ndf.empty:
            print(f"\n基金 {code}: 净值文件无有效日期")
            continue

        first_d = ndf["净值日期"].iloc[0]
        last_d = ndf["净值日期"].iloc[-1]
        first_nav = ndf["单位净值"].iloc[0]
        last_nav = ndf["单位净值"].iloc[-1]

        # 2025 年净值
        ndf_2025 = ndf[ndf["净值日期"].dt.year == 2025]
        n_2025 = len(ndf_2025)

        # 规模为 0 的季度
        qzeros = zero_2025[zero_2025["基金代码"] == code]["日期"].drop_duplicates()
        qlist = sorted(qzeros.dt.strftime("%Y-%m-%d").tolist()) if len(qzeros) else []

        print(f"\n基金 {code}:")
        print(f"  净值: {len(ndf)} 条, {first_d.strftime('%Y-%m-%d')} ~ {last_d.strftime('%Y-%m-%d')}")
        print(f"  首净={first_nav:.4f}, 末净={last_nav:.4f}")
        print(f"  2025年净值记录: {n_2025} 条")
        print(f"  规模为0的季度: {qlist}")

        # 规模首次为 0 前后的净值
        if qlist:
            q0 = qlist[0]
            before = ndf[ndf["净值日期"] < q0].tail(3)
            after = ndf[ndf["净值日期"] >= q0].head(3)
            if not before.empty:
                print(f"  规模为0前最近净值: {before[['净值日期','单位净值']].to_dict('records')}")
            if not after.empty:
                print(f"  规模为0后最近净值: {after[['净值日期','单位净值']].to_dict('records')}")


if __name__ == "__main__":
    main()
