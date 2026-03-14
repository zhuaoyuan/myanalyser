#!/usr/bin/env python3
"""
基于 fund_gmbd 数据，筛选出 2025 年规模变为 0、此前曾大于 2 亿元的基金，
并查看其净值数据情况。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="筛选 2025 年规模归零的基金并查看净值")
    parser.add_argument(
        "--gmbd",
        type=Path,
        default=Path("finance-runs/run_20260310_191534/data/versions/20260310_191534/fund_etl/fund_gmbd.csv"),
        help="fund_gmbd.csv 路径",
    )
    parser.add_argument(
        "--nav-dir",
        type=Path,
        default=Path("finance-runs/run_20260310_191534/data/versions/20260310_191534/fund_etl/fund_adjusted_nav_by_code"),
        help="fund_adjusted_nav_by_code 目录路径",
    )
    parser.add_argument("--threshold", type=float, default=2.0, help="此前规模阈值（亿元）")
    args = parser.parse_args()

    gmbd_path = args.gmbd.resolve()
    nav_dir = args.nav_dir.resolve()

    if not gmbd_path.exists():
        print(f"未找到规模数据: {gmbd_path}")
        return

    # 使用 pandas 读取规模数据（处理混合类型、空值等）
    nav_col = "期末净资产（亿元）"
    df = pd.read_csv(
        gmbd_path, encoding="utf-8", low_memory=False, dtype={"基金代码": str}
    )
    if nav_col not in df.columns:
        nav_col = "期末净资产（亿）"
    if nav_col not in df.columns:
        print("未找到期末净资产列，可用列:", list(df.columns))
        return

    df["基金代码"] = df["基金代码"].astype(str).str.strip()
    df["日期"] = df["日期"].astype(str)
    df["净资产"] = pd.to_numeric(df[nav_col], errors="coerce")

    # 2025 年规模 <= 0.001 视为归零
    MIN_ZERO = 0.001
    df_2025 = df[df["日期"].str.startswith("2025", na=False)]
    zero_2025_codes = set(
        df_2025.loc[df_2025["净资产"].notna() & (df_2025["净资产"] <= MIN_ZERO), "基金代码"]
    )

    # 此前曾有规模 > threshold 的基金
    df_before = df[~df["日期"].str.startswith("2025", na=False)]
    large_scale_codes = set(
        df_before.loc[df_before["净资产"].notna() & (df_before["净资产"] > args.threshold), "基金代码"]
    )

    zero_scale_funds = sorted(zero_2025_codes & large_scale_codes)
    print(f"符合条件（2025 年规模为 0，此前曾 > {args.threshold} 亿元）的基金共 {len(zero_scale_funds)} 只:")
    for c in zero_scale_funds[:50]:  # 先列前 50 只
        print(f"  {c}")
    if len(zero_scale_funds) > 50:
        print(f"  ... 等共 {len(zero_scale_funds)} 只")

    # 查看这些基金的净值情况：优先展示有净值文件的
    print("\n" + "=" * 60)
    print("净值数据概况:")
    print("=" * 60)

    # 先找出有净值文件的基金
    funds_with_nav = [
        c for c in zero_scale_funds
        if (nav_dir / f"{str(c).zfill(6)}.csv").exists()
    ]
    print(f"其中 {len(funds_with_nav)} 只有净值数据，展示如下（最多 10 只）:")

    for code in funds_with_nav[:10]:
        code_padded = str(code).zfill(6)
        nav_file = nav_dir / f"{code_padded}.csv"
        if not nav_file.exists():
            print(f"\n【{code}】净值文件不存在: {nav_file}")
            continue

        with nav_file.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            print(f"\n【{code}】净值文件为空")
            continue

        # 按日期排序（取最近的）
        def sort_key(r: dict) -> str:
            return r.get("净值日期", "") or r.get("日期", "")

        rows_sorted = sorted(rows, key=sort_key)
        first_date = sort_key(rows_sorted[0])
        last_date = sort_key(rows_sorted[-1])
        last_nav = rows_sorted[-1].get("单位净值") or rows_sorted[-1].get("复权净值", "")

        print(f"\n【{code}】")
        print(f"  记录条数: {len(rows_sorted)}")
        print(f"  日期范围: {first_date} ~ {last_date}")
        print(f"  最新净值: {last_nav}")
        if len(rows_sorted) >= 5:
            print("  最近 5 条:")
            for r in rows_sorted[-5:]:
                print(f"    {r.get('净值日期', r.get('日期'))} 单位净值={r.get('单位净值')} 复权净值={r.get('复权净值')}")


if __name__ == "__main__":
    main()
