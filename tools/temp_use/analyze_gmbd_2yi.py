#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 fund_gmbd.csv：
1. 找出所有期末净资产达到过2亿，后来又低于2亿的基金
2. 统计2023年以前期末净资产达到2亿的 case 中，3年后仍在2亿以上的占比
3. 统计2015～2023年、成立时间>=3年、期末净资产达到2亿的 case 中，1年/3年后仍在2亿的占比
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def load_gmbd(csv_path: str | Path) -> pd.DataFrame:
    """加载 fund_gmbd.csv，解析日期和期末净资产。"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    # 列名可能带 BOM 或空格
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    # 期末净资产（亿元）
    col = "期末净资产（亿元）"
    if col not in df.columns:
        # 尝试其他可能列名
        for c in df.columns:
            if "期末净资产" in c and "亿元" in c:
                col = c
                break
        else:
            raise ValueError(f"未找到期末净资产列，现有列: {list(df.columns)}")
    df["nav_yi"] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df.dropna(subset=["nav_yi", "date"])
    return df


def task1_fund_reached_then_below(df: pd.DataFrame, threshold: float = 2.0) -> list[str]:
    """
    找出所有期末净资产达到过 threshold 亿，后来又低于 threshold 亿的基金。
    """
    codes = []
    for code, g in df.groupby("基金代码"):
        nav = g.sort_values("date")["nav_yi"]
        reached = (nav >= threshold).any()
        below_later = False
        if reached:
            # 找到首次达到 threshold 的日期
            first_reach_idx = (nav >= threshold).idxmax()
            first_reach_date = g.loc[first_reach_idx, "date"]
            # 检查之后是否有低于 threshold 的记录
            after = g[g["date"] > first_reach_date]
            if len(after) > 0 and (after["nav_yi"] < threshold).any():
                below_later = True
        if reached and below_later:
            codes.append(str(code))
    return sorted(codes)


def task2_three_year_survival(df: pd.DataFrame, threshold: float = 2.0, cutoff_year: int = 2023) -> dict:
    """
    在 2023 年以前期末净资产达到 2 亿的 case 中，统计 3 年后仍在 2 亿以上的占比。
    """
    cutoff = pd.Timestamp(f"{cutoff_year}-01-01")
    cases = []
    grouped = df.groupby("基金代码")
    for code, g in grouped:
        g = g.sort_values("date").reset_index(drop=True)
        before = g[g["date"] < cutoff]
        if before.empty:
            continue
        above = before[before["nav_yi"] >= threshold]
        if above.empty:
            continue
        first_reach = above.iloc[0]["date"]
        three_years_later = first_reach + pd.DateOffset(years=3)
        future = g[g["date"] >= three_years_later]
        if len(future) == 0:
            continue
        nearest = future.iloc[0]
        still_above = nearest["nav_yi"] >= threshold
        cases.append((code, first_reach, still_above))
    if not cases:
        return {"total": 0, "still_above": 0, "pct": 0.0}
    total = len(cases)
    still_above = sum(1 for _, _, sa in cases if sa)
    return {"total": total, "still_above": still_above, "pct": 100.0 * still_above / total}


def _parse_foundation_date(raw_value: object) -> pd.Timestamp | None:
    """解析成立日期，如 2013年03月08日 / 52.127亿份"""
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    match = re.search(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", text)
    if match:
        y, m, d = match.groups()
        dt = pd.to_datetime(f"{y}-{int(m):02d}-{int(d):02d}", errors="coerce")
        return None if pd.isna(dt) else dt
    match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if match:
        dt = pd.to_datetime(match.group(0), errors="coerce")
        return None if pd.isna(dt) else dt
    return None


def load_inception_dates(overview_path: Path) -> dict[str, pd.Timestamp]:
    """从 fund_overview.csv 加载基金成立日期。"""
    df = pd.read_csv(overview_path, dtype={"基金代码": str}, encoding="utf-8-sig")
    col = "成立日期/规模" if "成立日期/规模" in df.columns else "成立日期"
    if col not in df.columns:
        raise ValueError(f"未找到成立日期列: {list(df.columns)}")
    out: dict[str, pd.Timestamp] = {}
    for row in df.to_dict("records"):
        code = str(row["基金代码"]).strip().zfill(6)
        dt = _parse_foundation_date(row.get(col))
        if dt is not None:
            out[code] = dt
    return out


def task3_survival_with_inception(
    df: pd.DataFrame,
    inception: dict[str, pd.Timestamp],
    threshold: float,
    start_year: int,
    end_year: int,
    min_age_years: float = 3.0,
) -> dict:
    """
    2015～2023年、成立时间>=3年、期末净资产达到2亿的 case 中，
    统计 1年后、3年后 仍在2亿以上的占比。

    语义：1年/3年后取报告期末日期 >= 基准日+n 年的第一条记录（非最近邻）。
    """
    start_d = pd.Timestamp(f"{start_year}-01-01")
    end_d = pd.Timestamp(f"{end_year}-12-31")
    window = df[(df["date"] >= start_d) & (df["date"] <= end_d)].copy()
    min_age_days = min_age_years * 365.25
    full_by_code = {c: g.sort_values("date") for c, g in df.groupby("基金代码")}

    cases: list[tuple[str, pd.Timestamp, bool | None, bool | None]] = []
    for code, g in window.groupby("基金代码"):
        inc = inception.get(str(code).zfill(6))
        if inc is None:
            continue
        g = g.sort_values("date").reset_index(drop=True)
        g_full = full_by_code.get(code)
        if g_full is None or g_full.empty:
            continue
        for _, row in g.iterrows():
            if row["nav_yi"] < threshold:
                continue
            d = row["date"]
            age_days = (d - inc).days
            if age_days < min_age_days:
                continue
            one_later = d + pd.DateOffset(years=1)
            three_later = d + pd.DateOffset(years=3)
            # 取目标日期之后的第一条报告（非最近邻）
            f1 = g_full[g_full["date"] >= one_later]
            f3 = g_full[g_full["date"] >= three_later]
            # 取目标日期之后的第一条报告（非最近邻）
            still_1: bool | None = f1.iloc[0]["nav_yi"] >= threshold if len(f1) > 0 else None
            still_3: bool | None = f3.iloc[0]["nav_yi"] >= threshold if len(f3) > 0 else None
            cases.append((code, d, still_1, still_3))

    total = len(cases)
    if total == 0:
        return {"total": 0, "still_1y": 0, "pct_1y": 0.0, "still_3y": 0, "pct_3y": 0.0}
    valid_1y = [c for c in cases if c[2] is not None]
    valid_3y = [c for c in cases if c[3] is not None]
    still_1y = sum(1 for c in valid_1y if c[2])
    still_3y = sum(1 for c in valid_3y if c[3])
    return {
        "total": total,
        "still_1y": still_1y,
        "valid_1y": len(valid_1y),
        "pct_1y": 100.0 * still_1y / len(valid_1y) if valid_1y else 0.0,
        "still_3y": still_3y,
        "valid_3y": len(valid_3y),
        "pct_3y": 100.0 * still_3y / len(valid_3y) if valid_3y else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="分析 fund_gmbd 期末净资产 2 亿相关统计")
    parser.add_argument(
        "csv",
        nargs="?",
        default="finance-runs/run_20260310_191534/data/versions/20260310_191534/fund_etl/fund_gmbd.csv",
        help="fund_gmbd.csv 路径",
    )
    parser.add_argument("--threshold", type=float, default=2.0, help="阈值（亿元）")
    parser.add_argument("--cutoff-year", type=int, default=2023, help="截止年份（不含）")
    parser.add_argument("--from-year", type=int, default=None, help="数据起始年份（含），如 2015 表示仅用 2015 年及以后的数据")
    parser.add_argument("--overview", help="fund_overview.csv 路径，提供则执行任务3（成立时间>=3年、2015～2023年达2亿的1年/3年后占比）")
    parser.add_argument("--task3-only", action="store_true", help="仅执行任务3，需配合 --overview")
    parser.add_argument("-o", "--out", help="任务1基金列表输出文件路径")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        csv_path = Path(__file__).resolve().parents[1] / ".." / csv_path
    if not csv_path.exists():
        print(f"文件不存在: {args.csv}", file=sys.stderr)
        return 1

    df = load_gmbd(csv_path)
    if args.from_year is not None:
        min_date = pd.Timestamp(f"{args.from_year}-01-01")
        df = df[df["date"] >= min_date].copy()
        print(f"限定 {args.from_year} 年及以后: {len(df)} 条记录，{df['基金代码'].nunique()} 只基金\n")
    else:
        print(f"加载 {len(df)} 条记录，{df['基金代码'].nunique()} 只基金\n")

    # 任务 1、2（--task3-only 时跳过）
    if not args.task3_only:
        codes = task1_fund_reached_then_below(df, args.threshold)
        print("=" * 60)
        print("任务 1：期末净资产达到过 2 亿、后来又低于 2 亿的基金")
        print("=" * 60)
        print(f"共 {len(codes)} 只")
        if args.out:
            Path(args.out).write_text("\n".join(codes), encoding="utf-8")
            print(f"完整列表已保存至: {args.out}")
        else:
            for c in codes[:30]:
                print(f"  {c}")
            if len(codes) > 30:
                print(f"  ... 等共 {len(codes)} 只")
        print()

        res = task2_three_year_survival(df, args.threshold, args.cutoff_year)
        print("=" * 60)
        time_desc = f"{args.from_year} 年及以后、" if args.from_year else ""
        print(f"任务 2：{time_desc}2023 年以前期末净资产达到 2 亿的 case，3 年后仍在 2 亿以上的占比")
        print("=" * 60)
        print(f"有效 case 数: {res['total']}")
        print(f"3 年后仍在 2 亿以上: {res['still_above']}")
        print(f"占比: {res['pct']:.2f}%")

    # 任务 3：2015～2023年、成立>=3年、达2亿的 case，1年/3年后占比
    if args.overview:
        ov_path = Path(args.overview)
        if not ov_path.exists():
            ov_path = csv_path.parent / "fund_overview.csv"
        if not ov_path.exists():
            print(f"\n[警告] fund_overview.csv 不存在: {args.overview}", file=sys.stderr)
        else:
            inception = load_inception_dates(ov_path)
            res3 = task3_survival_with_inception(
                df, inception, args.threshold, 2015, 2023, min_age_years=3.0
            )
            print()
            print("=" * 60)
            print("任务 3：2015～2023年、成立时间>=3年、期末净资产达2亿的 case")
            print("       1年后 / 3年后 仍在2亿以上的占比")
            print("=" * 60)
            print(f"有效 case 数: {res3['total']}")
            print(f"1年后仍在2亿以上: {res3['still_1y']} (有效追踪 {res3['valid_1y']} 个), 占比: {res3['pct_1y']:.2f}%")
            print(f"3年后仍在2亿以上: {res3['still_3y']} (有效追踪 {res3['valid_3y']} 个), 占比: {res3['pct_3y']:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
