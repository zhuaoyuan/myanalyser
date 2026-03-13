#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临时脚本：从 fund_fee_complete.csv 筛选符合条件的基金并分类。

条件：
1. 忽略 申购状态!=开放申购 或 赎回状态!=开放赎回 的数据
2. A类：赎回费率在持仓期限 [x,∞) 时为 0 的，按 x 归类：
   x<=30 → A类30天, 30<x<=60 → A类60天, 60<x<=180 → A类180天, 180<x<=365 → A类365天, 365<x<=730 → A类730天
   取最小金额阶梯的申购费率，赎回费率=0
3. C类：金额阶梯为空白或0时申购费率=0，且赎回费率=0的持仓期限 [x,∞)，按 x 归类：
   x<=30 → C类30天, 30<x<=60 → C类60天
   申购费率=0，赎回费率=0
4. 未命中上述的不计入结果

输出：类型, 基金编码, 申购费率, 赎回费率（每基金一行）
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _empty(s: Any) -> bool:
    v = "" if pd.isna(s) else str(s).strip()
    return v == "" or v.lower() == "nan"


def _parse_fee_pct(val: Any) -> float | None:
    """解析费率百分比，返回数值或 None。"""
    s = "" if pd.isna(val) else str(val).strip()
    if not s:
        return None
    m = re.match(r"([\d.]+)\s*%", s)
    if m:
        return float(m[1])
    return None


def _to_float(val: Any) -> float | None:
    s = "" if pd.isna(val) else str(val).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _classify_a_by_x(x: float) -> str | None:
    """按赎回费率为0的持仓期限起点 x 归类 A 类。"""
    if x <= 30:
        return "A类30天"
    if x <= 60:
        return "A类60天"
    if x <= 180:
        return "A类180天"
    if x <= 365:
        return "A类365天"
    if x <= 730:
        return "A类730天"
    return None  # x>730 不计入


def _classify_c_by_x(x: float) -> str | None:
    """按赎回费率为0的持仓期限起点 x 归类 C 类。"""
    if x <= 30:
        return "C类30天"
    if x <= 60:
        return "C类60天"
    return None


def run(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv, dtype=str, encoding="utf-8-sig")
    # 仅保留开放申购+开放赎回
    df = df[
        (df["申购状态"].fillna("").str.strip() == "开放申购")
        & (df["赎回状态"].fillna("").str.strip() == "开放赎回")
    ]

    # 按基金分组
    codes = df["基金编码"].drop_duplicates().tolist()
    results: list[dict[str, Any]] = []

    for code in codes:
        sub = df[df["基金编码"] == code]
        purchase_rows = sub[sub["数据类型"] == "申购费率"]
        redemption_rows = sub[sub["数据类型"] == "赎回费率"]

        # 找赎回费率为0且持仓期限终点为空的记录（即 [x, ∞) 档）
        redemption_zero_x: float | None = None
        for _, row in redemption_rows.iterrows():
            fee_pct = _parse_fee_pct(row.get("费率"))
            if fee_pct is not None and fee_pct == 0:
                start = _to_float(row.get("持仓期限阶梯起点"))
                end = row.get("持仓期限阶梯终点")
                if start is not None and _empty(end):  # [x, ∞)
                    redemption_zero_x = start
                    break

        if redemption_zero_x is None:
            continue  # 无赎回费率为0的 [x,∞) 档，跳过

        # 检查是否有金额阶梯为空白或0时申购费率=0
        purchase_zero_at_min_tier = False
        min_purchase_fee_str: str | None = None

        purchase_with_amt: list[tuple[float, str]] = []
        for _, row in purchase_rows.iterrows():
            amt_start = row.get("金额阶梯起点")
            fee_raw = str(row.get("费率", "")).strip()
            fee_pct = _parse_fee_pct(row.get("费率"))
            if _empty(amt_start) or (_to_float(amt_start) is not None and _to_float(amt_start) == 0):
                if fee_pct is not None and fee_pct == 0:
                    purchase_zero_at_min_tier = True
            amt_val = 0.0 if _empty(amt_start) else (_to_float(amt_start) or 0.0)
            if fee_raw:  # 包含 0%、百分比、每笔X元 等
                purchase_with_amt.append((amt_val, fee_raw))

        # 取最小金额阶梯的申购费率
        if purchase_with_amt:
            purchase_with_amt.sort(key=lambda t: t[0])
            min_purchase_fee_str = purchase_with_amt[0][1]

        # C 类：金额阶梯空白/0 时申购费率=0，且赎回0档 x 在 30 或 60 内
        if purchase_zero_at_min_tier:
            c_type = _classify_c_by_x(redemption_zero_x)
            if c_type:
                results.append({
                    "类型": c_type,
                    "基金编码": code,
                    "申购费率": "0%",
                    "赎回费率": "0%",
                })
                continue

        # A 类：赎回费率为0的 x 在有效范围内
        a_type = _classify_a_by_x(redemption_zero_x)
        if a_type and min_purchase_fee_str:
            results.append({
                "类型": a_type,
                "基金编码": code,
                "申购费率": min_purchase_fee_str,
                "赎回费率": "0%",
            })

    out_df = pd.DataFrame(results)
    if not out_df.empty:
        out_df = out_df[["类型", "基金编码", "申购费率", "赎回费率"]]
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"已写入 {output_csv}，共 {len(out_df)} 行")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从 fund_fee_complete.csv 筛选并分类基金费率"
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="fund_fee_complete.csv 路径",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="输出 CSV 路径，默认与输入同目录的 fund_fee_filtered.csv",
    )
    args = parser.parse_args()

    input_csv = args.input_csv.resolve()
    if not input_csv.exists():
        print(f"输入文件不存在: {input_csv}", file=sys.stderr)
        return 1

    out = args.output
    if out is None:
        out = input_csv.parent / "fund_fee_filtered.csv"
    else:
        out = out.resolve()

    run(input_csv, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
