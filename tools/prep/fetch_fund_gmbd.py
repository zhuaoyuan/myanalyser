#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基金规模变动数据抓取 CLI

从基金编码列表获取规模变动数据，输出 CSV。
输入：基金编码（命令行或 CSV 的 基金代码 列）
输出：包含 日期、期间申购（亿份）、期间赎回（亿份）、期末总份额（亿份）、期末净资产（亿元）、净资产变动率 的 CSV。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 项目路径：tools/prep 的 parents[2] 为 myanalyser
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
assert (_PROJECT_ROOT / "src").exists(), f"invalid project root: {_PROJECT_ROOT}"
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import pandas as pd

from fund_gmbd import _normalize_code, fund_gmbd_em


def _load_codes_from_csv(csv_path: Path, code_col: str = "基金代码") -> list[str]:
    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    if code_col not in df.columns:
        raise ValueError(f"CSV 缺少列 {code_col}: {csv_path}")
    codes = df[code_col].dropna().astype(str).str.strip()
    seen: set[str] = set()
    out: list[str] = []
    for c in codes:
        nc = _normalize_code(c)
        if nc and nc not in seen:
            seen.add(nc)
            out.append(nc)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="基金规模变动抓取，输出 CSV")
    parser.add_argument(
        "codes",
        nargs="*",
        default=[],
        help="基金编码列表，如 000198 110011",
    )
    parser.add_argument(
        "-i", "--input-csv",
        type=Path,
        help="从 CSV 读取基金代码列（列名默认 基金代码）",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="输出 CSV 路径，默认 stdout",
    )
    parser.add_argument(
        "--code-col",
        default="基金代码",
        help="CSV 中基金代码列名，默认 基金代码",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="请求间隔（秒），默认 0.3",
    )
    args = parser.parse_args()

    codes: list[str] = []
    if args.codes:
        codes = [_normalize_code(c) for c in args.codes if c]
        codes = list(dict.fromkeys(codes))
    if args.input_csv:
        if not args.input_csv.exists():
            print(f"错误：输入文件不存在 {args.input_csv}", file=sys.stderr)
            return 1
        codes = _load_codes_from_csv(args.input_csv, args.code_col)
    if not codes:
        parser.print_help()
        print("\n请指定基金编码或 -i CSV 文件", file=sys.stderr)
        return 1

    total = len(codes)
    step = max(1, min(100, total // 20))  # 每 100 条或约 5% 打印一次
    dfs: list[tuple[str, pd.DataFrame]] = []
    failed: list[str] = []
    for i, code in enumerate(codes):
        if (i + 1) % step == 0 or i == 0 or (i + 1) == total:
            pct = int(100 * (i + 1) / total)
            print(f"  [gmbd] 进度 {i + 1}/{total} ({pct}%)", file=sys.stderr)
        try:
            df = fund_gmbd_em(code)
            if not df.empty:
                df = df.copy()
                df.insert(0, "基金代码", code)
                dfs.append((code, df))
        except Exception as e:
            failed.append(code)
            print(f"  {code}: 失败 - {e}", file=sys.stderr)
        if i < len(codes) - 1 and args.delay > 0:
            time.sleep(args.delay)

    if failed:
        print(f"失败基金: {', '.join(failed)}", file=sys.stderr)

    if not dfs:
        print("未获取到任何数据", file=sys.stderr)
        # 全部返回空数据（非异常）：输出空 CSV，增量模式下下游可复用已有文件
        if args.output and not failed:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            empty_df = pd.DataFrame(
                columns=["基金代码", "日期", "期间申购（亿份）", "期间赎回（亿份）", "期末总份额（亿份）", "期末净资产（亿元）", "净资产变动率"]
            )
            empty_df.to_csv(args.output, index=False, encoding="utf-8-sig")
            print(f"已输出空 CSV -> {args.output}（{total} 只均无规模数据，可复用已有文件）", file=sys.stderr)
            return 0
        return 1

    combined = pd.concat([d[1] for d in dfs], ignore_index=True)
    out_csv = combined.to_csv(index=False, encoding="utf-8-sig")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_csv, encoding="utf-8-sig")
        print(f"已写入 {args.output}，共 {len(combined)} 行", file=sys.stderr)
    else:
        print(out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
