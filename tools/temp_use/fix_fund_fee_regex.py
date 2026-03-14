#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 fund_fee_complete.csv 中因正则模板缺失导致的错误记录。

因「大于1年，小于等于2年」「大于2年」等表述未被解析，部分赎回费率行持仓期限阶梯被错误置空，
导致业务主键重复或信息丢失。本脚本：
1. 识别 赎回费率 + 持仓期限阶梯全空 的基金（开放申购+开放赎回）
2. 重抓这些基金的赎回费率（使用已补全正则的 fetch_fund_fee）
3. 仅替换其 赎回费率 行，其余记录不变
4. 输出新文件 fund_fee_complete_fixed.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from fetch_fund_fee import (
    DEFAULT_REQUEST_DELAY,
    _empty_fee_row,
    _fetch_redemption_fee,
    _process_redemption_fee,
)


def _is_empty(s: Any) -> bool:
    v = pd.Series([s]).fillna("").astype(str).str.strip().iloc[0]
    return v == "" or v.lower() == "nan"


def _find_redemption_period_empty_codes(df: pd.DataFrame) -> set[str]:
    """找出赎回费率中存在持仓期限全空行的基金（开放申购+开放赎回）。"""
    cols = list(df.columns)
    if "基金编码" not in cols or "数据类型" not in cols:
        return set()

    codes: set[str] = set()
    for _, row in df.itertuples():
        code = str(row.基金编码 if hasattr(row, "基金编码") else row.get("基金编码", "")).strip()
        if not code:
            continue
        purchase_s = str(row.申购状态 if hasattr(row, "申购状态") else row.get("申购状态", "")).strip()
        redemption_s = str(row.赎回状态 if hasattr(row, "赎回状态") else row.get("赎回状态", "")).strip()
        if purchase_s != "开放申购" or redemption_s != "开放赎回":
            continue
        dtype = str(row.数据类型 if hasattr(row, "数据类型") else row.get("数据类型", "")).strip()
        if dtype != "赎回费率":
            continue
        fee = row.费率 if hasattr(row, "费率") else row.get("费率", "")
        if _is_empty(fee):
            continue
        period_start = row.持仓期限阶梯起点 if hasattr(row, "持仓期限阶梯起点") else row.get("持仓期限阶梯起点", "")
        period_end = row.持仓期限阶梯终点 if hasattr(row, "持仓期限阶梯终点") else row.get("持仓期限阶梯终点", "")
        if _is_empty(period_start) and _is_empty(period_end):
            codes.add(code)
    return codes


def _find_redemption_period_empty_codes_safe(df: pd.DataFrame) -> set[str]:
    """找出赎回费率中存在持仓期限全空行的基金（开放申购+开放赎回）。"""
    cols = list(df.columns)
    if "基金编码" not in cols or "数据类型" not in cols:
        return set()

    codes: set[str] = set()
    for _, row in df.iterrows():
        code = str(row.get("基金编码", "")).strip()
        if not code:
            continue
        purchase_s = str(row.get("申购状态", "")).strip()
        redemption_s = str(row.get("赎回状态", "")).strip()
        if purchase_s != "开放申购" or redemption_s != "开放赎回":
            continue
        dtype = str(row.get("数据类型", "")).strip()
        if dtype != "赎回费率":
            continue
        fee = row.get("费率", "")
        if _is_empty(fee):
            continue
        period_start = row.get("持仓期限阶梯起点", "")
        period_end = row.get("持仓期限阶梯终点", "")
        if _is_empty(period_start) and _is_empty(period_end):
            codes.add(code)
    return codes


def run(
    fee_csv: Path,
    output_csv: Path,
    logger: logging.Logger,
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> None:
    """主流程。"""
    df = pd.read_csv(fee_csv, dtype=str, encoding="utf-8-sig")
    cols_order = list(df.columns)

    fix_codes = _find_redemption_period_empty_codes_safe(df)
    if not fix_codes:
        logger.info("未发现需修复的基金（赎回费率+持仓期限全空），原样拷贝")
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return

    logger.info("待修复基金 %d 只（将重抓赎回费率）: %s", len(fix_codes), sorted(fix_codes)[:15])
    if len(fix_codes) > 15:
        logger.info("  ... 等共 %d 只", len(fix_codes))

    # 状态映射
    status_by_code: dict[str, tuple[str, str]] = {}
    for _, row in df.iterrows():
        code = str(row.get("基金编码", "")).strip()
        if code and code not in status_by_code:
            status_by_code[code] = (
                str(row.get("申购状态", "")).strip(),
                str(row.get("赎回状态", "")).strip(),
            )

    # 仅重抓赎回费率
    new_redemption_rows: list[dict[str, Any]] = []
    for i, code in enumerate(sorted(fix_codes)):
        if (i + 1) % 20 == 0 or i == 0:
            logger.info("重抓进度 %d/%d", i + 1, len(fix_codes))
        purchase_status, redemption_status = status_by_code.get(code, ("", ""))
        redemption_df = _fetch_redemption_fee(code, logger)
        if request_delay > 0:
            time.sleep(request_delay)
        rows = _process_redemption_fee(
            code, redemption_df, logger,
            purchase_status=purchase_status, redemption_status=redemption_status,
        ) if redemption_df is not None and not redemption_df.empty else []
        if not rows:
            new_redemption_rows.append(_empty_fee_row(code, "赎回费率", purchase_status, redemption_status))
        else:
            new_redemption_rows.extend(rows)

    # 合并：移除待修复基金的 赎回费率 行，插入新行；保留其余所有行
    mask_rest = ~(
        (df["基金编码"].isin(fix_codes)) & (df["数据类型"] == "赎回费率")
    )
    df_rest = df[mask_rest]
    new_df = pd.DataFrame(new_redemption_rows)
    if not new_df.empty:
        merged = pd.concat([df_rest, new_df[cols_order]], ignore_index=True)
    else:
        merged = df_rest

    merged = merged.sort_values(
        ["基金编码", "数据类型"],
        key=lambda c: c.map({"申购费率": 0, "赎回费率": 1}),
    )
    merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info("已写入 %s，共 %d 行（原 %d 行）", output_csv, len(merged), len(df))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="修复 fund_fee_complete.csv 中因正则缺失导致的赎回费率持仓期限错误"
    )
    parser.add_argument(
        "fee_csv",
        type=Path,
        help="fund_fee_complete.csv 路径",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="输出路径，默认为同目录 fund_fee_complete_fixed.csv",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        metavar="SEC",
        help="API 请求间隔秒数（默认 0.3）",
    )
    args = parser.parse_args()

    fee_csv = args.purchase_csv.resolve() if hasattr(args, "purchase_csv") else args.fee_csv.resolve()
    fee_csv = getattr(args, "fee_csv", args.fee_csv).resolve()
    if not fee_csv.exists():
        logger.error("输入文件不存在: %s", fee_csv)
        return 1

    out = args.output.resolve() if args.output else fee_csv.parent / "fund_fee_complete_fixed.csv"
    if args.output:
        out = args.output.resolve()
    else:
        out = fee_csv.parent / "fund_fee_complete_fixed.csv"

    run(fee_csv, out, logger, request_delay=args.delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
