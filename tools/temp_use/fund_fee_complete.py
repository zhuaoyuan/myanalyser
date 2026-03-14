#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拷贝 fund_fee_out.csv，识别疑似缺失的基金并重跑费率抓取，产出完整结果。

疑似缺失定义：
- 申购费率：费率非空 且 金额阶梯起点、金额阶梯终点 均空（排除非开放基金）
- 赎回费率：费率非空 且 持仓期限阶梯起点、持仓期限阶梯终点 均空（排除非开放基金）

仅对 开放申购+开放赎回 的基金重跑；非开放基金保持原空行。
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from fetch_fund_fee import (
    DEFAULT_REQUEST_DELAY,
    _empty_fee_row,
    _fetch_purchase_fee,
    _fetch_redemption_fee,
    _is_open_for_trade,
    _process_purchase_fee,
    _process_redemption_fee,
)


def _is_empty(s: Any) -> bool:
    v = pd.Series([s]).fillna("").astype(str).str.strip().iloc[0]
    return v == "" or v.lower() == "nan"


def _find_suspicious_missing(
    df: pd.DataFrame,
    redemption_only: bool = False,
) -> set[str]:
    """找出疑似阶梯缺失的基金编码集合（仅开放申购+开放赎回）。

    redemption_only: 若为 True，仅识别 赎回费率 持仓期限全空 的基金（更精准，API 调用更少）。
    """
    cols = list(df.columns)
    if "基金编码" not in cols or "数据类型" not in cols:
        return set()

    missing_codes: set[str] = set()
    for _, row in df.iterrows():
        code = str(row.get("基金编码", "")).strip()
        if not code:
            continue
        # 非开放基金不重跑
        purchase_s = str(row.get("申购状态", "")).strip()
        redemption_s = str(row.get("赎回状态", "")).strip()
        if purchase_s != "开放申购" or redemption_s != "开放赎回":
            continue

        dtype = str(row.get("数据类型", "")).strip()
        fee = row.get("费率", "")
        if _is_empty(fee):
            continue

        if dtype == "申购费率" and not redemption_only:
            amt_start = row.get("金额阶梯起点", "")
            amt_end = row.get("金额阶梯终点", "")
            if _is_empty(amt_start) and _is_empty(amt_end):
                missing_codes.add(code)
        elif dtype == "赎回费率":
            period_start = row.get("持仓期限阶梯起点", "")
            period_end = row.get("持仓期限阶梯终点", "")
            if _is_empty(period_start) and _is_empty(period_end):
                missing_codes.add(code)
    return missing_codes


def _rerun_funds(
    codes: list[str],
    status_by_code: dict[str, tuple[str, str]],
    logger: logging.Logger,
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> list[dict[str, Any]]:
    """对指定基金列表重跑费率抓取，返回新行。"""
    rows: list[dict[str, Any]] = []
    for i, code in enumerate(codes):
        if (i + 1) % 10 == 0 or i == 0:
            logger.info("重跑进度 %d/%d", i + 1, len(codes))
        purchase_status, redemption_status = status_by_code.get(code, ("", ""))

        purchase_df = _fetch_purchase_fee(code, logger)
        if request_delay > 0:
            time.sleep(request_delay)
        redemption_df = _fetch_redemption_fee(code, logger)
        if request_delay > 0:
            time.sleep(request_delay)

        purchase_rows = _process_purchase_fee(
            code, purchase_df, logger,
            purchase_status=purchase_status, redemption_status=redemption_status,
        ) if purchase_df is not None and not purchase_df.empty else []
        redemption_rows = _process_redemption_fee(
            code, redemption_df, logger,
            purchase_status=purchase_status, redemption_status=redemption_status,
        ) if redemption_df is not None and not redemption_df.empty else []

        if not purchase_rows and not redemption_rows:
            rows.append(_empty_fee_row(code, "申购费率", purchase_status, redemption_status))
            rows.append(_empty_fee_row(code, "赎回费率", purchase_status, redemption_status))
        else:
            rows.extend(purchase_rows)
            rows.extend(redemption_rows)
    return rows


def run(
    fee_csv: Path,
    output_csv: Path,
    logger: logging.Logger,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    backup: bool = True,
    redemption_only: bool = False,
) -> None:
    """主流程。"""
    df = pd.read_csv(fee_csv, dtype=str, encoding="utf-8-sig")
    cols_order = list(df.columns)

    missing_codes = _find_suspicious_missing(df, redemption_only=redemption_only)
    if not missing_codes:
        logger.info("未发现疑似缺失，原样拷贝")
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return

    logger.info("疑似缺失基金 %d 只: %s", len(missing_codes), sorted(missing_codes)[:20])
    if len(missing_codes) > 20:
        logger.info("  ... 等共 %d 只", len(missing_codes))

    # 备份
    if backup:
        backup_path = fee_csv.parent / f"{fee_csv.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{fee_csv.suffix}"
        shutil.copy2(fee_csv, backup_path)
        logger.info("已备份至 %s", backup_path)

    # 状态映射（从原 df 取）
    status_by_code: dict[str, tuple[str, str]] = {}
    for _, row in df.iterrows():
        code = str(row.get("基金编码", "")).strip()
        if code and code not in status_by_code:
            status_by_code[code] = (
                str(row.get("申购状态", "")).strip(),
                str(row.get("赎回状态", "")).strip(),
            )

    # 重跑
    new_rows = _rerun_funds(sorted(missing_codes), status_by_code, logger, request_delay)
    new_df = pd.DataFrame(new_rows)
    if new_df.empty:
        logger.warning("重跑无结果，保留原文件")
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return

    # 合并：移除缺失基金的行，插入新行
    df_rest = df[~df["基金编码"].isin(missing_codes)]
    merged = pd.concat([df_rest, new_df[cols_order]], ignore_index=True)
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
        description="拷贝 fund_fee_out.csv，对疑似缺失的基金重跑费率，产出完整结果"
    )
    parser.add_argument(
        "fee_csv",
        type=Path,
        help="fund_fee_out.csv 路径",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="输出 CSV 路径，默认覆盖原文件（建议先备份或指定 -o）",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不创建备份",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        metavar="SEC",
        help="API 请求间隔秒数（默认 0.3）",
    )
    parser.add_argument(
        "--redemption-only",
        action="store_true",
        help="仅对 赎回费率 持仓期限全空 的基金重跑（更精准，减少 API 调用）",
    )
    args = parser.parse_args()

    fee_csv = args.fee_csv.resolve()
    if not fee_csv.exists():
        logger.error("输入文件不存在: %s", fee_csv)
        return 1

    out = args.output
    if out is None:
        out = fee_csv
    else:
        out = out.resolve()

    run(
        fee_csv,
        out,
        logger,
        request_delay=args.delay,
        backup=not args.no_backup,
        redemption_only=args.redemption_only,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
