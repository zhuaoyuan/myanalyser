#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 fund_purchase.csv 读取基金编码，调用 akshare fund_fee_em 获取申购费率和赎回费率，
输出结构化 CSV，并记录无费率数据的基金到异常日志。
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# 需在项目虚拟环境中运行
try:
    import akshare as ak
except ImportError as e:
    raise RuntimeError("需要安装 akshare: pip install akshare") from e

# 默认请求间隔（秒），用于降低 akshare 限流风险
DEFAULT_REQUEST_DELAY = 0.3


def _normalize_code(code: str) -> str:
    raw = str(code).strip()
    if raw.isdigit():
        return raw.zfill(6)
    return raw


def _load_fund_records(purchase_csv: Path) -> list[dict[str, str]]:
    """从 fund_purchase.csv 读取基金记录（去重保序），每项含 基金编码、申购状态、赎回状态。"""
    df = pd.read_csv(purchase_csv, dtype=str, encoding="utf-8-sig")
    if "基金代码" not in df.columns:
        raise ValueError(f"缺少 基金代码 列: {purchase_csv}")
    has_purchase = "申购状态" in df.columns
    has_redemption = "赎回状态" in df.columns

    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        code = _normalize_code(row.get("基金代码", ""))
        if not code or code in seen:
            continue
        seen.add(code)
        purchase_s = str(row.get("申购状态", "")).strip() if has_purchase else ""
        redemption_s = str(row.get("赎回状态", "")).strip() if has_redemption else ""
        records.append({"基金编码": code, "申购状态": purchase_s, "赎回状态": redemption_s})
    return records


def _is_open_for_trade(rec: dict[str, str]) -> bool:
    """申购状态==开放申购 且 赎回状态==开放赎回 时才查询费率。"""
    return rec.get("申购状态") == "开放申购" and rec.get("赎回状态") == "开放赎回"


# ---- 自然语言解析 ----
# 金额单位统一为万元，期限单位统一为天

_AMOUNT_PATTERNS = [
    # 大于等于X万元，小于Y万元
    (re.compile(r"大于等于\s*([\d.]+)\s*万元\s*[，,]\s*小于\s*([\d.]+)\s*万元"), lambda m: (float(m[1]), float(m[2]))),
    # 小于X万元
    (re.compile(r"小于\s*([\d.]+)\s*万元"), lambda m: (0.0, float(m[1]))),
    # 大于等于X万元（无上限）
    (re.compile(r"大于等于\s*([\d.]+)\s*万元"), lambda m: (float(m[1]), None)),
]


def _parse_amount_tier(text: str | Any) -> tuple[float | None, float | None]:
    """解析适用金额，返回 (起点万元, 终点万元)，None 表示无该边界。"""
    s = str(text).strip() if pd.notna(text) else ""
    if not s or s == "---" or s == "—":
        return (None, None)
    for pat, extract in _AMOUNT_PATTERNS:
        m = pat.search(s)
        if m:
            return extract(m)
    return (None, None)


_PERIOD_PATTERNS = [
    # 大于等于X天，小于Y天
    (re.compile(r"大于等于\s*([\d.]+)\s*天\s*[，,]\s*小于\s*([\d.]+)\s*天"), lambda m: (float(m[1]), float(m[2]))),
    # 大于等于X天，小于Y年
    (re.compile(r"大于等于\s*([\d.]+)\s*天\s*[，,]\s*小于\s*([\d.]+)\s*年"), lambda m: (float(m[1]), float(m[2]) * 365)),
    # 大于等于X年，小于Y年
    (re.compile(r"大于等于\s*([\d.]+)\s*年\s*[，,]\s*小于\s*([\d.]+)\s*年"), lambda m: (float(m[1]) * 365, float(m[2]) * 365)),
    # 大于等于X天
    (re.compile(r"大于等于\s*([\d.]+)\s*天"), lambda m: (float(m[1]), None)),
    # 小于X天
    (re.compile(r"小于\s*([\d.]+)\s*天"), lambda m: (0.0, float(m[1]))),
    # 大于等于X年
    (re.compile(r"大于等于\s*([\d.]+)\s*年"), lambda m: (float(m[1]) * 365, None)),
    # 小于X年
    (re.compile(r"小于\s*([\d.]+)\s*年"), lambda m: (0.0, float(m[1]) * 365)),
]


def _parse_period_tier(text: str | Any) -> tuple[float | None, float | None]:
    """解析适用期限，返回 (起点天, 终点天)，None 表示无该边界。"""
    s = str(text).strip() if pd.notna(text) else ""
    if not s or s == "---" or s == "—":
        return (None, None)
    for pat, extract in _PERIOD_PATTERNS:
        m = pat.search(s)
        if m:
            return extract(m)
    return (None, None)


def _parse_fee_value(val: Any) -> tuple[str, float | None]:
    """解析费率。返回 (费率字符串, 数值型费率或 None)。数值型用于取最小值。"""
    s = str(val).strip() if pd.notna(val) else ""
    if not s:
        return ("", None)
    # 每笔固定金额
    fixed = re.match(r"每笔\s*([\d.]+)\s*元", s)
    if fixed:
        return (s, None)
    # 百分比 X%
    pct = re.match(r"([\d.]+)\s*%", s)
    if pct:
        n = float(pct[1])
        return (s, n)
    return (s, None)


def _choose_min_fee(row: pd.Series, fee_columns: list[str]) -> str:
    """同一阶梯下，从多列费率中取最小值（优惠费率如活期宝购买等）。"""
    candidates: list[tuple[str, float]] = []
    for col in fee_columns:
        if col not in row.index:
            continue
        raw, num = _parse_fee_value(row[col])
        if raw and num is not None:
            candidates.append((raw, num))
    if not candidates:
        for col in fee_columns:
            if col in row.index:
                raw, _ = _parse_fee_value(row[col])
                if raw:
                    return raw
        return ""
    return min(candidates, key=lambda x: x[1])[0]


def _fetch_purchase_fee(symbol: str, logger: logging.Logger) -> pd.DataFrame | None:
    """获取申购费率（前端）。"""
    try:
        df = ak.fund_fee_em(symbol=symbol, indicator="申购费率（前端）")
    except Exception as e:
        logger.warning("申购费率获取失败 symbol=%s: %s", symbol, e)
        return None
    if df is None or df.empty:
        return None
    return df


def _fetch_redemption_fee(symbol: str, logger: logging.Logger) -> pd.DataFrame | None:
    """获取赎回费率；若无数据则降级到「赎回费率（前端）」。"""
    for indicator in ("赎回费率", "赎回费率（前端）"):
        try:
            df = ak.fund_fee_em(symbol=symbol, indicator=indicator)
        except Exception as e:
            logger.debug("赎回费率 %s 获取失败 symbol=%s: %s", indicator, symbol, e)
            continue
        if df is not None and not df.empty:
            if indicator != "赎回费率":
                logger.info("symbol=%s 使用降级: %s", symbol, indicator)
            return df
    return None


def _empty_fee_row(
    code: str, data_type: str, purchase_status: str, redemption_status: str
) -> dict[str, Any]:
    """构造一条费率字段为空的记录（用于非开放申购/赎回的基金）。"""
    return {
        "基金编码": code,
        "申购状态": purchase_status,
        "赎回状态": redemption_status,
        "数据类型": data_type,
        "费率": "",
        "金额阶梯起点": "",
        "金额阶梯终点": "",
        "持仓期限阶梯起点": "",
        "持仓期限阶梯终点": "",
    }


def _process_purchase_fee(
    symbol: str,
    df: pd.DataFrame,
    logger: logging.Logger,
    purchase_status: str = "",
    redemption_status: str = "",
) -> list[dict[str, Any]]:
    """将申购费率 DataFrame 转为结构化记录。"""
    if df.empty or len(df.columns) == 0:
        return []
    rows: list[dict[str, Any]] = []
    has_std_cols = "适用金额" in df.columns and "适用期限" in df.columns
    amt_col = "适用金额" if has_std_cols else df.columns[0]
    period_col = "适用期限" if has_std_cols else (df.columns[1] if len(df.columns) > 1 else None)
    fee_cols = [c for c in df.columns if c not in (amt_col, period_col) and ("费率" in c or "费" in c)]
    if not fee_cols:
        fee_cols = [c for c in df.columns if c not in (amt_col, period_col)]

    for _, row in df.iterrows():
        amt_text = row.get(amt_col, "")
        period_text = row.get(period_col, "") if period_col else ""
        fee_str = _choose_min_fee(row, fee_cols)
        if not fee_str:
            continue
        amt_start, amt_end = _parse_amount_tier(amt_text)
        period_start, period_end = _parse_period_tier(period_text)
        rows.append({
            "基金编码": symbol,
            "申购状态": purchase_status,
            "赎回状态": redemption_status,
            "数据类型": "申购费率",
            "费率": fee_str,
            "金额阶梯起点": "" if amt_start is None else str(amt_start),
            "金额阶梯终点": "" if amt_end is None else str(amt_end),
            "持仓期限阶梯起点": "" if period_start is None else str(period_start),
            "持仓期限阶梯终点": "" if period_end is None else str(period_end),
        })
    return rows


def _process_redemption_fee(
    symbol: str,
    df: pd.DataFrame,
    logger: logging.Logger,
    purchase_status: str = "",
    redemption_status: str = "",
) -> list[dict[str, Any]]:
    """将赎回费率 DataFrame 转为结构化记录。"""
    if df.empty or len(df.columns) == 0:
        return []
    rows: list[dict[str, Any]] = []
    amt_col = "适用金额" if "适用金额" in df.columns else None
    period_col = "适用期限" if "适用期限" in df.columns else None
    fee_candidates = [c for c in df.columns if "费" in c]
    fee_col = "赎回费率" if "赎回费率" in df.columns else (
        fee_candidates[0] if fee_candidates else (df.columns[-1] if len(df.columns) > 0 else None)
    )
    if fee_col is None:
        return []

    for _, row in df.iterrows():
        amt_text = row.get(amt_col, "") if amt_col else ""
        period_text = row.get(period_col, "") if period_col else ""
        fee_str, _ = _parse_fee_value(row.get(fee_col, ""))
        if not fee_str:
            continue
        amt_start, amt_end = _parse_amount_tier(amt_text)
        period_start, period_end = _parse_period_tier(period_text)
        rows.append({
            "基金编码": symbol,
            "申购状态": purchase_status,
            "赎回状态": redemption_status,
            "数据类型": "赎回费率",
            "费率": fee_str,
            "金额阶梯起点": "" if amt_start is None else str(amt_start),
            "金额阶梯终点": "" if amt_end is None else str(amt_end),
            "持仓期限阶梯起点": "" if period_start is None else str(period_start),
            "持仓期限阶梯终点": "" if period_end is None else str(period_end),
        })
    return rows


def run(
    purchase_csv: Path,
    output_csv: Path,
    exception_log: Path,
    logger: logging.Logger,
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> None:
    """主流程。"""
    records = _load_fund_records(purchase_csv)
    logger.info("共 %d 只基金待处理", len(records))

    all_rows: list[dict[str, Any]] = []
    no_fee_codes: list[dict[str, Any]] = []

    for i, rec in enumerate(records):
        code = rec["基金编码"]
        purchase_status = rec.get("申购状态", "")
        redemption_status = rec.get("赎回状态", "")

        if (i + 1) % 50 == 0 or i == 0:
            logger.info("处理进度 %d/%d", i + 1, len(records))

        if not _is_open_for_trade(rec):
            all_rows.append(_empty_fee_row(code, "申购费率", purchase_status, redemption_status))
            all_rows.append(_empty_fee_row(code, "赎回费率", purchase_status, redemption_status))
            continue

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
            no_fee_codes.append({
                "基金编码": code,
                "申购费率": "无数据" if purchase_df is None or purchase_df.empty else "解析无结果",
                "赎回费率": "无数据" if redemption_df is None or redemption_df.empty else "解析无结果",
            })
            continue

        all_rows.extend(purchase_rows)
        all_rows.extend(redemption_rows)

    out_df = pd.DataFrame(all_rows)
    if not out_df.empty:
        cols = ["基金编码", "申购状态", "赎回状态", "数据类型", "费率", "金额阶梯起点", "金额阶梯终点", "持仓期限阶梯起点", "持仓期限阶梯终点"]
        out_df = out_df[cols]
        out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        logger.info("已写入 %s，共 %d 行", output_csv, len(out_df))
    else:
        logger.warning("无有效费率数据，未生成输出 CSV")

    if no_fee_codes:
        exc_df = pd.DataFrame(no_fee_codes)
        exc_df.to_csv(exception_log, index=False, encoding="utf-8-sig")
        logger.info("无费率数据基金已记录到 %s，共 %d 只", exception_log, len(no_fee_codes))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="从 fund_purchase.csv 获取基金申购/赎回费率并输出结构化 CSV")
    parser.add_argument(
        "purchase_csv",
        type=Path,
        help="fund_purchase.csv 路径，例: finance-runs/run_xxx/data/versions/xxx/fund_etl/fund_purchase.csv",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="输出 CSV 路径，默认与 purchase_csv 同目录下的 fund_fee_structured.csv",
    )
    parser.add_argument(
        "-e", "--exception-log",
        type=Path,
        default=None,
        help="无费率数据异常日志路径，默认与 output 同目录下的 fund_fee_exceptions.csv",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        metavar="SEC",
        help="每次 API 请求后休眠秒数，用于降低限流风险（默认 0.3）",
    )
    args = parser.parse_args()

    purchase_csv = args.purchase_csv.resolve()
    if not purchase_csv.exists():
        logger.error("输入文件不存在: %s", purchase_csv)
        return 1

    out = args.output
    if out is None:
        out = purchase_csv.parent / "fund_fee_structured.csv"
    else:
        out = out.resolve()

    exc = args.exception_log
    if exc is None:
        exc = out.parent / "fund_fee_exceptions.csv"
    else:
        exc = exc.resolve()

    run(purchase_csv, out, exc, logger, request_delay=args.delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
