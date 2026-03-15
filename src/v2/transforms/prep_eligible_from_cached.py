#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 预备数据 eligible 计算（仅使用本地缓存，不联网）。

规则（全局硬过滤，不按 T）：
- c.1 必须存在
- b: 全历史曾规模 >2 亿
- e: 成立日期可解析，且距今 > 3 年
- a: 不应用

输入 work_dir 期望包含（至少）：
- fund_purchase.csv
- fund_gmbd.csv
- fund_fee_structured.csv（或已存在 fund_fee_filtered.csv）
- fund_overview.csv

命令样例：
python myanalyser/src/v2/transforms/prep_eligible_from_cached.py \
  --work-dir myanalyser/tmp/prep_work_v2

输出：
- prep_result.csv（默认写入 work_dir）
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import re
import sys
from pathlib import Path

import pandas as pd

_TRANSFORMS_DIR = Path(__file__).resolve().parent
_MYANALYSER = _TRANSFORMS_DIR.parent.parent.parent
assert _MYANALYSER.name == "myanalyser", f"unexpected _MYANALYSER: {_MYANALYSER}"

_PREP_TOOLS = _MYANALYSER / "tools" / "prep"
if str(_MYANALYSER / "src") not in sys.path:
    sys.path.insert(0, str(_MYANALYSER / "src"))
if str(_PREP_TOOLS) not in sys.path:
    sys.path.insert(0, str(_PREP_TOOLS))


def _safe_code(v: object) -> str:
    """基金代码归一化：仅接受 6 位数字（或可转换为 6 位数字）。非法返回空字符串。"""
    if v is None:
        return ""
    if isinstance(v, int):
        return f"{v:06d}"
    if isinstance(v, float) and v.is_integer():
        return f"{int(v):06d}"
    s = str(v).strip()
    if not s or s == "---":
        return ""
    if not s.isdigit():
        return ""
    if len(s) > 6:
        return ""
    return s.zfill(6)


def _parse_date(text: object) -> pd.Timestamp | None:
    """解析日期，支持 2013年03月20日、YYYY-MM-DD 等格式。"""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    s = str(text).strip()
    if not s or s == "---":
        return None
    zh = re.search(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", s)
    if zh:
        y, m, d = zh.groups()
        return pd.to_datetime(f"{y}-{int(m):02d}-{int(d):02d}", errors="coerce")
    num = re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", s)
    if num:
        return pd.to_datetime(num.group(0), errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _get_today() -> pd.Timestamp:
    """返回本地今日日期（归一到 00:00），便于测试时 monkeypatch。"""
    return pd.Timestamp.today().normalize()


def _ensure_fee_filtered(
    fee_structured_csv: Path,
    fee_filtered_csv: Path,
    logger: logging.Logger,
) -> Path:
    if fee_filtered_csv.exists():
        return fee_filtered_csv
    if not fee_structured_csv.exists():
        raise FileNotFoundError(f"missing fee structured csv: {fee_structured_csv}")

    script_path = _PREP_TOOLS / "filter_fund_fee_by_holding.py"
    if not script_path.exists():
        raise FileNotFoundError(f"missing fee filter script: {script_path}")
    spec = importlib.util.spec_from_file_location("fee_filter", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load fee filter module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.info("[c.1] 生成费率分类: %s -> %s", fee_structured_csv, fee_filtered_csv)
    module.run(fee_structured_csv, fee_filtered_csv)
    if not fee_filtered_csv.exists():
        raise RuntimeError("failed to generate fund_fee_filtered.csv")
    return fee_filtered_csv


def _apply_hard_filters(
    purchase_df: pd.DataFrame,
    fee_filtered_csv: Path,
    gmbd_csv: Path,
    overview_csv: Path,
    logger: logging.Logger,
    *,
    today: pd.Timestamp,
) -> pd.DataFrame:
    codes = {c for c in purchase_df["基金代码"].dropna().map(_safe_code).tolist() if c}
    logger.info("[eligible] 原始候选 %d 只", len(codes))

    # c.1: 必须在 fee 分类结果中存在
    c1_df = pd.read_csv(fee_filtered_csv, dtype=str, encoding="utf-8-sig")
    code_col = "基金编码" if "基金编码" in c1_df.columns else "基金代码"
    c1_codes = {c for c in c1_df[code_col].dropna().map(_safe_code).tolist() if c}
    codes &= c1_codes
    logger.info("[eligible] c.1 后 %d", len(codes))

    # b: 全历史曾规模 > 2 亿
    b_df = pd.read_csv(gmbd_csv, dtype=str, encoding="utf-8-sig")
    scale_col = "期末净资产（亿元）"
    if scale_col not in b_df.columns:
        raise ValueError(f"missing column in gmbd csv: {scale_col}")
    b_df["_scale"] = pd.to_numeric(b_df[scale_col], errors="coerce")
    include_b = {
        c
        for c in b_df.loc[b_df["_scale"] > 2, "基金代码"]
        .dropna()
        .map(_safe_code)
        .unique()
        .tolist()
        if c
    }
    codes &= include_b
    logger.info("[eligible] b(曾规模>2亿) 后 %d", len(codes))

    # e: 成立日期可解析，且距今 > 3 年
    e_df = pd.read_csv(overview_csv, dtype=str, encoding="utf-8-sig")
    col = "成立日期/规模" if "成立日期/规模" in e_df.columns else "成立日期"
    if col not in e_df.columns:
        raise ValueError(f"missing column in overview csv: {col}")
    e_df = e_df.copy()
    e_df["_code"] = e_df["基金代码"].map(_safe_code)
    e_df["_inc"] = e_df[col].map(_parse_date)
    inc_ok = e_df[e_df["_inc"].notna() & (e_df["_code"] != "")]
    inc_ok = inc_ok.sort_values("_inc", na_position="last")
    inc_dedup = inc_ok.drop_duplicates("_code", keep="first").set_index("_code")
    cutoff = today - pd.DateOffset(years=3)
    include_e = {c for c in inc_dedup.index[inc_dedup["_inc"] < cutoff] if c}
    codes &= include_e
    logger.info("[eligible] e(成立距今>3年) 后 %d", len(codes))

    result = purchase_df[purchase_df["基金代码"].map(_safe_code).isin(codes)].copy()
    logger.info("[eligible] 最终结果 %d 只", len(result))
    return result


def run(
    work_dir: Path,
    *,
    output_path: Path | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    log = logger or logging.getLogger(__name__)
    work_dir = work_dir.resolve()

    purchase_csv = work_dir / "fund_purchase.csv"
    fee_structured_csv = work_dir / "fund_fee_structured.csv"
    fee_filtered_csv = work_dir / "fund_fee_filtered.csv"
    gmbd_csv = work_dir / "fund_gmbd.csv"
    overview_csv = work_dir / "fund_overview.csv"

    if not purchase_csv.exists():
        raise FileNotFoundError(f"missing purchase csv: {purchase_csv}")
    if not gmbd_csv.exists():
        raise FileNotFoundError(f"missing gmbd csv: {gmbd_csv}")
    if not overview_csv.exists():
        raise FileNotFoundError(f"missing overview csv: {overview_csv}")

    _ensure_fee_filtered(fee_structured_csv, fee_filtered_csv, log)

    purchase_df = pd.read_csv(purchase_csv, dtype=str, encoding="utf-8-sig")
    today = _get_today()
    result = _apply_hard_filters(
        purchase_df,
        fee_filtered_csv,
        gmbd_csv,
        overview_csv,
        log,
        today=today,
    )

    output_path = (output_path or (work_dir / "prep_result.csv")).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info("[eligible] 写入 %s", output_path)
    return output_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="v2 预备数据 eligible 计算（仅使用本地缓存）")
    parser.add_argument("--work-dir", type=Path, required=True, help="缓存工作目录（含 fund_*.csv）")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出 CSV 路径，默认 work_dir/prep_result.csv")
    args = parser.parse_args()

    try:
        run(
            work_dir=args.work_dir,
            output_path=args.output,
            logger=logger,
        )
        return 0
    except Exception as e:
        logger.exception("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
