#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预备数据工作流结果验证脚本。

对照原始 purchase 清单与结果文件，抽样验证：
1. 在结果中的 100 条：确认满足筛选条件（c.1 存在、a 无机构>60%、b 曾规模>2亿、e date前成立）
2. 不在结果中的 100 条：确认满足至少一条过滤条件（被 c.1/a/b/e 任一排除）
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pandas as pd

_MYANALYSER = Path(__file__).resolve().parent.parent.parent
if str(_MYANALYSER / "src") not in sys.path:
    sys.path.insert(0, str(_MYANALYSER / "src"))


def _safe_code(v: object) -> str:
    return str(v).strip().zfill(6)


def _parse_date(text: object) -> pd.Timestamp | None:
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


def _parse_pct(val: object) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s == "---":
        return None
    m = re.match(r"([\d.]+)\s*%", s)
    return float(m[1]) if m else None


def _check_code_conditions(
    code: str,
    date_ts: pd.Timestamp,
    c1_codes: set[str],
    exclude_a: set[str],
    include_b: set[str],
    include_e: set[str],
    cyrjg_exists: bool,
) -> dict:
    """检查单只基金是否满足各项条件。返回 {条件: 是否满足, ...}"""
    code_n = _safe_code(code)
    return {
        "c1": code_n in c1_codes,
        "a": cyrjg_exists and code_n not in exclude_a,
        "b": code_n in include_b,
        "e": code_n in include_e,
    }


def run(
    purchase_csv: Path,
    result_csv: Path,
    prep_work_dir: Path,
    date_str: str = "2021-01-01",
    sample_size: int = 100,
    seed: int = 42,
) -> None:
    date_ts = pd.to_datetime(date_str)
    purchase = pd.read_csv(purchase_csv, dtype=str)
    result = pd.read_csv(result_csv, dtype=str)

    in_result_codes = set(result["基金代码"].dropna().map(_safe_code).tolist())
    all_codes = set(purchase["基金代码"].dropna().map(_safe_code).tolist())
    not_in_result_codes = all_codes - in_result_codes

    print("=" * 60)
    print("预备数据工作流结果验证")
    print("=" * 60)
    print(f"起始日期: {date_str}")
    print(f"purchase 总数: {len(purchase)}, 去重基金数: {len(all_codes)}")
    print(f"result 总数: {len(result)}, 去重基金数: {len(in_result_codes)}")
    print(f"不在结果中: {len(not_in_result_codes)}")
    print()

    # 加载中间数据
    fee_filtered = prep_work_dir / "fund_fee_filtered.csv"
    gmbd = prep_work_dir / "fund_gmbd.csv"
    overview = prep_work_dir / "fund_overview.csv"
    cyrjg_paths = [
        prep_work_dir / "fund_cyrjg.csv",
        prep_work_dir.parent.parent / "finance-runs" / "run_20260310_191534"
        / "data" / "versions" / "20260310_191534" / "fund_etl" / "cyrjg_out.csv",
        _MYANALYSER.parent / "finance-runs" / "run_20260310_191534"
        / "data" / "versions" / "20260310_191534" / "fund_etl" / "cyrjg_out.csv",
    ]
    cyrjg_csv = next((p for p in cyrjg_paths if p.exists()), None)
    cyrjg_exists = cyrjg_csv and cyrjg_csv.exists()

    # c.1
    c1_codes: set[str] = set()
    if fee_filtered.exists():
        c1_df = pd.read_csv(fee_filtered, dtype=str)
        code_col = "基金编码" if "基金编码" in c1_df.columns else "基金代码"
        c1_codes = set(c1_df[code_col].dropna().map(_safe_code).tolist())
        print(f"[c.1] 费率分类基金数: {len(c1_codes)}")
    else:
        print("[c.1] fund_fee_filtered.csv 不存在，跳过")

    # a
    exclude_a: set[str] = set()
    if cyrjg_csv and cyrjg_csv.exists():
        a_df = pd.read_csv(cyrjg_csv, dtype=str)
        date_col = "日期" if "日期" in a_df.columns else "公告日期"
        a_df = a_df.copy()
        a_df[date_col] = pd.to_datetime(a_df[date_col], errors="coerce")
        a_df = a_df[a_df[date_col] >= date_ts]
        a_df["_pct"] = a_df["机构持有比例"].map(_parse_pct)
        exclude_rows = a_df[a_df["_pct"].notna() & (a_df["_pct"] > 60)]
        exclude_a = set(exclude_rows["基金代码"].dropna().map(_safe_code))
        print(f"[a] 机构>60% 排除基金数: {len(exclude_a)}")
    else:
        print("[a] cyrjg 不存在，跳过（视为全部通过）")

    # b
    include_b: set[str] = set()
    if gmbd.exists():
        b_df = pd.read_csv(gmbd, dtype=str)
        b_df["日期"] = pd.to_datetime(b_df["日期"], errors="coerce")
        b_df = b_df[b_df["日期"] >= date_ts]
        scale_col = "期末净资产（亿元）"
        if scale_col in b_df.columns:
            b_df["_scale"] = pd.to_numeric(b_df[scale_col], errors="coerce")
            include_b = set(b_df[b_df["_scale"] > 2]["基金代码"].dropna().map(_safe_code).tolist())
        print(f"[b] 规模>2亿 保留基金数: {len(include_b)}")
    else:
        print("[b] fund_gmbd.csv 不存在，跳过")

    # e
    include_e: set[str] = set()
    if overview.exists():
        e_df = pd.read_csv(overview, dtype=str)
        col = "成立日期/规模" if "成立日期/规模" in e_df.columns else "成立日期"
        if col in e_df.columns:
            e_df["_inc_dt"] = e_df[col].map(_parse_date)
            include_rows = e_df[e_df["_inc_dt"].notna() & (e_df["_inc_dt"] < date_ts)]
            include_e = set(include_rows["基金代码"].dropna().map(_safe_code))
        print(f"[e] date前成立 保留基金数: {len(include_e)}")
    else:
        print("[e] fund_overview.csv 不存在，跳过")

    print()

    def passes_all(conds: dict) -> bool:
        if not cyrjg_exists:
            return conds["c1"] and conds["b"] and conds["e"]
        return conds["c1"] and conds["a"] and conds["b"] and conds["e"]

    def excluded_by_any(conds: dict) -> bool:
        return not conds["c1"] or (cyrjg_exists and not conds["a"]) or not conds["b"] or not conds["e"]

    # 抽样在结果中的
    in_list = list(in_result_codes)
    random.seed(seed)
    n_in = min(sample_size, len(in_list))
    sample_in = random.sample(in_list, n_in)

    print("-" * 60)
    print(f"【在结果中的抽样 {n_in} 条】应全部满足：c.1 + a + b + e")
    print("-" * 60)
    fail_in = 0
    for i, code in enumerate(sample_in):
        conds = _check_code_conditions(
            code, date_ts, c1_codes, exclude_a, include_b, include_e, cyrjg_exists
        )
        ok = passes_all(conds)
        if not ok:
            fail_in += 1
            flags = []
            if not conds["c1"]:
                flags.append("不在c.1")
            if cyrjg_exists and not conds["a"]:
                flags.append("机构>60%")
            if not conds["b"]:
                flags.append("规模未>2亿")
            if not conds["e"]:
                flags.append("非date前成立")
            print(f"  [{i+1}] {code} 不通过: {', '.join(flags)}")
    if fail_in == 0:
        print(f"  全部 {n_in} 条通过 ✓")
    else:
        print(f"  失败 {fail_in}/{n_in} ✗")

    # 抽样不在结果中的
    not_in_list = list(not_in_result_codes)
    n_out = min(sample_size, len(not_in_list))
    sample_out = random.sample(not_in_list, n_out)

    print()
    print("-" * 60)
    print(f"【不在结果中的抽样 {n_out} 条】应全部满足：被 c.1/a/b/e 至少一项排除")
    print("-" * 60)
    fail_out = 0
    for i, code in enumerate(sample_out):
        conds = _check_code_conditions(
            code, date_ts, c1_codes, exclude_a, include_b, include_e, cyrjg_exists
        )
        ok = excluded_by_any(conds)
        if not ok:
            fail_out += 1
            print(f"  [{i+1}] {code} 异常：满足全部条件却不在结果中")
    if fail_out == 0:
        print(f"  全部 {n_out} 条均符合过滤逻辑 ✓")
    else:
        print(f"  异常 {fail_out}/{n_out} ✗")

    print()
    print("=" * 60)
    if fail_in == 0 and fail_out == 0:
        print("验证通过")
    else:
        print(f"验证未通过：在结果中失败 {fail_in}，不在结果中异常 {fail_out}")
    print("=" * 60)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="验证 prep_data_workflow 产物")
    ap.add_argument("--purchase", type=Path, default=None, help="purchase CSV")
    ap.add_argument("--result", type=Path, default=None, help="result CSV")
    ap.add_argument("--prep-work-dir", type=Path, default=None, help="prep_work 目录")
    ap.add_argument("--date", default="2021-01-01", help="筛选起始日期")
    ap.add_argument("--sample", type=int, default=100, help="每类抽样数量")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    base = _MYANALYSER / "tmp" / "1"
    purchase = args.purchase or base / "prep_work" / "fund_purchase.csv"
    result = args.result or base / "prep_result_m.csv"
    prep_work = args.prep_work_dir or base / "prep_work"

    if not purchase.exists():
        print(f"错误：purchase 不存在 {purchase}")
        return 1
    if not result.exists():
        print(f"错误：result 不存在 {result}")
        return 1
    if not prep_work.exists():
        print(f"错误：prep_work 不存在 {prep_work}")
        return 1

    run(
        purchase_csv=purchase,
        result_csv=result,
        prep_work_dir=prep_work,
        date_str=args.date,
        sample_size=args.sample,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
