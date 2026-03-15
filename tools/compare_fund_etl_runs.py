#!/usr/bin/env python3
"""
比较两次 fund_etl 运行爬取的原始数据一致性。
比较范围：fund_cum_return_by_code、fund_nav_by_code、fund_personnel_by_code
仅比较两次运行有交集的 fund code 部分。
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

# 项目根目录（workspace，finance-runs 在其下）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_A = PROJECT_ROOT / "finance-runs" / "run_20260315_prep_result_run" / "data" / "versions" / "20260315_prep_result_run" / "fund_etl"
RUN_B = PROJECT_ROOT / "finance-runs" / "run_20260310_191534" / "data" / "versions" / "20260310_191534" / "fund_etl"

DATASETS = ["fund_cum_return_by_code", "fund_nav_by_code", "fund_personnel_by_code"]


def get_fund_codes(dir_path: Path) -> set[str]:
    """获取目录下所有 fund code（不含 .csv 后缀）"""
    if not dir_path.exists():
        return set()
    return {f.stem for f in dir_path.glob("*.csv")}


def file_hash(path: Path) -> str:
    """计算文件 SHA256 哈希"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_csv_content(path_a: Path, path_b: Path) -> tuple[bool, str]:
    """
    比较两个 CSV 文件内容是否一致（考虑 pandas 读入后可能存在的浮点/格式差异）。
    返回 (是否一致, 差异描述)。
    """
    try:
        df_a = pd.read_csv(path_a)
        df_b = pd.read_csv(path_b)
    except Exception as e:
        return False, f"读取出错: {e}"

    # 比较形状
    if df_a.shape != df_b.shape:
        return False, f"shape 不同: {df_a.shape} vs {df_b.shape}"

    # 比较列名
    if list(df_a.columns) != list(df_b.columns):
        return False, f"列名不同: {list(df_a.columns)} vs {list(df_b.columns)}"

    # 比较数据（数值列容差，字符串精确）
    for col in df_a.columns:
        a_vals = df_a[col]
        b_vals = df_b[col]

        if pd.api.types.is_numeric_dtype(a_vals) and pd.api.types.is_numeric_dtype(b_vals):
            if not pd.Series(a_vals).equals(pd.Series(b_vals)):
                # 尝试容差比较
                diff_mask = pd.Series(a_vals).fillna(-999) != pd.Series(b_vals).fillna(-999)
                if pd.api.types.is_float_dtype(a_vals):
                    diff_mask = (pd.Series(a_vals) - pd.Series(b_vals)).abs() > 1e-9
                if diff_mask.any():
                    n_diff = diff_mask.sum()
                    return False, f"列 '{col}' 有 {n_diff} 行数值不同"
        else:
            if not a_vals.equals(b_vals):
                diff_count = (a_vals.fillna("") != b_vals.fillna("")).sum()
                return False, f"列 '{col}' 有 {diff_count} 行不同"

    return True, "一致"


def compare_overlap_time_series(path_a: Path, path_b: Path, date_col_a: str, date_col_b: str) -> tuple[bool, str]:
    """
    比较时序数据在「历史重叠区间」是否一致：取双方共有日期，对比数值。
    用于 fund_cum_return_by_code、fund_nav_by_code。
    """
    try:
        df_a = pd.read_csv(path_a)
        df_b = pd.read_csv(path_b)
    except Exception as e:
        return False, f"读取出错: {e}"

    if date_col_a not in df_a.columns or date_col_b not in df_b.columns:
        return False, f"缺少日期列: {date_col_a} / {date_col_b}"

    dates_a = set(df_a[date_col_a].astype(str).unique())
    dates_b = set(df_b[date_col_b].astype(str).unique())
    common_dates = dates_a & dates_b

    if not common_dates:
        return False, "无共同日期"

    df_a_sub = df_a[df_a[date_col_a].astype(str).isin(common_dates)].copy()
    df_b_sub = df_b[df_b[date_col_b].astype(str).isin(common_dates)].copy()

    # 按日期排序列出，便于比较
    df_a_sub = df_a_sub.sort_values(date_col_a).reset_index(drop=True)
    df_b_sub = df_b_sub.sort_values(date_col_b).reset_index(drop=True)

    if len(df_a_sub) != len(df_b_sub):
        return False, f"共同日期行数不同: {len(df_a_sub)} vs {len(df_b_sub)}"

    # 比较各列（排除日期列本身，用数值列和键列）
    value_cols = [c for c in df_a_sub.columns if c != date_col_a and c in df_b_sub.columns]
    for col in value_cols:
        if col not in df_b_sub.columns:
            continue
        a_vals = df_a_sub[col]
        b_vals = df_b_sub[col]
        if pd.api.types.is_numeric_dtype(a_vals) and pd.api.types.is_numeric_dtype(b_vals):
            diff = (pd.Series(a_vals) - pd.Series(b_vals)).abs()
            if (diff > 1e-9).any():
                n_diff = (diff > 1e-9).sum()
                return False, f"重叠区间列 '{col}' 有 {n_diff} 行数值不同"
        else:
            if not a_vals.equals(b_vals):
                n_diff = (a_vals.fillna("") != b_vals.fillna("")).sum()
                return False, f"重叠区间列 '{col}' 有 {n_diff} 行不同"
    return True, f"重叠 {len(common_dates)} 天一致"


def main() -> int:
    report_lines: list[str] = []

    report_lines.append("# 两次 fund_etl 运行原始数据一致性对比报告")
    report_lines.append("")
    report_lines.append("## 1. 运行路径")
    report_lines.append(f"- **Run A (20260315)**: `{RUN_A}`")
    report_lines.append(f"- **Run B (20260310)**: `{RUN_B}`")
    report_lines.append("")

    if not RUN_A.exists():
        report_lines.append(f"错误: Run A 目录不存在: {RUN_A}")
        print("\n".join(report_lines))
        return 1
    if not RUN_B.exists():
        report_lines.append(f"错误: Run B 目录不存在: {RUN_B}")
        print("\n".join(report_lines))
        return 1

    report_lines.append("## 2. 各数据集文件数量")
    report_lines.append("")
    report_lines.append("| 数据集 | Run A (20260315) | Run B (20260310) | 交集数量 |")
    report_lines.append("|--------|------------------|------------------|----------|")

    intersection_by_dataset: dict[str, set[str]] = {}

    for ds in DATASETS:
        dir_a = RUN_A / ds
        dir_b = RUN_B / ds
        codes_a = get_fund_codes(dir_a)
        codes_b = get_fund_codes(dir_b)
        inter = codes_a & codes_b
        intersection_by_dataset[ds] = inter
        report_lines.append(f"| {ds} | {len(codes_a)} | {len(codes_b)} | {len(inter)} |")

    report_lines.append("")
    report_lines.append("## 3. 交集数据一致性检查")
    report_lines.append("")
    report_lines.append("对交集中的每个 fund code，逐文件比较内容。")
    report_lines.append("")

    all_consistent = True
    for ds in DATASETS:
        report_lines.append(f"### {ds}")
        dir_a = RUN_A / ds
        dir_b = RUN_B / ds
        inter = intersection_by_dataset[ds]

        # 策略：先按文件哈希快速筛选，哈希相同则一致；哈希不同再按内容比较
        same_hash = 0
        diff_hash_same_content = 0
        diff_content = []
        missing_a = []
        missing_b = []

        for code in sorted(inter):
            f_a = dir_a / f"{code}.csv"
            f_b = dir_b / f"{code}.csv"
            if not f_a.exists():
                missing_a.append(code)
                continue
            if not f_b.exists():
                missing_b.append(code)
                continue

            h_a = file_hash(f_a)
            h_b = file_hash(f_b)
            if h_a == h_b:
                same_hash += 1
                continue

            # 哈希不同，进一步比较内容（可能只是格式/空格差异）
            ok, msg = compare_csv_content(f_a, f_b)
            if ok:
                diff_hash_same_content += 1
            else:
                diff_content.append((code, msg))
                all_consistent = False

        total = len(inter)
        report_lines.append(f"- 交集数量: {total}")
        report_lines.append(f"- 文件哈希完全一致: {same_hash}")
        report_lines.append(f"- 哈希不同但内容一致（格式差异）: {diff_hash_same_content}")
        if missing_a:
            report_lines.append(f"- Run A 缺失（交集内）: {len(missing_a)} 个")
        if missing_b:
            report_lines.append(f"- Run B 缺失（交集内）: {len(missing_b)} 个")
        if diff_content:
            report_lines.append(f"- **内容不一致: {len(diff_content)} 个**")
            for code, msg in diff_content[:20]:  # 最多列 20 个
                report_lines.append(f"  - `{code}`: {msg}")
            if len(diff_content) > 20:
                report_lines.append(f"  - ... 另有 {len(diff_content) - 20} 个")
        else:
            report_lines.append("- **所有交集文件内容一致** ✓")
        report_lines.append("")

    report_lines.append("## 3.5 历史重叠区间一致性检查")
    report_lines.append("")
    report_lines.append("Run A (20260315) 比 Run B (20260310) 晚 5 天，全量比较会因行数差异报不一致。")
    report_lines.append("本段仅比较「双方共有日期」内的数据，验证爬虫对历史区间的可复现性。")
    report_lines.append("")

    overlap_ok_cum = 0
    overlap_ok_nav = 0
    overlap_diff_cum = []
    overlap_diff_nav = []

    for code in sorted(intersection_by_dataset["fund_cum_return_by_code"]):
        f_a = RUN_A / "fund_cum_return_by_code" / f"{code}.csv"
        f_b = RUN_B / "fund_cum_return_by_code" / f"{code}.csv"
        if f_a.exists() and f_b.exists():
            ok, msg = compare_overlap_time_series(f_a, f_b, "日期", "日期")
            if ok:
                overlap_ok_cum += 1
            else:
                overlap_diff_cum.append((code, msg))

    for code in sorted(intersection_by_dataset["fund_nav_by_code"]):
        f_a = RUN_A / "fund_nav_by_code" / f"{code}.csv"
        f_b = RUN_B / "fund_nav_by_code" / f"{code}.csv"
        if f_a.exists() and f_b.exists():
            ok, msg = compare_overlap_time_series(f_a, f_b, "净值日期", "净值日期")
            if ok:
                overlap_ok_nav += 1
            else:
                overlap_diff_nav.append((code, msg))

    report_lines.append("- **fund_cum_return_by_code**（重叠日期内）: 一致 {}/{}，不一致 {}".format(
        overlap_ok_cum, len(intersection_by_dataset["fund_cum_return_by_code"]), len(overlap_diff_cum)))
    if overlap_diff_cum:
        for code, msg in overlap_diff_cum[:10]:
            report_lines.append(f"  - `{code}`: {msg}")
        if len(overlap_diff_cum) > 10:
            report_lines.append(f"  - ... 另有 {len(overlap_diff_cum) - 10} 个")
    else:
        report_lines.append("  - ✓ 所有交集 fund 在重叠区间内数据一致")
    report_lines.append("")
    report_lines.append("- **fund_nav_by_code**（重叠日期内）: 一致 {}/{}，不一致 {}".format(
        overlap_ok_nav, len(intersection_by_dataset["fund_nav_by_code"]), len(overlap_diff_nav)))
    if overlap_diff_nav:
        for code, msg in overlap_diff_nav[:10]:
            report_lines.append(f"  - `{code}`: {msg}")
        if len(overlap_diff_nav) > 10:
            report_lines.append(f"  - ... 另有 {len(overlap_diff_nav) - 10} 个")
    else:
        report_lines.append("  - ✓ 所有交集 fund 在重叠区间内数据一致")
    report_lines.append("")

    report_lines.append("## 4. 结论")
    report_lines.append("")
    if all_consistent:
        report_lines.append("**两次运行在交集范围内的原始数据完全一致。**")
    else:
        report_lines.append("**全量内容不一致**：Run A (20260315) 比 Run B (20260310) 晚 5 天，包含更多交易日数据，导致行数/日期差异。")
        report_lines.append("")
        if not overlap_diff_cum and not overlap_diff_nav:
            report_lines.append("**历史重叠区间一致**：在双方共有的历史日期范围内，fund_cum_return 与 fund_nav 数据完全一致，爬虫对历史数据具有可复现性。")
        else:
            report_lines.append("**历史重叠区间存在差异**：部分 fund 在共有日期内的数据也不同，需进一步排查爬虫或数据源变动。")
    report_lines.append("")

    report_text = "\n".join(report_lines)
    print(report_text)

    # 写入报告文件
    out_path = PROJECT_ROOT / "finance-runs" / "run_20260315_prep_result_run" / "docs" / "其他"
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "20260315_两次fund_etl运行原始数据一致性对比报告.md"
    report_file.write_text(report_text, encoding="utf-8")
    print(f"\n报告已保存: {report_file}")

    return 0 if all_consistent else 1


if __name__ == "__main__":
    sys.exit(main())
