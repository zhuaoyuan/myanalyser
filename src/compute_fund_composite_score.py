#!/usr/bin/env python3
"""基金综合得分计算脚本。

接收形如 scoreboard/filtered 的 CSV 输入，对原始指标做线性归一化 + 分组 + 加权，
计算二级指标（风险控制、短期业绩、持有体验、长期业绩）及综合得分，输出新 CSV。

分组与权重：
  A. 风险控制 (35%): 近1年最大回撤40% + 近3年最长回撤修复天数40% + 近3年最大回撤20%
  B. 短期业绩 (30%): 近1年卡玛比率50% + 近1年年化收益30% + 最近一个月涨跌幅20%
  C. 持有体验 (20%): 近1年上涨星期比例30% + 近3年上涨月比例30% + 近1年周标准差40%
  D. 长期业绩 (15%): 近3年卡玛比率50% + 近3年年化收益30% + 近3年夏普比率20%
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# 指标定义: (CSV列名, 二级指标内权重, 方向: "asc"=越小越好需取反, "desc"=越大越好)
RISK_CONTROL_METRICS = [
    ("近1年最大回撤率", 0.4, "asc"),
    ("近3年最长回撤修复天数", 0.4, "asc"),
    ("近3年最大回撤率", 0.2, "asc"),
]
SHORT_TERM_METRICS = [
    ("近1年卡玛比率", 0.5, "desc"),
    ("近1年年化收益率", 0.3, "desc"),
    ("最近一个月涨跌幅", 0.2, "desc"),
]
HOLDING_EXPERIENCE_METRICS = [
    ("近1年上涨星期比例", 0.3, "desc"),
    ("近3年上涨月份比例", 0.3, "desc"),
    ("近1年周涨跌幅标准差", 0.4, "asc"),
]
LONG_TERM_METRICS = [
    ("近3年卡玛比率", 0.5, "desc"),
    ("近3年年化收益率", 0.3, "desc"),
    ("近3年夏普比率", 0.2, "desc"),
]

SECONDARY_GROUPS = [
    ("风险控制", 0.35, RISK_CONTROL_METRICS),
    ("短期业绩", 0.30, SHORT_TERM_METRICS),
    ("持有体验", 0.20, HOLDING_EXPERIENCE_METRICS),
    ("长期业绩", 0.15, LONG_TERM_METRICS),
]


def _linear_normalize(series: pd.Series, ascending: bool) -> pd.Series:
    """线性 min-max 归一化到 [0, 1]。ascending=True 表示原值越大越好；False 表示越小越好（取反）。"""
    valid = series.dropna()
    if valid.empty or valid.min() == valid.max():
        return pd.Series(index=series.index, dtype=float)
    r = (series - valid.min()) / (valid.max() - valid.min())
    if not ascending:
        r = 1.0 - r
    return r


def _compute_group_score(df: pd.DataFrame, metrics: list[tuple[str, float, str]]) -> pd.Series:
    """对一组指标加权求和得到二级得分。缺失值按 0.5 处理（中性，不偏向高/低）。"""
    total = pd.Series(0.0, index=df.index)
    weight_sum = 0.0
    for col, w, direction in metrics:
        if col not in df.columns:
            continue
        ascending = direction == "desc"
        norm = _linear_normalize(pd.to_numeric(df[col], errors="coerce"), ascending=ascending)
        norm = norm.fillna(0.5)
        total = total + w * norm
        weight_sum += w
    if weight_sum <= 0:
        return total
    return total / weight_sum


def compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """计算二级指标与综合得分，返回带新列的 DataFrame（不修改原 df）。"""
    out = df.copy()

    for group_name, group_weight, metrics in SECONDARY_GROUPS:
        col_score = f"得分_{group_name}"
        out[col_score] = _compute_group_score(out, metrics)

    # 综合得分
    composite = pd.Series(0.0, index=out.index)
    for group_name, group_weight, _ in SECONDARY_GROUPS:
        col = f"得分_{group_name}"
        if col in out.columns:
            composite = composite + group_weight * out[col].fillna(0)
    out["综合得分"] = composite

    # 排名（综合得分降序，越大越靠前）
    out["综合排名"] = out["综合得分"].rank(ascending=False, method="min").astype("Int64")

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="基金综合得分计算：对 filtered/scoreboard CSV 做归一化+分组加权，输出带得分的 CSV。"
    )
    parser.add_argument(
        "--input-csv",
        "-i",
        type=Path,
        required=True,
        help="输入 CSV 路径（含基金代码、基金名称及所需指标列）",
    )
    parser.add_argument(
        "--output-csv",
        "-o",
        type=Path,
        required=True,
        help="输出 CSV 路径（原列 + 得分_* + 综合得分 + 综合排名）",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="输入输出 CSV 编码（默认 utf-8-sig）",
    )
    args = parser.parse_args()

    if not args.input_csv.exists():
        print(f"错误：输入文件不存在: {args.input_csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.input_csv, dtype={"基金代码": str}, encoding=args.encoding)
    if df.empty:
        print("警告：输入为空，将输出空 CSV。", file=sys.stderr)

    result = compute_composite_score(df)
    result.to_csv(args.output_csv, index=False, encoding=args.encoding)
    print(f"已写入: {args.output_csv}，共 {len(result)} 行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
