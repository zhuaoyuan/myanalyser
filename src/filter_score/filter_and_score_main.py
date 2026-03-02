#!/usr/bin/env python3
"""筛选与打分入口脚本。

接收输入 CSV、0~多个过滤脚本、1 个算分脚本、工作目录，执行：
  1. 依次运行过滤策略，生成中间产物（含过滤结果）
  2. 剔除被过滤行后，运行算分策略，生成最终结果

模板方法：过滤与算分策略可扩展，通过加载外部 Python 脚本实现。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .base import load_filter_strategy, load_score_strategy

# 确保项目根在 path（支持从 myanalyser 或 finance 运行）
_PROJ_ROOT = Path(__file__).resolve().parents[2]  # src
_WS_ROOT = _PROJ_ROOT.parent  # myanalyser 或 finance 上层
for _p in (_PROJ_ROOT.parent, _WS_ROOT):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _ensure_work_dir(work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)


def _apply_filters(
    df: pd.DataFrame, filter_strategies: list, progress_interval: int = 500
) -> pd.DataFrame:
    """依次应用过滤策略，为每行添加各策略的 是否被过滤、过滤细节原因。"""
    out = df.copy()
    all_filtered = pd.Series(False, index=df.index)
    all_reasons: list[str] = [""] * len(df)

    for i, strategy in enumerate(filter_strategies):
        name = strategy.STRATEGY_NAME
        col_filtered = f"{name}_是否被过滤"
        col_reason = f"{name}_过滤细节原因"
        filtered_flags = []
        reasons = []
        total = len(df)
        for idx, (_, row) in enumerate(df.iterrows()):
            if progress_interval and (idx + 1) % progress_interval == 0:
                print(f"  过滤进度 [{name}]: {idx + 1}/{total}")
            is_f, reason = strategy.filter_one(row.to_dict())
            filtered_flags.append(is_f)
            reasons.append(reason)

        out[col_filtered] = filtered_flags
        out[col_reason] = reasons
        all_filtered = all_filtered | out[col_filtered]
        for j, r in enumerate(reasons):
            if r:
                all_reasons[j] = (all_reasons[j] + "; " + r).strip("; ")

    out["是否被过滤"] = all_filtered
    out["过滤细节原因"] = all_reasons
    return out


def _run_score(
    df: pd.DataFrame, score_strategy, progress_interval: int = 500
) -> pd.DataFrame:
    """运行算分策略，添加 计算策略名称。"""
    total = len(df)
    if progress_interval and total > 0:
        print(f"  算分进度: 0/{total}")
    result = score_strategy.compute_score(df)
    result["计算策略名称"] = score_strategy.STRATEGY_NAME
    if progress_interval and total > 0:
        print(f"  算分进度: {total}/{total}")
    return result


def run_pipeline(
    input_csv: Path,
    work_dir: Path,
    filter_scripts: list[Path],
    score_script: Path,
    *,
    encoding: str = "utf-8-sig",
    progress_interval: int = 500,
) -> int:
    """主流程。返回 0 成功，非 0 失败。"""
    print(f"[1/5] 加载输入: {input_csv}")
    if not input_csv.exists():
        print(f"错误：输入文件不存在: {input_csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(input_csv, dtype={"基金代码": str}, encoding=encoding)
    total_in = len(df)
    print(f"  共 {total_in} 条记录")

    _ensure_work_dir(work_dir)

    # 加载策略
    print("[2/5] 加载过滤与算分策略")
    filter_strategies = []
    for fp in filter_scripts:
        s = load_filter_strategy(fp)
        filter_strategies.append(s)
        print(f"  过滤: {s.STRATEGY_NAME} <- {fp}")
    score_strategy = load_score_strategy(score_script)
    print(f"  算分: {score_strategy.STRATEGY_NAME} <- {score_script}")

    # 应用过滤
    print("[3/5] 应用过滤策略")
    df_filtered = _apply_filters(df, filter_strategies, progress_interval)
    filter_result_path = work_dir / "filter_result.csv"
    df_filtered.to_csv(filter_result_path, index=False, encoding=encoding)
    n_filtered_out = int(df_filtered["是否被过滤"].sum())
    print(f"  被过滤: {n_filtered_out} 条，通过: {total_in - n_filtered_out} 条")
    print(f"  中间产物: {filter_result_path}")

    # 剔除被过滤行，去掉过滤过程列，保留原始输入列
    cols_to_drop = ["是否被过滤", "过滤细节原因"]
    for s in filter_strategies:
        cols_to_drop.append(f"{s.STRATEGY_NAME}_是否被过滤")
        cols_to_drop.append(f"{s.STRATEGY_NAME}_过滤细节原因")
    df_pass = df_filtered[~df_filtered["是否被过滤"]].drop(
        columns=cols_to_drop, errors="ignore"
    )

    print("[4/5] 计算得分")
    df_scored = _run_score(df_pass, score_strategy, progress_interval)
    scored_result_path = work_dir / "scored_result.csv"
    df_scored.to_csv(scored_result_path, index=False, encoding=encoding)
    print(f"  最终结果: {scored_result_path}，共 {len(df_scored)} 条")

    print("[5/5] 完成")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="筛选与打分：对基金榜单 CSV 应用可扩展的过滤与算分策略。"
    )
    parser.add_argument(
        "--input-csv",
        "-i",
        type=Path,
        required=True,
        help="输入 CSV 路径（形如 fund_scoreboard_*.csv）",
    )
    parser.add_argument(
        "--work-dir",
        "-w",
        type=Path,
        required=True,
        help="工作目录，中间产物与最终结果输出路径",
    )
    parser.add_argument(
        "--filter-script",
        "-f",
        type=Path,
        action="append",
        default=[],
        dest="filter_scripts",
        help="过滤脚本路径，可多次指定",
    )
    parser.add_argument(
        "--score-script",
        "-s",
        type=Path,
        required=True,
        help="算分脚本路径",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV 编码（默认 utf-8-sig）",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=500,
        help="进度打印间隔行数，0 禁用（默认 500）",
    )
    args = parser.parse_args()

    return run_pipeline(
        input_csv=args.input_csv,
        work_dir=args.work_dir,
        filter_scripts=args.filter_scripts or [],
        score_script=args.score_script,
        encoding=args.encoding,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    sys.exit(main())
