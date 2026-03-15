#!/usr/bin/env python3
"""诊断：007540 在某调仓日通过 most_stable 过滤时的实际指标。

用于回答「为何 007540 的波动率/胜率没被 most_stable 过滤」：
- 回测在选入日使用「截至选入日」的净值动态计算指标，与 scoreboard 的「全样本/当前」指标不同。
- 本脚本复现指定日期的指标计算并输出 filter_one 判定结果。

用法：
  cd /Users/zhuaoyuan/cursor-workspace/finance
  source myanalyser/.venv312/bin/activate
  python myanalyser/tools/diagnose_most_stable_007540.py \\
    --nav-dir ../finance-runs/run_20260310_191534/data \\
    --as-of 2024-09-10 \\
    --symbol 007540
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保 myanalyser/src 在 path（与 pybroker_fund_backtest 一致）
_src = Path(__file__).resolve().parents[2] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from most_stable_logic import filter_one

# 从 backtest 导入数据加载和指标计算
from backtest.data import load_fund_nav_data
from backtest.filters.most_stable_strategy import _compute_most_stable_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="诊断 most_stable 指标与过滤结果")
    parser.add_argument("--nav-dir", type=Path, required=True, help="净值数据目录（含 fund_etl/fund_adjusted_nav_by_code）")
    parser.add_argument("--as-of", default="2024-09-10", help="调仓日 YYYY-MM-DD（007540 首次入选日）")
    parser.add_argument("--symbol", default="007540", help="基金代码")
    args = parser.parse_args()

    as_of = args.as_of
    symbol = str(args.symbol).strip().zfill(6)

    print(f"[diagnose] 加载数据: {args.nav_dir}")
    data = load_fund_nav_data(
        args.nav_dir,
        max_funds=5000,
        start_date="2020-01-01",
        end_date="2025-12-31",
        allowed_codes={symbol},
    )

    df = data.by_symbol.get(symbol)
    if df is None or df.empty:
        print(f"[diagnose] 未找到 {symbol} 的净值数据")
        sys.exit(1)

    as_of_ts = __import__("pandas").Timestamp(as_of)
    mask = df["date"] <= as_of_ts
    df_hist = df.loc[mask]
    if df_hist.empty:
        print(f"[diagnose] {symbol} 在 {as_of} 之前无数据")
        sys.exit(1)

    print(f"[diagnose] {symbol} 截至 {as_of} 共 {len(df_hist)} 条净值，范围 {df_hist['date'].min().date()} ~ {df_hist['date'].max().date()}")

    row = _compute_most_stable_metrics(df_hist, as_of_ts)
    if not row:
        print(f"[diagnose] 无法计算指标（可能不足 3 年窗口）")
        sys.exit(1)

    print("\n[diagnose] most_stable 所需指标（选入日当日计算）：")
    for k, v in row.items():
        print(f"  {k}: {v}")

    is_filtered, reason = filter_one(row)
    print(f"\n[diagnose] filter_one 判定: {'被过滤' if is_filtered else '通过'}")
    if reason:
        print(f"  原因: {reason}")

    # 与 scoreboard 的差异说明
    print("\n[说明] 回测使用「截至选入日」的净值计算指标，与 scoreboard 全样本/当前日指标不同。")
    print("        若当前 scoreboard 显示 007540 波动率高或胜率低，可能是后续行情导致指标变化。")


if __name__ == "__main__":
    main()
