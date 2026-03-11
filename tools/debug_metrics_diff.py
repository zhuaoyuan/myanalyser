#!/usr/bin/env python3
"""深入分析 3 个指标差异的根因（用于文档，不纳入主流程）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ws = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ws))
sys.path.insert(0, str(_ws / "myanalyser" / "src"))

from myanalyser.src.scoreboard_metrics import load_nav_df

# 取 000003 做示例
NAV_DIR = _ws / "myanalyser/tests/baseline/mini_case/input/fund_etl/fund_adjusted_nav_by_code"
AS_OF = pd.Timestamp("2026-02-27")


def main():
    df = load_nav_df(NAV_DIR / "000003.csv")
    df = df[df["净值日期"] <= AS_OF].reset_index(drop=True)
    if df.empty or len(df) < 2:
        print("数据不足")
        return

    print("=" * 60)
    print("1. 近3年最长回撤修复天数 - 算法定义本质不同")
    print("=" * 60)
    print("""
Backtest (_longest_recovery_days):
  - 当价格收复前高时：days = dates[恢复日] - dates[峰顶日]
  - 含义：从【峰顶】到【收复峰顶】的总时长 = 整段「水下」时长（含下跌+回升）
  - 即：跌破前高后的「全程时长」

Scoreboard (_max_drawdown_recovery_days):
  - 对每个峰→谷→收复段：days = dates[收复日] - dates[谷底日]
  - 含义：从【谷底】到【收复峰顶】的时长 = 仅「回升阶段」时长
  - 即：行业常见的「回撤修复天数」

举例（示意）：
  峰顶 T0 ──下跌──> 谷底 T1 ──回升──> 收复 T2
  Backtest  = T2 - T0 = 全程 735 天
  Scoreboard = T2 - T1 = 回升 141 天

二者量纲不同，不是简单的交易日/自然日换算。
""")

    # 用实际数据演示「最近一个月」的时间段
    print("=" * 60)
    print("2. 最近一个月涨跌幅 - 完全不同的时间区间")
    print("=" * 60)
    last_date = df["净值日期"].iloc[-1]
    # Backtest: 最近 21 个交易日
    last_21 = df.tail(21)
    bt_start = last_21["净值日期"].iloc[0]
    bt_end = last_21["净值日期"].iloc[-1]
    bt_ret = (last_21["复权净值"].iloc[-1] / last_21["复权净值"].iloc[0] - 1) * 100

    # Scoreboard: 上个完整自然月
    curr_month_start = pd.Timestamp(last_date.year, last_date.month, 1)
    recent_month_end = curr_month_start - pd.Timedelta(days=1)
    recent_month_start = pd.Timestamp(recent_month_end.year, recent_month_end.month, 1)
    sb_df = df[(df["净值日期"] >= recent_month_start) & (df["净值日期"] <= recent_month_end)]
    sb_ret = (sb_df["复权净值"].iloc[-1] / sb_df["复权净值"].iloc[0] - 1) * 100 if len(sb_df) >= 2 else None

    print(f"截止日期: {last_date.date()}")
    print(f"\nBacktest: 最近 21 个交易日")
    print(f"  区间: {bt_start.date()} ~ {bt_end.date()}")
    print(f"  涨跌幅: {bt_ret:.2f}%")
    print(f"\nScoreboard: 上个完整自然月")
    print(f"  区间: {recent_month_start.date()} ~ {recent_month_end.date()}")
    print(f"  涨跌幅: {sb_ret:.2f}%" if sb_ret is not None else "  无数据")
    print("""
两段区间可能几乎不重叠！例如截止 2026-02-27 时：
- Backtest 21 日 ≈ 2026-01-xx ~ 2026-02-27（含 2 月）
- Scoreboard 上月 = 2026-01-01 ~ 2026-01-31（纯 1 月）
若 1 月涨、2 月跌，则一个为正、一个为负，差异可超过 10%。
""")

    # 年化收益率公式对比
    print("=" * 60)
    print("3. 年化收益率 - 指数基准不同导致数值差异")
    print("=" * 60)
    win = df.tail(252)  # 近 1 年
    if len(win) >= 2:
        start_val = float(win["复权净值"].iloc[0])
        end_val = float(win["复权净值"].iloc[-1])
        date_start = win["净值日期"].iloc[0]
        date_end = win["净值日期"].iloc[-1]
        cal_days = (date_end - date_start).days
        n_obs = len(win) - 1

        # Backtest: years = (n_obs) / 252
        years_bt = n_obs / 252
        cagr_bt = (end_val / start_val) ** (1 / years_bt) - 1 if years_bt > 0 else None

        # Scoreboard: 365/days 为指数
        cagr_sb = (end_val / start_val) ** (365.0 / cal_days) - 1 if cal_days > 0 else None

        print(f"近 252 条净值:")
        print(f"  首日: {date_start.date()}, 末日: {date_end.date()}")
        print(f"  自然日跨度: {cal_days} 天")
        print(f"  观测数(用于收益): n = {n_obs}")
        print()
        print("Backtest 公式:")
        print(f"  years = n / 252 = {n_obs}/252 = {years_bt:.4f}")
        print(f"  CAGR = (end/start)^(1/years) - 1 = {cagr_bt*100:.2f}%")
        print()
        print("Scoreboard 公式:")
        print(f"  years = cal_days/365 = {cal_days}/365 = {cal_days/365:.4f}")
        print(f"  CAGR = (end/start)^(365/days) - 1 = {cagr_sb*100:.2f}%")
        print()
        print("差异原因: 同一段收益")
        print("  - Backtest 用 252 天/年，相当于 n 个交易日 ≈ n/252 年")
        print("  - Scoreboard 用实际自然日，相当于 cal_days 天 ≈ cal_days/365 年")
        print("  当 cal_days ≠ n 时（有周末/节假日），两种 years 不同，CAGR 即不同。")
        print("  若 n=252 而 cal_days≈365，则 1/years_bt=1, 365/cal_days≈1，接近；")
        print("  若数据稀疏或窗口不同，差异可达数个百分点。")
    print()


if __name__ == "__main__":
    main()
