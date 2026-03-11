#!/usr/bin/env python3
"""对比 backtest/metrics + compute_composite_score 与 scoreboard_metrics 两套算分逻辑的差异。

用同一份净值数据，分别用两套逻辑计算指标与综合得分，输出逐项对比与排名差异。

用法：
  cd /Users/zhuaoyuan/cursor-workspace/finance
  source myanalyser/.venv312/bin/activate
  python myanalyser/tools/compare_metrics_logic.py \
    --nav-dir myanalyser/tests/baseline/mini_case/input/fund_etl/fund_adjusted_nav_by_code \
    --as-of-date 2026-02-27 \
    --max-funds 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ws = Path(__file__).resolve().parents[2]
if str(_ws) not in sys.path:
    sys.path.insert(0, str(_ws))
_src = _ws / "myanalyser" / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from myanalyser.src.backtest.metrics import compute_low_risk_debt_metrics
from myanalyser.src.compute_fund_composite_score import compute_composite_score
from myanalyser.src.scoreboard_metrics import compute_metrics, load_nav_df, window_metrics

# scoreboard_metrics 输出列名 -> compute_composite_score 所需中文列名
SCOREBOARD_TO_CN = {
    "max_drawdown_1y": "近1年最大回撤率",
    "max_drawdown_recovery_days_3y": "近3年最长回撤修复天数",
    "max_drawdown_3y": "近3年最大回撤率",
    "calmar_ratio_1y": "近1年卡玛比率",
    "annual_return_1y": "近1年年化收益率",
    "recent_month_return": "最近一个月涨跌幅",
    "up_week_ratio_1y": "近1年上涨星期比例",
    "up_month_ratio_3y": "近3年上涨月份比例",
    "week_return_std_1y": "近1年周涨跌幅标准差",
    "calmar_ratio_3y": "近3年卡玛比率",
    "annual_return_3y": "近3年年化收益率",
    "sharpe_ratio_3y": "近3年夏普比率",
}

COMPOSITE_METRICS = list(SCOREBOARD_TO_CN.values())


def _resolve_nav_dir(nav_dir: Path) -> Path:
    """解析净值目录（与 backtest.data 逻辑一致）。"""
    nav_dir = Path(nav_dir).resolve()
    if nav_dir.is_dir() and list(nav_dir.glob("*.csv")):
        return nav_dir
    for candidate in [nav_dir / "fund_etl" / "fund_adjusted_nav_by_code", nav_dir / "versions" / "latest" / "fund_etl" / "fund_adjusted_nav_by_code"]:
        if candidate.is_dir() and list(candidate.glob("*.csv")):
            return candidate
    return nav_dir


def _metrics_from_backtest(nav_dir: Path, as_of_date: pd.Timestamp, max_funds: int) -> pd.DataFrame:
    """用 backtest 路径：load_nav_df 加载 + compute_low_risk_debt_metrics（与 scoreboard 使用同一数据源）。"""
    nav_dir = _resolve_nav_dir(nav_dir)
    files = sorted(nav_dir.glob("*.csv"))[:max_funds]
    rows = []
    for p in files:
        df = load_nav_df(p)
        if df.empty:
            continue
        df = df[df["净值日期"] <= as_of_date].reset_index(drop=True)
        if len(df) < 2:
            continue
        code = str(p.stem).strip().zfill(6)
        dates = df["净值日期"].to_numpy(dtype="datetime64[D]")
        prices = df["复权净值"].to_numpy(dtype=float)
        m = compute_low_risk_debt_metrics(dates, prices)
        rows.append({"symbol": code, **m})
    return pd.DataFrame(rows)


def _metrics_from_scoreboard(nav_dir: Path, as_of_date: pd.Timestamp, max_funds: int) -> pd.DataFrame:
    """用 scoreboard 路径：load_nav_df + compute_metrics + window_metrics。"""
    nav_dir = _resolve_nav_dir(nav_dir)
    files = sorted(nav_dir.glob("*.csv"))[:max_funds]
    rows = []
    for p in files:
        df = load_nav_df(p)
        if df.empty:
            continue
        df = df[df["净值日期"] <= as_of_date].reset_index(drop=True)
        if len(df) < 2:
            continue
        code = str(p.stem).strip().zfill(6)
        end_date = df["净值日期"].iloc[-1]
        base = compute_metrics(df, end_date)
        m3 = window_metrics(df, end_date, years=3)
        m1 = window_metrics(df, end_date, years=1)
        row = {"symbol": code, **base, **m3, **m1}
        rows.append(row)
    return pd.DataFrame(rows)


def _scoreboard_to_composite_input(df: pd.DataFrame) -> pd.DataFrame:
    """将 scoreboard 指标列映射为 compute_composite_score 所需中文列名。"""
    out = df[["symbol"]].copy()
    for eng, cn in SCOREBOARD_TO_CN.items():
        if eng in df.columns:
            out[cn] = df[eng]
    return out


def _compare_one_metric(
    df_bt: pd.DataFrame,
    df_sb: pd.DataFrame,
    col: str,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> dict:
    """对比单指标：返回 match_count, diff_count, max_abs_diff, sample_diffs。"""
    if col not in df_bt.columns or col not in df_sb.columns:
        return {"status": "skip", "reason": "column missing"}
    merged = df_bt[["symbol", col]].merge(
        df_sb[["symbol", col]],
        on="symbol",
        how="inner",
        suffixes=("_bt", "_sb"),
    )
    if merged.empty:
        return {"status": "skip", "reason": "no common symbols"}
    a = merged[f"{col}_bt"]
    b = merged[f"{col}_sb"]
    both_nan = a.isna() & b.isna()
    either_nan = a.isna() | b.isna()
    num_ok = both_nan.sum()
    num_diff_nan = either_nan.sum() - both_nan.sum()
    numeric_mask = ~either_nan
    if numeric_mask.any():
        diff = (a - b).abs()
        max_diff = float(diff.max())
        close = ((a - b).abs() <= (atol + rtol * b.abs())).fillna(False)
        num_close = close.sum()
        num_far = numeric_mask.sum() - num_close
        sample = (
            merged[numeric_mask & ~close]
            .head(3)[["symbol", f"{col}_bt", f"{col}_sb"]]
            .rename(columns={f"{col}_bt": "backtest", f"{col}_sb": "scoreboard"})
        )
    else:
        max_diff = None
        num_close = 0
        num_far = 0
        sample = pd.DataFrame()
    return {
        "status": "pass" if num_far == 0 and num_diff_nan == 0 else "diff",
        "total": len(merged),
        "both_nan": int(num_ok),
        "diff_nan": int(num_diff_nan),
        "close_match": int(num_close),
        "far_diff": int(num_far),
        "max_abs_diff": max_diff,
        "sample_diffs": sample.to_dict("records") if not sample.empty else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="对比 backtest 与 scoreboard 两套算分逻辑")
    parser.add_argument("--nav-dir", type=Path, required=True, help="净值目录或 run data 目录")
    parser.add_argument("--as-of-date", default="2026-02-27", help="截止日期 YYYY-MM-DD")
    parser.add_argument("--max-funds", type=int, default=20, help="最多参与对比的基金数")
    parser.add_argument("--output", type=Path, default=None, help="输出报告 CSV（可选）")
    args = parser.parse_args()

    if not args.nav_dir.is_absolute():
        args.nav_dir = (_ws / args.nav_dir).resolve()
    if not args.nav_dir.exists():
        print(f"数据目录不存在: {args.nav_dir}", file=sys.stderr)
        return 1

    as_of = pd.Timestamp(args.as_of_date)
    print(f"[compare] nav_dir={args.nav_dir}")
    print(f"[compare] as_of_date={as_of.date()}, max_funds={args.max_funds}")

    # 1) 两套指标
    print("[compare] 计算 backtest 指标...")
    df_bt = _metrics_from_backtest(args.nav_dir, as_of, args.max_funds)
    print(f"[compare] backtest 基金数: {len(df_bt)}")
    if df_bt.empty:
        print("无有效数据，退出", file=sys.stderr)
        return 1

    print("[compare] 计算 scoreboard 指标...")
    df_sb_raw = _metrics_from_scoreboard(args.nav_dir, as_of, args.max_funds)
    df_sb = _scoreboard_to_composite_input(df_sb_raw)
    print(f"[compare] scoreboard 基金数: {len(df_sb)}")

    # 2) 逐指标对比
    common = set(df_bt["symbol"]) & set(df_sb["symbol"])
    print(f"[compare] 共同基金: {len(common)}")
    if not common:
        print("无共同基金，无法对比", file=sys.stderr)
        return 1

    metric_report = []
    for col in COMPOSITE_METRICS:
        r = _compare_one_metric(df_bt, df_sb, col)
        r["metric"] = col
        metric_report.append(r)
        status = r.get("status", "?")
        if status == "diff":
            print(f"  {col}: {status} (far_diff={r.get('far_diff')}, max_diff={r.get('max_abs_diff')})")
        else:
            print(f"  {col}: {status}")

    # 3) 综合得分与排名对比
    scored_bt = compute_composite_score(df_bt)
    scored_sb = compute_composite_score(df_sb)
    merged_score = scored_bt[["symbol", "综合得分", "综合排名"]].merge(
        scored_sb[["symbol", "综合得分", "综合排名"]],
        on="symbol",
        how="inner",
        suffixes=("_bt", "_sb"),
    )
    merged_score["rank_diff"] = merged_score["综合排名_bt"] - merged_score["综合排名_sb"]
    merged_score["score_diff"] = merged_score["综合得分_bt"] - merged_score["综合得分_sb"]
    rank_changes = (merged_score["rank_diff"] != 0).sum()
    max_rank_diff = merged_score["rank_diff"].abs().max()
    max_score_diff = merged_score["score_diff"].abs().max()
    print(f"\n[compare] 综合得分/排名:")
    print(f"  有排名变化的基金数: {int(rank_changes)} / {len(merged_score)}")
    print(f"  最大排名差: {int(max_rank_diff)}")
    print(f"  最大得分差: {max_score_diff:.6f}")

    # 4) 输出详细对比 CSV
    out_dir = args.output or (_ws / "myanalyser" / "output" / "metrics_compare")
    if hasattr(out_dir, "suffix") and out_dir.suffix == ".csv":
        out_dir = out_dir.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 指标并排对比
    detail = df_bt.merge(
        df_sb,
        on="symbol",
        how="outer",
        suffixes=("_bt", "_sb"),
        indicator=True,
    )
    detail.to_csv(out_dir / "metrics_detail.csv", index=False, encoding="utf-8-sig")
    merged_score.to_csv(out_dir / "score_rank_compare.csv", index=False, encoding="utf-8-sig")
    print(f"\n[compare] 明细已写入: {out_dir}")

    # 5) 汇总
    diff_metrics = [m["metric"] for m in metric_report if m.get("status") == "diff"]
    if diff_metrics:
        print(f"\n[compare] 结论: 存在差异的指标: {', '.join(diff_metrics)}")
        return 0
    print("\n[compare] 结论: 两套逻辑指标一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
