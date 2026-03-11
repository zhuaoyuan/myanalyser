#!/usr/bin/env python3
"""回测逻辑核验：用真实数据跑回测，并在关键节点抽样验证。

用法：
  cd /Users/zhuaoyuan/cursor-workspace/finance
  source myanalyser/.venv312/bin/activate
  python myanalyser/tools/verify_backtest_logic.py \
    --nav-dir finance-runs/run_20260310_191534/data \
    --max-funds 6 --start-date 2024-06-01 --end-date 2025-06-30
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ws = Path(__file__).resolve().parents[2]
if str(_ws) not in sys.path:
    sys.path.insert(0, str(_ws))
_src = _ws / "myanalyser" / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# 使用 myanalyser.src 风格导入，避免 CLI 的包结构问题
from myanalyser.src.backtest.data import BacktestData, load_fund_nav_data
from myanalyser.src.backtest.engine import BacktestConfig, run_backtest, write_reports
from myanalyser.src.backtest.metrics import compute_low_risk_debt_metrics
from myanalyser.src.backtest.strategies.low_risk_debt import build_bundle
from myanalyser.src.compute_fund_composite_score import compute_composite_score


def _manual_max_drawdown(prices: np.ndarray) -> float | None:
    """纯手工实现最大回撤，用于核验 metrics 模块。"""
    if len(prices) < 2:
        return None
    running_max = np.maximum.accumulate(prices.astype(float))
    drawdown = prices.astype(float) / running_max - 1.0
    return float(np.nanmin(drawdown))


def _manual_cagr(prices: np.ndarray, trading_days: int = 243) -> float | None:
    """纯手工实现年化收益率。"""
    if len(prices) < 2:
        return None
    start, end = float(prices[0]), float(prices[-1])
    if start <= 0 or end <= 0:
        return None
    years = (len(prices) - 1) / trading_days
    if years <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def verify_metrics_node(symbol: str, data: BacktestData, as_of_date: pd.Timestamp) -> dict:
    """在【指标计算】节点抽样核验：用纯 numpy 手工计算与 metrics 模块对比。"""
    df_sym = data.by_symbol.get(symbol)
    if df_sym is None or df_sym.empty:
        return {"status": "skip", "reason": "no data"}
    mask = df_sym["date"] <= as_of_date
    hist = df_sym.loc[mask]
    if hist.empty or len(hist) < 2:
        return {"status": "skip", "reason": "insufficient history"}
    dates = hist["date"].to_numpy(dtype="datetime64[D]")
    prices = hist["close"].to_numpy(dtype=float)
    # 调用当前代码
    metrics_out = compute_low_risk_debt_metrics(dates, prices)
    # 手工计算近1年最大回撤、近1年年化（与 fund_metrics_core 对齐：窗口不足 243 日则返回 None）
    win_1y = 243
    has_1y = len(prices) >= win_1y
    if has_1y:
        prices_1y = prices[-win_1y:]
        manual_max_dd = _manual_max_drawdown(prices_1y)
        manual_ann_ret = _manual_cagr(prices_1y)
    else:
        manual_max_dd = None
        manual_ann_ret = None
    code_max_dd = metrics_out.get("近1年最大回撤率")
    code_ann_ret = metrics_out.get("近1年年化收益率")
    dd_ok = (
        (manual_max_dd is None and code_max_dd is None)
        or (
            manual_max_dd is not None
            and code_max_dd is not None
            and abs(manual_max_dd - code_max_dd) < 1e-6
        )
    )
    ann_ok = (
        (manual_ann_ret is None and code_ann_ret is None)
        or (
            manual_ann_ret is not None
            and code_ann_ret is not None
            and abs(manual_ann_ret - code_ann_ret) < 1e-5
        )
    )
    return {
        "symbol": symbol,
        "as_of_date": str(as_of_date.date()),
        "manual_max_dd_1y": manual_max_dd,
        "code_max_dd_1y": code_max_dd,
        "dd_match": dd_ok,
        "manual_ann_ret_1y": manual_ann_ret,
        "code_ann_ret_1y": code_ann_ret,
        "ann_ret_match": ann_ok,
        "status": "pass" if (dd_ok and ann_ok) else "fail",
    }


def verify_composite_score_node(scored_df: pd.DataFrame, sample_idx: int = 0) -> dict:
    """在【综合得分】节点核验：得分降序排列、排名与得分一致。"""
    if scored_df.empty or sample_idx >= len(scored_df):
        return {"status": "skip", "reason": "empty or out of range"}
    row = scored_df.iloc[sample_idx]
    score = row.get("综合得分")
    rank = row.get("综合排名")
    # 核验：排名 1 应对应最高分
    if rank == 1:
        expected_max = scored_df["综合得分"].max()
        score_ok = pd.isna(score) or abs(float(score) - float(expected_max)) < 1e-9
    else:
        score_ok = True
    # 核验：排名应与得分降序一致
    sorted_by_score = scored_df.sort_values("综合得分", ascending=False).reset_index(drop=True)
    rank_consistency = all(
        sorted_by_score.iloc[i]["综合排名"] <= sorted_by_score.iloc[i + 1]["综合排名"]
        for i in range(len(sorted_by_score) - 1)
    )
    return {
        "sample_symbol": row.get("symbol"),
        "sample_score": float(score) if pd.notna(score) else None,
        "sample_rank": int(rank) if pd.notna(rank) else None,
        "rank_1_has_max_score": score_ok,
        "rank_consistency": rank_consistency,
        "status": "pass" if (score_ok and rank_consistency) else "fail",
    }


def verify_position_weights_node(weights: dict[str, float], top_n: int) -> dict:
    """在【仓位权重】节点核验：等权、和为 1。"""
    if not weights:
        return {"status": "pass", "reason": "empty weights ok"}
    total = sum(weights.values())
    n = len(weights)
    expected_each = 1.0 / n if n > 0 else 0.0
    all_equal = all(abs(w - expected_each) < 1e-9 for w in weights.values())
    sum_ok = abs(total - 1.0) < 1e-9
    return {
        "n_symbols": n,
        "total_weight": total,
        "expected_each": expected_each,
        "all_equal": all_equal,
        "sum_one": sum_ok,
        "status": "pass" if (all_equal and sum_ok) else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="回测逻辑核验")
    parser.add_argument("--nav-dir", type=Path, required=True)
    parser.add_argument("--max-funds", type=int, default=6)
    parser.add_argument("--start-date", default="2024-06-01")
    parser.add_argument("--end-date", default="2025-06-30")
    parser.add_argument("--rebalance", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.nav_dir.is_absolute():
        # 相对路径以 workspace (finance) 为基准
        args.nav_dir = (_ws / args.nav_dir).resolve()
    if not args.nav_dir.exists():
        print(f"数据目录不存在: {args.nav_dir}", file=sys.stderr)
        return 1

    print("[verify] 加载数据...")
    data = load_fund_nav_data(
        args.nav_dir,
        max_funds=args.max_funds,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    symbols = sorted(data.by_symbol.keys())
    print(f"[verify] 基金: {symbols}, 日期: {data.trading_dates[0].date()} ~ {data.trading_dates[-1].date()}")

    # 1) 指标计算节点抽样核验
    sample_date = data.trading_dates[len(data.trading_dates) // 2]
    metrics_checks = []
    for sym in symbols[:3]:
        r = verify_metrics_node(sym, data, sample_date)
        metrics_checks.append(r)
        print(f"[verify] 指标核验 {sym}: {r['status']} (dd_match={r.get('dd_match')}, ann_match={r.get('ann_ret_match')})")

    # 2) 跑完整回测
    bundle = build_bundle()
    print("[verify] 运行回测...")
    result = run_backtest(
        data,
        bundle,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        rebalance_period=args.rebalance,
        warmup=120,
        config=BacktestConfig(initial_cash=100_000),
    )

    # 3) 从 period_log 抽取首期，核验综合得分与仓位
    period_log = result.period_log
    score_check = weight_check = {"status": "skip"}
    if not period_log:
        print("[verify] 无调仓记录，跳过后续核验")
    else:
        first = period_log[0]
        stat_date = first["stat_date"]
        scores_top = first.get("scores_top", [])
        target_weights = first.get("target_weights", {})
        print(f"[verify] 首期调仓日: {stat_date}, 选基: {list(target_weights.keys())}")

        # 手工执行同一天策略，对比
        candidates = bundle.filter_strategy.filter_symbols(data, pd.Timestamp(stat_date), symbols)
        scored = bundle.score_strategy.score(data, pd.Timestamp(stat_date), candidates)
        weights = bundle.position_strategy.target_weights(scored, args.top_n)

        score_check = verify_composite_score_node(scored, 0)
        weight_check = verify_position_weights_node(weights, args.top_n)
        print(f"[verify] 综合得分核验: {score_check['status']}")
        print(f"[verify] 仓位权重核验: {weight_check['status']} (sum={weight_check.get('total_weight'):.6f})")

        # 核验 period_log 与手工结果一致
        weights_match = set(weights.keys()) == set(target_weights.keys()) and all(
            abs(weights.get(s, 0) - target_weights.get(s, 0)) < 1e-9 for s in set(weights) | set(target_weights)
        )
        print(f"[verify] period_log 与手工执行一致性: {'pass' if weights_match else 'fail'}")

    # 4) 写报表
    out_dir = args.output_dir or (_ws / "myanalyser" / "output" / "pybroker_backtest_verify")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_reports(out_dir, result, data)
    print(f"[verify] 报表: {out_dir}/summary.csv, {out_dir}/period_detail.csv")

    # 5) 输出核验报告（确保可 JSON 序列化）
    def _to_jsonable(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return bool(obj) if isinstance(obj, np.bool_) else int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(x) for x in obj]
        return obj

    report = {
        "metrics_checks": _to_jsonable(metrics_checks),
        "period_log_count": len(period_log),
        "output_dir": str(out_dir),
    }
    report_path = out_dir / "verify_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[verify] 核验报告: {report_path}")

    # 汇总
    all_ok = all(c.get("status") in ("pass", "skip") for c in metrics_checks)
    if period_log:
        all_ok = all_ok and score_check.get("status") == "pass" and weight_check.get("status") == "pass"
    print(f"\n[verify] 核验结论: {'全部通过' if all_ok else '存在失败'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
