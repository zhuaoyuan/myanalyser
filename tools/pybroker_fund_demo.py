#!/usr/bin/env python3
"""PyBroker 基金回测演示脚本。

使用 fund_adjusted_nav_by_code 目录下的复权净值 CSV 作为数据源，
跑通一个简单的轮动策略（按 20 日收益率排名，持有 Top2 基金）。

依赖：pip install lib-pybroker（项目 venv 已包含）

用法：
  python tools/pybroker_fund_demo.py
  python tools/pybroker_fund_demo.py --nav-dir /path/to/fund_adjusted_nav_by_code
  python tools/pybroker_fund_demo.py --max-funds 10 --start-date 2024-01-01
  python tools/pybroker_fund_demo.py --no-detail  # 仅输出汇总指标，不输出每期详情
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保能 import 项目模块（project_paths 在 src/）
_SCRIPT_DIR = Path(__file__).resolve().parent
_MYANALYSER_ROOT = _SCRIPT_DIR.parent
_SRC = _MYANALYSER_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from project_paths import project_root

# 默认数据源路径（用户指定的 run 产物）
_DEFAULT_NAV_DIR = (
    project_root().parent
    / "finance-runs"
    / "run_20260301_1_formal_retry_step4_rerun"
    / "data"
    / "versions"
    / "20260301_1_formal_retry_step4_rerun"
    / "fund_etl"
    / "fund_adjusted_nav_by_code"
)


def load_fund_nav_data(
    nav_dir: Path,
    max_funds: int = 20,
    start_date: str | None = None,
    end_date: str | None = None,
) -> "pd.DataFrame":
    """从 fund_adjusted_nav_by_code 目录加载复权净值，转换为 pybroker 所需格式。"""
    import pandas as pd

    nav_dir = Path(nav_dir).resolve()
    if not nav_dir.is_dir():
        raise FileNotFoundError(f"净值目录不存在: {nav_dir}")

    csv_files = sorted(nav_dir.glob("*.csv"))[:max_funds]
    if not csv_files:
        raise ValueError(f"净值目录下没有 CSV 文件: {nav_dir}")

    rows = []
    for p in csv_files:
        df = pd.read_csv(p, dtype={"基金代码": str}, encoding="utf-8-sig")
        if df.empty or "净值日期" not in df.columns or "复权净值" not in df.columns:
            continue
        df["净值日期"] = pd.to_datetime(df["净值日期"], errors="coerce")
        df = df.dropna(subset=["净值日期", "复权净值"])
        if start_date:
            df = df[df["净值日期"] >= start_date]
        if end_date:
            df = df[df["净值日期"] <= end_date]
        if df.empty:
            continue
        code = df["基金代码"].iloc[0] if "基金代码" in df.columns else p.stem
        for _, r in df.iterrows():
            nav = float(r["复权净值"])
            rows.append(
                {
                    "symbol": str(code).zfill(6),
                    "date": r["净值日期"],
                    "open": nav,
                    "high": nav,
                    "low": nav,
                    "close": nav,
                }
            )

    if not rows:
        raise ValueError("未加载到任何有效净值数据")

    out = pd.DataFrame(rows)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    return out


def run_simple_rotation_backtest(
    df: "pd.DataFrame",
    start_date: str,
    end_date: str,
    top_n: int = 2,
    rank_period: int = 20,
    hold_bars: int = 20,
    warmup: int = 30,
) -> "tuple[object, list[dict]]":
    """简单轮动策略：按 rank_period 日收益率排名，持有 TopN，每 hold_bars 日调仓。

    返回 (result, period_log)，period_log 记录每期的候选基金、统计值等，用于详细报告。
    """
    import numpy as np
    import pybroker as pyb
    from pybroker import ExecContext, Strategy, StrategyConfig

    symbols = df["symbol"].unique().tolist()
    if len(symbols) < top_n:
        top_n = len(symbols)

    # 收集每期决策数据，用于详细报告
    period_log: list[dict] = []

    def roc_indicator(data):
        close = np.asarray(data.close, dtype=float)
        out = np.full_like(close, np.nan)
        if len(close) > rank_period:
            out[rank_period:] = close[rank_period:] / close[:-rank_period] - 1.0
        return out

    roc = pyb.indicator("roc", roc_indicator)

    pyb.param("top_n", top_n)
    pyb.param("rank_period", rank_period)
    pyb.param("hold_bars", hold_bars)

    def rank_before_exec(ctxs: dict[str, ExecContext]):
        scores = {}
        for sym, ctx in ctxs.items():
            roc_val = ctx.indicator("roc")
            scores[sym] = float(roc_val[-1]) if len(roc_val) > 0 and roc_val[-1] == roc_val[-1] else -999
        sorted_syms = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)
        top_symbols = sorted_syms[:top_n]
        pyb.param("top_symbols", top_symbols)

        # 当前 bar 的统计日期（用于详细报告）
        stat_date = None
        first_ctx = next(iter(ctxs.values()), None)
        if first_ctx is not None:
            try:
                d = getattr(first_ctx, "date", None)
                if d is not None:
                    stat_date = d[-1] if hasattr(d, "__len__") and len(d) > 0 else d
            except Exception:
                pass

        period_log.append({
            "stat_date": stat_date,
            "top_symbols": top_symbols.copy(),
            "scores": {k: round(v, 6) for k, v in scores.items()},
        })

    def rotate(ctx: ExecContext):
        top_symbols = pyb.param("top_symbols")
        if ctx.long_pos():
            if ctx.symbol not in top_symbols:
                ctx.sell_all_shares()
        else:
            if ctx.symbol in top_symbols:
                target = 1.0 / pyb.param("top_n")
                ctx.buy_shares = ctx.calc_target_shares(target)
                ctx.hold_bars = pyb.param("hold_bars")

    config = StrategyConfig(
        initial_cash=100_000,
        max_long_positions=top_n,
        buy_delay=1,
        sell_delay=1,
        round_fill_price=False,  # 基金净值需完整精度，否则会截断到 2 位小数导致 PnL 失真
    )
    strategy = Strategy(df, start_date, end_date, config=config)
    strategy.set_before_exec(rank_before_exec)
    strategy.add_execution(rotate, symbols, indicators=roc)
    result = strategy.backtest(warmup=warmup)
    return result, period_log


def _format_detailed_report(
    result: "object",
    period_log: "list[dict]",
    df: "pd.DataFrame",
    rank_period: int = 20,
) -> None:
    """格式化打印每期详细报告。"""
    import pandas as pd

    if not period_log:
        return

    # 交易日历
    trading_dates = sorted(pd.Series(df["date"].unique()).dropna().tolist())
    if not trading_dates:
        return

    # 构建 fill_date -> 前一交易日（决策日）
    fill_to_stat = {}
    for i in range(1, len(trading_dates)):
        fill_to_stat[pd.Timestamp(trading_dates[i])] = pd.Timestamp(trading_dates[i - 1])

    # 按 stat_date 索引 period_log
    stat_to_period = {}
    for p in period_log:
        sd = p.get("stat_date")
        if sd is not None:
            try:
                ts = pd.Timestamp(sd)
                stat_to_period[ts] = p
            except Exception:
                pass

    # 从 result.orders 按 fill_date 分组（订单的 date 为成交日）
    orders_df = getattr(result, "orders", None)
    if orders_df is None or orders_df.empty:
        orders_df = pd.DataFrame(columns=["type", "symbol", "date", "shares", "fill_price"])

    # 标准化订单日期
    if "date" in orders_df.columns and not orders_df.empty:
        orders_df = orders_df.copy()
        orders_df["fill_date"] = pd.to_datetime(orders_df["date"], errors="coerce")
    else:
        orders_df["fill_date"] = pd.NaT

    portfolio_df = getattr(result, "portfolio", None)
    daily_ret = None
    if portfolio_df is not None and not portfolio_df.empty and "market_value" in portfolio_df.columns:
        mv = portfolio_df["market_value"].sort_index()
        daily_ret = mv.pct_change()

    # 仅在有买入/卖出的期输出详情
    periods_with_orders: set[pd.Timestamp] = set()
    for _, row in orders_df.iterrows():
        fd = row.get("fill_date")
        if pd.isna(fd):
            continue
        fd_ts = pd.Timestamp(fd)
        stat_ts = fill_to_stat.get(fd_ts)
        if stat_ts is not None:
            periods_with_orders.add(stat_ts)

    print("\n" + "=" * 80)
    print("每期决策详情（仅展示有调仓的期）")
    print("=" * 80)

    for stat_ts in sorted(stat_to_period.keys()):
        if stat_ts not in periods_with_orders:
            continue
        p = stat_to_period[stat_ts]
        stat_str = pd.Timestamp(stat_ts).strftime("%Y-%m-%d")
        next_idx = next(
            (i for i, d in enumerate(trading_dates) if pd.Timestamp(d) >= stat_ts),
            len(trading_dates) - 1,
        )
        next_date = trading_dates[min(next_idx + 1, len(trading_dates) - 1)] if next_idx < len(trading_dates) - 1 else None
        fill_str = pd.Timestamp(next_date).strftime("%Y-%m-%d") if next_date else "-"

        print(f"\n【统计日期】{stat_str}  成交日期(T+1) {fill_str}")

        print(f"  候选基金(Top{len(p['top_symbols'])}): {', '.join(p['top_symbols'])}")
        scores = p.get("scores", {})
        score_str = ", ".join(f"{s}:{v:.2%}" for s, v in sorted(scores.items(), key=lambda x: -x[1])[:10])
        print(f"  {rank_period}日收益率(决策统计值): {score_str}" + ("..." if len(scores) > 10 else ""))

        # 当日下达、次日成交的订单
        fill_ts = pd.Timestamp(next_date) if next_date else None
        buys = []
        sells = []
        if fill_ts is not None and "fill_date" in orders_df.columns:
            mask = orders_df["fill_date"] == fill_ts
            for _, r in orders_df[mask].iterrows():
                sym = r.get("symbol", "?")
                sh = r.get("shares", 0)
                fp = r.get("fill_price", 0)
                if r.get("type") == "buy":
                    buys.append((sym, fp, sh))
                else:
                    sells.append((sym, fp, sh))

        if buys:
            for sym, price, shares in buys:
                print(f"  买入: {sym} @ {price:.4f}  {shares:.2f} 份")
        if sells:
            for sym, price, shares in sells:
                print(f"  卖出: {sym} @ {price:.4f}  {shares:.2f} 份")

        if daily_ret is not None:
            try:
                ret = daily_ret.loc[stat_ts] if stat_ts in daily_ret.index else None
                if ret is not None and pd.notna(ret):
                    print(f"  当期收益: {float(ret):.2%}")
            except Exception:
                try:
                    idx = daily_ret.index.get_indexer([pd.Timestamp(stat_ts)], method="nearest")[0]
                    if idx >= 0:
                        ret = daily_ret.iloc[idx]
                        if pd.notna(ret):
                            print(f"  当期收益: {float(ret):.2%}")
                except Exception:
                    pass

    print("\n" + "=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyBroker 基金回测演示：基于复权净值的简单轮动策略"
    )
    parser.add_argument(
        "--nav-dir",
        type=Path,
        default=_DEFAULT_NAV_DIR,
        help=f"复权净值目录，默认 {_DEFAULT_NAV_DIR}",
    )
    parser.add_argument(
        "--max-funds",
        type=int,
        default=20,
        help="最多加载基金数量（为加速演示，默认 20）",
    )
    parser.add_argument(
        "--start-date",
        default="2023-01-01",
        help="回测起始日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        default="2025-12-31",
        help="回测结束日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=2,
        help="轮动持有基金数量",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="策略预热 bar 数",
    )
    parser.add_argument(
        "--no-detail",
        action="store_true",
        help="不输出每期详细报告，仅输出汇总指标",
    )
    args = parser.parse_args()

    print(f"[pybroker_fund_demo] 加载数据: {args.nav_dir}")
    df = load_fund_nav_data(
        args.nav_dir,
        max_funds=args.max_funds,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    symbols = df["symbol"].unique().tolist()
    date_range = f"{df['date'].min().date()} ~ {df['date'].max().date()}"
    print(f"[pybroker_fund_demo] 基金数: {len(symbols)}, 日期: {date_range}, 行数: {len(df)}")

    print("[pybroker_fund_demo] 运行轮动回测...")
    result, period_log = run_simple_rotation_backtest(
        df,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        rank_period=20,
        hold_bars=20,
        warmup=args.warmup,
    )

    print("\n=== 回测结果 ===")
    if hasattr(result, "metrics_df") and result.metrics_df is not None:
        for _, row in result.metrics_df.iterrows():
            print(f"  {row['name']}: {row['value']}")

    if hasattr(result, "orders") and result.orders is not None and not result.orders.empty:
        print(f"\n订单数: {len(result.orders)}")
        print(result.orders.head(10).to_string())

    if not args.no_detail:
        _format_detailed_report(result, period_log, df, rank_period=20)


if __name__ == "__main__":
    main()
