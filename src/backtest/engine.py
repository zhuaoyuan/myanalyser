"""回测执行引擎。"""

from __future__ import annotations

import json
import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pybroker as pyb
from pybroker import ExecContext, Strategy, StrategyConfig

from .data import BacktestData
from .strategies.base import FilterStrategy, StrategyBundle

# 低于此值串行，避免线程池创建与切换开销；高于此值并行以加速（与各 filter 典型计算量及线程成本相关）
UNIVERSE_PARALLEL_THRESHOLD = 100


def _split_chunks(lst: list[str], n: int) -> list[list[str]]:
    """将列表均分至最多 n 份。"""
    if n <= 1:
        return [lst]
    size = max(1, (len(lst) + n - 1) // n)
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _filter_symbols_with_parallel(
    filter_strategy: FilterStrategy,
    data: BacktestData,
    as_of_ts: pd.Timestamp,
    universe: list[str],
    *,
    threshold: int = UNIVERSE_PARALLEL_THRESHOLD,
    max_workers: int | None = None,
) -> list[str]:
    """通用：universe 大于阈值时并行调用 filter_symbols，否则串行。"""
    if len(universe) <= threshold:
        return filter_strategy.filter_symbols(data, as_of_ts, universe)
    n = max_workers or min(32, os.cpu_count() or 4)
    chunks = _split_chunks(universe, n)
    result: list[str] = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = {
            ex.submit(filter_strategy.filter_symbols, data, as_of_ts, ch): ch
            for ch in chunks
        }
        for future in as_completed(futures):
            try:
                result.extend(future.result())
            except Exception as e:
                chunk = futures.get(future, [])
                raise RuntimeError(
                    f"filter_symbols failed for chunk (len={len(chunk)}): {chunk[:5]}{'...' if len(chunk) > 5 else ''}"
                ) from e
    return sorted(result)


@dataclass(frozen=True)
class BacktestResult:
    result: object
    period_log: list[dict]


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000
    turnover_half: float = 0.5  # 单向换手近似


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# summary.csv name 字段中英文映射
_CONFIG_NAME_CN = {
    "strategy": "策略",
    "start_date": "起始日期",
    "end_date": "结束日期",
    "rebalance": "调仓周期",
    "top_n": "持仓数量",
    "warmup": "预热天数",
    "initial_cash": "初始资金",
    "max_funds": "最大基金数",
    "nav_dir": "净值目录",
}

# 回测相关环境变量，写入 summary.csv 便于复现
_BACKTEST_ENV_VARS = (
    "FUND_BACKTEST_FILTERS",
    "FILTERED_FUND_CANDIDATES_CSV",
    "FUND_BACKTEST_MAX_FUNDS",
)
_DATA_NAME_CN = {
    "symbols": "基金数量",
    "date_range": "日期范围",
}
_METRICS_PYBROKER_NAME_CN = {
    "trade_count": "交易次数",
    "initial_market_value": "初始市值",
    "end_market_value": "期末市值",
    "total_pnl": "总盈亏",
    "unrealized_pnl": "未实现盈亏",
    "total_return_pct": "总收益率",
    "total_profit": "总盈利",
    "total_loss": "总亏损",
    "total_fees": "总手续费",
    "max_drawdown": "最大回撤额",
    "max_drawdown_pct": "最大回撤率",
    "max_drawdown_date": "最大回撤日期",
    "win_rate": "胜率",
    "loss_rate": "亏损率",
    "winning_trades": "盈利交易数",
    "losing_trades": "亏损交易数",
    "sharpe": "夏普比率",
    "sortino": "索提诺比率",
    "profit_factor": "盈利因子",
    "avg_return_pct": "平均收益率",
    "avg_pnl": "平均盈亏",
    "avg_trade_bars": "平均持仓周期",
    "avg_profit": "平均盈利",
    "avg_profit_pct": "平均盈利比例",
    "avg_winning_trade_bars": "平均盈利持仓周期",
    "avg_loss": "平均亏损",
    "avg_loss_pct": "平均亏损比例",
    "avg_losing_trade_bars": "平均亏损持仓周期",
    "largest_win": "最大单笔盈利",
    "largest_win_pct": "最大单笔盈利比例",
    "largest_win_bars": "最大单笔盈利持仓周期",
    "largest_loss": "最大单笔亏损",
    "largest_loss_pct": "最大单笔亏损比例",
    "largest_loss_bars": "最大单笔亏损持仓周期",
    "max_wins": "连续盈利次数",
    "max_losses": "连续亏损次数",
    "ulcer_index": "溃疡指数",
    "upi": "溃疡绩效指数",
    "equity_r2": "净值R方",
    "std_error": "标准误差",
}


def _build_rebalance_dates(
    trading_dates: list[pd.Timestamp],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    rebalance_period: int,
) -> list[pd.Timestamp]:
    dates = [pd.Timestamp(d).normalize() for d in trading_dates]
    dates = [d for d in dates if start_date <= d <= end_date]
    if not dates:
        return []
    if rebalance_period <= 0:
        return [dates[0]]
    return dates[::rebalance_period]


def run_backtest(
    data: BacktestData,
    bundle: StrategyBundle,
    start_date: str,
    end_date: str,
    top_n: int,
    rebalance_period: int,
    warmup: int,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()

    symbols = sorted(data.by_symbol.keys())
    if len(symbols) < top_n:
        top_n = len(symbols)

    rebalance_dates = _build_rebalance_dates(
        data.trading_dates, start_ts, end_ts, rebalance_period
    )
    rebalance_set = {pd.Timestamp(d).normalize() for d in rebalance_dates}

    period_log: list[dict] = []
    prev_weights: dict[str, float] = {}

    def before_exec(ctxs: dict[str, ExecContext]):
        nonlocal prev_weights
        first_ctx = next(iter(ctxs.values()), None)
        if first_ctx is None:
            return
        current_date = getattr(first_ctx, "date", None)
        if current_date is None:
            return
        if hasattr(current_date, "__len__") and not isinstance(current_date, (str, bytes)):
            try:
                current_date = current_date[-1]
            except Exception:
                pass
        current_ts = pd.Timestamp(current_date).normalize()

        if current_ts not in rebalance_set:
            pyb.param("do_rebalance", False)
            return

        universe = symbols
        candidates = _filter_symbols_with_parallel(
            bundle.filter_strategy, data, current_ts, universe
        )
        scored = bundle.score_strategy.score(
            data, current_ts, candidates
        )
        weights = bundle.position_strategy.target_weights(scored, top_n)

        # 记录期内明细
        gross_turnover = sum(
            abs(weights.get(s, 0.0) - prev_weights.get(s, 0.0))
            for s in set(weights) | set(prev_weights)
        )
        turnover = cfg.turnover_half * gross_turnover
        cash_ratio = 1.0 - sum(weights.values())

        period_log.append(
            {
                "stat_date": current_ts,
                "universe_size": len(universe),
                "candidate_size": len(candidates),
                "top_n": top_n,
                "selected_symbols": list(weights.keys()),
                "target_weights": weights.copy(),
                "prev_weights": prev_weights.copy(),
                "scores_top": scored.head(top_n)[
                    ["symbol", "综合得分", "综合排名"]
                ].to_dict("records")
                if not scored.empty
                else [],
                "turnover": turnover,
                "gross_turnover": gross_turnover,
                "cash_ratio": cash_ratio,
            }
        )

        prev_weights = weights.copy()
        pyb.param("do_rebalance", True)
        pyb.param("target_weights", weights)
        pyb.param("rebalance_date", current_ts)

    def execute(ctx: ExecContext):
        if not pyb.param("do_rebalance"):
            return
        target_weights = pyb.param("target_weights") or {}
        if ctx.long_pos():
            target = target_weights.get(ctx.symbol, 0.0)
            if ctx.symbol not in target_weights or target <= 0:
                ctx.sell_all_shares()
            else:
                ctx.buy_shares = ctx.calc_target_shares(target)
                ctx.hold_bars = rebalance_period
        else:
            if ctx.symbol in target_weights:
                target = target_weights.get(ctx.symbol, 0.0)
                if target > 0:
                    ctx.buy_shares = ctx.calc_target_shares(target)
                    ctx.hold_bars = rebalance_period

    config = StrategyConfig(
        initial_cash=cfg.initial_cash,
        max_long_positions=top_n,
        buy_delay=1,
        sell_delay=1,
        round_fill_price=False,
        bars_per_year=243,  # A 股口径，与 fund_metrics_core 一致
    )

    pyb.param("do_rebalance", False)
    pyb.param("target_weights", {})

    strategy = Strategy(data.long_df, start_ts, end_ts, config=config)
    strategy.set_before_exec(before_exec)
    strategy.add_execution(execute, symbols)
    result = strategy.backtest(warmup=warmup)

    return BacktestResult(result=result, period_log=period_log)


def _extract_portfolio_snapshot(portfolio_df: pd.DataFrame, date: pd.Timestamp) -> dict:
    if portfolio_df is None or portfolio_df.empty:
        return {}
    if not isinstance(portfolio_df.index, pd.DatetimeIndex):
        return {}
    if date in portfolio_df.index:
        row = portfolio_df.loc[date]
    else:
        idx = portfolio_df.index.get_indexer([date], method="nearest")
        if idx.size == 0 or idx[0] < 0:
            return {}
        row = portfolio_df.iloc[idx[0]]
    out = {}
    for col in ["cash", "market_value", "total_equity"]:
        if col in row:
            out[col] = float(row[col])
    return out


def _build_equity_curve(
    portfolio_df: pd.DataFrame | None,
    trading_dates: list[pd.Timestamp],
    initial_cash: float,
) -> pd.DataFrame:
    """从 portfolio 构建每日净值曲线。仅支持 DatetimeIndex，非 DatetimeIndex 会尝试归一化。"""
    eq_aligned = pd.Series(dtype=float)
    if portfolio_df is not None and not portfolio_df.empty:
        col = "total_equity" if "total_equity" in portfolio_df.columns else "market_value"
        if col in portfolio_df.columns:
            pf = portfolio_df.copy()
            if not isinstance(pf.index, pd.DatetimeIndex):
                pf.index = pd.to_datetime(pf.index, errors="coerce")
            if isinstance(pf.index, pd.DatetimeIndex):
                eq = pf[col][~pf.index.duplicated(keep="last")].sort_index().dropna()
                eq_aligned = eq
    if eq_aligned.empty and trading_dates:
        eq_aligned = pd.Series(
            index=pd.DatetimeIndex(trading_dates),
            data=float(initial_cash),
        )
    if eq_aligned.empty:
        return pd.DataFrame(columns=["date", "equity", "cumulative_return"])
    first_val = float(eq_aligned.iloc[0])
    base = first_val if first_val > 0 and not math.isnan(first_val) else 1.0
    cum_ret = eq_aligned / base - 1.0
    return pd.DataFrame({
        "date": eq_aligned.index,
        "equity": eq_aligned.values,
        "cumulative_return": cum_ret.values,
    }).reset_index(drop=True)


def _compute_portfolio_metrics_fund_core(
    equity_curve: pd.DataFrame,
    trading_days_per_year: int = 243,
) -> dict[str, float | None]:
    """用 fund_metrics_core 口径计算组合指标。"""
    if equity_curve.empty or len(equity_curve) < 2:
        return {}
    try:
        from fund_metrics_core import compute_low_risk_debt_metrics, WindowConfig
    except ModuleNotFoundError as e:
        warnings.warn(f"fund_metrics_core 未安装，跳过组合指标计算: {e}")
        return {}

    dates = equity_curve["date"].to_numpy(dtype="datetime64[D]")
    prices = equity_curve["equity"].to_numpy(dtype=float)
    cfg = WindowConfig(trading_days_per_year=trading_days_per_year)
    out = compute_low_risk_debt_metrics(dates, prices, config=cfg)
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}


def _write_html_curves(
    output_dir: Path,
    equity_curve: pd.DataFrame,
    data: BacktestData,
    period_log: list[dict],
    orders_df: pd.DataFrame | None = None,
    max_fund_curves: int = 10,
) -> Path | None:
    """生成 Plotly HTML 收益曲线图。

    时间轴起点为回测第一次买入日期；包含三图：组合净值、组合 vs 成分基金、成分基金走势。
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    if equity_curve.empty or len(equity_curve) < 2:
        return None

    dates = pd.to_datetime(equity_curve["date"])
    portfolio_cum = equity_curve["cumulative_return"].values

    # 回测第一次买入日期作为时间轴起点
    first_buy_date: pd.Timestamp | None = None
    if orders_df is not None and not orders_df.empty and "fill_date" in orders_df.columns:
        buy_orders = orders_df[orders_df["type"] == "buy"]
        if not buy_orders.empty:
            first_buy_date = pd.to_datetime(buy_orders["fill_date"].min())
    if first_buy_date is None:
        first_buy_date = dates.min()

    # 截断到 first_buy_date 起，并重新归一化累计收益
    date_ge = pd.to_datetime(equity_curve["date"]) >= first_buy_date
    equity_trunc = equity_curve.loc[date_ge].reset_index(drop=True)
    if len(equity_trunc) < 2:
        return None
    dates = pd.to_datetime(equity_trunc["date"])
    base_equity = float(equity_trunc["equity"].iloc[0])
    base_equity = base_equity if base_equity > 0 and not math.isnan(base_equity) else 1.0
    portfolio_cum = (equity_trunc["equity"].values.astype(float) / base_equity - 1.0)

    # 收集曾持有的基金，按持有期数排序取前 max_fund_curves
    symbol_counts: dict[str, int] = {}
    for p in period_log:
        for s in p.get("selected_symbols", []):
            symbol_counts[s] = symbol_counts.get(s, 0) + 1
    top_symbols = sorted(symbol_counts, key=lambda x: -symbol_counts[x])[:max_fund_curves]

    # 基金累计收益（归一化到 first_buy_date 当日=1）
    start_ts = first_buy_date
    end_ts = dates.max()
    fund_curves: list[tuple[str, pd.Series]] = []
    for sym in top_symbols:
        df_sym = data.by_symbol.get(sym)
        if df_sym is None or df_sym.empty:
            continue
        sym_dates = pd.to_datetime(df_sym["date"])
        win_mask = (sym_dates >= start_ts) & (sym_dates <= end_ts)
        win = df_sym.loc[win_mask].sort_values("date")
        if len(win) < 2:
            continue
        base = float(win["close"].iloc[0])
        if base <= 0:
            continue
        cum = win["close"].values / base - 1.0
        s = pd.Series(cum, index=win["date"].values)
        fund_curves.append((sym, s))

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("组合净值曲线", "组合 vs 成分基金收益曲线", "成分基金走势"),
        vertical_spacing=0.10,
        row_heights=[0.35, 0.35, 0.30],
    )

    # 图1：组合累计收益
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=portfolio_cum * 100,
            name="组合",
            line=dict(color="#1f77b4", width=2),
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="累计收益率 (%)", row=1, col=1)

    # 图2：组合 + 各基金
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=portfolio_cum * 100,
            name="组合",
            line=dict(color="#1f77b4", width=2.5),
        ),
        row=2,
        col=1,
    )
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8"]
    for i, (sym, s) in enumerate(fund_curves):
        s_dates = pd.to_datetime(s.index)
        s_reindexed = pd.Series(s.values, index=s_dates).reindex(dates, method="ffill").dropna()
        if s_reindexed.empty:
            continue
        y = s_reindexed.values * 100
        x = s_reindexed.index
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=sym,
                line=dict(color=colors[i % len(colors)], width=1, dash="dot"),
            ),
            row=2,
            col=1,
        )
    fig.update_yaxes(title_text="累计收益率 (%)", row=2, col=1)

    # 图3：成分基金走势（仅基金，样本跨度时间内）
    for i, (sym, s) in enumerate(fund_curves):
        s_dates = pd.to_datetime(s.index)
        s_reindexed = pd.Series(s.values, index=s_dates).reindex(dates, method="ffill").dropna()
        if s_reindexed.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s_reindexed.index,
                y=s_reindexed.values * 100,
                name=sym,
                line=dict(color=colors[i % len(colors)], width=1.5),
            ),
            row=3,
            col=1,
        )
    fig.update_yaxes(title_text="累计收益率 (%)", row=3, col=1)

    fig.update_layout(
        height=850,
        title_text="回测收益曲线",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    path = output_dir / "backtest_curves.html"
    fig.write_html(str(path), config={"displayModeBar": True})
    return path


def _render_markdown_report(
    output_dir: Path,
    summary_rows: list[dict],
    detail_df: pd.DataFrame,
    run_config: dict[str, Any],
    curves_html_path: Path | None,
) -> Path:
    """生成 Markdown 报告。"""
    lines = [
        "# PyBroker 回测报告",
        "",
        "## 运行参数",
    ]
    for k, v in run_config.items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## 汇总指标", ""])

    # 按 section 分组
    by_section: dict[str, list[tuple[str, Any]]] = {}
    for r in summary_rows:
        sec = r.get("section", "other")
        name = r.get("name", "")
        val = r.get("value", "")
        if sec not in by_section:
            by_section[sec] = []
        by_section[sec].append((name, val))

    for sec in ["config", "data", "env", "metrics"]:
        if sec not in by_section:
            continue
        lines.append(f"### {sec}")
        for name, val in by_section[sec]:
            lines.append(f"- **{name}**: {val}")
        lines.append("")

    if not detail_df.empty:
        lines.extend(["", "## Top 3 调仓期（按 period_return 降序）", ""])
        if "period_return" in detail_df.columns:
            top = detail_df.dropna(subset=["period_return"]).nlargest(3, "period_return")
        else:
            top = detail_df.head(3)
        lines.append("| stat_date | fill_date | period_return | turnover |")
        lines.append("|-----------|-----------|---------------|----------|")
        for _, row in top.iterrows():
            pr = row.get("period_return", "")
            pr_str = f"{float(pr):.4f}" if pr not in (None, "") and str(pr) != "nan" else "-"
            lines.append(
                f"| {row.get('stat_date', '')} | {row.get('fill_date', '')} | {pr_str} | {row.get('turnover', '')} |"
            )
        lines.append("")

    lines.extend(["", "## 输出文件", ""])
    for f in ["summary.csv", "period_detail.csv", "equity_curve.csv", "orders.csv", "positions_flat.csv"]:
        p = output_dir / f
        lines.append(f"- {f}")
    if curves_html_path and curves_html_path.exists():
        lines.append(f"- [backtest_curves.html]({curves_html_path.name})（收益曲线可视化）")
    lines.append("")

    report_path = output_dir / "backtest_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_reports(
    output_dir: Path,
    backtest_result: BacktestResult,
    data: BacktestData,
    run_config: dict[str, Any] | None = None,
    initial_cash: float = 100_000,
) -> dict[str, Path]:
    _ensure_dir(output_dir)

    result = backtest_result.result
    period_log = backtest_result.period_log
    run_config = run_config or {}

    # 运行参数写入 summary（name 统一为中文）
    summary_rows = []
    for k, v in run_config.items():
        name_cn = _CONFIG_NAME_CN.get(str(k), str(k))
        summary_rows.append({"section": "config", "name": name_cn, "value": v if v is None else str(v)})
    summary_rows.append({"section": "data", "name": _DATA_NAME_CN["symbols"], "value": len(data.by_symbol)})
    # 记录回测相关环境变量
    for env_name in _BACKTEST_ENV_VARS:
        val = os.environ.get(env_name, "")
        summary_rows.append({"section": "env", "name": env_name, "value": val if val else "(未设置)"})
    if data.trading_dates:
        summary_rows.append({
            "section": "data",
            "name": _DATA_NAME_CN["date_range"],
            "value": f"{data.trading_dates[0].date()} ~ {data.trading_dates[-1].date()}",
        })

    orders_df = getattr(result, "orders", None)
    if orders_df is None or orders_df.empty:
        orders_df = pd.DataFrame(columns=["type", "symbol", "date", "shares", "fill_price"])
    orders_df = orders_df.copy()
    if "date" in orders_df.columns:
        orders_df["fill_date"] = pd.to_datetime(orders_df["date"], errors="coerce")
    else:
        orders_df["fill_date"] = pd.NaT

    # 独立 orders.csv
    orders_out = orders_df.copy()
    if "fill_date" in orders_out.columns:
        orders_out["fill_date"] = orders_out["fill_date"].dt.strftime("%Y-%m-%d")
    orders_path = output_dir / "orders.csv"
    orders_out.to_csv(orders_path, index=False, encoding="utf-8-sig")

    portfolio_df = getattr(result, "portfolio", None)
    equity_curve = _build_equity_curve(
        portfolio_df,
        data.trading_dates,
        initial_cash,
    )
    equity_path = output_dir / "equity_curve.csv"
    equity_curve.to_csv(equity_path, index=False, encoding="utf-8-sig")

    # fund_metrics_core 指标
    metrics_core = _compute_portfolio_metrics_fund_core(equity_curve)
    for name, val in metrics_core.items():
        if val is not None:
            summary_rows.append({"section": "metrics", "name": name, "value": val})

    # PyBroker 原生 metrics_df
    metrics_df = getattr(result, "metrics_df", None)
    if metrics_df is not None and not metrics_df.empty:
        for _, row in metrics_df.iterrows():
            name_en = row.get("name")
            name_cn = _METRICS_PYBROKER_NAME_CN.get(str(name_en), name_en) if name_en else name_en
            summary_rows.append(
                {"section": "metrics_pybroker", "name": name_cn, "value": row.get("value")}
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    trading_dates = data.trading_dates
    fill_map = {}
    for i in range(1, len(trading_dates)):
        fill_map[pd.Timestamp(trading_dates[i - 1])] = pd.Timestamp(trading_dates[i])

    # 构建 equity 按日期查找
    equity_by_date = {}
    if not equity_curve.empty:
        for _, r in equity_curve.iterrows():
            d = pd.Timestamp(r["date"]).normalize()
            equity_by_date[d] = float(r["equity"])

    stat_dates = [pd.Timestamp(p["stat_date"]).normalize() for p in period_log]

    detail_rows = []
    position_flat_rows = []
    for i, p in enumerate(period_log):
        stat_date = pd.Timestamp(p["stat_date"])
        fill_date = fill_map.get(stat_date)
        buys = []
        sells = []
        if fill_date is not None and not orders_df.empty:
            mask = orders_df["fill_date"] == fill_date
            for r in orders_df[mask].to_dict("records"):
                sym = r.get("symbol", "?")
                sh = r.get("shares", 0)
                fp = r.get("fill_price", 0)
                if r.get("type") == "buy":
                    buys.append({"symbol": sym, "shares": sh, "fill_price": fp})
                else:
                    sells.append({"symbol": sym, "shares": sh, "fill_price": fp})

        snapshot = _extract_portfolio_snapshot(portfolio_df, stat_date)
        eq_curr = snapshot.get("total_equity") or equity_by_date.get(stat_date)

        # period_return
        period_return = None
        if i + 1 < len(stat_dates):
            eq_next = equity_by_date.get(stat_dates[i + 1]) or _extract_portfolio_snapshot(portfolio_df, stat_dates[i + 1]).get("total_equity")
            if eq_curr is not None and eq_next is not None and float(eq_curr) > 0:
                period_return = float(eq_next) / float(eq_curr) - 1.0
        elif eq_curr is not None and len(equity_by_date) > 0:
            last_d = max(equity_by_date.keys())
            eq_next = equity_by_date.get(last_d)
            if eq_next is not None and float(eq_curr) > 0:
                period_return = float(eq_next) / float(eq_curr) - 1.0

        weights = p.get("target_weights", {})
        for rank, (sym, w) in enumerate(weights.items(), start=1):
            position_flat_rows.append({
                "stat_date": stat_date.strftime("%Y-%m-%d"),
                "symbol": sym,
                "weight": w,
                "rank": rank,
            })

        detail_rows.append(
            {
                "stat_date": stat_date.strftime("%Y-%m-%d"),
                "fill_date": fill_date.strftime("%Y-%m-%d") if fill_date else "",
                "universe_size": p["universe_size"],
                "candidate_size": p["candidate_size"],
                "top_n": p["top_n"],
                "period_return": period_return,
                "selected_symbols": json.dumps(p["selected_symbols"], ensure_ascii=False),
                "target_weights": json.dumps(p["target_weights"], ensure_ascii=False),
                "prev_weights": json.dumps(p["prev_weights"], ensure_ascii=False),
                "scores_top": json.dumps(p["scores_top"], ensure_ascii=False),
                "turnover": p["turnover"],
                "gross_turnover": p["gross_turnover"],
                "cash_ratio": p["cash_ratio"],
                "orders_buy": json.dumps(buys, ensure_ascii=False),
                "orders_sell": json.dumps(sells, ensure_ascii=False),
                "portfolio_cash": snapshot.get("cash"),
                "portfolio_market_value": snapshot.get("market_value"),
                "portfolio_total_equity": eq_curr,
            }
        )

    detail_columns = [
        "stat_date", "fill_date", "universe_size", "candidate_size", "top_n", "period_return",
        "selected_symbols", "target_weights", "prev_weights", "scores_top",
        "turnover", "gross_turnover", "cash_ratio",
        "orders_buy", "orders_sell",
        "portfolio_cash", "portfolio_market_value", "portfolio_total_equity",
    ]
    detail_df = pd.DataFrame(detail_rows, columns=detail_columns)
    detail_path = output_dir / "period_detail.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    positions_path = output_dir / "positions_flat.csv"
    pd.DataFrame(
        position_flat_rows,
        columns=["stat_date", "symbol", "weight", "rank"],
    ).to_csv(positions_path, index=False, encoding="utf-8-sig")

    # HTML 曲线
    curves_path = _write_html_curves(
        output_dir, equity_curve, data, period_log, orders_df=orders_df
    )

    # Markdown 报告
    md_path = _render_markdown_report(
        output_dir,
        summary_rows,
        detail_df,
        run_config,
        curves_path,
    )

    out: dict[str, Path] = {
        "summary": summary_path,
        "detail": detail_path,
        "equity_curve": equity_path,
        "orders": orders_path,
        "positions_flat": positions_path,
        "report_md": md_path,
    }
    if curves_path is not None:
        out["curves_html"] = curves_path
    return out
