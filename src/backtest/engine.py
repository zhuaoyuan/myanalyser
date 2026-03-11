"""回测执行引擎。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pybroker as pyb
from pybroker import ExecContext, Strategy, StrategyConfig

from .data import BacktestData
from .strategies.base import StrategyBundle


@dataclass(frozen=True)
class BacktestResult:
    result: object
    period_log: list[dict]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
) -> BacktestResult:
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
        if hasattr(current_date, "__len__"):
            current_date = current_date[-1]
        current_ts = pd.Timestamp(current_date).normalize()

        if current_ts not in rebalance_set:
            pyb.param("do_rebalance", False)
            return

        universe = symbols
        candidates = bundle.filter_strategy.filter_symbols(
            data, current_ts, universe
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
        turnover = 0.5 * gross_turnover
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
            if ctx.symbol not in target_weights:
                ctx.sell_all_shares()
            else:
                target = target_weights.get(ctx.symbol, 0.0)
                ctx.buy_shares = ctx.calc_target_shares(target)
                ctx.hold_bars = rebalance_period
        else:
            if ctx.symbol in target_weights:
                target = target_weights.get(ctx.symbol, 0.0)
                ctx.buy_shares = ctx.calc_target_shares(target)
                ctx.hold_bars = rebalance_period

    config = StrategyConfig(
        initial_cash=100_000,
        max_long_positions=top_n,
        buy_delay=1,
        sell_delay=1,
        round_fill_price=False,
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


def write_reports(
    output_dir: Path,
    backtest_result: BacktestResult,
    data: BacktestData,
) -> dict[str, Path]:
    _ensure_dir(output_dir)

    result = backtest_result.result
    period_log = backtest_result.period_log

    summary_rows = []
    summary_rows.append({"section": "data", "name": "symbols", "value": len(data.by_symbol)})
    if data.trading_dates:
        summary_rows.append({
            "section": "data",
            "name": "date_range",
            "value": f"{data.trading_dates[0].date()} ~ {data.trading_dates[-1].date()}",
        })

    metrics_df = getattr(result, "metrics_df", None)
    if metrics_df is not None and not metrics_df.empty:
        for _, row in metrics_df.iterrows():
            summary_rows.append(
                {"section": "metrics", "name": row.get("name"), "value": row.get("value")}
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    orders_df = getattr(result, "orders", None)
    if orders_df is None or orders_df.empty:
        orders_df = pd.DataFrame(columns=["type", "symbol", "date", "shares", "fill_price"])
    orders_df = orders_df.copy()
    if "date" in orders_df.columns:
        orders_df["fill_date"] = pd.to_datetime(orders_df["date"], errors="coerce")
    else:
        orders_df["fill_date"] = pd.NaT

    portfolio_df = getattr(result, "portfolio", None)

    detail_rows = []
    trading_dates = data.trading_dates
    fill_map = {}
    for i in range(1, len(trading_dates)):
        fill_map[pd.Timestamp(trading_dates[i - 1])] = pd.Timestamp(trading_dates[i])

    for p in period_log:
        stat_date = pd.Timestamp(p["stat_date"])
        fill_date = fill_map.get(stat_date)
        buys = []
        sells = []
        if fill_date is not None and not orders_df.empty:
            mask = orders_df["fill_date"] == fill_date
            for _, r in orders_df[mask].iterrows():
                sym = r.get("symbol", "?")
                sh = r.get("shares", 0)
                fp = r.get("fill_price", 0)
                if r.get("type") == "buy":
                    buys.append({"symbol": sym, "shares": sh, "fill_price": fp})
                else:
                    sells.append({"symbol": sym, "shares": sh, "fill_price": fp})

        snapshot = _extract_portfolio_snapshot(portfolio_df, stat_date)

        detail_rows.append(
            {
                "stat_date": stat_date.strftime("%Y-%m-%d"),
                "fill_date": fill_date.strftime("%Y-%m-%d") if fill_date else "",
                "universe_size": p["universe_size"],
                "candidate_size": p["candidate_size"],
                "top_n": p["top_n"],
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
                "portfolio_total_equity": snapshot.get("total_equity"),
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    detail_path = output_dir / "period_detail.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    return {"summary": summary_path, "detail": detail_path}
