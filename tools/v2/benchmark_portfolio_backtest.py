#!/usr/bin/env python3
"""固定基金组合净值计算：模拟被动组合的建仓与周期性再平衡。

与 multi_t_backtest 不同，本脚本不做基金筛选/评分，仅对指定固定权重组合
做净值追踪和周期性再平衡模拟，适用于比较基准等被动策略的回测。

用法:
  python myanalyser/tools/v2/benchmark_portfolio_backtest.py \
    --fund-etl-dir myanalyser/data/versions/RUN_ID/fund_etl \
    --start-date 2015-02-27 --end-date 2026-03-13 \
    --rebalance-interval 243 \
    --portfolio "161119:0.70,510300:0.30" \
    --output-dir myanalyser/artifacts/backtest_base/RUN_ID/稳健型_A
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_MYANALYSER_ROOT = _SCRIPT_DIR.parent.parent
_SRC = _MYANALYSER_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from check_trade_day_data_integrity import compute_integrity_for_fund, load_trade_days
from fund_metrics_core import WindowConfig, compute_holding_period_metrics

logger = logging.getLogger(__name__)

DEFAULT_INITIAL_CAPITAL = 1_000_000.0
DEFAULT_INTEGRITY_THRESHOLD = 0.95
DEFAULT_COMPARE_THRESHOLD = 0.80
DEFAULT_TRADE_DATES_CSV = _MYANALYSER_ROOT / "data" / "common" / "trade_dates.csv"


# ---------------------------------------------------------------------------
# Portfolio parsing
# ---------------------------------------------------------------------------

def parse_portfolio(portfolio_str: str) -> list[tuple[str, float]]:
    """解析 'code:weight,code:weight,...' 为 [(code, weight), ...]。"""
    items: list[tuple[str, float]] = []
    seen: set[str] = set()
    for part in portfolio_str.split(","):
        part = part.strip()
        if not part:
            continue
        pieces = part.split(":")
        if len(pieces) != 2:
            raise ValueError(f"格式错误（应为 code:weight）: {part}")
        code = pieces[0].strip().zfill(6)
        weight = float(pieces[1].strip())
        if code in seen:
            raise ValueError(f"重复基金代码: {code}")
        seen.add(code)
        items.append((code, weight))
    if not items:
        raise ValueError("组合为空")
    total_weight = sum(w for _, w in items)
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"权重之和 {total_weight:.8f} != 1.0（容差 1e-6）")
    return items


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

def validate_integrity(
    fund_etl_dir: Path,
    fund_codes: list[str],
    start_date: str,
    end_date: str,
    threshold: float,
    trade_dates_csv: Path,
) -> list[str]:
    """校验各基金在 [start, end] 内的交易日数据完整性。返回错误列表。"""
    errors: list[str] = []
    adjusted_nav_dir = fund_etl_dir / "fund_adjusted_nav_by_code"

    trade_days = load_trade_days(trade_dates_csv, start_date, end_date)
    if not trade_days:
        errors.append(f"交易日历在 [{start_date}, {end_date}] 内无交易日")
        return errors

    for code in fund_codes:
        fund_csv = adjusted_nav_dir / f"{code}.csv"
        if not fund_csv.exists():
            errors.append(f"基金 {code}: 复权净值文件不存在 ({fund_csv})")
            continue
        _, ratio, _ = compute_integrity_for_fund(fund_csv, trade_days)
        if ratio < threshold:
            errors.append(
                f"基金 {code}: 数据完整比例 {ratio:.4f} < 阈值 {threshold:.4f}"
            )
        else:
            logger.info("基金 %s: 数据完整比例 %.4f >= %.4f ✓", code, ratio, threshold)

    return errors


def validate_compare(
    fund_etl_dir: Path,
    fund_codes: list[str],
    start_date: str,
    end_date: str,
    threshold: float,
) -> list[str]:
    """校验复权净值与累计收益率一致性。无 cum_return 数据时降级跳过。"""
    cum_return_dir = fund_etl_dir / "fund_cum_return_by_code"
    adjusted_nav_dir = fund_etl_dir / "fund_adjusted_nav_by_code"

    if not cum_return_dir.is_dir():
        logger.warning(
            "fund_cum_return_by_code 目录不存在，跳过全部 compare 校验（降级为仅 integrity）"
        )
        return []

    errors: list[str] = []
    skipped: list[str] = []
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)

    for code in fund_codes:
        adjusted_csv = adjusted_nav_dir / f"{code}.csv"
        cum_csv = cum_return_dir / f"{code}.csv"

        if not cum_csv.exists():
            skipped.append(code)
            continue
        if not adjusted_csv.exists():
            errors.append(f"基金 {code}: 复权净值文件不存在")
            continue

        try:
            ratio = _per_fund_compare(adjusted_csv, cum_csv, start_ts, end_ts)
        except Exception as exc:
            errors.append(f"基金 {code}: compare 异常 — {exc}")
            continue

        if ratio is None:
            skipped.append(code)
            continue

        if ratio < threshold:
            errors.append(
                f"基金 {code}: <1%偏差占比 {ratio:.4f} < 阈值 {threshold:.4f}"
            )
        else:
            logger.info("基金 %s: <1%%偏差占比 %.4f >= %.4f ✓", code, ratio, threshold)

    if skipped:
        logger.warning(
            "以下基金无 cum_return 数据或窗口内无可比对数据，跳过 compare（降级）: %s",
            ", ".join(skipped),
        )
    return errors


def _per_fund_compare(
    adjusted_csv: Path,
    cum_csv: Path,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> float | None:
    """单只基金的简化 compare：返回 <1% 偏差占比，无可比数据时返回 None。"""
    adj_df = pd.read_csv(adjusted_csv, dtype=str, encoding="utf-8-sig")
    cum_df = pd.read_csv(cum_csv, dtype=str, encoding="utf-8-sig")

    adj_series = _to_series(adj_df, "净值日期", "复权净值", start_ts, end_ts)
    cum_series = _to_series(cum_df, "日期", "累计收益率", start_ts, end_ts)
    if adj_series.empty or cum_series.empty:
        return None

    common = sorted(set(adj_series.index) & set(cum_series.index))
    if len(common) < 2:
        return None

    window_end = common[-1]
    good, total = 0, 0
    # cum_return 以 100 为基准（即 100 + 累计收益率%）
    _CUM_RETURN_BASE = 100.0
    for d in common[:-1]:
        local_s = float(adj_series.loc[d])
        local_e = float(adj_series.loc[window_end])
        if local_s <= 0:
            continue
        local_ret = (local_e - local_s) / local_s

        remote_s = float(cum_series.loc[d])
        remote_e = float(cum_series.loc[window_end])
        denom = remote_s + _CUM_RETURN_BASE
        if abs(denom) < 1e-9:
            continue
        remote_ret = (remote_e - remote_s) / denom

        total += 1
        max_abs = max(abs(local_ret), abs(remote_ret))
        if max_abs < 1e-8:
            good += 1
            continue
        if abs(local_ret - remote_ret) / max_abs < 0.01:
            good += 1

    return (good / total) if total > 0 else None


def _to_series(
    df: pd.DataFrame, date_col: str, value_col: str,
    start_ts: pd.Timestamp, end_ts: pd.Timestamp,
) -> pd.Series:
    dates = pd.to_datetime(df[date_col], errors="coerce")
    vals = pd.to_numeric(df[value_col], errors="coerce")
    s = pd.Series(vals.values, index=dates).dropna().sort_index()
    return s[(s.index >= start_ts) & (s.index <= end_ts)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_adjusted_nav(
    fund_etl_dir: Path,
    fund_codes: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, pd.Series]:
    """加载各基金复权净值，返回 {code: Series(index=date, value=float)}。"""
    adjusted_nav_dir = fund_etl_dir / "fund_adjusted_nav_by_code"
    result: dict[str, pd.Series] = {}
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)

    for code in fund_codes:
        csv_path = adjusted_nav_dir / f"{code}.csv"
        df = pd.read_csv(csv_path, dtype={"基金代码": str}, encoding="utf-8-sig")
        df["净值日期"] = pd.to_datetime(df["净值日期"], errors="coerce")
        df["复权净值"] = pd.to_numeric(df["复权净值"], errors="coerce")
        df = df.dropna(subset=["净值日期", "复权净值"])
        df = df[(df["净值日期"] >= start_ts) & (df["净值日期"] <= end_ts)]
        df = df.sort_values("净值日期").drop_duplicates(subset=["净值日期"], keep="last")
        result[code] = pd.Series(df["复权净值"].values, index=df["净值日期"], dtype=float)

    return result


def load_trade_calendar(
    trade_dates_csv: Path, start_date: str, end_date: str,
) -> list[pd.Timestamp]:
    """加载交易日历，返回 [start, end] 内的交易日列表。"""
    td = pd.read_csv(trade_dates_csv, dtype={"trade_date": str}, encoding="utf-8-sig")
    dates = pd.to_datetime(td["trade_date"], errors="coerce").dropna().sort_values()
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
    return [d for d in dates if start_ts <= d <= end_ts]


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------

def simulate_portfolio(
    nav_data: dict[str, pd.Series],
    portfolio: list[tuple[str, float]],
    trade_dates: list[pd.Timestamp],
    rebalance_interval: int,
    initial_capital: float,
) -> pd.DataFrame:
    """模拟固定权重组合的建仓与周期再平衡。

    返回 DataFrame: date(str), equity(float), cumulative_return(float)
    """
    codes = [code for code, _ in portfolio]
    weights = np.array([w for _, w in portfolio])

    price_df = pd.DataFrame(index=trade_dates, dtype=float)
    for code in codes:
        price_df[code] = nav_data[code].reindex(trade_dates)

    for code in codes:
        if price_df[code].isna().all():
            raise ValueError(f"基金 {code} 在日期范围内无任何净值数据")
    price_df = price_df.ffill().dropna()
    if price_df.empty:
        raise ValueError("前向填充后无完整数据行")

    dates = list(price_df.index)
    prices = price_df.values  # (n_dates, n_funds)

    if np.any(prices[0] <= 0):
        bad = [codes[i] for i, p in enumerate(prices[0]) if p <= 0]
        raise ValueError(f"首日净值为零或负值: {bad}")
    shares = initial_capital * weights / prices[0]
    equity_values: list[dict] = []

    for i in range(len(dates)):
        pv = float(np.sum(shares * prices[i]))
        equity_values.append({"date": dates[i], "equity": pv})
        if i > 0 and i % rebalance_interval == 0:
            shares = pv * weights / prices[i]

    eq_df = pd.DataFrame(equity_values)
    base_eq = float(eq_df["equity"].iloc[0])
    eq_df["cumulative_return"] = eq_df["equity"] / base_eq - 1.0
    eq_df["date"] = pd.to_datetime(eq_df["date"]).dt.strftime("%Y-%m-%d")
    return eq_df


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_outputs(
    equity_curve: pd.DataFrame,
    portfolio: list[tuple[str, float]],
    output_dir: Path,
    start_date: str,
    end_date: str,
    rebalance_interval: int,
    initial_capital: float,
) -> None:
    """写入 equity_curve.csv / summary.csv / backtest_report.md / backtest_curves.html。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio_str = ", ".join(f"{c}:{w:.2%}" for c, w in portfolio)

    # equity_curve.csv
    equity_path = output_dir / "equity_curve.csv"
    equity_curve.to_csv(equity_path, index=False, encoding="utf-8-sig")
    logger.info("equity_curve -> %s (%d rows)", equity_path, len(equity_curve))

    # metrics
    dates_arr = pd.to_datetime(equity_curve["date"]).to_numpy(dtype="datetime64[D]")
    prices_arr = equity_curve["equity"].to_numpy(dtype=float)
    cfg = WindowConfig(trading_days_per_year=243)
    mh = compute_holding_period_metrics(dates_arr, prices_arr, config=cfg)

    first_eq = float(equity_curve["equity"].iloc[0])
    last_eq = float(equity_curve["equity"].iloc[-1])
    total_ret = (last_eq / first_eq - 1.0) if first_eq > 0 else 0.0

    # summary.csv
    rows: list[dict[str, str | float]] = [
        {"section": "config", "name": "组合", "value": portfolio_str},
        {"section": "config", "name": "起始日期", "value": start_date},
        {"section": "config", "name": "结束日期", "value": end_date},
        {"section": "config", "name": "再平衡间隔", "value": f"{rebalance_interval} 交易日"},
        {"section": "config", "name": "初始资金", "value": str(initial_capital)},
        {"section": "metrics_pybroker", "name": "初始市值", "value": first_eq},
        {"section": "metrics_pybroker", "name": "期末市值", "value": last_eq},
        {"section": "metrics_pybroker", "name": "总收益率", "value": total_ret},
    ]
    for k, v in mh.items():
        if v is not None:
            val = round(v, 6) if isinstance(v, float) else v
            rows.append({"section": "metrics_holding", "name": k, "value": val})

    pd.DataFrame(rows, columns=["section", "name", "value"]).to_csv(
        output_dir / "summary.csv", index=False, encoding="utf-8-sig",
    )
    logger.info("summary -> %s", output_dir / "summary.csv")

    # backtest_report.md
    lines = [
        "# 固定组合回测报告", "",
        "## 组合配置",
        f"- 成分: {portfolio_str}",
        f"- 日期范围: {start_date} ~ {end_date}",
        f"- 再平衡间隔: {rebalance_interval} 交易日",
        f"- 初始资金: {initial_capital:,.2f}", "",
        "## 回测结果",
        f"- 期初市值: {first_eq:,.2f}",
        f"- 期末市值: {last_eq:,.2f}",
        f"- 总收益率: {total_ret:.2%}",
    ]
    for label, key in [
        ("年化收益率", "年化收益率"), ("夏普比率", "夏普比率"),
        ("最大回撤率", "最大回撤率"), ("年化波动率", "年化波动率"),
    ]:
        val = mh.get(key)
        if isinstance(val, float):
            fmt = f"{val:.4f}" if "比率" in label else f"{val:.2%}"
            lines.append(f"- {label}: {fmt}")
    lines.append("")

    (output_dir / "backtest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("backtest_report -> %s", output_dir / "backtest_report.md")

    # backtest_curves.html
    _write_curves_html(equity_curve.copy(), output_dir, portfolio_str)


def _write_curves_html(
    equity_curve: pd.DataFrame, output_dir: Path, title: str,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly 未安装，跳过 backtest_curves.html")
        return
    if equity_curve.empty or len(equity_curve) < 2:
        return

    dates = pd.to_datetime(equity_curve["date"], errors="coerce")
    cum_ret = equity_curve["cumulative_return"].values * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=cum_ret, name="组合",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.update_layout(
        title_text=f"固定组合净值曲线 — {title}",
        xaxis_title="日期", yaxis_title="累计收益率 (%)",
        height=500, showlegend=True,
    )
    path = output_dir / "backtest_curves.html"
    fig.write_html(str(path), config={"displayModeBar": True})
    logger.info("backtest_curves -> %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="固定基金组合净值计算（被动比较基准回测）",
    )
    parser.add_argument("--fund-etl-dir", required=True, type=Path,
                        help="fund_etl 数据目录")
    parser.add_argument("--start-date", required=True,
                        help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", required=True,
                        help="结束日期 YYYY-MM-DD")
    parser.add_argument("--rebalance-interval", required=True, type=int,
                        help="再平衡间隔交易日数")
    parser.add_argument("--portfolio", required=True,
                        help="组合定义 code:weight,... 如 161119:0.70,510300:0.30")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="产物输出目录")
    parser.add_argument("--initial-capital", type=float,
                        default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--integrity-threshold", type=float,
                        default=DEFAULT_INTEGRITY_THRESHOLD,
                        help="数据完整性最低比例（默认 0.95）")
    parser.add_argument("--compare-threshold", type=float,
                        default=DEFAULT_COMPARE_THRESHOLD,
                        help="收益率比对 <1%%偏差占比 最低比例（默认 0.80）")
    parser.add_argument("--trade-dates-csv", type=Path,
                        default=DEFAULT_TRADE_DATES_CSV,
                        help="交易日历 CSV 路径")
    args = parser.parse_args()

    fund_etl_dir = args.fund_etl_dir.resolve()
    if not fund_etl_dir.is_dir():
        print(f"[ERROR] fund_etl_dir 不存在: {fund_etl_dir}")
        sys.exit(1)

    portfolio = parse_portfolio(args.portfolio)
    fund_codes = list(dict.fromkeys(c for c, _ in portfolio))

    print(f"[benchmark] 组合: {[(c, f'{w:.2%}') for c, w in portfolio]}")
    print(f"[benchmark] 日期: {args.start_date} ~ {args.end_date}")
    print(f"[benchmark] 再平衡: 每 {args.rebalance_interval} 交易日")

    # --- Phase 1: integrity ---
    integrity_errs = validate_integrity(
        fund_etl_dir, fund_codes,
        args.start_date, args.end_date,
        args.integrity_threshold, args.trade_dates_csv,
    )
    if integrity_errs:
        print("[FAIL] 数据完整性校验未通过:")
        for e in integrity_errs:
            print(f"  - {e}")
        sys.exit(1)
    print("[PASS] 数据完整性校验通过")

    # --- Phase 2: compare (degrade if no cum_return) ---
    compare_errs = validate_compare(
        fund_etl_dir, fund_codes,
        args.start_date, args.end_date,
        args.compare_threshold,
    )
    if compare_errs:
        print("[FAIL] 收益率比对校验未通过:")
        for e in compare_errs:
            print(f"  - {e}")
        sys.exit(1)
    print("[PASS] 收益率比对校验通过（或降级跳过）")

    # --- Phase 3: simulate ---
    nav_data = load_adjusted_nav(
        fund_etl_dir, fund_codes, args.start_date, args.end_date,
    )
    trade_dates = load_trade_calendar(
        args.trade_dates_csv, args.start_date, args.end_date,
    )
    equity_curve = simulate_portfolio(
        nav_data, portfolio, trade_dates,
        args.rebalance_interval, args.initial_capital,
    )

    # --- Phase 4: write ---
    write_outputs(
        equity_curve, portfolio, args.output_dir,
        args.start_date, args.end_date,
        args.rebalance_interval, args.initial_capital,
    )
    print(f"[benchmark] 完成 -> {args.output_dir}")


if __name__ == "__main__":
    main()
