#!/usr/bin/env python3
"""回测曲线对比：合并多组回测/基准目录，输出交集时间范围内的曲线对比图与 summary 对比表。

支持两类输入目录（均可多选）：
1. 回测结果目录：如 myanalyser/artifacts/backtest_multi/.../chain
2. 基准目录：如 myanalyser/artifacts/backtest_base/.../807200_申万债券、myanalyser/artifacts/backtest_base/.../保守型_A

各目录需含 equity_curve.csv（date, equity, cumulative_return）。

输出单一 HTML（backtest_curves.html），含 Summary 对比表（14 项指标）与 Plotly 净值曲线图。

用法:
  python myanalyser/tools/compare_backtest_curves.py \
    --backtest-dir myanalyser/artifacts/backtest_multi/20260315_123456_full_run_v2/20260316_2m/chain \
    --base-dir myanalyser/artifacts/backtest_base/20260315_123456_fund_index/807200_申万债券 \
    --base-dir myanalyser/artifacts/backtest_base/20260315_123456_full_run_v2/保守型_A \
    --output-dir myanalyser/artifacts/backtest_base/20260315_123456_full_run_v2/保守型_B
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_MYANALYSER_ROOT = _SCRIPT_DIR.parent
_SRC = _MYANALYSER_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fund_metrics_core import HOLDING_METRIC_NAMES, WindowConfig, compute_holding_period_metrics

logger = logging.getLogger(__name__)

# Summary 指标（排除标准误差），顺序与 fund_metrics_core 一致
SUMMARY_METRIC_KEYS = [k for k in HOLDING_METRIC_NAMES if k != "标准误差"]

# 指标方向：True=越大越好，False=越小越好
_METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "年化收益率": True,
    "夏普比率": True,
    "索提诺比率": True,
    "卡玛比率": True,
    "盈利因子": True,
    "溃疡指数": False,
    "溃疡绩效指数": True,
    "净值R方": True,
    "上涨星期比例": True,
    "上涨月份比例": True,
    "最大回撤率": False,
    "最长回撤修复天数": False,
    "年化波动率": False,
}

DEFAULT_TRADING_DAYS_PER_YEAR = 243

# Plotly 曲线颜色
_CURVE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _load_equity_curve(dir_path: Path) -> None | pd.DataFrame:
    """加载 equity_curve.csv，返回 (date, equity, cumulative_return) 的 DataFrame。"""
    csv_path = dir_path / "equity_curve.csv"
    if not csv_path.exists():
        logger.warning("缺少 equity_curve.csv: %s", dir_path)
        return None
    try:
        df = pd.read_csv(csv_path, dtype={"date": str, "equity": float}, encoding="utf-8-sig")
    except Exception as e:
        logger.warning("读取 equity_curve.csv 失败 %s: %s", dir_path, e)
        return None
    if df.empty or len(df) < 2:
        logger.warning("equity_curve 为空或不足 2 行: %s", dir_path)
        return None
    if "date" not in df.columns or "equity" not in df.columns:
        logger.warning("equity_curve 缺少 date/equity 列: %s", dir_path)
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "equity"])
    if df.empty or len(df) < 2:
        return None
    if "cumulative_return" not in df.columns:
        base = float(df["equity"].iloc[0])
        base = base if base > 0 and not np.isnan(base) else 1.0
        df["cumulative_return"] = df["equity"] / base - 1.0
    return df.sort_values("date").reset_index(drop=True)


def _get_label(dir_path: Path) -> str:
    """从目录路径提取标签名（如 chain、807200_申万债券、保守型_A）。"""
    return dir_path.name


def _compute_intersection_dates(curves: list[tuple[str, pd.DataFrame]]) -> pd.DatetimeIndex | None:
    """计算时间范围：起止对齐（取各曲线起止的交集区间），过程中间日期取并集。

    即 [max(各曲线起始), min(各曲线结束)] 内的所有日期（任一曲线有值即保留），
    某组合在某日无值时用前向填充，不影响其他组合展示。
    """
    if not curves:
        return None
    min_dates = [df["date"].min() for _, df in curves]
    max_dates = [df["date"].max() for _, df in curves]
    range_start = max(min_dates)
    range_end = min(max_dates)
    if range_start > range_end:
        return None
    # 并集：范围内任一曲线有值的日期
    all_in_range: set = set()
    for _, df in curves:
        mask = (df["date"] >= range_start) & (df["date"] <= range_end)
        all_in_range.update(df.loc[mask, "date"].dropna().unique().tolist())
    if not all_in_range:
        return None
    return pd.DatetimeIndex(sorted(all_in_range))


def _reindex_to_dates(df: pd.DataFrame, target_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """将 equity 按 target_dates 对齐，缺失日期前向填充。"""
    df = df.set_index("date").sort_index()
    reindexed = df.reindex(target_dates, method="ffill")
    # 丢弃首段无数据的行（交集首日之前该曲线无值）
    reindexed = reindexed.dropna(subset=["equity"]).dropna(how="all")
    return reindexed


def _compute_metrics_on_curve(df: pd.DataFrame, config: WindowConfig) -> dict[str, str | float | None]:
    """基于交集区间内的 equity 计算 metrics_holding 全部指标。"""
    if df.empty or len(df) < 2:
        return {k: None for k in SUMMARY_METRIC_KEYS}
    dates_arr = df.index.to_numpy(dtype="datetime64[D]")
    prices = df["equity"].to_numpy(dtype=float)
    out = compute_holding_period_metrics(dates_arr, prices, config=config)
    result: dict[str, str | float | None] = {}
    for k in SUMMARY_METRIC_KEYS:
        v = out.get(k)
        if v is not None:
            if isinstance(v, float):
                result[k] = round(v, 6)
            else:
                result[k] = v
        else:
            result[k] = None
    return result


def _format_metric_value(v: str | float | None) -> str:
    """格式化指标显示（非百分比列）。"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return f"{v:.4f}"
    return str(v)


def _to_float_or_none(v: str | float | None) -> float | None:
    """可转为数值则返回 float，否则 None。"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _build_summary_table_html(summary_rows: list[dict]) -> str:
    """构建 Summary 表 HTML，最佳标红、最差标绿。"""
    if not summary_rows:
        return ""
    df = pd.DataFrame(summary_rows)
    pct_cols = {"年化收益率", "最大回撤率", "年化波动率", "上涨星期比例", "上涨月份比例"}
    non_metric_cols = {"name", "起始日期", "结束日期"}

    # 确定各列 best/worst 行索引
    best_idx: dict[str, int] = {}
    worst_idx: dict[str, int] = {}
    for col in df.columns:
        if col in non_metric_cols:
            continue
        higher = _METRIC_HIGHER_IS_BETTER.get(col, True)
        vals: list[tuple[int, float]] = []
        for i, v in enumerate(df[col]):
            fv = _to_float_or_none(v)
            if fv is not None:
                vals.append((i, fv))
        if len(vals) < 2:
            continue
        sorted_vals = sorted(vals, key=lambda x: x[1], reverse=higher)
        # higher=True: 排序后 [大,...,小]，best=首，worst=末
        # higher=False: 排序后 [小,...,大]，best=末（最大），worst=首（最小，如最大回撤率 -15% 最差）
        best_idx[col] = sorted_vals[0][0] if higher else sorted_vals[-1][0]
        worst_idx[col] = sorted_vals[-1][0] if higher else sorted_vals[0][0]
        if best_idx[col] == worst_idx[col]:
            del worst_idx[col]  # 全部相同则不标绿

    # 构建 HTML
    lines: list[str] = []
    lines.append('<table border="1" class="compare-summary">')
    lines.append("<thead><tr>")
    for col in df.columns:
        lines.append(f"<th>{col}</th>")
    lines.append("</tr></thead><tbody>")
    for i, row in df.iterrows():
        lines.append("<tr>")
        for col in df.columns:
            v = row[col]
            if col in pct_cols:
                disp = f"{v:.2%}" if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)) else str(v) if v is not None else ""
            else:
                disp = _format_metric_value(v)
            style = ""
            if col in best_idx and best_idx[col] == i:
                style = ' style="color:red;font-weight:bold"'
            elif col in worst_idx and worst_idx[col] == i:
                style = ' style="color:green;font-weight:bold"'
            lines.append(f"<td{style}>{disp}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _write_unified_html(
    curves: list[tuple[str, pd.DataFrame]],
    summary_rows: list[dict],
    output_path: Path,
    title: str = "回测曲线对比",
) -> None:
    """生成单一 HTML：Summary 对比表 + Plotly 曲线图。"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly 未安装，无法生成 HTML")
        return
    if not curves:
        return

    # 1. Plotly 曲线图
    fig = go.Figure()
    for i, (label, df) in enumerate(curves):
        dates = df.index
        cum_ret = df["cumulative_return"].values * 100
        color = _CURVE_COLORS[i % len(_CURVE_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cum_ret,
                name=label,
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(
        title_text=title,
        xaxis_title="日期",
        yaxis_title="累计收益率 (%)",
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    plot_html = fig.to_html(full_html=True, config={"displayModeBar": True})

    # 2. Summary 对比表（最佳标红、最差标绿）
    table_html = _build_summary_table_html(summary_rows)

    # 3. 合并：在 <body> 后插入 Summary 表 + 净值曲线标题，Plotly div 紧随其后
    body_start = plot_html.find("<body>")
    if body_start >= 0:
        insert_pos = body_start + len("<body>")
        extra = f'\n<div style="margin:1em 0;"><h2>Summary 对比表</h2>{table_html}</div>\n<div style="margin:1em 0;"><h2>净值曲线</h2>\n'
        plot_html = plot_html[:insert_pos] + extra + plot_html[insert_pos:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(plot_html, encoding="utf-8")
    logger.info("backtest_curves.html -> %s", output_path)


def run(
    backtest_dirs: list[Path],
    base_dirs: list[Path],
    output_dir: Path,
    trading_days_per_year: int = DEFAULT_TRADING_DAYS_PER_YEAR,
) -> dict[str, Path]:
    """执行对比：加载曲线、取交集、生成 HTML 与 summary。"""
    all_dirs: list[Path] = []
    for d in backtest_dirs + base_dirs:
        p = Path(d).resolve()
        if not p.is_dir():
            logger.warning("目录不存在，跳过: %s", p)
            continue
        all_dirs.append(p)

    if not all_dirs:
        raise ValueError("无有效输入目录")

    curves_raw: list[tuple[str, pd.DataFrame]] = []
    for d in all_dirs:
        df = _load_equity_curve(d)
        if df is not None:
            label = _get_label(d)
            curves_raw.append((label, df))

    if not curves_raw:
        raise ValueError("无有效 equity_curve 数据")

    # 日期交集
    common_dates = _compute_intersection_dates(curves_raw)
    if common_dates is None or len(common_dates) < 2:
        raise ValueError("各输入曲线日期无交集或交集不足 2 交易日")

    logger.info("交集日期范围: %s ~ %s (%d 交易日)", common_dates.min(), common_dates.max(), len(common_dates))

    config = WindowConfig(trading_days_per_year=trading_days_per_year)

    # 对齐到交集区间
    curves_aligned: list[tuple[str, pd.DataFrame]] = []
    summary_rows: list[dict] = []

    for label, df in curves_raw:
        reindexed = _reindex_to_dates(df, common_dates)
        if reindexed.empty or len(reindexed) < 2:
            logger.warning("对齐后数据不足，跳过: %s", label)
            continue
        # 首日归一化 cumulative_return
        base_eq = float(reindexed["equity"].iloc[0])
        base_eq = base_eq if base_eq > 0 else 1.0
        reindexed["cumulative_return"] = reindexed["equity"] / base_eq - 1.0
        curves_aligned.append((label, reindexed))

        metrics = _compute_metrics_on_curve(reindexed, config)
        row: dict = {"name": label, "起始日期": str(common_dates.min().date()), "结束日期": str(common_dates.max().date())}
        row.update(metrics)
        summary_rows.append(row)

    if not curves_aligned:
        raise ValueError("对齐后无有效曲线")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "backtest_curves.html"
    _write_unified_html(
        curves_aligned,
        summary_rows,
        output_path,
        title="回测曲线对比（交集时间范围）",
    )

    return {"output": output_path}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="回测曲线对比：合并多组回测/基准目录，输出交集时间范围内的曲线对比图与 summary 对比表",
    )
    parser.add_argument(
        "--backtest-dir",
        action="append",
        default=[],
        type=Path,
        dest="backtest_dirs",
        help="回测结果目录（可多次指定）",
    )
    parser.add_argument(
        "--base-dir",
        action="append",
        default=[],
        type=Path,
        dest="base_dirs",
        help="基准目录（可多次指定）",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="输出目录（生成 backtest_curves.html，含 Summary 对比表与曲线图）",
    )
    parser.add_argument(
        "--trading-days-per-year",
        type=int,
        default=DEFAULT_TRADING_DAYS_PER_YEAR,
        help=f"年化交易日数（默认 {DEFAULT_TRADING_DAYS_PER_YEAR}）",
    )
    args = parser.parse_args()

    all_dirs = args.backtest_dirs + args.base_dirs
    if not all_dirs:
        print("请至少指定一个 --backtest-dir 或 --base-dir")
        sys.exit(1)

    try:
        out = run(
            backtest_dirs=args.backtest_dirs,
            base_dirs=args.base_dirs,
            output_dir=args.output_dir,
            trading_days_per_year=args.trading_days_per_year,
        )
        print(f"\n输出:")
        for k, p in out.items():
            print(f"  {k}: {p}")
    except Exception as e:
        logger.exception("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
