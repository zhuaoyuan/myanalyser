#!/usr/bin/env python3
"""多 T 调仓链式模拟：用前 T 期末市值作为后 T 期初市值，串联回测长期效果。

前提条件：前 T 期末日期 <= 后 T 期初日期。无交易的 T 跳过。
输出：与单 T 回测相同格式（summary.csv、equity_curve.csv、period_detail.csv、orders.csv、positions_flat.csv、backtest_report.md）

用法:
  python myanalyser/tools/v2/chain_multi_t_backtest.py \\
    --output-root myanalyser/artifacts/backtest_multi/RUN_ID/RULESET_VERSION \\
    [--chain-output-dir chain]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_MYANALYSER_ROOT = _SCRIPT_DIR.parent.parent
_SRC = _MYANALYSER_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

logger = logging.getLogger(__name__)


def _read_summary_metrics(summary_csv: Path) -> dict[str, float | str]:
    """从 summary.csv 提取 config 与 metrics_pybroker 的键值。"""
    df = pd.read_csv(summary_csv, dtype=str, encoding="utf-8-sig")
    out: dict[str, float | str] = {}
    for _, row in df.iterrows():
        sec = str(row.get("section", "")).strip()
        name = str(row.get("name", "")).strip()
        val = str(row.get("value", "")).strip()
        if not name:
            continue
        if sec == "config":
            out[name] = val
        elif sec == "metrics_pybroker":
            try:
                out[name] = float(val)
            except ValueError:
                out[name] = val
    return out


def _has_buy_orders(orders_csv: Path) -> bool:
    """orders.csv 是否包含买入。"""
    if not orders_csv.exists():
        return False
    df = pd.read_csv(orders_csv, dtype=str, encoding="utf-8-sig")
    if df.empty:
        return False
    if "type" not in df.columns:
        return False
    buys = df[df["type"].astype(str).str.strip().str.lower() == "buy"]
    return not buys.empty


def _list_t_dirs(output_root: Path) -> list[tuple[Path, str]]:
    """扫描日期子目录 YYYY-MM-DD，返回 (dir_path, as_of_str) 按日期排序。"""
    date_dirs = sorted(
        d for d in output_root.iterdir()
        if d.is_dir() and len(d.name) == 10 and d.name[4] == "-" and d.name[7] == "-"
    )
    return [(d, d.name) for d in date_dirs]


def _write_chain_curves_html(equity_curve: pd.DataFrame, output_dir: Path) -> Path | None:
    """生成仅含组合累计收益率的 Plotly HTML 曲线。"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if equity_curve.empty or len(equity_curve) < 2:
        return None

    dates = pd.to_datetime(equity_curve["date"], errors="coerce")
    cum_ret = equity_curve["cumulative_return"].values * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cum_ret,
            name="组合",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.update_layout(
        title_text="链式调仓组合净值曲线",
        xaxis_title="日期",
        yaxis_title="累计收益率 (%)",
        height=500,
        showlegend=True,
    )
    path = output_dir / "backtest_curves.html"
    fig.write_html(str(path), config={"displayModeBar": True})
    return path


def chain(output_root: Path, chain_output_dir: str = "chain") -> Path:
    """执行链式模拟，写入 chain 子目录。"""
    output_root = Path(output_root).resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"output_root 不存在: {output_root}")

    t_list = _list_t_dirs(output_root)
    if not t_list:
        raise FileNotFoundError(f"未找到日期子目录: {output_root}")

    # 加载每个 T 的元数据，筛选有交易且满足日期条件的
    rows: list[dict] = []
    for t_dir, as_of_str in t_list:
        summary_csv = t_dir / "summary.csv"
        orders_csv = t_dir / "orders.csv"
        equity_csv = t_dir / "equity_curve.csv"

        if not summary_csv.exists():
            logger.warning("[%s] 缺少 summary.csv，跳过", as_of_str)
            continue
        if not _has_buy_orders(orders_csv):
            logger.info("[%s] 无买入，跳过", as_of_str)
            continue
        if not equity_csv.exists():
            logger.warning("[%s] 缺少 equity_curve.csv，跳过", as_of_str)
            continue

        metrics = _read_summary_metrics(summary_csv)
        start_str = metrics.get("起始日期", "")
        end_str = metrics.get("结束日期", "")
        if not start_str or not end_str:
            logger.warning("[%s] 无法解析起始/结束日期，跳过", as_of_str)
            continue

        initial_mv = metrics.get("初始市值")
        end_mv = metrics.get("期末市值")
        if initial_mv is None or end_mv is None:
            logger.warning("[%s] 缺少 初始市值 或 期末市值，跳过", as_of_str)
            continue

        try:
            initial_mv = float(initial_mv)
            end_mv = float(end_mv)
        except (TypeError, ValueError):
            logger.warning("[%s] 初始/期末市值非数值，跳过", as_of_str)
            continue

        start_ts = pd.to_datetime(start_str).normalize()
        end_ts = pd.to_datetime(end_str).normalize()

        rows.append({
            "t_dir": t_dir,
            "as_of_str": as_of_str,
            "start_str": start_str,
            "end_str": end_str,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "initial_mv": initial_mv,
            "end_mv": end_mv,
            "metrics": metrics,
        })

    if not rows:
        raise ValueError("无有效 T 日可链式（均有交易且满足日期条件）")

    # 按日期排序，应用链式条件：prev_end <= curr_start
    valid: list[dict] = []
    prev_end_mv: float | None = None
    prev_end_ts: pd.Timestamp | None = None

    for r in rows:
        if prev_end_mv is None and prev_end_ts is None:
            # 首个 T，直接加入
            valid.append(r)
            prev_end_mv = r["end_mv"]
            prev_end_ts = r["end_ts"]
            continue

        if prev_end_ts is not None and prev_end_ts > r["start_ts"]:
            logger.info("[%s] 前 T 期末 %s > 本期初 %s，不链式，跳过", r["as_of_str"], prev_end_ts.date(), r["start_ts"].date())
            continue

        # prev_end_ts <= r["start_ts"]：用前 T 期末市值作为本期初，缩放比例
        scale = prev_end_mv / r["initial_mv"] if r["initial_mv"] > 0 else 1.0
        r["scale"] = scale
        valid.append(r)
        prev_end_mv = r["end_mv"] * scale
        prev_end_ts = r["end_ts"]

    if not valid:
        raise ValueError("无满足链式条件的 T")

    # 输出目录
    out_dir = output_root / chain_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 合并 equity_curve
    equity_parts: list[pd.DataFrame] = []
    for i, r in enumerate(valid):
        eq_df = pd.read_csv(r["t_dir"] / "equity_curve.csv", dtype={"date": str, "equity": float}, encoding="utf-8-sig")
        if eq_df.empty:
            continue
        eq_df["date"] = pd.to_datetime(eq_df["date"], errors="coerce")
        scale = r.get("scale", 1.0)
        eq_df["equity"] = eq_df["equity"] * scale
        base = float(eq_df["equity"].iloc[0]) if eq_df["equity"].iloc[0] > 0 else 1.0
        eq_df["cumulative_return"] = eq_df["equity"] / base - 1.0
        equity_parts.append(eq_df)

    if not equity_parts:
        raise ValueError("无有效 equity 数据")

    # 合并去重：同日期取最后一个 T 的值（新仓位）
    merged = pd.concat(equity_parts, ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    # 重新计算 cumulative_return：以首日为 1
    first_eq = float(merged["equity"].iloc[0])
    base_eq = first_eq if first_eq > 0 else 1.0
    merged["cumulative_return"] = merged["equity"] / base_eq - 1.0

    # 1.1 组合净值曲线 HTML（仅组合，无成分基金）
    curves_path = _write_chain_curves_html(merged.copy(), out_dir)
    if curves_path is not None:
        logger.info("[chain] backtest_curves.html -> %s", curves_path)

    equity_path = out_dir / "equity_curve.csv"
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    merged.to_csv(equity_path, index=False, encoding="utf-8-sig")
    logger.info("[chain] equity_curve -> %s (%d rows)", equity_path, len(merged))

    # 2. 合并 orders
    orders_parts: list[pd.DataFrame] = []
    for r in valid:
        fp = r["t_dir"] / "orders.csv"
        if fp.exists():
            df = pd.read_csv(fp, encoding="utf-8-sig")
            if not df.empty:
                orders_parts.append(df)
    orders_df = pd.concat(orders_parts, ignore_index=True) if orders_parts else pd.DataFrame()
    if not orders_df.empty and "fill_date" in orders_df.columns:
        orders_df = orders_df.sort_values("fill_date").reset_index(drop=True)
    orders_path = out_dir / "orders.csv"
    orders_df.to_csv(orders_path, index=False, encoding="utf-8-sig")
    logger.info("[chain] orders -> %s (%d rows)", orders_path, len(orders_df))

    # 3. 合并 period_detail
    detail_parts: list[pd.DataFrame] = []
    for r in valid:
        fp = r["t_dir"] / "period_detail.csv"
        if fp.exists():
            df = pd.read_csv(fp, encoding="utf-8-sig")
            if not df.empty:
                detail_parts.append(df)
    detail_df = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
    if not detail_df.empty and "stat_date" in detail_df.columns:
        detail_df = detail_df.sort_values("stat_date").reset_index(drop=True)
    detail_path = out_dir / "period_detail.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    logger.info("[chain] period_detail -> %s (%d rows)", detail_path, len(detail_df))

    # 4. 合并 positions_flat
    pos_parts: list[pd.DataFrame] = []
    for r in valid:
        fp = r["t_dir"] / "positions_flat.csv"
        if fp.exists():
            df = pd.read_csv(fp, encoding="utf-8-sig")
            if not df.empty:
                pos_parts.append(df)
    pos_df = pd.concat(pos_parts, ignore_index=True) if pos_parts else pd.DataFrame()
    if not pos_df.empty and "stat_date" in pos_df.columns:
        pos_df = pos_df.sort_values(["stat_date", "rank"]).reset_index(drop=True)
    pos_path = out_dir / "positions_flat.csv"
    pos_df.to_csv(pos_path, index=False, encoding="utf-8-sig")
    logger.info("[chain] positions_flat -> %s (%d rows)", pos_path, len(pos_df))

    # 5. 生成 summary.csv
    first_r = valid[0]["metrics"]
    chain_start = valid[0]["start_str"]
    chain_end = valid[-1]["end_str"]

    # config 白名单（multi_t_backtest 写入的 run_config）
    _CONFIG_KEYS = (
        "策略", "调仓周期", "持仓数量", "预热天数", "初始资金", "净值目录", "最大基金数",
        "run_id", "ruleset_version", "filter_start", "filter_end",
    )
    summary_rows: list[dict[str, str | float]] = []
    for k in _CONFIG_KEYS:
        v = first_r.get(k)
        if v is not None:
            summary_rows.append({"section": "config", "name": k, "value": str(v)})

    # metrics_holding：基于链式 equity 计算
    try:
        from fund_metrics_core import compute_holding_period_metrics, WindowConfig
    except ImportError:
        compute_holding_period_metrics = None
        WindowConfig = None

    summary_rows.append({"section": "config", "name": "起始日期", "value": chain_start})
    summary_rows.append({"section": "config", "name": "结束日期", "value": chain_end})
    summary_rows.append({"section": "config", "name": "chain_mode", "value": "链式调仓（前T期末->后T期初）"})
    for env_name in ("FUND_BACKTEST_FILTERS", "FILTERED_FUND_CANDIDATES_CSV", "FUND_BACKTEST_MAX_FUNDS"):
        val = os.environ.get(env_name, "")
        summary_rows.append({"section": "env", "name": env_name, "value": val if val else "(未设置)"})
    summary_rows.append({"section": "data", "name": "基金数量", "value": str(len(valid))})
    summary_rows.append({"section": "data", "name": "日期范围", "value": f"{chain_start} ~ {chain_end}"})

    # 链式 metrics_holding
    if compute_holding_period_metrics and WindowConfig and not merged.empty:
        merged_dates = pd.to_datetime(merged["date"], errors="coerce")
        merged_holding = merged[merged_dates >= pd.to_datetime(chain_start)]
        if len(merged_holding) >= 2:
            dates_arr = merged_holding["date"].to_numpy(dtype="datetime64[D]")
            prices = merged_holding["equity"].to_numpy(dtype=float)
            cfg = WindowConfig(trading_days_per_year=243)
            mh = compute_holding_period_metrics(dates_arr, prices, config=cfg)
            for k, v in mh.items():
                if v is not None:
                    summary_rows.append({"section": "metrics_holding", "name": k, "value": round(v, 6) if isinstance(v, float) else v})

    # metrics_pybroker 简化版
    first_eq = float(merged["equity"].iloc[0])
    last_eq = float(merged["equity"].iloc[-1])
    total_ret = (last_eq / first_eq - 1.0) if first_eq > 0 else 0.0
    summary_rows.append({"section": "metrics_pybroker", "name": "初始市值", "value": first_eq})
    summary_rows.append({"section": "metrics_pybroker", "name": "期末市值", "value": last_eq})
    summary_rows.append({"section": "metrics_pybroker", "name": "总收益率", "value": total_ret})

    summary_df = pd.DataFrame(summary_rows, columns=["section", "name", "value"])
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info("[chain] summary -> %s", summary_path)

    # 6. backtest_report.md
    report_lines = [
        "# 链式调仓回测报告",
        "",
        "## 说明",
        "多 T 调仓链式模拟：用前 T 期末市值作为后 T 期初市值，串联长期效果。",
        f"- 链式 T 数: {len(valid)}",
        f"- 日期范围: {chain_start} ~ {chain_end}",
        f"- 期初市值: {first_eq:.2f}",
        f"- 期末市值: {last_eq:.2f}",
        f"- 总收益率: {total_ret:.2%}",
        "",
        "## 输出文件",
        "- summary.csv",
        "- equity_curve.csv",
        "- period_detail.csv",
        "- orders.csv",
        "- positions_flat.csv",
        "- backtest_curves.html（组合净值曲线）",
        "",
    ]
    report_path = out_dir / "backtest_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    logger.info("[chain] backtest_report -> %s", report_path)

    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="多 T 调仓链式模拟：前 T 期末市值作为后 T 期初，输出与单 T 相同格式"
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="multi_t_backtest 产物根目录，如 artifacts/backtest_multi/RUN_ID/RULESET_VERSION",
    )
    parser.add_argument(
        "--chain-output-dir",
        default="chain",
        help="链式结果子目录名，默认 chain",
    )
    args = parser.parse_args()
    out_dir = chain(args.output_root, args.chain_output_dir)
    print(f"链式模拟完成 -> {out_dir}")


if __name__ == "__main__":
    main()
