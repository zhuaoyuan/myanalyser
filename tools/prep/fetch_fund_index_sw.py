#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
爬取申万宏源基金指数历史数据（日频）。

背景说明：
  Wind 885001（偏股混合型基金指数）、885003（偏债混合型基金指数）为万得专有指数，
  官方数据需通过 Wind 终端/数据库获取，目前未发现公开免费且天级更新的数据源。

  本脚本使用 AKShare 获取申万宏源基金指数，作为概念相近的公开替代：
  - 807100：申万宏源权益基金指数（偏股，与 Wind 885001 概念相近）
  - 807200：申万宏源债券基金指数（偏债，与 Wind 885003 概念相近）
  - 807300：申万宏源混合基金指数（混合）

  数据来源：申万宏源研究 https://www.swsresearch.com/institute_sw/allIndex/releasedIndex
  更新频率：交易日日频，与 Wind 指数类似。

样例：
python myanalyser/tools/prep/fetch_fund_index_sw.py \
  --start-date 2015-02-27 --end-date 2026-03-17 \
  --run-id 20260315_123456_fund_index

产物目录结构（对标 backtest_base/保守型_A）：
  output_root/{run_id}/{指数子目录}/
    equity_curve.csv
    summary.csv
    backtest_report.md
    backtest_curves.html
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import akshare as ak
except ImportError as e:
    raise RuntimeError("需要安装 akshare: pip install akshare") from e

# 默认请求间隔（秒），降低限流风险
DEFAULT_REQUEST_DELAY = 0.5

# 默认爬取的指数：申万基金指数（与 Wind 885001/885003 概念相近的公开替代）
# (code, name, subdir_short_name)
DEFAULT_INDICES = [
    ("807100", "申万宏源权益基金指数", "申万权益"),
    ("807200", "申万宏源债券基金指数", "申万债券"),
    ("807300", "申万宏源混合基金指数", "申万混合"),
]

LOG = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_MYANALYSER_ROOT = _SCRIPT_DIR.parent.parent
_SRC = _MYANALYSER_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
DEFAULT_OUTPUT_ROOT = _MYANALYSER_ROOT / "artifacts" / "backtest_base"
DEFAULT_END = datetime.date.today().isoformat()


def fetch_index_hist(symbol: str, period: str = "day") -> pd.DataFrame:
    """调用 AKShare 获取单只指数历史行情。"""
    return ak.index_hist_fund_sw(symbol=symbol, period=period)


def _df_to_equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    """将指数 close 转为 equity_curve 格式：date, equity, cumulative_return。"""
    close = df["close"].astype(float)
    base = float(close.iloc[0])
    equity = (close / base) if base > 0 else close
    cum_ret = equity - 1.0
    return pd.DataFrame({
        "date": df["date"].astype(str),
        "equity": equity.values,
        "cumulative_return": cum_ret.values,
    })


def _write_index_outputs(
    df: pd.DataFrame,
    output_dir: Path,
    code: str,
    name: str,
    start_date: str,
    end_date: str,
) -> None:
    """写入与 backtest_base 一致的四类产物：equity_curve.csv / summary.csv / backtest_report.md / backtest_curves.html。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    equity_curve = _df_to_equity_curve(df)

    # equity_curve.csv
    equity_curve.to_csv(output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    LOG.info("equity_curve -> %s (%d rows)", output_dir / "equity_curve.csv", len(equity_curve))

    # metrics
    try:
        from fund_metrics_core import WindowConfig, compute_holding_period_metrics
        dates_arr = pd.to_datetime(equity_curve["date"]).to_numpy(dtype="datetime64[D]")
        prices_arr = equity_curve["equity"].to_numpy(dtype=float)
        cfg = WindowConfig(trading_days_per_year=243)
        mh = compute_holding_period_metrics(dates_arr, prices_arr, config=cfg)
    except ImportError:
        mh = {}

    first_eq = float(equity_curve["equity"].iloc[0])
    last_eq = float(equity_curve["equity"].iloc[-1])
    total_ret = (last_eq / first_eq - 1.0) if first_eq > 0 else 0.0

    # summary.csv
    rows: list[dict[str, str | float]] = [
        {"section": "config", "name": "指数", "value": f"{code} {name}"},
        {"section": "config", "name": "起始日期", "value": start_date},
        {"section": "config", "name": "结束日期", "value": end_date},
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

    # backtest_report.md
    lines = [
        "# 基金指数行情报告", "",
        f"## 指数: {code} {name}",
        f"- 日期范围: {start_date} ~ {end_date}",
        f"- 期初净值: {first_eq:.6f}",
        f"- 期末净值: {last_eq:.6f}",
        f"- 总收益率: {total_ret:.2%}", "",
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

    # backtest_curves.html
    try:
        import plotly.graph_objects as go
        dates = pd.to_datetime(equity_curve["date"], errors="coerce")
        cum_ret = equity_curve["cumulative_return"].values * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=cum_ret, name=name,
            line=dict(color="#1f77b4", width=2),
        ))
        fig.update_layout(
            title_text=f"基金指数累计收益率 — {code} {name}",
            xaxis_title="日期", yaxis_title="累计收益率 (%)",
            height=500, showlegend=True,
        )
        fig.write_html(str(output_dir / "backtest_curves.html"), config={"displayModeBar": True})
    except ImportError:
        LOG.warning("plotly 未安装，跳过 backtest_curves.html")


def _filter_by_date_range(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """按起止日期过滤（双闭区间 [start, end]）。"""
    if start_date is None and end_date is None:
        return df
    dates = pd.to_datetime(df["date"], format="%Y-%m-%d")
    mask = pd.Series(True, index=df.index)
    if start_date is not None:
        mask &= dates >= start_date
    if end_date is not None:
        mask &= dates <= end_date
    return df.loc[mask].reset_index(drop=True)


def run(
    output_root: Path,
    run_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    indices: list[tuple[str, str, str]] | None = None,
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> dict[str, Path]:
    """
    爬取指定指数历史数据，按起止日期过滤，每个指数输出到独立子目录。

    Args:
        output_root: 产物根目录（如 artifacts/backtest_base）
        run_id: 运行标识（如 20260315_123456_fund_index）
        start_date: 起始日期 YYYY-MM-DD，None 表示不限制
        end_date: 结束日期 YYYY-MM-DD，None 表示不限制
        indices: [(code, name, subdir_name), ...]，默认使用 DEFAULT_INDICES
        request_delay: 请求间隔秒数

    Returns:
        {code: output_path} 成功写入的 CSV 路径
    """
    indices = indices or DEFAULT_INDICES
    output_root = Path(output_root)
    base_dir = output_root / run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {}
    for i, (code, name, subdir_name) in enumerate(indices):
        if i > 0:
            time.sleep(request_delay)
        try:
            df = fetch_index_hist(code, period="day")
            if df.empty:
                LOG.warning("指数 %s (%s) 返回空数据，跳过", code, name)
                continue
            # 统一列名便于下游使用
            df = df.rename(columns={
                "日期": "date",
                "收盘指数": "close",
                "开盘指数": "open",
                "最高指数": "high",
                "最低指数": "low",
                "涨跌幅": "pct_chg",
            })
            df = _filter_by_date_range(df, start_date, end_date)
            if df.empty:
                LOG.warning("指数 %s (%s) 在 %s ~ %s 区间无数据，跳过",
                            code, name, start_date or "?", end_date or "?")
                continue
            df["symbol"] = str(code)
            df["name"] = str(name)
            subdir = base_dir / f"{code}_{subdir_name}"
            actual_start = str(df["date"].iloc[0])
            actual_end = str(df["date"].iloc[-1])
            _write_index_outputs(df, subdir, code, name, actual_start, actual_end)
            result[code] = subdir / "equity_curve.csv"
            LOG.info("已写入 %s: %d 行, %s ~ %s", subdir, len(df), actual_start, actual_end)
        except Exception as e:
            LOG.exception("爬取指数 %s (%s) 失败: %s", code, name, e)
            raise

    return result


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="爬取申万宏源基金指数历史数据（日频），作为 Wind 885001/885003 的公开替代"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"产物根目录（默认: {DEFAULT_OUTPUT_ROOT}）",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="运行标识，默认用 日期_时间_fund_index（如 20260315_123456_fund_index）",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="起始日期 YYYY-MM-DD，不指定则取全量",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=DEFAULT_END,
        help=f"结束日期 YYYY-MM-DD（默认: {DEFAULT_END}）",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        help=f"请求间隔秒数（默认: {DEFAULT_REQUEST_DELAY}）",
    )
    args = parser.parse_args()

    run_id = args.run_id
    if run_id is None:
        now = datetime.datetime.now()
        run_id = now.strftime("%Y%m%d_%H%M%S") + "_fund_index"

    try:
        paths = run(
            output_root=args.output_root,
            run_id=run_id,
            start_date=args.start_date,
            end_date=args.end_date,
            request_delay=args.request_delay,
        )
        print(f"\n完成：共写入 {len(paths)} 个指数")
        print(f"产物根目录: {args.output_root / run_id}")
        for code, p in paths.items():
            print(f"  {code}: {p}")
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
