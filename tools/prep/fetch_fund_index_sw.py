#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
爬取申万宏源基金指数全量历史数据（日频）。

背景说明：
  Wind 885001（偏股混合型基金指数）、885003（偏债混合型基金指数）为万得专有指数，
  官方数据需通过 Wind 终端/数据库获取，目前未发现公开免费且天级更新的数据源。

  本脚本使用 AKShare 获取申万宏源基金指数，作为概念相近的公开替代：
  - 807100：申万宏源权益基金指数（偏股，与 Wind 885001 概念相近）
  - 807200：申万宏源债券基金指数（偏债，与 Wind 885003 概念相近）
  - 807300：申万宏源混合基金指数（混合）

  数据来源：申万宏源研究 https://www.swsresearch.com/institute_sw/allIndex/releasedIndex
  更新频率：交易日日频，与 Wind 指数类似。
"""
from __future__ import annotations

import argparse
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
DEFAULT_INDICES = [
    ("807100", "申万宏源权益基金指数"),
    ("807200", "申万宏源债券基金指数"),
    ("807300", "申万宏源混合基金指数"),
]

LOG = logging.getLogger(__name__)


def fetch_index_hist(symbol: str, period: str = "day") -> pd.DataFrame:
    """调用 AKShare 获取单只指数历史行情。"""
    return ak.index_hist_fund_sw(symbol=symbol, period=period)


def run(
    output_dir: Path,
    indices: list[tuple[str, str]] | None = None,
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> dict[str, Path]:
    """
    爬取指定指数全量历史数据，输出为 CSV。

    Args:
        output_dir: 输出目录，每个指数一个 CSV 文件
        indices: [(code, name), ...]，默认使用 DEFAULT_INDICES
        request_delay: 请求间隔秒数

    Returns:
        {code: output_path} 成功写入的路径
    """
    indices = indices or DEFAULT_INDICES
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {}
    for i, (code, name) in enumerate(indices):
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
            df["symbol"] = str(code)
            df["name"] = str(name)
            out_path = output_dir / f"fund_index_{code}.csv"
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            result[code] = out_path
            LOG.info("已写入 %s: %d 行, %s ~ %s", out_path.name, len(df),
                     df["date"].iloc[0], df["date"].iloc[-1])
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
        description="爬取申万宏源基金指数全量历史数据（日频），作为 Wind 885001/885003 的公开替代"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("myanalyser/data/fund_index_sw"),
        help="输出目录（默认: myanalyser/data/fund_index_sw）",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        help=f"请求间隔秒数（默认: {DEFAULT_REQUEST_DELAY}）",
    )
    args = parser.parse_args()

    try:
        paths = run(
            output_dir=args.output_dir,
            request_delay=args.request_delay,
        )
        print(f"\n完成：共写入 {len(paths)} 个指数")
        for code, p in paths.items():
            print(f"  {code}: {p}")
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
