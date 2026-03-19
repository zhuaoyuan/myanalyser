"""V2 基线生成与回归的共享逻辑。供 generate_baseline_expected.py 与 test_v2_baseline_regression 共同调用。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_cum_return_from_adjusted_nav(adj_dir: Path, cum_dir: Path) -> None:
    """从 fund_adjusted_nav_by_code 目录生成 fund_cum_return_by_code。累计收益率以百分点计（1=1%）。"""
    cum_dir.mkdir(parents=True, exist_ok=True)
    for p in adj_dir.glob("*.csv"):
        df = pd.read_csv(p, dtype=str, encoding="utf-8-sig")
        if "净值日期" not in df.columns or "复权净值" not in df.columns:
            continue
        df["净值日期"] = pd.to_datetime(df["净值日期"], errors="coerce")
        df["复权净值"] = pd.to_numeric(df["复权净值"], errors="coerce")
        df = df.dropna(subset=["净值日期", "复权净值"]).sort_values("净值日期")
        if df.empty or len(df) < 2:
            continue
        code = p.stem.zfill(6)
        base = float(df["复权净值"].iloc[0])
        if base <= 0:
            continue
        cum = (df["复权净值"] / base - 1) * 100  # percentage points
        out = cum_dir / f"{code}.csv"
        pd.DataFrame({
            "基金代码": code,
            "日期": df["净值日期"].dt.strftime("%Y-%m-%d"),
            "累计收益率": cum.round(6).astype(str),
        }).to_csv(out, index=False, encoding="utf-8-sig")
