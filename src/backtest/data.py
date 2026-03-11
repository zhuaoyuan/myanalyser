"""回测数据加载与组织。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class BacktestData:
    long_df: pd.DataFrame
    by_symbol: dict[str, pd.DataFrame]
    trading_dates: list[pd.Timestamp]


def _resolve_nav_dir(nav_dir: Path) -> Path:
    nav_dir = Path(nav_dir).resolve()
    if nav_dir.is_dir() and list(nav_dir.glob("*.csv")):
        return nav_dir

    # 允许传入 run data 根目录或 fund_etl 目录
    candidates = [nav_dir / "fund_etl" / "fund_adjusted_nav_by_code"]
    for c in candidates:
        if c.is_dir() and list(c.glob("*.csv")):
            return c

    versions_dir = nav_dir / "versions"
    if versions_dir.is_dir():
        for version_path in sorted(versions_dir.iterdir()):
            candidate = version_path / "fund_etl" / "fund_adjusted_nav_by_code"
            if candidate.is_dir() and list(candidate.glob("*.csv")):
                return candidate

    raise FileNotFoundError(f"无法定位净值目录: {nav_dir}")


def load_fund_nav_data(
    nav_dir: Path,
    max_funds: int = 200,
    start_date: str | None = None,
    end_date: str | None = None,
) -> BacktestData:
    """从 fund_adjusted_nav_by_code 目录加载复权净值并构建回测数据结构。"""

    nav_dir = _resolve_nav_dir(nav_dir)
    csv_files = sorted(nav_dir.glob("*.csv"))[:max_funds]
    if not csv_files:
        raise ValueError(f"净值目录下没有 CSV 文件: {nav_dir}")

    rows = []
    by_symbol: dict[str, pd.DataFrame] = {}

    for p in csv_files:
        df = pd.read_csv(p, dtype={"基金代码": str}, encoding="utf-8-sig")
        if df.empty or "净值日期" not in df.columns or "复权净值" not in df.columns:
            continue
        df = df.copy()
        df["净值日期"] = pd.to_datetime(df["净值日期"], errors="coerce")
        df = df.dropna(subset=["净值日期", "复权净值"])
        if start_date:
            df = df[df["净值日期"] >= start_date]
        if end_date:
            df = df[df["净值日期"] <= end_date]
        if df.empty:
            continue

        code = df["基金代码"].iloc[0] if "基金代码" in df.columns else p.stem
        symbol = str(code).zfill(6)
        df = df.sort_values("净值日期")
        df_symbol = pd.DataFrame(
            {
                "date": pd.to_datetime(df["净值日期"], errors="coerce"),
                "close": pd.to_numeric(df["复权净值"], errors="coerce"),
            }
        ).dropna()
        if df_symbol.empty:
            continue

        by_symbol[symbol] = df_symbol.reset_index(drop=True)
        for _, r in df_symbol.iterrows():
            nav = float(r["close"])
            rows.append(
                {
                    "symbol": symbol,
                    "date": r["date"],
                    "open": nav,
                    "high": nav,
                    "low": nav,
                    "close": nav,
                }
            )

    if not rows:
        raise ValueError("未加载到任何有效净值数据")

    long_df = pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)
    trading_dates = sorted(
        pd.Series(long_df["date"].unique()).dropna().map(lambda d: pd.Timestamp(d).normalize()).tolist()
    )

    return BacktestData(long_df=long_df, by_symbol=by_symbol, trading_dates=trading_dates)
