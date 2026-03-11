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


def get_available_symbols(nav_dir: Path) -> set[str]:
    """返回净值目录下可用的基金编码集合（6 位补零）。"""
    nav_dir = _resolve_nav_dir(nav_dir)
    return {str(p.stem).strip().zfill(6) for p in nav_dir.glob("*.csv")}


def load_fund_nav_data(
    nav_dir: Path,
    max_funds: int = 200,
    start_date: str | None = None,
    end_date: str | None = None,
    allowed_codes: set[str] | None = None,
) -> BacktestData:
    """从 fund_adjusted_nav_by_code 目录加载复权净值并构建回测数据结构。

    若传入 allowed_codes，仅加载该集合中的基金（与 nav_dir 下实际存在的文件取交集）。
    """

    nav_dir = _resolve_nav_dir(nav_dir)
    all_files = sorted(nav_dir.glob("*.csv"))
    if allowed_codes is not None:
        allowed = {str(c).strip().zfill(6) for c in allowed_codes}
        all_files = [p for p in all_files if str(p.stem).strip().zfill(6) in allowed]
    csv_files = all_files[:max_funds]
    if not csv_files:
        raise ValueError(f"净值目录下没有 CSV 文件: {nav_dir}")

    rows = []
    by_symbol: dict[str, pd.DataFrame] = {}
    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None

    for p in csv_files:
        df = pd.read_csv(p, dtype={"基金代码": str}, encoding="utf-8-sig")
        if df.empty or "净值日期" not in df.columns or "复权净值" not in df.columns:
            continue
        df = df.copy()
        df["净值日期"] = pd.to_datetime(df["净值日期"], errors="coerce")
        df = df.dropna(subset=["净值日期", "复权净值"])
        if start_ts is not None:
            df = df[df["净值日期"] >= start_ts]
        if end_ts is not None:
            df = df[df["净值日期"] <= end_ts]
        if df.empty:
            continue

        code = df["基金代码"].iloc[0] if "基金代码" in df.columns else p.stem
        symbol = str(code).zfill(6)
        df = df.sort_values("净值日期")
        df_symbol = pd.DataFrame(
            {
                "date": df["净值日期"],
                "close": pd.to_numeric(df["复权净值"], errors="coerce"),
            }
        ).dropna()
        if df_symbol.empty:
            continue

        by_symbol[symbol] = df_symbol.reset_index(drop=True)
        chunk = df_symbol.assign(
            symbol=symbol,
            open=df_symbol["close"],
            high=df_symbol["close"],
            low=df_symbol["close"],
        )
        rows.append(chunk[["symbol", "date", "open", "high", "low", "close"]])

    if not rows:
        raise ValueError("未加载到任何有效净值数据")

    long_df = pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    trading_dates = sorted(
        pd.Series(long_df["date"].unique()).dropna().map(lambda d: pd.Timestamp(d).normalize()).tolist()
    )

    return BacktestData(long_df=long_df, by_symbol=by_symbol, trading_dates=trading_dates)
