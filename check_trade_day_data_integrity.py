from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = ["基金编码", "数据完整比例"]
DETAIL_COLUMNS = ["交易日日期", "该日期数据是否存在"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查基金在交易日区间内的数据完整性")
    parser.add_argument("--base-dir", required=True, help="数据根目录，目录下需包含 fund_adjusted_nav_by_code")
    parser.add_argument("--start-date", required=True, help="开始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="结束日期，格式 YYYY-MM-DD")
    parser.add_argument(
        "--trade-dates-csv",
        default=str(Path(__file__).resolve().parent / "trade_dates.csv"),
        help="交易日历 CSV 文件路径，默认使用 myanalyser/trade_dates.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录，默认使用 base-dir 下的 trade_day_integrity_reports",
    )
    return parser.parse_args()


def load_trade_days(trade_dates_csv: Path, start_date: str, end_date: str) -> list[str]:
    trade_df = pd.read_csv(trade_dates_csv, dtype={"trade_date": str}, encoding="utf-8-sig")
    if "trade_date" not in trade_df.columns:
        raise ValueError(f"交易日历缺少 trade_date 列: {trade_dates_csv}")

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if start > end:
        raise ValueError(f"start-date 不能大于 end-date: {start_date} > {end_date}")

    trade_df["trade_date"] = pd.to_datetime(trade_df["trade_date"], errors="coerce")
    trade_df = trade_df.dropna(subset=["trade_date"])

    selected = trade_df[(trade_df["trade_date"] >= start) & (trade_df["trade_date"] <= end)]
    return selected["trade_date"].dt.strftime("%Y-%m-%d").tolist()


def _parse_foundation_date(raw_value: object) -> pd.Timestamp:
    if pd.isna(raw_value):
        return pd.NaT
    text = str(raw_value).strip()
    if not text:
        return pd.NaT

    match_cn = re.search(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", text)
    if match_cn:
        year, month, day = match_cn.groups()
        return pd.to_datetime(f"{year}-{month}-{day}", errors="coerce")

    match_iso = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", text)
    if match_iso:
        return pd.to_datetime(match_iso.group(1), errors="coerce")

    match_slash = re.search(r"(\d{4}/\d{1,2}/\d{1,2})", text)
    if match_slash:
        return pd.to_datetime(match_slash.group(1), errors="coerce")

    return pd.to_datetime(text, errors="coerce")


def load_eligible_fund_codes(overview_csv: Path, start_date: str) -> set[str]:
    overview_df = pd.read_csv(overview_csv, dtype={"基金代码": str}, encoding="utf-8-sig")
    if "基金代码" not in overview_df.columns:
        raise ValueError(f"fund_overview.csv 缺少 基金代码 列: {overview_csv}")

    foundation_col = "成立日期/规模" if "成立日期/规模" in overview_df.columns else "成立日期"
    if foundation_col not in overview_df.columns:
        raise ValueError(f"fund_overview.csv 缺少 成立日期/规模 或 成立日期 列: {overview_csv}")

    start = pd.to_datetime(start_date)
    eligible_codes: set[str] = set()
    for _, row in overview_df.iterrows():
        fund_code = str(row["基金代码"]).strip().zfill(6)
        foundation_date = _parse_foundation_date(row[foundation_col])
        # 仅在明确判断“成立时间晚于开始日期”时忽略；无法解析成立时间时保留。
        if pd.notna(foundation_date) and foundation_date > start:
            continue
        eligible_codes.add(fund_code)
    return eligible_codes


def compute_integrity_for_fund(fund_csv: Path, trade_days: list[str]) -> tuple[str, float, pd.DataFrame]:
    fund_code = fund_csv.stem

    nav_df = pd.read_csv(fund_csv, dtype={"净值日期": str}, encoding="utf-8-sig")
    if "净值日期" not in nav_df.columns:
        raise ValueError(f"基金文件缺少 净值日期 列: {fund_csv}")

    existing_dates = set(nav_df["净值日期"].dropna().astype(str).str.strip().tolist())

    detail_records = []
    existing_count = 0
    for day in trade_days:
        exists = day in existing_dates
        if exists:
            existing_count += 1
        detail_records.append({"交易日日期": day, "该日期数据是否存在": "是" if exists else "否"})

    total_days = len(trade_days)
    ratio = (existing_count / total_days) if total_days > 0 else 0.0
    detail_df = pd.DataFrame(detail_records, columns=DETAIL_COLUMNS)
    return fund_code, ratio, detail_df


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir).resolve()
    fund_dir = base_dir / "fund_adjusted_nav_by_code"
    overview_csv = base_dir / "fund_overview.csv"
    trade_dates_csv = Path(args.trade_dates_csv).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (base_dir / "trade_day_integrity_reports")

    if not fund_dir.is_dir():
        raise FileNotFoundError(f"未找到目录: {fund_dir}")
    if not overview_csv.is_file():
        raise FileNotFoundError(f"未找到文件: {overview_csv}")
    if not trade_dates_csv.is_file():
        raise FileNotFoundError(f"未找到交易日历文件: {trade_dates_csv}")

    trade_days = load_trade_days(trade_dates_csv, args.start_date, args.end_date)
    eligible_codes = load_eligible_fund_codes(overview_csv, args.start_date)

    summary_output = output_dir / f"trade_day_integrity_summary_{args.start_date}_{args.end_date}.csv"
    details_dir = output_dir / f"details_{args.start_date}_{args.end_date}"
    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []
    fund_files = sorted(fund_dir.glob("*.csv"))
    processed_count = 0

    for fund_csv in fund_files:
        if fund_csv.stem not in eligible_codes:
            continue
        fund_code, ratio, detail_df = compute_integrity_for_fund(fund_csv, trade_days)
        summary_records.append({"基金编码": fund_code, "数据完整比例": f"{ratio:.6f}"})
        processed_count += 1

        detail_output = details_dir / f"{fund_code}_{args.start_date}_{args.end_date}.csv"
        detail_df.to_csv(detail_output, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summary_records, columns=SUMMARY_COLUMNS)
    summary_df.to_csv(summary_output, index=False, encoding="utf-8-sig")

    print(f"交易日数量: {len(trade_days)}")
    print(f"基金文件数量: {len(fund_files)}")
    print(f"纳入检查基金数量: {processed_count}")
    print(f"汇总文件: {summary_output}")
    print(f"明细目录: {details_dir}")


if __name__ == "__main__":
    main()
