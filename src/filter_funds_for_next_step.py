from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from validators.validate_pipeline_artifacts import validate_stage_or_raise


OUTPUT_COLUMNS = ["基金编码", "是否过滤", "过滤原因"]


def _safe_code(value: object) -> str:
    return str(value).strip().zfill(6)


def _load_purchase_codes(purchase_csv: Path) -> list[str]:
    purchase_df = pd.read_csv(purchase_csv, dtype={"基金代码": str}, encoding="utf-8-sig")
    if "基金代码" not in purchase_df.columns:
        raise ValueError(f"fund_purchase.csv 缺少 基金代码 列: {purchase_csv}")
    codes = [_safe_code(code) for code in purchase_df["基金代码"].dropna().tolist()]
    return list(dict.fromkeys(codes))


def _load_overview_codes(overview_csv: Path) -> set[str]:
    if not overview_csv.exists():
        return set()
    overview_df = pd.read_csv(overview_csv, dtype={"基金代码": str}, encoding="utf-8-sig")
    if "基金代码" not in overview_df.columns:
        return set()
    return {_safe_code(code) for code in overview_df["基金代码"].dropna().tolist()}


def _load_code_stems_from_dir(csv_dir: Path) -> set[str]:
    if not csv_dir.is_dir():
        return set()
    return {path.stem for path in csv_dir.glob("*.csv")}


def _compare_detail_issue_reasons(
    detail_csv: Path,
    start_date: pd.Timestamp,
    max_abs_deviation: float,
) -> list[str]:
    if not detail_csv.exists():
        return ["规则4: compare details 缺失或无比对记录"]

    detail_df = pd.read_csv(detail_csv, dtype=str, encoding="utf-8-sig")
    if "期初日期" not in detail_df.columns:
        return ["规则4: compare details 缺少 期初日期 列"]
    if "本地远程收益率偏差" not in detail_df.columns:
        return ["规则4: compare details 缺少 本地远程收益率偏差 列"]

    detail_df["期初日期"] = pd.to_datetime(detail_df["期初日期"], errors="coerce")
    scoped = detail_df[detail_df["期初日期"] >= start_date].copy()
    if scoped.empty:
        return ["规则4: 指定日期后无任何比对记录"]

    scoped["本地远程收益率偏差"] = pd.to_numeric(scoped["本地远程收益率偏差"], errors="coerce")
    bad = scoped["本地远程收益率偏差"].abs() >= max_abs_deviation
    if bad.fillna(False).any():
        return [f"规则4: 指定日期后存在偏差>={max_abs_deviation:.2%}"]
    return []


def _integrity_issue_reasons(
    detail_csv: Path | None,
    start_date: pd.Timestamp,
) -> list[str]:
    if detail_csv is None or not detail_csv.exists():
        return ["规则5: trade day integrity details 缺失"]

    detail_df = pd.read_csv(detail_csv, dtype=str, encoding="utf-8-sig")
    if "交易日日期" not in detail_df.columns or "该日期数据是否存在" not in detail_df.columns:
        return ["规则5: trade day integrity details 列缺失"]

    detail_df["交易日日期"] = pd.to_datetime(detail_df["交易日日期"], errors="coerce")
    scoped = detail_df[detail_df["交易日日期"] >= start_date].copy()
    if scoped.empty:
        return []

    exists_flag = scoped["该日期数据是否存在"].fillna("").astype(str).str.strip()
    if (exists_flag != "是").any():
        return ["规则5: 指定日期后存在交易日数据不完整"]
    return []


def filter_funds_for_next_step(
    *,
    purchase_csv: Path,
    overview_csv: Path,
    nav_dir: Path,
    adjusted_nav_dir: Path,
    compare_details_dir: Path,
    integrity_details_dir: Path,
    start_date: str = "2023-01-01",
    max_abs_deviation: float = 0.02,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start_date)
    purchase_codes = _load_purchase_codes(purchase_csv)
    overview_codes = _load_overview_codes(overview_csv)
    nav_codes = _load_code_stems_from_dir(nav_dir)
    adjusted_nav_codes = _load_code_stems_from_dir(adjusted_nav_dir)
    rows: list[dict[str, str]] = []
    for code in purchase_codes:
        reasons: list[str] = []
        if code not in overview_codes:
            reasons.append("规则1: fund_overview.csv 中不存在该基金")
        if code not in nav_codes:
            reasons.append("规则2: fund_nav_by_code 中不存在该基金")
        if code not in adjusted_nav_codes:
            reasons.append("规则3: fund_adjusted_nav_by_code 中不存在该基金")

        compare_detail_csv = compare_details_dir / f"{code}.csv"
        reasons.extend(
            _compare_detail_issue_reasons(
                detail_csv=compare_detail_csv,
                start_date=start_ts,
                max_abs_deviation=max_abs_deviation,
            )
        )

        integrity_candidates = (
            sorted(integrity_details_dir.glob(f"{code}_*.csv"), key=lambda p: p.name, reverse=True)
            if integrity_details_dir.is_dir()
            else []
        )
        integrity_detail_csv = integrity_candidates[0] if integrity_candidates else None
        reasons.extend(_integrity_issue_reasons(detail_csv=integrity_detail_csv, start_date=start_ts))

        rows.append(
            {
                "基金编码": code,
                "是否过滤": "是" if reasons else "否",
                "过滤原因": "；".join(reasons),
            }
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 Step9 后数据质量过滤可进入下一步的基金列表")
    parser.add_argument("--base-dir", required=True, type=Path, help="fund_etl 目录，包含 fund_purchase.csv 等")
    parser.add_argument("--compare-details-dir", required=True, type=Path, help="Step9 compare 明细目录（.../fund_return_compare/details）")
    parser.add_argument(
        "--integrity-details-dir",
        required=True,
        type=Path,
        help="Step8 交易日完整性明细目录（.../trade_day_integrity_reports/details_YYYY-MM-DD_YYYY-MM-DD）",
    )
    parser.add_argument("--start-date", default="2023-01-01", help="过滤判定起始日期（默认 2023-01-01）")
    parser.add_argument(
        "--max-abs-deviation",
        default=0.02,
        type=float,
        help="规则4允许的最大偏差绝对值，超过或等于即过滤（默认 0.02）",
    )
    parser.add_argument("--output-csv", default=None, type=Path, help="输出 CSV 路径（默认 {base-dir}/filtered_fund_candidates.csv）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_csv = args.output_csv.resolve() if args.output_csv else (base_dir / "filtered_fund_candidates.csv")
    validate_stage_or_raise(
        "filter_input",
        purchase_csv=base_dir / "fund_purchase.csv",
        overview_csv=base_dir / "fund_overview.csv",
        nav_dir=base_dir / "fund_nav_by_code",
        adjusted_nav_dir=base_dir / "fund_adjusted_nav_by_code",
        compare_details_dir=args.compare_details_dir.resolve(),
        integrity_details_dir=args.integrity_details_dir.resolve(),
    )

    result_df = filter_funds_for_next_step(
        purchase_csv=base_dir / "fund_purchase.csv",
        overview_csv=base_dir / "fund_overview.csv",
        nav_dir=base_dir / "fund_nav_by_code",
        adjusted_nav_dir=base_dir / "fund_adjusted_nav_by_code",
        compare_details_dir=args.compare_details_dir.resolve(),
        integrity_details_dir=args.integrity_details_dir.resolve(),
        start_date=args.start_date,
        max_abs_deviation=args.max_abs_deviation,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    validate_stage_or_raise("filtered_candidates_output", filter_csv=output_csv)
    print(f"输出文件: {output_csv}")
    print(f"总基金数: {len(result_df)}")
    print(f"过滤基金数: {(result_df['是否过滤'] == '是').sum()}")


if __name__ == "__main__":
    main()
