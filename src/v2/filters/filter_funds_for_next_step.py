"""v2: 按数据质量过滤可进入下一步的基金列表（支持 end-date）。

协议约定（详见 docs/参考/v2日期与区间协议约定.md）：日期区间 [start_date, end_date] 双闭。

注意：基金代码使用 v2.utils.safe_fund_code，对非纯数字（如 'TBD'、'---'）返回空串。
旧版 str.zfill(6) 会得到 '000tbd' 等；若存在此类脏数据需先清洗。

规则：
  1. 规则1: 基金必须在 fund_overview.csv 中
  2. 规则2: 必须在 fund_nav_by_code 中存在 NAV 原始净值
  3. 规则3: 必须在 fund_adjusted_nav_by_code 中存在复权净值
  4. 规则4: Compare 明细在 [start_date, end_date] 内（end_date 建议延伸 hold_days 覆盖完整持仓周期）
     本地远程收益率偏差绝对值须 < max_abs_deviation
  5. 规则5: Integrity 明细在 [start_date, end_date] 内各交易日数据须完整
     （end_date 建议延伸 hold_days 覆盖完整持仓周期）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from validators.validate_pipeline_artifacts import validate_stage_or_raise

from v2.utils import safe_fund_code

OUTPUT_COLUMNS = ["基金编码", "是否过滤", "过滤原因"]


def _load_purchase_codes(purchase_csv: Path) -> list[str]:
    purchase_df = pd.read_csv(purchase_csv, dtype={"基金代码": str}, encoding="utf-8-sig")
    if "基金代码" not in purchase_df.columns:
        raise ValueError(f"fund_purchase.csv 缺少 基金代码 列: {purchase_csv}")
    codes = [safe_fund_code(code) for code in purchase_df["基金代码"].dropna().tolist()]
    return list(dict.fromkeys(codes))


def _load_overview_codes(overview_csv: Path) -> set[str]:
    if not overview_csv.exists():
        return set()
    overview_df = pd.read_csv(overview_csv, dtype={"基金代码": str}, encoding="utf-8-sig")
    if "基金代码" not in overview_df.columns:
        return set()
    return {safe_fund_code(code) for code in overview_df["基金代码"].dropna().tolist()}


def _load_code_stems_from_dir(csv_dir: Path) -> set[str]:
    if not csv_dir.is_dir():
        return set()
    return {path.stem for path in csv_dir.glob("*.csv")}


def _compare_detail_issue_reasons(
    detail_csv: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
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
    scoped = detail_df[(detail_df["期初日期"] >= start_date) & (detail_df["期初日期"] <= end_date)].copy()

    if "期末日期" in scoped.columns:
        scoped["期末日期"] = pd.to_datetime(scoped["期末日期"], errors="coerce")
        scoped = scoped[scoped["期末日期"].notna() & (scoped["期末日期"] <= end_date)]

    if scoped.empty:
        return ["规则4: 指定区间内无任何比对记录"]

    scoped["本地远程收益率偏差"] = pd.to_numeric(scoped["本地远程收益率偏差"], errors="coerce")
    bad = scoped["本地远程收益率偏差"].abs() >= max_abs_deviation
    if bad.fillna(True).any():
        return [f"规则4: 指定区间内存在偏差>={max_abs_deviation:.2%}或缺失"]
    return []


def _integrity_issue_reasons(
    detail_csv: Path | None,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[str]:
    if detail_csv is None or not detail_csv.exists():
        return ["规则5: trade day integrity details 缺失"]

    detail_df = pd.read_csv(detail_csv, dtype=str, encoding="utf-8-sig")
    if "交易日日期" not in detail_df.columns or "该日期数据是否存在" not in detail_df.columns:
        return ["规则5: trade day integrity details 列缺失"]

    detail_df["交易日日期"] = pd.to_datetime(detail_df["交易日日期"], errors="coerce")
    scoped = detail_df[(detail_df["交易日日期"] >= start_date) & (detail_df["交易日日期"] <= end_date)].copy()
    if scoped.empty:
        return []

    exists_flag = scoped["该日期数据是否存在"].fillna("").astype(str).str.strip()
    if (exists_flag != "是").any():
        return ["规则5: 指定区间内存在交易日数据不完整"]
    return []


def filter_funds_for_next_step(
    *,
    purchase_csv: Path,
    overview_csv: Path,
    nav_dir: Path,
    adjusted_nav_dir: Path,
    compare_details_dir: Path,
    integrity_details_dir: Path,
    start_date: str,
    end_date: str,
    max_abs_deviation: float = 0.02,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    if start_ts > end_ts:
        raise ValueError(f"start-date cannot be after end-date: {start_date} > {end_date}")

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
                end_date=end_ts,
                max_abs_deviation=max_abs_deviation,
            )
        )

        integrity_candidates = (
            sorted(integrity_details_dir.glob(f"{code}_*.csv"), key=lambda p: p.name, reverse=True)
            if integrity_details_dir.is_dir()
            else []
        )
        integrity_detail_csv = integrity_candidates[0] if integrity_candidates else None
        reasons.extend(
            _integrity_issue_reasons(detail_csv=integrity_detail_csv, start_date=start_ts, end_date=end_ts)
        )

        rows.append(
            {
                "基金编码": code,
                "是否过滤": "是" if reasons else "否",
                "过滤原因": "；".join(reasons),
            }
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2: 按窗口过滤可进入下一步的基金列表")
    parser.add_argument("--base-dir", required=True, type=Path, help="fund_etl 目录，包含 fund_purchase.csv 等")
    parser.add_argument(
        "--purchase-csv",
        default=None,
        type=Path,
        help="申购列表 CSV（默认优先 fund_purchase_effective.csv，否则 fund_purchase.csv）",
    )
    parser.add_argument("--compare-details-dir", required=True, type=Path)
    parser.add_argument("--integrity-details-dir", required=True, type=Path)
    parser.add_argument("--start-date", required=True, help="过滤判定起始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="过滤判定结束日期 YYYY-MM-DD")
    parser.add_argument(
        "--max-abs-deviation",
        default=0.02,
        type=float,
        help="规则4允许的最大偏差绝对值，超过或等于即过滤（默认 0.02）",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        type=Path,
        help="输出 CSV 路径（默认 {base-dir}/filtered_fund_candidates.csv）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    _effective = base_dir / "fund_purchase_effective.csv"
    _fallback = base_dir / "fund_purchase.csv"
    purchase_csv = (
        args.purchase_csv.resolve()
        if args.purchase_csv
        else (_effective if _effective.exists() else _fallback)
    )
    output_csv = args.output_csv.resolve() if args.output_csv else (base_dir / "filtered_fund_candidates.csv")

    validate_stage_or_raise(
        "filter_input",
        purchase_csv=purchase_csv,
        overview_csv=base_dir / "fund_overview.csv",
        nav_dir=base_dir / "fund_nav_by_code",
        adjusted_nav_dir=base_dir / "fund_adjusted_nav_by_code",
        compare_details_dir=args.compare_details_dir.resolve(),
        integrity_details_dir=args.integrity_details_dir.resolve(),
    )

    result_df = filter_funds_for_next_step(
        purchase_csv=purchase_csv,
        overview_csv=base_dir / "fund_overview.csv",
        nav_dir=base_dir / "fund_nav_by_code",
        adjusted_nav_dir=base_dir / "fund_adjusted_nav_by_code",
        compare_details_dir=args.compare_details_dir.resolve(),
        integrity_details_dir=args.integrity_details_dir.resolve(),
        start_date=args.start_date,
        end_date=args.end_date,
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
