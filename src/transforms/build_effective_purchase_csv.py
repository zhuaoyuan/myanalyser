"""从 fund_purchase 剔除黑名单基金，生成 fund_purchase_effective.csv。不修改原始 fund_purchase。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validators.validate_pipeline_artifacts import validate_stage_or_raise


def _load_blacklist_codes(blacklist_path: Path) -> set[str]:
    """从黑名单文件加载基金代码（支持 基金代码 列或纯代码列）。"""
    if not blacklist_path.exists():
        return set()
    df = pd.read_csv(blacklist_path, dtype=str, encoding="utf-8-sig")
    if df.empty:
        return set()
    # 优先使用 基金代码 列，否则尝试第一列
    if "基金代码" in df.columns:
        col = "基金代码"
    elif len(df.columns) > 0:
        col = df.columns[0]
    else:
        return set()
    codes = [
        str(v).strip().zfill(6)
        for v in df[col].dropna().astype(str).tolist()
        if str(v).strip()
    ]
    return set(codes)


def build_effective_purchase_csv(
    purchase_csv: Path,
    blacklist_csv: Path | None,
    output_csv: Path,
) -> dict[str, object]:
    """从 fund_purchase 剔除黑名单，输出 fund_purchase_effective.csv。"""
    validate_stage_or_raise("fund_etl_step2_input", purchase_csv=purchase_csv)

    purchase_df = pd.read_csv(purchase_csv, dtype={"基金代码": str}, encoding="utf-8-sig")
    purchase_df["基金代码"] = purchase_df["基金代码"].map(
        lambda v: str(v).strip().zfill(6)
    )
    original_count = len(purchase_df)

    blacklist_codes: set[str] = set()
    if blacklist_csv and blacklist_csv.exists():
        blacklist_codes = _load_blacklist_codes(blacklist_csv)

    kept_df = purchase_df[~purchase_df["基金代码"].isin(blacklist_codes)].copy()
    effective_count = len(kept_df)
    blacklist_removed = original_count - effective_count

    if kept_df.empty:
        raise ValueError(
            f"剔除黑名单后无剩余基金: purchase={purchase_csv} blacklist={blacklist_csv}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    kept_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    validate_stage_or_raise("fund_etl_step2_input", purchase_csv=output_csv)

    return {
        "original_count": original_count,
        "blacklist_removed": blacklist_removed,
        "effective_count": effective_count,
        "output_csv": output_csv,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build fund_purchase_effective.csv by excluding blacklisted funds"
    )
    parser.add_argument("--purchase-csv", type=Path, required=True)
    parser.add_argument("--blacklist-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_effective_purchase_csv(
        purchase_csv=args.purchase_csv,
        blacklist_csv=args.blacklist_csv,
        output_csv=args.output_csv,
    )
    print(f"original_count={result['original_count']}")
    print(f"blacklist_removed={result['blacklist_removed']}")
    print(f"effective_count={result['effective_count']}")
    print(f"effective_purchase_csv={result['output_csv']}")


if __name__ == "__main__":
    main()
