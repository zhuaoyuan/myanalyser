from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validators.validate_pipeline_artifacts import validate_stage_or_raise


def build_filtered_purchase_csv(
    purchase_csv: Path,
    filter_csv: Path,
    output_csv: Path,
) -> dict[str, object]:
    validate_stage_or_raise("fund_etl_step2_input", purchase_csv=purchase_csv)
    validate_stage_or_raise("filtered_candidates_output", filter_csv=filter_csv)

    purchase_df = pd.read_csv(purchase_csv, dtype={"基金代码": str}, encoding="utf-8-sig")
    filter_df = pd.read_csv(filter_csv, dtype={"基金编码": str}, encoding="utf-8-sig")

    purchase_df["基金代码"] = purchase_df["基金代码"].map(lambda v: str(v).strip().zfill(6))
    filter_df["基金编码"] = filter_df["基金编码"].map(lambda v: str(v).strip().zfill(6))

    kept_codes = set(filter_df.loc[filter_df["是否过滤"] == "否", "基金编码"].dropna().tolist())
    kept_df = purchase_df[purchase_df["基金代码"].isin(kept_codes)].copy()
    if kept_df.empty:
        raise ValueError("all funds are filtered out; cannot continue step10 pipeline")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    kept_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    validate_stage_or_raise("filtered_purchase_output", purchase_csv=output_csv)

    return {
        "rows": int(len(kept_df)),
        "output_csv": output_csv,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build filtered purchase csv from filter result")
    parser.add_argument("--purchase-csv", type=Path, required=True)
    parser.add_argument("--filter-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_filtered_purchase_csv(
        purchase_csv=args.purchase_csv,
        filter_csv=args.filter_csv,
        output_csv=args.output_csv,
    )
    print(f"filtered_purchase_rows={result['rows']}")
    print(f"filtered_purchase_csv={result['output_csv']}")


if __name__ == "__main__":
    main()
