from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contracts.pipeline_contracts import CONTRACTS, CsvContract, DirContract, STAGE_REQUIREMENTS


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def _norm_code(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.zfill(6)


def _validate_csv(path: Path, contract: CsvContract) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing file: {path}"]

    df = _read_csv(path)
    if len(df) < contract.min_rows:
        errors.append(f"row count too small: {path} rows={len(df)} min_rows={contract.min_rows}")

    missing = [c for c in contract.required_columns if c not in df.columns]
    if missing:
        errors.append(f"missing required columns: {path} -> {missing}")
        return errors

    work_df = df.copy()
    for col in [*contract.non_null_columns, *contract.unique_key_columns]:
        if col in work_df.columns:
            if col in {"基金代码", "基金编码"}:
                work_df[col] = _norm_code(work_df[col])
            else:
                work_df[col] = work_df[col].fillna("").astype(str).str.strip()

    for col in contract.non_null_columns:
        if col not in work_df.columns:
            continue
        bad = work_df[col].eq("")
        if bad.any():
            errors.append(f"non-null violated: {path} col={col} empty_rows={int(bad.sum())}")

    if contract.unique_key_columns:
        dup = work_df.duplicated(subset=list(contract.unique_key_columns), keep=False)
        if dup.any():
            errors.append(f"unique key violated: {path} keys={contract.unique_key_columns} duplicate_rows={int(dup.sum())}")

    for col in contract.numeric_columns:
        if col not in work_df.columns:
            continue
        scoped = work_df[col].replace("", pd.NA).dropna()
        if scoped.empty:
            continue
        parsed = pd.to_numeric(scoped, errors="coerce")
        bad = parsed.isna().sum()
        if bad > 0:
            errors.append(f"numeric parse failed: {path} col={col} bad_rows={int(bad)}")

    for col in contract.date_columns:
        if col not in work_df.columns:
            continue
        scoped = work_df[col].replace("", pd.NA).dropna()
        if scoped.empty:
            continue
        parsed = pd.to_datetime(scoped, errors="coerce")
        bad = parsed.isna().sum()
        if bad > 0:
            errors.append(f"date parse failed: {path} col={col} bad_rows={int(bad)}")

    for col, allowed in contract.allowed_values.items():
        if col not in work_df.columns:
            continue
        scoped = set(work_df[col].replace("", pd.NA).dropna().astype(str).str.strip().tolist())
        bad_values = sorted(v for v in scoped if v not in allowed)
        if bad_values:
            errors.append(f"invalid enum values: {path} col={col} bad={bad_values}")

    return errors


def _validate_dir(path: Path, contract: DirContract) -> list[str]:
    if not path.is_dir():
        return [f"missing directory: {path}"]
    count = len(list(path.glob("*.csv")))
    if count < contract.min_csv_files:
        return [f"csv file count too small: {path} csv_count={count} min={contract.min_csv_files}"]
    return []


def validate_stage(stage: str, artifacts: dict[str, Path]) -> list[str]:
    if stage not in STAGE_REQUIREMENTS:
        return [f"unsupported stage: {stage}"]

    errors: list[str] = []
    for contract_name, arg_key in STAGE_REQUIREMENTS[stage]:
        if arg_key not in artifacts:
            errors.append(f"missing artifact arg: stage={stage} key={arg_key}")
            continue
        path = Path(artifacts[arg_key]).resolve()
        contract = CONTRACTS[contract_name]
        if isinstance(contract, CsvContract):
            errors.extend(_validate_csv(path, contract))
        elif isinstance(contract, DirContract):
            errors.extend(_validate_dir(path, contract))
        else:
            errors.append(f"unknown contract type: {contract_name}")
    return errors


def validate_stage_or_raise(stage: str, **artifacts: Path) -> None:
    errors = validate_stage(stage=stage, artifacts=artifacts)
    if errors:
        raise ValueError("; ".join(errors))


def _parse_artifacts(items: list[str]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --artifact item (expect key=path): {item}")
        key, raw = item.split("=", 1)
        artifacts[key.strip()] = Path(raw.strip())
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate pipeline artifacts by stage contracts")
    parser.add_argument("--stage", required=True, choices=sorted(STAGE_REQUIREMENTS.keys()))
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="artifact binding, format: key=/path/to/file_or_dir; can be used multiple times",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = _parse_artifacts(args.artifact)
    errors = validate_stage(stage=args.stage, artifacts=artifacts)
    if errors:
        print("validate=failed")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)
    print("validate=ok")
    print(f"stage={args.stage}")


if __name__ == "__main__":
    main()
