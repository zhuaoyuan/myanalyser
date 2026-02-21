from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PRIORITY_COLUMNS = ["ts", "stage", "code", "error"]


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".csv")


def _load_jsonl_records(input_path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"line {line_no}: invalid json ({err})") from err
            if not isinstance(record, dict):
                raise ValueError(f"line {line_no}: json value must be object")
            records.append(record)
    return records


def _build_columns(records: list[dict[str, object]]) -> list[str]:
    all_keys: set[str] = set()
    for record in records:
        all_keys.update(record.keys())

    columns: list[str] = [name for name in PRIORITY_COLUMNS if name in all_keys]
    remaining = sorted(name for name in all_keys if name not in columns)
    return columns + remaining


def jsonl_to_csv(input_path: Path, output_path: Path) -> int:
    records = _load_jsonl_records(input_path)
    columns = _build_columns(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    return len(records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert failed log JSONL file (e.g. failed_nav.jsonl) to CSV"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Path to input JSONL log file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to output CSV file (default: same path with .csv suffix)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path: Path = args.input
    output_path: Path = args.output or _default_output_path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"input path is not a file: {input_path}")

    row_count = jsonl_to_csv(input_path=input_path, output_path=output_path)
    print(f"converted {row_count} rows: {output_path}")


if __name__ == "__main__":
    main()
