from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class DiffRow:
    kind: str
    code: str
    event_date: str
    history_version: str
    history_value: str
    current_value: str


BONUS_EVENT_COL = "权益登记日"
SPLIT_EVENT_COL = "拆分折算日"
BONUS_DIR = "fund_bonus_by_code"
SPLIT_DIR = "fund_split_by_code"
REPORT_NAME = "fund_bonus_split_diff_report.csv"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({k: ("" if v is None else str(v).strip()) for k, v in row.items()})
        return rows


def _normalize_code(code: str) -> str:
    raw = str(code).strip()
    if raw.isdigit():
        return raw.zfill(6)
    return raw


def _parse_date(value: str) -> date | None:
    value = str(value).strip()
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_history_cutoff(version_dir: Path) -> date | None:
    match = re.match(r"^(\d{8})", version_dir.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _group_rows_by_date(
    rows: Sequence[dict[str, str]],
    code: str,
    event_col: str,
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        event_date = row.get(event_col, "").strip()
        if not event_date:
            continue
        row = dict(row)
        row["基金代码"] = _normalize_code(row.get("基金代码") or code)
        grouped.setdefault(event_date, []).append(row)
    return grouped


def _serialize_rows(rows: Sequence[dict[str, str]]) -> str:
    if not rows:
        return "[]"
    return json.dumps(list(rows), ensure_ascii=False, sort_keys=True)


def _iter_codes(dir_path: Path) -> set[str]:
    if not dir_path.exists():
        return set()
    return {_normalize_code(p.stem) for p in dir_path.glob("*.csv") if p.is_file()}


def _load_code_rows(dir_path: Path, code: str) -> list[dict[str, str]]:
    path = dir_path / f"{code}.csv"
    return _read_csv_rows(path)


def _compare_one_history(
    *,
    kind: str,
    event_col: str,
    current_dir: Path,
    history_dir: Path,
    history_version: str,
    cutoff: date | None,
    report_rows: list[DiffRow],
) -> None:
    current_codes = _iter_codes(current_dir)
    history_codes = _iter_codes(history_dir)
    all_codes = sorted(current_codes | history_codes)
    for code in all_codes:
        current_rows = _load_code_rows(current_dir, code)
        history_rows = _load_code_rows(history_dir, code)
        current_grouped = _group_rows_by_date(current_rows, code, event_col)
        history_grouped = _group_rows_by_date(history_rows, code, event_col)
        date_keys = sorted(set(current_grouped) | set(history_grouped))
        for event_date in date_keys:
            if cutoff is not None:
                parsed = _parse_date(event_date)
                if parsed is None or parsed > cutoff:
                    continue
            history_list = history_grouped.get(event_date, [])
            current_list = current_grouped.get(event_date, [])
            if _serialize_rows(history_list) == _serialize_rows(current_list):
                continue
            report_rows.append(
                DiffRow(
                    kind=kind,
                    code=code,
                    event_date=event_date,
                    history_version=history_version,
                    history_value=_serialize_rows(history_list),
                    current_value=_serialize_rows(current_list),
                )
            )


def compare_bonus_split(history_root: Path, current_root: Path) -> Path:
    report_rows: list[DiffRow] = []
    history_root = history_root.resolve()
    current_root = current_root.resolve()
    history_versions = sorted(
        [p for p in history_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )

    for version_dir in history_versions:
        cutoff = _parse_history_cutoff(version_dir)
        history_fund_etl = version_dir / "fund_etl"
        history_bonus = history_fund_etl / BONUS_DIR
        history_split = history_fund_etl / SPLIT_DIR

        _compare_one_history(
            kind="分红",
            event_col=BONUS_EVENT_COL,
            current_dir=current_root / BONUS_DIR,
            history_dir=history_bonus,
            history_version=version_dir.name,
            cutoff=cutoff,
            report_rows=report_rows,
        )
        _compare_one_history(
            kind="拆分",
            event_col=SPLIT_EVENT_COL,
            current_dir=current_root / SPLIT_DIR,
            history_dir=history_split,
            history_version=version_dir.name,
            cutoff=cutoff,
            report_rows=report_rows,
        )

    report_path = current_root / REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["类型", "基金编码", "问题日期", "差异历史版本", "历史结果值", "本次结果值"])
        for row in report_rows:
            writer.writerow(
                [
                    row.kind,
                    row.code,
                    row.event_date,
                    row.history_version,
                    row.history_value,
                    row.current_value,
                ]
            )

    return report_path


def _default_history_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "common" / "revise"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fund bonus/split history vs current crawl.")
    parser.add_argument(
        "--history-root",
        type=Path,
        default=_default_history_root(),
        help="历史归档根目录（默认 myanalyser/data/common/revise）",
    )
    parser.add_argument(
        "--current-root",
        type=Path,
        required=True,
        help="待比对 fund_etl 目录",
    )
    args = parser.parse_args()

    report_path = compare_bonus_split(history_root=args.history_root, current_root=args.current_root)
    print(f"[audit] report saved: {report_path}")


if __name__ == "__main__":
    main()
