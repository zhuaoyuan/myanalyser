from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from fund_bonus_split_audit import compare_bonus_split


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def test_compare_bonus_split_respects_cutoff(tmp_path: Path) -> None:
    history_root = tmp_path / "history"
    current_root = tmp_path / "current" / "fund_etl"

    history_version = history_root / "20260301_120000_runid" / "fund_etl"
    history_bonus = history_version / "fund_bonus_by_code" / "000001.csv"
    current_bonus = current_root / "fund_bonus_by_code" / "000001.csv"

    header = ["基金代码", "年份", "权益登记日", "除息日", "每份分红", "分红发放日"]
    history_rows = [
        ["000001", "2026年", "2026-02-28", "2026-02-28", "每份派现金0.0100元", "2026-03-01"],
        ["000001", "2026年", "2026-03-02", "2026-03-02", "每份派现金0.0200元", "2026-03-03"],
    ]
    current_rows = [
        ["000001", "2026年", "2026-02-28", "2026-02-28", "每份派现金0.0150元", "2026-03-01"],
        ["000001", "2026年", "2026-03-02", "2026-03-02", "每份派现金0.0200元", "2026-03-03"],
    ]

    _write_csv(history_bonus, header, history_rows)
    _write_csv(current_bonus, header, current_rows)

    report_path = compare_bonus_split(history_root=history_root, current_root=current_root)
    assert report_path.exists()

    with report_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["类型", "基金编码", "问题日期", "差异历史版本", "历史结果值", "本次结果值"]
    assert len(rows) == 2
    assert rows[1][0] == "分红"
    assert rows[1][1] == "000001"
    assert rows[1][2] == "2026-02-28"
    assert rows[1][3] == "20260301_120000_runid"
