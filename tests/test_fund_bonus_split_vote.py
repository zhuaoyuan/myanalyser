from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from fund_bonus_split_vote import vote_category


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events


def _make_rerun(tmp_path: Path, contents_by_attempt: dict[int, str]):
    def _rerun(code: str, attempt: int) -> Path:
        content = contents_by_attempt[attempt]
        out_dir = tmp_path / f"rerun_{attempt}"
        out_path = out_dir / f"{code}.csv"
        _write_text(out_path, content)
        return out_path

    return _rerun


def test_vote_category_same_copy(tmp_path: Path) -> None:
    current_dir = tmp_path / "current"
    history_dir = tmp_path / "history"
    out_dir = tmp_path / "out"
    logs_dir = tmp_path / "logs"

    _write_text(current_dir / "000001.csv", "A")
    _write_text(history_dir / "000001.csv", "A")

    def _rerun_fail(code: str, attempt: int) -> Path:
        raise AssertionError("rerun should not be called for identical files")

    stats = vote_category(
        kind="bonus",
        current_dir=current_dir,
        history_dir=history_dir,
        out_dir=out_dir,
        logs_dir=logs_dir,
        rerun_times=3,
        rerun_fetch=_rerun_fail,
    )

    assert (out_dir / "000001.csv").read_text(encoding="utf-8") == "A"
    assert stats.same == 1


def test_vote_category_missing_skip(tmp_path: Path) -> None:
    current_dir = tmp_path / "current"
    history_dir = tmp_path / "history"
    out_dir = tmp_path / "out"
    logs_dir = tmp_path / "logs"

    _write_text(current_dir / "000001.csv", "A")

    stats = vote_category(
        kind="bonus",
        current_dir=current_dir,
        history_dir=history_dir,
        out_dir=out_dir,
        logs_dir=logs_dir,
        rerun_times=3,
        rerun_fetch=_make_rerun(tmp_path, {1: "A", 2: "A", 3: "A"}),
    )

    assert not (out_dir / "000001.csv").exists()
    events = _read_events(logs_dir / "vote_bonus.jsonl")
    assert any(evt.get("event") == "missing" for evt in events)
    assert stats.missing == 1


def test_vote_category_majority(tmp_path: Path) -> None:
    current_dir = tmp_path / "current"
    history_dir = tmp_path / "history"
    out_dir = tmp_path / "out"
    logs_dir = tmp_path / "logs"

    _write_text(current_dir / "000001.csv", "A")
    _write_text(history_dir / "000001.csv", "B")

    stats = vote_category(
        kind="bonus",
        current_dir=current_dir,
        history_dir=history_dir,
        out_dir=out_dir,
        logs_dir=logs_dir,
        rerun_times=3,
        rerun_fetch=_make_rerun(tmp_path, {1: "A", 2: "C", 3: "A"}),
    )

    assert (out_dir / "000001.csv").read_text(encoding="utf-8") == "A"
    assert stats.voted == 1


def test_vote_category_tie_skip(tmp_path: Path) -> None:
    current_dir = tmp_path / "current"
    history_dir = tmp_path / "history"
    out_dir = tmp_path / "out"
    logs_dir = tmp_path / "logs"

    _write_text(current_dir / "000001.csv", "A")
    _write_text(history_dir / "000001.csv", "B")

    stats = vote_category(
        kind="bonus",
        current_dir=current_dir,
        history_dir=history_dir,
        out_dir=out_dir,
        logs_dir=logs_dir,
        rerun_times=3,
        rerun_fetch=_make_rerun(tmp_path, {1: "A", 2: "B", 3: "C"}),
    )

    assert not (out_dir / "000001.csv").exists()
    events = _read_events(logs_dir / "vote_bonus.jsonl")
    assert any(evt.get("event") == "tie" for evt in events)
    assert stats.tie == 1


def test_vote_category_respects_codes_filter(tmp_path: Path) -> None:
    current_dir = tmp_path / "current"
    history_dir = tmp_path / "history"
    out_dir = tmp_path / "out"
    logs_dir = tmp_path / "logs"

    _write_text(current_dir / "000001.csv", "A")
    _write_text(history_dir / "000001.csv", "A")
    _write_text(current_dir / "000002.csv", "B")
    _write_text(history_dir / "000002.csv", "B")

    stats = vote_category(
        kind="bonus",
        current_dir=current_dir,
        history_dir=history_dir,
        out_dir=out_dir,
        logs_dir=logs_dir,
        rerun_times=3,
        rerun_fetch=_make_rerun(tmp_path, {1: "A", 2: "A", 3: "A"}),
        codes=["000001"],
    )

    assert (out_dir / "000001.csv").exists()
    assert not (out_dir / "000002.csv").exists()
    assert stats.total_codes == 1
