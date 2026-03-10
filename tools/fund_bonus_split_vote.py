from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from fund_etl import ProgressConfig, RetryConfig, run_step4_bonus, run_step5_split

BONUS_DIR = "fund_bonus_by_code"
SPLIT_DIR = "fund_split_by_code"


@dataclass(frozen=True)
class VoteStats:
    total_codes: int
    same: int
    missing: int
    rerun_missing: int
    all_different: int
    tie: int
    voted: int


def _now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalize_code(code: str) -> str:
    raw = str(code).strip()
    if raw.isdigit():
        return raw.zfill(6)
    return raw


def _build_code_map(dir_path: Path) -> dict[str, Path]:
    code_map: dict[str, Path] = {}
    if not dir_path.exists():
        return code_map
    for path in dir_path.glob("*.csv"):
        if not path.is_file():
            continue
        code = _normalize_code(path.stem)
        code_map.setdefault(code, path)
    return code_map


def _load_codes_from_purchase(purchase_csv: Path) -> list[str]:
    codes: list[str] = []
    if not purchase_csv.exists():
        return codes
    with purchase_csv.open("r", encoding="utf-8-sig") as f:
        header = f.readline().strip().split(",")
        if "基金代码" not in header:
            return codes
        code_idx = header.index("基金代码")
        for raw in f:
            row = raw.strip().split(",")
            if code_idx >= len(row):
                continue
            code = _normalize_code(row[code_idx])
            if code:
                codes.append(code)
    seen: set[str] = set()
    ordered: list[str] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            ordered.append(code)
    return ordered


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    return _hash_file(left) == _hash_file(right)


def _append_jsonl(path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload.setdefault("ts", _now_ts())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def _copy_to_out(src: Path, out_dir: Path, code: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{_normalize_code(code)}.csv"
    shutil.copy2(src, dst)


def vote_category(
    *,
    kind: str,
    current_dir: Path,
    history_dir: Path,
    out_dir: Path,
    logs_dir: Path,
    rerun_times: int,
    rerun_fetch: Callable[[str, int], Path | None],
    codes: Iterable[str] | None = None,
) -> VoteStats:
    current_map = _build_code_map(current_dir)
    history_map = _build_code_map(history_dir)
    if codes is None:
        codes = sorted(set(current_map) | set(history_map))
    else:
        codes = [_normalize_code(code) for code in codes]
    log_path = logs_dir / f"vote_{kind}.jsonl"

    same = 0
    missing = 0
    rerun_missing = 0
    all_different = 0
    tie = 0
    voted = 0

    for code in codes:
        current_path = current_map.get(code)
        history_path = history_map.get(code)
        if current_path is None or history_path is None:
            _append_jsonl(
                log_path,
                {
                    "event": "missing",
                    "kind": kind,
                    "code": code,
                    "current_exists": current_path is not None,
                    "history_exists": history_path is not None,
                },
            )
            missing += 1
            continue

        try:
            if _files_equal(current_path, history_path):
                _copy_to_out(current_path, out_dir, code)
                same += 1
                continue
        except Exception as err:  # noqa: BLE001
            _append_jsonl(
                log_path,
                {
                    "event": "compare_error",
                    "kind": kind,
                    "code": code,
                    "error": str(err),
                },
            )
            missing += 1
            continue

        rerun_paths: list[Path] = []
        rerun_ok = True
        for attempt in range(1, rerun_times + 1):
            path = rerun_fetch(code, attempt)
            if path is None or not path.exists():
                _append_jsonl(
                    log_path,
                    {
                        "event": "rerun_missing",
                        "kind": kind,
                        "code": code,
                        "attempt": attempt,
                        "path": "" if path is None else str(path),
                    },
                )
                rerun_ok = False
                break
            rerun_paths.append(path)

        if not rerun_ok:
            rerun_missing += 1
            continue

        candidates = [history_path, current_path] + rerun_paths
        hash_to_paths: dict[str, list[Path]] = {}
        for path in candidates:
            try:
                digest = _hash_file(path)
            except Exception as err:  # noqa: BLE001
                _append_jsonl(
                    log_path,
                    {
                        "event": "hash_error",
                        "kind": kind,
                        "code": code,
                        "path": str(path),
                        "error": str(err),
                    },
                )
                rerun_ok = False
                break
            hash_to_paths.setdefault(digest, []).append(path)

        if not rerun_ok:
            rerun_missing += 1
            continue

        counts = sorted(
            ((digest, len(paths)) for digest, paths in hash_to_paths.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        top_digest, top_count = counts[0]
        top_ties = [digest for digest, count in counts if count == top_count]

        if top_count == 1:
            _append_jsonl(
                log_path,
                {
                    "event": "all_different",
                    "kind": kind,
                    "code": code,
                    "hashes": {digest: len(paths) for digest, paths in hash_to_paths.items()},
                },
            )
            all_different += 1
            continue

        if len(top_ties) > 1:
            _append_jsonl(
                log_path,
                {
                    "event": "tie",
                    "kind": kind,
                    "code": code,
                    "hashes": {digest: len(paths) for digest, paths in hash_to_paths.items()},
                },
            )
            tie += 1
            continue

        chosen_path = hash_to_paths[top_digest][0]
        _copy_to_out(chosen_path, out_dir, code)
        voted += 1
        _append_jsonl(
            log_path,
            {
                "event": "voted",
                "kind": kind,
                "code": code,
                "chosen_path": str(chosen_path),
                "vote_count": top_count,
            },
        )

    return VoteStats(
        total_codes=len(codes),
        same=same,
        missing=missing,
        rerun_missing=rerun_missing,
        all_different=all_different,
        tie=tie,
        voted=voted,
    )


class RerunRunner:
    def __init__(
        self,
        *,
        purchase_csv: Path,
        tmp_root: Path,
        retry_cfg: RetryConfig,
        progress_cfg: ProgressConfig,
    ) -> None:
        self.purchase_csv = purchase_csv
        self.tmp_root = tmp_root
        self.retry_cfg = retry_cfg
        self.progress_cfg = progress_cfg
        self.logs_dir = tmp_root / "logs"

    def fetch_bonus(self, code: str, attempt: int) -> Path | None:
        run_dir = self.tmp_root / f"bonus_rerun_{attempt}"
        fail_log = self.logs_dir / f"bonus_rerun_{attempt}.jsonl"
        run_step4_bonus(
            purchase_csv=self.purchase_csv,
            bonus_dir=run_dir,
            fail_log=fail_log,
            retry_cfg=self.retry_cfg,
            progress_cfg=self.progress_cfg,
            only_codes=[code],
        )
        return run_dir / f"{_normalize_code(code)}.csv"

    def fetch_split(self, code: str, attempt: int) -> Path | None:
        run_dir = self.tmp_root / f"split_rerun_{attempt}"
        fail_log = self.logs_dir / f"split_rerun_{attempt}.jsonl"
        run_step5_split(
            purchase_csv=self.purchase_csv,
            split_dir=run_dir,
            fail_log=fail_log,
            retry_cfg=self.retry_cfg,
            progress_cfg=self.progress_cfg,
            only_codes=[code],
        )
        return run_dir / f"{_normalize_code(code)}.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vote fund bonus/split data by multiple crawls.")
    parser.add_argument("--current-root", type=Path, required=True, help="本次 fund_etl 目录")
    parser.add_argument("--history-root", type=Path, required=True, help="历史 fund_etl 目录")
    parser.add_argument("--out-root", type=Path, required=True, help="输出目录（含 fund_bonus_by_code、fund_split_by_code）")
    parser.add_argument("--purchase-csv", type=Path, required=True, help="fund_purchase.csv 路径")
    parser.add_argument("--out-bonus-dir", type=Path, default=None, help="覆盖输出的 fund_bonus_by_code 目录")
    parser.add_argument("--out-split-dir", type=Path, default=None, help="覆盖输出的 fund_split_by_code 目录")
    parser.add_argument("--rerun-times", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--progress-interval", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    current_root = args.current_root.resolve()
    history_root = args.history_root.resolve()
    out_root = args.out_root.resolve()
    purchase_csv = args.purchase_csv.resolve()

    if not purchase_csv.exists():
        raise FileNotFoundError(f"purchase_csv not found: {purchase_csv}")

    current_bonus = current_root / BONUS_DIR
    current_split = current_root / SPLIT_DIR
    history_bonus = history_root / BONUS_DIR
    history_split = history_root / SPLIT_DIR

    out_bonus = args.out_bonus_dir.resolve() if args.out_bonus_dir else (out_root / BONUS_DIR)
    out_split = args.out_split_dir.resolve() if args.out_split_dir else (out_root / SPLIT_DIR)
    logs_dir = out_root / "logs"
    tmp_root = out_root / "_rerun_tmp"

    retry_cfg = RetryConfig(max_retries=args.max_retries, retry_sleep_seconds=args.retry_sleep)
    progress_cfg = ProgressConfig(print_interval_seconds=args.progress_interval)
    rerunner = RerunRunner(
        purchase_csv=purchase_csv,
        tmp_root=tmp_root,
        retry_cfg=retry_cfg,
        progress_cfg=progress_cfg,
    )

    codes = _load_codes_from_purchase(purchase_csv)
    bonus_stats = vote_category(
        kind="bonus",
        current_dir=current_bonus,
        history_dir=history_bonus,
        out_dir=out_bonus,
        logs_dir=logs_dir,
        rerun_times=args.rerun_times,
        rerun_fetch=rerunner.fetch_bonus,
        codes=codes,
    )
    split_stats = vote_category(
        kind="split",
        current_dir=current_split,
        history_dir=history_split,
        out_dir=out_split,
        logs_dir=logs_dir,
        rerun_times=args.rerun_times,
        rerun_fetch=rerunner.fetch_split,
        codes=codes,
    )

    summary = {
        "bonus": bonus_stats.__dict__,
        "split": split_stats.__dict__,
    }
    summary_path = logs_dir / "vote_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[vote] summary saved: {summary_path}")
    print(f"[vote] bonus: {bonus_stats}")
    print(f"[vote] split: {split_stats}")


if __name__ == "__main__":
    main()
