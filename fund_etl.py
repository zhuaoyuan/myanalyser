from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

import akshare as ak
import pandas as pd

PURCHASE_COLUMNS = [
    "基金代码",
    "基金简称",
    "申购状态",
    "赎回状态",
    "下一开放日",
    "购买起点",
    "日累计限定金额",
    "手续费",
]

OVERVIEW_COLUMNS = [
    "基金代码",
    "基金简称",
    "基金全称",
    "基金类型",
    "发行日期",
    "成立日期/规模",
    "资产规模",
    "份额规模",
    "基金管理人",
    "基金托管人",
    "基金经理人",
    "成立来分红",
    "管理费率",
    "托管费率",
    "销售服务费率",
    "最高认购费率",
    "业绩比较基准",
    "跟踪标的",
]

NAV_COLUMNS = ["基金代码", "净值日期", "累计净值"]


@dataclass
class RetryConfig:
    max_retries: int = 3
    retry_sleep_seconds: float = 1.0


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_str_code(value: object) -> str:
    return str(value).strip().zfill(6)


def _extract_scalar(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns or df.empty:
        return ""
    value = df.iloc[0][column]
    if pd.isna(value):
        return ""
    return str(value).strip()


def _with_retry(call: Callable[[], pd.DataFrame], cfg: RetryConfig, code: str, stage: str) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            return call()
        except Exception as err:  # noqa: BLE001
            last_err = err
            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_sleep_seconds * attempt)
    raise RuntimeError(f"{stage} failed for {code}: {last_err}") from last_err


def _append_failure_log(log_path: Path, code: str, stage: str, error: str) -> None:
    _ensure_dir(log_path.parent)
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "code": code,
        "error": error,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def verify_interfaces(sample_code: str = "015641", nav_code: str = "166009") -> dict:
    purchase_df = ak.fund_purchase_em()
    overview_df = ak.fund_overview_em(symbol=sample_code)
    nav_df = ak.fund_open_fund_info_em(symbol=nav_code, indicator="累计净值走势")

    overview_cols = set(overview_df.columns)
    missing_overview = [
        name
        for name in [
            "基金代码",
            "基金简称",
            "基金全称",
            "基金类型",
            "发行日期",
            "成立日期/规模",
            "份额规模",
            "基金管理人",
            "基金托管人",
            "基金经理人",
            "成立来分红",
            "管理费率",
            "托管费率",
            "销售服务费率",
            "最高认购费率",
            "业绩比较基准",
            "跟踪标的",
        ]
        if name not in overview_cols
    ]

    # AkShare 当前返回列名是“净资产规模”，需求中的“资产规模”在落盘时做映射。
    asset_scale_source_column = "净资产规模" if "净资产规模" in overview_cols else ""

    result = {
        "fund_purchase_em": {
            "rows": int(len(purchase_df)),
            "columns": list(purchase_df.columns),
            "required_columns_present": all(col in purchase_df.columns for col in PURCHASE_COLUMNS),
            "missing_required": [col for col in PURCHASE_COLUMNS if col not in purchase_df.columns],
        },
        "fund_overview_em": {
            "rows": int(len(overview_df)),
            "columns": list(overview_df.columns),
            "missing_required_except_asset_scale": missing_overview,
            "asset_scale_source_column": asset_scale_source_column,
        },
        "fund_open_fund_info_em": {
            "rows": int(len(nav_df)),
            "columns": list(nav_df.columns),
            "required_columns_present": all(col in nav_df.columns for col in ["净值日期", "累计净值"]),
            "missing_required": [col for col in ["净值日期", "累计净值"] if col not in nav_df.columns],
        },
    }
    return result


def run_step1_purchase(output_csv: Path) -> pd.DataFrame:
    df = ak.fund_purchase_em().copy()
    missing = [c for c in PURCHASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"step1 missing columns: {missing}")
    df["基金代码"] = df["基金代码"].map(_safe_str_code)
    result_df = df[PURCHASE_COLUMNS].copy()
    _ensure_dir(output_csv.parent)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return result_df


def _load_codes_from_purchase(purchase_csv: Path) -> list[str]:
    df = pd.read_csv(purchase_csv, dtype={"基金代码": str})
    if "基金代码" not in df.columns:
        raise ValueError(f"purchase file has no 基金代码: {purchase_csv}")
    return sorted({_safe_str_code(code) for code in df["基金代码"].dropna().tolist()})


def _load_done_codes_from_overview(overview_csv: Path) -> set[str]:
    if not overview_csv.exists():
        return set()
    df = pd.read_csv(overview_csv, dtype={"基金代码": str})
    if "基金代码" not in df.columns:
        return set()
    return {_safe_str_code(code) for code in df["基金代码"].dropna().tolist()}


def _normalize_overview(df: pd.DataFrame, code: str) -> pd.DataFrame:
    normalized = pd.DataFrame(
        [
            {
                "基金代码": _safe_str_code(code),
                "基金简称": _extract_scalar(df, "基金简称"),
                "基金全称": _extract_scalar(df, "基金全称"),
                "基金类型": _extract_scalar(df, "基金类型"),
                "发行日期": _extract_scalar(df, "发行日期"),
                "成立日期/规模": _extract_scalar(df, "成立日期/规模"),
                "资产规模": _extract_scalar(df, "净资产规模"),
                "份额规模": _extract_scalar(df, "份额规模"),
                "基金管理人": _extract_scalar(df, "基金管理人"),
                "基金托管人": _extract_scalar(df, "基金托管人"),
                "基金经理人": _extract_scalar(df, "基金经理人"),
                "成立来分红": _extract_scalar(df, "成立来分红"),
                "管理费率": _extract_scalar(df, "管理费率"),
                "托管费率": _extract_scalar(df, "托管费率"),
                "销售服务费率": _extract_scalar(df, "销售服务费率"),
                "最高认购费率": _extract_scalar(df, "最高认购费率"),
                "业绩比较基准": _extract_scalar(df, "业绩比较基准"),
                "跟踪标的": _extract_scalar(df, "跟踪标的"),
            }
        ],
        columns=OVERVIEW_COLUMNS,
    )
    return normalized


def run_step2_overview(
    purchase_csv: Path,
    overview_csv: Path,
    fail_log: Path,
    retry_cfg: RetryConfig,
    only_codes: Sequence[str] | None = None,
) -> dict:
    all_codes = _load_codes_from_purchase(purchase_csv)
    done_codes = _load_done_codes_from_overview(overview_csv)

    if only_codes is not None:
        requested = {_safe_str_code(code) for code in only_codes}
        target_codes = [code for code in all_codes if code in requested]
    else:
        target_codes = all_codes

    to_fetch = [code for code in target_codes if code not in done_codes]

    frames: list[pd.DataFrame] = []
    success = 0
    failed = 0

    for code in to_fetch:
        try:
            raw_df = _with_retry(
                lambda c=code: ak.fund_overview_em(symbol=c),
                cfg=retry_cfg,
                code=code,
                stage="step2_overview",
            )
            one = _normalize_overview(raw_df, code)
            frames.append(one)
            success += 1
        except Exception as err:  # noqa: BLE001
            failed += 1
            _append_failure_log(fail_log, code=code, stage="step2_overview", error=str(err))

    _ensure_dir(overview_csv.parent)
    if frames:
        new_df = pd.concat(frames, ignore_index=True)
        write_header = not overview_csv.exists()
        new_df.to_csv(
            overview_csv,
            mode="a",
            index=False,
            header=write_header,
            encoding="utf-8-sig",
        )

    return {
        "total_codes": len(target_codes),
        "already_done": len(target_codes) - len(to_fetch),
        "fetched": success,
        "failed": failed,
    }


def _load_done_codes_from_nav(nav_dir: Path) -> set[str]:
    if not nav_dir.exists():
        return set()
    done = set()
    for path in nav_dir.glob("*.csv"):
        done.add(path.stem)
    return done


def _normalize_nav(df: pd.DataFrame, code: str) -> pd.DataFrame:
    if "净值日期" not in df.columns or "累计净值" not in df.columns:
        raise ValueError(f"step3 missing columns for {code}: {df.columns.tolist()}")
    out = df[["净值日期", "累计净值"]].copy()
    out.insert(0, "基金代码", _safe_str_code(code))
    out["净值日期"] = pd.to_datetime(out["净值日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["净值日期", "累计净值"])
    out = out[NAV_COLUMNS]
    return out


def run_step3_nav(
    purchase_csv: Path,
    nav_dir: Path,
    fail_log: Path,
    retry_cfg: RetryConfig,
    only_codes: Sequence[str] | None = None,
) -> dict:
    all_codes = _load_codes_from_purchase(purchase_csv)
    done_codes = _load_done_codes_from_nav(nav_dir)

    if only_codes is not None:
        requested = {_safe_str_code(code) for code in only_codes}
        target_codes = [code for code in all_codes if code in requested]
    else:
        target_codes = all_codes

    to_fetch = [code for code in target_codes if code not in done_codes]

    _ensure_dir(nav_dir)

    success = 0
    failed = 0
    total_rows = 0

    for code in to_fetch:
        try:
            raw_df = _with_retry(
                lambda c=code: ak.fund_open_fund_info_em(symbol=c, indicator="累计净值走势"),
                cfg=retry_cfg,
                code=code,
                stage="step3_nav",
            )
            one = _normalize_nav(raw_df, code)
            out_path = nav_dir / f"{code}.csv"
            one.to_csv(out_path, index=False, encoding="utf-8-sig")
            success += 1
            total_rows += len(one)
        except Exception as err:  # noqa: BLE001
            failed += 1
            _append_failure_log(fail_log, code=code, stage="step3_nav", error=str(err))

    return {
        "total_codes": len(target_codes),
        "already_done": len(target_codes) - len(to_fetch),
        "fetched": success,
        "failed": failed,
        "rows_written": total_rows,
    }


def _load_failed_codes(log_paths: Iterable[Path], stage: str) -> list[str]:
    failed: set[str] = set()
    for path in log_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("stage") == stage and rec.get("code"):
                    failed.add(_safe_str_code(rec["code"]))
    return sorted(failed)


def _default_paths(base_dir: Path) -> dict[str, Path]:
    return {
        "purchase_csv": base_dir / "fund_purchase_samples.csv",
        "overview_csv": base_dir / "fund_overview.csv",
        "nav_dir": base_dir / "fund_nav_by_code",
        "verify_json": base_dir / "verify_report.json",
        "fail_overview_log": base_dir / "failed_overview.jsonl",
        "fail_nav_log": base_dir / "failed_nav.jsonl",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AkShare 基金数据采集脚本")
    parser.add_argument("--base-dir", default="myanalyser/data/fund_etl", help="输出目录")
    parser.add_argument(
        "--mode",
        choices=["all", "step1", "step2", "step3", "verify", "retry-overview", "retry-nav", "retry-all"],
        default="all",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    paths = _default_paths(base_dir)
    retry_cfg = RetryConfig(max_retries=args.max_retries, retry_sleep_seconds=args.retry_sleep)

    if args.mode in {"verify", "all"}:
        report = verify_interfaces()
        _ensure_dir(paths["verify_json"].parent)
        with paths["verify_json"].open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[verify] report saved: {paths['verify_json']}")

    if args.mode in {"step1", "all"}:
        df = run_step1_purchase(paths["purchase_csv"])
        print(f"[step1] rows={len(df)} -> {paths['purchase_csv']}")

    if args.mode in {"step2", "all"}:
        summary = run_step2_overview(
            purchase_csv=paths["purchase_csv"],
            overview_csv=paths["overview_csv"],
            fail_log=paths["fail_overview_log"],
            retry_cfg=retry_cfg,
        )
        print(f"[step2] {summary}")

    if args.mode in {"step3", "all"}:
        summary = run_step3_nav(
            purchase_csv=paths["purchase_csv"],
            nav_dir=paths["nav_dir"],
            fail_log=paths["fail_nav_log"],
            retry_cfg=retry_cfg,
        )
        print(f"[step3] {summary}")

    if args.mode in {"retry-overview", "retry-all"}:
        failed_codes = _load_failed_codes([paths["fail_overview_log"]], stage="step2_overview")
        summary = run_step2_overview(
            purchase_csv=paths["purchase_csv"],
            overview_csv=paths["overview_csv"],
            fail_log=paths["fail_overview_log"],
            retry_cfg=retry_cfg,
            only_codes=failed_codes,
        )
        print(f"[retry-overview] failed_codes={len(failed_codes)} summary={summary}")

    if args.mode in {"retry-nav", "retry-all"}:
        failed_codes = _load_failed_codes([paths["fail_nav_log"]], stage="step3_nav")
        summary = run_step3_nav(
            purchase_csv=paths["purchase_csv"],
            nav_dir=paths["nav_dir"],
            fail_log=paths["fail_nav_log"],
            retry_cfg=retry_cfg,
            only_codes=failed_codes,
        )
        print(f"[retry-nav] failed_codes={len(failed_codes)} summary={summary}")


if __name__ == "__main__":
    main()
