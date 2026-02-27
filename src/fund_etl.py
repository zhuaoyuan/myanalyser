from __future__ import annotations

import argparse
import json
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

from project_paths import default_run_id, project_root
from validators.validate_pipeline_artifacts import validate_stage_or_raise

try:
    import akshare as ak
except Exception:  # noqa: BLE001
    # In restricted environments (e.g. sandbox CI), importing akshare may fail
    # due to optional binary deps. Keep module importable for mocked tests.
    ak = types.SimpleNamespace(
        fund_purchase_em=None,
        fund_overview_em=None,
        fund_open_fund_info_em=None,
        fund_announcement_personnel_em=None,
    )

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

UNIT_NAV_COLUMNS = ["基金代码", "净值日期", "单位净值", "日增长率"]
BONUS_COLUMNS = ["基金代码", "年份", "权益登记日", "除息日", "每份分红", "分红发放日"]
SPLIT_COLUMNS = ["基金代码", "年份", "拆分折算日", "拆分类型", "拆分折算比例"]
PERSONNEL_COLUMNS = ["基金代码", "公告标题", "基金名称", "公告日期", "报告ID"]
CUM_RETURN_COLUMNS = ["基金代码", "日期", "累计收益率"]


@dataclass
class RetryConfig:
    max_retries: int = 3
    retry_sleep_seconds: float = 1.0


@dataclass
class ProgressConfig:
    print_interval_seconds: float = 5.0


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


def _fetch_personnel_announcement(symbol: str) -> pd.DataFrame:
    try:
        return ak.fund_announcement_personnel_em(symbol=symbol)
    except ValueError as err:
        # AkShare 在部分基金“人事公告无数据”时会抛 Length mismatch，按空结果处理。
        if "Length mismatch" in str(err):
            return pd.DataFrame(columns=PERSONNEL_COLUMNS)
        raise


def verify_interfaces(sample_code: str = "015641", nav_code: str = "166009") -> dict:
    purchase_df = ak.fund_purchase_em()
    overview_df = ak.fund_overview_em(symbol=sample_code)
    nav_df = ak.fund_open_fund_info_em(symbol=nav_code, indicator="单位净值走势")
    bonus_df = ak.fund_open_fund_info_em(symbol=nav_code, indicator="分红送配详情")
    split_df = ak.fund_open_fund_info_em(symbol=nav_code, indicator="拆分详情")
    cum_return_df = ak.fund_open_fund_info_em(symbol=nav_code, indicator="累计收益率走势", period="成立来")
    personnel_df = _fetch_personnel_announcement(symbol=sample_code)

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
            "required_columns_present": all(col in nav_df.columns for col in ["净值日期", "单位净值", "日增长率"]),
            "missing_required": [col for col in ["净值日期", "单位净值", "日增长率"] if col not in nav_df.columns],
        },
        "fund_open_fund_bonus_em": {
            "rows": int(len(bonus_df)),
            "columns": list(bonus_df.columns),
            "required_columns_present": all(col in bonus_df.columns for col in ["年份", "权益登记日", "除息日", "每份分红", "分红发放日"]),
            "missing_required": [col for col in ["年份", "权益登记日", "除息日", "每份分红", "分红发放日"] if col not in bonus_df.columns],
        },
        "fund_open_fund_split_em": {
            "rows": int(len(split_df)),
            "columns": list(split_df.columns),
            "required_columns_present": all(col in split_df.columns for col in ["年份", "拆分折算日", "拆分类型", "拆分折算比例"]),
            "missing_required": [col for col in ["年份", "拆分折算日", "拆分类型", "拆分折算比例"] if col not in split_df.columns],
        },
        "fund_open_fund_cum_return_em": {
            "rows": int(len(cum_return_df)),
            "columns": list(cum_return_df.columns),
            "required_columns_present": all(col in cum_return_df.columns for col in ["日期", "累计收益率"]),
            "missing_required": [col for col in ["日期", "累计收益率"] if col not in cum_return_df.columns],
        },
        "fund_announcement_personnel_em": {
            "rows": int(len(personnel_df)),
            "columns": list(personnel_df.columns),
            "required_columns_present": all(
                col in personnel_df.columns for col in ["基金代码", "公告标题", "基金名称", "公告日期", "报告ID"]
            ),
            "missing_required": [
                col for col in ["基金代码", "公告标题", "基金名称", "公告日期", "报告ID"] if col not in personnel_df.columns
            ],
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


def _print_progress(
    stage: str,
    processed: int,
    total: int,
    success: int,
    failed: int,
    already_done: int,
) -> None:
    print(
        f"[{stage}] progress processed={processed}/{total} success={success} "
        f"failed={failed} already_done={already_done}"
    )


def run_step2_overview(
    purchase_csv: Path,
    overview_csv: Path,
    fail_log: Path,
    retry_cfg: RetryConfig,
    max_workers: int = 8,
    progress_cfg: ProgressConfig | None = None,
    only_codes: Sequence[str] | None = None,
) -> dict:
    progress_cfg = progress_cfg or ProgressConfig()
    all_codes = _load_codes_from_purchase(purchase_csv)
    done_codes = _load_done_codes_from_overview(overview_csv)

    if only_codes is not None:
        requested = {_safe_str_code(code) for code in only_codes}
        target_codes = [code for code in all_codes if code in requested]
    else:
        target_codes = all_codes

    to_fetch = [code for code in target_codes if code not in done_codes]

    success = 0
    failed = 0
    processed = len(target_codes) - len(to_fetch)
    last_print_ts = time.monotonic()

    _ensure_dir(overview_csv.parent)
    write_header = not overview_csv.exists() or overview_csv.stat().st_size == 0

    if to_fetch:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {
                executor.submit(
                    _with_retry,
                    lambda c=code: ak.fund_overview_em(symbol=c),
                    retry_cfg,
                    code,
                    "step2_overview",
                ): code
                for code in to_fetch
            }
            for future in as_completed(futures):
                code = futures[future]
                processed += 1
                try:
                    raw_df = future.result()
                    one = _normalize_overview(raw_df, code)
                    one.to_csv(
                        overview_csv,
                        mode="a",
                        index=False,
                        header=write_header,
                        encoding="utf-8-sig",
                    )
                    write_header = False
                    success += 1
                except Exception as err:  # noqa: BLE001
                    failed += 1
                    _append_failure_log(fail_log, code=code, stage="step2_overview", error=str(err))

                now = time.monotonic()
                if now - last_print_ts >= progress_cfg.print_interval_seconds:
                    _print_progress(
                        stage="step2",
                        processed=processed,
                        total=len(target_codes),
                        success=success,
                        failed=failed,
                        already_done=len(target_codes) - len(to_fetch),
                    )
                    last_print_ts = now

    _print_progress(
        stage="step2",
        processed=len(target_codes),
        total=len(target_codes),
        success=success,
        failed=failed,
        already_done=len(target_codes) - len(to_fetch),
    )

    return {
        "total_codes": len(target_codes),
        "already_done": len(target_codes) - len(to_fetch),
        "fetched": success,
        "failed": failed,
    }


def _load_done_codes_from_dir(data_dir: Path) -> set[str]:
    if not data_dir.exists():
        return set()
    done = set()
    for path in data_dir.glob("*.csv"):
        done.add(path.stem)
    return done


def _normalize_nav(df: pd.DataFrame, code: str) -> pd.DataFrame:
    if "净值日期" not in df.columns or "单位净值" not in df.columns:
        raise ValueError(f"step3 missing columns for {code}: {df.columns.tolist()}")
    out = df.copy()
    for col in ["净值日期", "单位净值", "日增长率"]:
        if col not in out.columns:
            out[col] = ""
    out = out[["净值日期", "单位净值", "日增长率"]].copy()
    out.insert(0, "基金代码", _safe_str_code(code))
    out["净值日期"] = pd.to_datetime(out["净值日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["单位净值"] = pd.to_numeric(out["单位净值"], errors="coerce")
    out["日增长率"] = pd.to_numeric(out["日增长率"], errors="coerce")
    out = out.dropna(subset=["净值日期", "单位净值"])
    out = out[UNIT_NAV_COLUMNS]
    return out


def run_step3_nav(
    purchase_csv: Path,
    nav_dir: Path,
    fail_log: Path,
    retry_cfg: RetryConfig,
    max_workers: int = 8,
    progress_cfg: ProgressConfig | None = None,
    only_codes: Sequence[str] | None = None,
) -> dict:
    progress_cfg = progress_cfg or ProgressConfig()
    all_codes = _load_codes_from_purchase(purchase_csv)
    done_codes = _load_done_codes_from_dir(nav_dir)

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
    processed = len(target_codes) - len(to_fetch)
    last_print_ts = time.monotonic()

    if max_workers > 1:
        print("[step3] ignore max_workers, run serial requests by design")

    for code in to_fetch:
        processed += 1
        try:
            raw_df = _with_retry(
                lambda c=code: ak.fund_open_fund_info_em(symbol=c, indicator="单位净值走势"),
                retry_cfg,
                code,
                "step3_nav",
            )
            one = _normalize_nav(raw_df, code)
            out_path = nav_dir / f"{code}.csv"
            one.to_csv(out_path, index=False, encoding="utf-8-sig")
            success += 1
            total_rows += len(one)
        except Exception as err:  # noqa: BLE001
            failed += 1
            _append_failure_log(fail_log, code=code, stage="step3_nav", error=str(err))

        now = time.monotonic()
        if now - last_print_ts >= progress_cfg.print_interval_seconds:
            _print_progress(
                stage="step3",
                processed=processed,
                total=len(target_codes),
                success=success,
                failed=failed,
                already_done=len(target_codes) - len(to_fetch),
            )
            last_print_ts = now

    _print_progress(
        stage="step3",
        processed=len(target_codes),
        total=len(target_codes),
        success=success,
        failed=failed,
        already_done=len(target_codes) - len(to_fetch),
    )

    return {
        "total_codes": len(target_codes),
        "already_done": len(target_codes) - len(to_fetch),
        "fetched": success,
        "failed": failed,
        "rows_written": total_rows,
    }


def _normalize_bonus(df: pd.DataFrame, code: str) -> pd.DataFrame:
    out = df.copy()
    for col in ["年份", "权益登记日", "除息日", "每份分红", "分红发放日"]:
        if col not in out.columns:
            out[col] = ""
    out = out[["年份", "权益登记日", "除息日", "每份分红", "分红发放日"]].copy()
    out.insert(0, "基金代码", _safe_str_code(code))
    for col in ["权益登记日", "除息日", "分红发放日"]:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out[BONUS_COLUMNS]


def _normalize_split(df: pd.DataFrame, code: str) -> pd.DataFrame:
    out = df.copy()
    for col in ["年份", "拆分折算日", "拆分类型", "拆分折算比例"]:
        if col not in out.columns:
            out[col] = ""
    out = out[["年份", "拆分折算日", "拆分类型", "拆分折算比例"]].copy()
    out.insert(0, "基金代码", _safe_str_code(code))
    out["拆分折算日"] = pd.to_datetime(out["拆分折算日"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out[SPLIT_COLUMNS]


def _run_step_fund_info_serial(
    *,
    purchase_csv: Path,
    out_dir: Path,
    fail_log: Path,
    retry_cfg: RetryConfig,
    progress_cfg: ProgressConfig,
    indicator: str,
    period: str | None,
    stage_name: str,
    normalize_fn: Callable[[pd.DataFrame, str], pd.DataFrame],
    only_codes: Sequence[str] | None,
) -> dict:
    all_codes = _load_codes_from_purchase(purchase_csv)
    done_codes = _load_done_codes_from_dir(out_dir)

    if only_codes is not None:
        requested = {_safe_str_code(code) for code in only_codes}
        target_codes = [code for code in all_codes if code in requested]
    else:
        target_codes = all_codes

    to_fetch = [code for code in target_codes if code not in done_codes]
    _ensure_dir(out_dir)

    success = 0
    failed = 0
    total_rows = 0
    processed = len(target_codes) - len(to_fetch)
    last_print_ts = time.monotonic()

    for code in to_fetch:
        processed += 1
        try:
            raw_df = _with_retry(
                (
                    lambda c=code, ind=indicator, p=period: ak.fund_open_fund_info_em(symbol=c, indicator=ind, period=p)
                    if p
                    else ak.fund_open_fund_info_em(symbol=c, indicator=ind)
                ),
                retry_cfg,
                code,
                stage_name,
            )
            one = normalize_fn(raw_df, code)
            out_path = out_dir / f"{code}.csv"
            one.to_csv(out_path, index=False, encoding="utf-8-sig")
            success += 1
            total_rows += len(one)
        except Exception as err:  # noqa: BLE001
            failed += 1
            _append_failure_log(fail_log, code=code, stage=stage_name, error=str(err))

        now = time.monotonic()
        if now - last_print_ts >= progress_cfg.print_interval_seconds:
            _print_progress(
                stage=stage_name.replace("_", "-"),
                processed=processed,
                total=len(target_codes),
                success=success,
                failed=failed,
                already_done=len(target_codes) - len(to_fetch),
            )
            last_print_ts = now

    _print_progress(
        stage=stage_name.replace("_", "-"),
        processed=len(target_codes),
        total=len(target_codes),
        success=success,
        failed=failed,
        already_done=len(target_codes) - len(to_fetch),
    )

    return {
        "total_codes": len(target_codes),
        "already_done": len(target_codes) - len(to_fetch),
        "fetched": success,
        "failed": failed,
        "rows_written": total_rows,
    }


def run_step4_bonus(
    purchase_csv: Path,
    bonus_dir: Path,
    fail_log: Path,
    retry_cfg: RetryConfig,
    progress_cfg: ProgressConfig | None = None,
    only_codes: Sequence[str] | None = None,
) -> dict:
    progress_cfg = progress_cfg or ProgressConfig()
    return _run_step_fund_info_serial(
        purchase_csv=purchase_csv,
        out_dir=bonus_dir,
        fail_log=fail_log,
        retry_cfg=retry_cfg,
        progress_cfg=progress_cfg,
        indicator="分红送配详情",
        period=None,
        stage_name="step4_bonus",
        normalize_fn=_normalize_bonus,
        only_codes=only_codes,
    )


def run_step5_split(
    purchase_csv: Path,
    split_dir: Path,
    fail_log: Path,
    retry_cfg: RetryConfig,
    progress_cfg: ProgressConfig | None = None,
    only_codes: Sequence[str] | None = None,
) -> dict:
    progress_cfg = progress_cfg or ProgressConfig()
    return _run_step_fund_info_serial(
        purchase_csv=purchase_csv,
        out_dir=split_dir,
        fail_log=fail_log,
        retry_cfg=retry_cfg,
        progress_cfg=progress_cfg,
        indicator="拆分详情",
        period=None,
        stage_name="step5_split",
        normalize_fn=_normalize_split,
        only_codes=only_codes,
    )


def _normalize_personnel(df: pd.DataFrame, code: str) -> pd.DataFrame:
    out = df.copy()
    for col in ["公告标题", "基金名称", "公告日期", "报告ID"]:
        if col not in out.columns:
            out[col] = ""
    out = out[["公告标题", "基金名称", "公告日期", "报告ID"]].copy()
    out.insert(0, "基金代码", _safe_str_code(code))
    out["公告日期"] = pd.to_datetime(out["公告日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out[PERSONNEL_COLUMNS]


def run_step6_personnel(
    purchase_csv: Path,
    personnel_dir: Path,
    fail_log: Path,
    retry_cfg: RetryConfig,
    progress_cfg: ProgressConfig | None = None,
    only_codes: Sequence[str] | None = None,
) -> dict:
    progress_cfg = progress_cfg or ProgressConfig()
    all_codes = _load_codes_from_purchase(purchase_csv)
    done_codes = _load_done_codes_from_dir(personnel_dir)

    if only_codes is not None:
        requested = {_safe_str_code(code) for code in only_codes}
        target_codes = [code for code in all_codes if code in requested]
    else:
        target_codes = all_codes

    to_fetch = [code for code in target_codes if code not in done_codes]
    _ensure_dir(personnel_dir)

    success = 0
    failed = 0
    total_rows = 0
    processed = len(target_codes) - len(to_fetch)
    last_print_ts = time.monotonic()

    for code in to_fetch:
        processed += 1
        try:
            raw_df = _with_retry(
                lambda c=code: _fetch_personnel_announcement(symbol=c),
                retry_cfg,
                code,
                "step6_personnel",
            )
            one = _normalize_personnel(raw_df, code)
            out_path = personnel_dir / f"{code}.csv"
            one.to_csv(out_path, index=False, encoding="utf-8-sig")
            success += 1
            total_rows += len(one)
        except Exception as err:  # noqa: BLE001
            failed += 1
            _append_failure_log(fail_log, code=code, stage="step6_personnel", error=str(err))

        now = time.monotonic()
        if now - last_print_ts >= progress_cfg.print_interval_seconds:
            _print_progress(
                stage="step6-personnel",
                processed=processed,
                total=len(target_codes),
                success=success,
                failed=failed,
                already_done=len(target_codes) - len(to_fetch),
            )
            last_print_ts = now

    _print_progress(
        stage="step6-personnel",
        processed=len(target_codes),
        total=len(target_codes),
        success=success,
        failed=failed,
        already_done=len(target_codes) - len(to_fetch),
    )

    return {
        "total_codes": len(target_codes),
        "already_done": len(target_codes) - len(to_fetch),
        "fetched": success,
        "failed": failed,
        "rows_written": total_rows,
    }


def _normalize_cum_return(df: pd.DataFrame, code: str) -> pd.DataFrame:
    out = df.copy()
    for col in ["日期", "累计收益率"]:
        if col not in out.columns:
            out[col] = ""
    out = out[["日期", "累计收益率"]].copy()
    out.insert(0, "基金代码", _safe_str_code(code))
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["累计收益率"] = pd.to_numeric(out["累计收益率"], errors="coerce")
    out = out.dropna(subset=["日期", "累计收益率"])
    return out[CUM_RETURN_COLUMNS]


def run_step7_cum_return(
    purchase_csv: Path,
    cum_return_dir: Path,
    fail_log: Path,
    retry_cfg: RetryConfig,
    progress_cfg: ProgressConfig | None = None,
    only_codes: Sequence[str] | None = None,
) -> dict:
    progress_cfg = progress_cfg or ProgressConfig()
    return _run_step_fund_info_serial(
        purchase_csv=purchase_csv,
        out_dir=cum_return_dir,
        fail_log=fail_log,
        retry_cfg=retry_cfg,
        progress_cfg=progress_cfg,
        indicator="累计收益率走势",
        period="成立来",
        stage_name="step7_cum_return",
        normalize_fn=_normalize_cum_return,
        only_codes=only_codes,
    )


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


def _default_paths(base_dir: Path, logs_dir: Path) -> dict[str, Path]:
    return {
        "purchase_csv": base_dir / "fund_purchase.csv",
        "overview_csv": base_dir / "fund_overview.csv",
        "nav_dir": base_dir / "fund_nav_by_code",
        "bonus_dir": base_dir / "fund_bonus_by_code",
        "split_dir": base_dir / "fund_split_by_code",
        "personnel_dir": base_dir / "fund_personnel_by_code",
        "cum_return_dir": base_dir / "fund_cum_return_by_code",
        "verify_json": base_dir / "verify_report.json",
        "fail_overview_log": logs_dir / "failed_overview.jsonl",
        "fail_nav_log": logs_dir / "failed_nav.jsonl",
        "fail_bonus_log": logs_dir / "failed_bonus.jsonl",
        "fail_split_log": logs_dir / "failed_split.jsonl",
        "fail_personnel_log": logs_dir / "failed_personnel.jsonl",
        "fail_cum_return_log": logs_dir / "failed_cum_return.jsonl",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AkShare 基金数据采集脚本")
    parser.add_argument(
        "--base-dir",
        default=None,
        help="ETL 输出目录（默认 data/versions/{run_id}/fund_etl）",
    )
    parser.add_argument("--run-id", default=None, help="数据版本目录名（默认 YYYYMMDD_HHMMSS[可选后缀]）")
    parser.add_argument("--run-id-suffix", default=None, help="run_id 后缀描述，会拼接到时间戳后")
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "step1",
            "step2",
            "step3",
            "step4",
            "step5",
            "step6",
            "step7",
            "verify",
            "retry-overview",
            "retry-nav",
            "retry-bonus",
            "retry-split",
            "retry-personnel",
            "retry-cum-return",
            "retry-all",
        ],
        default="all",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--progress-interval", type=float, default=5.0)
    args = parser.parse_args()

    root = project_root()
    run_id = args.run_id or default_run_id(args.run_id_suffix)
    base_dir = Path(args.base_dir).resolve() if args.base_dir else (root / "data" / "versions" / run_id / "fund_etl")
    logs_dir = root / "data" / "versions" / run_id / "logs"
    paths = _default_paths(base_dir=base_dir, logs_dir=logs_dir)
    retry_cfg = RetryConfig(max_retries=args.max_retries, retry_sleep_seconds=args.retry_sleep)
    progress_cfg = ProgressConfig(print_interval_seconds=args.progress_interval)
    print(f"[run] run_id={run_id}")
    print(f"[run] fund_etl_dir={base_dir}")
    print(f"[run] logs_dir={logs_dir}")

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
        validate_stage_or_raise("fund_etl_step2_input", purchase_csv=paths["purchase_csv"])
        summary = run_step2_overview(
            purchase_csv=paths["purchase_csv"],
            overview_csv=paths["overview_csv"],
            fail_log=paths["fail_overview_log"],
            retry_cfg=retry_cfg,
            max_workers=args.max_workers,
            progress_cfg=progress_cfg,
        )
        print(f"[step2] {summary}")

    if args.mode in {"step3", "all"}:
        validate_stage_or_raise("fund_etl_step3_input", purchase_csv=paths["purchase_csv"])
        summary = run_step3_nav(
            purchase_csv=paths["purchase_csv"],
            nav_dir=paths["nav_dir"],
            fail_log=paths["fail_nav_log"],
            retry_cfg=retry_cfg,
            max_workers=args.max_workers,
            progress_cfg=progress_cfg,
        )
        print(f"[step3] {summary}")

    if args.mode in {"step4", "all"}:
        validate_stage_or_raise("fund_etl_step4_input", purchase_csv=paths["purchase_csv"])
        summary = run_step4_bonus(
            purchase_csv=paths["purchase_csv"],
            bonus_dir=paths["bonus_dir"],
            fail_log=paths["fail_bonus_log"],
            retry_cfg=retry_cfg,
            progress_cfg=progress_cfg,
        )
        print(f"[step4] {summary}")

    if args.mode in {"step5", "all"}:
        validate_stage_or_raise("fund_etl_step5_input", purchase_csv=paths["purchase_csv"])
        summary = run_step5_split(
            purchase_csv=paths["purchase_csv"],
            split_dir=paths["split_dir"],
            fail_log=paths["fail_split_log"],
            retry_cfg=retry_cfg,
            progress_cfg=progress_cfg,
        )
        print(f"[step5] {summary}")

    if args.mode in {"step6", "all"}:
        validate_stage_or_raise("fund_etl_step6_input", purchase_csv=paths["purchase_csv"])
        summary = run_step6_personnel(
            purchase_csv=paths["purchase_csv"],
            personnel_dir=paths["personnel_dir"],
            fail_log=paths["fail_personnel_log"],
            retry_cfg=retry_cfg,
            progress_cfg=progress_cfg,
        )
        print(f"[step6] {summary}")

    if args.mode in {"step7", "all"}:
        validate_stage_or_raise("fund_etl_step7_input", purchase_csv=paths["purchase_csv"])
        summary = run_step7_cum_return(
            purchase_csv=paths["purchase_csv"],
            cum_return_dir=paths["cum_return_dir"],
            fail_log=paths["fail_cum_return_log"],
            retry_cfg=retry_cfg,
            progress_cfg=progress_cfg,
        )
        print(f"[step7] {summary}")

    if args.mode in {"retry-overview", "retry-all"}:
        validate_stage_or_raise("fund_etl_step2_input", purchase_csv=paths["purchase_csv"])
        failed_codes = _load_failed_codes([paths["fail_overview_log"]], stage="step2_overview")
        summary = run_step2_overview(
            purchase_csv=paths["purchase_csv"],
            overview_csv=paths["overview_csv"],
            fail_log=paths["fail_overview_log"],
            retry_cfg=retry_cfg,
            max_workers=args.max_workers,
            progress_cfg=progress_cfg,
            only_codes=failed_codes,
        )
        print(f"[retry-overview] failed_codes={len(failed_codes)} summary={summary}")

    if args.mode in {"retry-nav", "retry-all"}:
        validate_stage_or_raise("fund_etl_step3_input", purchase_csv=paths["purchase_csv"])
        failed_codes = _load_failed_codes([paths["fail_nav_log"]], stage="step3_nav")
        summary = run_step3_nav(
            purchase_csv=paths["purchase_csv"],
            nav_dir=paths["nav_dir"],
            fail_log=paths["fail_nav_log"],
            retry_cfg=retry_cfg,
            max_workers=args.max_workers,
            progress_cfg=progress_cfg,
            only_codes=failed_codes,
        )
        print(f"[retry-nav] failed_codes={len(failed_codes)} summary={summary}")

    if args.mode in {"retry-bonus", "retry-all"}:
        validate_stage_or_raise("fund_etl_step4_input", purchase_csv=paths["purchase_csv"])
        failed_codes = _load_failed_codes([paths["fail_bonus_log"]], stage="step4_bonus")
        summary = run_step4_bonus(
            purchase_csv=paths["purchase_csv"],
            bonus_dir=paths["bonus_dir"],
            fail_log=paths["fail_bonus_log"],
            retry_cfg=retry_cfg,
            progress_cfg=progress_cfg,
            only_codes=failed_codes,
        )
        print(f"[retry-bonus] failed_codes={len(failed_codes)} summary={summary}")

    if args.mode in {"retry-split", "retry-all"}:
        validate_stage_or_raise("fund_etl_step5_input", purchase_csv=paths["purchase_csv"])
        failed_codes = _load_failed_codes([paths["fail_split_log"]], stage="step5_split")
        summary = run_step5_split(
            purchase_csv=paths["purchase_csv"],
            split_dir=paths["split_dir"],
            fail_log=paths["fail_split_log"],
            retry_cfg=retry_cfg,
            progress_cfg=progress_cfg,
            only_codes=failed_codes,
        )
        print(f"[retry-split] failed_codes={len(failed_codes)} summary={summary}")

    if args.mode in {"retry-personnel", "retry-all"}:
        validate_stage_or_raise("fund_etl_step6_input", purchase_csv=paths["purchase_csv"])
        failed_codes = _load_failed_codes([paths["fail_personnel_log"]], stage="step6_personnel")
        summary = run_step6_personnel(
            purchase_csv=paths["purchase_csv"],
            personnel_dir=paths["personnel_dir"],
            fail_log=paths["fail_personnel_log"],
            retry_cfg=retry_cfg,
            progress_cfg=progress_cfg,
            only_codes=failed_codes,
        )
        print(f"[retry-personnel] failed_codes={len(failed_codes)} summary={summary}")

    if args.mode in {"retry-cum-return", "retry-all"}:
        validate_stage_or_raise("fund_etl_step7_input", purchase_csv=paths["purchase_csv"])
        failed_codes = _load_failed_codes([paths["fail_cum_return_log"]], stage="step7_cum_return")
        summary = run_step7_cum_return(
            purchase_csv=paths["purchase_csv"],
            cum_return_dir=paths["cum_return_dir"],
            fail_log=paths["fail_cum_return_log"],
            retry_cfg=retry_cfg,
            progress_cfg=progress_cfg,
            only_codes=failed_codes,
        )
        print(f"[retry-cum-return] failed_codes={len(failed_codes)} summary={summary}")


if __name__ == "__main__":
    main()
