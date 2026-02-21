from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

OUTPUT_COLUMNS = ["基金代码", "净值日期", "单位净值", "复权净值", "cumulative_factor"]


def _safe_code(value: object) -> str:
    return str(value).strip().zfill(6)


def _parse_amount(text: object) -> float | None:
    if text is None:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", str(text))
    return float(match.group(1)) if match else None


def _parse_split_ratio(text: object) -> float | None:
    if text is None:
        return None
    value = str(text).strip()
    if ":" in value:
        value = value.split(":")[-1].strip()
    try:
        return float(value)
    except ValueError:
        return None


def _append_failure_log(log_path: Path, code: str, error: str) -> None:
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "stage": "adjusted_nav",
        "code": code,
        "error": error,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_dates_from_missing_error(err_msg: str, prefix: str) -> list[pd.Timestamp]:
    if prefix not in err_msg:
        return []
    raw = err_msg.split(":", 1)[-1].strip()
    if not raw:
        return []
    dates = [pd.to_datetime(part.strip(), errors="coerce") for part in raw.split(",")]
    return [date for date in dates if pd.notna(date)]


def calculate_adjusted_nav(df_nav: pd.DataFrame, df_dividend: pd.DataFrame, df_split: pd.DataFrame) -> pd.DataFrame:
    if "净值日期" not in df_nav.columns or "单位净值" not in df_nav.columns:
        raise ValueError("df_nav must contain columns: 净值日期, 单位净值")

    nav = df_nav.copy()
    nav["净值日期"] = pd.to_datetime(nav["净值日期"], errors="coerce")
    nav["单位净值"] = pd.to_numeric(nav["单位净值"], errors="coerce")
    nav = nav.dropna(subset=["净值日期", "单位净值"]).sort_values("净值日期").reset_index(drop=True)

    nav["adj_factor"] = 1.0

    if not df_dividend.empty and "除息日" in df_dividend.columns and "每份分红" in df_dividend.columns:
        dividend = df_dividend.copy()
        dividend["除息日"] = pd.to_datetime(dividend["除息日"], errors="coerce")
        dividend["分红金额"] = dividend["每份分红"].map(_parse_amount)
        dividend = dividend.dropna(subset=["除息日", "分红金额"])
        nav_dates = set(nav["净值日期"].tolist())
        missing_div_dates = sorted({date for date in dividend["除息日"].tolist() if date not in nav_dates})
        if missing_div_dates:
            missing_txt = ",".join(pd.Series(missing_div_dates).dt.strftime("%Y-%m-%d").tolist())
            raise ValueError(f"missing nav data on dividend dates: {missing_txt}")

        for _, row in dividend.iterrows():
            div_date = row["除息日"]
            div_amt = float(row["分红金额"])
            mask = nav["净值日期"] == div_date
            nav_val = float(nav.loc[mask, "单位净值"].iloc[0])
            if nav_val <= 0:
                continue
            factor = (nav_val + div_amt) / nav_val
            # 同日若有多次分红，按份额再投资连续累乘。
            nav.loc[mask, "adj_factor"] = nav.loc[mask, "adj_factor"] * factor

    if not df_split.empty and "拆分折算日" in df_split.columns and "拆分折算比例" in df_split.columns:
        split = df_split.copy()
        split["拆分折算日"] = pd.to_datetime(split["拆分折算日"], errors="coerce")
        split["拆分比例"] = split["拆分折算比例"].map(_parse_split_ratio)
        split = split.dropna(subset=["拆分折算日", "拆分比例"])
        nav_dates = set(nav["净值日期"].tolist())
        missing_split_dates = sorted({date for date in split["拆分折算日"].tolist() if date not in nav_dates})
        if missing_split_dates:
            missing_txt = ",".join(pd.Series(missing_split_dates).dt.strftime("%Y-%m-%d").tolist())
            raise ValueError(f"missing nav data on split dates: {missing_txt}")

        for _, row in split.iterrows():
            split_date = row["拆分折算日"]
            split_ratio = float(row["拆分比例"])
            if split_ratio <= 0:
                continue
            mask = nav["净值日期"] == split_date
            nav.loc[mask, "adj_factor"] = nav.loc[mask, "adj_factor"] * split_ratio

    nav["cumulative_factor"] = nav["adj_factor"].cumprod()
    nav["复权净值"] = nav["单位净值"] * nav["cumulative_factor"]
    nav["净值日期"] = nav["净值日期"].dt.strftime("%Y-%m-%d")
    return nav[["净值日期", "单位净值", "复权净值", "cumulative_factor"]]


def process_one_fund(
    nav_csv: Path,
    bonus_csv: Path,
    split_csv: Path,
    output_csv: Path,
    allow_missing_event_until: pd.Timestamp | None = None,
) -> dict[str, object]:
    nav_df = pd.read_csv(nav_csv, dtype={"基金代码": str})
    if not bonus_csv.exists() :
        raise FileNotFoundError(f"missing bonus files: {bonus_csv}")
    if not split_csv.exists() :
        raise FileNotFoundError(f"missing split files: {split_csv}")
    bonus_df = pd.read_csv(bonus_csv, dtype={"基金代码": str}) if bonus_csv.exists() else pd.DataFrame()
    split_df = pd.read_csv(split_csv, dtype={"基金代码": str}) if split_csv.exists() else pd.DataFrame()

    if nav_df.empty:
        raise ValueError(f"empty nav data: {nav_csv}")

    code = _safe_code(nav_df["基金代码"].iloc[0]) if "基金代码" in nav_df.columns else nav_csv.stem
    while True:
        try:
            result_df = calculate_adjusted_nav(df_nav=nav_df, df_dividend=bonus_df, df_split=split_df)
            break
        except ValueError as err:
            err_msg = str(err)
            handled = False
            if allow_missing_event_until is not None and "missing nav data on dividend dates" in err_msg:
                missing_dates = _extract_dates_from_missing_error(err_msg, "missing nav data on dividend dates")
                if missing_dates and all(date.normalize() <= allow_missing_event_until.normalize() for date in missing_dates):
                    bonus_df = bonus_df.copy()
                    bonus_df["除息日"] = pd.to_datetime(bonus_df["除息日"], errors="coerce")
                    bonus_df = bonus_df[~bonus_df["除息日"].isin(missing_dates)]
                    handled = True
            if allow_missing_event_until is not None and (not handled) and "missing nav data on split dates" in err_msg:
                missing_dates = _extract_dates_from_missing_error(err_msg, "missing nav data on split dates")
                if missing_dates and all(date.normalize() <= allow_missing_event_until.normalize() for date in missing_dates):
                    split_df = split_df.copy()
                    split_df["拆分折算日"] = pd.to_datetime(split_df["拆分折算日"], errors="coerce")
                    split_df = split_df[~split_df["拆分折算日"].isin(missing_dates)]
                    handled = True
            if not handled:
                raise

    result_df.insert(0, "基金代码", code)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return {
        "code": code,
        "rows": int(len(result_df)),
        "start_date": result_df["净值日期"].iloc[0] if not result_df.empty else "",
        "end_date": result_df["净值日期"].iloc[-1] if not result_df.empty else "",
    }


def process_all_funds(
    nav_dir: Path,
    bonus_dir: Path,
    split_dir: Path,
    output_dir: Path,
    codes: list[str] | None = None,
    progress_interval_seconds: float = 5.0,
    allow_missing_event_until: pd.Timestamp | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fail_log = output_dir / "failed_adjusted_nav.jsonl"
    nav_files = sorted(nav_dir.glob("*.csv"))
    if codes:
        wanted = {_safe_code(code) for code in codes}
        nav_files = [path for path in nav_files if _safe_code(path.stem) in wanted]

    processed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    skipped = 0
    total = len(nav_files)
    done = 0
    last_print_ts = 0.0
    last_print_done = -1

    def _print_progress(force: bool = False) -> None:
        nonlocal last_print_ts, last_print_done
        now = time.time()
        should_print = force or now - last_print_ts >= progress_interval_seconds
        if should_print and done == last_print_done:
            return
        if should_print:
            print(
                f"progress: {done}/{total} processed "
                f"(success={len(processed)}, failed={len(failed)}, skipped={skipped})"
            )
            last_print_ts = now
            last_print_done = done

    for nav_csv in nav_files:
        code = _safe_code(nav_csv.stem)
        output_csv = output_dir / f"{code}.csv"
        if output_csv.exists():
            skipped += 1
            done += 1
            _print_progress()
            continue
        try:
            info = process_one_fund(
                nav_csv=nav_csv,
                bonus_csv=bonus_dir / f"{code}.csv",
                split_csv=split_dir / f"{code}.csv",
                output_csv=output_csv,
                allow_missing_event_until=allow_missing_event_until,
            )
            processed.append(info)
        except Exception as err:  # noqa: BLE001
            err_msg = str(err)
            _append_failure_log(log_path=fail_log, code=code, error=err_msg)
            failed.append({"code": code, "error": err_msg})
        finally:
            done += 1
            _print_progress()

    _print_progress(force=True)
    return {
        "funds": len(processed),
        "items": processed,
        "failed": len(failed),
        "failed_items": failed,
        "skipped": skipped,
        "total": total,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate adjusted NAV by fund code from nav/bonus/split CSV files")
    parser.add_argument("--nav-dir", type=Path, required=True, help="Input nav csv directory (fund_nav_by_code)")
    parser.add_argument("--bonus-dir", type=Path, required=True, help="Input bonus csv directory (fund_bonus_by_code)")
    parser.add_argument("--split-dir", type=Path, required=True, help="Input split csv directory (fund_split_by_code)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for adjusted nav csv files")
    parser.add_argument("--codes", nargs="*", default=None, help="Optional specific fund codes, e.g. 163402 000001")
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=5.0,
        help="Progress print interval in seconds",
    )
    parser.add_argument(
        "--allow-missing-event-until",
        type=str,
        default=None,
        help="Allow missing dividend/split event dates on or before this date (YYYY-MM-DD)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    allow_missing_event_until = None
    if args.allow_missing_event_until:
        parsed = pd.to_datetime(args.allow_missing_event_until, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"invalid --allow-missing-event-until: {args.allow_missing_event_until}")
        allow_missing_event_until = parsed
    summary = process_all_funds(
        nav_dir=args.nav_dir,
        bonus_dir=args.bonus_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        codes=args.codes,
        progress_interval_seconds=args.progress_interval_seconds,
        allow_missing_event_until=allow_missing_event_until,
    )
    print(f"processed funds: {summary['funds']}")
    if summary["skipped"] > 0:
        print(f"skipped funds: {summary['skipped']}")
    for item in summary["items"]:
        print(f"{item['code']}: rows={item['rows']} range={item['start_date']}~{item['end_date']}")
    if summary["failed"] > 0:
        print(f"failed funds: {summary['failed']} (details in failed_adjusted_nav.jsonl)")


if __name__ == "__main__":
    main()
