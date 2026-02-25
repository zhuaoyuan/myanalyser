from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

SUMMARY_COLUMNS = [
    "基金代码",
    "数据是否缺失",
    "参与比对收益率的天数",
    "因日期数据缺失跳过的天数",
    "<1%偏差占比",
    "1%～2%偏差占比",
    "2%～5%偏差占比",
    "5%～10%偏差占比",
    "10%以上偏差占比",
]

DETAIL_COLUMNS = [
    "期初日期",
    "期末日期",
    "本地期初值",
    "本地期末值",
    "远程期初值",
    "远程期末值",
    "该时段本地计算的收益率",
    "该时段远程收益率",
    "本地远程收益率偏差",
    "偏差类型",
]

MISSING_STAGE = "compare_adjusted_nav_cum_return"


def _safe_code(value: object) -> str:
    return str(value).strip().zfill(6)


def _append_error_log(log_path: Path, code: str, error: str) -> None:
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "stage": MISSING_STAGE,
        "code": _safe_code(code),
        "error": error,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_series(csv_path: Path, date_col: str, value_col: str) -> pd.Series:
    df = pd.read_csv(csv_path, dtype=str)
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"{csv_path} missing columns: {date_col}, {value_col}")

    parsed = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "value": pd.to_numeric(df[value_col], errors="coerce"),
        }
    ).dropna(subset=["date", "value"])

    if parsed.empty:
        return pd.Series(dtype=float)

    parsed = parsed.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return pd.Series(parsed["value"].to_numpy(), index=parsed["date"])


def _format_num(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.10f}"


def _format_ratio(value: float) -> str:
    return f"{value * 100:.2f}%"


def _deviation_type(deviation: float | None) -> str:
    if deviation is None:
        return "-"
    abs_dev = abs(deviation)
    if abs_dev < 0.01:
        return "<1%"
    if abs_dev < 0.02:
        return "1%～2%"
    if abs_dev < 0.05:
        return "2%～5%"
    if abs_dev < 0.10:
        return "5%～10%"
    return "10%以上"


def _calc_local_return(start_val: float, end_val: float) -> float | None:
    if start_val == 0:
        return None
    return (end_val - start_val) / start_val


def _calc_remote_return(start_val: float, end_val: float) -> float | None:
    denominator = start_val + 100.0
    if denominator == 0:
        return None
    return (end_val - start_val) / denominator


def _empty_summary_row(code: str, missing: str) -> dict[str, object]:
    return {
        "基金代码": _safe_code(code),
        "数据是否缺失": missing,
        "参与比对收益率的天数": 0,
        "因日期数据缺失跳过的天数": 0,
        "<1%偏差占比": "0.00%",
        "1%～2%偏差占比": "0.00%",
        "2%～5%偏差占比": "0.00%",
        "5%～10%偏差占比": "0.00%",
        "10%以上偏差占比": "0.00%",
    }


def compare_adjusted_nav_and_cum_return(base_dir: Path, output_dir: Path | None = None) -> dict[str, Path]:
    adjusted_dir = base_dir / "fund_adjusted_nav_by_code"
    cum_return_dir = base_dir / "fund_cum_return_by_code"
    if not adjusted_dir.is_dir():
        raise FileNotFoundError(f"missing directory: {adjusted_dir}")
    if not cum_return_dir.is_dir():
        raise FileNotFoundError(f"missing directory: {cum_return_dir}")

    out_dir = output_dir or (base_dir / "fund_return_compare")
    detail_dir = out_dir / "details"
    summary_csv = out_dir / "summary.csv"
    error_jsonl = out_dir / "errors.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)
    if error_jsonl.exists():
        error_jsonl.unlink()

    adjusted_files = {path.stem: path for path in adjusted_dir.glob("*.csv")}
    cum_files = {path.stem: path for path in cum_return_dir.glob("*.csv")}
    adjusted_codes = {_safe_code(code) for code in adjusted_files}
    cum_codes = {_safe_code(code) for code in cum_files}
    all_codes = sorted(adjusted_codes | cum_codes)

    summary_rows: list[dict[str, object]] = []
    for code in all_codes:
        adjusted_csv = adjusted_dir / f"{code}.csv"
        cum_csv = cum_return_dir / f"{code}.csv"

        if not adjusted_csv.exists() or not cum_csv.exists():
            side = "adjusted_nav_missing" if not adjusted_csv.exists() else "cum_return_missing"
            _append_error_log(error_jsonl, code, f"fund file missing: {side}")
            summary_rows.append(_empty_summary_row(code, "是"))
            continue

        try:
            adjusted_series = _parse_series(adjusted_csv, date_col="净值日期", value_col="复权净值")
            cum_series = _parse_series(cum_csv, date_col="日期", value_col="累计收益率")
        except Exception as err:  # noqa: BLE001
            _append_error_log(error_jsonl, code, f"parse_error: {err}")
            summary_rows.append(_empty_summary_row(code, "是"))
            continue

        adjusted_dates = set(adjusted_series.index.tolist())
        cum_dates = set(cum_series.index.tolist())
        common_dates = sorted(adjusted_dates & cum_dates)
        if not common_dates:
            _append_error_log(error_jsonl, code, "no common date found")
            summary_rows.append(_empty_summary_row(code, "是"))
            continue

        end_date = common_dates[-1]
        scan_dates = sorted(date for date in (adjusted_dates | cum_dates) if date <= end_date)

        skipped_missing_days = 0
        start_dates: list[pd.Timestamp] = []
        for date in scan_dates:
            if date in adjusted_dates and date in cum_dates:
                if date < end_date:
                    start_dates.append(date)
            else:
                skipped_missing_days += 1

        details: list[dict[str, object]] = []
        bucket_count = {"<1%": 0, "1%～2%": 0, "2%～5%": 0, "5%～10%": 0, "10%以上": 0}

        for start_date in start_dates:
            local_start = float(adjusted_series.loc[start_date])
            local_end = float(adjusted_series.loc[end_date])
            remote_start = float(cum_series.loc[start_date])
            remote_end = float(cum_series.loc[end_date])

            local_return = _calc_local_return(local_start, local_end)
            remote_return = _calc_remote_return(remote_start, remote_end)
            if local_return in (None, 0.0) or remote_return is None:
                deviation = None
            else:
                deviation = (local_return - remote_return) / local_return

            dev_type = _deviation_type(deviation)
            if dev_type in bucket_count:
                bucket_count[dev_type] += 1

            details.append(
                {
                    "期初日期": start_date.strftime("%Y-%m-%d"),
                    "期末日期": end_date.strftime("%Y-%m-%d"),
                    "本地期初值": _format_num(local_start),
                    "本地期末值": _format_num(local_end),
                    "远程期初值": _format_num(remote_start),
                    "远程期末值": _format_num(remote_end),
                    "该时段本地计算的收益率": _format_num(local_return),
                    "该时段远程收益率": _format_num(remote_return),
                    "本地远程收益率偏差": _format_num(deviation),
                    "偏差类型": dev_type,
                }
            )

        detail_df = pd.DataFrame(details, columns=DETAIL_COLUMNS)
        detail_df.to_csv(detail_dir / f"{code}.csv", index=False, encoding="utf-8-sig")

        compared_days = len(start_dates)
        if compared_days > 0:
            bucket_ratio = {name: _format_ratio(count / compared_days) for name, count in bucket_count.items()}
        else:
            bucket_ratio = {name: "0.00%" for name in bucket_count}

        summary_rows.append(
            {
                "基金代码": code,
                "数据是否缺失": "否",
                "参与比对收益率的天数": compared_days,
                "因日期数据缺失跳过的天数": skipped_missing_days,
                "<1%偏差占比": bucket_ratio["<1%"],
                "1%～2%偏差占比": bucket_ratio["1%～2%"],
                "2%～5%偏差占比": bucket_ratio["2%～5%"],
                "5%～10%偏差占比": bucket_ratio["5%～10%"],
                "10%以上偏差占比": bucket_ratio["10%以上"],
            }
        )

    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    return {
        "summary_csv": summary_csv,
        "detail_dir": detail_dir,
        "error_jsonl": error_jsonl,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local adjusted NAV return with remote cumulative return series by fund code."
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        type=Path,
        help="Base dir containing fund_adjusted_nav_by_code and fund_cum_return_by_code",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: {base-dir}/fund_return_compare)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = compare_adjusted_nav_and_cum_return(base_dir=args.base_dir, output_dir=args.output_dir)
    print(f"summary: {result['summary_csv']}")
    print(f"details: {result['detail_dir']}")
    print(f"errors: {result['error_jsonl']}")


if __name__ == "__main__":
    main()
