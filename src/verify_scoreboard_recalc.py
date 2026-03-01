from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scoreboard_metrics import (
    METRIC_DIRECTIONS,
    compute_metrics,
    load_nav_df,
    safe_code,
    window_metrics,
)
from validators.validate_pipeline_artifacts import validate_stage_or_raise

DEFAULT_MAX_INPUT_ROWS = 200
DETAIL_COLUMNS = ["核验数据项名称", "原结果", "新结果", "核验是否通过"]

# (中文字段名, 内部字段名, 展示格式)，与 pipeline_scoreboard.EXPORT_COLUMN_SPECS 中导出的列名对齐
VERIFY_FIELDS: list[tuple[str, str, str]] = [
    ("年化收益率", "annual_return", "percent2"),
    ("上涨季度比例", "up_quarter_ratio", "percent0"),
    ("上涨月份比例", "up_month_ratio", "percent0"),
    ("上涨星期比例", "up_week_ratio", "percent0"),
    ("季涨跌幅标准差", "quarter_return_std", "percent2"),
    ("月涨跌幅标准差", "month_return_std", "percent2"),
    ("周涨跌幅标准差", "week_return_std", "percent2"),
    ("最大回撤率", "max_drawdown", "percent2"),
    ("近3年年化收益率", "annual_return_3y", "percent2"),
    ("近3年上涨季度比例", "up_quarter_ratio_3y", "percent0"),
    ("近3年上涨月份比例", "up_month_ratio_3y", "percent0"),
    ("近3年上涨星期比例", "up_week_ratio_3y", "percent0"),
    ("近3年季涨跌幅标准差", "quarter_return_std_3y", "percent2"),
    ("近3年月涨跌幅标准差", "month_return_std_3y", "percent2"),
    ("近3年周涨跌幅标准差", "week_return_std_3y", "percent2"),
    ("近3年最大回撤率", "max_drawdown_3y", "percent2"),
    ("近1年年化收益率", "annual_return_1y", "percent2"),
    ("近1年上涨月份比例", "up_month_ratio_1y", "percent0"),
    ("近1年上涨星期比例", "up_week_ratio_1y", "percent0"),
    ("近1年月涨跌幅标准差", "month_return_std_1y", "percent2"),
    ("近1年周涨跌幅标准差", "week_return_std_1y", "percent2"),
    ("近1年最大回撤率", "max_drawdown_1y", "percent2"),
    ("最近一个月涨跌幅", "recent_month_return", "percent2"),
    ("近1年夏普比率", "sharpe_ratio_1y", "round2"),
    ("近3年夏普比率", "sharpe_ratio_3y", "round2"),
    ("近1年卡玛比率", "calmar_ratio_1y", "round2"),
    ("近3年卡玛比率", "calmar_ratio_3y", "round2"),
    ("全期最长回撤修复天数", "max_drawdown_recovery_days", "int"),
    ("近1年最长回撤修复天数", "max_drawdown_recovery_days_1y", "int"),
    ("近3年最长回撤修复天数", "max_drawdown_recovery_days_3y", "int"),
    ("全期最大单日跌幅", "max_single_day_drop", "percent2"),
    ("近1年最大单日跌幅", "max_single_day_drop_1y", "percent2"),
    ("近3年最大单日跌幅", "max_single_day_drop_3y", "percent2"),
]


def _to_display_value(value: object, style: str) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    if style == "int":
        return int(round(float(value)))
    if style == "round2":
        return round(float(value), 2)
    if style == "percent2":
        return round(float(value) * 100.0, 2)
    if style == "percent0":
        return int(round(float(value) * 100.0))
    return float(value)


def _parse_original_value(value: object, style: str) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    if style in {"int", "percent0"}:
        return int(round(float(text)))
    return round(float(text), 2)


def _is_equal(left: int | float | None, right: int | float | None) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    return abs(float(left) - float(right)) <= 1e-12


def _build_recalc_metrics_with_latest_nav_date(
    scoreboard_df: pd.DataFrame,
    nav_dir: Path,
    latest_nav_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code in scoreboard_df["基金代码"].astype(str).map(safe_code).tolist():
        nav_df = load_nav_df(nav_dir / f"{code}.csv")
        if latest_nav_date is not None:
            nav_df = nav_df[nav_df["净值日期"] <= latest_nav_date].reset_index(drop=True)
        row: dict[str, object] = {"基金代码": code}
        for k in METRIC_DIRECTIONS:
            row[k] = None
        if not nav_df.empty:
            end_date = nav_df["净值日期"].iloc[-1]
            row.update(compute_metrics(nav_df=nav_df, end_date=end_date))
            row.update(window_metrics(nav_df=nav_df, end_date=end_date, years=3))
            row.update(window_metrics(nav_df=nav_df, end_date=end_date, years=1))
        rows.append(row)

    return pd.DataFrame(rows)


def run_verification(
    scoreboard_csv: Path,
    fund_etl_dir: Path,
    output_dir: Path,
    max_input_rows: int = DEFAULT_MAX_INPUT_ROWS,
    latest_nav_date: pd.Timestamp | None = None,
) -> dict[str, Path]:
    validate_stage_or_raise(
        "verify_scoreboard_recalc_input",
        scoreboard_csv=scoreboard_csv,
        nav_dir=fund_etl_dir / "fund_adjusted_nav_by_code",
    )
    scoreboard_df = pd.read_csv(scoreboard_csv, dtype={"基金代码": str}, encoding="utf-8-sig")
    if scoreboard_df.shape[0] > max_input_rows:
        raise ValueError(
            f"scoreboard rows={scoreboard_df.shape[0]} exceeds max_input_rows={max_input_rows}; "
            "ranking verification must use full input without additional sampling"
        )

    missing_fields = [name for name, _, _ in VERIFY_FIELDS if name not in scoreboard_df.columns]
    if missing_fields:
        raise ValueError(f"scoreboard missing verify columns: {missing_fields}")
    if "基金代码" not in scoreboard_df.columns:
        raise ValueError("scoreboard missing required column: 基金代码")

    scoreboard_df = scoreboard_df.copy()
    scoreboard_df["基金代码"] = scoreboard_df["基金代码"].map(safe_code)

    nav_dir = fund_etl_dir / "fund_adjusted_nav_by_code"
    recalc_df = _build_recalc_metrics_with_latest_nav_date(
        scoreboard_df=scoreboard_df,
        nav_dir=nav_dir,
        latest_nav_date=latest_nav_date,
    )
    merged = scoreboard_df.merge(recalc_df, on="基金代码", how="left", suffixes=("", "_recalc"))

    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir = output_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        code = row["基金代码"]
        detail_rows: list[dict[str, object]] = []
        failed_fields: list[str] = []
        for cn_name, internal_name, style in VERIFY_FIELDS:
            old_display = _parse_original_value(row.get(cn_name), style=style)
            new_display = _to_display_value(row.get(internal_name), style=style)
            passed = _is_equal(old_display, new_display)
            if not passed:
                failed_fields.append(cn_name)
            detail_rows.append(
                {
                    "核验数据项名称": cn_name,
                    "原结果": old_display,
                    "新结果": new_display,
                    "核验是否通过": "是" if passed else "否",
                }
            )

        detail_df = pd.DataFrame(detail_rows, columns=DETAIL_COLUMNS)
        detail_df.to_csv(details_dir / f"{code}.csv", index=False, encoding="utf-8-sig")
        summary_rows.append(
            {
                "基金代码": code,
                "待核验字段是否全部核验通过": "是" if len(failed_fields) == 0 else "否",
                "未通过字段名": ",".join(failed_fields),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows,
        columns=["基金代码", "待核验字段是否全部核验通过", "未通过字段名"],
    )
    summary_csv = output_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    metrics_cols = ["基金代码"] + list(METRIC_DIRECTIONS.keys())
    metrics_csv = output_dir / "metrics_recalc_sample.csv"
    recalc_df[metrics_cols].to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    return {
        "summary_csv": summary_csv,
        "details_dir": details_dir,
        "metrics_recalc_sample_csv": metrics_csv,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recalculate scoreboard metrics from fund_etl intermediate data and verify against exported scoreboard csv"
    )
    parser.add_argument("--scoreboard-csv", type=Path, required=True)
    parser.add_argument("--fund-etl-dir", type=Path, required=True, help=".../data/versions/{run_id}/fund_etl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-input-rows", type=int, default=DEFAULT_MAX_INPUT_ROWS)
    parser.add_argument("--latest-nav-date", type=str, default=None, help="可选，按净值日期截断重算，格式 YYYY-MM-DD")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    latest_nav_date = pd.to_datetime(args.latest_nav_date) if args.latest_nav_date else None
    result = run_verification(
        scoreboard_csv=args.scoreboard_csv,
        fund_etl_dir=args.fund_etl_dir,
        output_dir=args.output_dir,
        max_input_rows=args.max_input_rows,
        latest_nav_date=latest_nav_date,
    )
    print("alignment: window_start=end_date-DateOffset(years=N), freq={week:W-FRI, month:ME, quarter:QE}")
    print("alignment: metric computation consistent with pipeline_scoreboard.py")
    print(f"summary_csv={result['summary_csv']}")
    print(f"details_dir={result['details_dir']}")
    print(f"metrics_recalc_sample_csv={result['metrics_recalc_sample_csv']}")


if __name__ == "__main__":
    main()
