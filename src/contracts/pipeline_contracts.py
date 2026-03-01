from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CsvContract:
    required_columns: tuple[str, ...]
    non_null_columns: tuple[str, ...] = ()
    unique_key_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    date_columns: tuple[str, ...] = ()
    allowed_values: dict[str, set[str]] = field(default_factory=dict)
    min_rows: int = 1


@dataclass(frozen=True)
class DirContract:
    min_csv_files: int = 1


CONTRACTS: dict[str, CsvContract | DirContract] = {
    "fund_purchase_csv": CsvContract(
        required_columns=("基金代码", "基金简称", "申购状态", "赎回状态", "购买起点", "日累计限定金额", "手续费"),
        non_null_columns=("基金代码",),
        unique_key_columns=("基金代码",),
        numeric_columns=("购买起点", "日累计限定金额", "手续费"),
    ),
    "fund_purchase_effective_csv": CsvContract(
        required_columns=("基金代码", "基金简称", "申购状态", "赎回状态", "购买起点", "日累计限定金额", "手续费"),
        non_null_columns=("基金代码",),
        unique_key_columns=("基金代码",),
        numeric_columns=("购买起点", "日累计限定金额", "手续费"),
    ),
    "fund_blacklist_csv": CsvContract(
        required_columns=("基金代码",),
        non_null_columns=(),
        unique_key_columns=(),
        min_rows=0,
    ),
    "fund_overview_csv": CsvContract(
        required_columns=("基金代码", "成立日期/规模"),
        non_null_columns=("基金代码",),
        unique_key_columns=("基金代码",),
    ),
    "filtered_fund_candidates_csv": CsvContract(
        required_columns=("基金编码", "是否过滤", "过滤原因"),
        non_null_columns=("基金编码", "是否过滤"),
        unique_key_columns=("基金编码",),
        allowed_values={"是否过滤": {"是", "否"}},
    ),
    "fund_purchase_filtered_csv": CsvContract(
        required_columns=("基金代码",),
        non_null_columns=("基金代码",),
        unique_key_columns=("基金代码",),
    ),
    "fund_scoreboard_csv": CsvContract(
        required_columns=(
            "基金代码",
            "基金名称",
            "期初日期",
            "期末日期",
            "年化收益率",
            "最大回撤率",
            "近1年夏普比率",
            "近3年夏普比率",
            "近1年卡玛比率",
            "近3年卡玛比率",
            "全期最长回撤修复天数",
            "全期最大单日跌幅",
        ),
        non_null_columns=("基金代码", "基金名称", "期初日期", "期末日期"),
        unique_key_columns=("基金代码",),
        numeric_columns=("年化收益率", "最大回撤率", "近1年夏普比率", "近3年夏普比率", "近1年卡玛比率", "近3年卡玛比率"),
        date_columns=("期初日期", "期末日期"),
        min_rows=0,
    ),
    "fund_scoreboard_recalc_input_csv": CsvContract(
        required_columns=(
            "基金代码",
            "年化收益率",
            "上涨季度比例",
            "上涨月份比例",
            "上涨星期比例",
            "季涨跌幅标准差",
            "月涨跌幅标准差",
            "周涨跌幅标准差",
            "最大回撤率",
            "近3年年化收益率",
            "近3年上涨季度比例",
            "近3年上涨月份比例",
            "近3年上涨星期比例",
            "近3年季涨跌幅标准差",
            "近3年月涨跌幅标准差",
            "近3年周涨跌幅标准差",
            "近3年最大回撤率",
            "近1年年化收益率",
            "近1年上涨月份比例",
            "近1年上涨星期比例",
            "近1年月涨跌幅标准差",
            "近1年周涨跌幅标准差",
            "近1年最大回撤率",
            "最近一个月涨跌幅",
            "近1年夏普比率",
            "近3年夏普比率",
            "近1年卡玛比率",
            "近3年卡玛比率",
            "全期最长回撤修复天数",
            "近1年最长回撤修复天数",
            "近3年最长回撤修复天数",
            "全期最大单日跌幅",
            "近1年最大单日跌幅",
            "近3年最大单日跌幅",
        ),
        non_null_columns=("基金代码",),
        unique_key_columns=("基金代码",),
    ),
    "trade_dates_csv": CsvContract(
        required_columns=("trade_date",),
        non_null_columns=("trade_date",),
        date_columns=("trade_date",),
    ),
    "fund_nav_dir": DirContract(min_csv_files=1),
    "fund_bonus_dir": DirContract(min_csv_files=1),
    "fund_split_dir": DirContract(min_csv_files=1),
    "fund_personnel_dir": DirContract(min_csv_files=1),
    "fund_cum_return_dir": DirContract(min_csv_files=1),
    "fund_adjusted_nav_dir": DirContract(min_csv_files=1),
    "compare_details_dir": DirContract(min_csv_files=1),
    "integrity_details_dir": DirContract(min_csv_files=1),
}


STAGE_REQUIREMENTS: dict[str, list[tuple[str, str]]] = {
    "fund_etl_step2_input": [("fund_purchase_csv", "purchase_csv")],
    "fund_etl_step3_input": [("fund_purchase_csv", "purchase_csv")],
    "fund_etl_step4_input": [("fund_purchase_csv", "purchase_csv")],
    "fund_etl_step5_input": [("fund_purchase_csv", "purchase_csv")],
    "fund_etl_step6_input": [("fund_purchase_csv", "purchase_csv")],
    "fund_etl_step7_input": [("fund_purchase_csv", "purchase_csv")],
    "fund_etl_step2_input_effective": [("fund_purchase_effective_csv", "purchase_csv")],
    "fund_etl_step3_input_effective": [("fund_purchase_effective_csv", "purchase_csv")],
    "fund_etl_step4_input_effective": [("fund_purchase_effective_csv", "purchase_csv")],
    "fund_etl_step5_input_effective": [("fund_purchase_effective_csv", "purchase_csv")],
    "fund_etl_step6_input_effective": [("fund_purchase_effective_csv", "purchase_csv")],
    "fund_etl_step7_input_effective": [("fund_purchase_effective_csv", "purchase_csv")],
    "build_effective_purchase_input": [
        ("fund_purchase_csv", "purchase_csv"),
    ],
    "build_effective_purchase_output": [("fund_purchase_effective_csv", "output_csv")],
    "adjusted_nav_input": [
        ("fund_nav_dir", "nav_dir"),
        ("fund_bonus_dir", "bonus_dir"),
        ("fund_split_dir", "split_dir"),
    ],
    "integrity_input": [
        ("fund_adjusted_nav_dir", "adjusted_nav_dir"),
        ("fund_overview_csv", "overview_csv"),
        ("trade_dates_csv", "trade_dates_csv"),
    ],
    "compare_input": [
        ("fund_adjusted_nav_dir", "adjusted_nav_dir"),
        ("fund_cum_return_dir", "cum_return_dir"),
    ],
    "filter_input": [
        ("fund_purchase_csv", "purchase_csv"),
        ("fund_overview_csv", "overview_csv"),
        ("fund_nav_dir", "nav_dir"),
        ("fund_adjusted_nav_dir", "adjusted_nav_dir"),
        ("compare_details_dir", "compare_details_dir"),
        ("integrity_details_dir", "integrity_details_dir"),
    ],
    "scoreboard_input": [
        ("fund_purchase_filtered_csv", "purchase_csv"),
        ("fund_overview_csv", "overview_csv"),
        ("fund_personnel_dir", "personnel_dir"),
        ("fund_adjusted_nav_dir", "nav_dir"),
    ],
    "scoreboard_output": [("fund_scoreboard_csv", "scoreboard_csv")],
    "filtered_candidates_output": [("filtered_fund_candidates_csv", "filter_csv")],
    "filtered_purchase_output": [("fund_purchase_filtered_csv", "purchase_csv")],
    "verify_scoreboard_recalc_input": [
        ("fund_scoreboard_recalc_input_csv", "scoreboard_csv"),
        ("fund_adjusted_nav_dir", "nav_dir"),
    ],
}
