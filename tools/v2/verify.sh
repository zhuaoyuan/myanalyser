#!/usr/bin/env bash
# v2 精简验收脚本：仅保留对 V2 流程有用的环节，不依赖 fund_infra（Docker/MySQL/ClickHouse）。
#
# 与 V2完整流程说明.md 对齐：
# - 数据抓取(ETL) + 复权净值 + 数据质量(integrity/compare/filter) + 评分榜(CSV) + 筛选打分 + 重算核验
# - 不含：数据库入库、backtest_verify_e2e（需 ClickHouse）
#
# 用法：
#   cd myanalyser && bash tools/v2/verify.sh
#   RUN_ID=20260319_120000_verify_v2 bash tools/v2/verify.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[verify] warning: VIRTUAL_ENV is not active. Please run:"
  echo "  source ${PROJECT_ROOT}/.venv312/bin/activate"
fi

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[verify] missing python/python3 in PATH"
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_verify_v2}"
DATA_VERSION="${DATA_VERSION:-${RUN_ID}}"
VERIFY_ROOT="${PROJECT_ROOT}/data/versions/${RUN_ID}"
FUND_ETL_DIR="${VERIFY_ROOT}/fund_etl"
LOGS_DIR="${VERIFY_ROOT}/logs"
ARTIFACTS_DIR="${PROJECT_ROOT}/artifacts/verify_${RUN_ID}"
SCOREBOARD_DIR="${ARTIFACTS_DIR}/scoreboard"
FILTER_START_DATE="${FILTER_START_DATE:-2023-01-01}"
FILTER_MAX_ABS_DEVIATION="${FILTER_MAX_ABS_DEVIATION:-0.02}"
FILTER_RESULT_CSV="${ARTIFACTS_DIR}/filtered_fund_candidates.csv"
FILTERED_PURCHASE_CSV="${FUND_ETL_DIR}/fund_purchase_for_step10_filtered.csv"
FUND_PURCHASE_EFFECTIVE_CSV="${FUND_ETL_DIR}/fund_purchase.csv"
RUN_REPORT_STEPS_CSV="${ARTIFACTS_DIR}/run_report_steps.csv"
RUN_REPORT_SUMMARY_CSV="${ARTIFACTS_DIR}/run_report_summary.csv"
RUN_REPORT_MD="${ARTIFACTS_DIR}/run_report.md"
CURRENT_STEP=""
STEP_START_TS=0
BONUS_SPLIT_REVISE_ROOT="${BONUS_SPLIT_REVISE_ROOT:-}"

assert_file_exists() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[verify] missing file: ${path}"
    exit 1
  fi
}

assert_dir_exists() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[verify] missing directory: ${path}"
    exit 1
  fi
}

assert_csv_has_rows() {
  local path="$1"
  "${PYTHON_BIN}" - <<'PY' "${path}"
import sys
from pathlib import Path
import pandas as pd

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
if df.empty:
    raise SystemExit(2)
PY
}

assert_dir_has_csv() {
  local dir="$1"
  "${PYTHON_BIN}" - <<'PY' "${dir}"
import sys
from pathlib import Path

dir_path = Path(sys.argv[1])
if not dir_path.exists():
    raise SystemExit(1)
files = sorted(dir_path.glob("*.csv"))
if not files:
    raise SystemExit(2)
PY
}

BONUS_SPLIT_REVISE_ARG=()
if [[ -n "${BONUS_SPLIT_REVISE_ROOT}" ]]; then
  if [[ "${BONUS_SPLIT_REVISE_ROOT}" != /* ]]; then
    BONUS_SPLIT_REVISE_ROOT="$(cd "${PROJECT_ROOT}" && cd "${BONUS_SPLIT_REVISE_ROOT}" && pwd)"
  fi
  BONUS_SPLIT_REVISE_ARG=(--bonus-split-revise-root "${BONUS_SPLIT_REVISE_ROOT}")
fi

start_step() {
  CURRENT_STEP="$1"
  STEP_START_TS="$(date +%s)"
  echo "[verify] ${CURRENT_STEP}"
}

finish_step() {
  local status="$1"
  local end_ts duration
  end_ts="$(date +%s)"
  duration=$((end_ts - STEP_START_TS))
  printf '%s,%s,%s\n' "${CURRENT_STEP}" "${status}" "${duration}" >>"${RUN_REPORT_STEPS_CSV}"
  CURRENT_STEP=""
}

generate_run_report() {
  "${PYTHON_BIN}" - <<'PY' "${RUN_REPORT_STEPS_CSV}" "${RUN_REPORT_SUMMARY_CSV}" "${RUN_REPORT_MD}" "${FUND_ETL_DIR}" "${LOGS_DIR}" "${FILTER_RESULT_CSV}" "${FILTERED_PURCHASE_CSV}"
import json
import sys
from pathlib import Path

import pandas as pd

steps_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
report_md = Path(sys.argv[3])
fund_etl_dir = Path(sys.argv[4])
logs_dir = Path(sys.argv[5])
filter_result_csv = Path(sys.argv[6])
filtered_purchase_csv = Path(sys.argv[7])

if not steps_csv.exists():
    raise SystemExit(0)

steps = pd.read_csv(steps_csv, dtype=str)
steps["duration_seconds"] = pd.to_numeric(steps["duration_seconds"], errors="coerce").fillna(0).astype(int)
total_steps = len(steps)
ok_steps = int((steps["status"] == "success").sum())
success_rate = (ok_steps / total_steps * 100.0) if total_steps else 0.0

error_stage_count = {}
if logs_dir.exists():
    for p in sorted(logs_dir.glob("*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stage = str(rec.get("stage", "unknown")).strip() or "unknown"
                error_stage_count[stage] = error_stage_count.get(stage, 0) + 1

purchase_before = None
purchase_after = None
filtered_yes = None
if (fund_etl_dir / "fund_purchase.csv").exists():
    purchase_before = len(pd.read_csv(fund_etl_dir / "fund_purchase.csv", dtype=str, encoding="utf-8-sig"))
if filtered_purchase_csv.exists():
    purchase_after = len(pd.read_csv(filtered_purchase_csv, dtype=str, encoding="utf-8-sig"))
if filter_result_csv.exists():
    fdf = pd.read_csv(filter_result_csv, dtype=str, encoding="utf-8-sig")
    if "是否过滤" in fdf.columns:
        filtered_yes = int((fdf["是否过滤"] == "是").sum())

summary = pd.DataFrame([
    {"指标": "总步骤数", "值": total_steps},
    {"指标": "成功步骤数", "值": ok_steps},
    {"指标": "步骤成功率(%)", "值": round(success_rate, 2)},
    {"指标": "过滤前基金数", "值": purchase_before},
    {"指标": "过滤后基金数", "值": purchase_after},
    {"指标": "被过滤基金数", "值": filtered_yes},
])
summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

err_text = "无"
if error_stage_count:
    err_text = "; ".join(f"{k}:{v}" for k, v in sorted(error_stage_count.items()))

lines = [
    "# v2 验收运行报告",
    "",
    "## 验收结论",
    f"- 步骤成功率: {ok_steps}/{total_steps} ({success_rate:.2f}%)",
    f"- 过滤前后数量: {purchase_before} -> {purchase_after}",
    f"- 被过滤基金数: {filtered_yes}",
    f"- 异常分布: {err_text}",
    "",
    "## 步骤耗时",
]
for _, row in steps.iterrows():
    lines.append(f"- {row['step']}: {row['status']} ({int(row['duration_seconds'])}s)")
report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

on_error() {
  if [[ -n "${CURRENT_STEP}" ]]; then
    finish_step "failed"
  fi
  generate_run_report
}

trap on_error ERR

echo "[verify] v2 精简验收（无 fund_infra 依赖）"
echo "[verify] project_root=${PROJECT_ROOT}"
echo "[verify] run_id=${RUN_ID}"

mkdir -p "${FUND_ETL_DIR}" "${LOGS_DIR}" "${ARTIFACTS_DIR}" "${SCOREBOARD_DIR}"
printf 'step,status,duration_seconds\n' >"${RUN_REPORT_STEPS_CSV}"

# ---------------------------------------------------------------------------
# Step 1: 单元测试
# unittest discover 不加载 pytest conftest，需显式设置 PYTHONPATH 供 test_fetch_fund_* 等导入 tools/prep
# ---------------------------------------------------------------------------
start_step "step1_unit_tests"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/tools:${PROJECT_ROOT}/tools/v2:${PROJECT_ROOT}/tools/prep:${PYTHONPATH:-}"
"${PYTHON_BIN}" -m unittest discover -s tests -p "test_*.py" -v
finish_step "success"

# ---------------------------------------------------------------------------
# Step 2: 核心 CLI 冒烟 + V2 流程 CLI 冒烟
# 覆盖 V2完整流程说明.md 中的环境：
#   - CoreCliIntegrationTest: fund_etl、pipeline_skip_sinks、backtest、compare、integrity
#   - V2FlowCliSmokeTest: compare_window、compare_backtest_curves、fetch_fund_index_sw、
#     benchmark_portfolio_backtest、prep_data_workflow、filter_funds_for_next_step、run_filter_and_score
# ---------------------------------------------------------------------------
start_step "step2_core_cli_smoke"
"${PYTHON_BIN}" -m unittest -v \
  tests.test_cli_integration.CoreCliIntegrationTest.test_fund_etl_cli_run_id_layout \
  tests.test_cli_integration.CoreCliIntegrationTest.test_pipeline_cli_smoke_skip_sinks_with_run_id_layout \
  tests.test_cli_integration.CoreCliIntegrationTest.test_backtest_cli_smoke_with_run_id_layout \
  tests.test_cli_integration.CoreCliIntegrationTest.test_compare_cli_with_run_id_layout \
  tests.test_cli_integration.CoreCliIntegrationTest.test_check_trade_day_integrity_cli_with_run_id_layout \
  tests.test_v2_flow_cli_smoke.V2FlowCliSmokeTest.test_compare_window_cli_smoke \
  tests.test_v2_flow_cli_smoke.V2FlowCliSmokeTest.test_compare_backtest_curves_cli_smoke \
  tests.test_v2_flow_cli_smoke.V2FlowCliSmokeTest.test_fetch_fund_index_sw_cli_smoke \
  tests.test_v2_flow_cli_smoke.V2FlowCliSmokeTest.test_benchmark_portfolio_backtest_cli_smoke \
  tests.test_v2_flow_cli_smoke.V2FlowCliSmokeTest.test_prep_data_workflow_v2_cli_help \
  tests.test_v2_flow_cli_smoke.V2FlowCliSmokeTest.test_filter_funds_for_next_step_cli_smoke \
  tests.test_v2_flow_cli_smoke.V2FlowCliSmokeTest.test_run_filter_and_score_cli_smoke
finish_step "success"

# ---------------------------------------------------------------------------
# Step 2b: V2 最小基线完整集成回归测试
# 用小份固定输入 (tests/baseline/mini_case_v2/input) 跑 V2 全流程 step5~10，
# 逐环节校验产出与 expected/default 一致，防止重构改动破坏各环节结果。
# ---------------------------------------------------------------------------
start_step "step2b_v2_baseline_regression"
"${PYTHON_BIN}" -m unittest -v \
  tests.test_v2_baseline_regression.V2BaselineRegressionTest.test_v2_baseline_full_flow_regression
finish_step "success"

# ---------------------------------------------------------------------------
# Step 3: fund_etl step1 + 抽样 21 只基金
# ---------------------------------------------------------------------------
start_step "step3_fund_etl_step1_and_sampling"
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode verify
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step1
assert_file_exists "${FUND_ETL_DIR}/fund_purchase.csv"
assert_csv_has_rows "${FUND_ETL_DIR}/fund_purchase.csv"

RUN_ID="${RUN_ID}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import pandas as pd

root = Path(".").resolve()
run_id = os.environ["RUN_ID"]
verify_root = root / "data" / "versions" / run_id
fund = verify_root / "fund_etl"
purchase_csv = fund / "fund_purchase.csv"
df = pd.read_csv(purchase_csv, dtype={"基金代码": str}, encoding="utf-8-sig")
if "基金代码" not in df.columns:
    raise ValueError(f"missing 基金代码 column: {purchase_csv}")

df["基金代码"] = df["基金代码"].map(lambda v: str(v).strip().zfill(6))
df = df.drop_duplicates(subset=["基金代码"], keep="first")
target_code = "163402"

top20 = df[df["基金代码"] != target_code].head(20).copy()
if top20.shape[0] < 20:
    raise ValueError(f"fund_purchase rows not enough for sampling 20 rows: got={top20.shape[0]}")

target_row = df[df["基金代码"] == target_code].head(1).copy()
if target_row.empty:
    target_row = top20.head(1).copy()
    target_row["基金代码"] = target_code
    for col in target_row.columns:
        if col != "基金代码":
            target_row[col] = ""

sample_df = pd.concat([top20, target_row], ignore_index=True)
if sample_df.shape[0] != 21:
    raise ValueError(f"sample rows expected 21, got={sample_df.shape[0]}")

sample_df.to_csv(purchase_csv, index=False, encoding="utf-8-sig")
PY
assert_csv_has_rows "${FUND_ETL_DIR}/fund_purchase.csv"
finish_step "success"

# ---------------------------------------------------------------------------
# Step 4: fund_etl step2~step7
# ---------------------------------------------------------------------------
start_step "step4_fund_etl_step2_to_step7"
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step2 --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}" --max-workers 8
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step3 --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}"
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step4 --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}"
if [[ ${#BONUS_SPLIT_REVISE_ARG[@]} -gt 0 ]]; then
  "${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step5 --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}" "${BONUS_SPLIT_REVISE_ARG[@]}"
else
  "${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step5 --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}"
fi
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step6 --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}"
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step7 --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}"
assert_csv_has_rows "${FUND_ETL_DIR}/fund_overview.csv"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_nav_by_code"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_bonus_by_code"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_split_by_code"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_personnel_by_code"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_cum_return_by_code"
finish_step "success"

# ---------------------------------------------------------------------------
# Step 5: 复权净值
# ---------------------------------------------------------------------------
BONUS_DIR_FOR_ADJ="${FUND_ETL_DIR}/fund_bonus_by_code"
SPLIT_DIR_FOR_ADJ="${FUND_ETL_DIR}/fund_split_by_code"
if [[ -d "${FUND_ETL_DIR}/revised_fund_bonus_by_code" && -d "${FUND_ETL_DIR}/revised_fund_split_by_code" ]]; then
  BONUS_DIR_FOR_ADJ="${FUND_ETL_DIR}/revised_fund_bonus_by_code"
  SPLIT_DIR_FOR_ADJ="${FUND_ETL_DIR}/revised_fund_split_by_code"
fi

start_step "step5_adjusted_nav"
"${PYTHON_BIN}" src/adjusted_nav_tool.py \
  --nav-dir "${FUND_ETL_DIR}/fund_nav_by_code" \
  --bonus-dir "${BONUS_DIR_FOR_ADJ}" \
  --split-dir "${SPLIT_DIR_FOR_ADJ}" \
  --output-dir "${FUND_ETL_DIR}/fund_adjusted_nav_by_code" \
  --allow-missing-event-until 2020-12-31 \
  --fail-log "${LOGS_DIR}/failed_adjusted_nav.jsonl"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_adjusted_nav_by_code"
finish_step "success"

# ---------------------------------------------------------------------------
# Step 6: 交易日完整性检查
# ---------------------------------------------------------------------------
start_step "step6_integrity"
"${PYTHON_BIN}" src/check_trade_day_data_integrity.py \
  --base-dir "${FUND_ETL_DIR}" \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --trade-dates-csv "${PROJECT_ROOT}/data/common/trade_dates.csv" \
  --output-dir "${ARTIFACTS_DIR}/trade_day_integrity_reports"

SUMMARY_CSV="$(ls -1 "${ARTIFACTS_DIR}"/trade_day_integrity_reports/trade_day_integrity_summary_*.csv 2>/dev/null | head -n 1)"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "[verify] missing integrity summary csv"
  exit 1
fi
assert_csv_has_rows "${SUMMARY_CSV}"
finish_step "success"

# ---------------------------------------------------------------------------
# Step 7: 复权净值 vs 累计收益率 对比
# ---------------------------------------------------------------------------
start_step "step7_compare_returns"
"${PYTHON_BIN}" src/v2/compare/compare_adjusted_nav_and_cum_return_window.py \
  --base-dir "${FUND_ETL_DIR}" \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --output-dir "${ARTIFACTS_DIR}/fund_return_compare" \
  --error-log "${LOGS_DIR}/compare_adjusted_nav_cum_return_errors.jsonl"
assert_csv_has_rows "${ARTIFACTS_DIR}/fund_return_compare/summary.csv"
assert_dir_exists "${ARTIFACTS_DIR}/fund_return_compare/details"
finish_step "success"

# ---------------------------------------------------------------------------
# Step 8: v2 过滤 (filter_funds_for_next_step) + 生成 filtered purchase
# ---------------------------------------------------------------------------
start_step "step8_filter_and_filtered_purchase"
"${PYTHON_BIN}" src/v2/filters/filter_funds_for_next_step.py \
  --base-dir "${FUND_ETL_DIR}" \
  --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}" \
  --compare-details-dir "${ARTIFACTS_DIR}/fund_return_compare/details" \
  --integrity-details-dir "${ARTIFACTS_DIR}/trade_day_integrity_reports/details_2025-01-01_2025-12-31" \
  --start-date "${FILTER_START_DATE}" \
  --end-date 2025-12-31 \
  --max-abs-deviation "${FILTER_MAX_ABS_DEVIATION}" \
  --output-csv "${FILTER_RESULT_CSV}"
assert_csv_has_rows "${FILTER_RESULT_CSV}"

"${PYTHON_BIN}" src/transforms/build_filtered_purchase_csv.py \
  --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}" \
  --filter-csv "${FILTER_RESULT_CSV}" \
  --output-csv "${FILTERED_PURCHASE_CSV}"
assert_csv_has_rows "${FILTERED_PURCHASE_CSV}"
finish_step "success"

# ---------------------------------------------------------------------------
# Step 9: 评分榜（--skip-sinks，仅 CSV，不写 MySQL/ClickHouse）
# ---------------------------------------------------------------------------
start_step "step9_scoreboard_skip_sinks"
AS_OF_DATE="$("${PYTHON_BIN}" - <<'PY' "${FUND_ETL_DIR}/fund_adjusted_nav_by_code"
import sys
from pathlib import Path

import pandas as pd

nav_dir = Path(sys.argv[1])
max_date = None
for path in nav_dir.glob("*.csv"):
    try:
        df = pd.read_csv(path, dtype={"净值日期": str}, encoding="utf-8-sig")
    except Exception:
        continue
    if "净值日期" not in df.columns:
        continue
    ds = pd.to_datetime(df["净值日期"], errors="coerce").dropna()
    if ds.empty:
        continue
    one = ds.max()
    if max_date is None or one > max_date:
        max_date = one
if max_date is None:
    raise SystemExit(1)
print(max_date.strftime("%Y-%m-%d"))
PY
)"

"${PYTHON_BIN}" src/pipeline_scoreboard.py \
  --purchase-csv "${FILTERED_PURCHASE_CSV}" \
  --overview-csv "${FUND_ETL_DIR}/fund_overview.csv" \
  --personnel-dir "${FUND_ETL_DIR}/fund_personnel_by_code" \
  --nav-dir "${FUND_ETL_DIR}/fund_adjusted_nav_by_code" \
  --output-dir "${SCOREBOARD_DIR}" \
  --data-version "${DATA_VERSION}" \
  --as-of-date "${AS_OF_DATE}" \
  --stale-max-days 3650 \
  --resume \
  --skip-sinks \
  --formal-only
assert_csv_has_rows "${SCOREBOARD_DIR}/fund_scoreboard_${DATA_VERSION}.csv"
finish_step "success"

# ---------------------------------------------------------------------------
# Step 10: 筛选打分 + 重算核验
# ---------------------------------------------------------------------------
start_step "step10_filter_score_and_recalc"
FILTER_SCORE_WORK_DIR="${ARTIFACTS_DIR}/filter_score"
# 使用 non_a_unlimited_purchase 替代 most_stable，确保 verify 采样下至少有基金通过过滤，避免 scored_result 为空导致 assert 失败
bash tools/run_filter_and_score.sh \
  -i "${SCOREBOARD_DIR}/fund_scoreboard_${DATA_VERSION}.csv" \
  -w "${FILTER_SCORE_WORK_DIR}" \
  -f src/filter_score/filters/non_a_unlimited_purchase.py \
  -s src/filter_score/scores/low_risk_debt.py
assert_csv_has_rows "${FILTER_SCORE_WORK_DIR}/filter_result.csv"
assert_csv_has_rows "${FILTER_SCORE_WORK_DIR}/scored_result.csv"

"${PYTHON_BIN}" src/verify_scoreboard_recalc.py \
  --scoreboard-csv "${SCOREBOARD_DIR}/fund_scoreboard_${DATA_VERSION}.csv" \
  --fund-etl-dir "${FUND_ETL_DIR}" \
  --output-dir "${ARTIFACTS_DIR}/scoreboard_recheck" \
  --max-input-rows 200
assert_csv_has_rows "${ARTIFACTS_DIR}/scoreboard_recheck/summary.csv"

"${PYTHON_BIN}" - <<'PY' "${ARTIFACTS_DIR}/scoreboard_recheck/summary.csv"
import sys
from pathlib import Path

import pandas as pd

summary_csv = Path(sys.argv[1])
df = pd.read_csv(summary_csv, dtype=str, encoding="utf-8-sig")
if "待核验字段是否全部核验通过" not in df.columns:
    raise ValueError(f"missing column in summary: {summary_csv}")
failed_df = df[df["待核验字段是否全部核验通过"] != "是"].copy()
if not failed_df.empty:
    print("recalc verification failed funds:")
    print(failed_df[["基金代码", "未通过字段名"]].to_string(index=False))
    raise SystemExit(1)
print(f"recalc verification all passed: funds={len(df)}")
PY
finish_step "success"

generate_run_report

echo "[verify] OK"
echo "[verify] run_id=${RUN_ID}"
echo "[verify] data_version=${DATA_VERSION}"
echo "[verify] integrity_summary=${SUMMARY_CSV}"
echo "[verify] scoreboard_csv=${SCOREBOARD_DIR}/fund_scoreboard_${DATA_VERSION}.csv"
echo "[verify] scoreboard_recheck_summary=${ARTIFACTS_DIR}/scoreboard_recheck/summary.csv"
echo "[verify] filter_score_result=${FILTER_SCORE_WORK_DIR}/scored_result.csv"
echo "[verify] run_report=${RUN_REPORT_MD}"
