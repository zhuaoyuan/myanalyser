#!/usr/bin/env bash

# ================================
# run_full_pipeline.sh 使用说明（完整示例）
# ================================
# 1) 最小可用（使用默认参数）：
#    bash myanalyser/tools/run_full_pipeline.sh
#
# 2) 指定本地 purchase csv（参数写成 @<path>）：
#    bash myanalyser/tools/run_full_pipeline.sh @/absolute/or/relative/path/to/fund_purchase.csv
#
# 3) 完整参数示例（推荐复制后按需改值）：
#    RUN_ID=20260228_233000_full_run \
#    DATA_VERSION=20260228_233000_db \
#    ETL_MAX_RETRIES=3 \
#    ETL_RETRY_SLEEP=1.0 \
#    ETL_MAX_WORKERS=8 \
#    ETL_PROGRESS_INTERVAL=5.0 \
#    STALE_MAX_DAYS=2 \
#    INTEGRITY_START_DATE=2020-01-01 \
#    INTEGRITY_END_DATE=2026-02-28 \
#    FILTER_START_DATE=2023-01-01 \
#    FILTER_MAX_ABS_DEVIATION=0.02 \
#    bash myanalyser/tools/run_full_pipeline.sh @/absolute/path/to/fund_purchase.csv
#
# 4) 后台运行并写日志（长任务常用）：
#    nohup bash -lc '
#      export RUN_ID=20260228_233000_full_run
#      export DATA_VERSION=20260228_233000_db
#      export ETL_MAX_WORKERS=16
#      export FILTER_START_DATE=2023-01-01
#      bash myanalyser/tools/run_full_pipeline.sh @/absolute/path/to/fund_purchase.csv
#    ' > /tmp/run_full_pipeline.log 2>&1 &
#
# 5) 变量含义（本脚本内支持覆盖）：
#    RUN_ID                 运行批次 ID（默认：当前时间_full_run）
#    DATA_VERSION           入库/产物版本（默认：${RUN_ID}_db）
#    ETL_MAX_RETRIES        ETL 重试次数（默认：3）
#    ETL_RETRY_SLEEP        ETL 重试等待秒数（默认：1.0）
#    ETL_MAX_WORKERS        ETL 并发数（默认：8）
#    ETL_PROGRESS_INTERVAL  ETL 进度打印间隔秒（默认：5.0）
#    STALE_MAX_DAYS         有效性校验允许滞后天数（默认：2）
#    INTEGRITY_START_DATE   交易日完整性校验开始日（默认：2020-01-01）
#    INTEGRITY_END_DATE     交易日完整性校验结束日（默认：当天）
#    FILTER_START_DATE      收益比较过滤起始日（默认：2023-01-01）
#    FILTER_MAX_ABS_DEVIATION  收益偏差过滤阈值（默认：0.02）
#
# 注意：
# - 仅支持 0 或 1 个位置参数；若传入，必须是 @<csv_path> 格式。
# - 建议先激活 venv，或确保 python/python3 在 PATH 中可用。
# - 默认输出目录在 myanalyser/artifacts 和 myanalyser/data/versions 下。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

find_db_infra_dir() {
  local cursor="$1"
  while true; do
    local candidate="${cursor}/fund_db_infra"
    if [[ -f "${candidate}/docker-compose.yml" ]]; then
      echo "${candidate}"
      return 0
    fi
    local parent
    parent="$(cd "${cursor}/.." && pwd)"
    if [[ "${parent}" == "${cursor}" ]]; then
      break
    fi
    cursor="${parent}"
  done
  return 1
}

if [[ -n "${DB_INFRA_DIR:-}" ]]; then
  if [[ "${DB_INFRA_DIR}" != /* ]]; then
    DB_INFRA_DIR="$(cd "${PROJECT_ROOT}" && cd "${DB_INFRA_DIR}" && pwd)"
  fi
else
  DB_INFRA_DIR="$(find_db_infra_dir "${PROJECT_ROOT}")" || {
    echo "[full-run] cannot locate fund_db_infra from ${PROJECT_ROOT}; set DB_INFRA_DIR explicitly"
    exit 1
  }
fi

cd "${PROJECT_ROOT}"

LOCAL_PURCHASE_CSV=""
if [[ $# -gt 0 ]]; then
  case "$1" in
    @*)
      LOCAL_PURCHASE_CSV="${1#@}"
      shift
      ;;
  esac
fi

if [[ $# -gt 0 ]]; then
  echo "[full-run] usage: $0 [@/absolute/or/relative/path/to/fund_purchase.csv]"
  exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[full-run] warning: VIRTUAL_ENV is not active. Please run:"
  echo "  source /Users/zhuaoyuan/cursor-workspace/finance/myanalyser/.venv312/bin/activate"
fi

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[full-run] missing python/python3 in PATH"
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_full_run}"
DATA_VERSION="${DATA_VERSION:-${RUN_ID}_db}"
VERIFY_ROOT="${PROJECT_ROOT}/data/versions/${RUN_ID}"
FUND_ETL_DIR="${VERIFY_ROOT}/fund_etl"
LOGS_DIR="${VERIFY_ROOT}/logs"
ARTIFACTS_DIR="${PROJECT_ROOT}/artifacts/full_run_${RUN_ID}"
SCOREBOARD_DIR="${ARTIFACTS_DIR}/scoreboard"
MYSQL_DDL="${DB_INFRA_DIR}/sql/mysql_schema.sql"
CLICKHOUSE_DDL="${DB_INFRA_DIR}/sql/clickhouse_schema.sql"
CHECKPOINT_DIR="${ARTIFACTS_DIR}/.checkpoints"

ETL_MAX_RETRIES="${ETL_MAX_RETRIES:-3}"
ETL_RETRY_SLEEP="${ETL_RETRY_SLEEP:-1.0}"
ETL_MAX_WORKERS="${ETL_MAX_WORKERS:-8}"
ETL_PROGRESS_INTERVAL="${ETL_PROGRESS_INTERVAL:-5.0}"
STALE_MAX_DAYS="${STALE_MAX_DAYS:-2}"

INTEGRITY_START_DATE="${INTEGRITY_START_DATE:-2020-01-01}"
INTEGRITY_END_DATE="${INTEGRITY_END_DATE:-$(date +%Y-%m-%d)}"

FILTER_START_DATE="${FILTER_START_DATE:-2023-01-01}"
FILTER_MAX_ABS_DEVIATION="${FILTER_MAX_ABS_DEVIATION:-0.02}"
FILTER_RESULT_CSV="${ARTIFACTS_DIR}/filtered_fund_candidates.csv"
FILTERED_PURCHASE_CSV="${FUND_ETL_DIR}/fund_purchase_for_step10_filtered.csv"
INTEGRITY_DETAILS_DIR="${ARTIFACTS_DIR}/trade_day_integrity_reports/details_${INTEGRITY_START_DATE}_${INTEGRITY_END_DATE}"
INTEGRITY_SUMMARY_CSV="${ARTIFACTS_DIR}/trade_day_integrity_reports/trade_day_integrity_summary_${INTEGRITY_START_DATE}_${INTEGRITY_END_DATE}.csv"
COMPARE_SUMMARY_CSV="${ARTIFACTS_DIR}/fund_return_compare/summary.csv"
SCOREBOARD_CSV="${SCOREBOARD_DIR}/fund_scoreboard_${DATA_VERSION}.csv"
RUN_REPORT_STEPS_CSV="${ARTIFACTS_DIR}/run_report_steps.csv"
RUN_REPORT_SUMMARY_CSV="${ARTIFACTS_DIR}/run_report_summary.csv"
RUN_REPORT_MD="${ARTIFACTS_DIR}/run_report.md"
CURRENT_STEP=""
STEP_START_TS=0

assert_file_exists() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[full-run] missing file: ${path}"
    exit 1
  fi
}

assert_dir_exists() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[full-run] missing directory: ${path}"
    exit 1
  fi
}

assert_csv_has_rows() {
  local path="$1"
  "${PYTHON_BIN}" - <<'PY' "${path}"
from pathlib import Path
import pandas as pd
import sys

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
from pathlib import Path
import sys

dir_path = Path(sys.argv[1])
if not dir_path.exists():
    raise SystemExit(1)
files = sorted(dir_path.glob("*.csv"))
if not files:
    raise SystemExit(2)
PY
}

assert_purchase_csv_valid() {
  local path="$1"
  "${PYTHON_BIN}" - <<'PY' "${path}"
from pathlib import Path
import pandas as pd
import sys

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
if "基金代码" not in df.columns:
    raise SystemExit(2)
if df.empty:
    raise SystemExit(3)
PY
}

checkpoint_path() {
  local step="$1"
  echo "${CHECKPOINT_DIR}/${step}.ok"
}

mark_checkpoint() {
  local step="$1"
  mkdir -p "${CHECKPOINT_DIR}"
  printf 'run_id=%s\ndata_version=%s\nts=%s\n' "${RUN_ID}" "${DATA_VERSION}" "$(date +%Y-%m-%dT%H:%M:%S)" >"$(checkpoint_path "${step}")"
}

has_checkpoint() {
  local step="$1"
  local path
  path="$(checkpoint_path "${step}")"
  [[ -f "${path}" ]]
}

docker_compose_cmd() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  else
    docker-compose "$@"
  fi
}

wait_mysql_ready() {
  local max_wait=120
  local waited=0
  while true; do
    if docker exec fund_mysql mysqladmin ping -h127.0.0.1 -uroot -pyour_strong_password --silent >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
    if (( waited >= max_wait )); then
      echo "[full-run] mysql not ready after ${max_wait}s"
      exit 1
    fi
  done
}

wait_clickhouse_ready() {
  local max_wait=120
  local waited=0
  while true; do
    if docker exec fund_clickhouse clickhouse-client --query "SELECT 1" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
    if (( waited >= max_wait )); then
      echo "[full-run] clickhouse not ready after ${max_wait}s"
      exit 1
    fi
  done
}

start_step() {
  CURRENT_STEP="$1"
  STEP_START_TS="$(date +%s)"
  echo "[full-run] ${CURRENT_STEP}"
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
from pathlib import Path
import json
import sys
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

error_stage_count: dict[str, int] = {}
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

summary = pd.DataFrame(
    [
        {"指标": "总步骤数", "值": total_steps},
        {"指标": "成功步骤数", "值": ok_steps},
        {"指标": "步骤成功率(%)", "值": round(success_rate, 2)},
        {"指标": "过滤前基金数", "值": purchase_before},
        {"指标": "过滤后基金数", "值": purchase_after},
        {"指标": "被过滤基金数", "值": filtered_yes},
    ]
)
summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

err_text = "无"
if error_stage_count:
    err_text = "; ".join(f"{k}:{v}" for k, v in sorted(error_stage_count.items()))

lines = [
    "# 运行报告汇总",
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

echo "[full-run] project_root=${PROJECT_ROOT}"
echo "[full-run] run_id=${RUN_ID}"
echo "[full-run] data_version=${DATA_VERSION}"

mkdir -p "${FUND_ETL_DIR}" "${LOGS_DIR}" "${ARTIFACTS_DIR}" "${SCOREBOARD_DIR}" "${CHECKPOINT_DIR}"
printf 'step,status,duration_seconds\n' >"${RUN_REPORT_STEPS_CSV}"

if [[ -n "${LOCAL_PURCHASE_CSV}" ]]; then
  if [[ ! -f "${LOCAL_PURCHASE_CSV}" ]]; then
    echo "[full-run] local purchase csv not found: ${LOCAL_PURCHASE_CSV}"
    exit 1
  fi
  assert_purchase_csv_valid "${LOCAL_PURCHASE_CSV}" || {
    echo "[full-run] local purchase csv is invalid (need non-empty file with 基金代码 column): ${LOCAL_PURCHASE_CSV}"
    exit 1
  }
  echo "[full-run] local purchase csv mode enabled: ${LOCAL_PURCHASE_CSV}"
fi

start_step "step1_start_db"
assert_file_exists "${DB_INFRA_DIR}/docker-compose.yml"
docker_compose_cmd -f "${DB_INFRA_DIR}/docker-compose.yml" up -d
wait_mysql_ready
wait_clickhouse_ready
finish_step "success"

start_step "step2_fund_etl"
if has_checkpoint "step2_fund_etl"; then
  echo "[full-run] step 2/7: checkpoint hit, skip fund_etl"
  assert_csv_has_rows "${FUND_ETL_DIR}/fund_purchase.csv"
  assert_csv_has_rows "${FUND_ETL_DIR}/fund_overview.csv"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_nav_by_code"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_bonus_by_code"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_split_by_code"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_personnel_by_code"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_cum_return_by_code"
else
  if [[ -n "${LOCAL_PURCHASE_CSV}" ]]; then
    echo "[full-run] step 2/7: local purchase csv + fund_etl (verify + step2~step7)"
    cp "${LOCAL_PURCHASE_CSV}" "${FUND_ETL_DIR}/fund_purchase.csv"
    assert_purchase_csv_valid "${FUND_ETL_DIR}/fund_purchase.csv" || {
      echo "[full-run] copied purchase csv is invalid: ${FUND_ETL_DIR}/fund_purchase.csv"
      exit 1
    }
    "${PYTHON_BIN}" src/fund_etl.py \
      --run-id "${RUN_ID}" \
      --mode verify \
      --max-retries "${ETL_MAX_RETRIES}" \
      --retry-sleep "${ETL_RETRY_SLEEP}" \
      --max-workers "${ETL_MAX_WORKERS}" \
      --progress-interval "${ETL_PROGRESS_INTERVAL}"
    for mode in step2 step3 step4 step5 step6 step7; do
      "${PYTHON_BIN}" src/fund_etl.py \
        --run-id "${RUN_ID}" \
        --mode "${mode}" \
        --max-retries "${ETL_MAX_RETRIES}" \
        --retry-sleep "${ETL_RETRY_SLEEP}" \
        --max-workers "${ETL_MAX_WORKERS}" \
        --progress-interval "${ETL_PROGRESS_INTERVAL}"
    done
  else
    echo "[full-run] step 2/7: full fund_etl (verify + step1~step7)"
    "${PYTHON_BIN}" src/fund_etl.py \
      --run-id "${RUN_ID}" \
      --mode all \
      --max-retries "${ETL_MAX_RETRIES}" \
      --retry-sleep "${ETL_RETRY_SLEEP}" \
      --max-workers "${ETL_MAX_WORKERS}" \
      --progress-interval "${ETL_PROGRESS_INTERVAL}"
  fi
  assert_csv_has_rows "${FUND_ETL_DIR}/fund_purchase.csv"
  assert_csv_has_rows "${FUND_ETL_DIR}/fund_overview.csv"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_nav_by_code"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_bonus_by_code"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_split_by_code"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_personnel_by_code"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_cum_return_by_code"
  mark_checkpoint "step2_fund_etl"
fi
finish_step "success"

start_step "step3_adjusted_nav"
if has_checkpoint "step3_adjusted_nav"; then
  echo "[full-run] step 3/7: checkpoint hit, skip adjusted nav"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_adjusted_nav_by_code"
else
  echo "[full-run] step 3/7: calculate adjusted nav"
  "${PYTHON_BIN}" src/adjusted_nav_tool.py \
    --nav-dir "${FUND_ETL_DIR}/fund_nav_by_code" \
    --bonus-dir "${FUND_ETL_DIR}/fund_bonus_by_code" \
    --split-dir "${FUND_ETL_DIR}/fund_split_by_code" \
    --output-dir "${FUND_ETL_DIR}/fund_adjusted_nav_by_code" \
    --allow-missing-event-until 2020-12-31 \
    --fail-log "${LOGS_DIR}/failed_adjusted_nav.jsonl"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_adjusted_nav_by_code"
  mark_checkpoint "step3_adjusted_nav"
fi
finish_step "success"

start_step "step4_integrity"
if has_checkpoint "step4_integrity"; then
  echo "[full-run] step 4/7: checkpoint hit, skip trade-day integrity"
  assert_csv_has_rows "${INTEGRITY_SUMMARY_CSV}"
  assert_dir_exists "${INTEGRITY_DETAILS_DIR}"
else
  echo "[full-run] step 4/7: trade-day integrity check"
  "${PYTHON_BIN}" src/check_trade_day_data_integrity.py \
    --base-dir "${FUND_ETL_DIR}" \
    --start-date "${INTEGRITY_START_DATE}" \
    --end-date "${INTEGRITY_END_DATE}" \
    --trade-dates-csv "${PROJECT_ROOT}/data/common/trade_dates.csv" \
    --output-dir "${ARTIFACTS_DIR}/trade_day_integrity_reports"
  assert_csv_has_rows "${INTEGRITY_SUMMARY_CSV}"
  assert_dir_exists "${INTEGRITY_DETAILS_DIR}"
  mark_checkpoint "step4_integrity"
fi
finish_step "success"

start_step "step5_compare"
if has_checkpoint "step5_compare"; then
  echo "[full-run] step 5/7: checkpoint hit, skip compare adjusted nav vs cum return"
  assert_csv_has_rows "${COMPARE_SUMMARY_CSV}"
  assert_dir_exists "${ARTIFACTS_DIR}/fund_return_compare/details"
else
  echo "[full-run] step 5/7: compare adjusted nav vs cum return"
  "${PYTHON_BIN}" src/compare_adjusted_nav_and_cum_return.py \
    --base-dir "${FUND_ETL_DIR}" \
    --output-dir "${ARTIFACTS_DIR}/fund_return_compare" \
    --error-log "${LOGS_DIR}/compare_adjusted_nav_cum_return_errors.jsonl"
  assert_csv_has_rows "${COMPARE_SUMMARY_CSV}"
  assert_dir_exists "${ARTIFACTS_DIR}/fund_return_compare/details"
  mark_checkpoint "step5_compare"
fi
finish_step "success"

start_step "step6_filter"
if has_checkpoint "step6_filter"; then
  echo "[full-run] step 6/7: checkpoint hit, skip fund filter"
  assert_csv_has_rows "${FILTER_RESULT_CSV}"
else
  echo "[full-run] step 6/7: filter fund list before step10"
  "${PYTHON_BIN}" src/filter_funds_for_next_step.py \
    --base-dir "${FUND_ETL_DIR}" \
    --compare-details-dir "${ARTIFACTS_DIR}/fund_return_compare/details" \
    --integrity-details-dir "${INTEGRITY_DETAILS_DIR}" \
    --start-date "${FILTER_START_DATE}" \
    --max-abs-deviation "${FILTER_MAX_ABS_DEVIATION}" \
    --output-csv "${FILTER_RESULT_CSV}"
  assert_csv_has_rows "${FILTER_RESULT_CSV}"
  mark_checkpoint "step6_filter"
fi
finish_step "success"

start_step "step6b_filtered_purchase"
if has_checkpoint "step6b_filtered_purchase"; then
  echo "[full-run] step 6b: checkpoint hit, skip filtered purchase generation"
else
  "${PYTHON_BIN}" src/transforms/build_filtered_purchase_csv.py \
    --purchase-csv "${FUND_ETL_DIR}/fund_purchase.csv" \
    --filter-csv "${FILTER_RESULT_CSV}" \
    --output-csv "${FILTERED_PURCHASE_CSV}"
  mark_checkpoint "step6b_filtered_purchase"
fi
assert_csv_has_rows "${FILTERED_PURCHASE_CSV}"
finish_step "success"

AS_OF_DATE="$("${PYTHON_BIN}" - <<'PY' "${FUND_ETL_DIR}/fund_adjusted_nav_by_code"
from pathlib import Path
import sys

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

start_step "step7_scoreboard"
if has_checkpoint "step7_scoreboard"; then
  echo "[full-run] step 7/7: checkpoint hit, skip pipeline scoreboard -> db"
  assert_csv_has_rows "${SCOREBOARD_CSV}"
else
  echo "[full-run] step 7/7: pipeline scoreboard (formal-only, no DB)"
  "${PYTHON_BIN}" src/pipeline_scoreboard.py \
    --purchase-csv "${FILTERED_PURCHASE_CSV}" \
    --overview-csv "${FUND_ETL_DIR}/fund_overview.csv" \
    --personnel-dir "${FUND_ETL_DIR}/fund_personnel_by_code" \
    --nav-dir "${FUND_ETL_DIR}/fund_adjusted_nav_by_code" \
    --output-dir "${SCOREBOARD_DIR}" \
    --data-version "${DATA_VERSION}" \
    --as-of-date "${AS_OF_DATE}" \
    --stale-max-days "${STALE_MAX_DAYS}" \
    --formal-only
  assert_csv_has_rows "${SCOREBOARD_CSV}"
  mark_checkpoint "step7_scoreboard"
fi
finish_step "success"

generate_run_report

echo "[full-run] OK"
echo "[full-run] run_id=${RUN_ID}"
echo "[full-run] data_version=${DATA_VERSION}"
echo "[full-run] integrity_summary=${INTEGRITY_SUMMARY_CSV}"
echo "[full-run] filtered_fund_csv=${FILTER_RESULT_CSV}"
echo "[full-run] filtered_purchase_csv=${FILTERED_PURCHASE_CSV}"
echo "[full-run] scoreboard_csv=${SCOREBOARD_CSV}"
echo "[full-run] run_report=${RUN_REPORT_MD}"
