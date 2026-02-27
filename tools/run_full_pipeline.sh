#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DB_INFRA_DIR="${WORKSPACE_ROOT}/fund_db_infra"

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
STALE_MAX_DAYS="${STALE_MAX_DAYS:-3650}"

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

echo "[full-run] project_root=${PROJECT_ROOT}"
echo "[full-run] run_id=${RUN_ID}"
echo "[full-run] data_version=${DATA_VERSION}"

mkdir -p "${FUND_ETL_DIR}" "${LOGS_DIR}" "${ARTIFACTS_DIR}" "${SCOREBOARD_DIR}" "${CHECKPOINT_DIR}"

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

echo "[full-run] step 1/7: start db infra"
assert_file_exists "${WORKSPACE_ROOT}/fund_db_infra/docker-compose.yml"
docker_compose_cmd -f "${WORKSPACE_ROOT}/fund_db_infra/docker-compose.yml" up -d
wait_mysql_ready
wait_clickhouse_ready

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

if has_checkpoint "step6b_filtered_purchase"; then
  echo "[full-run] step 6b: checkpoint hit, skip filtered purchase generation"
else
  "${PYTHON_BIN}" - <<'PY' "${FUND_ETL_DIR}/fund_purchase.csv" "${FILTER_RESULT_CSV}" "${FILTERED_PURCHASE_CSV}"
from pathlib import Path
import sys

import pandas as pd

purchase_csv = Path(sys.argv[1])
filter_csv = Path(sys.argv[2])
output_csv = Path(sys.argv[3])

purchase_df = pd.read_csv(purchase_csv, dtype={"基金代码": str}, encoding="utf-8-sig")
filter_df = pd.read_csv(filter_csv, dtype={"基金编码": str}, encoding="utf-8-sig")

if "基金代码" not in purchase_df.columns:
    raise ValueError(f"missing 基金代码 column: {purchase_csv}")
if "基金编码" not in filter_df.columns or "是否过滤" not in filter_df.columns:
    raise ValueError(f"missing 基金编码/是否过滤 columns: {filter_csv}")

purchase_df["基金代码"] = purchase_df["基金代码"].map(lambda v: str(v).strip().zfill(6))
filter_df["基金编码"] = filter_df["基金编码"].map(lambda v: str(v).strip().zfill(6))

kept_codes = set(filter_df.loc[filter_df["是否过滤"] == "否", "基金编码"].dropna().tolist())
kept_df = purchase_df[purchase_df["基金代码"].isin(kept_codes)].copy()
if kept_df.empty:
    raise ValueError("all funds are filtered out; cannot continue step10 pipeline")

kept_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"filtered_purchase_rows={len(kept_df)}")
print(f"filtered_purchase_csv={output_csv}")
PY
  mark_checkpoint "step6b_filtered_purchase"
fi
assert_csv_has_rows "${FILTERED_PURCHASE_CSV}"

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

if has_checkpoint "step7_scoreboard"; then
  echo "[full-run] step 7/7: checkpoint hit, skip pipeline scoreboard -> db"
  assert_csv_has_rows "${SCOREBOARD_CSV}"
else
  echo "[full-run] step 7/7: pipeline scoreboard -> db"
  "${PYTHON_BIN}" src/pipeline_scoreboard.py \
    --purchase-csv "${FILTERED_PURCHASE_CSV}" \
    --overview-csv "${FUND_ETL_DIR}/fund_overview.csv" \
    --personnel-dir "${FUND_ETL_DIR}/fund_personnel_by_code" \
    --nav-dir "${FUND_ETL_DIR}/fund_adjusted_nav_by_code" \
    --output-dir "${SCOREBOARD_DIR}" \
    --data-version "${DATA_VERSION}" \
    --as-of-date "${AS_OF_DATE}" \
    --stale-max-days "${STALE_MAX_DAYS}" \
    --apply-ddl \
    --mysql-ddl "${MYSQL_DDL}" \
    --clickhouse-ddl "${CLICKHOUSE_DDL}" \
    --mysql-host 127.0.0.1 \
    --mysql-port 3306 \
    --mysql-user root \
    --mysql-password your_strong_password \
    --mysql-db fund_analysis \
    --clickhouse-db fund_analysis \
    --clickhouse-container fund_clickhouse
  assert_csv_has_rows "${SCOREBOARD_CSV}"
  mark_checkpoint "step7_scoreboard"
fi

echo "[full-run] OK"
echo "[full-run] run_id=${RUN_ID}"
echo "[full-run] data_version=${DATA_VERSION}"
echo "[full-run] integrity_summary=${INTEGRITY_SUMMARY_CSV}"
echo "[full-run] filtered_fund_csv=${FILTER_RESULT_CSV}"
echo "[full-run] filtered_purchase_csv=${FILTERED_PURCHASE_CSV}"
echo "[full-run] scoreboard_csv=${SCOREBOARD_CSV}"
