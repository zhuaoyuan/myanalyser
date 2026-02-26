#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DB_INFRA_DIR="${WORKSPACE_ROOT}/fund_db_infra"

cd "${PROJECT_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[verify] warning: VIRTUAL_ENV is not active. Please run:"
  echo "  source /Users/zhuaoyuan/cursor-workspace/finance/myanalyser/.venv312/bin/activate"
fi


if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[verify] missing python/python3 in PATH"
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_verify_e2e}"
DATA_VERSION="${DATA_VERSION:-${RUN_ID}_db}"
VERIFY_ROOT="${PROJECT_ROOT}/data/versions/${RUN_ID}"
FUND_ETL_DIR="${VERIFY_ROOT}/fund_etl"
LOGS_DIR="${VERIFY_ROOT}/logs"
ARTIFACTS_DIR="${PROJECT_ROOT}/artifacts/verify_${RUN_ID}"
SCOREBOARD_DIR="${ARTIFACTS_DIR}/scoreboard"
BACKTEST_DIR="${ARTIFACTS_DIR}/backtest"
MYSQL_DDL="${DB_INFRA_DIR}/sql/mysql_schema.sql"
CLICKHOUSE_DDL="${DB_INFRA_DIR}/sql/clickhouse_schema.sql"

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
      echo "[verify] mysql not ready after ${max_wait}s"
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
      echo "[verify] clickhouse not ready after ${max_wait}s"
      exit 1
    fi
  done
}

echo "[verify] project_root=${PROJECT_ROOT}"
echo "[verify] run_id=${RUN_ID}"
echo "[verify] data_version=${DATA_VERSION}"

mkdir -p "${FUND_ETL_DIR}" "${LOGS_DIR}" "${ARTIFACTS_DIR}" "${SCOREBOARD_DIR}" "${BACKTEST_DIR}"

echo "[verify] step 1/10: unit tests"
"${PYTHON_BIN}" -m unittest discover -s tests -p "test_*.py" -v

echo "[verify] step 2/10: core CLI smoke tests (5 CLIs)"
"${PYTHON_BIN}" -m unittest -v \
  tests.test_cli_integration.CoreCliIntegrationTest.test_fund_etl_cli_run_id_layout \
  tests.test_cli_integration.CoreCliIntegrationTest.test_pipeline_cli_smoke_skip_sinks_with_run_id_layout \
  tests.test_cli_integration.CoreCliIntegrationTest.test_backtest_cli_smoke_with_run_id_layout \
  tests.test_cli_integration.CoreCliIntegrationTest.test_compare_cli_with_run_id_layout \
  tests.test_cli_integration.CoreCliIntegrationTest.test_check_trade_day_integrity_cli_with_run_id_layout

echo "[verify] step 3/10: start db infra"
assert_file_exists "${WORKSPACE_ROOT}/fund_db_infra/docker-compose.yml"
docker_compose_cmd -f "${WORKSPACE_ROOT}/fund_db_infra/docker-compose.yml" up -d
wait_mysql_ready
wait_clickhouse_ready

echo "[verify] step 4/10: fund_etl verify + step1"
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode verify
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step1
assert_file_exists "${FUND_ETL_DIR}/fund_purchase.csv"
assert_csv_has_rows "${FUND_ETL_DIR}/fund_purchase.csv"

echo "[verify] step 5/10: sampling 101 funds (top100 + 163402)"
RUN_ID="${RUN_ID}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os

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

top100 = df[df["基金代码"] != target_code].head(100).copy()
if top100.shape[0] < 100:
    raise ValueError(f"fund_purchase rows not enough for sampling 100 rows: got={top100.shape[0]}")

target_row = df[df["基金代码"] == target_code].head(1).copy()
if target_row.empty:
    target_row = top100.head(1).copy()
    target_row["基金代码"] = target_code
    for col in target_row.columns:
        if col != "基金代码":
            target_row[col] = ""

sample_df = pd.concat([top100, target_row], ignore_index=True)
if sample_df.shape[0] != 101:
    raise ValueError(f"sample rows expected 101, got={sample_df.shape[0]}")

sample_df.to_csv(purchase_csv, index=False, encoding="utf-8-sig")
PY
assert_csv_has_rows "${FUND_ETL_DIR}/fund_purchase.csv"

echo "[verify] step 6/10: fund_etl step2~step7 on sampled funds"
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step2 --max-workers 8
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step3
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step4
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step5
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step6
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step7
assert_csv_has_rows "${FUND_ETL_DIR}/fund_overview.csv"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_nav_by_code"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_bonus_by_code"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_split_by_code"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_personnel_by_code"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_cum_return_by_code"

echo "[verify] step 7/10: calculate adjusted nav"
"${PYTHON_BIN}" src/adjusted_nav_tool.py \
  --nav-dir "${FUND_ETL_DIR}/fund_nav_by_code" \
  --bonus-dir "${FUND_ETL_DIR}/fund_bonus_by_code" \
  --split-dir "${FUND_ETL_DIR}/fund_split_by_code" \
  --output-dir "${FUND_ETL_DIR}/fund_adjusted_nav_by_code" \
  --allow-missing-event-until 2020-12-31 \
  --fail-log "${LOGS_DIR}/failed_adjusted_nav.jsonl"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_adjusted_nav_by_code"

echo "[verify] step 8/10: data integrity reports"
"${PYTHON_BIN}" src/check_trade_day_data_integrity.py \
  --base-dir "${FUND_ETL_DIR}" \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --trade-dates-csv "${PROJECT_ROOT}/data/common/trade_dates.csv" \
  --output-dir "${ARTIFACTS_DIR}/trade_day_integrity_reports"

SUMMARY_CSV="$(ls -1 "${ARTIFACTS_DIR}"/trade_day_integrity_reports/trade_day_integrity_summary_*.csv | head -n 1)"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "[verify] missing integrity summary csv"
  exit 1
fi
assert_csv_has_rows "${SUMMARY_CSV}"

echo "[verify] step 9/10: compare adjusted nav vs cum return"
"${PYTHON_BIN}" src/compare_adjusted_nav_and_cum_return.py \
  --base-dir "${FUND_ETL_DIR}" \
  --output-dir "${ARTIFACTS_DIR}/fund_return_compare" \
  --error-log "${LOGS_DIR}/compare_adjusted_nav_cum_return_errors.jsonl"
assert_csv_has_rows "${ARTIFACTS_DIR}/fund_return_compare/summary.csv"
assert_dir_exists "${ARTIFACTS_DIR}/fund_return_compare/details"

echo "[verify] step 10/10: pipeline scoreboard -> db + backtest"
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

"${PYTHON_BIN}" src/pipeline_scoreboard.py \
  --purchase-csv "${FUND_ETL_DIR}/fund_purchase.csv" \
  --overview-csv "${FUND_ETL_DIR}/fund_overview.csv" \
  --personnel-dir "${FUND_ETL_DIR}/fund_personnel_by_code" \
  --nav-dir "${FUND_ETL_DIR}/fund_adjusted_nav_by_code" \
  --output-dir "${SCOREBOARD_DIR}" \
  --data-version "${DATA_VERSION}" \
  --as-of-date "${AS_OF_DATE}" \
  --stale-max-days 3650 \
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

assert_csv_has_rows "${SCOREBOARD_DIR}/fund_scoreboard_${DATA_VERSION}.csv"

"${PYTHON_BIN}" src/backtest_portfolio.py \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --output-dir "${BACKTEST_DIR}" \
  --trade-dates-csv "${PROJECT_ROOT}/data/common/trade_dates.csv" \
  --selection-rule-id "verify_e2e_top5" \
  --selection-data-version "${DATA_VERSION}" \
  --selection-where "1" \
  --selection-order-by "annual_return_rank ASC, fund_code ASC" \
  --selection-limit 5 \
  --nav-data-version "${DATA_VERSION}" \
  --clickhouse-db fund_analysis \
  --clickhouse-container fund_clickhouse

assert_csv_has_rows "${BACKTEST_DIR}/backtest_window_detail.csv"
assert_file_exists "${BACKTEST_DIR}/backtest_report.md"

echo "[verify] OK"
echo "[verify] run_id=${RUN_ID}"
echo "[verify] data_version=${DATA_VERSION}"
echo "[verify] integrity_summary=${SUMMARY_CSV}"
echo "[verify] scoreboard_csv=${SCOREBOARD_DIR}/fund_scoreboard_${DATA_VERSION}.csv"
echo "[verify] backtest_report=${BACKTEST_DIR}/backtest_report.md"
