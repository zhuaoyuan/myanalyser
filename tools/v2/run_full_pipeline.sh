#!/usr/bin/env bash

# v2: L1-only pipeline (fund_etl + adjusted_nav). No integrity/compare/filter/scoreboard/backtest.
# Usage:
#   bash myanalyser/tools/v2/run_full_pipeline.sh
#   bash myanalyser/tools/v2/run_full_pipeline.sh @/abs/path/to/fund_purchase.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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
  echo "[v2-full-run] usage: $0 [@/absolute/or/relative/path/to/fund_purchase.csv]"
  exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[v2-full-run] warning: VIRTUAL_ENV is not active. Please run:"
  echo "  source /Users/zhuaoyuan/cursor-workspace/finance/myanalyser/.venv312/bin/activate"
fi

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[v2-full-run] missing python/python3 in PATH"
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_full_run_v2}"
DATA_VERSION="${DATA_VERSION:-${RUN_ID}_db}"
VERIFY_ROOT="${PROJECT_ROOT}/data/versions/${RUN_ID}"
FUND_ETL_DIR="${VERIFY_ROOT}/fund_etl"
LOGS_DIR="${VERIFY_ROOT}/logs"
CHECKPOINT_DIR="${VERIFY_ROOT}/.checkpoints"

ETL_MAX_RETRIES="${ETL_MAX_RETRIES:-3}"
ETL_RETRY_SLEEP="${ETL_RETRY_SLEEP:-1.0}"
ETL_MAX_WORKERS="${ETL_MAX_WORKERS:-8}"
ETL_PROGRESS_INTERVAL="${ETL_PROGRESS_INTERVAL:-5.0}"

FUND_PURCHASE_EFFECTIVE_CSV="${FUND_ETL_DIR}/fund_purchase.csv"

assert_file_exists() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[v2-full-run] missing file: ${path}"
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

start_step() {
  local step="$1"
  echo "[v2-full-run] ${step} start"
}

finish_step() {
  local status="$1"
  echo "[v2-full-run] ${status}"
}

mkdir -p "${FUND_ETL_DIR}" "${LOGS_DIR}" "${CHECKPOINT_DIR}"

if [[ -n "${LOCAL_PURCHASE_CSV}" ]]; then
  if [[ ! -f "${LOCAL_PURCHASE_CSV}" ]]; then
    echo "[v2-full-run] local purchase csv not found: ${LOCAL_PURCHASE_CSV}"
    exit 1
  fi
  assert_purchase_csv_valid "${LOCAL_PURCHASE_CSV}" || {
    echo "[v2-full-run] local purchase csv invalid: ${LOCAL_PURCHASE_CSV}"
    exit 1
  }
  echo "[v2-full-run] local purchase csv mode: ${LOCAL_PURCHASE_CSV}"
fi

BONUS_SPLIT_REVISE_ARG=()
if [[ -n "${BONUS_SPLIT_REVISE_ROOT:-}" ]]; then
  if [[ "${BONUS_SPLIT_REVISE_ROOT}" != /* ]]; then
    BONUS_SPLIT_REVISE_ROOT="$(cd "${PROJECT_ROOT}" && cd "${BONUS_SPLIT_REVISE_ROOT}" && pwd)"
  fi
  BONUS_SPLIT_REVISE_ARG=(--bonus-split-revise-root "${BONUS_SPLIT_REVISE_ROOT}")
fi

start_step "step1_prepare_purchase"
if has_checkpoint "step2_fund_etl"; then
  echo "[v2-full-run] step1: checkpoint hit, purchase exists"
  assert_csv_has_rows "${FUND_ETL_DIR}/fund_purchase.csv"
elif [[ -n "${LOCAL_PURCHASE_CSV}" ]]; then
  echo "[v2-full-run] step1: copy local purchase csv"
  cp "${LOCAL_PURCHASE_CSV}" "${FUND_ETL_DIR}/fund_purchase.csv"
  assert_purchase_csv_valid "${FUND_ETL_DIR}/fund_purchase.csv" || {
    echo "[v2-full-run] copied purchase csv invalid: ${FUND_ETL_DIR}/fund_purchase.csv"
    exit 1
  }
else
  echo "[v2-full-run] step1: fund_etl step1 (fetch purchase)"
  "${PYTHON_BIN}" src/fund_etl.py \
    --run-id "${RUN_ID}" \
    --mode step1 \
    --max-retries "${ETL_MAX_RETRIES}" \
    --retry-sleep "${ETL_RETRY_SLEEP}"
fi
assert_csv_has_rows "${FUND_ETL_DIR}/fund_purchase.csv"
finish_step "success"

start_step "step2_fund_etl"
if has_checkpoint "step2_fund_etl"; then
  echo "[v2-full-run] step2: checkpoint hit, skip fund_etl"
  assert_csv_has_rows "${FUND_ETL_DIR}/fund_purchase.csv"
  assert_csv_has_rows "${FUND_ETL_DIR}/fund_overview.csv"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_nav_by_code"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_bonus_by_code"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_split_by_code"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_personnel_by_code"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_cum_return_by_code"
else
  echo "[v2-full-run] step2: fund_etl verify + step2~step7"
  "${PYTHON_BIN}" src/fund_etl.py \
    --run-id "${RUN_ID}" \
    --mode verify \
    --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}" \
    --max-retries "${ETL_MAX_RETRIES}" \
    --retry-sleep "${ETL_RETRY_SLEEP}" \
    --max-workers "${ETL_MAX_WORKERS}" \
    --progress-interval "${ETL_PROGRESS_INTERVAL}"
  for mode in step2 step3 step4 step5 step6 step7; do
    if [[ ${#BONUS_SPLIT_REVISE_ARG[@]} -gt 0 ]]; then
      "${PYTHON_BIN}" src/fund_etl.py \
        --run-id "${RUN_ID}" \
        --mode "${mode}" \
        --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}" \
        --max-retries "${ETL_MAX_RETRIES}" \
        --retry-sleep "${ETL_RETRY_SLEEP}" \
        --max-workers "${ETL_MAX_WORKERS}" \
        --progress-interval "${ETL_PROGRESS_INTERVAL}" \
        "${BONUS_SPLIT_REVISE_ARG[@]}"
    else
      "${PYTHON_BIN}" src/fund_etl.py \
        --run-id "${RUN_ID}" \
        --mode "${mode}" \
        --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}" \
        --max-retries "${ETL_MAX_RETRIES}" \
        --retry-sleep "${ETL_RETRY_SLEEP}" \
        --max-workers "${ETL_MAX_WORKERS}" \
        --progress-interval "${ETL_PROGRESS_INTERVAL}"
    fi
  done

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

BONUS_DIR_FOR_ADJ="${FUND_ETL_DIR}/fund_bonus_by_code"
SPLIT_DIR_FOR_ADJ="${FUND_ETL_DIR}/fund_split_by_code"
if [[ -d "${FUND_ETL_DIR}/revised_fund_bonus_by_code" && -d "${FUND_ETL_DIR}/revised_fund_split_by_code" ]]; then
  BONUS_DIR_FOR_ADJ="${FUND_ETL_DIR}/revised_fund_bonus_by_code"
  SPLIT_DIR_FOR_ADJ="${FUND_ETL_DIR}/revised_fund_split_by_code"
fi

start_step "step3_adjusted_nav"
if has_checkpoint "step3_adjusted_nav"; then
  echo "[v2-full-run] step3: checkpoint hit, skip adjusted nav"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_adjusted_nav_by_code"
else
  echo "[v2-full-run] step3: calculate adjusted nav"
  "${PYTHON_BIN}" src/adjusted_nav_tool.py \
    --nav-dir "${FUND_ETL_DIR}/fund_nav_by_code" \
    --bonus-dir "${BONUS_DIR_FOR_ADJ}" \
    --split-dir "${SPLIT_DIR_FOR_ADJ}" \
    --output-dir "${FUND_ETL_DIR}/fund_adjusted_nav_by_code" \
    --allow-missing-event-until 2020-12-31 \
    --fail-log "${LOGS_DIR}/failed_adjusted_nav.jsonl"
  assert_dir_has_csv "${FUND_ETL_DIR}/fund_adjusted_nav_by_code"
  mark_checkpoint "step3_adjusted_nav"
fi
finish_step "success"

echo "[v2-full-run] done: fund_etl_dir=${FUND_ETL_DIR}"
