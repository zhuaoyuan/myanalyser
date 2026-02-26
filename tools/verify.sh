#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[verify] warning: VIRTUAL_ENV is not active. Please run:"
  echo "  source /Users/zhuaoyuan/cursor-workspace/finance/myanalyser/.venv312/bin/activate"
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_verify}"
VERIFY_ROOT="${PROJECT_ROOT}/data/versions/${RUN_ID}"
FUND_ETL_DIR="${VERIFY_ROOT}/fund_etl"
LOGS_DIR="${VERIFY_ROOT}/logs"
ARTIFACTS_DIR="${PROJECT_ROOT}/artifacts/verify_${RUN_ID}"

echo "[verify] project_root=${PROJECT_ROOT}"
echo "[verify] run_id=${RUN_ID}"

mkdir -p "${FUND_ETL_DIR}" "${LOGS_DIR}" "${ARTIFACTS_DIR}"

echo "[verify] step 1/4: unit tests"
python -m unittest discover -s tests -p "test_*.py" -v

echo "[verify] step 2/4: core CLI smoke tests (5 CLIs)"
python -m unittest -v \
  tests.test_cli_integration.CoreCliIntegrationTest.test_fund_etl_cli_run_id_layout \
  tests.test_cli_integration.CoreCliIntegrationTest.test_pipeline_cli_smoke_skip_sinks_with_run_id_layout \
  tests.test_cli_integration.CoreCliIntegrationTest.test_backtest_cli_smoke_with_run_id_layout \
  tests.test_cli_integration.CoreCliIntegrationTest.test_compare_cli_with_run_id_layout \
  tests.test_cli_integration.CoreCliIntegrationTest.test_check_trade_day_integrity_cli_with_run_id_layout

echo "[verify] step 3/4: build tiny dataset under data/versions/${RUN_ID}"
RUN_ID="${RUN_ID}" python - <<'PY'
from pathlib import Path
import pandas as pd
import os

root = Path(".").resolve()
run_id = os.environ["RUN_ID"]
verify_root = root / "data" / "versions" / run_id
fund = verify_root / "fund_etl"
(fund / "fund_adjusted_nav_by_code").mkdir(parents=True, exist_ok=True)

pd.DataFrame(
    [
        {"基金代码": "163402", "净值日期": "2024-01-02", "单位净值": 1.0, "复权净值": 1.0, "cumulative_factor": 1.0},
        {"基金代码": "163402", "净值日期": "2024-01-03", "单位净值": 1.1, "复权净值": 1.1, "cumulative_factor": 1.0},
    ]
).to_csv(fund / "fund_adjusted_nav_by_code" / "163402.csv", index=False, encoding="utf-8-sig")

pd.DataFrame([{"基金代码": "163402", "成立日期/规模": "2010-01-01"}]).to_csv(
    fund / "fund_overview.csv",
    index=False,
    encoding="utf-8-sig",
)
PY

echo "[verify] step 4/4: data integrity check CLI"
python src/check_trade_day_data_integrity.py \
  --base-dir "${FUND_ETL_DIR}" \
  --start-date 2024-01-02 \
  --end-date 2024-01-03 \
  --trade-dates-csv "${PROJECT_ROOT}/data/common/trade_dates.csv" \
  --output-dir "${ARTIFACTS_DIR}/trade_day_integrity_reports"

SUMMARY_CSV="$(ls -1 "${ARTIFACTS_DIR}"/trade_day_integrity_reports/trade_day_integrity_summary_*.csv | head -n 1)"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "[verify] missing integrity summary csv"
  exit 1
fi

echo "[verify] OK"
echo "[verify] run_id=${RUN_ID}"
echo "[verify] integrity_summary=${SUMMARY_CSV}"
