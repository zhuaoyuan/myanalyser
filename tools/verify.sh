#!/usr/bin/env bash

# `myanalyser/tools/verify.sh` 当前是一个**端到端验收脚本**，目标是把“数据抓取→清洗校验→入库→策略回测”整条链路跑一遍，并在关键节点做可执行断言。下面按流程拆开。

# ## 整体定位

# - 这是一个 10 步串行流水线，失败即退出（`set -euo pipefail`）。
# - 核心产物分三类：
#   - 数据版本目录：`data/versions/${RUN_ID}`
#   - 验收产物目录：`artifacts/verify_${RUN_ID}`
#   - 数据库落库结果（MySQL + ClickHouse）
# - 设计思路是：
#   - 先验证代码可用（单测 + CLI 冒烟）
#   - 再验证外部依赖可用（Docker DB）
#   - 再验证数据可用（ETL + 各种 CSV 断言）
#   - 最后验证业务可用（评分榜入库 + 回测输出）

# ---

# ## 脚本启动与公共准备

# ### 1) 路径和环境约定

# - `PROJECT_ROOT` 指向 `myanalyser`，`WORKSPACE_ROOT` 指向上层 `finance`。
# - 数据库基础设施固定用 `fund_db_infra/docker-compose.yml`。
# - 若未激活虚拟环境，只给 warning，不强制失败（可跑，但存在依赖版本漂移风险）。

# ### 2) Python 和运行参数

# - 自动选 `python` 或 `python3`，都没有就失败退出。
# - 默认变量：
#   - `RUN_ID`: `YYYYMMDD_HHMMSS_verify_e2e`
#   - `DATA_VERSION`: `${RUN_ID}_db`
# - 关键目录：
#   - `FUND_ETL_DIR`: `data/versions/${RUN_ID}/fund_etl`
#   - `LOGS_DIR`: `data/versions/${RUN_ID}/logs`
#   - `ARTIFACTS_DIR`: `artifacts/verify_${RUN_ID}`

# ### 3) 内置断言函数

# - `assert_file_exists` / `assert_dir_exists`
# - `assert_csv_has_rows`: 用 pandas 读 CSV，要求非空
# - `assert_dir_has_csv`: 目录至少有一个 CSV
# - `wait_mysql_ready` / `wait_clickhouse_ready`: 最长等待 120s

# ---

# ## 10 步执行流程（业务视角）

# ### Step 1/11：单元测试

# - 执行 `tests/test_*.py` 全量单测。
# - 业务意义：先守住函数级/模块级正确性，避免后续长链路浪费时间。

# ### Step 2/11：核心 CLI 冒烟（5 个）

# - 指定执行 5 个关键 CLI 集成测试（fund_etl/pipeline/backtest/compare/integrity）。
# - 业务意义：确认“最关键入口命令”在当前代码状态能启动并跑通基础流程。

# ### Step 3/11：启动数据库基础设施

# - 启动 `fund_db_infra` 的 docker compose。
# - 等待 MySQL / ClickHouse readiness。
# - 业务意义：后续 scoreboard 入库和回测选基依赖 DB，可提前暴露环境问题。

# ### Step 4/11：fund_etl 接口探测 + step1 全量清单

# - 执行 `fund_etl.py --mode verify`（接口列结构检查报告）
# - 执行 `fund_etl.py --mode step1`（抓基金购买清单）
# - 断言 `fund_purchase.csv` 存在且有行。
# - 业务意义：确认上游数据接口可用、字段没明显漂移。

# ### Step 5/10：抽样 101 只基金（Top100 + 163402）

# - 从 step1 输出中抽样：
#   - 先取不含 `163402` 的前 100
#   - 再补 1 条 `163402`（若不存在则构造空白占位）
# - 覆盖回写 `fund_purchase.csv`，强制后续 ETL 只跑样本集。
# - 业务意义：控制验收时长，同时保留一个指定目标基金用于回归对比。

# ### Step 6/10：对样本跑 step2~step7

# - 依次执行 ETL 各明细步骤：overview/nav/bonus/split/personnel/cum_return。
# - 对产物做存在性断言（overview 非空，分目录有 CSV）。
# - 业务意义：验证各业务主题数据都能拉取并落盘。

# ### Step 7/10：计算复权净值

# - `adjusted_nav_tool.py` 基于 nav + 分红 + 拆分计算复权净值。
# - 使用 `--allow-missing-event-until 2020-12-31`，并记录失败日志。
# - 断言复权目录有 CSV。
# - 业务意义：把原始净值转成可用于收益比较/回测的统一口径序列。

# ### Step 8/10：交易日完整性检查（2025 全年）

# - 对 ETL 数据做交易日覆盖检查，输出完整性报告目录。
# - 自动抓第一份 summary CSV 并断言非空。
# - 业务意义：检查时序数据是否缺日/断档，避免策略结果失真。

# ### Step 9/10：复权净值 vs 累计收益率 对比

# - 执行一致性比较，输出 `summary.csv` + `details/`。
# - 断言 summary 非空、details 目录存在。
# - 业务意义：交叉校验两条独立来源/口径的收益信息是否相互印证。

# ### Step 9.5/10：按 Step 9 结果过滤基金清单
#
# - 执行 `filter_funds_for_next_step.py`，综合 overview/nav/adjusted_nav、收益比对明细、交易日完整性明细打标过滤。
# - 产出过滤结果 CSV，并生成仅保留“不过滤”基金的 `fund_purchase_for_step10_filtered.csv`。
# - 业务意义：把质量约束前置到入库和回测前，避免明显异常基金进入后续策略链路。
#
# ### Step 10/10：评分榜入库 + 回测（消费过滤后清单）

# - 先扫描 `fund_adjusted_nav_by_code` 自动计算 `AS_OF_DATE`（最大净值日期）。
# - 运行 `pipeline_scoreboard.py`：
#   - 生成评分榜 CSV
#   - 连接 MySQL/ClickHouse
#   - `--apply-ddl` 自动建表
#   - 将当前 `DATA_VERSION` 数据入库
# - 再运行 `backtest_portfolio.py`：
#   - 2025 年区间
#   - 固定规则 `verify_e2e_top5`
#   - 从 ClickHouse 取选基和净值
# - 断言回测明细与报告文件存在。
# - 业务意义：验证“数据生产→入库→策略消费→结果输出”闭环可用。

# ---

# ## 关键输入/输出关系（便于审计）

# - 输入依赖：
#   - AkShare 接口（fund_etl 各步骤）
#   - Docker + MySQL + ClickHouse
#   - `fund_db_infra/sql/*.sql`
#   - `data/common/trade_dates.csv`
# - 关键输出：
#   - `fund_etl/fund_purchase.csv`, `fund_overview.csv`, 各 `*_by_code/*.csv`
#   - `fund_adjusted_nav_by_code/*.csv`
#   - `artifacts/verify_${RUN_ID}/trade_day_integrity_reports/*`
#   - `artifacts/verify_${RUN_ID}/fund_return_compare/*`
#   - `artifacts/verify_${RUN_ID}/scoreboard/fund_scoreboard_${DATA_VERSION}.csv`
#   - `artifacts/verify_${RUN_ID}/backtest/backtest_report.md`

# ---

# ## 从“业务合理性”看当前脚本的优点

# - 覆盖面完整：接口、数据、质量、入库、消费全链路都覆盖。
# - 有明确断言：不是只跑命令，而是对关键产物做非空/存在检查。
# - 有可追溯版本：`RUN_ID` + `DATA_VERSION` 将验收数据和业务版本绑定。
# - 有交叉验证：`adjusted_nav` 与 `cum_return` 的一致性检查是很实用的业务约束。

# ---


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
FILTER_START_DATE="${FILTER_START_DATE:-2023-01-01}"
FILTER_MAX_ABS_DEVIATION="${FILTER_MAX_ABS_DEVIATION:-0.02}"
VERIFY_SCOREBOARD_CH_WRITE_PROFILE="${VERIFY_SCOREBOARD_CH_WRITE_PROFILE:-fast}"
FILTER_RESULT_CSV="${ARTIFACTS_DIR}/filtered_fund_candidates.csv"
FILTERED_PURCHASE_CSV="${FUND_ETL_DIR}/fund_purchase_for_step10_filtered.csv"
FUND_PURCHASE_EFFECTIVE_CSV="${FUND_ETL_DIR}/fund_purchase_effective.csv"
FUND_BLACKLIST_PATH="${FUND_BLACKLIST_PATH:-${PROJECT_ROOT}/data/common/fund_blacklist.csv}"
RUN_REPORT_STEPS_CSV="${ARTIFACTS_DIR}/run_report_steps.csv"
RUN_REPORT_SUMMARY_CSV="${ARTIFACTS_DIR}/run_report_summary.csv"
RUN_REPORT_MD="${ARTIFACTS_DIR}/run_report.md"
CURRENT_STEP=""
STEP_START_TS=0

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

echo "[verify] project_root=${PROJECT_ROOT}"
echo "[verify] run_id=${RUN_ID}"
echo "[verify] data_version=${DATA_VERSION}"

mkdir -p "${FUND_ETL_DIR}" "${LOGS_DIR}" "${ARTIFACTS_DIR}" "${SCOREBOARD_DIR}" "${BACKTEST_DIR}"
printf 'step,status,duration_seconds\n' >"${RUN_REPORT_STEPS_CSV}"

start_step "step1_unit_tests"
"${PYTHON_BIN}" -m unittest discover -s tests -p "test_*.py" -v
finish_step "success"

start_step "step2_core_cli_smoke"
"${PYTHON_BIN}" -m unittest -v \
  tests.test_cli_integration.CoreCliIntegrationTest.test_fund_etl_cli_run_id_layout \
  tests.test_cli_integration.CoreCliIntegrationTest.test_pipeline_cli_smoke_skip_sinks_with_run_id_layout \
  tests.test_cli_integration.CoreCliIntegrationTest.test_backtest_cli_smoke_with_run_id_layout \
  tests.test_cli_integration.CoreCliIntegrationTest.test_compare_cli_with_run_id_layout \
  tests.test_cli_integration.CoreCliIntegrationTest.test_check_trade_day_integrity_cli_with_run_id_layout
finish_step "success"

start_step "step3_start_db"
assert_file_exists "${WORKSPACE_ROOT}/fund_db_infra/docker-compose.yml"
docker_compose_cmd -f "${WORKSPACE_ROOT}/fund_db_infra/docker-compose.yml" up -d
wait_mysql_ready
wait_clickhouse_ready
finish_step "success"

start_step "step4_fund_etl_verify_step1"
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode verify
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step1
assert_file_exists "${FUND_ETL_DIR}/fund_purchase.csv"
assert_csv_has_rows "${FUND_ETL_DIR}/fund_purchase.csv"
finish_step "success"

start_step "step5_sampling_purchase"
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

start_step "step5b_build_effective_purchase"
"${PYTHON_BIN}" src/transforms/build_effective_purchase_csv.py \
  --purchase-csv "${FUND_ETL_DIR}/fund_purchase.csv" \
  --blacklist-csv "${FUND_BLACKLIST_PATH}" \
  --output-csv "${FUND_PURCHASE_EFFECTIVE_CSV}"
assert_csv_has_rows "${FUND_PURCHASE_EFFECTIVE_CSV}"
finish_step "success"

start_step "step6_fund_etl_step2_to_step7"
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step2 --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}" --max-workers 8
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step3 --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}"
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step4 --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}"
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step5 --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}"
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step6 --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}"
"${PYTHON_BIN}" src/fund_etl.py --run-id "${RUN_ID}" --mode step7 --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}"
assert_csv_has_rows "${FUND_ETL_DIR}/fund_overview.csv"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_nav_by_code"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_bonus_by_code"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_split_by_code"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_personnel_by_code"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_cum_return_by_code"
finish_step "success"

start_step "step7_adjusted_nav"
"${PYTHON_BIN}" src/adjusted_nav_tool.py \
  --nav-dir "${FUND_ETL_DIR}/fund_nav_by_code" \
  --bonus-dir "${FUND_ETL_DIR}/fund_bonus_by_code" \
  --split-dir "${FUND_ETL_DIR}/fund_split_by_code" \
  --output-dir "${FUND_ETL_DIR}/fund_adjusted_nav_by_code" \
  --allow-missing-event-until 2020-12-31 \
  --fail-log "${LOGS_DIR}/failed_adjusted_nav.jsonl"
assert_dir_has_csv "${FUND_ETL_DIR}/fund_adjusted_nav_by_code"
finish_step "success"

start_step "step8_integrity"
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
finish_step "success"

start_step "step9_compare_returns"
"${PYTHON_BIN}" src/compare_adjusted_nav_and_cum_return.py \
  --base-dir "${FUND_ETL_DIR}" \
  --output-dir "${ARTIFACTS_DIR}/fund_return_compare" \
  --error-log "${LOGS_DIR}/compare_adjusted_nav_cum_return_errors.jsonl"
assert_csv_has_rows "${ARTIFACTS_DIR}/fund_return_compare/summary.csv"
assert_dir_exists "${ARTIFACTS_DIR}/fund_return_compare/details"
finish_step "success"

start_step "step9_5_filter_and_filtered_purchase"
"${PYTHON_BIN}" src/filter_funds_for_next_step.py \
  --base-dir "${FUND_ETL_DIR}" \
  --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}" \
  --compare-details-dir "${ARTIFACTS_DIR}/fund_return_compare/details" \
  --integrity-details-dir "${ARTIFACTS_DIR}/trade_day_integrity_reports/details_2025-01-01_2025-12-31" \
  --start-date "${FILTER_START_DATE}" \
  --max-abs-deviation "${FILTER_MAX_ABS_DEVIATION}" \
  --output-csv "${FILTER_RESULT_CSV}"
assert_csv_has_rows "${FILTER_RESULT_CSV}"

"${PYTHON_BIN}" src/transforms/build_filtered_purchase_csv.py \
  --purchase-csv "${FUND_PURCHASE_EFFECTIVE_CSV}" \
  --filter-csv "${FILTER_RESULT_CSV}" \
  --output-csv "${FILTERED_PURCHASE_CSV}"
assert_csv_has_rows "${FILTERED_PURCHASE_CSV}"
finish_step "success"

start_step "step10_scoreboard_and_backtest"
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

STEP10_SCOREBOARD_START_TS="$(date +%s)"
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
  --apply-ddl \
  --mysql-ddl "${MYSQL_DDL}" \
  --clickhouse-ddl "${CLICKHOUSE_DDL}" \
  --mysql-host 127.0.0.1 \
  --mysql-port 3306 \
  --mysql-user root \
  --mysql-password your_strong_password \
  --mysql-db fund_analysis \
  --clickhouse-db fund_analysis \
  --clickhouse-container fund_clickhouse \
  --clickhouse-write-profile "${VERIFY_SCOREBOARD_CH_WRITE_PROFILE}" \
  --clickhouse-write-scope verify_minimal
STEP10_SCOREBOARD_END_TS="$(date +%s)"
echo "[verify] step10 scoreboard_seconds=$((STEP10_SCOREBOARD_END_TS - STEP10_SCOREBOARD_START_TS))"

assert_csv_has_rows "${SCOREBOARD_DIR}/fund_scoreboard_${DATA_VERSION}.csv"

STEP10_BACKTEST_START_TS="$(date +%s)"
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
STEP10_BACKTEST_END_TS="$(date +%s)"
echo "[verify] step10 backtest_seconds=$((STEP10_BACKTEST_END_TS - STEP10_BACKTEST_START_TS))"

assert_csv_has_rows "${BACKTEST_DIR}/backtest_window_detail.csv"
assert_file_exists "${BACKTEST_DIR}/backtest_report.md"
finish_step "success"

start_step "step11_scoreboard_recalc_verify"
"${PYTHON_BIN}" src/verify_scoreboard_recalc.py \
  --scoreboard-csv "${SCOREBOARD_DIR}/fund_scoreboard_${DATA_VERSION}.csv" \
  --fund-etl-dir "${FUND_ETL_DIR}" \
  --output-dir "${ARTIFACTS_DIR}/scoreboard_recheck" \
  --max-input-rows 200
assert_csv_has_rows "${ARTIFACTS_DIR}/scoreboard_recheck/summary.csv"

"${PYTHON_BIN}" - <<'PY' "${ARTIFACTS_DIR}/scoreboard_recheck/summary.csv"
from pathlib import Path
import sys

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
echo "[verify] backtest_report=${BACKTEST_DIR}/backtest_report.md"
echo "[verify] scoreboard_recheck_summary=${ARTIFACTS_DIR}/scoreboard_recheck/summary.csv"
echo "[verify] run_report=${RUN_REPORT_MD}"
