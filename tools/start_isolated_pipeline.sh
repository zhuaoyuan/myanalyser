#!/usr/bin/env bash

set -euo pipefail

# 用法速览（在仓库根目录执行）：
# 1) 默认启动（要求工作区干净）：
#    bash myanalyser/tools/start_isolated_pipeline.sh
# 2) 允许未提交改动（仅记录状态，不会进入冻结运行代码）：
#    bash myanalyser/tools/start_isolated_pipeline.sh --allow-dirty
# 3) 指定本次运行的 purchase csv：
#    bash myanalyser/tools/start_isolated_pipeline.sh @/abs/path/fund_purchase.csv
# 4) 指定运行目录：
#    bash myanalyser/tools/start_isolated_pipeline.sh --target-dir /path/to/run_dir
# 5) 指定共享虚拟环境（推荐）：
#    bash myanalyser/tools/start_isolated_pipeline.sh --venv /Users/zhuaoyuan/cursor-workspace/finance/myanalyser/.venv312

usage() {
  cat <<'EOF'
Usage:
  start_isolated_pipeline.sh [--target-dir DIR] [--allow-dirty] [--venv VENV_DIR] [@/path/to/fund_purchase.csv]

Behavior:
  1) Create a detached git worktree from current HEAD
  2) Record run metadata under the target worktree
  3) Start tools/run_full_pipeline.sh with nohup in background

Examples:
  bash myanalyser/tools/start_isolated_pipeline.sh
  bash myanalyser/tools/start_isolated_pipeline.sh --allow-dirty
  bash myanalyser/tools/start_isolated_pipeline.sh --venv /Users/zhuaoyuan/cursor-workspace/finance/myanalyser/.venv312
  bash myanalyser/tools/start_isolated_pipeline.sh @/abs/path/fund_purchase.csv
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
WORKSPACE_PARENT="$(cd "${REPO_ROOT}/.." && pwd)"

TARGET_DIR=""
ALLOW_DIRTY=0
PIPELINE_ARG=""
VENV_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-dir)
      if [[ $# -lt 2 ]]; then
        echo "[isolated-run] --target-dir requires a value"
        exit 1
      fi
      TARGET_DIR="$2"
      shift 2
      ;;
    --allow-dirty)
      ALLOW_DIRTY=1
      shift
      ;;
    --venv)
      if [[ $# -lt 2 ]]; then
        echo "[isolated-run] --venv requires a value"
        exit 1
      fi
      VENV_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    @*)
      if [[ -n "${PIPELINE_ARG}" ]]; then
        echo "[isolated-run] only one @purchase-csv argument is supported"
        exit 1
      fi
      CSV_PATH="${1#@}"
      if [[ "${CSV_PATH}" = /* ]]; then
        PIPELINE_ARG="@$CSV_PATH"
      else
        PIPELINE_ARG="@$(cd "${PWD}" && cd "$(dirname "${CSV_PATH}")" && pwd)/$(basename "${CSV_PATH}")"
      fi
      shift
      ;;
    *)
      echo "[isolated-run] unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

RUN_ID="$(date +%Y%m%d_%H%M%S)"
COMMIT_HASH="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
BRANCH_NAME="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)"
STATUS_SHORT="$(git -C "${REPO_ROOT}" status --short || true)"

if [[ -z "${TARGET_DIR}" ]]; then
  TARGET_DIR="${WORKSPACE_PARENT}/finance-runs/run_${RUN_ID}"
fi

if [[ "${ALLOW_DIRTY}" -eq 0 && -n "${STATUS_SHORT}" ]]; then
  echo "[isolated-run] working tree is dirty; commit/stash first, or pass --allow-dirty"
  exit 1
fi

if [[ -e "${TARGET_DIR}" ]]; then
  echo "[isolated-run] target already exists: ${TARGET_DIR}"
  exit 1
fi

if [[ -n "${VENV_DIR}" ]]; then
  if [[ "${VENV_DIR}" != /* ]]; then
    VENV_DIR="$(cd "${PWD}" && cd "${VENV_DIR}" && pwd)"
  fi
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[isolated-run] invalid venv: ${VENV_DIR} (missing ${VENV_DIR}/bin/python)"
    exit 1
  fi
fi

mkdir -p "$(dirname "${TARGET_DIR}")"
git -C "${REPO_ROOT}" worktree add --detach "${TARGET_DIR}" "${COMMIT_HASH}" >/dev/null

{
  echo "Run Date: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "Run ID: ${RUN_ID}"
  echo "Commit Hash: ${COMMIT_HASH}"
  echo "Branch: ${BRANCH_NAME}"
  echo "Repo Root: ${REPO_ROOT}"
  echo "Worktree Dir: ${TARGET_DIR}"
  echo "Launcher PID: $$"
  echo "Dirty Working Tree: $([[ -n "${STATUS_SHORT}" ]] && echo yes || echo no)"
  echo "Status:"
  if [[ -n "${STATUS_SHORT}" ]]; then
    printf '%s\n' "${STATUS_SHORT}"
  else
    echo "(clean)"
  fi
} > "${TARGET_DIR}/VERSION_INFO"

{
  echo "PIPELINE_ARG=${PIPELINE_ARG}"
  echo "VENV_DIR=${VENV_DIR}"
} > "${TARGET_DIR}/LAUNCH_INFO"

cd "${TARGET_DIR}"
LOG_FILE="${TARGET_DIR}/pipeline.log"
CMD=(bash tools/run_full_pipeline.sh)
if [[ -n "${PIPELINE_ARG}" ]]; then
  CMD+=("${PIPELINE_ARG}")
fi

if [[ -n "${VENV_DIR}" ]]; then
  nohup env "VIRTUAL_ENV=${VENV_DIR}" "PATH=${VENV_DIR}/bin:${PATH}" "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
else
  nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
fi
PIPELINE_PID=$!

{
  echo "Pipeline PID: ${PIPELINE_PID}"
  echo "Log File: ${LOG_FILE}"
  echo "Start Date: $(date '+%Y-%m-%d %H:%M:%S %z')"
  if [[ -n "${VENV_DIR}" ]]; then
    echo "Runtime Venv: ${VENV_DIR}"
  else
    echo "Runtime Venv: (inherit current PATH)"
  fi
} >> "${TARGET_DIR}/VERSION_INFO"

echo "[isolated-run] pipeline started"
echo "[isolated-run] target dir: ${TARGET_DIR}"
echo "[isolated-run] pid: ${PIPELINE_PID}"
echo "[isolated-run] log: ${LOG_FILE}"
echo "[isolated-run] follow log: tail -f '${LOG_FILE}'"
