#!/usr/bin/env bash
# 筛选与打分入口脚本包装器。
# 用法示例：
#   cd myanalyser && source .venv312/bin/activate
#   bash tools/run_filter_and_score.sh -i result_example/fund_scoreboard_xxx.csv -w artifacts/filter_score_run -f src/filter_score/filters/most_stable.py -s src/filter_score/scores/low_risk_debt.py

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

cd "${PROJECT_ROOT}"

# 确保 myanalyser 可导入
export PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH:-}"

exec python -m myanalyser.src.filter_score.filter_and_score_main "$@"
