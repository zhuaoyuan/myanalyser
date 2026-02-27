# myanalyser 项目说明

本项目用于面向自用需求的基金数据采集、复权净值计算、收益率一致性校验，以及评分榜单生成/入库。

## 目录结构（重构后）

```text
myanalyser/
  src/        # 业务代码（CLI + 核心逻辑）
  tests/      # 单测/集成测试
  data/
    common/   # 公共数据（如交易日历）
    versions/ # 按 run_id 存放每次跑数版本
    samples/  # 小样本数据
  artifacts/  # 预留目录（运行产物）
  docs/       # 文档
  tools/      # 一次性脚本
```

## 数据目录约定

- `run_id` 默认格式：`YYYYMMDD_HHMMSS`
- 可追加描述后缀：`YYYYMMDD_HHMMSS_desc`
- 每次跑数建议独立目录：`data/versions/{run_id}/`
- `fund_etl` 结果目录：`data/versions/{run_id}/fund_etl`
- 错误日志目录（与结果分离）：`data/versions/{run_id}/logs`
- 公共交易日历：`data/common/trade_dates.csv`

## 核心脚本

- `src/fund_etl.py`：AkShare 拉数（step1~step7 + retry）
- `src/adjusted_nav_tool.py`：复权净值计算
- `src/compare_adjusted_nav_and_cum_return.py`：复权收益率一致性比对
- `src/check_trade_day_data_integrity.py`：交易日完整性检查
- `src/pipeline_scoreboard.py`：评分榜单计算、导出与入库（支持 `--skip-sinks`）
- `src/backtest_portfolio.py`：按规则回测组合
- `src/verify_scoreboard_recalc.py`：榜单指标独立重算核验（从 fund_etl 中间数据重算并与导出榜单比对）

## 常用命令

```bash
# 1) 拉取原始数据（自动生成 run_id）
python src/fund_etl.py --mode all

# 2) 拉取原始数据（指定 run_id + 后缀）
python src/fund_etl.py --mode all --run-id 20260226_210000_test
# 或
python src/fund_etl.py --mode all --run-id-suffix smoke
```

```bash
# 3) 计算复权净值（示例 run_id）
RUN_ID=20260226_210000_smoke
python src/adjusted_nav_tool.py \
  --nav-dir data/versions/${RUN_ID}/fund_etl/fund_nav_by_code \
  --bonus-dir data/versions/${RUN_ID}/fund_etl/fund_bonus_by_code \
  --split-dir data/versions/${RUN_ID}/fund_etl/fund_split_by_code \
  --output-dir data/versions/${RUN_ID}/fund_etl/fund_adjusted_nav_by_code \
  --fail-log data/versions/${RUN_ID}/logs/failed_adjusted_nav.jsonl
```

```bash
# 4) 比对复权收益率与累计收益率
RUN_ID=20260226_210000_smoke
python src/compare_adjusted_nav_and_cum_return.py \
  --base-dir data/versions/${RUN_ID}/fund_etl \
  --output-dir data/versions/${RUN_ID}/fund_etl/fund_return_compare \
  --error-log data/versions/${RUN_ID}/logs/compare_adjusted_nav_cum_return_errors.jsonl
```

```bash
# 5) 交易日完整性检查
RUN_ID=20260226_210000_smoke
python src/check_trade_day_data_integrity.py \
  --base-dir data/versions/${RUN_ID}/fund_etl \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --trade-dates-csv data/common/trade_dates.csv
```

```bash
# 6) 榜单计算（仅导出 CSV，不入库）
RUN_ID=20260226_210000_smoke
python src/pipeline_scoreboard.py \
  --purchase-csv data/versions/${RUN_ID}/fund_etl/fund_purchase.csv \
  --overview-csv data/versions/${RUN_ID}/fund_etl/fund_overview.csv \
  --personnel-dir data/versions/${RUN_ID}/fund_etl/fund_personnel_by_code \
  --nav-dir data/versions/${RUN_ID}/fund_etl/fund_adjusted_nav_by_code \
  --output-dir artifacts/scoreboard_${RUN_ID} \
  --data-version ${RUN_ID} \
  --as-of-date 2026-02-26 \
  --skip-sinks
```

```bash
# 7) 榜单指标独立重算核验（验证 pipeline_scoreboard 计算正确性）
RUN_ID=20260226_210000_smoke
python src/verify_scoreboard_recalc.py \
  --scoreboard-csv artifacts/scoreboard_${RUN_ID}/scoreboard.csv \
  --fund-etl-dir data/versions/${RUN_ID}/fund_etl \
  --output-dir artifacts/scoreboard_${RUN_ID}/scoreboard_recheck
```

核验脚本从 `fund_adjusted_nav_by_code` 重算年化收益、夏普比率、最大回撤等指标及排名，与导出榜单逐项比对。产物：`summary.csv`（每只基金是否全部通过）、`details/{基金代码}.csv`（逐项明细）、`metrics_recalc_sample.csv`。默认 `--max-input-rows 200`，超过会报错（重算需全量输入，不支持抽样）。

## 依赖

```bash
pip install akshare pandas numpy pymysql
```

数据库联调需本地可用 Docker（用于 `fund_db_infra` 的 MySQL + ClickHouse）。

## 两类全流程脚本

### 1) 验收跑（`tools/verify.sh`）

用于代码和流程回归验收，重点是“快而全地证明链路可用”。

- 单测回归（`tests/test_*.py`）
- 核心 CLI smoke（`fund_etl`、`pipeline_scoreboard`、`backtest_portfolio`、`compare_adjusted_nav_and_cum_return`、`check_trade_day_data_integrity`）
- 启动 `fund_db_infra`（MySQL + ClickHouse）
- ETL 抽样数据链路（step1~step7，抽样 21 只：前 20 + `163402`）
- 复权净值计算、交易日完整性检查、复权收益率一致性比对
- Step 9.5 基金过滤（过滤结果会用于后续评分与回测）
- 评分榜单入库与导出、回测报告生成
- Step 11 独立重算核验（`verify_scoreboard_recalc.py`），要求 `scoreboard_recheck/summary.csv` 全部通过

```bash
cd /Users/zhuaoyuan/cursor-workspace/finance/myanalyser
source /Users/zhuaoyuan/cursor-workspace/finance/myanalyser/.venv312/bin/activate
bash tools/verify.sh
```

如需固定本次验收目录名，可显式传入 `RUN_ID`：

```bash
RUN_ID=20260226_220000_verify bash tools/verify.sh
```

如需固定数据库版本号，可同时指定 `DATA_VERSION`：

```bash
RUN_ID=20260226_220000_verify DATA_VERSION=20260226_verify_db bash tools/verify.sh
```

主要产物位置：

- `data/versions/{RUN_ID}/fund_etl`
- `data/versions/{RUN_ID}/logs`
- `artifacts/verify_{RUN_ID}/scoreboard`
- `artifacts/verify_{RUN_ID}/backtest`
- `artifacts/verify_{RUN_ID}/scoreboard_recheck`

### 2) 正式跑（`tools/run_full_pipeline.sh`）

用于真实生产/正式跑数，重点是“全量数据 + 全链路落库 + 可配置时间窗口”。

- 不跑单测和 CLI smoke，不做验收抽样（验收跑为 21 只），直接全量 ETL（`fund_etl --mode all`）
- 包含交易日完整性检查、复权收益率一致性比对、Step 9.5 过滤
- 评分流程直接消费过滤后的 `fund_purchase_for_step10_filtered.csv`
- 支持同 `RUN_ID` 断点续跑：各步骤成功后写 checkpoint，再次运行时优先复用已完成产物
- 不包含 backtest（回测部分建议独立执行）

```bash
cd /Users/zhuaoyuan/cursor-workspace/finance/myanalyser
source /Users/zhuaoyuan/cursor-workspace/finance/myanalyser/.venv312/bin/activate
bash tools/run_full_pipeline.sh
```

常用参数通过环境变量传入：

```bash
RUN_ID=20260226_230000_formal \
DATA_VERSION=20260226_formal_db \
INTEGRITY_START_DATE=2015-01-01 \
INTEGRITY_END_DATE=2025-12-31 \
FILTER_START_DATE=2023-01-01 \
FILTER_MAX_ABS_DEVIATION=0.02 \
bash tools/run_full_pipeline.sh
```

主要产物位置：

- `data/versions/{RUN_ID}/fund_etl`
- `data/versions/{RUN_ID}/logs`
- `artifacts/full_run_{RUN_ID}/filtered_fund_candidates.csv`
- `artifacts/full_run_{RUN_ID}/scoreboard`
- `artifacts/full_run_{RUN_ID}/.checkpoints`（步骤完成标记，用于断点续跑）