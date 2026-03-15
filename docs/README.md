# myanalyser 项目说明

本项目用于面向自用需求的基金数据采集、复权净值计算、收益率一致性校验，以及评分榜单生成/入库。

## 目录结构（重构后）

```text
myanalyser/
  src/        # 业务代码（CLI + 核心逻辑）
    contracts/   # 中间产物 schema 契约定义
    validators/  # 按 stage 执行契约校验
    transforms/  # 流程复用的通用转换脚本
  tests/      # 单测/集成测试
  data/
    common/   # 公共数据（如交易日历）
    versions/ # 按 run_id 存放每次跑数版本
    samples/  # 小样本数据
  artifacts/  # 预留目录（运行产物）
  docs/       # 文档
  tools/      # 工具脚本
    temp_use/  # 一次性脚本
```

## 数据目录约定

- `run_id` 默认格式：`YYYYMMDD_HHMMSS`
- 可追加描述后缀：`YYYYMMDD_HHMMSS_desc`
- 每次跑数建议独立目录：`data/versions/{run_id}/`
- `fund_etl` 结果目录：`data/versions/{run_id}/fund_etl`
- 错误日志目录（与结果分离）：`data/versions/{run_id}/logs`
- 分红/拆分校对结果：`data/versions/{run_id}/fund_etl/revised_fund_bonus_by_code`、`data/versions/{run_id}/fund_etl/revised_fund_split_by_code`
- 校对留档目录：`data/common/revise/{YYYYMMDD}_{run_id}_revised/fund_etl`
- 公共交易日历：`data/common/trade_dates.csv`
- 基金黑名单：`data/common/fund_blacklist.csv`（可选，格式含 `基金代码` 列；通过 `FUND_BLACKLIST_PATH` 覆盖路径）

## 核心脚本

- `src/fund_etl.py`：AkShare 拉数（step1~step7 + retry）
- `tools/fund_bonus_split_vote.py`：分红/拆分数据多次爬取投票校对（供 fund_etl step5 后自动调用）
- `src/adjusted_nav_tool.py`：复权净值计算
- `src/compare_adjusted_nav_and_cum_return.py`：复权收益率一致性比对
- `src/check_trade_day_data_integrity.py`：交易日完整性检查
- `src/pipeline_scoreboard.py`：评分榜单计算、导出与入库（支持 `--formal-only`、`--skip-sinks`、`--latest-nav-date`、`--clickhouse-write-profile`、`--clickhouse-write-scope`）
- `src/scoreboard_metrics.py`：评分榜指标计算共享模块（供 pipeline 与 verify 共用；与 backtest 共用 `fund_metrics_core` 保证口径一致）
- `src/fund_metrics_core.py`：基金指标计算核心逻辑（Backtest 与 Scoreboard 共用，A 股口径：243 交易日/年、20 交易日/月）
- `src/backtest_portfolio.py`：按规则回测组合
- `src/fund_gmbd.py`：基金规模变动数据抓取（东方财富 FundArchivesDatas API，akshare 风格 `fund_gmbd_em(code)`）；CLI `tools/prep/fetch_fund_gmbd.py`
- `src/fund_cyrjg.py`：基金持有人结构数据抓取（东方财富 FundArchivesDatas API，akshare 风格 `fund_cyrjg_em(code)`）；CLI `tools/prep/fetch_fund_cyrjg.py`
- `tools/prep/prep_data_workflow.py`：**预备数据工作流**，从指定日期起拉取并筛选基金预备数据（购买 x → 持有人比例 a、规模 b、费率 c → 基金分类 c.1 → 详情 e → 多条件筛选 d.1）；支持已有 CSV 增量复用
- `src/verify_scoreboard_recalc.py`：榜单指标独立重算核验（从 fund_etl 中间数据重算并与导出榜单比对，支持 `--latest-nav-date`）
- `src/contracts/pipeline_contracts.py`：关键中间产物契约（列名/类型/非空/唯一键、目录 CSV 文件数量）
- `src/validators/validate_pipeline_artifacts.py`：按 stage 执行契约校验（失败返回非 0）
- `src/transforms/build_effective_purchase_csv.py`：从 `fund_purchase` 剔除黑名单生成 `fund_purchase_effective.csv`（不修改原始 purchase）
- `src/transforms/build_filtered_purchase_csv.py`：根据过滤结果生成 `fund_purchase_for_step10_filtered.csv`
- `src/compute_fund_composite_score.py`：基金综合得分计算（对 filtered/scoreboard CSV 做归一化+分组加权，输出带得分的 CSV）
- `src/filter_score/`：筛选与打分模块（入口 `filter_and_score_main.py`，可扩展过滤与算分策略，内置样例：最稳健原则过滤、低风险偏债得分）
- `src/backtest/`：PyBroker 回测框架（数据加载、指标计算、策略包、**多过滤器链**）；CLI `tools/pybroker_fund_backtest.py`。策略包 `low_risk_debt_most_stable` 在筛选阶段应用最稳健原则（基于目标日前净值动态计算指标，与 `low_risk_debt` 类似）

### PyBroker 回测过滤器链

通过环境变量 `FUND_BACKTEST_FILTERS` 指定链式过滤器（逗号分隔），主流程不感知具体实现：

| 过滤器名 | 环境变量 | 说明 |
|----------|----------|------|
| `filtered_candidates` | `FILTERED_FUND_CANDIDATES_CSV` | 从 filtered_fund_candidates.csv 取 是否过滤=否 的基金编码 |
| `max_funds` | `FUND_BACKTEST_MAX_FUNDS` | 按数量截断（取前 N 个，按编码排序） |

示例：
```bash
export FUND_BACKTEST_FILTERS=filtered_candidates,max_funds
export FILTERED_FUND_CANDIDATES_CSV=finance-runs/run_xxx/artifacts/full_run_xxx/filtered_fund_candidates.csv
export FUND_BACKTEST_MAX_FUNDS=50
python myanalyser/tools/pybroker_fund_backtest.py --nav-dir finance-runs/run_xxx/data ...
```

策略包 `low_risk_debt_most_stable` 在筛选阶段应用最稳健原则（基于目标日前净值动态计算指标，无需预计算 CSV）：
```bash
python myanalyser/tools/pybroker_fund_backtest.py --strategy low_risk_debt_most_stable --nav-dir ... ...
```

### PyBroker 回测输出文件

| 文件 | 说明 |
|------|------|
| summary.csv | 汇总：运行参数、数据范围、fund_metrics_core 指标（年化、回撤、夏普等） |
| period_detail.csv | 每期调仓明细（含 period_return、换手、订单） |
| equity_curve.csv | 每日净值与累计收益率 |
| orders.csv | 独立订单明细 |
| positions_flat.csv | 扁平持仓（stat_date, symbol, weight, rank） |
| backtest_report.md | Markdown 报告（运行参数、Top 3 调仓期、输出文件索引） |
| backtest_curves.html | Plotly 收益曲线图（组合 + 成分基金对照，需 `plotly` 依赖） |

## 常用命令

```bash
# 1) 拉取原始数据（自动生成 run_id）
python src/fund_etl.py --mode all

# 2) 拉取原始数据（指定 run_id + 后缀）
python src/fund_etl.py --mode all --run-id 20260226_210000_test
# 或
python src/fund_etl.py --mode all --run-id-suffix smoke

# 2.1) 启用分红/拆分校对（step5 后自动投票产出 revised 目录）
python src/fund_etl.py --mode all \
  --bonus-split-revise-root data/common/revise/20260310_final
```

```bash
# 3) 计算复权净值（示例 run_id）
RUN_ID=20260226_210000_smoke
python src/adjusted_nav_tool.py \
  --nav-dir data/versions/${RUN_ID}/fund_etl/fund_nav_by_code \
  --bonus-dir data/versions/${RUN_ID}/fund_etl/revised_fund_bonus_by_code \
  --split-dir data/versions/${RUN_ID}/fund_etl/revised_fund_split_by_code \
  --output-dir data/versions/${RUN_ID}/fund_etl/fund_adjusted_nav_by_code \
  --fail-log data/versions/${RUN_ID}/logs/failed_adjusted_nav.jsonl
```

> 若未启用分红/拆分校对（或校对目录为空），可退回使用 `fund_bonus_by_code` / `fund_split_by_code`。

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
# 6) 榜单计算（正式计算模式：仅 Python 计算，不入库，显著提速）
RUN_ID=20260226_210000_smoke
python src/pipeline_scoreboard.py \
  --purchase-csv data/versions/${RUN_ID}/fund_etl/fund_purchase.csv \
  --overview-csv data/versions/${RUN_ID}/fund_etl/fund_overview.csv \
  --personnel-dir data/versions/${RUN_ID}/fund_etl/fund_personnel_by_code \
  --nav-dir data/versions/${RUN_ID}/fund_etl/fund_adjusted_nav_by_code \
  --output-dir artifacts/scoreboard_${RUN_ID} \
  --data-version ${RUN_ID} \
  --as-of-date 2026-02-26 \
  --formal-only
# 或使用 --skip-sinks 保留 nav/period 构建但跳过 DB 写入

# 历史截断模式（用于更公正的回测）：
# 仅使用 <= latest-nav-date 的净值与人事数据计算榜单。
# 设置 --latest-nav-date 时，--resume 会自动禁用，避免复用不匹配 checkpoint。
python src/pipeline_scoreboard.py \
  --purchase-csv data/versions/${RUN_ID}/fund_etl/fund_purchase.csv \
  --overview-csv data/versions/${RUN_ID}/fund_etl/fund_overview.csv \
  --personnel-dir data/versions/${RUN_ID}/fund_etl/fund_personnel_by_code \
  --nav-dir data/versions/${RUN_ID}/fund_etl/fund_adjusted_nav_by_code \
  --output-dir artifacts/scoreboard_${RUN_ID}_hist_20251231 \
  --data-version ${RUN_ID}_hist_20251231 \
  --as-of-date 2025-12-31 \
  --latest-nav-date 2025-12-31 \
  --formal-only

# 验收/联调场景可控制 ClickHouse 写入策略
# --clickhouse-write-profile: auto|safe|fast（默认 auto）
# --clickhouse-write-scope: full|verify_minimal（仅写 nav_daily + scoreboard）
```

```bash
# 7) 榜单指标独立重算核验（验证 pipeline_scoreboard 计算正确性）
RUN_ID=20260226_210000_smoke
python src/verify_scoreboard_recalc.py \
  --scoreboard-csv artifacts/scoreboard_${RUN_ID}/scoreboard.csv \
  --fund-etl-dir data/versions/${RUN_ID}/fund_etl \
  --output-dir artifacts/scoreboard_${RUN_ID}/scoreboard_recheck
# 若 scoreboard 是历史截断口径，需传入相同 latest-nav-date 以保持一致：
#   --latest-nav-date 2025-12-31
```

核验脚本从 `fund_adjusted_nav_by_code` 重算年化收益、夏普比率、最大回撤等指标及排名，与导出榜单逐项比对。产物：`summary.csv`（每只基金是否全部通过）、`details/{基金代码}.csv`（逐项明细）、`metrics_recalc_sample.csv`。默认 `--max-input-rows 200`，超过会报错（重算需全量输入，不支持抽样）。

```bash
# 8) 按 stage 执行中间产物契约校验（示例）
python src/validators/validate_pipeline_artifacts.py \
  --stage scoreboard_input \
  --artifact purchase_csv=data/versions/${RUN_ID}/fund_etl/fund_purchase_for_step10_filtered.csv \
  --artifact overview_csv=data/versions/${RUN_ID}/fund_etl/fund_overview.csv \
  --artifact personnel_dir=data/versions/${RUN_ID}/fund_etl/fund_personnel_by_code \
  --artifact nav_dir=data/versions/${RUN_ID}/fund_etl/fund_adjusted_nav_by_code
```

```bash
# 9) 由过滤结果生成过滤后 purchase（供 step10 消费）
python src/transforms/build_filtered_purchase_csv.py \
  --purchase-csv data/versions/${RUN_ID}/fund_etl/fund_purchase.csv \
  --filter-csv artifacts/verify_${RUN_ID}/filtered_fund_candidates.csv \
  --output-csv data/versions/${RUN_ID}/fund_etl/fund_purchase_for_step10_filtered.csv
```

```bash
# 10) 基金综合得分计算（对 filtered/scoreboard CSV 做归一化+分组加权）
python src/compute_fund_composite_score.py \
  -i result_example/0301_manual_filtered.csv \
  -o artifacts/composite_score_output.csv
```

```bash
# 10.1) 基金规模变动抓取（东方财富 FundArchivesDatas API，akshare 风格）
python tools/prep/fetch_fund_gmbd.py 000198 110011 -o artifacts/fund_gmbd.csv
# 或从 CSV 读取基金代码列：-i fund_purchase.csv -o output.csv [--delay 0.3]
```

```bash
# 11) 筛选与打分模块化流水线（可扩展过滤策略 + 算分策略）
# 输入：fund_scoreboard CSV、0~多个过滤脚本、1 个算分脚本、工作目录
# 产物：filter_result.csv（中间，含过滤结果）、scored_result.csv（最终）
bash tools/run_filter_and_score.sh \
  -i result_example/fund_scoreboard_20260301_1_formal_retry_step4_rerun_db.csv \
  -w artifacts/filter_score_run \
  -f src/filter_score/filters/most_stable.py \
  -s src/filter_score/scores/low_risk_debt.py
# 可多次 -f 指定多个过滤脚本
```

```bash
# 生成 scored_result 静态 HTML + ECharts 可视化
python tools/gen_scoreboard_html.py \
  -i artifacts/filter_score_run_3/scored_result.csv \
  -o artifacts/filter_score_run_3/scoreboard.html
# 可选 -f 指定 fund_etl 目录以显示净值走势图（需含 fund_adjusted_nav_by_code、fund_personnel_by_code）
python tools/gen_scoreboard_html.py -i scored_result.csv -o scoreboard.html \
  -f finance-runs/run_xxx/data/versions/xxx/fund_etl
# 输出单文件 HTML，包含：综合排名、雷达图、风险-收益散点（近1年）、明细表（全列筛选/排序/勾选显示）、净值走势图（可选）
```

## 依赖

### Python 版本

- 建议使用 **Python 3.12.x**，项目通过 `myanalyser/.python-version` 锁定为 `3.12.12`（pyenv 自动读取）。
- 虚拟环境路径：`myanalyser/.venv312`

### pip 依赖

**复现环境（推荐，精确锁定版本）：**

```bash
source myanalyser/.venv312/bin/activate
pip install -r myanalyser/requirements-lock.txt
```

**仅安装顶层依赖（允许传递依赖自动解析）：**

```bash
pip install -r myanalyser/requirements.txt
```

### Docker 基础设施

数据库联调需本地可用 Docker（用于 `fund_db_infra` 的 MySQL + ClickHouse）。镜像已锁定：`mysql:8.4`、`clickhouse/clickhouse-server:25.8`。

## 两类全流程脚本

### 1) 验收跑（`tools/verify.sh`）

用于代码和流程回归验收，重点是“快而全地证明链路可用”。

- 单测回归（`tests/test_*.py`）
- 核心 CLI smoke（`fund_etl`、`pipeline_scoreboard`、`backtest_portfolio`、`compare_adjusted_nav_and_cum_return`、`check_trade_day_data_integrity`）
- Step 10b 筛选与打分（消费 scoreboard，产出 `filter_result.csv`、`scored_result.csv`）
- 启动 `fund_db_infra`（MySQL + ClickHouse）
- ETL 抽样数据链路（step1~step7，抽样 21 只：前 20 + `163402`）
- 复权净值计算、交易日完整性检查、复权收益率一致性比对
- Step 9.5 基金过滤（过滤结果会用于后续评分与回测）
- 评分榜单入库与导出、回测报告生成
  - Step10 默认以快速模式写 ClickHouse：`--clickhouse-write-profile fast`
  - Step10 默认使用最小写入范围：`--clickhouse-write-scope verify_minimal`（仅 `fact_fund_nav_daily` + `fact_fund_scoreboard_snapshot`）
- Step 11 独立重算核验（`verify_scoreboard_recalc.py`），要求 `scoreboard_recheck/summary.csv` 全部通过
- 自动生成运行报告汇总（每步耗时、步骤成功率、异常分布、过滤前后数量变化）

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

如需覆盖 Step10 写库策略，可通过环境变量修改：

```bash
VERIFY_SCOREBOARD_CH_WRITE_PROFILE=safe bash tools/verify.sh
```

主要产物位置：

- `data/versions/{RUN_ID}/fund_etl`
- `data/versions/{RUN_ID}/logs`
- `artifacts/verify_{RUN_ID}/scoreboard`
- `artifacts/verify_{RUN_ID}/backtest`
- `artifacts/verify_{RUN_ID}/filter_score`（filter_result.csv、scored_result.csv）
- `artifacts/verify_{RUN_ID}/scoreboard_recheck`
- `artifacts/verify_{RUN_ID}/run_report_steps.csv`
- `artifacts/verify_{RUN_ID}/run_report_summary.csv`
- `artifacts/verify_{RUN_ID}/run_report.md`

Step10 还会在终端打印分段耗时：`scoreboard_seconds` 与 `backtest_seconds`，用于快速定位瓶颈。

### 2) 正式跑（`tools/run_full_pipeline.sh`）

用于真实生产/正式跑数，重点是“全量数据 + 正式计算 + 可配置时间窗口”。

- 不跑单测和 CLI smoke，不做验收抽样（验收跑为 21 只），直接全量 ETL（`fund_etl --mode all`）
- 包含交易日完整性检查、复权收益率一致性比对、Step 9.5 过滤
- 评分流程使用 `--formal-only`（纯 Python 计算，不写 DB），直接消费过滤后的 `fund_purchase_for_step10_filtered.csv`
- 支持同 `RUN_ID` 断点续跑：各步骤成功后写 checkpoint，再次运行时优先复用已完成产物
- 不包含 backtest、不包含核验（核验仅在 `verify.sh` 中执行）
- 自动生成运行报告汇总（每步耗时、步骤成功率、异常分布、过滤前后数量变化）

```bash
cd /Users/zhuaoyuan/cursor-workspace/finance/myanalyser
source /Users/zhuaoyuan/cursor-workspace/finance/myanalyser/.venv312/bin/activate
bash tools/run_full_pipeline.sh
```

长时任务推荐使用隔离启动脚本（会自动创建 `git worktree`、记录元数据并后台运行）：

```bash
cd /Users/zhuaoyuan/cursor-workspace/finance
bash myanalyser/tools/start_isolated_pipeline.sh \
  --venv /Users/zhuaoyuan/cursor-workspace/finance/myanalyser/.venv312
```

隔离启动脚本常用参数：

- `--venv /path/to/venv`：为后台进程注入 `VIRTUAL_ENV` 与 `PATH`。
- `--allow-dirty`：允许当前开发工作区有未提交改动（会记录到 `VERSION_INFO`，但运行代码仍固定在当前 `HEAD`）。
- `--target-dir /path/to/run_dir`：指定 worktree 运行目录。
- `@/path/to/fund_purchase.csv`：传入本次运行使用的 purchase 文件。

通过隔离脚本传递 `run_full_pipeline.sh` 环境变量（推荐命令前缀）：

```bash
ETL_MAX_WORKERS=16 \
FILTER_START_DATE=2023-01-01 \
DATA_VERSION=202602_custom \
bash myanalyser/tools/start_isolated_pipeline.sh \
  --venv /Users/zhuaoyuan/cursor-workspace/finance/myanalyser/.venv312
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
- `artifacts/full_run_{RUN_ID}/run_report_steps.csv`
- `artifacts/full_run_{RUN_ID}/run_report_summary.csv`
- `artifacts/full_run_{RUN_ID}/run_report.md`

隔离启动额外产物（位于隔离目录根，例如 `../finance-runs/run_YYYYMMDD_HHMMSS/`）：

- `VERSION_INFO`：运行时间、commit、branch、运行 PID、venv 等元信息
- `LAUNCH_INFO`：启动参数记录（如 `PIPELINE_ARG`、`VENV_DIR`）
- `pipeline.log`：后台运行日志

## 最小回归基线

- 基线目录：`tests/baseline/mini_case/`
- 输入数据来自历史 `data/versions` 与 `artifacts` 产物抽样，不使用 mock
- 期望输出目录默认：`tests/baseline/mini_case/expected/default`
- 可通过环境变量切换期望目录：`MYANALYSER_BASELINE_EXPECTED_DIR=/abs/path/to/expected`
- 关键回归用例：`tests/test_pipeline_regression_baseline.py`
