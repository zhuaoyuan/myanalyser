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
- 基金黑名单：`data/common/fund_blacklist.csv`（可选，格式含 `基金代码` 列；在 **prep_data_workflow** 阶段剔除，主流水线不再使用）

## 核心脚本

- `src/fund_etl.py`：AkShare 拉数（step1~step7 + retry）
- `tools/fund_bonus_split_vote.py`：分红/拆分数据多次爬取投票校对（供 fund_etl step5 后自动调用）
- `src/adjusted_nav_tool.py`：复权净值计算
- `src/compare_adjusted_nav_and_cum_return.py`：复权收益率一致性比对
- `src/check_trade_day_data_integrity.py`：交易日完整性检查
- `src/pipeline_scoreboard.py`：评分榜单计算、导出与入库（支持 `--formal-only`、`--skip-sinks`、`--latest-nav-date`）
- `src/scoreboard_metrics.py`：评分榜指标计算共享模块（供 pipeline 与 verify 共用；与 backtest 共用 `fund_metrics_core` 保证口径一致）
- `src/fund_metrics_core.py`：基金指标计算核心逻辑（Backtest 与 Scoreboard 共用，A 股口径：243 交易日/年、20 交易日/月）
- `src/fund_gmbd.py`：基金规模变动数据抓取（东方财富 FundArchivesDatas API，akshare 风格 `fund_gmbd_em(code)`）；CLI `tools/prep/fetch_fund_gmbd.py`
- `src/fund_cyrjg.py`：基金持有人结构数据抓取（东方财富 FundArchivesDatas API，akshare 风格 `fund_cyrjg_em(code)`）；CLI `tools/prep/fetch_fund_cyrjg.py`
- `tools/prep/prep_data_workflow.py`：**预备数据工作流**，从指定日期起拉取并筛选基金预备数据（购买 x → 持有人比例 a、规模 b、费率 c → 基金分类 c.1 → 详情 e → 多条件筛选 d.1）；支持已有 CSV 增量复用
- `src/verify_scoreboard_recalc.py`：榜单指标独立重算核验（从 fund_etl 中间数据重算并与导出榜单比对，支持 `--latest-nav-date`）
- `src/contracts/pipeline_contracts.py`：关键中间产物契约（列名/类型/非空/唯一键、目录 CSV 文件数量）
- `src/validators/validate_pipeline_artifacts.py`：按 stage 执行契约校验（失败返回非 0）
- `src/transforms/build_effective_purchase_csv.py`：从 `fund_purchase` 剔除黑名单生成 `fund_purchase_effective.csv`（供 prep 或独立调用；主流水线已不依赖，黑名单在 prep 阶段处理）
- `src/transforms/build_filtered_purchase_csv.py`：根据过滤结果生成 `fund_purchase_for_step10_filtered.csv`
- `src/compute_fund_composite_score.py`：基金综合得分计算（对 filtered/scoreboard CSV 做归一化+分组加权，输出带得分的 CSV）
- `src/filter_score/`：筛选与打分模块（入口 `filter_and_score_main.py`，可扩展过滤与算分策略，内置样例：最稳健原则过滤、低风险偏债得分）；最稳健规则逻辑位于共享模块 `most_stable_logic`
- `src/backtest/`：PyBroker 回测框架（数据加载、指标计算、策略包、**多过滤器链**）；CLI `tools/pybroker_fund_backtest.py`。策略包：`low_risk_debt`、`low_risk_debt_most_stable`（最稳健原则）、`steady_debt`（稳健型，依据 docs/参考/分类型的硬约束和主次目标.md）

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

策略包 `low_risk_debt_most_stable` 在筛选阶段应用最稳健原则（基于目标日前净值动态计算指标，无需预计算 CSV）。策略包 `steady_debt` 为稳健型（低波动偏债）：硬约束最大回撤≥-8%、年化≥5%、夏普≥0.5，主目标卡玛比率。
```bash
python myanalyser/tools/pybroker_fund_backtest.py --strategy low_risk_debt_most_stable --nav-dir ... ...
python myanalyser/tools/pybroker_fund_backtest.py --strategy steady_debt --nav-dir ... ...
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

### 多 T 调仓链式模拟（chain_multi_t_backtest）

基于 `multi_t_backtest` 产物，用前 T 期末市值作为后 T 期初市值，串联长期效果。前提：前 T 期末日期 ≤ 后 T 期初日期；无买入的 T 跳过。输出与单 T 相同格式。

```bash
python myanalyser/tools/v2/chain_multi_t_backtest.py \
  --output-root myanalyser/artifacts/backtest_multi/RUN_ID/RULESET_VERSION \
  [--chain-output-dir chain]
```

产物写入 `{output-root}/chain/`（默认），含 summary.csv、equity_curve.csv、period_detail.csv、orders.csv、positions_flat.csv、backtest_report.md。

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
# 或使用 --skip-sinks 保留 nav/period 构建但跳过 DB 写入（v2 verify 使用此模式）

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

核验脚本从 `fund_adjusted_nav_by_code` 重算年化收益、夏普比率、最大回撤等指标及排名，与导出榜单逐项比对。产物：`summary.csv`（每只基金是否全部通过）、`details/{基金代码}.csv`（逐项明细）、`metrics_recalc_sample.csv`。默认 `--max-input-rows 200`，超过会报错（重算需全量输入，不支持抽样）。v2 verify step10 会调用此脚本。

```bash
# 8) 由过滤结果生成过滤后 purchase（供 step10 消费）
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

## V2 全流程脚本

### 1) 验收跑（`tools/v2/verify.sh`）

用于代码和流程回归验收，**不依赖** fund_infra（Docker/MySQL/ClickHouse）。

- 单测回归（`tests/test_*.py`）
- 核心 CLI smoke（fund_etl、pipeline、backtest、compare、integrity、filter_funds_for_next_step、run_filter_and_score）
- V2 基线回归（step5~10 与 expected 对比）
- fund_etl step1 + 抽样 21 只
- fund_etl step2~step7
- 复权净值、交易日完整性、复权 vs 累计收益比对
- v2 过滤（filter_funds_for_next_step）+ filtered purchase
- 评分榜（`--skip-sinks`，仅 CSV）
- 筛选打分（step10 使用 `non_a_unlimited_purchase`）+ 重算核验

```bash
cd /Users/zhuaoyuan/cursor-workspace/finance/myanalyser
source .venv312/bin/activate
bash tools/v2/verify.sh
```

```bash
RUN_ID=20260319_120000_verify_v2 bash tools/v2/verify.sh
```

### 2) 正式跑（`tools/v2/run_full_pipeline.sh`）

全量 ETL + integrity/compare/filter/scoreboard。详见 `docs/V2完整流程说明.md`。

```bash
cd myanalyser && source .venv312/bin/activate
bash tools/v2/run_full_pipeline.sh @myanalyser/tmp/prep_work_v2/fund_purchase.csv
```

## 最小回归基线

- **mini_case_v2**：`tests/baseline/mini_case_v2/`，小份固定输入跑 step5~10，逐环节与 `expected/default` 对比
  - 输入：fund_etl（nav、bonus、split、overview、purchase、personnel、cum_return）
  - 回归用例：`tests/test_v2_baseline_regression.py`，由 `v2/verify.sh` step2b 调用
  - 生成 expected：`python tools/v2/generate_baseline_expected.py`（修改流程后需重跑以更新基线）
  - filter 区分：`v2/verify.sh` step10 使用 `non_a_unlimited_purchase`（验收用，确保 scored_result 非空）；baseline 回归与 generate 使用 `most_stable`
- 期望目录切换：`MYANALYSER_BASELINE_EXPECTED_DIR=/abs/path/to/expected`
