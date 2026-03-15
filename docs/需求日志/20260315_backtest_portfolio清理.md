# 20260315：backtest_portfolio 清理（阶段 A）

## 需求简述

按 `docs/其他/旧回测代码引用梳理_清理准备.md` 执行阶段 A：仅替换 backtest_portfolio，filter_score 保留。

## 实现方案

采用 **A3 适配层** 方案：

1. 新建 `src/backtest_verify_e2e.py`：从 ClickHouse 取选基、使用本地净值目录（fund_adjusted_nav_by_code）、调用 PyBroker 引擎
2. 新建 `src/backtest/strategies/verify_e2e_top5.py`：固定选基策略包（每调仓日持有同一 top5）
3. `tools/backtest_verify_e2e.py` 为薄包装，调用 src 模块的 main()

## 变更清单

| 文件 | 操作 |
|------|------|
| `src/backtest_portfolio.py` | **删除** |
| `src/backtest_verify_e2e.py` | **新增** |
| `src/backtest/strategies/verify_e2e_top5.py` | **新增** |
| `tools/backtest_verify_e2e.py` | **新增**（薄包装） |
| `tools/verify.sh` | step10 切换至 backtest_verify_e2e，断言 period_detail.csv |
| `tests/test_backtest_portfolio.py` | **删除** |
| `tests/test_cli_integration.py` | backtest smoke 改为 backtest_verify_e2e |
| `docs/系统设计.md` | backtest_portfolio → backtest_verify_e2e |
| `docs/README.md` | 同上 |

## 验收清单

- [x] `make verify` 全量通过（含 step10 回测）
- [x] `pytest myanalyser/tests/ -v` 单测通过
- [x] `docs/系统设计.md`、`docs/README.md` 已更新
- [x] `tools/verify.sh` 已切换至新 backtest
- [x] 本需求已记入 `docs/需求日志/20260315_backtest_portfolio清理.md`

## 说明

- verify step10 产出由 `backtest_window_detail.csv` 变为 `period_detail.csv`（与 PyBroker 引擎统一）
- 选基仍从 ClickHouse `fact_fund_scoreboard_snapshot` 取，净值从本地 `fund_adjusted_nav_by_code` 加载

---

## 阶段 B：filter_one 解耦（2026-03-15）

### 需求简述

按 `docs/其他/20260315_阶段B_filter_one解耦技术评审.md` 执行阶段 B：backtest 解除对 filter_score 的依赖，将 `filter_one` 抽成共享模块。

### 实现方案

采用「保留 filter_score 流水线」路径：

1. 新建 `src/most_stable_logic.py`：承载 filter_one 及 _RULES、_to_float
2. `filter_score/filters/most_stable.py` 改为从共享模块 re-export（保证 step10b 动态加载不变）
3. `most_stable_strategy.py`、`test_backtest_pybroker_comprehensive.py`、`diagnose_most_stable_007540.py`、`test_filter_score` 改为从共享模块导入

**本次仅迁移 most_stable**，`non_a_unlimited`、`steady_aggressive` 等策略未改动。

### 变更清单

| 文件 | 操作 |
|------|------|
| `src/most_stable_logic.py` | **新增**（共享 filter_one 逻辑） |
| `src/filter_score/filters/most_stable.py` | 改为 re-export |
| `src/backtest/filters/most_stable_strategy.py` | 导入改为 `from most_stable_logic import filter_one` |
| `tests/test_backtest_pybroker_comprehensive.py` | 导入改为 `from myanalyser.src.most_stable_logic import filter_one` |
| `tests/test_filter_score.py` | 导入改为 `from myanalyser.src.most_stable_logic import filter_one as filter_most_stable` |
| `tools/temp_use/diagnose_most_stable_007540.py` | 导入改为 `from most_stable_logic import filter_one`；修复 sys.path（parents[2]/src） |
| `docs/系统设计.md` | 补充 most_stable_logic 说明 |
| `docs/README.md` | 同上 |

### 验收清单

- [x] `pytest myanalyser/tests/test_filter_score.py -v` 全通过
- [x] `pytest myanalyser/tests/test_backtest_engine_parallel.py -v -k most_stable` 通过
- [x] `tools/temp_use/diagnose_most_stable_007540.py` 在更新导入后可正常运行（需有效 --nav-dir）
- [x] `pytest myanalyser/tests/ -v -k "most_stable or filter"` 全通过
- [x] 需求日志中记录：本次仅迁移 most_stable，其余 filter 策略未改动
