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
- 阶段 B（filter_score 弱化/移除）未在本次执行
