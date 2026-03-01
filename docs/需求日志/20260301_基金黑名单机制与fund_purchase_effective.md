# 需求日志：基金黑名单机制与 fund_purchase_effective

**日期**：2026-03-01  
**需求描述**：引入黑名单机制，在需要 fund_purchase 时，从中剔除黑名单基金，以加快整体处理速度；不修改 fund_purchase 原始数据，用 (fund_purchase − 黑名单) 代替 fund_purchase 的功能。

## 设计确认

| 决策点 | 结论 |
|--------|------|
| 黑名单文件 | `myanalyser/data/common/fund_blacklist.csv`，`FUND_BLACKLIST_PATH` 可覆盖 |
| 有效列表生成 | 始终生成 `fund_purchase_effective.csv`，后续步骤均依赖它 |
| 契约与测试 | `fund_purchase_effective_csv` 写入 pipeline_contracts，作为回归基线一部分 |
| 运行报告 | 记录「原始基金数 / 黑名单剔除数 / 有效基金数」 |

## 实现概要

- 新增 `src/transforms/build_effective_purchase_csv.py`：从 fund_purchase 剔除黑名单生成 effective
- 新增 `data/common/fund_blacklist.csv`（默认空，仅表头）
- `pipeline_contracts`：新增 `fund_purchase_effective_csv`、`fund_blacklist_csv` 及对应 stage
- `fund_etl.py`：新增 `--purchase-csv` 覆盖默认路径
- `filter_funds_for_next_step.py`：新增 `--purchase-csv`，默认优先 `fund_purchase_effective.csv`
- `run_full_pipeline.sh`：新增 step1a（准备 purchase）、step1b（build effective），step2~7 及 step6b 均使用 effective
- `verify.sh`：新增 step5b（build effective），后续步骤使用 effective；回归 verify 确认新逻辑正常
- `generate_run_report`：增加原始/黑名单剔除/有效基金数统计
