# 需求日志：PyBroker 回测报告增强与 HTML 收益曲线

- 日期：2026-03-11
- 范围：`src/backtest/engine.py`、`tools/pybroker_fund_backtest.py`、`requirements*.txt`、`tests/test_backtest_pybroker_comprehensive.py`

## 1. 背景

此前 pybroker_fund_backtest 仅输出 summary.csv 与 period_detail.csv，缺少人类可读报告、运行参数持久化、控制台摘要、净值曲线、以及组合与成分基金的收益曲线可视化。

## 2. 开发者诉求

1. 报告形式改进：Markdown 报告、运行参数写回 summary、控制台输出核心指标
2. 报告内容增强：period_return、equity_curve.csv、orders.csv、positions_flat.csv、fund_metrics_core 指标对齐
3. HTML 可视化：组合收益曲线 + 成分基金收益曲线（Plotly）作为对照

## 3. AI 方案与实现

### 3.1 write_reports 扩展

- 新增参数：`run_config`（运行参数字典）、`initial_cash`
- 输出文件：summary.csv、period_detail.csv、equity_curve.csv、orders.csv、positions_flat.csv、backtest_report.md、backtest_curves.html（可选）
- period_detail 增加 `period_return` 列
- summary 增加 config section（strategy、rebalance、top_n 等）与 fund_metrics_core 指标（年化、回撤、夏普、卡玛等）
- 使用 fund_metrics_core 基于组合净值曲线计算指标，与 Scoreboard 口径一致

### 3.2 Plotly HTML 曲线

- 双图布局：① 组合净值曲线 ② 组合 vs 成分基金（最多 10 只，按持有期排序）
- 归一化到「1 元投入」累计收益率，便于对比
- 自包含 HTML，需 `plotly>=5.0` 依赖

### 3.3 pybroker_fund_backtest 调用方

- 传入 run_config 与 initial_cash
- 控制台输出核心指标（年化、回撤、夏普等）及所有输出路径
- 依赖：requirements.txt / requirements-lock.txt 增加 plotly

### 3.4 边界处理

- period_log 为空时，detail/equity_curve 仍写出带表头的空 CSV
- 无 plotly 时，curves_html 不生成，不影响其余输出

## 4. 验收

- `test_write_reports_creates_summary_and_detail`：验证 summary、detail、equity_curve、orders、positions_flat、report_md 存在，detail 含 period_return 列
- 27 个 test_backtest_pybroker_comprehensive 单测全部通过

## 5. 变更文件

| 文件 | 变更 |
|------|------|
| src/backtest/engine.py | write_reports 大改：run_config、equity_curve、orders、positions_flat、fund_metrics_core、Markdown、Plotly HTML |
| tools/pybroker_fund_backtest.py | run_config 传入、控制台摘要、pandas 导入 |
| requirements.txt | 新增 plotly>=5.0 |
| requirements-lock.txt | 新增 plotly==5.24.1 |
| tests/test_backtest_pybroker_comprehensive.py | 扩展 test_write_reports 断言 |
| docs/README.md | 新增「PyBroker 回测输出文件」表 |
| docs/需求日志/20260311_pybroker回测报告增强与HTML曲线.md | 本需求日志 |
