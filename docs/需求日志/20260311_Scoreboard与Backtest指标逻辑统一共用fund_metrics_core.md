# 需求日志：Scoreboard 与 Backtest 指标逻辑统一共用 fund_metrics_core

- 日期：2026-03-11
- 范围：`src/fund_metrics_core.py`（新增）、`src/backtest/metrics.py`、`src/scoreboard_metrics.py`、`tests/conftest.py`、`tests/baseline/mini_case/expected/default/`、`tools/compare_metrics_logic.py`

## 1. 背景

回测策略（`low_risk_debt`）与评分榜 pipeline（`scoreboard_metrics`）此前使用两套独立实现的指标计算逻辑，导致：
- 年化收益：Backtest 用 252 交易日/年，Scoreboard 用 365 自然日
- 最近一个月：Backtest 用最近 21 个交易日，Scoreboard 用「上个完整自然月」
- 回撤修复天数：Backtest 算「峰顶→收复」全程时长，Scoreboard 算「谷底→收复」回升时长

同一份数据下指标差异可达 10% 以上，综合得分与排名可能不一致。

## 2. 开发者诉求（按轮次）

1. **第一轮**：澄清算分过程是否重新调用 pipeline_scoreboard——确认为否，两套逻辑独立。
2. **第二轮**：用同一份数据验算两套逻辑差异，输出对比报告。
3. **第三轮**：希望将 Scoreboard 逻辑改为与 Backtest 一致，方式是将 Backtest 计算逻辑提取出来，双方共用；完成后再次比对。
4. **第四轮**：针对 review 意见做调整（sys.path、max_drawdown NaN、window_metrics years 校验等）。
5. **第五轮**：针对最新 review（up_month/up_week 分支遗漏、conftest 仅 pytest 生效）评估并确认调整。
6. **第六轮**：将本次变更记录到需求日志。

## 3. AI 方案与实现

### 3.1 新增共享模块 fund_metrics_core.py

- 从 `backtest/metrics` 提取核心计算函数：`cagr`、`max_drawdown`、`longest_recovery_days`、`return_over_period`、`rolling_returns`、`sharpe_ratio`。
- 统一口径：252 交易日/年、21 日/月、5 日/周；回撤修复天数 = 峰顶到收复全程（含下跌+回升）。
- 主入口：`compute_low_risk_debt_metrics(dates, prices)`，返回 12 项中文列名指标；`CN_TO_EN_LOW_RISK` 供 Scoreboard 映射英文列名。

### 3.2 backtest/metrics.py 改造

- 移除本地实现，改为从 `fund_metrics_core` 导入并导出。
- 增加 `try/except ModuleNotFoundError` 兜底：当 src 不在 path 时（CLI/非 pytest 直接运行），自动注入 src 后重试导入。

### 3.3 scoreboard_metrics.py 改造

- `compute_metrics`：年化收益、最大回撤、回撤修复天数、最近一个月涨跌幅改用 fund_metrics_core。
- `window_metrics`：窗口由「自然日区间」改为「最近 252/756 个交易日」；重叠指标全部使用 `compute_low_risk_debt_metrics`；仅支持 `years=1` 或 `3`，增加 `ValueError` 校验。
- 移除 `_TRADING_DAYS_*` 魔法数字，改用 `WindowConfig()` 常量。

### 3.4 验算工具 compare_metrics_logic.py

- 新增脚本：加载同一 nav 数据，分别调用 backtest 与 scoreboard 两套路径计算指标与综合得分，逐项对比并输出明细 CSV。
- 用法：`python myanalyser/tools/compare_metrics_logic.py --nav-dir <path> --as-of-date YYYY-MM-DD`

### 3.5 测试与 review 相关调整

- 新增 `tests/conftest.py`：pytest 启动时注入 `myanalyser/src` 到 `sys.path`，避免在业务代码中修改 path。
- `fund_metrics_core.max_drawdown`：输入含 NaN 或结果为 NaN 时返回 `None`，风格与其他函数一致。
- `scoreboard_metrics.window_metrics`：`years not in (1, 3)` 时抛出 `ValueError`。
- `verify_backtest_logic.py`：在 `if period_log` 前初始化 `score_check = weight_check = {"status": "skip"}`，避免异常路径下 `NameError`。
- 新增单测：`test_window_metrics_years_validation`。
- 回归基线：更新 `tests/baseline/mini_case/expected/default/` 下 fund_scoreboard、scoreboard_recheck 等 CSV。

## 4. 验收与证据

1. **compare_metrics_logic 比对结果**：
   - 12 项指标全部 pass；
   - 综合得分差 0.000000，排名变化 0；
   - 结论：两套逻辑指标一致。

2. **单测**：
   - `test_scoreboard_metrics`、`test_backtest_pybroker_comprehensive`、`test_pipeline_regression_baseline` 通过；
   - `test_window_metrics_years_validation` 验证 `years=2` 抛出 `ValueError`。

3. **CLI/pipeline 直接运行**：
   - `pybroker_fund_backtest`、`pipeline_scoreboard` 等入口在无 conftest 场景下通过 backtest/metrics 的 fallback 导入正常执行。

## 5. 变更文件清单

| 文件 | 变更类型 |
|------|----------|
| `src/fund_metrics_core.py` | 新增 |
| `src/backtest/metrics.py` | 精简为导入 + fallback |
| `src/scoreboard_metrics.py` | 改用 fund_metrics_core |
| `tests/conftest.py` | 新增 |
| `tests/test_scoreboard_metrics.py` | 新增 years 校验单测 |
| `tools/compare_metrics_logic.py` | 新增 |
| `tools/debug_metrics_diff.py` | 新增（分析用，可选） |
| `tools/verify_backtest_logic.py` | 防御性初始化 |
| `tests/baseline/mini_case/expected/default/*` | 更新 4 个 CSV |

## 6. 元文件同步

本轮已更新：

- `docs/需求日志/20260311_Scoreboard与Backtest指标逻辑统一共用fund_metrics_core.md`：本需求日志。
- `docs/README.md`：补充 `fund_metrics_core`、`scoreboard_metrics` 与 backtest 共用口径说明。
- `docs/系统设计.md`：同上。

验算工具用法（可纳入 README 常用命令）：

```bash
python myanalyser/tools/compare_metrics_logic.py \
  --nav-dir <fund_adjusted_nav_by_code 路径> \
  --as-of-date YYYY-MM-DD \
  --max-funds 20
```
