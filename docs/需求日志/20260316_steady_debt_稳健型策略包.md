# 需求日志：steady_debt 稳健型策略包

**日期**: 2026-03-16  
**需求描述**: 依据 `docs/参考/分类型的硬约束和主次目标.md` 第二节「稳健型（低波动偏债）」策略描述，编写 `multi_t_backtest.py` 可用的筛选+打分策略 bundle，命名为 `steady_debt`。

---

## 1. 背景与痛点

当前已有 `low_risk_debt`、`low_risk_debt_most_stable` 策略包，后者在筛选阶段应用最稳健原则（9 条规则，如年化>3%、夏普>1、卡玛>1 等）。参考文档定义了四种投资风格的主次目标模板，其中「稳健型」与 most_stable 定位接近但硬约束和打分侧重不同：

| 层级 | 稳健型（低波动偏债） | 说明 |
|------|----------------------|------|
| **硬约束** | 最大回撤 ≥ -8% | 回撤不能比 -8% 更差 |
| | 年化收益 ≥ 5% | |
| | 夏普比率 ≥ 0.5 | |
| **主目标** | 卡玛比率（越大越好） | 收益/回撤性价比 |
| **次目标 1** | 夏普比率（越大越好） | |
| **次目标 2** | 最大回撤（越浅越好） | |

希望新增独立策略包，与文档定义一一对应，便于后续按风格扩展（固收型、均衡型、进取型）。

---

## 2. 方案与实现

### 2.1 设计

- **筛选（Filter）**：新建 `steady_debt_logic.py` 实现 `filter_one`，硬约束三条（全部满足才通过）
- **打分（Score）**：主目标卡玛 60%，次目标夏普 25%、回撤 15%（回撤越浅越好，asc 方向）
- **仓位**：复用 `EqualWeightPosition`

### 2.2 指标口径

- 指标为 `fund_metrics_core.compute_low_risk_debt_metrics` 输出的小数形式
- 近 3 年窗口：243×3 交易日
- 最大回撤 ≥ -8%：`近3年最大回撤率 >= -0.08`
- 年化 ≥ 5%：`近3年年化收益率 >= 0.05`
- 夏普 ≥ 0.5：`近3年夏普比率 >= 0.5`

### 2.3 变更文件

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `myanalyser/src/steady_debt_logic.py` | 新增 | 稳健型 filter_one，区分 None/阈值 reason |
| `myanalyser/src/backtest/filters/steady_debt_strategy.py` | 新增 | SteadyDebtFilterStrategy |
| `myanalyser/src/backtest/strategies/steady_debt.py` | 新增 | SteadyDebtScoreStrategy + build_bundle_steady_debt |
| `myanalyser/src/backtest/strategies/registry.py` | 修改 | 注册 steady_debt |
| `myanalyser/src/backtest/filters/__init__.py` | 修改 | 导出 SteadyDebtFilterStrategy |
| `myanalyser/tests/test_backtest_pybroker_comprehensive.py` | 修改 | 新增 test_registry_steady_debt、test_steady_debt_logic_filter_one（含边界） |
| `myanalyser/docs/README.md` | 修改 | 补充 steady_debt 策略说明 |

---

## 3. 实现要点

- Filter 与 most_stable 类似：基于目标日前净值动态计算指标，调用 `filter_one` 判定
- Score 使用 `compute_composite_score`，传入自定义 `secondary_groups` 实现主次目标权重
- 导入：与 most_stable 一致，`steady_debt_logic`、`compute_fund_composite_score` 等依赖运行时 PYTHONPATH 含 `src`（conftest / sys.path 统一处理）
- `SteadyDebtScoreStrategy.score` docstring 明确：调用方应保证 symbols 已通过 filter（含 3y 窗口）

---

## 4. Code Review 反馈与调整

| # | 意见 | 评估 | 处理 |
|---|------|------|------|
| 1 | 导入改相对路径 | 不采纳 | 与 most_stable 一致，steady_debt_logic 与 backtest 同级，无法用 `...` 相对导入 |
| 2 | WindowConfig 移至模块顶部 | 采纳 | 已调整 |
| 3 | compute_fund_composite_score 导入可移植性 | 不采纳 | 与 low_risk_debt 相同 |
| 4 | max_dd reason 区分 None/阈值 | 采纳 | 拆为「缺失」与「<-8%」 |
| 5 | ScoreStrategy docstring 说明 3y 前置 | 采纳 | 已补充 |
| 6 | filter_symbols 性能 | 不采纳 | 与 most_stable 一致，当前可接受 |
| 7 | 边界测试 -0.08/-0.0801 | 采纳 | 已补充 |
| 8 | 抽取共享 compute_steady_debt_metrics | 不采纳 | 与 most_stable 结构一致 |
| 9 | list_strategy_names 下游兼容 | 采纳 | README 已更新 |
| 10 | _to_float 类型注解 | 采纳 | 添加 `val: Any` |

---

## 5. 验收证据

### 5.1 单测

```bash
cd /Users/zhuaoyuan/cursor-workspace/finance
source myanalyser/.venv312/bin/activate
cd myanalyser
pytest tests/test_backtest_pybroker_comprehensive.py tests/test_backtest_low_risk_debt.py -v -k "steady_debt or registry or low_risk"
```

- `test_registry_steady_debt_uses_steady_debt_filter`：验证 steady_debt 使用 SteadyDebtFilterStrategy
- `test_steady_debt_logic_filter_one`：硬约束通过/过滤、边界（-0.08 通过、-0.0801 过滤）

### 5.2 使用方式

```bash
python myanalyser/tools/v2/multi_t_backtest.py \
  --run-id "..." \
  --ruleset-version "..." \
  --strategy "steady_debt" \
  --trading-calendar-csv "myanalyser/data/common/trade_dates.csv" \
  --prep-work-dir "myanalyser/tmp/prep_work_v2" \
  ...
```

---

## 6. 项目元文件更新

- `myanalyser/docs/README.md`：策略包说明中补充 `steady_debt`，与 `low_risk_debt`、`low_risk_debt_most_stable` 并列
- `list_strategy_names()` 现返回 `['low_risk_debt', 'low_risk_debt_most_stable', 'steady_debt']`，依赖该列表的脚本/文档已确认兼容
