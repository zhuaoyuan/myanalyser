# 阶段 B filter_one 解耦技术评审报告

> 评审对象：20260315_backtest_portfolio 清理 — 阶段 B（backtest 解除对 filter_score 的依赖）  
> 评审日期：2026-03-15

## 原报告内容
结论：可以执行阶段 B
前提与路径如下：

采用“保留 filter_score 流水线”路径
step10b 仍用 run_filter_and_score.sh，因此：

保留 run_filter_and_score.sh、filter_and_score_main.py、test_filter_score
只做 backtest 对 filter_score 的解除依赖
先完成 filter_one 解耦，再改引用

新建共享模块（如 src/most_stable_logic.py），把 filter_one 及其依赖迁入
filter_score/filters/most_stable.py 改为从共享模块 re-export，保证 filter_and_score 动态加载仍可用
most_stable_strategy.py、test_backtest_pybroker_comprehensive.py、diagnose_most_stable_007540.py 改为从共享模块导入
暂不执行“完全移除 filter_score”

需要迁移所有 filter/score 策略并重写 step10b 验收，工程量较大
建议在 filter_one 解耦与 backtest 解耦验证通过后再规划
执行建议顺序
抽取 filter_one 到 src/most_stable_logic.py
修改 filter_score/filters/most_stable.py 为 re-export
修改 most_stable_strategy.py、test_backtest_pybroker_comprehensive.py、diagnose_most_stable_007540.py 的导入
运行 make verify 与相关单测（含 test_filter_score、test_backtest_pybroker_comprehensive）
更新文档（如 系统设计.md、README.md）
预估工作量（含测试与文档）
抽取与迁移：约 1–2 小时
验证与文档更新：约 0.5–1 小时

---

## 一、总体评价

**结论：方案可行，可执行。** 整体设计（保留 filter_score 流水线、仅将 backtest 依赖抽离到共享模块）清晰，风险可控，工作量评估合理。建议在以下点补充收紧后执行。

---

## 二、正确性验证

### 2.1 动态加载与 re-export 的兼容性

`load_filter_strategy` 通过 `importlib.util.spec_from_file_location` 按 **文件路径** 加载脚本，要求被加载模块暴露 `STRATEGY_NAME` 和 `filter_one`。

采用 re-export 形式：

```python
# filter_score/filters/most_stable.py
from ??? import STRATEGY_NAME, filter_one
```

在加载时会执行该模块，只要能成功 import 并暴露这两个符号即可，逻辑上可行。

**需在验收中明确**：step10b 仍使用 `-f src/filter_score/filters/most_stable.py`，`load_filter_strategy(Path("src/filter_score/filters/most_stable.py"))` 能正常加载，且 `test_filter_score` 全通过。

### 2.2 导入路径统一

当前各调用方导入方式不一致：

| 调用方 | 当前导入 |
|--------|----------|
| `most_stable_strategy.py` | `from filter_score.filters.most_stable import filter_one` |
| `test_backtest_pybroker_comprehensive.py` | `from myanalyser.src.filter_score.filters.most_stable import filter_one` |
| `diagnose_most_stable_007540.py` | `from filter_score.filters.most_stable import filter_one` |

迁移后统一为 `from ...most_stable_logic import filter_one`（具体包路径需结合 `sys.path` 约定）。注意：`filter_and_score_main` 启动时 `sys.path` 含 `finance`（workspace root），可 import `myanalyser.src.most_stable_logic`；而多数测试/工具将 `myanalyser/src` 加入 path，应使用 `from most_stable_logic import ...`。建议在实现前确定**单一路径约定**，并在所有迁移点统一采用，避免后续因 PYTHONPATH 差异再次分裂。

### 2.3 test_filter_score 的引用

`test_filter_score.py` 中有直接 import `filter_most_stable`，以及通过 `Path(...)/"src/filter_score/filters/most_stable.py"` 做动态加载测试。迁移后需更新直接 import，动态加载测试仍加载 `most_stable.py`（因 re-export 后行为不变），但需确认 `test_filter_score` 的 import 与现有用例一致。

---

## 三、风险与遗漏

### 3.1 循环依赖

`most_stable_strategy.py` 依赖 `filter_one`、`fund_metrics_core`、`scoreboard_metrics`、`compute_low_risk_debt_metrics` 等；`filter_one` 本身为纯规则函数，无外部依赖。将 `filter_one` 及 `_RULES`、`_to_float` 迁入 `most_stable_logic.py` 不会引入循环依赖。

### 3.2 test_backtest_engine_parallel

`test_backtest_engine_parallel.py` 含 `test_parallel_most_stable_filter_consistency`，需确认其是否直接或间接依赖 `filter_one`；若有直接 import，需一并改为从共享模块导入。

### 3.3 其他 filter 策略

计划仅迁移 `most_stable` 的 `filter_one`。`filter_score/filters/` 下尚有 `non_a_unlimited`、`steady_aggressive` 等，均实现 `filter_one` 接口。阶段 B 明确不迁移这些，建议在需求日志中注明：**本次仅解耦 most_stable，其余 filter 保持现状**，避免后续误解。

### 3.4 命名与放置

`src/most_stable_logic.py` 仅承载「最稳健原则」的规则逻辑，命名合理。若未来有更多 filter 迁移，可考虑规划 `src/filter_logic/` 目录；本次可先单文件实现，后续再重构。

---

## 四、执行顺序微调建议

原计划顺序合理，建议在「修改 backtest 引用」前增加一步：**先单独跑 `test_filter_score`**，确认 step10b 动态加载无虞，再批量改 `most_stable_strategy` / 测试 / 诊断脚本。这样若 step10b 出问题，可快速定位为 re-export 问题，而非 backtest 侧改动。

### 建议执行顺序

1. 抽取 `filter_one` 到 `src/most_stable_logic.py`
2. 修改 `filter_score/filters/most_stable.py` 为 re-export
3. **先单独跑 `test_filter_score`**，确认 step10b 动态加载无误
4. 再批量改 `most_stable_strategy.py`、`test_backtest_pybroker_comprehensive.py`、`diagnose_most_stable_007540.py`（及 `test_filter_score` 若有直接 import）
5. 运行 `make verify` 与 `test_backtest_engine_parallel`
6. 更新文档

---

## 五、验收清单补充

在现有验收基础上，建议补充：

- [ ] `pytest myanalyser/tests/test_filter_score.py -v` 全通过
- [ ] `pytest myanalyser/tests/test_backtest_engine_parallel.py -v -k most_stable` 通过（若存在）
- [ ] `tools/temp_use/diagnose_most_stable_007540.py` 在更新导入后仍可正常运行
- [ ] 需求日志中记录：本次仅迁移 most_stable，`non_a_unlimited`、`steady_aggressive` 等策略未改动

---

## 六、工作量评估

- 抽取与迁移：约 1–2 小时 —— **合理**（逻辑简单、无外部依赖）
- 验证与文档：约 0.5–1 小时 —— **合理**（建议预留 1 小时，以防 PYTHONPATH/导入问题排查）

总体 2–3 小时偏保守，可作为计划参考。

---

## 七、评审结论与建议

| 维度 | 结论 |
|------|------|
| 设计合理性 | ✅ 通过；re-export 保持 step10b 不变 |
| 风险可控性 | ✅ 低；无循环依赖，改动面清晰 |
| 执行顺序 | 建议先验证 filter_score 再改 backtest |
| 验收完整性 | 建议补充 test_filter_score、test_backtest_engine_parallel、diagnose 等验收项 |
| 文档更新 | 需覆盖 `系统设计.md`、`README.md` 及需求日志 |

**建议**：实现前在需求日志中明确「单一路径约定」及本次迁移范围；实现时按上述微调顺序执行，每步可单独验证。完成后可安全进入下一阶段。
