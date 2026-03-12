# 20260312 most_stable 回测多线程优化与并行逻辑上移

## 需求背景
- `low_risk_debt_most_stable` 策略在 universe 约 4082 只基金、55 个调仓日场景下，回测耗时约 12 分钟。
- 瓶颈：每次调仓日对 4082 只基金逐只调用 `_compute_most_stable_metrics`（含 `compute_low_risk_debt_metrics` + `window_metrics`），计算量约 22.5 万次指标计算。

## 设计与讨论
- **多线程 vs 多进程**：filter_symbols 主要调用 numpy/pandas（会释放 GIL），多线程即可有效加速；多进程需 pickle 大量 DataFrame，序列化开销大。
- **并行位置**：将多线程逻辑放在 engine 主流程、而非 MostStableFilterStrategy 内部，使任意 FilterStrategy 自动受益，职责更清晰。
- **实现方式**：主流程将 universe 分块，每块调用 `filter_symbols(data, as_of_ts, chunk)`，多线程并行，主线程合并结果。`filter_symbols` 支持任意 universe 子集，无需改接口。

## 实现要点

### 1. engine 中增加通用并行包装
- `_split_chunks(lst, n)`：将 universe 均分为 n 份。
- `_filter_symbols_with_parallel(filter_strategy, data, as_of_ts, universe, threshold, max_workers)`：
  - universe ≤ threshold：串行调用 `filter_symbols`。
  - universe > threshold：分块并行，主线程收集结果后 `sorted()` 保证输出顺序确定。
- 调用处：`candidates = _filter_symbols_with_parallel(bundle.filter_strategy, data, current_ts, universe)`。

### 2. MostStableFilterStrategy 恢复为顺序实现
- 移除 `os`、`concurrent.futures` 导入。
- 移除 `_process_chunk`、`_split_chunks`。
- 恢复为单纯 `for symbol in universe` 循环。

### 3. Code Review 调整
- **魔法数字**：提取 `UNIVERSE_PARALLEL_THRESHOLD = 100` 模块级常量，注释说明与各 filter 典型计算量及线程成本相关。
- **线程安全约束**：FilterStrategy 协议 docstring 注明 `filter_symbols 须为无状态/线程安全`。
- **max_workers**：由 `(os.cpu_count() or 1) + 4` 改为 `min(32, os.cpu_count() or 4)`，避免 CPU 密集型场景过度订阅。
- **异常上下文**：`future.result()` 外层 try/except，失败时抛出带 chunk 信息的 `RuntimeError`，便于定位。

## 验收命令
```bash
cd /Users/zhuaoyuan/cursor-workspace/finance
source myanalyser/.venv312/bin/activate
pytest myanalyser/tests/test_backtest_pybroker_comprehensive.py myanalyser/tests/test_filter_score.py -v -k "most_stable or filter"
```

## 变更文件
- `myanalyser/src/backtest/engine.py`：新增 `UNIVERSE_PARALLEL_THRESHOLD`、`_split_chunks`、`_filter_symbols_with_parallel`，调用处改为并行包装。
- `myanalyser/src/backtest/filters/most_stable_strategy.py`：移除多线程逻辑，恢复顺序实现。
- `myanalyser/src/backtest/strategies/base.py`：FilterStrategy 协议增加线程安全 docstring。

## 备注
- 所有实现 FilterStrategy 协议的筛选策略（PassThroughFilter、MostStableFilterStrategy 及后续新增）在 universe > 100 时均自动使用多线程并行。
- 新增 FilterStrategy 须保证 `filter_symbols` 为无状态或线程安全。
