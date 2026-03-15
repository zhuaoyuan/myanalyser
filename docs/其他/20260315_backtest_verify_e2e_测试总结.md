# backtest_verify_e2e / verify_e2e_top5 测试总结

> 基于 diff 20260315_1145 / 20260315_1148（阶段 B filter_one 解耦 + SQL 注入防护 + 基金代码校验）

---

## 一、需求变更与测试范围

### 1.1 主要变更点（来自 diff）

| 模块 | 变更内容 |
|------|----------|
| `verify_e2e_top5.py` | `build_bundle_verify_e2e`：仅保留 6~8 位纯数字基金代码，排除全零（000000） |
| `backtest_verify_e2e.py` | SQL 注入防护：`_validate_clickhouse_db`、`_validate_container_name`、`_validate_selection_where`、`_validate_and_build_order_by` |
| `backtest_verify_e2e.py` | `_run_clickhouse_query` 增加 timeout、`CalledProcessError`/`TimeoutExpired` 异常处理 |
| `most_stable_logic` | filter_one 抽离到共享模块，filter_score re-export |

### 1.2 测试文件

- `myanalyser/tests/test_backtest_verify_e2e.py`（40 个用例）

---

## 二、场景清单

### 2.1 正常场景（Normal Scenarios）

| ID | 场景 | 输入/条件 | 预期输出 |
|----|------|-----------|----------|
| N1 | `_quote_sql` 普通字符串 | `"202401"` | `"'202401'"` |
| N2 | `_quote_sql` 单引号转义 | `"a'b"` | `"'a\\'b'"` |
| N3 | `_quote_sql` 反斜杠转义 | `"a\\b"` | `"'a\\\\b'"` |
| N4 | `_build_status_filter` 空列表 | `col, []` | `"1"` |
| N5 | `_build_status_filter` 单值 | `subscribe_status, ["暂停申购"]` | 包含列名与值 |
| N6 | `_build_status_filter` 多值 | `redeem_status, ["暂停赎回","封闭期"]` | 包含全部值 |
| N7 | `_fetch_fund_selection` 正常查询 | ClickHouse 返回多基金 | DataFrame 含 fund_code、weight，等权 |
| N8 | main 端到端成功 | 有效 nav-dir、选基 | 产出 period_detail.csv、backtest_report.md |
| N9 | FixedSelectionFilterStrategy 全命中 | universe 全在 allowed | 返回全部 symbol |
| N10 | FixedSelectionFilterStrategy 部分命中 | universe 部分在 allowed | 返回交集 |
| N11 | FixedSelectionScoreStrategy symbols 为空 | `symbols=[]` | 空 DataFrame |
| N12 | FixedSelectionScoreStrategy 顺序打分 | 按 allowed 顺序 | 综合得分/排名符合顺序 |
| N13 | build_bundle_verify_e2e 正常 | `["000001","000002"]` | 策略包 name/filter/score/position 非空 |
| N14 | `_validate_clickhouse_db` 合法 | `fund_analysis`, `db123_abc` | 无异常 |
| N15 | `_validate_container_name` 合法 | `fund_clickhouse`, `clickhouse-1.2` | 无异常 |
| N16 | `_validate_selection_where` 合法 | `"1"` | 无异常 |
| N17 | `_validate_and_build_order_by` 合法 | `annual_return DESC` / 多列 / 大写列名 | 返回正确 ORDER BY（列名 lower） |

### 2.2 异常场景（Exception Scenarios）

| ID | 场景 | 输入/条件 | 预期输出 |
|----|------|----------|----------|
| E1 | 选基为空 | ClickHouse 返回空 | `SystemExit` |
| E2 | 净值目录无有效数据 | CSV 格式错误 / 缺列 | `ValueError` |
| E3 | 查询返回空 | mock 返回空 DataFrame | 空 DataFrame（fund_code, weight） |
| E4 | 非法数据库名 | `fund;drop`, `db-name`, `db.name` | `ValueError("非法数据库名...")` |
| E5 | 非法容器名 | `bad;container` | `ValueError("非法容器名...")` |
| E6 | selection_where SQL 注入特征 | 含 `;`、`--`、`/*`、`*/`、`'`、`"`、`\` | `ValueError("非法字符...")` |
| E7 | ORDER BY 非法列名 | `evil_col DESC` | `ValueError("ORDER BY 不允许的列...")` |
| E8 | ORDER BY 非法方向 | `annual_return RANDOM` | `ValueError("ORDER BY 方向非法...")` |
| E9 | ORDER BY 为空 | `""` | `ValueError("ORDER BY 不能为空...")` |
| E10 | ORDER BY 仅逗号/空格 | `"  ,  , "` | `ValueError("不能为空...")` |

### 2.3 边界条件（Boundary Conditions）

| ID | 场景 | 输入/条件 | 预期输出 |
|----|------|----------|----------|
| B1 | allowed_symbols 为空 | `[]` | `allowed_symbols=()` |
| B2 | 基金代码补齐 6 位 | `["1","000002"]` | `("000001","000002")` |
| B3 | 去除前后空格 | `["  000001  ","000002"]` | `("000001","000002")` |
| B4 | 空字符串过滤 | `["000001","","000002"]` | `("000001","000002")` |
| B5 | 非数字代码剔除 | `["ABC123","000001"]` | `("000001",)` |
| B6 | 全零代码剔除 | `["000000","000001"]` | `("000001",)` |
| B7 | 8 位数字保留 | `["12345678"]` | `("12345678",)` |
| B8 | 超 8 位数字剔除 | `["123456789","000001"]` | `("000001",)` |
| B9 | universe 为空 | filter 时 universe=[] | `[]` |
| B10 | allowed 为空 universe 有值 | allowed=(), universe=["000001"] | `[]` |
| B11 | 单只基金 | top_n=1，1 只基金 | period_detail 非空 |
| B12 | top_n 大于选基数量 | top_n=10，2 只基金 | 取 min，产出正常 |
| B13 | exclude 为空字符串 | `--exclude-subscribe-status ""` | 解析为空列表，main 正常 |

---

## 三、测试执行结果

### Summary

| 指标 | 值 |
|------|-----|
| Total Tests | 40 |
| Passed | 40 |
| Failed | 0 |
| Coverage | 单文件覆盖未单独测量（可执行 `pytest --cov=...` 补充） |

### Failures Detail

**无失败用例。** 所有 40 个测试均通过。

---

## 四、失败分析（本次无）

| Test ID | Scenario | Input | Expected | Actual | Traceback/Error | 根因 |
|---------|----------|-------|----------|--------|-----------------|------|
| — | — | — | — | — | — | 无 |

*说明：若后续出现失败，可区分为「代码实现错误」或「需求理解歧义」填入上表。*

---

## 五、验收清单与执行命令

```bash
cd /Users/zhuaoyuan/cursor-workspace/finance
source myanalyser/.venv312/bin/activate
pytest myanalyser/tests/test_backtest_verify_e2e.py -v
```

---

## 六、附录：未覆盖/潜在扩展

- **并发冲突**：当前无多进程/多线程选基逻辑，未设计并发测试。
- **_run_clickhouse_query 超时/失败**：通过 mock 已避开真实 docker 调用，未单测 `TimeoutExpired`/`CalledProcessError` 分支（可补充 mock 异常用例）。
- **project_paths**：main 依赖 `project_root()`，测试通过 patch 与临时目录规避，未测路径解析逻辑。
