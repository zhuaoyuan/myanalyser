# multi_t_backtest 基金过滤步骤追踪

> 基于 `multi_t_backtest.py` 与 `--strategy low_risk_debt_most_stable` 的完整基金过滤流程梳理。

## 命令示例

```bash
python myanalyser/tools/v2/multi_t_backtest.py \
  --run-id "20260315_123456_full_run_v2" \
  --ruleset-version "20260316_v1" \
  --t-start 2025-01-01 --t-end 2025-03-01 --t-step 25 \
  --trading-calendar-csv "myanalyser/data/common/trade_dates.csv" \
  --prep-work-dir "myanalyser/tmp/prep_work_v2" \
  --lookback-years 3 \
  --hold-days 243 \
  --strategy "low_risk_debt_most_stable" \
  --max-funds 5000 \
  --rebalance 0 \
  --top-n 5 \
  --warmup 0 \
  --initial-cash 100000
```

---

## 过滤步骤总览

| 步骤 | 模块/函数 | 输入 | 输出 | 过滤前 | 过滤后 | 过滤条件 |
|-----|----------|------|------|--------|--------|----------|
| 1 | prep_eligible_window | fund_purchase.csv | eligible_fund_candidates.csv | 见下 | 见下 | c.1 + a + b + e |
| 2 | filter_funds_for_next_step | eligible_fund_candidates.csv | filtered_fund_candidates.csv | eligible 数量 | 是否过滤=否 数量 | 规则1-5 |
| 3 | build_filtered_purchase_csv | eligible + filter | fund_purchase_for_step10_filtered.csv | eligible 行数 | 保留行数 | 是否过滤=否 |
| 4 | load_fund_nav_data | allowed_codes + max_funds | BacktestData | allowed 数量 | min( allowed, max_funds ) | 仅加载 allowed 内基金，上限 max_funds |
| 5 | MostStableFilterStrategy | universe | candidates | 加载的基金数 | 通过 filter_one 的基金数 | 最稳健原则 9 条规则 |

---

## 步骤 1：prep_eligible_window（预备可申购候选）

**位置**: `src/v2/filters/prep_eligible_window.py` → `run()`

**输入**:
- `prep_work_dir/fund_purchase.csv`（基金代码列）
- `fund_cyrjg.csv`、`fund_gmbd.csv`、`fund_fee_structured.csv`、`fund_overview.csv`

**输出**: `eligible_fund_candidates.csv`

### 过滤条件与数量变化

| 阶段 | 过滤条件 | 过滤前 | 过滤后 |
|-----|----------|--------|--------|
| 原始 | - | `len(fund_purchase 去重基金代码)` | - |
| c.1 | 必须在 `fund_fee_filtered.csv` 中存在 | 原始数量 | `codes &= c1_codes` |
| a | 排除：在 `[成立+2年, 窗口 end_date]` 内机构持仓比例**连续两次**>60% 的基金（不做 start_ts 裁剪） | c.1 后 | `codes -= exclude_a` |
| b | 仅保留：`end_date` 前**最新一条**规模 > 2 亿的基金（非窗口内任一条） | a 后 | `codes &= include_b` |
| e | 仅保留：`start_date` 之前成立的基金（有成立日期且成立日 < start_date） | b 后 | `codes &= include_e` |

**日志示例**:
```
[eligible] 原始候选 XXXX 只
[eligible] c.1 后 XXXX
[eligible] a([成立+2年,end_date]内连续两次>60%排除) 后 XXXX，排除 XX
[eligible] b(end_date前最新规模>2亿) 后 XXXX
[eligible] e(start_date前成立) 后 XXXX
[eligible] 最终结果 XXXX 只
```

---

## 步骤 2：filter_funds_for_next_step（数据质量过滤）

**位置**: `src/v2/filters/filter_funds_for_next_step.py` → `filter_funds_for_next_step()`

**输入**: `eligible_fund_candidates.csv`（步骤 1 输出）

**输出**: `filtered_fund_candidates.csv`，列：`基金编码`、`是否过滤`、`过滤原因`

### 过滤条件（规则 1–5）

任一条不满足则 `是否过滤 = "是"`：

| 规则 | 条件 |
|-----|------|
| 规则 1 | 基金必须在 `fund_overview.csv`（fund_etl 目录）中 |
| 规则 2 | 必须在 `fund_nav_by_code` 目录中存在 NAV 原始净值 |
| 规则 3 | 必须在 `fund_adjusted_nav_by_code` 目录中存在复权净值 |
| 规则 4 | Compare 明细在 `[start_date, end_date + hold_days]` 内（按交易日历延伸），本地/远程收益率偏差绝对值 < `max_abs_deviation`（默认 0.02） |
| 规则 5 | Integrity 明细在 `[start_date, end_date + hold_days]` 内（按交易日历延伸），各交易日数据完整（该日期数据是否存在 = "是"） |

**数量说明**:
- 过滤前：eligible 中基金数
- 过滤后：`是否过滤 == "否"` 的基金数（即 `allowed_codes`）

**获取方式**: 读取 `filtered_fund_candidates.csv` 统计 `是否过滤` 列。

---

## 步骤 3：build_filtered_purchase_csv（生成 step10 申购清单）

**位置**: `src/transforms/build_filtered_purchase_csv.py` → `build_filtered_purchase_csv()`

**输入**:
- `eligible_fund_candidates.csv`
- `filtered_fund_candidates.csv`

**输出**: `fund_purchase_for_step10_filtered.csv`

**过滤逻辑**: 保留 `基金代码 in { 是否过滤=="否" 的基金编码 }` 的行。

- 过滤前：eligible 的行数
- 过滤后：`kept_df` 的行数（与 allowed_codes 对应的基金）

---

## 步骤 4：load_fund_nav_data（净值加载与数量限制）

**位置**: `src/backtest/data.py` → `load_fund_nav_data()`

**输入**:
- `allowed_codes`：步骤 2 中 `是否过滤=="否"` 的基金集合
- `max_funds=5000`（命令行）

**逻辑**:
1. 取 `nav_dir` 下所有 `*.csv`，与 `allowed_codes` 取交集
2. 按文件名排序，取前 `max_funds` 个
3. 加载这些基金的复权净值

**数量说明**:
- 过滤前：`len(allowed_codes)`（与 nav_dir 交集后的理论数量）
- 过滤后：`min(len(allowed ∩ nav_files), max_funds)`，且只加载格式正确、有有效净值数据的基金

---

## 步骤 5：MostStableFilterStrategy（策略内筛选）

**位置**: `src/backtest/filters/most_stable_strategy.py` → `filter_symbols()`

**调用时机**: 回测每个调仓日（`rebalance_period=0` 时仅首日）在 `engine.run_backtest` 的 `before_exec` 中。

**输入**: `universe = sorted(data.by_symbol.keys())`（步骤 4 加载的基金列表）

**输出**: `candidates`（通过 `filter_one` 的基金列表）

### filter_one 规则（src/most_stable_logic.py）

全部满足才通过，任一条不满足即过滤：

| 指标 | 条件 |
|------|------|
| 近3年年化收益率 | > 3（%） |
| 近1年年化收益率 | > 3（%） |
| 近3年上涨季度比例 | > 80（%） |
| 近3年上涨月份比例 | > 70（%） |
| 近3年月涨跌幅标准差 | < 1.5（%） |
| 近1年夏普比率 | > 1 |
| 近3年夏普比率 | > 1 |
| 近1年卡玛比率 | > 1 |
| 近3年卡玛比率 | > 1 |

**数量说明**:
- 过滤前：`len(universe)`（加载的基金数）
- 过滤后：`len(candidates)`（通过 `filter_one` 的基金数）

**日志**: `period_log` 中每期有 `universe_size`、`candidate_size`。

---

## 如何获取实际数量

### 1. 从日志

运行命令后查看：
- `[eligible]` 相关日志：步骤 1 各阶段数量
- 其他步骤当前未打“过滤前/后”日志

### 2. 从缓存 CSV

假设 T 日为 `2025-01-02`，窗口为 `2022-01-02_2025-01-02`：

```
data/versions/{run_id}/cache/v2/
├── prep_eligible/{ruleset_version}/{start}_{end}/
│   └── eligible_fund_candidates.csv          # 步骤1 输出
├── filter/{ruleset_version}/{start}_{end}/
│   ├── filtered_fund_candidates.csv          # 步骤2 输出（含 是否过滤、过滤原因）
│   └── fund_purchase_for_step10_filtered.csv # 步骤3 输出
```

统计方式：
```bash
# 步骤1 后数量
wc -l eligible_fund_candidates.csv

# 步骤2 过滤前、过滤后
# 过滤前 = 总行数 - 1（表头）
# 过滤后 = 是否过滤=="否" 的行数
grep -c '"否"' filtered_fund_candidates.csv
```

### 3. 从 period_log

回测结果中的 `period_log` 有每期 `universe_size`、`candidate_size`，对应步骤 4、5 的过滤前后。

---

## 数据流简图

```
fund_purchase.csv (prep_work_dir)
    │
    ▼ [步骤1: prep_eligible_window]
eligible_fund_candidates.csv
    │
    ▼ [步骤2: filter_funds_for_next_step]
filtered_fund_candidates.csv (是否过滤, 过滤原因)
    │
    ├─► allowed_codes = { 是否过滤=="否" }
    │
    ▼ [步骤3: build_filtered_purchase_csv]
fund_purchase_for_step10_filtered.csv  (供 scoreboard)
    │
    ▼ [步骤4: load_fund_nav_data(allowed_codes, max_funds)]
BacktestData (by_symbol)
    │
    ▼ [步骤5: MostStableFilterStrategy.filter_symbols]
candidates → 评分 → top_n 持仓
```

---

## 附录：相关文件路径

| 模块 | 文件 |
|------|------|
| prep_eligible | `myanalyser/src/v2/filters/prep_eligible_window.py` |
| filter_funds_for_next_step | `myanalyser/src/v2/filters/filter_funds_for_next_step.py` |
| build_filtered_purchase_csv | `myanalyser/src/transforms/build_filtered_purchase_csv.py` |
| load_fund_nav_data | `myanalyser/src/backtest/data.py` |
| MostStableFilterStrategy | `myanalyser/src/backtest/filters/most_stable_strategy.py` |
| filter_one 规则 | `myanalyser/src/most_stable_logic.py` |
