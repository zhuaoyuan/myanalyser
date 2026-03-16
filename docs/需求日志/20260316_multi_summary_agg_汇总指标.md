# 需求日志：multi_summary_agg.csv 增加汇总分析指标

**日期**: 2026-03-16  
**需求描述**: 在 multi_t_backtest 结束后，使输出的 `multi_summary.csv` 目录下直接包含跨 T 的汇总分析指标 `multi_summary_agg.csv`，无需再单独跑 pandas 脚本即可评估策略整体表现。

---

## 1. 背景与痛点

当前 `multi_summary.csv` 每行对应一个 T 日回测，列包含：`as_of_date`、`filter_start`、`filter_end`、`backtest_start`、`backtest_end`、`allowed_funds`、各 `metrics_holding` 指标（年化收益率、最大回撤率、夏普比率等）及若干路径列。

当 T 日较多（如 t-step=25 时有几十组）时，需自行用 pandas 做聚合分析才能得到：
- 各指标在全 T 样本的均值、标准差、分位数等
- 正收益比例、极端回撤比例等稳健性指标

希望这些汇总指标能在 multi_summary 输出时自动产生，方便快速评估策略。

---

## 2. 方案与实现

### 2.1 输出形态

- `multi_summary.csv`：保持当前结构，仅 per-T 明细行
- `multi_summary_agg.csv`：同一目录下新增，宽表多行，记录跨 T 的汇总统计

### 2.2 汇总指标内容

针对 `multi_summary.csv` 中已有的数值型 `metrics_holding` 列，在 `multi_summary_agg.csv` 中输出：

| 类别 | 指标名称 | 说明 |
|------|----------|------|
| **描述统计** | mean | 均值 |
| | median | 中位数 |
| | std | 标准差 |
| | min | 最小值 |
| | max | 最大值 |
| | p25 | 25 分位数 |
| | p75 | 75 分位数 |
| | count | 有效样本数 |
| **稳健性** | win_rate | 年化收益率 > 0 的比例 |
| **元信息** | t_count | T 日总数 |

实际输出仅包含 `multi_summary` 中存在的列。

### 2.3 文件格式

宽表多行：
```
stat_type,年化收益率,最大回撤率,夏普比率,...
mean,0.052,-0.028,1.15,...
median,0.048,-0.025,1.08,...
std,0.018,0.012,0.32,...
win_rate,0.666667,,,...
t_count,3,,,...
```

### 2.4 变更文件

| 文件 | 变更类型 |
|------|----------|
| `myanalyser/tools/v2/multi_t_backtest.py` | 修改：新增 `_write_multi_summary_agg` |
| `myanalyser/tests/test_multi_summary_agg.py` | 新增：单测 |

---

## 3. 实现要点

- T 循环结束后、写入 `multi_summary.csv` 之后调用 `_write_multi_summary_agg(summary_df, output_root)`
- 解析 `summary_df` 中数值型 metrics 列，空串/非数值按 NaN 处理
- 按需求计算 mean/median/std/min/max/p25/p75/count、win_rate、t_count
- 某指标列全为 NaN 时对应 agg 列为空；T=0 时不生成文件；单 T 时 std 为空
- 日志打印 `[multi] agg summary -> {path}`

---

## 4. 验收证据

### 4.1 单测

```bash
cd /Users/zhuaoyuan/cursor-workspace/finance
source myanalyser/.venv312/bin/activate
cd myanalyser
pytest tests/test_multi_summary_agg.py -v
```

结果：5 passed（含 agg 与 df.describe 一致性、t_count、空 summary 跳过、单 T std、win_rate）

### 4.2 multi_t_backtest 运行

```bash
python myanalyser/tools/v2/multi_t_backtest.py \
  --run-id "20260315_123456_full_run_v2" \
  --ruleset-version "20260315_v1" \
  --t-list "2023-01-03,2023-07-03,2024-01-02" \
  --trading-calendar-csv "myanalyser/data/common/trade_dates.csv" \
  --prep-work-dir "myanalyser/tmp/prep_work_v2" \
  --lookback-years 3 \
  --hold-days 21 \
  --strategy "low_risk_debt_most_stable" \
  ...
```

验收：
- [x] `output_root` 下存在 `multi_summary_agg.csv`
- [x] 汇总指标与对 `multi_summary.csv` 手动 `df.describe()` 等结果一致（容差 1e-6）
- [x] t_count 等于 `multi_summary.csv` 行数

---

## 5. 项目元文件更新建议

（需经开发者确认后写入）

- `myanalyser/docs/系统设计.md`：补充 multi_summary_agg.csv 产物说明
- `myanalyser/docs/README.md`：如已有 multi_t_backtest 输出说明，增加 multi_summary_agg.csv 描述
- `myanalyser/docs/其他/multi_t_backtest_时序图.md`：产物列表增加 multi_summary_agg.csv
