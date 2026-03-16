# 需求日志：prep_eligible_window 人事变动过滤

**日期**: 2026-03-16  
**需求描述**: 在 prep_eligible_window 过滤阶段新增规则 f，排除在 [目标窗口 end-1年, 目标窗口 end] 期间有人事变动记录的基金。结果缓存方式同现有其他条件。

---

## 1. 需求背景

- 目标窗口 end 即为 T（选基日），不包括 hold-days
- 人事信息来源于 `fund_etl/fund_personnel_by_code`，每只基金对应 `{code}.csv`
- 人事 CSV 含 `公告日期` 列，每条记录表示一次人事变动

---

## 2. 实现内容

### 规则 f

| 项目 | 内容 |
|------|------|
| **规则** | 排除在 [end_date-1年, end_date] 内有人事变动记录的基金 |
| **数据源** | `fund_personnel_by_code` 目录下的 `{code}.csv` |
| **日期列** | `公告日期`（兼容 `日期`） |
| **缓存** | 拆分为 `eligible_base_{start}_{end}.csv`（不依赖 personnel）+ `personnel_excluded_{start}_{end}.csv`，加载时合并；与 base 共用 `eligible_fund_candidates.csv` 最终输出 |

### 调用方式

- `run()` 新增可选参数 `personnel_dir: Path | None = None`
- `personnel_dir` 为 `None` 或目录不存在时，跳过规则 f
- `multi_t_backtest` 传入 `personnel_dir=fund_etl_dir / "fund_personnel_by_code"`
- `fund_personnel_by_code` 由 run_full_pipeline step6 产出；若 ETL 未执行该步骤则 rule f 自动跳过

---

## 3. 变更文件列表

| 文件 | 变更类型 |
|------|----------|
| `myanalyser/src/v2/filters/prep_eligible_window.py` | 修改 |
| `myanalyser/tools/v2/multi_t_backtest.py` | 修改 |
| `myanalyser/tools/v2/verify_filter_flow_report.py` | 修改 |
| `myanalyser/tests/test_v2_phase0_2_wide_store_narrow_use.py` | 新增 rule f 测试 |

---

## 4. 验收

- `test_rule_f_personnel_in_window_excluded`：窗口内有人事变动 → 排除
- `test_rule_f_personnel_outside_window_retained`：人事变动在窗口外 → 保留
- TestPrepEligibleWindow 全部 18 个用例通过

---

## 5. Review 意见处理（2026-03-16）

| 意见 | 处理 |
|------|------|
| `except Exception` 过宽 | 限定为 `(OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError)`，并记录 debug log |
| `for code in list(codes)` 冗余 | 改为 `for code in codes` |
| `_has_personnel_in_window` 未做 code 校验 | 增加 `if not code or not code.isdigit() or len(code) > 6: return False` |
| 测试断言不够明确 | 为 `test_rule_f_personnel_outside_window_retained` 增加 assert 提示消息 |
| 明确 fund_personnel_by_code 产出时机 | 在 multi_t_backtest 中增加注释说明 |

---

## 6. 增强：批量 I/O + 并发 + 拆分缓存（2026-03-16 续）

### 6.1 批量 I/O 与并发

- 仅对 `personnel_dir` 中**存在**的 `{code}.csv` 做读取（`existent = [c for c in codes if (personnel_dir / f"{c}.csv").exists()]`）
- 使用 `ThreadPoolExecutor`（`_MAX_PERSONNEL_WORKERS=16`）并发读取人事文件
- 新增 `_check_personnel_one(path, window_start, window_end)` 供并发调用，`_compute_personnel_excluded()` 聚合结果

### 6.2 拆分缓存机制

| 缓存文件 | 说明 |
|----------|------|
| `eligible_base_{start}_{end}.csv` | c.1+a+b+e 结果，**不依赖** personnel_dir |
| `personnel_excluded_{start}_{end}.csv` | 规则 f 排除的基金编码（列：基金编码） |
| `eligible_fund_candidates.csv` | base − personnel_excluded，最终输出 |

- `multi_t_backtest`：在 `eligible_csv` 不存在时，优先尝试从 `base_path` + `personnel_excluded_path` 合并生成，避免调用 `run_prep_eligible_window`

### 6.3 变更文件

| 文件 | 变更 |
|------|------|
| `prep_eligible_window.py` | `_compute_personnel_excluded`、拆分缓存逻辑 |
| `multi_t_backtest.py` | 缓存合并分支 |

---

## 7. 项目元文件更新

- `myanalyser/docs/其他/multi_t_backtest_基金过滤追踪.md`：step1 条件包含规则 f，缓存结构已更新
- 本文件作为本次需求完整记录
