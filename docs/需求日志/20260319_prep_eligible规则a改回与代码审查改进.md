# 需求日志：prep_eligible 规则 a 改回与代码审查改进

**日期**: 2026-03-19  
**需求描述**: 将 prep_eligible_window 规则 a（机构持仓）逻辑改回 [max(成立+2年, start_ts), end_ts]，并落实四项代码审查改进建议。

---

## 1. 规则 a（机构持仓）逻辑回退

| 项目 | 内容 |
|------|------|
| **原状态**（20260316 调整后） | [成立+2年, 窗口 end_date] 内检查机构持仓 |
| **本次修改** | 改回 [max(成立+2年, start_ts), end_ts] 内检查 |
| **实现** | `a_df` 按 [start_ts, end_ts] 裁剪；每只基金 cutoff = max(成立+2年, start_ts)，仅检查 [cutoff, end_ts] 内连续两次 > 60% |

**实现**：`myanalyser/src/v2/filters/prep_eligible_window.py`
- `a_df = a_df[(a_df[date_col] >= start_ts) & (a_df[date_col] <= end_ts)]`
- `cutoff = max(inc + two_years, start_ts)`
- `sub = grp[grp[date_col] >= cutoff]`

---

## 2. 代码审查改进

针对规则 a 相关逻辑，落实四项改进：

| 审查项 | 处理 |
|--------|------|
| **cutoff > end_ts 空集短路** | `if cutoff > end_ts: continue`，避免对必然为空的 sub 做 groupby/迭代 |
| **timezone 约定** | 在规则 a 注释中约定：a_df[date_col]、start_ts、end_ts 均为 naive datetime |
| **逻辑变更需清缓存** | 在模块 docstring 缓存策略中补充：规则 a/b/e 等筛选逻辑变更后，需手动删除 cache 目录 |
| **inc_by_code 缺失** | `if inc is None: continue` 处增加注释：overview 无成立日期时跳过，规则 e 另行处理 |

---

## 3. 变更文件列表

| 文件 | 变更类型 |
|------|----------|
| `myanalyser/src/v2/filters/prep_eligible_window.py` | 修改（规则 a 逻辑、短路、注释、docstring） |

---

## 4. 验收

- 已有 test_v2_phase0_2_wide_store_narrow_use 中 `test_rule_a_consecutive_over_60_excluded` 等单测覆盖规则 a
- 规则逻辑变更后，需手动删除相关 cache 目录使 base 缓存失效（见 docstring）
