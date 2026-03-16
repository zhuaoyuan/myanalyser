# 需求日志：multi_t_backtest 基金类型筛选参数

**日期**: 2026-03-16  
**需求描述**: 为 multi_t_backtest 增加 `--fund-types` 参数，支持从 prep-work-dir 的 fund_fee_filtered.csv 中按基金类型筛选。

---

## 1. 需求与实现

### 1.1 新增参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--fund-types` | str | 逗号分隔的基金类型，可多选；可选值见 `prep-work-dir/fund_fee_filtered.csv` 的「类型」列 |

**可选值示例**（来自 fund_fee_filtered.csv 类型枚举）：
- A类180天、A类30天、A类365天、A类60天、A类730天
- C类30天、C类60天
- 场内交易

### 1.2 使用方式

在 pipeline 中，从 `prep-work-dir/fund_fee_filtered.csv` 筛选出传入类型列表的基金，与 eligible 取交集后作为 filter 步骤的输入。

**示例**：
```bash
# 仅 A 类 730 天
--fund-types "A类730天"

# A 类与 C 类 30 天
--fund-types "A类730天,C类30天"
```

不传 `--fund-types` 时保持原行为，不做类型过滤。

### 1.3 缓存策略

- `filter_dir` 路径包含 `fund_types` 的 hash 后缀，不同 `--fund-types` 组合使用独立缓存
- `type_allowed_codes` 在 T 循环外加载一次，避免重复读取 fund_fee_filtered.csv

---

## 2. 变更文件列表

| 文件 | 变更类型 |
|------|----------|
| `myanalyser/tools/v2/multi_t_backtest.py` | 修改 |

---

## 3. 实现要点

- `_parse_fund_types(raw)`：解析逗号分隔类型列表，空值返回 `[]`
- `_load_type_filtered_codes(prep_work_dir, fund_types)`：从 fund_fee_filtered.csv 筛选指定类型的基金编码
- `_build_purchase_csv_for_filter(...)`：按类型与 eligible 取交集，返回供 filter 使用的 purchase CSV 路径
- `code_col` 显式检查：eligible CSV 至少存在「基金编码」或「基金代码」之一，否则抛出含列名的 `ValueError`

---

## 4. Code Review 采纳

| 意见 | 处理 |
|------|------|
| code_col 回退逻辑有漏洞（两列都不存在时 KeyError） | 使用 if/elif/else 显式检查并抛出 ValueError |
| type_allowed_codes 循环内重复读取 | 在 T 循环外加载一次 |
| _parse_fund_types 重复 str(raw) | 先 `s = str(raw).strip() if raw is not None else ""` 再解析 |
| fund_types 分支约 30 行可读性一般 | 抽取为 `_build_purchase_csv_for_filter` 独立函数 |
| MD5 8 位碰撞风险 | 保持 MD5（review 认为可接受） |

---

## 5. 验收

- 依赖 fund_fee_filtered.csv 存在且含「类型」「基金编码」列
- 指定类型在 CSV 中无匹配时抛 ValueError
- eligible 与类型取交后为空时抛 ValueError

---

## 6. 提交记录

- v0316.18 (038a24c)：基金类型筛选参数及 review 相关修改
