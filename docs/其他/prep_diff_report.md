# prep_data_workflow 两次运行产物校对报告

**Run1:** `myanalyser/tmp/1`  
**Run2:** `myanalyser/tmp/2`  
**执行参数：** 一致

---

## 1. 总体结论

- **最终输出 `prep_result_*.csv` 完全一致**：2281 行，0 单元格差异
- 中间产物在内容上基本一致，仅存在一处可解释的中间态差异
- **未发现远程查询导致的实质性数据偏差**

---

## 2. 文件级比对

| 文件 | Run1 行数 | Run2 行数 | 内容差异 | 说明 |
|------|-----------|-----------|----------|------|
| prep_work/fund_purchase.csv | 25942 | 25942 | 无 | 输入一致 |
| prep_work/fund_gmbd.csv | 619348 | 619348 | 无 | 规模历史，Key(基金代码,日期) 一致，0 内容差 |
| prep_work/fund_fee_structured.csv | 116340 | 116340 | 无 | 费率结构，排序后 0 差异 |
| prep_work/fund_fee_filtered.csv | 17181 | 17181 | 无 | 本地分类 |
| prep_work/fund_overview.csv | 25964 | 25964 | 无 | 按基金代码排序后 0 差异（此前 66 为行序导致的假阳性） |
| prep_work/_tmp_gmbd_purchase.csv | 259 | 272 | 行数差 13 | 见下文 |
| prep_work/_tmp_fee_purchase.csv | 21860 | 21860 | 无 | 待抓取费率列表一致 |
| **prep_result_*.csv** | **2281** | **2281** | **无** | 最终筛选结果完全一致 |

---

## 3. 差异说明

### 3.1 _tmp_gmbd_purchase 行数差异（260 vs 273）

- `_tmp_gmbd_purchase.csv` 是**待抓取规模**的基金代码临时列表
- Run2 比 Run1 多 13 个待抓取代码：  
  `022471, 025218, 025219, 026343, 026344, 026396, 026397, 026430, 026431, 026558, 026559, 026732, 026733`

**可能原因：**

1. 两次运行时 `--gmbd-csv` 指向的已有规模数据不同（如运行间隔内文件被更新）
2. 增量逻辑 `to_fetch = codes - done_codes` 中，`done_codes` 来源不同

**影响：**

- `fund_gmbd.csv` 行数与内容完全一致（619348 行，0 差异）
- 说明无论 to_fetch 列表如何，最终合并后的规模数据一致，对后续筛选无影响

---

## 4. 远程查询相关结论

| 数据源 | 脚本/步骤 | 一致性 | 备注 |
|--------|-----------|--------|------|
| 持有人比例 (cyrjg) | fetch_fund_cyrjg | 未单独比对 | 通常由 --cyrjg-csv 直接提供 |
| 规模 (gmbd) | fetch_fund_gmbd | 完全一致 | 最终 fund_gmbd 0 差异 |
| 费率 (fee) | fetch_fund_fee | 完全一致 | fund_fee_structured 0 差异 |
| 基金详情 (overview) | run_step2_overview | 完全一致 | 按基金代码排序后 0 差异 |

**结论：** 在当前两次运行中，远程查询未引入可观测的偏差，最终 `prep_result` 与中间产物在内容上均一致。
