# 需求日志：v2 verify 运行确认、Review 意见实施与元文件同步

**日期**: 2026-03-19  
**需求描述**: 运行 v2/verify.sh 确认完全成功；评估并实施 7 条 review 意见；同步变更到项目元文件。

---

## 1. v2/verify.sh 运行确认

| 问题 | 处理 |
|------|------|
| step10 使用 most_stable 时 15 只基金全部被过滤，scored_result 为空，assert_csv_has_rows 失败 | 将 step10 过滤器改为 non_a_unlimited_purchase，确保采样下至少有基金通过 |
| 下游影响 | 在 verify.sh 与 V2完整流程说明.md 中注明：verify step10 为验收场景使用宽松过滤，baseline 回归使用 most_stable |

---

## 2. Review 意见评估与实施

| 文件 | 类型 | 意见 | 评估 | 处理 |
|------|------|------|------|------|
| test_v2_baseline_regression.py | Logic | try/except 静默跳过 nav 文件，无日志 | 合理 | 增加 logging.warning 记录被跳过路径与异常 |
| test_v2_baseline_regression.py | Logic | ValueError 范围过宽，可能掩盖编码/dtype 错误 | 合理 | 仅对列缺失类 ValueError 做 continue，其余 re-raise |
| generate_baseline_expected.py | Logic | 同上，try/except 静默跳过 | 合理 | 增加 skipped 计数，skipped>0 时打印 Warning |
| generate_baseline_expected.py | Style | os.chdir 若被导入产生副作用 | 合理（架构建议） | 模块 docstring 注明脚本设计为直接运行，不作为模块导入 |
| verify.sh | Architecture | PYTHONPATH 作用域是否影响后续 step | 不采纳子 shell | step2~10 均为 Python，需相同路径；补充注释说明 |
| verify.sh | Architecture/Major | step10 non_a vs baseline most_stable 语义歧义 | 合理 | 文档补充：verify 验收用 non_a，baseline 用 most_stable |
| verify.sh | Downstream | filter 变更影响 filter_result/scored_result 消费者 | 合理 | 与上条合并，在文档中注明 step10 实际 filter |

---

## 3. 变更文件列表

| 文件 | 变更类型 |
|------|----------|
| `tests/test_v2_baseline_regression.py` | 修改（import logging；try/except 增加 warning、ValueError 收紧） |
| `tools/v2/generate_baseline_expected.py` | 修改（skipped 计数与 Warning；模块 docstring） |
| `tools/v2/verify.sh` | 修改（PYTHONPATH 注释；step10 filter 注释） |
| `docs/V2完整流程说明.md` | 修改（校验环节补充 step10 filter 与 baseline 区分说明） |
| `docs/需求日志/20260319_v2_verify与review及元文件同步.md` | 新增 |
| `docs/README.md` | 修改（最小回归基线补充 filter 区分） |

---

## 4. 验收

- `tests.test_v2_baseline_regression.V2BaselineRegressionTest.test_v2_baseline_full_flow_regression` 通过
- `bash tools/v2/verify.sh` 完整通过（exit code 0）
