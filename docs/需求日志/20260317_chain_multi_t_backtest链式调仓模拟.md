# 需求日志：chain_multi_t_backtest 链式调仓模拟

**日期**: 2026-03-17  
**需求描述**: 基于 multi_t_backtest 产物，新增工具脚本 chain_multi_t_backtest，用前 T 期末市值作为后 T 期初市值，模拟调仓后的长期效果。

## 需求要点

1. **输入**: `artifacts/backtest_multi/{run_id}/{ruleset_version}/` 下各 T 日子目录（如 2015-02-27、2015-04-29 等）
2. **链式逻辑**: 
   - 前 T 期末市值 → 后 T 期初市值（缩放 equity）
   - 前提：前 T 期末日期 <= 后 T 期初日期
   - 无交易的 T 跳过（orders 中无 buy）
3. **输出**: 与单 T 回测相同格式的产物，放在 `chain/` 子目录：
   - summary.csv、equity_curve.csv、period_detail.csv、orders.csv、positions_flat.csv、backtest_report.md

## 变更文件

| 路径 | 操作 |
|------|------|
| `myanalyser/tools/v2/chain_multi_t_backtest.py` | 新增 |
| `myanalyser/tests/test_chain_multi_t_backtest.py` | 新增 |
| `myanalyser/docs/README.md` | 更新：补充 chain 工具说明 |

## 用法

```bash
python myanalyser/tools/v2/chain_multi_t_backtest.py \
  --output-root myanalyser/artifacts/backtest_multi/20260315_123456_full_run_v2/20260316_2m \
  [--chain-output-dir chain]
```

## 验收

- 5 个单测通过：`pytest myanalyser/tests/test_chain_multi_t_backtest.py -v`
- 在 20260316_2m 产物上运行，生成 chain/ 目录，equity 期初 100000、期末约 16 万（约 60% 总收益）
