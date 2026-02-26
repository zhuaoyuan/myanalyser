# myanalyser 项目说明

本项目用于面向自用需求的基金数据采集、复权净值计算、收益率一致性校验，以及评分榜单生成/入库。

## 目录结构（重构后）

```text
myanalyser/
  src/        # 业务代码（CLI + 核心逻辑）
  tests/      # 单测/集成测试
  data/
    common/   # 公共数据（如交易日历）
    versions/ # 按 run_id 存放每次跑数版本
    samples/  # 小样本数据
  artifacts/  # 预留目录（运行产物）
  docs/       # 文档
  tools/      # 一次性脚本
```

## 数据目录约定

- `run_id` 默认格式：`YYYYMMDD_HHMMSS`
- 可追加描述后缀：`YYYYMMDD_HHMMSS_desc`
- 每次跑数建议独立目录：`data/versions/{run_id}/`
- `fund_etl` 结果目录：`data/versions/{run_id}/fund_etl`
- 错误日志目录（与结果分离）：`data/versions/{run_id}/logs`
- 公共交易日历：`data/common/trade_dates.csv`

## 核心脚本

- `src/fund_etl.py`：AkShare 拉数（step1~step7 + retry）
- `src/adjusted_nav_tool.py`：复权净值计算
- `src/compare_adjusted_nav_and_cum_return.py`：复权收益率一致性比对
- `src/check_trade_day_data_integrity.py`：交易日完整性检查
- `src/pipeline_scoreboard.py`：评分榜单计算、导出与入库（支持 `--skip-sinks`）
- `src/backtest_portfolio.py`：按规则回测组合

## 常用命令

```bash
# 1) 拉取原始数据（自动生成 run_id）
python src/fund_etl.py --mode all

# 2) 拉取原始数据（指定 run_id + 后缀）
python src/fund_etl.py --mode all --run-id 20260226_210000_test
# 或
python src/fund_etl.py --mode all --run-id-suffix smoke
```

```bash
# 3) 计算复权净值（示例 run_id）
RUN_ID=20260226_210000_smoke
python src/adjusted_nav_tool.py \
  --nav-dir data/versions/${RUN_ID}/fund_etl/fund_nav_by_code \
  --bonus-dir data/versions/${RUN_ID}/fund_etl/fund_bonus_by_code \
  --split-dir data/versions/${RUN_ID}/fund_etl/fund_split_by_code \
  --output-dir data/versions/${RUN_ID}/fund_etl/fund_adjusted_nav_by_code \
  --fail-log data/versions/${RUN_ID}/logs/failed_adjusted_nav.jsonl
```

```bash
# 4) 比对复权收益率与累计收益率
RUN_ID=20260226_210000_smoke
python src/compare_adjusted_nav_and_cum_return.py \
  --base-dir data/versions/${RUN_ID}/fund_etl \
  --output-dir data/versions/${RUN_ID}/fund_etl/fund_return_compare \
  --error-log data/versions/${RUN_ID}/logs/compare_adjusted_nav_cum_return_errors.jsonl
```

```bash
# 5) 交易日完整性检查
RUN_ID=20260226_210000_smoke
python src/check_trade_day_data_integrity.py \
  --base-dir data/versions/${RUN_ID}/fund_etl \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --trade-dates-csv data/common/trade_dates.csv
```

```bash
# 6) 榜单计算（仅导出 CSV，不入库）
RUN_ID=20260226_210000_smoke
python src/pipeline_scoreboard.py \
  --purchase-csv data/versions/${RUN_ID}/fund_etl/fund_purchase.csv \
  --overview-csv data/versions/${RUN_ID}/fund_etl/fund_overview.csv \
  --personnel-dir data/versions/${RUN_ID}/fund_etl/fund_personnel_by_code \
  --nav-dir data/versions/${RUN_ID}/fund_etl/fund_adjusted_nav_by_code \
  --output-dir artifacts/scoreboard_${RUN_ID} \
  --data-version ${RUN_ID} \
  --as-of-date 2026-02-26 \
  --skip-sinks
```

## 依赖

```bash
pip install akshare pandas numpy pymysql
```

## 统一验收命令

项目提供统一验收脚本 `tools/verify.sh`，用于在新目录结构下执行完整回归闭环：

- 单测回归（`tests/test_*.py`）
- 核心 CLI smoke（`fund_etl`、`pipeline_scoreboard`、`backtest_portfolio`、`compare_adjusted_nav_and_cum_return`、`check_trade_day_data_integrity`）
- 数据完整性检查（真实调用 `src/check_trade_day_data_integrity.py`）

```bash
cd /Users/zhuaoyuan/cursor-workspace/finance/myanalyser
source /Users/zhuaoyuan/cursor-workspace/finance/myanalyser/.venv312/bin/activate
bash tools/verify.sh
```

如需固定本次验收目录名，可显式传入 `RUN_ID`：

```bash
RUN_ID=20260226_220000_verify bash tools/verify.sh
```
