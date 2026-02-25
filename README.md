# myanalyser 项目说明

本项目用于基金数据采集、复权净值计算、收益率一致性校验，以及最终评分榜单入库。选取数据范围和处理方法都从个人用途出发，有很大的个人倾向。

## 为什么要做这个项目？

在常见的个人资产管理需求下，需要对国内开放式公募基金的数据采集、整理、以及进一步分析。
东方财富、同花顺等平台是公开的数据提供者，但产品设计无法满足定制化的需求。
akshare是最常用的此类数据接口api集成者，对平台api做了标准化封装。
本项目在akshare的数据上，进行进一步定制化处理，例如：
- 原始数据中的单位净值无法直接用于计算收益率，累计净值未考虑分红复投，平台累计收益率数据为复权后数据，但步长太长，无法支持精确的分析或回测。因此需要自行计算天级复权净值。
- 据称部分平台数据有偏差，因此本地尽可能对数据再做交叉比对，剔除问题数据。
- 按需支持自定义的筛选和排序方法。


## 未处理的问题
- 尚未支持货币基金的净值数据处理。暂时不是关注重点，后续补上。
- 历史数据接口总是按基金编码全量获取，而没有增量方案，单次更新数据代价较大。当前基于基金投资的特点，暂时按月为周期执行一次全量分析。
- 部分基金的历史单位净值数据步长大于一天，可能跳过分红日/拆分日，而导致复权净值无法计算。可以关注近3年或近5年数据，减小此类影响面。近期仍存在此类异常的基金则认为数据不可靠，先剔除。
- 部分基金的历史分红数据不符合预期，导致复权净值计算错误。目前通过本地计算的复权净值和平台计算的累计收益率比对发现并排查。

## 脚本总览

1. `fund_etl.py`：从 AkShare 拉取原始基金数据并落盘（按步骤执行，支持失败重试）。
2. `adjusted_nav_tool.py`：基于单位净值 + 分红 + 拆分计算复权净值。
3. `compare_adjusted_nav_and_cum_return.py`：将本地复权净值推导收益率与远端累计收益率对比，产出偏差报告。
4. `pipeline_scoreboard.py`：计算绩效指标、生成榜单 CSV，并写入 MySQL/ClickHouse。

## 目录约定

默认 ETL 输出目录是 `myanalyser/data/fund_etl`，关键子目录/文件：

- `fund_purchase.csv`
- `fund_overview.csv`
- `fund_nav_by_code/*.csv`
- `fund_bonus_by_code/*.csv`
- `fund_split_by_code/*.csv`
- `fund_personnel_by_code/*.csv`
- `fund_cum_return_by_code/*.csv`
- `failed_*.jsonl`

## 1) fund_etl.py

### 作用

从 AkShare 拉取基金基础与行情相关数据，拆分为 7 个步骤：

- `step1`：基金申购列表（`fund_purchase.csv`）
- `step2`：基金概况（`fund_overview.csv`）
- `step3`：单位净值走势（`fund_nav_by_code`）
- `step4`：分红送配详情（`fund_bonus_by_code`）
- `step5`：拆分详情（`fund_split_by_code`）
- `step6`：基金人事公告（`fund_personnel_by_code`）
- `step7`：累计收益率走势（`fund_cum_return_by_code`）

支持 `retry-*` 模式按失败日志自动重跑。

### 常用参数

- `--base-dir`：输出目录，默认 `myanalyser/data/fund_etl`
- `--mode`：`all/step1...step7/verify/retry-*`
- `--max-retries`：单请求重试次数，默认 `3`
- `--retry-sleep`：重试退避基数（秒），默认 `1.0`
- `--max-workers`：并发数（主要用于 step2），默认 `8`
- `--progress-interval`：进度打印间隔（秒），默认 `5.0`

### 调用示例

```bash
# 全量执行（含接口列验证）
python myanalyser/fund_etl.py --mode all --base-dir myanalyser/data/fund_etl

# 仅跑单位净值
python myanalyser/fund_etl.py --mode step3 --base-dir myanalyser/data/fund_etl

# 仅重试累计收益率失败基金
python myanalyser/fund_etl.py --mode retry-cum-return --base-dir myanalyser/data/fund_etl
```

## 2) adjusted_nav_tool.py

### 作用

按基金逐只读取：

- `fund_nav_by_code/*.csv`
- `fund_bonus_by_code/*.csv`
- `fund_split_by_code/*.csv`

计算 `复权净值` 与 `cumulative_factor`，输出到 `fund_adjusted_nav_by_code/*.csv`。

如果分红/拆分日期在净值表中缺失，默认报错；可通过阈值日期放宽历史缺失。

### 常用参数

- `--nav-dir`：单位净值目录（必填）
- `--bonus-dir`：分红目录（必填）
- `--split-dir`：拆分目录（必填）
- `--output-dir`：复权净值输出目录（必填）
- `--codes`：仅处理指定基金代码，可多值
- `--progress-interval-seconds`：进度打印间隔，默认 `5`
- `--allow-missing-event-until`：允许忽略该日期及以前缺失事件（`YYYY-MM-DD`）

### 调用示例

```bash
# 全量计算复权净值
python myanalyser/adjusted_nav_tool.py \
  --nav-dir myanalyser/data/fund_etl/fund_nav_by_code \
  --bonus-dir myanalyser/data/fund_etl/fund_bonus_by_code \
  --split-dir myanalyser/data/fund_etl/fund_split_by_code \
  --output-dir myanalyser/data/fund_etl/fund_adjusted_nav_by_code

# 仅计算指定基金，且放宽历史缺失日期
python myanalyser/adjusted_nav_tool.py \
  --nav-dir myanalyser/data/fund_etl/fund_nav_by_code \
  --bonus-dir myanalyser/data/fund_etl/fund_bonus_by_code \
  --split-dir myanalyser/data/fund_etl/fund_split_by_code \
  --output-dir myanalyser/data/fund_etl/fund_adjusted_nav_by_code \
  --codes 000001 163402 \
  --allow-missing-event-until 2015-12-31
```

## 3) compare_adjusted_nav_and_cum_return.py

### 作用

对比：

- 本地复权净值（`fund_adjusted_nav_by_code`）推导收益率
- 远端累计收益率（`fund_cum_return_by_code`）

输出：

- `summary.csv`：每只基金偏差分桶占比
- `details/{code}.csv`：基金级明细
- `errors.jsonl`：缺文件/解析失败/无公共日期等错误

默认输出目录：`{base-dir}/fund_return_compare`。

### 参数

- `--base-dir`：必须包含 `fund_adjusted_nav_by_code` 和 `fund_cum_return_by_code`
- `--output-dir`：可选，覆盖默认输出目录

### 调用示例

```bash
python myanalyser/compare_adjusted_nav_and_cum_return.py \
  --base-dir myanalyser/data/fund_etl
```

## 4) pipeline_scoreboard.py

### 作用

读取基金维度信息与复权净值，计算多维绩效指标（年化、胜率、波动、回撤、夏普、卡玛等），输出榜单并落 MySQL/ClickHouse：

- 导出 CSV：
  - `fund_scoreboard_{data_version}.csv`
  - `fund_exclusion_detail_{data_version}.csv`
  - `fund_exclusion_summary_{data_version}.csv`
- 写入 MySQL：维表与剔除明细/汇总
- 写入 ClickHouse：日净值、周期收益、指标快照、榜单快照

### 常用参数

- 输入参数（必填）：
  - `--purchase-csv`
  - `--overview-csv`
  - `--personnel-dir`
  - `--nav-dir`（应指向复权净值目录）
  - `--output-dir`
  - `--data-version`
  - `--as-of-date`（如 `2026-02-24`）
- 计算控制：
  - `--stale-max-days`（默认 `2`）
  - `--code-limit`（调试时限制基金数）
- 建表：
  - `--apply-ddl`
  - `--mysql-ddl`
  - `--clickhouse-ddl`
- 数据库连接：
  - `--mysql-*`
  - `--clickhouse-*`
  - `--clickhouse-container`

### 调用示例

```bash
python myanalyser/pipeline_scoreboard.py \
  --purchase-csv myanalyser/data/fund_etl/fund_purchase.csv \
  --overview-csv myanalyser/data/fund_etl/fund_overview.csv \
  --personnel-dir myanalyser/data/fund_etl/fund_personnel_by_code \
  --nav-dir myanalyser/data/fund_etl/fund_adjusted_nav_by_code \
  --output-dir myanalyser/output \
  --data-version 202602 \
  --as-of-date 2026-02-24 \
  --mysql-password your_strong_password
```

如需首次初始化表结构：

```bash
python myanalyser/pipeline_scoreboard.py \
  --purchase-csv myanalyser/data/fund_etl/fund_purchase.csv \
  --overview-csv myanalyser/data/fund_etl/fund_overview.csv \
  --personnel-dir myanalyser/data/fund_etl/fund_personnel_by_code \
  --nav-dir myanalyser/data/fund_etl/fund_adjusted_nav_by_code \
  --output-dir myanalyser/output \
  --data-version 202602 \
  --as-of-date 2026-02-24 \
  --apply-ddl \
  --mysql-password your_strong_password
```

## 推荐执行顺序

```bash
# 1) 拉取原始数据
python myanalyser/fund_etl.py --mode all --base-dir myanalyser/data/fund_etl

# 2) 计算复权净值
python myanalyser/adjusted_nav_tool.py \
  --nav-dir myanalyser/data/fund_etl/fund_nav_by_code \
  --bonus-dir myanalyser/data/fund_etl/fund_bonus_by_code \
  --split-dir myanalyser/data/fund_etl/fund_split_by_code \
  --output-dir myanalyser/data/fund_etl/fund_adjusted_nav_by_code

# 3) （可选）比对复权净值收益率与累计收益率
python myanalyser/compare_adjusted_nav_and_cum_return.py \
  --base-dir myanalyser/data/fund_etl

# 4) 生成榜单并入库
python myanalyser/pipeline_scoreboard.py \
  --purchase-csv myanalyser/data/fund_etl/fund_purchase.csv \
  --overview-csv myanalyser/data/fund_etl/fund_overview.csv \
  --personnel-dir myanalyser/data/fund_etl/fund_personnel_by_code \
  --nav-dir myanalyser/data/fund_etl/fund_adjusted_nav_by_code \
  --output-dir myanalyser/output \
  --data-version 202602 \
  --as-of-date 2026-02-24 \
  --mysql-password your_strong_password
```

## 依赖

```bash
pip install akshare pandas numpy pymysql
```

如果需要写入 ClickHouse，请确保本地可执行 `docker exec <container> clickhouse-client`。
