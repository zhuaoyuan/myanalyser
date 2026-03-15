# multi_t_backtest 时序图

以单个 T 日为例，描述 multi_t_backtest、backtest 组件、落盘缓存之间的流程与数据读写关系。

```mermaid
sequenceDiagram
    autonumber
    participant M as multi_t_backtest
    participant Cmp as compare_adjusted_nav
    participant Int as integrity_check
    participant Elig as prep_eligible_window
    participant Flt as filter_funds
    participant Sb as pipeline_scoreboard
    participant Load as load_fund_nav_data
    participant Run as run_backtest
    participant FS as filter_strategy
    participant SS as score_strategy
    participant PS as position_strategy

    box rgb(230,245,255) 缓存 / 落盘
        participant CacheC as cache/compare
        participant CacheI as cache/integrity
        participant CacheE as cache/prep_eligible
        participant CacheF as cache/filter
        participant CacheS as cache/scoreboard
        participant FE as fund_etl
        participant PW as prep_work_dir
        participant Out as artifacts
    end

    Note over M,Out: 循环开始：对每个 T 日

    rect rgb(255,248,220)
        Note over M,CacheC: Phase 1: Compare
        M->>CacheC: 检查 summary.csv + details/
        alt 缓存未命中
            M->>Cmp: compare_adjusted_nav_and_cum_return_window
            Cmp->>FE: 读 fund_adjusted_nav_by_code, fund_cum_return_by_code
            Cmp->>CacheC: 写 summary.csv, details/{code}.csv
            Cmp-->>M: summary_csv, detail_dir
        else 缓存命中
            CacheC-->>M: 返回已有 summary_csv, detail_dir
        end
    end

    rect rgb(255,248,220)
        Note over M,CacheI: Phase 2: Integrity
        M->>CacheI: 检查 integrity_summary + details/
        alt 缓存未命中
            M->>Int: _run_integrity_window
            Int->>FE: 读 fund_adjusted_nav_by_code, fund_overview.csv
            Int->>Int: 读 trade_dates.csv
            Int->>CacheI: 写 integrity_summary, details_{code}_*.csv
            Int-->>M: summary_csv, details_dir
        else 缓存命中
            CacheI-->>M: 返回已有路径
        end
    end

    rect rgb(255,248,220)
        Note over M,CacheE: Phase 3: Prep Eligible
        M->>CacheE: 检查 eligible_fund_candidates.csv
        alt 缓存未命中
            M->>Elig: run_prep_eligible_window
            Elig->>PW: 读 fund_cyrjg, fund_gmbd, fund_overview, fund_fee_filtered 等
            Elig->>CacheE: 写 eligible_fund_candidates.csv
            Elig-->>M: eligible_csv
        else 缓存命中
            CacheE-->>M: 返回 eligible_csv
        end
    end

    rect rgb(255,248,220)
        Note over M,CacheF: Phase 4: Filter
        M->>CacheF: 检查 filtered_fund_candidates.csv
        alt 缓存未命中
            M->>Flt: filter_funds_for_next_step
            Flt->>CacheC: 读 compare details/{code}.csv
            Flt->>CacheI: 读 integrity details/{code}_*.csv
            Flt->>CacheE: 读 eligible_fund_candidates.csv
            Flt->>FE: 读 fund_overview, fund_nav_by_code, fund_adjusted_nav_by_code
            Flt->>CacheF: 写 filtered_fund_candidates.csv
            M->>CacheF: 写 fund_purchase_for_step10_filtered.csv
            Flt-->>M: filter_csv
        else 缓存命中
            CacheF-->>M: 返回 filter_csv
        end
        M->>M: _read_allowed_codes → allowed_codes
    end

    rect rgb(255,248,220)
        Note over M,CacheS: Phase 5: Scoreboard
        M->>CacheS: 检查 fund_scoreboard_*.csv
        alt 缓存未命中
            M->>Sb: _run_scoreboard
            Sb->>CacheF: 读 fund_purchase_for_step10_filtered.csv
            Sb->>FE: 读 fund_overview, fund_personnel_by_code, fund_adjusted_nav_by_code
            Sb->>CacheS: 写 fund_scoreboard_{data_version}.csv
            Sb-->>M: scoreboard_csv
        else 缓存命中
            CacheS-->>M: 返回 scoreboard_csv
        end
    end

    rect rgb(220,255,220)
        Note over M,Run: Phase 6: Backtest（backtest 目录组件）
        M->>Load: load_fund_nav_data(nav_dir, allowed_codes)
        Load->>FE: 读 fund_adjusted_nav_by_code/{code}.csv（仅 allowed_codes）
        Load-->>M: BacktestData
        M->>Run: run_backtest(data, bundle, start_date, end_date, top_n, ...)
    end

    rect rgb(220,255,220)
        Note over Run,PS: run_backtest 内部：每个调仓日
        loop 每个 rebalance_date
            Run->>FS: filter_symbols(data, as_of_ts, universe)
            FS->>FS: 从 data.by_symbol 取 NAV，计算指标
            FS-->>Run: candidates
            Run->>SS: score(data, as_of_ts, candidates)
            SS->>SS: 从 data.by_symbol 取 NAV，compute_low_risk_debt_metrics
            SS-->>Run: scored
            Run->>PS: target_weights(scored, top_n)
            PS-->>Run: weights
        end
        Run->>Run: PyBroker strategy.backtest()
        Run-->>M: BacktestResult
    end

    M->>Out: write_reports(t_output_dir, result, data, run_config)
    Out->>Out: 写 summary.csv, detail.csv, report.md, curves
    M->>Out: 追加 multi_summary.csv 行

    Note over M,Out: 循环结束，下一个 T
```

---

## 参与对象说明

| 对象 | 类型 | 说明 |
|------|------|------|
| **multi_t_backtest** | 脚本 | 主流程编排，循环每个 T 日 |
| **compare_adjusted_nav** | v2.compare | 本地复权 vs 远程 cum_return 一致性比对 |
| **integrity_check** | check_trade_day_data_integrity | 交易日数据完整性检查 |
| **prep_eligible_window** | v2.filters | 从 prep_work 做 eligible 筛选 |
| **filter_funds** | v2.filters | 结合 compare+integrity 做规则 1–5 过滤 |
| **pipeline_scoreboard** | pipeline_scoreboard | 打分排序（本流程中 skip_sinks，仅写 CSV） |
| **load_fund_nav_data** | backtest.data | 加载 NAV 为 BacktestData |
| **run_backtest** | backtest.engine | 回测执行，调 PyBroker |
| **filter_strategy** | bundle 内 | PassThroughFilter / MostStableFilterStrategy |
| **score_strategy** | bundle 内 | LowRiskDebtScoreStrategy |
| **position_strategy** | bundle 内 | EqualWeightPosition |
| **cache/*** | 磁盘 | compare/integrity/prep_eligible/filter/scoreboard 缓存目录 |
| **fund_etl** | 磁盘 | run_full_pipeline 产出的 L1 数据 |
| **prep_work_dir** | 磁盘 | prep_data_workflow 产出的 cyrjg/gmbd/overview/fee 等 |
| **artifacts** | 磁盘 | 回测报告、multi_summary.csv |

---

## 数据传递关系简图

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                    fund_etl (L1)                          │
                    │  fund_adjusted_nav_by_code, fund_cum_return_by_code,      │
                    │  fund_overview, fund_nav_by_code, fund_personnel_by_code  │
                    └──────────────────────────────────────────────────────────┘
                         │           │           │           │           │
                         ▼           ▼           ▼           ▼           ▼
              compare   integrity  filter    scoreboard   load_fund_nav_data
                         │           │           │
                         ▼           ▼           ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │                    prep_work_dir                         │
                    │  fund_cyrjg, fund_gmbd, fund_overview, fund_fee_filtered  │
                    └──────────────────────────────────────────────────────────┘
                         │
                         ▼
                    prep_eligible
                         │
                         ▼
                    filter (allowed_codes) ──────────────────────────────────────┐
                         │                                                      │
                         ▼                                                      ▼
                    load_fund_nav_data(allowed_codes) ─────► BacktestData ──► run_backtest
                                                                                  │
                                                                                  ▼
                                                                            filter_strategy
                                                                            score_strategy
                                                                            position_strategy
                                                                                  │
                                                                                  ▼
                                                                            PyBroker
```

---

## 缓存目录结构

```
data/versions/{run_id}/cache/v2/
├── compare/{ruleset_version}/{start}_{end}/
│   ├── summary.csv
│   └── details/{code}.csv
├── integrity/{ruleset_version}/{start}_{end}/
│   ├── trade_day_integrity_summary_{start}_{end}.csv
│   └── details_{start}_{end}/{code}_{start}_{end}.csv
├── prep_eligible/{ruleset_version}/{start}_{end}/
│   └── eligible_fund_candidates.csv
├── filter/{ruleset_version}/{start}_{end}/
│   ├── filtered_fund_candidates.csv
│   └── fund_purchase_for_step10_filtered.csv
└── scoreboard/{ruleset_version}/{as_of_date}/{start}_{end}/
    └── fund_scoreboard_{run_id}_{ruleset}_{as_of}.csv

artifacts/backtest_multi/{run_id}/{ruleset_version}/
├── {as_of_date}/          # 每个 T 日
│   ├── summary.csv
│   ├── detail.csv
│   ├── report.md
│   └── ...
└── multi_summary.csv
```
