# myanalyser - Claude Code 项目配置

## 项目概述

本项目用于面向自用需求的基金数据采集、复权净值计算、收益率一致性校验，以及评分榜单生成/入库。

**技术栈**: Python 3.12.12, pandas, pytest, PyBroker

## 环境配置

- **虚拟环境**: `myanalyser/.venv312`
- **激活命令**: `source myanalyser/.venv312/bin/activate`
- **Python 版本**: 3.12.12（通过 `.python-version` 锁定，pyenv 自动读取）
- **依赖锁定**: `pip install -r myanalyser/requirements-lock.txt`

## 协作流程 (强制)

需求 → 设计与风险讨论 → AI实现 → 验收证据 → 开发者确认 → 更新元文件

**关键约束**:
- 代码/脚本/数据/文档变更必须走完整流程
- 设计阶段需给出"验收清单"并经确认后才能实现
- 实现必须包含：变更文件列表、测试点、本地命令、输出样例
- 验收需通过 `bash tools/v2/verify.sh` 跑完回归

## 常用命令

```bash
# 激活环境
cd /Users/zhuaoyuan/cursor-workspace/finance
source myanalyser/.venv312/bin/activate

# V2 回归验证
bash myanalyser/tools/v2/verify.sh

# 全流程运行流程见 `myanalyser/docs/V2完整流程说明.md`

```

## 关键文档

- `myanalyser/docs/项目通用约束.md` - 项目原则性约束（每次会话默认读取）
- `myanalyser/docs/README.md` - 基本功能和命令
- `myanalyser/docs/V2完整流程说明.md` - V2 流程详细说明

## 验收标准

- 单测回归（`tests/test_*.py`）
- 核心 CLI smoke 测试
- V2 基线回归（step5~10 与 expected 对比）
- 统一验收命令: `bash tools/v2/verify.sh`

## 目录结构

```
myanalyser/
  src/        # 业务代码（CLI + 核心逻辑）
  tests/      # 单测/集成测试
  data/       # 数据目录
    common/   # 公共数据（交易日历等）
    versions/ # 按 run_id 存放每次跑数版本
  artifacts/  # 运行产物
  docs/       # 文档
  tools/      # 工具脚本
```

## 日期协议

- v2 系列日期与区间协议见 `docs/参考/v2日期与区间协议约定.md`
- lookback 统一为交易日口径：1年 = 243 交易日
- 日期区间为双闭 [start, end]
- run_id 默认格式：`YYYYMMDD_HHMMSS`
