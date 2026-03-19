#!/bin/bash

# 1. 确保目录存在
mkdir -p tmp

# 2. 获取变量
# 获取最新的 Commit ID (短哈希，前7位)
COMMIT_ID=$(git rev-parse --short HEAD)

# 获取当前时间戳，格式为：YYYYMMDD_HHMM (例如：20260311_0945)
TIMESTAMP=$(date +"%Y%m%d_%H%M")

# 定义文件名
FILENAME="gitdiff_tmp/temp_diff_${TIMESTAMP}_${COMMIT_ID}.diff"

# 3. 检查提交次数
COMMIT_COUNT=$(git rev-list --count HEAD 2>/dev/null)
if [ -z "$COMMIT_COUNT" ] || [ "$COMMIT_COUNT" -lt 2 ]; then
    echo "错误: 提交次数不足（需要至少2次），无法生成 diff。"
    exit 1
fi

# 4. 执行 diff 并保存
git diff HEAD~1..HEAD > "$FILENAME"

# 5. 反馈结果
if [ $? -eq 0 ]; then
    echo "成功生成文件: $FILENAME"
    # 打印一下文件大小确认非空
    ls -lh "$FILENAME"
else
    echo "生成失败。"
    exit 1
fi