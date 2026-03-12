#!/bin/bash

# 1. 获取当前月日 (例如 0311)
DATE=$(date +%m%d)

# 2. 获取当天提交数量 (确保只抓取以 vMMDD 开头的 commit)
# 加上 10# 前缀可以强制 Bash 按十进制解析
SEQ_RAW=$(git log --oneline --since="midnight" --grep="v$DATE" | wc -l)

# 3. 序列号自增 1
# 10#$SEQ_RAW 的意思是：无论 SEQ_RAW 是什么，都按 10 进制处理
NEXT_SEQ=$(printf "%02d" $((10#$SEQ_RAW + 1)))

# 4. 拼接最终的 Commit Message
COMMIT_MSG="v${DATE}.${NEXT_SEQ}"

echo "🚀 正在提交，版本号为: $COMMIT_MSG"

# 5. 执行你的命令组合
git add . && \
git commit -m "$COMMIT_MSG" && \
./tools/gitdiff.sh && \
git push origin main