#!/bin/bash

# 1. 获取当前月日 (例如 0311)
DATE=$(date +%m%d)

# 2. 获取当天已经存在的、符合格式的 tag 或提交记录数量，以此生成序列号
# 这里我们通过 git log 统计当天生成的 vMMDD 开头的提交信息数量
SEQ=$(git log --oneline --since="midnight" | grep -c "v$DATE" || echo 0)

# 3. 序列号自增 1 (如果是当天第一次提交，SEQ 为 0+1=1)
# 使用 printf 确保序号是两位数，例如 01, 10
NEXT_SEQ=$(printf "%02d" $((SEQ + 1)))

# 4. 拼接最终的 Commit Message
COMMIT_MSG="v${DATE}.${NEXT_SEQ}"

echo "🚀 正在提交，版本号为: $COMMIT_MSG"

# 5. 执行你的命令组合
git add . && \
git commit -m "$COMMIT_MSG" && \
./tools/git/gitdiff.sh && \
git push origin main