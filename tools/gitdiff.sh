#!/bin/bash

# 1. 确保目标目录存在
mkdir -p tmp

# 2. 检查 git 提交记录是否至少有两次
COMMIT_COUNT=$(git rev-list --count HEAD 2>/dev/null)

if [ -z "$COMMIT_COUNT" ]; then
    echo "错误: 当前目录似乎不是一个 Git 仓库。"
    exit 1
fi

if [ "$COMMIT_COUNT" -lt 2 ]; then
    echo "提示: 当前分支提交次数少于 2 次，无法进行对比。"
    # 如果只有一次提交，可以选择输出该提交本身的内容
    # git show HEAD > tmp/temp_diff.diff
    exit 1
fi

# 3. 执行 diff
# HEAD   代表最新一次 commit (v0311.2)
# HEAD~1 代表倒数第二次 commit (v0311)
git diff HEAD~1..HEAD > tmp/temp_diff.diff

# 4. 检查执行结果
if [ $? -eq 0 ]; then
    echo "成功: 已将 HEAD~1 与 HEAD 的差异保存至 tmp/temp_diff.diff。"
else
    echo "失败: git diff 执行出错。"
    exit 1
fi