#!/usr/bin/env bash
# MiniMax API 认证诊断脚本
# 用于排查 Claude Code CLI 401 问题：测试不同 endpoint 和认证方式

set -e
API_KEY="${1:-}"
if [ -z "$API_KEY" ]; then
  echo "用法: $0 <YOUR_MINIMAX_API_KEY>"
  echo "或: source ~/.claude/settings.json 后从 env 读取（需手动传入）"
  exit 1
fi

# Anthropic Messages API 格式的 minimal 请求体
# 注意：若需代理访问，请先执行 proxyon
BODY='{"model":"MiniMax-M2.7","max_tokens":10,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}'

echo "=========================================="
echo "MiniMax API 认证诊断"
echo "=========================================="
echo ""

for BASE in "https://api.minimaxi.com/anthropic" "https://api.minimax.io/anthropic"; do
  echo ">>> 测试 endpoint: $BASE"
  echo ""
  
  echo "  [1] x-api-key header (ANTHROPIC_API_KEY 方式):"
  resp=$(curl -s -w "\n%{http_code}" -X POST "$BASE/v1/messages" \
    -H "Content-Type: application/json" \
    -H "x-api-key: $API_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -d "$BODY" 2>/dev/null)
  code=$(echo "$resp" | tail -n1)
  body=$(echo "$resp" | sed '$d')
  if [ "$code" = "200" ]; then
    echo "      ✓ 成功 (HTTP $code)"
  else
    echo "      ✗ 失败 (HTTP $code)"
    echo "$body" | head -c 300
    echo ""
  fi
  echo ""
  
  echo "  [2] Authorization: Bearer (ANTHROPIC_AUTH_TOKEN 方式):"
  resp=$(curl -s -w "\n%{http_code}" -X POST "$BASE/v1/messages" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -d "$BODY" 2>/dev/null)
  code=$(echo "$resp" | tail -n1)
  body=$(echo "$resp" | sed '$d')
  if [ "$code" = "200" ]; then
    echo "      ✓ 成功 (HTTP $code)"
  else
    echo "      ✗ 失败 (HTTP $code)"
    echo "$body" | head -c 300
    echo ""
  fi
  echo ""
done

echo "=========================================="
echo "结论："
echo "- API Key 必须与 endpoint 匹配："
echo "  - platform.minimaxi.com 创建的 key → 用 api.minimaxi.com"
echo "  - platform.minimax.io 创建的 key  → 用 api.minimax.io"
echo "- 若两种认证方式都失败，请检查：key 是否有效、是否开通按量计费、网络/代理"
echo "=========================================="
