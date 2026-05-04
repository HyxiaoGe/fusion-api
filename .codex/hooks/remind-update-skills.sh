#!/bin/bash
# Stop hook：提醒检查 SKILL.md 是否需要同步更新
# 检查本次会话是否修改了关键模块代码

CHANGED_FILES=$(git diff --name-only HEAD 2>/dev/null)
if [ -z "$CHANGED_FILES" ]; then
  exit 0
fi

NEED_UPDATE=""

# 检查是否改了 API 路由
echo "$CHANGED_FILES" | grep -q "app/api/" && NEED_UPDATE="$NEED_UPDATE\n  - app/api/ 变更 → 检查 api-reference/SKILL.md"

# 检查是否改了流式处理
echo "$CHANGED_FILES" | grep -q -E "stream_handler|stream_state|task_manager|lua/" && NEED_UPDATE="$NEED_UPDATE\n  - 流式模块变更 → 检查 debug-stream/SKILL.md"

# 检查是否改了 LLM 管理
echo "$CHANGED_FILES" | grep -q "llm_manager" && NEED_UPDATE="$NEED_UPDATE\n  - LLM 管理变更 → 检查 add-provider/SKILL.md"

# 检查是否改了核心架构
echo "$CHANGED_FILES" | grep -q -E "chat_service|main.py|app/core/" && NEED_UPDATE="$NEED_UPDATE\n  - 核心架构变更 → 检查 api-overview/SKILL.md"

if [ -n "$NEED_UPDATE" ]; then
  echo "提醒：本次修改了模块代码，请确认是否需要同步更新 SKILL.md：" >&2
  echo -e "$NEED_UPDATE" >&2
fi

exit 0
