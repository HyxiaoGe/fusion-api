#!/bin/bash
# 阻止危险命令：rm -rf、DROP TABLE、TRUNCATE、git push --force 等
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

if echo "$COMMAND" | grep -iE 'rm\s+-rf\s+/|drop\s+table|truncate\s+table|git\s+push\s+--force|git\s+reset\s+--hard' >/dev/null 2>&1; then
  echo "⚠️ 危险命令被阻止: $COMMAND" >&2
  exit 2
fi

exit 0
