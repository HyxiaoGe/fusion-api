#!/bin/bash
# 编辑 Python 文件后检查语法错误
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [[ "$FILE_PATH" == *.py ]] && [[ -f "$FILE_PATH" ]]; then
  OUTPUT=$(python3 -m py_compile "$FILE_PATH" 2>&1)
  if [[ $? -ne 0 ]]; then
    echo "⚠️ Python 语法错误: $OUTPUT" >&2
    exit 2
  fi
fi

exit 0
