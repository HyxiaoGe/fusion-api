#!/bin/bash
# UserPromptSubmit hook：收到用户需求时，提醒开发前自查
# 每次用户发消息都会触发，简短提醒不打断流程

echo "💡 开发前确认：" >&2
echo "  1. 涉及不熟悉的技术/效果/算法 → 先 WebSearch 搜参考实现" >&2
echo "  2. 改动前想清楚关联位置，一次改全，不要分多次" >&2
echo "  3. 想好怎么验证（后端 curl / 前端走读渲染路径）" >&2
exit 0
