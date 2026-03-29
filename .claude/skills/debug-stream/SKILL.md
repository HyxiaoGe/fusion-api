---
name: debug-stream
description: Debug streaming issues (stop not working, content loss, reconnect problems). Use when SSE streaming has bugs.
allowed-tools: Bash, Read, Grep
---

# 调试流式输出问题

## 前置：获取 Token

```bash
TOKEN=$(ssh dev "curl -s -X POST http://localhost:8100/auth/login \
  -H 'Content-Type: application/json' \
  -d '{\"email\":\"claude-test@fusion.dev\",\"password\":\"test123456\",\"client_id\":\"app_a93ea0569cafafe6299c7f660669a5b7\"}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)[\"access_token\"])'")
```

## 1. 检查后端日志

```bash
# 最近的 app 日志（过滤噪音）
ssh dev "docker logs fusion-api 2>&1" | grep -E 'app - |POST /api/chat/(send|stop)|stream-status|/stream/' | tail -30

# 特定会话
ssh dev "docker logs fusion-api 2>&1" | grep '{conversation_id}'
```

## 2. 检查 Redis 流状态

```bash
# 查看所有活跃流
ssh dev "docker exec middleware-redis redis-cli keys 'stream:*'"

# 查看特定会话的 meta
ssh dev "docker exec middleware-redis redis-cli hgetall 'stream:meta:{conv_id}'"

# 查看 lock
ssh dev "docker exec middleware-redis redis-cli get 'stream:lock:{conv_id}'"

# 查看 Stream 内容（最近 5 条）
ssh dev "docker exec middleware-redis redis-cli xrevrange 'stream:chunks:{conv_id}' + - COUNT 5"
```

## 3. 端到端测试流程

```bash
# 发消息
RESPONSE=$(ssh dev "timeout 3 curl -s -N \
  -H 'Authorization: Bearer $TOKEN' \
  -H 'Content-Type: application/json' \
  -X POST http://localhost:8002/api/chat/send \
  -d '{\"model_id\":\"qwen3-235b-a22b\",\"message\":\"说一个字\",\"stream\":true}' 2>&1 || true")

# 提取 IDs
CONV_ID=$(echo "$RESPONSE" | python3 -c "import sys,json
for l in sys.stdin:
    if 'conversation_id' in l and 'data:' in l:
        print(json.loads(l.split('data: ')[1])['conversation_id']); break" 2>/dev/null)
MSG_ID=$(echo "$RESPONSE" | python3 -c "import sys,json
for l in sys.stdin:
    if 'data:' in l and '\"id\"' in l:
        print(json.loads(l.split('data: ')[1])['id']); break" 2>/dev/null)
echo "conv=$CONV_ID msg=$MSG_ID"

# 检查 stream-status
ssh dev "curl -s -H 'Authorization: Bearer $TOKEN' \
  'http://localhost:8002/api/chat/stream-status/$CONV_ID'"

# 发 stop（带 message_id 防误杀）
ssh dev "curl -s -H 'Authorization: Bearer $TOKEN' \
  -X POST 'http://localhost:8002/api/chat/stop/$CONV_ID?message_id=$MSG_ID'"

# 验证 stop 后状态
sleep 1
ssh dev "curl -s -H 'Authorization: Bearer $TOKEN' \
  'http://localhost:8002/api/chat/stream-status/$CONV_ID'"
```

## 4. 常见问题排查

### stop 后刷新仍继续输出
- 检查 `stream-status` 是否返回 `cancelled`
- 如果返回 `streaming`，说明 `cancel_stream` 没执行成功
- 检查 `cancel_stream.lua` 是否因为 meta 状态或 message_id 不匹配而跳过

### 切换对话时报"模型调用失败"
- 检查日志中是否有"任务被踢掉"
- 检查 `cancel_stream` 是否误杀了新一轮的流（message_id 校验问题）

### 重连后内容为空
- 检查 `init_stream` 是否在 SSE 读取器之前执行（应在 `process_message` 中同步调用）
- 检查 Redis Stream 是否有旧轮次的残留数据

### 思考内容复用
- 检查 `init_stream` 是否正确清除了上一轮的 Stream 数据（`redis.delete(stream_chunks_key)`）

## 5. 关键代码路径

| 文件 | 职责 |
|------|------|
| `app/services/chat_service.py` | 入口：init_stream → create_task → StreamingResponse |
| `app/services/stream_handler.py` | Part A: generate_to_redis / Part B: stream_redis_as_sse |
| `app/services/stream_state_service.py` | Redis 操作：init/append/finalize/cancel/check_lock |
| `app/services/task_manager.py` | 进程内任务注册与取消（注意：跨 worker 不可见） |
| `app/core/lua/` | Lua 原子脚本：finalize_stream / cancel_stream / release_lock |
| `app/api/chat.py` | stop 端点：cancel_task + cancel_stream 双通道 |
