---
name: dev-test-api
description: 用 curl 测试 dev 服务器 API 端点。Use to verify API behavior, test streaming, or debug issues end-to-end.
argument-hint: [测试场景描述]
allowed-tools: Bash
---

# Dev 服务器 API 测试

通过 curl 直接调用 dev 服务器的 API，验证功能闭环。

## 前置：获取 Token

```bash
TOKEN=$(ssh dev "curl -s -X POST http://localhost:8100/auth/login \
  -H 'Content-Type: application/json' \
  -d '{\"email\":\"Codex-test@fusion.dev\",\"password\":\"test123456\",\"client_id\":\"app_a93ea0569cafafe6299c7f660669a5b7\"}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)[\"access_token\"])'")
```

测试账号：`Codex-test@fusion.dev` / `test123456`，token 有效期 24 小时。

## 常用测试

### 发消息（流式）
```bash
ssh dev "timeout 5 curl -s -N \
  -H 'Authorization: Bearer $TOKEN' \
  -H 'Content-Type: application/json' \
  -X POST http://localhost:8002/api/chat/send \
  -d '{\"model_id\":\"qwen3-235b-a22b\",\"message\":\"你好\",\"stream\":true}'"
```

### 查询流状态
```bash
ssh dev "curl -s -H 'Authorization: Bearer $TOKEN' \
  'http://localhost:8002/api/chat/stream-status/{conv_id}'"
```

### 停止流
```bash
ssh dev "curl -s -H 'Authorization: Bearer $TOKEN' \
  -X POST 'http://localhost:8002/api/chat/stop/{conv_id}?message_id={msg_id}'"
```

### 获取对话列表
```bash
ssh dev "curl -s -H 'Authorization: Bearer $TOKEN' \
  'http://localhost:8002/api/chat/conversations?page=1&page_size=5'"
```

### 获取对话详情
```bash
ssh dev "curl -s -H 'Authorization: Bearer $TOKEN' \
  'http://localhost:8002/api/chat/conversations/{conv_id}'"
```

## 端到端测试：发消息 → stop → 验证

```bash
# 1. 发消息，读 3 秒后断开
RESPONSE=$(ssh dev "timeout 3 curl -s -N \
  -H 'Authorization: Bearer $TOKEN' \
  -H 'Content-Type: application/json' \
  -X POST http://localhost:8002/api/chat/send \
  -d '{\"model_id\":\"qwen3-235b-a22b\",\"message\":\"说一个字\",\"stream\":true}' 2>&1 || true")

# 2. 提取 IDs
CONV_ID=$(echo "$RESPONSE" | python3 -c "import sys,json
for l in sys.stdin:
    if 'conversation_id' in l and 'data:' in l:
        print(json.loads(l.split('data: ')[1])['conversation_id']); break" 2>/dev/null)
MSG_ID=$(echo "$RESPONSE" | python3 -c "import sys,json
for l in sys.stdin:
    if 'data:' in l and '\"id\"' in l:
        print(json.loads(l.split('data: ')[1])['id']); break" 2>/dev/null)
echo "conv=$CONV_ID msg=$MSG_ID"

# 3. 检查 stream-status（应为 streaming）
ssh dev "curl -s -H 'Authorization: Bearer $TOKEN' \
  'http://localhost:8002/api/chat/stream-status/$CONV_ID'"

# 4. 发 stop
ssh dev "curl -s -H 'Authorization: Bearer $TOKEN' \
  -X POST 'http://localhost:8002/api/chat/stop/$CONV_ID?message_id=$MSG_ID'"

# 5. 验证状态变为 cancelled
sleep 1
ssh dev "curl -s -H 'Authorization: Bearer $TOKEN' \
  'http://localhost:8002/api/chat/stream-status/$CONV_ID'"

# 6. 检查 Redis
ssh dev "docker exec middleware-redis redis-cli hgetall 'stream:meta:$CONV_ID'"
```

## 注意事项

- 所有请求通过 `ssh dev` 在服务器本地执行（localhost:8002）
- token 变量需要在每条 ssh 命令中传递，或先 export
- `timeout N curl` 用于限制流式读取时间
- 不要直接修改 dev 服务器代码
