---
name: dev-logs
description: 查看 dev 服务器 fusion-api 日志。Use when debugging backend issues, checking errors, or tracing request flow.
argument-hint: [conversation_id 或关键词]
allowed-tools: Bash
---

# 查看 Dev 服务器日志

通过 `ssh dev` 连接开发服务器查看 fusion-api 容器日志。

## 常用命令

### 最近日志（过滤 LiteLLM 噪音）
```bash
ssh dev "docker logs fusion-api 2>&1 | tail -50" | grep -v 'LiteLLM\|litellm\|backoff\|ImportError\|Module\|During\|Traceback\|File "/usr\|proxy_server\|cold_storage'
```

### 应用级日志（只看 app logger）
```bash
ssh dev "docker logs fusion-api 2>&1" | grep 'app - ' | tail -30
```

### 按会话 ID 过滤
```bash
ssh dev "docker logs fusion-api 2>&1" | grep '{conversation_id}' | grep -v OPTIONS
```

### 按时间范围
```bash
ssh dev "docker logs fusion-api 2>&1" | grep -E '08:1[0-9]' | grep -v 'GET /docs'
```

### 查看请求流水
```bash
ssh dev "docker logs fusion-api 2>&1" | grep -E 'POST /api/chat/(send|stop)|stream-status|/stream/' | tail -20
```

### 查看错误
```bash
ssh dev "docker logs fusion-api 2>&1" | grep -E 'ERROR|异常|失败' | tail -20
```

## 注意事项

- 容器名固定为 `fusion-api`
- API 端口映射为 8002（不是 8000）
- 日志中 LiteLLM 的 backoff 报错是误报（缺少 litellm[proxy] 依赖），可忽略
- 用 `ssh dev` 直连开发服务器
