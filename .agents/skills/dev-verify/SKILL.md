---
name: dev-verify
description: 验证 dev 服务器部署状态：容器、健康检查、Redis、CI。Use after pushing code to verify deployment.
allowed-tools: Bash
---

# 验证 Dev 服务器部署

推送代码后，按以下步骤验证部署状态。

## 1. 检查 CI

```bash
# fusion-api
gh run list --repo HyxiaoGe/fusion-api --limit 1

# fusion-ui
gh run list --repo HyxiaoGe/fusion-ui --limit 1
```

如果状态是 `queued` 或 `in_progress`，等待完成：
```bash
gh run view {run_id} --repo HyxiaoGe/fusion-api
```

## 2. 检查容器状态

```bash
ssh dev "docker ps --filter name=fusion --format '{{.Names}}: {{.Status}}'"
```

## 3. 健康检查

```bash
ssh dev "curl -s http://localhost:8002/health"
```

期望返回：`{"status":"healthy","database":"connected",...}`

## 4. 检查 Redis 连接

```bash
ssh dev "docker logs fusion-api 2>&1 | grep 'Redis' | tail -5"
```

期望看到：`Redis 连接池初始化成功`

## 5. 检查 Redis 流状态

```bash
# 查看所有活跃流
ssh dev "docker exec middleware-redis redis-cli keys 'stream:*'"

# 查看特定流的 meta
ssh dev "docker exec middleware-redis redis-cli hgetall 'stream:meta:{conv_id}'"
```

## 6. 验证代码是否生效

```bash
# 检查容器内文件内容
ssh dev "docker exec fusion-api grep '关键代码' /app/path/to/file.py"

# 检查 Lua 脚本
ssh dev "docker exec fusion-api cat /app/app/core/lua/cancel_stream.lua"
```

## 关键信息

- **fusion-api 端口**: 8002（宿主机） → 8000（容器内）
- **fusion-ui 端口**: Nginx 代理
- **Redis 容器名**: middleware-redis
- **网络**: middleware_default + postgres_default
- 不要直接修改 dev 服务器上的代码，走 git + CI
