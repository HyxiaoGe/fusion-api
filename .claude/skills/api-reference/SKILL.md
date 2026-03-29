---
name: api-reference
description: Quick reference for all API endpoints. Use when you need to know available endpoints, their methods, and purposes.
---

# API 端点速查

## Chat (`/api/chat`)

| Method | Path | 说明 |
|--------|------|------|
| POST | `/send` | 发送消息（流式/非流式） |
| GET | `/conversations` | 分页对话列表（page, page_size） |
| GET | `/conversations/{id}` | 对话详情（含完整消息） |
| DELETE | `/conversations/{id}` | 删除对话 |
| PUT | `/conversations/{id}/messages/{msg_id}` | 编辑消息 |
| POST | `/generate-title` | 自动生成对话标题 |
| POST | `/suggest-questions` | 生成推荐后续问题 |
| GET | `/stream-status/{conv_id}` | 查询流状态（streaming/cancelled/done） |
| GET | `/stream/{conv_id}` | 断线重连 SSE（last_entry_id 参数） |
| POST | `/stop/{conv_id}` | 停止生成（message_id 参数防误杀） |

### 关键请求体

**POST /send:**
```json
{
  "model_id": "qwen3-235b-a22b",
  "message": "你好",
  "conversation_id": null,
  "stream": true,
  "options": {"use_reasoning": true},
  "file_ids": ["uuid"]
}
```

**SSE 帧格式:**
```
id: {redis_entry_id}
data: {"id": "{msg_id}", "conversation_id": "{conv_id}", "choices": [{"delta": {"content": [{"type": "text", "id": "blk_xxx", "text": "..."}]}, "finish_reason": null}]}
```

## Files (`/api/files`)

| Method | Path | 说明 |
|--------|------|------|
| POST | `/upload` | 上传文件（multipart/form-data） |
| GET | `/` | 用户文件列表 |
| GET | `/conversation/{id}` | 对话关联文件 |
| GET | `/{file_id}/status` | 文件处理状态 |
| DELETE | `/{file_id}` | 删除文件 |

## Models (`/api/models`)

| Method | Path | 说明 |
|--------|------|------|
| GET | `/` | 模型列表 |
| GET | `/{model_id}` | 模型详情 |
| POST | `/` | 创建模型 |
| PUT | `/{model_id}` | 更新模型 |
| DELETE | `/{model_id}` | 删除模型 |
| GET | `/{model_id}/credentials` | 凭证列表 |
| POST | `/{model_id}/credentials` | 创建凭证 |
| PUT | `/credentials/{cred_id}` | 更新凭证 |
| DELETE | `/credentials/{cred_id}` | 删除凭证 |
| POST | `/credentials/test` | 测试凭证 |

## Auth (`/api/auth`)

| Method | Path | 说明 |
|--------|------|------|
| GET | `/me` | 当前用户信息 |

## Health

| Method | Path | 说明 |
|--------|------|------|
| GET | `/health` | 健康检查（数据库连接状态） |

## 认证

所有端点（除 `/health`）需要 JWT：
```
Authorization: Bearer <access_token>
```

Token 从 auth-service（端口 8100）获取。
