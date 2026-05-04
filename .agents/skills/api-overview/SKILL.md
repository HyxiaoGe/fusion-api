---
name: api-overview
description: Fusion API 项目架构、技术栈、核心组件总览。Use when you need to understand how the backend works, its architecture, or tech stack.
---

# Fusion API 项目总览

## 定位

基于 FastAPI 的 AI 对话集成平台，通过 LiteLLM 提供统一的多 LLM 提供商接口。

## 技术栈

- **框架**: FastAPI + Uvicorn (4 workers)
- **LLM 接口**: LiteLLM（7 个提供商，其中 4 个走 OpenRouter 路由）
- **搜索增强**: Web Search tool_call + 私有 search-service（Tavily）
- **数据库**: PostgreSQL + SQLAlchemy ORM
- **缓存/流**: Redis Stream（流式输出解耦架构）
- **认证**: JWT + JWKS（独立 auth-service）
- **部署**: Docker Compose / Railway

## 分层架构

```
API Layer (app/api/)           ← FastAPI 路由
  ↓
Service Layer (app/services/)  ← 业务逻辑
  ↓
AI Layer (app/ai/)             ← LLMManager (LiteLLM)
  ↓
Data Layer (app/db/)           ← SQLAlchemy ORM + Repository
```

## 核心组件

### AI 集成 (`app/ai/`)
- `llm_manager.py` — LLM 调用管理器，`PROVIDER_LITELLM_PREFIX` 字典映射提供商
  - OpenAI/Anthropic/Google/xAI 走 `openrouter/` 前缀路由
  - DeepSeek/Qwen/Volcengine 直连
- `resolve_model()` 根据 model_id 查凭证，构造 LiteLLM 参数
- `tools.py` — LLM Tool 定义（web_search）
- `prompts/` — 提示词模板管理

### API (`app/api/`)
- `chat.py` — 对话（send/stop/stream-status/stream/conversations CRUD）
- `files.py` — 文件上传与管理
- `models.py` — 模型与凭证 CRUD
- `auth.py` — 用户认证

### Service (`app/services/`)
- `chat_service.py` — 对话业务逻辑（传递 capabilities 给 stream_handler）
- `stream_handler.py` — Redis Stream 两段式流架构（核心），支持 web_search tool_call 检测
- `stream_state_service.py` — Redis Stream 状态管理（Lua 原子脚本）
- `search_client.py` — 搜索服务 HTTP 客户端（调用 search-service）
- `memory_service.py` — 数据库持久化
- `task_manager.py` — 后台任务注册与取消
- `file_service.py` — 文件处理

### 核心工具 (`app/core/`)
- `config.py` — Pydantic Settings
- `redis.py` — 连接池 + Lua 脚本加载
- `security.py` — JWT/JWKS 验证
- `lua/` — Redis Lua 原子脚本

### 数据模型 (`app/db/models.py`)
- users, social_accounts, conversations, messages (JSONB content blocks)
- files, conversation_files, model_sources, model_credentials

## 关键架构：Redis Stream 两段式流

```
POST /send
  ├─ create_task(generate_to_redis)  # 后台任务，独立于 HTTP
  │    └─ LLM chunk → XADD Redis Stream → 完成后写 PostgreSQL
  └─ StreamingResponse(stream_redis_as_sse)  # SSE 只读 Redis
       └─ XREAD → 推送给客户端
```

客户端断线不影响生成，重连通过 `GET /stream/{conv_id}` 续读。

## Web Search 增强架构

```
generate_to_redis() 后台任务
  ├─ 模型支持 functionCalling?
  │   ├─ 否 → 纯流式路径（不变）
  │   └─ 是 → 第一轮 LLM (stream=true, tools=[web_search])
  │       ├─ delta.content → 正常输出（thinking 缓冲不推送）
  │       └─ delta.tool_calls → 累积 → 调 search-service → 注入正文 → 第二轮流式
  ↓
落库 PostgreSQL (content: [ThinkingBlock?, SearchBlock?, TextBlock])
```

- 搜索由 LLM 自主决策（tool_call），无需前端开关
- search-service 使用 Tavily advanced 模式，返回网页正文（~2000 字/条）
- 第一轮 thinking 缓冲：不推送到 Redis，tool_call 时丢弃，正常回答时回放
- SSE 新增 `thinking_pending`、`search_start`、`search_complete` 事件类型

## 常用命令

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload  # 开发
docker-compose up -d                                    # Docker
pytest test/                                            # 测试
```
