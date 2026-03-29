# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言和开发规范

### 语言要求

- **所有回复必须使用中文**：包括代码注释、解释说明、错误信息等
- **Git 提交信息必须使用中文**：提交标题和描述都使用中文
- **文档和注释使用中文**：所有新创建的文档、代码注释都使用中文

### Git 提交规范

- **提交信息格式要求**：

  - **必须使用类型前缀 + 中文摘要**：格式固定为 `<type>: <中文描述>`
  - 示例：
    - `feat: 添加 RAG 检索增强功能`
    - `fix: 修复 SSE 收尾边界问题`
    - `refactor: 重构聊天 SSE 流式协议`

  ```
  feat: 添加RAG检索增强功能
  
  - 实现向量数据库集成
  - 优化文档分块策略
  - 添加混合搜索支持
  
  🤖 Generated with [Claude Code](https://claude.ai/code)
  
  Co-Authored-By: Claude <noreply@anthropic.com>
  ```

- **必须包含 Co-author 信息**：每个提交都要包含 `Co-authored-by: Claude Code <noreply@anthropic.com>`

- **使用中文提交类型**：
  - `feat`: 新功能
  - `fix`: 修复bug
  - `docs`: 文档更新
  - `style`: 代码格式调整
  - `refactor`: 重构代码
  - `test`: 测试相关
  - `chore`: 构建工具或辅助工具的变动

### 个人开发偏好

- **代码风格**：使用4个空格缩进，不使用Tab
- **函数命名**：使用动词开头，如 `获取用户信息()`, `处理文档()`
- **错误处理**：优先使用 try-except，提供中文错误信息
- **日志格式**：使用中文日志信息，便于调试
- **注释语言**：所有代码注释使用中文
- **变量命名**：使用英文，但注释说明使用中文
- **函数设计**：单个函数不超过50行，职责单一
- **导入顺序**：标准库 → 第三方库 → 本地模块，每组之间空一行
- **字符串处理**：优先使用 f-string 格式化，避免使用 % 格式化
- **文件路径**：使用 `pathlib.Path` 而不是 `os.path`
- **配置管理**：使用 `.env` 文件管理环境变量，敏感信息不写入代码
- **依赖管理**：使用 `requirements.txt` 锁定版本，重要依赖添加中文注释说明用途

### 文档和注释偏好

- **函数文档**：所有函数必须有中文docstring，说明参数、返回值、异常
- **类文档**：类的作用、主要方法、使用示例都用中文描述
- **复杂逻辑**：超过5行的复杂逻辑必须添加中文注释解释
- **TODO标记**：使用中文 `# TODO: 待实现功能描述` 格式
- **代码示例**：在文档中提供中文注释的完整代码示例

### 测试和质量保证

- **测试覆盖**：重要函数必须有对应的测试用例
- **测试命名**：测试函数使用中文描述，如 `test_用户登录_成功场景()`
- **断言信息**：断言失败时提供中文错误信息
- **测试数据**：使用中文测试数据，更贴近实际使用场景
- **性能测试**：关键算法需要添加性能测试和基准测试

### 调试和日志偏好

- **调试信息**：使用中文debug信息，便于定位问题
- **日志级别**：开发环境使用DEBUG，生产环境使用INFO
- **异常捕获**：捕获异常时记录中文上下文信息
- **打印调试**：临时调试可以使用print，但正式代码必须使用logging
- **错误追踪**：重要错误必须记录完整的中文错误堆栈

### 安全和性能偏好

- **输入验证**：所有外部输入必须验证，提供中文错误提示
- **密码处理**：使用bcrypt等安全算法，不明文存储
- **API限流**：重要接口添加速率限制
- **缓存策略**：合理使用缓存，避免重复计算
- **资源清理**：及时关闭文件、数据库连接等资源

### 项目结构偏好

- **目录命名**：使用中文拼音或英文，避免中文目录名
- **文件分类**：工具函数放在 `utils/`，配置文件放在 `config/`
- **模块划分**：按功能模块划分，每个模块职责清晰
- **常量定义**：所有魔法数字和字符串定义为有意义的常量
- **环境隔离**：开发、测试、生产环境严格隔离

## Project Overview

Fusion API 是基于 FastAPI 的 AI 对话集成平台，通过 **LiteLLM** 提供统一的多 LLM 提供商接口。

支持的 LLM 提供商（9 个）：

| 提供商 | LiteLLM 前缀 | 接口类型 | 备注 |
|--------|-------------|---------|------|
| OpenAI | `openai` | 原生 | GPT 系列 |
| Anthropic | `anthropic` | 原生 | Claude 系列 |
| DeepSeek | `deepseek` | 原生 | |
| Google | `gemini` | 原生 | Gemini 系列 |
| Qwen（通义千问） | `openai` | OpenAI 兼容 | 阿里云 DashScope |
| Volcengine（火山引擎） | `openai` | OpenAI 兼容 | 字节跳动 |
| Wenxin（文心一言） | `openai` | OpenAI 兼容 | 百度 |
| Hunyuan（混元） | `openai` | OpenAI 兼容 | 腾讯 |
| xAI | `xai` | 原生 | Grok 系列 |

需要自定义 `api_base` 的提供商：qwen、volcengine、wenxin、hunyuan（从数据库凭证中读取 `base_url`）。

## Key Commands

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server with hot reload
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run with Docker
docker-compose up -d

# View Docker logs
docker-compose logs -f

# Rebuild and restart Docker containers
docker-compose build && docker-compose up -d
```

### Production
```bash
# Run production server (4 workers)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Architecture Overview

应用采用分层架构，职责清晰分离：

```
┌─────────────────────────────────────┐
│   API Layer (app/api/)              │
│   chat / files / models / auth      │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   Service Layer (app/services/)     │
│   ChatService / StreamHandler /     │
│   FileService / MemoryService       │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   AI Layer (app/ai/)                │
│   LLMManager (LiteLLM 统一接口)     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   Data Layer                        │
│   PostgreSQL + Redis Stream         │
└─────────────────────────────────────┘
```

### 核心组件

1. **AI 集成层** (`app/ai/`)
   - `llm_manager.py`：统一 LLM 调用管理器，基于 LiteLLM
   - `PROVIDER_LITELLM_PREFIX` 字典定义提供商到 LiteLLM 前缀的映射
   - `resolve_model()` 根据 `model_id` 从数据库查询凭证，构造 LiteLLM 调用参数
   - `prompts/`：提示词模板管理（`prompt_manager.py`、`templates.py`）
   - `adapters/`：文件适配器

2. **API 层** (`app/api/`)
   - `chat.py`：对话端点（发送消息、会话管理、流状态查询、流重连、停止生成）
   - `files.py`：文件上传与管理
   - `models.py`：模型与凭证 CRUD
   - `auth.py`：用户认证（JWT + auth-service）

3. **Service 层** (`app/services/`)
   - `chat_service.py`：对话业务逻辑（消息处理、会话管理、标题生成、推荐问题）
   - `stream_handler.py`：**Redis Stream 两段式流架构**（核心）
     - Part A: `generate_to_redis()` — 后台任务，调用 LLM 写 Redis Stream + 落库 PostgreSQL
     - Part B: `stream_redis_as_sse()` — SSE 读取器，从 Redis Stream 消费推送给客户端
   - `stream_state_service.py`：Redis Stream 状态管理（init/append/finalize/read）
   - `memory_service.py`：数据库持久化（对话/消息落库）
   - `task_manager.py`：后台任务注册与取消
   - `file_service.py`：文件上传、解析、存储

4. **数据层** (`app/db/`)
   - SQLAlchemy ORM 模型（`models.py`）
   - Repository 模式数据访问（`repositories.py`）
   - 数据库初始化（`init_db.py`、`database.py`）

5. **核心工具** (`app/core/`)
   - `config.py`：Pydantic Settings 配置管理
   - `redis.py`：Redis 异步连接池（lifespan 启动/关闭）
   - `security.py`：JWT 认证 + auth-service JWKS 验证
   - `logger.py`：日志配置

### 关键架构：Redis Stream 两段式流

这是系统最核心的设计，解耦了 LLM 生成与 HTTP 连接生命周期：

```
POST /send
  ├─ asyncio.create_task(generate_to_redis())   # 后台任务，独立生命周期
  │    └─ LLM chunk → XADD Redis Stream
  │    └─ 生成完成 → 写 PostgreSQL → XADD done
  │
  └─ StreamingResponse(stream_redis_as_sse())   # SSE 只读 Redis，不调 LLM
       └─ XREAD Redis Stream → 推送给客户端

客户端断线 → SSE reader 取消 → 后台任务继续运行
客户端重连 → GET /stream/{conv_id} → 从断点 ID 续读
```

### 数据库模型（PostgreSQL）

| 表名 | 说明 | 关键字段 |
|------|------|---------|
| `users` | 用户账户 | id, username, nickname, email, avatar |
| `social_accounts` | OAuth 关联 | user_id, provider, provider_user_id |
| `conversations` | 对话会话 | id, user_id, title, model_id |
| `messages` | 消息（JSONB content blocks） | id, conversation_id, role, content, usage |
| `files` | 上传文件 | id, user_id, filename, status, parsed_content |
| `conversation_files` | 文件-对话关联表 | conversation_id, file_id |
| `model_sources` | 模型定义与能力 | model_id, provider, capabilities, pricing |
| `model_credentials` | API 凭证（JSON） | model_id, credentials, is_default |

注意：
- `messages.content` 是 JSONB 类型，存储 content blocks 数组（TextBlock / ThinkingBlock）
- 主键使用 UUID
- 时间戳使用中国时区（+8）
- 关联关系 cascade 删除

### 认证机制

- 独立的 auth-service（默认 `http://localhost:8100`）负责用户注册/登录/OAuth
- fusion-api 通过 JWKS 验证 JWT token（`security.py`）
- 支持 GitHub、Google OAuth 登录
- Token 有效期默认 8 天（11520 分钟）

### API 端点一览

**Chat (`/api/chat`):**
- `POST /send` — 发送消息（支持流式/非流式）
- `GET /conversations` — 分页对话列表
- `GET /conversations/{id}` — 对话详情
- `DELETE /conversations/{id}` — 删除对话
- `PUT /conversations/{id}/messages/{msg_id}` — 编辑消息
- `POST /generate-title` — 自动生成标题
- `POST /suggest-questions` — 推荐后续问题
- `GET /stream-status/{conv_id}` — 查询流状态
- `GET /stream/{conv_id}` — 重连 SSE 流
- `POST /stop/{conv_id}` — 停止生成

**Files (`/api/files`):**
- `POST /upload` — 上传文件
- `GET /` — 用户文件列表
- `GET /conversation/{id}` — 对话文件列表
- `GET /{file_id}/status` — 文件处理状态
- `DELETE /{file_id}` — 删除文件

**Models (`/api/models`):**
- CRUD 模型定义 + 凭证管理 + 凭证测试

**Auth (`/api/auth`):**
- `GET /me` — 获取当前用户信息

### 测试

- 测试文件在 `test/` 目录，使用 pytest
- 运行测试：`pytest test/`
- 覆盖范围：security、llm_manager、chat_service、file_service、stream_handler、memory_service、repositories、chat utils、file_processor 等
- LLM 调用使用 mock，不依赖真实凭证
- Redis 测试使用 fakeredis

### 常见开发任务

1. **添加新 LLM 提供商**：
   - 在 `app/ai/llm_manager.py` 的 `PROVIDER_LITELLM_PREFIX` 字典中添加映射
   - 如需自定义 api_base，加入 `CUSTOM_BASE_URL_PROVIDERS` 集合
   - 在 `.env` 和 `docker-compose.yml` 中添加 API key 环境变量

2. **添加新 API 端点**：
   - 在 `app/api/` 创建路由
   - 在 `app/services/` 实现业务逻辑
   - 在 `app/schemas/` 添加 Pydantic 模型
   - 在 `main.py` 注册路由

3. **数据库变更**：
   - 修改 `app/db/models.py` 中的 ORM 模型
   - 应用启动时自动建表（`init_db.py`）
   - 生产环境建议使用 Alembic 迁移

### 性能与部署

- **Docker 资源限制**：CPU 1 核，内存 2GB
- **请求超时**：10 秒超时中间件
- **数据库连接池**：SQLAlchemy 连接池
- **Redis**：异步连接池，max_connections=20
- **生产模式**：4 worker Uvicorn
- **部署方式**：Docker Compose / Railway
- **网络**：`fusion_fusion_network`（外部）、`middleware_default`（外部，连接 Redis 等中间件）

### 安全

- API 凭证加密存储
- CORS 通过 `CORS_ORIGINS` 环境变量配置
- 敏感配置通过 `.env` 管理
- JWT token 认证 + JWKS 验证
- 支持的文件类型白名单（图片、PDF、Word、文本等）
