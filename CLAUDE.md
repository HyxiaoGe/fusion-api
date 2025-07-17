# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言和开发规范

### 语言要求

- **所有回复必须使用中文**：包括代码注释、解释说明、错误信息等
- **Git 提交信息必须使用中文**：提交标题和描述都使用中文
- **文档和注释使用中文**：所有新创建的文档、代码注释都使用中文

### Git 提交规范

- **提交信息格式要求**：

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

Fusion API is a Python-based AI chat integration platform built with FastAPI that provides a unified interface for multiple Large Language Model (LLM) providers including Anthropic, OpenAI, Google Gemini, DeepSeek, and various Chinese AI services (Qwen, Wenxin, Hunyuan, etc.).

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

The application follows a clean architecture pattern with clear separation of concerns:

### Core Components

1. **AI Integration Layer** (`app/ai/`)
   - `llm_manager.py`: Central manager for all LLM integrations
   - Provider-specific adapters in `providers/` subdirectory
   - Each provider adapter implements a common interface for chat completions

2. **API Layer** (`app/api/`)
   - FastAPI routers for different resource endpoints
   - Main endpoints: chat, files, settings, prompts, models, credentials
   - Authentication endpoints for OAuth flows

3. **Service Layer** (`app/services/`)
   - Business logic separated from API layer
   - Key services:
     - `conversation_service.py`: Manages chat sessions and history
     - `file_service.py`: Handles file uploads and processing
     - `auto_title_service.py`: Generates conversation titles using LLMs
     - `hot_topic_service.py`: Manages trending topics
     - `scheduled_task_service.py`: Background task scheduling

4. **Data Layer** (`app/db/`)
   - SQLAlchemy models for PostgreSQL
   - Repository pattern for data access
   - Models: Conversation, Message, File, Setting, User, etc.

5. **Vector Search** (`app/services/chromadb_service.py`)
   - ChromaDB integration for semantic search
   - Embeddings storage and retrieval
   - RAG (Retrieval Augmented Generation) support

### Key Design Patterns

1. **Adapter Pattern**: Each LLM provider has an adapter implementing a common interface
2. **Repository Pattern**: Database access is abstracted through repository classes
3. **Dependency Injection**: FastAPI's dependency injection for services and database sessions
4. **Middleware Pipeline**: Request timeout, CORS, and session management

### Important Implementation Details

1. **Stream Support**: All LLM integrations support streaming responses
2. **Error Handling**: Custom exceptions with proper HTTP status codes
3. **Configuration**: Environment-based configuration via `.env` file
4. **Authentication**: JWT tokens with OAuth provider support (GitHub, Google)
5. **File Processing**: Supports PDF, DOCX, and text file uploads with content extraction

### Database Schema

The application uses PostgreSQL with the following main tables:
- `conversations`: Chat sessions with metadata
- `messages`: Individual messages in conversations
- `files`: Uploaded files with vector embeddings
- `settings`: Application and user settings
- `users`: User accounts with OAuth associations
- `credentials`: Encrypted API credentials for LLM providers

### Testing

Currently, the project has minimal test infrastructure. When adding tests:
- Create test files in the `test/` directory
- Consider adding pytest to requirements.txt
- Test API endpoints using FastAPI's TestClient
- Mock external LLM API calls to avoid using real credentials

### Common Development Tasks

1. **Adding a New LLM Provider**:
   - Create a new adapter in `app/ai/providers/`
   - Implement the base adapter interface
   - Register in `LLMManager` in `app/ai/llm_manager.py`
   - Add configuration constants in `app/constants/`

2. **Adding New API Endpoints**:
   - Create router in `app/api/`
   - Implement service logic in `app/services/`
   - Add Pydantic schemas in `app/schemas/`
   - Register router in `main.py`

3. **Database Migrations**:
   - Modify models in `app/db/models/`
   - The app auto-creates tables on startup (see `app/db/base.py`)
   - For production, consider using Alembic for migrations

### Performance Considerations

1. **Docker Resource Limits**: CPU limited to 1 core, memory to 1GB
2. **Request Timeout**: 10-second timeout middleware applied
3. **Database Connections**: Connection pooling via SQLAlchemy
4. **Streaming**: Use streaming for long LLM responses to improve UX

### Security Notes

1. API credentials are encrypted before storage
2. CORS is currently open - restrict for production
3. Environment variables for sensitive configuration
4. JWT tokens for authentication with configurable expiry