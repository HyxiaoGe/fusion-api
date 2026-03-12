# Fusion API - Chat Core

## 📖 项目介绍

Fusion API 是 Fusion 的聊天核心后端，提供认证、模型管理、会话管理、流式回复和文件辅助聊天能力。

## ✨ 主要功能

### 核心功能
- **多模型支持**：集成 DeepSeek、OpenAI、Google、Anthropic、通义千问、文心一言、火山引擎、讯飞星火等模型
- **流式响应**：支持实时流式输出，提供更好的用户体验
- **会话管理**：保存完整的对话历史，支持多轮对话
- **文件处理**：支持上传和处理 PDF、Word、文本等格式文件
- **用户认证**：支持 GitHub / Google OAuth 和 JWT
- **模型管理**：动态配置和管理不同的 AI 模型

### 实用功能
- **自动生成标题**：基于对话内容智能生成对话标题
- **推荐问题**：根据当前对话生成相关的推荐问题

## 🔧 技术栈

- **后端框架**：FastAPI
- **数据库**：PostgreSQL + SQLAlchemy ORM
- **异步支持**：asyncio + httpx
- **AI框架**：LangChain
- **容器化**：Docker & Docker Compose
- **认证**：JWT + OAuth 2.0

## 🚀 快速开始

### 使用 Docker 部署（推荐）

1. 克隆项目
```bash
git clone <repository-url>
cd fusion-api
```

2. 配置环境变量
创建 `.env` 文件，配置数据库、OAuth 和至少一个模型凭证：
```env
# 数据库配置
DATABASE_URL=postgresql://fusion:fusion123!!@fusion_postgres:5432/fusion

# OAuth
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret

# AI 模型 API 密钥（根据需要配置）
DEEPSEEK_API_KEY=your_deepseek_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
# ... 其他模型密钥
```

3. 启动服务
```bash
docker-compose up -d
```

4. 访问 API 文档
```
http://localhost:8000/docs
```

### 手动安装

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 配置数据库
确保 PostgreSQL 已安装并创建数据库

3. 启动应用
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📚 API 使用示例

### 发送消息
```bash
POST /api/chat/send
{
  "provider": "deepseek",
  "model": "deepseek-chat",
  "message": "你好",
  "stream": true,
  "options": {
    "use_reasoning": true
  }
}
```

### 获取会话历史
```bash
GET /api/chat/conversations/{conversation_id}
```

### 生成对话标题
```bash
POST /api/chat/generate-title
{
  "conversation_id": "xxx"
}
```

## 当前范围

- 默认运行面仅暴露 `chat / auth / files / models`
- RSS、热点、摘要、调度等内容平台能力已退出当前主产品范围
- `web search` 和 `function call` 作为后续可控增强能力保留在代码边缘，不是当前默认产品面

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
