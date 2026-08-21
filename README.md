# Fusion API - Chat Core

## 📖 项目介绍

Fusion API 是 Fusion 的聊天核心后端，提供认证、模型管理、会话管理、流式回复和文件辅助聊天能力。

## ✨ 主要功能

### 核心功能
- **多模型支持**：集成 DeepSeek、OpenAI、Google、Anthropic、通义千问、文心一言、火山引擎、讯飞星火等模型
- **流式响应**：支持实时流式输出，提供更好的用户体验
- **会话管理**：保存完整的对话历史，支持多轮对话
- **文件处理**：支持上传和处理 PDF、Word、文本等格式文件
- **知识库**：独立管理文档，使用持久 Worker 预索引并通过 Milvus 检索（默认关闭）
- **用户认证**：支持 GitHub / Google OAuth 和 JWT
- **模型管理**：动态配置和管理不同的 AI 模型

知识库 v1 的配置、状态机、非 root Milvus 初始化、独立 Worker、真实验收和故障恢复见
[知识库运行手册](docs/KNOWLEDGE_BASE.md)。

LiteLLM 全模型健康探测（`/health` 会产生真实 completion 费用）的开关、多实例协调
与迁移/回滚见 [LiteLLM 健康探测成本治理](docs/LITELLM_HEALTH.md)。

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

### 发布安全与回滚

`master` push 和未填写回滚参数的手动运行都属于正常发布。dev 部署在任何 migration 或候选
容器变更前，会同时保存 `fusion-api`、`fusion-flyai-adapter` 两个运行中容器的完整镜像引用
（`.Config.Image`）和实际内容 ID（`.Image`）；任一回滚目标缺失时 fail-closed。

候选部署开始后，镜像身份、健康检查或 deployment smoke 任一失败都会恢复两个旧镜像。恢复后
必须精确核对旧镜像引用与内容 ID，并重新执行容器内 API/adapter health 和
`scripts/deployment_smoke.py`。自动回滚成功不会掩盖原发布失败，回滚失败也不会被忽略。旧镜像
只在整次发布成功后清理本地副本，ACR 中的 SHA 标签继续作为手动回滚来源。

手动回滚时，在 GitHub Actions 的 `Fusion API Windows CI` 中填写已经成功发布过的 40 位小写
`rollback_sha` 和非空 `rollback_reason`。该路径跳过 Windows runner 上的构建、registry 登录、
镜像推送及 Alembic migration；整个 Windows publish job 不会排队，因此 Windows runner 离线不
阻断手动回滚。GitHub-hosted prepare job 负责校验输入，finalize job 只把“回滚模式、publish
skipped、deploy success”判为成功；普通发布仍必须同时满足 publish 与 deploy success。部署、
指标和通知统一记录实际回滚 SHA。

```bash
gh workflow run deploy.yml --ref master \
  -f rollback_sha=0123456789abcdef0123456789abcdef01234567 \
  -f rollback_reason='候选版本健康检查失败'
```

数据库只允许 expand/contract 演进：先发布兼容 schema 扩展，确认所有可回滚版本不再依赖旧
结构后再独立删除。镜像回滚绝不执行 `alembic downgrade`，手动回滚前必须确认目标 SHA 与当前
schema 兼容。

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

- 运行面暴露 `chat / auth / files / models` 四个 API 路由
- RSS、热点、摘要、调度、web search、function call 等扩展能力已清理，代码中不再保留

## 数据流文档

- 聊天核心数据流说明见 [`CHAT_CORE_DATA_FLOW.md`](/Users/sean/code/fusion/fusion-api/CHAT_CORE_DATA_FLOW.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
