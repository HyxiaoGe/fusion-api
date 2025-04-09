# Fusion API - AI聊天集成平台

<p align="center">
  <img src="https://img.shields.io/badge/版本-0.1.0-blue.svg" alt="版本">
  <img src="https://img.shields.io/badge/许可证-MIT-green.svg" alt="许可证">
  <img src="https://img.shields.io/badge/Python-3.8+-brightgreen.svg" alt="Python版本">
</p>

## 📖 项目介绍

Fusion API是一个强大的AI聊天集成平台，支持多种大型语言模型（LLM）如文心一言、通义千问、OpenAI等，提供统一的API接口进行交互。系统集成了向量数据库功能，支持知识检索和上下文增强，让AI回答更加精准。

## ✨ 主要特性

- 🤖 **多模型集成**：支持文心一言、通义千问、DeepSeek、OpenAI等多种AI模型
- 🔄 **统一API接口**：提供一致的交互体验，无论使用哪种底层模型
- 💬 **会话管理**：完整的会话历史记录和管理功能
- 📝 **自动标题生成**：基于对话内容智能生成对话标题
- 🔍 **向量检索功能**：集成ChromaDB实现高效的语义搜索
- 📁 **文件处理**：支持上传PDF、Word、文本等多种格式文件进行分析
- 🛠️ **可定制提示词**：支持自定义提示词模板，优化AI输出

## 🔧 技术栈

- **后端框架**：FastAPI
- **数据库**：PostgreSQL + SQLAlchemy
- **向量数据库**：ChromaDB
- **AI/LLM集成**：LangChain框架
- **容器化**：Docker & Docker Compose
- **Web服务**：Nginx, Uvicorn

## 🚀 快速开始

### 前置条件

- Python 3.8+
- Docker & Docker Compose
- PostgreSQL数据库
- 各AI服务商的API密钥

### 环境变量配置

创建`.env`文件并配置以下环境变量：

```
# 数据库配置
DATABASE_URL=postgresql://fusion:fusion123!!@postgres:5432/fusion

# API密钥
WENXIN_API_KEY=你的文心一言API密钥
WENXIN_SECRET_KEY=你的文心一言密钥
DASHSCOPE_API_KEY=你的通义千问API密钥
DEEPSEEK_API_KEY=你的DeepSeek API密钥
OPENAI_API_KEY=你的OpenAI API密钥

# 其他配置
ENABLE_VECTOR_EMBEDDINGS=true
```

### 使用Docker部署

1. 克隆仓库
```bash
git clone https://github.com/yourusername/fusion-api.git
cd fusion-api
```

2. 启动服务
```bash
docker-compose up -d
```

3. 访问API文档
```
http://localhost:8000/docs
```

### 手动安装

1. 克隆仓库
```bash
git clone https://github.com/yourusername/fusion-api.git
cd fusion-api
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 启动应用
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
``` 