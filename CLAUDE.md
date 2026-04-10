# CLAUDE.md — fusion-api 导航文件

## 语言

所有回复、注释、提交信息使用中文。Git 格式：`<type>: <中文描述>`，必须包含 Co-Authored-By。

## 快速命令

```bash
pip install -r requirements.txt          # 安装依赖
uvicorn main:app --reload                # 开发服务器
python -m pytest test/                   # 运行测试
python -m ruff check . && python -m ruff format --check .  # 代码检查
docker-compose up -d                     # Docker 启动
```

## 架构速览

四层架构，依赖只能向下：API(`app/api/`) → Service(`app/services/`) → AI(`app/ai/`) → Data(`app/db/`)

核心设计：Redis Stream 两段式流，LLM 生成与 HTTP 连接完全解耦。详见 → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## 工作流程

1. **变更前**：超过 3 个文件的改动，先输出影响分析，等人类确认
2. **编码中**：遵守 [docs/ARCHITECTURE_RULES.md](docs/ARCHITECTURE_RULES.md)，参考 [docs/CODING_CONVENTIONS.md](docs/CODING_CONVENTIONS.md)
3. **变更后**：运行测试 + ruff 检查，涉及数据流变更则更新对应文档
4. **提交前**：确认改动已 push 且部署通过，不能只改本地就让用户测试

## 常见开发任务

- **添加新 LLM 提供商** → [docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md#添加新-llm-提供商)
- **添加新 API 端点** → [docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md#添加新-api-端点)
- **数据库变更** → [docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md#数据库变更)

## 详细文档索引

| 文档 | 内容 |
|------|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 分层架构、核心组件、Redis Stream 数据流、数据库模型 |
| [docs/ARCHITECTURE_RULES.md](docs/ARCHITECTURE_RULES.md) | 架构约束（依赖方向、禁止操作、模块边界） |
| [docs/CODING_CONVENTIONS.md](docs/CODING_CONVENTIONS.md) | 编码风格、命名规范、日志、测试约定 |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | API 端点一览、认证机制 |
| [docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md) | 开发任务指南、部署配置、性能与安全 |

## 扩展触发条件

以下规则当前不实施，达到阈值时启用：

- **Python 文件 >80 个**：引入 entropy 扫描机制（重复函数、超长函数、TODO 过期）
- **LLM 提供商 >15 个**：拆分 `llm_manager.py` 为 provider 插件架构
