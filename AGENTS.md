# AGENTS.md — fusion-api 导航文件

## 语言

所有回复、注释、提交信息使用中文。Git 格式：`<type>: <中文描述>`，必须包含 Co-Authored-By。

## 快速命令

以下启动命令仅供人工本地开发参考。AI 协作者默认不得启动本地 Fusion 服务；调查、验收优先使用测试、CI、远端 dev 日志/状态和已有运行服务。

```bash
pip install -r requirements-dev.txt      # 安装本地开发与测试依赖
uvicorn main:app --reload                # 开发服务器
python -m pytest test/                   # 运行测试
python -m ruff check . && python -m ruff format --check .  # 代码检查
docker-compose up -d                     # Docker 启动
```

## 架构速览

四层架构，依赖只能向下：API(`app/api/`) → Service(`app/services/`) → AI(`app/ai/`) → Data(`app/db/`)

核心设计：Redis Stream 两段式流，LLM 生成与 HTTP 连接完全解耦。详见 → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## 工作流程

1. **默认执行闭环**：用户说“开始”“继续”“修下”“按你说的来”“提交/部署”等，视为授权继续完成实现、验证、提交、push 和 CI/CD 跟踪；不要为常规下一步反复等待确认。
2. **先定位根因**：bug、线上异常、CI 失败、日志报错必须先读错误、查调用链、对照近期改动和远端证据，确认根因后再改。
3. **复杂变更先计划**：多文件、跨模块、数据流或协议变化先写简短 implementation plan；必要时更新对应 spec/文档。
4. **默认 Subagent-Driven**：适合拆分的开发任务默认用 Subagent-Driven；主 Agent 负责拆分、协调、复核，子 Agent 负责独立实现/审查。若工具额度或任务规模不适合启用，需说明原因并按同等 checklist 自审。
5. **TDD 优先**：bugfix 和行为变更先补能失败的回归测试，再实现；已有测试不足时至少补覆盖核心行为的单测。
6. **编码中**：遵守 [docs/ARCHITECTURE_RULES.md](docs/ARCHITECTURE_RULES.md)，参考 [docs/CODING_CONVENTIONS.md](docs/CODING_CONVENTIONS.md)，保持改动范围最小，不回滚无关用户改动。
7. **禁止默认本地启动**：不得为调查或验收启动 `uvicorn`、本地 Docker 或其他 Fusion 服务；优先使用单元测试、ruff、CI、dev 日志和远端健康检查。只有用户明确要求本地启动时才可执行。
8. **真实 Chrome 回归**：涉及登录态、联网回答、消息流、设置页或其他用户可见链路时，只能复用用户已经打开的、与任务匹配的 Chrome 标签做补充回归。**严禁打开新的 Chrome、新标签、新窗口、`about:blank`、isolated context，严禁使用会创建或切到独立浏览器目标的 DevTools/CDP 通道。**若没有可复用的匹配标签，必须停止并说明阻塞，不能自行新开页面；只验证关键路径、真实网络和可见异常，不用本地 Fusion 服务替代。
9. **变更后验证**：运行与改动匹配的 pytest/ruff；涉及数据流、联网、Redis Stream、认证或部署逻辑时扩大回归范围并更新文档。
10. **CI/CD 收尾**：按正常 Git 流程中文提交并包含 `Co-Authored-By`，push 后持续监控 GitHub Actions 和 dev 部署；失败时拉日志定位并修复。
11. **下一步建议必须查执行记录**：用户问“下一步”“接下来做什么”“还有什么优化”“还能怎么加强”时，必须先读 [docs/EXECUTION_LEDGER.md](docs/EXECUTION_LEDGER.md)，再查 `fusion-api` / `fusion-ui` 最近 `git log` 和相关 `docs/superpowers/specs`。禁止凭 memory 或印象回答，禁止重复建议台账里已经完成的方向。可使用 repo skill `fusion-next-step`。

## Code Review Rules

### 阻塞边界

- 只提交 P0/P1 finding：问题必须由当前 PR 引入、存在当前可达的触发路径，并会造成明确的正确性、安全、权限、数据、兼容性或发布后果；评论必须说明触发条件、实际影响和最小安全路径，证据不足则不报告。
- P2/P3、纯防御性加固、需要未来维护者同时修改规则与测试才成立的假设、测试还可增加更多 fixture、lint/格式/措辞/命名或无当前影响的重构默认不报告，也不得仅因建议有价值就阻塞合并。

### 项目重点

- 重点检查 `API → Service → AI → Data` 单向依赖、后台生成与 HTTP 连接解耦、Redis Stream 断点续读、终态持久化，以及发布/回滚是否保持数据兼容；机械一致性继续交给确定性 CI。

### Few-shot

正例：客户端断线会取消仍在执行的后台生成，导致 Redis Stream 没有终态且刷新后任务永久停留；这是当前可达的数据与用户路径后果，应提交 P1。

反例：还可以为同一 Redis Stream helper 增加另一种参数组合的 fixture，但现有支持路径和失败语义没有错误；这属于 P2/P3 加固，不提交 finding。

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
