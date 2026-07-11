# 管理员审计中心 v1 设计

## 目标

为 Fusion 提供独立、只读、可审计的管理员内容观察面：管理员可以检索全部用户与对话，分页查看已持久化消息、usage、Agent run/step、工具安全投影和文件元数据，并查看独立留存的压测汇总。

这不是普通聊天页的管理员模式，也不是数据归档或用户冒充系统。用户删除的数据继续删除，管理员页面不复制聊天正文。

## 核心安全边界

1. 新增独立 `conversation_auditor` 服务端依赖，v1 映射现有 `is_superuser`；普通聊天接口的 `user_id` 所有权语义不改变。
2. 管理内容接口只提供 GET；压测汇总仅提供管理员显式导入，不提供消息修改、重试、续写、停止、删除、导出或冒充用户能力。
3. 列表与详情读取由后端自动写 `admin_audit_events`。敏感详情审计失败时 fail closed，不返回内容。
4. 数据库 JSON 不直接透传。消息、错误、工具参数、工具结果和 run config 统一经过递归脱敏与长度上限。
5. 永不返回文件 `path`、`storage_key`、`thumbnail_key`、`parsed_content`、签名 token、Provider 原始请求响应、完整 resolved prompt 或任何凭据。
6. 所有管理响应使用 `Cache-Control: private, no-store`。`/admin` 页面使用 `frame-ancestors 'none'` 与 `X-Frame-Options: DENY` 防止点击劫持。前端使用隔离的组件局部状态，不写普通聊天 Redux、Dexie、localStorage 或 stream 状态。

## v1 产品范围

### 用户

- 分页列出全部用户。
- 按用户 ID、用户名、昵称、邮箱检索。
- 展示脱敏邮箱、管理员标记、注册/最近活跃时间、对话/消息/工具数量和 token 汇总。
- 用户详情可显示完整邮箱和当前自定义 system prompt，但属于敏感详情，必须单独审计；system prompt 默认折叠。

### 对话

- 按用户、模型、标题/用户关键词、时间、是否包含工具/文件筛选。
- 展示用户摘要、模型、消息/工具/文件数量、最近 Agent 状态、usage 汇总。
- 详情分区分页读取：消息、工具调用、Agent runs/steps、文件元数据。
- 消息只呈现数据库中已持久化的 content blocks；未知 block 保留安全 JSON 视图，不能静默丢失。

### 工具与 Agent

- Agent session 展示状态、模型、provider、步骤数、工具数、耗时、limit reason、脱敏错误与安全 config。
- Agent step 通过 `trace_id == agent_sessions.id` 关联，历史孤立 step 安全忽略并可在诊断计数中体现。
- 已知工具展示字段白名单安全投影；未知工具默认只展示名称、状态、耗时和脱敏标记，不返回原始参数或结果。

### 文件

- v1 只展示仍与会话关联的文件元数据：原始文件名、MIME、大小、状态、尺寸、创建时间。
- 不提供内容、预览、下载或导出。若后续加入图片预览，必须走管理员专用 Bearer 代理和 `no-store` Blob 生命周期，不能复用普通签名 URL。

### 压测记录

- 新增 `performance_runs`，独立保存 runner 脱敏汇总；清理测试聊天后仍可查看。
- 管理员页面通过 JSON 导入完成留存，`run_id` 幂等；v1 不引入长期 ingest secret。
- 不保存压测邮箱、密码、token、消息正文、conversation ID 清单或逐请求 payload。
- 首轮 2026-07-11 生产基线通过管理员页面显式导入。

## 后端架构

```text
app/api/admin_audit.py
  -> app/services/admin_audit_service.py
    -> app/db/admin_audit_repository.py
      -> User / Conversation / Message / ToolCallLog / AgentSession /
         AgentStep / AgentProgressSnapshot / File / AdminAuditEvent /
         PerformanceRun
```

普通 `ChatService`、`ConversationService` 和用户接口不增加管理员旁路。

### 数据模型

`admin_audit_events`：

- `id`
- `admin_user_id`（保留字符串快照，不随用户删除）
- `admin_snapshot`（仅 ID、用户名、脱敏邮箱）
- `action`
- `resource_type`
- `resource_id`
- `target_user_id`
- `request_id`
- `reason`
- `metadata`（仅安全筛选摘要，不含正文）
- `created_at`

`performance_runs`：

- `run_id`（主键）
- `environment`
- `model_id`
- `status`
- `schema_version`
- `safe_summary`
- `imported_by_user_id`
- `started_at`
- `finished_at`
- `created_at`

数据关联迁移采用非破坏策略：为 Agent/消息关键关系增加索引；`agent_steps.trace_id -> agent_sessions.id ON DELETE CASCADE` 在 PostgreSQL 先以 `NOT VALID` 约束落地，阻止新增孤立数据但不删除历史孤立记录。历史清理与约束验证属于单独高风险运维任务。

### API

```text
GET  /api/admin/audit/users
GET  /api/admin/audit/users/{user_id}
GET  /api/admin/audit/conversations
GET  /api/admin/audit/conversations/{conversation_id}
GET  /api/admin/audit/conversations/{conversation_id}/messages
GET  /api/admin/audit/conversations/{conversation_id}/tool-calls
GET  /api/admin/audit/conversations/{conversation_id}/agent-runs
GET  /api/admin/audit/conversations/{conversation_id}/files
GET  /api/admin/audit/events
GET  /api/admin/audit/performance-runs
GET  /api/admin/audit/performance-runs/{run_id}
POST /api/admin/audit/performance-runs/import
```

所有列表使用 `page/page_size`，默认 25、最大 100，排序必须以 ID 作为稳定次级键。消息正文全文搜索不在 v1 使用 JSONB `ILIKE`；后续需要时单独设计 PostgreSQL FTS。

## 前端架构

独立顶层 `/admin`，不加载普通聊天侧栏：

```text
/admin
  ├─ 用户
  ├─ 对话
  │   └─ 详情：消息 / Agent 与工具 / 文件元数据
  ├─ 压测
  └─ 访问审计
```

主要边界：

- `src/lib/api/adminAudit.ts`：唯一 API 入口。
- `src/types/adminAudit.ts`：协议类型。
- `src/components/admin/AdminGuard.tsx`：只负责 UX 保护，真正权限仍由后端决定。
- `src/components/admin/AdminAuditCenter.tsx`：tab、筛选和页面装配。
- `AdminMessageCard` / `AdminExecutionInspector`：专用只读组件，可复用 Markdown、Reasoning 等叶组件，但不能复用 `ChatMessageList`、`ChatMessage`、`UserMessage`、普通文件组件或聊天 hydration。
- 任一 403 立即清空当前敏感详情；离页和登出清空内存状态。

用户头像菜单仅在已确认的管理员 profile 下显示“管理中心”。本地 profile 可伪造，因此页面仍必须处理后端 403。

## 验收矩阵

| 类别 | 必测 |
|---|---|
| 权限 | 未登录 401、普通用户 403、管理员 200、降权/吊销后立即失效 |
| 只读 | 管理内容无 mutation 路由；页面无编辑、重试、发送、停止、删除、导出 |
| 审计 | 列表与详情均记 actor/action/target/request ID；写失败时敏感内容不返回 |
| 脱敏 | 嵌套 secret、Bearer、JWT、cookie、敏感 URL query、错误文本和超长 output |
| 用户 | 空列表、搜索、分页、完整/脱敏邮箱边界、统计与 usage 缺失降级 |
| 对话 | 跨用户列表、组合筛选、空/长对话、相同时间稳定排序、删除后 404 |
| 消息 | 所有已知 block、未知 block、thinking、usage、分页无重复遗漏 |
| Agent/工具 | 多 run、所有终态、孤立 step、未知工具、无 snapshot、脱敏与截断 |
| 文件 | 只返回安全元数据；存储路径、parsed content 和签名 token 永不出现 |
| 前端隔离 | 不读写普通 chat Redux、stream、Dexie；不建 SSE；403 清空详情 |
| 压测 | JSON 校验、幂等导入、清理后汇总仍在、结果不含账号/token/正文 |
| 性能 | 列表查询有界、无 N+1、page_size 上限、长对话分区加载 |
| 生产 | 真实普通用户新数据 + 真实管理员 Chrome，验证列表、详情、刷新、登出、network/console |

## 非目标

- 聊天或文件导出。
- 批量下载、原始文件预览。
- 管理员修改或删除用户内容。
- 恢复已删除对话。
- 消息正文全文搜索。
- Provider 原始报文和完整 PromptHub/system/developer prompt。
- 接管正在运行的 SSE。
