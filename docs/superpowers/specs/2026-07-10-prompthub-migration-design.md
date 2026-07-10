# Fusion PromptHub 正式迁移设计

## 背景

Fusion 当前有 11 个业务 Prompt 通过 `runtime_config_entries` 的 `prompt_template` 命名空间落库，代码常量作为最终回退。PromptHub 已在 dev 运行，但尚无 `fusion` 项目，Fusion 也没有 PromptHub 客户端、内部网络或服务凭证。

本次迁移只覆盖现有 11 个 Runtime Prompt。运行时日期、搜索结果拼装、用户个性化 Prompt 包装、文件上下文包装等动态或安全边界代码继续留在 Fusion，避免把迁移扩大为 Prompt 重写。

## 目标

1. PromptHub 成为 11 个 Runtime Prompt 的版本与发布事实源。
2. Fusion 不在聊天热路径实时请求 PromptHub，而是后台同步完整已发布 bundle 到本地持久化 LKG。
3. PromptHub 不可用、认证失败、返回坏版本或 bundle 不完整时，Fusion 继续使用最后可用 bundle，再回退旧 Runtime Config 和代码默认值。
4. 每次 Agent run 和独立 Prompt 调用可追踪 bundle revision、Prompt slug 和版本。
5. 迁移支持 `disabled -> shadow -> apply` 分阶段切换和快速回滚。

## 非目标

- 不迁移运行时生成的日期 Prompt、搜索结果 opening、用户个性化 Prompt 包装和文件上下文包装。
- 不改变现有 Python `str.format()` 变量语法。
- 不让 PromptHub 成为单次聊天请求的同步强依赖。
- 不把 Runtime Config 页面改成写入控制台。
- 不复用现有 PromptHub admin key，也不向宿主机重新暴露 PromptHub 端口。

## 当前基线与前置治理

- Fusion dev 的 11 条 active Prompt 与代码默认值 checksum 一致。
- PromptHub dev checkout 停在 `13b9525`，GitHub master 为 `a789a7e`；dev 的 `docker-compose.yml` 有未提交的“移除 8200 端口”安全改动。
- PromptHub master 当前被架构检查拦截：API 层直接 `db.flush()`。
- PromptHub 的 `Prompt.content` 与 `current_version` 对应版本内容可能漂移，消费者必须读取 `prompt_versions` 中的 current published 版本。
- PromptHub 与 Fusion 当前没有共同 Docker 网络。
- PromptHub 现有 user API key 是明文且无项目级 scope，不能作为 Fusion 服务凭证。

因此发布顺序必须是：先恢复 PromptHub 门禁与安全读取契约，再导入数据，最后接入 Fusion shadow/apply。

## Prompt 清单与映射

| Fusion key | PromptHub slug | 变量 |
|---|---|---|
| `app_identity` | `app-identity` | 无 |
| `tool_usage_contract` | `tool-usage-contract` | 无 |
| `no_tool_network_boundary` | `no-tool-network-boundary` | 无 |
| `no_vision_file_boundary` | `no-vision-file-boundary` | 无 |
| `url_read_tool_description` | `url-read-tool-description` | 无 |
| `limit_summary` | `limit-summary` | 无 |
| `continuation_system` | `continuation-system` | 无 |
| `generate_title` | `generate-title` | `content` |
| `generate_suggested_questions` | `generate-suggested-questions` | `content` |
| `file_analysis` | `file-analysis` | `query`, `file_content` |
| `file_content_enhancement` | `file-content-enhancement` | `query`, `file_content` |

PromptHub 中统一使用 `format=text`、`template_engine=none`、`tags=[fusion,runtime-config]`、`is_shared=false`。首次导入内容必须来自 Fusion dev 实际 active 值并逐项校验 SHA-256。

## PromptHub 已发布 bundle 契约

新增只读接口：

```http
GET /api/v1/projects/by-slug/{project_slug}/prompts/published
Authorization: Bearer <project-bound service token>
```

响应只从 `prompt_versions` 读取 `current_version` 且状态为 `published` 的内容，按 slug 稳定排序：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "project_id": "uuid",
    "project_slug": "fusion",
    "revision": "sha256",
    "prompts": [
      {
        "id": "uuid",
        "slug": "app-identity",
        "name": "Fusion 应用身份",
        "version": "1.0.0",
        "status": "published",
        "content": "...",
        "variables": [],
        "format": "text",
        "template_engine": "none",
        "published_at": "2026-07-10T00:00:00Z"
      }
    ]
  }
}
```

`revision` 由项目 slug 和排序后的 Prompt slug、version、content checksum、variables 计算。任何 current version 缺失或不是 published 时整包失败，不能静默返回半包。

## 服务凭证

PromptHub 新增 project-bound service token：

- 数据库只保存 SHA-256 token hash，不保存明文。
- 明文 token 强制使用 `phs_` 独立前缀；用户 API 遇到该前缀直接拒绝，数据库日志隐藏绑定参数。
- token 绑定单个 `project_id` 和 `prompts:read` scope。
- service token 只能访问 published bundle；调用任何用户写接口仍返回 401/403。
- Fusion 使用独立随机 key，通过 GitHub Actions Secret 和容器环境注入，禁止写日志、API 响应或仓库。

现有 user key 兼容保留；固定开发 key 从 seed 脚本移除并轮换，不用于 Fusion。

## 内部网络

创建专用 external Docker 网络 `fusion-prompthub`，只连接：

- `fusion-api`
- `prompthub-backend`

Fusion 使用 `PROMPTHUB_BASE_URL=http://prompthub-backend:8000`。不恢复 PromptHub 宿主机端口，不走公网/nginx，也不把 Fusion 加入完整 `ai-audio-network`。

## Fusion 同步与 LKG

配置：

```text
PROMPTHUB_SYNC_MODE=disabled|shadow|apply
PROMPTHUB_BASE_URL=http://prompthub-backend:8000
PROMPTHUB_API_KEY=<secret>
PROMPTHUB_PROJECT_SLUG=fusion
PROMPTHUB_REQUEST_TIMEOUT_SECONDS=3
PROMPTHUB_SYNC_INTERVAL_SECONDS=300
PROMPTHUB_SYNC_ON_STARTUP=true
```

同步流程：

1. 启动后 best-effort 同步，并每 5 分钟轮询。
2. 一次请求获取完整 published bundle。
3. 校验 11 个 slug、变量集合、非空模板、固定 marker 和 checksum。
   对 4 个带变量的模板同时实际执行 Python `str.format()` 契约校验，拒绝未知、缺失、嵌套或非法转换占位符。
4. `shadow` 只比较 PromptHub 与当前有效模板，记录差异，不切换。
5. `apply` 在一个事务中写入并激活 `namespace=prompt_bundle,key=fusion,version=<revision>`，旧 bundle 保留为回滚点。
6. 相同 revision 幂等；多 worker 使用 PostgreSQL advisory lock 避免并发切换。
7. Prompt 读取顺序：active bundle -> 旧 per-key Runtime Config -> 代码 fallback。

`prompt_bundle` 是同步服务专用保留域，通用 admin create/activate/status API 永久只读；`apply` 模式同时禁止新建或激活旧 `prompt_template`，避免绕过 PromptHub 发布事实源。

现有 Prompt getter API 保持不变，只替换底层 resolver。

## 可观测性与管理边界

- Agent run 的 `runtime_config_versions` 增加 `prompt_bundle/fusion=<revision>`。
- 独立标题、推荐问题和文件解析 LLM 调用记录 `prompt_slug`、`prompt_version`、`prompt_revision`，版本不作为高基数 metrics tag。
- admin 只读快照展示同步模式、最近成功时间、revision、差异和最近错误，绝不返回 token。
- `apply` 稳定后，Fusion admin 禁止新建或激活 `prompt_template`，避免双事实源；紧急回滚通过切回 `shadow/disabled` 或重新激活旧 bundle。

## 验收矩阵

| case_id | 场景 | 通过标准 |
|---|---|---|
| `PH-CI-01` | PromptHub 门禁 | 架构、ruff、backend、SDK、Alembic、build 全绿 |
| `PH-AUTH-01` | service token | 正确 key 只能读 fusion bundle；错误 key 401；跨项目 403；写操作 401/403 |
| `PH-BUNDLE-01` | published bundle | 只返回 current published 版本，顺序和 revision 稳定 |
| `PH-BUNDLE-02` | draft/content 漂移 | `Prompt.content` 与 version 不同时仍返回版本表内容 |
| `PH-NET-01` | 内部网络 | Fusion 容器访问 PromptHub 200；宿主机无业务端口 |
| `PH-DATA-01` | 首次导入 | 恰好 11 项，slug/变量/content hash 与 Fusion active 值一致 |
| `FUS-SYNC-01` | shadow | 零漂移且不写 active bundle |
| `FUS-SYNC-02` | apply | 整包原子激活，相同 revision 幂等 |
| `FUS-SYNC-03` | 失败降级 | timeout/401/5xx/坏 bundle 保留 LKG，健康和聊天正常 |
| `FUS-RUN-01` | 热路径 | 单次聊天期间没有 PromptHub HTTP 请求 |
| `FUS-OBS-01` | 版本追踪 | Agent run 和独立 Prompt 调用可定位 revision/version |
| `CHROME-01` | 身份与普通对话 | 新对话行为符合迁移前契约，刷新后保持 |
| `CHROME-02` | 联网工具契约 | 实时问题正常搜索、深读并展示来源 |
| `CHROME-03` | 无工具边界 | 无联网能力模型不会声称已搜索 |
| `CHROME-04` | 异步 Prompt | 标题和推荐问题正常，console error 为 0 |

## 回滚

1. 将 `PROMPTHUB_SYNC_MODE` 切回 `shadow` 或 `disabled`。
2. Fusion 立即回到旧 per-key Runtime Config；不需要等待 PromptHub 恢复。
3. 保留 PromptHub 项目和版本数据，不做破坏性删除。
4. 若 PromptHub 发布坏版本，同步校验应在切换前整包拒绝；必要时在 PromptHub 发布 patch 恢复。
