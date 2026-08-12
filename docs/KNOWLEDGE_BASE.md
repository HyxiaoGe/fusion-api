# 知识库 v1 运行手册

## 边界与架构

知识库是独立于聊天附件的资源。它只复用对象存储抽象，不复用 `files`、
`conversation_files`、LLM 文件摘要或 Web 进程内的 `asyncio.create_task`。PostgreSQL 保存知识库、
文档、不可变索引版本和持久任务；Milvus 只保存可重建的向量及受控标量字段。

API 进程只负责鉴权、CRUD、上传入队和检索。`python -m scripts.run_knowledge_worker` 是无端口的独立
进程，通过 PostgreSQL `FOR UPDATE SKIP LOCKED` 领取任务，并用随机 lease token 的 SHA-256、
heartbeat、过期回收和终态 fencing 防止旧 Worker 写入新状态。Redis 不承担任务事实源。

文档索引状态依次为 `queued → parsing → chunking → embedding → writing → ready`。解析、Embedding
或 Milvus 写入失败时进入有界重试，耗尽后为 `failed`；用户可调用 retry 创建新的不可变索引版本。
新版本只有在所有向量写入成功且当前 lease 与 `desired_index_version` 同时匹配时才原子激活，旧版本
随后由独立清理任务删除。

## API

所有接口位于 `/api/knowledge-bases`，使用现有 Bearer 认证和统一响应结构。

- `POST/GET /`：创建、分页列出知识库。
- `GET/PATCH/DELETE /{knowledge_base_id}`：详情、更新、异步删除。
- `POST/GET /{knowledge_base_id}/documents`：上传、分页列出文档。
- `GET/DELETE /{knowledge_base_id}/documents/{document_id}`：详情、异步删除。
- `POST /{knowledge_base_id}/documents/{document_id}/retry`：仅重试终态失败文档。
- `POST /{knowledge_base_id}/documents/{document_id}/rebuild`：用当前不可变 Embedding revision 重建。
- `GET /tasks/{task_id}`：本人异步任务状态与稳定错误码。
- `POST /search`：在 1 至 5 个本人知识库中检索，`top_k` 为 1 至 50。

同一知识库内活动文档按 SHA-256 去重，重复内容返回 HTTP 409 和
`KNOWLEDGE_DOCUMENT_DUPLICATE`；跨知识库允许重复。文档完成物理和向量清理后会释放去重键。不存在
和非本人资源统一返回 404。搜索会在返回前用 PostgreSQL 重新验证 owner、知识库、文档 ready 状态、
当前激活版本和 chunk manifest，最终正文与来源也以 PostgreSQL 为准；任一指定知识库不可用时整体失败，
不返回部分结果。

v1 支持 UTF-8 TXT、Markdown、CSV、带文字层的 PDF 和 DOCX，不支持 OCR、扫描 PDF、旧版 DOC、
网页抓取、音视频、图片理解、rerank 或聊天知识库选择器。

## 配置与 fail-closed

完整默认值见 `.env.example`。启用前至少配置：

```dotenv
KNOWLEDGE_BASE_ENABLED=true
KNOWLEDGE_EMBEDDING_PROVIDER=litellm
KNOWLEDGE_EMBEDDING_MODEL=受治理的-embedding-alias
KNOWLEDGE_EMBEDDING_REVISION=该-alias-不可变的发布版本
KNOWLEDGE_EMBEDDING_DIMENSION=1024
KNOWLEDGE_EMBEDDING_ALLOWED_DIMENSIONS=1024
MILVUS_URI=http://fusion-knowledge-milvus:19530
MILVUS_USERNAME=fusion_knowledge
MILVUS_PASSWORD=使用密钥系统注入
MILVUS_DATABASE=fusion_knowledge
MILVUS_COLLECTION_PREFIX=fusion_knowledge_chunks
```

启用时会集中校验 chunk、batch、lease/heartbeat、Embedding profile、COSINE、Milvus URI、数据库和
应用账号；缺项返回稳定 503，Worker 拒绝进入运行态。应用账号不能是 `root`，collection 名由服务端
前缀和允许维度生成，客户端不能指定。

## 本地真实 Milvus 2.6.21 验收

以下命令会拉起官方 standalone 依赖，并仅把 Milvus gRPC/健康端口绑定到 loopback：

```bash
docker compose -f ops/knowledge/milvus-compose.yml up -d
```

root 只用于一次性 bootstrap。应用运行和验收必须使用非 root 账号：

```bash
export MILVUS_URI=http://127.0.0.1:19530
export MILVUS_BOOTSTRAP_PASSWORD=Milvus
export MILVUS_USERNAME=fusion_knowledge
export MILVUS_PASSWORD='替换为满足 Milvus 策略的密码'
export MILVUS_DATABASE=fusion_knowledge
python -m scripts.bootstrap_knowledge_milvus
```

启动 API 与独立 Worker 的组合覆盖：

```bash
docker compose -f docker-compose.yml \
  -f ops/knowledge/fusion-compose.override.yml up -d app knowledge-worker
```

真实验收测试是显式 opt-in，要求服务端版本精确为 2.6.21，执行建 collection、upsert、带 owner/KB/
版本过滤的 search 及异步清理：

```bash
RUN_KNOWLEDGE_MILVUS_INTEGRATION=1 \
KNOWLEDGE_MILVUS_EXPECTED_VERSION=2.6.21 \
python -m unittest test.test_knowledge_milvus_integration
```

## 发布、观测与恢复

`master` 发布流水线使用同一不可变 API 镜像启动 `fusion-api` 和 `fusion-knowledge-worker`，检查二者
镜像引用、内容 ID、健康文件和数据库连接。Worker 与 API 共享原文件挂载并连接私有 Milvus 网络，
不发布端口。第一次发布前在 dev Environment 配置 `KNOWLEDGE_BASE_ENABLED`、Embedding/Milvus
Variables 和 `MILVUS_PASSWORD` Secret；开关默认 false。

`/health` 仍只代表 API 的数据库与 Redis 就绪，不把 Milvus 故障扩散成全部 CRUD 下线。Worker 健康和
知识库检索会独立暴露 Milvus/配置问题。日志不得记录文档正文、向量、密码或 lease token；排障使用
knowledge_base_id、document_id、task_id、index_version、phase 和稳定 error_code。

删除失败不会先删除 PostgreSQL 事实记录；任务会重试并保留可见错误。Embedding profile 或维度变更
必须通过 retry/重建生成新版本，不能原地改 collection schema。镜像回滚不执行 Alembic downgrade；
迁移是 expand-only，旧镜像会忽略新增表。自动回滚会恢复原 API/Adapter/Worker 镜像身份；若原版本
没有知识库 Worker，则删除候选 Worker 容器。
