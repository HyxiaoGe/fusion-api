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

原始对象 key 使用 `.../objects/{checksum}/{document_id}`：checksum 用于内容归组，document_id 隔离每次
上传代际。对象上传前先写入带过期保护的持久 cleanup intent；文档和索引任务提交成功后清除 intent，
配额/唯一冲突的同步删除失败或数据库结果不确定时由独立 Worker 通过租约、引用复查和退避重试收敛。
代际隔离保证旧清理请求即使晚到，也不会删除并发或后续上传的对象。
对象存储暂时断连、超时或限流时，上传接口返回稳定 503 `KNOWLEDGE_STORAGE_UNAVAILABLE`，同时保留
cleanup intent 继续处理不确定的远端写入结果。

v1 支持 UTF-8 TXT、Markdown、CSV、带文字层的 PDF 和 DOCX，不支持 OCR、扫描 PDF、旧版 DOC、
网页抓取、音视频、图片理解、rerank 或聊天知识库选择器。

## 配置与 fail-closed

完整默认值见 `.env.example`。启用前至少配置：

```dotenv
KNOWLEDGE_BASE_ENABLED=true
KNOWLEDGE_CHUNKER_VERSION=chunker-v2
KNOWLEDGE_EMBEDDING_PROVIDER=litellm
KNOWLEDGE_EMBEDDING_MODEL=embedding-v1
KNOWLEDGE_EMBEDDING_REVISION=embedding-r1
KNOWLEDGE_EMBEDDING_REVISION_ROUTES={"embedding-v1@embedding-r1":"embedding-v1-r1-immutable"}
KNOWLEDGE_EMBEDDING_DIMENSION=1024
KNOWLEDGE_EMBEDDING_ALLOWED_DIMENSIONS=1024
KNOWLEDGE_EMBEDDING_BATCH_SIZE=32
KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS=30
KNOWLEDGE_SEARCH_MAX_PROFILES=8
KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT=10000
KNOWLEDGE_WORKER_RETRY_BASE_SECONDS=5
KNOWLEDGE_WORKER_RETRY_MAX_SECONDS=300
MILVUS_URI=http://fusion-knowledge-milvus:19530
MILVUS_USERNAME=fusion_knowledge
MILVUS_PASSWORD=使用密钥系统注入
MILVUS_DATABASE=fusion_knowledge
MILVUS_COLLECTION_PREFIX=fusion_knowledge_chunks
```

`KNOWLEDGE_EMBEDDING_REVISION_ROUTES` 是 append-only JSON registry：键是持久化到知识库和索引版本的
`model@revision`，值是 LiteLLM Proxy 中绑定具体供应商模型版本、禁止原地改指向的不可变 alias。
Embedding 请求会按每个索引版本的持久化键解析 registry 后再调用 Proxy，因此 revision 不只是展示标签。
当前 `model@revision` 必须存在；历史索引版本仍可能用于检索、重试或清理时，对应 route 禁止删除或改指向。
需要轮换模型时添加新 route，并同时更新当前 model/revision 后通过 rebuild 生成新索引版本。
发布 preflight 会把候选 registry 与部署前从 API/Worker 捕获的 registry 对账，任何历史键删除或值改写都会
fail closed；pre-#41 镜像按空 registry 兼容，自动回滚仍恢复部署前完整快照。

当前边界偏好与最小推进算法的不可变标识是 `chunker-v2`。候选部署始终从 Environment Variable 注入
该版本，不继承服务器旧 `.env` 中的 `chunker-v1`；配置为其他值时 preflight fail closed。历史 active
`chunker-v1` 索引仍可检索，因为检索读取其已落库的 manifest 与向量；历史 building v1 任务不能按 v2
算法原地续跑，Worker 会以稳定不支持错误终止并登记该未完成索引版本的清理任务。自动回滚则使用部署前
快照中的版本；对没有知识库环境变量的旧镜像，快照兼容默认仍为 `chunker-v1`。

启用时会集中校验上传上限、chunk 大小/重叠比例/最小步长/单文档总量、batch、搜索 profile 总量、
Worker poll、lease/heartbeat/retry、
Embedding profile/revision route/有限超时、COSINE、Milvus URI/有限超时、数据库和应用账号；缺项返回稳定
503，Worker 拒绝进入运行态。`KNOWLEDGE_MAX_FILE_SIZE` 上限为 50 MiB，Worker poll 上限为 60 秒，
Worker 重试需满足 `0 < base <= max <= 3600 秒`，Embedding 超时上限为 120 秒，Milvus 超时上限为
60 秒。每用户知识库数量限制为 1 到 1000，每知识库文档数量限制为 1 到 10000；单次 Embedding batch
限制为 1 到 128；单文档 chunk 总量限制为 1 到 10000；搜索最多允许
1 到 16 个 profile，默认 8，超过配置上限会在外部调用前以稳定 503 fail closed。搜索路由总超时按
`ceil(max_profiles / 4) × (Embedding 超时 + 2 × Milvus 超时) + 5 秒` 计算，覆盖每个 profile 冷路径的
Milvus client 构造与 search 两次有限超时以及全部并发批次。应用账号不能是
`root`，collection 名由服务端
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
Variables（包括 `KNOWLEDGE_CHUNKER_VERSION=chunker-v2`、完整的
`KNOWLEDGE_EMBEDDING_REVISION_ROUTES`、`KNOWLEDGE_SEARCH_MAX_PROFILES`、
`KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT`）和 `MILVUS_PASSWORD` Secret；开关默认
false。自动回滚快照会保留当时的完整 revision registry，使旧索引仍能路由到部署前的不可变 alias。

`/health` 仍只代表 API 的数据库与 Redis 就绪，不把 Milvus 故障扩散成全部 CRUD 下线。Worker 健康和
知识库检索会独立暴露 Milvus/配置问题。日志不得记录文档正文、向量、密码或 lease token；排障使用
knowledge_base_id、document_id、task_id、index_version、phase 和稳定 error_code。

删除失败不会先删除 PostgreSQL 事实记录；任务会重试并保留可见错误。Embedding profile 或维度变更
必须通过 retry/重建生成新版本，不能原地改 collection schema。镜像回滚不执行 Alembic downgrade；
迁移是 expand-only，旧镜像会忽略新增表。自动回滚会恢复原 API/Adapter/Worker 镜像身份；若原版本
没有知识库 Worker，则删除候选 Worker 容器。
