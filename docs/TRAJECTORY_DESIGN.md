# Fusion Trajectory 集成设计（DSH 风格 Agent 执行轨迹）

> 状态：设计稿 v0.11（终审通过；P0 实施预检补充稳定 turn 分组键）
> 范围：fusion-api + fusion-ui 集成 Agent 执行轨迹展示；langchain-trajectory-mvp 降级为原型实验室与联调夹具。
> 本文档是实施 P0 的正式依据；P0 已在独立实施分支开始，P1–P3 不在本轮范围内。
>
> **仓库处理决定**：正式实施分支通过 `.gitignore` 的 `!docs/TRAJECTORY_DESIGN.md` 例外跟踪本文档。

## 1. 背景与目标

### 1.1 背景

- fusion 已有自实现 Agent 运行时（零 LangChain）：多轮对话、工具调用、计划协调、推理流、上下文管理，并已发出 21 种 `agent_event`（`run/step/tool` 三级层级、`sequence`、`trace_id`）。
- 但这些事件经 `AgentEventCompositeWriter` 写入 Redis Stream，**带 TTL 即蒸发**（进行中 600s / 结束后 60s，见 `app/core/redis.py`），不可回放、不可查询、不可刷新恢复。
- 目标对标 DeepSeek Harness 的 Trajectory：在 fusion-ui 中提供独立的「Agent 执行轨迹」视图——按轮次组织的账本、Overview 时间线、逐条检查器，与聊天双向定位。
- `langchain-trajectory-mvp` 已验证「追加账本 + SSE 补发 + 投影纪律 + 瀑布图 UI」三层能力；其数据源是演示链，不适合作为 fusion 的数据源。**MVP 转型为：① UI 原型实验室（快速试交互）；② P2 阶段的联调夹具。**

### 1.2 目标形态

```
对话页（现有聊天原样保留）
  ├─ 聊天流：plan 卡片 / 工具卡片 / 推理 / 回答（低噪声用户视图，AgentRunTimeline 不动）
  └─ 轨迹侧栏（新增，独立调试视图，可折叠 Tab）
       ├─ 轮次账本：conversation → message → run attempt → step → tool/llm round
       │    #N 索引 / 类型标记 / 单行摘要 / 耗时 / 状态
       ├─ Overview 时间线：真实耗时比例投影，拖选聚焦、缩放、hover 精确计时
       ├─ 检查器：输入/输出/错误/工具 schema/证据/token（TTFT、cache、thinking）
       └─ 聊天 ↔ 轨迹双向定位（hover 工具调用 → 轨迹节点，反之亦然）
```

### 1.3 设计硬约束（评审结论，不可放宽）

1. **TrajectoryRecorder 必须使用独立数据库 Session**，不复用 Agent Loop 的业务 Session；不得对请求 Session 执行 commit/rollback。
2. **账本与投影分离**：`TrajectoryRecorder` 只做不可变追加，投影由查询侧 `TrajectoryProjector` 完成；`AgentProgressRecorder` 是写时折叠快照，三者是同一个事件协议的不同消费者。
3. **`trajectory_status` 独立于 `AgentSession.status`**：轨迹观测缺失（`recording/complete/degraded/legacy`）与模型回答不完整（`incomplete`）语义必须分离。
4. **账本完成边界 = emitter seal + recorder finalize**：`run_completed` 不是最后一条事件；由 Emitter 在同一把锁内 `seal_and_get_last_sequence()` 封口并返回 `_sequence - 1`，再由编排层调用 `finalize(expected_last_sequence)`。同步 v1 称 **commit/completeness barrier**；异步队列的 flush barrier 仅作后续升级方案。
5. **`schema_version` 提升到公共 Event Envelope**，所有 `agent_event` 统一携带；部署窗口内兼容缺失版本。
6. **安全分级是服务端契约**：新增生命周期事件必须天然是用户安全数据；落库为**脱敏事件账本**，禁止依赖前端隐藏字段实现权限。
7. **首屏延迟有界**：同步 v1 承认并限制延迟回归（**不承诺零影响**）——同步落账允许 §9.1 阈值内的有界回归；Recorder 超时/熔断必须有**可执行机制**（fusion 是同步 SQLAlchemy + psycopg2：`wait_for` 无法中断阻塞调用、`connect_timeout` 限整数秒、超时不得取消底层任务防 permit 泄漏、`CancelledError` 必须显式捕获，见 §5.1）；`first_output_delta` 按 delta 类型分别定义发送顺序（§3.3）；实测超阈值则 P0 切换 §5.4 异步队列（§2.1、§5.1）。

## 2. 总架构与数据流

### 2.1 事件流（写侧，改造后）

```
AgentEventEmitter（锁内分配 sequence + ts，唯一发送方；seal 封口）
        ↓ await
AgentEventCompositeWriter
  ├─ Redis Stream（AgentEventRedisWriter）→ 实时 SSE（现有）
  ├─ AgentProgressRecorder（现有）→ agent_progress_snapshots（写时折叠）
  └─ TrajectoryRecorder（新增）→ agent_events（只追加，独立 Session）
```

**sink 失败语义分类**（必须显式区分，不得统一吞异常）：

| sink | 类别 | 失败语义 |
|---|---|---|
| Redis：权威 `plan_snapshot` | required | **fail-closed**：现有语义保持——重试后仍失败则抛 `StreamWriteUnavailableError` 并终止生成（`tool_executor.py`）。CompositeWriter 不得吞掉该异常 |
| Redis：非权威事件（其余 20 种） | auxiliary | fail-open：记日志，不阻塞 |
| AgentProgressRecorder | auxiliary | fail-open：记日志，不影响主链路（现有行为保持） |
| TrajectoryRecorder | auxiliary | 对主链路 fail-open；但对**轨迹可信度** fail-closed：失败（含取消）必须反映到 `trajectory_status = degraded`（见 §5.3） |

**首屏延迟边界（硬约束 #7）**：同步 v1 位于 Emitter 的顺序 await 路径（`emitter._emit` 在锁内 await writer → CompositeWriter 顺序 await Redis + 同步 Recorder），因此 `run_started / step_started / llm_round_started` 等首个输出前事件的数据库提交**必然计入端到端 TTFT**——Redis 先写只保证顺序，不实现异步解耦。本文档**不承诺零延迟影响**，而是：

- 承认并限制有界回归：同步落账允许 §9.1 阈值内的延迟回归；
- Recorder 超时/熔断采用**可执行机制**（§5.1：有界线程池 + 非阻塞准入闸门 + `wait_for` + `shield` + `CancelledError` 分支 + 数据库级超时），失败 fail-open 并标 degraded；
- `first_output_delta` 的发送顺序按 delta 类型分别定义（§3.3）；
- 实测超过 §9.1 阈值 → P0 切换 §5.4 异步队列（切换是既定路径，不是可选优化）。

顺序保证：三个 sink 在 emitter 的 `asyncio.Lock` 内顺序执行，Redis、折叠快照、账本对同一 run 的事件顺序一致；sequence 从 **0** 开始，范围 `0..N-1`，**单调递增、永不复用**。

### 2.2 查询流（读侧，P1）

```
Trajectory API（GET /api/conversations/{id}/runs/{run_id}/trajectory）—— 仅历史快照，不声称实时
        ↓ 读取 agent_events（有序，LIMIT+1 截断保护）
TrajectoryProjector（查询时 O(n) 纯函数投影，不落冗余 span 数据）
        ↓
Trajectory 快照 JSON（事件 + span/record 投影 + 完整性状态 + truncated，按用户/诊断分级）
        ↓
fusion-ui 轨迹侧栏（运行中实时归并 = P3，复用现有聊天 SSE）
```

### 2.3 数据关系（一个协议，三个消费者）

```
agent_event 原始协议（schema_version 统一版本）
    ├─ AgentProgressRecorder → 写时折叠的实时快照（现有，用户进度视图）
    ├─ TrajectoryRecorder    → 只追加脱敏事件账本（新增，真相视图）
    └─ TrajectoryProjector   → 查询时生成 span / record / waterfall（新增）
```

- 原始事件不可变：账本只追加；任何视图都是投影，不反向修改事件。
- Schema 校验只在 Emitter 创建 Event Model 时执行一次；落库保存已序列化（已脱敏）结果；历史读取时按 `schema_version` 做兼容性校验；缺失版本的事件按旧协议解释并展示为兼容记录。

## 3. 事件协议（P0 核心）

### 3.1 Envelope 升级

`AgentEventBase`（`app/services/agent/events.py`）统一增加：

```text
schema_version: int   # 协议版本，本次定 1
```

- `protocol_version` 保留给 progress 类事件做兼容，新协议以 `schema_version` 为准。
- **部署窗口兼容**：前端对缺失 `schema_version` 的事件按旧协议解释，不因字段缺失崩溃；历史 run（账本建立前）展示为 `legacy/not_recorded`。

### 3.2 事件分类：生命周期 vs Annotation

投影器只对**生命周期事件**建 span；annotation 事件作为附加信息附着到对应 span 或单独展示。

| 分类 | 事件 | 投影行为 |
|---|---|---|
| 生命周期 | `run_started / run_completed / run_failed / run_limit_reached / run_interrupted` | run 级 span（轮次）；**`run_completed` 只关闭 run span，不触发账本 finalize**（见 §5.2） |
| 生命周期 | `step_started / step_completed` | step 级 span |
| 生命周期 | `tool_call_started / tool_call_completed` | tool 级 span（`tool_call_delta` 为流式 annotation） |
| **新增** | `llm_round_started / llm_round_first_output_delta / llm_round_completed / llm_round_failed / llm_round_cancelled` | llm 级 span（TTFT = first_output_delta − started；tokens/cache 在 completed） |
| **新增** | `retrieval_started / retrieval_completed / retrieval_failed / retrieval_cancelled` | 知识库检索 span |
| **新增** | `tool_attempt_started / tool_attempt_completed` | 工具重试的 attempt 子 span（`attempt_index` 归组） |
| Annotation | `run_progress_updated / plan_snapshot / plan_step_updated` | 计划快照区段（plan 卡片，不建 span） |
| Annotation | `tool_result_digest` | 附着到对应 tool span |
| Annotation | `evidence_item_upserted` | 证据区段 |
| Annotation | `content_block_upserted / content_block_discarded` | 消息组装块 |
| Annotation | `context_status_updated` | 上下文管理区段（含 `removed_turns/removed_messages/removed_tool_transactions`——即 compaction 数据） |
| Annotation | `context_required / context_result` | 上下文握手 |
| Annotation | `suggested_questions_pending` | run 后异步辅助事件；计入账本 sequence 完整性范围（见 §5.2） |

### 3.3 新增生命周期事件的完整 Schema（P0 可实施定义）

所有新增事件复用 `AgentEventBase` 公共字段（`run_id / step_id / parent_step_id / sequence / trace_id / ts / schema_version`），并满足**用户安全**要求（§7.3：不携带 prompt、完整输入输出、工具 schema 原文）。

**LLM Round（`kind=llm`）**

```text
llm_round_started:
  llm_round_id: str          # 本轮 LLM 调用唯一 id（同一次 round 内各事件共享）
  round_index: int           # run 内第几个 LLM round（1-based）
  model: str                 # 模型 id（已脱敏，如 deepseek/deepseek-chat）
  provider: str
  parent_step_id: str | null

llm_round_first_output_delta:
  llm_round_id: str
  delta_kind: Literal["reasoning" | "content" | "tool_call"]
  ttft_ms: int               # 非空：仅当真实 delta 到达时才发送本事件
```

> **「首 token」语义与发送顺序（已通过终审，按 delta 类型分两条状态机）**：
>
> **通用**：只有真实 reasoning/content/tool-call delta 到达时才发送本事件，`ttft_ms` 非空；空流不发送，由 `llm_round_completed.ttft_ms = null` 表达；现有 `_has_text_delta` 只识别 reasoning/content（已核验），`delta_kind=tool_call` 需要扩展检测逻辑——属 P0 新能力。
>
> **reasoning / content 分支（受顺序约束）**：
> ```text
> 1. 记录模型 delta 到达时间（ttft 基准）
> 2. await 写入可见 Redis chunk（现有 llm 流式渠道 append_chunk）
> 3. 编排层 emit llm_round_first_output_delta（经 emitter → CompositeWriter → Redis + 账本）
> ```
> 顺序可测试：断言 Redis Stream 中可见 chunk 条目先于 `first_output_delta` 条目。
>
> **tool_call 分支（不受顺序约束）**：tool-call-only 轮次**没有用户可见文本 chunk**，检测到首个 tool_call delta 即 emit（TTFT 为模型侧测量），不要求先写任何可见 chunk；文档明确该分支独立于首屏约束（工具调用无首屏延迟敏感度）。

```text
llm_round_completed:
  llm_round_id: str
  status: Literal["success"]
  finish_reason: str | null  # stop / length / content_filter ...
  input_tokens / output_tokens / total_tokens: int
  cache_read_tokens / cache_write_tokens: int | null   # P0 新能力
  ttft_ms: int | null        # 空流时为 null；有 delta 时 = first_output_delta.ttft_ms
  duration_ms: int

llm_round_failed:
  llm_round_id: str
  status: Literal["failed"]
  error_code: str | null     # 如 rate_limit / timeout / provider_error
  message: str | null        # 已脱敏错误摘要（不含密钥/URL 参数）

llm_round_cancelled:
  llm_round_id: str
  status: Literal["cancelled"]
  reason: Literal["user_cancelled" | "superseded" | "shutdown"]
```

> **cache token 是 P0 新能力，不是复用现有测量**：现有 `Usage`（`app/schemas/chat.py`）只有 `input/output/context`，**无 cache 字段**（已核验）。`cache_read_tokens / cache_write_tokens` 需要新增异步提取逻辑（从供应商 usage 扩展字段读取）与协议字段，单列为 P0 新能力交付；提取失败降级为 null，不报错。

**知识库检索（`kind=retrieval`）**

```text
retrieval_started:
  retrieval_id: str
  query_summary: str | null  # 已脱敏查询摘要（≤120 字符），不存全文
  parent_step_id: str | null

retrieval_completed:
  retrieval_id: str
  status: Literal["success"]
  document_count: int
  duration_ms: int

retrieval_failed:
  retrieval_id: str
  status: Literal["failed"]
  error_code: str | null
  message: str | null

retrieval_cancelled:
  retrieval_id: str
  status: Literal["cancelled"]
  reason: Literal["user_cancelled" | "superseded" | "shutdown"]
```

**工具重试 attempt（`kind=tool_attempt`）**

```text
tool_attempt_started:
  tool_attempt_id: str
  tool_call_id: str          # 关联到所属 tool_call（同一 tool_call 可有多个 attempt）
  tool_name: str
  attempt_index: int         # 1-based，同 tool_call 下递增

tool_attempt_completed:
  tool_attempt_id: str
  status: Literal["success" | "failed" | "cancelled" | "timeout"]
  error_code: str | null
  duration_ms: int
```

**Orphan / 未正常收口规则**（投影器强制语义）：

- run 进入终态时，任何 `started` 未配 `completed/failed/cancelled` 的 span（llm round / retrieval / tool attempt / step）由投影器**按 run 终态推导关闭**，且推导结果必须标注来源：
  - 推导出的 span 携带 `terminal_source: "inferred"` + `inferred_reason`（`run_completed_without_close` / `run_failed_without_close` / `run_interrupted_without_close`）；由真实事件关闭的 span 携带 `terminal_source: "recorded"`。
  - **成功 run 中的孤儿标 `unknown/incomplete`（reason=run_completed_without_close），不得伪装成真实 `cancelled`**；失败 run 中的孤儿标 `failed`；中断 run 中的孤儿标 `cancelled`（reason=run_interrupted_without_close）。
- 禁止出现无终态、无 `terminal_source` 的悬挂 span。

### 3.4 层级模型（run attempt 语义与并发分配）

```
conversation → message/turn → run attempt → step → tool / llm round
```

- **不是「一条消息 = 一个 run」**：retry、continue、重新生成会让同一消息产生多个 run attempt。
- **实施预检修订（v0.11）**：现有 `AgentSession.message_id` 是 assistant 消息锚点，已被消息展示、审计、继续生成等路径使用，不能改成 user turn id；而“首次生成失败后重试”会预分配新的 assistant id。为避免同一 user turn 被拆成多个 attempt 组，`AgentSession` 增加三列，`message_id` 继续保持现有语义：

```text
turn_message_id: str | null  # 稳定 user turn 分组键；新 run 写 user message id
previous_run_id: str | null   # 被接续的 run id；初始 run 为 NULL
attempt_index: int            # 同一 message 下 run 的序号，1-based
```

- 新 run 的唯一 attempt 组键是 `turn_message_id`；`message_id` 仍指向本次 assistant 消息，用于现有 UI/审计/消息恢复消费者。
- 历史行无法无歧义反推 user turn，迁移时令 `turn_message_id = message_id`；不伪造跨 assistant id 的历史 lineage。

赋值规则与来源（每个入口在创建 run 时**显式传入被接续的 run_id**，不得事后推断）：

| 场景 | previous_run_id | attempt_index | 来源 |
|---|---|---|---|
| initial（消息首次执行） | NULL | 1 | 无 |
| retry（同消息重试） | 被重试的 run_id | 同消息下递增 | 请求携带（两阶段兼容，见下） |
| continue（继续生成，`agent_loop_continuation`） | 被接续的 run_id | 同消息下递增 | continue 请求已支持 `previous_run_id` |
| regenerate（重新生成，取代旧回答） | 被取代的 run_id | 同消息下递增 | 请求携带（两阶段兼容，见下） |

**retry/regenerate 的 previous_run_id 两阶段兼容**：

- 现状：`ChatRequest` 只有 `retry_user_message_id / retry_assistant_message_id`（按消息 ID 重试），**无 `previous_run_id`**；仅 continue 路径已支持（已核验 `app/schemas/chat.py`）。
- **阶段 A（兼容期）**：后端 `ChatRequest` 先增加 optional `previous_run_id`；前端从 assistant 消息的 `agent_run.run_id` 发送。请求缺失时，**在同一个 `SELECT FOR UPDATE` 事务内按 `turn_message_id` 解析最新 run** 作为 previous，并记录兼容性指标（统计缺失率）。
- **阶段 B（收紧期）**：缺失率稳定后，retry/regenerate 改为必填，删除事务内兜底解析。
- 分阶段发布期间旧前端不因新增必填字段断约。

**并发分配规则**：

- 建 `UNIQUE(turn_message_id, attempt_index)` **部分唯一约束**（`turn_message_id IS NOT NULL AND attempt_index IS NOT NULL`）。
- attempt_index 分配必须**原子**：锁定所属 conversation 行（assistant 消息可能尚未持久化，不能只锁 message 行）后，按 `turn_message_id` 取当前最大 attempt_index + 1，再插入 AgentSession；禁止「读-算-写」非原子路径。
- `previous_run_id` 自引用外键 `agent_sessions(id)`，删除语义 `ON DELETE SET NULL`（被接续的 run 删除时不得级联删除新 run）。
- 同 turn 并发 retry/continue 由唯一约束兜底，冲突时重试分配。

### 3.5 层级约束：LLM 观测不得反向依赖 emitter

- `LLMRoundObservation` 位于 `app/ai/`，**架构规则禁止 `app/ai` import `app/services`**（含 emitter）。LLM round 事件不能由观测对象直接发出。
- 正确方式：**Service 编排层**（`app/services/stream/` 的 llm 调用路径）把「观测 → 事件」回调注入观测对象，或消费观测结果后在编排层调用 emitter 发出 `llm_round_*` 事件。观测对象只负责测量（started_at / first_output_delta / tokens），不负责发事件。
- 工具 attempt、检索事件同理：由各自的 Service 执行路径发出，不在底层适配器内发。

## 4. 数据模型（P0）

### 4.1 agent_events 表（脱敏追加账本）

```sql
CREATE TABLE agent_events (
    event_id        UUID PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    message_id      TEXT,
    run_id          TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
    sequence        INTEGER NOT NULL,          -- 0-based，范围 0..N-1，单调递增永不复用
    event_type      TEXT NOT NULL,
    schema_version  INTEGER NOT NULL DEFAULT 1,
    step_id         TEXT,
    tool_call_id    TEXT,
    parent_step_id  TEXT,
    trace_id        TEXT,
    event_ts        TIMESTAMPTZ NOT NULL,      -- 事件发生时间（emitter 分配）
    recorded_at     TIMESTAMPTZ NOT NULL,      -- 落账时间
    payload         JSONB NOT NULL,            -- 已脱敏、已截断的完整事件体（脱敏事件账本，非原始事件）
    UNIQUE (run_id, sequence)
);
CREATE INDEX ix_agent_events_conversation_ts ON agent_events (conversation_id, event_ts);
CREATE INDEX ix_agent_events_run ON agent_events (run_id);
```

- 外键：`run_id → agent_sessions.id ON DELETE CASCADE`、`conversation_id → conversations.id ON DELETE CASCADE`——随会话/run 删除级联清理，不留孤儿数据。
- `payload` 落库前经**按事件类型的 allowlist 脱敏**（见 §7.3），非通用 sanitizer。
- `recorded_at` 保留列；**TTL 清理索引（`recorded_at`）推迟到 TTL 方案实施时再建**（见 §4.3）。

### 4.2 run 级轨迹完整性（独立于 AgentSession.status）

```sql
CREATE TABLE run_trajectory_meta (
    run_id                  TEXT PRIMARY KEY REFERENCES agent_sessions(id) ON DELETE CASCADE,
    conversation_id         TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    message_id              TEXT,
    trajectory_status       TEXT NOT NULL,  -- 'recording' | 'complete' | 'degraded' | 'legacy'
    event_count             INTEGER NOT NULL DEFAULT 0,
    expected_last_sequence  INTEGER,        -- seal 封口时的 _sequence - 1；finalize 依据
    first_event_ts          TIMESTAMPTZ,
    last_event_ts           TIMESTAMPTZ,
    finalized_at            TIMESTAMPTZ,    -- 仅 complete 时有值
    degraded_reason         TEXT,           -- 仅 degraded 时有值
    updated_at              TIMESTAMPTZ NOT NULL
);

CREATE TABLE trajectory_ledger_settings (
    singleton_key           TEXT PRIMARY KEY,          -- 固定为 'default'
    ledger_enabled_at       TIMESTAMPTZ NOT NULL,      -- 各环境迁移实际执行水位
    created_at              TIMESTAMPTZ NOT NULL
);
```

状态机与语义：

| 状态 | 含义 | 进入条件 |
|---|---|---|
| `recording` | 账本写入中 | run 开始记录事件 |
| `complete` | 账本完整 | **显式 seal + finalize 成功**：degraded latch 未置位 **且** 数据库校验 `COUNT = expected+1 ∧ MIN(sequence)=0 ∧ MAX(sequence)=expected`（§5.2），且 `finalized_at` 落库 |
| `degraded` | 观测缺失 | **latch 置位（含取消/迟到事务场景）** / 任意落账失败 / 超时熔断 / 准入满载 / stale 协调发现缺尾部事件 / 新 run 缺 meta（`meta_missing`） |
| `legacy` | 账本建立前的历史 run | `run.created_at < ledger_enabled_at` 且无 meta（**仅此一种判据**，见 §5.3） |

**只有显式 seal/finalize 成功才能进入 complete，且 latch 置位时 finalize 一律落 degraded**（§5.2）；stale recording 无持久化 `expected_last_sequence` 时只能保守标 `degraded`，不得猜测 complete。

### 4.3 保留策略（P0 范围决策）

- **P0 只做随会话 CASCADE，不做独立 TTL**。理由：fusion 当前没有消息级 TTL；只删 events 保留 meta 会出现「状态 complete 但事件为空」的假象，删 meta 又会被误判为 degraded/legacy——独立 TTL 的正确语义（原子删除 events+meta 并引入 `expired` 状态）是后续工作，不塞进 P0。
- 后续需要 TTL 时：基于 `recorded_at` 的清理 Worker（挂 `scheduler_service` 体系）原子删除 events 与对应 meta，并增加 `expired` 状态与清理索引；清理失败按审计留存语义处理，不静默。
- 脱敏原则：prompt、完整输入输出、工具 schema **不入账本**；content blocks / 证据全文只存摘要 + 引用，详情按需从消息/证据库读取。

### 4.4 AgentSession 迁移与回填（P0 必做）

现有 `agent_sessions` 已有历史数据且 `message_id` 可空，按 Alembic expand/backfill/contract 顺序执行：

1. **expand**：`ALTER TABLE agent_sessions ADD COLUMN turn_message_id TEXT NULL`、`ADD COLUMN previous_run_id TEXT NULL`、`ADD COLUMN attempt_index INTEGER NULL`（均可空，不动存量行）。
2. **backfill**：
   - `turn_message_id = message_id`：历史行仅保留可证实的 assistant 锚点，不跨 assistant id 推断 user turn；
   - `attempt_index`：按 `(turn_message_id, created_at, id)` 用窗口函数 `row_number()` 回填，同一历史 assistant 锚点内的 run 得到稳定序号；
   - `message_id IS NULL` 的历史行：`turn_message_id / attempt_index` 保持 NULL（不参与 turn 归组，部分唯一索引不受影响）；
   - **历史 `previous_run_id` 保持 NULL，不伪造 lineage**（无法可靠推断，宁可缺失）。
3. **contract**：
   - 建部分唯一索引 `UNIQUE (turn_message_id, attempt_index) WHERE turn_message_id IS NOT NULL AND attempt_index IS NOT NULL`；
   - 新数据约束由应用层强制（新建 run 必须写 `attempt_index`；`previous_run_id` 由两阶段兼容控制），数据库层不强制 NOT NULL（避免迁移期间断约）。
4. **`ledger_enabled_at` 不得是源码配置常量**：P0 新建单例表 `trajectory_ledger_settings` 持久化各环境的实际启用时间；legacy 判定只读取该不可变水位。不同环境（dev/prod）各自由迁移记录，运行时配置管理不得覆盖。

## 5. 采集层：TrajectoryRecorder（P0）

### 5.1 v1：同步写入（独立 Session + 原子事务 + 可执行超时熔断 + 取消分支）

- 新增 `app/services/agent/trajectory_recorder.py`，实现与 `AgentProgressRecorder` 同构的 `record_chunk(conversation_id, chunk_type, payload)`。
- **独立短生命周期 Session**：每次写入自己开 session、自己 commit、自己 rollback；绝不触碰请求 Session；以后改后台队列也不得把请求 Session 传入后台任务。
- **事件写入与 meta 计数是同一个原子事务**——单 Session、单事务：

```text
begin
  幂等创建 run_trajectory_meta（INSERT ... ON CONFLICT DO NOTHING，状态 recording）
  INSERT agent_event
  仅当 INSERT 实际成功时：UPDATE meta SET event_count = event_count + 1,
      first_event_ts = COALESCE(first_event_ts, :ts), last_event_ts = :ts
commit
```

任一步失败整体 rollback；**禁止「先插 event 再单独计数」的非原子路径**。
- 追加语义：只 INSERT `agent_events` 与维护 meta 计数，不做任何投影。
- 接入点：`AgentEventCompositeWriter` 增加 `trajectory_recorder` 可选 sink（与现有 `recorder` 并列）；CompositeWriter 不吞 Redis 权威事件异常（§2.1）。

**超时/熔断的可执行机制 + 取消分支（终审定稿）**：

- **前提事实**：fusion 数据库层是**同步 SQLAlchemy**（`create_engine` + `sessionmaker`，已核验 `app/db/database.py`），驱动为 **psycopg2-binary 2.9.9**（已核验）——同步调用阻塞事件循环，`asyncio.wait_for` 无法中断阻塞调用；且 psycopg2 的 `connect_timeout` 只接受**整数秒**，**无法配置 500ms**。因此主链路隔离不依赖 connect 超时：
  1. **有界线程池 + 非阻塞准入闸门（BoundedSemaphore）**：进程级 `ThreadPoolExecutor(max_workers=4)` **必须配 `BoundedSemaphore(4)` 做提交准入**——`sem.acquire(blocking=False)` 成功才提交任务；**仅凭 `max_workers` 无法实现「满载降级」：线程池任务队列本身无界（只限制工作线程数）**；准入失败（semaphore 满）→ 直接 fail-open（置 degraded latch），不排队、不阻塞事件循环；
  2. **主链路 250ms 隔离 = `asyncio.wait_for(0.25)` + 准入闸门**（进程内机制，与 DB 无关）；**超时不得取消底层任务（permit 泄漏防护）**：`wait_for` 直接包装 executor future 时，超时可能取消**尚未开始执行**的任务——worker 的 `finally` 不会运行，permit 永久泄漏。必须用 `asyncio.shield` 包裹；
  3. **`CancelledError` 必须显式捕获（终审定稿）**：`asyncio.CancelledError` 继承 `BaseException`，**不会进入 `except Exception`**——用户取消或 shutdown 时会漏掉。取消路径必须：置 degraded latch → 消费迟到 future → 传播取消。

```python
# v1 落账的固定形态（permit 生命周期四规则 + 取消分支）
async def _record(run_id: str, payload: dict) -> None:
    if not _sem.acquire(blocking=False):                 # ① 准入失败 → fail-open
        mark_degraded(run_id, "admission_full"); return
    try:
        future = asyncio.get_running_loop().run_in_executor(
            _executor, _db_worker, run_id, payload)      # _db_worker 的 finally 里 _sem.release()
    except Exception:                                    # ② submit 失败（executor 关闭等）→ 调用方立即释放
        _sem.release()
        mark_degraded(run_id, "write_failed"); return
    try:
        await asyncio.wait_for(asyncio.shield(future), 0.25)   # ③ 超时不取消底层任务
    except asyncio.CancelledError:                       # ④ 取消分支（终审定稿）：CancelledError 继承 BaseException
        mark_degraded(run_id, "recorder_cancelled")
        _consume_late(future)                            #    迟到任务继续跑，finally 释放 permit；异常被消费
        raise                                            #    必须重新抛出，不得吞掉取消
    except asyncio.TimeoutError:
        mark_degraded(run_id, "recorder_timeout")
        _consume_late(future)                            #    迟到结果/异常必须被消费（防 never-retrieved）
    except Exception:
        mark_degraded(run_id, "write_failed")
```

**permit 生命周期四规则**：
- `_sem.release()` 只在 `_db_worker` 的 `finally` 中执行（正常 / 异常 / 迟到 / 取消后迟到路径统一）；
- `submit` 失败（executor 已关闭等）→ 调用方**立即释放**，不进入 worker；
- 超时/取消**不取消底层任务**（`shield`），迟到任务继续执行、其 `finally` 释放 permit；
- 迟到任务的异常必须被消费（取 exception 或后台 gather 兜底），避免「exception was never retrieved」；**迟到成功也不得反转 degraded latch**（§5.2）。

**取消后的 sequence 语义（终审定稿）**：
- emitter 的 `_sequence` 单调递增；被取消事件**已分配的 sequence 永不复用**——后续事件（含终态事件）使用严格更大的序号；
- 若被取消事件未落库：账本出现空洞，finalize 三断言 `MAX/COUNT` 不匹配 → `degraded`；
- 若被取消事件已落库（写入完成但调用被取消）：latch 已置位（`recorder_cancelled`），finalize latch 优先 → `degraded`；
- 两条路径都收敛到 `degraded`，**不会错标 complete，也不会出现 sequence 重复**（唯一约束 `(run_id, sequence)` 兜底）。

- 数据库侧配置可用的超时：`statement_timeout=200`（ms，整数，经连接 options 下发）、`lock_timeout=100`（ms，整数）、`connect_timeout=1`（s，psycopg2 最小粒度，仅作连接兜底，**非主隔离手段**）。
- **备选方案（决策点）**：真正 `AsyncSession`（asyncpg/psycopg3 async 驱动）。若线程池方案实测不达 §9.1 阈值，P0 记录决策并切换；不默认采用（fusion 现有层为同步 Session，切换成本高）。
- **首屏边界**：sink 顺序保持 Redis 先于本 Recorder（§2.1 硬约束 #7）；`first_output_delta` 的 reasoning/content 分支在可见 chunk 写 Redis 之后才落账（§3.3）。同步 v1 承认有界回归（§2.1），不承诺零影响。

### 5.2 账本完成边界：emitter seal + recorder finalize

- **`run_completed` 不是最后一条事件**：现有顺序是 `run_completed` →（可能）`suggested_questions_pending`（`agent_loop_run_completion.py` 末端）。
- **Emitter 的 sequence 预留时机（终审定稿）**：当前实现在 `await writer` **返回后**才递增 `_sequence`，取消发生在 `await writer` 内时递增不执行，后续终态事件会**复用相同 sequence**。改为在**序列化与校验完成后、第一次可取消的 `await` 之前**预留并递增：

```python
async def _emit(self, event, ...):
    async with self._lock:
        sequence = self._sequence          # 预留（此后该序号永不复用）
        event.sequence = sequence
        event.ts = time.time()
        payload = event.model_dump(mode="json")
        validate_payload(payload)          # 校验必须在递增前完成
        self._sequence = sequence + 1      # 第一次可取消 await 之前递增
        await self._writer.append_chunk(...)   # 取消发生在此处 → 该序号已消耗，形成空洞而非复用
```

- Emitter 新增封口接口（同一把锁内）：

```text
async def seal_and_get_last_sequence() -> int
    # 在 self._lock 内：
    #   - 置 sealed=True，此后任何 emit 抛 RuntimeError（禁止后续事件）
    #   - 返回 self._sequence - 1
```

- 编排层流程：
  1. 所有同步辅助事件（含 `suggested_questions_pending`）发出完毕；
  2. `last_seq = await emitter.seal_and_get_last_sequence()`；
  3. `await trajectory_recorder.finalize(last_seq)`。
- **finalize 的数据库校验（latch 优先，再验三断言）**：

```text
if recorder.degraded_latch(run_id):              # ← 第一步：latch 优先，终审定稿
    trajectory_status = 'degraded'
    degraded_reason = latch 原因（如 recorder_cancelled / recorder_timeout / admission_full / write_failed）
    return  # 禁止进入 complete，即使 COUNT/MIN/MAX 重新满足

SELECT COUNT(*), MIN(sequence), MAX(sequence) FROM agent_events WHERE run_id = :run_id
断言：COUNT(*) = expected_last_sequence + 1
      MIN(sequence) = 0
      MAX(sequence) = expected_last_sequence
通过 → trajectory_status = 'complete'，finalized_at = now
不通过 → trajectory_status = 'degraded'，degraded_reason = 'finalize_mismatch'
```

> **迟到事务竞态**：最后一条事件（含终态辅助事件）落账超时后，迟到事务可能随后成功提交，使三断言重新满足——**只要 latch 曾置位（含取消），本次 run 只能落为 `degraded`**。需增加「最后一条事件迟到提交」与「取消后 sequence 不复用」的竞态测试（§9.2）。

- **取消路径**：`run_failed / run_limit_reached / run_interrupted` 与用户取消、shutdown 全部在统一 `finally` 中执行 seal + finalize，并用**有界 `asyncio.shield`** 包裹，防止二次取消打断收口。
- 同步 v1 的 seal+finalize 称为 **commit/completeness barrier**；「flush barrier」一词保留给 §5.4 异步队列方案。

### 5.3 落账失败、degraded 与 legacy 的判定边界

事件 INSERT 因 Postgres 不可用失败时，随后更新 meta 也可能失败——「失败必然可见」不能依赖同一次 DB 写。因此：

1. **内存降级 latch**：Recorder 进程内维护 per-run degraded 标记；首次落账失败/超时/准入失败/取消置位（§5.1），后续事件跳过落账（不重复报错），并在 DB 恢复后重试写 meta（幂等）。**迟到事务不得反转 latch**（§5.1、§5.2）。
2. **stale-recording 协调任务**：周期性扫描 `recording` 状态、但对应 run 已不在运行/已终态的 meta；按 run 实际终态收敛——latch 未置位且 `expected_last_sequence` 已持久化且数据库校验通过（§5.2 三断言）→ `complete`；否则 → `degraded`（含进程崩溃丢失内存队列的场景）。
3. **legacy 判定唯一判据**：
   - `run.created_at < ledger_enabled_at`（迁移持久化的各环境启用时间，§4.4）且无 meta → `legacy/not_recorded`（账本上线前，正常）；
   - `run.created_at >= ledger_enabled_at` 且无 meta → `degraded/meta_missing`（数据库故障被显式暴露，**不得伪装成历史运行**）。
4. **可见性**：`degraded` 在 UI / API / 审计三处一致可见（§7、§8）。

### 5.4 异步化设计（前置定义，v1 不实现）

**切换条件**：P0 实测超过 §9.1 阈值即切换到本方案（同步 v1 只用于验证，不追求长期维持）；按以下边界升级：

- **有界队列**（per-run，如 1000 条）；队满策略：**降级**（标记 degraded 并丢弃新事件，绝不阻塞 emitter 锁）。
- **flush barrier**：与 v1 seal+finalize 同位置、同语义（所有终态路径经过 flush，成功才置 `complete`）。
- **崩溃恢复**：进程崩溃时内存队列丢失；启动时扫描 `recording` 且 run 已不在运行的 meta，按 §5.3 收敛为 `complete` 或 `degraded`。
- **有序性**：后台 writer 保证同一 run 严格按 sequence 顺序落库（单 run 单 writer 或按 run_id 分区）；跨 run 可并发。
- 后台任务使用自己的 Session，不传请求 Session。

## 6. 投影层：TrajectoryProjector（P1）

### 6.1 原则

- 查询时 O(n) 纯函数投影，不落冗余 span 数据（不建 span 表）。
- 支持两类事件形态：
  - **配对生命周期**：`started / completed|failed|cancelled` 按 id（`llm_round_id / retrieval_id / tool_attempt_id / run_id / step_id / tool_call_id`）配对，计算 duration、TTFT。
  - **自带摘要**：`tool_call_completed / step_completed` 已携带 `duration_ms / status / result_summary`，可直接建 span，配对只用于补参数与流式 delta。
- **Orphan 收口**：按 §3.3 规则推导关闭未配对 span；推导 span 必须携带 `terminal_source: recorded|inferred` 与 `inferred_reason`；**成功 run 中的孤儿标 `unknown/incomplete`，不得伪装成真实 cancelled**；TTFT 允许为 null（无任何 output delta 的轮次）。
- annotation 事件不建 span，按类型挂到对应 span 的附加区段（plan/evidence/context）。
- 结果模型：`TrajectorySnapshot { run, records[], spans[], completeness, message 归组, attempt 归组, truncated }`。

### 6.2 与现有持久化模型的关系

- `AgentSession / AgentStep / ToolCallLog` 是**权威摘要**（用户可见、审计用）；`agent_events` 是**脱敏轨迹账本**（调试用）。两者不互相覆盖：摘要表继续由现有写路径维护，账本只服务轨迹视图。
- 快照 API 合并两者：摘要表提供高置信度字段，账本提供完整时序与详情（按权限分级下发，§7.3）。

## 7. API 层（P1）

### 7.1 端点与截断保护（P1 = 仅历史快照，不声称实时）

```text
GET /api/conversations/{conversation_id}/runs
    → run attempt 列表，按 message 归组、按 attempt_index 排序
      [{run_id, message_id, turn_message_id, attempt_index, status, trajectory_status,
        total_steps, total_tool_calls, duration_ms, started_at, ended_at}]

GET /api/conversations/{conversation_id}/runs/{run_id}/trajectory
    → TrajectorySnapshot（事件 + 投影 + completeness + degraded_reason + truncated）
      —— 普通用户 DTO 与管理员诊断 DTO 分离（§7.3）
```

**截断保护**：

- 定义 `MAX_TRAJECTORY_EVENTS_PER_RUN`（默认 5000，配置常量）；
- 查询使用数据库 `LIMIT max+1` 探测超限；
- 超限时响应携带 `truncated=true`，投影只基于已加载前缀；**禁止对截断数据生成看似完整的 span**（未配对的 started 一律按 §3.3 标注 inferred/unknown，且整体标记 truncated）；
- 运行列表接口同样受 `MAX_TRAJECTORY_RUNS_PER_CONVERSATION` 保护（超限返回最近 N 条 + `truncated=true`）。

**实时性边界**：P1 只提供历史快照，**不声称实时**；「打开面板拉快照」之外，运行中实时 SSE 归并属于 **P3**（复用现有聊天 SSE + 受控事件回调通道）。P1 文档与接口注释不得出现实时语义。

### 7.2 鉴权与越权

- 普通端点必须按 `(conversation_id, current_user.id)` 查询，并校验 run 属于该会话；**越权统一返回 404**（不暴露资源存在性）。
- 管理员诊断固定使用 `/api/admin/audit/.../trajectory`，接入现有 `get_conversation_auditor` 与访问审计（`app/api/admin_audit.py` 体系）；**普通端点不得按角色切换返回内容**。
- 权限分级只发生在端点层，不在同一端点内分支。

### 7.3 安全分级（服务端契约，不是前端约定）

- **新生命周期事件必须天然是用户安全数据**：经聊天 SSE 发给普通用户的 `llm_round_* / retrieval_* / tool_attempt_*` 不得携带 prompt、完整输入输出、工具 schema 原文、检索查询全文。敏感字段只出现在管理员诊断 DTO。
- **落库 allowlist**：`TrajectoryRecorder` 按事件类型定义允许入库的字段白名单（现有 `sanitize_arguments` 是工具名定向脱敏，不是通用事件脱敏器，不能直接当账本脱敏用）。
- **管理员 DTO 的字段来源**：prompt、工具 schema、完整输入/输出**不在账本中**，v1 管理员 DTO 只能承诺从 `ToolCallLog / messages / 工具注册表` 可还原的字段；无法还原的字段**从 v1 承诺中删除**，不得声称账本可提供。
- **DTO 分离**：普通用户 `TrajectorySnapshot` 与管理员诊断 DTO 分离（不同端点，§7.2）；**禁止依赖前端隐藏字段实现权限**。
- 命名：账本称「**脱敏事件账本**」，文档与 UI 不得声称是完整原始事件。

## 8. 前端（P3）

- **独立调试侧栏/Tab**，不扩充 `AgentRunTimeline`（它是普通用户的低噪声执行状态展示，成功路径主动压缩信息，职责不同）。
- 组件拆分（移植自 MVP 原型，按 fusion-ui 的 Redux/设计规范重写状态接入）：
  - `TrajectoryLedger`：轮次账本（#N / 类型标记 / 单行摘要 / 状态 / 耗时）
  - `TrajectoryTimeline`：Overview 时间线（真实耗时比例、拖选聚焦、缩放、hover 计时；TTFT 分段）
  - `TrajectoryInspector`：检查器（用户级字段 / 管理员诊断字段分级渲染）
  - `TrajectorySidebar`：编排 + 聊天双向定位（hover 工具调用 ↔ 轨迹节点）
- **受控事件回调通道（P3 交付，随侧栏一起验收）**：现有 `dispatchAgentEvent` 是 switch 无 default，未知事件被直接丢弃（`fusion-ui/src/lib/api/chat.ts`）。P3 增加受控通用回调（如 `onAgentEvent(type, payload)`）供轨迹侧栏实时消费；**只转发经过公共事件 union、`schema_version` 与 allowlist 校验的 DTO，不允许任意未知事件直通**；不改变现有已知事件的既有处理路径。
  - P3 的实时归并语义：打开面板拉快照（P1 端点）+ 运行中消费现有聊天 SSE 增量归并；`truncated / degraded / legacy` 状态必须随视图明示。
- 聊天 ↔ 轨迹双向定位是 DSH 体验的关键，P3 必须实现（`tool_call_id / step_id / message_id / llm_round_id` 足够做链接键）。
- MVP 前端继续作为原型实验室：交互创新（Overview 拖选、搜索、折叠）先在 MVP 快速迭代，稳定后移植。

## 9. P0 验收标准

P0 验收不依赖 P2（MVP Adapter 联调属 P2 验收）。

### 9.1 延迟与首屏基准（先测基线，再定阈值；有界回归，不承诺零影响）

1. 改造前先记录基线：agent_event 写入延迟 p50/p95/p99、模型 TTFT、整轮耗时、DB 提交耗时/连接数/锁等待。
2. 改造后同场景复测，判定阈值：
   - 相对回归：p95/p99 不劣于基线的 **1.2 倍**；
   - 绝对门限（基线建立后确认）：emitter 锁内 agent_event 写入 p95 ≤ 5ms、p99 ≤ 15ms（若基线本身更高，以相对回归为准并记录原因）。
3. 独立 Session 无争用：连接数、锁等待不随事件量线性增长。
4. **端到端 TTFT（有界回归）**：记录客户端实际收到首个 reasoning/content/tool SSE 的时间，与模型内部 `first_output_delta.ttft_ms` 对比；**不得只测模型内部时间**。改造前后的端到端 TTFT 差值必须落在 §9.1 阈值内；Recorder 失败（fail-open）不影响首个 SSE chunk 到达（有断言/测试）。**超过阈值 → P0 切换 §5.4**。

### 9.2 账本正确性

- **sequence 语义（统一验收口径，终审定稿）**：
  - `complete` 的 run：严格 **`0..N-1` 连续**、无空洞、无重复（emitter 从 0 开始）；
  - `degraded` 的 run：只保证**单调、唯一、不复用**（取消/熔断/丢事件可产生空洞——空洞正是 degraded 的成因之一）；
  - 取消后已分配 sequence 不复用（后续事件序号严格更大）——由「await 前预留」（§5.2）保证。
- **seal + finalize 语义**：`seal_and_get_last_sequence()` 在同一把锁内封口并返回 `_sequence - 1`；finalize 在**所有同步辅助事件（含 `suggested_questions_pending`）之后**执行；`run_completed` 不触发 finalize；seal 后任何 emit 必须抛错。
- **finalize 三断言 + latch 优先**：finalize 先查 degraded latch，置位（含取消 `recorder_cancelled`）则一律落 `degraded`（即使三断言满足）；未置位才验 `COUNT = expected+1 ∧ MIN(sequence)=0 ∧ MAX(sequence)=expected`（§5.2）——两条路径都有单测。
- **迟到事务竞态**：「最后一条事件落账超时 → 迟到提交成功 → 三断言满足」仍必须 `degraded`（latch 权威性）有专门竞态测试。
- **取消路径竞态**：worker 运行中 / 尚未启动时取消——permit 最终守恒（无泄漏）、迟到异常被消费、latch 保持 degraded、finalize 不得 complete、**sequence 不重复**（有专门竞态测试）。
- **原子事务**：事件 INSERT 与 meta 计数同事务（§5.1）有故障注入测试——INSERT 成功后 commit 前失败，必须整体回滚，不得出现「有事件无计数」。
- **超时熔断 + permit 生命周期**：`BoundedSemaphore(4)` 非阻塞准入——满载直接 fail-open（有测试：sem 满时新事件不排队、置 degraded）；`wait_for(250ms)` 在 DB 挂起时真正超时（非假超时）；**「任务尚未开始执行便超时」不得泄漏 permit**（有测试：连续 N 次超时后 semaphore 仍可继续 acquire，permit 数守恒）；`shield` 保证超时不取消底层任务；迟到任务异常被消费（无 never-retrieved 告警）；psycopg2 `connect_timeout` 整数秒限制已说明（500ms 不可配，主隔离靠 wait_for+闸门）。
- **取消路径**：`run_failed / run_limit_reached / run_interrupted / 用户取消 / shutdown` 都在统一 `finally` + 有界 `asyncio.shield` 中完成 seal+finalize；二次取消不得打断收口；**`CancelledError` 分支置 degraded 并 re-raise**（§5.1）有单测。
- **完整性矩阵**：成功 / 失败 / 中断 / 继续执行（retry、regenerate）后的轨迹完整性——**seal/finalize 后无新事件；run 终态事件后允许同步辅助事件**。
- **attempt_index 并发**：同 `turn_message_id` 并发 retry/continue 不产生相同 `attempt_index`（部分唯一约束 + conversation `SELECT FOR UPDATE` 原子分配）；迁移回填后历史行 `turn_message_id/attempt_index` 正确、`message_id=NULL` 行不受影响。
- **legacy 判定**：`run.created_at < ledger_enabled_at`（迁移持久化值）→ `legacy`；`>= ledger_enabled_at` 且 meta 缺失 → `degraded/meta_missing`；两类有测试覆盖。

### 9.3 协议与回归

- 所有事件携带 `schema_version`；缺失版本事件被前端/投影器兼容解释。
- `llm_round_first_output_delta`：真实 DeepSeek 流式路径上至少观察到一次；`delta_kind` 覆盖 reasoning/content/tool_call 至少两种（可多轮构造）；**空流不发送该事件、`llm_round_completed.ttft_ms=null`** 有单测。
- **发送顺序断言**：reasoning/content 分支——Redis Stream 中可见 chunk 条目先于 `first_output_delta` 条目（有测试）；tool_call 分支——不要求可见 chunk 前置（有测试覆盖该分支不违反约束）。
- `retrieval_cancelled` 有真实/模拟取消路径覆盖。
- cache token：`cache_read_tokens / cache_write_tokens` 提取逻辑有单测（供应商不返回时为 null，不报错）。
- **Orphan 收口**：推导 span 带 `terminal_source/ inferred_reason`；成功 run 孤儿标 `unknown/incomplete`（不伪装 cancelled）有单测。
- 回归：现有聊天流、进度快照、SSE、管理审计全量测试通过；`app/ai` 无新增对 `app/services` 的依赖（架构规则）。

## 10. 分阶段计划

| 阶段 | 交付物 | 退出标准 |
|---|---|---|
| P0 | 协议升级（schema_version + 新增生命周期事件含 `first_output_delta` 双分支语义、`retrieval_cancelled`、cache token 提取）；`agent_events` + `run_trajectory_meta`（FK、expected/finalized、latch 优先）；**AgentSession 迁移与回填（§4.4）**；`TrajectoryRecorder`（独立 Session + 原子事务 + **BoundedSemaphore 准入 + wait_for/shield/CancelledError/DB 超时熔断 + permit 生命周期四规则** + seal/finalize 三断言 + latch 优先）；emitter `seal_and_get_last_sequence`；`turn_message_id/previous_run_id/attempt_index`（稳定 turn 分组 + 两阶段兼容 + 并发分配）；落库 allowlist 脱敏；保留策略 = 级联删除（无独立 TTL） | §9 验收全绿（同步 v1 有界回归在阈值内；超阈值按 §5.4 切换）；不含 P2 依赖 |
| P1 | `TrajectoryProjector`（orphan 收口 + `terminal_source` + 截断保护）；快照 API（用户/管理员端点分离、越权 404、`truncated` 语义）；**仅历史快照，不声称实时** | 真实数据可回放、可钻取、degraded/legacy/truncated 明示 |
| P2 | MVP 侧 `FusionTrajectoryAdapter`（不污染 fusion 正式 API）；真实数据联调 | 协议与交互模型在 MVP 上钉死 |
| P3 | fusion-ui 轨迹侧栏（账本 + 时间线 + 检查器 + 双向定位）+ 受控事件回调通道 + 运行中实时归并 | 与 DSH 轨迹对标的功能验收 |

## 11. 对抗式审查清单（终审确认用）

- [ ] 独立 Session 未被共享对象污染（全局 session、请求依赖注入）。
- [ ] emitter 锁内新增 sink 后最坏路径延迟达标（§9.1 **有界回归**，不承诺零首屏影响）；Redis 权威事件异常未被 CompositeWriter 吞掉。
- [ ] **超时熔断可执行 + permit 无泄漏**：`BoundedSemaphore(4)` 非阻塞准入（满载 fail-open、不排队）；`wait_for(asyncio.shield(future))`——超时不取消底层任务；submit 失败立即释放；worker `finally` 释放；迟到异常被消费；「任务未开始便超时」不泄漏 permit（permit 守恒测试）。
- [ ] **`CancelledError` 分支**：捕获置 degraded（`recorder_cancelled`）→ 消费迟到 future → **re-raise**（不吞取消）；取消后已分配 sequence 不复用；「worker 运行中/未启动时取消」竞态测试覆盖 permit 守恒、latch 保持 degraded、finalize 不得 complete、sequence 不重复。
- [ ] **finalize latch 优先**：latch 置位（含取消）→ 一律 degraded（即使三断言满足）；「最后一条事件迟到提交」竞态测试存在。
- [ ] **sequence 在第一次可取消 await 前预留**：校验完成后、await writer 前递增；取消的 emit 消耗序号形成空洞（degraded）而非复用；complete 严格 `0..N-1` 连续、degraded 单调唯一允许空洞——两侧语义有测试。
- [ ] `seal_and_get_last_sequence` 与 emit 互斥（同一把锁）；seal 后 emit 抛错有测试；取消路径 shield 有测试。
- [ ] finalize 调用点在**所有同步辅助事件之后**；`run_completed` 不触发 finalize；finalize 三断言有单测。
- [ ] 事件 INSERT 与 meta 计数同事务；故障注入整体回滚有测试。
- [ ] degraded 在 DB 不可用时仍可达（内存 latch + 协调任务）；legacy 唯一判据是迁移持久化的 `ledger_enabled_at`；新 run 缺 meta 必须 `degraded/meta_missing`。
- [ ] `first_output_delta` 只在真实 delta 时发送且 `ttft_ms` 非空；空流由 `completed.ttft_ms=null` 表达；**reasoning/content 分支可见 chunk 先于事件（顺序断言）**；**tool_call 分支不受该约束**。
- [ ] `retrieval_cancelled` 覆盖；orphan 推导带 `terminal_source/inferred_reason`，成功 run 孤儿标 `unknown/incomplete`。
- [ ] `app/ai` 无反向依赖 `app/services`；LLM round 事件由编排层发出。
- [ ] 落库 allowlist 是单点强制；新事件天然用户安全；管理员端点固定 `/api/admin/audit/.../trajectory` 接入访问审计；普通端点越权 404、不按角色切换内容。
- [ ] `turn_message_id/previous_run_id/attempt_index` 四场景全覆盖 + 两阶段兼容（阶段 A 缺省时 FOR UPDATE 内解析 + 兼容指标；阶段 B 收紧）；部分唯一约束 + 并发分配有测试。
- [ ] 迁移回填：历史 `turn_message_id = message_id`，row_number 按 `(turn_message_id, created_at, id)`、`message_id=NULL` 行处理、历史跨 assistant lineage 不伪造、`ledger_enabled_at` 由迁移持久化。
- [ ] P0 无独立 TTL；级联删除无孤儿；`expected_last_sequence / finalized_at` 持久化完整。
- [ ] `MAX_TRAJECTORY_EVENTS_PER_RUN` 截断保护：`truncated=true`，截断数据不生成看似完整的 span。
- [ ] `schema_version` 缺失兼容；部署窗口无破坏。
- [ ] 快照 API 的 O(n) 投影在长会话下可接受；有上限保护。

## 12. 风险与开放问题

- **事件体积**：content blocks / 证据全文不入账本，摘要 + 引用；必要时按类型设 cap。
- **协议破坏性变更**：`schema_version` 加入 AgentEventBase 是破坏性的；部署窗口前端需兼容缺失版本（§3.1）。
- **cache token 提取**：供应商 usage 扩展字段格式未知，P0 单独列为新能力；提取失败降级为 null 不报错。
- **两阶段兼容**：阶段 A 的缺失率需持续监控；阶段 B 收紧前必须覆盖稳定，否则旧客户端断约。
- **同步 DB 线程池**：fusion 为同步 SQLAlchemy + psycopg2（`connect_timeout` 限整数秒），超时隔离靠 `BoundedSemaphore` 准入 + `wait_for(asyncio.shield(...))` + 显式 `CancelledError` 分支；permit 泄漏由四规则 + 守恒测试兜底；若线程池方案实测不达 §9.1 阈值，记录决策切换 AsyncSession（§5.1 备选）。
- **sequence 预留时机**：emitter 在第一次可取消 await 前预留并递增（§5.2）；取消/熔断导致的空洞只允许出现在 degraded run；complete 严格要求连续（§9.2）。
- **同步 v1 延迟**：承认有界回归（不承诺零首屏影响）；若超 §9.1 阈值，P0 切换 §5.4 异步队列（切换条件已定义，实施成本可控）。
- **TTL 推迟**：P0 只做级联删除；独立 TTL（含 `expired` 状态）留待后续，避免「complete 但事件为空」的假象。
- **MVP 协议差异**（命名、sequence 0-based、span 含义）：由 P2 的 `FusionTrajectoryAdapter` 吸收，不反向污染 fusion API。
- **仓库状态**：`.gitignore` 例外与文档跟踪按文首决定，在正式实施分支首个提交中一并处理；不在 detached HEAD 操作。
