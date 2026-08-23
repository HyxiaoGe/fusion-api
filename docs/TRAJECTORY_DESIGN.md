# Fusion Trajectory 集成设计（DSH 风格 Agent 执行轨迹）

> 状态：设计稿 v0.21（P0 已 dev 验收并采用 `QueuedTrajectoryRecorder`；P1 已由 PR #54 合并并部署验证；P2 独立联调夹具已完成既有真实 API 验证；P3 实施中；v0.21 P3 决策已迁入本正式 `feat/trajectory-p3-contract-guard` 分支）
> 范围：fusion-api + fusion-ui 集成 Agent 执行轨迹展示；`langchain-trajectory-mvp` 仅保留为历史联调夹具，不进入 Fusion P3 运行时或交付链路。
> 本文档记录 P0–P3 的统一架构；P3 直接消费已发布 P1，不重新实施后端投影器或快照 API。
>
> **仓库处理决定**：正式实施分支通过 `.gitignore` 的 `!docs/TRAJECTORY_DESIGN.md` 例外跟踪本文档。
> **P3 仓库落位**：本文档以 `fusion-api/origin/master` 的 v0.15 为底，仅合入 P3 决策，保留已经发布的 P0/P1 事实；前端在 `fusion-ui/origin/master` 的独立 P3 worktree 实施。

## 1. 背景与目标

### 1.1 背景

- fusion 已有自实现 Agent 运行时（零 LangChain）：多轮对话、工具调用、计划协调、推理流、上下文管理，并已发出 21 种 `agent_event`（`run/step/tool` 三级层级、`sequence`、`trace_id`）。
- P0 之前这些事件只进入带 TTL 的 Redis Stream；P0 已将脱敏事件按 sequence 追加到 PostgreSQL，Redis/SSE 继续负责实时投递，P1 已提供有界历史查询和 span 投影。
- 目标对标 DeepSeek Harness 的 Trajectory：在 fusion-ui 中提供独立的「Agent 执行轨迹」视图——按轮次组织的账本、Overview 时间线、逐条检查器，与聊天双向定位。
- `langchain-trajectory-mvp` 已验证「追加账本 + SSE 补发 + 投影纪律 + 瀑布图 UI」三层能力；其数据源是演示链，不适合作为 fusion 的数据源。**MVP 转型为：① UI 原型实验室（快速试交互）；② P2 阶段的联调夹具。**

### 1.2 目标形态

```
ConversationViewShell（同一份 messages + 脱敏事件账本）
  ├─ Chat Tab：用户提问 / 助手回答 / 轻量运行状态 / 「查看轨迹」
  ├─ Trajectory Tab：全高、按 turn 组织的会话级有界轨迹视图
  │    ├─ 最近 N 个 Run 的骨架与摘要带
  │    └─ 当前选中 Run 的 records / spans 水合、虚拟账本、瀑布图与检查器
  └─ ChatInput：会话壳唯一实例，跨 Tab 始终挂载
```

### 1.3 设计硬约束（评审结论，不可放宽）

1. **TrajectoryRecorder 必须使用独立数据库 Session**，不复用 Agent Loop 的业务 Session；不得对请求 Session 执行 commit/rollback。
2. **账本与投影分离**：`TrajectoryRecorder` 只做不可变追加，投影由查询侧 `TrajectoryProjector` 完成；`AgentProgressRecorder` 是写时折叠快照，三者是同一个事件协议的不同消费者。
3. **`trajectory_status` 独立于 `AgentSession.status`**：轨迹观测缺失（`recording/complete/degraded/legacy`）与模型回答不完整（`incomplete`）语义必须分离。
4. **账本完成边界 = emitter seal + recorder finalize**：`run_completed` 不是最后一条事件；由 Emitter 在同一把锁内 `seal_and_get_last_sequence()` 封口并返回 `_sequence - 1`，再由编排层调用 `finalize(expected_last_sequence)`。P0 当前由 `QueuedTrajectoryRecorder` 先执行 **flush barrier**，再调用同步核心的 commit/completeness barrier。
5. **`schema_version` 提升到公共 Event Envelope**，所有 `agent_event` 统一携带；部署窗口内兼容缺失版本。
6. **安全分级是服务端契约**：新增生命周期事件必须天然是用户安全数据；落库为**脱敏事件账本**，禁止依赖前端隐藏字段实现权限。
7. **首屏延迟有界**：dev 同步基线已超过 §9.1 绝对门限，P0 已启用 §5.4 有界异步接纳；同步 `TrajectoryRecorder` 保留独立 Session、超时/熔断与取消语义，由每个 run 的单消费者顺序调用。`first_output_delta` 仍按 delta 类型分别定义发送顺序（§3.3）。

## 2. 总架构与数据流

### 2.1 事件流（写侧，改造后）

```
AgentEventEmitter（锁内分配 sequence + ts，唯一发送方；seal 封口）
        ↓ await
AgentEventCompositeWriter
  ├─ Redis Stream（AgentEventRedisWriter）→ 实时 SSE（现有）
  ├─ AgentProgressRecorder（现有）→ agent_progress_snapshots（写时折叠）
  └─ QueuedTrajectoryRecorder（P0 当前生产接纳边界）
       └─ 单 run 单消费者 → TrajectoryRecorder 同步核心 → agent_events（只追加，独立 Session）
```

**sink 失败语义分类**（必须显式区分，不得统一吞异常）：

| sink | 类别 | 失败语义 |
|---|---|---|
| Redis：权威 `plan_snapshot` | required | **fail-closed**：现有语义保持——重试后仍失败则抛 `StreamWriteUnavailableError` 并终止生成（`tool_executor.py`）。CompositeWriter 不得吞掉该异常 |
| Redis：非权威事件（其余 20 种） | auxiliary | fail-open：记日志，不阻塞 |
| AgentProgressRecorder | auxiliary | fail-open：记日志，不影响主链路（现有行为保持） |
| QueuedTrajectoryRecorder / TrajectoryRecorder | auxiliary | 对主链路 fail-open；但对**轨迹可信度** fail-closed：队满、flush 超时、取消或 worker 异常必须反映到 `trajectory_status = degraded`（见 §5.3、§5.4） |

**首屏延迟边界（硬约束 #7）**：`emitter._emit` 仍在锁内 await writer，CompositeWriter 先完成 Redis 与 progress，再调用 `QueuedTrajectoryRecorder.record_chunk()`；该调用只做有界队列的非阻塞接纳，不等待同步数据库写入。数据库提交由单 run 消费者在锁外顺序执行，因此不再把每次同步提交直接计入端到端 TTFT。本文档仍不承诺零延迟影响，而是：

- 生产接纳边界执行 §9.1 的 p95/p99 绝对门限；
- 队列包装器与同步核心分别执行 §5.4、§5.1 的有界、超时、取消和 degraded 语义；
- `first_output_delta` 的发送顺序按 delta 类型分别定义（§3.3）；
- dev 同步基线已经触发 §5.4，P0 当前生产路径已完成该切换。

顺序保证：Redis 与折叠快照先于轨迹队列接纳可见；轨迹账本由单 run 单消费者严格按接纳顺序落库。sequence 从 **0** 开始，范围 `0..N-1`，**单调递增、永不复用**。

### 2.2 查询流（读侧，P1）

```
Trajectory API（GET /api/conversations/{id}/runs/{run_id}/trajectory）—— 仅历史快照，不声称实时
        ↓ 读取 agent_events（有序，LIMIT+1 截断保护）
TrajectoryProjector（查询时 O(n) 纯函数投影，不落冗余 span 数据）
        ↓
Trajectory 快照 JSON（事件 + span/record 投影 + 完整性状态 + truncated，按用户/诊断分级）
        ↓
fusion-ui Trajectory Tab（运行中实时归并 = P3，复用现有聊天 SSE）
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
    terminal_intent_id      TEXT,           -- 每次 finalize 唯一所有权 token
    terminal_intent_status  TEXT,           -- pending 目标：'complete' | 'degraded'
    terminal_intent_reason  TEXT,           -- pending degraded 的首次原因
    terminal_intent_version INTEGER,        -- 当前协议为 1；五列清空表示已 ack
    terminal_intent_pending_at TIMESTAMPTZ,  -- 非空表示终态仍待确认/协调
    updated_at              TIMESTAMPTZ NOT NULL
);
CREATE INDEX ix_run_trajectory_meta_terminal_intent_pending
    ON run_trajectory_meta (trajectory_status, terminal_intent_pending_at)
    WHERE terminal_intent_pending_at IS NOT NULL;

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
| `complete` | 账本完整 | **显式 seal + finalize 成功**：degraded latch 未置位 **且** 数据库校验 `COUNT = expected+1 ∧ MIN(sequence)=0 ∧ MAX(sequence)=expected`（§5.2），`finalized_at` 落库，且 `terminal_intent_pending_at IS NULL`；`complete + pending` 只是待协调的暂存终态，不得对外宣称完整 |
| `degraded` | 观测缺失 | **latch 置位（含取消/迟到事务场景）** / 任意落账失败 / 超时熔断 / 准入满载 / stale 协调发现缺尾部事件 / 新 run 缺 meta（`meta_missing`） |
| `legacy` | 账本建立前的历史 run | `run.created_at < ledger_enabled_at` 且无 meta（**仅此一种判据**，见 §5.3） |

**只有显式 seal/finalize 成功且 terminal intent 已独立 ack 才能确认 complete，且 latch 置位时 finalize 一律落 degraded**（§5.2）；stale recording 无持久化 `expected_last_sequence` 时只能保守标 `degraded`，不得猜测 complete。`recording|complete|degraded + terminal_intent_pending_at` 都是 Task 6 必须扫描的未确认状态；任何新 Recorder 都不得冒领、覆盖或清除已有 pending。

### 4.3 保留策略（P0 范围决策）

- **P0 只做随会话 CASCADE，不做独立 TTL**。理由：fusion 当前没有消息级 TTL；只删 events 保留 meta 会出现「状态 complete 但事件为空」的假象，删 meta 又会被误判为 degraded/legacy——独立 TTL 的正确语义（原子删除 events+meta 并引入 `expired` 状态）是后续工作，不塞进 P0。
- 后续需要 TTL 时：基于 `recorded_at` 的清理 Worker（挂 `scheduler_service` 体系）原子删除 events 与对应 meta，并增加 `expired` 状态与清理索引；清理失败按审计留存语义处理，不静默。
- 脱敏原则：prompt、完整输入输出、工具 schema **不入账本**；content blocks / 证据全文只存摘要 + 引用，详情按需从消息/证据库读取。

### 4.4 AgentSession 迁移与回填（P0 必做）

现有 `agent_sessions` 已有历史数据且 `message_id` 可空，按 Alembic expand/backfill/contract 顺序执行：

1. **expand**：`ALTER TABLE agent_sessions ADD COLUMN turn_message_id TEXT NULL`、`ADD COLUMN previous_run_id TEXT NULL`、`ADD COLUMN attempt_index INTEGER NULL`、`ADD COLUMN terminal_at TIMESTAMPTZ NULL`（均可空，不动存量行）。
2. **backfill**：
   - `turn_message_id = message_id`：历史行仅保留可证实的 assistant 锚点，不跨 assistant id 推断 user turn；
   - `attempt_index`：按 `(turn_message_id, created_at, id)` 用窗口函数 `row_number()` 回填，同一历史 assistant 锚点内的 run 得到稳定序号；
   - `message_id IS NULL` 的历史行：`turn_message_id / attempt_index` 保持 NULL（不参与 turn 归组，部分唯一索引不受影响）；
   - **历史 `previous_run_id` 保持 NULL，不伪造 lineage**（无法可靠推断，宁可缺失）。
   - 历史非 `running` 行以 `created_at` 保守回填 `terminal_at`；历史 `running` 行保持 NULL，避免把仍在执行业务链路的 run 误当作可协调终态。
3. **contract**：
   - 建部分唯一索引 `UNIQUE (turn_message_id, attempt_index) WHERE turn_message_id IS NOT NULL AND attempt_index IS NOT NULL`；
   - 建普通索引 `ix_agent_sessions_terminal_at (terminal_at)`，供 stale/pending/meta-missing 协调任务按终态年龄筛选；
   - 新数据约束由应用层强制（新建 run 必须写 `attempt_index`；`previous_run_id` 由两阶段兼容控制），数据库层不强制 NOT NULL（避免迁移期间断约）。
4. **`ledger_enabled_at` 不得是源码配置常量**：P0 新建单例表 `trajectory_ledger_settings` 持久化各环境的实际启用时间；legacy 判定只读取该不可变水位。不同环境（dev/prod）各自由迁移记录，运行时配置管理不得覆盖。
5. **运行时 `terminal_at` 不变量**：新建 run 或重新进入 `running` 时必须清为 NULL；所有业务终态写入（completed / limit / error / interrupted 及兼容写入旁路）必须在同一事务写入 aware UTC。协调器不得用 `updated_at` 或 `created_at` 替代新 run 的业务终态时刻。

## 5. 采集层：TrajectoryRecorder（P0）

### 5.1 P0 当前实现：异步接纳包装同步核心

- `app/services/agent/queued_trajectory_recorder.py` 的 `QueuedTrajectoryRecorder` 是 CompositeWriter 当前使用的生产接纳边界；它包装 `app/services/agent/trajectory_recorder.py` 的同步核心，不改动后者的事务与完整性职责。
- **独立短生命周期 Session**：同步核心每次写入自己开 session、自己 commit、自己 rollback；绝不触碰请求 Session，队列消费者也不接收请求 Session。
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
- 接入点：`AgentEventCompositeWriter.trajectory_recorder` 注入 `QueuedTrajectoryRecorder`（与现有 progress `recorder` 并列）；CompositeWriter 不吞 Redis 权威事件异常（§2.1）。

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
- **首屏边界**：sink 顺序保持 Redis、progress 先于队列接纳（§2.1 硬约束 #7）；`first_output_delta` 的 reasoning/content 分支在可见 chunk 写 Redis 之后才接纳落账（§3.3）。同步核心的数据库等待只发生在单 run 消费者中。

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
- **finalize 的数据库校验与 durable intent（latch 优先，再验三断言）**：

```text
独立事务 A（assessment 前必须明确 commit）：
  UPSERT meta(recording)
  为本次 finalize 生成唯一 :terminal_intent_id
  仅 UPDATE trajectory_status = recording 且五个 intent 字段均为 NULL 的行
  SET expected_last_sequence = :expected,
      terminal_intent_id = :terminal_intent_id,
      terminal_intent_status = degraded（已有 latch）否则 complete,
      terminal_intent_reason = 首次 latch 原因或 NULL,
      terminal_intent_version = 1,
      terminal_intent_pending_at = now

SELECT COUNT(*), MIN(sequence), MAX(sequence) FROM agent_events WHERE run_id = :run_id
断言：COUNT(*) = expected_last_sequence + 1
      MIN(sequence) = 0
      MAX(sequence) = expected_last_sequence
assessment/事件 latch、timeout、cancel、unknown：
  用新的独立事务按同一 terminal_intent_id CAS，把 pending target 单调升级为 degraded（首次原因不可覆盖）

风险终态事务 B（禁止清 intent）：
  通过且无 latch → 按同一 terminal_intent_id CAS，从 recording 更新为 complete，finalized_at = now
  否则 → 按同一 terminal_intent_id CAS，将 recording（或同 expected 的待纠正 complete）更新为 degraded

只有 worker 明确确认事务 B 的真实结果并在线程锁内封住后到 latch 后：
  独立幂等事务 C 按同一 terminal_intent_id CAS，清除五个 terminal_intent_* 字段（terminal ack）
```

事务 A、B、C 必须分别使用独立短 Session。A 的 `rowcount=0` 是明确的所有权前置失败，**不得**走 response-loss readback；只有 commit/连接等导致结果不确定时，才允许用新 Session 按本次 `terminal_intent_id` 对账。B、C、degraded upgrade、pending DTO 与所有 readback 必须绑定同一 `terminal_intent_id`；错误 token 的写入必须零影响。B 的 commit 响应丢失时只允许用同 token 的新 Session 对账或有限幂等重试；**B 绝不得顺带清 intent**。C 失败、响应丢失或对账失败时保留 pending；宁可由 Task 6 保守降级，也不得把未确认的 `complete + pending` 当作权威 complete。终态结果仍 unknown 时不得封住 terminal timeout/cancel：后到原因可继续把本次 pending 单调升级为 degraded。历史 `recording|complete|degraded + pending` 只归 Task 6 协调，新 Recorder 不得接管。

> **迟到事务竞态**：最后一条事件（含终态辅助事件）落账超时后，迟到事务可能随后成功提交，使三断言重新满足——**只要 latch 曾置位（含取消），本次 run 只能落为 `degraded`**。需增加「最后一条事件迟到提交」与「取消后 sequence 不复用」的竞态测试（§9.2）。

- **取消路径**：`run_failed / run_limit_reached / run_interrupted` 与用户取消、shutdown 全部在统一 `finally` 中执行 seal + finalize，并用**有界 `asyncio.shield`** 包裹，防止二次取消打断收口。
- `QueuedTrajectoryRecorder.finalize()` 先执行 §5.4 **flush barrier**，再调用同步核心的 commit/completeness barrier。

### 5.3 落账失败、degraded 与 legacy 的判定边界

事件 INSERT 因 Postgres 不可用失败时，随后更新 meta 也可能失败——「失败必然可见」不能依赖同一次 DB 写。因此：

1. **内存降级 latch**：Recorder 进程内维护 per-run degraded 标记；首次落账失败/超时/准入失败/取消置位（§5.1），后续事件跳过落账（不重复报错），并在 DB 恢复后重试写 meta（幂等）。**迟到事务不得反转 latch**（§5.1、§5.2）。
2. **stale/pending 协调任务**：以持久化的 `AgentSession.terminal_at` 作为业务终态年龄证据，并使用 60 秒 grace（`stale_before = now - 60s`）。候选 run 必须 `status != running AND terminal_at IS NOT NULL AND terminal_at <= stale_before`；此外：
   - 无 pending 的普通 stale `recording` 必须同时 `meta.updated_at <= stale_before`，再按 `expected_last_sequence +` §5.2 三断言判定 complete/degraded；
   - 任意 `trajectory_status` 只要 `terminal_intent_pending_at IS NOT NULL`，都属于 pending 候选，并必须同时满足 `terminal_intent_pending_at <= stale_before AND meta.updated_at <= stale_before`。**任何到期遗留 pending 都表示 Recorder 未明确 ack，Task 6 必须保守收敛为 degraded**（已有 degraded 保持 degraded；未知 status/reason/version 也安全降级），并在同一原子更新中清除五个 intent 字段；
   - `terminal_intent_pending_at` 虽已过期、但 `updated_at` 仍新鲜时不得协调，给正常 finalize 的 B→C、C ack 或迟到纠偏保留完整 grace；若 C 未发生，则从最后一次持久写入再经过完整 grace 后才保守降级；
   - 新 run 缺 meta 也必须满足相同 `terminal_at <= stale_before` 才写 `degraded/meta_missing`。插入使用 PostgreSQL `ON CONFLICT (run_id) DO NOTHING`，仅 `rowcount=1` 计入处理数；若并发 Recorder 先创建 meta，协调器不得覆盖，且同批其他候选仍正常提交。
   扫描在 PostgreSQL 使用 `FOR UPDATE SKIP LOCKED` 分批执行；业务 `running` run 绝不处理。这样既保护正常 finalize/late first-write，也使进程崩溃遗留状态在 grace 后幂等收敛。
3. **legacy 判定唯一判据**：
   - `run.created_at < ledger_enabled_at`（迁移持久化的各环境启用时间，§4.4）且无 meta → `legacy/not_recorded`（账本上线前，正常）；
   - `run.created_at >= ledger_enabled_at` 且无 meta → `degraded/meta_missing`（数据库故障被显式暴露，**不得伪装成历史运行**）。
4. **可见性**：`degraded` 在 UI / API / 审计三处一致可见（§7、§8）。

### 5.4 P0 有界异步接纳（当前实现）

dev 同步基线已经触发切换；`QueuedTrajectoryRecorder` 固定以下生产语义：

- **容量与消费者**：每个 run 一个容量 **1000** 的 `asyncio.Queue`，按需启动且只启动一个消费者；同一 run 严格按接纳顺序调用同步核心，跨 run 可并发。
- **队满**：`put_nowait` 失败立即置 `admission_full` degraded latch 并丢弃新事件，绝不阻塞 emitter 锁；latch 置位后不再接纳该 run 的后续事件。
- **10 秒 flush barrier**：seal 后编排层调用 `finalize(expected_last_sequence)`；包装器关闭接纳、向单消费者发送终止哨兵，并最多等待 **10 秒**清空已接纳事件。flush 成功后才调用同步核心 finalize 判定 `complete`。
- **超时**：flush 超过 10 秒置 `recorder_timeout`，取消 flush/消费者后台 task，随后仍调用同步核心 finalize，使 latch 优先收敛为 `degraded`；不得遗留后台 task。
- **取消**：flush 被取消时置 `recorder_cancelled`，取消并等待 flush/消费者 task，调用同步核心 finalize 后重新抛出 `CancelledError`；不得吞取消或泄漏 task。
- **worker 异常**：消费者捕获异常并置 `write_failed`，包装器保持 fail-open；finalize 仍关闭同步核心，轨迹不得标为 `complete`。
- **Session 与崩溃恢复**：后台消费者不持有请求 Session；同步核心每次操作继续使用独立短 Session。进程崩溃导致内存队列丢失时，由 §5.3 stale/pending 协调路径保守收敛。

## 6. 投影层：TrajectoryProjector（P1）

### 6.1 原则

- 查询时 O(n) 纯函数投影，不落冗余 span 数据（不建 span 表）。
- 支持两类事件形态：
  - **配对生命周期**：`started / completed|failed|cancelled` 按 id（`llm_round_id / retrieval_id / tool_attempt_id / run_id / step_id / tool_call_id`）配对，计算 duration、TTFT。
  - **自带摘要**：`tool_call_completed / step_completed` 已携带 `duration_ms / status / result_summary`，可直接建 span，配对只用于补参数与流式 delta。
- **Orphan 收口**：按 §3.3 规则推导关闭未配对 span；推导 span 必须携带 `terminal_source: recorded|inferred` 与 `inferred_reason`；**成功 run 中的孤儿标 `unknown/incomplete`，不得伪装成真实 cancelled**；TTFT 允许为 null（无任何 output delta 的轮次）。
- annotation 事件不建 span，按类型挂到对应 span 的附加区段（plan/evidence/context）。
- **P3 分层边界**：P1 已发布 `records + spans + completeness + truncated`，不新增第二套后端 cell 投影。P3 在浏览器端用 `TrajectoryCellProjection` 将会话 messages、P1 run/snapshot 与受控 SSE 增量投影为用户可读 cell；P1 spans 仍是历史瀑布图的权威投影。
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

GET /api/admin/audit/conversations/{conversation_id}/runs/{run_id}/trajectory
    → AdminTrajectorySnapshot({snapshot: TrajectorySnapshot, tool_calls: [...]})
      —— 仅审计员；成功审计写入后才返回诊断
```

普通 `TrajectorySnapshot` 固定包含 `run`、`records`、`spans`、`completeness` 与 `truncated`，不因角色增加诊断字段。管理员 `AdminTrajectorySnapshot` 以该普通快照作为嵌套 `snapshot`，额外给出独立的 `tool_calls` 与 `tool_calls_truncated`；每项固定为 `{association: "run", id, message_id, step_number, tool_name, status, duration_ms, model_id, provider, arguments, result_preview, error, redacted_fields, created_at}`。

两个读取上限默认值固定为：`MAX_TRAJECTORY_EVENTS_PER_RUN=5000`、`MAX_TRAJECTORY_RUNS_PER_CONVERSATION=500`。事件、run 列表和管理员 `ToolCallLog` 诊断均使用 `LIMIT max+1`；前两者设置 `truncated=true`，工具诊断设置 `tool_calls_truncated=true`。

**截断保护**：

- 定义 `MAX_TRAJECTORY_EVENTS_PER_RUN`（默认 5000，配置常量）；
- 查询使用数据库 `LIMIT max+1` 探测超限；
- 超限时响应携带 `truncated=true`，投影只基于已加载前缀；**禁止对截断数据生成看似完整的 span**（未配对的 started 一律标注 `terminal_source=inferred`、`status=unknown`、`inferred_reason=truncated_prefix`，且整体标记 truncated）；
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
- **run 级关联限制**：`ToolCallLog` 只按 `trace_id=run_id` 查询并按 `step_number / created_at / id` 排序。它没有 ledger `tool_call_id` 外键，管理员工具项必须固定 `association="run"`，不得伪造 span 或 `tool_call_id` 精确关联。
- **历史兼容**：管理员工具项的 `created_at` 可为 null；`ToolCallLog.created_at` 的历史 naive 值按既有**北京时间墙钟**业务语义赋 `Asia/Shanghai` 后转 UTC，原本 aware 的值按其 instant 转 UTC。该兼容只作用于 P1 读侧 DTO，未宣称已迁移或统一修复 PostgreSQL 历史 `TIMESTAMP WITHOUT TIME ZONE` 数据。`step_number` 与 `created_at` 的 NULL 排在非空值之后，再用 `id` 稳定兜底，保证 SQLite/PostgreSQL 一致。
- **审计闭环**：管理员轨迹读取必须复用 `get_conversation_auditor` 与 `X-Admin-Audit-Reason`；在返回 `tool_calls` 前写入 `admin.audit.trajectory.view`，审计失败返回既有 503，不得 fail-open。
- 命名：账本称「**脱敏事件账本**」，文档与 UI 不得声称是完整原始事件。

### 7.4 Trajectory Node Detail 契约（v0.21 定稿，P3 只承诺 Tool）

> 本节与 §8 的 Network 式三区交互共享同一 Detail 契约，分别验收。硬约束：**只支持精确关联，不支持历史猜测；旧数据允许缺失，新数据缺失必须显式暴露。**
> **P3 实现范围（v0.21，YAGNI）**：后端 `node-detail` 端点只承诺 `node_type=tool`；LLM/Message/Run/Step 的 Summary/Timing 由前端投影直接生成，不经 Node Detail API；统一 Envelope 为未来扩展保留其他 `node_type`。

```typescript
type TrajectoryNodeDetailResponse = {
  status: "available" | "pending" | "not_recorded" | "degraded";
  node_type: "tool" | "llm" | "message" | "run" | "step";
  available_sections: Array<"summary" | "payload" | "result" | "timing" | "schema">;
  detail: ToolNodeDetail | LlmNodeDetail | MessageNodeDetail | RunNodeDetail | StepNodeDetail | null;
  redacted_fields: string[];
  reason: string | null;
};
```

- `status` 表示关联数据是否可信，`available_sections` 表示可展示的详情页签，二者不得混用；P3 后端只实现 Tool 分支，非 Tool 节点不因没有 Tool Payload 而标记 degraded。
- 普通端点为 `GET /api/conversations/{conversation_id}/runs/{run_id}/node-detail/{node_type}/{node_id}`，Tool 的 `node_id` 为 `tool_call_id`；管理员端点位于 `/api/admin/audit/conversations/{conversation_id}/runs/{run_id}/node-detail/{node_type}/{node_id}`，要求访问原因和审计记录。
- `trajectory_detail_enabled_at` 是独立于 `ledger_enabled_at` 的持久化水位。先迁移列/索引、部署写入链路、确认实例开始写 `tool_call_id`，再显式激活该水位；激活前数据统一为 `not_recorded`，不得在迁移开始时提前生效。
- Tool 的服务端状态按以下确定性顺序判定：① 精确 `(run_id, tool_call_id)` 行存在为 `available`；② `run.created_at < trajectory_detail_enabled_at` 为 `not_recorded`；③ run 仍在运行或④ 已终态但仍在 `detail_settle_grace` 内为 `pending`；⑤ 其余终态且超宽限期仍无精确行才为 `degraded`。
- `ToolCallLog` 是 fire-and-forget 写入，Run finalize 不等待它；`detail_settle_grace` 为数秒级有限值（例如 5s），宽限期后缺失收敛为 `degraded`，不得无限刷新，也不得改变 P0 ledger finalize barrier。
- 新增 nullable `ToolCallLog.tool_call_id`，链路为 `tool_call["id"] → log_tool_execution → handler.log → log_tool_call → ToolCallLog.tool_call_id`；不得复用日志记录 ID。新写入必须同时具备 `trace_id` 与 `tool_call_id`，并建立部分唯一索引 `UNIQUE (trace_id, tool_call_id) WHERE trace_id IS NOT NULL AND tool_call_id IS NOT NULL`。
- 查询只允许精确匹配 `conversation_id + user_id + run_id/trace_id + tool_call_id`，禁止按 step/name/time 做历史启发式 fallback。普通用户脱敏由服务端执行，并在 `redacted_fields` 明示。
- `ToolCallLog` 每个逻辑 Tool Call 只保留最终一条；Attempt 只来自 `tool_attempt_started/completed`。Payload/Result 归属逻辑 Tool 节点，Attempt 仅展示状态、错误码和 Timing；单次成功 Attempt 默认折叠。
- 历史 run 缺精确详情时展示基础账本与 Timing，并以 `not_recorded` 说明 Payload/Result 不可用，不标 degraded、不猜测关联；新 run 缺失必须显式 `pending` 或 `degraded`。
- 本节不回填旧数据、不把 Payload/Result 复制进 `agent_events`、不实施 cursor 分页或 Schema UI。

## 8. 前端（P3）：Chat Tab + 会话级有界 Trajectory Tab

### 8.1 定位、范围与 DSH 对标

- Trajectory 是会话级全高 Tab，不是常驻窄侧栏；Chat 与 Trajectory 共享 conversation 数据和同一个 Composer。
- 首版是**会话级有界轨迹视图**：Run 列表只覆盖 P1 返回的最近 N 个 Run；单 Run 只覆盖 P1 返回的 sequence 前缀。任何 `truncated` 都必须由粘性横幅常驻明示，禁止“全部历史”“完整轨迹”等误导措辞。
- 按 `turn_message_id` 分组、按 `attempt_index` 排序；P1 普通 run DTO 不含 `previous_run_id`，P3 首版不绘制精确 lineage edge。cursor 与精确 lineage 属于 P1.1 后续增强。
- 对标 DSH 的部分：会话视图 Tab、turn 组织、用户可读账本、Chat → Trajectory one-shot inspect、长列表虚拟化。`plan/context` cell 与 Trajectory → Chat reveal 属于 Fusion 扩展。
- 普通会话 Tab 只消费普通用户 `TrajectorySnapshot`；管理员诊断继续走独立 admin-audit 页面/端点，不在普通组件中按角色切换 DTO，也不把 run 级管理员工具记录伪装成精确 tool/span 关联。
- v0.21 的视图布局为 Network 式三区：顶部 `TrajectoryOverview` 负责会话时间域与联动；中部 `TrajectoryTable` 是按 Turn 组织的虚拟化账本；右侧 `TrajectoryNodeDetailPanel` 展示选中节点。Overview、记录表与详情使用同一选中状态，避免再引入独立 Span 瀑布图加重复列表。

### 8.2 规范化事件、实时合并与终态对账

P1 record 与 SSE event 必须先经过独立 allowlist adapter，统一为：

```text
NormalizedTrajectoryEvent = {
  runId, sequence, eventType, schemaVersion, timestamp,
  stepId, toolCallId, parentStepId, traceId, payload
}
identity = (runId, sequence)
```

- `schema_version` 缺失按 legacy 版本兼容；未知 schema 或不在公共事件 union/字段 allowlist 中的对象不得进入 Redux。
- `trajectorySlice` 以 conversation 为根，在 Tab 未挂载时也持续接收受控事件；初次发送、continuation、页面 reconnect 三条 `createAgentStreamEventHandlers` 装配路径必须全部接入。
- merge 以 `(runId, sequence)` 幂等。snapshot 覆盖其 durable 前缀，保留 sequence 更大的 live tail；相同 identity 若出现 payload 冲突，以 durable snapshot 为准并显式标记 reconciliation conflict。
- parser 内 `lastSequenceByRun` 只负责单次 SSE 读取防重，不能替代 trajectory slice 的跨快照/重连去重。
- `run_completed / run_failed / run_interrupted / run_limit_reached` 到达后，将该 run 标为 `reconciling` 并重拉 P1 快照；只有 P1 `completeness/truncated` 能确认最终可信度。终态 refetch 前不得先宣称轨迹 complete。
- 验收必须覆盖：先订阅后取快照的交叠、Tab 晚打开、重连重放、snapshot 落后 live、终态 refetch 与同 key 冲突。

### 8.3 TrajectoryCellProjection 与 message join

`TrajectoryCellProjection` 是无 fetch、无 Redux、可单测的纯函数，输入为 messages、P1 run summaries、已加载 snapshots 与 live tail，输出：

```text
TrajectoryCell =
  UserCell | MessageCell | RunCell | PlanCell | ContextCell |
  ToolCell | SubtoolCell | CompactedCell
```

- 新数据确定性 join：`turn_message_id → user message`，`message_id → assistant message`；message 正文只引用 messages，不复制进账本。
- legacy 回填可能令 `turn_message_id` 等于 assistant id：先用 `message_id` 命中 assistant，再回看同一会话中相邻 user。仍无法归组的 run 进入“未关联运行”，不得猜测。
- Run 未加载 snapshot 时只生成 Turn/Run 骨架；选中后按需水合详细 cell。禁止一次请求或一次渲染拼接所有 Run 的全部事件。
- `legacy/degraded/truncated` 不参加完整字段 parity：可以用持久化 `message.agent_run` 的高置信度摘要生成带来源标记的 fallback，但不得伪造缺失事件、span 或消除完整性横幅。
- 主状态来自 `AgentSession.status`（运行中/完成/失败/中断/回答不完整）；轨迹 badge 独立来自 `trajectory_status/truncated`（记录中/完整/降级/历史未记录/已截断），两者不得互相覆盖。

### 8.4 会话壳与 Composer 单实例

```text
ConversationViewShell
  ├─ ViewTabs(Chat | Trajectory)
  ├─ ActiveViewBody
  └─ ChatInput（唯一实例，始终挂载）
```

- Tab 只切换 body；Composer 不 remount，草稿、附件、知识库选择、模型与焦点状态保持。
- 普通发送始终创建会话新 turn，与 `selectedRunId` 无关；从 Trajectory Tab 发送后保持当前 Tab，使用户可以直接观察新 run 实时进入账本。
- stop/steering 只作用于当前 active stream/run，不作用于所选历史 run；无 active stream 时不展示运行控制。
- 轨迹滚动区的底部边界由唯一 Composer 所在会话壳自然约束，不复制第二份 Composer 或固定双份 padding。

### 8.5 组件、懒加载与虚拟化

- `TrajectoryTabView`：全高编排 Network 式三区、Run 选择、完整性粘性横幅、终态动作与错误/空态。
- `TrajectoryOverview`：记录级时间总览（Input/Model/Tools 轨道、投影模式切换、区间选择、缩放/平移、hover 精确时刻）。P3 只实现 `sequence`（等宽、查看执行顺序）与 `actual`（真实 start/end、查看时间分布）两种模式；`duration/time` 后置，不纳入本期验收。
- Overview 的会话域由 Run summary 构成：当前聚焦 Run 展开 Input/Model/Tools 详细记录，未水合 Run 仅显示粗粒度占位区段；切换焦点时替换详细展开区段，不把 LRU 缓存详情永久拼入 Overview。跨未水合 Run 拖选时先聚焦并加载对应 Run，再展示详细匹配；不得假装已过滤全部记录，也不新增 Overview API。
- `TrajectoryTable`：按 Turn 分隔的高密度虚拟化记录表（user/message/tool 等 cell，#N、类型标记、单行摘要、状态、耗时）。记录行点击高亮 Overview；Overview 区段点击定位记录表并打开右侧详情。
- `TrajectoryNodeDetailPanel`：消费 §7.4 Tool Node Detail DTO，展示 Summary/Payload/Result/Timing；Payload/Result 按点击懒加载，诊断字段折叠展示。LLM/Message/Run/Step 的 Summary/Timing 由前端投影生成，不调用 Node Detail API。
- 区间选择清除必须同时提供可见「清除范围」按钮、Escape 和触屏入口；右键清除只能作为快捷方式，不能是唯一入口。
- 首载只拉 run list；只拉当前选中 Run snapshot；切换时按需加载并使用最多 8 个 Run 的 LRU 缓存。
- `TrajectoryTable` 使用固定行高窗口化，overscan 后 DOM 记录行始终 ≤200；支持 Home/End/方向键与 `aria-posinset/aria-setsize`。inspect 可先计算目标 index，再滚动到尚未挂载的虚拟行。
- 5000 事件固定 fixture 的 CI 门禁：纯投影批次 ≤750ms、1000 条受控增量合并批次 ≤500ms、DOM 记录行 ≤200；同时验证首/中/尾定位和连续 SSE 合并。

### 8.6 InspectRequest 与 revealInChat

- **Chat → Trajectory one-shot inspect**：

```text
InspectRequest = {
  nonce, conversationId, messageId, runId?,
  target?: { kind: message | tool, id }
}
```

- 消费顺序固定：切 Tab → 确定 Run → 必要时加载 snapshot → `scrollToIndex` → 高亮 → 清除 request。持久 `selectedMessageId/selectedRunId/selectedSpanId` 与一次性 request 分开。
- 聊天轻量状态行的 `messageId` 是稳定入口；只有仍存在稳定 `tool_call_id` 的用户内容卡片才允许发 tool target，不承诺聊天中存在 step/LLM 锚点。
- target 因 `truncated/legacy/缺失` 不可见时定位到 Run 头并提示“该节点不在当前有界快照中”，不得无限等待。
- **Trajectory → Chat** 使用独立 `revealInChat` Fusion 增强：保证切回 Chat 并定位 message；只有聊天中确有稳定 DOM target 时才做细定位，不称为“双向 one-shot”。
- Tab 往返保持 selection、ledger/timeline scroll 和 inspector 状态。

### 8.7 retry/continue 与聊天轻量状态行

- 消息级“重新发送/重新生成”保留在 Chat，继续覆盖非 Agent、发送失败与普通消息；“唯一入口”只约束 Agent run 级 retry/continue。
- Agent run 级 retry/continue 只出现在 Trajectory 终态横幅。按钮仅在所选 run 属于会话最后一轮、是该 assistant 的最新 attempt、当前无 active stream 且既有模型/知识库前置条件满足时可用；continue 额外要求 `status=limit_reached`。历史 attempt 只读。
- run retry 请求必须显式发送 `previous_run_id=selectedRunId`；continue 继续显式发送 selected run id。新 stream 的 `run_started.run_id` 到达后自动选中真实新 run，不用 `attempt_index + 1` 猜测。
- Chat 中 Agent 状态行固定为：状态点及名称、耗时、一个最高优先级异常、独立轨迹 badge、“查看轨迹”。禁止 plan、工具列表、Evidence、Step、Token、TTFT、run retry/continue 与可展开过程。

### 8.8 一致性闸门与旧“过程”迁移

一致性测试拆成四组，不能用一个“同源”断言混合不同语义：

1. `event projection parity`：同一已知事件集的 plan/tool/evidence/run 规范化字段；主 cohort 仅 `complete && !truncated && supported schema`。
2. `live ↔ durable reconciliation`：同一 complete、非 truncated run 在终态 refetch 后一致。
3. `message join invariants`：user/message cell 与 messages 一致，未关联 run 不猜测。
4. `action policy`：消息级 retry 与 Agent run retry/continue 的 eligibility、目标 message/run id 和 `previous_run_id` 分别测试。

`legacy/degraded/truncated` 单独验收“状态不静默、已知摘要仍可见、未知不伪造”，不要求与完整 progress 摘要逐字段相等。

迁移分两次交付：

1. P3 产品 PR 移除 `AssistantResponseStack` 对 `AgentRunTimeline` 的内联挂载，保留 `AnswerEvidence`、`StructuredToolResults` 与消息级 `MessageActions`；旧过程实现暂时留在仓库，便于 dev 回归对照。
2. 一致性测试通过、新能力可达、真实 dev run 回归通过后，再用独立 cleanup PR 物理删除零引用的 `AgentRunTimeline / ExecutionProcess / executionProcessModel`；`RunBanner/RunHeader/...` 按实际复用情况保留或删除。

### 8.9 P3 退出标准

- 会话壳双 Tab、唯一 Composer、Tab 往返状态保持；Trajectory Tab 全高且只声称有界视图。
- 多 turn/run attempt 按 `turn_message_id + attempt_index` 组织；未加载 Run 骨架 → 选中水合；最多 8 个 snapshot 缓存。
- P1 snapshot + SSE 合并、终态对账、reconciliation conflict、`degraded/legacy/truncated` 全部可见。
- 5000 事件性能与虚拟化门禁通过；锚点可定位未挂载行，键盘/ARIA 有效。
- Chat → Trajectory inspect 与 Trajectory → Chat message reveal 分别通过 e2e；Tab 切换不终止 SSE。
- Agent run retry/continue 只在 Trajectory，消息级 retry 保留；请求目标与 `previous_run_id` 正确，新 attempt 自动选中。
- 四组一致性闸门通过；真实 dev 新 Run 覆盖工具、知识库、失败/触顶或 continuation、刷新恢复与 console 0 error。
- Network 联动验收：顶部区段点击定位记录表并打开右侧详情，记录行点击高亮 Overview；区间拖选仅显示/聚焦时间重叠记录；可访问清除范围可用；`sequence/actual` 可切换；未水合 Run 占位、焦点替换展开与跨 Run 拖选按 §8.5 收敛；Tool Node Detail 的 `available/pending/not_recorded/degraded` 四态与 settle grace 正确。

## 9. P0 验收标准

P0 验收不依赖 P2（MVP Adapter 联调属 P2 验收）。

### 9.1 延迟与首屏基准（dev 同步基线已触发异步切换）

1. dev 真实 PostgreSQL 同步核心基线：**200 事件，p50 7.428ms、p95 10.781ms、p99 11.124ms**。p95 超过 5ms 绝对门限，已触发 P0 切换 §5.4；该记录是同步 DB 基线。
2. 切换后同场景复测，判定阈值：
   - 相对回归：p95/p99 不劣于基线的 **1.2 倍**；
   - 绝对门限：Emitter 锁内 `AgentEventCompositeWriter → QueuedTrajectoryRecorder.record_chunk()` 接纳 p95 ≤ 5ms、p99 ≤ 15ms。
   - 本地性能 runner 用每次等待 20ms 的可控同步核心验证 250 次生产包装器接纳边界；它不连接真实数据库，**不得冒充 dev DB 基线**。
3. 独立 Session 无争用：连接数、锁等待不随事件量线性增长。
4. **端到端 TTFT（有界回归）**：记录客户端实际收到首个 reasoning/content/tool SSE 的时间，与模型内部 `first_output_delta.ttft_ms` 对比；**不得只测模型内部时间**。改造前后的端到端 TTFT 差值必须落在 §9.1 阈值内；Recorder 失败（fail-open）不影响首个 SSE chunk 到达（有断言/测试）。

### 9.2 账本正确性

- **sequence 语义（统一验收口径，终审定稿）**：
  - `complete` 的 run：严格 **`0..N-1` 连续**、无空洞、无重复（emitter 从 0 开始）；
  - `degraded` 的 run：只保证**单调、唯一、不复用**（取消/熔断/丢事件可产生空洞——空洞正是 degraded 的成因之一）；
  - 取消后已分配 sequence 不复用（后续事件序号严格更大）——由「await 前预留」（§5.2）保证。
- **seal + finalize 语义**：`seal_and_get_last_sequence()` 在同一把锁内封口并返回 `_sequence - 1`；finalize 在**所有同步辅助事件（含 `suggested_questions_pending`）之后**执行；`run_completed` 不触发 finalize；seal 后任何 emit 必须抛错。
- **finalize 三断言 + latch 优先**：finalize 先查 degraded latch，置位（含取消 `recorder_cancelled`）则一律落 `degraded`（即使三断言满足）；未置位才验 `COUNT = expected+1 ∧ MIN(sequence)=0 ∧ MAX(sequence)=expected`（§5.2）——两条路径都有单测。
- **迟到事务竞态**：「最后一条事件落账超时 → 迟到提交成功 → 三断言满足」仍必须 `degraded`（latch 权威性）有专门竞态测试。
- **durable terminal intent**：每次 finalize 生成唯一 `terminal_intent_id`；事务 A 只能从 `recording + no pending` 创建 `expected + pending(v1)`，明确前置失败不得 readback；风险终态事务不清 intent；B/C/升级/readback 全部按同 token CAS，正常 complete 仅在独立 ack 后清五字段。覆盖遗留 `recording|complete|degraded + pending` 不被新 Recorder 清理、A commit 响应丢失同 token 恢复、错误 token 零影响、terminal commit 响应丢失 + 首次/持续对账失败、unknown 后 cancel、ack 提交失败与进程崩溃前 DB 状态；断言 Task 6 可扫描 `recording|complete|degraded + pending`，permit 最终守恒。
- **协调 grace 与正常 ack 保护**：`terminal_at` 是唯一业务终态年龄证据；新建/重入 running 清 NULL，所有终态写 aware UTC。stale recording 需要 `terminal_at + meta.updated_at` 都过 60 秒 grace；pending 需要 `terminal_at + pending_at + meta.updated_at` 都过 grace。覆盖「pending_at 旧但 updated_at 新」的 recording/complete/degraded 不被协调、complete 可正常 C ack；若 C 未发生且 updated_at 再过完整 grace，pending 才保守 degraded 并原子清五字段。
- **取消路径竞态**：worker 运行中 / 尚未启动时取消——permit 最终守恒（无泄漏）、迟到异常被消费、latch 保持 degraded、finalize 不得 complete、**sequence 不重复**（有专门竞态测试）。
- **原子事务**：事件 INSERT 与 meta 计数同事务（§5.1）有故障注入测试——INSERT 成功后 commit 前失败，必须整体回滚，不得出现「有事件无计数」。
- **超时熔断 + permit 生命周期**：`BoundedSemaphore(4)` 非阻塞准入——满载直接 fail-open（有测试：sem 满时新事件不排队、置 degraded）；`wait_for(250ms)` 在 DB 挂起时真正超时（非假超时）；**「任务尚未开始执行便超时」不得泄漏 permit**（有测试：连续 N 次超时后 semaphore 仍可继续 acquire，permit 数守恒）；`shield` 保证超时不取消底层任务；迟到任务异常被消费（无 never-retrieved 告警）；psycopg2 `connect_timeout` 整数秒限制已说明（500ms 不可配，主隔离靠 wait_for+闸门）。
- **取消路径**：`run_failed / run_limit_reached / run_interrupted / 用户取消 / shutdown` 都在统一 `finally` + 有界 `asyncio.shield` 中完成 seal+finalize；二次取消不得打断收口；**`CancelledError` 分支置 degraded 并 re-raise**（§5.1）有单测。
- **完整性矩阵**：成功 / 失败 / 中断 / 继续执行（retry、regenerate）后的轨迹完整性——**seal/finalize 后无新事件；run 终态事件后允许同步辅助事件**。
- **attempt_index 并发**：同 `turn_message_id` 并发 retry/continue 不产生相同 `attempt_index`（部分唯一约束 + conversation `SELECT FOR UPDATE` 原子分配）；迁移回填后历史行 `turn_message_id/attempt_index` 正确、`message_id=NULL` 行不受影响。
- **legacy 判定**：`run.created_at < ledger_enabled_at`（迁移持久化值）→ `legacy`；`>= ledger_enabled_at` 且 meta 缺失 → `degraded/meta_missing`；两类有测试覆盖。
- **meta_missing 并发安全**：新终态 run 缺 meta 仅在 `terminal_at` 过 grace 后候选；PostgreSQL `ON CONFLICT DO NOTHING` 不覆盖并发 Recorder 首写，仅实际插入行计数，并保证同批其他 stale/pending 候选提交。

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
| P0 | 协议升级（schema_version + 新增生命周期事件含 `first_output_delta` 双分支语义、`retrieval_cancelled`、cache token 提取）；`agent_events` + `run_trajectory_meta`（FK、expected/finalized、latch 优先、**durable terminal intent + pending 扫描索引**）；**AgentSession 迁移与回填（§4.4，含 `terminal_at`、UTC 终态写入与 60 秒协调 grace）**；`QueuedTrajectoryRecorder` **有界异步接纳已启用**（容量 1000、单 run 单消费者、10 秒 flush barrier）并包装 `TrajectoryRecorder` 同步核心（独立 Session + 原子事务 + **BoundedSemaphore 准入 + wait_for/shield/CancelledError/DB 超时熔断 + permit 生命周期四规则** + seal/finalize 三断言 + latch 优先 + 独立 terminal ack）；emitter `seal_and_get_last_sequence`；`turn_message_id/previous_run_id/attempt_index`（稳定 turn 分组 + 两阶段兼容 + 并发分配）；落库 allowlist 脱敏；保留策略 = 级联删除（无独立 TTL） | §9 验收全绿；不含 P2 依赖 |
| P1 | **已完成（PR #54）**：`TrajectoryProjector`（orphan 收口 + `terminal_source` + 截断保护）；普通/管理员快照 API 分离；仅历史、有界、`truncated` 明示 | P3 直接消费，不重新实施 |
| P1.1（后续可选） | Run/event cursor 分页与精确 `previous_run_id` lineage DTO | P3 首版不依赖；需要突破有界历史时再做 |
| P2 | **已完成历史验证**：独立 MVP `FusionTrajectoryAdapter` 只读 P1 快照；不进入 Fusion P3 runtime，不复制其 SQLite/SSE/投影器 | 仅作为既有联调证据保留 |
| P3 | fusion-ui 双 Tab；前端 `TrajectoryCellProjection`；Network 三区的 `TrajectoryOverview + TrajectoryTable + TrajectoryNodeDetailPanel`；Tool-only Node Detail 精确关联、四态与 settle grace；会话级有界账本；按 Run 水合/LRU；虚拟化；P1+SSE 归并与终态对账；inspect/reveal；一致性闸门；替代旧“过程” | §8.9 全部通过；旧组件 cleanup 在真实 dev 回归后独立提交 |

**P3 实施顺序（v0.21 定稿，Node Detail 后端前置以先验证真实契约）**：

1. 将 v0.21 文档迁入正式 P3 分支；
2. `ToolCallLog.tool_call_id` 迁移与写入链路（§7.4）；
3. Tool Node Detail 端点、脱敏、四态与 settle grace（§7.4）；
4. 前端 Detail client/type（§7.4 Envelope）；
5. `TrajectoryOverview` + `TrajectoryTable`（§8.5）；
6. `TrajectoryNodeDetailPanel` 接真实 API；
7. SSE 实时归并与尾部跟随（§8.2/§8.5）；
8. 一致性闸门（§8.8）；
9. dev 真实 Run 回归；
10. 最后删除旧「过程」组件（三条件齐备）。

## 11. 对抗式审查清单（终审确认用）

- [ ] 独立 Session 未被共享对象污染（全局 session、请求依赖注入）。
- [ ] emitter 锁内生产队列接纳边界延迟达标（§9.1 **有界回归**，不承诺零首屏影响）；Redis 权威事件异常未被 CompositeWriter 吞掉。
- [ ] **超时熔断可执行 + permit 无泄漏**：`BoundedSemaphore(4)` 非阻塞准入（满载 fail-open、不排队）；`wait_for(asyncio.shield(future))`——超时不取消底层任务；submit 失败立即释放；worker `finally` 释放；迟到异常被消费；「任务未开始便超时」不泄漏 permit（permit 守恒测试）。
- [ ] **`CancelledError` 分支**：捕获置 degraded（`recorder_cancelled`）→ 消费迟到 future → **re-raise**（不吞取消）；取消后已分配 sequence 不复用；「worker 运行中/未启动时取消」竞态测试覆盖 permit 守恒、latch 保持 degraded、finalize 不得 complete、sequence 不重复。
- [ ] **finalize latch 优先**：latch 置位（含取消）→ 一律 degraded（即使三断言满足）；「最后一条事件迟到提交」竞态测试存在。
- [ ] **durable terminal intent 所有权**：每次 finalize 唯一 token；A 仅从 recording + no pending 创建，rowcount=0 不 readback；B/C/升级/readback 同 token CAS；历史 recording|complete|degraded pending 不被新 Recorder 冒领。明确终态后才独立 ack；commit/ack 结果未知或进程崩溃后，Task 6 扫描三种状态的 pending 并保守收敛，unknown 不吞后到 timeout/cancel。
- [ ] **协调 grace 与并发首写**：`AgentSession.terminal_at` 迁移回填、running 清 NULL、所有终态写 aware UTC；stale/pending/meta-missing 都要求 terminal_at 过 60 秒 grace，pending 额外要求 pending_at 与 updated_at 均过 cutoff，普通 recording 要求 updated_at 过 cutoff。新鲜 B→C/C ack 不被抢占；到期 pending 原子降级并清五字段；meta_missing 用 `ON CONFLICT DO NOTHING`，不覆盖并发 Recorder 首写。
- [ ] **sequence 在第一次可取消 await 前预留**：校验完成后、await writer 前递增；取消的 emit 消耗序号形成空洞（degraded）而非复用；complete 严格 `0..N-1` 连续、degraded 单调唯一允许空洞——两侧语义有测试。
- [ ] `seal_and_get_last_sequence` 与 emit 互斥（同一把锁）；seal 后 emit 抛错有测试；取消路径 shield 有测试。
- [ ] finalize 调用点在**所有同步辅助事件之后**；`run_completed` 不触发 finalize；finalize 三断言有单测。
- [ ] 事件 INSERT 与 meta 计数同事务；故障注入整体回滚有测试。
- [ ] degraded 在 DB 不可用时仍可达（内存 latch + 协调任务）；协调器使用 PostgreSQL `FOR UPDATE SKIP LOCKED` 幂等分批，业务 running 永不处理；legacy 唯一判据是迁移持久化的 `ledger_enabled_at`；新 run 缺 meta 仅在 terminal_at 过 grace 后写 `degraded/meta_missing`。
- [ ] `first_output_delta` 只在真实 delta 时发送且 `ttft_ms` 非空；空流由 `completed.ttft_ms=null` 表达；**reasoning/content 分支可见 chunk 先于事件（顺序断言）**；**tool_call 分支不受该约束**。
- [ ] `retrieval_cancelled` 覆盖；orphan 推导带 `terminal_source/inferred_reason`，成功 run 孤儿标 `unknown/incomplete`。
- [ ] `app/ai` 无反向依赖 `app/services`；LLM round 事件由编排层发出。
- [ ] 落库 allowlist 是单点强制；新事件天然用户安全；管理员端点固定 `/api/admin/audit/.../trajectory` 接入访问审计；普通端点越权 404、不按角色切换内容。
- [ ] `turn_message_id/previous_run_id/attempt_index` 四场景全覆盖 + 两阶段兼容（阶段 A 缺省时 FOR UPDATE 内解析 + 兼容指标；阶段 B 收紧）；部分唯一约束 + 并发分配有测试。
- [ ] 迁移回填：历史 `turn_message_id = message_id`，row_number 按 `(turn_message_id, created_at, id)`、`message_id=NULL` 行处理、历史跨 assistant lineage 不伪造、`ledger_enabled_at` 由迁移持久化。
- [ ] P0 无独立 TTL；级联删除无孤儿；`expected_last_sequence / finalized_at / terminal_intent_*` 持久化完整。
- [ ] `MAX_TRAJECTORY_EVENTS_PER_RUN` 截断保护：`truncated=true`，截断数据不生成看似完整的 span。
- [ ] `schema_version` 缺失兼容；部署窗口无破坏。
- [ ] 快照 API 的 O(n) 投影在长会话下可接受；有上限保护。

## 12. 风险与开放问题

- **事件体积**：content blocks / 证据全文不入账本，摘要 + 引用；必要时按类型设 cap。
- **协议破坏性变更**：`schema_version` 加入 AgentEventBase 是破坏性的；部署窗口前端需兼容缺失版本（§3.1）。
- **cache token 提取**：供应商 usage 扩展字段格式未知，P0 单独列为新能力；提取失败降级为 null 不报错。
- **两阶段兼容**：阶段 A 的缺失率需持续监控；阶段 B 收紧前必须覆盖稳定，否则旧客户端断约。
- **同步 DB 线程池**：fusion 为同步 SQLAlchemy + psycopg2（`connect_timeout` 限整数秒），同步核心的超时隔离仍靠 `BoundedSemaphore` 准入 + `wait_for(asyncio.shield(...))` + 显式 `CancelledError` 分支；permit 泄漏由四规则 + 守恒测试兜底。主链路已由 `QueuedTrajectoryRecorder` 异步接纳隔离。
- **sequence 预留时机**：emitter 在第一次可取消 await 前预留并递增（§5.2）；取消/熔断导致的空洞只允许出现在 degraded run；complete 严格要求连续（§9.2）。
- **异步接纳延迟**：dev 同步基线已触发 §5.4；P0 当前以生产包装器接纳 p95/p99 与 dev 端到端 TTFT 作为门禁。
- **协调 grace 的可见性窗口**：终态异常最多延后约「60 秒 grace + 一次 scheduler 周期」才收敛；这是保护正常 finalize B→C/C ack 与迟到首写的有意取舍。`terminal_at` 缺失时协调器当前不得猜测或破坏性更新，只能保守跳过；为此类跳过补充不泄露敏感信息的聚合日志/指标属于后续运维项，缺少该观测能力是当前已知风险。
- **TTL 推迟**：P0 只做级联删除；独立 TTL（含 `expired` 状态）留待后续，避免「complete 但事件为空」的假象。
- **MVP 协议差异**（命名、sequence 0-based、span 含义）：由 P2 的 `FusionTrajectoryAdapter` 吸收，不反向污染 fusion API。
- **仓库状态**：`.gitignore` 例外与文档跟踪按文首决定，在正式实施分支首个提交中一并处理；不在 detached HEAD 操作。
