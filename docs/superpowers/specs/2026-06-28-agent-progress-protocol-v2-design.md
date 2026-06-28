# Agent Progress Protocol v2 设计

## 背景

Fusion 当前已经有 agent loop 基础协议。后端通过 Redis Stream 推送 `agent_event`，前端可以实时展示 run、step、tool call 和触顶终态。上一轮 continuation 已经补齐了触顶后继续执行的基础能力。

现在的缺口不在传输层，而在语义层：用户能看到“调用了工具”，但不容易看懂“任务现在做到哪、还差什么、哪些工具结果真的支撑了回答”。如果继续只把工具参数和简单 `result_summary` 推给前端，时间线会偏工程日志，不会形成可读的 agent 过程。

本设计定义 Agent Progress Protocol v2：在现有 `chunk_type=agent_event` 之内增加计划、进度、证据和工具摘要事件，并持久化 compact progress snapshot，让断线重连、刷新历史和 continuation 都能恢复用户可理解的任务时间线。

## 目标

- 保留现有 Redis Stream 到 SSE 的两段式架构，不切 WebSocket，不改变外层 envelope。
- 在 `agent_event` 内扩展 v2 事件，表达计划、进度、证据和工具结果摘要。
- 让前端不再解析 reasoning 文本或 raw tool output 来猜任务状态。
- 支持 Redis replay 下的断线重连，并支持 Redis TTL 之后的历史会话恢复。
- 保持旧客户端兼容：未知 v2 事件可以被忽略，v1 事件语义不变。
- 对用户展示的是安全、短小、可理解的摘要，不泄露 prompt、原始工具大结果、密钥、内部服务名或调试字段。

## 非目标

- 不做发送前强制“先规划、等确认、再执行”的长任务计划模式。
- 不增加额外 LLM planning call 作为首版必需路径。
- 不让用户编辑计划、跳过步骤、替换工具或手动调整预算。
- 不把完整工具 raw output 作为 progress 事件或历史快照持久化。
- 不重写现有 `run_started`、`step_started`、`tool_call_*`、`run_completed` 事件。
- 不启动本地 Fusion 服务作为验收路径；实现阶段仍使用单测、ruff、CI/CD、远端部署和正式域名 Chrome 回归。

## 现状问题

### 1. 工具过程有了，任务语义还不够

现有事件可以表达“第几步调了什么工具”，但计划和目标只存在于模型思考或最终回答里。前端没有稳定字段能显示“准备查资料、读取关键来源、整理结论”这些用户可理解阶段。

### 2. `result_summary` 偏工具元数据

当前 `tool_call_completed.result_summary` 主要服务工具 chip，例如 count、title、truncated。它不能表达关键发现、来源是否被采用、为什么这个工具结果重要。

### 3. Redis replay 不能覆盖历史刷新

Redis Stream 适合流式和断线重连，但 TTL 过后无法恢复 timeline。历史会话只拿到 message content 和最近 agent run summary，不能完整恢复计划和证据链。

### 4. continuation 需要知道还差什么

触顶 continuation 已经可以继续执行，但用户最好能看到“已完成哪些计划、还剩哪些计划、继续会补什么”，否则“继续查”仍然只是边界按钮。

## 推荐方案

采用“兼容扩展 agent_event + compact snapshot 持久化”的方案。

后端继续通过 `AgentEventEmitter` 分配单 run 内递增 `sequence`，写入 Redis Stream。v2 新事件仍走 `chunk_type=agent_event`，前端按 `type` 二级分发。与此同时，后端新增 progress recorder，把 v2 事件折叠成 compact snapshot，按 run 持久化到数据库。会话历史接口返回最近 run 的 progress snapshot，前端刷新后可以恢复可读时间线。

## 协议兼容策略

### 外层 SSE 不变

```text
id: <redis_entry_id>
data: {"chunk_type": "agent_event", "data": {...}}
```

`reasoning`、`answering`、`done`、`error` 等 chunk type 不变。

### 内层 agent_event 增量扩展

所有 v1 事件继续保留。v2 事件增加 `protocol_version: 2`。旧客户端遇到未知 `type` 会按当前逻辑 warn 并忽略。

现有 v1 事件可以继续不带 `protocol_version`，前端按 `protocol_version ?? 1` 处理。

### 顺序和幂等

- `sequence` 仍然是单 run 内严格递增。
- 前端仍以 `run_id + sequence` 防重放。
- v2 更新类事件必须包含稳定 ID 和 revision，前端可以重复应用同一 snapshot 而不重复渲染。

## 新增事件

### `run_progress_updated`

表达 run 级别当前阶段和预算进度。

```json
{
  "type": "run_progress_updated",
  "protocol_version": 2,
  "run_id": "run-1",
  "trace_id": "run-1",
  "step_id": null,
  "tool_call_id": null,
  "sequence": 3,
  "ts": 1782650000.0,
  "phase": "researching",
  "label": "正在搜索相关资料",
  "completed_steps": 1,
  "total_steps": 4,
  "completed_tool_calls": 2,
  "max_tool_calls": 20
}
```

字段约束：

- `phase`: `planning | thinking | researching | reading | synthesizing | answering | recovering`
- `label`: 面向用户的短文案，最长 40 个中文字符。
- `total_steps` 可以为空；没有明确计划时前端只显示当前阶段。
- `completed_tool_calls` 和 `max_tool_calls` 只展示预算进度，不展示内部限流原因。

### `plan_snapshot`

表达当前计划的全量快照。用于首帧、重连和修正计划。

```json
{
  "type": "plan_snapshot",
  "protocol_version": 2,
  "run_id": "run-1",
  "trace_id": "run-1",
  "step_id": null,
  "tool_call_id": null,
  "sequence": 4,
  "ts": 1782650001.0,
  "plan_id": "plan-run-1",
  "revision": 1,
  "items": [
    {
      "id": "understand",
      "title": "理解问题",
      "status": "completed",
      "kind": "reasoning",
      "summary": "已明确需要比较近期资料",
      "tool_names": [],
      "evidence_item_ids": []
    },
    {
      "id": "search",
      "title": "搜索资料",
      "status": "running",
      "kind": "search",
      "summary": "正在查找权威来源",
      "tool_names": ["web_search"],
      "evidence_item_ids": []
    }
  ]
}
```

`items[].status`: `pending | running | completed | failed | skipped | blocked`

`items[].kind`: `reasoning | search | read | synthesis | answer | other`

首版计划可以由后端根据实际 agent loop 阶段生成，不要求额外 LLM planning call。后续长任务计划模式可以复用同一个事件，把 LLM 预生成计划作为 `plan_snapshot` 发出。

### `plan_step_updated`

表达单个计划项更新，避免频繁发送全量计划。

```json
{
  "type": "plan_step_updated",
  "protocol_version": 2,
  "run_id": "run-1",
  "trace_id": "run-1",
  "step_id": "agent-step-2",
  "tool_call_id": null,
  "sequence": 8,
  "ts": 1782650004.0,
  "plan_id": "plan-run-1",
  "revision": 2,
  "item": {
    "id": "search",
    "title": "搜索资料",
    "status": "completed",
    "kind": "search",
    "summary": "找到 5 条候选来源",
    "tool_names": ["web_search"],
    "evidence_item_ids": ["ev-1", "ev-2"]
  }
}
```

前端应用规则：

- 如果不存在 `plan_id`，忽略。
- 如果 `revision` 小于等于已应用 revision，忽略。
- 如果 item 不存在，追加；存在则按 id 覆盖。

### `tool_result_digest`

表达工具结果的用户可读摘要，和 raw tool result 分离。

```json
{
  "type": "tool_result_digest",
  "protocol_version": 2,
  "run_id": "run-1",
  "trace_id": "run-1",
  "step_id": "agent-step-2",
  "tool_call_id": "tool-1",
  "sequence": 10,
  "ts": 1782650005.0,
  "tool_name": "web_search",
  "status": "success",
  "title": "找到 5 条搜索结果",
  "summary": "优先保留官方和一手来源，剔除了重复转载。",
  "key_findings": [
    "官方公告确认发布时间为 2026 年 6 月。",
    "多家媒体引用同一份原始资料。"
  ],
  "source_refs": ["ev-1", "ev-2"],
  "truncated": false
}
```

字段约束：

- `summary` 最长 120 个中文字符。
- `key_findings` 最多 5 条，每条最长 80 个中文字符。
- `source_refs` 只引用 evidence ID，不内嵌完整来源列表。
- `status`: `success | failed | degraded | interrupted`

### `evidence_item_upserted`

表达可读证据项。使用 upsert 而不是 added，是为了后续把候选证据标记为 used 或 discarded。

```json
{
  "type": "evidence_item_upserted",
  "protocol_version": 2,
  "run_id": "run-1",
  "trace_id": "run-1",
  "step_id": "agent-step-2",
  "tool_call_id": "tool-1",
  "sequence": 11,
  "ts": 1782650006.0,
  "evidence": {
    "id": "ev-1",
    "kind": "web",
    "status": "candidate",
    "title": "官方发布页",
    "url": "https://example.com/news",
    "domain": "example.com",
    "claim": "官方发布页确认了事项的发布时间。",
    "snippet": "页面摘要的安全短摘录。",
    "used_by_final_answer": false
  }
}
```

字段约束：

- `kind`: `web | file | tool | model`
- `status`: `candidate | used | discarded`
- `claim` 最长 120 个中文字符。
- `snippet` 最长 180 个中文字符；不能包含长段原文或用户隐私。
- `url` 必须是公开来源或系统已允许展示的文件引用；不展示内部服务 URL。

## 持久化设计

### 新表：`agent_progress_snapshots`

新增一张 compact snapshot 表，按 run 持久化可读进度。

字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string primary key | snapshot id |
| `run_id` | string index unique | 对应 `agent_sessions.id` |
| `conversation_id` | string index | 会话 id |
| `message_id` | string index | assistant message id |
| `user_id` | string index | 用户 id |
| `protocol_version` | integer | 当前为 2 |
| `state` | JSONB | compact progress state |
| `created_at` | datetime | 创建时间 |
| `updated_at` | datetime | 更新时间 |

索引：

- `uq_agent_progress_snapshots_run_id`
- `ix_agent_progress_message_updated` on `(conversation_id, message_id, updated_at)`

### `state` 结构

```json
{
  "run_id": "run-1",
  "message_id": "msg-1",
  "status": "running",
  "progress": {
    "phase": "researching",
    "label": "正在搜索相关资料",
    "completed_steps": 1,
    "total_steps": 4,
    "completed_tool_calls": 2,
    "max_tool_calls": 20
  },
  "plan": {
    "plan_id": "plan-run-1",
    "revision": 2,
    "items": []
  },
  "tool_digests": [],
  "evidence": [],
  "updated_at": "2026-06-28T21:30:00+08:00"
}
```

存储限制：

- 每个 run 最多保留 12 个 evidence item。
- 每个 run 最多保留 20 个 tool digest。
- 超限时保留 `used` 和最近项，丢弃低价值 candidate。
- 所有字符串字段入库前走长度裁剪。

### 写入策略

新增 `AgentProgressRecorder`，作为 `AgentEventEmitter` 的旁路 sink 使用。推荐通过 `AgentEventCompositeWriter` 接入：

1. `AgentEventEmitter` 仍然只负责构造事件、分配 sequence、调用 writer。
2. `AgentEventCompositeWriter.append_chunk()` 先写 Redis Stream，再把 payload 交给 recorder。
3. recorder 把 v2 事件折叠进内存 state，并在这些时机写 DB：
   - `plan_snapshot`
   - `plan_step_updated`
   - `tool_result_digest`
   - `evidence_item_upserted`
   - `run_completed`
   - `run_failed`
   - `run_interrupted`
4. recorder 写入失败只记录 warning，不中断主生成链路。

这样可以避免把 DB session 塞进 emitter，也避免让 Redis writer 知道 progress state 的细节。

## 后端模块边界

新增或修改建议：

- `app/services/agent/events.py`
  - 新增 v2 Pydantic event models。
  - `AnyAgentEvent` union 纳入 v2 事件。
- `app/services/agent/emitter.py`
  - 新增方法：`run_progress_updated()`、`plan_snapshot()`、`plan_step_updated()`、`tool_result_digest()`、`evidence_item_upserted()`。
  - 仍然保持 sequence lock。
- `app/services/agent/progress_state.py`
  - 纯函数 reducer：`apply_progress_event(state, event)`。
  - 负责幂等、revision、裁剪。
- `app/services/agent/progress_recorder.py`
  - 接收事件 payload，维护 snapshot state，写 `agent_progress_snapshots`。
- `app/services/stream/tool_executor.py`
  - 从 `ToolExecutionRecord` 派生 `tool_result_digest` 和 `evidence_item_upserted`。
- `app/services/stream/agent_loop_lifecycle.py`
  - run 开始后发初始 `run_progress_updated` 和首个 `plan_snapshot`。
- `app/services/stream/step_lifecycle.py`
  - step 开始/完成时同步更新计划项和 run progress。
- `app/db/repositories.py`
  - `AgentRunSummary` 增加 progress snapshot 字段，返回给会话详情。

## 计划生成策略

首版不引入额外 LLM planning call。计划由后端根据 agent loop 能力和实际状态生成：

- run 开始：生成默认计划 `理解问题`、`查找资料`、`读取关键来源`、`整理回答`。
- 如果当前模型或请求没有工具，计划降级为 `理解问题`、`整理回答`。
- 进入 tool call round 时，把对应计划项标为 running。
- 工具完成时，用 digest/evidence 更新计划项 summary。
- 最终回答开始时，把 `整理回答` 标为 running。
- run 终态时，把仍 running 的计划项派生为 completed、failed、blocked 或 skipped。

后续长任务计划模式可以把 LLM 生成的计划替换默认计划，但仍使用同一组事件和前端 UI。

## 安全和隐私边界

- v2 事件只允许面向用户的摘要，不允许写入完整 prompt、system prompt、原始 HTML、完整网页正文、工具原始 JSON 或 API key。
- `tool_result_digest` 和 `evidence` 必须经过统一 sanitizer。
- URL 只展示公开目标 URL；内部 reader-service URL、Redis key、容器名、trace 内部路径不进入事件。
- 错误信息对普通用户只保留短文案；底层异常继续留在日志或管理员诊断接口。
- snippet 不能超过 180 个中文字符，避免长段版权内容进入 SSE 和数据库。

## 错误处理

- v2 event emit 失败：记录 warning，不影响 v1 事件、正文 token 和最终回答。
- progress snapshot 写 DB 失败：Redis/SSE 仍继续；历史恢复缺失时前端显示 v1 timeline 或 content evidence。
- 前端未知 v2 事件：忽略并 warn，不影响回答。
- revision 倒退：前端和 recorder 都忽略。
- evidence 超限：保留 used 项和最近 candidate，设置 `truncated=true`。

## 测试计划

### 后端单元测试

- `test/services/agent/test_events.py`
  - v2 event models 校验必填字段、枚举和 extra forbid。
- `test/services/agent/test_emitter.py`
  - v2 事件 sequence 单调。
  - run-level v2 事件 `step_id` 为空，step-level v2 事件继承当前 step。
- `test/services/agent/test_progress_state.py`
  - `plan_snapshot` 覆盖旧计划。
  - `plan_step_updated` 按 revision 幂等。
  - evidence upsert 不重复。
  - 超限裁剪保留 used 项。
- `test/services/agent/test_progress_recorder.py`
  - recorder 折叠 v2 事件并写 snapshot。
  - DB 写失败不抛到调用方。

### 后端集成测试

- `test/services/stream/test_agent_loop_contract.py`
  - agent run 会发初始 progress 和 plan snapshot。
  - 工具完成后发 digest 和 evidence。
  - Redis Stream replay 中 v1/v2 事件顺序稳定。
  - run 完成后 snapshot status 更新为 completed。
- `test/test_repositories.py`
  - conversation detail 返回 latest agent run progress snapshot。
  - 无 progress snapshot 的历史消息兼容返回 `progress=null`。

### 迁移测试

- Alembic migration 能在 SQLite 测试库创建新表。
- `ruff check .` 不报导入顺序问题。

## 验收标准

- 普通短对话不因为 v2 事件增加额外 LLM 调用。
- Agent run 的 SSE 中同时存在 v1 timeline 事件和 v2 progress 事件。
- 前端断线重连可通过 Redis replay 恢复 plan/evidence。
- Redis TTL 之后，conversation detail 仍能返回 compact progress snapshot。
- continuation 新 run 能生成自己的 progress snapshot，且归属同一 assistant message。
- 旧前端忽略未知 v2 事件后仍可正常显示回答。

## 分阶段实施

### 第一阶段：协议和持久化

- 新增 v2 event models、emitter 方法、progress state reducer、snapshot 表。
- 在 run/step/tool 生命周期中发出最小可用 v2 事件。
- conversation detail 暴露 latest progress snapshot。

### 第二阶段：前端可读时间线

- 前端解析 v2 事件并写 Redux。
- `AgentRunTimeline` 增加计划、进度、证据摘要展示。
- 历史 hydration 支持 `agent_run.progress`。

### 第三阶段：长任务计划模式

- 只在协议稳定后引入发送前计划模式。
- 可以复用 `plan_snapshot`，但需要单独设计交互节奏、成本和用户确认策略。
