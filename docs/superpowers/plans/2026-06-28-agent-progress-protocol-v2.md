# Agent Progress Protocol v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 Redis Stream/SSE agent_event 协议内增加 v2 进度、计划、工具摘要和证据事件，并持久化 compact progress snapshot 供历史会话恢复。

**Architecture:** `AgentEventEmitter` 继续作为唯一事件发送方，新增 v2 Pydantic 事件和 emitter 方法；`progress_state.py` 负责纯函数折叠与裁剪，`AgentProgressRecorder` 作为 writer 旁路 sink 写入 `agent_progress_snapshots`。生命周期模块只发语义事件，不直接知道 DB snapshot 细节。

**Tech Stack:** FastAPI, SQLAlchemy, Alembic, PostgreSQL JSONB, Redis Stream, Pydantic v2, pytest/unittest, ruff.

---

## 文件结构

- 修改 `app/services/agent/events.py`：新增 v2 事件模型、枚举和 `AnyAgentEvent` union。
- 修改 `app/services/agent/emitter.py`：新增 v2 emit 方法，保持单 run sequence lock。
- 新建 `app/services/agent/progress_state.py`：纯函数 reducer、字段裁剪、snapshot 限额。
- 新建 `app/services/agent/progress_recorder.py`：接收 agent_event payload，折叠并 upsert snapshot。
- 修改 `app/services/stream/tool_executor.py`：新增 `AgentEventCompositeWriter`，并从工具执行记录派生 digest/evidence。
- 修改 `app/services/stream/agent_loop_execution.py`：构造 composite writer 与 recorder。
- 修改 `app/services/stream/agent_loop_lifecycle.py`：run 开始发初始 progress 和默认 plan。
- 修改 `app/services/stream/step_lifecycle.py`：step 开始/完成发 plan/progress 更新。
- 修改 `app/services/stream/run_finalizer.py`：终态前发 progress/plan 收尾事件。
- 修改 `app/db/models.py`：新增 `AgentProgressSnapshot` ORM。
- 新建 Alembic migration：`alembic/versions/9c1f2a3b4d5e_add_agent_progress_snapshots.py`，`down_revision = "4c6a1f2b8d90"`。
- 修改 `app/schemas/chat.py`：`AgentRunSummary` 增加 `progress`。
- 修改 `app/db/repositories.py`：会话详情返回最新 run 的 progress snapshot。
- 新增/修改测试：
  - `test/services/agent/test_events.py`
  - `test/services/agent/test_emitter.py`
  - `test/services/agent/test_progress_state.py`
  - `test/services/agent/test_progress_recorder.py`
  - `test/services/stream/test_agent_loop_contract.py`
  - `test/services/stream/test_step_lifecycle.py`
  - `test/test_tool_executor.py`
  - `test/test_repositories.py`

## 实施约束

- 不改变外层 SSE envelope：仍是 `chunk_type=agent_event`。
- 不新增首轮 LLM planning call。
- v2 写入失败只记录 warning，不中断正文 token、v1 事件和最终回答。
- 所有用户可见摘要必须短文本，不写入 prompt、raw HTML、raw tool JSON、内部 URL、密钥或容器/Redis 细节。
- 本阶段不启动本地 Fusion 服务；验证只跑 pytest、ruff、migration 相关命令和后续 CI/CD。
- 为减少 CI/CD 浪费，本计划执行时按测试检查点推进，最终合并为一个 API 功能提交。

## Task 1: v2 事件模型

**Files:**
- Modify: `app/services/agent/events.py`
- Modify: `test/services/agent/test_events.py`

- [ ] **Step 1: 写失败测试**

在 `test/services/agent/test_events.py` 追加：

```python
class AgentProgressV2EventModelTests(unittest.TestCase):
    def _common(self):
        return dict(run_id="r1", step_id=None, tool_call_id=None, sequence=0, trace_id="r1", ts=1.0)

    def test_run_progress_updated_requires_protocol_version_2(self):
        from app.services.agent.events import RunProgressUpdated

        ev = RunProgressUpdated(
            type="run_progress_updated",
            protocol_version=2,
            phase="researching",
            label="正在搜索相关资料",
            completed_steps=1,
            total_steps=4,
            completed_tool_calls=2,
            max_tool_calls=20,
            **self._common(),
        )

        self.assertEqual(ev.protocol_version, 2)
        self.assertEqual(ev.phase, "researching")

    def test_plan_snapshot_forbids_unknown_fields(self):
        from app.services.agent.events import PlanSnapshot

        with self.assertRaises(ValidationError):
            PlanSnapshot(
                type="plan_snapshot",
                protocol_version=2,
                plan_id="plan-r1",
                revision=1,
                items=[],
                unexpected=True,
                **self._common(),
            )

    def test_tool_result_digest_and_evidence_models(self):
        from app.services.agent.events import EvidenceItemUpserted, ToolResultDigest

        digest = ToolResultDigest(
            type="tool_result_digest",
            protocol_version=2,
            step_id="s1",
            tool_call_id="tc1",
            tool_name="web_search",
            status="success",
            title="找到 2 条结果",
            summary="优先保留官方来源。",
            key_findings=["官方页面确认发布时间。"],
            source_refs=["ev-1"],
            truncated=False,
            **{k: v for k, v in self._common().items() if k not in {"step_id", "tool_call_id"}},
        )
        self.assertEqual(digest.status, "success")

        evidence = EvidenceItemUpserted(
            type="evidence_item_upserted",
            protocol_version=2,
            step_id="s1",
            tool_call_id="tc1",
            evidence={
                "id": "ev-1",
                "kind": "web",
                "status": "candidate",
                "title": "官方发布页",
                "url": "https://example.com/news",
                "domain": "example.com",
                "claim": "官方发布页确认发布时间。",
                "snippet": "页面摘要。",
                "used_by_final_answer": False,
            },
            **{k: v for k, v in self._common().items() if k not in {"step_id", "tool_call_id"}},
        )
        self.assertEqual(evidence.evidence.id, "ev-1")
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
python -m pytest test/services/agent/test_events.py -q
```

Expected: FAIL，原因是 `RunProgressUpdated`、`PlanSnapshot`、`ToolResultDigest`、`EvidenceItemUpserted` 还不存在。

- [ ] **Step 3: 实现事件模型**

在 `app/services/agent/events.py` 增加：

```python
AgentProgressPhase = Literal["planning", "thinking", "researching", "reading", "synthesizing", "answering", "recovering"]
AgentPlanItemStatus = Literal["pending", "running", "completed", "failed", "skipped", "blocked"]
AgentPlanItemKind = Literal["reasoning", "search", "read", "synthesis", "answer", "other"]


class AgentPlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    status: AgentPlanItemStatus
    kind: AgentPlanItemKind
    summary: str | None = None
    tool_names: list[str] = Field(default_factory=list)
    evidence_item_ids: list[str] = Field(default_factory=list)


class AgentEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["web", "file", "tool", "model"]
    status: Literal["candidate", "used", "discarded"]
    title: str
    url: str | None = None
    domain: str | None = None
    claim: str
    snippet: str | None = None
    used_by_final_answer: bool = False
```

并新增五个事件类：`RunProgressUpdated`、`PlanSnapshot`、`PlanStepUpdated`、`ToolResultDigest`、`EvidenceItemUpserted`，全部带 `protocol_version: Literal[2]`，再加入 `AnyAgentEvent`。

- [ ] **Step 4: 验证通过**

Run:

```bash
python -m pytest test/services/agent/test_events.py -q
```

Expected: PASS。

## Task 2: emitter v2 方法和 sequence 语义

**Files:**
- Modify: `app/services/agent/emitter.py`
- Modify: `test/services/agent/test_emitter.py`

- [ ] **Step 1: 写失败测试**

在 `test/services/agent/test_emitter.py` 追加：

```python
    async def test_v2_events_use_same_sequence_stream(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", redis_writer=writer)

        await em.run_started(message_id="m1", model="gpt", tools=["web_search"], config={"max_steps": 8})
        await em.run_progress_updated(phase="planning", label="正在理解问题", completed_steps=0, total_steps=4)
        await em.plan_snapshot(plan_id="plan-r1", revision=1, items=[
            {"id": "understand", "title": "理解问题", "status": "running", "kind": "reasoning", "tool_names": [], "evidence_item_ids": []}
        ])

        events = [call.args[2] for call in writer.append_chunk.call_args_list]
        self.assertEqual([event["sequence"] for event in events], [0, 1, 2])
        self.assertEqual(events[1]["type"], "run_progress_updated")
        self.assertEqual(events[1]["protocol_version"], 2)
        self.assertIsNone(events[1]["step_id"])

    async def test_step_level_plan_update_inherits_current_step(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", redis_writer=writer)
        step_id = await em.step_started(step_number=1)

        await em.plan_step_updated(
            plan_id="plan-r1",
            revision=2,
            item={"id": "search", "title": "搜索资料", "status": "completed", "kind": "search", "tool_names": ["web_search"], "evidence_item_ids": []},
        )

        payload = writer.append_chunk.call_args_list[-1].args[2]
        self.assertEqual(payload["type"], "plan_step_updated")
        self.assertEqual(payload["step_id"], step_id)
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
python -m pytest test/services/agent/test_emitter.py -q
```

Expected: FAIL，原因是 emitter 尚无 v2 方法。

- [ ] **Step 3: 实现最小 emitter 方法**

在 `AgentEventEmitter` 增加方法：

```python
    async def run_progress_updated(self, *, phase: str, label: str, completed_steps: int | None = None,
                                   total_steps: int | None = None, completed_tool_calls: int | None = None,
                                   max_tool_calls: int | None = None) -> None:
        await self._emit(ev.RunProgressUpdated(
            type="run_progress_updated",
            protocol_version=2,
            phase=phase,
            label=label,
            completed_steps=completed_steps,
            total_steps=total_steps,
            completed_tool_calls=completed_tool_calls,
            max_tool_calls=max_tool_calls,
            **self._envelope(step_id=None),
        ))
```

同样实现 `plan_snapshot()`、`plan_step_updated()`、`tool_result_digest()`、`evidence_item_upserted()`。run-level progress/snapshot 显式 `step_id=None`；step/tool 级事件默认继承 current step 或传入 `tool_call_id`。

- [ ] **Step 4: 验证通过**

Run:

```bash
python -m pytest test/services/agent/test_emitter.py -q
```

Expected: PASS。

## Task 3: progress state reducer 和裁剪规则

**Files:**
- Create: `app/services/agent/progress_state.py`
- Create: `test/services/agent/test_progress_state.py`

- [ ] **Step 1: 写失败测试**

新建 `test/services/agent/test_progress_state.py`：

```python
from app.services.agent.progress_state import apply_progress_event, empty_progress_state


def test_plan_snapshot_replaces_existing_plan():
    state = empty_progress_state(run_id="r1", message_id="m1")
    state = apply_progress_event(state, {
        "type": "plan_snapshot",
        "protocol_version": 2,
        "plan_id": "plan-r1",
        "revision": 1,
        "items": [{"id": "a", "title": "理解问题", "status": "running", "kind": "reasoning", "tool_names": [], "evidence_item_ids": []}],
    })
    state = apply_progress_event(state, {
        "type": "plan_snapshot",
        "protocol_version": 2,
        "plan_id": "plan-r1",
        "revision": 2,
        "items": [{"id": "b", "title": "整理回答", "status": "pending", "kind": "answer", "tool_names": [], "evidence_item_ids": []}],
    })

    assert state["plan"]["revision"] == 2
    assert [item["id"] for item in state["plan"]["items"]] == ["b"]


def test_plan_step_update_ignores_stale_revision():
    state = empty_progress_state(run_id="r1", message_id="m1")
    state = apply_progress_event(state, {"type": "plan_snapshot", "protocol_version": 2, "plan_id": "plan-r1", "revision": 2, "items": []})
    stale = apply_progress_event(state, {
        "type": "plan_step_updated",
        "protocol_version": 2,
        "plan_id": "plan-r1",
        "revision": 2,
        "item": {"id": "search", "title": "搜索资料", "status": "running", "kind": "search", "tool_names": [], "evidence_item_ids": []},
    })

    assert stale["plan"]["items"] == []


def test_evidence_and_tool_digest_upsert_and_cap():
    state = empty_progress_state(run_id="r1", message_id="m1")
    for index in range(14):
        state = apply_progress_event(state, {
            "type": "evidence_item_upserted",
            "protocol_version": 2,
            "evidence": {
                "id": f"ev-{index}",
                "kind": "web",
                "status": "used" if index == 0 else "candidate",
                "title": "t" * 80,
                "domain": "example.com",
                "claim": "c" * 200,
                "snippet": "s" * 300,
                "used_by_final_answer": index == 0,
            },
        })

    ids = [item["id"] for item in state["evidence"]]
    assert len(ids) == 12
    assert "ev-0" in ids
    assert state["evidence"][0]["claim"] == "c" * 120
    assert len(state["evidence"][0]["snippet"]) == 180
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
python -m pytest test/services/agent/test_progress_state.py -q
```

Expected: FAIL，原因是 `progress_state.py` 不存在。

- [ ] **Step 3: 实现 reducer**

创建 `app/services/agent/progress_state.py`，导出：

```python
MAX_EVIDENCE_ITEMS = 12
MAX_TOOL_DIGESTS = 20


def empty_progress_state(*, run_id: str, message_id: str) -> dict:
    return {
        "run_id": run_id,
        "message_id": message_id,
        "status": "running",
        "progress": None,
        "plan": None,
        "tool_digests": [],
        "evidence": [],
        "updated_at": None,
    }
```

`apply_progress_event(state, event)` 只处理 v2 事件和终态 v1 事件。实现细节：

- `run_progress_updated` 覆盖 `state["progress"]`。
- `plan_snapshot` 覆盖 `state["plan"]`。
- `plan_step_updated` 在 `revision > current revision` 时按 `item.id` upsert。
- `tool_result_digest` 按 `tool_call_id` upsert，最多 20 条。
- `evidence_item_upserted` 按 `evidence.id` upsert，最多 12 条，保留 `used/used_by_final_answer` 和最近 candidate。
- `run_completed/run_failed/run_interrupted` 更新 `state["status"]`。
- 字符串裁剪：`label=40`、`summary=120`、`key_findings[]=80`、`claim=120`、`snippet=180`、标题不超过 80。

- [ ] **Step 4: 验证通过**

Run:

```bash
python -m pytest test/services/agent/test_progress_state.py -q
```

Expected: PASS。

## Task 4: snapshot 表、migration 和 recorder

**Files:**
- Modify: `app/db/models.py`
- Create: `alembic/versions/9c1f2a3b4d5e_add_agent_progress_snapshots.py`
- Create: `app/services/agent/progress_recorder.py`
- Create: `test/services/agent/test_progress_recorder.py`

- [ ] **Step 1: 写 recorder 失败测试**

新建 `test/services/agent/test_progress_recorder.py`，用 fake session 验证行为：

```python
from unittest.mock import Mock

from app.services.agent.progress_recorder import AgentProgressRecorder


def test_recorder_ignores_non_agent_event_chunks():
    db = Mock()
    recorder = AgentProgressRecorder(db=db, run_id="r1", conversation_id="c1", message_id="m1", user_id="u1")

    recorder.record_chunk("c1", "answering", {"delta": "x"})

    db.merge.assert_not_called()


def test_recorder_upserts_snapshot_for_v2_event():
    db = Mock()
    recorder = AgentProgressRecorder(db=db, run_id="r1", conversation_id="c1", message_id="m1", user_id="u1")

    recorder.record_chunk("c1", "agent_event", {
        "type": "run_progress_updated",
        "protocol_version": 2,
        "phase": "planning",
        "label": "正在理解问题",
    })

    db.merge.assert_called_once()
    db.commit.assert_called_once()
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
python -m pytest test/services/agent/test_progress_recorder.py -q
```

Expected: FAIL，原因是 recorder/model 尚不存在。

- [ ] **Step 3: 新增 ORM 和 migration**

在 `app/db/models.py` 增加：

```python
class AgentProgressSnapshot(Base):
    """Agent 可读进度 compact snapshot — 按 run_id 唯一保存最新折叠状态"""

    __tablename__ = "agent_progress_snapshots"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String, ForeignKey("agent_sessions.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    message_id = Column(String, nullable=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    protocol_version = Column(Integer, nullable=False, default=2)
    state = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=get_china_time, index=True)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time, index=True)

    __table_args__ = (
        Index("ix_agent_progress_message_updated", "conversation_id", "message_id", "updated_at"),
    )
```

Migration 文件头使用：

```python
revision: str = "9c1f2a3b4d5e"
down_revision: Union[str, Sequence[str], None] = "4c6a1f2b8d90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None
```

Migration 内容使用 `op.create_table()` 创建同名表、`op.create_index()` 创建 `ix_agent_progress_message_updated`，并在 downgrade 删除索引和表。

- [ ] **Step 4: 实现 recorder**

`AgentProgressRecorder` 构造参数：`db`、`run_id`、`conversation_id`、`message_id`、`user_id`、`logger`。方法：

```python
def record_chunk(self, conversation_id: str, chunk_type: str, payload: dict) -> None:
    if conversation_id != self.conversation_id or chunk_type != "agent_event":
        return
    next_state = apply_progress_event(self._state, payload)
    if next_state is self._state:
        return
    self._state = next_state
    self._upsert_snapshot()
```

`_upsert_snapshot()` 查询或 merge `AgentProgressSnapshot`，`commit()`；异常时 `rollback()` 并 warning，不 re-raise。

- [ ] **Step 5: 验证通过**

Run:

```bash
python -m pytest test/services/agent/test_progress_recorder.py -q
```

Expected: PASS。

## Task 5: writer 接入和生命周期最小 v2 事件

**Files:**
- Modify: `app/services/stream/tool_executor.py`
- Modify: `app/services/stream/agent_loop_execution.py`
- Modify: `app/services/stream/agent_loop_lifecycle.py`
- Modify: `app/services/stream/step_lifecycle.py`
- Modify: `app/services/stream/run_finalizer.py`
- Modify: `test/services/stream/test_agent_loop_contract.py`
- Modify: `test/services/stream/test_step_lifecycle.py`

- [ ] **Step 1: 写失败测试：lifecycle 初始 progress/plan**

在 `test/services/stream/test_agent_loop_contract.py` 增加断言：一次 agent run 的事件序列中，`run_started` 后出现 `run_progress_updated` 和 `plan_snapshot`，且不改变现有 v1 事件顺序约束。

核心断言：

```python
types = [payload["type"] for payload in emitted_agent_events]
assert "run_started" in types
assert "run_progress_updated" in types
assert "plan_snapshot" in types
assert types.index("run_started") < types.index("run_progress_updated") < types.index("plan_snapshot")
```

- [ ] **Step 2: 写失败测试：step 更新 plan/progress**

在 `test/services/stream/test_step_lifecycle.py` 增加 fake emitter，要求 `start_agent_step()` 调用 `plan_step_updated()`，`complete_agent_step()` 调用 `run_progress_updated()`。

- [ ] **Step 3: 运行测试确认失败**

Run:

```bash
python -m pytest test/services/stream/test_agent_loop_contract.py test/services/stream/test_step_lifecycle.py -q
```

Expected: FAIL，原因是生命周期尚未发 v2 事件。

- [ ] **Step 4: 实现 composite writer**

在 `app/services/stream/tool_executor.py` 增加：

```python
class AgentEventCompositeWriter:
    def __init__(self, *, redis_writer: AgentEventRedisWriter, recorder=None) -> None:
        self.redis_writer = redis_writer
        self.recorder = recorder

    async def append_chunk(self, conversation_id: str, chunk_type: str, payload: dict) -> None:
        await self.redis_writer.append_chunk(conversation_id, chunk_type, payload)
        if self.recorder is not None:
            self.recorder.record_chunk(conversation_id, chunk_type, payload)
```

在 `agent_loop_execution.py` 使用 recorder 包装 writer；测试里没有 DB 时保留原 writer。

- [ ] **Step 5: 实现默认计划工具**

在 lifecycle 内部增加私有函数：

```python
def _default_plan_items(tools: list[str]) -> list[dict]:
    if tools:
        return [
            {"id": "understand", "title": "理解问题", "status": "running", "kind": "reasoning", "tool_names": [], "evidence_item_ids": []},
            {"id": "search", "title": "查找资料", "status": "pending", "kind": "search", "tool_names": tools, "evidence_item_ids": []},
            {"id": "read", "title": "读取关键来源", "status": "pending", "kind": "read", "tool_names": tools, "evidence_item_ids": []},
            {"id": "answer", "title": "整理回答", "status": "pending", "kind": "answer", "tool_names": [], "evidence_item_ids": []},
        ]
    return [
        {"id": "understand", "title": "理解问题", "status": "running", "kind": "reasoning", "tool_names": [], "evidence_item_ids": []},
        {"id": "answer", "title": "整理回答", "status": "pending", "kind": "answer", "tool_names": [], "evidence_item_ids": []},
    ]
```

在 `_start_run()` 的 `start_agent_run_fn` 之后发：

```python
await execution.emitter.run_progress_updated(phase="planning", label="正在理解问题", completed_steps=0, total_steps=len(plan_items), completed_tool_calls=0, max_tool_calls=request.limits.max_tool_calls)
await execution.emitter.plan_snapshot(plan_id=f"plan-{execution.run_id}", revision=1, items=plan_items)
```

- [ ] **Step 6: step 和终态更新**

`start_agent_step()` 进入工具轮时把 `search/read` 标记 running；`complete_agent_step()` 根据 `tool_call_count` 更新 completed steps/tool count；`run_finalizer.py` 在 completed/failed/interrupted/limit 前把仍 running 的 plan item 收尾为 `completed/blocked/failed`。

- [ ] **Step 7: 验证通过**

Run:

```bash
python -m pytest test/services/stream/test_agent_loop_contract.py test/services/stream/test_step_lifecycle.py -q
```

Expected: PASS。

## Task 6: 工具结果 digest 和 evidence 派生

**Files:**
- Create: `app/services/agent/progress_digest.py`
- Modify: `app/services/stream/tool_executor.py`
- Modify: `test/test_tool_executor.py`

- [ ] **Step 1: 写失败测试**

在 `test/test_tool_executor.py` 增加一个 fake emitter 测试：工具成功后会收到 `tool_result_digest()`，web/search 类结果会收到不超过 12 条 `evidence_item_upserted()`。

核心断言：

```python
assert emitter.tool_result_digest.await_count == 1
payload = emitter.tool_result_digest.await_args.kwargs
assert payload["tool_call_id"] == "tc1"
assert payload["summary"]
assert len(payload["key_findings"]) <= 5
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
python -m pytest test/test_tool_executor.py -q
```

Expected: FAIL，原因是工具执行尚未发 digest/evidence。

- [ ] **Step 3: 实现 digest helper**

`progress_digest.py` 导出：

```python
def build_tool_result_digest(record: ToolExecutionRecord) -> dict:
    return {
        "tool_call_id": record.tool_call["id"],
        "tool_name": record.tool_call["name"],
        "status": _map_status(record.result.status),
        "title": _safe_title(record),
        "summary": _safe_summary(record),
        "key_findings": _safe_findings(record)[:5],
        "source_refs": [],
        "truncated": bool(getattr(record.result, "truncated", False)),
    }


def build_evidence_items(record: ToolExecutionRecord) -> list[dict]:
    return _extract_public_sources(record)[:12]
```

helper 只从 handler summary、公开 URL/title/domain/snippet 取安全短文本；无法提取时返回空 evidence，但仍发 digest。

- [ ] **Step 4: 在 tool executor 发事件**

`execute_one_tool_call()` 返回 record 前，如果 `request.emitter` 存在：

```python
digest = build_tool_result_digest(record)
await request.emitter.tool_result_digest(**digest)
for evidence in build_evidence_items(record):
    await request.emitter.evidence_item_upserted(tool_call_id=tool_call["id"], evidence=evidence)
```

预算失败路径和未知工具路径也发 digest，status 映射为 `failed` 或 `degraded`。

- [ ] **Step 5: 验证通过**

Run:

```bash
python -m pytest test/test_tool_executor.py -q
```

Expected: PASS。

## Task 7: repository 和 API schema 暴露 progress

**Files:**
- Modify: `app/schemas/chat.py`
- Modify: `app/db/repositories.py`
- Modify: `test/test_repositories.py`

- [ ] **Step 1: 写失败测试**

在 `test/test_repositories.py` 增加：当某 assistant message 有最新 `AgentSession` 和对应 `AgentProgressSnapshot` 时，`message.agent_run.progress` 等于 snapshot state。

核心断言：

```python
message = repo.get_conversation(conv_id, user_id=user_id).messages[-1]
assert message.agent_run.run_id == "run-new"
assert message.agent_run.progress["plan"]["plan_id"] == "plan-run-new"
```

另加兼容测试：没有 snapshot 时 `progress is None`。

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
python -m pytest test/test_repositories.py -q
```

Expected: FAIL，原因是 schema 和 repository 尚未返回 progress。

- [ ] **Step 3: 修改 schema/repository**

`AgentRunSummary` 增加：

```python
progress: Optional[Dict[str, Any]] = None
```

`_latest_agent_runs_for_messages()` 查询最新 run 后收集 `run_ids`，再批量查 `AgentProgressSnapshot.run_id.in_(run_ids)`，构造 `snapshots_by_run_id`，然后填入：

```python
progress=snapshots_by_run_id.get(row.id).state if row.id in snapshots_by_run_id else None
```

- [ ] **Step 4: 验证通过**

Run:

```bash
python -m pytest test/test_repositories.py -q
```

Expected: PASS。

## Task 8: 后端集成验证和提交

**Files:**
- All backend files above.

- [ ] **Step 1: 跑目标测试集**

Run:

```bash
python -m pytest \
  test/services/agent/test_events.py \
  test/services/agent/test_emitter.py \
  test/services/agent/test_progress_state.py \
  test/services/agent/test_progress_recorder.py \
  test/services/stream/test_agent_loop_contract.py \
  test/services/stream/test_step_lifecycle.py \
  test/test_tool_executor.py \
  test/test_repositories.py \
  -q
```

Expected: PASS。

- [ ] **Step 2: 跑后端全量检查**

Run:

```bash
python -m pytest test/ -q
python -m ruff check .
python -m ruff format --check .
git diff --check
```

Expected: 全部 exit 0。

- [ ] **Step 3: 自审要求覆盖**

逐项确认：

- SSE 外层 envelope 未变。
- v1 事件仍发出，sequence 与 v2 共享单调序列。
- recorder 失败不影响 Redis writer。
- snapshot 字段裁剪和数量上限生效。
- repository 无 snapshot 时兼容旧消息。
- 没有新增本地服务启动命令或依赖本地 dev 数据。

- [ ] **Step 4: 提交 API 功能**

Run:

```bash
git status --short
git add app/services/agent/events.py app/services/agent/emitter.py app/services/agent/progress_state.py app/services/agent/progress_recorder.py app/services/agent/progress_digest.py app/services/stream/tool_executor.py app/services/stream/agent_loop_execution.py app/services/stream/agent_loop_lifecycle.py app/services/stream/step_lifecycle.py app/services/stream/run_finalizer.py app/db/models.py app/schemas/chat.py app/db/repositories.py alembic/versions/*_add_agent_progress_snapshots.py test/services/agent/test_events.py test/services/agent/test_emitter.py test/services/agent/test_progress_state.py test/services/agent/test_progress_recorder.py test/services/stream/test_agent_loop_contract.py test/services/stream/test_step_lifecycle.py test/test_tool_executor.py test/test_repositories.py
git commit -m "feat: 支持 agent progress protocol v2" -m "Co-Authored-By: Codex <noreply@anthropic.com>"
```

Expected: 生成一个 API 功能提交。
