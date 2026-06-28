# Agent Run 继续执行 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `limit_reached` 的 agent 回答增加“继续查”，用新的 run 续写同一条 assistant 消息。

**Architecture:** 后端新增 continuation API，校验原 assistant message 和最近一次 `limit_reached` run 后，用同一 `assistant_message_id` 启动新的 agent loop。agent loop 支持传入已有 content blocks 和 continuation system prompt；前端用独立 hook 调 continuation SSE，并用 streamSlice 的 static blocks 保留旧内容后追加新 delta。

**Tech Stack:** FastAPI、SQLAlchemy、Alembic、Redis Stream、pytest、Next.js 15、React 19、Redux Toolkit、Vitest、Testing Library。

---

## Scope Check

这份 spec 横跨 `fusion-api` 和 `fusion-ui`，但它是一个单一用户能力：触顶后继续同一条 assistant 消息。后端 API 和前端 CTA 必须一起交付才能形成可用闭环，因此保持一个 implementation plan。实施时使用一个工作分支；最终按子仓分别提交，避免把半成品多次推送触发流水线。

## File Structure

### fusion-api

- Create: `alembic/versions/4c6a1f2b8d90_add_agent_session_config.py`
  - 给 `agent_sessions` 增加 `config` JSONB，用于持久化上一轮预算。
- Modify: `app/db/models.py`
  - `AgentSession` 增加 `run_config = Column("config", JSONB, nullable=True)`。
- Modify: `app/services/agent/session_cache.py`
  - `write_session_started()` 接收并写入 `run_config`。
- Modify: `app/services/stream/run_finalizer.py`
  - `start_agent_run()` 把 `config` 同步传给 session cache。
- Create: `app/services/agent/continuation.py`
  - 封装 continuation 权限前置数据、最近触顶 run 查询、预算恢复、content block 反序列化和 continuation prompt 注入。
- Modify: `app/schemas/chat.py`
  - 新增 `ContinueAgentRunRequest`。
- Modify: `app/services/chat_service.py`
  - 新增 `continue_agent_run()`，复用 StreamHandler 启动 SSE。
- Modify: `app/api/chat.py`
  - 新增 `POST /api/chat/conversations/{conversation_id}/messages/{message_id}/continue`。
- Modify: `app/services/stream/agent_loop_wiring.py`
  - `AgentLoopRunInput` 增加已有 content blocks、额外 system prompts、是否预处理用户输入。
- Modify: `app/services/stream/agent_loop_lifecycle.py`
  - 生命周期 request 携带初始 content blocks 和额外 system prompts。
- Modify: `app/services/stream/agent_loop_request_prep.py`
  - 支持跳过 file/url 预处理并注入 continuation system prompt。
- Modify: `app/services/stream/runner.py`
  - `generate_to_redis()` 接收 continuation 参数并传入 wiring。
- Test: `test/services/agent/test_continuation.py`
- Test: `test/services/stream/test_agent_loop_request_prep.py`
- Test: `test/test_chat_continue.py`
- Test: `test/test_stream_handler.py`

### fusion-ui

- Modify: `src/lib/api/chat.ts`
  - 新增 `continueAgentRunStream()`，复用 SSE parser。
- Modify: `src/redux/slices/streamSlice.ts`
  - `startStream` 支持 `staticBlocks`，selectors 返回旧 blocks + 新增流式 blocks。
- Create: `src/hooks/useContinueAgentRun.ts`
  - 封装 continuation API 调用、SSE callbacks、消息更新和错误处理。
- Modify: `src/components/chat/agent/RunBanner.tsx`
  - `limit_reached` 三种 reason 显示“继续查”。
- Modify: `src/components/chat/agent/AgentRunTimeline.tsx`
  - 增加 `onContinue`，保留 `onRetry` 给失败/不完整路径。
- Modify: `src/components/chat/AssistantResponseStack.tsx`
  - 传递 `onContinueAgentRun`。
- Modify: `src/components/chat/AssistantMessage.tsx`
  - 接收并传递 continuation handler。
- Modify: `src/components/chat/ChatMessage.tsx`
  - 接收并传递 continuation handler。
- Modify: `src/components/chat/ChatMessageList.tsx`
  - 接收并传递 continuation handler。
- Modify: `src/app/(app)/chat/[chatId]/page.tsx`
  - 使用 `useContinueAgentRun()` 接线。
- Test: `src/lib/api/chat.test.ts`
- Test: `src/redux/slices/streamSlice.test.ts`
- Test: `src/components/chat/agent/RunBanner.test.tsx`
- Test: `src/components/chat/AssistantResponseStack.test.tsx`
- Test: `src/hooks/useContinueAgentRun.test.ts`
- Test: `src/app/(app)/chat/[chatId]/page.test.tsx`

---

## Task 1: 持久化 agent run config

**Files:**
- Create: `fusion-api/alembic/versions/4c6a1f2b8d90_add_agent_session_config.py`
- Modify: `fusion-api/app/db/models.py`
- Modify: `fusion-api/app/services/agent/session_cache.py`
- Modify: `fusion-api/app/services/stream/run_finalizer.py`
- Test: `fusion-api/test/services/agent/test_session_cache.py`
- Test: `fusion-api/test/services/stream/test_agent_loop_lifecycle.py`

- [ ] **Step 1: 写 migration**

Create `fusion-api/alembic/versions/4c6a1f2b8d90_add_agent_session_config.py`:

```python
"""add agent session config

Revision ID: 4c6a1f2b8d90
Revises: 3b4c8a7d2f10
Create Date: 2026-06-28
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "4c6a1f2b8d90"
down_revision: Union[str, Sequence[str], None] = "3b4c8a7d2f10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_sessions",
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_sessions", "config")
```

- [ ] **Step 2: 更新 ORM model**

In `fusion-api/app/db/models.py`, inside `class AgentSession` after `provider`:

```python
    run_config = Column("config", JSONB, nullable=True)
```

- [ ] **Step 3: 更新 session cache 协议和实现**

In `fusion-api/app/services/agent/session_cache.py`, change `write_session_started` signature:

```python
async def write_session_started(
    *,
    run_id: str,
    conversation_id: str,
    user_id: str,
    model_id: str,
    provider: str,
    message_id: str | None = None,
    run_config: dict | None = None,
) -> None:
```

Set `existing.run_config = run_config` in the existing-row path, and pass `run_config=run_config` when creating `AgentSession`.

- [ ] **Step 4: 更新 run finalizer protocol**

In `fusion-api/app/services/stream/run_finalizer.py`, update `AgentRunSessionCache.write_session_started` protocol:

```python
    async def write_session_started(
        self,
        *,
        run_id: str,
        conversation_id: str,
        user_id: str,
        model_id: str,
        provider: str,
        message_id: str,
        run_config: dict | None = None,
    ) -> None: ...
```

In `start_agent_run()`, pass:

```python
        run_config=config,
```

- [ ] **Step 5: 写 session cache 测试**

Create or extend `fusion-api/test/services/agent/test_session_cache.py` with a DB-free unit around the model mutation by monkeypatching `SessionLocal` to a fake context manager:

```python
import pytest

from app.db.models import AgentSession
from app.services.agent import session_cache


class FakeSession:
    def __init__(self):
        self.row = None
        self.added = None
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, model, key):
        assert model is AgentSession
        return self.row

    def add(self, row):
        self.added = row

    def commit(self):
        self.commits += 1


@pytest.mark.asyncio
async def test_write_session_started_persists_run_config(monkeypatch):
    fake = FakeSession()
    monkeypatch.setattr(session_cache, "SessionLocal", lambda: fake)

    await session_cache.write_session_started(
        run_id="run-1",
        conversation_id="conv-1",
        user_id="user-1",
        model_id="deepseek-chat",
        provider="deepseek",
        message_id="msg-1",
        run_config={"max_steps": 8, "max_tool_calls": 20, "timeout_s": 300},
    )

    assert fake.added.run_config == {"max_steps": 8, "max_tool_calls": 20, "timeout_s": 300}
    assert fake.commits == 1
```

- [ ] **Step 6: 运行后端 focused tests**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest test/services/agent/test_session_cache.py test/services/stream/test_agent_loop_lifecycle.py -q
```

Expected: tests pass after implementation. Before implementation, new test fails because `run_config` is not written.

---

## Task 2: 后端 continuation 上下文服务

**Files:**
- Create: `fusion-api/app/services/agent/continuation.py`
- Test: `fusion-api/test/services/agent/test_continuation.py`

- [ ] **Step 1: 写 continuation 单测**

Create `fusion-api/test/services/agent/test_continuation.py`:

```python
import pytest

from app.schemas.chat import TextBlock
from app.services.agent.continuation import (
    CONTINUATION_SYSTEM_PROMPT,
    deserialize_content_blocks,
    inject_continuation_prompt,
    resolve_continuation_limits,
)
from app.services.stream.agent_loop_policy import AgentLoopLimits


def test_deserialize_content_blocks_preserves_existing_block_id():
    blocks = deserialize_content_blocks([
        {"type": "text", "id": "blk_old", "text": "旧回答"},
    ])

    assert blocks == [TextBlock(type="text", id="blk_old", text="旧回答")]


def test_inject_continuation_prompt_after_existing_system_messages():
    messages = [
        {"role": "system", "content": "用户自定义系统提示"},
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "旧回答"},
    ]

    result = inject_continuation_prompt(messages)

    assert result[0]["content"] == "用户自定义系统提示"
    assert result[1] == {"role": "system", "content": CONTINUATION_SYSTEM_PROMPT}
    assert result[2]["role"] == "user"


def test_resolve_continuation_limits_uses_session_config():
    session = type("Session", (), {
        "run_config": {"max_steps": 4, "max_tool_calls": 7, "timeout_s": 90},
    })()

    limits = resolve_continuation_limits(
        session,
        default_limits=AgentLoopLimits(max_steps=8, max_tool_calls=20, total_timeout_s=300),
    )

    assert limits == AgentLoopLimits(max_steps=4, max_tool_calls=7, total_timeout_s=90)


def test_resolve_continuation_limits_falls_back_to_default_for_missing_config():
    session = type("Session", (), {"run_config": None})()
    default_limits = AgentLoopLimits(max_steps=8, max_tool_calls=20, total_timeout_s=300)

    assert resolve_continuation_limits(session, default_limits=default_limits) == default_limits
```

- [ ] **Step 2: 运行测试验证失败**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest test/services/agent/test_continuation.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.agent.continuation'`.

- [ ] **Step 3: 实现 continuation helper**

Create `fusion-api/app/services/agent/continuation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter
from sqlalchemy.orm import Session

from app.db.models import AgentSession, Message as MessageModel
from app.schemas.chat import ContentBlock
from app.schemas.response import ApiException
from app.services.stream.agent_loop_policy import AgentLoopLimits

CONTINUATION_SYSTEM_PROMPT = (
    "你正在继续上一轮因运行上限而停止的回答。请基于已有对话、已有回答和已有工具结果继续补充，"
    "不要重写或总结已完成的部分。若需要更多资料，可以继续调用可用工具。"
    "输出应自然衔接在上一段回答之后。"
)

_CONTENT_BLOCKS_ADAPTER = TypeAdapter(list[ContentBlock])


@dataclass(frozen=True)
class AgentContinuationContext:
    assistant_message: MessageModel
    previous_session: AgentSession
    limits: AgentLoopLimits
    initial_content_blocks: list[ContentBlock]


def deserialize_content_blocks(raw_blocks: list[dict[str, Any]] | None) -> list[ContentBlock]:
    return _CONTENT_BLOCKS_ADAPTER.validate_python(raw_blocks or [])


def inject_continuation_prompt(messages: list[dict]) -> list[dict]:
    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    prompt = {"role": "system", "content": CONTINUATION_SYSTEM_PROMPT}
    return [*messages[:insert_at], prompt, *messages[insert_at:]]


def resolve_continuation_limits(session: AgentSession, *, default_limits: AgentLoopLimits) -> AgentLoopLimits:
    config = session.run_config if isinstance(session.run_config, dict) else {}
    try:
        return AgentLoopLimits(
            max_steps=int(config.get("max_steps", default_limits.max_steps)),
            max_tool_calls=int(config.get("max_tool_calls", default_limits.max_tool_calls)),
            total_timeout_s=float(config.get("timeout_s", default_limits.total_timeout_s)),
        )
    except (TypeError, ValueError):
        return default_limits


def find_latest_limit_reached_session(
    db: Session,
    *,
    conversation_id: str,
    message_id: str,
    previous_run_id: str | None = None,
) -> AgentSession:
    query = db.query(AgentSession).filter(
        AgentSession.conversation_id == conversation_id,
        AgentSession.message_id == message_id,
        AgentSession.status == "limit_reached",
    )
    if previous_run_id:
        query = query.filter(AgentSession.id == previous_run_id)
    session = query.order_by(AgentSession.created_at.desc()).first()
    if session is None:
        raise ApiException.bad_request("这条回答当前不能继续执行")
    return session


def build_continuation_context(
    db: Session,
    *,
    conversation_id: str,
    message_id: str,
    previous_run_id: str | None,
    default_limits: AgentLoopLimits,
) -> AgentContinuationContext:
    assistant_message = db.query(MessageModel).filter(
        MessageModel.id == message_id,
        MessageModel.conversation_id == conversation_id,
        MessageModel.role == "assistant",
    ).first()
    if assistant_message is None:
        raise ApiException.not_found("会话消息不存在或无权访问")

    previous_session = find_latest_limit_reached_session(
        db,
        conversation_id=conversation_id,
        message_id=message_id,
        previous_run_id=previous_run_id,
    )

    return AgentContinuationContext(
        assistant_message=assistant_message,
        previous_session=previous_session,
        limits=resolve_continuation_limits(previous_session, default_limits=default_limits),
        initial_content_blocks=deserialize_content_blocks(assistant_message.content),
    )
```

- [ ] **Step 4: 运行 continuation tests**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest test/services/agent/test_continuation.py -q
```

Expected: PASS.

---

## Task 3: agent loop 支持初始内容和 continuation prompt

**Files:**
- Modify: `fusion-api/app/services/stream/agent_loop_wiring.py`
- Modify: `fusion-api/app/services/stream/agent_loop_lifecycle.py`
- Modify: `fusion-api/app/services/stream/agent_loop_request_prep.py`
- Modify: `fusion-api/app/services/stream/runner.py`
- Test: `fusion-api/test/services/stream/test_agent_loop_request_prep.py`
- Test: `fusion-api/test/services/stream/test_agent_loop_lifecycle.py`

- [ ] **Step 1: 写 request prep 测试**

Extend `fusion-api/test/services/stream/test_agent_loop_request_prep.py`:

```python
async def test_prepare_agent_loop_messages_injects_extra_system_prompts_without_user_preprocess():
    async def fake_build_llm_messages(raw_messages, has_vision, file_repo, user_system_prompt):
        return [{"role": "user", "content": "原问题"}, {"role": "assistant", "content": "旧回答"}]

    async def should_not_preprocess_url(*args, **kwargs):
        raise AssertionError("continuation 不应重新跑 URL 预处理")

    result = await prepare_agent_loop_messages(
        db=object(),
        user_id="user-1",
        raw_messages=[],
        has_vision=False,
        file_ids=["file-1"],
        original_message="https://example.com",
        call_config=AgentLoopCallConfig(
            should_use_reasoning=False,
            supports_function_calling=True,
            call_kwargs={"tools": []},
            announced_tools=[],
        ),
        file_repo_factory=lambda db: object(),
        load_user_system_prompt_fn=lambda db, user_id: None,
        build_llm_messages_fn=fake_build_llm_messages,
        preprocess_url_in_message_fn=should_not_preprocess_url,
        preprocess_user_input=False,
        extra_system_prompts=["继续执行，不要重写前文"],
    )

    assert result.initial_content_blocks == []
    assert result.messages[0] == {"role": "system", "content": "继续执行，不要重写前文"}
    assert result.messages[1]["role"] == "user"
```

- [ ] **Step 2: 运行测试验证失败**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest test/services/stream/test_agent_loop_request_prep.py::test_prepare_agent_loop_messages_injects_extra_system_prompts_without_user_preprocess -q
```

Expected: FAIL because `prepare_agent_loop_messages()` does not accept `preprocess_user_input` or `extra_system_prompts`.

- [ ] **Step 3: 扩展 request prep**

In `fusion-api/app/services/stream/agent_loop_request_prep.py`, add parameters to `prepare_agent_loop_messages()`:

```python
    preprocess_user_input: bool = True,
    extra_system_prompts: list[str] | None = None,
```

After `build_llm_messages_fn(...)`, insert prompts:

```python
    messages = inject_extra_system_prompts(messages, extra_system_prompts or [])
```

Wrap file/url preprocess:

```python
    if preprocess_user_input:
        messages = _inject_non_image_file_contents(...)
        messages, initial_content_blocks = await _prepare_url_context(...)
    else:
        initial_content_blocks = []
```

Add helper:

```python
def inject_extra_system_prompts(messages: list[dict], prompts: list[str]) -> list[dict]:
    if not prompts:
        return messages
    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    prompt_messages = [{"role": "system", "content": prompt} for prompt in prompts]
    return [*messages[:insert_at], *prompt_messages, *messages[insert_at:]]
```

- [ ] **Step 4: 扩展 wiring request dataclasses**

In `fusion-api/app/services/stream/agent_loop_wiring.py`, add fields to `AgentLoopRunInput`:

```python
    initial_content_blocks: list | None = None
    extra_system_prompts: list[str] | None = None
    preprocess_user_input: bool = True
```

Add the same fields to `AgentLoopLifecycleRequest` construction:

```python
            initial_content_blocks=self.initial_content_blocks or [],
            extra_system_prompts=self.extra_system_prompts or [],
            preprocess_user_input=self.preprocess_user_input,
```

- [ ] **Step 5: 扩展 lifecycle request**

In `fusion-api/app/services/stream/agent_loop_lifecycle.py`, add fields:

```python
    initial_content_blocks: list
    extra_system_prompts: list[str]
    preprocess_user_input: bool
```

In `_run_success_path()`, replace:

```python
    execution.state.content_blocks.extend(prepared_messages.initial_content_blocks)
```

with:

```python
    execution.state.content_blocks.extend(request.initial_content_blocks)
    execution.state.content_blocks.extend(prepared_messages.initial_content_blocks)
```

In `_prepare_messages()`, pass:

```python
        extra_system_prompts=request.extra_system_prompts,
        preprocess_user_input=request.preprocess_user_input,
```

- [ ] **Step 6: 扩展 StreamHandler 参数**

In `fusion-api/app/services/stream/runner.py`, add optional parameters to `generate_to_redis()`:

```python
        initial_content_blocks: Optional[list] = None,
        extra_system_prompts: Optional[list[str]] = None,
        preprocess_user_input: bool = True,
```

Pass them into `AgentLoopRunInput`.

- [ ] **Step 7: 运行 stream focused tests**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest test/services/stream/test_agent_loop_request_prep.py test/services/stream/test_agent_loop_lifecycle.py test/services/stream/test_agent_loop_wiring.py -q
```

Expected: PASS.

---

## Task 4: 后端 continuation API

**Files:**
- Modify: `fusion-api/app/schemas/chat.py`
- Modify: `fusion-api/app/services/chat_service.py`
- Modify: `fusion-api/app/api/chat.py`
- Test: `fusion-api/test/test_chat_continue.py`
- Test: `fusion-api/test/test_stream_handler.py`

- [ ] **Step 1: 写 API/service 行为测试**

Create `fusion-api/test/test_chat_continue.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.response import ApiException
from app.services.chat_service import ChatService


@pytest.mark.asyncio
async def test_continue_agent_run_reuses_assistant_message_id():
    db = MagicMock()
    service = ChatService(db)
    conversation = type("Conversation", (), {
        "id": "conv-1",
        "user_id": "user-1",
        "model_id": "deepseek-chat",
        "messages": [],
    })()
    service.conversation_service.get_conversation = MagicMock(return_value=conversation)

    continuation_context = type("ContinuationContext", (), {
        "initial_content_blocks": [],
        "limits": type("Limits", (), {
            "max_steps": 8,
            "max_tool_calls": 20,
            "total_timeout_s": 300,
        })(),
    })()

    with patch("app.services.chat_service.llm_manager.resolve_model", return_value=("deepseek/deepseek-chat", "deepseek", {})), \
        patch("app.services.chat_service.litellm_catalog.get_capabilities", return_value={"functionCalling": True}), \
        patch("app.services.chat_service.build_continuation_context", return_value=continuation_context), \
        patch("app.services.chat_service.init_stream", new=AsyncMock()) as init_stream_mock, \
        patch("app.services.chat_service.register_task") as register_task_mock, \
        patch("app.services.chat_service.asyncio.create_task") as create_task_mock:
        create_task_mock.return_value = object()

        response = await service.continue_agent_run(
            conversation_id="conv-1",
            assistant_message_id="msg-1",
            user_id="user-1",
            previous_run_id="run-old",
            trace_id="trace-1",
        )

    init_stream_mock.assert_awaited_once()
    assert init_stream_mock.await_args.args[3] == "msg-1"
    register_task_mock.assert_called_once()
    assert response.media_type == "text/event-stream"


@pytest.mark.asyncio
async def test_continue_agent_run_rejects_missing_conversation():
    service = ChatService(MagicMock())
    service.conversation_service.get_conversation = MagicMock(return_value=None)

    with pytest.raises(ApiException):
        await service.continue_agent_run(
            conversation_id="missing",
            assistant_message_id="msg-1",
            user_id="user-1",
            previous_run_id=None,
            trace_id="trace-1",
        )
```

- [ ] **Step 2: 运行测试验证失败**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest test/test_chat_continue.py -q
```

Expected: FAIL because `ChatService.continue_agent_run` does not exist.

- [ ] **Step 3: 新增 schema**

In `fusion-api/app/schemas/chat.py`, add near `ChatRequest`:

```python
class ContinueAgentRunRequest(BaseModel):
    previous_run_id: Optional[str] = None
    stream: bool = True
```

- [ ] **Step 4: 实现 ChatService.continue_agent_run**

In `fusion-api/app/services/chat_service.py`, import:

```python
from app.services.agent.continuation import (
    CONTINUATION_SYSTEM_PROMPT,
    build_continuation_context,
)
from app.services.stream.runner import _agent_loop_limits
```

Add method:

```python
    async def continue_agent_run(
        self,
        *,
        conversation_id: str,
        assistant_message_id: str,
        user_id: str,
        previous_run_id: str | None = None,
        trace_id: str | None = None,
    ) -> StreamingResponse:
        conversation = self.conversation_service.get_conversation(conversation_id, user_id)
        if not conversation:
            raise ApiException.not_found("会话不存在或无权访问")

        model_id = conversation.model_id
        litellm_model, provider, litellm_kwargs = llm_manager.resolve_model(model_id)
        capabilities = litellm_catalog.get_capabilities(model_id)
        has_vision = capabilities.get("vision", False)

        continuation = build_continuation_context(
            self.db,
            conversation_id=conversation_id,
            message_id=assistant_message_id,
            previous_run_id=previous_run_id,
            default_limits=_agent_loop_limits(),
        )

        task_id = str(uuid_mod.uuid4())
        await init_stream(conversation_id, str(user_id), model_id, assistant_message_id, task_id)

        task = asyncio.create_task(
            self.stream_handler.generate_to_redis(
                conversation_id=conversation_id,
                user_id=user_id,
                model_id=model_id,
                litellm_model=litellm_model,
                litellm_kwargs=litellm_kwargs,
                provider=provider,
                raw_messages=conversation.messages,
                has_vision=has_vision,
                file_ids=None,
                original_message="",
                assistant_message_id=assistant_message_id,
                task_id=task_id,
                options={},
                capabilities=capabilities,
                trace_id=trace_id,
                initial_content_blocks=continuation.initial_content_blocks,
                extra_system_prompts=[CONTINUATION_SYSTEM_PROMPT],
                preprocess_user_input=False,
                limits=continuation.limits,
            )
        )
        register_task(conversation_id, task, task_id)

        return StreamingResponse(
            stream_redis_as_sse(conversation_id=conversation_id, message_id=assistant_message_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
```

This step also requires `StreamHandler.generate_to_redis()` to accept an optional `limits` override. Add:

```python
        limits: Optional[AgentLoopLimits] = None,
```

and pass `limits=limits or _agent_loop_limits()` into `build_agent_loop_lifecycle_call`.

- [ ] **Step 5: 新增 route**

In `fusion-api/app/api/chat.py`, import `ContinueAgentRunRequest` and add before message diagnostics route:

```python
@router.post("/conversations/{conversation_id}/messages/{message_id}/continue")
async def continue_agent_run(
    conversation_id: str,
    message_id: str,
    continue_request: ContinueAgentRunRequest,
    request: Request,
    chat_service: ChatService = Depends(get_chat_service),
    current_user: User = Depends(get_current_user),
):
    if not continue_request.stream:
        raise ApiException.bad_request("continue 仅支持流式响应")
    return await chat_service.continue_agent_run(
        conversation_id=conversation_id,
        assistant_message_id=message_id,
        user_id=current_user.id,
        previous_run_id=continue_request.previous_run_id,
        trace_id=request.state.request_id,
    )
```

- [ ] **Step 6: 补并发 409**

Before `init_stream`, call `get_stream_meta(conversation_id)` and if current status is `"streaming"`, raise:

```python
raise ApiException.conflict("当前会话已有回答正在生成，请结束后再继续")
```

If `ApiException.conflict` does not exist, add it in `app/schemas/response.py` following existing factory style and test it in the focused API test.

- [ ] **Step 7: 运行后端 API tests**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest test/test_chat_continue.py test/test_stream_handler.py -q
```

Expected: PASS.

---

## Task 5: 前端 streamSlice 支持旧内容静态块

**Files:**
- Modify: `fusion-ui/src/redux/slices/streamSlice.ts`
- Test: `fusion-ui/src/redux/slices/streamSlice.test.ts`

- [ ] **Step 1: 写 reducer/selector 测试**

Extend `fusion-ui/src/redux/slices/streamSlice.test.ts`:

```ts
import reducer, {
  appendTextDelta,
  selectFullStreamContentBlocks,
  startStream,
} from './streamSlice';

it('keeps static blocks before continuation deltas', () => {
  let state = reducer(undefined, startStream({
    conversationId: 'conv-1',
    messageId: 'msg-1',
    staticBlocks: [{ type: 'text', id: 'blk_old', text: '旧回答' }],
  }));

  state = reducer(state, appendTextDelta({
    blockId: 'blk_new',
    delta: '新补充',
    runId: 'run-2',
    stepId: 'step-1',
  }));

  expect(selectFullStreamContentBlocks(state)).toEqual([
    { type: 'text', id: 'blk_old', text: '旧回答' },
    { type: 'text', id: 'blk_new', text: '新补充' },
  ]);
});
```

- [ ] **Step 2: 运行测试验证失败**

Run:

```bash
cd /Users/sean/code/fusion/fusion-ui
npm test -- streamSlice.test.ts
```

Expected: FAIL because `startStream` does not accept `staticBlocks`.

- [ ] **Step 3: 实现 staticBlocks**

In `fusion-ui/src/redux/slices/streamSlice.ts`, add to `StreamState`:

```ts
  staticBlocks: ContentBlock[];
```

Add to `initialState`:

```ts
  staticBlocks: [],
```

Change `startStream` payload:

```ts
action: PayloadAction<{ conversationId: string; messageId: string; staticBlocks?: ContentBlock[] }>
```

Inside reducer:

```ts
state.staticBlocks = action.payload.staticBlocks ?? [];
```

Update `selectStreamContentBlocks()` and `selectFullStreamContentBlocks()` to start with:

```ts
const blocks: ContentBlock[] = [...state.staticBlocks];
```

- [ ] **Step 4: 运行 streamSlice tests**

Run:

```bash
cd /Users/sean/code/fusion/fusion-ui
npm test -- streamSlice.test.ts
```

Expected: PASS.

---

## Task 6: 前端 continuation API 和 hook

**Files:**
- Modify: `fusion-ui/src/lib/api/chat.ts`
- Create: `fusion-ui/src/hooks/useContinueAgentRun.ts`
- Test: `fusion-ui/src/lib/api/chat.test.ts`
- Test: `fusion-ui/src/hooks/useContinueAgentRun.test.ts`

- [ ] **Step 1: 写 API client 测试**

Extend `fusion-ui/src/lib/api/chat.test.ts`:

```ts
import { continueAgentRunStream } from './chat';

it('continueAgentRunStream posts to continuation endpoint', async () => {
  const callbacks = buildCallbacks();
  fetchMock.mockResponseOnce(sse([
    agentEvent({ type: 'run_started', run_id: 'run-2', message_id: 'msg-1', conversation_id: 'conv-1', config: { max_steps: 8, max_tool_calls: 20, timeout_s: 300 } }),
    done(),
  ]));

  await continueAgentRunStream({
    conversationId: 'conv-1',
    messageId: 'msg-1',
    previousRunId: 'run-1',
  }, callbacks);

  expect(fetchMock).toHaveBeenCalledWith(
    expect.stringContaining('/api/chat/conversations/conv-1/messages/msg-1/continue'),
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ previous_run_id: 'run-1', stream: true }),
    }),
  );
});
```

Use existing test helpers in `chat.test.ts`; if helper names differ, add local helpers that emit valid SSE envelope strings.

- [ ] **Step 2: 实现 API client**

In `fusion-ui/src/lib/api/chat.ts`, add:

```ts
export interface ContinueAgentRunRequest {
  conversationId: string;
  messageId: string;
  previousRunId?: string;
}

export async function continueAgentRunStream(
  data: ContinueAgentRunRequest,
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetchWithAuth(
    `${API_BASE_URL}/api/chat/conversations/${encodeURIComponent(data.conversationId)}/messages/${encodeURIComponent(data.messageId)}/continue`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal,
      body: JSON.stringify({
        previous_run_id: data.previousRunId ?? null,
        stream: true,
      }),
    },
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const body = errorData as { code?: string; message?: string; detail?: string };
    throw new Error(body.message || body.detail || '继续执行失败');
  }
  if (!response.body) throw new Error('响应体为空');

  const reader = response.body.getReader();
  await parseSseEnvelopeStream(reader, callbacks, {
    fallbackConversationId: data.conversationId,
    doneConversationId: () => data.conversationId,
  });
}
```

- [ ] **Step 3: 写 hook 测试**

Create `fusion-ui/src/hooks/useContinueAgentRun.test.ts` with a minimal renderHook test:

```ts
import { renderHook, act } from '@testing-library/react';
import { vi, expect, it } from 'vitest';
import { useContinueAgentRun } from './useContinueAgentRun';

vi.mock('@/lib/api/chat', () => ({
  continueAgentRunStream: vi.fn(),
  getConversation: vi.fn(),
}));

it('starts continuation stream with existing assistant content', async () => {
  const dispatch = vi.fn();
  const store = {
    getState: () => ({
      conversation: {
        byId: {
          'conv-1': {
            id: 'conv-1',
            messages: [
              { id: 'msg-1', role: 'assistant', content: [{ type: 'text', id: 'blk_old', text: '旧回答' }] },
            ],
          },
        },
      },
      stream: {
        currentRun: null,
        staticBlocks: [{ type: 'text', id: 'blk_old', text: '旧回答' }],
        textBlocks: {},
        thinkingBlocks: {},
        blockOrder: [],
        blockTypes: {},
      },
    }),
  };

  const { continueAgentRunStream } = await import('@/lib/api/chat');
  vi.mocked(continueAgentRunStream).mockImplementation(async (_payload, callbacks) => {
    callbacks.onReady({ messageId: 'msg-1', conversationId: 'conv-1' });
    callbacks.onDone({ messageId: 'msg-1', conversationId: 'conv-1' });
  });

  const { result } = renderHook(() => useContinueAgentRun({ dispatch, store: store as never }));

  await act(async () => {
    await result.current.continueAgentRun({
      conversationId: 'conv-1',
      assistantMessageId: 'msg-1',
      previousRunId: 'run-1',
    });
  });

  expect(dispatch).toHaveBeenCalledWith(expect.objectContaining({
    type: 'stream/startStream',
    payload: expect.objectContaining({
      conversationId: 'conv-1',
      messageId: 'msg-1',
      staticBlocks: [{ type: 'text', id: 'blk_old', text: '旧回答' }],
    }),
  }));
});
```

- [ ] **Step 4: 实现 hook**

Create `fusion-ui/src/hooks/useContinueAgentRun.ts`. Keep it focused:

```ts
import { useCallback, useRef } from 'react';
import { useStore } from 'react-redux';
import { useAppDispatch } from '@/redux/hooks';
import {
  appendTextDelta,
  appendThinkingDelta,
  completeThinkingPhase,
  endStream,
  finalizeRun,
  finalizeStep,
  finalizeToolCall,
  initRun,
  markLimitReached,
  pushStep,
  pushToolCall,
  selectFullStreamContentBlocks,
  setStreamError,
  startStream,
} from '@/redux/slices/streamSlice';
import { updateMessage } from '@/redux/slices/conversationSlice';
import { continueAgentRunStream, getConversation } from '@/lib/api/chat';
import { getRunStatusFromFinishReason } from '@/lib/agent/finishReason';
import type { FinalizeToolCallStatus, LimitReachedReason, ToolCallResultSummary } from '@/types/agentRun';
import type { Conversation } from '@/types/conversation';

interface ContinueAgentRunInput {
  conversationId: string;
  assistantMessageId: string;
  previousRunId?: string;
}

interface HookDeps {
  dispatch?: ReturnType<typeof useAppDispatch>;
  store?: ReturnType<typeof useStore>;
}

export function useContinueAgentRun(deps: HookDeps = {}) {
  const realDispatch = useAppDispatch();
  const realStore = useStore();
  const dispatch = deps.dispatch ?? realDispatch;
  const store = deps.store ?? realStore;
  const abortControllerRef = useRef<AbortController | null>(null);

  const continueAgentRun = useCallback(async ({
    conversationId,
    assistantMessageId,
    previousRunId,
  }: ContinueAgentRunInput) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const state = store.getState() as { conversation: { byId: Record<string, Conversation> }, stream: any };
    const conversation = state.conversation.byId[conversationId];
    const assistantMessage = conversation?.messages.find(m => m.id === assistantMessageId);
    const staticBlocks = assistantMessage?.content ?? [];

    const controller = new AbortController();
    abortControllerRef.current = controller;
    dispatch(startStream({ conversationId, messageId: assistantMessageId, staticBlocks }));

    try {
      await continueAgentRunStream({ conversationId, messageId: assistantMessageId, previousRunId }, {
        onReady: () => {},
        onReasoning: payload => dispatch(appendThinkingDelta({ blockId: payload.block_id, delta: payload.delta, runId: payload.run_id, stepId: payload.step_id })),
        onAnswering: payload => {
          const streamState = (store.getState() as { stream: any }).stream;
          if (streamState.isStreamingReasoning) dispatch(completeThinkingPhase());
          dispatch(appendTextDelta({ blockId: payload.block_id, delta: payload.delta, runId: payload.run_id, stepId: payload.step_id }));
        },
        onRunStarted: ev => dispatch(initRun({
          runId: ev.run_id,
          messageId: assistantMessageId,
          serverMessageId: ev.message_id,
          config: {
            maxSteps: (ev.config.max_steps as number) ?? 0,
            maxToolCalls: (ev.config.max_tool_calls as number) ?? 0,
            timeoutS: (ev.config.timeout_s as number) ?? 0,
          },
          sequence: ev.sequence,
        })),
        onStepStarted: ev => {
          if (ev.step_id) dispatch(pushStep({ runId: ev.run_id, stepId: ev.step_id, stepNumber: ev.step_number, sequence: ev.sequence }));
        },
        onToolCallStarted: ev => {
          if (ev.step_id && ev.tool_call_id) dispatch(pushToolCall({ runId: ev.run_id, stepId: ev.step_id, toolCallId: ev.tool_call_id, toolName: ev.tool_name, arguments: ev.arguments, sequence: ev.sequence }));
        },
        onToolCallDelta: () => {},
        onToolCallCompleted: ev => {
          if (ev.tool_call_id) dispatch(finalizeToolCall({ runId: ev.run_id, toolCallId: ev.tool_call_id, status: ev.status as FinalizeToolCallStatus, durationMs: ev.duration_ms, resultSummary: ev.result_summary as unknown as ToolCallResultSummary | undefined, error: ev.error ?? null, sequence: ev.sequence }));
        },
        onStepCompleted: ev => {
          if (ev.step_id) dispatch(finalizeStep({ runId: ev.run_id, stepId: ev.step_id, sequence: ev.sequence }));
        },
        onRunLimitReached: ev => dispatch(markLimitReached({ runId: ev.run_id, reason: ev.reason as LimitReachedReason, sequence: ev.sequence })),
        onRunInterrupted: ev => dispatch(finalizeRun({ runId: ev.run_id, status: 'interrupted', reason: ev.reason, sequence: ev.sequence })),
        onRunFailed: ev => dispatch(finalizeRun({ runId: ev.run_id, status: 'failed', failure: { code: ev.error_code, message: ev.message }, sequence: ev.sequence })),
        onRunCompleted: ev => dispatch(finalizeRun({ runId: ev.run_id, status: getRunStatusFromFinishReason(ev.finish_reason), sequence: ev.sequence })),
        onDone: async () => {
          const streamState = (store.getState() as { stream: any }).stream;
          const finalBlocks = selectFullStreamContentBlocks(streamState);
          dispatch(updateMessage({ conversationId, messageId: assistantMessageId, patch: { content: finalBlocks } }));
          dispatch(endStream());
          if ((streamState.currentRun?.totalToolCalls ?? 0) > 0) {
            const refreshed = await getConversation(conversationId) as Conversation;
            const dbMessage = refreshed.messages.find(m => m.id === assistantMessageId);
            if (dbMessage) {
              dispatch(updateMessage({ conversationId, messageId: assistantMessageId, patch: { content: dbMessage.content, model_id: dbMessage.model_id, usage: dbMessage.usage } }));
            }
          }
        },
        onError: (message, payload) => dispatch(setStreamError({ message, code: payload?.code, data: payload?.data })),
      }, controller.signal);
    } catch (error) {
      if (!controller.signal.aborted) {
        dispatch(setStreamError({ message: error instanceof Error ? error.message : '继续执行失败' }));
        dispatch(endStream());
      }
    } finally {
      abortControllerRef.current = null;
    }
  }, [dispatch, store]);

  return { continueAgentRun };
}
```

After the hook is green, compare the duplicated SSE event handling with `useSendMessage`. If both files share more than 80 lines of identical callback dispatching, extract only the shared event-dispatch body into `src/hooks/useAgentStreamEvents.ts` in the same task.

- [ ] **Step 5: 运行 hook/API tests**

Run:

```bash
cd /Users/sean/code/fusion/fusion-ui
npm test -- chat.test.ts useContinueAgentRun.test.ts
```

Expected: PASS.

---

## Task 7: 前端 CTA 接线

**Files:**
- Modify: `fusion-ui/src/components/chat/agent/RunBanner.tsx`
- Modify: `fusion-ui/src/components/chat/agent/AgentRunTimeline.tsx`
- Modify: `fusion-ui/src/components/chat/AssistantResponseStack.tsx`
- Modify: `fusion-ui/src/components/chat/AssistantMessage.tsx`
- Modify: `fusion-ui/src/components/chat/ChatMessage.tsx`
- Modify: `fusion-ui/src/components/chat/ChatMessageList.tsx`
- Modify: `fusion-ui/src/app/(app)/chat/[chatId]/page.tsx`
- Test: `fusion-ui/src/components/chat/agent/RunBanner.test.tsx`
- Test: `fusion-ui/src/components/chat/AssistantResponseStack.test.tsx`
- Test: `fusion-ui/src/app/(app)/chat/[chatId]/page.test.tsx`

- [ ] **Step 1: 写 RunBanner 测试**

Extend `fusion-ui/src/components/chat/agent/RunBanner.test.tsx`:

```tsx
it.each(['max_steps', 'max_tool_calls', 'timeout'] as const)(
  'limit_reached + %s 显示继续查按钮',
  reason => {
    const onContinue = vi.fn();
    render(<RunBanner run={run({ status: 'limit_reached', limitReachedReason: reason })} onContinue={onContinue} />);

    fireEvent.click(screen.getByRole('button', { name: '继续查' }));

    expect(onContinue).toHaveBeenCalledTimes(1);
  },
);
```

- [ ] **Step 2: 实现 RunBanner props**

In `RunBanner.tsx`, change props:

```ts
interface RunBannerProps {
  run: AgentRunState;
  onRetry?: () => void;
  onContinue?: () => void;
}
```

For `limit_reached`, render:

```tsx
{onContinue && (
  <button
    type="button"
    onClick={onContinue}
    className="shrink-0 inline-flex items-center gap-1 px-2 py-1 rounded border border-warn/30 text-xs text-warn hover:bg-warn/10 transition-colors duration-fast"
  >
    <RotateCw className="w-3 h-3" />
    继续查
  </button>
)}
```

Keep `failed` and `incomplete` using `onRetry`.

- [ ] **Step 3: 逐层传递 onContinueAgentRun**

Use this prop shape through the chat stack:

```ts
onContinueAgentRun?: (messageId: string, previousRunId?: string) => void;
```

At `AgentRunTimeline`, convert to:

```tsx
<RunBanner
  run={run}
  onRetry={onRetry}
  onContinue={onContinue ? () => onContinue(run.runId) : undefined}
/>
```

At `AssistantMessageFrame`, bind:

```ts
const handleContinue = useMemo(
  () => onContinueAgentRun ? (previousRunId?: string) => onContinueAgentRun(message.id, previousRunId) : undefined,
  [message.id, onContinueAgentRun],
);
```

Pass `handleContinue` into `AssistantResponseStack`.

- [ ] **Step 4: 接入 ChatPage**

In `fusion-ui/src/app/(app)/chat/[chatId]/page.tsx`, import and use:

```ts
const { continueAgentRun } = useContinueAgentRun();
```

Add handler:

```ts
const handleContinueAgentRun = useCallback((messageId: string, previousRunId?: string) => {
  if (!chatId || isStreaming) return;
  void continueAgentRun({
    conversationId: chatId,
    assistantMessageId: messageId,
    previousRunId,
  });
}, [chatId, continueAgentRun, isStreaming]);
```

Pass to `ChatMessageList`:

```tsx
onContinueAgentRun={handleContinueAgentRun}
```

- [ ] **Step 5: 运行 UI focused tests**

Run:

```bash
cd /Users/sean/code/fusion/fusion-ui
npm test -- RunBanner.test.tsx AssistantResponseStack.test.tsx \"page.test.tsx\"
```

Expected: PASS.

---

## Task 8: 集成验证、CI/CD 和真实 Chrome 回归

**Files:**
- Modify as produced by Tasks 1-7.

- [ ] **Step 1: 后端 focused suite**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest \
  test/services/agent/test_continuation.py \
  test/services/agent/test_session_cache.py \
  test/services/stream/test_agent_loop_request_prep.py \
  test/services/stream/test_agent_loop_lifecycle.py \
  test/test_chat_continue.py \
  test/test_stream_handler.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: 后端 full suite and lint**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest -q
/opt/homebrew/bin/ruff check .
/opt/homebrew/bin/ruff format --check app test
/opt/homebrew/bin/python3.11 scripts/check_architecture.py
/opt/homebrew/bin/python3.11 scripts/check_quality.py
git diff --check
```

Expected: pytest PASS; ruff PASS; architecture check PASS; quality check has no new stream redlines.

- [ ] **Step 3: 前端 focused suite**

Run:

```bash
cd /Users/sean/code/fusion/fusion-ui
npm test -- \
  chat.test.ts \
  streamSlice.test.ts \
  useContinueAgentRun.test.ts \
  RunBanner.test.tsx \
  AssistantResponseStack.test.tsx \
  page.test.tsx
```

Expected: PASS.

- [ ] **Step 4: 前端 full validation**

Run:

```bash
cd /Users/sean/code/fusion/fusion-ui
npm test
npm run lint
npm run build
git diff --check
```

Expected: all commands pass.

- [ ] **Step 5: 提交和 push**

Use one final commit per repo, with Chinese messages and Co-Authored-By. For `fusion-api`, include the already-created spec commit in branch history and create the implementation commit after code passes:

```bash
cd /Users/sean/code/fusion/fusion-api
git status --short
git add app alembic test docs/superpowers/plans/2026-06-28-agent-run-continuation.md
git commit -m "feat: 支持 agent run 继续执行" -m "Co-Authored-By: Codex <noreply@anthropic.com>"
git push origin master
```

For `fusion-ui`:

```bash
cd /Users/sean/code/fusion/fusion-ui
git status --short
git add src
git commit -m "feat: 接入 agent run 继续执行" -m "Co-Authored-By: Codex <noreply@anthropic.com>"
git push origin master
```

- [ ] **Step 6: 监控 CI/CD**

Run:

```bash
gh run list --repo HyxiaoGe/fusion-api --branch master --limit 5
gh run list --repo HyxiaoGe/fusion-ui --branch master --limit 5
```

Then inspect the run IDs for the pushed commits:

```bash
gh run view <api-run-id> --repo HyxiaoGe/fusion-api --log-failed
gh run view <ui-run-id> --repo HyxiaoGe/fusion-ui --log-failed
```

Expected: both workflows finish success. If either fails, inspect logs, fix, rerun focused tests, and push one correction commit in the affected repo.

- [ ] **Step 7: 远端 dev 健康验证**

After CI deploy succeeds, run read-only checks:

```bash
ssh dev 'docker inspect fusion-api --format "image={{.Config.Image}} status={{.State.Status}} started={{.State.StartedAt}}"'
ssh dev 'docker inspect fusion-ui --format "image={{.Config.Image}} status={{.State.Status}} started={{.State.StartedAt}}"'
ssh dev 'curl -fsS http://127.0.0.1:8002/health'
```

Expected: API and UI containers use the pushed image tags; API health returns `{"status":"healthy",...}`.

- [ ] **Step 8: 真实 Chrome 回归**

Use existing logged-in Chrome tab only. Do not open deprecated dev domains. Use `https://fusion.seanfield.org`.

Regression path:

1. In an existing official Fusion tab, create or use a conversation with a prompt likely to hit a low continuation budget only if the deployed config/test path can safely trigger it. If production config cannot safely force a limit, use an existing triggered `limit_reached` conversation from dev logs.
2. Confirm `limit_reached` banner shows `继续查`.
3. Click `继续查`.
4. Confirm no user message or new assistant message is inserted.
5. Confirm original assistant message content grows.
6. Confirm Chrome console has no error/warning entries.
7. Confirm dev logs show two different run IDs with the same `message_id`, and the continuation run reaches `finish_reason=stop` or a second `limit_reached`.

Log command:

```bash
ssh dev 'docker logs --since=20m fusion-api 2>&1 | grep -E "AGENT_ROUND_SUMMARY|run_limit_reached|ERROR|Traceback|Exception" | tail -160'
```

Expected: no traceback/error for the regression conversation.

---

## Self-Review

- Spec coverage: continuation API, same assistant message append, original-budget continuation, no fake user message, errors, stop behavior, tests, CI/CD and Chrome regression are covered by Tasks 1-8.
- Data gap handled: `agent_sessions.config` is added because previous `run_started.config` was not durably stored.
- Type consistency: backend request uses `previous_run_id`; frontend request uses `previousRunId` and maps to snake case at the API boundary.
- Scope kept narrow: failed-run retry, single-tool retry, budget picker, historical multi-run timeline, and tool marketplace work are excluded.
