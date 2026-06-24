# 会话级联网诊断 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让历史联网回答可回看本次用了哪些工具、耗时多少、哪些失败或降级，并按普通用户/管理员权限展示不同粒度。

**Architecture:** 后端先修正 `tool_call_logs.message_id` 关联，再新增只读 diagnostics 聚合 API，从 `agent_sessions`、`agent_steps`、`tool_call_logs` 按 assistant message 拼出诊断模型。前端不改变实时 `agent_event` timeline，只在回答依据侧栏懒加载 diagnostics，并展示普通用户摘要和管理员明细。

**Tech Stack:** FastAPI、SQLAlchemy、Pydantic、pytest、Next.js 15、React 19、Redux、Vitest、Testing Library。

---

## 文件结构

### 后端 `fusion-api`

- Modify: `app/services/tool_handlers/base.py`
  - 让 `BaseToolHandler.log()` 接收 `message_id` 并传给 `log_tool_call()`。
- Modify: `app/services/stream/tool_executor.py`
  - 保持现有 `message_id=message_id` 调用路径；只在必要时调整类型注解。
- Create: `app/schemas/network_diagnostics.py`
  - 定义 diagnostics 响应 schema。
- Create: `app/services/network_diagnostics_service.py`
  - 聚合 agent session、step、tool call log，输出脱敏/管理员模型。
- Modify: `app/api/deps.py`
  - 增加 `get_network_diagnostics_service()`。
- Modify: `app/api/chat.py`
  - 新增 `GET /conversations/{conversation_id}/messages/{message_id}/diagnostics`。
- Test: `test/test_tool_handlers.py`
  - 覆盖 `BaseToolHandler.log()` 透传 message_id。
- Test: `test/test_tool_executor.py`
  - 保留并确认 `execute_tools_parallel()` 传 message_id 给 handler.log。
- Test: `test/test_network_diagnostics.py`
  - 覆盖权限、空状态、聚合、脱敏、管理员明细。

### 前端 `fusion-ui`

- Create: `src/types/networkDiagnostics.ts`
  - 定义后端响应类型。
- Create: `src/lib/api/chatDiagnostics.ts`
  - 新增 diagnostics API client。
- Create: `src/lib/api/chatDiagnostics.test.ts`
  - 覆盖请求路径和错误传播。
- Create: `src/components/chat/networkDiagnosticsModel.ts`
  - 纯函数派生摘要、异常列表、管理员明细展示模型。
- Create: `src/components/chat/networkDiagnosticsModel.test.ts`
  - 覆盖文案和权限粒度。
- Create: `src/components/chat/NetworkDiagnosticsPanel.tsx`
  - 侧栏中的联网诊断分区。
- Modify: `src/components/chat/AnswerEvidenceSidebar.tsx`
  - 接收 diagnostics model/loading/error，并渲染 `NetworkDiagnosticsPanel`。
- Modify: `src/components/chat/AssistantMessage.tsx`
  - 侧栏打开时懒加载 diagnostics。
- Modify: `src/components/chat/AnswerEvidence.test.tsx`
  - 保证无来源但有诊断异常入口时仍可打开侧栏。
- Modify: `src/components/chat/AnswerEvidenceSidebar.test.tsx`
  - 覆盖联网诊断分区。
- Modify: `src/components/chat/AssistantMessage.test.tsx`
  - 覆盖打开侧栏触发 diagnostics 拉取。

---

## Task 1: 后端修复工具日志 message_id 关联

**Files:**
- Modify: `fusion-api/app/services/tool_handlers/base.py`
- Test: `fusion-api/test/test_tool_handlers.py`
- Verify: `fusion-api/test/test_tool_executor.py`

- [ ] **Step 1: 写失败测试**

在 `test/test_tool_handlers.py` 的 `ExecuteWithEmitterTests` 前新增测试类：

```python
class ToolHandlerLogTests(unittest.IsolatedAsyncioTestCase):
    async def test_log_passes_message_id_to_agent_logger(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from app.services.tool_handlers.base import BaseToolHandler, ToolResult

        class _Stub(BaseToolHandler):
            tool_name = "web_search"
            sse_event_prefix = "search"

            async def execute(self, args):
                return ToolResult(status="success", data={})

            def build_content_block(self, result, block_id, log_id):
                return MagicMock()

            def format_llm_context(self, result):
                return ""

        handler = _Stub()
        result = ToolResult(status="success", duration_ms=12, data={"result_count": 1})

        with patch(
            "app.services.tool_handlers.base.log_tool_call",
            new_callable=AsyncMock,
        ) as mock_log_tool_call:
            await handler.log(
                log_id="log-1",
                conversation_id="conv-1",
                user_id="user-1",
                model_id="model-1",
                provider="provider-1",
                result=result,
                input_params={"query": "redis"},
                trace_id="trace-1",
                step_number=1,
                message_id="assistant-1",
            )

        mock_log_tool_call.assert_awaited_once()
        assert mock_log_tool_call.await_args.kwargs["message_id"] == "assistant-1"
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest test/test_tool_handlers.py::ToolHandlerLogTests::test_log_passes_message_id_to_agent_logger -q
```

Expected: FAIL，原因是 `BaseToolHandler.log()` 不接受 `message_id` 参数。

- [ ] **Step 3: 实现最小修复**

修改 `app/services/tool_handlers/base.py`：

```python
    async def log(
        self,
        log_id: str,
        conversation_id: str,
        user_id: str,
        model_id: str,
        provider: str,
        result: ToolResult,
        input_params: dict,
        trace_id: str = None,
        step_number: int = None,
        message_id: str | None = None,
    ) -> None:
        """异步记录 ToolCallLog"""
        task = asyncio.create_task(
            log_tool_call(
                log_id=log_id,
                conversation_id=conversation_id,
                message_id=message_id,
                user_id=user_id,
                tool_name=self.tool_name,
                status=result.status,
                duration_ms=result.duration_ms,
                model_id=model_id,
                provider=provider,
                input_params=input_params,
                output_data=_serialize_for_json(result.data),
                error_message=result.error_message,
                trace_id=trace_id,
                step_number=step_number,
            )
        )
        task.add_done_callback(_task_done_callback)
```

- [ ] **Step 4: 运行目标测试确认通过**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest test/test_tool_handlers.py::ToolHandlerLogTests::test_log_passes_message_id_to_agent_logger test/test_tool_executor.py::ToolExecutorMessageIdTests::test_execute_tools_parallel_passes_message_id_to_handler_log -q
```

Expected: `2 passed`。

- [ ] **Step 5: 提交**

```bash
cd /Users/sean/code/fusion/fusion-api
git add app/services/tool_handlers/base.py test/test_tool_handlers.py test/test_tool_executor.py
git commit -m "fix: 关联工具日志消息ID" -m "Co-Authored-By: Codex <noreply@anthropic.com>"
```

---

## Task 2: 后端新增 diagnostics schema 和聚合服务

**Files:**
- Create: `fusion-api/app/schemas/network_diagnostics.py`
- Create: `fusion-api/app/services/network_diagnostics_service.py`
- Test: `fusion-api/test/test_network_diagnostics.py`

- [ ] **Step 1: 写 schema 文件**

Create `app/schemas/network_diagnostics.py`:

```python
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


ToolStatus = Literal["success", "failed", "degraded", "interrupted"]


class NetworkDiagnosticsSummary(BaseModel):
    total_duration_ms: int | None = None
    total_steps: int = 0
    total_tool_calls: int = 0
    search_calls: int = 0
    url_read_calls: int = 0
    success_count: int = 0
    failed_count: int = 0
    degraded_count: int = 0
    interrupted_count: int = 0
    limit_reason: str | None = None
    run_status: str | None = None


class NetworkDiagnosticsToolItem(BaseModel):
    tool_call_log_id: str
    tool_name: str
    status: ToolStatus
    duration_ms: int | None = None
    target: str = ""
    result_count: int | None = None
    reason: str | None = None
    started_at: datetime | None = None
    admin: dict[str, Any] | None = None


class NetworkDiagnosticsResponse(BaseModel):
    conversation_id: str
    message_id: str
    run_id: str | None = None
    visibility: Literal["user", "admin"] = "user"
    summary: NetworkDiagnosticsSummary = Field(default_factory=NetworkDiagnosticsSummary)
    tools: list[NetworkDiagnosticsToolItem] = Field(default_factory=list)
    is_empty: bool = False
```

- [ ] **Step 2: 写服务测试**

Create `test/test_network_diagnostics.py` with these core tests first:

```python
import os
import unittest

os.environ.setdefault("DATABASE_URL", "sqlite:///./fusion-test.db")

from app.db.models import AgentSession, Message, ToolCallLog  # noqa: E402
from app.services.network_diagnostics_service import NetworkDiagnosticsService  # noqa: E402


class NetworkDiagnosticsServiceTests(unittest.TestCase):
    def test_empty_response_for_message_without_agent_rows(self):
        db = unittest.mock.MagicMock()
        db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        service = NetworkDiagnosticsService(db)

        result = service.build_for_message(
            conversation_id="conv-1",
            message_id="assistant-1",
            is_admin=False,
        )

        self.assertTrue(result.is_empty)
        self.assertEqual(result.summary.total_tool_calls, 0)
        self.assertEqual(result.tools, [])

    def test_tool_item_is_sanitized_for_user(self):
        service = NetworkDiagnosticsService(unittest.mock.MagicMock())
        log = ToolCallLog(
            id="log-1",
            conversation_id="conv-1",
            message_id="assistant-1",
            user_id="user-1",
            tool_name="web_search",
            status="success",
            duration_ms=123,
            model_id="model",
            provider="provider",
            input_params={"query": "redis stream"},
            output_data={"result_count": 5, "secret": "hidden"},
            trace_id="trace-1",
            step_number=2,
        )

        item = service._tool_item_from_log(log, is_admin=False)

        self.assertEqual(item.target, "redis stream")
        self.assertEqual(item.result_count, 5)
        self.assertIsNone(item.admin)

    def test_tool_item_includes_admin_fields_for_admin(self):
        service = NetworkDiagnosticsService(unittest.mock.MagicMock())
        log = ToolCallLog(
            id="log-1",
            conversation_id="conv-1",
            message_id="assistant-1",
            user_id="user-1",
            tool_name="url_read",
            status="failed",
            error_message="timeout",
            duration_ms=5000,
            model_id="model",
            provider="provider",
            input_params={"url": "https://example.com/a"},
            output_data={"content": "must not leak"},
            trace_id="trace-1",
            step_number=1,
        )

        item = service._tool_item_from_log(log, is_admin=True)

        self.assertEqual(item.target, "https://example.com/a")
        self.assertEqual(item.reason, "timeout")
        self.assertEqual(item.admin["trace_id"], "trace-1")
        self.assertEqual(item.admin["input_params"], {"url": "https://example.com/a"})
        self.assertNotIn("output_data", item.admin)
```

在文件顶部补：

```python
import unittest.mock
```

- [ ] **Step 3: 运行测试确认失败**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest test/test_network_diagnostics.py -q
```

Expected: FAIL，原因是 `app.services.network_diagnostics_service` 不存在。

- [ ] **Step 4: 实现服务**

Create `app/services/network_diagnostics_service.py`:

```python
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db.models import AgentSession, ToolCallLog
from app.schemas.network_diagnostics import (
    NetworkDiagnosticsResponse,
    NetworkDiagnosticsSummary,
    NetworkDiagnosticsToolItem,
)


class NetworkDiagnosticsService:
    def __init__(self, db: Session):
        self.db = db

    def build_for_message(
        self,
        *,
        conversation_id: str,
        message_id: str,
        is_admin: bool,
    ) -> NetworkDiagnosticsResponse:
        session = (
            self.db.query(AgentSession)
            .filter(
                AgentSession.conversation_id == conversation_id,
                AgentSession.message_id == message_id,
            )
            .order_by(AgentSession.created_at.desc())
            .first()
        )
        logs = (
            self.db.query(ToolCallLog)
            .filter(
                ToolCallLog.conversation_id == conversation_id,
                ToolCallLog.message_id == message_id,
            )
            .order_by(ToolCallLog.created_at.asc())
            .all()
        )

        tools = [self._tool_item_from_log(log, is_admin=is_admin) for log in logs]
        is_empty = session is None and not tools
        return NetworkDiagnosticsResponse(
            conversation_id=conversation_id,
            message_id=message_id,
            run_id=session.id if session else None,
            visibility="admin" if is_admin else "user",
            summary=self._build_summary(session, tools),
            tools=tools,
            is_empty=is_empty,
        )

    def _build_summary(
        self,
        session: AgentSession | None,
        tools: list[NetworkDiagnosticsToolItem],
    ) -> NetworkDiagnosticsSummary:
        return NetworkDiagnosticsSummary(
            total_duration_ms=session.total_duration_ms if session else None,
            total_steps=session.total_steps if session else 0,
            total_tool_calls=len(tools),
            search_calls=sum(1 for item in tools if item.tool_name == "web_search"),
            url_read_calls=sum(1 for item in tools if item.tool_name == "url_read"),
            success_count=sum(1 for item in tools if item.status == "success"),
            failed_count=sum(1 for item in tools if item.status == "failed"),
            degraded_count=sum(1 for item in tools if item.status == "degraded"),
            interrupted_count=sum(1 for item in tools if item.status == "interrupted"),
            limit_reason=session.limit_reason if session else None,
            run_status=session.status if session else None,
        )

    def _tool_item_from_log(
        self,
        log: ToolCallLog,
        *,
        is_admin: bool,
    ) -> NetworkDiagnosticsToolItem:
        input_params = log.input_params or {}
        output_data = log.output_data or {}
        admin: dict[str, Any] | None = None
        if is_admin:
            admin = {
                "trace_id": log.trace_id,
                "step_number": log.step_number,
                "input_params": self._sanitize_input_params(input_params),
                "error_message": log.error_message,
                "created_at": log.created_at.isoformat() if log.created_at else None,
            }

        return NetworkDiagnosticsToolItem(
            tool_call_log_id=log.id,
            tool_name=log.tool_name,
            status=self._normalize_status(log.status),
            duration_ms=log.duration_ms,
            target=self._derive_target(log.tool_name, input_params),
            result_count=self._derive_result_count(log.tool_name, output_data),
            reason=self._derive_reason(log.status, log.error_message),
            started_at=log.created_at,
            admin=admin,
        )

    def _normalize_status(self, status: str) -> str:
        if status in ("success", "failed", "degraded", "interrupted"):
            return status
        return "failed"

    def _derive_target(self, tool_name: str, input_params: dict[str, Any]) -> str:
        if tool_name == "web_search":
            return str(input_params.get("query") or "").strip()
        if tool_name == "url_read":
            return str(input_params.get("url") or "").strip()
        return tool_name

    def _derive_result_count(self, tool_name: str, output_data: dict[str, Any]) -> int | None:
        if "result_count" in output_data and isinstance(output_data["result_count"], int):
            return output_data["result_count"]
        sources = output_data.get("sources")
        if tool_name == "web_search" and isinstance(sources, list):
            return len(sources)
        return None

    def _derive_reason(self, status: str, error_message: str | None) -> str | None:
        if error_message and error_message.strip():
            return error_message.strip()
        if status == "degraded":
            return "部分内容不可用，已降级处理"
        if status == "failed":
            return "未取得可用内容"
        if status == "interrupted":
            return "工具调用已中断"
        return None

    def _sanitize_input_params(self, input_params: dict[str, Any]) -> dict[str, Any]:
        allowed_keys = {"query", "url"}
        return {key: value for key, value in input_params.items() if key in allowed_keys}
```

- [ ] **Step 5: 运行服务测试确认通过**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest test/test_network_diagnostics.py -q
```

Expected: service tests pass.

- [ ] **Step 6: 提交**

```bash
cd /Users/sean/code/fusion/fusion-api
git add app/schemas/network_diagnostics.py app/services/network_diagnostics_service.py test/test_network_diagnostics.py
git commit -m "feat: 聚合联网诊断数据" -m "Co-Authored-By: Codex <noreply@anthropic.com>"
```

---

## Task 3: 后端新增 diagnostics API

**Files:**
- Modify: `fusion-api/app/api/deps.py`
- Modify: `fusion-api/app/api/chat.py`
- Test: `fusion-api/test/test_network_diagnostics_api.py`

- [ ] **Step 1: 写 API 测试**

Create `test/test_network_diagnostics_api.py` with focused endpoint tests. Use existing `TestClient` fixtures from nearby chat API tests; if no shared fixture fits, create a local minimal setup with dependency overrides.

Core assertions:

```python
def test_diagnostics_rejects_user_message(client, db, auth_user):
    response = client.get("/api/chat/conversations/conv-1/messages/user-1/diagnostics")
    assert response.status_code == 404


def test_diagnostics_returns_empty_for_assistant_without_logs(client, db, auth_user):
    response = client.get("/api/chat/conversations/conv-1/messages/assistant-1/diagnostics")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["is_empty"] is True
    assert data["tools"] == []


def test_diagnostics_admin_gets_admin_field(client, db, admin_user):
    response = client.get("/api/chat/conversations/conv-1/messages/assistant-1/diagnostics")
    assert response.status_code == 200
    tool = response.json()["data"]["tools"][0]
    assert "admin" in tool
    assert tool["admin"]["trace_id"] == "trace-1"
```

- [ ] **Step 2: 运行 API 测试确认失败**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest test/test_network_diagnostics_api.py -q
```

Expected: FAIL，原因是路由不存在。

- [ ] **Step 3: 增加依赖工厂**

Modify `app/api/deps.py`:

```python
from app.services.network_diagnostics_service import NetworkDiagnosticsService


def get_network_diagnostics_service(db: Session = Depends(get_db)) -> NetworkDiagnosticsService:
    return NetworkDiagnosticsService(db)
```

- [ ] **Step 4: 增加路由**

Modify `app/api/chat.py` imports:

```python
from app.api.deps import get_chat_service, get_current_user, get_network_diagnostics_service
from app.services.network_diagnostics_service import NetworkDiagnosticsService
```

Add route before `update_message`:

```python
@router.get("/conversations/{conversation_id}/messages/{message_id}/diagnostics")
def get_message_network_diagnostics(
    conversation_id: str,
    message_id: str,
    request: Request,
    chat_service: ChatService = Depends(get_chat_service),
    diagnostics_service: NetworkDiagnosticsService = Depends(get_network_diagnostics_service),
    current_user: User = Depends(get_current_user),
):
    """获取单条 assistant 消息的联网诊断。"""
    conversation = chat_service.get_conversation(conversation_id, user_id=current_user.id)
    if not conversation:
        raise ApiException.not_found("会话不存在或无权访问")
    message = next((msg for msg in conversation.messages if msg.id == message_id), None)
    if message is None or message.role != "assistant":
        raise ApiException.not_found("消息不存在或无权访问")

    data = diagnostics_service.build_for_message(
        conversation_id=conversation_id,
        message_id=message_id,
        is_admin=bool(getattr(current_user, "is_superuser", False)),
    )
    return success(data=data, request_id=request.state.request_id)
```

- [ ] **Step 5: 运行 API + service 测试**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest test/test_network_diagnostics.py test/test_network_diagnostics_api.py -q
```

Expected: tests pass.

- [ ] **Step 6: 运行后端相关回归**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest test/test_tool_handlers.py test/test_tool_executor.py test/test_network_diagnostics.py test/test_network_diagnostics_api.py -q
.venv/bin/ruff check app/services/tool_handlers/base.py app/services/network_diagnostics_service.py app/schemas/network_diagnostics.py app/api/chat.py app/api/deps.py test/test_network_diagnostics.py test/test_network_diagnostics_api.py
```

Expected: pytest and ruff pass.

- [ ] **Step 7: 提交**

```bash
cd /Users/sean/code/fusion/fusion-api
git add app/api/deps.py app/api/chat.py test/test_network_diagnostics_api.py
git commit -m "feat: 提供联网诊断接口" -m "Co-Authored-By: Codex <noreply@anthropic.com>"
```

---

## Task 4: 前端新增 diagnostics 类型、API client 和展示模型

**Files:**
- Create: `fusion-ui/src/types/networkDiagnostics.ts`
- Create: `fusion-ui/src/lib/api/chatDiagnostics.ts`
- Create: `fusion-ui/src/lib/api/chatDiagnostics.test.ts`
- Create: `fusion-ui/src/components/chat/networkDiagnosticsModel.ts`
- Create: `fusion-ui/src/components/chat/networkDiagnosticsModel.test.ts`

- [ ] **Step 1: 新增类型**

Create `src/types/networkDiagnostics.ts`:

```ts
export type NetworkDiagnosticsStatus = 'success' | 'failed' | 'degraded' | 'interrupted';

export interface NetworkDiagnosticsSummary {
  total_duration_ms: number | null;
  total_steps: number;
  total_tool_calls: number;
  search_calls: number;
  url_read_calls: number;
  success_count: number;
  failed_count: number;
  degraded_count: number;
  interrupted_count: number;
  limit_reason?: string | null;
  run_status?: string | null;
}

export interface NetworkDiagnosticsToolItem {
  tool_call_log_id: string;
  tool_name: string;
  status: NetworkDiagnosticsStatus;
  duration_ms: number | null;
  target: string;
  result_count?: number | null;
  reason?: string | null;
  started_at?: string | null;
  admin?: Record<string, unknown> | null;
}

export interface NetworkDiagnosticsResponse {
  conversation_id: string;
  message_id: string;
  run_id: string | null;
  visibility: 'user' | 'admin';
  summary: NetworkDiagnosticsSummary;
  tools: NetworkDiagnosticsToolItem[];
  is_empty: boolean;
}
```

- [ ] **Step 2: 写 API client 测试**

Create `src/lib/api/chatDiagnostics.test.ts`:

```ts
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { getMessageNetworkDiagnostics } from './chatDiagnostics';
import { apiRequest } from './fetchWithAuth';

vi.mock('./fetchWithAuth', () => ({
  apiRequest: vi.fn(),
}));

describe('getMessageNetworkDiagnostics', () => {
  beforeEach(() => {
    vi.mocked(apiRequest).mockReset();
  });

  it('请求单条消息 diagnostics 路径', async () => {
    vi.mocked(apiRequest).mockResolvedValueOnce({ is_empty: true });

    await getMessageNetworkDiagnostics('conv-1', 'msg-1');

    expect(apiRequest).toHaveBeenCalledWith(
      expect.stringContaining('/api/chat/conversations/conv-1/messages/msg-1/diagnostics'),
    );
  });
});
```

- [ ] **Step 3: 实现 API client**

Create `src/lib/api/chatDiagnostics.ts`:

```ts
import { API_CONFIG } from '../config';
import { apiRequest } from './fetchWithAuth';
import type { NetworkDiagnosticsResponse } from '@/types/networkDiagnostics';

export function getMessageNetworkDiagnostics(
  conversationId: string,
  messageId: string,
): Promise<NetworkDiagnosticsResponse> {
  const path = `/api/chat/conversations/${encodeURIComponent(conversationId)}/messages/${encodeURIComponent(messageId)}/diagnostics`;
  return apiRequest<NetworkDiagnosticsResponse>(`${API_CONFIG.BASE_URL}${path}`);
}
```

- [ ] **Step 4: 写展示模型测试**

Create `src/components/chat/networkDiagnosticsModel.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { deriveNetworkDiagnosticsModel } from './networkDiagnosticsModel';
import type { NetworkDiagnosticsResponse } from '@/types/networkDiagnostics';

const base: NetworkDiagnosticsResponse = {
  conversation_id: 'conv-1',
  message_id: 'msg-1',
  run_id: 'run-1',
  visibility: 'user',
  is_empty: false,
  summary: {
    total_duration_ms: 4200,
    total_steps: 2,
    total_tool_calls: 3,
    search_calls: 2,
    url_read_calls: 1,
    success_count: 2,
    failed_count: 0,
    degraded_count: 1,
    interrupted_count: 0,
  },
  tools: [
    {
      tool_call_log_id: 'log-1',
      tool_name: 'web_search',
      status: 'success',
      duration_ms: 1200,
      target: 'G7 AI',
      result_count: 5,
    },
    {
      tool_call_log_id: 'log-2',
      tool_name: 'url_read',
      status: 'degraded',
      duration_ms: 3000,
      target: 'https://example.com',
      reason: 'reader-service 暂时未返回内容',
    },
  ],
};

describe('deriveNetworkDiagnosticsModel', () => {
  it('生成用户摘要和异常列表', () => {
    const model = deriveNetworkDiagnosticsModel(base);
    expect(model?.summaryText).toBe('联网诊断 · 搜索 2 次 · 读取 1 个网页 · 用时 4.2s');
    expect(model?.issueItems).toHaveLength(1);
    expect(model?.issueItems[0].reason).toContain('reader-service');
  });

  it('空 diagnostics 不渲染', () => {
    expect(deriveNetworkDiagnosticsModel({ ...base, is_empty: true, tools: [] })).toBeNull();
  });

  it('管理员可展开明细', () => {
    const model = deriveNetworkDiagnosticsModel({
      ...base,
      visibility: 'admin',
      tools: [{ ...base.tools[0], admin: { trace_id: 'trace-1' } }],
    });
    expect(model?.canShowAdminDetails).toBe(true);
  });
});
```

- [ ] **Step 5: 实现展示模型**

Create `src/components/chat/networkDiagnosticsModel.ts`:

```ts
import type {
  NetworkDiagnosticsResponse,
  NetworkDiagnosticsToolItem,
} from '@/types/networkDiagnostics';

export interface NetworkDiagnosticsIssueItem {
  id: string;
  toolName: string;
  title: string;
  status: NetworkDiagnosticsToolItem['status'];
  reason: string;
}

export interface NetworkDiagnosticsModel {
  summaryText: string;
  issueItems: NetworkDiagnosticsIssueItem[];
  tools: NetworkDiagnosticsToolItem[];
  canShowAdminDetails: boolean;
}

export function deriveNetworkDiagnosticsModel(
  diagnostics: NetworkDiagnosticsResponse | null,
): NetworkDiagnosticsModel | null {
  if (!diagnostics || diagnostics.is_empty || diagnostics.summary.total_tool_calls === 0) return null;
  const issueItems = diagnostics.tools
    .filter(item => item.status === 'failed' || item.status === 'degraded' || item.status === 'interrupted')
    .map(item => ({
      id: item.tool_call_log_id,
      toolName: item.tool_name,
      title: item.target || getToolLabel(item.tool_name),
      status: item.status,
      reason: item.reason || getFallbackReason(item.status),
    }));

  return {
    summaryText: buildSummaryText(diagnostics),
    issueItems,
    tools: diagnostics.tools,
    canShowAdminDetails: diagnostics.visibility === 'admin'
      && diagnostics.tools.some(item => item.admin),
  };
}

function buildSummaryText(diagnostics: NetworkDiagnosticsResponse): string {
  const parts = ['联网诊断'];
  if (diagnostics.summary.search_calls > 0) parts.push(`搜索 ${diagnostics.summary.search_calls} 次`);
  if (diagnostics.summary.url_read_calls > 0) parts.push(`读取 ${diagnostics.summary.url_read_calls} 个网页`);
  if (diagnostics.summary.total_duration_ms !== null) {
    parts.push(`用时 ${formatDuration(diagnostics.summary.total_duration_ms)}`);
  }
  return parts.join(' · ');
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function getFallbackReason(status: NetworkDiagnosticsToolItem['status']): string {
  if (status === 'degraded') return '部分内容不可用，已降级处理';
  if (status === 'interrupted') return '工具调用已中断';
  return '未取得可用内容';
}

function getToolLabel(toolName: string): string {
  if (toolName === 'web_search') return '搜索';
  if (toolName === 'url_read') return '网页读取';
  return toolName;
}
```

- [ ] **Step 6: 运行前端模型/API 测试**

Run:

```bash
cd /Users/sean/code/fusion/fusion-ui
npm test -- src/lib/api/chatDiagnostics.test.ts src/components/chat/networkDiagnosticsModel.test.ts
```

Expected: tests pass.

- [ ] **Step 7: 提交**

```bash
cd /Users/sean/code/fusion/fusion-ui
git add src/types/networkDiagnostics.ts src/lib/api/chatDiagnostics.ts src/lib/api/chatDiagnostics.test.ts src/components/chat/networkDiagnosticsModel.ts src/components/chat/networkDiagnosticsModel.test.ts
git commit -m "feat: 增加联网诊断前端模型" -m "Co-Authored-By: Codex <noreply@anthropic.com>"
```

---

## Task 5: 前端接入回答依据侧栏

**Files:**
- Create: `fusion-ui/src/components/chat/NetworkDiagnosticsPanel.tsx`
- Modify: `fusion-ui/src/components/chat/AnswerEvidenceSidebar.tsx`
- Modify: `fusion-ui/src/components/chat/AssistantMessage.tsx`
- Modify: `fusion-ui/src/components/chat/AnswerEvidenceSidebar.test.tsx`
- Modify: `fusion-ui/src/components/chat/AssistantMessage.test.tsx`

- [ ] **Step 1: 新增 panel 组件**

Create `src/components/chat/NetworkDiagnosticsPanel.tsx`:

```tsx
'use client';

import { ChevronDown, ChevronRight, Clock, Wrench } from 'lucide-react';
import { useState } from 'react';
import { cn } from '@/lib/utils';
import type { NetworkDiagnosticsModel } from './networkDiagnosticsModel';

interface NetworkDiagnosticsPanelProps {
  model: NetworkDiagnosticsModel | null;
  isLoading?: boolean;
  error?: string | null;
}

export default function NetworkDiagnosticsPanel({
  model,
  isLoading = false,
  error = null,
}: NetworkDiagnosticsPanelProps) {
  const [expanded, setExpanded] = useState(false);

  if (isLoading) {
    return <section className="mt-5 text-xs text-muted-foreground">正在读取联网诊断...</section>;
  }
  if (error) {
    return <section className="mt-5 text-xs text-muted-foreground">联网诊断暂不可用</section>;
  }
  if (!model) return null;

  return (
    <section className="mt-5" data-testid="network-diagnostics-panel">
      <h4 className="mb-2 flex items-center gap-1.5 text-xs font-medium text-foreground">
        <Wrench className="h-3.5 w-3.5" aria-hidden="true" />
        联网诊断
      </h4>
      <div className="rounded-md border border-border/40 bg-muted/10 px-3 py-2">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Clock className="h-3.5 w-3.5" aria-hidden="true" />
          <span>{model.summaryText}</span>
        </div>
        {model.issueItems.length > 0 ? (
          <div className="mt-2 space-y-1">
            {model.issueItems.map(item => (
              <div key={item.id} className="text-xs text-muted-foreground">
                <span className={cn(
                  item.status === 'failed' ? 'text-danger'
                    : item.status === 'degraded' ? 'text-warn'
                      : 'text-muted-foreground',
                )}>
                  {item.status === 'failed' ? '失败' : item.status === 'degraded' ? '降级' : '中断'}
                </span>
                <span> · {item.title}：{item.reason}</span>
              </div>
            ))}
          </div>
        ) : null}
        {model.canShowAdminDetails ? (
          <button
            type="button"
            className="mt-2 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
            onClick={() => setExpanded(value => !value)}
          >
            {expanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
            管理员明细
          </button>
        ) : null}
        {expanded ? (
          <div className="mt-2 space-y-1 border-t border-border/40 pt-2">
            {model.tools.map(tool => (
              <div key={tool.tool_call_log_id} className="text-xs text-muted-foreground">
                {tool.tool_name} · {tool.status} · {tool.duration_ms ?? '-'}ms · {tool.target || '-'}
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </section>
  );
}
```

- [ ] **Step 2: 修改 AnswerEvidenceSidebar props**

Modify `AnswerEvidenceSidebar.tsx`:

```tsx
import NetworkDiagnosticsPanel from './NetworkDiagnosticsPanel';
import type { NetworkDiagnosticsModel } from './networkDiagnosticsModel';

interface AnswerEvidenceSidebarProps {
  model: AnswerEvidenceSidebarModel | null;
  diagnostics?: NetworkDiagnosticsModel | null;
  diagnosticsLoading?: boolean;
  diagnosticsError?: string | null;
  isOpen: boolean;
  onClose: () => void;
  highlightIndex?: number;
  highlightTick?: number;
}
```

Render after issue section:

```tsx
<NetworkDiagnosticsPanel
  model={diagnostics ?? null}
  isLoading={diagnosticsLoading}
  error={diagnosticsError}
/>
```

- [ ] **Step 3: 在 AssistantMessage 懒加载 diagnostics**

Modify `AssistantMessage.tsx` imports:

```tsx
import { getMessageNetworkDiagnostics } from '@/lib/api/chatDiagnostics';
import { deriveNetworkDiagnosticsModel } from './networkDiagnosticsModel';
import type { NetworkDiagnosticsResponse } from '@/types/networkDiagnostics';
```

Add state inside `AssistantMessageFrame`:

```tsx
const [networkDiagnostics, setNetworkDiagnostics] = useState<NetworkDiagnosticsResponse | null>(null);
const [networkDiagnosticsLoading, setNetworkDiagnosticsLoading] = useState(false);
const [networkDiagnosticsError, setNetworkDiagnosticsError] = useState<string | null>(null);
```

Add derived model:

```tsx
const networkDiagnosticsModel = useMemo(
  () => deriveNetworkDiagnosticsModel(networkDiagnostics),
  [networkDiagnostics],
);
```

Add lazy load effect:

```tsx
useEffect(() => {
  if (!answerEvidenceSidebarOpen || !activeChatId || networkDiagnostics || networkDiagnosticsLoading) return;
  let cancelled = false;
  setNetworkDiagnosticsLoading(true);
  setNetworkDiagnosticsError(null);
  getMessageNetworkDiagnostics(activeChatId, message.id)
    .then(data => {
      if (!cancelled) setNetworkDiagnostics(data);
    })
    .catch(error => {
      if (!cancelled) setNetworkDiagnosticsError(error instanceof Error ? error.message : '联网诊断暂不可用');
    })
    .finally(() => {
      if (!cancelled) setNetworkDiagnosticsLoading(false);
    });
  return () => {
    cancelled = true;
  };
}, [
  activeChatId,
  answerEvidenceSidebarOpen,
  message.id,
  networkDiagnostics,
  networkDiagnosticsLoading,
]);
```

Pass props to sidebar:

```tsx
<AnswerEvidenceSidebar
  model={answerEvidenceSidebar}
  diagnostics={networkDiagnosticsModel}
  diagnosticsLoading={networkDiagnosticsLoading}
  diagnosticsError={networkDiagnosticsError}
  isOpen={answerEvidenceSidebarOpen}
  onClose={handleSourcesClose}
  highlightIndex={citationHighlight.index}
  highlightTick={citationHighlight.tick}
/>
```

- [ ] **Step 4: 更新组件测试**

In `AnswerEvidenceSidebar.test.tsx`, add:

```tsx
it('渲染联网诊断分区', () => {
  render(
    <AnswerEvidenceSidebar
      model={model}
      diagnostics={{
        summaryText: '联网诊断 · 搜索 1 次 · 用时 1.2s',
        issueItems: [],
        tools: [],
        canShowAdminDetails: false,
      }}
      isOpen={true}
      onClose={vi.fn()}
    />,
  );

  expect(screen.getByText('联网诊断')).toBeInTheDocument();
  expect(screen.getByText('联网诊断 · 搜索 1 次 · 用时 1.2s')).toBeInTheDocument();
});
```

In `AssistantMessage.test.tsx`, mock `getMessageNetworkDiagnostics` and assert it is called after opening sources.

- [ ] **Step 5: 运行前端聊天测试**

Run:

```bash
cd /Users/sean/code/fusion/fusion-ui
npm test -- src/components/chat/networkDiagnosticsModel.test.ts src/lib/api/chatDiagnostics.test.ts src/components/chat/AnswerEvidenceSidebar.test.tsx src/components/chat/AssistantMessage.test.tsx src/components/chat/ChatMessage.test.tsx
```

Expected: tests pass.

- [ ] **Step 6: 提交**

```bash
cd /Users/sean/code/fusion/fusion-ui
git add src/components/chat/NetworkDiagnosticsPanel.tsx src/components/chat/AnswerEvidenceSidebar.tsx src/components/chat/AssistantMessage.tsx src/components/chat/AnswerEvidenceSidebar.test.tsx src/components/chat/AssistantMessage.test.tsx
git commit -m "feat: 接入联网诊断侧栏" -m "Co-Authored-By: Codex <noreply@anthropic.com>"
```

---

## Task 6: 全量验证、push 和 CI 跟进

**Files:**
- No source edits unless verification exposes a real issue.

- [ ] **Step 1: 后端验证**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest test/test_tool_handlers.py test/test_tool_executor.py test/test_network_diagnostics.py test/test_network_diagnostics_api.py -q
.venv/bin/ruff check .
```

Expected: tests and ruff pass.

- [ ] **Step 2: 前端验证**

Run:

```bash
cd /Users/sean/code/fusion/fusion-ui
npm test -- src/lib/api/chatDiagnostics.test.ts src/components/chat/networkDiagnosticsModel.test.ts src/components/chat/AnswerEvidenceSidebar.test.tsx src/components/chat/AssistantMessage.test.tsx src/components/chat/ChatMessage.test.tsx
npm run lint
```

Expected: tests and lint pass.

- [ ] **Step 3: 推送两个仓库**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
git push origin master

cd /Users/sean/code/fusion/fusion-ui
git push origin master
```

- [ ] **Step 4: 监听 CI**

Run:

```bash
gh run list --repo HyxiaoGe/fusion-api --branch master --limit 3
gh run list --repo HyxiaoGe/fusion-ui --branch master --limit 3
```

Follow the newest runs until success or a concrete failure is identified.

- [ ] **Step 5: 完成审计**

Verify these requirements against current evidence:

- `ToolCallLog.message_id` is populated for new web tool calls.
- Diagnostics API returns empty model for non-network old messages.
- Diagnostics API returns user summary without raw params for normal users.
- Diagnostics API returns admin field for superuser.
- Answer evidence sidebar shows network diagnostics after lazy load.
- Existing realtime Agent timeline still renders from `agent_event`.
- No local Fusion dev server was started.

---

## Plan Self-Review

- Spec coverage: Task 1 covers message_id association. Tasks 2-3 cover backend schema/service/API, permissions, empty state, user/admin visibility. Tasks 4-5 cover frontend types, API client, model, lazy loading, side panel display. Task 6 covers validation, push, and CI.
- Placeholder scan: no unresolved placeholder wording; every task has concrete files, commands, and expected output.
- Type consistency: backend uses `NetworkDiagnosticsResponse`, `NetworkDiagnosticsSummary`, `NetworkDiagnosticsToolItem`; frontend mirrors `NetworkDiagnosticsResponse`, `NetworkDiagnosticsSummary`, `NetworkDiagnosticsToolItem` and derives `NetworkDiagnosticsModel`.
