# Agent Loop Contract Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补齐 agent-loop contract test suite，把事件顺序、DB partial persist、Redis Stream 终态、agent session/step 状态变成明确断言。

**Architecture:** 新增一个聚合型测试文件 `test/services/stream/test_agent_loop_contract.py`，复用现有 `StreamHandler.generate_to_redis()` 入口，以 fake LLM round、fake tool executor、fake Redis、captured session cache 写入构造真实编排契约。生产代码默认不改；只有 contract 测试暴露真实缺陷时才修实现。

**Tech Stack:** Python 3.11、unittest/pytest、fakeredis、FastAPI 后端现有 stream/agent 模块。

---

## File Structure

- Create: `test/services/stream/test_agent_loop_contract.py`
  - 提供 `AgentLoopContractTests`。
  - 覆盖跨模块契约：agent event sequence、partial persist order、Redis Stream done/error terminal、agent session/step terminal state。
- Create: `test/services/stream/__init__.py`
  - 让 `python -m unittest discover -s test -t .` 能递归发现 `test/services/stream` 下的 `unittest.TestCase` 测试。
- Modify only if needed: `app/services/stream/*.py`
  - 仅当新增契约测试发现真实回归时修改。

## Task 1: Contract Harness

**Files:**
- Create: `test/services/stream/test_agent_loop_contract.py`

- [x] **Step 1: Write a failing smoke contract harness**

Create a test class that invokes `StreamHandler.generate_to_redis()` with:

```python
class AgentLoopContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_contract_harness_records_events_persist_and_session_state(self):
        result = await self._run_agent_contract(
            rounds=[("", "Final answer", [], "stop", None)],
        )

        self.assertEqual(result.event_types, ["run_started", "step_started", "step_completed", "run_completed"])
        self.assertEqual(result.finalize_calls[-1], {"conversation_id": "conv-contract", "success": True, "task_id": "task-contract", "error_msg": ""})
```

- [x] **Step 2: Run RED**

Run:

```bash
DATABASE_URL=sqlite:////tmp/fusion_api_agent_loop_contract.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_agent_loop_contract.py -q
```

Expected: fail because the new helper/result type is not implemented.

- [x] **Step 3: Implement test-only harness**

Implement helpers inside the test file:

```python
@dataclass
class AgentLoopContractResult:
    events: list[dict]
    event_types: list[str]
    append_calls: list[dict]
    persist_calls: list[dict]
    finalize_calls: list[dict]
    session_started_calls: list[dict]
    session_status_calls: list[dict]
    step_started_calls: list[dict]
    step_completed_calls: list[dict]
    step_terminal_calls: list[dict]
```

The harness must patch Redis writes, persistence, session cache writes, LLM rounds, tool execution, DB session, and message building without starting local services.

- [x] **Step 4: Run GREEN**

Run the same test. Expected: pass.

## Task 2: Event And Partial Persist Contract

**Files:**
- Modify: `test/services/stream/test_agent_loop_contract.py`

- [x] **Step 1: Write failing tool round contract test**

Add a test that runs one tool round followed by one stop round and asserts:

```python
self.assertEqual(
    result.event_types,
    [
        "run_started",
        "step_started",
        "tool_call_started",
        "tool_call_completed",
        "step_completed",
        "step_started",
        "step_completed",
        "run_completed",
    ],
)
self.assertEqual([event["sequence"] for event in result.events], list(range(len(result.events))))
self.assertEqual(result.persist_calls[0]["partial"], True)
self.assertEqual(result.persist_calls[0]["block_types"], ["thinking"])
self.assertEqual(result.tool_execute_calls[0]["message_id"], "msg-contract")
self.assertEqual(result.persist_calls[-1]["partial"], False)
```

- [x] **Step 2: Run RED**

Run the new test. Expected: fail until harness records tool execution arguments and persist block types.

- [x] **Step 3: Extend harness minimally**

Record `execute_tools_parallel` kwargs and persist content block types.

- [x] **Step 4: Run GREEN**

Run the contract file. Expected: pass.

## Task 3: Redis Stream Terminal Contract

**Files:**
- Modify: `test/services/stream/test_agent_loop_contract.py`

- [x] **Step 1: Write failing Redis terminal tests**

Add tests asserting:

```python
self.assertEqual(result.redis_entry_types[0], "start")
self.assertIn("preparing", result.redis_entry_types)
self.assertEqual(result.redis_entry_types[-1], "done")
self.assertEqual(result.redis_meta["status"], "done")
```

And for failure:

```python
with self.assertRaises(RuntimeError):
    await self._run_agent_contract(rounds=RuntimeError("LLM 5xx"), use_real_redis_stream=True)
self.assertEqual(result.redis_entry_types[-1], "error")
self.assertEqual(result.redis_entries[-1][1]["content"], "LLM 5xx")
self.assertEqual(result.redis_meta["status"], "error")
```

- [x] **Step 2: Run RED**

Run the targeted tests. Expected: fail until the harness supports `use_real_redis_stream`.

- [x] **Step 3: Add real Redis capture and failure capture**

Patch `stream_state_service.get_redis_pool` to a `fakeredis.aioredis.FakeRedis`, call `init_stream()` before `generate_to_redis()`, then read `stream:chunks:conv-contract` and `stream:meta:conv-contract` before propagating expected exceptions.

- [x] **Step 4: Run GREEN**

Run the contract file. Expected: pass.

## Task 4: Session And Step State Contract

**Files:**
- Modify: `test/services/stream/test_agent_loop_contract.py`

- [x] **Step 1: Write failing session/step status tests**

Add tests for:

```python
self.assertEqual(result.session_started_calls[0]["message_id"], "msg-contract")
self.assertEqual(result.session_status_calls[-1]["status"], "completed")
self.assertEqual(result.session_status_calls[-1]["total_steps"], 2)
self.assertEqual(result.session_status_calls[-1]["total_tool_calls"], 1)
self.assertEqual([c["step_number"] for c in result.step_started_calls], [1, 2])
self.assertEqual([c["tool_calls_count"] for c in result.step_completed_calls], [1, 0])
```

Add failed path assertions:

```python
self.assertEqual(result.step_terminal_calls[-1]["status"], "failed")
self.assertEqual(result.session_status_calls[-1]["status"], "error")
```

- [x] **Step 2: Run RED**

Run targeted tests. Expected: fail until step completed/terminal data is captured fully.

- [x] **Step 3: Extend harness minimally**

Capture step completed and step terminal calls.

- [x] **Step 4: Run GREEN**

Run the contract file. Expected: pass.

## Task 5: Final Verification

**Files:**
- Test: `test/services/stream/test_agent_loop_contract.py`
- Test package marker: `test/services/stream/__init__.py`
- Test: `test/services/stream test/test_stream_handler.py test/test_tool_executor.py`

- [x] **Step 1: Run contract suite**

```bash
DATABASE_URL=sqlite:////tmp/fusion_api_agent_loop_contract.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_agent_loop_contract.py -q
```

- [x] **Step 2: Run stream regression suite**

```bash
DATABASE_URL=sqlite:////tmp/fusion_api_agent_loop_contract_regression.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream test/test_stream_handler.py test/test_tool_executor.py -q
```

- [x] **Step 3: Run lint/format checks for touched files**

```bash
/opt/homebrew/bin/python3.11 -m ruff check test/services/stream/__init__.py test/services/stream/test_agent_loop_contract.py
/opt/homebrew/bin/python3.11 -m ruff format --check test/services/stream/__init__.py test/services/stream/test_agent_loop_contract.py
```

- [x] **Step 4: Run CI-equivalent unittest discover**

```bash
DATABASE_URL=sqlite:////tmp/fusion_api_agent_loop_contract_full_unittest.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -u -m unittest discover -s test -t . -v
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-06-27-agent-loop-contract-tests.md test/services/stream/test_agent_loop_contract.py
git commit -m "test: 补充 agent loop 契约测试"
```

Commit body:

```text
Co-Authored-By: Codex <noreply@anthropic.com>
```
