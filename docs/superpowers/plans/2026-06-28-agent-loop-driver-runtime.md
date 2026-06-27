# Agent Loop Driver Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `runner.py` 里的 agent loop `while True` 状态机和触顶总结迁到独立 driver/runtime，保留现有 Redis Stream、事件、落库和终态行为。

**Architecture:** `runner.py` 继续负责外围准备、消息构建、异常/finally、最终落库与 run 终态。新增 `AgentLoopRuntime` 收拢循环运行时依赖，新增 `run_agent_loop()` 只执行每轮 LLM、工具回合、superseded 识别和触顶总结，并返回 outcome 供 `runner.py` 决定是否继续最终收尾。

**Tech Stack:** FastAPI 服务层 Python 3.11、pytest/unittest、Redis Stream agent event helpers。

---

### Task 1: Driver 契约测试

**Files:**
- Create: `test/services/stream/test_agent_loop_driver.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest
from dataclasses import dataclass

from app.schemas.chat import Usage
from app.services.stream.agent_loop_driver import AgentLoopExit, run_agent_loop
from app.services.stream.agent_loop_policy import AgentLoopLimits
from app.services.stream.agent_loop_runtime import AgentLoopRuntime
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.agent_round import AgentRoundResult


@dataclass
class DummyStepContext:
    step_id: str
    thinking_block_id: str
    text_block_id: str


class DummyEmitter:
    def __init__(self):
        self.limit_reasons = []

    async def run_limit_reached(self, *, reason):
        self.limit_reasons.append(reason)


class AgentLoopDriverTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_round_completes_text_step_and_returns_completed(self):
        state = AgentLoopState()
        started = []
        completed = []
        rounds = []

        async def start_step_fn(**kwargs):
            started.append(kwargs["step_number"])
            return DummyStepContext("step-1", "thinking-1", "text-1")

        async def complete_step_fn(**kwargs):
            completed.append(kwargs["context"].step_id)

        async def run_round_fn(**kwargs):
            rounds.append(kwargs["step_number"])
            return AgentRoundResult(
                reasoning_buf="思考",
                content_buf="回答",
                tool_calls=[],
                finish_reason="stop",
                accumulated_usage=Usage(input_tokens=3, output_tokens=5),
            )

        runtime = AgentLoopRuntime(
            conversation_id="conv",
            task_id="task",
            run_id="run",
            user_id="user",
            model_id="model",
            provider="provider",
            litellm_model="litellm",
            litellm_kwargs={},
            should_use_reasoning=True,
            call_kwargs={},
            assistant_message_id="assistant",
            run_start=0,
            limits=AgentLoopLimits(max_steps=8, max_tool_calls=20, total_timeout_s=300),
            emitter=DummyEmitter(),
            session_cache=object(),
            network_budget=object(),
            start_step_fn=start_step_fn,
            complete_step_fn=complete_step_fn,
            run_round_fn=run_round_fn,
            handle_tool_calls_round_fn=None,
            run_limit_summary_step_fn=None,
            llm_call_fn=None,
            stream_round_fn=None,
            execute_tools_fn=None,
            persist_message_fn=None,
            log_round_summary_fn=lambda **kwargs: None,
            warning_fn=lambda message: None,
            clock=lambda: 1,
        )

        outcome = await run_agent_loop(db=object(), messages=[], state=state, runtime=runtime)

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        self.assertEqual(started, [1])
        self.assertEqual(rounds, [1])
        self.assertEqual(completed, ["step-1"])
        self.assertEqual(state.current_step_id, None)
        self.assertEqual(state.accumulated_usage.input_tokens, 3)
        self.assertEqual(len(state.content_blocks), 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
DATABASE_URL=sqlite:////tmp/fusion_api_agent_loop_driver_red.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_agent_loop_driver.py -q
```

Expected: FAIL because `app.services.stream.agent_loop_driver` does not exist.

### Task 2: Driver/Runtime 抽取

**Files:**
- Create: `app/services/stream/agent_loop_runtime.py`
- Create: `app/services/stream/agent_loop_driver.py`
- Modify: `app/services/stream/runner.py`
- Modify: `test/services/stream/test_agent_loop_driver.py`

- [ ] **Step 1: Implement runtime/outcome types**

Create `AgentLoopRuntime` as a dataclass that carries ids/config/dependencies currently used only by the loop. Create `AgentLoopExit` with `COMPLETED` and `SUPERSEDED`, plus `AgentLoopOutcome(exit, error_msg=None)`.

- [ ] **Step 2: Move the loop body into `run_agent_loop()`**

Move these existing behaviors out of `runner.py` without semantic changes:
- limit check order: timeout > max_steps > max_tool_calls via `check_agent_loop_limit`
- normal stop: append reasoning/text blocks, complete text step, clear current step
- cancelled: append blocks, keep current step id, return `SUPERSEDED` for `runner.py` to persist/interrupted/finalize once
- tool calls: delegate to `handle_tool_calls_round_fn`, record tool count through state callback, clear current step, continue
- unknown finish reason: append blocks, mark unknown terminated, complete text step, clear current step
- limit summary: start independent summary step through `run_limit_summary_step_fn`, update usage, clear current step

- [ ] **Step 3: Replace `runner.py` loop with runtime construction and driver call**

`runner.py` should construct `AgentLoopRuntime`, call `run_agent_loop()`, and return early only for `SUPERSEDED`. Final persist, `map_run_terminal_state()`, `complete_agent_run()` and `finalize_stream(success=True)` remain in `runner.py`.

- [ ] **Step 4: Expand driver tests**

Add focused tests for:
- immediate max-step limit calls summary and emits `run_limit_reached`
- cancelled round returns `SUPERSEDED` without terminal side effects, plus runner-level test covers persist/interrupted/finalize

### Task 3: Verification and CI/CD

**Files:**
- Modify as required by implementation only.

- [ ] **Step 1: Run focused tests**

```bash
DATABASE_URL=sqlite:////tmp/fusion_api_agent_loop_driver_focus.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_agent_loop_driver.py test/services/stream/test_agent_loop_contract.py test/test_stream_handler.py -q
```

- [ ] **Step 2: Run architecture and lint checks**

```bash
/Users/sean/code/fusion/fusion-api/.venv311/bin/python scripts/check_architecture.py
/opt/homebrew/bin/python3.11 -m ruff check .
/opt/homebrew/bin/python3.11 -m ruff format --check app/services/stream/agent_loop_runtime.py app/services/stream/agent_loop_driver.py app/services/stream/runner.py test/services/stream/test_agent_loop_driver.py
git diff --check
```

- [ ] **Step 3: Commit and push**

Commit message:

```text
refactor: 抽出 agent loop driver runtime

Co-Authored-By: Codex <noreply@anthropic.com>
```

- [ ] **Step 4: Follow CI/CD**

Push through normal Git flow, monitor GitHub Actions and deployed dev image. Do not start local Fusion services.
