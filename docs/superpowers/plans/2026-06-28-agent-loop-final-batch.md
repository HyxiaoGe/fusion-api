# Agent Loop Final Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 一次性收尾 agent loop 剩余结构优化：抽出 round outcome 分发、集中 ToolRoundRequest/LimitSummaryStepRequest 构造、收薄 execution/wiring builder。

**Architecture:** 新增 `agent_loop_outcome.py` 保存 outcome 类型，新增 `agent_loop_round_outcome.py` 承担 stop/cancelled/tool_calls/unknown 分发，新增 `agent_loop_step_requests.py` 集中构造 tool round 与 limit summary 请求。`agent_loop_driver.py` 只保留循环、limit 检查、step 启动和 round 调用；`agent_loop_wiring.py` / `agent_loop_execution.py` 把字段映射下沉到数据对象/小 helper。

**Tech Stack:** Python 3.11、pytest、dataclasses、ruff、Fusion agent loop。

---

## File Structure

- Create: `app/services/stream/agent_loop_outcome.py`
  - 保存 `AgentLoopExit` 和 `AgentLoopOutcome`，避免 outcome 分发模块反向依赖 driver。
- Create: `app/services/stream/agent_loop_round_outcome.py`
  - 保存 `AgentRoundOutcomeRequest` 与 `handle_agent_round_outcome()`，封装 stop/cancelled/tool_calls/unknown 分发。
- Create: `app/services/stream/agent_loop_step_requests.py`
  - 保存 `build_tool_round_request()` 与 `build_limit_summary_step_request()`。
- Modify: `app/services/stream/agent_loop_driver.py`
  - 使用新模块，移除 round 分支副作用和 request 构造。
- Modify: `app/services/stream/agent_loop_lifecycle.py`
  - 从 `agent_loop_outcome.py` 导入 `AgentLoopExit`。
- Modify: `app/services/stream/agent_loop_execution.py`
  - 增加 `AgentLoopExecutionParts` / `build_agent_loop_runtime()`，让 `build_agent_loop_execution()` 只装配主要部件。
- Modify: `app/services/stream/agent_loop_wiring.py`
  - 给 `AgentLoopRunInput` / `AgentLoopWiringDependencies` 增加转换方法，收薄 `build_agent_loop_lifecycle_call()`。
- Test: `test/services/stream/test_agent_loop_step_requests.py`
- Test: `test/services/stream/test_agent_loop_round_outcome.py`
- Modify tests: `test/services/stream/test_agent_loop_execution.py`, `test/services/stream/test_agent_loop_wiring.py`

## Task 1: Request Builders

- [ ] **Step 1: Write failing tests**

Add `test/services/stream/test_agent_loop_step_requests.py` with tests importing:

```python
from app.services.stream.agent_loop_step_requests import (
    build_limit_summary_step_request,
    build_tool_round_request,
)
```

Assert `build_tool_round_request()` copies db/messages/state/runtime/step/round fields into `ToolRoundRequest`, and `build_limit_summary_step_request()` copies state/runtime/messages into `LimitSummaryStepRequest`.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_agent_loop_step_requests.py -q
```

Expected: FAIL because `agent_loop_step_requests` does not exist.

- [ ] **Step 3: Implement request builders**

Create `app/services/stream/agent_loop_step_requests.py` with:

```python
def build_tool_round_request(...): ...
def build_limit_summary_step_request(...): ...
```

- [ ] **Step 4: Verify green**

Run the same test; expected PASS.

## Task 2: Round Outcome Dispatcher

- [ ] **Step 1: Write failing tests**

Add `test/services/stream/test_agent_loop_round_outcome.py` with tests importing:

```python
from app.services.stream.agent_loop_round_outcome import AgentRoundOutcomeRequest, handle_agent_round_outcome
```

Cover stop, cancelled, tool_calls, and unknown fallback behavior.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_agent_loop_round_outcome.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement dispatcher and outcome module**

Create `agent_loop_outcome.py`; move outcome types there and re-export them from driver. Create `agent_loop_round_outcome.py` and update `agent_loop_driver.py` to call it.

- [ ] **Step 4: Verify green**

Run round outcome and driver tests; expected PASS.

## Task 3: Builder Thin-Out

- [ ] **Step 1: Write failing tests**

Extend execution/wiring tests to import and assert:

```python
from app.services.stream.agent_loop_execution import AgentLoopExecutionParts, build_agent_loop_runtime
```

and call:

```python
run_input.to_execution_request(...)
run_input.to_lifecycle_request(...)
dependencies.to_execution_dependencies()
dependencies.to_lifecycle_dependencies()
```

Expected: methods/classes missing.

- [ ] **Step 2: Implement builder helpers**

Add `AgentLoopExecutionParts`, `build_agent_loop_runtime()`, `AgentLoopRunInput.to_*()` and `AgentLoopWiringDependencies.to_*()`.

- [ ] **Step 3: Verify builder tests**

Run:

```bash
/Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_agent_loop_execution.py test/services/stream/test_agent_loop_wiring.py -q
```

Expected: PASS.

## Task 4: Final Verification and Single Delivery

- [ ] Run focused tests:

```bash
/Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_agent_loop_step_requests.py test/services/stream/test_agent_loop_round_outcome.py test/services/stream/test_agent_loop_driver.py test/services/stream/test_agent_loop_execution.py test/services/stream/test_agent_loop_wiring.py test/services/stream/test_agent_loop_contract.py -q
```

- [ ] Run full stream tests:

```bash
/Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream -q
```

- [ ] Run full repo tests and checks:

```bash
/Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest -q
/opt/homebrew/bin/ruff check .
/opt/homebrew/bin/ruff format --check app/services/stream test/services/stream/test_agent_loop_step_requests.py test/services/stream/test_agent_loop_round_outcome.py test/services/stream/test_agent_loop_execution.py test/services/stream/test_agent_loop_wiring.py
/opt/homebrew/bin/python3.11 scripts/check_architecture.py
/opt/homebrew/bin/python3.11 scripts/check_quality.py
git diff --check
```

- [ ] Single commit:

```bash
git commit -m "refactor: 收拢 agent loop 最终批次" -m "Co-Authored-By: Codex <noreply@anthropic.com>"
```

- [ ] Merge to master, push once, watch one CI/CD run, verify dev image/health, run one Chrome regression on `https://fusion.seanfield.org` using the existing logged-in Chrome tab.
