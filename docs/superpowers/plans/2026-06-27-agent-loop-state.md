# Agent Loop State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `StreamHandler.generate_to_redis` 内散落的 agent loop 状态集中到纯状态对象 `AgentLoopState`，保持运行行为零变化。

**Architecture:** 新增 `app/services/stream/agent_loop_state.py`，只管理内存状态和小型状态转移，不触碰 DB、Redis、LLM、emitter 或 session_cache。`runner.py` 继续负责编排和副作用，但通过 `state` 读写 step、usage、content_blocks、terminal 标记和统计。现有 contract suite 负责守住事件顺序、partial persist、Redis 终态、session/step 状态。

**Tech Stack:** Python 3.11、unittest、FastAPI 后端现有 `app.schemas.chat.Usage` 和 stream 测试目录。

---

### Task 1: 新增 AgentLoopState 纯状态对象

**Files:**
- Create: `app/services/stream/agent_loop_state.py`
- Create: `test/services/stream/test_agent_loop_state.py`

- [x] **Step 1: 写失败测试**

新增 `test/services/stream/test_agent_loop_state.py`：

```python
import unittest

from app.schemas.chat import TextBlock, Usage
from app.services.stream.agent_loop_state import AgentLoopState


class AgentLoopStateTests(unittest.TestCase):
    def test_initial_state_matches_runner_defaults(self):
        state = AgentLoopState()

        self.assertEqual(state.content_blocks, [])
        self.assertEqual(state.accumulated_usage, Usage(input_tokens=0, output_tokens=0))
        self.assertEqual(state.step, 0)
        self.assertEqual(state.total_tool_calls, 0)
        self.assertEqual(state.current_step_id, None)
        self.assertEqual(state.finish_reason, "stop")
        self.assertEqual(state.limit_reason, None)
        self.assertEqual(state.unknown_terminated, False)
        self.assertEqual(state.terminal_emitted, False)

    def test_step_and_tool_call_mutations_are_explicit(self):
        state = AgentLoopState()

        self.assertEqual(state.next_step_number(), 1)
        state.mark_current_step("step-1")
        state.record_executed_tool_calls(2)
        state.clear_current_step()

        self.assertEqual(state.step, 1)
        self.assertEqual(state.total_tool_calls, 2)
        self.assertEqual(state.current_step_id, None)
        self.assertEqual(state.run_stats("run-1").total_steps, 1)
        self.assertEqual(state.run_stats("run-1").total_tool_calls, 2)

    def test_usage_content_and_terminal_mutations_are_explicit(self):
        state = AgentLoopState()
        block = TextBlock(type="text", id="blk-1", text="answer")

        state.content_blocks.append(block)
        state.update_usage(Usage(input_tokens=3, output_tokens=5))
        state.mark_unknown_terminated()
        state.mark_terminal_emitted()

        self.assertEqual(state.content_blocks, [block])
        self.assertEqual(state.final_usage(), Usage(input_tokens=3, output_tokens=5))
        self.assertEqual(state.unknown_terminated, True)
        self.assertEqual(state.terminal_emitted, True)

    def test_final_usage_omits_zero_input_usage(self):
        state = AgentLoopState()

        self.assertIsNone(state.final_usage())


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: 跑测试确认红灯**

Run:

```bash
DATABASE_URL=sqlite:////tmp/fusion_api_agent_loop_state_red.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m unittest test.services.stream.test_agent_loop_state -v
```

Expected: FAIL，原因是 `app.services.stream.agent_loop_state` 还不存在。

- [x] **Step 3: 写最小实现**

新增 `app/services/stream/agent_loop_state.py`：

```python
"""Agent loop 内存状态与纯状态转移。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.chat import Usage
from app.services.stream.agent_loop_policy import AgentLoopLimitReason
from app.services.stream.run_finalizer import AgentRunStats


@dataclass
class AgentLoopState:
    content_blocks: list = field(default_factory=list)
    accumulated_usage: Usage = field(default_factory=lambda: Usage(input_tokens=0, output_tokens=0))
    step: int = 0
    total_tool_calls: int = 0
    current_step_id: str | None = None
    finish_reason: str = "stop"
    limit_reason: AgentLoopLimitReason | None = None
    unknown_terminated: bool = False
    terminal_emitted: bool = False

    def next_step_number(self) -> int:
        self.step += 1
        return self.step

    def mark_current_step(self, step_id: str) -> None:
        self.current_step_id = step_id

    def clear_current_step(self) -> None:
        self.current_step_id = None

    def record_executed_tool_calls(self, tool_call_count: int) -> None:
        self.total_tool_calls += tool_call_count

    def update_usage(self, usage: Usage) -> None:
        self.accumulated_usage = usage

    def final_usage(self) -> Usage | None:
        if self.accumulated_usage.input_tokens <= 0:
            return None
        return self.accumulated_usage

    def mark_unknown_terminated(self) -> None:
        self.unknown_terminated = True

    def mark_terminal_emitted(self) -> None:
        self.terminal_emitted = True

    def run_stats(self, run_id: str) -> AgentRunStats:
        return AgentRunStats(
            run_id=run_id,
            total_steps=self.step,
            total_tool_calls=self.total_tool_calls,
        )
```

- [x] **Step 4: 跑测试确认绿灯**

Run:

```bash
DATABASE_URL=sqlite:////tmp/fusion_api_agent_loop_state_green.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m unittest test.services.stream.test_agent_loop_state -v
```

Expected: PASS。

### Task 2: 将 runner 接入 AgentLoopState

**Files:**
- Modify: `app/services/stream/runner.py`
- Test: `test/services/stream/test_agent_loop_contract.py`
- Test: `test/test_stream_handler.py`

- [x] **Step 1: 替换散落状态变量**

在 `runner.py` 中导入 `AgentLoopState`，用 `state = AgentLoopState()` 替代以下局部变量：

```python
content_blocks = []
accumulated_usage = Usage(input_tokens=0, output_tokens=0)
step = 0
total_tool_calls = 0
finish_reason = "stop"
current_step_id = None
terminal_emitted = False
limit_reason = None
unknown_terminated = False
```

关键替换规则：

```python
content_blocks -> state.content_blocks
accumulated_usage -> state.accumulated_usage
step -> state.step
total_tool_calls -> state.total_tool_calls
finish_reason -> state.finish_reason
current_step_id -> state.current_step_id
terminal_emitted -> state.terminal_emitted
limit_reason -> state.limit_reason
unknown_terminated -> state.unknown_terminated
_run_stats() -> state.run_stats(run_id)
_record_executed_tool_calls -> state.record_executed_tool_calls
```

每次开启 step 使用：

```python
step_number = state.next_step_number()
step_context = await start_agent_step(..., step_number=step_number, on_step_started=state.mark_current_step)
state.mark_current_step(step_context.step_id)
```

每个正常闭合 step 后调用：

```python
state.clear_current_step()
```

每个终态成功写出后调用：

```python
state.mark_terminal_emitted()
```

- [x] **Step 2: 跑 focused 契约测试**

Run:

```bash
DATABASE_URL=sqlite:////tmp/fusion_api_agent_loop_state_contract.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_agent_loop_contract.py test/test_stream_handler.py -q
```

Expected: 全部 PASS，尤其是事件顺序、partial persist、Redis terminal、session/step 状态不变。

### Task 3: 全量验证、提交和 CI/CD

**Files:**
- Modify: `docs/superpowers/plans/2026-06-27-agent-loop-state.md`

- [x] **Step 1: 跑完整后端测试和 lint**

Run:

```bash
DATABASE_URL=sqlite:////tmp/fusion_api_agent_loop_state_full.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -u -m unittest discover -s test -t . -v
/opt/homebrew/bin/python3.11 -m ruff check .
git diff --check
```

Expected: unittest 341+ tests OK，ruff 无错误，`git diff --check` 无输出。

- [ ] **Step 2: 提交并推送**

Run:

```bash
git add -f docs/superpowers/plans/2026-06-27-agent-loop-state.md
git add app/services/stream/agent_loop_state.py app/services/stream/runner.py test/services/stream/test_agent_loop_state.py
git commit -m "refactor: 抽出 agent loop 状态对象" -m "Co-Authored-By: Codex <noreply@anthropic.com>"
git push -u origin refactor/agent-loop-state
```

- [ ] **Step 3: 监控 GitHub Actions**

Run:

```bash
gh run list --repo HyxiaoGe/fusion-api --branch refactor/agent-loop-state --limit 3
gh run watch <run_id> --repo HyxiaoGe/fusion-api --interval 10 --exit-status
```

Expected: 分支 CI success。若用户继续要求部署，则按正常 CI/CD 合入 master 并监控 dev 部署。

---

## Self-Review

- 覆盖目标：计划只抽 `AgentLoopState`，不引入 `AgentLoopRuntime`，保持行为零变化。
- TDD：先新增 `test_agent_loop_state.py` 并确认红灯，再新增生产代码。
- 风险控制：contract suite 和 `test_stream_handler.py` 作为行为保护网。
- 本地服务：不启动 uvicorn、Next、Docker 或本地 Fusion 服务。
