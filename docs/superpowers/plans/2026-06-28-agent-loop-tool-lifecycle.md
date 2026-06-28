# Agent Loop Tool Lifecycle 拆分 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 拆分 `execute_tool_with_lifecycle()` 的工具执行、耗时测量、失败结果构造和 completed 事件发送职责，保持现有工具事件协议不变。

**Architecture:** `execute_tool_with_lifecycle(...)` 保持外部入口不变，内部只负责无 emitter 快路径、started 事件和调用 helper。新增 `ToolLifecycleAttempt` 承载单次执行结果，`run_tool_attempt()` 封装计时/异常映射，`complete_tool_lifecycle()` 统一发送 completed 事件并补齐成功结果 duration。

**Tech Stack:** Python 3.11、pytest、ruff、Fusion agent_event 工具调用协议。

---

### Task 1: 拆分 tool_call lifecycle 执行 attempt 和 completed 收尾

**Files:**
- Modify: `app/services/stream/tool_call_lifecycle.py`
- Modify: `test/services/stream/test_tool_call_lifecycle.py`

- [x] **Step 1: 写先失败的 helper 边界测试**

在 `test/services/stream/test_tool_call_lifecycle.py` 增加：
- `run_tool_attempt(target, args, execute)` 成功时返回 `ToolLifecycleAttempt(result, duration_ms, cancelled_error=None)`，并把测量耗时写入 `duration_ms`。
- `run_tool_attempt(...)` 遇到普通异常时返回 `failed` ToolResult，不抛出。
- `complete_tool_lifecycle(...)` 成功时补齐 `result.duration_ms` 并只发送一次 completed 事件。

- [x] **Step 2: 运行测试确认失败**

Run:
```bash
DATABASE_URL=sqlite:////tmp/fusion_api_tool_lifecycle_red.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_tool_call_lifecycle.py -q
```
Expected: FAIL，原因是 `ToolLifecycleAttempt` / `run_tool_attempt` / `complete_tool_lifecycle` 尚未导出。

- [x] **Step 3: 最小实现 helper 拆分**

在 `tool_call_lifecycle.py` 增加：
- `@dataclass(frozen=True) ToolLifecycleAttempt`
- `measure_duration_ms(start_mono: float) -> int`
- `run_tool_attempt(...)`
- `complete_tool_lifecycle(...)`

保持 `execute_tool_with_lifecycle(...)` 行为：
- `emitter is None` 时仍只执行 tool，不构造 summary，不写 duration。
- 有 emitter 时先发 started。
- 成功结果若 `duration_ms is None` 则补测量值。
- 普通异常返回 failed ToolResult 并发送 completed。
- `asyncio.CancelledError` 发送 failed completed 后继续抛出。

- [x] **Step 4: 验证保持行为**

Run:
```bash
DATABASE_URL=sqlite:////tmp/fusion_api_tool_lifecycle_green.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_tool_call_lifecycle.py test/test_tool_executor.py test/services/stream/test_tool_round.py test/test_stream_handler.py -q
/opt/homebrew/bin/python3.11 -m ruff check app/services/stream/tool_call_lifecycle.py test/services/stream/test_tool_call_lifecycle.py
/opt/homebrew/bin/python3.11 -m ruff format --check app/services/stream/tool_call_lifecycle.py test/services/stream/test_tool_call_lifecycle.py
/opt/homebrew/bin/python3.11 scripts/check_quality.py
```

验收标准：
- `execute_tool_with_lifecycle()` 不再出现在质量扫描超长函数列表。
- tool_call started/completed 事件字段、顺序和 error 映射不变。
- 相关工具执行和 agent-loop 回归测试通过。
