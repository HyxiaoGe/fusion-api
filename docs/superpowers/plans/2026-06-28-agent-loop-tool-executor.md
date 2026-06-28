# Agent Loop 工具执行拆分 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 拆分 `tool_executor` 中的并发调度、单工具执行、预算处理、事件发送、日志写入和 record 构造，保持现有工具执行行为不变。

**Architecture:** `execute_tools_parallel(...)` 保持现有外部签名，内部只负责组装 `ToolExecutionBatchRequest` 并委托 `execute_tool_batch()`。单个工具调用由 `execute_one_tool_call()` 编排，具体职责拆成参数解析、预算准备、预算降级事件、handler 执行、日志写入和 `ToolExecutionRecord` 构造 helper。

**Tech Stack:** FastAPI 后端内部 Python 3.11、pytest、ruff、现有 agent-loop Redis Stream 工具事件协议。

---

### Task 1: 建立工具执行 request/helper 边界

**Files:**
- Modify: `test/test_tool_executor.py`
- Modify: `app/services/stream/tool_executor.py`

- [ ] **Step 1: 写先失败的 helper 接口测试**

新增测试使用 `ToolExecutionBatchRequest` 调用 `execute_tool_batch(request, [tool_call])`，断言成功路径仍返回 `ToolExecutionRecord`，并且 `handler.log` 收到 `message_id`、`trace_id`、`step_number` 与输入参数。

- [ ] **Step 2: 运行测试确认失败**

Run:
```bash
DATABASE_URL=sqlite:////tmp/fusion_api_tool_executor_red.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/test_tool_executor.py -q
```
Expected: FAIL，因为 `ToolExecutionBatchRequest` 或 `execute_tool_batch` 尚未导出。

- [ ] **Step 3: 最小实现 request 和 batch helper**

在 `tool_executor.py` 增加：
- `ToolExecutionBatchRequest`
- `ToolExecutionIds`
- `execute_tool_batch()`
- `execute_one_tool_call()`
- `resolve_tool_handler()`
- `parse_tool_arguments()`
- `prepare_tool_arguments()`
- `emit_budget_result()`
- `log_tool_execution()`
- `build_tool_execution_record()`

`execute_tools_parallel(...)` 只组装 request 并调用 `execute_tool_batch()`。

- [ ] **Step 4: 验证保持行为**

Run:
```bash
DATABASE_URL=sqlite:////tmp/fusion_api_tool_executor_green.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/test_tool_executor.py test/services/stream/test_tool_call_lifecycle.py test/services/stream/test_agent_loop_contract.py test/test_stream_handler.py -q
/opt/homebrew/bin/python3.11 scripts/check_quality.py
/opt/homebrew/bin/python3.11 scripts/check_architecture.py
/opt/homebrew/bin/python3.11 -m ruff check app/services/stream/tool_executor.py test/test_tool_executor.py
/opt/homebrew/bin/python3.11 -m ruff format --check app/services/stream/tool_executor.py test/test_tool_executor.py
```

验收标准：
- `app/services/stream/tool_executor.py:execute_tools_parallel()` 和原嵌套 `_run_one()` 不再出现在质量扫描的超长函数列表。
- 工具执行相关测试保持通过。
- 真实外部签名不变，`runner -> tool_round -> execute_tools_parallel` 调用链不需要调整。
- 不启动本地 Fusion 服务。
