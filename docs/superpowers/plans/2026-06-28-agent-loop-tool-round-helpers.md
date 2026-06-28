# Agent Loop 工具回合拆分 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `tool_round` 从大函数编排拆成 request 对象和小职责 helper，保持工具回合事件顺序、持久化顺序、LLM 上下文注入和 reasoning 恢复语义不变。

**Architecture:** `handle_tool_calls_round()` 保留为工具回合入口，但改为接收 `ToolRoundRequest`。具体职责拆到同模块 helper：追加 thinking、预持久化、执行工具、注入 assistant/tool 消息、完成 step、恢复 reasoning 参数。`agent_loop_driver` 只负责把 runtime/state/round_result 组装成 request，不直接展开工具回合所有参数。

**Tech Stack:** FastAPI 后端内部 Python 3.11、pytest、ruff、现有 Redis Stream agent-loop 模块。

---

### Task 1: 锁住 request 接口和工具结果注入顺序

**Files:**
- Modify: `test/services/stream/test_tool_round.py`
- Modify: `app/services/stream/tool_round.py`
- Modify: `app/services/stream/agent_loop_driver.py`

- [ ] **Step 1: 写先失败的接口测试**

新增测试使用 `ToolRoundRequest` 调用 `handle_tool_calls_round(request=request)`，断言：
- 第一次 persist 发生在 execute 前，且只有 thinking block。
- execute 后先记录工具调用数，再注入 assistant/tool 消息，再第二次 partial persist。
- `complete_step_fn` 收到执行结果里的 tool_names 和 `len(results)`。
- outcome 仍返回 `len(tool_calls)` 和 tool_names。

- [ ] **Step 2: 运行测试确认失败**

Run:
```bash
DATABASE_URL=sqlite:////tmp/fusion_api_tool_round_red.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_tool_round.py -q
```
Expected: FAIL，因为 `ToolRoundRequest` 尚未导出或 `handle_tool_calls_round()` 还不接受 `request`。

- [ ] **Step 3: 最小实现 request 对象和 helper 拆分**

在 `tool_round.py` 增加 `ToolRoundRequest` dataclass，把原入口参数收敛到 request；拆出：
- `append_tool_round_reasoning()`
- `persist_tool_round_checkpoint()`
- `execute_tool_round_tools()`
- `append_tool_round_messages()`
- `complete_tool_round_step()`

`handle_tool_calls_round()` 只保留顺序编排。

- [ ] **Step 4: driver 改为 request 调用**

`agent_loop_driver._handle_tool_calls_round()` 构造 `ToolRoundRequest(...)` 并调用 `runtime.handle_tool_calls_round_fn(request=request)`。

- [ ] **Step 5: 验证**

Run:
```bash
DATABASE_URL=sqlite:////tmp/fusion_api_tool_round_green.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_tool_round.py test/services/stream/test_agent_loop_driver.py test/services/stream/test_agent_loop_contract.py test/test_stream_handler.py -q
/opt/homebrew/bin/python3.11 scripts/check_quality.py
/opt/homebrew/bin/python3.11 scripts/check_architecture.py
/Users/sean/code/fusion/fusion-api/.venv311/bin/python -m ruff check .
/Users/sean/code/fusion/fusion-api/.venv311/bin/python -m ruff format --check .
```

验收标准：
- `app/services/stream/tool_round.py:handle_tool_calls_round()` 不再出现在质量扫描的超长函数列表。
- 聚焦测试保持 43 个通过。
- 架构检查和 ruff 通过。
- 不启动本地 Fusion 服务。
