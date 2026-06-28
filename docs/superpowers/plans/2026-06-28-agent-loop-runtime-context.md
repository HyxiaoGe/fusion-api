# Agent Loop Runtime Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `StreamHandler.generate_to_redis()` 中的 `AgentLoopState`、`AgentLoopRuntime`、`AgentLoopRunCompletionContext` 装配逻辑移动到独立模块，降低 runner 的状态/依赖构建负担。

**Architecture:** 新增 `app/services/stream/agent_loop_execution.py`，定义 `AgentLoopDependencies` 和 `AgentLoopExecutionContext`，集中创建 state、network budget、emitter、runtime、completion context。`runner.py` 保持对现有可 patch 依赖的导入，并把这些依赖显式传给 builder，避免破坏既有测试 patch 路径。

**Tech Stack:** Python 3.11、pytest、dataclasses、Fusion agent loop。

---

### Task 1: 抽取 Agent Loop runtime/completion 装配

**Files:**
- Create: `app/services/stream/agent_loop_execution.py`
- Create: `test/services/stream/test_agent_loop_execution.py`
- Modify: `app/services/stream/runner.py`

- [x] **Step 1: 写先失败的 execution context 测试**

新增测试覆盖：
- `build_agent_loop_execution(...)` 使用传入 `trace_id` 作为 `run_id`。
- `runtime` 与 `completion_context` 共享同一个 `state`、`emitter`、`session_cache`。
- `runtime.limits`、`runtime.call_kwargs`、`runtime.should_use_reasoning` 来自传入配置。
- `duration_ms_factory()` 使用同一个 `run_start` 和 clock 计算运行耗时。

Run:
```bash
DATABASE_URL=sqlite:////tmp/fusion_api_agent_loop_execution_red.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_agent_loop_execution.py -q
```
Expected: FAIL，原因是 `app.services.stream.agent_loop_execution` 尚不存在。

- [x] **Step 2: 最小实现 `agent_loop_execution.py`**

实现：
- `AgentLoopDependencies`：保存 runner 传入的 session cache、step/run/tool/LLM/persist/log/warning/clock 依赖。
- `AgentLoopExecutionContext`：保存 `state`、`network_budget`、`emitter`、`runtime`、`completion_context`。
- `build_agent_loop_execution(...)`：创建 state、network budget、emitter、duration factory、completion context 和 runtime。

- [x] **Step 3: 改 runner 使用 execution builder**

在 `generate_to_redis()` 中：
- 保留 `call_config` 和 `db = SessionLocal()`。
- 用 `build_agent_loop_execution(...)` 创建 `execution`。
- `start_agent_run`、`prepare_agent_loop_messages`、`run_agent_loop`、finalize 分支都使用 `execution.state`、`execution.runtime`、`execution.completion_context`、`execution.emitter`。
- 不改变 run_started tools 时序，不改变异常重抛和 finally fallback 行为。

- [x] **Step 4: 验证保持行为**

Run:
```bash
DATABASE_URL=sqlite:////tmp/fusion_api_agent_loop_execution_green.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_agent_loop_execution.py test/services/stream/test_agent_loop_driver.py test/services/stream/test_agent_loop_run_completion.py test/test_stream_handler.py -q
/opt/homebrew/bin/python3.11 -m ruff check app/services/stream/agent_loop_execution.py app/services/stream/runner.py test/services/stream/test_agent_loop_execution.py
/opt/homebrew/bin/python3.11 -m ruff format --check app/services/stream/agent_loop_execution.py app/services/stream/runner.py test/services/stream/test_agent_loop_execution.py
/opt/homebrew/bin/python3.11 scripts/check_quality.py
```

验收标准：
- `generate_to_redis()` 行数下降，runner 不再直接构造 `AgentLoopRuntime` 和 `AgentLoopRunCompletionContext`。
- 现有 `test_stream_handler.py` 行为测试继续通过。
- 不启动本地 Fusion 服务。
