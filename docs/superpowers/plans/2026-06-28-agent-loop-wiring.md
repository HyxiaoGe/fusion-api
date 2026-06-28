# Agent Loop Wiring Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `StreamHandler.generate_to_redis()` 中的 execution/lifecycle dependency wiring 抽到独立模块，让 runner 只保留入口适配、limits 和 DB session 外层生命周期。

**Architecture:** 新增 `app/services/stream/agent_loop_wiring.py`，定义 run 输入、runner 注入依赖、以及 `build_agent_loop_lifecycle_call()`。新模块只装配 `AgentLoopExecutionRequest`、`AgentLoopDependencies`、`AgentLoopLifecycleRequest` 和 `AgentLoopLifecycleDependencies`；具体函数仍由 `runner.py` 按当前模块变量传入，保持现有 `app.services.stream.runner.*` patch 路径有效。

**Tech Stack:** Python 3.11、pytest、dataclasses、Fusion agent loop。

---

### Task 1: 抽取 agent loop wiring builder

**Files:**
- Create: `app/services/stream/agent_loop_wiring.py`
- Create: `test/services/stream/test_agent_loop_wiring.py`
- Modify: `app/services/stream/runner.py`
- Modify: `test/test_stream_handler.py`

- [x] **Step 1: 写先失败的 wiring 测试**

新增 `test/services/stream/test_agent_loop_wiring.py`，覆盖：
- `build_agent_loop_lifecycle_call()` 使用输入参数构造 `AgentLoopLifecycleRequest`。
- `options=None` / `capabilities=None` 时传给 call config builder 的是空 dict。
- `AgentLoopExecutionRequest` 保留 db、conversation_id、user_id、model_id、provider、message_id、task_id、trace_id。
- `AgentLoopDependencies` 使用 runner 注入的 session cache、redis writer factory、round/tool/summary/llm/stream/persist/log/warning/clock 函数。
- `AgentLoopLifecycleDependencies` 使用 runner 注入的 append/start/prepare/run/finalize/fallback 函数。

Run:
```bash
DATABASE_URL=sqlite:////tmp/fusion_api_agent_loop_wiring_red.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_agent_loop_wiring.py -q
```
Expected: FAIL，原因是 `app.services.stream.agent_loop_wiring` 尚不存在。

- [x] **Step 2: 实现 `agent_loop_wiring.py`**

实现：
- `AgentLoopRunInput`：保存 `generate_to_redis()` 的业务输入。
- `AgentLoopWiringDependencies`：保存 runner 注入的 builder、service、tool、finalizer、logger、clock、redis writer factory。
- `AgentLoopLifecycleCall`：保存 `request`、`execution`、`dependencies` 三元组。
- `build_agent_loop_lifecycle_call(run_input, db, limits, dependencies)`：集中创建 call config、execution context 和 lifecycle dependencies。

约束：
- 不从新模块静态 import `append_chunk`、`finalize_stream`、`llm_call_with_retry`、`execute_tools_parallel`、`persist_message` 等生产函数。
- 新模块可以 import 类型和 dataclass，以及 `AgentLoopExecutionRequest`、`AgentLoopDependencies`、`AgentLoopLifecycleRequest`、`AgentLoopLifecycleDependencies`。

- [x] **Step 3: 改 runner 调用 wiring builder**

在 `runner.py` 中：
- 继续保留 `AGENT_MAX_STEPS`、`AGENT_MAX_TOOL_CALLS`、`AGENT_TOTAL_TIMEOUT`，保证现有测试 patch constants 仍生效。
- 新增 `_agent_loop_limits()` 返回 `AgentLoopLimits(...)`。
- 新增 `_agent_loop_wiring_dependencies()` 返回 `AgentLoopWiringDependencies(...)`，所有函数引用都来自 runner 当前模块变量。
- `generate_to_redis()` 只构造 `AgentLoopRunInput`，打开 `SessionLocal()`，调用 `build_agent_loop_lifecycle_call(...)`，再调用 `run_agent_loop_lifecycle(...)`，最后 `db.close()`。

- [x] **Step 4: 补 runner patch 路径回归测试**

在 `test/test_stream_handler.py` 增加或调整测试：
- patch `app.services.stream.runner.run_agent_loop_lifecycle` 捕获 `execution` 和 `dependencies`。
- patch `app.services.stream.runner.llm_call_with_retry`、`stream_round`、`execute_tools_parallel`、`persist_message`、`append_chunk`、`finalize_stream` 为 sentinel/mock。
- 调用 `generate_to_redis()` 后断言 `execution.runtime.llm_call_fn`、`execution.runtime.stream_round_fn`、`execution.runtime.execute_tools_fn`、`execution.runtime.persist_message_fn`、`dependencies.append_chunk_fn`、`dependencies.finalize_stream_fn` 均为 runner patched 对象。

- [x] **Step 5: 验证保持行为**

Run:
```bash
DATABASE_URL=sqlite:////tmp/fusion_api_agent_loop_wiring_green.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_agent_loop_wiring.py test/services/stream/test_agent_loop_lifecycle.py test/services/stream/test_agent_loop_execution.py test/services/stream/test_agent_loop_run_completion.py test/services/stream/test_agent_loop_driver.py test/test_stream_handler.py -q
/opt/homebrew/bin/python3.11 -m ruff check app/services/stream/agent_loop_wiring.py app/services/stream/runner.py test/services/stream/test_agent_loop_wiring.py test/test_stream_handler.py
/opt/homebrew/bin/python3.11 -m ruff format --check app/services/stream/agent_loop_wiring.py app/services/stream/runner.py test/services/stream/test_agent_loop_wiring.py test/test_stream_handler.py
/opt/homebrew/bin/python3.11 scripts/check_quality.py
```

验收标准：
- `runner.generate_to_redis()` 不再直接拼 `AgentLoopDependencies(...)` / `AgentLoopLifecycleDependencies(...)`。
- `runner.generate_to_redis()` 行数显著下降，目标低于 50 行。
- 旧 `runner.*` patch 路径仍由测试锁住。
- 不启动本地 Fusion 服务。
