# Agent Loop Lifecycle Facade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `StreamHandler.generate_to_redis()` 中的 run lifecycle 编排抽到独立 facade，runner 只保留输入配置、DB session 生命周期和依赖注入入口。

**Architecture:** 新增 `app/services/stream/agent_loop_lifecycle.py`，定义 `AgentLoopLifecycleRequest`、`AgentLoopLifecycleDependencies` 和 `run_agent_loop_lifecycle()`。该 facade 负责 append preparing、start run、prepare messages、run driver、terminal finalize、cancel/fail/fallback 处理。`runner.py` 继续把当前模块变量传入依赖对象，保持既有 `app.services.stream.runner.*` patch 路径有效。

**Tech Stack:** Python 3.11、pytest、dataclasses、Fusion agent loop。

---

### Task 1: 抽取 agent loop lifecycle facade

**Files:**
- Create: `app/services/stream/agent_loop_lifecycle.py`
- Create: `test/services/stream/test_agent_loop_lifecycle.py`
- Modify: `app/services/stream/runner.py`

- [x] **Step 1: 写先失败的 lifecycle 测试**

新增测试覆盖：
- completed 路径顺序：`append preparing` → `start_agent_run` → `prepare_agent_loop_messages` → `run_agent_loop` → `finalize_completed_run`。
- prepared initial content blocks 被追加到 `execution.state.content_blocks` 后再进入 driver。
- superseded 路径调用 `finalize_superseded_run` 且不调用 completed finalize。
- exception wrapper 保持 CancelledError/Exception finalize 后 re-raise，并执行 fallback。

Run:
```bash
DATABASE_URL=sqlite:////tmp/fusion_api_agent_loop_lifecycle_red.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_agent_loop_lifecycle.py -q
```
Expected: FAIL，原因是 `app.services.stream.agent_loop_lifecycle` 尚不存在。

- [x] **Step 2: 最小实现 `agent_loop_lifecycle.py`**

实现：
- `AgentLoopLifecycleRequest`：保存 raw messages、vision/file/original message、call config、limits。
- `AgentLoopLifecycleDependencies`：保存 runner 传入的 append/start/prepare/run/finalize/fallback/logger 依赖。
- `run_agent_loop_lifecycle(...)`：包住 try/except/finally，调用内部 `_run_success_path(...)`、`_finalize_cancelled(...)`、`_finalize_failed(...)`、`_write_fallback(...)`。
- 内部 completed/superseded finalize 继续复用 `agent_loop_run_completion.py` 里的函数，terminal state 继续用 `map_run_terminal_state(...)`。

- [x] **Step 3: 改 runner 使用 lifecycle facade**

在 `generate_to_redis()` 中：
- 保留 `call_config`、`db = SessionLocal()`、`limits`、`execution = build_agent_loop_execution(...)`。
- 用 `run_agent_loop_lifecycle(...)` 替代原 try/except/finally 中除 `db.close()` 以外的 lifecycle 逻辑。
- `runner.py` 继续导入并传入 `append_chunk`、`finalize_stream`、`start_agent_run`、`prepare_agent_loop_messages`、`run_agent_loop`、`persist_message` 等当前模块变量，保证旧 patch 路径不变。

- [x] **Step 4: 验证保持行为**

Run:
```bash
DATABASE_URL=sqlite:////tmp/fusion_api_agent_loop_lifecycle_green.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_agent_loop_lifecycle.py test/services/stream/test_agent_loop_execution.py test/services/stream/test_agent_loop_run_completion.py test/services/stream/test_agent_loop_driver.py test/test_stream_handler.py -q
/opt/homebrew/bin/python3.11 -m ruff check app/services/stream/agent_loop_lifecycle.py app/services/stream/runner.py test/services/stream/test_agent_loop_lifecycle.py
/opt/homebrew/bin/python3.11 -m ruff format --check app/services/stream/agent_loop_lifecycle.py app/services/stream/runner.py test/services/stream/test_agent_loop_lifecycle.py
/opt/homebrew/bin/python3.11 scripts/check_quality.py
```

验收标准：
- `generate_to_redis()` 行数继续下降，runner 不再直接编排 start/prepare/run/finalize/except/fallback。
- 新增 lifecycle 模块不产生新的超长函数热点。
- `test_stream_handler.py` 和契约测试继续通过。
- 不启动本地 Fusion 服务。
