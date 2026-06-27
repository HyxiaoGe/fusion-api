# Agent Loop Request Prep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 agent loop 进入 driver 前的 call config 与 message prep 从 `runner.py` 拆到独立模块，保留现有 run_started、URL 预处理、文件注入和工具契约行为。

**Architecture:** `runner.py` 继续负责 DB 生命周期、preparing、run_started、driver 调用、终态落库和异常/finally 收尾。新增 `agent_loop_request_prep.py` 提供两个阶段：`build_agent_loop_call_config()` 在 `start_agent_run()` 前同步构造 reasoning/tool call 配置并捕获 announced tools；`prepare_agent_loop_messages()` 在 `start_agent_run()` 后执行 DB/user prompt、文件注入、URL 预处理和工具契约注入。

**Tech Stack:** FastAPI 服务层 Python 3.11、pytest/unittest、现有 stream contract tests。

---

### Task 1: Request Prep 契约测试

**Files:**
- Create: `test/services/stream/test_agent_loop_request_prep.py`

- [ ] **Step 1: Write failing tests**

Add tests that import the intended API before it exists:

```python
from app.services.stream.agent_loop_request_prep import (
    AgentLoopCallConfig,
    build_agent_loop_call_config,
    prepare_agent_loop_messages,
)
```

Cover:
- `functionCalling=True` adds `web_search`, `tool_choice="auto"`, captures `announced_tools=["web_search"]`.
- `provider="volcengine"` plus reasoning adds disabled-thinking `extra_body`.
- `use_reasoning=False` overrides `capabilities.deepThinking=True`.
- message prep injects non-image file content, inserts URL context before final user message, prepends the web-search contract after existing system messages, and returns initial content blocks for URL read.

- [ ] **Step 2: Run RED**

```bash
DATABASE_URL=sqlite:////tmp/fusion_api_request_prep_red.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_agent_loop_request_prep.py -q
```

Expected: FAIL because `app.services.stream.agent_loop_request_prep` does not exist.

### Task 2: Extract Request Prep

**Files:**
- Create: `app/services/stream/agent_loop_request_prep.py`
- Modify: `app/services/stream/runner.py`
- Modify: `test/test_stream_handler.py`
- Modify: `test/services/stream/test_agent_loop_contract.py`

- [ ] **Step 1: Implement call config**

Create:

```python
@dataclass(frozen=True)
class AgentLoopCallConfig:
    should_use_reasoning: bool
    supports_function_calling: bool
    call_kwargs: dict
    announced_tools: list[str]
```

`build_agent_loop_call_config()` must preserve existing behavior:
- default `should_use_reasoning` follows `capabilities["deepThinking"]`
- explicit `options["use_reasoning"]` overrides the default
- function calling adds `build_web_search_tool()` and `tool_choice="auto"`
- volcengine + reasoning applies `merge_extra_body(..., {"thinking": {"type": "disabled"}})`
- `announced_tools` is captured before later URL preprocessing mutates `call_kwargs`

- [ ] **Step 2: Implement message prep**

Create:

```python
@dataclass(frozen=True)
class AgentLoopPreparedMessages:
    messages: list[dict]
    initial_content_blocks: list
```

`prepare_agent_loop_messages()` must preserve existing behavior:
- create `FileRepository(db)`
- load `User.system_prompt`
- call `build_llm_messages(...)`
- inject non-image file contents through `inject_file_content(...)`
- run `preprocess_url_in_message(original_message, supports_function_calling, call_kwargs)`
- insert URL context with `messages.insert(-1, url_context_msg)`
- append successful URL block to `initial_content_blocks`
- inject the web-search consistency contract only when `web_search` is present and not already injected

- [ ] **Step 3: Wire runner**

`runner.py` should:
- call `build_agent_loop_call_config()` before `start_agent_run()`
- pass `call_config.announced_tools` to `start_agent_run()`
- call `prepare_agent_loop_messages()` after `start_agent_run()`
- extend `state.content_blocks` with `prepared_messages.initial_content_blocks`
- pass `call_config.should_use_reasoning` and `call_config.call_kwargs` into `AgentLoopRuntime`

Do not move DB session lifecycle, `append_chunk("preparing")`, `start_agent_run()`, final persist, run finalizer, `finalize_stream()`, exception paths, or `finally`.

### Task 3: Verification and CI/CD

**Files:**
- Modify as required by implementation only.

- [ ] **Step 1: Run focused tests**

```bash
DATABASE_URL=sqlite:////tmp/fusion_api_request_prep_focus.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_agent_loop_request_prep.py test/services/stream/test_agent_loop_contract.py test/test_stream_handler.py -q
```

- [ ] **Step 2: Run architecture and lint checks**

```bash
/Users/sean/code/fusion/fusion-api/.venv311/bin/python scripts/check_architecture.py
/opt/homebrew/bin/python3.11 -m ruff check .
/opt/homebrew/bin/python3.11 -m ruff format --check app/services/stream/agent_loop_request_prep.py app/services/stream/runner.py test/services/stream/test_agent_loop_request_prep.py test/services/stream/test_agent_loop_contract.py test/test_stream_handler.py
git diff --check
```

- [ ] **Step 3: Run full tests**

```bash
DATABASE_URL=sqlite:////tmp/fusion_api_request_prep_full.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m unittest discover -s test -t .
```

- [ ] **Step 4: Commit and push**

Commit message:

```text
refactor: 抽出 agent loop 请求准备

Co-Authored-By: Codex <noreply@anthropic.com>
```

Push `master` and follow CI/CD to dev. Do not start local Fusion services.
