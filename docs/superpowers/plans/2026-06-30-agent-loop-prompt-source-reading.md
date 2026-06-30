# Agent Loop Prompt Source Reading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 调整 agent loop 的联网提示词，使搜索结果不再阻止后续 `url_read`，并让模型优先深读高价值来源。

**Architecture:** 先把 agent loop 相关提示词集中到 `app/ai/prompts/agent_loop.py`，再由 message builder、tool schema 和 tool handler 引用。当前仍保留代码内常量，后续迁移 PromptHub 时只替换该模块的提供方。

**Tech Stack:** FastAPI, LiteLLM function calling, Python unittest/pytest.

---

### Task 1: 集中 Prompt 常量并锁定搜索上下文行为

**Files:**
- Create: `app/ai/prompts/agent_loop.py`
- Modify: `app/services/chat/message_builder.py`
- Modify: `app/services/stream/agent_loop_request_prep.py`
- Modify: `app/services/tool_handlers/web_search.py`
- Test: `test/test_tool_handlers.py`
- Test: `test/services/stream/test_agent_loop_request_prep.py`

- [ ] **Step 1: Write failing tests**

Add assertions that `WebSearchHandler.format_llm_context()` no longer contains `不要再发起搜索或输出任何工具调用指令`, and does contain guidance to call `url_read` for key facts, official announcements, and original details. Add assertions that `inject_tool_usage_contract()` still injects the centralized contract prompt.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest test/test_tool_handlers.py::WebSearchHandlerTests::test_format_llm_context_allows_targeted_url_read_after_search test/services/stream/test_agent_loop_request_prep.py::AgentLoopRequestPrepTests::test_tool_usage_contract_uses_centralized_prompt -q
```

Expected: tests fail because the current search context forbids follow-up tool calls and no centralized prompt module exists.

- [ ] **Step 3: Implement minimal code**

Create `app/ai/prompts/agent_loop.py` with constants for current-date prompt building, tool usage contract, search result context instructions, and source reading guidance. Import those constants/functions from the existing call sites.

- [ ] **Step 4: Run tests to verify GREEN**

Run the same targeted pytest command. Expected: both tests pass.

### Task 2: 强化 `url_read` Tool 描述

**Files:**
- Modify: `app/ai/tools.py`
- Test: `test/test_ai_tools.py`

- [ ] **Step 1: Write failing test**

Add assertions that `URL_READ_TOOL` tells the model to prioritize official sources, original announcements, high-relevance pages, and down-rank video/forum/low-relevance results after search.

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
python -m pytest test/test_ai_tools.py::AiToolSchemaTests::test_url_read_schema_prioritizes_high_value_search_results -q
```

Expected: test fails because the current description only says “搜索结果中的某个链接需要深入阅读”.

- [ ] **Step 3: Implement minimal code**

Update `URL_READ_TOOL` description to include high-value source selection rules while preserving the existing URL/reason schema.

- [ ] **Step 4: Run test to verify GREEN**

Run the same targeted pytest command. Expected: test passes.

### Task 3: 回归、提交、部署和真实对话验证

**Files:**
- No additional files expected.

- [ ] **Step 1: Run backend regression**

Run:

```bash
python -m pytest test/test_ai_tools.py test/test_tool_handlers.py test/services/stream/test_agent_loop_request_prep.py -q
python -m ruff check app test
git diff --check
```

Expected: targeted tests, ruff, and whitespace check pass.

- [ ] **Step 2: Commit and push**

Commit with a structured Chinese commit message including `背景：`, `改动：`, `验证：`, and `Co-Authored-By: Codex <noreply@anthropic.com>`, then push `fusion-api` master.

- [ ] **Step 3: Track CI/CD**

Use `gh run view/list` to confirm build/test/image push/dev deploy success. Do not start local Fusion services.

- [ ] **Step 4: Real conversation regression**

After deploy, create or reuse a real deployed conversation without opening a new Chrome target. Verify a current-information query produces search results and, when useful, targeted `url_read` calls for high-value sources. Record input, conversation/message id, expected behavior, actual tool logs, and conclusion.
