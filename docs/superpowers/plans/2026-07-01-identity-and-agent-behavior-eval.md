# 身份一致性热修与 Agent 行为 Eval 矩阵 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复模型自称错误身份的问题，并把关键 Agent 行为场景固化为可自动化评估矩阵。

**Architecture:** 身份一致性通过全局 system prompt 注入到所有用户对话的 LLM 消息中，避免只在工具模式生效。Agent 行为 eval 使用独立脚本和 fixture 描述样本、期望工具策略、期望 UI surface 与禁止泄露项，既可用于单元测试，也可承接真实 Chrome 回归记录。

**Tech Stack:** FastAPI 后端、LiteLLM 消息构造、Python unittest/pytest、JSON fixture。

---

### Task 1: 身份一致性约束

**Files:**
- Modify: `app/ai/prompts/agent_loop.py`
- Modify: `app/services/chat/message_builder.py`
- Create: `test/services/chat/test_message_builder.py`

- [ ] **Step 1: Write failing tests**
  - `build_llm_messages()` 必须在日期 prompt 后注入 Fusion 身份 prompt。
  - 用户自定义 system prompt 必须排在身份 prompt 后。
  - 身份 prompt 必须禁止自称 Claude / Anthropic / OpenAI / DeepSeek 等单一供应商身份。

- [ ] **Step 2: Run target tests and verify RED**
  - Run: `.venv311/bin/python -m pytest test/services/chat/test_message_builder.py -q`
  - Expected: fail because identity prompt is not injected yet.

- [ ] **Step 3: Implement minimal prompt injection**
  - Add `APP_IDENTITY_PROMPT` constant.
  - Inject it in `build_llm_messages()` immediately after current-date prompt.

- [ ] **Step 4: Run target tests and verify GREEN**
  - Run: `.venv311/bin/python -m pytest test/services/chat/test_message_builder.py -q`
  - Expected: pass.

### Task 2: Agent 行为 Eval 矩阵

**Files:**
- Create: `scripts/agent_behavior_eval.py`
- Create: `test/fixtures/agent_behavior_eval_samples.json`
- Create: `test/test_agent_behavior_eval.py`

- [ ] **Step 1: Write failing tests**
  - loader 拒绝重复 id、缺少必填字段、非法策略。
  - 默认样本至少覆盖：身份问答、简单数学、实时产品功能、搜索失败降级、读取失败跳过、刷新恢复。
  - scorer 能判定 no_search 场景不应有工具调用/执行过程/回答依据。
  - scorer 能判定 search 场景必须有 web_search、搜索关键词、依据来源且不得泄露内部服务名。

- [ ] **Step 2: Run target tests and verify RED**
  - Run: `.venv311/bin/python -m pytest test/test_agent_behavior_eval.py -q`
  - Expected: fail because eval script and fixture do not exist yet.

- [ ] **Step 3: Implement minimal eval script and fixture**
  - Add sample loader and observation scorer.
  - Keep dry-run JSONL output for CI/manual inspection.
  - Do not call external services.

- [ ] **Step 4: Run target tests and verify GREEN**
  - Run: `.venv311/bin/python -m pytest test/test_agent_behavior_eval.py -q`
  - Expected: pass.

### Task 3: 回归验证与交付

**Files:**
- Modify only files above unless tests expose a necessary adjacent change.

- [ ] **Step 1: Run focused tests**
  - `.venv311/bin/python -m pytest test/services/chat/test_message_builder.py test/test_agent_behavior_eval.py test/services/stream/test_agent_loop_request_prep.py test/test_stream_handler.py -q`

- [ ] **Step 2: Run full API verification**
  - `.venv311/bin/python -m pytest -q`
  - `.venv/bin/ruff check .`

- [ ] **Step 3: Commit and push**
  - Structured Chinese commit body with `背景：` / `改动：` and `Co-Authored-By: Codex <noreply@anthropic.com>`.

- [ ] **Step 4: CI/CD and real Chrome regression**
  - Watch GitHub Actions and dev deploy.
  - Reuse existing user-opened `fusion.seanfield.org` Chrome tab only.
  - New real conversations:
    - `你好，你是谁？` must not claim Claude/Anthropic/OpenAI/DeepSeek identity.
    - `1+1等于几？` must not show execution/evidence.
    - `微信A2A互通怎么用？` must autonomously search and show evidence.
