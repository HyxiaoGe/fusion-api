# Model Capability Contract v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 LiteLLM 模型能力元数据、后端运行约束和前端用户提示使用同一套能力契约。

**Architecture:** `fusion-api` 继续以 LiteLLM `/model/info` metadata 为事实来源，在 `litellm_catalog.normalize_capabilities()` 中派生 Fusion 运行时能力。`searchCapable` 是 v1 新增的产品语义字段，表示 Fusion 会实际下发 `web_search/url_read` 工具；旧 `webSearch` 保留为兼容别名，但由同一派生结果填充。`fusion-ui` 只消费 API 返回的契约字段，不自行猜测工具能力。

**Tech Stack:** Python 3.11, FastAPI, LiteLLM Proxy metadata, unittest, TypeScript, React, Vitest.

---

## Test Matrix

| Case | 层级 | 输入 | 期望 |
| --- | --- | --- | --- |
| MCC-01 | API catalog | `functionCalling=true`，未显式 `agentTools` | `agentTools=true`, `searchCapable=true`, `webSearch=true` |
| MCC-02 | API catalog | `functionCalling=false`, `agentTools=true` | `agentTools=false`, `searchCapable=false`, `webSearch=false` |
| MCC-03 | API catalog | `qwen-vl-max`, `functionCalling=true`, `vision=true` | `vision=true`, `agentTools=false`, `searchCapable=false` |
| MCC-04 | API catalog | metadata 旧字段 `webSearch=true` 但无 agent tools | 不把旧字段误解释为可联网工具 |
| MCC-05 | Runtime | `searchCapable=true` | agent loop 下发 `web_search`, `tool_choice=auto` |
| MCC-06 | Runtime | `searchCapable=false` | 不下发工具，注入无联网边界提示词 |
| MCC-07 | Runtime | `vision=true` | 图片 FileBlock 可注入 LLM |
| MCC-08 | Runtime | `vision=false` | 图片 FileBlock 不注入 LLM |
| MCC-09 | UI | `searchCapable=true` | 模型卡片/按钮展示“可联网” |
| MCC-10 | UI | `searchCapable=false` 且 `functionCalling=true` | 展示“不可联网”，不展示“工具” |
| MCC-11 | UI | `vision=false` | 上传入口和已有文件发送拦截仍提示切换视觉模型 |
| MCC-12 | Deploy smoke | `/api/models/` | 每个模型能力对象包含 `functionCalling`, `agentTools`, `searchCapable`, `webSearch`, `vision` |

## Task 1: Backend Capability Normalization

**Files:**
- Modify: `fusion-api/app/ai/litellm_catalog.py`
- Modify: `fusion-api/app/api/models.py`
- Test: `fusion-api/test/test_litellm_catalog.py`
- Test: `fusion-api/test/test_models.py`
- Test: `fusion-api/test/test_deployment_smoke.py`

- [ ] **Step 1: Write failing tests**
  - Add catalog tests for MCC-01 to MCC-04.
  - Add `/api/models` card test asserting `searchCapable` is present and mirrors runtime tool availability.
  - Add deployment smoke test requiring `searchCapable`.

- [ ] **Step 2: Run backend target tests and confirm RED**
  - Run: `.venv311/bin/python -m unittest test.test_litellm_catalog test.test_models test.test_deployment_smoke`
  - Expected: fails because `searchCapable` is missing and explicit `agentTools=true` can still pass when `functionCalling=false`.

- [ ] **Step 3: Implement normalization**
  - In `normalize_capabilities()`, coerce booleans for `functionCalling`, `vision`, `deepThinking`, `fileSupport`, `imageGen`.
  - Derive `agentTools` as `functionCalling && explicit_or_default_agent_tools`.
  - Derive `searchCapable` as `agentTools`.
  - Set `webSearch` to `searchCapable` for backwards compatibility.
  - Keep denylist behavior for `qwen-vl-max`.

- [ ] **Step 4: Expose contract through `/api/models/`**
  - Add `searchCapable` to capabilities payload.
  - Keep `webSearch` but source it from normalized capabilities.

- [ ] **Step 5: Verify GREEN**
  - Run: `.venv311/bin/python -m unittest test.test_litellm_catalog test.test_models test.test_deployment_smoke`
  - Expected: PASS.

## Task 2: Backend Runtime Contract

**Files:**
- Modify: `fusion-api/app/services/stream/agent_loop_request_prep.py`
- Test: `fusion-api/test/services/stream/test_agent_loop_request_prep.py`
- Test: `fusion-api/test/services/chat/test_message_builder.py`

- [ ] **Step 1: Write failing runtime tests**
  - Add MCC-05: `searchCapable=true` enables `web_search`.
  - Add MCC-06: `searchCapable=false` disables tools even if `functionCalling=true`.
  - Add MCC-07/MCC-08 message builder tests for image injection gated by `has_vision`.

- [ ] **Step 2: Run target tests and confirm RED**
  - Run: `.venv311/bin/python -m unittest test.services.stream.test_agent_loop_request_prep test.services.chat.test_message_builder`
  - Expected: fails where `searchCapable` is not consumed directly.

- [ ] **Step 3: Implement runtime helper**
  - Add a small helper in request prep, `supports_search_tools(capabilities)`, returning `bool(capabilities.get("searchCapable", capabilities.get("agentTools", False))) && functionCalling`.
  - Use the helper for tool enablement.
  - Keep no-tool boundary prompt injection behavior unchanged.

- [ ] **Step 4: Verify GREEN**
  - Run: `.venv311/bin/python -m unittest test.services.stream.test_agent_loop_request_prep test.services.chat.test_message_builder`
  - Expected: PASS.

## Task 3: Frontend Capability Presentation

**Files:**
- Modify: `fusion-ui/src/lib/config/modelConfig.ts`
- Modify: `fusion-ui/src/lib/models/modelCapabilityPresentation.ts`
- Test: `fusion-ui/src/lib/models/modelCapabilityPresentation.test.ts`
- Test: `fusion-ui/src/components/models/ModelSelectorPanel.test.tsx`
- Test: `fusion-ui/src/components/models/ModelSelector.integration.test.tsx`

- [ ] **Step 1: Write failing UI tests**
  - Add `searchCapable` to mock models.
  - Assert `searchCapable=true` shows “可联网”.
  - Assert `webSearch=true` without `searchCapable` no longer makes the model look联网-capable.

- [ ] **Step 2: Run UI target tests and confirm RED**
  - Run: `npm test -- src/lib/models/modelCapabilityPresentation.test.ts src/components/models/ModelSelectorPanel.test.tsx src/components/models/ModelSelector.integration.test.tsx`
  - Expected: fails before UI switches to `searchCapable`.

- [ ] **Step 3: Implement UI contract consumption**
  - Add `searchCapable?: boolean` to `ModelCapability`.
  - Update `supportsAgentTools()` to prefer `searchCapable`, then fallback to `agentTools` only for old payloads.
  - Keep labels and tooltip copy stable.

- [ ] **Step 4: Verify GREEN**
  - Run: `npm test -- src/lib/models/modelCapabilityPresentation.test.ts src/components/models/ModelSelectorPanel.test.tsx src/components/models/ModelSelector.integration.test.tsx`
  - Expected: PASS.

## Task 4: Verification and Release

**Files:**
- Both repos as changed above.

- [ ] **Step 1: Backend verification**
  - Run: `.venv311/bin/python -m unittest test.test_litellm_catalog test.test_models test.test_deployment_smoke test.services.stream.test_agent_loop_request_prep test.services.chat.test_message_builder`
  - Run: `DATABASE_URL=sqlite:///./fusion-test.db .venv311/bin/python -m unittest discover -s test`
  - Run: `.venv/bin/ruff check .`
  - Run: `.venv311/bin/python scripts/check_architecture.py`

- [ ] **Step 2: Frontend verification**
  - Run target Vitest files from Task 3.
  - Run: `npm test`
  - Run: `npm run build`

- [ ] **Step 3: Commit and push**
  - Commit backend and frontend separately with structured Chinese messages.
  - Do not push docs-only; push after code and docs are included.

- [ ] **Step 4: CI/CD and dev smoke**
  - Monitor both GitHub Actions runs.
  - Verify dev `/api/models/` contains `searchCapable` and model picker smoke still passes.

## Self-Review

- Spec coverage: all three requested layers are covered: source normalization, runtime constraints, and user-facing UI presentation.
- Placeholder scan: no TBD/TODO placeholders.
- Type consistency: `searchCapable` is the new canonical user/runtime search field; `webSearch` remains a compatibility alias.
