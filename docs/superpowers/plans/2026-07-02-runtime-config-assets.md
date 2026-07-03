# Runtime Config Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 统一迁移 Fusion 的产品/Agent 策略和可运营 Prompt 资产到后端运行时配置层，并让前端模型能力展示消费后端派生结果。

**Architecture:** LiteLLM 继续作为模型目录事实源；Fusion API 新增 `runtime_config_entries` 表和带 fallback 的 core 配置读取服务。Agent/search/read/ranker/prompt/model presentation 都从同一配置层读取默认 profile，UI 优先渲染 `/api/models/` 返回的展示结构。

**Tech Stack:** FastAPI + SQLAlchemy + Alembic + Python unittest；Next.js 15 + React 19 + TypeScript + Vitest。

---

## File Structure

- Create: `fusion-api/app/services/runtime_config_defaults.py`
  - 保存 `DEFAULT_AGENT_STRATEGY_CONFIG`、`DEFAULT_MODEL_PRESENTATION_CONFIG`、`DEFAULT_PROMPT_TEMPLATES`。
- Create: `fusion-api/app/core/runtime_config.py`
  - 读取 `runtime_config_entries`，做 deep merge、缓存和 fallback。
- Create: `fusion-api/app/services/runtime_config_service.py`
  - 兼容旧 import 的 re-export，避免 AI 层反向依赖 service 层。
- Create: `fusion-api/app/services/model_presentation.py`
  - 根据模型 card + 配置生成 `capabilityPresentation`。
- Create: `fusion-api/app/services/agent_strategy_config.py`
  - 提供 Agent/search/read/ranker 统一配置访问函数。
- Modify: `fusion-api/app/db/models.py`
  - 新增 `RuntimeConfigEntry` ORM。
- Create: `fusion-api/alembic/versions/7d2f8a1c9b30_add_runtime_config_entries.py`
  - 建表并 seed 默认配置。
- Modify: `fusion-api/app/api/models.py`
  - `/api/models/` card 增加 `capabilityPresentation`。
- Modify: `fusion-api/app/ai/litellm_catalog.py`
  - 保持 AI 层纯 normalize，支持由 service/API 层注入 agent tools 默认禁用名单。
- Modify: `fusion-api/app/services/search_budget.py`
  - 搜索 budget、intent keywords、阈值读取配置。
- Modify: `fusion-api/app/services/search_read_planner.py`
  - 推荐深读数量和需核验 reason 读取配置。
- Modify: `fusion-api/app/services/source_candidate_ranker.py`
  - 域名分层、权重、优先级阈值读取配置。
- Modify: `fusion-api/app/services/stream/network_budget.py`
  - 搜索/读取上限、repair、recency clamp 读取配置。
- Modify: `fusion-api/app/services/tool_handlers/web_search.py`
  - 上下文注入上限和单域名限制读取配置。
- Modify: `fusion-api/app/services/tool_handlers/url_read.py`
  - 内容截断和 reason 长度读取配置。
- Modify: `fusion-api/app/ai/prompts/prompt_manager.py`
  - Prompt 模板从 runtime config 读取，缺失 fallback。
- Modify: `fusion-api/app/ai/prompts/agent_loop.py`
  - 增加 agent prompt getter，保留常量作为 fallback。
- Modify: `fusion-api/app/ai/tools.py`
  - tool description 使用 getter。
- Modify: `fusion-api/app/services/stream/limit_summary.py`
  - limit summary prompt 使用 getter。
- Modify: `fusion-api/app/services/agent/continuation.py`
  - continuation prompt 使用 getter。
- Modify: `fusion-api/app/services/stream/agent_loop_lifecycle.py`
  - `AgentSession.config` 写入 runtime config 版本。
- Modify: `fusion-ui/src/lib/config/modelConfig.ts`
  - 类型和转换保留 `capabilityPresentation`。
- Modify: `fusion-ui/src/lib/models/modelCapabilityPresentation.ts`
  - 优先使用后端展示字段，保留本地 fallback。
- Modify: `fusion-ui/src/components/models/ModelSelectorPanel.tsx`
  - 使用后端展示字段生成能力分、标签和 tooltip。

## Task 1: Runtime Config DB Layer

**Files:**
- Create: `fusion-api/test/test_runtime_config_service.py`
- Create: `fusion-api/app/services/runtime_config_defaults.py`
- Create: `fusion-api/app/services/runtime_config_service.py`
- Modify: `fusion-api/app/db/models.py`
- Create: `fusion-api/alembic/versions/7d2f8a1c9b30_add_runtime_config_entries.py`

- [ ] **Step 1: Write failing tests**

Cover:

```python
def test_get_runtime_config_payload_returns_default_when_db_unavailable():
    payload, meta = get_runtime_config_payload("agent_strategy", "default", {"a": 1}, session_factory=failing_session)
    assert payload == {"a": 1}
    assert meta["source"] == "default"

def test_get_runtime_config_payload_deep_merges_active_payload():
    db_payload = {"search": {"standard_budget": {"requested_count": 7}}}
    payload, meta = get_runtime_config_payload("agent_strategy", "default", default, session_factory=fake_session(db_payload))
    assert payload["search"]["standard_budget"]["requested_count"] == 7
    assert payload["search"]["standard_budget"]["context_source_limit"] == default["search"]["standard_budget"]["context_source_limit"]
    assert meta["source"] == "db"
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
python -m pytest test/test_runtime_config_service.py -v
```

Expected: import/module failure because service does not exist.

- [ ] **Step 3: Implement minimal DB layer**

Implement:

- `RuntimeConfigEntry` ORM.
- `deep_merge_config(base, override)`.
- `get_runtime_config_payload(namespace, key, default, session_factory=SessionLocal)`.
- `clear_runtime_config_cache()`.
- Alembic table creation and seed rows.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest test/test_runtime_config_service.py -v
```

Expected: PASS.

## Task 2: Backend Model Presentation

**Files:**
- Modify: `fusion-api/test/test_models.py`
- Create: `fusion-api/test/test_model_presentation.py`
- Create: `fusion-api/app/services/model_presentation.py`
- Modify: `fusion-api/app/api/models.py`

- [ ] **Step 1: Write failing tests**

Cover:

```python
def test_entry_to_card_includes_backend_capability_presentation():
    card = _entry_to_card("deepseek-chat", entry)
    presentation = card["capabilityPresentation"]
    assert presentation["score"] >= 70
    assert any(label["text"] == "可联网" for label in presentation["labels"])
    assert "可按问题需要自主联网" in presentation["tooltip"]

def test_model_presentation_uses_configured_weights():
    presentation = build_model_capability_presentation(card, config={"weights": {"base": 10, "network": 80}})
    assert presentation["score"] == 90
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest test/test_models.py test/test_model_presentation.py -v
```

Expected: missing `capabilityPresentation` / missing module failure.

- [ ] **Step 3: Implement presentation service**

Implement backend equivalent of current UI scoring, labels and tooltip, driven by `model_presentation/default`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest test/test_models.py test/test_model_presentation.py -v
```

Expected: PASS.

## Task 3: Agent Strategy Config Consumption

**Files:**
- Create: `fusion-api/test/test_agent_strategy_config.py`
- Modify: `fusion-api/test/test_litellm_catalog.py`
- Modify: `fusion-api/test/services/stream/test_network_budget.py`
- Modify: `fusion-api/test/test_source_candidate_ranker.py`
- Create: `fusion-api/app/services/agent_strategy_config.py`
- Modify: strategy consumer modules listed above.

- [ ] **Step 1: Write failing tests**

Cover:

```python
def test_litellm_catalog_uses_runtime_disabled_agent_tool_aliases():
    with patch_config({"model_runtime": {"agent_tools_disabled_aliases": ["deepseek-chat"]}}):
        capabilities = normalize_capabilities("deepseek-chat", {"functionCalling": True})
        assert capabilities["agentTools"] is False

def test_network_budget_uses_configured_standard_budget():
    with patch_config({"search": {"standard_budget": {"requested_count": 7, "context_source_limit": 6}}}):
        args, degraded = NetworkToolBudget().prepare_web_search_args({"query": "redis"})
        assert degraded is None
        assert args["count"] == 7
        assert args["context_source_limit"] == 6

def test_ranker_uses_configured_authority_domain():
    with patch_config({"source_ranker": {"authority_media_domains": ["example.com"]}}):
        plan = rank_search_sources([...])
        assert "权威媒体" in plan.recommended[0].reasons
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest test/test_litellm_catalog.py test/services/stream/test_network_budget.py test/test_source_candidate_ranker.py -v
```

Expected: new override tests fail because consumers still use hardcoded constants.

- [ ] **Step 3: Implement config consumption**

Move tunable values into `DEFAULT_AGENT_STRATEGY_CONFIG` and replace direct constants with config-backed helper reads. Keep validation conservative and fallback to existing defaults.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest test/test_litellm_catalog.py test/services/stream/test_network_budget.py test/test_source_candidate_ranker.py test/test_agent_strategy_config.py -v
```

Expected: PASS.

## Task 4: Prompt Template Runtime Adapter

**Files:**
- Create: `fusion-api/test/test_prompt_runtime_templates.py`
- Modify: `fusion-api/app/ai/prompts/prompt_manager.py`
- Modify: `fusion-api/app/ai/prompts/agent_loop.py`
- Modify: `fusion-api/app/ai/tools.py`
- Modify: `fusion-api/app/services/stream/limit_summary.py`
- Modify: `fusion-api/app/services/agent/continuation.py`
- Update affected prompt tests.

- [ ] **Step 1: Write failing tests**

Cover:

```python
def test_prompt_manager_uses_runtime_template_override():
    with patch_prompt("generate_title", "标题：{content}"):
        assert prompt_manager.format_prompt("generate_title", content="abc") == "标题：abc"

def test_tool_description_uses_runtime_prompt_override():
    with patch_prompt("url_read_tool_description", "读取网页：{policy}"):
        assert "读取网页" in URL_READ_TOOL["function"]["description"]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest test/test_prompt_runtime_templates.py test/test_ai_tools.py test/services/stream/test_limit_summary.py test/services/agent/test_continuation.py -v
```

Expected: override tests fail.

- [ ] **Step 3: Implement prompt adapter**

Add prompt template getters, keep current constants as fallback, and avoid changing public prompt names.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest test/test_prompt_runtime_templates.py test/test_ai_tools.py test/services/stream/test_limit_summary.py test/services/agent/test_continuation.py -v
```

Expected: PASS.

## Task 5: Frontend Consumes Backend Presentation

**Files:**
- Modify: `fusion-ui/src/lib/config/modelConfig.test.ts`
- Modify: `fusion-ui/src/lib/models/modelCapabilityPresentation.test.ts`
- Modify: `fusion-ui/src/components/models/ModelSelectorPanel.test.tsx`
- Modify: `fusion-ui/src/lib/config/modelConfig.ts`
- Modify: `fusion-ui/src/lib/models/modelCapabilityPresentation.ts`
- Modify: `fusion-ui/src/components/models/ModelSelectorPanel.tsx`

- [ ] **Step 1: Write failing tests**

Cover:

```ts
it('preserves backend capabilityPresentation from /api/models', () => {
  const model = convertApiModelToModelInfo(apiModelWithPresentation);
  expect(model.capabilityPresentation?.score).toBe(88);
});

it('prefers backend capabilityPresentation over local fallback', () => {
  const recommendation = buildModelCapabilityRecommendation(modelWithPresentation);
  expect(recommendation.score).toBe(88);
  expect(recommendation.headline).toBe('后端推荐文案');
});
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cd /Users/sean/code/fusion/fusion-ui
npm test -- src/lib/config/modelConfig.test.ts src/lib/models/modelCapabilityPresentation.test.ts src/components/models/ModelSelectorPanel.test.tsx
```

Expected: missing type/field failure.

- [ ] **Step 3: Implement frontend consumption**

Add `CapabilityPresentation` types, persist API field, and make presentation helpers prefer backend values with fallback.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
npm test -- src/lib/config/modelConfig.test.ts src/lib/models/modelCapabilityPresentation.test.ts src/components/models/ModelSelectorPanel.test.tsx
```

Expected: PASS.

## Task 6: Final Validation

**Files:**
- All touched files.

- [ ] **Step 1: Run API targeted tests**

```bash
cd /Users/sean/code/fusion/fusion-api
python -m pytest test/test_runtime_config_service.py test/test_model_presentation.py test/test_models.py test/test_litellm_catalog.py test/services/stream/test_network_budget.py test/test_source_candidate_ranker.py test/test_prompt_runtime_templates.py test/test_ai_tools.py test/services/stream/test_limit_summary.py test/services/agent/test_continuation.py -v
```

- [ ] **Step 2: Run API static checks**

```bash
ruff check app test
```

- [ ] **Step 3: Run UI targeted tests**

```bash
cd /Users/sean/code/fusion/fusion-ui
npm test -- src/lib/config/modelConfig.test.ts src/lib/models/modelCapabilityPresentation.test.ts src/components/models/ModelSelectorPanel.test.tsx
```

- [ ] **Step 4: Run UI build if targeted tests pass**

```bash
npm run build
```

- [ ] **Step 5: Commit both repos, push once, monitor CI/CD**

Use Chinese structured commit body with `背景：` and `改动：` sections. Do not push docs-only separately.

- [ ] **Step 6: Dev real regression after deployment**

Use only an already-open logged-in Chrome tab. If none exists, record blocker instead of opening a new tab.
