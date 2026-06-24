# 动态联网搜索与读取预算 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `web_search` 支持模型动态指定候选数量和搜索意图，同时用后端预算限制 search/read 次数，并让诊断面板展示关键联网参数。

**Architecture:** `fusion-api` 负责工具 schema、参数归一化、单轮联网预算和诊断字段；`search-service` 已支持 count/domain/freshness，第一版只由 `fusion-api` 适配调用参数；`fusion-ui` 只消费后端诊断字段并展示，不新增用户设置。

**Tech Stack:** FastAPI, Pydantic, Python unittest, Next.js, React, TypeScript, Vitest.

---

## 文件结构

- `fusion-api/app/ai/tools.py`：扩展 `web_search` 和 `url_read` 工具 schema。
- `fusion-api/app/services/external/search_client.py`：扩展 search-service 请求参数。
- `fusion-api/app/services/stream/network_budget.py`：新增单轮联网预算和参数归一化 helper。
- `fusion-api/app/services/stream/tool_executor.py`：在执行工具前应用预算和归一化参数。
- `fusion-api/app/services/stream/runner.py`：每个 assistant run 创建一个 `NetworkToolBudget`。
- `fusion-api/app/services/tool_handlers/web_search.py`：使用归一化参数，记录 metadata，限制注入 LLM 的搜索摘要数量。
- `fusion-api/app/services/tool_handlers/url_read.py`：保留并截断 `reason`，写入 block 和日志数据。
- `fusion-api/app/schemas/chat.py`：为 `SearchBlock`、`UrlBlock` 增加联网 metadata 字段。
- `fusion-api/app/schemas/network_diagnostics.py`：为诊断工具项增加 count、intent、domains、recency_days、context_count、budget_limited。
- `fusion-api/app/services/network_diagnostics_service.py`：从工具日志 input/output 派生新诊断字段。
- `fusion-ui/src/types/conversation.ts`：补齐 SearchBlock/UrlBlock metadata 类型。
- `fusion-ui/src/types/networkDiagnostics.ts`：补齐诊断字段类型。
- `fusion-ui/src/components/chat/networkDiagnosticsModel.ts`：把新字段派生成展示文案。
- `fusion-ui/src/components/chat/NetworkDiagnosticsPanel.tsx`：展示 count、intent、使用数、reason、触顶信息。

## Task 1: 后端工具 schema 与参数归一化

**Files:**
- Modify: `fusion-api/app/ai/tools.py`
- Create: `fusion-api/app/services/stream/network_budget.py`
- Test: `fusion-api/test/test_ai_tools.py`
- Test: `fusion-api/test/services/stream/test_network_budget.py`

- [ ] **Step 1: Write failing schema tests**

Add tests that assert `web_search` exposes optional `count`、`intent`、`domains`、`recency_days`, and `url_read` exposes optional `reason`.

Run: `cd fusion-api && python3 -m unittest test.test_ai_tools -v`

Expected: FAIL because those schema fields are absent.

- [ ] **Step 2: Write failing budget tests**

Create `fusion-api/test/services/stream/test_network_budget.py` covering:

- `count` defaults to 5.
- `count` clamps to 3..10.
- unsupported `intent` is dropped.
- `domains` keeps at most 5 valid domains.
- `recency_days` clamps to 1..365.
- more than 3 search calls returns a degraded `ToolResult`.
- more than 5 url reads returns a degraded `ToolResult`.

Run: `cd fusion-api && python3 -m unittest test.services.stream.test_network_budget -v`

Expected: FAIL because `NetworkToolBudget` does not exist.

- [ ] **Step 3: Implement minimal schema and budget helper**

Add `NetworkToolBudget` with methods:

- `prepare_web_search_args(args: dict) -> tuple[dict, ToolResult | None]`
- `prepare_url_read_args(args: dict) -> tuple[dict, ToolResult | None]`

The helper should return normalized args and `None` when execution may continue; when budget is exhausted, return original-ish args and a degraded `ToolResult`.

- [ ] **Step 4: Verify Task 1 tests**

Run:

```bash
cd fusion-api
python3 -m unittest test.test_ai_tools test.services.stream.test_network_budget -v
```

Expected: PASS.

## Task 2: 后端工具执行、search-service 参数透传和上下文裁剪

**Files:**
- Modify: `fusion-api/app/services/external/search_client.py`
- Modify: `fusion-api/app/services/stream/tool_executor.py`
- Modify: `fusion-api/app/services/stream/runner.py`
- Modify: `fusion-api/app/services/tool_handlers/web_search.py`
- Modify: `fusion-api/app/services/tool_handlers/url_read.py`
- Modify: `fusion-api/app/schemas/chat.py`
- Test: `fusion-api/test/test_search_client.py`
- Test: `fusion-api/test/test_tool_executor.py`
- Test: `fusion-api/test/test_tool_handlers.py`

- [ ] **Step 1: Write failing search client tests**

Extend `test_search_client.py` so `search_web("q", count=8, domains=["openai.com"], recency_days=30)` posts:

- `count: 8`
- `domain_filters: ["openai.com"]`
- `freshness: "pm"` for 30 days

Run: `cd fusion-api && python3 -m unittest test.test_search_client -v`

Expected: FAIL because the function only accepts `query, count` and always sends `freshness: "pw"`.

- [ ] **Step 2: Write failing executor budget tests**

Extend `test_tool_executor.py` with a budget object and assert:

- handler receives normalized args, not raw args.
- handler.log records normalized `input_params`.
- a fourth `web_search` returns degraded without invoking handler.execute.

Run: `cd fusion-api && python3 -m unittest test.test_tool_executor -v`

Expected: FAIL because `execute_tools_parallel` has no budget parameter.

- [ ] **Step 3: Write failing handler metadata tests**

Extend `test_tool_handlers.py` to assert:

- `WebSearchHandler.execute()` passes normalized count/domains/recency to `search_web`.
- `SearchBlock` includes `intent`、`requested_count`、`actual_count`、`context_source_count`、`budget_limited`。
- `format_llm_context()` includes at most 8 search sources.
- `UrlReadHandler.execute()` stores truncated `reason` in result data.
- `UrlBlock` includes `reason`。

Run: `cd fusion-api && python3 -m unittest test.test_tool_handlers -v`

Expected: FAIL because these fields are missing.

- [ ] **Step 4: Implement search client and executor integration**

Update `search_web()` signature to:

```python
async def search_web(
    query: str,
    count: int = 5,
    *,
    domains: list[str] | None = None,
    recency_days: int | None = None,
) -> list[SearchSource]:
```

Map `recency_days` to search-service freshness:

- `<= 1`: `pd`
- `<= 7`: `pw`
- `<= 31`: `pm`
- otherwise: `py`

Pass `domain_filters` only when domains are non-empty.

Update `execute_tools_parallel(..., network_budget: NetworkToolBudget | None = None)` so budget is applied after JSON parsing and before emitting `tool_call_started`.

- [ ] **Step 5: Implement handler metadata and context limits**

`WebSearchHandler.execute()` should read normalized `count`、`intent`、`domains`、`recency_days` from args, call `search_web()`, and write metadata into `result.data`.

`WebSearchHandler.format_llm_context()` should use `context_sources = sources[:8]` and mention when search results were truncated for context.

`UrlReadHandler.execute()` should normalize `reason` to max 160 chars and keep it in result data; `build_content_block()` should copy it to `UrlBlock`.

- [ ] **Step 6: Verify Task 2 tests**

Run:

```bash
cd fusion-api
python3 -m unittest test.test_search_client test.test_tool_executor test.test_tool_handlers -v
```

Expected: PASS.

## Task 3: 后端联网诊断字段

**Files:**
- Modify: `fusion-api/app/schemas/network_diagnostics.py`
- Modify: `fusion-api/app/services/network_diagnostics_service.py`
- Test: `fusion-api/test/test_network_diagnostics.py`
- Test: `fusion-api/test/test_network_diagnostics_api.py`

- [ ] **Step 1: Write failing diagnostics tests**

Extend diagnostics tests so a `web_search` ToolCallLog with input/output metadata returns:

- `requested_count`
- `actual_count`
- `context_count`
- `intent`
- `domains`
- `recency_days`
- `budget_limited`

Extend url_read diagnostics so input `reason` is returned as `reason` for successful reads.

Run:

```bash
cd fusion-api
python3 -m unittest test.test_network_diagnostics test.test_network_diagnostics_api -v
```

Expected: FAIL because schema and service do not expose those fields.

- [ ] **Step 2: Implement diagnostics mapping**

Add optional fields to `NetworkDiagnosticsToolItem` and derive them from `log.input_params` plus `log.output_data`.

For `url_read`, prefer explicit input `reason` when `log.error_message` is empty.

- [ ] **Step 3: Verify Task 3 tests**

Run:

```bash
cd fusion-api
python3 -m unittest test.test_network_diagnostics test.test_network_diagnostics_api -v
```

Expected: PASS.

## Task 4: 前端类型和诊断展示

**Files:**
- Modify: `fusion-ui/src/types/conversation.ts`
- Modify: `fusion-ui/src/types/networkDiagnostics.ts`
- Modify: `fusion-ui/src/components/chat/networkDiagnosticsModel.ts`
- Modify: `fusion-ui/src/components/chat/NetworkDiagnosticsPanel.tsx`
- Test: `fusion-ui/src/components/chat/networkDiagnosticsModel.test.ts`
- Test: `fusion-ui/src/components/chat/NetworkDiagnosticsPanel.test.tsx`

- [ ] **Step 1: Write failing frontend model tests**

Add tests asserting diagnostics process items include readable details:

- `intent: comparison`
- `请求 8 条`
- `返回 7 条`
- `用于上下文 6 条`
- `读取原因：需要核实官方原文细节`
- budget limited item shows `已达联网预算`

Run:

```bash
cd fusion-ui
npm test -- src/components/chat/networkDiagnosticsModel.test.ts src/components/chat/NetworkDiagnosticsPanel.test.tsx
```

Expected: FAIL because model and panel do not expose these details.

- [ ] **Step 2: Implement frontend model and panel**

Add optional fields to TS types and derive `detailParts: string[]` on process items. Render those parts as compact secondary text/chips under each process item.

- [ ] **Step 3: Verify Task 4 tests**

Run:

```bash
cd fusion-ui
npm test -- src/components/chat/networkDiagnosticsModel.test.ts src/components/chat/NetworkDiagnosticsPanel.test.tsx
```

Expected: PASS.

## Task 5: Full verification and commits

**Files:**
- All modified files from Tasks 1-4.

- [ ] **Step 1: Run backend targeted tests**

```bash
cd fusion-api
python3 -m unittest \
  test.test_ai_tools \
  test.services.stream.test_network_budget \
  test.test_search_client \
  test.test_tool_executor \
  test.test_tool_handlers \
  test.test_network_diagnostics \
  test.test_network_diagnostics_api \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run frontend targeted tests**

```bash
cd fusion-ui
npm test -- \
  src/components/chat/networkDiagnosticsModel.test.ts \
  src/components/chat/NetworkDiagnosticsPanel.test.tsx
```

Expected: PASS.

- [ ] **Step 3: Run broader checks**

```bash
cd fusion-api && python3 -m unittest discover test -v
cd fusion-ui && npm test
```

Expected: PASS.

- [ ] **Step 4: Commit per repo**

Use Chinese commit messages and include:

```text
Co-Authored-By: Codex <noreply@anthropic.com>
```

Suggested commits:

- `feat: 支持动态联网搜索预算`
- `feat: 展示联网搜索预算诊断`

