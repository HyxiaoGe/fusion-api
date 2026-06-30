# Search / Read Planner + Evidence Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 Search / Read Planner + Evidence Ledger 最小版，让搜索候选、推荐深读和网页读取状态使用统一 evidence 生命周期。

**Architecture:** 后端新增轻量 evidence ledger helper，生成稳定 URL evidence id，并让 `web_search`、`SourceCandidateRanker`、`url_read` upsert 同一条 evidence。前端只扩展 evidence status 类型和恢复逻辑，保持现有 UI 结构。

**Tech Stack:** FastAPI service layer, Pydantic event schema, Redux/React TypeScript, pytest, Vitest, ruff.

---

### Task 1: 后端 Evidence Ledger Helper

**Files:**
- Create: `app/services/source_evidence_ledger.py`
- Modify: `app/services/agent/progress_digest.py`
- Test: `test/services/agent/test_source_evidence_ledger.py`

- [ ] **Step 1: Write failing backend tests**

新增测试覆盖：

```python
def test_search_and_url_read_share_stable_evidence_id():
    search_item = build_search_source_evidence_item(...)
    read_item = build_url_read_evidence_item(...)
    assert search_item["id"] == read_item["id"]
```

```python
def test_search_candidate_uses_candidate_status():
    item = build_search_source_evidence_item(...)
    assert item["status"] == "candidate"
```

```python
def test_url_read_status_maps_to_read_lifecycle():
    assert success_item["status"] == "read_success"
    assert degraded_item["status"] == "read_degraded"
    assert failed_item["status"] == "read_failed"
```

Run:

```bash
.venv311/bin/python -m pytest test/services/agent/test_source_evidence_ledger.py -q
```

Expected: fail because helper does not exist.

- [ ] **Step 2: Implement helper**

`source_evidence_ledger.py` must provide:

```python
def canonicalize_evidence_url(url: str) -> str:
    ...

def stable_web_evidence_id(url: str, *, fallback: str) -> str:
    ...

def build_search_source_evidence_item(source, *, tool_call_id: str, source_index: int) -> dict:
    ...

def build_url_read_evidence_item(result_data: dict, *, status: str, tool_call_id: str) -> dict | None:
    ...
```

Status mapping:

```python
success -> read_success
degraded -> read_degraded
failed -> read_failed
```

- [ ] **Step 3: Wire `progress_digest.build_evidence_items`**

Replace per-record evidence construction with helper:

- `web_search` returns candidate items from sources.
- `url_read` returns one read lifecycle item when URL exists.
- `tool_result_digest.source_refs` uses stable evidence ids.

- [ ] **Step 4: Verify task**

Run:

```bash
.venv311/bin/python -m pytest test/services/agent/test_source_evidence_ledger.py test/services/agent/test_progress_digest.py -q
```

Expected: pass.

### Task 2: 后端 Source Selection Upsert

**Files:**
- Modify: `app/services/stream/tool_round.py`
- Modify: `app/services/source_evidence_ledger.py`
- Test: `test/services/stream/test_tool_round.py`

- [ ] **Step 1: Write failing tests**

Add test:

```python
async def test_tool_round_emits_selected_evidence_for_ranker_recommendations():
    ...
    await handle_tool_calls_round(...)
    emitter.evidence_item_upserted.assert_any_await(
        tool_call_id="tc-search",
        evidence=ANY,
    )
    assert selected_event["evidence"]["status"] == "selected"
```

Run:

```bash
.venv311/bin/python -m pytest test/services/stream/test_tool_round.py::ToolRoundTests::test_tool_round_emits_selected_evidence_for_ranker_recommendations -q
```

Expected: fail because selected evidence is not emitted.

- [ ] **Step 2: Implement selected evidence builder**

Add helper:

```python
def build_selected_source_evidence_item(candidate: RankedSourceCandidate) -> dict:
    ...
```

Claim format:

```text
建议深读：{reason1} / {reason2}
```

- [ ] **Step 3: Emit selected evidence in tool round**

In `handle_tool_calls_round`, after `execute_tool_round_tools` and before `append_tool_round_messages`, emit selected evidence events for successful search ranker recommendations. Event failure must not break tool main chain; log warning only.

- [ ] **Step 4: Verify task**

Run:

```bash
.venv311/bin/python -m pytest test/services/stream/test_tool_round.py test/test_source_candidate_ranker.py -q
```

Expected: pass.

### Task 3: Agent Event Schema And Snapshot Reducer

**Files:**
- Modify: `app/services/agent/events.py`
- Modify: `app/services/agent/progress_state.py`
- Test: `test/services/agent/test_progress_state.py`
- Test: `test/services/agent/test_events.py`

- [ ] **Step 1: Write failing tests**

Add tests:

```python
def test_progress_state_accepts_read_success_evidence():
    state = apply_progress_event(empty_progress_state(...), evidence_event("read_success"))
    assert state["evidence"][0]["status"] == "read_success"
```

```python
def test_progress_state_caps_candidates_but_keeps_selected_and_read_success():
    ...
    assert "selected-id" in kept_ids
    assert "read-success-id" in kept_ids
```

Run:

```bash
.venv311/bin/python -m pytest test/services/agent/test_progress_state.py test/services/agent/test_events.py -q
```

Expected: fail because new statuses are not accepted/prioritized.

- [ ] **Step 2: Extend schema literals**

`AgentEvidenceItem.status` accepts:

```text
candidate | selected | read_success | read_degraded | read_failed | used | discarded
```

- [ ] **Step 3: Update cap priority**

Keep priority:

```text
used / used_by_final_answer > read_success > selected > read_degraded/read_failed > candidate/discarded
```

- [ ] **Step 4: Verify task**

Run:

```bash
.venv311/bin/python -m pytest test/services/agent/test_progress_state.py test/services/agent/test_events.py -q
```

Expected: pass.

### Task 4: Frontend Status Compatibility

**Files:**
- Modify: `../fusion-ui/src/types/agentRun.ts`
- Modify: `../fusion-ui/src/lib/agent/streamEventHandlers.ts`
- Modify: `../fusion-ui/src/lib/chat/conversationHydration.ts`
- Modify: `../fusion-ui/src/components/chat/agent/executionProcessModel.ts`
- Test: `../fusion-ui/src/redux/slices/streamSlice.test.ts`
- Test: `../fusion-ui/src/lib/chat/conversationHydration.test.ts`
- Test: `../fusion-ui/src/components/chat/agent/executionProcessModel.test.ts`

- [ ] **Step 1: Write failing frontend tests**

Add tests:

```ts
it('接收 selected/read_success evidence status', () => ...)
```

```ts
it('历史恢复保留 read_success evidence status', () => ...)
```

```ts
it('execution process 把 selected/read_success evidence 当作可展示来源', () => ...)
```

Run:

```bash
npm test -- --run src/redux/slices/streamSlice.test.ts src/lib/chat/conversationHydration.test.ts src/components/chat/agent/executionProcessModel.test.ts
```

Expected: fail because types/status assumptions not updated.

- [ ] **Step 2: Extend TS status union**

Update `AgentEvidenceItem.status`.

- [ ] **Step 3: Keep render semantics**

`isRenderableSearchEvidence` continues hiding only `discarded`; selected/read statuses are visible.

- [ ] **Step 4: Verify task**

Run:

```bash
npm test -- --run src/redux/slices/streamSlice.test.ts src/lib/chat/conversationHydration.test.ts src/components/chat/agent/executionProcessModel.test.ts
```

Expected: pass.

### Task 5: Full Validation, Commit, CI/CD, Real Regression

**Files:**
- No new source files expected beyond previous tasks.

- [ ] **Step 1: Backend validation**

Run:

```bash
cd /Users/sean/code/fusion/fusion-api
.venv311/bin/python -m pytest test/ -q
ruff check app test
ruff format --check app/services/source_evidence_ledger.py app/services/agent/progress_digest.py app/services/agent/progress_state.py app/services/agent/events.py app/services/stream/tool_round.py test/services/agent/test_source_evidence_ledger.py test/services/agent/test_progress_state.py test/services/stream/test_tool_round.py
git diff --check
```

- [ ] **Step 2: Frontend validation**

Run:

```bash
cd /Users/sean/code/fusion/fusion-ui
npm test -- --run src/redux/slices/streamSlice.test.ts src/lib/chat/conversationHydration.test.ts src/components/chat/agent/executionProcessModel.test.ts
npm run lint
npm run build
git diff --check
```

- [ ] **Step 3: Commit and push**

Commit backend and frontend separately if both repos changed. Use structured Chinese commit messages with `背景：` / `改动：` / `验证：` and `Co-Authored-By: Codex <noreply@anthropic.com>`.

- [ ] **Step 4: Monitor CI/CD**

Use `gh run list`, `gh run watch --exit-status`, and dev container checks. Do not treat push as completion.

- [ ] **Step 5: Real Chrome regression**

Reuse the existing logged-in Chrome Fusion tab only. Test a new deployed conversation:

```text
请联网检索 OpenAI 2026年6月 GPT-5.6 Sol 最新官方公告，并对照一条权威媒体报道。请优先深读官方公告、官方技术报告和高相关媒体原文，最后说明你选择深读这些来源的理由。
```

Record:

- case id
- input
- conversation URL
- expected
- actual UI execution/evidence summary
- backend tool logs
- evidence lifecycle snapshot
- console errors
- refresh result
- conclusion
