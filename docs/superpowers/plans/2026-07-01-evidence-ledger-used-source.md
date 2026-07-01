# Evidence Ledger 来源使用判定增强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Fusion 能区分“最终回答真正使用的来源”和“搜索/深读候选来源”。

**Architecture:** 后端在最终文本回合完成时运行确定性 used-source matcher，通过现有 `evidence_item_upserted` upsert `status=used` evidence；前端优先用 agent evidence 的 used/candidate 状态派生回答依据模型，历史数据继续 fallback 到 content block source refs。

**Tech Stack:** FastAPI service layer, Pydantic event protocol, pytest, Next.js/React/Vitest, existing AgentRun evidence snapshot.

---

## Task 1: Backend Final Answer Evidence Matcher

**Files:**
- Create: `app/services/final_answer_evidence.py`
- Modify: `app/services/stream/agent_loop_round_outcome.py`
- Test: `test/services/stream/test_final_answer_evidence.py`
- Test: `test/services/stream/test_agent_loop_round_outcome.py`

- [ ] Add failing tests for `[1]`, `⟦2⟧`, exact URL, unique domain, ambiguous domain, single-read fallback, and no-match multi-source.
- [ ] Implement `build_used_final_answer_evidence(content_blocks, answer_text)`.
- [ ] Call matcher after final text blocks are appended and before step/run completion events finish.
- [ ] Emit `evidence_item_upserted` with `status="used"` and `used_by_final_answer=true`.
- [ ] Verify targeted backend tests.

## Task 2: Frontend Evidence Model Split

**Files:**
- Modify: `src/components/chat/answerEvidenceModel.ts`
- Modify: `src/components/chat/useAssistantMessageViewModel.ts`
- Modify: `src/components/chat/answerEvidenceSidebarModel.ts`
- Modify: `src/components/chat/AnswerEvidenceSidebar.tsx`
- Test: `src/components/chat/answerEvidenceModel.test.ts`
- Test: `src/components/chat/answerEvidenceSidebarModel.test.ts`
- Test: `src/components/chat/useAssistantMessageViewModel.test.tsx`

- [ ] Add failing tests showing run evidence with used and candidate sources.
- [ ] Add `usedItems` / `candidateItems` to `AnswerEvidenceModel`.
- [ ] Prefer `currentRun.evidence` when it contains renderable web evidence.
- [ ] Preserve fallback for older messages without evidence.
- [ ] Update sidebar labels to show “已使用来源” and “候选来源”.
- [ ] Verify targeted frontend tests.

## Task 3: Verification And Rollout

**Files:**
- Modify as needed based on integration findings.

- [ ] Run backend targeted tests.
- [ ] Run backend full `pytest` and `ruff`.
- [ ] Run frontend targeted Vitest.
- [ ] Run frontend build or existing required validation for changed TSX/model files.
- [ ] Commit backend and frontend changes with structured Chinese messages.
- [ ] Push both repos and monitor CI/CD.
- [ ] Reuse existing Chrome `fusion.seanfield.org` tab for deployed regression:
  - New realtime search answer with citations.
  - Check answer evidence summary and sidebar used/candidate split.
  - Refresh the conversation and confirm used/candidate state persists.
  - Confirm console has no new errors.
