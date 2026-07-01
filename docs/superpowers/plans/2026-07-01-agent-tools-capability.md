# Agent Tools Capability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Fusion 模型目录和 agent loop 增加 `agentTools` 能力位，避免不适合工具路径的模型被默认当成联网 agent。

**Architecture:** 在 LiteLLM catalog 薄缓存层归一化 capabilities，`/api/models` 透出该字段，agent loop request prep 使用 `functionCalling && agentTools` 作为工具启用条件。eval runner 同步读取该字段，避免把非 agent 模型的 autonomous search 当作工具失败。

**Tech Stack:** FastAPI, LiteLLM metadata, Python unittest/pytest, Fusion stream agent loop.

---

## Files

- Modify: `app/ai/litellm_catalog.py`
  - 增加 `normalize_capabilities()`。
  - 增加短期 agent tool denylist。
- Modify: `app/api/models.py`
  - `/api/models` 输出 `capabilities.agentTools`。
- Modify: `app/services/stream/agent_loop_request_prep.py`
  - 用 `functionCalling && agentTools` 判断是否下发工具。
- Modify: `scripts/model_catalog_eval_baseline.py`
  - 记录 `agent_tools_supported`。
  - 非 agent 模型不再被 expected tool 场景误判。
- Test: `test/test_litellm_catalog.py`
- Test: `test/test_models.py`
- Test: `test/services/stream/test_agent_loop_request_prep.py`
- Test: `test/test_model_catalog_eval_baseline.py`

## Test Matrix

| Case | 输入 | 预期 |
| --- | --- | --- |
| ATC-01 | `functionCalling=true` 且无显式 `agentTools` | 默认 `agentTools=true` |
| ATC-02 | `qwen-vl-max` + `functionCalling=true` | 默认 `agentTools=false` |
| ATC-03 | metadata 显式 `agentTools=false` | 尊重显式 false |
| ATC-04 | `/api/models` card | 输出 `capabilities.agentTools` |
| ATC-05 | `agentTools=false` 的 call config | 不下发 `tools` 和 `tool_choice` |
| ATC-06 | eval autonomous search + `agentTools=false` | 无工具调用不算 mismatch |

## Tasks

- [x] 写 ATC-01 到 ATC-06 失败测试。
- [x] 实现 capabilities 归一化和 API 输出。
- [x] 实现 agent loop 工具启用条件收敛。
- [x] 同步 eval runner 工具期望判定。
- [ ] 跑目标测试、lint、全量后端测试和架构检查。
- [ ] commit + push，监控 CI/CD 和 dev 部署。
- [ ] 部署后验证 `/api/models` 与 qwen-vl autonomous search 行为。
