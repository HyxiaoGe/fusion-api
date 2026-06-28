# Agent Loop LLM Stream 拆分 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 拆分 `llm_stream.stream_round()` 的流式协议解析、chunk 写入、usage 收集、tool_call 累积和锁检查职责，保持现有 public API 与 agent-loop 行为不变。

**Architecture:** `stream_round(...)` 保持外部签名不变，内部构造 `LLMStreamRequest` 并调用 `consume_stream_round()`。`consume_stream_round()` 只负责编排；具体逻辑拆到 usage 提取、tool_call 累积、reasoning/content delta 提取、Redis chunk append、锁检查和 outcome 构造 helper。

**Tech Stack:** FastAPI 后端内部 Python 3.11、pytest、ruff、Redis Stream SSE chunk 协议、LiteLLM streaming response。

---

### Task 1: 建立 LLM streaming request/helper 边界

**Files:**
- Create: `test/services/stream/test_llm_stream.py`
- Modify: `app/services/stream/llm_stream.py`

- [x] **Step 1: 写先失败的 helper 接口测试**

新增测试通过 `LLMStreamRequest` 调用 `consume_stream_round(response, request)`，断言：
- reasoning delta 写入 `reasoning` chunk，并透传 `run_id` / `step_id`。
- content delta 写入 `answering` chunk。
- usage-only chunk 仍更新 `Usage`。
- 分片 tool calls 按 index 累积，并按 index 排序返回。
- `finish_reason="tool_calls"` 不写空 content chunk。

- [x] **Step 2: 运行测试确认失败**

Run:
```bash
DATABASE_URL=sqlite:////tmp/fusion_api_llm_stream_red.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_llm_stream.py -q
```
Expected: FAIL，因为 `LLMStreamRequest` 或 `consume_stream_round` 尚未导出。

- [x] **Step 3: 最小实现 request/outcome 和 helper 拆分**

在 `llm_stream.py` 增加：
- `LLMStreamRequest`
- `LLMStreamState`
- `LLMStreamOutcome`
- `consume_stream_round()`
- `extract_usage()`
- `accumulate_tool_calls()`
- `extract_reasoning_delta()`
- `extract_content_delta()`
- `append_stream_delta()`
- `maybe_check_lock_owner()`
- `build_tool_calls_list()`

`stream_round(...)` 只组装 request 并返回 outcome tuple。

- [x] **Step 4: 验证保持行为**

Run:
```bash
DATABASE_URL=sqlite:////tmp/fusion_api_llm_stream_green.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_llm_stream.py test/services/stream/test_agent_round.py test/services/stream/test_limit_summary.py test/services/stream/test_agent_loop_contract.py test/test_stream_handler.py test/test_stream_state_service.py -q
/opt/homebrew/bin/python3.11 scripts/check_quality.py
/opt/homebrew/bin/python3.11 scripts/check_architecture.py
/opt/homebrew/bin/python3.11 -m ruff check app/services/stream/llm_stream.py test/services/stream/test_llm_stream.py
/opt/homebrew/bin/python3.11 -m ruff format --check app/services/stream/llm_stream.py test/services/stream/test_llm_stream.py
```

验收标准：
- `app/services/stream/llm_stream.py:stream_round()` 不再出现在质量扫描的超长函数列表。
- 现有 `stream_round(...)` public 签名不变。
- usage、reasoning/content chunk、tool_calls、lock cancellation 语义不变。
- 不启动本地 Fusion 服务。
