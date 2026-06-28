# Agent Loop Limit Summary Helpers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 拆分触顶强制总结 step 的内部职责，降低 `run_limit_summary_step()` 的修改风险。

**Architecture:** 用 `LimitSummaryStepRequest` 收敛 driver 到 summary 的内部调用边界。新增模块内 helper 负责 prompt append、summary LLM/stream 调用、timeout fallback、usage 累加、content block append 和 step complete，runner 不参与本轮改动。

**Tech Stack:** Python 3.11, FastAPI service modules, unittest/pytest, ruff.

---

### Task 1: 补齐 helper 行为测试

**Files:**
- Modify: `test/services/stream/test_limit_summary.py`

- [ ] **Step 1: 新增 helper 导入**

```python
from app.services.stream.limit_summary import (
    append_summary_content_blocks,
    accumulate_summary_usage,
)
```

- [ ] **Step 2: 新增 usage 累加测试**

```python
def test_accumulate_summary_usage_adds_usage_data(self):
    result = accumulate_summary_usage(
        Usage(input_tokens=2, output_tokens=3),
        Usage(input_tokens=5, output_tokens=7),
    )

    self.assertEqual(result, Usage(input_tokens=7, output_tokens=10))
```

- [ ] **Step 3: 新增 content block append 测试**

```python
def test_append_summary_content_blocks_adds_reasoning_and_text(self):
    content_blocks = []

    append_summary_content_blocks(
        content_blocks=content_blocks,
        reasoning_buf="推理",
        content_buf="总结正文",
        thinking_block_id="blk-thinking",
        text_block_id="blk-text",
    )

    self.assertEqual([block.type for block in content_blocks], ["thinking", "text"])
```

- [ ] **Step 4: 运行 RED**

Run:

```bash
DATABASE_URL=sqlite:////tmp/fusion_api_limit_summary_red.db .venv311/bin/python -m pytest test/services/stream/test_limit_summary.py -q
```

Expected: import error for missing helper functions.

### Task 2: 拆分 `limit_summary.py`

**Files:**
- Modify: `app/services/stream/limit_summary.py`
- Modify: `app/services/stream/agent_loop_driver.py`

- [ ] **Step 1: 新增 helper**

新增：

```python
@dataclass(frozen=True)
class LimitSummaryStepRequest: ...
def append_limit_summary_prompt(messages: list[dict]) -> None: ...
async def call_limit_summary_round(...) -> tuple[str, str, Usage | None]: ...
def accumulate_summary_usage(accumulated_usage: Usage, usage_data: Usage | None) -> Usage: ...
def append_summary_content_blocks(...) -> None: ...
async def complete_limit_summary_step(...) -> None: ...
```

- [ ] **Step 2: 收敛 `run_limit_summary_step()`**

`run_limit_summary_step(request=...)` 只保留：

1. start step
2. append prompt
3. compute timeout
4. call summary round with timeout fallback
5. accumulate usage
6. append blocks
7. complete step
8. return outcome

- [ ] **Step 3: 保持外部行为不变**

保留：

- `build_limit_summary_call_kwargs()` 复制并移除 `tools/tool_choice`
- timeout warning 文案
- timeout 后不写 log summary、不 append block
- `complete_step_fn(... tool_names=[], tool_call_count=0)`
- driver 只负责把 `AgentLoopRuntime` / `AgentLoopState` 转成 `LimitSummaryStepRequest`

### Task 3: 验证与发布

**Files:**
- No extra code files.

- [ ] **Step 1: 聚焦测试**

```bash
DATABASE_URL=sqlite:////tmp/fusion_api_limit_summary_focus.db .venv311/bin/python -m pytest test/services/stream/test_limit_summary.py test/services/stream/test_agent_loop_driver.py test/test_stream_handler.py -q
```

- [ ] **Step 2: 质量与格式**

```bash
.venv311/bin/python scripts/check_architecture.py
/opt/homebrew/bin/python3.11 -m ruff check .
/opt/homebrew/bin/python3.11 -m ruff format --check app/services/stream/limit_summary.py test/services/stream/test_limit_summary.py
```

- [ ] **Step 3: 全量 unittest**

```bash
DATABASE_URL=sqlite:////tmp/fusion_api_limit_summary_full.db .venv311/bin/python -m unittest discover -s test -t .
```

- [ ] **Step 4: 提交、推送、监控 CI/CD**

Commit:

```bash
git commit -m "refactor: 拆分 agent loop 触顶总结"
```

Push to `master` and watch GitHub Actions plus dev health.
