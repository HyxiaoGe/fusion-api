# Agent Loop URL 预处理拆分 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 拆分 `preprocess_url_in_message()` 的 URL 提取、策略处理、reader 调用、fallback tool 注入和成功结果构建职责，保持 URL 路径 A 的外部行为不变。

**Architecture:** `preprocess_url_in_message(...)` 保持返回三元组 `(UrlBlock | None, context_msg | None, detected_url | None)`。内部改为小型 helper：`extract_first_url()`、`ensure_url_read_tool()`、`read_url_for_context()`、`build_url_context_message()`、`build_url_read_block()`、`remove_disabled_thinking()`；runner/request prep 调用合同不变。

**Tech Stack:** Python 3.11、pytest、ruff、Fusion agent loop、reader-service、URL 安全策略。

---

### Task 1: 拆分 URL 路径 A 预处理 helper

**Files:**
- Create: `test/services/stream/test_url_preprocess.py`
- Modify: `app/services/stream/persistence.py`

- [x] **Step 1: 写先失败的 helper 边界测试**

新增测试覆盖：
- `extract_first_url(message)` 只返回第一个 http/https URL，无 URL 时返回 `None`。
- `ensure_url_read_tool(call_kwargs)` 追加 `URL_READ_TOOL` 且重复调用不重复追加。
- `build_url_context_message(...)` 生成 `role="user"` 的不可信 web context，使用 reader URL 或 normalized URL 兜底。
- `build_url_read_block(...)` 生成 `UrlBlock` 并沿用 reader 返回的 title/favicon。
- `remove_disabled_thinking(call_kwargs)` 只删除 `extra_body.thinking.type == "disabled"` 的兼容项。

- [x] **Step 2: 运行测试确认失败**

Run:
```bash
DATABASE_URL=sqlite:////tmp/fusion_api_url_preprocess_red.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_url_preprocess.py -q
```
Expected: FAIL，原因是 helper 尚未导出。

- [x] **Step 3: 最小实现 helper 拆分**

在 `persistence.py` 中：
- 提升 URL 正则为模块常量。
- 将 `_append_url_read_tool()` 改为 `ensure_url_read_tool()`。
- 新增 reader URL 解析、reader 调用、context message、UrlBlock 和 thinking 清理 helper。
- 保持 `preprocess_url_in_message(...)` 行为：不支持 FC 直接跳过；无 URL/策略拒绝/reader 失败都追加 URL_READ_TOOL；成功时插入 user web context、返回 UrlBlock、删除 volcengine disabled thinking。

- [x] **Step 4: 验证保持行为**

Run:
```bash
DATABASE_URL=sqlite:////tmp/fusion_api_url_preprocess_green.db /Users/sean/code/fusion/fusion-api/.venv311/bin/python -m pytest test/services/stream/test_url_preprocess.py test/test_stream_handler.py::UrlPreprocessTests test/services/stream/test_agent_loop_request_prep.py test/test_stream_handler.py -q
/opt/homebrew/bin/python3.11 -m ruff check app/services/stream/persistence.py test/services/stream/test_url_preprocess.py
/opt/homebrew/bin/python3.11 -m ruff format --check app/services/stream/persistence.py test/services/stream/test_url_preprocess.py
/opt/homebrew/bin/python3.11 scripts/check_quality.py
```

验收标准：
- `app/services/stream/persistence.py:preprocess_url_in_message()` 不再出现在质量扫描超长函数列表。
- URL 路径 A 成功、拒绝、无 URL、reader 失败的外部三元组和 call_kwargs 语义不变。
- 不启动本地 Fusion 服务。
