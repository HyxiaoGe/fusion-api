# Multi Model Real Acceptance v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把现有多模型 runner 从“文本/搜索质量基线”升级为“按模型能力分层的真实验收矩阵”。

**Architecture:** 继续复用 `scripts/model_catalog_eval_baseline.py` 的独立脚本形态，不改业务 API。场景成为带能力门槛的矩阵项：脚本先从 `/api/models/` 读取能力契约，再决定某个模型是否适合运行该场景；需要图片的场景先通过 `/api/files/upload` 上传内置测试图，再用同一 `conversation_id` 调 `/api/chat/send`。结果 JSONL 增加跳过原因、附件信息和内容断言风险，summary/report 聚合能力覆盖、跳过项和风险。

**Tech Stack:** Python 3.11, httpx, unittest, Fusion `/api/models/`, Fusion `/api/files/upload`, Fusion `/api/chat/send`, SSE envelope.

---

## Files

- Modify: `scripts/model_catalog_eval_baseline.py`
  - 增加场景能力门槛、跳过结果、图片上传、内容断言、能力覆盖 summary/report。
- Modify: `test/test_model_catalog_eval_baseline.py`
  - 增加矩阵适用性、图片上传、跳过项、内容断言和报告测试。

## Requirements

- 默认场景矩阵继续覆盖基础对话、事实问答、代码/推理、自主搜索、不应搜索、长回答。
- 新增读图正向场景：只对 `vision=true` 模型运行，自动上传内置 PNG，要求回答命中图片里的 `FUSION`。
- 新增无读图降级场景：只对 `vision=false` 模型运行，自动上传同一张图，要求模型明确说明无法读取/理解图片，避免臆测。
- 新增长上下文契约场景：只对 `longContext=true` 模型运行，用轻量长上下文任务验证该能力进入验收矩阵；v1 不构造 128k 超大输入，避免成本和供应商波动。
- 场景不适用时生成 `skipped=true` 结果，而不是误记为失败；summary/report 要展示跳过数量和原因。
- dry-run 必须显示每个模型/场景的 `eligible`、`skip_reason`、`required_capabilities` 和 `excluded_capabilities`。
- apply 模式中有附件的场景必须先上传，再把 `file_ids` 传给 `/api/chat/send`；上传失败应归类为失败结果。
- JSONL 不输出 auth token，不输出图片 base64。
- 真实 Chrome 登录态回归仍作为补充记录模板保留；脚本不自动打开 Chrome。

## Test Matrix

| Case | 输入 | 预期 |
| --- | --- | --- |
| MMRA-01 | 默认场景 | 包含 `vision_image_understanding`、`no_vision_image_boundary`、`long_context_contract` |
| MMRA-02 | `vision=false` 模型 + 读图正向场景 | 生成 skipped 结果，原因说明缺少 `vision` |
| MMRA-03 | `vision=true` 模型 + 无读图降级场景 | 生成 skipped 结果，原因说明不满足排除能力 |
| MMRA-04 | dry-run | 每行包含 eligible、skip_reason、附件类型和能力门槛 |
| MMRA-05 | 有图片附件场景 | 先调用 `/api/files/upload`，再调用 `/api/chat/send`，同一 conversation_id 传入 file_ids |
| MMRA-06 | 回答未包含期望关键词 | success 保持 true，但 quality_flags 增加 `expected_answer_missing` |
| MMRA-07 | 无读图降级回答未说明能力边界 | quality_flags 增加 `missing_no_vision_boundary` |
| MMRA-08 | summary | 聚合 skipped_count、by_capability_bucket、scenario_matrix |
| MMRA-09 | report | 展示跳过项、能力覆盖和真实 Chrome 补充记录模板 |

## Tasks

### Task 1: 场景矩阵与跳过语义

- [ ] 写失败测试：默认场景包含读图、无读图降级、长上下文契约。
- [ ] 写失败测试：能力不匹配时生成 skipped 结果。
- [ ] 实现 `required_capabilities`、`excluded_capabilities`、`build_skipped_result()` 和 dry-run 适用性输出。
- [ ] 跑目标测试确认通过。

### Task 2: 图片附件真实链路

- [ ] 写失败测试：图片场景先上传文件，再把 file_ids 传给 chat send。
- [ ] 实现内置 PNG 测试图和 `upload_scenario_files()`。
- [ ] 修改 stream/nonstream 调用支持 file_ids。
- [ ] 跑目标测试确认通过。

### Task 3: 内容断言与能力覆盖报告

- [ ] 写失败测试：答案缺期望关键词时标记 `expected_answer_missing`。
- [ ] 写失败测试：无读图降级答案缺边界说明时标记 `missing_no_vision_boundary`。
- [ ] 扩展 summary/report，展示 skipped、能力 bucket、场景矩阵。
- [ ] 跑目标测试和后端关键测试。

### Task 4: 发布闭环

- [ ] `.venv311/bin/python -m unittest test.test_model_catalog_eval_baseline`
- [ ] `.venv/bin/ruff check scripts/model_catalog_eval_baseline.py test/test_model_catalog_eval_baseline.py`
- [ ] `.venv/bin/ruff format --check scripts/model_catalog_eval_baseline.py test/test_model_catalog_eval_baseline.py`
- [ ] 视改动风险运行后端全量 unittest、ruff、architecture。
- [ ] commit + push，监控 GitHub Actions 和 dev deploy smoke。
- [ ] 若存在可复用已登录 Chrome 标签，再按报告模板补真实 UI 回归；否则记录阻塞。

## Self-Review

- Spec coverage: 自动化矩阵覆盖文本、搜索、读图、无读图降级和长上下文契约；Chrome 回归仍遵守“只复用已有标签”约束。
- Placeholder scan: 无 TBD/TODO。
- Type consistency: `skipped`, `skip_reason`, `required_capabilities`, `excluded_capabilities`, `attachment_kind` 贯穿 dry-run、JSONL、summary 和 report。
