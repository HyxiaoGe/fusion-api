# Model Catalog Audit Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 增加模型目录巡检/同步 v1，输出 LiteLLM、Fusion `/api/models/`、Fusion virtual key allowlist 的一致性报告，并支持显式 apply 同步 allowlist。

**Architecture:** 新增独立脚本 `scripts/audit_litellm_model_catalog.py`，复用 LiteLLM 管理 API 和 Fusion 公开 `/api/models/`，把巡检规则、同步计划和 I/O 分开。v1 只允许同步 virtual key allowlist，不自动注册或删除模型，避免把 provider 官方模型变更误应用到生产。

**Tech Stack:** Python 3.11, httpx, unittest/pytest, LiteLLM Proxy 管理 API, Fusion `/api/models/`。

---

## Files

- Create: `scripts/audit_litellm_model_catalog.py`
  - 拉取 LiteLLM `/model/info`、Fusion `/api/models/`、LiteLLM `/key/info`。
  - 构建巡检报告和 allowlist 同步计划。
  - CLI 默认 dry-run，`--apply` 才调用 `/key/update`。
- Create: `test/test_litellm_model_catalog_audit.py`
  - 覆盖目录一致性、Fusion 多余模型、key 缺失、退役模型、metadata 缺失、序列化脱敏、apply 边界。
- Add: `docs/superpowers/specs/2026-07-01-model-catalog-audit-sync-design.md`
  - 记录 v1 设计边界和验收标准。

## Requirements

- 默认 `--dry-run`，不写 LiteLLM。
- `--apply` 只能同步 virtual key allowlist，不能调用 `/model/new` 或 `/model/delete`。
- 报告包含 `summary`、`issues`、`sync_plan`。
- 报告不能输出 master key、virtual key、provider API key。
- `db_model=true` 是 Fusion 业务模型的唯一来源。
- Fusion `/api/models/` 展示模型必须是 LiteLLM 业务模型子集。
- virtual key allowlist 缺少业务模型时给 `error`，同步计划加入 `add`。
- virtual key allowlist 含已知退役模型时给 `error`，同步计划加入 `remove`。
- 业务模型缺 `provider_key`、`provider_display`、`capabilities` 或 `pricing` 时给 `warning`。

## Test Matrix

| Case | Test |
| --- | --- |
| AUDIT-01 | `test_clean_catalog_has_no_issues` |
| AUDIT-02 | `test_fusion_unknown_model_is_error` |
| AUDIT-03 | `test_key_missing_db_model_is_error_and_sync_adds_it` |
| AUDIT-04 | `test_deprecated_key_model_is_error_and_sync_removes_it` |
| AUDIT-05 | `test_missing_metadata_is_warning` |
| AUDIT-06 | `test_serialize_report_does_not_include_secrets` |
| AUDIT-07 | `test_apply_sync_only_updates_key_models` |

## Tasks

### Task 1: 写失败测试

- [ ] 新建 `test/test_litellm_model_catalog_audit.py`，定义上述 7 个测试。
- [ ] 运行 `.venv311/bin/python -m pytest test/test_litellm_model_catalog_audit.py -q`。
- [ ] 预期失败：`scripts.audit_litellm_model_catalog` 不存在。

### Task 2: 实现巡检核心

- [ ] 新建 `scripts/audit_litellm_model_catalog.py`。
- [ ] 定义 `CatalogIssue`、`SyncPlan`、`AuditReport` dataclass。
- [ ] 实现 `extract_db_models()`、`extract_fusion_model_ids()`、`build_allowlist_sync_plan()`、`audit_catalog()`。
- [ ] 跑目标测试，确认从 import failure 进入行为断言。

### Task 3: 实现 I/O 和 CLI

- [ ] 实现 `fetch_litellm_models()`、`fetch_fusion_models()`、`fetch_key_models()`、`update_key_models()`。
- [ ] 实现 CLI 参数：`--dry-run`、`--apply`、`--litellm-base-url`、`--fusion-base-url`、`--master-key`、`--virtual-key`。
- [ ] `--apply` 无 `--virtual-key` 时抛明确错误。
- [ ] 输出 JSON 使用 `serialize_report()`，不输出任何 secret。

### Task 4: 验证和发布

- [ ] 运行目标测试。
- [ ] 运行全量 `pytest test/ -q`。
- [ ] 运行 `ruff check .` 和 `ruff format --check`。
- [ ] 运行 `scripts/check_architecture.py`。
- [ ] 在生产配置上执行 dry-run，不打印密钥。
- [ ] commit、push，监控 GitHub Actions 和 dev 部署。
