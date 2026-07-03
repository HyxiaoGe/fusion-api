# Runtime Config 治理闭环 v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 runtime config 增加后端治理闭环，确保配置可查、可校验、可禁用并能安全回退。

**Architecture:** 在 `app.core` 放轻量 schema 校验和主链路读取保护；在 `app.services` 放治理快照、校验、安全写入和安全激活；`app.api.admin` 只做管理员路由和请求模型。

**Tech Stack:** FastAPI、SQLAlchemy ORM、pytest/unittest、现有 `runtime_config_entries` 表。

---

### Task 1: 主链路 schema 校验和坏版本回退

**Files:**
- Create: `app/core/runtime_config_schema.py`
- Modify: `app/core/runtime_config.py`
- Test: `test/test_runtime_config_service.py`

- [x] 写失败测试：最新 active prompt payload 类型错误时跳过，使用上一条有效 active 版本。
- [x] 写失败测试：所有 active model presentation payload 无效时回退代码默认值。
- [x] 写失败测试：schema 校验返回可读问题。
- [x] 实现 `validate_runtime_config_payload()`。
- [x] 修改 `get_runtime_config_payload()`，最多检查 10 条 active 候选并记录 `skipped_versions`。
- [x] 聚焦测试通过。

### Task 2: 治理 service

**Files:**
- Create: `app/services/runtime_config_governance.py`
- Test: `test/test_runtime_config_governance.py`

- [x] 写失败测试：治理快照列出 entries 和 effective version。
- [x] 写失败测试：active 状态切换会更新 row、commit、refresh、清理缓存。
- [x] 实现 `build_runtime_config_snapshot()`。
- [x] 实现 `validate_runtime_config_candidate()`。
- [x] 实现 `set_runtime_config_entry_active()`。
- [x] 聚焦测试通过。

### Task 3: Admin API

**Files:**
- Modify: `app/api/admin.py`
- Test: `test/test_runtime_config_governance_api.py`

- [x] 写失败测试：非管理员访问 runtime config 返回 403。
- [x] 写失败测试：管理员可读取治理快照。
- [x] 写失败测试：管理员可无写入校验 payload。
- [x] 写失败测试：管理员可切换条目 active 状态。
- [x] 实现 `GET /api/admin/runtime-config`。
- [x] 实现 `POST /api/admin/runtime-config/validate`。
- [x] 实现 `PATCH /api/admin/runtime-config/{entry_id}/status`。
- [x] 聚焦测试通过。

### Task 4: 安全写入和安全激活

**Files:**
- Modify: `app/services/runtime_config_governance.py`
- Modify: `app/api/admin.py`
- Test: `test/test_runtime_config_governance.py`
- Test: `test/test_runtime_config_governance_api.py`

- [x] 写失败测试：创建新版本会先校验 payload，通过后写入 inactive 版本。
- [x] 写失败测试：非法 payload 返回 `INVALID_PARAM`，不 add、不 commit。
- [x] 写失败测试：重复 `(namespace,key,version)` 返回 `CONFLICT`，不写入。
- [x] 写失败测试：激活版本时关闭同一 `namespace/key` 的旧 active 版本。
- [x] 写失败测试：坏版本不可激活，旧 active 状态保持不变。
- [x] 写失败测试：旧 `status=true` 入口复用安全激活语义。
- [x] 实现 `create_runtime_config_entry()`。
- [x] 实现 `activate_runtime_config_entry()`。
- [x] 实现 `POST /api/admin/runtime-config`。
- [x] 实现 `POST /api/admin/runtime-config/{entry_id}/activate`。
- [x] 聚焦测试通过。

### Task 5: 验证和发布

**Files:**
- Modify only files above and docs.

- [x] 运行架构检查：`/opt/homebrew/bin/python3.11 scripts/check_architecture.py`
- [x] 运行 lint：`/opt/homebrew/bin/python3.11 -m ruff check .`
- [x] 运行全量测试：`/opt/homebrew/bin/python3.11 -m pytest`
- [x] 检查 git diff，确认未提交无关文件。
- [ ] 中文结构化 commit，包含 `Co-Authored-By: Codex <noreply@anthropic.com>`。
- [ ] push 后监控 GitHub Actions 和 dev 部署门禁。
