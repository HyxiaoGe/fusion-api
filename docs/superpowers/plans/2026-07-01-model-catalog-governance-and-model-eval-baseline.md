# Model Catalog Governance And Model Eval Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 Fusion 可选模型目录与 LiteLLM 实际可调用模型不一致的问题，并建立多模型测验基线。

**Architecture:** Fusion 继续只展示 LiteLLM `/model/info` 中 `db_model=true` 的显式业务别名，不把 wildcard 路由直接暴露给前端。新增一个可重复运行的模型目录治理脚本，用声明式清单生成 LiteLLM `/model/new`、`/model/delete` 和 `/key/update` 操作，先替换已退役的小米 V2 模型并同步 Fusion virtual key 白名单，再用独立脚本对 `/api/models` 中健康模型执行统一 smoke/eval。

**Tech Stack:** FastAPI, LiteLLM Proxy 管理 API, Python 3.11, pytest/unittest, httpx, GitHub Actions/CI/CD, 真实生产 Fusion 回归。

---

## Files

- Create: `scripts/govern_litellm_model_catalog.py`  
  声明式治理小米模型目录和 Fusion virtual key 模型白名单，支持 `--dry-run` 和 `--apply`，默认只打印脱敏计划。
- Create: `scripts/model_catalog_eval_baseline.py`  
  对 Fusion `/api/models` 返回的可用模型执行统一非流式 smoke，输出 JSONL 基线。
- Create: `test/test_litellm_model_catalog_governance.py`  
  覆盖 deprecated 模型删除、新模型注册 payload、已有模型跳过、缺 key 报错。
- Create: `test/test_model_catalog_eval_baseline.py`  
  覆盖模型筛选、健康模型选择、结果 JSONL 结构和失败记录。
- Modify: `docs/superpowers/plans/2026-07-01-model-catalog-governance-and-model-eval-baseline.md`  
  记录实施计划和验收矩阵。

## Requirements

- 不改变 `/api/models` 的产品语义：仍只展示 `db_model=true` 的显式业务别名。
- 不暴露 `xiaomi/*` wildcard 到前端模型选择器。
- 下线或删除旧小米 V2 别名：`mimo-v2-flash`, `mimo-v2-pro`。
- 注册新小米文本模型：`mimo-v2.5-pro`, `mimo-v2.5-pro-ultraspeed`。
- 新小米模型 metadata 必须包含 `provider_key=xiaomi`, `provider_display=小米 MiMo`, `source=fusion-governance`, `capabilities`, `pricing`, `knowledge_cutoff`, `recommended_for`。
- 治理脚本默认 dry-run，不触碰生产 LiteLLM；只有显式 `--apply` 才写入。
- 治理脚本 dry-run 不允许打印真实 API key。
- 如果传入 `--virtual-key`，治理脚本必须把 key allowlist 中的旧小米模型替换成 V2.5 模型。
- 多模型测验基线必须从 Fusion `/api/models` 拉当前可选模型，默认只测 health=healthy 的模型。
- 基线脚本必须记录：model_id、provider、输入、是否成功、耗时、错误、回答摘要。

## Test Matrix

| Case | 输入状态 | 预期 |
| --- | --- | --- |
| GOV-01 | LiteLLM 有旧小米模型 UUID | 生成 delete 动作 |
| GOV-02 | LiteLLM 缺新小米模型且有 `XIAOMI_API_KEY` | 生成 create 动作，payload 使用 `openai/<alias>` + 小米 api_base |
| GOV-03 | LiteLLM 已存在新模型 | 不重复注册 |
| GOV-04 | 需要注册新模型但缺 `XIAOMI_API_KEY` | 抛出明确错误 |
| GOV-05 | 旧模型没有 UUID | 不执行 delete，记录 skip |
| GOV-06 | dry-run 输出 create payload | `api_key` 必须脱敏为 `***` |
| GOV-07 | Fusion virtual key allowlist 含旧小米 | 删除旧小米，补入两个 V2.5 模型 |
| EVAL-01 | `/api/models` 有 healthy/unhealthy 混合模型 | 默认只选 healthy |
| EVAL-02 | 模型调用成功 | 输出 success JSONL，含耗时和回答摘要 |
| EVAL-03 | 模型调用失败 | 输出 failure JSONL，含错误类型和错误消息 |

## Task 1: 写治理脚本测试

- [ ] **Step 1: Add failing tests**

Create `test/test_litellm_model_catalog_governance.py` with tests for:

```python
import unittest

from scripts import govern_litellm_model_catalog as catalog


class ModelCatalogGovernanceTests(unittest.TestCase):
    def test_plan_deletes_deprecated_xiaomi_models(self):
        entries = [
            {
                "model_name": "mimo-v2-pro",
                "model_info": {
                    "id": "uuid-old-pro",
                    "metadata": {"provider_key": "xiaomi", "source": "fusion-migration"},
                },
            }
        ]

        plan = catalog.build_governance_plan(entries, {"XIAOMI_API_KEY": "sk-xiaomi"})

        delete_actions = [a for a in plan.actions if a.action == "delete"]
        self.assertEqual(len(delete_actions), 1)
        self.assertEqual(delete_actions[0].model_name, "mimo-v2-pro")
        self.assertEqual(delete_actions[0].model_uuid, "uuid-old-pro")

    def test_plan_registers_missing_xiaomi_v25_models(self):
        plan = catalog.build_governance_plan([], {"XIAOMI_API_KEY": "sk-xiaomi"})

        create_actions = [a for a in plan.actions if a.action == "create"]
        self.assertEqual([a.model_name for a in create_actions], ["mimo-v2.5-pro", "mimo-v2.5-pro-ultraspeed"])
        payload = create_actions[0].payload
        self.assertEqual(payload["model_name"], "mimo-v2.5-pro")
        self.assertEqual(payload["litellm_params"]["model"], "openai/mimo-v2.5-pro")
        self.assertEqual(payload["litellm_params"]["api_base"], "https://api.xiaomimimo.com/v1")
        self.assertEqual(payload["litellm_params"]["api_key"], "sk-xiaomi")
        self.assertEqual(payload["model_info"]["metadata"]["provider_key"], "xiaomi")
        self.assertEqual(payload["model_info"]["metadata"]["source"], "fusion-governance")

    def test_plan_skips_existing_xiaomi_v25_models(self):
        entries = [
            {"model_name": "mimo-v2.5-pro", "model_info": {"id": "uuid-new-pro", "metadata": {"provider_key": "xiaomi"}}},
            {
                "model_name": "mimo-v2.5-pro-ultraspeed",
                "model_info": {"id": "uuid-new-fast", "metadata": {"provider_key": "xiaomi"}},
            },
        ]

        plan = catalog.build_governance_plan(entries, {"XIAOMI_API_KEY": "sk-xiaomi"})

        self.assertFalse([a for a in plan.actions if a.action == "create"])

    def test_missing_xiaomi_key_is_clear_when_registration_needed(self):
        with self.assertRaisesRegex(RuntimeError, "XIAOMI_API_KEY"):
            catalog.build_governance_plan([], {})

    def test_serialized_plan_redacts_api_key(self):
        plan = catalog.build_governance_plan([], {"XIAOMI_API_KEY": "sk-secret-xiaomi"})
        action = [action for action in plan.actions if action.action == "create"][0]

        serialized = catalog.serialize_action(action)

        self.assertEqual(serialized["payload"]["litellm_params"]["api_key"], "***")
        self.assertEqual(action.payload["litellm_params"]["api_key"], "sk-secret-xiaomi")

    def test_replace_deprecated_xiaomi_models_in_key_allowlist(self):
        models = ["deepseek-chat", "mimo-v2-flash", "mimo-v2-pro", "qwen-max-latest"]

        updated = catalog.replace_deprecated_models_in_allowlist(models)

        self.assertEqual(
            updated,
            ["deepseek-chat", "qwen-max-latest", "mimo-v2.5-pro", "mimo-v2.5-pro-ultraspeed"],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run red test**

Run:

```bash
.venv311/bin/python -m pytest test/test_litellm_model_catalog_governance.py -q
```

Expected: fails because `scripts/govern_litellm_model_catalog.py` does not exist.

## Task 2: 实现治理脚本

- [ ] **Step 1: Create `scripts/govern_litellm_model_catalog.py`**

Implement:

- `TARGET_MODELS`
- `DEPRECATED_MODELS`
- `CatalogAction`
- `CatalogPlan`
- `build_governance_plan(entries, env)`
- `fetch_existing_models(base_url, key)`
- `apply_plan(base_url, key, plan)`
- `replace_deprecated_models_in_allowlist(models)`
- `fetch_key_models(base_url, master_key, virtual_key)`
- `update_key_models(base_url, master_key, virtual_key, models)`
- CLI with `--dry-run` default and `--apply`

- [ ] **Step 2: Run governance tests**

Run:

```bash
.venv311/bin/python -m pytest test/test_litellm_model_catalog_governance.py -q
```

Expected: all tests pass.

## Task 3: 写多模型测验基线测试

- [ ] **Step 1: Add failing tests**

Create `test/test_model_catalog_eval_baseline.py` with tests for healthy filtering and result serialization.

- [ ] **Step 2: Run red test**

Run:

```bash
.venv311/bin/python -m pytest test/test_model_catalog_eval_baseline.py -q
```

Expected: fails because `scripts/model_catalog_eval_baseline.py` does not exist.

## Task 4: 实现多模型测验基线脚本

- [ ] **Step 1: Create `scripts/model_catalog_eval_baseline.py`**

Implement:

- fetch Fusion `/api/models`
- filter healthy models by default
- call Fusion `/api/chat/send` non-streaming with a configurable bearer token
- write JSONL results
- support `--dry-run`, `--models`, `--include-unhealthy`, `--question`

- [ ] **Step 2: Run eval baseline tests**

Run:

```bash
.venv311/bin/python -m pytest test/test_model_catalog_eval_baseline.py -q
```

Expected: all tests pass.

## Task 5: 验证、部署、真实回归

- [ ] **Step 1: Run targeted tests**

```bash
.venv311/bin/python -m pytest test/test_litellm_model_catalog_governance.py test/test_model_catalog_eval_baseline.py -q
```

- [ ] **Step 2: Run lint/format checks**

```bash
.venv/bin/python -m ruff check scripts/govern_litellm_model_catalog.py scripts/model_catalog_eval_baseline.py test/test_litellm_model_catalog_governance.py test/test_model_catalog_eval_baseline.py
.venv/bin/python -m ruff format --check scripts/govern_litellm_model_catalog.py scripts/model_catalog_eval_baseline.py test/test_litellm_model_catalog_governance.py test/test_model_catalog_eval_baseline.py
```

- [ ] **Step 3: Dry-run production governance plan**

```bash
LITELLM_BASE_URL=<dev/prod-litellm-url> LITELLM_MASTER_KEY=<key> XIAOMI_API_KEY=<key> LITELLM_VIRTUAL_KEY=<fusion-key> .venv311/bin/python scripts/govern_litellm_model_catalog.py --dry-run --virtual-key "$LITELLM_VIRTUAL_KEY"
```

Expected: prints delete actions for old V2 models, create actions for missing V2.5 models, and a key allowlist update summary. Printed payload must redact `api_key`.

- [ ] **Step 4: Apply after dry-run review**

```bash
LITELLM_BASE_URL=<dev/prod-litellm-url> LITELLM_MASTER_KEY=<key> XIAOMI_API_KEY=<key> LITELLM_VIRTUAL_KEY=<fusion-key> .venv311/bin/python scripts/govern_litellm_model_catalog.py --apply --virtual-key "$LITELLM_VIRTUAL_KEY"
```

- [ ] **Step 5: Verify `/api/models`**

Expected:

- `mimo-v2-flash` absent
- `mimo-v2-pro` absent
- `mimo-v2.5-pro` present
- `mimo-v2.5-pro-ultraspeed` present

- [ ] **Step 6: Run baseline smoke**

Run against deployed Fusion with a real auth token and record JSONL output.
