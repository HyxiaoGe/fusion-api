import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import httpx


class PromptHubMigrationScriptTests(unittest.TestCase):
    def test_loads_exact_effective_runtime_prompts_with_hashes(self):
        from scripts.migrate_prompts_to_prompthub import load_effective_prompt_manifest

        manifest = load_effective_prompt_manifest(
            runtime_loader=lambda namespace, key, default, **kwargs: (
                {"template": f"effective:{key}"},
                {"source": "db", "version": "v1"},
            )
        )

        self.assertEqual(len(manifest), 11)
        title = next(item for item in manifest if item.key == "generate_title")
        self.assertEqual(title.slug, "generate-title")
        self.assertEqual(title.variables, ("content",))
        self.assertEqual(
            title.content_sha256,
            hashlib.sha256(b"effective:generate_title").hexdigest(),
        )

    def test_dry_run_plans_creation_without_writes(self):
        from scripts.migrate_prompts_to_prompthub import (
            MigrationPrompt,
            PromptHubAdminClient,
            migrate_prompts,
        )

        methods = []

        def handler(request: httpx.Request) -> httpx.Response:
            methods.append(request.method)
            return httpx.Response(200, json={"code": 0, "message": "success", "data": [], "meta": {}})

        prompt = MigrationPrompt(
            key="generate_title",
            slug="generate-title",
            name="生成会话标题",
            content="标题：{content}",
            variables=("content",),
            content_sha256=hashlib.sha256("标题：{content}".encode()).hexdigest(),
            source="db",
            source_version="v1",
        )
        client = PromptHubAdminClient(
            base_url="http://prompthub.local",
            api_key="admin-secret",
            transport=httpx.MockTransport(handler),
        )

        result = migrate_prompts(client, [prompt], apply=False)

        self.assertTrue(result["create_project"])
        self.assertEqual(result["prompts"][0]["action"], "create")
        self.assertEqual(methods, ["GET"])
        self.assertNotIn("admin-secret", str(result))

    def test_apply_creates_project_and_prompt_and_verifies_hash(self):
        from scripts.migrate_prompts_to_prompthub import (
            MigrationPrompt,
            PromptHubAdminClient,
            migrate_prompts,
        )

        content = "标题：{content}"
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path, request.headers.get("Authorization")))
            if request.method == "GET" and request.url.path == "/api/v1/projects":
                return httpx.Response(200, json={"code": 0, "message": "success", "data": [], "meta": {}})
            if request.method == "POST" and request.url.path == "/api/v1/projects":
                return httpx.Response(
                    200,
                    json={"code": 0, "message": "success", "data": {"id": "project-1", "slug": "fusion"}},
                )
            if request.method == "GET" and request.url.path == "/api/v1/projects/project-1/prompts":
                return httpx.Response(200, json={"code": 0, "message": "success", "data": [], "meta": {}})
            if request.method == "POST" and request.url.path == "/api/v1/prompts":
                payload = __import__("json").loads(request.content)
                self.assertEqual(payload["template_engine"], "none")
                self.assertEqual(payload["tags"], ["fusion", "runtime-config"])
                self.assertFalse(payload["is_shared"])
                return httpx.Response(
                    200,
                    json={
                        "code": 0,
                        "message": "success",
                        "data": {
                            "id": "prompt-1",
                            "current_version": "1.0.0",
                            "content": payload["content"],
                            "variables": payload["variables"],
                        },
                    },
                )
            raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

        prompt = MigrationPrompt(
            key="generate_title",
            slug="generate-title",
            name="生成会话标题",
            content=content,
            variables=("content",),
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            source="db",
            source_version="v1",
        )
        client = PromptHubAdminClient(
            base_url="http://prompthub.local",
            api_key="admin-secret",
            transport=httpx.MockTransport(handler),
        )

        result = migrate_prompts(client, [prompt], apply=True)

        self.assertEqual(result["prompts"][0]["status"], "verified")
        self.assertEqual(result["prompts"][0]["remote_sha256"], prompt.content_sha256)
        self.assertTrue(all(auth == "Bearer admin-secret" for _, _, auth in requests))

    def test_existing_prompt_uses_published_rendering_snapshot_to_decide_patch(self):
        from scripts.migrate_prompts_to_prompthub import MigrationPrompt, migrate_prompts

        content = "标题：{content}"
        prompt = MigrationPrompt(
            key="generate_title",
            slug="generate-title",
            name="生成会话标题",
            content=content,
            variables=("content",),
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            source="db",
            source_version="v1",
        )
        client = SimpleNamespace(
            list_projects=lambda: [{"id": "project-1", "slug": "fusion"}],
            list_project_prompts=lambda project_id: [{"id": "prompt-1", "slug": "generate-title"}],
            get_prompt=lambda prompt_id: {
                "id": prompt_id,
                "current_version": "1.0.0",
                "format": "text",
                "template_engine": "none",
                "tags": ["fusion", "runtime-config"],
                "is_shared": False,
            },
            get_version=lambda prompt_id, version: {
                "content": content,
                "variables": [{"name": "content"}],
                "status": "published",
                "format": "text",
                "template_engine": "jinja2",
            },
            update_prompt=Mock(),
            publish_patch=Mock(return_value={"content": content}),
        )

        result = migrate_prompts(client, [prompt], apply=True)

        self.assertEqual(result["prompts"][0]["action"], "publish_patch")
        client.update_prompt.assert_called_once()
        client.publish_patch.assert_called_once()

    def test_rejects_unexpected_project_prompt_slug_before_writing(self):
        from scripts.migrate_prompts_to_prompthub import MigrationPrompt, migrate_prompts

        prompt = MigrationPrompt(
            key="generate_title",
            slug="generate-title",
            name="生成会话标题",
            content="标题：{content}",
            variables=("content",),
            content_sha256=hashlib.sha256("标题：{content}".encode()).hexdigest(),
            source="db",
            source_version="v1",
        )
        client = SimpleNamespace(
            list_projects=lambda: [{"id": "project-1", "slug": "fusion"}],
            list_project_prompts=lambda project_id: [
                {"id": "prompt-1", "slug": "generate-title"},
                {"id": "prompt-extra", "slug": "unexpected"},
            ],
        )

        with self.assertRaisesRegex(RuntimeError, "额外 Prompt slug"):
            migrate_prompts(client, [prompt], apply=True)


if __name__ == "__main__":
    unittest.main()
