import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def _published_bundle(*, revision: str = "b" * 64):
    from app.core.prompt_catalog import PROMPT_SPECS
    from app.services.external.prompthub_client import PromptHubBundle, PromptHubBundleItem
    from app.services.runtime_config_defaults import DEFAULT_PROMPT_TEMPLATES

    return PromptHubBundle(
        project_id="project-1",
        project_slug="fusion",
        revision=revision,
        prompts=tuple(
            PromptHubBundleItem(
                id=f"id-{spec.slug}",
                slug=spec.slug,
                name=spec.key,
                version="1.0.0",
                status="published",
                content=DEFAULT_PROMPT_TEMPLATES[spec.key],
                variables=spec.variables,
                format="text",
                template_engine="none",
                published_at="2026-07-10T00:00:00Z",
            )
            for spec in PROMPT_SPECS
        ),
    )


class PromptBundleValidationTests(unittest.TestCase):
    def test_catalog_has_exactly_the_eleven_runtime_prompts(self):
        from app.core.prompt_catalog import PROMPT_SPECS

        self.assertEqual(len(PROMPT_SPECS), 11)
        self.assertEqual(
            {spec.key for spec in PROMPT_SPECS},
            {
                "app_identity",
                "tool_usage_contract",
                "no_tool_network_boundary",
                "no_vision_file_boundary",
                "url_read_tool_description",
                "limit_summary",
                "continuation_system",
                "generate_title",
                "generate_suggested_questions",
                "file_analysis",
                "file_content_enhancement",
            },
        )

    def test_validates_and_serializes_complete_bundle_with_local_checksums(self):
        from app.core.prompt_bundle import validate_published_bundle

        validated = validate_published_bundle(_published_bundle())

        self.assertEqual(validated["schema_version"], 1)
        self.assertEqual(validated["revision"], "b" * 64)
        self.assertEqual(len(validated["prompts"]), 11)
        self.assertEqual(len(validated["prompts"]["generate_title"]["content_sha256"]), 64)
        self.assertEqual(validated["prompts"]["generate_title"]["variables"], ["content"])

    def test_rejects_missing_duplicate_or_unknown_slug_as_whole_bundle(self):
        from app.core.prompt_bundle import PromptBundleValidationError, validate_published_bundle

        base = _published_bundle()
        unknown = SimpleNamespace(**{**vars(base.prompts[-1]), "slug": "unknown"})
        invalid_prompt_sets = [
            base.prompts[:-1],
            (*base.prompts[:-1], base.prompts[0]),
            (*base.prompts[:-1], unknown),
        ]

        for prompts in invalid_prompt_sets:
            with self.subTest(slugs=[prompt.slug for prompt in prompts]):
                invalid = SimpleNamespace(**{**vars(base), "prompts": tuple(prompts)})
                with self.assertRaises(PromptBundleValidationError):
                    validate_published_bundle(invalid)

    def test_rejects_bad_revision_variables_marker_and_prompt_contract(self):
        from app.core.prompt_bundle import PromptBundleValidationError, validate_published_bundle

        base = _published_bundle()
        bad_items = [
            ("revision", SimpleNamespace(**{**vars(base), "revision": "not-sha256"})),
            (
                "variables",
                SimpleNamespace(
                    **{
                        **vars(base),
                        "prompts": (
                            *base.prompts[:-4],
                            SimpleNamespace(**{**vars(base.prompts[-4]), "variables": ()}),
                            *base.prompts[-3:],
                        ),
                    }
                ),
            ),
            (
                "marker",
                SimpleNamespace(
                    **{
                        **vars(base),
                        "prompts": (
                            SimpleNamespace(**{**vars(base.prompts[0]), "content": "没有固定标记"}),
                            *base.prompts[1:],
                        ),
                    }
                ),
            ),
            (
                "status",
                SimpleNamespace(
                    **{
                        **vars(base),
                        "prompts": (
                            SimpleNamespace(**{**vars(base.prompts[0]), "status": "draft"}),
                            *base.prompts[1:],
                        ),
                    }
                ),
            ),
        ]

        for name, invalid in bad_items:
            with self.subTest(name=name), self.assertRaises(PromptBundleValidationError):
                validate_published_bundle(invalid)

    def test_rejects_unknown_missing_or_malformed_python_placeholders(self):
        from app.core.prompt_bundle import PromptBundleValidationError, validate_published_bundle

        for content in (
            "对话内容：{unexpected}",
            "对话内容：没有变量",
            "对话内容：{content",
            "对话内容：{content:{unexpected}}",
            "对话内容：{content!z}",
        ):
            with self.subTest(content=content):
                base = _published_bundle()
                prompts = list(base.prompts)
                index = next(i for i, item in enumerate(prompts) if item.slug == "generate-title")
                prompts[index] = SimpleNamespace(**{**vars(prompts[index]), "content": content})
                invalid = SimpleNamespace(**{**vars(base), "prompts": tuple(prompts)})

                with self.assertRaises(PromptBundleValidationError):
                    validate_published_bundle(invalid)

    def test_stored_bundle_requires_each_prompt_version(self):
        from app.core.prompt_bundle import validate_published_bundle, validate_stored_bundle_payload

        payload = validate_published_bundle(_published_bundle())
        payload["prompts"]["generate_title"]["version"] = ""

        self.assertFalse(validate_stored_bundle_payload(payload))


class PromptBundleResolverTests(unittest.TestCase):
    def test_resolver_exposes_prompthub_version_metadata_without_remote_request(self):
        from app.core.prompt_bundle import resolve_prompt_template_with_metadata

        payload = {
            "schema_version": 1,
            "project_slug": "fusion",
            "revision": "c" * 64,
            "prompts": {
                "generate_title": {
                    "slug": "generate-title",
                    "version": "1.2.3",
                    "content": "标题模板 {content}",
                    "variables": ["content"],
                    "content_sha256": hashlib.sha256("标题模板 {content}".encode()).hexdigest(),
                    "published_at": "2026-07-10T00:00:00Z",
                }
            },
        }

        with (
            patch("app.core.prompt_bundle.settings.PROMPTHUB_SYNC_MODE", "apply"),
            patch("app.core.prompt_bundle._load_active_bundle_payload", return_value=payload),
        ):
            template, metadata = resolve_prompt_template_with_metadata(
                "generate_title",
                "代码默认值",
                legacy_loader=unittest.mock.Mock(),
            )

        self.assertEqual(template, "标题模板 {content}")
        self.assertEqual(
            metadata,
            {
                "source": "prompthub",
                "prompt_slug": "generate-title",
                "prompt_version": "1.2.3",
                "prompt_revision": "c" * 64,
            },
        )

    def test_apply_reads_active_bundle_before_legacy_runtime_config(self):
        from app.core.prompt_bundle import resolve_prompt_template

        payload = {
            "schema_version": 1,
            "project_slug": "fusion",
            "revision": "c" * 64,
            "prompts": {
                "generate_title": {
                    "slug": "generate-title",
                    "version": "2.0.0",
                    "content": "Bundle 标题：{content}",
                    "variables": ["content"],
                    "content_sha256": hashlib.sha256("Bundle 标题：{content}".encode()).hexdigest(),
                    "published_at": None,
                }
            },
        }
        legacy_loader = unittest.mock.Mock(return_value=({"template": "旧模板"}, {"source": "db"}))

        with (
            patch("app.core.prompt_bundle.settings.PROMPTHUB_SYNC_MODE", "apply"),
            patch("app.core.prompt_bundle._load_active_bundle_payload", return_value=payload),
        ):
            template = resolve_prompt_template("generate_title", "代码默认值", legacy_loader=legacy_loader)

        self.assertEqual(template, "Bundle 标题：{content}")
        legacy_loader.assert_not_called()

    def test_shadow_disabled_and_invalid_bundle_fall_back_to_legacy(self):
        from app.core.prompt_bundle import resolve_prompt_template

        for mode, active in (("disabled", None), ("shadow", None), ("apply", {"prompts": {}})):
            with self.subTest(mode=mode):
                legacy_loader = unittest.mock.Mock(return_value=({"template": "旧模板"}, {"source": "db"}))
                with (
                    patch("app.core.prompt_bundle.settings.PROMPTHUB_SYNC_MODE", mode),
                    patch("app.core.prompt_bundle._load_active_bundle_payload", return_value=active),
                ):
                    template = resolve_prompt_template("generate_title", "代码默认值", legacy_loader=legacy_loader)

                self.assertEqual(template, "旧模板")
                legacy_loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
