import json
import unittest
from unittest.mock import Mock

from scripts import audit_litellm_model_catalog as audit


def litellm_entry(
    model_name: str,
    *,
    db_model: bool = True,
    metadata: dict | None = None,
    model_uuid: str | None = None,
    underlying_model: str | None = None,
) -> dict:
    return {
        "model_name": model_name,
        "model_info": {
            "id": model_uuid or f"uuid-{model_name}",
            "db_model": db_model,
            "metadata": metadata
            if metadata is not None
            else {
                "provider_key": "deepseek",
                "provider_display": "DeepSeek",
                "capabilities": {"functionCalling": True},
                "pricing": {"input": 0.001, "output": 0.002, "unit": "USD"},
            },
        },
        "litellm_params": {"model": underlying_model or f"openai/{model_name}"},
    }


class ModelCatalogAuditTests(unittest.TestCase):
    def test_clean_catalog_has_no_issues(self):
        report = audit.audit_catalog(
            litellm_entries=[litellm_entry("deepseek-chat")],
            fusion_models=[{"modelId": "deepseek-chat"}],
            key_models=["deepseek-chat"],
        )

        self.assertEqual(report.summary["issue_count"], 0)
        self.assertEqual(report.issues, [])
        self.assertEqual(report.sync_plan.add, [])
        self.assertEqual(report.sync_plan.remove, [])

    def test_fusion_unknown_model_is_error(self):
        report = audit.audit_catalog(
            litellm_entries=[litellm_entry("deepseek-chat")],
            fusion_models=[{"modelId": "ghost-model"}],
            key_models=["deepseek-chat"],
        )

        self.assertEqual(report.issues[0].code, "fusion_unknown_model")
        self.assertEqual(report.issues[0].severity, "error")
        self.assertEqual(report.issues[0].model_name, "ghost-model")

    def test_key_missing_db_model_is_error_and_sync_adds_it(self):
        report = audit.audit_catalog(
            litellm_entries=[litellm_entry("deepseek-chat"), litellm_entry("mimo-v2.5-pro")],
            fusion_models=[{"modelId": "deepseek-chat"}, {"modelId": "mimo-v2.5-pro"}],
            key_models=["deepseek-chat"],
        )

        self.assertIn("mimo-v2.5-pro", report.sync_plan.add)
        self.assertIn("mimo-v2.5-pro", report.sync_plan.allowlist_after)
        issue = [issue for issue in report.issues if issue.code == "key_missing_db_model"][0]
        self.assertEqual(issue.severity, "error")
        self.assertEqual(issue.model_name, "mimo-v2.5-pro")

    def test_sync_plan_deduplicates_duplicate_db_model_entries(self):
        plan = audit.build_allowlist_sync_plan(
            db_model_names=["deepseek-chat", "mimo-v2.5-pro", "mimo-v2.5-pro"],
            key_models=["deepseek-chat"],
        )

        self.assertEqual(plan.add, ["mimo-v2.5-pro"])
        self.assertEqual(plan.allowlist_after, ["deepseek-chat", "mimo-v2.5-pro"])

    def test_identical_duplicate_db_model_entries_are_counted_once_without_warning(self):
        report = audit.audit_catalog(
            litellm_entries=[litellm_entry("deepseek-chat"), litellm_entry("deepseek-chat")],
            fusion_models=[{"modelId": "deepseek-chat"}],
            key_models=["deepseek-chat"],
        )

        self.assertEqual(report.summary["litellm_db_models"], 1)
        self.assertFalse([issue for issue in report.issues if issue.code == "db_model_duplicate"])

    def test_conflicting_duplicate_db_model_entries_are_warned_and_counted_once(self):
        report = audit.audit_catalog(
            litellm_entries=[
                litellm_entry(
                    "deepseek-chat",
                    model_uuid="uuid-deepseek-chat-old",
                    underlying_model="deepseek/deepseek-v3",
                ),
                litellm_entry(
                    "deepseek-chat",
                    model_uuid="uuid-deepseek-chat-new",
                    underlying_model="deepseek/deepseek-v4-flash",
                ),
            ],
            fusion_models=[{"modelId": "deepseek-chat"}],
            key_models=["deepseek-chat"],
        )

        self.assertEqual(report.summary["litellm_db_models"], 1)
        issue = [issue for issue in report.issues if issue.code == "db_model_duplicate"][0]
        self.assertEqual(issue.severity, "warning")
        self.assertEqual(issue.model_name, "deepseek-chat")

    def test_hidden_incomplete_db_model_is_warning_not_sync_target(self):
        report = audit.audit_catalog(
            litellm_entries=[
                litellm_entry("deepseek-chat"),
                litellm_entry(
                    "chat-default",
                    metadata={"provider_display": "Google"},
                ),
            ],
            fusion_models=[{"modelId": "deepseek-chat"}],
            key_models=["deepseek-chat"],
        )

        self.assertNotIn("chat-default", report.sync_plan.add)
        self.assertFalse([issue for issue in report.issues if issue.code == "key_missing_db_model"])
        metadata_issue = [issue for issue in report.issues if issue.code == "metadata_missing"][0]
        self.assertEqual(metadata_issue.model_name, "chat-default")

    def test_deprecated_key_model_is_error_and_sync_removes_it(self):
        report = audit.audit_catalog(
            litellm_entries=[litellm_entry("mimo-v2.5-pro")],
            fusion_models=[{"modelId": "mimo-v2.5-pro"}],
            key_models=["mimo-v2-flash", "mimo-v2.5-pro"],
        )

        self.assertIn("mimo-v2-flash", report.sync_plan.remove)
        self.assertNotIn("mimo-v2-flash", report.sync_plan.allowlist_after)
        issue = [issue for issue in report.issues if issue.code == "key_deprecated_model"][0]
        self.assertEqual(issue.severity, "error")
        self.assertEqual(issue.model_name, "mimo-v2-flash")

    def test_missing_metadata_is_warning(self):
        report = audit.audit_catalog(
            litellm_entries=[litellm_entry("bad-model", metadata={"provider_key": "bad"})],
            fusion_models=[{"modelId": "bad-model"}],
            key_models=["bad-model"],
        )

        issue = [issue for issue in report.issues if issue.code == "metadata_missing"][0]
        self.assertEqual(issue.severity, "warning")
        self.assertEqual(issue.model_name, "bad-model")
        self.assertIn("pricing", issue.message)

    def test_serialize_report_does_not_include_secrets(self):
        report = audit.audit_catalog(
            litellm_entries=[litellm_entry("deepseek-chat")],
            fusion_models=[{"modelId": "deepseek-chat"}],
            key_models=["deepseek-chat"],
        )

        serialized = audit.serialize_report(
            report,
            context={
                "litellm_base_url": "http://litellm-proxy:4000",
                "master_key": "sk-master-secret",
                "virtual_key": "sk-virtual-secret",
            },
        )
        payload = json.dumps(serialized, ensure_ascii=False)

        self.assertIn("litellm_base_url", serialized["context"])
        self.assertNotIn("sk-master-secret", payload)
        self.assertNotIn("sk-virtual-secret", payload)

    def test_apply_sync_only_updates_key_models(self):
        client = Mock()
        report = audit.audit_catalog(
            litellm_entries=[litellm_entry("deepseek-chat"), litellm_entry("mimo-v2.5-pro")],
            fusion_models=[{"modelId": "deepseek-chat"}, {"modelId": "mimo-v2.5-pro"}],
            key_models=["deepseek-chat"],
        )

        audit.apply_sync(
            base_url="http://litellm-proxy:4000",
            master_key="sk-master",
            virtual_key="sk-virtual",
            sync_plan=report.sync_plan,
            client=client,
        )

        client.post.assert_called_once()
        url = client.post.call_args.args[0]
        body = client.post.call_args.kwargs["json"]
        self.assertEqual(url, "http://litellm-proxy:4000/key/update")
        self.assertEqual(body["key"], "sk-virtual")
        self.assertEqual(body["models"], ["deepseek-chat", "mimo-v2.5-pro"])


if __name__ == "__main__":
    unittest.main()
