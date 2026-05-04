import unittest

from app.ai.litellm_utils import ProviderOfflineError, merge_extra_body


class MergeExtraBodyTests(unittest.TestCase):
    def test_merge_into_empty_kwargs(self):
        kwargs = {}
        merge_extra_body(kwargs, {"api_key": "sk-1"})
        self.assertEqual(kwargs["extra_body"], {"api_key": "sk-1"})

    def test_merge_preserves_existing_field(self):
        kwargs = {"extra_body": {"api_key": "sk-1"}}
        merge_extra_body(kwargs, {"thinking": {"type": "disabled"}})
        self.assertEqual(kwargs["extra_body"], {"api_key": "sk-1", "thinking": {"type": "disabled"}})

    def test_new_field_overrides_same_key(self):
        kwargs = {"extra_body": {"api_key": "sk-old"}}
        merge_extra_body(kwargs, {"api_key": "sk-new"})
        self.assertEqual(kwargs["extra_body"]["api_key"], "sk-new")

    def test_handles_none_extra_body(self):
        kwargs = {"extra_body": None}
        merge_extra_body(kwargs, {"api_key": "sk-1"})
        self.assertEqual(kwargs["extra_body"], {"api_key": "sk-1"})


class ProviderOfflineErrorTests(unittest.TestCase):
    def test_carries_provider_id_and_reason(self):
        exc = ProviderOfflineError(provider_id="xai", reason="quota_exceeded", message="余额耗尽")
        self.assertEqual(exc.provider_id, "xai")
        self.assertEqual(exc.reason, "quota_exceeded")
        self.assertEqual(exc.message, "余额耗尽")
        self.assertIn("xai", str(exc))
