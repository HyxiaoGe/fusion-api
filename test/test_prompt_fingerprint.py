"""有效 system 消息指纹的确定性与边界测试。"""

import hashlib
import unittest


class PromptFingerprintTests(unittest.TestCase):
    def test_fingerprint_uses_only_ordered_system_messages_and_preserves_boundaries(self):
        from app.utils.prompt_fingerprint import fingerprint_system_messages

        messages = [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "不应参与指纹"},
            {"role": "system", "content": "bc"},
        ]
        expected = hashlib.sha256(b'[{"content":"a","role":"system"},{"content":"bc","role":"system"}]').hexdigest()
        assert fingerprint_system_messages(messages) == expected
        assert fingerprint_system_messages(messages) != fingerprint_system_messages(
            [{"role": "system", "content": "ab"}, {"role": "system", "content": "c"}]
        )
        assert fingerprint_system_messages(messages) != fingerprint_system_messages(list(reversed(messages)))
        assert fingerprint_system_messages([]) == hashlib.sha256(b"[]").hexdigest()

    def test_fingerprint_keeps_structured_content_and_message_fields_without_mutation(self):
        from app.utils.prompt_fingerprint import fingerprint_system_messages

        messages = [{"role": "system", "content": [{"type": "text", "text": "规则"}], "name": "fusion"}]
        reordered = [{"name": "fusion", "content": [{"text": "规则", "type": "text"}], "role": "system"}]
        assert fingerprint_system_messages(messages) == fingerprint_system_messages(reordered)
        assert fingerprint_system_messages(messages) != fingerprint_system_messages(
            [{"role": "system", "content": "规则", "name": "fusion"}]
        )
        assert messages == [{"role": "system", "content": [{"type": "text", "text": "规则"}], "name": "fusion"}]
