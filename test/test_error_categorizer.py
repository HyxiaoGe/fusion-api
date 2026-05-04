import unittest

from app.services.error_categorizer import ErrorKind, categorize


def make_litellm_exc(status_code: int, message: str = "test err"):
    """制造一个伪 litellm 异常，带 status_code 属性。"""
    exc = Exception(message)
    exc.status_code = status_code
    return exc


class ErrorCategorizerTests(unittest.TestCase):
    def test_401_is_key_invalid(self):
        kind, msg = categorize(make_litellm_exc(401, "Unauthorized"))
        self.assertEqual(kind, ErrorKind.KEY_INVALID)
        self.assertIn("Unauthorized", msg)

    def test_402_is_quota_exceeded(self):
        kind, _ = categorize(make_litellm_exc(402, "Payment required"))
        self.assertEqual(kind, ErrorKind.QUOTA_EXCEEDED)

    def test_quota_message_overrides(self):
        kind, _ = categorize(make_litellm_exc(400, "insufficient credits"))
        self.assertEqual(kind, ErrorKind.QUOTA_EXCEEDED)

    def test_403_tos_message_is_tos_blocked(self):
        kind, _ = categorize(make_litellm_exc(403, "Terms of Service violation"))
        self.assertEqual(kind, ErrorKind.TOS_BLOCKED)

    def test_403_default_is_key_invalid(self):
        kind, _ = categorize(make_litellm_exc(403, "Forbidden"))
        self.assertEqual(kind, ErrorKind.KEY_INVALID)

    def test_429_is_transient(self):
        kind, _ = categorize(make_litellm_exc(429, "rate limited"))
        self.assertEqual(kind, ErrorKind.TRANSIENT)

    def test_500_is_transient(self):
        kind, _ = categorize(make_litellm_exc(500, "server error"))
        self.assertEqual(kind, ErrorKind.TRANSIENT)

    def test_unknown_falls_back(self):
        kind, _ = categorize(Exception("random"))
        self.assertEqual(kind, ErrorKind.UNKNOWN)
