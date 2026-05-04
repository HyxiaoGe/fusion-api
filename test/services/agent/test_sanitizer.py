"""sanitizer 单元测试"""
import json
import unittest

from app.services.agent.sanitizer import cap_and_truncate, sanitize_arguments


class SanitizeArgumentsTests(unittest.TestCase):
    def test_v1_pass_through_web_search(self):
        args = {"query": "GPT-5 评测"}
        self.assertEqual(sanitize_arguments("web_search", args), args)

    def test_v1_pass_through_url_read(self):
        args = {"url": "https://example.com"}
        self.assertEqual(sanitize_arguments("url_read", args), args)

    def test_unknown_tool_pass_through(self):
        args = {"x": 1}
        self.assertEqual(sanitize_arguments("future_tool", args), args)


class CapAndTruncateTests(unittest.TestCase):
    def test_under_limit_unchanged(self):
        payload = {"kind": "search", "title": "abc", "count": 1, "truncated": False}
        out = cap_and_truncate(payload, max_bytes=1024)
        self.assertFalse(out["truncated"])
        self.assertEqual(out["title"], "abc")

    def test_over_limit_truncates_string_fields(self):
        payload = {"kind": "search", "title": "x" * 5000, "count": 1, "truncated": False}
        out = cap_and_truncate(payload, max_bytes=1024)
        self.assertTrue(out["truncated"])
        self.assertLessEqual(len(json.dumps(out, ensure_ascii=False).encode("utf-8")), 1024)

    def test_non_string_fields_preserved(self):
        payload = {"kind": "search", "title": "x" * 5000, "count": 99, "truncated": False}
        out = cap_and_truncate(payload, max_bytes=1024)
        self.assertEqual(out["count"], 99)
        self.assertEqual(out["kind"], "search")
