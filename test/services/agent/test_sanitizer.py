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

    def test_under_limit_returns_same_object(self):
        """under-limit 路径必须零拷贝（return 同一对象）"""
        payload = {"kind": "search", "title": "abc", "truncated": False}
        out = cap_and_truncate(payload, max_bytes=1024)
        self.assertIs(out, payload)

    def test_empty_dict(self):
        """空 dict 应返回 {} 不动"""
        out = cap_and_truncate({}, max_bytes=1024)
        self.assertEqual(out, {})

    def test_nested_dict_not_truncated(self):
        """v1 仅截断顶层 string；嵌套 dict 不参与截断（文档化当前行为）"""
        payload = {
            "kind": "search",
            "meta": {"snippet": "x" * 5000},  # 嵌套 dict 内的长 string
            "truncated": False,
        }
        out = cap_and_truncate(payload, max_bytes=1024)
        # 顶层 truncated 标志被设
        self.assertTrue(out["truncated"])
        # 但嵌套字段不会被裁
        self.assertEqual(len(out["meta"]["snippet"]), 5000)
