"""sanitizer 单元测试"""
import json
import unittest

from app.services.agent.sanitizer import cap_and_truncate, sanitize_arguments


class SanitizeArgumentsTests(unittest.TestCase):
    def test_v1_pass_through_web_search(self):
        args = {"query": "GPT-5 评测"}
        self.assertEqual(sanitize_arguments("web_search", args), args)

    def test_url_read_sanitizes_sensitive_url(self):
        args = {"url": "https://example.com/page?token=secret&safe=1", "reason": "核实原文"}
        sanitized = sanitize_arguments("url_read", args)

        self.assertEqual(sanitized["url"], "https://example.com/page")
        self.assertEqual(sanitized["reason"], "核实原文")
        self.assertNotIn("token", str(sanitized))
        self.assertEqual(args["url"], "https://example.com/page?token=secret&safe=1")

    def test_url_read_sanitize_malformed_url_is_best_effort(self):
        args = {"url": "http://[::1", "reason": "核实原文"}
        sanitized = sanitize_arguments("url_read", args)

        self.assertEqual(sanitized["url"], "")
        self.assertEqual(sanitized["url_policy_reason"], "invalid_url")

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

    def test_nested_dict_string_truncated_under_limit(self):
        """嵌套 dict 中的长 string 也被递归截断到 ≤ max_bytes"""
        payload = {
            "kind": "search",
            "meta": {"snippet": "x" * 5000},
            "truncated": False,
        }
        out = cap_and_truncate(payload, max_bytes=1024)
        self.assertTrue(out["truncated"])
        self.assertLessEqual(
            len(json.dumps(out, ensure_ascii=False).encode("utf-8")),
            1024,
        )
        # 嵌套结构仍存在但 snippet 被裁短
        self.assertIn("meta", out)
        self.assertLess(len(out["meta"]["snippet"]), 5000)

    def test_deeply_nested_list_strings_truncated(self):
        """多层嵌套 dict + list 内的长 string 都被递归截断"""
        payload = {
            "kind": "search",
            "items": [{"snippet": "y" * 3000}, {"snippet": "z" * 3000}],
            "truncated": False,
        }
        out = cap_and_truncate(payload, max_bytes=1024)
        self.assertTrue(out["truncated"])
        self.assertLessEqual(
            len(json.dumps(out, ensure_ascii=False).encode("utf-8")),
            1024,
        )

    def test_extreme_oversized_falls_back_to_minimal(self):
        """无法靠字符串截断达标时，删除嵌套 container；仍超时回退到 minimal"""
        payload = {
            "kind": "search",
            "many_items": [{"a": str(i) * 100} for i in range(50)],
        }
        out = cap_and_truncate(payload, max_bytes=64)
        self.assertLessEqual(
            len(json.dumps(out, ensure_ascii=False).encode("utf-8")),
            64,
        )
        self.assertTrue(out.get("truncated", False))
        self.assertEqual(out.get("kind"), "search")  # kind 应被保留

    def test_payload_not_mutated_in_place(self):
        """deep copy 保证调用方原 dict 不被修改"""
        nested = {"snippet": "x" * 5000}
        payload = {"kind": "search", "meta": nested, "truncated": False}
        cap_and_truncate(payload, max_bytes=1024)
        # 原嵌套对象应不变
        self.assertEqual(len(nested["snippet"]), 5000)
        self.assertFalse(payload["truncated"])
