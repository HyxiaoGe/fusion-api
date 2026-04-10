import unittest

from app.schemas.response import ApiException, ApiResponse, ErrorCode, generate_request_id, success


class TestGenerateRequestId(unittest.TestCase):
    def test_format(self):
        rid = generate_request_id()
        self.assertTrue(rid.startswith("req_"))
        self.assertEqual(len(rid), 16)  # "req_" + 12 hex chars

    def test_unique(self):
        ids = {generate_request_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)


class TestErrorCode(unittest.TestCase):
    def test_success_value(self):
        self.assertEqual(ErrorCode.SUCCESS, "SUCCESS")
        self.assertEqual(ErrorCode.NOT_FOUND, "NOT_FOUND")

    def test_is_string(self):
        self.assertIsInstance(ErrorCode.SUCCESS, str)


class TestApiResponse(unittest.TestCase):
    def test_success_with_data(self):
        resp = ApiResponse(data={"key": "value"}, request_id="req_abc123def456")
        self.assertEqual(resp.code, "SUCCESS")
        self.assertEqual(resp.message, "ok")
        self.assertEqual(resp.data, {"key": "value"})
        self.assertEqual(resp.request_id, "req_abc123def456")

    def test_success_null_data(self):
        resp = ApiResponse(request_id="req_abc123def456")
        self.assertIsNone(resp.data)

    def test_serialization(self):
        resp = ApiResponse(data=[1, 2, 3], request_id="req_abc123def456")
        d = resp.model_dump()
        self.assertEqual(d["code"], "SUCCESS")
        self.assertEqual(d["data"], [1, 2, 3])


class TestSuccessHelper(unittest.TestCase):
    def test_default(self):
        resp = success(request_id="req_abc123def456")
        self.assertEqual(resp.code, "SUCCESS")
        self.assertEqual(resp.message, "ok")
        self.assertIsNone(resp.data)

    def test_with_data_and_message(self):
        resp = success(data={"id": 1}, message="已创建", request_id="req_abc123def456")
        self.assertEqual(resp.message, "已创建")
        self.assertEqual(resp.data, {"id": 1})


class TestApiException(unittest.TestCase):
    def test_fields(self):
        exc = ApiException(ErrorCode.NOT_FOUND, "会话不存在", 404)
        self.assertEqual(exc.code, "NOT_FOUND")
        self.assertEqual(exc.message, "会话不存在")
        self.assertEqual(exc.status_code, 404)

    def test_default_status_code(self):
        exc = ApiException(ErrorCode.INVALID_PARAM, "参数错误")
        self.assertEqual(exc.status_code, 400)


if __name__ == "__main__":
    unittest.main()
