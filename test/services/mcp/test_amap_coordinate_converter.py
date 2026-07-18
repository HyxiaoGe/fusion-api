import os
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import httpx

from app.services.agent.context_broker import Geolocation
from app.services.mcp.amap_coordinate_converter import (
    AmapCoordinateConversionError,
    convert_wgs84_to_gcj02,
)


class AmapCoordinateConverterTests(IsolatedAsyncioTestCase):
    async def test_uses_official_api_and_limits_coordinate_precision(self):
        captured = {}

        async def handle(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            return httpx.Response(200, json={"status": "1", "locations": "114.1234567,22.7654321"})

        location = Geolocation(latitude=22.7, longitude=114.1, accuracy_m=10, acquired_at=1_700_000_000)
        with self.assertLogs("httpx", level="INFO") as http_logs:
            result = await convert_wgs84_to_gcj02(
                location,
                api_key="SECRET_AMAP_KEY",
                transport=httpx.MockTransport(handle),
            )

        request = captured["request"]
        self.assertEqual(request.url.path, "/v3/assistant/coordinate/convert")
        self.assertEqual(request.url.params["coordsys"], "gps")
        self.assertEqual(request.url.params["locations"], "114.100000,22.700000")
        self.assertEqual(result, "114.123457,22.765432")
        serialized_logs = "\n".join(http_logs.output)
        self.assertNotIn("SECRET_AMAP_KEY", serialized_logs)
        self.assertNotIn("114.100000", serialized_logs)
        self.assertNotIn("22.700000", serialized_logs)

    async def test_missing_key_and_invalid_payload_fail_without_echoing_key(self):
        location = Geolocation(latitude=22.7, longitude=114.1, accuracy_m=10, acquired_at=1_700_000_000)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(AmapCoordinateConversionError) as missing:
                await convert_wgs84_to_gcj02(location)
        self.assertEqual(str(missing.exception), "高德坐标转换不可用")

        async def handle(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "0", "info": "SECRET_AMAP_KEY"})

        with self.assertRaises(AmapCoordinateConversionError) as invalid:
            await convert_wgs84_to_gcj02(
                location,
                api_key="SECRET_AMAP_KEY",
                transport=httpx.MockTransport(handle),
            )
        self.assertNotIn("SECRET_AMAP_KEY", str(invalid.exception))
