import json
import unittest
from datetime import datetime, timezone

from app.services.mcp.amap_weather_cache import WEATHER_CACHE_TTL_SECONDS, AmapWeatherCache


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.setex_calls = []

    async def get(self, key):
        return self.values.get(key)

    async def setex(self, key, ttl, value):
        self.setex_calls.append((key, ttl, value))
        self.values[key] = value


def cache_core():
    return {
        "resolved_location": "深圳市",
        "forecast_days": [
            {
                "date": "2026-07-23",
                "weekday": 4,
                "day_weather": "多云",
                "night_weather": "阵雨",
                "high_c": 32,
                "low_c": 27,
            }
        ],
        "fetched_at": datetime(2026, 7, 23, 8, tzinfo=timezone.utc).isoformat(),
        "limitations": ["仅返回 1 天有效预报"],
    }


class AmapWeatherCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_uses_digest_and_adcode_key_with_exact_ttl_and_safe_value(self):
        redis = FakeRedis()
        cache = AmapWeatherCache(
            service_identity="server-private:7:definition-private",
            redis_getter=lambda: redis,
        )

        await cache.set("440300", cache_core())

        self.assertEqual(len(redis.setex_calls), 1)
        key, ttl, raw = redis.setex_calls[0]
        self.assertRegex(key, r"^mcp:weather:v1:[0-9a-f]{24}:440300$")
        self.assertNotIn("server-private", key)
        self.assertEqual(ttl, WEATHER_CACHE_TTL_SECONDS)
        payload = json.loads(raw)
        self.assertEqual(
            set(payload),
            {"resolved_location", "forecast_days", "fetched_at", "limitations"},
        )
        self.assertNotIn("query", raw)
        self.assertNotIn("tool_call_log_id", raw)
        self.assertNotIn("440300", raw)
        restored = await cache.get("440300")
        self.assertEqual(restored["resolved_location"], "深圳市")

    async def test_cache_rejects_unknown_fields_unsorted_days_and_backend_failures(self):
        redis = FakeRedis()
        cache = AmapWeatherCache(service_identity="service", redis_getter=lambda: redis)
        await cache.set("440300", {**cache_core(), "query": "深圳"})
        self.assertEqual(redis.setex_calls, [])

        invalid = cache_core()
        invalid["forecast_days"] = [
            {
                "date": "2026-07-24",
                "weekday": 5,
                "day_weather": "多云",
                "night_weather": "多云",
                "high_c": 31,
                "low_c": 26,
            },
            invalid["forecast_days"][0],
        ]
        await cache.set("440300", invalid)
        self.assertEqual(redis.setex_calls, [])

        class BrokenRedis:
            async def get(self, _key):
                raise RuntimeError("down")

            async def setex(self, *_args):
                raise RuntimeError("down")

        broken = AmapWeatherCache(service_identity="service", redis_getter=BrokenRedis)
        self.assertIsNone(await broken.get("440300"))
        await broken.set("440300", cache_core())


if __name__ == "__main__":
    unittest.main()
