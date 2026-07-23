import unittest
from datetime import datetime, timezone

from app.schemas.chat import PlaceResultsBlock, SearchBlock, TextBlock, UnsupportedContentBlock, WeatherResultsBlock
from app.schemas.content_block_registry import deserialize_content_blocks


class ContentBlockRegistryTests(unittest.TestCase):
    def test_deserialize_blocks_isolates_unknown_corrupt_and_future_blocks(self):
        raw_blocks = [
            {"type": "text", "id": "text-1", "text": "保留的回答"},
            {"type": "future_widget", "id": "future-1", "payload": {"value": 1}},
            {
                "type": "place_results",
                "id": "place-future",
                "schema_version": 2,
                "provider": "future-provider",
            },
            {"type": "route_results", "id": "route-corrupt", "schema_version": 1},
            {"type": "text", "id": "text-2", "text": "后续回答"},
        ]

        blocks = deserialize_content_blocks(raw_blocks)

        self.assertEqual(
            [block.type for block in blocks],
            [
                "text",
                "unsupported_result",
                "unsupported_result",
                "unsupported_result",
                "text",
            ],
        )
        self.assertEqual(blocks[0], TextBlock(type="text", id="text-1", text="保留的回答"))
        self.assertEqual(blocks[1].source_type, "future_widget")
        self.assertEqual(blocks[1].reason, "unsupported_type")
        self.assertEqual(blocks[2].source_type, "place_results")
        self.assertEqual(blocks[2].source_schema_version, 2)
        self.assertEqual(blocks[2].reason, "unsupported_version")
        self.assertEqual(blocks[3].source_type, "route_results")
        self.assertEqual(blocks[3].source_schema_version, 1)
        self.assertEqual(blocks[3].reason, "invalid_payload")
        self.assertEqual(blocks[4], TextBlock(type="text", id="text-2", text="后续回答"))
        self.assertEqual(raw_blocks[1]["type"], "future_widget")

    def test_unsupported_fallback_is_safe_stable_and_unique(self):
        raw_blocks = [
            {"type": "future_widget", "payload": {"provider": "PRIVATE_PROVIDER"}},
            {"type": "future_widget", "payload": {"provider": "PRIVATE_PROVIDER"}},
            {"type": "future_widget", "id": "duplicate-id", "tool_name": "PRIVATE_TOOL"},
            {"type": "future_widget", "id": "duplicate-id", "tool_name": "PRIVATE_TOOL"},
        ]

        first = deserialize_content_blocks(raw_blocks)
        second = deserialize_content_blocks(raw_blocks)

        self.assertTrue(all(isinstance(block, UnsupportedContentBlock) for block in first))
        self.assertEqual([block.id for block in first], [block.id for block in second])
        self.assertEqual(len({block.id for block in first}), len(first))
        serialized = [block.model_dump(mode="json") for block in first]
        self.assertNotIn("PRIVATE_PROVIDER", str(serialized))
        self.assertNotIn("PRIVATE_TOOL", str(serialized))
        self.assertTrue(
            all(
                set(block)
                <= {
                    "type",
                    "id",
                    "source_type",
                    "source_schema_version",
                    "reason",
                }
                for block in serialized
            )
        )

    def test_deserialize_blocks_keeps_legacy_search_defaults(self):
        blocks = deserialize_content_blocks(
            [
                {
                    "type": "search",
                    "tool_call_log_id": "log-legacy",
                }
            ]
        )

        self.assertEqual(len(blocks), 1)
        self.assertIsInstance(blocks[0], SearchBlock)
        self.assertEqual(blocks[0].query, "")
        self.assertEqual(blocks[0].sources, [])
        self.assertEqual(blocks[0].tool_call_log_id, "log-legacy")

    def test_deserialize_blocks_accepts_current_rich_schema_version(self):
        blocks = deserialize_content_blocks(
            [
                {
                    "type": "place_results",
                    "id": "place-1",
                    "schema_version": 1,
                    "provider": "amap",
                    "query": "咖啡",
                    "status": "success",
                    "result_count": 0,
                    "places": [],
                }
            ]
        )

        self.assertEqual(len(blocks), 1)
        self.assertIsInstance(blocks[0], PlaceResultsBlock)
        self.assertEqual(blocks[0].schema_version, 1)

    def test_deserialize_blocks_accepts_weather_schema_v1(self):
        blocks = deserialize_content_blocks(
            [
                {
                    "type": "weather_results",
                    "id": "weather-1",
                    "schema_version": 1,
                    "provider": "amap",
                    "status": "degraded",
                    "query": "深圳市",
                    "resolved_location": "深圳市",
                    "day_count": 1,
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
                }
            ]
        )

        self.assertEqual(len(blocks), 1)
        self.assertIsInstance(blocks[0], WeatherResultsBlock)
        self.assertEqual(blocks[0].forecast_days[0].weekday, 4)

    def test_deserialize_blocks_treats_non_list_payload_as_empty(self):
        self.assertEqual(deserialize_content_blocks(None), [])
        self.assertEqual(deserialize_content_blocks({"type": "text"}), [])
        self.assertEqual(deserialize_content_blocks("invalid"), [])

    def test_malformed_block_becomes_fallback_instead_of_empty_message(self):
        blocks = deserialize_content_blocks([{"type": "place_results", "schema_version": 9}])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].type, "unsupported_result")
        self.assertEqual(blocks[0].source_type, "place_results")
        self.assertEqual(blocks[0].source_schema_version, 9)
        self.assertEqual(blocks[0].reason, "unsupported_version")


if __name__ == "__main__":
    unittest.main()
