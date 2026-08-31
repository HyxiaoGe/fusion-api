import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.ai.skills.registry import SkillReleasePin
from app.services.stream.previous_run_skill_release import (
    UNRESTORABLE_SKILL_RELEASE_PINS,
    load_previous_run_skill_release_pins,
)


class PreviousRunSkillReleaseTests(unittest.TestCase):
    def test_loaded_skill_is_restored_from_owner_scoped_previous_run(self):
        payload = {
            "schema_version": 2,
            "router_version": "2026-08-31.1",
            "package_id": "verified_web",
            "confidence": "high",
            "resolution_mode": "routed",
            "reason_codes": ["verified_source_request"],
            "external_tool_names": ["web_search", "url_read"],
            "effective_plan_mode": "auto",
            "include_current_date": True,
            "network_boundary_required": False,
            "bundle_fingerprint": "sha256:" + "b" * 64,
            "skill_resolution": {
                "status": "loaded",
                "activation_source": "capability_package",
                "requested_skill_ids": ["verified-research"],
                "skills": [
                    {
                        "skill_id": "verified-research",
                        "version": "0.9.0",
                        "content_sha256": "a" * 64,
                        "allowed_tool_names": ["web_search", "url_read"],
                        "section_id": "skill:verified-research@0.9.0",
                        "char_count": 128,
                    }
                ],
                "duration_ms": 4,
                "error_code": None,
            },
        }

        with patch("app.services.stream.previous_run_skill_release.TrajectoryRepository") as repository_cls:
            repository_cls.return_value.get_run.return_value = SimpleNamespace(capability_resolution=payload)
            result = load_previous_run_skill_release_pins(
                "db",
                conversation_id="conv-1",
                user_id="user-1",
                previous_run_id="run-1",
            )

        repository_cls.assert_called_once_with("db")
        repository_cls.return_value.get_run.assert_called_once_with("conv-1", "run-1", "user-1")
        self.assertEqual(
            result,
            (
                SkillReleasePin(
                    skill_id="verified-research",
                    version="0.9.0",
                    content_sha256="a" * 64,
                ),
            ),
        )

    def test_invalid_or_historical_resolution_fails_closed_instead_of_loading_current_release(self):
        with patch("app.services.stream.previous_run_skill_release.TrajectoryRepository") as repository_cls:
            repository_cls.return_value.get_run.return_value = SimpleNamespace(
                capability_resolution={"schema_version": 2, "package_id": "verified_web"}
            )
            result = load_previous_run_skill_release_pins(
                "db",
                conversation_id="conv-1",
                user_id="user-1",
                previous_run_id="run-1",
            )

        self.assertEqual(result, UNRESTORABLE_SKILL_RELEASE_PINS)

    def test_explicit_not_selected_previous_run_is_the_only_empty_restore_result(self):
        payload = {
            "schema_version": 2,
            "router_version": "2026-08-31.1",
            "package_id": "direct",
            "confidence": "high",
            "resolution_mode": "routed",
            "reason_codes": ["direct_greeting"],
            "external_tool_names": [],
            "effective_plan_mode": "off",
            "include_current_date": False,
            "network_boundary_required": False,
            "bundle_fingerprint": "sha256:" + "b" * 64,
            "skill_resolution": {
                "status": "not_selected",
                "activation_source": "capability_package",
                "requested_skill_ids": [],
                "skills": [],
                "duration_ms": 0,
                "error_code": None,
            },
        }
        with patch("app.services.stream.previous_run_skill_release.TrajectoryRepository") as repository_cls:
            repository_cls.return_value.get_run.return_value = SimpleNamespace(capability_resolution=payload)
            result = load_previous_run_skill_release_pins(
                "db",
                conversation_id="conv-1",
                user_id="user-1",
                previous_run_id="run-1",
            )

        self.assertEqual(result, ())


if __name__ == "__main__":
    unittest.main()
