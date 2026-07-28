import unittest

from app.schemas.response import ApiException
from app.services.stream.agent_task_policy import (
    DEEP_RESEARCH_EVIDENCE_POLICY,
    DEEP_RESEARCH_NETWORK_PROFILE,
    resolve_agent_task_policy,
)


class AgentTaskPolicyTests(unittest.TestCase):
    def test_standard_mode_preserves_legacy_plan_mode(self):
        policy = resolve_agent_task_policy(
            options={"plan_mode": "off"},
            capabilities={"functionCalling": True, "searchCapable": True},
        )

        self.assertEqual(policy.task_mode, "standard")
        self.assertEqual(policy.plan_mode, "off")
        self.assertEqual(policy.network_profile, "standard")
        self.assertEqual(policy.evidence_policy, "standard")

    def test_deep_research_forces_plan_and_controlled_profiles(self):
        policy = resolve_agent_task_policy(
            options={"task_mode": "deep_research", "plan_mode": "off"},
            capabilities={"functionCalling": True, "searchCapable": True},
        )

        self.assertEqual(policy.task_mode, "deep_research")
        self.assertEqual(policy.plan_mode, "on")
        self.assertEqual(policy.network_profile, DEEP_RESEARCH_NETWORK_PROFILE)
        self.assertEqual(policy.evidence_policy, DEEP_RESEARCH_EVIDENCE_POLICY)

    def test_unknown_task_mode_safely_falls_back_to_standard(self):
        policy = resolve_agent_task_policy(
            options={"task_mode": "future_mode", "plan_mode": "on"},
            capabilities={"functionCalling": True, "searchCapable": True},
        )

        self.assertEqual(policy.task_mode, "standard")
        self.assertEqual(policy.plan_mode, "on")

    def test_deep_research_rejects_missing_search_capability(self):
        with self.assertRaises(ApiException) as raised:
            resolve_agent_task_policy(
                options={"task_mode": "deep_research"},
                capabilities={"functionCalling": True, "searchCapable": False},
            )

        self.assertEqual(raised.exception.code, "MODEL_UNAVAILABLE")
        self.assertEqual(raised.exception.status_code, 400)

    def test_deep_research_rejects_disabled_tools(self):
        with self.assertRaises(ApiException):
            resolve_agent_task_policy(
                options={"task_mode": "deep_research", "disable_tools": True},
                capabilities={"functionCalling": True, "searchCapable": True},
            )


if __name__ == "__main__":
    unittest.main()
