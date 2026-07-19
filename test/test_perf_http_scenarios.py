import json
import unittest
from collections import deque
from dataclasses import dataclass

from scripts.perf.core import StopPolicy
from scripts.perf.http_scenarios import (
    JsonScenario,
    build_l1_scenarios,
    run_scenario_ladder,
    run_scenario_stage,
    sample_json_request,
)

VALID_INTERNAL_TOKEN = "test-internal-auth-token-0123456789abcdef"


@dataclass(frozen=True)
class FakeResponse:
    status: int
    data: dict


class FakeJsonClient:
    def __init__(self, responses=None, error=None):
        self.responses = deque(responses or [FakeResponse(200, {"code": "SUCCESS"})])
        self.error = error
        self.calls = []

    def request_json(self, method, url, *, payload=None, token=None, extra_headers=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "payload": payload,
                "token": token,
                "extra_headers": extra_headers,
            }
        )
        if self.error is not None:
            raise self.error
        if len(self.responses) > 1:
            return self.responses.popleft()
        return self.responses[0]


class HttpScenarioDefinitionTests(unittest.TestCase):
    def test_builds_l1_scenarios_without_registration_and_hides_credentials_from_repr(self):
        scenarios = build_l1_scenarios(
            target_url="https://fusion.example",
            auth_url="https://auth.example",
            email="perf-user@example.com",
            password="password-secret",
            client_id="fusion-client",
            access_token="access-secret",
            internal_auth_token=VALID_INTERNAL_TOKEN,
            conversation_id="conv/unsafe id",
            page_size=25,
        )

        self.assertEqual(set(scenarios), {"auth_login", "models", "conversation_list", "conversation_detail"})
        self.assertEqual(scenarios["auth_login"].method, "POST")
        self.assertEqual(scenarios["auth_login"].url, "https://auth.example/auth/login")
        self.assertNotIn("register", scenarios["auth_login"].url)
        self.assertEqual(
            scenarios["auth_login"].extra_headers,
            {"X-Fusion-Internal-Auth": VALID_INTERNAL_TOKEN},
        )
        self.assertEqual(scenarios["models"].url, "https://fusion.example/api/models/")
        self.assertEqual(
            scenarios["conversation_list"].url,
            "https://fusion.example/api/chat/conversations?page=1&page_size=25",
        )
        self.assertEqual(
            scenarios["conversation_detail"].url,
            "https://fusion.example/api/chat/conversations/conv%2Funsafe%20id",
        )
        serialized_repr = repr(scenarios)
        self.assertNotIn("perf-user@example.com", serialized_repr)
        self.assertNotIn("password-secret", serialized_repr)
        self.assertNotIn("access-secret", serialized_repr)
        self.assertNotIn(VALID_INTERNAL_TOKEN, serialized_repr)

    def test_omits_conversation_detail_when_no_existing_id_is_provided(self):
        scenarios = build_l1_scenarios(
            target_url="https://fusion.example",
            auth_url="https://auth.example",
            email="perf-user@example.com",
            password="password-secret",
            client_id="fusion-client",
            access_token="access-secret",
            internal_auth_token=VALID_INTERNAL_TOKEN,
        )

        self.assertEqual(set(scenarios), {"auth_login", "models", "conversation_list"})

    def test_l1_login_scenario_fails_closed_without_internal_token(self):
        with self.assertRaisesRegex(ValueError, "FUSION_PERF_INTERNAL_AUTH_TOKEN"):
            build_l1_scenarios(
                target_url="https://fusion.example",
                auth_url="https://auth.example",
                email="perf-user@example.com",
                password="password-secret",
                client_id="fusion-client",
                access_token="access-secret",
                internal_auth_token=None,
            )


class HttpScenarioSamplingTests(unittest.TestCase):
    def test_samples_get_without_token_and_never_returns_response_body(self):
        client = FakeJsonClient(
            [FakeResponse(200, {"models": [], "email": "secret@example.com", "access_token": "secret-token"})]
        )
        scenario = JsonScenario(name="models", method="GET", url="https://fusion.example/api/models/")

        sample = sample_json_request(client, scenario)

        self.assertEqual(sample.status, 200)
        self.assertIsNone(sample.error)
        self.assertEqual(client.calls[0]["token"], None)
        self.assertEqual(client.calls[0]["payload"], None)
        self.assertNotIn("secret", repr(sample))

    def test_samples_post_payload_and_authenticated_get(self):
        login_client = FakeJsonClient([FakeResponse(200, {"access_token": "returned-secret"})])
        login = JsonScenario(
            name="auth_login",
            method="POST",
            url="https://auth.example/auth/login",
            payload={"email": "perf@example.com", "password": "secret"},
            extra_headers={"X-Fusion-Internal-Auth": VALID_INTERNAL_TOKEN},
            response_validator=lambda data: bool(data.get("access_token")),
        )
        authenticated_client = FakeJsonClient([FakeResponse(200, {"items": []})])
        conversation_list = JsonScenario(
            name="conversation_list",
            method="GET",
            url="https://fusion.example/api/chat/conversations",
            token="access-secret",
        )
        authenticated_post_client = FakeJsonClient([FakeResponse(201, {"code": "SUCCESS"})])
        authenticated_post = JsonScenario(
            name="authenticated_post",
            method="POST",
            url="https://fusion.example/api/example",
            payload={"value": 1},
            token="access-secret",
            expected_statuses=(201,),
        )

        login_sample = sample_json_request(login_client, login)
        list_sample = sample_json_request(authenticated_client, conversation_list)
        post_sample = sample_json_request(authenticated_post_client, authenticated_post)

        self.assertIsNone(login_sample.error)
        self.assertEqual(login_client.calls[0]["payload"]["email"], "perf@example.com")
        self.assertEqual(
            login_client.calls[0]["extra_headers"],
            {"X-Fusion-Internal-Auth": VALID_INTERNAL_TOKEN},
        )
        self.assertNotIn(VALID_INTERNAL_TOKEN, repr(login))
        self.assertNotIn(VALID_INTERNAL_TOKEN, repr(login_sample))
        self.assertEqual(authenticated_client.calls[0]["token"], "access-secret")
        self.assertIsNone(list_sample.error)
        self.assertEqual(authenticated_post_client.calls[0]["payload"], {"value": 1})
        self.assertEqual(authenticated_post_client.calls[0]["token"], "access-secret")
        self.assertIsNone(post_sample.error)

    def test_marks_unexpected_status_invalid_json_contract_and_timeout_safely(self):
        unexpected = sample_json_request(
            FakeJsonClient([FakeResponse(503, {"detail": "Bearer response-secret"})]),
            JsonScenario(name="models", method="GET", url="https://fusion.example/api/models/"),
        )
        invalid = sample_json_request(
            FakeJsonClient([FakeResponse(200, {"email": "secret@example.com"})]),
            JsonScenario(
                name="auth_login",
                method="POST",
                url="https://auth.example/auth/login",
                response_validator=lambda data: "access_token" in data,
            ),
        )
        timed_out = sample_json_request(
            FakeJsonClient(error=TimeoutError("timeout with secret@example.com")),
            JsonScenario(name="models", method="GET", url="https://fusion.example/api/models/"),
        )

        self.assertEqual(unexpected.error, "http_503")
        self.assertEqual(invalid.error, "invalid_response")
        self.assertEqual(timed_out.error, "timeout")
        self.assertTrue(timed_out.timed_out)
        self.assertNotIn("secret", json.dumps([unexpected.__dict__, invalid.__dict__, timed_out.__dict__]))


class HttpScenarioStageTests(unittest.TestCase):
    def test_stage_and_ladder_return_only_safe_aggregates(self):
        client = FakeJsonClient(
            [FakeResponse(200, {"access_token": "response-secret", "email": "perf-user@example.com"})]
        )
        scenario = JsonScenario(
            name="conversation_list",
            method="GET",
            url="https://fusion.example/api/chat/conversations",
            token="access-secret",
        )

        stage, consecutive = run_scenario_stage(client, scenario, concurrency=2, requests=4)
        ladder = run_scenario_ladder(
            client,
            scenario,
            concurrencies=[1, 2],
            requests_per_stage=2,
            stop_policy=StopPolicy(min_samples=1, max_error_rate=1.0, max_timeout_rate=1.0),
        )

        self.assertEqual(stage["scenario"], "conversation_list")
        self.assertEqual(stage["kind"], "http")
        self.assertEqual(stage["concurrency"], 2)
        self.assertEqual(stage["requests"], 4)
        self.assertEqual(stage["successful"], 4)
        self.assertEqual(consecutive, 0)
        self.assertEqual(len(ladder["stages"]), 2)
        self.assertFalse(ladder["stopped"])
        serialized = json.dumps({"stage": stage, "ladder": ladder})
        self.assertNotIn("response-secret", serialized)
        self.assertNotIn("perf-user@example.com", serialized)
        self.assertNotIn("access-secret", serialized)
        self.assertNotIn("https://", serialized)

    def test_ladder_stops_after_policy_failure_without_exposing_exception(self):
        client = FakeJsonClient(error=RuntimeError("database failed for secret@example.com"))
        scenario = JsonScenario(name="models", method="GET", url="https://fusion.example/api/models/")

        result = run_scenario_ladder(
            client,
            scenario,
            concurrencies=[1, 5, 10],
            requests_per_stage=2,
            stop_policy=StopPolicy(min_samples=1, max_error_rate=0.5, max_timeout_rate=0.5),
        )

        self.assertTrue(result["stopped"])
        self.assertEqual(len(result["stages"]), 1)
        self.assertEqual(result["stop_reasons"], ["models:error_rate"])
        self.assertNotIn("secret@example.com", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
