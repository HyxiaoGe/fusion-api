import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import ValidationError

from app.schemas.response import ApiException


class AgentContextRequestSchemaTests(unittest.TestCase):
    def test_provided_requires_strict_location_and_non_provided_requires_reason(self):
        from app.schemas.chat import AgentContextResultRequest

        provided = AgentContextResultRequest(
            context_type="geolocation",
            status="provided",
            location={
                "latitude": 22.616,
                "longitude": 114.031,
                "accuracy_m": 25,
                "acquired_at": 100,
            },
        )
        self.assertEqual(provided.location.latitude, 22.616)

        timeout = AgentContextResultRequest(
            context_type="geolocation",
            status="timeout",
            reason="client_timeout",
        )
        self.assertEqual(timeout.status, "timeout")

        invalid_payloads = (
            {"context_type": "geolocation", "status": "provided"},
            {"context_type": "geolocation", "status": "denied"},
            {
                "context_type": "geolocation",
                "status": "denied",
                "reason": "permission_denied",
                "location": {"latitude": 0, "longitude": 0, "accuracy_m": 1, "acquired_at": 1},
            },
            {
                "context_type": "geolocation",
                "status": "provided",
                "location": {"latitude": 91, "longitude": 0, "accuracy_m": 1, "acquired_at": 1},
            },
            {"status": "timeout", "reason": "client_timeout"},
            {"context_type": "location", "status": "timeout", "reason": "client_timeout"},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                AgentContextResultRequest(**payload)


class ChatContextServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_context_validates_conversation_and_maps_broker_outcomes(self):
        from app.services.chat_service import ChatService

        service = ChatService(MagicMock())
        service.conversation_service.get_conversation = MagicMock(return_value=SimpleNamespace(id="conv-1"))

        with patch(
            "app.services.chat_service.submit_context_result",
            new=AsyncMock(
                return_value=SimpleNamespace(
                    outcome="accepted",
                    request_id="ctx-1",
                    context_type="geolocation",
                    status="denied",
                    model_dump=lambda mode=None: {
                        "outcome": "accepted",
                        "request_id": "ctx-1",
                        "context_type": "geolocation",
                        "status": "denied",
                    },
                )
            ),
        ) as broker:
            result = await service.submit_agent_context_result(
                conversation_id="conv-1",
                run_id="run-1",
                request_id="ctx-1",
                user_id="user-1",
                status="denied",
                location=None,
                reason="permission_denied",
            )

        self.assertEqual(result["status"], "denied")
        broker.assert_awaited_once()

        mappings = (("conflict", 409), ("expired", 410), ("stale", 409), ("not_found", 404), ("forbidden", 404))
        for outcome, status_code in mappings:
            with self.subTest(outcome=outcome):
                submission = SimpleNamespace(outcome=outcome)
                with patch(
                    "app.services.chat_service.submit_context_result",
                    new=AsyncMock(return_value=submission),
                ):
                    with self.assertRaises(ApiException) as raised:
                        await service.submit_agent_context_result(
                            conversation_id="conv-1",
                            run_id="run-1",
                            request_id="ctx-1",
                            user_id="user-1",
                            status="denied",
                            location=None,
                            reason="permission_denied",
                        )
                self.assertEqual(raised.exception.status_code, status_code)

    async def test_submit_context_route_delegates_authenticated_user_and_never_echoes_location(self):
        from app.api.chat import submit_agent_context
        from app.schemas.chat import AgentContextResultRequest

        request = SimpleNamespace(state=SimpleNamespace(request_id="http-1"))
        service = SimpleNamespace(
            submit_agent_context_result=AsyncMock(
                return_value={
                    "outcome": "accepted",
                    "request_id": "ctx-1",
                    "context_type": "geolocation",
                    "status": "provided",
                }
            )
        )
        body = AgentContextResultRequest(
            context_type="geolocation",
            status="provided",
            location={"latitude": 22.616, "longitude": 114.031, "accuracy_m": 25, "acquired_at": 100},
        )

        response = await submit_agent_context(
            conversation_id="conv-1",
            run_id="run-1",
            context_request_id="ctx-1",
            context_request=body,
            request=request,
            chat_service=service,
            current_user=SimpleNamespace(id="user-1"),
        )

        self.assertEqual(response.data["status"], "provided")
        self.assertNotIn("latitude", str(response.data))
        service.submit_agent_context_result.assert_awaited_once_with(
            conversation_id="conv-1",
            run_id="run-1",
            request_id="ctx-1",
            user_id="user-1",
            status="provided",
            location=body.location.model_dump(mode="json"),
            reason=None,
        )


if __name__ == "__main__":
    unittest.main()
