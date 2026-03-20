import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from app.api import chat as chat_api
from app.schemas.chat import ChatRequest
from app.services.chat_service import FunctionCallsNotSupportedError


class ChatApiTests(unittest.TestCase):
    def test_send_message_maps_function_call_error_to_http_400(self):
        request = ChatRequest(provider="qwen", model="qwen-max", message="hello", stream=True)

        with patch(
            "app.api.chat.ChatService.process_message",
            new=AsyncMock(side_effect=FunctionCallsNotSupportedError("工具调用功能当前未启用")),
        ):
            with self.assertRaises(HTTPException) as context:
                asyncio.run(
                    chat_api.send_message(
                        request,
                        db=MagicMock(),
                        current_user=SimpleNamespace(id="user-1"),
                    )
                )

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.detail, "工具调用功能当前未启用")
