import json
from typing import Optional


class StreamSerializer:
    @staticmethod
    def _serialize(payload) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @staticmethod
    def _chunk(
        message_id: str,
        conversation_id: str,
        delta: dict,
        finish_reason=None,
        error_message: Optional[str] = None,
    ) -> str:
        payload = {
            "id": message_id,
            "conversation_id": conversation_id,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if error_message is not None:
            payload["error"] = {"message": error_message}
        return StreamSerializer._serialize(payload)

    @staticmethod
    def content_chunk(message_id: str, conversation_id: str, content: str) -> str:
        return StreamSerializer._chunk(message_id, conversation_id, {"content": content})

    @staticmethod
    def init_chunk(message_id: str, conversation_id: str) -> str:
        return StreamSerializer._chunk(message_id, conversation_id, {})

    @staticmethod
    def reasoning_chunk(message_id: str, conversation_id: str, reasoning_content: str) -> str:
        return StreamSerializer._chunk(
            message_id,
            conversation_id,
            {"reasoning_content": reasoning_content},
        )

    @staticmethod
    def finish_chunk(message_id: str, conversation_id: str, finish_reason: str = "stop") -> str:
        return StreamSerializer._chunk(message_id, conversation_id, {}, finish_reason=finish_reason)

    @staticmethod
    def error_chunk(message_id: str, conversation_id: str, error_message: str) -> str:
        return StreamSerializer._chunk(
            message_id,
            conversation_id,
            {},
            finish_reason="error",
            error_message=error_message,
        )

    @staticmethod
    def done_marker() -> str:
        return "data: [DONE]\n\n"
