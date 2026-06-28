"""Agent progress snapshot 旁路记录器。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.core.logger import app_logger
from app.db.models import AgentProgressSnapshot
from app.services.agent.progress_state import apply_progress_event, empty_progress_state


class AgentProgressRecorder:
    """把 v2 agent_event 折叠成 compact snapshot 并写入数据库。"""

    def __init__(
        self,
        *,
        db: Any,
        run_id: str,
        conversation_id: str,
        message_id: str,
        user_id: str,
        logger: Any = app_logger,
    ) -> None:
        self.db = db
        self.run_id = run_id
        self.conversation_id = conversation_id
        self.message_id = message_id
        self.user_id = user_id
        self.logger = logger
        self._state = empty_progress_state(run_id=run_id, message_id=message_id)

    def record_chunk(self, conversation_id: str, chunk_type: str, payload: dict[str, Any]) -> None:
        if conversation_id != self.conversation_id or chunk_type != "agent_event":
            return

        next_state = apply_progress_event(self._state, payload)
        if next_state is self._state:
            return

        self._state = next_state
        self._upsert_snapshot()

    def _upsert_snapshot(self) -> None:
        try:
            row = self.db.query(AgentProgressSnapshot).filter(AgentProgressSnapshot.run_id == self.run_id).first()
            if row is None:
                row = AgentProgressSnapshot(
                    run_id=self.run_id,
                    conversation_id=self.conversation_id,
                    message_id=self.message_id,
                    user_id=self.user_id,
                    protocol_version=2,
                    state=deepcopy(self._state),
                )
                self.db.add(row)
            else:
                row.conversation_id = self.conversation_id
                row.message_id = self.message_id
                row.user_id = self.user_id
                row.protocol_version = 2
                row.state = deepcopy(self._state)
            self.db.commit()
        except Exception as error:  # pragma: no cover - 日志内容不影响业务断言
            rollback = getattr(self.db, "rollback", None)
            if rollback is not None:
                rollback()
            self.logger.warning(f"Agent progress snapshot 写入失败: run_id={self.run_id}, error={error}")
