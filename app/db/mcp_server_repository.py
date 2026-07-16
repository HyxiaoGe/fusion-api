from typing import Any

from sqlalchemy import update as sqlalchemy_update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import McpServer


class McpServerRepository:
    def __init__(self, db: Session):
        self.db = db

    def list_all(self) -> list[McpServer]:
        return self.db.query(McpServer).order_by(McpServer.created_at.desc(), McpServer.id.desc()).all()

    def get(self, server_id: str) -> McpServer | None:
        return (
            self.db.query(McpServer).execution_options(populate_existing=True).filter(McpServer.id == server_id).first()
        )

    def get_by_name(self, name: str) -> McpServer | None:
        return self.db.query(McpServer).filter(McpServer.name == name).first()

    def create(self, values: dict[str, Any]) -> McpServer:
        row = McpServer(**values)
        self.db.add(row)
        self._commit()
        self.db.refresh(row)
        return row

    def update(self, row: McpServer, values: dict[str, Any]) -> McpServer:
        statement = (
            sqlalchemy_update(McpServer)
            .where(McpServer.id == row.id)
            .values(**values, config_version=McpServer.config_version + 1)
        )
        self.db.execute(statement)
        self._commit()
        self.db.expire_all()
        return self.get(row.id)

    def update_if_version(
        self,
        server_id: str,
        expected_version: int,
        values: dict[str, Any],
    ) -> McpServer | None:
        """仅在配置版本未变化时保存远端检测结果，避免覆盖并发管理操作。"""

        statement = (
            sqlalchemy_update(McpServer)
            .where(McpServer.id == server_id, McpServer.config_version == expected_version)
            .values(**values, config_version=McpServer.config_version + 1)
        )
        result = self.db.execute(statement)
        if result.rowcount != 1:
            self.db.rollback()
            self.db.expire_all()
            return None
        self._commit()
        self.db.expire_all()
        return self.get(server_id)

    def _commit(self) -> None:
        try:
            self.db.commit()
        except IntegrityError:
            self.db.rollback()
            raise
