from __future__ import annotations

from typing import Any

from sqlalchemy import update as sqlalchemy_update
from sqlalchemy.orm import Session

from app.db.models import ModelCatalogControl


class ModelCatalogControlRepository:
    """模型选择器控制记录的数据访问层；事务由服务层统一提交。"""

    def __init__(self, db: Session):
        self.db = db

    def get(self, model_id: str) -> ModelCatalogControl | None:
        return self.db.query(ModelCatalogControl).filter(ModelCatalogControl.model_id == model_id).first()

    def get_by_model_ids(self, model_ids: list[str]) -> dict[str, ModelCatalogControl]:
        normalized = sorted(set(model_ids))
        if not normalized:
            return {}
        rows = self.db.query(ModelCatalogControl).filter(ModelCatalogControl.model_id.in_(normalized)).all()
        return {row.model_id: row for row in rows}

    def add(self, values: dict[str, Any]) -> ModelCatalogControl:
        row = ModelCatalogControl(**values)
        self.db.add(row)
        self.db.flush()
        return row

    def update_if_revision(
        self,
        *,
        model_id: str,
        expected_revision: int,
        values: dict[str, Any],
    ) -> ModelCatalogControl | None:
        statement = (
            sqlalchemy_update(ModelCatalogControl)
            .where(
                ModelCatalogControl.model_id == model_id,
                ModelCatalogControl.revision == expected_revision,
            )
            .values(
                **{
                    **values,
                    "routable": True,
                    "revision": ModelCatalogControl.revision + 1,
                }
            )
        )
        result = self.db.execute(statement)
        if result.rowcount != 1:
            return None
        self.db.expire_all()
        return self.get(model_id)
