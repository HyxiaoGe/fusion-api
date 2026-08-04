from __future__ import annotations

from datetime import datetime

from sqlalchemy import case, or_
from sqlalchemy.orm import Session

from app.db.models import ModelAdmissionOperation


class ModelAdmissionOperationRepository:
    """模型准入操作队列；提交和审计事务由服务层控制。"""

    def __init__(self, db: Session):
        self.db = db

    def get(self, operation_id: str) -> ModelAdmissionOperation | None:
        return (
            self.db.query(ModelAdmissionOperation)
            .execution_options(populate_existing=True)
            .filter(ModelAdmissionOperation.id == operation_id)
            .first()
        )

    def get_by_candidate_run(self, fingerprint: str, run_id: str) -> ModelAdmissionOperation | None:
        return (
            self.db.query(ModelAdmissionOperation)
            .filter(
                ModelAdmissionOperation.candidate_fingerprint == fingerprint,
                ModelAdmissionOperation.governance_run_id == run_id,
            )
            .first()
        )

    def list_recent(self, limit: int = 100) -> list[ModelAdmissionOperation]:
        return (
            self.db.query(ModelAdmissionOperation)
            .order_by(
                ModelAdmissionOperation.updated_at.desc(),
                ModelAdmissionOperation.created_at.desc(),
                ModelAdmissionOperation.id.desc(),
            )
            .limit(max(1, min(limit, 100)))
            .all()
        )

    def add(self, row: ModelAdmissionOperation) -> ModelAdmissionOperation:
        self.db.add(row)
        self.db.flush()
        return row

    def lock_claimable(self, now: datetime) -> ModelAdmissionOperation | None:
        return (
            self.db.query(ModelAdmissionOperation)
            .filter(
                or_(
                    ModelAdmissionOperation.status == "pending",
                    (
                        (ModelAdmissionOperation.status == "running")
                        & (
                            ModelAdmissionOperation.lease_expires_at.is_(None)
                            | (ModelAdmissionOperation.lease_expires_at <= now)
                        )
                    ),
                )
            )
            .order_by(
                case((ModelAdmissionOperation.status == "running", 0), else_=1),
                ModelAdmissionOperation.created_at.asc(),
                ModelAdmissionOperation.id.asc(),
            )
            .with_for_update(skip_locked=True)
            .first()
        )

    def has_active_running(self, now: datetime) -> bool:
        return (
            self.db.query(ModelAdmissionOperation.id)
            .filter(
                ModelAdmissionOperation.status == "running",
                ModelAdmissionOperation.lease_expires_at > now,
            )
            .with_for_update(skip_locked=True)
            .first()
            is not None
        )

    def lock(self, operation_id: str) -> ModelAdmissionOperation | None:
        return (
            self.db.query(ModelAdmissionOperation)
            .filter(ModelAdmissionOperation.id == operation_id)
            .with_for_update()
            .first()
        )
