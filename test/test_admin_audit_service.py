from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

from app.schemas.response import ApiException
from app.services.admin_audit_service import AdminAuditService


def test_audit_event_list_batches_target_users_and_preserves_historical_snapshots():
    repository = Mock()
    repository.list_audit_events.return_value = (
        [
            SimpleNamespace(
                id="event-live",
                admin_user_id="admin-old-id",
                admin_snapshot={
                    "id": "admin-old-id",
                    "username": "historical-admin",
                    "email_masked": "h***@example.com",
                    "email": "historical-admin-private@example.com",
                    "unknown_pii": "historical-admin-id-card",
                },
                action="admin.audit.user.view",
                resource_type="user",
                resource_id="user-1",
                target_user_id="user-1",
                request_id="request-old-live",
                reason=None,
                extra_metadata={
                    "page": 1,
                    "model_id": "gpt-5",
                    "query": {"present": True, "length": 17, "raw": "alice-private@example.com"},
                    "customer_email": "customer-private@example.com",
                    "future_secret": "future-private-value",
                },
                created_at=datetime(2026, 7, 11, 12, 0, 0),
            ),
            SimpleNamespace(
                id="event-deleted",
                admin_user_id="admin-old-id",
                admin_snapshot={"id": "admin-old-id", "username": "historical-admin"},
                action="admin.audit.user.view",
                resource_type="user",
                resource_id="deleted-user",
                target_user_id="deleted-user",
                request_id="request-old-deleted",
                reason=None,
                extra_metadata={},
                created_at=datetime(2026, 7, 11, 11, 0, 0),
            ),
        ],
        2,
    )
    repository.get_users_by_ids.return_value = {
        "user-1": SimpleNamespace(
            id="user-1",
            username="alice",
            nickname="Alice",
            email="alice@example.com",
        )
    }
    service = AdminAuditService(repository)

    result = service.list_audit_events(
        admin=SimpleNamespace(id="admin-current", username="current-admin", email="current@example.com"),
        request_id="request-list",
        reason=None,
        page=1,
        page_size=25,
        admin_user_id=None,
        target_user_id=None,
        action=None,
        resource_type=None,
        created_from=None,
        created_to=None,
    )

    repository.get_users_by_ids.assert_called_once_with(["deleted-user", "user-1"])
    repository.create_audit_event.assert_called_once()
    assert result["items"][0]["target_user"] == {
        "id": "user-1",
        "username": "alice",
        "nickname": "Alice",
    }
    assert result["items"][0]["admin_snapshot"]["username"] == "historical-admin"
    assert result["items"][0]["admin_snapshot"] == {
        "id": "admin-old-id",
        "username": "historical-admin",
    }
    assert "email" not in result["items"][0]["admin_snapshot"]
    assert "email_masked" not in result["items"][0]["admin_snapshot"]
    assert "email" not in result["items"][0]["target_user"]
    assert "email_masked" not in result["items"][0]["target_user"]
    assert result["items"][0]["metadata"] == {
        "page": 1,
        "model_id": "gpt-5",
        "query": {"present": True, "length": 17},
    }
    assert result["items"][1]["target_user_id"] == "deleted-user"
    assert result["items"][1]["target_user"] is None


def test_model_detail_rejects_control_character_id_before_catalog_or_database_access():
    repository = Mock()
    service = AdminAuditService(repository)

    try:
        service.get_model(
            "bad\nmodel",
            admin=SimpleNamespace(id="admin-1", username="root", email="root@example.com"),
            request_id="request-invalid-model",
            reason=None,
        )
    except ApiException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("控制字符模型 ID 应被拒绝")

    repository.list_model_operation_stats.assert_not_called()
