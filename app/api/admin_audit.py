"""独立、只读、可审计的管理员内容观察 API。"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Header, Query, Request

from app.api.deps import get_admin_audit_service, get_conversation_auditor
from app.db.models import User
from app.schemas.admin_audit import AdminPerformanceRunImport
from app.schemas.response import success
from app.services.admin_audit_service import AdminAuditService

router = APIRouter()


def _context(request: Request, reason: str | None) -> dict:
    return {"request_id": request.state.request_id, "reason": reason}


@router.get("/users")
def list_users(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    q: str | None = Query(None, max_length=200),
    is_superuser: bool | None = None,
    created_from: datetime | None = None,
    created_to: datetime | None = None,
    reason: str | None = Header(None, alias="X-Admin-Audit-Reason", max_length=300),
    service: AdminAuditService = Depends(get_admin_audit_service),
    auditor: User = Depends(get_conversation_auditor),
):
    data = service.list_users(
        admin=auditor,
        **_context(request, reason),
        page=page,
        page_size=page_size,
        query=q,
        is_superuser=is_superuser,
        created_from=created_from,
        created_to=created_to,
    )
    return success(data=data, request_id=request.state.request_id)


@router.get("/users/{user_id}")
def get_user(
    user_id: str,
    request: Request,
    reason: str | None = Header(None, alias="X-Admin-Audit-Reason", max_length=300),
    service: AdminAuditService = Depends(get_admin_audit_service),
    auditor: User = Depends(get_conversation_auditor),
):
    return success(
        data=service.get_user(user_id, admin=auditor, **_context(request, reason)),
        request_id=request.state.request_id,
    )


@router.get("/conversations")
def list_conversations(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    q: str | None = Query(None, max_length=200),
    user_id: str | None = None,
    model_id: str | None = None,
    has_tools: bool | None = None,
    has_files: bool | None = None,
    created_from: datetime | None = None,
    created_to: datetime | None = None,
    updated_from: datetime | None = None,
    updated_to: datetime | None = None,
    reason: str | None = Header(None, alias="X-Admin-Audit-Reason", max_length=300),
    service: AdminAuditService = Depends(get_admin_audit_service),
    auditor: User = Depends(get_conversation_auditor),
):
    data = service.list_conversations(
        admin=auditor,
        **_context(request, reason),
        page=page,
        page_size=page_size,
        query=q,
        user_id=user_id,
        model_id=model_id,
        has_tools=has_tools,
        has_files=has_files,
        created_from=created_from,
        created_to=created_to,
        updated_from=updated_from,
        updated_to=updated_to,
    )
    return success(data=data, request_id=request.state.request_id)


@router.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    request: Request,
    reason: str | None = Header(None, alias="X-Admin-Audit-Reason", max_length=300),
    service: AdminAuditService = Depends(get_admin_audit_service),
    auditor: User = Depends(get_conversation_auditor),
):
    return success(
        data=service.get_conversation(conversation_id, admin=auditor, **_context(request, reason)),
        request_id=request.state.request_id,
    )


def _conversation_page(
    method_name: str,
    conversation_id: str,
    request: Request,
    page: int,
    page_size: int,
    reason: str | None,
    service: AdminAuditService,
    auditor: User,
):
    method = getattr(service, method_name)
    return success(
        data=method(
            conversation_id,
            page=page,
            page_size=page_size,
            admin=auditor,
            **_context(request, reason),
        ),
        request_id=request.state.request_id,
    )


@router.get("/conversations/{conversation_id}/messages")
def list_messages(
    conversation_id: str,
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    reason: str | None = Header(None, alias="X-Admin-Audit-Reason", max_length=300),
    service: AdminAuditService = Depends(get_admin_audit_service),
    auditor: User = Depends(get_conversation_auditor),
):
    return _conversation_page("list_messages", conversation_id, request, page, page_size, reason, service, auditor)


@router.get("/conversations/{conversation_id}/tool-calls")
def list_tool_calls(
    conversation_id: str,
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    reason: str | None = Header(None, alias="X-Admin-Audit-Reason", max_length=300),
    service: AdminAuditService = Depends(get_admin_audit_service),
    auditor: User = Depends(get_conversation_auditor),
):
    return _conversation_page("list_tool_calls", conversation_id, request, page, page_size, reason, service, auditor)


@router.get("/conversations/{conversation_id}/agent-runs")
def list_agent_runs(
    conversation_id: str,
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    reason: str | None = Header(None, alias="X-Admin-Audit-Reason", max_length=300),
    service: AdminAuditService = Depends(get_admin_audit_service),
    auditor: User = Depends(get_conversation_auditor),
):
    return _conversation_page("list_agent_runs", conversation_id, request, page, page_size, reason, service, auditor)


@router.get("/conversations/{conversation_id}/files")
def list_files(
    conversation_id: str,
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    reason: str | None = Header(None, alias="X-Admin-Audit-Reason", max_length=300),
    service: AdminAuditService = Depends(get_admin_audit_service),
    auditor: User = Depends(get_conversation_auditor),
):
    return _conversation_page("list_files", conversation_id, request, page, page_size, reason, service, auditor)


@router.get("/events")
def list_events(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    admin_user_id: str | None = None,
    target_user_id: str | None = None,
    action: str | None = None,
    resource_type: str | None = None,
    created_from: datetime | None = None,
    created_to: datetime | None = None,
    reason: str | None = Header(None, alias="X-Admin-Audit-Reason", max_length=300),
    service: AdminAuditService = Depends(get_admin_audit_service),
    auditor: User = Depends(get_conversation_auditor),
):
    data = service.list_audit_events(
        admin=auditor,
        **_context(request, reason),
        page=page,
        page_size=page_size,
        admin_user_id=admin_user_id,
        target_user_id=target_user_id,
        action=action,
        resource_type=resource_type,
        created_from=created_from,
        created_to=created_to,
    )
    return success(data=data, request_id=request.state.request_id)


@router.post("/performance-runs/import")
def import_performance_run(
    payload: AdminPerformanceRunImport,
    request: Request,
    reason: str | None = Header(None, alias="X-Admin-Audit-Reason", max_length=300),
    service: AdminAuditService = Depends(get_admin_audit_service),
    auditor: User = Depends(get_conversation_auditor),
):
    return success(
        data=service.import_performance_run(payload, admin=auditor, **_context(request, reason)),
        request_id=request.state.request_id,
    )


@router.get("/performance-runs")
def list_performance_runs(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    environment: str | None = None,
    status: str | None = None,
    model_id: str | None = None,
    reason: str | None = Header(None, alias="X-Admin-Audit-Reason", max_length=300),
    service: AdminAuditService = Depends(get_admin_audit_service),
    auditor: User = Depends(get_conversation_auditor),
):
    data = service.list_performance_runs(
        admin=auditor,
        **_context(request, reason),
        page=page,
        page_size=page_size,
        environment=environment,
        status=status,
        model_id=model_id,
    )
    return success(data=data, request_id=request.state.request_id)


@router.get("/performance-runs/{run_id}")
def get_performance_run(
    run_id: str,
    request: Request,
    reason: str | None = Header(None, alias="X-Admin-Audit-Reason", max_length=300),
    service: AdminAuditService = Depends(get_admin_audit_service),
    auditor: User = Depends(get_conversation_auditor),
):
    return success(
        data=service.get_performance_run(run_id, admin=auditor, **_context(request, reason)),
        request_id=request.state.request_id,
    )
