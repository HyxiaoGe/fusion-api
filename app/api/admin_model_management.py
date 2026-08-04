from fastapi import APIRouter, Depends, Header, Request, Response, status

from app.api.deps import (
    get_current_admin_user,
    get_model_management_service,
    require_model_admission_worker,
)
from app.db.models import User
from app.schemas.model_management import (
    AdmissionCompletionRequest,
    CandidateAdmissionRequest,
    ModelVisibilityRequest,
)
from app.schemas.response import success
from app.services.model_management_service import ModelManagementService

router = APIRouter()
internal_router = APIRouter()


@router.get("")
def get_model_management_snapshot(
    request: Request,
    _admin: User = Depends(get_current_admin_user),
    service: ModelManagementService = Depends(get_model_management_service),
):
    return success(data=service.get_snapshot(), request_id=request.state.request_id)


@router.patch("/models/visibility")
def update_model_visibility(
    payload: ModelVisibilityRequest,
    request: Request,
    admin: User = Depends(get_current_admin_user),
    service: ModelManagementService = Depends(get_model_management_service),
):
    row = service.set_visibility(
        model_id=payload.model_id,
        selectable=payload.selectable,
        expected_revision=payload.expected_revision,
        reason=payload.reason,
        admin=admin,
        request_id=request.state.request_id,
    )
    return success(
        data={
            "model_id": row.model_id,
            "selectable": bool(row.selectable),
            "routable": True,
            "revision": row.revision,
            "reason": row.reason,
            "updated_at": row.updated_at,
        },
        request_id=request.state.request_id,
    )


@router.post("/candidates/{fingerprint}/admit", status_code=status.HTTP_202_ACCEPTED)
def request_candidate_admission(
    fingerprint: str,
    payload: CandidateAdmissionRequest,
    request: Request,
    admin: User = Depends(get_current_admin_user),
    service: ModelManagementService = Depends(get_model_management_service),
):
    operation = service.request_admission(
        fingerprint=fingerprint,
        model_id=payload.model_id,
        expected_run_id=payload.expected_run_id,
        reason=payload.reason,
        admin=admin,
        request_id=request.state.request_id,
    )
    return success(data=service.operation_payload(operation), request_id=request.state.request_id)


@internal_router.post("/admissions/claim")
def claim_candidate_admission(
    response: Response,
    _worker: None = Depends(require_model_admission_worker),
    service: ModelManagementService = Depends(get_model_management_service),
):
    claimed = service.claim_operation()
    if claimed is None:
        response.status_code = status.HTTP_204_NO_CONTENT
        return None
    operation, lease_token = claimed
    return {
        "operation_id": operation.id,
        "lease_token": lease_token,
        "run_id": operation.governance_run_id,
        "candidate_fingerprint": operation.candidate_fingerprint,
        "model_id": operation.model_id,
    }


@internal_router.post("/admissions/{operation_id}/invalidate-catalog")
def invalidate_catalog_for_admission(
    operation_id: str,
    operation_lease: str = Header(..., alias="X-Operation-Lease"),
    _worker: None = Depends(require_model_admission_worker),
    service: ModelManagementService = Depends(get_model_management_service),
):
    generation = service.invalidate_catalog(operation_id=operation_id, lease_token=operation_lease)
    return {"generation": generation}


@internal_router.post("/admissions/{operation_id}/complete")
def complete_candidate_admission(
    operation_id: str,
    payload: AdmissionCompletionRequest,
    operation_lease: str = Header(..., alias="X-Operation-Lease"),
    _worker: None = Depends(require_model_admission_worker),
    service: ModelManagementService = Depends(get_model_management_service),
):
    operation = service.complete_operation(
        operation_id=operation_id,
        lease_token=operation_lease,
        status=payload.status,
        phase=payload.phase,
        error_code=payload.error_code,
        completed_phases=payload.completed_phases,
        writes_performed=payload.writes_performed,
        compensation=payload.compensation,
    )
    return service.operation_payload(operation)
