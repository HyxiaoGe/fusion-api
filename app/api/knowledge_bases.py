from fastapi import APIRouter, Depends, File, Query, Request, UploadFile

from app.api.deps import get_current_user, get_knowledge_service
from app.db.models import User
from app.schemas.knowledge import (
    KnowledgeBaseCreate,
    KnowledgeBaseEnvelope,
    KnowledgeBasePage,
    KnowledgeBaseUpdate,
    KnowledgeDocumentEnvelope,
    KnowledgeDocumentPage,
    KnowledgeDocumentUploadResult,
    KnowledgeRetrievalRequest,
    KnowledgeRetrievalResult,
    KnowledgeTaskEnvelope,
)
from app.schemas.response import ApiResponse, success
from app.services.knowledge.service import KnowledgeService

router = APIRouter()

ERROR_RESPONSES = {
    400: {"model": ApiResponse[None], "description": "参数或文档格式无效"},
    401: {"model": ApiResponse[None], "description": "未认证"},
    404: {"model": ApiResponse[None], "description": "资源不存在或无权访问"},
    409: {"model": ApiResponse[None], "description": "资源状态或重复内容冲突"},
    413: {"model": ApiResponse[None], "description": "文档超过大小上限"},
    503: {"model": ApiResponse[None], "description": "知识库未启用或依赖不可用"},
}


@router.post(
    "/",
    status_code=201,
    response_model=ApiResponse[KnowledgeBaseEnvelope],
    responses=ERROR_RESPONSES,
)
def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    row = service.create_knowledge_base(current_user.id, payload)
    return success(data={"knowledge_base": row}, message="知识库已创建", request_id=request.state.request_id)


@router.get("/", response_model=ApiResponse[KnowledgeBasePage], responses=ERROR_RESPONSES)
def list_knowledge_bases(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    result = service.list_knowledge_bases(current_user.id, page=page, page_size=page_size)
    return success(data=result, request_id=request.state.request_id)


@router.get(
    "/{knowledge_base_id}",
    response_model=ApiResponse[KnowledgeBaseEnvelope],
    responses=ERROR_RESPONSES,
)
def get_knowledge_base(
    knowledge_base_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    row = service.get_knowledge_base(current_user.id, knowledge_base_id)
    return success(data={"knowledge_base": row}, request_id=request.state.request_id)


@router.patch(
    "/{knowledge_base_id}",
    response_model=ApiResponse[KnowledgeBaseEnvelope],
    responses=ERROR_RESPONSES,
)
def update_knowledge_base(
    knowledge_base_id: str,
    payload: KnowledgeBaseUpdate,
    request: Request,
    current_user: User = Depends(get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    row = service.update_knowledge_base(current_user.id, knowledge_base_id, payload)
    return success(data={"knowledge_base": row}, message="知识库已更新", request_id=request.state.request_id)


@router.delete(
    "/{knowledge_base_id}",
    status_code=202,
    response_model=ApiResponse[KnowledgeTaskEnvelope],
    responses=ERROR_RESPONSES,
)
def delete_knowledge_base(
    knowledge_base_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    task = service.delete_knowledge_base(current_user.id, knowledge_base_id)
    return success(data={"task": task}, message="知识库删除任务已排队", request_id=request.state.request_id)


@router.post(
    "/{knowledge_base_id}/documents",
    status_code=202,
    response_model=ApiResponse[KnowledgeDocumentUploadResult],
    responses=ERROR_RESPONSES,
)
async def upload_knowledge_document(
    knowledge_base_id: str,
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    result = await service.upload_document(current_user.id, knowledge_base_id, file)
    return success(data=result, message="文档已上传并进入索引队列", request_id=request.state.request_id)


@router.get(
    "/{knowledge_base_id}/documents",
    response_model=ApiResponse[KnowledgeDocumentPage],
    responses=ERROR_RESPONSES,
)
def list_knowledge_documents(
    knowledge_base_id: str,
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    result = service.list_documents(
        current_user.id,
        knowledge_base_id,
        page=page,
        page_size=page_size,
    )
    return success(data=result, request_id=request.state.request_id)


@router.get(
    "/{knowledge_base_id}/documents/{document_id}",
    response_model=ApiResponse[KnowledgeDocumentEnvelope],
    responses=ERROR_RESPONSES,
)
def get_knowledge_document(
    knowledge_base_id: str,
    document_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    row = service.get_document(current_user.id, knowledge_base_id, document_id)
    return success(data={"document": row}, request_id=request.state.request_id)


@router.delete(
    "/{knowledge_base_id}/documents/{document_id}",
    status_code=202,
    response_model=ApiResponse[KnowledgeTaskEnvelope],
    responses=ERROR_RESPONSES,
)
def delete_knowledge_document(
    knowledge_base_id: str,
    document_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    task = service.delete_document(current_user.id, knowledge_base_id, document_id)
    return success(data={"task": task}, message="文档删除任务已排队", request_id=request.state.request_id)


@router.post(
    "/{knowledge_base_id}/documents/{document_id}/retry",
    status_code=202,
    response_model=ApiResponse[KnowledgeTaskEnvelope],
    responses=ERROR_RESPONSES,
)
def retry_knowledge_document(
    knowledge_base_id: str,
    document_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    task = service.retry_document(current_user.id, knowledge_base_id, document_id)
    return success(data={"task": task}, message="文档索引重试已排队", request_id=request.state.request_id)


@router.post(
    "/{knowledge_base_id}/documents/{document_id}/rebuild",
    status_code=202,
    response_model=ApiResponse[KnowledgeTaskEnvelope],
    responses=ERROR_RESPONSES,
)
def rebuild_knowledge_document(
    knowledge_base_id: str,
    document_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    task = service.rebuild_document(current_user.id, knowledge_base_id, document_id)
    return success(data={"task": task}, message="文档索引重建已排队", request_id=request.state.request_id)


@router.post(
    "/search",
    response_model=ApiResponse[KnowledgeRetrievalResult],
    responses=ERROR_RESPONSES,
)
async def search_knowledge_bases(
    payload: KnowledgeRetrievalRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    result = await service.retrieve(current_user.id, payload)
    return success(data=result, request_id=request.state.request_id)


@router.get(
    "/tasks/{task_id}",
    response_model=ApiResponse[KnowledgeTaskEnvelope],
    responses=ERROR_RESPONSES,
)
def get_knowledge_task(
    task_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    task = service.get_task(current_user.id, task_id)
    return success(data={"task": task}, request_id=request.state.request_id)
