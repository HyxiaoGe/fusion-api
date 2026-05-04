"""用户级 API key 管理（BYOK）"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.api.deps import (
    get_current_user,
    get_model_source_repo,
    get_provider_repo,
    get_user_credential_repo,
)
from app.db.database import get_db
from app.db.models import User as UserModel
from app.db.repositories import (
    ModelSourceRepository,
    ProviderRepository,
    UserCredentialRepository,
)
from app.schemas.credentials import (
    UserCredentialTestRequest,
    UserCredentialTestResult,
    UserCredentialUpsert,
)
from app.schemas.response import success

router = APIRouter()


@router.get("")
async def list_credentials(
    request: Request,
    current_user: UserModel = Depends(get_current_user),
    repo: UserCredentialRepository = Depends(get_user_credential_repo),
):
    """列出当前用户已设置的所有 provider credentials（掩码 + 失败状态）"""
    creds = repo.list_by_user(current_user.id)
    items = [repo.to_masked_schema(c).model_dump() for c in creds]
    return success(data={"credentials": items}, request_id=request.state.request_id)


@router.put("/{provider_id}")
async def upsert_credential(
    provider_id: str,
    body: UserCredentialUpsert,
    request: Request,
    current_user: UserModel = Depends(get_current_user),
    repo: UserCredentialRepository = Depends(get_user_credential_repo),
    provider_repo: ProviderRepository = Depends(get_provider_repo),
):
    """新建或覆盖某 provider 的 user key（自动重置失败状态，等价 reactivate）"""
    if not provider_repo.get_by_id(provider_id):
        raise HTTPException(status_code=404, detail=f"未知 provider: {provider_id}")
    cred = repo.upsert(current_user.id, provider_id, body.api_key, body.is_active)
    return success(data=repo.to_masked_schema(cred).model_dump(), request_id=request.state.request_id)


@router.delete("/{provider_id}")
async def delete_credential(
    provider_id: str,
    request: Request,
    current_user: UserModel = Depends(get_current_user),
    repo: UserCredentialRepository = Depends(get_user_credential_repo),
):
    """删除某 provider 的 user key（删后自动 fallback 系统 key）"""
    deleted = repo.delete(current_user.id, provider_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="该 provider 还没设置 key")
    return success(data={"deleted": True}, request_id=request.state.request_id)


@router.post("/{provider_id}/test")
async def test_credential(
    provider_id: str,
    body: UserCredentialTestRequest,
    request: Request,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
    repo: UserCredentialRepository = Depends(get_user_credential_repo),
    model_source_repo: ModelSourceRepository = Depends(get_model_source_repo),
):
    """验证 key 有效性。优先用 body.api_key（临时测试未保存的输入），没传则用已存的。
    本接口不写任何 health 状态。"""
    from app.ai.llm_manager import llm_manager

    api_key_to_test = body.api_key
    if not api_key_to_test:
        cred = repo.get(current_user.id, provider_id)
        if cred and cred.is_active:
            try:
                api_key_to_test = repo.decrypt(cred.api_key)
            except Exception:
                api_key_to_test = None

    if not api_key_to_test:
        raise HTTPException(status_code=400, detail="没有可测试的 key（body 没传，存量也没有有效 key）")

    # 取该 provider 下任一启用模型（按优先级排序，取第一个）
    models = model_source_repo.get_all(provider=provider_id, enabled=True)
    if not models:
        raise HTTPException(status_code=404, detail=f"provider {provider_id} 下没有可用模型")

    result = await llm_manager.test_credentials(
        provider=provider_id,
        model_id=models[0].model_id,
        credentials={"api_key": api_key_to_test},
        db=db,
    )
    return success(data=UserCredentialTestResult(**result).model_dump(), request_id=request.state.request_id)
