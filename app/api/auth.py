from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy.orm import Session
from authlib.integrations.starlette_client import OAuth
from app.db.repositories import UserRepository, SocialAccountRepository
from app.db.models import User as UserModel
from app.db.database import get_db
from app.core import security
from app.schemas.auth import User as UserSchema
from app.core.config import settings
import urllib.parse

router = APIRouter()

oauth = OAuth()

oauth.register(
    name='github',
    client_id=settings.GITHUB_CLIENT_ID,
    client_secret=settings.GITHUB_CLIENT_SECRET,
    access_token_url='https://github.com/login/oauth/access_token',
    authorize_url='https://github.com/login/oauth/authorize',
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'},
)

oauth.register(
    name='google',
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    access_token_url='https://oauth2.googleapis.com/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    api_base_url='https://www.googleapis.com/oauth2/v2/',
    client_kwargs={'scope': 'email profile'},
)

@router.get("/login/{provider}")
async def login(request: Request, provider: str):
    try:
        # 使用配置的服务器地址构建回调URL，而不是依赖request.url_for
        base_url = settings.SERVER_HOST.rstrip('/')
        redirect_uri = f"{base_url}/api/auth/callback/{provider}"
        
        client = oauth.create_client(provider)
        return await client.authorize_redirect(request, redirect_uri)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create authorization URL: {str(e)}")


@router.get("/callback/{provider}")
async def callback(request: Request, provider: str, db: Session = Depends(get_db)):
    try:
        client = oauth.create_client(provider)
        token = await client.authorize_access_token(request)
        
        # 根据不同的提供商获取用户信息
        if provider == 'google':
            # Google OAuth 2.0 API调用获取用户信息
            resp = await client.get('userinfo', token=token)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Could not fetch user info from provider"
                )
            profile = resp.json()
            provider_user_id = str(profile['id'])  # Google使用'id'作为用户ID
            username = profile.get("email", "").split('@')[0]  # 使用email前缀作为用户名
            user_data = {
                "username": username,
                "email": profile.get("email"),
                "nickname": profile.get("name"),
                "avatar": profile.get("picture"),
            }
        elif provider == 'github':
            # GitHub的处理逻辑
            resp = await client.get('user', token=token)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Could not fetch user info from provider"
                )
            profile = resp.json()
            provider_user_id = str(profile['id'])
            username = profile.get("login")
            
            # 获取GitHub用户email - 如果公开email为空，尝试获取私有emails
            user_email = profile.get("email")
            if not user_email:
                try:
                    emails_resp = await client.get('user/emails', token=token)
                    if emails_resp.status_code == 200:
                        emails = emails_resp.json()
                        # 查找主要邮箱
                        for email_info in emails:
                            if email_info.get('primary', False):
                                user_email = email_info.get('email')
                                break
                        # 如果没有主要邮箱，使用第一个验证过的邮箱
                        if not user_email:
                            for email_info in emails:
                                if email_info.get('verified', False):
                                    user_email = email_info.get('email')
                                    break
                except Exception as e:
                    pass
            
            user_data = {
                "username": username,
                "email": user_email,
                "nickname": profile.get("name"),
                "avatar": profile.get("avatar_url"),
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported provider: {provider}"
            )
        
        user_repo = UserRepository(db)
        social_account_repo = SocialAccountRepository(db)
        social_account = social_account_repo.get_by_provider(provider=provider, provider_user_id=provider_user_id)
        
        if social_account:
            # 更新现有用户信息
            user = social_account.user
            
            # 更新nickname和avatar
            if user_data.get("nickname") and not user.nickname:
                user.nickname = user_data.get("nickname")
            if user_data.get("avatar") and not user.avatar:
                user.avatar = user_data.get("avatar")
            
            # 更新email - 如果获取到了email且用户当前没有email，则更新
            new_email = user_data.get("email")
            if new_email and new_email.strip() and not user.email:
                user.email = new_email
            
            db.commit()
            db.refresh(user)
        else:
            # 创建新用户或关联现有用户
            # 优先通过邮箱查找现有用户（支持多OAuth提供商账户合并）
            user = None
            email = user_data.get("email")
            
            # 如果有email，先通过邮箱查找
            if email and email.strip():
                user = db.query(UserModel).filter(UserModel.email == email).first()
            
            # 如果邮箱未找到用户，再通过用户名查找
            if not user:
                user = user_repo.get_by_username(username=username)
            
            # 如果仍未找到用户，创建新用户
            if not user:
                # 过滤掉None值
                user_data = {k: v for k, v in user_data.items() if v is not None}
                user = user_repo.create(obj_in=user_data)
                # 立即flush以获取用户ID
                db.flush()
                db.refresh(user)
            else:
                # 如果找到现有用户，更新用户信息（合并账户信息）
                if user_data.get("nickname") and not user.nickname:
                    user.nickname = user_data.get("nickname")
                if user_data.get("avatar") and not user.avatar:
                    user.avatar = user_data.get("avatar")
                # 只有当新email不为空且用户当前没有email时才更新
                new_email = user_data.get("email")
                if new_email and new_email.strip() and not user.email:
                    user.email = new_email
                db.flush()
                db.refresh(user)
            
            # 创建社交账户关联
            social_account_data = {
                "user_id": user.id,
                "provider": provider,
                "provider_user_id": provider_user_id
            }
            social_account_repo.create(obj_in=social_account_data)
            db.commit()
            db.refresh(user)
        
        # 生成访问令牌并重定向
        access_token = security.create_access_token(data={"sub": str(user.id)})
        redirect_url = settings.FRONTEND_AUTH_CALLBACK_URL
        params = {"token": access_token, "token_type": "bearer"}
        url_parts = list(urllib.parse.urlparse(redirect_url))
        query = dict(urllib.parse.parse_qsl(url_parts[4]))
        query.update(params)
        url_parts[4] = urllib.parse.urlencode(query)
        final_redirect_url = urllib.parse.urlunparse(url_parts)
        return RedirectResponse(url=final_redirect_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail="OAuth callback failed.") 