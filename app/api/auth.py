import logging
from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy.orm import Session
from authlib.integrations.starlette_client import OAuth
from app.db.repositories import UserRepository, SocialAccountRepository
from app.db.database import get_db
from app.core import security
from app.schemas.auth import User as UserSchema
from app.core.config import settings
import urllib.parse

router = APIRouter()
logger = logging.getLogger(__name__)

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

@router.get("/login/{provider}")
async def login(request: Request, provider: str):
    try:
        redirect_uri = str(request.url_for('callback', provider=provider))
        client = oauth.create_client(provider)
        return await client.authorize_redirect(request, redirect_uri)
    except Exception as e:
        logger.error(f"Error during login for provider {provider}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create authorization URL.")


@router.get("/callback/{provider}")
async def callback(request: Request, provider: str, db: Session = Depends(get_db)):
    try:
        client = oauth.create_client(provider)
        token = await client.authorize_access_token(request)
        resp = await client.get('user', token=token)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not fetch user info from provider"
            )
        profile = resp.json()
        provider_user_id = str(profile['id'])
        user_repo = UserRepository(db)
        social_account_repo = SocialAccountRepository(db)
        social_account = social_account_repo.get_by_provider(provider=provider, provider_user_id=provider_user_id)
        if social_account:
            user = social_account.user
            update_data = {
                "nickname": profile.get("name"),
                "avatar": profile.get("avatar_url")
            }
            for key, value in update_data.items():
                if value is not None:
                    setattr(user, key, value)
            db.commit()
            db.refresh(user)
        else:
            username = profile.get("login")
            user = user_repo.get_by_username(username=username)
            if not user:
                user_data = {
                    "username": username,
                    "email": profile.get("email"),
                    "nickname": profile.get("name"),
                    "avatar": profile.get("avatar_url"),
                }
                user_data = {k: v for k, v in user_data.items() if v is not None}
                user = user_repo.create(obj_in=user_data)
            social_account_data = {
                "user_id": user.id,
                "provider": provider,
                "provider_user_id": provider_user_id
            }
            social_account_repo.create(obj_in=social_account_data)
            db.commit()
            db.refresh(user)
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
        logger.error(f"Error during callback for provider {provider}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="OAuth callback failed.") 