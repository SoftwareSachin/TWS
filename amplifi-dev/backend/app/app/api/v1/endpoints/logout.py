from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import HTTPBearer
from redis import Redis

from app.api.deps import get_redis_client
from app.be_core import security
from app.be_core.logger import logger

router = APIRouter()
loginradius = security.get_loginradius_client()
bearerSecurity = HTTPBearer()


@router.get("/logout", status_code=204)
async def logout(
    token: str = Depends(bearerSecurity),
    redis_client: Redis = Depends(get_redis_client),
) -> Response:
    """
    This API invalidates the user token.
    """
    access_token = token.credentials
    await redis_client.delete(access_token)

    try:
        loginradius.authentication.auth_in_validate_access_token(
            access_token=access_token
        )
    except Exception as e:
        logger.error(f"Error invalidating token: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error while invalidating the token.",
        )

    logger.info("User successfully logged out.")
    return Response(status_code=204)
