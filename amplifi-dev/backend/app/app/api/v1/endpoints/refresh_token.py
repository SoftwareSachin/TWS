from fastapi import APIRouter, Depends, HTTPException
from redis import Redis

from app import crud
from app.api.deps import get_redis_client
from app.be_core import security
from app.be_core.logger import logger
from app.be_core.security import generate_jwt_token
from app.schemas.refresh_token_schema import RefreshTokenSchema
from app.schemas.token_schema import TokenRead

loginradius = security.get_loginradius_client()
router = APIRouter()


@router.post("/refresh_access_token")
async def refresh_token(
    body: RefreshTokenSchema,
    redis_client: Redis = Depends(get_redis_client),
) -> TokenRead:
    """
    Retrieves New access token once the previous one is expired .
    """
    try:
        refresh_token_response = (
            loginradius.account.refresh_access_token_by_refresh_token(
                refresh__token=body.refresh_token
            )
        )
    except Exception as e:
        logger.error(f"Error refreshing access token: {str(e)}")
        raise HTTPException(401, "Session expired. Please login again.")

    if not refresh_token_response or "access_token" not in refresh_token_response:
        logger.error("Error validating refresh token: Invalid response structure")
        raise HTTPException(401, "Session expired. Please login again.")

    profile_response = loginradius.authentication.get_profile_by_access_token(
        access_token=refresh_token_response.get("access_token")
    )

    user = await crud.user.get_by_email(email=profile_response["Email"][0]["Value"])
    user_data = security.build_user_info_from_lr_response(profile_response, user.id)
    if user.organization_id:
        user_data.organization_id = str(user.organization_id)

    await security.cache_put_user(
        redis_client=redis_client,
        access_token=refresh_token_response.get("access_token"),
        user_data=user_data,
        expiration=security.calculate_token_expiration(refresh_token_response),
    )

    jwt_token = await generate_jwt_token(
        response={"access_token": refresh_token_response["access_token"]}
    )
    token_read = TokenRead(
        access_token=refresh_token_response.get("access_token"),
        refresh_token=refresh_token_response.get("refresh_token"),
        token_type="bearer",
        jwt_token=jwt_token,
    )  # nosec
    return token_read
