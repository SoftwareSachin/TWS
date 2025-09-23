from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer

from app import crud
from app.api import deps
from app.be_core.logger import logger
from app.be_core.security import IsResetPasswordRequired, get_loginradius_client
from app.models.user_model import User
from app.schemas.passwordResetSchema import PasswordData, PasswordReqFetch
from app.schemas.role_schema import IRoleEnum

loginradius = get_loginradius_client()
bearerSecurity = HTTPBearer()
router = APIRouter()


@router.post("/is_reset_password_required")
async def checkResetPassword(
    email: PasswordReqFetch,
    _: User = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.developer, IRoleEnum.member]
        )
    ),
):
    return {
        "IsPasswordResetRequired": await IsResetPasswordRequired(email.email),
    }


@router.post("/reset_password")
async def resetPassword(
    password: PasswordData,
    token: str = Depends(bearerSecurity),
    _: User = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.developer, IRoleEnum.member]
        )
    ),
):
    access_token = token.credentials
    # Removed sensitive access_token logging for security
    result = loginradius.authentication.change_password(
        access_token=access_token,
        old_password=password.oldPassword,
        new_password=password.newPassword,
    )

    if password.oldPassword == password.newPassword:
        raise HTTPException(
            status_code=400, detail="New password cannot be same as Old password."
        )
    logger.debug("reset %s", result)

    response = loginradius.authentication.get_profile_by_access_token(access_token)
    logger.debug("reset %s", response)
    if result.get("IsPosted"):
        user = await crud.user.get_by_email(email=response.get("Email")[0].get("Value"))
        logger.debug("reset %s", user)
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User with email {response.get('Email')[0].get('Value')} not found.",
            )

        # Step 2: Update the user's state
        await crud.user.update(
            obj_current=user,
            obj_new={"state": "Active"},  # Update only the 'state' field
        )
        return {
            "message": "You’ve successfully reset your password and activated your account."
        }

    raise HTTPException(status_code=400, detail="Your old password is incorrect")
