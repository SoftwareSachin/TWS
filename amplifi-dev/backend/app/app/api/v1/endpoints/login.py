from datetime import timedelta

import requests
from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from jwt import DecodeError, ExpiredSignatureError, MissingRequiredClaimError
from pydantic import EmailStr
from redis.asyncio import Redis

from app import crud
from app.api import deps
from app.api.deps import check_loginradius_errors, get_redis_client
from app.be_core import security
from app.be_core.config import settings
from app.be_core.logger import logger
from app.be_core.security import (
    build_user_info_from_lr_response,
    calculate_token_expiration,
    check_env,
    decode_token,
    generate_jwt_token,
    is_amplifi_generated_token,
    validate_token,
)
from app.schemas.common_schema import IMetaGeneral, TokenType
from app.schemas.mfa_schema import (
    MFAValidationRequest,
)
from app.schemas.passwordResetSchema import PasswordReqFetch, Vtoken, changePassword
from app.schemas.response_schema import IPostResponseBase, create_response
from app.schemas.token_schema import AccessTokenRead, RefreshToken, Token, TokenRead
from app.utils.redis_token import add_token_to_redis, get_valid_tokens

router = APIRouter()

loginradius = security.get_loginradius_client()


@router.post("")
async def login(
    email: EmailStr = Body(...),
    password: str = Body(...),
    meta_data: IMetaGeneral = Depends(deps.get_general_meta),
    redis_client: Redis = Depends(get_redis_client),
) -> IPostResponseBase[Token]:
    """
    Login for all users
    """
    user = await crud.user.authenticate(email=email, password=password)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    elif not user.is_active:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        user.id, expires_delta=access_token_expires
    )
    refresh_token = security.create_refresh_token(
        user.id, expires_delta=refresh_token_expires
    )
    # Ignore - [B106: hardcoded_password_funcarg]
    # There is no hardcoded password 'bearer' in this code.
    data = Token(
        access_token=access_token,
        token_type="bearer",
        refresh_token=refresh_token,
        user=user,
    )  # nosec
    valid_access_tokens = await get_valid_tokens(
        redis_client, user.id, TokenType.ACCESS
    )
    if valid_access_tokens:
        await add_token_to_redis(
            redis_client,
            user,
            access_token,
            TokenType.ACCESS,
            settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        )
    valid_refresh_tokens = await get_valid_tokens(
        redis_client, user.id, TokenType.REFRESH
    )
    if valid_refresh_tokens:
        await add_token_to_redis(
            redis_client,
            user,
            refresh_token,
            TokenType.REFRESH,
            settings.REFRESH_TOKEN_EXPIRE_MINUTES,
        )

    return create_response(meta=meta_data, data=data, message="Login correctly")


# @router.post("/change_password")
# async def change_password(
#     current_password: str = Body(...),
#     new_password: str = Body(...),
#     current_user: UserData= Depends(deps.get_current_user()),
#     redis_client: Redis = Depends(get_redis_client),
# ) -> IPostResponseBase[Token]:
#     """
#     Change password
#     """

#     if not verify_password(current_password, current_user.hashed_password):
#         raise HTTPException(status_code=400, detail="Invalid Current Password")

#     if verify_password(new_password, current_user.hashed_password):
#         raise HTTPException(
#             status_code=400,
#             detail="New Password should be different that the current one",
#         )

#     new_hashed_password = get_password_hash(new_password)
#     await crud.user.update(
#         obj_current=current_user, obj_new={"hashed_password": new_hashed_password}
#     )

#     access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
#     refresh_token_expires = timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
#     access_token = security.create_access_token(
#         current_user.id, expires_delta=access_token_expires
#     )
#     refresh_token = security.create_refresh_token(
#         current_user.id, expires_delta=refresh_token_expires
#     )
#     # Ignore - [B106: hardcoded_password_funcarg]
#     # There is no hardcoded password 'bearer' in this code.
#     data = Token(
#         access_token=access_token,
#         token_type="bearer",
#         refresh_token=refresh_token,
#         user=current_user,
#     )  # nosec

#     await delete_tokens(redis_client, current_user, TokenType.ACCESS)
#     await delete_tokens(redis_client, current_user, TokenType.REFRESH)
#     await add_token_to_redis(
#         redis_client,
#         current_user,
#         access_token,
#         TokenType.ACCESS,
#         settings.ACCESS_TOKEN_EXPIRE_MINUTES,
#     )
#     await add_token_to_redis(
#         redis_client,
#         current_user,
#         refresh_token,
#         TokenType.REFRESH,
#         settings.REFRESH_TOKEN_EXPIRE_MINUTES,
#     )

#     return create_response(data=data, message="New password generated")


@router.post("/new_access_token", status_code=201)
async def get_new_access_token(
    body: RefreshToken = Body(...),
    redis_client: Redis = Depends(get_redis_client),
) -> IPostResponseBase[AccessTokenRead]:
    """
    Gets a new access token using the refresh token for future requests
    """
    if not is_amplifi_generated_token(body.refresh_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="There is no required headers in your token. Please contact the administrator.",
        )
    try:
        payload = decode_token(body.refresh_token)
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your token has expired. Please log in again.",
        )
    except DecodeError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Error when decoding the token. Please check your request.",
        )
    except MissingRequiredClaimError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="There is no required field in your token. Please contact the administrator.",
        )

    if payload["type"] == "refresh":
        user_id = payload["sub"]
        valid_refresh_tokens = await get_valid_tokens(
            redis_client, user_id, TokenType.REFRESH
        )
        if valid_refresh_tokens and body.refresh_token not in valid_refresh_tokens:
            raise HTTPException(status_code=403, detail="Refresh token invalid")

        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        user = await crud.user.get(id=user_id)
        if user.is_active:
            access_token = security.create_access_token(
                payload["sub"], expires_delta=access_token_expires
            )
            valid_access_get_valid_tokens = await get_valid_tokens(
                redis_client, user.id, TokenType.ACCESS
            )
            if valid_access_get_valid_tokens:
                await add_token_to_redis(
                    redis_client,
                    user,
                    access_token,
                    TokenType.ACCESS,
                    settings.ACCESS_TOKEN_EXPIRE_MINUTES,
                )
            # Ignore - [B106: hardcoded_password_funcarg]
            # There is no hardcoded password 'bearer' in this code.
            return create_response(
                data=AccessTokenRead(
                    access_token=access_token, token_type="bearer"
                ),  # nosec
                message="Access token generated correctly",
            )
        else:
            raise HTTPException(status_code=404, detail="User inactive")
    else:
        raise HTTPException(status_code=404, detail="Incorrect token")


# @router.post("/mfa-login")
# async def mfa_login(
#     form_data: OAuth2PasswordRequestForm = Depends(),
#     redis_client: Redis = Depends(get_redis_client),
# ) -> TwoFactorSchema:
#     """
#     use this api only when the user has configured his mfa
#     """
#     response = loginradius.mfa.mfa_login_by_email(
#         email=form_data.username, password=form_data.password
#     )

#     logger.debug("mfa-login %s ", response)
#     two_factor_token = TwoFactorSchema(
#         TwoFactAuthenticationToken=response["SecondFactorAuthentication"][
#             "SecondFactorAuthenticationToken"
#         ]
#     )

#     return two_factor_token


@router.post("/access-token", response_model_exclude_none=True)
async def login_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    redis_client: Redis = Depends(get_redis_client),
) -> TokenRead:
    """
    OAuth2 compatible token login, get an access token for future requests and storing it in redis client.

    use this api , when either mfa is disabled or user has not setup his mfa.

    Note:
    - If user is present in redis cache then there will be no call to loginradius
    - Else it will invalidate User

    """
    isMFaEnabled = False
    login_data = {
        "email": form_data.username,
        "password": form_data.password,
        "securityanswer": {"Project Name": "Amplifi"},
    }
    try:
        response = loginradius.authentication.login_by_email(login_data)

    except HTTPException:
        logger.warning("Something wrong with login radius")
        raise HTTPException(
            status_code=404,
            detail=response.get("Message"),
        )

    logger.debug("profile %s ", response.get("Profile"))

    await check_loginradius_errors(response)

    if "SecondFactorAuthentication" not in response:
        isMFaEnabled = False
    else:
        isMFaEnabled = True
    second_factor = response.get("SecondFactorAuthentication")
    if second_factor and second_factor.get("SecondFactorAuthenticationToken"):
        response = loginradius.mfa.mfa_login_by_email(
            email=form_data.username, password=form_data.password
        )

        logger.debug("mfa-login %s ", response)
        two_factor_token = response["SecondFactorAuthentication"][
            "SecondFactorAuthenticationToken"
        ]
        token_read = TokenRead(
            token_type="bearer",
            secondFactorAuthenticationToken=two_factor_token,
            isMfaEnabled=isMFaEnabled,
        )  # nosec

        return token_read

    user = await crud.user.get_by_email(email=form_data.username)
    if not user:
        # Log the sync issue but don't reveal it to prevent user enumeration
        logger.error(
            f"User {form_data.username} authenticated in LoginRadius but not found in local DB. Check database sync."
        )
        raise HTTPException(status_code=400, detail="Invalid credentials")
    user_data = build_user_info_from_lr_response(response["Profile"], user.id)
    if user.organization_id:
        user_data.organization_id = str(user.organization_id)

    jwtToken = None
    firstLogin = None
    if user.state == "Active":
        firstLogin = False
        jwtToken = await generate_jwt_token(response=response)
    firstLogin = True
    qrcode = loginradius.mfa.mfa_configure_by_access_token(
        access_token=response["access_token"]
    )
    await security.cache_put_user(
        redis_client=redis_client,
        access_token=response["access_token"],
        user_data=user_data,
        expiration=calculate_token_expiration(response),
        jwtToken=jwtToken,
    )

    token_read = TokenRead(
        access_token=response["access_token"],
        refresh_token=response["refresh_token"],
        token_type="bearer",
        jwt_token=jwtToken,
        QrCode=qrcode.get("QRCode"),
        isMfaEnabled=isMFaEnabled,
        FirstLogin=firstLogin,
    )  # nosec

    if "SecondFactorAuthentication" in response:
        token_read.secondFactorAuthenticationToken = response[
            "SecondFactorAuthentication"
        ]
        if response["SecondFactorAuthentication"] is None:
            token_read.secondFactorAuthenticationToken = "null"

    return token_read


# @router.post("/get-qr-code")
# def get_qr_code(
#     authorization: Optional[str] = Header(None),
#     current_user: UserData = Depends(
#         deps.get_current_user(
#             required_roles=[IRoleEnum.admin, IRoleEnum.developer, IRoleEnum.member]
#         )
#     ),
# ) -> QRCode:
#     """
#     call this api when user is setting up his mfa for the first time
#     """
#     logger.info(f"Retrieving QR Code for MFA for user {current_user.email}")
#     if not authorization.startswith("Bearer "):
#         raise HTTPException(status_code=400, detail="Invalid authorization header")

#     access_token = authorization.replace("Bearer ", "")
#     try:
#         response = loginradius.mfa.mfa_configure_by_access_token(
#             access_token=access_token
#         )
#         logger.debug("qr %s ", response)
#     except requests.RequestException as e:
#         logger.error(f"Request to LoginRadius failed: {str(e)}")
#         raise HTTPException(status_code=400, detail="Something went wrong")
#     qrCode = QRCode(QRcode=response["QRCode"])

#     return qrCode


@router.post("/validate-code")
async def validate_code(
    payload: MFAValidationRequest,
    redis_client: Redis = Depends(get_redis_client),
) -> TokenRead:
    """
    this api will be called after the /mfa-login api
    """
    logger.info("Validating the MFA code")

    try:

        multi_factor_auth_model_by_authenticator_code = {
            "authenticatorcode": payload.authenticatorcode
        }
        if payload.secondfactorauthenticationtoken:
            response = loginradius.mfa.mfa_validate_authenticator_code(
                secondfactorauthenticationtoken=payload.secondfactorauthenticationtoken,
                multi_factor_auth_model_by_authenticator_code=multi_factor_auth_model_by_authenticator_code,
            )
            logger.debug("validate %s ", response.get("Profile"))
        else:
            response = response = loginradius.mfa.mfa_verify_authenticator_code(
                access_token=payload.access_token,
                multi_factor_auth_model_by_authenticator_code_security_answer=multi_factor_auth_model_by_authenticator_code,
            )
            response_token = validate_token(access_token=payload.access_token)
            token_data, _ = response_token
            response = {
                "Profile": response,
                "access_token": token_data["access_token"],
                "refresh_token": token_data["refresh_token"],
                "expires_in": token_data["expires_in"],
            }
            logger.debug("verify %s ", response.get("Profile"))
    except requests.RequestException as e:
        logger.error(f"request failed: {str(e)}")
        raise HTTPException(status_code=502, detail="API error")

    user = await crud.user.get_by_email(email=response["Profile"]["Email"][0]["Value"])
    if not user:
        # Log the sync issue but don't reveal it to prevent user enumeration
        logger.error(
            f"User {response['Profile']['Email'][0]['Value']} authenticated in LoginRadius but not found in local DB. Check database sync."
        )
        raise HTTPException(status_code=400, detail="Invalid credentials")
    user_data = build_user_info_from_lr_response(response["Profile"], user.id)
    if user.organization_id:
        user_data.organization_id = str(user.organization_id)

    jwtToken = await generate_jwt_token(response=response)

    await security.cache_put_user(
        redis_client=redis_client,
        access_token=response["access_token"],
        user_data=user_data,
        expiration=calculate_token_expiration(response),
        jwtToken=jwtToken,
    )

    token_read = TokenRead(
        access_token=response["access_token"],
        refresh_token=response["refresh_token"],
        token_type="bearer",
        jwt_token=jwtToken,
    )  # nosec

    return token_read


# @router.post("/verify-code")
# async def verify_code(
#     authCode: VerifyAuthCode,
#     authorization: Optional[str] = Header(None),
#     current_user: UserData = Depends(
#         deps.get_current_user(
#             required_roles=[IRoleEnum.admin, IRoleEnum.developer, IRoleEnum.member]
#         )
#     ),
# ):
#     """
#     this api is called once when the user is setting up his mfa
#     """
#     logger.info("Verifying the mfa code for the first time")

#     if not authorization.startswith("Bearer "):
#         raise HTTPException(status_code=400, detail="Invalid authorization header")

#     access_token = authorization.replace("Bearer ", "")

#     try:
#         multi_factor_auth_model_by_authenticator_code = {
#             "authenticatorcode": authCode.authCode
#         }
#         response = loginradius.mfa.mfa_verify_authenticator_code(
#             access_token=access_token,
#             multi_factor_auth_model_by_authenticator_code_security_answer=multi_factor_auth_model_by_authenticator_code,
#         )

#         logger.debug("verify %s ", response)
#     except requests.RequestException as e:
#         logger.error(f"request failed: {str(e)}")
#         raise HTTPException(status_code=502, detail="API error")

#     return response


@router.post("/verify-email")
async def verify_email(
    vtoken: Vtoken,
    redis_client: Redis = Depends(get_redis_client),
) -> TokenRead:
    logger.info("verifying the user email....")
    try:
        response = loginradius.authentication.verify_email(
            verification_token=vtoken.vtoken
        )
        logger.debug(
            "verify -email response %s", response.get("Data", {}).get("Profile")
        )
    except HTTPException:
        logger.warning("the email has been expired")
        raise HTTPException(
            status_code=400,
            detail=response.get("Message"),
        )
    if response.get("ErrorCode") == 974:
        raise HTTPException(
            status_code=400,
            detail="Email has been already verified, please login.",
        )

    user = await crud.user.get_by_email(
        email=response["Data"]["Profile"]["Email"][0]["Value"]
    )
    logger.debug("usremail %s", response["Data"]["Profile"]["Email"][0]["Value"])
    logger.debug("verify -email user %s", user)
    user_data = build_user_info_from_lr_response(response["Data"]["Profile"], user.id)
    logger.debug("verify -email usrdata %s", user_data)
    if user.organization_id:
        user_data.organization_id = str(user.organization_id)

    await security.cache_put_user_if_not_exists(
        redis_client=redis_client,
        access_token=response["Data"].get("access_token"),
        user_data=user_data,
        expiration=calculate_token_expiration(response["Data"]),
    )

    token_read = TokenRead(
        access_token=response["Data"].get("access_token"),
        refresh_token=response["Data"].get("refresh_token"),
        token_type="bearer",
        email=user_data.email,
    )  # nosec

    return token_read


@router.post(
    "/forgot-password",
    dependencies=[Depends(check_env)],
    include_in_schema=settings.DEPLOYED_ENV in {"local", "azure_dev", "azure_prod"},
)
async def forgot_password(email: PasswordReqFetch = Body(...)):

    logger.info(f"sending invite link to the user email: {email.email}")
    user = await crud.user.get_by_email(email=email.email)

    # Always return success message to prevent user enumeration
    if user is None or user.state != "Active":
        # Don't reveal whether user exists or is inactive
        return {"message": "Reset link sent."}

    try:
        response = loginradius.authentication.forgot_password(
            email=email.email,
            email_template="forgotpassword-amplifi",
            reset_password_url=settings.RESET_PASSWORD_URL,
        )

        logger.debug(response)
        if response.get("ErrorCode") == 1122:
            # Log the rate limit but still return success message to prevent enumeration
            logger.warning(f"Rate limit reached for password reset: {email.email}")

    except Exception as e:
        # Log the error but don't expose it to prevent information leakage
        logger.error(f"Error sending password reset email: {str(e)}")

    return {"message": "Reset link sent."}


@router.post(
    "/change-password",
    dependencies=[Depends(check_env)],
    include_in_schema=settings.DEPLOYED_ENV in {"local", "azure_dev", "azure_prod"},
)
async def change_password(data: changePassword):
    """
    This api will reset the password for a user.
    """

    reset_password_by_reset_token_model = {
        "resettoken": data.vtoken,
        "password": data.password,
        "welcomeemailtemplate": "",
        "resetpasswordemailtemplate": "",
    }

    response = loginradius.authentication.reset_password_by_reset_token(
        reset_password_by_reset_token_model=reset_password_by_reset_token_model
    )

    logger.debug("change-password %s ", response)
    if response.get("ErrorCode") in [974, 975]:
        logger.warning("Link has expired")
        raise HTTPException(
            status_code=400, detail="The reset password link has already been used."
        )
    elif response.get("ErrorCode") == 1015:
        logger.warning("password is too similar")
        raise HTTPException(status_code=400, detail=response.get("Description"))
    return create_response(data=response, message="Password changed successfully!")
