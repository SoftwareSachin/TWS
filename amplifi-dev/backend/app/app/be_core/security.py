import base64
import json
import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

import bcrypt
import jwt
import requests
from cryptography.fernet import Fernet
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from jwt import DecodeError, ExpiredSignatureError, MissingRequiredClaimError
from jwt.algorithms import RSAAlgorithm
from LoginRadius import LoginRadius as LR
from redis import Redis

from app import crud
from app.be_core.config import settings
from app.be_core.logger import logger
from app.models import User
from app.schemas.common_schema import TokenType
from app.schemas.user_schema import CachedValue, IUserInvite, UserData
from app.utils.redis_token import get_valid_tokens

fernet = Fernet(str.encode(settings.ENCRYPT_KEY))

JWT_ALGORITHM = "HS256"
valid_tenants = [f"{settings.AZURE_TENANT_ID}"]


def get_loginradius_client() -> LR | None:
    try:
        LR.API_KEY = settings.LOGINRADIUS_API_KEY
        LR.API_SECRET = settings.LOGINRADIUS_API_SECRET
        loginradius = LR()
        return loginradius
    except Exception as e:
        logger.error(f"Error initializing LoginRadius client: {e}")
        return None


loginradius = get_loginradius_client()


def validate_token(access_token):
    try:
        response = loginradius.authentication.auth_validate_access_token(access_token)
        if response.get("ErrorCode") in {905, 906}:
            logger.info("Access token is invalid: %s", access_token)
            raise HTTPException(
                status_code=401, detail="Access token is invalid"
            )  # Raise HTTPException here

        profile_response = loginradius.authentication.get_profile_by_access_token(
            access_token
        )
        return response, profile_response

    except HTTPException:
        raise  # Don't handle it, just re-raise it
    except Exception as e:
        logger.error(f"Error Validating token: {str(e)}")
        raise HTTPException(status_code=401, detail="Error validating token")


def validate_ms_token(access_token):
    jwks_url = f"https://login.microsoftonline.com/{settings.AZURE_TENANT_ID}/discovery/v2.0/keys"
    jwks = requests.get(jwks_url, timeout=10).json()

    # Decode token header to find `kid`
    unverified_header = jwt.get_unverified_header(access_token)
    rsa_key = {}

    for key in jwks["keys"]:
        if key["kid"] == unverified_header["kid"]:
            rsa_key = {
                "kty": key["kty"],
                "kid": key["kid"],
                "use": key["use"],
                "n": key["n"],
                "e": key["e"],
            }

    # Find the correct key
    if rsa_key:
        for _ in valid_tenants:
            payload = jwt.decode(
                access_token,
                RSAAlgorithm.from_jwk(rsa_key),
                algorithms=["RS256"],
                audience=f"api://{settings.MSAL_AZURE_CLIENT_ID}",
                issuer=f"https://sts.windows.net/{settings.AZURE_TENANT_ID}/",
            )
            return payload
    raise Exception("Unable to find appropriate key")


def calculate_token_expiration(lr_response: dict) -> int:
    expires_in = lr_response.get("expires_in")
    expires_in_time = datetime.fromisoformat(expires_in.replace("Z", "+00:00"))
    current_utc_time = datetime.now(timezone.utc)
    return int((expires_in_time - current_utc_time).total_seconds())


def calculate_ms_token_expiration(payload: dict) -> int:
    expires_in = payload.get("iat")
    expires_in_time = payload.get("exp")
    return int(expires_in_time - expires_in)


def build_user_info_from_lr_response(profile_response: dict, user_id: UUID) -> UserData:
    return UserData(
        id=str(user_id),
        email=profile_response["Email"][0]["Value"],
        role=profile_response.get("Roles"),
        is_active=profile_response["IsActive"],
        full_name=profile_response["FullName"],
        first_name=profile_response["FirstName"],
        last_name=profile_response["LastName"],
    )


def build_user_info_from_ms_response(profile_response: dict, user_id: UUID) -> UserData:
    email = profile_response.get("email", None)
    if not email:
        email = profile_response.get("unique_name")

    return UserData(
        id=str(user_id),
        email=email,
        role=profile_response.get("roles"),
        full_name=profile_response.get("name"),
        first_name=profile_response.get("given_name"),
        last_name=profile_response.get("family_name"),
        organization_id=profile_response.get("tid"),
    )


def build_user_data_from_user(user: User) -> UserData:
    return UserData(
        **user.model_dump(exclude={"role"}),
        role=[user.role.name],
        full_name=user.first_name + " " + user.last_name,
    )


async def cache_get_user(access_token: str, redis_client: Redis) -> CachedValue | None:
    cached_data = await redis_client.get(access_token)
    if cached_data is None:
        return None
    return CachedValue.parse_raw(cached_data)


async def cache_put_user_if_not_exists(
    redis_client: Redis,
    access_token: str,
    user_data: UserData,
    jwtToken: Optional[str] = None,
    expiration: int = None,
) -> CachedValue | None:
    cached_value = await cache_get_user(access_token, redis_client)
    if not cached_value:
        cached_value = CachedValue(
            user_data=user_data, access_token=access_token, jwtToken=jwtToken
        )
        await redis_client.set(access_token, cached_value.json(), ex=expiration)
    return cached_value


async def cache_put_user(
    redis_client: Redis,
    access_token: str,
    user_data: UserData,
    jwtToken: Optional[str] = None,
    expiration: int = None,
) -> CachedValue:
    cached_value = CachedValue(
        user_data=user_data, access_token=access_token, jwtToken=jwtToken
    )
    await redis_client.set(access_token, cached_value.json(), ex=expiration)
    return cached_value


async def generate_jwt_token(response: dict):
    params = {"access_token": response["access_token"]}
    url = f"https://{settings.LOGINRADIUS_SITEURL}/api/jwt/{settings.LOGINRADIUS_JWTAPPNAME}/token"
    logger.debug("jwt_url: %s", url)
    try:
        response_jwt = requests.get(url, params=params, timeout=10)
        response_jwt.raise_for_status()  # Raise HTTPError for bad status codes
        data = response_jwt.json()

        if data.get("errorCode") == 905:
            raise HTTPException(
                status_code=500,
                detail="JWT configuration not found in LoginRadius (Error Code: 905).",
            )

        jwtToken = data.get("signature", "null")

    except requests.exceptions.RequestException as e:
        logger.error("Error fetching JWT token: %s", e)
        raise HTTPException(status_code=500, detail="Failed to fetch JWT token.")

    return jwtToken


def create_access_token(subject: str | Any, expires_delta: timedelta = None) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode = {"exp": expire, "sub": str(subject), "type": "access"}
    headers = {"amplifi_idp": True}
    return jwt.encode(
        payload=to_encode,
        key=settings.ENCRYPT_KEY,
        algorithm=JWT_ALGORITHM,
        headers=headers,
    )


def create_refresh_token(subject: str | Any, expires_delta: timedelta = None) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES
        )
    to_encode = {"exp": expire, "sub": str(subject), "type": "refresh"}
    headers = {"amplifi_idp": True}
    return jwt.encode(
        payload=to_encode,
        key=settings.ENCRYPT_KEY,
        algorithm=JWT_ALGORITHM,
        headers=headers,
    )


def decode_token(token: str) -> dict[str, Any]:
    return jwt.decode(
        jwt=token,
        key=settings.ENCRYPT_KEY,
        algorithms=[JWT_ALGORITHM],
    )


def verify_password(plain_password: str | bytes, hashed_password: str | bytes) -> bool:
    if isinstance(plain_password, str):
        plain_password = plain_password.encode()
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode()

    return bcrypt.checkpw(plain_password, hashed_password)


def get_password_hash(plain_password: str | bytes) -> str:
    if isinstance(plain_password, str):
        plain_password = plain_password.encode()

    return bcrypt.hashpw(plain_password, bcrypt.gensalt()).decode()


def get_data_encrypt(data) -> str:
    data = fernet.encrypt(data)
    return data.decode()


def get_content(variable: str) -> str:
    return fernet.decrypt(variable.encode()).decode()


def is_jwt(token: str) -> bool:
    """
    Check if a given string is a valid JWT token format.
    This does NOT verify the token's authenticity or correctness.
    """
    if not isinstance(token, str):
        return False

    # JWT should have exactly two dots separating three parts
    parts = token.split(".")
    if len(parts) != 3:
        return False

    try:
        # Decode header and payload to check if they are valid JSON
        header = base64.urlsafe_b64decode(parts[0] + "==").decode("utf-8")
        payload = base64.urlsafe_b64decode(parts[1] + "==").decode("utf-8")

        json.loads(header)  # Check if header is valid JSON
        json.loads(payload)  # Check if payload is valid JSON

        return True
    except (ValueError, json.JSONDecodeError):
        return False


def is_amplifi_generated_token(access_token: str) -> bool:
    if not is_jwt(access_token):
        return False

    try:
        unverified_header = jwt.get_unverified_header(access_token)
        if unverified_header:
            return unverified_header.get("amplifi_idp") is True
    except DecodeError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Error when decoding the token. Please check your request.",
        )
    return False


def get_user_status(info: dict) -> str:
    if info["IsDeleted"]:
        return "Deleted"
    if info["IsLoginLocked"]:
        return "Login Locked"
    if info["IsPasswordBreached"]:
        return "Password Breached"
    if info["IsActive"]:
        if not info["FirstLogin"] and not info["EmailVerified"]:
            return "Invited"
        return "Active"
    return "UnknownStatus"


def IsResetPasswordRequired(email: str):
    return checkPasswordReset(email)


async def checkPasswordReset(email: str):
    user = await crud.user.get_by_email(email=email)
    return user.state == "Invited"


async def organization_exists(new_user: IUserInvite):
    try:
        organization_id = await crud.organization.get_organization_by_id(
            organization_id=new_user.organization_id
        )
        if not organization_id:
            logger.error(f"Invalid organization ID: {new_user.organization_id}")
            return JSONResponse(
                content={"error": "The provided organization does not exist."},
                status_code=404,
            )
    except Exception as e:
        logger.error(f"Error checking organization ID {new_user.organization_id}: {e}")
        return JSONResponse(
            content={"error": "Failed to validate the organization ID."},
            status_code=500,
        )


async def user_exists(new_user: IUserInvite):
    existing_user = await crud.user.get_by_email(email=new_user.email)
    if existing_user:
        logger.info(
            f"User with email {new_user.email} already exists. Skipping invite."
        )
        return JSONResponse(
            content={"message": "This user already has an account."},
            status_code=400,
        )


async def microsoft_authentication(
    access_token: str, redis_client: Redis, required_roles: list[str] = None
) -> UserData:
    try:
        cached_value = await cache_get_user(access_token, redis_client)
        if cached_value:
            user_data = cached_value.user_data
        else:
            payload = validate_ms_token(access_token)
            email = payload.get("email") or payload.get("unique_name")

            user = await crud.user.get_by_email(email=email)
            if not user:
                raise HTTPException(status.HTTP_403_FORBIDDEN, "User not found in DB.")

            user_data = build_user_info_from_ms_response(payload, user.id)

            # Cache the user data
            await cache_put_user(
                redis_client,
                access_token,
                user_data,
                calculate_ms_token_expiration(payload),
            )

    except jwt.PyJWTError as e:
        error_messages = {
            jwt.ExpiredSignatureError: "Your token has expired. Please log in again.",
            jwt.DecodeError: "Error decoding the token. Please check your request.",
            jwt.MissingRequiredClaimError: "Required field missing in token. Contact admin.",
            jwt.InvalidTokenError: "Invalid Token",
        }
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=error_messages.get(type(e), "Invalid Token"),
        )

    except Exception as e:
        logger.error(f"Error validating the token: {e}")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Error validating token. Contact admin.",
        )

    # Role verification
    if required_roles and not any(role in required_roles for role in user_data.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role {required_roles} is required for this action",
        )

    return user_data


async def loginradius_authentication(
    access_token: str, redis_client: Redis, required_roles: list[str] = None
) -> UserData:
    try:
        cached_value = await cache_get_user(access_token, redis_client)
        if cached_value is None:
            # Access token is not found in cache
            # For login with microsoft, validate the token
            response, profile_response = validate_token(access_token)

            user = await crud.user.get_by_email(
                email=profile_response["Email"][0]["Value"]
            )

            user_data = build_user_info_from_lr_response(profile_response, user.id)
            user_data.organization_id = str(user.organization_id)

            #  Store token in the cache

            jwtToken = await generate_jwt_token(response=response)
            cached_value = await cache_put_user(
                redis_client=redis_client,
                access_token=response["access_token"],
                user_data=user_data,
                expiration=calculate_token_expiration(response),
                jwtToken=jwtToken,
            )

        # Access token is found in cache
        # Return User info and roles
        user_data = cached_value.user_data

        if required_roles and not any(
            role in required_roles for role in (user_data.role or [])
        ):
            raise HTTPException(
                status_code=403,
                detail=f"Role {required_roles} is required for this action",
            )
        return user_data
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your token has expired. Please log in again.",
        )
    except jwt.DecodeError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Error when decoding the token. Please check your request.",
        )
    except jwt.MissingRequiredClaimError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="There is no required field in your token. Please contact the administrator.",
        )


async def amplifi_authentication(
    access_token: str, redis_client: Redis, required_roles: list[str] = None
) -> UserData:
    try:
        payload = decode_token(access_token)
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

    user_id = payload["sub"]
    valid_access_tokens = await get_valid_tokens(
        redis_client, user_id, TokenType.ACCESS
    )
    if valid_access_tokens and access_token not in valid_access_tokens:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    user: User = await crud.user.get(id=user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    if required_roles and user.role.name not in required_roles:
        raise HTTPException(
            status_code=403,
            detail=f'Role "{required_roles}" is required for this action',
        )

    return build_user_data_from_user(user)


def generate_temp_password(length=12, special_chars: bool = True):
    alphabet = string.ascii_letters + string.digits
    if special_chars:
        alphabet += "!@#$%^&*"
    password = "".join(secrets.choice(alphabet) for i in range(length))
    return password


def check_env():
    allowed_envs = {"local", "azure_dev", "azure_prod"}
    if settings.DEPLOYED_ENV not in allowed_envs:
        raise HTTPException(
            status_code=403, detail="Access denied for this environment"
        )
