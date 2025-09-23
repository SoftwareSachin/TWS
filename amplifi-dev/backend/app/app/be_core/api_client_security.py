from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

import jwt
from fastapi import HTTPException, status
from jwt import DecodeError, ExpiredSignatureError, MissingRequiredClaimError

from app.be_core.config import settings
from app.models.api_client_model import ApiClient
from app.schemas.user_schema import UserData


def create_api_client_token(
    client_id: str,
    organization_id: UUID,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT token for API client authentication.

    Args:
        client_id: The API client ID
        organization_id: The organization ID
        expires_delta: Optional expiration time

    Returns:
        JWT token string
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode = {
        "sub": client_id,
        "type": "api_client",
        "organization_id": str(organization_id),
        "exp": expire,
        "iat": datetime.utcnow(),
    }

    encoded_jwt = jwt.encode(to_encode, settings.ENCRYPT_KEY, algorithm="HS256")
    return encoded_jwt


def decode_api_client_token(token: str) -> dict:
    """
    Decode and validate a JWT token for API client authentication.

    Args:
        token: JWT token string

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid, expired, or missing required claims
    """
    try:
        payload = jwt.decode(token, settings.ENCRYPT_KEY, algorithms=["HS256"])
        return payload
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your token has expired. Please authenticate again.",
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


def validate_api_client_token(token: str) -> dict:
    """
    Validate a JWT token and ensure it's for API client authentication.

    Args:
        token: JWT token string

    Returns:
        Validated token payload

    Raises:
        HTTPException: If token is invalid or not for API client
    """
    payload = decode_api_client_token(token)

    # Check if this is an API client token
    if payload.get("type") != "api_client":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid token type. This endpoint requires API client authentication.",
        )

    # Check if required fields are present
    if not payload.get("sub") or not payload.get("organization_id"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token missing required fields.",
        )

    return payload


def build_api_client_user_data(api_client: ApiClient) -> UserData:
    """
    Build UserData object from API client for compatibility with existing middleware.

    Args:
        api_client: The authenticated API client

    Returns:
        UserData object representing the API client
    """
    return UserData(
        id=api_client.id,
        email=f"api-client-{api_client.client_id}@system.local",
        first_name="API",
        last_name="Client",
        is_active=True,  # API clients are always considered active
        is_superuser=False,
        role=["api_client"],
        organization_id=str(api_client.organization_id),
        phone=None,
        state=None,
        country=None,
        address=None,
    )


def is_api_client_token(token: str) -> bool:
    """
    Check if a token is an API client token.

    Args:
        token: JWT token string

    Returns:
        True if it's an API client token, False otherwise
    """
    try:
        payload = decode_api_client_token(token)
        return payload.get("type") == "api_client"
    except HTTPException:
        return False
