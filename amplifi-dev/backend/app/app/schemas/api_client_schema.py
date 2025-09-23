from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.utils.datetime_utils import ensure_naive_datetime


class ApiClientBase(BaseModel):
    """Base schema for API client data"""

    name: str = Field(..., description="Human-readable name for the API client")
    description: Optional[str] = Field(
        None, description="Optional description of the client"
    )
    expires_at: Optional[datetime] = Field(
        None, description="Expiration date for this client"
    )


class ApiClientCreate(ApiClientBase):
    """Schema for creating a new API client"""

    @field_validator("expires_at")
    @classmethod
    def expires_at_must_be_future(cls, v):
        if v is not None and ensure_naive_datetime(v) <= ensure_naive_datetime(
            datetime.utcnow()
        ):
            raise ValueError("expires_at must be in the future.")
        return v


class ApiClientUpdate(BaseModel):
    """Schema for updating an API client"""

    name: Optional[str] = Field(
        None, description="Human-readable name for the API client"
    )
    description: Optional[str] = Field(
        None, description="Optional description of the client"
    )
    expires_at: Optional[datetime] = Field(
        None, description="Expiration date for this client"
    )

    @field_validator("expires_at")
    @classmethod
    def expires_at_must_be_future(cls, v):
        if v is not None and ensure_naive_datetime(v) <= ensure_naive_datetime(
            datetime.utcnow()
        ):
            raise ValueError("expires_at must be in the future.")
        return v


class ApiClientRead(ApiClientBase):
    """Schema for reading API client data (without sensitive information)"""

    id: UUID
    client_id: str
    organization_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    @field_validator(
        "created_at", "updated_at", "last_used_at", "expires_at", mode="before"
    )
    @classmethod
    def normalize_datetimes(cls, v):
        if v is not None:
            return ensure_naive_datetime(v)
        return v

    class Config:
        from_attributes = True


class ApiClientCreateResponse(BaseModel):
    """Schema for API client creation response (includes client_secret)"""

    id: UUID
    client_id: str
    client_secret: str
    name: str
    description: Optional[str] = None
    organization_id: UUID
    expires_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ApiClientTokenRequest(BaseModel):
    """Schema for API client token request"""

    client_id: str = Field(..., description="API client ID")
    client_secret: str = Field(..., description="API client secret")


class ApiClientTokenResponse(BaseModel):
    """Schema for API client token response"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    client_id: str
