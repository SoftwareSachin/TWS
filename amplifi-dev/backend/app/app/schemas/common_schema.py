from enum import Enum
from uuid import UUID

from pydantic import BaseModel, field_validator

from app.schemas.role_schema import IRoleRead
from app.utils.uuid6 import uuid7


class IGenderEnum(str, Enum):
    female = "female"
    male = "male"
    other = "other"


class IMetaGeneral(BaseModel):
    roles: list[IRoleRead]


class IOrderEnum(str, Enum):
    ascendent = "ascendent"
    descendent = "descendent"


class TokenType(str, Enum):
    ACCESS = "access_token"
    REFRESH = "refresh_token"


class IUserMessage(BaseModel):
    """User message schema."""

    user_id: UUID | None = None
    message: str


class IChatResponse(BaseModel):
    """Chat response schema."""

    id: str
    message_id: str
    sender: str
    message: str
    type: str

    @field_validator("id", "message_id")
    def check_ids(cls, v):
        if v == "" or v is None:
            return str(uuid7())
        return v

    @field_validator("sender")
    def sender_must_be_bot_or_you(cls, v):
        if v not in ["bot", "you"]:
            raise ValueError("sender must be bot or you")
        return v

    @field_validator("type")
    def validate_message_type(cls, v):
        if v not in ["start", "stream", "end", "error", "info"]:
            raise ValueError("type must be start, stream or end")
        return v
