from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr
from sqlmodel import Column, Field, String

from app.models.group_model import GroupBase
from app.models.user_model import UserBase

from .image_media_schema import IImageMediaRead
from .role_schema import IRoleRead


class UserData(BaseModel):
    id: Optional[UUID] = None
    email: str
    organization_id: Optional[UUID] = None
    role: List[str]
    is_active: Optional[bool] = True
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_superuser: bool = False


class UserDataWithWorkspace(UserData):
    workspace_id: List[UUID]


class CachedValue(BaseModel):
    user_data: UserData
    access_token: str
    jwtToken: Optional[str] = None


class IUserCreate(UserBase):
    password: str

    class Config:
        hashed_password = None


class IUserUpdate(UserBase):
    pass


class IUserInvite(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr = Field(sa_column=Column(String, index=True, unique=True))
    role: str
    organization_id: UUID


# This schema is used to avoid circular import
class IGroupReadBasic(GroupBase):
    id: UUID


class IUserRead(UserBase):
    id: UUID
    role: IRoleRead | None = None
    groups: list[IGroupReadBasic] | None = []
    image: IImageMediaRead | None
    follower_count: int | None = 0
    following_count: int | None = 0


class IUserReadWithWorkspaces(BaseModel):
    id: UUID
    first_name: str
    last_name: str
    email: EmailStr
    is_active: bool
    is_superuser: bool
    organization_id: UUID
    role_id: UUID
    role: str
    workspace_ids: List[UUID] = []


class IUserReadWithoutGroups(UserBase):
    id: UUID
    role: IRoleRead | None = None
    image: IImageMediaRead | None
    follower_count: int | None = 0
    following_count: int | None = 0


class IUserBasicInfo(BaseModel):
    id: UUID
    first_name: str
    last_name: str


class IUserStatus(str, Enum):
    active = "active"
    inactive = "inactive"
