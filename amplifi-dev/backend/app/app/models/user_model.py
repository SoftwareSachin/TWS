from datetime import datetime
from typing import TYPE_CHECKING, List, Optional
from uuid import UUID

from pydantic import EmailStr
from sqlmodel import BigInteger, Column, DateTime, Field, Relationship, SQLModel, String

from app.models.audit_log_model import AuditLog
from app.models.base_uuid_model import BaseUUIDModel
from app.models.chat_session_model import ChatSession
from app.models.image_media_model import ImageMedia
from app.models.links_model import LinkGroupUser
from app.models.user_workspace_link_model import UserWorkspaceLink

if TYPE_CHECKING:
    from app.models.chat_session_model import ChatSession
    from app.models.group_model import Group
    from app.models.organization_model import Organization
    from app.models.role_model import Role
    from app.models.workspace_model import Workspace


class UserBase(SQLModel):
    first_name: str
    last_name: str
    email: EmailStr = Field(sa_column=Column(String, index=True, unique=True))
    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)
    birthdate: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )  # birthday with timezone
    role_id: UUID | None = Field(default=None, foreign_key="Role.id")
    organization_id: UUID | None = Field(default=None, foreign_key="organizations.id")
    phone: str | None = None
    # gender: IGenderEnum | None = Field(
    #     default=IGenderEnum.other,
    #     sa_column=Column(ChoiceType(IGenderEnum, impl=String())),
    # )
    state: str | None = None
    country: str | None = None
    address: str | None = None


class User(BaseUUIDModel, UserBase, table=True):
    hashed_password: str | None = Field(default=None, nullable=False, index=True)
    role: Optional["Role"] = Relationship(  # noqa: F821
        back_populates="users", sa_relationship_kwargs={"lazy": "joined"}
    )

    organization: Optional["Organization"] = Relationship(  # noqa: F821
        back_populates="users", sa_relationship_kwargs={"lazy": "joined"}
    )
    groups: list["Group"] = Relationship(  # noqa: F821
        back_populates="users",
        link_model=LinkGroupUser,
        sa_relationship_kwargs={"lazy": "selectin"},
    )
    image_id: UUID | None = Field(default=None, foreign_key="ImageMedia.id")
    image: ImageMedia = Relationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "User.image_id==ImageMedia.id",
        }
    )
    follower_count: int | None = Field(
        default=None, sa_column=Column(BigInteger(), server_default="0")
    )
    following_count: int | None = Field(
        default=None, sa_column=Column(BigInteger(), server_default="0")
    )
    audit_logs: List["AuditLog"] = Relationship(back_populates="user")
    chat_sessions: List["ChatSession"] = Relationship(back_populates="user")
    workspaces: List["Workspace"] = Relationship(
        back_populates="users",
        link_model=UserWorkspaceLink,
        sa_relationship_kwargs={"lazy": "selectin"},
    )
