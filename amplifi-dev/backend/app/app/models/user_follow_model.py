from uuid import UUID

from sqlmodel import Boolean, Column, Field

from app.models.base_uuid_model import BaseUUIDModel, SQLModel


class UserFollowBase(SQLModel):
    user_id: UUID = Field(nullable=False)
    target_user_id: UUID = Field(nullable=False)


class UserFollow(BaseUUIDModel, UserFollowBase, table=True):
    is_mutual: bool | None = Field(
        default=None, sa_column=Column(Boolean(), server_default="0")
    )
