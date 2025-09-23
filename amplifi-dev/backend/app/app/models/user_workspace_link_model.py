from typing import TYPE_CHECKING
from uuid import UUID

from sqlmodel import Field, SQLModel

if TYPE_CHECKING:
    pass


class UserWorkspaceLink(SQLModel, table=True):
    __tablename__ = "user_workspace_link"
    user_id: UUID = Field(foreign_key="User.id", primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", primary_key=True)
