from typing import TYPE_CHECKING, List, Optional
from uuid import UUID

from sqlmodel import Field, Relationship, SQLModel

from app.models.base_uuid_model import BaseUUIDModel
from app.models.user_workspace_link_model import UserWorkspaceLink
from app.models.workspace_agent_model import WorkspaceAgent
from app.models.workspace_tools import WorkspaceTool

if TYPE_CHECKING:
    from app.models.chat_app_model import ChatApp
    from app.models.dataset_model import Dataset
    from app.models.file_model import File
    from app.models.organization_model import Organization
    from app.models.source_model import Source
    from app.models.user_model import User


class WorkspaceBase(SQLModel):
    name: str = Field(..., index=True)


class Workspace(BaseUUIDModel, WorkspaceBase, table=True):
    __tablename__ = "workspaces"

    organization_id: UUID = Field(foreign_key="organizations.id")
    is_active: bool = Field(default=True)
    description: Optional[str] = Field(default=None)
    processed_chunks: int = Field(default=0)

    organization: "Organization" = Relationship(
        back_populates="workspaces",
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "Workspace.organization_id == Organization.id",
        },
    )

    sources: List["Source"] = Relationship(
        back_populates="workspace",
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "Workspace.id == Source.workspace_id",
        },
    )

    files: List["File"] = Relationship(
        back_populates="workspace",
        sa_relationship_kwargs={
            "lazy": "select",  # This will load lazily, only when accessed
            "primaryjoin": "Workspace.id == File.workspace_id",
        },
    )
    datasets: List["Dataset"] = Relationship(
        back_populates="workspace",
        sa_relationship_kwargs={
            "lazy": "select",
            "primaryjoin": "Workspace.id == Dataset.workspace_id",
        },
    )
    chatapps: List["ChatApp"] = Relationship(
        back_populates="workspace",
        sa_relationship_kwargs={
            "lazy": "select",
            "primaryjoin": "Workspace.id == ChatApp.workspace_id",
        },
    )
    users: List["User"] = Relationship(
        back_populates="workspaces",
        link_model=UserWorkspaceLink,
        sa_relationship_kwargs={"lazy": "selectin"},
    )

    tools: List["WorkspaceTool"] = Relationship(
        back_populates="workspace",
        sa_relationship_kwargs={
            "lazy": "select",
        },
    )

    workspace_links: List["WorkspaceAgent"] = Relationship(
        back_populates="workspace", sa_relationship_kwargs={"lazy": "selectin"}
    )

    # agents: List["Agent"] = Relationship(
    #     back_populates="workspace_links",
    #     link_model=WorkspaceAgent,
    #     sa_relationship_kwargs={"lazy": "selectin"},
    # )
