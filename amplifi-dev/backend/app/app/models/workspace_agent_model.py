from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlmodel import Field, Relationship

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.agent_model import Agent
    from app.models.workspace_model import Workspace


class WorkspaceAgent(BaseUUIDModel, table=True):
    __tablename__ = "workspace_agents"

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    workspace_id: Optional[UUID] = Field(default=None, foreign_key="workspaces.id")
    agent_id: Optional[UUID] = Field(default=None, foreign_key="agents.id")

    workspace: "Workspace" = Relationship(
        back_populates="workspace_links", sa_relationship_kwargs={"lazy": "selectin"}
    )

    agent: "Agent" = Relationship(
        back_populates="workspace_links", sa_relationship_kwargs={"lazy": "selectin"}
    )
