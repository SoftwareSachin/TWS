from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlmodel import Field, Relationship

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.agent_model import Agent
    from app.models.workspace_tools import WorkspaceTool


class AgentTool(BaseUUIDModel, table=True):
    __tablename__ = "agent_tools"

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    agent_id: Optional[UUID] = Field(default=None, foreign_key="agents.id")
    workspace_tool_id: Optional[UUID] = Field(
        default=None, foreign_key="workspace_tools.id"
    )

    # relationships
    agent: "Agent" = Relationship(
        back_populates="tools", sa_relationship_kwargs={"lazy": "selectin"}
    )
    workspace_tool: Optional["WorkspaceTool"] = Relationship(
        back_populates="agent_tools", sa_relationship_kwargs={"lazy": "selectin"}
    )
