from typing import TYPE_CHECKING, List, Optional
from uuid import UUID

from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlmodel import Field, Relationship, SQLModel

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.agent_tools_model import AgentTool
    from app.models.tools_models import Tool
    from app.models.workspace_model import Workspace


class WorkspaceToolBase(SQLModel):
    dataset_ids: Optional[List[UUID]] = Field(
        sa_column=Column(ARRAY(PG_UUID)),
        default=None,
        description="Dataset ids on which this tool can be applied",
    )
    name: str = Field(index=True, max_length=100)
    description: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Optional description for this workspace tool",
    )
    mcp_tools: Optional[List[str]] = Field(
        default=None,
        sa_column=Column(ARRAY(String)),
        description="Optional list of MCP tools",
    )

    tool_ids: List[UUID] = Field(
        sa_column=Column(ARRAY(PG_UUID)),
        description="List of tool UUIDs associated with this workspace tool",
    )


class WorkspaceTool(BaseUUIDModel, WorkspaceToolBase, table=True):
    __tablename__ = "workspace_tools"

    workspace_id: UUID = Field(foreign_key="workspaces.id")

    workspace: "Workspace" = Relationship(back_populates="tools")

    tools: List["Tool"] = Relationship(
        sa_relationship_kwargs={
            "primaryjoin": "foreign(Tool.id) == any_(WorkspaceTool.tool_ids)",
            "lazy": "selectin",
            "viewonly": True,
        }
    )

    agent_tools: List["AgentTool"] = Relationship(
        back_populates="workspace_tool", sa_relationship_kwargs={"lazy": "selectin"}
    )
