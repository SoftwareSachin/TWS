from enum import Enum as PyEnum
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlalchemy import JSON, Column
from sqlalchemy.types import Enum as SQLEnum
from sqlmodel import Field, Relationship, SQLModel

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.mcp_tools import MCPTool
    from app.models.organization_model import Organization
    from app.models.system_tools import SystemTool


class ToolType(PyEnum):
    system = "system"
    mcp = "mcp"


class ToolBase(SQLModel):
    name: str = Field(index=True, max_length=100)
    description: Optional[str] = None
    deprecated: bool = Field(default=False)
    tool_kind: ToolType = Field(
        sa_column=Column(
            SQLEnum(ToolType, name="tooltype", create_type=False), nullable=False
        ),
        description="Either 'system' or 'mcp'",
    )
    # renamed from 'metadata' to 'tool_metadata'
    tool_metadata: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    dataset_required: bool = Field(
        default=False, description="Whether this tool requires a dataset"
    )


class Tool(BaseUUIDModel, ToolBase, table=True):
    __tablename__ = "tools"

    # Organization ID for external MCP tools only (null for system and internal MCP tools)
    organization_id: Optional[UUID] = Field(
        default=None,
        foreign_key="organizations.id",
        nullable=True,
        description="Organization ID for external MCP tools (null for global tools)",
    )

    system_tool: Optional["SystemTool"] = Relationship(back_populates="tool")
    mcp_tool: Optional["MCPTool"] = Relationship(back_populates="tool")
    organization: Optional["Organization"] = Relationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "foreign_keys": "[Tool.organization_id]",
        }
    )
    # workspace_links: List["WorkspaceTool"] = Relationship(back_populates="tool")
