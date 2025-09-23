from enum import Enum as PyEnum
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlalchemy import JSON, Column
from sqlalchemy.types import Enum as SQLEnum
from sqlmodel import Field, Relationship, SQLModel

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.tools_models import Tool


class MCPType(PyEnum):
    internal = "internal"
    external = "external"


class MCPToolBase(SQLModel):
    mcp_subtype: MCPType = Field(
        sa_column=Column(
            SQLEnum(MCPType, name="mcptype", create_type=False), nullable=False
        ),
        description="Either 'internal' or 'external'",
    )
    mcp_server_config: dict = Field(
        ...,
        description="MCP configuration object in JSON",
        sa_column=Column(JSON),
    )
    timeout_secs: int = Field(
        default=30, description="Optional request timeout in seconds"
    )
    description: Optional[str] = Field(
        default=None, description="Human-readable description of what this tool does"
    )


class MCPTool(BaseUUIDModel, MCPToolBase, table=True):
    __tablename__ = "mcp_tools"

    tool_id: UUID = Field(
        foreign_key="tools.id",
        unique=True,
        description="Foreign key to tools table",
        nullable=False,
    )
    tool: "Tool" = Relationship(back_populates="mcp_tool")
