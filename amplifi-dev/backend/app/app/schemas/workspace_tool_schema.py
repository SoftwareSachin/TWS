from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.mcp_tools import MCPType
from app.models.tools_models import ToolType


class IWorkspaceToolAdd(BaseModel):
    name: Optional[str] = Field(
        default=None, max_length=100, description="Name of the workspace tool"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Optional description for the workspace tool",
    )
    tool_ids: List[UUID]
    dataset_ids: Optional[List[UUID]] = None
    mcp_tools: Optional[List[str]] = Field(
        default=None, description="Optional list of MCP tools"
    )

    class Config:
        schema_extra = {
            "example": {
                "name": "Example Tool",
                "description": "This is a custom configuration of the tool for this workspace",
                "tool_ids": [
                    "11111111-1111-1111-1111-111111111111",
                    "22222222-2222-2222-2222-222222222222",
                ],
                "dataset_ids": [
                    "e9132a0b-8d5a-4d7f-9d16-6e63e6b2b2fa",
                    "33ab7be2-27e9-4f95-b72e-5b08f25f739e",
                ],
                "mcp_tools": ["tool_a", "tool_b"],
            }
        }


class IToolAssignmentResponse(BaseModel):
    id: UUID
    tool_ids: List[UUID]
    tool_names: List[str]


# Unified response for listing a workspace's tools


class ISystemToolRead(BaseModel):
    tool_id: UUID
    tool_type: ToolType
    name: str
    description: Optional[str]
    dataset_required: bool
    deprecated: Optional[bool] = False
    metadata: Optional[dict] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    python_module: Optional[str] = None
    function_name: Optional[str] = None
    is_async: Optional[bool] = None
    input_schema: Optional[str] = None
    output_schema: Optional[str] = None
    function_signature: Optional[str] = None

    @classmethod
    def from_model(cls, tool):
        system_tool = tool.system_tool
        return cls(
            tool_id=tool.id,
            tool_type=ToolType.system,
            name=tool.name,
            description=tool.description,
            dataset_required=tool.dataset_required,
            deprecated=tool.deprecated,
            metadata=tool.tool_metadata,
            created_at=tool.created_at.isoformat() if tool.created_at else None,
            updated_at=tool.updated_at.isoformat() if tool.updated_at else None,
            python_module=system_tool.python_module,
            function_name=system_tool.function_name,
            is_async=system_tool.is_async,
            input_schema=system_tool.input_schema,
            output_schema=system_tool.output_schema,
            function_signature=system_tool.function_signature,
        )


class IMCPToolRead(BaseModel):
    tool_id: UUID
    tool_type: ToolType
    name: str
    description: Optional[str]
    dataset_required: bool
    deprecated: Optional[bool] = False
    metadata: Optional[dict] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    mcp_subtype: Optional[MCPType] = None
    mcp_server_config: Optional[Dict[str, Any]] = None
    timeout_secs: Optional[int] = None

    @classmethod
    def from_model(cls, tool):
        mcp_tool = tool.mcp_tool
        return cls(
            tool_id=tool.id,
            tool_type=ToolType.mcp,
            name=tool.name,
            description=tool.description,
            dataset_required=tool.dataset_required,
            deprecated=tool.deprecated,
            metadata=tool.tool_metadata,
            created_at=tool.created_at.isoformat() if tool.created_at else None,
            updated_at=tool.updated_at.isoformat() if tool.updated_at else None,
            mcp_subtype=mcp_tool.mcp_subtype,
            mcp_server_config=mcp_tool.mcp_server_config,
            timeout_secs=mcp_tool.timeout_secs,
        )


ToolReadUnion = Union[ISystemToolRead, IMCPToolRead]


class IWorkspaceToolListRead(BaseModel):
    id: UUID
    name: Optional[str] = None
    description: Optional[str] = None
    tools: List[ToolReadUnion]
    dataset_ids: List[UUID] = Field(default_factory=list)
    mcp_tools: Optional[List[str]] = Field(default_factory=list)
