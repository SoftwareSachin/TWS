from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.mcp_tools import MCPType
from app.models.tools_models import ToolType


class IToolBase(BaseModel):
    name: str = Field(..., max_length=100)
    description: str
    deprecated: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    dataset_required: bool = False


class ISystemToolCreate(BaseModel):
    python_module: str
    function_name: str
    is_async: bool
    input_schema: Optional[str] = None
    output_schema: Optional[str] = None
    function_signature: Optional[str] = None


class IMCPToolCreate(BaseModel):
    mcp_subtype: MCPType
    mcp_server_config: Dict[str, Any]
    timeout_secs: Optional[int] = 60


class IToolCreate(IToolBase):
    tool_kind: ToolType
    system_tool: Optional[ISystemToolCreate] = None
    mcp_tool: Optional[IMCPToolCreate] = None
    organization_id: Optional[UUID] = Field(
        default=None,
        description="Organization ID for external MCP tools (auto-set from current user)",
    )

    class Config:
        schema_extra = {
            "example": {
                "name": "Sample Tool",
                "description": "Performs complex transformation",
                "tool_kind": "system",
                "system_tool": {
                    "python_module": "mytools.module",
                    "function_name": "run",
                    "is_async": True,
                    "input_schema": '{"type": "object"}',
                    "output_schema": '{"type": "object"}',
                    "function_signature": "run(input: dict) -> dict",
                },
            }
        }


class IToolCreatedResponse(BaseModel):
    id: UUID
    name: str
    tool_kind: ToolType
    dataset_required: bool


class IToolUpdate(BaseModel):
    name: str
    description: str
    deprecated: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None
    system_tool: Optional[ISystemToolCreate] = None
    mcp_tool: Optional[IMCPToolCreate] = None
    dataset_required: Optional[bool] = None


class IToolIdsList(BaseModel):
    tool_ids: List[UUID]
