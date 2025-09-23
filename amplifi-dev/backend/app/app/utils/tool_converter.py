from fastapi import HTTPException

from app.models.tools_models import Tool, ToolType
from app.schemas.workspace_tool_schema import (
    IMCPToolRead,
    ISystemToolRead,
    ToolReadUnion,
)


def convert_tool_to_schema(tool: Tool) -> ToolReadUnion:
    if tool.tool_kind == ToolType.system:
        return ISystemToolRead.from_model(tool)
    elif tool.tool_kind == ToolType.mcp:
        return IMCPToolRead.from_model(tool)
    else:
        raise HTTPException(status_code=400, detail="Unsupported tool kind")
