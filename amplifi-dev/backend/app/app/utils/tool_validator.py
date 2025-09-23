from typing import List
from uuid import UUID

from fastapi import HTTPException, status

from app.models.tools_models import Tool, ToolType
from app.schemas.workspace_tool_schema import IWorkspaceToolAdd


def validate_tools_exist(tools: List[Tool], requested_ids: List[UUID]) -> None:
    """
    Ensure all requested tool IDs exist in the database.
    """
    if len(tools) != len(requested_ids):
        found_ids = {str(t.id) for t in tools}
        missing_ids = [str(tid) for tid in requested_ids if str(tid) not in found_ids]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tools not found: {missing_ids}",
        )


def validate_tool_kinds(tools: List[Tool]) -> ToolType:
    """
    Ensure all tools are of the same kind and return the common ToolType.
    """
    tool_kinds = {tool.tool_kind for tool in tools}
    if len(tool_kinds) > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="All tools must be of the same kind (either all 'system' or all 'mcp')",
        )
    return tool_kinds.pop()


def validate_tool_specific_constraints(
    tool_kind: ToolType, obj_in: IWorkspaceToolAdd
) -> None:
    """
    Validate tool-specific constraints on dataset_ids and mcp_tools.
    """
    if tool_kind == ToolType.mcp and obj_in.dataset_ids is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="dataset_ids should not be provided for MCP tools",
        )
    if tool_kind == ToolType.system and obj_in.mcp_tools is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="mcp_tools should not be provided for System tools",
        )
