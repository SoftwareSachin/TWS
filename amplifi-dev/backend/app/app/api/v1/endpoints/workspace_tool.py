from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi_pagination import Params
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.api import deps
from app.be_core.logger import logger
from app.models.tools_models import Tool, ToolType
from app.schemas.response_schema import (
    IGetResponseBase,
    IGetResponsePaginated,
    IPostResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData
from app.schemas.workspace_tool_schema import (
    IMCPToolRead,
    ISystemToolRead,
    IToolAssignmentResponse,
    IWorkspaceToolAdd,
    IWorkspaceToolListRead,
    ToolReadUnion,
)
from app.utils.tool_validator import (
    validate_tool_kinds,
    validate_tool_specific_constraints,
    validate_tools_exist,
)

router = APIRouter()


@router.post(
    "/workspace/{workspace_id}/tool",
    response_model=IPostResponseBase[IToolAssignmentResponse],
    summary="Assign multiple tools to a workspace, with optional dataset IDs",
)
async def assign_tool_to_workspace(
    workspace_id: UUID,
    obj_in: IWorkspaceToolAdd,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.workspace_check),
    db: AsyncSession = Depends(deps.get_db),
) -> IPostResponseBase[IToolAssignmentResponse]:
    """
    Assign multiple tools to a workspace with validation on tool kind (system or mcp).

    Required roles:
    - admin
    - developer
    """

    try:
        result = await db.execute(select(Tool).where(Tool.id.in_(obj_in.tool_ids)))
        tools = result.scalars().all()

        validate_tools_exist(tools, obj_in.tool_ids)
        tool_kind = validate_tool_kinds(tools)
        validate_tool_specific_constraints(tool_kind, obj_in)

        assigned_tool = await crud.workspace_tool_crud.assign_tool_to_workspace(
            workspace_id=workspace_id,
            obj_in=obj_in,
            db_session=db,
        )

        if not assigned_tool:
            raise HTTPException(status_code=500, detail="Failed to assign tools")

        return create_response(
            data=IToolAssignmentResponse(
                id=assigned_tool.id,
                tool_ids=obj_in.tool_ids,
                tool_names=[tool.name for tool in tools],
            ),
            message="Tools assigned successfully",
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception(
            f"Unexpected error assigning tools to workspace {workspace_id}"
        )
        raise HTTPException(status_code=500, detail="Unexpected error")


@router.get(
    "/workspace/{workspace_id}/workspace_tool/{workspace_tool_id}",
    response_model=IGetResponseBase[IWorkspaceToolListRead],
    summary="Get a workspace tool by ID",
    description="Retrieve details of a specific workspace tool using its ID.",
)
async def get_workspace_tool_by_id(
    workspace_id: UUID,
    workspace_tool_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.workspace_check),
):
    workspace_tool = await crud.workspace_tool_crud.get_workspace_tool_by_id(
        db_session=db,
        workspace_id=workspace_id,
        workspace_tool_id=workspace_tool_id,
    )

    if workspace_tool is None:
        raise HTTPException(status_code=404, detail="Workspace tool not found.")

    tool_objs = (
        workspace_tool.tools
    )  # assumes many-to-one relationship or list fetched in query
    if not tool_objs:
        raise HTTPException(status_code=404, detail="Associated tools not found.")

    # Validate tool types are consistent
    tool_kinds = {tool.tool_kind for tool in tool_objs}
    if len(tool_kinds) > 1:
        raise HTTPException(
            status_code=400,
            detail="Workspace tool contains mixed tool types. This is unsupported.",
        )

    tool_kind = tool_kinds.pop()
    tool_schemas: List[ToolReadUnion] = []

    for tool in tool_objs:
        if tool_kind == ToolType.system:
            tool_schemas.append(ISystemToolRead.from_model(tool))
        elif tool_kind == ToolType.mcp:
            tool_schemas.append(IMCPToolRead.from_model(tool))
        else:
            raise HTTPException(status_code=400, detail="Unsupported tool type.")

    return create_response(
        data=IWorkspaceToolListRead(
            id=workspace_tool.id,
            name=workspace_tool.name,
            description=workspace_tool.description,
            tools=tool_schemas,  # updated to list[ToolReadUnion]
            dataset_ids=workspace_tool.dataset_ids or [],
            mcp_tools=workspace_tool.mcp_tools or [],
        ),
        message="Workspace tool retrieved successfully.",
    )


@router.get(
    "/workspace/{workspace_id}/tool",
    response_model=IGetResponsePaginated[IWorkspaceToolListRead],
    summary="Get tools for a workspace",
)
async def get_workspace_tools(
    workspace_id: UUID,
    params: Params = Depends(),
    search: Optional[str] = Query(
        None, description="Search by name or description of tool or workspace tool"
    ),
    tool_kind: Optional[ToolType] = Query(
        None, description="Filter by tool kind (system or mcp)"
    ),
    db: AsyncSession = Depends(deps.get_db),
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.workspace_check),
):
    """
    Get all tools assigned to a specific workspace (paginated).

    Optional:
    - Filter by `tool_type` (system or mcp).
    - Search by name/description.

    Required roles:
    - admin
    - developer
    """
    page = await crud.workspace_tool_crud.get_workspace_tools(
        db_session=db,
        workspace_id=workspace_id,
        params=params,
        search=search,
        tool_kind=tool_kind,
    )

    mapped_items = []
    for workspace_tool in page.items:
        tools = getattr(workspace_tool, "tools", [])
        if not tools:
            continue

        tool_kinds = {tool.tool_kind for tool in tools}
        if len(tool_kinds) > 1:
            continue  # skip if mixed kinds

        kind = tool_kinds.pop()

        if tool_kind and kind != tool_kind:
            continue

        tool_schemas = []
        for tool in tools:
            if kind == ToolType.system:
                tool_schemas.append(ISystemToolRead.from_model(tool))
            elif kind == ToolType.mcp:
                tool_schemas.append(IMCPToolRead.from_model(tool))

        mapped_items.append(
            IWorkspaceToolListRead(
                id=workspace_tool.id,
                name=workspace_tool.name,
                description=workspace_tool.description,
                tools=tool_schemas,
                dataset_ids=workspace_tool.dataset_ids or [],
                mcp_tools=workspace_tool.mcp_tools or [],
            )
        )

    page.items = mapped_items
    return create_response(data=page, message="Data Got correctly")


@router.put(
    "/workspace/{workspace_id}/workspace_tool/{workspace_tool_id}",
    response_model=IPostResponseBase[IToolAssignmentResponse],
    summary="Update a workspace tool by ID",
    description="Update name, datasets, tool IDs, or MCP tools for a workspace tool.",
)
async def update_workspace_tool_by_id(
    workspace_id: UUID,
    workspace_tool_id: UUID,
    obj_in: IWorkspaceToolAdd,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.workspace_check),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Update the assigned tools within a workspace.

    Required roles:
    - admin
    - developer
    """
    try:
        updated_tool = await crud.workspace_tool_crud.update_workspace_tool_by_id(
            workspace_id=workspace_id,
            workspace_tool_id=workspace_tool_id,
            obj_in=obj_in,
            db_session=db,
        )

        if not updated_tool:
            raise HTTPException(status_code=404, detail="Workspace tool not found")

        # Get tool names for response
        result = await db.execute(
            select(Tool).where(Tool.id.in_(updated_tool.tool_ids))
        )
        tools = result.scalars().all()

        return create_response(
            data=IToolAssignmentResponse(
                id=updated_tool.id,
                tool_ids=updated_tool.tool_ids,
                tool_names=[tool.name for tool in tools],
            ),
            message="Workspace tool updated successfully",
        )

    except HTTPException as e:
        raise e
    except Exception:
        logger.exception(
            f"Unexpected error updating workspace_tool {workspace_tool_id}"
        )
        raise HTTPException(status_code=500, detail="Unexpected error")


@router.delete(
    "/workspace/{workspace_id}/workspace_tool/{workspace_tool_id}",
    status_code=204,
    summary="Unassign tools from a workspace",
    description="Soft deletes the workspace tool association, effectively unassigning all tools under it.",
)
async def unassign_tool_from_workspace(
    workspace_id: UUID,
    workspace_tool_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.user_workspace_access_check),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Unassign tools from the workspace by soft-deleting the workspace tool record.

    Required roles:
    - admin
    - developer
    """
    await crud.workspace_tool_crud.delete_workspace_tool(
        workspace_id=workspace_id,
        workspace_tool_id=workspace_tool_id,
        db_session=db,
    )
    return Response(status_code=204)


@router.get(
    "/system-tools",
    response_model=IGetResponsePaginated[ISystemToolRead],
    summary="Get all system tools",
)
async def get_system_tools(
    search: Optional[str] = None,
    params: Params = Depends(),
    db: AsyncSession = Depends(deps.get_db),
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
):
    """
    Returns all system tools (paginated).

    Required roles:
    - admin
    - developer
    """
    page = await crud.workspace_tool_crud.get_system_tools(
        db_session=db,
        params=params,
        search=search,
    )

    mapped_items = []
    for system_tool in page.items:
        tool = system_tool.tool
        if tool is None:
            continue

        schema = ISystemToolRead.from_model(tool)
        mapped_items.append(schema)

    page.items = mapped_items
    return create_response(data=page, message="Data Got correctly")
