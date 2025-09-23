import asyncio
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi_pagination import Params
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

try:
    from mcp.shared.exceptions import McpError
except ImportError:
    # Fallback if MCP is not available
    class McpError(Exception):
        pass


from app.api import deps
from app.be_core.logger import logger
from app.crud.tool_crud import tool_crud
from app.models.mcp_tools import MCPType
from app.models.tools_models import ToolType
from app.schemas.mcp_schema import (
    MCPConfigValidationRequest,
    MCPConfigValidationResponse,
)
from app.schemas.response_schema import (
    IGetResponseBase,
    IGetResponsePaginated,
    IPostResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.tool_schema import (
    IToolCreate,
    IToolCreatedResponse,
    IToolIdsList,
    IToolUpdate,
)
from app.schemas.user_schema import UserData
from app.schemas.workspace_tool_schema import (
    ToolReadUnion,
)
from app.utils.mcp_server_manager import MCPServerManager
from app.utils.tool_converter import convert_tool_to_schema

router = APIRouter()


@router.post(
    "/tool/validate-mcp-config",
    response_model=IPostResponseBase[MCPConfigValidationResponse],
    summary="Validate MCP server configuration and list available tools",
)
async def validate_mcp_config(
    request: MCPConfigValidationRequest,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
) -> IPostResponseBase[MCPConfigValidationResponse]:
    """
    Validate an MCP server configuration by attempting to connect to it
    and list all available tools. Optionally test actual tool execution
    to verify authentication and permissions are working correctly.
    This endpoint helps verify that MCP server configurations are
    working before saving them.
    """
    server_name = "unknown"

    try:
        logger.info(f"Validating MCP server configuration: {request.mcp_server_config}")

        # Initialize MCP server manager
        manager = MCPServerManager()

        # Convert Pydantic model to dictionary if needed
        config_dict = request.mcp_server_config
        if hasattr(config_dict, "model_dump"):
            config_dict = config_dict.model_dump()
        elif not isinstance(config_dict, dict):
            # Try to convert to dict if it's some other object
            config_dict = dict(config_dict) if hasattr(config_dict, "__iter__") else {}

        # Extract server name from config (guaranteed to have at least one key due to validation)
        server_name = list(config_dict.keys())[0]

        # Pass the actual config dictionary with timeout only if provided
        if request.timeout_seconds is not None:
            tools = await manager.list_tools(config_dict, request.timeout_seconds)
        else:
            tools = await manager.list_tools(config_dict)

        # Optionally test tool execution if requested
        tool_test_results = None
        if request.test_tool_execution:
            logger.info("Testing tool execution as requested")
            try:
                tool_test_results = await manager.test_tool_execution(
                    config_dict,
                    timeout_seconds=request.timeout_seconds or 30,
                    max_retries=request.max_retries or 3,
                    enable_retries=(
                        request.enable_retries
                        if request.enable_retries is not None
                        else True
                    ),
                )

                # Count successful vs failed tests
                successful_tests = sum(
                    1 for test in tool_test_results if test["test_status"] == "success"
                )
                total_tests = len(tool_test_results)

                logger.info(
                    f"Tool execution tests completed: {successful_tests}/{total_tests} tools working"
                )
            except Exception as e:
                logger.warning(f"Tool execution testing failed: {str(e)}")
                # Don't fail the whole validation if tool testing fails
                tool_test_results = [{"error": f"Tool testing failed: {str(e)}"}]

        logger.info(f"MCP server validation successful. Found {len(tools)} tools")

        # Create response message based on whether tool testing was performed
        if (
            tool_test_results
            and isinstance(tool_test_results, list)
            and len(tool_test_results) > 0
            and "error" not in tool_test_results[0]
        ):
            successful_tests = sum(
                1 for test in tool_test_results if test["test_status"] == "success"
            )
            rate_limited_tests = sum(
                1
                for test in tool_test_results
                if test["test_status"] in ["rate_limited", "rate_limited_after_retries"]
            )
            rate_limited_after_retries = sum(
                1
                for test in tool_test_results
                if test["test_status"] == "rate_limited_after_retries"
            )
            total_tests = len(tool_test_results)

            if rate_limited_tests > 0:
                if rate_limited_after_retries > 0:
                    message = f"Successfully connected to MCP server '{server_name}'. Found {len(tools)} tools. Tool execution test: {successful_tests}/{total_tests} tools working correctly, {rate_limited_tests} rate limited ({rate_limited_after_retries} failed after retries)."
                else:
                    message = f"Successfully connected to MCP server '{server_name}'. Found {len(tools)} tools. Tool execution test: {successful_tests}/{total_tests} tools working correctly, {rate_limited_tests} rate limited."
            else:
                message = f"Successfully connected to MCP server '{server_name}'. Found {len(tools)} tools. Tool execution test: {successful_tests}/{total_tests} tools working correctly."
        else:
            message = f"Successfully connected to MCP server '{server_name}'. Found {len(tools)} tools."

        response_data = MCPConfigValidationResponse(
            valid=True,
            server_name=server_name,
            tools_count=len(tools),
            available_tools=tools,
            message=message,
            tool_test_results=tool_test_results,
        )

        return create_response(
            data=response_data,
            message="MCP configuration validated successfully",
        )

    except ValidationError as e:
        logger.warning(
            f"MCP configuration validation failed - Invalid request: {str(e)}"
        )
        # Extract validation error details
        error_details = "; ".join(
            [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid MCP server configuration format",
                "error_details": error_details,
                "server_name": server_name,
            },
        )

    except ValueError as e:
        logger.warning(
            f"MCP configuration validation failed - Invalid config: {str(e)}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid MCP server configuration",
                "error_details": str(e),
                "server_name": server_name,
            },
        )

    except FileNotFoundError as e:
        logger.error(f"MCP server executable not found: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail={
                "message": "MCP server executable not found",
                "error_details": f"Executable not found: {str(e)}",
                "server_name": server_name,
            },
        )

    except McpError as e:
        logger.error(f"MCP protocol error: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail={
                "message": "MCP server protocol error",
                "error_details": f"MCP error: {str(e)}",
                "server_name": server_name,
            },
        )

    except asyncio.TimeoutError:
        logger.error(f"MCP server connection timeout for {server_name}")
        raise HTTPException(
            status_code=408,
            detail={
                "message": "MCP server connection timeout",
                "error_details": f"Connection to MCP server '{server_name}' timed out",
                "server_name": server_name,
            },
        )

    except ConnectionError as e:
        logger.error(f"Failed to connect to MCP server: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Failed to connect to MCP server",
                "error_details": f"Connection error: {str(e)}",
                "server_name": server_name,
            },
        )

    except Exception as e:
        logger.error(f"Unexpected error during MCP validation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Unexpected error during validation",
                "error_details": str(e),
                "server_name": server_name,
            },
        )


@router.post(
    "/tool",
    response_model=IPostResponseBase[IToolCreatedResponse],
    summary="Create a new tool (System or MCP)",
)
async def create_tool(
    obj_in: IToolCreate,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin])
    ),
    db: AsyncSession = Depends(deps.get_db),
) -> IPostResponseBase[IToolCreatedResponse]:
    """
    Create a new tool and its subtype (System or MCP).
    Only admin role is allowed.
    """
    try:
        # Set organization_id for external MCP tools
        if (
            obj_in.tool_kind == ToolType.mcp
            and obj_in.mcp_tool
            and obj_in.mcp_tool.mcp_subtype == MCPType.external
        ):
            obj_in.organization_id = current_user.organization_id
        else:
            # Ensure organization_id is None for system tools and internal MCP tools
            obj_in.organization_id = None

        tool = await tool_crud.create_tool(db_session=db, obj_in=obj_in)
        return create_response(
            data=IToolCreatedResponse(
                id=tool.id,
                name=tool.name,
                tool_kind=tool.tool_kind,
                dataset_required=tool.dataset_required,
            ),
            message="Tool created successfully",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Failed to create tool")
        raise HTTPException(status_code=500, detail="Failed to create tool")


@router.get(
    "/tool",
    response_model=IGetResponsePaginated[ToolReadUnion],
    summary="Get all tools (paginated)",
)
async def get_all_tools(
    params: Params = Depends(),
    search: Optional[str] = Query(
        None, description="Search tools by name or description"
    ),
    tool_kind: Optional[ToolType] = Query(
        None, description="Filter by tool kind (system or mcp)"
    ),
    mcp_type: Optional[MCPType] = Query(
        None, description="If tool_kind is 'mcp', filter by mcp type"
    ),
    dataset_required: Optional[bool] = Query(
        None, description="Filter by dataset requirement"
    ),
    db: AsyncSession = Depends(deps.get_db),
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
):
    page = await tool_crud.get_all_tools(
        db_session=db,
        params=params,
        search=search,
        tool_kind=tool_kind,
        mcp_type=mcp_type,
        dataset_required=dataset_required,
        current_user_org_id=current_user.organization_id,
    )

    converted = []
    for tool in page.items:
        try:
            converted.append(convert_tool_to_schema(tool))
        except HTTPException:
            continue

    page.items = converted
    return create_response(data=page, message="Tools retrieved successfully.")


@router.get(
    "/tool/{tool_id}",
    response_model=IGetResponseBase[ToolReadUnion],
    summary="Get tool details by ID",
)
async def get_tool_by_id(
    tool_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
):
    tool_data = await tool_crud.get_tool_by_id(
        db_session=db, tool_id=tool_id, current_user_org_id=current_user.organization_id
    )
    return create_response(data=tool_data, message="Tool retrieved successfully")


@router.get(
    "/tool/{tool_id}/mcp-tools",
    response_model=IGetResponseBase[List[dict]],
    summary="List available tools for an MCP server",
)
async def list_mcp_tools(
    tool_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
):
    """List all available tools for a specific MCP server"""
    try:
        # Get tool using existing function with organization filtering
        tool = await tool_crud.get_tool_by_id(
            db_session=db,
            tool_id=tool_id,
            current_user_org_id=current_user.organization_id,
        )
        logger.info(f"Retrieved tool: {tool.tool_type}")

        # Verify it's an MCP tool
        if tool.tool_type != ToolType.mcp:
            raise HTTPException(
                status_code=400,
                detail="Tool is not an MCP tool",
            )

        # Initialize manager and list tools using the mcp_server_config
        manager = MCPServerManager()
        tools = await manager.list_tools(tool.mcp_server_config)

        return create_response(
            data=tools,
            message="MCP tools listed successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing MCP tools: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing MCP tools: {str(e)}",
        )


@router.post(
    "/tool/batch/mcp-tools",
    response_model=IGetResponseBase[Dict[str, List[dict]]],
    summary="List available tools for multiple MCP servers",
)
async def list_batch_mcp_tools(
    request: IToolIdsList,
    db: AsyncSession = Depends(deps.get_db),
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
):
    """List all available tools for multiple MCP servers"""
    try:
        result = {}
        manager = MCPServerManager()

        for tool_id in request.tool_ids:
            try:
                # Get tool using existing function with organization filtering
                tool = await tool_crud.get_tool_by_id(
                    db_session=db,
                    tool_id=tool_id,
                    current_user_org_id=current_user.organization_id,
                )

                # Verify it's an MCP tool
                if tool.tool_type != ToolType.mcp:
                    logger.warning(f"Tool {tool_id} is not an MCP tool - skipping")
                    result[str(tool_id)] = {"error": "Tool is not an MCP tool"}
                    continue

                # List tools using the mcp_server_config
                tools = await manager.list_tools(tool.mcp_server_config)
                result[str(tool_id)] = tools

            except Exception as e:
                logger.error(f"Error processing MCP tool {tool_id}: {str(e)}")
                result[str(tool_id)] = {"error": str(e)}

        return create_response(
            data=result,
            message="Batch MCP tools listed successfully",
        )

    except Exception as e:
        logger.error(f"Error listing batch MCP tools: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing batch MCP tools: {str(e)}",
        )


@router.put(
    "/tool/{tool_id}",
    response_model=IPostResponseBase[IToolCreatedResponse],
    summary="Update an existing tool (System or MCP)",
)
async def update_tool(
    tool_id: UUID,
    obj_in: IToolUpdate,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin])
    ),
    db: AsyncSession = Depends(deps.get_db),
) -> IPostResponseBase[IToolCreatedResponse]:
    try:
        tool = await tool_crud.update_tool(
            db_session=db, tool_id=tool_id, obj_in=obj_in
        )
        return create_response(
            data=IToolCreatedResponse(
                id=tool.id,
                name=tool.name,
                tool_kind=tool.tool_kind,
                dataset_required=tool.dataset_required,
            ),
            message="Tool updated successfully",
        )
    except HTTPException as e:
        raise e  # Let FastAPI handle structured HTTP exceptions
    except Exception:
        logger.exception("Failed to update tool")
        raise HTTPException(status_code=500, detail="Failed to update tool")


@router.delete(
    "/tool/{tool_id}",
    status_code=204,
    summary="Soft delete a tool",
    description="Marks a tool as deleted (soft delete) using deleted_at timestamp.",
)
async def delete_tool(
    tool_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin])
    ),
    db: AsyncSession = Depends(deps.get_db),
) -> Response:
    await tool_crud.soft_delete_tool(db_session=db, tool_id=tool_id)
    return Response(status_code=204)
