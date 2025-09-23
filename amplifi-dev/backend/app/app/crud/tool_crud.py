from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import HTTPException
from fastapi_pagination import Params
from fastapi_pagination.ext.sqlalchemy import paginate
from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.crud.base_crud import CRUDBase
from app.models.mcp_tools import MCPTool, MCPType
from app.models.system_tools import SystemTool
from app.models.tools_models import Tool, ToolType
from app.schemas.tool_schema import IToolCreate, IToolUpdate
from app.schemas.workspace_tool_schema import (
    ToolReadUnion,
)
from app.utils.tool_converter import convert_tool_to_schema


class CRUDTool(CRUDBase[Tool, IToolCreate, IToolUpdate]):
    async def create_tool(self, db_session: AsyncSession, obj_in: IToolCreate) -> Tool:
        try:
            result = await db_session.execute(
                select(Tool).where(
                    func.lower(Tool.name) == obj_in.name.lower(),
                    Tool.deleted_at.is_(None),
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                raise ValueError(
                    f"A tool with the name '{obj_in.name}' already exists."
                )

            # Set organization_id only for external MCP tools
            organization_id = None
            if obj_in.tool_kind == ToolType.mcp and obj_in.mcp_tool:
                from app.models.mcp_tools import MCPType

                if obj_in.mcp_tool.mcp_subtype == MCPType.external:
                    organization_id = getattr(obj_in, "organization_id", None)

            db_tool = Tool(
                name=obj_in.name,
                description=obj_in.description,
                deprecated=obj_in.deprecated,
                tool_kind=obj_in.tool_kind,
                tool_metadata=obj_in.metadata or {},
                organization_id=organization_id,
                dataset_required=obj_in.dataset_required,
            )
            db_session.add(db_tool)
            await db_session.flush()

            if obj_in.tool_kind == ToolType.system:
                if not obj_in.system_tool:
                    raise ValueError("System tool details must be provided")
                system = obj_in.system_tool
                db_session.add(
                    SystemTool(
                        tool_id=db_tool.id,
                        python_module=system.python_module,
                        function_name=system.function_name,
                        is_async=system.is_async,
                        input_schema=system.input_schema,
                        output_schema=system.output_schema,
                        function_signature=system.function_signature,
                    )
                )
            elif obj_in.tool_kind == ToolType.mcp:
                mcp = obj_in.mcp_tool
                db_session.add(
                    MCPTool(
                        tool_id=db_tool.id,
                        mcp_subtype=mcp.mcp_subtype,
                        mcp_server_config=mcp.mcp_server_config,
                        timeout_secs=mcp.timeout_secs,
                    )
                )

            await db_session.commit()
            await db_session.refresh(db_tool)
            return db_tool
        except Exception as e:
            await db_session.rollback()
            raise HTTPException(status_code=400, detail=str(e))

    async def get_tool_by_id(
        self,
        db_session: AsyncSession,
        tool_id: UUID,
        current_user_org_id: Optional[UUID] = None,
    ) -> ToolReadUnion:
        query = (
            select(Tool)
            .where(Tool.id == tool_id, Tool.deleted_at.is_(None))
            .options(
                selectinload(Tool.system_tool),
                selectinload(Tool.mcp_tool),
            )
        )

        # Apply organization filtering for external MCP tools
        if current_user_org_id:
            query = query.where(
                or_(
                    Tool.organization_id.is_(None),
                    Tool.organization_id == current_user_org_id,
                )
            )
        else:
            query = query.where(Tool.organization_id.is_(None))

        result = await db_session.execute(query)
        tool = result.scalar_one_or_none()

        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")

        return convert_tool_to_schema(tool)

    async def get_tools_by_ids(
        self,
        tool_ids: list[UUID],
        db_session: Optional[AsyncSession] = None,
        current_user_org_id: Optional[UUID] = None,
    ) -> list[ToolReadUnion]:
        db_session = db_session or super().get_db().session
        query = (
            select(Tool)
            .where(Tool.id.in_(tool_ids), Tool.deleted_at.is_(None))
            .options(
                selectinload(Tool.system_tool),
                selectinload(Tool.mcp_tool),
            )
        )

        # Apply organization filtering for external MCP tools
        if current_user_org_id:
            query = query.where(
                or_(
                    Tool.organization_id.is_(
                        None
                    ),  # Global tools (system and internal MCP)
                    Tool.organization_id
                    == current_user_org_id,  # Organization's external MCP tools
                )
            )
        else:
            # If no user org provided, only show global tools
            query = query.where(Tool.organization_id.is_(None))

        result = await db_session.execute(query)
        tools = result.scalars().all()

        if not tools:
            raise HTTPException(status_code=404, detail="No tools found")

        return [convert_tool_to_schema(tool) for tool in tools]

    async def get_all_tools(
        self,
        *,
        db_session: AsyncSession,
        params: Params,
        search: Optional[str] = None,
        tool_kind: Optional[ToolType] = None,
        mcp_type: Optional[MCPType] = None,
        dataset_required: Optional[bool] = None,
        current_user_org_id: Optional[UUID] = None,
    ):
        query = (
            select(Tool)
            .where(Tool.deleted_at.is_(None))
            .options(
                selectinload(Tool.system_tool),
                selectinload(Tool.mcp_tool),
            )
            .order_by(Tool.created_at.desc())
        )

        # Filter tools based on organization access:
        # - System tools: always visible (organization_id is NULL)
        # - Internal MCP tools: always visible (organization_id is NULL)
        # - External MCP tools: only visible to the organization that created them
        if current_user_org_id:
            query = query.where(
                or_(
                    Tool.organization_id.is_(
                        None
                    ),  # Global tools (system and internal MCP)
                    Tool.organization_id
                    == current_user_org_id,  # Organization's external MCP tools
                )
            )
        else:
            # If no user org provided, only show global tools
            query = query.where(Tool.organization_id.is_(None))

        if search:
            search_pattern = f"%{search.lower()}%"
            query = query.where(
                or_(
                    func.lower(Tool.name).like(search_pattern),
                    func.lower(Tool.description).like(search_pattern),
                )
            )

        if tool_kind:
            query = query.where(Tool.tool_kind == tool_kind)

        if tool_kind == ToolType.mcp and mcp_type:
            query = query.where(
                Tool.mcp_tool.has(MCPTool.mcp_subtype == mcp_type)
            )  # Use relationship filter

        if dataset_required is not None:
            query = query.where(Tool.dataset_required == dataset_required)

        return await paginate(db_session, query, params)

    async def update_tool(
        self, db_session: AsyncSession, tool_id: UUID, obj_in: IToolUpdate
    ) -> Tool:
        db_tool = await db_session.get(Tool, tool_id)

        if not db_tool or db_tool.deleted_at is not None:
            raise HTTPException(
                status_code=404, detail="Tool not found or already deleted"
            )

        result = await db_session.execute(
            select(Tool).where(
                func.lower(Tool.name) == obj_in.name.lower(),
                Tool.id != tool_id,
                Tool.deleted_at.is_(None),
            )
        )
        duplicate = result.scalar_one_or_none()
        if duplicate:
            raise HTTPException(
                status_code=400,
                detail=f"A tool with the name '{obj_in.name}' already exists.",
            )

        # Update base fields
        db_tool.name = obj_in.name
        db_tool.description = obj_in.description
        db_tool.deprecated = obj_in.deprecated
        db_tool.tool_metadata = obj_in.metadata or {}
        if obj_in.dataset_required is not None:
            db_tool.dataset_required = obj_in.dataset_required

        # Update subtype-specific data
        if db_tool.tool_kind == ToolType.system:
            result = await db_session.execute(
                select(SystemTool).where(SystemTool.tool_id == tool_id)
            )
            system_tool = result.scalar_one_or_none()
            if not system_tool:
                raise HTTPException(status_code=404, detail="System tool not found")
            if not obj_in.system_tool:
                raise HTTPException(status_code=422, detail="Missing system tool input")

            system_tool.python_module = obj_in.system_tool.python_module
            system_tool.function_name = obj_in.system_tool.function_name
            system_tool.is_async = obj_in.system_tool.is_async
            system_tool.input_schema = obj_in.system_tool.input_schema
            system_tool.output_schema = obj_in.system_tool.output_schema
            system_tool.function_signature = obj_in.system_tool.function_signature

        elif db_tool.tool_kind == ToolType.mcp:
            result = await db_session.execute(
                select(MCPTool).where(MCPTool.tool_id == tool_id)
            )
            mcp_tool = result.scalar_one_or_none()
            if not mcp_tool:
                raise HTTPException(status_code=404, detail="MCP tool not found")
            if not obj_in.mcp_tool:
                raise HTTPException(status_code=422, detail="Missing MCP tool input")

            mcp_tool.mcp_subtype = obj_in.mcp_tool.mcp_subtype
            mcp_tool.mcp_server_config = obj_in.mcp_tool.mcp_server_config
            mcp_tool.timeout_secs = obj_in.mcp_tool.timeout_secs

        await db_session.commit()
        await db_session.refresh(db_tool)
        return db_tool

    async def soft_delete_tool(self, db_session: AsyncSession, tool_id: UUID) -> None:
        stmt = (
            update(Tool)
            .where(Tool.id == tool_id, Tool.deleted_at.is_(None))
            .values(deleted_at=datetime.utcnow())
            .execution_options(synchronize_session="fetch")
        )
        result = await db_session.execute(stmt)
        await db_session.commit()

        if result.rowcount == 0:
            raise HTTPException(
                status_code=404, detail="Tool not found or already deleted"
            )

    async def get_tool_by_name(
        self, db_session: AsyncSession, name: str
    ) -> Optional[Tool]:
        query = select(Tool).where(Tool.name == name, Tool.deleted_at.is_(None))
        result = await db_session.execute(query)
        return result.scalar_one_or_none()


tool_crud = CRUDTool(Tool)
