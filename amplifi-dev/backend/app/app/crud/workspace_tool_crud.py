from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import HTTPException
from fastapi_pagination import Page, Params
from fastapi_pagination.ext.sqlmodel import paginate
from sqlalchemy import and_, func, or_, update
from sqlalchemy.orm import selectinload
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.crud.base_crud import CRUDBase
from app.models.system_tools import SystemTool
from app.models.tools_models import Tool, ToolType
from app.models.workspace_tools import WorkspaceTool
from app.schemas.workspace_tool_schema import IWorkspaceToolAdd
from app.utils.tool_validator import (
    validate_tool_kinds,
    validate_tool_specific_constraints,
    validate_tools_exist,
)


class CRUDWorkspaceTool(CRUDBase[WorkspaceTool, IWorkspaceToolAdd, None]):
    async def assign_tool_to_workspace(
        self,
        workspace_id: UUID,
        obj_in: IWorkspaceToolAdd,
        db_session: AsyncSession,
    ) -> WorkspaceTool:
        result = await db_session.execute(
            select(WorkspaceTool).where(
                and_(
                    WorkspaceTool.workspace_id == workspace_id,
                    WorkspaceTool.name.ilike(obj_in.name),
                    WorkspaceTool.deleted_at.is_(None),
                )
            )
        )
        existing = result.scalars().first()
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"A tool with name '{obj_in.name}' already exists in this workspace.",
            )

        result = await db_session.execute(
            select(Tool).where(Tool.id.in_(obj_in.tool_ids))
        )
        tools = result.scalars().all()

        validate_tools_exist(tools, obj_in.tool_ids)
        tool_kind = validate_tool_kinds(tools)
        validate_tool_specific_constraints(tool_kind, obj_in)

        db_obj = WorkspaceTool(
            workspace_id=workspace_id,
            name=obj_in.name,
            description=obj_in.description,
            tool_ids=obj_in.tool_ids,
            dataset_ids=obj_in.dataset_ids if tool_kind == ToolType.system else [],
            mcp_tools=obj_in.mcp_tools if tool_kind == ToolType.mcp else [],
        )
        db_session.add(db_obj)
        await db_session.commit()
        await db_session.refresh(db_obj)
        return db_obj

    async def get_workspace_tool_by_id(
        self,
        *,
        db_session: AsyncSession,
        workspace_id: UUID,
        workspace_tool_id: UUID,
    ) -> Optional[WorkspaceTool]:
        query = select(WorkspaceTool).where(
            WorkspaceTool.id == workspace_tool_id,
            WorkspaceTool.workspace_id == workspace_id,
            WorkspaceTool.deleted_at.is_(None),
        )
        result = await db_session.execute(query)
        workspace_tool = result.scalar_one_or_none()

        if not workspace_tool or not workspace_tool.tool_ids:
            return workspace_tool

        tool_query = (
            select(Tool)
            .where(Tool.id.in_(workspace_tool.tool_ids))
            .options(
                selectinload(Tool.system_tool),
                selectinload(Tool.mcp_tool),
            )
        )
        tool_result = await db_session.execute(tool_query)
        tools = tool_result.scalars().all()

        # Attach tools manually for response use
        workspace_tool.tools = tools
        return workspace_tool

    async def get_workspace_tools_by_ids(
        self,
        *,
        workspace_id: UUID,
        workspace_tool_ids: list[UUID],
        db_session: Optional[AsyncSession] = None,
    ) -> list[WorkspaceTool]:
        # Step 1: Fetch WorkspaceTools
        db_session = db_session or super().get_db().session

        query = select(WorkspaceTool).where(
            WorkspaceTool.id.in_(workspace_tool_ids),
            WorkspaceTool.workspace_id == workspace_id,
            WorkspaceTool.deleted_at.is_(None),
        )
        result = await db_session.execute(query)
        workspace_tools = result.scalars().all()

        if not workspace_tools:
            return []

        # Step 2: Collect all tool_ids from all workspace_tools
        all_tool_ids = set()
        for wt in workspace_tools:
            all_tool_ids.update(getattr(wt, "tool_ids", []))

        if not all_tool_ids:
            return workspace_tools

        # Step 3: Fetch Tools in bulk
        tool_query = (
            select(Tool)
            .where(Tool.id.in_(all_tool_ids))
            .options(
                selectinload(Tool.system_tool),
                selectinload(Tool.mcp_tool),
            )
        )
        tool_result = await db_session.execute(tool_query)
        all_tools = tool_result.scalars().all()

        # Step 4: Create a tool lookup dictionary
        tool_map = {tool.id: tool for tool in all_tools}

        # Step 5: Attach tools to each workspace_tool
        for wt in workspace_tools:
            wt.tools = [
                tool_map[tid] for tid in getattr(wt, "tool_ids", []) if tid in tool_map
            ]

        return workspace_tools

    async def get_workspace_tools(
        self,
        *,
        db_session: AsyncSession,
        workspace_id: UUID,
        params: Params,
        search: Optional[str] = None,
        tool_kind: Optional[ToolType] = None,
    ):
        query = (
            select(WorkspaceTool)
            .where(
                WorkspaceTool.workspace_id == workspace_id,
                WorkspaceTool.deleted_at.is_(None),
            )
            .options(
                selectinload(WorkspaceTool.tools).selectinload(Tool.system_tool),
                selectinload(WorkspaceTool.tools).selectinload(Tool.mcp_tool),
            )
            .order_by(WorkspaceTool.created_at.desc())
        )

        if search or tool_kind:
            query = query.join(WorkspaceTool.tools)

        if search:
            search_pattern = f"%{search.lower()}%"
            query = query.where(
                or_(
                    func.lower(WorkspaceTool.name).like(search_pattern),
                    func.lower(WorkspaceTool.description).like(search_pattern),
                    func.lower(Tool.name).like(search_pattern),
                    func.lower(Tool.description).like(search_pattern),
                )
            )

        if tool_kind:
            query = query.where(Tool.tool_kind == tool_kind)
        return await paginate(db_session, query, params)

    async def update_workspace_tool_by_id(
        self,
        workspace_id: UUID,
        workspace_tool_id: UUID,
        obj_in: IWorkspaceToolAdd,
        db_session: AsyncSession,
    ) -> Optional[WorkspaceTool]:
        if not obj_in.tool_ids:
            raise HTTPException(status_code=400, detail="tool_ids must be provided")

        result = await db_session.execute(
            select(WorkspaceTool).where(
                and_(
                    WorkspaceTool.name.ilike(obj_in.name),
                    WorkspaceTool.workspace_id == workspace_id,
                    WorkspaceTool.deleted_at.is_(None),
                    WorkspaceTool.id != workspace_tool_id,
                )
            )
        )
        duplicate_tool = result.scalar_one_or_none()
        if duplicate_tool:
            raise HTTPException(
                status_code=400,
                detail=f"A workspace tool with name '{obj_in.name}' already exists in this workspace.",
            )

        result = await db_session.execute(
            select(WorkspaceTool).where(
                WorkspaceTool.id == workspace_tool_id,
                WorkspaceTool.workspace_id == workspace_id,
                WorkspaceTool.deleted_at.is_(None),
            )
        )
        db_obj = result.scalar_one_or_none()
        if db_obj is None:
            return None

        result = await db_session.execute(
            select(Tool).where(Tool.id.in_(obj_in.tool_ids))
        )
        tools = result.scalars().all()

        validate_tools_exist(tools, obj_in.tool_ids)
        tool_kind = validate_tool_kinds(tools)
        validate_tool_specific_constraints(tool_kind, obj_in)

        db_obj.name = obj_in.name
        db_obj.description = obj_in.description
        db_obj.tool_ids = obj_in.tool_ids
        db_obj.dataset_ids = obj_in.dataset_ids if tool_kind == ToolType.system else []
        db_obj.mcp_tools = obj_in.mcp_tools if tool_kind == ToolType.mcp else []

        db_session.add(db_obj)
        await db_session.commit()
        await db_session.refresh(db_obj)
        return db_obj

    async def delete_workspace_tool(
        self, workspace_id: UUID, workspace_tool_id: UUID, db_session: AsyncSession
    ) -> None:
        statement = (
            update(WorkspaceTool)
            .where(
                WorkspaceTool.workspace_id == workspace_id,
                WorkspaceTool.id == workspace_tool_id,
                WorkspaceTool.deleted_at.is_(None),
            )
            .values(deleted_at=datetime.utcnow())
            .execution_options(synchronize_session="fetch")
        )
        result = await db_session.execute(statement)
        await db_session.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Workspace tool not found.")

    async def get_system_tools(
        self,
        db_session: AsyncSession,
        params: Params,
        search: Optional[str] = None,
    ) -> Page[SystemTool]:
        query = (
            select(SystemTool)
            .options(selectinload(SystemTool.tool).selectinload(Tool.system_tool))
            .join(SystemTool.tool)
            .where(SystemTool.deleted_at.is_(None))
            .order_by(SystemTool.created_at.desc())
        )

        if search:
            search_pattern = f"%{search.lower()}%"
            query = query.where(
                or_(
                    func.lower(Tool.name).like(search_pattern),
                    func.lower(Tool.description).like(search_pattern),
                )
            )

        return await paginate(db_session, query, params)


workspace_tool_crud = CRUDWorkspaceTool(WorkspaceTool)
