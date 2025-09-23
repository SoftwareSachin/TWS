from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import HTTPException
from fastapi_pagination import Params
from fastapi_pagination.ext.async_sqlalchemy import paginate
from sqlalchemy import any_, delete, func, select, update
from sqlalchemy.orm import selectinload
from sqlmodel import asc, desc
from sqlmodel.ext.asyncio.session import AsyncSession

from app.crud.base_crud import CRUDBase
from app.models.agent_model import Agent
from app.models.agent_tools_model import AgentTool
from app.models.tools_models import Tool
from app.models.workspace_agent_model import WorkspaceAgent
from app.models.workspace_tools import WorkspaceTool
from app.schemas.agent_schema import (
    IAgentCreate,
    IAgentUpdate,
    IWorkspaceAgentRead,
    WorkspaceToolInfo,
)
from app.schemas.common_schema import IOrderEnum

DEFAULT_PROMPT_INSTRUCTIONS = (
    "You are a helpful AI agent. Answer queries accurately and concisely."
)


def convert_agent_to_schema(
    agent: Agent,
    workspace_tool_ids: List[UUID],
    tool_type: Optional[str] = None,
) -> IWorkspaceAgentRead:
    return IWorkspaceAgentRead.from_model(
        agent,
        workspace_tool_ids=workspace_tool_ids,
        tool_type=tool_type,
    )


class CRUDAgent(CRUDBase[Agent, IAgentCreate, IWorkspaceAgentRead]):

    async def _get_workspace_tools_info(
        self,
        db_session: AsyncSession,
        workspace_tool_ids: List[UUID],
    ) -> List[WorkspaceToolInfo]:
        """Fetch workspace tool information for given tool IDs."""
        if not workspace_tool_ids:
            return []

        result = await db_session.execute(
            select(
                WorkspaceTool.id, WorkspaceTool.name, WorkspaceTool.description
            ).where(WorkspaceTool.id.in_(workspace_tool_ids))
        )

        return [
            WorkspaceToolInfo(id=row[0], name=row[1], description=row[2])
            for row in result.fetchall()
        ]

    async def create_agent(
        self,
        *,
        obj_in: IAgentCreate,
        workspace_id: Optional[UUID] = None,
        db_session: Optional[AsyncSession] = None,
    ) -> IWorkspaceAgentRead:
        db_session = db_session or super().get_db().session

        if not obj_in.workspace_tool_ids:
            raise HTTPException(
                status_code=400,
                detail="At least one tool must be provided to create an agent.",
            )

        # Check for agent name uniqueness in this workspace
        result = await db_session.execute(
            select(Agent)
            .join(WorkspaceAgent, WorkspaceAgent.agent_id == Agent.id)
            .where(
                Agent.name.ilike(obj_in.name),
                WorkspaceAgent.workspace_id == workspace_id,
                Agent.deleted_at.is_(None),
                WorkspaceAgent.deleted_at.is_(None),
            )
        )
        if result.scalars().first():
            raise HTTPException(
                status_code=400,
                detail=f"An agent with name '{obj_in.name}' already exists in this workspace.",
            )

        # Validate tools exist and belong to workspace
        await self._validate_workspace_tools(
            db_session, workspace_id, obj_in.workspace_tool_ids
        )

        # Create Agent
        agent = Agent(
            name=obj_in.name,
            description=obj_in.description,
            prompt_instructions=obj_in.prompt_instructions
            or DEFAULT_PROMPT_INSTRUCTIONS,
            llm_provider=obj_in.llm_provider,
            llm_model=obj_in.llm_model,
            temperature=obj_in.temperature or 0.7,
            system_prompt=obj_in.system_prompt,
            memory_enabled=obj_in.memory_enabled or False,
            agent_metadata=obj_in.agent_metadata,
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        # Link to workspace
        if workspace_id:
            db_session.add(WorkspaceAgent(workspace_id=workspace_id, agent_id=agent.id))
            await db_session.commit()

        # Link tools
        db_session.add_all(
            [
                AgentTool(agent_id=agent.id, workspace_tool_id=tool_id)
                for tool_id in obj_in.workspace_tool_ids
            ]
        )
        await db_session.commit()

        # Fetch workspace_tool_ids for response
        result = await db_session.execute(
            select(AgentTool.workspace_tool_id).where(AgentTool.agent_id == agent.id)
        )
        workspace_tool_ids = [row[0] for row in result.fetchall()]

        workspace_tools = await self._get_workspace_tools_info(
            db_session, workspace_tool_ids
        )

        # Fetch tool_type of first associated tool
        tool_type_result = await db_session.execute(
            select(Tool.tool_kind)
            .join(WorkspaceTool, Tool.id == any_(WorkspaceTool.tool_ids))
            .join(AgentTool, AgentTool.workspace_tool_id == WorkspaceTool.id)
            .where(AgentTool.agent_id == agent.id)
            .limit(1)
        )
        tool_type = tool_type_result.scalar_one_or_none()

        return IWorkspaceAgentRead.from_model(
            agent,
            workspace_tool_ids=workspace_tool_ids,
            workspace_tools=workspace_tools,
            tool_type=tool_type.value if tool_type else None,
        )

    async def _validate_workspace_tools(
        self,
        db_session: AsyncSession,
        workspace_id: UUID,
        workspace_tool_ids: list[UUID],
    ):
        result = await db_session.execute(
            select(WorkspaceTool.id).where(
                WorkspaceTool.workspace_id == workspace_id,
                WorkspaceTool.id.in_(workspace_tool_ids),
                WorkspaceTool.deleted_at.is_(None),
            )
        )
        valid_ids = {row[0] for row in result.fetchall()}
        invalid_ids = set(workspace_tool_ids) - valid_ids

        if invalid_ids:
            raise HTTPException(
                status_code=400,
                detail=f"The following tool IDs are not available in workspace {workspace_id}: {list(invalid_ids)}",
            )

    async def get_agent_by_id_in_workspace(
        self,
        *,
        db_session: AsyncSession | None = None,
        agent_id: UUID,
        workspace_id: UUID,
    ) -> IWorkspaceAgentRead:
        db_session = db_session or super().get_db().session
        query = (
            select(Agent)
            .join(WorkspaceAgent, WorkspaceAgent.agent_id == Agent.id)
            .where(
                Agent.id == agent_id,
                WorkspaceAgent.workspace_id == workspace_id,
                Agent.deleted_at.is_(None),
            )
        )
        result = await db_session.execute(query)
        agent = result.scalar_one_or_none()

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Fetch related workspace_tool_ids
        tools_query = select(AgentTool.workspace_tool_id).where(
            AgentTool.agent_id == agent_id
        )
        tools_result = await db_session.execute(tools_query)
        workspace_tool_ids = [row[0] for row in tools_result.fetchall()]

        workspace_tools = await self._get_workspace_tools_info(
            db_session, workspace_tool_ids
        )

        # Fetch tool_type of first associated tool
        tool_type = None
        if workspace_tool_ids:
            tool_type_result = await db_session.execute(
                select(Tool.tool_kind)
                .join(WorkspaceTool, Tool.id == any_(WorkspaceTool.tool_ids))
                .where(WorkspaceTool.id.in_(workspace_tool_ids))
                .limit(1)
            )
            tool_type = tool_type_result.scalar_one_or_none()

        return IWorkspaceAgentRead.from_model(
            agent,
            workspace_tool_ids=workspace_tool_ids,
            workspace_tools=workspace_tools,
            tool_type=tool_type.value if tool_type else None,
        )

    async def get_agents(
        self,
        *,
        db_session: AsyncSession,
        workspace_id: UUID,
        search: Optional[str] = None,
        params: Params,
        order: IOrderEnum = IOrderEnum.ascendent,
        order_by: str = "created_at",
    ):
        columns = Agent.__table__.columns
        if order_by not in columns:
            order_by = "created_at"

        order_clause = (
            asc(columns[order_by])
            if order == IOrderEnum.ascendent
            else desc(columns[order_by])
        )

        query = (
            select(Agent)
            .join(WorkspaceAgent, WorkspaceAgent.agent_id == Agent.id)
            .options(selectinload(Agent.tools))  # Load related AgentTool entries
            .where(
                WorkspaceAgent.workspace_id == workspace_id,
                WorkspaceAgent.deleted_at.is_(None),
            )
            .order_by(order_clause)
        )

        if search:
            query = query.where(func.lower(Agent.name).like(f"%{search.lower()}%"))

        page = await paginate(db_session, query, params)

        converted = []
        for agent in page.items:
            workspace_tool_ids = [
                tool.workspace_tool_id
                for tool in agent.tools
                if tool.workspace_tool_id is not None
            ]

            workspace_tools = await self._get_workspace_tools_info(
                db_session, workspace_tool_ids
            )

            tool_type = None
            if workspace_tool_ids:
                tool_type_result = await db_session.execute(
                    select(Tool.tool_kind)
                    .join(WorkspaceTool, Tool.id == any_(WorkspaceTool.tool_ids))
                    .where(WorkspaceTool.id.in_(workspace_tool_ids))
                    .limit(1)
                )
                tool_type = tool_type_result.scalar_one_or_none()

            converted.append(
                IWorkspaceAgentRead.from_model(
                    agent,
                    workspace_tool_ids=workspace_tool_ids,
                    workspace_tools=workspace_tools,
                    tool_type=tool_type.value if tool_type else None,
                )
            )

        page.items = converted
        return page

    async def update_agent(
        self,
        *,
        agent_id: UUID,
        obj_in: IAgentUpdate,
        db_session: AsyncSession,
        workspace_id: Optional[UUID] = None,
    ) -> IWorkspaceAgentRead:
        agent = await db_session.get(Agent, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Optional: prevent duplicate name if name is changing
        if obj_in.name and obj_in.name != agent.name:
            result = await db_session.execute(
                select(Agent).where(
                    Agent.name.ilike(obj_in.name), Agent.deleted_at.is_(None)
                )
            )
            if result.scalars().first():
                raise HTTPException(
                    status_code=400,
                    detail=f"Agent with name '{obj_in.name}' already exists.",
                )

        agent_fields = {column.name for column in Agent.__table__.columns}

        for field, value in obj_in.dict(exclude_unset=True).items():
            if field in agent_fields:
                setattr(agent, field, value)

        # Update tool mapping if workspace_tool_ids provided
        if obj_in.workspace_tool_ids is not None:
            if not obj_in.workspace_tool_ids:
                raise HTTPException(
                    status_code=400,
                    detail="At least one tool must be provided to update agent.",
                )

            await self._validate_workspace_tools(
                db_session, workspace_id, obj_in.workspace_tool_ids
            )

            # Delete existing tools
            await db_session.execute(
                delete(AgentTool).where(AgentTool.agent_id == agent_id)
            )
            # Reassign tools
            db_session.add_all(
                [
                    AgentTool(agent_id=agent.id, workspace_tool_id=wt_id)
                    for wt_id in obj_in.workspace_tool_ids
                ]
            )

        await db_session.commit()
        await db_session.refresh(agent)

        workspace_tool_ids = [
            row[0]
            for row in (
                await db_session.execute(
                    select(AgentTool.workspace_tool_id).where(
                        AgentTool.agent_id == agent.id
                    )
                )
            ).fetchall()
        ]

        workspace_tools = await self._get_workspace_tools_info(
            db_session, workspace_tool_ids
        )

        # Fetch tool_type of first associated tool
        tool_type = None
        if workspace_tool_ids:
            tool_type_result = await db_session.execute(
                select(Tool.tool_kind)
                .join(WorkspaceTool, Tool.id == any_(WorkspaceTool.tool_ids))
                .join(AgentTool, AgentTool.workspace_tool_id == WorkspaceTool.id)
                .where(AgentTool.agent_id == agent.id)
                .limit(1)
            )
            tool_type = tool_type_result.scalar_one_or_none()

        return IWorkspaceAgentRead.from_model(
            agent,
            workspace_tool_ids=workspace_tool_ids,
            workspace_tools=workspace_tools,
            tool_type=tool_type.value if tool_type else None,
        )

    async def delete_workspace_agent(
        self,
        workspace_id: UUID,
        agent_id: UUID,
        db_session: AsyncSession,
    ) -> bool:
        statement = (
            update(WorkspaceAgent)
            .where(
                WorkspaceAgent.workspace_id == workspace_id,
                WorkspaceAgent.agent_id == agent_id,
                WorkspaceAgent.deleted_at.is_(None),
            )
            .values(deleted_at=datetime.utcnow())
            .execution_options(synchronize_session="fetch")
        )
        result = await db_session.execute(statement)
        await db_session.commit()
        return result.rowcount > 0


agent = CRUDAgent(Agent)
