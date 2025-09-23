from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response
from fastapi_pagination import Params
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.api import deps
from app.be_core.logger import logger
from app.models.workspace_agent_model import WorkspaceAgent
from app.schemas.agent_schema import (
    IAgentCreate,
    IAgentUpdate,
    IWorkspaceAgentRead,
)
from app.schemas.common_schema import IOrderEnum
from app.schemas.response_schema import (
    IGetResponseBase,
    IGetResponsePaginated,
    IPostResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData
from app.utils.exceptions import IdNotFoundException

router = APIRouter()


@router.post("/workspace/{workspace_id}/agent")
async def create_agent(
    workspace_id: UUID,
    obj_in: IAgentCreate,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.user_workspace_access_check),
    db: AsyncSession = Depends(deps.get_db),
) -> IPostResponseBase[IWorkspaceAgentRead]:
    """
    Create a new agent for the workspace.

    Required roles:
    - admin
    - developer
    """
    logger.info(f"Begin create_agent for workspace {workspace_id}")
    agent = await crud.agent.create_agent(
        db_session=db, workspace_id=workspace_id, obj_in=obj_in  # <-- changed here
    )
    # await log_crud.log_operation(
    #     operation=OperationType.Create,
    #     entity=EntityType.Agent,
    #     entity_id=agent.id,
    #     entity_name=agent.name,
    #     user_id=current_user.id,
    #     user_name=current_user.email,
    # )

    return create_response(data=agent, message="Agent created successfully")


@router.get(
    "/workspace/{workspace_id}/agent/{agent_id}",
    response_model=IGetResponseBase[IWorkspaceAgentRead],
    summary="Get agent by ID within a workspace",
)
async def get_agent_by_id(
    workspace_id: UUID,
    agent_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
):
    agent_data = await crud.agent.get_agent_by_id_in_workspace(
        db_session=db,
        agent_id=agent_id,
        workspace_id=workspace_id,
    )
    return create_response(data=agent_data, message="Agent retrieved successfully")


@router.get(
    "/workspace/{workspace_id}/agent",
    response_model=IGetResponsePaginated[IWorkspaceAgentRead],
    summary="List agents in workspace (paginated)",
)
async def list_agents(
    workspace_id: UUID,
    search: Optional[str] = Query(None, description="Search by agent name"),
    order: IOrderEnum = Query(IOrderEnum.ascendent),
    order_by: str = Query("created_at"),
    params: Params = Depends(),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.developer, IRoleEnum.member]
        )
    ),
    _=Depends(deps.user_workspace_access_check),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    List all agents for a workspace with pagination, search, and ordering support.

    Required roles:
    - admin
    - developer
    - member
    """
    page = await crud.agent.get_agents(
        db_session=db,
        workspace_id=workspace_id,
        search=search,
        order=order,
        order_by=order_by,
        params=params,
    )
    return create_response(data=page, message="Agents retrieved successfully.")


@router.delete(
    "/workspace/{workspace_id}/agent/{agent_id}",
    status_code=204,
    summary="Unassign an agent from workspace",
    description="Soft deletes the association between a workspace and an agent.",
)
async def delete_agent(
    workspace_id: UUID,
    agent_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.user_workspace_access_check),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Soft deletes the workspace-agent association.

    Required roles:
    - admin
    - developer
    """
    deleted = await crud.agent.delete_workspace_agent(
        workspace_id=workspace_id,
        agent_id=agent_id,
        db_session=db,
    )

    if not deleted:
        raise IdNotFoundException(
            WorkspaceAgent, f"{workspace_id}::{agent_id} (or it was already deleted)"
        )

    # await log_crud.log_operation(
    #     operation=OperationType.Delete,
    #     entity=EntityType.WorkspaceAgent,
    #     entity_id=str(agent_id),
    #     entity_name=f"Agent {agent_id} from Workspace {workspace_id}",
    #     user_id=current_user.id,
    #     user_name=current_user.email,
    # )

    return Response(status_code=204)


@router.put("/workspace/{workspace_id}/agent/{agent_id}")
async def update_agent(
    workspace_id: UUID,
    agent_id: UUID,
    obj_in: IAgentUpdate,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.user_workspace_access_check),
    db: AsyncSession = Depends(deps.get_db),
) -> IPostResponseBase[IWorkspaceAgentRead]:
    """
    Update an existing agent in the workspace.
    """
    agent = await crud.agent.update_agent(
        agent_id=agent_id,
        obj_in=obj_in,
        db_session=db,
        workspace_id=workspace_id,
    )
    return create_response(data=agent, message="Agent updated successfully")
