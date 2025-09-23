from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi_pagination import Page, Params
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.api import deps
from app.be_core.config import settings
from app.be_core.logger import logger
from app.crud.log_crud import log_crud
from app.models.audit_log_model import EntityType, OperationType
from app.models.workspace_model import Workspace
from app.schemas.common_schema import IOrderEnum
from app.schemas.response_schema import (
    IGetResponseBase,
    IGetResponsePaginated,
    IPostResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData
from app.schemas.workspace_schema import (
    IWorkspaceCreate,
    IWorkspaceRead,
    IWorkspaceUpdate,
    WorkspaceUserIDList,
)
from app.utils.exceptions import IdNotFoundException

# from schemas.workspace import IWorkspaceCreate, IWorkspaceRead


router = APIRouter()


# Create Workspace Route
@router.post("/organization/{organization_id}/workspace")
async def create_workspace(
    obj_in: IWorkspaceCreate,
    organization_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.organization_check),
    db: AsyncSession = Depends(deps.get_db),  # Inject the db session
) -> IPostResponseBase[IWorkspaceRead]:
    """
    Creates a new workspace under a specific organization.

    Required roles:
    - admin
    - developer
    """
    if obj_in.description and len(obj_in.description) > settings.MAX_DESCRIPTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Description must not exceed {settings.MAX_DESCRIPTION_LENGTH} characters.",
        )

    if obj_in.name and len(obj_in.name) > settings.MAX_NAME_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Workspace name must not exceed {settings.MAX_NAME_LENGTH} characters.",
        )

    logger.info(f"Begin create_workspace for {organization_id}; Inside FastAPI Func")
    new_workspace = await crud.workspace_crud.workspace.create_workspace(
        organization_id=organization_id,
        obj_in=obj_in,  # Updated param name to match the function signature
        db_session=db,  # Passing the db session explicitly
    )
    # Add user to workspace
    await crud.workspace_crud.workspace.add_user_in_workspace(
        user_id=current_user.id, workspace_id=new_workspace.id
    )
    await log_crud.log_operation(
        operation=OperationType.Create,
        entity=EntityType.Workspace,
        entity_id=new_workspace.id,
        entity_name=new_workspace.name,
        user_id=current_user.id,
        user_name=current_user.email,
    )

    return create_response(data=new_workspace, message="Workspace created successfully")


@router.put("/organization/{organization_id}/workspace/{workspace_id}")
async def update_workspace(
    organization_id: UUID,
    workspace_id: UUID,
    obj_in: IWorkspaceUpdate,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.user_workspace_access_check),
    db: AsyncSession = Depends(deps.get_db),
) -> IGetResponseBase[IWorkspaceRead]:
    """
    Updates a workspace's details (name, description, is_active).

    Required roles:
    - admin
    - developer
    """
    if obj_in.description and len(obj_in.description) > settings.MAX_DESCRIPTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Description must not exceed {settings.MAX_DESCRIPTION_LENGTH} characters.",
        )

    if obj_in.name and len(obj_in.name) > settings.MAX_NAME_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Workspace name must not exceed {settings.MAX_NAME_LENGTH} characters.",
        )
    updated_workspace = await crud.workspace_crud.workspace.update_workspace(
        workspace_id=workspace_id,
        organization_id=organization_id,
        obj_in=obj_in,
        db_session=db,
    )
    if not updated_workspace:
        raise IdNotFoundException(Workspace, workspace_id)

    await log_crud.log_operation(
        operation=OperationType.Update,
        entity=EntityType.Workspace,
        entity_id=updated_workspace.id,
        entity_name=updated_workspace.name,
        user_id=current_user.id,
        user_name=current_user.email,
    )

    return create_response(
        data=updated_workspace, message="Workspace updated successfully"
    )


# Get Workspace by ID
@router.get("/organization/{organization_id}/workspace/{workspace_id}")
async def get_workspace_by_id(
    workspace_id: UUID,
    organization_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.developer, IRoleEnum.member]
        )
    ),
    _=Depends(deps.user_workspace_access_check),
    db: AsyncSession = Depends(deps.get_db),
) -> IGetResponseBase[IWorkspaceRead]:
    """
    Gets a workspace by its unique ID and the organization it belongs to.

    Required roles:
    - admin
    - developer
    - member
    """
    workspace = await crud.workspace_crud.workspace.get_workspace(
        workspace_id=workspace_id,
        organization_id=organization_id,
    )
    if workspace:
        return create_response(data=workspace)
    else:
        raise IdNotFoundException(Workspace, workspace_id)


@router.get(
    "/organization/{organization_id}/workspace",
    response_model=IGetResponsePaginated[IWorkspaceRead],
)
async def get_workspaces(
    organization_id: UUID,
    params: Params = Depends(),
    search: Optional[str] = Query(None, description="Search by workspace name"),
    order: IOrderEnum = Query(IOrderEnum.ascendent),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.developer, IRoleEnum.member]
        )
    ),
    _=Depends(deps.organization_check),
    db: AsyncSession = Depends(deps.get_db),
) -> IGetResponsePaginated[IWorkspaceRead]:
    """
    Gets a paginated list of all workspaces under a specific organization.

    Optional:
    - Search by workspace name.
    Required roles:
    - admin
    - developer
    - member
    """
    logger.info(f"Getting workspaces for organization {organization_id}")

    try:
        logger.info("Fetching workspaces from database")
        paginated_workspaces = await crud.workspace_crud.workspace.get_workspaces(
            organization_id=organization_id,
            params=params,
            order=order,
            user_id=current_user.id,
            user_role=current_user.role,
            search=search,
        )

        logger.info(
            f"Database query successful - found {len(paginated_workspaces.items)} workspaces"
        )

        workspaces_with_total_files = [
            IWorkspaceRead(
                **workspace.model_dump(),
                total_files=None,
            )
            for workspace in paginated_workspaces.items
        ]

        paginated_workspaces.items = workspaces_with_total_files
        logger.debug(
            f"Processed {len(workspaces_with_total_files)} workspaces with file counts"
        )

        return create_response(data=paginated_workspaces)

    except Exception as e:
        logger.error(
            f"Error in get_workspaces for organization {organization_id}: {str(e)}"
        )
        raise


# Delete Workspace by ID
@router.delete("/organization/{organization_id}/workspace/{workspace_id}")
async def delete_workspace(
    workspace_id: UUID,
    organization_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.user_workspace_access_check),
    db: AsyncSession = Depends(deps.get_db),
) -> IGetResponseBase[IWorkspaceRead]:
    """
    Deletes a workspace by its unique ID.

    Required roles:
    - admin
    - developer
    """
    workspace = await crud.workspace_crud.workspace.delete_workspace(
        workspace_id=workspace_id, organization_id=organization_id, db_session=db
    )
    if workspace:
        await log_crud.log_operation(
            operation=OperationType.Delete,
            entity=EntityType.Workspace,
            entity_id=workspace.id,
            entity_name=workspace.name,
            user_id=current_user.id,
            user_name=current_user.email,
        )

        return create_response(data=workspace, message="Workspace deleted successfully")
    else:
        raise IdNotFoundException(Workspace, workspace_id)


@router.post("/organization/{organization_id}/workspace/{workspace_id}/add_users")
async def add_users_in_workspace(
    organization_id: UUID,
    workspace_id: UUID,
    obj_in: WorkspaceUserIDList,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.user_workspace_access_check),
) -> IPostResponseBase[WorkspaceUserIDList]:
    """
    Adds new users under a specific workspace.

    Required roles:
    - Admin
    - Developer
    """
    added_users = []
    for user_id in obj_in.user_ids:
        belongs_to_org_check = (
            await crud.workspace_crud.workspace.user_belongs_to_organization(
                user_id=user_id, organization_id=organization_id
            )
        )
        if not belongs_to_org_check:
            raise HTTPException(
                status_code=400,
                detail=f"User {user_id} does not belong to organization and cannot be added to workspace {workspace_id}.",
            )

        logger.info(f"Adding user:{user_id}")
        await crud.workspace_crud.workspace.add_user_in_workspace(
            user_id=user_id, workspace_id=workspace_id
        )
        added_users.append(user_id)
    return create_response(
        data=WorkspaceUserIDList(user_ids=added_users),
        message="Added users to the workspace successfully",
    )


@router.get("/organization/{organization_id}/workspace/{workspace_id}/get_users")
async def get_users_in_workspace(
    organization_id: UUID,
    workspace_id: UUID,
    params: Params = Depends(),
    order: IOrderEnum = Query(IOrderEnum.ascendent),
    order_by: str = "created_at",
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin])
    ),
    _=Depends(deps.user_workspace_access_check),
) -> IGetResponsePaginated[UserData]:
    """
    Get all users for a workspace.

    Required roles:
    - admin
    """
    users_page = await crud.workspace.get_users_in_workspace(
        workspace_id=workspace_id, params=params, order=order, order_by=order_by
    )
    user_data_items = [
        UserData(
            id=user.id,
            email=user.email,
            organization_id=user.organization_id,
            role=([user.role.name] if user.role else []),
            is_active=user.is_active,
            full_name=(
                f"{user.first_name} {user.last_name}"
                if user.first_name and user.last_name
                else None
            ),
            first_name=user.first_name,
            last_name=user.last_name,
        )
        for user in users_page.items
    ]

    users_data_page = Page[UserData](
        items=user_data_items,
        total=users_page.total,
        page=users_page.page,
        size=users_page.size,
    )

    return create_response(data=users_data_page)


@router.post("/organization/{organization_id}/workspace/{workspace_id}/remove_users")
async def remove_users_in_workspace(
    organization_id: UUID,
    workspace_id: UUID,
    obj_in: WorkspaceUserIDList,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin])
    ),
    _=Depends(deps.user_workspace_access_check),
) -> IGetResponseBase[WorkspaceUserIDList]:
    """
    Remove users from a specific workspace.

    Required roles:
    - admin
    """
    deleted_user_id = await crud.workspace_crud.workspace.delete_users_from_workspaces(
        workspace_id=workspace_id, user_ids=obj_in.user_ids
    )
    return create_response(
        data=WorkspaceUserIDList(user_ids=deleted_user_id),
        message="Users deleted successfully",
    )
