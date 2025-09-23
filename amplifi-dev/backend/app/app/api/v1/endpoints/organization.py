from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi_pagination import Params
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.api import deps
from app.schemas.organization_schema import (
    IOrganizationCreate,
    IOrganizationRead,
    IOrganizationUpdate,
)
from app.schemas.response_schema import (
    IDeleteResponseBase,
    IGetResponseBase,
    IGetResponsePaginated,
    IPostResponseBase,
    IPutResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData
from app.utils.exceptions.common_exception import IdNotFoundException

router = APIRouter()


@router.post("/organization")
async def create_organization(
    obj_in: IOrganizationCreate,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    db: AsyncSession = Depends(deps.get_db),  # Inject the db session
) -> IPostResponseBase[IOrganizationRead]:
    """
    Creates a new Organization.

    Required roles:
    - admin
    - developer
    """
    new_organization = await crud.organization.create_organization(
        obj_in=obj_in,  # Updated param name to match the function signature
        db_session=db,  # Passing the db session explicitly
    )
    if not new_organization:
        raise HTTPException(
            status_code=500, detail=f"Failed to create Organization {obj_in.name}"
        )
    return create_response(
        data=new_organization, message="Organization created successfully"
    )


@router.get("/organization")
async def get_organization(
    params: Params = Depends(),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    db: AsyncSession = Depends(deps.get_db),  # Inject the db session
) -> IGetResponsePaginated[IOrganizationRead]:
    """
    Retrieves a paginated list of all organizations.

    Required roles:
    - admin
    - member
    - developer
    """

    organizations = await crud.organization.get_multi_paginated(params=params)
    return create_response(data=organizations)


@router.get("/organization/{organization_id}")
async def get_organization_by_id(
    organization_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
) -> IGetResponseBase[IOrganizationRead]:
    """
    Retrieves an Organization by its unique ID.

    Required roles:
    - admin
    - member
    - developer
    """
    organization = await crud.organization_crud.organization.get_organization_by_id(
        organization_id=organization_id
    )
    if organization:
        return create_response(data=organization)
    else:
        raise IdNotFoundException(organization, organization_id)


@router.put("/organization/{organization_id}")
async def update_organization(
    organization_id: UUID,
    obj_in: IOrganizationUpdate,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
) -> IPutResponseBase[IOrganizationRead]:
    """
    Updates an Organization by its unique ID.

    Required roles:
    - admin
    - developer
    """
    updated_organization = await crud.organization.update_organization(
        organization_id=organization_id,
        obj_in=obj_in,
    )
    return create_response(data=updated_organization)


@router.delete("/organization/{organization_id}")
async def delete_organization(
    organization_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin])
    ),
) -> IDeleteResponseBase[IOrganizationRead]:
    """
    Deletes an Organization by its unique ID.

    Required roles:
    - must be an admin and superuser
    """
    if current_user.is_superuser is False:
        raise HTTPException(
            status_code=403, detail="You do not have permission to perform this action."
        )
    deleted_organization = await crud.organization.soft_delete_organization(
        organization_id=organization_id
    )
    return create_response(data=deleted_organization)
