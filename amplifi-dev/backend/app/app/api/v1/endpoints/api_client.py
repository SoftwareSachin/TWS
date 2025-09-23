from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi_pagination import Params
from sqlmodel.ext.asyncio.session import AsyncSession

from app import crud
from app.api import deps
from app.be_core.api_client_security import create_api_client_token
from app.be_core.logger import logger
from app.schemas.api_client_schema import (
    ApiClientCreate,
    ApiClientCreateResponse,
    ApiClientRead,
    ApiClientTokenRequest,
    ApiClientTokenResponse,
    ApiClientUpdate,
)
from app.schemas.response_schema import (
    IDeleteResponseBase,
    IGetResponseBase,
    IGetResponsePaginated,
    IPostResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData

router = APIRouter()


@router.post(
    "/api_client/token",
    response_model=IPostResponseBase[ApiClientTokenResponse],
)
async def get_api_client_token(
    request: ApiClientTokenRequest,
    db: AsyncSession = Depends(deps.get_db),
) -> IPostResponseBase[ApiClientTokenResponse]:
    """
    Get JWT token for API client authentication.
    This endpoint allows API clients to authenticate using client_id and client_secret.
    """
    try:
        # Authenticate the API client
        api_client = await crud.api_client.authenticate_client(
            client_id=request.client_id,
            client_secret=request.client_secret,
            db_session=db,
        )

        if not api_client:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid client credentials",
            )

        # Create JWT token
        access_token = create_api_client_token(
            client_id=api_client.client_id,
            organization_id=api_client.organization_id,
        )

        # Ignore - [B106: hardcoded_password_funcarg]
        # There is no hardcoded password 'bearer' in this code.
        token_response = ApiClientTokenResponse(
            access_token=access_token,
            token_type="bearer",  # nosec
            expires_in=60 * 60,  # 1 hour
            client_id=api_client.client_id,
        )

        logger.info(f"API client {api_client.client_id} authenticated successfully")

        return create_response(
            data=token_response,
            message="API client authenticated successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error authenticating API client: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication",
        )


@router.post(
    "/organization/{organization_id}/api_client",
    response_model=IPostResponseBase[ApiClientCreateResponse],
)
async def create_api_client(
    *,
    organization_id: UUID,
    obj_in: ApiClientCreate,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    db: AsyncSession = Depends(deps.get_db),
) -> IPostResponseBase[ApiClientCreateResponse]:
    """
    Create a new API client.

    Required roles:
    - admin
    - developer
    """
    try:
        # Verify user belongs to the organization
        if str(current_user.organization_id) != str(organization_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only create API clients for your own organization",
            )

        # Set the organization_id from the path parameter
        # obj_in.organization_id = organization_id

        # Create the API client
        api_client, client_secret = await crud.api_client.create(
            obj_in=obj_in,
            organization_id=organization_id,
            created_by=current_user.id,
            db_session=db,
        )

        # Build response with client_secret (only shown during creation)
        response_data = ApiClientCreateResponse(
            id=api_client.id,
            client_id=api_client.client_id,
            client_secret=client_secret,
            name=api_client.name,
            description=api_client.description,
            organization_id=api_client.organization_id,
            expires_at=api_client.expires_at,
            created_at=api_client.created_at,
        )

        logger.info(
            f"API client {api_client.client_id} created by user {current_user.id}"
        )

        return create_response(
            data=response_data,
            message="API client created successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating API client: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during API client creation",
        )


@router.get(
    "/organization/{organization_id}/api_client",
    response_model=IGetResponsePaginated[ApiClientRead],
)
async def get_api_clients(
    *,
    organization_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.organization_check),
    params: Params = Depends(),
    db: AsyncSession = Depends(deps.get_db),
) -> IGetResponsePaginated[ApiClientRead]:
    """
    Get all API clients for the current user's organization.

    Required roles:
    - admin
    - developer
    """
    try:
        # Verify user belongs to the organization
        if str(current_user.organization_id) != str(organization_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only access API clients from your own organization",
            )

        api_clients_page = await crud.api_client.get_by_organization(
            organization_id=organization_id,
            pagination_params=params,
            db_session=db,
        )

        # Convert to response format - api_clients_page is a Page object
        response_data = [
            ApiClientRead.model_validate(client) for client in api_clients_page.items
        ]

        logger.info(f"API clients: {response_data}")
        paginated_response = IGetResponsePaginated.create(
            items=response_data,
            total=api_clients_page.total,
            params=params,
        )
        logger.info(f"paginated_response: {paginated_response}")
        return create_response(data=paginated_response)

    except Exception as e:
        logger.error(f"Error retrieving API clients: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while retrieving API clients",
        )


@router.get(
    "/organization/{organization_id}/api_client/{api_client_id}",
    response_model=IGetResponseBase[ApiClientRead],
)
async def get_api_client(
    *,
    organization_id: UUID,
    api_client_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.organization_check),
    db: AsyncSession = Depends(deps.get_db),
) -> IGetResponseBase[ApiClientRead]:
    """
    Get a specific API client by ID.

    Required roles:
    - admin
    - developer
    """
    try:
        api_client = await crud.api_client.get(id=api_client_id, db_session=db)

        if not api_client:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API client not found",
            )

        # Verify the API client belongs to the specified organization
        if str(api_client.organization_id) != str(organization_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only access API clients from your own organization",
            )
        response_data = ApiClientRead(
            id=api_client.id,
            client_id=api_client.client_id,
            name=api_client.name,
            description=api_client.description,
            organization_id=api_client.organization_id,
            expires_at=api_client.expires_at,
            created_at=api_client.created_at,
            updated_at=api_client.updated_at,
            last_used_at=api_client.last_used_at,
        )

        return create_response(
            data=response_data, message="API client retrieved successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving API client {api_client_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while retrieving API client",
        )


@router.put(
    "/organization/{organization_id}/api_client/{api_client_id}",
    response_model=IPostResponseBase[ApiClientRead],
)
async def update_api_client(
    *,
    organization_id: UUID,
    api_client_id: UUID,
    obj_in: ApiClientUpdate,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.organization_check),
    db: AsyncSession = Depends(deps.get_db),
) -> IPostResponseBase[ApiClientRead]:
    """
    Update an API client.

    Required roles:
    - admin
    - developer
    """
    try:
        # Get the existing API client
        api_client = await crud.api_client.get(id=api_client_id, db_session=db)

        if not api_client:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API client not found",
            )

        # Verify user belongs to the same organization
        if str(api_client.organization_id) != str(organization_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only update API clients from your own organization",
            )

        # Verify the API client belongs to the specified organization
        if str(api_client.organization_id) != str(organization_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API client not found in the specified organization",
            )

        # Update the API client
        updated_client = await crud.api_client.update(
            db_obj=api_client,
            obj_in=obj_in,
            db_session=db,
        )

        response_data = ApiClientRead(
            id=updated_client.id,
            client_id=updated_client.client_id,
            name=updated_client.name,
            description=updated_client.description,
            organization_id=updated_client.organization_id,
            expires_at=updated_client.expires_at,
            created_at=updated_client.created_at,
            updated_at=updated_client.updated_at,
            last_used_at=updated_client.last_used_at,
        )

        logger.info(f"API client {api_client_id} updated by user {current_user.id}")

        return create_response(
            data=response_data,
            message="API client updated successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating API client {api_client_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while updating API client",
        )


@router.delete(
    "/organization/{organization_id}/api_client/{api_client_id}",
    response_model=IDeleteResponseBase,
)
async def delete_api_client(
    *,
    organization_id: UUID,
    api_client_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.organization_check),
    db: AsyncSession = Depends(deps.get_db),
) -> IDeleteResponseBase:
    """
    Delete an API client.

    Required roles:
    - admin
    - developer
    """
    try:
        # Get the existing API client
        api_client = await crud.api_client.get(id=api_client_id, db_session=db)

        if not api_client:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API client not found",
            )

        # Verify user belongs to the same organization
        if str(api_client.organization_id) != str(current_user.organization_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only delete API clients from your own organization",
            )

        # Delete the API client
        await crud.api_client.delete(id=api_client_id, db_session=db)

        logger.info(f"API client {api_client_id} deleted by user {current_user.id}")

        return IDeleteResponseBase(
            message="API client deleted successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting API client {api_client_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while deleting API client",
        )


@router.post(
    "/organization/{organization_id}/api_client/{api_client_id}/regenerate-secret",
    response_model=IPostResponseBase[dict],
)
async def regenerate_api_client_secret(
    *,
    organization_id: UUID,
    api_client_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.organization_check),
    db: AsyncSession = Depends(deps.get_db),
) -> IPostResponseBase[dict]:
    """
    Regenerate the client_secret for an API client.

    Required roles:
    - admin
    - developer
    """
    try:
        # Get the existing API client
        api_client = await crud.api_client.get(id=api_client_id, db_session=db)

        if not api_client:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API client not found",
            )

        # Verify user belongs to the same organization
        if str(api_client.organization_id) != str(current_user.organization_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only regenerate secrets for API clients from your own organization",
            )

        # Regenerate the client_secret
        new_secret = await crud.api_client.regenerate_secret(
            id=api_client_id, db_session=db
        )

        logger.info(
            f"API client {api_client_id} secret regenerated by user {current_user.id}"
        )

        return create_response(
            data={"client_secret": new_secret},
            message="API client secret regenerated successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error regenerating API client secret {api_client_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while regenerating API client secret",
        )
