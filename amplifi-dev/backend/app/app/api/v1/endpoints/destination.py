from typing import Any, List
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Response
from fastapi_pagination import Params
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.api import deps
from app.be_core.logger import logger
from app.models.destination_model import Destination
from app.models.workflow_model import Workflow
from app.schemas.connection_status_schema import (
    ConnectorStatus,
    IDestinationConnectorStatusRead,
)
from app.schemas.destination_schema import (
    IDestinationCreate,
    IDestinationRead,
)
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


@router.post(
    "/organization/{organization_id}/destination",
    response_model=IPostResponseBase[IDestinationRead],
)
async def create_destination(
    obj_in: IDestinationCreate,
    organization_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.organization_check),
    db: AsyncSession = Depends(deps.get_db),
) -> IPostResponseBase[IDestinationRead]:
    """
    Creates a new destination for the specified organization after validating the connection.

    Required roles:
    - admin
    - developer

    Note:
    - Supports both PgVector and Databricks types.
    """
    try:
        destination_type = _get_destination_type(obj_in)
        connection_details = _get_connection_details(obj_in, destination_type)

        await _validate_destination_connection(connection_details, destination_type)

        new_destination = await crud.destination_crud.create_destination(
            organization_id=organization_id,
            obj_in=obj_in,
            destination_type=destination_type,
            db_session=db,
        )

        if not new_destination:
            raise HTTPException(status_code=500, detail="Failed to create destination")

        return create_response(
            data=IDestinationRead.from_orm(new_destination),
            message="Destination created successfully",
        )

    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create destination")


def _get_destination_type(obj_in: IDestinationCreate) -> str:
    if obj_in.pg_vector:
        return "pg_vector"
    if obj_in.databricks:
        return "databricks"
    raise HTTPException(
        status_code=400,
        detail="Either pg_vector or databricks information must be provided.",
    )


def _get_connection_details(obj_in: IDestinationCreate, destination_type: str) -> Any:
    if destination_type == "pg_vector":
        pg = obj_in.pg_vector
        return f"postgresql://{pg.username}:{pg.password}@{pg.host}:{pg.port}/{pg.database_name}"
    if destination_type == "databricks":
        dbx = obj_in.databricks
        return {
            "workspace_url": dbx.workspace_url,
            "token": dbx.token,
            "warehouse_id": dbx.warehouse_id,
        }
    raise HTTPException(status_code=400, detail="Invalid destination type")


async def _validate_destination_connection(
    connection_details: Any, destination_type: str
):
    is_valid, message = await crud.destination_crud.check_destination_connection(
        connection_details, destination_type
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Failed to connect: {message}")


@router.get(
    "/organization/{organization_id}/destination",
    response_model=IGetResponsePaginated[IDestinationRead],
)
async def get_destinations(
    organization_id: UUID,
    params: Params = Depends(),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.organization_check),
    db: AsyncSession = Depends(deps.get_db),
) -> IGetResponsePaginated[IDestinationRead]:
    """
    Retrieves all destinations for the specified organization.

    Required roles:
    - admin
    - member
    - developer

    Note:
    - Supports paginated results ordered by created_at.
    """
    destinations: List[Destination] = await crud.destination_crud.get_destinations(
        organization_id=organization_id, params=params, db_session=db
    )

    # Use list comprehension to create Pydantic models
    pydantic_destinations = [
        IDestinationRead.from_orm(destination) for destination in destinations
    ]

    total_destinations = len(pydantic_destinations)
    paginated_response = IGetResponsePaginated.create(
        items=pydantic_destinations, total=total_destinations, params=params
    )
    return create_response(data=paginated_response)


@router.get(
    "/organization/{organization_id}/destination/{destination_id}",
    response_model=IGetResponseBase[IDestinationRead],
)
async def get_destination_by_id(
    destination_id: UUID,
    organization_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.organization_check),
    db: AsyncSession = Depends(deps.get_db),
) -> IGetResponseBase[IDestinationRead]:
    """
    Retrieves details of a specific destination by ID.

    Required roles:
    - admin
    - member
    - developer
    """
    # Retrieve the basic destination info
    destination = await crud.destination_crud.get_destination_by_id(
        destination_id=destination_id, organization_id=organization_id, db_session=db
    )

    if not destination:
        raise IdNotFoundException(Destination, destination_id)

    # Prepare detailed information based on destination_type
    destination_data = IDestinationRead.from_orm(destination)

    if destination.destination_type == "pg_vector":
        destination_data.pg_vector = await crud.destination_crud._get_pg_vector_details(
            destination_id=destination_id, db_session=db
        )
    elif destination.destination_type == "databricks":
        destination_data.databricks = (
            await crud.destination_crud._get_databricks_details(
                destination_id=destination_id, db_session=db
            )
        )

    # Count active workflows associated with this destination
    active_workflows_count = await db.execute(
        select(func.count()).where(
            Workflow.destination_id == destination_id,
            Workflow.is_active == True,  # noqa: E712
            Workflow.deleted_at.is_(None),
        )
    )
    destination_data.active_workflows = active_workflows_count.scalar()
    return create_response(data=destination_data)


@router.delete(
    "/organization/{organization_id}/destination/{destination_id}",
    status_code=204,
)
async def delete_destination(
    destination_id: UUID,
    organization_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.organization_check),
    db: AsyncSession = Depends(deps.get_db),
) -> Response:
    """
    Deletes a destination by its ID.

    Required roles:
    - admin
    - developer
    """
    try:
        deleted_destination = await crud.destination_crud.delete_destination(
            destination_id=destination_id,
            organization_id=organization_id,
            db_session=db,
        )

        if not deleted_destination:
            logger.warning(f"Destination {destination_id} not found.")
            raise HTTPException(status_code=404, detail="Destination not found")

        logger.info(f"Destination {destination_id} successfully deleted.")
        return Response(status_code=204)

    except HTTPException as exc:
        logger.error(f"HTTPException occurred: {exc.detail}")
        raise exc
    except Exception as e:
        logger.error(
            f"Unexpected error while deleting destination: {str(e)}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get(
    "/organization/{organization_id}/destination/{destination_id}/connection_status",
    response_model=IGetResponseBase[IDestinationConnectorStatusRead],
)
async def get_destination_connection_status(
    organization_id: UUID = Path(..., description="The UUID of the organization"),
    destination_id: UUID = Path(
        ..., description="The UUID of the destination connector"
    ),
    db_session: AsyncSession = Depends(deps.get_db),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
):
    """
    Get the connection status of a specific destination connector in a organization.

    Returns:
    - Connection status of the destinationn connector, including whether the connection
    is successful and an appropriate message.

    Required roles:
    - admin
    - member
    - developer
    """
    try:
        destination = await crud.destination_crud.get_destination_info(
            organization_id=organization_id,
            destination_id=destination_id,
            db_session=db_session,
        )
        if not destination:
            raise HTTPException(status_code=404, detail="Destination not found")

        if destination.destination_type == "pg_vector":
            connection_details = await crud.destination_crud._get_pg_vector_details(
                destination_id=destination_id, db_session=db_session
            )

            encoded_password = quote(connection_details["password"])

            db_url = (
                f"postgresql://{connection_details['username']}:{encoded_password}@"
                f"{connection_details['host']}:{connection_details['port']}/"
                f"{connection_details['database_name']}?sslmode=require"
            )

            is_connected, message = (
                await crud.destination_crud.check_destination_connection(
                    db_url, "pg_vector"
                )
            )
        elif destination.destination_type == "databricks":
            connection_details = await crud.destination_crud._get_databricks_details(
                destination_id=destination_id, db_session=db_session
            )

            is_connected, message = (
                await crud.destination_crud.check_destination_connection(
                    connection_details, "databricks"
                )
            )
        else:
            raise HTTPException(
                status_code=501,
                detail="Unsupported or missing destination type details",
            )

        status = ConnectorStatus.success if is_connected else ConnectorStatus.failed
        connection_status_data = IDestinationConnectorStatusRead(
            status=status, message=message
        )
        return IGetResponseBase(
            message="Data got correctly", meta={}, data=connection_status_data
        )
    except HTTPException as e:
        logger.error(f"HTTPException occurred: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )
