import base64
import os
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Dict, Optional, Union
from urllib.parse import unquote
from uuid import UUID

import hvac  # type: ignore
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, UploadFile
from fastapi_pagination import Params
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.be_core.config import settings
from app.be_core.logger import logger
from app.crud.log_crud import log_crud
from app.crud.source_connector_crud import CRUDSource
from app.models.audit_log_model import EntityType, OperationType
from app.models.azure_storage_model import AzureStorage
from app.models.file_model import File
from app.models.groove_source_model import GrooveSource
from app.models.pull_status_model import PullStatusEnum, SourcePullStatus
from app.models.source_model import Source
from app.schemas.common_schema import IOrderEnum
from app.schemas.connection_status_schema import (
    ConnectorStatus,
    ISourceConnectorStatusRead,
)
from app.schemas.file_metadata_schema import FileMetadataRead, PGTableMetadataRead
from app.schemas.file_schema import FileStatusEnum, IFileUploadFailed, IFileUploadRead
from app.schemas.response_schema import (
    IGetResponseBase,
    IGetResponsePaginated,
    IPostResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.source_schema import (
    IAutoDetectionStatus,
    IAWSS3Source,
    IAzureStorageSource,
    IFileStatistics,
    IGrooveSource,
    IMonitoringHealth,
    ISourceConnectorResponse,
    ISourceResponse,
    ISQLSource,
    IStorageAccountInfo,
    MonitoringHealthStatusEnum,
)
from app.schemas.user_schema import UserData
from app.utils.feature_flags import FeatureFlags, is_feature_enabled_for_workspace
from app.utils.optional_params import OptionalParams

vault_client = hvac.Client(url=settings.VAULT_ADDR, token=settings.VAULT_TOKEN)
DEFAULT_UPLOAD_FOLDER = settings.DEFAULT_UPLOAD_FOLDER
router = APIRouter()
crud = CRUDSource(Source)


@router.post("/workspace/{workspace_id}/source", response_model=ISourceResponse)
async def create_source(
    workspace_id: UUID,
    source: IAzureStorageSource | IAWSS3Source | ISQLSource | IGrooveSource,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.user_workspace_access_check),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Creates a source connector for a workspace.

    - First, it checks the connection.
    - If successful, it creates the source.
    - If creation is successful, it triggers pull_files.
    """
    try:
        logger.info(f"Received source_type: {source.source_type}")

        # Feature gate: groove_source availability via Flagsmith
        if source.source_type == "groove_source":
            enabled = await is_feature_enabled_for_workspace(
                feature_name=FeatureFlags.GROOVE_CONNECTOR_FEATURE,
            )
            if not enabled:
                raise HTTPException(
                    status_code=403,
                    detail="groove_source is not enabled for this organization.",
                )

        connection_checkers = {
            "azure_storage": lambda: crud.check_azure_connection(
                container_name=source.container_name,
                sas_url=base64.b64decode(source.sas_url).decode("utf-8"),
                # sas_url=try_base64_decode(source.sas_url)
            ),
            "aws_s3": lambda: crud.check_s3_connection(
                bucket_name=source.bucket_name,
                access_id=base64.b64decode(source.access_id).decode("utf-8"),
                access_secret=base64.b64decode(source.access_secret).decode("utf-8"),
            ),
            "azure_fabric": lambda: crud.check_azure_fabric_connection(
                container_name=source.container_name,
                sas_url=base64.b64decode(source.sas_url).decode("utf-8"),
            ),
            "pg_db": lambda: crud.check_sql_db_connection(
                "pg_db",
                host=source.host,
                port=source.port,
                database_name=source.database_name,
                username=source.username,
                password=source.password,
            ),
            "mysql_db": lambda: crud.check_sql_db_connection(
                "mysql_db",
                host=source.host,
                port=source.port,
                database_name=source.database_name,
                username=source.username,
                password=source.password,
                ssl_mode=source.ssl_mode,
            ),
            "groove_source": lambda: crud.check_groove_connection(
                api_key=base64.b64decode(source.groove_api_key).decode("utf-8")
            ),
        }

        if source.source_type not in connection_checkers:
            raise HTTPException(status_code=400, detail="Unsupported source type.")

        success, message = await connection_checkers[source.source_type]()
        if not success:
            logger.error(f"Connection Failed: {message}")
            raise HTTPException(status_code=400, detail=f"Connection Failed: {message}")

        # Mapping of source types to source creation functions
        source_creators = {
            "azure_storage": crud.create_with_azure_storage,
            "aws_s3": crud.create_with_s3_storage,
            "azure_fabric": crud.create_with_azure_fabric,
            "pg_db": crud.create_with_sql_db,
            "mysql_db": crud.create_with_sql_db,
            "groove_source": crud.create_with_groove_source,
        }

        source_db = await source_creators[source.source_type](
            workspace_id=workspace_id, obj_in=source, db_session=db
        )

        pull_status_entry = SourcePullStatus(
            source_id=source_db.id,
            pull_status=PullStatusEnum.NOT_STARTED,
        )
        db.add(pull_status_entry)

        await db.commit()
        await db.refresh(source_db)

        # Call pull_files
        await crud.pull_files(
            workspace_id=workspace_id,
            user_id=current_user.id,
            source_id=source_db.id,
            db_session=db,
        )

        await log_crud.log_operation(
            operation=OperationType.Create,
            entity=EntityType.Source_Connector,
            entity_id=source_db.id,
            entity_name=source_db.source_type,
            user_id=current_user.id,
            user_name=current_user.email,
        )

        return ISourceResponse(
            message="Source created successfully",
            meta={},
            data={"id": str(source_db.id)},
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in create_source: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/workspace/{workspace_id}/source/{source_id}", response_model=ISourceResponse
)
async def get_source(
    workspace_id: UUID,
    source_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.user_workspace_access_check),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Retrieves details of a specific source connector.
    Roles: admin, member, developer
    """
    try:
        source_db = await crud.get_by_id(
            workspace_id=workspace_id, source_id=source_id, db_session=db
        )
        if not source_db:
            raise HTTPException(status_code=404, detail="Source not found.")

        source_type = source_db.source_type

        if source_type == "azure_storage":
            await crud.get_azure_storage_details(source_id, db_session=db)
            return ISourceResponse(
                message="Data got correctly",
                meta={},
                data={
                    "source_type": source_type,
                    # "container_name": azure_storage["container_name"],
                    # "sas_url": azure_storage["sas_url"],
                },
            )

        elif source_type == "azure_fabric":
            azure_fabric = await crud.get_azure_fabric_details(source_id, db_session=db)
            if not azure_fabric:
                raise HTTPException(
                    status_code=404, detail="Azure Fabric details not found."
                )
            return ISourceResponse(
                message="Azure Fabric details fetched successfully.",
                meta={},
                data={
                    "source_type": source_type,
                    # "container_name": azure_fabric["container_name"],
                    # "sas_url": azure_fabric["sas_url"],
                },
            )

        elif source_type == "aws_s3":
            s3_storage_details = await crud.get_s3_storage_details(
                source_id, db_session=db
            )
            return ISourceResponse(
                message="Source details fetched successfully",
                meta={},
                data={
                    "source_type": source_type,
                    "bucket_name": s3_storage_details["bucket_name"],
                    # "access_key": s3_storage_details["access_key"],
                    # "secret_key": s3_storage_details["secret_key"],
                },
            )

        elif source_type in ["pg_db", "mysql_db"]:
            sql_details = await crud.get_sql_db_details(
                source_id, db_type=source_type, db_session=db
            )
            if not sql_details:
                raise HTTPException(
                    status_code=404, detail=f"{source_type.upper()} details not found."
                )
            return ISourceResponse(
                message=f"{source_type.upper()} details fetched successfully",
                meta={},
                data={
                    "source_type": source_type,
                    "host": sql_details["host"],
                    "port": sql_details["port"],
                    "database_name": sql_details["database_name"],
                    # "username": sql_details["username"],
                    # "password": sql_details["password"],
                },
            )

        elif source_type == "groove_source":
            groove_details = await crud.get_groove_details(source_id, db_session=db)
            if not groove_details:
                raise HTTPException(
                    status_code=404, detail="Groove source details not found."
                )
            return ISourceResponse(
                message="Groove source details fetched successfully.",
                meta={},
                data={
                    "source_type": source_type,
                    # "source_name": source_db.source_name,  # Include the source_name from the database
                    "api_key": groove_details["api_key"],
                },
            )

        else:
            raise HTTPException(status_code=400, detail="Unsupported source type.")

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error(f"Unexpected error in get_source: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/workspace/{workspace_id}/source/{source_id}", response_model=ISourceResponse
)
async def update_source(
    workspace_id: UUID,
    source_id: UUID,
    source: IAzureStorageSource | IAWSS3Source | ISQLSource | IGrooveSource,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.user_workspace_access_check),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Updates a specific source connector for a workspace.

    1. Checks the connection for the given source_type.
    2. Updates the source in DB.
    3. Clears old files related to that source.
    4. Triggers the pull_files logic.
    """
    try:
        logger.info(f"Updating source_id={source_id} of type={source.source_type}")

        # Feature gate: groove_source availability via Flagsmith
        if source.source_type == "groove_source":
            enabled = await is_feature_enabled_for_workspace(
                feature_name=FeatureFlags.GROOVE_CONNECTOR_FEATURE,
            )
            if not enabled:
                raise HTTPException(
                    status_code=403,
                    detail="groove_source is not enabled for this organization.",
                )

        connection_checkers = {
            "azure_storage": lambda: crud.check_azure_connection(
                container_name=source.container_name,
                sas_url=base64.b64decode(source.sas_url).decode("utf-8"),
                # sas_url=try_base64_decode(source.sas_url),
            ),
            "aws_s3": lambda: crud.check_s3_connection(
                bucket_name=source.bucket_name,
                access_id=base64.b64decode(source.access_id).decode("utf-8"),
                access_secret=base64.b64decode(source.access_secret).decode("utf-8"),
            ),
            "azure_fabric": lambda: crud.check_azucheck_sql_db_connectionre_fabric_connection(
                container_name=source.container_name,
                sas_url=base64.b64decode(source.sas_url).decode("utf-8"),
            ),
            "pg_db": lambda: crud.check_sql_db_connection(
                "pg_db",
                host=source.host,
                port=source.port,
                database_name=source.database_name,
                username=source.username,
                password=source.password,
            ),
            "mysql_db": lambda: crud.check_sql_db_connection(
                "mysql_db",
                host=source.host,
                port=source.port,
                database_name=source.database_name,
                username=source.username,
                password=source.password,
                ssl_mode=source.ssl_mode,
            ),
            "groove_source": lambda: crud.check_groove_connection(
                api_key=base64.b64decode(source.groove_api_key).decode("utf-8")
            ),
        }

        if source.source_type not in connection_checkers:
            raise HTTPException(status_code=400, detail="Unsupported source type.")

        success, message = await connection_checkers[source.source_type]()
        if not success:
            logger.error(f"Connection failed: {message}")
            raise HTTPException(status_code=400, detail=f"Connection failed: {message}")

        source_db = await crud.get_by_id(
            workspace_id=workspace_id, source_id=source_id, db_session=db
        )
        if not source_db:
            logger.error(
                f"Source with id={source_id} not found in workspace={workspace_id}"
            )
            raise HTTPException(
                status_code=400, detail=f"Source with id {source_id} not found."
            )

        updater_functions = {
            "azure_storage": crud.update_source_with_azure,
            "aws_s3": crud.update_source_with_s3,
            "azure_fabric": crud.update_source_with_azure_fabric,
            "pg_db": partial(crud.update_sql_db, source_type="pg_db"),
            "mysql_db": partial(crud.update_sql_db, source_type="mysql_db"),
            "groove_source": crud.update_source_with_groove,
        }

        await updater_functions[source.source_type](
            workspace_id=workspace_id,
            source_id=source_id,
            obj_in=source,
            db_session=db,
        )

        await crud.clear_old_files(
            workspace_id=workspace_id, source_id=source_id, db_session=db
        )

        await crud.pull_files(
            workspace_id=workspace_id,
            user_id=current_user.id,
            source_id=source_id,
            db_session=db,
        )

        await log_crud.log_operation(
            operation=OperationType.Update,
            entity=EntityType.Source_Connector,
            entity_id=source_id,
            entity_name=source.source_type,
            user_id=current_user.id,
            user_name=current_user.email,
        )

        return ISourceResponse(
            message="Source updated successfully",
            meta={},
            data={"id": str(source_db.id)},
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in update_source: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Connection Failed: {str(e)}")


MAX_FILE_SIZE = 300 * 1024 * 1024


@router.post(
    "/workspace/{workspace_id}/file_upload",
    response_model=IPostResponseBase[list[IFileUploadRead | IFileUploadFailed]],
)
async def upload_file(
    workspace_id: UUID,
    files: list[UploadFile] = [],  # Accept file input
    # target_path: str = None,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.user_workspace_access_check),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Uploads a file to the specified workspace.

    Required roles:
    - admin
    - developer

    Note:
    - Max file size is MAX_FILE_SIZE_UPLOADED_FILES (default 50mb).
    - Allowed types in :ALLOWED_FILE_EXTENSIONS, ALLOWED_MIME_TYPES
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Check the file size
    if sum([file.size for file in files]) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File sizes exceed the maximum limit. (Note: sum of file sizes must be under {MAX_FILE_SIZE})",
        )

    # upload_folder = target_path if target_path else DEFAULT_UPLOAD_FOLDER
    upload_folder = DEFAULT_UPLOAD_FOLDER
    print("Upload Folder:", upload_folder)

    if not upload_folder:
        logger.error("Upload folder not configured", exc_info=True)
        raise HTTPException(status_code=500, detail="Upload folder not configured.")

    os.makedirs(upload_folder, exist_ok=True)
    file_upload_responses: list[IFileUploadRead] = []
    for file in files:
        try:
            # Call the CRUD function to handle the file upload
            file_db = await crud.upload_file(
                workspace_id=workspace_id,
                file=file,
                db_session=db,
                target_path=upload_folder,
            )
            await log_crud.log_operation(
                operation=OperationType.Create,
                entity=EntityType.File,
                entity_id=file_db.id,
                entity_name=file_db.filename,
                user_id=current_user.id,
                user_name=current_user.email,
            )
            file_upload_responses.append(
                IFileUploadRead(
                    filename=file_db.filename,
                    mimetype=file_db.mimetype,
                    size=file_db.size,
                    id=file_db.id,
                    status=file_db.status,
                )
            )
        except Exception as e:
            logger.warning(f"Error During File Upload: {str(e)}")
            file_upload_responses.append(
                IFileUploadFailed(
                    filename=file.filename,
                    mimetype=file.content_type or "",
                    size=file.size,
                    error=str(e),
                )
            )
    return create_response(data=file_upload_responses, message="Files processed.")


@router.get(
    "/workspace/{workspace_id}/source",
    response_model=IGetResponsePaginated[ISourceConnectorResponse],
)
async def get_sources(
    workspace_id: UUID,
    order: IOrderEnum = Query(
        default=IOrderEnum.ascendent, description="Optional. Default is ascendent"
    ),
    params: Params = Depends(),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.user_workspace_access_check),
) -> IGetResponsePaginated[ISourceConnectorResponse]:
    """
    Retrieves all source connectors for a workspace, with pagination support.
    Required roles: admin, member, developer.
    """
    sources = await crud.get_sources_paginated_ordered(
        workspace_id=workspace_id, params=params, order=order, order_by="created_at"
    )

    pydantic_sources = []
    for source in sources.items:
        processed_source = await crud.process_source(source)
        if processed_source:
            pydantic_sources.append(processed_source)

    return create_response(
        data=IGetResponsePaginated.create(
            items=pydantic_sources, total=sources.total, params=params
        )
    )


@router.get(
    "/workspace/{workspace_id}/file",
    response_model=IGetResponsePaginated[IFileUploadRead]
    | IGetResponseBase[list[IFileUploadRead]],
)
async def get_files(
    workspace_id: UUID,
    params: OptionalParams = Depends(),
    order: IOrderEnum = Query(IOrderEnum.ascendent),
    only_uploaded: bool = Query(
        True,
        description="If true, only show uploaded files (source_id is null). If false, show all files.",
    ),
    search: Optional[str] = Query(None, description="Search by filename"),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.user_workspace_access_check),
):
    """
    Retrieves files for a workspace.
    Query Parameters:
    - only_uploaded=True: Only files uploaded directly (source_id is null)
    - only_uploaded=False: All files (both uploaded and source files)
    - search: Search by filename (optional)
    - default it is true
    Required roles:
    - admin
    - member
    - developer

    Note:
    - Supports paginated results ordered by created_at.
    - If `page` and `size` are passed, paginated response is returned.
    - If not passed, all files are returned (infinite scroll behavior).
    - Search functionality filters files by filename (case-insensitive).
    """
    files_result = await crud.get_files_by_workspace(
        workspace_id=workspace_id,
        params=params,
        order=order,
        only_uploaded=only_uploaded,
        search=search,
    )

    if params.page is None and params.size is None and isinstance(files_result, list):
        pydantic_files = [
            IFileUploadRead(
                filename=file.filename,
                mimetype=file.mimetype,
                size=file.size,
                status=file.status,
                id=file.id,
            )
            for file in files_result
        ]
        return create_response(
            data=pydantic_files, message="Files retrieved without pagination"
        )

    # Paginated
    pydantic_files = [
        IFileUploadRead(
            filename=file.filename,
            mimetype=file.mimetype,
            size=file.size,
            status=file.status,
            id=file.id,
        )
        for file in files_result.items
    ]
    return create_response(
        data=IGetResponsePaginated.create(
            items=pydantic_files,
            total=files_result.total,
            params=params,
        ),
        message="Files retrieved with pagination",
    )


@router.get(
    "/workspace/{workspace_id}/file/{file_id}",
    response_model=IFileUploadRead,
)
async def get_file_by_id_in_workspace(
    workspace_id: UUID,
    file_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.user_workspace_access_check),
):
    """
    Get a single file by ID in a workspace.
    """
    try:
        file = await crud.get_file_by_id_in_workspace(
            file_id=file_id, workspace_id=workspace_id
        )

        if not file:
            raise HTTPException(
                status_code=404, detail="File not found in this workspace."
            )

        return IFileUploadRead(
            filename=deps.get_filename(file.filename),
            mimetype=file.mimetype,
            size=file.size,
            status=file.status,
            id=file.id,
        )
    except HTTPException as exc:
        raise exc
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get(
    "/workspace/{workspace_id}/source/{source_id}/files_metadata",
    response_model=IGetResponsePaginated[Union[PGTableMetadataRead, FileMetadataRead]],
)
async def get_file_metadata_by_source_id(
    workspace_id: UUID = Path(..., description="The UUID id of the workspace"),
    source_id: UUID = Path(..., description="The UUID id of the source connector"),
    params: Params = Depends(),
    order_by: str | None = Query(
        default="id", description="Field to order by (default is id)"
    ),
    order: IOrderEnum | None = Query(
        default=IOrderEnum.ascendent,
        description="Order direction (ascendent or descendent)",
    ),
    search: Optional[str] = Query(None, description="Search by filename"),
    db_session: AsyncSession = Depends(deps.get_db),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.user_workspace_access_check),
):
    """
    Retrieves metadata for files in a source connector.

    Query Parameters:
    - search: Search by filename (optional, case-insensitive)
    Required roles:
    - admin
    - member
    - developer

    Note:
    - Supports paginated results.
    - Search functionality filters files by filename (case-insensitive).
    """

    source = await crud.get_source_info(workspace_id, source_id, db_session=db_session)
    file_metadata = await crud.get_file_metadata_by_source_id(
        source_id=source_id,
        params=params,
        search=search,
        order_by=order_by,
        order=order,
        db_session=db_session,
    )

    pydantic_metadata = []
    for metadata in file_metadata.items:
        updated_filename = metadata.filename
        if source.source_type == "pg_db":
            pydantic_metadata.append(
                PGTableMetadataRead(
                    id=metadata.id,
                    filename=updated_filename,
                    mimetype=metadata.mimetype,
                    size=metadata.size,
                    status=metadata.status,
                    rows=metadata.rows or 0,
                    columns=metadata.columns or 0,
                )
            )
        else:
            data = FileMetadataRead.from_orm(metadata)
            data.filename = updated_filename
            pydantic_metadata.append(data)

    return create_response(
        data=IGetResponsePaginated.create(
            items=pydantic_metadata, total=file_metadata.total, params=params
        )
    )


@router.get(
    "/workspace/{workspace_id}/source/{source_id}/connection_status",
    response_model=IGetResponseBase[ISourceConnectorStatusRead],
)
async def get_connection_status(
    workspace_id: UUID = Path(..., description="The UUID id of the workspace"),
    source_id: UUID = Path(..., description="The UUID id of the source connector"),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.user_workspace_access_check),
    db_session: AsyncSession = Depends(deps.get_db),
):
    """Get the connection status of a specific source connector in a workspace."""
    try:
        source = await _get_source_or_404(workspace_id, source_id, db_session)
        processed_source = await _process_source_or_501(source)

        checker = _get_connection_checker(source.source_type)
        args = await _get_connection_args(
            source, processed_source, source_id, db_session
        )

        is_connected, message = await checker(**args)

        return IGetResponseBase(
            data=ISourceConnectorStatusRead(
                status=(
                    ConnectorStatus.success if is_connected else ConnectorStatus.failed
                ),
                message=message,
            )
        )

    except HTTPException as e:
        logger.error(f"HTTPException occurred: {str(e)}")
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error in get_connection_status: {str(e)}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


async def _get_source_or_404(workspace_id, source_id, db_session):
    source = await crud.get_source_info(workspace_id, source_id, db_session)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    return source


async def _process_source_or_501(source):
    processed = await crud.process_source(source)
    if not processed:
        raise HTTPException(
            status_code=501, detail="Unsupported or missing source type details"
        )
    return processed


def _get_connection_checker(source_type: str):
    checkers = {
        "azure_storage": crud.check_azure_connection,
        "aws_s3": crud.check_s3_connection,
        "azure_fabric": crud.check_azure_fabric_connection,
        "pg_db": lambda **kwargs: crud.check_sql_db_connection("pg_db", **kwargs),
        "mysql_db": lambda **kwargs: crud.check_sql_db_connection("mysql_db", **kwargs),
        "groove_source": crud.check_groove_connection,
    }
    if source_type not in checkers:
        raise HTTPException(status_code=501, detail="Unsupported source type")
    return checkers[source_type]


async def _get_connection_args(source, processed_source, source_id, db_session):
    if source.source_type == "aws_s3":
        credentials = await crud.get_s3_storage_details(source_id, db_session)
        if not credentials:
            raise HTTPException(
                status_code=500, detail="Failed to retrieve AWS S3 credentials."
            )
        return {
            "bucket_name": credentials["bucket_name"],
            "access_id": credentials["access_key"],
            "access_secret": credentials["secret_key"],
        }

    elif source.source_type in ["pg_db", "mysql_db"]:
        credentials = await crud.get_sql_db_details(
            source_id=source_id, db_type=source.source_type, db_session=db_session
        )

        if not credentials:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve credentials for {source.source_type}.",
            )

        return {
            "host": credentials["host"],
            "port": credentials["port"],
            "database_name": credentials["database_name"],
            "username": credentials["username"],
            "password": credentials["password"],
        }

    elif source.source_type == "groove_source":
        credentials = await crud.get_groove_details(source_id, db_session)
        if not credentials:
            raise HTTPException(
                status_code=500, detail="Failed to retrieve Groove credentials."
            )
        return {"api_key": credentials["api_key"]}

    elif source.source_type in ["azure_storage", "azure_fabric"]:
        # For azure_storage and azure_fabric
        sas_url_ref = (
            unquote(source.azure_storage.sas_url).replace("vault:", "").strip()
        )
        sas_url = await crud.fetch_secret_value(sas_url_ref, "sas_url")
        return {
            "container_name": processed_source.sources.container_name,
            "sas_url": sas_url,
        }

    else:
        raise HTTPException(
            status_code=501, detail=f"Unsupported source type: {source.source_type}"
        )


@router.delete(
    "/workspace/{workspace_id}/source/{source_id}",
    status_code=204,
    response_class=Response,
)
async def delete_source(
    source_id: UUID,
    workspace_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    db: AsyncSession = Depends(deps.get_db),
    _=Depends(deps.user_workspace_access_check),
) -> Response:
    """
    Deletes a source by its ID based on source_type.
    """
    try:
        deleted_source = await crud.delete_source(
            source_id=source_id,
            workspace_id=workspace_id,
            db_session=db,
        )

        if not deleted_source:
            logger.warning(f"Source {source_id} not found.")
            raise HTTPException(status_code=404, detail="Source not found")

        logger.info(f"Source {source_id} successfully deleted.")
        return Response(status_code=204)

    except HTTPException as exc:
        logger.error(f"HTTPException occurred: {exc.detail}")
        raise exc
    except Exception as e:
        logger.error(f"Unexpected error while deleting source: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.delete(
    "/workspace/{workspace_id}/file/{file_id}",
    status_code=204,
    response_class=Response,
)
async def delete_file(
    file_id: UUID,
    workspace_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    db: AsyncSession = Depends(deps.get_db),
    _=Depends(deps.user_workspace_access_check),
) -> Response:
    """
    Soft deletes a file by setting its `deleted_at` timestamp.
    """
    try:
        deleted = await crud.delete_file(
            file_id=file_id,
            workspace_id=workspace_id,
            db_session=db,
        )

        if not deleted:
            logger.warning(f"File {file_id} not found or already deleted.")
            raise HTTPException(status_code=404, detail="File not found")

        logger.info(f"File {file_id} successfully soft deleted.")
        return Response(status_code=204)

    except HTTPException as exc:
        logger.error(f"HTTPException occurred: {exc.detail}")
        raise exc
    except Exception as e:
        logger.error(
            f"Unexpected error while soft deleting file: {str(e)}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal Server Error")


## combining the azure and groove source auto-detection configure funcitons
@router.patch(
    "/workspace/{workspace_id}/source/{source_id}/auto-detection",
    response_model=ISourceResponse,
)
async def configure_auto_detection(
    workspace_id: UUID,
    source_id: UUID,
    auto_detection_config: dict,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.user_workspace_access_check),
    db: AsyncSession = Depends(deps.get_db),
):
    """Configure auto-detection for sources using polling mechanism."""

    async def _configure_azure_auto_detection():
        """Configure auto-detection for Azure Storage sources."""
        azure_storage = source.azure_storage

        if auto_detection_config.get("enabled"):
            # Enable auto-detection
            try:
                azure_storage.auto_detection_enabled = True
                azure_storage.monitoring_frequency_minutes = auto_detection_config.get(
                    "frequency_minutes", 30
                )
                azure_storage.re_ingest_updated_blobs = auto_detection_config.get(
                    "re_ingest_updated_blobs", False
                )
                await db.commit()
                await db.refresh(source)
                logger.info(
                    f"Auto-detection enabled for Azure source {source_id} with "
                    f"{azure_storage.monitoring_frequency_minutes} minute polling and "
                    f"re-ingestion {'enabled' if azure_storage.re_ingest_updated_blobs else 'disabled'}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to enable auto-detection for Azure source {source_id}: {str(e)}"
                )
                raise HTTPException(
                    status_code=500, detail=f"Failed to enable auto-detection: {str(e)}"
                )
        else:
            # Disable auto-detection
            try:
                azure_storage.auto_detection_enabled = False
                azure_storage.webhook_url = None
                await db.commit()
                await db.refresh(source)
                logger.info(f"Auto-detection disabled for Azure source {source_id}")
            except Exception as e:
                logger.error(
                    f"Failed to disable auto-detection for Azure source {source_id}: {str(e)}",
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to disable auto-detection: {str(e)}",
                )

    async def _configure_groove_auto_detection():
        """Configure auto-detection for Groove sources."""
        groove_source = source.groove_source

        if auto_detection_config.get("enabled"):
            # Enable auto-detection
            try:
                groove_source.auto_detection_enabled = True
                groove_source.monitoring_frequency_minutes = auto_detection_config.get(
                    "frequency_minutes", 30
                )
                groove_source.ticket_batch_size = auto_detection_config.get(
                    "ticket_batch_size", 10
                )
                groove_source.re_ingest_updated_tickets = auto_detection_config.get(
                    "re_ingest_updated_tickets", False
                )
                await db.commit()
                await db.refresh(source)
                logger.info(
                    f"Auto-detection enabled for Groove source {source_id} with "
                    f"{groove_source.monitoring_frequency_minutes} minute polling"
                )
            except Exception as e:
                logger.error(
                    f"Failed to enable auto-detection for Groove source {source_id}: {str(e)}"
                )
                raise HTTPException(
                    status_code=500, detail=f"Failed to enable auto-detection: {str(e)}"
                )
        else:
            # Disable auto-detection
            try:
                groove_source.auto_detection_enabled = False
                await db.commit()
                await db.refresh(source)
                logger.info(f"Auto-detection disabled for Groove source {source_id}")
            except Exception as e:
                logger.error(
                    f"Failed to disable auto-detection for Groove source {source_id}: {str(e)}",
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to disable auto-detection: {str(e)}",
                )

    # Main function logic
    source = await crud.get_source_info(workspace_id, source_id, db)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    # Handle different source types
    if source.source_type == "azure_storage":
        await _configure_azure_auto_detection()
    elif source.source_type == "groove_source":
        await _configure_groove_auto_detection()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Auto-detection not supported for source type: {source.source_type}",
        )

    return ISourceResponse(
        message="Auto-detection configuration updated successfully",
        meta={},
        data={"id": str(source_id)},
    )


@router.get(
    "/workspace/{workspace_id}/source/{source_id}/auto-detection/status",
    response_model=IAutoDetectionStatus,
)
async def get_auto_detection_status(
    workspace_id: UUID,
    source_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.user_workspace_access_check),
    db: AsyncSession = Depends(deps.get_db),
):
    """Get auto-detection status for Azure Storage and Groove sources using polling mechanism."""

    source = await crud.get_source_info(workspace_id, source_id, db)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    if source.source_type == "azure_storage":
        return await _get_azure_auto_detection_status(source, source_id, db)
    elif source.source_type == "groove_source":
        return await _get_groove_auto_detection_status(source, source_id, db)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Auto-detection status not supported for source type: {source.source_type}",
        )


async def _get_azure_auto_detection_status(
    source, source_id: UUID, db: AsyncSession
) -> IAutoDetectionStatus:
    """Get auto-detection status for Azure Storage sources."""
    azure_storage = source.azure_storage

    # Calculate next monitoring time if auto-detection is enabled
    next_monitoring_time = None
    time_until_next_monitoring_minutes = None

    if azure_storage.auto_detection_enabled and azure_storage.last_monitored:
        next_monitoring = azure_storage.last_monitored + timedelta(
            minutes=azure_storage.monitoring_frequency_minutes
        )
        next_monitoring_time = next_monitoring.isoformat()
        time_until_next_monitoring_minutes = max(
            0, (next_monitoring - datetime.utcnow()).total_seconds() / 60
        )

    # Get storage account information from SAS URL
    try:
        sas_url = azure_storage.sas_url
        storage_account_name = sas_url.split("//")[1].split(".")[0]
        storage_account_info = IStorageAccountInfo(
            storage_account_name=storage_account_name,
            container_name=azure_storage.container_name,
            sas_url_configured=bool(azure_storage.sas_url),
        )
    except Exception:
        storage_account_info = IStorageAccountInfo(
            storage_account_name="error",
            container_name="error",
            sas_url_configured=False,
        )

    # Get file statistics for this source
    try:
        file_stats = await _get_source_file_statistics(source_id, db)
        file_statistics = IFileStatistics(**file_stats)
    except Exception:
        file_statistics = IFileStatistics(
            total_files=0,
            status_breakdown={},
            files_last_24_hours=0,
            successful_files=0,
            failed_files=0,
            processing_files=0,
        )

    # Get monitoring health status
    monitoring_health = _get_monitoring_health_status(azure_storage)

    return IAutoDetectionStatus(
        auto_detection_enabled=azure_storage.auto_detection_enabled or False,
        monitoring_frequency_minutes=azure_storage.monitoring_frequency_minutes or 30,
        last_monitored=(
            azure_storage.last_monitored.isoformat()
            if azure_storage.last_monitored
            else None
        ),
        detection_method="polling",
        next_monitoring_time=next_monitoring_time,
        time_until_next_monitoring_minutes=time_until_next_monitoring_minutes,
        re_ingest_updated_blobs=azure_storage.re_ingest_updated_blobs or False,
        storage_account_info=storage_account_info,
        file_statistics=file_statistics,
        monitoring_health=monitoring_health,
    )


async def _get_groove_auto_detection_status(
    source, source_id: UUID, db: AsyncSession
) -> IAutoDetectionStatus:
    """Get auto-detection status for Groove sources."""
    groove_source = source.groove_source

    # Calculate next monitoring time if auto-detection is enabled
    next_monitoring_time = None
    time_until_next_monitoring_minutes = None

    if groove_source.auto_detection_enabled and groove_source.last_monitored:
        next_monitoring = groove_source.last_monitored + timedelta(
            minutes=groove_source.monitoring_frequency_minutes
        )
        next_monitoring_time = next_monitoring.isoformat()
        time_until_next_monitoring_minutes = max(
            0, (next_monitoring - datetime.utcnow()).total_seconds() / 60
        )

    # Get Groove source information
    groove_info = _get_groove_source_info(groove_source)

    # Get file statistics for this source
    try:
        file_stats = await _get_source_file_statistics(source_id, db)
        file_statistics = IFileStatistics(**file_stats)
    except Exception:
        file_statistics = IFileStatistics(
            total_files=0,
            status_breakdown={},
            files_last_24_hours=0,
            successful_files=0,
            failed_files=0,
            processing_files=0,
        )

    # Get monitoring health status
    monitoring_health = _get_groove_monitoring_health_status(groove_source)

    return IAutoDetectionStatus(
        auto_detection_enabled=groove_source.auto_detection_enabled or False,
        monitoring_frequency_minutes=groove_source.monitoring_frequency_minutes or 30,
        last_monitored=(
            groove_source.last_monitored.isoformat()
            if groove_source.last_monitored
            else None
        ),
        detection_method="polling",
        next_monitoring_time=next_monitoring_time,
        time_until_next_monitoring_minutes=time_until_next_monitoring_minutes,
        re_ingest_updated_blobs=groove_source.re_ingest_updated_tickets
        or False,  # Using tickets for Groove
        storage_account_info=groove_info,
        file_statistics=file_statistics,
        monitoring_health=monitoring_health,
    )


async def _get_source_file_statistics(
    source_id: UUID, db: AsyncSession
) -> Dict[str, Any]:
    """Get file statistics for a source."""
    # Get total file count
    total_files = await db.execute(
        select(func.count(File.id)).where(File.source_id == source_id)
    )
    total_count = total_files.scalar() or 0

    # Get files by status
    status_counts = await db.execute(
        select(File.status, func.count(File.id))
        .where(File.source_id == source_id)
        .group_by(File.status)
    )
    # Fixed C416: use dict() instead of dict comprehension
    status_breakdown = dict(status_counts.all())

    # Get recent files (last 24 hours)
    yesterday = datetime.utcnow() - timedelta(days=1)
    recent_files = await db.execute(
        select(func.count(File.id))
        .where(File.source_id == source_id)
        .where(File.created_at >= yesterday)
    )
    recent_count = recent_files.scalar() or 0

    return {
        "total_files": total_count,
        "status_breakdown": status_breakdown,
        "files_last_24_hours": recent_count,
        "successful_files": status_breakdown.get(FileStatusEnum.Uploaded, 0),
        "failed_files": status_breakdown.get(FileStatusEnum.Failed, 0),
        "processing_files": status_breakdown.get(FileStatusEnum.Processing, 0),
    }


def _get_monitoring_health_status(azure_storage: AzureStorage) -> IMonitoringHealth:
    """Get monitoring health status for Azure Storage sources."""
    if not azure_storage.auto_detection_enabled:
        return IMonitoringHealth(
            status=MonitoringHealthStatusEnum.DISABLED,
            message="Auto-detection is not enabled for this source",
        )

    if not azure_storage.last_monitored:
        return IMonitoringHealth(
            status=MonitoringHealthStatusEnum.PENDING_FIRST_SCAN,
            message="Waiting for first monitoring scan",
        )

    # Check if monitoring is overdue
    frequency_minutes = azure_storage.monitoring_frequency_minutes or 30
    expected_last_monitoring = datetime.utcnow() - timedelta(
        minutes=frequency_minutes * 2
    )  # Allow 2x frequency as buffer

    if azure_storage.last_monitored < expected_last_monitoring:
        return IMonitoringHealth(
            status=MonitoringHealthStatusEnum.OVERDUE,
            message=f"Last monitoring was {azure_storage.last_monitored.isoformat()}, expected within {frequency_minutes} minutes",
            last_monitoring_age_minutes=(
                datetime.utcnow() - azure_storage.last_monitored
            ).total_seconds()
            / 60,
        )

    # Check if monitoring is due soon
    next_expected = azure_storage.last_monitored + timedelta(minutes=frequency_minutes)
    if datetime.utcnow() > next_expected:
        return IMonitoringHealth(
            status=MonitoringHealthStatusEnum.DUE,
            message=f"Monitoring is due (last: {azure_storage.last_monitored.isoformat()})",
            minutes_overdue=(datetime.utcnow() - next_expected).total_seconds() / 60,
        )

    return IMonitoringHealth(
        status=MonitoringHealthStatusEnum.HEALTHY,
        message=f"Monitoring is healthy (last: {azure_storage.last_monitored.isoformat()})",
        next_monitoring_in_minutes=(next_expected - datetime.utcnow()).total_seconds()
        / 60,
    )


def _get_groove_source_info(groove_source: GrooveSource) -> IStorageAccountInfo:
    """Get Groove source information equivalent to storage account info."""
    try:
        return IStorageAccountInfo(
            storage_account_name=groove_source.source_name,
            container_name=(
                f"Last Ticket: {groove_source.last_ticket_number}"
                if groove_source.last_ticket_number
                else "No tickets processed"
            ),
            sas_url_configured=bool(groove_source.api_key_vault_path),
        )
    except Exception:
        return IStorageAccountInfo(
            storage_account_name="error",
            container_name="error",
            sas_url_configured=False,
        )


def _get_groove_monitoring_health_status(
    groove_source: GrooveSource,
) -> IMonitoringHealth:
    """Get monitoring health status for Groove sources."""
    if not groove_source.auto_detection_enabled:
        return IMonitoringHealth(
            status=MonitoringHealthStatusEnum.DISABLED,
            message="Auto-detection is not enabled for this source",
        )

    if not groove_source.last_monitored:
        return IMonitoringHealth(
            status=MonitoringHealthStatusEnum.PENDING_FIRST_SCAN,
            message="Waiting for first monitoring scan",
        )

    # Check if monitoring is overdue
    frequency_minutes = groove_source.monitoring_frequency_minutes or 30
    expected_last_monitoring = datetime.utcnow() - timedelta(
        minutes=frequency_minutes * 2
    )  # Allow 2x frequency as buffer

    if groove_source.last_monitored < expected_last_monitoring:
        return IMonitoringHealth(
            status=MonitoringHealthStatusEnum.OVERDUE,
            message=f"Last monitoring was {groove_source.last_monitored.isoformat()}, expected within {frequency_minutes} minutes",
            last_monitoring_age_minutes=(
                datetime.utcnow() - groove_source.last_monitored
            ).total_seconds()
            / 60,
        )

    # Check if monitoring is due soon
    next_expected = groove_source.last_monitored + timedelta(minutes=frequency_minutes)
    if datetime.utcnow() > next_expected:
        return IMonitoringHealth(
            status=MonitoringHealthStatusEnum.DUE,
            message=f"Monitoring is due (last: {groove_source.last_monitored.isoformat()})",
            minutes_overdue=(datetime.utcnow() - next_expected).total_seconds() / 60,
        )

    return IMonitoringHealth(
        status=MonitoringHealthStatusEnum.HEALTHY,
        message=f"Monitoring is healthy (last: {groove_source.last_monitored.isoformat()})",
        next_monitoring_in_minutes=(next_expected - datetime.utcnow()).total_seconds()
        / 60,
    )
