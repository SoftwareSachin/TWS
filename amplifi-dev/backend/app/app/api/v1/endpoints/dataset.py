# import asyncio
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi_pagination import Params
from sqlalchemy import Integer, String, func, select

from app.api import deps
from app.be_core.logger import logger
from app.crud.dataset_crud import dataset
from app.schemas.common_schema import IOrderEnum

# from app.crud.workspace_crud import workspace as workspace_crud
from app.schemas.dataset_schema import (
    IDatasetRead,
)
from app.schemas.response_schema import (
    IGetResponseBase,
    IGetResponsePaginated,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData
from app.utils.exceptions.common_exception import IdNotFoundException
from app.utils.optional_params import OptionalParams

router = APIRouter()
crud = dataset


@router.get(
    "/workspace/{workspace_id}/dataset/{dataset_id}",
    response_model=IGetResponseBase[IDatasetRead],
)
async def get_dataset(
    workspace_id: UUID,
    dataset_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.dataset_check),
) -> IGetResponseBase[IDatasetRead]:
    """
    Retrieves a dataset by its ID.

    Required roles:
    - admin
    - member
    - developer
    """
    dataset_record = await crud.get_dataset(
        workspace_id=workspace_id, dataset_id=dataset_id
    )
    if not dataset_record:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return create_response(
        data=dataset_record, message="Dataset retrieved successfully"
    )


class chunk_Params(Params):
    page: int = Query(1, ge=1, description="Page number")
    size: int = Query(10, ge=1, le=100, description="Page size")


def validate_partial_vectors(
    include_vectors: bool = False, partial_vectors: bool = Query(default=None)
) -> bool | None:
    if not include_vectors:
        return None
    return partial_vectors if partial_vectors is not None else True


async def _get_chunk_count_and_pagination(db, document_id: UUID, params: chunk_Params):
    """Get total chunk count and calculate pagination."""
    from app.models.document_chunk_model import DocumentChunk

    # Get total count of chunks
    count_query = select(func.count()).where(
        DocumentChunk.document_id == document_id, DocumentChunk.deleted_at.is_(None)
    )
    count_result = await db.execute(count_query)
    total_chunks = count_result.scalar_one()

    if total_chunks == 0:
        logger.warning(f"No chunks found for the document {document_id}")
        return 0, None, None

    # Calculate pagination
    offset = (params.page - 1) * params.size
    limit = params.size
    return total_chunks, offset, limit


async def _get_document_chunks(db, document_id: UUID, offset: int, limit: int):
    """Get paginated and ordered document chunks."""
    from app.models.document_chunk_model import DocumentChunk

    chunks_query = (
        select(DocumentChunk)
        .where(
            DocumentChunk.document_id == document_id, DocumentChunk.deleted_at.is_(None)
        )
        .order_by(
            func.coalesce(
                func.cast(
                    func.cast(DocumentChunk.chunk_metadata["chunk_order"], String),
                    Integer,
                ),
                999999,
            ),
            DocumentChunk.id,
        )
        .offset(offset)
        .limit(limit)
    )
    chunks_result = await db.execute(chunks_query)
    return chunks_result.scalars().all()


@router.get(
    "/workspace/{workspace_id}/dataset",
    response_model=IGetResponseBase[list[IDatasetRead]]
    | IGetResponsePaginated[IDatasetRead],
)
async def get_datasets(
    workspace_id: UUID,
    ingested: Optional[bool] = None,
    type: Optional[str] = Query(None),
    params: OptionalParams = Depends(),  # Do not default this!
    order: IOrderEnum = Query(IOrderEnum.ascendent),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.user_workspace_access_check),
):
    """
    Retrieves datasets for a workspace with optional filters:
    - 'ingested' = True: only return datasets that are ingestion-ready.
    - 'dataset_type' = "sql": return datasets linked to PostgreSQL sources.
    - 'dataset_type' = "unstructured": return datasets linked to any other source type.
    - If both are None, return all datasets.

    Required roles:
    - admin
    - member
    - developer
    """
    dataset_result = await crud.get_datasets(
        workspace_id=workspace_id,
        ingested=ingested,
        type=type,
        params=params,
        order=order,
    )

    if params.page is None and params.size is None and isinstance(dataset_result, list):
        return create_response(
            data=dataset_result, message="Datasets retrieved without pagination"
        )

    return create_response(
        data=IGetResponsePaginated.create(
            items=dataset_result.items,
            total=dataset_result.total,
            params=params,
        ),
        message="Datasets retrieved with pagination",
    )


@router.get(
    "/organization/{organization_id}/dataset",
    response_model=IGetResponsePaginated[IDatasetRead],
)
async def get_datasets_by_organization(
    organization_id: UUID,
    params: Params = Depends(),
    order: IOrderEnum = Query(IOrderEnum.ascendent),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.organization_check),
) -> IGetResponsePaginated[IDatasetRead]:
    """
    Retrieves all datasets for an organization.

    Required roles:
    - admin
    - member
    - developer
    """
    paginated_datasets = await crud.get_datasets_by_organization(
        organization_id=organization_id,
        params=params,
        order=order,
    )

    pydantic_datasets = [
        IDatasetRead.from_orm(dataset) for dataset in paginated_datasets.items
    ]

    paginated_response = IGetResponsePaginated.create(
        items=pydantic_datasets,
        total=paginated_datasets.total,
        params=params,
    )

    if paginated_response:
        return create_response(data=paginated_response)
    else:
        raise IdNotFoundException(
            "No datasets found for this organization.", organization_id
        )
