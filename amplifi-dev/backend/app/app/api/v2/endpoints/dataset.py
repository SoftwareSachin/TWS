from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi_pagination import Params
from pydantic import BaseModel
from sqlalchemy import Integer, String, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.be_core.logger import logger
from app.crud.dataset_crud_v2 import dataset_v2
from app.crud.ingest_crud import ingestion_crud
from app.db.session import SessionLocal
from app.models.document_chunk_model import ChunkTypeEnum, DocumentChunk
from app.models.document_model import Document

# from app.crud.workspace_crud import workspace as workspace_crud
from app.schemas.dataset_schema import (
    IChunkInfoRead,
    IDatasetCreate,
    IDatasetRead,
    IDatasetUpdate,
)
from app.schemas.response_schema import (
    IDeleteResponseBase,
    IGetResponsePaginated,
    IPostResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData

router = APIRouter()
crud_v2 = dataset_v2


class FileRemoveRequest(BaseModel):
    file_ids: List[UUID]


class chunk_Params(Params):
    page: int = Query(1, ge=1, description="Page number")
    size: int = Query(10, ge=1, le=100, description="Page size")


async def _get_file_metadata(db, file_id: UUID):
    """Get file record and metadata for chunks."""
    from app.models.file_model import File

    file_query = select(File).where(File.id == file_id)
    file_result = await db.execute(file_query)
    file_record = file_result.scalar_one_or_none()

    if not file_record:
        logger.warning(f"File record not found for file {file_id}")
        return None
    return file_record


async def _get_document_for_file(db, file_id: UUID, dataset_id: UUID):
    """Check if file has an associated document (image/audio)."""
    doc_query = select(Document).where(
        Document.file_id == file_id,
        Document.dataset_id == dataset_id,
        Document.deleted_at.is_(None),
    )
    doc_result = await db.execute(doc_query)
    return doc_result.scalar_one_or_none()


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


def _create_chunk_data(
    chunk,
    file_record,
    dataset_id: UUID,
    include_vectors: bool,
    partial_vectors: bool | None,
):
    """Create standardized chunk data structure."""
    chunk_metadata = chunk.chunk_metadata or {}
    chunk_type = (
        chunk.chunk_type.value
        if hasattr(chunk.chunk_type, "value")
        else str(chunk.chunk_type)
    )

    chunk_data = {
        "text": chunk.chunk_text,
        "chunk_id": str(chunk.id),
        "document_id": str(chunk.document_id),
        "vector": None,
        "chunk_order": chunk_metadata.get("chunk_order"),
        "dataset_id": str(dataset_id),
        "file": {
            "filename": file_record.filename,
            "mime_type": file_record.mimetype,
            "file_id": str(file_record.id),
            "source_id": str(file_record.source_id) if file_record.source_id else None,
            "workspace_id": (
                str(file_record.workspace_id) if file_record.workspace_id else None
            ),
        },
        "metadata": {
            "chunk_type": chunk_type,
            "confidence": chunk_metadata.get("confidence"),
            "chunked_by_engine": chunk_metadata.get("chunked_by_engine"),
            **{
                k: v
                for k, v in chunk_metadata.items()
                if k not in ["chunk_order", "confidence", "chunked_by_engine"]
            },
        },
    }

    if include_vectors:
        chunk_data["vector"] = _process_chunk_vectors(
            chunk.chunk_embedding, partial_vectors
        )

    return chunk_data


def _process_chunk_vectors(chunk_embedding, partial_vectors: bool | None):
    """Process vector data with proper handling of NumPy arrays."""
    import numpy as np

    if chunk_embedding is None:
        return None

    # Convert to list if it's a NumPy array
    if isinstance(chunk_embedding, np.ndarray):
        chunk_embedding = chunk_embedding.tolist()

    if not isinstance(chunk_embedding, list):
        return None

    if partial_vectors:
        vector_limit = min(5, len(chunk_embedding))
        return chunk_embedding[:vector_limit]

    return chunk_embedding


async def _get_document_chunksV2(db, document_id: UUID, offset: int, limit: int):
    """Get paginated and custom-ordered document chunks.

    Priority Order:
    1. PDFText
    2. PDFTable
    3. All other types
    """

    # Define a CASE for custom ordering: PDFText (0), PDFTable (1), others (2)
    chunk_type_order = case(
        (DocumentChunk.chunk_type == ChunkTypeEnum.PDFText.value, 0),
        (DocumentChunk.chunk_type == ChunkTypeEnum.PDFTable.value, 1),
        else_=2,
    )

    chunks_query = (
        select(DocumentChunk)
        .where(
            DocumentChunk.document_id == document_id, DocumentChunk.deleted_at.is_(None)
        )
        .order_by(
            chunk_type_order,  # First sort by custom chunk_type priority
            func.coalesce(  # Then sort by chunk_order metadata (if available)
                func.cast(
                    func.cast(DocumentChunk.chunk_metadata["chunk_order"], String),
                    Integer,
                ),
                999999,
            ),
            DocumentChunk.id,  # Finally sort by chunk id to guarantee stable order
        )
        .offset(offset)
        .limit(limit)
    )

    chunks_result = await db.execute(chunks_query)
    return chunks_result.scalars().all()


async def process_document_chunksV2(
    db,
    file_id: UUID,
    document_id: UUID,
    dataset_id: UUID,
    file_ingestion_id: str,
    params: chunk_Params,
    include_vectors: bool,
    partial_vectors: bool | None,
):
    """Process chunks from an image and audio document stored in our database."""
    # Get file metadata
    file_record = await _get_file_metadata(db, file_id)
    if not file_record:
        return None, 0

    # Get chunk count and pagination
    total_chunks, offset, limit = await _get_chunk_count_and_pagination(
        db, document_id, params
    )
    if not total_chunks:
        return None, 0

    # Get paginated chunks
    chunks = await _get_document_chunksV2(db, document_id, offset, limit)

    # Process chunks into the expected format
    chunk_list = [
        _create_chunk_data(
            chunk, file_record, dataset_id, include_vectors, partial_vectors
        )
        for chunk in chunks
    ]

    # Create final chunk info object
    chunk_info = {
        "file_id": str(file_id),
        "filename": deps.get_filename(file_record.filename),
        "ingestion_id": file_ingestion_id,
        "chunks": chunk_list,
        "total_chunks": total_chunks,
    }

    return chunk_info, total_chunks


async def _check_file_ingestion_status(
    db: AsyncSession, file_id: UUID, dataset_id: UUID
):
    """Check if a file has been successfully ingested using the Document table."""
    statement = select(Document).where(
        Document.file_id == file_id,
        Document.dataset_id == dataset_id,
        Document.deleted_at.is_(None),
    )
    result = await db.execute(statement)
    document = result.scalar_one_or_none()

    if not document or document.processing_status != document.processing_status.Success:
        logger.warning(
            f"File {file_id} hasn't been successfully ingested yet. "
            f"Status: {document.processing_status if document else 'Not found'}"
        )
        return None

    return document


async def process_file_chunks(
    file_id: UUID,
    dataset_id: UUID,
    params: chunk_Params,
    include_vectors: bool,
    partial_vectors: bool | None,
):
    """
    Always process chunks as image/audio documents for all files in the dataset.
    """
    async with SessionLocal() as db:
        # Check ingestion status
        ingestion_record = await _check_file_ingestion_status(db, file_id, dataset_id)
        if not ingestion_record:
            return None, 0

        file_ingestion_id = ingestion_record.id

        # Always get the document for the file (image/audio doc logic)
        document = await _get_document_for_file(db, file_id, dataset_id)
        if not document:
            logger.warning(
                f"No document found for file {file_id} in dataset {dataset_id}"
            )
            return None, 0

        logger.info(f"Processing document chunks for file {file_id}")
        return await process_document_chunksV2(
            db=db,
            file_id=file_id,
            document_id=document.id,
            dataset_id=dataset_id,
            file_ingestion_id=file_ingestion_id,
            params=params,
            include_vectors=include_vectors,
            partial_vectors=partial_vectors,
        )


def validate_partial_vectors(
    include_vectors: bool = False, partial_vectors: bool = Query(default=None)
) -> bool | None:
    if not include_vectors:
        return None
    return partial_vectors if partial_vectors is not None else True


@router.get("/workspace/{workspace_id}/dataset/{dataset_id}/chunks")
async def get_chunks_v2(
    workspace_id: UUID,
    dataset_id: UUID,
    file_id: UUID | None = None,
    include_vectors: bool = False,
    partial_vectors: bool | None = Depends(validate_partial_vectors),
    params: chunk_Params = Depends(),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.dataset_check),
) -> IGetResponsePaginated[IChunkInfoRead]:
    """
    Retrieves chunks of data for a dataset or file.

    Required roles:
    - admin
    - member
    - developer

    Note:
    - It takes in the dataset id and retrieves chunks for all the files in that dataset.
    - If the file id is passed for any particular file in that dataset, then only the chunks for that file are fetched.
    - By default, vectors are not included in chunk response but can be retrieved by passing the include_vectors parameter as True.
    - To get full vectors, partial_vectors parameter needs to be changed to False. By default, only a limited number of vector data points are fetched.
    - For split files, chunks are returned from first split only (for simplicity).
    """

    chunk_info_reads = []
    chunk_num = 0

    # Process either a specific file or all files in the dataset
    if file_id:
        # Check if file belongs to dataset
        await crud_v2.file_belongs_to_dataset_check(
            dataset_id=dataset_id, file_id=file_id
        )
        file_list = [file_id]
    else:
        # Get all files in the dataset
        file_records = await ingestion_crud.get_files_for_dataset(dataset_id=dataset_id)
        file_list = [file.id for file in file_records]

    # Process each file
    for current_file_id in file_list:
        # Try to get chunks for this file - either from splits or the original file
        try:
            # Call separate function to handle the file processing
            file_chunk_info, file_chunk_count = await process_file_chunks(
                file_id=current_file_id,
                dataset_id=dataset_id,
                params=params,
                include_vectors=include_vectors,
                partial_vectors=partial_vectors,
            )

            if file_chunk_info:
                chunk_info_reads.append(file_chunk_info)
                chunk_num += file_chunk_count
        except Exception as e:
            logger.warning(
                f"Failed to retrieve chunks for file {current_file_id}: {str(e)}"
            )

    # Check if we found any chunks
    if not chunk_info_reads:
        error_detail = f"No chunks found for {'file ' + str(file_id) if file_id else 'dataset ' + str(dataset_id)}"
        raise HTTPException(status_code=404, detail=error_detail)

    # Create paginated response
    data = IGetResponsePaginated.create(
        items=chunk_info_reads, total=chunk_num, params=params
    )
    data.message = "Chunks fetched"
    return data


@router.post(
    "/workspace/{workspace_id}/dataset", response_model=IPostResponseBase[IDatasetRead]
)
async def create_dataset(
    workspace_id: UUID,
    request_data: IDatasetCreate,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.user_workspace_access_check),
) -> IPostResponseBase[IDatasetRead]:
    """
    Creates a new dataset in a workspace.

    Required roles:
    - admin
    - developer
    """

    new_dataset = await crud_v2.create_dataset_v2(
        obj_in=request_data,
        organization_id=current_user.organization_id,
        workspace_id=workspace_id,
    )
    logger.info(f"Dataset {new_dataset.name} created.")

    return create_response(data=new_dataset, message="Dataset created successfully")


@router.put("/workspace/{workspace_id}/dataset/{dataset_id}")
async def update_dataset_v2(
    workspace_id: UUID,
    dataset_id: UUID,
    request_obj: IDatasetUpdate,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.dataset_check),
) -> IPostResponseBase[IDatasetRead]:
    """
    Update dataset using V2 API (without R2R operations).

    Required roles:
    - admin
    - developer

    Note:
    - Only updates basic fields like name and description
    - Does not perform any R2R operations
    - File associations remain unchanged
    """
    try:
        updated_dataset = await crud_v2.update_dataset_v2(
            obj_in=request_obj,
            dataset_id=dataset_id,
            workspace_id=workspace_id,
        )

        return create_response(
            data=updated_dataset,
            message="Dataset updated successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/workspace/{workspace_id}/dataset/{dataset_id}")
async def delete_dataset_v2(
    workspace_id: UUID,
    dataset_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.dataset_check),
) -> IDeleteResponseBase[IDatasetRead]:
    """
    Delete dataset using V2 API (without R2R operations).

    Required roles:
    - admin
    - developer

    Note:
    - Only performs soft delete on the dataset record
    - Does not perform any R2R operations
    - Removes dataset-file associations
    """
    try:
        deleted_dataset = await crud_v2.delete_dataset_v2(
            dataset_id=dataset_id,
            workspace_id=workspace_id,
        )

        return create_response(
            data=deleted_dataset,
            message="Dataset deleted successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/workspace/{workspace_id}/dataset/{dataset_id}/files")
async def remove_files_from_dataset(
    workspace_id: UUID,
    dataset_id: UUID,
    request: FileRemoveRequest,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.dataset_check),
) -> IDeleteResponseBase[dict]:
    """
    Remove one or multiple files from a dataset without deleting them from the files section.

    Required roles:
    - admin
    - developer

    Notes:
    - Only removes the file(s) from the dataset, does not delete the file(s) themselves
    - Cleans up associated documents, chunks, and splits for this dataset
    """
    removed_files = []
    not_found_files = []

    try:
        for file_id in request.file_ids:
            try:
                file_in_dataset = await crud_v2.file_belongs_to_dataset_check(
                    dataset_id=dataset_id, file_id=file_id
                )
                if not file_in_dataset:
                    not_found_files.append(str(file_id))
                    continue

                await crud_v2.remove_single_file_from_dataset(
                    dataset_id=dataset_id, file_id=file_id
                )
                removed_files.append(str(file_id))

            except Exception as e:
                logger.error(
                    f"Error removing file {file_id} from dataset {dataset_id}: {str(e)}"
                )
                not_found_files.append(str(file_id))

        return create_response(
            data={
                "dataset_id": dataset_id,
                "removed_files": removed_files,
                "not_found_or_failed": not_found_files,
            },
            message="File removed from dataset successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing files from dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
