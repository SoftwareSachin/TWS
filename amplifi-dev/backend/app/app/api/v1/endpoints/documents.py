from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.crud.document_crud import document as document_crud
from app.models.document_chunk_model import ChunkTypeEnum
from app.models.document_model import (
    DocumentProcessingStatusEnum,
    DocumentTypeEnum,
)
from app.schemas.document_schema import (
    DocumentChunkRead,
    DocumentProcessingStatusResponse,
    DocumentRead,
    DocumentUpdate,
    DocumentWithChunks,
)
from app.schemas.response_schema import (
    IDeleteResponseBase,
    IGetResponseBase,
    IPutResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData

router = APIRouter()


@router.get(
    "/{dataset_id}/documents", response_model=IGetResponseBase[List[DocumentRead]]
)
async def get_documents(
    dataset_id: UUID,
    document_type: Optional[DocumentTypeEnum] = None,
    status: Optional[DocumentProcessingStatusEnum] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Get all documents for a dataset with optional filtering by document type and status.

    Required roles:
    - admin
    - member
    - developer
    """
    documents = await document_crud.get_documents_by_dataset(
        db=db,
        dataset_id=dataset_id,
        document_type=document_type,
        status=status,
        skip=skip,
        limit=limit,
    )

    return create_response(
        data=documents, message=f"Retrieved {len(documents)} documents"
    )


@router.get(
    "/{dataset_id}/documents/{document_id}",
    response_model=IGetResponseBase[DocumentWithChunks],
)
async def get_document_with_chunks(
    dataset_id: UUID,
    document_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Get a document by ID, including its chunks.

    Required roles:
    - admin
    - member
    - developer
    """
    # Get the document
    document = await document_crud.get_document(db=db, document_id=document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    # Verify the document belongs to the specified dataset
    if document.dataset_id != dataset_id:
        raise HTTPException(
            status_code=403, detail="Document does not belong to the specified dataset"
        )

    # Get the chunks for this document
    chunks = await document_crud.get_chunks_by_document(db=db, document_id=document_id)

    # Create a DocumentWithChunks model
    document_with_chunks = DocumentWithChunks.model_validate(document)
    document_with_chunks.chunks = chunks

    return create_response(
        data=document_with_chunks,
        message=f"Retrieved document {document_id} with {len(chunks)} chunks",
    )


@router.get(
    "/{dataset_id}/documents/{document_id}/chunks",
    response_model=IGetResponseBase[List[DocumentChunkRead]],
)
async def get_document_chunks(
    dataset_id: UUID,
    document_id: UUID,
    chunk_type: Optional[ChunkTypeEnum] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Get all chunks for a document with optional filtering by chunk type.

    Required roles:
    - admin
    - member
    - developer
    """
    # Verify document exists and belongs to dataset
    document = await document_crud.get_document(db=db, document_id=document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    if document.dataset_id != dataset_id:
        raise HTTPException(
            status_code=403, detail="Document does not belong to the specified dataset"
        )

    # Get the chunks
    chunks = await document_crud.get_chunks_by_document(
        db=db, document_id=document_id, chunk_type=chunk_type, skip=skip, limit=limit
    )

    return create_response(
        data=chunks,
        message=f"Retrieved {len(chunks)} chunks for document {document_id}",
    )


@router.get(
    "/{dataset_id}/documents/{document_id}/status",
    response_model=IGetResponseBase[DocumentProcessingStatusResponse],
)
async def get_document_status(
    dataset_id: UUID,
    document_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Get the processing status of a document.

    Required roles:
    - admin
    - member
    - developer
    """
    document = await document_crud.get_document(db=db, document_id=document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    if document.dataset_id != dataset_id:
        raise HTTPException(
            status_code=403, detail="Document does not belong to the specified dataset"
        )

    status_response = DocumentProcessingStatusResponse(
        document_id=document.id,
        status=document.processing_status,
        file_id=document.file_id,
        document_type=document.document_type,
        processed_at=document.processed_at,
        error_message=document.error_message,
    )

    return create_response(
        data=status_response,
        message=f"Status for document {document_id}: {document.processing_status.value}",
    )


@router.delete(
    "/{dataset_id}/documents/{document_id}", response_model=IDeleteResponseBase
)
async def delete_document(
    dataset_id: UUID,
    document_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Delete a document and all its chunks.

    Required roles:
    - admin
    - developer
    """
    # Verify document exists and belongs to dataset
    document = await document_crud.get_document(db=db, document_id=document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    if document.dataset_id != dataset_id:
        raise HTTPException(
            status_code=403, detail="Document does not belong to the specified dataset"
        )

    # Delete the document and its chunks
    success = await document_crud.delete_document(db=db, document_id=document_id)

    if not success:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document {document_id}"
        )

    return create_response(
        data={"deleted": True}, message=f"Document {document_id} deleted successfully"
    )


@router.put(
    "/{dataset_id}/documents/{document_id}/reprocess",
    response_model=IPutResponseBase[DocumentRead],
)
async def reprocess_document(
    dataset_id: UUID,
    document_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Reprocess an existing document by resetting its status and triggering the processing pipeline again.

    Required roles:
    - admin
    - developer
    """
    # Verify document exists and belongs to dataset
    document = await document_crud.get_document(db=db, document_id=document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    if document.dataset_id != dataset_id:
        raise HTTPException(
            status_code=403, detail="Document does not belong to the specified dataset"
        )

    # Reset document status
    document_update = DocumentUpdate(
        processing_status=DocumentProcessingStatusEnum.Queued, error_message=None
    )
    updated_document = await document_crud.update_document(
        db=db, db_obj=document, obj_in=document_update
    )

    # Delete existing chunks
    await document_crud.delete_chunks_by_document(db=db, document_id=document_id)

    # Trigger reprocessing using Celery
    from app.be_core.celery import celery

    task = celery.signature(
        "tasks.process_image_document_task",
        kwargs={
            "file_id": str(document.file_id),
            "file_path": document.file_path,
            "ingestion_id": (
                str(document.ingestion_id) if document.ingestion_id else None
            ),
            "dataset_id": str(dataset_id),
            # "chunking_config": {},  # Use default chunking config
            "metadata": document.metadata,
        },
    ).apply_async()

    # Update document with the new task ID
    document_update = DocumentUpdate(task_id=task.id)
    updated_document = await document_crud.update_document(
        db=db, db_obj=updated_document, obj_in=document_update
    )

    return create_response(
        data=updated_document, message=f"Document {document_id} reprocessing initiated"
    )
