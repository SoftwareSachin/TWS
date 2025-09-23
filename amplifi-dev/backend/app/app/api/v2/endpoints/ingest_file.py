import os
from datetime import UTC, datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api import deps
from app.api.deps import get_redis_client
from app.be_core.celery import celery
from app.be_core.logger import logger
from app.crud.dataset_crud_v2 import dataset_v2
from app.crud.ingest_crud_v2 import CRUDIngestionV2
from app.db.session import SyncSessionLocal
from app.models.document_model import Document, DocumentProcessingStatusEnum
from app.models.file_model import File
from app.models.source_model import Source
from app.schemas.ingest_request_schema import IIngestDatasetCreate
from app.schemas.response_schema import (
    IGetResponseBase,
    IIngestFilesOperationRead,
    IPostResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData
from app.utils.feature_flags import is_video_ingestion_enabled
from app.utils.ingestion_status_propagation import cleanup_processing_flag
from app.utils.uuid6 import uuid7

router = APIRouter()
crud = CRUDIngestionV2(Document)


def process_files_background(
    files: List[File],
    dataset_id: UUID,
    organization_id: UUID,
    ingestion_id: str,
    chunking_config_instance: dict,
    metadata: dict,
    user_id: UUID,
    skip_successful_files: bool = True,
):
    """
    Process files in the background using the unified ingestion approach.

    Args:
        files: List of files to process
        dataset_id: Dataset ID
        organization_id: Organization ID
        ingestion_id: Ingestion ID
        chunking_config_instance: Chunking configuration
        metadata: Metadata to associate with files
        user_id: User ID for websocket notifications
        skip_successful_files: Whether to skip files that are already successfully ingested
    """
    task_ids = []
    # Track which files we've already submitted in this batch to prevent duplicates
    submitted_files = set()

    for file in files:
        try:
            # Create unique file key for deduplication
            file_key = f"{file.id}:{dataset_id}:{ingestion_id}"

            if file_key in submitted_files:
                logger.warning(f"Skipping duplicate file submission: {file.id}")
                continue

            submitted_files.add(file_key)

            # Create metadata for this file
            file_metadata = {
                "dataset_id": str(dataset_id),
                "file_id": str(file.id),
                **metadata,
            }

            # Check file type
            is_image = _is_image_file(file.mimetype, file.filename)
            is_audio = _is_audio_file(file.mimetype, file.filename)
            is_video = _is_video_file(file.mimetype, file.filename)
            is_document = _is_document_file(file.mimetype, file.filename)

            logger.debug(
                f"Preparing ingestion task for file: file_id={file.id}, file_path={file.file_path}, "
                f"is_image={is_image}, is_audio={is_audio}, is_video={is_video}, is_document={is_document}"
            )

            # Add file path and size to metadata for media files
            if is_image or is_audio or is_video:
                # For video files, check if original file still exists
                # If not, we'll use stored metadata or segment-based processing
                if is_video and not os.path.exists(file.file_path):
                    logger.info(
                        f"Original video file no longer exists: {file.file_path}"
                    )
                    # For video files, we can still process using existing segments
                    # The video ingestion task will handle this case
                    file_metadata.update(
                        {
                            "file_path": file.file_path,  # Keep original path for reference
                            "file_size": file.size
                            or 0,  # Use stored size from database
                            "original_file_deleted": True,  # Flag to indicate original is gone
                        }
                    )
                else:
                    # Original file exists, use actual file system data
                    try:
                        actual_file_size = os.path.getsize(file.file_path)
                        file_metadata.update(
                            {
                                "file_path": file.file_path,
                                "file_size": actual_file_size,
                            }
                        )
                    except (OSError, IOError) as e:
                        logger.warning(
                            f"Could not get file size for {file.file_path}: {e}"
                        )
                        # Fallback to stored size
                        file_metadata.update(
                            {
                                "file_path": file.file_path,
                                "file_size": file.size or 0,
                            }
                        )

            if is_image:
                # Use the image document processing pipeline - use v2 version
                logger.debug(
                    f"Processing image file with document processor: {file.filename}"
                )
                task = celery.signature(
                    "tasks.image_ingestion_task_v2",
                    kwargs={
                        "file_id": str(file.id),
                        "file_path": file.file_path,
                        "ingestion_id": ingestion_id,
                        "dataset_id": str(dataset_id),
                        "metadata": file_metadata,
                        "user_id": str(user_id),
                        "skip_successful_files": skip_successful_files,
                    },
                ).apply_async()
            elif is_audio:
                # Use the audio document processing pipeline
                logger.debug(
                    f"Processing audio file with document processor: {file.filename}"
                )
                task = celery.signature(
                    "tasks.audio_ingestion_task_v2",
                    kwargs={
                        "file_id": str(file.id),
                        "file_path": file.file_path,
                        "ingestion_id": ingestion_id,
                        "dataset_id": str(dataset_id),
                        "metadata": file_metadata,
                        "user_id": str(user_id),
                        "skip_successful_files": skip_successful_files,
                        "chunk_size": chunking_config_instance.get(
                            "max_characters", 100
                        ),
                        "chunk_overlap": chunking_config_instance.get("overlap", 20),
                    },
                ).apply_async()
            elif is_video:
                # Check if video ingestion is enabled
                if not is_video_ingestion_enabled():
                    logger.warning(
                        f"Video ingestion is disabled - skipping video file: {file.filename}"
                    )
                    continue

                # Use the video document processing pipeline
                logger.debug(
                    f"Processing video file with video processor: {file.filename}"
                )
                task = celery.signature(
                    "tasks.video_ingestion_task",
                    kwargs={
                        "file_id": str(file.id),
                        "file_path": file.file_path,
                        "ingestion_id": ingestion_id,
                        "dataset_id": str(dataset_id),
                        "metadata": file_metadata,
                        "user_id": str(user_id),
                        "skip_successful_files": skip_successful_files,
                    },
                ).apply_async()
            elif is_document:
                # For document files (PDF, DOCX, XLSX, PPTX, MD, HTML, CSV), use prepare_split_ingestion_task
                logger.debug(
                    f"Using prepare_split_ingestion_task for document file: {file.filename}"
                )
                task = celery.signature(
                    "tasks.prepare_split_ingestion_task_v2",
                    kwargs={
                        "file_id": str(file.id),
                        "file_path": file.file_path,
                        "ingestion_id": ingestion_id,
                        "dataset_id": str(dataset_id),
                        "user_id": str(user_id),
                        "organization_id": str(organization_id),
                        "chunking_config": chunking_config_instance,
                        "skip_successful_files": skip_successful_files,
                        "file_metadata": file_metadata,
                    },
                ).apply_async()
            else:
                # For unsupported file types, log a warning and skip
                logger.warning(
                    f"Unsupported file type for {file.filename} (mimetype: {file.mimetype}). Skipping."
                )
                continue

            task_ids.append(task.id)
            logger.info(f"Submitted task {task.id} for file {file.id}")

        except Exception as e:
            logger.error(
                f"Error submitting task for file {file.id}: {str(e)}", exc_info=True
            )
            continue


def _is_image_file(mimetype: str, filename: str) -> bool:
    """Check if a file is an image based on mimetype and extension"""
    # Supported MIME types
    supported_mimetypes = {"image/jpeg", "image/jpg", "image/png"}

    # Check MIME type first (exact match)
    if mimetype and mimetype in supported_mimetypes:
        return True

    # Fallback to extension check
    if filename:
        # Extract extension more safely
        ext = os.path.splitext(filename.lower())[1]
        if ext in {".jpg", ".jpeg", ".png"}:
            return True

    return False


def _is_audio_file(mimetype: str, filename: str) -> bool:
    """Check if a file is an audio based on mimetype and extension"""

    # Supported MIME types for audio files
    supported_mimetypes = {
        "audio/wav",
        "audio/mp3",
        "audio/aac",
        "audio/mpeg",
    }
    # Check if mimetype is in supported types
    if mimetype and mimetype in supported_mimetypes:
        return True

    # Fallback to extension check
    if filename:
        ext = os.path.splitext(filename.lower())[1]
        if ext in {".wav", ".mp3", ".aac", ".mpeg"}:
            return True

    return False


def _is_document_file(mimetype: str, filename: str) -> bool:
    """Check if a file is a supported document type based on mimetype and extension"""

    # Supported MIME types for document files
    supported_mimetypes = {
        # PDF
        "application/pdf",
        # Microsoft Office formats
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # XLSX
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # PPTX
        # Legacy Office formats
        "application/msword",  # DOC
        "application/vnd.ms-excel",  # XLS
        "application/vnd.ms-powerpoint",  # PPT
        # Text formats
        "text/markdown",
        "text/html",
        "text/csv",
        "application/csv",
        # Generic text types that might be markdown/html
        "text/plain",
    }

    # Check MIME type first
    if mimetype and mimetype in supported_mimetypes:
        return True

    # Fallback to extension check
    if filename:
        ext = os.path.splitext(filename.lower())[1]
        supported_extensions = {
            ".pdf",
            ".docx",
            ".doc",
            ".xlsx",
            ".xls",
            ".pptx",
            ".ppt",
            ".md",
            ".markdown",
            ".html",
            ".htm",
            ".csv",
        }
        if ext in supported_extensions:
            return True

    return False


def _is_video_file(mimetype: str, filename: str) -> bool:
    """Check if a file is a video based on mimetype and extension"""

    # Supported MIME types for video files
    supported_mimetypes = {
        "video/mp4",
        "video/x-msvideo",  # AVI
        "video/quicktime",  # MOV
        "video/x-ms-wmv",  # WMV
        "video/x-flv",  # FLV
        "video/webm",  # WebM
        "video/x-matroska",  # MKV
    }

    # Check if mimetype is in supported types
    if mimetype and mimetype in supported_mimetypes:
        return True

    # Fallback to extension check
    if filename:
        ext = os.path.splitext(filename.lower())[1]
        if ext in {".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".mkv"}:
            return True

    return False


@router.post(
    "/dataset/{dataset_id}/ingest",
    response_model=IPostResponseBase[List[IIngestFilesOperationRead]],
)
async def ingest_files(
    dataset_id: UUID,
    request_body: IIngestDatasetCreate,
    skip_successful_files: Optional[bool] = True,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.dataset_check),
) -> IPostResponseBase[List[IIngestFilesOperationRead]]:
    """
    Initiates file ingestion for a specified dataset.

    Required roles:
    - admin
    - developer

    Args:
        dataset_id: Dataset ID to process
        request_body: Ingestion details and metadata
        skip_successful_files: If True, skips files that are already successfully processed

    Note:
    - It initiates file ingestion for all files in that dataset and returns response including
      an ingestion_id that can be used for checking the ingestion status later.
    """

    # Enhanced deduplication logic
    request_key = None
    processing_key = None
    redis_client = None
    try:
        redis_client = await get_redis_client()

        # Create dataset-level processing key (independent of request content)
        processing_key = f"ingestion:processing:{dataset_id}:{current_user.id}"

        # Create request-level key (for exact duplicate detection)
        request_key = f"ingestion:request:{dataset_id}:{current_user.id}:{hash(str(request_body.dict()))}"

        # Check if dataset is currently being processed
        processing_exists = await redis_client.exists(processing_key)
        if processing_exists:
            processing_info = await redis_client.get(processing_key)
            processing_info_str = (
                processing_info.decode()
                if isinstance(processing_info, bytes)
                else str(processing_info)
            )
            logger.warning(
                f"Dataset {dataset_id} is already being processed by {processing_info_str}"
            )
            return create_response(
                data=[],
                message="Dataset is currently being processed. Please wait for completion.",
            )

        # Check for exact duplicate request
        request_exists = await redis_client.exists(request_key)
        if request_exists:
            logger.warning(f"Duplicate request detected for dataset {dataset_id}")
            return create_response(
                data=[],
                message="Identical ingestion request already submitted recently",
            )

        # Set both flags atomically
        await redis_client.setex(
            processing_key, 3600, f"user:{current_user.id}"
        )  # 1 hour
        await redis_client.setex(request_key, 300, "processing")  # 5 minutes

        logger.info(
            f"Set processing flags - processing: {processing_key}, request: {request_key}"
        )

    except Exception as e:
        logger.warning(f"Redis deduplication not available: {str(e)}")
        request_key = None
        processing_key = None
    finally:
        # Close Redis connection after setting flags
        if redis_client:
            try:
                await redis_client.close()
            except Exception as e:
                logger.warning(f"Failed to close Redis connection: {str(e)}")

    try:
        # Get all files in the dataset
        all_files = await crud.get_files_for_dataset(dataset_id=dataset_id)
        if not all_files:
            raise HTTPException(
                status_code=404, detail="No files found for the dataset"
            )

        # Get chunking configuration
        chunking_config_instance, config_id = (
            await crud.get_chunking_config_for_dataset(dataset_id=dataset_id)
        )

        ingestion_id = str(uuid7())
        created_time = datetime.now(UTC)

        # Determine which files to process
        files_to_process = []

        # Standard processing logic
        if skip_successful_files:
            # Only process files that are not already successfully ingested
            file_ids = [file.id for file in all_files]
            successful_files = await crud.get_successful_files(
                file_ids=file_ids, dataset_id=dataset_id
            )
            files_to_process = [f for f in all_files if f.id not in successful_files]

            logger.info(
                f"Processing {len(files_to_process)} files, skipping {len(successful_files)} successful files"
            )
        else:
            # Process all files
            files_to_process = all_files
            logger.info(
                f"Processing all {len(all_files)} files (skip_successful_files=False)"
            )

        # If no files to process, return early
        if not files_to_process:
            cleanup_processing_flag(dataset_id, current_user.id)
            return create_response(data=[], message="No files to process")

        # Create document records in database
        await crud.create_or_update_document_records(
            ingestion_id=ingestion_id,
            dataset_id=dataset_id,
            created_time=created_time,
            status=DocumentProcessingStatusEnum.Not_Started,
            file_ids=[f.id for f in files_to_process],
            task_ids=[None] * len(files_to_process),
        )

        # Start background processing
        process_files_background(
            files=files_to_process,
            dataset_id=dataset_id,
            organization_id=current_user.organization_id,
            ingestion_id=ingestion_id,
            chunking_config_instance=chunking_config_instance,
            metadata=request_body.metadata,
            skip_successful_files=skip_successful_files,
            user_id=current_user.id,
        )

        # Create response
        ingestion_response = await crud.create_ingestion_initiation_response(
            files=files_to_process,
            ingestion_id=ingestion_id,
            chunking_config_id=config_id,
            created_time=created_time,
        )

        return create_response(
            data=ingestion_response, message="Ingestion process initiated"
        )

    except Exception as e:
        # If ingestion fails, clean up processing flag immediately
        if processing_key:
            try:
                cleanup_redis = await get_redis_client()
                await cleanup_redis.delete(processing_key)
                await cleanup_redis.close()
                logger.info(
                    f"Cleaned up processing flag {processing_key} due to error: {e}"
                )
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to clean up processing flag on error: {str(cleanup_error)}"
                )
        raise

    finally:
        # Only clean up request flag, keep processing flag until completion
        if request_key:
            try:
                cleanup_redis = await get_redis_client()
                await cleanup_redis.delete(request_key)
                await cleanup_redis.close()
                logger.debug(f"Cleaned up request flag: {request_key}")
            except Exception as e:
                logger.warning(f"Failed to clean up request flag: {str(e)}")


@router.get(
    "/dataset/{dataset_id}/ingestion_status",
    response_model=IGetResponseBase[List[IIngestFilesOperationRead]],
)
async def get_ingestion_status(
    dataset_id: UUID,
    ingestion_id: Optional[UUID] = None,
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.dataset_check),
) -> IGetResponseBase[List[IIngestFilesOperationRead]]:
    """
    Retrieves the ingestion status of a dataset.

    Required roles:
    - admin
    - member
    - developer

    Note:
    - It takes in dataset id and returns the ingestion status for all the files present in that dataset.
    - If ingestion id is passed as parameter, it will return the ingestion status for the files ingested in that particular ingestion task.
    """
    db_session: Session = SyncSessionLocal()
    source_id = await dataset_v2.get_source_id_by_dataset_id(dataset_id=dataset_id)
    logger.info(f"Source ID: {source_id}")

    # Get source type from source ID
    source_result = db_session.execute(
        select(Source.source_type).where(Source.id == source_id)
    )
    source_type = source_result.scalars().first()
    logger.info(f"Source type: {source_type}")

    # Close the database session
    db_session.close()
    return await crud.get_ingestion_status(
        dataset_id=dataset_id,
        ingestion_id=ingestion_id,
        source_type=source_type,
    )
