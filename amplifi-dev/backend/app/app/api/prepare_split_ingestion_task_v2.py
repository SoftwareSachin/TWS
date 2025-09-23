"""
Celery task for preparing file splitting and deciding whether to split files for ingestion.
"""

from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import attributes

from app.api.deps import get_redis_client_sync
from app.be_core.celery import celery
from app.be_core.config import settings
from app.be_core.logger import logger
from app.crud.ingest_crud_v2 import ingestion_crud
from app.db.session import SyncSessionLocal
from app.models.document_model import (
    Document,
    DocumentProcessingStatusEnum,
    DocumentTypeEnum,
)
from app.models.file_split_model import FileSplit, SplitFileStatusType
from app.utils.ingestion_benchmark import IngestionBenchmark
from app.utils.ingestion_utils_v2 import (
    check_file_needs_splitting,
    check_splits_valid,
    clean_existing_splits,
    create_splits_for_file,
    get_splitting_config_hash,
    update_task_id,
)
from app.utils.processing_lock_utils import (
    _cleanup_processing_lock,
    _handle_stale_lock,
)


def _get_document_record(
    db, file_id: str, file_path: str, ingestion_id: str, dataset_id: str
) -> Optional[Document]:
    """
    Helper function to get and validate the document record.

    Args:
        db: Database session
        file_id: ID of the file
        file_path: Path to the file
        ingestion_id: ID of the ingestion process
        dataset_id: ID of the dataset

    Returns:
        The document record or None if not found
    """  # First check if document record exists with SELECT FOR UPDATE to prevent race conditions
    document = (
        db.query(Document)
        .filter(
            Document.file_id == UUID(file_id),
            Document.ingestion_id == UUID(ingestion_id),
            Document.dataset_id == UUID(dataset_id),
        )
        .with_for_update()  # Lock the query to prevent race conditions
        .first()
    )

    if not document:
        error_msg = (
            f"No document record found for file {file_id} "
            f"in ingestion {ingestion_id} for dataset {dataset_id}"
        )
        logger.error(error_msg)

        # Try to create a record if it doesn't exist
        try:
            logger.info(
                f"Attempting to create missing document record for file {file_id}"
            )

            # Get file info to populate required fields
            from app.models.file_model import File

            file_record = db.query(File).filter(File.id == UUID(file_id)).first()

            if not file_record:
                logger.error(f"File record {file_id} not found")
                return None

            # figure out document type based on mime type
            document_type = (
                DocumentTypeEnum.PDF
            )  # Default to PDF, can be adjusted based on file type
            if file_record.mime_type.startswith("image/"):
                document_type = DocumentTypeEnum.Image
            elif file_record.mime_type.startswith("audio/"):
                document_type = DocumentTypeEnum.Audio
            elif file_record.mime_type.startswith("video/"):
                document_type = DocumentTypeEnum.Video

            document = Document(
                file_id=UUID(file_id),
                ingestion_id=UUID(ingestion_id),
                dataset_id=UUID(dataset_id),
                document_type=document_type,
                processing_status=DocumentProcessingStatusEnum.Processing,
                file_path=file_path
                or file_record.file_path,  # Use provided path or file record path
                file_size=file_record.file_size,
                mime_type=file_record.mime_type,
            )
            db.add(document)
            db.commit()
            db.refresh(document)
            logger.info(f"Created missing document record for file {file_id}")
            return document
        except Exception as e:
            logger.error(f"Failed to create missing document record: {str(e)}")
            return None

    return document


def _process_direct_ingestion(
    db,
    file_id: str,
    file_path: str,
    file_metadata: dict,
    ingestion_id: str,
    dataset_id: str,
    user_id: UUID,
    organization_id: Optional[str],
    chunking_config: Optional[dict],
    skip_successful_files: bool,
    config_hash: str,
    task_id: str,
) -> Dict[str, Any]:
    """
    Helper function to process a file directly without splitting.

    Args:
        db: Database session
        file_id: ID of the file
        file_path: Path to the file
        file_metadata: Metadata for the file
        ingestion_id: ID of the ingestion process
        dataset_id: ID of the dataset
        organization_id: ID of the organization
        chunking_config: Configuration for chunking
        skip_successful_files: Whether to skip successful files
        user_id: ID of the user who initiated the ingestion
        config_hash: Hash of the current splitting configuration
        task_id: ID of the task

    Returns:
        Dictionary with results of the direct ingestion
    """
    logger.info(f"File {file_id} does not need splitting, processing directly")

    # Note about PDF image extraction handling
    logger.info(
        "If this is a PDF with images, they will be extracted and processed as child documents"
    )

    # Check for and clean existing splits
    existing_splits = (
        db.query(FileSplit).filter(FileSplit.original_file_id == UUID(file_id)).all()
    )

    if existing_splits:
        logger.info(
            f"File no longer needs splitting but has {len(existing_splits)} old splits - cleaning up"
        )
        clean_existing_splits(db, UUID(file_id), UUID(dataset_id))

    # Schedule the ingestion task (which will handle PDF image extraction if needed)
    celery.signature(
        "tasks.ingest_files_task_v2",
        kwargs={
            "file_path": file_path,
            "file_id": file_id,
            "metadata": file_metadata,
            "chunking_config": chunking_config,
            "ingestion_id": ingestion_id,
            "dataset_id": dataset_id,
            "organization_id": organization_id,
            "skip_successful_files": skip_successful_files,
            "user_id": user_id,
        },
    ).apply_async()

    return {
        "file_id": file_id,
        "ingestion_id": ingestion_id,
        "task_id": task_id,
        "success": True,
        "needs_splitting": False,
        "config_hash": config_hash,
        "message": "File does not need splitting, using regular ingestion",
    }


def _prepare_splits(
    db,
    file_id: str,
    file_path: str,
    document_id: UUID,
    dataset_id: str,
    splits_valid: bool,
) -> List[FileSplit]:
    """
    Helper function to prepare splits for a file.

    Args:
        db: Database session
        file_id: ID of the file
        file_path: Path to the file
        document_id: ID of the document associated with the file
        dataset_id: ID of the dataset
        splits_valid: Whether existing splits are valid

    Returns:
        List of file splits
    """
    # Check if splits already exist for this specific file+dataset combination
    existing_splits = (
        db.query(FileSplit)
        .filter(
            FileSplit.original_file_id == UUID(file_id),
            FileSplit.dataset_id == UUID(dataset_id),
        )
        .all()
    )

    if existing_splits and splits_valid:
        # Use existing splits for this dataset
        logger.info(
            f"Using existing valid splits for file {file_id} in dataset {dataset_id}"
        )
        return existing_splits

    if existing_splits and not splits_valid:
        # Clean up existing splits for this dataset using clean_existing_splits
        logger.info(
            f"File {file_id} needs splitting and existing splits for dataset {dataset_id} are invalid - cleaning up"
        )
        # Use the clean_existing_splits with dataset_id parameter
        clean_existing_splits(
            db=db,
            file_id=UUID(file_id),
            dataset_id=UUID(dataset_id),
            document_id=document_id,
        )

    # Create new splits
    logger.info(
        f"Creating new splits for file {file_id} in dataset {dataset_id} with current configuration"
    )
    splits = create_splits_for_file(
        db=db,
        file_id=UUID(file_id),
        file_path=file_path,
        document_id=document_id,
        dataset_id=UUID(dataset_id),
    )

    # Handle case where no splits are created
    if not splits or len(splits) == 0:
        logger.warning(
            f"No splits created for file {file_id} - this may be a problematic file"
        )
        logger.info(
            "File will be processed directly despite size estimation indicating splitting needed"
        )
        return []  # Return empty list to trigger direct processing fallback

    return splits


def _schedule_split_ingestions(
    db,
    splits: List[FileSplit],
    original_file_id: str,
    file_metadata: dict,
    ingestion_id: str,
    dataset_id: str,
    organization_id: Optional[str],
    chunking_config: Optional[dict],
    skip_successful_files: bool,
    config_hash: str,
    user_id: UUID,
) -> None:
    """
    Helper function to schedule ingestion tasks for each split.

    Args:
        db: Database session
        splits: List of file splits
        original_file_id: ID of the original file
        file_metadata: Metadata for the file
        ingestion_id: ID of the ingestion process
        dataset_id: ID of the dataset
        organization_id: ID of the organization
        chunking_config: Configuration for chunking
        skip_successful_files: Whether to skip successful files
        config_hash: Hash of the current splitting configuration
        user_id: ID of the user who initiated the ingestion
    """
    # Add metadata for original document to each split
    split_metadata = file_metadata.copy() if file_metadata else {}
    split_metadata["is_split"] = True
    split_metadata["original_file_id"] = original_file_id

    # Track scheduled vs skipped splits
    scheduled_count = 0
    skipped_count = 0

    # Schedule split ingestion tasks for each split
    for split in splits:
        # Check if this specific split should be skipped
        if skip_successful_files and split.status == SplitFileStatusType.Success:
            logger.info(
                f"Skipping split {split.id} (index {split.split_index}) - "
                f"already successful"
            )
            skipped_count += 1
            continue

        logger.info(
            f"Scheduling split {split.id} (index {split.split_index}) - "
            f"current status: {split.status}"
        )
        scheduled_count += 1

        # Reset split metadata for failed/exception splits BEFORE setting new metadata
        if split.status in [SplitFileStatusType.Failed, SplitFileStatusType.Exception]:
            logger.info(f"Resetting metadata for failed split {split.id}")
            split.split_metadata = None  # Clear all previous metadata

        # Now prepare clean metadata for this processing attempt
        this_split_metadata = split_metadata.copy()
        this_split_metadata["split_id"] = str(split.id)
        this_split_metadata["split_index"] = split.split_index
        this_split_metadata["total_splits"] = split.total_splits
        this_split_metadata["config_hash"] = config_hash

        # Schedule the ingest_split task
        celery.signature(
            "tasks.ingest_split_task_v2",
            kwargs={
                "split_id": str(split.id),
                "file_path": split.split_file_path,
                "original_file_id": original_file_id,
                "ingestion_id": ingestion_id,
                "dataset_id": dataset_id,
                "organization_id": organization_id,
                "chunking_config": chunking_config,
                "skip_successful_files": skip_successful_files,
                "metadata": this_split_metadata,
                "retry_count": 0,
                "max_retries": settings.MAX_INGESTION_RETRIES,
                "user_id": user_id,
            },
        ).apply_async()

        # update status for splits that are being processed
        split.status = SplitFileStatusType.Processing
        split.task_id = None  # Will be updated by the task
        attributes.flag_modified(split, "split_metadata")
        db.add(split)

    logger.info(
        f"Split scheduling summary for file {original_file_id}: "
        f"scheduled={scheduled_count}, skipped={skipped_count}, total={len(splits)}"
    )


def _handle_preparation_error(
    e, db, self_request_id, file_id, ingestion_id, dataset_id
):
    """
    Handle errors in the preparation process.

    Args:
        e: The exception that occurred
        db: Database session
        self_request_id: ID of the current task
        file_id: ID of the file being processed
        ingestion_id: ID of the ingestion process
        dataset_id: ID of the dataset

    Returns:
        Error response dictionary
    """
    db.rollback()
    logger.error(f"Error in prepare_split_ingestion: {str(e)}", exc_info=True)

    # Update document status - with better error handling
    try:
        document = _get_document_record(
            db=db,
            file_id=file_id,
            file_path=None,  # Pass None as we don't have file_path in this context
            ingestion_id=ingestion_id,
            dataset_id=dataset_id,
        )
        if document:
            document.processing_status = DocumentProcessingStatusEnum.Exception
            document.error_message = str(e)
            document.processed_at = datetime.now(UTC)
            db.commit()
    except Exception as status_error:
        logger.error(f"Failed to update status to Exception: {str(status_error)}")

    return {
        "file_id": file_id,
        "error": str(e),
        "success": False,
    }


def _ensure_document_record(
    db,
    file_id,
    file_path,
    ingestion_id,
    dataset_id,
    task_id,
    skip_successful_files=True,
):
    """
    Ensure that a document record exists and is up to date.

    Args:
        db: Database session
        file_id: ID of the file
        file_path: Path to the file
        ingestion_id: ID of the ingestion process
        dataset_id: ID of the dataset
        task_id: ID of the current task

    Returns:
        Tuple of (success, document_record or error_message)
    """
    # Get the document record
    document = _get_document_record(db, file_id, file_path, ingestion_id, dataset_id)

    if not document:
        # If we couldn't get or create a record, try one more approach
        try:
            # Get file record to populate required fields
            from app.models.file_model import File

            file_record = db.query(File).filter(File.id == UUID(file_id)).first()

            if not file_record:
                error_msg = f"File record {file_id} not found"
                logger.error(error_msg)
                return False, error_msg

            # Try to create or update the record using the CRUD method
            ingestion_crud.create_or_update_document_records(
                ingestion_id=ingestion_id,
                dataset_id=UUID(dataset_id),
                created_time=datetime.now(UTC),
                status=DocumentProcessingStatusEnum.Processing,
                name="Auto-created during task",
                file_ids=[UUID(file_id)],
                task_ids=[task_id],
                db_session=db,
            )
            # Try to get the record again
            document = _get_document_record(
                db, file_id, file_path, ingestion_id, dataset_id
            )
        except Exception as inner_e:
            logger.error(f"Failed final attempt to create record: {str(inner_e)}")

    if not document:
        # If we still don't have a record, we can't proceed
        error_msg = "Could not find or create document record"
        logger.error(error_msg)
        return False, error_msg

    if not skip_successful_files:
        logger.info(
            f"Fresh ingestion requested (skip_successful_files=False) - resetting document metadata for {document.id}"
        )

        # Preserve essential fields that shouldn't be reset
        essential_fields = {}
        if document.document_metadata:
            essential_fields = {
                k: v
                for k, v in document.document_metadata.items()
                if k
                in [
                    "original_pdf_summary",
                    "original_pdf_summary_preserved",
                    "file_path",
                    "file_name",
                    "file_size",
                    "dataset_id",
                    "mime_type",
                    "processing_time",
                ]
            }

        # Reset to clean slate with preserved fields
        document.document_metadata = essential_fields.copy()

        # Initialize fresh image summary
        document.document_metadata["image_summary"] = {
            "total_images": 0,
            "images_by_split": {},
            "images_by_page": {},
            "all_image_info": [],
            "processing_status": "pending",
        }

        # Reset split counts as well
        document.successful_splits_count = 0
        document.total_splits_count = 0

        attributes.flag_modified(document, "document_metadata")
        logger.info(
            f"Reset document metadata for re-ingestion of document {document.id}"
        )

    # Update status to Processing
    document.processing_status = DocumentProcessingStatusEnum.Queued
    document.task_id = task_id
    attributes.flag_modified(document, "document_metadata")
    db.commit()

    return True, document


def _acquire_processing_lock_atomic(
    redis_client, processing_key: str, task_id: str, ttl: int = 3600
) -> tuple[bool, Optional[str]]:
    """
    Atomically acquire processing lock with task ID verification.

    Args:
        redis_client: Redis client instance
        processing_key: The key to use for locking
        task_id: Current task ID
        ttl: Time to live for the lock in seconds

    Returns:
        tuple: (lock_acquired, existing_task_id)
    """
    try:
        # Try atomic acquisition first
        lock_acquired = redis_client.set(
            processing_key,
            task_id,
            nx=True,  # Only set if not exists
            ex=ttl,  # TTL in seconds
        )

        if lock_acquired:
            logger.info(f"Acquired processing lock: {processing_key}")
            return True, None

        # Lock exists - get the task that owns it
        existing_task_id = redis_client.get(processing_key)
        if existing_task_id:
            existing_task_id = (
                existing_task_id.decode()
                if isinstance(existing_task_id, bytes)
                else existing_task_id
            )

        return False, existing_task_id

    except Exception as e:
        logger.warning(f"Redis lock acquisition failed: {str(e)}")
        return True, None  # Fallback: allow processing if Redis unavailable


@celery.task(name="tasks.prepare_split_ingestion_task_v2", bind=True, acks_late=True)
def prepare_split_ingestion_task_v2(
    self,
    file_id: str,
    file_path: str,
    ingestion_id: str,
    dataset_id: str,
    user_id: UUID,
    organization_id: Optional[str] = None,
    chunking_config: Optional[dict] = None,
    skip_successful_files: bool = True,
    file_metadata: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Prepare a file for ingestion, splitting if necessary.

    This task:
    1. Checks if the file needs splitting
    2. If it does, creates splits and schedules individual ingestion tasks for each split
    3. If not, processes the file directly

    Args:
        file_id: ID of the file to ingest
        file_path: Path to the file
        ingestion_id: ID of the ingestion process
        dataset_id: ID of the dataset
        organization_id: ID of the organization
        chunking_config: Configuration for chunking
        skip_successful_files: Whether to skip successful files
        file_metadata: Optional metadata for the file
        user_id: ID of the user who initiated the ingestion

    Returns:
        Dictionary with results of the preparation process
    """
    # Enhanced task-level deduplication with atomic lock acquisition
    processing_key = None
    redis_client = None

    try:
        redis_client = get_redis_client_sync()

        # Create unique processing key
        processing_key = (
            f"processing:file:{file_id}:ingestion:{ingestion_id}:dataset:{dataset_id}"
        )

        # Attempt atomic lock acquisition
        lock_acquired, existing_task_id = _acquire_processing_lock_atomic(
            redis_client, processing_key, self.request.id, ttl=1800  # 30 minutes
        )

        if not lock_acquired:
            # Try to handle stale lock
            if not _handle_stale_lock(
                redis_client, processing_key, existing_task_id, self.request.id
            ):
                logger.info(
                    f"File {file_id} is being processed by active task {existing_task_id}"
                )
                return {
                    "file_id": file_id,
                    "success": True,
                    "skipped": True,
                    "reason": "already_processing",
                    "active_task": existing_task_id,
                }

        logger.info(f"Acquired processing lock for file {file_id}")

    except Exception as e:
        # Don't fail the task if Redis is unavailable, just log it
        logger.warning(f"Redis deduplication not available: {str(e)}")
        processing_key = None
    finally:
        # Close Redis connection after lock acquisition attempt
        if redis_client:
            try:
                redis_client.close()
            except Exception as e:
                logger.warning(f"Failed to close Redis connection: {str(e)}")
            redis_client = None

    try:
        logger.info(f"Preparing split ingestion for file: {file_id}")
        benchmark = IngestionBenchmark(file_id=file_id, ingestion_type="split")
        benchmark.start("total_split_preparation")

        # Initialize metadata if not provided
        if file_metadata is None:
            file_metadata = {}

        # Register task ID with ingestion process
        try:
            update_task_id(
                file_id=UUID(file_id),
                task_id=self.request.id,
                ingestion_id=ingestion_id,
                dataset_id=UUID(dataset_id),
            )
        except Exception as e:
            logger.error(f"Failed to update task ID: {str(e)}")
            # Continue execution as we'll try to get/create the record next

        # Open a database session
        with SyncSessionLocal() as db:
            try:
                # Ensure we have a valid document record
                success, result = _ensure_document_record(
                    db=db,
                    file_id=file_id,
                    file_path=file_path,
                    ingestion_id=ingestion_id,
                    dataset_id=dataset_id,
                    task_id=self.request.id,
                    skip_successful_files=skip_successful_files,
                )
                if not success:
                    return {"error": result, "success": False}

                document = result

                # Check if file needs splitting and get splitting configuration
                needs_splitting = check_file_needs_splitting(file_path)
                current_config_hash = get_splitting_config_hash()
                logger.info(
                    f"Current splitting configuration hash: {current_config_hash}"
                )
                splits_valid = check_splits_valid(db, UUID(file_id), document.id)

                # Process based on splitting needs
                if not needs_splitting:
                    # Process the file directly without splitting
                    return _process_direct_ingestion(
                        db=db,
                        file_id=file_id,
                        file_path=file_path,
                        file_metadata=file_metadata,
                        ingestion_id=ingestion_id,
                        dataset_id=dataset_id,
                        organization_id=organization_id,
                        chunking_config=chunking_config,
                        skip_successful_files=skip_successful_files,
                        user_id=user_id,
                        config_hash=current_config_hash,
                        task_id=self.request.id,
                    )

                # File needs splitting - prepare splits and schedule ingestion tasks
                splits = _prepare_splits(
                    db=db,
                    file_id=file_id,
                    file_path=file_path,
                    document_id=document.id,
                    dataset_id=dataset_id,
                    splits_valid=splits_valid,
                )

                # Handle case where no splits are created
                if not splits or len(splits) == 0:
                    logger.warning(
                        f"File {file_id} needs splitting but no splits were created - falling back to direct ingestion"
                    )

                    # Update document to indicate direct processing
                    document.is_split_document = False
                    document.processing_status = DocumentProcessingStatusEnum.Processing
                    db.commit()

                    # Process directly as fallback
                    return _process_direct_ingestion(
                        db=db,
                        file_id=file_id,
                        file_path=file_path,
                        file_metadata=file_metadata,
                        ingestion_id=ingestion_id,
                        dataset_id=dataset_id,
                        organization_id=organization_id,
                        chunking_config=chunking_config,
                        skip_successful_files=skip_successful_files,
                        user_id=user_id,
                        config_hash=current_config_hash,
                        task_id=self.request.id,
                    )

                # Schedule split ingestion tasks
                _schedule_split_ingestions(
                    db=db,
                    splits=splits,
                    original_file_id=file_id,
                    file_metadata=file_metadata,
                    ingestion_id=ingestion_id,
                    dataset_id=dataset_id,
                    organization_id=organization_id,
                    chunking_config=chunking_config,
                    skip_successful_files=skip_successful_files,
                    config_hash=current_config_hash,
                    user_id=user_id,
                )

                db.commit()

                result = {
                    "file_id": file_id,
                    "ingestion_id": ingestion_id,
                    "task_id": self.request.id,
                    "success": True,
                    "needs_splitting": True,
                    "splits_count": len(splits),
                    "config_hash": current_config_hash,
                    "message": f"File split into {len(splits)} parts with configuration {current_config_hash}",
                }
                benchmark.end("total_split_preparation")
                return result
            except Exception as db_error:
                db.rollback()
                logger.error(
                    f"Database error in prepare_split_ingestion: {str(db_error)}",
                    exc_info=True,
                )
                raise db_error

    except Exception as e:
        benchmark.end("total_split_preparation")
        return _handle_preparation_error(
            e, db, self.request.id, file_id, ingestion_id, dataset_id
        )
    finally:
        # Enhanced atomic cleanup - only delete if we still own the lock
        if processing_key:
            try:
                cleanup_redis = get_redis_client_sync()

                # Use atomic cleanup function
                lock_released = _cleanup_processing_lock(
                    cleanup_redis, processing_key, self.request.id
                )

                if lock_released:
                    logger.debug(
                        f"Successfully released processing lock for file {file_id}"
                    )
                else:
                    logger.debug(
                        f"Lock for file {file_id} was not owned by this task or already released"
                    )

            except Exception as e:
                logger.warning(f"Failed to clean up processing lock: {str(e)}")
            finally:
                try:
                    if "cleanup_redis" in locals():
                        cleanup_redis.close()
                except Exception as e:
                    logger.warning(
                        f"Failed to close cleanup Redis connection: {str(e)}"
                    )
