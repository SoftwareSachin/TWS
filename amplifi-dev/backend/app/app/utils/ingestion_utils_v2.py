"""
Utility functions for file ingestion, supporting both regular and split-based ingestion.
"""

import json
import os
from datetime import datetime

# from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from limits import RateLimitItemPerMinute, storage, strategies

from app.be_core.config import settings
from app.be_core.logger import logger
from app.crud.file_split_crud_v2 import file_split_crud
from app.crud.ingest_crud_v2 import ingestion_crud
from app.db.session import SyncSessionLocal
from app.models.document_chunk_model import DocumentChunk
from app.models.document_model import Document, DocumentProcessingStatusEnum
from app.models.file_model import File
from app.models.file_split_model import FileSplit

# from app.utils.rate_limiter import rate_limiter
from app.schemas.long_task_schema import ITaskType
from app.utils.datetime_utils import serialize_datetime
from app.utils.file_splitter import FileSplitter

# Constants
POLL_INTERVAL = 10
TERMINAL_STATUSES = ["success", "failed"]

# Initialize the file splitter for token counting
file_splitter = FileSplitter()

if settings.REDIS_MODE == "cluster":
    # Generate Redis URI from settings
    REDIS_URI = f"redis+cluster://{settings.REDIS_HOST}:{settings.REDIS_PORT}"

    # Initialize rate limiter
    redis_storage = storage.RedisClusterStorage(REDIS_URI)
else:
    # Generate Redis URI from settings
    REDIS_URI = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0"

    # Initialize rate limiter
    redis_storage = storage.RedisStorage(REDIS_URI)

# limiter = strategies.MovingWindowRateLimiter(redis_storage)
limiter = strategies.FixedWindowRateLimiter(redis_storage)

INGESTION_RATE_LIMIT = RateLimitItemPerMinute(
    settings.AZURE_RATE_LIMIT_TOKENS_PER_MINUTE
)


def acquire_tokens_with_retry(token_count: int) -> Tuple[bool, Optional[int]]:
    """
    Token acquisition with PROPER time calculation
    Returns: (success, remaining_retry_time_seconds)
    """
    try:
        from time import time

        current_time = time()
        success = limiter.hit(
            INGESTION_RATE_LIMIT, "global_ingestion", cost=token_count
        )
        stats = limiter.get_window_stats(INGESTION_RATE_LIMIT, "global_ingestion")

        # Calculate remaining time properly
        remaining_time = max(0, (stats[0] - current_time)) if stats and stats[0] else 60

        if success:
            logger.info(
                f"RateLimit: Acquired {token_count} tokens | "
                f"Used: {settings.AZURE_RATE_LIMIT_TOKENS_PER_MINUTE - stats[1]} | "
                f"Resets in: {remaining_time:.1f}s"
            )
            return True, None

        # Cap retry time at 60s for minute-based windows
        safe_retry_time = min(remaining_time, 60)
        logger.warning(
            f"RateLimit: Denied {token_count} tokens | "
            f"Retry in: {safe_retry_time:.1f}s | "
            f"Used: {settings.AZURE_RATE_LIMIT_TOKENS_PER_MINUTE - stats[1]}"
        )
        return False, safe_retry_time

    except Exception as e:
        logger.error(f"Error while acquiring tokens: {str(e)}", exc_info=True)
        return False, 60  # Return default retry time of 60 seconds on error


def update_task_id(
    file_id: UUID, task_id: str, ingestion_id: UUID, dataset_id: UUID
) -> None:
    """
    Update the task ID for a document.

    Args:
        file_id: ID of the file
        task_id: New task ID
        ingestion_id: ID of the ingestion process
        dataset_id: ID of the dataset
    """
    try:
        ingestion_crud.update_document_task_id_sync(
            ingestion_id=ingestion_id,
            file_id=file_id,
            task_id=task_id,
            dataset_id=dataset_id,
        )
        logger.info(
            f"Updated task_id {task_id} for file {file_id} in ingestion {ingestion_id}"
        )
    except Exception as e:
        logger.error(f"Error updating task ID: {str(e)}", exc_info=True)


def update_document_status(
    task_id: str,
    status: DocumentProcessingStatusEnum,
    error_message: str,
    doc: dict = None,
) -> None:
    """
    Update the status of a document record.

    Args:
        task_id: ID of the Celery task
        status: New status to set
        doc: Optional document data, which can be a dict or object
    """
    try:
        from sqlalchemy import update

        with SyncSessionLocal() as db:
            # Get the processed_at timestamp from the document or use current time
            processed_at = datetime.now()
            if doc:
                # Handle both object and dictionary formats
                if isinstance(doc, dict):
                    # Dictionary format
                    processed_at = (
                        doc.get("updated_at")
                        or doc.get("processed_at")
                        or datetime.now()
                    )
                else:
                    # Object format
                    processed_at = (
                        getattr(doc, "updated_at", None)
                        or getattr(doc, "processed_at", None)
                        or datetime.now()
                    )

            # Update the document record
            stmt = (
                update(Document)
                .where(Document.task_id == task_id)
                .values(
                    processing_status=status,
                    processed_at=processed_at,
                    error_message=error_message,
                )
            )
            db.execute(stmt)
            db.commit()

        logger.info(f"Updated document status for task {task_id} to {status.value}")
    except Exception as e:
        logger.error(f"Error updating document status: {str(e)}", exc_info=True)


def publish_ingestion_status(
    user_id: UUID, ingestion_id: str, task_id: str, ingestion_result: Dict[str, Any]
) -> None:
    """
    Publish ingestion status to WebSocket.

    Args:
        user_id: ID of the user who initiated the ingestion
        ingestion_id: ID of the ingestion process
        task_id: ID of the Celery task
        ingestion_result: Results of the ingestion process
    """
    from app.api.deps import publish_websocket_message

    ws_result = {
        k: v
        for k, v in ingestion_result.items()
        if k not in ["size_in_bytes", "metadata", "ingestion_attempt_number"]
    }
    ws_result["ingestion_id"] = ingestion_id
    ws_result["task_id"] = f"{task_id}"
    publish_websocket_message(
        f"{user_id}:{ITaskType.ingestion}",
        json.dumps(ws_result, default=serialize_datetime),
    )


def check_file_needs_splitting(file_path: str) -> bool:
    """
    Determine if a file needs to be split based on its content using the centralized
    token counting approach.

    Args:
        file_path: Path to the file to check

    Returns:
        True if the file needs to be split, False otherwise
    """
    # Get file size
    file_size = os.path.getsize(file_path)

    # Very small files don't need splitting
    # if file_size < file_splitter.SMALL_FILE_THRESHOLD:
    #     logger.debug(
    #         f"File {file_path} is smaller than {file_splitter.SMALL_FILE_THRESHOLD/1024/1024} MB, no splitting needed"
    #     )
    #     return False

    # Use the centralized token counting method from FileSplitter
    try:
        # Count estimated tokens for the file
        estimated_tokens = file_splitter.count_file_tokens(file_path)

        # Log the estimated token count
        logger.info(f"File {file_path} estimated tokens: {estimated_tokens}")

        # Determine if splitting is needed based on token count
        needs_splitting = estimated_tokens > settings.MAX_TOKENS_PER_SPLIT

        if needs_splitting:
            logger.info(
                f"File {file_path} needs splitting ({estimated_tokens} tokens > {settings.MAX_TOKENS_PER_SPLIT})"
            )
        else:
            logger.info(
                f"File {file_path} does not need splitting ({estimated_tokens} tokens <= {settings.MAX_TOKENS_PER_SPLIT})"
            )

        return needs_splitting

    except Exception as e:
        logger.warning(f"Error estimating tokens for file {file_path}: {e}")
        # For files over 5MB, assume splitting is needed when estimation fails
        needs_splitting = file_size > 5 * 1024 * 1024
        if needs_splitting:
            logger.info("Falling back to size-based decision: file needs splitting")
        else:
            logger.info(
                "Falling back to size-based decision: file does not need splitting"
            )
        return needs_splitting


def cleanup_split_files(db, file_id: UUID) -> int:
    """
    Clean up split files for a document.
    Will only perform cleanup if SPLIT_CLEANUP_ENABLED setting is True.
    Only deletes the temporary files without removing database records.

    Args:
        db: Database session
        file_id: ID of the original file

    Returns:
        Number of temporary files deleted
    """
    # Check if split cleanup is enabled
    if not settings.SPLIT_CLEANUP_ENABLED:
        logger.info(
            f"Split cleanup is disabled. Skipping split files cleanup for file {file_id}"
        )
        return 0

    try:
        # Get all splits for this file
        splits = file_split_crud.get_splits_for_file(db, file_id)

        if not splits:
            logger.debug(f"No splits found to clean up for file {file_id}")
            return 0

        # Track file deletion count
        file_delete_count = 0

        # Delete each split file from filesystem but keep database records
        for split in splits:
            split_path = split.split_file_path
            try:
                if os.path.exists(split_path):
                    os.remove(split_path)
                    logger.debug(f"Removed existing split file: {split_path}")
                    file_delete_count += 1
                else:
                    logger.warning(f"Existing split file not found: {split_path}")
            except Exception as e:
                logger.error(
                    f"Error removing existing split file {split_path}: {str(e)}"
                )
                # Continue with cleanup even if one file fails

        logger.info(
            f"Cleaned up {file_delete_count} temporary split files for file {file_id} while keeping database records"
        )
        return file_delete_count

    except Exception as e:
        logger.error(
            f"Error during temporary split files cleanup: {str(e)}", exc_info=True
        )
        return 0


def get_splitting_config_hash() -> str:
    """
    Generate a hash based on the current file splitting configuration settings.
    This hash helps track which configuration was used to create a set of splits.
    Note: This hash is for configuration tracking, not security purposes.

    Returns:
        A string hash representing the current configuration
    """
    import hashlib
    import json

    from app.utils.file_splitter import (
        LARGE_FILE_THRESHOLD,
        MAX_SAMPLES,
        MEDIUM_FILE_THRESHOLD,
        SAMPLE_SIZE_LARGE,
        SAMPLE_SIZE_MEDIUM,
        SAMPLE_SIZE_SMALL,
        SMALL_FILE_THRESHOLD,
    )

    # Collect all splitting-related settings
    config_dict = {
        "MAX_TOKENS_PER_SPLIT": settings.MAX_TOKENS_PER_SPLIT,
        "MIN_SPLIT_SIZE": settings.MIN_SPLIT_SIZE,
        # Include any other settings that affect splitting behavior
        "SMALL_FILE_THRESHOLD": SMALL_FILE_THRESHOLD,
        "MEDIUM_FILE_THRESHOLD": MEDIUM_FILE_THRESHOLD,
        "LARGE_FILE_THRESHOLD": LARGE_FILE_THRESHOLD,
        "SAMPLE_SIZE_SMALL": SAMPLE_SIZE_SMALL,
        "SAMPLE_SIZE_MEDIUM": SAMPLE_SIZE_MEDIUM,
        "SAMPLE_SIZE_LARGE": SAMPLE_SIZE_LARGE,
        "MAX_SAMPLES": MAX_SAMPLES,
    }

    # Create a deterministic JSON string (sorted keys)
    config_json = json.dumps(config_dict, sort_keys=True)

    # Generate a hash (specify this is not for security purposes)
    try:
        # For Python 3.9+, use usedforsecurity parameter
        config_hash = hashlib.md5(
            config_json.encode(), usedforsecurity=False
        ).hexdigest()
    except TypeError:
        # For older Python versions, just use md5 (we're using it for checksums, not security)
        config_hash = hashlib.md5(config_json.encode()).hexdigest()  # nosec B324

    return config_hash


def check_splits_valid(db, file_id: UUID, document_id: UUID) -> bool:
    """
    Check if existing splits for a file are valid with the current configuration.
    This includes checking if the split files actually exist on disk and
    ensuring that there are no partial split sets from interrupted operations.

    Args:
        db: Database session
        file_id: Original file ID
        document_id: ID of the current document

    Returns:
        True if splits are valid and complete, False otherwise
    """
    # Step 1: Check if the document is in Splitting state
    if _is_document_in_splitting_state(db, document_id):
        return False
    # Step 2: Get the splits for this file with proper filtering
    splits = (
        db.query(FileSplit)
        .filter(
            FileSplit.original_file_id == file_id, FileSplit.document_id == document_id
        )
        .all()
    )
    if not splits:
        logger.info(f"No existing splits found for file {file_id}")
        return False

    # Step 3: Check if we have the expected number of splits
    if not _has_expected_split_count(db, document_id, splits):
        return False

    # Step 4: Validate each split against current configuration
    return _are_all_splits_valid(splits)


def _is_document_in_splitting_state(db, document_id: UUID) -> bool:
    """Check if the document is in Splitting state."""
    document = db.query(Document).filter(Document.id == document_id).first()

    if (
        document
        and document.processing_status == DocumentProcessingStatusEnum.Splitting
    ):
        logger.info(
            f"Document {document_id} is in Splitting state, splits are not valid"
        )
        return True

    return False


def _has_expected_split_count(db, document_id: UUID, splits: List) -> bool:
    """Check if we have all the expected splits."""
    # Get the document to check total_splits_count
    document = db.query(Document).filter(Document.id == document_id).first()

    if not document:
        logger.error(f"Document {document_id} not found")
        return False

    # If document has total_splits_count set, compare with actual splits
    if document.total_splits_count and document.total_splits_count > 0:
        if len(splits) != document.total_splits_count:
            logger.warning(
                f"Split count mismatch: found {len(splits)} splits but expected {document.total_splits_count} "
                f"for document {document_id}"
            )
            return False

    # If all splits have the same total_splits value, use that as expected count
    if splits and all(split.total_splits == splits[0].total_splits for split in splits):
        expected_count = splits[0].total_splits
        if len(splits) != expected_count:
            logger.warning(
                f"Split count mismatch: found {len(splits)} splits but expected {expected_count} "
                f"according to splits' total_splits value for document {document_id}"
            )
            return False

    return True


def _are_all_splits_valid(splits: List) -> bool:
    """Check if all splits are valid (have physical files and correct config)."""
    import os

    # Get current configuration hash
    current_config_hash = get_splitting_config_hash()

    for split in splits:
        # Check if split file exists
        if not os.path.exists(split.split_file_path):
            logger.warning(f"Split file missing: {split.split_file_path}")
            return False

        # Check if config hash matches (if one is set)
        if split.config_hash and split.config_hash != current_config_hash:
            logger.warning(
                f"Split {split.id} was created with different configuration "
                f"(hash: {split.config_hash} vs current: {current_config_hash})"
            )
            return False

    return True


def clean_existing_splits(
    db,
    file_id: UUID,
    dataset_id: Optional[UUID] = None,
    document_id: Optional[UUID] = None,
) -> None:
    """
    Clean up existing splits for a file, optionally filtered by dataset and document.
    Refactored to reduce cyclomatic complexity by using helper functions.

    Args:
        db: Database session
        file_id: UUID of the file whose splits should be cleaned
        dataset_id: Optional UUID of the dataset to filter splits by
        document_id: Optional UUID of the document to filter splits by"""
    try:
        # Find and validate splits to delete
        splits_to_delete = _find_splits_to_delete(db, file_id, dataset_id, document_id)
        if not splits_to_delete:
            return

        # Clean up document chunks that reference these splits
        _cleanup_split_document_chunks(db, splits_to_delete)

        # Clean up split files from filesystem
        _cleanup_split_filesystem_files(splits_to_delete)

        # Clean up split database records
        _cleanup_split_database_records(db, file_id, dataset_id, document_id)

        logger.info(
            f"Successfully cleaned up {len(splits_to_delete)} split records for file {file_id}"
        )

    except Exception as e:
        db.rollback()
        logger.error(
            f"Error cleaning existing splits for file {file_id}: {str(e)}",
            exc_info=True,
        )
        raise


def _find_splits_to_delete(
    db, file_id: UUID, dataset_id: Optional[UUID], document_id: Optional[UUID]
) -> Optional[List]:
    """Find splits to be deleted with optional dataset and document filtering."""
    # Build query for splits to delete
    split_query = db.query(FileSplit).filter(FileSplit.original_file_id == file_id)

    if dataset_id:
        split_query = split_query.filter(FileSplit.dataset_id == dataset_id)

    if document_id:
        split_query = split_query.filter(FileSplit.document_id == document_id)
        logger.info(
            f"Cleaning existing splits for file {file_id} in dataset {dataset_id} for document {document_id}"
        )
    elif dataset_id:
        logger.info(
            f"Cleaning existing splits for file {file_id} in dataset {dataset_id}"
        )
    else:
        logger.info(f"Cleaning existing splits for file {file_id} (all datasets)")

    splits_to_delete = split_query.all()

    if not splits_to_delete:
        logger.info(f"No existing splits found for file {file_id}")
        return None

    split_ids = [split.id for split in splits_to_delete]
    logger.info(f"Found {len(splits_to_delete)} splits to clean up: {split_ids}")

    return splits_to_delete


def _cleanup_split_document_chunks(db, splits_to_delete: List) -> None:
    """Clean up document chunks that reference the splits to be deleted."""
    split_ids = [split.id for split in splits_to_delete]

    # Find all chunks that reference these splits
    chunks_to_delete = (
        db.query(DocumentChunk).filter(DocumentChunk.split_id.in_(split_ids)).all()
    )

    if not chunks_to_delete:
        return

    logger.info(
        f"Found {len(chunks_to_delete)} document chunks referencing splits - deleting them first"
    )

    # Delete chunks in batches to avoid memory issues
    _delete_chunks_in_batches(db, chunks_to_delete)

    # Commit chunk deletions
    db.commit()
    logger.info(f"Successfully deleted {len(chunks_to_delete)} document chunks")


def _delete_chunks_in_batches(db, chunks_to_delete: List) -> None:
    """Delete document chunks in batches to avoid memory issues."""
    batch_size = 100

    for i in range(0, len(chunks_to_delete), batch_size):
        batch = chunks_to_delete[i : i + batch_size]
        chunk_ids = [chunk.id for chunk in batch]

        # Delete the batch
        db.query(DocumentChunk).filter(DocumentChunk.id.in_(chunk_ids)).delete(
            synchronize_session=False
        )
        logger.debug(f"Deleted chunk batch {i//batch_size + 1}: {len(batch)} chunks")


def _cleanup_split_filesystem_files(splits_to_delete: List) -> None:
    """Clean up split files from the filesystem."""
    deleted_files = []

    for split in splits_to_delete:
        if split.split_file_path and Path(split.split_file_path).exists():
            try:
                Path(split.split_file_path).unlink()
                deleted_files.append(split.split_file_path)
            except Exception as e:
                logger.warning(
                    f"Could not delete split file {split.split_file_path}: {str(e)}"
                )

    if deleted_files:
        logger.info(f"Deleted {len(deleted_files)} split files from filesystem")


def _cleanup_split_database_records(
    db, file_id: UUID, dataset_id: Optional[UUID], document_id: Optional[UUID]
) -> None:
    """Clean up split database records (safe to do after chunks are deleted)."""
    # Rebuild the query for final deletion
    split_query = db.query(FileSplit).filter(FileSplit.original_file_id == file_id)

    if dataset_id:
        split_query = split_query.filter(FileSplit.dataset_id == dataset_id)

    if document_id:
        split_query = split_query.filter(FileSplit.document_id == document_id)

    # Delete the split records (now safe since no chunks reference them)
    _deleted_count = split_query.delete(synchronize_session=False)
    db.commit()


def create_splits_for_file(
    db,
    file_id: UUID,
    file_path: str,
    document_id: UUID,
    dataset_id: UUID,
) -> List:
    """
    Create new splits for a file.

    Args:
        db: Database session
        file_id: ID of the original file
        file_path: Path to the original file
        document_id: ID of the document
        dataset_id: ID of the dataset

    Returns:
        List of created split records
    """

    # Update document status to Splitting
    document = db.query(Document).filter(Document.id == document_id).first()

    # Check if this file ingestion was previously in the Splitting state
    # This indicates a pod restart or failure during splitting
    if (
        document
        and document.processing_status == DocumentProcessingStatusEnum.Splitting
    ):
        logger.warning(
            f"Detected interrupted split creation for file {file_id}. "
            f"Cleaning up partial splits and starting fresh."
        )
        # Clean up any partial splits from the previous attempt
        clean_existing_splits(db=db, file_id=file_id, dataset_id=dataset_id)

    # Mark the document as Splitting
    if document:
        document.processing_status = DocumentProcessingStatusEnum.Splitting
        document.is_split_document = True
        db.commit()

    try:
        splitter = FileSplitter()
        splits_info = splitter.split_file(file_path)

        # Generate config hash for tracking
        config_hash = get_splitting_config_hash()
        logger.info(f"Creating splits with config hash: {config_hash}")

        splits = []
        for split_info in splits_info:
            # Use our CRUD function to create each split with dataset_id
            split = file_split_crud.create_split(
                db=db,
                original_file_id=file_id,
                document_id=document_id,
                dataset_id=dataset_id,  # Include dataset_id
                split_file_path=split_info["split_file_path"],
                split_index=split_info["split_index"],
                total_splits=split_info["total_splits"],
                size=split_info["size"],
                token_count=split_info["token_count"],
                config_hash=config_hash,
            )
            splits.append(split)
            # Commit after each split to ensure progress is saved
            db.commit()

        # Update the file to indicate it requires splitting
        file = db.query(File).filter(File.id == file_id).first()
        if file:
            file.requires_splitting = True

        # Update document with split info
        if document:
            document.is_split_document = True
            document.total_splits_count = len(splits)
            document.successful_splits_count = 0
            document.processing_status = DocumentProcessingStatusEnum.Processing
            db.add(document)

        db.commit()
        logger.info(f"Created {len(splits)} splits for file {file_id}")
        return splits

    except Exception as e:
        logger.error(f"Error creating splits for file {file_id}: {str(e)}")
        db.rollback()

        # Update document status to Failed
        if document:
            document.processing_status = DocumentProcessingStatusEnum.Failed
            document.error_message = f"Failed to create splits: {str(e)}"
            db.commit()

        raise
