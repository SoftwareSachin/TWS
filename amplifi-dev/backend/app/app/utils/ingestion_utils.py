"""
Utility functions for file ingestion, supporting both regular and split-based ingestion.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from limits import RateLimitItemPerMinute, storage, strategies

from app.be_core.config import settings

# from datetime import timedelta
from app.be_core.logger import logger
from app.crud.file_split_crud import file_split_crud
from app.crud.ingest_crud import ingestion_crud
from app.db.session import SyncSessionLocal
from app.models.file_ingestion_model import FileIngestion, FileIngestionStatusType

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
        logger.error(f"RateLimit Error: {str(e)}", exc_info=True)
        return False, 60


def monitor_document_status(client, doc_id: str) -> Tuple[str, dict]:
    """
    Monitor the status of a document in R2R until it reaches a terminal state.

    Args:
        client: R2R client
        doc_id: ID of the document to monitor

    Returns:
        Tuple of (status, document_data)
    """
    while True:
        logger.info(f"Polling status for document {doc_id}")
        time.sleep(POLL_INTERVAL)

        try:
            doc_response = client.documents.retrieve(id=doc_id)
            doc = doc_response.results

            if doc.ingestion_status.value in TERMINAL_STATUSES:
                doc_data = {
                    "file_id": str(doc.id),
                    "file_name": doc.title,
                    "status": doc.ingestion_status.value,
                    "created_at": doc.created_at,
                    "updated_at": doc.updated_at,
                    "finished_at": doc.updated_at,
                    "metadata": doc.metadata or {},
                    "size_in_bytes": doc.size_in_bytes,
                    "ingestion_attempt_number": doc.ingestion_attempt_number or 0,
                }

                return doc.ingestion_status.value, doc_data

        except Exception as e:
            logger.error(f"Error monitoring document {doc_id}: {e}", exc_info=True)
            return "failed", {"error": str(e)}


def update_task_id(
    file_id: UUID, task_id: str, ingestion_id: str, dataset_id: UUID
) -> None:
    """
    Update the task ID for a file ingestion record.

    Args:
        file_id: ID of the file
        task_id: ID of the Celery task
        ingestion_id: ID of the ingestion process
        dataset_id: ID of the dataset
    """
    if file_id:
        logger.debug(
            f"Updating task ID {task_id} for file {file_id} "
            f"in ingestion {ingestion_id} for dataset {dataset_id}"
        )
        ingestion_crud.update_file_ingestion_task_id_sync(
            ingestion_id=ingestion_id,
            file_id=file_id,
            task_id=task_id,
            dataset_id=dataset_id,
        )


def update_file_ingestion_status(
    task_id: str, status: FileIngestionStatusType, doc: dict = None
) -> None:
    """
    Update the status of a file ingestion record.

    Args:
        task_id: ID of the Celery task
        status: New status to set
        doc: Optional document data, which can be a dict or object
    """
    try:
        with SyncSessionLocal() as db:
            from sqlalchemy import update

            # Get the finished timestamp from the document or use current time
            finished_at = datetime.now()
            if doc:
                # Handle both object and dictionary formats
                if isinstance(doc, dict):
                    # Dictionary format (from monitor_document_status)
                    finished_at = (
                        doc.get("updated_at")
                        or doc.get("finished_at")
                        or datetime.now()
                    )
                else:
                    # Object format (direct from R2R)
                    finished_at = (
                        getattr(doc, "updated_at", None)
                        or getattr(doc, "finished_at", None)
                        or datetime.now()
                    )

            # Update the file ingestion record
            stmt = (
                update(FileIngestion)
                .where(FileIngestion.task_id == task_id)
                .values(
                    status=status.value,
                    finished_at=finished_at,
                )
            )
            db.execute(stmt)
            db.commit()

        logger.info(f"Updated ingestion status for task {task_id} to {status.value}")
    except Exception as e:
        logger.error(f"Error updating ingestion status: {str(e)}", exc_info=True)


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
    if file_size < file_splitter.SMALL_FILE_THRESHOLD:
        logger.debug(
            f"File {file_path} is smaller than {file_splitter.SMALL_FILE_THRESHOLD/1024/1024}MB, no splitting needed"
        )
        return False

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
        logger.info(
            f"Falling back to size-based decision: file {needs_splitting and 'needs' or 'does not need'} splitting"
        )
        return needs_splitting


def update_parent_ingestion_status(
    db,
    ingestion_id: UUID,
    file_id: UUID,
    dataset_id: UUID,
) -> None:
    """
    Update the status of the parent file ingestion based on the status of its splits.
    Only considers an ingestion successful if ALL splits are successfully ingested.
    Also sends WebSocket notifications about the parent file status changes.

    Args:
        db: Database session
        ingestion_id: ID of the ingestion process
        file_id: ID of the original file
        dataset_id: ID of the dataset
    """
    try:
        # Get the file ingestion record
        file_ingestion = (
            db.query(FileIngestion)
            .filter(
                FileIngestion.ingestion_id == ingestion_id,
                FileIngestion.file_id == file_id,
                FileIngestion.dataset_id == dataset_id,
            )
            .first()
        )

        if not file_ingestion:
            logger.error(
                f"No file ingestion record found for file {file_id} in ingestion {ingestion_id}"
            )
            return

        # Use the CRUD function to update the status
        new_status = file_split_crud.update_parent_ingestion_status(
            db, file_ingestion.id
        )

        # Log the status change for better tracking retries and overall progress
        if new_status:
            logger.info(
                f"Updated parent ingestion status to {new_status} for file {file_id} in dataset {dataset_id}"
            )

            # If any splits are still being retried or processing, the status should remain as Processing
            from app.models.file_split_model import SplitFileStatusType

            splits = file_split_crud.get_splits_for_ingestion(db, file_ingestion.id)

            processing_splits = sum(
                1 for s in splits if s.status == SplitFileStatusType.Processing
            )

            if processing_splits > 0 and new_status in [
                FileIngestionStatusType.Failed,
                FileIngestionStatusType.Exception,
            ]:
                logger.info(
                    f"File {file_id} has {processing_splits} splits still processing or being retried. "
                    f"Keeping parent status as Processing until all splits complete or fail."
                )
                file_ingestion.status = FileIngestionStatusType.Processing
                db.commit()

        # Reload the file ingestion record to get the updated status
        db.refresh(file_ingestion)
    except Exception as e:
        logger.error(f"Error updating parent ingestion status: {str(e)}", exc_info=True)
        db.rollback()


def cleanup_split_files(db, file_ingestion_id: UUID) -> int:
    """
    Clean up split files for a file ingestion.
    Will only perform cleanup if SPLIT_CLEANUP_ENABLED setting is True.
    Only deletes the temporary files without removing database records.

    Args:
        db: Database session
        file_ingestion_id: ID of the file ingestion

    Returns:
        Number of temporary files deleted
    """
    # Check if split cleanup is enabled
    if not settings.SPLIT_CLEANUP_ENABLED:
        logger.info(
            f"Split cleanup is disabled. Skipping split files cleanup for file ingestion {file_ingestion_id}"
        )
        return 0

    try:
        # Get all splits for this file ingestion

        splits = file_split_crud.get_splits_for_ingestion(db, file_ingestion_id)

        if not splits:
            logger.debug(
                f"No splits found to clean up for ingestion {file_ingestion_id}"
            )
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
            f"Cleaned up {file_delete_count} temporary split files for ingestion {file_ingestion_id} while keeping database records"
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


def check_splits_valid(db, file_id: UUID, file_ingestion_id: UUID) -> bool:
    """
    Check if existing splits for a file are valid with the current configuration.
    This includes checking if the split files actually exist on disk and
    ensuring that there are no partial split sets from interrupted operations.

    Args:
        db: Database session
        file_id: Original file ID
        file_ingestion_id: ID of the current file ingestion

    Returns:
        True if splits are valid and complete, False otherwise
    """
    # Step 1: Check if the file ingestion is in Splitting state
    if _is_ingestion_in_splitting_state(db, file_ingestion_id):
        return False

    # Step 2: Get the splits for this file
    splits = file_split_crud.get_splits_for_file(db, file_id)
    if not splits:
        logger.info(f"No existing splits found for file {file_id}")
        return False

    # Step 3: Check if we have the expected number of splits
    if not _has_expected_split_count(db, file_ingestion_id, splits):
        return False

    # Step 4: Validate each split against current configuration
    return _are_all_splits_valid(splits)


def _is_ingestion_in_splitting_state(db, file_ingestion_id: UUID) -> bool:
    """Check if the file ingestion is in Splitting state."""
    from app.models.file_ingestion_model import FileIngestion, FileIngestionStatusType

    file_ingestion = (
        db.query(FileIngestion).filter(FileIngestion.id == file_ingestion_id).first()
    )

    if file_ingestion and file_ingestion.status == FileIngestionStatusType.Splitting:
        logger.info(
            f"File ingestion {file_ingestion_id} is in Splitting state, splits are not valid"
        )
        return True

    return False


def _has_expected_split_count(db, file_ingestion_id: UUID, splits: list) -> bool:
    """Check if we have the expected number of splits."""
    from app.models.file_ingestion_model import FileIngestion

    file_ingestion = (
        db.query(FileIngestion).filter(FileIngestion.id == file_ingestion_id).first()
    )

    if (
        file_ingestion
        and file_ingestion.total_splits_count
        and len(splits) != file_ingestion.total_splits_count
    ):
        logger.warning(
            f"Incomplete split set detected: found {len(splits)} splits but expected {file_ingestion.total_splits_count}"
        )
        return False

    return True


def _are_all_splits_valid(splits: list) -> bool:
    """Validate that all splits are consistent and valid."""
    # Get the current configuration hash
    current_config_hash = get_splitting_config_hash()

    # Get the expected total splits from the first split
    expected_total = splits[0].total_splits if splits else 0

    # Check each split for validity
    for split in splits:
        # Check if split has valid configuration hash
        if split.config_hash != current_config_hash:
            logger.info(
                f"Split {split.id} configuration hash doesn't match current: {split.config_hash} vs {current_config_hash}"
            )
            return False

        # Check if split file exists on disk
        if not os.path.exists(split.split_file_path):
            logger.info(f"Split file doesn't exist on disk: {split.split_file_path}")
            return False

        # Check if this split has the same total_splits as other splits
        if split.total_splits != expected_total:
            logger.warning(
                f"Inconsistent total_splits value in split {split.id}: {split.total_splits} vs expected {expected_total}"
            )
            return False

    logger.info(
        f"All splits for file {splits[0].original_file_id} are valid with current configuration"
    )
    return True


def clean_existing_splits(
    db, file_id: UUID, dataset_id: Optional[UUID]
) -> Tuple[int, int]:
    """
    Remove existing splits for a file from both the database and filesystem.

    Args:
        db: Database session
        file_id: ID of the original file

    Returns:
        Tuple containing (number of DB records deleted, number of files deleted)
    """
    # Use our CRUD function for this operation
    return file_split_crud.clean_existing_splits(
        db, file_id, delete_files=True, dataset_id=dataset_id
    )


def create_splits_for_file(
    db, file_id: UUID, file_path: str, file_ingestion_id: UUID, dataset_id: UUID
) -> List:
    """
    Create splits for a file and save them to the database.
    This function handles pod restarts by detecting and cleaning up incomplete splits.

    Args:
        db: Database session
        file_id: ID of the original file
        file_path: Path to the original file
        file_ingestion_id: ID of the file ingestion record
        dataset_id: ID of the dataset

    Returns:
        List of created FileSplit objects
    """
    from app.models.file_ingestion_model import FileIngestion, FileIngestionStatusType
    from app.models.file_model import File

    # Get the file ingestion record
    file_ingestion = (
        db.query(FileIngestion).filter(FileIngestion.id == file_ingestion_id).first()
    )

    # Check if this file ingestion was previously in the Splitting state
    # This indicates a pod restart or failure during splitting
    if file_ingestion and file_ingestion.status == FileIngestionStatusType.Splitting:
        logger.warning(
            f"Detected interrupted split creation for file {file_id}. "
            f"Cleaning up partial splits and starting fresh."
        )
        # Clean up any partial splits from the previous attempt
        clean_existing_splits(db=db, file_id=file_id, dataset_id=dataset_id)

    # Mark the ingestion as splitting in progress
    if file_ingestion:
        file_ingestion.status = FileIngestionStatusType.Splitting
        file_ingestion.is_split_ingestion = True
        db.commit()

    try:
        splitter = FileSplitter()
        splits_info = splitter.split_file(file_path)

        # Generate a hash of the current configuration
        config_hash = get_splitting_config_hash()
        logger.info(f"Creating splits with configuration hash: {config_hash}")

        file_splits = []
        for split_info in splits_info:
            # Use our CRUD function to create each split with dataset_id
            split = file_split_crud.create_split(
                db=db,
                original_file_id=file_id,
                file_ingestion_id=file_ingestion_id,
                dataset_id=dataset_id,  # Include dataset_id
                split_file_path=split_info["split_file_path"],
                split_index=split_info["split_index"],
                total_splits=split_info["total_splits"],
                size=split_info["size"],
                token_count=split_info["token_count"],
                config_hash=config_hash,
            )
            file_splits.append(split)
            # Commit after each split to ensure progress is saved
            db.commit()

        # Update the file to indicate it requires splitting
        file = db.query(File).filter(File.id == file_id).first()
        if file:
            file.requires_splitting = True

        # Update the file ingestion record to track split ingestion
        if file_ingestion:
            file_ingestion.is_split_ingestion = True
            file_ingestion.total_splits_count = len(splits_info)
            file_ingestion.successful_splits_count = 0
            # Reset status to Pending since splitting is complete
            file_ingestion.status = FileIngestionStatusType.Processing
            db.add(file_ingestion)

        db.commit()
        return file_splits

    except Exception as e:
        logger.error(
            f"Error creating splits for file {file_id}: {str(e)}", exc_info=True
        )
        # If there's an error, mark the file ingestion as failed
        if file_ingestion:
            file_ingestion.status = FileIngestionStatusType.Failed
            file_ingestion.error_message = f"Split creation failed: {str(e)}"
            db.commit()
        raise
