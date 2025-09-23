"""
Celery task for ingesting individual file splits using Document model (v2).
Uses the Document model directly instead of FileIngestion model.
Supports PDF, DOCX, XLSX, PPTX, Markdown, HTML, and CSV files.
Simplified version with centralized status management.
"""

import secrets
from datetime import UTC, datetime
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import attributes

from app.api.deps import get_redis_client_sync
from app.be_core.celery import celery
from app.be_core.config import settings
from app.be_core.logger import logger
from app.crud.file_split_crud_v2 import file_split_crud
from app.db.session import SyncSessionLocal
from app.models.document_chunk_model import DocumentChunk
from app.models.document_model import (
    Document,
    DocumentProcessingStatusEnum,
)
from app.models.file_split_model import FileSplit, SplitFileStatusType
from app.utils.document_processing_utils import (
    process_document,  # Use the new unified function
)
from app.utils.document_processing_utils import (
    _get_document_type_from_file_path,
)
from app.utils.ingestion_status_propagation import (
    propagate_ingestion_status,
    should_skip_processing_due_to_timeout,
)
from app.utils.ingestion_utils_v2 import (
    acquire_tokens_with_retry,
    file_splitter,
)
from app.utils.processing_lock_utils import (
    _cleanup_processing_lock,
    _handle_stale_lock,
)


def _acquire_processing_lock_atomic(
    redis_client, processing_key: str, task_id: str, expiry_seconds: int = 3600
) -> tuple[bool, Optional[str]]:
    """
    Atomically acquire a processing lock using Redis SET NX EX.

    Args:
        redis_client: Redis client instance
        processing_key: The key to lock
        task_id: Current task ID to store in the lock
        expiry_seconds: Lock expiry time in seconds

    Returns:
        tuple: (acquired, existing_task_id)
        - acquired: True if lock was acquired, False if already exists
        - existing_task_id: Task ID that currently holds the lock (if any)
    """
    try:
        # Try to atomically set the key only if it doesn't exist (NX) with expiration (EX)
        result = redis_client.set(processing_key, task_id, nx=True, ex=expiry_seconds)

        if result:
            logger.info(
                f"Successfully acquired split processing lock: {processing_key}"
            )
            return True, None
        else:
            # Lock exists, get the current task ID
            existing_task_id = redis_client.get(processing_key)
            if existing_task_id:
                # Handle both bytes and string types
                if isinstance(existing_task_id, bytes):
                    existing_task_id = existing_task_id.decode("utf-8")
                else:
                    existing_task_id = str(existing_task_id)

            logger.info(
                f"Split processing lock already exists for key {processing_key}, held by task: {existing_task_id}"
            )
            return False, existing_task_id

    except Exception as e:
        logger.error(
            f"Error acquiring split processing lock {processing_key}: {str(e)}"
        )
        # If Redis is unavailable, allow processing to continue
        return True, None


class SplitStatusManager:
    """Centralized status management for split processing"""

    def __init__(
        self,
        split_id: str,
        original_file_id: str,
        dataset_id: str,
        ingestion_id: str,
        user_id: UUID,
    ):
        self.split_id = split_id
        self.original_file_id = original_file_id
        self.dataset_id = dataset_id
        self.ingestion_id = ingestion_id
        self.user_id = user_id

    def mark_processing(self, task_id: str, metadata: Dict[str, Any]) -> None:
        """Mark split as processing"""
        with SyncSessionLocal() as db:
            split = file_split_crud.update_split_status(
                db=db,
                split_id=UUID(self.split_id),
                status=SplitFileStatusType.Processing,
                task_id=task_id,
            )

            if split:
                if split.split_metadata is None:
                    split.split_metadata = metadata or {}
                split.split_metadata.update(
                    {
                        "processing_started_at": datetime.now(UTC).isoformat(),
                        "task_id": task_id,
                    }
                )
                attributes.flag_modified(split, "split_metadata")
                db.commit()
            logger.info(f"Split {self.split_id} reset and marked as Processing")

    def mark_success(self, has_images: bool = False) -> None:
        """Mark split as successful, handle image dependencies"""
        with SyncSessionLocal() as db:
            # Get current split with lock to prevent race conditions
            current_split = (
                db.query(FileSplit)
                .filter(FileSplit.id == UUID(self.split_id))
                .with_for_update()
                .first()
            )

            if not current_split:
                logger.error(f"Split {self.split_id} not found")
                return

            # If split is already Success, still need to check propagation
            if current_split.status == SplitFileStatusType.Success:
                logger.info(
                    f"Split {self.split_id} already marked as Success - checking propagation"
                )
                # Update document status in case this is a late completion
                self._update_document_from_splits(db)
                # Still propagate if ready - this split completion might trigger document completion
                self._propagate_if_ready(db)
                return

            # Check if split was marked as failed due to timeout
            if (
                current_split.split_metadata
                and current_split.split_metadata.get("timeout_failure")
                and current_split.split_metadata.get("timeout_failure", {}).get(
                    "parent_document_timeout"
                )
            ):

                # If split has timeout metadata but is not Failed, fix the status
                if current_split.status != SplitFileStatusType.Failed:
                    logger.warning(
                        f"Split {self.split_id} has timeout metadata but status is {current_split.status} - fixing to Failed"
                    )
                    file_split_crud.update_split_status(
                        db=db,
                        split_id=UUID(self.split_id),
                        status=SplitFileStatusType.Failed,
                    )
                else:
                    logger.warning(
                        f"Split {self.split_id} was marked as failed due to parent document timeout - not changing status"
                    )

                self._update_document_from_splits(db)
                self._propagate_if_ready(db)
                return

            # If split is Failed/Exception, don't override
            if current_split.status in [
                SplitFileStatusType.Failed,
                SplitFileStatusType.Exception,
            ]:
                logger.info(
                    f"Split {self.split_id} is in {current_split.status} state - not changing"
                )
                return

            # Update split status
            if has_images:
                status = SplitFileStatusType.Processing  # Wait for images
                logger.info(f"Split {self.split_id} has images - keeping as Processing")
            else:
                status = SplitFileStatusType.Success
                logger.info(f"Split {self.split_id} completed - no images")

            file_split_crud.update_split_status(
                db=db, split_id=UUID(self.split_id), status=status
            )

            # Update document status
            self._update_document_from_splits(db)

            # Handle propagation
            if status == SplitFileStatusType.Success:
                self._propagate_if_ready(db)

    def mark_failed(self, error_message: str) -> None:
        """Mark split as failed"""
        with SyncSessionLocal() as db:
            split = file_split_crud.update_split_status(
                db=db,
                split_id=UUID(self.split_id),
                status=SplitFileStatusType.Failed,
            )
            if split:
                split.split_metadata = {
                    "processing_failed_at": datetime.now(UTC).isoformat(),
                    "error_message": error_message,
                    "processing_status": "failed",
                    "failure_reason": "processing_error",
                }
                attributes.flag_modified(split, "split_metadata")
                db.commit()
            logger.error(f"Split {self.split_id} marked as Failed: {error_message}")

            # Update document status
            self._update_document_from_splits(db)

            # Always propagate failures immediately
            document = self._get_document(db)
            if document:
                propagate_ingestion_status(db, "document", document.id, self.user_id)

    def _update_document_from_splits(self, db) -> None:
        """Single place for document status aggregation"""
        document = self._get_document(db)
        if not document:
            return

        # Get all splits for this document
        splits = (
            db.query(FileSplit)
            .filter(
                FileSplit.original_file_id == UUID(self.original_file_id),
                FileSplit.dataset_id == UUID(self.dataset_id),
                FileSplit.document_id == document.id,
            )
            .all()
        )

        if not splits:
            document.processing_status = DocumentProcessingStatusEnum.Success
            document.processed_at = datetime.now(UTC)
            db.commit()
            return

        # Calculate status counts
        success_count = sum(
            1 for s in splits if s.status == SplitFileStatusType.Success
        )
        failed_count = sum(
            1
            for s in splits
            if s.status in [SplitFileStatusType.Failed, SplitFileStatusType.Exception]
        )
        processing_count = len(splits) - success_count - failed_count

        # Update document metadata
        document.successful_splits_count = success_count
        document.total_splits_count = len(splits)

        document.document_metadata = document.document_metadata or {}
        document.document_metadata.update(
            {
                "last_split_status_update": datetime.now(UTC).isoformat(),
            }
        )

        # Update document status
        if failed_count > 0:
            document.processing_status = DocumentProcessingStatusEnum.Failed
            document.processed_at = datetime.now(UTC)
        elif processing_count > 0:
            document.processing_status = DocumentProcessingStatusEnum.Processing
        elif success_count == len(splits):
            document.processing_status = DocumentProcessingStatusEnum.Success
            document.processed_at = datetime.now(UTC)

        logger.info(
            f"Document {document.id} status updated: "
            f"Success: {success_count}, Failed: {failed_count}, Processing: {processing_count}"
        )

        db.add(document)
        attributes.flag_modified(document, "document_metadata")
        db.commit()
        db.refresh(document)

    def _get_document(self, db) -> Optional[Document]:
        """Get the document for this split's original file"""
        return (
            db.query(Document)
            .filter(
                Document.file_id == UUID(self.original_file_id),
                Document.dataset_id == UUID(self.dataset_id),
            )
            .with_for_update()
            .first()
        )

    def _propagate_if_ready(self, db) -> None:
        """Propagate status only if document is ready"""
        document = self._get_document(db)
        if document and not document.document_metadata.get("waiting_for_images"):
            logger.info(
                f"Split {self.split_id} completed with no images - propagating status"
            )
            propagate_ingestion_status(db, "document", document.id, self.user_id)


def _prepare_split_metadata(
    metadata: Optional[dict],
    original_file_id: str,
    split_id: str,
    dataset_id: str,
    split: FileSplit,
) -> Dict[str, Any]:
    """Single function to prepare all split metadata"""
    if not metadata:
        metadata = {}

    metadata.update(
        {
            "is_split": True,
            "split_id": split_id,
            "original_file_id": original_file_id,
            "dataset_id": dataset_id,
            "split_index": split.split_index,
            "total_splits": split.total_splits,
            "token_count": split.token_count,
            "file_size": split.size,
        }
    )
    return metadata


def _get_split_with_validation(
    split_id: str,
) -> Tuple[bool, Optional[FileSplit], Optional[str]]:
    """Get split and return (success, split, error_message)"""
    try:
        with SyncSessionLocal() as db:
            split = file_split_crud.get_by_id(db, UUID(split_id))
            if not split:
                return False, None, f"Split {split_id} not found"
            return True, split, None
    except Exception as e:
        logger.error(f"Error getting split {split_id}: {str(e)}")
        return False, None, f"Error getting split: {str(e)}"


def _can_skip_split(split: FileSplit, skip_successful_files: bool) -> bool:
    """Simple boolean check for skip logic"""
    return skip_successful_files and split.status == SplitFileStatusType.Success


def _handle_token_acquisition_with_retry(
    file_path: str,
    retry_count: int,
    max_retries: int,
    split_id: str,
    original_file_id: str,
    ingestion_id: str,
    dataset_id: str,
    user_id: UUID,
    organization_id: Optional[str] = None,
    chunking_config: Optional[dict] = None,
    skip_successful_files: bool = True,
    metadata: Optional[dict] = None,
) -> Tuple[bool, str]:
    """Handle token acquisition with retry logic

    Returns:
        Tuple[bool, str]: (success, status)
        - success: True if tokens acquired, False if failed or retry scheduled
        - status: "success", "retry_scheduled", "max_retries_exceeded", or "error"
    """
    try:
        token_count = file_splitter.count_file_tokens(file_path)
        logger.info(f"Estimated token count for split {split_id}: {token_count}")

        success, remaining_time = acquire_tokens_with_retry(token_count)

        if not success:
            if retry_count < max_retries:
                # Calculate backoff with jitter
                base_delay = remaining_time or settings.RATE_LIMIT_RETRY_SECONDS * (
                    2**retry_count
                )
                max_backoff = 300  # 5 minutes max
                backoff = min(base_delay, max_backoff)
                jitter = (float(secrets.randbelow(20) - 10) / 100.0) * backoff
                countdown = max(1, backoff + jitter)

                logger.info(
                    f"Could not acquire tokens for split {split_id}. "
                    f"Retry {retry_count+1}/{max_retries} scheduled in {countdown:.1f}s"
                )

                # Schedule retry by re-invoking the task
                kwargs = {
                    "split_id": split_id,
                    "file_path": file_path,
                    "original_file_id": original_file_id,
                    "ingestion_id": ingestion_id,
                    "dataset_id": dataset_id,
                    "user_id": user_id,
                    "organization_id": organization_id,
                    "chunking_config": chunking_config,
                    "skip_successful_files": skip_successful_files,
                    "metadata": metadata,
                    "retry_count": retry_count + 1,
                    "max_retries": max_retries,
                    "retry_reason": "rate_limit",
                }

                celery.signature(
                    "tasks.ingest_split_task_v2", kwargs=kwargs
                ).apply_async(countdown=countdown)

                return (
                    False,
                    "retry_scheduled",
                )  # Current task should exit, retry is scheduled
            else:
                logger.error(
                    f"Failed to acquire tokens for split {split_id} after {max_retries} retries"
                )
                return False, "max_retries_exceeded"

        return True, "success"

    except Exception as e:
        logger.error(f"Error acquiring tokens for split {split_id}: {str(e)}")
        return False, "error"


def _delete_split_chunks(db, document_id: UUID, split_id: str) -> int:
    """Delete chunks that belong to a specific split."""
    try:
        deleted_count = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.document_id == document_id,
                DocumentChunk.split_id == split_id,
            )
            .delete(synchronize_session=False)
        )

        if deleted_count > 0:
            logger.info(
                f"Deleted {deleted_count} chunks for split {split_id} of document {document_id}"
            )
        else:
            logger.info(f"No existing chunks found for split {split_id}")

        return deleted_count

    except Exception as e:
        logger.error(f"Error deleting chunks for split {split_id}: {str(e)}")
        raise


def _get_or_create_document_for_file(
    db,
    original_file_id: str,
    dataset_id: str,
    split_id: str,
    skip_successful_files: bool,
    metadata: Dict[str, Any],
    task_id: str,
    ingestion_id: Optional[str] = None,
    original_file_path: Optional[str] = None,
) -> Optional[Document]:
    """Get existing document for the original file or create a new one."""

    # Get the document for the ORIGINAL FILE, not the split
    document = (
        db.query(Document)
        .filter(
            Document.file_id == UUID(original_file_id),
            Document.dataset_id == UUID(dataset_id),
        )
        .with_for_update()  # Lock the query to prevent race conditions
        .first()
    )

    if document:
        # Document exists - determine if we need to clean chunks for this split
        should_clean_chunks = False

        if not skip_successful_files:
            # Force reprocessing mode - clean chunks for this split
            should_clean_chunks = True
            logger.info(
                f"Force reprocessing mode - will clean chunks for split {split_id}"
            )
        else:
            # Selective reprocessing - only clean chunks if split is not successful
            split = db.query(FileSplit).filter(FileSplit.id == UUID(split_id)).first()
            if split and split.status != SplitFileStatusType.Success:
                should_clean_chunks = True
                logger.info(
                    f"Selective reprocessing - split {split_id} status is {split.status}, will clean chunks"
                )
            else:
                logger.info(
                    f"Selective reprocessing - split {split_id} is successful, keeping existing chunks"
                )

        # Clean chunks if needed
        if should_clean_chunks:
            deleted_count = _delete_split_chunks(db, document.id, split_id)
            # Also clean up split references from document metadata
            _cleanup_split_from_document_metadata(db, document, split_id)
            logger.info(
                f"Cleaned {deleted_count} chunks and metadata for split {split_id} before reprocessing"
            )

        # Update document processing status
        # Do not update documents that are already in failed/exception state
        if document.processing_status in [
            DocumentProcessingStatusEnum.Failed,
            DocumentProcessingStatusEnum.Exception,
        ]:
            logger.info(
                f"Document {document.id} is in {document.processing_status} state - not updating status or timestamp"
            )
        else:
            document.task_id = task_id
            document.updated_at = datetime.now(UTC)
            document.processing_status = DocumentProcessingStatusEnum.Processing

        db.commit()
        db.refresh(document)
        logger.info(
            f"Using existing document {document.id} for original file {original_file_id}"
        )
        return document

    # Create new document for the ORIGINAL FILE
    # This should only happen for the first split of a file
    from app.models.file_model import File

    # Get original file info
    original_file = db.query(File).filter(File.id == UUID(original_file_id)).first()
    if not original_file:
        logger.error(f"Original file {original_file_id} not found")
        return None

    # Determine document type from the original file
    document_type = _get_document_type_from_file_path(
        original_file_path or original_file.file_path
    )

    document = Document(
        file_id=UUID(original_file_id),
        dataset_id=UUID(dataset_id),
        document_type=document_type,  # Use detected document type
        processing_status=DocumentProcessingStatusEnum.Processing,
        file_path=original_file.file_path,  # Use original file path
        file_size=original_file.file_size,
        mime_type=original_file.mime_type,
        document_metadata=metadata or {},
        task_id=task_id,
        ingestion_id=UUID(ingestion_id) if ingestion_id else None,
        is_split_document=False,  # This represents the original file
        # Note: description and description_embedding will be set by the first split (index 0)
    )

    db.add(document)
    db.commit()
    db.refresh(document)
    logger.info(
        f"Created {document_type} document {document.id} for original file {original_file_id} - description will be generated by first split"
    )
    return document


def _cleanup_split_from_document_metadata(
    db, document: Document, split_id: str
) -> None:
    """Clean up split references from document metadata when reprocessing."""
    try:
        if not document.document_metadata:
            return

        # Clean up image summary if it exists
        image_summary = document.document_metadata.get("image_summary", {})
        if image_summary:
            # Remove this split from images_by_split
            images_by_split = image_summary.get("images_by_split", {})
            if split_id in images_by_split:
                logger.info(f"Removing split {split_id} from document image summary")
                del images_by_split[split_id]

            # Remove split references from images_by_page
            images_by_page = image_summary.get("images_by_page", {})
            for page_num, page_images in list(images_by_page.items()):
                images_by_page[page_num] = [
                    img for img in page_images if img.get("split_id") != split_id
                ]
                # Remove empty page entries
                if not images_by_page[page_num]:
                    del images_by_page[page_num]

            # Remove split references from all_image_info
            all_image_info = image_summary.get("all_image_info", [])
            image_summary["all_image_info"] = [
                img for img in all_image_info if img.get("split_id") != split_id
            ]

            # Recalculate totals
            total_images = sum(
                split_info.get("count", 0) for split_info in images_by_split.values()
            )
            image_summary["total_images"] = total_images

            # Update document
            document.document_metadata["image_summary"] = image_summary

        # Mark for database update
        db.add(document)
        attributes.flag_modified(document, "document_metadata")

        logger.info(f"Cleaned up split {split_id} references from document metadata")

    except Exception as e:
        logger.error(f"Error cleaning up split metadata: {str(e)}")


@celery.task(name="tasks.ingest_split_task_v2", bind=True, acks_late=True)
def ingest_split_task_v2(
    self,
    split_id: str,
    file_path: str,
    original_file_id: str,
    ingestion_id: str,
    dataset_id: str,
    user_id: UUID,
    organization_id: Optional[str] = None,
    chunking_config: Optional[dict] = None,
    skip_successful_files: bool = True,
    metadata: Optional[dict] = None,
    retry_count: int = 0,
    max_retries: int = 10,
    retry_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a single document split using the unified document processing pipeline.
    Supports PDF, DOCX, XLSX, PPTX, Markdown, HTML, and CSV files.

    Args:
        split_id: ID of the file split
        file_path: Path to the split file
        original_file_id: ID of the original file
        ingestion_id: ID of the ingestion process
        dataset_id: ID of the dataset
        user_id: ID of the user who initiated the ingestion
        organization_id: ID of the organization
        chunking_config: Configuration for chunking
        skip_successful_files: Whether to skip if split was already successful
        metadata: Metadata to associate with the split
        retry_count: Current retry attempt count for token acquisition
        max_retries: Maximum number of retries before giving up
        retry_reason: Reason for retry (e.g., "timeout", "rate_limit")

    Returns:
        Dictionary with processing results
    """
    # Detect document type for logging
    document_type = _get_document_type_from_file_path(file_path)

    logger.info(
        f"Starting split ingestion task for {document_type.value} split: {split_id} "
        f"(file: {file_path}, retry #{retry_count})"
    )

    # Initialize status manager
    status_manager = SplitStatusManager(
        split_id=split_id,
        original_file_id=original_file_id,
        dataset_id=dataset_id,
        ingestion_id=ingestion_id,
        user_id=user_id,
    )

    # Robust task-level deduplication
    processing_key = None
    redis_client = None

    try:
        redis_client = get_redis_client_sync()

        # Create unique processing key for this split
        processing_key = f"processing:split:{split_id}:file:{original_file_id}"

        # Try to atomically acquire processing lock
        acquired, existing_task_id = _acquire_processing_lock_atomic(
            redis_client, processing_key, self.request.id
        )

        if not acquired:
            # Check if original task is still alive
            if existing_task_id and not _handle_stale_lock(
                redis_client, processing_key, existing_task_id, self.request.id
            ):
                logger.info(
                    f"Split {split_id} is being processed by active task {existing_task_id} - skipping"
                )
                redis_client.close()

                return {
                    "split_id": split_id,
                    "original_file_id": original_file_id,
                    "success": True,
                    "skipped": True,
                    "reason": "already_processing",
                    "processing_task_id": existing_task_id,
                }
            # If we reach here, either we took over a stale lock or the original task is dead
            logger.info(
                f"Acquired split processing lock after stale lock handling: {processing_key}"
            )

        # logger.info(f"Successfully acquired split processing lock: {processing_key}")

    except Exception as e:
        # Don't fail the task if Redis is unavailable, just log it
        logger.warning(f"Redis deduplication not available: {str(e)}")
        processing_key = None
        redis_client = None

    document_id = None

    try:
        # Step 1: Get and validate split
        split_valid, split, error_msg = _get_split_with_validation(split_id)
        if not split_valid:
            return {
                "split_id": split_id,
                "original_file_id": original_file_id,
                "task_id": self.request.id,
                "success": False,
                "error": error_msg,
                "document_type": document_type.value,
            }

        # Step 2: Check if we can skip this split
        if _can_skip_split(split, skip_successful_files):
            logger.info(f"Split {split_id} already successful - skipping")
            return {
                "split_id": split_id,
                "original_file_id": original_file_id,
                "task_id": self.request.id,
                "success": True,
                "skipped": True,
                "reason": "already_successful",
                "document_type": document_type.value,
            }

        # Step 3: Prepare metadata
        split_metadata = _prepare_split_metadata(
            metadata, original_file_id, split_id, dataset_id, split
        )

        # Step 4: Get or create document for the original file FIRST to check timeout
        with SyncSessionLocal() as db:
            document = _get_or_create_document_for_file(
                db=db,
                original_file_id=original_file_id,
                dataset_id=dataset_id,
                split_id=split_id,
                skip_successful_files=skip_successful_files,
                metadata=split_metadata,
                task_id=self.request.id,
                ingestion_id=ingestion_id,
                original_file_path=file_path,  # Pass the split file path for type detection
            )

            if not document:
                error_msg = (
                    f"Failed to get or create document for file {original_file_id}"
                )
                status_manager.mark_failed(error_msg)
                return {
                    "split_id": split_id,
                    "success": False,
                    "error": error_msg,
                    "document_type": document_type.value,
                }

            document_id = str(document.id)

            # Check if this document was failed due to timeout - skip processing BEFORE marking as processing
            logger.info(f"CHECKING if document {document.id} is failed due to timeout")
            should_skip, skip_reason = should_skip_processing_due_to_timeout(
                document.id
            )
            if should_skip:
                logger.warning(
                    f"⏰ SKIPPING split processing for {split_id}: {skip_reason}"
                )
                return {
                    "split_id": split_id,
                    "original_file_id": original_file_id,
                    "success": True,
                    "skipped": True,
                    "reason": "document_timeout_failed",
                    "message": skip_reason,
                    "document_id": str(document.id),
                    "document_type": document_type.value,
                }

            # Also check if this specific split was failed due to timeout
            logger.info(f"CHECKING if split {split_id} is failed due to timeout")
            from app.utils.ingestion_status_propagation import (
                should_skip_split_processing_due_to_timeout,
            )

            split_should_skip, split_skip_reason = (
                should_skip_split_processing_due_to_timeout(UUID(split_id))
            )
            if split_should_skip:
                logger.warning(
                    f"⏰ SKIPPING split processing for {split_id}: {split_skip_reason}"
                )
                return {
                    "split_id": split_id,
                    "original_file_id": original_file_id,
                    "success": True,
                    "skipped": True,
                    "reason": "split_timeout_failed",
                    "message": split_skip_reason,
                    "document_id": str(document.id),
                    "document_type": document_type.value,
                }

        # Step 5: Mark split as processing (only if timeout check passed)
        if retry_count == 0:
            status_manager.mark_processing(self.request.id, split_metadata)

        # Step 6: Handle token acquisition with retry logic
        token_success, token_status = _handle_token_acquisition_with_retry(
            file_path=file_path,
            retry_count=retry_count,
            max_retries=max_retries,
            split_id=split_id,
            original_file_id=original_file_id,
            ingestion_id=ingestion_id,
            dataset_id=dataset_id,
            user_id=user_id,
            organization_id=organization_id,
            chunking_config=chunking_config,
            skip_successful_files=skip_successful_files,
            metadata=metadata,
        )

        if not token_success:
            if token_status == "retry_scheduled":  # nosec
                # Retry was scheduled, exit gracefully without marking as failed
                logger.info(
                    f"Token acquisition retry scheduled for split {split_id} - exiting current task"
                )
                return {
                    "split_id": split_id,
                    "original_file_id": original_file_id,
                    "task_id": self.request.id,
                    "success": True,
                    "retry_scheduled": True,
                    "retry_count": retry_count + 1,
                    "document_type": document_type.value,
                }
            else:
                # Actual failure (max retries exceeded or error)
                error_msg = (
                    f"Failed to acquire tokens for split {split_id}: {token_status}"
                )
                status_manager.mark_failed(error_msg)
                return {
                    "split_id": split_id,
                    "original_file_id": original_file_id,
                    "task_id": self.request.id,
                    "success": False,
                    "error": error_msg,
                    "document_type": document_type.value,
                }

        # Step 7: Process the split using unified document processing
        logger.info(
            f"Processing {document_type.value} split {split_id} with document {document_id}"
        )

        result = process_document(
            file_path=file_path,
            file_id=UUID(original_file_id),
            document_id=UUID(document_id),
            dataset_id=UUID(dataset_id),
            chunking_config=chunking_config,
            ingestion_id=UUID(ingestion_id) if ingestion_id else None,
            user_id=user_id,
            split_id=split_id,  # This is a split
            skip_successful_files=skip_successful_files,
        )

        # Step 8: Update split status based on processing results
        if result["success"]:
            has_images = result.get("has_images", False)
            status_manager.mark_success(has_images=has_images)

            logger.info(
                f"Successfully processed {document_type.value} split {split_id}: "
                f"chunks={result.get('chunk_counts', {})}, has_images={has_images}"
            )
        else:
            error_msg = (
                f"Split processing failed: {result.get('error', 'Unknown error')}"
            )
            status_manager.mark_failed(error_msg)

        # Return final result
        final_result = {
            "split_id": split_id,
            "document_id": document_id,
            "success": result["success"],
            "has_images": result.get("has_images", False),
            "processing_type": "split",
            "document_type": document_type.value,
            "chunk_counts": result.get("chunk_counts", {}),
            "task_id": self.request.id,
            "ingestion_id": ingestion_id,
        }

        if not result["success"]:
            final_result["error"] = result.get("error", "Unknown processing error")

        return final_result

    except Exception as e:
        logger.error(
            f"Error processing {document_type.value} split {split_id}: {str(e)}",
            exc_info=True,
        )
        status_manager.mark_failed(str(e))
        return {
            "split_id": split_id,
            "document_id": document_id,
            "success": False,
            "error": str(e),
            "document_type": document_type.value,
            "task_id": self.request.id,
            "ingestion_id": ingestion_id,
        }

    finally:
        # Clean up processing deduplication flag (only if we own it)
        if processing_key and redis_client:
            try:
                _cleanup_processing_lock(redis_client, processing_key, self.request.id)
            except Exception as e:
                logger.warning(f"Failed to clean up split processing lock: {str(e)}")

        # Close Redis connection
        if redis_client:
            try:
                redis_client.close()
            except Exception as e:
                logger.warning(f"Failed to close Redis connection: {str(e)}")
